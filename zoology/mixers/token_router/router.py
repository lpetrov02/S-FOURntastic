import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from einops import repeat, rearrange


def stochastic_depth(input: torch.Tensor, p: float, mode: str, training: bool = True):
    """
    Implements Stochastic Depth.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim

    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate).div_(survival_rate)
    return input * noise


class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie: bool = True, transposed: bool = True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d style)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability has to be in [0, 1), but got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if not self.training:
            return X

        if not self.transposed:
            X = rearrange(X, "b ... d -> b d ...")

        mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
        mask = (torch.rand(*mask_shape, device=X.device) < (1.0 - self.p)).to(X.dtype)
        X = X * mask * (1.0 / (1.0 - self.p))

        if not self.transposed:
            X = rearrange(X, "b d ... -> b ... d")
        return X


def sample_gumbel_like(x: torch.Tensor) -> torch.Tensor:
    """Sample standard Gumbel noise with same shape/device/dtype as x."""
    u = torch.rand_like(x).clamp_(1e-6, 1.0 - 1e-6)
    return -torch.log(-torch.log(u))


class TokenTopKRouter(nn.Module):
    """
    Token-level router.
    Input:  u of shape (B, H, L)
    Output: alpha/logits of shape (B, L, E)

    lb_strategy: "none" | "lbl" | "aux_free"
      - "lbl":      adds Switch-Transformer-style load-balanced auxiliary loss
      - "aux_free": updates a per-expert bias buffer to steer routing without
                    any gradient-based loss term
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        lb_strategy: str = "none",
        lb_coef: float = 0.01,
        aux_free_bias_step: float = 1e-3,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.lb_strategy = lb_strategy
        self.lb_coef = lb_coef
        self.aux_free_bias_step = aux_free_bias_step
        self.proj = nn.Conv1d(d_model, n_experts, kernel_size=1)

        if lb_strategy == "aux_free":
            self.register_buffer("expert_bias", torch.zeros(n_experts))

        self.last_lb_loss = None

    def forward(
        self,
        u: torch.Tensor,
        noise_scale: float = 1.0,
        mixture_temperature: float = 1.0,
        straight_through: bool = True,
        uniform_topk_eps: float = 0.0,
    ):
        """
        u: (B, H, L)
        returns:
          alpha:  (B, L, E) sparse convex weights
          logits: (B, L, E)
        """
        logits = self.proj(u).transpose(1, 2)  # (B, L, E)

        # aux_free: bias selection scores only; weights come from unbiased logits
        if self.lb_strategy == "aux_free":
            selection_logits = logits + self.expert_bias
        else:
            selection_logits = logits

        # Selection scores: noisy during training, deterministic at eval
        if self.training:
            noisy_logits = selection_logits + noise_scale * sample_gumbel_like(selection_logits)
        else:
            noisy_logits = selection_logits

        k = min(self.top_k, self.n_experts)
        _, top_idx = noisy_logits.topk(k, dim=-1)  # (B, L, K)

        # Mixture weights computed from unbiased logits
        selected_logits = logits.gather(-1, top_idx)  # (B, L, K)

        tau = max(float(mixture_temperature), 1e-4)
        top_alpha = F.softmax(selected_logits / tau, dim=-1)  # (B, L, K)

        # Optional anti-collapse smoothing within top-k
        if uniform_topk_eps > 0.0:
            top_alpha = (1.0 - uniform_topk_eps) * top_alpha + uniform_topk_eps / k

        hard = torch.zeros_like(logits).scatter_(-1, top_idx, top_alpha)  # (B, L, E)

        if straight_through:
            # Dense surrogate gradient, sparse forward — always from unbiased logits
            soft_full = F.softmax(logits / tau, dim=-1)
            alpha = hard + soft_full - soft_full.detach()
        else:
            alpha = hard

        if self.training:
            B, L_seq = logits.shape[0], logits.shape[1]
            n_tokens = B * L_seq
            dispatch = torch.zeros(
                self.n_experts, device=logits.device, dtype=logits.dtype
            )
            dispatch.scatter_add_(
                0,
                top_idx.reshape(-1),
                torch.ones(n_tokens * k, device=logits.device, dtype=logits.dtype),
            )
            f = dispatch / (n_tokens * k)  # (E,) fraction dispatched per expert

            if self.lb_strategy == "lbl":
                # P_i: mean soft routing probability (differentiable)
                P = F.softmax(logits / tau, dim=-1).mean(dim=(0, 1))  # (E,)
                self.last_lb_loss = self.lb_coef * self.n_experts * (f.detach() * P).sum()
            else:
                self.last_lb_loss = None

            if self.lb_strategy == "aux_free":
                with torch.no_grad():
                    self.expert_bias.add_(
                        -self.aux_free_bias_step
                        * (f.float() - 1.0 / self.n_experts).sign()
                    )
        else:
            self.last_lb_loss = None

        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)  # (B, L, E)
            self.last_entropy = -(probs * torch.log(probs + 1e-9)).sum(-1).mean().item()

        return alpha, logits

    def get_auxiliary_loss(self):
        if self.lb_strategy == "lbl" and self.last_lb_loss is not None:
            return self.last_lb_loss
        return self.proj.weight.new_zeros(())
