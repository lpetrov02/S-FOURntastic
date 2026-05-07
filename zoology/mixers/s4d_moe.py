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


class S4DMoEv1(nn.Module):
    """
    MoE-style S4D: tokens are routed to top-k experts; only chosen experts
    receive the token as input (masked-input approach).  Each expert is a
    full S4D model with d_inner channels.

    Training: FFT convolution over all E×H expert-channel pairs at once,
    then weight outputs by routing scores and sum across experts.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        num_experts: int = 4,
        top_k: int = 1,
        dropout: float = 0.0,
        dt_min: float = 1e-4,
        dt_max: float = 1e-1,
        layer_idx=None,
        learnable_A_imag: bool = True,
        lb_strategy: str = "none",
        lb_coef: float = 0.01,
        aux_free_bias_step: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        assert d_state % 2 == 0, "d_state must be even for complex conjugate pairs"
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.n2 = d_state // 2  # complex conjugate pairs per channel
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.top_k = top_k

        H, N2, E = self.d_inner, self.n2, num_experts

        self.router = TokenTopKRouter(H, E, top_k, lb_strategy, lb_coef, aux_free_bias_step)

        # Input expands to (x, z) like Mamba; z serves as the GLU gate
        self.in_proj = nn.Linear(d_model, 2 * H, bias=False)
        self.out_proj = nn.Linear(H, d_model, bias=False)

        # SSM parameters: (E, H, ...) — one independent set per expert
        log_dt = torch.rand(E, H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)                           # (E, H)

        log_A_real = torch.log(0.5 * torch.ones(E, H, N2))
        self.log_A_real = nn.Parameter(log_A_real)                   # (E, H, N2)

        A_imag = math.pi * repeat(torch.arange(N2, dtype=torch.float32), "n -> e h n", e=E, h=H)
        if learnable_A_imag:
            self.A_imag = nn.Parameter(A_imag)                       # (E, H, N2)
        else:
            self.register_buffer("A_imag", A_imag)

        B = torch.ones(E, H, N2, dtype=torch.cfloat)
        self.B = nn.Parameter(torch.view_as_real(B))                 # (E, H, N2, 2)

        C = torch.randn(E, H, N2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))                 # (E, H, N2, 2)

        self.D = nn.Parameter(torch.randn(E, H))                     # (E, H)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _ssm_kernel(self, L: int) -> torch.Tensor:
        """
        Compute SSM convolution kernels for all experts via ZOH discretization.

        K[e, h, t] = 2 * Re( sum_n  C[e,h,n] * Bbar[e,h,n] * Abar[e,h,n]^t )

        Returns: (E*H, L) real tensor
        """
        dt = torch.exp(self.log_dt)                                   # (E, H)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag           # (E, H, N2) complex
        B = torch.view_as_complex(self.B.contiguous())                # (E, H, N2) complex
        C = torch.view_as_complex(self.C.contiguous())                # (E, H, N2) complex

        # ZOH discretization
        dtA = A * dt.unsqueeze(-1)                                    # (E, H, N2)
        Abar = torch.exp(dtA)                                         # (E, H, N2)
        A_safe = torch.where(A.abs() < 1e-6, A + 1e-6, A)
        Bbar = B * (Abar - 1.0) / A_safe                             # (E, H, N2)

        log_Abar = torch.log(Abar + 1e-30)                           # (E, H, N2) complex
        t = torch.arange(L, device=self.log_dt.device, dtype=torch.float32)
        powers = torch.exp(t.view(1, 1, 1, L) * log_Abar.unsqueeze(-1))  # (E, H, N2, L)

        CB = (C * Bbar).unsqueeze(-1)                                 # (E, H, N2, 1)
        K = 2.0 * (CB * powers).sum(dim=2).real                      # (E, H, L)
        return K.reshape(self.num_experts * self.d_inner, L)          # (E*H, L)

    def _fft_conv(self, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Causal linear convolution via FFT.
        u: (B, E*H, L),  K: (E*H, L)  ->  (B, E*H, L)
        """
        L = u.shape[-1]
        fft_len = 2 * L  # zero-pad to avoid circular wrap-around
        U = torch.fft.rfft(u.float(), n=fft_len)
        K_ = torch.fft.rfft(K.float(), n=fft_len)
        Y = U * K_.unsqueeze(0)
        return torch.fft.irfft(Y, n=fft_len)[..., :L].to(u.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        noise_scale: float = 1.0,
        mixture_temperature: float = 1.0,
        straight_through: bool = True,
        uniform_topk_eps: float = 0.0,
        **kwargs,
    ):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        B, L, _ = hidden_states.shape
        E, H = self.num_experts, self.d_inner

        # Project to (x, z): x goes through SSM, z gates the output
        xz = self.in_proj(hidden_states)                              # (B, L, 2*H)
        x, z = xz.chunk(2, dim=-1)                                   # (B, L, H) each
        x = x.transpose(1, 2)                                        # (B, H, L)

        alpha, _ = self.router(
            x,
            noise_scale=noise_scale,
            mixture_temperature=mixture_temperature,
            straight_through=straight_through,
            uniform_topk_eps=uniform_topk_eps,
        )  # alpha: (B, L, E)

        # All experts see the full input; routing scores gate at output.
        # Avoiding a binary mask here is critical: mask = (alpha > 0) would zero the
        # SSM output for unchosen experts, destroying the gradient signal that the
        # straight-through estimator needs to train the router.
        x_flat = x.unsqueeze(1).expand(-1, E, -1, -1).contiguous()  # (B, E, H, L)
        x_flat = x_flat.reshape(B, E * H, L)                         # (B, E*H, L)

        K = self._ssm_kernel(L)                                       # (E*H, L)
        D_flat = self.D.reshape(E * H, 1)
        y = self._fft_conv(x_flat, K) + D_flat * x_flat             # (B, E*H, L)

        # Weight by routing scores and sum over experts
        y = y.reshape(B, E, H, L)
        alpha_bel = alpha.permute(0, 2, 1).unsqueeze(2)             # (B, E, 1, L)
        y = (y * alpha_bel).sum(dim=1)                               # (B, H, L)

        y = y.transpose(1, 2)                                        # (B, L, H)
        y = self.act(y) * torch.sigmoid(z)                           # GLU gate
        y = self.dropout(y)
        y = self.out_proj(y)                                         # (B, L, D)
        return y

    def state_size(self, sequence_length: int = 2048) -> int:
        return self.d_inner * self.d_state * self.num_experts
    

class S4DMoEv2(nn.Module):
    """
    MoE-style S4D: tokens are routed to top-k experts; only chosen experts
    receive the token as input (masked-input approach).  Each expert is a
    full S4D model with d_inner channels.

    Training: FFT convolution over all E×H expert-channel pairs at once,
    then weight outputs by routing scores and sum across experts.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        num_experts: int = 4,
        dropout: float = 0.0,
        dt_min: float = 1e-4,
        dt_max: float = 1e-1,
        layer_idx=None,
        learnable_A_imag: bool = True,
        lb_strategy: str = "none",
        lb_coef: float = 0.01,
        aux_free_bias_step: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        assert d_state % 2 == 0, "d_state must be even for complex conjugate pairs"
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.n2 = d_state // 2  # complex conjugate pairs per channel
        self.layer_idx = layer_idx
        self.num_experts = num_experts

        H, N2, E = self.d_inner, self.n2, num_experts

        self.router = TokenTopKRouter(H, E, 1, lb_strategy, lb_coef, aux_free_bias_step)

        # Input expands to (x, z) like Mamba; z serves as the GLU gate
        self.in_proj = nn.Linear(d_model, 2 * H, bias=False)
        self.out_proj = nn.Linear(H, d_model, bias=False)

        # SSM parameters: (E, H, ...) — one independent set per expert
        log_dt = torch.rand(E, H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)                           # (E, H)

        log_A_real = torch.log(0.5 * torch.ones(E, H, N2))
        self.log_A_real = nn.Parameter(log_A_real)                   # (E, H, N2)

        A_imag = math.pi * repeat(torch.arange(N2, dtype=torch.float32), "n -> e h n", e=E, h=H)
        if learnable_A_imag:
            self.A_imag = nn.Parameter(A_imag)                       # (E, H, N2)
        else:
            self.register_buffer("A_imag", A_imag)

        B = torch.ones(E, H, N2, dtype=torch.cfloat)
        self.B = nn.Parameter(torch.view_as_real(B))                 # (E, H, N2, 2)

        C = torch.randn(E, H, N2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))                 # (E, H, N2, 2)

        self.D = nn.Parameter(torch.randn(E, H))                     # (E, H)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _ssm_kernel(self, L: int, chosen_experts: torch.Tensor) -> torch.Tensor:
        """
        Compute SSM convolution kernels for all experts via ZOH discretization.

        K[e, h, t] = 2 * Re( sum_n  C[e,h,n] * Bbar[e,h,n] * Abar[e,h,n]^t )

        Returns: (B, H, L) real tensor
        """
        dt = torch.exp(self.log_dt)                                   # (E, H)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag           # (E, H, N2) complex
        B = torch.view_as_complex(self.B.contiguous())                # (E, H, N2) complex
        C = torch.view_as_complex(self.C.contiguous())                # (E, H, N2) complex

        # ZOH discretization
        dtA = A * dt.unsqueeze(-1)                                    # (E, H, N2)
        Abar = torch.exp(dtA)                                         # (E, H, N2)
        A_safe = torch.where(A.abs() < 1e-6, A + 1e-6, A)
        Bbar = B * (Abar - 1.0) / A_safe                             # (E, H, N2)

        log_Abar = torch.log(Abar + 1e-30)                           # (E, H, N2) complex
        powers = torch.cumsum(log_Abar[chosen_experts], dim=1)       # (B, L, H, N2)

        CB = (C * Bbar)[chosen_experts]                              # (B, L, H, N2)
        K = 2.0 * (CB * powers).sum(dim=-1).real.permute(0, 2, 1)    # (B, H, L)
        return K

    def _fft_conv(self, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Causal linear convolution via FFT.
        u: (B, E*H, L),  K: (E*H, L)  ->  (B, E*H, L)
        """
        L = u.shape[-1]
        fft_len = 2 * L  # zero-pad to avoid circular wrap-around
        U = torch.fft.rfft(u.float(), n=fft_len)
        K_ = torch.fft.rfft(K.float(), n=fft_len)
        Y = U * K_
        return torch.fft.irfft(Y, n=fft_len)[..., :L].to(u.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        noise_scale: float = 1.0,
        mixture_temperature: float = 1.0,
        straight_through: bool = True,
        uniform_topk_eps: float = 0.0,
        **kwargs,
    ):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        B, L, _ = hidden_states.shape
        E, H = self.num_experts, self.d_inner

        # Project to (x, z): x goes through SSM, z gates the output
        xz = self.in_proj(hidden_states)                              # (B, L, 2*H)
        x, z = xz.chunk(2, dim=-1)                                   # (B, L, H) each
        x = x.transpose(1, 2)                                        # (B, H, L)

        alpha, _ = self.router(
            x,
            noise_scale=noise_scale,
            mixture_temperature=mixture_temperature,
            straight_through=straight_through,
            uniform_topk_eps=uniform_topk_eps,
        )  # alpha: (B, L, E)

        chosen_experts = alpha.argmax(dim=-1)  # consistent with router's actual selection

        K = self._ssm_kernel(L, chosen_experts)                                       # (E*H, L)
        y = self._fft_conv(x, K) + self.D[chosen_experts].permute(0, 2, 1) * x
        y = y * alpha.sum(-1).unsqueeze(1)

        # Weight by routing scores and sum over experts                                      # (B, L, H)
        y = self.act(y) * torch.sigmoid(z)                           # GLU gate
        y = self.dropout(y)
        y = self.out_proj(y) 
        return y

    def state_size(self, sequence_length: int = 2048) -> int:
        return self.d_inner * self.d_state * self.num_experts


class S4DMoEv1Block(nn.Module):
    def __init__(
        self, config, residual_in_fp32=True, norm_epsilon=1e-5, **factory_kwargs
    ):
        """
        Block wrapping S4DMoEv1 with LayerNorm and residual connection, mirroring MambaBlock.

        Structure (prenorm):  Add -> LN -> S4DMoEv1
        Returns both hidden_states and residual so the caller can chain blocks.
        """
        super().__init__()
        d_model = config.d_model
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = S4DMoEv1(d_model, **factory_kwargs, **config.sequence_mixer.kwargs)
        self.norm = nn.LayerNorm(d_model, eps=norm_epsilon)

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None
    ):
        """
        hidden_states: the sequence to the encoder layer (required).
        residual: hidden_states = Mixer(LN(residual))
        """
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


class S4DMoEv2Block(nn.Module):
    def __init__(
        self, config, residual_in_fp32=True, norm_epsilon=1e-5, **factory_kwargs
    ):
        """
        Block wrapping S4DMoEv2 with LayerNorm and residual connection, mirroring MambaBlock.

        Structure (prenorm):  Add -> LN -> S4DMoEv2
        Returns both hidden_states and residual so the caller can chain blocks.
        """
        super().__init__()
        d_model = config.d_model
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = S4DMoEv2(d_model, **factory_kwargs, **config.sequence_mixer.kwargs)
        self.norm = nn.LayerNorm(d_model, eps=norm_epsilon)

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None
    ):
        """
        hidden_states: the sequence to the encoder layer (required).
        residual: hidden_states = Mixer(LN(residual))
        """
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


def S4DMoEInit(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if "out_proj.weight" in name:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)
