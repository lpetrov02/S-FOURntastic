import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from einops import repeat, rearrange

from zoology.mixers.token_router.router import TokenTopKRouter


class CausalConv1d(nn.Module):
    """Conv1d with causal masking via left-padding."""

    def __init__(self, dim: int, kernel_size: int, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        # Causal pad = (kernel_size - 1) on the left only
        self._pad = kernel_size - 1
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        x = F.pad(x, (self._pad, 0))   # pad left only → causal
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 4,
        activation: str = "silu",      # "silu" | "gelu" | "relu"
        norm: str = "layer",           # "layer" | "rms"
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()


        # ── Normalisation ────────────────────────────────────────────────────
        if norm == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif norm == "rms":
            self.norm = nn.RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown norm: {norm!r}")

        # ── Activation ───────────────────────────────────────────────────────
        _acts = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}
        if activation not in _acts:
            raise ValueError(f"Unknown activation: {activation!r}")
        self.act = _acts[activation]()

        # ── Causal Conv1d ────────────────────────────────────────────────────
        self.conv = CausalConv1d(d_model, kernel_size, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, L]
        Returns:
            [B, D, L]  (residual connection included)
        """
        residual = x
        x = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.act(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x + residual


class TokenRoutedS4D(nn.Module):
    """
    Token-routed diagonal SSM.

    Faster version:
      - chunked parallel scan instead of Python time loop
      - avoids materializing (B, T, 2H, H) in output projection
      - uses packed bank mixing
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        n_experts: int = 32,
        top_k: int = 4,
        dropout: float = 0.0,
        transposed: bool = True,
        dt_min: float = 1e-4,
        dt_max: float = 1e-1,
        layer_idx=None,
        learnable_A_imag: bool = True,
        use_state_for_routing: bool = False,
        pre_routing_kernel_size: int = 4,
        num_conv_blocks: int = 0,
        lb_strategy: str = "none",
        lb_coef: float = 0.01,
        aux_free_bias_step: float = 1e-3,
    ):
        super().__init__()

        assert d_state % 2 == 0, "d_state must be even for complex conjugate parameterization."
        self.layer_idx = layer_idx
        self.h = d_model
        self.n = d_state
        self.n2 = d_state // 2
        self.d_output = d_model
        self.transposed = transposed
        self.n_experts = n_experts
        self.top_k = top_k
        self.lb_strategy = lb_strategy
        self.lb_coef = lb_coef
        self.last_lb_loss = None

        self.n_convs = num_conv_blocks
        self.kernel_size = pre_routing_kernel_size

        E, H, N2 = n_experts, d_model, self.n2

        # Router
        self.router = TokenTopKRouter(H, E, top_k, N2, use_state_for_routing, lb_strategy, aux_free_bias_step)
        self.conv = nn.Sequential(
            *[ConvBlock(d_model, kernel_size=self.kernel_size) for _ in range(self.n_convs)]
        ) if self.n_convs > 0 else nn.Identity()

        # Expert banks
        log_dt = torch.rand(E, H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)  # (E, H)

        log_A_real = torch.log(0.5 * torch.ones(E, H, N2))
        self.log_A_real = nn.Parameter(log_A_real)  # (E, H, N2)

        A_imag = math.pi * repeat(torch.arange(N2, dtype=torch.float32), "n -> e h n", e=E, h=H)
        if learnable_A_imag:
            self.A_imag = nn.Parameter(A_imag)  # (E, H, N2)
        else:
            self.register_buffer("A_imag", A_imag)

        B = torch.ones(E, H, N2, dtype=torch.cfloat)
        self.B = nn.Parameter(torch.view_as_real(B))  # (E, H, N2, 2)

        C = torch.randn(E, H, N2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))  # (E, H, N2, 2)

        self.D = nn.Parameter(torch.randn(E, H))  # (E, H)

        # Expert pointwise output projection: equivalent to Conv1d(H, 2H, 1) + GLU
        self.out_W = nn.Parameter(torch.empty(E, 2 * H, H))  # (E, 2H, H)
        self.out_b = nn.Parameter(torch.empty(E, 2 * H))     # (E, 2H)

        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        self.last_router_aux = None
        self.reset_parameters()

    def reset_parameters(self):
        for e in range(self.n_experts):
            nn.init.kaiming_uniform_(self.out_W[e], a=math.sqrt(5))
            fan_in = self.out_W[e].size(-1)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.out_b[e], -bound, bound)

    # ---------------- Mixing helpers ----------------

    @staticmethod
    def _mix_eo(alpha: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """
        alpha: (B, T, E)
        bank:  (E, O)
        out:   (B, T, O)
        """
        B, T, E = alpha.shape
        O = bank.shape[1]
        out = alpha.reshape(B * T, E).to(bank.dtype) @ bank
        return out.reshape(B, T, O)

    # ---------------- Materialization ----------------

    def _materialize_expert_banks(self):
        dt_bank = torch.exp(self.log_dt)                         # (E, H)
        A_bank = -torch.exp(self.log_A_real) + 1j * self.A_imag # (E, H, N2)
        B_bank = torch.view_as_complex(self.B)                  # (E, H, N2)
        C_bank = torch.view_as_complex(self.C)                  # (E, H, N2)
        D_bank = self.D                                         # (E, H)

        # Real banks packed together
        real_bank = torch.cat([dt_bank, D_bank], dim=-1).contiguous()  # (E, 2H)

        # Complex banks packed together
        E = dt_bank.shape[0]
        complex_bank = torch.cat(
            [
                A_bank.reshape(E, -1),
                B_bank.reshape(E, -1),
                C_bank.reshape(E, -1),
            ],
            dim=-1,
        ).contiguous()  # (E, 3 * H * N2)

        return real_bank, complex_bank

    # ---------------- Core scan ----------------

    def _scan_chunk(
        self,
        u_chunk: torch.Tensor,
        alpha_chunk: torch.Tensor,
        state: torch.Tensor,
        real_bank: torch.Tensor,
        complex_bank: torch.Tensor,
    ):
        """
        Exact chunked scan, but with packed bank mixing.

        u_chunk:     (B, H, T)
        alpha_chunk: (B, T, E)
        state:       (B, H, N2) complex
        """
        Bsz, H, T = u_chunk.shape
        N2 = self.n2

        u_t = u_chunk.transpose(1, 2).contiguous()  # (B, T, H)
        alpha2d = alpha_chunk.reshape(Bsz * T, self.n_experts)

        # Real mix: dt, D
        real_mix = alpha2d.to(real_bank.dtype) @ real_bank      # (BT, 2H)
        dt, D = real_mix.split(H, dim=-1)
        dt = dt.view(Bsz, T, H, 1)
        D = D.view(Bsz, T, H)

        # Complex mix: A, B, C
        complex_mix = alpha2d.to(complex_bank.dtype) @ complex_bank  # (BT, 3HN2)
        A, Bp, C = complex_mix.split(H * N2, dim=-1)
        A = A.view(Bsz, T, H, N2)
        Bp = Bp.view(Bsz, T, H, N2)
        C = C.view(Bsz, T, H, N2)

        # Discretize
        A_dt = A * dt
        Abar = torch.exp(A_dt)

        A_denom = torch.where(
            A.abs() < 1e-6,
            A + (1e-6 + 0.0j),
            A,
        )
        Bbar = ((Abar - 1.0) / A_denom) * Bp

        bu = Bbar * u_t.unsqueeze(-1)  # (B, T, H, N2)

        # Parallel scan: x_t = P_t * (x0 + cumsum(b_t / P_t))
        P = torch.cumprod(Abar, dim=1)
        P_safe = torch.where(
            P.abs() < 1e-12,
            P + (1e-12 + 0.0j),
            P,
        )

        S = torch.cumsum(bu / P_safe, dim=1)
        state_seq = P * (state.unsqueeze(1) + S)

        y = 2.0 * (C * state_seq).sum(dim=-1).real + D * u_t
        y_chunk = y.transpose(1, 2).contiguous()
        state = state_seq[:, -1]
        return y_chunk, state

    # ---------------- Output projection ----------------

    def _dynamic_output_projection(
        self,
        y_chunk: torch.Tensor,
        alpha_chunk: torch.Tensor,
    ):
        """
        Exact same math, but avoids materializing large token-specific weight tensors.
        """
        y_chunk = self.dropout(self.activation(y_chunk))
        y_t = y_chunk.transpose(1, 2).contiguous()  # (B, T, H)

        # z[b,t,o] = sum_e alpha[b,t,e] * sum_h out_W[e,o,h] * y_t[b,t,h]
        z = torch.einsum("bte,eoh,bth->bto", alpha_chunk, self.out_W, y_t)

        # bias term: sum_e alpha[b,t,e] * out_b[e,o]
        z = z + self._mix_eo(alpha_chunk, self.out_b)

        a, g = z.chunk(2, dim=-1)
        y_out = (a * torch.sigmoid(g)).transpose(1, 2).contiguous()
        return y_out

    # ---------------- Forward ----------------

    def forward(
        self,
        u: torch.Tensor,
        noise_scale: float = 1.0,
        mixture_temperature: float = 1.0,
        chunk_size: int = 64,
        straight_through: bool = True,
        return_router_aux: bool = False,
        uniform_topk_eps: float = 0.0,
        **kwargs,
    ):
        """
        Accept either:
          (B, H, L) or (B, L, H)
        and canonicalize internally to (B, H, L).
        """
        if u.ndim != 3:
            raise RuntimeError(f"Expected 3D input, got shape {tuple(u.shape)}")

        input_was_blh = False

        if u.size(1) == self.h:
            u = u.contiguous()  # already (B, H, L)
        elif u.size(-1) == self.h:
            u = u.transpose(-1, -2).contiguous()  # (B, L, H) -> (B, H, L)
            input_was_blh = True
        else:
            raise RuntimeError(
                f"Cannot infer layout for input shape {tuple(u.shape)} with d_model={self.h}. "
                "Expected either (B, H, L) or (B, L, H)."
            )

        Bsz, H, L = u.shape
        if chunk_size is None or chunk_size <= 0:
            chunk_size = L

        # Complex recurrent state
        state = torch.zeros(
            Bsz,
            H,
            self.n2,
            device=u.device,
            dtype=torch.cfloat,
        )

        # Materialize expert banks once
        real_bank, complex_bank = self._materialize_expert_banks()


        outputs = []
        accum_logits = []
        accum_alpha = []
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            u_chunk = u[:, :, start:end].contiguous()          # (B, H, T)

            # Token-level routing over full sequence
            alpha, logits = self.router(
                self.conv(u_chunk),
                noise_scale=noise_scale,
                mixture_temperature=mixture_temperature,
                straight_through=straight_through,
                uniform_topk_eps=uniform_topk_eps,
                state=state,
            )
            accum_logits.append(logits)
            accum_alpha.append(alpha)

            y_chunk, state = self._scan_chunk(
                u_chunk=u_chunk,
                alpha_chunk=alpha,
                state=state,
                real_bank=real_bank,
                complex_bank=complex_bank,
            )

            y_chunk = self._dynamic_output_projection(y_chunk, alpha)
            outputs.append(y_chunk)
        full_logits = torch.cat(accum_logits, dim=1)   # (B, L, E)
        full_alpha = torch.cat(accum_alpha, dim=1)     # (B, L, E)

        # LBL: compute from full-sequence routing data so all chunks contribute equally
        if self.training and self.lb_strategy == "lbl":
            tau = max(float(mixture_temperature), 1e-4)
            k = min(self.top_k, self.n_experts)
            Bsz, L_full = full_logits.shape[0], full_logits.shape[1]
            n_tokens = Bsz * L_full
            top_idx = full_alpha.topk(k, dim=-1).indices          # (B, L, K)
            dispatch = torch.zeros(
                self.n_experts, device=full_logits.device, dtype=full_logits.dtype
            )
            dispatch.scatter_add_(
                0,
                top_idx.reshape(-1),
                torch.ones(n_tokens * k, device=full_logits.device, dtype=full_logits.dtype),
            )
            f = dispatch / (n_tokens * k)
            P = F.softmax(full_logits / tau, dim=-1).mean(dim=(0, 1))  # (E,)
            self.last_lb_loss = self.lb_coef * self.n_experts * (f.detach() * P).sum()
        else:
            self.last_lb_loss = None

        y = torch.cat(outputs, dim=-1)  # (B, H, L)

        if input_was_blh:
            y = y.transpose(-1, -2).contiguous()  # back to (B, L, H)

        aux = {
            "routing_weights": full_alpha,
            "routing_logits": full_logits,
        }
        self.last_router_aux = aux

        if return_router_aux:
            return y, aux
        return y

    def get_auxiliary_loss(self):
        if self.lb_strategy == "lbl" and self.last_lb_loss is not None:
            return self.last_lb_loss
        return self.log_dt.new_zeros(())
    
    def state_size(self, sequence_length: int = 2048) -> int:
        return self.h * self.n * self.n_experts


class TokenRoutedS4DBlock(nn.Module):
    def __init__(
        self, config, residual_in_fp32=True, norm_epsilon=1e-5, **factory_kwargs
    ):
        """
        Block wrapping TokenRoutedS4D with LayerNorm and residual connection,
        mirroring MambaBlock and S4DBlock.

        Structure (prenorm):  Add -> LN -> TokenRoutedS4D
        Returns both hidden_states and residual so the caller can chain blocks.
        """
        super().__init__()
        d_model = config.d_model
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = TokenRoutedS4D(d_model, **factory_kwargs, **config.sequence_mixer.kwargs)
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


def TokenRoutedS4DInit(
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
            # out_W is the per-expert output projection (E, 2H, H), analogous to out_proj.weight
            if "out_W" in name:
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)
