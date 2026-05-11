import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from einops import repeat, rearrange


class LowRankMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, low_rank_dim, vdim=None):
        super().__init__()
        self.low_rank_dim = low_rank_dim  # 16 total, so 2-dim per head with 8 heads
        self.vdim = vdim or embed_dim

        # Q and K project DOWN to low-rank space
        self.q_proj = nn.Linear(embed_dim, low_rank_dim)
        self.k_proj = nn.Linear(embed_dim, low_rank_dim)

        self.scale = self.low_rank_dim ** -0.5

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        B, T, _ = q.shape
        S = k.shape[1]

        Q = self.q_proj(q)
        K = self.k_proj(k)

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, heads, T, S)

        if attn_mask is not None:
            attn = attn + attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

        attn = attn.softmax(dim=-1)
        return attn @ v


class S4DMoEAttn(nn.Module):
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
        attn_dim=16,
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

        self.router = LowRankMultiheadAttention(
            H,
            low_rank_dim=attn_dim,
        )

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

        CB = (C * Bbar).unsqueeze(-1)
        K = 2.0 * (CB * powers).sum(dim=-2).real
        return K

    def _fft_conv(self, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Causal linear convolution via FFT.
        u: (B, E, H, L),  K: (E, H, L)  ->  (B, E, H, L)
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

        x_flat = x.unsqueeze(1).expand(-1, E, -1, -1).contiguous()  # (B, E, H, L)

        K = self._ssm_kernel(L)                               # (H, L), (H, N2, L)
        y = self._fft_conv(x_flat, K)                              # (B, E, H, L)
        y = y + self.D[None, :, :, None] * x_flat

        # Weight by routing scores and sum over experts
        y = y.permute(0, 3, 1, 2).flatten(0, 1)  # (BL, E, H)
        x = x.permute(0, 2, 1).flatten(0, 1).unsqueeze(1)  # (BL, 1, H)
        # y = self.router(x, y, y).squeeze(1).reshape(B, L, H)  # (BL, 1, H)
        y = y.mean(1).reshape(B, L, H)

        y = self.act(y) * torch.sigmoid(z)                           # GLU gate
        y = self.dropout(y)
        y = self.out_proj(y)                                         # (B, L, D)
        return y

    def state_size(self, sequence_length: int = 2048) -> int:
        return self.d_inner * self.d_state * self.num_experts


class S4DMoEAttnBlock(nn.Module):
    def __init__(
        self, config, residual_in_fp32=True, norm_epsilon=1e-5, **factory_kwargs
    ):
        """
        Block wrapping S4DMoEv3 with LayerNorm and residual connection, mirroring MambaBlock.

        Structure (prenorm):  Add -> LN -> S4DMoEAttnM
        Returns both hidden_states and residual so the caller can chain blocks.
        """
        super().__init__()
        d_model = config.d_model
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = S4DMoEAttn(d_model, **factory_kwargs, **config.sequence_mixer.kwargs)
        self.norm = nn.LayerNorm(d_model, eps=norm_epsilon)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
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

def S4DMoEAttnInit(
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
