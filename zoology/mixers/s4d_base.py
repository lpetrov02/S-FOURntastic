import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from einops import repeat


class S4D(nn.Module):
    """
    Base S4D mixer: Structured State Spaces with Diagonal SSMs.

    Simplification of Mamba:
      - Removes causal conv1d, x_proj, dt_proj (Mamba's input-dependent scan).
      - Replaces selective scan with a fixed diagonal SSM (A, B, C, D, log_dt are
        learned but not input-dependent).
      - Uses ZOH discretization and FFT-based convolution for efficient training.

    Parameters are per-channel (d_inner channels, N/2 complex conjugate pairs each),
    initialized with S4D-Lin: A_n = -1/2 + i*pi*n.

    Input/output: (B, L, D)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        dropout: float = 0.0,
        dt_min: float = 1e-4,
        dt_max: float = 1e-1,
        layer_idx=None,
        learnable_A_imag: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert d_state % 2 == 0, "d_state must be even for complex conjugate pairs"
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.n2 = d_state // 2  # complex conjugate pairs per channel
        self.layer_idx = layer_idx

        H, N2 = self.d_inner, self.n2

        # Input expands to (x, z) like Mamba; z serves as the GLU gate
        self.in_proj = nn.Linear(d_model, 2 * H, bias=False)
        self.out_proj = nn.Linear(H, d_model, bias=False)

        # Step size: one scalar per channel
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # A real part: kept negative via -exp(log_A_real); init at -1/2
        log_A_real = torch.log(0.5 * torch.ones(H, N2))
        self.log_A_real = nn.Parameter(log_A_real)

        # A imaginary part: S4D-Lin initialization pi*n
        A_imag = math.pi * repeat(torch.arange(N2, dtype=torch.float32), "n -> h n", h=H)
        if learnable_A_imag:
            self.A_imag = nn.Parameter(A_imag)
        else:
            self.register_buffer("A_imag", A_imag)

        # B: input-to-state (fixed, not input-dependent unlike Mamba)
        B = torch.ones(H, N2, dtype=torch.cfloat)
        self.B = nn.Parameter(torch.view_as_real(B))

        # C: state-to-output
        C = torch.randn(H, N2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))

        # D: skip connection (u passes directly to output)
        self.D = nn.Parameter(torch.ones(H))

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _ssm_kernel(self, L: int) -> torch.Tensor:
        """
        Compute the causal SSM convolution kernel via ZOH discretization.

        K[h, t] = 2 * Re( sum_n  C[h,n] * Bbar[h,n] * Abar[h,n]^t )

        Returns: (H, L) real tensor
        """
        dt = torch.exp(self.log_dt)                              # (H,)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag      # (H, N2) complex
        B = torch.view_as_complex(self.B.contiguous())           # (H, N2) complex
        C = torch.view_as_complex(self.C.contiguous())           # (H, N2) complex

        # ZOH discretization
        dtA = A * dt.unsqueeze(-1)                               # (H, N2)
        Abar = torch.exp(dtA)                                    # (H, N2)
        # Guard against A ≈ 0 to avoid division by zero in (Abar-1)/A
        A_safe = torch.where(A.abs() < 1e-6, A + 1e-6, A)
        Bbar = B * (Abar - 1.0) / A_safe                        # (H, N2)

        # Raise Abar to integer powers 0..L-1 using log for numerical stability
        log_Abar = torch.log(Abar + 1e-30)                       # (H, N2) complex
        t = torch.arange(L, device=self.log_dt.device, dtype=torch.float32)
        powers = torch.exp(t.view(1, 1, L) * log_Abar.unsqueeze(-1))  # (H, N2, L)

        CB = (C * Bbar).unsqueeze(-1)                            # (H, N2, 1)
        K = 2.0 * (CB * powers).sum(dim=1).real                 # (H, L)
        return K

    def _fft_conv(self, u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Causal linear convolution via FFT.
        u: (B, H, L),  K: (H, L)  ->  (B, H, L)
        """
        L = u.shape[-1]
        fft_len = 2 * L  # zero-pad to avoid circular wrap-around
        U = torch.fft.rfft(u.float(), n=fft_len)
        K_ = torch.fft.rfft(K.float(), n=fft_len)
        Y = U * K_.unsqueeze(0)
        return torch.fft.irfft(Y, n=fft_len)[..., :L].to(u.dtype)

    def forward(self, hidden_states, **kwargs):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        B, L, D = hidden_states.shape

        # Project to (x, z): x goes through SSM, z gates the output
        xz = self.in_proj(hidden_states)                         # (B, L, 2*H)
        x, z = xz.chunk(2, dim=-1)                              # (B, L, H) each

        x = x.transpose(1, 2)                                   # (B, H, L)

        K = self._ssm_kernel(L)                                  # (H, L)
        y = self._fft_conv(x, K) + self.D.unsqueeze(-1) * x    # (B, H, L)

        y = y.transpose(1, 2)                                   # (B, L, H)
        y = self.act(y) * torch.sigmoid(z)                      # GLU gate
        y = self.dropout(y)
        y = self.out_proj(y)                                     # (B, L, D)
        return y

    def state_size(self, sequence_length: int = 2048) -> int:
        return self.d_inner * self.d_state


class S4DBlock(nn.Module):
    def __init__(
        self, config, residual_in_fp32=True, norm_epsilon=1e-5, **factory_kwargs
    ):
        """
        Block wrapping S4D with LayerNorm and residual connection, mirroring MambaBlock.

        Structure (prenorm):  Add -> LN -> S4D
        Returns both hidden_states and residual so the caller can chain blocks.
        """
        super().__init__()
        d_model = config.d_model
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = S4D(d_model, **factory_kwargs, **config.sequence_mixer.kwargs)
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


def S4DInit(
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
