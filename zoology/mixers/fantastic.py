# Copyright (c) 2023, Albert Gu, Tri Dao.
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from einops import rearrange, repeat
from pydantic import validate_call

from zoology.mixers.mamba_ssm.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
try:
    from causal_conv1d import causal_conv1d_fn
except:
    assert 0, print(f"Need to install causal_conv1d: pip install causal_conv1d")
try:
    from zoology.mixers.mamba_ssm.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except:
    assert 0, print(f"Need to install selective_scan_interface: pip install mamba_ssm")

from zoology.mixers.token_router.router import TokenTopKRouter


class FantasticV1(nn.Module):

    @validate_call
    def __init__(
        self,
        d_model,
        d_state: int = 16,
        d_conv:int = 4,
        expand: int = 2,
        num_experts: int = 4,
        top_k: int = 1,
        dropout: float = 0.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        conv_bias: bool = True,
        bias: bool = False,
        layer_idx = None,
        lb_strategy: str = "none",
        lb_coef: float = 0.01,
        aux_free_bias_step: float = 1e-3,
        device = None,
        dtype = None,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.layer_idx = layer_idx
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.top_k = top_k

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.router = TokenTopKRouter(
            self.d_inner,
            self.num_experts,
            self.top_k,
            lb_strategy,
            lb_coef,
            aux_free_bias_step
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # DELTA
        log_dt = torch.rand(self.num_experts, self.d_inner) \
            * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # A-matrix
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        
        # B, C, D
        B = torch.ones(self.num_experts, self.d_state, dtype=torch.float32)
        self.B = nn.Parameter(B)
        C = torch.randn(self.num_experts, self.d_state, dtype=torch.float32)
        self.C = nn.Parameter(C)
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        
    def forward(self,
        hidden_states,
        noise_scale: float = 1.0,
        mixture_temperature: float = 1.0,
        straight_through: bool = True,
        uniform_topk_eps: float = 0.0,
        inference_params=None
    ):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # print(f"mixer: {type(hidden_states)}")
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)  # (B, L, H) each

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Compute short convolution
        if conv_state is not None:
            conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )

        alpha, _ = self.router(
            x,
            noise_scale=noise_scale,
            mixture_temperature=mixture_temperature,
            straight_through=straight_through,
            uniform_topk_eps=uniform_topk_eps,
        )  # alpha: (B, L, E)

        dt = alpha @ torch.exp(self.log_dt)
        B = (alpha @ self.B).transpose(-1, -2)  # (B, N, L)
        C = (alpha @ self.C).transpose(-1, -2)  # (B, N, L)

        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

    def state_size(self, sequence_length: int=2048):
        return 2 * self.d_model * self.d_state


class FantasticV1Block(nn.Module):
    def __init__(
        self, config, fused_add_norm=True, residual_in_fp32=True, norm_epsilon=1e-5, **factory_kwargs
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        d_model = config.d_model
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        #self.mixer = config.sequence_mixer.instantiate(d_model=d_model, **factory_kwargs)
        self.mixer = FantasticV1(d_model, **factory_kwargs, **config.sequence_mixer.kwargs)
        from zoology.mixers.mamba_ssm.triton.layernorm import RMSNorm
        self.norm = RMSNorm(d_model, eps=norm_epsilon)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        noise_scale: float = 1.0,
        mixture_temperature: float = 1.0,
        straight_through: bool = True,
        uniform_topk_eps: float = 0.0,
        inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(
            hidden_states,
            inference_params=inference_params,
            noise_scale=noise_scale,
            mixture_temperature=mixture_temperature,
            straight_through=straight_through,
            uniform_topk_eps=uniform_topk_eps,
        )
        return hidden_states, residual


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def FantasticV1Init(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            # print(f"{name=}")
            if "out_proj.weight" in name or "fc2.weight" in name:
                print(f"found in initialization phase - {name=}!")
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

