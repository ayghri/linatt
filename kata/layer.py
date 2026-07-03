"""KATA attention layer. nn.Module mirroring fla.layers.LinearAttention.

Differences from FLA LinearAttention:
- feature_map can be 'positive' | 'lorentz' | 'spd_concat' (KATA-specific)
- always runs with denominator normalization (`do_feature_map_norm=True`)
- uses fused recurrent for short prefills, chunk for long, just like the FLA
  reference, so prefill and decode both work via FLA's KV cache helpers.
"""

from typing import TYPE_CHECKING

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import RMSNorm, RotaryEmbedding, ShortConvolution
from fla.modules.l2norm import l2norm  # the function (fla.modules.l2norm is the submodule)
from fla.ops.linear_attn import (
    chunk_linear_attn,
    fused_chunk_linear_attn,
    fused_recurrent_linear_attn,
)

from kata.feature_maps import FeatureMap, feature_map_out_dim
from kata.parallel_kata_attn import (
    parallel_kata_attn,
    parallel_kata_attn_decay,
    parallel_kata_attn_fwd,
    parallel_kata_attn_sum,
    parallel_kata_attn_sum_fwd_impl,
)

if TYPE_CHECKING:
    from fla.models.utils import Cache


class KataAttention(nn.Module):

    def __init__(
        self,
        mode: str = "chunk",
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        feature_map: str = "positive",
        spd_num_groups: int = 1,
        spd_use_kernel: bool = False,
        spd_chunk_size: int = 32,
        use_offset_gate: bool = False,
        use_decay: bool = False,
        feature_map_eps: float = 1e-6,
        qk_norm: bool = False,
        norm_q: str = "rmsnorm",
        norm_k: str = "rmsnorm",
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        output_norm: str = "rmsnorm",
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        rope_group: bool = False,
        max_position_embeddings: int = 2048,
        layer_idx: int | None = None,
        **kwargs,
    ):
        super().__init__()

        if mode not in ("chunk", "fused_chunk", "fused_recurrent"):
            raise ValueError(f"unsupported mode {mode!r}")

        self.hidden_size = hidden_size
        self.mode = mode
        self.num_heads = num_heads
        self.num_kv_heads = (
            num_kv_heads if num_kv_heads is not None else num_heads
        )
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups

        if self.key_dim % num_heads != 0:
            raise ValueError(
                f"key_dim {self.key_dim} not divisible by num_heads {num_heads}"
            )
        if self.value_dim % num_heads != 0:
            raise ValueError(
                f"value_dim {self.value_dim} not divisible by num_heads {num_heads}"
            )

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx

        # qk-norm on q/k before the score (after conv, same position as GDN).
        # qk_norm (bool) = master on/off; norm_q/norm_k = the TYPE per side:
        #   "rmsnorm" -> transformer-style RMSNorm (learnable scale)
        #   "l2"      -> GDN-style parameter-free l2-norm (x/||x||): q.k in [-1,1] so
        #               the SPD square (q.k)^2 stays in [0,1] (well-conditioned).
        # Backward-compat: pre-API configs passed bool norm_q/norm_k; qk_norm now
        # gates whether they are used, so coerce any bool to a valid type string.
        if isinstance(norm_q, bool):
            norm_q = "rmsnorm"
        if isinstance(norm_k, bool):
            norm_k = "rmsnorm"
        for _nm, _v in (("norm_q", norm_q), ("norm_k", norm_k)):
            if _v not in ("rmsnorm", "l2"):
                raise ValueError(f"{_nm} must be 'rmsnorm' or 'l2', got {_v!r}")
        self.qk_norm = qk_norm
        if qk_norm and norm_q == "rmsnorm":
            self.q_norm = RMSNorm(self.head_k_dim, dtype=torch.float32)
        if qk_norm and norm_k == "rmsnorm":
            self.k_norm = RMSNorm(self.head_k_dim, dtype=torch.float32)

        # RoPE applied to q,k before the SPD score (same as self-attention).
        self.use_rope = use_rope
        self.rope_group = rope_group
        self.max_position_embeddings = max_position_embeddings
        if use_rope:
            if rope_group:
                # Per-group RoPE: rotate each E-dim group independently with the SAME
                # frequencies (RotaryEmbedding(dim=E)). Every group is then full-spectrum
                # AND relative-position, with an IDENTICAL positional factor across groups.
                self.rotary = RotaryEmbedding(
                    dim=self.head_k_dim // spd_num_groups, base=rope_theta
                )
            else:
                # Full-head RoPE, interleaved=True: pairs (2i,2i+1) stay within a contiguous
                # group so each per-group dot is relative-position, but groups get DISJOINT
                # frequency bands (low->group0, high->groupM-1). interleaved=False would
                # straddle the group boundary and leak absolute position.
                self.rotary = RotaryEmbedding(
                    dim=self.head_k_dim, base=rope_theta, interleaved=True
                )

        self.feature_map_name = feature_map
        self.spd_num_groups = spd_num_groups
        self.spd_use_kernel = spd_use_kernel
        self.spd_chunk_size = spd_chunk_size

        if spd_use_kernel:
            raise NotImplementedError(
                "spd_use_kernel (the chunk_kata_spd path) was removed; use "
                "feature_map='kata_quadratic' (parallel_kata_attn) instead."
            )

        self.feature_map = FeatureMap(
            name=feature_map,
            num_groups=spd_num_groups,
            eps=feature_map_eps,
        )
        self.head_q_dim = feature_map_out_dim(
            feature_map, self.head_k_dim, spd_num_groups
        )

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(
            hidden_size, self.value_dim_per_group, bias=False
        )

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        if output_norm == "rmsnorm":
            self.norm = RMSNorm(
                hidden_size=self.head_v_dim,
                elementwise_affine=elementwise_affine,
                eps=norm_eps,
                dtype=torch.float32,
            )
        elif output_norm == "identity":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"unsupported output_norm {output_norm!r}")

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Data-dependent affine offset for the SPD score: A=(a+<q,k>)^2 with
        # a_ts = sigmoid(a_proj(x_t)) * sigmoid(a_proj(x_s)) in [0,1]. Implemented
        # by appending the (scaled) gate as an extra coordinate per group + zero-pad
        # to a multiple of 16, so the existing squared-dot kernel computes the offset
        # with no kernel change; autograd flows the gate gradient back to a_proj.
        self.use_offset_gate = use_offset_gate
        if use_offset_gate:
            if feature_map not in ("kata_quadratic", "kata_quadratic_sum"):
                raise ValueError(
                    "use_offset_gate requires feature_map in "
                    "{kata_quadratic, kata_quadratic_sum}"
                )
            if self.num_kv_groups != 1:
                raise ValueError("use_offset_gate currently requires MHA (num_kv_heads=num_heads)")
            self._off_E = self.head_k_dim // spd_num_groups
            # kernel requires the group dim to be a power of 2; pad to next pow2 > E
            self._off_Epad = max(16, 1 << self._off_E.bit_length())
            self._off_scale = 1.0 / math.sqrt(self._off_E)          # = kernel default for orig E
            self.a_proj = nn.Linear(hidden_size, num_heads, bias=True)

        # Data-dependent GDN-style recency decay: score *= exp(c_t - c_s),
        # c = cumsum( -exp(A_log)*softplus(dt_proj(x)+dt_bias) ). Soft "replacement".
        self.use_decay = use_decay
        if use_decay:
            if feature_map != "kata_quadratic":
                raise ValueError("use_decay currently requires feature_map=kata_quadratic (concat)")
            self.dt_proj = nn.Linear(hidden_size, num_heads, bias=False)
            A = torch.empty(num_heads).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
            dt = torch.exp(
                torch.rand(num_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            ).clamp(min=1e-4)
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus

        self.norm_q = norm_q
        self.norm_k = norm_k

    def _augment_offset(self, q, k, off):
        """Per-token, per-head additive offset: append off_t to each group of q and a
        constant 1 to each group of k (zero-pad the group to a power-of-2 dim), so the
        squared-dot kernel yields (<q_i,k_i> + off_t)^2 per group. off: (B,T,H)."""
        B, T, H, _ = q.shape
        M, E, Ep = self.spd_num_groups, self._off_E, self._off_Epad

        def _pack(x, col):                      # col: (B,T,H) -> one coord per group
            xg = x.view(B, T, H, M, E)
            c = col[..., None, None].expand(B, T, H, M, 1).to(x.dtype)
            z = x.new_zeros(B, T, H, M, Ep - E - 1)
            return torch.cat([xg, c, z], dim=-1).reshape(B, T, H, M * Ep)

        return _pack(q, off), _pack(k, torch.ones_like(off))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: "Cache | None" = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        # short prefill / decode -> recurrent kernel; long -> requested chunked path.
        mode = "fused_recurrent" if hidden_states.shape[1] <= 64 else self.mode
        last_state = get_layer_cache(self, past_key_values)

        if self.use_short_conv:
            conv_state_q = conv_state_k = conv_state_v = None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state.get(
                    "conv_state", (None, None, None)
                )
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        if attention_mask is not None:
            v = v.mul(attention_mask[:, -v.shape[-2] :, None])

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        if self.num_kv_groups > 1:
            k = repeat(
                k,
                "... (h d) -> ... (h g) d",
                d=self.head_k_dim,
                g=self.num_kv_groups,
            )
            v = repeat(
                v,
                "... (h d) -> ... (h g) d",
                d=self.head_v_dim,
                g=self.num_kv_groups,
            )
        else:
            k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
            v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        # qk-norm THEN RoPE on q,k (B,T,H,head_k_dim) before the SPD score -- exact
        # transformer order (qk_norm at fla Attention L106, rotary at L124).
        if self.qk_norm:
            # l2 -> fla's FUSED l2norm (single Triton kernel, fp32 internal, no extra
            # bf16<->fp32 copies); rmsnorm -> the RMSNorm module.
            q = l2norm(q) if self.norm_q == "l2" else self.q_norm(q)
            k = l2norm(k) if self.norm_k == "l2" else self.k_norm(k)
        if self.use_rope:
            seqlen_offset = 0
            if past_key_values is not None and self.layer_idx is not None:
                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = max(
                q.shape[1] + seqlen_offset, self.max_position_embeddings
            )
            if self.rope_group:
                # treat each E-dim group as its own head: (B,T,H,D) -> (B,T,H*M,E),
                # rotate with RotaryEmbedding(dim=E), reshape back.
                Bz, Tl, Hq = q.shape[0], q.shape[1], q.shape[2]
                Hk = k.shape[2]
                E = self.head_k_dim // self.spd_num_groups
                qg = q.reshape(Bz, Tl, Hq * self.spd_num_groups, E)
                kg = k.reshape(Bz, Tl, Hk * self.spd_num_groups, E)
                qg, kg = self.rotary(
                    qg, kg, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen
                )
                q = qg.reshape(Bz, Tl, Hq, self.head_k_dim)
                k = kg.reshape(Bz, Tl, Hk, self.head_k_dim)
            else:
                q, k = self.rotary(
                    q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen
                )

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        if self.feature_map_name in ("kata_quadratic", "kata_quadratic_sum"):
            # FlashAttention-style quadratic kata; psi never materialized.
            # `kata_quadratic`     -> concat-SPD score Sum_i (q_i.k_i)^2
            # `kata_quadratic_sum` -> sum-SPD score   Sum_{i,j} (q_i.k_j)^2
            #                         (matches paper KATA-SPD-4 math).
            is_sum = self.feature_map_name == "kata_quadratic_sum"
            # data-dependent affine offset (sign/suppression): append sigmoid gate
            # as an extra group coordinate; score becomes (<q,k> + g_t g_s)^2.
            q_in, k_in, scale_in = q, k, None
            if self.use_offset_gate:
                off = torch.sigmoid(self.a_proj(hidden_states))    # (B,T,H) in [0,1], per token/head
                q_in, k_in = self._augment_offset(q, k, off)
                scale_in = self._off_scale
            grad_on = torch.is_grad_enabled() and any(
                t.requires_grad for t in (q_in, k_in, v)
            )
            if self.use_decay:
                dt = F.softplus(self.dt_proj(hidden_states) + self.dt_bias)   # (B,T,H) >0
                c = (-self.A_log.float().exp() * dt.float()).cumsum(dim=1).contiguous()
                o = parallel_kata_attn_decay(
                    q_in, k_in, v, c,
                    num_groups=self.spd_num_groups, scale=scale_in,
                )
            elif grad_on:
                if is_sum:
                    o = parallel_kata_attn_sum(
                        q_in,
                        k_in,
                        v,
                        num_groups=self.spd_num_groups,
                        scale=scale_in,
                    )
                else:
                    o = parallel_kata_attn(
                        q_in,
                        k_in,
                        v,
                        num_groups=self.spd_num_groups,
                        scale=scale_in,
                    )
            else:
                if is_sum:
                    o, _ = parallel_kata_attn_sum_fwd_impl(
                        q_in,
                        k_in,
                        v,
                        self.spd_num_groups,
                        scale_in,
                        64,
                        64,
                    )
                else:
                    o, _ = parallel_kata_attn_fwd(
                        q_in,
                        k_in,
                        v,
                        num_groups=self.spd_num_groups,
                        scale=scale_in,
                    )
            final_state = None
        else:
            # pre-materialize psi(q), psi(k) and route through FLA kernels
            q = self.feature_map(q)
            k = self.feature_map(k)
            if self.qk_norm:  # legacy psi-feature sum-norm (positive/lorentz maps)
                q = q / (q.sum(-1, True) + 1e-4)
                k = k / (k.sum(-1, True) + 1e-4)
            if mode == "chunk":
                o, final_state = chunk_linear_attn(
                    q=q,
                    k=k,
                    v=v,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    normalize=True,
                )
            elif mode == "fused_chunk":
                o, final_state = fused_chunk_linear_attn(
                    q=q,
                    k=k,
                    v=v,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    normalize=True,
                )
            elif mode == "fused_recurrent":
                o, final_state = fused_recurrent_linear_attn(
                    q=q,
                    k=k,
                    v=v,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    normalize=True,
                )
            else:
                raise NotImplementedError

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=final_state,
            offset=q.shape[1],
        )

        o = self.norm(o)
        o = rearrange(o, "... h d -> ... (h d)")
        o = self.o_proj(o)
        return o, None, past_key_values
