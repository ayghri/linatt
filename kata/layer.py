"""KATA attention layer. nn.Module mirroring fla.layers.LinearAttention.

Differences from FLA LinearAttention:
- feature_map can be 'positive' | 'lorentz' | 'spd_concat' (KATA-specific)
- always runs with denominator normalization (`do_feature_map_norm=True`)
- uses fused recurrent for short prefills, chunk for long, just like the FLA
  reference, so prefill and decode both work via FLA's KV cache helpers.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange, repeat

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules import RMSNorm, RotaryEmbedding, ShortConvolution
from fla.ops.linear_attn import (
    chunk_linear_attn,
    fused_chunk_linear_attn,
    fused_recurrent_linear_attn,
)

from kata.feature_maps import FeatureMap, feature_map_out_dim
from kata.parallel_kata_attn import (
    parallel_kata_attn,
    parallel_kata_attn_fwd,
    parallel_kata_attn_sum,
    parallel_kata_attn_sum_fwd_impl,
)
from kata.spd_kernels import chunk_kata_spd

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
        feature_map_eps: float = 1e-6,
        norm_q: bool = False,
        norm_k: bool = False,
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

        # norm_q/norm_k: RMSNorm(head_dim, fp32) on q/k before the score, applied the
        # SAME way self-attention applies qk-norm (fla Attention q_norm/k_norm). No psi.
        if norm_q:
            self.q_norm = RMSNorm(self.head_k_dim, dtype=torch.float32)
        if norm_k:
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
                self.rotary = RotaryEmbedding(dim=self.head_k_dim // spd_num_groups, base=rope_theta)
            else:
                # Full-head RoPE, interleaved=True: pairs (2i,2i+1) stay within a contiguous
                # group so each per-group dot is relative-position, but groups get DISJOINT
                # frequency bands (low->group0, high->groupM-1). interleaved=False would
                # straddle the group boundary and leak absolute position.
                self.rotary = RotaryEmbedding(dim=self.head_k_dim, base=rope_theta, interleaved=True)

        self.feature_map_name = feature_map
        self.spd_num_groups = spd_num_groups
        self.spd_use_kernel = spd_use_kernel
        self.spd_chunk_size = spd_chunk_size

        if spd_use_kernel:
            if feature_map != "spd_concat":
                raise ValueError(
                    'spd_use_kernel requires feature_map="spd_concat"'
                )
            if self.head_k_dim % spd_num_groups != 0:
                raise ValueError(
                    f"head_k_dim {self.head_k_dim} not divisible by "
                    f"spd_num_groups {spd_num_groups}",
                )
            E_per_group = self.head_k_dim // spd_num_groups
            if E_per_group < 16:
                raise ValueError(
                    f"E_per_group {E_per_group} < 16; lower spd_num_groups or "
                    f"raise head_k_dim",
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
                hidden_size=self.key_dim, kernel_size=conv_size,
                bias=conv_bias, activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim_per_group, kernel_size=conv_size,
                bias=conv_bias, activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim_per_group, kernel_size=conv_size,
                bias=conv_bias, activation="silu",
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

        self.norm_q = norm_q
        self.norm_k = norm_k

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
                x=self.q_proj(hidden_states), cache=conv_state_q,
                output_final_state=use_cache,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states), cache=conv_state_k,
                output_final_state=use_cache,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states), cache=conv_state_v,
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

        # norm_q/norm_k THEN RoPE on q,k (B,T,H,head_k_dim) before the SPD score --
        # exact transformer order (qk_norm at fla Attention L106, rotary at L124).
        if self.norm_q:
            q = self.q_norm(q)
        if self.norm_k:
            k = self.k_norm(k)
        if self.use_rope:
            seqlen_offset = 0
            if past_key_values is not None and self.layer_idx is not None:
                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = max(q.shape[1] + seqlen_offset, self.max_position_embeddings)
            if self.rope_group:
                # treat each E-dim group as its own head: (B,T,H,D) -> (B,T,H*M,E),
                # rotate with RotaryEmbedding(dim=E), reshape back.
                Bz, Tl, Hq = q.shape[0], q.shape[1], q.shape[2]
                Hk = k.shape[2]
                E = self.head_k_dim // self.spd_num_groups
                qg = q.reshape(Bz, Tl, Hq * self.spd_num_groups, E)
                kg = k.reshape(Bz, Tl, Hk * self.spd_num_groups, E)
                qg, kg = self.rotary(qg, kg, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen)
                q = qg.reshape(Bz, Tl, Hq, self.head_k_dim)
                k = kg.reshape(Bz, Tl, Hk, self.head_k_dim)
            else:
                q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen)

        # SPD kernel path: skip psi materialization, hand raw projections to
        # the on-the-fly Triton kernel. Only valid for chunk mode (no varlen,
        # no caching support yet — those use the pre-materialize fallback).
        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )
        use_kernel = (
            self.spd_use_kernel
            and self.feature_map_name == "spd_concat"
            and mode == "chunk"
            and not use_cache
            and recurrent_state is None
        )
        if use_kernel:
            o = chunk_kata_spd(
                q,
                k,
                v,
                chunk_size=self.spd_chunk_size,
                num_groups=self.spd_num_groups,
            )
            final_state = None
        elif self.feature_map_name in ("kata_quadratic", "kata_quadratic_sum"):
            # FlashAttention-style quadratic kata; psi never materialized.
            # `kata_quadratic`     -> concat-SPD score Sum_i (q_i.k_i)^2
            # `kata_quadratic_sum` -> sum-SPD score   Sum_{i,j} (q_i.k_j)^2
            #                         (matches paper KATA-SPD-4 math).
            is_sum = self.feature_map_name == "kata_quadratic_sum"
            grad_on = torch.is_grad_enabled() and any(
                t.requires_grad for t in (q, k, v)
            )
            if grad_on:
                if is_sum:
                    o = parallel_kata_attn_sum(
                        q, k, v, num_groups=self.spd_num_groups, scale=None,
                    )
                else:
                    o = parallel_kata_attn(
                        q, k, v, num_groups=self.spd_num_groups, scale=None,
                    )
            else:
                if is_sum:
                    o, _ = parallel_kata_attn_sum_fwd_impl(
                        q, k, v, self.spd_num_groups, None, 64, 64,
                    )
                else:
                    o, _ = parallel_kata_attn_fwd(
                        q, k, v, num_groups=self.spd_num_groups, scale=None,
                    )
            final_state = None
        else:
            # pre-materialize psi(q), psi(k) and route through FLA kernels
            q = self.feature_map(q)
            k = self.feature_map(k)
            if self.norm_q:
                q = q / (q.sum(-1, True) + 1e-4)
            if self.norm_k:
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
