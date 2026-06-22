"""KATA chunk attention. Wraps FLA's chunk_simple_gla + normalize_with_z_state.

Forward:
    Given raw q_hat, k_hat, v of shape [B, T, H, D] / [B, T, H, D] / [B, T, H, V],
    apply feature map psi to q_hat, k_hat -> q, k of shape [B, T, H, q] (q can
    differ from D for spd_concat). Run chunked linear attention with running
    denominator <q_t, sum_{s<=t} k_s>.

The chunked log(T/C) reduction comes from FLA's chunk_simple_gla; the
denominator comes from FLA's normalize_with_z_state. Both have correct backward
passes — autograd handles the chain rule through psi.

Note on Hopper Triton:
    fla #640 reports wrong gradients in chunk_bwd_dqkwg (gated delta path,
    Triton 3.4+). chunk_simple_gla used here is a *different* kernel
    (no q-side gating) and has not been reported to suffer from that bug. The
    LinAtt setup script still installs tilelang on Hopper as a safety net for
    other paths.
"""

import torch
from fla.ops.linear_attn import chunk_linear_attn
from kata.feature_maps import FeatureMap


def chunk_kata(
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    feature_map: FeatureMap,
    scale: float | None = None,
    initial_state: torch.Tensor | tuple | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | tuple | None]:
    """Apply psi to (q_hat, k_hat) and run normalized linear attention.

    Args mirror fla.ops.linear_attn.chunk_linear_attn except the feature_map.
    Output normalization (denominator <q_t, Z_t>) is always on for KATA.
    """
    q = feature_map(q_hat)
    k = feature_map(k_hat)
    return chunk_linear_attn(
        q=q,
        k=k,
        v=v,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        normalize=True,
        cu_seqlens=cu_seqlens,
    )
