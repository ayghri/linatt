"""Optimized chunked (GDN-like) KATA-SPD forward: bf16 tensor-core matmuls.

The previous chunked kernel ran fp32 tl.dot (no tensor cores) -> slow. This one
runs every matmul bf16-in / fp32-accumulate to use the tensor cores, which is the
dominant speed lever vs GatedDeltaNet.

Note on symmetric packing: psi(x_g)=vec(x_g⊗x_g) is symmetric, so in principle
only E(E+1)/2 entries are unique. But Triton block shapes must be powers of two,
and E is a power of two -> E^2 is already a power of two while E(E+1)/2 is not and
pads back up to E^2. So packing yields no Triton-level saving here; we keep the
dense E^2 feature, which tiles cleanly.

State S (E^2 x Dv) and z (E^2) are carried across chunks per (batch-head, group).
Validated vs kata.spd_attn.spd_parallel_ref.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_fast_fwd_kernel(
    q,
    k,
    v,
    num,
    den,
    s_scale,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    P: tl.constexpr,
    DV: tl.constexpr,
    C: tl.constexpr,
    NG: tl.constexpr,
    MODE: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    i_g = tl.program_id(1)
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    bos_ng = (i_bh * NG + i_g) * T
    NC = tl.cdiv(T, C)

    S = tl.zeros([P, DV], dtype=tl.float32)
    Z = tl.zeros([P], dtype=tl.float32)

    for i_c in range(NC):
        c0 = i_c * C
        o_row = c0 + tl.arange(0, C)
        row_ok = o_row < T
        # bf16 for tensor cores
        b_v = tl.load(
            tl.make_block_ptr(v + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
            boundary_check=(0, 1),
        ).to(tl.bfloat16)

        # dense outer-product feature (bf16): one group for CONCAT, summed for SUM
        psi_q = tl.zeros([C, P], dtype=tl.bfloat16)
        psi_k = tl.zeros([C, P], dtype=tl.bfloat16)
        for g in range(M):
            use = (g == i_g) if MODE == 0 else True
            if use:
                b_qg = (
                    tl.load(
                        tl.make_block_ptr(
                            q + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                b_kg = (
                    tl.load(
                        tl.make_block_ptr(
                            k + bos_qk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)
                        ),
                        boundary_check=(0, 1),
                    )
                    * s_scale
                )
                psi_q += tl.reshape(b_qg[:, :, None] * b_qg[:, None, :], (C, P)).to(
                    tl.bfloat16
                )
                psi_k += tl.reshape(b_kg[:, :, None] * b_kg[:, None, :], (C, P)).to(
                    tl.bfloat16
                )
        psi_k = tl.where(row_ok[:, None], psi_k, 0.0)

        # inter-chunk (carried state); all dots bf16-in / fp32-acc
        b_num = tl.dot(psi_q, S.to(tl.bfloat16))
        b_den = tl.sum(psi_q.to(tl.float32) * Z[None, :], axis=1)
        # intra-chunk causal
        A = tl.dot(psi_q, tl.trans(psi_k))
        A = tl.where((o_row[:, None] >= o_row[None, :]) & row_ok[None, :], A, 0.0)
        b_num += tl.dot(A.to(tl.bfloat16), b_v)
        b_den += tl.sum(A, axis=1)

        tl.store(
            tl.make_block_ptr(
                num + bos_ng * DV, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)
            ),
            b_num.to(num.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            tl.make_block_ptr(den + bos_ng, (T,), (1,), (c0,), (C,), (0,)),
            b_den.to(den.dtype.element_ty),
            boundary_check=(0,),
        )

        # update state (fp32 accumulate)
        b_vm = tl.where(row_ok[:, None], b_v, 0.0)
        S += tl.dot(tl.trans(psi_k), b_vm)
        Z += tl.sum(psi_k.to(tl.float32), axis=0)


def spd_attn_chunked_fast(q, k, v, M, mode, scale=None, C=64, eps=1e-6):
    B, H, T, D = q.shape
    DV = v.shape[-1]
    E = D // M
    P = E * E
    NG = M if mode == "concat" else 1
    if scale is None:
        scale = 1.0 / E
    q, k, v = (x.contiguous() for x in (q, k, v))
    num = torch.empty(B, H, NG, T, DV, device=q.device, dtype=torch.float32)
    den = torch.empty(B, H, NG, T, device=q.device, dtype=torch.float32)
    _chunk_fast_fwd_kernel[(B * H, NG)](
        q,
        k,
        v,
        num,
        den,
        scale**0.5,
        T,
        H=H,
        D=D,
        E=E,
        M=M,
        P=P,
        DV=DV,
        C=C,
        NG=NG,
        MODE=0 if mode == "concat" else 1,
        num_warps=4,
        num_stages=2,
    )
    num = num.sum(2)
    den = den.sum(2).clamp_min(eps)
    return (num / den[..., None]).to(q.dtype), den
