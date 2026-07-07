"""Triton chunked (GDN-like, state-carrying) forward for KATA-SPD attention.

Linear-in-T. Grid is (batch-head, n_state): each program owns ONE 2D state
S (E^2 x Dv) + z (E^2), walks chunks left->right, and writes a per-state partial
(num_g, den_g). The quadratic feature psi(x_g)=vec(x_g⊗x_g) (dim E^2) is
materialized only within the current chunk, so the state stays a 2D tile and each
chunk is ordinary linear-attention math:

  inter_num = psi(Q_c) @ S      inter_den = psi(Q_c) @ z
  intra: A = psi(Q_c) @ psi(K_c)^T (causal) -> A@V, rowsum
  num_g = inter_num + intra_num     den_g = inter_den + intra_den
  S += psi(K_c)^T @ V               z += sum_s psi(K_c)[s]

CONCAT: n_state = M, program g uses query/key group g only.
SUM:    n_state = 1, the program forms the summed feature over all M groups.
The host sums the partials over states and divides. Validated vs spd_chunked_ref.
Forward only here (bwd is the follow-up).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _spd_chunk_fwd_kernel(
    q, k, v, num, den,
    s_scale, T,
    H: tl.constexpr, D: tl.constexpr, E: tl.constexpr, M: tl.constexpr,
    P: tl.constexpr, DV: tl.constexpr, C: tl.constexpr,
    NG: tl.constexpr, MODE: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    i_g = tl.program_id(1)              # which state (concat: group; sum: 0)
    bos_qk = i_bh * T * D
    bos_v = i_bh * T * DV
    bos_ng = (i_bh * NG + i_g) * T      # partial-output base for this (bh, state)
    NC = tl.cdiv(T, C)

    S = tl.zeros([P, DV], dtype=tl.float32)
    Z = tl.zeros([P], dtype=tl.float32)

    for i_c in range(NC):
        c0 = i_c * C
        o_row = c0 + tl.arange(0, C)
        row_ok = o_row < T
        b_v = tl.load(tl.make_block_ptr(v + bos_v, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
                      boundary_check=(0, 1))

        # build psi_q, psi_k (C, E^2): one group for CONCAT, summed for SUM
        psi_q = tl.zeros([C, P], dtype=tl.float32)
        psi_k = tl.zeros([C, P], dtype=tl.float32)
        for g in range(M):
            use = (g == i_g) if MODE == 0 else True
            if use:
                b_qg = tl.load(tl.make_block_ptr(q + bos_qk, (T, D), (D, 1),
                                                 (c0, g * E), (C, E), (1, 0)),
                               boundary_check=(0, 1)) * s_scale
                b_kg = tl.load(tl.make_block_ptr(k + bos_qk, (T, D), (D, 1),
                                                 (c0, g * E), (C, E), (1, 0)),
                               boundary_check=(0, 1)) * s_scale
                psi_q += tl.reshape(b_qg[:, :, None] * b_qg[:, None, :], (C, P))
                psi_k += tl.reshape(b_kg[:, :, None] * b_kg[:, None, :], (C, P))
        psi_k = tl.where(row_ok[:, None], psi_k, 0.0)

        # inter-chunk (carried state)
        b_num = tl.dot(psi_q, S)
        b_den = tl.sum(psi_q * Z[None, :], axis=1)
        # intra-chunk (within this chunk, causal)
        A = tl.dot(psi_q, tl.trans(psi_k))
        A = tl.where((o_row[:, None] >= o_row[None, :]) & row_ok[None, :], A, 0.0)
        b_num += tl.dot(A.to(b_v.dtype), b_v)
        b_den += tl.sum(A, axis=1)

        tl.store(tl.make_block_ptr(num + bos_ng * DV, (T, DV), (DV, 1), (c0, 0), (C, DV), (1, 0)),
                 b_num.to(num.dtype.element_ty), boundary_check=(0, 1))
        tl.store(tl.make_block_ptr(den + bos_ng, (T,), (1,), (c0,), (C,), (0,)),
                 b_den.to(den.dtype.element_ty), boundary_check=(0,))

        # update state (psi_k accumulates in fp32; keep V fp32 for the matmul)
        b_vm = tl.where(row_ok[:, None], b_v, 0.0).to(tl.float32)
        S += tl.dot(tl.trans(psi_k), b_vm)
        Z += tl.sum(psi_k, axis=0)


def spd_attn_chunked_fwd(q, k, v, M, mode, scale=None, C=32, eps=1e-6):
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
    _spd_chunk_fwd_kernel[(B * H, NG)](
        q, k, v, num, den, scale ** 0.5, T,
        H=H, D=D, E=E, M=M, P=P, DV=DV, C=C, NG=NG,
        MODE=0 if mode == "concat" else 1,
        num_warps=4, num_stages=1,
    )
    num = num.sum(2)                                   # (B,H,T,DV)
    den = den.sum(2).clamp_min(eps)                    # (B,H,T)
    o = (num / den[..., None]).to(q.dtype)
    return o, den
