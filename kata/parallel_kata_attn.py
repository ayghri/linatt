"""Parallel (quadratic) KATA-SPD attention in Triton — FlashAttention-style.

Forks the structure of `fla.ops.attn.parallel.parallel_attn_fwd_kernel` but
replaces the softmax score path with the SPD-concat identity

    A[t, s] = Sum_{i=1..M} <q_i[t], k_i[s]> ** 2
            = <psi_concat(q_t), psi_concat(k_s)>

where psi_concat splits the d-dim head into M groups of E = d/M and
concatenates vec(g_i g_i^T). No psi is ever materialized in HBM — the M
per-group dot products are computed inside SRAM from the same Q, K we'd
load for vanilla attention.

Compared to softmax-FA:
- no online max tracking (kata scores are nonneg; no overflow stabilization)
- denominator is just Sum_s A[t, s] (no exp, no log-sum-exp)
- otherwise the tile-fused recipe is identical: load Q tile, iterate K/V
  blocks, accumulate normalized output

Forward-only for now; backward to follow once forward throughput is
verified on H100.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------
# Triton autotune configs. Tune (BT, BS, num_warps, num_stages) for both
# fwd and bwd kernels. Small configs first so they're tried before larger
# ones that OOR on smaller SRAM (e.g., RTX 3090's 100KB shared).
# ---------------------------------------------------------------------

_FWD_CONFIGS = [
    # 32x32
    triton.Config({"BT": 32,  "BS": 32}, num_warps=4, num_stages=1),
    triton.Config({"BT": 32,  "BS": 32}, num_warps=4, num_stages=2),
    triton.Config({"BT": 32,  "BS": 32}, num_warps=4, num_stages=3),
    # 32x64
    triton.Config({"BT": 32,  "BS": 64}, num_warps=4, num_stages=1),
    triton.Config({"BT": 32,  "BS": 64}, num_warps=4, num_stages=2),
    # 64x32
    triton.Config({"BT": 64,  "BS": 32}, num_warps=4, num_stages=1),
    triton.Config({"BT": 64,  "BS": 32}, num_warps=4, num_stages=2),
    # 64x64
    triton.Config({"BT": 64,  "BS": 64}, num_warps=4, num_stages=1),
    triton.Config({"BT": 64,  "BS": 64}, num_warps=4, num_stages=2),
    triton.Config({"BT": 64,  "BS": 64}, num_warps=4, num_stages=3),
    triton.Config({"BT": 64,  "BS": 64}, num_warps=8, num_stages=1),
    triton.Config({"BT": 64,  "BS": 64}, num_warps=8, num_stages=2),
    # 64x128
    triton.Config({"BT": 64,  "BS": 128}, num_warps=8, num_stages=1),
    triton.Config({"BT": 64,  "BS": 128}, num_warps=8, num_stages=2),
    # 128x64
    triton.Config({"BT": 128, "BS": 64}, num_warps=8, num_stages=1),
    triton.Config({"BT": 128, "BS": 64}, num_warps=8, num_stages=2),
    # 128x128
    triton.Config({"BT": 128, "BS": 128}, num_warps=8, num_stages=1),
    triton.Config({"BT": 128, "BS": 128}, num_warps=8, num_stages=2),
]

# Bwd kernels carry more live tiles per program (per-group dq/dk
# accumulators); SRAM-friendly configs first.
_BWD_CONFIGS = [
    # 16x16 (most conservative)
    triton.Config({"BT": 16, "BS": 16}, num_warps=4, num_stages=1),
    triton.Config({"BT": 32, "BS": 16}, num_warps=4, num_stages=1),
    triton.Config({"BT": 16, "BS": 32}, num_warps=4, num_stages=1),
    # 32x32
    triton.Config({"BT": 32, "BS": 32}, num_warps=4, num_stages=1),
    triton.Config({"BT": 32, "BS": 32}, num_warps=4, num_stages=2),
    triton.Config({"BT": 32, "BS": 32}, num_warps=8, num_stages=1),
    # 32x64 / 64x32
    triton.Config({"BT": 32, "BS": 64}, num_warps=4, num_stages=1),
    triton.Config({"BT": 64, "BS": 32}, num_warps=4, num_stages=1),
    # 64x64
    triton.Config({"BT": 64, "BS": 64}, num_warps=4, num_stages=1),
    triton.Config({"BT": 64, "BS": 64}, num_warps=4, num_stages=2),
    triton.Config({"BT": 64, "BS": 64}, num_warps=8, num_stages=1),
    triton.Config({"BT": 64, "BS": 64}, num_warps=8, num_stages=2),
    # 128x64 / 64x128 (may OOR on smaller SRAM; Triton drops)
    triton.Config({"BT": 128, "BS": 64}, num_warps=8, num_stages=1),
    triton.Config({"BT": 64, "BS": 128}, num_warps=8, num_stages=1),
    triton.Config({"BT": 128, "BS": 128}, num_warps=8, num_stages=1),
]

_FWD_KEY = ["T", "K_d", "V_d", "M"]
_BWD_KEY_M1 = ["T", "K_d", "V_d"]
_BWD_KEY_ME = ["T", "K_d", "V_d", "E"]


@triton.autotune(configs=_FWD_CONFIGS, key=_FWD_KEY)
@triton.jit
def parallel_kata_attn_fwd_kernel(
    q,                  # (B, T, HQ, K) bf16
    k,                  # (B, T, H, K)  bf16
    v,                  # (B, T, H, V)  bf16
    o,                  # (B, T, HQ, V) bf16
    den,                # (B, T, HQ)    fp32 — denominator Sum_s A[t,s], saved for bwd
    scale,              # softmax_scale (applied to qk *before* squaring)
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,    # = HQ // H (GQA grouping)
    K_d: tl.constexpr,  # head_k dim (= M * E)
    V_d: tl.constexpr,  # head_v dim
    BV: tl.constexpr,
    M: tl.constexpr,    # SPD num_groups
    E: tl.constexpr,    # = K_d / M
    BT: tl.constexpr,   # query-block rows  (autotuned)
    BS: tl.constexpr,   # key-block rows    (autotuned)
):
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos = i_b * T

    p_o = tl.make_block_ptr(
        o + (bos * HQ + i_hq) * V_d, (T, V_d), (HQ * V_d, 1),
        (i_t * BT, i_v * BV), (BT, BV), (1, 0),
    )
    p_den = tl.make_block_ptr(
        den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,),
    )

    # Running accumulators
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    o_q = i_t * BT + tl.arange(0, BT)

    # Phase 1: strictly earlier key blocks (no causal mask needed)
    for i_s in range(0, i_t * BT, BS):
        b_s = tl.zeros([BT, BS], dtype=tl.float32)
        for m_id in tl.static_range(M):
            p_q_g = tl.make_block_ptr(
                q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1),
                (i_t * BT, m_id * E), (BT, E), (1, 0),
            )
            p_k_g = tl.make_block_ptr(
                k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d),
                (m_id * E, i_s), (E, BS), (0, 1),
            )
            b_q_g = tl.load(p_q_g, boundary_check=(0, 1))
            b_k_g = tl.load(p_k_g, boundary_check=(0, 1))
            qk_g = tl.dot(b_q_g, b_k_g) * scale
            b_s += qk_g * qk_g

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_s = tl.where(m_k[None, :], b_s, 0.0)

        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d, (T, V_d), (H * V_d, 1),
            (i_s, i_v * BV), (BS, BV), (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_acc += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)

    # Phase 2: on-diagonal block (causal mask)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        b_s = tl.zeros([BT, BS], dtype=tl.float32)
        for m_id in tl.static_range(M):
            p_q_g = tl.make_block_ptr(
                q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1),
                (i_t * BT, m_id * E), (BT, E), (1, 0),
            )
            p_k_g = tl.make_block_ptr(
                k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d),
                (m_id * E, i_s), (E, BS), (0, 1),
            )
            b_q_g = tl.load(p_q_g, boundary_check=(0, 1))
            b_k_g = tl.load(p_k_g, boundary_check=(0, 1))
            qk_g = tl.dot(b_q_g, b_k_g) * scale
            b_s += qk_g * qk_g

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_s = tl.where(m_s, b_s, 0.0)

        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d, (T, V_d), (H * V_d, 1),
            (i_s, i_v * BV), (BS, BV), (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_acc += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)

    # Normalize
    b_acc_safe = tl.where(b_acc > 0, b_acc, 1.0)
    b_o = b_o / b_acc_safe[:, None]

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_den, b_acc.to(p_den.dtype.element_ty), boundary_check=(0,))


@triton.autotune(configs=_FWD_CONFIGS, key=_FWD_KEY)
@triton.jit
def parallel_kata_attn_sum_fwd_kernel(
    q, k, v, o, den,
    scale, T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BV: tl.constexpr,
    M: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    """Sum-SPD score: A[t,s] = Sum_{i,j} (scale * q_i[t] . k_j[s])^2.

    M^2 per-group dot products per (t,s) tile (vs the concat kernel's M).
    Implicit psi(q) = vec(Sum_i q_i q_i^T) of dim d_head^2 / M^2 (smaller
    state than concat). Matches the math used by paper KATA-SPD-4
    (psi_packed variant='four_rank' in kata/reference.py).
    """
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos = i_b * T

    p_o = tl.make_block_ptr(
        o + (bos * HQ + i_hq) * V_d, (T, V_d), (HQ * V_d, 1),
        (i_t * BT, i_v * BV), (BT, BV), (1, 0),
    )
    p_den = tl.make_block_ptr(
        den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,),
    )

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)
    o_q = i_t * BT + tl.arange(0, BT)

    # Phase 1: strictly earlier key blocks
    for i_s in range(0, i_t * BT, BS):
        b_s = tl.zeros([BT, BS], dtype=tl.float32)
        for i in tl.static_range(M):
            p_q_i = tl.make_block_ptr(
                q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1),
                (i_t * BT, i * E), (BT, E), (1, 0),
            )
            b_q_i = tl.load(p_q_i, boundary_check=(0, 1))
            for j in tl.static_range(M):
                p_k_j = tl.make_block_ptr(
                    k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d),
                    (j * E, i_s), (E, BS), (0, 1),
                )
                b_k_j = tl.load(p_k_j, boundary_check=(0, 1))
                qk_ij = tl.dot(b_q_i, b_k_j) * scale
                b_s += qk_ij * qk_ij

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_s = tl.where(m_k[None, :], b_s, 0.0)

        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d, (T, V_d), (H * V_d, 1),
            (i_s, i_v * BV), (BS, BV), (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_acc += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)

    # Phase 2: on-diagonal block (causal)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        b_s = tl.zeros([BT, BS], dtype=tl.float32)
        for i in tl.static_range(M):
            p_q_i = tl.make_block_ptr(
                q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1),
                (i_t * BT, i * E), (BT, E), (1, 0),
            )
            b_q_i = tl.load(p_q_i, boundary_check=(0, 1))
            for j in tl.static_range(M):
                p_k_j = tl.make_block_ptr(
                    k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d),
                    (j * E, i_s), (E, BS), (0, 1),
                )
                b_k_j = tl.load(p_k_j, boundary_check=(0, 1))
                qk_ij = tl.dot(b_q_i, b_k_j) * scale
                b_s += qk_ij * qk_ij

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_s = tl.where(m_s, b_s, 0.0)

        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d, (T, V_d), (H * V_d, 1),
            (i_s, i_v * BV), (BS, BV), (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_acc += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v)

    b_acc_safe = tl.where(b_acc > 0, b_acc, 1.0)
    b_o = b_o / b_acc_safe[:, None]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_den, b_acc.to(p_den.dtype.element_ty), boundary_check=(0,))


@triton.jit
def parallel_kata_attn_bwd_preprocess(
    o, do, delta,
    NEL: tl.constexpr,
    V: tl.constexpr,
):
    """delta[i] = <o[i,:], do[i,:]>  for each row i  (grid: total rows)."""
    i_n = tl.program_id(0)
    o_d = tl.arange(0, NEL)
    m_d = o_d < V
    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    tl.store(delta + i_n, tl.sum(b_o * b_do).to(delta.dtype.element_ty))




def parallel_kata_attn_bwd(
    q, k, v, o, den, do,
    num_groups, scale,
    BT=64, BS=64,
):
    """Backward for kata-quadratic attention.

    Currently a pure-pytorch reference (recomputes A in fp32, materializes the
    (T,T) score matrix). Correct but O(T²) memory in the score tensor — fine
    for T<=2048 at moderate B but not the eventual production path. A fused
    two-pass Triton bwd is planned (see fused_kernel_plan.md §4 Tier C).
    """
    B, T, HQ, K_d = q.shape
    H = k.shape[2]
    V_d = v.shape[-1]
    M = num_groups
    E = K_d // M

    # Promote to fp32 for the bwd math.
    qf = q.float()
    kf = k.float()
    vf = v.float()
    dof = do.float()
    Df = den.float().clamp_min(1e-12)

    # Expand K, V to HQ via GQA replication (assume G = HQ // H).
    G = HQ // H
    if G > 1:
        kf = kf.unsqueeze(3).expand(B, T, H, G, K_d).reshape(B, T, HQ, K_d).contiguous()
        vf = vf.unsqueeze(3).expand(B, T, H, G, V_d).reshape(B, T, HQ, V_d).contiguous()

    # Per-group dot products: qk[b, h, t, s, m] = scale * sum_e q[b,t,h,m*E+e] k[b,s,h,m*E+e]
    qg = qf.view(B, T, HQ, M, E)
    kg = kf.view(B, T, HQ, M, E)
    qk = torch.einsum("bthme,bshme->bhtsm", qg, kg) * scale  # (B, HQ, T, T, M)
    A = (qk * qk).sum(-1)  # (B, HQ, T, T)

    # Causal mask
    causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    A = A * causal.unsqueeze(0).unsqueeze(0)
    P = A / Df.permute(0, 2, 1).unsqueeze(-1)  # (B, HQ, T, T)

    # delta[b, h, t] = <o[t], do[t]>
    of = o.float()
    delta = (of * dof).sum(-1)  # (B, T, HQ)
    delta = delta.permute(0, 2, 1)  # (B, HQ, T)

    # dV[b, s, h, d] = Σ_t P[t,s] dO[t, d]
    do_p = dof.permute(0, 2, 1, 3)  # (B, HQ, T, V_d)
    dV = torch.einsum("bhts,bhtv->bhsv", P, do_p)  # (B, HQ, T, V_d)

    # dA[t,s] = (<dO[t], V[s]> - delta[t]) / D[t]
    v_p = vf.permute(0, 2, 1, 3)  # (B, HQ, T, V_d)
    dN_V = torch.einsum("bhtv,bhsv->bhts", do_p, v_p)
    Df_bht = Df.permute(0, 2, 1)  # (B, HQ, T)
    dA = (dN_V - delta.unsqueeze(-1)) / Df_bht.unsqueeze(-1)
    dA = dA * causal.unsqueeze(0).unsqueeze(0)

    # da[b, h, t, s, m] = 2 * qk * dA
    da = 2.0 * qk * dA.unsqueeze(-1)  # (B, HQ, T, T, M)

    # dq_i[t, e] = scale * Σ_s da[t,s,i] * k_i[s, e]
    dQ = scale * torch.einsum("bhtsm,bshme->bthme", da, kg)  # (B, T, HQ, M, E)
    dQ = dQ.reshape(B, T, HQ, K_d).contiguous()

    # dk_i[s, e] = scale * Σ_t da[t,s,i] * q_i[t, e]
    dK_full = scale * torch.einsum("bhtsm,bthme->bshme", da, qg)  # (B, T, HQ, M, E)
    dK_full = dK_full.reshape(B, T, HQ, K_d)

    if G > 1:
        # Sum dk, dv across query-head groups that share the same kv head.
        dK = dK_full.view(B, T, H, G, K_d).sum(3).contiguous()
        dV_p = dV.permute(0, 2, 1, 3)  # (B, T, HQ, V_d)
        dV = dV_p.view(B, T, H, G, V_d).sum(3).contiguous()
    else:
        dK = dK_full.contiguous()
        dV = dV.permute(0, 2, 1, 3).contiguous()

    return dQ.to(q.dtype), dK.to(k.dtype), dV.to(v.dtype)


class ParallelKataAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, num_groups, scale, BT, BS, use_triton_bwd):
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1] // num_groups)
        o, den = parallel_kata_attn_fwd_impl(q, k, v, num_groups, scale, BT, BS)
        ctx.save_for_backward(q, k, v, o, den)
        ctx.num_groups = num_groups
        ctx.scale = scale
        ctx.BT = BT
        ctx.BS = BS
        ctx.use_triton_bwd = use_triton_bwd
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, den = ctx.saved_tensors
        H = k.shape[2]
        HQ = q.shape[2]
        # Triton bwd supports M in {1, 2, 4}, MHA only (G=1).
        if (
            ctx.use_triton_bwd
            and ctx.num_groups in (1, 2, 4)
            and H == HQ
        ):
            dq, dk, dv = parallel_kata_attn_bwd_triton(
                q, k, v, o, den, do.contiguous(),
                ctx.num_groups, ctx.scale, ctx.BT, ctx.BS,
            )
        else:
            dq, dk, dv = parallel_kata_attn_bwd(
                q, k, v, o, den, do.contiguous(),
                ctx.num_groups, ctx.scale, ctx.BT, ctx.BS,
            )
        return dq, dk, dv, None, None, None, None, None


def parallel_kata_attn(q, k, v, num_groups=4, scale=None, BT=64, BS=64,
                       use_triton_bwd=True):
    """Autograd-wrapped quadratic kata-attention (fwd + bwd).

    Triton bwd is currently available only for M=4 MHA (no GQA). Falls back
    to the pytorch reference otherwise.
    """
    return ParallelKataAttnFunction.apply(q, k, v, num_groups, scale, BT, BS,
                                          use_triton_bwd)


def parallel_kata_attn_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_groups: int,
    scale: float,
    BT: int,
    BS: int,
):
    """Quadratic kata-attention forward (causal). Returns (o, den).

    o   : (B, T, HQ, V_d) same dtype as v
    den : (B, T, HQ) fp32 — Sum_{s<=t} <psi(q_t), psi(k_s)>, saved for bwd
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    B, T, HQ, K_d = q.shape
    _, _, H, _ = k.shape
    V_d = v.shape[-1]
    assert k.shape[:2] == (B, T) and v.shape[:2] == (B, T)
    assert HQ % H == 0, "GQA requires HQ multiple of H"
    G = HQ // H
    assert K_d % num_groups == 0, f"K_d={K_d} not divisible by M={num_groups}"
    M = num_groups
    E = K_d // M
    if scale is None:
        # We square the score: keep scale = 1 / sqrt(E) so per-group
        # standardized inner products stay O(1) before squaring.
        scale = 1.0 / math.sqrt(E)

    BV = V_d
    NV = 1

    dev = q.device
    o = torch.empty_like(v) if v.shape[2] == HQ else torch.empty(
        B, T, HQ, V_d, device=dev, dtype=v.dtype,
    )
    den = torch.empty(B, T, HQ, device=dev, dtype=torch.float32)

    # BT is autotuned: pass grid as a meta-aware lambda.
    grid = lambda meta: (NV, triton.cdiv(T, meta["BT"]), B * HQ)
    parallel_kata_attn_fwd_kernel[grid](
        q, k, v, o, den,
        scale, T,
        H=H, HQ=HQ, G=G,
        K_d=K_d, V_d=V_d,
        BV=BV,
        M=M, E=E,
    )
    return o, den


def parallel_kata_attn_fwd(q, k, v, num_groups=4, scale=None, BT=64, BS=64):
    """No-autograd wrapper around the forward kernel (used by inference/bench)."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1] // num_groups)
    return parallel_kata_attn_fwd_impl(q, k, v, num_groups, scale, BT, BS)


# =====================================================================
# Working M=4-unrolled backward kernels.
# Avoids the unsupported slice-add idiom by keeping 4 separate per-group
# accumulators (b_dq0..b_dq3 and b_dk0..b_dk3) and emitting 4 stores at
# the end. M is hardcoded to 4. For other M values, write a separate kernel
# or use the pytorch reference path in `parallel_kata_attn_bwd_torch`.
# =====================================================================


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_bwd_kernel_dq_M4(
    q, k, v,
    den, delta, do, dq,
    scale, T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    E: tl.constexpr,
):
    """dQ for kata-quadratic attention, M=4. Grid: (1, NT, B*HQ)."""
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos = i_b * T

    p_do = tl.make_block_ptr(
        do + (bos * HQ + i_hq) * V_d, (T, V_d), (HQ * V_d, 1),
        (i_t * BT, 0), (BT, BV), (1, 0),
    )
    p_den = tl.make_block_ptr(
        den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,),
    )
    p_delta = tl.make_block_ptr(
        delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,),
    )
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_D = tl.load(p_den, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_D_safe = tl.where(b_D > 0, b_D, 1.0)

    # M=4 per-group dq accumulators
    b_dq0 = tl.zeros([BT, E], dtype=tl.float32)
    b_dq1 = tl.zeros([BT, E], dtype=tl.float32)
    b_dq2 = tl.zeros([BT, E], dtype=tl.float32)
    b_dq3 = tl.zeros([BT, E], dtype=tl.float32)

    o_q = i_t * BT + tl.arange(0, BT)

    for i_s in range(0, min((i_t + 1) * BT, T), BS):
        # Per-group Q (BT, E) and K (E, BS).
        p_q0 = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 0 * E), (BT, E), (1, 0))
        p_q1 = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 1 * E), (BT, E), (1, 0))
        p_q2 = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 2 * E), (BT, E), (1, 0))
        p_q3 = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 3 * E), (BT, E), (1, 0))
        p_k0 = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (0 * E, i_s), (E, BS), (0, 1))
        p_k1 = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (1 * E, i_s), (E, BS), (0, 1))
        p_k2 = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (2 * E, i_s), (E, BS), (0, 1))
        p_k3 = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (3 * E, i_s), (E, BS), (0, 1))

        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_q2 = tl.load(p_q2, boundary_check=(0, 1))
        b_q3 = tl.load(p_q3, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_k3 = tl.load(p_k3, boundary_check=(0, 1))

        qk0 = tl.dot(b_q0, b_k0) * scale
        qk1 = tl.dot(b_q1, b_k1) * scale
        qk2 = tl.dot(b_q2, b_k2) * scale
        qk3 = tl.dot(b_q3, b_k3) * scale

        # b_A not needed in dq kernel (we use dA directly).

        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d, (V_d, T), (1, H * V_d),
            (0, i_s), (BV, BS), (0, 1),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)  # (BT, BS)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_dA = tl.where(causal, b_dA, 0.0)

        da0 = 2.0 * qk0 * b_dA
        da1 = 2.0 * qk1 * b_dA
        da2 = 2.0 * qk2 * b_dA
        da3 = 2.0 * qk3 * b_dA

        # b_k_g is (E, BS); transpose to (BS, E) for the dot
        b_dq0 += tl.dot(da0.to(b_k0.dtype), tl.trans(b_k0)) * scale
        b_dq1 += tl.dot(da1.to(b_k1.dtype), tl.trans(b_k1)) * scale
        b_dq2 += tl.dot(da2.to(b_k2.dtype), tl.trans(b_k2)) * scale
        b_dq3 += tl.dot(da3.to(b_k3.dtype), tl.trans(b_k3)) * scale

    # Store the four (BT, E) slices of dQ.
    p_dq0 = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 0 * E), (BT, E), (1, 0))
    p_dq1 = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 1 * E), (BT, E), (1, 0))
    p_dq2 = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 2 * E), (BT, E), (1, 0))
    p_dq3 = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 3 * E), (BT, E), (1, 0))
    tl.store(p_dq0, b_dq0.to(p_dq0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq1, b_dq1.to(p_dq1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq3, b_dq3.to(p_dq3.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_bwd_kernel_dkdv_M4(
    q, k, v,
    den, delta, do, dk, dv,
    scale, T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,   # assumed = 1 (MHA); for GQA, post-sum in python
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    E: tl.constexpr,
):
    """dK + dV for kata-quadratic attention, M=4. Grid: (1, NS, B*HQ).

    Note: G=1 assumption. For GQA (G>1) we'd accumulate one dK/dV per
    query head and sum after — the wrapper handles that path in pytorch.
    """
    i_s = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos = i_b * T

    b_dk0 = tl.zeros([BS, E], dtype=tl.float32)
    b_dk1 = tl.zeros([BS, E], dtype=tl.float32)
    b_dk2 = tl.zeros([BS, E], dtype=tl.float32)
    b_dk3 = tl.zeros([BS, E], dtype=tl.float32)
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    o_k = i_s * BS + tl.arange(0, BS)

    t_start = (i_s * BS) // BT
    NT = tl.cdiv(T, BT)
    for i_t in range(t_start, NT):
        p_do = tl.make_block_ptr(
            do + (bos * HQ + i_hq) * V_d, (T, V_d), (HQ * V_d, 1),
            (i_t * BT, 0), (BT, BV), (1, 0),
        )
        p_den = tl.make_block_ptr(den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_D = tl.load(p_den, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_D_safe = tl.where(b_D > 0, b_D, 1.0)

        # Per-group Q (BT, E) and K (E, BS).
        p_q0 = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 0 * E), (BT, E), (1, 0))
        p_q1 = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 1 * E), (BT, E), (1, 0))
        p_q2 = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 2 * E), (BT, E), (1, 0))
        p_q3 = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 3 * E), (BT, E), (1, 0))
        p_k0 = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (0 * E, i_s * BS), (E, BS), (0, 1))
        p_k1 = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (1 * E, i_s * BS), (E, BS), (0, 1))
        p_k2 = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (2 * E, i_s * BS), (E, BS), (0, 1))
        p_k3 = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (3 * E, i_s * BS), (E, BS), (0, 1))

        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_q2 = tl.load(p_q2, boundary_check=(0, 1))
        b_q3 = tl.load(p_q3, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_k3 = tl.load(p_k3, boundary_check=(0, 1))

        qk0 = tl.dot(b_q0, b_k0) * scale
        qk1 = tl.dot(b_q1, b_k1) * scale
        qk2 = tl.dot(b_q2, b_k2) * scale
        qk3 = tl.dot(b_q3, b_k3) * scale
        b_A = qk0 * qk0 + qk1 * qk1 + qk2 * qk2 + qk3 * qk3

        o_q = i_t * BT + tl.arange(0, BT)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]

        # dV first.  P[t,s] = A[t,s] / D[t]
        b_P = tl.where(causal, b_A / b_D_safe[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

        # dA = (<dO, V> - delta) / D
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V_d, (V_d, T), (1, H * V_d),
            (0, i_s * BS), (BV, BS), (0, 1),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        b_dA = tl.where(causal, b_dA, 0.0)

        # da per group, then dk_g += scale * sum_t da[t,s] * q_g[t,:]  ⇒  scale * da^T @ q_g (BS, E)
        da0 = 2.0 * qk0 * b_dA
        da1 = 2.0 * qk1 * b_dA
        da2 = 2.0 * qk2 * b_dA
        da3 = 2.0 * qk3 * b_dA

        b_dk0 += tl.dot(tl.trans(da0.to(b_q0.dtype)), b_q0) * scale
        b_dk1 += tl.dot(tl.trans(da1.to(b_q1.dtype)), b_q1) * scale
        b_dk2 += tl.dot(tl.trans(da2.to(b_q2.dtype)), b_q2) * scale
        b_dk3 += tl.dot(tl.trans(da3.to(b_q3.dtype)), b_q3) * scale

    # Stores
    p_dk0 = tl.make_block_ptr(dk + (bos * H + i_h) * K_d, (T, K_d), (H * K_d, 1), (i_s * BS, 0 * E), (BS, E), (1, 0))
    p_dk1 = tl.make_block_ptr(dk + (bos * H + i_h) * K_d, (T, K_d), (H * K_d, 1), (i_s * BS, 1 * E), (BS, E), (1, 0))
    p_dk2 = tl.make_block_ptr(dk + (bos * H + i_h) * K_d, (T, K_d), (H * K_d, 1), (i_s * BS, 2 * E), (BS, E), (1, 0))
    p_dk3 = tl.make_block_ptr(dk + (bos * H + i_h) * K_d, (T, K_d), (H * K_d, 1), (i_s * BS, 3 * E), (BS, E), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V_d, (T, V_d), (H * V_d, 1), (i_s * BS, 0), (BS, BV), (1, 0))
    tl.store(p_dk0, b_dk0.to(p_dk0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk1, b_dk1.to(p_dk1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk3, b_dk3.to(p_dk3.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def parallel_kata_attn_bwd_triton_M4(
    q, k, v, o, den, do,
    scale,
    BT=64, BS=64,
):
    """Triton bwd, M=4 hardcoded, MHA only (G=1).

    Returns (dq, dk, dv) in input dtypes.
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    do = do.contiguous()
    B, T, HQ, K_d = q.shape
    H = k.shape[2]
    V_d = v.shape[-1]
    assert H == HQ, "M=4 Triton bwd assumes MHA (G=1); use pytorch ref for GQA"
    M = 4
    E = K_d // M
    assert K_d == M * E, f"K_d={K_d} not divisible by M={M}"

    dev = q.device
    # Preprocess: delta[t] = <o[t], do[t]>
    delta = torch.empty(B * T * HQ, device=dev, dtype=torch.float32)
    NEL = triton.next_power_of_2(V_d)
    parallel_kata_attn_bwd_preprocess[(B * T * HQ,)](
        o.contiguous().view(-1, V_d), do.view(-1, V_d), delta,
        NEL=NEL, V=V_d,
        num_warps=4, num_stages=1,
    )
    delta = delta.view(B, T, HQ)

    dq = torch.empty(B, T, HQ, K_d, device=dev, dtype=q.dtype)
    dk = torch.empty(B, T, H, K_d, device=dev, dtype=k.dtype)
    dv = torch.empty(B, T, H, V_d, device=dev, dtype=v.dtype)

    parallel_kata_attn_bwd_kernel_dq_M4[lambda meta: (1, triton.cdiv(T, meta["BT"]), B * HQ)](
        q, k, v, den, delta, do, dq,
        scale, T,
        H=H, HQ=HQ, G=1, K_d=K_d, V_d=V_d,
        BV=V_d, E=E,
    )
    parallel_kata_attn_bwd_kernel_dkdv_M4[lambda meta: (1, triton.cdiv(T, meta["BS"]), B * HQ)](
        q, k, v, den, delta, do, dk, dv,
        scale, T,
        H=H, HQ=HQ, G=1, K_d=K_d, V_d=V_d,
        BV=V_d, E=E,
    )
    return dq, dk, dv


# =====================================================================
# M=1 backward kernels (degenerate: K_d == E, no grouping)
# =====================================================================


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_M1)
@triton.jit
def parallel_kata_attn_bwd_kernel_dq_M1(
    q, k, v,
    den, delta, do, dq,
    scale, T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
):
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos = i_b * T

    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V_d, (T, V_d), (HQ * V_d, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_den = tl.make_block_ptr(den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_D = tl.load(p_den, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_D_safe = tl.where(b_D > 0, b_D, 1.0)

    b_dq = tl.zeros([BT, K_d], dtype=tl.float32)
    o_q = i_t * BT + tl.arange(0, BT)

    for i_s in range(0, min((i_t + 1) * BT, T), BS):
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 0), (BT, K_d), (1, 0))
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (0, i_s), (K_d, BS), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        qk = tl.dot(b_q, b_k) * scale

        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V_d, (V_d, T), (1, H * V_d), (0, i_s), (BV, BS), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_dA = tl.where(causal, b_dA, 0.0)

        da = 2.0 * qk * b_dA
        b_dq += tl.dot(da.to(b_k.dtype), tl.trans(b_k)) * scale

    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 0), (BT, K_d), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_M1)
@triton.jit
def parallel_kata_attn_bwd_kernel_dkdv_M1(
    q, k, v,
    den, delta, do, dk, dv,
    scale, T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
):
    i_s = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    bos = i_b * T

    b_dk = tl.zeros([BS, K_d], dtype=tl.float32)
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)
    o_k = i_s * BS + tl.arange(0, BS)
    t_start = (i_s * BS) // BT
    NT = tl.cdiv(T, BT)

    for i_t in range(t_start, NT):
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V_d, (T, V_d), (HQ * V_d, 1), (i_t * BT, 0), (BT, BV), (1, 0))
        p_den = tl.make_block_ptr(den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_D = tl.load(p_den, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_D_safe = tl.where(b_D > 0, b_D, 1.0)

        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 0), (BT, K_d), (1, 0))
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (0, i_s * BS), (K_d, BS), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        qk = tl.dot(b_q, b_k) * scale
        b_A = qk * qk

        o_q = i_t * BT + tl.arange(0, BT)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_P = tl.where(causal, b_A / b_D_safe[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V_d, (V_d, T), (1, H * V_d), (0, i_s * BS), (BV, BS), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        b_dA = tl.where(causal, b_dA, 0.0)
        da = 2.0 * qk * b_dA
        b_dk += tl.dot(tl.trans(da.to(b_q.dtype)), b_q) * scale

    p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K_d, (T, K_d), (H * K_d, 1), (i_s * BS, 0), (BS, K_d), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V_d, (T, V_d), (H * V_d, 1), (i_s * BS, 0), (BS, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# =====================================================================
# M=2 backward kernels (two per-group accumulators)
# =====================================================================


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_bwd_kernel_dq_M2(
    q, k, v,
    den, delta, do, dq,
    scale, T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    E: tl.constexpr,
):
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V_d, (T, V_d), (HQ * V_d, 1), (i_t * BT, 0), (BT, BV), (1, 0))
    p_den = tl.make_block_ptr(den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_D = tl.load(p_den, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_D_safe = tl.where(b_D > 0, b_D, 1.0)

    b_dq0 = tl.zeros([BT, E], dtype=tl.float32)
    b_dq1 = tl.zeros([BT, E], dtype=tl.float32)
    o_q = i_t * BT + tl.arange(0, BT)

    for i_s in range(0, min((i_t + 1) * BT, T), BS):
        p_q0 = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 0 * E), (BT, E), (1, 0))
        p_q1 = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 1 * E), (BT, E), (1, 0))
        p_k0 = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (0 * E, i_s), (E, BS), (0, 1))
        p_k1 = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (1 * E, i_s), (E, BS), (0, 1))
        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        qk0 = tl.dot(b_q0, b_k0) * scale
        qk1 = tl.dot(b_q1, b_k1) * scale

        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V_d, (V_d, T), (1, H * V_d), (0, i_s), (BV, BS), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_dA = tl.where(causal, b_dA, 0.0)
        da0 = 2.0 * qk0 * b_dA
        da1 = 2.0 * qk1 * b_dA
        b_dq0 += tl.dot(da0.to(b_k0.dtype), tl.trans(b_k0)) * scale
        b_dq1 += tl.dot(da1.to(b_k1.dtype), tl.trans(b_k1)) * scale

    p_dq0 = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 0 * E), (BT, E), (1, 0))
    p_dq1 = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 1 * E), (BT, E), (1, 0))
    tl.store(p_dq0, b_dq0.to(p_dq0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq1, b_dq1.to(p_dq1.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_bwd_kernel_dkdv_M2(
    q, k, v,
    den, delta, do, dk, dv,
    scale, T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    E: tl.constexpr,
):
    i_s = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    b_dk0 = tl.zeros([BS, E], dtype=tl.float32)
    b_dk1 = tl.zeros([BS, E], dtype=tl.float32)
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)
    o_k = i_s * BS + tl.arange(0, BS)
    t_start = (i_s * BS) // BT
    NT = tl.cdiv(T, BT)

    for i_t in range(t_start, NT):
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V_d, (T, V_d), (HQ * V_d, 1), (i_t * BT, 0), (BT, BV), (1, 0))
        p_den = tl.make_block_ptr(den + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_D = tl.load(p_den, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_D_safe = tl.where(b_D > 0, b_D, 1.0)

        p_q0 = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 0 * E), (BT, E), (1, 0))
        p_q1 = tl.make_block_ptr(q + (bos * HQ + i_hq) * K_d, (T, K_d), (HQ * K_d, 1), (i_t * BT, 1 * E), (BT, E), (1, 0))
        p_k0 = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (0 * E, i_s * BS), (E, BS), (0, 1))
        p_k1 = tl.make_block_ptr(k + (bos * H + i_h) * K_d, (K_d, T), (1, H * K_d), (1 * E, i_s * BS), (E, BS), (0, 1))
        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        qk0 = tl.dot(b_q0, b_k0) * scale
        qk1 = tl.dot(b_q1, b_k1) * scale
        b_A = qk0 * qk0 + qk1 * qk1

        o_q = i_t * BT + tl.arange(0, BT)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_P = tl.where(causal, b_A / b_D_safe[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V_d, (V_d, T), (1, H * V_d), (0, i_s * BS), (BV, BS), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        b_dA = tl.where(causal, b_dA, 0.0)
        da0 = 2.0 * qk0 * b_dA
        da1 = 2.0 * qk1 * b_dA
        b_dk0 += tl.dot(tl.trans(da0.to(b_q0.dtype)), b_q0) * scale
        b_dk1 += tl.dot(tl.trans(da1.to(b_q1.dtype)), b_q1) * scale

    p_dk0 = tl.make_block_ptr(dk + (bos * H + i_h) * K_d, (T, K_d), (H * K_d, 1), (i_s * BS, 0 * E), (BS, E), (1, 0))
    p_dk1 = tl.make_block_ptr(dk + (bos * H + i_h) * K_d, (T, K_d), (H * K_d, 1), (i_s * BS, 1 * E), (BS, E), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V_d, (T, V_d), (H * V_d, 1), (i_s * BS, 0), (BS, BV), (1, 0))
    tl.store(p_dk0, b_dk0.to(p_dk0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk1, b_dk1.to(p_dk1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def parallel_kata_attn_bwd_triton(q, k, v, o, den, do, num_groups, scale, BT=64, BS=64):
    """Triton bwd dispatcher for M in {1, 2, 4}, MHA only (G=1).

    Falls back to the pytorch reference for other M or for GQA.
    """
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); do = do.contiguous()
    B, T, HQ, K_d = q.shape
    H = k.shape[2]
    V_d = v.shape[-1]
    M = num_groups
    if H != HQ or M not in (1, 2, 4):
        return parallel_kata_attn_bwd(q, k, v, o, den, do, num_groups, scale, BT, BS)

    E = K_d // M
    dev = q.device
    delta = torch.empty(B * T * HQ, device=dev, dtype=torch.float32)
    NEL = triton.next_power_of_2(V_d)
    parallel_kata_attn_bwd_preprocess[(B * T * HQ,)](
        o.contiguous().view(-1, V_d), do.view(-1, V_d), delta,
        NEL=NEL, V=V_d, num_warps=4, num_stages=1,
    )
    delta = delta.view(B, T, HQ)

    dq = torch.empty(B, T, HQ, K_d, device=dev, dtype=q.dtype)
    dk = torch.empty(B, T, H, K_d, device=dev, dtype=k.dtype)
    dv = torch.empty(B, T, H, V_d, device=dev, dtype=v.dtype)

    NT = triton.cdiv(T, BT)
    NS = triton.cdiv(T, BS)
    if M == 1:
        parallel_kata_attn_bwd_kernel_dq_M1[lambda meta: (1, triton.cdiv(T, meta["BT"]), B * HQ)](
            q, k, v, den, delta, do, dq, scale, T,
            H=H, HQ=HQ, G=1, K_d=K_d, V_d=V_d,
            BV=V_d,
        )
        parallel_kata_attn_bwd_kernel_dkdv_M1[lambda meta: (1, triton.cdiv(T, meta["BS"]), B * HQ)](
            q, k, v, den, delta, do, dk, dv, scale, T,
            H=H, HQ=HQ, G=1, K_d=K_d, V_d=V_d,
            BV=V_d,
        )
    elif M == 2:
        parallel_kata_attn_bwd_kernel_dq_M2[lambda meta: (1, triton.cdiv(T, meta["BT"]), B * HQ)](
            q, k, v, den, delta, do, dq, scale, T,
            H=H, HQ=HQ, G=1, K_d=K_d, V_d=V_d,
            BV=V_d, E=E,
        )
        parallel_kata_attn_bwd_kernel_dkdv_M2[lambda meta: (1, triton.cdiv(T, meta["BS"]), B * HQ)](
            q, k, v, den, delta, do, dk, dv, scale, T,
            H=H, HQ=HQ, G=1, K_d=K_d, V_d=V_d,
            BV=V_d, E=E,
        )
    else:  # M == 4
        parallel_kata_attn_bwd_kernel_dq_M4[lambda meta: (1, triton.cdiv(T, meta["BT"]), B * HQ)](
            q, k, v, den, delta, do, dq, scale, T,
            H=H, HQ=HQ, G=1, K_d=K_d, V_d=V_d,
            BV=V_d, E=E,
        )
        parallel_kata_attn_bwd_kernel_dkdv_M4[lambda meta: (1, triton.cdiv(T, meta["BS"]), B * HQ)](
            q, k, v, den, delta, do, dk, dv, scale, T,
            H=H, HQ=HQ, G=1, K_d=K_d, V_d=V_d,
            BV=V_d, E=E,
        )
    return dq, dk, dv


# =====================================================================
# Sum-SPD variant: same FlashAttention skeleton, score = Sum_{i,j} (q_i·k_j)²
# (M^2 inner products per (t,s) pair, vs concat's M).
# Matches the math used by paper KATA-SPD-4 (psi_packed variant="four_rank").
# =====================================================================


def parallel_kata_attn_sum_fwd_impl(q, k, v, num_groups, scale, BT, BS):
    """Sum-SPD forward. See parallel_kata_attn_sum_fwd_kernel."""
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    B, T, HQ, K_d = q.shape
    H = k.shape[2]
    V_d = v.shape[-1]
    assert HQ % H == 0
    G = HQ // H
    assert K_d % num_groups == 0
    M = num_groups
    E = K_d // M
    if scale is None:
        scale = 1.0 / math.sqrt(E)
    BV = V_d
    dev = q.device
    o = torch.empty(B, T, HQ, V_d, device=dev, dtype=v.dtype)
    den = torch.empty(B, T, HQ, device=dev, dtype=torch.float32)
    grid = lambda meta: (1, triton.cdiv(T, meta["BT"]), B * HQ)
    parallel_kata_attn_sum_fwd_kernel[grid](
        q, k, v, o, den,
        scale, T,
        H=H, HQ=HQ, G=G,
        K_d=K_d, V_d=V_d,
        BV=BV,
        M=M, E=E,
    )
    return o, den


def parallel_kata_attn_bwd_sum_torch(q, k, v, o, den, do, num_groups, scale):
    """Pytorch-reference bwd for sum-SPD attention.

    Materializes the (B, HQ, T, T, M, M) score tensor in fp32 → memory-heavy
    at T=2048 with M=4 (B*HQ*T²*M² fp32 = B*HQ*16M*64 bytes per layer).
    Use only for MQAR-scale correctness and small-T training; production
    bwd needs a Triton kernel (TODO).
    """
    B, T, HQ, K_d = q.shape
    H = k.shape[2]
    V_d = v.shape[-1]
    G = HQ // H
    M = num_groups
    E = K_d // M

    qf = q.float(); kf = k.float(); vf = v.float(); dof = do.float()
    Df = den.float().clamp_min(1e-12)
    if G > 1:
        kf = kf.unsqueeze(3).expand(B, T, H, G, K_d).reshape(B, T, HQ, K_d).contiguous()
        vf = vf.unsqueeze(3).expand(B, T, H, G, V_d).reshape(B, T, HQ, V_d).contiguous()

    qg = qf.view(B, T, HQ, M, E)
    kg = kf.view(B, T, HQ, M, E)
    # qk[b,h,t,s,i,j] = scale * sum_e q_i[b,t,h,e] * k_j[b,s,h,e]
    qk = torch.einsum("bthie,bshje->bhtsij", qg, kg) * scale  # (B,HQ,T,T,M,M)
    A = (qk * qk).sum(dim=(-2, -1))  # (B, HQ, T, T)

    causal = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    A = A * causal.unsqueeze(0).unsqueeze(0)
    P = A / Df.permute(0, 2, 1).unsqueeze(-1)

    delta = (o.float() * dof).sum(-1)  # (B, T, HQ)
    delta = delta.permute(0, 2, 1)

    do_p = dof.permute(0, 2, 1, 3)  # (B, HQ, T, V)
    v_p = vf.permute(0, 2, 1, 3)
    dV = torch.einsum("bhts,bhtv->bhsv", P, do_p)  # (B, HQ, T, V)

    dN_V = torch.einsum("bhtv,bhsv->bhts", do_p, v_p)
    Df_bht = Df.permute(0, 2, 1)
    dA = (dN_V - delta.unsqueeze(-1)) / Df_bht.unsqueeze(-1)
    dA = dA * causal.unsqueeze(0).unsqueeze(0)

    # da[b,h,t,s,i,j] = 2 * qk_ij * dA[t,s]
    da = 2.0 * qk * dA.unsqueeze(-1).unsqueeze(-1)  # (B,HQ,T,T,M,M)

    # dq_i[t,e] = scale * sum_s sum_j da[t,s,i,j] * k_j[s,e]
    dQ = scale * torch.einsum("bhtsij,bshje->bthie", da, kg)  # (B,T,HQ,M,E)
    dQ = dQ.reshape(B, T, HQ, K_d).contiguous()

    # dk_j[s,e] = scale * sum_t sum_i da[t,s,i,j] * q_i[t,e]
    dK_full = scale * torch.einsum("bhtsij,bthie->bshje", da, qg)
    dK_full = dK_full.reshape(B, T, HQ, K_d)

    if G > 1:
        dK = dK_full.view(B, T, H, G, K_d).sum(3).contiguous()
        dV_t = dV.permute(0, 2, 1, 3).view(B, T, H, G, V_d).sum(3).contiguous()
    else:
        dK = dK_full.contiguous()
        dV_t = dV.permute(0, 2, 1, 3).contiguous()

    return dQ.to(q.dtype), dK.to(k.dtype), dV_t.to(v.dtype)


class ParallelKataAttnSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, num_groups, scale, BT, BS, use_triton_bwd):
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1] // num_groups)
        o, den = parallel_kata_attn_sum_fwd_impl(q, k, v, num_groups, scale, BT, BS)
        # Only Q, K, V, O, den persisted (no (T,T,M,M) score in HBM).
        # The bwd recomputes A and dA inside each Triton chunk in SRAM.
        ctx.save_for_backward(q, k, v, o, den)
        ctx.num_groups = num_groups
        ctx.scale = scale
        ctx.use_triton_bwd = use_triton_bwd
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, den = ctx.saved_tensors
        H = k.shape[2]
        HQ = q.shape[2]
        if ctx.use_triton_bwd and H == HQ:
            if ctx.num_groups == 4:
                dq, dk, dv = parallel_kata_attn_sum_bwd_triton_M4(
                    q, k, v, o, den, do.contiguous(), ctx.scale,
                )
            elif ctx.num_groups == 2:
                dq, dk, dv = parallel_kata_attn_sum_bwd_triton_M2(
                    q, k, v, o, den, do.contiguous(), ctx.scale,
                )
            elif ctx.num_groups == 1:
                # M=1: sum == concat == single squared dot.
                dq, dk, dv = parallel_kata_attn_bwd_triton(
                    q, k, v, o, den, do.contiguous(),
                    1, ctx.scale, 32, 32,
                )
            else:
                dq, dk, dv = parallel_kata_attn_bwd_sum_torch(
                    q, k, v, o, den, do.contiguous(),
                    ctx.num_groups, ctx.scale,
                )
        else:
            dq, dk, dv = parallel_kata_attn_bwd_sum_torch(
                q, k, v, o, den, do.contiguous(),
                ctx.num_groups, ctx.scale,
            )
        return dq, dk, dv, None, None, None, None, None


def parallel_kata_attn_sum(q, k, v, num_groups=4, scale=None, BT=64, BS=64,
                          use_triton_bwd=True):
    """Autograd-wrapped sum-SPD quadratic kata-attention (fwd + bwd, Triton).

    fwd: `parallel_kata_attn_sum_fwd_kernel` — M² inner products per (t,s)
         tile, no psi in HBM.
    bwd (M=4): `parallel_kata_attn_sum_bwd_kernel_dq_M4` +
               `parallel_kata_attn_sum_bwd_kernel_dkdv_M4` — only Q, K, V, O,
               den read from HBM; A and dA recomputed in SRAM per chunk.
    bwd (M=1): falls back to the (mathematically identical) concat M=1 kernel.
    bwd (M=2, M=3, ... or GQA): pytorch reference (slow, OOMs at production B).
    """
    return ParallelKataAttnSumFunction.apply(q, k, v, num_groups, scale, BT, BS,
                                             use_triton_bwd)


# =====================================================================
# Triton backward kernels for sum-SPD attention, M=4 (production case).
# 16 (i, j) pairs per (t, s) tile: 4 dq accumulators + 4 dk accumulators
# + 1 dv accumulator. Hardcoded M=4 unroll.
# =====================================================================


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_sum_bwd_kernel_dq_M4(
    q, k, v, den, delta, do, dq,
    scale, T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BV: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    """Sum-SPD dQ kernel, M=4. Grid: (1, NT, B*HQ)."""
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    p_do = tl.make_block_ptr(do + (bos*HQ+i_hq)*V_d, (T, V_d), (HQ*V_d, 1), (i_t*BT, 0), (BT, BV), (1, 0))
    p_den = tl.make_block_ptr(den + bos*HQ+i_hq, (T,), (HQ,), (i_t*BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos*HQ+i_hq, (T,), (HQ,), (i_t*BT,), (BT,), (0,))
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_D = tl.load(p_den, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_D_safe = tl.where(b_D > 0, b_D, 1.0)

    b_dq0 = tl.zeros([BT, E], dtype=tl.float32)
    b_dq1 = tl.zeros([BT, E], dtype=tl.float32)
    b_dq2 = tl.zeros([BT, E], dtype=tl.float32)
    b_dq3 = tl.zeros([BT, E], dtype=tl.float32)
    o_q = i_t*BT + tl.arange(0, BT)

    for i_s in range(0, min((i_t+1)*BT, T), BS):
        # Load Q groups (BT, E) and K groups (E, BS) for this chunk.
        p_q0 = tl.make_block_ptr(q + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 0*E), (BT, E), (1, 0))
        p_q1 = tl.make_block_ptr(q + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 1*E), (BT, E), (1, 0))
        p_q2 = tl.make_block_ptr(q + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 2*E), (BT, E), (1, 0))
        p_q3 = tl.make_block_ptr(q + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 3*E), (BT, E), (1, 0))
        p_k0 = tl.make_block_ptr(k + (bos*H+i_h)*K_d, (K_d, T), (1, H*K_d), (0*E, i_s), (E, BS), (0, 1))
        p_k1 = tl.make_block_ptr(k + (bos*H+i_h)*K_d, (K_d, T), (1, H*K_d), (1*E, i_s), (E, BS), (0, 1))
        p_k2 = tl.make_block_ptr(k + (bos*H+i_h)*K_d, (K_d, T), (1, H*K_d), (2*E, i_s), (E, BS), (0, 1))
        p_k3 = tl.make_block_ptr(k + (bos*H+i_h)*K_d, (K_d, T), (1, H*K_d), (3*E, i_s), (E, BS), (0, 1))
        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_q2 = tl.load(p_q2, boundary_check=(0, 1))
        b_q3 = tl.load(p_q3, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_k3 = tl.load(p_k3, boundary_check=(0, 1))

        # dA from accumulated A is the same regardless of (i, j) — load V, dO once.
        p_v = tl.make_block_ptr(v + (bos*H+i_h)*V_d, (V_d, T), (1, H*V_d), (0, i_s), (BV, BS), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_dA = tl.where(causal, b_dA, 0.0)

        # Per (i, j): dq[i] += scale * (2 * qk_ij * dA) @ k_j^T
        # i=0
        qk = tl.dot(b_q0, b_k0) * scale; da = 2.0*qk*b_dA; b_dq0 += tl.dot(da.to(b_k0.dtype), tl.trans(b_k0)) * scale
        qk = tl.dot(b_q0, b_k1) * scale; da = 2.0*qk*b_dA; b_dq0 += tl.dot(da.to(b_k1.dtype), tl.trans(b_k1)) * scale
        qk = tl.dot(b_q0, b_k2) * scale; da = 2.0*qk*b_dA; b_dq0 += tl.dot(da.to(b_k2.dtype), tl.trans(b_k2)) * scale
        qk = tl.dot(b_q0, b_k3) * scale; da = 2.0*qk*b_dA; b_dq0 += tl.dot(da.to(b_k3.dtype), tl.trans(b_k3)) * scale
        # i=1
        qk = tl.dot(b_q1, b_k0) * scale; da = 2.0*qk*b_dA; b_dq1 += tl.dot(da.to(b_k0.dtype), tl.trans(b_k0)) * scale
        qk = tl.dot(b_q1, b_k1) * scale; da = 2.0*qk*b_dA; b_dq1 += tl.dot(da.to(b_k1.dtype), tl.trans(b_k1)) * scale
        qk = tl.dot(b_q1, b_k2) * scale; da = 2.0*qk*b_dA; b_dq1 += tl.dot(da.to(b_k2.dtype), tl.trans(b_k2)) * scale
        qk = tl.dot(b_q1, b_k3) * scale; da = 2.0*qk*b_dA; b_dq1 += tl.dot(da.to(b_k3.dtype), tl.trans(b_k3)) * scale
        # i=2
        qk = tl.dot(b_q2, b_k0) * scale; da = 2.0*qk*b_dA; b_dq2 += tl.dot(da.to(b_k0.dtype), tl.trans(b_k0)) * scale
        qk = tl.dot(b_q2, b_k1) * scale; da = 2.0*qk*b_dA; b_dq2 += tl.dot(da.to(b_k1.dtype), tl.trans(b_k1)) * scale
        qk = tl.dot(b_q2, b_k2) * scale; da = 2.0*qk*b_dA; b_dq2 += tl.dot(da.to(b_k2.dtype), tl.trans(b_k2)) * scale
        qk = tl.dot(b_q2, b_k3) * scale; da = 2.0*qk*b_dA; b_dq2 += tl.dot(da.to(b_k3.dtype), tl.trans(b_k3)) * scale
        # i=3
        qk = tl.dot(b_q3, b_k0) * scale; da = 2.0*qk*b_dA; b_dq3 += tl.dot(da.to(b_k0.dtype), tl.trans(b_k0)) * scale
        qk = tl.dot(b_q3, b_k1) * scale; da = 2.0*qk*b_dA; b_dq3 += tl.dot(da.to(b_k1.dtype), tl.trans(b_k1)) * scale
        qk = tl.dot(b_q3, b_k2) * scale; da = 2.0*qk*b_dA; b_dq3 += tl.dot(da.to(b_k2.dtype), tl.trans(b_k2)) * scale
        qk = tl.dot(b_q3, b_k3) * scale; da = 2.0*qk*b_dA; b_dq3 += tl.dot(da.to(b_k3.dtype), tl.trans(b_k3)) * scale

    p_dq0 = tl.make_block_ptr(dq + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 0*E), (BT, E), (1, 0))
    p_dq1 = tl.make_block_ptr(dq + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 1*E), (BT, E), (1, 0))
    p_dq2 = tl.make_block_ptr(dq + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 2*E), (BT, E), (1, 0))
    p_dq3 = tl.make_block_ptr(dq + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 3*E), (BT, E), (1, 0))
    tl.store(p_dq0, b_dq0.to(p_dq0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq1, b_dq1.to(p_dq1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq3, b_dq3.to(p_dq3.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_sum_bwd_kernel_dkdv_M4(
    q, k, v, den, delta, do, dk, dv,
    scale, T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K_d: tl.constexpr,
    V_d: tl.constexpr,
    BV: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    """Sum-SPD dK+dV kernel, M=4. Grid: (1, NS, B*HQ)."""
    i_s = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    b_dk0 = tl.zeros([BS, E], dtype=tl.float32)
    b_dk1 = tl.zeros([BS, E], dtype=tl.float32)
    b_dk2 = tl.zeros([BS, E], dtype=tl.float32)
    b_dk3 = tl.zeros([BS, E], dtype=tl.float32)
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)
    o_k = i_s*BS + tl.arange(0, BS)
    t_start = (i_s*BS) // BT
    NT = tl.cdiv(T, BT)

    for i_t in range(t_start, NT):
        p_do = tl.make_block_ptr(do + (bos*HQ+i_hq)*V_d, (T, V_d), (HQ*V_d, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_den = tl.make_block_ptr(den + bos*HQ+i_hq, (T,), (HQ,), (i_t*BT,), (BT,), (0,))
        p_delta = tl.make_block_ptr(delta + bos*HQ+i_hq, (T,), (HQ,), (i_t*BT,), (BT,), (0,))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_D = tl.load(p_den, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_D_safe = tl.where(b_D > 0, b_D, 1.0)

        p_q0 = tl.make_block_ptr(q + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 0*E), (BT, E), (1, 0))
        p_q1 = tl.make_block_ptr(q + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 1*E), (BT, E), (1, 0))
        p_q2 = tl.make_block_ptr(q + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 2*E), (BT, E), (1, 0))
        p_q3 = tl.make_block_ptr(q + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 3*E), (BT, E), (1, 0))
        p_k0 = tl.make_block_ptr(k + (bos*H+i_h)*K_d, (K_d, T), (1, H*K_d), (0*E, i_s*BS), (E, BS), (0, 1))
        p_k1 = tl.make_block_ptr(k + (bos*H+i_h)*K_d, (K_d, T), (1, H*K_d), (1*E, i_s*BS), (E, BS), (0, 1))
        p_k2 = tl.make_block_ptr(k + (bos*H+i_h)*K_d, (K_d, T), (1, H*K_d), (2*E, i_s*BS), (E, BS), (0, 1))
        p_k3 = tl.make_block_ptr(k + (bos*H+i_h)*K_d, (K_d, T), (1, H*K_d), (3*E, i_s*BS), (E, BS), (0, 1))
        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_q2 = tl.load(p_q2, boundary_check=(0, 1))
        b_q3 = tl.load(p_q3, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_k3 = tl.load(p_k3, boundary_check=(0, 1))

        # A and dv
        qk_00 = tl.dot(b_q0, b_k0) * scale; qk_01 = tl.dot(b_q0, b_k1) * scale
        qk_02 = tl.dot(b_q0, b_k2) * scale; qk_03 = tl.dot(b_q0, b_k3) * scale
        qk_10 = tl.dot(b_q1, b_k0) * scale; qk_11 = tl.dot(b_q1, b_k1) * scale
        qk_12 = tl.dot(b_q1, b_k2) * scale; qk_13 = tl.dot(b_q1, b_k3) * scale
        qk_20 = tl.dot(b_q2, b_k0) * scale; qk_21 = tl.dot(b_q2, b_k1) * scale
        qk_22 = tl.dot(b_q2, b_k2) * scale; qk_23 = tl.dot(b_q2, b_k3) * scale
        qk_30 = tl.dot(b_q3, b_k0) * scale; qk_31 = tl.dot(b_q3, b_k1) * scale
        qk_32 = tl.dot(b_q3, b_k2) * scale; qk_33 = tl.dot(b_q3, b_k3) * scale
        b_A = (qk_00*qk_00 + qk_01*qk_01 + qk_02*qk_02 + qk_03*qk_03
             + qk_10*qk_10 + qk_11*qk_11 + qk_12*qk_12 + qk_13*qk_13
             + qk_20*qk_20 + qk_21*qk_21 + qk_22*qk_22 + qk_23*qk_23
             + qk_30*qk_30 + qk_31*qk_31 + qk_32*qk_32 + qk_33*qk_33)

        o_q = i_t*BT + tl.arange(0, BT)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_P = tl.where(causal, b_A / b_D_safe[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

        p_v = tl.make_block_ptr(v + (bos*H+i_h)*V_d, (V_d, T), (1, H*V_d), (0, i_s*BS), (BV, BS), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        b_dA = tl.where(causal, b_dA, 0.0)

        # dk[j] += scale * sum_i (2 * qk_ij * dA)^T @ q_i
        # j=0: i=0..3
        da = 2.0*qk_00*b_dA; b_dk0 += tl.dot(tl.trans(da.to(b_q0.dtype)), b_q0) * scale
        da = 2.0*qk_10*b_dA; b_dk0 += tl.dot(tl.trans(da.to(b_q1.dtype)), b_q1) * scale
        da = 2.0*qk_20*b_dA; b_dk0 += tl.dot(tl.trans(da.to(b_q2.dtype)), b_q2) * scale
        da = 2.0*qk_30*b_dA; b_dk0 += tl.dot(tl.trans(da.to(b_q3.dtype)), b_q3) * scale
        # j=1
        da = 2.0*qk_01*b_dA; b_dk1 += tl.dot(tl.trans(da.to(b_q0.dtype)), b_q0) * scale
        da = 2.0*qk_11*b_dA; b_dk1 += tl.dot(tl.trans(da.to(b_q1.dtype)), b_q1) * scale
        da = 2.0*qk_21*b_dA; b_dk1 += tl.dot(tl.trans(da.to(b_q2.dtype)), b_q2) * scale
        da = 2.0*qk_31*b_dA; b_dk1 += tl.dot(tl.trans(da.to(b_q3.dtype)), b_q3) * scale
        # j=2
        da = 2.0*qk_02*b_dA; b_dk2 += tl.dot(tl.trans(da.to(b_q0.dtype)), b_q0) * scale
        da = 2.0*qk_12*b_dA; b_dk2 += tl.dot(tl.trans(da.to(b_q1.dtype)), b_q1) * scale
        da = 2.0*qk_22*b_dA; b_dk2 += tl.dot(tl.trans(da.to(b_q2.dtype)), b_q2) * scale
        da = 2.0*qk_32*b_dA; b_dk2 += tl.dot(tl.trans(da.to(b_q3.dtype)), b_q3) * scale
        # j=3
        da = 2.0*qk_03*b_dA; b_dk3 += tl.dot(tl.trans(da.to(b_q0.dtype)), b_q0) * scale
        da = 2.0*qk_13*b_dA; b_dk3 += tl.dot(tl.trans(da.to(b_q1.dtype)), b_q1) * scale
        da = 2.0*qk_23*b_dA; b_dk3 += tl.dot(tl.trans(da.to(b_q2.dtype)), b_q2) * scale
        da = 2.0*qk_33*b_dA; b_dk3 += tl.dot(tl.trans(da.to(b_q3.dtype)), b_q3) * scale

    p_dk0 = tl.make_block_ptr(dk + (bos*H+i_h)*K_d, (T, K_d), (H*K_d, 1), (i_s*BS, 0*E), (BS, E), (1, 0))
    p_dk1 = tl.make_block_ptr(dk + (bos*H+i_h)*K_d, (T, K_d), (H*K_d, 1), (i_s*BS, 1*E), (BS, E), (1, 0))
    p_dk2 = tl.make_block_ptr(dk + (bos*H+i_h)*K_d, (T, K_d), (H*K_d, 1), (i_s*BS, 2*E), (BS, E), (1, 0))
    p_dk3 = tl.make_block_ptr(dk + (bos*H+i_h)*K_d, (T, K_d), (H*K_d, 1), (i_s*BS, 3*E), (BS, E), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos*H+i_h)*V_d, (T, V_d), (H*V_d, 1), (i_s*BS, 0), (BS, BV), (1, 0))
    tl.store(p_dk0, b_dk0.to(p_dk0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk1, b_dk1.to(p_dk1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk3, b_dk3.to(p_dk3.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def parallel_kata_attn_sum_bwd_triton_M4(q, k, v, o, den, do, scale):
    """Triton sum-SPD bwd, M=4, MHA only."""
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); do = do.contiguous()
    B, T, HQ, K_d = q.shape
    H = k.shape[2]
    V_d = v.shape[-1]
    assert H == HQ, "sum-SPD M=4 Triton bwd assumes MHA"
    M = 4
    E = K_d // M
    dev = q.device

    delta = torch.empty(B*T*HQ, device=dev, dtype=torch.float32)
    NEL = triton.next_power_of_2(V_d)
    parallel_kata_attn_bwd_preprocess[(B*T*HQ,)](
        o.contiguous().view(-1, V_d), do.view(-1, V_d), delta,
        NEL=NEL, V=V_d, num_warps=4, num_stages=1,
    )
    delta = delta.view(B, T, HQ)

    dq = torch.empty(B, T, HQ, K_d, device=dev, dtype=q.dtype)
    dk = torch.empty(B, T, H, K_d, device=dev, dtype=k.dtype)
    dv = torch.empty(B, T, H, V_d, device=dev, dtype=v.dtype)

    parallel_kata_attn_sum_bwd_kernel_dq_M4[lambda meta: (1, triton.cdiv(T, meta["BT"]), B*HQ)](
        q, k, v, den, delta, do, dq, scale, T,
        H=H, HQ=HQ, G=1, K_d=K_d, V_d=V_d, BV=V_d, E=E,
    )
    parallel_kata_attn_sum_bwd_kernel_dkdv_M4[lambda meta: (1, triton.cdiv(T, meta["BS"]), B*HQ)](
        q, k, v, den, delta, do, dk, dv, scale, T,
        H=H, HQ=HQ, G=1, K_d=K_d, V_d=V_d, BV=V_d, E=E,
    )
    return dq, dk, dv


# =====================================================================
# Triton backward kernels for sum-SPD attention, M=2.
# 4 (i, j) pairs per (t, s) tile.
# =====================================================================


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_sum_bwd_kernel_dq_M2(
    q, k, v, den, delta, do, dq,
    scale, T,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K_d: tl.constexpr, V_d: tl.constexpr, BV: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr,
):
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    p_do = tl.make_block_ptr(do + (bos*HQ+i_hq)*V_d, (T, V_d), (HQ*V_d, 1), (i_t*BT, 0), (BT, BV), (1, 0))
    p_den = tl.make_block_ptr(den + bos*HQ+i_hq, (T,), (HQ,), (i_t*BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos*HQ+i_hq, (T,), (HQ,), (i_t*BT,), (BT,), (0,))
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_D = tl.load(p_den, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))
    b_D_safe = tl.where(b_D > 0, b_D, 1.0)

    b_dq0 = tl.zeros([BT, E], dtype=tl.float32)
    b_dq1 = tl.zeros([BT, E], dtype=tl.float32)
    o_q = i_t*BT + tl.arange(0, BT)

    for i_s in range(0, min((i_t+1)*BT, T), BS):
        p_q0 = tl.make_block_ptr(q + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 0*E), (BT, E), (1, 0))
        p_q1 = tl.make_block_ptr(q + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 1*E), (BT, E), (1, 0))
        p_k0 = tl.make_block_ptr(k + (bos*H+i_h)*K_d, (K_d, T), (1, H*K_d), (0*E, i_s), (E, BS), (0, 1))
        p_k1 = tl.make_block_ptr(k + (bos*H+i_h)*K_d, (K_d, T), (1, H*K_d), (1*E, i_s), (E, BS), (0, 1))
        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))

        p_v = tl.make_block_ptr(v + (bos*H+i_h)*V_d, (V_d, T), (1, H*V_d), (0, i_s), (BV, BS), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_dA = tl.where(causal, b_dA, 0.0)

        # i=0, j=0..1 → b_dq0
        qk = tl.dot(b_q0, b_k0) * scale; da = 2.0*qk*b_dA; b_dq0 += tl.dot(da.to(b_k0.dtype), tl.trans(b_k0)) * scale
        qk = tl.dot(b_q0, b_k1) * scale; da = 2.0*qk*b_dA; b_dq0 += tl.dot(da.to(b_k1.dtype), tl.trans(b_k1)) * scale
        # i=1
        qk = tl.dot(b_q1, b_k0) * scale; da = 2.0*qk*b_dA; b_dq1 += tl.dot(da.to(b_k0.dtype), tl.trans(b_k0)) * scale
        qk = tl.dot(b_q1, b_k1) * scale; da = 2.0*qk*b_dA; b_dq1 += tl.dot(da.to(b_k1.dtype), tl.trans(b_k1)) * scale

    p_dq0 = tl.make_block_ptr(dq + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 0*E), (BT, E), (1, 0))
    p_dq1 = tl.make_block_ptr(dq + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 1*E), (BT, E), (1, 0))
    tl.store(p_dq0, b_dq0.to(p_dq0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq1, b_dq1.to(p_dq1.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY_ME)
@triton.jit
def parallel_kata_attn_sum_bwd_kernel_dkdv_M2(
    q, k, v, den, delta, do, dk, dv,
    scale, T,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K_d: tl.constexpr, V_d: tl.constexpr, BV: tl.constexpr,
    E: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr,
):
    i_s = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    b_dk0 = tl.zeros([BS, E], dtype=tl.float32)
    b_dk1 = tl.zeros([BS, E], dtype=tl.float32)
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)
    o_k = i_s*BS + tl.arange(0, BS)
    t_start = (i_s*BS) // BT
    NT = tl.cdiv(T, BT)

    for i_t in range(t_start, NT):
        p_do = tl.make_block_ptr(do + (bos*HQ+i_hq)*V_d, (T, V_d), (HQ*V_d, 1), (i_t*BT, 0), (BT, BV), (1, 0))
        p_den = tl.make_block_ptr(den + bos*HQ+i_hq, (T,), (HQ,), (i_t*BT,), (BT,), (0,))
        p_delta = tl.make_block_ptr(delta + bos*HQ+i_hq, (T,), (HQ,), (i_t*BT,), (BT,), (0,))
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
        b_D = tl.load(p_den, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_D_safe = tl.where(b_D > 0, b_D, 1.0)

        p_q0 = tl.make_block_ptr(q + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 0*E), (BT, E), (1, 0))
        p_q1 = tl.make_block_ptr(q + (bos*HQ+i_hq)*K_d, (T, K_d), (HQ*K_d, 1), (i_t*BT, 1*E), (BT, E), (1, 0))
        p_k0 = tl.make_block_ptr(k + (bos*H+i_h)*K_d, (K_d, T), (1, H*K_d), (0*E, i_s*BS), (E, BS), (0, 1))
        p_k1 = tl.make_block_ptr(k + (bos*H+i_h)*K_d, (K_d, T), (1, H*K_d), (1*E, i_s*BS), (E, BS), (0, 1))
        b_q0 = tl.load(p_q0, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1))
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))

        qk_00 = tl.dot(b_q0, b_k0) * scale
        qk_01 = tl.dot(b_q0, b_k1) * scale
        qk_10 = tl.dot(b_q1, b_k0) * scale
        qk_11 = tl.dot(b_q1, b_k1) * scale
        b_A = qk_00*qk_00 + qk_01*qk_01 + qk_10*qk_10 + qk_11*qk_11

        o_q = i_t*BT + tl.arange(0, BT)
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        b_P = tl.where(causal, b_A / b_D_safe[:, None], 0.0)
        b_dv += tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)

        p_v = tl.make_block_ptr(v + (bos*H+i_h)*V_d, (V_d, T), (1, H*V_d), (0, i_s*BS), (BV, BS), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_dN_V = tl.dot(b_do, b_v)
        b_dA = (b_dN_V - b_delta[:, None]) / b_D_safe[:, None]
        b_dA = tl.where(causal, b_dA, 0.0)

        # j=0: sum over i=0,1 of (2 * qk_i0 * dA)^T @ q_i  → b_dk0
        da = 2.0*qk_00*b_dA; b_dk0 += tl.dot(tl.trans(da.to(b_q0.dtype)), b_q0) * scale
        da = 2.0*qk_10*b_dA; b_dk0 += tl.dot(tl.trans(da.to(b_q1.dtype)), b_q1) * scale
        # j=1
        da = 2.0*qk_01*b_dA; b_dk1 += tl.dot(tl.trans(da.to(b_q0.dtype)), b_q0) * scale
        da = 2.0*qk_11*b_dA; b_dk1 += tl.dot(tl.trans(da.to(b_q1.dtype)), b_q1) * scale

    p_dk0 = tl.make_block_ptr(dk + (bos*H+i_h)*K_d, (T, K_d), (H*K_d, 1), (i_s*BS, 0*E), (BS, E), (1, 0))
    p_dk1 = tl.make_block_ptr(dk + (bos*H+i_h)*K_d, (T, K_d), (H*K_d, 1), (i_s*BS, 1*E), (BS, E), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos*H+i_h)*V_d, (T, V_d), (H*V_d, 1), (i_s*BS, 0), (BS, BV), (1, 0))
    tl.store(p_dk0, b_dk0.to(p_dk0.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk1, b_dk1.to(p_dk1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


def parallel_kata_attn_sum_bwd_triton_M2(q, k, v, o, den, do, scale):
    """Triton sum-SPD bwd, M=2."""
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous(); do = do.contiguous()
    B, T, HQ, K_d = q.shape
    H = k.shape[2]; V_d = v.shape[-1]
    assert H == HQ
    M = 2; E = K_d // M
    dev = q.device

    delta = torch.empty(B*T*HQ, device=dev, dtype=torch.float32)
    NEL = triton.next_power_of_2(V_d)
    parallel_kata_attn_bwd_preprocess[(B*T*HQ,)](
        o.contiguous().view(-1, V_d), do.view(-1, V_d), delta,
        NEL=NEL, V=V_d, num_warps=4, num_stages=1,
    )
    delta = delta.view(B, T, HQ)
    dq = torch.empty(B, T, HQ, K_d, device=dev, dtype=q.dtype)
    dk = torch.empty(B, T, H, K_d, device=dev, dtype=k.dtype)
    dv = torch.empty(B, T, H, V_d, device=dev, dtype=v.dtype)

    parallel_kata_attn_sum_bwd_kernel_dq_M2[lambda meta: (1, triton.cdiv(T, meta["BT"]), B*HQ)](
        q, k, v, den, delta, do, dq, scale, T,
        H=H, HQ=HQ, G=1, K_d=K_d, V_d=V_d, BV=V_d, E=E,
    )
    parallel_kata_attn_sum_bwd_kernel_dkdv_M2[lambda meta: (1, triton.cdiv(T, meta["BS"]), B*HQ)](
        q, k, v, den, delta, do, dk, dv, scale, T,
        H=H, HQ=HQ, G=1, K_d=K_d, V_d=V_d, BV=V_d, E=E,
    )
    return dq, dk, dv
