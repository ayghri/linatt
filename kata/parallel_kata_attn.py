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

import math
import os

import torch
import triton


# Triton kernels (fwd/bwd, decay-aware) live in kata/kernels/. Normally this file
# is imported as `kata.parallel_kata_attn` and the package import works. But it is
# also loaded STANDALONE by file path (zoology's KataQuadraticMixer does this to
# avoid a two-`kata`-package name clash), where `kata` resolves to a different
# package — so fall back to loading kernels/ as a synthetic package.
def _load_kernels():
    g = globals()
    try:
        import kata.kernels.parallel_kata as _pk
        import kata.kernels.parallel_kata_sum as _ps
    except ModuleNotFoundError:
        import sys
        import importlib
        import importlib.util
        kdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
        pkg = "_linatt_kata_kernels"
        if pkg not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                pkg, os.path.join(kdir, "__init__.py"),
                submodule_search_locations=[kdir])
            mod = importlib.util.module_from_spec(spec)
            sys.modules[pkg] = mod
            spec.loader.exec_module(mod)
        _pk = importlib.import_module(f"{pkg}.parallel_kata")
        _ps = importlib.import_module(f"{pkg}.parallel_kata_sum")
    for _m in (_pk, _ps):
        g.update({k: v for k, v in vars(_m).items() if not k.startswith("_")})


_load_kernels()


def parallel_kata_attn_bwd(
    q,
    k,
    v,
    o,
    den,
    do,
    num_groups,
    scale,
    BT=64,
    BS=64,
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
        if ctx.use_triton_bwd and ctx.num_groups in (1, 2, 4) and H == HQ:
            dq, dk, dv = parallel_kata_attn_bwd_triton(
                q,
                k,
                v,
                o,
                den,
                do.contiguous(),
                ctx.num_groups,
                ctx.scale,
                ctx.BT,
                ctx.BS,
            )
        else:
            dq, dk, dv = parallel_kata_attn_bwd(
                q,
                k,
                v,
                o,
                den,
                do.contiguous(),
                ctx.num_groups,
                ctx.scale,
                ctx.BT,
                ctx.BS,
            )
        return dq, dk, dv, None, None, None, None, None


def parallel_kata_attn(
    q, k, v, num_groups=4, scale=None, BT=64, BS=64, use_triton_bwd=True
):
    """Autograd-wrapped quadratic kata-attention (fwd + bwd).

    Triton bwd is currently available only for M=4 MHA (no GQA). Falls back
    to the pytorch reference otherwise.
    """
    return ParallelKataAttnFunction.apply(
        q, k, v, num_groups, scale, BT, BS, use_triton_bwd
    )


def parallel_kata_attn_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_groups: int,
    scale: float,
    BT: int,
    BS: int,
    c: torch.Tensor = None,
):
    """Quadratic kata-attention forward (causal). Returns (o, den).

    c : optional (B, T, HQ) fp32 cumulative log-decay; if given the score is
        multiplied by exp(c_t - c_s) (GDN-style recency decay).

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
    o = (
        torch.empty_like(v)
        if v.shape[2] == HQ
        else torch.empty(
            B,
            T,
            HQ,
            V_d,
            device=dev,
            dtype=v.dtype,
        )
    )
    den = torch.empty(B, T, HQ, device=dev, dtype=torch.float32)
    has_decay = c is not None
    c_arg = c.contiguous() if has_decay else den  # dummy when unused (not read)

    # BT is autotuned: pass grid as a meta-aware lambda.
    grid = lambda meta: (NV, triton.cdiv(T, meta["BT"]), B * HQ)
    parallel_kata_attn_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        den,
        c_arg,
        scale,
        T,
        H=H,
        M=M,
        K_d=K_d,
        V_d=V_d,
        HQ=HQ,
        G=G,
        BV=BV,
        HAS_DECAY=has_decay,
    )
    return o, den


def _kata_decay_ref(q, k, v, c, num_groups, scale):
    """Differentiable PyTorch reference for the decayed normalized SPD attention
    (matches the Triton fwd: score (scale*q.k)^2 per group, * exp(min(c_t-c_s,0)),
    causal, sum-normalized). Materializes T^2 -- used only in the bwd recompute.
    """
    B, T, H, Kd = q.shape
    M = num_groups
    E = Kd // M
    qg = q.view(B, T, H, M, E)
    kg = k.view(B, T, H, M, E)
    qk = torch.einsum("bthme,bshme->bhtsm", qg, kg)
    A = ((scale * qk) ** 2).sum(-1)  # (B,H,T,T)
    cH = c.transpose(1, 2)  # (B,H,T)
    dec = torch.exp(torch.clamp(cH[:, :, :, None] - cH[:, :, None, :], max=0.0))
    tri = torch.tril(torch.ones(T, T, device=q.device, dtype=A.dtype))
    A = A * dec * tri
    o = torch.einsum("bhts,bshd->bthd", A, v) / (
        A.sum(-1).transpose(1, 2)[..., None] + 1e-20
    )
    return o


class _KataDecayFunction(torch.autograd.Function):
    """Triton forward (fast) + autograd-recompute backward (correct, incl. dc).
    The Triton decay backward is a perf TODO; this is correctness-complete."""

    @staticmethod
    def forward(ctx, q, k, v, c, num_groups, scale):
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1] // num_groups)
        o, _ = parallel_kata_attn_fwd_impl(q, k, v, num_groups, scale, 64, 64, c=c)
        ctx.save_for_backward(q, k, v, c)
        ctx.num_groups, ctx.scale = num_groups, scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, c = ctx.saved_tensors
        with torch.enable_grad():
            qd = q.detach().float().requires_grad_(True)
            kd = k.detach().float().requires_grad_(True)
            vd = v.detach().float().requires_grad_(True)
            cd = c.detach().float().requires_grad_(True)
            o = _kata_decay_ref(qd, kd, vd, cd, ctx.num_groups, ctx.scale)
            dq, dk, dv, dc = torch.autograd.grad(o, [qd, kd, vd, cd], do.float())
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dc.to(c.dtype),
            None,
            None,
        )


def parallel_kata_attn_decay(q, k, v, c, num_groups=2, scale=None):
    """Decayed SPD attention: o_t = sum_{s<=t} exp(c_t-c_s)(q.k)^2 v_s / sum(...).
    c: (B,T,HQ) fp32 cumulative log-decay. Triton fwd + autograd-recompute bwd.
    """
    return _KataDecayFunction.apply(q, k, v, c, num_groups, scale)


def parallel_kata_attn_fwd(q, k, v, num_groups=4, scale=None, BT=64, BS=64):
    """No-autograd wrapper around the forward kernel (used by inference/bench)."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1] // num_groups)
    return parallel_kata_attn_fwd_impl(q, k, v, num_groups, scale, BT, BS)



def parallel_kata_attn_bwd_triton(q, k, v, o, den, do, num_groups, scale, BT=64, BS=64):
    """Triton bwd dispatcher for M in {1, 2, 4}, MHA only (G=1).

    Falls back to the pytorch reference for other M or for GQA.
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    do = do.contiguous()
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
        o.contiguous().view(-1, V_d),
        do.view(-1, V_d),
        delta,
        NEL=NEL,
        V=V_d,
        num_warps=4,
        num_stages=1,
    )
    delta = delta.view(B, T, HQ)

    dq = torch.empty(B, T, HQ, K_d, device=dev, dtype=q.dtype)
    dk = torch.empty(B, T, H, K_d, device=dev, dtype=k.dtype)
    dv = torch.empty(B, T, H, V_d, device=dev, dtype=v.dtype)

    NT = triton.cdiv(T, BT)
    NS = triton.cdiv(T, BS)
    if M == 1:
        parallel_kata_attn_bwd_kernel_dq_M1[
            lambda meta: (1, triton.cdiv(T, meta["BT"]), B * HQ)
        ](
            q,
            k,
            v,
            den,
            delta,
            do,
            dq,
            scale,
            T,
            H=H,
            HQ=HQ,
            G=1,
            K_d=K_d,
            V_d=V_d,
            BV=V_d,
        )
        parallel_kata_attn_bwd_kernel_dkdv_M1[
            lambda meta: (1, triton.cdiv(T, meta["BS"]), B * HQ)
        ](
            q,
            k,
            v,
            den,
            delta,
            do,
            dk,
            dv,
            scale,
            T,
            H=H,
            HQ=HQ,
            G=1,
            K_d=K_d,
            V_d=V_d,
            BV=V_d,
        )
    elif M == 2:
        parallel_kata_attn_bwd_kernel_dq_M2[
            lambda meta: (1, triton.cdiv(T, meta["BT"]), B * HQ)
        ](
            q,
            k,
            v,
            den,
            delta,
            do,
            dq,
            scale,
            T,
            H=H,
            HQ=HQ,
            G=1,
            K_d=K_d,
            V_d=V_d,
            BV=V_d,
            E=E,
        )
        parallel_kata_attn_bwd_kernel_dkdv_M2[
            lambda meta: (1, triton.cdiv(T, meta["BS"]), B * HQ)
        ](
            q,
            k,
            v,
            den,
            delta,
            do,
            dk,
            dv,
            scale,
            T,
            H=H,
            HQ=HQ,
            G=1,
            K_d=K_d,
            V_d=V_d,
            BV=V_d,
            E=E,
        )
    else:  # M == 4
        parallel_kata_attn_bwd_kernel_dq_M4[
            lambda meta: (1, triton.cdiv(T, meta["BT"]), B * HQ)
        ](
            q,
            k,
            v,
            den,
            delta,
            do,
            dq,
            scale,
            T,
            H=H,
            HQ=HQ,
            G=1,
            K_d=K_d,
            V_d=V_d,
            BV=V_d,
            E=E,
        )
        parallel_kata_attn_bwd_kernel_dkdv_M4[
            lambda meta: (1, triton.cdiv(T, meta["BS"]), B * HQ)
        ](
            q,
            k,
            v,
            den,
            delta,
            do,
            dk,
            dv,
            scale,
            T,
            H=H,
            HQ=HQ,
            G=1,
            K_d=K_d,
            V_d=V_d,
            BV=V_d,
            E=E,
        )
    return dq, dk, dv


# =====================================================================
# Sum-SPD variant: same FlashAttention skeleton, score = Sum_{i,j} (q_i·k_j)²
# (M^2 inner products per (t,s) pair, vs concat's M).
# Matches the math used by paper KATA-SPD-4 (psi_packed variant="four_rank").
# =====================================================================


def parallel_kata_attn_sum_fwd_impl(q, k, v, num_groups, scale, BT, BS):
    """Sum-SPD forward. See parallel_kata_attn_sum_fwd_kernel."""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
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
        q,
        k,
        v,
        o,
        den,
        scale,
        T,
        H=H,
        HQ=HQ,
        G=G,
        K_d=K_d,
        V_d=V_d,
        BV=BV,
        M=M,
        E=E,
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

    qf = q.float()
    kf = k.float()
    vf = v.float()
    dof = do.float()
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
                    q,
                    k,
                    v,
                    o,
                    den,
                    do.contiguous(),
                    ctx.scale,
                )
            elif ctx.num_groups == 2:
                dq, dk, dv = parallel_kata_attn_sum_bwd_triton_M2(
                    q,
                    k,
                    v,
                    o,
                    den,
                    do.contiguous(),
                    ctx.scale,
                )
            elif ctx.num_groups == 1:
                # M=1: sum == concat == single squared dot.
                dq, dk, dv = parallel_kata_attn_bwd_triton(
                    q,
                    k,
                    v,
                    o,
                    den,
                    do.contiguous(),
                    1,
                    ctx.scale,
                    32,
                    32,
                )
            else:
                dq, dk, dv = parallel_kata_attn_bwd_sum_torch(
                    q,
                    k,
                    v,
                    o,
                    den,
                    do.contiguous(),
                    ctx.num_groups,
                    ctx.scale,
                )
        else:
            dq, dk, dv = parallel_kata_attn_bwd_sum_torch(
                q,
                k,
                v,
                o,
                den,
                do.contiguous(),
                ctx.num_groups,
                ctx.scale,
            )
        return dq, dk, dv, None, None, None, None, None


def parallel_kata_attn_sum(
    q, k, v, num_groups=4, scale=None, BT=64, BS=64, use_triton_bwd=True
):
    """Autograd-wrapped sum-SPD quadratic kata-attention (fwd + bwd, Triton).

    fwd: `parallel_kata_attn_sum_fwd_kernel` — M² inner products per (t,s)
         tile, no psi in HBM.
    bwd (M=4): `parallel_kata_attn_sum_bwd_kernel_dq_M4` +
               `parallel_kata_attn_sum_bwd_kernel_dkdv_M4` — only Q, K, V, O,
               den read from HBM; A and dA recomputed in SRAM per chunk.
    bwd (M=1): falls back to the (mathematically identical) concat M=1 kernel.
    bwd (M=2, M=3, ... or GQA): pytorch reference (slow, OOMs at production B).
    """
    return ParallelKataAttnSumFunction.apply(
        q, k, v, num_groups, scale, BT, BS, use_triton_bwd
    )





def parallel_kata_attn_sum_bwd_triton_M4(q, k, v, o, den, do, scale):
    """Triton sum-SPD bwd, M=4, MHA only."""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    do = do.contiguous()
    B, T, HQ, K_d = q.shape
    H = k.shape[2]
    V_d = v.shape[-1]
    assert H == HQ, "sum-SPD M=4 Triton bwd assumes MHA"
    M = 4
    E = K_d // M
    dev = q.device

    delta = torch.empty(B * T * HQ, device=dev, dtype=torch.float32)
    NEL = triton.next_power_of_2(V_d)
    parallel_kata_attn_bwd_preprocess[(B * T * HQ,)](
        o.contiguous().view(-1, V_d),
        do.view(-1, V_d),
        delta,
        NEL=NEL,
        V=V_d,
        num_warps=4,
        num_stages=1,
    )
    delta = delta.view(B, T, HQ)

    dq = torch.empty(B, T, HQ, K_d, device=dev, dtype=q.dtype)
    dk = torch.empty(B, T, H, K_d, device=dev, dtype=k.dtype)
    dv = torch.empty(B, T, H, V_d, device=dev, dtype=v.dtype)

    parallel_kata_attn_sum_bwd_kernel_dq_M4[
        lambda meta: (1, triton.cdiv(T, meta["BT"]), B * HQ)
    ](
        q,
        k,
        v,
        den,
        delta,
        do,
        dq,
        scale,
        T,
        H=H,
        HQ=HQ,
        G=1,
        K_d=K_d,
        V_d=V_d,
        BV=V_d,
        E=E,
    )
    parallel_kata_attn_sum_bwd_kernel_dkdv_M4[
        lambda meta: (1, triton.cdiv(T, meta["BS"]), B * HQ)
    ](
        q,
        k,
        v,
        den,
        delta,
        do,
        dk,
        dv,
        scale,
        T,
        H=H,
        HQ=HQ,
        G=1,
        K_d=K_d,
        V_d=V_d,
        BV=V_d,
        E=E,
    )
    return dq, dk, dv




def parallel_kata_attn_sum_bwd_triton_M2(q, k, v, o, den, do, scale):
    """Triton sum-SPD bwd, M=2."""
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    do = do.contiguous()
    B, T, HQ, K_d = q.shape
    H = k.shape[2]
    V_d = v.shape[-1]
    assert H == HQ
    M = 2
    E = K_d // M
    dev = q.device

    delta = torch.empty(B * T * HQ, device=dev, dtype=torch.float32)
    NEL = triton.next_power_of_2(V_d)
    parallel_kata_attn_bwd_preprocess[(B * T * HQ,)](
        o.contiguous().view(-1, V_d),
        do.view(-1, V_d),
        delta,
        NEL=NEL,
        V=V_d,
        num_warps=4,
        num_stages=1,
    )
    delta = delta.view(B, T, HQ)
    dq = torch.empty(B, T, HQ, K_d, device=dev, dtype=q.dtype)
    dk = torch.empty(B, T, H, K_d, device=dev, dtype=k.dtype)
    dv = torch.empty(B, T, H, V_d, device=dev, dtype=v.dtype)

    parallel_kata_attn_sum_bwd_kernel_dq_M2[
        lambda meta: (1, triton.cdiv(T, meta["BT"]), B * HQ)
    ](
        q,
        k,
        v,
        den,
        delta,
        do,
        dq,
        scale,
        T,
        H=H,
        HQ=HQ,
        G=1,
        K_d=K_d,
        V_d=V_d,
        BV=V_d,
        E=E,
    )
    parallel_kata_attn_sum_bwd_kernel_dkdv_M2[
        lambda meta: (1, triton.cdiv(T, meta["BS"]), B * HQ)
    ](
        q,
        k,
        v,
        den,
        delta,
        do,
        dk,
        dv,
        scale,
        T,
        H=H,
        HQ=HQ,
        G=1,
        K_d=K_d,
        V_d=V_d,
        BV=V_d,
        E=E,
    )
    return dq, dk, dv
