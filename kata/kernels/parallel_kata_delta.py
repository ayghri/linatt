"""Chunked SPD delta-rule Triton kernel (style of parallel_kata.py; M=2 specialized).

Pure SPD delta-rule (no decay), psi-normalized (q,k pre-scaled outside). Chunked with
implicit E^2 state (2D tl.dot throughout, M-specialized named accumulators G0/G1).

Per chunk (state G_m = sum_s k_s,m (x) k_s,m U_s, U = WY-corrected values):
  read_k_i = sum_m k_i,m^T G_m k_i,m ; read_q_i = sum_m q_i,m^T G_m q_i,m   (state before chunk)
  Skk[i,j]=sum_m (k_i.k_j)^2 ; Sqk[i,j]=sum_m (q_i.k_j)^2                    (intra, per-group dot^2)
  A = beta * tril(Skk,-1);  U = (I+A)^{-1} (beta*(V - read_k))               (WY, nilpotent-doubling)
  o_i = read_q_i + sum_{j<=i} Sqk[i,j] U_j
  G_m += sum_i (k_i,m (x) k_i,m) U_i

(I+A)^{-1} = prod_{i<log2 C} (I + P^{2^i}),  P=-A  (P strictly-lower nilpotent, P^C=0). All tl.dot.
"""
import math
import torch
import triton
import triton.language as tl

_TF = True   # tf32 for 3090 correctness; H100 can use fp32


@triton.jit
def _read(b_x, Gm, C: tl.constexpr, E: tl.constexpr, BV: tl.constexpr, TF: tl.constexpr):
    tmp = tl.dot(b_x, Gm, allow_tf32=TF)                 # (C, E*BV)
    tmp3 = tl.reshape(tmp, (C, E, BV))
    return tl.sum(b_x[:, :, None] * tmp3, axis=1)        # (C, BV)


@triton.jit
def _upd(b_k, U, C: tl.constexpr, E: tl.constexpr, BV: tl.constexpr, TF: tl.constexpr):
    ku = tl.reshape(b_k[:, :, None] * U[:, None, :], (C, E * BV))   # (C, E*BV)
    return tl.dot(tl.trans(b_k), ku, allow_tf32=TF)                 # (E, E*BV)


@triton.jit
def spd_delta_chunk_fwd_kernel_M2(
    q, k, v, beta, o, T,
    H: tl.constexpr, E: tl.constexpr, DV: tl.constexpr,
    C: tl.constexpr, BV: tl.constexpr, LOGC: tl.constexpr, TF: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_dv = tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    Kd: tl.constexpr = 2 * E
    bos = i_b * T
    o_c = tl.arange(0, C)
    eye = (o_c[:, None] == o_c[None, :]).to(tl.float32)
    strict = (o_c[:, None] > o_c[None, :]).to(tl.float32)     # strictly lower
    incl = (o_c[:, None] >= o_c[None, :]).to(tl.float32)      # lower incl

    G0 = tl.zeros([E, E * BV], dtype=tl.float32)
    G1 = tl.zeros([E, E * BV], dtype=tl.float32)

    for i_c in range(0, tl.cdiv(T, C)):
        t0 = i_c * C
        cmask = (t0 + o_c) < T
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * DV, (T, DV), (H * DV, 1), (t0, i_dv * BV), (C, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        p_be = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (t0,), (C,), (0,))
        b_be = tl.load(p_be, boundary_check=(0,)).to(tl.float32)          # (C,)

        p_q0 = tl.make_block_ptr(q + (bos * H + i_h) * Kd, (T, Kd), (H * Kd, 1), (t0, 0), (C, E), (1, 0))
        p_k0 = tl.make_block_ptr(k + (bos * H + i_h) * Kd, (T, Kd), (H * Kd, 1), (t0, 0), (C, E), (1, 0))
        p_q1 = tl.make_block_ptr(q + (bos * H + i_h) * Kd, (T, Kd), (H * Kd, 1), (t0, E), (C, E), (1, 0))
        p_k1 = tl.make_block_ptr(k + (bos * H + i_h) * Kd, (T, Kd), (H * Kd, 1), (t0, E), (C, E), (1, 0))
        b_q0 = tl.load(p_q0, boundary_check=(0, 1)).to(tl.float32)
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
        b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)

        # inter-chunk reads from state (before this chunk)
        read_k = _read(b_k0, G0, C, E, BV, TF) + _read(b_k1, G1, C, E, BV, TF)
        read_q = _read(b_q0, G0, C, E, BV, TF) + _read(b_q1, G1, C, E, BV, TF)
        # intra grams
        qk0 = tl.dot(b_q0, tl.trans(b_k0), allow_tf32=TF); qk1 = tl.dot(b_q1, tl.trans(b_k1), allow_tf32=TF)
        kk0 = tl.dot(b_k0, tl.trans(b_k0), allow_tf32=TF); kk1 = tl.dot(b_k1, tl.trans(b_k1), allow_tf32=TF)
        Sqk = qk0 * qk0 + qk1 * qk1
        Skk = kk0 * kk0 + kk1 * kk1
        # WY: A = beta * strict(Skk); T=(I+A)^-1 = prod (I + P^{2^i}), P=-A
        A = b_be[:, None] * (Skk * strict)
        P = -A
        Tm = eye + P
        for _i in tl.static_range(1, LOGC):
            P = tl.dot(P, P, allow_tf32=TF)
            Tm = Tm + tl.dot(Tm, P, allow_tf32=TF)
        rhs = b_be[:, None] * (b_v - read_k)                 # (C, BV)
        U = tl.dot(Tm, rhs, allow_tf32=TF)                   # (C, BV)
        # output
        Sqk_c = tl.where(incl > 0, Sqk, 0.0) * (cmask[None, :]).to(tl.float32)
        b_o = read_q + tl.dot(Sqk_c, U, allow_tf32=TF)
        p_o = tl.make_block_ptr(o + (bos * H + i_h) * DV, (T, DV), (H * DV, 1), (t0, i_dv * BV), (C, BV), (1, 0))
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        # state update with U
        G0 += _upd(b_k0, U, C, E, BV, TF)
        G1 += _upd(b_k1, U, C, E, BV, TF)


def _prescale(x):
    n2 = (x * x).sum(-1); a = (n2 * n2).sum(-1).clamp_min(1e-12).pow(-0.25)
    return x * a[..., None, None]


def spd_delta(q, k, v, beta, chunk_size=32, BV=16, prescale=True):
    """q,k (B,T,H,2,E)  v (B,T,H,DV)  beta (B,T,H). Pure SPD delta-rule (psi-norm)."""
    B, T, H, M, E = q.shape
    assert M == 2 and (chunk_size & (chunk_size - 1)) == 0
    if prescale:
        q, k = _prescale(q), _prescale(k)
    DV = v.shape[-1]
    o = torch.empty(B, T, H, DV, device=q.device, dtype=torch.float32)
    grid = (B * H, DV // BV)
    spd_delta_chunk_fwd_kernel_M2[grid](
        q.reshape(B, T, H, M * E).contiguous(), k.reshape(B, T, H, M * E).contiguous(),
        v.contiguous(), beta.contiguous(), o,
        T, H=H, E=E, DV=DV, C=chunk_size, BV=BV, LOGC=int(math.log2(chunk_size)), TF=_TF,
    )
    return o

@triton.jit
def spd_delta_U_kernel_M2(
    q, k, v, beta, Uout, T,
    H: tl.constexpr, E: tl.constexpr, DV: tl.constexpr,
    C: tl.constexpr, BV: tl.constexpr, LOGC: tl.constexpr, TF: tl.constexpr,
):
    # computes ONLY U (WY-corrected values); output o = parallel_kata(q,k,U) done elsewhere.
    i_bh = tl.program_id(0); i_dv = tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    Kd: tl.constexpr = 2 * E; bos = i_b * T
    o_c = tl.arange(0, C)
    eye = (o_c[:, None] == o_c[None, :]).to(tl.float32)
    strict = (o_c[:, None] > o_c[None, :]).to(tl.float32)
    G0 = tl.zeros([E, E * BV], dtype=tl.float32); G1 = tl.zeros([E, E * BV], dtype=tl.float32)
    for i_c in range(0, tl.cdiv(T, C)):
        t0 = i_c * C
        p_v = tl.make_block_ptr(v + (bos*H+i_h)*DV, (T,DV), (H*DV,1), (t0,i_dv*BV), (C,BV), (1,0))
        b_v = tl.load(p_v, boundary_check=(0,1)).to(tl.float32)
        p_be = tl.make_block_ptr(beta + bos*H+i_h, (T,), (H,), (t0,), (C,), (0,))
        b_be = tl.load(p_be, boundary_check=(0,)).to(tl.float32)
        p_k0 = tl.make_block_ptr(k+(bos*H+i_h)*Kd,(T,Kd),(H*Kd,1),(t0,0),(C,E),(1,0))
        p_k1 = tl.make_block_ptr(k+(bos*H+i_h)*Kd,(T,Kd),(H*Kd,1),(t0,E),(C,E),(1,0))
        b_k0 = tl.load(p_k0, boundary_check=(0,1)).to(tl.float32)
        b_k1 = tl.load(p_k1, boundary_check=(0,1)).to(tl.float32)
        read_k = _read(b_k0,G0,C,E,BV,TF) + _read(b_k1,G1,C,E,BV,TF)
        kk0 = tl.dot(b_k0, tl.trans(b_k0), allow_tf32=TF); kk1 = tl.dot(b_k1, tl.trans(b_k1), allow_tf32=TF)
        Skk = kk0*kk0 + kk1*kk1
        A = b_be[:,None]*(Skk*strict); P = -A; Tm = eye + P
        for _i in tl.static_range(1, LOGC):
            P = tl.dot(P,P,allow_tf32=TF); Tm = Tm + tl.dot(Tm,P,allow_tf32=TF)
        U = tl.dot(Tm, b_be[:,None]*(b_v-read_k), allow_tf32=TF)
        p_U = tl.make_block_ptr(Uout+(bos*H+i_h)*DV,(T,DV),(H*DV,1),(t0,i_dv*BV),(C,BV),(1,0))
        tl.store(p_U, U.to(p_U.dtype.element_ty), boundary_check=(0,1))
        G0 += _upd(b_k0,U,C,E,BV,TF); G1 += _upd(b_k1,U,C,E,BV,TF)


def spd_delta_fast(q, k, v, beta, chunk_size=64, BV=16, prescale=True):
    """U via the state kernel, then o = parallel_kata(q,k,U)*den (reuses the fast SPD-attn)."""
    try:
        from kata.parallel_kata_attn import parallel_kata_attn_fwd_impl
    except ImportError:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from parallel_kata_attn import parallel_kata_attn_fwd_impl
    B, T, H, M, E = q.shape; DV = v.shape[-1]
    if prescale:
        q, k = _prescale(q), _prescale(k)
    U = torch.empty(B, T, H, DV, device=q.device, dtype=torch.float32)
    spd_delta_U_kernel_M2[(B*H, DV//BV)](
        q.reshape(B,T,H,M*E).contiguous(), k.reshape(B,T,H,M*E).contiguous(),
        v.contiguous(), beta.contiguous(), U,
        T, H=H, E=E, DV=DV, C=chunk_size, BV=BV, LOGC=int(math.log2(chunk_size)), TF=_TF)
    o_norm, den = parallel_kata_attn_fwd_impl(
        q.reshape(B,T,H,M*E).bfloat16(), k.reshape(B,T,H,M*E).bfloat16(), U.bfloat16(), M, 1.0, 64, 64)
    return o_norm.float() * den.float()[..., None]
