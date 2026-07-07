"""Chunked-STATE SPD delta: carry S = E^2 x DV across chunks (GDN-style, O(T)), materialize
psi (E^2 feature). DV-tiling splits S across programs so the E^2 state fits SRAM for M>=2.
Cost O(T*E^2*DV) beats flash O(T^2*DV) once E^2 < T (M>=2 at T=2048)."""
from __future__ import annotations
import torch


def _psi(x, M, s):
    """psi(x) = sum_g (s x_g) (x) (s x_g) in R^{E^2}, so <psi(a),psi(b)> = sum_gh (s^2 a_g.b_h)^2."""
    B, H, T, D = x.shape
    E = D // M
    xg = (x * s).reshape(B, H, T, M, E)
    outer = xg.unsqueeze(-1) * xg.unsqueeze(-2)      # (B,H,T,M,E,E)
    return outer.sum(3).reshape(B, H, T, E * E)      # sum over groups -> (B,H,T,E^2)


def spd_delta_state_ref(q, k, v, beta, M, scale=None, C=64):
    B, H, T, D = q.shape
    DV = v.shape[-1]; E = D // M
    s = (1.0 / E if scale is None else scale) ** 0.5
    NC = T // C
    pq = _psi(q.float(), M, s); pk = _psi(k.float(), M, s)
    S = torch.zeros(B, H, E * E, DV, device=q.device, dtype=torch.float64)
    o = torch.zeros(B, H, T, DV, device=q.device, dtype=torch.float64)
    eyeC = torch.eye(C, device=q.device, dtype=torch.float64)
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        pkc = pk[:, :, sl].double(); pqc = pq[:, :, sl].double()
        vc = v[:, :, sl].double(); bc = beta[:, :, sl].double()
        Vp = vc - pkc @ S                                        # inter-chunk erase via state
        Akk = pkc @ pkc.transpose(-1, -2)                        # (B,H,C,C) intra gram
        Tm = eyeC + bc[..., None] * torch.tril(Akk, -1)
        U = torch.linalg.solve_triangular(Tm, bc[..., None] * Vp, upper=False)
        Aqk = pqc @ pkc.transpose(-1, -2)
        o[:, :, sl] = pqc @ S + torch.tril(Aqk) @ U              # state readout + intra
        S = S + pkc.transpose(-1, -2) @ U                        # state update
    return o


import triton
import triton.language as tl

_SCFG = [triton.Config({}, num_warps=w, num_stages=1) for w in (2, 4, 8)]


@triton.autotune(configs=_SCFG, key=['T', 'C', 'M', 'DV', 'DVB', 'E2'])
@triton.jit
def _state_fwd_kernel(q, k, v, beta, o, s_scale, T,
                      H: tl.constexpr, D: tl.constexpr, E: tl.constexpr, M: tl.constexpr,
                      DV: tl.constexpr, DVB: tl.constexpr, C: tl.constexpr, LOGC: tl.constexpr,
                      E2: tl.constexpr):
    """Chunked-STATE SPD delta. Carry S (E2 x DVB) in SRAM across chunks (O(T)). DV-tiled by
    program_id(1). psi(K_c),psi(Q_c) materialized C x E2 (small C so the tile fits)."""
    i_bh = tl.program_id(0).to(tl.int64)
    dv_off = tl.program_id(1) * DVB
    bqk = i_bh * T * D
    bv = i_bh * T * DV
    NC = tl.cdiv(T, C)
    o_row = tl.arange(0, C)
    eye = (o_row[:, None] == o_row[None, :]).to(tl.float32)
    strict = (o_row[:, None] > o_row[None, :]).to(tl.float32)
    re2 = tl.arange(0, E2)
    rdv = tl.arange(0, DVB)
    S = tl.zeros([E2, DVB], dtype=tl.float32)
    for c in range(NC):
        c0 = c * C
        rowd = (c0 + o_row)[:, None] * DV + (dv_off + rdv)[None, :]
        b_v = tl.load(v + bv + rowd, mask=(c0 + o_row)[:, None] < T, other=0.0).to(tl.float32)
        b_beta = tl.load(beta + i_bh * T + c0 + o_row, mask=(c0 + o_row) < T, other=0.0).to(tl.float32)
        # materialize psi(K_c), psi(Q_c)  (C, E2)
        psiK = tl.zeros([C, E2], dtype=tl.float32)
        psiQ = tl.zeros([C, E2], dtype=tl.float32)
        for g in range(M):
            kg = tl.load(tl.make_block_ptr(k + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
            qg = tl.load(tl.make_block_ptr(q + bqk, (T, D), (D, 1), (c0, g * E), (C, E), (1, 0)), boundary_check=(0, 1)) * s_scale
            psiK += tl.reshape(kg[:, :, None] * kg[:, None, :], (C, E2))
            psiQ += tl.reshape(qg[:, :, None] * qg[:, None, :], (C, E2))
        psiKb = psiK.to(tl.bfloat16)
        psiQb = psiQ.to(tl.bfloat16)
        Sb = S.to(tl.bfloat16)
        # inter-chunk erase + intra solve
        Ec = tl.dot(psiKb, Sb)                                   # (C,DVB)
        Vp = b_v - Ec
        Akk = tl.dot(psiKb, tl.trans(psiKb))                     # (C,C)
        Pm = -(b_beta[:, None] * (Akk * strict))
        Tm = eye + Pm
        for _i in tl.static_range(1, LOGC):
            Pm = tl.dot(Pm.to(tl.bfloat16), Pm.to(tl.bfloat16))
            Tm = Tm + tl.dot(Tm.to(tl.bfloat16), Pm.to(tl.bfloat16))
        U = tl.dot(Tm.to(tl.bfloat16), (b_beta[:, None] * Vp).to(tl.bfloat16))   # (C,DVB)
        # readout
        Aqk = tl.dot(psiQb, tl.trans(psiKb))                     # (C,C)
        oc = tl.dot(psiQb, Sb) + tl.dot((tl.where(o_row[:, None] >= o_row[None, :], Aqk, 0.0)).to(tl.bfloat16), U.to(tl.bfloat16))
        tl.store(o + bv + rowd, oc.to(o.dtype.element_ty), mask=(c0 + o_row)[:, None] < T)
        # state update
        S += tl.dot(tl.trans(psiKb), U.to(tl.bfloat16))          # (E2,DVB)


def spd_delta_state(q, k, v, beta, M, scale=None, C=32, DVB=16):
    import math
    B, H, T, D = q.shape
    DV = v.shape[-1]; E = D // M; E2 = E * E
    if scale is None:
        scale = 1.0 / E
    DVB = min(DVB, DV)
    while DV % DVB != 0:
        DVB //= 2
    q, k, v, beta = (x.contiguous() for x in (q, k, v, beta))
    o = torch.empty(B, H, T, DV, device=q.device, dtype=torch.float32)
    _state_fwd_kernel[(B * H, DV // DVB)](q, k, v, beta, o, scale ** 0.5, T,
                                          H=H, D=D, E=E, M=M, DV=DV, DVB=DVB, C=C,
                                          LOGC=int(math.log2(C)), E2=E2)
    return o
