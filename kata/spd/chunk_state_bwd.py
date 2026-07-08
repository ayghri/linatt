"""O(T) chunked-STATE SPD delta BACKWARD (reference).

Reverse-scan over chunks carrying dS (the state gradient), the mirror of the forward
state recurrence S_{c+1} = S_c + psi(K_c)^T U_c. Makes fwd AND bwd both O(T) (GDN-style).

Forward (per chunk c, S_0=0):
    E   = P_k S_c                      P_k=psi(K_c), P_q=psi(Q_c)   (C x E^2)
    Vp  = V_c - E ;  bVp = beta_c ⊙ Vp
    Tm  = I + beta_c ⊙ tril(A_kk,-1) ,  A_kk = P_k P_k^T
    U   = Tm^{-1} bVp
    o_c = P_q S_c + tril(A_qk) U ,      A_qk = P_q P_k^T
    S_{c+1} = S_c + P_k^T U

This module is the correctness oracle for the Triton O(T) backward kernel.
"""
from __future__ import annotations
import torch

try:
    from .chunk_state_delta import _psi
except ImportError:
    from chunk_state_delta import _psi


def _psi_bwd(dP, x, M, s):
    """VJP of psi: given dP (.,C,E^2) return dx (.,C,D). psi[t,ij]=sum_g (s x_g[i])(s x_g[j])."""
    B, H, C, D = x.shape
    E = D // M
    dPm = dP.reshape(B, H, C, E, E)
    dPsym = dPm + dPm.transpose(-1, -2)                       # (B,H,C,E,E)
    xg = x.reshape(B, H, C, M, E)
    dxg = (s * s) * torch.einsum("bhcij,bhcgj->bhcgi", dPsym, xg)   # per group
    return dxg.reshape(B, H, C, D)


def spd_delta_state_bwd_ref(q, k, v, beta, do, M, scale=None, C=64):
    B, H, T, D = q.shape
    DV = v.shape[-1]; E = D // M
    s = (1.0 / E if scale is None else scale) ** 0.5
    NC = T // C
    dt = torch.float64
    q, k, v, beta, do = (x.to(dt) for x in (q, k, v, beta, do))
    pq = _psi(q, M, s); pk = _psi(k, M, s)                    # (B,H,T,E^2)
    eyeC = torch.eye(C, device=q.device, dtype=dt)
    # ---- forward, storing S_c (state BEFORE chunk c) and U_c ----
    S = torch.zeros(B, H, E * E, DV, device=q.device, dtype=dt)
    Sc, Uc, Tmc = [], [], []
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        Pk = pk[:, :, sl]; Pq = pq[:, :, sl]
        vc = v[:, :, sl]; bc = beta[:, :, sl]
        Sc.append(S.clone())
        Ec = Pk @ S
        Vp = vc - Ec
        Akk = Pk @ Pk.transpose(-1, -2)
        Tm = eyeC + bc[..., None] * torch.tril(Akk, -1)
        U = torch.linalg.solve_triangular(Tm, bc[..., None] * Vp, upper=False)
        Uc.append(U); Tmc.append(Tm)
        S = S + Pk.transpose(-1, -2) @ U
    # ---- reverse scan carrying dS (grad wrt S_{c+1}) ----
    dq = torch.zeros_like(q); dk = torch.zeros_like(k); dv = torch.zeros_like(v)
    dbeta = torch.zeros_like(beta)
    dS = torch.zeros(B, H, E * E, DV, device=q.device, dtype=dt)
    for c in reversed(range(NC)):
        sl = slice(c * C, (c + 1) * C)
        Pk = pk[:, :, sl]; Pq = pq[:, :, sl]
        vc = v[:, :, sl]; bc = beta[:, :, sl]; doc = do[:, :, sl]
        S_c = Sc[c]; U = Uc[c]; Tm = Tmc[c]
        Akk = Pk @ Pk.transpose(-1, -2)
        Aqk = Pq @ Pk.transpose(-1, -2); Lqk = torch.tril(Aqk)
        # --- readout o = Pq S_c + Lqk U ---
        dPq = doc @ S_c.transpose(-1, -2)                    # from Pq S_c
        dS_c = Pq.transpose(-1, -2) @ doc                    # grad wrt S_c (readout)
        dU = Lqk.transpose(-1, -2) @ doc                     # from Lqk U
        dLqk = torch.tril(doc @ U.transpose(-1, -2))
        # --- state update S_{c+1} = S_c + Pk^T U (dS = grad wrt S_{c+1}) ---
        dU = dU + Pk @ dS                                    # U feeds the update
        dPk = U @ dS.transpose(-1, -2)                       # Pk^T U -> dPk (C,E^2)
        # --- U = Tm^{-1} bVp ---
        dbVp = torch.linalg.solve_triangular(Tm.transpose(-1, -2), dU, upper=True)
        dTm = -dbVp @ U.transpose(-1, -2)
        Vp = vc - Pk @ S_c
        dbeta_c = (Vp * dbVp).sum(-1)                        # from bVp = beta ⊙ Vp
        dVp = bc[..., None] * dbVp
        dv[:, :, sl] = dVp                                   # Vp = v - E
        dE = -dVp
        dPk = dPk + dE @ S_c.transpose(-1, -2)               # E = Pk S_c
        dS_c = dS_c + Pk.transpose(-1, -2) @ dE              # grad wrt S_c (erase)
        # --- Tm = I + beta ⊙ tril(Akk,-1) ---
        dAkk = bc[..., None] * torch.tril(dTm, -1)
        dbeta_c = dbeta_c + (torch.tril(Akk, -1) * dTm).sum(-1)
        dPk = dPk + (dAkk + dAkk.transpose(-1, -2)) @ Pk     # Akk = Pk Pk^T
        # --- Aqk = Pq Pk^T ---
        dPq = dPq + dLqk @ Pk
        dPk = dPk + dLqk.transpose(-1, -2) @ Pq
        # --- psi VJP -> dq, dk ---
        dq[:, :, sl] = _psi_bwd(dPq, q[:, :, sl], M, s)
        dk[:, :, sl] = _psi_bwd(dPk, k[:, :, sl], M, s)
        dbeta[:, :, sl] = dbeta_c
        # pass dS to previous chunk (S_c = S_{(c-1)+1})
        dS = dS + dS_c
    return dq, dk, dv, dbeta
