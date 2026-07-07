"""KATA-SPD attention: references + Triton kernels, FLA-structured.

Two score variants (q, k split into M groups along the head dim, E = d_head / M):
  - CONCAT (same-group):  A[t,s] = sum_i  <q_i, k_s_i>^2          (i = 1..M)
  - SUM   (cross-group):  A[t,s] = sum_ij <q_i, k_s_j>^2          (i,j = 1..M)

Both are inner products of an implicit quadratic feature map psi:
  - CONCAT: psi(x) = concat_i vec(x_i ⊗ x_i)   -> dim  M * E^2
  - SUM:    psi(x) = sum_i    vec(x_i ⊗ x_i)    -> dim  E^2
so  A[t,s] = <psi(q_t), psi(k_s)>  >= 0, which gives two equivalent computations:
  - PARALLEL (flash-like): materialize A blockwise, causal-mask, normalize by the
    row sum, multiply by V.  O(T^2), no state.
  - RECURRENT (GDN-like):  carry state  S = sum_s psi(k_s) v_s^T  and
    z = sum_s psi(k_s); then o_t = (psi(q_t) S) / (psi(q_t) z).  Linear in T.

This module holds the pure-torch references (autograd-differentiable, ground
truth). Triton kernels live alongside and are checked against these.
"""
from __future__ import annotations

import torch
from torch import Tensor

EPS = 1e-6


# --------------------------------------------------------------------------- #
# explicit quadratic feature maps (for the recurrent/state reference)
# --------------------------------------------------------------------------- #
def psi_concat(x: Tensor, M: int) -> Tensor:
    """concat-SPD feature map: concat_i vec(x_i ⊗ x_i).  (..., d) -> (..., M*E^2)."""
    *b, d = x.shape
    E = d // M
    xg = x.reshape(*b, M, E)
    outer = xg.unsqueeze(-1) * xg.unsqueeze(-2)          # (..., M, E, E)
    return outer.reshape(*b, M * E * E)


def psi_sum(x: Tensor, M: int) -> Tensor:
    """sum-SPD feature map: sum_i vec(x_i ⊗ x_i).  (..., d) -> (..., E^2)."""
    *b, d = x.shape
    E = d // M
    xg = x.reshape(*b, M, E)
    outer = (xg.unsqueeze(-1) * xg.unsqueeze(-2)).sum(-3)  # (..., E, E)
    return outer.reshape(*b, E * E)


# --------------------------------------------------------------------------- #
# scores  A[t,s]  (the thing the parallel kernel computes blockwise)
# --------------------------------------------------------------------------- #
def spd_scores(q: Tensor, k: Tensor, M: int, mode: str) -> Tensor:
    """A[...,t,s] from q,k of shape (..., T, d). mode in {'concat','sum'}."""
    qs = q.chunk(M, dim=-1)
    ks = k.chunk(M, dim=-1)
    if mode == "concat":
        return sum((qs[i] @ ks[i].transpose(-1, -2)) ** 2 for i in range(M))
    if mode == "sum":
        return sum(
            (qs[i] @ ks[j].transpose(-1, -2)) ** 2
            for i in range(M) for j in range(M)
        )
    raise ValueError(mode)


# --------------------------------------------------------------------------- #
# PARALLEL reference (flash-like): O(T^2), exact ground truth
# --------------------------------------------------------------------------- #
def spd_parallel_ref(q, k, v, M: int, mode: str, scale: float | None = None) -> Tensor:
    """q,k: (B,H,T,d)  v: (B,H,T,dv) -> o: (B,H,T,dv). Causal, sum-normalized."""
    if scale is None:
        scale = 1.0 / (q.shape[-1] // M)
    A = spd_scores(q, k, M, mode) * (scale ** 2)          # scale applies per dot, squared
    T = q.shape[-2]
    causal = torch.ones(T, T, device=q.device, dtype=A.dtype).tril()
    A = A * causal
    den = A.sum(-1, keepdim=True).clamp_min(EPS)
    return (A @ v) / den


# --------------------------------------------------------------------------- #
# RECURRENT reference (GDN-like): linear scan via explicit psi; must match parallel
# --------------------------------------------------------------------------- #
def spd_recurrent_ref(q, k, v, M: int, mode: str, scale: float | None = None,
                      return_state: bool = False):
    """Token scan with state S = sum psi(k) v^T, z = sum psi(k)."""
    if scale is None:
        scale = 1.0 / (q.shape[-1] // M)
    feat = psi_concat if mode == "concat" else psi_sum
    pq = feat(q, M) * scale                               # scale once per feature
    pk = feat(k, M) * scale
    B, H, T, _ = q.shape
    P, Dv = pq.shape[-1], v.shape[-1]
    S = q.new_zeros(B, H, P, Dv)
    z = q.new_zeros(B, H, P)
    out = q.new_empty(B, H, T, Dv)
    for t in range(T):
        S = S + pk[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
        z = z + pk[:, :, t]
        num = (pq[:, :, t].unsqueeze(-1) * S).sum(-2)
        den = (pq[:, :, t] * z).sum(-1, keepdim=True).clamp_min(EPS)
        out[:, :, t] = num / den
    return (out, S, z) if return_state else out


# --------------------------------------------------------------------------- #
# CHUNKED reference (GDN-like): the algorithm the chunk kernel implements.
# Per-group state S_g (E,E,Dv), z_g (E,E). Inter-chunk term is a bilinear form
#   o_inter[t] = sum_g q_g[t]^T S_g q_g[t]      (no E^2 feature materialized)
# concat: M independent states (group i <-> state i);
# sum:    ONE shared state, updated by all key groups, queried by all q groups.
# scale is folded as q' = sqrt(scale)*q, k' = sqrt(scale)*k so A = <psi(q'),psi(k')>.
# --------------------------------------------------------------------------- #
def spd_chunked_ref(q, k, v, M: int, mode: str, scale: float | None = None,
                    C: int = 16) -> Tensor:
    if scale is None:
        scale = 1.0 / (q.shape[-1] // M)
    B, H, T, D = q.shape
    E = D // M
    Dv = v.shape[-1]
    s = scale ** 0.5
    qf, kf = q * s, k * s                                  # fold scale into features
    nG = M if mode == "concat" else 1
    S = q.new_zeros(B, H, nG, E, E, Dv)
    z = q.new_zeros(B, H, nG, E, E)
    out = q.new_empty(B, H, T, Dv)

    for c0 in range(0, T, C):
        c1 = min(c0 + C, T)
        Qf, Kf, Vc = qf[:, :, c0:c1], kf[:, :, c0:c1], v[:, :, c0:c1]
        cT = c1 - c0
        # ---- inter-chunk: bilinear form against the carried state ----
        num_i = q.new_zeros(B, H, cT, Dv)
        den_i = q.new_zeros(B, H, cT)
        for gi in range(M):
            qi = Qf[..., gi * E:(gi + 1) * E]
            sg = gi if mode == "concat" else 0
            num_i += torch.einsum("bhtx,bhty,bhxyv->bhtv", qi, qi, S[:, :, sg])
            den_i += torch.einsum("bhtx,bhty,bhxy->bht", qi, qi, z[:, :, sg])
        # ---- intra-chunk: causal quadratic scores within the chunk ----
        A = spd_scores(Qf, Kf, M, mode)                   # scale already folded
        A = A * torch.ones(cT, cT, device=q.device, dtype=A.dtype).tril()
        num_a = A @ Vc
        den_a = A.sum(-1)
        den = (den_i + den_a).clamp_min(EPS)
        out[:, :, c0:c1] = (num_i + num_a) / den.unsqueeze(-1)
        # ---- update state with this chunk's keys ----
        for gj in range(M):
            kj = Kf[..., gj * E:(gj + 1) * E]
            sg = gj if mode == "concat" else 0
            S[:, :, sg] += torch.einsum("bhsx,bhsy,bhsv->bhxyv", kj, kj, Vc)
            z[:, :, sg] += torch.einsum("bhsx,bhsy->bhxy", kj, kj)
    return out


# --------------------------------------------------------------------------- #
# self-check: parallel == recurrent (validates the math + feature maps)
# --------------------------------------------------------------------------- #
def _rel(a, b):
    return ((a - b).abs().mean() / b.abs().mean().clamp_min(1e-12)).item()


def _selfcheck():
    torch.manual_seed(0)
    B, H, T, d, dv = 2, 2, 48, 16, 16
    for M in (2, 4):
        for mode in ("concat", "sum"):
            q = torch.randn(B, H, T, d, dtype=torch.float64) * 0.3
            k = torch.randn(B, H, T, d, dtype=torch.float64) * 0.3
            v = torch.randn(B, H, T, dv, dtype=torch.float64)
            o_par = spd_parallel_ref(q, k, v, M, mode)
            o_rec = spd_recurrent_ref(q, k, v, M, mode)
            o_chk = spd_chunked_ref(q, k, v, M, mode, C=16)
            r_rec = _rel(o_rec, o_par)
            r_chk = _rel(o_chk, o_par)
            print(f"  M={M} mode={mode:>6}: recurrent rel={r_rec:.1e}  chunked rel={r_chk:.1e}")
            assert max(r_rec, r_chk) < 1e-10, "references disagree"
    print("OK: parallel == recurrent == chunked for all variants")


if __name__ == "__main__":
    _selfcheck()
