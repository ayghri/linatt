"""KATA feature maps psi: R^d -> R^q.

These are nonnegative finite feature maps used in place of softmax for linear
attention. The choice of psi controls the cone the attention weights live in
and, per the paper, the recall capacity under the Welch floor.

Implementations are pure torch v1; matched-precision triton variants land in a
later iteration once the LM-scale wiring is validated.

Variants
--------
positive
    psi(x) = relu(x) + eps. Output dim q = d.
lorentz
    psi(x) = [x[..., :d-1], ||x[..., :d-1]|| * (1 + x[..., d-1]**2)]. q = d.
    Image lies on the Lorentz cone {x_d >= ||x_{1..d-1}||}.
spd_concat
    Split x in R^d into M groups of E = d/M dims. For each group g_i,
    return the lower-triangle of g_i g_i^T (E*(E+1)/2 entries). Concat over M
    groups. Output dim q = M * E*(E+1)/2.
    The off-diagonals are scaled by sqrt(2) so that <psi(x), psi(y)> equals
    sum_i <g_i, h_i>**2 -- the squared inner product per group.

All maps are nonnegative when they need to be (positive: yes; lorentz: yes
because the last coord dominates; spd_concat: diagonals are nonneg, and
nonnegativity of <psi(x), psi(y)> follows from <g_i,h_i>**2 >= 0).
"""

import math

import torch
import torch.nn as nn


def positive(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.relu(x) + eps


def lorentz(x: torch.Tensor) -> torch.Tensor:
    head = x[..., :-1]
    tail = x[..., -1:]
    norm = head.norm(dim=-1, keepdim=True)
    return torch.cat([head, norm * (1.0 + tail * tail)], dim=-1)


def spd_concat(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Concatenated SPD one-rank feature map.

    Args:
        x: (..., d). d must be divisible by num_groups.
        num_groups: M. groups have dim E = d / M.

    Returns:
        (..., M * E*(E+1)//2). For each group g of shape (E,):
        [diag(g g^T), sqrt(2) * tril(g g^T, -1) flattened] = squared
        coordinates concatenated with sqrt(2) * pairwise products.
        The sqrt(2) on off-diagonals matches <psi(g), psi(h)> = <g, h>**2.
    """
    *prefix, d = x.shape
    if d % num_groups != 0:
        raise ValueError(
            f"x last dim {d} not divisible by num_groups {num_groups}"
        )
    e = d // num_groups
    g = x.view(*prefix, num_groups, e)

    diag = g * g  # (..., M, E)

    if e == 1:
        return diag.reshape(*prefix, num_groups * e)

    # off-diag: outer product, then take strict lower-tri, scaled by sqrt(2).
    outer = g.unsqueeze(-1) * g.unsqueeze(-2)  # (..., M, E, E)
    ii, jj = torch.tril_indices(e, e, offset=-1, device=x.device)
    off = outer[..., ii, jj] * math.sqrt(2.0)  # (..., M, E*(E-1)/2)

    packed = torch.cat([diag, off], dim=-1)  # (..., M, E*(E+1)/2)
    return packed.reshape(*prefix, num_groups * (e * (e + 1) // 2))


def spd_full(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Full outer-product SPD feature map (no sqrt(2) packing).

    Output dim per group is E² (vs E·(E+1)/2 for packed). Wasteful by 2x but
    yields a power-of-2 output dim whenever E is a power of 2 — which our
    triton tile shapes (tl.arange) require.

    <psi_full(g), psi_full(h)> = sum_{i,j} g_i g_j h_i h_j = (g·h)²
    matching the packed form's inner product.
    """
    *prefix, d = x.shape
    if d % num_groups != 0:
        raise ValueError(f"d={d} not divisible by M={num_groups}")
    e = d // num_groups
    g = x.view(*prefix, num_groups, e)
    out = (g.unsqueeze(-1) * g.unsqueeze(-2)).reshape(
        *prefix, num_groups * e * e
    )
    return out


def spd_sum(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Multi-group SPD with summed outer products, FULL E×E storage.

    Split x in R^d into M groups of E = d/M. Return the flattened
    full Sum_i g_i g_i^T  (E×E matrix). Has redundant symmetric storage;
    see `spd_sum_packed` for the half-storage variant.

    Output dim q = E² = d²/M². Kernel identity:
        <psi(x), psi(y)> = Sum_{i,j} (g_i · h_j)²
    """
    *prefix, d = x.shape
    if d % num_groups != 0:
        raise ValueError(f"d={d} not divisible by M={num_groups}")
    e = d // num_groups
    g = x.view(*prefix, num_groups, e)
    out = torch.einsum("...mi,...mj->...ij", g, g)
    return out.reshape(*prefix, e * e)


def spd_sum_packed(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Multi-group SPD with summed outer products, PACKED storage.

    Sum_i g_i g_i^T is symmetric so we keep only the lower triangle:
    output dim q = E(E+1)/2 instead of E². Off-diagonal entries scaled
    by sqrt(2) so that <packed(x), packed(y)> = Frobenius(M_x, M_y) =
    Sum_{i,j} (g_i · h_j)².

    For d=64, M=4: q = 16*17/2 = 136 (vs 256 for spd_sum).
    For d=64, M=8: q = 8*9/2  = 36  (vs 64).
    For d=128, M=4: q = 32*33/2 = 528 (vs 1024).

    Args:
        x: (..., d). d must be divisible by num_groups.
        num_groups: M.
    """
    *prefix, d = x.shape
    if d % num_groups != 0:
        raise ValueError(f"d={d} not divisible by M={num_groups}")
    e = d // num_groups
    g = x.view(*prefix, num_groups, e)
    # Summed outer product: (..., E, E).
    summed = torch.einsum("...mi,...mj->...ij", g, g)
    if e == 1:
        return summed.reshape(*prefix, 1)
    diag = torch.diagonal(summed, dim1=-2, dim2=-1)
    ii, jj = torch.tril_indices(e, e, offset=-1, device=x.device)
    off = summed[..., ii, jj] * math.sqrt(2.0)
    return torch.cat([diag, off], dim=-1)


def feature_map_out_dim(name: str, head_k_dim: int, num_groups: int = 1) -> int:
    """Return q = output dim of psi given raw head_k_dim and (for spd) num_groups."""
    if name in ("positive", "lorentz"):
        return head_k_dim
    if name == "spd_concat":
        if head_k_dim % num_groups != 0:
            raise ValueError(
                f"head_k_dim {head_k_dim} not divisible by num_groups {num_groups}"
            )
        e = head_k_dim // num_groups
        return num_groups * e * (e + 1) // 2
    if name == "spd_full":
        if head_k_dim % num_groups != 0:
            raise ValueError(f"d={head_k_dim} not divisible by M={num_groups}")
        e = head_k_dim // num_groups
        return num_groups * e * e
    if name == "spd_sum":
        if head_k_dim % num_groups != 0:
            raise ValueError(f"d={head_k_dim} not divisible by M={num_groups}")
        e = head_k_dim // num_groups
        return e * e
    if name == "spd_sum_packed":
        if head_k_dim % num_groups != 0:
            raise ValueError(f"d={head_k_dim} not divisible by M={num_groups}")
        e = head_k_dim // num_groups
        return e * (e + 1) // 2
    if name == "kata_quadratic":
        # Implicit qd; never materialized. Bookkeeping: concat-SPD qd = M*E².
        if head_k_dim % num_groups != 0:
            raise ValueError(f"d={head_k_dim} not divisible by M={num_groups}")
        e = head_k_dim // num_groups
        return num_groups * e * e
    if name == "kata_quadratic_sum":
        # Implicit qd; never materialized. Bookkeeping: sum-SPD qd = E².
        if head_k_dim % num_groups != 0:
            raise ValueError(f"d={head_k_dim} not divisible by M={num_groups}")
        e = head_k_dim // num_groups
        return e * e
    raise ValueError(f"unknown feature map {name!r}")


class FeatureMap(nn.Module):
    """Stateless wrapper. nn.Module so it lives inside attention layer."""

    def __init__(
        self, name: str = "positive", num_groups: int = 1, eps: float = 1e-6
    ):
        super().__init__()
        if name not in (
            "positive", "lorentz", "spd_concat", "spd_full",
            "spd_sum", "spd_sum_packed",
            "kata_quadratic", "kata_quadratic_sum",
        ):
            raise ValueError(f"unknown feature map {name!r}")
        self.name = name
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "positive":
            return positive(x, self.eps)
        if self.name == "lorentz":
            return lorentz(x)
        if self.name == "spd_concat":
            return spd_concat(x, self.num_groups)
        if self.name == "spd_full":
            return spd_full(x, self.num_groups)
        if self.name == "spd_sum":
            return spd_sum(x, self.num_groups)
        return spd_sum_packed(x, self.num_groups)

    def extra_repr(self) -> str:
        if self.name in ("spd_concat", "spd_full", "spd_sum", "spd_sum_packed"):
            return f"name={self.name}, num_groups={self.num_groups}"
        return f"name={self.name}"
