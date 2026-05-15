from __future__ import annotations

import torch

from src.complexes.incidence import sparse_mm, sparse_transpose_mm


def sheaf_hodge_laplacian_apply(complex, k: int, x: torch.Tensor) -> torch.Tensor:
    """Apply unsigned cellular Hodge Laplacian stencil to k-cochains."""
    out = torch.zeros_like(x)
    if k in complex.coboundary:
        d = complex.coboundary[k]
        out = out + sparse_transpose_mm(d, sparse_mm(d, x))
    if k - 1 in complex.coboundary:
        dm = complex.coboundary[k - 1]
        out = out + sparse_mm(dm, sparse_transpose_mm(dm, x))
    return out
