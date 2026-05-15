from __future__ import annotations

import torch
from torch import nn

from src.complexes.incidence import sparse_mm, sparse_transpose_mm
from .restrictions import SharedIncidenceRestriction
from .sheaf_laplacian import sheaf_hodge_laplacian_apply


class CellularSheafMessageBlock(nn.Module):
    """Incidence-first cellular sheaf block on dimensions 0..3."""

    def __init__(self, channels: int, *, use_sheaf_laplacian: bool = False, use_geometry_conditioned_restrictions: bool = False):
        super().__init__()
        self.use_sheaf_laplacian = use_sheaf_laplacian
        self.self_maps = nn.ModuleDict({str(k): nn.Linear(channels, channels) for k in range(4)})
        self.up_maps = nn.ModuleDict({str(k): SharedIncidenceRestriction(channels, use_geometry_conditioned_restrictions) for k in range(1, 4)})
        self.down_maps = nn.ModuleDict({str(k): SharedIncidenceRestriction(channels, use_geometry_conditioned_restrictions) for k in range(0, 3)})
        self.lap_maps = nn.ModuleDict({str(k): nn.Linear(channels, channels, bias=False) for k in range(4)})
        self.norms = nn.ModuleDict({str(k): nn.LayerNorm(channels) for k in range(4)})
        self.act = nn.GELU()

    def forward(self, h: dict[int, torch.Tensor], complex) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
        for k in range(4):
            x = h[k]
            y = self.self_maps[str(k)](x)
            if k - 1 in complex.coboundary:
                msg = sparse_mm(complex.coboundary[k - 1], h[k - 1])
                y = y + self.up_maps[str(k)](msg, complex.hodge.get(k))
            if k in complex.coboundary:
                msg = sparse_transpose_mm(complex.coboundary[k], h[k + 1])
                y = y + self.down_maps[str(k)](msg, complex.hodge.get(k))
            if self.use_sheaf_laplacian:
                y = y + self.lap_maps[str(k)](sheaf_hodge_laplacian_apply(complex, k, x))
            out[k] = self.act(self.norms[str(k)](y + x))
        return out
