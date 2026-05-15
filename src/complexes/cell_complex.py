from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import torch


@dataclass
class CellComplex:
    """Arbitrary oriented finite cell complex with sparse incidence/coboundary maps.

    ``boundary[k]`` is ∂_k: C_k -> C_{k-1}. ``coboundary[k]`` is
    d_k = ∂_{k+1}^T: C^k -> C^{k+1}.  Geometry and Hodge data are diagonal
    approximations by dimension; cubical complexes use primal measures and simple
    dual-volume approximations where exact duals are not constructed.
    """

    cells_by_dim: dict[int, Any]
    boundary: dict[int, torch.Tensor]
    geometry: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    hodge: dict[int, torch.Tensor] = field(default_factory=dict)
    boundary_tags: dict[int, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.boundary = {int(k): v.coalesce() for k, v in self.boundary.items()}
        self.coboundary = {k - 1: v.transpose(0, 1).coalesce() for k, v in self.boundary.items()}

    def num_cells(self, dim: int) -> int:
        cells = self.cells_by_dim.get(dim)
        if isinstance(cells, int):
            return cells
        if hasattr(cells, "__len__"):
            return len(cells)
        if dim in self.hodge:
            return int(self.hodge[dim].numel())
        return 0

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "CellComplex":
        dtype = dtype or next(iter(self.boundary.values())).dtype
        boundary = {k: v.to(device=device, dtype=dtype).coalesce() for k, v in self.boundary.items()}
        geometry = {d: {name: val.to(device=device, dtype=dtype) for name, val in geom.items()} for d, geom in self.geometry.items()}
        hodge = {d: val.to(device=device, dtype=dtype) for d, val in self.hodge.items()}
        tags = {d: val.to(device=device) for d, val in self.boundary_tags.items()}
        return CellComplex(self.cells_by_dim, boundary, geometry, hodge, tags, dict(self.metadata))

    def d(self, k: int) -> torch.Tensor:
        return self.coboundary[k]

    def boundary_matrix(self, k: int) -> torch.Tensor:
        return self.boundary[k]

    def incidence_summary(self) -> dict[str, Any]:
        return {
            "complex_type": self.metadata.get("complex_type", "arbitrary"),
            "grid_shape": list(self.metadata.get("grid_shape", [])),
            "cells_by_dim": {str(k): self.num_cells(k) for k in range(4)},
            "boundary_nnz": {f"partial_{k}": int(v._nnz()) for k, v in self.boundary.items()},
            "coboundary_nnz": {f"d_{k}": int(v._nnz()) for k, v in self.coboundary.items()},
        }

    def max_d_next_d_error(self, k: int) -> float:
        """Return max absolute entry of d_{k+1} d_k without dense materialization.

        The check is intentionally performed on CPU in float32.  CUDA sparse
        addmm does not implement every AMP dtype (notably bfloat16), and dense
        materialization is prohibitive for realistic 3-D grids.
        """
        if k not in self.coboundary or k + 1 not in self.coboundary:
            return 0.0
        left = self.coboundary[k + 1].detach().to(device="cpu", dtype=torch.float32).coalesce()
        right = self.coboundary[k].detach().to(device="cpu", dtype=torch.float32).coalesce()
        with warnings.catch_warnings(), torch.amp.autocast(device_type="cpu", enabled=False):
            warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*")
            prod = torch.sparse.mm(left, right).coalesce()
        if prod._nnz() == 0:
            return 0.0
        return float(prod.values().abs().max().item())
