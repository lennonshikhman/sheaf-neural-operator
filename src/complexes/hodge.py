from __future__ import annotations

import torch


def diagonal_hodge_weights(num_cells: dict[int, int], spacing: tuple[float, float, float]) -> dict[int, torch.Tensor]:
    """Simple diagonal geometric Hodge weights for cubical grids.

    These are primal measure approximations (vertex dual volumes are approximated
    by the cell volume, edges by length, faces by area, cells by volume).  Exact
    circumcentric duals are intentionally not constructed for rectangular grids.
    """
    dx, dy, dz = spacing
    return {
        0: torch.full((num_cells.get(0, 0),), dx * dy * dz),
        1: torch.ones(num_cells.get(1, 0)),
        2: torch.ones(num_cells.get(2, 0)),
        3: torch.full((num_cells.get(3, 0),), dx * dy * dz),
    }
