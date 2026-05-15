"""Curl operators used by vector-potential magnetic updates."""
from __future__ import annotations

import torch

from .finite_differences import periodic_diff_x_3d, periodic_diff_y_3d, periodic_diff_z_3d


def curl_vector_potential_3d(ax: torch.Tensor, ay: torch.Tensor, az: torch.Tensor, dx: float, dy: float, dz: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bx = periodic_diff_y_3d(az, dy) - periodic_diff_z_3d(ay, dz)
    by = periodic_diff_z_3d(ax, dz) - periodic_diff_x_3d(az, dx)
    bz = periodic_diff_x_3d(ay, dx) - periodic_diff_y_3d(ax, dy)
    return bx, by, bz
