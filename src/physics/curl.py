"""Curl operators used by the Sheaf Neural Operator constrained heads."""
from __future__ import annotations
import torch
from .finite_differences import periodic_diff_x_2d, periodic_diff_y_2d, periodic_diff_x_3d, periodic_diff_y_3d, periodic_diff_z_3d


def periodic_curl_scalar_2d(a: torch.Tensor, dy: float, dz: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (d a / dz, -d a / dy) for scalar EMF a[..., H, W]."""
    delta_by = periodic_diff_y_2d(a, dz)
    delta_bz = -periodic_diff_x_2d(a, dy)
    return delta_by, delta_bz


def curl_vector_potential_3d(ax: torch.Tensor, ay: torch.Tensor, az: torch.Tensor, dx: float, dy: float, dz: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bx = periodic_diff_y_3d(az, dy) - periodic_diff_z_3d(ay, dz)
    by = periodic_diff_z_3d(ax, dz) - periodic_diff_x_3d(az, dx)
    bz = periodic_diff_x_3d(ay, dx) - periodic_diff_y_3d(ax, dy)
    return bx, by, bz
