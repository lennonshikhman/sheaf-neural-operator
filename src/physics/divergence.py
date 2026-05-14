"""Magnetic divergence diagnostics."""
from __future__ import annotations
import torch
from .finite_differences import periodic_diff_x_2d, periodic_diff_y_2d, periodic_diff_x_3d, periodic_diff_y_3d, periodic_diff_z_3d


def periodic_divergence_2d(by: torch.Tensor, bz: torch.Tensor, dy: float, dz: float) -> torch.Tensor:
    """div B = d By / dy + d Bz / dz for Orszag-Tang axes [y,z]."""
    return periodic_diff_x_2d(by, dy) + periodic_diff_y_2d(bz, dz)


def periodic_divergence_3d(bx: torch.Tensor, by: torch.Tensor, bz: torch.Tensor, dx: float, dy: float, dz: float) -> torch.Tensor:
    return periodic_diff_x_3d(bx, dx) + periodic_diff_y_3d(by, dy) + periodic_diff_z_3d(bz, dz)
