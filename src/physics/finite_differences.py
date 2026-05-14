"""Periodic centered finite differences on structured grids."""
from __future__ import annotations
import torch


def _centered(u: torch.Tensor, dim: int, h: float) -> torch.Tensor:
    return (torch.roll(u, shifts=-1, dims=dim) - torch.roll(u, shifts=1, dims=dim)) / (2.0 * h)


def periodic_diff_x_2d(u: torch.Tensor, dx: float) -> torch.Tensor:
    """Differentiate along axis 0 / height for tensors [..., H, W]."""
    return _centered(u, -2, dx)


def periodic_diff_y_2d(u: torch.Tensor, dy: float) -> torch.Tensor:
    """Differentiate along axis 1 / width for tensors [..., H, W]."""
    return _centered(u, -1, dy)


def periodic_diff_x_3d(u: torch.Tensor, dx: float) -> torch.Tensor:
    return _centered(u, -3, dx)


def periodic_diff_y_3d(u: torch.Tensor, dy: float) -> torch.Tensor:
    return _centered(u, -2, dy)


def periodic_diff_z_3d(u: torch.Tensor, dz: float) -> torch.Tensor:
    return _centered(u, -1, dz)
