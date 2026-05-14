"""Metrics for MHD surrogate evaluation."""
from __future__ import annotations
import torch
from .divergence import periodic_divergence_2d, periodic_divergence_3d

EPS = 1e-12

def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)

def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - target) / (torch.linalg.vector_norm(target) + EPS)

def per_channel_relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(2, pred.ndim))
    num = torch.linalg.vector_norm(pred - target, dim=dims)
    den = torch.linalg.vector_norm(target, dim=dims) + EPS
    return (num / den).mean(dim=0)

def magnetic_divergence_l2(field: torch.Tensor, magnetic_field_indices: list[int] | None, spacing: list[float]) -> torch.Tensor:
    if not magnetic_field_indices:
        return torch.tensor(float('nan'), device=field.device)
    if field.ndim == 4 and len(magnetic_field_indices) >= 2:
        by = field[:, magnetic_field_indices[0]]; bz = field[:, magnetic_field_indices[1]]
        div = periodic_divergence_2d(by, bz, spacing[0], spacing[1])
    elif field.ndim == 5 and len(magnetic_field_indices) >= 3:
        bx, by, bz = [field[:, i] for i in magnetic_field_indices[:3]]
        div = periodic_divergence_3d(bx, by, bz, spacing[0], spacing[1], spacing[2])
    else:
        return torch.tensor(float('nan'), device=field.device)
    return torch.sqrt(torch.mean(div ** 2))

def magnetic_divergence_relative(field: torch.Tensor, magnetic_field_indices: list[int] | None, spacing: list[float]) -> torch.Tensor:
    div_l2 = magnetic_divergence_l2(field, magnetic_field_indices, spacing)
    if not magnetic_field_indices or torch.isnan(div_l2):
        return div_l2
    b = field[:, magnetic_field_indices]
    return div_l2 / (torch.sqrt(torch.mean(b ** 2)) + EPS)

def energy_like_quantity(field: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(field ** 2)

def energy_drift(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (energy_like_quantity(pred) - energy_like_quantity(target)).abs() / (energy_like_quantity(target).abs() + EPS)

def spectrum_2d(field: torch.Tensor) -> torch.Tensor:
    fft = torch.fft.rfft2(field, dim=(-2, -1))
    return torch.mean(torch.abs(fft) ** 2, dim=tuple(range(field.ndim - 2)))

def rollout_relative_l2(preds: list[torch.Tensor], targets: list[torch.Tensor]) -> list[float]:
    return [float(relative_l2(p, t).detach().cpu()) for p, t in zip(preds, targets)]
