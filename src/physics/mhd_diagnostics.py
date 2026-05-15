"""Metrics for MHD surrogate evaluation."""
from __future__ import annotations

import torch

from .divergence import periodic_divergence_3d
from .spectra import spectral_error_3d

EPS = 1e-12


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - target) / (torch.linalg.vector_norm(target) + EPS)


def per_channel_relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim <= 2:
        return torch.mean(torch.abs(pred - target) / (torch.abs(target) + EPS), dim=0)
    dims = tuple(range(2, pred.ndim))
    num = torch.linalg.vector_norm(pred - target, dim=dims)
    den = torch.linalg.vector_norm(target, dim=dims) + EPS
    return (num / den).mean(dim=0)


def magnetic_divergence_l2(field: torch.Tensor, magnetic_field_indices: list[int] | None, spacing: list[float]) -> torch.Tensor:
    if not magnetic_field_indices:
        return torch.tensor(float("nan"), device=field.device)
    if field.ndim == 5 and len(magnetic_field_indices) >= 3:
        bx, by, bz = [field[:, i] for i in magnetic_field_indices[:3]]
        div = periodic_divergence_3d(bx, by, bz, spacing[0], spacing[1], spacing[2])
    else:
        return torch.tensor(float("nan"), device=field.device)
    return torch.sqrt(torch.mean(div**2))


def magnetic_divergence_relative(field: torch.Tensor, magnetic_field_indices: list[int] | None, spacing: list[float]) -> torch.Tensor:
    div_l2 = magnetic_divergence_l2(field, magnetic_field_indices, spacing)
    if not magnetic_field_indices or torch.isnan(div_l2):
        return div_l2
    b = field[:, magnetic_field_indices]
    return div_l2 / (torch.sqrt(torch.mean(b**2)) + EPS)


def energy_like_quantity(field: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(field**2)


def energy_drift(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (energy_like_quantity(pred) - energy_like_quantity(target)).abs() / (energy_like_quantity(target).abs() + EPS)


def spectral_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 5:
        return spectral_error_3d(pred, target)
    return torch.tensor(float("nan"), device=pred.device)


def rollout_relative_l2(preds: list[torch.Tensor], targets: list[torch.Tensor]) -> list[float]:
    return [float(relative_l2(p, t).detach().cpu()) for p, t in zip(preds, targets, strict=False)]
