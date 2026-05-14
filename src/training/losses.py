from __future__ import annotations
import torch
from src.physics.mhd_diagnostics import relative_l2, magnetic_divergence_l2

def mhd_loss(pred: torch.Tensor, target: torch.Tensor, lambda_rel: float=0.1, lambda_div: float=0.0,
             magnetic_field_indices: list[int]|None=None, spacing: list[float]|None=None) -> torch.Tensor:
    loss=torch.mean((pred-target)**2) + lambda_rel*relative_l2(pred,target)
    if lambda_div and magnetic_field_indices:
        div=magnetic_divergence_l2(pred, magnetic_field_indices, spacing or [1.0]*(pred.ndim-2))
        if not torch.isnan(div): loss = loss + lambda_div * div**2
    return loss
