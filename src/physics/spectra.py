"""Spectral diagnostics for 3D grid surrogates."""
from __future__ import annotations

import torch

EPS = 1e-12


def isotropic_power_spectrum_3d(field: torch.Tensor, bins: int | None = None) -> torch.Tensor:
    """Compute a simple isotropic 3D power spectrum averaged over batch/channels.

    ``field`` is expected to have shape [B,C,X,Y,Z] or [C,X,Y,Z].
    """
    if field.ndim == 4:
        field = field.unsqueeze(0)
    if field.ndim != 5:
        raise ValueError(f"Expected 4D/5D field for 3D spectrum, got {tuple(field.shape)}")
    _, _, nx, ny, nz = field.shape
    with torch.autocast(device_type=field.device.type, enabled=False):
        field = field.float()
        fft = torch.fft.fftn(field, dim=(-3, -2, -1))
        power = torch.abs(fft) ** 2
        kx = torch.fft.fftfreq(nx, device=field.device) * nx
        ky = torch.fft.fftfreq(ny, device=field.device) * ny
        kz = torch.fft.fftfreq(nz, device=field.device) * nz
        kk = torch.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
        max_bin = int(torch.ceil(kk.max()).item()) + 1
        nbins = bins or max_bin
        edges = torch.linspace(0, kk.max() + EPS, nbins + 1, device=field.device)
        spectrum = []
        for i in range(nbins):
            mask = (kk >= edges[i]) & (kk < edges[i + 1])
            spectrum.append(power[..., mask].mean() if mask.any() else torch.tensor(float("nan"), device=field.device))
        return torch.stack(spectrum)


def spectral_error_3d(pred: torch.Tensor, target: torch.Tensor, bins: int | None = None) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    ps_pred = isotropic_power_spectrum_3d(pred, bins)
    ps_target = isotropic_power_spectrum_3d(target, bins)
    mask = torch.isfinite(ps_pred) & torch.isfinite(ps_target)
    if not mask.any():
        return torch.tensor(float("nan"), device=pred.device)
    return torch.linalg.vector_norm(ps_pred[mask] - ps_target[mask]) / (torch.linalg.vector_norm(ps_target[mask]) + EPS)
