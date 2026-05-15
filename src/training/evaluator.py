from __future__ import annotations

import time
from contextlib import nullcontext

import torch

from src.physics.mhd_diagnostics import (
    energy_drift,
    mae,
    magnetic_divergence_l2,
    magnetic_divergence_relative,
    mse,
    per_channel_relative_l2,
    relative_l2,
    spectral_error,
)


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, loader, device, magnetic_field_indices=None, spacing=None, include_spectral: bool = True, use_amp: bool = False, amp_dtype=torch.bfloat16) -> dict:
    model.eval()
    ms: list[float] = []
    mas: list[float] = []
    rel: list[float] = []
    div: list[float] = []
    divr: list[float] = []
    edrift: list[float] = []
    spec: list[float] = []
    per = []
    t0 = time.perf_counter()
    nb = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp and device.type == "cuda" else nullcontext()
        with ctx:
            pred = model(x)
        nb += 1
        ms.append(float(mse(pred, y).cpu()))
        mas.append(float(mae(pred, y).cpu()))
        rel.append(float(relative_l2(pred, y).cpu()))
        per.append(per_channel_relative_l2(pred, y).detach().cpu())
        edrift.append(float(energy_drift(pred, y).cpu()))
        if pred.ndim >= 4:
            div.append(float(magnetic_divergence_l2(pred, magnetic_field_indices, spacing or [1.0] * (pred.ndim - 2)).detach().cpu()))
            divr.append(float(magnetic_divergence_relative(pred, magnetic_field_indices, spacing or [1.0] * (pred.ndim - 2)).detach().cpu()))
            if include_spectral and pred.ndim == 5:
                with torch.autocast(device_type=device.type, enabled=False):
                    spec.append(float(spectral_error(pred.float(), y.float()).detach().cpu()))
    import numpy as np

    return {
        "mse": float(np.nanmean(ms)) if ms else float("nan"),
        "mae": float(np.nanmean(mas)) if mas else float("nan"),
        "relative_l2": float(np.nanmean(rel)) if rel else float("nan"),
        "per_channel_relative_l2": torch.stack(per).nanmean(0).tolist() if per else [],
        "per_target_relative_error": torch.stack(per).nanmean(0).tolist() if per and per[0].ndim == 1 else [],
        "magnetic_divergence_l2": float(np.nanmean(div)) if div else float("nan"),
        "magnetic_divergence_relative": float(np.nanmean(divr)) if divr else float("nan"),
        "energy_like_drift": float(np.nanmean(edrift)) if edrift else float("nan"),
        "spectral_error_3d": float(np.nanmean(spec)) if spec else float("nan"),
        "inference_time_per_batch": (time.perf_counter() - t0) / max(1, nb),
        "parameter_count": parameter_count(model),
    }
