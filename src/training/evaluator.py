from __future__ import annotations
import time, torch
from src.physics.mhd_diagnostics import mse, relative_l2, per_channel_relative_l2, magnetic_divergence_l2, magnetic_divergence_relative

def parameter_count(model) -> int: return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def evaluate(model, loader, device, magnetic_field_indices=None, spacing=None) -> dict:
    model.eval(); ms=[]; rel=[]; div=[]; divr=[]; per=[]; t0=time.perf_counter(); nb=0
    for batch in loader:
        x=batch['x'].to(device); y=batch['y'].to(device); pred=model(x); nb+=1
        ms.append(float(mse(pred,y).cpu())); rel.append(float(relative_l2(pred,y).cpu()))
        per.append(per_channel_relative_l2(pred,y).detach().cpu())
        div.append(float(magnetic_divergence_l2(pred, magnetic_field_indices, spacing or [1.0]*(pred.ndim-2)).detach().cpu()))
        divr.append(float(magnetic_divergence_relative(pred, magnetic_field_indices, spacing or [1.0]*(pred.ndim-2)).detach().cpu()))
    import numpy as np
    return {'mse': float(np.nanmean(ms)) if ms else float('nan'), 'relative_l2': float(np.nanmean(rel)) if rel else float('nan'),
            'per_channel_relative_l2': torch.stack(per).nanmean(0).tolist() if per else [],
            'magnetic_divergence_l2': float(np.nanmean(div)) if div else float('nan'),
            'magnetic_divergence_relative': float(np.nanmean(divr)) if divr else float('nan'),
            'inference_time_per_batch': (time.perf_counter()-t0)/max(1,nb), 'parameter_count': parameter_count(model)}
