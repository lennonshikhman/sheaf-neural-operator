from __future__ import annotations
import math, torch
from src.physics.mhd_diagnostics import mse, relative_l2, magnetic_divergence_l2

@torch.no_grad()
def rollout_evaluate(model, loader, device, steps:int=5, magnetic_field_indices=None, spacing=None) -> dict:
    model.eval(); rel_by=[[] for _ in range(steps)]; mse_by=[[] for _ in range(steps)]; div_by=[[] for _ in range(steps)]; unstable=False
    for batch in loader:
        x=batch['x'].to(device); y=batch['y'].to(device); cur=x
        for s in range(steps):
            pred=model(cur)
            if not torch.isfinite(pred).all() or pred.abs().max() > 1e6: unstable=True
            rel_by[s].append(float(relative_l2(pred,y).cpu())); mse_by[s].append(float(mse(pred,y).cpu()))
            div_by[s].append(float(magnetic_divergence_l2(pred, magnetic_field_indices, spacing or [1.0]*(pred.ndim-2)).cpu()))
            cur = pred if pred.shape[1] == cur.shape[1] else torch.cat([cur[:, pred.shape[1]:], pred], dim=1)
        break
    import numpy as np
    rel=[float(np.nanmean(v)) for v in rel_by]; ms=[float(np.nanmean(v)) for v in mse_by]; dv=[float(np.nanmean(v)) for v in div_by]
    return {'rollout_relative_l2_by_step': rel, 'rollout_mse_by_step': ms, 'rollout_divergence_by_step': dv,
            'final_step_relative_l2': rel[-1] if rel else math.nan, 'mean_rollout_relative_l2': float(np.nanmean(rel)) if rel else math.nan,
            'long_horizon_instability_flag': bool(unstable)}
