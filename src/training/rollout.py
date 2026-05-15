from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from src.physics.mhd_diagnostics import mse, relative_l2, magnetic_divergence_l2


def _target_at_step(batch: dict, default_target: torch.Tensor, step: int, device: torch.device) -> torch.Tensor:
    """Return rollout target for ``step`` when a dataset exposes a target sequence.

    Supported optional keys are ``y_sequence`` / ``rollout_y`` with shape
    [B, T, C, ...] or a Python sequence of tensors. If absent, the one-step target
    is reused; this keeps one-step datasets evaluable while fully supporting
    multi-frame datasets when they provide rollout targets.
    """
    for key in ("y_sequence", "rollout_y"):
        if key not in batch:
            continue
        seq = batch[key]
        if torch.is_tensor(seq):
            return seq[:, min(step, seq.shape[1] - 1)].to(device)
        if isinstance(seq, Sequence) and seq:
            return seq[min(step, len(seq) - 1)].to(device)
    return default_target


def _mark_compiled_step_begin() -> None:
    compiler = getattr(torch, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None) if compiler is not None else None
    if marker is not None:
        marker()


def _autoregressive_input(current: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Update autoregressive state while preserving input-channel width.

    Compiled CUDA-graph models may reuse their output storage on the next replay.
    The rollout state must therefore own cloned storage before it is fed back as
    the next input; otherwise torch.compile can raise an overwritten-output error.
    """
    if prediction.shape[1] == current.shape[1]:
        return prediction.detach().clone()
    if prediction.shape[1] > current.shape[1]:
        return prediction[:, -current.shape[1] :].detach().clone()
    return torch.cat([current[:, prediction.shape[1] :], prediction], dim=1).detach().clone()


@torch.no_grad()
def rollout_evaluate(model, loader, device, steps: int = 5, magnetic_field_indices=None, spacing=None) -> dict:
    model.eval()
    rel_by = [[] for _ in range(steps)]
    mse_by = [[] for _ in range(steps)]
    div_by = [[] for _ in range(steps)]
    unstable = False
    batches = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        cur = x
        batches += 1
        for step in range(steps):
            _mark_compiled_step_begin()
            pred = model(cur)
            target = _target_at_step(batch, y, step, device)
            finite = torch.isfinite(pred).all()
            if not finite or pred.detach().abs().max() > 1e6:
                unstable = True
            rel_by[step].append(float(relative_l2(pred, target).cpu()))
            mse_by[step].append(float(mse(pred, target).cpu()))
            div_by[step].append(
                float(magnetic_divergence_l2(pred, magnetic_field_indices, spacing or [1.0] * (pred.ndim - 2)).cpu())
            )
            cur = _autoregressive_input(cur, pred)
    import numpy as np

    rel = [float(np.nanmean(v)) if v else math.nan for v in rel_by]
    ms = [float(np.nanmean(v)) if v else math.nan for v in mse_by]
    dv = [float(np.nanmean(v)) if v else math.nan for v in div_by]
    return {
        "rollout_relative_l2_by_step": rel,
        "rollout_mse_by_step": ms,
        "rollout_divergence_by_step": dv,
        "final_step_relative_l2": rel[-1] if rel else math.nan,
        "mean_rollout_relative_l2": float(np.nanmean(rel)) if rel else math.nan,
        "long_horizon_instability_flag": bool(unstable),
        "num_rollout_batches": batches,
    }
