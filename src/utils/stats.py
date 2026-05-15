from __future__ import annotations

import importlib
import importlib.util
import math

import numpy as np
import pandas as pd

scipy_stats = importlib.import_module("scipy.stats") if importlib.util.find_spec("scipy") is not None else None


def summarize(values, bootstrap: int = 10000, seed: int = 123):
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = len(x)
    if n == 0:
        return dict(mean=np.nan, std=np.nan, se=np.nan, ci95_low=np.nan, ci95_high=np.nan, boot_ci95_low=np.nan, boot_ci95_high=np.nan, median=np.nan, iqr=np.nan, n=0)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n else np.nan
    tcrit = float(scipy_stats.t.ppf(0.975, n - 1)) if scipy_stats and n > 1 else 1.96
    rng = np.random.default_rng(seed)
    reps = min(bootstrap, 10000)
    boots = rng.choice(x, size=(reps, n), replace=True).mean(axis=1)
    q75, q25 = np.percentile(x, [75, 25])
    return dict(
        mean=mean,
        std=std,
        se=se,
        ci95_low=mean - tcrit * se,
        ci95_high=mean + tcrit * se,
        boot_ci95_low=float(np.percentile(boots, 2.5)),
        boot_ci95_high=float(np.percentile(boots, 97.5)),
        median=float(np.median(x)),
        iqr=float(q75 - q25),
        n=n,
    )


def aggregate_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in raw.columns if c not in {"dataset", "model", "seed", "status", "error", "track"} and pd.api.types.is_numeric_dtype(raw[c])]
    rows = []
    for keys, g in raw.groupby(["dataset", "model"]):
        ds, model = keys
        for metric in metrics:
            rows.append({"dataset": ds, "model": model, "metric": metric, **summarize(g[metric].to_numpy())})
    return pd.DataFrame(rows)


def _comparison_pairs(dataset: str) -> list[tuple[str, str]]:
    if dataset == "constellaration_equilibrium":
        return [("sheaf_equilibrium", "mlp")]
    return [("sheaf_mhd", "unet3d"), ("sheaf_mhd", "fno3d")]


def pairwise_comparisons(raw: pd.DataFrame, metrics=None) -> pd.DataFrame:
    metrics = metrics or ["relative_l2", "mse", "mae", "magnetic_divergence_l2", "mean_rollout_relative_l2", "spectral_error_3d"]
    rows = []
    for ds, gds in raw.groupby("dataset"):
        for challenger, baseline in _comparison_pairs(ds):
            for metric in metrics:
                if metric not in gds:
                    continue
                left = gds[gds.model == challenger][["seed", metric]]
                right = gds[gds.model == baseline][["seed", metric]]
                s = left.merge(right, on="seed", suffixes=("_challenger", "_baseline"))
                if s.empty:
                    continue
                diff = s[f"{metric}_challenger"] - s[f"{metric}_baseline"]
                mean_base = s[f"{metric}_baseline"].mean()
                summ = summarize(diff)
                p = float(scipy_stats.ttest_rel(s[f"{metric}_challenger"], s[f"{metric}_baseline"], nan_policy="omit").pvalue) if scipy_stats and len(s) > 1 else np.nan
                rows.append(
                    {
                        "dataset": ds,
                        "comparison": f"{challenger}_vs_{baseline}",
                        "metric": metric,
                        "mean_difference": summ["mean"],
                        "relative_percent_improvement": float(-100 * summ["mean"] / mean_base) if mean_base else np.nan,
                        "p_value": p,
                        **{f"diff_{k}": v for k, v in summ.items()},
                    }
                )
    return pd.DataFrame(rows)
