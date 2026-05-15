from __future__ import annotations

from pathlib import Path

import pandas as pd

DISPLAY_NAMES = {"sheaf_mhd": "Sheaf Neural Operator", "unet3d": "3D U-Net", "fno3d": "3D FNO", "mlp": "MLP", "sheaf_equilibrium": "Sheaf Equilibrium MLP"}


def _fmt(row) -> str:
    return f"{row['mean']:.4g} [{row['ci95_low']:.4g}, {row['ci95_high']:.4g}]"


def _wide_table(agg: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for (ds, model), g in agg.groupby(["dataset", "model"]):
        r = {"dataset": ds, "model": DISPLAY_NAMES.get(model, model)}
        for metric in metrics:
            gm = g[g.metric == metric]
            r[metric] = _fmt(gm.iloc[0]) if not gm.empty else "not_applicable"
        rows.append(r)
    return pd.DataFrame(rows)


def write_tables(agg: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    main_metrics = ["relative_l2", "mse", "mae", "magnetic_divergence_l2", "mean_rollout_relative_l2", "spectral_error_3d", "parameter_count", "inference_time_per_batch"]
    main = _wide_table(agg, main_metrics)
    main.to_markdown(out / "main_results.md", index=False)
    main.to_latex(out / "main_results.tex", index=False, escape=False)
    table_specs = {
        "divergence_results.tex": ["magnetic_divergence_l2", "magnetic_divergence_relative"],
        "rollout_results.tex": ["mean_rollout_relative_l2", "final_step_relative_l2"],
        "swigs_results.tex": ["relative_l2", "mse", "magnetic_divergence_l2", "spectral_error_3d"],
        "constellaration_results.tex": ["relative_l2", "mse", "mae"],
    }
    for filename, metrics in table_specs.items():
        sub = _wide_table(agg, metrics)
        if filename == "swigs_results.tex":
            sub = sub[sub.dataset == "swigs_gorgon"]
        if filename == "constellaration_results.tex":
            sub = sub[sub.dataset == "constellaration_equilibrium"]
        sub.to_latex(out / filename, index=False, escape=False)
