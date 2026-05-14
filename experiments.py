"""One-command experimental suite for Sheaf Neural Operators for MHD.

Run the complete default suite with:
    python experiments.py

Developers may manually set SMOKE_TEST=True below for a local 1-epoch pipeline check; no
command-line flags are required or used.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import traceback

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.orszag_tang import OrszagTangDataset
from src.datasets.well_mhd import WellMHD64Dataset
from src.models import FNO2D, FNO3D, SheafMHDOperator, UNet2D, UNet3D
from src.physics.divergence import periodic_divergence_2d, periodic_divergence_3d
from src.training.evaluator import evaluate
from src.training.rollout import rollout_evaluate
from src.training.trainer import Trainer
from src.utils.config import load_yaml, save_json
from src.utils.logging import setup_logger
from src.utils.plotting import plot_aggregate_bars, plot_divergence_map, plot_error_heatmap, plot_loss_curve, plot_prediction_example, plot_rollout_error
from src.utils.seed import seed_everything
from src.utils.stats import aggregate_metrics, pairwise_comparisons
from src.utils.tables import write_tables

FULL_EXPERIMENT = True
SMOKE_TEST = False


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def apply_smoke(cfg: dict) -> dict:
    if not SMOKE_TEST:
        return cfg
    cfg = json.loads(json.dumps(cfg))
    cfg["seeds"] = [0]
    for d in cfg["datasets"].values():
        d["epochs"] = 1
        d["max_train_samples"] = 4
        d["max_valid_samples"] = 2
        d["max_test_samples"] = 2
        d["hidden_channels"] = min(8, d.get("hidden_channels", 8))
        d["num_layers"] = 1
        d["modes"] = min(4, d.get("modes", 4))
    return cfg


def build_dataset(name: str, dc: dict, split: str):
    cap = dc.get({"train": "max_train_samples", "valid": "max_valid_samples", "test": "max_test_samples"}[split])
    if name == "orszag_tang":
        return OrszagTangDataset(dc["data_root"], split, dc.get("n_input_frames", 1), dc.get("target_frame", 1), cap, normalize=True)
    if name == "wells_mhd64":
        return WellMHD64Dataset(
            dc["data_root"],
            split,
            dc.get("n_input_frames", 1),
            dc.get("n_output_frames", 1),
            cap,
            True,
            None,
            dc.get("magnetic_field_indices"),
        )
    raise KeyError(name)


def build_model(model_name: str, dim: int, in_ch: int, out_ch: int, dc: dict):
    common = dict(in_channels=in_ch, out_channels=out_ch, hidden_channels=dc["hidden_channels"], num_layers=dc["num_layers"], modes=dc["modes"])
    if model_name == "unet":
        return (UNet2D if dim == 2 else UNet3D)(**common)
    if model_name == "fno":
        return (FNO2D if dim == 2 else FNO3D)(**common)
    if model_name == "sheaf_mhd":
        magnetic = dc.get("magnetic_field_indices") or []
        constrained = dim == 2 or (dim == 3 and len(magnetic) >= 3)
        return SheafMHDOperator(
            dim=dim,
            periodic=dc.get("periodic", True),
            dt=dc.get("dt", 1.0),
            spacing=dc.get("spacing"),
            magnetic_field_indices=dc.get("magnetic_field_indices"),
            fluid_field_indices=dc.get("fluid_field_indices"),
            constrained_magnetic_update=constrained,
            backbone_type=dc.get("sheaf_backbone_type", "cnn"),
            **common,
        )
    raise KeyError(model_name)


def run_prediction_figures(model, loader, device, run_dir: Path, magnetic_field_indices: list[int] | None, spacing: list[float] | None, logger) -> None:
    """Generate deterministic per-run diagnostic figures and report failures explicitly."""
    batch = next(iter(loader))
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    with torch.no_grad():
        pred = model(x)
    fig_dir = run_dir / "figures"
    plot_prediction_example(pred, y, fig_dir / "prediction_example.png")
    plot_error_heatmap(pred, y, fig_dir / "error_heatmap.png")
    if magnetic_field_indices:
        if pred.ndim == 4 and len(magnetic_field_indices) >= 2:
            div = periodic_divergence_2d(pred[:, magnetic_field_indices[0]], pred[:, magnetic_field_indices[1]], spacing[0], spacing[1])
            plot_divergence_map(div, fig_dir / "divergence_map.png")
        elif pred.ndim == 5 and len(magnetic_field_indices) >= 3:
            bx, by, bz = [pred[:, idx] for idx in magnetic_field_indices[:3]]
            div = periodic_divergence_3d(bx, by, bz, spacing[0], spacing[1], spacing[2])
            plot_divergence_map(div, fig_dir / "divergence_map.png")
        else:
            logger.warning("Skipping divergence_map for %s because magnetic_field_indices=%s are incompatible with prediction shape %s", run_dir, magnetic_field_indices, tuple(pred.shape))


def write_empty_tables(out_root: Path, reason: str) -> None:
    save_json({"records": [], "reason": reason}, out_root / "aggregate_metrics.json")
    pd.DataFrame().to_csv(out_root / "aggregate_metrics.csv", index=False)
    for name in ["main_results.tex", "divergence_results.tex", "rollout_results.tex", "pairwise_comparisons.tex"]:
        (out_root / "paper_tables" / name).write_text(f"% {reason}\n", encoding="utf-8")
    (out_root / "paper_tables" / "main_results.md").write_text(f"No completed runs. Reason: {reason}\n", encoding="utf-8")


def main() -> None:
    cfg = apply_smoke(load_yaml("configs/default_experiment.yaml"))
    out_root = Path("outputs") / (("smoke_experiment_" if SMOKE_TEST else "full_experiment_") + utc_timestamp())
    for sub in [
        "paper_tables",
        "figures/loss_curves",
        "figures/prediction_examples",
        "figures/divergence_maps",
        "figures/rollout_curves",
        "figures/aggregate_barplots",
        "runs",
    ]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_root / "experiment_log.txt")
    save_json(cfg, out_root / "resolved_experiment_config.json")
    logger.info("Starting Sheaf Neural Operators for MHD suite. FULL_EXPERIMENT=%s SMOKE_TEST=%s", FULL_EXPERIMENT, SMOKE_TEST)
    raw_rows: list[dict] = []
    failures: list[dict] = []
    completed = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    for dataset_name, dc in cfg["datasets"].items():
        root = Path(dc["data_root"])
        if not root.exists():
            msg = f"Dataset root missing: {root}. Data are not downloaded by this codebase."
            logger.error("%s failed dataset check: %s", dataset_name, msg)
            for model_name in cfg["models"]:
                for seed in cfg["seeds"]:
                    row = {"dataset": dataset_name, "model": model_name, "seed": seed, "status": "failed", "error": msg}
                    raw_rows.append(row)
                    failures.append(row)
            continue
        try:
            train_ds = build_dataset(dataset_name, dc, "train")
            valid_ds = build_dataset(dataset_name, dc, "valid")
            test_ds = build_dataset(dataset_name, dc, "test")
            sample = train_ds[0]
            in_ch = sample["x"].shape[0]
            out_ch = sample["y"].shape[0]
            logger.info(
                "%s shapes: x=%s y=%s n_train=%d n_valid=%d n_test=%d",
                dataset_name,
                tuple(sample["x"].shape),
                tuple(sample["y"].shape),
                len(train_ds),
                len(valid_ds),
                len(test_ds),
            )
        except Exception as exc:
            msg = f"Dataset inspection failed: {exc}"
            logger.error("%s", msg)
            for model_name in cfg["models"]:
                for seed in cfg["seeds"]:
                    row = {"dataset": dataset_name, "model": model_name, "seed": seed, "status": "failed", "error": msg}
                    raw_rows.append(row)
                    failures.append(row)
            continue

        for model_name in cfg["models"]:
            for seed in cfg["seeds"]:
                run_dir = out_root / "runs" / dataset_name / model_name / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "figures").mkdir(exist_ok=True)
                run_cfg = {
                    **dc,
                    **cfg.get("training", {}),
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": seed,
                    "lambda_div": dc.get("lambda_div_sheaf", 0.0) if model_name == "sheaf_mhd" else dc.get("lambda_div_baseline", 0.0),
                }
                save_json(run_cfg, run_dir / "config_resolved.json")
                try:
                    seed_everything(seed)
                    gen = torch.Generator().manual_seed(seed)
                    loaders = {
                        split: DataLoader(
                            ds,
                            batch_size=dc["batch_size"],
                            shuffle=(split == "train"),
                            num_workers=cfg["training"].get("num_workers", 0),
                            generator=gen,
                            pin_memory=(device.type == "cuda"),
                        )
                        for split, ds in [("train", train_ds), ("valid", valid_ds), ("test", test_ds)]
                    }
                    model = build_model(model_name, dc["dim"], in_ch, out_ch, dc)
                    trainer = Trainer(model, loaders["train"], loaders["valid"], run_dir, device, run_cfg)
                    trainer.fit()
                    valid_metrics = evaluate(model, loaders["valid"], device, dc.get("magnetic_field_indices"), dc.get("spacing"))
                    test_metrics = evaluate(model, loaders["test"], device, dc.get("magnetic_field_indices"), dc.get("spacing"))
                    rollout = rollout_evaluate(model, loaders["test"], device, cfg["training"].get("rollout_steps", 5), dc.get("magnetic_field_indices"), dc.get("spacing"))
                    save_json(valid_metrics, run_dir / "metrics_valid.json")
                    save_json(test_metrics, run_dir / "metrics_test.json")
                    save_json(rollout, run_dir / "rollout_metrics.json")
                    plot_loss_curve(run_dir / "train_log.csv", run_dir / "figures/loss_curve.png")
                    plot_rollout_error(rollout, run_dir / "figures/rollout_error.png")
                    run_prediction_figures(model, loaders["test"], device, run_dir, dc.get("magnetic_field_indices"), dc.get("spacing"), logger)
                    row = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "seed": seed,
                        "status": "completed",
                        **test_metrics,
                        "final_step_relative_l2": rollout.get("final_step_relative_l2"),
                        "mean_rollout_relative_l2": rollout.get("mean_rollout_relative_l2"),
                    }
                    raw_rows.append(row)
                    completed += 1
                    logger.info("Completed %s/%s seed %s", dataset_name, model_name, seed)
                except Exception as exc:
                    err = {"dataset": dataset_name, "model": model_name, "seed": seed, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
                    save_json(err, run_dir / "failure.json")
                    raw_rows.append(err)
                    failures.append(err)
                    logger.error("Run failed: %s", err)

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out_root / "raw_metrics.csv", index=False)
    numeric_raw = raw[raw.get("status", "") == "completed"].copy() if not raw.empty and "status" in raw else pd.DataFrame()
    if not numeric_raw.empty:
        agg = aggregate_metrics(numeric_raw)
        agg.to_csv(out_root / "aggregate_metrics.csv", index=False)
        save_json({"records": agg.to_dict(orient="records")}, out_root / "aggregate_metrics.json")
        write_tables(agg, out_root / "paper_tables")
        pair = pairwise_comparisons(numeric_raw)
        pair.to_latex(out_root / "paper_tables" / "pairwise_comparisons.tex", index=False)
        for metric in ["relative_l2", "mse", "magnetic_divergence_l2", "mean_rollout_relative_l2"]:
            try:
                plot_aggregate_bars(agg, out_root / f"figures/aggregate_barplots/{metric}.png", metric)
            except ValueError as exc:
                logger.warning("Skipping aggregate plot for %s: %s", metric, exc)
    else:
        write_empty_tables(out_root, "No completed runs; inspect raw_metrics.csv and experiment_summary.json for recorded failures.")

    summary = {
        "outputs_saved": str(out_root),
        "completed_runs": completed,
        "failed_runs": len(failures),
        "failures": failures,
        "aggregate_metrics_csv": str(out_root / "aggregate_metrics.csv"),
        "main_results_tex": str(out_root / "paper_tables/main_results.tex"),
        "figures_dir": str(out_root / "figures"),
    }
    save_json(summary, out_root / "experiment_summary.json")
    logger.info("Final summary: %s", summary)
    print(f"Outputs saved: {out_root}")
    print(f"Completed runs: {completed}")
    print(f"Failed runs: {len(failures)}")
    print(f"Aggregate metrics CSV: {out_root/'aggregate_metrics.csv'}")
    print(f"Main results TeX: {out_root/'paper_tables/main_results.tex'}")
    print(f"Figures directory: {out_root/'figures'}")


if __name__ == "__main__":
    main()
