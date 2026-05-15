"""One-command experimental suite for Sheaf Neural Operators for MHD surrogate modeling.

Run the complete default suite with:
    python experiments.py

No command-line flags are required or used. Developers can manually set
``SMOKE_TEST = True`` below for a local 1-epoch pipeline check.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import traceback
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.constellaration import ConStellarationDataset
from src.datasets.swigs_gorgon import SWIGSGorgonDataset
from src.datasets.well_mhd import WellMHD64Dataset
from src.models import FNO3D, MLPRegressor, SheafEquilibriumMLP, SheafMHDOperator, UNet3D
from src.physics.divergence import periodic_divergence_3d
from src.training.evaluator import evaluate
from src.training.rollout import rollout_evaluate
from src.training.trainer import Trainer
from src.utils.config import load_yaml, save_json
from src.utils.logging import setup_logger
from src.utils.plotting import (
    plot_aggregate_bars,
    plot_divergence_map,
    plot_error_heatmap,
    plot_loss_curve,
    plot_prediction_example,
    plot_rollout_error,
    plot_spectra,
)
from src.utils.seed import seed_everything
from src.utils.stats import aggregate_metrics, pairwise_comparisons
from src.utils.tables import write_tables

SMOKE_TEST = False
RUN_THE_WELL = True
RUN_SWIGS = True
RUN_CONSTELLARATION = True


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def apply_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    if not SMOKE_TEST:
        return cfg
    cfg = json.loads(json.dumps(cfg))
    cfg["seeds"] = [0]
    for dcfg in cfg["datasets"].values():
        dcfg["epochs"] = 1
        dcfg["max_train_samples"] = 4
        dcfg["max_valid_samples"] = 2
        dcfg["max_test_samples"] = 2
        if "hidden_channels" in dcfg:
            dcfg["hidden_channels"] = min(16, dcfg["hidden_channels"])
        if "num_layers" in dcfg:
            dcfg["num_layers"] = 1
        if "modes" in dcfg:
            dcfg["modes"] = min(4, dcfg["modes"])
    return cfg


def sample_cap(dcfg: dict[str, Any], split: str) -> int | None:
    return dcfg.get({"train": "max_train_samples", "valid": "max_valid_samples", "test": "max_test_samples"}[split])


def build_time_dataset(name: str, dcfg: dict[str, Any], split: str):
    if name == "wells_mhd64":
        return WellMHD64Dataset(
            dcfg["data_root"],
            split,
            dcfg.get("n_input_frames", 1),
            dcfg.get("n_output_frames", 1),
            sample_cap(dcfg, split),
            True,
            None,
            dcfg.get("magnetic_field_indices"),
        )
    if name == "swigs_gorgon":
        return SWIGSGorgonDataset(
            dcfg["data_root"],
            split,
            dcfg.get("n_input_frames", 1),
            dcfg.get("n_output_frames", 1),
            sample_cap(dcfg, split),
            True,
            dcfg.get("field_mode", "ms_required"),
            required_fields=dcfg.get("required_fields"),
            spatial_downsample=dcfg.get("spatial_downsample", 8),
            crop_shape=dcfg.get("crop_shape"),
        )
    raise KeyError(name)


def build_equilibrium_dataset(dcfg: dict[str, Any], split: str):
    return ConStellarationDataset(dcfg["data_root"], split, sample_cap(dcfg, split), normalize=True)


def build_time_model(model_name: str, in_ch: int, out_ch: int, dcfg: dict[str, Any]):
    common = dict(in_channels=in_ch, out_channels=out_ch, hidden_channels=dcfg["hidden_channels"], num_layers=dcfg["num_layers"])
    if model_name == "unet3d":
        return UNet3D(**common)
    if model_name == "fno3d":
        return FNO3D(**common, modes=dcfg["modes"])
    if model_name == "sheaf_mhd":
        return SheafMHDOperator(
            dim=3,
            modes=dcfg["modes"],
            periodic=dcfg.get("periodic", True),
            dt=dcfg.get("dt", 1.0),
            spacing=dcfg.get("spacing"),
            magnetic_field_indices=dcfg.get("magnetic_field_indices"),
            fluid_field_indices=dcfg.get("fluid_field_indices"),
            constrained_magnetic_update=dcfg.get("constrained_magnetic_update", "direct_with_divergence_features"),
            backbone_type=dcfg.get("sheaf_backbone_type", "cnn"),
            **common,
        )
    raise KeyError(model_name)


def build_equilibrium_model(model_name: str, in_dim: int, out_dim: int, dcfg: dict[str, Any]):
    common = dict(in_channels=in_dim, out_channels=out_dim, hidden_channels=dcfg["hidden_channels"], num_layers=dcfg["num_layers"])
    if model_name == "mlp":
        return MLPRegressor(**common)
    if model_name == "sheaf_equilibrium":
        return SheafEquilibriumMLP(**common)
    raise KeyError(model_name)


def inspect_dataset_sample(ds, name: str) -> dict[str, Any]:
    sample = ds[0]
    info = {
        "name": name,
        "num_samples": len(ds),
        "x_shape": list(sample["x"].shape),
        "y_shape": list(sample["y"].shape),
        "meta": sample.get("meta", {}),
    }
    for attr in ("field_names", "field_keys", "magnetic_field_indices", "inferred_field_mapping", "feature_dim", "target_dim"):
        if hasattr(ds, attr):
            info[attr] = getattr(ds, attr)
    if hasattr(ds, "inspection"):
        info["inspection"] = getattr(ds, "inspection")
    if hasattr(ds, "skipped_files"):
        info["skipped_files"] = getattr(ds, "skipped_files")
    if hasattr(ds, "template_file"):
        info["template_file"] = getattr(ds, "template_file")
    return info


def run_grid_figures(model, loader, device, run_dir: Path, global_figures: Path, label: str, magnetic_field_indices: list[int] | None, spacing: list[float] | None, logger) -> None:
    batch = next(iter(loader))
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    with torch.no_grad():
        pred = model(x)
    fig_dir = run_dir / "figures"
    plot_prediction_example(pred, y, fig_dir / "prediction_example.png")
    plot_prediction_example(pred, y, global_figures / "prediction_examples" / f"{label}.png")
    plot_error_heatmap(pred, y, fig_dir / "error_heatmap.png")
    if pred.ndim == 5:
        plot_spectra(pred, y, fig_dir / "spectra.png")
        plot_spectra(pred, y, global_figures / "spectra" / f"{label}.png")
    if magnetic_field_indices and len(magnetic_field_indices) >= 3 and pred.ndim == 5:
        bx, by, bz = [pred[:, idx] for idx in magnetic_field_indices[:3]]
        div = periodic_divergence_3d(bx, by, bz, *(spacing or [1.0, 1.0, 1.0]))
        plot_divergence_map(div, fig_dir / "divergence_map.png")
        plot_divergence_map(div, global_figures / "divergence_maps" / f"{label}.png")
    elif magnetic_field_indices:
        logger.warning("Divergence map requested but magnetic indices %s are incompatible with prediction shape %s", magnetic_field_indices, tuple(pred.shape))


def write_empty_tables(out_root: Path, reason: str) -> None:
    save_json({"records": [], "reason": reason}, out_root / "aggregate_metrics.json")
    pd.DataFrame().to_csv(out_root / "aggregate_metrics.csv", index=False)
    for name in [
        "main_results.tex",
        "divergence_results.tex",
        "rollout_results.tex",
        "swigs_results.tex",
        "constellaration_results.tex",
        "pairwise_comparisons.tex",
    ]:
        (out_root / "paper_tables" / name).write_text(f"% {reason}\n", encoding="utf-8")
    (out_root / "paper_tables" / "main_results.md").write_text(f"No completed runs. Reason: {reason}\n", encoding="utf-8")


def make_loaders(datasets: dict[str, Any], batch_size: int, num_workers: int, seed: int, device: torch.device) -> dict[str, DataLoader]:
    gen = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            generator=gen,
            pin_memory=(device.type == "cuda"),
        )
        for split, ds in datasets.items()
    }


def main() -> None:
    cfg = apply_smoke(load_yaml("configs/default_experiment.yaml"))
    out_root = Path("outputs") / (("smoke_experiment_" if SMOKE_TEST else "full_experiment_") + utc_timestamp())
    for sub in [
        "paper_tables",
        "figures/loss_curves",
        "figures/prediction_examples",
        "figures/divergence_maps",
        "figures/spectra",
        "figures/rollout_curves",
        "figures/aggregate_barplots",
        "runs",
    ]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_root / "experiment_log.txt")
    save_json(cfg, out_root / "resolved_experiment_config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting suite on device=%s SMOKE_TEST=%s", device, SMOKE_TEST)

    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped_optional: list[dict[str, Any]] = []
    dataset_inspection: dict[str, Any] = {}
    swigs_inferred_mapping = False
    completed = 0

    time_dataset_names = []
    if RUN_THE_WELL:
        time_dataset_names.append("wells_mhd64")
    if RUN_SWIGS:
        time_dataset_names.append("swigs_gorgon")

    for dataset_name in time_dataset_names:
        dcfg = dict(cfg["datasets"][dataset_name])
        if not Path(dcfg["data_root"]).exists():
            msg = f"Dataset root missing: {dcfg['data_root']}. Data are not downloaded by this codebase."
            logger.error("%s skipped/failed: %s", dataset_name, msg)
            for model_name in cfg["time_dependent_models"]:
                for seed in cfg["seeds"]:
                    row = {"dataset": dataset_name, "model": model_name, "seed": seed, "track": "time_dependent", "status": "failed", "error": msg}
                    raw_rows.append(row)
                    failures.append(row)
            continue
        try:
            datasets = {split: build_time_dataset(dataset_name, dcfg, split) for split in ("train", "valid", "test")}
            dataset_inspection[dataset_name] = {split: inspect_dataset_sample(ds, f"{dataset_name}/{split}") for split, ds in datasets.items()}
            if dataset_name == "swigs_gorgon":
                train_ds = datasets["train"]
                dcfg["magnetic_field_indices"] = dcfg.get("magnetic_field_indices") or getattr(train_ds, "magnetic_field_indices", None)
                swigs_inferred_mapping = swigs_inferred_mapping or bool(getattr(train_ds, "inferred_field_mapping", False))
                if swigs_inferred_mapping:
                    logger.warning("SWIGS field mapping was inferred automatically; inspect dataset_inspection.json and .swigs_index.json.")
            sample = datasets["train"][0]
            in_ch = sample["x"].shape[0]
            out_ch = sample["y"].shape[0]
            logger.info("%s inspected: x=%s y=%s", dataset_name, tuple(sample["x"].shape), tuple(sample["y"].shape))
        except Exception as exc:
            msg = f"Dataset inspection failed: {exc}"
            logger.error("%s", msg)
            for model_name in cfg["time_dependent_models"]:
                for seed in cfg["seeds"]:
                    row = {"dataset": dataset_name, "model": model_name, "seed": seed, "track": "time_dependent", "status": "failed", "error": msg}
                    raw_rows.append(row)
                    failures.append(row)
            continue

        for model_name in cfg["time_dependent_models"]:
            for seed in cfg["seeds"]:
                seed_everything(seed)
                run_dir = out_root / "runs" / dataset_name / model_name / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "figures").mkdir(exist_ok=True)
                lambda_div = dcfg.get("lambda_div", 0.0) if dcfg.get("magnetic_field_indices") else 0.0
                run_cfg = {**dcfg, **cfg.get("training", {}), "dataset": dataset_name, "model": model_name, "seed": seed, "lambda_div": lambda_div}
                save_json(run_cfg, run_dir / "config_resolved.json")
                try:
                    loaders = make_loaders(datasets, dcfg["batch_size"], cfg["training"].get("num_workers", 0), seed, device)
                    model = build_time_model(model_name, in_ch, out_ch, dcfg)
                    trainer = Trainer(model, loaders["train"], loaders["valid"], run_dir, device, run_cfg)
                    trainer.fit()
                    valid_metrics = evaluate(model, loaders["valid"], device, dcfg.get("magnetic_field_indices"), dcfg.get("spacing"))
                    test_metrics = evaluate(model, loaders["test"], device, dcfg.get("magnetic_field_indices"), dcfg.get("spacing"))
                    rollout = rollout_evaluate(model, loaders["test"], device, cfg["training"].get("rollout_steps", 5), dcfg.get("magnetic_field_indices"), dcfg.get("spacing"))
                    save_json(valid_metrics, run_dir / "metrics_valid.json")
                    save_json(test_metrics, run_dir / "metrics_test.json")
                    save_json(rollout, run_dir / "rollout_metrics.json")
                    plot_loss_curve(run_dir / "train_log.csv", run_dir / "figures/loss_curve.png")
                    plot_loss_curve(run_dir / "train_log.csv", out_root / "figures/loss_curves" / f"{dataset_name}_{model_name}_seed_{seed}.png")
                    plot_rollout_error(rollout, run_dir / "figures/rollout_error.png")
                    plot_rollout_error(rollout, out_root / "figures/rollout_curves" / f"{dataset_name}_{model_name}_seed_{seed}.png")
                    run_grid_figures(model, loaders["test"], device, run_dir, out_root / "figures", f"{dataset_name}_{model_name}_seed_{seed}", dcfg.get("magnetic_field_indices"), dcfg.get("spacing"), logger)
                    raw_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "seed": seed,
                            "track": "time_dependent",
                            "status": "completed",
                            **test_metrics,
                            "final_step_relative_l2": rollout.get("final_step_relative_l2"),
                            "mean_rollout_relative_l2": rollout.get("mean_rollout_relative_l2"),
                        }
                    )
                    completed += 1
                    logger.info("Completed %s/%s seed %s", dataset_name, model_name, seed)
                except Exception as exc:
                    err = {"dataset": dataset_name, "model": model_name, "seed": seed, "track": "time_dependent", "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
                    save_json(err, run_dir / "failure.json")
                    raw_rows.append(err)
                    failures.append(err)
                    logger.error("Run failed: %s", err)

    if RUN_CONSTELLARATION:
        dataset_name = "constellaration_equilibrium"
        dcfg = dict(cfg["datasets"][dataset_name])
        root = Path(dcfg["data_root"])
        if not root.exists():
            msg = f"Optional ConStellaration subset missing at {root}; skipping only this optional track."
            logger.warning(msg)
            skipped_optional.append({"dataset": dataset_name, "reason": msg})
        else:
            try:
                datasets = {split: build_equilibrium_dataset(dcfg, split) for split in ("train", "valid", "test")}
                dataset_inspection[dataset_name] = {split: inspect_dataset_sample(ds, f"{dataset_name}/{split}") for split, ds in datasets.items()}
                sample = datasets["train"][0]
                in_dim = sample["x"].numel()
                out_dim = sample["y"].numel()
            except Exception as exc:
                msg = f"ConStellaration inspection failed: {exc}"
                logger.error(msg)
                skipped_optional.append({"dataset": dataset_name, "reason": msg})
                datasets = None
            if datasets is not None:
                for model_name in cfg["equilibrium_models"]:
                    for seed in cfg["seeds"]:
                        seed_everything(seed)
                        run_dir = out_root / "runs" / dataset_name / model_name / f"seed_{seed}"
                        run_dir.mkdir(parents=True, exist_ok=True)
                        run_cfg = {**dcfg, **cfg.get("training", {}), "dataset": dataset_name, "model": model_name, "seed": seed, "lambda_div": 0.0, "grad_clip_norm": cfg["training"].get("grad_clip_norm", 1.0)}
                        save_json(run_cfg, run_dir / "config_resolved.json")
                        try:
                            loaders = make_loaders(datasets, dcfg["batch_size"], cfg["training"].get("num_workers", 0), seed, device)
                            model = build_equilibrium_model(model_name, in_dim, out_dim, dcfg)
                            trainer = Trainer(model, loaders["train"], loaders["valid"], run_dir, device, run_cfg)
                            trainer.fit()
                            valid_metrics = evaluate(model, loaders["valid"], device, None, None, include_spectral=False)
                            test_metrics = evaluate(model, loaders["test"], device, None, None, include_spectral=False)
                            save_json(valid_metrics, run_dir / "metrics_valid.json")
                            save_json(test_metrics, run_dir / "metrics_test.json")
                            plot_loss_curve(run_dir / "train_log.csv", run_dir / "figures/loss_curve.png")
                            plot_loss_curve(run_dir / "train_log.csv", out_root / "figures/loss_curves" / f"{dataset_name}_{model_name}_seed_{seed}.png")
                            raw_rows.append({"dataset": dataset_name, "model": model_name, "seed": seed, "track": "equilibrium", "status": "completed", **test_metrics})
                            completed += 1
                            logger.info("Completed %s/%s seed %s", dataset_name, model_name, seed)
                        except Exception as exc:
                            err = {"dataset": dataset_name, "model": model_name, "seed": seed, "track": "equilibrium", "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
                            save_json(err, run_dir / "failure.json")
                            raw_rows.append(err)
                            failures.append(err)
                            logger.error("Run failed: %s", err)

    save_json(dataset_inspection, out_root / "dataset_inspection.json")
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
        for metric in ["relative_l2", "mse", "mae", "magnetic_divergence_l2", "mean_rollout_relative_l2", "spectral_error_3d"]:
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
        "skipped_optional_runs": len(skipped_optional),
        "skipped_optional": skipped_optional,
        "failures": failures,
        "raw_metrics_csv": str(out_root / "raw_metrics.csv"),
        "aggregate_metrics_csv": str(out_root / "aggregate_metrics.csv"),
        "main_results_tex": str(out_root / "paper_tables/main_results.tex"),
        "figures_dir": str(out_root / "figures"),
        "constellaration_skipped": bool(skipped_optional),
        "swigs_field_mappings_inferred": swigs_inferred_mapping,
    }
    save_json(summary, out_root / "experiment_summary.json")
    logger.info("Final summary: %s", summary)
    print(f"Completed runs: {completed}")
    print(f"Failed runs: {len(failures)}")
    print(f"Skipped optional runs: {len(skipped_optional)}")
    print(f"Raw metrics CSV: {out_root/'raw_metrics.csv'}")
    print(f"Aggregate metrics CSV: {out_root/'aggregate_metrics.csv'}")
    print(f"Main results TeX: {out_root/'paper_tables/main_results.tex'}")
    print(f"Figures directory: {out_root/'figures'}")
    if skipped_optional:
        print("WARNING: ConStellaration optional track was skipped; see experiment_summary.json.")
    if swigs_inferred_mapping:
        print("WARNING: SWIGS field mappings were inferred automatically; inspect dataset_inspection.json and .swigs_index.json.")


if __name__ == "__main__":
    main()
