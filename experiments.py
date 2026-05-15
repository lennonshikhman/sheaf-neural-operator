"""One-command experimental suite for Sheaf Neural Operators for MHD surrogate modeling.

Run the default fast development protocol with:
    python experiments.py

No command-line flags are required or used. Set ``FINAL_RUN = True`` and
``FAST_DEV_RUN = False`` below only for the full paper-scale protocol.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import time
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

FAST_DEV_RUN = True
FINAL_RUN = False
RUN_THE_WELL = True
RUN_SWIGS = True
RUN_CONSTELLARATION = True


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def configure_torch_startup() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def apply_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    if FAST_DEV_RUN == FINAL_RUN:
        raise ValueError("Exactly one of FAST_DEV_RUN or FINAL_RUN must be True.")
    cfg = json.loads(json.dumps(cfg))
    if FAST_DEV_RUN:
        cfg["seeds"] = [0, 1, 2]
        cfg["time_dependent_models"] = ["unet3d", "fno3d", "sheaf_mhd"]
        for dcfg in cfg["datasets"].values():
            dcfg["epochs"] = 5
            dcfg["max_train_samples"] = 256
            dcfg["max_valid_samples"] = 64
            dcfg["max_test_samples"] = 64
    if FINAL_RUN:
        cfg["seeds"] = list(range(10))
        for name, dcfg in cfg["datasets"].items():
            dcfg["epochs"] = 100 if name == "constellaration_equilibrium" else 20
            dcfg["max_train_samples"] = None
            dcfg["max_valid_samples"] = None
            dcfg["max_test_samples"] = None
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
            crop_size=dcfg.get("crop_size", 48),
            use_cache=dcfg.get("use_cache", True),
            rebuild_cache=dcfg.get("rebuild_cache", False),
            cache_root=dcfg.get("cache_root", "datasets/cache"),
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
            optional_fields=dcfg.get("optional_fields"),
            spatial_downsample=dcfg.get("spatial_downsample", dcfg.get("downsample_by", 4)),
            downsample_by=dcfg.get("downsample_by", 4),
            crop_shape=dcfg.get("crop_shape"),
            use_cache=dcfg.get("use_cache", True),
            rebuild_cache=dcfg.get("rebuild_cache", False),
            cache_root=dcfg.get("cache_root", "datasets/cache"),
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
    info = {"name": name, "num_samples": len(ds), "x_shape": list(sample["x"].shape), "y_shape": list(sample["y"].shape), "meta": sample.get("meta", {})}
    for attr in ("field_names", "field_keys", "magnetic_field_indices", "inferred_field_mapping", "feature_dim", "target_dim", "cache_dir", "cache_used", "cache_rebuilt"):
        if hasattr(ds, attr):
            value = getattr(ds, attr)
            info[attr] = str(value) if isinstance(value, Path) else value
    if hasattr(ds, "inspection"):
        info["inspection"] = getattr(ds, "inspection")
    if hasattr(ds, "skipped_files"):
        info["skipped_files"] = getattr(ds, "skipped_files")
    if hasattr(ds, "template_file"):
        info["template_file"] = getattr(ds, "template_file")
    return info


def amp_dtype_from_cfg(cfg: dict[str, Any]) -> torch.dtype:
    return torch.bfloat16 if str(cfg.get("amp_dtype", "bf16")).lower() in {"bf16", "bfloat16", "torch.bfloat16"} else torch.float16


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
    for name in ["main_results.tex", "divergence_results.tex", "rollout_results.tex", "swigs_results.tex", "constellaration_results.tex", "pairwise_comparisons.tex"]:
        (out_root / "paper_tables" / name).write_text(f"% {reason}\n", encoding="utf-8")
    (out_root / "paper_tables" / "main_results.md").write_text(f"No completed runs. Reason: {reason}\n", encoding="utf-8")



def sample_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "meta": [item.get("meta", {}) for item in batch],
    }

def make_loaders(datasets: dict[str, Any], batch_size: int, loader_cfg: dict[str, Any], seed: int, device: torch.device) -> dict[str, DataLoader]:
    gen = torch.Generator().manual_seed(seed)
    num_workers = int(loader_cfg.get("num_workers", 0))
    persistent_workers = bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "generator": gen,
        "pin_memory": bool(loader_cfg.get("pin_memory", device.type == "cuda")),
        "persistent_workers": persistent_workers,
        "collate_fn": sample_collate,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 2))
    return {split: DataLoader(ds, shuffle=(split == "train"), **kwargs) for split, ds in datasets.items()}


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or ("CUDA out of memory" in str(exc) or "cuda runtime error" in str(exc).lower() and "out of memory" in str(exc).lower())


def log_run_start(logger, run_cfg: dict[str, Any]) -> None:
    fields = [
        "dataset", "model", "seed", "batch_size_requested", "batch_size_effective", "use_amp", "amp_dtype",
        "use_compile", "compile_mode", "use_cache", "cache_path", "crop_size", "downsample_by",
        "num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
    ]
    logger.info("Run configuration:\n%s", "\n".join(f"  {key}: {run_cfg.get(key)}" for key in fields))


def run_time_dependent_once(dataset_name: str, model_name: str, seed: int, dcfg: dict[str, Any], cfg: dict[str, Any], datasets: dict[str, Any], in_ch: int, out_ch: int, run_dir: Path, device: torch.device, logger) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    loader_cfg = {**cfg.get("training", {}), **dcfg.get("dataloader", {})}
    requested_bs = int(dcfg["batch_size"])
    batch_size = requested_bs
    last_exc: BaseException | None = None
    while batch_size >= 1:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        run_cfg = {**dcfg, **cfg.get("training", {}), "dataset": dataset_name, "model": model_name, "seed": seed, "lambda_div": dcfg.get("lambda_div", 0.0) if dcfg.get("magnetic_field_indices") else 0.0}
        run_cfg.update({
            "batch_size_requested": requested_bs,
            "batch_size_effective": batch_size,
            "cache_path": str(getattr(datasets["train"], "cache_dir", dcfg.get("cache_root", "datasets/cache"))),
        })
        save_json(run_cfg, run_dir / "config_resolved.json")
        log_run_start(logger, run_cfg)
        try:
            loaders = make_loaders(datasets, batch_size, loader_cfg, seed, device)
            model = build_time_model(model_name, in_ch, out_ch, dcfg)
            trainer = Trainer(model, loaders["train"], loaders["valid"], run_dir, device, run_cfg)
            rows = trainer.fit()
            model_for_eval = trainer.model
            amp_dtype = amp_dtype_from_cfg(run_cfg)
            valid_metrics = evaluate(model_for_eval, loaders["valid"], device, dcfg.get("magnetic_field_indices"), dcfg.get("spacing"), use_amp=run_cfg.get("use_amp", False), amp_dtype=amp_dtype)
            test_metrics = evaluate(model_for_eval, loaders["test"], device, dcfg.get("magnetic_field_indices"), dcfg.get("spacing"), use_amp=run_cfg.get("use_amp", False), amp_dtype=amp_dtype)
            rollout = rollout_evaluate(model_for_eval, loaders["test"], device, cfg["training"].get("rollout_steps", 5), dcfg.get("magnetic_field_indices"), dcfg.get("spacing"))
            save_json(valid_metrics, run_dir / "metrics_valid.json")
            save_json(test_metrics, run_dir / "metrics_test.json")
            save_json(rollout, run_dir / "rollout_metrics.json")
            return {"test_metrics": test_metrics, "rollout": rollout, "effective_batch_size": batch_size}, rows, model_for_eval
        except Exception as exc:
            last_exc = exc
            if device.type == "cuda" and is_cuda_oom(exc) and batch_size > 1:
                logger.warning("CUDA OOM for %s/%s seed %s batch_size=%s; retrying with batch_size=%s", dataset_name, model_name, seed, batch_size, max(1, batch_size // 2))
                batch_size = max(1, batch_size // 2)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def main() -> None:
    start_suite = time.perf_counter()
    configure_torch_startup()
    cfg = apply_protocol(load_yaml("configs/default_experiment.yaml"))
    out_root = Path("outputs") / (("fast_dev_experiment_" if FAST_DEV_RUN else "final_experiment_") + utc_timestamp())
    for sub in ["paper_tables", "figures/loss_curves", "figures/prediction_examples", "figures/divergence_maps", "figures/spectra", "figures/rollout_curves", "figures/aggregate_barplots", "runs"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_root / "experiment_log.txt")
    save_json(cfg, out_root / "resolved_experiment_config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Protocol active: {'FAST_DEV_RUN' if FAST_DEV_RUN else 'FINAL_RUN'}")
    logger.info("Starting suite on device=%s FAST_DEV_RUN=%s FINAL_RUN=%s", device, FAST_DEV_RUN, FINAL_RUN)

    raw_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped_optional: list[dict[str, Any]] = []
    dataset_inspection: dict[str, Any] = {}
    completed = 0
    epoch_times: dict[str, list[float]] = {}
    effective_batch_sizes: list[dict[str, Any]] = []
    cache_summary: dict[str, Any] = {}
    swigs_inferred_mapping = False

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
                    raw_rows.append(row); failures.append(row)
            continue
        try:
            datasets = {split: build_time_dataset(dataset_name, dcfg, split) for split in ("train", "valid", "test")}
            dataset_inspection[dataset_name] = {split: inspect_dataset_sample(ds, f"{dataset_name}/{split}") for split, ds in datasets.items()}
            cache_summary[dataset_name] = {split: {"used": getattr(ds, "cache_used", False), "rebuilt": getattr(ds, "cache_rebuilt", False), "path": str(getattr(ds, "cache_dir", ""))} for split, ds in datasets.items()}
            if dataset_name == "swigs_gorgon":
                train_ds = datasets["train"]
                dcfg["magnetic_field_indices"] = dcfg.get("magnetic_field_indices") or getattr(train_ds, "magnetic_field_indices", None)
                swigs_inferred_mapping = swigs_inferred_mapping or bool(getattr(train_ds, "inferred_field_mapping", False))
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
                    raw_rows.append(row); failures.append(row)
            continue

        for model_name in cfg["time_dependent_models"]:
            for seed in cfg["seeds"]:
                seed_everything(seed)
                run_dir = out_root / "runs" / dataset_name / model_name / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "figures").mkdir(exist_ok=True)
                try:
                    result, rows, model_for_eval = run_time_dependent_once(dataset_name, model_name, seed, dcfg, cfg, datasets, in_ch, out_ch, run_dir, device, logger)
                    test_metrics = result["test_metrics"]
                    rollout = result["rollout"]
                    effective_batch_sizes.append({"dataset": dataset_name, "model": model_name, "seed": seed, "batch_size": result["effective_batch_size"]})
                    epoch_times.setdefault(f"{dataset_name}/{model_name}", []).extend(float(r.get("total_epoch_time", 0.0)) for r in rows)
                    plot_loss_curve(run_dir / "train_log.csv", run_dir / "figures/loss_curve.png")
                    plot_loss_curve(run_dir / "train_log.csv", out_root / "figures/loss_curves" / f"{dataset_name}_{model_name}_seed_{seed}.png")
                    plot_rollout_error(rollout, run_dir / "figures/rollout_error.png")
                    plot_rollout_error(rollout, out_root / "figures/rollout_curves" / f"{dataset_name}_{model_name}_seed_{seed}.png")
                    run_grid_figures(model_for_eval, make_loaders(datasets, result["effective_batch_size"], {**cfg.get("training", {}), **dcfg.get("dataloader", {})}, seed, device)["test"], device, run_dir, out_root / "figures", f"{dataset_name}_{model_name}_seed_{seed}", dcfg.get("magnetic_field_indices"), dcfg.get("spacing"), logger)
                    raw_rows.append({"dataset": dataset_name, "model": model_name, "seed": seed, "track": "time_dependent", "status": "completed", **test_metrics, "final_step_relative_l2": rollout.get("final_step_relative_l2"), "mean_rollout_relative_l2": rollout.get("mean_rollout_relative_l2")})
                    completed += 1
                    logger.info("Completed %s/%s seed %s", dataset_name, model_name, seed)
                except Exception as exc:
                    err = {"dataset": dataset_name, "model": model_name, "seed": seed, "track": "time_dependent", "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
                    save_json(err, run_dir / "failure.json")
                    raw_rows.append(err); failures.append(err)
                    logger.error("Run failed: %s", err)

    if RUN_CONSTELLARATION:
        dataset_name = "constellaration_equilibrium"
        dcfg = dict(cfg["datasets"][dataset_name])
        root = Path(dcfg["data_root"])
        if not root.exists():
            msg = f"Optional ConStellaration subset missing at {root}; skipping only this optional track."
            logger.warning(msg); skipped_optional.append({"dataset": dataset_name, "reason": msg})
        else:
            try:
                datasets = {split: build_equilibrium_dataset(dcfg, split) for split in ("train", "valid", "test")}
                dataset_inspection[dataset_name] = {split: inspect_dataset_sample(ds, f"{dataset_name}/{split}") for split, ds in datasets.items()}
                sample = datasets["train"][0]
                in_dim = sample["x"].numel(); out_dim = sample["y"].numel()
            except Exception as exc:
                msg = f"ConStellaration inspection failed: {exc}"
                logger.error(msg); skipped_optional.append({"dataset": dataset_name, "reason": msg}); datasets = None
            if datasets is not None:
                for model_name in cfg["equilibrium_models"]:
                    for seed in cfg["seeds"]:
                        seed_everything(seed)
                        run_dir = out_root / "runs" / dataset_name / model_name / f"seed_{seed}"
                        run_dir.mkdir(parents=True, exist_ok=True); (run_dir / "figures").mkdir(exist_ok=True)
                        run_cfg = {**dcfg, **cfg.get("training", {}), "dataset": dataset_name, "model": model_name, "seed": seed, "lambda_div": 0.0, "batch_size_requested": dcfg["batch_size"], "batch_size_effective": dcfg["batch_size"], "cache_path": None}
                        save_json(run_cfg, run_dir / "config_resolved.json"); log_run_start(logger, run_cfg)
                        try:
                            requested_bs = int(dcfg["batch_size"])
                            batch_size = requested_bs
                            while True:
                                try:
                                    run_cfg["batch_size_requested"] = requested_bs
                                    run_cfg["batch_size_effective"] = batch_size
                                    save_json(run_cfg, run_dir / "config_resolved.json")
                                    loaders = make_loaders(datasets, batch_size, {**cfg.get("training", {}), **dcfg.get("dataloader", {})}, seed, device)
                                    trainer = Trainer(build_equilibrium_model(model_name, in_dim, out_dim, dcfg), loaders["train"], loaders["valid"], run_dir, device, run_cfg)
                                    rows = trainer.fit(); model_for_eval = trainer.model
                                    test_metrics = evaluate(model_for_eval, loaders["test"], device, None, None, include_spectral=False, use_amp=run_cfg.get("use_amp", False), amp_dtype=amp_dtype_from_cfg(run_cfg))
                                    break
                                except Exception as exc:
                                    if device.type == "cuda" and is_cuda_oom(exc) and batch_size > 1:
                                        logger.warning("CUDA OOM for %s/%s seed %s batch_size=%s; retrying with batch_size=%s", dataset_name, model_name, seed, batch_size, max(1, batch_size // 2))
                                        batch_size = max(1, batch_size // 2)
                                        torch.cuda.empty_cache()
                                        continue
                                    raise
                            save_json(test_metrics, run_dir / "metrics_test.json")
                            plot_loss_curve(run_dir / "train_log.csv", run_dir / "figures/loss_curve.png")
                            raw_rows.append({"dataset": dataset_name, "model": model_name, "seed": seed, "track": "equilibrium", "status": "completed", **test_metrics})
                            effective_batch_sizes.append({"dataset": dataset_name, "model": model_name, "seed": seed, "batch_size": batch_size})
                            epoch_times.setdefault(f"{dataset_name}/{model_name}", []).extend(float(r.get("total_epoch_time", 0.0)) for r in rows)
                            completed += 1
                        except Exception as exc:
                            err = {"dataset": dataset_name, "model": model_name, "seed": seed, "track": "equilibrium", "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
                            save_json(err, run_dir / "failure.json"); raw_rows.append(err); failures.append(err); logger.error("Run failed: %s", err)

    save_json(dataset_inspection, out_root / "dataset_inspection.json")
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out_root / "raw_metrics.csv", index=False)
    numeric_raw = raw[raw.get("status", "") == "completed"].copy() if not raw.empty and "status" in raw else pd.DataFrame()
    if not numeric_raw.empty:
        agg = aggregate_metrics(numeric_raw); agg.to_csv(out_root / "aggregate_metrics.csv", index=False); save_json({"records": agg.to_dict(orient="records")}, out_root / "aggregate_metrics.json")
        write_tables(agg, out_root / "paper_tables")
        pair = pairwise_comparisons(numeric_raw); pair.to_latex(out_root / "paper_tables" / "pairwise_comparisons.tex", index=False)
        for metric in ["relative_l2", "mse", "mae", "magnetic_divergence_l2", "mean_rollout_relative_l2", "spectral_error_3d"]:
            try:
                plot_aggregate_bars(agg, out_root / f"figures/aggregate_barplots/{metric}.png", metric)
            except ValueError as exc:
                logger.warning("Skipping aggregate plot for %s: %s", metric, exc)
    else:
        write_empty_tables(out_root, "No completed runs; inspect raw_metrics.csv and experiment_summary.json for recorded failures.")

    total_runtime = time.perf_counter() - start_suite
    average_epoch_time = {key: sum(vals) / max(1, len(vals)) for key, vals in epoch_times.items()}
    summary = {
        "outputs_saved": str(out_root),
        "protocol": "FAST_DEV_RUN" if FAST_DEV_RUN else "FINAL_RUN",
        "total_runtime_seconds": total_runtime,
        "average_epoch_time_seconds": average_epoch_time,
        "effective_batch_sizes": effective_batch_sizes,
        "cache_summary": cache_summary,
        "completed_runs": completed,
        "failed_runs": len(failures),
        "skipped_optional_runs": len(skipped_optional),
        "skipped_optional": skipped_optional,
        "failures": failures,
        "raw_metrics_csv": str(out_root / "raw_metrics.csv"),
        "aggregate_metrics_csv": str(out_root / "aggregate_metrics.csv"),
        "figures_dir": str(out_root / "figures"),
        "swigs_field_mappings_inferred": swigs_inferred_mapping,
    }
    save_json(summary, out_root / "experiment_summary.json")
    logger.info("Final summary: %s", summary)
    print(f"Total runtime: {total_runtime:.2f}s")
    print(f"Average epoch time per dataset/model: {average_epoch_time}")
    print(f"Effective batch sizes: {effective_batch_sizes}")
    print(f"Cache summary: {cache_summary}")
    print(f"Completed runs: {completed}")
    print(f"Failed runs: {len(failures)}")
    print(f"Skipped optional runs: {len(skipped_optional)}")
    print(f"Raw metrics CSV: {out_root/'raw_metrics.csv'}")
    print(f"Aggregate metrics CSV: {out_root/'aggregate_metrics.csv'}")
    print(f"Figures directory: {out_root/'figures'}")


if __name__ == "__main__":
    main()
