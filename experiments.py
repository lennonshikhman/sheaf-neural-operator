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
import os
import time
import traceback
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.constellaration import ConStellarationDataset
from src.datasets.swigs_gorgon import SWIGSGorgonDataset
from src.datasets.well_mhd import WellMHD64Dataset
from src.models import FNO3D, MLPRegressor, SheafEquilibriumMLP, SheafMHDOperator, CellularMHDSheafNeuralOperator, UNet3D
from src.physics.divergence import periodic_divergence_3d
from src.training.evaluator import evaluate
from src.training.rollout import rollout_evaluate
from src.training.trainer import Trainer
from src.training.losses import mhd_loss
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

FAST_DEV_RUN = False
FINAL_RUN = True
RUN_THE_WELL = True
RUN_SWIGS = True
RUN_CONSTELLARATION = True
RUN_VALIDATION = False


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def protocol_output_prefix() -> str:
    return "fast_dev_experiment_" if FAST_DEV_RUN else "final_experiment_"


def select_output_root() -> tuple[Path, bool]:
    """Return the output root for this run and whether it is resuming an existing suite.

    Experiments are long-running, so by default a restarted ``python experiments.py``
    resumes the newest output directory for the active protocol instead of creating
    a fresh timestamped directory. Set ``EXPERIMENTS_FORCE_NEW=1`` to opt into a
    new directory when an intentionally independent repeat is desired.
    """
    prefix = protocol_output_prefix()
    outputs = Path("outputs")
    force_new = os.environ.get("EXPERIMENTS_FORCE_NEW", "").lower() in {"1", "true", "yes"}
    if not force_new and outputs.exists():
        candidates = sorted(
            (path for path in outputs.glob(f"{prefix}*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0], True
    return outputs / (prefix + utc_timestamp()), False


def read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def epoch_times_from_train_log(run_dir: Path) -> list[float]:
    train_log = run_dir / "train_log.csv"
    if not train_log.exists():
        return []
    try:
        frame = pd.read_csv(train_log)
    except Exception:
        return []
    if "total_epoch_time" not in frame:
        return []
    return [float(value) for value in frame["total_epoch_time"].dropna()]


def completion_marker_path(run_dir: Path) -> Path:
    return run_dir / "complete_run.json"


def save_completion_marker(run_dir: Path, row: dict[str, Any], effective_batch_size: int | None) -> None:
    save_json(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "row": row,
            "effective_batch_size": effective_batch_size,
            "required_files": sorted(path.name for path in run_dir.iterdir() if path.is_file()),
        },
        completion_marker_path(run_dir),
    )


def completed_run_from_disk(run_dir: Path, dataset_name: str, model_name: str, seed: int, track: str) -> tuple[dict[str, Any], int | None, list[float]] | None:
    """Load a completed experiment row if all required result files are present.

    Newer runs write ``complete_run.json`` after all metrics are saved. For
    compatibility with already-completed runs from older versions, this also
    accepts the complete set of metric files as proof of completion. Partial
    directories from crashes are intentionally ignored and rerun.
    """
    marker = read_json_if_present(completion_marker_path(run_dir))
    if marker and marker.get("status") == "completed" and isinstance(marker.get("row"), dict):
        row = dict(marker["row"])
        effective_batch_size = marker.get("effective_batch_size")
        return row, int(effective_batch_size) if effective_batch_size is not None else None, epoch_times_from_train_log(run_dir)

    config = read_json_if_present(run_dir / "config_resolved.json") or {}
    test_metrics = read_json_if_present(run_dir / "metrics_test.json")
    if not test_metrics:
        return None

    if track == "time_dependent":
        rollout = read_json_if_present(run_dir / "rollout_metrics.json")
        if not rollout:
            return None
        row = {
            "dataset": dataset_name,
            "model": model_name,
            "seed": seed,
            "track": track,
            "status": "completed",
            **test_metrics,
            "final_step_relative_l2": rollout.get("final_step_relative_l2"),
            "mean_rollout_relative_l2": rollout.get("mean_rollout_relative_l2"),
        }
    elif track == "equilibrium":
        row = {"dataset": dataset_name, "model": model_name, "seed": seed, "track": track, "status": "completed", **test_metrics}
    else:
        return None

    effective_batch_size = config.get("batch_size_effective")
    return row, int(effective_batch_size) if effective_batch_size is not None else None, epoch_times_from_train_log(run_dir)


def record_completed_skip(
    run_dir: Path,
    dataset_name: str,
    model_name: str,
    seed: int,
    track: str,
    raw_rows: list[dict[str, Any]],
    effective_batch_sizes: list[dict[str, Any]],
    epoch_times: dict[str, list[float]],
    logger,
) -> bool:
    completed = completed_run_from_disk(run_dir, dataset_name, model_name, seed, track)
    if completed is None:
        return False
    row, effective_batch_size, run_epoch_times = completed
    raw_rows.append(row)
    if effective_batch_size is not None:
        effective_batch_sizes.append({"dataset": dataset_name, "model": model_name, "seed": seed, "batch_size": effective_batch_size})
    if run_epoch_times:
        epoch_times.setdefault(f"{dataset_name}/{model_name}", []).extend(run_epoch_times)
    logger.info("Skipping completed %s/%s seed %s; loaded complete results from %s", dataset_name, model_name, seed, run_dir)
    return True


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
    if model_name in {"sheaf_mhd", "cellular_mhd_sno"}:
        return CellularMHDSheafNeuralOperator(
            dim=3,
            modes=dcfg["modes"],
            periodic=dcfg.get("periodic", True),
            dt=dcfg.get("dt", 1.0),
            spacing=dcfg.get("spacing"),
            magnetic_field_indices=dcfg.get("magnetic_field_indices"),
            fluid_field_indices=dcfg.get("fluid_field_indices"),
            max_internal_cells=dcfg.get("max_internal_cells", 32768),
            use_sheaf_laplacian=dcfg.get("use_sheaf_laplacian", False),
            use_geometry_conditioned_restrictions=dcfg.get("use_geometry_conditioned_restrictions", False),
            use_geometric_hodge=dcfg.get("use_geometric_hodge", True),
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



def resolve_compile_request(model_name: str, dcfg: dict[str, Any], training_cfg: dict[str, Any]) -> bool:
    if "use_compile" in dcfg:
        return bool(dcfg["use_compile"])
    defaults = training_cfg.get("model_compile_defaults", {})
    if model_name == "cellular_mhd_sno" and "sheaf_mhd" in defaults:
        return bool(defaults["sheaf_mhd"])
    if model_name in defaults:
        return bool(defaults[model_name])
    return bool(training_cfg.get("use_compile", False))


def build_run_cfg(dcfg: dict[str, Any], training_cfg: dict[str, Any], dataset_name: str, model_name: str, seed: int, lambda_div: float, batch_size_requested: int, batch_size_effective: int, cache_path: str | None) -> dict[str, Any]:
    use_compile_requested = resolve_compile_request(model_name, dcfg, training_cfg)
    run_cfg = {**dcfg, **training_cfg, "dataset": dataset_name, "model": model_name, "seed": seed, "lambda_div": lambda_div}
    run_cfg.update({
        "batch_size_requested": batch_size_requested,
        "batch_size_effective": batch_size_effective,
        "cache_path": cache_path,
        "use_compile_requested": use_compile_requested,
        "use_compile": use_compile_requested,
        "use_compile_effective": False,
        "compile_failure_reason": None,
    })
    return run_cfg



def enrich_run_cfg_from_model(run_cfg: dict[str, Any], model: Any, logger=None) -> None:
    target = getattr(model, "module", model)
    if hasattr(target, "complex_summary"):
        summary = target.complex_summary()
        if summary:
            run_cfg["model_backend"] = summary.get("model_backend", "cellular_mhd_sno")
            run_cfg["cell_complex"] = summary
            if logger is not None:
                logger.info(
                    "model backend: %s complex type: %s grid shape: %s cells: %s coboundary nnz: %s "
                    "magnetic placement: C^2 faces EMF placement: C^1 edges fluid placement: C^3 cells "
                    "exact d2d1 check max error: %s use_sheaf_laplacian=%s "
                    "use_geometry_conditioned_restrictions=%s use_geometric_hodge=%s",
                    summary.get("model_backend"), summary.get("complex_type"), summary.get("grid_shape"),
                    summary.get("cells_by_dim"), summary.get("coboundary_nnz"),
                    summary.get("exact_d2d1_check_max_error"), summary.get("use_sheaf_laplacian"),
                    summary.get("use_geometry_conditioned_restrictions"), summary.get("use_geometric_hodge"),
                )

def save_failed_run(run_dir: Path, err: dict[str, Any]) -> None:
    save_json(err, run_dir / "failed_run.json")

def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or ("CUDA out of memory" in str(exc) or "cuda runtime error" in str(exc).lower() and "out of memory" in str(exc).lower())


def log_run_start(logger, run_cfg: dict[str, Any]) -> None:
    fields = [
        "dataset", "model", "seed", "batch_size_requested", "batch_size_effective", "use_amp", "amp_dtype",
        "use_compile_requested", "use_compile_effective", "compile_mode", "compile_failure_reason",
        "use_cache", "cache_path", "crop_size", "downsample_by",
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
        lambda_div = dcfg.get("lambda_div", 0.0) if dcfg.get("magnetic_field_indices") else 0.0
        run_cfg = build_run_cfg(
            dcfg,
            cfg.get("training", {}),
            dataset_name,
            model_name,
            seed,
            lambda_div,
            requested_bs,
            batch_size,
            str(getattr(datasets["train"], "cache_dir", dcfg.get("cache_root", "datasets/cache"))),
        )
        save_json(run_cfg, run_dir / "config_resolved.json")
        try:
            loaders = make_loaders(datasets, batch_size, loader_cfg, seed, device)
            model = build_time_model(model_name, in_ch, out_ch, dcfg)
            warmup_batch = next(iter(loaders["train"]))
            trainer = Trainer(model, loaders["train"], loaders["valid"], run_dir, device, run_cfg, compile_warmup_x=warmup_batch["x"])
            with torch.no_grad(), trainer._amp_context():
                _ = trainer.model(warmup_batch["x"][:1].to(device))
            enrich_run_cfg_from_model(run_cfg, trainer.original_model, logger)
            save_json(run_cfg, run_dir / "config_resolved.json")
            log_run_start(logger, run_cfg)
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



def validate_time_dataset_model_pairs(cfg: dict[str, Any], time_dataset_names: list[str], device: torch.device, logger) -> None:
    """Run one forward/loss/evaluation batch per available time-dependent pair before FINAL_RUN."""
    if not FINAL_RUN:
        return
    logger.info("Running FINAL_RUN startup validation before launching full seeds.")
    validation_root = Path("outputs") / "startup_validation_tmp"
    validation_root.mkdir(parents=True, exist_ok=True)
    for dataset_name in time_dataset_names:
        dcfg = dict(cfg["datasets"][dataset_name])
        if not Path(dcfg["data_root"]).exists():
            logger.warning("Skipping startup validation for missing dataset root: %s", dcfg["data_root"])
            continue
        datasets = {split: build_time_dataset(dataset_name, dcfg, split) for split in ("train", "valid", "test")}
        if dataset_name == "swigs_gorgon":
            train_ds = datasets["train"]
            dcfg["magnetic_field_indices"] = dcfg.get("magnetic_field_indices") or getattr(train_ds, "magnetic_field_indices", None)
        sample = datasets["train"][0]
        in_ch = sample["x"].shape[0]
        out_ch = sample["y"].shape[0]
        loaders = make_loaders(datasets, 1, {**cfg.get("training", {}), "num_workers": 0, "persistent_workers": False}, 0, device)
        batch = next(iter(loaders["train"]))
        eval_loader = DataLoader(datasets["valid"], batch_size=1, shuffle=False, num_workers=0, collate_fn=sample_collate)
        for model_name in cfg["time_dependent_models"]:
            run_cfg = build_run_cfg(
                dcfg,
                cfg.get("training", {}),
                dataset_name,
                model_name,
                0,
                dcfg.get("lambda_div", 0.0) if dcfg.get("magnetic_field_indices") else 0.0,
                1,
                1,
                str(getattr(datasets["train"], "cache_dir", dcfg.get("cache_root", "datasets/cache"))),
            )
            run_dir = validation_root / dataset_name / model_name
            try:
                model = build_time_model(model_name, in_ch, out_ch, dcfg)
                trainer = Trainer(model, loaders["train"], loaders["valid"], run_dir, device, run_cfg, compile_warmup_x=batch["x"])
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                with torch.no_grad(), trainer._amp_context():
                    pred = trainer.model(x)
                    enrich_run_cfg_from_model(run_cfg, trainer.original_model, logger)
                    save_json(run_cfg, run_dir / "config_resolved.json")
                    loss = mhd_loss(pred, y, run_cfg.get("lambda_rel", 0.1), run_cfg.get("lambda_div", 0.0), dcfg.get("magnetic_field_indices"), dcfg.get("spacing"))
                if not torch.isfinite(loss):
                    raise FloatingPointError("startup validation produced NaN/Inf loss")
                metrics = evaluate(trainer.model, eval_loader, device, dcfg.get("magnetic_field_indices"), dcfg.get("spacing"), include_spectral=True, use_amp=run_cfg.get("use_amp", False), amp_dtype=amp_dtype_from_cfg(run_cfg))
                if pred.ndim == 5 and not torch.isfinite(torch.tensor(metrics.get("spectral_error_3d", float("nan")))):
                    logger.warning("Startup validation spectral_error_3d is non-finite for %s/%s: %s", dataset_name, model_name, metrics.get("spectral_error_3d"))
                logger.info("Startup validation passed for %s/%s compile_effective=%s", dataset_name, model_name, run_cfg.get("use_compile_effective"))
            except Exception as exc:
                err = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": 0,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "resolved_config": run_cfg,
                }
                save_failed_run(run_dir, err)
                logger.error("Startup validation failed: %s", err)
                raise RuntimeError(f"Startup validation failed for {dataset_name}/{model_name}; stopping before FINAL_RUN") from exc

    if RUN_CONSTELLARATION:
        dataset_name = "constellaration_equilibrium"
        dcfg = dict(cfg["datasets"][dataset_name])
        if not Path(dcfg["data_root"]).exists():
            logger.warning("Skipping startup validation for missing optional dataset root: %s", dcfg["data_root"])
            return
        datasets = {split: build_equilibrium_dataset(dcfg, split) for split in ("train", "valid", "test")}
        sample = datasets["train"][0]
        in_dim = sample["x"].numel()
        out_dim = sample["y"].numel()
        loaders = make_loaders(datasets, 1, {**cfg.get("training", {}), "num_workers": 0, "persistent_workers": False}, 0, device)
        batch = next(iter(loaders["train"]))
        eval_loader = DataLoader(datasets["valid"], batch_size=1, shuffle=False, num_workers=0, collate_fn=sample_collate)
        for model_name in cfg["equilibrium_models"]:
            run_cfg = build_run_cfg(dcfg, cfg.get("training", {}), dataset_name, model_name, 0, 0.0, 1, 1, None)
            run_dir = validation_root / dataset_name / model_name
            try:
                model = build_equilibrium_model(model_name, in_dim, out_dim, dcfg)
                trainer = Trainer(model, loaders["train"], loaders["valid"], run_dir, device, run_cfg, compile_warmup_x=batch["x"])
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                with torch.no_grad(), trainer._amp_context():
                    pred = trainer.model(x)
                    loss = mhd_loss(pred, y, run_cfg.get("lambda_rel", 0.1), 0.0, None, None)
                if not torch.isfinite(loss):
                    raise FloatingPointError("startup validation produced NaN/Inf loss")
                evaluate(trainer.model, eval_loader, device, None, None, include_spectral=False, use_amp=run_cfg.get("use_amp", False), amp_dtype=amp_dtype_from_cfg(run_cfg))
                logger.info("Startup validation passed for %s/%s compile_effective=%s", dataset_name, model_name, run_cfg.get("use_compile_effective"))
            except Exception as exc:
                err = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": 0,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "resolved_config": run_cfg,
                }
                save_failed_run(run_dir, err)
                logger.error("Startup validation failed: %s", err)
                raise RuntimeError(f"Startup validation failed for {dataset_name}/{model_name}; stopping before FINAL_RUN") from exc

def main() -> None:
    start_suite = time.perf_counter()
    configure_torch_startup()
    cfg = apply_protocol(load_yaml("configs/default_experiment.yaml"))
    out_root, resumed_output_root = select_output_root()
    for sub in ["paper_tables", "figures/loss_curves", "figures/prediction_examples", "figures/divergence_maps", "figures/spectra", "figures/rollout_curves", "figures/aggregate_barplots", "runs"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_root / "experiment_log.txt")
    save_json(cfg, out_root / "resolved_experiment_config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Protocol active: {'FAST_DEV_RUN' if FAST_DEV_RUN else 'FINAL_RUN'}")
    logger.info("Starting suite on device=%s FAST_DEV_RUN=%s FINAL_RUN=%s output_root=%s resumed=%s", device, FAST_DEV_RUN, FINAL_RUN, out_root, resumed_output_root)

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
        
    if RUN_VALIDATION:
        validate_time_dataset_model_pairs(cfg, time_dataset_names, device, logger)

    for dataset_name in time_dataset_names:
        dcfg = dict(cfg["datasets"][dataset_name])
        if not Path(dcfg["data_root"]).exists():
            msg = f"Dataset root missing: {dcfg['data_root']}. Data are not downloaded by this codebase."
            logger.error("%s skipped/failed: %s", dataset_name, msg)
            for model_name in cfg["time_dependent_models"]:
                for seed in cfg["seeds"]:
                    row = {"dataset": dataset_name, "model": model_name, "seed": seed, "track": "time_dependent", "status": "failed", "error": msg}
                    run_dir = out_root / "runs" / dataset_name / model_name / f"seed_{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    save_failed_run(run_dir, row)
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
                    run_dir = out_root / "runs" / dataset_name / model_name / f"seed_{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    save_failed_run(run_dir, row)
                    raw_rows.append(row); failures.append(row)
            continue

        for model_name in cfg["time_dependent_models"]:
            for seed in cfg["seeds"]:
                seed_everything(seed)
                run_dir = out_root / "runs" / dataset_name / model_name / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "figures").mkdir(exist_ok=True)
                if record_completed_skip(run_dir, dataset_name, model_name, seed, "time_dependent", raw_rows, effective_batch_sizes, epoch_times, logger):
                    completed += 1
                    continue
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
                    row = {"dataset": dataset_name, "model": model_name, "seed": seed, "track": "time_dependent", "status": "completed", **test_metrics, "final_step_relative_l2": rollout.get("final_step_relative_l2"), "mean_rollout_relative_l2": rollout.get("mean_rollout_relative_l2")}
                    raw_rows.append(row)
                    save_completion_marker(run_dir, row, result["effective_batch_size"])
                    completed += 1
                    logger.info("Completed %s/%s seed %s", dataset_name, model_name, seed)
                except Exception as exc:
                    resolved_config = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8")) if (run_dir / "config_resolved.json").exists() else {}
                    err = {"dataset": dataset_name, "model": model_name, "seed": seed, "track": "time_dependent", "status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "resolved_config": resolved_config}
                    save_failed_run(run_dir, err)
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
                        if record_completed_skip(run_dir, dataset_name, model_name, seed, "equilibrium", raw_rows, effective_batch_sizes, epoch_times, logger):
                            completed += 1
                            continue
                        run_cfg = build_run_cfg(dcfg, cfg.get("training", {}), dataset_name, model_name, seed, 0.0, dcfg["batch_size"], dcfg["batch_size"], None)
                        save_json(run_cfg, run_dir / "config_resolved.json")
                        try:
                            requested_bs = int(dcfg["batch_size"])
                            batch_size = requested_bs
                            while True:
                                try:
                                    run_cfg["batch_size_requested"] = requested_bs
                                    run_cfg["batch_size_effective"] = batch_size
                                    save_json(run_cfg, run_dir / "config_resolved.json")
                                    loaders = make_loaders(datasets, batch_size, {**cfg.get("training", {}), **dcfg.get("dataloader", {})}, seed, device)
                                    warmup_batch = next(iter(loaders["train"]))
                                    trainer = Trainer(build_equilibrium_model(model_name, in_dim, out_dim, dcfg), loaders["train"], loaders["valid"], run_dir, device, run_cfg, compile_warmup_x=warmup_batch["x"])
                                    save_json(run_cfg, run_dir / "config_resolved.json")
                                    log_run_start(logger, run_cfg)
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
                            row = {"dataset": dataset_name, "model": model_name, "seed": seed, "track": "equilibrium", "status": "completed", **test_metrics}
                            raw_rows.append(row)
                            save_completion_marker(run_dir, row, batch_size)
                            effective_batch_sizes.append({"dataset": dataset_name, "model": model_name, "seed": seed, "batch_size": batch_size})
                            epoch_times.setdefault(f"{dataset_name}/{model_name}", []).extend(float(r.get("total_epoch_time", 0.0)) for r in rows)
                            completed += 1
                        except Exception as exc:
                            resolved_config = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8")) if (run_dir / "config_resolved.json").exists() else {}
                            err = {"dataset": dataset_name, "model": model_name, "seed": seed, "track": "equilibrium", "status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "resolved_config": resolved_config}
                            save_failed_run(run_dir, err); raw_rows.append(err); failures.append(err); logger.error("Run failed: %s", err)

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
