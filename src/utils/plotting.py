from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _image_slice(a):
    arr = a.detach().cpu() if hasattr(a, "detach") else np.asarray(a)
    while arr.ndim > 2:
        arr = arr[..., arr.shape[-1] // 2]
    return arr


def plot_loss_curve(train_log_csv, out_path):
    df = pd.read_csv(train_log_csv)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train")
    if "valid_relative_l2" in df:
        plt.plot(df["epoch"], df["valid_relative_l2"], label="valid relative L2")
    plt.xlabel("epoch")
    plt.legend()
    plt.title("Sheaf Neural Operators for MHD loss")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_prediction_example(pred, target, out_path, channel: int = 0):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    p = _image_slice(pred[0, channel])
    t = _image_slice(target[0, channel])
    plt.figure(figsize=(9, 3))
    for i, (a, title) in enumerate([(p, "prediction"), (t, "target"), (np.abs(p - t), "abs error")]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(a)
        plt.title(title)
        plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_error_heatmap(pred, target, out_path):
    err = (pred - target).detach().abs().mean(dim=1).cpu()[0]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(_image_slice(err))
    plt.title("Mean absolute error")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_divergence_map(div, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(_image_slice(div[0]))
    plt.title("Magnetic divergence map")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_rollout_error(metrics, out_path):
    y = metrics.get("rollout_relative_l2_by_step", [])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(y) + 1), y, marker="o")
    plt.xlabel("rollout step")
    plt.ylabel("relative L2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_aggregate_bars(agg, out_path, metric="relative_l2"):
    sub = agg[agg.metric == metric]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if sub.empty:
        raise ValueError(f"Cannot plot aggregate bars: no rows for metric {metric!r}")
    labels = [f"{r.dataset}\n{r.model}" for r in sub.itertuples()]
    y = sub["mean"].to_numpy()
    err = np.vstack([y - sub["ci95_low"].to_numpy(), sub["ci95_high"].to_numpy() - y])
    plt.figure(figsize=(max(6, len(labels)), 4))
    plt.bar(labels, y, yerr=err)
    plt.ylabel(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
