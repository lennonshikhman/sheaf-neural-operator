"""Optional ConStellaration equilibrium surrogate dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.json_utils import flatten_numeric_json, read_jsonl


class ConStellarationDataset(Dataset):
    """JSONL equilibrium regression dataset with deterministic splits."""

    def __init__(self, data_root: str | Path, split: str, max_samples: int | None = None, normalize: bool = True) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.normalize = normalize
        boundary_path = self.data_root / "boundaries_and_metrics.jsonl"
        target_path = self.data_root / "vmecpp_wout_finite_beta_3pct.jsonl"
        if not self.data_root.exists():
            raise FileNotFoundError(f"ConStellaration subset not found: {self.data_root}")
        if not boundary_path.exists() or not target_path.exists():
            raise FileNotFoundError(f"Expected {boundary_path.name} and {target_path.name} under {self.data_root}")
        rows = self._join_rows(read_jsonl(boundary_path), read_jsonl(target_path))
        if not rows:
            raise ValueError("ConStellaration JSONL files could not be joined into supervised rows.")
        self.x_keys, self.y_keys, x_all, y_all, metas = self._vectorize(rows)
        indices = self._split_indices(len(rows), split)
        if max_samples is not None:
            indices = indices[:max_samples]
        self.x = x_all[indices]
        self.y = y_all[indices]
        self.metas = [metas[i] for i in indices]
        if normalize:
            self.x_mean = x_all.mean(axis=0)
            self.x_std = np.maximum(x_all.std(axis=0), 1e-6)
            self.y_mean = y_all.mean(axis=0)
            self.y_std = np.maximum(y_all.std(axis=0), 1e-6)
            self.x = (self.x - self.x_mean) / self.x_std
            self.y = (self.y - self.y_mean) / self.y_std
        self.feature_dim = int(self.x.shape[1])
        self.target_dim = int(self.y.shape[1])

    def _row_id(self, row: dict[str, Any]) -> str | None:
        for key in ("plasma_config_id", "source_plasma_config_id", "config_id", "id"):
            if key in row:
                return str(row[key])
        return None

    def _join_rows(self, inputs: list[dict[str, Any]], targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        target_by_id = {self._row_id(row): row for row in targets if self._row_id(row) is not None}
        joined = []
        for i, inp in enumerate(inputs):
            rid = self._row_id(inp)
            target = target_by_id.get(rid) if rid is not None else (targets[i] if i < len(targets) else None)
            if target is not None:
                joined.append({"input": inp, "target": target, "id": rid or str(i)})
        return joined

    def _vectorize(self, rows: list[dict[str, Any]]):
        x_maps = [flatten_numeric_json(row["input"]) for row in rows]
        y_maps = [flatten_numeric_json(row["target"]) for row in rows]
        x_keys = sorted({key for item in x_maps for key in item})
        y_keys = sorted({key for item in y_maps for key in item})
        if not x_keys or not y_keys:
            raise ValueError("No numeric ConStellaration features or targets were found after flattening JSON.")
        x = np.asarray([[item.get(key, 0.0) for key in x_keys] for item in x_maps], dtype=np.float32)
        y = np.asarray([[item.get(key, 0.0) for key in y_keys] for item in y_maps], dtype=np.float32)
        metas = [{"id": row["id"], "input_keys": len(x_keys), "target_keys": len(y_keys)} for row in rows]
        return x_keys, y_keys, x, y, metas

    def _split_indices(self, n: int, split: str) -> np.ndarray:
        rng = np.random.default_rng(12345)
        order = rng.permutation(n)
        train_end = max(1, int(0.7 * n))
        valid_end = max(train_end + 1, int(0.85 * n)) if n > 2 else n
        if split == "train":
            return order[:train_end]
        if split in {"valid", "validation"}:
            return order[train_end:valid_end]
        if split == "test":
            return order[valid_end:] if valid_end < n else order[-1:]
        raise ValueError(f"Unknown split {split!r}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"x": torch.tensor(self.x[idx], dtype=torch.float32), "y": torch.tensor(self.y[idx], dtype=torch.float32), "meta": self.metas[idx]}
