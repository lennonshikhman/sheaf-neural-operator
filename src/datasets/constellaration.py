"""Optional ConStellaration equilibrium surrogate dataset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.json_utils import flatten_numeric_json, read_jsonl

JSON_STRING_KEYS = {"boundary.json", "metrics.json", "json"}


class ConStellarationDataset(Dataset):
    """JSONL equilibrium regression dataset with deterministic splits.

    ConStellaration is treated as tabular/JSON regression rather than a
    time-dependent tensor problem. JSON-valued string columns in the local files
    are parsed and numeric leaves from the parsed structures are flattened into
    feature/target vectors.
    """

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
        inputs = [self._parse_json_strings(row) for row in read_jsonl(boundary_path)]
        rows = self._join_rows(inputs, target_path)
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

    def _parse_json_strings(self, row: dict[str, Any]) -> dict[str, Any]:
        parsed = dict(row)
        for key, value in list(row.items()):
            if key not in JSON_STRING_KEYS or not isinstance(value, str):
                continue
            stripped = value.strip()
            if not stripped or stripped[0] not in "[{":
                continue
            try:
                parsed[key] = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"ConStellaration column {key!r} contains invalid JSON text: {exc}") from exc
        return parsed

    def _row_ids(self, row: dict[str, Any]) -> Iterable[str]:
        for key in ("plasma_config_id", "source_plasma_config_id", "config_id", "id", "misc.source_plasma_config_id", "misc.vmecpp_wout_id"):
            if key in row and row[key] is not None:
                yield str(row[key])
        misc = row.get("misc")
        if isinstance(misc, dict):
            for key in ("source_plasma_config_id", "vmecpp_wout_id"):
                if key in misc and misc[key] is not None:
                    yield str(misc[key])

    def _primary_id(self, row: dict[str, Any]) -> str | None:
        return next(iter(self._row_ids(row)), None)

    def _join_rows(self, inputs: list[dict[str, Any]], target_path: Path) -> list[dict[str, Any]]:
        inputs_by_id: dict[str, dict[str, Any]] = {}
        for row in inputs:
            for row_id in self._row_ids(row):
                inputs_by_id[row_id] = row
        remaining = set(inputs_by_id)
        joined_by_input_id: dict[str, dict[str, Any]] = {}
        with open(target_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    target = self._parse_json_strings(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} of {target_path}: {exc}") from exc
                matching_ids = [row_id for row_id in self._row_ids(target) if row_id in inputs_by_id]
                for row_id in matching_ids:
                    joined_by_input_id.setdefault(row_id, {"input": inputs_by_id[row_id], "target": target, "id": row_id})
                    remaining.discard(row_id)
                if not remaining:
                    break
        if joined_by_input_id:
            return list(joined_by_input_id.values())
        if target_path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError(
                f"Could not join ConStellaration rows by IDs from {target_path}; refusing positional fallback "
                "on a large WOut JSONL file. Check plasma_config_id, misc.vmecpp_wout_id, and target id fields."
            )
        # Fallback for tiny synthetic or reordered subsets without matching IDs.
        targets = [self._parse_json_strings(row) for row in read_jsonl(target_path)]
        joined = []
        for i, inp in enumerate(inputs[: len(targets)]):
            joined.append({"input": inp, "target": targets[i], "id": self._primary_id(inp) or str(i)})
        return joined

    def _vectorize(self, rows: list[dict[str, Any]]):
        x_maps = [flatten_numeric_json(row["input"]) for row in rows]
        y_maps = [flatten_numeric_json(row["target"]) for row in rows]
        x_keys = sorted({key for item in x_maps for key in item})
        y_keys = sorted({key for item in y_maps for key in item})
        if not x_keys or not y_keys:
            raise ValueError("No numeric ConStellaration features or targets were found after parsing JSON strings and flattening JSON.")
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
