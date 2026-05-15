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
ID_KEY_PARTS = ("plasma_config_id", "source_plasma_config_id", "vmecpp_wout_id")
DIRECT_ID_KEYS = {"plasma_config_id", "source_plasma_config_id", "id", "wout_id", "vmecpp_wout_id", "config_id"}


class ConStellarationDataset(Dataset):
    """JSONL equilibrium regression dataset with robust WOut join fallback."""

    def __init__(self, data_root: str | Path, split: str, max_samples: int | None = None, normalize: bool = True) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.normalize = normalize
        self.join_mode = "unknown"
        boundary_path = self.data_root / "boundaries_and_metrics.jsonl"
        target_path = self.data_root / "vmecpp_wout_finite_beta_3pct.jsonl"
        if not self.data_root.exists():
            raise FileNotFoundError(f"ConStellaration subset not found: {self.data_root}")
        if not boundary_path.exists() or not target_path.exists():
            raise FileNotFoundError(f"Expected {boundary_path.name} and {target_path.name} under {self.data_root}")
        inputs = [self._parse_json_strings(row) for row in read_jsonl(boundary_path)]
        rows, report = self._join_rows(inputs, target_path)
        if not rows:
            raise ValueError("ConStellaration JSONL files could not be converted into supervised rows.")
        self.join_mode = report["join_mode"]
        self.x_keys, self.y_keys, x_all, y_all, metas = self._vectorize(rows)
        report.update({"x_feature_count": len(self.x_keys), "y_target_count": len(self.y_keys)})
        self.join_report = report
        self._write_join_report(report)
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
        self.inspection = {"join_mode": self.join_mode, "join_report_path": str(self.report_path), "feature_dim": self.feature_dim, "target_dim": self.target_dim}

    def _write_join_report(self, report: dict[str, Any]) -> None:
        out_dir = Path("outputs") / "constellaration_join_reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = out_dir / f"constellaration_join_report_{self.split}.json"
        self.report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

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

    def _walk(self, obj: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                yield path, value
                yield from self._walk(value, path)
        elif isinstance(obj, list):
            for i, value in enumerate(obj[:128]):
                yield from self._walk(value, f"{prefix}.{i}" if prefix else str(i))

    def _row_ids(self, row: dict[str, Any]) -> Iterable[str]:
        seen: set[str] = set()
        for path, value in self._walk(row):
            key = path.split(".")[-1]
            include = key in DIRECT_ID_KEYS or path.endswith(".id") or any(part in path for part in ID_KEY_PARTS)
            if include and value is not None and not isinstance(value, (dict, list)):
                text = str(value)
                if text not in seen:
                    seen.add(text)
                    yield text

    def _primary_id(self, row: dict[str, Any]) -> str | None:
        return next(iter(self._row_ids(row)), None)

    def _join_rows(self, inputs: list[dict[str, Any]], target_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        input_ids = {i: list(self._row_ids(row)) for i, row in enumerate(inputs)}
        inputs_by_id: dict[str, list[int]] = {}
        for i, ids in input_ids.items():
            for row_id in ids:
                inputs_by_id.setdefault(row_id, []).append(i)
        joined: dict[int, dict[str, Any]] = {}
        target_rows_scanned = 0
        example_target_ids: list[str] = []
        with open(target_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    target = self._parse_json_strings(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} of {target_path}: {exc}") from exc
                target_rows_scanned += 1
                tids = list(self._row_ids(target))
                if tids and len(example_target_ids) < 8:
                    example_target_ids.extend(tids[: 8 - len(example_target_ids)])
                for tid in tids:
                    for input_index in inputs_by_id.get(tid, []):
                        joined.setdefault(input_index, {"input": inputs[input_index], "target": target, "id": tid, "join_mode": "id_join"})
        if joined:
            rows = list(joined.values())
            mode = "id_join"
        else:
            rows = self._metrics_only_rows(inputs)
            mode = "metrics_only_fallback"
            print(
                "WARNING: ConStellaration WOut ID join failed; using metrics_only_fallback from boundaries_and_metrics.jsonl "
                "instead of unsafe positional WOut fallback."
            )
        report = {
            "input_rows": len(inputs),
            "target_rows_scanned": target_rows_scanned,
            "joined_rows": len(joined),
            "join_mode": mode,
            "example_input_candidate_ids": [rid for ids in list(input_ids.values())[:8] for rid in ids[:2]][:16],
            "example_target_candidate_ids": example_target_ids[:16],
            "target_path": str(target_path),
        }
        return rows, report

    def _metrics_only_rows(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for i, inp in enumerate(inputs):
            flat = flatten_numeric_json(inp)
            metrics = {key: value for key, value in flat.items() if key.startswith("metrics") or ".metrics" in key or key.startswith("metrics.json")}
            # Include all numeric metrics; common paper targets such as aspect_ratio,
            # qi, and vacuum_well remain present when available.
            target = metrics
            if not target:
                # Last-resort supervised target from numeric leaves; still explicit in metadata.
                target = {f"numeric_summary.{k}": v for k, v in list(flat.items())[: min(16, len(flat))]}
            rows.append({"input": inp, "target": target, "id": self._primary_id(inp) or str(i), "join_mode": "metrics_only_fallback"})
        return rows

    def _vectorize(self, rows: list[dict[str, Any]]):
        x_maps = [flatten_numeric_json(row["input"]) for row in rows]
        y_maps = [flatten_numeric_json(row["target"]) for row in rows]
        x_keys = sorted({key for item in x_maps for key in item})
        y_keys = sorted({key for item in y_maps for key in item})
        if not x_keys or not y_keys:
            raise ValueError("No numeric ConStellaration features or targets were found after parsing JSON strings and flattening JSON.")
        x = np.asarray([[item.get(key, 0.0) for key in x_keys] for item in x_maps], dtype=np.float32)
        y = np.asarray([[item.get(key, 0.0) for key in y_keys] for item in y_maps], dtype=np.float32)
        metas = [{"id": row["id"], "input_keys": len(x_keys), "target_keys": len(y_keys), "join_mode": row.get("join_mode", self.join_mode)} for row in rows]
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
