"""JSON/JSONL helpers for tabular scientific metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def flatten_numeric_json(obj: Any, prefix: str = "", max_depth: int = 8) -> dict[str, float]:
    out: dict[str, float] = {}
    if max_depth < 0:
        return out
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out[prefix or "value"] = float(obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_numeric_json(value, child, max_depth - 1))
    elif isinstance(obj, (list, tuple)):
        if len(obj) <= 256:
            for i, value in enumerate(obj):
                child = f"{prefix}[{i}]" if prefix else f"[{i}]"
                out.update(flatten_numeric_json(value, child, max_depth - 1))
    return out
