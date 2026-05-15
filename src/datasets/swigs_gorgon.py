"""SWIGS/Gorgon HDF5 dataset adapter for 3D MHD surrogate modeling."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

MHD_PRIORITY = {
    "density": ("density", "rho", "mass_density", "dens"),
    "pressure": ("pressure", "press", "p"),
    "vx": ("vx", "v_x", "velx", "velocity_x", "x_velocity"),
    "vy": ("vy", "v_y", "vely", "velocity_y", "y_velocity"),
    "vz": ("vz", "v_z", "velz", "velocity_z", "z_velocity"),
    "bx": ("bx", "b_x", "magx", "magnetic_x", "x_magnetic", "bfieldx"),
    "by": ("by", "b_y", "magy", "magnetic_y", "y_magnetic", "bfieldy"),
    "bz": ("bz", "b_z", "magz", "magnetic_z", "z_magnetic", "bfieldz"),
    "jx": ("jx", "j_x", "current_x"),
    "jy": ("jy", "j_y", "current_y"),
    "jz": ("jz", "j_z", "current_z"),
}


def inspect_hdf5_file(path: str | Path) -> dict[str, Any]:
    """Return a nested summary of groups and datasets in an HDF5 file."""
    path = Path(path)
    summary: dict[str, Any] = {"path": str(path), "datasets": {}, "groups": []}
    with h5py.File(path, "r") as h5:
        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                summary["datasets"][name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
            elif isinstance(obj, h5py.Group):
                summary["groups"].append(name)
        h5.visititems(visitor)
    return summary


def _canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower().split("/")[-1])


def _extract_time(path: Path) -> float | None:
    candidates = re.findall(r"(?i)(?:time|t|step)[_=-]?([0-9]+(?:\.[0-9]+)?)", path.stem)
    if candidates:
        return float(candidates[-1])
    nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", path.stem)
    return float(nums[-1]) if nums else None


def _compatible_shape(shape: list[int]) -> bool:
    return len(shape) == 3 and min(shape) > 1


class SWIGSGorgonDataset(Dataset):
    """Recursively discovers SWIGS/Gorgon HDF5 files and returns adjacent-time 3D field pairs."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        n_input_frames: int = 1,
        n_output_frames: int = 1,
        max_samples: int | None = None,
        normalize: bool = True,
        field_mode: str = "core_mhd",
        index_cache_name: str = ".swigs_index.json",
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.n_input_frames = n_input_frames
        self.n_output_frames = n_output_frames
        self.normalize = normalize
        self.field_mode = field_mode
        self.index_path = self.data_root / index_cache_name
        if not self.data_root.exists():
            raise FileNotFoundError(f"SWIGS/Gorgon root not found: {self.data_root}")
        index = self._load_or_build_index()
        self.inspection = index["inspection"]
        self.field_names = index["field_names"]
        self.field_keys = index["field_keys"]
        self.magnetic_field_indices = index.get("magnetic_field_indices")
        self.inferred_field_mapping = bool(index.get("inferred_field_mapping", False))
        self.samples = self._split_samples(index["samples"], split)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise FileNotFoundError(f"No SWIGS/Gorgon samples for split={split} under {self.data_root}")
        self.mean, self.std = self._estimate_stats() if normalize else (None, None)

    def _load_or_build_index(self) -> dict[str, Any]:
        h5_files = sorted([*self.data_root.rglob("*.h5"), *self.data_root.rglob("*.hdf5"), *self.data_root.rglob("*.hdf")])
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found recursively under {self.data_root}")
        cache_valid = self.index_path.exists()
        if cache_valid:
            cached = json.loads(self.index_path.read_text(encoding="utf-8"))
            cached_files = cached.get("source_files", [])
            if cached_files == [str(p) for p in h5_files] and cached.get("field_mode") == self.field_mode:
                return cached
        index = self._build_index(h5_files)
        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        return index

    def _build_index(self, files: list[Path]) -> dict[str, Any]:
        inspections = [inspect_hdf5_file(path) for path in files]
        first = inspections[0]
        candidates = {k: v for k, v in first["datasets"].items() if _compatible_shape(v["shape"])}
        if len(candidates) < 4:
            raise ValueError(f"SWIGS/Gorgon file {first['path']} has fewer than 4 compatible 3D field arrays; found {list(candidates)}")
        field_keys, field_names, inferred = self._select_fields(candidates)
        times = [_extract_time(path) for path in files]
        order = sorted(range(len(files)), key=lambda i: (float("inf") if times[i] is None else times[i], str(files[i])))
        samples = []
        for a, b in zip(order[:-1], order[1:], strict=False):
            samples.append({"file_x": str(files[a]), "file_y": str(files[b]), "time_x": times[a], "time_y": times[b]})
        magnetic = [i for i, name in enumerate(field_names) if name in {"bx", "by", "bz"}]
        magnetic_indices = magnetic if len(magnetic) >= 3 else None
        return {
            "source_files": [str(p) for p in files],
            "field_mode": self.field_mode,
            "inspection": inspections,
            "field_keys": field_keys,
            "field_names": field_names,
            "inferred_field_mapping": inferred,
            "magnetic_field_indices": magnetic_indices,
            "samples": samples,
        }

    def _select_fields(self, candidates: dict[str, dict[str, Any]]) -> tuple[list[str], list[str], bool]:
        canonical_to_key = {_canonical(key): key for key in candidates}
        selected: list[tuple[str, str]] = []
        for physical, aliases in MHD_PRIORITY.items():
            for alias in aliases:
                can = _canonical(alias)
                matches = [key for cname, key in canonical_to_key.items() if can == cname or can in cname]
                if matches:
                    selected.append((physical, sorted(matches)[0]))
                    break
        names = [name for name, _ in selected]
        has_core = ({"density", "pressure"} & set(names)) and {"vx", "vy", "vz"}.issubset(names) and {"bx", "by", "bz"}.issubset(names)
        if self.field_mode == "core_mhd" and has_core:
            keep = [item for item in selected if item[0] in {"density", "pressure", "vx", "vy", "vz", "bx", "by", "bz"}]
            return [key for _, key in keep], [name for name, _ in keep], False
        sorted_candidates = sorted(candidates.items(), key=lambda kv: (np.prod(kv[1]["shape"]), kv[0]), reverse=True)
        if len(sorted_candidates) < 4:
            raise ValueError("Could not infer at least four SWIGS/Gorgon 3D fields.")
        keys = [key for key, _ in sorted_candidates]
        names = [Path(key).name for key in keys]
        return keys, names, True

    def _split_samples(self, samples: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        n = len(samples)
        train_end = max(1, int(0.7 * n))
        valid_end = max(train_end + 1, int(0.85 * n)) if n > 2 else n
        if split == "train":
            return samples[:train_end]
        if split in {"valid", "validation"}:
            return samples[train_end:valid_end]
        if split == "test":
            return samples[valid_end:] or samples[-1:]
        raise ValueError(f"Unknown split {split!r}")

    def _read_fields(self, path: str | Path) -> torch.Tensor:
        arrays = []
        with h5py.File(path, "r") as h5:
            for key in self.field_keys:
                arr = np.asarray(h5[key][...], dtype=np.float32)
                if arr.ndim != 3:
                    raise ValueError(f"Selected SWIGS field {key} in {path} is shape {arr.shape}, expected 3D")
                arrays.append(arr)
        return torch.tensor(np.stack(arrays), dtype=torch.float32)

    def _estimate_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        vals = [self._read_fields(sample["file_x"]) for sample in self.samples[: min(8, len(self.samples))]]
        stack = torch.stack(vals)
        dims = (0, 2, 3, 4)
        return stack.mean(dim=dims), stack.std(dim=dims).clamp_min(1e-6)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        x = self._read_fields(sample["file_x"])
        y = self._read_fields(sample["file_y"])
        if self.normalize and self.mean is not None and self.std is not None:
            mean = self.mean.view(-1, 1, 1, 1)
            std = self.std.view(-1, 1, 1, 1)
            x = (x - mean) / std
            y = (y - mean) / std
        return {
            "x": x,
            "y": y,
            "meta": {**sample, "field_names": self.field_names, "field_keys": self.field_keys, "inferred_field_mapping": self.inferred_field_mapping},
        }
