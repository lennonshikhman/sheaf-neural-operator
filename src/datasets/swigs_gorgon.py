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
    squeezed = [dim for dim in shape if dim != 1]
    return len(squeezed) == 3 and min(squeezed) > 1


def _slice_field_key(dataset_key: str, axis: int, index: int) -> str:
    return f"{dataset_key}::axis{axis}[{index}]"


def _parse_field_key(field_key: str) -> tuple[str, int | None, int | None]:
    match = re.match(r"^(?P<dataset>.*)::axis(?P<axis>-?\d+)\[(?P<index>\d+)\]$", field_key)
    if match is None:
        return field_key, None, None
    return match.group("dataset"), int(match.group("axis")), int(match.group("index"))


def _candidate_fields(datasets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Expand HDF5 datasets into readable 3D field candidates.

    Gorgon exports are not completely uniform: some files store each physical
    field as its own 3D dataset, some include singleton dimensions, and some
    pack several components into a leading or trailing channel axis.  This
    normalizes those layouts into field keys that ``_read_fields`` can load.
    """
    candidates: dict[str, dict[str, Any]] = {}
    for key, value in datasets.items():
        shape = list(value["shape"])
        if _compatible_shape(shape):
            entry = dict(value)
            entry["field_shape"] = [dim for dim in shape if dim != 1]
            candidates[key] = entry
            continue
        if len(shape) == 4:
            for axis, dim in enumerate(shape):
                remaining = [size for i, size in enumerate(shape) if i != axis and size != 1]
                if 1 < dim <= 32 and len(remaining) == 3 and min(remaining) > 1:
                    for index in range(dim):
                        field_key = _slice_field_key(key, axis, index)
                        entry = dict(value)
                        entry["field_shape"] = remaining
                        entry["source_dataset"] = key
                        entry["slice_axis"] = axis
                        entry["slice_index"] = index
                        candidates[field_key] = entry
                    break
    return candidates


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
        self.skipped_files = index.get("skipped_files", [])
        self.template_file = index.get("template_file")
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
        usable: list[tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]] = []
        skipped: list[dict[str, Any]] = []
        for path, inspection in zip(files, inspections, strict=True):
            candidates = _candidate_fields(inspection["datasets"])
            if len(candidates) >= 4:
                usable.append((path, inspection, candidates))
            else:
                skipped.append({"path": str(path), "reason": "fewer than four compatible 3D fields", "compatible_fields": list(candidates)})
        if not usable:
            raise ValueError(
                f"No SWIGS/Gorgon HDF5 files under {self.data_root} contained at least four compatible 3D fields. "
                "Parameter-only IS files are skipped automatically; check dataset_inspection.json for file contents."
            )

        # Choose the richest schema as the canonical field set, then use only files
        # that contain those keys.  This skips SWIGS parameter files while retaining
        # Gorgon state dumps such as *_MS_params_*.hdf5 from the tree.
        template_path, _, template_candidates = max(usable, key=lambda item: (len(item[2]), str(item[0])))
        field_keys, field_names, inferred = self._select_fields(template_candidates)
        compatible_files = []
        for path, _, candidates in usable:
            missing = [key for key in field_keys if key not in candidates]
            if missing:
                skipped.append({"path": str(path), "reason": "missing selected field keys", "missing_fields": missing})
            else:
                compatible_files.append(path)
        if len(compatible_files) < self.n_input_frames + self.n_output_frames:
            raise ValueError(
                f"Need at least {self.n_input_frames + self.n_output_frames} SWIGS/Gorgon files with schema from {template_path}; "
                f"found {len(compatible_files)}."
            )

        samples = []
        for group_files in self._group_files_for_sequences(compatible_files):
            times = [_extract_time(path) for path in group_files]
            order = sorted(range(len(group_files)), key=lambda i: (float("inf") if times[i] is None else times[i], str(group_files[i])))
            window = self.n_input_frames + self.n_output_frames
            for start in range(0, len(order) - window + 1):
                in_idx = order[start : start + self.n_input_frames]
                out_idx = order[start + self.n_input_frames : start + window]
                samples.append({
                    "files_x": [str(group_files[i]) for i in in_idx],
                    "files_y": [str(group_files[i]) for i in out_idx],
                    "file_x": str(group_files[in_idx[-1]]),
                    "file_y": str(group_files[out_idx[-1]]),
                    "time_x": times[in_idx[-1]],
                    "time_y": times[out_idx[-1]],
                    "sequence": str(group_files[0].parent.relative_to(self.data_root)) if group_files[0].is_relative_to(self.data_root) else str(group_files[0].parent),
                })
        if not samples:
            raise ValueError(f"No adjacent SWIGS/Gorgon time pairs could be built from {len(compatible_files)} compatible files.")
        samples.sort(key=lambda sample: (sample["sequence"], float("inf") if sample["time_x"] is None else sample["time_x"], sample["file_x"]))
        magnetic = [i for i, name in enumerate(field_names) if name in {"bx", "by", "bz"}]
        magnetic_indices = magnetic if len(magnetic) >= 3 else None
        return {
            "source_files": [str(p) for p in files],
            "field_mode": self.field_mode,
            "inspection": inspections,
            "skipped_files": skipped,
            "template_file": str(template_path),
            "field_keys": field_keys,
            "field_names": field_names,
            "inferred_field_mapping": inferred,
            "magnetic_field_indices": magnetic_indices,
            "samples": samples,
        }

    def _group_files_for_sequences(self, files: list[Path]) -> list[list[Path]]:
        groups: dict[Path, list[Path]] = {}
        for path in files:
            groups.setdefault(path.parent, []).append(path)
        return [sorted(group) for _, group in sorted(groups.items(), key=lambda item: str(item[0])) if len(group) >= self.n_input_frames + self.n_output_frames]

    def _select_fields(self, candidates: dict[str, dict[str, Any]]) -> tuple[list[str], list[str], bool]:
        canonical_to_key = {_canonical(_parse_field_key(key)[0]): key for key in candidates}
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
        names = []
        for key in keys:
            dataset_key, slice_axis, slice_index = _parse_field_key(key)
            base_name = Path(dataset_key).name
            names.append(base_name if slice_axis is None else f"{base_name}_{slice_index}")
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
                dataset_key, slice_axis, slice_index = _parse_field_key(key)
                arr = np.asarray(h5[dataset_key][...], dtype=np.float32)
                if slice_axis is not None and slice_index is not None:
                    arr = np.take(arr, slice_index, axis=slice_axis)
                arr = np.squeeze(arr)
                if arr.ndim != 3:
                    raise ValueError(f"Selected SWIGS field {key} in {path} is shape {arr.shape}, expected 3D after slicing/squeezing")
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
        x_frames = [self._read_fields(path) for path in sample.get("files_x", [sample["file_x"]])]
        y_frames = [self._read_fields(path) for path in sample.get("files_y", [sample["file_y"]])]
        x = torch.cat(x_frames, dim=0)
        y = torch.cat(y_frames, dim=0)
        if self.normalize and self.mean is not None and self.std is not None:
            mean = self.mean.repeat(len(x_frames)).view(-1, 1, 1, 1)
            std = self.std.repeat(len(x_frames)).view(-1, 1, 1, 1)
            x = (x - mean) / std
            mean_y = self.mean.repeat(len(y_frames)).view(-1, 1, 1, 1)
            std_y = self.std.repeat(len(y_frames)).view(-1, 1, 1, 1)
            y = (y - mean_y) / std_y
        return {
            "x": x,
            "y": y,
            "meta": {**sample, "field_names": self.field_names, "field_keys": self.field_keys, "inferred_field_mapping": self.inferred_field_mapping},
        }
