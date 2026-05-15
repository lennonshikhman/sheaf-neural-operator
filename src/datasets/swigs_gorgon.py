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

DEFAULT_REQUIRED_FIELDS = ["P", "Bvec_c"]


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
    nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", path.stem)
    return float(nums[-1]) if nums else None


def _is_ms_file(path: Path) -> bool:
    stem = path.stem.lower()
    return "_ms_" in stem and "_is_" not in stem


def _compatible_field_shape(shape: list[int]) -> bool:
    if len(shape) == 3:
        return min(shape) > 1
    if len(shape) == 4 and shape[-1] in {2, 3}:
        return min(shape[:-1]) > 1
    return False


def _field_channel_names(key: str, shape: list[int]) -> list[str]:
    base = Path(key).name
    if len(shape) == 4 and shape[-1] == 3:
        return [f"{base}_x", f"{base}_y", f"{base}_z"]
    if len(shape) == 4 and shape[-1] == 2:
        return [f"{base}_0", f"{base}_1"]
    return [base]


class SWIGSGorgonDataset(Dataset):
    """Return same-field SWIGS/Gorgon MS states at adjacent timestamps.

    The first supported local schema intentionally ignores ionosphere ``IS``
    files. It indexes magnetosphere ``MS`` files by shock directory and timestamp,
    keeps timestamps that contain the required fields, and builds samples only
    when the full required field set exists at both input and output times.
    Large ``480x320x320`` fields are downsampled by default before training.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        n_input_frames: int = 1,
        n_output_frames: int = 1,
        max_samples: int | None = None,
        normalize: bool = True,
        field_mode: str = "ms_required",
        index_cache_name: str = ".swigs_index.json",
        required_fields: list[str] | None = None,
        spatial_downsample: int = 8,
        crop_shape: list[int] | tuple[int, int, int] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.n_input_frames = n_input_frames
        self.n_output_frames = n_output_frames
        self.normalize = normalize
        self.field_mode = field_mode
        self.required_fields = required_fields or DEFAULT_REQUIRED_FIELDS
        self.spatial_downsample = max(1, int(spatial_downsample))
        self.crop_shape = tuple(crop_shape) if crop_shape is not None else None
        self.index_path = self.data_root / index_cache_name
        if not self.data_root.exists():
            raise FileNotFoundError(f"SWIGS/Gorgon root not found: {self.data_root}")
        index = self._load_or_build_index()
        self.inspection = index["inspection"]
        self.skipped_files = index.get("skipped_files", [])
        self.field_keys = index["field_keys"]
        self.field_names = index["field_names"]
        self.magnetic_field_indices = index.get("magnetic_field_indices")
        self.inferred_field_mapping = False
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
        signature = {
            "source_files": [str(p) for p in h5_files],
            "field_mode": self.field_mode,
            "required_fields": self.required_fields,
        }
        if self.index_path.exists():
            cached = json.loads(self.index_path.read_text(encoding="utf-8"))
            if all(cached.get(key) == value for key, value in signature.items()):
                return cached
        index = self._build_index(h5_files)
        index.update(signature)
        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        return index

    def _build_index(self, files: list[Path]) -> dict[str, Any]:
        inspections = [inspect_hdf5_file(path) for path in files]
        required_canonical = {_canonical(field): field for field in self.required_fields}
        records: dict[str, dict[float, dict[str, Any]]] = {}
        skipped: list[dict[str, Any]] = []
        field_shapes: dict[str, list[int]] = {}

        for path, inspection in zip(files, inspections, strict=True):
            if not _is_ms_file(path):
                skipped.append({"path": str(path), "reason": "not an MS magnetosphere file"})
                continue
            compatible = {key: value for key, value in inspection["datasets"].items() if _compatible_field_shape(value["shape"])}
            available = {_canonical(key): key for key in compatible}
            present = {required_name: available[canonical] for canonical, required_name in required_canonical.items() if canonical in available}
            if not present:
                skipped.append({"path": str(path), "reason": "no required MS fields", "compatible_fields": list(compatible)})
                continue
            time_value = _extract_time(path)
            if time_value is None:
                skipped.append({"path": str(path), "reason": "could not infer timestamp"})
                continue
            sequence = str(path.parent.relative_to(self.data_root)) if path.is_relative_to(self.data_root) else str(path.parent)
            record = records.setdefault(sequence, {}).setdefault(time_value, {"time": time_value, "sequence": sequence, "fields": {}})
            for field_name, key in present.items():
                record["fields"][field_name] = {"file": str(path), "key": key, "shape": compatible[key]["shape"]}
                field_shapes.setdefault(field_name, compatible[key]["shape"])

        samples: list[dict[str, Any]] = []
        window = self.n_input_frames + self.n_output_frames
        for sequence, by_time in sorted(records.items()):
            complete_times = sorted(time for time, record in by_time.items() if all(field in record["fields"] for field in self.required_fields))
            for start in range(max(0, len(complete_times) - window + 1)):
                input_times = complete_times[start : start + self.n_input_frames]
                output_times = complete_times[start + self.n_input_frames : start + window]
                samples.append({
                    "sequence": sequence,
                    "times_x": input_times,
                    "times_y": output_times,
                    "time_x": input_times[-1],
                    "time_y": output_times[-1],
                    "records_x": [by_time[time]["fields"] for time in input_times],
                    "records_y": [by_time[time]["fields"] for time in output_times],
                })

        if not samples:
            raise ValueError(
                f"No SWIGS/Gorgon MS samples under {self.data_root} had required fields {self.required_fields} "
                "at both input and output timestamps. IS ionosphere files are ignored."
            )

        field_names = [channel for field in self.required_fields for channel in _field_channel_names(field, field_shapes[field])]
        magnetic_indices = [i for i, name in enumerate(field_names) if name.startswith("Bvec_c")]
        return {
            "inspection": inspections,
            "skipped_files": skipped,
            "field_keys": self.required_fields,
            "field_names": field_names,
            "magnetic_field_indices": magnetic_indices if len(magnetic_indices) >= 3 else None,
            "samples": samples,
        }

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

    def _downsample_and_crop(self, arr: np.ndarray) -> np.ndarray:
        if self.crop_shape is not None:
            slices = tuple(slice(0, min(size, arr.shape[axis])) for axis, size in enumerate(self.crop_shape))
            if arr.ndim == 4:
                arr = arr[*slices, :]
            else:
                arr = arr[slices]
        step = self.spatial_downsample
        if step > 1:
            if arr.ndim == 4:
                arr = arr[::step, ::step, ::step, :]
            else:
                arr = arr[::step, ::step, ::step]
        return np.ascontiguousarray(arr)

    def _read_record(self, record: dict[str, Any]) -> torch.Tensor:
        arrays: list[np.ndarray] = []
        for field in self.required_fields:
            spec = record[field]
            with h5py.File(spec["file"], "r") as h5:
                arr = np.asarray(h5[spec["key"]][...], dtype=np.float32)
            arr = self._downsample_and_crop(arr)
            if arr.ndim == 4:
                arr = np.moveaxis(arr, -1, 0)
            else:
                arr = arr[None, ...]
            arrays.append(arr)
        return torch.tensor(np.concatenate(arrays, axis=0), dtype=torch.float32)

    def _estimate_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        vals = [self._read_record(sample["records_x"][0]) for sample in self.samples[: min(4, len(self.samples))]]
        stack = torch.stack(vals)
        dims = (0, 2, 3, 4)
        return stack.mean(dim=dims), stack.std(dim=dims).clamp_min(1e-6)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        x_frames = [self._read_record(record) for record in sample["records_x"]]
        y_frames = [self._read_record(record) for record in sample["records_y"]]
        x = torch.cat(x_frames, dim=0)
        y = torch.cat(y_frames, dim=0)
        if self.normalize and self.mean is not None and self.std is not None:
            mean_x = self.mean.repeat(len(x_frames)).view(-1, 1, 1, 1)
            std_x = self.std.repeat(len(x_frames)).view(-1, 1, 1, 1)
            mean_y = self.mean.repeat(len(y_frames)).view(-1, 1, 1, 1)
            std_y = self.std.repeat(len(y_frames)).view(-1, 1, 1, 1)
            x = (x - mean_x) / std_x
            y = (y - mean_y) / std_y
        return {
            "x": x,
            "y": y,
            "meta": {**sample, "field_names": self.field_names, "field_keys": self.field_keys, "spatial_downsample": self.spatial_downsample},
        }
