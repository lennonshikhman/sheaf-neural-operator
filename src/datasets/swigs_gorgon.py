"""SWIGS/Gorgon HDF5 dataset adapter for cached 3D MHD surrogate modeling."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_REQUIRED_FIELDS = ["P", "Bvec_c"]
DEFAULT_OPTIONAL_FIELDS = ["jvec"]


def inspect_hdf5_file(path: str | Path) -> dict[str, Any]:
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

    Ionosphere ``IS`` files are ignored. Magnetosphere ``MS`` files are grouped
    by sequence and timestamp, required fields must exist at both input and
    output timestamps, and large ``480x320x320`` fields are strided by
    ``downsample_by``/``spatial_downsample`` before channel-first caching.
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
        optional_fields: list[str] | None = None,
        spatial_downsample: int = 4,
        downsample_by: int | None = None,
        crop_shape: list[int] | tuple[int, int, int] | None = None,
        use_cache: bool = True,
        rebuild_cache: bool = False,
        cache_root: str | Path = "datasets/cache",
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.n_input_frames = n_input_frames
        self.n_output_frames = n_output_frames
        self.normalize = normalize
        self.field_mode = field_mode
        self.required_base_fields = required_fields or DEFAULT_REQUIRED_FIELDS
        self.optional_fields = optional_fields if optional_fields is not None else DEFAULT_OPTIONAL_FIELDS
        self.spatial_downsample = max(1, int(downsample_by if downsample_by is not None else spatial_downsample))
        self.crop_shape = tuple(crop_shape) if crop_shape is not None else None
        self.index_path = self.data_root / index_cache_name
        self.use_cache = bool(use_cache)
        self.rebuild_cache = bool(rebuild_cache)
        self.cache_root = Path(cache_root)
        self.cache_dir = self.cache_root / "swigs_gorgon" / split
        self.cache_used = False
        self.cache_rebuilt = False
        self.cached_files: list[Path] = []
        if not self.data_root.exists():
            raise FileNotFoundError(f"SWIGS/Gorgon root not found: {self.data_root}")
        index = self._load_or_build_index()
        self.inspection = index["inspection"]
        self.skipped_files = index.get("skipped_files", [])
        self.field_keys = index["field_keys"]
        self.required_fields = self.field_keys
        self.field_names = index["field_names"]
        self.magnetic_field_indices = index.get("magnetic_field_indices")
        self.inferred_field_mapping = False
        self.samples = self._split_samples(index["samples"], split)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise FileNotFoundError(f"No SWIGS/Gorgon samples for split={split} under {self.data_root}")
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        if self.use_cache:
            self._prepare_cache()
        else:
            self.mean, self.std = self._estimate_stats() if normalize else (None, None)

    def _load_or_build_index(self) -> dict[str, Any]:
        h5_files = sorted([*self.data_root.rglob("*.h5"), *self.data_root.rglob("*.hdf5"), *self.data_root.rglob("*.hdf")])
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found recursively under {self.data_root}")
        signature = {
            "source_files": [str(p) for p in h5_files],
            "field_mode": self.field_mode,
            "required_fields": self.required_base_fields,
            "optional_fields": self.optional_fields,
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
        wanted = list(dict.fromkeys([*self.required_base_fields, *self.optional_fields]))
        wanted_canonical = {_canonical(field): field for field in wanted}
        records: dict[str, dict[float, dict[str, Any]]] = {}
        skipped: list[dict[str, Any]] = []
        field_shapes: dict[str, list[int]] = {}

        for path, inspection in zip(files, inspections, strict=True):
            if not _is_ms_file(path):
                skipped.append({"path": str(path), "reason": "not an MS magnetosphere file"})
                continue
            compatible = {key: value for key, value in inspection["datasets"].items() if _compatible_field_shape(value["shape"])}
            available = {_canonical(key): key for key in compatible}
            present = {field: available[canonical] for canonical, field in wanted_canonical.items() if canonical in available}
            if not any(field in present for field in self.required_base_fields):
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
            complete_times = sorted(time for time, record in by_time.items() if all(field in record["fields"] for field in self.required_base_fields))
            for start in range(max(0, len(complete_times) - window + 1)):
                input_times = complete_times[start : start + self.n_input_frames]
                output_times = complete_times[start + self.n_input_frames : start + window]
                records_x = [by_time[time]["fields"] for time in input_times]
                records_y = [by_time[time]["fields"] for time in output_times]
                consistently_optional = [field for field in self.optional_fields if all(field in rec for rec in [*records_x, *records_y])]
                fields = [*self.required_base_fields, *consistently_optional]
                samples.append({
                    "sequence": sequence,
                    "times_x": input_times,
                    "times_y": output_times,
                    "time_x": input_times[-1],
                    "time_y": output_times[-1],
                    "records_x": records_x,
                    "records_y": records_y,
                    "fields": fields,
                })

        if not samples:
            raise ValueError(f"No SWIGS/Gorgon MS samples under {self.data_root} had required fields {self.required_base_fields} at both input and output timestamps. IS ionosphere files are ignored.")
        effective_fields = [*self.required_base_fields]
        for field in self.optional_fields:
            if all(field in sample["fields"] for sample in samples):
                effective_fields.append(field)
        for sample in samples:
            sample["fields"] = effective_fields
        field_names = [channel for field in effective_fields for channel in _field_channel_names(field, field_shapes[field])]
        magnetic_indices = [i for i, name in enumerate(field_names) if name.startswith("Bvec_c")]
        return {
            "inspection": inspections,
            "skipped_files": skipped,
            "field_keys": effective_fields,
            "field_names": field_names,
            "magnetic_field_indices": magnetic_indices if len(magnetic_indices) >= 3 else None,
            "samples": samples,
        }

    def _cache_signature(self) -> dict[str, Any]:
        return {
            "dataset": "swigs_gorgon",
            "split": self.split,
            "n_input_frames": self.n_input_frames,
            "n_output_frames": self.n_output_frames,
            "max_samples": len(self.samples),
            "normalize": self.normalize,
            "field_keys": self.field_keys,
            "downsample_by": self.spatial_downsample,
            "crop_shape": self.crop_shape,
            "source_files": sorted({spec["file"] for sample in self.samples for rec in [*sample["records_x"], *sample["records_y"]] for spec in rec.values()}),
        }

    def _prepare_cache(self) -> None:
        manifest_path = self.cache_dir / "manifest.json"
        signature = self._cache_signature()
        if self.rebuild_cache and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("signature") == signature:
                files = [self.cache_dir / name for name in manifest.get("samples", [])]
                if files and all(path.exists() for path in files):
                    self.cached_files = files
                    self.mean = torch.tensor(manifest["normalization"]["mean"], dtype=torch.float32) if manifest.get("normalization") else None
                    self.std = torch.tensor(manifest["normalization"]["std"], dtype=torch.float32) if manifest.get("normalization") else None
                    self.cache_used = True
                    return
        self._build_cache(manifest_path, signature)
        self.cache_used = True
        self.cache_rebuilt = True

    def _build_cache(self, manifest_path: Path, signature: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for old in self.cache_dir.glob("sample_*.pt"):
            old.unlink()
        raw_items = [self._assemble_sample(i, normalize=False) for i in range(len(self.samples))]
        if self.normalize:
            c = len(self.field_names)
            stack = torch.stack([item["x"][:c] for item in raw_items[: min(4, len(raw_items))]])
            self.mean = stack.mean(dim=(0, 2, 3, 4))
            self.std = stack.std(dim=(0, 2, 3, 4)).clamp_min(1e-6)
        else:
            self.mean, self.std = None, None
        self.cached_files = []
        for idx, item in enumerate(raw_items):
            if self.normalize and self.mean is not None and self.std is not None:
                item["x"] = self._normalize_frames(item["x"], self.n_input_frames)
                item["y"] = self._normalize_frames(item["y"], self.n_output_frames)
                item["meta"]["normalization"] = {"mean": self.mean.tolist(), "std": self.std.tolist()}
            path = self.cache_dir / f"sample_{idx:08d}.pt"
            torch.save(item, path)
            self.cached_files.append(path)
        manifest = {
            "signature": signature,
            "samples": [path.name for path in self.cached_files],
            "normalization": {"mean": self.mean.tolist(), "std": self.std.tolist()} if self.mean is not None and self.std is not None else None,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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
        original_ndim = arr.ndim
        if self.crop_shape is not None:
            slices = tuple(slice(0, min(size, arr.shape[axis])) for axis, size in enumerate(self.crop_shape))
            arr = arr[(*slices, slice(None))] if original_ndim == 4 else arr[slices]
        step = self.spatial_downsample
        if step > 1:
            arr = arr[::step, ::step, ::step, :] if original_ndim == 4 else arr[::step, ::step, ::step]
        return np.ascontiguousarray(arr)

    def _read_record(self, record: dict[str, Any], fields: list[str]) -> torch.Tensor:
        arrays: list[np.ndarray] = []
        for field in fields:
            spec = record[field]
            with h5py.File(spec["file"], "r") as h5:
                arr = np.asarray(h5[spec["key"]][...], dtype=np.float32)
            arr = self._downsample_and_crop(arr)
            arrays.append(np.moveaxis(arr, -1, 0) if arr.ndim == 4 else arr[None, ...])
        return torch.tensor(np.concatenate(arrays, axis=0), dtype=torch.float32)

    def _estimate_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        vals = [self._read_record(sample["records_x"][0], sample["fields"]) for sample in self.samples[: min(4, len(self.samples))]]
        stack = torch.stack(vals)
        return stack.mean(dim=(0, 2, 3, 4)), stack.std(dim=(0, 2, 3, 4)).clamp_min(1e-6)

    def _normalize_frames(self, tensor: torch.Tensor, n_frames: int) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return tensor
        return (tensor - self.mean.repeat(n_frames).view(-1, 1, 1, 1)) / self.std.repeat(n_frames).view(-1, 1, 1, 1)

    def _assemble_sample(self, idx: int, normalize: bool = True) -> dict[str, Any]:
        sample = self.samples[idx]
        fields = sample["fields"]
        x_frames = [self._read_record(record, fields) for record in sample["records_x"]]
        y_frames = [self._read_record(record, fields) for record in sample["records_y"]]
        x = torch.cat(x_frames, dim=0).float()
        y = torch.cat(y_frames, dim=0).float()
        if normalize and self.normalize and self.mean is not None and self.std is not None:
            x = self._normalize_frames(x, self.n_input_frames)
            y = self._normalize_frames(y, self.n_output_frames)
        first_shape = next(iter(sample["records_x"][0].values()))["shape"]
        return {
            "x": x,
            "y": y,
            "meta": {
                **sample,
                "source_files": sorted({spec["file"] for record in [*sample["records_x"], *sample["records_y"]] for spec in record.values()}),
                "timestamps": {"x": sample["times_x"], "y": sample["times_y"]},
                "field_names": self.field_names,
                "field_keys": self.field_keys,
                "channel_order": self.field_names,
                "original_shape": first_shape[:3],
                "cached_shape": {"x": list(x.shape), "y": list(y.shape)},
                "downsample_by": self.spatial_downsample,
                "crop_shape": self.crop_shape,
            },
        }

    def __len__(self) -> int:
        return len(self.cached_files) if self.cache_used else len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.cache_used:
            return torch.load(self.cached_files[idx], map_location="cpu", weights_only=False)
        if self.mean is None and self.normalize:
            self.mean, self.std = self._estimate_stats()
        return self._assemble_sample(idx, normalize=True)
