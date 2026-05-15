"""The Well MHD_64 HDF5 dataset adapter with optional preprocessed sample cache."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class WellMHD64Dataset(Dataset):
    """Read local The Well ``MHD_64`` files as channel-first 3D states.

    The local HDF5 layout stores density under ``t0_fields`` and vector fields
    under ``t1_fields``::

        t0_fields/density        [n_traj, time, x, y, z]
        t1_fields/magnetic_field [n_traj, time, x, y, z, 3]
        t1_fields/velocity       [n_traj, time, x, y, z, 3]

    Each one-frame state is returned as seven channels ordered as
    ``density, Bx, By, Bz, vx, vy, vz``. Input frames are folded into the
    channel axis, so cached/default samples have shape
    ``x=[7*n_input_frames, 48, 48, 48]`` and ``y=[7*n_output_frames, 48, 48, 48]``
    when ``crop_size=48``.
    """

    field_names = ["density", "bx", "by", "bz", "vx", "vy", "vz"]
    field_keys = ["t0_fields/density", "t1_fields/magnetic_field", "t1_fields/velocity"]

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        n_input_frames: int = 1,
        n_output_frames: int = 1,
        max_samples: int | None = None,
        normalize: bool = True,
        channels: list[int] | None = None,
        magnetic_field_indices: list[int] | None = None,
        crop_size: int | None = 48,
        use_cache: bool = True,
        rebuild_cache: bool = False,
        cache_root: str | Path = "datasets/cache",
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.n_input_frames = n_input_frames
        self.n_output_frames = n_output_frames
        self.normalize = normalize
        self.channels = channels
        self.magnetic_field_indices = magnetic_field_indices or [1, 2, 3]
        self.crop_size = int(crop_size) if crop_size else None
        self.use_cache = bool(use_cache)
        self.rebuild_cache = bool(rebuild_cache)
        self.cache_root = Path(cache_root)
        self.cache_dir = self.cache_root / "wells_mhd64" / split
        self.cache_used = False
        self.cache_rebuilt = False
        if not self.data_root.exists():
            raise FileNotFoundError(f"The Well root not found: {self.data_root}")
        self.well_base_path = self._resolve_well_base_path()
        self.split_dir = self.well_base_path / "MHD_64" / "data" / split
        self.samples = self._build_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise FileNotFoundError(f"No The Well MHD_64 samples for split={split} under {self.split_dir}")
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.cached_files: list[Path] = []
        if self.use_cache:
            self._prepare_cache()
        else:
            self.mean, self.std = self._estimate_stats() if normalize else (None, None)

    def _resolve_well_base_path(self) -> Path:
        candidates = [
            self.data_root,
            self.data_root / "datasets",
            self.data_root.parent if self.data_root.name == "MHD_64" else self.data_root,
        ]
        for candidate in candidates:
            split_dir = candidate / "MHD_64" / "data" / self.split
            if split_dir.is_dir() and any(split_dir.glob("*.hdf5")):
                return candidate
        expected = " or ".join(str(c / "MHD_64" / "data" / self.split) for c in candidates[:2])
        raise FileNotFoundError(f"No The Well MHD_64 HDF5 files found; expected files under {expected}")

    def _cache_signature(self) -> dict[str, Any]:
        return {
            "dataset": "wells_mhd64",
            "split": self.split,
            "n_input_frames": self.n_input_frames,
            "n_output_frames": self.n_output_frames,
            "max_samples": len(self.samples),
            "normalize": self.normalize,
            "channels": self.channels,
            "crop_size": self.crop_size,
            "field_keys": self.field_keys,
            "source_files": sorted({s["file"] for s in self.samples}),
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
        raw_items: list[dict[str, Any]] = [self._assemble_sample(i, normalize=False) for i in range(len(self.samples))]
        if self.normalize:
            stack = torch.stack([item["x"][: len(self.field_names)] for item in raw_items[: min(8, len(raw_items))]])
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

    def _build_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        window = self.n_input_frames + self.n_output_frames
        for path in sorted(self.split_dir.glob("*.hdf5")):
            with h5py.File(path, "r") as h5:
                missing = [key for key in self.field_keys if key not in h5]
                if missing:
                    raise KeyError(f"The Well file {path} is missing required datasets: {missing}")
                density_shape = h5["t0_fields/density"].shape
                magnetic_shape = h5["t1_fields/magnetic_field"].shape
                velocity_shape = h5["t1_fields/velocity"].shape
                if len(density_shape) != 5 or len(magnetic_shape) != 6 or len(velocity_shape) != 6:
                    raise ValueError(f"Unexpected The Well shapes in {path}: density={density_shape}, magnetic_field={magnetic_shape}, velocity={velocity_shape}")
                if magnetic_shape[-1] != 3 or velocity_shape[-1] != 3:
                    raise ValueError(f"Expected vector fields with trailing size 3 in {path}")
                n_traj = min(density_shape[0], magnetic_shape[0], velocity_shape[0])
                n_time = min(density_shape[1], magnetic_shape[1], velocity_shape[1])
                original_shape = list(density_shape[2:5])
            for trajectory in range(n_traj):
                for start_time in range(max(0, n_time - window + 1)):
                    samples.append({"file": str(path), "trajectory": trajectory, "start_time": start_time, "original_shape": original_shape})
        return samples

    def _crop_slices(self, shape: tuple[int, int, int], idx: int) -> tuple[tuple[slice, slice, slice], dict[str, Any]]:
        if self.crop_size is None:
            starts = (0, 0, 0)
            sizes = shape
        else:
            sizes = tuple(min(self.crop_size, dim) for dim in shape)
            if self.split == "train":
                rng = random.Random(idx)
                starts = tuple(rng.randint(0, max(0, dim - size)) for dim, size in zip(shape, sizes, strict=True))
            else:
                starts = tuple(max(0, (dim - size) // 2) for dim, size in zip(shape, sizes, strict=True))
        stops = tuple(start + size for start, size in zip(starts, sizes, strict=True))
        return tuple(slice(start, stop) for start, stop in zip(starts, stops, strict=True)), {"crop_size": self.crop_size, "crop_starts": list(starts), "crop_stops": list(stops), "crop_mode": "random" if self.split == "train" else "center"}

    def _read_state(self, path: str | Path, trajectory: int, time_index: int, crop: tuple[slice, slice, slice] | None = None) -> torch.Tensor:
        with h5py.File(path, "r") as h5:
            if crop is None:
                density = np.asarray(h5["t0_fields/density"][trajectory, time_index], dtype=np.float32)[None, ...]
                magnetic = np.asarray(h5["t1_fields/magnetic_field"][trajectory, time_index], dtype=np.float32)
                velocity = np.asarray(h5["t1_fields/velocity"][trajectory, time_index], dtype=np.float32)
            else:
                density = np.asarray(h5["t0_fields/density"][(trajectory, time_index, *crop)], dtype=np.float32)[None, ...]
                magnetic = np.asarray(h5["t1_fields/magnetic_field"][(trajectory, time_index, *crop, slice(None))], dtype=np.float32)
                velocity = np.asarray(h5["t1_fields/velocity"][(trajectory, time_index, *crop, slice(None))], dtype=np.float32)
        channels = np.concatenate([density, np.moveaxis(magnetic, -1, 0), np.moveaxis(velocity, -1, 0)], axis=0)
        tensor = torch.tensor(channels, dtype=torch.float32)
        if self.channels is not None:
            tensor = tensor[self.channels]
        return tensor

    def _estimate_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        states = []
        for idx, sample in enumerate(self.samples[: min(8, len(self.samples))]):
            crop, _ = self._crop_slices(tuple(sample["original_shape"]), idx)
            states.append(self._read_state(sample["file"], sample["trajectory"], sample["start_time"], crop))
        stack = torch.stack(states)
        return stack.mean(dim=(0, 2, 3, 4)), stack.std(dim=(0, 2, 3, 4)).clamp_min(1e-6)

    def _normalize_frames(self, tensor: torch.Tensor, n_frames: int) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return tensor
        mean = self.mean.repeat(n_frames).view(-1, 1, 1, 1)
        std = self.std.repeat(n_frames).view(-1, 1, 1, 1)
        return (tensor - mean) / std

    def _assemble_sample(self, idx: int, normalize: bool = True) -> dict[str, Any]:
        sample = self.samples[idx]
        crop, crop_meta = self._crop_slices(tuple(sample["original_shape"]), idx)
        start = sample["start_time"]
        x_frames = [self._read_state(sample["file"], sample["trajectory"], start + offset, crop) for offset in range(self.n_input_frames)]
        y_frames = [self._read_state(sample["file"], sample["trajectory"], start + self.n_input_frames + offset, crop) for offset in range(self.n_output_frames)]
        x = torch.cat(x_frames, dim=0).float()
        y = torch.cat(y_frames, dim=0).float()
        if normalize and self.normalize and self.mean is not None and self.std is not None:
            x = self._normalize_frames(x, self.n_input_frames)
            y = self._normalize_frames(y, self.n_output_frames)
        return {
            "x": x,
            "y": y,
            "meta": {
                **sample,
                "split": self.split,
                "source_files": [sample["file"]],
                "timestamps": list(range(start, start + self.n_input_frames + self.n_output_frames)),
                "field_names": self.field_names,
                "field_keys": self.field_keys,
                "channel_order": self.field_names,
                "magnetic_field_indices": self.magnetic_field_indices,
                "original_shape": sample["original_shape"],
                "cached_shape": {"x": list(x.shape), "y": list(y.shape)},
                **crop_meta,
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
