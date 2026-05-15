"""Orszag-Tang FARGO3D processed dataset loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FIELDS = ["density", "vy", "vz", "by", "bz"]


class OrszagTangDataset(Dataset):
    """Loads samples as x=[5*n_input_frames,H,W], y=[5,H,W].

    The loader supports split subdirectories, flat processed field directories,
    .npy/.npz/.h5/.hdf5 files, and deterministic train/valid partitioning when no
    validation split is present.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        n_input_frames: int = 1,
        target_frame: int = 1,
        max_samples: int | None = None,
        normalize: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.n_input_frames = n_input_frames
        self.target_frame = target_frame
        self.normalize = normalize
        self.base = self.data_root / "input_data"
        if not self.base.exists():
            raise FileNotFoundError(f"Orszag-Tang input_data directory not found: {self.base}")
        self.files = self._discover_files(split)
        if max_samples is not None:
            self.files = self.files[:max_samples]
        if not self.files:
            raise FileNotFoundError(f"No Orszag-Tang samples found for split={split} under {self.base}")
        self.mean, self.std = self._estimate_stats() if normalize else (None, None)

    def _field_files(self, field: str) -> list[Path]:
        root = self.base / field
        patterns = ("*.npy", "*.npz", "*.h5", "*.hdf5")
        out: list[Path] = []
        for pattern in patterns:
            out.extend(root.rglob(pattern))
        return sorted(out)

    def _discover_files(self, split: str) -> list[dict[str, Path]]:
        per_field = {field: self._field_files(field) for field in FIELDS}
        n = min((len(v) for v in per_field.values()), default=0)
        samples = [{field: per_field[field][i] for field in FIELDS} for i in range(n)]

        def has_split(sample: dict[str, Path], name: str) -> bool:
            return any(name in path.parts for path in sample.values())

        explicit = [sample for sample in samples if has_split(sample, split)]
        if explicit:
            return explicit
        train_like = [sample for sample in samples if not has_split(sample, "test")]
        test_like = [sample for sample in samples if has_split(sample, "test")]
        if split == "test" and test_like:
            return test_like
        if split in {"train", "valid"}:
            pool = train_like or samples
            cut = max(1, int(0.8 * len(pool)))
            return pool[:cut] if split == "train" else pool[cut:]
        return samples

    def _load_array(self, path: Path) -> np.ndarray:
        if path.suffix == ".npy":
            arr = np.load(path)
        elif path.suffix == ".npz":
            z = np.load(path)
            arr = z[z.files[0]]
        else:
            import h5py

            with h5py.File(path, "r") as h5:
                datasets = []
                h5.visititems(lambda _name, obj: datasets.append(obj) if hasattr(obj, "shape") else None)
                if not datasets:
                    raise ValueError(f"No array-like dataset found inside {path}")
                arr = datasets[0][...]
        return np.asarray(arr, dtype=np.float32)

    def _frames(self, arr: np.ndarray) -> np.ndarray:
        """Normalize arrays to [T,H,W] with strict shape validation."""
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return arr[None]
        if arr.ndim != 3:
            raise ValueError(f"Cannot infer Orszag-Tang array shape {arr.shape}; expected [H,W], [T,H,W], or singleton-channel variants")
        # If one axis is a singleton channel after partial squeeze, move/strip it; otherwise assume [T,H,W].
        if 1 in arr.shape:
            squeezed = np.squeeze(arr)
            if squeezed.ndim == 2:
                return squeezed[None]
            if squeezed.ndim == 3:
                arr = squeezed
        # Heuristic guard: the time axis should usually be the smallest axis. If a channel-last [H,W,T]
        # layout is detected, transpose it to [T,H,W].
        if arr.shape[-1] < arr.shape[0] and arr.shape[-1] <= 512:
            arr = np.moveaxis(arr, -1, 0)
        return arr

    def _estimate_stats(self):
        stack = []
        for i in range(min(len(self.files), 16)):
            fields = [self._frames(self._load_array(self.files[i][field]))[-1] for field in FIELDS]
            stack.append(np.stack(fields))
        a = np.stack(stack)
        return torch.tensor(a.mean(axis=(0, 2, 3)), dtype=torch.float32), torch.tensor(a.std(axis=(0, 2, 3)) + 1e-6, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        sample = self.files[idx]
        xs, ys = [], []
        for field in FIELDS:
            frames = self._frames(self._load_array(sample[field]))
            if frames.shape[0] < self.n_input_frames + self.target_frame:
                xframes = np.repeat(frames[-1:], self.n_input_frames, axis=0)
                yframe = frames[-1]
            else:
                xframes = frames[-self.target_frame - self.n_input_frames : -self.target_frame]
                yframe = frames[-self.target_frame]
            xs.extend(list(xframes))
            ys.append(yframe)
        x = torch.tensor(np.stack(xs), dtype=torch.float32)
        y = torch.tensor(np.stack(ys), dtype=torch.float32)
        if self.normalize and self.mean is not None and self.std is not None:
            mean_x = self.mean.repeat_interleave(self.n_input_frames).view(-1, 1, 1)
            std_x = self.std.repeat_interleave(self.n_input_frames).view(-1, 1, 1)
            x = (x - mean_x) / std_x
            y = (y - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)
        return {"x": x, "y": y, "meta": {"fields": FIELDS, "split": self.split, "paths": {k: str(v) for k, v in sample.items()}}}
