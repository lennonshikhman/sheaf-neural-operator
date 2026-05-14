"""Orszag-Tang FARGO3D processed dataset loader."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

FIELDS = ["density", "vy", "vz", "by", "bz"]

class OrszagTangDataset(Dataset):
    """Loads samples as x=[5*n_input_frames,H,W], y=[5,H,W].

    The loader supports split subdirectories, flat processed field directories, .npy/.npz/.h5 files,
    and deterministic train/valid partitioning when no validation split is present.
    """
    def __init__(self, data_root: str | Path, split: str, n_input_frames: int = 1, target_frame: int = 1,
                 max_samples: int | None = None, normalize: bool = True) -> None:
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
        pats = ["*.npy", "*.npz", "*.h5", "*.hdf5"]
        out: list[Path] = []
        for p in pats:
            out.extend(root.rglob(p))
        return sorted(out)

    def _discover_files(self, split: str) -> list[dict[str, Path]]:
        per_field = {f: self._field_files(f) for f in FIELDS}
        n = min(len(v) for v in per_field.values()) if per_field else 0
        samples = [{f: per_field[f][i] for f in FIELDS} for i in range(n)]
        def has_split(sample, name): return any(name in p.parts for p in sample.values())
        explicit = [s for s in samples if has_split(s, split)]
        if explicit:
            return explicit
        train_like = [s for s in samples if not has_split(s, "test")]
        test_like = [s for s in samples if has_split(s, "test")]
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
            z = np.load(path); arr = z[z.files[0]]
        else:
            import h5py
            with h5py.File(path, "r") as h5:
                key = next(k for k in h5.keys() if hasattr(h5[k], "shape"))
                arr = h5[key][...]
        return np.asarray(arr, dtype=np.float32)

    def _frames(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = arr[None]
        elif arr.ndim == 3:
            # Assume [T,H,W]; if a singleton channel is present squeeze it.
            pass
        elif arr.ndim == 4 and 1 in (arr.shape[0], arr.shape[-1]):
            arr = np.squeeze(arr)
            if arr.ndim == 2: arr = arr[None]
        else:
            raise ValueError(f"Cannot infer Orszag-Tang array shape {arr.shape}; expected [H,W] or [T,H,W]")
        return arr

    def _estimate_stats(self):
        stack = []
        for i in range(min(len(self.files), 16)):
            y = [self._frames(self._load_array(self.files[i][f]))[-1] for f in FIELDS]
            stack.append(np.stack(y))
        a = np.stack(stack)
        return torch.tensor(a.mean(axis=(0,2,3)), dtype=torch.float32), torch.tensor(a.std(axis=(0,2,3)) + 1e-6, dtype=torch.float32)

    def __len__(self) -> int: return len(self.files)

    def __getitem__(self, idx: int):
        sample = self.files[idx]
        xs, ys = [], []
        for ci, f in enumerate(FIELDS):
            frames = self._frames(self._load_array(sample[f]))
            if frames.shape[0] < self.n_input_frames + self.target_frame:
                xframes = np.repeat(frames[-1:], self.n_input_frames, axis=0)
                yframe = frames[-1]
            else:
                xframes = frames[-self.target_frame-self.n_input_frames:-self.target_frame]
                yframe = frames[-self.target_frame]
            xs.extend(list(xframes)); ys.append(yframe)
        x = torch.tensor(np.stack(xs), dtype=torch.float32)
        y = torch.tensor(np.stack(ys), dtype=torch.float32)
        if self.normalize and self.mean is not None:
            mean_x = self.mean.repeat_interleave(self.n_input_frames).view(-1,1,1)
            std_x = self.std.repeat_interleave(self.n_input_frames).view(-1,1,1)
            x = (x - mean_x) / std_x
            y = (y - self.mean.view(-1,1,1)) / self.std.view(-1,1,1)
        return {"x": x, "y": y, "meta": {"fields": FIELDS, "split": self.split, "paths": {k: str(v) for k,v in sample.items()}}}
