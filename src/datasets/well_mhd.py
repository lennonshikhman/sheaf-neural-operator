"""The Well MHD_64 HDF5 dataset adapter."""
from __future__ import annotations

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
    channel axis, so ``x`` has shape ``[7*n_input_frames, X, Y, Z]`` and ``y``
    has shape ``[7*n_output_frames, X, Y, Z]``.
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
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.n_input_frames = n_input_frames
        self.n_output_frames = n_output_frames
        self.normalize = normalize
        self.channels = channels
        self.magnetic_field_indices = magnetic_field_indices or [1, 2, 3]
        if not self.data_root.exists():
            raise FileNotFoundError(f"The Well root not found: {self.data_root}")
        self.well_base_path = self._resolve_well_base_path()
        self.split_dir = self.well_base_path / "MHD_64" / "data" / split
        self.samples = self._build_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise FileNotFoundError(f"No The Well MHD_64 samples for split={split} under {self.split_dir}")
        self.mean, self.std = self._estimate_stats() if normalize else (None, None)

    def _resolve_well_base_path(self) -> Path:
        """Return the directory that contains the ``MHD_64`` dataset folder."""
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
                    raise ValueError(
                        f"Unexpected The Well shapes in {path}: density={density_shape}, "
                        f"magnetic_field={magnetic_shape}, velocity={velocity_shape}"
                    )
                if magnetic_shape[-1] != 3 or velocity_shape[-1] != 3:
                    raise ValueError(f"Expected vector fields with trailing size 3 in {path}")
                n_traj = min(density_shape[0], magnetic_shape[0], velocity_shape[0])
                n_time = min(density_shape[1], magnetic_shape[1], velocity_shape[1])
            for trajectory in range(n_traj):
                for start_time in range(max(0, n_time - window + 1)):
                    samples.append({"file": str(path), "trajectory": trajectory, "start_time": start_time})
        return samples

    def _read_state(self, path: str | Path, trajectory: int, time_index: int) -> torch.Tensor:
        with h5py.File(path, "r") as h5:
            density = np.asarray(h5["t0_fields/density"][trajectory, time_index], dtype=np.float32)[None, ...]
            magnetic = np.asarray(h5["t1_fields/magnetic_field"][trajectory, time_index], dtype=np.float32)
            velocity = np.asarray(h5["t1_fields/velocity"][trajectory, time_index], dtype=np.float32)
        channels = np.concatenate([density, np.moveaxis(magnetic, -1, 0), np.moveaxis(velocity, -1, 0)], axis=0)
        tensor = torch.tensor(channels, dtype=torch.float32)
        if self.channels is not None:
            tensor = tensor[self.channels]
        return tensor

    def _estimate_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        states = []
        for sample in self.samples[: min(8, len(self.samples))]:
            states.append(self._read_state(sample["file"], sample["trajectory"], sample["start_time"]))
        stack = torch.stack(states)
        dims = (0, 2, 3, 4)
        return stack.mean(dim=dims), stack.std(dim=dims).clamp_min(1e-6)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        start = sample["start_time"]
        x_frames = [self._read_state(sample["file"], sample["trajectory"], start + offset) for offset in range(self.n_input_frames)]
        y_frames = [
            self._read_state(sample["file"], sample["trajectory"], start + self.n_input_frames + offset)
            for offset in range(self.n_output_frames)
        ]
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
            "meta": {
                **sample,
                "split": self.split,
                "field_names": self.field_names,
                "field_keys": self.field_keys,
                "magnetic_field_indices": self.magnetic_field_indices,
            },
        }
