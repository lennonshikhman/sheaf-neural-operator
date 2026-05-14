"""The Well MHD_64 dataset adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset

class WellMHD64Dataset(Dataset):
    """Channel-first adapter for ``the_well.data.WellDataset``.

    Internal 3D convention is x=[C_in,X,Y,Z], y=[C_out,X,Y,Z]. The adapter inspects
    tensors returned by The Well and folds input time into channels.
    """
    def __init__(self, data_root: str | Path, split: str, n_input_frames: int = 1, n_output_frames: int = 1,
                 max_samples: int | None = None, normalize: bool = True, channels: list[int] | None = None,
                 magnetic_field_indices: list[int] | None = None) -> None:
        self.data_root = Path(data_root); self.split = split; self.n_input_frames = n_input_frames
        self.n_output_frames = n_output_frames; self.max_samples = max_samples; self.normalize = normalize
        self.channels = channels; self.magnetic_field_indices = magnetic_field_indices
        if not self.data_root.exists():
            raise FileNotFoundError(f"The Well root not found: {self.data_root}")
        try:
            from the_well.data import WellDataset
        except ImportError as exc:
            raise ImportError("Install the_well to load MHD_64: pip install the_well") from exc
        self.ds = WellDataset(well_base_path=str(self.data_root), well_dataset_name="MHD_64", well_split_name=split)
        self._len = len(self.ds) if max_samples is None else min(len(self.ds), max_samples)
        self.mean = None; self.std = None

    def __len__(self) -> int: return self._len

    def _extract_tensor(self, item: Any) -> torch.Tensor:
        if isinstance(item, torch.Tensor):
            return item.float()
        if isinstance(item, dict):
            for key in ("x", "input", "data", "fields", "u", "trajectory"):
                if key in item and torch.is_tensor(item[key]):
                    return item[key].float()
            tensors = [v for v in item.values() if torch.is_tensor(v)]
            if tensors:
                return tensors[0].float()
        if isinstance(item, (tuple, list)):
            tensors = [v for v in item if torch.is_tensor(v)]
            if tensors:
                return tensors[0].float()
        raise TypeError(f"Could not find tensor in WellDataset item of type {type(item)}")

    def _to_time_channel_grid(self, u: torch.Tensor) -> torch.Tensor:
        # Accept common layouts [T,C,X,Y,Z], [T,X,Y,Z,C], [C,T,X,Y,Z], [X,Y,Z,C], [C,X,Y,Z].
        if u.ndim == 4:
            # Static sample. Infer channel axis as smallest dimension unless first already channel-like.
            c_axis = 0 if u.shape[0] <= 32 else -1
            if c_axis == -1: u = u.permute(3,0,1,2)
            u = u.unsqueeze(0)  # [T=1,C,X,Y,Z]
        elif u.ndim == 5:
            shapes = list(u.shape)
            small_axes = [i for i,s in enumerate(shapes) if s <= 64]
            if len(small_axes) < 2:
                raise ValueError(f"Cannot infer time/channel axes for Well tensor shape {tuple(u.shape)}")
            # Prefer time first if present; channel is the smallest non-time axis.
            t_axis = 0
            c_candidates = [i for i in small_axes if i != t_axis]
            c_axis = min(c_candidates, key=lambda i: shapes[i]) if c_candidates else 1
            grid_axes = [i for i in range(5) if i not in (t_axis, c_axis)]
            u = u.permute(t_axis, c_axis, *grid_axes)
        else:
            raise ValueError(f"Expected 4D/5D Well tensor, got shape {tuple(u.shape)}")
        if self.channels is not None:
            u = u[:, self.channels]
        if u.ndim != 5:
            raise ValueError(f"Well tensor normalization failed, got {tuple(u.shape)}")
        return u.contiguous()

    def __getitem__(self, idx: int):
        raw = self.ds[idx]
        u = self._to_time_channel_grid(self._extract_tensor(raw))
        if u.shape[0] < self.n_input_frames + self.n_output_frames:
            raise ValueError(f"Well sample has {u.shape[0]} frames but needs {self.n_input_frames+self.n_output_frames}")
        x = u[:self.n_input_frames].reshape(-1, *u.shape[2:])
        y = u[self.n_input_frames:self.n_input_frames+self.n_output_frames].reshape(-1, *u.shape[2:])
        return {"x": x, "y": y, "meta": {"split": self.split, "magnetic_field_indices": self.magnetic_field_indices}}
