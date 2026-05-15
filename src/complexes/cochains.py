from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CochainBatch:
    """Batched cochain features by dimension, each [B, n_k, channels]."""

    features: dict[int, torch.Tensor]
    metadata: dict | None = None

    def __getitem__(self, dim: int) -> torch.Tensor:
        return self.features[dim]

    def get(self, dim: int, default=None):
        return self.features.get(dim, default)

    def to(self, device: torch.device | str) -> "CochainBatch":
        return CochainBatch({k: v.to(device) for k, v in self.features.items()}, self.metadata)
