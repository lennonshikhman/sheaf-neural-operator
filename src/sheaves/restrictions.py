from __future__ import annotations

import torch
from torch import nn


class SharedIncidenceRestriction(nn.Module):
    """Shared learned restriction/message map for one incidence type.

    Option C: a shared linear map per incidence type with optional diagonal
    geometry conditioning.  The signed sparse incidence performs topology; this
    module transforms the transported fiber values.
    """

    def __init__(self, channels: int, use_geometry_conditioning: bool = False):
        super().__init__()
        self.linear = nn.Linear(channels, channels)
        self.use_geometry_conditioning = use_geometry_conditioning
        self.geometry_gate = nn.Linear(1, channels) if use_geometry_conditioning else None

    def forward(self, x: torch.Tensor, geometry_weight: torch.Tensor | None = None) -> torch.Tensor:
        y = self.linear(x)
        if self.geometry_gate is not None and geometry_weight is not None:
            gate = torch.sigmoid(self.geometry_gate(geometry_weight.to(device=x.device, dtype=x.dtype).view(1, -1, 1)))
            y = y * gate
        return y
