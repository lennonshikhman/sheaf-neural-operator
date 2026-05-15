"""MLP models for ConStellaration equilibrium surrogate experiments."""
from __future__ import annotations

import torch
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 256, num_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        width = in_channels
        for _ in range(max(1, num_layers)):
            layers.extend([nn.Linear(width, hidden_channels), nn.GELU(), nn.LayerNorm(hidden_channels)])
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = hidden_channels
        layers.append(nn.Linear(width, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SheafEquilibriumMLP(nn.Module):
    """Simple sheaf-style equilibrium MLP with geometry/profile/metric fibers."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 256, num_layers: int = 4):
        super().__init__()
        self.in_channels = in_channels
        splits = torch.linspace(0, in_channels, steps=4).long().tolist()
        self.slices = [(splits[i], splits[i + 1]) for i in range(3)]
        self.fiber_lifts = nn.ModuleList([nn.Linear(max(1, b - a), hidden_channels) for a, b in self.slices])
        self.restrictions = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(6)])
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.GELU(), nn.LayerNorm(hidden_channels)) for _ in range(3)])
                for _ in range(max(1, num_layers))
            ]
        )
        self.head = nn.Sequential(nn.Linear(3 * hidden_channels, hidden_channels), nn.GELU(), nn.Linear(hidden_channels, out_channels))

    def _slice(self, x: torch.Tensor, a: int, b: int) -> torch.Tensor:
        part = x[:, a:b]
        if part.shape[1] == 0:
            return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        return part

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fibers = [lift(self._slice(x, a, b)) for lift, (a, b) in zip(self.fiber_lifts, self.slices, strict=False)]
        for block in self.blocks:
            old = fibers
            coupled = []
            r = iter(self.restrictions)
            for i in range(3):
                msg = block[i](old[i])
                for j in range(3):
                    if i != j:
                        msg = msg + next(r)(old[j])
                coupled.append(msg)
            fibers = coupled
        return self.head(torch.cat(fibers, dim=1))
