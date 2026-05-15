from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _Block3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 32, **_: object):
        super().__init__()
        h = hidden_channels
        self.pool = nn.MaxPool3d(2)
        self.e1 = _Block3D(in_channels, h)
        self.e2 = _Block3D(h, 2 * h)
        self.mid = _Block3D(2 * h, 4 * h)
        self.u2 = nn.Conv3d(4 * h, 2 * h, 1)
        self.d2 = _Block3D(4 * h, 2 * h)
        self.u1 = nn.Conv3d(2 * h, h, 1)
        self.d1 = _Block3D(2 * h, h)
        self.out = nn.Conv3d(h, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        m = self.mid(self.pool(e2))
        u2 = F.interpolate(self.u2(m), size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(self.u1(d2), size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], dim=1))
        return self.out(d1)
