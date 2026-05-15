"""Minimal Fourier layers for 3D FNO baselines."""
from __future__ import annotations

import torch
from torch import nn


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / max(1, in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, d, h, w = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(b, self.out_channels, d, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        md = min(self.modes, d)
        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)
        out_ft[:, :, :md, :mh, :mw] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :md, :mh, :mw], self.weights[:, :, :md, :mh, :mw])
        return torch.fft.irfftn(out_ft, s=(d, h, w), dim=(-3, -2, -1))
