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
        orig_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            x_ft = torch.fft.rfftn(x_fp32, dim=(-3, -2, -1)).to(torch.complex64)
            out_ft = torch.zeros(b, self.out_channels, d, h, w // 2 + 1, dtype=torch.complex64, device=x.device)
            md = min(self.modes, d)
            mh = min(self.modes, h)
            mw = min(self.modes, w // 2 + 1)
            weights = self.weights[:, :, :md, :mh, :mw].to(torch.complex64)
            out_ft[:, :, :md, :mh, :mw] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :md, :mh, :mw], weights)
            out = torch.fft.irfftn(out_ft, s=(d, h, w), dim=(-3, -2, -1))
        return out.to(orig_dtype) if orig_dtype in (torch.float16, torch.bfloat16) else out
