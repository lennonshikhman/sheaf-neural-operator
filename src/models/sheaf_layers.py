"""Sheaf-style local fiber layers and learned restriction maps."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .spectral_layers import SpectralConv2d, SpectralConv3d


def conv(dim: int):
    return nn.Conv2d if dim == 2 else nn.Conv3d


class RestrictionMap(nn.Module):
    """Learned 1x1 sheaf restriction/coupling map between local fibers."""

    def __init__(self, dim: int, in_channels: int, out_channels: int):
        super().__init__()
        self.map = conv(dim)(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.map(x)


class FiberOperator(nn.Module):
    """Local operator on one fiber, implemented either as CNN or FNO."""

    def __init__(self, dim: int, channels: int, backbone_type: str = "cnn", modes: int = 16):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        C = conv(dim)
        if self.backbone_type == "cnn":
            self.net = nn.Sequential(
                C(channels, channels, 3, padding=1),
                nn.GELU(),
                C(channels, channels, 3, padding=1),
            )
            self.pointwise = None
        elif self.backbone_type == "fno":
            spectral = SpectralConv2d if dim == 2 else SpectralConv3d
            self.net = spectral(channels, channels, modes)
            self.pointwise = C(channels, channels, 1)
        else:
            raise ValueError(f"Unknown sheaf fiber backbone_type={backbone_type!r}; expected 'cnn' or 'fno'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pointwise is None:
            return self.net(x)
        return self.net(x) + self.pointwise(x)


class SheafMessageBlock(nn.Module):
    """Updates fluid and magnetic fibers with local operators plus cross-fiber restrictions."""

    def __init__(self, dim: int, channels: int, use_restriction_maps: bool = True, backbone_type: str = "cnn", modes: int = 16):
        super().__init__()
        self.use_restriction_maps = use_restriction_maps
        self.fluid_local = FiberOperator(dim, channels, backbone_type, modes)
        self.magnetic_local = FiberOperator(dim, channels, backbone_type, modes)
        self.f2m = RestrictionMap(dim, channels, channels)
        self.m2f = RestrictionMap(dim, channels, channels)
        self.norm_f = nn.GroupNorm(1, channels)
        self.norm_m = nn.GroupNorm(1, channels)

    def forward(self, fluid: torch.Tensor, magnetic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_msg = self.fluid_local(fluid)
        m_msg = self.magnetic_local(magnetic)
        if self.use_restriction_maps:
            f_msg = f_msg + self.m2f(magnetic)
            m_msg = m_msg + self.f2m(fluid)
        return F.gelu(self.norm_f(fluid + f_msg)), F.gelu(self.norm_m(magnetic + m_msg))
