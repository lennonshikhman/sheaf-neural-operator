"""Sheaf-style local fiber layers and learned restriction maps."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def conv(dim:int): return nn.Conv2d if dim==2 else nn.Conv3d

class RestrictionMap(nn.Module):
    """Learned 1x1 sheaf restriction/coupling map between local fibers."""
    def __init__(self, dim:int, in_channels:int, out_channels:int):
        super().__init__(); self.map = conv(dim)(in_channels, out_channels, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.map(x)

class SheafMessageBlock(nn.Module):
    """Updates fluid and magnetic fibers with local operators plus cross-fiber restrictions."""
    def __init__(self, dim:int, channels:int, use_restriction_maps:bool=True):
        super().__init__(); C=conv(dim); self.use_restriction_maps=use_restriction_maps
        self.fluid_local=nn.Sequential(C(channels,channels,3,padding=1), nn.GELU(), C(channels,channels,3,padding=1))
        self.magnetic_local=nn.Sequential(C(channels,channels,3,padding=1), nn.GELU(), C(channels,channels,3,padding=1))
        self.f2m=RestrictionMap(dim, channels, channels); self.m2f=RestrictionMap(dim, channels, channels)
        self.norm_f=nn.GroupNorm(1, channels); self.norm_m=nn.GroupNorm(1, channels)
    def forward(self, fluid: torch.Tensor, magnetic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f_msg=self.fluid_local(fluid); m_msg=self.magnetic_local(magnetic)
        if self.use_restriction_maps:
            f_msg = f_msg + self.m2f(magnetic); m_msg = m_msg + self.f2m(fluid)
        return F.gelu(self.norm_f(fluid + f_msg)), F.gelu(self.norm_m(magnetic + m_msg))
