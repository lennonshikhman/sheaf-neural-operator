from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .spectral_layers import SpectralConv2d

class FNO2D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, hidden_channels:int=64, num_layers:int=4, modes:int=16):
        super().__init__(); self.lift=nn.Conv2d(in_channels, hidden_channels, 1)
        self.spec=nn.ModuleList([SpectralConv2d(hidden_channels, hidden_channels, modes) for _ in range(num_layers)])
        self.pw=nn.ModuleList([nn.Conv2d(hidden_channels, hidden_channels, 1) for _ in range(num_layers)])
        self.proj=nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels,1), nn.GELU(), nn.Conv2d(hidden_channels,out_channels,1))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.lift(x)
        for s,w in zip(self.spec,self.pw): x=F.gelu(s(x)+w(x))
        return self.proj(x)
