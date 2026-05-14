from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def _conv(dim:int): return nn.Conv2d if dim==2 else nn.Conv3d
def _pool(dim:int): return nn.MaxPool2d if dim==2 else nn.MaxPool3d

def _interp(x, size): return F.interpolate(x, size=size, mode='bilinear' if x.ndim==4 else 'trilinear', align_corners=False)

class _Block(nn.Module):
    def __init__(self, dim:int, a:int, b:int):
        super().__init__(); C=_conv(dim)
        self.net=nn.Sequential(C(a,b,3,padding=1), nn.GroupNorm(1,b), nn.GELU(), C(b,b,3,padding=1), nn.GroupNorm(1,b), nn.GELU())
    def forward(self,x): return self.net(x)

class _UNet(nn.Module):
    def __init__(self, dim:int, in_channels:int, out_channels:int, hidden_channels:int=64):
        super().__init__(); C=_conv(dim); P=_pool(dim); h=hidden_channels
        self.e1=_Block(dim,in_channels,h); self.e2=_Block(dim,h,2*h); self.pool=P(2)
        self.mid=_Block(dim,2*h,4*h); self.u2=C(4*h,2*h,1); self.d2=_Block(dim,4*h,2*h)
        self.u1=C(2*h,h,1); self.d1=_Block(dim,2*h,h); self.out=C(h,out_channels,1)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        e1=self.e1(x); e2=self.e2(self.pool(e1)); m=self.mid(self.pool(e2))
        u2=_interp(self.u2(m), e2.shape[2:]); d2=self.d2(torch.cat([u2,e2],1))
        u1=_interp(self.u1(d2), e1.shape[2:]); d1=self.d1(torch.cat([u1,e1],1))
        return self.out(d1)

class UNet2D(_UNet):
    def __init__(self, in_channels:int, out_channels:int, hidden_channels:int=64, **kw): super().__init__(2,in_channels,out_channels,hidden_channels)
class UNet3D(_UNet):
    def __init__(self, in_channels:int, out_channels:int, hidden_channels:int=32, **kw): super().__init__(3,in_channels,out_channels,hidden_channels)
