"""Sheaf Neural Operator for structure-preserving MHD."""
from __future__ import annotations
import torch
from torch import nn
from .sheaf_layers import SheafMessageBlock, conv
from src.physics.curl import periodic_curl_scalar_2d, curl_vector_potential_3d
from src.physics.divergence import periodic_divergence_2d

class SheafMHDOperator(nn.Module):
    """Sheaf Neural Operator with fluid/magnetic fibers and curl-constrained 2D magnetic updates.

    Public model name: Sheaf Neural Operator. Internal identifier: sheaf_mhd.
    """
    def __init__(self, dim:int, in_channels:int, out_channels:int, hidden_channels:int=64, num_layers:int=4,
                 backbone_type:str='cnn', modes:int=16, periodic:bool=True, dt:float=1.0, spacing:list[float]|None=None,
                 magnetic_field_indices:list[int]|None=None, fluid_field_indices:list[int]|None=None,
                 constrained_magnetic_update:bool=True, use_restriction_maps:bool=True,
                 use_incidence_features:bool=True):
        super().__init__(); self.dim=dim; self.out_channels=out_channels; self.dt=dt; self.spacing=spacing or [1.0]*dim
        self.magnetic_field_indices=magnetic_field_indices or ([] if dim==3 else [3,4])
        self.fluid_field_indices=fluid_field_indices or [i for i in range(out_channels) if i not in self.magnetic_field_indices]
        self.constrained_magnetic_update=constrained_magnetic_update and dim==2 and len(self.magnetic_field_indices)>=2
        self.use_incidence_features=use_incidence_features; C=conv(dim)
        extra = 1 if (dim==2 and use_incidence_features and len(self.magnetic_field_indices)>=2) else 0
        self.fluid_lift=C(in_channels+extra, hidden_channels, 1); self.mag_lift=C(in_channels+extra, hidden_channels, 1)
        self.blocks=nn.ModuleList([SheafMessageBlock(dim, hidden_channels, use_restriction_maps) for _ in range(num_layers)])
        self.nonmag_head=nn.Sequential(C(hidden_channels, hidden_channels,3,padding=1), nn.GELU(), C(hidden_channels, len(self.fluid_field_indices),1))
        if self.constrained_magnetic_update:
            self.emf_head=nn.Sequential(C(hidden_channels, hidden_channels,3,padding=1), nn.GELU(), C(hidden_channels,1,1))
        else:
            mag_out = len(self.magnetic_field_indices) if self.magnetic_field_indices else out_channels - len(self.fluid_field_indices)
            self.mag_head=nn.Sequential(C(hidden_channels, hidden_channels,3,padding=1), nn.GELU(), C(hidden_channels, mag_out,1))

    def _incidence_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim==2 and self.use_incidence_features and len(self.magnetic_field_indices)>=2:
            by=x[:, self.magnetic_field_indices[0] % x.shape[1]]; bz=x[:, self.magnetic_field_indices[1] % x.shape[1]]
            div=periodic_divergence_2d(by,bz,self.spacing[0],self.spacing[1]).unsqueeze(1)
            return torch.cat([x, div], dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z=self._incidence_features(x)
        fluid=self.fluid_lift(z); magnetic=self.mag_lift(z)
        for block in self.blocks: fluid, magnetic = block(fluid, magnetic)
        out=torch.zeros(x.shape[0], self.out_channels, *x.shape[2:], device=x.device, dtype=x.dtype)
        fluid_delta=self.nonmag_head(fluid)
        for j, idx in enumerate(self.fluid_field_indices[:fluid_delta.shape[1]]):
            base=x[:, idx] if idx < x.shape[1] else 0.0; out[:, idx]=base + fluid_delta[:, j]
        if self.constrained_magnetic_update:
            a=self.emf_head(magnetic)[:,0]
            dby, dbz=periodic_curl_scalar_2d(a, self.spacing[0], self.spacing[1])
            by_idx,bz_idx=self.magnetic_field_indices[:2]
            out[:, by_idx]=x[:, by_idx] + self.dt*dby
            out[:, bz_idx]=x[:, bz_idx] + self.dt*dbz
        else:
            md=self.mag_head(magnetic)
            for j, idx in enumerate(self.magnetic_field_indices[:md.shape[1]]):
                base=x[:, idx] if idx < x.shape[1] else 0.0; out[:, idx]=base + md[:, j]
        return out
