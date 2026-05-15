"""Sheaf Neural Operator for 3D MHD surrogate modeling."""
from __future__ import annotations

import torch
from torch import nn

from .sheaf_layers import SheafMessageBlock, conv
from src.physics.curl import curl_vector_potential_3d
from src.physics.divergence import periodic_divergence_3d


class SheafMHDOperator(nn.Module):
    """Sheaf Neural Operator with fluid/magnetic fibers for 3D MHD grids.

    Public model name: Sheaf Neural Operator. Internal identifier: ``sheaf_mhd``.
    Magnetic channels can be updated either through a vector-potential residual
    ``curl(A)`` or through a direct residual head augmented with divergence features.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 4,
        backbone_type: str = "cnn",
        modes: int = 8,
        periodic: bool = True,
        dt: float = 1.0,
        spacing: list[float] | None = None,
        magnetic_field_indices: list[int] | None = None,
        fluid_field_indices: list[int] | None = None,
        constrained_magnetic_update: str | bool = "direct_with_divergence_features",
        use_restriction_maps: bool = True,
        use_incidence_features: bool = True,
    ):
        super().__init__()
        if dim != 3:
            raise ValueError(f"SheafMHDOperator is configured for 3D MHD grids; got dim={dim}.")
        self.dim = dim
        self.out_channels = out_channels
        self.dt = dt
        self.periodic = periodic
        self.spacing = spacing or [1.0, 1.0, 1.0]
        self.magnetic_field_indices = magnetic_field_indices or []
        self.fluid_field_indices = fluid_field_indices or [i for i in range(out_channels) if i not in self.magnetic_field_indices]
        self.use_incidence_features = use_incidence_features
        if constrained_magnetic_update is True:
            constrained_magnetic_update = "vector_potential"
        if constrained_magnetic_update is False:
            constrained_magnetic_update = "direct_with_divergence_features"
        if constrained_magnetic_update not in {"vector_potential", "direct_with_divergence_features"}:
            raise ValueError("constrained_magnetic_update must be 'vector_potential' or 'direct_with_divergence_features'.")
        self.constrained_magnetic_update = constrained_magnetic_update
        self.use_vector_potential = constrained_magnetic_update == "vector_potential" and len(self.magnetic_field_indices) >= 3

        C = conv(dim)
        incidence_channels = 1 if use_incidence_features and len(self.magnetic_field_indices) >= 3 else 0
        self.fluid_lift = C(in_channels + incidence_channels, hidden_channels, 1)
        self.mag_lift = C(in_channels + incidence_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [
                SheafMessageBlock(
                    dim,
                    hidden_channels,
                    use_restriction_maps=use_restriction_maps,
                    backbone_type=backbone_type,
                    modes=modes,
                )
                for _ in range(num_layers)
            ]
        )
        self.nonmag_head = nn.Sequential(C(hidden_channels, hidden_channels, 3, padding=1), nn.GELU(), C(hidden_channels, len(self.fluid_field_indices), 1))
        if self.use_vector_potential:
            self.vector_potential_head = nn.Sequential(C(hidden_channels, hidden_channels, 3, padding=1), nn.GELU(), C(hidden_channels, 3, 1))
        else:
            self.mag_head = nn.Sequential(C(hidden_channels, hidden_channels, 3, padding=1), nn.GELU(), C(hidden_channels, len(self.magnetic_field_indices), 1))

    def _incidence_features(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_incidence_features or len(self.magnetic_field_indices) < 3:
            return x
        bx, by, bz = [x[:, idx % x.shape[1]] for idx in self.magnetic_field_indices[:3]]
        div = periodic_divergence_3d(bx, by, bz, self.spacing[0], self.spacing[1], self.spacing[2]).unsqueeze(1)
        return torch.cat([x, div], dim=1)

    def _base_channel(self, x: torch.Tensor, idx: int) -> torch.Tensor | float:
        return x[:, idx] if idx < x.shape[1] else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._incidence_features(x)
        fluid = self.fluid_lift(z)
        magnetic = self.mag_lift(z)
        for block in self.blocks:
            fluid, magnetic = block(fluid, magnetic)

        out = torch.zeros(x.shape[0], self.out_channels, *x.shape[2:], device=x.device, dtype=x.dtype)
        fluid_delta = self.nonmag_head(fluid)
        for j, idx in enumerate(self.fluid_field_indices[: fluid_delta.shape[1]]):
            out[:, idx] = self._base_channel(x, idx) + fluid_delta[:, j]

        if self.use_vector_potential:
            potential = self.vector_potential_head(magnetic)
            dbx, dby, dbz = curl_vector_potential_3d(
                potential[:, 0], potential[:, 1], potential[:, 2], self.spacing[0], self.spacing[1], self.spacing[2]
            )
            for idx, delta in zip(self.magnetic_field_indices[:3], (dbx, dby, dbz), strict=False):
                out[:, idx] = self._base_channel(x, idx) + self.dt * delta
        elif self.magnetic_field_indices:
            md = self.mag_head(magnetic)
            for j, idx in enumerate(self.magnetic_field_indices[: md.shape[1]]):
                out[:, idx] = self._base_channel(x, idx) + md[:, j]
        return out
