from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from src.complexes import CochainBatch, get_cached_cubical_complex
from src.complexes.incidence import sparse_mm
from src.sheaves import CellularSheafMessageBlock


class CellularMHDSheafNeuralOperator(nn.Module):
    """Cellular/cochain sheaf neural operator for MHD on oriented cell complexes.

    Public experiment names ``sheaf_mhd`` and ``cellular_mhd_sno`` instantiate
    this backend.  Magnetic flux is represented as a face 2-cochain and updated
    exactly by ``B_next = B - dt d1 E``; fluid/conserved state is represented as
    a volume 3-cochain and updated by ``U_next = U - dt d2 F + dt S``.
    """

    backend_name = "cellular_mhd_sno"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 4,
        dim: int = 3,
        modes: int | None = None,
        periodic: bool = False,
        dt: float = 1.0,
        spacing: list[float] | tuple[float, float, float] | None = None,
        magnetic_field_indices: list[int] | None = None,
        fluid_field_indices: list[int] | None = None,
        grid_shape: tuple[int, int, int] | None = None,
        max_internal_cells: int = 32768,
        use_sheaf_laplacian: bool = False,
        use_geometry_conditioned_restrictions: bool = False,
        use_geometric_hodge: bool = True,
        **_: Any,
    ):
        super().__init__()
        if dim != 3:
            raise ValueError(f"CellularMHDSheafNeuralOperator requires dim=3, got {dim}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.periodic = bool(periodic)
        self.dt = float(dt)
        self.spacing = tuple(float(s) for s in (spacing or (1.0, 1.0, 1.0)))
        self.magnetic_field_indices = magnetic_field_indices or []
        self.fluid_field_indices = fluid_field_indices or [i for i in range(out_channels) if i not in self.magnetic_field_indices]
        self.grid_shape = tuple(grid_shape) if grid_shape is not None else None
        self.max_internal_cells = int(max_internal_cells)
        self.use_sheaf_laplacian = bool(use_sheaf_laplacian)
        self.use_geometry_conditioned_restrictions = bool(use_geometry_conditioned_restrictions)
        self.use_geometric_hodge = bool(use_geometric_hodge)
        self._complex_cache: dict[tuple[Any, ...], Any] = {}
        self._last_complex_summary: dict[str, Any] = {}

        self.vertex_lift = nn.Linear(1, hidden_channels)
        self.edge_lift = nn.Linear(1, hidden_channels)
        self.face_lift = nn.Linear(1, hidden_channels)
        self.cell_lift = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList([
            CellularSheafMessageBlock(
                hidden_channels,
                use_sheaf_laplacian=self.use_sheaf_laplacian,
                use_geometry_conditioned_restrictions=self.use_geometry_conditioned_restrictions,
            )
            for _ in range(num_layers)
        ])
        self.edge_emf_head = nn.Linear(hidden_channels, 1)
        self.face_flux_head = nn.Linear(hidden_channels, len(self.fluid_field_indices))
        self.cell_source_head = nn.Linear(hidden_channels, len(self.fluid_field_indices))

    def _internal_shape(self, shape: tuple[int, int, int]) -> tuple[int, int, int]:
        n = shape[0] * shape[1] * shape[2]
        if n <= self.max_internal_cells:
            return shape
        scale = (self.max_internal_cells / float(n)) ** (1.0 / 3.0)
        return tuple(max(4, int(round(s * scale))) for s in shape)

    def _complex(self, shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype):
        key = (shape, self.spacing, self.periodic, str(device), str(dtype))
        if key not in self._complex_cache:
            cx = get_cached_cubical_complex(*shape, spacing=self.spacing, periodic=self.periodic, device=device, dtype=dtype)
            self._complex_cache[key] = cx
            self._last_complex_summary = self._make_summary(cx, shape)
        return self._complex_cache[key]

    def _make_summary(self, cx, shape) -> dict[str, Any]:
        summary = cx.incidence_summary()
        summary.update({
            "model_backend": self.backend_name,
            "magnetic_placement": "C^2 faces",
            "emf_placement": "C^1 edges",
            "fluid_placement": "C^3 cells",
            "exact_d2d1_check_max_error": cx.max_d_next_d_error(1),
            "use_sheaf_laplacian": self.use_sheaf_laplacian,
            "use_geometry_conditioned_restrictions": self.use_geometry_conditioned_restrictions,
            "use_geometric_hodge": self.use_geometric_hodge,
            "internal_grid_shape": list(shape),
        })
        return summary

    def complex_summary(self) -> dict[str, Any]:
        return dict(self._last_complex_summary)

    def _maybe_downsample(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        original = tuple(int(s) for s in x.shape[2:])
        internal = self._internal_shape(original if self.grid_shape is None else self.grid_shape)
        if internal != original:
            x = F.interpolate(x.float(), size=internal, mode="trilinear", align_corners=False).to(dtype=x.dtype)
        return x, original

    def _grid_to_cell_features(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], -1, x.shape[1])

    def _cell_features_to_grid(self, u: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
        return u.reshape(u.shape[0], *shape, u.shape[2]).permute(0, 4, 1, 2, 3).contiguous()

    def _magnetic_faces_from_grid(self, x: torch.Tensor, cx) -> torch.Tensor:
        bsz, _, nx, ny, nz = x.shape
        counts = cx.metadata["face_counts"]
        bx = x[:, self.magnetic_field_indices[0] % x.shape[1]] if len(self.magnetic_field_indices) > 0 else torch.zeros(bsz, nx, ny, nz, device=x.device, dtype=x.dtype)
        by = x[:, self.magnetic_field_indices[1] % x.shape[1]] if len(self.magnetic_field_indices) > 1 else torch.zeros_like(bx)
        bz = x[:, self.magnetic_field_indices[2] % x.shape[1]] if len(self.magnetic_field_indices) > 2 else torch.zeros_like(bx)
        if self.periodic:
            parts = [bx.reshape(bsz, -1), by.reshape(bsz, -1), bz.reshape(bsz, -1)]
        else:
            fx = torch.empty(bsz, nx + 1, ny, nz, device=x.device, dtype=x.dtype)
            fy = torch.empty(bsz, nx, ny + 1, nz, device=x.device, dtype=x.dtype)
            fz = torch.empty(bsz, nx, ny, nz + 1, device=x.device, dtype=x.dtype)
            fx[:, 1:nx] = 0.5 * (bx[:, 1:] + bx[:, :-1]); fx[:, 0] = bx[:, 0]; fx[:, nx] = bx[:, -1]
            fy[:, :, 1:ny] = 0.5 * (by[:, :, 1:] + by[:, :, :-1]); fy[:, :, 0] = by[:, :, 0]; fy[:, :, ny] = by[:, :, -1]
            fz[:, :, :, 1:nz] = 0.5 * (bz[:, :, :, 1:] + bz[:, :, :, :-1]); fz[:, :, :, 0] = bz[:, :, :, 0]; fz[:, :, :, nz] = bz[:, :, :, -1]
            parts = [fx.reshape(bsz, -1), fy.reshape(bsz, -1), fz.reshape(bsz, -1)]
        out = torch.cat(parts, dim=1).unsqueeze(-1)
        if out.shape[1] != sum(counts):
            raise RuntimeError("face projection produced inconsistent face count")
        return out

    def _magnetic_grid_from_faces(self, faces: torch.Tensor, shape: tuple[int, int, int], cx) -> torch.Tensor:
        bsz = faces.shape[0]
        nx, ny, nz = shape
        nfx, nfy, nfz = cx.metadata["face_counts"]
        fx, fy, fz = faces[:, :nfx, 0], faces[:, nfx:nfx+nfy, 0], faces[:, nfx+nfy:nfx+nfy+nfz, 0]
        if self.periodic:
            bx = fx.reshape(bsz, nx, ny, nz); by = fy.reshape(bsz, nx, ny, nz); bz = fz.reshape(bsz, nx, ny, nz)
        else:
            fx = fx.reshape(bsz, nx + 1, ny, nz); fy = fy.reshape(bsz, nx, ny + 1, nz); fz = fz.reshape(bsz, nx, ny, nz + 1)
            bx = 0.5 * (fx[:, 1:] + fx[:, :-1])
            by = 0.5 * (fy[:, :, 1:] + fy[:, :, :-1])
            bz = 0.5 * (fz[:, :, :, 1:] + fz[:, :, :, :-1])
        return torch.stack([bx, by, bz], dim=1)

    def grid_to_cochains(self, x: torch.Tensor, cx) -> CochainBatch:
        bsz = x.shape[0]
        cells = self._grid_to_cell_features(x)
        faces = self._magnetic_faces_from_grid(x, cx)
        edges = torch.zeros(bsz, cx.num_cells(1), 1, device=x.device, dtype=x.dtype)
        verts = torch.zeros(bsz, cx.num_cells(0), 1, device=x.device, dtype=x.dtype)
        return CochainBatch({0: verts, 1: edges, 2: faces, 3: cells})

    def cochains_to_grid(self, cell_state: torch.Tensor, face_b: torch.Tensor, shape: tuple[int, int, int], cx, base: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(base.shape[0], self.out_channels, *shape, device=base.device, dtype=base.dtype)
        cell_grid = self._cell_features_to_grid(cell_state, shape)
        for idx in range(min(self.out_channels, cell_grid.shape[1])):
            out[:, idx] = cell_grid[:, idx]
        if self.magnetic_field_indices:
            bgrid = self._magnetic_grid_from_faces(face_b, shape, cx)
            for j, idx in enumerate(self.magnetic_field_indices[:3]):
                if idx < self.out_channels:
                    out[:, idx] = bgrid[:, j]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected channel-first [B,C,X,Y,Z] tensor, got {tuple(x.shape)}")
        x_internal, original_shape = self._maybe_downsample(x)
        internal_shape = tuple(int(s) for s in x_internal.shape[2:])
        cx = self._complex(internal_shape, x.device, x.dtype)
        cochains = self.grid_to_cochains(x_internal, cx)
        h = {
            0: self.vertex_lift(cochains[0]),
            1: self.edge_lift(cochains[1]),
            2: self.face_lift(cochains[2]),
            3: self.cell_lift(cochains[3]),
        }
        for block in self.blocks:
            h = block(h, cx)

        e_theta = self.edge_emf_head(h[1])
        b_next = cochains[2] - self.dt * sparse_mm(cx.d(1), e_theta)

        fluid_flux = self.face_flux_head(h[2])
        source = self.cell_source_head(h[3])
        div_flux = sparse_mm(cx.d(2), fluid_flux)
        cell_state = torch.zeros(x_internal.shape[0], cx.num_cells(3), self.out_channels, device=x.device, dtype=x.dtype)
        copy_ch = min(self.out_channels, x_internal.shape[1])
        cell_state[:, :, :copy_ch] = cochains[3][:, :, :copy_ch]
        fluid_next = cell_state[:, :, self.fluid_field_indices] - self.dt * div_flux + self.dt * source if self.fluid_field_indices else torch.empty(0, device=x.device)
        for j, idx in enumerate(self.fluid_field_indices):
            if idx < self.out_channels:
                cell_state[:, :, idx] = fluid_next[:, :, j]

        out_internal = self.cochains_to_grid(cell_state, b_next, internal_shape, cx, x_internal)
        if internal_shape != original_shape:
            out_internal = F.interpolate(out_internal.float(), size=original_shape, mode="trilinear", align_corners=False).to(dtype=x.dtype)
        return out_internal

    @torch.no_grad()
    def verify_magnetic_preservation(self, x: torch.Tensor) -> float:
        x_internal, _ = self._maybe_downsample(x)
        shape = tuple(int(s) for s in x_internal.shape[2:])
        cx = self._complex(shape, x.device, x.dtype)
        b = self._magnetic_faces_from_grid(x_internal, cx)
        e = torch.randn(x.shape[0], cx.num_cells(1), 1, device=x.device, dtype=x.dtype)
        b_next = b - self.dt * sparse_mm(cx.d(1), e)
        return float((sparse_mm(cx.d(2), b_next) - sparse_mm(cx.d(2), b)).abs().max().detach().cpu())
