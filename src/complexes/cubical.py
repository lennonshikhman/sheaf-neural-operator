from __future__ import annotations

from functools import lru_cache

import torch

from .cell_complex import CellComplex
from .hodge import diagonal_hodge_weights
from .incidence import coalesce_sparse


def _append(rows: list[int], cols: list[int], vals: list[float], r: int, c: int, v: float) -> None:
    rows.append(r); cols.append(c); vals.append(v)


def cubical_complex_3d(nx: int, ny: int, nz: int, spacing: tuple[float, float, float] | list[float] = (1.0, 1.0, 1.0), periodic: bool = False, *, dtype: torch.dtype = torch.float32) -> CellComplex:
    """Construct an oriented cubical complex for a 3-D Cartesian grid.

    Cochain ordering is grouped by orientation: x/y/z edges, then x/y/z normal
    faces, then volume cells.  Non-periodic complexes include boundary faces and
    edges. Periodic complexes identify opposite sides and use modulo indexing.
    """
    nx, ny, nz = int(nx), int(ny), int(nz)
    if min(nx, ny, nz) <= 0:
        raise ValueError(f"Invalid cubical shape {(nx, ny, nz)}")
    spacing = tuple(float(s) for s in spacing)

    if periodic:
        n_v = nx * ny * nz
        n_ex = nx * ny * nz; n_ey = nx * ny * nz; n_ez = nx * ny * nz
        n_fx = nx * ny * nz; n_fy = nx * ny * nz; n_fz = nx * ny * nz
        def v(i,j,k): return ((i % nx) * ny + (j % ny)) * nz + (k % nz)
        def ex(i,j,k): return ((i % nx) * ny + (j % ny)) * nz + (k % nz)
        def ey(i,j,k): return n_ex + ((i % nx) * ny + (j % ny)) * nz + (k % nz)
        def ez(i,j,k): return n_ex + n_ey + ((i % nx) * ny + (j % ny)) * nz + (k % nz)
        def fx(i,j,k): return ((i % nx) * ny + (j % ny)) * nz + (k % nz)
        def fy(i,j,k): return n_fx + ((i % nx) * ny + (j % ny)) * nz + (k % nz)
        def fz(i,j,k): return n_fx + n_fy + ((i % nx) * ny + (j % ny)) * nz + (k % nz)
    else:
        n_v = (nx + 1) * (ny + 1) * (nz + 1)
        n_ex = nx * (ny + 1) * (nz + 1); n_ey = (nx + 1) * ny * (nz + 1); n_ez = (nx + 1) * (ny + 1) * nz
        n_fx = (nx + 1) * ny * nz; n_fy = nx * (ny + 1) * nz; n_fz = nx * ny * (nz + 1)
        def v(i,j,k): return (i * (ny + 1) + j) * (nz + 1) + k
        def ex(i,j,k): return (i * (ny + 1) + j) * (nz + 1) + k
        def ey(i,j,k): return n_ex + (i * ny + j) * (nz + 1) + k
        def ez(i,j,k): return n_ex + n_ey + (i * (ny + 1) + j) * nz + k
        def fx(i,j,k): return (i * ny + j) * nz + k
        def fy(i,j,k): return n_fx + (i * (ny + 1) + j) * nz + k
        def fz(i,j,k): return n_fx + n_fy + (i * ny + j) * (nz + 1) + k

    n_edges = n_ex + n_ey + n_ez
    n_faces = n_fx + n_fy + n_fz
    n_cells = nx * ny * nz
    cell = lambda i, j, k: (i * ny + j) * nz + k

    # ∂1: edges -> vertices.
    rows: list[int] = []; cols: list[int] = []; vals: list[float] = []
    for i in range(nx):
        for j in range(ny if periodic else ny + 1):
            for k in range(nz if periodic else nz + 1):
                c = ex(i,j,k); _append(rows, cols, vals, v(i+1,j,k), c, +1.0); _append(rows, cols, vals, v(i,j,k), c, -1.0)
    for i in range(nx if periodic else nx + 1):
        for j in range(ny):
            for k in range(nz if periodic else nz + 1):
                c = ey(i,j,k); _append(rows, cols, vals, v(i,j+1,k), c, +1.0); _append(rows, cols, vals, v(i,j,k), c, -1.0)
    for i in range(nx if periodic else nx + 1):
        for j in range(ny if periodic else ny + 1):
            for k in range(nz):
                c = ez(i,j,k); _append(rows, cols, vals, v(i,j,k+1), c, +1.0); _append(rows, cols, vals, v(i,j,k), c, -1.0)
    partial1 = coalesce_sparse(torch.tensor([rows, cols]), torch.tensor(vals), (n_v, n_edges), dtype=dtype)

    # ∂2: faces -> edges from exterior-derivative sign convention.
    rows = []; cols = []; vals = []
    # x-normal yz faces: dE = Ez(y+1)-Ez(y)-Ey(z+1)+Ey(z)
    for i in range(nx if periodic else nx + 1):
        for j in range(ny):
            for k in range(nz):
                c = fx(i,j,k)
                for e, s in [(ez(i,j+1,k), +1), (ez(i,j,k), -1), (ey(i,j,k+1), -1), (ey(i,j,k), +1)]:
                    _append(rows, cols, vals, e, c, float(s))
    # y-normal zx faces: dE = Ex(z+1)-Ex(z)-Ez(x+1)+Ez(x)
    for i in range(nx):
        for j in range(ny if periodic else ny + 1):
            for k in range(nz):
                c = fy(i,j,k)
                for e, s in [(ex(i,j,k+1), +1), (ex(i,j,k), -1), (ez(i+1,j,k), -1), (ez(i,j,k), +1)]:
                    _append(rows, cols, vals, e, c, float(s))
    # z-normal xy faces: dE = Ey(x+1)-Ey(x)-Ex(y+1)+Ex(y)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz if periodic else nz + 1):
                c = fz(i,j,k)
                for e, s in [(ey(i+1,j,k), +1), (ey(i,j,k), -1), (ex(i,j+1,k), -1), (ex(i,j,k), +1)]:
                    _append(rows, cols, vals, e, c, float(s))
    partial2 = coalesce_sparse(torch.tensor([rows, cols]), torch.tensor(vals), (n_edges, n_faces), dtype=dtype)

    # ∂3: cells -> faces; d2 is finite-volume divergence.
    rows = []; cols = []; vals = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = cell(i,j,k)
                for fidx, s in [(fx(i+1,j,k), +1), (fx(i,j,k), -1), (fy(i,j+1,k), +1), (fy(i,j,k), -1), (fz(i,j,k+1), +1), (fz(i,j,k), -1)]:
                    _append(rows, cols, vals, fidx, c, float(s))
    partial3 = coalesce_sparse(torch.tensor([rows, cols]), torch.tensor(vals), (n_faces, n_cells), dtype=dtype)

    n_by_dim = {0: n_v, 1: n_edges, 2: n_faces, 3: n_cells}
    hodge = diagonal_hodge_weights(n_by_dim, spacing)
    dx, dy, dz = spacing
    # More specific primal measures by orientation for edges/faces.
    hodge[1] = torch.cat([torch.full((n_ex,), dx), torch.full((n_ey,), dy), torch.full((n_ez,), dz)]).to(dtype)
    hodge[2] = torch.cat([torch.full((n_fx,), dy * dz), torch.full((n_fy,), dz * dx), torch.full((n_fz,), dx * dy)]).to(dtype)
    hodge[3] = hodge[3].to(dtype); hodge[0] = hodge[0].to(dtype)
    metadata = {
        "complex_type": "cubical", "grid_shape": [nx, ny, nz], "spacing": list(spacing), "periodic": periodic,
        "edge_counts": [n_ex, n_ey, n_ez], "face_counts": [n_fx, n_fy, n_fz],
    }
    return CellComplex({0: n_v, 1: n_edges, 2: n_faces, 3: n_cells}, {1: partial1, 2: partial2, 3: partial3}, {}, hodge, {}, metadata)


@lru_cache(maxsize=32)
def _cached_cpu(nx: int, ny: int, nz: int, spacing: tuple[float, float, float], periodic: bool) -> CellComplex:
    return cubical_complex_3d(nx, ny, nz, spacing, periodic)


def get_cached_cubical_complex(nx: int, ny: int, nz: int, spacing=(1.0, 1.0, 1.0), periodic: bool = False, *, device=None, dtype=torch.float32) -> CellComplex:
    spacing_t = tuple(float(s) for s in spacing)
    cx = _cached_cpu(int(nx), int(ny), int(nz), spacing_t, bool(periodic))
    if device is None or str(device) == "cpu":
        return cx.to("cpu", dtype)
    return cx.to(device, dtype)
