import torch

from src.complexes import cubical_complex_3d
from src.complexes.incidence import sparse_mm


def test_d2_d1_zero_numerically_and_magnetic_update_preserves_divergence():
    cx = cubical_complex_3d(4, 4, 4, periodic=False)
    dense = torch.sparse.mm(cx.d(2), cx.d(1).to_dense())
    assert torch.allclose(dense, torch.zeros_like(dense))
    b = torch.randn(2, cx.num_cells(2), 1)
    e = torch.randn(2, cx.num_cells(1), 1)
    b_next = b - 0.25 * sparse_mm(cx.d(1), e)
    before = sparse_mm(cx.d(2), b)
    after = sparse_mm(cx.d(2), b_next)
    assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)
