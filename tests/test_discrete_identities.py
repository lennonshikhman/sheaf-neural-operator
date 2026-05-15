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


def test_sparse_identity_check_and_sparse_mm_tolerate_bfloat16():
    cx = cubical_complex_3d(4, 4, 4, periodic=False).to("cpu", torch.bfloat16)
    assert cx.max_d_next_d_error(1) == 0.0
    e = torch.randn(1, cx.num_cells(1), 1, dtype=torch.bfloat16)
    curl = sparse_mm(cx.d(1), e)
    assert curl.dtype == torch.bfloat16
    assert curl.shape == (1, cx.num_cells(2), 1)


def test_sparse_mm_disables_autocast_for_bfloat16():
    cx = cubical_complex_3d(4, 4, 4, periodic=False)
    e = torch.randn(1, cx.num_cells(1), 1)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        curl = sparse_mm(cx.d(1), e)
    assert curl.dtype == e.dtype
    assert curl.shape == (1, cx.num_cells(2), 1)


def test_sparse_mm_disables_cuda_autocast_for_bfloat16_if_available():
    if not torch.cuda.is_available():
        return
    cx = cubical_complex_3d(4, 4, 4, periodic=False).to("cuda", torch.float32)
    e = torch.randn(1, cx.num_cells(1), 1, device="cuda", dtype=torch.bfloat16)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        curl = sparse_mm(cx.d(1), e)
    assert curl.dtype == torch.bfloat16
    assert curl.shape == (1, cx.num_cells(2), 1)
