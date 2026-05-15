import torch

from src.complexes import cubical_complex_3d


def test_cubical_complex_counts_and_d_squared_zero():
    cx = cubical_complex_3d(4, 4, 4, periodic=False)
    assert cx.num_cells(3) == 64
    assert cx.d(1).shape == (cx.num_cells(2), cx.num_cells(1))
    assert cx.d(2).shape == (cx.num_cells(3), cx.num_cells(2))
    assert cx.max_d_next_d_error(0) == 0.0
    assert cx.max_d_next_d_error(1) == 0.0


def test_periodic_cubical_complex_d_squared_zero():
    cx = cubical_complex_3d(4, 4, 4, periodic=True)
    assert cx.max_d_next_d_error(0) == 0.0
    assert cx.max_d_next_d_error(1) == 0.0
