from __future__ import annotations

import torch


def coalesce_sparse(indices: torch.Tensor, values: torch.Tensor, shape: tuple[int, int], *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Create a coalesced COO sparse tensor with a pure PyTorch fallback path."""
    if indices.numel() == 0:
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=dtype, device=device)
    else:
        indices = indices.to(device=device, dtype=torch.long)
        values = values.to(device=device, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, shape, device=device, dtype=dtype).coalesce()


def sparse_mm(matrix: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Apply sparse matrix [M,N] to batched cochains [B,N,C] -> [B,M,C]."""
    if features.ndim != 3:
        raise ValueError(f"Expected [B,N,C] cochain features, got {tuple(features.shape)}")
    bsz, n_cells, n_chan = features.shape
    if matrix.shape[1] != n_cells:
        raise ValueError(f"Sparse matrix width {matrix.shape[1]} does not match feature cells {n_cells}")
    mat = matrix.to(device=features.device, dtype=features.dtype)
    flat = features.permute(1, 0, 2).reshape(n_cells, bsz * n_chan)
    out = torch.sparse.mm(mat, flat)
    return out.reshape(matrix.shape[0], bsz, n_chan).permute(1, 0, 2).contiguous()


def sparse_transpose_mm(matrix: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Apply sparse matrix transpose to batched cochains."""
    return sparse_mm(matrix.transpose(0, 1).coalesce(), features)
