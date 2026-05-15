"""Compatibility module: sheaf_mhd now uses the cellular MHD SNO backend."""
from __future__ import annotations

from .cellular_mhd_sno import CellularMHDSheafNeuralOperator


class SheafMHDOperator(CellularMHDSheafNeuralOperator):
    """Backward-compatible class name for the cellular/cochain MHD SNO."""

    pass
