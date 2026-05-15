import torch

from src.models import CellularMHDSheafNeuralOperator, FNO3D, UNet3D


def test_grid_cochain_projection_shapes_and_forward():
    model = CellularMHDSheafNeuralOperator(
        in_channels=7,
        out_channels=7,
        hidden_channels=4,
        num_layers=1,
        magnetic_field_indices=[1, 2, 3],
        periodic=False,
        max_internal_cells=10_000,
    )
    x = torch.randn(2, 7, 4, 4, 4)
    y = model(x)
    assert y.shape == (2, 7, 4, 4, 4)
    assert model.complex_summary()["exact_d2d1_check_max_error"] == 0.0
    assert model.verify_magnetic_preservation(x) < 1e-5


def test_unet3d_and_fno3d_still_run():
    x = torch.randn(1, 7, 8, 8, 8)
    assert UNet3D(7, 7, hidden_channels=4)(x).shape == (1, 7, 8, 8, 8)
    assert FNO3D(7, 7, hidden_channels=4, num_layers=1, modes=2)(x).shape == (1, 7, 8, 8, 8)


def test_rollout_autoregressive_input_clones_prediction_storage():
    from src.training.rollout import _autoregressive_input

    cur = torch.zeros(1, 3, 2, 2, 2)
    pred = torch.ones_like(cur)
    nxt = _autoregressive_input(cur, pred)
    assert torch.equal(nxt, pred)
    assert nxt.data_ptr() != pred.data_ptr()

    short_pred = torch.ones(1, 2, 2, 2, 2)
    mixed = _autoregressive_input(cur, short_pred)
    assert mixed.shape == cur.shape
    assert mixed.data_ptr() != short_pred.data_ptr()
