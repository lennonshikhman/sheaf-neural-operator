# Sheaf Neural Operators for Structure-Preserving Magnetohydrodynamics

This repository implements a full experimental suite for **Sheaf Neural Operators for MHD**. The research goal is to compare standard neural PDE surrogate baselines against a **Sheaf Neural Operator** that encodes magnetohydrodynamic structure, with particular emphasis on controlling the magnetic divergence constraint

\[
\nabla \cdot B = 0.
\]

## Why Sheaf Neural Operators for MHD?

MHD fields should not be treated only as arbitrary image channels. Density, velocity, and magnetic variables play different geometric roles and satisfy different compatibility constraints. The **Sheaf Neural Operator** implemented here reflects this by:

- representing fluid variables and magnetic variables as different local fibers;
- coupling those fibers through learned 1x1 restriction maps;
- adding incidence/Hodge/de Rham-inspired finite-difference features such as magnetic divergence;
- using a curl/EMF magnetic update on the 2D Orszag-Tang grid so the learned update is divergence-free under periodic finite differences;
- comparing against UNet and FNO baselines that operate more directly on channel stacks.

The public-facing method name is **Sheaf Neural Operator**. The internal model key is `sheaf_mhd`, implemented by `SheafMHDOperator`.

## Datasets

The one-command experiment expects the following local data layouts. The code does **not** download data.

### The Well `MHD_64`

```text
datasets/wells/
```

Loaded with:

```python
from the_well.data import WellDataset
```

The dataset adapter inspects tensor shapes and converts samples to the internal 3D channel-first convention:

```text
x: [C_in, X, Y, Z]
y: [C_out, X, Y, Z]
```

If magnetic channel indices are provided in the config, divergence diagnostics are computed and the 3D Sheaf Neural Operator can use a vector-potential curl head for divergence-controlled magnetic updates. If channel metadata are unavailable, the configuration intentionally leaves the indices unset rather than guessing physical channels.

### Orszag-Tang FARGO3D processed dataset

```text
datasets/orszag_tang/input_data/
  density/
  vy/
  vz/
  by/
  bz/
```

The expected field order is:

```text
[density, vy, vz, by, bz]
```

The spatial axes are interpreted as axis 0 = `y` and axis 1 = `z`. Samples are returned as:

```text
x: [5 * n_input_frames, H, W]
y: [5, H, W]
```

## One-command full experiment

Run from the repository root:

```bash
python experiments.py
```

No command-line flags are required or used. By default, `experiments.py` sets:

```python
FULL_EXPERIMENT = True
SMOKE_TEST = False
```

The default suite covers both datasets, three models (`unet`, `fno`, `sheaf_mhd`), and ten random seeds (`0` through `9`). Developers can manually set `SMOKE_TEST = True` at the top of `experiments.py` for a 1-epoch local pipeline check with small sample caps, then restore it to `False` before running the full suite.

## Models

### UNet baseline

`UNet2D` and `UNet3D` provide compact convolutional encoder-decoder baselines.

### FNO baseline

`FNO2D` and `FNO3D` implement Fourier Neural Operator layers directly in PyTorch using spectral convolutions, pointwise 1x1 convolutions, and GELU activations. The code does not depend on `neuraloperator`.

### Sheaf Neural Operator

`SheafMHDOperator` separates fluid and magnetic fibers, updates each fiber with local CNN or FNO operators, exchanges information through learned sheaf-style restriction maps, and augments magnetic runs with incidence-inspired divergence information when magnetic channels are known.

For 2D Orszag-Tang, the magnetic head predicts a scalar EMF-like potential `a(y,z)` and applies:

```text
delta_by =  d a / dz
delta_bz = -d a / dy
by_next  = by + dt * delta_by
bz_next  = bz + dt * delta_bz
```

Because the update is a discrete curl, `div(delta_B)` is approximately zero under the same periodic finite-difference operators used for diagnostics.

## Outputs

Each run creates a timestamped directory:

```text
outputs/full_experiment_<timestamp>/
```

Important artifacts include:

```text
resolved_experiment_config.json
experiment_log.txt
raw_metrics.csv
aggregate_metrics.csv
aggregate_metrics.json
experiment_summary.json
paper_tables/
  main_results.tex
  main_results.md
  divergence_results.tex
  rollout_results.tex
  pairwise_comparisons.tex
figures/
  loss_curves/
  prediction_examples/
  divergence_maps/
  rollout_curves/
  aggregate_barplots/
runs/<dataset>/<model>/seed_<seed>/
  config_resolved.json
  train_log.csv
  metrics_valid.json
  metrics_test.json
  rollout_metrics.json
  best_model.pt
  last_model.pt
  figures/
```

If a dataset is missing or a run fails, the failure is recorded in JSON/CSV outputs and in `experiment_summary.json`; failures are not silently skipped.

## Metrics and statistics

The evaluation pipeline reports:

- MSE;
- relative L2;
- per-channel relative L2;
- magnetic divergence L2;
- relative magnetic divergence;
- inference time per batch;
- parameter count;
- rollout relative L2/MSE/divergence where possible.

Aggregation over seeds includes mean, standard deviation, standard error, 95% t-confidence intervals, deterministic bootstrap confidence intervals, median, and interquartile range. Pairwise comparisons evaluate `sheaf_mhd` against `unet` and `fno` with paired tests when matching seeds are available.

## Limitations

- The Well `MHD_64` is a uniform-grid benchmark, so it tests structure preservation and 3D scalability more than arbitrary cell-complex generality.
- Orszag-Tang is 2D and uses a simplified field set.
- This implementation builds the sheaf/cell-complex structure on structured grids. Unstructured-mesh experiments would require explicit mesh cells, faces, edges, and incidence matrices supplied by the dataset.
- The 2D Orszag-Tang magnetic update is the most directly constrained path because the field set contains exactly the two magnetic components needed by the scalar EMF update. The 3D vector-potential curl head is available when reliable magnetic channel indices are supplied for The Well.
