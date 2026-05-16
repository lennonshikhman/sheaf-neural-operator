# Cellular Sheaf Neural Operators for Structure-Preserving Magnetohydrodynamic Surrogate Modeling

This repository implements a one-command experimental suite for comparing standard neural PDE surrogate models against a **Cellular Sheaf Neural Operator** for magnetohydrodynamic (MHD) surrogate modeling.

The central scientific claim is not that the current datasets are already arbitrary unstructured cell complexes. Instead, the claim is that MHD fields have different physical roles, constraints, and compatibility relations. The Cellular Sheaf Neural Operator represents these coupled variables through local fibers, learned restriction maps, and incidence/Hodge-inspired communication features rather than treating every field as an interchangeable image channel.

## Why Cellular Sheaf Neural Operators for MHD?

MHD couples density, pressure, velocity, magnetic fields, current density, and boundary/interface effects. A Cellular Sheaf Neural Operator is appropriate because it can:

- maintain separate local fibers for fluid and magnetic variables;
- learn restriction/coupling maps between those fibers;
- use incidence-inspired features such as magnetic divergence;
- use vector-potential magnetic residual updates when periodic 3D magnetic channels are known;
- provide ablations with or without restriction maps and incidence features;
- support both homogeneous turbulence-style benchmarks and bounded/coupled MHD systems.

The paper-facing model name is **Cellular Sheaf Neural Operator**. The internal model key is `sheaf_mhd`, implemented by `SheafMHDOperator`.

## Datasets

The code does **not** download data and does not require internet access. Place datasets under the following local paths.

### The Well `MHD_64`

```text
datasets/wells/datasets/MHD_64/
```

The loader also accepts the older parent root `datasets/wells/` and resolves the nested `datasets/` directory automatically. It reads the local HDF5 schema directly, using `t0_fields/density`, `t1_fields/magnetic_field`, and `t1_fields/velocity` even though the scalar and vector fields live under different groups. Each one-frame state is ordered as seven channels:

```text
density + magnetic_field[3] + velocity[3]
x: [7*n_input_frames, X, Y, Z]
y: [7*n_output_frames, X, Y, Z]
```

The Well `MHD_64` is a homogeneous/periodic 3D MHD benchmark used for prediction accuracy, rollout stability, spectra/correlation diagnostics, divergence diagnostics, and scalability.

### SWIGS Gorgon MHD

```text
datasets/swigs_gorgon/
```

The SWIGS/Gorgon loader recursively discovers `.h5`, `.hdf5`, and `.hdf` files under this root. The first supported schema intentionally ignores ionosphere `IS` files, indexes only magnetosphere `MS` files by shock directory and timestamp, requires `P` plus `Bvec_c` at both `t` and `t+dt`, and downsamples the large `480x320x320` arrays by default before training. It caches an index at:

```text
datasets/swigs_gorgon/.swigs_index.json
```

SWIGS is a bounded/coupled magnetosphere-ionosphere MHD benchmark. It is the more natural dataset for the sheaf/restriction-map story because different variables and regions can be coupled through boundary/interface-type relations.

### Optional ConStellaration equilibrium subset

```text
datasets/constellaration_subset/
  boundaries_and_metrics.jsonl
  vmecpp_wout_finite_beta_3pct.jsonl
```

This optional track is a supervised fusion-equilibrium regression problem, not a time-evolution rollout benchmark. The loader joins JSONL rows by configuration identifiers when possible, parses JSON-valued string columns such as `boundary.json`, `metrics.json`, and WOut `json`, flattens numeric leaves into input/output vectors, standardizes features, and evaluates equilibrium surrogate models separately from time-dependent MHD rollouts.

If the folder is missing, only the ConStellaration track is skipped with a logged warning.

## One-command full experiment

Run from the repository root:

```bash
python experiments.py
```

No command-line flags are required or used. The top of `experiments.py` exposes simple manual constants:

```python
SMOKE_TEST = False
RUN_THE_WELL = True
RUN_SWIGS = True
RUN_CONSTELLARATION = True
```

When `SMOKE_TEST = True`, the script uses one seed, one epoch, and small sample caps for quick local checks. The committed default is `SMOKE_TEST = False`.

## Default suite

Time-dependent datasets:

- `wells_mhd64`
- `swigs_gorgon`

Time-dependent models:

- `unet3d`
- `fno3d`
- `sheaf_mhd` / **Cellular Sheaf Neural Operator**

Optional equilibrium dataset:

- `constellaration_equilibrium`

Equilibrium models:

- `mlp`
- `sheaf_equilibrium`

Default seeds are `0` through `9`.

## Outputs

Each run creates:

```text
outputs/full_experiment_<timestamp>/
  resolved_experiment_config.json
  experiment_log.txt
  dataset_inspection.json
  raw_metrics.csv
  aggregate_metrics.csv
  aggregate_metrics.json
  experiment_summary.json
  paper_tables/
    main_results.tex
    main_results.md
    divergence_results.tex
    rollout_results.tex
    swigs_results.tex
    constellaration_results.tex
    pairwise_comparisons.tex
  figures/
    loss_curves/
    prediction_examples/
    divergence_maps/
    spectra/
    rollout_curves/
    aggregate_barplots/
  runs/
    wells_mhd64/
      unet3d/
      fno3d/
      sheaf_mhd/
    swigs_gorgon/
      unet3d/
      fno3d/
      sheaf_mhd/
    constellaration_equilibrium/
      mlp/
      sheaf_equilibrium/
```

Per-seed run directories contain resolved configs, CSV logs, validation/test metrics, checkpoints, rollout metrics for time-dependent data, and representative figures.

## Metrics

For The Well and SWIGS, the suite reports MSE, MAE, relative L2, per-channel relative L2, magnetic divergence metrics when magnetic channels are known, energy-like drift, spectral error, rollout metrics, inference time, and parameter count.

For ConStellaration, the suite reports MSE, MAE, relative L2, per-target relative error, inference time, and parameter count.

Seed aggregation includes means, standard deviations, standard errors, Student-t confidence intervals, deterministic bootstrap confidence intervals, medians, and interquartile ranges. Pairwise comparisons include `sheaf_mhd` vs `unet3d`, `sheaf_mhd` vs `fno3d`, and `sheaf_equilibrium` vs `mlp`.

## Limitations

- The Well `MHD_64` is a uniform-grid benchmark, so it tests structure preservation and 3D scalability more than arbitrary cell-complex geometry.
- SWIGS is more relevant for bounded/coupled MHD structure, but HDF5 field-name conventions may require dataset inspection and automatic field mapping.
- ConStellaration is an equilibrium surrogate problem, not a time-evolution benchmark.
- This implementation approximates sheaf/cell-complex structure on structured arrays; a later unstructured version should use explicit cells, faces, edges, and incidence matrices.
