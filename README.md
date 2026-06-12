# Cellular Sheaf Neural Operators for Structure-Preserving PDE Surrogate Modeling

This repository implements an experimental suite for evaluating **Cellular Sheaf Neural Operators (CSNOs)** as structure-preserving surrogate models for constrained PDE systems. The problem is that many neural PDE surrogates treat simulation states as homogeneous grid-channel tensors, even when the underlying numerical discretization stores different physical quantities on cells, faces, edges, vertices, boundaries, or interfaces. CSNO addresses this mismatch by representing PDE states as cochain-valued fields on a cell complex, coupling local feature spaces through learned restriction maps, and using incidence/Hodge-informed communication and compatibility-preserving update heads.

Magnetohydrodynamics (MHD) is used as a demanding testbed because it combines fluid variables, magnetic fields, geometric constraints, fluxes, and long-horizon rollout behavior. The same computational idea is broader: learned surrogate models should respect the algebraic and geometric organization of the discretization when that structure matters for stability, compatibility, or physical consistency.

The paper-facing model name is **Cellular Sheaf Neural Operator**. The internal model key is `sheaf_mhd`, implemented by `SheafMHDOperator`.

## Why Cellular Sheaf Neural Operators?

Many scientific-computing problems involve PDE states with typed geometric roles:

- conserved quantities stored on volume cells;
- fluxes naturally associated with faces;
- circulations or electromotive quantities associated with edges;
- potentials, boundary values, or interface quantities associated with vertices or lower-dimensional cells;
- compatibility constraints arising from discrete differential identities.

Standard neural PDE surrogates often flatten these distinctions into a single grid-channel representation. That is convenient for U-Nets, FNOs, and other tensor backbones, but it can obscure the computational structure used by finite-volume methods, discrete exterior calculus, finite element exterior calculus, compatible discretizations, and constrained-transport schemes.

CSNO is designed to make this structure explicit. It can:

- represent PDE variables as cochains on cells of different dimensions;
- maintain separate learned feature spaces, or fibers, for different cell types;
- learn restriction/coupling maps across incidences;
- use incidence- and Hodge-informed message passing;
- route selected learned updates through coboundary or flux maps;
- preserve selected compatibility residuals by construction in the native cochain representation;
- evaluate models using rollout, spectral, divergence, parameter-efficiency, and structure-sensitive diagnostics rather than only one-step tensor error.

The MHD implementation specializes this general idea by representing magnetic flux on faces, learned electromotive quantities on edges, and fluid variables on volume cells. This provides a concrete constrained-PDE setting where cochain placement and compatibility-preserving updates are meaningful.

## Repository Scope

This codebase supports three experimental tracks:

1. **The Well `MHD_64`**
   - Main time-dependent constrained-PDE benchmark.
   - Used to evaluate one-step prediction, rollout stability, spectral behavior, magnetic-divergence diagnostics, parameter count, and inference cost.

2. **SWIGS/Gorgon MHD**
   - Experimental bounded/coupled MHD loader.
   - Useful for exploring more complex magnetosphere-ionosphere-style data organization.
   - Not necessarily part of the main paper-facing result set unless explicitly enabled and documented.

3. **ConStellaration equilibrium subset**
   - Optional structured equilibrium-regression track.
   - Tests whether grouped sheaf-style representations and learned restriction-style coupling help beyond time-dependent rollout prediction.

## Datasets

The code does **not** download data and does not require internet access. Place datasets under the following local paths.

### The Well `MHD_64`

```text
datasets/wells/datasets/MHD_64/
```

The loader also accepts the older parent root:

```text
datasets/wells/
```

and resolves the nested `datasets/` directory automatically.

The loader reads the local HDF5 schema directly, using:

```text
t0_fields/density
t1_fields/magnetic_field
t1_fields/velocity
```

even though scalar and vector fields live under different groups. Each one-frame state is ordered as seven channels:

```text
density + magnetic_field[3] + velocity[3]
x: [7*n_input_frames, X, Y, Z]
y: [7*n_output_frames, X, Y, Z]
```

The Well `MHD_64` is treated as a structured-grid, time-dependent PDE surrogate-modeling benchmark. For CSNO, the preprocessed grid is associated with a cubical cell complex. Cell-centered variables are mapped to volume-cell cochains, magnetic components are mapped to face-based magnetic-flux cochains, and learned electromotive quantities are represented on edges. Predictions are projected back to the common channel-first tensor format so CSNO, U-Net, and FNO can be evaluated on the same supervised targets.

### SWIGS/Gorgon MHD

```text
datasets/swigs_gorgon/
```

The SWIGS/Gorgon loader recursively discovers `.h5`, `.hdf5`, and `.hdf` files under this root. The current supported schema intentionally ignores ionosphere `IS` files, indexes magnetosphere `MS` files by shock directory and timestamp, requires `P` plus `Bvec_c` at both `t` and `t+dt`, and downsamples the large `480x320x320` arrays by default before training.

It caches an index at:

```text
datasets/swigs_gorgon/.swigs_index.json
```

SWIGS is included as an experimental bounded/coupled MHD extension. It is relevant to the broader sheaf/restriction-map story because different variables, regions, and boundary/interface-type quantities may need to be coupled. Treat this track as less standardized than The Well unless the exact data schema has been inspected and confirmed.

### Optional ConStellaration Equilibrium Subset

```text
datasets/constellaration_subset/
  boundaries_and_metrics.jsonl
  vmecpp_wout_finite_beta_3pct.jsonl
```

This optional track is a supervised fusion-equilibrium regression problem, not a time-evolution rollout benchmark. The loader joins JSONL rows by configuration identifiers when possible, parses JSON-valued string columns such as `boundary.json`, `metrics.json`, and WOut `json`, flattens numeric leaves into input/output vectors, standardizes features, and evaluates equilibrium surrogate models separately from time-dependent PDE rollouts.

If the folder is missing, only the ConStellaration track is skipped with a logged warning.

## One-Command Full Experiment

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

When `SMOKE_TEST = True`, the script uses one seed, one epoch, and small sample caps for quick local checks. The committed default is:

```python
SMOKE_TEST = False
```

## Default Suite

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

For time-dependent PDE surrogate modeling, the suite reports:

- MSE;
- MAE;
- relative L2 error;
- per-channel relative L2 error;
- magnetic-divergence diagnostics when magnetic channels are known;
- energy-like drift;
- spectral error;
- autoregressive rollout metrics;
- inference time;
- parameter count.

For ConStellaration, the suite reports:

- MSE;
- MAE;
- relative L2 error;
- per-target relative error;
- inference time;
- parameter count.

Seed aggregation includes means, standard deviations, standard errors, Student-t confidence intervals, deterministic bootstrap confidence intervals, medians, and interquartile ranges. Pairwise comparisons include:

```text
sheaf_mhd vs unet3d
sheaf_mhd vs fno3d
sheaf_equilibrium vs mlp
```

## Interpreting the Results

The intended interpretation is computational rather than purely application-specific.

A model that minimizes one-step pointwise error is not necessarily the best learned time-stepper. For scientific-computing use cases such as forecasting, design iteration, uncertainty propagation, control, and inverse problems, the surrogate is often used repeatedly. In that setting, rollout growth, compatibility drift, spectral distortion, parameter efficiency, and constraint violation can matter as much as or more than one-step reconstruction.

CSNO is designed to test whether encoding discretization structure into the model can improve these structure-sensitive diagnostics. The expected tradeoff is:

- dense tensor models such as U-Net may be strong one-step regressors;
- FNO-style models may be strong global spectral baselines;
- CSNO may be preferable when typed variables, compatibility constraints, rollout behavior, and compact parameterization are important.

## Limitations

- The Well `MHD_64` is a uniform-grid benchmark, so it tests structure-aware surrogate modeling and 3D scalability more than arbitrary unstructured cell-complex geometry.
- The current CSNO implementation maps structured arrays to cubical cochain representations; a later unstructured version should use explicit cells, faces, edges, incidence matrices, and geometry-dependent Hodge data from production meshes.
- Exact preservation claims apply to native cochain updates, not necessarily to every projected grid-space diagnostic.
- Projected magnetic-divergence metrics depend on interpolation, projection, finite-difference diagnostics, and boundary handling.
- CSNO introduces additional modeling choices beyond U-Net or FNO, including variable placement, learned restriction parameterization, Hodge weights, cochain projection, and task-specific update heads.
- SWIGS is more relevant for bounded/coupled MHD structure, but HDF5 field-name conventions may require dataset inspection and automatic field mapping.
- ConStellaration is an equilibrium surrogate problem, not a time-evolution benchmark.
- CSNO may be less attractive when the only goal is fastest inference or minimum one-step tensor error.

## Future Directions

Natural extensions include:

- unstructured meshes and non-cubical cell complexes;
- adaptive meshes and boundary-aware projections;
- explicit geometry-conditioned restriction maps;
- stronger ablations of restriction maps, Hodge terms, projection choices, and compatibility-preserving heads;
- optimized sparse and mixed dense-sparse kernels;
- longer rollout horizons;
- additional constrained PDE systems beyond MHD;
- integration with conservative, compatible, or finite-element-style surrogate-modeling pipelines.

## Citation

If you use this repository in academic work, please cite the arXiv/SSRN preprint:

```bibtex
@misc{shikhman2026csno,
      title={Cellular Sheaf Neural Operators for Structure-Preserving Surrogate Modeling of Constrained PDEs}, 
      author={Lennon J. Shikhman and Shane Gilbertie},
      year={2026},
      eprint={2606.00937},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.00937}, 
}
```
