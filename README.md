# RLS

This repository contains Julia research code for pseudospectrum computation on complex grids. The code evaluates `sigma_min(C - zI)` over a recursion-ordered grid to compute pseudospectral level-set data and compares three solvers:

- `psInvLanc`: inverse Lanczos baseline.
- `psTriRecy`: triangular recursive recycling method.
- `psPrecRecy`: preconditioned recursive recycling method.

The repository is script-oriented rather than packaged as a Julia module. The implementation is loaded with `include("src/RLS.jl")`.

## Requirements

The code was tested with Julia 1.12.4. Install the Julia dependencies from the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The dependency list is in `Project.toml`. The most important external packages are `KrylovKit`, `MKLSparse`, `MatrixMarket`, `MAT`, `Plots`, `ToeplitzMatrices`, `AMD`, `Bumper`, `DataStructures`, and `Parameters`.

## Code and Data Files

Top-level files:

- `Project.toml`: Julia project environment and direct dependency list.
- `README.md`: this file.
- `LICENSE`: MIT license.

Core implementation:

- `src/RLS.jl`: loads all source files and external dependencies.
- `src/baseline/psInvLanc.jl`: inverse Lanczos baseline and `PsInvLancControl`.
- `src/core/psTriRecy.jl`: triangular recursive recycling solver and `PsTriRecyControl`.
- `src/core/psPrecRecy.jl`: preconditioned recursive recycling solver and `PsPrecRecyControl`.
- `src/core/recuGrid.jl`: recursion grid construction, recursion ordering, and helpers for converting results back to natural grid order.
- `src/core/recySpace.jl`: recycled subspace storage and update routines.
- `src/core/lobpcgSvd.jl`: LOBPCG-based singular-value iteration used by the recycling solvers.
- `src/core/fastRaylRitz.jl`: Rayleigh-Ritz helper routines.
- `src/types/RLSMatrices.jl`: matrix wrapper types used by shifted and preconditioned operators.
- `src/types/RLSLinearAlgebra.jl`: local linear algebra helpers.
- `src/preconditioner/RLSPreconditioner.jl`: preconditioner interface and shifted-preconditioner construction.
- `src/preconditioner/hsl_preconditioner.jl`, `ichsd.jl`, `ictkl3.jl`, `ictkl3_new.jl`, `permute_matrix.jl`, `scale.jl`: incomplete-factorization and scaling routines used by `psPrecRecy`.
- `src/utils/loadmtx.jl`: Matrix Market data loader used by the `af23560` example.

Example drivers:

- `example/af23560/af23560.jl`: loads `data/af23560.mtx`, sets `normC = 645.74001457933`, and defines the default grid.
- `example/sprand/sprand.jl`: builds a sparse random test matrix with `N = 4000`, `Random.seed!(20260318)`, `normC = 3.2`, and the default grid.
- `example/skewlap3d/skewlap3d.jl`: builds the 3D skew Laplacian test matrix with `N = 25`, `normC = 7471.59`, and the default grid.
- `example/laser/laser.jl`: builds the Landau laser matrix with `landau_demo(5000, 40*pi)` and the default grid.
- `example/basor/basor.jl`: builds the Toeplitz BASOR matrix with `basor_mat(2000)` and the default grid.
- `example/*/*_psInvLanc.jl`: runs `psInvLanc` for the corresponding matrix.
- `example/*/*_psTriRecy.jl`: runs `psTriRecy` for the corresponding matrix.
- `example/*/*_psPrecRecy.jl`: runs `psPrecRecy` where that method is included for the example.
- `example/grid_cli.jl`: parses grid overrides from command-line arguments.
- `example/invlan_control_cli.jl`, `tri_control_cli.jl`, `pred_control_cli.jl`: parse solver-control overrides from command-line arguments.
- `example/output_paths.jl`: controls output directory selection.

Data files:

- `data/af23560.mtx`: Matrix Market sparse matrix used by `example/af23560/af23560.jl`.

The other examples generate their matrices directly in Julia code, so no additional data files are required.

## Default Pseudospectrum Settings

The following table summarizes the default paper-scale matrix and grid settings. These values are encoded in the corresponding `example/<case>/<case>.jl` files and can be overridden from the command line.

| Case | Matrix definition | Grid rectangle | Grid size | Recursion depth |
| --- | --- | --- | --- | --- |
| `af23560` | `data/af23560.mtx` | `[-1.6, 0.8] x [-1.55, 1.55]` | `200 x 200` | `k = 20` for recycling methods, `k = 1` for `psInvLanc` |
| `sprand` | sparse random matrix, `N = 4000`, seed `20260318` | `[0.45, 0.55] x [0.0, 0.1]` | `200 x 200` | `k = 20` for recycling methods, `k = 1` for `psInvLanc` |
| `skewlap3d` | 3D skew Laplacian, `N = 25` | `[-550.0, 100.0] x [-200.0, 200.0]` | `200 x 200` | `k = 20` for recycling methods, `k = 1` for `psInvLanc` |
| `laser` | `landau_demo(5000, 40*pi)` | `[-1.1, 1.2] x [-1.1, 1.1]` | `200 x 200` | `k = 20` for `psTriRecy`, `k = 1` for `psInvLanc` |
| `basor` | `basor_mat(2000)` | `[-4.0, 6.5] x [-5.0, 2.0]` | `200 x 200` | `k = 20` for `psTriRecy`, `k = 1` for `psInvLanc` |

The solver-control defaults are defined in each entry script as `default_control`. The common paper-scale values are:

- `psInvLanc`: `PsInvLancControl(tol = 1e-4, continue_ = false)`.
- `psTriRecy`: `tol = 1e-4`, `p = 60`, `p_min = 1` where set explicitly, `recy_tau = 1.6`, `implict_recycle = true`; `normC` is set per example.
- `psPrecRecy`: `tol = 1e-4`, `main_tau = 1.6`, `gap_tau = 1e-10`, `tol_pred = 1e-3`, `pred_tau = 1.6`, `idx_cutoff_pred = 0.01`, `r_main = 6`, `p_main = 60`, `r_pred = 40`, `p_pred = 20`, `hsl_tol = 1e-3`; the HSL-style threshold parameters are set in each `*_psPrecRecy.jl` script.

For exact parameter values used by a run, inspect the `default_grid` and `default_control` blocks in the entry script, or keep the run log printed to stdout. The recycling solvers print their grid and control fields at startup.

## Running Smoke Tests

These commands use small grids and are intended only to confirm that the environment and scripts run:

```sh
julia --project=. example/sprand/sprand_psInvLanc.jl n_x=3 n_y=3 k=2
julia --project=. example/sprand/sprand_psTriRecy.jl n_x=3 n_y=3 k=2 p=6 r=2
julia --project=. example/sprand/sprand_psPrecRecy.jl n_x=3 n_y=3 k=2 p_main=6 r_main=2 p_pred=4 r_pred=2
```

## Reproducing the Numerical Runs

Run commands from the repository root. By default, outputs are written into the same directory as the script. To keep results organized, set `RLS_RUN_DIR` for each run:

```sh
mkdir -p results
```

The full default runs are:

```sh
RLS_RUN_DIR=results/af23560/psInvLanc  julia --project=. example/af23560/af23560_psInvLanc.jl
RLS_RUN_DIR=results/af23560/psTriRecy  julia --project=. example/af23560/af23560_psTriRecy.jl
RLS_RUN_DIR=results/af23560/psPrecRecy julia --project=. example/af23560/af23560_psPrecRecy.jl

RLS_RUN_DIR=results/sprand/psInvLanc  julia --project=. example/sprand/sprand_psInvLanc.jl
RLS_RUN_DIR=results/sprand/psTriRecy  julia --project=. example/sprand/sprand_psTriRecy.jl
RLS_RUN_DIR=results/sprand/psPrecRecy julia --project=. example/sprand/sprand_psPrecRecy.jl

RLS_RUN_DIR=results/skewlap3d/psInvLanc  julia --project=. example/skewlap3d/skewlap3d_psInvLanc.jl
RLS_RUN_DIR=results/skewlap3d/psTriRecy  julia --project=. example/skewlap3d/skewlap3d_psTriRecy.jl
RLS_RUN_DIR=results/skewlap3d/psPrecRecy julia --project=. example/skewlap3d/skewlap3d_psPrecRecy.jl

RLS_RUN_DIR=results/laser/psInvLanc julia --project=. example/laser/laser_psInvLanc.jl
RLS_RUN_DIR=results/laser/psTriRecy julia --project=. example/laser/laser_psTriRecy.jl

RLS_RUN_DIR=results/basor/psInvLanc julia --project=. example/basor/basor_psInvLanc.jl
RLS_RUN_DIR=results/basor/psTriRecy julia --project=. example/basor/basor_psTriRecy.jl
```

The `laser_psTriRecy.jl` script also accepts an optional `seed=<integer>` argument before the matrix is constructed:

```sh
RLS_RUN_DIR=results/laser/psTriRecy_seed20260318 julia --project=. example/laser/laser_psTriRecy.jl seed=20260318
```

## Command-Line Overrides

All example entry scripts accept grid overrides:

```sh
julia --project=. example/af23560/af23560_psTriRecy.jl n_x=50 n_y=50 k=10
```

Supported grid keys are `x_st`, `x_ed`, `y_st`, `y_ed`, `n_x`, `n_y`, and `k`.

Solver-control fields can be overridden either field-by-field:

```sh
julia --project=. example/sprand/sprand_psTriRecy.jl p=40 r=8 tol=1e-4
```

or with a full control expression:

```sh
julia --project=. example/sprand/sprand_psTriRecy.jl 'control=PsTriRecyControl(tol=1e-4,p=40,r=8)'
```

The allowed field names are the fields of `PsInvLancControl`, `PsTriRecyControl`, or `PsPrecRecyControl`, depending on the script.

## License

This code is released under the MIT License. See `LICENSE` for details.
