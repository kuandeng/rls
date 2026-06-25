# RLS

This repository contains Julia research code for parameterized singular-value experiments with recursive-grid evaluation and recycling. The main algorithms are:

- `psInvLanc`: inverse Lanczos baseline.
- `psTriRecy`: triangular recursive recycling method.
- `psPrecRecy`: preconditioned recursive recycling method.

## Setup

The code was tested with Julia 1.12.4. From the repository root, instantiate the project environment:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The source tree is script-oriented rather than packaged as a Julia module. Load the implementation with:

```julia
include("src/RLS.jl")
```

## Repository Layout

- `src/`: core implementation, matrix wrappers, preconditioners, and baseline methods.
- `example/`: runnable experiment drivers.
- `data/`: matrix data used by the included examples.

## Running Examples

Each example driver accepts command-line overrides of grid sizes and control fields. Outputs are written next to the script by default. Set `RLS_RUN_DIR` to redirect generated files.

Small smoke-test examples:

```sh
julia --project=. example/sprand/sprand_psInvLanc.jl n_x=3 n_y=3 k=2
julia --project=. example/sprand/sprand_psTriRecy.jl n_x=3 n_y=3 k=2 p=6 r=2
julia --project=. example/sprand/sprand_psPrecRecy.jl n_x=3 n_y=3 k=2 p=6 r=2
```

Available experiment families:

- `example/af23560/`: matrix-market data example.
- `example/sprand/`: sparse random matrix example.
- `example/skewlap3d/`: 3D skew Laplacian example.
- `example/laser/`: laser example.
- `example/basor/`: Toeplitz/BASOR example.

For larger paper-scale runs, start from the corresponding example file and override grid and control parameters on the command line, for example:

```sh
RLS_RUN_DIR=logs/sprand julia --project=. example/sprand/sprand_psTriRecy.jl n_x=100 n_y=100 k=20 p=60 r=10
```

Control structures can also be overridden through `control=...`, for example:

```sh
julia --project=. example/sprand/sprand_psTriRecy.jl 'control=PsTriRecyControl(tol=1e-4,p=40,r=8)'
```

## License

This code is released under the MIT License. See `LICENSE` for details.
