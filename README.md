# AstroFD

AstroFD is a 3D relativistic fluid dynamics code for jets and SN-lite scenarios.
It supports SRHD/RMHD in flat spacetime and GRHD/GRMHD on fixed backgrounds,
with configurable nozzle injection, GLM divergence cleaning, causal dissipation,
non-ideal MHD terms, two-temperature relaxation, and H/He non-equilibrium
chemistry. The code is designed for small-to-medium research simulations and
validation-driven development.

## Features at a Glance

Physics:
- SRHD, RMHD, GRHD, GRMHD with Valencia formulation on fixed metrics.
- Jet nozzle injection profiles: top-hat, taper, parabolic, with optional perturbations.
- GLM divergence cleaning (hyperbolic + damping).
- Causal dissipation (Israel-Stewart type) with IMEX subcycling.
- Two-temperature closure with electron-ion equilibration.
- H/He non-equilibrium ion chemistry with cooling/heating source terms.
- Resistive RMHD and non-ideal MHD add-ons (Hall, ambipolar, hyper-resistivity).
- SN-lite physics: parametric gravity, gain-region heating/cooling, lightbulb neutrinos.

Numerics:
- Finite-volume update on Cartesian grids.
- Reconstruction: MUSCL (MC), PPM, WENO5; limiters: MC, minmod, van Leer.
- Riemann solvers: HLLE, HLLC (hydro), HLLD and full HLLD (RMHD).
- SSPRK2/SSPRK3 time integration; IMEX subcycling for stiff sources.
- Robust primitive recovery with fallback paths and safety floors/caps.

Diagnostics and tools:
- Built-in runtime diagnostics for Lorentz factor, fluxes, divB, and SN-lite metrics.
- Cocoon and mixing diagnostics for jets.
- Post-processing: quicklook plots, spectra, structure functions, flux budgets.

## Repository Layout

- `solvers/`: main solver entrypoints (MPI).
- `core/`: reconstruction, Riemann solvers, physics, EOS, source terms.
- `config/`: example configurations for jets, GR, and SN-lite.
- `tools/`: validation suites and analysis tools.
- `docs/`: user guide, technical reference, and LaTeX writeup.
- `results/`: output directory (created at runtime).

## Requirements

- Python 3.10+ recommended.
- MPI runtime (OpenMPI or MPICH).
- `mpi4py`, `numpy`, `numba`, `scipy`, `matplotlib` (see `requirements.txt`).

Optional:
- CuPy for GPU-accelerated post-processing.

## Installation

Create and install into an isolated environment:

```bash
scripts/setup_env.sh
source scripts/env.sh
```

`scripts/env.sh` also sets `MPLCONFIGDIR` to a local writable cache (`.mplcache`) and defaults `MPLBACKEND=Agg` for headless runs.

For optional CuPy support:

```bash
INSTALL_CUPY=1 scripts/setup_env.sh
```

## Quickstart

Run a tiny SRHD test:

```bash
python solvers/srhd3d_mpi_muscl.py --config config/config_two_temp.json
```

Run a small RMHD case:

```bash
python solvers/srhd3d_mpi_muscl.py --config config/config_rmhd.json
```

Run GRHD on Kerr-Schild:

```bash
python solvers/srhd3d_mpi_muscl.py --config config/config_grhd_ks.json
```

MPI (2 ranks):

```bash
mpiexec -n 2 python solvers/srhd3d_mpi_muscl.py --config config/config_rmhd.json
```

## Configuration

Configurations live in `config/*.json`. Common keys:

Numerics:
- `RECON`: `muscl`, `ppm`, `weno`
- `LIMITER`: `mc`, `minmod`, `vanleer`
- `RIEMANN`: `hlle`, `hllc`, `hlld`, `hlld_full`
- `RK_ORDER`: `2` or `3`
- `HALO_EXCHANGE`: `blocking` or `nonblocking`

Physics:
- `PHYSICS`: `hydro`, `rmhd`, `grhd`, `grmhd`, `sn`
- `GLM_CH`, `GLM_CP`: GLM cleaning parameters
- `DISSIPATION_ENABLED`
- `TWO_TEMPERATURE`, `TEI_TAU`
- `CHEMISTRY_ENABLED`
- `RESISTIVE_ENABLED`, `RESISTIVITY`
- `NONIDEAL_MHD_ENABLED`, `HALL_*`, `AMBIPOLAR_*`, `HYPERRESIST_*`

SN-lite:
- `SN_GRAVITY_ENABLED`, `SN_GRAVITY_MODEL`
- `SN_HEATING_ENABLED`, `SN_HEATING_MODEL`
- `NEUTRINO_ENABLED`, `NEUTRINO_MODEL`
- `SN_EOS_GAMMA`, `EOS_MODE`

## Outputs

Simulation outputs live in `results/YYYY-MM-DD[/HH-MM-SS]` and include:
- `jet3d_rankXXXX_stepNNNNNN.npz`: snapshots
- `diagnostics.csv`: global Lorentz factor and inlet flux
- `centerline.csv`: axis profiles
- `divb.csv`: divB statistics for RMHD/GRMHD
- `sn_diagnostics.csv`: SN-lite metrics

## Validation

Run the full validation suite:

```bash
python tools/run_validation_suite.py
```

Quick mode:

```bash
python tools/run_validation_suite.py --quick
```

Baseline regression checks (small hydro + RMHD cases):

```bash
python tools/validate_baselines.py
```

To refresh baselines after controlled updates:

```bash
python tools/validate_baselines.py --update
```

Error-norm regression checks (shock tube + Orszag–Tang):

```bash
python tools/validate_error_norms.py
```

To refresh error-norm references after controlled updates:

```bash
python tools/validate_error_norms.py --update
```

CI uses the quick mode on each push. For a local CI-like run:

```bash
scripts/ci_quick.sh
```

Nightly CI runs the full validation suite (including baselines and MPI restart checks).
If a CI run fails, open the Actions log and look for the first non-zero exit step.

## Documentation

- `docs/USER_GUIDE.md`: setup, usage, troubleshooting
- `docs/TECHNICAL_REFERENCE.md`: equations, numerics, algorithms
- `docs/PRODUCTION_READINESS.md`: roadmap checklist
- `docs/astrofd.tex`: LaTeX writeup with references

Build the PDF writeup:

```bash
cd docs
pdflatex astrofd.tex
bibtex astrofd
pdflatex astrofd.tex
pdflatex astrofd.tex
```

## Troubleshooting

- MPI warnings about socket binding may appear in sandboxed environments;
  runs typically complete despite the warnings.
- For CuPy post-processing, install CuPy and pass `--backend cupy` to tools.

## Citation

If you use AstroFD in publications, cite this repository and the references
listed in `docs/astrofd_refs.bib`.
