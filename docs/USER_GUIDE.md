# AstroFD User Guide

## Overview

AstroFD is a 3D relativistic fluid dynamics code with SRHD, RMHD, GRHD, and GRMHD solvers, plus SN-lite physics, two-temperature closures, resistive and non-ideal MHD, ion chemistry, and diagnostics suited for relativistic jets and core-collapse scenarios.

## Installation and Setup

1) Create the Python environment and install dependencies:

```bash
scripts/setup_env.sh
```

2) Apply default thread settings (optional but recommended):

```bash
source scripts/env.sh
```

Dependencies are listed in `requirements.txt`. For GPU-accelerated post-processing, install CuPy:

```bash
INSTALL_CUPY=1 scripts/setup_env.sh
```

MPI runtime (OpenMPI or MPICH) is required for multi-rank runs; this repository uses `mpi4py` for MPI bindings.

## Running Simulations

All runs use:

```bash
python solvers/srhd3d_mpi_muscl.py --config <config.json>
```

Example runs:

- SRHD jet (baseline): `config/config.json`
- RMHD jet (GLM + poloidal field): `config/config_rmhd.json`
- GRHD Kerr-Schild: `config/config_grhd_ks.json`
- GRMHD Kerr-Schild: `config/config_grmhd.json`
- SN-lite stalled shock: `config/config_sn_stalled_shock.json`
- SN-lite neutrino lightbulb: `config/config_neutrino_lightbulb.json`
- Static nested refinement: `config/config_adaptivity_smoke.json`

MPI usage (2 ranks):

```bash
mpiexec -n 2 python solvers/srhd3d_mpi_muscl.py --config config/config_rmhd.json
```

## Key Configuration Options

Core numerics:
- `RECON`: `muscl`, `ppm`, `weno`
- `LIMITER`: `mc`, `minmod`, `vanleer`
- `RIEMANN`: `hlle`, `hllc`, `hlld`, `hlld_full`
- `RK_ORDER`: `2` or `3`
- `HALO_EXCHANGE`: `blocking` or `nonblocking`

Physics:
- `PHYSICS`: `hydro`, `rmhd`, `grhd`, `grmhd`, `sn`
- `GLM_CH`, `GLM_CP`: GLM divergence cleaning parameters
- `DISSIPATION_ENABLED`: Israel-Stewart causal dissipation (hydro/grhd)
- `TWO_TEMPERATURE`, `TEI_TAU`: two-temperature relaxation
- `CHEMISTRY_ENABLED`: H/He non-equilibrium ionization
- `RESISTIVE_ENABLED`, `RESISTIVITY`: resistive RMHD
- `NONIDEAL_MHD_ENABLED`, `HALL_*`, `AMBIPOLAR_*`, `HYPERRESIST_*`
- `RADIATION_COUPLING_ENABLED`: simple gas/electron radiation coupling
- `KINETIC_EFFECTS_ENABLED`: shear/current heating

SN-lite:
- `SN_GRAVITY_ENABLED`, `SN_GRAVITY_MODEL` (`point_mass`, `monopole`, `monopole_plus_point`)
- `SN_HEATING_ENABLED`, `SN_HEATING_MODEL` (gain profiles)
- `NEUTRINO_ENABLED`, `NEUTRINO_MODEL` (`lightbulb`)
- `SN_EOS_GAMMA`, `EOS_MODE` (`gamma`, `piecewise`, `table`)

Adaptivity (single rank):
- `ADAPTIVITY_ENABLED`: enable static nested refinement
- `ADAPTIVITY_REFINEMENT`: integer refinement factor
- `ADAPTIVITY_REGION`: `[xlo,xhi,ylo,yhi,zlo,zhi]`

## Output and Diagnostics

Outputs live in `results/YYYY-MM-DD[/HH-MM-SS]` and include:
- `jet3d_rankXXXX_stepNNNNNN.npz`: snapshot data
- `diagnostics.csv`: global max Gamma and inlet flux
- `centerline.csv`: midline profile along x
- `divb.csv`: divB statistics (RMHD/GRMHD)
- `sn_diagnostics.csv`: SN shock radius, gain mass, heating efficiency
- `cocoon.csv`, `mixing.csv`: mixing and cocoon diagnostics when enabled
- `perf.csv`: optional performance data

Quick-look plots:

```bash
python tools/quickview.py
```

## Validation and Tests

Run the main suite:

```bash
python tools/run_validation_suite.py
```

Quick mode:

```bash
python tools/run_validation_suite.py --quick
```

Individual suites:
- `python tools/run_smoke_suite.py`
- `python tools/validate_schemes.py`
- `python tools/validate_hlld.py`
- `python tools/validate_gr.py`
- `python tools/validate_gr_orthonormal.py`
- `python tools/validate_restart.py`
- `python tools/run_sn_tests.py`

## Post-processing

- `tools/flux_budget.py`: fluxes through planes or shells
- `tools/spectra.py`: velocity spectrum
- `tools/structure_function.py`: 2nd-order structure function

Use `--backend cupy` if CuPy is available.
