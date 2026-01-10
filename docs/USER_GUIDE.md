# AstroFD User Guide

## Overview

AstroFD is a 3D relativistic fluid dynamics solver focused on jets and
SN-lite scenarios. It advances SRHD/RMHD in flat spacetime and GRHD/GRMHD
on fixed backgrounds, with configurable nozzle injection, multiple
reconstruction and Riemann options, and extended physics modules.

This guide covers installation, running simulations, configuration, and
basic troubleshooting.

## Installation

1) Create and install the Python environment:

```bash
scripts/setup_env.sh
```

2) Apply default thread settings (optional but recommended):

```bash
source scripts/env.sh
```

Optional CuPy for post-processing:

```bash
INSTALL_CUPY=1 scripts/setup_env.sh
```

## Running Simulations

All runs use:

```bash
python solvers/srhd3d_mpi_muscl.py --config <config.json>
```

Example runs:
- SRHD jet: `config/config.json`
- RMHD jet: `config/config_rmhd.json`
- GRHD Kerr-Schild: `config/config_grhd_ks.json`
- GRMHD Kerr-Schild: `config/config_grmhd.json`
- SN-lite stalled shock: `config/config_sn_stalled_shock.json`
- SN-lite neutrino lightbulb: `config/config_neutrino_lightbulb.json`
- Static nested refinement: `config/config_adaptivity_smoke.json`

MPI usage (2 ranks):

```bash
mpiexec -n 2 python solvers/srhd3d_mpi_muscl.py --config config/config_rmhd.json
```

## Tutorials

These walkthroughs use small grids for quick validation runs and point to
the relevant output files. Scale up later by increasing `NX/NY/NZ` and
adjusting the end time.

### Tutorial A: SRHD Jet Smoke Test

1) Run a small SRHD jet:

```bash
python solvers/srhd3d_mpi_muscl.py --config config/config.json
```

2) Inspect outputs in the latest `results/YYYY-MM-DD/` directory:
   - `diagnostics.csv` for max Lorentz factor and inlet flux
   - `centerline.csv` for axial profiles

3) Quicklook slice plot:

```bash
python tools/quickview.py --field rho
```

### Tutorial B: RMHD Jet with GLM

1) Run RMHD with GLM and a poloidal field:

```bash
python solvers/srhd3d_mpi_muscl.py --config config/config_rmhd.json
```

2) Check `divb.csv` and confirm divB remains small relative to mean B.

3) Optional mixing diagnostics (if tracers enabled):

```bash
python tools/mixing_layer.py --tracer-index 0
python tools/cocoon_pressure.py --tracer-index 0
```

### Tutorial C: GRHD Kerr-Schild

1) Run GRHD on a fixed Kerr-Schild metric:

```bash
python solvers/srhd3d_mpi_muscl.py --config config/config_grhd_ks.json
```

2) Review `diagnostics.csv` and ensure stability with the chosen CFL.

### Tutorial D: SN-lite Stalled Shock

1) Run the stalled-shock configuration:

```bash
python solvers/srhd3d_mpi_muscl.py --config config/config_sn_stalled_shock.json
```

2) Review SN diagnostics:
   - `sn_diagnostics.csv` for shock radius and gain mass
   - `diagnostics.csv` for global quantities

### Tutorial E: SN-lite Lightbulb Heating

1) Enable the lightbulb model:

```bash
python solvers/srhd3d_mpi_muscl.py --config config/config_neutrino_lightbulb.json
```

2) Compare shock radius evolution to the stalled-shock baseline.

## Configuration Structure

Configurations are JSON files in `config/`. The following groups are
common across configurations.

### Grid and Time

- `NX`, `NY`, `NZ`: grid resolution.
- `Lx`, `Ly`, `Lz`: domain size.
- `T_END`: end time.
- `OUT_EVERY`: output cadence in steps.
- `PRINT_EVERY`: diagnostic print cadence.
- `CFL`: CFL number.

### Jet Nozzle and Ambient

- `JET_RADIUS`, `JET_CENTER`: nozzle geometry.
- `NOZZLE_PROFILE`: `top_hat`, `taper`, `parabolic`.
- `SHEAR_THICK`: shear layer thickness.
- `NOZZLE_TURB`: enable inlet perturbations.
- `NOZZLE_PERTURB`: `random`, `sinusoidal`.
- `TURB_VAMP`, `TURB_PAMP`: perturbation amplitudes.
- `GAMMA_JET`, `ETA_RHO`, `P_EQ`: jet parameters.
- `RHO_AMB`, `P_AMB`, `VX_AMB`, `VY_AMB`, `VZ_AMB`: ambient state.

### Numerics

- `RECON`: `muscl`, `ppm`, `weno`.
- `LIMITER`: `mc`, `minmod`, `vanleer` (for MUSCL).
- `RIEMANN`: `hlle`, `hllc`, `hlld`, `hlld_full`.
- `RK_ORDER`: `2` or `3`.
- `HALO_EXCHANGE`: `blocking` or `nonblocking`.
- `NG`: ghost zones.
- `P_MAX`, `V_MAX`: safety caps.

### Adaptivity

- `ADAPTIVITY_ENABLED`: enable nested refinement (single-rank or MPI x-slab).
- `ADAPTIVITY_MODE`: `nested_static`, `nested_dynamic`, `nested_static_mpi`, `nested_dynamic_mpi`.
- `nested_dynamic_mpi` computes a global refine box via MPI reductions and refines only ranks that intersect it.
- `ADAPTIVITY_REFINEMENT`: integer refinement factor (>= 2).
- `ADAPTIVITY_REGION`: `[xlo,xhi,ylo,yhi,zlo,zhi]` (static mode).
- `ADAPTIVITY_FIELD`: field used for dynamic refinement (`rho` or `p`).
- `ADAPTIVITY_GRAD_THRESHOLD`: gradient threshold for dynamic region selection.
- `ADAPTIVITY_BUFFER`: buffer cells around detected gradients.
- `ADAPTIVITY_UPDATE_EVERY`: update cadence for dynamic refinement.

### Physics Selection

- `PHYSICS`: `hydro`, `rmhd`, `grhd`, `grmhd`, `sn`.
- `GLM_CH`, `GLM_CP`: GLM divergence cleaning.
- `B_INIT`: `none`, `poloidal`, `toroidal`.
- `B0`: nozzle field magnitude.
- `DISSIPATION_ENABLED`: causal dissipation.
- `TWO_TEMPERATURE`, `TEI_TAU`: two-temperature relaxation.
- `CHEMISTRY_ENABLED`: H/He ionization network.
- `RESISTIVE_ENABLED`, `RESISTIVITY`: resistive RMHD.
- `NONIDEAL_MHD_ENABLED`: Hall/ambipolar/hyper-resistive terms.
- `RMHD_INIT`: `uniform` or `riemann` (1D RMHD shock-tube setup).
- `RMHD_RIEMANN_X0`: split location in x for the Riemann problem.
- `RMHD_RIEMANN_LEFT`, `RMHD_RIEMANN_RIGHT`: 9-component RMHD states
  `[rho, vx, vy, vz, p, Bx, By, Bz, psi]`.

Passive tracers:
- `N_TRACERS`: number of passive scalars.
- `TRACER_NAMES`: optional list of names.
- `TRACER_NOZZLE_VALUES`, `TRACER_AMB_VALUES`: inlet/ambient values.
Diagnostics for tracers:
- `DIAG_COCOON_TRACER_IDX`, `DIAG_COCOON_TRACER_MIN`, `DIAG_COCOON_TRACER_MAX`.
- `DIAG_MIXING_TRACER_IDX`.
Additional diagnostics:
- `DIAG_PLANE_ENABLED`, `DIAG_PLANE_X`: plane flux budgets (mass, momentum, energy).
- `DIAG_SPECTRA_ENABLED`: 1D centerline spectra for ρ and v_x.
- `DIAG_STRUCTURE_ENABLED`, `DIAG_STRUCTURE_MAX_LAG`: centerline structure functions.

### GR and Metric Controls

- `GR_METRIC`: `minkowski`, `schwarzschild`, `kerr_schild`.
- `GR_MASS`, `GR_SPIN`: parameters for fixed metrics.
- `GR_ORTHONORMAL`: enable orthonormal-frame flux evaluation.

### SN-lite Controls

- `SN_GRAVITY_ENABLED`, `SN_GRAVITY_MODEL`.
- `SN_HEATING_ENABLED`, `SN_HEATING_MODEL`.
- `NEUTRINO_ENABLED`, `NEUTRINO_MODEL`.
- `SN_EOS_GAMMA`, `EOS_MODE`.

### Restart and Checkpointing

- `CHECKPOINT_EVERY`: periodic checkpoints (0 disables).
- `RESTART_PATH`: path to checkpoint to resume.

## Outputs and Diagnostics

Outputs are written to `results/YYYY-MM-DD[/HH-MM-SS]`:
- `jet3d_rankXXXX_stepNNNNNN.npz`: field snapshots.
- `diagnostics.csv`: global max Lorentz factor and inlet flux.
- `centerline.csv`: axial centerline profile.
- `divb.csv`: divB statistics for RMHD/GRMHD.
- `sn_diagnostics.csv`: shock radius, gain mass, heating efficiency.
- `cocoon.csv`, `mixing.csv`: jet cocoon and mixing metrics.

Quicklook plots:

```bash
python tools/quickview.py
```

## Validation

Run all suites:

```bash
python tools/run_validation_suite.py
```

Quick mode:

```bash
python tools/run_validation_suite.py --quick
```

Targeted suites:
- `python tools/validate_schemes.py`
- `python tools/validate_hlld.py`
- `python tools/validate_gr.py`
- `python tools/validate_gr_orthonormal.py`
- `python tools/validate_restart.py`
- `python tools/run_sn_tests.py`

## Documentation and Paper Assets

Generate paper-ready figures and diagnostics tables from the latest run:

```bash
python tools/make_doc_figures.py
```

To use a specific run directory:

```bash
python tools/make_doc_figures.py --run-dir results/2026-01-08
```

Outputs are written to `docs/figures/` and are referenced by
`docs/astrofd.tex`. Rebuild the PDF with:

```bash
cd docs
pdflatex astrofd.tex
bibtex astrofd
pdflatex astrofd.tex
pdflatex astrofd.tex
```

Generate the full configuration table (CSV + LaTeX):

```bash
python tools/gen_config_table.py
```

## Troubleshooting

- MPI warnings about socket binding can appear in sandboxed or restricted
  environments; runs typically complete despite the warnings.
- CI failures are based on non-zero exit codes; warnings do not fail runs.
- For nightly CI failures, check the first failed step in the Actions log and
  rerun that tool locally with the same command.
- If a run crashes early, reduce `CFL` and tighten `P_MAX`/`V_MAX`.
- For large grids, disable `DEBUG` and reduce output cadence.
- For CuPy backends, ensure the CUDA toolkit is installed and compatible
  with the CuPy version.
