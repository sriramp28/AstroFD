# AstroFD Production Readiness Checklist

This checklist summarizes remaining work to move from research code to
production-grade. Items are grouped by priority.

## Validation and Verification

- Standard SRHD/RMHD shock tube suite (1D/2D) with error norms.
- Baseline regression checks (initial hydro + RMHD cases in place; expand coverage).
- SN-lite and GR baselines now tracked; expand to more cases and higher resolutions.
- RMHD Orszag-Tang vortex and rotor tests.
- Alfvén and fast/slow magnetosonic convergence tests.
- GRMHD benchmarks (magnetized Bondi, Fishbone-Moncrief torus).
- Quantitative thresholds for divB, shock radius, and Lorentz factor.
- Regression baselines for PPM/WENO + RK3 combinations.

## Physical Fidelity

- Expanded EOS options (tabulated gamma profiles, relativistic Synge).
- Improved RMHD primitive recovery for extreme magnetization.
- Extended chemistry and cooling networks for SN-lite.
- Optional non-equilibrium radiation transport (beyond lightbulb).

## Performance and Scaling

- Profile hotspots for large grids.
- Improve strong/weak scaling for multi-node MPI.
- Optional shared-memory parallelism within ranks.
- Solver-side GPU backend (not only post-processing).

## Robustness and Reproducibility

- Deterministic inlet perturbations and fixed seeds.
- Stronger restart coverage and versioned checkpoint metadata (MPI restart checks added).
- Parameter validation with schema checks.
- Unit tests for EOS, reconstruction, and primitive recovery.

## Engineering

- CI pipelines (quick smoke + docs build in place; expand to lint + full validation).
- Structured logging and metadata outputs.
- Release tagging and semantic versioning.

## Documentation

- Expanded user guide with step-by-step tutorials.
- Detailed LaTeX writeup with method derivations and test results.
- Example notebooks or scripts for analysis workflows.
