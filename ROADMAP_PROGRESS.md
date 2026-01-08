# AstroFD Roadmap Progress

This file tracks the current plan and completion status so work can resume easily.

## Current focus (sn-lite-plasma branch)

1. SN-lite physics accuracy
   - Monopole/enclosed-mass gravity (done)
   - Detailed cooling/heating profiles (done)
2. Proper EOS integration (piecewise/tabulated, gamma-law fallback)
   - Variable-gamma EOS module wired into SRHD/GRHD/RMHD/GRMHD (done)
3. Numerics/engineering
   - Checkpoint/restart (done)
   - Performance hooks (done)
   - Static nested refinement (done; single-rank)
   - Threading hooks (done; Numba/OpenMP settings)
   - Nonblocking halo exchange option (done)
4. Diagnostics/analysis
   - SN-specific metrics (shock radius, gain mass, heating efficiency) (done)
   - Expanded validation suite (done)
   - Cocoon pressure + mixing diagnostics (done)
   - Shell flux budgets (done)
5. Neutrino transport (deferred)

## Completed highlights

- SRHD/RMHD/GRHD/GRMHD core with recon/riemann switches, GLM, HLLD full option.
- Israel–Stewart dissipation (IMEX relaxation + advection).
- Two-temperature, cooling/heating, resistive RMHD, passive tracers.
- H/He nonequilibrium ion chemistry with temperature coupling.
- SN-lite core: gravity, heating/cooling, gamma override, composition passives, basic test cases.
- SN-lite upgrades: monopole gravity + gain-region heating/cooling profiles.
- Non-ideal MHD microphysics (Hall, ambipolar, hyper-resistive), radiation coupling, kinetic heating models.
- Diagnostics/analysis tools + smoke suites.

## Notes

- Keep commit messages referencing this roadmap file when updating progress.
- TODO: tighten diagnostics thresholds (divB, shock radius trends) once we set expected ranges.
- TODO: add GPU backends and stronger scaling optimizations.
- GPU backend (post-processing tools) now supports optional CuPy acceleration.
- TODO: improve RMHD primitive recovery robustness + diagnostics expansions (cocoon/mixing/shell fluxes).
