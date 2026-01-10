# AstroFD Roadmap Progress

This file tracks the current plan and completion status so work can resume easily.

## Current focus (sn-lite-plasma branch)

### Priority work (in progress)
1. RMHD solver fidelity
   - Full HLLD implementation (done; RMHD HLLD with star states + fallbacks)
   - HLLD benchmark validation suite (done; Riemann init + verify script)
   - Primitive recovery stress testing + fallback verification in CI (done)
2. Diagnostics/analysis
   - Spectra/structure functions, flux budgets, entrainment/mixing reports (done)
3. Non-ideal MHD validation
   - Resistive + Hall/ambipolar/hyper-resistive test cases (done)
4. SN physics fidelity
   - EOS tables and neutrino source term upgrades (done)
5. AMR
   - Real AMR (error indicators, multi-rank) beyond static nested refinement (in progress; MPI global dynamic nested refinement now supported)

## Completed highlights

- SRHD/RMHD/GRHD/GRMHD core with recon/riemann switches, GLM cleaning.
- Israel–Stewart dissipation (IMEX relaxation + advection).
- Two-temperature, cooling/heating, resistive RMHD hooks, passive tracers.
- H/He nonequilibrium ion chemistry with temperature coupling.
- SN-lite core: gravity, heating/cooling, gamma override, composition passives, basic test cases.
- SN-lite upgrades: monopole gravity + gain-region heating/cooling profiles.
- Non-ideal MHD microphysics hooks (Hall, ambipolar, hyper-resistive), radiation coupling, kinetic heating models.
- Diagnostics/analysis tools + smoke suites.

## Notes

- HLLD full solver path exists with validation; additional regression cases are planned.
- Static nested refinement exists (single-rank); MPI supports global dynamic nested boxes, but full patch-based AMR is not implemented yet.
- Diagnostics thresholds tightened (divB rel default 100, SN eff default 5).
- GPU backend (post-processing tools) now supports optional CuPy acceleration.
- Baseline regression checks added for small hydro/RMHD cases.
- Baseline regression checks expanded to SN-lite and GR cases; MPI restart validation added.
- New validation configs added for hydro/RMHD Riemann, Orszag–Tang, and RMHD rotor tests.
