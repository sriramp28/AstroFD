# AstroFD Technical Reference

## Physics Overview

AstroFD advances conservative SRHD/RMHD systems in flat spacetime and GRHD/GRMHD on fixed backgrounds. For jets, the inlet boundary uses configurable nozzle profiles with optional perturbations and tracers. For SN-lite, Newtonian gravity, gain-region heating/cooling, and optional lightbulb neutrino source terms are provided.

## Governing Equations (Summary)

SRHD:
- Conservative variables `(D, S_i, tau)` from primitive `(rho, v_i, p)`.
- Fluxes computed with HLLE/HLLC and high-order reconstruction.

RMHD:
- Adds magnetic fields `(Bx, By, Bz)` and GLM scalar `psi`.
- Fluxes include magnetic pressure and tension.
- `divB` controlled by GLM with parameters `GLM_CH` and `GLM_CP`.

GRHD/GRMHD:
- Valencia formulation on fixed metrics (Minkowski, Schwarzschild, Kerr-Schild).
- Optional orthonormal-frame flux evaluation for improved stability.

SN-lite:
- Newtonian gravity source terms (point mass or monopole).
- Parametric gain heating/cooling and optional lightbulb neutrino transport.

## Numerics

Discretization:
- Finite-volume, conservative update with ghost zones.
- Reconstruction: MUSCL (MC limiter), PPM, WENO5.
- Limiters: MC, minmod, van Leer.

Riemann solvers:
- HLLE (robust baseline).
- HLLC (hydro).
- HLLD / full HLLD (RMHD).

Time integration:
- SSPRK2 and SSPRK3.
- IMEX subcycling for causal dissipation.

Robustness:
- Pressure/density floors.
- Velocity caps.
- Recovery fallback paths for RMHD primitives.

## Diagnostics

Built-in diagnostics include:
- Global max Lorentz factor, inlet energy flux.
- RMHD `divB` statistics.
- SN-lite shock radius and gain mass.
- Cocoon pressure and mixing layer thickness.

## Validation

Validation suites live in `tools/`:
- Scheme tests: limiter, recon, RK, HLLC/HLLD.
- GR Kerr-Schild runs and orthonormal flux verification.
- Restart regression.
- SN-lite freefall, Sedov, stalled shock, and neutrino lightbulb.

## LaTeX Writeup

See `docs/astrofd.tex` and `docs/astrofd_refs.bib` for a full description with references.
