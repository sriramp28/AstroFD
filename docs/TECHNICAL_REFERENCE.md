# AstroFD Technical Reference

## Scope and Notation

AstroFD solves relativistic fluid dynamics in Cartesian coordinates with
finite-volume methods. Units use c = 1. Greek indices denote spacetime
components and Latin indices denote spatial components. The 3-velocity is
v_i, Lorentz factor is W = (1 - v^2)^(-1/2), specific enthalpy is
h = 1 + e + p / rho, and the ideal-gas EOS is p = (gamma - 1) * rho * e.

## Governing Equations

### SRHD

Conserved variables:
- D = rho * W
- S_i = rho * h * W^2 * v_i
- tau = rho * h * W^2 - p - D

The SRHD system is advanced with conservative fluxes in each direction
and a source-free evolution in flat spacetime.

### RMHD

RMHD extends SRHD with magnetic fields B_i and a GLM scalar psi. The
magnetic field contributes to momentum and energy via magnetic pressure
and tension. GLM evolves psi with hyperbolic wave speed ch and damping
cp to control divB.

### GRHD/GRMHD

GRHD/GRMHD use the Valencia formulation on fixed backgrounds. The code
supports Minkowski, Schwarzschild, and Kerr-Schild metrics. Optional
orthonormal-frame flux evaluation improves stability in strong curvature.

### Causal Dissipation

Israel-Stewart causal dissipation evolves bulk, shear, and heat flux
variables with relaxation times. Stiff source terms are integrated with
subcycled IMEX updates.

### Two-Temperature Closure

Separate electron and ion internal energies evolve with relaxation time
TEI_TAU. The model is optional and can be enabled with TWO_TEMPERATURE.

### H/He Non-Equilibrium Chemistry

A simplified H/He ionization network evolves species fractions with
ionization and recombination source terms. Cooling and heating are
applied consistently with the ion state.

### Resistive and Non-Ideal MHD

Resistive RMHD adds Ohmic diffusion with resistivity eta. Optional
Hall, ambipolar, and hyper-resistive terms are included as explicit
non-ideal corrections.

### SN-lite Source Terms

SN-lite physics provides parametric gravity (point mass or monopole),
gain-region heating/cooling, and a lightbulb neutrino source term for
approximate energy deposition.

## Numerical Methods

### Finite-Volume Discretization

Cell-averaged conserved variables are updated using reconstructed face
states and a Riemann solver. Source terms are applied in split fashion.

### Reconstruction

- MUSCL with MC/minmod/van Leer limiters.
- PPM for sharper shocks.
- WENO5 for high-order smooth-region accuracy.

### Riemann Solvers

- HLLE: robust baseline.
- HLLC: hydro contact-resolving solver.
- HLLD and full HLLD: RMHD solvers that resolve additional wave families.

### Time Integration

- SSPRK2 and SSPRK3 integrators.
- IMEX subcycling for stiff dissipation or chemistry.

### Primitive Recovery

RMHD primitive recovery uses iterative inversion with fallbacks. If
recovery fails or produces unphysical states, floors and caps are applied.

## Boundary Conditions and Injection

Jet inflow is applied at the x-min boundary with configurable profiles:
- top-hat + shear layer
- taper
- parabolic

Optional perturbations seed instabilities; magnetic fields may be poloidal
or toroidal at the inlet. Outflow boundaries are used elsewhere.

## Diagnostics

Diagnostics include:
- maximum Lorentz factor and inlet fluxes
- divB statistics for RMHD/GRMHD
- cocoon pressure and mixing layer thickness
- SN shock radius, gain mass, heating efficiency
- optional performance counters

## Validation and Regression

Validation suites in `tools/` include:
- reconstruction, limiter, RK, and Riemann checks
- GR Kerr-Schild and orthonormal flux verification
- restart regression
- SN-lite freefall, Sedov, stalled shock, and lightbulb tests

## Reference Materials

The LaTeX writeup (`docs/astrofd.tex`) contains a full description and
citations for the methods. Bibliography entries are in
`docs/astrofd_refs.bib`.
