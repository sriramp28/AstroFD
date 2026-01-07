# AstroFD

## Quick smoke tests

Two tiny configs are included for fast checks on an 8^3 grid:

- `config/config_two_temp.json`: SRHD with two-temperature relaxation enabled.
- `config/config_resist.json`: RMHD with resistive diffusion enabled.

Example runs:

```bash
python solvers/srhd3d_mpi_muscl.py --config config/config_two_temp.json --nx 8 --ny 8 --nz 8 --t-end 0.001 --out-every 1 --print-every 1
python solvers/srhd3d_mpi_muscl.py --config config/config_resist.json --nx 8 --ny 8 --nz 8 --t-end 0.001 --out-every 1 --print-every 1
```

## Phase 5 config keys

Extended thermodynamics and resistive RMHD are controlled through the following keys:

- `TWO_TEMPERATURE` (bool): enable Te/Ti passive fields and relaxation to gas temperature.
- `TEI_TAU` (float): relaxation time for Te/Ti toward `T = p/rho`.
- `TE_AMB`, `TI_AMB` (float): ambient electron/ion temperature.
- `TE_NOZZLE`, `TI_NOZZLE` (float): inlet electron/ion temperature.
- `COOLING_ENABLED` (bool): enable simple cooling/heating source on pressure.
- `COOLING_LAMBDA` (float): cooling rate coefficient (applied to pressure).
- `HEATING_RATE` (float): constant heating rate (applied to pressure).
- `RESISTIVE_ENABLED` (bool): enable Ohmic diffusion of B in RMHD.
- `RESISTIVITY` (float): resistivity coefficient for Laplacian diffusion of B.
