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
