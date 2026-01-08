# AstroFD

Relativistic (SR/GR) hydro and magnetohydrodynamics with jet and SN-lite physics.

## Quickstart

```bash
scripts/setup_env.sh
source scripts/env.sh
```

Run a tiny SRHD test:

```bash
python solvers/srhd3d_mpi_muscl.py --config config/config_two_temp.json
```

Run the full validation suite (small tests):

```bash
python tools/run_validation_suite.py --quick
```

## Documentation

- `docs/USER_GUIDE.md`
- `docs/TECHNICAL_REFERENCE.md`
- `docs/PRODUCTION_READINESS.md`
- `docs/astrofd.tex` (LaTeX writeup)

## Notes

- MPI warnings about socket binding may appear in sandboxed environments; runs still complete.
- For CuPy-accelerated post-processing, install CuPy and pass `--backend cupy` to tools.
