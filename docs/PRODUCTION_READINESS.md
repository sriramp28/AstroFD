# AstroFD Production Readiness Checklist

This checklist summarizes remaining work to move from research code to production-grade.

## Validation and Verification

- Add standard SRHD/RMHD shock tube suite (1D/2D) with error norms.
- Add RMHD Orszag-Tang vortex and rotor tests.
- Add Alfvén and fast/slow magnetosonic wave convergence tests.
- Add GRMHD benchmark (e.g., magnetized Bondi or Fishbone-Moncrief torus).
- Establish quantitative thresholds for key diagnostics (divB, shock radius trends).

## Performance and Scaling

- Profile hotspots and memory usage for large grids.
- Improve strong/weak scaling on multi-node MPI.
- Optional threading in compute kernels (beyond Numba JIT).
- GPU solver backends (not just post-processing).

## Reproducibility and Robustness

- Fixed seeds and deterministic nozzle perturbations.
- Stronger restart coverage and versioned checkpoint metadata.
- Parameter validation and auto-generated config reports.
- Unit tests for EOS, reconstruction, and primitive recovery.

## Engineering

- CI pipeline (lint + unit + smoke + validation).
- Structured logging and metadata outputs.
- Release tagging and versioning.

## Documentation

- Full user guide and tutorial walkthroughs.
- Expanded LaTeX technical report with test results.
