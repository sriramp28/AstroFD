#!/usr/bin/env python3
import argparse
import subprocess
import sys


CASES = [
    ("hydro_hllc", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_hllc_small.json"]),
    ("hydro_minmod", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_minmod.json"]),
    ("hydro_vanleer", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_vanleer.json"]),
    ("hydro_ppm", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_ppm_small.json"]),
    ("hydro_weno", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_weno_small.json"]),
    ("hydro_rk3", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_rk3_smoke.json"]),
    ("rmhd_hlld_full", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_hlld_full.json"]),
    ("rmhd_ppm", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_rmhd_ppm_small.json"]),
    ("rmhd_weno", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_rmhd_weno_small.json"]),
]


def main():
    ap = argparse.ArgumentParser(description="Run validation smoke tests for schemes.")
    ap.add_argument("--skip-rmhd", action="store_true", help="skip RMHD HLLD test")
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    args = ap.parse_args()

    failed = []
    for name, cmd in CASES:
        if args.skip_rmhd and "rmhd" in name:
            continue
        print(f"[validate] {name}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            failed.append(name)

    if failed:
        print(f"[validate] failed: {failed}")
        raise SystemExit(1)
    print("[validate] all cases passed")


if __name__ == "__main__":
    main()
