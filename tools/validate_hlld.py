#!/usr/bin/env python3
import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Run HLLD regression smoke test.")
    ap.add_argument("--skip-run", action="store_true", help="skip solver run, only verify latest RMHD output")
    ap.add_argument("--python", default="python")
    args = ap.parse_args()

    if not args.skip_run:
        cmd = [
            args.python,
            "solvers/srhd3d_mpi_muscl.py",
            "--config",
            "config/config_hlld_full.json",
            "--nx",
            "8",
            "--ny",
            "8",
            "--nz",
            "8",
            "--t-end",
            "0.001",
            "--out-every",
            "1",
            "--print-every",
            "1",
        ]
        print(f"[hlld] {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    cmd = [args.python, "tools/verify_rmhd.py", "--max-divb-rel", "200.0", "--max-psi", "1e-2"]
    print(f"[hlld] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    print("[hlld] ok")


if __name__ == "__main__":
    main()
