#!/usr/bin/env python3
import argparse
import glob
import os
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

    run_dir = latest_run_dir()
    cmd = [args.python, "tools/verify_rmhd.py", "--max-divb-rel", "100.0", "--max-psi", "1e-2"]
    if run_dir:
        cmd.extend(["--run-dir", run_dir])
    print(f"[hlld] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    print("[hlld] ok")


def latest_run_dir(base="results"):
    runs = sorted([d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)])
    if not runs:
        return None
    last = runs[-1]
    subs = sorted([d for d in glob.glob(os.path.join(last, "*")) if os.path.isdir(d)])
    return subs[-1] if subs else last


if __name__ == "__main__":
    main()
