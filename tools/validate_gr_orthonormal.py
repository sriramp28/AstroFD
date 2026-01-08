#!/usr/bin/env python3
import argparse
import glob
import os
import subprocess
import sys


CASES = [
    ("grhd_ortho", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_grhd_ks.json"]),
    ("grhd_coord", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_grhd_ks_coord.json"]),
    ("grmhd_ortho", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_grmhd.json"]),
    ("grmhd_coord", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_grmhd_ks_coord.json"]),
]


def main():
    ap = argparse.ArgumentParser(description="Validate GR orthonormal flux toggle.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default="python")
    args = ap.parse_args()

    failed = []
    for name, cmd in CASES:
        cmd = [args.python if c == "python" else c for c in cmd]
        print(f"[gr-ortho] {name}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            failed.append(name)
            continue
        run_dir = latest_run_dir()
        if "grmhd" in name:
            vcmd = [args.python, "tools/verify_rmhd.py", "--max-divb-rel", "200.0", "--max-psi", "1e-2"]
        else:
            vcmd = [args.python, "tools/verify_run.py"]
        if run_dir:
            vcmd.extend(["--run-dir", run_dir])
        print(f"[gr-ortho] verify: {' '.join(vcmd)}", flush=True)
        vproc = subprocess.run(vcmd, stdout=sys.stdout, stderr=sys.stderr)
        if vproc.returncode != 0:
            failed.append(name)

    if failed:
        print(f"[gr-ortho] failed: {failed}")
        raise SystemExit(1)
    print("[gr-ortho] all cases passed")


def latest_run_dir(base="results"):
    runs = sorted([d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)])
    if not runs:
        return None
    last = runs[-1]
    subs = sorted([d for d in glob.glob(os.path.join(last, "*")) if os.path.isdir(d)])
    return subs[-1] if subs else last


if __name__ == "__main__":
    main()
