#!/usr/bin/env python3
import argparse
import glob
import os
import subprocess
import sys

import numpy as np


def latest_run_dir(base="results"):
    runs = sorted([d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)])
    if not runs:
        return None
    last = runs[-1]
    subs = sorted([d for d in glob.glob(os.path.join(last, "*")) if os.path.isdir(d)])
    return subs[-1] if subs else last


def magnetic_energy(path):
    data = np.load(path)
    Bx = data["Bx"]; By = data["By"]; Bz = data["Bz"]
    B2 = Bx*Bx + By*By + Bz*Bz
    return float(np.mean(B2))


def check_case(python, cfg, min_drop=None, max_growth=None):
    cmd = [python, "solvers/srhd3d_mpi_muscl.py", "--config", cfg]
    print(f"[nonideal] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    run_dir = latest_run_dir()
    if not run_dir:
        raise SystemExit("[nonideal] no results dir found")
    files = sorted(glob.glob(os.path.join(run_dir, "jet3d_rank0000_step*.npz")))
    if len(files) < 2:
        raise SystemExit(f"[nonideal] not enough outputs in {run_dir}")
    e0 = magnetic_energy(files[0])
    e1 = magnetic_energy(files[-1])
    if not np.isfinite([e0, e1]).all():
        raise SystemExit("[nonideal] non-finite magnetic energy")

    if min_drop is not None:
        if e1 > e0 * (1.0 - min_drop):
            raise SystemExit(f"[nonideal] magnetic energy did not decay: {e0:.3e} -> {e1:.3e}")
    if max_growth is not None:
        if e1 > e0 * max_growth:
            raise SystemExit(f"[nonideal] magnetic energy grew too much: {e0:.3e} -> {e1:.3e}")

    print(f"[nonideal] ok: {cfg} B2 {e0:.3e} -> {e1:.3e}")


def main():
    ap = argparse.ArgumentParser(description="Validate non-ideal RMHD terms.")
    ap.add_argument("--python", default="python")
    args = ap.parse_args()

    check_case(args.python, "config/config_nonideal_resist.json", min_drop=0.01)
    check_case(args.python, "config/config_nonideal_hall.json", max_growth=5.0)


if __name__ == "__main__":
    main()
