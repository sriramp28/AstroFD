#!/usr/bin/env python3
import argparse
import os
import sys


def latest_run_dir(base="results"):
    if not os.path.isdir(base):
        return None
    dirs = [os.path.join(base, d) for d in os.listdir(base)]
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        return None
    return max(dirs, key=lambda d: os.path.getmtime(d))


def parse_last_row(path):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 2:
        return None
    parts = lines[-1].split(",")
    try:
        values = [float(p) for p in parts[1:]]
    except Exception:
        return None
    return values


def main():
    ap = argparse.ArgumentParser(description="Verify SN diagnostics outputs.")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    if run_dir is None:
        print("[sn-verify] no run directory found")
        return 1

    diag_path = os.path.join(run_dir, "sn_diagnostics.csv")
    if not os.path.exists(diag_path):
        print(f"[sn-verify] missing {diag_path}")
        return 1

    values = parse_last_row(diag_path)
    if values is None or len(values) < 6:
        print(f"[sn-verify] unable to parse last row of {diag_path}")
        return 1

    shock_r = values[1]
    gain_mass = values[2]
    heat_power = values[3]
    heat_abs = values[4]
    eff = values[5]

    if shock_r < 0.0 or gain_mass < 0.0 or heat_abs < 0.0:
        print(f"[sn-verify] invalid values in {diag_path}")
        return 1
    if abs(eff) > 10.0:
        print(f"[sn-verify] suspicious heating efficiency: {eff}")
        return 1

    print(f"[sn-verify] ok: {diag_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
