#!/usr/bin/env python3
import argparse
import subprocess
import sys


CASES = [
    "config/config_gamma_table.json",
    "config/config_resist_eos.json",
    "config/config_grmhd_eos_smoke.json",
]


def main():
    ap = argparse.ArgumentParser(description="Run EOS smoke tests.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-gr", action="store_true", help="skip GRMHD EOS case")
    ap.add_argument("--python", default="python")
    args = ap.parse_args()

    failed = []
    for cfg in CASES:
        if args.skip_gr and "grmhd" in cfg:
            continue
        cmd = [args.python, "solvers/srhd3d_mpi_muscl.py", "--config", cfg]
        print(f"[eos-smoke] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            failed.append(cfg)

    if failed:
        print(f"[eos-smoke] failed: {failed}")
        raise SystemExit(1)
    print("[eos-smoke] all cases passed")


if __name__ == "__main__":
    main()
