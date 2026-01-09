#!/usr/bin/env python3
import argparse
import subprocess
import sys


CASES = [
    "config/config_two_temp.json",
    "config/config_resist.json",
    "config/config_dissipation.json",
    "config/config_grmhd.json",
    "config/config_hllc_small.json",
    "config/config_minmod.json",
    "config/config_vanleer.json",
    "config/config_hlld_full.json",
    "config/config_chemistry.json",
    "config/config_chemistry_hot.json",
    "config/config_sn_lite.json",
    "config/config_adaptivity_smoke.json",
    "config/config_adaptivity_dynamic.json",
]


def main():
    ap = argparse.ArgumentParser(description="Run all tiny smoke-test configs.")
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    ap.add_argument("--skip-rmhd", action="store_true", help="skip RMHD/GRMHD cases")
    ap.add_argument("--skip-gr", action="store_true", help="skip GRMHD case")
    ap.add_argument("--python", default="python", help="python executable")
    args = ap.parse_args()

    failed = []
    for cfg in CASES:
        if args.skip_rmhd and ("rmhd" in cfg or "hlld" in cfg or "resist" in cfg):
            continue
        if args.skip_gr and "grmhd" in cfg:
            continue
        cmd = [args.python, "solvers/srhd3d_mpi_muscl.py", "--config", cfg]
        print(f"[smoke] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            failed.append(cfg)

    if failed:
        print(f"[smoke] failed: {failed}")
        raise SystemExit(1)
    print("[smoke] all cases passed")


if __name__ == "__main__":
    main()
