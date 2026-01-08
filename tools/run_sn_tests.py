#!/usr/bin/env python3
import argparse
import subprocess
import sys


CASES = [
    "config/config_sn_freefall.json",
    "config/config_sn_sedov.json",
    "config/config_sn_stalled_shock.json",
    "config/config_neutrino_lightbulb.json",
]


def main():
    ap = argparse.ArgumentParser(description="Run SN-lite test problems.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip diagnostics verification")
    ap.add_argument("--python", default="python")
    args = ap.parse_args()

    failed = []
    for cfg in CASES:
        cmd = [args.python, "solvers/srhd3d_mpi_muscl.py", "--config", cfg]
        print(f"[sn-test] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            failed.append(cfg)
            continue
        if not args.no_verify:
            vcmd = [args.python, "tools/verify_sn_diagnostics.py"]
            print(f"[sn-test] {' '.join(vcmd)}", flush=True)
            vproc = subprocess.run(vcmd, stdout=sys.stdout, stderr=sys.stderr)
            if vproc.returncode != 0:
                failed.append(cfg)

    if failed:
        print(f"[sn-test] failed: {failed}")
        raise SystemExit(1)
    print("[sn-test] all cases passed")


if __name__ == "__main__":
    main()
