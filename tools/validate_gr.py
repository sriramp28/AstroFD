#!/usr/bin/env python3
import argparse
import subprocess
import sys


CASES = [
    ("grhd_ks", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_grhd_ks.json"]),
    ("grmhd_ks", ["python", "solvers/srhd3d_mpi_muscl.py", "--config", "config/config_grmhd.json"]),
]


def main():
    ap = argparse.ArgumentParser(description="Validate GRHD/GRMHD Kerr-Schild runs.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default="python")
    args = ap.parse_args()

    failed = []
    for name, cmd in CASES:
        cmd = [args.python if c == "python" else c for c in cmd]
        print(f"[gr-validate] {name}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            failed.append(name)
            continue
        if name == "grmhd_ks":
            vcmd = [args.python, "tools/verify_rmhd.py", "--max-divb-rel", "200.0", "--max-psi", "1e-2"]
            print(f"[gr-validate] verify: {' '.join(vcmd)}", flush=True)
            vproc = subprocess.run(vcmd, stdout=sys.stdout, stderr=sys.stderr)
            if vproc.returncode != 0:
                failed.append(name)
        else:
            vcmd = [args.python, "tools/verify_run.py"]
            print(f"[gr-validate] verify: {' '.join(vcmd)}", flush=True)
            vproc = subprocess.run(vcmd, stdout=sys.stdout, stderr=sys.stderr)
            if vproc.returncode != 0:
                failed.append(name)

    if failed:
        print(f"[gr-validate] failed: {failed}")
        raise SystemExit(1)
    print("[gr-validate] all cases passed")


if __name__ == "__main__":
    main()
