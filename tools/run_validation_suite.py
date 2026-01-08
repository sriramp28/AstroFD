#!/usr/bin/env python3
import argparse
import subprocess
import sys


STEPS = [
    ("smoke", ["python", "tools/run_smoke_suite.py"]),
    ("schemes", ["python", "tools/validate_schemes.py"]),
    ("hlld", ["python", "tools/validate_hlld.py"]),
    ("gr", ["python", "tools/validate_gr.py"]),
    ("sn", ["python", "tools/run_sn_tests.py"]),
    ("eos", ["python", "tools/run_eos_smoke.py"]),
    ("rmhd_recovery", ["python", "tools/stress_rmhd_recovery.py", "--n", "200"]),
]


def main():
    ap = argparse.ArgumentParser(description="Run expanded validation suite.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quick", action="store_true", help="run a reduced, faster subset")
    ap.add_argument("--skip-rmhd", action="store_true", help="skip RMHD/GRMHD validations")
    ap.add_argument("--skip-gr", action="store_true", help="skip GRMHD validations")
    ap.add_argument("--skip-sn", action="store_true", help="skip SN-lite validations")
    ap.add_argument("--skip-eos", action="store_true", help="skip EOS smoke tests")
    ap.add_argument("--python", default="python")
    args = ap.parse_args()

    failed = []
    steps = STEPS
    if args.quick:
        steps = [
            ("smoke", ["python", "tools/run_smoke_suite.py"]),
            ("sn", ["python", "tools/run_sn_tests.py"]),
            ("eos", ["python", "tools/run_eos_smoke.py"]),
            ("rmhd_recovery", ["python", "tools/stress_rmhd_recovery.py", "--n", "50"]),
        ]

    for name, cmd in steps:
        if name == "smoke":
            cmd = [args.python, "tools/run_smoke_suite.py"]
            if args.skip_rmhd:
                cmd.append("--skip-rmhd")
            if args.skip_gr:
                cmd.append("--skip-gr")
            if args.quick:
                cmd.extend(["--skip-rmhd", "--skip-gr"])
        if name == "schemes" and args.skip_rmhd:
            cmd = [args.python, "tools/validate_schemes.py", "--skip-rmhd"]
        if name == "gr" and args.quick:
            continue
        if name in ("hlld", "rmhd_recovery") and args.skip_rmhd:
            continue
        if name == "eos":
            cmd = [args.python, "tools/run_eos_smoke.py"]
            if args.skip_gr:
                cmd.append("--skip-gr")
            if args.quick:
                cmd.append("--skip-gr")
        if name == "gr" and args.skip_gr:
            continue
        if name == "sn" and args.quick:
            cmd = [args.python, "tools/run_sn_tests.py", "--no-verify"]
        if args.skip_sn and name == "sn":
            continue
        if args.skip_eos and name == "eos":
            continue

        cmd = [args.python if c == "python" else c for c in cmd]
        print(f"[validate-all] {name}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            failed.append(name)

    if failed:
        print(f"[validate-all] failed: {failed}")
        raise SystemExit(1)
    print("[validate-all] all checks passed")


if __name__ == "__main__":
    main()
