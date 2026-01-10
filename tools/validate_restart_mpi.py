#!/usr/bin/env python3
import argparse
import glob
import json
import os
import subprocess
import sys


def latest_run_dir(base="results"):
    runs = sorted([d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)])
    if not runs:
        return None
    last = runs[-1]
    subs = sorted([d for d in glob.glob(os.path.join(last, "*")) if os.path.isdir(d)])
    return subs[-1] if subs else last


def has_checkpoint_step(run_dir, step_min, nranks):
    if run_dir is None:
        return False
    for r in range(nranks):
        pats = glob.glob(os.path.join(run_dir, f"checkpoint_rank{r:04d}_step*.npz"))
        ok = False
        for path in pats:
            base = os.path.basename(path)
            if "step" not in base:
                continue
            try:
                step = int(base.split("step")[1].split(".")[0])
            except Exception:
                continue
            if step >= step_min:
                ok = True
                break
        if not ok:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Validate MPI checkpoint/restart flow.")
    ap.add_argument("--python", default="python")
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--np", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_cfg = "config/config_restart_smoke.json"
    with open(base_cfg, "r") as f:
        cfg = json.load(f)

    tmp_run = "config/.restart_mpi_run.json"
    cfg_run = dict(cfg)
    cfg_run["RESTART_PATH"] = None
    cfg_run["CHECKPOINT_EVERY"] = 1
    with open(tmp_run, "w") as f:
        json.dump(cfg_run, f, indent=2)

    cmd1 = [
        args.mpiexec, "-n", str(args.np),
        args.python, "solvers/srhd3d_mpi_muscl.py",
        "--config", tmp_run,
        "--t-end", "0.0001",
        "--out-every", "1",
        "--print-every", "1",
    ]
    cmd3 = [args.python, "tools/verify_run.py"]

    for label, cmd in [("run", cmd1)]:
        print(f"[restart-mpi] {label}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    run_dir = latest_run_dir()
    if run_dir is None:
        raise SystemExit("[restart-mpi] no run directory after first run")

    tmp_restart = "config/.restart_mpi_resume.json"
    cfg_restart = dict(cfg)
    cfg_restart["RESTART_PATH"] = run_dir
    cfg_restart["CHECKPOINT_EVERY"] = 1
    with open(tmp_restart, "w") as f:
        json.dump(cfg_restart, f, indent=2)

    cmd2 = [
        args.mpiexec, "-n", str(args.np),
        args.python, "solvers/srhd3d_mpi_muscl.py",
        "--config", tmp_restart,
        "--t-end", "0.0002",
        "--out-every", "1",
        "--print-every", "1",
    ]
    for label, cmd in [("restart", cmd2)]:
        print(f"[restart-mpi] {label}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    run_dir = latest_run_dir()
    if run_dir:
        cmd3.extend(["--run-dir", run_dir])
    for label, cmd in [("verify", cmd3)]:
        print(f"[restart-mpi] {label}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    if not has_checkpoint_step(run_dir, 2, args.np):
        raise SystemExit("[restart-mpi] missing checkpoint at step >= 2 on all ranks")

    for path in (tmp_run, tmp_restart):
        try:
            os.remove(path)
        except Exception:
            pass

    print("[restart-mpi] ok")


if __name__ == "__main__":
    main()
