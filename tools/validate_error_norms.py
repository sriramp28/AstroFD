#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np


CASES = [
    {
        "name": "hydro_riemann",
        "config": "config/config_hydro_riemann.json",
        "ref": "tests/references/hydro_riemann_ref.npz",
        "fields": ["rho", "p", "vx"],
        "mode": "centerline",
        "tol_l1": 5e-2,
        "tol_l2": 5e-2,
    },
    {
        "name": "rmhd_orszag_tang",
        "config": "config/config_rmhd_orszag_tang.json",
        "ref": "tests/references/rmhd_orszag_tang_ref.npz",
        "fields": ["rho", "Bmag"],
        "mode": "midplane",
        "tol_l1": 5e-2,
        "tol_l2": 5e-2,
    },
]


def _leaf_run_dirs(base="results"):
    runs = []
    if not os.path.exists(base):
        return runs
    for d in sorted(os.listdir(base)):
        path = os.path.join(base, d)
        if not os.path.isdir(path):
            continue
        subs = [os.path.join(path, s) for s in os.listdir(path) if os.path.isdir(os.path.join(path, s))]
        if subs:
            runs.extend(sorted(subs))
        else:
            runs.append(path)
    return runs


def _latest_run_dir(after_dirs, before_dirs):
    new_dirs = [d for d in after_dirs if d not in before_dirs]
    if new_dirs:
        new_dirs.sort(key=lambda p: os.path.getmtime(p))
        return new_dirs[-1]
    if after_dirs:
        after_dirs.sort(key=lambda p: os.path.getmtime(p))
        return after_dirs[-1]
    raise SystemExit("No run directories found after execution.")


def _latest_npz(run_dir):
    files = sorted([f for f in os.listdir(run_dir) if f.startswith("jet3d_rank0000_step") and f.endswith(".npz")])
    if not files:
        raise SystemExit(f"No NPZ files in {run_dir}")
    return os.path.join(run_dir, files[-1])


def _write_temp_config(cfg):
    cfg = dict(cfg)
    cfg["RESULTS_UNIQUE"] = True
    cfg["NOZZLE_TURB"] = False
    cfg["NOZZLE_PERTURB"] = "none"
    cfg["NOZZLE_ENABLED"] = False
    fd, path = tempfile.mkstemp(prefix="astrofd_cfg_", suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def _load_config(path):
    try:
        import json5
        with open(path, "r") as f:
            return json5.load(f)
    except Exception:
        with open(path, "r") as f:
            return json.load(f)


def _run_case(python, cfg_path):
    before = _leaf_run_dirs()
    cmd = [python, "solvers/srhd3d_mpi_muscl.py", "--config", cfg_path]
    print(f"[norms] run: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    time.sleep(0.1)
    after = _leaf_run_dirs()
    return _latest_run_dir(after, before)


def _extract_fields(npz_path, mode, fields):
    data = np.load(npz_path)
    NG = 2
    if mode == "centerline":
        rho = data["rho"]; p = data["p"]; vx = data["vx"]
        ny = rho.shape[1]; nz = rho.shape[2]
        j = ny // 2
        k = nz // 2
        sl = slice(NG, -NG)
        out = {}
        if "rho" in fields:
            out["rho"] = rho[sl, j, k].copy()
        if "p" in fields:
            out["p"] = p[sl, j, k].copy()
        if "vx" in fields:
            out["vx"] = vx[sl, j, k].copy()
        return out
    if mode == "midplane":
        rho = data["rho"]; Bx = data["Bx"]; By = data["By"]; Bz = data["Bz"]
        k = rho.shape[2] // 2
        slx = slice(NG, -NG)
        sly = slice(NG, -NG)
        out = {}
        if "rho" in fields:
            out["rho"] = rho[slx, sly, k].copy()
        if "Bmag" in fields:
            Bmag = np.sqrt(Bx[slx, sly, k]**2 + By[slx, sly, k]**2 + Bz[slx, sly, k]**2)
            out["Bmag"] = Bmag
        return out
    raise ValueError(f"Unknown mode {mode}")


def _norms(a, ref):
    diff = a - ref
    l1 = float(np.sum(np.abs(diff)) / max(np.sum(np.abs(ref)), 1e-12))
    l2 = float(np.sqrt(np.sum(diff * diff) / max(np.sum(ref * ref), 1e-12)))
    return l1, l2


def _load_reference(ref_path):
    if not os.path.exists(ref_path):
        raise SystemExit(f"Missing reference: {ref_path} (run with --update)")
    return np.load(ref_path)


def _write_reference(ref_path, arrays):
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    np.savez(ref_path, **arrays)
    print(f"[norms] wrote {ref_path}")


def main():
    ap = argparse.ArgumentParser(description="Validate error norms for shock tubes and Orszag–Tang.")
    ap.add_argument("--python", default="python")
    ap.add_argument("--update", action="store_true", help="write reference snapshots")
    args = ap.parse_args()

    errors = []
    for case in CASES:
        cfg = _load_config(case["config"])
        tmp_cfg = _write_temp_config(cfg)
        try:
            run_dir = _run_case(args.python, tmp_cfg)
        finally:
            os.remove(tmp_cfg)

        npz_path = _latest_npz(run_dir)
        arrays = _extract_fields(npz_path, case["mode"], case["fields"])
        if args.update:
            _write_reference(case["ref"], arrays)
            continue

        ref = _load_reference(case["ref"])
        for key in case["fields"]:
            if key not in ref:
                errors.append(f"{case['name']}: missing ref {key}")
                continue
            l1, l2 = _norms(arrays[key], ref[key])
            print(f"[norms] {case['name']} {key} L1={l1:.3e} L2={l2:.3e}")
            if l1 > case["tol_l1"]:
                errors.append(f"{case['name']} {key} L1 {l1:.3e} > {case['tol_l1']:.3e}")
            if l2 > case["tol_l2"]:
                errors.append(f"{case['name']} {key} L2 {l2:.3e} > {case['tol_l2']:.3e}")

    if errors:
        for err in errors:
            print(f"[norms] {err}")
        raise SystemExit(1)
    if not args.update:
        print("[norms] all error-norm checks passed")


if __name__ == "__main__":
    main()
