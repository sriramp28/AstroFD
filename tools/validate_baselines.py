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
    ("hydro_hllc_small", "config/config_hllc_small.json", {}),
    ("rmhd_hlld_full", "config/config_hlld_full.json", {}),
    ("grhd_small", "config/config_grhd.json", {"NX": 8, "NY": 8, "NZ": 8, "T_END": 0.001, "OUT_EVERY": 1, "PRINT_EVERY": 1}),
    ("grmhd_small", "config/config_grmhd.json", {"NX": 8, "NY": 8, "NZ": 8, "T_END": 0.001, "OUT_EVERY": 1, "PRINT_EVERY": 1}),
    ("sn_lite_small", "config/config_sn_lite.json", {"NX": 16, "NY": 16, "NZ": 16, "T_END": 0.001, "OUT_EVERY": 1, "PRINT_EVERY": 1}),
]


DEFAULT_TOLS = {
    "max_gamma": {"rel": 5e-2, "abs": 1e-3},
    "max_v": {"rel": 5e-2, "abs": 1e-3},
    "mean_rho": {"rel": 5e-2, "abs": 1e-4},
    "mean_p": {"rel": 5e-2, "abs": 1e-5},
    "inlet_flux_abs": {"rel": 1e-1, "abs": 1e-6},
    "mean_b": {"rel": 1e-1, "abs": 1e-6},
    "max_b": {"rel": 1e-1, "abs": 1e-6},
    "max_psi": {"rel": 2e-1, "abs": 1e-5},
    "rel_divb": {"rel": 2e-1, "abs": 1e-6},
    "shock_radius": {"rel": 1e-1, "abs": 1e-3},
    "heat_eff": {"rel": 2e-1, "abs": 1e-4},
}


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


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _latest_npz(run_dir):
    files = sorted([f for f in os.listdir(run_dir) if f.startswith("jet3d_rank0000_step") and f.endswith(".npz")])
    if not files:
        raise SystemExit(f"No NPZ files in {run_dir}")
    return os.path.join(run_dir, files[-1])


def _compute_metrics_hydro(npz_path):
    data = np.load(npz_path)
    rho = data["rho"]; vx = data["vx"]; vy = data["vy"]; vz = data["vz"]; p = data["p"]
    NG = 2
    sl = (slice(NG, -NG), slice(NG, -NG), slice(NG, -NG))
    v2 = vx[sl]**2 + vy[sl]**2 + vz[sl]**2
    v2 = np.clip(v2, 0, 1 - 1e-14)
    G = 1.0 / np.sqrt(1.0 - v2)
    max_v = float(np.max(np.sqrt(v2)))
    max_g = float(np.max(G))
    mean_rho = float(np.mean(rho[sl]))
    mean_p = float(np.mean(p[sl]))

    try:
        dx, dy, dz, _t = data["meta"]
    except Exception:
        ny = rho.shape[1] - 2*NG
        nz = rho.shape[2] - 2*NG
        dy = 1.0 / max(ny, 1)
        dz = 1.0 / max(nz, 1)
    i = NG
    gamma = 5.0/3.0
    h = 1.0 + gamma/(gamma-1.0) * (p[i,:,:] / np.maximum(rho[i,:,:], 1e-12))
    w = rho[i,:,:] * h
    W2 = 1.0 / (1.0 - np.clip(vx[i,:,:]**2 + vy[i,:,:]**2 + vz[i,:,:]**2, 0, 1 - 1e-14))
    Sx = w * W2 * vx[i,:,:]
    inlet_flux = float(np.sum(Sx[NG:-NG, NG:-NG]) * dy * dz)

    return {
        "max_gamma": max_g,
        "max_v": max_v,
        "mean_rho": mean_rho,
        "mean_p": mean_p,
        "inlet_flux_abs": abs(inlet_flux),
    }


def _compute_metrics_rmhd(npz_path):
    data = np.load(npz_path)
    rho = data["rho"]; vx = data["vx"]; vy = data["vy"]; vz = data["vz"]; p = data["p"]
    Bx = data["Bx"]; By = data["By"]; Bz = data["Bz"]; psi = data["psi"]
    NG = 2
    sl = (slice(NG, -NG), slice(NG, -NG), slice(NG, -NG))
    v2 = vx[sl]**2 + vy[sl]**2 + vz[sl]**2
    v2 = np.clip(v2, 0, 1 - 1e-14)
    G = 1.0 / np.sqrt(1.0 - v2)
    B2 = Bx[sl]**2 + By[sl]**2 + Bz[sl]**2
    max_v = float(np.max(np.sqrt(v2)))
    max_g = float(np.max(G))
    mean_rho = float(np.mean(rho[sl]))
    mean_p = float(np.mean(p[sl]))
    mean_b = float(np.mean(np.sqrt(B2)))
    max_b = float(np.max(np.sqrt(B2)))
    max_psi = float(np.max(np.abs(psi[sl])))

    divb = (
        (Bx[NG+1:-NG+1, NG:-NG, NG:-NG] - Bx[NG-1:-NG-1, NG:-NG, NG:-NG]) / 2.0 +
        (By[NG:-NG, NG+1:-NG+1, NG:-NG] - By[NG:-NG, NG-1:-NG-1, NG:-NG]) / 2.0 +
        (Bz[NG:-NG, NG:-NG, NG+1:-NG+1] - Bz[NG:-NG, NG:-NG, NG-1:-NG-1]) / 2.0
    )
    divb_max = float(np.max(np.abs(divb)))
    try:
        dx, dy, dz, _t = data["meta"]
    except Exception:
        nx = rho.shape[0] - 2*NG
        dx = 1.0 / max(nx, 1)
    bnorm = np.sqrt(B2)
    denom = np.maximum(bnorm / max(dx, 1e-12), 1e-12)
    rel_divb = float(divb_max / np.max(denom))

    return {
        "max_gamma": max_g,
        "max_v": max_v,
        "mean_rho": mean_rho,
        "mean_p": mean_p,
        "mean_b": mean_b,
        "max_b": max_b,
        "max_psi": max_psi,
        "rel_divb": rel_divb,
    }


def _load_config(path):
    try:
        import json5
        with open(path, "r") as f:
            return json5.load(f)
    except Exception:
        with open(path, "r") as f:
            return json.load(f)


def _write_temp_config(cfg):
    cfg = dict(cfg)
    cfg["RESULTS_UNIQUE"] = True
    if "NOZZLE_TURB" in cfg:
        cfg["NOZZLE_TURB"] = False
    cfg["NOZZLE_PERTURB"] = "none"
    fd, path = tempfile.mkstemp(prefix="astrofd_cfg_", suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def _write_temp_config_with_overrides(cfg, overrides):
    cfg_tmp = dict(cfg)
    if overrides:
        cfg_tmp.update(overrides)
    return _write_temp_config(cfg_tmp)


def _latest_sn_diag(run_dir):
    path = os.path.join(run_dir, "sn_diagnostics.csv")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) < 2:
        return None
    last = lines[-1].split(",")
    if len(last) < 7:
        return None
    return {
        "shock_radius": float(last[2]),
        "heat_eff": float(last[6]),
    }


def _run_case(python, cfg_path):
    before = _leaf_run_dirs()
    cmd = [python, "solvers/srhd3d_mpi_muscl.py", "--config", cfg_path]
    print(f"[baseline] run: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    time.sleep(0.1)
    after = _leaf_run_dirs()
    return _latest_run_dir(after, before)


def _get_metrics(run_dir, physics):
    npz = _latest_npz(run_dir)
    if physics == "sn":
        sn = _latest_sn_diag(run_dir)
        if sn is None:
            raise SystemExit(f"missing sn_diagnostics.csv in {run_dir}")
        return sn
    if physics in ("rmhd", "grmhd"):
        return _compute_metrics_rmhd(npz)
    return _compute_metrics_hydro(npz)


def _compare_metrics(name, metrics, baseline, tolerances):
    errors = []
    for key, ref in baseline.items():
        if key not in metrics:
            errors.append(f"{name}: missing metric {key}")
            continue
        val = metrics[key]
        tol = tolerances.get(key, DEFAULT_TOLS.get(key, {"rel": 0.1, "abs": 1e-6}))
        rel = tol.get("rel", 0.1)
        abs_tol = tol.get("abs", 1e-6)
        diff = abs(val - ref)
        allowed = abs_tol + rel * max(abs(ref), 1e-12)
        if diff > allowed:
            errors.append(f"{name}: {key} {val:.6e} not within {allowed:.3e} of {ref:.6e}")
    return errors


def main():
    ap = argparse.ArgumentParser(description="Run regression baselines and compare metrics.")
    ap.add_argument("--baseline", default="tests/baselines.json", help="baseline JSON path")
    ap.add_argument("--update", action="store_true", help="write new baselines from current runs")
    ap.add_argument("--python", default="python")
    args = ap.parse_args()

    if args.update:
        out = {"cases": {}}
        for name, cfg_path, overrides in CASES:
            cfg = _load_config(cfg_path)
            physics = str(cfg.get("PHYSICS", "hydro")).lower()
            tmp_cfg = _write_temp_config_with_overrides(cfg, overrides)
            try:
                run_dir = _run_case(args.python, tmp_cfg)
            finally:
                os.remove(tmp_cfg)
            metrics = _get_metrics(run_dir, physics)
            out["cases"][name] = {
                "config": cfg_path,
                "physics": physics,
                "metrics": metrics,
                "tolerances": DEFAULT_TOLS,
            }
        _write_json(args.baseline, out)
        print(f"[baseline] wrote {args.baseline}")
        return

    if not os.path.exists(args.baseline):
        raise SystemExit(f"missing baseline file: {args.baseline} (run with --update)")
    base = _load_json(args.baseline)
    errors = []
    for name, cfg_path, overrides in CASES:
        if "cases" not in base or name not in base["cases"]:
            errors.append(f"{name}: missing in baseline file")
            continue
        entry = base["cases"][name]
        cfg = _load_config(cfg_path)
        physics = str(cfg.get("PHYSICS", entry.get("physics", "hydro"))).lower()
        tmp_cfg = _write_temp_config_with_overrides(cfg, overrides)
        try:
            run_dir = _run_case(args.python, tmp_cfg)
        finally:
            os.remove(tmp_cfg)
        metrics = _get_metrics(run_dir, physics)
        tols = entry.get("tolerances", {})
        errors.extend(_compare_metrics(name, metrics, entry["metrics"], tols))

    if errors:
        for err in errors:
            print(f"[baseline] {err}")
        raise SystemExit(1)
    print("[baseline] all baseline checks passed")


if __name__ == "__main__":
    main()
