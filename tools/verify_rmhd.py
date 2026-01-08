#!/usr/bin/env python3
import argparse
import os, glob, numpy as np

def latest_run_dir(base="results"):
    runs = sorted([d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)])
    if not runs: raise SystemExit("No results/ runs found.")
    last = runs[-1]
    subs = sorted([d for d in glob.glob(os.path.join(last, "*")) if os.path.isdir(d)])
    return subs[-1] if subs else last

def main():
    ap = argparse.ArgumentParser(description="Verify RMHD output sanity checks.")
    ap.add_argument("--max-divb-rel", type=float, default=None)
    ap.add_argument("--max-psi", type=float, default=None)
    ap.add_argument("--max-b", type=float, default=None)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    files = sorted(glob.glob(os.path.join(run_dir, "jet3d_rank0000_step*.npz")))
    if not files: raise SystemExit(f"No NPZ files in {run_dir}")
    npz = files[-1]
    print(f"[verify-rmhd] using {npz}")

    data = np.load(npz)
    rho = data["rho"]; vx = data["vx"]; vy = data["vy"]; vz = data["vz"]; p = data["p"]
    Bx = data["Bx"]; By = data["By"]; Bz = data["Bz"]; psi = data["psi"]
    NG = 2

    sl = (slice(NG, -NG), slice(NG, -NG), slice(NG, -NG))
    v2 = vx[sl]**2 + vy[sl]**2 + vz[sl]**2
    v2 = np.clip(v2, 0, 1 - 1e-14)
    G  = 1.0/np.sqrt(1.0 - v2)
    B2 = Bx[sl]**2 + By[sl]**2 + Bz[sl]**2

    print(f"[verify-rmhd] interior max |v| = {np.max(np.sqrt(v2)):.6e}")
    print(f"[verify-rmhd] interior max  Γ  = {np.max(G):.6f}")
    print(f"[verify-rmhd] interior mean ρ, p = {np.mean(rho[sl]):.6e}, {np.mean(p[sl]):.6e}")
    print(f"[verify-rmhd] interior mean |B| = {np.mean(np.sqrt(B2)):.6e}")
    print(f"[verify-rmhd] interior max  |B| = {np.max(np.sqrt(B2)):.6e}")
    print(f"[verify-rmhd] interior max  |psi| = {np.max(np.abs(psi[sl])):.6e}")

    # basic divB estimate at interior cells
    divb = (
        (Bx[NG+1:-NG+1, NG:-NG, NG:-NG] - Bx[NG-1:-NG-1, NG:-NG, NG:-NG]) / 2.0 +
        (By[NG:-NG, NG+1:-NG+1, NG:-NG] - By[NG:-NG, NG-1:-NG-1, NG:-NG]) / 2.0 +
        (Bz[NG:-NG, NG:-NG, NG+1:-NG+1] - Bz[NG:-NG, NG:-NG, NG-1:-NG-1]) / 2.0
    )
    divb_max = np.max(np.abs(divb))
    print(f"[verify-rmhd] divB max = {divb_max:.6e}")

    # normalized divB: divB / (|B|/dx)
    try:
        dx, dy, dz, _t = data["meta"]
    except Exception:
        nx = rho.shape[0] - 2*NG
        dx = 1.0 / max(nx, 1)
    bnorm = np.sqrt(B2)
    denom = np.maximum(bnorm / max(dx, 1e-12), 1e-12)
    rel_divb = divb_max / np.max(denom)
    print(f"[verify-rmhd] divB max / (|B|/dx) = {rel_divb:.6e}")

    if args.max_divb_rel is not None and rel_divb > args.max_divb_rel:
        raise SystemExit(f"divB relative {rel_divb:.3e} > {args.max_divb_rel}")
    if args.max_psi is not None and np.max(np.abs(psi[sl])) > args.max_psi:
        raise SystemExit(f"psi max {np.max(np.abs(psi[sl])):.3e} > {args.max_psi}")
    if args.max_b is not None and np.max(np.sqrt(B2)) > args.max_b:
        raise SystemExit(f"|B| max {np.max(np.sqrt(B2)):.3e} > {args.max_b}")

if __name__ == "__main__":
    main()
