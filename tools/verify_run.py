#!/usr/bin/env python3
import argparse
import os, glob, numpy as np

def latest_run_dir(base="results"):
    runs = sorted([d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)])
    if not runs: raise SystemExit("No results/ runs found.")
    # If you used daily folders with unique subfolders, pick the deepest latest
    # Otherwise the last entry is fine.
    # This handles both results/YYYY-MM-DD/ and results/YYYY-MM-DD/HH-MM-SS/
    last = runs[-1]
    subs = sorted([d for d in glob.glob(os.path.join(last, "*")) if os.path.isdir(d)])
    return subs[-1] if subs else last

def main():
    ap = argparse.ArgumentParser(description="Verify hydro/GRHD output sanity checks.")
    ap.add_argument("--max-gamma", type=float, default=None)
    ap.add_argument("--max-vel", type=float, default=None)
    args = ap.parse_args()

    run_dir = latest_run_dir()
    # Pick rank0 latest file
    files = sorted(glob.glob(os.path.join(run_dir, "jet3d_rank0000_step*.npz")))
    if not files: raise SystemExit(f"No NPZ files in {run_dir}")
    npz = files[-1]
    print(f"[verify] using {npz}")

    data = np.load(npz)
    rho = data["rho"]; vx = data["vx"]; vy = data["vy"]; vz = data["vz"]; p = data["p"]
    NG = 2  # our solver’s ghost thickness

    # Basic stats over interior
    sl = (slice(NG, -NG), slice(NG, -NG), slice(NG, -NG))
    v2 = vx[sl]**2 + vy[sl]**2 + vz[sl]**2
    v2 = np.clip(v2, 0, 1 - 1e-14)
    G  = 1.0/np.sqrt(1.0 - v2)

    max_v = np.max(np.abs(vx[sl]))
    print(f"[verify] interior max |vx| = {max_v:.6e}")
    max_g = np.max(G)
    print(f"[verify] interior max  Γ   = {max_g:.6f}")
    print(f"[verify] interior mean ρ, p = {np.mean(rho[sl]):.6e}, {np.mean(p[sl]):.6e}")

    # First interior inlet plane (i = NG)
    i = NG
    v2_face = vx[i,:,:]**2 + vy[i,:,:]**2 + vz[i,:,:]**2
    G_face  = 1.0/np.sqrt(1.0 - np.clip(v2_face, 0, 1 - 1e-14))
    max_g_face = np.max(G_face)
    print(f"[verify] inlet plane max Γ (i=NG) = {max_g_face:.6f}")

    # Recompute inlet energy flux by summation over the *inlet plane* cells
    # Sx = (from prim_to_cons): w*W^2*vx with w = rho * (1 + gamma/(gamma-1) * p/rho)
    gamma = 5.0/3.0
    h  = 1.0 + gamma/(gamma-1.0) * (p[i,:,:]/np.maximum(rho[i,:,:], 1e-12))
    w  = rho[i,:,:] * h
    W2 = G_face**2
    Sx = w * W2 * vx[i,:,:]
    # Recover dy, dz from meta (dx,dy,dz,t) if present; else infer assuming unit box and NY=NZ = shape-2*NG
    try:
        dx, dy, dz, t = data["meta"]
    except Exception:
        ny = rho.shape[1] - 2*NG; nz = rho.shape[2] - 2*NG
        dy = 1.0/ny; dz = 1.0/nz
    inlet_flux = np.sum(Sx[NG:-NG, NG:-NG]) * dy * dz
    print(f"[verify] recomputed inlet flux = {inlet_flux:.6e}")

    if args.max_gamma is not None and max_g > args.max_gamma:
        raise SystemExit(f"max Gamma {max_g:.3f} > {args.max_gamma}")
    if args.max_vel is not None and max_v > args.max_vel:
        raise SystemExit(f"max |vx| {max_v:.3e} > {args.max_vel}")

if __name__ == "__main__":
    main()
