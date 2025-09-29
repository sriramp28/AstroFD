#!/usr/bin/env python3
import os, glob, numpy as np
import matplotlib.pyplot as plt

def latest_run_dir(base="results"):
    runs = sorted([d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)])
    if not runs: raise SystemExit("No results/ runs found.")
    return runs[-1]

def load_any_rank_npz(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "jet3d_rank*_step*.npz")))
    if not files: raise SystemExit(f"No NPZ files in {run_dir}")
    return files[-1]

def main():
    run_dir = latest_run_dir()
    npz_path = load_any_rank_npz(run_dir)
    print(f"[quickview] Using {npz_path}")

    blk = np.load(npz_path)
    rho = blk["rho"]   # (nx_loc+2*NG, NY+2*NG, NZ+2*NG)
    vx  = blk["vx"]; vy = blk["vy"]; vz = blk["vz"]

    # midplane slice (z-mid)
    k = rho.shape[2]//2
    slc = rho[:, :, k].T
    plt.figure()
    plt.imshow(slc, origin="lower", aspect="auto")
    plt.colorbar(label="rho")
    out_png = os.path.join(run_dir, "rho_midZ.png")
    plt.title("rho (mid-Z)")
    plt.savefig(out_png, dpi=150)
    print(f"[quickview] wrote {out_png}")

    # centerline Lorentz factor along x at y,z mid
    j = rho.shape[1]//2
    v2 = vx[:, j, k]**2 + vy[:, j, k]**2 + vz[:, j, k]**2
    v2 = np.clip(v2, 0, 1-1e-14)
    Gamma = 1.0/np.sqrt(1.0 - v2)
    plt.figure()
    plt.plot(Gamma)
    plt.xlabel("i (local)")
    plt.ylabel("Gamma (centerline)")
    out_png2 = os.path.join(run_dir, "centerline_Gamma.png")
    plt.title("Centerline Lorentz factor")
    plt.savefig(out_png2, dpi=150)
    print(f"[quickview] wrote {out_png2}")

if __name__ == "__main__":
    main()
