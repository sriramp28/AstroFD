#!/usr/bin/env python3
import argparse
import numpy as np
from utils.backend import get_backend, to_numpy


def _load_npz(path):
    data = np.load(path)
    meta = data["meta"] if "meta" in data.files else None
    dx = dy = dz = None
    if meta is not None and len(meta) >= 3:
        dx, dy, dz = float(meta[0]), float(meta[1]), float(meta[2])
    return data, dx, dy, dz


def main():
    ap = argparse.ArgumentParser(description="Compute 2nd-order velocity structure function along x.")
    ap.add_argument("npz", help="snapshot .npz")
    ap.add_argument("--ng", type=int, default=2, help="ghost layers")
    ap.add_argument("--rmax", type=int, default=16, help="max separation in grid cells")
    ap.add_argument("--backend", default="numpy", help="numpy or cupy")
    args = ap.parse_args()

    data, dx, dy, dz = _load_npz(args.npz)
    vx = data["vx"]; vy = data["vy"]; vz = data["vz"]
    xp, backend = get_backend(args.backend)
    ng = args.ng
    vx = xp.asarray(vx[ng:-ng, ng:-ng, ng:-ng])
    vy = xp.asarray(vy[ng:-ng, ng:-ng, ng:-ng])
    vz = xp.asarray(vz[ng:-ng, ng:-ng, ng:-ng])

    nx, ny, nz = vx.shape
    rmax = min(args.rmax, nx - 1)

    print("r,S2,count")
    for r in range(1, rmax + 1):
        dvx = vx[r:, :, :] - vx[:-r, :, :]
        dvy = vy[r:, :, :] - vy[:-r, :, :]
        dvz = vz[r:, :, :] - vz[:-r, :, :]
        s2 = dvx*dvx + dvy*dvy + dvz*dvz
        mean_s2 = float(to_numpy(xp.mean(s2)))
        count = s2.size
        print(f"{r*dx:.6e},{mean_s2:.6e},{count}")


if __name__ == "__main__":
    main()
