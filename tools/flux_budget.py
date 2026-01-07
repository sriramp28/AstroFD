#!/usr/bin/env python3
import argparse
import numpy as np

from core import srhd_core
from core import rmhd_core


def _load_npz(path):
    data = np.load(path)
    meta = data["meta"] if "meta" in data.files else None
    dx = dy = dz = None
    if meta is not None and len(meta) >= 3:
        dx, dy, dz = float(meta[0]), float(meta[1]), float(meta[2])
    return data, dx, dy, dz


def _plane_index(axis, coord, dx, dy, dz, n):
    if coord is None:
        return None
    d = {"x": dx, "y": dy, "z": dz}[axis]
    idx = int(np.floor(coord / max(d, 1e-12)))
    if idx < 0:
        idx = 0
    if idx > n - 1:
        idx = n - 1
    return idx


def main():
    ap = argparse.ArgumentParser(description="Flux budget through a plane for SRHD/RMHD outputs.")
    ap.add_argument("npz", help="snapshot .npz")
    ap.add_argument("--axis", choices=["x", "y", "z"], default="x", help="plane normal axis")
    ap.add_argument("--plane-index", type=int, default=None, help="plane index in interior coords")
    ap.add_argument("--plane-coord", type=float, default=None, help="plane coordinate (same units as domain)")
    ap.add_argument("--ng", type=int, default=2, help="ghost layers")
    args = ap.parse_args()

    data, dx, dy, dz = _load_npz(args.npz)
    rho = data["rho"]; vx = data["vx"]; vy = data["vy"]; vz = data["vz"]; p = data["p"]
    is_rmhd = "Bx" in data.files
    Bx = By = Bz = psi = None
    if is_rmhd:
        Bx = data["Bx"]; By = data["By"]; Bz = data["Bz"]; psi = data["psi"]

    ng = args.ng
    nx = rho.shape[0] - 2*ng
    ny = rho.shape[1] - 2*ng
    nz = rho.shape[2] - 2*ng

    if args.plane_index is None:
        idx = _plane_index(args.axis, args.plane_coord, dx, dy, dz, {"x": nx, "y": ny, "z": nz}[args.axis])
    else:
        idx = args.plane_index
    if idx is None:
        idx = 0
    if idx < 0:
        idx = 0

    if args.axis == "x":
        i = ng + min(idx, nx - 1)
        j0, j1 = ng, ng + ny
        k0, k1 = ng, ng + nz
        dA = dy * dz
        flux = np.zeros(9 if is_rmhd else 5)
        for j in range(j0, j1):
            for k in range(k0, k1):
                if is_rmhd:
                    prim = np.array([rho[i,j,k], vx[i,j,k], vy[i,j,k], vz[i,j,k], p[i,j,k],
                                     Bx[i,j,k], By[i,j,k], Bz[i,j,k], psi[i,j,k]])
                    F = rmhd_core.flux_rmhd_x(prim)
                else:
                    F = srhd_core.flux_x(rho[i,j,k], vx[i,j,k], vy[i,j,k], vz[i,j,k], p[i,j,k])
                flux[:F.size] += F * dA
    elif args.axis == "y":
        j = ng + min(idx, ny - 1)
        i0, i1 = ng, ng + nx
        k0, k1 = ng, ng + nz
        dA = dx * dz
        flux = np.zeros(9 if is_rmhd else 5)
        for i in range(i0, i1):
            for k in range(k0, k1):
                if is_rmhd:
                    prim = np.array([rho[i,j,k], vx[i,j,k], vy[i,j,k], vz[i,j,k], p[i,j,k],
                                     Bx[i,j,k], By[i,j,k], Bz[i,j,k], psi[i,j,k]])
                    F = rmhd_core.flux_rmhd_y(prim)
                else:
                    F = srhd_core.flux_y(rho[i,j,k], vx[i,j,k], vy[i,j,k], vz[i,j,k], p[i,j,k])
                flux[:F.size] += F * dA
    else:
        k = ng + min(idx, nz - 1)
        i0, i1 = ng, ng + nx
        j0, j1 = ng, ng + ny
        dA = dx * dy
        flux = np.zeros(9 if is_rmhd else 5)
        for i in range(i0, i1):
            for j in range(j0, j1):
                if is_rmhd:
                    prim = np.array([rho[i,j,k], vx[i,j,k], vy[i,j,k], vz[i,j,k], p[i,j,k],
                                     Bx[i,j,k], By[i,j,k], Bz[i,j,k], psi[i,j,k]])
                    F = rmhd_core.flux_rmhd_z(prim)
                else:
                    F = srhd_core.flux_z(rho[i,j,k], vx[i,j,k], vy[i,j,k], vz[i,j,k], p[i,j,k])
                flux[:F.size] += F * dA

    if is_rmhd:
        print(f"axis={args.axis} idx={idx} mass={flux[0]:.6e} "
              f"momx={flux[1]:.6e} momy={flux[2]:.6e} momz={flux[3]:.6e} "
              f"energy={flux[4]:.6e}")
    else:
        print(f"axis={args.axis} idx={idx} mass={flux[0]:.6e} "
              f"momx={flux[1]:.6e} momy={flux[2]:.6e} momz={flux[3]:.6e} "
              f"energy={flux[4]:.6e}")


if __name__ == "__main__":
    main()
