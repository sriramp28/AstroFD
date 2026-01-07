#!/usr/bin/env python3
import argparse
import numpy as np


def _load_npz(path):
    data = np.load(path)
    meta = data["meta"] if "meta" in data.files else None
    dx = dy = dz = None
    if meta is not None and len(meta) >= 3:
        dx, dy, dz = float(meta[0]), float(meta[1]), float(meta[2])
    return data, dx, dy, dz


def main():
    ap = argparse.ArgumentParser(description="Compute isotropic velocity spectrum from a snapshot.")
    ap.add_argument("npz", help="snapshot .npz")
    ap.add_argument("--ng", type=int, default=2, help="ghost layers")
    ap.add_argument("--nbins", type=int, default=32, help="k bins")
    args = ap.parse_args()

    data, dx, dy, dz = _load_npz(args.npz)
    vx = data["vx"]; vy = data["vy"]; vz = data["vz"]
    ng = args.ng
    vx = vx[ng:-ng, ng:-ng, ng:-ng]
    vy = vy[ng:-ng, ng:-ng, ng:-ng]
    vz = vz[ng:-ng, ng:-ng, ng:-ng]

    nx, ny, nz = vx.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
    kx3, ky3, kz3 = np.meshgrid(kx, ky, kz, indexing="ij")
    kmag = np.sqrt(kx3*kx3 + ky3*ky3 + kz3*kz3)

    vxk = np.fft.fftn(vx)
    vyk = np.fft.fftn(vy)
    vzk = np.fft.fftn(vz)
    ek = 0.5 * (np.abs(vxk)**2 + np.abs(vyk)**2 + np.abs(vzk)**2) / (nx*ny*nz)

    kmax = np.max(kmag)
    edges = np.linspace(0.0, kmax, args.nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    spec = np.zeros(args.nbins)
    counts = np.zeros(args.nbins, dtype=np.int64)

    flat_k = kmag.ravel()
    flat_e = ek.ravel()
    bin_idx = np.searchsorted(edges, flat_k, side="right") - 1
    for idx, e in zip(bin_idx, flat_e):
        if 0 <= idx < args.nbins:
            spec[idx] += e
            counts[idx] += 1

    nzmask = counts > 0
    spec[nzmask] /= counts[nzmask]

    print("k,E_k,count")
    for k, e, c in zip(centers, spec, counts):
        print(f"{k:.6e},{e:.6e},{c}")


if __name__ == "__main__":
    main()
