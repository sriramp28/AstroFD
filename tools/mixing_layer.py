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


def _get_tracer(data, name, index):
    if name is not None:
        if name in data.files:
            return data[name]
        return None
    if index is not None:
        key = f"tracer{index}"
        if key in data.files:
            return data[key]
    if "tracer0" in data.files:
        return data["tracer0"]
    return None


def main():
    ap = argparse.ArgumentParser(description="Estimate mixing layer thickness from a tracer.")
    ap.add_argument("npz", help="snapshot .npz")
    ap.add_argument("--ng", type=int, default=2, help="ghost layers")
    ap.add_argument("--tracer-name", type=str, default=None, help="tracer field name")
    ap.add_argument("--tracer-index", type=int, default=None, help="tracer index (tracer0, tracer1, ...)")
    ap.add_argument("--hi", type=float, default=0.9, help="upper tracer threshold")
    ap.add_argument("--lo", type=float, default=0.1, help="lower tracer threshold")
    ap.add_argument("--nbins", type=int, default=32, help="radial bins")
    ap.add_argument("--y0", type=float, default=None, help="center y")
    ap.add_argument("--z0", type=float, default=None, help="center z")
    args = ap.parse_args()

    data, dx, dy, dz = _load_npz(args.npz)
    tracer = _get_tracer(data, args.tracer_name, args.tracer_index)
    if tracer is None:
        raise SystemExit("no tracer field found (use --tracer-name or --tracer-index)")

    ng = args.ng
    nx = tracer.shape[0] - 2*ng
    ny = tracer.shape[1] - 2*ng
    nz = tracer.shape[2] - 2*ng

    y0 = args.y0 if args.y0 is not None else 0.5 * ny * dy
    z0 = args.z0 if args.z0 is not None else 0.5 * nz * dz

    rmax = np.sqrt(((ny * dy) * 0.5)**2 + ((nz * dz) * 0.5)**2)
    edges = np.linspace(0.0, rmax, args.nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    print("i,x,r_hi,r_lo,thickness,mixing_index")
    for i in range(ng, ng + nx):
        sums = np.zeros(args.nbins)
        counts = np.zeros(args.nbins, dtype=np.int64)
        mix_sum = 0.0
        mix_count = 0
        for j in range(ng, ng + ny):
            y = (j - ng + 0.5) * dy
            for k in range(ng, ng + nz):
                z = (k - ng + 0.5) * dz
                r = np.sqrt((y - y0)**2 + (z - z0)**2)
                b = np.searchsorted(edges, r, side="right") - 1
                if b < 0 or b >= args.nbins:
                    continue
                tv = tracer[i, j, k]
                sums[b] += tv
                counts[b] += 1
                mix_sum += tv * (1.0 - tv)
                mix_count += 1
        mean = np.zeros(args.nbins)
        nonzero = counts > 0
        mean[nonzero] = sums[nonzero] / counts[nonzero]

        r_hi = np.nan
        r_lo = np.nan
        for b in range(args.nbins - 1):
            if mean[b] >= args.hi and mean[b+1] < args.hi:
                frac = (args.hi - mean[b+1]) / max(mean[b] - mean[b+1], 1e-12)
                r_hi = centers[b+1] + frac * (centers[b] - centers[b+1])
                break
        for b in range(args.nbins - 1):
            if mean[b] >= args.lo and mean[b+1] < args.lo:
                frac = (args.lo - mean[b+1]) / max(mean[b] - mean[b+1], 1e-12)
                r_lo = centers[b+1] + frac * (centers[b] - centers[b+1])
                break

        thickness = r_lo - r_hi if np.isfinite(r_hi) and np.isfinite(r_lo) else np.nan
        mix_index = mix_sum / mix_count if mix_count > 0 else np.nan
        x = (i - ng + 0.5) * dx
        print(f"{i-ng},{x:.6e},{r_hi:.6e},{r_lo:.6e},{thickness:.6e},{mix_index:.6e}")


if __name__ == "__main__":
    main()
