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
    return None


def main():
    ap = argparse.ArgumentParser(description="Estimate cocoon pressure from a snapshot.")
    ap.add_argument("npz", help="snapshot .npz")
    ap.add_argument("--ng", type=int, default=2, help="ghost layers")
    ap.add_argument("--tracer-name", type=str, default=None, help="tracer field name")
    ap.add_argument("--tracer-index", type=int, default=None, help="tracer index (tracer0, tracer1, ...)")
    ap.add_argument("--tracer-min", type=float, default=0.05, help="min tracer value for selection")
    ap.add_argument("--tracer-max", type=float, default=0.95, help="max tracer value for selection")
    ap.add_argument("--rmin", type=float, default=None, help="min radius from center")
    ap.add_argument("--rmax", type=float, default=None, help="max radius from center")
    ap.add_argument("--y0", type=float, default=None, help="center y for radius filter")
    ap.add_argument("--z0", type=float, default=None, help="center z for radius filter")
    ap.add_argument("--per-x", action="store_true", help="write per-x mean pressure")
    args = ap.parse_args()

    data, dx, dy, dz = _load_npz(args.npz)
    p = data["p"]
    ng = args.ng
    nx = p.shape[0] - 2*ng
    ny = p.shape[1] - 2*ng
    nz = p.shape[2] - 2*ng

    tracer = _get_tracer(data, args.tracer_name, args.tracer_index)
    use_tracer = tracer is not None

    y0 = args.y0 if args.y0 is not None else 0.5 * ny * dy
    z0 = args.z0 if args.z0 is not None else 0.5 * nz * dz

    def cell_pos(j, k):
        y = (j - ng + 0.5) * dy
        z = (k - ng + 0.5) * dz
        return y, z

    if args.per_x:
        print("i,x,mean_p,count")
        for i in range(ng, ng + nx):
            vals = []
            for j in range(ng, ng + ny):
                for k in range(ng, ng + nz):
                    if use_tracer:
                        tv = tracer[i, j, k]
                        if tv < args.tracer_min or tv > args.tracer_max:
                            continue
                    if args.rmin is not None or args.rmax is not None:
                        y, z = cell_pos(j, k)
                        r = np.sqrt((y - y0)**2 + (z - z0)**2)
                        if args.rmin is not None and r < args.rmin:
                            continue
                        if args.rmax is not None and r > args.rmax:
                            continue
                    vals.append(p[i, j, k])
            if vals:
                mean_p = float(np.mean(vals))
                x = (i - ng + 0.5) * dx
                print(f"{i-ng},{x:.6e},{mean_p:.6e},{len(vals)}")
        return

    vals = []
    for i in range(ng, ng + nx):
        for j in range(ng, ng + ny):
            for k in range(ng, ng + nz):
                if use_tracer:
                    tv = tracer[i, j, k]
                    if tv < args.tracer_min or tv > args.tracer_max:
                        continue
                if args.rmin is not None or args.rmax is not None:
                    y, z = cell_pos(j, k)
                    r = np.sqrt((y - y0)**2 + (z - z0)**2)
                    if args.rmin is not None and r < args.rmin:
                        continue
                    if args.rmax is not None and r > args.rmax:
                        continue
                vals.append(p[i, j, k])

    if vals:
        print(f"mean_p={float(np.mean(vals)):.6e} count={len(vals)}")
    else:
        print("mean_p=nan count=0")


if __name__ == "__main__":
    main()
