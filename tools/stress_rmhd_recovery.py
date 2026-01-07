#!/usr/bin/env python3
import argparse
import math
import numpy as np

from core import rmhd_core


def main():
    ap = argparse.ArgumentParser(description="Stress-test RMHD primitive recovery.")
    ap.add_argument("--n", type=int, default=1000, help="number of random states")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tol", type=float, default=1e-2, help="relative tolerance on rho,p,v")
    ap.add_argument("--max-v", type=float, default=0.5)
    ap.add_argument("--max-p", type=float, default=1.0)
    ap.add_argument("--max-b", type=float, default=0.05)
    ap.add_argument("--max-fail-frac", type=float, default=0.2)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rmhd_core.configure({"GAMMA": 5.0/3.0, "P_MAX": 100.0, "V_MAX": 0.9999})

    n = args.n
    tol = args.tol
    max_v = args.max_v
    max_p = args.max_p
    max_b = args.max_b

    bad = 0
    worst = 0.0

    for _ in range(n):
        rho = 10.0 ** rng.uniform(-2.0, 1.0)
        p = 10.0 ** rng.uniform(-3.0, math.log10(max_p))
        vx, vy, vz = rng.uniform(-max_v, max_v, size=3)
        v2 = vx*vx + vy*vy + vz*vz
        if v2 >= max_v*max_v:
            fac = max_v / math.sqrt(v2 + 1e-32)
            vx *= fac; vy *= fac; vz *= fac
        Bx, By, Bz = rng.uniform(-max_b, max_b, size=3)
        psi = 0.0

        D, Sx, Sy, Sz, tau, _, _, _, _ = rmhd_core.prim_to_cons_rmhd(
            rho, vx, vy, vz, p, Bx, By, Bz, psi, rmhd_core.GAMMA
        )

        r2, vx2, vy2, vz2, p2, _, _, _, _ = rmhd_core.cons_to_prim_rmhd(D, Sx, Sy, Sz, tau, Bx, By, Bz, psi)

        if not np.isfinite([r2, vx2, vy2, vz2, p2]).all():
            bad += 1
            continue

        dv = max(abs(vx2 - vx), abs(vy2 - vy), abs(vz2 - vz))
        dr = abs(r2 - rho) / max(rho, 1e-12)
        dp = abs(p2 - p) / max(p, 1e-12)
        err = max(dv, dr, dp)
        if err > worst:
            worst = err
        if err > tol:
            bad += 1

    frac = bad / max(n, 1)
    print(f"[rmhd-recovery] n={n} bad={bad} frac={frac:.3f} worst_err={worst:.3e} tol={tol:.3e}")
    if frac > args.max_fail_frac:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
