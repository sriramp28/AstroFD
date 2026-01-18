#!/usr/bin/env python3
import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from core import rmhd_core


def main():
    ap = argparse.ArgumentParser(description="Stress-test RMHD primitive recovery.")
    ap.add_argument("--n", type=int, default=1000, help="number of random states per regime")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tol", type=float, default=1e-2, help="relative tolerance on rho,p,v")
    ap.add_argument("--regimes", type=str, default="mild,relativistic,magnetized",
                    help="comma-separated regimes: mild,relativistic,magnetized,cold,hot")
    ap.add_argument("--max-v", type=float, default=0.5)
    ap.add_argument("--max-p", type=float, default=1.0)
    ap.add_argument("--max-b", type=float, default=0.05)
    ap.add_argument("--cons-tol", type=float, default=5e-3,
                    help="relative tolerance on D,S,tau for conservative check")
    ap.add_argument("--max-fail-frac", type=float, default=0.2)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rmhd_core.configure({"GAMMA": 5.0/3.0, "P_MAX": 100.0, "V_MAX": 0.9999})

    n = args.n
    tol = args.tol
    regimes = [r.strip().lower() for r in args.regimes.split(",") if r.strip()]

    def sample_state(regime):
        if regime == "relativistic":
            max_v = 0.98
            max_b = max(args.max_b, 0.2)
            pmax = max(args.max_p, 10.0)
            rho = 10.0 ** rng.uniform(-2.0, 1.0)
            p = 10.0 ** rng.uniform(-3.0, math.log10(pmax))
        elif regime == "magnetized":
            max_v = max(args.max_v, 0.6)
            max_b = max(args.max_b, 1.0)
            pmax = max(args.max_p, 1.0)
            rho = 10.0 ** rng.uniform(-3.0, 1.0)
            p = 10.0 ** rng.uniform(-5.0, math.log10(pmax))
        elif regime == "cold":
            max_v = max(args.max_v, 0.5)
            max_b = max(args.max_b, 0.2)
            rho = 10.0 ** rng.uniform(-1.0, 1.0)
            p = 10.0 ** rng.uniform(-6.0, -3.0)
        elif regime == "hot":
            max_v = max(args.max_v, 0.5)
            max_b = max(args.max_b, 0.2)
            rho = 10.0 ** rng.uniform(-2.0, 0.5)
            p = 10.0 ** rng.uniform(-2.0, math.log10(max(args.max_p, 10.0)))
        else:
            max_v = args.max_v
            max_b = args.max_b
            rho = 10.0 ** rng.uniform(-2.0, 1.0)
            p = 10.0 ** rng.uniform(-3.0, math.log10(max(args.max_p, 1e-6)))

        vx, vy, vz = rng.uniform(-max_v, max_v, size=3)
        v2 = vx*vx + vy*vy + vz*vz
        if v2 >= max_v*max_v:
            fac = max_v / math.sqrt(v2 + 1e-32)
            vx *= fac; vy *= fac; vz *= fac
        Bx, By, Bz = rng.uniform(-max_b, max_b, size=3)
        return rho, vx, vy, vz, p, Bx, By, Bz

    overall_bad = 0
    overall = 0
    worst = 0.0
    status_counts = {"bisection": 0, "fallback": 0, "vclip": 0}
    cons_rescued = 0

    for regime in regimes:
        bad = 0
        for _ in range(n):
            rho, vx, vy, vz, p, Bx, By, Bz = sample_state(regime)
            psi = 0.0

            D, Sx, Sy, Sz, tau, _, _, _, _ = rmhd_core.prim_to_cons_rmhd(
                rho, vx, vy, vz, p, Bx, By, Bz, psi, rmhd_core.GAMMA
            )

            r2, vx2, vy2, vz2, p2, _, _, _, _, status = rmhd_core.cons_to_prim_rmhd_status(
                D, Sx, Sy, Sz, tau, Bx, By, Bz, psi
            )

            if status & 1:
                status_counts["bisection"] += 1
            if status & 2:
                status_counts["fallback"] += 1
            if status & 4:
                status_counts["vclip"] += 1

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
                D2, Sx2, Sy2, Sz2, tau2, _, _, _, _ = rmhd_core.prim_to_cons_rmhd(
                    r2, vx2, vy2, vz2, p2, Bx, By, Bz, psi, rmhd_core.GAMMA
                )
                cD = abs(D2 - D) / max(abs(D), 1e-12)
                cSx = abs(Sx2 - Sx) / max(abs(Sx), 1e-12)
                cSy = abs(Sy2 - Sy) / max(abs(Sy), 1e-12)
                cSz = abs(Sz2 - Sz) / max(abs(Sz), 1e-12)
                cT = abs(tau2 - tau) / max(abs(tau), 1e-12)
                if max(cD, cSx, cSy, cSz, cT) <= args.cons_tol:
                    cons_rescued += 1
                else:
                    bad += 1

        frac = bad / max(n, 1)
        overall_bad += bad
        overall += n
        print(f"[rmhd-recovery:{regime}] n={n} bad={bad} frac={frac:.3f} tol={tol:.3e}")
        if frac > args.max_fail_frac:
            raise SystemExit(1)

    frac = overall_bad / max(overall, 1)
    print(f"[rmhd-recovery] total={overall} bad={overall_bad} frac={frac:.3f} worst_err={worst:.3e}")
    print(f"[rmhd-recovery] cons-rescued={cons_rescued}")
    print(f"[rmhd-recovery] status bisection={status_counts['bisection']} fallback={status_counts['fallback']} vclip={status_counts['vclip']}")
    if frac > args.max_fail_frac:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
