#!/usr/bin/env python3
import argparse
import json

from core import chemistry


def _load_cfg(path):
    try:
        import json5
        with open(path, "r") as f:
            return json5.load(f)
    except Exception:
        with open(path, "r") as f:
            return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Single-cell ion chemistry relaxation test.")
    ap.add_argument("--config", required=True, help="config JSON/JSON5")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--xHII", type=float, default=None)
    ap.add_argument("--xHeII", type=float, default=None)
    ap.add_argument("--xHeIII", type=float, default=None)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    cfg = dict(cfg)
    cfg["CHEMISTRY_ENABLED"] = True
    cfg["NG"] = 1

    # compute offsets for a hydro-only single cell
    base = 5
    cfg["CHEM_OFFSET"] = base
    cfg["N_CHEM"] = 3

    rho = args.rho if args.rho is not None else float(cfg.get("RHO_AMB", 1.0))
    p = args.p if args.p is not None else float(cfg.get("P_AMB", cfg.get("P_EQ", 1.0e-2)))
    xHII = args.xHII if args.xHII is not None else float(cfg.get("CHEM_X_HII_AMB", 0.0))
    xHeII = args.xHeII if args.xHeII is not None else float(cfg.get("CHEM_X_HEII_AMB", 0.0))
    xHeIII = args.xHeIII if args.xHeIII is not None else float(cfg.get("CHEM_X_HEIII_AMB", 0.0))

    import numpy as np
    n = 1 + 2 * cfg["NG"]
    pr = np.zeros((base + 3, n, n, n), dtype=np.float64)
    pr[0, 1, 1, 1] = rho
    pr[4, 1, 1, 1] = p
    pr[base + 0, 1, 1, 1] = xHII
    pr[base + 1, 1, 1, 1] = xHeII
    pr[base + 2, 1, 1, 1] = xHeIII

    print("step,p,xHII,xHeII,xHeIII")
    for n in range(args.steps + 1):
        xHII = pr[base + 0][1][1][1]
        xHeII = pr[base + 1][1][1][1]
        xHeIII = pr[base + 2][1][1][1]
        p = pr[4][1][1][1]
        print(f"{n},{p:.6e},{xHII:.6e},{xHeII:.6e},{xHeIII:.6e}")
        pr = chemistry.apply_ion_chemistry(pr, args.dt, cfg)


if __name__ == "__main__":
    main()
