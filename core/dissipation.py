#!/usr/bin/env python3
# core/dissipation.py
# Causal dissipation (Israel-Stewart) with IMEX relaxation for bulk pressure.

def apply_causal_dissipation(pr, dt, cfg):
    """
    Apply causal dissipation updates (placeholder).
    Expected to evolve additional dissipative variables with relaxation time.
    """
    if not cfg.get("DISSIPATION_ENABLED", False):
        return pr
    if cfg.get("PHYSICS") not in ("hydro", "grhd"):
        return pr

    ng = cfg.get("NG", 2)
    zeta = float(cfg.get("BULK_ZETA", 0.0))
    tau = float(cfg.get("RELAX_TAU", 0.1))
    if zeta == 0.0 or tau <= 0.0:
        return pr

    # Bulk pressure Pi stored in the last component for hydro/GRHD runs.
    if pr.shape[0] < 6:
        return pr
    pi_idx = 5

    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    for i in range(ng, nx-ng):
        for j in range(ng, ny-ng):
            for k in range(ng, nz-ng):
                dvx = (pr[1, i+1, j, k] - pr[1, i-1, j, k]) * 0.5
                dvy = (pr[2, i, j+1, k] - pr[2, i, j-1, k]) * 0.5
                dvz = (pr[3, i, j, k+1] - pr[3, i, j, k-1]) * 0.5
                divv = dvx + dvy + dvz

                pi_eq = -zeta * divv
                pi_old = pr[pi_idx, i, j, k]
                pi_new = (pi_old + dt * pi_eq / tau) / (1.0 + dt / tau)

                pmax = float(cfg.get("P_MAX", 1.0))
                if pi_new >  0.5*pmax: pi_new =  0.5*pmax
                if pi_new < -0.5*pmax: pi_new = -0.5*pmax
                pr[pi_idx, i, j, k] = pi_new

    return pr
