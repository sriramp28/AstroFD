#!/usr/bin/env python3
# core/source_terms.py
# Simple source terms: cooling/heating and two-temperature relaxation.

def apply_cooling_heating(pr, dt, cfg):
    if not cfg.get("COOLING_ENABLED", False):
        return pr
    cool = float(cfg.get("COOLING_LAMBDA", 0.0))
    heat = float(cfg.get("HEATING_RATE", 0.0))
    if cool == 0.0 and heat == 0.0:
        return pr
    ng = cfg.get("NG", 2)
    pmax = float(cfg.get("P_MAX", 1.0))

    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    for i in range(ng, nx-ng):
        for j in range(ng, ny-ng):
            for k in range(ng, nz-ng):
                rho = pr[0, i, j, k]
                p = pr[4, i, j, k]
                dp = dt * (heat - cool * rho * p)
                p_new = p + dp
                if p_new < 1e-12:
                    p_new = 1e-12
                if p_new > pmax:
                    p_new = pmax
                pr[4, i, j, k] = p_new
    return pr

def apply_two_temperature(pr, dt, cfg):
    if not cfg.get("TWO_TEMPERATURE", False):
        return pr
    ntr = int(cfg.get("N_TRACERS", 0))
    off = int(cfg.get("THERMO_OFFSET", 0))
    if pr.shape[0] < off + 2:
        return pr
    tau = float(cfg.get("TEI_TAU", 0.5))
    if tau <= 0.0:
        return pr
    ng = cfg.get("NG", 2)

    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    for i in range(ng, nx-ng):
        for j in range(ng, ny-ng):
            for k in range(ng, nz-ng):
                rho = pr[0, i, j, k]
                p = pr[4, i, j, k]
                T = p / max(rho, 1e-12)
                Te = pr[off, i, j, k]
                Ti = pr[off+1, i, j, k]
                Te += dt * (T - Te) / tau
                Ti += dt * (T - Ti) / tau
                pr[off, i, j, k] = Te
                pr[off+1, i, j, k] = Ti
    return pr
