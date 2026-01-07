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


def apply_sn_heating(pr, dt, dx, dy, dz, cfg, offs_x, ng):
    """
    Parametric SN-lite heating/cooling applied to pressure.
    Models a gain region with optional radial profiles.
    """
    if not cfg.get("SN_HEATING_ENABLED", False):
        return pr
    if cfg.get("PHYSICS") not in ("sn",):
        return pr

    model = str(cfg.get("SN_HEATING_MODEL", "gain_spherical")).lower()
    h0 = float(cfg.get("SN_HEATING_RATE", 0.0))
    c0 = float(cfg.get("SN_COOLING_RATE", 0.0))
    r0 = float(cfg.get("SN_GAIN_RADIUS", 0.2))
    r1 = float(cfg.get("SN_GAIN_WIDTH", 0.1))
    gamma = float(cfg.get("GAMMA", 5.0/3.0))
    pmax = float(cfg.get("P_MAX", 1.0))
    center = cfg.get("SN_GRAVITY_CENTER", [0.5*cfg.get("Lx", 1.0), 0.5*cfg.get("Ly", 1.0), 0.5*cfg.get("Lz", 1.0)])
    x0, y0, z0 = float(center[0]), float(center[1]), float(center[2])

    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]

    for i in range(ng, nx - ng):
        x = (offs_x + (i - ng) + 0.5) * dx
        for j in range(ng, ny - ng):
            y = (j - ng + 0.5) * dy
            for k in range(ng, nz - ng):
                z = (k - ng + 0.5) * dz
                rx = x - x0
                ry = y - y0
                rz = z - z0
                r = (rx*rx + ry*ry + rz*rz) ** 0.5

                heat = 0.0
                cool = 0.0
                if model == "gain_spherical":
                    if r > r0:
                        xi = (r - r0) / max(r1, 1e-12)
                        weight = 1.0 / (1.0 + xi*xi)
                        heat = h0 * weight
                        cool = c0 * weight
                elif model == "constant":
                    heat = h0
                    cool = c0

                if heat != 0.0 or cool != 0.0:
                    rho = pr[0, i, j, k]
                    if rho <= 0.0:
                        continue
                    dp = (gamma - 1.0) * (heat - cool) * dt
                    p = pr[4, i, j, k] + dp
                    if p < 1e-12:
                        p = 1e-12
                    if p > pmax:
                        p = pmax
                    pr[4, i, j, k] = p

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
