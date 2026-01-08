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


def apply_sn_heating(pr, dt, dx, dy, dz, cfg, offs_x, ng, t):
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
    rp = float(cfg.get("SN_GAIN_POW", 2.0))
    rho_exp = float(cfg.get("SN_HEATING_RHO_EXP", 0.0))
    p_exp = float(cfg.get("SN_HEATING_P_EXP", 0.0))
    tau = float(cfg.get("SN_HEATING_TIME_TAU", 0.0))
    t0 = float(cfg.get("SN_HEATING_TIME_T0", 0.0))
    cool_texp = float(cfg.get("SN_COOLING_TEXP", 0.0))
    cool_tref = float(cfg.get("SN_COOLING_TREF", 1.0))
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
                elif model == "gain_powerlaw":
                    if r >= r0:
                        weight = (r0 / max(r, 1e-12)) ** max(rp, 0.0)
                        heat = h0 * weight
                    else:
                        weight = (r0 / max(r, 1e-12)) ** max(rp, 0.0)
                        cool = c0 * weight
                elif model == "gain_exponential":
                    if r >= r0:
                        xi = (r - r0) / max(r1, 1e-12)
                        weight = pow(2.718281828, -xi)
                        heat = h0 * weight
                    else:
                        xi = (r0 - r) / max(r1, 1e-12)
                        weight = pow(2.718281828, -xi)
                        cool = c0 * weight
                elif model == "gain_gaussian":
                    if r >= r0:
                        xi = (r - r0) / max(r1, 1e-12)
                        weight = pow(2.718281828, -(xi * xi))
                        heat = h0 * weight
                    else:
                        xi = (r0 - r) / max(r1, 1e-12)
                        weight = pow(2.718281828, -(xi * xi))
                        cool = c0 * weight
                elif model == "constant":
                    heat = h0
                    cool = c0

                if heat != 0.0 or cool != 0.0:
                    rho = pr[0, i, j, k]
                    if rho <= 0.0:
                        continue
                    p_local = pr[4, i, j, k]
                    scale = 1.0
                    if rho_exp != 0.0:
                        scale *= rho ** rho_exp
                    if p_exp != 0.0:
                        scale *= max(p_local, 1e-12) ** p_exp
                    if tau > 0.0:
                        tnorm = (t - t0) / max(tau, 1e-12)
                        if tnorm > 0.0:
                            scale *= pow(2.718281828, -tnorm)
                    if cool_texp != 0.0:
                        tgas = p_local / max(rho, 1e-12)
                        scale *= (max(tgas, 1e-12) / max(cool_tref, 1e-12)) ** cool_texp
                    dp = (gamma - 1.0) * (heat - cool) * scale * dt
                    p = p_local + dp
                    if p < 1e-12:
                        p = 1e-12
                    if p > pmax:
                        p = pmax
                    pr[4, i, j, k] = p

    return pr


def apply_radiation_coupling(pr, dt, cfg):
    """
    Simple radiation-plasma coupling: relax gas temperature toward T_rad.
    T is approximated as p/rho in code units.
    """
    if not cfg.get("RADIATION_COUPLING_ENABLED", False):
        return pr

    coeff = float(cfg.get("RADIATION_COEFF", 0.0))
    t_rad = float(cfg.get("RADIATION_T_RAD", 0.0))
    if coeff == 0.0:
        return pr
    gamma = float(cfg.get("GAMMA", 5.0/3.0))
    pmax = float(cfg.get("P_MAX", 1.0))
    ng = int(cfg.get("NG", 2))
    target = str(cfg.get("RADIATION_COUPLING_TARGET", "gas")).lower()
    two_temp = bool(cfg.get("TWO_TEMPERATURE", False))
    thermo_off = int(cfg.get("THERMO_OFFSET", 0))

    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    for i in range(ng, nx - ng):
        for j in range(ng, ny - ng):
            for k in range(ng, nz - ng):
                rho = pr[0, i, j, k]
                if rho <= 0.0:
                    continue
                if target in ("electrons", "electron") and two_temp and pr.shape[0] >= thermo_off + 2:
                    te = pr[thermo_off, i, j, k]
                    dT = (t_rad - te) * coeff * dt
                    te_new = te + dT
                    if te_new < 1e-12:
                        te_new = 1e-12
                    pr[thermo_off, i, j, k] = te_new
                    if target == "electrons":
                        continue
                t_gas = pr[4, i, j, k] / max(rho, 1e-12)
                dT = (t_rad - t_gas) * coeff * dt
                dp = (gamma - 1.0) * rho * dT
                p = pr[4, i, j, k] + dp
                if p < 1e-12:
                    p = 1e-12
                if p > pmax:
                    p = pmax
                pr[4, i, j, k] = p

    return pr


def apply_kinetic_effects(pr, dt, dx, dy, dz, cfg, ng):
    """
    Effective kinetic heating using local shear or current magnitude.
    """
    if not cfg.get("KINETIC_EFFECTS_ENABLED", False):
        return pr

    model = str(cfg.get("KINETIC_MODEL", "shear")).lower()
    coeff = float(cfg.get("KINETIC_COEFF", 0.0))
    cap = float(cfg.get("KINETIC_CAP", 0.0))
    if coeff == 0.0:
        return pr

    gamma = float(cfg.get("GAMMA", 5.0/3.0))
    pmax = float(cfg.get("P_MAX", 1.0))
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]

    if model == "current" and pr.shape[0] >= 8:
        for i in range(ng, nx - ng):
            for j in range(ng, ny - ng):
                for k in range(ng, nz - ng):
                    dBz_dy = (pr[7, i, j+1, k] - pr[7, i, j-1, k]) / (2.0*dy)
                    dBy_dz = (pr[6, i, j, k+1] - pr[6, i, j, k-1]) / (2.0*dz)
                    dBx_dz = (pr[5, i, j, k+1] - pr[5, i, j, k-1]) / (2.0*dz)
                    dBz_dx = (pr[7, i+1, j, k] - pr[7, i-1, j, k]) / (2.0*dx)
                    dBy_dx = (pr[6, i+1, j, k] - pr[6, i-1, j, k]) / (2.0*dx)
                    dBx_dy = (pr[5, i, j+1, k] - pr[5, i, j-1, k]) / (2.0*dy)
                    jx = dBz_dy - dBy_dz
                    jy = dBx_dz - dBz_dx
                    jz = dBy_dx - dBx_dy
                    j2 = jx*jx + jy*jy + jz*jz
                    dp = (gamma - 1.0) * coeff * j2 * dt
                    if cap > 0.0 and dp > cap:
                        dp = cap
                    p = pr[4, i, j, k] + dp
                    if p < 1e-12:
                        p = 1e-12
                    if p > pmax:
                        p = pmax
                    pr[4, i, j, k] = p
        return pr

    # shear-based heating
    for i in range(ng, nx - ng):
        for j in range(ng, ny - ng):
            for k in range(ng, nz - ng):
                dvx_dx = (pr[1, i+1, j, k] - pr[1, i-1, j, k]) / (2.0*dx)
                dvy_dy = (pr[2, i, j+1, k] - pr[2, i, j-1, k]) / (2.0*dy)
                dvz_dz = (pr[3, i, j, k+1] - pr[3, i, j, k-1]) / (2.0*dz)
                shear2 = dvx_dx*dvx_dx + dvy_dy*dvy_dy + dvz_dz*dvz_dz
                dp = (gamma - 1.0) * coeff * shear2 * dt
                if cap > 0.0 and dp > cap:
                    dp = cap
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
