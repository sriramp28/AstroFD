#!/usr/bin/env python3
# core/gravity.py
# Simple Newtonian gravity source terms for SN-lite.
import math


def apply_gravity(pr, dt, dx, dy, dz, cfg, offs_x, ng):
    if not cfg.get("SN_GRAVITY_ENABLED", False):
        return pr
    if cfg.get("PHYSICS") not in ("sn",):
        return pr

    gconst = float(cfg.get("SN_GRAVITY_G", 1.0))
    mass = float(cfg.get("SN_GRAVITY_MASS", 1.0))
    soften = float(cfg.get("SN_GRAVITY_SOFTEN", 0.0))
    center = cfg.get("SN_GRAVITY_CENTER", [0.0, 0.0, 0.0])
    x0, y0, z0 = float(center[0]), float(center[1]), float(center[2])
    v_max = float(cfg.get("V_MAX", 0.999))
    p_max = float(cfg.get("P_MAX", 1.0))
    gamma = float(cfg.get("GAMMA", 5.0/3.0))
    energy_couple = bool(cfg.get("SN_GRAVITY_ENERGY", False))

    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    eps2 = soften * soften

    for i in range(ng, nx - ng):
        x = (offs_x + (i - ng) + 0.5) * dx
        for j in range(ng, ny - ng):
            y = (j - ng + 0.5) * dy
            for k in range(ng, nz - ng):
                z = (k - ng + 0.5) * dz
                rx = x - x0
                ry = y - y0
                rz = z - z0
                r2 = rx*rx + ry*ry + rz*rz + eps2
                r = math.sqrt(r2)
                if r == 0.0:
                    continue
                fac = -gconst * mass / (r2 * r)
                gx = fac * rx
                gy = fac * ry
                gz = fac * rz

                rho = pr[0, i, j, k]
                vx = pr[1, i, j, k]
                vy = pr[2, i, j, k]
                vz = pr[3, i, j, k]

                if energy_couple:
                    work = rho * (vx*gx + vy*gy + vz*gz) * dt
                    dp = (gamma - 1.0) * work
                    p = pr[4, i, j, k] + dp
                    if p < 1e-12:
                        p = 1e-12
                    if p > p_max:
                        p = p_max
                    pr[4, i, j, k] = p

                vx += dt * gx
                vy += dt * gy
                vz += dt * gz
                v2 = vx*vx + vy*vy + vz*vz
                vmax2 = v_max * v_max
                if v2 >= vmax2:
                    facv = v_max / math.sqrt(v2 + 1e-32)
                    vx *= facv; vy *= facv; vz *= facv

                pr[1, i, j, k] = vx
                pr[2, i, j, k] = vy
                pr[3, i, j, k] = vz

    return pr
