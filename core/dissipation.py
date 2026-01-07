#!/usr/bin/env python3
# core/dissipation.py
# Causal dissipation (Israel-Stewart) with IMEX relaxation for bulk/shear/heat.

def apply_causal_dissipation(pr, dt, dx, dy, dz, cfg):
    """
    Apply causal dissipation updates with relaxation time.
    This uses a local IMEX step toward Navier-Stokes targets.
    """
    if not cfg.get("DISSIPATION_ENABLED", False):
        return pr
    if cfg.get("PHYSICS") not in ("hydro", "grhd"):
        return pr

    ng = cfg.get("NG", 2)
    zeta = float(cfg.get("BULK_ZETA", 0.0))
    eta = float(cfg.get("SHEAR_ETA", 0.0))
    kappa = float(cfg.get("HEAT_KAPPA", 0.0))
    tau_bulk = float(cfg.get("RELAX_TAU_BULK", cfg.get("RELAX_TAU", 0.1)) or cfg.get("RELAX_TAU", 0.1))
    tau_shear = float(cfg.get("RELAX_TAU_SHEAR", cfg.get("RELAX_TAU", 0.1)) or cfg.get("RELAX_TAU", 0.1))
    tau_heat = float(cfg.get("RELAX_TAU_HEAT", cfg.get("RELAX_TAU", 0.1)) or cfg.get("RELAX_TAU", 0.1))
    if (zeta == 0.0 and eta == 0.0 and kappa == 0.0) or min(tau_bulk, tau_shear, tau_heat) <= 0.0:
        return pr

    # Dissipation layout: [pi, pixx, piyy, pizz, pixy, pixz, piyz, qx, qy, qz]
    if pr.shape[0] < 15:
        return pr
    pi_idx = 5
    pixx_idx = 6
    piyy_idx = 7
    pizz_idx = 8
    pixy_idx = 9
    pixz_idx = 10
    piyz_idx = 11
    qx_idx = 12
    qy_idx = 13
    qz_idx = 14

    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    inv2dx = 0.5 / max(dx, 1e-12)
    inv2dy = 0.5 / max(dy, 1e-12)
    inv2dz = 0.5 / max(dz, 1e-12)
    cap_frac = float(cfg.get("DISSIPATION_CAP_FRAC", 0.5))
    pmax = float(cfg.get("P_MAX", 1.0))
    advect = bool(cfg.get("DISSIPATION_ADVECT", True))

    if advect:
        # First-order upwind advection for dissipative variables (explicit step).
        pi_adv = pr[pi_idx].copy()
        pixx_adv = pr[pixx_idx].copy()
        piyy_adv = pr[piyy_idx].copy()
        pizz_adv = pr[pizz_idx].copy()
        pixy_adv = pr[pixy_idx].copy()
        pixz_adv = pr[pixz_idx].copy()
        piyz_adv = pr[piyz_idx].copy()
        qx_adv = pr[qx_idx].copy()
        qy_adv = pr[qy_idx].copy()
        qz_adv = pr[qz_idx].copy()

        for i in range(ng, nx-ng):
            for j in range(ng, ny-ng):
                for k in range(ng, nz-ng):
                    vx = pr[1, i, j, k]
                    vy = pr[2, i, j, k]
                    vz = pr[3, i, j, k]

                    for arr_adv, arr in (
                        (pi_adv, pr[pi_idx]),
                        (pixx_adv, pr[pixx_idx]),
                        (piyy_adv, pr[piyy_idx]),
                        (pizz_adv, pr[pizz_idx]),
                        (pixy_adv, pr[pixy_idx]),
                        (pixz_adv, pr[pixz_idx]),
                        (piyz_adv, pr[piyz_idx]),
                        (qx_adv, pr[qx_idx]),
                        (qy_adv, pr[qy_idx]),
                        (qz_adv, pr[qz_idx]),
                    ):
                        u0 = arr[i, j, k]
                        if vx > 0.0:
                            dudx = (u0 - arr[i-1, j, k]) / max(dx, 1e-12)
                        else:
                            dudx = (arr[i+1, j, k] - u0) / max(dx, 1e-12)
                        if vy > 0.0:
                            dudy = (u0 - arr[i, j-1, k]) / max(dy, 1e-12)
                        else:
                            dudy = (arr[i, j+1, k] - u0) / max(dy, 1e-12)
                        if vz > 0.0:
                            dudz = (u0 - arr[i, j, k-1]) / max(dz, 1e-12)
                        else:
                            dudz = (arr[i, j, k+1] - u0) / max(dz, 1e-12)
                        arr_adv[i, j, k] = u0 - dt * (vx*dudx + vy*dudy + vz*dudz)

        pr[pi_idx] = pi_adv
        pr[pixx_idx] = pixx_adv
        pr[piyy_idx] = piyy_adv
        pr[pizz_idx] = pizz_adv
        pr[pixy_idx] = pixy_adv
        pr[pixz_idx] = pixz_adv
        pr[piyz_idx] = piyz_adv
        pr[qx_idx] = qx_adv
        pr[qy_idx] = qy_adv
        pr[qz_idx] = qz_adv

    for i in range(ng, nx-ng):
        for j in range(ng, ny-ng):
            for k in range(ng, nz-ng):
                dvx_dx = (pr[1, i+1, j, k] - pr[1, i-1, j, k]) * inv2dx
                dvx_dy = (pr[1, i, j+1, k] - pr[1, i, j-1, k]) * inv2dy
                dvx_dz = (pr[1, i, j, k+1] - pr[1, i, j, k-1]) * inv2dz

                dvy_dx = (pr[2, i+1, j, k] - pr[2, i-1, j, k]) * inv2dx
                dvy_dy = (pr[2, i, j+1, k] - pr[2, i, j-1, k]) * inv2dy
                dvy_dz = (pr[2, i, j, k+1] - pr[2, i, j, k-1]) * inv2dz

                dvz_dx = (pr[3, i+1, j, k] - pr[3, i-1, j, k]) * inv2dx
                dvz_dy = (pr[3, i, j+1, k] - pr[3, i, j-1, k]) * inv2dy
                dvz_dz = (pr[3, i, j, k+1] - pr[3, i, j, k-1]) * inv2dz

                divv = dvx_dx + dvy_dy + dvz_dz

                # Bulk pressure
                if zeta != 0.0:
                    pi_eq = -zeta * divv
                    pi_old = pr[pi_idx, i, j, k]
                    pi_new = (pi_old + dt * pi_eq / tau_bulk) / (1.0 + dt / tau_bulk)
                    cap = cap_frac * pmax
                    if pi_new >  cap: pi_new =  cap
                    if pi_new < -cap: pi_new = -cap
                    pr[pi_idx, i, j, k] = pi_new

                # Shear stress (symmetric, traceless)
                if eta != 0.0:
                    sxx = dvx_dx - (1.0/3.0)*divv
                    syy = dvy_dy - (1.0/3.0)*divv
                    szz = dvz_dz - (1.0/3.0)*divv
                    sxy = 0.5*(dvx_dy + dvy_dx)
                    sxz = 0.5*(dvx_dz + dvz_dx)
                    syz = 0.5*(dvy_dz + dvz_dy)

                    pixx_eq = -2.0*eta*sxx
                    piyy_eq = -2.0*eta*syy
                    pizz_eq = -2.0*eta*szz
                    pixy_eq = -2.0*eta*sxy
                    pixz_eq = -2.0*eta*sxz
                    piyz_eq = -2.0*eta*syz

                    pixx = (pr[pixx_idx, i, j, k] + dt * pixx_eq / tau_shear) / (1.0 + dt / tau_shear)
                    piyy = (pr[piyy_idx, i, j, k] + dt * piyy_eq / tau_shear) / (1.0 + dt / tau_shear)
                    pizz = (pr[pizz_idx, i, j, k] + dt * pizz_eq / tau_shear) / (1.0 + dt / tau_shear)
                    pixy = (pr[pixy_idx, i, j, k] + dt * pixy_eq / tau_shear) / (1.0 + dt / tau_shear)
                    pixz = (pr[pixz_idx, i, j, k] + dt * pixz_eq / tau_shear) / (1.0 + dt / tau_shear)
                    piyz = (pr[piyz_idx, i, j, k] + dt * piyz_eq / tau_shear) / (1.0 + dt / tau_shear)

                    # enforce tracelessness
                    pizz = -(pixx + piyy)

                    cap = cap_frac * pmax
                    if pixx >  cap: pixx =  cap
                    if pixx < -cap: pixx = -cap
                    if piyy >  cap: piyy =  cap
                    if piyy < -cap: piyy = -cap
                    if pizz >  cap: pizz =  cap
                    if pizz < -cap: pizz = -cap
                    if pixy >  cap: pixy =  cap
                    if pixy < -cap: pixy = -cap
                    if pixz >  cap: pixz =  cap
                    if pixz < -cap: pixz = -cap
                    if piyz >  cap: piyz =  cap
                    if piyz < -cap: piyz = -cap

                    pr[pixx_idx, i, j, k] = pixx
                    pr[piyy_idx, i, j, k] = piyy
                    pr[pizz_idx, i, j, k] = pizz
                    pr[pixy_idx, i, j, k] = pixy
                    pr[pixz_idx, i, j, k] = pixz
                    pr[piyz_idx, i, j, k] = piyz

                # Heat flux (temperature ~ p/rho)
                if kappa != 0.0:
                    rho = pr[0, i, j, k]
                    p = pr[4, i, j, k]
                    tloc = p / max(rho, 1e-12)
                    txp = pr[4, i+1, j, k] / max(pr[0, i+1, j, k], 1e-12)
                    txm = pr[4, i-1, j, k] / max(pr[0, i-1, j, k], 1e-12)
                    typ = pr[4, i, j+1, k] / max(pr[0, i, j+1, k], 1e-12)
                    tym = pr[4, i, j-1, k] / max(pr[0, i, j-1, k], 1e-12)
                    tzp = pr[4, i, j, k+1] / max(pr[0, i, j, k+1], 1e-12)
                    tzm = pr[4, i, j, k-1] / max(pr[0, i, j, k-1], 1e-12)

                    dTdx = (txp - txm) * inv2dx
                    dTdy = (typ - tym) * inv2dy
                    dTdz = (tzp - tzm) * inv2dz

                    qx_eq = -kappa * dTdx
                    qy_eq = -kappa * dTdy
                    qz_eq = -kappa * dTdz

                    qx = (pr[qx_idx, i, j, k] + dt * qx_eq / tau_heat) / (1.0 + dt / tau_heat)
                    qy = (pr[qy_idx, i, j, k] + dt * qy_eq / tau_heat) / (1.0 + dt / tau_heat)
                    qz = (pr[qz_idx, i, j, k] + dt * qz_eq / tau_heat) / (1.0 + dt / tau_heat)

                    cap = cap_frac * pmax
                    if qx >  cap: qx =  cap
                    if qx < -cap: qx = -cap
                    if qy >  cap: qy =  cap
                    if qy < -cap: qy = -cap
                    if qz >  cap: qz =  cap
                    if qz < -cap: qz = -cap

                    pr[qx_idx, i, j, k] = qx
                    pr[qy_idx, i, j, k] = qy
                    pr[qz_idx, i, j, k] = qz

    return pr
