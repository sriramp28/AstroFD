#!/usr/bin/env python3
# core/nozzle.py
import math

def nozzle_weight(r, radius, shear_thick, profile):
    """
    Return smooth weight s in [0,1] for nozzle profile.
    profile: "top_hat" (with shear), "taper", "parabolic".
    """
    if radius <= 0.0:
        return 0.0
    if profile in ("taper", "smooth"):
        if r >= radius:
            return 0.0
        return 0.5 * (1.0 + math.cos(math.pi * r / radius))
    if profile in ("parabolic", "parabola"):
        if r >= radius:
            return 0.0
        x = r / radius
        return max(0.0, 1.0 - x*x)

    # default: top-hat with shear layer
    if shear_thick > 0.0:
        return 0.5 * (1.0 - math.tanh((r - radius) / shear_thick))
    return 1.0 if r <= radius else 0.0

def apply_nozzle_left_x(pr, dx, dy, dz, ny_loc, nz_loc, y0, z0, rng, cfg):
    """
    Apply nozzle inflow on left ghost layers for SRHD.
    cfg keys: NG, JET_RADIUS, SHEAR_THICK, NOZZLE_PROFILE, GAMMA_JET, ETA_RHO, P_EQ,
              RHO_AMB, VX_AMB, VY_AMB, VZ_AMB,
              NOZZLE_PERTURB, TURB_VAMP, TURB_PAMP
    """
    ng = cfg["NG"]
    profile = cfg.get("NOZZLE_PROFILE", "top_hat")
    perturb = cfg.get("NOZZLE_PERTURB", "none")
    for g in range(ng):
        for j in range(ng, ng + ny_loc):
            y = (j - ng + 0.5) * dy
            for k in range(ng, ng + nz_loc):
                z = (k - ng + 0.5) * dz
                rr = math.sqrt((y - y0)**2 + (z - z0)**2)
                s = nozzle_weight(rr, cfg["JET_RADIUS"], cfg["SHEAR_THICK"], profile)
                rho = cfg["ETA_RHO"]*cfg["RHO_AMB"] * s + cfg["RHO_AMB"]*(1.0 - s)
                p   = cfg["P_EQ"]
                beta= math.sqrt(1.0 - 1.0/(cfg["GAMMA_JET"]*cfg["GAMMA_JET"]))
                vx  = beta * s + cfg["VX_AMB"]*(1.0 - s)
                vy  = cfg["VY_AMB"]
                vz  = cfg["VZ_AMB"]
                if perturb != "none" and s > 0.0:
                    vy += cfg["TURB_VAMP"] * (2.0*rng.random()-1.0)
                    vz += cfg["TURB_VAMP"] * (2.0*rng.random()-1.0)
                    p  *= (1.0 + cfg["TURB_PAMP"] * (2.0*rng.random()-1.0))
                pr[0, g, j, k] = rho
                pr[1, g, j, k] = vx
                pr[2, g, j, k] = vy
                pr[3, g, j, k] = vz
                pr[4, g, j, k] = p
                if pr.shape[0] >= 15:
                    pr[5:15, g, j, k] = 0.0
                ntr = int(cfg.get("N_TRACERS", 0))
                if ntr > 0:
                    off = int(cfg.get("TRACER_OFFSET", 5))
                    nozzle_vals = cfg.get("TRACER_NOZZLE_VALUES", [])
                    amb_vals = cfg.get("TRACER_AMB_VALUES", [])
                    for t in range(ntr):
                        t_noz = nozzle_vals[t] if t < len(nozzle_vals) else (1.0 if t == 0 else 0.0)
                        t_amb = amb_vals[t] if t < len(amb_vals) else 0.0
                        pr[off + t, g, j, k] = t_amb + (t_noz - t_amb) * s
                if cfg.get("TWO_TEMPERATURE", False):
                    toff = int(cfg.get("THERMO_OFFSET", 0))
                    if pr.shape[0] > toff + 1:
                        te = float(cfg.get("TE_NOZZLE", cfg.get("TE_AMB", 0.0)))
                        ti = float(cfg.get("TI_NOZZLE", cfg.get("TI_AMB", 0.0)))
                        pr[toff, g, j, k] = te
                        pr[toff+1, g, j, k] = ti
                if cfg.get("CHEMISTRY_ENABLED", False):
                    coff = int(cfg.get("CHEM_OFFSET", 0))
                    if pr.shape[0] > coff + 2:
                        pr[coff + 0, g, j, k] = float(cfg.get("CHEM_X_HII_NOZZLE", 0.0))
                        pr[coff + 1, g, j, k] = float(cfg.get("CHEM_X_HEII_NOZZLE", 0.0))
                        pr[coff + 2, g, j, k] = float(cfg.get("CHEM_X_HEIII_NOZZLE", 0.0))
                if cfg.get("PHYSICS") == "sn":
                    names = cfg.get("SN_COMP_NAMES", [])
                    off = int(cfg.get("SN_COMP_OFFSET", 5))
                    noz = cfg.get("SN_COMP_NOZZLE_VALUES", [])
                    amb = cfg.get("SN_COMP_AMB_VALUES", [])
                    for ci in range(len(names)):
                        v_noz = noz[ci] if ci < len(noz) else 0.0
                        v_amb = amb[ci] if ci < len(amb) else 0.0
                        pr[off + ci, g, j, k] = v_amb + (v_noz - v_amb) * s
                if cfg.get("PHYSICS") in ("rmhd", "grmhd"):
                    Bx = By = Bz = 0.0
                    if cfg.get("B_INIT") == "poloidal":
                        Bx = cfg.get("B0", 0.0) * s
                    elif cfg.get("B_INIT") == "toroidal":
                        By = cfg.get("B0", 0.0) * s
                    pr[5, g, j, k] = Bx
                    pr[6, g, j, k] = By
                    pr[7, g, j, k] = Bz
                    pr[8, g, j, k] = 0.0
