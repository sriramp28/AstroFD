#!/usr/bin/env python3
# core/gr_metric.py
# Simple Schwarzschild isotropic metric helpers.
import math

def schwarzschild_iso_lapse_and_grad(x, y, z, M):
    r = math.sqrt(x*x + y*y + z*z) + 1e-12
    a = M / (2.0 * r)
    alpha = (1.0 - a) / (1.0 + a)
    # d ln alpha / dr = 2a / (r*(1 - a^2))
    denom = (1.0 - a*a)
    dlnadr = 0.0 if denom == 0.0 else (2.0 * a) / (r * denom)
    dlnadx = dlnadr * (x / r)
    dlnady = dlnadr * (y / r)
    dlnadz = dlnadr * (z / r)
    return alpha, dlnadx, dlnady, dlnadz

def schwarzschild_ks_lapse_shift_and_grad(x, y, z, M):
    # Kerr-Schild (a=0) in Cartesian coordinates.
    r = math.sqrt(x*x + y*y + z*z) + 1e-12
    H = M / r
    alpha = 1.0 / math.sqrt(1.0 + 2.0*H)
    fac = 2.0*H / (1.0 + 2.0*H)
    lx = x / r; ly = y / r; lz = z / r
    beta_x = fac * lx
    beta_y = fac * ly
    beta_z = fac * lz

    # d ln alpha / dr
    dlnadr = H / (r * (1.0 + 2.0*H))
    dlnadx = dlnadr * lx
    dlnady = dlnadr * ly
    dlnadz = dlnadr * lz
    return alpha, beta_x, beta_y, beta_z, dlnadx, dlnady, dlnadz

def _kerr_schild_alpha_beta(x, y, z, M, a):
    x2y2 = x*x + y*y
    z2 = z*z
    a2 = a*a
    tmp = x2y2 + z2 - a2
    r2 = 0.5 * (tmp + math.sqrt(tmp*tmp + 4.0*a2*z2))
    if r2 < 1e-12:
        r2 = 1e-12
    r = math.sqrt(r2)
    denom = r2 + a2
    if denom < 1e-12:
        denom = 1e-12

    H = M * r*r*r / (r2*r2 + a2*z2 + 1e-12)
    alpha = 1.0 / math.sqrt(1.0 + 2.0*H)
    fac = 2.0*H / (1.0 + 2.0*H)
    lx = (r*x + a*y) / denom
    ly = (r*y - a*x) / denom
    lz = z / r
    beta_x = fac * lx
    beta_y = fac * ly
    beta_z = fac * lz
    return alpha, beta_x, beta_y, beta_z

def kerr_schild_lapse_shift_and_grad(x, y, z, M, a):
    # Kerr-Schild lapse/shift with spin; gradients from finite differences.
    alpha, beta_x, beta_y, beta_z = _kerr_schild_alpha_beta(x, y, z, M, a)
    r = math.sqrt(x*x + y*y + z*z) + 1e-12
    eps = 1e-5 * max(1.0, r)

    ap, _, _, _ = _kerr_schild_alpha_beta(x + eps, y, z, M, a)
    am, _, _, _ = _kerr_schild_alpha_beta(x - eps, y, z, M, a)
    bp, _, _, _ = _kerr_schild_alpha_beta(x, y + eps, z, M, a)
    bm, _, _, _ = _kerr_schild_alpha_beta(x, y - eps, z, M, a)
    cp, _, _, _ = _kerr_schild_alpha_beta(x, y, z + eps, M, a)
    cm, _, _, _ = _kerr_schild_alpha_beta(x, y, z - eps, M, a)

    dlnadx = (math.log(ap) - math.log(am)) / (2.0 * eps)
    dlnady = (math.log(bp) - math.log(bm)) / (2.0 * eps)
    dlnadz = (math.log(cp) - math.log(cm)) / (2.0 * eps)
    return alpha, beta_x, beta_y, beta_z, dlnadx, dlnady, dlnadz
