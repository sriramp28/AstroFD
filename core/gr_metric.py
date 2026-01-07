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

def kerr_schild_lapse_shift_and_grad(x, y, z, M, a):
    # Approximate Kerr-Schild; for now treat as Schwarzschild KS if spin is small.
    return schwarzschild_ks_lapse_shift_and_grad(x, y, z, M)
