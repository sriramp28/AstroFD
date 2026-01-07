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
