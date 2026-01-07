#!/usr/bin/env python3
# core/chemistry.py
# Simple H/He nonequilibrium ionization with IMEX relaxation and energy coupling.
import math

SMALL = 1e-20


def _clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _rate_ci_h(T):
    # Collisional ionization (H) - Cen 1992 style fit, T in K
    return 5.85e-11 * math.sqrt(T) * math.exp(-157809.1 / max(T, 1.0)) / (1.0 + math.sqrt(T / 1.0e5))


def _rate_ci_hei(T):
    return 2.38e-11 * math.sqrt(T) * math.exp(-285335.4 / max(T, 1.0)) / (1.0 + math.sqrt(T / 1.0e5))


def _rate_ci_heii(T):
    return 5.68e-12 * math.sqrt(T) * math.exp(-631515.0 / max(T, 1.0)) / (1.0 + math.sqrt(T / 1.0e5))


def _rate_rec_h(T):
    # Case B recombination, T in K
    return 2.59e-13 * (T / 1.0e4) ** -0.7


def _rate_rec_heii(T):
    return 1.50e-10 * (T / 1.0e4) ** -0.6353


def _rate_rec_heiii(T):
    return 3.36e-10 * (T / 1.0e4) ** -0.5


def apply_ion_chemistry(pr, dt, cfg):
    """
    Update H/He ion fractions with a simple IMEX step and couple to pressure.
    Species layout: [xHII, xHeII, xHeIII] stored in passive fields.
    """
    if not cfg.get("CHEMISTRY_ENABLED", False):
        return pr

    nchem = int(cfg.get("N_CHEM", 0))
    if nchem < 3:
        return pr

    off = int(cfg.get("CHEM_OFFSET", 0))
    if pr.shape[0] <= off + 2:
        return pr

    ng = int(cfg.get("NG", 2))
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]

    X = float(cfg.get("CHEM_X", 0.76))
    Y = float(cfg.get("CHEM_Y", 0.24))
    tunit = float(cfg.get("CHEM_TUNIT_K", 1.0e4))
    rate_scale = float(cfg.get("CHEM_RATE_SCALE", 1.0))
    energy_scale = float(cfg.get("CHEM_ENERGY_SCALE", 1.0))
    tmin = float(cfg.get("CHEM_TMIN_K", 1.0))
    gamma = float(cfg.get("GAMMA", 5.0/3.0))

    # Ionization potentials in K (E/kB): H=157809 K, HeI=285335 K, HeII=631515 K
    e_h = 157809.1
    e_he1 = 285335.4
    e_he2 = 631515.0

    for i in range(ng, nx - ng):
        for j in range(ng, ny - ng):
            for k in range(ng, nz - ng):
                rho = pr[0, i, j, k]
                p = pr[4, i, j, k]
                if rho <= 0.0 or p <= 0.0:
                    continue

                T = (p / max(rho, SMALL)) * tunit
                if T < tmin:
                    T = tmin

                # number densities (assuming rho ~ number density)
                nH = X * rho
                nHe = Y * rho / 4.0

                xHII = _clamp(pr[off + 0, i, j, k], 0.0, 1.0)
                xHeII = _clamp(pr[off + 1, i, j, k], 0.0, 1.0)
                xHeIII = _clamp(pr[off + 2, i, j, k], 0.0, 1.0)

                xHeII = min(xHeII, 1.0 - xHeIII)
                xHeI = max(0.0, 1.0 - xHeII - xHeIII)
                xHI = 1.0 - xHII

                ne = nH * xHII + nHe * (xHeII + 2.0 * xHeIII)
                ne = max(ne, SMALL)

                C_H = rate_scale * _rate_ci_h(T)
                C_HeI = rate_scale * _rate_ci_hei(T)
                C_HeII = rate_scale * _rate_ci_heii(T)
                A_H = rate_scale * _rate_rec_h(T)
                A_HeII = rate_scale * _rate_rec_heii(T)
                A_HeIII = rate_scale * _rate_rec_heiii(T)

                # IMEX updates (explicit ionization, implicit recombination)
                xHII_new = (xHII + dt * ne * C_H * xHI) / (1.0 + dt * ne * A_H)

                xHeIII_new = (xHeIII + dt * ne * C_HeII * xHeII) / (1.0 + dt * ne * A_HeIII)
                xHeIII_new = _clamp(xHeIII_new, 0.0, 1.0)

                xHeII_new = (xHeII + dt * ne * (C_HeI * xHeI + A_HeIII * xHeIII_new)) / (1.0 + dt * ne * (A_HeII + C_HeII))

                xHII_new = _clamp(xHII_new, 0.0, 1.0)
                xHeII_new = _clamp(xHeII_new, 0.0, 1.0 - xHeIII_new)

                # energy coupling to pressure
                dH = (xHII_new - xHII)
                dHeII = (xHeII_new - xHeII)
                dHeIII = (xHeIII_new - xHeIII)
                dE = energy_scale * (
                    nH * e_h * dH +
                    nHe * e_he1 * dHeII +
                    nHe * e_he2 * dHeIII
                )
                dp = -(gamma - 1.0) * dE / max(tunit, SMALL)
                p_new = p + dp
                if p_new < SMALL:
                    p_new = SMALL

                pr[4, i, j, k] = p_new
                pr[off + 0, i, j, k] = xHII_new
                pr[off + 1, i, j, k] = xHeII_new
                pr[off + 2, i, j, k] = xHeIII_new

    return pr
