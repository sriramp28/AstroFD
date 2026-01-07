#!/usr/bin/env python3
# core/plasma_microphysics.py
# Non-ideal MHD microphysics: Hall + ambipolar + hyper-resistive corrections.
import numpy as np

SMALL = 1e-20


def apply_nonideal_mhd(pr, dt, dx, dy, dz, cfg, ng):
    if not cfg.get("NONIDEAL_MHD_ENABLED", False):
        return pr
    if cfg.get("PHYSICS") not in ("rmhd", "grmhd"):
        return pr
    if pr.shape[0] < 8:
        return pr

    hall = bool(cfg.get("HALL_ENABLED", False))
    ambi = bool(cfg.get("AMBIPOLAR_ENABLED", False))
    hyper = bool(cfg.get("HYPERRESIST_ENABLED", False))
    eta_hall = float(cfg.get("HALL_COEFF", 0.0))
    eta_ambi = float(cfg.get("AMBIPOLAR_COEFF", 0.0))
    eta_hyper = float(cfg.get("HYPERRESIST_COEFF", 0.0))
    joule = bool(cfg.get("JOULE_HEAT_ENABLED", False))
    joule_eff = float(cfg.get("JOULE_HEAT_EFF", 0.0))
    gamma = float(cfg.get("GAMMA", 5.0/3.0))
    pmax = float(cfg.get("P_MAX", 1.0))

    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]

    # current J = curl B (centered)
    Jx = np.zeros((nx, ny, nz))
    Jy = np.zeros((nx, ny, nz))
    Jz = np.zeros((nx, ny, nz))
    for i in range(ng, nx - ng):
        for j in range(ng, ny - ng):
            for k in range(ng, nz - ng):
                dBz_dy = (pr[7, i, j+1, k] - pr[7, i, j-1, k]) / (2.0*dy)
                dBy_dz = (pr[6, i, j, k+1] - pr[6, i, j, k-1]) / (2.0*dz)
                dBx_dz = (pr[5, i, j, k+1] - pr[5, i, j, k-1]) / (2.0*dz)
                dBz_dx = (pr[7, i+1, j, k] - pr[7, i-1, j, k]) / (2.0*dx)
                dBy_dx = (pr[6, i+1, j, k] - pr[6, i-1, j, k]) / (2.0*dx)
                dBx_dy = (pr[5, i, j+1, k] - pr[5, i, j-1, k]) / (2.0*dy)
                Jx[i, j, k] = dBz_dy - dBy_dz
                Jy[i, j, k] = dBx_dz - dBz_dx
                Jz[i, j, k] = dBy_dx - dBx_dy

    # E = E_hall + E_ambi + E_hyper
    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    Ez = np.zeros((nx, ny, nz))

    if hall or ambi:
        for i in range(ng, nx - ng):
            for j in range(ng, ny - ng):
                for k in range(ng, nz - ng):
                    rho = pr[0, i, j, k]
                    inv_rho = 1.0 / max(rho, SMALL)
                    Bx = pr[5, i, j, k]
                    By = pr[6, i, j, k]
                    Bz = pr[7, i, j, k]
                    Jx0 = Jx[i, j, k]
                    Jy0 = Jy[i, j, k]
                    Jz0 = Jz[i, j, k]
                    if hall and eta_hall != 0.0:
                        # Hall: J x B
                        hx = Jy0*Bz - Jz0*By
                        hy = Jz0*Bx - Jx0*Bz
                        hz = Jx0*By - Jy0*Bx
                        Ex[i, j, k] += eta_hall * hx * inv_rho
                        Ey[i, j, k] += eta_hall * hy * inv_rho
                        Ez[i, j, k] += eta_hall * hz * inv_rho
                    if ambi and eta_ambi != 0.0:
                        # Ambipolar: (J x B) x B
                        cx = Jy0*Bz - Jz0*By
                        cy = Jz0*Bx - Jx0*Bz
                        cz = Jx0*By - Jy0*Bx
                        ax = cy*Bz - cz*By
                        ay = cz*Bx - cx*Bz
                        az = cx*By - cy*Bx
                        Ex[i, j, k] += eta_ambi * ax * inv_rho
                        Ey[i, j, k] += eta_ambi * ay * inv_rho
                        Ez[i, j, k] += eta_ambi * az * inv_rho

    if hyper and eta_hyper != 0.0:
        # Hyper-resistive term: E_hyper = -eta4 * curl(J) ~ -eta4 * laplacian(B)
        invdx2 = 1.0 / (dx*dx + SMALL)
        invdy2 = 1.0 / (dy*dy + SMALL)
        invdz2 = 1.0 / (dz*dz + SMALL)
        for i in range(ng, nx - ng):
            for j in range(ng, ny - ng):
                for k in range(ng, nz - ng):
                    lapBx = (
                        (pr[5, i+1, j, k] - 2.0*pr[5, i, j, k] + pr[5, i-1, j, k]) * invdx2 +
                        (pr[5, i, j+1, k] - 2.0*pr[5, i, j, k] + pr[5, i, j-1, k]) * invdy2 +
                        (pr[5, i, j, k+1] - 2.0*pr[5, i, j, k] + pr[5, i, j, k-1]) * invdz2
                    )
                    lapBy = (
                        (pr[6, i+1, j, k] - 2.0*pr[6, i, j, k] + pr[6, i-1, j, k]) * invdx2 +
                        (pr[6, i, j+1, k] - 2.0*pr[6, i, j, k] + pr[6, i, j-1, k]) * invdy2 +
                        (pr[6, i, j, k+1] - 2.0*pr[6, i, j, k] + pr[6, i, j, k-1]) * invdz2
                    )
                    lapBz = (
                        (pr[7, i+1, j, k] - 2.0*pr[7, i, j, k] + pr[7, i-1, j, k]) * invdx2 +
                        (pr[7, i, j+1, k] - 2.0*pr[7, i, j, k] + pr[7, i, j-1, k]) * invdy2 +
                        (pr[7, i, j, k+1] - 2.0*pr[7, i, j, k] + pr[7, i, j, k-1]) * invdz2
                    )
                    Ex[i, j, k] += -eta_hyper * lapBx
                    Ey[i, j, k] += -eta_hyper * lapBy
                    Ez[i, j, k] += -eta_hyper * lapBz

    # Update B: dB/dt = -curl(E)
    for i in range(ng, nx - ng):
        for j in range(ng, ny - ng):
            for k in range(ng, nz - ng):
                dEz_dy = (Ez[i, j+1, k] - Ez[i, j-1, k]) / (2.0*dy)
                dEy_dz = (Ey[i, j, k+1] - Ey[i, j, k-1]) / (2.0*dz)
                dEx_dz = (Ex[i, j, k+1] - Ex[i, j, k-1]) / (2.0*dz)
                dEz_dx = (Ez[i+1, j, k] - Ez[i-1, j, k]) / (2.0*dx)
                dEy_dx = (Ey[i+1, j, k] - Ey[i-1, j, k]) / (2.0*dx)
                dEx_dy = (Ex[i, j+1, k] - Ex[i, j-1, k]) / (2.0*dy)

                pr[5, i, j, k] += -dt * (dEz_dy - dEy_dz)
                pr[6, i, j, k] += -dt * (dEx_dz - dEz_dx)
                pr[7, i, j, k] += -dt * (dEy_dx - dEx_dy)

    if joule and joule_eff > 0.0:
        for i in range(ng, nx - ng):
            for j in range(ng, ny - ng):
                for k in range(ng, nz - ng):
                    j2 = Jx[i, j, k]**2 + Jy[i, j, k]**2 + Jz[i, j, k]**2
                    dp = (gamma - 1.0) * joule_eff * j2 * dt
                    p = pr[4, i, j, k] + dp
                    if p < 1e-12:
                        p = 1e-12
                    if p > pmax:
                        p = pmax
                    pr[4, i, j, k] = p

    return pr
