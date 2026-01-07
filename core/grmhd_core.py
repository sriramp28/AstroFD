#!/usr/bin/env python3
# core/grmhd_core.py
# GRMHD (fixed metric) with simple Schwarzschild source terms (approximate).
import numpy as np
from core import rmhd_core
from core import gr_metric

GR_METRIC = "minkowski"
GR_MASS = 1.0

def configure(params):
    global GR_METRIC, GR_MASS
    GR_METRIC = str(params.get("GR_METRIC", "minkowski")).lower()
    GR_MASS = float(params.get("GR_MASS", GR_MASS))

def compute_rhs_grmhd(pr, dx, dy, dz, offs_x, ng):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    if rmhd_core.RECON_ID == 1:
        rhs = rmhd_core.compute_rhs_ppm(pr, nx, ny, nz, dx, dy, dz)
    elif rmhd_core.RECON_ID == 2:
        rhs = rmhd_core.compute_rhs_weno(pr, nx, ny, nz, dx, dy, dz)
    else:
        rhs = rmhd_core.compute_rhs_rmhd(pr, nx, ny, nz, dx, dy, dz)

    if GR_METRIC == "minkowski":
        return rhs

    for i in range(ng, nx-ng):
        x = (offs_x + (i - ng) + 0.5) * dx
        for j in range(ng, ny-ng):
            y = (j - ng + 0.5) * dy
            for k in range(ng, nz-ng):
                z = (k - ng + 0.5) * dz
                _alpha, dlnadx, dlnady, dlnadz = gr_metric.schwarzschild_iso_lapse_and_grad(x, y, z, GR_MASS)

                rho, vx, vy, vz, p = pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k], pr[3,i,j,k], pr[4,i,j,k]
                Bx, By, Bz = pr[5,i,j,k], pr[6,i,j,k], pr[7,i,j,k]
                v2 = vx*vx + vy*vy + vz*vz
                if v2 >= 1.0:
                    v2 = 1.0 - 1e-14
                W = 1.0 / np.sqrt(1.0 - v2)
                h = 1.0 + rmhd_core.GAMMA/(rmhd_core.GAMMA-1.0) * p / max(rho, rmhd_core.SMALL)
                B2 = Bx*Bx + By*By + Bz*Bz
                w = rho * h * W * W + B2 + p
                _, Sx, Sy, Sz, _, _, _, _, _ = rmhd_core.prim_to_cons_rmhd(rho, vx, vy, vz, p, Bx, By, Bz, pr[8,i,j,k], rmhd_core.GAMMA)

                rhs[1,i,j,k] += w * dlnadx
                rhs[2,i,j,k] += w * dlnady
                rhs[3,i,j,k] += w * dlnadz
                rhs[4,i,j,k] += Sx*dlnadx + Sy*dlnady + Sz*dlnadz

    return rhs

def step_ssprk2(pr, dx, dy, dz, dt, offs_x, ng):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    U0 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                U0[:,i,j,k] = rmhd_core.prim_to_cons_rmhd(pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k],
                                                          pr[3,i,j,k], pr[4,i,j,k], pr[5,i,j,k],
                                                          pr[6,i,j,k], pr[7,i,j,k], pr[8,i,j,k], rmhd_core.GAMMA)

    rhs1 = compute_rhs_grmhd(pr, dx, dy, dz, offs_x, ng)
    U1   = U0 + dt*rhs1

    pr1 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr1[:,i,j,k] = rmhd_core.cons_to_prim_rmhd(U1[0,i,j,k], U1[1,i,j,k], U1[2,i,j,k],
                                                          U1[3,i,j,k], U1[4,i,j,k], U1[5,i,j,k],
                                                          U1[6,i,j,k], U1[7,i,j,k], U1[8,i,j,k])

    rhs2 = compute_rhs_grmhd(pr1, dx, dy, dz, offs_x, ng)
    U2   = 0.5*(U0 + U1 + dt*rhs2)

    out  = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[:,i,j,k] = rmhd_core.cons_to_prim_rmhd(U2[0,i,j,k], U2[1,i,j,k], U2[2,i,j,k],
                                                          U2[3,i,j,k], U2[4,i,j,k], U2[5,i,j,k],
                                                          U2[6,i,j,k], U2[7,i,j,k], U2[8,i,j,k])
    return out

def step_ssprk3(pr, dx, dy, dz, dt, offs_x, ng):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    U0 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                U0[:,i,j,k] = rmhd_core.prim_to_cons_rmhd(pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k],
                                                          pr[3,i,j,k], pr[4,i,j,k], pr[5,i,j,k],
                                                          pr[6,i,j,k], pr[7,i,j,k], pr[8,i,j,k], rmhd_core.GAMMA)

    rhs1 = compute_rhs_grmhd(pr, dx, dy, dz, offs_x, ng)
    U1 = U0 + dt*rhs1

    pr1 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr1[:,i,j,k] = rmhd_core.cons_to_prim_rmhd(U1[0,i,j,k], U1[1,i,j,k], U1[2,i,j,k],
                                                          U1[3,i,j,k], U1[4,i,j,k], U1[5,i,j,k],
                                                          U1[6,i,j,k], U1[7,i,j,k], U1[8,i,j,k])

    rhs2 = compute_rhs_grmhd(pr1, dx, dy, dz, offs_x, ng)
    U2 = 0.75*U0 + 0.25*(U1 + dt*rhs2)

    pr2 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr2[:,i,j,k] = rmhd_core.cons_to_prim_rmhd(U2[0,i,j,k], U2[1,i,j,k], U2[2,i,j,k],
                                                          U2[3,i,j,k], U2[4,i,j,k], U2[5,i,j,k],
                                                          U2[6,i,j,k], U2[7,i,j,k], U2[8,i,j,k])

    rhs3 = compute_rhs_grmhd(pr2, dx, dy, dz, offs_x, ng)
    U3 = (1.0/3.0)*U0 + (2.0/3.0)*(U2 + dt*rhs3)

    out  = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[:,i,j,k] = rmhd_core.cons_to_prim_rmhd(U3[0,i,j,k], U3[1,i,j,k], U3[2,i,j,k],
                                                          U3[3,i,j,k], U3[4,i,j,k], U3[5,i,j,k],
                                                          U3[6,i,j,k], U3[7,i,j,k], U3[8,i,j,k])
    return out
