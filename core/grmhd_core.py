#!/usr/bin/env python3
# core/grmhd_core.py
# GRMHD (fixed metric) with simple Schwarzschild source terms (approximate).
import numpy as np
from core import rmhd_core
from core import eos
from core import gr_metric

GR_METRIC = "minkowski"
GR_MASS = 1.0
ORTHONORMAL_FLUX = True
GR_SPIN = 0.0

def configure(params):
    global GR_METRIC, GR_MASS, ORTHONORMAL_FLUX, GR_SPIN
    GR_METRIC = str(params.get("GR_METRIC", "minkowski")).lower()
    GR_MASS = float(params.get("GR_MASS", GR_MASS))
    GR_SPIN = float(params.get("GR_SPIN", GR_SPIN))
    ORTHONORMAL_FLUX = bool(params.get("ORTHONORMAL_FLUX", ORTHONORMAL_FLUX))

def compute_rhs_grmhd(pr, dx, dy, dz, offs_x, ng):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    pr_use = pr
    if ORTHONORMAL_FLUX and GR_METRIC in ("schwarzschild", "kerr-schild"):
        pr_use = pr.copy()
        for i in range(ng, nx-ng):
            x = (offs_x + (i - ng) + 0.5) * dx
            for j in range(ng, ny-ng):
                y = (j - ng + 0.5) * dy
                for k in range(ng, nz-ng):
                    z = (k - ng + 0.5) * dz
                    if GR_METRIC == "schwarzschild":
                        alpha, _bx, _by, _bz, _dax, _day, _daz = gr_metric.schwarzschild_ks_lapse_shift_and_grad(x, y, z, GR_MASS)
                    else:
                        alpha, _bx, _by, _bz, _dax, _day, _daz = gr_metric.kerr_schild_lapse_shift_and_grad(x, y, z, GR_MASS, GR_SPIN)
                    ax = max(alpha, 1e-12)
                    pr_use[1, i, j, k] = pr[1, i, j, k] / ax
                    pr_use[2, i, j, k] = pr[2, i, j, k] / ax
                    pr_use[3, i, j, k] = pr[3, i, j, k] / ax

    if rmhd_core.RECON_ID == 1:
        rhs = rmhd_core.compute_rhs_ppm(pr_use, nx, ny, nz, dx, dy, dz)
    elif rmhd_core.RECON_ID == 2:
        rhs = rmhd_core.compute_rhs_weno(pr_use, nx, ny, nz, dx, dy, dz)
    else:
        rhs = rmhd_core.compute_rhs_rmhd(pr_use, nx, ny, nz, dx, dy, dz)

    if GR_METRIC == "minkowski":
        return rhs

    for i in range(ng, nx-ng):
        x = (offs_x + (i - ng) + 0.5) * dx
        for j in range(ng, ny-ng):
            y = (j - ng + 0.5) * dy
            for k in range(ng, nz-ng):
                z = (k - ng + 0.5) * dz
                if GR_METRIC == "schwarzschild":
                    _alpha, dlnadx, dlnady, dlnadz = gr_metric.schwarzschild_iso_lapse_and_grad(x, y, z, GR_MASS)
                else:
                    _alpha, _bx, _by, _bz, dlnadx, dlnady, dlnadz = gr_metric.kerr_schild_lapse_shift_and_grad(x, y, z, GR_MASS, GR_SPIN)

                rho, vx, vy, vz, p = pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k], pr[3,i,j,k], pr[4,i,j,k]
                Bx, By, Bz = pr[5,i,j,k], pr[6,i,j,k], pr[7,i,j,k]
                v2 = vx*vx + vy*vy + vz*vz
                if v2 >= 1.0:
                    v2 = 1.0 - 1e-14
                W = 1.0 / np.sqrt(1.0 - v2)
                h = eos.enthalpy(rho, p)
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
    npass = rmhd_core.N_PASSIVE
    U0 = np.zeros((9, nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                U0[:,i,j,k] = rmhd_core.prim_to_cons_rmhd(
                    pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k], pr[3,i,j,k], pr[4,i,j,k],
                    pr[5,i,j,k], pr[6,i,j,k], pr[7,i,j,k], pr[8,i,j,k], rmhd_core.GAMMA
                )

    rhs1 = compute_rhs_grmhd(pr, dx, dy, dz, offs_x, ng)
    U1   = U0 + dt*rhs1[0:9]

    pr1 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr1[0:9,i,j,k] = rmhd_core.cons_to_prim_rmhd(
                    U1[0,i,j,k], U1[1,i,j,k], U1[2,i,j,k], U1[3,i,j,k], U1[4,i,j,k],
                    U1[5,i,j,k], U1[6,i,j,k], U1[7,i,j,k], U1[8,i,j,k]
                )
    if npass > 0:
        pr1[rmhd_core.PASSIVE_OFFSET:rmhd_core.PASSIVE_OFFSET+npass] = (
            pr[rmhd_core.PASSIVE_OFFSET:rmhd_core.PASSIVE_OFFSET+npass] + dt*rhs1[9:9+npass]
        )

    rhs2 = compute_rhs_grmhd(pr1, dx, dy, dz, offs_x, ng)
    U2   = 0.5*(U0 + U1 + dt*rhs2[0:9])

    out  = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[0:9,i,j,k] = rmhd_core.cons_to_prim_rmhd(
                    U2[0,i,j,k], U2[1,i,j,k], U2[2,i,j,k], U2[3,i,j,k], U2[4,i,j,k],
                    U2[5,i,j,k], U2[6,i,j,k], U2[7,i,j,k], U2[8,i,j,k]
                )
    if npass > 0:
        t0 = pr[rmhd_core.PASSIVE_OFFSET:rmhd_core.PASSIVE_OFFSET+npass]
        t1 = pr1[rmhd_core.PASSIVE_OFFSET:rmhd_core.PASSIVE_OFFSET+npass]
        out[rmhd_core.PASSIVE_OFFSET:rmhd_core.PASSIVE_OFFSET+npass] = 0.5*(t0 + t1 + dt*rhs2[9:9+npass])
    return out

def step_ssprk3(pr, dx, dy, dz, dt, offs_x, ng):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    npass = rmhd_core.N_PASSIVE
    U0 = np.zeros((9, nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                U0[:,i,j,k] = rmhd_core.prim_to_cons_rmhd(
                    pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k], pr[3,i,j,k], pr[4,i,j,k],
                    pr[5,i,j,k], pr[6,i,j,k], pr[7,i,j,k], pr[8,i,j,k], rmhd_core.GAMMA
                )

    rhs1 = compute_rhs_grmhd(pr, dx, dy, dz, offs_x, ng)
    U1 = U0 + dt*rhs1[0:9]

    pr1 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr1[0:9,i,j,k] = rmhd_core.cons_to_prim_rmhd(
                    U1[0,i,j,k], U1[1,i,j,k], U1[2,i,j,k], U1[3,i,j,k], U1[4,i,j,k],
                    U1[5,i,j,k], U1[6,i,j,k], U1[7,i,j,k], U1[8,i,j,k]
                )
    if npass > 0:
        pr1[rmhd_core.PASSIVE_OFFSET:rmhd_core.PASSIVE_OFFSET+npass] = (
            pr[rmhd_core.PASSIVE_OFFSET:rmhd_core.PASSIVE_OFFSET+npass] + dt*rhs1[9:9+npass]
        )

    rhs2 = compute_rhs_grmhd(pr1, dx, dy, dz, offs_x, ng)
    U2 = 0.75*U0 + 0.25*(U1 + dt*rhs2[0:9])

    pr2 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr2[0:9,i,j,k] = rmhd_core.cons_to_prim_rmhd(
                    U2[0,i,j,k], U2[1,i,j,k], U2[2,i,j,k], U2[3,i,j,k], U2[4,i,j,k],
                    U2[5,i,j,k], U2[6,i,j,k], U2[7,i,j,k], U2[8,i,j,k]
                )
    if npass > 0:
        t0 = pr[rmhd_core.PASSIVE_OFFSET:rmhd_core.PASSIVE_OFFSET+npass]
        t1 = pr1[rmhd_core.PASSIVE_OFFSET:rmhd_core.PASSIVE_OFFSET+npass]
        pr2[rmhd_core.PASSIVE_OFFSET:rmhd_core.PASSIVE_OFFSET+npass] = 0.75*t0 + 0.25*(t1 + dt*rhs2[9:9+npass])

    rhs3 = compute_rhs_grmhd(pr2, dx, dy, dz, offs_x, ng)
    U3 = (1.0/3.0)*U0 + (2.0/3.0)*(U2 + dt*rhs3[0:9])

    out  = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[0:9,i,j,k] = rmhd_core.cons_to_prim_rmhd(
                    U3[0,i,j,k], U3[1,i,j,k], U3[2,i,j,k], U3[3,i,j,k], U3[4,i,j,k],
                    U3[5,i,j,k], U3[6,i,j,k], U3[7,i,j,k], U3[8,i,j,k]
                )
    if npass > 0:
        t0 = pr[rmhd_core.PASSIVE_OFFSET:rmhd_core.PASSIVE_OFFSET+npass]
        t2 = pr2[rmhd_core.PASSIVE_OFFSET:rmhd_core.PASSIVE_OFFSET+npass]
        out[rmhd_core.PASSIVE_OFFSET:rmhd_core.PASSIVE_OFFSET+npass] = (1.0/3.0)*t0 + (2.0/3.0)*(t2 + dt*rhs3[9:9+npass])
    return out
