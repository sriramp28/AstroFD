#!/usr/bin/env python3
# core/grhd_core.py
# GRHD (fixed metric) with simple Schwarzschild source terms (approximate).
import numpy as np
from core import srhd_core
from core import gr_metric

GR_METRIC = "minkowski"
GR_MASS = 1.0

def configure(params):
    global GR_METRIC, GR_MASS
    GR_METRIC = str(params.get("GR_METRIC", "minkowski")).lower()
    GR_MASS = float(params.get("GR_MASS", GR_MASS))

def compute_rhs_grhd(pr, dx, dy, dz, offs_x, ng):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    if srhd_core.RECON_ID == 1:
        rhs = srhd_core.compute_rhs_ppm(pr, nx, ny, nz, dx, dy, dz)
    elif srhd_core.RECON_ID == 2:
        rhs = srhd_core.compute_rhs_weno(pr, nx, ny, nz, dx, dy, dz)
    else:
        rhs = srhd_core.compute_rhs_muscl(pr, nx, ny, nz, dx, dy, dz)

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
                v2 = vx*vx + vy*vy + vz*vz
                if v2 >= 1.0:
                    v2 = 1.0 - 1e-14
                W = 1.0 / np.sqrt(1.0 - v2)
                h = 1.0 + srhd_core.GAMMA/(srhd_core.GAMMA-1.0) * p / max(rho, srhd_core.SMALL)
                w = rho * h * W * W + p
                _, Sx, Sy, Sz, _ = srhd_core.prim_to_cons(rho, vx, vy, vz, p)

                rhs[1,i,j,k] += w * dlnadx
                rhs[2,i,j,k] += w * dlnady
                rhs[3,i,j,k] += w * dlnadz
                rhs[4,i,j,k] += Sx*dlnadx + Sy*dlnady + Sz*dlnadz

    return rhs

def step_ssprk2(pr, dx, dy, dz, dt, offs_x, ng):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    nt = srhd_core.N_TRACERS
    U0 = np.zeros((5, nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                U0[:,i,j,k] = srhd_core.prim_to_cons(pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k], pr[3,i,j,k], pr[4,i,j,k])

    rhs1 = compute_rhs_grhd(pr, dx, dy, dz, offs_x, ng)
    U1   = U0 + dt*rhs1[0:5]

    pr1 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr1[0:5,i,j,k] = srhd_core.cons_to_prim(U1[0,i,j,k], U1[1,i,j,k], U1[2,i,j,k], U1[3,i,j,k], U1[4,i,j,k])
    if pr.shape[0] > 5 and srhd_core.TRACER_OFFSET > 5:
        pr1[5:srhd_core.TRACER_OFFSET] = pr[5:srhd_core.TRACER_OFFSET]
    if nt > 0:
        pr1[srhd_core.TRACER_OFFSET:srhd_core.TRACER_OFFSET+nt] = pr[srhd_core.TRACER_OFFSET:srhd_core.TRACER_OFFSET+nt] + dt*rhs1[5:5+nt]

    rhs2 = compute_rhs_grhd(pr1, dx, dy, dz, offs_x, ng)
    U2   = 0.5*(U0 + U1 + dt*rhs2[0:5])

    out  = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[0:5,i,j,k] = srhd_core.cons_to_prim(U2[0,i,j,k], U2[1,i,j,k], U2[2,i,j,k], U2[3,i,j,k], U2[4,i,j,k])
    if pr.shape[0] > 5 and srhd_core.TRACER_OFFSET > 5:
        out[5:srhd_core.TRACER_OFFSET] = pr1[5:srhd_core.TRACER_OFFSET]
    if nt > 0:
        t0 = pr[srhd_core.TRACER_OFFSET:srhd_core.TRACER_OFFSET+nt]
        t1 = pr1[srhd_core.TRACER_OFFSET:srhd_core.TRACER_OFFSET+nt]
        out[srhd_core.TRACER_OFFSET:srhd_core.TRACER_OFFSET+nt] = 0.5*(t0 + t1 + dt*rhs2[5:5+nt])
    return out

def step_ssprk3(pr, dx, dy, dz, dt, offs_x, ng):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    nt = srhd_core.N_TRACERS
    U0 = np.zeros((5, nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                U0[:,i,j,k] = srhd_core.prim_to_cons(pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k], pr[3,i,j,k], pr[4,i,j,k])

    rhs1 = compute_rhs_grhd(pr, dx, dy, dz, offs_x, ng)
    U1 = U0 + dt*rhs1[0:5]

    pr1 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr1[0:5,i,j,k] = srhd_core.cons_to_prim(U1[0,i,j,k], U1[1,i,j,k], U1[2,i,j,k], U1[3,i,j,k], U1[4,i,j,k])
    if pr.shape[0] > 5 and srhd_core.TRACER_OFFSET > 5:
        pr1[5:srhd_core.TRACER_OFFSET] = pr[5:srhd_core.TRACER_OFFSET]
    if nt > 0:
        pr1[srhd_core.TRACER_OFFSET:srhd_core.TRACER_OFFSET+nt] = pr[srhd_core.TRACER_OFFSET:srhd_core.TRACER_OFFSET+nt] + dt*rhs1[5:5+nt]

    rhs2 = compute_rhs_grhd(pr1, dx, dy, dz, offs_x, ng)
    U2 = 0.75*U0 + 0.25*(U1 + dt*rhs2[0:5])

    pr2 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr2[0:5,i,j,k] = srhd_core.cons_to_prim(U2[0,i,j,k], U2[1,i,j,k], U2[2,i,j,k], U2[3,i,j,k], U2[4,i,j,k])
    if pr.shape[0] > 5 and srhd_core.TRACER_OFFSET > 5:
        pr2[5:srhd_core.TRACER_OFFSET] = pr1[5:srhd_core.TRACER_OFFSET]
    if nt > 0:
        t0 = pr[srhd_core.TRACER_OFFSET:srhd_core.TRACER_OFFSET+nt]
        t1 = pr1[srhd_core.TRACER_OFFSET:srhd_core.TRACER_OFFSET+nt]
        pr2[srhd_core.TRACER_OFFSET:srhd_core.TRACER_OFFSET+nt] = 0.75*t0 + 0.25*(t1 + dt*rhs2[5:5+nt])

    rhs3 = compute_rhs_grhd(pr2, dx, dy, dz, offs_x, ng)
    U3 = (1.0/3.0)*U0 + (2.0/3.0)*(U2 + dt*rhs3[0:5])

    out = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[0:5,i,j,k] = srhd_core.cons_to_prim(U3[0,i,j,k], U3[1,i,j,k], U3[2,i,j,k], U3[3,i,j,k], U3[4,i,j,k])
    if pr.shape[0] > 5 and srhd_core.TRACER_OFFSET > 5:
        out[5:srhd_core.TRACER_OFFSET] = pr2[5:srhd_core.TRACER_OFFSET]
    if nt > 0:
        t0 = pr[srhd_core.TRACER_OFFSET:srhd_core.TRACER_OFFSET+nt]
        t2 = pr2[srhd_core.TRACER_OFFSET:srhd_core.TRACER_OFFSET+nt]
        out[srhd_core.TRACER_OFFSET:srhd_core.TRACER_OFFSET+nt] = (1.0/3.0)*t0 + (2.0/3.0)*(t2 + dt*rhs3[5:5+nt])
    return out
