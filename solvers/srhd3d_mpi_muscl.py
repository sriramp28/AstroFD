#!/usr/bin/env python3
# srhd3d_mpi_muscl.py  — 3D SRHD (flat), MUSCL + HLLE + SSPRK2
# MPI x-slab decomposition, blocking Sendrecv halo exchange, jet nozzle inflow
# Includes simple diagnostics: global max Lorentz factor and inlet energy flux
#
# Usage:
#   python3 -m pip install numpy numba mpi4py
#   mpirun -np 2 python3 srhd3d_mpi_muscl.py

import os, math, time
import numpy as np
from mpi4py import MPI
import numba as nb

from utils.settings import load_settings
from utils.io_utils import make_run_dir

settings = load_settings()

# assign globals from settings (Numba reads them before first JIT use)
NX, NY, NZ = settings["NX"], settings["NY"], settings["NZ"]
Lx, Ly, Lz = settings["Lx"], settings["Ly"], settings["Lz"]
T_END, OUT_EVERY, PRINT_EVERY = settings["T_END"], settings["OUT_EVERY"], settings["PRINT_EVERY"]
NG, CFL, GAMMA = settings["NG"], settings["CFL"], settings["GAMMA"]

JET_RADIUS    = settings["JET_RADIUS"]
JET_CENTER    = settings["JET_CENTER"]      # [x,y,z] (we use y,z)
GAMMA_JET     = settings["GAMMA_JET"]
ETA_RHO       = settings["ETA_RHO"]
P_EQ          = settings["P_EQ"]
SHEAR_THICK   = settings["SHEAR_THICK"]
NOZZLE_TURB   = settings["NOZZLE_TURB"]
TURB_VAMP     = settings["TURB_VAMP"]
TURB_PAMP     = settings["TURB_PAMP"]

RHO_AMB       = settings["RHO_AMB"]
P_AMB         = settings["P_AMB"]
VX_AMB        = settings["VX_AMB"]
VY_AMB        = settings["VY_AMB"]
VZ_AMB        = settings["VZ_AMB"]

DEBUG         = settings["DEBUG"]
ASSERTS       = settings["ASSERTS"]
CHECK_NAN_EVERY = settings["CHECK_NAN_EVERY"]

SMALL = 1e-12

# ------------------------
# SRHD helpers (Numba)
# ------------------------
@nb.njit(fastmath=True)
def prim_to_cons(rho, vx, vy, vz, p, gamma=GAMMA):
    v2 = vx*vx + vy*vy + vz*vz
    if v2 >= 1.0: v2 = 1.0 - 1e-14
    W  = 1.0/np.sqrt(1.0 - v2)
    h  = 1.0 + gamma/(gamma-1.0)*p/np.maximum(rho, SMALL)
    w  = rho*h
    D  = rho*W
    Sx = w*W*W*vx
    Sy = w*W*W*vy
    Sz = w*W*W*vz
    tau= w*W*W - p - D
    return D, Sx, Sy, Sz, tau

@nb.njit(fastmath=True)
def cons_to_prim(D, Sx, Sy, Sz, tau, gamma=GAMMA):
    E = tau + D
    S2 = Sx*Sx + Sy*Sy + Sz*Sz
    p = (gamma-1.0)*(E - D)
    if p < SMALL: p = SMALL
    for _ in range(60):
        Wm = E + p
        v2 = S2 / (Wm*Wm + SMALL)
        if v2 >= 1.0: v2 = 1.0 - 1e-14
        W  = 1.0/np.sqrt(1.0 - v2)
        rho= D/np.maximum(W, SMALL)
        h  = 1.0 + gamma/(gamma-1.0)*p/np.maximum(rho, SMALL)
        w  = rho*h
        f  = Wm - w*W*W
        dfdp = 1.0
        dp = -f/dfdp
        if dp >  0.5*p: dp =  0.5*p
        if dp < -0.5*p: dp = -0.5*p
        p_new = p + dp
        if p_new < SMALL: p_new = SMALL
        if abs(dp) < 1e-12*max(1.0, p_new):
            p = p_new; break
        p = p_new
    Wm = E + p
    vx = Sx/np.maximum(Wm, SMALL)
    vy = Sy/np.maximum(Wm, SMALL)
    vz = Sz/np.maximum(Wm, SMALL)
    v2 = vx*vx + vy*vy + vz*vz
    if v2 >= 1.0:
        fac = (1.0 - 1e-14)/np.sqrt(v2)
        vx *= fac; vy *= fac; vz *= fac
    v2 = vx*vx + vy*vy + vz*vz
    W  = 1.0/np.sqrt(1.0 - v2)
    rho= D/np.maximum(W, SMALL)
    if rho < SMALL: rho = SMALL
    if p   < SMALL: p   = SMALL
    return rho, vx, vy, vz, p

@nb.njit(fastmath=True)
def sound_speed(rho, p, gamma=GAMMA):
    h  = 1.0 + gamma/(gamma-1.0)*p/np.maximum(rho, SMALL)
    w  = rho*h
    cs2= gamma*p/np.maximum(w, SMALL)
    if cs2 < 0.0: cs2 = 0.0
    if cs2 > 1.0 - 1e-14: cs2 = 1.0 - 1e-14
    return np.sqrt(cs2)

@nb.njit(fastmath=True)
def eig_speeds(vn, cs):
    dp = 1.0 + vn*cs
    dm = 1.0 - vn*cs
    if dp == 0.0: dp = SMALL
    if dm == 0.0: dm = SMALL
    lp = (vn + cs)/dp
    lm = (vn - cs)/dm
    if lp >  1.0: lp =  1.0
    if lp < -1.0: lp = -1.0
    if lm >  1.0: lm =  1.0
    if lm < -1.0: lm = -1.0
    return lm, lp

@nb.njit(fastmath=True)
def flux_x(rho, vx, vy, vz, p):
    D,Sx,Sy,Sz,tau = prim_to_cons(rho,vx,vy,vz,p)
    return np.array([D*vx, Sx*vx + p, Sy*vx, Sz*vx, (tau+p)*vx])

@nb.njit(fastmath=True)
def flux_y(rho, vx, vy, vz, p):
    D,Sx,Sy,Sz,tau = prim_to_cons(rho,vx,vy,vz,p)
    return np.array([D*vy, Sx*vy, Sy*vy + p, Sz*vy, (tau+p)*vy])

@nb.njit(fastmath=True)
def flux_z(rho, vx, vy, vz, p):
    D,Sx,Sy,Sz,tau = prim_to_cons(rho,vx,vy,vz,p)
    return np.array([D*vz, Sx*vz, Sy*vz, Sz*vz + p, (tau+p)*vz])

@nb.njit(fastmath=True)
def hlle(UL, UR, FL, FR, sL, sR):
    if sL >= 0.0:
        return FL
    elif sR <= 0.0:
        return FR
    else:
        return (sR*FL - sL*FR + sL*sR*(UR-UL)) / (sR - sL + SMALL)

# ------------------------
# MUSCL (MC limiter) helpers
# ------------------------
@nb.njit(fastmath=True)
def minmod(a, b):
    if a*b <= 0.0:
        return 0.0
    else:
        if abs(a) < abs(b): return a
        else:               return b

@nb.njit(fastmath=True)
def mc_lim(dqL, dqR):
    mm = minmod(dqL, dqR)
    return minmod(0.5*(dqL + dqR), 2.0*mm)

@nb.njit(fastmath=True)
def floor_prim(rho, vx, vy, vz, p):
    if rho < SMALL: rho = SMALL
    if p   < SMALL: p   = SMALL
    v2 = vx*vx + vy*vy + vz*vz
    if v2 >= 1.0:
        fac = (1.0 - 1e-14)/np.sqrt(v2)
        vx *= fac; vy *= fac; vz *= fac
    return rho, vx, vy, vz, p

# ------------------------
# RHS with MUSCL reconstruction in x/y/z
# Full implementation inlined for standalone file
# ------------------------
@nb.njit(parallel=True, fastmath=True)
def compute_rhs_muscl(pr, nx, ny, nz, dx, dy, dz):
    """
    Compute RHS with MUSCL (MC limiter) reconstruction.
    IMPORTANT: loops start at 2 and stop at N-2 so all i±2, j±2, k±2 are in-bounds.
    Requires NG >= 2 ghost layers.
    """
    rhs = np.zeros((5, nx, ny, nz))

    i0, i1 = 2, nx-2
    j0, j1 = 2, ny-2
    k0, k1 = 2, nz-2

    # X faces
    for i in range(i0, i1):
        for j in range(j0, j1):
            for k in range(k0, k1):
                # slopes around i-1 and i
                dqL_r = pr[0,i-1,j,k] - pr[0,i-2,j,k]
                dqR_r = pr[0,i  ,j,k] - pr[0,i-1,j,k]
                slL_r = mc_lim(dqL_r, dqR_r)
                dqL_r = pr[0,i  ,j,k] - pr[0,i-1,j,k]
                dqR_r = pr[0,i+1,j,k] - pr[0,i  ,j,k]
                slR_r = mc_lim(dqL_r, dqR_r)

                dqL_vx = pr[1,i-1,j,k] - pr[1,i-2,j,k]
                dqR_vx = pr[1,i  ,j,k] - pr[1,i-1,j,k]
                slL_vx = mc_lim(dqL_vx, dqR_vx)
                dqL_vx = pr[1,i  ,j,k] - pr[1,i-1,j,k]
                dqR_vx = pr[1,i+1,j,k] - pr[1,i  ,j,k]
                slR_vx = mc_lim(dqL_vx, dqR_vx)

                dqL_vy = pr[2,i-1,j,k] - pr[2,i-2,j,k]
                dqR_vy = pr[2,i  ,j,k] - pr[2,i-1,j,k]
                slL_vy = mc_lim(dqL_vy, dqR_vy)
                dqL_vy = pr[2,i  ,j,k] - pr[2,i-1,j,k]
                dqR_vy = pr[2,i+1,j,k] - pr[2,i  ,j,k]
                slR_vy = mc_lim(dqL_vy, dqR_vy)

                dqL_vz = pr[3,i-1,j,k] - pr[3,i-2,j,k]
                dqR_vz = pr[3,i  ,j,k] - pr[3,i-1,j,k]
                slL_vz = mc_lim(dqL_vz, dqR_vz)
                dqL_vz = pr[3,i  ,j,k] - pr[3,i-1,j,k]
                dqR_vz = pr[3,i+1,j,k] - pr[3,i  ,j,k]
                slR_vz = mc_lim(dqL_vz, dqR_vz)

                dqL_p = pr[4,i-1,j,k] - pr[4,i-2,j,k]
                dqR_p = pr[4,i  ,j,k] - pr[4,i-1,j,k]
                slL_p = mc_lim(dqL_p, dqR_p)
                dqL_p = pr[4,i  ,j,k] - pr[4,i-1,j,k]
                dqR_p = pr[4,i+1,j,k] - pr[4,i  ,j,k]
                slR_p = mc_lim(dqL_p, dqR_p)

                # left face (i-1/2)
                rL = pr[0,i-1,j,k] + 0.5*slL_r
                vxL= pr[1,i-1,j,k] + 0.5*slL_vx
                vyL= pr[2,i-1,j,k] + 0.5*slL_vy
                vzL= pr[3,i-1,j,k] + 0.5*slL_vz
                pL = pr[4,i-1,j,k] + 0.5*slL_p

                # right state for same face from cell i
                rR = pr[0,i  ,j,k] - 0.5*slR_r
                vxR= pr[1,i  ,j,k] - 0.5*slR_vx
                vyR= pr[2,i  ,j,k] - 0.5*slR_vy
                vzR= pr[3,i  ,j,k] - 0.5*slR_vz
                pR = pr[4,i  ,j,k] - 0.5*slR_p

                rL,vxL,vyL,vzL,pL = floor_prim(rL,vxL,vyL,vzL,pL)
                rR,vxR,vyR,vzR,pR = floor_prim(rR,vxR,vyR,vzR,pR)

                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL))
                UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_x(rL,vxL,vyL,vzL,pL)
                FR = flux_x(rR,vxR,vyR,vzR,pR)
                csL= sound_speed(rL,pL); csR= sound_speed(rR,pR)
                lmL, lpL = eig_speeds(vxL, csL)
                lmR, lpR = eig_speeds(vxR, csR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                FxL = hlle(UL, UR, FL, FR, sL, sR)

                # right face (i+1/2): need slopes centered at i+1
                dqL_r = pr[0,i+1,j,k] - pr[0,i  ,j,k]
                dqR_r = pr[0,i+2,j,k] - pr[0,i+1,j,k]
                slRp1_r = mc_lim(dqL_r, dqR_r)

                dqL_vx = pr[1,i+1,j,k] - pr[1,i  ,j,k]
                dqR_vx = pr[1,i+2,j,k] - pr[1,i+1,j,k]
                slRp1_vx = mc_lim(dqL_vx, dqR_vx)

                dqL_vy = pr[2,i+1,j,k] - pr[2,i  ,j,k]
                dqR_vy = pr[2,i+2,j,k] - pr[2,i+1,j,k]
                slRp1_vy = mc_lim(dqL_vy, dqR_vy)

                dqL_vz = pr[3,i+1,j,k] - pr[3,i  ,j,k]
                dqR_vz = pr[3,i+2,j,k] - pr[3,i+1,j,k]
                slRp1_vz = mc_lim(dqL_vz, dqR_vz)

                dqL_p = pr[4,i+1,j,k] - pr[4,i  ,j,k]
                dqR_p = pr[4,i+2,j,k] - pr[4,i+1,j,k]
                slRp1_p = mc_lim(dqL_p, dqR_p)

                rL = pr[0,i  ,j,k] + 0.5*slR_r
                vxL= pr[1,i  ,j,k] + 0.5*slR_vx
                vyL= pr[2,i  ,j,k] + 0.5*slR_vy
                vzL= pr[3,i  ,j,k] + 0.5*slR_vz
                pL = pr[4,i  ,j,k] + 0.5*slR_p

                rR = pr[0,i+1,j,k] - 0.5*slRp1_r
                vxR= pr[1,i+1,j,k] - 0.5*slRp1_vx
                vyR= pr[2,i+1,j,k] - 0.5*slRp1_vy
                vzR= pr[3,i+1,j,k] - 0.5*slRp1_vz
                pR = pr[4,i+1,j,k] - 0.5*slRp1_p

                rL,vxL,vyL,vzL,pL = floor_prim(rL,vxL,vyL,vzL,pL)
                rR,vxR,vyR,vzR,pR = floor_prim(rR,vxR,vyR,vzR,pR)

                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL))
                UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_x(rL,vxL,vyL,vzL,pL)
                FR = flux_x(rR,vxR,vyR,vzR,pR)
                csL= sound_speed(rL,pL); csR= sound_speed(rR,pR)
                lmL, lpL = eig_speeds(vxL, csL)
                lmR, lpR = eig_speeds(vxR, csR)
                sL2 = min(lmL, lmR); sR2 = max(lpL, lpR)
                FxR = hlle(UL, UR, FL, FR, sL2, sR2)

                rhs[:,i,j,k] -= (FxR - FxL)/dx

    # Y faces
    for i in range(i0, i1):
        for j in range(j0, j1):
            for k in range(k0, k1):
                dqL_r = pr[0,i,j-1,k] - pr[0,i,j-2,k]
                dqR_r = pr[0,i,j  ,k] - pr[0,i,j-1,k]
                slL_r = mc_lim(dqL_r, dqR_r)
                dqL_r = pr[0,i,j  ,k] - pr[0,i,j-1,k]
                dqR_r = pr[0,i,j+1,k] - pr[0,i,j  ,k]
                slR_r = mc_lim(dqL_r, dqR_r)

                dqL_vx = pr[1,i,j-1,k] - pr[1,i,j-2,k]
                dqR_vx = pr[1,i,j  ,k] - pr[1,i,j-1,k]
                slL_vx = mc_lim(dqL_vx, dqR_vx)
                dqL_vx = pr[1,i,j  ,k] - pr[1,i,j-1,k]
                dqR_vx = pr[1,i,j+1,k] - pr[1,i,j  ,k]
                slR_vx = mc_lim(dqL_vx, dqR_vx)

                dqL_vy = pr[2,i,j-1,k] - pr[2,i,j-2,k]
                dqR_vy = pr[2,i,j  ,k] - pr[2,i,j-1,k]
                slL_vy = mc_lim(dqL_vy, dqR_vy)
                dqL_vy = pr[2,i,j  ,k] - pr[2,i,j-1,k]
                dqR_vy = pr[2,i,j+1,k] - pr[2,i,j  ,k]
                slR_vy = mc_lim(dqL_vy, dqR_vy)

                dqL_vz = pr[3,i,j-1,k] - pr[3,i,j-2,k]
                dqR_vz = pr[3,i,j  ,k] - pr[3,i,j-1,k]
                slL_vz = mc_lim(dqL_vz, dqR_vz)
                dqL_vz = pr[3,i,j  ,k] - pr[3,i,j-1,k]
                dqR_vz = pr[3,i,j+1,k] - pr[3,i,j  ,k]
                slR_vz = mc_lim(dqL_vz, dqR_vz)

                dqL_p = pr[4,i,j-1,k] - pr[4,i,j-2,k]
                dqR_p = pr[4,i,j  ,k] - pr[4,i,j-1,k]
                slL_p = mc_lim(dqL_p, dqR_p)
                dqL_p = pr[4,i,j  ,k] - pr[4,i,j-1,k]
                dqR_p = pr[4,i,j+1,k] - pr[4,i,j  ,k]
                slR_p = mc_lim(dqL_p, dqR_p)

                rL = pr[0,i,j-1,k] + 0.5*slL_r
                vxL= pr[1,i,j-1,k] + 0.5*slL_vx
                vyL= pr[2,i,j-1,k] + 0.5*slL_vy
                vzL= pr[3,i,j-1,k] + 0.5*slL_vz
                pL = pr[4,i,j-1,k] + 0.5*slL_p

                rR = pr[0,i,j  ,k] - 0.5*slR_r
                vxR= pr[1,i,j  ,k] - 0.5*slR_vx
                vyR= pr[2,i,j  ,k] - 0.5*slR_vy
                vzR= pr[3,i,j  ,k] - 0.5*slR_vz
                pR = pr[4,i,j  ,k] - 0.5*slR_p

                rL,vxL,vyL,vzL,pL = floor_prim(rL,vxL,vyL,vzL,pL)
                rR,vxR,vyR,vzR,pR = floor_prim(rR,vxR,vyR,vzR,pR)

                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL))
                UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_y(rL,vxL,vyL,vzL,pL)
                FR = flux_y(rR,vxR,vyR,vzR,pR)
                csL= sound_speed(rL,pL); csR= sound_speed(rR,pR)
                lmL, lpL = eig_speeds(vyL, csL)
                lmR, lpR = eig_speeds(vyR, csR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                FyD = hlle(UL, UR, FL, FR, sL, sR)

                dqL_r = pr[0,i,j+1,k] - pr[0,i,j  ,k]
                dqR_r = pr[0,i,j+2,k] - pr[0,i,j+1,k]
                slUp_r = mc_lim(dqL_r, dqR_r)

                dqL_vx = pr[1,i,j+1,k] - pr[1,i,j  ,k]
                dqR_vx = pr[1,i,j+2,k] - pr[1,i,j+1,k]
                slUp_vx = mc_lim(dqL_vx, dqR_vx)

                dqL_vy = pr[2,i,j+1,k] - pr[2,i,j  ,k]
                dqR_vy = pr[2,i,j+2,k] - pr[2,i,j+1,k]
                slUp_vy = mc_lim(dqL_vy, dqR_vy)

                dqL_vz = pr[3,i,j+1,k] - pr[3,i,j  ,k]
                dqR_vz = pr[3,i,j+2,k] - pr[3,i,j+1,k]
                slUp_vz = mc_lim(dqL_vz, dqR_vz)

                dqL_p = pr[4,i,j+1,k] - pr[4,i,j  ,k]
                dqR_p = pr[4,i,j+2,k] - pr[4,i,j+1,k]
                slUp_p = mc_lim(dqL_p, dqR_p)

                rL = pr[0,i,j  ,k] + 0.5*slR_r
                vxL= pr[1,i,j  ,k] + 0.5*slR_vx
                vyL= pr[2,i,j  ,k] + 0.5*slR_vy
                vzL= pr[3,i,j  ,k] + 0.5*slR_vz
                pL = pr[4,i,j  ,k] + 0.5*slR_p

                rR = pr[0,i,j+1,k] - 0.5*slUp_r
                vxR= pr[1,i,j+1,k] - 0.5*slUp_vx
                vyR= pr[2,i,j+1,k] - 0.5*slUp_vy
                vzR= pr[3,i,j+1,k] - 0.5*slUp_vz
                pR = pr[4,i,j+1,k] - 0.5*slUp_p

                rL,vxL,vyL,vzL,pL = floor_prim(rL,vxL,vyL,vzL,pL)
                rR,vxR,vyR,vzR,pR = floor_prim(rR,vxR,vyR,vzR,pR)

                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL))
                UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_y(rL,vxL,vyL,vzL,pL)
                FR = flux_y(rR,vxR,vyR,vzR,pR)
                csL= sound_speed(rL,pL); csR= sound_speed(rR,pR)
                lmL, lpL = eig_speeds(vyL, csL)
                lmR, lpR = eig_speeds(vyR, csR)
                sL2 = min(lmL, lmR); sR2 = max(lpL, lpR)
                FyU = hlle(UL, UR, FL, FR, sL2, sR2)

                rhs[:,i,j,k] -= (FyU - FyD)/dy

    # Z faces
    for i in range(i0, i1):
        for j in range(j0, j1):
            for k in range(k0, k1):
                dqL_r = pr[0,i,j,k-1] - pr[0,i,j,k-2]
                dqR_r = pr[0,i,j,k  ] - pr[0,i,j,k-1]
                slL_r = mc_lim(dqL_r, dqR_r)
                dqL_r = pr[0,i,j,k  ] - pr[0,i,j,k-1]
                dqR_r = pr[0,i,j,k+1] - pr[0,i,j,k  ]
                slR_r = mc_lim(dqL_r, dqR_r)

                dqL_vx = pr[1,i,j,k-1] - pr[1,i,j,k-2]
                dqR_vx = pr[1,i,j,k  ] - pr[1,i,j,k-1]
                slL_vx = mc_lim(dqL_vx, dqR_vx)
                dqL_vx = pr[1,i,j,k  ] - pr[1,i,j,k-1]
                dqR_vx = pr[1,i,j,k+1] - pr[1,i,j,k  ]
                slR_vx = mc_lim(dqL_vx, dqR_vx)

                dqL_vy = pr[2,i,j,k-1] - pr[2,i,j,k-2]
                dqR_vy = pr[2,i,j,k  ] - pr[2,i,j,k-1]
                slL_vy = mc_lim(dqL_vy, dqR_vy)
                dqL_vy = pr[2,i,j,k  ] - pr[2,i,j,k-1]
                dqR_vy = pr[2,i,j,k+1] - pr[2,i,j,k  ]
                slR_vy = mc_lim(dqL_vy, dqR_vy)

                dqL_vz = pr[3,i,j,k-1] - pr[3,i,j,k-2]
                dqR_vz = pr[3,i,j,k  ] - pr[3,i,j,k-1]
                slL_vz = mc_lim(dqL_vz, dqR_vz)
                dqL_vz = pr[3,i,j,k  ] - pr[3,i,j,k-1]
                dqR_vz = pr[3,i,j,k+1] - pr[3,i,j,k  ]
                slR_vz = mc_lim(dqL_vz, dqR_vz)

                dqL_p = pr[4,i,j,k-1] - pr[4,i,j,k-2]
                dqR_p = pr[4,i,j,k  ] - pr[4,i,j,k-1]
                slL_p = mc_lim(dqL_p, dqR_p)
                dqL_p = pr[4,i,j,k  ] - pr[4,i,j,k-1]
                dqR_p = pr[4,i,j,k+1] - pr[4,i,j,k  ]
                slR_p = mc_lim(dqL_p, dqR_p)

                rL = pr[0,i,j,k-1] + 0.5*slL_r
                vxL= pr[1,i,j,k-1] + 0.5*slL_vx
                vyL= pr[2,i,j,k-1] + 0.5*slL_vy
                vzL= pr[3,i,j,k-1] + 0.5*slL_vz
                pL = pr[4,i,j,k-1] + 0.5*slL_p

                rR = pr[0,i,j,k  ] - 0.5*slR_r
                vxR= pr[1,i,j,k  ] - 0.5*slR_vx
                vyR= pr[2,i,j,k  ] - 0.5*slR_vy
                vzR= pr[3,i,j,k  ] - 0.5*slR_vz
                pR = pr[4,i,j,k  ] - 0.5*slR_p

                rL,vxL,vyL,vzL,pL = floor_prim(rL,vxL,vyL,vzL,pL)
                rR,vxR,vyR,vzR,pR = floor_prim(rR,vxR,vyR,vzR,pR)

                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL))
                UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_z(rL,vxL,vyL,vzL,pL)
                FR = flux_z(rR,vxR,vyR,vzR,pR)
                csL= sound_speed(rL,pL); csR= sound_speed(rR,pR)
                lmL, lpL = eig_speeds(vzL, csL)
                lmR, lpR = eig_speeds(vzR, csR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                FzB = hlle(UL, UR, FL, FR, sL, sR)

                dqL_r = pr[0,i,j,k+1] - pr[0,i,j,k  ]
                dqR_r = pr[0,i,j,k+2] - pr[0,i,j,k+1]
                slFp_r = mc_lim(dqL_r, dqR_r)

                dqL_vx = pr[1,i,j,k+1] - pr[1,i,j,k  ]
                dqR_vx = pr[1,i,j,k+2] - pr[1,i,j,k+1]
                slFp_vx = mc_lim(dqL_vx, dqR_vx)

                dqL_vy = pr[2,i,j,k+1] - pr[2,i,j,k  ]
                dqR_vy = pr[2,i,j,k+2] - pr[2,i,j,k+1]
                slFp_vy = mc_lim(dqL_vy, dqR_vy)

                dqL_vz = pr[3,i,j,k+1] - pr[3,i,j,k  ]
                dqR_vz = pr[3,i,j,k+2] - pr[3,i,j,k+1]
                slFp_vz = mc_lim(dqL_vz, dqR_vz)

                dqL_p = pr[4,i,j,k+1] - pr[4,i,j,k  ]
                dqR_p = pr[4,i,j,k+2] - pr[4,i,j,k+1]
                slFp_p = mc_lim(dqL_p, dqR_p)

                rL = pr[0,i,j,k  ] + 0.5*slR_r
                vxL= pr[1,i,j,k  ] + 0.5*slR_vx
                vyL= pr[2,i,j,k  ] + 0.5*slR_vy
                vzL= pr[3,i,j,k  ] + 0.5*slR_vz
                pL = pr[4,i,j,k  ] + 0.5*slR_p

                rR = pr[0,i,j,k+1] - 0.5*slFp_r
                vxR= pr[1,i,j,k+1] - 0.5*slFp_vx
                vyR= pr[2,i,j,k+1] - 0.5*slFp_vy
                vzR= pr[3,i,j,k+1] - 0.5*slFp_vz
                pR = pr[4,i,j,k+1] - 0.5*slFp_p

                rL,vxL,vyL,vzL,pL = floor_prim(rL,vxL,vyL,vzL,pL)
                rR,vxR,vyR,vzR,pR = floor_prim(rR,vxR,vyR,vzR,pR)

                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL))
                UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_z(rL,vxL,vyL,vzL,pL)
                FR = flux_z(rR,vxR,vyR,vzR,pR)
                csL= sound_speed(rL,pL); csR= sound_speed(rR,pR)
                lmL, lpL = eig_speeds(vzL, csL)
                lmR, lpR = eig_speeds(vzR, csR)
                sL2 = min(lmL, lmR); sR2 = max(lpL, lpR)
                FzF = hlle(UL, UR, FL, FR, sL2, sR2)

                rhs[:,i,j,k] -= (FzF - FzB)/dz

    return rhs
# ------------------------
# max char speed
# ------------------------
@nb.njit(fastmath=True)
def max_char_speed(pr, nx, ny, nz):
    amax = 0.0
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                rho, vx, vy, vz, p = pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k], pr[3,i,j,k], pr[4,i,j,k]
                cs = sound_speed(rho, p)
                axm, axp = eig_speeds(vx, cs)
                aym, ayp = eig_speeds(vy, cs)
                azm, azp = eig_speeds(vz, cs)
                loc = max(abs(axm),abs(axp),abs(aym),abs(ayp),abs(azm),abs(azp))
                if loc > amax: amax = loc
    if amax < SMALL: amax = SMALL
    return amax

# ------------------------
# MPI utilities: x-slab decomposition
# ------------------------
def decompose_x(nx_glob, comm):
    size = comm.Get_size(); rank = comm.Get_rank()
    counts = [nx_glob // size]*size
    for r in range(nx_glob % size): counts[r] += 1
    offsets = [0]*size
    for r in range(1,size): offsets[r] = offsets[r-1] + counts[r-1]
    return counts[rank], offsets[rank], counts, offsets

# ------------------------
# Blocking halo exchange using Sendrecv (robust + simple)
# ------------------------
def exchange_halos(pr, comm, left, right):
    """
    Exchange NG ghost layers along x with neighbors using blocking Sendrecv.
    pr shape: (5, nx_loc + 2*NG, NY + 2*NG, NZ + 2*NG)
    """
    # phase 1: exchange with left neighbor
    if left is not None:
        sendL = np.ascontiguousarray(pr[:, NG:2*NG, :, :])     # our first interior slab
        recvL = np.empty_like(sendL)                            # will hold neighbor's right interior slab
        comm.Sendrecv(sendbuf=sendL, dest=left,  sendtag=21,
                      recvbuf=recvL, source=left, recvtag=20)
        pr[:, 0:NG, :, :] = recvL                               # copy into left ghosts

    # phase 2: exchange with right neighbor
    if right is not None:
        sendR = np.ascontiguousarray(pr[:, -2*NG:-NG, :, :])    # our last interior slab
        recvR = np.empty_like(sendR)                            # neighbor's left interior slab
        comm.Sendrecv(sendbuf=sendR, dest=right, sendtag=20,
                      recvbuf=recvR, source=right, recvtag=21)
        pr[:, -NG:, :, :] = recvR                               # copy into right ghosts

# ------------------------
# BCs & nozzle (same as earlier)
# ------------------------
def apply_periodic_yz(pr):
    pr[:, :, 0:NG, :] = pr[:, :, -2*NG:-NG, :]
    pr[:, :, -NG:, :] = pr[:, :, NG:2*NG, :]
    pr[:, :, :, 0:NG] = pr[:, :, :, -2*NG:-NG]
    pr[:, :, :, -NG:] = pr[:, :, :, NG:2*NG]

def apply_outflow_right_x(pr):
    pr[:, -NG:, :, :] = pr[:, -NG-1:-NG, :, :]

def apply_nozzle_left_x(pr, dx, dy, dz, ny_loc, nz_loc, y0, z0, rng):
    for g in range(NG):
        for j in range(NG, NG+ny_loc):
            y = (j-NG + 0.5)*dy
            for k in range(NG, NG+nz_loc):
                z = (k-NG + 0.5)*dz
                rr = math.sqrt((y - y0)**2 + (z - z0)**2)
                s = 0.5*(1.0 - math.tanh((rr - JET_RADIUS)/SHEAR_THICK)) if SHEAR_THICK>0.0 else (1.0 if rr<=JET_RADIUS else 0.0)
                rho = ETA_RHO*RHO_AMB * s + RHO_AMB*(1.0 - s)
                p   = P_EQ
                beta= math.sqrt(1.0 - 1.0/(GAMMA_JET*GAMMA_JET))
                vx  = beta * s + VX_AMB*(1.0 - s)
                vy  = VY_AMB
                vz  = VZ_AMB
                if NOZZLE_TURB and s>0.0:
                    vy += TURB_VAMP * (2.0*rng.random()-1.0)
                    vz += TURB_VAMP * (2.0*rng.random()-1.0)
                    p  *= (1.0 + TURB_PAMP * (2.0*rng.random()-1.0))
                pr[0, g, j, k] = rho
                pr[1, g, j, k] = vx
                pr[2, g, j, k] = vy
                pr[3, g, j, k] = vz
                pr[4, g, j, k] = p

# ------------------------
# Initialization
# ------------------------
def init_block(nx_loc, ny_loc, nz_loc):
    pr = np.zeros((5, nx_loc + 2*NG, ny_loc + 2*NG, nz_loc + 2*NG), dtype=np.float64)
    pr[0, :, :, :] = RHO_AMB
    pr[1, :, :, :] = VX_AMB
    pr[2, :, :, :] = VY_AMB
    pr[3, :, :, :] = VZ_AMB
    pr[4, :, :, :] = P_AMB
    return pr

# ------------------------
# Time stepping (SSPRK2)
# ------------------------
def step_ssprk2(pr, dx, dy, dz, dt):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    U0 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                U0[:,i,j,k] = prim_to_cons(pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k], pr[3,i,j,k], pr[4,i,j,k])

    rhs1 = compute_rhs_muscl(pr, nx, ny, nz, dx, dy, dz)
    U1   = U0 + dt*rhs1

    pr1 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr1[:,i,j,k] = cons_to_prim(U1[0,i,j,k], U1[1,i,j,k], U1[2,i,j,k], U1[3,i,j,k], U1[4,i,j,k])

    rhs2 = compute_rhs_muscl(pr1, nx, ny, nz, dx, dy, dz)
    U2   = 0.5*(U0 + U1 + dt*rhs2)

    out  = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[:,i,j,k] = cons_to_prim(U2[0,i,j,k], U2[1,i,j,k], U2[2,i,j,k], U2[3,i,j,k], U2[4,i,j,k])
    return out

# ------------------------
# Diagnostics: max Lorentz factor (global) and inlet energy flux (rank 0)
# ------------------------
def compute_diagnostics_and_write(pr, dx, dy, dz, offs_x, counts, comm, rank, step, t, dt, amax, run_dir):
    """
    pr has ghosts. We'll:
      - compute local max Lorentz factor across interior cells,
      - reduce global max to rank 0,
      - on rank 0 compute inlet energy flux (Sx integrated over j,k at first interior i),
      - append to diagnostics.csv on rank 0.
    """
    nx_loc = pr.shape[1] - 2*NG
    ny_loc = pr.shape[2] - 2*NG
    nz_loc = pr.shape[3] - 2*NG

    # local max gamma
    local_maxG = 0.0
    for i in range(NG, NG + nx_loc):
        for j in range(NG, NG + ny_loc):
            for k in range(NG, NG + nz_loc):
                vx = pr[1,i,j,k]; vy = pr[2,i,j,k]; vz = pr[3,i,j,k]
                v2 = vx*vx + vy*vy + vz*vz
                if v2 >= 1.0: v2 = 1.0 - 1e-14
                W = 1.0 / math.sqrt(1.0 - v2)
                if W > local_maxG: local_maxG = W

    global_maxG = comm.allreduce(local_maxG, op=MPI.MAX)

    inlet_flux = 0.0
    # inlet plane lives on rank 0 only: first interior i = NG
    if rank == 0:
        i = NG
        for j in range(NG, NG + ny_loc):
            for k in range(NG, NG + nz_loc):
                rho = pr[0,i,j,k]; vx = pr[1,i,j,k]; vy = pr[2,i,j,k]; vz = pr[3,i,j,k]; p = pr[4,i,j,k]
                _, Sx, _, _, _ = prim_to_cons(rho, vx, vy, vz, p)
                inlet_flux += Sx * dy * dz

    # gather inlet_flux global (rank 0 only needs it)
    total_inlet_flux = comm.allreduce(inlet_flux, op=MPI.SUM)

    # write diagnostics on rank 0
    if rank == 0:
        fn = os.path.join(run_dir, "diagnostics.csv")
        header = False
        if not os.path.exists(fn):
            header = True
        with open(fn, "a") as f:
            if header:
                f.write("step,time,dt,amax,maxGamma,inletEnergyFlux\n")
            f.write(f"{step},{t:.8e},{dt:.8e},{amax:.6e},{global_maxG:.6e},{total_inlet_flux:.6e}\n")

    return global_maxG, total_inlet_flux

# ------------------------
# Main
# ------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- create output dir ---
    RUN_DIR = make_run_dir(base="results", unique=settings.get("RESULTS_UNIQUE", False))
    if rank == 0:
        print(f"[startup] run directory: {RUN_DIR}", flush=True)

    # --- startup banner ---
    if rank == 0:
        print(f"[startup] ranks={size} grid={NX}x{NY}x{NZ} NG={NG} debug={DEBUG}", flush=True)
        if DEBUG:
            try:
                from numba import config as nbconfig
                print(f"[startup] numba threads={nbconfig.NUMBA_NUM_THREADS}", flush=True)
            except Exception:
                pass

    # domain decomposition in x
    nx_loc, x0, counts, offs = decompose_x(NX, comm)
    left  = rank-1 if rank-1 >= 0     else None
    right = rank+1 if rank+1 < size   else None

    dx, dy, dz = Lx/NX, Ly/NY, Lz/NZ
    ny_loc, nz_loc = NY, NZ

    # allocate primitives
    pr = init_block(nx_loc, ny_loc, nz_loc)
    rng = np.random.default_rng(1234 + rank*777)

    t = 0.0
    step = 0

    # initial BCs
    apply_periodic_yz(pr)
    if rank == size-1: apply_outflow_right_x(pr)
    if rank == 0:      apply_nozzle_left_x(pr, dx, dy, dz, ny_loc, nz_loc,
                                           JET_CENTER[1], JET_CENTER[2], rng)

    # --- DEBUG: JIT warm-up ---
    if DEBUG:
        _ = compute_rhs_muscl(pr, pr.shape[1], pr.shape[2], pr.shape[3], dx, dy, dz)
        comm.Barrier()
        if rank == 0:
            print("[jit] compute_rhs_muscl compiled and first call done.", flush=True)

    # --- time loop ---
    while t < T_END:
        # CFL timestep
        amax_local = max_char_speed(pr, pr.shape[1], pr.shape[2], pr.shape[3])
        amax = comm.allreduce(amax_local, op=MPI.MAX)
        dt = CFL * min(dx, dy, dz) / max(amax, SMALL)
        if t + dt > T_END:
            dt = T_END - t

        # halo exchange + BCs
        exchange_halos(pr, comm, left, right)
        apply_periodic_yz(pr)
        if rank == size-1: apply_outflow_right_x(pr)
        if rank == 0:      apply_nozzle_left_x(pr, dx, dy, dz, ny_loc, nz_loc,
                                               JET_CENTER[1], JET_CENTER[2], rng)

        # advance one step
        pr = step_ssprk2(pr, dx, dy, dz, dt)

        # re-apply BCs after update
        apply_periodic_yz(pr)
        if rank == size-1: apply_outflow_right_x(pr)
        if rank == 0:      apply_nozzle_left_x(pr, dx, dy, dz, ny_loc, nz_loc,
                                               JET_CENTER[1], JET_CENTER[2], rng)

        # update time, step count
        t += dt
        step += 1

        # progress print
        if rank == 0 and (step % PRINT_EVERY == 0 or abs(t - T_END) < 1e-14):
            print(f"[rank0] t={t:.5f} dt={dt:.3e} amax={amax:.3f} step={step}", flush=True)

        # optional DEBUG health checks
        if CHECK_NAN_EVERY > 0 and (step % CHECK_NAN_EVERY == 0):
            bad = (not np.isfinite(pr).all())
            bad_any = comm.allreduce(1 if bad else 0, op=MPI.SUM)
            if bad_any > 0 and rank == 0:
                print("[warn] NaN/Inf detected in primitives!", flush=True)
                if ASSERTS:
                    raise RuntimeError("NaN/Inf in primitives")

        # output + diagnostics
        if step % OUT_EVERY == 0 or abs(t - T_END) < 1e-14:
            fname = os.path.join(RUN_DIR, f"jet3d_rank{rank:04d}_step{step:06d}.npz")
            np.savez(
                fname,
                rho = pr[0], vx=pr[1], vy=pr[2], vz=pr[3], p=pr[4],
                meta = np.array([dx, dy, dz, t], dtype=np.float64),
                comment = "3D SRHD MUSCL block (with ghosts)."
            )
            if rank == 0:
                print(f"[io] wrote {fname}", flush=True)

            global_maxG, inlet_flux = compute_diagnostics_and_write(
                pr, dx, dy, dz, offs[rank], counts, comm, rank,
                step, t, dt, amax
            )
            if rank == 0:
                print(f"[diag] step={step} maxGamma={global_maxG:.3f} "
                      f"inletFlux={inlet_flux:.6e}", flush=True)

    if rank == 0:
        print("Done.", flush=True)

if __name__ == "__main__":
    main()
