#!/usr/bin/env python3
# core/srhd_core.py
# SRHD numerics core: primitives/conserved, MUSCL+HLLE fluxes, SSPRK2.
import numpy as np
import numba as nb

# Module-level parameters set by configure() before first JIT call.
GAMMA = 5.0 / 3.0
P_MAX = 1.0
V_MAX = 0.999
LIMITER_ID = 0  # 0=mc, 1=minmod, 2=vanleer
RECON_ID = 0    # 0=muscl, 1=ppm, 2=weno
RIEMANN_ID = 0  # 0=hlle, 1=hllc
SMALL = 1e-12
N_TRACERS = 0
TRACER_OFFSET = 5

def configure(params):
    """Set global parameters for Numba-compiled kernels."""
    global GAMMA, P_MAX, V_MAX, LIMITER_ID, RECON_ID, RIEMANN_ID, N_TRACERS, TRACER_OFFSET
    GAMMA = float(params.get("GAMMA", GAMMA))
    P_MAX = float(params.get("P_MAX", P_MAX))
    V_MAX = float(params.get("V_MAX", V_MAX))
    limiter = str(params.get("LIMITER", "mc")).lower()
    if limiter == "minmod":
        LIMITER_ID = 1
    elif limiter == "vanleer":
        LIMITER_ID = 2
    else:
        LIMITER_ID = 0
    recon = str(params.get("RECON", "muscl")).lower()
    if recon == "ppm":
        RECON_ID = 1
    elif recon == "weno":
        RECON_ID = 2
    else:
        RECON_ID = 0
    riemann = str(params.get("RIEMANN", "hlle")).lower()
    RIEMANN_ID = 1 if riemann == "hllc" else 0
    N_TRACERS = int(params.get("N_TRACERS", N_TRACERS))
    TRACER_OFFSET = int(params.get("TRACER_OFFSET", TRACER_OFFSET))

# ------------------------
# SRHD helpers (Numba)
# ------------------------
@nb.njit(fastmath=True)
def prim_to_cons(rho, vx, vy, vz, p, gamma=GAMMA):
    v2 = vx*vx + vy*vy + vz*vz
    if v2 >= 1.0:
        v2 = 1.0 - 1e-14
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
    E  = tau + D
    S2 = Sx*Sx + Sy*Sy + Sz*Sz

    # initial guess: ideal-gas-like
    p = (gamma - 1.0) * (E - D)
    if p < SMALL: p = SMALL
    if p > P_MAX: p = P_MAX

    ok = False
    for _ in range(60):
        Wm = E + p                      # Wm = E + p  (our unknown enters here)
        v2 = S2 / (Wm*Wm + SMALL)
        if v2 >= V_MAX*V_MAX:
            v2 = V_MAX*V_MAX - 1e-14
        W  = 1.0 / np.sqrt(1.0 - v2)
        rho= D / max(W, SMALL)
        h  = 1.0 + gamma/(gamma-1.0) * p / max(rho, SMALL)
        w  = rho * h
        # residual: Wm - w W^2 = 0
        f  = Wm - w * W * W

        # simple, safe slope; we damp anyway
        dfdp = 1.0
        dp = -f / dfdp

        # damp update and clamp into [P_MIN, P_MAX]
        if dp >  0.5*p: dp =  0.5*p
        if dp < -0.5*p: dp = -0.5*p
        p_new = p + dp
        if p_new < SMALL: p_new = SMALL
        if p_new > P_MAX: p_new = P_MAX

        if np.abs(dp) < 1e-12 * max(1.0, p_new):
            p = p_new
            ok = True
            break
        p = p_new

    if not ok:
        # fallback: energy-based clamp (very robust)
        p = (gamma - 1.0) * (E - D)
        if p < SMALL: p = SMALL
        if p > P_MAX: p = P_MAX

    # recover velocities with cap
    Wm = E + p
    vx = Sx / max(Wm, SMALL)
    vy = Sy / max(Wm, SMALL)
    vz = Sz / max(Wm, SMALL)

    # cap velocity to V_MAX
    v2 = vx*vx + vy*vy + vz*vz
    vmax2 = V_MAX*V_MAX
    if v2 >= vmax2:
        fac = V_MAX / np.sqrt(v2 + 1e-32)
        vx *= fac; vy *= fac; vz *= fac

    # recompute rho with final Γ
    v2 = vx*vx + vy*vy + vz*vz
    if v2 >= 1.0: v2 = 1.0 - 1e-14
    W  = 1.0 / np.sqrt(1.0 - v2)
    rho= D / max(W, SMALL)
    if rho < SMALL: rho = SMALL

    # final pressure cap (in case)
    if p < SMALL: p = SMALL
    if p > P_MAX: p = P_MAX

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

@nb.njit(fastmath=True)
def hlle_scalar(qL, qR, fL, fR, sL, sR):
    if sL >= 0.0:
        return fL
    if sR <= 0.0:
        return fR
    return (sR*fL - sL*fR + sL*sR*(qR - qL)) / (sR - sL + SMALL)

@nb.njit(fastmath=True)
def hllc_flux_x(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    DL, SxL, SyL, SzL, tauL = UL
    DR, SxR, SyR, SzR, tauR = UR
    denom = (DL*(sL - vL) - DR*(sR - vR)) + SMALL
    sM = (pR - pL + SxL*(sL - vL) - SxR*(sR - vR)) / denom
    pStar = 0.5*(pL + pR + DL*(sL - vL)*(sM - vL) + DR*(sR - vR)*(sM - vR))

    if sL >= 0.0:
        return FL
    if sR <= 0.0:
        return FR

    if sM >= 0.0:
        fac = (sL - vL) / (sL - sM + SMALL)
        Dst = DL * fac
        Sxst = SxL * fac + (pStar - pL) / (sL - sM + SMALL)
        Syst = SyL * fac
        Szst = SzL * fac
        taust = tauL * fac + (pStar*sM - pL*vL) / (sL - sM + SMALL)
        Ust = np.array([Dst, Sxst, Syst, Szst, taust])
        return FL + sL*(Ust - UL)
    else:
        fac = (sR - vR) / (sR - sM + SMALL)
        Dst = DR * fac
        Sxst = SxR * fac + (pStar - pR) / (sR - sM + SMALL)
        Syst = SyR * fac
        Szst = SzR * fac
        taust = tauR * fac + (pStar*sM - pR*vR) / (sR - sM + SMALL)
        Ust = np.array([Dst, Sxst, Syst, Szst, taust])
        return FR + sR*(Ust - UR)

@nb.njit(fastmath=True)
def hllc_flux_y(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    DL, SxL, SyL, SzL, tauL = UL
    DR, SxR, SyR, SzR, tauR = UR
    denom = (DL*(sL - vL) - DR*(sR - vR)) + SMALL
    sM = (pR - pL + SyL*(sL - vL) - SyR*(sR - vR)) / denom
    pStar = 0.5*(pL + pR + DL*(sL - vL)*(sM - vL) + DR*(sR - vR)*(sM - vR))

    if sL >= 0.0:
        return FL
    if sR <= 0.0:
        return FR

    if sM >= 0.0:
        fac = (sL - vL) / (sL - sM + SMALL)
        Dst = DL * fac
        Syst = SyL * fac + (pStar - pL) / (sL - sM + SMALL)
        Sxst = SxL * fac
        Szst = SzL * fac
        taust = tauL * fac + (pStar*sM - pL*vL) / (sL - sM + SMALL)
        Ust = np.array([Dst, Sxst, Syst, Szst, taust])
        return FL + sL*(Ust - UL)
    else:
        fac = (sR - vR) / (sR - sM + SMALL)
        Dst = DR * fac
        Syst = SyR * fac + (pStar - pR) / (sR - sM + SMALL)
        Sxst = SxR * fac
        Szst = SzR * fac
        taust = tauR * fac + (pStar*sM - pR*vR) / (sR - sM + SMALL)
        Ust = np.array([Dst, Sxst, Syst, Szst, taust])
        return FR + sR*(Ust - UR)

@nb.njit(fastmath=True)
def hllc_flux_z(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    DL, SxL, SyL, SzL, tauL = UL
    DR, SxR, SyR, SzR, tauR = UR
    denom = (DL*(sL - vL) - DR*(sR - vR)) + SMALL
    sM = (pR - pL + SzL*(sL - vL) - SzR*(sR - vR)) / denom
    pStar = 0.5*(pL + pR + DL*(sL - vL)*(sM - vL) + DR*(sR - vR)*(sM - vR))

    if sL >= 0.0:
        return FL
    if sR <= 0.0:
        return FR

    if sM >= 0.0:
        fac = (sL - vL) / (sL - sM + SMALL)
        Dst = DL * fac
        Szst = SzL * fac + (pStar - pL) / (sL - sM + SMALL)
        Sxst = SxL * fac
        Syst = SyL * fac
        taust = tauL * fac + (pStar*sM - pL*vL) / (sL - sM + SMALL)
        Ust = np.array([Dst, Sxst, Syst, Szst, taust])
        return FL + sL*(Ust - UL)
    else:
        fac = (sR - vR) / (sR - sM + SMALL)
        Dst = DR * fac
        Szst = SzR * fac + (pStar - pR) / (sR - sM + SMALL)
        Sxst = SxR * fac
        Syst = SyR * fac
        taust = tauR * fac + (pStar*sM - pR*vR) / (sR - sM + SMALL)
        Ust = np.array([Dst, Sxst, Syst, Szst, taust])
        return FR + sR*(Ust - UR)

@nb.njit(fastmath=True)
def riemann_flux_x(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    if RIEMANN_ID == 1:
        return hllc_flux_x(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR)
    return hlle(UL, UR, FL, FR, sL, sR)

@nb.njit(fastmath=True)
def riemann_flux_y(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    if RIEMANN_ID == 1:
        return hllc_flux_y(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR)
    return hlle(UL, UR, FL, FR, sL, sR)

@nb.njit(fastmath=True)
def riemann_flux_z(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    if RIEMANN_ID == 1:
        return hllc_flux_z(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR)
    return hlle(UL, UR, FL, FR, sL, sR)

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
    if LIMITER_ID == 1:
        return minmod(dqL, dqR)
    if LIMITER_ID == 2:
        if dqL*dqR <= 0.0:
            return 0.0
        denom = dqL + dqR
        if denom == 0.0:
            return 0.0
        return (2.0*dqL*dqR) / denom
    mm = minmod(dqL, dqR)
    return minmod(0.5*(dqL + dqR), 2.0*mm)

@nb.njit(fastmath=True)
def ppm_reconstruct(q_im2, q_im1, q_i, q_ip1, q_ip2):
    # PPM interface values with monotonicity clamp
    qL = (7.0/12.0)*(q_im1 + q_i)   - (1.0/12.0)*(q_im2 + q_ip1)
    qR = (7.0/12.0)*(q_i   + q_ip1) - (1.0/12.0)*(q_im1 + q_ip2)
    qmax = max(q_im1, q_i, q_ip1)
    qmin = min(q_im1, q_i, q_ip1)
    if qL > qmax: qL = qmax
    if qL < qmin: qL = qmin
    if qR > qmax: qR = qmax
    if qR < qmin: qR = qmin
    if (qR - q_i) * (q_i - qL) <= 0.0:
        qL = q_i
        qR = q_i
    return qL, qR

@nb.njit(fastmath=True)
def weno5_left(q_im2, q_im1, q_i, q_ip1, q_ip2):
    eps = 1e-12
    IS0 = (13.0/12.0)*(q_im2 - 2.0*q_im1 + q_i)**2 + 0.25*(q_im2 - 4.0*q_im1 + 3.0*q_i)**2
    IS1 = (13.0/12.0)*(q_im1 - 2.0*q_i + q_ip1)**2 + 0.25*(q_im1 - q_ip1)**2
    IS2 = (13.0/12.0)*(q_i - 2.0*q_ip1 + q_ip2)**2 + 0.25*(3.0*q_i - 4.0*q_ip1 + q_ip2)**2
    a0 = 0.1 / (eps + IS0)**2
    a1 = 0.6 / (eps + IS1)**2
    a2 = 0.3 / (eps + IS2)**2
    s = a0 + a1 + a2
    w0 = a0 / s
    w1 = a1 / s
    w2 = a2 / s
    p0 = (1.0/3.0)*q_im2 - (7.0/6.0)*q_im1 + (11.0/6.0)*q_i
    p1 = (-1.0/6.0)*q_im1 + (5.0/6.0)*q_i + (1.0/3.0)*q_ip1
    p2 = (1.0/3.0)*q_i + (5.0/6.0)*q_ip1 - (1.0/6.0)*q_ip2
    return w0*p0 + w1*p1 + w2*p2

@nb.njit(fastmath=True)
def weno5_right(q_ip2, q_ip1, q_i, q_im1, q_im2):
    eps = 1e-12
    IS0 = (13.0/12.0)*(q_ip2 - 2.0*q_ip1 + q_i)**2 + 0.25*(q_ip2 - 4.0*q_ip1 + 3.0*q_i)**2
    IS1 = (13.0/12.0)*(q_ip1 - 2.0*q_i + q_im1)**2 + 0.25*(q_ip1 - q_im1)**2
    IS2 = (13.0/12.0)*(q_i - 2.0*q_im1 + q_im2)**2 + 0.25*(3.0*q_i - 4.0*q_im1 + q_im2)**2
    a0 = 0.1 / (eps + IS0)**2
    a1 = 0.6 / (eps + IS1)**2
    a2 = 0.3 / (eps + IS2)**2
    s = a0 + a1 + a2
    w0 = a0 / s
    w1 = a1 / s
    w2 = a2 / s
    p0 = (1.0/3.0)*q_ip2 - (7.0/6.0)*q_ip1 + (11.0/6.0)*q_i
    p1 = (-1.0/6.0)*q_ip1 + (5.0/6.0)*q_i + (1.0/3.0)*q_im1
    p2 = (1.0/3.0)*q_i + (5.0/6.0)*q_im1 - (1.0/6.0)*q_im2
    return w0*p0 + w1*p1 + w2*p2

@nb.njit(fastmath=True)
def floor_prim(rho, vx, vy, vz, p):
    if rho < SMALL: rho = SMALL
    if p   < SMALL: p   = SMALL
    # caps
    if p   > P_MAX: p   = P_MAX
    # limit |v| < V_MAX
    v2 = vx*vx + vy*vy + vz*vz
    vmax2 = V_MAX*V_MAX
    if v2 >= vmax2:
        fac = V_MAX / np.sqrt(v2 + 1e-32)
        vx *= fac; vy *= fac; vz *= fac
    return rho, vx, vy, vz, p

@nb.njit(fastmath=True)
def add_pi_lr(pr, pL, pR, iL, jL, kL, iR, jR, kR):
    if TRACER_OFFSET > 5:
        pL += pr[5, iL, jL, kL]
        pR += pr[5, iR, jR, kR]
    return pL, pR

@nb.njit(fastmath=True)
def get_diss_vars(pr, i, j, k):
    if pr.shape[0] >= 15:
        pi = pr[5, i, j, k]
        pixx = pr[6, i, j, k]
        piyy = pr[7, i, j, k]
        pizz = pr[8, i, j, k]
        pixy = pr[9, i, j, k]
        pixz = pr[10, i, j, k]
        piyz = pr[11, i, j, k]
        qx = pr[12, i, j, k]
        qy = pr[13, i, j, k]
        qz = pr[14, i, j, k]
        return pi, pixx, piyy, pizz, pixy, pixz, piyz, qx, qy, qz
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

@nb.njit(fastmath=True)
def add_diss_flux_x(F, vx, vy, vz, pi, pixx, piyy, pizz, pixy, pixz, piyz, qx, qy, qz):
    out = F.copy()
    out[1] += pi + pixx
    out[2] += pixy
    out[3] += pixz
    out[4] += vx*(pi + pixx) + vy*pixy + vz*pixz + qx
    return out

@nb.njit(fastmath=True)
def add_diss_flux_y(F, vx, vy, vz, pi, pixx, piyy, pizz, pixy, pixz, piyz, qx, qy, qz):
    out = F.copy()
    out[1] += pixy
    out[2] += pi + piyy
    out[3] += piyz
    out[4] += vx*pixy + vy*(pi + piyy) + vz*piyz + qy
    return out

@nb.njit(fastmath=True)
def add_diss_flux_z(F, vx, vy, vz, pi, pixx, piyy, pizz, pixy, pixz, piyz, qx, qy, qz):
    out = F.copy()
    out[1] += pixz
    out[2] += piyz
    out[3] += pi + pizz
    out[4] += vx*pixz + vy*piyz + vz*(pi + pizz) + qz
    return out

@nb.njit(fastmath=True)
def apply_diss_flux_x(pr, F, vx, vy, vz, i, j, k):
    if pr.shape[0] < 15:
        return F
    pi, pixx, piyy, pizz, pixy, pixz, piyz, qx, qy, qz = get_diss_vars(pr, i, j, k)
    return add_diss_flux_x(F, vx, vy, vz, pi, pixx, piyy, pizz, pixy, pixz, piyz, qx, qy, qz)

@nb.njit(fastmath=True)
def apply_diss_flux_y(pr, F, vx, vy, vz, i, j, k):
    if pr.shape[0] < 15:
        return F
    pi, pixx, piyy, pizz, pixy, pixz, piyz, qx, qy, qz = get_diss_vars(pr, i, j, k)
    return add_diss_flux_y(F, vx, vy, vz, pi, pixx, piyy, pizz, pixy, pixz, piyz, qx, qy, qz)

@nb.njit(fastmath=True)
def apply_diss_flux_z(pr, F, vx, vy, vz, i, j, k):
    if pr.shape[0] < 15:
        return F
    pi, pixx, piyy, pizz, pixy, pixz, piyz, qx, qy, qz = get_diss_vars(pr, i, j, k)
    return add_diss_flux_z(F, vx, vy, vz, pi, pixx, piyy, pizz, pixy, pixz, piyz, qx, qy, qz)

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
    rhs = np.zeros((5 + N_TRACERS, nx, ny, nz))

    i0, i1 = 2, nx-3
    j0, j1 = 2, ny-3
    k0, k1 = 2, nz-3

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
                pL, pR = add_pi_lr(pr, pL, pR, i-1, j, k, i, j, k)

                rL,vxL,vyL,vzL,pL = floor_prim(rL,vxL,vyL,vzL,pL)
                rR,vxR,vyR,vzR,pR = floor_prim(rR,vxR,vyR,vzR,pR)

                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL))
                UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_x(rL,vxL,vyL,vzL,pL)
                FR = flux_x(rR,vxR,vyR,vzR,pR)
                FL = apply_diss_flux_x(pr, FL, vxL, vyL, vzL, i-1, j, k)
                FR = apply_diss_flux_x(pr, FR, vxR, vyR, vzR, i, j, k)
                csL= sound_speed(rL,pL); csR= sound_speed(rR,pR)
                lmL, lpL = eig_speeds(vxL, csL)
                lmR, lpR = eig_speeds(vxR, csR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                FxL = riemann_flux_x(UL, UR, FL, FR, sL, sR, pL, pR, vxL, vxR)
                if N_TRACERS > 0:
                    FxL_tr = np.empty(N_TRACERS)
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        tLq = weno5_left(pr[idx,i-3,j,k], pr[idx,i-2,j,k], pr[idx,i-1,j,k], pr[idx,i,j,k], pr[idx,i+1,j,k])
                        tRq = weno5_right(pr[idx,i+2,j,k], pr[idx,i+1,j,k], pr[idx,i,j,k], pr[idx,i-1,j,k], pr[idx,i-2,j,k])
                        FxL_tr[t] = hlle_scalar(tLq, tRq, tLq*vxL, tRq*vxR, sL, sR)
                if N_TRACERS > 0:
                    FxL_tr = np.empty(N_TRACERS)
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        l_im1, r_im1 = ppm_reconstruct(pr[idx,i-3,j,k], pr[idx,i-2,j,k], pr[idx,i-1,j,k],
                                                       pr[idx,i-0,j,k], pr[idx,i+1,j,k])
                        l_i, r_i = ppm_reconstruct(pr[idx,i-2,j,k], pr[idx,i-1,j,k], pr[idx,i,j,k],
                                                   pr[idx,i+1,j,k], pr[idx,i+2,j,k])
                        tLq = r_im1
                        tRq = l_i
                        FxL_tr[t] = hlle_scalar(tLq, tRq, tLq*vxL, tRq*vxR, sL, sR)
                if N_TRACERS > 0:
                    FxL_tr = np.empty(N_TRACERS)
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        dqL_t = pr[idx, i-1, j, k] - pr[idx, i-2, j, k]
                        dqR_t = pr[idx, i,   j, k] - pr[idx, i-1, j, k]
                        slL_t = mc_lim(dqL_t, dqR_t)
                        dqL_t = pr[idx, i,   j, k] - pr[idx, i-1, j, k]
                        dqR_t = pr[idx, i+1, j, k] - pr[idx, i,   j, k]
                        slR_t = mc_lim(dqL_t, dqR_t)
                        tL = pr[idx, i-1, j, k] + 0.5*slL_t
                        tR = pr[idx, i,   j, k] - 0.5*slR_t
                        FxL_tr[t] = hlle_scalar(tL, tR, tL*vxL, tR*vxR, sL, sR)

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
                pL, pR = add_pi_lr(pr, pL, pR, i, j, k, i+1, j, k)

                rL,vxL,vyL,vzL,pL = floor_prim(rL,vxL,vyL,vzL,pL)
                rR,vxR,vyR,vzR,pR = floor_prim(rR,vxR,vyR,vzR,pR)

                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL))
                UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_x(rL,vxL,vyL,vzL,pL)
                FR = flux_x(rR,vxR,vyR,vzR,pR)
                FL = apply_diss_flux_x(pr, FL, vxL, vyL, vzL, i, j, k)
                FR = apply_diss_flux_x(pr, FR, vxR, vyR, vzR, i+1, j, k)
                csL= sound_speed(rL,pL); csR= sound_speed(rR,pR)
                lmL, lpL = eig_speeds(vxL, csL)
                lmR, lpR = eig_speeds(vxR, csR)
                sL2 = min(lmL, lmR); sR2 = max(lpL, lpR)
                FxR = riemann_flux_x(UL, UR, FL, FR, sL2, sR2, pL, pR, vxL, vxR)
                if N_TRACERS > 0:
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        tLq = weno5_left(pr[idx,i-2,j,k], pr[idx,i-1,j,k], pr[idx,i,j,k], pr[idx,i+1,j,k], pr[idx,i+2,j,k])
                        tRq = weno5_right(pr[idx,i+3,j,k], pr[idx,i+2,j,k], pr[idx,i+1,j,k], pr[idx,i,j,k], pr[idx,i-1,j,k])
                        FxR_tr = hlle_scalar(tLq, tRq, tLq*vxL, tRq*vxR, sL2, sR2)
                        rhs[5 + t, i, j, k] -= (FxR_tr - FxL_tr[t]) / dx
                if N_TRACERS > 0:
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        l_i, r_i = ppm_reconstruct(pr[idx,i-2,j,k], pr[idx,i-1,j,k], pr[idx,i,j,k],
                                                   pr[idx,i+1,j,k], pr[idx,i+2,j,k])
                        l_ip1, r_ip1 = ppm_reconstruct(pr[idx,i-1,j,k], pr[idx,i,j,k], pr[idx,i+1,j,k],
                                                       pr[idx,i+2,j,k], pr[idx,i+3,j,k])
                        tLq = r_i
                        tRq = l_ip1
                        FxR_tr = hlle_scalar(tLq, tRq, tLq*vxL, tRq*vxR, sL2, sR2)
                        rhs[5 + t, i, j, k] -= (FxR_tr - FxL_tr[t]) / dx
                if N_TRACERS > 0:
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        dqL_t = pr[idx, i+1, j, k] - pr[idx, i,   j, k]
                        dqR_t = pr[idx, i+2, j, k] - pr[idx, i+1, j, k]
                        slRp1_t = mc_lim(dqL_t, dqR_t)
                        dqL_t = pr[idx, i,   j, k] - pr[idx, i-1, j, k]
                        dqR_t = pr[idx, i+1, j, k] - pr[idx, i,   j, k]
                        slR_t = mc_lim(dqL_t, dqR_t)
                        tL = pr[idx, i,   j, k] + 0.5*slR_t
                        tR = pr[idx, i+1, j, k] - 0.5*slRp1_t
                        FxR_tr = hlle_scalar(tL, tR, tL*vxL, tR*vxR, sL2, sR2)
                        rhs[5 + t, i, j, k] -= (FxR_tr - FxL_tr[t]) / dx

                rhs[0:5,i,j,k] -= (FxR - FxL)/dx

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
                pL, pR = add_pi_lr(pr, pL, pR, i, j-1, k, i, j, k)

                rL,vxL,vyL,vzL,pL = floor_prim(rL,vxL,vyL,vzL,pL)
                rR,vxR,vyR,vzR,pR = floor_prim(rR,vxR,vyR,vzR,pR)

                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL))
                UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_y(rL,vxL,vyL,vzL,pL)
                FR = flux_y(rR,vxR,vyR,vzR,pR)
                FL = apply_diss_flux_y(pr, FL, vxL, vyL, vzL, i, j-1, k)
                FR = apply_diss_flux_y(pr, FR, vxR, vyR, vzR, i, j, k)
                csL= sound_speed(rL,pL); csR= sound_speed(rR,pR)
                lmL, lpL = eig_speeds(vyL, csL)
                lmR, lpR = eig_speeds(vyR, csR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                FyD = riemann_flux_y(UL, UR, FL, FR, sL, sR, pL, pR, vyL, vyR)

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
                pL, pR = add_pi_lr(pr, pL, pR, i, j, k, i, j+1, k)

                rL,vxL,vyL,vzL,pL = floor_prim(rL,vxL,vyL,vzL,pL)
                rR,vxR,vyR,vzR,pR = floor_prim(rR,vxR,vyR,vzR,pR)

                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL))
                UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_y(rL,vxL,vyL,vzL,pL)
                FR = flux_y(rR,vxR,vyR,vzR,pR)
                FL = apply_diss_flux_y(pr, FL, vxL, vyL, vzL, i, j, k)
                FR = apply_diss_flux_y(pr, FR, vxR, vyR, vzR, i, j+1, k)
                csL= sound_speed(rL,pL); csR= sound_speed(rR,pR)
                lmL, lpL = eig_speeds(vyL, csL)
                lmR, lpR = eig_speeds(vyR, csR)
                sL2 = min(lmL, lmR); sR2 = max(lpL, lpR)
                FyU = riemann_flux_y(UL, UR, FL, FR, sL2, sR2, pL, pR, vyL, vyR)

                rhs[0:5,i,j,k] -= (FyU - FyD)/dy

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
                pL, pR = add_pi_lr(pr, pL, pR, i, j, k-1, i, j, k)

                rL,vxL,vyL,vzL,pL = floor_prim(rL,vxL,vyL,vzL,pL)
                rR,vxR,vyR,vzR,pR = floor_prim(rR,vxR,vyR,vzR,pR)

                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL))
                UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_z(rL,vxL,vyL,vzL,pL)
                FR = flux_z(rR,vxR,vyR,vzR,pR)
                FL = apply_diss_flux_z(pr, FL, vxL, vyL, vzL, i, j, k-1)
                FR = apply_diss_flux_z(pr, FR, vxR, vyR, vzR, i, j, k)
                csL= sound_speed(rL,pL); csR= sound_speed(rR,pR)
                lmL, lpL = eig_speeds(vzL, csL)
                lmR, lpR = eig_speeds(vzR, csR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                FzB = riemann_flux_z(UL, UR, FL, FR, sL, sR, pL, pR, vzL, vzR)
                if N_TRACERS > 0:
                    FzB_tr = np.empty(N_TRACERS)
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        dqL_t = pr[idx, i, j, k-1] - pr[idx, i, j, k-2]
                        dqR_t = pr[idx, i, j, k]   - pr[idx, i, j, k-1]
                        slL_t = mc_lim(dqL_t, dqR_t)
                        dqL_t = pr[idx, i, j, k]   - pr[idx, i, j, k-1]
                        dqR_t = pr[idx, i, j, k+1] - pr[idx, i, j, k]
                        slR_t = mc_lim(dqL_t, dqR_t)
                        tLq = pr[idx, i, j, k-1] + 0.5*slL_t
                        tRq = pr[idx, i, j, k]   - 0.5*slR_t
                        FzB_tr[t] = hlle_scalar(tLq, tRq, tLq*vzL, tRq*vzR, sL, sR)

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
                pL, pR = add_pi_lr(pr, pL, pR, i, j, k, i, j, k+1)

                rL,vxL,vyL,vzL,pL = floor_prim(rL,vxL,vyL,vzL,pL)
                rR,vxR,vyR,vzR,pR = floor_prim(rR,vxR,vyR,vzR,pR)

                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL))
                UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_z(rL,vxL,vyL,vzL,pL)
                FR = flux_z(rR,vxR,vyR,vzR,pR)
                FL = apply_diss_flux_z(pr, FL, vxL, vyL, vzL, i, j, k)
                FR = apply_diss_flux_z(pr, FR, vxR, vyR, vzR, i, j, k+1)
                csL= sound_speed(rL,pL); csR= sound_speed(rR,pR)
                lmL, lpL = eig_speeds(vzL, csL)
                lmR, lpR = eig_speeds(vzR, csR)
                sL2 = min(lmL, lmR); sR2 = max(lpL, lpR)
                FzF = riemann_flux_z(UL, UR, FL, FR, sL2, sR2, pL, pR, vzL, vzR)
                if N_TRACERS > 0:
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        dqL_t = pr[idx, i, j, k+1] - pr[idx, i, j, k]
                        dqR_t = pr[idx, i, j, k+2] - pr[idx, i, j, k+1]
                        slRp1_t = mc_lim(dqL_t, dqR_t)
                        dqL_t = pr[idx, i, j, k]   - pr[idx, i, j, k-1]
                        dqR_t = pr[idx, i, j, k+1] - pr[idx, i, j, k]
                        slR_t = mc_lim(dqL_t, dqR_t)
                        tLq = pr[idx, i, j, k]   + 0.5*slR_t
                        tRq = pr[idx, i, j, k+1] - 0.5*slRp1_t
                        FzF_tr = hlle_scalar(tLq, tRq, tLq*vzL, tRq*vzR, sL2, sR2)
                        rhs[5 + t, i, j, k] -= (FzF_tr - FzB_tr[t]) / dz

                rhs[0:5,i,j,k] -= (FzF - FzB)/dz

    return rhs

@nb.njit(parallel=True, fastmath=True)
def compute_rhs_ppm(pr, nx, ny, nz, dx, dy, dz):
    rhs = np.zeros((5 + N_TRACERS, nx, ny, nz))
    i0, i1 = 3, nx-3
    j0, j1 = 3, ny-3
    k0, k1 = 3, nz-3

    for i in range(i0, i1):
        for j in range(j0, j1):
            for k in range(k0, k1):
                qL = np.empty(5)
                qR = np.empty(5)
                qL2 = np.empty(5)
                qR2 = np.empty(5)

                # x faces: i-1/2 and i+1/2
                for v in range(5):
                    l_im1, r_im1 = ppm_reconstruct(pr[v,i-3,j,k], pr[v,i-2,j,k], pr[v,i-1,j,k],
                                                   pr[v,i-0,j,k], pr[v,i+1,j,k])
                    l_i, r_i = ppm_reconstruct(pr[v,i-2,j,k], pr[v,i-1,j,k], pr[v,i,j,k],
                                               pr[v,i+1,j,k], pr[v,i+2,j,k])
                    l_ip1, r_ip1 = ppm_reconstruct(pr[v,i-1,j,k], pr[v,i,j,k], pr[v,i+1,j,k],
                                                   pr[v,i+2,j,k], pr[v,i+3,j,k])
                    qL[v] = r_im1
                    qR[v] = l_i
                    qL2[v] = r_i
                    qR2[v] = l_ip1

                rL,vxL,vyL,vzL,pL = floor_prim(qL[0], qL[1], qL[2], qL[3], qL[4])
                rR,vxR,vyR,vzR,pR = floor_prim(qR[0], qR[1], qR[2], qR[3], qR[4])
                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_x(rL,vxL,vyL,vzL,pL); FR = flux_x(rR,vxR,vyR,vzR,pR)
                FL = apply_diss_flux_x(pr, FL, vxL, vyL, vzL, i-1, j, k)
                FR = apply_diss_flux_x(pr, FR, vxR, vyR, vzR, i, j, k)
                FL = apply_diss_flux_x(pr, FL, vxL, vyL, vzL, i-1, j, k)
                FR = apply_diss_flux_x(pr, FR, vxR, vyR, vzR, i, j, k)
                csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
                lamLm, lamLp = eig_speeds(vxL, csL); lamRm, lamRp = eig_speeds(vxR, csR)
                sL = min(lamLm, lamRm); sR = max(lamLp, lamRp)
                FxL = riemann_flux_x(UL, UR, FL, FR, sL, sR, pL, pR, vxL, vxR)

                rL,vxL,vyL,vzL,pL = floor_prim(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4])
                rR,vxR,vyR,vzR,pR = floor_prim(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4])
                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_x(rL,vxL,vyL,vzL,pL); FR = flux_x(rR,vxR,vyR,vzR,pR)
                FL = apply_diss_flux_x(pr, FL, vxL, vyL, vzL, i, j, k)
                FR = apply_diss_flux_x(pr, FR, vxR, vyR, vzR, i+1, j, k)
                FL = apply_diss_flux_x(pr, FL, vxL, vyL, vzL, i, j, k)
                FR = apply_diss_flux_x(pr, FR, vxR, vyR, vzR, i+1, j, k)
                csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
                lamLm, lamLp = eig_speeds(vxL, csL); lamRm, lamRp = eig_speeds(vxR, csR)
                sL2 = min(lamLm, lamRm); sR2 = max(lamLp, lamRp)
                FxR = riemann_flux_x(UL, UR, FL, FR, sL2, sR2, pL, pR, vxL, vxR)

                rhs[0:5,i,j,k] -= (FxR - FxL)/dx

                # y faces
                for v in range(5):
                    l_jm1, r_jm1 = ppm_reconstruct(pr[v,i,j-3,k], pr[v,i,j-2,k], pr[v,i,j-1,k],
                                                   pr[v,i,j-0,k], pr[v,i,j+1,k])
                    l_j, r_j = ppm_reconstruct(pr[v,i,j-2,k], pr[v,i,j-1,k], pr[v,i,j,k],
                                               pr[v,i,j+1,k], pr[v,i,j+2,k])
                    l_jp1, r_jp1 = ppm_reconstruct(pr[v,i,j-1,k], pr[v,i,j,k], pr[v,i,j+1,k],
                                                   pr[v,i,j+2,k], pr[v,i,j+3,k])
                    qL[v] = r_jm1
                    qR[v] = l_j
                    qL2[v] = r_j
                    qR2[v] = l_jp1

                rL,vxL,vyL,vzL,pL = floor_prim(qL[0], qL[1], qL[2], qL[3], qL[4])
                rR,vxR,vyR,vzR,pR = floor_prim(qR[0], qR[1], qR[2], qR[3], qR[4])
                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_y(rL,vxL,vyL,vzL,pL); FR = flux_y(rR,vxR,vyR,vzR,pR)
                FL = apply_diss_flux_y(pr, FL, vxL, vyL, vzL, i, j-1, k)
                FR = apply_diss_flux_y(pr, FR, vxR, vyR, vzR, i, j, k)
                FL = apply_diss_flux_y(pr, FL, vxL, vyL, vzL, i, j-1, k)
                FR = apply_diss_flux_y(pr, FR, vxR, vyR, vzR, i, j, k)
                csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
                lamLm, lamLp = eig_speeds(vyL, csL); lamRm, lamRp = eig_speeds(vyR, csR)
                tL = min(lamLm, lamRm); tR = max(lamLp, lamRp)
                FyD = riemann_flux_y(UL, UR, FL, FR, tL, tR, pL, pR, vyL, vyR)
                if N_TRACERS > 0:
                    FyD_tr = np.empty(N_TRACERS)
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        tLq = weno5_left(pr[idx,i,j-3,k], pr[idx,i,j-2,k], pr[idx,i,j-1,k], pr[idx,i,j,k], pr[idx,i,j+1,k])
                        tRq = weno5_right(pr[idx,i,j+2,k], pr[idx,i,j+1,k], pr[idx,i,j,k], pr[idx,i,j-1,k], pr[idx,i,j-2,k])
                        FyD_tr[t] = hlle_scalar(tLq, tRq, tLq*vyL, tRq*vyR, tL, tR)
                if N_TRACERS > 0:
                    FyD_tr = np.empty(N_TRACERS)
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        l_jm1, r_jm1 = ppm_reconstruct(pr[idx,i,j-3,k], pr[idx,i,j-2,k], pr[idx,i,j-1,k],
                                                       pr[idx,i,j-0,k], pr[idx,i,j+1,k])
                        l_j, r_j = ppm_reconstruct(pr[idx,i,j-2,k], pr[idx,i,j-1,k], pr[idx,i,j,k],
                                                   pr[idx,i,j+1,k], pr[idx,i,j+2,k])
                        tLq = r_jm1
                        tRq = l_j
                        FyD_tr[t] = hlle_scalar(tLq, tRq, tLq*vyL, tRq*vyR, tL, tR)
                if N_TRACERS > 0:
                    FyD_tr = np.empty(N_TRACERS)
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        dqL_t = pr[idx, i, j-1, k] - pr[idx, i, j-2, k]
                        dqR_t = pr[idx, i, j,   k] - pr[idx, i, j-1, k]
                        slL_t = mc_lim(dqL_t, dqR_t)
                        dqL_t = pr[idx, i, j,   k] - pr[idx, i, j-1, k]
                        dqR_t = pr[idx, i, j+1, k] - pr[idx, i, j,   k]
                        slR_t = mc_lim(dqL_t, dqR_t)
                        tLq = pr[idx, i, j-1, k] + 0.5*slL_t
                        tRq = pr[idx, i, j,   k] - 0.5*slR_t
                        FyD_tr[t] = hlle_scalar(tLq, tRq, tLq*vyL, tRq*vyR, tL, tR)

                rL,vxL,vyL,vzL,pL = floor_prim(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4])
                rR,vxR,vyR,vzR,pR = floor_prim(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4])
                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_y(rL,vxL,vyL,vzL,pL); FR = flux_y(rR,vxR,vyR,vzR,pR)
                FL = apply_diss_flux_y(pr, FL, vxL, vyL, vzL, i, j, k)
                FR = apply_diss_flux_y(pr, FR, vxR, vyR, vzR, i, j+1, k)
                FL = apply_diss_flux_y(pr, FL, vxL, vyL, vzL, i, j, k)
                FR = apply_diss_flux_y(pr, FR, vxR, vyR, vzR, i, j+1, k)
                csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
                lamLm, lamLp = eig_speeds(vyL, csL); lamRm, lamRp = eig_speeds(vyR, csR)
                tL2 = min(lamLm, lamRm); tR2 = max(lamLp, lamRp)
                FyU = riemann_flux_y(UL, UR, FL, FR, tL2, tR2, pL, pR, vyL, vyR)
                if N_TRACERS > 0:
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        tLq = weno5_left(pr[idx,i,j-2,k], pr[idx,i,j-1,k], pr[idx,i,j,k], pr[idx,i,j+1,k], pr[idx,i,j+2,k])
                        tRq = weno5_right(pr[idx,i,j+3,k], pr[idx,i,j+2,k], pr[idx,i,j+1,k], pr[idx,i,j,k], pr[idx,i,j-1,k])
                        FyU_tr = hlle_scalar(tLq, tRq, tLq*vyL, tRq*vyR, tL2, tR2)
                        rhs[5 + t, i, j, k] -= (FyU_tr - FyD_tr[t]) / dy
                if N_TRACERS > 0:
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        l_j, r_j = ppm_reconstruct(pr[idx,i,j-2,k], pr[idx,i,j-1,k], pr[idx,i,j,k],
                                                   pr[idx,i,j+1,k], pr[idx,i,j+2,k])
                        l_jp1, r_jp1 = ppm_reconstruct(pr[idx,i,j-1,k], pr[idx,i,j,k], pr[idx,i,j+1,k],
                                                       pr[idx,i,j+2,k], pr[idx,i,j+3,k])
                        tLq = r_j
                        tRq = l_jp1
                        FyU_tr = hlle_scalar(tLq, tRq, tLq*vyL, tRq*vyR, tL2, tR2)
                        rhs[5 + t, i, j, k] -= (FyU_tr - FyD_tr[t]) / dy
                if N_TRACERS > 0:
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        dqL_t = pr[idx, i, j+1, k] - pr[idx, i, j,   k]
                        dqR_t = pr[idx, i, j+2, k] - pr[idx, i, j+1, k]
                        slRp1_t = mc_lim(dqL_t, dqR_t)
                        dqL_t = pr[idx, i, j,   k] - pr[idx, i, j-1, k]
                        dqR_t = pr[idx, i, j+1, k] - pr[idx, i, j,   k]
                        slR_t = mc_lim(dqL_t, dqR_t)
                        tLq = pr[idx, i, j,   k] + 0.5*slR_t
                        tRq = pr[idx, i, j+1, k] - 0.5*slRp1_t
                        FyU_tr = hlle_scalar(tLq, tRq, tLq*vyL, tRq*vyR, tL2, tR2)
                        rhs[5 + t, i, j, k] -= (FyU_tr - FyD_tr[t]) / dy

                rhs[0:5,i,j,k] -= (FyU - FyD)/dy

                # z faces
                for v in range(5):
                    l_km1, r_km1 = ppm_reconstruct(pr[v,i,j,k-3], pr[v,i,j,k-2], pr[v,i,j,k-1],
                                                   pr[v,i,j,k-0], pr[v,i,j,k+1])
                    l_k, r_k = ppm_reconstruct(pr[v,i,j,k-2], pr[v,i,j,k-1], pr[v,i,j,k],
                                               pr[v,i,j,k+1], pr[v,i,j,k+2])
                    l_kp1, r_kp1 = ppm_reconstruct(pr[v,i,j,k-1], pr[v,i,j,k], pr[v,i,j,k+1],
                                                   pr[v,i,j,k+2], pr[v,i,j,k+3])
                    qL[v] = r_km1
                    qR[v] = l_k
                    qL2[v] = r_k
                    qR2[v] = l_kp1

                rL,vxL,vyL,vzL,pL = floor_prim(qL[0], qL[1], qL[2], qL[3], qL[4])
                rR,vxR,vyR,vzR,pR = floor_prim(qR[0], qR[1], qR[2], qR[3], qR[4])
                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_z(rL,vxL,vyL,vzL,pL); FR = flux_z(rR,vxR,vyR,vzR,pR)
                FL = apply_diss_flux_z(pr, FL, vxL, vyL, vzL, i, j, k-1)
                FR = apply_diss_flux_z(pr, FR, vxR, vyR, vzR, i, j, k)
                FL = apply_diss_flux_z(pr, FL, vxL, vyL, vzL, i, j, k-1)
                FR = apply_diss_flux_z(pr, FR, vxR, vyR, vzR, i, j, k)
                csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
                lamLm, lamLp = eig_speeds(vzL, csL); lamRm, lamRp = eig_speeds(vzR, csR)
                uL = min(lamLm, lamRm); uR = max(lamLp, lamRp)
                FzB = riemann_flux_z(UL, UR, FL, FR, uL, uR, pL, pR, vzL, vzR)
                if N_TRACERS > 0:
                    FzB_tr = np.empty(N_TRACERS)
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        tLq = weno5_left(pr[idx,i,j,k-3], pr[idx,i,j,k-2], pr[idx,i,j,k-1], pr[idx,i,j,k], pr[idx,i,j,k+1])
                        tRq = weno5_right(pr[idx,i,j,k+2], pr[idx,i,j,k+1], pr[idx,i,j,k], pr[idx,i,j,k-1], pr[idx,i,j,k-2])
                        FzB_tr[t] = hlle_scalar(tLq, tRq, tLq*vzL, tRq*vzR, uL, uR)
                if N_TRACERS > 0:
                    FzB_tr = np.empty(N_TRACERS)
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        l_km1, r_km1 = ppm_reconstruct(pr[idx,i,j,k-3], pr[idx,i,j,k-2], pr[idx,i,j,k-1],
                                                       pr[idx,i,j,k-0], pr[idx,i,j,k+1])
                        l_k, r_k = ppm_reconstruct(pr[idx,i,j,k-2], pr[idx,i,j,k-1], pr[idx,i,j,k],
                                                   pr[idx,i,j,k+1], pr[idx,i,j,k+2])
                        tLq = r_km1
                        tRq = l_k
                        FzB_tr[t] = hlle_scalar(tLq, tRq, tLq*vzL, tRq*vzR, uL, uR)

                rL,vxL,vyL,vzL,pL = floor_prim(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4])
                rR,vxR,vyR,vzR,pR = floor_prim(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4])
                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_z(rL,vxL,vyL,vzL,pL); FR = flux_z(rR,vxR,vyR,vzR,pR)
                FL = apply_diss_flux_z(pr, FL, vxL, vyL, vzL, i, j, k)
                FR = apply_diss_flux_z(pr, FR, vxR, vyR, vzR, i, j, k+1)
                FL = apply_diss_flux_z(pr, FL, vxL, vyL, vzL, i, j, k)
                FR = apply_diss_flux_z(pr, FR, vxR, vyR, vzR, i, j, k+1)
                csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
                lamLm, lamLp = eig_speeds(vzL, csL); lamRm, lamRp = eig_speeds(vzR, csR)
                uL2 = min(lamLm, lamRm); uR2 = max(lamLp, lamRp)
                FzF = riemann_flux_z(UL, UR, FL, FR, uL2, uR2, pL, pR, vzL, vzR)
                if N_TRACERS > 0:
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        tLq = weno5_left(pr[idx,i,j,k-2], pr[idx,i,j,k-1], pr[idx,i,j,k], pr[idx,i,j,k+1], pr[idx,i,j,k+2])
                        tRq = weno5_right(pr[idx,i,j,k+3], pr[idx,i,j,k+2], pr[idx,i,j,k+1], pr[idx,i,j,k], pr[idx,i,j,k-1])
                        FzF_tr = hlle_scalar(tLq, tRq, tLq*vzL, tRq*vzR, uL2, uR2)
                        rhs[5 + t, i, j, k] -= (FzF_tr - FzB_tr[t]) / dz
                if N_TRACERS > 0:
                    for t in range(N_TRACERS):
                        idx = TRACER_OFFSET + t
                        l_k, r_k = ppm_reconstruct(pr[idx,i,j,k-2], pr[idx,i,j,k-1], pr[idx,i,j,k],
                                                   pr[idx,i,j,k+1], pr[idx,i,j,k+2])
                        l_kp1, r_kp1 = ppm_reconstruct(pr[idx,i,j,k-1], pr[idx,i,j,k], pr[idx,i,j,k+1],
                                                       pr[idx,i,j,k+2], pr[idx,i,j,k+3])
                        tLq = r_k
                        tRq = l_kp1
                        FzF_tr = hlle_scalar(tLq, tRq, tLq*vzL, tRq*vzR, uL2, uR2)
                        rhs[5 + t, i, j, k] -= (FzF_tr - FzB_tr[t]) / dz

                rhs[0:5,i,j,k] -= (FzF - FzB)/dz

    return rhs
# ------------------------
# WENO5 reconstruction RHS
# ------------------------
@nb.njit(parallel=True, fastmath=True)
def compute_rhs_weno(pr, nx, ny, nz, dx, dy, dz):
    rhs = np.zeros((5 + N_TRACERS, nx, ny, nz))
    i0, i1 = 3, nx-3
    j0, j1 = 3, ny-3
    k0, k1 = 3, nz-3

    for i in range(i0, i1):
        for j in range(j0, j1):
            for k in range(k0, k1):
                qL = np.empty(5)
                qR = np.empty(5)
                qL2 = np.empty(5)
                qR2 = np.empty(5)

                # x faces
                for v in range(5):
                    qL[v] = weno5_left(pr[v,i-3,j,k], pr[v,i-2,j,k], pr[v,i-1,j,k], pr[v,i,j,k], pr[v,i+1,j,k])
                    qR[v] = weno5_right(pr[v,i+2,j,k], pr[v,i+1,j,k], pr[v,i,j,k], pr[v,i-1,j,k], pr[v,i-2,j,k])
                    qL2[v] = weno5_left(pr[v,i-2,j,k], pr[v,i-1,j,k], pr[v,i,j,k], pr[v,i+1,j,k], pr[v,i+2,j,k])
                    qR2[v] = weno5_right(pr[v,i+3,j,k], pr[v,i+2,j,k], pr[v,i+1,j,k], pr[v,i,j,k], pr[v,i-1,j,k])

                rL,vxL,vyL,vzL,pL = floor_prim(qL[0], qL[1], qL[2], qL[3], qL[4])
                rR,vxR,vyR,vzR,pR = floor_prim(qR[0], qR[1], qR[2], qR[3], qR[4])
                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_x(rL,vxL,vyL,vzL,pL); FR = flux_x(rR,vxR,vyR,vzR,pR)
                csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
                lamLm, lamLp = eig_speeds(vxL, csL); lamRm, lamRp = eig_speeds(vxR, csR)
                sL = min(lamLm, lamRm); sR = max(lamLp, lamRp)
                FxL = riemann_flux_x(UL, UR, FL, FR, sL, sR, pL, pR, vxL, vxR)

                rL,vxL,vyL,vzL,pL = floor_prim(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4])
                rR,vxR,vyR,vzR,pR = floor_prim(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4])
                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_x(rL,vxL,vyL,vzL,pL); FR = flux_x(rR,vxR,vyR,vzR,pR)
                csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
                lamLm, lamLp = eig_speeds(vxL, csL); lamRm, lamRp = eig_speeds(vxR, csR)
                sL2 = min(lamLm, lamRm); sR2 = max(lamLp, lamRp)
                FxR = riemann_flux_x(UL, UR, FL, FR, sL2, sR2, pL, pR, vxL, vxR)

                rhs[0:5,i,j,k] -= (FxR - FxL)/dx

                # y faces
                for v in range(5):
                    qL[v] = weno5_left(pr[v,i,j-3,k], pr[v,i,j-2,k], pr[v,i,j-1,k], pr[v,i,j,k], pr[v,i,j+1,k])
                    qR[v] = weno5_right(pr[v,i,j+2,k], pr[v,i,j+1,k], pr[v,i,j,k], pr[v,i,j-1,k], pr[v,i,j-2,k])
                    qL2[v] = weno5_left(pr[v,i,j-2,k], pr[v,i,j-1,k], pr[v,i,j,k], pr[v,i,j+1,k], pr[v,i,j+2,k])
                    qR2[v] = weno5_right(pr[v,i,j+3,k], pr[v,i,j+2,k], pr[v,i,j+1,k], pr[v,i,j,k], pr[v,i,j-1,k])

                rL,vxL,vyL,vzL,pL = floor_prim(qL[0], qL[1], qL[2], qL[3], qL[4])
                rR,vxR,vyR,vzR,pR = floor_prim(qR[0], qR[1], qR[2], qR[3], qR[4])
                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_y(rL,vxL,vyL,vzL,pL); FR = flux_y(rR,vxR,vyR,vzR,pR)
                csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
                lamLm, lamLp = eig_speeds(vyL, csL); lamRm, lamRp = eig_speeds(vyR, csR)
                tL = min(lamLm, lamRm); tR = max(lamLp, lamRp)
                FyD = riemann_flux_y(UL, UR, FL, FR, tL, tR, pL, pR, vyL, vyR)

                rL,vxL,vyL,vzL,pL = floor_prim(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4])
                rR,vxR,vyR,vzR,pR = floor_prim(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4])
                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_y(rL,vxL,vyL,vzL,pL); FR = flux_y(rR,vxR,vyR,vzR,pR)
                csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
                lamLm, lamLp = eig_speeds(vyL, csL); lamRm, lamRp = eig_speeds(vyR, csR)
                tL2 = min(lamLm, lamRm); tR2 = max(lamLp, lamRp)
                FyU = riemann_flux_y(UL, UR, FL, FR, tL2, tR2, pL, pR, vyL, vyR)

                rhs[0:5,i,j,k] -= (FyU - FyD)/dy

                # z faces
                for v in range(5):
                    qL[v] = weno5_left(pr[v,i,j,k-3], pr[v,i,j,k-2], pr[v,i,j,k-1], pr[v,i,j,k], pr[v,i,j,k+1])
                    qR[v] = weno5_right(pr[v,i,j,k+2], pr[v,i,j,k+1], pr[v,i,j,k], pr[v,i,j,k-1], pr[v,i,j,k-2])
                    qL2[v] = weno5_left(pr[v,i,j,k-2], pr[v,i,j,k-1], pr[v,i,j,k], pr[v,i,j,k+1], pr[v,i,j,k+2])
                    qR2[v] = weno5_right(pr[v,i,j,k+3], pr[v,i,j,k+2], pr[v,i,j,k+1], pr[v,i,j,k], pr[v,i,j,k-1])

                rL,vxL,vyL,vzL,pL = floor_prim(qL[0], qL[1], qL[2], qL[3], qL[4])
                rR,vxR,vyR,vzR,pR = floor_prim(qR[0], qR[1], qR[2], qR[3], qR[4])
                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_z(rL,vxL,vyL,vzL,pL); FR = flux_z(rR,vxR,vyR,vzR,pR)
                csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
                lamLm, lamLp = eig_speeds(vzL, csL); lamRm, lamRp = eig_speeds(vzR, csR)
                uL = min(lamLm, lamRm); uR = max(lamLp, lamRp)
                FzB = riemann_flux_z(UL, UR, FL, FR, uL, uR, pL, pR, vzL, vzR)

                rL,vxL,vyL,vzL,pL = floor_prim(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4])
                rR,vxR,vyR,vzR,pR = floor_prim(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4])
                UL = np.array(prim_to_cons(rL,vxL,vyL,vzL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,vzR,pR))
                FL = flux_z(rL,vxL,vyL,vzL,pL); FR = flux_z(rR,vxR,vyR,vzR,pR)
                csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
                lamLm, lamLp = eig_speeds(vzL, csL); lamRm, lamRp = eig_speeds(vzR, csR)
                uL2 = min(lamLm, lamRm); uR2 = max(lamLp, lamRp)
                FzF = riemann_flux_z(UL, UR, FL, FR, uL2, uR2, pL, pR, vzL, vzR)

                rhs[0:5,i,j,k] -= (FzF - FzB)/dz

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
# Time stepping (SSPRK2)
# ------------------------
def step_ssprk2(pr, dx, dy, dz, dt):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    nt = N_TRACERS
    U0 = np.zeros((5, nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                U0[:,i,j,k] = prim_to_cons(pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k], pr[3,i,j,k], pr[4,i,j,k])

    if RECON_ID == 1:
        rhs1 = compute_rhs_ppm(pr, nx, ny, nz, dx, dy, dz)
    elif RECON_ID == 2:
        rhs1 = compute_rhs_weno(pr, nx, ny, nz, dx, dy, dz)
    else:
        rhs1 = compute_rhs_muscl(pr, nx, ny, nz, dx, dy, dz)
    U1   = U0 + dt*rhs1[0:5]

    pr1 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr1[0:5,i,j,k] = cons_to_prim(U1[0,i,j,k], U1[1,i,j,k], U1[2,i,j,k], U1[3,i,j,k], U1[4,i,j,k])
    if pr.shape[0] > 5 and TRACER_OFFSET > 5:
        pr1[5:TRACER_OFFSET] = pr[5:TRACER_OFFSET]
    if nt > 0:
        pr1[TRACER_OFFSET:TRACER_OFFSET+nt] = pr[TRACER_OFFSET:TRACER_OFFSET+nt] + dt*rhs1[5:5+nt]

    if RECON_ID == 1:
        rhs2 = compute_rhs_ppm(pr1, nx, ny, nz, dx, dy, dz)
    elif RECON_ID == 2:
        rhs2 = compute_rhs_weno(pr1, nx, ny, nz, dx, dy, dz)
    else:
        rhs2 = compute_rhs_muscl(pr1, nx, ny, nz, dx, dy, dz)
    U2   = 0.5*(U0 + U1 + dt*rhs2[0:5])

    out  = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[0:5,i,j,k] = cons_to_prim(U2[0,i,j,k], U2[1,i,j,k], U2[2,i,j,k], U2[3,i,j,k], U2[4,i,j,k])
    if pr.shape[0] > 5 and TRACER_OFFSET > 5:
        out[5:TRACER_OFFSET] = pr1[5:TRACER_OFFSET]
    if nt > 0:
        t0 = pr[TRACER_OFFSET:TRACER_OFFSET+nt]
        t1 = pr1[TRACER_OFFSET:TRACER_OFFSET+nt]
        out[TRACER_OFFSET:TRACER_OFFSET+nt] = 0.5*(t0 + t1 + dt*rhs2[5:5+nt])
    return out

def step_ssprk3(pr, dx, dy, dz, dt):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    nt = N_TRACERS
    U0 = np.zeros((5, nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                U0[:,i,j,k] = prim_to_cons(pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k], pr[3,i,j,k], pr[4,i,j,k])

    if RECON_ID == 1:
        rhs1 = compute_rhs_ppm(pr, nx, ny, nz, dx, dy, dz)
    elif RECON_ID == 2:
        rhs1 = compute_rhs_weno(pr, nx, ny, nz, dx, dy, dz)
    else:
        rhs1 = compute_rhs_muscl(pr, nx, ny, nz, dx, dy, dz)
    U1 = U0 + dt*rhs1[0:5]

    pr1 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr1[0:5,i,j,k] = cons_to_prim(U1[0,i,j,k], U1[1,i,j,k], U1[2,i,j,k], U1[3,i,j,k], U1[4,i,j,k])
    if pr.shape[0] > 5 and TRACER_OFFSET > 5:
        pr1[5:TRACER_OFFSET] = pr[5:TRACER_OFFSET]
    if nt > 0:
        pr1[TRACER_OFFSET:TRACER_OFFSET+nt] = pr[TRACER_OFFSET:TRACER_OFFSET+nt] + dt*rhs1[5:5+nt]

    if RECON_ID == 1:
        rhs2 = compute_rhs_ppm(pr1, nx, ny, nz, dx, dy, dz)
    elif RECON_ID == 2:
        rhs2 = compute_rhs_weno(pr1, nx, ny, nz, dx, dy, dz)
    else:
        rhs2 = compute_rhs_muscl(pr1, nx, ny, nz, dx, dy, dz)
    U2 = 0.75*U0 + 0.25*(U1 + dt*rhs2[0:5])

    pr2 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr2[0:5,i,j,k] = cons_to_prim(U2[0,i,j,k], U2[1,i,j,k], U2[2,i,j,k], U2[3,i,j,k], U2[4,i,j,k])
    if pr.shape[0] > 5 and TRACER_OFFSET > 5:
        pr2[5:TRACER_OFFSET] = pr1[5:TRACER_OFFSET]
    if nt > 0:
        t0 = pr[TRACER_OFFSET:TRACER_OFFSET+nt]
        t1 = pr1[TRACER_OFFSET:TRACER_OFFSET+nt]
        pr2[TRACER_OFFSET:TRACER_OFFSET+nt] = 0.75*t0 + 0.25*(t1 + dt*rhs2[5:5+nt])

    if RECON_ID == 1:
        rhs3 = compute_rhs_ppm(pr2, nx, ny, nz, dx, dy, dz)
    elif RECON_ID == 2:
        rhs3 = compute_rhs_weno(pr2, nx, ny, nz, dx, dy, dz)
    else:
        rhs3 = compute_rhs_muscl(pr2, nx, ny, nz, dx, dy, dz)
    U3 = (1.0/3.0)*U0 + (2.0/3.0)*(U2 + dt*rhs3[0:5])

    out = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[0:5,i,j,k] = cons_to_prim(U3[0,i,j,k], U3[1,i,j,k], U3[2,i,j,k], U3[3,i,j,k], U3[4,i,j,k])
    if pr.shape[0] > 5 and TRACER_OFFSET > 5:
        out[5:TRACER_OFFSET] = pr2[5:TRACER_OFFSET]
    if nt > 0:
        t0 = pr[TRACER_OFFSET:TRACER_OFFSET+nt]
        t2 = pr2[TRACER_OFFSET:TRACER_OFFSET+nt]
        out[TRACER_OFFSET:TRACER_OFFSET+nt] = (1.0/3.0)*t0 + (2.0/3.0)*(t2 + dt*rhs3[5:5+nt])
    return out
