#!/usr/bin/env python3
# core/rmhd_core.py
# RMHD core: prim<->cons, HLLE fluxes, GLM psi damping, SSPRK2 (first-order).
import numpy as np
import numba as nb

GAMMA = 5.0 / 3.0
P_MAX = 1.0
V_MAX = 0.999
GLM_CH = 1.0
GLM_CP = 0.1
LIMITER_ID = 0  # 0=mc, 1=minmod, 2=vanleer
RECON_ID = 0    # 0=muscl, 1=ppm, 2=weno
RIEMANN_ID = 0  # 0=hlle, 1=hlld-like
SMALL = 1e-12

def configure(params):
    global GAMMA, P_MAX, V_MAX, GLM_CH, GLM_CP, LIMITER_ID, RECON_ID, RIEMANN_ID
    GAMMA = float(params.get("GAMMA", GAMMA))
    P_MAX = float(params.get("P_MAX", P_MAX))
    V_MAX = float(params.get("V_MAX", V_MAX))
    GLM_CH = float(params.get("GLM_CH", GLM_CH))
    GLM_CP = float(params.get("GLM_CP", GLM_CP))
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
    RIEMANN_ID = 1 if riemann == "hlld" else 0

@nb.njit(fastmath=True)
def minmod(a, b):
    if a*b <= 0.0:
        return 0.0
    else:
        if abs(a) < abs(b): return a
        else:               return b

@nb.njit(fastmath=True)
def limiter(dqL, dqR):
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
def floor_prim_rmhd(rho, vx, vy, vz, p, Bx, By, Bz, psi):
    if rho < SMALL: rho = SMALL
    if p < SMALL: p = SMALL
    if p > P_MAX: p = P_MAX
    v2 = vx*vx + vy*vy + vz*vz
    vmax2 = V_MAX*V_MAX
    if v2 >= vmax2:
        fac = V_MAX / np.sqrt(v2 + 1e-32)
        vx *= fac; vy *= fac; vz *= fac
    return rho, vx, vy, vz, p, Bx, By, Bz, psi

@nb.njit(fastmath=True)
def prim_to_cons_rmhd(rho, vx, vy, vz, p, Bx, By, Bz, psi, gamma=GAMMA):
    # cap inputs
    v2 = vx*vx + vy*vy + vz*vz
    vmax2 = V_MAX*V_MAX
    if v2 >= vmax2:
        fac = V_MAX/np.sqrt(v2 + 1e-32); vx*=fac; vy*=fac; vz*=fac
    if p < SMALL: p = SMALL
    if p > P_MAX: p = P_MAX

    B2 = Bx*Bx + By*By + Bz*Bz
    W  = 1.0/np.sqrt(1.0 - (vx*vx + vy*vy + vz*vz))
    h  = 1.0 + gamma/(gamma-1.0)*p/max(rho, SMALL)
    rhoh = rho*h

    pt = p + 0.5*B2
    vb = vx*Bx + vy*By + vz*Bz
    b0 = W*vb
    bx = (Bx + b0*vx)/W
    by = (By + b0*vy)/W
    bz = (Bz + b0*vz)/W
    b2 = (bx*bx + by*by + bz*bz) - b0*b0

    wtot = rhoh*W*W + B2

    D   = rho*W
    Sx  = wtot*vx - (Bx*vb)
    Sy  = wtot*vy - (By*vb)
    Sz  = wtot*vz - (Bz*vb)
    tau = rhoh*W*W - p + 0.5*B2 + 0.5*b2 - D

    return D, Sx, Sy, Sz, tau, Bx, By, Bz, psi

@nb.njit(fastmath=True)
def _cons_to_prim_hydro(D, Sx, Sy, Sz, tau, gamma=GAMMA):
    E  = tau + D
    S2 = Sx*Sx + Sy*Sy + Sz*Sz

    p = (gamma - 1.0) * (E - D)
    if p < SMALL: p = SMALL
    if p > P_MAX: p = P_MAX

    ok = False
    for _ in range(60):
        Wm = E + p
        v2 = S2 / (Wm*Wm + SMALL)
        if v2 >= V_MAX*V_MAX:
            v2 = V_MAX*V_MAX - 1e-14
        W  = 1.0 / np.sqrt(1.0 - v2)
        rho= D / max(W, SMALL)
        h  = 1.0 + gamma/(gamma-1.0) * p / max(rho, SMALL)
        w  = rho * h
        f  = Wm - w * W * W
        dfdp = 1.0
        dp = -f / dfdp
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
        p = (gamma - 1.0) * (E - D)
        if p < SMALL: p = SMALL
        if p > P_MAX: p = P_MAX

    Wm = E + p
    vx = Sx / max(Wm, SMALL)
    vy = Sy / max(Wm, SMALL)
    vz = Sz / max(Wm, SMALL)
    v2 = vx*vx + vy*vy + vz*vz
    vmax2 = V_MAX*V_MAX
    if v2 >= vmax2:
        fac = V_MAX / np.sqrt(v2 + 1e-32)
        vx *= fac; vy *= fac; vz *= fac

    v2 = vx*vx + vy*vy + vz*vz
    if v2 >= 1.0: v2 = 1.0 - 1e-14
    W  = 1.0 / np.sqrt(1.0 - v2)
    rho= D / max(W, SMALL)
    if rho < SMALL: rho = SMALL
    if p < SMALL: p = SMALL
    if p > P_MAX: p = P_MAX

    return rho, vx, vy, vz, p

@nb.njit(fastmath=True)
def cons_to_prim_rmhd(D, Sx, Sy, Sz, tau, Bx, By, Bz, psi):
    # Improved recovery: solve for Z = rho h W^2 using a damped Newton iteration.
    B2 = Bx*Bx + By*By + Bz*Bz
    SB = Sx*Bx + Sy*By + Sz*Bz
    S2 = Sx*Sx + Sy*Sy + Sz*Sz

    # initial guess from hydro recovery
    tau_h = tau - 0.5*B2
    if tau_h < SMALL:
        tau_h = SMALL
    rho0, vx0, vy0, vz0, p0 = _cons_to_prim_hydro(D, Sx, Sy, Sz, tau_h, GAMMA)
    v20 = vx0*vx0 + vy0*vy0 + vz0*vz0
    if v20 >= V_MAX*V_MAX:
        v20 = V_MAX*V_MAX - 1e-14
    W0 = 1.0 / np.sqrt(1.0 - v20)
    h0 = 1.0 + GAMMA/(GAMMA-1.0)*p0/max(rho0, SMALL)
    Z = max(rho0*h0*W0*W0, SMALL)

    for _ in range(50):
        denomZ = max(Z, SMALL)
        vb = SB / denomZ
        ZpB = Z + B2
        v2 = (S2 + (SB*SB) * (2.0*Z + B2) / (denomZ*denomZ)) / (ZpB*ZpB + SMALL)
        vmax2 = V_MAX*V_MAX
        if v2 >= vmax2:
            v2 = vmax2 - 1e-14
        W2 = 1.0 / (1.0 - v2)
        W = np.sqrt(W2)
        rho = D / max(W, SMALL)
        h = Z / max(rho*W2, SMALL)
        p = (h - 1.0) * rho * (GAMMA - 1.0) / GAMMA
        if p < SMALL: p = SMALL
        if p > P_MAX: p = P_MAX
        b2 = B2 / W2 + vb*vb
        tau_calc = Z - p + 0.5*B2 + 0.5*b2 - D
        f = tau_calc - tau
        if np.abs(f) < 1e-10 * max(1.0, tau):
            break

        eps = 1e-6 * max(1.0, Z)
        Zp = Z + eps
        denomZp = max(Zp, SMALL)
        vb_p = SB / denomZp
        ZpB_p = Zp + B2
        v2p = (S2 + (SB*SB) * (2.0*Zp + B2) / (denomZp*denomZp)) / (ZpB_p*ZpB_p + SMALL)
        if v2p >= vmax2:
            v2p = vmax2 - 1e-14
        W2p = 1.0 / (1.0 - v2p)
        Wp = np.sqrt(W2p)
        rhop = D / max(Wp, SMALL)
        hp = Zp / max(rhop*W2p, SMALL)
        pp = (hp - 1.0) * rhop * (GAMMA - 1.0) / GAMMA
        if pp < SMALL: pp = SMALL
        if pp > P_MAX: pp = P_MAX
        b2p = B2 / W2p + vb_p*vb_p
        tau_calc_p = Zp - pp + 0.5*B2 + 0.5*b2p - D
        fp = tau_calc_p - tau

        dfdZ = (fp - f) / eps
        if dfdZ == 0.0:
            break
        dZ = -f / dfdZ
        if dZ >  0.5*Z: dZ =  0.5*Z
        if dZ < -0.5*Z: dZ = -0.5*Z
        Z += dZ
        if Z < SMALL:
            Z = SMALL

    # recover velocities
    ZpB = Z + B2
    vb = SB / max(Z, SMALL)
    vx = (Sx + vb*Bx) / max(ZpB, SMALL)
    vy = (Sy + vb*By) / max(ZpB, SMALL)
    vz = (Sz + vb*Bz) / max(ZpB, SMALL)

    v2 = vx*vx + vy*vy + vz*vz
    vmax2 = V_MAX*V_MAX
    if v2 >= vmax2:
        fac = V_MAX / np.sqrt(v2 + 1e-32)
        vx *= fac; vy *= fac; vz *= fac
    W = 1.0 / np.sqrt(1.0 - (vx*vx + vy*vy + vz*vz))
    rho = D / max(W, SMALL)
    h = Z / max(rho*W*W, SMALL)
    p = (h - 1.0) * rho * (GAMMA - 1.0) / GAMMA

    return floor_prim_rmhd(rho, vx, vy, vz, p, Bx, By, Bz, psi)

@nb.njit(fastmath=True)
def flux_rmhd_x(prim):
    rho,vx,vy,vz,p,Bx,By,Bz,psi = prim
    D,Sx,Sy,Sz,tau,_,_,_,_ = prim_to_cons_rmhd(rho,vx,vy,vz,p,Bx,By,Bz,psi,GAMMA)
    B2 = Bx*Bx + By*By + Bz*Bz
    pt = p + 0.5*B2
    F = np.zeros(9)
    F[0] = D*vx
    F[1] = Sx*vx + pt - Bx*Bx
    F[2] = Sy*vx - Bx*By
    F[3] = Sz*vx - Bx*Bz
    F[4] = (tau + pt)*vx - (Bx*(Bx*vx + By*vy + Bz*vz))
    F[5] = psi
    F[6] = vy*Bx - vx*By
    F[7] = vz*Bx - vx*Bz
    F[8] = GLM_CH*GLM_CH * Bx
    return F

@nb.njit(fastmath=True)
def flux_rmhd_y(prim):
    rho,vx,vy,vz,p,Bx,By,Bz,psi = prim
    D,Sx,Sy,Sz,tau,_,_,_,_ = prim_to_cons_rmhd(rho,vx,vy,vz,p,Bx,By,Bz,psi,GAMMA)
    B2 = Bx*Bx + By*By + Bz*Bz
    pt = p + 0.5*B2
    F = np.zeros(9)
    F[0] = D*vy
    F[1] = Sx*vy - By*Bx
    F[2] = Sy*vy + pt - By*By
    F[3] = Sz*vy - By*Bz
    F[4] = (tau + pt)*vy - (By*(Bx*vx + By*vy + Bz*vz))
    F[5] = vx*By - vy*Bx
    F[6] = psi
    F[7] = vz*By - vy*Bz
    F[8] = GLM_CH*GLM_CH * By
    return F

@nb.njit(fastmath=True)
def flux_rmhd_z(prim):
    rho,vx,vy,vz,p,Bx,By,Bz,psi = prim
    D,Sx,Sy,Sz,tau,_,_,_,_ = prim_to_cons_rmhd(rho,vx,vy,vz,p,Bx,By,Bz,psi,GAMMA)
    B2 = Bx*Bx + By*By + Bz*Bz
    pt = p + 0.5*B2
    F = np.zeros(9)
    F[0] = D*vz
    F[1] = Sx*vz - Bz*Bx
    F[2] = Sy*vz - Bz*By
    F[3] = Sz*vz + pt - Bz*Bz
    F[4] = (tau + pt)*vz - (Bz*(Bx*vx + By*vy + Bz*vz))
    F[5] = vx*Bz - vz*Bx
    F[6] = vy*Bz - vz*By
    F[7] = psi
    F[8] = GLM_CH*GLM_CH * Bz
    return F

@nb.njit(fastmath=True)
def hlle_bounds_fast():
    # robust bounds for first pass: use ±c
    return -1.0, 1.0

@nb.njit(fastmath=True)
def hlle(UL, UR, FL, FR, sL, sR):
    if sL >= 0.0:
        return FL
    elif sR <= 0.0:
        return FR
    else:
        return (sR*FL - sL*FR + sL*sR*(UR-UL)) / (sR - sL + SMALL)

@nb.njit(fastmath=True)
def hllc_hydro_flux_x(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    DL, SxL, SyL, SzL, tauL = UL[0], UL[1], UL[2], UL[3], UL[4]
    DR, SxR, SyR, SzR, tauR = UR[0], UR[1], UR[2], UR[3], UR[4]
    denom = (DL*(sL - vL) - DR*(sR - vR)) + SMALL
    sM = (pR - pL + SxL*(sL - vL) - SxR*(sR - vR)) / denom
    pStar = 0.5*(pL + pR + DL*(sL - vL)*(sM - vL) + DR*(sR - vR)*(sM - vR))

    if sL >= 0.0:
        return FL[:5]
    if sR <= 0.0:
        return FR[:5]

    if sM >= 0.0:
        fac = (sL - vL) / (sL - sM + SMALL)
        Dst = DL * fac
        Sxst = SxL * fac + (pStar - pL) / (sL - sM + SMALL)
        Syst = SyL * fac
        Szst = SzL * fac
        taust = tauL * fac + (pStar*sM - pL*vL) / (sL - sM + SMALL)
        Ust = np.array([Dst, Sxst, Syst, Szst, taust])
        return FL[:5] + sL*(Ust - UL[:5])
    else:
        fac = (sR - vR) / (sR - sM + SMALL)
        Dst = DR * fac
        Sxst = SxR * fac + (pStar - pR) / (sR - sM + SMALL)
        Syst = SyR * fac
        Szst = SzR * fac
        taust = tauR * fac + (pStar*sM - pR*vR) / (sR - sM + SMALL)
        Ust = np.array([Dst, Sxst, Syst, Szst, taust])
        return FR[:5] + sR*(Ust - UR[:5])

@nb.njit(fastmath=True)
def hllc_hydro_flux_y(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    DL, SxL, SyL, SzL, tauL = UL[0], UL[1], UL[2], UL[3], UL[4]
    DR, SxR, SyR, SzR, tauR = UR[0], UR[1], UR[2], UR[3], UR[4]
    denom = (DL*(sL - vL) - DR*(sR - vR)) + SMALL
    sM = (pR - pL + SyL*(sL - vL) - SyR*(sR - vR)) / denom
    pStar = 0.5*(pL + pR + DL*(sL - vL)*(sM - vL) + DR*(sR - vR)*(sM - vR))

    if sL >= 0.0:
        return FL[:5]
    if sR <= 0.0:
        return FR[:5]

    if sM >= 0.0:
        fac = (sL - vL) / (sL - sM + SMALL)
        Dst = DL * fac
        Syst = SyL * fac + (pStar - pL) / (sL - sM + SMALL)
        Sxst = SxL * fac
        Szst = SzL * fac
        taust = tauL * fac + (pStar*sM - pL*vL) / (sL - sM + SMALL)
        Ust = np.array([Dst, Sxst, Syst, Szst, taust])
        return FL[:5] + sL*(Ust - UL[:5])
    else:
        fac = (sR - vR) / (sR - sM + SMALL)
        Dst = DR * fac
        Syst = SyR * fac + (pStar - pR) / (sR - sM + SMALL)
        Sxst = SxR * fac
        Szst = SzR * fac
        taust = tauR * fac + (pStar*sM - pR*vR) / (sR - sM + SMALL)
        Ust = np.array([Dst, Sxst, Syst, Szst, taust])
        return FR[:5] + sR*(Ust - UR[:5])

@nb.njit(fastmath=True)
def hllc_hydro_flux_z(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    DL, SxL, SyL, SzL, tauL = UL[0], UL[1], UL[2], UL[3], UL[4]
    DR, SxR, SyR, SzR, tauR = UR[0], UR[1], UR[2], UR[3], UR[4]
    denom = (DL*(sL - vL) - DR*(sR - vR)) + SMALL
    sM = (pR - pL + SzL*(sL - vL) - SzR*(sR - vR)) / denom
    pStar = 0.5*(pL + pR + DL*(sL - vL)*(sM - vL) + DR*(sR - vR)*(sM - vR))

    if sL >= 0.0:
        return FL[:5]
    if sR <= 0.0:
        return FR[:5]

    if sM >= 0.0:
        fac = (sL - vL) / (sL - sM + SMALL)
        Dst = DL * fac
        Szst = SzL * fac + (pStar - pL) / (sL - sM + SMALL)
        Sxst = SxL * fac
        Syst = SyL * fac
        taust = tauL * fac + (pStar*sM - pL*vL) / (sL - sM + SMALL)
        Ust = np.array([Dst, Sxst, Syst, Szst, taust])
        return FL[:5] + sL*(Ust - UL[:5])
    else:
        fac = (sR - vR) / (sR - sM + SMALL)
        Dst = DR * fac
        Szst = SzR * fac + (pStar - pR) / (sR - sM + SMALL)
        Sxst = SxR * fac
        Syst = SyR * fac
        taust = tauR * fac + (pStar*sM - pR*vR) / (sR - sM + SMALL)
        Ust = np.array([Dst, Sxst, Syst, Szst, taust])
        return FR[:5] + sR*(Ust - UR[:5])

@nb.njit(fastmath=True)
def riemann_flux_rmhd_x(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    if RIEMANN_ID == 1:
        hllc = hllc_hydro_flux_x(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR)
        out = hlle(UL, UR, FL, FR, sL, sR)
        out[0:5] = hllc
        return out
    return hlle(UL, UR, FL, FR, sL, sR)

@nb.njit(fastmath=True)
def riemann_flux_rmhd_y(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    if RIEMANN_ID == 1:
        hllc = hllc_hydro_flux_y(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR)
        out = hlle(UL, UR, FL, FR, sL, sR)
        out[0:5] = hllc
        return out
    return hlle(UL, UR, FL, FR, sL, sR)

@nb.njit(fastmath=True)
def riemann_flux_rmhd_z(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    if RIEMANN_ID == 1:
        hllc = hllc_hydro_flux_z(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR)
        out = hlle(UL, UR, FL, FR, sL, sR)
        out[0:5] = hllc
        return out
    return hlle(UL, UR, FL, FR, sL, sR)

@nb.njit(parallel=True, fastmath=True)
def compute_rhs_rmhd(pr, nx, ny, nz, dx, dy, dz):
    rhs = np.zeros((9, nx, ny, nz))
    i0, i1 = 2, nx-2
    j0, j1 = 2, ny-2
    k0, k1 = 2, nz-2
    sL, sR = hlle_bounds_fast()

    for i in range(i0, i1):
        for j in range(j0, j1):
            for k in range(k0, k1):
                slR = np.empty(9)
                slRp1 = np.empty(9)

                qL = np.empty(9)
                qR = np.empty(9)

                # x left face (i-1/2)
                for v in range(9):
                    dqL = pr[v, i-1, j, k] - pr[v, i-2, j, k]
                    dqR = pr[v, i,   j, k] - pr[v, i-1, j, k]
                    slL = limiter(dqL, dqR)
                    dqL2 = pr[v, i,   j, k] - pr[v, i-1, j, k]
                    dqR2 = pr[v, i+1, j, k] - pr[v, i,   j, k]
                    slR[v] = limiter(dqL2, dqR2)
                    qL[v] = pr[v, i-1, j, k] + 0.5*slL
                    qR[v] = pr[v, i,   j, k] - 0.5*slR[v]

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(
                    qL[0], qL[1], qL[2], qL[3], qL[4], qL[5], qL[6], qL[7], qL[8]
                )
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(
                    qR[0], qR[1], qR[2], qR[3], qR[4], qR[5], qR[6], qR[7], qR[8]
                )

                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_x(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_x(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FxL = riemann_flux_rmhd_x(UL, UR, FL, FR, sL, sR, pL, pR, vxL, vxR)

                # x right face (i+1/2)
                for v in range(9):
                    dqL = pr[v, i+1, j, k] - pr[v, i,   j, k]
                    dqR = pr[v, i+2, j, k] - pr[v, i+1, j, k]
                    slRp1[v] = limiter(dqL, dqR)
                    qL[v] = pr[v, i,   j, k] + 0.5*slR[v]
                    qR[v] = pr[v, i+1, j, k] - 0.5*slRp1[v]

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(
                    qL[0], qL[1], qL[2], qL[3], qL[4], qL[5], qL[6], qL[7], qL[8]
                )
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(
                    qR[0], qR[1], qR[2], qR[3], qR[4], qR[5], qR[6], qR[7], qR[8]
                )

                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_x(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_x(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FxR = riemann_flux_rmhd_x(UL, UR, FL, FR, sL, sR, pL, pR, vxL, vxR)

                rhs[:, i, j, k] -= (FxR - FxL) / dx

                # y faces
                for v in range(9):
                    dqL = pr[v, i, j-1, k] - pr[v, i, j-2, k]
                    dqR = pr[v, i, j,   k] - pr[v, i, j-1, k]
                    slL = limiter(dqL, dqR)
                    dqL2 = pr[v, i, j,   k] - pr[v, i, j-1, k]
                    dqR2 = pr[v, i, j+1, k] - pr[v, i, j,   k]
                    slR[v] = limiter(dqL2, dqR2)
                    qL[v] = pr[v, i, j-1, k] + 0.5*slL
                    qR[v] = pr[v, i, j,   k] - 0.5*slR[v]

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(
                    qL[0], qL[1], qL[2], qL[3], qL[4], qL[5], qL[6], qL[7], qL[8]
                )
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(
                    qR[0], qR[1], qR[2], qR[3], qR[4], qR[5], qR[6], qR[7], qR[8]
                )

                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FyD = riemann_flux_rmhd_y(UL, UR, FL, FR, sL, sR, pL, pR, vyL, vyR)

                for v in range(9):
                    dqL = pr[v, i, j+1, k] - pr[v, i, j,   k]
                    dqR = pr[v, i, j+2, k] - pr[v, i, j+1, k]
                    slRp1[v] = limiter(dqL, dqR)
                    qL[v] = pr[v, i, j,   k] + 0.5*slR[v]
                    qR[v] = pr[v, i, j+1, k] - 0.5*slRp1[v]

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(
                    qL[0], qL[1], qL[2], qL[3], qL[4], qL[5], qL[6], qL[7], qL[8]
                )
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(
                    qR[0], qR[1], qR[2], qR[3], qR[4], qR[5], qR[6], qR[7], qR[8]
                )

                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FyU = riemann_flux_rmhd_y(UL, UR, FL, FR, sL, sR, pL, pR, vyL, vyR)

                rhs[:, i, j, k] -= (FyU - FyD) / dy

                # z faces
                for v in range(9):
                    dqL = pr[v, i, j, k-1] - pr[v, i, j, k-2]
                    dqR = pr[v, i, j, k  ] - pr[v, i, j, k-1]
                    slL = limiter(dqL, dqR)
                    dqL2 = pr[v, i, j, k  ] - pr[v, i, j, k-1]
                    dqR2 = pr[v, i, j, k+1] - pr[v, i, j, k  ]
                    slR[v] = limiter(dqL2, dqR2)
                    qL[v] = pr[v, i, j, k-1] + 0.5*slL
                    qR[v] = pr[v, i, j, k  ] - 0.5*slR[v]

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(
                    qL[0], qL[1], qL[2], qL[3], qL[4], qL[5], qL[6], qL[7], qL[8]
                )
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(
                    qR[0], qR[1], qR[2], qR[3], qR[4], qR[5], qR[6], qR[7], qR[8]
                )

                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FzB = riemann_flux_rmhd_z(UL, UR, FL, FR, sL, sR, pL, pR, vzL, vzR)

                for v in range(9):
                    dqL = pr[v, i, j, k+1] - pr[v, i, j, k  ]
                    dqR = pr[v, i, j, k+2] - pr[v, i, j, k+1]
                    slRp1[v] = limiter(dqL, dqR)
                    qL[v] = pr[v, i, j, k  ] + 0.5*slR[v]
                    qR[v] = pr[v, i, j, k+1] - 0.5*slRp1[v]

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(
                    qL[0], qL[1], qL[2], qL[3], qL[4], qL[5], qL[6], qL[7], qL[8]
                )
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(
                    qR[0], qR[1], qR[2], qR[3], qR[4], qR[5], qR[6], qR[7], qR[8]
                )

                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FzF = riemann_flux_rmhd_z(UL, UR, FL, FR, sL, sR, pL, pR, vzL, vzR)

                rhs[:, i, j, k] -= (FzF - FzB) / dz

                # GLM damping source term
                rhs[8, i, j, k] -= GLM_CP*GLM_CP * pr[8, i, j, k]

    return rhs

@nb.njit(parallel=True, fastmath=True)
def compute_rhs_ppm(pr, nx, ny, nz, dx, dy, dz):
    rhs = np.zeros((9, nx, ny, nz))
    i0, i1 = 3, nx-3
    j0, j1 = 3, ny-3
    k0, k1 = 3, nz-3
    sL, sR = hlle_bounds_fast()

    for i in range(i0, i1):
        for j in range(j0, j1):
            for k in range(k0, k1):
                qL = np.empty(9)
                qR = np.empty(9)
                qL2 = np.empty(9)
                qR2 = np.empty(9)

                # x faces
                for v in range(9):
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

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL[0], qL[1], qL[2], qL[3], qL[4], qL[5], qL[6], qL[7], qL[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR[0], qR[1], qR[2], qR[3], qR[4], qR[5], qR[6], qR[7], qR[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_x(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_x(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FxL = riemann_flux_rmhd_x(UL, UR, FL, FR, sL, sR, pL, pR, vxL, vxR)

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4], qL2[5], qL2[6], qL2[7], qL2[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4], qR2[5], qR2[6], qR2[7], qR2[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_x(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_x(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FxR = riemann_flux_rmhd_x(UL, UR, FL, FR, sL, sR, pL, pR, vxL, vxR)

                rhs[:, i, j, k] -= (FxR - FxL) / dx

                # y faces
                for v in range(9):
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

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL[0], qL[1], qL[2], qL[3], qL[4], qL[5], qL[6], qL[7], qL[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR[0], qR[1], qR[2], qR[3], qR[4], qR[5], qR[6], qR[7], qR[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FyD = riemann_flux_rmhd_y(UL, UR, FL, FR, sL, sR, pL, pR, vyL, vyR)

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4], qL2[5], qL2[6], qL2[7], qL2[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4], qR2[5], qR2[6], qR2[7], qR2[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FyU = riemann_flux_rmhd_y(UL, UR, FL, FR, sL, sR, pL, pR, vyL, vyR)

                rhs[:, i, j, k] -= (FyU - FyD) / dy

                # z faces
                for v in range(9):
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

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL[0], qL[1], qL[2], qL[3], qL[4], qL[5], qL[6], qL[7], qL[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR[0], qR[1], qR[2], qR[3], qR[4], qR[5], qR[6], qR[7], qR[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FzB = riemann_flux_rmhd_z(UL, UR, FL, FR, sL, sR, pL, pR, vzL, vzR)

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4], qL2[5], qL2[6], qL2[7], qL2[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4], qR2[5], qR2[6], qR2[7], qR2[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FzF = riemann_flux_rmhd_z(UL, UR, FL, FR, sL, sR, pL, pR, vzL, vzR)

                rhs[:, i, j, k] -= (FzF - FzB) / dz

                rhs[8, i, j, k] -= GLM_CP*GLM_CP * pr[8, i, j, k]

    return rhs

@nb.njit(parallel=True, fastmath=True)
def compute_rhs_weno(pr, nx, ny, nz, dx, dy, dz):
    rhs = np.zeros((9, nx, ny, nz))
    i0, i1 = 3, nx-3
    j0, j1 = 3, ny-3
    k0, k1 = 3, nz-3
    sL, sR = hlle_bounds_fast()

    for i in range(i0, i1):
        for j in range(j0, j1):
            for k in range(k0, k1):
                qL = np.empty(9)
                qR = np.empty(9)
                qL2 = np.empty(9)
                qR2 = np.empty(9)

                # x faces
                for v in range(9):
                    qL[v] = weno5_left(pr[v,i-3,j,k], pr[v,i-2,j,k], pr[v,i-1,j,k], pr[v,i,j,k], pr[v,i+1,j,k])
                    qR[v] = weno5_right(pr[v,i+2,j,k], pr[v,i+1,j,k], pr[v,i,j,k], pr[v,i-1,j,k], pr[v,i-2,j,k])
                    qL2[v] = weno5_left(pr[v,i-2,j,k], pr[v,i-1,j,k], pr[v,i,j,k], pr[v,i+1,j,k], pr[v,i+2,j,k])
                    qR2[v] = weno5_right(pr[v,i+3,j,k], pr[v,i+2,j,k], pr[v,i+1,j,k], pr[v,i,j,k], pr[v,i-1,j,k])

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL[0], qL[1], qL[2], qL[3], qL[4], qL[5], qL[6], qL[7], qL[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR[0], qR[1], qR[2], qR[3], qR[4], qR[5], qR[6], qR[7], qR[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_x(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_x(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FxL = riemann_flux_rmhd_x(UL, UR, FL, FR, sL, sR, pL, pR, vxL, vxR)

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4], qL2[5], qL2[6], qL2[7], qL2[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4], qR2[5], qR2[6], qR2[7], qR2[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_x(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_x(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FxR = riemann_flux_rmhd_x(UL, UR, FL, FR, sL, sR, pL, pR, vxL, vxR)

                rhs[:, i, j, k] -= (FxR - FxL) / dx

                # y faces
                for v in range(9):
                    qL[v] = weno5_left(pr[v,i,j-3,k], pr[v,i,j-2,k], pr[v,i,j-1,k], pr[v,i,j,k], pr[v,i,j+1,k])
                    qR[v] = weno5_right(pr[v,i,j+2,k], pr[v,i,j+1,k], pr[v,i,j,k], pr[v,i,j-1,k], pr[v,i,j-2,k])
                    qL2[v] = weno5_left(pr[v,i,j-2,k], pr[v,i,j-1,k], pr[v,i,j,k], pr[v,i,j+1,k], pr[v,i,j+2,k])
                    qR2[v] = weno5_right(pr[v,i,j+3,k], pr[v,i,j+2,k], pr[v,i,j+1,k], pr[v,i,j,k], pr[v,i,j-1,k])

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL[0], qL[1], qL[2], qL[3], qL[4], qL[5], qL[6], qL[7], qL[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR[0], qR[1], qR[2], qR[3], qR[4], qR[5], qR[6], qR[7], qR[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FyD = riemann_flux_rmhd_y(UL, UR, FL, FR, sL, sR, pL, pR, vyL, vyR)

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4], qL2[5], qL2[6], qL2[7], qL2[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4], qR2[5], qR2[6], qR2[7], qR2[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FyU = riemann_flux_rmhd_y(UL, UR, FL, FR, sL, sR, pL, pR, vyL, vyR)

                rhs[:, i, j, k] -= (FyU - FyD) / dy

                # z faces
                for v in range(9):
                    qL[v] = weno5_left(pr[v,i,j,k-3], pr[v,i,j,k-2], pr[v,i,j,k-1], pr[v,i,j,k], pr[v,i,j,k+1])
                    qR[v] = weno5_right(pr[v,i,j,k+2], pr[v,i,j,k+1], pr[v,i,j,k], pr[v,i,j,k-1], pr[v,i,j,k-2])
                    qL2[v] = weno5_left(pr[v,i,j,k-2], pr[v,i,j,k-1], pr[v,i,j,k], pr[v,i,j,k+1], pr[v,i,j,k+2])
                    qR2[v] = weno5_right(pr[v,i,j,k+3], pr[v,i,j,k+2], pr[v,i,j,k+1], pr[v,i,j,k], pr[v,i,j,k-1])

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL[0], qL[1], qL[2], qL[3], qL[4], qL[5], qL[6], qL[7], qL[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR[0], qR[1], qR[2], qR[3], qR[4], qR[5], qR[6], qR[7], qR[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FzB = riemann_flux_rmhd_z(UL, UR, FL, FR, sL, sR, pL, pR, vzL, vzR)

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4], qL2[5], qL2[6], qL2[7], qL2[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4], qR2[5], qR2[6], qR2[7], qR2[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                FzF = riemann_flux_rmhd_z(UL, UR, FL, FR, sL, sR, pL, pR, vzL, vzR)

                rhs[:, i, j, k] -= (FzF - FzB) / dz

                rhs[8, i, j, k] -= GLM_CP*GLM_CP * pr[8, i, j, k]

    return rhs

def step_ssprk2(pr, dx, dy, dz, dt):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    U0 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                U0[:,i,j,k] = prim_to_cons_rmhd(pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k],
                                                pr[3,i,j,k], pr[4,i,j,k], pr[5,i,j,k],
                                                pr[6,i,j,k], pr[7,i,j,k], pr[8,i,j,k], GAMMA)

    if RECON_ID == 1:
        rhs1 = compute_rhs_ppm(pr, nx, ny, nz, dx, dy, dz)
    elif RECON_ID == 2:
        rhs1 = compute_rhs_weno(pr, nx, ny, nz, dx, dy, dz)
    else:
        rhs1 = compute_rhs_rmhd(pr, nx, ny, nz, dx, dy, dz)
    U1   = U0 + dt*rhs1

    pr1 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr1[:,i,j,k] = cons_to_prim_rmhd(U1[0,i,j,k], U1[1,i,j,k], U1[2,i,j,k],
                                                U1[3,i,j,k], U1[4,i,j,k], U1[5,i,j,k],
                                                U1[6,i,j,k], U1[7,i,j,k], U1[8,i,j,k])

    if RECON_ID == 1:
        rhs2 = compute_rhs_ppm(pr1, nx, ny, nz, dx, dy, dz)
    elif RECON_ID == 2:
        rhs2 = compute_rhs_weno(pr1, nx, ny, nz, dx, dy, dz)
    else:
        rhs2 = compute_rhs_rmhd(pr1, nx, ny, nz, dx, dy, dz)
    U2   = 0.5*(U0 + U1 + dt*rhs2)

    out  = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[:,i,j,k] = cons_to_prim_rmhd(U2[0,i,j,k], U2[1,i,j,k], U2[2,i,j,k],
                                                 U2[3,i,j,k], U2[4,i,j,k], U2[5,i,j,k],
                                                 U2[6,i,j,k], U2[7,i,j,k], U2[8,i,j,k])
    return out

def step_ssprk3(pr, dx, dy, dz, dt):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    U0 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                U0[:,i,j,k] = prim_to_cons_rmhd(pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k],
                                                pr[3,i,j,k], pr[4,i,j,k], pr[5,i,j,k],
                                                pr[6,i,j,k], pr[7,i,j,k], pr[8,i,j,k], GAMMA)

    if RECON_ID == 1:
        rhs1 = compute_rhs_ppm(pr, nx, ny, nz, dx, dy, dz)
    elif RECON_ID == 2:
        rhs1 = compute_rhs_weno(pr, nx, ny, nz, dx, dy, dz)
    else:
        rhs1 = compute_rhs_rmhd(pr, nx, ny, nz, dx, dy, dz)
    U1 = U0 + dt*rhs1

    pr1 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr1[:,i,j,k] = cons_to_prim_rmhd(U1[0,i,j,k], U1[1,i,j,k], U1[2,i,j,k],
                                                U1[3,i,j,k], U1[4,i,j,k], U1[5,i,j,k],
                                                U1[6,i,j,k], U1[7,i,j,k], U1[8,i,j,k])

    if RECON_ID == 1:
        rhs2 = compute_rhs_ppm(pr1, nx, ny, nz, dx, dy, dz)
    elif RECON_ID == 2:
        rhs2 = compute_rhs_weno(pr1, nx, ny, nz, dx, dy, dz)
    else:
        rhs2 = compute_rhs_rmhd(pr1, nx, ny, nz, dx, dy, dz)
    U2 = 0.75*U0 + 0.25*(U1 + dt*rhs2)

    pr2 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr2[:,i,j,k] = cons_to_prim_rmhd(U2[0,i,j,k], U2[1,i,j,k], U2[2,i,j,k],
                                                 U2[3,i,j,k], U2[4,i,j,k], U2[5,i,j,k],
                                                 U2[6,i,j,k], U2[7,i,j,k], U2[8,i,j,k])

    if RECON_ID == 1:
        rhs3 = compute_rhs_ppm(pr2, nx, ny, nz, dx, dy, dz)
    elif RECON_ID == 2:
        rhs3 = compute_rhs_weno(pr2, nx, ny, nz, dx, dy, dz)
    else:
        rhs3 = compute_rhs_rmhd(pr2, nx, ny, nz, dx, dy, dz)
    U3 = (1.0/3.0)*U0 + (2.0/3.0)*(U2 + dt*rhs3)

    out = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[:,i,j,k] = cons_to_prim_rmhd(U3[0,i,j,k], U3[1,i,j,k], U3[2,i,j,k],
                                                 U3[3,i,j,k], U3[4,i,j,k], U3[5,i,j,k],
                                                 U3[6,i,j,k], U3[7,i,j,k], U3[8,i,j,k])
    return out
