#!/usr/bin/env python3
# core/rmhd_core.py
# RMHD core: prim<->cons, HLLE fluxes, GLM psi damping, SSPRK2 (first-order).
import numpy as np
import numba as nb
from core import eos

GAMMA = 5.0 / 3.0
P_MAX = 1.0
V_MAX = 0.999
GLM_CH = 1.0
GLM_CP = 0.1
LIMITER_ID = 0  # 0=mc, 1=minmod, 2=vanleer
RECON_ID = 0    # 0=muscl, 1=ppm, 2=weno
RIEMANN_ID = 0  # 0=hlle, 1=hlld-like
SMALL = 1e-12
RESISTIVE_ENABLED = False
RESISTIVITY = 0.0
N_PASSIVE = 0
PASSIVE_OFFSET = 9

def configure(params):
    global GAMMA, P_MAX, V_MAX, GLM_CH, GLM_CP, LIMITER_ID, RECON_ID, RIEMANN_ID, N_PASSIVE, PASSIVE_OFFSET, RESISTIVE_ENABLED, RESISTIVITY
    GAMMA = float(params.get("GAMMA", GAMMA))
    P_MAX = float(params.get("P_MAX", P_MAX))
    V_MAX = float(params.get("V_MAX", V_MAX))
    eos.configure(params)
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
    if riemann == "hlld_full":
        RIEMANN_ID = 2
    else:
        RIEMANN_ID = 1 if riemann == "hlld" else 0
    RESISTIVE_ENABLED = bool(params.get("RESISTIVE_ENABLED", RESISTIVE_ENABLED))
    RESISTIVITY = float(params.get("RESISTIVITY", RESISTIVITY))
    N_PASSIVE = int(params.get("N_PASSIVE", N_PASSIVE))
    PASSIVE_OFFSET = int(params.get("PASSIVE_OFFSET", PASSIVE_OFFSET))

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
    h  = eos.enthalpy(rho, p)
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

    gamma0 = eos.gamma_eff(max(D, SMALL))
    p = (gamma0 - 1.0) * (E - D)
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
        h  = eos.enthalpy(rho, p)
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
        gamma0 = eos.gamma_eff(max(D, SMALL))
        p = (gamma0 - 1.0) * (E - D)
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
def rmhd_f_of_Z(Z, D, tau, B2, SB, S2):
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
    gamma = eos.gamma_eff(rho)
    p = (h - 1.0) * rho * (gamma - 1.0) / gamma
    if p < SMALL: p = SMALL
    if p > P_MAX: p = P_MAX
    b2 = B2 / W2 + vb*vb
    tau_calc = Z - p + 0.5*B2 + 0.5*b2 - D
    return tau_calc - tau

@nb.njit(fastmath=True)
def _cons_to_prim_rmhd_impl(D, Sx, Sy, Sz, tau, Bx, By, Bz, psi):
    # Improved recovery: solve for Z = rho h W^2 using a damped Newton iteration.
    # status bits: 1=bisection used, 2=fallback to hydro, 4=velocity clipped.
    status = 0
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
    h0 = eos.enthalpy(rho0, p0)
    Z0 = max(rho0*h0*W0*W0, SMALL)
    Z1 = max(D + tau + B2, SMALL)

    ok = False
    Z = Z0
    for attempt in range(2):
        if attempt == 1:
            Z = Z1
        for _ in range(50):
            f = rmhd_f_of_Z(Z, D, tau, B2, SB, S2)
            if np.abs(f) < 1e-10 * max(1.0, tau):
                ok = True
                break

            eps = 1e-6 * max(1.0, Z)
            fp = rmhd_f_of_Z(Z + eps, D, tau, B2, SB, S2)

            dfdZ = (fp - f) / eps
            if dfdZ == 0.0:
                break
            dZ = -f / dfdZ
            if dZ >  0.5*Z: dZ =  0.5*Z
            if dZ < -0.5*Z: dZ = -0.5*Z
            Z += dZ
            if Z < SMALL:
                Z = SMALL
        if ok:
            break

    if not ok:
        # Fallback to bisection if Newton stalls.
        status |= 1
        Zmin = SMALL
        Zmax = max(Z, Z1, 1.0)
        fmin = rmhd_f_of_Z(Zmin, D, tau, B2, SB, S2)
        fmax = rmhd_f_of_Z(Zmax, D, tau, B2, SB, S2)
        for _ in range(40):
            if fmin * fmax <= 0.0:
                break
            Zmax *= 2.0
            fmax = rmhd_f_of_Z(Zmax, D, tau, B2, SB, S2)
        if fmin * fmax <= 0.0:
            for _ in range(60):
                Zmid = 0.5*(Zmin + Zmax)
                fmid = rmhd_f_of_Z(Zmid, D, tau, B2, SB, S2)
                if np.abs(fmid) < 1e-10 * max(1.0, tau):
                    Z = Zmid
                    break
                if fmin * fmid <= 0.0:
                    Zmax = Zmid
                    fmax = fmid
                else:
                    Zmin = Zmid
                    fmin = fmid
                Z = 0.5*(Zmin + Zmax)

    # recover velocities
    ZpB = Z + B2
    vb = SB / max(Z, SMALL)
    vx = (Sx + vb*Bx) / max(ZpB, SMALL)
    vy = (Sy + vb*By) / max(ZpB, SMALL)
    vz = (Sz + vb*Bz) / max(ZpB, SMALL)

    v2 = vx*vx + vy*vy + vz*vz
    vmax2 = V_MAX*V_MAX
    if v2 >= vmax2:
        status |= 4
        fac = V_MAX / np.sqrt(v2 + 1e-32)
        vx *= fac; vy *= fac; vz *= fac
    W = 1.0 / np.sqrt(1.0 - (vx*vx + vy*vy + vz*vz))
    rho = D / max(W, SMALL)
    h = Z / max(rho*W*W, SMALL)
    gamma = eos.gamma_eff(rho)
    p = (h - 1.0) * rho * (gamma - 1.0) / gamma

    # If recovery yields invalid values, fall back to hydro guess with B kept.
    if not np.isfinite(rho) or not np.isfinite(p) or rho <= 0.0 or p <= 0.0:
        status |= 2
        rho, vx, vy, vz, p = rho0, vx0, vy0, vz0, p0

    rho, vx, vy, vz, p, Bx, By, Bz, psi = floor_prim_rmhd(rho, vx, vy, vz, p, Bx, By, Bz, psi)
    return rho, vx, vy, vz, p, Bx, By, Bz, psi, status

@nb.njit(fastmath=True)
def cons_to_prim_rmhd(D, Sx, Sy, Sz, tau, Bx, By, Bz, psi):
    rho, vx, vy, vz, p, Bx, By, Bz, psi, _ = _cons_to_prim_rmhd_impl(
        D, Sx, Sy, Sz, tau, Bx, By, Bz, psi
    )
    return rho, vx, vy, vz, p, Bx, By, Bz, psi

@nb.njit(fastmath=True)
def cons_to_prim_rmhd_status(D, Sx, Sy, Sz, tau, Bx, By, Bz, psi):
    return _cons_to_prim_rmhd_impl(D, Sx, Sy, Sz, tau, Bx, By, Bz, psi)

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
def hlle_scalar(qL, qR, fL, fR, sL, sR):
    if sL >= 0.0:
        return fL
    if sR <= 0.0:
        return fR
    return (sR*fL - sL*fR + sL*sR*(qR - qL)) / (sR - sL + SMALL)

@nb.njit(fastmath=True)
def rmhd_fast_speed(rho, p, vx, vy, vz, Bx, By, Bz):
    # Approximate fast magnetosonic speed in fluid frame.
    v2 = vx*vx + vy*vy + vz*vz
    if v2 >= V_MAX*V_MAX:
        v2 = V_MAX*V_MAX - 1e-14
    W = 1.0 / np.sqrt(1.0 - v2)
    if rho < SMALL:
        rho = SMALL
    if p < SMALL:
        p = SMALL
    if p > P_MAX:
        p = P_MAX

    h = eos.enthalpy(rho, p)
    rhoh = rho*h
    B2 = Bx*Bx + By*By + Bz*Bz
    vb = vx*Bx + vy*By + vz*Bz
    b0 = W*vb
    bx = (Bx + b0*vx)/W
    by = (By + b0*vy)/W
    bz = (Bz + b0*vz)/W
    b2 = (bx*bx + by*by + bz*bz) - b0*b0

    gamma = eos.gamma_eff(rho)
    cs2 = gamma*p/max(rhoh, SMALL)
    if cs2 < 0.0: cs2 = 0.0
    if cs2 > 1.0 - 1e-14: cs2 = 1.0 - 1e-14
    ca2 = b2 / max(rhoh + b2, SMALL)
    if ca2 < 0.0: ca2 = 0.0
    if ca2 > 1.0 - 1e-14: ca2 = 1.0 - 1e-14

    a = cs2 + ca2 - cs2*ca2
    disc = a*a - 4.0*cs2*ca2
    if disc < 0.0:
        disc = 0.0
    cf2 = 0.5*(a + np.sqrt(disc))
    if cf2 < 0.0: cf2 = 0.0
    if cf2 > 1.0 - 1e-14: cf2 = 1.0 - 1e-14
    return np.sqrt(cf2)

@nb.njit(fastmath=True)
def rmhd_eig_speeds(vn, cf):
    dp = 1.0 + vn*cf
    dm = 1.0 - vn*cf
    if dp == 0.0: dp = SMALL
    if dm == 0.0: dm = SMALL
    lp = (vn + cf)/dp
    lm = (vn - cf)/dm
    if lp >  1.0: lp =  1.0
    if lp < -1.0: lp = -1.0
    if lm >  1.0: lm =  1.0
    if lm < -1.0: lm = -1.0
    return lm, lp

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
    if RIEMANN_ID == 2:
        return hll_d_flux_dir(0, UL, UR, FL, FR, sL, sR, pL, pR, vL, vR)
    if RIEMANN_ID == 1:
        hllc = hllc_hydro_flux_x(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR)
        out = hlle(UL, UR, FL, FR, sL, sR)
        out[0:5] = hllc
        return out
    return hlle(UL, UR, FL, FR, sL, sR)

@nb.njit(fastmath=True)
def hllc_contact_speed_x(UL, UR, sL, sR, pL, pR, vL, vR):
    DL = UL[0]; SxL = UL[1]; SyL = UL[2]; SzL = UL[3]
    DR = UR[0]; SxR = UR[1]; SyR = UR[2]; SzR = UR[3]
    denom = (DL*(sL - vL) - DR*(sR - vR)) + SMALL
    return (pR - pL + SxL*(sL - vL) - SxR*(sR - vR)) / denom

@nb.njit(fastmath=True)
def hllc_contact_speed_y(UL, UR, sL, sR, pL, pR, vL, vR):
    DL = UL[0]; SxL = UL[1]; SyL = UL[2]; SzL = UL[3]
    DR = UR[0]; SxR = UR[1]; SyR = UR[2]; SzR = UR[3]
    denom = (DL*(sL - vL) - DR*(sR - vR)) + SMALL
    return (pR - pL + SyL*(sL - vL) - SyR*(sR - vR)) / denom

@nb.njit(fastmath=True)
def hllc_contact_speed_z(UL, UR, sL, sR, pL, pR, vL, vR):
    DL = UL[0]; SxL = UL[1]; SyL = UL[2]; SzL = UL[3]
    DR = UR[0]; SxR = UR[1]; SyR = UR[2]; SzR = UR[3]
    denom = (DL*(sL - vL) - DR*(sR - vR)) + SMALL
    return (pR - pL + SzL*(sL - vL) - SzR*(sR - vR)) / denom

@nb.njit(fastmath=True)
def hll_d_flux_dir(n, UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    # Experimental RMHD HLLD (approximate). Falls back to HLLE on invalid states.
    if sL >= 0.0:
        return FL
    if sR <= 0.0:
        return FR

    # index mapping
    if n == 0:
        vn = 1; vt1 = 2; vt2 = 3
        bn = 5; bt1 = 6; bt2 = 7
    elif n == 1:
        vn = 2; vt1 = 1; vt2 = 3
        bn = 6; bt1 = 5; bt2 = 7
    else:
        vn = 3; vt1 = 1; vt2 = 2
        bn = 7; bt1 = 5; bt2 = 6

    # Unpack primitives from fluxes (we don't have prim arrays here; infer from FL/FR using UL/UR).
    # Use UL/UR as conserved and pL/pR for thermal pressure to construct approximate primitives.
    # Approximate: use vL/vR for normal velocity and infer tangentials from flux ratios.
    vL_n = vL
    vR_n = vR
    if UL[0] <= SMALL or UR[0] <= SMALL:
        return hlle(UL, UR, FL, FR, sL, sR)

    # Reconstruct magnetic fields from fluxes: use Bn from UL/UR (stored in UL[5:8]).
    BnL = UL[bn]; BnR = UR[bn]
    Bt1L = UL[bt1]; Bt2L = UL[bt2]
    Bt1R = UR[bt1]; Bt2R = UR[bt2]
    Bn = 0.5*(BnL + BnR)
    if abs(Bn) < 1e-8:
        return hlle(UL, UR, FL, FR, sL, sR)

    # Contact speed from HLLC
    if n == 0:
        sM = hllc_contact_speed_x(UL, UR, sL, sR, pL, pR, vL_n, vR_n)
    elif n == 1:
        sM = hllc_contact_speed_y(UL, UR, sL, sR, pL, pR, vL_n, vR_n)
    else:
        sM = hllc_contact_speed_z(UL, UR, sL, sR, pL, pR, vL_n, vR_n)

    DL = UL[0]; DR = UR[0]
    DstarL = DL*(sL - vL_n)/(sL - sM + SMALL)
    DstarR = DR*(sR - vR_n)/(sR - sM + SMALL)
    if DstarL <= SMALL or DstarR <= SMALL:
        return hlle(UL, UR, FL, FR, sL, sR)

    # Tangential fields and velocities in star region
    Bt1L_star = Bt1L*(sL - vL_n)/(sL - sM + SMALL)
    Bt2L_star = Bt2L*(sL - vL_n)/(sL - sM + SMALL)
    Bt1R_star = Bt1R*(sR - vR_n)/(sR - sM + SMALL)
    Bt2R_star = Bt2R*(sR - vR_n)/(sR - sM + SMALL)

    denomL = DL*(sL - vL_n) - Bn*Bn
    denomR = DR*(sR - vR_n) - Bn*Bn
    if abs(denomL) < SMALL or abs(denomR) < SMALL:
        return hlle(UL, UR, FL, FR, sL, sR)

    vt1L = UL[vt1] / max(DL, SMALL)
    vt2L = UL[vt2] / max(DL, SMALL)
    vt1R = UR[vt1] / max(DR, SMALL)
    vt2R = UR[vt2] / max(DR, SMALL)

    vt1L_star = vt1L - Bn*(Bt1L - Bt1L_star)/denomL
    vt2L_star = vt2L - Bn*(Bt2L - Bt2L_star)/denomL
    vt1R_star = vt1R - Bn*(Bt1R - Bt1R_star)/denomR
    vt2R_star = vt2R - Bn*(Bt2R - Bt2R_star)/denomR

    # Alfven speeds
    sL_star = sM - abs(Bn)/np.sqrt(DstarL + SMALL)
    sR_star = sM + abs(Bn)/np.sqrt(DstarR + SMALL)

    sqrtDL = np.sqrt(DstarL)
    sqrtDR = np.sqrt(DstarR)
    sgn = 1.0 if Bn >= 0.0 else -1.0
    vt1_starstar = (sqrtDL*vt1L_star + sqrtDR*vt1R_star + sgn*(Bt1R_star - Bt1L_star)) / (sqrtDL + sqrtDR + SMALL)
    vt2_starstar = (sqrtDL*vt2L_star + sqrtDR*vt2R_star + sgn*(Bt2R_star - Bt2L_star)) / (sqrtDL + sqrtDR + SMALL)
    Bt1_starstar = (sqrtDL*Bt1R_star + sqrtDR*Bt1L_star + sgn*sqrtDL*sqrtDR*(vt1L_star - vt1R_star)) / (sqrtDL + sqrtDR + SMALL)
    Bt2_starstar = (sqrtDL*Bt2R_star + sqrtDR*Bt2L_star + sgn*sqrtDL*sqrtDR*(vt2L_star - vt2R_star)) / (sqrtDL + sqrtDR + SMALL)

    # Build star primitives (approximate) and conserved states for flux construction.
    def build_state(Dstar, vtn1, vtn2, Bt1s, Bt2s, psi):
        rho = max(Dstar, SMALL)
        if n == 0:
            vx = sM; vy = vtn1; vz = vtn2; Bx = Bn; By = Bt1s; Bz = Bt2s
        elif n == 1:
            vy = sM; vx = vtn1; vz = vtn2; By = Bn; Bx = Bt1s; Bz = Bt2s
        else:
            vz = sM; vx = vtn1; vy = vtn2; Bz = Bn; Bx = Bt1s; By = Bt2s
        p = pL if Dstar == DstarL else pR
        return rho, vx, vy, vz, p, Bx, By, Bz, psi

    primL_star = build_state(DstarL, vt1L_star, vt2L_star, Bt1L_star, Bt2L_star, UL[8])
    primR_star = build_state(DstarR, vt1R_star, vt2R_star, Bt1R_star, Bt2R_star, UR[8])
    prim_starstar = build_state(0.5*(DstarL + DstarR), vt1_starstar, vt2_starstar, Bt1_starstar, Bt2_starstar, 0.5*(UL[8] + UR[8]))

    ULs = np.array(prim_to_cons_rmhd(*primL_star, GAMMA))
    URs = np.array(prim_to_cons_rmhd(*primR_star, GAMMA))
    Us = np.array(prim_to_cons_rmhd(*prim_starstar, GAMMA))

    if n == 0:
        FLs = flux_rmhd_x(np.array(primL_star))
        FRs = flux_rmhd_x(np.array(primR_star))
        Fs = flux_rmhd_x(np.array(prim_starstar))
    elif n == 1:
        FLs = flux_rmhd_y(np.array(primL_star))
        FRs = flux_rmhd_y(np.array(primR_star))
        Fs = flux_rmhd_y(np.array(prim_starstar))
    else:
        FLs = flux_rmhd_z(np.array(primL_star))
        FRs = flux_rmhd_z(np.array(primR_star))
        Fs = flux_rmhd_z(np.array(prim_starstar))

    if sL <= 0.0 and 0.0 <= sL_star:
        return FL + sL*(ULs - UL)
    if sL_star <= 0.0 and 0.0 <= sM:
        return FL + sL*(ULs - UL) + sL_star*(Us - ULs)
    if sM <= 0.0 and 0.0 <= sR_star:
        return FR + sR*(URs - UR) + sR_star*(Us - URs)
    if sR_star <= 0.0 and 0.0 <= sR:
        return FR + sR*(URs - UR)
    return hlle(UL, UR, FL, FR, sL, sR)

@nb.njit(fastmath=True)
def riemann_flux_rmhd_y(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    if RIEMANN_ID == 2:
        return hll_d_flux_dir(1, UL, UR, FL, FR, sL, sR, pL, pR, vL, vR)
    if RIEMANN_ID == 1:
        hllc = hllc_hydro_flux_y(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR)
        out = hlle(UL, UR, FL, FR, sL, sR)
        out[0:5] = hllc
        return out
    return hlle(UL, UR, FL, FR, sL, sR)

@nb.njit(fastmath=True)
def riemann_flux_rmhd_z(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR):
    if RIEMANN_ID == 2:
        return hll_d_flux_dir(2, UL, UR, FL, FR, sL, sR, pL, pR, vL, vR)
    if RIEMANN_ID == 1:
        hllc = hllc_hydro_flux_z(UL, UR, FL, FR, sL, sR, pL, pR, vL, vR)
        out = hlle(UL, UR, FL, FR, sL, sR)
        out[0:5] = hllc
        return out
    return hlle(UL, UR, FL, FR, sL, sR)

@nb.njit(parallel=True, fastmath=True)
def compute_rhs_rmhd(pr, nx, ny, nz, dx, dy, dz):
    rhs = np.zeros((9 + N_PASSIVE, nx, ny, nz))
    i0, i1 = 2, nx-2
    j0, j1 = 2, ny-2
    k0, k1 = 2, nz-2
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
                cfL = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL, lpL = rmhd_eig_speeds(vxL, cfL)
                lmR, lpR = rmhd_eig_speeds(vxR, cfR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                cfL = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL, lpL = rmhd_eig_speeds(vxL, cfL)
                lmR, lpR = rmhd_eig_speeds(vxR, cfR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                cfL = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL, lpL = rmhd_eig_speeds(vxL, cfL)
                lmR, lpR = rmhd_eig_speeds(vxR, cfR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                cfL = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL, lpL = rmhd_eig_speeds(vxL, cfL)
                lmR, lpR = rmhd_eig_speeds(vxR, cfR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                cfL = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL, lpL = rmhd_eig_speeds(vxL, cfL)
                lmR, lpR = rmhd_eig_speeds(vxR, cfR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                FxL = riemann_flux_rmhd_x(UL, UR, FL, FR, sL, sR, pL, pR, vxL, vxR)
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_x(UL, UR, sL, sR, pL, pR, vxL, vxR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_x(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_x(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FxL[5:9] = Fm[5:9]
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_x(UL, UR, sL, sR, pL, pR, vxL, vxR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_x(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_x(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FxL[5:9] = Fm[5:9]
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_x(UL, UR, sL, sR, pL, pR, vxL, vxR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_x(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_x(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FxL[5:9] = Fm[5:9]
                if N_PASSIVE > 0:
                    FxL_tr = np.empty(N_PASSIVE)
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        tLq = weno5_left(pr[idx,i-3,j,k], pr[idx,i-2,j,k], pr[idx,i-1,j,k], pr[idx,i,j,k], pr[idx,i+1,j,k])
                        tRq = weno5_right(pr[idx,i+2,j,k], pr[idx,i+1,j,k], pr[idx,i,j,k], pr[idx,i-1,j,k], pr[idx,i-2,j,k])
                        FxL_tr[t] = hlle_scalar(tLq, tRq, tLq*vxL, tRq*vxR, sL, sR)
                if N_PASSIVE > 0:
                    FxL_tr = np.empty(N_PASSIVE)
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        l_im1, r_im1 = ppm_reconstruct(pr[idx,i-3,j,k], pr[idx,i-2,j,k], pr[idx,i-1,j,k],
                                                       pr[idx,i-0,j,k], pr[idx,i+1,j,k])
                        l_i, r_i = ppm_reconstruct(pr[idx,i-2,j,k], pr[idx,i-1,j,k], pr[idx,i,j,k],
                                                   pr[idx,i+1,j,k], pr[idx,i+2,j,k])
                        tLq = r_im1
                        tRq = l_i
                        FxL_tr[t] = hlle_scalar(tLq, tRq, tLq*vxL, tRq*vxR, sL, sR)
                if N_PASSIVE > 0:
                    FxL_tr = np.empty(N_PASSIVE)
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        dqL = pr[idx, i-1, j, k] - pr[idx, i-2, j, k]
                        dqR = pr[idx, i,   j, k] - pr[idx, i-1, j, k]
                        slL_t = limiter(dqL, dqR)
                        dqL2 = pr[idx, i,   j, k] - pr[idx, i-1, j, k]
                        dqR2 = pr[idx, i+1, j, k] - pr[idx, i,   j, k]
                        slR_t = limiter(dqL2, dqR2)
                        tL = pr[idx, i-1, j, k] + 0.5*slL_t
                        tR = pr[idx, i,   j, k] - 0.5*slR_t
                        FxL_tr[t] = hlle_scalar(tL, tR, tL*vxL, tR*vxR, sL, sR)

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
                cfL2 = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR2 = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL2, lpL2 = rmhd_eig_speeds(vxL, cfL2)
                lmR2, lpR2 = rmhd_eig_speeds(vxR, cfR2)
                sL2 = min(lmL2, lmR2); sR2 = max(lpL2, lpR2)
                FxR = riemann_flux_rmhd_x(UL, UR, FL, FR, sL2, sR2, pL, pR, vxL, vxR)
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_x(UL, UR, sL2, sR2, pL, pR, vxL, vxR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_x(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_x(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FxR[5:9] = Fm[5:9]
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_x(UL, UR, sL2, sR2, pL, pR, vxL, vxR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_x(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_x(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FxR[5:9] = Fm[5:9]
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_x(UL, UR, sL2, sR2, pL, pR, vxL, vxR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_x(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_x(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FxR[5:9] = Fm[5:9]
                if N_PASSIVE > 0:
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        tLq = weno5_left(pr[idx,i-2,j,k], pr[idx,i-1,j,k], pr[idx,i,j,k], pr[idx,i+1,j,k], pr[idx,i+2,j,k])
                        tRq = weno5_right(pr[idx,i+3,j,k], pr[idx,i+2,j,k], pr[idx,i+1,j,k], pr[idx,i,j,k], pr[idx,i-1,j,k])
                        FxR_tr = hlle_scalar(tLq, tRq, tLq*vxL, tRq*vxR, sL, sR)
                        rhs[9 + t, i, j, k] -= (FxR_tr - FxL_tr[t]) / dx
                if N_PASSIVE > 0:
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        l_i, r_i = ppm_reconstruct(pr[idx,i-2,j,k], pr[idx,i-1,j,k], pr[idx,i,j,k],
                                                   pr[idx,i+1,j,k], pr[idx,i+2,j,k])
                        l_ip1, r_ip1 = ppm_reconstruct(pr[idx,i-1,j,k], pr[idx,i,j,k], pr[idx,i+1,j,k],
                                                       pr[idx,i+2,j,k], pr[idx,i+3,j,k])
                        tLq = r_i
                        tRq = l_ip1
                        FxR_tr = hlle_scalar(tLq, tRq, tLq*vxL, tRq*vxR, sL, sR)
                        rhs[9 + t, i, j, k] -= (FxR_tr - FxL_tr[t]) / dx
                if N_PASSIVE > 0:
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        dqL = pr[idx, i+1, j, k] - pr[idx, i,   j, k]
                        dqR = pr[idx, i+2, j, k] - pr[idx, i+1, j, k]
                        slRp1_t = limiter(dqL, dqR)
                        dqL2 = pr[idx, i,   j, k] - pr[idx, i-1, j, k]
                        dqR2 = pr[idx, i+1, j, k] - pr[idx, i,   j, k]
                        slR_t = limiter(dqL2, dqR2)
                        tL = pr[idx, i,   j, k] + 0.5*slR_t
                        tR = pr[idx, i+1, j, k] - 0.5*slRp1_t
                        FxR_tr = hlle_scalar(tL, tR, tL*vxL, tR*vxR, sL, sR)
                        rhs[9 + t, i, j, k] -= (FxR_tr - FxL_tr[t]) / dx

                rhs[0:9, i, j, k] -= (FxR - FxL) / dx

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
                cfL = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL, lpL = rmhd_eig_speeds(vyL, cfL)
                lmR, lpR = rmhd_eig_speeds(vyR, cfR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                FyD = riemann_flux_rmhd_y(UL, UR, FL, FR, sL, sR, pL, pR, vyL, vyR)
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_y(UL, UR, sL, sR, pL, pR, vyL, vyR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FyD[5:9] = Fm[5:9]
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_y(UL, UR, sL, sR, pL, pR, vyL, vyR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FyD[5:9] = Fm[5:9]
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_y(UL, UR, sL, sR, pL, pR, vyL, vyR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FyD[5:9] = Fm[5:9]
                if N_PASSIVE > 0:
                    FyD_tr = np.empty(N_PASSIVE)
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        tLq = weno5_left(pr[idx,i,j-3,k], pr[idx,i,j-2,k], pr[idx,i,j-1,k], pr[idx,i,j,k], pr[idx,i,j+1,k])
                        tRq = weno5_right(pr[idx,i,j+2,k], pr[idx,i,j+1,k], pr[idx,i,j,k], pr[idx,i,j-1,k], pr[idx,i,j-2,k])
                        FyD_tr[t] = hlle_scalar(tLq, tRq, tLq*vyL, tRq*vyR, sL, sR)
                if N_PASSIVE > 0:
                    FyD_tr = np.empty(N_PASSIVE)
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        l_jm1, r_jm1 = ppm_reconstruct(pr[idx,i,j-3,k], pr[idx,i,j-2,k], pr[idx,i,j-1,k],
                                                       pr[idx,i,j-0,k], pr[idx,i,j+1,k])
                        l_j, r_j = ppm_reconstruct(pr[idx,i,j-2,k], pr[idx,i,j-1,k], pr[idx,i,j,k],
                                                   pr[idx,i,j+1,k], pr[idx,i,j+2,k])
                        tLq = r_jm1
                        tRq = l_j
                        FyD_tr[t] = hlle_scalar(tLq, tRq, tLq*vyL, tRq*vyR, sL, sR)
                if N_PASSIVE > 0:
                    FyD_tr = np.empty(N_PASSIVE)
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        dqL = pr[idx, i, j-1, k] - pr[idx, i, j-2, k]
                        dqR = pr[idx, i, j,   k] - pr[idx, i, j-1, k]
                        slL_t = limiter(dqL, dqR)
                        dqL2 = pr[idx, i, j,   k] - pr[idx, i, j-1, k]
                        dqR2 = pr[idx, i, j+1, k] - pr[idx, i, j,   k]
                        slR_t = limiter(dqL2, dqR2)
                        tL = pr[idx, i, j-1, k] + 0.5*slL_t
                        tR = pr[idx, i, j,   k] - 0.5*slR_t
                        FyD_tr[t] = hlle_scalar(tL, tR, tL*vyL, tR*vyR, sL, sR)

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
                cfL2 = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR2 = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL2, lpL2 = rmhd_eig_speeds(vyL, cfL2)
                lmR2, lpR2 = rmhd_eig_speeds(vyR, cfR2)
                sL2 = min(lmL2, lmR2); sR2 = max(lpL2, lpR2)
                FyU = riemann_flux_rmhd_y(UL, UR, FL, FR, sL2, sR2, pL, pR, vyL, vyR)
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_y(UL, UR, sL2, sR2, pL, pR, vyL, vyR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FyU[5:9] = Fm[5:9]
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_y(UL, UR, sL2, sR2, pL, pR, vyL, vyR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FyU[5:9] = Fm[5:9]
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_y(UL, UR, sL2, sR2, pL, pR, vyL, vyR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FyU[5:9] = Fm[5:9]
                if N_PASSIVE > 0:
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        tLq = weno5_left(pr[idx,i,j-2,k], pr[idx,i,j-1,k], pr[idx,i,j,k], pr[idx,i,j+1,k], pr[idx,i,j+2,k])
                        tRq = weno5_right(pr[idx,i,j+3,k], pr[idx,i,j+2,k], pr[idx,i,j+1,k], pr[idx,i,j,k], pr[idx,i,j-1,k])
                        FyU_tr = hlle_scalar(tLq, tRq, tLq*vyL, tRq*vyR, sL, sR)
                        rhs[9 + t, i, j, k] -= (FyU_tr - FyD_tr[t]) / dy
                if N_PASSIVE > 0:
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        l_j, r_j = ppm_reconstruct(pr[idx,i,j-2,k], pr[idx,i,j-1,k], pr[idx,i,j,k],
                                                   pr[idx,i,j+1,k], pr[idx,i,j+2,k])
                        l_jp1, r_jp1 = ppm_reconstruct(pr[idx,i,j-1,k], pr[idx,i,j,k], pr[idx,i,j+1,k],
                                                       pr[idx,i,j+2,k], pr[idx,i,j+3,k])
                        tLq = r_j
                        tRq = l_jp1
                        FyU_tr = hlle_scalar(tLq, tRq, tLq*vyL, tRq*vyR, sL, sR)
                        rhs[9 + t, i, j, k] -= (FyU_tr - FyD_tr[t]) / dy
                if N_PASSIVE > 0:
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        dqL = pr[idx, i, j+1, k] - pr[idx, i, j,   k]
                        dqR = pr[idx, i, j+2, k] - pr[idx, i, j+1, k]
                        slRp1_t = limiter(dqL, dqR)
                        dqL2 = pr[idx, i, j,   k] - pr[idx, i, j-1, k]
                        dqR2 = pr[idx, i, j+1, k] - pr[idx, i, j,   k]
                        slR_t = limiter(dqL2, dqR2)
                        tL = pr[idx, i, j,   k] + 0.5*slR_t
                        tR = pr[idx, i, j+1, k] - 0.5*slRp1_t
                        FyU_tr = hlle_scalar(tL, tR, tL*vyL, tR*vyR, sL, sR)
                        rhs[9 + t, i, j, k] -= (FyU_tr - FyD_tr[t]) / dy

                rhs[0:9, i, j, k] -= (FyU - FyD) / dy

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
                cfL = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL, lpL = rmhd_eig_speeds(vzL, cfL)
                lmR, lpR = rmhd_eig_speeds(vzR, cfR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                FzB = riemann_flux_rmhd_z(UL, UR, FL, FR, sL, sR, pL, pR, vzL, vzR)
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_z(UL, UR, sL, sR, pL, pR, vzL, vzR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FzB[5:9] = Fm[5:9]
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_z(UL, UR, sL, sR, pL, pR, vzL, vzR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FzB[5:9] = Fm[5:9]
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_z(UL, UR, sL, sR, pL, pR, vzL, vzR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FzB[5:9] = Fm[5:9]
                if N_PASSIVE > 0:
                    FzB_tr = np.empty(N_PASSIVE)
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        tLq = weno5_left(pr[idx,i,j,k-3], pr[idx,i,j,k-2], pr[idx,i,j,k-1], pr[idx,i,j,k], pr[idx,i,j,k+1])
                        tRq = weno5_right(pr[idx,i,j,k+2], pr[idx,i,j,k+1], pr[idx,i,j,k], pr[idx,i,j,k-1], pr[idx,i,j,k-2])
                        FzB_tr[t] = hlle_scalar(tLq, tRq, tLq*vzL, tRq*vzR, sL, sR)
                if N_PASSIVE > 0:
                    FzB_tr = np.empty(N_PASSIVE)
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        l_km1, r_km1 = ppm_reconstruct(pr[idx,i,j,k-3], pr[idx,i,j,k-2], pr[idx,i,j,k-1],
                                                       pr[idx,i,j,k-0], pr[idx,i,j,k+1])
                        l_k, r_k = ppm_reconstruct(pr[idx,i,j,k-2], pr[idx,i,j,k-1], pr[idx,i,j,k],
                                                   pr[idx,i,j,k+1], pr[idx,i,j,k+2])
                        tLq = r_km1
                        tRq = l_k
                        FzB_tr[t] = hlle_scalar(tLq, tRq, tLq*vzL, tRq*vzR, sL, sR)
                if N_PASSIVE > 0:
                    FzB_tr = np.empty(N_PASSIVE)
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        dqL = pr[idx, i, j, k-1] - pr[idx, i, j, k-2]
                        dqR = pr[idx, i, j, k  ] - pr[idx, i, j, k-1]
                        slL_t = limiter(dqL, dqR)
                        dqL2 = pr[idx, i, j, k  ] - pr[idx, i, j, k-1]
                        dqR2 = pr[idx, i, j, k+1] - pr[idx, i, j, k  ]
                        slR_t = limiter(dqL2, dqR2)
                        tL = pr[idx, i, j, k-1] + 0.5*slL_t
                        tR = pr[idx, i, j, k  ] - 0.5*slR_t
                        FzB_tr[t] = hlle_scalar(tL, tR, tL*vzL, tR*vzR, sL, sR)

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
                cfL2 = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR2 = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL2, lpL2 = rmhd_eig_speeds(vzL, cfL2)
                lmR2, lpR2 = rmhd_eig_speeds(vzR, cfR2)
                sL2 = min(lmL2, lmR2); sR2 = max(lpL2, lpR2)
                FzF = riemann_flux_rmhd_z(UL, UR, FL, FR, sL2, sR2, pL, pR, vzL, vzR)
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_z(UL, UR, sL2, sR2, pL, pR, vzL, vzR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FzF[5:9] = Fm[5:9]
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_z(UL, UR, sL2, sR2, pL, pR, vzL, vzR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FzF[5:9] = Fm[5:9]
                if RIEMANN_ID == 1:
                    sM = hllc_contact_speed_z(UL, UR, sL2, sR2, pL, pR, vzL, vzR)
                    if sM >= 0.0:
                        Fm = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                    else:
                        Fm = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                    FzF[5:9] = Fm[5:9]
                if N_PASSIVE > 0:
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        tLq = weno5_left(pr[idx,i,j,k-2], pr[idx,i,j,k-1], pr[idx,i,j,k], pr[idx,i,j,k+1], pr[idx,i,j,k+2])
                        tRq = weno5_right(pr[idx,i,j,k+3], pr[idx,i,j,k+2], pr[idx,i,j,k+1], pr[idx,i,j,k], pr[idx,i,j,k-1])
                        FzF_tr = hlle_scalar(tLq, tRq, tLq*vzL, tRq*vzR, sL, sR)
                        rhs[9 + t, i, j, k] -= (FzF_tr - FzB_tr[t]) / dz
                if N_PASSIVE > 0:
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        l_k, r_k = ppm_reconstruct(pr[idx,i,j,k-2], pr[idx,i,j,k-1], pr[idx,i,j,k],
                                                   pr[idx,i,j,k+1], pr[idx,i,j,k+2])
                        l_kp1, r_kp1 = ppm_reconstruct(pr[idx,i,j,k-1], pr[idx,i,j,k], pr[idx,i,j,k+1],
                                                       pr[idx,i,j,k+2], pr[idx,i,j,k+3])
                        tLq = r_k
                        tRq = l_kp1
                        FzF_tr = hlle_scalar(tLq, tRq, tLq*vzL, tRq*vzR, sL, sR)
                        rhs[9 + t, i, j, k] -= (FzF_tr - FzB_tr[t]) / dz
                if N_PASSIVE > 0:
                    for t in range(N_PASSIVE):
                        idx = PASSIVE_OFFSET + t
                        dqL = pr[idx, i, j, k+1] - pr[idx, i, j, k  ]
                        dqR = pr[idx, i, j, k+2] - pr[idx, i, j, k+1]
                        slRp1_t = limiter(dqL, dqR)
                        dqL2 = pr[idx, i, j, k  ] - pr[idx, i, j, k-1]
                        dqR2 = pr[idx, i, j, k+1] - pr[idx, i, j, k  ]
                        slR_t = limiter(dqL2, dqR2)
                        tL = pr[idx, i, j, k  ] + 0.5*slR_t
                        tR = pr[idx, i, j, k+1] - 0.5*slRp1_t
                        FzF_tr = hlle_scalar(tL, tR, tL*vzL, tR*vzR, sL, sR)
                        rhs[9 + t, i, j, k] -= (FzF_tr - FzB_tr[t]) / dz

                rhs[0:9, i, j, k] -= (FzF - FzB) / dz
                # GLM damping source term
                rhs[8, i, j, k] -= GLM_CP*GLM_CP * pr[8, i, j, k]
                if RESISTIVE_ENABLED and RESISTIVITY > 0.0:
                    invdx2 = 1.0 / (dx*dx + SMALL)
                    invdy2 = 1.0 / (dy*dy + SMALL)
                    invdz2 = 1.0 / (dz*dz + SMALL)
                    for comp in range(5, 8):
                        lap = (
                            (pr[comp, i+1, j, k] - 2.0*pr[comp, i, j, k] + pr[comp, i-1, j, k]) * invdx2 +
                            (pr[comp, i, j+1, k] - 2.0*pr[comp, i, j, k] + pr[comp, i, j-1, k]) * invdy2 +
                            (pr[comp, i, j, k+1] - 2.0*pr[comp, i, j, k] + pr[comp, i, j, k-1]) * invdz2
                        )
                        rhs[comp, i, j, k] += RESISTIVITY * lap

    return rhs

@nb.njit(parallel=True, fastmath=True)
def compute_rhs_ppm(pr, nx, ny, nz, dx, dy, dz):
    rhs = np.zeros((9 + N_PASSIVE, nx, ny, nz))
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
                cfL2 = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR2 = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL2, lpL2 = rmhd_eig_speeds(vxL, cfL2)
                lmR2, lpR2 = rmhd_eig_speeds(vxR, cfR2)
                sL2 = min(lmL2, lmR2); sR2 = max(lpL2, lpR2)
                FxR = riemann_flux_rmhd_x(UL, UR, FL, FR, sL2, sR2, pL, pR, vxL, vxR)

                rhs[0:9, i, j, k] -= (FxR - FxL) / dx

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
                cfL = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL, lpL = rmhd_eig_speeds(vyL, cfL)
                lmR, lpR = rmhd_eig_speeds(vyR, cfR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                FyD = riemann_flux_rmhd_y(UL, UR, FL, FR, sL, sR, pL, pR, vyL, vyR)

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4], qL2[5], qL2[6], qL2[7], qL2[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4], qR2[5], qR2[6], qR2[7], qR2[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                cfL2 = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR2 = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL2, lpL2 = rmhd_eig_speeds(vyL, cfL2)
                lmR2, lpR2 = rmhd_eig_speeds(vyR, cfR2)
                sL2 = min(lmL2, lmR2); sR2 = max(lpL2, lpR2)
                FyU = riemann_flux_rmhd_y(UL, UR, FL, FR, sL2, sR2, pL, pR, vyL, vyR)

                rhs[0:9, i, j, k] -= (FyU - FyD) / dy

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
                cfL = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL, lpL = rmhd_eig_speeds(vzL, cfL)
                lmR, lpR = rmhd_eig_speeds(vzR, cfR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                FzB = riemann_flux_rmhd_z(UL, UR, FL, FR, sL, sR, pL, pR, vzL, vzR)

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4], qL2[5], qL2[6], qL2[7], qL2[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4], qR2[5], qR2[6], qR2[7], qR2[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                cfL2 = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR2 = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL2, lpL2 = rmhd_eig_speeds(vzL, cfL2)
                lmR2, lpR2 = rmhd_eig_speeds(vzR, cfR2)
                sL2 = min(lmL2, lmR2); sR2 = max(lpL2, lpR2)
                FzF = riemann_flux_rmhd_z(UL, UR, FL, FR, sL2, sR2, pL, pR, vzL, vzR)

                rhs[0:9, i, j, k] -= (FzF - FzB) / dz

                rhs[8, i, j, k] -= GLM_CP*GLM_CP * pr[8, i, j, k]
                if RESISTIVE_ENABLED and RESISTIVITY > 0.0:
                    invdx2 = 1.0 / (dx*dx + SMALL)
                    invdy2 = 1.0 / (dy*dy + SMALL)
                    invdz2 = 1.0 / (dz*dz + SMALL)
                    for comp in range(5, 8):
                        lap = (
                            (pr[comp, i+1, j, k] - 2.0*pr[comp, i, j, k] + pr[comp, i-1, j, k]) * invdx2 +
                            (pr[comp, i, j+1, k] - 2.0*pr[comp, i, j, k] + pr[comp, i, j-1, k]) * invdy2 +
                            (pr[comp, i, j, k+1] - 2.0*pr[comp, i, j, k] + pr[comp, i, j, k-1]) * invdz2
                        )
                        rhs[comp, i, j, k] += RESISTIVITY * lap
                if RESISTIVE_ENABLED and RESISTIVITY > 0.0:
                    invdx2 = 1.0 / (dx*dx + SMALL)
                    invdy2 = 1.0 / (dy*dy + SMALL)
                    invdz2 = 1.0 / (dz*dz + SMALL)
                    for comp in range(5, 8):
                        lap = (
                            (pr[comp, i+1, j, k] - 2.0*pr[comp, i, j, k] + pr[comp, i-1, j, k]) * invdx2 +
                            (pr[comp, i, j+1, k] - 2.0*pr[comp, i, j, k] + pr[comp, i, j-1, k]) * invdy2 +
                            (pr[comp, i, j, k+1] - 2.0*pr[comp, i, j, k] + pr[comp, i, j, k-1]) * invdz2
                        )
                        rhs[comp, i, j, k] += RESISTIVITY * lap

    return rhs

@nb.njit(parallel=True, fastmath=True)
def compute_rhs_weno(pr, nx, ny, nz, dx, dy, dz):
    rhs = np.zeros((9 + N_PASSIVE, nx, ny, nz))
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
                cfL2 = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR2 = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL2, lpL2 = rmhd_eig_speeds(vxL, cfL2)
                lmR2, lpR2 = rmhd_eig_speeds(vxR, cfR2)
                sL2 = min(lmL2, lmR2); sR2 = max(lpL2, lpR2)
                FxR = riemann_flux_rmhd_x(UL, UR, FL, FR, sL2, sR2, pL, pR, vxL, vxR)

                rhs[0:9, i, j, k] -= (FxR - FxL) / dx

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
                cfL = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL, lpL = rmhd_eig_speeds(vyL, cfL)
                lmR, lpR = rmhd_eig_speeds(vyR, cfR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                FyD = riemann_flux_rmhd_y(UL, UR, FL, FR, sL, sR, pL, pR, vyL, vyR)

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4], qL2[5], qL2[6], qL2[7], qL2[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4], qR2[5], qR2[6], qR2[7], qR2[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_y(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_y(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                cfL2 = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR2 = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL2, lpL2 = rmhd_eig_speeds(vyL, cfL2)
                lmR2, lpR2 = rmhd_eig_speeds(vyR, cfR2)
                sL2 = min(lmL2, lmR2); sR2 = max(lpL2, lpR2)
                FyU = riemann_flux_rmhd_y(UL, UR, FL, FR, sL2, sR2, pL, pR, vyL, vyR)

                rhs[0:9, i, j, k] -= (FyU - FyD) / dy

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
                cfL = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL, lpL = rmhd_eig_speeds(vzL, cfL)
                lmR, lpR = rmhd_eig_speeds(vzR, cfR)
                sL = min(lmL, lmR); sR = max(lpL, lpR)
                FzB = riemann_flux_rmhd_z(UL, UR, FL, FR, sL, sR, pL, pR, vzL, vzR)

                rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL = floor_prim_rmhd(qL2[0], qL2[1], qL2[2], qL2[3], qL2[4], qL2[5], qL2[6], qL2[7], qL2[8])
                rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR = floor_prim_rmhd(qR2[0], qR2[1], qR2[2], qR2[3], qR2[4], qR2[5], qR2[6], qR2[7], qR2[8])
                UL = np.array(prim_to_cons_rmhd(rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL,GAMMA))
                UR = np.array(prim_to_cons_rmhd(rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR,GAMMA))
                FL = flux_rmhd_z(np.array([rL,vxL,vyL,vzL,pL,BxL,ByL,BzL,psiL]))
                FR = flux_rmhd_z(np.array([rR,vxR,vyR,vzR,pR,BxR,ByR,BzR,psiR]))
                cfL2 = rmhd_fast_speed(rL, pL, vxL, vyL, vzL, BxL, ByL, BzL)
                cfR2 = rmhd_fast_speed(rR, pR, vxR, vyR, vzR, BxR, ByR, BzR)
                lmL2, lpL2 = rmhd_eig_speeds(vzL, cfL2)
                lmR2, lpR2 = rmhd_eig_speeds(vzR, cfR2)
                sL2 = min(lmL2, lmR2); sR2 = max(lpL2, lpR2)
                FzF = riemann_flux_rmhd_z(UL, UR, FL, FR, sL2, sR2, pL, pR, vzL, vzR)

                rhs[0:9, i, j, k] -= (FzF - FzB) / dz

                rhs[8, i, j, k] -= GLM_CP*GLM_CP * pr[8, i, j, k]

    return rhs

def step_ssprk2(pr, dx, dy, dz, dt):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    npass = N_PASSIVE
    U0 = np.zeros((9, nx, ny, nz))
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
    U1   = U0 + dt*rhs1[0:9]

    pr1 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr1[0:9,i,j,k] = cons_to_prim_rmhd(U1[0,i,j,k], U1[1,i,j,k], U1[2,i,j,k],
                                                  U1[3,i,j,k], U1[4,i,j,k], U1[5,i,j,k],
                                                  U1[6,i,j,k], U1[7,i,j,k], U1[8,i,j,k])
    if npass > 0:
        pr1[PASSIVE_OFFSET:PASSIVE_OFFSET+npass] = pr[PASSIVE_OFFSET:PASSIVE_OFFSET+npass] + dt*rhs1[9:9+npass]

    if RECON_ID == 1:
        rhs2 = compute_rhs_ppm(pr1, nx, ny, nz, dx, dy, dz)
    elif RECON_ID == 2:
        rhs2 = compute_rhs_weno(pr1, nx, ny, nz, dx, dy, dz)
    else:
        rhs2 = compute_rhs_rmhd(pr1, nx, ny, nz, dx, dy, dz)
    U2   = 0.5*(U0 + U1 + dt*rhs2[0:9])

    out  = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[0:9,i,j,k] = cons_to_prim_rmhd(U2[0,i,j,k], U2[1,i,j,k], U2[2,i,j,k],
                                                   U2[3,i,j,k], U2[4,i,j,k], U2[5,i,j,k],
                                                   U2[6,i,j,k], U2[7,i,j,k], U2[8,i,j,k])
    if npass > 0:
        t0 = pr[PASSIVE_OFFSET:PASSIVE_OFFSET+npass]
        t1 = pr1[PASSIVE_OFFSET:PASSIVE_OFFSET+npass]
        out[PASSIVE_OFFSET:PASSIVE_OFFSET+npass] = 0.5*(t0 + t1 + dt*rhs2[9:9+npass])
    return out

def step_ssprk3(pr, dx, dy, dz, dt):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    npass = N_PASSIVE
    U0 = np.zeros((9, nx, ny, nz))
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
    U1 = U0 + dt*rhs1[0:9]

    pr1 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr1[0:9,i,j,k] = cons_to_prim_rmhd(U1[0,i,j,k], U1[1,i,j,k], U1[2,i,j,k],
                                                  U1[3,i,j,k], U1[4,i,j,k], U1[5,i,j,k],
                                                  U1[6,i,j,k], U1[7,i,j,k], U1[8,i,j,k])
    if npass > 0:
        pr1[PASSIVE_OFFSET:PASSIVE_OFFSET+npass] = pr[PASSIVE_OFFSET:PASSIVE_OFFSET+npass] + dt*rhs1[9:9+npass]

    if RECON_ID == 1:
        rhs2 = compute_rhs_ppm(pr1, nx, ny, nz, dx, dy, dz)
    elif RECON_ID == 2:
        rhs2 = compute_rhs_weno(pr1, nx, ny, nz, dx, dy, dz)
    else:
        rhs2 = compute_rhs_rmhd(pr1, nx, ny, nz, dx, dy, dz)
    U2 = 0.75*U0 + 0.25*(U1 + dt*rhs2[0:9])

    pr2 = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pr2[0:9,i,j,k] = cons_to_prim_rmhd(U2[0,i,j,k], U2[1,i,j,k], U2[2,i,j,k],
                                                  U2[3,i,j,k], U2[4,i,j,k], U2[5,i,j,k],
                                                  U2[6,i,j,k], U2[7,i,j,k], U2[8,i,j,k])
    if npass > 0:
        t0 = pr[PASSIVE_OFFSET:PASSIVE_OFFSET+npass]
        t1 = pr1[PASSIVE_OFFSET:PASSIVE_OFFSET+npass]
        pr2[PASSIVE_OFFSET:PASSIVE_OFFSET+npass] = 0.75*t0 + 0.25*(t1 + dt*rhs2[9:9+npass])

    if RECON_ID == 1:
        rhs3 = compute_rhs_ppm(pr2, nx, ny, nz, dx, dy, dz)
    elif RECON_ID == 2:
        rhs3 = compute_rhs_weno(pr2, nx, ny, nz, dx, dy, dz)
    else:
        rhs3 = compute_rhs_rmhd(pr2, nx, ny, nz, dx, dy, dz)
    U3 = (1.0/3.0)*U0 + (2.0/3.0)*(U2 + dt*rhs3[0:9])

    out = np.zeros_like(pr)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[0:9,i,j,k] = cons_to_prim_rmhd(U3[0,i,j,k], U3[1,i,j,k], U3[2,i,j,k],
                                                  U3[3,i,j,k], U3[4,i,j,k], U3[5,i,j,k],
                                                  U3[6,i,j,k], U3[7,i,j,k], U3[8,i,j,k])
    if npass > 0:
        t0 = pr[PASSIVE_OFFSET:PASSIVE_OFFSET+npass]
        t2 = pr2[PASSIVE_OFFSET:PASSIVE_OFFSET+npass]
        out[PASSIVE_OFFSET:PASSIVE_OFFSET+npass] = (1.0/3.0)*t0 + (2.0/3.0)*(t2 + dt*rhs3[9:9+npass])
    return out
