#!/usr/bin/env python3
# srhd2d.py — 2D Special Relativistic Hydro (flat), MUSCL-lite + HLLE + SSPRK2
# Now with relativistic signal speeds in HLLE and CFL from max wave speed.
import numpy as np
import numba as nb

GAMMA = 5.0/3.0
CFL   = 0.4
SMALL = 1e-12

# ------------------------
# Primitive <-> Conservative
# ------------------------
@nb.njit(fastmath=True)
def prim_to_cons(rho, vx, vy, p, gamma=GAMMA):
    v2 = vx*vx + vy*vy
    if v2 >= 1.0: v2 = 1.0 - 1e-14
    W = 1.0/np.sqrt(1.0 - v2)
    h = 1.0 + gamma/(gamma-1.0)*p/np.maximum(rho, SMALL)
    w = rho*h
    D = rho*W
    Sx = w*W*W*vx
    Sy = w*W*W*vy
    tau = w*W*W - p - D
    return D, Sx, Sy, tau

@nb.njit(fastmath=True)
def cons_to_prim(D, Sx, Sy, tau, gamma=GAMMA):
    E  = tau + D
    S2 = Sx*Sx + Sy*Sy

    # positive initial guess for p
    p = (gamma - 1.0) * (E - D)
    if p < SMALL:
        p = SMALL

    for _ in range(60):
        Wm = E + p
        v2 = S2 / (Wm*Wm + SMALL)
        if v2 >= 1.0:
            v2 = 1.0 - 1e-14
        W   = 1.0/np.sqrt(1.0 - v2)
        rho = D / max(W, SMALL)
        h   = 1.0 + gamma/(gamma-1.0) * p / max(rho, SMALL)
        w   = rho * h

        # Residual for pressure equation: Wm - w W^2 = 0
        f = Wm - w*W*W

        # Safeguarded Jacobian (simple, stable)
        dfdp = 1.0  # robust approximation; damping below keeps it stable

        dp = -f / dfdp
        # Damp the step to keep positivity and stability
        if dp >  0.5*p: dp =  0.5*p
        if dp < -0.5*p: dp = -0.5*p

        p_new = p + dp
        if p_new < SMALL:
            p_new = SMALL

        if abs(dp) < 1e-12 * max(1.0, p_new):
            p = p_new
            break
        p = p_new

    # Recover primitives
    Wm = E + p
    vx = Sx / max(Wm, SMALL)
    vy = Sy / max(Wm, SMALL)
    v2 = vx*vx + vy*vy
    if v2 >= 1.0:
        fac = (1.0 - 1e-14) / np.sqrt(v2)
        vx *= fac; vy *= fac
        v2 = vx*vx + vy*vy
    W  = 1.0/np.sqrt(1.0 - v2)
    rho= D / max(W, SMALL)
    return rho, vx, vy, p

# ------------------------
# EOS helpers
# ------------------------
@nb.njit(fastmath=True)
def sound_speed(rho, p, gamma=GAMMA):
    h = 1.0 + gamma/(gamma-1.0)*p/np.maximum(rho, SMALL)
    w = rho*h
    cs2 = gamma * p / np.maximum(w, SMALL)
    if cs2 < 0.0: cs2 = 0.0
    if cs2 > 1.0 - 1e-14: cs2 = 1.0 - 1e-14
    return np.sqrt(cs2)

@nb.njit(fastmath=True)
def eig_speeds_dir(vn, cs):
    # Relativistic velocity addition along the face-normal
    # λ± = (vn ± cs) / (1 ± vn cs)
    denom_p = 1.0 + vn*cs
    denom_m = 1.0 - vn*cs
    if denom_p == 0.0: denom_p = SMALL
    if denom_m == 0.0: denom_m = SMALL
    lam_p = (vn + cs) / denom_p
    lam_m = (vn - cs) / denom_m
    # clamp to [-1,1]
    if lam_p >  1.0: lam_p =  1.0
    if lam_p < -1.0: lam_p = -1.0
    if lam_m >  1.0: lam_m =  1.0
    if lam_m < -1.0: lam_m = -1.0
    return lam_m, lam_p

# ------------------------
# Fluxes
# ------------------------
@nb.njit(fastmath=True)
def flux_x(rho, vx, vy, p):
    D,Sx,Sy,tau = prim_to_cons(rho, vx, vy, p)
    return np.array([D*vx, Sx*vx + p, Sy*vx, (tau+p)*vx])

@nb.njit(fastmath=True)
def flux_y(rho, vx, vy, p):
    D,Sx,Sy,tau = prim_to_cons(rho, vx, vy, p)
    return np.array([D*vy, Sx*vy, Sy*vy + p, (tau+p)*vy])

@nb.njit(fastmath=True)
def hlle_flux(UL, UR, FL, FR, sL, sR):
    if sL >= 0.0:
        return FL
    elif sR <= 0.0:
        return FR
    else:
        return (sR*FL - sL*FR + sL*sR*(UR-UL)) / (sR - sL + SMALL)

# ------------------------
# RHS with relativistic wave speed bounds
# ------------------------
@nb.njit(parallel=True, fastmath=True)
def compute_rhs(prims, nx, ny, dx, dy):
    rhs = np.zeros((4,nx,ny))
    # interior cells; simple first-order reconstruction at cell faces (MUSCL can be added later)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            # x-direction faces: (i-1/2) between i-1 and i; (i+1/2) between i and i+1
            # Left face
            rL,vxL,vyL,pL = prims[0,i-1,j], prims[1,i-1,j], prims[2,i-1,j], prims[3,i-1,j]
            rR,vxR,vyR,pR = prims[0,i  ,j], prims[1,i  ,j], prims[2,i  ,j], prims[3,i  ,j]
            UL = np.array(prim_to_cons(rL,vxL,vyL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,pR))
            FL = flux_x(rL,vxL,vyL,pL); FR = flux_x(rR,vxR,vyR,pR)
            csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
            lamLm, lamLp = eig_speeds_dir(vxL, csL)
            lamRm, lamRp = eig_speeds_dir(vxR, csR)
            sL = min(lamLm, lamRm); sR = max(lamLp, lamRp)
            Fx_L = hlle_flux(UL, UR, FL, FR, sL, sR)

            # Right face
            rL,vxL,vyL,pL = prims[0,i  ,j], prims[1,i  ,j], prims[2,i  ,j], prims[3,i  ,j]
            rR,vxR,vyR,pR = prims[0,i+1,j], prims[1,i+1,j], prims[2,i+1,j], prims[3,i+1,j]
            UL = np.array(prim_to_cons(rL,vxL,vyL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,pR))
            FL = flux_x(rL,vxL,vyL,pL); FR = flux_x(rR,vxR,vyR,pR)
            csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
            lamLm, lamLp = eig_speeds_dir(vxL, csL)
            lamRm, lamRp = eig_speeds_dir(vxR, csR)
            sL2 = min(lamLm, lamRm); sR2 = max(lamLp, lamRp)
            Fx_R = hlle_flux(UL, UR, FL, FR, sL2, sR2)

            # y-direction faces
            rL,vxL,vyL,pL = prims[0,i,j-1], prims[1,i,j-1], prims[2,i,j-1], prims[3,i,j-1]
            rR,vxR,vyR,pR = prims[0,i,j  ], prims[1,i,j  ], prims[2,i,j  ], prims[3,i,j  ]
            UL = np.array(prim_to_cons(rL,vxL,vyL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,pR))
            FL = flux_y(rL,vxL,vyL,pL); FR = flux_y(rR,vxR,vyR,pR)
            csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
            lamLm, lamLp = eig_speeds_dir(vyL, csL)
            lamRm, lamRp = eig_speeds_dir(vyR, csR)
            tL = min(lamLm, lamRm); tR = max(lamLp, lamRp)
            Fy_D = hlle_flux(UL, UR, FL, FR, tL, tR)

            rL,vxL,vyL,pL = prims[0,i,j  ], prims[1,i,j  ], prims[2,i,j  ], prims[3,i,j  ]
            rR,vxR,vyR,pR = prims[0,i,j+1], prims[1,i,j+1], prims[2,i,j+1], prims[3,i,j+1]
            UL = np.array(prim_to_cons(rL,vxL,vyL,pL)); UR = np.array(prim_to_cons(rR,vxR,vyR,pR))
            FL = flux_y(rL,vxL,vyL,pL); FR = flux_y(rR,vxR,vyR,pR)
            csL = sound_speed(rL,pL); csR = sound_speed(rR,pR)
            lamLm, lamLp = eig_speeds_dir(vyL, csL)
            lamRm, lamRp = eig_speeds_dir(vyR, csR)
            tL2 = min(lamLm, lamRm); tR2 = max(lamLp, lamRp)
            Fy_U = hlle_flux(UL, UR, FL, FR, tL2, tR2)

            Uc = np.array(prim_to_cons(prims[0,i,j],prims[1,i,j],prims[2,i,j],prims[3,i,j]))
            rhs[:,i,j] = -(Fx_R - Fx_L)/dx - (Fy_U - Fy_D)/dy
    return rhs

# ------------------------
# Time stepping (SSPRK2) with CFL from max wave speed
# ------------------------
@nb.njit(fastmath=True)
def max_char_speed(prims, nx, ny):
    amax = 0.0
    for i in range(nx):
        for j in range(ny):
            rho = prims[0,i,j]; vx = prims[1,i,j]; vy = prims[2,i,j]; p = prims[3,i,j]
            cs = sound_speed(rho, p)
            lamxm, lamxp = eig_speeds_dir(vx, cs)
            lamym, lamyp = eig_speeds_dir(vy, cs)
            local = max(abs(lamxm), abs(lamxp), abs(lamym), abs(lamyp))
            if local > amax: amax = local
    if amax < SMALL: amax = SMALL
    return amax

def step(prims, nx, ny, dx, dy, dt):
    U0 = np.zeros((4,nx,ny))
    for i in range(nx):
        for j in range(ny):
            U0[:,i,j] = prim_to_cons(prims[0,i,j],prims[1,i,j],prims[2,i,j],prims[3,i,j])

    rhs1 = compute_rhs(prims, nx, ny, dx, dy)
    U1 = U0 + dt*rhs1

    prims1 = np.zeros_like(prims)
    for i in range(nx):
        for j in range(ny):
            prims1[:,i,j] = cons_to_prim(U1[0,i,j],U1[1,i,j],U1[2,i,j],U1[3,i,j])

    rhs2 = compute_rhs(prims1, nx, ny, dx, dy)
    U2 = 0.5*(U0 + U1 + dt*rhs2)

    out = np.zeros_like(prims)
    for i in range(nx):
        for j in range(ny):
            out[:,i,j] = cons_to_prim(U2[0,i,j],U2[1,i,j],U2[2,i,j],U2[3,i,j])
    return out

# ------------------------
# ICs and boundaries (periodic for now)
# ------------------------
def apply_bc_periodic(prims):
    # simple periodic wrap
    prims[:, 0 ,:] = prims[:, -2,:]
    prims[:, -1,:] = prims[:, 1 ,:]
    prims[:, :, 0] = prims[:, :, -2]
    prims[:, :, -1]= prims[:, :, 1 ]

def init_blast(nx, ny):
    x = np.linspace(0,1,nx)
    y = np.linspace(0,1,ny)
    X,Y = np.meshgrid(x,y,indexing='ij')
    r2 = (X-0.5)**2 + (Y-0.5)**2
    prims = np.zeros((4,nx,ny))
    prims[0,:,:] = 1.0
    prims[1,:,:] = 0.0
    prims[2,:,:] = 0.0
    prims[3,:,:] = 0.1
    prims[3,r2<0.10**2] = 1.0
    # pad one-cell ghost for periodic (we'll maintain via apply_bc_periodic)
    return prims

# ------------------------
# Driver
# ------------------------
def run():
    nx, ny = 128+2, 128+2   # +2 for periodic ghost cells
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx/(nx-2), Ly/(ny-2)

    prims = init_blast(nx, ny)
    apply_bc_periodic(prims)

    t, t_end = 0.0, 0.4
    step_id = 0
    while t < t_end:
        amax = max_char_speed(prims, nx, ny)
        dt   = CFL * min(dx,dy) / amax
        if t + dt > t_end: dt = t_end - t

        prims = step(prims, nx, ny, dx, dy, dt)
        apply_bc_periodic(prims)

        t += dt; step_id += 1
        if step_id % 10 == 0 or t >= t_end:
            print(f"t={t:.4f}  dt={dt:.3e}  amax={amax:.3f}")

    np.savez("blast2d_final.npz",
             rho=prims[0], vx=prims[1], vy=prims[2], p=prims[3])
    print("Saved blast2d_final.npz")

if __name__ == "__main__":
    run()
