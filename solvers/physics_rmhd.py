# solvers/physics_rmhd.py
import numpy as np
SMALL = 1e-12

# ---- primitives & conserved ----
# prim_rmhd: [rho, vx, vy, vz, p, Bx, By, Bz, psi]   (psi used for GLM)
# cons_rmhd: [D, Sx, Sy, Sz, tau, Bx, By, Bz, psi]   (Bx.. are area-averaged in CT, but here cell-centered + GLM)

def prim_to_cons_rmhd(rho, vx, vy, vz, p, Bx, By, Bz, psi, gamma, V_MAX, P_MAX):
    # cap inputs
    v2 = vx*vx + vy*vy + vz*vz
    vmax2 = V_MAX*V_MAX
    if v2 >= vmax2:
        fac = V_MAX/np.sqrt(v2 + 1e-32); vx*=fac; vy*=fac; vz*=fac
    if p < SMALL: p = SMALL
    if p > P_MAX: p = P_MAX

    # magnetic pressure & energy in lab frame
    B2 = Bx*Bx + By*By + Bz*Bz
    W  = 1.0/np.sqrt(1.0 - (vx*vx + vy*vy + vz*vz))
    h  = 1.0 + gamma/(gamma-1.0)*p/max(rho, SMALL)
    rhoh = rho*h

    # RMHD “Valencia-like” cons (special relativistic flat space)
    # total pressure: pt = p + 0.5*B^2
    pt = p + 0.5*B2
    # b^0, b^i (comoving) simplification for flat SR:
    vb = vx*Bx + vy*By + vz*Bz
    b0 = W*vb
    bx = (Bx + b0*vx)/W
    by = (By + b0*vy)/W
    bz = (Bz + b0*vz)/W
    b2 = (bx*bx + by*by + bz*bz) - b0*b0

    # effective enthalpy density
    wtot = rhoh*W*W + B2

    D   = rho*W
    Sx  = wtot*vx - (Bx*vb)
    Sy  = wtot*vy - (By*vb)
    Sz  = wtot*vz - (Bz*vb)
    tau = rhoh*W*W - p + 0.5*B2 + 0.5*b2 - D

    return D, Sx, Sy, Sz, tau, Bx, By, Bz, psi

def cons_to_prim_rmhd(U, gamma, V_MAX, P_MAX):
    # robust placeholder: recover hydro primitives, carry B, psi across.
    # (we’ll replace with full RMHD recovery in the next patch.)
    D,Sx,Sy,Sz,tau,Bx,By,Bz,psi = U
    # use hydro cons->prim on (D,S,tau), then attach B,psi
    # you will pass the hydro cons_to_prim into this module from the solver.
    return None  # to be wired by solver: call hydro recovery, then append Bx,By,Bz,psi

# ---- fluxes (HLLE scaffold) ----
def flux_rmhd_x(prim, gamma):
    rho,vx,vy,vz,p,Bx,By,Bz,psi = prim
    # total pressure
    B2 = Bx*Bx + By*By + Bz*Bz
    pt = p + 0.5*B2
    # basic flux skeleton (momentum-energy + induction + GLM psi coupling)
    # Return 9-component flux consistent with cons vector
    # For the first compiling pass, fill safe approximations; we’ll refine next patch.
    D,Sx,Sy,Sz,tau,_,_,_,_ = prim_to_cons_rmhd(rho,vx,vy,vz,p,Bx,By,Bz,psi,gamma,0.9999,1e9)
    F = np.zeros(9)
    F[0] = D*vx
    F[1] = Sx*vx + pt - Bx*Bx
    F[2] = Sy*vx - Bx*By
    F[3] = Sz*vx - Bx*Bz
    F[4] = (tau + pt)*vx - (Bx*(Bx*vx + By*vy + Bz*vz))
    # induction + GLM (Dedner): ∂t B + ∇(vx B - Bx v) + ∇ψ = 0 in x-flux
    F[5] = psi                          # Bx flux includes ψ (for GLM hyperbolic)
    F[6] = vy*Bx - vx*By               # By flux
    F[7] = vz*Bx - vx*Bz               # Bz flux
    # ψ equation: ∂t ψ + c_h^2 ∇·B = -c_p^2 ψ  → in x-flux: c_h^2 * Bx
    F[8] = 0.0  # ψ’s x-flux will be set as c_h^2 * Bx in the solver with GLM_CH
    return F

# y,z fluxes are analogous
def flux_rmhd_y(prim, gamma):
    rho,vx,vy,vz,p,Bx,By,Bz,psi = prim
    D,Sx,Sy,Sz,tau,_,_,_,_ = prim_to_cons_rmhd(rho,vx,vy,vz,p,Bx,By,Bz,psi,gamma,0.9999,1e9)
    F = np.zeros(9)
    F[0] = D*vy
    B2 = Bx*Bx + By*By + Bz*Bz
    pt = p + 0.5*B2
    F[1] = Sx*vy - By*Bx
    F[2] = Sy*vy + pt - By*By
    F[3] = Sz*vy - By*Bz
    F[4] = (tau + pt)*vy - (By*(Bx*vx + By*vy + Bz*vz))
    F[5] = vx*By - vy*Bx
    F[6] = psi
    F[7] = vz*By - vy*Bz
    F[8] = 0.0
    return F

def flux_rmhd_z(prim, gamma):
    rho,vx,vy,vz,p,Bx,By,Bz,psi = prim
    D,Sx,Sy,Sz,tau,_,_,_,_ = prim_to_cons_rmhd(rho,vx,vy,vz,p,Bx,By,Bz,psi,gamma,0.9999,1e9)
    F = np.zeros(9)
    F[0] = D*vz
    B2 = Bx*Bx + By*By + Bz*Bz
    pt = p + 0.5*B2
    F[1] = Sx*vz - Bz*Bx
    F[2] = Sy*vz - Bz*By
    F[3] = Sz*vz + pt - Bz*Bz
    F[4] = (tau + pt)*vz - (Bz*(Bx*vx + By*vy + Bz*vz))
    F[5] = vx*Bz - vz*Bx
    F[6] = vy*Bz - vz*By
    F[7] = psi
    F[8] = 0.0
    return F

def hlle_bounds_fast(primL, primR, c_cap=1.0):
    # robust bounds for first pass: use ±c_cap (≤ 1.0). We’ll refine later.
    return -c_cap, c_cap
