#!/usr/bin/env python3
# srhd3d_mpi_muscl.py  — 3D SRHD (flat), MUSCL + HLLE + SSPRK2
# MPI x-slab decomposition, blocking Sendrecv halo exchange, jet nozzle inflow
# Includes simple diagnostics: global max Lorentz factor and inlet energy flux
#
# Usage:
#   python3 -m pip install numpy numba mpi4py
#   mpirun -np 2 python3 srhd3d_mpi_muscl.py

import os
import sys
import glob
import numpy as np
from mpi4py import MPI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.settings import load_settings
from utils.io_utils import make_run_dir
from utils import diagnostics
from core import srhd_core
from core import rmhd_core
from core import grhd_core
from core import grmhd_core
from core import dissipation
from core import source_terms
from core import chemistry
from core import plasma_microphysics
from core import gravity
from core import boundary
from core import nozzle

settings = load_settings()
if settings.get("PHYSICS") == "sn":
    # Override gamma for SN-lite EOS without affecting other modes.
    settings["GAMMA"] = float(settings.get("SN_EOS_GAMMA", settings.get("GAMMA", 5.0/3.0)))
srhd_core.configure(settings)
rmhd_core.configure(settings)
grhd_core.configure(settings)
grmhd_core.configure(settings)

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
P_MAX         = settings["P_MAX"]
V_MAX         = settings["V_MAX"]

DEBUG         = settings["DEBUG"]
ASSERTS       = settings["ASSERTS"]
CHECK_NAN_EVERY = settings["CHECK_NAN_EVERY"]

PHYSICS      = settings.get("PHYSICS", "hydro")   # "hydro" | "rmhd" | "sn"
GLM_CH       = settings.get("GLM_CH", 1.0)
GLM_CP       = settings.get("GLM_CP", 0.1)
B_INIT       = settings.get("B_INIT", "none")     # "none" | "poloidal" | "toroidal"
B0           = settings.get("B0", 0.0)
RK_ORDER     = int(settings.get("RK_ORDER", 2))
CHECKPOINT_EVERY = int(settings.get("CHECKPOINT_EVERY", 0))
RESTART_PATH = settings.get("RESTART_PATH")
DISSIPATION_ENABLED = settings.get("DISSIPATION_ENABLED", False)
BULK_ZETA = settings.get("BULK_ZETA", 0.0)
N_TRACERS = int(settings.get("N_TRACERS", 0))
TRACER_OFFSET = int(settings.get("TRACER_OFFSET", 5))
TRACER_NAMES = settings.get("TRACER_NAMES", [])
TRACER_AMB_VALUES = settings.get("TRACER_AMB_VALUES", [])
N_THERMO = int(settings.get("N_THERMO", 0))
THERMO_OFFSET = int(settings.get("THERMO_OFFSET", TRACER_OFFSET + N_TRACERS))
N_CHEM = int(settings.get("N_CHEM", 0))
CHEM_OFFSET = int(settings.get("CHEM_OFFSET", THERMO_OFFSET + N_THERMO))
CHEM_NAMES = settings.get("CHEM_NAMES", ["xHII", "xHeII", "xHeIII"])
SN_COMP_NAMES = settings.get("SN_COMP_NAMES", [])
SN_COMP_AMB_VALUES = settings.get("SN_COMP_AMB_VALUES", [])
SN_COMP_NOZZLE_VALUES = settings.get("SN_COMP_NOZZLE_VALUES", [])
SN_COMP_OFFSET = int(settings.get("SN_COMP_OFFSET", TRACER_OFFSET))

SMALL = 1e-12
 
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
    pr shape: (NV, nx_loc + 2*NG, NY + 2*NG, NZ + 2*NG)
    Note: NV = 5 for base hydro, NV = 15 for hydro+IS dissipation, NV = 9 for RMHD.
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
# Initialization
# ------------------------
def init_block(nx_loc, ny_loc, nz_loc, x0, dx, dy, dz):
    nv = 9 if PHYSICS in ("rmhd", "grmhd") else 5
    if DISSIPATION_ENABLED and PHYSICS in ("hydro", "grhd"):
        nv = 15
    if PHYSICS == "sn":
        nv = 5
    if N_TRACERS > 0:
        nv += N_TRACERS
    if N_THERMO > 0:
        nv += N_THERMO
    if N_CHEM > 0:
        nv += N_CHEM
    if PHYSICS == "sn" and len(SN_COMP_NAMES) > 0:
        nv += len(SN_COMP_NAMES)
    pr = np.zeros((nv, nx_loc + 2*NG, ny_loc + 2*NG, nz_loc + 2*NG), dtype=np.float64)

    # base hydro fields
    pr[0, :, :, :] = RHO_AMB
    pr[1, :, :, :] = VX_AMB
    pr[2, :, :, :] = VY_AMB
    pr[3, :, :, :] = VZ_AMB
    pr[4, :, :, :] = P_AMB
    if PHYSICS in ("rmhd", "grmhd"):
        pr[5:, :, :, :] = 0.0
    if DISSIPATION_ENABLED and PHYSICS in ("hydro", "grhd"):
        pr[5:, :, :, :] = 0.0
    if N_TRACERS > 0:
        for t in range(N_TRACERS):
            val = TRACER_AMB_VALUES[t] if t < len(TRACER_AMB_VALUES) else 0.0
            pr[TRACER_OFFSET + t, :, :, :] = val
    if N_THERMO > 0:
        te = float(settings.get("TE_AMB", 0.0))
        ti = float(settings.get("TI_AMB", 0.0))
        pr[THERMO_OFFSET, :, :, :] = te
        pr[THERMO_OFFSET + 1, :, :, :] = ti
    if N_CHEM > 0:
        pr[CHEM_OFFSET + 0, :, :, :] = float(settings.get("CHEM_X_HII_AMB", 0.0))
        pr[CHEM_OFFSET + 1, :, :, :] = float(settings.get("CHEM_X_HEII_AMB", 0.0))
        pr[CHEM_OFFSET + 2, :, :, :] = float(settings.get("CHEM_X_HEIII_AMB", 0.0))
    if PHYSICS == "sn" and len(SN_COMP_NAMES) > 0:
        for ci, val in enumerate(SN_COMP_AMB_VALUES):
            pr[SN_COMP_OFFSET + ci, :, :, :] = val

    if PHYSICS == "sn":
        init = str(settings.get("SN_INIT", "uniform")).lower()
        if init in ("sedov", "shock_sphere"):
            center = settings.get("SN_GRAVITY_CENTER") or [0.5*Lx, 0.5*Ly, 0.5*Lz]
            cx, cy, cz = center
            for i in range(pr.shape[1]):
                x = (x0 + (i - NG) + 0.5) * dx
                for j in range(pr.shape[2]):
                    y = (j - NG + 0.5) * dy
                    for k in range(pr.shape[3]):
                        z = (k - NG + 0.5) * dz
                        r = ((x - cx)**2 + (y - cy)**2 + (z - cz)**2) ** 0.5
                        if init == "sedov":
                            r0 = float(settings.get("SN_SEDOV_RADIUS", 0.05))
                            dp = float(settings.get("SN_SEDOV_DP", 1.0))
                            if r <= r0:
                                pr[4, i, j, k] += dp
                        else:
                            r0 = float(settings.get("SN_SPHERE_RADIUS", 0.2))
                            if r <= r0:
                                pr[0, i, j, k] = float(settings.get("SN_RHO_IN", pr[0, i, j, k]))
                                pr[4, i, j, k] = float(settings.get("SN_P_IN", pr[4, i, j, k]))
                            else:
                                pr[0, i, j, k] = float(settings.get("SN_RHO_OUT", pr[0, i, j, k]))
                                pr[4, i, j, k] = float(settings.get("SN_P_OUT", pr[4, i, j, k]))
    return pr

def latest_run_dir(base="results"):
    runs = sorted([d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)])
    if not runs:
        return None
    return runs[-1]

def latest_checkpoint_in_dir(run_dir, rank):
    pat = os.path.join(run_dir, f"checkpoint_rank{rank:04d}_step*.npz")
    files = sorted(glob.glob(pat))
    return files[-1] if files else None

def resolve_restart_path(restart_path, rank):
    if restart_path is None:
        return None
    if isinstance(restart_path, str) and restart_path.lower() in ("none", ""):
        return None
    if isinstance(restart_path, str) and restart_path.lower() == "latest":
        run_dir = latest_run_dir()
        if run_dir is None:
            return None
        return latest_checkpoint_in_dir(run_dir, rank)
    if os.path.isdir(restart_path):
        return latest_checkpoint_in_dir(restart_path, rank)
    if os.path.isfile(restart_path):
        return restart_path
    return None

def load_checkpoint(path):
    data = np.load(path)
    pr = data["prim"]
    meta = data["meta"] if "meta" in data.files else None
    if meta is None or len(meta) < 5:
        return pr, 0.0, 0
    t = float(meta[3])
    step = int(meta[4])
    return pr, t, step

# ------------------------
# Main
# ------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- create output dir (or resume) ---
    restart_path = resolve_restart_path(RESTART_PATH, rank)
    if restart_path:
        RUN_DIR = os.path.dirname(restart_path)
    else:
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
        print("[startup] params:"
              f" P_EQ={P_EQ:.6g}, P_AMB={P_AMB:.6g}, RHO_AMB={RHO_AMB:.6g},"
              f" GAMMA_JET={GAMMA_JET:.6g}, ETA_RHO={ETA_RHO:.6g},"
              f" JET_RADIUS={JET_RADIUS:.3f}, NOZZLE_TURB={NOZZLE_TURB}", flush=True)
        print(f"[startup] physics={PHYSICS}  B_INIT={B_INIT}  B0={B0}", flush=True)

    # domain decomposition in x
    nx_loc, x0, counts, offs = decompose_x(NX, comm)
    left  = rank-1 if rank-1 >= 0     else None
    right = rank+1 if rank+1 < size   else None

    dx, dy, dz = Lx/NX, Ly/NY, Lz/NZ
    ny_loc, nz_loc = NY, NZ

    # allocate primitives
    pr = init_block(nx_loc, ny_loc, nz_loc, x0, dx, dy, dz)
    rng = np.random.default_rng(1234 + rank*777)

    t = 0.0
    step = 0
    if restart_path:
        pr, t, step = load_checkpoint(restart_path)
        if rank == 0:
            print(f"[restart] loaded {restart_path} at t={t:.6e} step={step}", flush=True)

    if not restart_path:
        # initial BCs
        boundary.apply_periodic_yz(pr, NG)
        if rank == size-1: boundary.apply_outflow_right_x(pr, NG)
        if rank == 0 and PHYSICS != "sn":
            nozzle.apply_nozzle_left_x(
                pr, dx, dy, dz, ny_loc, nz_loc, JET_CENTER[1], JET_CENTER[2], rng, settings
            )

    # --- DEBUG: JIT warm-up ---
    if DEBUG:
        if PHYSICS == "hydro":
            _ = srhd_core.compute_rhs_muscl(pr, pr.shape[1], pr.shape[2], pr.shape[3], dx, dy, dz)
        elif PHYSICS == "rmhd":
            _ = rmhd_core.compute_rhs_rmhd(pr, pr.shape[1], pr.shape[2], pr.shape[3], dx, dy, dz)
        elif PHYSICS == "grhd":
            _ = grhd_core.compute_rhs_grhd(pr, dx, dy, dz, offs[rank], NG)
        else:
            _ = grmhd_core.compute_rhs_grmhd(pr, dx, dy, dz, offs[rank], NG)
        comm.Barrier()
        if rank == 0:
            print("[jit] RHS kernel compiled and first call done.", flush=True)

    # --- time loop ---
    while t < T_END:
        # CFL timestep
        if PHYSICS in ("hydro", "grhd", "sn"):
            amax_local = srhd_core.max_char_speed(pr, pr.shape[1], pr.shape[2], pr.shape[3])
        else:
            amax_local = 1.0
        amax = comm.allreduce(amax_local, op=MPI.MAX)
        dt = CFL * min(dx, dy, dz) / max(amax, SMALL)
        if t + dt > T_END:
            dt = T_END - t

        # halo exchange + BCs
        exchange_halos(pr, comm, left, right)
        boundary.apply_periodic_yz(pr, NG)
        if rank == size-1: boundary.apply_outflow_right_x(pr, NG)
        if rank == 0:      nozzle.apply_nozzle_left_x(
            pr, dx, dy, dz, ny_loc, nz_loc, JET_CENTER[1], JET_CENTER[2], rng, settings
        )

        # advance one step
        if PHYSICS in ("hydro", "sn"):
            if RK_ORDER == 3:
                pr = srhd_core.step_ssprk3(pr, dx, dy, dz, dt)
            else:
                pr = srhd_core.step_ssprk2(pr, dx, dy, dz, dt)
        elif PHYSICS == "rmhd":
            if RK_ORDER == 3:
                pr = rmhd_core.step_ssprk3(pr, dx, dy, dz, dt)
            else:
                pr = rmhd_core.step_ssprk2(pr, dx, dy, dz, dt)
        elif PHYSICS == "grhd":
            if RK_ORDER == 3:
                pr = grhd_core.step_ssprk3(pr, dx, dy, dz, dt, offs[rank], NG)
            else:
                pr = grhd_core.step_ssprk2(pr, dx, dy, dz, dt, offs[rank], NG)
        else:
            if RK_ORDER == 3:
                pr = grmhd_core.step_ssprk3(pr, dx, dy, dz, dt, offs[rank], NG)
            else:
                pr = grmhd_core.step_ssprk2(pr, dx, dy, dz, dt, offs[rank], NG)

        # re-apply BCs after update
        boundary.apply_periodic_yz(pr, NG)
        if rank == size-1: boundary.apply_outflow_right_x(pr, NG)
        if rank == 0 and PHYSICS != "sn":
            nozzle.apply_nozzle_left_x(
                pr, dx, dy, dz, ny_loc, nz_loc, JET_CENTER[1], JET_CENTER[2], rng, settings
            )
        pr = dissipation.apply_causal_dissipation(pr, dt, dx, dy, dz, settings)
        pr = source_terms.apply_cooling_heating(pr, dt, settings)
        pr = source_terms.apply_two_temperature(pr, dt, settings)
        pr = chemistry.apply_ion_chemistry(pr, dt, settings)
        pr = gravity.apply_gravity(pr, dt, dx, dy, dz, settings, offs[rank], NG)
        pr = source_terms.apply_sn_heating(pr, dt, dx, dy, dz, settings, offs[rank], NG)
        pr = plasma_microphysics.apply_nonideal_mhd(pr, dt, dx, dy, dz, settings, NG)
        pr = source_terms.apply_radiation_coupling(pr, dt, settings)

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
            tracer_fields = {}
            if N_TRACERS > 0:
                for ti in range(N_TRACERS):
                    name = TRACER_NAMES[ti] if ti < len(TRACER_NAMES) else f"tracer{ti}"
                    tracer_fields[name] = pr[TRACER_OFFSET + ti]
                tracer_fields["tracer_names"] = np.array(TRACER_NAMES, dtype=object)
            if N_THERMO > 0:
                tracer_fields["Te"] = pr[THERMO_OFFSET]
                tracer_fields["Ti"] = pr[THERMO_OFFSET + 1]
            if N_CHEM > 0:
                for ci in range(N_CHEM):
                    name = CHEM_NAMES[ci] if ci < len(CHEM_NAMES) else f"chem{ci}"
                    tracer_fields[name] = pr[CHEM_OFFSET + ci]
            if PHYSICS in ("hydro", "grhd", "sn"):
                if TRACER_OFFSET > 5:
                    np.savez(
                        fname,
                        rho = pr[0], vx=pr[1], vy=pr[2], vz=pr[3], p=pr[4],
                        pi=pr[5], pixx=pr[6], piyy=pr[7], pizz=pr[8],
                        pixy=pr[9], pixz=pr[10], piyz=pr[11],
                        qx=pr[12], qy=pr[13], qz=pr[14],
                        **tracer_fields,
                        meta = np.array([dx, dy, dz, t], dtype=np.float64),
                        comment = "3D SRHD/GRHD block (with ghosts)."
                    )
                else:
                    np.savez(
                        fname,
                        rho = pr[0], vx=pr[1], vy=pr[2], vz=pr[3], p=pr[4],
                        **tracer_fields,
                        meta = np.array([dx, dy, dz, t], dtype=np.float64),
                        comment = "3D SRHD/GRHD block (with ghosts)."
                    )
            else:
                np.savez(
                    fname,
                    rho = pr[0], vx=pr[1], vy=pr[2], vz=pr[3], p=pr[4],
                    Bx = pr[5], By=pr[6], Bz=pr[7], psi=pr[8],
                    **tracer_fields,
                    meta = np.array([dx, dy, dz, t], dtype=np.float64),
                    comment = "3D RMHD/GRMHD block (with ghosts)."
                )
            if rank == 0:
                print(f"[io] wrote {fname}", flush=True)

            if PHYSICS in ("hydro", "grhd", "sn"):
                global_maxG, flux_signed, flux_abs = diagnostics.compute_diagnostics_and_write(
                    pr, dx, dy, dz, offs[rank], counts, comm, rank,
                    step, t, dt, amax, RUN_DIR,
                    srhd_core.prim_to_cons, NG, False
                )
            else:
                global_maxG, flux_signed, flux_abs = diagnostics.compute_diagnostics_and_write(
                    pr, dx, dy, dz, offs[rank], counts, comm, rank,
                    step, t, dt, amax, RUN_DIR,
                    rmhd_core.prim_to_cons_rmhd, NG, True
                )
            if rank == 0:
                print(f"[diag] step={step} maxGamma={global_maxG:.3f} "
                      f"inletFlux_signed={flux_signed:.3e} inletFlux_abs={flux_abs:.3e}",
                      flush=True)

            # centerline diagnostics (write only; no return so no need to assign to any object)
            diagnostics.compute_centerline_and_write(
                pr, dx, dy, dz, offs[rank], counts, comm, rank,
                step, t, RUN_DIR, V_MAX, NG
            )
            if PHYSICS in ("rmhd", "grmhd"):
                diagnostics.compute_divb_and_write(
                    pr, dx, dy, dz, offs[rank], counts, comm, rank,
                    step, t, RUN_DIR, NG
                )

        if CHECKPOINT_EVERY > 0 and (step % CHECKPOINT_EVERY == 0):
            ckpt = os.path.join(RUN_DIR, f"checkpoint_rank{rank:04d}_step{step:06d}.npz")
            np.savez(
                ckpt,
                prim = pr,
                meta = np.array([dx, dy, dz, t, step], dtype=np.float64),
                comment = "Checkpoint (with ghosts)."
            )
            if rank == 0:
                print(f"[ckpt] wrote {ckpt}", flush=True)
    if rank == 0:
        print("Done.", flush=True)

if __name__ == "__main__":
    main()
