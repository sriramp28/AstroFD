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
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.settings import load_settings

settings = load_settings()
if settings.get("OMP_NUM_THREADS") is not None:
    nthreads = int(settings["OMP_NUM_THREADS"])
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
if settings.get("NUMBA_NUM_THREADS") is not None:
    try:
        import numba as nb
        nb.set_num_threads(int(settings["NUMBA_NUM_THREADS"]))
    except Exception:
        pass

import numpy as np
import numba as nb
from mpi4py import MPI
from utils.io_utils import make_run_dir, write_run_config
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
from core import adaptivity
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
RESTART_STRICT = bool(settings.get("RESTART_STRICT", False))
PERF_ENABLED = bool(settings.get("PERF_ENABLED", False))
PERF_EVERY = int(settings.get("PERF_EVERY", 10))
PERF_RESET_EVERY = int(settings.get("PERF_RESET_EVERY", 0))
DISSIPATION_ENABLED = settings.get("DISSIPATION_ENABLED", False)
BULK_ZETA = settings.get("BULK_ZETA", 0.0)
ADAPTIVITY_ENABLED = bool(settings.get("ADAPTIVITY_ENABLED", False))
ADAPTIVITY_MODE = str(settings.get("ADAPTIVITY_MODE", "nested_static")).lower()
ADAPTIVITY_SUBCYCLES = settings.get("ADAPTIVITY_SUBCYCLES", None)
HALO_EXCHANGE = str(settings.get("HALO_EXCHANGE", "blocking")).lower()
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
def exchange_halos(pr, comm, left, right, mode="blocking"):
    """
    Exchange NG ghost layers along x with neighbors using blocking Sendrecv.
    pr shape: (NV, nx_loc + 2*NG, NY + 2*NG, NZ + 2*NG)
    Note: NV = 5 for base hydro, NV = 15 for hydro+IS dissipation, NV = 9 for RMHD.
    """
    if mode == "nonblocking":
        reqs = []
        if left is not None:
            sendL = np.ascontiguousarray(pr[:, NG:2*NG, :, :])
            recvL = np.empty_like(sendL)
            reqs.append(comm.Irecv(recvL, source=left, tag=20))
            reqs.append(comm.Isend(sendL, dest=left, tag=21))
        if right is not None:
            sendR = np.ascontiguousarray(pr[:, -2*NG:-NG, :, :])
            recvR = np.empty_like(sendR)
            reqs.append(comm.Irecv(recvR, source=right, tag=21))
            reqs.append(comm.Isend(sendR, dest=right, tag=20))
        MPI.Request.Waitall(reqs)
        if left is not None:
            pr[:, 0:NG, :, :] = recvL
        if right is not None:
            pr[:, -NG:, :, :] = recvR
        return
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

def compute_monopole_mass(pr, dx, dy, dz, offs_x, ng, cfg, comm):
    nbins = int(cfg.get("SN_GRAVITY_BINS", 64))
    center = cfg.get("SN_GRAVITY_CENTER") or [0.5*Lx, 0.5*Ly, 0.5*Lz]
    x0, y0, z0 = float(center[0]), float(center[1]), float(center[2])
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    # max radius to domain corners
    corners = [
        (0.0, 0.0, 0.0),
        (Lx, 0.0, 0.0),
        (0.0, Ly, 0.0),
        (0.0, 0.0, Lz),
        (Lx, Ly, 0.0),
        (Lx, 0.0, Lz),
        (0.0, Ly, Lz),
        (Lx, Ly, Lz),
    ]
    rmax = 0.0
    for cx, cy, cz in corners:
        r = ((cx - x0)**2 + (cy - y0)**2 + (cz - z0)**2) ** 0.5
        if r > rmax:
            rmax = r
    if rmax <= 0.0:
        rmax = 1.0
    r_edges = np.linspace(0.0, rmax, nbins + 1)

    local_bins = np.zeros(nbins)
    dV = dx * dy * dz
    for i in range(ng, nx - ng):
        x = (offs_x + (i - ng) + 0.5) * dx
        for j in range(ng, ny - ng):
            y = (j - ng + 0.5) * dy
            for k in range(ng, nz - ng):
                z = (k - ng + 0.5) * dz
                r = ((x - x0)**2 + (y - y0)**2 + (z - z0)**2) ** 0.5
                idx = np.searchsorted(r_edges, r, side="right") - 1
                if idx < 0:
                    idx = 0
                if idx >= nbins:
                    idx = nbins - 1
                local_bins[idx] += pr[0, i, j, k] * dV

    global_bins = np.zeros(nbins)
    comm.Allreduce(local_bins, global_bins, op=MPI.SUM)
    menc = np.cumsum(global_bins)
    return menc, r_edges

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

    if PHYSICS in ("rmhd", "grmhd"):
        init = str(settings.get("RMHD_INIT", "uniform")).lower()
        if init == "riemann":
            x_split = float(settings.get("RMHD_RIEMANN_X0", 0.5 * Lx))
            left = settings.get("RMHD_RIEMANN_LEFT", [1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.0])
            right = settings.get("RMHD_RIEMANN_RIGHT", [0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0, 0.0])
            if len(left) < 9:
                left = list(left) + [0.0] * (9 - len(left))
            if len(right) < 9:
                right = list(right) + [0.0] * (9 - len(right))
            for i in range(pr.shape[1]):
                x = (x0 + (i - NG) + 0.5) * dx
                vals = left if x < x_split else right
                pr[0, i, :, :] = vals[0]
                pr[1, i, :, :] = vals[1]
                pr[2, i, :, :] = vals[2]
                pr[3, i, :, :] = vals[3]
                pr[4, i, :, :] = vals[4]
                pr[5, i, :, :] = vals[5]
                pr[6, i, :, :] = vals[6]
                pr[7, i, :, :] = vals[7]
                pr[8, i, :, :] = vals[8]

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

def expected_nv():
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
    return nv

def latest_run_dir(base="results"):
    runs = sorted([d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)])
    if not runs:
        return None
    last = runs[-1]
    subs = sorted([d for d in glob.glob(os.path.join(last, "*")) if os.path.isdir(d)])
    return subs[-1] if subs else last

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
    shape = data["shape"] if "shape" in data.files else None
    if meta is None or len(meta) < 5:
        return pr, 0.0, 0, shape
    t = float(meta[3])
    step = int(meta[4])
    return pr, t, step, shape


def advance_block(pr, dx, dy, dz, dt, offs_x, ng):
    if PHYSICS in ("hydro", "sn"):
        if RK_ORDER == 3:
            return srhd_core.step_ssprk3(pr, dx, dy, dz, dt)
        return srhd_core.step_ssprk2(pr, dx, dy, dz, dt)
    if PHYSICS == "rmhd":
        if RK_ORDER == 3:
            return rmhd_core.step_ssprk3(pr, dx, dy, dz, dt)
        return rmhd_core.step_ssprk2(pr, dx, dy, dz, dt)
    if PHYSICS == "grhd":
        if RK_ORDER == 3:
            return grhd_core.step_ssprk3(pr, dx, dy, dz, dt, offs_x, ng)
        return grhd_core.step_ssprk2(pr, dx, dy, dz, dt, offs_x, ng)
    if RK_ORDER == 3:
        return grmhd_core.step_ssprk3(pr, dx, dy, dz, dt, offs_x, ng)
    return grmhd_core.step_ssprk2(pr, dx, dy, dz, dt, offs_x, ng)


def apply_sources(pr, dt, dx, dy, dz, offs_x, ng, t, step, menc, r_edges, comm):
    pr = dissipation.apply_causal_dissipation(pr, dt, dx, dy, dz, settings)
    pr = source_terms.apply_cooling_heating(pr, dt, settings)
    pr = source_terms.apply_two_temperature(pr, dt, settings)
    pr = chemistry.apply_ion_chemistry(pr, dt, settings)
    if (PHYSICS == "sn" and settings.get("SN_GRAVITY_ENABLED", False)
            and str(settings.get("SN_GRAVITY_MODEL", "point_mass")).lower() == "monopole"):
        every = int(settings.get("SN_GRAVITY_UPDATE_EVERY", 1))
        if every <= 0:
            every = 1
        if step % every == 0 or menc is None:
            menc, r_edges = compute_monopole_mass(pr, dx, dy, dz, offs_x, ng, settings, comm)
    pr = gravity.apply_gravity(pr, dt, dx, dy, dz, settings, offs_x, ng, menc, r_edges)
    pr = source_terms.apply_sn_heating(pr, dt, dx, dy, dz, settings, offs_x, ng, t)
    pr = source_terms.apply_neutrino_transport(pr, dt, dx, dy, dz, settings, offs_x, ng, t)
    pr = plasma_microphysics.apply_nonideal_mhd(pr, dt, dx, dy, dz, settings, ng)
    pr = source_terms.apply_radiation_coupling(pr, dt, settings)
    pr = source_terms.apply_kinetic_effects(pr, dt, dx, dy, dz, settings, ng)
    return pr, menc, r_edges

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
        if settings.get("SAVE_RUN_CONFIG", True) and not restart_path:
            write_run_config(RUN_DIR, settings)

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
        if ADAPTIVITY_ENABLED:
            print(f"[startup] adaptivity={ADAPTIVITY_MODE} refinement={settings.get('ADAPTIVITY_REFINEMENT')}",
                  flush=True)

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
    menc = None
    r_edges = None
    if restart_path:
        pr, t, step, shape = load_checkpoint(restart_path)
        if RESTART_STRICT and shape is not None:
            expected = np.array([expected_nv(), nx_loc + 2*NG, ny_loc + 2*NG, nz_loc + 2*NG], dtype=np.int64)
            if not np.all(shape == expected):
                raise RuntimeError(f"restart shape mismatch: checkpoint {shape} vs expected {expected}")
        if rank == 0:
            print(f"[restart] loaded {restart_path} at t={t:.6e} step={step}", flush=True)

    # adaptivity setup (single-rank nested refinement)
    fine_info = None
    pr_f = None
    fine_subcycles = None
    if ADAPTIVITY_ENABLED:
        if size != 1:
            raise RuntimeError("ADAPTIVITY_ENABLED currently requires a single MPI rank.")
        if ADAPTIVITY_MODE != "nested_static":
            raise RuntimeError("ADAPTIVITY_MODE must be 'nested_static'.")
        fine_info = adaptivity.build_refine_info(settings, dx, dy, dz, NG, NX, NY, NZ)
        nx_f, ny_f, nz_f = fine_info["fine_shape"]
        pr_f = np.zeros((pr.shape[0], nx_f + 2*NG, ny_f + 2*NG, nz_f + 2*NG), dtype=np.float64)
        adaptivity.fill_fine_from_coarse(pr_f, pr, fine_info, NG)
        if ADAPTIVITY_SUBCYCLES is None:
            fine_subcycles = int(fine_info["refine"])
        else:
            fine_subcycles = int(ADAPTIVITY_SUBCYCLES)
        if fine_subcycles < 1:
            fine_subcycles = 1
        if rank == 0:
            print(f"[adaptivity] region box={fine_info['box']} subcycles={fine_subcycles}", flush=True)

    if not restart_path:
        # initial BCs
        boundary.apply_periodic_yz(pr, NG)
        if rank == size-1: boundary.apply_outflow_right_x(pr, NG)
        if rank == 0 and PHYSICS != "sn":
            nozzle.apply_nozzle_left_x(
                pr, dx, dy, dz, ny_loc, nz_loc, JET_CENTER[1], JET_CENTER[2], rng, settings
            )
    if ADAPTIVITY_ENABLED and pr_f is not None:
        adaptivity.fill_fine_from_coarse(pr_f, pr, fine_info, NG)

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
    perf_step_accum = 0.0
    perf_io_accum = 0.0
    perf_diag_accum = 0.0
    perf_count = 0

    while t < T_END:
        step_start = time.perf_counter() if PERF_ENABLED else 0.0
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
        exchange_halos(pr, comm, left, right, HALO_EXCHANGE)
        boundary.apply_periodic_yz(pr, NG)
        if rank == size-1: boundary.apply_outflow_right_x(pr, NG)
        if rank == 0 and PHYSICS != "sn":
            nozzle.apply_nozzle_left_x(
                pr, dx, dy, dz, ny_loc, nz_loc, JET_CENTER[1], JET_CENTER[2], rng, settings
            )

        # advance one step
        pr = advance_block(pr, dx, dy, dz, dt, offs[rank], NG)

        # re-apply BCs after update
        boundary.apply_periodic_yz(pr, NG)
        if rank == size-1: boundary.apply_outflow_right_x(pr, NG)
        if rank == 0 and PHYSICS != "sn":
            nozzle.apply_nozzle_left_x(
                pr, dx, dy, dz, ny_loc, nz_loc, JET_CENTER[1], JET_CENTER[2], rng, settings
            )
        pr, menc, r_edges = apply_sources(pr, dt, dx, dy, dz, offs[rank], NG, t, step, menc, r_edges, comm)

        # nested refinement update (single-rank)
        if ADAPTIVITY_ENABLED and pr_f is not None:
            dx_f, dy_f, dz_f = fine_info["fine_spacing"]
            dt_f = dt / float(fine_subcycles)
            for _ in range(fine_subcycles):
                adaptivity.fill_fine_ghosts_from_coarse(pr_f, pr, fine_info, NG)
                pr_f = advance_block(pr_f, dx_f, dy_f, dz_f, dt_f, 0, NG)
                pr_f, _, _ = apply_sources(pr_f, dt_f, dx_f, dy_f, dz_f, 0, NG, t, step, None, None, comm)
            adaptivity.restrict_coarse_from_fine(pr, pr_f, fine_info, NG)

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
            io_start = time.perf_counter() if PERF_ENABLED else 0.0
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
            if PERF_ENABLED:
                perf_io_accum += time.perf_counter() - io_start

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
            diag_start = time.perf_counter() if PERF_ENABLED else 0.0
            diagnostics.compute_centerline_and_write(
                pr, dx, dy, dz, offs[rank], counts, comm, rank,
                step, t, RUN_DIR, V_MAX, NG
            )
            if PHYSICS == "sn":
                diagnostics.compute_sn_diagnostics_and_write(
                    pr, dx, dy, dz, offs[rank], counts, comm, rank,
                    step, t, RUN_DIR, settings, NG
                )
            if PHYSICS in ("rmhd", "grmhd"):
                diagnostics.compute_divb_and_write(
                    pr, dx, dy, dz, offs[rank], counts, comm, rank,
                    step, t, RUN_DIR, NG
                )
            if settings.get("DIAG_COCOON_ENABLED", False):
                diagnostics.compute_cocoon_diagnostics_and_write(
                    pr, dx, dy, dz, offs[rank], counts, comm, rank,
                    step, t, RUN_DIR, settings, NG, TRACER_OFFSET
                )
            if settings.get("DIAG_MIXING_ENABLED", False):
                diagnostics.compute_mixing_diagnostics_and_write(
                    pr, dx, dy, dz, offs[rank], counts, comm, rank,
                    step, t, RUN_DIR, settings, NG, TRACER_OFFSET
                )
            if PERF_ENABLED:
                perf_diag_accum += time.perf_counter() - diag_start

        if CHECKPOINT_EVERY > 0 and (step % CHECKPOINT_EVERY == 0):
            ckpt = os.path.join(RUN_DIR, f"checkpoint_rank{rank:04d}_step{step:06d}.npz")
            np.savez(
                ckpt,
                prim = pr,
                meta = np.array([dx, dy, dz, t, step], dtype=np.float64),
                shape = np.array(pr.shape, dtype=np.int64),
                comment = "Checkpoint (with ghosts)."
            )
            if rank == 0:
                print(f"[ckpt] wrote {ckpt}", flush=True)
        if PERF_ENABLED:
            perf_step_accum += time.perf_counter() - step_start
            perf_count += 1
            if PERF_EVERY > 0 and (step % PERF_EVERY == 0):
                if rank == 0:
                    perf_path = os.path.join(RUN_DIR, "perf.csv")
                    new = not os.path.exists(perf_path)
                    with open(perf_path, "a") as f:
                        if new:
                            f.write("step,step_time,io_time,diag_time,count\n")
                        f.write(f"{step},{perf_step_accum:.6e},{perf_io_accum:.6e},"
                                f"{perf_diag_accum:.6e},{perf_count}\n")
                if PERF_RESET_EVERY > 0 and (step % PERF_RESET_EVERY == 0):
                    perf_step_accum = 0.0
                    perf_io_accum = 0.0
                    perf_diag_accum = 0.0
                    perf_count = 0
    if rank == 0:
        print("Done.", flush=True)

if __name__ == "__main__":
    main()
