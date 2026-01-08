# utils/diagnostics.py
import os, math, numpy as np
from mpi4py import MPI

_MIXING_HISTORY = {"time": None, "mass": None}

def compute_diagnostics_and_write(pr, dx, dy, dz,
                                  offs_x, counts, comm, rank,
                                  step, t, dt, amax, run_dir, prim_to_cons, NG, is_rmhd=False):
    """
    Compute and append global diagnostics (max Γ, inlet flux).
    """
    nx_loc = pr.shape[1] - 2*NG
    ny_loc = pr.shape[2] - 2*NG
    nz_loc = pr.shape[3] - 2*NG

    # local max gamma
    local_maxG = 0.0
    for i in range(NG, NG + nx_loc):
        for j in range(NG, NG + ny_loc):
            for k in range(NG, NG + nz_loc):
                vx, vy, vz = pr[1,i,j,k], pr[2,i,j,k], pr[3,i,j,k]
                v2 = vx*vx + vy*vy + vz*vz
                if v2 >= 1.0:
                    v2 = 1.0 - 1e-14
                W = 1.0 / math.sqrt(1.0 - v2)
                if W > local_maxG:
                    local_maxG = W

    global_maxG = comm.allreduce(local_maxG, op=MPI.MAX)

    # inlet plane (rank 0)
    inlet_flux = 0.0
    if rank == 0:
        i = NG
        for j in range(NG, NG + ny_loc):
            for k in range(NG, NG + nz_loc):
                rho, vx, vy, vz, p = pr[0,i,j,k], pr[1,i,j,k], pr[2,i,j,k], pr[3,i,j,k], pr[4,i,j,k]
                if is_rmhd:
                    Bx, By, Bz, psi = pr[5,i,j,k], pr[6,i,j,k], pr[7,i,j,k], pr[8,i,j,k]
                    _, Sx, _, _, _, _, _, _, _ = prim_to_cons(rho, vx, vy, vz, p, Bx, By, Bz, psi)
                else:
                    _, Sx, _, _, _ = prim_to_cons(rho, vx, vy, vz, p)
                inlet_flux += Sx * dy * dz

    total_inlet_flux = comm.allreduce(inlet_flux, op=MPI.SUM)
    signed_flux = total_inlet_flux
    abs_flux = abs(total_inlet_flux)

    if rank == 0:
        fn = os.path.join(run_dir, "diagnostics.csv")
        new = not os.path.exists(fn)
        with open(fn, "a") as f:
            if new:
                f.write("step,time,dt,amax,maxGamma,inletFlux_signed,inletFlux_abs\n")
            f.write(f"{step},{t:.8e},{dt:.8e},{amax:.6e},{global_maxG:.6e},"
                    f"{signed_flux:.6e},{abs_flux:.6e}\n")

    return global_maxG, signed_flux, abs_flux

def compute_centerline_and_write(pr, dx, dy, dz,
                                 offs_x, counts, comm, rank,
                                 step, t, run_dir,
                                 V_MAX, NG):
    """
    Gather and write the mid-plane (y,z = mid) centerline profile across x.
    Each rank contributes its interior slab; rank 0 writes a single CSV.

    CSV columns: step,time,i_global,x,Gamma,rho,p,vx
    """
    ny_loc = pr.shape[2] - 2*NG
    nz_loc = pr.shape[3] - 2*NG
    nx_loc = pr.shape[1] - 2*NG

    # mid y,z indices
    j = NG + ny_loc // 2
    k = NG + nz_loc // 2

    i_start = offs_x
    i_end   = offs_x + nx_loc

    i_loc = slice(NG, NG + nx_loc)
    rho_l = pr[0, i_loc, j, k].copy()
    vx_l  = pr[1, i_loc, j, k].copy()
    vy_l  = pr[2, i_loc, j, k].copy()
    vz_l  = pr[3, i_loc, j, k].copy()
    p_l   = pr[4, i_loc, j, k].copy()

    v2 = vx_l*vx_l + vy_l*vy_l + vz_l*vz_l
    vmax2 = V_MAX*V_MAX
    mask = v2 >= vmax2
    if np.any(mask):
        fac = V_MAX/np.sqrt(v2[mask] + 1e-32)
        vx_l[mask] *= fac; vy_l[mask] *= fac; vz_l[mask] *= fac
        v2[mask] = vmax2 - 1e-14
    Gamma_l = 1.0/np.sqrt(1.0 - np.clip(v2, 0, 1 - 1e-14))

    # Gather all pieces onto rank 0
    count = np.int32(nx_loc)
    counts_all = comm.gather(count, root=0)
    if rank == 0:
        total = np.sum(counts_all)
        disp = np.zeros_like(counts_all, dtype=np.int32)
        for r in range(1, len(counts_all)):
            disp[r] = disp[r-1] + counts_all[r-1]
        out_rho = np.empty(total); out_p = np.empty(total)
        out_vx = np.empty(total); out_G = np.empty(total)
    else:
        out_rho = out_p = out_vx = out_G = None; disp = None

    comm.Gatherv(sendbuf=rho_l, recvbuf=(out_rho,(counts_all,disp)) if rank==0 else None, root=0)
    comm.Gatherv(sendbuf=p_l,   recvbuf=(out_p,(counts_all,disp))   if rank==0 else None, root=0)
    comm.Gatherv(sendbuf=vx_l,  recvbuf=(out_vx,(counts_all,disp))  if rank==0 else None, root=0)
    comm.Gatherv(sendbuf=Gamma_l,recvbuf=(out_G,(counts_all,disp))  if rank==0 else None, root=0)

    if rank == 0:
        total = int(total)
        i_global = np.arange(total)
        x = (i_global + 0.5)*dx
        path = os.path.join(run_dir, "centerline.csv")
        new = not os.path.exists(path)
        with open(path, "a") as f:
            if new:
                f.write("step,time,i,x,Gamma,rho,p,vx\n")
            for ii in range(total):
                f.write(f"{step},{t:.8e},{i_global[ii]},{x[ii]:.8e},"
                        f"{out_G[ii]:.8e},{out_rho[ii]:.8e},"
                        f"{out_p[ii]:.8e},{out_vx[ii]:.8e}\n")

def compute_divb_and_write(pr, dx, dy, dz,
                           offs_x, counts, comm, rank,
                           step, t, run_dir, NG):
    """
    Compute and append divB diagnostics for RMHD runs.
    CSV columns: step,time,divB_max,divB_rms,divB_mean
    """
    if pr.shape[0] < 8:
        return None, None, None

    nx_loc = pr.shape[1] - 2*NG
    ny_loc = pr.shape[2] - 2*NG
    nz_loc = pr.shape[3] - 2*NG

    divb_vals = []
    for i in range(NG, NG + nx_loc):
        for j in range(NG, NG + ny_loc):
            for k in range(NG, NG + nz_loc):
                dBx = (pr[5, i+1, j, k] - pr[5, i-1, j, k]) / (2.0*dx)
                dBy = (pr[6, i, j+1, k] - pr[6, i, j-1, k]) / (2.0*dy)
                dBz = (pr[7, i, j, k+1] - pr[7, i, j, k-1]) / (2.0*dz)
                divb_vals.append(dBx + dBy + dBz)

    if divb_vals:
        divb_arr = np.array(divb_vals, dtype=np.float64)
        local_max = np.max(np.abs(divb_arr))
        local_sum = np.sum(divb_arr)
        local_sumsq = np.sum(divb_arr*divb_arr)
        local_n = divb_arr.size
    else:
        local_max = 0.0
        local_sum = 0.0
        local_sumsq = 0.0
        local_n = 0

    global_max = comm.allreduce(local_max, op=MPI.MAX)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_sumsq = comm.allreduce(local_sumsq, op=MPI.SUM)
    global_n = comm.allreduce(local_n, op=MPI.SUM)

    if global_n > 0:
        global_mean = global_sum / global_n
        global_rms = np.sqrt(global_sumsq / global_n)
    else:
        global_mean = 0.0
        global_rms = 0.0

    if rank == 0:
        fn = os.path.join(run_dir, "divb.csv")
        new = not os.path.exists(fn)
        with open(fn, "a") as f:
            if new:
                f.write("step,time,divB_max,divB_rms,divB_mean\n")
            f.write(f"{step},{t:.8e},{global_max:.6e},{global_rms:.6e},{global_mean:.6e}\n")

    return global_max, global_rms, global_mean


def compute_sn_diagnostics_and_write(pr, dx, dy, dz,
                                     offs_x, counts, comm, rank,
                                     step, t, run_dir, cfg, NG):
    """
    SN-lite diagnostics: shock radius, gain mass, heating efficiency.
    CSV columns: step,time,shock_radius,gain_mass,heat_power,heat_abs,heating_eff
    """
    if cfg.get("PHYSICS") not in ("sn",):
        return None

    nx_loc = pr.shape[1] - 2*NG
    ny_loc = pr.shape[2] - 2*NG
    nz_loc = pr.shape[3] - 2*NG

    center = cfg.get("SN_GRAVITY_CENTER") or [0.5*cfg.get("Lx", 1.0),
                                              0.5*cfg.get("Ly", 1.0),
                                              0.5*cfg.get("Lz", 1.0)]
    x0, y0, z0 = float(center[0]), float(center[1]), float(center[2])
    p_ref = cfg.get("SN_SHOCK_P_REF")
    if p_ref is None:
        p_ref = cfg.get("P_AMB", cfg.get("P_EQ", 1.0e-2))
    p_ratio = float(cfg.get("SN_SHOCK_P_RATIO", 1.2))
    p_thresh = p_ratio * float(p_ref)

    heat_enabled = bool(cfg.get("SN_HEATING_ENABLED", False))
    model = str(cfg.get("SN_HEATING_MODEL", "gain_spherical")).lower()
    h0 = float(cfg.get("SN_HEATING_RATE", 0.0))
    c0 = float(cfg.get("SN_COOLING_RATE", 0.0))
    r0 = float(cfg.get("SN_GAIN_RADIUS", 0.2))
    r1 = float(cfg.get("SN_GAIN_WIDTH", 0.1))
    rho_exp = float(cfg.get("SN_HEATING_RHO_EXP", 0.0))
    p_exp = float(cfg.get("SN_HEATING_P_EXP", 0.0))

    dV = dx * dy * dz
    local_rsh = 0.0
    gain_mass = 0.0
    heat_power = 0.0
    heat_abs = 0.0

    for i in range(NG, NG + nx_loc):
        x = (offs_x + (i - NG) + 0.5) * dx
        for j in range(NG, NG + ny_loc):
            y = (j - NG + 0.5) * dy
            for k in range(NG, NG + nz_loc):
                z = (k - NG + 0.5) * dz
                rx = x - x0
                ry = y - y0
                rz = z - z0
                r = (rx*rx + ry*ry + rz*rz) ** 0.5
                p = pr[4, i, j, k]
                if p >= p_thresh and r > local_rsh:
                    local_rsh = r

                if not heat_enabled:
                    continue

                heat = 0.0
                cool = 0.0
                if model == "gain_spherical":
                    if r > r0:
                        xi = (r - r0) / max(r1, 1e-12)
                        weight = 1.0 / (1.0 + xi*xi)
                        heat = h0 * weight
                        cool = c0 * weight
                elif model == "gain_exponential":
                    if r >= r0:
                        xi = (r - r0) / max(r1, 1e-12)
                        weight = pow(2.718281828, -xi)
                        heat = h0 * weight
                    else:
                        xi = (r0 - r) / max(r1, 1e-12)
                        weight = pow(2.718281828, -xi)
                        cool = c0 * weight
                elif model == "gain_gaussian":
                    if r >= r0:
                        xi = (r - r0) / max(r1, 1e-12)
                        weight = pow(2.718281828, -(xi * xi))
                        heat = h0 * weight
                    else:
                        xi = (r0 - r) / max(r1, 1e-12)
                        weight = pow(2.718281828, -(xi * xi))
                        cool = c0 * weight
                elif model == "constant":
                    heat = h0
                    cool = c0

                if heat == 0.0 and cool == 0.0:
                    continue

                rho = pr[0, i, j, k]
                if rho <= 0.0:
                    continue
                scale = 1.0
                if rho_exp != 0.0:
                    scale *= rho ** rho_exp
                if p_exp != 0.0:
                    scale *= max(p, 1e-12) ** p_exp
                net = (heat - cool) * scale
                if net > 0.0:
                    gain_mass += rho * dV
                heat_power += net * dV
                heat_abs += (abs(heat) + abs(cool)) * scale * dV

    global_rsh = comm.allreduce(local_rsh, op=MPI.MAX)
    global_gain_mass = comm.allreduce(gain_mass, op=MPI.SUM)
    global_heat_power = comm.allreduce(heat_power, op=MPI.SUM)
    global_heat_abs = comm.allreduce(heat_abs, op=MPI.SUM)
    eff = global_heat_power / global_heat_abs if global_heat_abs > 0.0 else 0.0

    if rank == 0:
        fn = os.path.join(run_dir, "sn_diagnostics.csv")
        new = not os.path.exists(fn)
        with open(fn, "a") as f:
            if new:
                f.write("step,time,shock_radius,gain_mass,heat_power,heat_abs,heating_eff\n")
            f.write(f"{step},{t:.8e},{global_rsh:.8e},{global_gain_mass:.8e},"
                    f"{global_heat_power:.8e},{global_heat_abs:.8e},{eff:.8e}\n")

    return global_rsh, global_gain_mass, global_heat_power, global_heat_abs, eff


def compute_cocoon_diagnostics_and_write(pr, dx, dy, dz,
                                         offs_x, counts, comm, rank,
                                         step, t, run_dir, cfg, NG, tracer_offset):
    if not cfg.get("DIAG_COCOON_ENABLED", False):
        return None
    ntr = int(cfg.get("N_TRACERS", 0))
    if ntr <= 0:
        return None
    tidx = int(cfg.get("DIAG_COCOON_TRACER_IDX", 0))
    if tidx < 0 or tidx >= ntr:
        return None
    tmin = float(cfg.get("DIAG_COCOON_TRACER_MIN", 0.05))
    tmax = float(cfg.get("DIAG_COCOON_TRACER_MAX", 1.0))
    tmin = max(0.0, min(1.0, tmin))
    tmax = max(tmin, min(1.0, tmax))

    nx_loc = pr.shape[1] - 2*NG
    ny_loc = pr.shape[2] - 2*NG
    nz_loc = pr.shape[3] - 2*NG
    dV = dx * dy * dz

    local_sum_p = 0.0
    local_vol = 0.0
    local_max_p = 0.0
    idx = tracer_offset + tidx
    for i in range(NG, NG + nx_loc):
        for j in range(NG, NG + ny_loc):
            for k in range(NG, NG + nz_loc):
                tr = pr[idx, i, j, k]
                if tmin <= tr <= tmax:
                    p = pr[4, i, j, k]
                    local_sum_p += p * dV
                    local_vol += dV
                    if p > local_max_p:
                        local_max_p = p

    global_sum_p = comm.allreduce(local_sum_p, op=MPI.SUM)
    global_vol = comm.allreduce(local_vol, op=MPI.SUM)
    global_max_p = comm.allreduce(local_max_p, op=MPI.MAX)
    avg_p = global_sum_p / global_vol if global_vol > 0.0 else 0.0

    if rank == 0:
        fn = os.path.join(run_dir, "cocoon.csv")
        new = not os.path.exists(fn)
        with open(fn, "a") as f:
            if new:
                f.write("step,time,cocoon_p_avg,cocoon_p_max,cocoon_volume\n")
            f.write(f"{step},{t:.8e},{avg_p:.8e},{global_max_p:.8e},{global_vol:.8e}\n")

    return avg_p, global_max_p, global_vol


def compute_mixing_diagnostics_and_write(pr, dx, dy, dz,
                                         offs_x, counts, comm, rank,
                                         step, t, run_dir, cfg, NG, tracer_offset):
    if not cfg.get("DIAG_MIXING_ENABLED", False):
        return None
    ntr = int(cfg.get("N_TRACERS", 0))
    if ntr <= 0:
        return None
    tidx = int(cfg.get("DIAG_MIXING_TRACER_IDX", 0))
    if tidx < 0 or tidx >= ntr:
        return None
    tmin = float(cfg.get("DIAG_MIXING_MIN", 0.05))
    tmax = float(cfg.get("DIAG_MIXING_MAX", 0.95))
    tmin = max(0.0, min(1.0, tmin))
    tmax = max(tmin, min(1.0, tmax))
    xfrac = float(cfg.get("DIAG_MIXING_X_FRAC", 0.5))
    xfrac = max(0.0, min(1.0, xfrac))

    nx = int(cfg.get("NX", pr.shape[1] - 2*NG))
    ny_loc = pr.shape[2] - 2*NG
    nz_loc = pr.shape[3] - 2*NG
    i_glob = int(xfrac * nx)
    i_glob = max(0, min(nx - 1, i_glob))
    i_loc = i_glob - offs_x
    if i_loc < 0 or i_loc >= pr.shape[1] - 2*NG:
        local_min_r = float("inf")
        local_max_r = 0.0
        local_mass = 0.0
    else:
        i = NG + i_loc
        center = cfg.get("JET_CENTER") or [0.0, 0.5*cfg.get("Ly", 1.0), 0.5*cfg.get("Lz", 1.0)]
        y0, z0 = float(center[1]), float(center[2])
        local_min_r = float("inf")
        local_max_r = 0.0
        local_mass = 0.0
        idx = tracer_offset + tidx
        dV = dx * dy * dz
        for j in range(NG, NG + ny_loc):
            y = (j - NG + 0.5) * dy
            for k in range(NG, NG + nz_loc):
                z = (k - NG + 0.5) * dz
                tr = pr[idx, i, j, k]
                if tmin <= tr <= tmax:
                    r = math.sqrt((y - y0)**2 + (z - z0)**2)
                    local_min_r = min(local_min_r, r)
                    local_max_r = max(local_max_r, r)
                    local_mass += pr[0, i, j, k] * dV

    global_min_r = comm.allreduce(local_min_r, op=MPI.MIN)
    global_max_r = comm.allreduce(local_max_r, op=MPI.MAX)
    global_mass = comm.allreduce(local_mass, op=MPI.SUM)
    thickness = max(0.0, global_max_r - global_min_r) if np.isfinite(global_min_r) else 0.0

    if rank == 0:
        prev_t = _MIXING_HISTORY["time"]
        prev_m = _MIXING_HISTORY["mass"]
        if prev_t is None or prev_m is None or t <= prev_t:
            rate = 0.0
        else:
            rate = (global_mass - prev_m) / (t - prev_t)
        _MIXING_HISTORY["time"] = t
        _MIXING_HISTORY["mass"] = global_mass
        fn = os.path.join(run_dir, "mixing.csv")
        new = not os.path.exists(fn)
        with open(fn, "a") as f:
            if new:
                f.write("step,time,x_index,mixing_thickness,mixing_mass,mixing_rate\n")
            f.write(f"{step},{t:.8e},{i_glob},{thickness:.8e},{global_mass:.8e},{rate:.8e}\n")

    return thickness, global_mass
