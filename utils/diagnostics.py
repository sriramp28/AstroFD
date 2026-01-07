# utils/diagnostics.py
import os, math, numpy as np
from mpi4py import MPI

def compute_diagnostics_and_write(pr, dx, dy, dz,
                                  offs_x, counts, comm, rank,
                                  step, t, dt, amax, run_dir, prim_to_cons, NG):
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
                if pr.shape[0] >= 9:
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
