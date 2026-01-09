#!/usr/bin/env python3
# core/adaptivity.py
# Static nested-grid refinement helpers (single-rank).
import math
import numpy as np


def _default_region(Lx, Ly, Lz):
    return [0.25 * Lx, 0.75 * Lx, 0.25 * Ly, 0.75 * Ly, 0.25 * Lz, 0.75 * Lz]


def build_refine_info(cfg, dx, dy, dz, ng, nx, ny, nz):
    region = cfg.get("ADAPTIVITY_REGION")
    if region is None:
        region = _default_region(cfg["Lx"], cfg["Ly"], cfg["Lz"])
    if not (isinstance(region, (list, tuple)) and len(region) == 6):
        raise ValueError("ADAPTIVITY_REGION must be [xlo,xhi,ylo,yhi,zlo,zhi]")
    refine = int(cfg.get("ADAPTIVITY_REFINEMENT", 2))
    if refine < 2:
        raise ValueError("ADAPTIVITY_REFINEMENT must be >= 2")

    xlo, xhi, ylo, yhi, zlo, zhi = [float(v) for v in region]
    if not (0.0 <= xlo < xhi <= cfg["Lx"] and 0.0 <= ylo < yhi <= cfg["Ly"] and 0.0 <= zlo < zhi <= cfg["Lz"]):
        raise ValueError("ADAPTIVITY_REGION must lie within domain bounds")

    i0 = max(ng, int(math.floor(xlo / dx + ng - 0.5)))
    i1 = min(nx + ng - 1, int(math.floor(xhi / dx + ng - 0.5)))
    j0 = max(ng, int(math.floor(ylo / dy + ng - 0.5)))
    j1 = min(ny + ng - 1, int(math.floor(yhi / dy + ng - 0.5)))
    k0 = max(ng, int(math.floor(zlo / dz + ng - 0.5)))
    k1 = min(nz + ng - 1, int(math.floor(zhi / dz + ng - 0.5)))
    if i1 < i0 or j1 < j0 or k1 < k0:
        raise ValueError("ADAPTIVITY_REGION yields empty refine box")

    nx_f = (i1 - i0 + 1) * refine
    ny_f = (j1 - j0 + 1) * refine
    nz_f = (k1 - k0 + 1) * refine
    dx_f, dy_f, dz_f = dx / refine, dy / refine, dz / refine
    x0_f = (i0 - ng) * dx
    y0_f = (j0 - ng) * dy
    z0_f = (k0 - ng) * dz

    return dict(
        refine=refine,
        box=(i0, i1, j0, j1, k0, k1),
        fine_shape=(nx_f, ny_f, nz_f),
        fine_spacing=(dx_f, dy_f, dz_f),
        fine_origin=(x0_f, y0_f, z0_f),
    )


def build_refine_info_local(cfg, dx, dy, dz, ng, nx_loc, ny, nz, offs_x):
    region = cfg.get("ADAPTIVITY_REGION")
    if region is None:
        region = _default_region(cfg["Lx"], cfg["Ly"], cfg["Lz"])
    if not (isinstance(region, (list, tuple)) and len(region) == 6):
        raise ValueError("ADAPTIVITY_REGION must be [xlo,xhi,ylo,yhi,zlo,zhi]")
    refine = int(cfg.get("ADAPTIVITY_REFINEMENT", 2))
    if refine < 2:
        raise ValueError("ADAPTIVITY_REFINEMENT must be >= 2")

    xlo, xhi, ylo, yhi, zlo, zhi = [float(v) for v in region]
    if not (0.0 <= xlo < xhi <= cfg["Lx"] and 0.0 <= ylo < yhi <= cfg["Ly"] and 0.0 <= zlo < zhi <= cfg["Lz"]):
        raise ValueError("ADAPTIVITY_REGION must lie within domain bounds")

    i0g = max(0, int(math.floor(xlo / dx + 0.5)))
    i1g = min(cfg["NX"] - 1, int(math.floor(xhi / dx - 0.5)))
    j0g = max(0, int(math.floor(ylo / dy + 0.5)))
    j1g = min(ny - 1, int(math.floor(yhi / dy - 0.5)))
    k0g = max(0, int(math.floor(zlo / dz + 0.5)))
    k1g = min(nz - 1, int(math.floor(zhi / dz - 0.5)))

    # map global -> local slab indices
    i0 = i0g - offs_x
    i1 = i1g - offs_x
    if i1 < 0 or i0 >= nx_loc:
        return None
    i0 = max(0, i0); i1 = min(nx_loc - 1, i1)
    if i1 < i0 or j1g < j0g or k1g < k0g:
        return None

    # shift to array indices with ghosts
    i0a = i0 + ng
    i1a = i1 + ng
    j0a = j0g + ng
    j1a = j1g + ng
    k0a = k0g + ng
    k1a = k1g + ng

    nx_f = (i1a - i0a + 1) * refine
    ny_f = (j1a - j0a + 1) * refine
    nz_f = (k1a - k0a + 1) * refine
    dx_f, dy_f, dz_f = dx / refine, dy / refine, dz / refine
    x0_f = (i0a - ng + offs_x) * dx
    y0_f = (j0a - ng) * dy
    z0_f = (k0a - ng) * dz

    return dict(
        refine=refine,
        box=(i0a, i1a, j0a, j1a, k0a, k1a),
        fine_shape=(nx_f, ny_f, nz_f),
        fine_spacing=(dx_f, dy_f, dz_f),
        fine_origin=(x0_f, y0_f, z0_f),
        offs_x=offs_x,
    )


def _field_index(field):
    if isinstance(field, (int, np.integer)):
        return int(field)
    name = str(field).lower()
    if name in ("rho", "density"):
        return 0
    if name in ("p", "pressure"):
        return 4
    return 0


def _estimate_dynamic_region(pr, cfg, dx, dy, dz, ng):
    thresh = cfg.get("ADAPTIVITY_GRAD_THRESHOLD", None)
    buffer_cells = int(cfg.get("ADAPTIVITY_BUFFER", 2))
    fidx = _field_index(cfg.get("ADAPTIVITY_FIELD", 0))

    nx = pr.shape[1] - 2*ng
    ny = pr.shape[2] - 2*ng
    nz = pr.shape[3] - 2*ng
    if nx <= 2 or ny <= 2 or nz <= 2:
        return None

    max_g = 0.0
    box = None
    for i in range(ng + 1, ng + nx - 1):
        for j in range(ng + 1, ng + ny - 1):
            for k in range(ng + 1, ng + nz - 1):
                ddx = (pr[fidx, i+1, j, k] - pr[fidx, i-1, j, k]) / (2.0*dx)
                ddy = (pr[fidx, i, j+1, k] - pr[fidx, i, j-1, k]) / (2.0*dy)
                ddz = (pr[fidx, i, j, k+1] - pr[fidx, i, j, k-1]) / (2.0*dz)
                g = math.sqrt(ddx*ddx + ddy*ddy + ddz*ddz)
                if thresh is None:
                    if g > max_g:
                        max_g = g
                        box = (i, i, j, j, k, k)
                elif g > thresh:
                    if box is None:
                        box = (i, i, j, j, k, k)
                    else:
                        i0, i1, j0, j1, k0, k1 = box
                        box = (min(i0, i), max(i1, i), min(j0, j), max(j1, j), min(k0, k), max(k1, k))

    if box is None:
        return None

    i0, i1, j0, j1, k0, k1 = box
    i0 = max(ng, i0 - buffer_cells)
    j0 = max(ng, j0 - buffer_cells)
    k0 = max(ng, k0 - buffer_cells)
    i1 = min(ng + nx - 1, i1 + buffer_cells)
    j1 = min(ng + ny - 1, j1 + buffer_cells)
    k1 = min(ng + nz - 1, k1 + buffer_cells)
    return (i0, i1, j0, j1, k0, k1)


def build_dynamic_refine_info(pr, cfg, dx, dy, dz, ng, nx, ny, nz, prev_info=None):
    box = _estimate_dynamic_region(pr, cfg, dx, dy, dz, ng)
    if box is None:
        return prev_info, False

    i0, i1, j0, j1, k0, k1 = box
    xlo = (i0 - ng) * dx
    xhi = (i1 - ng + 1) * dx
    ylo = (j0 - ng) * dy
    yhi = (j1 - ng + 1) * dy
    zlo = (k0 - ng) * dz
    zhi = (k1 - ng + 1) * dz
    region = [xlo, xhi, ylo, yhi, zlo, zhi]
    cfg_tmp = dict(cfg)
    cfg_tmp["ADAPTIVITY_REGION"] = region
    info = build_refine_info(cfg_tmp, dx, dy, dz, ng, nx, ny, nz)

    if prev_info and prev_info.get("box") == info.get("box"):
        return prev_info, False
    return info, True


def build_dynamic_refine_info_local(pr, cfg, dx, dy, dz, ng, nx_loc, ny, nz, offs_x, prev_info=None):
    box = _estimate_dynamic_region(pr, cfg, dx, dy, dz, ng)
    if box is None:
        return prev_info, False

    i0, i1, j0, j1, k0, k1 = box
    xlo = (i0 - ng + offs_x) * dx
    xhi = (i1 - ng + 1 + offs_x) * dx
    ylo = (j0 - ng) * dy
    yhi = (j1 - ng + 1) * dy
    zlo = (k0 - ng) * dz
    zhi = (k1 - ng + 1) * dz
    region = [xlo, xhi, ylo, yhi, zlo, zhi]
    cfg_tmp = dict(cfg)
    cfg_tmp["ADAPTIVITY_REGION"] = region
    info = build_refine_info_local(cfg_tmp, dx, dy, dz, ng, nx_loc, ny, nz, offs_x)
    if info is None:
        return prev_info, False
    if prev_info and prev_info.get("box") == info.get("box"):
        return prev_info, False
    return info, True


def _trilinear_sample(pr, ic, jc, kc):
    nx, ny, nz = pr.shape[1], pr.shape[2], pr.shape[3]
    i0 = int(math.floor(ic)); j0 = int(math.floor(jc)); k0 = int(math.floor(kc))
    i0 = max(0, min(nx - 2, i0))
    j0 = max(0, min(ny - 2, j0))
    k0 = max(0, min(nz - 2, k0))
    fx = ic - i0; fy = jc - j0; fz = kc - k0
    c000 = pr[:, i0, j0, k0]
    c100 = pr[:, i0 + 1, j0, k0]
    c010 = pr[:, i0, j0 + 1, k0]
    c110 = pr[:, i0 + 1, j0 + 1, k0]
    c001 = pr[:, i0, j0, k0 + 1]
    c101 = pr[:, i0 + 1, j0, k0 + 1]
    c011 = pr[:, i0, j0 + 1, k0 + 1]
    c111 = pr[:, i0 + 1, j0 + 1, k0 + 1]
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


def fill_fine_from_coarse(pr_f, pr_c, info, ng):
    dx_f, dy_f, dz_f = info["fine_spacing"]
    x0_f, y0_f, z0_f = info["fine_origin"]
    nx_f, ny_f, nz_f = info["fine_shape"]
    dx, dy, dz = dx_f * info["refine"], dy_f * info["refine"], dz_f * info["refine"]
    for i in range(ng, ng + nx_f):
        x = x0_f + (i - ng + 0.5) * dx_f
        ic = x / dx + ng - 0.5
        for j in range(ng, ng + ny_f):
            y = y0_f + (j - ng + 0.5) * dy_f
            jc = y / dy + ng - 0.5
            for k in range(ng, ng + nz_f):
                z = z0_f + (k - ng + 0.5) * dz_f
                kc = z / dz + ng - 0.5
                pr_f[:, i, j, k] = _trilinear_sample(pr_c, ic, jc, kc)


def fill_fine_ghosts_from_coarse(pr_f, pr_c, info, ng):
    dx_f, dy_f, dz_f = info["fine_spacing"]
    x0_f, y0_f, z0_f = info["fine_origin"]
    nx_f, ny_f, nz_f = info["fine_shape"]
    dx, dy, dz = dx_f * info["refine"], dy_f * info["refine"], dz_f * info["refine"]
    nx_tot, ny_tot, nz_tot = pr_f.shape[1], pr_f.shape[2], pr_f.shape[3]
    i_in = range(ng, ng + nx_f)
    j_in = range(ng, ng + ny_f)
    k_in = range(ng, ng + nz_f)
    for i in range(nx_tot):
        x = x0_f + (i - ng + 0.5) * dx_f
        ic = x / dx + ng - 0.5
        for j in range(ny_tot):
            y = y0_f + (j - ng + 0.5) * dy_f
            jc = y / dy + ng - 0.5
            for k in range(nz_tot):
                if i in i_in and j in j_in and k in k_in:
                    continue
                z = z0_f + (k - ng + 0.5) * dz_f
                kc = z / dz + ng - 0.5
                pr_f[:, i, j, k] = _trilinear_sample(pr_c, ic, jc, kc)


def restrict_coarse_from_fine(pr_c, pr_f, info, ng):
    refine = info["refine"]
    i0, i1, j0, j1, k0, k1 = info["box"]
    nx_f, ny_f, nz_f = info["fine_shape"]
    for i in range(i0, i1 + 1):
        fi0 = (i - i0) * refine
        for j in range(j0, j1 + 1):
            fj0 = (j - j0) * refine
            for k in range(k0, k1 + 1):
                fk0 = (k - k0) * refine
                block = pr_f[:, ng + fi0:ng + fi0 + refine,
                             ng + fj0:ng + fj0 + refine,
                             ng + fk0:ng + fk0 + refine]
                pr_c[:, i, j, k] = np.mean(block, axis=(1, 2, 3))
