#!/usr/bin/env python3
# core/eos.py
# Lightweight EOS helpers (gamma-law with optional variable gamma tables).
import numpy as np
import numba as nb

SMALL = 1e-12
GAMMA = 5.0 / 3.0
MODE = 0  # 0=gamma, 1=piecewise_gamma, 2=table_gamma
RHO_EDGES = np.array([0.0], dtype=np.float64)
GAMMA_SEG = np.array([GAMMA], dtype=np.float64)
TABLE_RHO = np.array([0.0], dtype=np.float64)
TABLE_GAMMA = np.array([GAMMA], dtype=np.float64)
TABLE_LOG = True


def configure(params):
    global GAMMA, MODE, RHO_EDGES, GAMMA_SEG, TABLE_RHO, TABLE_GAMMA, TABLE_LOG
    GAMMA = float(params.get("GAMMA", GAMMA))
    mode = str(params.get("EOS_MODE", "gamma")).lower()
    if mode in ("piecewise", "piecewise_gamma", "pw_gamma"):
        MODE = 1
    elif mode in ("table", "table_gamma", "tabulated"):
        MODE = 2
    else:
        MODE = 0
    TABLE_LOG = bool(params.get("EOS_TABLE_LOG", True))

    if MODE == 1:
        edges = params.get("EOS_PIECEWISE_RHO", []) or []
        gammas = params.get("EOS_PIECEWISE_GAMMA", []) or []
        try:
            edges = [float(v) for v in edges]
        except Exception:
            edges = []
        try:
            gammas = [float(v) for v in gammas]
        except Exception:
            gammas = []
        if len(edges) == 0 or len(gammas) == 0:
            MODE = 0
            RHO_EDGES = np.array([0.0], dtype=np.float64)
            GAMMA_SEG = np.array([GAMMA], dtype=np.float64)
            return
        edges = sorted(edges)
        if len(gammas) < len(edges) + 1:
            gammas = gammas + [gammas[-1]] * (len(edges) + 1 - len(gammas))
        GAMMA_SEG = np.array(gammas[:len(edges) + 1], dtype=np.float64)
        RHO_EDGES = np.array(edges, dtype=np.float64)
    elif MODE == 2:
        rho = params.get("EOS_GAMMA_TABLE_RHO", []) or []
        gam = params.get("EOS_GAMMA_TABLE_VAL", []) or []
        try:
            rho = [float(v) for v in rho]
            gam = [float(v) for v in gam]
        except Exception:
            rho, gam = [], []
        if len(rho) < 2 or len(gam) < 2:
            MODE = 0
            TABLE_RHO = np.array([0.0], dtype=np.float64)
            TABLE_GAMMA = np.array([GAMMA], dtype=np.float64)
            return
        pairs = sorted(zip(rho, gam), key=lambda x: x[0])
        TABLE_RHO = np.array([p[0] for p in pairs], dtype=np.float64)
        TABLE_GAMMA = np.array([p[1] for p in pairs], dtype=np.float64)


@nb.njit(fastmath=True)
def gamma_eff(rho):
    if MODE == 0:
        return GAMMA
    if MODE == 1:
        idx = np.searchsorted(RHO_EDGES, rho, side="right")
        if idx < 0:
            idx = 0
        if idx >= GAMMA_SEG.shape[0]:
            idx = GAMMA_SEG.shape[0] - 1
        return GAMMA_SEG[idx]
    if MODE == 2:
        n = TABLE_RHO.shape[0]
        if n < 2:
            return GAMMA
        if rho <= TABLE_RHO[0]:
            return TABLE_GAMMA[0]
        if rho >= TABLE_RHO[n-1]:
            return TABLE_GAMMA[n-1]
        if TABLE_LOG:
            lr = np.log10(max(rho, SMALL))
            l0 = np.log10(max(TABLE_RHO[0], SMALL))
            ln = np.log10(max(TABLE_RHO[n-1], SMALL))
            t = (lr - l0) / max(ln - l0, SMALL)
            idx = int(t * (n - 1))
            if idx < 0:
                idx = 0
            if idx >= n - 1:
                idx = n - 2
            r0 = np.log10(max(TABLE_RHO[idx], SMALL))
            r1 = np.log10(max(TABLE_RHO[idx + 1], SMALL))
            f = (lr - r0) / max(r1 - r0, SMALL)
        else:
            idx = 0
            for i in range(n - 1):
                if rho >= TABLE_RHO[i] and rho <= TABLE_RHO[i + 1]:
                    idx = i
                    break
            r0 = TABLE_RHO[idx]
            r1 = TABLE_RHO[idx + 1]
            f = (rho - r0) / max(r1 - r0, SMALL)
        g0 = TABLE_GAMMA[idx]
        g1 = TABLE_GAMMA[idx + 1]
        return g0 + f * (g1 - g0)
    return GAMMA


@nb.njit(fastmath=True)
def enthalpy(rho, p):
    gamma = gamma_eff(rho)
    return 1.0 + gamma / (gamma - 1.0) * p / max(rho, SMALL)


@nb.njit(fastmath=True)
def sound_speed(rho, p):
    gamma = gamma_eff(rho)
    h = enthalpy(rho, p)
    w = rho * h
    cs2 = gamma * p / max(w, SMALL)
    if cs2 < 0.0:
        cs2 = 0.0
    if cs2 > 1.0 - 1e-14:
        cs2 = 1.0 - 1e-14
    return np.sqrt(cs2)
