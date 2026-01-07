# utils/settings.py
import argparse, os, json, sys

def _load_json5_or_json(path: str) -> dict:
    try:
        import json5
        with open(path, "r") as f:
            return json5.load(f)
    except Exception as e:
        print(f"[settings] failed to parse {path}: {e}")
        return {}

def load_settings():
    print("[settings] sys.argv =", sys.argv)
    defaults = dict(
        # grid & numerics
        NX=128, NY=96, NZ=96, Lx=1.0, Ly=1.0, Lz=1.0,
        T_END=0.25, OUT_EVERY=50, PRINT_EVERY=10,
        NG=2, CFL=0.35, GAMMA=5.0/3.0,
        # jet physics
        JET_RADIUS=0.12,
        JET_CENTER=None,
        GAMMA_JET=6.0, ETA_RHO=0.05, P_EQ=1.0e-2,
        SHEAR_THICK=0.02,
        NOZZLE_TURB=True, TURB_VAMP=0.02, TURB_PAMP=0.00,
        NOZZLE_PROFILE="top_hat",
        NOZZLE_PERTURB=None,
        RECON="muscl",
        LIMITER="mc",
        RIEMANN="hlle",
        RK_ORDER=2,
        CHECKPOINT_EVERY=0,
        RESTART_PATH=None,
        # ambient
        RHO_AMB=1.0, P_AMB=None,
        VX_AMB=0.0, VY_AMB=0.0, VZ_AMB=0.0,
        P_MAX=1.0, V_MAX=0.999,
        # physics mode (RMHD scaffold)
        PHYSICS="hydro",      # "hydro" | "rmhd"
        GLM_CH=1.0,           # hyperbolic cleaning speed
        GLM_CP=0.1,           # damping coefficient
        B_INIT="none",        # "none" | "poloidal" | "toroidal"
        B0=0.0,               # base magnetic field magnitude
        # GR placeholders
        GR_METRIC="minkowski",   # "minkowski" | "schwarzschild" | "kerr-schild"
        GR_MASS=1.0,
        GR_SPIN=0.0,
        # dissipation placeholders
        DISSIPATION_ENABLED=False,
        DISSIPATION_MODEL="israel_stewart",
        RELAX_TAU=0.1,
        BULK_ZETA=0.0,
        # results
        RESULTS_UNIQUE=False,
        # debug
        DEBUG=False, ASSERTS=False, CHECK_NAN_EVERY=0
    )

    ap = argparse.ArgumentParser(description="3D SRHD jet (MUSCL + HLLE + MPI)")
    ap.add_argument("--config", type=str, help="path to JSON/JSON5 config")
    ap.add_argument("--debug", action="store_true", help="verbose logging + JIT warm-up")
    # quick overrides
    ap.add_argument("--nx", type=int); ap.add_argument("--ny", type=int); ap.add_argument("--nz", type=int)
    ap.add_argument("--t-end", type=float); ap.add_argument("--out-every", type=int)
    ap.add_argument("--print-every", type=int)
    args = ap.parse_args()

    cfg = {}
    print("[settings] args.config =", args.config)
    print("[settings] exists? =", os.path.exists(args.config))
    print("[settings] cwd =", os.getcwd())
    if args.config and os.path.exists(args.config):
        cfg = _load_json5_or_json(args.config)
        print("[settings] loaded config dict =", cfg)
        # --- debug: show what was read from the config file ---
        print("[settings] loaded keys:", list(cfg.keys()))
        if "PHYSICS" in cfg:
            print(f"[settings] PHYSICS in file = {cfg['PHYSICS']}")
        else:
            print("[settings] no PHYSICS key found in file")

    s = {**defaults, **cfg}
    if args.debug: s["DEBUG"] = True
    if args.nx is not None: s["NX"] = args.nx
    if args.ny is not None: s["NY"] = args.ny
    if args.nz is not None: s["NZ"] = args.nz
    if args.t_end is not None: s["T_END"] = args.t_end
    if args.out_every is not None: s["OUT_EVERY"] = args.out_every
    if args.print_every is not None: s["PRINT_EVERY"] = args.print_every

    # dependent defaults
    if s["P_AMB"] is None:
        s["P_AMB"] = s["P_EQ"]
    if s["JET_CENTER"] is None:
        s["JET_CENTER"] = [0.0, 0.5*s["Ly"], 0.5*s["Lz"]]
    else:
        jc = s["JET_CENTER"]
        if isinstance(jc, dict):
            y = jc.get("y", 0.5*s["Ly"]); z = jc.get("z", 0.5*s["Lz"])
            s["JET_CENTER"] = [0.0, float(y), float(z)]
        else:
            if not (isinstance(jc, (list, tuple)) and len(jc) == 3):
                raise ValueError("JET_CENTER must be [x,y,z] or {'y':..., 'z':...}")
            s["JET_CENTER"] = [float(jc[0]), float(jc[1]), float(jc[2])]

    # light validation
    if s["NG"] < 2:
        raise ValueError("NG must be >= 2 for MUSCL (uses i±2, j±2, k±2).")
    if not (0.0 < s["CFL"] <= 0.9):
        raise ValueError("CFL should be in (0, 0.9].")
    if min(s["NX"], s["NY"], s["NZ"]) < 8:
        raise ValueError("NX, NY, NZ must be ≥ 8.")
    if s.get("DISSIPATION_ENABLED", False):
        if s.get("PHYSICS") in ("rmhd", "grmhd"):
            raise ValueError("DISSIPATION_ENABLED is supported only for hydro/grhd.")
        if s.get("RECON") != "muscl":
            raise ValueError("DISSIPATION_ENABLED currently requires RECON='muscl'.")

    # normalize nozzle perturbations: keep legacy NOZZLE_TURB behavior
    if s.get("NOZZLE_PERTURB") is None:
        s["NOZZLE_PERTURB"] = "random" if s.get("NOZZLE_TURB", False) else "none"
    if isinstance(s.get("NOZZLE_PROFILE"), str):
        s["NOZZLE_PROFILE"] = s["NOZZLE_PROFILE"].lower()
    if isinstance(s.get("NOZZLE_PERTURB"), str):
        s["NOZZLE_PERTURB"] = s["NOZZLE_PERTURB"].lower()
    if isinstance(s.get("LIMITER"), str):
        s["LIMITER"] = s["LIMITER"].lower()
    if isinstance(s.get("RECON"), str):
        s["RECON"] = s["RECON"].lower()
    if isinstance(s.get("RIEMANN"), str):
        s["RIEMANN"] = s["RIEMANN"].lower()

    return s
