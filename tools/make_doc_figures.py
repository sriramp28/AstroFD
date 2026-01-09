#!/usr/bin/env python3
import argparse
import csv
import glob
import os

import numpy as np
import matplotlib.pyplot as plt


def latest_run_dir(base="results"):
    runs = sorted([d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)])
    if not runs:
        raise SystemExit("No results/ runs found.")
    return runs[-1]


def _latest_npz_in_dir(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "jet3d_rank*_step*.npz")))
    if files:
        return files[-1]
    sub_files = sorted(glob.glob(os.path.join(run_dir, "*", "jet3d_rank*_step*.npz")))
    if sub_files:
        return sub_files[-1]
    return None


def _load_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            rows.append(row)
    return header, rows


def _last_monotonic_segment(times):
    if len(times) == 0:
        return slice(0, 0)
    start = 0
    for i in range(1, len(times)):
        if times[i] < times[i - 1]:
            start = i
    return slice(start, None)


def plot_jet_slice(npz_path, out_dir):
    blk = np.load(npz_path)
    rho = blk["rho"]
    vx = blk["vx"]
    vy = blk["vy"]
    vz = blk["vz"]

    k = rho.shape[2] // 2
    slc = rho[:, :, k].T
    plt.figure()
    plt.imshow(slc, origin="lower", aspect="auto")
    plt.colorbar(label="rho")
    plt.title("rho (mid-Z)")
    out_png = os.path.join(out_dir, "jet_rho_midZ.png")
    plt.savefig(out_png, dpi=200)
    plt.close()

    j = rho.shape[1] // 2
    v2 = vx[:, j, k] ** 2 + vy[:, j, k] ** 2 + vz[:, j, k] ** 2
    v2 = np.clip(v2, 0.0, 1.0 - 1e-12)
    gamma = 1.0 / np.sqrt(1.0 - v2)
    plt.figure()
    plt.plot(gamma)
    plt.xlabel("i (local)")
    plt.ylabel("Gamma (centerline)")
    plt.title("Centerline Lorentz factor")
    out_png2 = os.path.join(out_dir, "centerline_gamma.png")
    plt.savefig(out_png2, dpi=200)
    plt.close()

    return out_png, out_png2


def plot_divb(run_dir, out_dir):
    path = os.path.join(run_dir, "divb.csv")
    if not os.path.exists(path):
        return None
    header, rows = _load_csv(path)
    if not rows:
        return None
    time = np.array([float(r[1]) for r in rows])
    divb_max = np.array([float(r[2]) for r in rows])
    seg = _last_monotonic_segment(time)
    time = time[seg]
    divb_max = divb_max[seg]

    plt.figure()
    plt.plot(time, divb_max)
    plt.xlabel("time")
    plt.ylabel("divB max")
    plt.title("RMHD divB max")
    out_png = os.path.join(out_dir, "divb_max.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png


def plot_sn_shock(run_dir, out_dir):
    path = os.path.join(run_dir, "sn_diagnostics.csv")
    if not os.path.exists(path):
        return None
    header, rows = _load_csv(path)
    if not rows:
        return None
    time = np.array([float(r[1]) for r in rows])
    shock = np.array([float(r[2]) for r in rows])
    seg = _last_monotonic_segment(time)
    time = time[seg]
    shock = shock[seg]

    plt.figure()
    plt.plot(time, shock)
    plt.xlabel("time")
    plt.ylabel("shock radius")
    plt.title("SN-lite shock radius")
    out_png = os.path.join(out_dir, "sn_shock_radius.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png


def write_summary_table(run_dir, out_dir):
    diag_path = os.path.join(run_dir, "diagnostics.csv")
    divb_path = os.path.join(run_dir, "divb.csv")
    sn_path = os.path.join(run_dir, "sn_diagnostics.csv")

    max_gamma = np.nan
    amax = np.nan
    inlet_abs = np.nan
    if os.path.exists(diag_path):
        _, rows = _load_csv(diag_path)
        if rows:
            time = np.array([float(r[1]) for r in rows])
            seg = _last_monotonic_segment(time)
            max_gamma = np.max(np.array([float(r[4]) for r in rows])[seg])
            amax = np.max(np.array([float(r[3]) for r in rows])[seg])
            inlet_abs = np.mean(np.array([float(r[6]) for r in rows])[seg])

    divb_max = np.nan
    divb_rms = np.nan
    if os.path.exists(divb_path):
        _, rows = _load_csv(divb_path)
        if rows:
            time = np.array([float(r[1]) for r in rows])
            seg = _last_monotonic_segment(time)
            divb_max = np.max(np.array([float(r[2]) for r in rows])[seg])
            divb_rms = np.mean(np.array([float(r[3]) for r in rows])[seg])

    shock_max = np.nan
    heat_eff = np.nan
    if os.path.exists(sn_path):
        _, rows = _load_csv(sn_path)
        if rows:
            time = np.array([float(r[1]) for r in rows])
            seg = _last_monotonic_segment(time)
            shock_max = np.max(np.array([float(r[2]) for r in rows])[seg])
            heat_eff = np.mean(np.array([float(r[6]) for r in rows])[seg])

    def _fmt(val):
        if np.isfinite(val):
            return f"{val:.3e}"
        return "N/A"

    out_path = os.path.join(out_dir, "validation_summary.tex")
    with open(out_path, "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l l}\n")
        f.write("\\hline\n")
        f.write("Metric & Value \\\\ \n")
        f.write("\\hline\n")
        f.write(f"Max Lorentz factor & {_fmt(max_gamma)} \\\\ \n")
        f.write(f"Max signal speed & {_fmt(amax)} \\\\ \n")
        f.write(f"Mean inlet flux (abs) & {_fmt(inlet_abs)} \\\\ \n")
        f.write(f"divB max & {_fmt(divb_max)} \\\\ \n")
        f.write(f"divB rms & {_fmt(divb_rms)} \\\\ \n")
        f.write(f"SN shock radius max & {_fmt(shock_max)} \\\\ \n")
        f.write(f"SN heating efficiency mean & {_fmt(heat_eff)} \\\\ \n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Summary diagnostics from the latest run segment.}\n")
        f.write("\\label{tab:validation-summary}\n")
        f.write("\\end{table}\n")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Generate documentation figures from results.")
    ap.add_argument("--run-dir", default=None, help="results run directory (default: latest)")
    ap.add_argument("--output-dir", default="docs/figures", help="output directory")
    args = ap.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    os.makedirs(args.output_dir, exist_ok=True)

    npz_path = _latest_npz_in_dir(run_dir)
    if not npz_path:
        raise SystemExit(f"No NPZ files found in {run_dir}")

    print(f"[doc-figures] Using run dir: {run_dir}")
    print(f"[doc-figures] Using NPZ: {npz_path}")

    plot_jet_slice(npz_path, args.output_dir)
    plot_divb(run_dir, args.output_dir)
    plot_sn_shock(run_dir, args.output_dir)
    write_summary_table(run_dir, args.output_dir)

    print(f"[doc-figures] Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
