

!/usr/bin/env python3
"""
plots/generate_plots.py
========================
Reads strong_scaling.csv and weak_scaling.csv from output/ and produces:
  1. Speedup plot (strong scaling) vs ideal
  2. Parallel efficiency plot (strong scaling)
  3. Weak scaling efficiency plot
  4. Phase breakdown bar chart (stacked timing per config)
  5. Roofline-style OpenMP thread scaling

Usage:
    python3 plots/generate_plots.py [--output-dir output/figs]
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.alpha":       0.35,
})
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

def load(path):
    if not os.path.exists(path):
        print(f"[WARN] {path} not found – skipping related plots.")
        return None
    return pd.read_csv(path)

# ── 1 & 2: Strong scaling – speedup + efficiency ─────────────────────────────
def plot_strong_scaling(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for omp_t, grp in df.groupby("omp_threads"):
        grp = grp.sort_values("mpi_procs")
        p   = grp["total_procs"].values.astype(float)
        sp  = grp["speedup"].values.astype(float)
        eff = grp["efficiency"].values.astype(float)
        lbl = f"OMP={omp_t}"
        axes[0].plot(p, sp,  "o-", label=lbl)
        axes[1].plot(p, eff, "s-", label=lbl)

    # Ideal lines
    p_range = np.array(sorted(df["total_procs"].unique()))
    axes[0].plot(p_range, p_range, "k--", linewidth=1, label="Ideal")
    axes[1].axhline(1.0, color="k", linestyle="--", linewidth=1, label="Ideal (E=1)")

    axes[0].set_xlabel("Total processes (MPI × OMP)")
    axes[0].set_ylabel("Speedup  $S(p) = T_1 / T_p$")
    axes[0].set_title("Strong Scaling – Speedup")
    axes[0].legend()

    axes[1].set_xlabel("Total processes (MPI × OMP)")
    axes[1].set_ylabel("Parallel Efficiency  $E(p) = S(p)/p$")
    axes[1].set_title("Strong Scaling – Efficiency")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(out_dir, "strong_scaling.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)

# ── 3: Weak scaling efficiency ───────────────────────────────────────────────
def plot_weak_scaling(df, out_dir):
    # Weak scaling efficiency = T_1 / T_p (ideal: constant)
    baseline = df[(df.mpi_procs == 1) & (df.omp_threads == 1)]["t_total"].values
    if len(baseline) == 0:
        print("[WARN] No baseline (1 proc) in weak scaling data.")
        return
    T1 = baseline[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    for omp_t, grp in df.groupby("omp_threads"):
        grp  = grp.sort_values("mpi_procs")
        p    = grp["total_procs"].values.astype(float)
        eff  = T1 / grp["t_total"].values.astype(float)
        ax.plot(p, eff, "o-", label=f"OMP={omp_t}")

    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Ideal")
    ax.set_xlabel("Total processes (MPI × OMP)")
    ax.set_ylabel("Weak Scaling Efficiency  $T_1 / T_p$")
    ax.set_title("Weak Scaling – Iso-efficiency")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "weak_scaling.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)

# ── 4: Phase breakdown stacked bar ───────────────────────────────────────────
def plot_phase_breakdown(df, out_dir):
    phases = ["t_load", "t_degree", "t_laplacian", "t_eigen", "t_kmeans"]
    labels = ["Load/Similarity", "Degree", "Laplacian", "Eigen", "K-means"]

    # Pick OMP=1 rows for clarity
    sub = df[df.omp_threads == 1].sort_values("mpi_procs").copy()
    if sub.empty: sub = df.sort_values("total_procs")

    x    = np.arange(len(sub))
    xlbl = [f"MPI={r.mpi_procs}\nOMP={r.omp_threads}" for _, r in sub.iterrows()]

    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(sub))
    for ph, lbl, col in zip(phases, labels, COLORS):
        vals = sub[ph].values.astype(float)
        ax.bar(x, vals, bottom=bottom, label=lbl, color=col, edgecolor="white", width=0.6)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(xlbl, fontsize=8)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Phase-level Timing Breakdown (Strong Scaling, OMP=1)")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    path = os.path.join(out_dir, "phase_breakdown.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)

# ── 5: OMP thread scaling at fixed MPI ───────────────────────────────────────
def plot_omp_scaling(df, out_dir):
    """For a fixed MPI count (largest available), show effect of OMP threads."""
    max_mpi = df["mpi_procs"].max()
    sub = df[df.mpi_procs == max_mpi].sort_values("omp_threads")
    if sub.empty: return

    T1 = sub[sub.omp_threads == 1]["t_total"].values
    if len(T1) == 0: return
    T1 = T1[0]

    fig, ax = plt.subplots(figsize=(5, 4))
    t   = sub["omp_threads"].values.astype(float)
    sp  = T1 / sub["t_total"].values.astype(float)
    ax.plot(t, sp, "o-", color=COLORS[0], label=f"MPI={max_mpi}")
    ax.plot(t, t,  "k--", linewidth=1, label="Ideal")
    ax.set_xlabel("OMP threads per rank")
    ax.set_ylabel("Speedup relative to OMP=1")
    ax.set_title(f"OpenMP Thread Scaling (MPI={max_mpi})")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "omp_thread_scaling.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/figs")
    parser.add_argument("--strong-csv", default="output/strong_scaling.csv")
    parser.add_argument("--weak-csv",   default="output/weak_scaling.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    strong = load(args.strong_csv)
    weak   = load(args.weak_csv)

    if strong is not None:
        plot_strong_scaling(strong, args.output_dir)
        plot_phase_breakdown(strong, args.output_dir)
        plot_omp_scaling(strong, args.output_dir)

    if weak is not None:
        plot_weak_scaling(weak, args.output_dir)

    print("Done.")

if __name__ == "__main__":
    main()