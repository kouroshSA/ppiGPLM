#!/usr/bin/env python3
"""Composite (N-model ensemble) LES trajectory plots — mean +/- SD.

For each metric (ROC-AUC and Best-F1) this reads the per-checkpoint value from
every per-model LES folder under --parent, then draws a single publication-quality
trajectory of the ACROSS-MODEL MEAN at each training checkpoint, with a shaded band
spanning +/-1 standard deviation (population SD, ddof=0).

The per-model Learning Efficiency Score (LES) is the area under that model's own
metric-vs-iteration curve (iterations min-max normalized to [0,1], trapezoidal rule),
exactly as computed by LES-wrapper_v2.py. The figure title reports the ensemble LES
as mean +/- SD across the models.

Each per-model folder is `--folder-template` with `{k}` substituted, and must contain
`summary_table.csv` (columns: checkpoint,iteration,AUC,Best_F1,...). Works for any LES
layout — the native V3 ensemble (`LES_results_V3-{k}_6k`) or, e.g., the homodimer-removed
run (`LES_V3-{k}`).

Examples:
  # native 10-model ensemble (original layout)
  python make_ensemble_les_trajectories.py \
      --parent /home/ksa/Dropbox/LES_and_V3_Datasets/LES_results \
      --folder-template "LES_results_V3-{k}_6k" \
      --out    /home/ksa/Dropbox/LES_and_V3_Datasets/LES_results/Enselmble_10_V3_models

  # homodimer-removed run
  python make_ensemble_les_trajectories.py \
      --parent /home/ksa/Dropbox/LES_and_V3_Datasets/LES_results_V3_no_homodimers \
      --folder-template "LES_V3-{k}" --out <parent>/composite
"""
import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB_DPI = 600
AUC_C = "#1f4e9c"   # blue  (matches per-model LES-wrapper_v2 style)
F1_C = "#2c8a3d"    # green

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.titlesize': 16,
    'axes.labelsize': 15, 'axes.linewidth': 1.2, 'xtick.labelsize': 12,
    'ytick.labelsize': 12, 'legend.fontsize': 12, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def read_summary(parent, template, k):
    """Return (iterations, AUC[], Best_F1[]) for model k from its summary_table.csv."""
    path = os.path.join(parent, template.format(k=k), "summary_table.csv")
    its, auc, f1 = [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if not row["checkpoint"].startswith("ckpt_"):
                continue
            its.append(int(row["iteration"]))
            auc.append(float(row["AUC"]))
            f1.append(float(row["Best_F1"]))
    return np.array(its), np.array(auc), np.array(f1)


def compute_les(iterations, values):
    """Area under the metric-vs-iteration curve; iterations min-max normalized."""
    it = np.asarray(iterations, float)
    v = np.asarray(values, float)
    itn = (it - it.min()) / (it.max() - it.min())
    trap = getattr(np, "trapezoid", np.trapz)
    return trap(v, itn)


def _annotate_nodes(ax, xs, ys):
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color="0.15")


def make_trajectory(iters_ref, mat, color, metric_name, les_vals, base, n_models,
                    floor=False):
    mean = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    if floor:
        ax.fill_between(iters_ref, 0, mean, color=color, alpha=0.18,
                        label="Area under curve (LES)")
        ax.errorbar(iters_ref, mean, yerr=sd, fmt="o-", color=color, lw=2.5,
                    markersize=7, capsize=4, elinewidth=1.4, ecolor=color,
                    label=f"Ensemble mean +/-1 SD (n = {n_models})")
    else:
        lo = np.clip(mean - sd, 0, 1)
        hi = np.clip(mean + sd, 0, 1)
        ax.fill_between(iters_ref, lo, hi, color=color, alpha=0.20,
                        label=f"+/-1 SD (n = {n_models} models)")
        ax.plot(iters_ref, mean, "o-", color=color, lw=2.5, markersize=7,
                label="Ensemble mean")
    _annotate_nodes(ax, iters_ref, mean)
    ax.set_xlabel("Training iteration", fontsize=15)
    ax.set_ylabel(metric_name, fontsize=15)
    ax.set_ylim(0, 1)
    ax.set_xlim(iters_ref.min() - 150, iters_ref.max() + 150)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title(f"{metric_name} trajectory — {n_models}-model ensemble\n"
                 f"LES-{metric_name} = {les_vals.mean():.4f} +/- "
                 f"{les_vals.std(ddof=0):.4f} (mean +/- SD, n = {n_models})",
                 fontsize=15, pad=12)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=PUB_DPI)
    plt.close(fig)
    print(f"  wrote {base}.png/.pdf   LES = {les_vals.mean():.4f} +/- "
          f"{les_vals.std(ddof=0):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", required=True,
                    help="directory containing the per-model LES folders")
    ap.add_argument("--folder-template", default="LES_results_V3-{k}_6k",
                    help="per-model folder name with {k} placeholder")
    ap.add_argument("--models", type=int, default=10, help="number of models (1..N)")
    ap.add_argument("--out", default=None, help="output dir (default: <parent>/composite)")
    ap.add_argument("--prefix", default="composite", help="output filename prefix")
    a = ap.parse_args()
    out = a.out or os.path.join(a.parent, "composite")
    os.makedirs(out, exist_ok=True)

    iters_ref, auc_mat, f1_mat, les_auc, les_f1 = None, [], [], [], []
    for k in range(1, a.models + 1):
        its, auc, f1 = read_summary(a.parent, a.folder_template, k)
        if iters_ref is None:
            iters_ref = its
        assert np.array_equal(its, iters_ref), f"iteration grid mismatch in model {k}"
        auc_mat.append(auc); f1_mat.append(f1)
        les_auc.append(compute_les(its, auc)); les_f1.append(compute_les(its, f1))
    auc_mat = np.vstack(auc_mat); f1_mat = np.vstack(f1_mat)
    les_auc = np.array(les_auc); les_f1 = np.array(les_f1)
    n = a.models

    make_trajectory(iters_ref, auc_mat, AUC_C, "AUC", les_auc,
                    os.path.join(out, f"{a.prefix}_trajectory_AUC"), n)
    make_trajectory(iters_ref, f1_mat, F1_C, "Best F1", les_f1,
                    os.path.join(out, f"{a.prefix}_trajectory_F1"), n)
    make_trajectory(iters_ref, auc_mat, AUC_C, "AUC", les_auc,
                    os.path.join(out, f"{a.prefix}_trajectory_AUC_area"), n, floor=True)
    make_trajectory(iters_ref, f1_mat, F1_C, "Best F1", les_f1,
                    os.path.join(out, f"{a.prefix}_trajectory_F1_area"), n, floor=True)

    with open(os.path.join(out, f"{a.prefix}_trajectory_data.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iteration", "AUC_mean", "AUC_SD", "F1_mean", "F1_SD"])
        for i, it in enumerate(iters_ref):
            w.writerow([it,
                        f"{auc_mat[:, i].mean():.6f}", f"{auc_mat[:, i].std(ddof=0):.6f}",
                        f"{f1_mat[:, i].mean():.6f}", f"{f1_mat[:, i].std(ddof=0):.6f}"])
        w.writerow([])
        w.writerow(["LES_AUC_mean", f"{les_auc.mean():.6f}",
                    "LES_AUC_SD", f"{les_auc.std(ddof=0):.6f}"])
        w.writerow(["LES_F1_mean", f"{les_f1.mean():.6f}",
                    "LES_F1_SD", f"{les_f1.std(ddof=0):.6f}"])
    print("DONE ->", out)


if __name__ == "__main__":
    main()
