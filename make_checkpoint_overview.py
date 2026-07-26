#!/usr/bin/env python
"""
Cross-checkpoint overview built from the per-checkpoint outputs of
`make_composite_roc.py`.

Reads every `<iter>_ROCs/composite_ROC_iter<iter>_mean_curve.csv` and
`..._per_model.csv` already written into this folder and produces:

  composite_ROC_all_checkpoints.png / .pdf   mean ROC of the 10-model ensemble
                                             at each checkpoint, coloured by
                                             training iteration on a single
                                             light-blue -> dark-green gradient,
                                             each with its own +/- 1 SD band
  composite_ROC_checkpoint_grid.png / .pdf   the same curves as small multiples,
                                             one panel per checkpoint
  AUC_vs_iteration.png / .pdf                mean +/- SD AUC across the 10
                                             models as a function of iteration
  all_checkpoints_summary.csv                one row per checkpoint (mean, SD,
                                             SEM, min, max of AUC and best F1)

Curves in the two ROC figures are lightly smoothed for legibility (Gaussian
kernel over the FPR grid, followed by a running maximum so the curve stays
monotonic and still starts at (0,0) and ends at (1,1)). Smoothing is cosmetic
only: all AUC/F1 numbers are computed from the raw, unsmoothed data.

Usage:  python make_checkpoint_overview.py
"""

import csv
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
PUB_DPI = 600
COL_MEAN = "#08519c"
COL_BAND = "#4292c6"

# Single perceptual gradient for the training axis: light blue -> dark green.
CKPT_CMAP = LinearSegmentedColormap.from_list(
    "blue_green",
    ["#a8dced", "#5cb8d6", "#3d9bb5", "#3aa17e", "#2e8b57", "#1b6b3a", "#0b3d20"],
)

# Gaussian smoothing width, in FPR-grid points (grid = 501 points over 0-1),
# so 5 points ~ 0.01 FPR. Purely cosmetic.
SMOOTH_SIGMA_PTS = 5.0


def smooth(y, sigma=SMOOTH_SIGMA_PTS, monotonic=False):
    """Gaussian-smooth a curve sampled on the FPR grid.

    Edges are handled by edge-padding so the endpoints are not dragged inward.
    With monotonic=True a running maximum is applied afterwards, which keeps the
    smoothed curve a valid (non-decreasing) ROC curve.
    """
    if sigma <= 0:
        return np.asarray(y, float)
    y = np.asarray(y, float)
    rad = int(np.ceil(3 * sigma))
    x = np.arange(-rad, rad + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = np.convolve(np.pad(y, rad, mode="edge"), k, mode="valid")
    if monotonic:
        out = np.maximum.accumulate(out)
    out = np.clip(out, 0.0, 1.0)
    out[0], out[-1] = y[0], y[-1]      # keep (0,0) and (1,1) exactly
    return out


def lighten(color, frac=0.62):
    """Blend a colour towards white — used for each curve's own variance band."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def set_publication_style():
    plt.rcParams.update({
        "savefig.dpi": PUB_DPI, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05, "font.family": "DejaVu Sans",
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 15,
        "axes.linewidth": 1.2, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 11, "lines.linewidth": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def discover():
    """Return sorted iterations that have a completed per-checkpoint folder."""
    its = []
    for name in os.listdir(HERE):
        m = re.fullmatch(r"(\d+)_ROCs", name)
        if not m:
            continue
        it = int(m.group(1))
        if os.path.isfile(os.path.join(HERE, name,
                                       f"composite_ROC_iter{it}_mean_curve.csv")):
            its.append(it)
    return sorted(its)


def load(it):
    d = os.path.join(HERE, f"{it}_ROCs")
    fpr, tpr, sd = [], [], []
    with open(os.path.join(d, f"composite_ROC_iter{it}_mean_curve.csv")) as fh:
        for row in csv.DictReader(fh):
            fpr.append(float(row["FPR"]))
            tpr.append(float(row["TPR_mean"]))
            sd.append(float(row["TPR_sd"]))
    aucs, f1s = [], []
    with open(os.path.join(d, f"composite_ROC_iter{it}_per_model.csv")) as fh:
        for row in csv.DictReader(fh):
            aucs.append(float(row["AUC"]))
            f1s.append(float(row["Best_F1"]))
    return (np.array(fpr), np.array(tpr), np.array(sd),
            np.array(aucs), np.array(f1s))


def main():
    global HERE
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=HERE,
                    help="folder holding the <iter>_ROCs subdirs and receiving the "
                         "overview figures (default: this script's folder)")
    HERE = os.path.abspath(ap.parse_args().dir)
    set_publication_style()
    its = discover()
    if not its:
        raise SystemExit("No <iter>_ROCs folders with results found.")
    data = {it: load(it) for it in its}

    norm = Normalize(vmin=min(its), vmax=max(its))
    cmap = CKPT_CMAP
    colors = {it: cmap(norm(it)) for it in its}

    # Smoothed mean and +/- 1 SD envelope per checkpoint (display only).
    smoothed = {}
    for it in its:
        fpr, tpr, sd, _, _ = data[it]
        m = smooth(tpr, monotonic=True)
        s = smooth(sd)
        smoothed[it] = (fpr, m, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1))

    # ---------------- figure 1: mean ROC per checkpoint -------------------- #
    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    # All variance bands first, all mean curves on top: otherwise a late
    # checkpoint's band paints over an earlier checkpoint's curve.
    for i, it in enumerate(its):
        fpr, _, lo, hi = smoothed[it]
        # Each checkpoint gets its own band in a lighter shade of its own colour.
        ax.fill_between(fpr, lo, hi, color=lighten(colors[it]), alpha=0.45,
                        lw=0, zorder=2 + i)
    for i, it in enumerate(its):
        fpr, m, _, _ = smoothed[it]
        aucs = data[it][3]
        ax.plot(fpr, m, color=colors[it], lw=2.2, zorder=40 + i,
                solid_capstyle="round",
                label=f"{it} (AUC {aucs.mean():.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, ls="--", zorder=1)
    ax.set_xlim(0, 1)
    # Exactly [0, 1]: headroom above 1.0 lets the line stroke render above the
    # y = 1 gridline, which reads as a TPR greater than 1.
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Mean ROC of the 10-model V3 ensemble\nacross training checkpoints",
                 fontsize=15, pad=10)
    ax.grid(True, ls="--", lw=0.5, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Training iteration", fontsize=14)
    cbar.set_ticks(its)
    cbar.ax.tick_params(labelsize=10)
    leg = ax.legend(loc="lower right", fontsize=8.5, ncol=2, framealpha=0.96,
                    edgecolor="0.8", title="iteration (mean AUC)",
                    title_fontsize=9, borderpad=0.6, labelspacing=0.4)
    leg.set_zorder(60)
    ax.text(0.985, 0.30, "shading: $\\pm$ 1 SD across the 10 models",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color="0.35", zorder=60)
    fig.savefig(os.path.join(HERE, "composite_ROC_all_checkpoints.png"),
                dpi=PUB_DPI)
    fig.savefig(os.path.join(HERE, "composite_ROC_all_checkpoints.pdf"))
    plt.close(fig)

    # ---------------- figure 1b: small multiples --------------------------- #
    ncol = 4
    nrow = int(np.ceil(len(its) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.15 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for axi, it in zip(axes, its):
        fpr, m, lo, hi = smoothed[it]
        aucs = data[it][3]
        axi.fill_between(fpr, lo, hi, color=lighten(colors[it], 0.55),
                         alpha=0.85, lw=0, zorder=2)
        axi.plot(fpr, m, color=colors[it], lw=2.0, zorder=3)
        axi.plot([0, 1], [0, 1], color="gray", lw=1.0, ls="--", zorder=1)
        axi.set_title(f"iteration {it}", fontsize=12, pad=6)
        axi.text(0.96, 0.06,
                 f"AUC {aucs.mean():.3f} $\\pm$ {aucs.std(ddof=1):.3f}",
                 transform=axi.transAxes, ha="right", va="bottom", fontsize=9.5)
        axi.set_xlim(0, 1)
        axi.set_ylim(0, 1.0)
        axi.set_xticks([0, 0.5, 1.0])
        axi.set_yticks([0, 0.5, 1.0])
        axi.set_aspect("equal", adjustable="box")
        axi.grid(True, ls="--", lw=0.5, alpha=0.5)
        for side in ("top", "right"):
            axi.spines[side].set_visible(False)
    for axi in axes[len(its):]:
        axi.set_visible(False)
    fig.supxlabel("False positive rate", fontsize=14)
    fig.supylabel("True positive rate", fontsize=14)
    fig.suptitle("Mean ROC $\\pm$ 1 SD of the 10-model V3 ensemble, per checkpoint",
                 fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "composite_ROC_checkpoint_grid.png"),
                dpi=PUB_DPI)
    fig.savefig(os.path.join(HERE, "composite_ROC_checkpoint_grid.pdf"))
    plt.close(fig)

    # ---------------- figure 2: AUC vs iteration --------------------------- #
    mean = np.array([data[it][3].mean() for it in its])
    sd = np.array([data[it][3].std(ddof=1) for it in its])
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.fill_between(its, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1),
                    color=COL_BAND, alpha=0.30, lw=0)
    ax.plot(its, mean, "o-", color=COL_MEAN, lw=2.5, ms=7,
            markeredgecolor="white", markeredgewidth=1.0)
    for it in its:
        ax.scatter([it] * len(data[it][3]), data[it][3], s=14, color=COL_MEAN,
                   alpha=0.35, zorder=2, linewidths=0)
    ax.axhline(0.5, color="gray", lw=1.5, ls="--")
    ax.text(max(its), 0.505, "chance", ha="right", va="bottom",
            fontsize=11, color="gray")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("AUC")
    ax.set_title("Ensemble AUC across training (mean $\\pm$ 1 SD, n = 10 models)",
                 fontsize=14, pad=10)
    ax.set_xticks(its)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(os.path.join(HERE, "AUC_vs_iteration.png"), dpi=PUB_DPI)
    fig.savefig(os.path.join(HERE, "AUC_vs_iteration.pdf"))
    plt.close(fig)

    # ---------------- combined summary table ------------------------------- #
    out = os.path.join(HERE, "all_checkpoints_summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iteration", "n_models", "AUC_mean", "AUC_sd", "AUC_sem",
                    "AUC_min", "AUC_max", "BestF1_mean", "BestF1_sd",
                    "BestF1_min", "BestF1_max"])
        for it in its:
            a, f = data[it][3], data[it][4]
            w.writerow([it, len(a), f"{a.mean():.4f}", f"{a.std(ddof=1):.4f}",
                        f"{a.std(ddof=1)/np.sqrt(len(a)):.4f}",
                        f"{a.min():.4f}", f"{a.max():.4f}",
                        f"{f.mean():.4f}", f"{f.std(ddof=1):.4f}",
                        f"{f.min():.4f}", f"{f.max():.4f}"])
    print(f"checkpoints: {its}")
    print(f"wrote overview figures and {os.path.basename(out)} to {HERE}")


if __name__ == "__main__":
    main()
