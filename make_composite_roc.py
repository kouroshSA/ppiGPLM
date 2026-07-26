#!/usr/bin/env python
"""
Composite (ensemble) ROC curve for the 10 native V3 models at a single checkpoint.

Default checkpoint: iteration 5000.

For each of the 10 independently trained V3 models
(`LES_results_V3-{1..10}_6k`) this script re-reads the raw per-pair
P(interaction) values produced by `LES-wrapper_v2.py`
(`ckpt_<iter>/combined_probabilities_iter<iter>.csv`, column 0 = PRS / positive,
column 1 = RRS / negative), recomputes the ROC curve exactly as the wrapper
does (PRS -> label 1, RRS -> label 0, score = Probability_of_1), and then
aggregates the 10 curves by **vertical averaging**: TPR is linearly
interpolated onto a common FPR grid and the mean +/- 1 SD across models is
plotted as a line with a shaded band.

Outputs (written next to this script):
  composite_ROC_iter<iter>.png / .pdf            main publication figure
  composite_ROC_iter<iter>_panel.png / .pdf      two-panel version: ROC (+ pooled
                                                 curve) and per-model AUC spread
  composite_ROC_iter<iter>_mean_curve.csv        mean/SD TPR on the FPR grid
  composite_ROC_iter<iter>_per_model.csv         per-model AUC / best-F1
  composite_ROC_iter<iter>_stats.txt             summary statistics

Excluded on purpose: the `_ps1_random`, `_ps2_random`, `_ps1-ps2_random`
shuffled-sequence controls and the rejected `V3-3_6k_lr1e6` learning-rate
variant.

Usage:
    python make_composite_roc.py                 # iteration 5000
    python make_composite_roc.py --iter 6000     # any other checkpoint
"""

import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc, f1_score

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
LES_ROOT = "/home/ksa/Dropbox/LES_and_V3_Datasets/LES_results"
MODELS = [f"V3-{i}" for i in range(1, 11)]          # the 10 native models
FOLDER_TMPL = "LES_results_{model}_6k"

# Publication defaults — kept consistent with LES-wrapper_v2.py
PUB_DPI = 600
COL_MEAN = "#08519c"      # mean ROC (same blue as the wrapper's ROC curves)
COL_BAND = "#4292c6"      # +/- 1 SD band
COL_INDIV = "#9ecae1"     # individual model curves
COL_POOL = "#b2182b"      # pooled-data ROC (RRS red from the wrapper palette)
FPR_GRID = np.linspace(0.0, 1.0, 501)


def set_publication_style():
    plt.rcParams.update({
        "savefig.dpi": PUB_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "axes.linewidth": 1.2,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "lines.linewidth": 2.5,
        "pdf.fonttype": 42,     # embed as TrueType -> editable text in Illustrator
        "ps.fonttype": 42,
    })


# --------------------------------------------------------------------------- #
# Data loading / ROC computation
# --------------------------------------------------------------------------- #
def read_combined_csv(path):
    """Read a headerless combined_probabilities CSV.

    Column 0 = PRS (positive) P(interaction), column 1 = RRS (negative).
    Blank cells are allowed (unequal PRS/RRS counts), matching
    LES-wrapper_v2.run_roc_analysis_internal().
    """
    prs, rrs = [], []
    with open(path, "r") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            a, b = row[0].strip(), row[1].strip()
            if a:
                prs.append(float(a))
            if b:
                rrs.append(float(b))
    return np.asarray(prs), np.asarray(rrs)


def roc_from_probs(prs, rrs):
    """ROC curve, AUC and best F1 for one model (wrapper-identical logic)."""
    probs = np.concatenate([prs, rrs])
    labels = np.concatenate([np.ones(len(prs), int), np.zeros(len(rrs), int)])

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Best F1 scanned over every unique score (decision rule: prob >= t -> 1),
    # identical to LES-wrapper_v2.py.
    best_f1 = -1.0
    for thresh in np.unique(probs):
        f1 = f1_score(labels, (probs >= thresh).astype(int), zero_division=0)
        if f1 >= best_f1:
            best_f1 = f1
    return fpr, tpr, roc_auc, best_f1


def interp_tpr(fpr, tpr, grid=FPR_GRID):
    """Vertical averaging helper: TPR on a common FPR grid, forced through (0,0)."""
    out = np.interp(grid, fpr, tpr)
    out[0] = 0.0
    return out


def smooth(y, sigma, monotonic=False):
    """Gaussian-smooth a curve sampled on the FPR grid (display only).

    Edges are edge-padded so the endpoints are not dragged inward. With
    monotonic=True a running maximum is applied afterwards, which keeps the
    smoothed curve a valid (non-decreasing) ROC curve. sigma is in grid points;
    sigma <= 0 returns the curve unchanged.
    """
    y = np.asarray(y, float)
    if sigma is None or sigma <= 0:
        return y
    rad = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.convolve(np.pad(y, rad, mode="edge"), k, mode="valid")
    if monotonic:
        out = np.maximum.accumulate(out)
    out = np.clip(out, 0.0, 1.0)
    out[0], out[-1] = y[0], y[-1]      # keep (0,0) and (1,1) exactly
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def draw_base(ax, curves, mean_tpr, sd_tpr, aucs, iteration, sigma=0.0,
              auc_in_legend=True):
    if sigma > 0:
        curves = np.vstack([smooth(c, sigma, monotonic=True) for c in curves])
        mean_tpr = smooth(mean_tpr, sigma, monotonic=True)
        sd_tpr = smooth(sd_tpr, sigma)
    lo = np.clip(mean_tpr - sd_tpr, 0.0, 1.0)
    hi = np.clip(mean_tpr + sd_tpr, 0.0, 1.0)

    # individual models (thin, translucent)
    for c in curves:
        ax.plot(FPR_GRID, c, color=COL_INDIV, lw=1.0, alpha=0.75, zorder=2)

    ax.fill_between(FPR_GRID, lo, hi, color=COL_BAND, alpha=0.30,
                    lw=0, zorder=3)
    ax.plot(FPR_GRID, mean_tpr, color=COL_MEAN, lw=3.0, zorder=5,
            solid_capstyle="round")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, ls="--", zorder=1)

    # Limits are exactly [0, 1] on both axes: with headroom above 1.0 the 3 pt
    # stroke of the mean curve renders above the y = 1 gridline and reads as a
    # TPR greater than 1. Clipping at 1.0 makes the curve end flush with the top.
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Composite ROC — V3 ensemble (iteration {iteration})",
                 fontsize=15, pad=10)
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # In the two-panel figure the AUC is already stated above panel B, so the
    # legend there just identifies the colour.
    mean_label = ("Mean ROC "
                  f"(AUC = {np.mean(aucs):.3f} $\\pm$ {np.std(aucs, ddof=1):.3f})"
                  if auc_in_legend else "Mean ROC")
    handles = [
        Line2D([], [], color=COL_MEAN, lw=3.0, label=mean_label),
        Patch(facecolor=COL_BAND, alpha=0.30, label="$\\pm$ 1 SD across models"),
        Line2D([], [], color=COL_INDIV, lw=1.2,
               label=f"Individual models (n = {len(aucs)})"),
        Line2D([], [], color="gray", lw=1.5, ls="--", label="Chance"),
    ]
    return handles


def make_main_figure(curves, mean_tpr, sd_tpr, aucs, iteration, out_stem,
                     sigma=0.0):
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    handles = draw_base(ax, curves, mean_tpr, sd_tpr, aucs, iteration,
                        sigma=sigma)
    ax.legend(handles=handles, loc="lower right", frameon=True,
              framealpha=0.95, edgecolor="0.8", borderpad=0.7,
              labelspacing=0.55)
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=PUB_DPI)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)


def make_panel_figure(curves, mean_tpr, sd_tpr, aucs, f1s, iteration,
                      pooled, out_stem, rng_seed=0, sigma=0.0):
    """Two-panel version: (A) the composite ROC plus the pooled-data ROC,
    (B) the distribution of the 10 per-model AUCs (box + individual points).

    A side panel is used rather than an inset because at early checkpoints the
    ROC curves run close to the diagonal, where an inset would occlude data.
    """
    pooled_fpr, pooled_tpr, pooled_auc = pooled

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(10.4, 6.4),
        gridspec_kw=dict(width_ratios=[2.6, 1.0], wspace=0.32))

    # ---- panel A: composite ROC ------------------------------------------- #
    handles = draw_base(ax, curves, mean_tpr, sd_tpr, aucs, iteration,
                        sigma=sigma, auc_in_legend=False)
    if sigma > 0:
        # Put the pooled curve on the common grid so it is smoothed the same way.
        pooled_tpr = smooth(interp_tpr(pooled_fpr, pooled_tpr), sigma,
                            monotonic=True)
        pooled_fpr = FPR_GRID
    ax.plot(pooled_fpr, pooled_tpr, color=COL_POOL, lw=1.8, ls=(0, (5, 2)),
            zorder=4)
    handles.insert(2, Line2D([], [], color=COL_POOL, lw=1.8, ls=(0, (5, 2)),
                             label=f"Pooled data (AUC = {pooled_auc:.3f})"))
    ax.legend(handles=handles, loc="lower right", frameon=True,
              framealpha=0.95, edgecolor="0.8", borderpad=0.7,
              labelspacing=0.5, fontsize=10.5)

    # ---- panel B: per-model AUC distribution ------------------------------ #
    axb.boxplot([aucs], vert=True, widths=0.5, patch_artist=True,
                showfliers=False, medianprops=dict(color=COL_MEAN, lw=2.0),
                boxprops=dict(facecolor=COL_BAND, alpha=0.35,
                              edgecolor="0.35", lw=1.0),
                whiskerprops=dict(color="0.35", lw=1.0),
                capprops=dict(color="0.35", lw=1.0))
    rng = np.random.default_rng(rng_seed)
    jitter = 1 + rng.uniform(-0.15, 0.15, size=len(aucs))
    axb.scatter(jitter, aucs, s=42, color=COL_MEAN, alpha=0.9,
                edgecolor="white", linewidth=0.8, zorder=4)
    axb.errorbar([1.40], [aucs.mean()], yerr=[aucs.std(ddof=1)], fmt="o",
                 ms=7, color="0.2", ecolor="0.2", elinewidth=1.4, capsize=5,
                 zorder=5)
    axb.axhline(0.5, color="gray", lw=1.2, ls="--", zorder=1)
    axb.set_xlim(0.5, 1.95)
    # Fixed y-range so panel B is directly comparable between checkpoints.
    axb.set_ylim(0.45, 1.0)
    axb.set_xticks([])
    axb.set_ylabel("AUC")
    axb.set_title(f"Per-model AUC\n(mean {aucs.mean():.3f} $\\pm$ "
                  f"{aucs.std(ddof=1):.3f} SD)", fontsize=13, pad=10)
    axb.grid(axis="y", ls="--", lw=0.5, alpha=0.6)
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)
    axb.text(1.50, aucs.mean(), "mean $\\pm$ SD", va="center", ha="left",
             fontsize=9.5, color="0.2")

    del f1s
    fig.savefig(out_stem + ".png", dpi=PUB_DPI, bbox_inches="tight")
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iter", type=int, default=5000,
                    help="checkpoint iteration to aggregate (default: 5000)")
    ap.add_argument("--les-root", default=LES_ROOT,
                    help="folder holding the per-model LES directories")
    ap.add_argument("--folder-template", default=FOLDER_TMPL,
                    help="per-model folder name with {model} placeholder "
                         f"(default: {FOLDER_TMPL!r}). For the homodimer-removed "
                         "run use 'LES_{model}'.")
    ap.add_argument("--models", type=int, default=len(MODELS),
                    help=f"number of V3 models, 1..N (default: {len(MODELS)})")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="output directory (default: this script's folder)")
    ap.add_argument("--smooth", type=float, default=0.0, metavar="SIGMA",
                    help="Gaussian smoothing width for the plotted curves, in "
                         "FPR-grid points (5 ~ 0.01 FPR). Display only: all "
                         "reported numbers stay raw. 0 = no smoothing (default)")
    args = ap.parse_args()

    it = args.iter
    os.makedirs(args.outdir, exist_ok=True)
    set_publication_style()

    curves, aucs, f1s, names = [], [], [], []
    pooled_prs, pooled_rrs = [], []

    models = [f"V3-{i}" for i in range(1, args.models + 1)]
    for model in models:
        path = os.path.join(args.les_root, args.folder_template.format(model=model),
                            f"ckpt_{it}", f"combined_probabilities_iter{it}.csv")
        if not os.path.isfile(path):
            print(f"[WARN] missing, skipped: {path}")
            continue
        prs, rrs = read_combined_csv(path)
        fpr, tpr, roc_auc, best_f1 = roc_from_probs(prs, rrs)
        curves.append(interp_tpr(fpr, tpr))
        aucs.append(roc_auc)
        f1s.append(best_f1)
        names.append(model)
        pooled_prs.append(prs)
        pooled_rrs.append(rrs)
        print(f"{model:>6}  n_PRS={len(prs):4d}  n_RRS={len(rrs):4d}  "
              f"AUC={roc_auc:.4f}  bestF1={best_f1:.4f}")

    if len(curves) < 2:
        raise SystemExit("Need at least 2 models to build a composite curve.")

    curves = np.vstack(curves)
    aucs = np.asarray(aucs)
    f1s = np.asarray(f1s)
    mean_tpr = curves.mean(axis=0)
    sd_tpr = curves.std(axis=0, ddof=1)
    mean_tpr[0], sd_tpr[0] = 0.0, 0.0          # anchor at the origin

    # pooled ROC (all models' pairs thrown into one classifier evaluation)
    p_prs = np.concatenate(pooled_prs)
    p_rrs = np.concatenate(pooled_rrs)
    p_fpr, p_tpr, p_auc, p_f1 = roc_from_probs(p_prs, p_rrs)

    stem = os.path.join(args.outdir, f"composite_ROC_iter{it}")
    make_main_figure(curves, mean_tpr, sd_tpr, aucs, it, stem,
                     sigma=args.smooth)
    make_panel_figure(curves, mean_tpr, sd_tpr, aucs, f1s, it,
                      (p_fpr, p_tpr, p_auc), stem + "_panel",
                      sigma=args.smooth)

    # ---- data exports ----------------------------------------------------- #
    with open(stem + "_mean_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["FPR", "TPR_mean", "TPR_sd", "TPR_mean_minus_sd",
                    "TPR_mean_plus_sd", "TPR_min", "TPR_max"])
        lo = np.clip(mean_tpr - sd_tpr, 0, 1)
        hi = np.clip(mean_tpr + sd_tpr, 0, 1)
        for i, x in enumerate(FPR_GRID):
            w.writerow([f"{x:.4f}", f"{mean_tpr[i]:.6f}", f"{sd_tpr[i]:.6f}",
                        f"{lo[i]:.6f}", f"{hi[i]:.6f}",
                        f"{curves[:, i].min():.6f}", f"{curves[:, i].max():.6f}"])

    with open(stem + "_per_model.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "iteration", "AUC", "Best_F1",
                    "PRS_samples", "RRS_samples"])
        for i, m in enumerate(names):
            w.writerow([m, it, f"{aucs[i]:.6f}", f"{f1s[i]:.6f}",
                        len(pooled_prs[i]), len(pooled_rrs[i])])

    sd = aucs.std(ddof=1)
    sem = sd / np.sqrt(len(aucs))
    lines = [
        f"Composite ROC — checkpoint {it}",
        f"models (n = {len(names)}): {', '.join(names)}",
        f"positives per model (PRS): {len(pooled_prs[0])}    "
        f"negatives per model (RRS): {len(pooled_rrs[0])}",
        "",
        f"AUC  mean = {aucs.mean():.4f}   SD = {sd:.4f}   SEM = {sem:.4f}",
        f"AUC  median = {np.median(aucs):.4f}   min = {aucs.min():.4f}   "
        f"max = {aucs.max():.4f}",
        f"AUC  95% CI (t, df={len(aucs)-1}) = "
        f"[{aucs.mean() - 2.262 * sem:.4f}, {aucs.mean() + 2.262 * sem:.4f}]",
        "",
        f"Best-F1  mean = {f1s.mean():.4f}   SD = {f1s.std(ddof=1):.4f}   "
        f"min = {f1s.min():.4f}   max = {f1s.max():.4f}",
        "",
        f"Pooled data ({len(p_prs)} PRS + {len(p_rrs)} RRS): "
        f"AUC = {p_auc:.4f}   Best F1 = {p_f1:.4f}",
        "",
        ("figure curves: raw (unsmoothed)" if args.smooth <= 0 else
         f"figure curves: Gaussian-smoothed for display, sigma = "
         f"{args.smooth:g} FPR-grid points (~{args.smooth / (len(FPR_GRID) - 1):.4f} "
         f"FPR). All numbers above are from the raw data."),
    ]
    with open(stem + "_stats.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote figures and tables to {args.outdir}")


if __name__ == "__main__":
    main()
