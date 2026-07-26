#!/usr/bin/env python3
"""Composite (N-model ensemble) LES violin plots.

POOLS the PRS and RRS interaction scores P(token=1) across all models under
--parent at each training checkpoint (each model contributes its own reference
pairs), and draws the combined PRS-vs-RRS violin figure across checkpoints,
matching the per-model summary_prob_distributions_combined style.

By default it draws the native PRS-vs-RRS violin. If the randomized-control LES
folders exist alongside the native ones (suffixes _ps1_random / _ps2_random /
_ps1-ps2_random on the folder template), pass --controls to draw those too; any
missing control type is skipped with a warning.

Each per-model folder is `--folder-template` with `{k}` substituted; a control set
appends the suffix to that folder name. Reads
`<folder>/ckpt_<it>/{PRS,RRS}_iter<it>_probabilities.csv` (Probability_of_1 is the
second-to-last column, robust to commas in the prompt).

Example (homodimer-removed run, native only):
  python make_ensemble_les_violins.py \
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

PRS_C, RRS_C = "#2166ac", "#b2182b"
DEFAULT_CKPTS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
CONTROL_TYPES = [
    ("_ps1_random",     "ps1_random",     "Control - partner-1 randomized (ps1_random)"),
    ("_ps2_random",     "ps2_random",     "Control - partner-2 randomized (ps2_random)"),
    ("_ps1-ps2_random", "ps1-ps2_random", "Control - both partners randomized (ps1-ps2_random)"),
]


def read_p1(path):
    vals = []
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2:
                try:
                    vals.append(float(row[-2]))   # Probability_of_1
                except ValueError:
                    pass
    return vals


def pooled(parent, template, suffix, kind, it, models):
    out = []
    for k in range(1, models + 1):
        folder = template.format(k=k) + suffix
        fp = os.path.join(parent, folder, f"ckpt_{it}", f"{kind}_iter{it}_probabilities.csv")
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        out.extend(read_p1(fp))
    return np.array(out)


def make_plot(parent, template, suffix, title, base, ckpts, models):
    prs = [pooled(parent, template, suffix, "PRS", it, models) for it in ckpts]
    rrs = [pooled(parent, template, suffix, "RRS", it, models) for it in ckpts]
    n_prs = len(prs[0]); n_rrs = len(rrs[0])
    gap = 2.6
    prs_pos = np.arange(1, len(ckpts) + 1, dtype=float)
    rrs_pos = prs_pos + len(ckpts) + gap
    divider = len(ckpts) + gap / 2.0 + 0.5
    fig, ax = plt.subplots(figsize=(17, 6.6))

    def viol(data, pos, color):
        parts = ax.violinplot(data, positions=pos, widths=0.9,
                              showmedians=True, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(color); b.set_alpha(0.5)
            b.set_edgecolor(color); b.set_linewidth(0.7)
        if "cmedians" in parts:
            parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.1)

    viol(prs, prs_pos, PRS_C)
    viol(rrs, rrs_pos, RRS_C)
    ax.axvline(divider, color="0.5", ls="--", lw=1.3)
    ax.set_xticks(list(prs_pos) + list(rrs_pos))
    ax.set_xticklabels([str(c) for c in ckpts] * 2, rotation=90, fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0.2, rrs_pos[-1] + 0.8)
    ax.set_ylabel("P(interaction) = P(token = 1)", fontsize=14)
    ax.set_xlabel("Training checkpoint (iteration)", fontsize=13)
    ax.yaxis.grid(True, ls="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.text(prs_pos.mean(), 1.06, "PRS (positives)", ha="center", va="bottom",
            fontsize=14, color=PRS_C, fontweight="bold")
    ax.text(rrs_pos.mean(), 1.06, "RRS (negatives)", ha="center", va="bottom",
            fontsize=14, color=RRS_C, fontweight="bold")
    ax.set_title(f"{models}-model composite - {title}\n"
                 f"pooled across models; PRS n = {n_prs}, RRS n = {n_rrs} "
                 f"per checkpoint",
                 fontsize=14, pad=30)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {base}.png/.pdf  (PRS n={n_prs}, RRS n={n_rrs}/ckpt)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--folder-template", default="LES_results_V3-{k}_6k")
    ap.add_argument("--models", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default="composite")
    ap.add_argument("--controls", action="store_true",
                    help="also draw the 3 randomized-control violins if present")
    ap.add_argument("--ckpts", type=int, nargs="+", default=DEFAULT_CKPTS)
    a = ap.parse_args()
    out = a.out or os.path.join(a.parent, "composite")
    os.makedirs(out, exist_ok=True)

    # native (always)
    make_plot(a.parent, a.folder_template, "", "Native - PRS vs RRS",
              os.path.join(out, f"{a.prefix}_native_PRS-RRS_violins"), a.ckpts, a.models)

    if a.controls:
        for suffix, tag, title in CONTROL_TYPES:
            try:
                make_plot(a.parent, a.folder_template, suffix, title,
                          os.path.join(out, f"{a.prefix}_{tag}_PRS-RRS_violins"),
                          a.ckpts, a.models)
            except FileNotFoundError as e:
                print(f"  [skip] control '{tag}' not found ({e})")
    print("DONE ->", out)


if __name__ == "__main__":
    main()
