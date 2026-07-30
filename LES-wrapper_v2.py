#!/usr/bin/env python
"""
LES-wrapper_v2.py — Learning Efficiency Score evaluation across training checkpoints.

Runs inference on PRS (Positive Reference Set) and RRS (Random Reference Set)
prompts at every saved checkpoint in a directory, computes ROC-AUC and Best-F1 at
each checkpoint, then integrates these into a single Learning Efficiency Score
(LES) per metric — the area under the metric-vs-iteration curve.

v2 change vs LES-wrapper.py: the optimal-F1 *threshold* metric is no longer
reported. The best-F1 threshold was a degenerate/uninformative diagnostic (for
non-discriminating controls it collapses toward 0, i.e. "predict everything
positive"), so it added noise to the report. Removed here: the trajectory_Threshold
plot, LES-Threshold, the Best_F1_Threshold summary column, the manifest Threshold
entry, and the threshold panel of the combined figure. ROC-AUC and Best-F1 (and how
they are computed) are unchanged.

ppiGPLM is a vanilla GPT-2 model (no HOPE/Titan/CMS). Inference is run with
sample_fasta3.3_softmax_error_handling3f.py, the plain vanilla sampler.

Basic usage:
    python LES-wrapper_v2.py \\
        --checkpoint_dir <dir> \\
        --prs_file <prs.txt> \\
        --rrs_file <rrs.txt> \\
        --output_dir <out>
"""

import os
import sys
import re
import glob
import argparse
import subprocess
import csv
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score

# -----------------------------------------------------------------------------
# Publication-quality figure defaults
# -----------------------------------------------------------------------------
PUB_DPI = 600

def set_publication_style():
    """Apply consistent, publication-quality matplotlib defaults (300 dpi,
    tight bounding box, larger readable fonts, heavier axis lines)."""
    plt.rcParams.update({
        'savefig.dpi': PUB_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'font.family': 'DejaVu Sans',
        'font.size': 13,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'axes.linewidth': 1.2,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
        'lines.markersize': 7,
    })

# -----------------------------------------------------------------------------
# Parse command-line arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='LES-wrapper: Learning Efficiency Score evaluation across checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python LES-wrapper_v2.py --checkpoint_dir out-model --prs_file prs.txt --rrs_file rrs.txt --output_dir results
  python LES-wrapper_v2.py --checkpoint_dir out-model --prs_file prs.txt --rrs_file rrs.txt --output_dir results --color_threshold
        """
    )

    # Required arguments
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing model checkpoints (ckpt_*.pt files)')
    parser.add_argument('--prs_file', type=str, required=True,
                        help='Path to Positive Reference Set prompts file')
    parser.add_argument('--rrs_file', type=str, required=True,
                        help='Path to Random Reference Set prompts file')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='LES_results',
                        help='Directory to save all outputs (default: LES_results)')

    # Checkpoint pattern
    parser.add_argument('--checkpoint_pattern', type=str, default='ckpt_*.pt',
                        help='Pattern to match checkpoint files (default: ckpt_*.pt)')

    # Include final checkpoint
    parser.add_argument('--include_final', action='store_true',
                        help='Also include ckpt.pt (final checkpoint) if present')

    # Control flow
    parser.add_argument('--no_metrics', action='store_true',
                        help='Skip ROC-AUC and Best-F1 (and the LES / AUC-F1 trajectory '
                             'plots that derive from them). Use for random-substituted '
                             'CONTROL sets: there neither reference file contains true '
                             'positives (both are random pairs), so ranking metrics like '
                             'AUC/F1 are not meaningful. Probability distributions '
                             '(violins) and the raw probability CSVs are still produced. '
                             'Auto-enabled when the PRS/RRS filenames contain "random".')
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip inference step (use existing probability files)')

    # Plotting options
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating trajectory plots')
    parser.add_argument('--color_threshold', action='store_true',
                        help='Color the ROC curve by decision threshold and add a '
                             'colorbar (default: plain single-color curve, no scale)')

    return parser.parse_known_args()

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def extract_iteration_from_checkpoint(ckpt_name):
    """Extract iteration number from checkpoint filename."""
    # Match patterns like ckpt_1000.pt, ckpt_iter_1000.pt, etc.
    match = re.search(r'ckpt_?(?:iter_)?(\d+)\.pt$', ckpt_name)
    if match:
        return int(match.group(1))
    # For ckpt.pt (final), return a large number
    if ckpt_name == 'ckpt.pt':
        return float('inf')
    return None

def get_checkpoints(checkpoint_dir, pattern, include_final=False):
    """Get sorted list of checkpoint files with their iterations."""
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, pattern))

    if include_final:
        final_ckpt = os.path.join(checkpoint_dir, 'ckpt.pt')
        if os.path.exists(final_ckpt) and final_ckpt not in ckpt_files:
            ckpt_files.append(final_ckpt)

    checkpoints = []
    for ckpt_path in ckpt_files:
        ckpt_name = os.path.basename(ckpt_path)
        iteration = extract_iteration_from_checkpoint(ckpt_name)
        if iteration is not None:
            checkpoints.append((ckpt_name, iteration, ckpt_path))

    # Sort by iteration
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints

def run_inference(sample_script, model_dir, ckpt_name, input_file, output_dir,
                  output_prefix, extra_args):
    """Run inference using the vanilla-GPT sample script."""
    cmd = [
        sys.executable, sample_script,
        '--input_file', input_file,
        '--output_dir', output_dir,
        '--output_prefix', output_prefix,
        '--model_dir', model_dir,
        '--ckpt_name', ckpt_name,
    ] + extra_args

    print(f"  Running: {' '.join(cmd[:8])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: Inference failed for {ckpt_name}")
        print(f"  stderr: {result.stderr[:500]}")
        return False

    return True

def extract_probabilities_from_csv(csv_path):
    """Extract probability of '1' from inference output CSV.

    Note: The prompts may contain commas (e.g., '<ps1>,SEQ,<ps2>,SEQ,<')
    so the probability columns are at the end of each row, not at fixed positions.
    The format is: Prompt, Probability_of_1, Probability_of_0
    But due to commas in prompts, probabilities are in columns [-2] and [-1].
    """
    probabilities = []
    if not os.path.exists(csv_path):
        print(f"  WARNING: File not found: {csv_path}")
        return probabilities

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header
        for row in reader:
            if len(row) >= 2:
                # Probabilities are at the END of the row due to commas in prompts
                # Second-to-last column is Probability_of_1
                try:
                    prob_1 = float(row[-2])  # Use negative indexing to get 2nd from end
                    probabilities.append(prob_1)
                except (ValueError, IndexError):
                    continue
    return probabilities

def combine_probabilities(prs_probs, rrs_probs, output_path):
    """Combine PRS and RRS probabilities into a single CSV for ROC analysis."""
    max_len = max(len(prs_probs), len(rrs_probs))

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # No header - roc.py expects raw values
        for i in range(max_len):
            prs_val = prs_probs[i] if i < len(prs_probs) else ''
            rrs_val = rrs_probs[i] if i < len(rrs_probs) else ''
            writer.writerow([prs_val, rrs_val])

    return output_path

def run_roc_analysis_internal(combined_csv_path, output_plot_path, color_threshold=False):
    """Run ROC analysis and return (roc_auc, best_f1)."""
    # Read probabilities
    prs_probs = []
    rrs_probs = []

    with open(combined_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                prs_val = row[0].strip()
                rrs_val = row[1].strip()
                if prs_val:
                    prs_probs.append(float(prs_val))
                if rrs_val:
                    rrs_probs.append(float(rrs_val))

    if not prs_probs or not rrs_probs:
        return None, None

    # Assign labels (PRS = 1, RRS = 0)
    prs_labels = [1] * len(prs_probs)
    rrs_labels = [0] * len(rrs_probs)

    probs = np.array(prs_probs + rrs_probs)
    labels = np.array(prs_labels + rrs_labels)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Filter valid thresholds (only used for the optional color-by-threshold ROC plot)
    finite_idxs = np.where(np.isfinite(thresholds))[0]
    fpr = fpr[finite_idxs]
    tpr = tpr[finite_idxs]
    thresholds = thresholds[finite_idxs]

    valid_thresholds_idxs = np.where((thresholds >= 0) & (thresholds <= 1))[0]
    fpr = fpr[valid_thresholds_idxs]
    tpr = tpr[valid_thresholds_idxs]
    thresholds = thresholds[valid_thresholds_idxs]

    # Best F1 over ALL candidate thresholds — every unique score, not just the
    # ROC vertices. roc_curve(drop_intermediate=True) prunes collinear vertices;
    # scanning the full unique-score set guarantees the true F1-optimal cutoff is
    # considered. Decision rule: prob >= threshold => positive.
    best_f1 = -1.0
    for thresh in np.unique(probs):
        current_f1 = f1_score(labels, (probs >= thresh).astype(int), zero_division=0)
        if current_f1 >= best_f1:
            best_f1 = current_f1

    # ROC plot. Default: clean single-color curve. With color_threshold: the
    # curve is colored by decision threshold and a colorbar is added.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    if color_threshold:
        norm = plt.Normalize(vmin=thresholds.min(), vmax=thresholds.max())
        cmap = plt.cm.viridis
        for i in range(len(fpr) - 1):
            ax.plot(fpr[i:i + 2], tpr[i:i + 2], color=cmap(norm(thresholds[i])), lw=2.5)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Threshold', fontsize=15)
    else:
        ax.plot(fpr, tpr, color='#08519c', lw=2.5)

    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')

    # Show AUC and Best F1 on the curve.
    legend_text = f'AUC = {roc_auc:.3f}, Best F1 = {best_f1:.3f}'
    ax.legend([legend_text], loc="lower right")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=PUB_DPI, format='png')
    # Also emit a vector PDF (scalable, no pixelation) for publication.
    plt.savefig(os.path.splitext(output_plot_path)[0] + '.pdf', format='pdf')
    plt.close(fig)

    return roc_auc, best_f1

def compute_les(iterations, values):
    """Compute Learning Efficiency Score (area under curve)."""
    if len(iterations) < 2 or len(values) < 2:
        return 0.0

    # Use numpy trapezoid integration
    # Normalize iterations to [0, 1] for comparable LES across different training lengths
    iters = np.array(iterations, dtype=float)
    vals = np.array(values, dtype=float)

    # Remove any inf iterations (final checkpoint)
    valid_mask = np.isfinite(iters)
    iters = iters[valid_mask]
    vals = vals[valid_mask]

    if len(iters) < 2:
        return 0.0

    # Normalize iterations
    iters_normalized = (iters - iters.min()) / (iters.max() - iters.min())

    # Compute area under curve using trapezoidal rule
    les = np.trapezoid(vals, iters_normalized)

    return les

def plot_metric_trajectory(iterations, values, metric_name, output_path, les_value,
                           les_label=None):
    """Plot a single metric trajectory across checkpoints (y-axis fixed to 0..1)."""
    plt.figure(figsize=(10, 6))

    # Filter out inf iterations
    valid_mask = [i < float('inf') for i in iterations]
    plot_iters = [it for it, v in zip(iterations, valid_mask) if v]
    plot_vals = [val for val, v in zip(values, valid_mask) if v]

    plt.plot(plot_iters, plot_vals, 'bo-', linewidth=2, markersize=8)
    plt.fill_between(plot_iters, plot_vals, alpha=0.3)

    plt.xlabel('Training Iteration', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    label = les_label if les_label is not None else f'LES-{metric_name}'
    plt.title(f'{metric_name} vs Training Iteration\n{label} = {les_value:.4f}', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add value annotations
    for i, (it, val) in enumerate(zip(plot_iters, plot_vals)):
        if i % max(1, len(plot_iters) // 10) == 0:  # Annotate every ~10% of points
            plt.annotate(f'{val:.3f}', (it, val), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PUB_DPI)
    plt.savefig(os.path.splitext(output_path)[0] + '.pdf', format='pdf')
    plt.close()

def plot_combined_trajectories(iterations, auc_values, f1_values, output_path, les_values):
    """Plot AUC and Best-F1 metrics on a single figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Filter out inf iterations
    valid_mask = [i < float('inf') for i in iterations]
    plot_iters = [it for it, v in zip(iterations, valid_mask) if v]
    plot_auc = [val for val, v in zip(auc_values, valid_mask) if v]
    plot_f1 = [val for val, v in zip(f1_values, valid_mask) if v]

    # AUC plot
    axes[0].plot(plot_iters, plot_auc, 'bo-', linewidth=2, markersize=6)
    axes[0].fill_between(plot_iters, plot_auc, alpha=0.3)
    axes[0].set_xlabel('Training Iteration')
    axes[0].set_ylabel('AUC')
    axes[0].set_title(f'AUC Trajectory\nLES-AUC = {les_values["AUC"]:.4f}')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_ylim([0, 1])

    # F1 plot
    axes[1].plot(plot_iters, plot_f1, 'go-', linewidth=2, markersize=6)
    axes[1].fill_between(plot_iters, plot_f1, alpha=0.3, color='green')
    axes[1].set_xlabel('Training Iteration')
    axes[1].set_ylabel('Best F1')
    axes[1].set_title(f'Best F1 Trajectory\nArea under the curve = {les_values["F1"]:.4f}')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=PUB_DPI)
    plt.savefig(os.path.splitext(output_path)[0] + '.pdf', format='pdf')
    plt.close()

def plot_probability_distribution(prs_probs, rrs_probs, output_path, iter_str, ax=None):
    """Probability-distribution plot for one checkpoint: P(interaction) for PRS
    (positives) vs RRS (negatives). y-axis = probability, fixed to [0, 1].
    Violin (distribution shape) + jittered points. Draws into `ax` if given
    (used by the summary grid); otherwise makes and saves its own figure.
    """
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 6))

    data = [prs_probs, rrs_probs]
    colors = ['#2166ac', '#b2182b']            # PRS blue, RRS red
    parts = ax.violinplot(data, positions=[1, 2], showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.35); pc.set_edgecolor(colors[i])
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.5)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        if len(d):
            x = (i + 1) + (rng.random(len(d)) - 0.5) * 0.16
            ax.scatter(x, d, s=8, color=colors[i], alpha=0.5, edgecolors='none')
    ax.set_xticks([1, 2]); ax.set_xticklabels(['PRS', 'RRS'])
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('P(interaction)')
    ax.set_title(f'iter {iter_str}')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    if own:
        plt.tight_layout()
        plt.savefig(output_path, dpi=PUB_DPI)
        plt.savefig(os.path.splitext(output_path)[0] + '.pdf', format='pdf')
        plt.close(fig)


def plot_summary_distributions(dist_data, output_path):
    """One summary figure: the per-checkpoint probability distributions (PRS vs RRS),
    one panel per checkpoint, every panel with y-axis fixed to [0, 1].
    dist_data: list of (iter_str, prs_probs, rrs_probs) in checkpoint order.
    """
    n = len(dist_data)
    if n == 0:
        return
    ncols = min(6, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows),
                             squeeze=False)
    for idx, (iter_str, prs, rrs) in enumerate(dist_data):
        r, c = divmod(idx, ncols)
        plot_probability_distribution(prs, rrs, None, iter_str, ax=axes[r][c])
    for idx in range(n, nrows * ncols):          # hide unused panels
        r, c = divmod(idx, ncols)
        axes[r][c].axis('off')
    fig.suptitle('Probability distributions across checkpoints (PRS vs RRS)',
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=PUB_DPI)
    fig.savefig(os.path.splitext(output_path)[0] + '.pdf', format='pdf')
    plt.close(fig)


def plot_summary_distributions_combined(dist_data, output_path):
    """All per-checkpoint probability distributions on ONE axes (publication quality):
    the x-axis is split into two sections — the PRS violins (left, one per checkpoint,
    labelled by its training iteration) then the RRS violins (right, same). y-axis is
    P(interaction), fixed to [0, 1]. Makes PRS-stays-high / RRS-stays-low visible at a
    glance across training.
    """
    n = len(dist_data)
    if n == 0:
        return
    iters = [d[0] for d in dist_data]
    prs_list = [d[1] for d in dist_data]
    rrs_list = [d[2] for d in dist_data]

    prs_pos = list(range(1, n + 1))
    rrs_pos = list(range(n + 2, 2 * n + 2))     # gap of 1 at n+1 for the divider
    divider = n + 1

    fig, ax = plt.subplots(figsize=(max(11.0, 0.62 * (2 * n + 2)), 7.0))

    def _violins(data, positions, color):
        parts = ax.violinplot(data, positions=positions, showmedians=True,
                              showextrema=False, widths=0.82)
        for pc in parts['bodies']:
            pc.set_facecolor(color); pc.set_alpha(0.40); pc.set_edgecolor(color)
            pc.set_linewidth(0.8)
        if 'cmedians' in parts:
            parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.1)

    def _points(data, positions, color):
        # overlay the individual data points as jittered dots (as in the
        # per-checkpoint prob_dist plots)
        rng = np.random.default_rng(0)
        for d, pos in zip(data, positions):
            if len(d):
                x = pos + (rng.random(len(d)) - 0.5) * 0.16
                ax.scatter(x, d, s=6, color=color, alpha=0.5, edgecolors='none',
                           zorder=3)

    _violins(prs_list, prs_pos, '#2166ac')       # PRS blue (positives)
    _violins(rrs_list, rrs_pos, '#b2182b')       # RRS red (negatives)
    _points(prs_list, prs_pos, '#2166ac')
    _points(rrs_list, rrs_pos, '#b2182b')

    ax.axvline(divider, color='gray', linestyle='--', linewidth=1.3)

    ax.set_xticks(prs_pos + rrs_pos)
    ax.set_xticklabels([str(it) for it in iters] * 2, rotation=90, fontsize=10)
    ax.set_xlim(0.3, 2 * n + 2 - 0.3)
    ax.set_ylim(0, 1)
    ax.set_ylabel('P(interaction)')
    ax.set_xlabel('Training iteration')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # section headers above each group (axis-fraction y, data-coord x)
    tr = ax.get_xaxis_transform()
    ax.text((1 + n) / 2.0, 1.03, 'PRS (positives)', ha='center', va='bottom',
            fontsize=15, color='#2166ac', fontweight='bold', transform=tr)
    ax.text((n + 2 + 2 * n + 1) / 2.0, 1.03, 'RRS (negatives)', ha='center',
            va='bottom', fontsize=15, color='#b2182b', fontweight='bold', transform=tr)
    ax.set_title('Probability distributions across checkpoints — PRS vs RRS', pad=34)

    fig.tight_layout()
    fig.savefig(output_path, dpi=PUB_DPI)
    fig.savefig(os.path.splitext(output_path)[0] + '.pdf', format='pdf')
    plt.close(fig)


def write_analysis_readme(output_dir):
    """Write a README.md legend for the analysis-level plots (not per-checkpoint)."""
    text = """# LES analysis — plot legend

This folder is the Learning Efficiency Score (LES) analysis for one model evaluated
on one PRS (Positive Reference Set) / RRS (Random Reference Set) pair, across all
saved training checkpoints. **PRS = blue (positives); RRS = red (negatives).** The
interaction score for each pair is `P(next token = 1)` in [0, 1] — see the next
section for exactly what that number is.

## What "probability" / `P(interaction)` means here (read this)
The value on every y-axis — labelled `P(interaction)` — is **not** an empirical
frequency or a calibrated statistical probability that two proteins interact. It is the
**model's next-token softmax probability for the token `1`**.

For a prompt holding the two sequences (`<ps1>,SEQ1,<ps2>,SEQ2,<`), the model emits
logits over its entire vocabulary at the next position; a softmax turns those logits
into a probability for every token, and we read off the mass on the single token `1`:

```
P(interaction) = softmax(last-position logits)[ id("1") ]
```

- It is a genuine probability **of the next generated token being `1`**, in [0, 1] — i.e.
  a softmax-normalised logit, computed once per pair from a single forward pass.
- The softmax is over the **whole vocabulary**, not renormalised over just `{0, 1}`, so
  `P(1)` and `P(0)` need **not** sum to 1 (a trained model puts almost all mass on
  `0`/`1`, so in practice they nearly do). Both are saved in each per-checkpoint
  `*_probabilities.csv` (`Probability_of_1`, `Probability_of_0`).
- **No temperature** is applied to this number. The sampler's `temperature=0.1` affects
  only a separately-generated example continuation (the classification `.txt`), never
  the probability that is scored and plotted.
- **It is `P(1)` for *both* PRS and RRS** — a single, shared score function, *not*
  "`P(1)` for PRS and `P(0)` for RRS". That shared score is what makes the ROC/AUC
  valid: every pair gets the same score `P(1)`, PRS pairs are labelled 1 and RRS pairs 0,
  and AUC measures how well `P(1)` ranks true interactors above random ones. A good model
  pushes `P(1)` high for PRS (blue, near 1) and low for RRS (red, near 0 — equivalently,
  its mass goes on `0`). `Probability_of_0` is recorded but is **not** used for AUC, F1,
  LES, or any plot.

## Summary figures (this folder)
- **`trajectory_AUC.png`** — ROC-AUC vs training iteration (y-axis 0-1). Subtitle
  `LES-AUC` is the area under this AUC-vs-iteration curve (the learning-efficiency
  score for AUC).
- **`trajectory_F1.png`** — Best-F1 vs training iteration (y-axis 0-1). Subtitle
  `Area under the curve` is the integral of the F1-vs-iteration curve.
- **`trajectory_combined.png`** — the AUC and Best-F1 trajectories side by side.
- **`summary_prob_distributions.png`** — a grid, one panel per checkpoint; each panel
  shows the P(interaction) distribution for PRS vs RRS (violin + jittered points),
  y-axis 0-1.
- **`summary_prob_distributions_combined.png`** — the same distributions on a single
  axes with the x-axis split into two sections: all PRS violins (left, one per
  checkpoint, labelled by iteration) then all RRS violins (right). Lets you see PRS
  staying high and RRS staying low across training in one view.
- **`summary_table.csv`** — per-checkpoint AUC and Best-F1, plus a final LES row.
- **`manifest.json`** — run metadata and LES values.

## How to read the violin plots
Each violin summarises the 100 P(interaction) scores for one reference set (PRS or
RRS) at one checkpoint. The anatomy:

- **Width (the shaded shape):** a kernel-density estimate (KDE) of the score
  distribution. The violin is *wider at probability values where more of the 100
  pairs fall* and narrows where few pairs do — so a bulge near the top means many
  pairs scored close to 1, a bulge near the bottom means many scored close to 0. The
  width is a *relative* density (each violin is scaled to the same maximum width); it
  shows the *shape* of the distribution, not an absolute count. The KDE can extend
  slightly beyond the individual points because it smooths them.
- **Horizontal black line:** the **median** of the 100 scores (the 50th percentile —
  half the pairs score above it, half below). It is *not* the mean; with skewed,
  piled-up distributions the median is the more robust centre. A PRS median near 1 and
  an RRS median near 0 is the signature of a well-separated, discriminating model.
- **Dots:** the individual pairs — all 100 raw scores, jittered horizontally only (the
  jitter is cosmetic, to unstack points; vertical position is the true probability).
  They let you see the actual sample behind the smoothed shape, including outliers the
  KDE glosses over (e.g. a handful of PRS pairs scoring low).
- **Colour:** blue = PRS (positives, should sit high), red = RRS (negatives, should
  sit low). No whiskers/quartile boxes or min/max bars are drawn — only the density,
  the median line, and the points.

## Per-checkpoint folders (`ckpt_<iter>/`)
Each holds that checkpoint's ROC curve (`ROC_iter<iter>.png`), its probability
distribution (`prob_dist_iter<iter>.png`), and the raw PRS/RRS/combined probability
CSVs. (These folders intentionally have no README.)

## Reading it
A discriminating model shows PRS probabilities clustered near 1 and RRS near 0
(clear separation in the distribution plots), an AUC that rises and plateaus, and —
for the fully-randomized `ps1-ps2` control — AUC ~ 0.5 with PRS/RRS distributions
overlapping (the expected null).
"""
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(text)


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main():
    args, extra_args = parse_args()

    # Random-substituted control sets have no true positives (both reference
    # files are random pairs), so ROC-AUC / Best-F1 / LES are not meaningful.
    # Honour --no_metrics, and auto-enable it when the filenames say "random".
    _names = (os.path.basename(args.prs_file) + os.path.basename(args.rrs_file)).lower()
    if not args.no_metrics and 'random' in _names:
        args.no_metrics = True
        print("[no_metrics] random-control reference set detected -> skipping "
              "AUC/F1/LES; reporting probability distributions only.")

    set_publication_style()

    # Validate inputs
    if not os.path.exists(args.checkpoint_dir):
        print(f"ERROR: Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)

    if not os.path.exists(args.prs_file):
        print(f"ERROR: PRS file not found: {args.prs_file}")
        sys.exit(1)

    if not os.path.exists(args.rrs_file):
        print(f"ERROR: RRS file not found: {args.rrs_file}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find sample script. ppiGPLM is a vanilla GPT-2 model (no HOPE/Titan/CMS),
    # so we use the plain vanilla inference script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_script = os.path.join(script_dir, 'sample_fasta3.3_softmax_error_handling3f.py')
    model_type = "Vanilla GPT"

    if not os.path.exists(sample_script):
        print(f"ERROR: Sample script not found: {sample_script}")
        sys.exit(1)

    # Get checkpoints
    checkpoints = get_checkpoints(args.checkpoint_dir, args.checkpoint_pattern, args.include_final)

    if not checkpoints:
        print(f"ERROR: No checkpoints found matching pattern '{args.checkpoint_pattern}' in {args.checkpoint_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("LES-wrapper: Learning Efficiency Score Evaluation")
    print(f"{'='*60}")
    print(f"Model type: {model_type}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"PRS file: {args.prs_file}")
    print(f"RRS file: {args.rrs_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"{'='*60}\n")

    # Any extra arguments passed through to the sample script.
    sample_extra_args = list(extra_args)

    # Results storage
    results = []
    iterations = []
    auc_values = []
    f1_values = []
    dist_data = []   # (iter_str, prs_probs, rrs_probs) per checkpoint, for the summary

    # Process each checkpoint
    for idx, (ckpt_name, iteration, ckpt_path) in enumerate(checkpoints):
        iter_str = str(iteration) if iteration < float('inf') else 'final'
        print(f"\n[{idx+1}/{len(checkpoints)}] Processing checkpoint: {ckpt_name} (iteration {iter_str})")

        # Create subdirectory for this checkpoint
        ckpt_subdir = os.path.join(args.output_dir, f"ckpt_{iter_str}")
        os.makedirs(ckpt_subdir, exist_ok=True)

        prs_prefix = f"PRS_iter{iter_str}"
        rrs_prefix = f"RRS_iter{iter_str}"

        prs_csv = os.path.join(ckpt_subdir, f"{prs_prefix}_probabilities.csv")
        rrs_csv = os.path.join(ckpt_subdir, f"{rrs_prefix}_probabilities.csv")

        if not args.skip_inference:
            # Run inference for PRS
            print(f"  Running PRS inference...")
            if not run_inference(sample_script, args.checkpoint_dir, ckpt_name,
                               args.prs_file, ckpt_subdir, prs_prefix, sample_extra_args):
                print(f"  SKIPPING checkpoint due to inference error")
                continue

            # Run inference for RRS
            print(f"  Running RRS inference...")
            if not run_inference(sample_script, args.checkpoint_dir, ckpt_name,
                               args.rrs_file, ckpt_subdir, rrs_prefix, sample_extra_args):
                print(f"  SKIPPING checkpoint due to inference error")
                continue

        # Extract probabilities
        print(f"  Extracting probabilities...")
        prs_probs = extract_probabilities_from_csv(prs_csv)
        rrs_probs = extract_probabilities_from_csv(rrs_csv)

        if not prs_probs or not rrs_probs:
            print(f"  WARNING: Could not extract probabilities, skipping")
            continue

        print(f"  PRS samples: {len(prs_probs)}, RRS samples: {len(rrs_probs)}")

        # Combine probabilities
        combined_csv = os.path.join(ckpt_subdir, f"combined_probabilities_iter{iter_str}.csv")
        combine_probabilities(prs_probs, rrs_probs, combined_csv)

        # Per-checkpoint probability-distribution plot (PRS vs RRS), y in [0, 1],
        # from this checkpoint's combined_probabilities_iter<N>.csv values.
        plot_probability_distribution(
            prs_probs, rrs_probs,
            os.path.join(ckpt_subdir, f"prob_dist_iter{iter_str}.png"), iter_str)
        dist_data.append((iter_str, list(prs_probs), list(rrs_probs)))

        # Run ROC analysis (skipped for random controls — no true positives)
        if args.no_metrics:
            roc_auc = best_f1 = None
            print(f"  (no_metrics) skipping ROC/AUC/F1; distribution only")
        else:
            print(f"  Running ROC analysis...")
            roc_plot = os.path.join(ckpt_subdir, f"ROC_iter{iter_str}.png")
            roc_auc, best_f1 = run_roc_analysis_internal(
                combined_csv, roc_plot, color_threshold=args.color_threshold)
            if roc_auc is None:
                print(f"  WARNING: ROC analysis failed, skipping")
                continue
            print(f"  Results: AUC={roc_auc:.4f}, F1={best_f1:.4f}")

        # Store results
        results.append({
            'checkpoint': ckpt_name,
            'iteration': iteration if iteration < float('inf') else 'final',
            'AUC': roc_auc,
            'Best_F1': best_f1,
            'PRS_samples': len(prs_probs),
            'RRS_samples': len(rrs_probs)
        })

        iterations.append(iteration)
        if not args.no_metrics:
            auc_values.append(roc_auc)
            f1_values.append(best_f1)

    # Compute LES values (only meaningful with >= 2 checkpoints). With a single
    # checkpoint, do the per-checkpoint analysis only and skip all summaries.
    # LES + AUC/F1 trajectories need >= 2 checkpoints AND meaningful metrics
    # (random-control sets have none). Distribution summaries only need >= 2 ckpts.
    multi = len(iterations) >= 2
    have_metrics = (not args.no_metrics) and len(auc_values) >= 2
    if have_metrics:
        print(f"\n{'='*60}")
        print("Computing Learning Efficiency Scores (LES)")
        print(f"{'='*60}")
        les_auc = compute_les(iterations, auc_values)
        les_f1 = compute_les(iterations, f1_values)
        les_values = {'AUC': les_auc, 'F1': les_f1}
        print(f"  LES-AUC: {les_auc:.6f}")
        print(f"  LES-F1: {les_f1:.6f}")
    else:
        les_auc = les_f1 = None
        les_values = {}
        if args.no_metrics:
            print("\n[no_metrics] random control — skipping LES and AUC/F1 trajectories; "
                  "probability distributions are still produced.")
        else:
            print(f"\nOnly {len(iterations)} checkpoint(s) analyzed — skipping LES / "
                  f"trajectories (need >= 2 checkpoints).")

    # Plots: AUC/F1 trajectories only when metrics are meaningful; the PRS-vs-RRS
    # probability-distribution summaries are produced whenever there are >= 2 ckpts.
    if not args.no_plots:
        if have_metrics:
            print(f"\nGenerating trajectory plots...")
            plot_metric_trajectory(iterations, auc_values, 'AUC',
                                  os.path.join(args.output_dir, 'trajectory_AUC.png'), les_auc,
                                  les_label='LES-AUC')
            plot_metric_trajectory(iterations, f1_values, 'Best F1',
                                  os.path.join(args.output_dir, 'trajectory_F1.png'), les_f1,
                                  les_label='Area under the curve')
            plot_combined_trajectories(iterations, auc_values, f1_values,
                                      os.path.join(args.output_dir, 'trajectory_combined.png'), les_values)
        if multi:
            plot_summary_distributions(dist_data,
                                       os.path.join(args.output_dir, 'summary_prob_distributions.png'))
            plot_summary_distributions_combined(
                dist_data,
                os.path.join(args.output_dir, 'summary_prob_distributions_combined.png'))
            write_analysis_readme(args.output_dir)
            print(f"  Saved distribution plots to {args.output_dir}")

    # Generate summary table
    print(f"\nGenerating summary table...")
    summary_csv = os.path.join(args.output_dir, 'summary_table.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['checkpoint', 'iteration', 'AUC', 'Best_F1',
                                               'PRS_samples', 'RRS_samples'])
        writer.writeheader()
        writer.writerows(results)

    # Add LES row (only when metrics are meaningful and >= 2 checkpoints)
    if have_metrics:
        with open(summary_csv, 'a', newline='') as f:
            f.write(f"\nLES (Learning Efficiency Score),---,{les_auc:.6f},{les_f1:.6f},---,---\n")

    print(f"  Saved summary table to {summary_csv}")

    # Generate JSON manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint_dir': args.checkpoint_dir,
        'prs_file': args.prs_file,
        'rrs_file': args.rrs_file,
        'output_dir': args.output_dir,
        'num_checkpoints': len(checkpoints),
        'num_successful': len(results),
        'LES': ({'AUC': les_auc, 'F1': les_f1} if have_metrics else None),
        'metrics_excluded': bool(args.no_metrics),
        'results': results
    }

    manifest_path = os.path.join(args.output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"  Saved manifest to {manifest_path}")

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Checkpoints processed: {len(results)}/{len(checkpoints)}")
    if have_metrics:
        print(f"\nLearning Efficiency Scores (LES):")
        print(f"  LES-AUC:       {les_auc:.6f}")
        print(f"  LES-F1:        {les_f1:.6f}")
    elif args.no_metrics:
        print("\n(Random control: AUC/F1/LES excluded; probability distributions produced.)")
    else:
        print("\n(Single checkpoint: LES and trajectory/distribution summaries skipped.)")

    if results:
        final_result = results[-1]
        print(f"\nFinal Checkpoint Performance:")
        print(f"  AUC:       {final_result['AUC']:.4f}")
        print(f"  Best F1:   {final_result['Best_F1']:.4f}")

    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
