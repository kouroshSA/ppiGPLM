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

def plot_metric_trajectory(iterations, values, metric_name, output_path, les_value):
    """Plot a single metric trajectory across checkpoints."""
    plt.figure(figsize=(10, 6))

    # Filter out inf iterations
    valid_mask = [i < float('inf') for i in iterations]
    plot_iters = [it for it, v in zip(iterations, valid_mask) if v]
    plot_vals = [val for val, v in zip(values, valid_mask) if v]

    plt.plot(plot_iters, plot_vals, 'bo-', linewidth=2, markersize=8)
    plt.fill_between(plot_iters, plot_vals, alpha=0.3)

    plt.xlabel('Training Iteration', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.title(f'{metric_name} vs Training Iteration\nLES-{metric_name} = {les_value:.4f}', fontsize=14)
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
    axes[0].set_ylim([0, 1.05])

    # F1 plot
    axes[1].plot(plot_iters, plot_f1, 'go-', linewidth=2, markersize=6)
    axes[1].fill_between(plot_iters, plot_f1, alpha=0.3, color='green')
    axes[1].set_xlabel('Training Iteration')
    axes[1].set_ylabel('Best F1')
    axes[1].set_title(f'Best F1 Trajectory\nLES-F1 = {les_values["F1"]:.4f}')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=PUB_DPI)
    plt.savefig(os.path.splitext(output_path)[0] + '.pdf', format='pdf')
    plt.close()

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main():
    args, extra_args = parse_args()

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

        # Run ROC analysis
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
        auc_values.append(roc_auc)
        f1_values.append(best_f1)

    # Compute LES values
    print(f"\n{'='*60}")
    print("Computing Learning Efficiency Scores (LES)")
    print(f"{'='*60}")

    les_auc = compute_les(iterations, auc_values)
    les_f1 = compute_les(iterations, f1_values)

    les_values = {
        'AUC': les_auc,
        'F1': les_f1,
    }

    print(f"  LES-AUC: {les_auc:.6f}")
    print(f"  LES-F1: {les_f1:.6f}")

    # Generate trajectory plots
    if not args.no_plots and len(iterations) >= 2:
        print(f"\nGenerating trajectory plots...")

        # Individual plots
        plot_metric_trajectory(iterations, auc_values, 'AUC',
                              os.path.join(args.output_dir, 'trajectory_AUC.png'), les_auc)
        plot_metric_trajectory(iterations, f1_values, 'Best F1',
                              os.path.join(args.output_dir, 'trajectory_F1.png'), les_f1)

        # Combined plot
        plot_combined_trajectories(iterations, auc_values, f1_values,
                                  os.path.join(args.output_dir, 'trajectory_combined.png'), les_values)

        print(f"  Saved trajectory plots to {args.output_dir}")

    # Generate summary table
    print(f"\nGenerating summary table...")
    summary_csv = os.path.join(args.output_dir, 'summary_table.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['checkpoint', 'iteration', 'AUC', 'Best_F1',
                                               'PRS_samples', 'RRS_samples'])
        writer.writeheader()
        writer.writerows(results)

    # Add LES row
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
        'LES': {
            'AUC': les_auc,
            'F1': les_f1
        },
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
    print(f"\nLearning Efficiency Scores (LES):")
    print(f"  LES-AUC:       {les_auc:.6f}")
    print(f"  LES-F1:        {les_f1:.6f}")

    if results:
        final_result = results[-1]
        print(f"\nFinal Checkpoint Performance:")
        print(f"  AUC:       {final_result['AUC']:.4f}")
        print(f"  Best F1:   {final_result['Best_F1']:.4f}")

    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
