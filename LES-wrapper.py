#!/usr/bin/env python
"""
LES-wrapper.py — Learning Efficiency Score evaluation across training checkpoints.

Runs inference on PRS (Positive Reference Set) and RRS (Random Reference Set)
prompts at every saved checkpoint in a directory, computes ROC-AUC, optimal-F1
threshold, and Best-F1 at each checkpoint, then integrates these into a single
Learning Efficiency Score (LES) per metric — the area under the
metric-vs-iteration curve.

Note: This script supports both vanilla GPT checkpoints (use --vanilla) and
HOPE/Titan checkpoints (the --use_titan_in_forward, --enable_surprise_updates,
--adapt_mode, --teach_* flags). When evaluating ppiGPLM models, use --vanilla;
the HOPE-specific flags are no-ops for vanilla checkpoints.

Basic usage:
    python LES-wrapper.py \\
        --checkpoint_dir <dir> \\
        --prs_file <prs.txt> \\
        --rrs_file <rrs.txt> \\
        --output_dir <out> \\
        --vanilla
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
# Parse command-line arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='LES-wrapper: Learning Efficiency Score evaluation across checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python LES-wrapper.py --checkpoint_dir out-model --prs_file prs.txt --rrs_file rrs.txt --output_dir results
  python LES-wrapper.py --checkpoint_dir out-model --prs_file prs.txt --rrs_file rrs.txt --use_titan_in_forward=1
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

    # Titan/HOPE arguments (passed to sample script)
    parser.add_argument('--use_titan_in_forward', type=int, default=-1,
                        help='Override use_titan_in_forward (-1=use checkpoint value)')
    parser.add_argument('--enable_surprise_updates', type=int, default=0,
                        help='Enable Titan surprise updates (0/1)')
    parser.add_argument('--surprise_update_in_eval', type=int, default=0,
                        help='Allow memory updates during eval (0/1)')
    parser.add_argument('--adapt_mode', type=str, default='none',
                        choices=['none', 'prefix'], help='Adaptation mode')
    parser.add_argument('--adapt_steps', type=int, default=0,
                        help='Number of adaptation steps (0=disabled). When --teach_file is provided, this means teaching epochs.')

    # Memory state
    parser.add_argument('--memory_state_in', type=str, default='',
                        help='Path to load memory-only state file')

    # Teaching file arguments
    parser.add_argument('--teach_file', type=str, default='',
                        help='Path to teaching CSV file for supervised adaptation')
    parser.add_argument('--teach_delim', type=str, default='|',
                        help='Delimiter for teaching CSV (default: |)')
    parser.add_argument('--teach_has_header', type=int, default=1,
                        help='Whether teaching CSV has header row (default: 1)')
    parser.add_argument('--teach_reset_policy', type=str, default='pair',
                        choices=['pair', 'file', 'none'],
                        help='Memory reset policy during teaching')
    parser.add_argument('--teach_shuffle', type=int, default=1,
                        help='Shuffle teaching examples each epoch (default: 1)')
    parser.add_argument('--teach_max_rows', type=int, default=0,
                        help='Limit number of teaching rows loaded (0=all)')

    # Parallel processing
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip inference step (use existing probability files)')

    # Plotting options
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating trajectory plots')

    # Vanilla mode (use standard GPT model without HOPE features)
    parser.add_argument('--vanilla', action='store_true',
                        help='Use vanilla GPT model (no HOPE/CMS/Titan features)')

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
                  output_prefix, extra_args, vanilla=False):
    """Run inference using the sample script.

    Both vanilla and HOPE models use the same sample script (sample_fasta3.3_softmax_error_handling3e_hope_v3.py)
    since models trained with train_hope_v3.py --vanilla use the same checkpoint format.
    The vanilla flag just controls whether HOPE-specific args are passed.
    """
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

def run_roc_analysis_internal(combined_csv_path, output_plot_path):
    """Run ROC analysis and return metrics (internal implementation)."""
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
        return None, None, None

    # Assign labels (PRS = 1, RRS = 0)
    prs_labels = [1] * len(prs_probs)
    rrs_labels = [0] * len(rrs_probs)

    probs = np.array(prs_probs + rrs_probs)
    labels = np.array(prs_labels + rrs_labels)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Filter valid thresholds
    finite_idxs = np.where(np.isfinite(thresholds))[0]
    fpr = fpr[finite_idxs]
    tpr = tpr[finite_idxs]
    thresholds = thresholds[finite_idxs]

    valid_thresholds_idxs = np.where((thresholds >= 0) & (thresholds <= 1))[0]
    fpr = fpr[valid_thresholds_idxs]
    tpr = tpr[valid_thresholds_idxs]
    thresholds = thresholds[valid_thresholds_idxs]

    # Compute best F1 score
    best_f1 = -1.0
    best_thresh = None
    for thresh in thresholds:
        predicted_labels = (probs >= thresh).astype(int)
        current_f1 = f1_score(labels, predicted_labels)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh

    # Generate ROC plot using figure and axes approach (like original roc.py)
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams['font.family'] = 'DejaVu Sans'  # More portable than Arial

    # Color-coded by threshold
    norm = plt.Normalize(vmin=thresholds.min(), vmax=thresholds.max())
    cmap = plt.cm.viridis

    for i in range(len(fpr) - 1):
        x = fpr[i:i + 2]
        y = tpr[i:i + 2]
        z = thresholds[i]
        ax.plot(x, y, color=cmap(norm(z)), lw=2.5)

    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Threshold', fontsize=14)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve', fontsize=16)

    legend_text = f'AUC = {roc_auc:.3f}, Best F1 = {best_f1:.3f}, Threshold = {best_thresh:.3f}'
    ax.legend([legend_text], loc="lower right", fontsize=11)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150, format='png')
    plt.close(fig)

    return roc_auc, best_f1, best_thresh

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
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_combined_trajectories(iterations, auc_vals, f1_vals, thresh_vals, output_path, les_values):
    """Plot all metrics on a single figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Filter out inf iterations
    valid_mask = [i < float('inf') for i in iterations]
    plot_iters = [it for it, v in zip(iterations, valid_mask) if v]
    plot_auc = [val for val, v in zip(auc_vals, valid_mask) if v]
    plot_f1 = [val for val, v in zip(f1_vals, valid_mask) if v]
    plot_thresh = [val for val, v in zip(thresh_vals, valid_mask) if v]

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

    # Threshold plot
    axes[2].plot(plot_iters, plot_thresh, 'ro-', linewidth=2, markersize=6)
    axes[2].fill_between(plot_iters, plot_thresh, alpha=0.3, color='red')
    axes[2].set_xlabel('Training Iteration')
    axes[2].set_ylabel('Best F1 Threshold')
    axes[2].set_title(f'Threshold Trajectory\nLES-Threshold = {les_values["Threshold"]:.4f}')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main():
    args, extra_args = parse_args()

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

    # Find sample script
    # Note: Both vanilla and HOPE models trained with train_hope_v3.py use the same
    # checkpoint format, so we always use the HOPE sample script. For vanilla models,
    # we just skip the HOPE-specific arguments (Titan, surprise updates, etc.)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_script = os.path.join(script_dir, 'sample_fasta3.3_softmax_error_handling3e_hope_v3.py')
    model_type = "Vanilla GPT (trained with train_hope_v3.py)" if args.vanilla else "HOPE"

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

    # Build extra args for sample script (only for HOPE models)
    sample_extra_args = []
    if not args.vanilla:
        # HOPE-specific arguments
        if args.use_titan_in_forward >= 0:
            sample_extra_args.extend(['--use_titan_in_forward', str(args.use_titan_in_forward)])
        if args.enable_surprise_updates:
            sample_extra_args.extend(['--enable_surprise_updates', str(args.enable_surprise_updates)])
        if args.surprise_update_in_eval:
            sample_extra_args.extend(['--surprise_update_in_eval', str(args.surprise_update_in_eval)])
        if args.adapt_mode != 'none':
            sample_extra_args.extend(['--adapt_mode', args.adapt_mode])
        if args.adapt_steps > 0:
            sample_extra_args.extend(['--adapt_steps', str(args.adapt_steps)])
        if args.memory_state_in:
            sample_extra_args.extend(['--memory_state_in', args.memory_state_in])

        # Teaching file arguments
        if args.teach_file:
            sample_extra_args.extend(['--teach_file', args.teach_file])
            sample_extra_args.extend(['--teach_delim', args.teach_delim])
            sample_extra_args.extend(['--teach_has_header', str(args.teach_has_header)])
            sample_extra_args.extend(['--teach_reset_policy', args.teach_reset_policy])
            sample_extra_args.extend(['--teach_shuffle', str(args.teach_shuffle)])
            if args.teach_max_rows > 0:
                sample_extra_args.extend(['--teach_max_rows', str(args.teach_max_rows)])

    # Add any extra arguments passed through
    sample_extra_args.extend(extra_args)

    # Results storage
    results = []
    iterations = []
    auc_values = []
    f1_values = []
    thresh_values = []

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
                               args.prs_file, ckpt_subdir, prs_prefix, sample_extra_args,
                               vanilla=args.vanilla):
                print(f"  SKIPPING checkpoint due to inference error")
                continue

            # Run inference for RRS
            print(f"  Running RRS inference...")
            if not run_inference(sample_script, args.checkpoint_dir, ckpt_name,
                               args.rrs_file, ckpt_subdir, rrs_prefix, sample_extra_args,
                               vanilla=args.vanilla):
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
        roc_auc, best_f1, best_thresh = run_roc_analysis_internal(combined_csv, roc_plot)

        if roc_auc is None:
            print(f"  WARNING: ROC analysis failed, skipping")
            continue

        print(f"  Results: AUC={roc_auc:.4f}, F1={best_f1:.4f}, Threshold={best_thresh:.4f}")

        # Store results
        results.append({
            'checkpoint': ckpt_name,
            'iteration': iteration if iteration < float('inf') else 'final',
            'AUC': roc_auc,
            'Best_F1': best_f1,
            'Best_F1_Threshold': best_thresh,
            'PRS_samples': len(prs_probs),
            'RRS_samples': len(rrs_probs)
        })

        iterations.append(iteration)
        auc_values.append(roc_auc)
        f1_values.append(best_f1)
        thresh_values.append(best_thresh)

    # Compute LES values
    print(f"\n{'='*60}")
    print("Computing Learning Efficiency Scores (LES)")
    print(f"{'='*60}")

    les_auc = compute_les(iterations, auc_values)
    les_f1 = compute_les(iterations, f1_values)
    les_thresh = compute_les(iterations, thresh_values)

    les_values = {
        'AUC': les_auc,
        'F1': les_f1,
        'Threshold': les_thresh
    }

    print(f"  LES-AUC: {les_auc:.6f}")
    print(f"  LES-F1: {les_f1:.6f}")
    print(f"  LES-Threshold: {les_thresh:.6f}")

    # Generate trajectory plots
    if not args.no_plots and len(iterations) >= 2:
        print(f"\nGenerating trajectory plots...")

        # Individual plots
        plot_metric_trajectory(iterations, auc_values, 'AUC',
                              os.path.join(args.output_dir, 'trajectory_AUC.png'), les_auc)
        plot_metric_trajectory(iterations, f1_values, 'Best F1',
                              os.path.join(args.output_dir, 'trajectory_F1.png'), les_f1)
        plot_metric_trajectory(iterations, thresh_values, 'Best F1 Threshold',
                              os.path.join(args.output_dir, 'trajectory_Threshold.png'), les_thresh)

        # Combined plot
        plot_combined_trajectories(iterations, auc_values, f1_values, thresh_values,
                                  os.path.join(args.output_dir, 'trajectory_combined.png'), les_values)

        print(f"  Saved trajectory plots to {args.output_dir}")

    # Generate summary table
    print(f"\nGenerating summary table...")
    summary_csv = os.path.join(args.output_dir, 'summary_table.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['checkpoint', 'iteration', 'AUC', 'Best_F1',
                                               'Best_F1_Threshold', 'PRS_samples', 'RRS_samples'])
        writer.writeheader()
        writer.writerows(results)

    # Add LES row
    with open(summary_csv, 'a', newline='') as f:
        f.write(f"\nLES (Learning Efficiency Score),---,{les_auc:.6f},{les_f1:.6f},{les_thresh:.6f},---,---\n")

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
            'F1': les_f1,
            'Threshold': les_thresh
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
    print(f"  LES-Threshold: {les_thresh:.6f}")

    if results:
        final_result = results[-1]
        print(f"\nFinal Checkpoint Performance:")
        print(f"  AUC:       {final_result['AUC']:.4f}")
        print(f"  Best F1:   {final_result['Best_F1']:.4f}")
        print(f"  Threshold: {final_result['Best_F1_Threshold']:.4f}")

    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
