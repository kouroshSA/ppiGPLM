# LES-wrapper: Learning Efficiency Score Evaluation

## Overview

The **LES-wrapper** automates the evaluation of model trainability across multiple
training checkpoints. It runs inference on PRS (Positive Reference Set) and RRS
(Random Reference Set) datasets at each checkpoint, computes ROC metrics, and
derives integrated learning efficiency scores.

## What is LES?

LES (Learning Efficiency Score) is defined as the **area under the metric-vs-iteration
curve**. Unlike metrics that measure only final performance, LES captures the entire
learning trajectory:

- **LES-AUC**: Area under the AUC trajectory curve
- **LES-F1**: Area under the Best-F1 trajectory curve
- **LES-Threshold**: Area under the optimal-threshold trajectory curve

Higher LES values indicate faster learning, better overall performance across training,
and more efficient use of training iterations.

## Workflow

For each checkpoint the wrapper:

1. Runs inference on PRS and RRS prompt files
2. Extracts softmax probabilities for the positive class
3. Combines probabilities into a single file for ROC analysis
4. Computes AUC, Best-F1, and optimal threshold
5. Generates a color-coded ROC curve plot
6. Aggregates results into a summary table
7. Plots metric trajectories across checkpoints
8. Computes LES values for each metric

## Installation

```bash
conda activate gpt
pip install scikit-learn matplotlib numpy
```

## Basic Usage

```bash
python LES-wrapper.py \
    --checkpoint_dir out_ppiGPLM_MED4 \
    --prs_file MED4_Int_100pairs_prompts.txt \
    --rrs_file MED4_100_RND_prompts.txt \
    --output_dir LES_results_MED4 \
    --vanilla
```

The `--vanilla` flag is required when evaluating standard ppiGPLM checkpoints
(i.e., checkpoints trained with `train_.py` rather than a HOPE-variant trainer).

## Common Patterns

### Selecting Specific Checkpoints

```bash
# Only checkpoints at iterations 1000, 2000, and 5000
python LES-wrapper.py \
    --checkpoint_dir out_ppiGPLM_MED4 \
    --prs_file prs.txt \
    --rrs_file rrs.txt \
    --output_dir results \
    --checkpoint_pattern "ckpt_[125]000.pt" \
    --vanilla

# Every 5000 iterations
python LES-wrapper.py \
    --checkpoint_dir out_ppiGPLM_MED4 \
    --prs_file prs.txt \
    --rrs_file rrs.txt \
    --output_dir results \
    --checkpoint_pattern "ckpt_*000.pt" \
    --vanilla
```

### Skipping Inference (Re-computing Metrics Only)

If you have already run inference and just want to recompute metrics or plots:

```bash
python LES-wrapper.py \
    --checkpoint_dir out_ppiGPLM_MED4 \
    --prs_file MED4_Int_100pairs_prompts.txt \
    --rrs_file MED4_100_RND_prompts.txt \
    --output_dir LES_results_MED4 \
    --skip_inference \
    --vanilla
```

### Plot Customization

Use `--no_plots` to skip trajectory figure generation when you only need the
summary CSV.

## Command-Line Arguments (Vanilla Flags)

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_dir` | *(required)* | Directory containing checkpoint files (`ckpt_*.pt`) |
| `--prs_file` | *(required)* | Path to Positive Reference Set prompts file |
| `--rrs_file` | *(required)* | Path to Random Reference Set prompts file |
| `--output_dir` | `LES_results` | Directory for all output files |
| `--checkpoint_pattern` | `ckpt_*.pt` | Glob pattern to select checkpoints |
| `--include_final` | False | Also evaluate `ckpt.pt` (the final checkpoint) |
| `--no_plots` | False | Skip generating trajectory plots |
| `--skip_inference` | False | Skip inference; reuse existing probability files |
| `--vanilla` | False | Use standard GPT checkpoint format (required for ppiGPLM) |

## Output Structure

```
LES_results/
├── ckpt_1000/
│   ├── PRS_iter1000_probabilities.csv
│   ├── PRS_iter1000_classifications.txt
│   ├── RRS_iter1000_probabilities.csv
│   ├── RRS_iter1000_classifications.txt
│   ├── combined_probabilities_iter1000.csv
│   ├── ROC_iter1000.png
│   └── inference_log.md
├── ckpt_2000/ ...
├── trajectory_AUC.png
├── trajectory_F1.png
├── trajectory_Threshold.png
├── trajectory_combined.png
├── summary_table.csv
└── manifest.json
```

`summary_table.csv` contains per-checkpoint metrics plus a final row with the
integrated LES values. `manifest.json` records complete run metadata.

---

## Appendix: HOPE/Titan Checkpoint Support (not used for vanilla ppiGPLM)

The script also supports checkpoints from HOPE/Titan-variant trainers. These flags
are no-ops when `--vanilla` is passed, so they do not affect standard ppiGPLM runs.

| Flag | Description |
|------|-------------|
| `--use_titan_in_forward` | Override Titan memory-in-forward flag (-1 = use checkpoint value) |
| `--enable_surprise_updates` | Enable Titan surprise-based memory updates (0/1) |
| `--surprise_update_in_eval` | Allow memory updates during evaluation (0/1) |
| `--adapt_mode` | Prefix-adaptation mode: `none` or `prefix` |
| `--adapt_steps` | Number of adaptation steps or teaching epochs |
| `--memory_state_in` | Path to a saved memory-only state file |
| `--teach_file` | CSV of supervised teaching pairs for pre-evaluation conditioning |
| `--teach_delim` | Delimiter for the teaching CSV (default `\|`) |
| `--teach_has_header` | Whether the teaching CSV has a header row (0/1) |
| `--teach_reset_policy` | Memory reset policy during teaching: `pair`, `file`, or `none` |
| `--teach_shuffle` | Shuffle teaching examples each epoch (0/1) |
| `--teach_max_rows` | Limit teaching rows; 0 = use all |

These flags are relevant to the HOPE project (if/when it has its own public repo).
