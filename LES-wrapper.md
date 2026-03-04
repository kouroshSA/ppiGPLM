# LES-wrapper: Learning Efficiency Score Evaluation Tool

## Overview

The **LES-wrapper** (Learning Efficiency Score wrapper) is a Python tool that automates the evaluation of model trainability across multiple training checkpoints. It provides a comprehensive pipeline for assessing how well a model learns over time by computing ROC metrics at each checkpoint and deriving integrated learning efficiency scores.

## What is LES (Learning Efficiency Score)?

LES is defined as the **area under the metric-vs-iteration curve**. Unlike traditional metrics that only measure final model performance, LES captures the entire learning trajectory, summarizing:

- **Trainability**: How quickly the model improves
- **Stability**: How consistently the model learns
- **Learning efficiency**: Overall learning behavior across training

Three LES metrics are computed:
- **LES-AUC**: Area under the AUC trajectory curve
- **LES-F1**: Area under the Best F1 trajectory curve
- **LES-Threshold**: Area under the optimal threshold trajectory curve

## Workflow

The wrapper performs the following steps for each checkpoint:

1. **Run inference** on PRS (Positive Reference Set) and RRS (Random Reference Set) datasets
2. **Extract probabilities** for the positive class from inference outputs
3. **Combine probabilities** into a single file for ROC analysis
4. **Run ROC analysis** to compute AUC, Best F1, and optimal threshold
5. **Generate ROC curve plots** with color-coded thresholds
6. **Aggregate results** into a summary table
7. **Plot metric trajectories** across checkpoints
8. **Compute LES values** for each metric

## Installation

Ensure the required dependencies are installed in your conda environment:

```bash
conda activate hope
pip install scikit-learn matplotlib numpy
```

## Usage

### Basic Usage

```bash
python LES-wrapper.py \
    --checkpoint_dir <path_to_checkpoints> \
    --prs_file <prs_prompts.txt> \
    --rrs_file <rrs_prompts.txt> \
    --output_dir <output_directory>
```

### Full Example

```bash
python LES-wrapper.py \
    --checkpoint_dir out-hope-v2 \
    --prs_file MED4_Int_100pairs_prompts.txt \
    --rrs_file MED4_100_RND_prompts.txt \
    --output_dir LES_results_MED4 \
    --checkpoint_pattern "ckpt_*.pt" \
    --include_final
```

### For Vanilla GPT Models (No HOPE Features)

Use the `--vanilla` flag to evaluate models trained with `train_hope_v3.py --vanilla` (standard GPT without HOPE/CMS/Titan features):

```bash
python LES-wrapper.py \
    --checkpoint_dir out_ppiGPLM_MED4_vanilla-40ki_1k_384 \
    --prs_file MED4_100_PRS.txt \
    --rrs_file MED4_100_RRS.txt \
    --output_dir LES_results_vanilla \
    --vanilla
```

**Note**: The sample script auto-detects vanilla vs HOPE checkpoints based on the saved model_args. The `--vanilla` flag ensures HOPE-specific arguments are not passed to the sample script.

### With HOPE/Titan Features Enabled

```bash
python LES-wrapper.py \
    --checkpoint_dir out-ppiGPLM_MED4_hope-60ki_1k_384_ful \
    --prs_file MED4_Int_100pairs_prompts.txt \
    --rrs_file MED4_100_RND_prompts.txt \
    --output_dir LES_results_Titan \
    --use_titan_in_forward=1 \
    --enable_surprise_updates=1
```

### With Prefix Adaptation (In-Context Learning)

```bash
python LES-wrapper.py \
    --checkpoint_dir out-hope-v2 \
    --prs_file MED4_Int_100pairs_prompts.txt \
    --rrs_file MED4_100_RND_prompts.txt \
    --output_dir LES_results_adapted \
    --use_titan_in_forward=1 \
    --enable_surprise_updates=1 \
    --adapt_mode prefix \
    --adapt_steps 3
```

### With Teaching File (Supervised Pre-Training)

```bash
python LES-wrapper.py \
    --checkpoint_dir out-hope-v2 \
    --prs_file MED4_PRS_100pairs.txt \
    --rrs_file MED4_RRS_100pairs.txt \
    --output_dir LES_results_teaching \
    --use_titan_in_forward=1 \
    --enable_surprise_updates=1 \
    --teach_file teaching_pairs.csv \
    --teach_delim "|" \
    --teach_has_header 1 \
    --teach_reset_policy pair \
    --teach_shuffle 1 \
    --adapt_steps 2
```

**Teaching file format** (teaching_pairs.csv):
```csv
protein1|protein2|interaction
<ps1>,ATCG...,<ps2>,GCTA...,<1>
<ps1>,TTAG...,<ps2>,CCGA...,<0>
```
- Each row must end with `<1>` or `<0>` label token
- Teaching happens before each checkpoint evaluation
- Memory state carries into PRS/RRS evaluation

### Skip Inference (Use Existing Outputs)

If you've already run inference and just want to recompute metrics:

```bash
python LES-wrapper.py \
    --checkpoint_dir out-hope-v2 \
    --prs_file MED4_Int_100pairs_prompts.txt \
    --rrs_file MED4_100_RND_prompts.txt \
    --output_dir LES_results \
    --skip_inference
```

### Select Specific Checkpoints

Use glob patterns to select specific checkpoints:

```bash
# Only checkpoints 1000, 2000, 5000
python LES-wrapper.py \
    --checkpoint_dir out-hope-v2 \
    --prs_file prs.txt \
    --rrs_file rrs.txt \
    --output_dir results \
    --checkpoint_pattern "ckpt_[125]000.pt"

# Every 5000 iterations
python LES-wrapper.py \
    --checkpoint_dir out-hope-v2 \
    --prs_file prs.txt \
    --rrs_file rrs.txt \
    --output_dir results \
    --checkpoint_pattern "ckpt_*000.pt"
```

## Command-Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--checkpoint_dir` | Directory containing model checkpoints (ckpt_*.pt files) |
| `--prs_file` | Path to Positive Reference Set prompts file |
| `--rrs_file` | Path to Random Reference Set prompts file |

### Output Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `LES_results` | Directory to save all outputs |
| `--checkpoint_pattern` | `ckpt_*.pt` | Glob pattern to match checkpoint files |
| `--include_final` | False | Also include ckpt.pt (final checkpoint) if present |
| `--no_plots` | False | Skip generating trajectory plots |
| `--skip_inference` | False | Skip inference step (use existing probability files) |

### Model Type

| Argument | Default | Description |
|----------|---------|-------------|
| `--vanilla` | False | Use vanilla GPT model (no HOPE/CMS/Titan features) |

### HOPE/Titan Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_titan_in_forward` | -1 | Override use_titan_in_forward (-1=use checkpoint value) |
| `--enable_surprise_updates` | 0 | Enable Titan surprise updates (0/1) |
| `--surprise_update_in_eval` | 0 | Allow memory updates during eval (0/1) |
| `--adapt_mode` | `none` | Adaptation mode: `none` or `prefix` |
| `--adapt_steps` | 0 | Number of teaching epochs (with --teach_file) or adaptation steps |
| `--memory_state_in` | `` | Path to load memory-only state file |

### Teaching File Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--teach_file` | `` | Path to teaching CSV file for supervised pre-training |
| `--teach_delim` | `\|` | Delimiter for teaching CSV |
| `--teach_has_header` | 1 | Whether teaching CSV has header row (0/1) |
| `--teach_reset_policy` | `pair` | Memory reset during teaching: `pair`, `file`, `none` |
| `--teach_shuffle` | 1 | Shuffle teaching examples each epoch (0/1) |
| `--teach_max_rows` | 0 | Limit number of teaching rows (0=all) |

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
├── ckpt_2000/
│   └── ... (same structure)
├── ckpt_5000/
│   └── ... (same structure)
├── trajectory_AUC.png
├── trajectory_F1.png
├── trajectory_Threshold.png
├── trajectory_combined.png
├── summary_table.csv
└── manifest.json
```

## Output Files

### summary_table.csv

Contains per-checkpoint metrics and LES values:

```csv
checkpoint,iteration,AUC,Best_F1,Best_F1_Threshold,PRS_samples,RRS_samples
ckpt_1000.pt,1000,0.7731,0.7094,0.2812,101,101
ckpt_2000.pt,2000,0.7914,0.7326,0.0331,101,101
ckpt_5000.pt,5000,0.8701,0.8019,0.0683,101,101

LES (Learning Efficiency Score),---,0.818621,0.755662,0.077289,---,---
```

### manifest.json

JSON file with complete run metadata:

```json
{
  "timestamp": "2024-12-25T19:10:00",
  "checkpoint_dir": "out-hope-v2",
  "prs_file": "MED4_Int_100pairs_prompts.txt",
  "rrs_file": "MED4_100_RND_prompts.txt",
  "output_dir": "LES_results",
  "num_checkpoints": 3,
  "num_successful": 3,
  "LES": {
    "AUC": 0.818621,
    "F1": 0.755662,
    "Threshold": 0.077289
  },
  "results": [...]
}
```

### Trajectory Plots

- **trajectory_AUC.png**: AUC vs training iteration with LES annotation
- **trajectory_F1.png**: Best F1 vs training iteration with LES annotation
- **trajectory_Threshold.png**: Optimal threshold vs training iteration
- **trajectory_combined.png**: All three metrics in a single figure

### Per-Checkpoint ROC Plots

Each checkpoint subdirectory contains a color-coded ROC curve (`ROC_iter*.png`) where the curve color represents the classification threshold, making it easy to identify optimal operating points.

## Interpreting Results

### LES Values

LES values range from 0 to 1 (when metrics are normalized). Higher LES indicates:
- Faster learning (quick improvement early in training)
- Better overall performance across training
- More efficient use of training iterations

### Comparing Models

To compare different model variants (e.g., vanilla GPT-2 vs CMS vs CMS+Titan):

1. Run LES-wrapper on each model's checkpoint directory
2. Compare the final LES values
3. Examine trajectory plots to understand learning dynamics

Example comparison:
| Model Variant | LES-AUC | LES-F1 | Final AUC |
|--------------|---------|--------|-----------|
| Vanilla GPT-2 | 0.72 | 0.68 | 0.81 |
| CMS | 0.78 | 0.73 | 0.85 |
| CMS+Titan | 0.82 | 0.76 | 0.87 |
| CMS+Titan+Surprise | 0.85 | 0.79 | 0.89 |

## Troubleshooting

### "No checkpoints found"

Ensure your checkpoint pattern matches the files in the directory:
```bash
ls out-hope-v2/*.pt  # Check available checkpoints
```

### "Could not extract probabilities"

The inference output CSV format may have changed. Check that the probability columns are correctly positioned in the output files.

### Memory issues

For large datasets or many checkpoints, consider:
- Processing checkpoints in batches
- Using `--skip_inference` after initial run
- Reducing the number of samples in PRS/RRS files

## Dependencies

- Python 3.8+
- numpy
- matplotlib
- scikit-learn
- PyTorch (for inference)

## Related Files

- `sample_fasta3.3_softmax_error_handling3e_hope_v3.py`: Inference script used by the wrapper
- `roc.py`: Standalone ROC analysis script (wrapper has built-in implementation)
- `train_hope_v3.py`: Training script that generates checkpoints

## Author

Developed for the HOPE (Hierarchical Optimization with Persistent Embeddings) architecture evaluation pipeline.
