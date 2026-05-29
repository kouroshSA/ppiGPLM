---
license: mit
library_name: pytorch
tags:
  - protein-protein-interaction
  - ppi
  - protein-language-model
  - gpt-2
  - nanogpt
  - character-level
  - trained-from-scratch
  - bioinformatics
  - biology
pipeline_tag: text-generation
---

# ppiGPLM

A GPT-2 small protein language model trained from scratch on protein-pair prompts and used for binary protein-protein interaction (PPI) classification via next-token prediction. The implementation is based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy, with character-level tokenization over amino acids.

![ppiGPLM](assets/ppiGPLM.png)

## Overview

ppiGPLM uses a GPT-2 small architecture (12 layers, 12 attention heads, 768 embedding dimensions) with character-level tokenization to predict whether two proteins interact. Rather than using a separate classification head, ppiGPLM frames PPI prediction as next-token prediction: given a structured prompt encoding a protein pair, the model predicts a binary label (`0` or `1`) as the next token. Softmax probabilities over the label tokens provide continuous interaction scores.

The model was developed for the *Prochlorococcus marinus* MED4 interactome, where it serves as one component of a tri-model consensus framework for computational PPI screening.

## Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | GPT-2 small |
| Layers | 12 |
| Attention heads | 12 |
| Embedding dimension | 768 |
| Context length | 4,096 tokens |
| Tokenization | Character-level (one token per amino acid) |
| Dropout | 0.2 |
| Optimizer | AdamW (lr = 5e-4, beta2 = 0.99) |
| Training iterations | 8,000 |

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- conda (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/kouroshSA/ppiGPLM.git
cd ppiGPLM

# Create a conda environment
conda create -n gpt python=3.10
conda activate gpt
pip install -r requirements.txt
```

## Repository Structure

```
ppiGPLM/
|-- model.py                          # GPT model definition
|-- train_.py                         # Training loop
|-- sample_fasta3.3_softmax_error_handling3e.py  # Batch inference script
|-- LES-wrapper.py                    # Learning Efficiency Score evaluation wrapper
|-- LES-wrapper.md                    # LES-wrapper documentation
|-- roc_analysis_color_threshold_F1e.py  # ROC curve analysis
|-- configurator.py                   # Configuration utility
|-- config/
|   |-- train_par_gpt2-s_scratch.py   # Training config (GPT-2 small, from scratch)
|   +-- finetune_label3.py            # Fine-tuning config
|-- data/
|   +-- MED4_char/                    # MED4 PPI dataset
|       |-- prepare.py                # Character-level tokenizer
|       +-- meta.pkl                  # Vocabulary (stoi/itos mappings)
|-- assets/
|   |-- ppiGPLM.png                  # ASCII workflow diagram
|   |-- tri_model_consensus.svg      # Tri-model consensus framework (SVG)
|   +-- tri_model_consensus.png      # Tri-model consensus framework (PNG)
|-- requirements.txt
|-- LICENSE
+-- README.md
```

## Usage

### Quick start: fetch the checkpoint from Hugging Face

The released MED4 checkpoint (`checkpoints/ppiGPLM_ckpt_7e.pt`, epoch ≈ 71)
lives on this Hugging Face repo. To pull it without cloning the GitHub
mirror:

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="kouroshSA/ppiGPLM",
    filename="checkpoints/ppiGPLM_ckpt_7e.pt",
)
meta_path = hf_hub_download(
    repo_id="kouroshSA/ppiGPLM",
    filename="data/MED4_char/meta.pkl",
)
```

`meta.pkl` carries the character vocabulary (`stoi`/`itos`) the inference
script needs to encode protein sequences.

#### Wiring the checkpoint into the inference script

`sample_fasta3.3_softmax_error_handling3e.py` loads from
`<model_dir>/ckpt.pt`, where `model_dir = 'out'` is set inline near the
top of the script and the trailing `ckpt.pt` filename is **hardcoded**.
Two ways to make the downloaded file work:

**Option A — place the file where the defaults expect it:**

```bash
mkdir -p out
cp /path/to/ppiGPLM_ckpt_7e.pt out/ckpt.pt   # note the required rename
```

then run the inference command below as-is.

**Option B — override `model_dir` via the poor-man's configurator
(`configurator.py`):**

```bash
mkdir -p my_ckpts
cp /path/to/ppiGPLM_ckpt_7e.pt my_ckpts/ckpt.pt   # still needs to be ckpt.pt
python sample_fasta3.3_softmax_error_handling3e.py \
    --input_file MED4-PPIs-low-confidence_ppiGPLM_prompts.csv \
    --output_dir ppi_results \
    --output_prefix my_predictions \
    --model_dir=my_ckpts
```

Either way, the on-disk filename must be `ckpt.pt`; editing it out of the
script is also possible (change the `model_dir` default near the top, or
the literal `'ckpt.pt'` further down) but the rename above is simpler.

The character vocabulary (`meta.pkl`) is read from
`data/<dataset>/meta.pkl`, where `<dataset>` comes from
`checkpoint['config']['dataset']` (`MED4_char` for this checkpoint). Make
sure that path exists — either keep the `data/MED4_char/` directory from
the GitHub clone, or place the downloaded `meta_path` there.

### Input file format

Each line of `--input_file` is one structured prompt (one protein pair),
not a free-form FASTA record. The schema is:

```
<ps1>,SEQ_A,<ps2>,SEQ_B,<l1>,LEN_A,<l2>,LEN_B,<l3>,<
```

- `<ps1>`, `<ps2>`: protein-sequence delimiter tokens
- `<l1>`, `<l2>`, `<l3>`: length-field delimiter tokens
- The trailing `,<` is **the cue**: it tells the model the next token to
  generate is the classification label (`1` = interacting, `0` = not).
  Don't omit it.

A ready-made example is shipped with the repo:
[`MED4-PPIs-low-confidence_ppiGPLM_prompts.csv`](MED4-PPIs-low-confidence_ppiGPLM_prompts.csv)
— inspect or copy its format when building your own input file.

### Batch Inference

Run inference on a file of prompts:

```bash
python sample_fasta3.3_softmax_error_handling3e.py \
    --input_file MED4-PPIs-low-confidence_ppiGPLM_prompts.csv \
    --output_dir ppi_results \
    --output_prefix my_predictions
```

This produces:
- `*_classifications.txt`: Full model output in FASTA-like format
- `*_probabilities.csv`: Per-pair probabilities for class 1 and class 0

#### About the inference script

`sample_fasta3.3_softmax_error_handling3e.py` is derived from Karpathy's
nanoGPT `sample.py` — it reuses the same `GPTConfig`/`GPT` classes from
`model.py`, the `init_from = 'resume'` checkpoint-loading idiom, and the
`_orig_mod.` prefix strip for `torch.compile`-wrapped state dicts. It is
**not** a drop-in copy, however. The modifications make it a batch
classifier rather than a generic sampler:

- batch input: one prompt per line read from `--input_file`, processed
  sequentially with no interactive loop;
- classifier-style output: per-prompt softmax probabilities of the next
  token being `"1"` vs `"0"`, written to `*_probabilities.csv` alongside
  the conventional `generate()` output dump in `*_classifications.txt`;
- robustness against real-world inputs: automatic block-size detection
  (`checkpoint['model_args']['block_size']` or
  `model.config.n_positions`), head-clipping when a prompt exceeds the
  context window so the trailing `<` label-cue token survives, and
  out-of-vocabulary character replacement (defaults to `A`).

The lineage is **GPT-2 → nanoGPT → ppiGPLM's batch-classifier sampler**.

### Training

#### Prepare data

```bash
python data/MED4_char/prepare.py
```

This creates `train.bin`, `val.bin`, and `meta.pkl` from the input training data.

#### Train the model

```bash
# Single GPU
python train_.py config/train_par_gpt2-s_scratch.py

# Multi-GPU (2 GPUs)
torchrun --standalone --nproc_per_node=2 train_.py config/train_par_gpt2-s_scratch.py
```

### Learning Efficiency Score (LES) Evaluation

The LES-wrapper automates evaluation across multiple training checkpoints, computing ROC-AUC, F1, and optimal threshold at each checkpoint and deriving integrated Learning Efficiency Scores:

```bash
python LES-wrapper.py \
    --checkpoint_dir out \
    --prs_file PRS.txt \
    --rrs_file RRS.txt \
    --output_dir LES_results \
    --vanilla
```

See [LES-wrapper.md](LES-wrapper.md) for full documentation.

### Standalone ROC Analysis

```bash
python roc_analysis_color_threshold_F1e.py \
    --prs_file ppi_results/PRS_probabilities.csv \
    --rrs_file ppi_results/RRS_probabilities.csv
```

## Architecture Diagrams

The ASCII workflow diagram (`assets/ppiGPLM.png`) covers:
- **A.** Prompt-based input strategy (character-level tokenization)
- **B.** Model architecture (GPT-2 small, causal self-attention)
- **C.** Training pipeline
- **D.** Inference pipeline with LES evaluation

> Note: the diagram lists "Flash Attention" — this path is taken automatically
> when running on PyTorch ≥ 2.0; older versions fall back to the manual
> scaled-dot-product implementation. Numerical results are equivalent.

See `assets/tri_model_consensus.svg` for the tri-model consensus framework with [ppiDCE](https://github.com/kouroshSA/ppiDCE) and [ppiBTEP](https://github.com/kouroshSA/ppiBTEP).

## Citation

If you use this software, please cite:

```
Daakour, S. et al. (2026).
```

This software is built on nanoGPT:

```
Karpathy, A. (2022). nanoGPT. https://github.com/karpathy/nanoGPT
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The original nanoGPT framework is by Andrej Karpathy (MIT License, 2022). Modifications and additions for protein-protein interaction prediction are by Kourosh Salehi-Ashtiani (MIT License, 2026).
