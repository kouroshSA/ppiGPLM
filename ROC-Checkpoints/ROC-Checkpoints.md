# ppiGPLM ROC / LES trajectory checkpoints

**Per-iteration training checkpoints used to trace the model's classification
performance over training — the inputs to the Learning Efficiency Score (LES)
and ROC analysis produced by `LES-wrapper.py`.**

Unlike [`checkpoints/`](../checkpoints/), which holds the single best checkpoint
used for *screening* the MED4 interactome, this set captures the **full training
trajectory** so that ROC-AUC, Best-F1, and the optimal-F1 threshold can be
computed at every saved iteration and integrated into a single LES per metric
(the area under the metric-vs-iteration curve).

## Provenance

| | |
|---|---|
| Model | ppiGPLM (GPT-2 small, character-level tokenization, trained from scratch on protein-pair prompts) |
| Architecture | 12 layers, 12 attention heads, 768 emb dim, 4,096 token context |
| Mode | Vanilla GPT (no HOPE / Titan / CMS) |
| Training run | `out_ppiGPLM_Med4-solo_hope.v3-8ki_4k_768_vanilla` (max_iters 8000, lr 5e-4, batch 12, dropout 0.2, bf16) |
| Training data | MED4 PPI dataset (`data/MED4_char/`) — prepared by `data/MED4_char/prepare.py` |
| Tokenization | Per-amino-acid characters (one token per residue), prompts wrapped with `<ps1>`, `<ps2>`, `<l1>`, `<l2>`, `<l3>` delimiter tokens |
| Total parameters | 85,078,272 (~85 M) |

## The checkpoint set

| | |
|---|---|
| Iterations | `ckpt_250.pt`, `ckpt_500.pt`, … `ckpt_8000.pt` — every 250 iterations |
| Count | 32 per-iteration checkpoints + `ckpt.pt` (final, == iteration 8000) |
| File size | ~1.06 GB each |
| Final loss (iter 8000) | train 0.0202 / val 0.0821 |
| Naming | nanoGPT-style `ckpt_<iter>.pt`; `LES-wrapper.py` parses the iteration from the filename for the x-axis of every LES curve |

## Download (Hugging Face)

These trajectory checkpoints are large (~34 GB total, 33 × ~1.06 GB) and are
hosted on Hugging Face rather than in this Git repository. They live in the
`ROC-Checkpoints/` folder of the main model repo
[kouroshSA/ppiGPLM](https://huggingface.co/kouroshSA/ppiGPLM/tree/main/ROC-Checkpoints).

```bash
# Download just the ROC-Checkpoints folder into ./ROC-Checkpoints
hf download kouroshSA/ppiGPLM --repo-type model \
    --include "ROC-Checkpoints/*" --local-dir .
```

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="kouroshSA/ppiGPLM",
    repo_type="model",
    allow_patterns="ROC-Checkpoints/*",
    local_dir=".",
)
```

Related Hugging Face resources:
[ppiGPLM](https://huggingface.co/kouroshSA/ppiGPLM) ·
[ppiBTEP](https://huggingface.co/kouroshSA/ppiBTEP) ·
[ppiDCE](https://huggingface.co/kouroshSA/ppiDCE)

## Intended use

Reproducing the LES / ROC analysis for ppiGPLM. Point `LES-wrapper.py` at this
folder and supply a PRS (Positive Reference Set) and RRS (Random Reference Set):

```bash
python LES-wrapper.py \
    --checkpoint_dir ROC-Checkpoints \
    --prs_file MED4_100_PRS.csv \
    --rrs_file MED4_100_RRS.csv \
    --output_dir LES_results
```

The wrapper runs inference at each checkpoint with
`sample_fasta3.3_softmax_error_handling3f.py` (the vanilla sampler), computes
ROC-AUC / Best-F1 / optimal threshold per checkpoint, and integrates them into
LES values, emitting publication-quality (600 dpi PNG + vector PDF) ROC and
trajectory plots. The vocabulary (`stoi` / `itos`) is read from
`data/MED4_char/meta.pkl`.
