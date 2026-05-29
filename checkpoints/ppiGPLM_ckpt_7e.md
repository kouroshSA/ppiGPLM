# ppiGPLM_ckpt_7e.pt

**Checkpoint used for screening low-confidence Y2H pairs in the *Prochlorococcus marinus* MED4 interactome.**

## Provenance

| | |
|---|---|
| Model | ppiGPLM (GPT-2 small, character-level tokenization, trained from scratch on protein-pair prompts) |
| Architecture | 12 layers, 12 attention heads, 768 emb dim, 4,096 token context |
| File size | ~1009 MB |
| Training run | `out_7e` |
| Training data | MED4 PPI dataset (`data/MED4_char/`) — prepared by `data/MED4_char/prepare.py` from the project's input CSV/TXT |
| Tokenization | Per-amino-acid characters (one token per residue), prompts wrapped with `<ps1>`, `<ps2>`, `<l1>`, `<l2>`, `<l3>` delimiter tokens |

## Intended use

Inference / screening of candidate MED4 protein–protein interactions that
were originally flagged as **low-confidence Y2H hits**. The model is prompted
with `<ps1>,Seq_A,<ps2>,Seq_B,<l1>,len_A,<l2>,len_B,<l3>` and the next-token
softmax over `0` / `1` is used (alongside the other tri-model components,
[ppiBTEP](https://huggingface.co/kouroshSA/ppiBTEP) and
[ppiDCE](https://huggingface.co/kouroshSA/ppiDCE)) to retain or discard the
pair.

## Loading

Use `sample_fasta3.3_softmax_error_handling3e.py` from the parent repo, or
load the checkpoint directly with `train_.py` / `model.py`. The vocabulary
(`stoi` / `itos` mappings) is read from `data/MED4_char/meta.pkl`.
