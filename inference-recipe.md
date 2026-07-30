# ppiGPLM V3 inference / LES recipe

How to score trained ppiGPLM checkpoints on the PRS/RRS reference sets and produce
the Learning-Efficiency-Score (LES) analysis. ppiGPLM is a character-level GPT-2
(nanoGPT) that reads a **joint prompt** `<ps1>,SEQ1,<ps2>,SEQ2,<` and predicts the
interaction token; the LES x-axis is the training **iteration** (checkpoints
`ckpt_<iter>.pt`, e.g. 500…6000).

## Run LES on a checkpoint directory

```bash
# MKL/libgomp clash on this machine: set the threading layer first.
export MKL_THREADING_LAYER=GNU
PY=/home/ksa/anaconda3/envs/gpt/bin/python
$PY LES-wrapper_v2.py \
    --checkpoint_dir out_V3-1_6k \
    --prs_file V3_PRS-RRS/PRS-RRS/PRS-V3-1.csv \
    --rrs_file V3_PRS-RRS/PRS-RRS/RRS-V3-1.csv \
    --output_dir LES_results_V3-1 \
    --checkpoint_pattern 'ckpt_*.pt' --include_final
```

`LES-wrapper_v2.py` runs the vanilla-GPT sampler
(`sample_fasta3.3_softmax_error_handling3f.py`) per checkpoint; unrecognized flags
are forwarded to it. Unlike the ESM models there is no `--num_layers` /
`--max_length` to match — the model config lives in the checkpoint. The
positive-class probability is `Probability_of_1` (the joint prompt itself contains
commas, so it is read as the second-to-last column).

Per checkpoint it writes `ckpt_<iter>/` with the probability CSVs, a PRS-vs-RRS
probability-distribution **violin** (`prob_dist_iter<iter>.png`) and (for real
positive sets) `ROC_iter<iter>.png`; across checkpoints it writes the AUC / Best-F1
trajectories, `summary_prob_distributions*` violins, `summary_table.csv`
(per-iteration AUC/Best-F1 + an LES row), and `manifest.json`.

## Reference-set conditions

`V3_PRS-RRS/` provides, per replicate (ppiGPLM 5-column format
`<ps1>,SEQ1,<ps2>,SEQ2,<`):

| condition | PRS / RRS files |
|---|---|
| regular | `PRS-RRS/{PRS,RRS}-V3-k.csv` |
| homodimer-depleted | `PRS-RRS_no_homodimers/{PRS,RRS}-V3-k.csv` |
| homodimers-only | `PRS-RRS_homodimers_only/{PRS,RRS}-V3-k.csv` |
| ps1 / ps2 / ps1-ps2 random | `random_controls/{PRS,RRS}-V3-k_{ps1,ps2,ps1-ps2}_random.csv` |

## Random controls: AUC-ROC and Best-F1 are excluded

For the random-substituted controls (`ps1_random`, `ps2_random`,
`ps1-ps2_random`) **neither** reference file contains true positives — both the
"PRS" and the "RRS" are random pairs — so ROC-AUC and Best-F1, which measure
positive-vs-negative ranking, are **not meaningful** and are **not reported**.

`LES-wrapper_v2.py` handles this automatically:

- It **auto-enables `--no_metrics`** when the PRS/RRS filenames contain `random`
  (or pass `--no_metrics` explicitly).
- In that mode it **skips ROC-AUC, Best-F1, the LES integral, the ROC plots, and
  the AUC/F1 trajectory plots**; the `summary_table.csv` AUC/Best_F1 columns are
  left blank (no LES row), and `manifest.json` records `"metrics_excluded": true`.
- It **still produces** the per-checkpoint and summary **probability-distribution
  violins** and the raw probability CSVs — the meaningful read-out for a control.

The inference/sampler script needs no change — the metric exclusion lives in the
LES-wrapper, which is where AUC/F1/LES are computed.
