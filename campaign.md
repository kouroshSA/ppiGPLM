# ppiGPLM V3 campaign — LES, composite & ensemble analyses

End-to-end recipe for the ppiGPLM V3 evaluation campaign: train the ten MCCV
replicates, run LES against every reference-set condition, and build the
cross-replicate composite / ensemble figures. See `inference-recipe.md` for the
per-run LES details this ties together.

ppiGPLM is a character-level GPT-2 (joint prompt `<ps1>,SEQ1,<ps2>,SEQ2,<`); the
LES x-axis is the training **iteration** (`ckpt_<iter>.pt`, 500…6000). Env: the
`gpt` conda env, with `export MKL_THREADING_LAYER=GNU`.

## 0. Reference sets

All PRS/RRS conditions are in `V3_PRS-RRS/` (5-column ppiGPLM format):
`PRS-RRS/` (regular), `PRS-RRS_no_homodimers/`, `PRS-RRS_homodimers_only/`,
`random_controls/`. Training sets are not committed.

## 1. Per-model LES across conditions

For each replicate's checkpoint dir (`out_V3-k_6k/`), run the LES-wrapper on each
condition (see `inference-recipe.md`):

```bash
export MKL_THREADING_LAYER=GNU
PY=/home/ksa/anaconda3/envs/gpt/bin/python
$PY LES-wrapper_v2.py --checkpoint_dir out_V3-k_6k \
    --prs_file V3_PRS-RRS/<cond>/PRS-V3-k.csv \
    --rrs_file V3_PRS-RRS/<cond>/RRS-V3-k.csv \
    --output_dir LES_results_V3-k_6k --checkpoint_pattern 'ckpt_*.pt' --include_final
```

Conditions: `regular`, `no_homodimers`, `homodimers_only`, and the three random
controls. The random controls **auto-skip ROC-AUC / Best-F1 / LES** (filenames
contain `random`) and emit only the probability-distribution **violins** + raw
CSVs — those sets have no true positives.

## 2. Ensemble LES: trajectories + pooled violins

Across the ten per-model LES folders (mean ± SD over models):

```bash
$PY make_ensemble_les_trajectories.py --parent <parent> \
    --folder-template 'LES_results_V3-{k}_6k' --out <parent>/composite
$PY make_ensemble_les_violins.py --parent <parent> \
    --folder-template 'LES_results_V3-{k}_6k' --out <parent>/composite
```

`make_ensemble_les_trajectories.py` draws mean AUC/Best-F1 vs iteration (± SD) and
the ensemble LES; `make_ensemble_les_violins.py` pools PRS-vs-RRS P(interaction)
across models per checkpoint. Both take `--folder-template` so any layout works
(e.g. `LES_V3-{k}` for the homodimer-removed run).

## 3. Composite ROC across models

Vertically-averaged composite ROC per checkpoint + a cross-checkpoint overview:

```bash
LES_ROOT=<parent> FOLDER_TEMPLATE='LES_results_V3-{model}_6k' \
OUTDIR=<parent>/ROC_composite  bash run_all_checkpoints.sh
```

`run_all_checkpoints.sh` calls `make_composite_roc.py` (per-iteration composite ROC,
raw + smoothed) then `make_checkpoint_overview.py` (all-checkpoints ROC, checkpoint
grid, AUC-vs-iteration). Skip for random controls (no ROC).

## Summary of outputs

| step | script(s) | output |
|---|---|---|
| per-model LES | `LES-wrapper_v2.py` | `LES_results_V3-k_6k/` (per-iteration ROC/violin + LES) |
| ensemble LES | `make_ensemble_les_trajectories.py`, `make_ensemble_les_violins.py` | mean±SD trajectories, pooled violins |
| composite ROC | `run_all_checkpoints.sh` → `make_composite_roc.py`, `make_checkpoint_overview.py` | per-iteration composite ROC + overview |
