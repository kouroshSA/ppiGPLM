#!/usr/bin/env bash
# Regenerate every per-checkpoint composite ROC folder plus the cross-checkpoint
# overview figures.
#
#   # native 10-model ensemble (defaults, run from this dir):
#   bash run_all_checkpoints.sh
#
#   # homodimer-removed run (folders named LES_V3-{k}, output elsewhere):
#   LES_ROOT=/home/ksa/Dropbox/LES_and_V3_Datasets/LES_results_V3_no_homodimers \
#   FOLDER_TEMPLATE='LES_{model}' \
#   OUTDIR=/home/ksa/Dropbox/LES_and_V3_Datasets/LES_results_V3_no_homodimers/ROC_composite \
#       bash run_all_checkpoints.sh
#
# Requires numpy / matplotlib / scikit-learn. The `gpt` conda env on this
# machine has a working scikit-learn; the anaconda base env does NOT
# (numpy/sklearn ABI mismatch).
set -euo pipefail

PY="${PY:-/home/ksa/anaconda3/envs/gpt/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Where the per-model LES folders live, and how they are named. Defaults
# reproduce the original native-ensemble run.
LES_ROOT="${LES_ROOT:-/home/ksa/Dropbox/LES_and_V3_Datasets/LES_results}"
# NB: keep the {model}-containing default OUT of a ${VAR:-default} expansion —
# the embedded '}' would prematurely close the expansion in bash.
FOLDER_TEMPLATE="${FOLDER_TEMPLATE:-}"
[ -z "$FOLDER_TEMPLATE" ] && FOLDER_TEMPLATE='LES_results_{model}_6k'
MODELS="${MODELS:-10}"

# Where the <iter>_ROCs subfolders and overview figures are written. Defaults to
# this script's folder (the original in-place behaviour).
OUTDIR="${OUTDIR:-$HERE}"
mkdir -p "$OUTDIR"

# Gaussian smoothing width for the "_smoothed" variants, in FPR-grid points
# (grid = 501 points over 0-1, so 5 ~ 0.01 FPR). Display only.
SIGMA="${SIGMA:-5}"

for it in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000; do
    echo ">>> iteration ${it}"

    # raw curves
    mkdir -p "${OUTDIR}/${it}_ROCs"
    "$PY" "$HERE/make_composite_roc.py" --iter "$it" \
        --les-root "$LES_ROOT" --folder-template "$FOLDER_TEMPLATE" \
        --models "$MODELS" --outdir "${OUTDIR}/${it}_ROCs" \
        | tee "${OUTDIR}/${it}_ROCs/_run_log.txt"

    # same data, curves smoothed for display
    mkdir -p "${OUTDIR}/${it}_ROCs_smoothed"
    "$PY" "$HERE/make_composite_roc.py" --iter "$it" \
        --les-root "$LES_ROOT" --folder-template "$FOLDER_TEMPLATE" \
        --models "$MODELS" --outdir "${OUTDIR}/${it}_ROCs_smoothed" \
        --smooth "$SIGMA" | tee "${OUTDIR}/${it}_ROCs_smoothed/_run_log.txt"
done

# Cross-checkpoint overview figures (always smoothed; see README).
"$PY" "$HERE/make_checkpoint_overview.py" --dir "$OUTDIR"
