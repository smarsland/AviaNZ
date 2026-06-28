#!/usr/bin/env bash
# run_full_doc_experiment.sh
#
# Train RegNet and AST on the full DOC dataset (all species, built by
# build_full_doc_dataset.sh) and evaluate on the matched human-reviewed
# test splits.
#
# This is the missing counterpart to the scaling experiments: instead of
# restricting to 9 matched classes, the model sees every mappable DOC species
# and we test how well it generalises to the matched split vocabulary.
#
# Uses the best-performing configuration from the scaling experiments:
#   Log transform + background subtraction + kbird-prior 2.0
#
# Results land in ${FULL_DOC_TESTS}/ and are picked up automatically by
# scripts/analyze_all_results.py.
#
# Prerequisites:
#   bash build_full_doc_dataset.sh
#
# Usage:
#   bash run_full_doc_experiment.sh
#   SEED=1 bash run_full_doc_experiment.sh
#   MODELS=regnet bash run_full_doc_experiment.sh   # skip AST

set -euo pipefail

BASE="/local/scratch/freangi"
FULL_DOC_TRAIN="${BASE}/full_doc/doc_large"
MATCHED="${BASE}/matched"
MATCHED_AVIANZ_TEST="${MATCHED}/avianz_split/test"
MATCHED_DOC_TEST="${MATCHED}/doc_split/test"
FULL_DOC_TESTS="${BASE}/full_doc_tests"
PRETRAINED="BirdClefModels/model_fold0.pth"
SEED="${SEED:-0}"
MODELS="${MODELS:-both}"

# --- sanity checks ---
check_path() { [[ -d "$1" ]] || { echo "ERROR: directory not found: $1"; exit 1; }; }
check_path "${FULL_DOC_TRAIN}"
check_path "${MATCHED_AVIANZ_TEST}"
check_path "${MATCHED_DOC_TEST}"

mkdir -p "${FULL_DOC_TESTS}"

echo "================================================================"
echo " Full DOC training: all DOC species → matched test"
echo "  Train data    : ${FULL_DOC_TRAIN}"
echo "  AviaNZ test   : ${MATCHED_AVIANZ_TEST}"
echo "  DOC test      : ${MATCHED_DOC_TEST}"
echo "  Output root   : ${FULL_DOC_TESTS}"
echo "  Seed          : ${SEED}"
echo "  Architectures : ${MODELS}"
echo "================================================================"

# ── RegNet ──────────────────────────────────────────────────────────
if [[ "${MODELS}" != "ast" ]]; then
    [[ -f "${PRETRAINED}" ]] || { echo "ERROR: pretrained weights not found: ${PRETRAINED}"; exit 1; }
    OUT="${FULL_DOC_TESTS}/regnet_full_doc_bgsubtract_seed${SEED}"
    echo ""
    echo "------------------------------------------------------------"
    echo " RegNet  →  ${OUT}"
    echo "------------------------------------------------------------"

    PYTHONPATH=. python3 train.py \
        "${FULL_DOC_TRAIN}" \
        "${OUT}" \
        --model-type  regnet \
        --pretrained  "${PRETRAINED}" \
        --spec-transform Log \
        --bg-subtract \
        --kbird-prior 2.0 \
        --epochs      40 \
        --patience    15 \
        --seed        "${SEED}" \
        --test-folder  "${MATCHED_AVIANZ_TEST}" \
        --test-folder2 "${MATCHED_DOC_TEST}"

    echo " Finished RegNet"
fi

# ── AST ─────────────────────────────────────────────────────────────
if [[ "${MODELS}" != "regnet" ]]; then
    OUT="${FULL_DOC_TESTS}/ast_full_doc_bgsubtract_seed${SEED}"
    echo ""
    echo "------------------------------------------------------------"
    echo " AST  →  ${OUT}"
    echo "------------------------------------------------------------"

    PYTHONPATH=. python3 train.py \
        "${FULL_DOC_TRAIN}" \
        "${OUT}" \
        --model-type   ast \
        --spec-transform Log \
        --bg-subtract \
        --kbird-prior  2.0 \
        --epochs       40 \
        --patience     15 \
        --seed         "${SEED}" \
        --test-folder  "${MATCHED_AVIANZ_TEST}" \
        --test-folder2 "${MATCHED_DOC_TEST}"

    echo " Finished AST"
fi

echo ""
echo "================================================================"
echo " Full DOC training complete."
echo " Checkpoints in: ${FULL_DOC_TESTS}/"
echo " Run  python3 scripts/analyze_all_results.py  to compare."
echo "================================================================"
