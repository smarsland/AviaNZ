#!/usr/bin/env bash
# run_combined_experiment.sh
#
# Pre-train RegNet and AST on the combined DOC + AviaNZ dataset
# (all available species, non-Joe_mo AviaNZ data merged with DOC,
# up to 5000 samples/class) built by build_combined_dataset.sh.
#
# Both models use background subtraction and kbird-prior=2.0, matching
# the best-performing configuration in the scaling experiments.
#
# Outputs (under ${COMBINED_TESTS}):
#   regnet_combined_bgsubtract_seed{SEED}/   – RegNet checkpoint
#   ast_combined_bgsubtract_seed{SEED}/      – AST checkpoint
#
# Prerequisites:
#   bash build_combined_dataset.sh
#
# Usage:
#   bash run_combined_experiment.sh
#   SEED=1 bash run_combined_experiment.sh
#   MODELS=regnet bash run_combined_experiment.sh   # skip AST

set -euo pipefail

BASE="/local/scratch/freangi"
COMBINED_TRAIN="${BASE}/combined_dataset/combined_large"
MATCHED="${BASE}/matched"
MATCHED_AVIANZ_TEST="${MATCHED}/avianz_split/test"
MATCHED_DOC_TEST="${MATCHED}/doc_split/test"
COMBINED_TESTS="${BASE}/combined_tests"
PRETRAINED="BirdClefModels/model_fold0.pth"
SEED="${SEED:-0}"
MODELS="${MODELS:-both}"
BG_SUBTRACT="${BG_SUBTRACT:-1}"   # set BG_SUBTRACT=0 to skip background subtraction

# Build the config tag and optional flag
if [[ "${BG_SUBTRACT}" == "1" ]]; then
    CONFIG_TAG="bgsubtract"
    BG_FLAG="--bg-subtract"
else
    CONFIG_TAG="nobgsub"
    BG_FLAG=""
fi

# --- sanity checks ---
check_path() { [[ -d "$1" ]] || { echo "ERROR: directory not found: $1"; exit 1; }; }
check_path "${COMBINED_TRAIN}"
check_path "${MATCHED_AVIANZ_TEST}"
check_path "${MATCHED_DOC_TEST}"

mkdir -p "${COMBINED_TESTS}"

echo "================================================================"
echo " Combined pre-training: DOC + AviaNZ full vocab → matched test"
echo "  Train data    : ${COMBINED_TRAIN}"
echo "  AviaNZ test   : ${MATCHED_AVIANZ_TEST}"
echo "  DOC test      : ${MATCHED_DOC_TEST}"
echo "  Output root   : ${COMBINED_TESTS}"
echo "  Seed          : ${SEED}"
echo "  Architectures : ${MODELS}"
echo "================================================================"

# ── RegNet ──────────────────────────────────────────────────────────
if [[ "${MODELS}" != "ast" ]]; then
    [[ -f "${PRETRAINED}" ]] || { echo "ERROR: pretrained weights not found: ${PRETRAINED}"; exit 1; }
    OUT="${COMBINED_TESTS}/regnet_combined_${CONFIG_TAG}_seed${SEED}"
    echo ""
    echo "------------------------------------------------------------"
    echo " RegNet  →  ${OUT}"
    echo "------------------------------------------------------------"

    PYTHONPATH=. python3 train.py \
        "${COMBINED_TRAIN}" \
        "${OUT}" \
        --model-type  regnet \
        --pretrained  "${PRETRAINED}" \
        --spec-transform Log \
        ${BG_FLAG} \
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
    OUT="${COMBINED_TESTS}/ast_combined_${CONFIG_TAG}_seed${SEED}"
    echo ""
    echo "------------------------------------------------------------"
    echo " AST  →  ${OUT}"
    echo "------------------------------------------------------------"

    PYTHONPATH=. python3 train.py \
        "${COMBINED_TRAIN}" \
        "${OUT}" \
        --model-type   ast \
        --spec-transform Log \
        ${BG_FLAG} \
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
echo " Combined pre-training complete."
echo " Checkpoints in: ${COMBINED_TESTS}/"
echo " Next step: bash run_combined_finetune_experiment.sh"
echo "================================================================"
