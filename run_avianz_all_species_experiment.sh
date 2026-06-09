#!/usr/bin/env bash
# run_avianz_all_species_experiment.sh
#
# Train RegNet and AST on the full-vocabulary AviaNZ dataset (~N species,
# 2000 segments/class max) built by build_avianz_all_species_dataset.sh.
#
# This is the AviaNZ-domain counterpart to run_all_species_experiment.sh.
# The hypothesis: pre-training on a large in-domain AviaNZ dataset produces
# better features for AviaNZ-test transfer than DOC-only pre-training.
#
# Both models use background subtraction and kbird-prior=2.0, matching the
# best-performing configuration in the scaling experiments.
#
# Outputs (under ${AVIANZ_ALL_SPECIES_TESTS}):
#   regnet_avianz_all_species_bgsubtract_seed{SEED}/
#   ast_avianz_all_species_bgsubtract_seed{SEED}/
#
# Prerequisites:
#   bash build_avianz_all_species_dataset.sh
#
# Usage:
#   bash run_avianz_all_species_experiment.sh
#   SEED=1 bash run_avianz_all_species_experiment.sh
#   MODELS=regnet bash run_avianz_all_species_experiment.sh

set -euo pipefail

BASE="/local/scratch/freangi"
AVIANZ_ALL_SPECIES_TRAIN="${BASE}/avianz_all_species/avianz_large"
MATCHED="${BASE}/matched"
MATCHED_AVIANZ_TEST="${MATCHED}/avianz_split/test"
MATCHED_DOC_TEST="${MATCHED}/doc_split/test"
AVIANZ_ALL_SPECIES_TESTS="${BASE}/avianz_all_species_tests"
PRETRAINED="BirdClefModels/model_fold0.pth"
SEED="${SEED:-0}"
MODELS="${MODELS:-both}"

# --- sanity checks ---
check_path() { [[ -d "$1" ]] || { echo "ERROR: directory not found: $1"; exit 1; }; }
check_path "${AVIANZ_ALL_SPECIES_TRAIN}"
check_path "${MATCHED_AVIANZ_TEST}"
check_path "${MATCHED_DOC_TEST}"

mkdir -p "${AVIANZ_ALL_SPECIES_TESTS}"

echo "================================================================"
echo " AviaNZ all-species training: full AviaNZ vocab → matched test"
echo "  Train data    : ${AVIANZ_ALL_SPECIES_TRAIN}"
echo "  AviaNZ test   : ${MATCHED_AVIANZ_TEST}"
echo "  DOC test      : ${MATCHED_DOC_TEST}"
echo "  Output root   : ${AVIANZ_ALL_SPECIES_TESTS}"
echo "  Seed          : ${SEED}"
echo "  Architectures : ${MODELS}"
echo "================================================================"

# ── RegNet ──────────────────────────────────────────────────────────
if [[ "${MODELS}" != "ast" ]]; then
    [[ -f "${PRETRAINED}" ]] || { echo "ERROR: pretrained weights not found: ${PRETRAINED}"; exit 1; }
    OUT="${AVIANZ_ALL_SPECIES_TESTS}/regnet_avianz_all_species_bgsubtract_seed${SEED}"
    echo ""
    echo "------------------------------------------------------------"
    echo " RegNet  →  ${OUT}"
    echo "------------------------------------------------------------"

    PYTHONPATH=. python3 train.py \
        "${AVIANZ_ALL_SPECIES_TRAIN}" \
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
    OUT="${AVIANZ_ALL_SPECIES_TESTS}/ast_avianz_all_species_bgsubtract_seed${SEED}"
    echo ""
    echo "------------------------------------------------------------"
    echo " AST  →  ${OUT}"
    echo "------------------------------------------------------------"

    PYTHONPATH=. python3 train.py \
        "${AVIANZ_ALL_SPECIES_TRAIN}" \
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
echo " AviaNZ all-species training complete."
echo " Checkpoints in: ${AVIANZ_ALL_SPECIES_TESTS}/"
echo " Next step: bash run_avianz_all_species_finetune_experiment.sh"
echo "================================================================"
