#!/usr/bin/env bash
# run_all_species_experiment.sh
#
# Pre-train RegNet and AST on the full-vocabulary DOC dataset (~130 species,
# 8000 samples/class) built by build_all_species_dataset.sh.
#
# The hypothesis: a model that sees many more species during pre-training will
# learn richer bird-audio features and fine-tune better on the 12 matched-test
# species than one pre-trained on only those 12 species.
#
# Both models use background subtraction and kbird-prior=2.0, matching the
# best-performing configuration in the scaling experiments.
#
# Outputs (under ${ALL_SPECIES_TESTS}):
#   regnet_all_species_bgsubtract_seed{SEED}/   – RegNet checkpoint
#   ast_all_species_bgsubtract_seed{SEED}/      – AST checkpoint
#
# Prerequisites:
#   bash build_all_species_dataset.sh
#
# Usage:
#   bash run_all_species_experiment.sh
#   SEED=1 bash run_all_species_experiment.sh
#   MODELS=regnet bash run_all_species_experiment.sh   # skip AST

set -euo pipefail

BASE="/local/scratch/freangi"
ALL_SPECIES_TRAIN="${BASE}/scaling_all_species/doc_large"
MATCHED="${BASE}/matched"
MATCHED_AVIANZ_TEST="${MATCHED}/avianz_split/test"
MATCHED_DOC_TEST="${MATCHED}/doc_split/test"
ALL_SPECIES_TESTS="${BASE}/all_species_tests"
PRETRAINED="BirdClefModels/model_fold0.pth"
SEED="${SEED:-0}"
# Set MODELS=regnet or MODELS=ast to run only one architecture.
MODELS="${MODELS:-both}"

# --- sanity checks ---
check_path() { [[ -d "$1" ]] || { echo "ERROR: directory not found: $1"; exit 1; }; }
check_path "${ALL_SPECIES_TRAIN}"
check_path "${MATCHED_AVIANZ_TEST}"
check_path "${MATCHED_DOC_TEST}"

mkdir -p "${ALL_SPECIES_TESTS}"

echo "================================================================"
echo " All-species pre-training: DOC full vocab → matched test"
echo "  Train data    : ${ALL_SPECIES_TRAIN}"
echo "  AviaNZ test   : ${MATCHED_AVIANZ_TEST}"
echo "  DOC test      : ${MATCHED_DOC_TEST}"
echo "  Output root   : ${ALL_SPECIES_TESTS}"
echo "  Seed          : ${SEED}"
echo "  Architectures : ${MODELS}"
echo "================================================================"

# ── RegNet ──────────────────────────────────────────────────────────
if [[ "${MODELS}" != "ast" ]]; then
    [[ -f "${PRETRAINED}" ]] || { echo "ERROR: pretrained weights not found: ${PRETRAINED}"; exit 1; }
    OUT="${ALL_SPECIES_TESTS}/regnet_all_species_bgsubtract_seed${SEED}"
    echo ""
    echo "------------------------------------------------------------"
    echo " RegNet  →  ${OUT}"
    echo "------------------------------------------------------------"

    PYTHONPATH=. python3 train.py \
        "${ALL_SPECIES_TRAIN}" \
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
    OUT="${ALL_SPECIES_TESTS}/ast_all_species_bgsubtract_seed${SEED}"
    echo ""
    echo "------------------------------------------------------------"
    echo " AST  →  ${OUT}"
    echo "------------------------------------------------------------"

    PYTHONPATH=. python3 train.py \
        "${ALL_SPECIES_TRAIN}" \
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
echo " All-species pre-training complete."
echo " Checkpoints in: ${ALL_SPECIES_TESTS}/"
echo " Next step: bash run_all_species_finetune_experiment.sh"
echo "================================================================"
