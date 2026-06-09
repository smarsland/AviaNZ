#!/usr/bin/env bash
# run_avianz_all_species_finetune_experiment.sh
#
# Fine-tune the AviaNZ all-species pre-trained models (from
# run_avianz_all_species_experiment.sh) on the corrected 12-class matched
# DOC training set.
#
# The classifier head is automatically replaced (shape mismatch → skipped on
# load) so the model adapts from the full AviaNZ vocabulary down to 12 classes.
#
# RegNet: full backbone frozen  (--freeze-backbone), same as other fine-tunes.
# AST:    first 8 of 12 layers frozen (--freeze-layers 8), same as AST scaling.
#
# Outputs (under ${AVIANZ_ALL_SPECIES_TESTS}):
#   regnet_avianz_all_species_bgsubtract_ft_seed{SEED}/
#   ast_avianz_all_species_bgsubtract_ft_seed{SEED}/
#
# Prerequisites:
#   bash run_avianz_all_species_experiment.sh
#
# Usage:
#   bash run_avianz_all_species_finetune_experiment.sh
#   SEED=1 bash run_avianz_all_species_finetune_experiment.sh
#   MODELS=regnet bash run_avianz_all_species_finetune_experiment.sh

set -euo pipefail

BASE="/local/scratch/freangi"
MATCHED="${BASE}/matched"
MATCHED_DOC_TRAIN="${MATCHED}/doc_split/train"
MATCHED_AVIANZ_TEST="${MATCHED}/avianz_split/test"
MATCHED_DOC_TEST="${MATCHED}/doc_split/test"
AVIANZ_ALL_SPECIES_TESTS="${BASE}/avianz_all_species_tests"
SEED="${SEED:-0}"
MODELS="${MODELS:-both}"

# --- sanity checks ---
check_path() { [[ -d "$1" ]] || { echo "ERROR: directory not found: $1"; exit 1; }; }
check_path "${MATCHED_DOC_TRAIN}"
check_path "${MATCHED_AVIANZ_TEST}"
check_path "${MATCHED_DOC_TEST}"

mkdir -p "${AVIANZ_ALL_SPECIES_TESTS}"

echo "================================================================"
echo " AviaNZ all-species fine-tune: pre-trained → 12-class matched"
echo "  Fine-tune train : ${MATCHED_DOC_TRAIN}"
echo "  AviaNZ test     : ${MATCHED_AVIANZ_TEST}"
echo "  DOC test        : ${MATCHED_DOC_TEST}"
echo "  Output root     : ${AVIANZ_ALL_SPECIES_TESTS}"
echo "  Seed            : ${SEED}"
echo "  Architectures   : ${MODELS}"
echo "================================================================"

# ── RegNet ──────────────────────────────────────────────────────────
if [[ "${MODELS}" != "ast" ]]; then
    CKPT="${AVIANZ_ALL_SPECIES_TESTS}/regnet_avianz_all_species_bgsubtract_seed${SEED}/regnet_model_best.pt"
    OUT="${AVIANZ_ALL_SPECIES_TESTS}/regnet_avianz_all_species_bgsubtract_ft_seed${SEED}"

    echo ""
    echo "------------------------------------------------------------"
    echo " RegNet  checkpoint: ${CKPT}"
    echo "         →  ${OUT}"
    echo "------------------------------------------------------------"

    if [[ ! -f "${CKPT}" ]]; then
        echo " SKIP: checkpoint not found — run run_avianz_all_species_experiment.sh first"
    else
        PYTHONPATH=. python3 train.py \
            "${MATCHED_DOC_TRAIN}" \
            "${OUT}" \
            --model-type  regnet \
            --pretrained  "${CKPT}" \
            --spec-transform Log \
            --bg-subtract \
            --kbird-prior 2.0 \
            --freeze-backbone \
            --lr          1e-4 \
            --epochs      20 \
            --patience    10 \
            --seed        "${SEED}" \
            --test-folder  "${MATCHED_AVIANZ_TEST}" \
            --test-folder2 "${MATCHED_DOC_TEST}"

        echo " Finished RegNet fine-tune"
    fi
fi

# ── AST ─────────────────────────────────────────────────────────────
if [[ "${MODELS}" != "regnet" ]]; then
    CKPT="${AVIANZ_ALL_SPECIES_TESTS}/ast_avianz_all_species_bgsubtract_seed${SEED}/ast_model_best.pt"
    OUT="${AVIANZ_ALL_SPECIES_TESTS}/ast_avianz_all_species_bgsubtract_ft_seed${SEED}"

    echo ""
    echo "------------------------------------------------------------"
    echo " AST  checkpoint: ${CKPT}"
    echo "      →  ${OUT}"
    echo "------------------------------------------------------------"

    if [[ ! -f "${CKPT}" ]]; then
        echo " SKIP: checkpoint not found — run run_avianz_all_species_experiment.sh first"
    else
        PYTHONPATH=. python3 train.py \
            "${MATCHED_DOC_TRAIN}" \
            "${OUT}" \
            --model-type   ast \
            --pretrained   "${CKPT}" \
            --spec-transform Log \
            --bg-subtract \
            --kbird-prior  2.0 \
            --freeze-layers 8 \
            --lr           1e-4 \
            --epochs       20 \
            --patience     10 \
            --seed         "${SEED}" \
            --test-folder  "${MATCHED_AVIANZ_TEST}" \
            --test-folder2 "${MATCHED_DOC_TEST}"

        echo " Finished AST fine-tune"
    fi
fi

echo ""
echo "================================================================"
echo " AviaNZ all-species fine-tune complete."
echo " Checkpoints in: ${AVIANZ_ALL_SPECIES_TESTS}/"
echo "================================================================"
