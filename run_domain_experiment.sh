#!/bin/bash
set -euo pipefail

# Train a domain classifier (AviaNZ vs DOC) and generate Grad-CAM heatmaps.
# Run after build_domain_dataset.sh.
#
# The model is trained to predict which dataset a spectrogram comes from.
# Grad-CAM heatmaps reveal what spectral / temporal features differ
# systematically between the two datasets.
#
# Usage:
#   ./run_domain_experiment.sh                  # run both regnet and ast
#   ./run_domain_experiment.sh --model regnet   # regnet only
#   ./run_domain_experiment.sh --model ast      # ast only
#   ./run_domain_experiment.sh --dry-run

DRY_RUN=0
MODELS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=1; shift ;;
        --model) MODELS+=("$2"); shift 2 ;;
        *) echo "Unknown argument: $1"; echo "Valid options: --dry-run, --model {regnet,ast}"; exit 1 ;;
    esac
done

if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(regnet ast)
fi

BASE="/local/scratch/freangi"
DOMAIN="${BASE}/domain"
OUTPUT_BASE="${BASE}/domain_tests"

if [ ! -d "${DOMAIN}/domain_train" ] || [ ! -f "${DOMAIN}/domain_train/labels.json" ]; then
    echo "ERROR: Domain datasets not found at ${DOMAIN}. Run build_domain_dataset.sh first."
    exit 1
fi

if [ ! -d "${DOMAIN}/avianz_split/test" ] || [ ! -d "${DOMAIN}/doc_split/test" ]; then
    echo "ERROR: Per-domain test sets not found at ${DOMAIN}. Run build_domain_dataset.sh first."
    exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN — no commands will be executed"
fi

run_model() {
    local model=$1
    local output="${OUTPUT_BASE}/${model}_domain"

    echo ""
    echo "============================================================"
    echo " Train domain classifier: AviaNZ vs DOC  [${model}]"
    echo "============================================================"
    echo "  Train : ${DOMAIN}/domain_train"
    echo "  Test  : ${DOMAIN}/avianz_split/test  +  ${DOMAIN}/doc_split/test"
    echo "  Output: ${output}"
    echo "============================================================"
    echo ""

    mkdir -p "$(dirname "$output")"

    CMD=(
        python train.py
        "${DOMAIN}/domain_train"
        "$output"
        --test-folder  "${DOMAIN}/avianz_split/test"
        --test-folder2 "${DOMAIN}/doc_split/test"
        --model-type   "$model"
        --epochs       30
        --patience     15
        --mixup        0.25
        --visualize-attention
        --viz-samples  10
    )

    echo "  \$ ${CMD[*]}"
    if [ "$DRY_RUN" -eq 0 ]; then
        PYTHONPATH="$PWD" "${CMD[@]}"
    fi

    echo ""
    echo "============================================================"
    echo " Done [${model}]. Results in: $output"
    echo " Grad-CAM heatmaps in: ${output}/attention_avianz_split/"
    echo "                  and: ${output}/attention_doc_split/"
    echo "============================================================"
}

for model in "${MODELS[@]}"; do
    run_model "$model"
done
