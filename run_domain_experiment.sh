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
#   ./run_domain_experiment.sh
#   ./run_domain_experiment.sh --dry-run

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

BASE="/local/scratch/freangi"
DOMAIN="${BASE}/domain"
OUTPUT="${BASE}/domain_tests/regnet_domain"

if [ ! -d "${DOMAIN}/domain_train" ] || [ ! -f "${DOMAIN}/domain_train/labels.json" ]; then
    echo "ERROR: Domain datasets not found at ${DOMAIN}. Run build_domain_dataset.sh first."
    exit 1
fi

if [ ! -d "${DOMAIN}/avianz_split/test" ] || [ ! -d "${DOMAIN}/doc_split/test" ]; then
    echo "ERROR: Per-domain test sets not found at ${DOMAIN}. Run build_domain_dataset.sh first."
    exit 1
fi

echo "============================================================"
echo " Train domain classifier: AviaNZ vs DOC"
echo "============================================================"
echo "  Train : ${DOMAIN}/domain_train"
echo "  Test  : ${DOMAIN}/domain_test"
echo "  Output: ${OUTPUT}"
echo "============================================================"
echo ""

mkdir -p "$(dirname "$OUTPUT")"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN — command that would run:"
    echo ""
fi

CMD=(
    python train.py
    "${DOMAIN}/domain_train"
    "$OUTPUT"
    --test-folder  "${DOMAIN}/avianz_split/test"
    --test-folder2 "${DOMAIN}/doc_split/test"
    --model-type  regnet
    --epochs      30
    --patience    15
    --mixup       0.25
    --visualize-attention
    --viz-samples 10
)

echo "  \$ ${CMD[*]}"
if [ "$DRY_RUN" -eq 0 ]; then
    PYTHONPATH="$PWD" "${CMD[@]}"
fi

echo ""
echo "============================================================"
echo " Done. Results in: $OUTPUT"
echo " Grad-CAM heatmaps in: ${OUTPUT}/attention_avianz_split/"
echo "                  and: ${OUTPUT}/attention_doc_split/"
echo "============================================================"
