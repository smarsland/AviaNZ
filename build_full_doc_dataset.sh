#!/bin/bash
# Build a DOC-only dataset using ALL available species and up to
# MAX_PER_SPECIES samples per class.
#
# Unlike build_scaling_dataset.sh this does NOT restrict to the 9 matched
# classes — every mappable DOC species is included.  A label remap is applied
# so that kaka and tui/bellbird use the same names as the matched test splits,
# enabling direct evaluation against those splits.
#
# Output: ${OUTPUT}/doc_large/   (labels.json + data/)
# The Trainer does its own 80/20 train/val split internally.
#
# Usage:
#   bash build_full_doc_dataset.sh
#   bash build_full_doc_dataset.sh --overwrite
#   bash build_full_doc_dataset.sh --max-per-species 3000

set -euo pipefail

BASE="/local/scratch/freangi"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"
OUTPUT="${BASE}/full_doc"
MAX_PER_SPECIES="${MAX_PER_SPECIES:-5000}"
MAPPING="data/DOC_bird_naming_map.csv"
OVERWRITE_FLAG=""

# Remap DOC names to match the matched-dataset labels for kaka and tui/bellbird
LABEL_REMAP="new zealand kaka:kaka,tui:tui/bellbird,bellbird:tui/bellbird"

while [[ $# -gt 0 ]]; do
    case $1 in
        --overwrite)        OVERWRITE_FLAG="--overwrite"; shift ;;
        --max-per-species)  MAX_PER_SPECIES="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Valid options: --overwrite, --max-per-species N"
            exit 1 ;;
    esac
done

echo "============================================================"
echo " Build full DOC dataset (all species, ${MAX_PER_SPECIES}/class max)"
echo "  DOC raw : ${DOC_RAW}"
echo "  Output  : ${OUTPUT}/doc_large"
echo "  Mapping : ${MAPPING}"
echo "============================================================"

mkdir -p "${OUTPUT}"

PYTHONPATH=. python3 src/experiments/build_large_datasets.py \
    --doc-raw    "${DOC_RAW}" \
    --output     "${OUTPUT}" \
    --mapping    "${MAPPING}" \
    --doc-only \
    --label-remap      "${LABEL_REMAP}" \
    --max-per-species  "${MAX_PER_SPECIES}" \
    --no-audio \
    --spec-type   Standard \
    --window-type Hamming \
    --sg-scale    "Mel Frequency" \
    ${OVERWRITE_FLAG:+$OVERWRITE_FLAG}

echo ""
echo "Done. Training data at: ${OUTPUT}/doc_large"
echo "Run: bash run_full_doc_experiment.sh"
