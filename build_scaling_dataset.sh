#!/bin/bash
# Build a DOC-only dataset for the data-scaling experiment.
#
# Uses the 9 classes from the matched experiments:
#   blackbird, chaffinch, fantail, grey warbler, kaka,
#   morepork, silvereye, tomtit, tui/bellbird
#
# Saves up to MAX_PER_SPECIES (3000) samples per class so that the
# run_scaling_experiment.sh script can subsample down to smaller N
# at train time using --max-samples-per-class.
#
# Output: ${OUTPUT}/doc_large/   (labels.json + data/)
# The Trainer does its own 80/20 train/val split internally.
#
# Usage:
#   bash build_scaling_dataset.sh            # uses default paths
#   DOC_RAW=/my/path bash build_scaling_dataset.sh

set -euo pipefail

BASE="/local/scratch/freangi"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"
OUTPUT="${BASE}/scaling"
MAX_PER_SPECIES=3000
MAPPING="data/DOC_bird_naming_map.csv"

# The 9 class labels as they appear in the matched dataset labels.json
# restrict-classes uses the POST-remap names; label-remap converts DOC names to match
CLASSES="blackbird,chaffinch,fantail,grey warbler,kaka,morepork,silvereye,tomtit,tui/bellbird"

# DOC name mapping: ebird_to_common gives "New Zealand Kaka", "Tui", "Bellbird"
# Remap to match the merged labels used in the matched dataset
LABEL_REMAP="new zealand kaka:kaka,tui:tui/bellbird,bellbird:tui/bellbird"

echo "============================================================"
echo " Build scaling dataset (DOC only, 9 classes, ${MAX_PER_SPECIES}/class)"
echo "  DOC raw : ${DOC_RAW}"
echo "  Output  : ${OUTPUT}/doc_large"
echo "  Classes : ${CLASSES}"
echo "============================================================"

mkdir -p "${OUTPUT}"

PYTHONPATH=. python3 src/experiments/build_large_datasets.py \
    --doc-raw    "${DOC_RAW}" \
    --output     "${OUTPUT}" \
    --mapping    "${MAPPING}" \
    --doc-only \
    --restrict-classes "${CLASSES}" \
    --label-remap      "${LABEL_REMAP}" \
    --max-per-species  "${MAX_PER_SPECIES}" \
    --no-audio \
    --spec-type  Standard \
    --window-type Hamming \
    --sg-scale   "Mel Frequency" \
    --overwrite

echo ""
echo "Done. Training data at: ${OUTPUT}/doc_large"
echo "Run: bash run_scaling_experiment.sh"
