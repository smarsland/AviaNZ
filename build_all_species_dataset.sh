#!/usr/bin/env bash
# build_all_species_dataset.sh
#
# Build a DOC-only dataset using ALL available species (not restricted to the
# 12 matched-test species).  The model trained on this dataset will see a much
# richer vocabulary (~130 species) before being fine-tuned on the 12-class
# matched set, acting as a stronger feature pre-trainer than the 9/12-class
# scaling dataset.
#
# Saves up to MAX_PER_SPECIES (8000) samples per class so that
# run_all_species_experiment.sh can subsample at train time if needed.
#
# Output: ${OUTPUT}/doc_all_species/   (labels.json + data/)
# The Trainer does its own 80/20 train/val split internally.
#
# Compare with build_scaling_dataset.sh which restricts to 9 classes.
#
# Usage:
#   bash build_all_species_dataset.sh
#   DOC_RAW=/my/path bash build_all_species_dataset.sh

set -euo pipefail

BASE="/local/scratch/freangi"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"
# Separate root so output lands at ${OUTPUT}/doc_large (the builder always uses
# that subfolder name) without clobbering the 9-class scaling dataset.
OUTPUT="${BASE}/scaling_all_species"
MAX_PER_SPECIES=2000
MAPPING="data/DOC_bird_naming_map.csv"

# Label remaps to align DOC names with the matched-test vocabulary.
# (same remaps used in build_scaling_dataset.sh)
LABEL_REMAP="new zealand kaka:kaka,tui:tui/bellbird,bellbird:tui/bellbird"

echo "============================================================"
echo " Build all-species DOC dataset (${MAX_PER_SPECIES}/class max)"
echo "  DOC raw : ${DOC_RAW}"
echo "  Output  : ${OUTPUT}/doc_large"
echo "  Mapping : ${MAPPING}"
echo "  Remaps  : ${LABEL_REMAP}"
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
    --spec-type  Standard \
    --window-type Hamming \
    --sg-scale   "Mel Frequency" \
    --overwrite

echo ""
echo "Done. Training data at: ${OUTPUT}/doc_large"
echo "Run: bash run_all_species_experiment.sh"
