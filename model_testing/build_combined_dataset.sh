#!/usr/bin/env bash
# build_combined_dataset.sh
#
# Build a single training dataset that merges DOC data (all species from
# NZBirds) with AviaNZ data from every annotated folder EXCEPT Joe_MoDone?
# (which is reserved for matched-test evaluation).
#
# Up to MAX_PER_SPECIES samples per class are kept across the combined pool,
# so abundant species are balanced while rare species contribute everything
# they have.
#
# Output: ${OUTPUT}/combined_large/   (labels.json + data/)
# The Trainer does its own 80/20 train/val split internally.
#
# Usage:
#   bash build_combined_dataset.sh
#   bash build_combined_dataset.sh --overwrite
#   MAX_PER_SPECIES=3000 bash build_combined_dataset.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE="/local/scratch/freangi"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"
DRIVE1="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_01"

if [[ ! -d "$DOC_RAW" ]]; then
    echo "ERROR: DOC raw data directory not found: $DOC_RAW"
    echo "Mount or update the path in this script before running the build."
    exit 1
fi
DRIVE2="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02"
DRIVE3="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_03"

# Scratch space for the intermediate per-source datasets.
# Combined output lands at ${OUTPUT}/combined_large.
OUTPUT="${BASE}/combined_dataset"
MAX_PER_SPECIES="${MAX_PER_SPECIES:-10000}"
MAPPING="$REPO_ROOT/model_testing/data/DOC_bird_naming_map.csv"
OVERWRITE_FLAG=""

if [[ "${1:-}" == "--overwrite" ]]; then
    OVERWRITE_FLAG="--overwrite"
fi

# All annotated AviaNZ folders, excluding Joe_MoDone? (matched-test source)
# and non-bird sources (bats, NZBirds).
# ECS_acoustic_01 and ECS_acoustic_03 are scanned as whole-drive roots so that
# any annotated subfolders are picked up automatically.
AVIANZ_FOLDERS=(
    "${DRIVE1}"
    "${DRIVE2}"
    "${DRIVE3}"
)

# Build --avianz-raw flags for each folder.
AVIANZ_ARGS=()
for FOLDER in "${AVIANZ_FOLDERS[@]}"; do
    AVIANZ_ARGS+=( "--avianz-raw" "${FOLDER}" )
done

echo "============================================================"
echo " Build combined DOC + AviaNZ dataset (${MAX_PER_SPECIES}/class max)"
echo "  DOC raw    : ${DOC_RAW}"
echo "  AviaNZ src : ${#AVIANZ_FOLDERS[@]} folders (excl. Joe_MoDone?)"
echo "  Output     : ${OUTPUT}/combined_large"
echo "  Mapping    : ${MAPPING}"
echo "============================================================"

mkdir -p "${OUTPUT}"

PYTHONPATH="$REPO_ROOT" python3 "$REPO_ROOT/model_testing/src/experiments/build_large_datasets.py" \
    --doc-raw        "${DOC_RAW}" \
    "${AVIANZ_ARGS[@]}" \
    --output         "${OUTPUT}" \
    --combined-out   "${OUTPUT}" \
    --mapping        "${MAPPING}" \
    --max-per-species "${MAX_PER_SPECIES}" \
    --label-remap "new zealand kaka:kaka,tui:tui/bellbird,bellbird:tui/bellbird" \
    --no-audio \
    --spec-type   Standard \
    --window-type Hamming \
    --sg-scale    "Mel Frequency" \
    ${OVERWRITE_FLAG}

echo ""
echo "Done. Training data at: ${OUTPUT}/combined_large"
