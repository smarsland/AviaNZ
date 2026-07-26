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

BASE="${AVIA_NZ_BASE:-$REPO_ROOT/model_testing/output}"
DOC_RAW="${DOC_RAW_DIR:-}"
DRIVE1="${AVIANZ_DRIVE1:-}"
DRIVE2="${AVIANZ_DRIVE2:-}"
DRIVE3="${AVIANZ_DRIVE3:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --overwrite) OVERWRITE_FLAG="--overwrite"; shift ;;
        --doc-raw) DOC_RAW="$2"; shift 2 ;;
        --output-base) BASE="$2"; shift 2 ;;
        --avianz-drive1) DRIVE1="$2"; shift 2 ;;
        --avianz-drive2) DRIVE2="$2"; shift 2 ;;
        --avianz-drive3) DRIVE3="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$DOC_RAW" ]]; then
    DOC_RAW="$(find /data /mnt /media /home /workspace /srv /tmp -maxdepth 6 -type d -name 'NZBirds' 2>/dev/null | head -n 1 || true)"
fi

if [[ ! -d "$DOC_RAW" ]]; then
    echo "ERROR: DOC raw data directory not found: $DOC_RAW"
    echo "Mount or update the path in this script before running the build."
    exit 1
fi

if [[ -z "$DRIVE1" ]]; then
    DRIVE1="$(find /data /mnt /media /home /workspace /srv /tmp -maxdepth 6 -type d -name 'ECS_acoustic_01' 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$DRIVE2" ]]; then
    DRIVE2="$(find /data /mnt /media /home /workspace /srv /tmp -maxdepth 6 -type d -name 'ECS_acoustic_02' 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$DRIVE3" ]]; then
    DRIVE3="$(find /data /mnt /media /home /workspace /srv /tmp -maxdepth 6 -type d -name 'ECS_acoustic_03' 2>/dev/null | head -n 1 || true)"
fi

# Scratch space for the intermediate per-source datasets.
# Combined output lands at ${OUTPUT}/combined_large.
OUTPUT="${BASE}/combined_dataset"
MAX_PER_SPECIES="${MAX_PER_SPECIES:-10000}"
MAPPING="$REPO_ROOT/model_testing/data/DOC_bird_naming_map.csv"
OVERWRITE_FLAG=""

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
