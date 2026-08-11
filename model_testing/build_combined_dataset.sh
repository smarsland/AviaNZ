#!/usr/bin/env bash
# build_combined_dataset.sh
#
# Build DOC + AviaNZ training datasets and merge their training splits.
#
# Output:
#   ${OUTPUT}/doc_large/
#   ${OUTPUT}/avianz_large/
#   ${OUTPUT}/doc_split/
#   ${OUTPUT}/avianz_split/
#   ${OUTPUT}/merged_train/
#
# AviaNZ recordings in matched/avianz_matched are excluded because they are
# reserved for matched-test evaluation.
#
# Usage:
#   bash build_combined_dataset.sh
#   bash build_combined_dataset.sh --overwrite
#   MAX_PER_SPECIES=3000 bash build_combined_dataset.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SERVER_PREFIX="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic"

BASE="${AVIA_NZ_BASE:-/local/scratch/freangi}"
DOC_RAW="${DOC_RAW_DIR:-${SERVER_PREFIX}_02/NZBirds}"

DRIVE1="${AVIANZ_DRIVE1:-${SERVER_PREFIX}_01}"
DRIVE2="${AVIANZ_DRIVE2:-${SERVER_PREFIX}_02}"
DRIVE3="${AVIANZ_DRIVE3:-${SERVER_PREFIX}_03}"

MATCHED_AVIANZ="${BASE}/matched/avianz_matched/labels.json"

OVERWRITE_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --overwrite)
            OVERWRITE_FLAG="--overwrite"
            shift
            ;;
        --doc-raw)
            DOC_RAW="$2"
            shift 2
            ;;
        --output-base)
            BASE="$2"
            shift 2
            ;;
        --avianz-drive1)
            DRIVE1="$2"
            shift 2
            ;;
        --avianz-drive2)
            DRIVE2="$2"
            shift 2
            ;;
        --avianz-drive3)
            DRIVE3="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ ! -d "$DOC_RAW" ]]; then
    echo "ERROR: DOC raw data directory not found: $DOC_RAW"
    exit 1
fi

if [[ ! -f "$MATCHED_AVIANZ" ]]; then
    echo "ERROR: matched AviaNZ labels not found: $MATCHED_AVIANZ"
    exit 1
fi


OUTPUT="${BASE}/combined_dataset"
MAX_PER_SPECIES="${MAX_PER_SPECIES:-5000}"
MAPPING="$REPO_ROOT/model_testing/data/DOC_bird_naming_map.csv"


AVIANZ_FOLDERS=(
    "${DRIVE1}"
    "${DRIVE2}"
    "${DRIVE3}"
)

AVIANZ_ARGS=()
for FOLDER in "${AVIANZ_FOLDERS[@]}"; do
    AVIANZ_ARGS+=( "--avianz-raw" "${FOLDER}" )
done


echo "============================================================"
echo " Build DOC + AviaNZ merged training dataset"
echo "  DOC raw       : ${DOC_RAW}"
echo "  AviaNZ src    : ${#AVIANZ_FOLDERS[@]} folders"
echo "  Excluding     : ${MATCHED_AVIANZ}"
echo "  Output        : ${OUTPUT}"
echo "  Max/species   : ${MAX_PER_SPECIES}"
echo "  Mapping       : ${MAPPING}"
echo "============================================================"


mkdir -p "${OUTPUT}"


PYTHONPATH="$REPO_ROOT" python3 \
"$REPO_ROOT/model_testing/src/experiments/build_large_datasets.py" \
    --doc-raw "${DOC_RAW}" \
    "${AVIANZ_ARGS[@]}" \
    --output "${OUTPUT}" \
    --mapping "${MAPPING}" \
    --max-per-species "${MAX_PER_SPECIES}" \
    --exclude-source-files "${MATCHED_AVIANZ}" \
    --label-remap "new zealand kaka:kaka,tui:tui/bellbird,bellbird:tui/bellbird" \
    --spec-type Standard \
    --window-type Hamming \
    --sg-scale "Mel Frequency" \
    ${OVERWRITE_FLAG}


echo ""
echo "Done."
echo "Merged training dataset:"
echo "  ${OUTPUT}/merged_train"