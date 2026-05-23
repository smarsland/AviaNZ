#!/bin/bash
set -e

# Run BirdNET inference on both test sets and produce accuracy results.
# Run this after build_dataset.sh (requires audio/ subfolders).
# Must be run from the AviaNZ project root.
#
# Usage:
#   ./run_birdnet_eval.sh              # evaluate on matched test sets
#   ./run_birdnet_eval.sh --large      # evaluate on large (full) test sets
#   ./run_birdnet_eval.sh --min-confidence 0.25

MIN_CONF_FLAG=""
LARGE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --large) LARGE=true; shift ;;
        --min-confidence) MIN_CONF_FLAG="--min-confidence $2"; shift 2 ;;
        *) echo "Unknown option: $1"; echo "Valid options: --large, --min-confidence FLOAT"; exit 1 ;;
    esac
done

OUTPUT_BASE="/local/scratch/freangi"

if [ "$LARGE" = true ]; then
    AVIANZ_TEST="${OUTPUT_BASE}/large/avianz_split/test"
    DOC_TEST="${OUTPUT_BASE}/large/doc_split/test"
    OUTPUT="${OUTPUT_BASE}/large_tests/birdnet_pretrained_seed0"
else
    AVIANZ_TEST="${OUTPUT_BASE}/matched/avianz_split/test"
    DOC_TEST="${OUTPUT_BASE}/matched/doc_split/test"
    OUTPUT="${OUTPUT_BASE}/matched_tests/birdnet_pretrained_seed0"
fi

echo "============================================================"
echo " BirdNET evaluation"
echo "============================================================"
echo "  AviaNZ test : $AVIANZ_TEST"
echo "  DOC test    : $DOC_TEST"
echo "  Output      : $OUTPUT"
echo "============================================================"
echo ""

if [ ! -d "$AVIANZ_TEST/audio" ] || [ ! -d "$DOC_TEST/audio" ]; then
    echo "ERROR: audio/ subfolder missing from test sets."
    echo "Re-run build_dataset.sh (it saves audio automatically)."
    exit 1
fi

mkdir -p "$OUTPUT"

PYTHONPATH="$PWD" python3 scripts/evaluate_birdnet.py \
    "$AVIANZ_TEST" \
    "$DOC_TEST" \
    --output "$OUTPUT" \
    ${MIN_CONF_FLAG}

echo ""
echo "============================================================"
echo " Done. Results in $OUTPUT"
echo "============================================================"
