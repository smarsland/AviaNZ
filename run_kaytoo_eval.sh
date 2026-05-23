#!/bin/bash
set -e

# Run Kaytoo inference on both test sets and produce accuracy results.
# Run this after build_dataset.sh and run_experiments.sh.
# Must be run from the AviaNZ project root.
#
# Usage:
#   ./run_kaytoo_eval.sh              # evaluate on matched test sets
#   ./run_kaytoo_eval.sh --large      # evaluate on large (full) test sets
#   ./run_kaytoo_eval.sh --cpu        # force CPU inference

CPU_FLAG=""
LARGE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)   CPU_FLAG="--cpu"; shift ;;
        --large) LARGE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

OUTPUT_BASE="/local/scratch/freangi"
KAYTOO_ROOT="$(pwd)/../Kaytoo"
KAYTOO_PYTHON="${KAYTOO_ROOT}/venv_kay/bin/python"

MAPPING="data/DOC_bird_naming_map.csv"

if [ "$LARGE" = true ]; then
    AVIANZ_TEST="${OUTPUT_BASE}/large/avianz_split/test"
    DOC_TEST="${OUTPUT_BASE}/large/doc_split/test"
    OUTPUT="${OUTPUT_BASE}/large_tests/kaytoo_pretrained_seed0"
else
    AVIANZ_TEST="${OUTPUT_BASE}/matched/avianz_split/test"
    DOC_TEST="${OUTPUT_BASE}/matched/doc_split/test"
    OUTPUT="${OUTPUT_BASE}/matched_tests/kaytoo_pretrained_seed0"
fi

echo "============================================================"
echo " Kaytoo evaluation"
echo "============================================================"
echo "  Kaytoo root : $KAYTOO_ROOT"
echo "  Python      : $KAYTOO_PYTHON"
echo "  AviaNZ test : $AVIANZ_TEST"
echo "  DOC test    : $DOC_TEST"
echo "  Output      : $OUTPUT"
echo "============================================================"
echo ""

if [ ! -f "$KAYTOO_PYTHON" ]; then
    echo "ERROR: Kaytoo Python not found at $KAYTOO_PYTHON"
    echo "Check that venv_kay exists inside $KAYTOO_ROOT"
    exit 1
fi

if [ ! -d "$AVIANZ_TEST/audio" ] || [ ! -d "$DOC_TEST/audio" ]; then
    echo "ERROR: audio/ subfolder missing from test sets."
    echo "Re-run build_dataset.sh (it saves audio automatically)."
    exit 1
fi

mkdir -p "$OUTPUT"

PYTHONPATH="$PWD" "$KAYTOO_PYTHON" scripts/evaluate_kaytoo.py \
    "$AVIANZ_TEST" \
    "$DOC_TEST" \
    --kaytoo-root "$KAYTOO_ROOT" \
    --mapping     "$MAPPING" \
    --output      "$OUTPUT" \
    $CPU_FLAG

echo ""
echo "============================================================"
echo " Done. Results in $OUTPUT"
echo "============================================================"
