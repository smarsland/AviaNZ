#!/usr/bin/env bash
# run_large_experiment.sh
#
# Train RegNet on the large DOC and large AviaNZ datasets (built by
# build_large_dataset.sh) and evaluate on the large dataset's own test splits.
#
# Uses the same spectrogram config as the best-performing model:
#   Reassigned spectrogram · Hamming window · Linear scale
#   Log normalization + background subtraction + median filter
#
# Results land in $BASE/large_tests/ and are picked up automatically by
# scripts/analyze_all_results.py.
#
# Usage:
#   bash run_large_experiment.sh
#   bash run_large_experiment.sh --dry-run
# ---------------------------------------------------------------------------

set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE="/local/scratch/freangi"

LARGE_BASE="$BASE/large"
LARGE_DOC_TRAIN="$LARGE_BASE/doc_split/train"
LARGE_AVIANZ_TRAIN="$LARGE_BASE/avianz_split/train"
LARGE_AVIANZ_TEST="$LARGE_BASE/avianz_split/test"
LARGE_DOC_TEST="$LARGE_BASE/doc_split/test"

MATCHED_BASE="$BASE/matched"
MATCHED_AVIANZ_TEST="$MATCHED_BASE/avianz_split/test"
MATCHED_DOC_TEST="$MATCHED_BASE/doc_split/test"

TESTS_BASE="$BASE/large_tests"
MATCHED_TESTS_BASE="$BASE/matched_tests"

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS=100
PATIENCE=15
MIXUP=0.25
MODEL_TYPE="regnet"
VIZ_SAMPLES=3

# ── Checks ────────────────────────────────────────────────────────────────────
check_path() {
    if [ ! -e "$1" ]; then
        echo "ERROR: required path does not exist: $1"
        exit 1
    fi
}

if [ "$DRY_RUN" -eq 0 ]; then
    check_path "$LARGE_DOC_TRAIN"
    check_path "$LARGE_AVIANZ_TRAIN"
    check_path "$LARGE_AVIANZ_TEST"
    check_path "$LARGE_DOC_TEST"
    check_path "$MATCHED_AVIANZ_TEST"
    check_path "$MATCHED_DOC_TEST"
fi

run_cmd() {
    echo "  \$ $*"
    if [ "$DRY_RUN" -eq 0 ]; then
        PYTHONPATH="$(pwd)" "$@"
    fi
}

echo ""
echo "############################################################"
echo "  Large dataset experiments"
echo "  DOC train       : $LARGE_DOC_TRAIN"
echo "  AviaNZ train    : $LARGE_AVIANZ_TRAIN"
echo "  Large test 1    : $LARGE_AVIANZ_TEST"
echo "  Large test 2    : $LARGE_DOC_TEST"
echo "  Matched test 1  : $MATCHED_AVIANZ_TEST"
echo "  Matched test 2  : $MATCHED_DOC_TEST"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "  DRY RUN — no commands will be executed"
fi
echo "############################################################"
echo ""

if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$TESTS_BASE"
fi

# ── RegNet on large DOC — best config from matched sweep ─────────────────────
# (Box-Cox + background subtraction + median filter, Reassigned/Hamming/Mel defaults)
RUN_NAME="${MODEL_TYPE}_on_large_doc_bgmed_sed_head_like_kaytoo_spec"
OUT_DIR="$TESTS_BASE/$RUN_NAME"

if [ -f "$OUT_DIR/${MODEL_TYPE}_model.pt" ] && [ "$DRY_RUN" -eq 0 ]; then
    echo "  [skip] $RUN_NAME (already trained)"
else
    echo ""
    echo "============================================================"
    echo "  $RUN_NAME"
    echo "============================================================"
    run_cmd python train.py \
        "$LARGE_DOC_TRAIN" "$OUT_DIR" \
        --test-folder  "$LARGE_AVIANZ_TEST" \
        --test-folder2 "$LARGE_DOC_TEST" \
        --visualize-attention \
        --viz-samples  "$VIZ_SAMPLES" \
        --epochs       "$EPOCHS" \
        --patience     "$PATIENCE" \
        --mixup        "$MIXUP" \
        --model-type   "$MODEL_TYPE" \
        --spec-transform "LogMinMax" --bg-subtract --median-filter
fi

# Reload the just-trained (or previously-trained) model and evaluate on matched splits
MATCHED_OUT_DIR="$MATCHED_TESTS_BASE/$RUN_NAME"
[ "$DRY_RUN" -eq 0 ] && mkdir -p "$MATCHED_OUT_DIR"
echo ""
echo "  -- Reloading $RUN_NAME → evaluate on matched test splits --"
run_cmd python train.py \
    "$LARGE_DOC_TRAIN" "$MATCHED_OUT_DIR" \
    --eval-only \
    --checkpoint   "$OUT_DIR/${MODEL_TYPE}_model.pt" \
    --test-folder  "$MATCHED_AVIANZ_TEST" \
    --test-folder2 "$MATCHED_DOC_TEST" \
    --model-type   "$MODEL_TYPE" \
    --spec-transform "LogMinMax" --bg-subtract --median-filter

echo ""
echo "############################################################"
echo "  Large training run complete."
echo "  Large results   : $TESTS_BASE"
echo "  Matched results : $MATCHED_TESTS_BASE"
echo "  Run  python3 scripts/analyze_all_results.py  to compare."
echo "############################################################"
echo ""
