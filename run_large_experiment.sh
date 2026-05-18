#!/usr/bin/env bash
# run_large_experiment.sh
#
# Train RegNet on the large DOC dataset (built by build_large_dataset.sh) and
# evaluate on the large dataset's own test splits (avianz_split/test and
# doc_split/test from the same build).
#
# Training uses the same spectrogram config as the best-performing model:
#   Reassigned spectrogram · Hamming window · Linear scale
#   Runs all 3 normalization variants: Log / PCEN / Box-Cox
#
# Results land in $TESTS_BASE/regnet_on_large_doc_{norm}/ and are picked up
# automatically by scripts/analyze_all_results.py.
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

# Large dataset built by build_large_dataset.sh
LARGE_BASE="$BASE/large"
LARGE_DOC_TRAIN="$LARGE_BASE/doc_split/train"

# Matched test sets built by build_dataset.sh
MATCHED_BASE="$BASE/matched"
AVIANZ_TEST="$MATCHED_BASE/avianz_split/test"
DOC_TEST="$MATCHED_BASE/doc_split/test"

# Output directory for trained models (same layout as sweep_tests)
TESTS_BASE="$BASE/sweep_tests"

# ── Training hyperparameters (matching run_sweep.py) ──────────────────────────
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
    check_path "$AVIANZ_TEST"
    check_path "$DOC_TEST"
fi

run_cmd() {
    echo "  \$ $*"
    if [ "$DRY_RUN" -eq 0 ]; then
        PYTHONPATH="$(pwd)" "$@"
    fi
}

# ── Normalization variants (same 3 as run_sweep.py) ──────────────────────────
declare -A NORMS
NORMS["Log"]="log"
NORMS["PCEN"]="pcen"
NORMS["Box-Cox"]="boxcox"

echo ""
echo "############################################################"
echo "  Large DOC experiment"
echo "  Training data : $LARGE_DOC_TRAIN"
echo "  Test set 1    : $AVIANZ_TEST"
echo "  Test set 2    : $DOC_TEST"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "  DRY RUN — no commands will be executed"
fi
echo "############################################################"
echo ""

mkdir -p "$TESTS_BASE"

for NORM_LABEL in "Log" "PCEN" "Box-Cox"; do
    NORM_SLUG="${NORMS[$NORM_LABEL]}"
    RUN_NAME="${MODEL_TYPE}_on_large_doc_${NORM_SLUG}"
    OUT_DIR="$TESTS_BASE/$RUN_NAME"

    # Skip if already trained
    if [ -f "$OUT_DIR/${MODEL_TYPE}_model.pt" ] && [ "$DRY_RUN" -eq 0 ]; then
        echo "  [skip] $RUN_NAME (already trained)"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  $RUN_NAME"
    echo "============================================================"

    run_cmd python train.py \
        "$LARGE_DOC_TRAIN" "$OUT_DIR" \
        --test-folder  "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST" \
        --visualize-attention \
        --viz-samples  "$VIZ_SAMPLES" \
        --epochs       "$EPOCHS" \
        --patience     "$PATIENCE" \
        --mixup        "$MIXUP" \
        --model-type   "$MODEL_TYPE" \
        --spec-transform "$NORM_LABEL"

done

echo ""
echo "############################################################"
echo "  All large-doc training runs complete."
echo "  Run  python3 scripts/analyze_all_results.py  to compare."
echo "############################################################"
echo ""
