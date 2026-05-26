#!/bin/bash
set -e
#
# Train RegNet models on the matched dataset and evaluate on both test sets.
# Run after build_dataset.sh.
#
# Ten runs (all on DOC matched train):
#    1. baseline             — Log transform
#    2. boxcox               — Box-Cox transform
#    3. kbird2               — baseline + k-bird prior 2
#    4. kbird4               — baseline + k-bird prior 4
#    5. bgsub                — baseline + background subtraction
#    6. bgmed                — baseline + background subtraction + median filter
#    7. no_background        — baseline + no background samples
#    8. delta                — baseline + delta + delta-delta channels
#    9. sed_head             — baseline + SED head
#   10. logminmax            — LogMinMax transform (Kaytoo-style)
#
# Results land in $OUTPUT and are picked up by scripts/analyze_all_results.py.
#
# Usage:
#   bash run_experiments.sh
#   bash run_experiments.sh --dry-run

set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

BASE="/local/scratch/freangi"
MATCHED="${BASE}/matched"
OUTPUT="${BASE}/matched_tests"

AVIANZ_TRAIN="${MATCHED}/avianz_split/train"
AVIANZ_TEST="${MATCHED}/avianz_split/test"
DOC_TRAIN="${MATCHED}/doc_split/train"
DOC_TEST="${MATCHED}/doc_split/test"

EPOCHS=30
PATIENCE=15
MIXUP=0.25
VIZ_SAMPLES=3

run_cmd() {
    echo "  \$ $*"
    if [ "$DRY_RUN" -eq 0 ]; then
        PYTHONPATH="$(pwd)" "$@"
    fi
}

run_experiment() {
    local model=$1
    local train_name=$2
    local train_dir=$3
    local transform_name=$4
    shift 4

    local out_dir="${OUTPUT}/${model}_on_${train_name}_${transform_name}"

    # Skip if already trained
    if [ -f "${out_dir}/${model}_model.pt" ] && [ "$DRY_RUN" -eq 0 ]; then
        echo "  [skip] ${model}_on_${train_name}_${transform_name} (already trained)"
        return
    fi

    echo ""
    echo "============================================================"
    echo " ${model}_on_${train_name}_${transform_name}"
    echo "============================================================"

    run_cmd python train.py \
        "$train_dir" \
        "$out_dir" \
        --test-folder  "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST" \
        --visualize-attention \
        --viz-samples  $VIZ_SAMPLES \
        --epochs       $EPOCHS \
        --patience     $PATIENCE \
        --mixup        $MIXUP \
        --model-type   "$model" \
        "$@"
}

mkdir -p "$OUTPUT"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN — no commands will be executed"
fi

# 1. Baseline — Log transform (default)
run_experiment regnet doc "$DOC_TRAIN" baseline \
    --spec-transform "Log"

# 2. Alternative — Box-Cox transform
run_experiment regnet doc "$DOC_TRAIN" boxcox \
    --spec-transform "Box-Cox"

# 3. Baseline + k-bird prior of 2
run_experiment regnet doc "$DOC_TRAIN" kbird2 \
    --spec-transform "Log" --kbird-prior 2.0

# 4. Baseline + k-bird prior of 4
run_experiment regnet doc "$DOC_TRAIN" kbird4 \
    --spec-transform "Log" --kbird-prior 4.0

# 5. Baseline + background subtraction
run_experiment regnet doc "$DOC_TRAIN" bgsub \
    --spec-transform "Log" --bg-subtract

# 6. Baseline + background subtraction + median filter
run_experiment regnet doc "$DOC_TRAIN" bgmed \
    --spec-transform "Log" --bg-subtract --median-filter

# 7. Baseline + no background samples
run_experiment regnet doc "$DOC_TRAIN" no_background \
    --spec-transform "Log" --no-background

# 8. Baseline + delta + delta-delta channels
run_experiment regnet doc "$DOC_TRAIN" delta \
    --spec-transform "Log" --deltas

# 9. Baseline + SED head
run_experiment regnet doc "$DOC_TRAIN" sed_head \
    --spec-transform "Log" --sed-head

# 10. Baseline — LogMinMax transform (Kaytoo-style)
run_experiment regnet doc "$DOC_TRAIN" logminmax \
    --spec-transform "LogMinMax"
echo ""
echo "============================================================"
echo " All matched experiments complete."
echo " Results: $OUTPUT"
echo ""
echo " Next steps:"
echo "   ./run_kaytoo_eval.sh   — Kaytoo baseline on matched test sets"
echo "   ./run_birdnet_eval.sh  — BirdNET baseline on matched test sets"
echo "============================================================"
