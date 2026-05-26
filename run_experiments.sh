#!/bin/bash
set -e
#
# Train RegNet models on the matched dataset and evaluate on both test sets.
# Run after build_dataset.sh.
#
# Four runs:
#   1. DOC matched train — normal RegNet (Box-Cox, baseline)
#   2. DOC matched train — k-bird prior (max-4 normalisation; soft species-count constraint)
#   3. DOC matched train — background subtraction + median filter
#   4. AviaNZ matched train — normal RegNet (domain-shift comparison vs run 1)
#   5. DOC matched train — no background samples (train on labelled-only)
#   6. DOC matched train — delta + delta-delta channels (3-ch input, recording-condition robustness)
#   7. AviaNZ matched train — delta + delta-delta channels
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

EPOCHS=100
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

run_experiment regnet doc "$DOC_TRAIN" baseline_doc \
    --spec-transform "Box-Cox" --kbird-prior 4.0

# run_experiment regnet doc "$DOC_TRAIN" bgmed_doc \
#     --spec-transform "Box-Cox" --bg-subtract --median-filter --kbird-prior 4.0

# run_experiment regnet avianz "$AVIANZ_TRAIN" baseline_avianz \
#     --spec-transform "Box-Cox" --kbird-prior 4.0 

# run_experiment regnet doc "$DOC_TRAIN" no_background_doc \
#     --spec-transform "Box-Cox" --no-background --bg-subtract --median-filter --kbird-prior 4.0

# run_experiment regnet doc "$DOC_TRAIN" delta_doc \
#     --spec-transform "Box-Cox" --deltas --kbird-prior 4.0

# run_experiment regnet doc "$DOC_TRAIN" sed_head_doc \
#     --spec-transform "Box-Cox" --kbird-prior 4.0 --sed-head

echo ""
echo "============================================================"
echo " All matched experiments complete."
echo " Results: $OUTPUT"
echo ""
echo " Next steps:"
echo "   ./run_kaytoo_eval.sh   — Kaytoo baseline on matched test sets"
echo "   ./run_birdnet_eval.sh  — BirdNET baseline on matched test sets"
echo "============================================================"
