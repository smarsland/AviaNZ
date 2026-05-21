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

# ── 1. Baseline: RegNet on matched DOC data (Box-Cox) ────────────────────────
run_experiment regnet doc "$DOC_TRAIN" boxcox \
    --spec-transform "Box-Cox"

# ── 2. K-bird prior: soft constraint that ≤4 species are active per segment ──
#    Probabilities are normalised so their sum never exceeds k=4, encoding the
#    prior that at most ~4 birds call simultaneously.  Unlike the old k*softmax
#    approach this does not force competition between classes.
run_experiment regnet doc "$DOC_TRAIN" boxcox_kbird4 \
    --spec-transform "Box-Cox" --kbird-prior 4.0

# ── 3. Background normalisation: median filter + background subtraction ───────
run_experiment regnet doc "$DOC_TRAIN" boxcox_bgmed \
    --spec-transform "Box-Cox" --bg-subtract --median-filter

# ── 4. AviaNZ domain-shift comparison ────────────────────────────────────────
#    Same as run 1 but trained on AviaNZ matched data.  The gap between this and
#    run 1 (DOC) on the two test sets quantifies cross-dataset domain shift.
run_experiment regnet avianz "$AVIANZ_TRAIN" boxcox \
    --spec-transform "Box-Cox"

echo ""
echo "============================================================"
echo " All matched experiments complete."
echo " Results: $OUTPUT"
echo ""
echo " Next steps:"
echo "   ./run_kaytoo_eval.sh   — Kaytoo baseline on matched test sets"
echo "   ./run_birdnet_eval.sh  — BirdNET baseline on matched test sets"
echo "============================================================"
