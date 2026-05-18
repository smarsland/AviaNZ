#!/bin/bash
set -e

main() {

BASE="/local/scratch/freangi"
MATCHED="${BASE}/matched"
OUTPUT="${BASE}/matched_tests"

AVIANZ_TRAIN="${MATCHED}/avianz_split/train"
AVIANZ_TEST="${MATCHED}/avianz_split/test"
DOC_TRAIN="${MATCHED}/doc_split/train"
DOC_TEST="${MATCHED}/doc_split/test"
LARGE_DOC_TRAIN="${BASE}/large/doc_split/train"
EPOCHS=100
PATIENCE=15
MIXUP=0.25
VIZ_SAMPLES=3

run_experiment() {
    local model=$1
    local train_name=$2
    local train_dir=$3
    local transform_name=$4
    shift 4
    local extra_flags=("$@")

    local out_dir="${OUTPUT}/${model}_on_${train_name}_${transform_name}"

    echo "============================================================"
    echo " Model: $model | Train: $train_name | Transform: $transform_name"
    echo " Output: $out_dir"
    echo "============================================================"

    PYTHONPATH="$PWD" python train.py \
        "$train_dir" \
        "$out_dir" \
        --test-folder "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST" \
        --visualize-attention \
        --viz-samples $VIZ_SAMPLES \
        --epochs $EPOCHS \
        --patience $PATIENCE \
        --mixup $MIXUP \
        --model-type "$model" \
        "${extra_flags[@]}"
}

# RegNet trained on matched DOC data
run_experiment regnet    doc       "$DOC_TRAIN"       log_norm_med --bg-subtract --median-filter

# RegNet trained on matched AviaNZ data (benchmark)
run_experiment regnet    avianz    "$AVIANZ_TRAIN"    log_norm_med --bg-subtract --median-filter

# AST trained on matched DOC data
run_experiment ast       doc       "$DOC_TRAIN"       log_norm_med --bg-subtract --median-filter --per-chunk-norm

# RegNet trained on full (large) DOC data, evaluated on matched test sets
run_experiment regnet    large_doc "$LARGE_DOC_TRAIN" log_norm_med --bg-subtract --median-filter

}

main "$@"

echo ""
echo "To run Kaytoo baseline evaluation on the same test sets:"
echo "  ./run_kaytoo_eval.sh"
