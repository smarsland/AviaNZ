#!/bin/bash
set -e

BASE="/local/scratch/freangi"
MATCHED="${BASE}/matched"
VIZ_BASE="${BASE}/visualizations"

AVIANZ_TRAIN="${MATCHED}/avianz_split/train"
AVIANZ_TEST="${MATCHED}/avianz_split/test"
DOC_TRAIN="${MATCHED}/doc_split/train"
DOC_TEST="${MATCHED}/doc_split/test"

EPOCHS=100
PATIENCE=15
MIXUP=0.25
VIZ_SAMPLES=30

run_experiment() {
    local model=$1
    local train_name=$2
    local train_dir=$3
    local transform_name=$4
    shift 4
    local extra_flags=("$@")

    local out_dir="${VIZ_BASE}/${model}_on_${train_name}_${transform_name}"

    echo "============================================================"
    echo " Model: $model | Train: $train_name | Transform: $transform_name"
    echo " Output: $out_dir"
    echo "============================================================"

    python train.py \
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

# Default log transform
run_experiment ast    doc    "$DOC_TRAIN"    log
#run_experiment ast    avianz "$AVIANZ_TRAIN" log
#run_experiment regnet doc    "$DOC_TRAIN"    log
#run_experiment regnet avianz "$AVIANZ_TRAIN" log

# Log + background subtraction + median filter
#run_experiment ast    doc    "$DOC_TRAIN"    log_norm_med --bg-subtract --median-filter
#run_experiment ast    avianz "$AVIANZ_TRAIN" log_norm_med --bg-subtract --median-filter
#run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med --bg-subtract --median-filter
#run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med --bg-subtract --median-filter
