#!/bin/bash
set -e

main() {

BASE="/local/scratch/freangi"
MATCHED="${BASE}/matched"
OUTPUT="${BASE}/tests"

AVIANZ_TRAIN="${MATCHED}/avianz_split/train"
AVIANZ_TEST="${MATCHED}/avianz_split/test"
DOC_TRAIN="${MATCHED}/doc_split/train"
DOC_TEST="${MATCHED}/doc_split/test"
EPOCHS=100
PATIENCE=15
MIXUP=0.25
VIZ_SAMPLES=10

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

    # Freeze most of the backbone, leaving the last few layers trainable
    local freeze_flags=()
    if [ "$model" = "ast" ]; then
        freeze_flags=(--freeze-layers 8)
    elif [ "$model" = "regnet" ]; then
        freeze_flags=(--freeze-stages 3)
    fi

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
        "${freeze_flags[@]}" \
        "${extra_flags[@]}"
}

# Default log transform
# run_experiment ast    doc    "$DOC_TRAIN"    log
# run_experiment ast    avianz "$AVIANZ_TRAIN" log
run_experiment regnet doc    "$DOC_TRAIN"    log
run_experiment regnet avianz "$AVIANZ_TRAIN" log

# # Log + background subtraction + median filter
# run_experiment ast    doc    "$DOC_TRAIN"    log_norm_med --bg-subtract --median-filter
# run_experiment ast    avianz "$AVIANZ_TRAIN" log_norm_med --bg-subtract --median-filter
# run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med --bg-subtract --median-filter
# run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med --bg-subtract --median-filter

# # With CNN adapter: default log transform
# run_experiment ast    doc    "$DOC_TRAIN"    log_cnn    --cnn-adapter
# run_experiment ast    avianz "$AVIANZ_TRAIN" log_cnn    --cnn-adapter
# run_experiment regnet doc    "$DOC_TRAIN"    log_cnn    --cnn-adapter
# run_experiment regnet avianz "$AVIANZ_TRAIN" log_cnn    --cnn-adapter

# # With CNN adapter: log + background subtraction + median filter
# run_experiment ast    doc    "$DOC_TRAIN"    log_norm_med_cnn --bg-subtract --median-filter --cnn-adapter
# run_experiment ast    avianz "$AVIANZ_TRAIN" log_norm_med_cnn --bg-subtract --median-filter --cnn-adapter
# run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med_cnn --bg-subtract --median-filter --cnn-adapter
# run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med_cnn --bg-subtract --median-filter --cnn-adapter

}

main "$@"

echo ""
echo "To run Kaytoo baseline evaluation on the same test sets:"
echo "  ./run_kaytoo_eval.sh"
