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
MERGED_TRAIN="${MATCHED}/merged_train"
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

# # Default log transform
# run_experiment ast    doc    "$DOC_TRAIN"    log
# run_experiment ast    avianz "$AVIANZ_TRAIN" log
# run_experiment regnet doc    "$DOC_TRAIN"    log
# run_experiment regnet avianz "$AVIANZ_TRAIN" log

# # Per-clip normalization: replaces global AudioSet stats (tests normalization hypothesis)
# run_experiment ast    doc    "$DOC_TRAIN"    log_clip_norm --per-chunk-norm
# run_experiment ast    avianz "$AVIANZ_TRAIN" log_clip_norm --per-chunk-norm

# # Log + background subtraction + median filter
# run_experiment ast    doc    "$DOC_TRAIN"    log_norm_med --bg-subtract --median-filter
# run_experiment ast    avianz "$AVIANZ_TRAIN" log_norm_med --bg-subtract --median-filter
# run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med --bg-subtract --median-filter
# run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med --bg-subtract --median-filter

# # Merged with log + background subtraction + median filter
# run_experiment ast    merged  "$MERGED_TRAIN" log_norm_med --bg-subtract --median-filter
# run_experiment regnet merged  "$MERGED_TRAIN" log_norm_med --bg-subtract --median-filter

# # CNN adapter: learnable CNN front-end prepended to backbone (trained at 10x LR)
# run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med_cnn --bg-subtract --median-filter --cnn-adapter
# run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med_cnn --bg-subtract --median-filter --cnn-adapter
# run_experiment ast    doc    "$DOC_TRAIN"    log_norm_med_cnn --bg-subtract --median-filter --cnn-adapter
# run_experiment ast    avianz "$AVIANZ_TRAIN" log_norm_med_cnn --bg-subtract --median-filter --cnn-adapter

# # Target-domain noise: real acoustic noise from the other domain (0.2 mixing ratio)
# run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med_tgt_noise --bg-subtract --median-filter --noise 0.2 --noise-folder "$AVIANZ_TRAIN"
# run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med_tgt_noise --bg-subtract --median-filter --noise 0.2 --noise-folder "$DOC_TRAIN"

# # Partial backbone freezing: keep pretrained early stages fixed, fine-tune upper stages only
# run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med_freeze2 --bg-subtract --median-filter --freeze-stages 2
# run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med_freeze2 --bg-subtract --median-filter --freeze-stages 2
# run_experiment ast    doc    "$DOC_TRAIN"    log_freeze6 --freeze-layers 6
# run_experiment ast    avianz "$AVIANZ_TRAIN" log_freeze6 --freeze-layers 6

# # Per-class temporal attention head (SED-style): replaces global avg pool so each
# # class attends to its own time windows instead of averaging over everything.
# run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med_sed --bg-subtract --median-filter --sed-head
# run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med_sed --bg-subtract --median-filter --sed-head
# run_experiment regnet merged  "$MERGED_TRAIN" log_norm_med_sed --bg-subtract --median-filter --sed-head
# run_experiment ast    doc    "$DOC_TRAIN"    log_norm_med_sed --bg-subtract --median-filter --sed-head
# run_experiment ast    avianz "$AVIANZ_TRAIN" log_norm_med_sed --bg-subtract --median-filter --sed-head
# run_experiment ast    merged  "$MERGED_TRAIN" log_norm_med_sed --bg-subtract --median-filter --sed-head

run_pseudo_experiment() {
    local model=$1
    local source_name=$2
    local source_dir=$3
    local target_name=$4
    local target_dir=$5
    local transform_name=$6
    local pseudo_pct=$7
    shift 7
    local extra_flags=("$@")

    local pct_int=$(python3 -c "print(int($pseudo_pct * 100))")
    local out_dir="${OUTPUT}/${model}_pseudo_${source_name}_to_${target_name}_${transform_name}_pct${pct_int}"

    echo "============================================================"
    echo " Pseudo: $model | Source: $source_name | Target: $target_name | Subset: ${pct_int}%"
    echo " Output: $out_dir"
    echo "============================================================"

    python pseudo_train.py \
        "$source_dir" \
        "$target_dir" \
        "$out_dir" \
        --test-folder "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST" \
        --epochs $EPOCHS \
        --patience $PATIENCE \
        --mixup $MIXUP \
        --model-type "$model" \
        --pseudo-pct "$pseudo_pct" \
        "${extra_flags[@]}"
}

# Pseudo-label training: train on doc, adapt to avianz via pseudo labels
# Three subset sizes: 25%, 50%, 100% of avianz used in phase 2 (real labels before pseudo generation)
run_pseudo_experiment regnet doc "$DOC_TRAIN" avianz "$AVIANZ_TRAIN" log_norm_med 0.25 --bg-subtract --median-filter
run_pseudo_experiment regnet doc "$DOC_TRAIN" avianz "$AVIANZ_TRAIN" log_norm_med 0.50 --bg-subtract --median-filter
run_pseudo_experiment regnet doc "$DOC_TRAIN" avianz "$AVIANZ_TRAIN" log_norm_med 1.00 --bg-subtract --median-filter
run_pseudo_experiment regnet avianz "$AVIANZ_TRAIN" doc "$DOC_TRAIN" log_norm_med 0.25 --bg-subtract --median-filter
run_pseudo_experiment regnet avianz "$AVIANZ_TRAIN" doc "$DOC_TRAIN" log_norm_med 0.50 --bg-subtract --median-filter
run_pseudo_experiment regnet avianz "$AVIANZ_TRAIN" doc "$DOC_TRAIN" log_norm_med 1.00 --bg-subtract --median-filter

}

main "$@"

echo ""
echo "To run Kaytoo baseline evaluation on the same test sets:"
echo "  ./run_kaytoo_eval.sh"
