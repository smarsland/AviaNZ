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
# run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med_no_bg --bg-subtract --median-filter --no-background
# run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med_no_bg --bg-subtract --median-filter --no-background

# Two-stage gated head: bird-presence gate + species classifier.
# Background samples are kept so the gate can learn to distinguish them.
run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med_gated --bg-subtract --median-filter --gated-head
run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med_gated --bg-subtract --median-filter --gated-head

# ---------------------------------------------------------------------------
# NEW EXPERIMENTS: Diagnosing why background data hurts performance
# ---------------------------------------------------------------------------

# EXP 1: Asymmetric Loss (ASL)
# ASL clips the gradient from easy negatives (the background problem in a nutshell:
# every background sample pushes every class logit down equally).
# With background KEPT, ASL's margin zeros out easy-negative gradients so the
# background samples stop suppressing rare species.
run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med_asl   --bg-subtract --median-filter --use-asl
run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med_asl   --bg-subtract --median-filter --use-asl

# EXP 2: no-background + class weights
# Removing background fixes gradient suppression; adding class weights then addresses
# the next problem: imbalance between common and rare species in the labelled set.
run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med_no_bg_cw --bg-subtract --median-filter --no-background --class-weights
run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med_no_bg_cw --bg-subtract --median-filter --no-background --class-weights

# EXP 3: no-background + SED head
# The SED head's per-class temporal attention was previously being poisoned by background
# samples (attending to silence). With no background, it gets a clean learning signal.
run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med_no_bg_sed --bg-subtract --median-filter --no-background --sed-head
run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med_no_bg_sed --bg-subtract --median-filter --no-background --sed-head

# EXP 4: no-background + freeze early stages
# BirdClef pretraining already gives good low-level features. Freezing stages 1-2
# prevents the backbone from overwriting them to fit source-domain quirks, which
# should improve cross-domain transfer to the held-out test set.
run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med_no_bg_freeze2 --bg-subtract --median-filter --no-background --freeze-stages 2
run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med_no_bg_freeze2 --bg-subtract --median-filter --no-background --freeze-stages 2

# EXP 5: Cross-domain noise augmentation
# Train on one domain, mix in 20% of the OTHER domain's audio at each step
# (labels ignored — used as acoustic noise only). Soft domain adaptation:
# the model hears target-domain recording conditions without DANN complexity.
run_experiment regnet doc    "$DOC_TRAIN"    log_norm_med_xnoise --bg-subtract --median-filter --noise 0.2 --noise-folder "$AVIANZ_TRAIN"
run_experiment regnet avianz "$AVIANZ_TRAIN" log_norm_med_xnoise --bg-subtract --median-filter --noise 0.2 --noise-folder "$DOC_TRAIN"

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

}

main "$@"

echo ""
echo "To run Kaytoo baseline evaluation on the same test sets:"
echo "  ./run_kaytoo_eval.sh"
