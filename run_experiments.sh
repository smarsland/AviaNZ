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

# Pre-computed AST attention map cache (flat dir, one map per spectrogram file)
AST_ATTN_DIR="${BASE}/ast_attn_cache_boxcox"

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

# Pre-compute AST attention maps for all data splits (run once; skips existing files).
# Maps are stored flat in AST_ATTN_DIR so train and test files share one directory.
precompute_ast_attention() {
    echo "============================================================"
    echo " Precomputing raw AST attention maps (Box-Cox)"
    echo " Output: $AST_ATTN_DIR"
    echo "============================================================"
    PYTHONPATH="$PWD" python scripts/precompute_ast_attention.py \
        "$AVIANZ_TRAIN" "$AVIANZ_TEST" "$DOC_TRAIN" "$DOC_TEST" \
        --out-dir "$AST_ATTN_DIR" \
        --spec-transform "Box-Cox"
}

# # RegNet trained on matched DOC data  (Box-Cox = best config from sweep)
# run_experiment regnet    doc       "$DOC_TRAIN"       boxcox --spec-transform "Box-Cox"

# # RegNet trained on matched AviaNZ data (benchmark)
# run_experiment regnet    avianz    "$AVIANZ_TRAIN"    boxcox --spec-transform "Box-Cox"

# RegNet with softmax-4 output: each segment is treated as containing ≤~4 birds.
# Instead of independent sigmoids, the model outputs k*softmax(logits) so
# probabilities sum to k≈4, providing a soft upper-bound on predicted species count.
# run_experiment regnet    doc       "$DOC_TRAIN"       boxcox_softmax4 \
#     --spec-transform "Box-Cox" --softmax-scale 4.0

# RegNet with raw AST attention as a second input channel.
# Step 1: pre-compute attention maps from the frozen AudioSet-pretrained AST
#         (run once; subsequent calls skip existing files).
precompute_ast_attention
# Step 2: train with 2-channel input (spectrogram + attention map)
run_experiment regnet    doc       "$DOC_TRAIN"       boxcox_astchan \
    --spec-transform "Box-Cox" --ast-channel-dir "$AST_ATTN_DIR"

# # AST trained on matched DOC data
# run_experiment ast       doc       "$DOC_TRAIN"       boxcox --spec-transform "Box-Cox" --per-chunk-norm

# # RegNet trained on full (large) DOC data, evaluated on matched test sets
# run_experiment regnet    large_doc "$LARGE_DOC_TRAIN" boxcox --spec-transform "Box-Cox"

}

main "$@"

echo ""
echo "To run Kaytoo baseline evaluation on the same test sets:"
echo "  ./run_kaytoo_eval.sh"
