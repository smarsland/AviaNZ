#!/bin/bash
set -e

BASE="/local/scratch/freangi"
MATCHED="${BASE}/matched"
VIZ_BASE="${BASE}/visualizations"

AVIANZ_TEST="${MATCHED}/avianz_split/test"
DOC_TEST="${MATCHED}/doc_split/test"

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-}"

run_regen() {
    local model=$1
    local train_name=$2
    local transform_name=$3
    shift 3
    local extra_flags=("$@")

    local run_dir="${VIZ_BASE}/${model}_on_${train_name}_${transform_name}"

    echo "============================================================"
    echo " Regenerating attention | Model: $model | Train: $train_name | Transform: $transform_name"
    echo " Run dir: $run_dir"
    echo "============================================================"

    local cmd=(
        "$PYTHON_BIN" scripts/regenerate_attention.py "$run_dir"
        --dataset "$AVIANZ_TEST"
        --dataset "$DOC_TEST"
        --num-samples "$NUM_SAMPLES"
        --batch-size "$BATCH_SIZE"
    )

    if [[ -n "$DEVICE" ]]; then
        cmd+=(--device "$DEVICE")
    fi

    if [[ ${#extra_flags[@]} -gt 0 ]]; then
        cmd+=("${extra_flags[@]}")
    fi

    "${cmd[@]}"
}

# Default log transform
run_regen ast    doc    log
run_regen ast    avianz log
run_regen regnet doc    log
run_regen regnet avianz log

# Log + background subtraction + median filter
run_regen ast    doc    log_norm_med
run_regen ast    avianz log_norm_med
run_regen regnet doc    log_norm_med
run_regen regnet avianz log_norm_med