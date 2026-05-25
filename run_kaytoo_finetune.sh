#!/bin/bash
set -e

# Fine-tune the pre-trained Kaytoo model on the matched dataset with corrected
# labels, then evaluate on both test sets.
# Run this after build_dataset.sh.
# Must be run from the AviaNZ project root.
#
# Usage:
#   ./run_kaytoo_finetune.sh              # fine-tune + evaluate on matched sets
#   ./run_kaytoo_finetune.sh --large      # fine-tune + evaluate on large sets
#   ./run_kaytoo_finetune.sh --cpu        # force CPU
#   ./run_kaytoo_finetune.sh --epochs 20  # override epoch count (default: 10)
#   ./run_kaytoo_finetune.sh --lr 5e-5    # override peak learning rate

CPU_FLAG=""
LARGE=false
EPOCHS=50
LR=1e-4
BATCH_SIZE=16
NUM_WORKERS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)        CPU_FLAG="--cpu"; shift ;;
        --large)      LARGE=true; shift ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --lr)         LR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --workers)    NUM_WORKERS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

OUTPUT_BASE="/local/scratch/freangi"
KAYTOO_ROOT="$(pwd)/../Kaytoo"
KAYTOO_PYTHON="${KAYTOO_ROOT}/venv_kay/bin/python"

MAPPING="data/DOC_bird_naming_map.csv"

if [ "$LARGE" = true ]; then
    AVIANZ_TRAIN="${OUTPUT_BASE}/large/avianz_split/train"
    AVIANZ_TEST="${OUTPUT_BASE}/large/avianz_split/test"
    DOC_TRAIN="${OUTPUT_BASE}/large/doc_split/train"
    DOC_TEST="${OUTPUT_BASE}/large/doc_split/test"
    OUTPUT="${OUTPUT_BASE}/large_tests/kaytoo_finetuned_seed0"
else
    AVIANZ_TRAIN="${OUTPUT_BASE}/matched/avianz_split/train"
    AVIANZ_TEST="${OUTPUT_BASE}/matched/avianz_split/test"
    DOC_TRAIN="${OUTPUT_BASE}/matched/doc_split/train"
    DOC_TEST="${OUTPUT_BASE}/matched/doc_split/test"
    OUTPUT="${OUTPUT_BASE}/matched_tests/kaytoo_finetuned_seed0"
fi

echo "============================================================"
echo " Kaytoo fine-tuning + evaluation"
echo "============================================================"
echo "  Kaytoo root   : $KAYTOO_ROOT"
echo "  Python        : $KAYTOO_PYTHON"
echo "  AviaNZ train  : $AVIANZ_TRAIN"
echo "  AviaNZ test   : $AVIANZ_TEST"
echo "  DOC train     : $DOC_TRAIN"
echo "  DOC test      : $DOC_TEST"
echo "  Output        : $OUTPUT"
echo "  Epochs        : $EPOCHS"
echo "  Learning rate : $LR"
echo "  Batch size    : $BATCH_SIZE"
echo "============================================================"
echo ""

if [ ! -f "$KAYTOO_PYTHON" ]; then
    echo "ERROR: Kaytoo Python not found at $KAYTOO_PYTHON"
    echo "Check that venv_kay exists inside $KAYTOO_ROOT"
    exit 1
fi

for SPLIT_DIR in "$AVIANZ_TRAIN" "$AVIANZ_TEST" "$DOC_TRAIN" "$DOC_TEST"; do
    if [ ! -d "${SPLIT_DIR}/audio" ]; then
        echo "ERROR: audio/ subfolder missing from $SPLIT_DIR"
        echo "Re-run build_dataset.sh (it saves audio automatically)."
        exit 1
    fi
done

mkdir -p "$OUTPUT"

PYTHONPATH="$PWD" "$KAYTOO_PYTHON" scripts/finetune_kaytoo.py \
    --avianz-train "$AVIANZ_TRAIN" \
    --avianz-test  "$AVIANZ_TEST" \
    --doc-train    "$DOC_TRAIN" \
    --doc-test     "$DOC_TEST" \
    --kaytoo-root  "$KAYTOO_ROOT" \
    --mapping      "$MAPPING" \
    --output       "$OUTPUT" \
    --epochs       "$EPOCHS" \
    --lr           "$LR" \
    --batch-size   "$BATCH_SIZE" \
    --num-workers  "$NUM_WORKERS" \
    $CPU_FLAG

echo ""
echo "============================================================"
echo " Done. Results in $OUTPUT"
echo "============================================================"
