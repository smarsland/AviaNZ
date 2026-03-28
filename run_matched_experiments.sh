#!/bin/bash
set -e

# ============================================================
# Clean experiment pipeline
# 
# Does:
#   1. Build matched datasets (if not exist)
#   2. Split datasets (if not exist)
#   3. Run ALL experiments (Python handles caching)
#
# No flags. No skip logic. Just checks file existence.
# Want to rerun? Delete the folder.
# ============================================================

# Paths
AVIANZ_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Joe_MoDone?"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"
OUTPUT_BASE="/local/scratch/freangi"

REVIEWED_CSV="doc_reviewed.csv"
MAPPING="DOC_bird_naming_map.csv"

MATCHED_BASE="${OUTPUT_BASE}/matched"
DOC_MATCHED="${MATCHED_BASE}/doc_matched"
AVIANZ_MATCHED="${MATCHED_BASE}/avianz_matched"

DOC_SPLIT_BASE="${MATCHED_BASE}/doc_split"
DOC_TRAIN="${DOC_SPLIT_BASE}/train"
DOC_TEST="${DOC_SPLIT_BASE}/test"

AVIANZ_SPLIT_BASE="${MATCHED_BASE}/avianz_split"
AVIANZ_TRAIN="${AVIANZ_SPLIT_BASE}/train"
AVIANZ_TEST="${AVIANZ_SPLIT_BASE}/test"

NOISE_FOLDER="${OUTPUT_BASE}/noise"
RESULTS="${OUTPUT_BASE}/experiments_matched"

# Training config
EPOCHS=100
BATCH_SIZE=16
MIXUP_ALPHA=0.25
TEST_SIZE=0.25

echo "============================================================"
echo " Domain Shift Experiments"
echo "============================================================"
echo "  AviaNZ raw : $AVIANZ_RAW"
echo "  DOC raw    : $DOC_RAW"
echo "  Output     : $OUTPUT_BASE"
echo "============================================================"
echo ""

# ============================================================
# PHASE 1: DATASET PREPARATION
# ============================================================

# Build matched datasets (if not exist)
if [ ! -d "$DOC_MATCHED" ] || [ ! -d "$AVIANZ_MATCHED" ]; then
    echo "=== Building matched datasets ==="
    python3 build_matched_datasets.py \
        --reviewed-csv "$REVIEWED_CSV" \
        --doc-raw      "$DOC_RAW" \
        --avianz-raw   "$AVIANZ_RAW" \
        --output       "$MATCHED_BASE" \
        --mapping      "$MAPPING"
else
    echo "=== Matched datasets exist, skipping build ==="
fi

# Split datasets (if not exist)
if [ ! -d "$DOC_TRAIN" ] || [ ! -d "$AVIANZ_TRAIN" ]; then
    echo ""
    echo "=== Splitting datasets ==="
    python3 split_dataset.py "$DOC_MATCHED" "$DOC_SPLIT_BASE" \
        --test-ratio $TEST_SIZE \
        --overwrite
    
    python3 split_dataset.py "$AVIANZ_MATCHED" "$AVIANZ_SPLIT_BASE" \
        --test-ratio $TEST_SIZE \
        --overwrite
    
    echo ""
    echo "=== Validating splits ==="
    python3 validate_splits.py "$AVIANZ_TRAIN" "$AVIANZ_TEST" "$DOC_TRAIN" "$DOC_TEST"
else
    echo "=== Splits exist, skipping ==="
fi

# ============================================================
# PHASE 2: RUN ALL EXPERIMENTS
# ============================================================

echo ""
echo "============================================================"
echo " Running all experiments"
echo "============================================================"
echo " The Python script will run:"
echo "   - Normalization comparison (12 experiments)"
echo "   - DANN domain adaptation (2 experiments)"
echo "   - Noise augmentation sweep (10 experiments)"
echo ""
echo " Each experiment caches results automatically."
echo " To rerun: delete experiment folders in $RESULTS"
echo "============================================================"
echo ""

python3 run_cross_dataset_experiments.py \
    --avianz-train "$AVIANZ_TRAIN" \
    --avianz-test  "$AVIANZ_TEST" \
    --doc-train    "$DOC_TRAIN" \
    --doc-test     "$DOC_TEST" \
    --noise-folder "$NOISE_FOLDER" \
    --output       "$RESULTS" \
    --epochs       $EPOCHS \
    --batch-size   $BATCH_SIZE \
    --mixup        $MIXUP_ALPHA

echo ""
echo "============================================================"
echo " ALL DONE"
echo "============================================================"
echo " Results: $RESULTS"
echo "============================================================"
