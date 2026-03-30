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
#
# Usage:
#   ./run_matched_experiments.sh                      # Variable-length spectrograms
#   ./run_matched_experiments.sh --fixed-length       # Fixed-length mode (trim to 1024 bins)
#   ./run_matched_experiments.sh --parallel 2         # Run 2 experiments in parallel (use GPUs)
#   ./run_matched_experiments.sh --fixed-length --parallel 0  # Auto-detect GPUs
# ============================================================

# Parse arguments
FIXED_LENGTH_FLAG=""
PARALLEL_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --fixed-length)
            FIXED_LENGTH_FLAG="--fixed-length"
            echo "Fixed-length mode enabled"
            shift
            ;;
        --parallel)
            PARALLEL_FLAG="--parallel $2"
            echo "Parallel mode: $2 workers"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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
        --mapping      "$MAPPING" \
        $FIXED_LENGTH_FLAG
else
    echo "=== Matched datasets exist, skipping build ==="
fi

# Split datasets (if not exist)
# Uses file-level splitting for AviaNZ and distribution-matched splitting for DOC
# 
# AviaNZ: All segments from the same audio file go to train OR test (never both)
#         This prevents data leakage from similar recording conditions
# 
# DOC:    Split to match the species distribution that resulted from AviaNZ split
#         This ensures both datasets have similar class balance in train/test
if [ ! -d "$DOC_TRAIN" ] || [ ! -d "$AVIANZ_TRAIN" ]; then
    echo ""
    echo "=== Splitting datasets (file-level + distribution-matched) ==="
    python3 split_matched_datasets.py \
        "$AVIANZ_MATCHED" \
        "$DOC_MATCHED" \
        "$MATCHED_BASE" \
        --test-ratio $TEST_SIZE \
        --seed 42 \
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
echo "   - Noise intensity sweep (10 experiments)"
echo "   - Noise variety sweep (10 experiments)"
echo ""
echo " Total: ~34 experiments"
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
    --mixup        $MIXUP_ALPHA \
    $PARALLEL_FLAG

echo ""
echo "============================================================"
echo " ALL DONE"
echo "============================================================"
echo " Results: $RESULTS"
echo "============================================================"
