#!/bin/bash
set -e

# ============================================================
# Clean experiment pipeline
# 
# Does:
#   1. Build matched datasets (if not exist)
#   2. Split datasets (if not exist)
#   3. Merge train datasets (if not exist)
#   4. Run ALL experiments (Python handles caching)
#
# Usage:
#   ./run_matched_experiments.sh                         # Use all defaults
#   ./run_matched_experiments.sh --parallel 1           # Force single-threaded
#   ./run_matched_experiments.sh --results-dir ./output # Custom results location
#
# Multi-machine setup:
#   Just run on each machine - results automatically saved to ~/results (shared)
#   Large model files (.pt) stay in /local/scratch (machine-specific)
#   Small result files (JSON, CSV) copied to ~/results (shared across machines)
# ============================================================

# Default settings
PARALLEL_FLAG="--parallel 0"  # Auto-detect GPUs
RESULTS_DIR="$HOME/results"

# Parse arguments (to override defaults if needed)
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL_FLAG="--parallel $2"
            echo "Parallel mode: $2 workers"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            echo "Shared results directory: $RESULTS_DIR"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Valid options: --parallel N, --results-dir DIR"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo " Configuration"
echo "============================================================"
echo "  Parallel mode    : ${PARALLEL_FLAG#--parallel }"
echo "  Results directory: $RESULTS_DIR"
echo "============================================================"
echo ""

# Paths
AVIANZ_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Joe_MoDone?"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"
OUTPUT_BASE="/local/scratch/freangi"

REVIEWED_CSV="data/doc_reviewed.csv"
MAPPING="data/DOC_bird_naming_map.csv"

MATCHED_BASE="${OUTPUT_BASE}/matched"
DOC_MATCHED="${MATCHED_BASE}/doc_matched"
AVIANZ_MATCHED="${MATCHED_BASE}/avianz_matched"  # Waitākere dataset

DOC_SPLIT_BASE="${MATCHED_BASE}/doc_split"
DOC_TRAIN="${DOC_SPLIT_BASE}/train"
DOC_TEST="${DOC_SPLIT_BASE}/test"

AVIANZ_SPLIT_BASE="${MATCHED_BASE}/avianz_split"  # Waitākere dataset splits
AVIANZ_TRAIN="${AVIANZ_SPLIT_BASE}/train"
AVIANZ_TEST="${AVIANZ_SPLIT_BASE}/test"

MERGED_TRAIN="${MATCHED_BASE}/merged_train"  # Combined DOC + Waitākere training data

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
echo "  Waitākere raw : $AVIANZ_RAW"
echo "  DOC raw       : $DOC_RAW"
echo "  Output        : $OUTPUT_BASE"
echo "============================================================"
echo ""

# ============================================================
# PHASE 1: DATASET PREPARATION
# ============================================================

# Build matched datasets (if not exist)
if [ ! -d "$DOC_MATCHED" ] || [ ! -f "$DOC_MATCHED/labels.json" ] || [ ! -d "$AVIANZ_MATCHED" ] || [ ! -f "$AVIANZ_MATCHED/labels.json" ]; then
    echo "=== Building matched datasets (fixed-length spectrograms) ==="
    PYTHONPATH="$PWD" python3 src/experiments/build_matched_datasets.py \
        --reviewed-csv "$REVIEWED_CSV" \
        --doc-raw      "$DOC_RAW" \
        --avianz-raw   "$AVIANZ_RAW" \
        --output       "$MATCHED_BASE" \
        --mapping      "$MAPPING" \
        --fixed-length
else
    echo "=== Matched datasets exist, skipping build ==="
fi

# Split datasets (if not exist)
# Uses file-level splitting for Waitākere and distribution-matched splitting for DOC
# 
# Waitākere: All segments from the same audio file go to train OR test (never both)
#            This prevents data leakage from similar recording conditions
# 
# DOC:       Split to match the species distribution that resulted from Waitākere split
#            This ensures both datasets have similar class balance in train/test
if [ ! -d "$DOC_TRAIN" ] || [ ! -f "$DOC_TRAIN/labels.json" ] || [ ! -d "$AVIANZ_TRAIN" ] || [ ! -f "$AVIANZ_TRAIN/labels.json" ]; then
    echo ""
    echo "=== Splitting datasets (file-level + distribution-matched) ==="
    PYTHONPATH="$PWD" python3 src/experiments/split_matched_datasets.py \
        "$AVIANZ_MATCHED" \
        "$DOC_MATCHED" \
        "$MATCHED_BASE" \
        --test-ratio $TEST_SIZE \
        --seed 42 \
        --overwrite
    
    echo ""
    echo "=== Validating splits ==="
    PYTHONPATH="$PWD" python3 src/experiments/validate_splits.py "$AVIANZ_TRAIN" "$AVIANZ_TEST" "$DOC_TRAIN" "$DOC_TEST"
else
    echo "=== Splits exist, skipping ==="
fi

# Merge train datasets (DOC + Waitākere) for combined training experiments
if [ ! -d "$MERGED_TRAIN" ] || [ ! -f "$MERGED_TRAIN/labels.json" ]; then
    echo ""
    echo "=== Merging training datasets (DOC + Waitākere) ==="
    PYTHONPATH="$PWD" python3 src/experiments/merge_datasets.py \
        "$DOC_TRAIN" \
        "$AVIANZ_TRAIN" \
        "$MERGED_TRAIN" \
        --symlink \
        --no-audio
else
    echo ""
    echo "=== Merged training dataset exists, skipping ==="
fi

# ============================================================
# PHASE 2: LOAD NOISE DATA (OPTIONAL)
# ============================================================

# Noise data path from shared storage
NOISE_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/freefield"
NUM_NOISE_SAMPLES=1000

# Noise data is OPTIONAL - used for noise augmentation experiments (suites 3-4)
if [ ! -d "$NOISE_FOLDER" ] || [ ! -f "$NOISE_FOLDER/labels.json" ]; then
    # Check if raw noise data is available
    if [ -d "$NOISE_RAW" ]; then
        echo ""
        echo "=== Loading noise data from freefield recordings ==="
        PYTHONPATH="$PWD" python3 -m src.data.dataset_builder noise "$NOISE_RAW" "$NOISE_FOLDER" --samples $NUM_NOISE_SAMPLES
        
        if [ -f "$NOISE_FOLDER/labels.json" ]; then
            noise_count=$(find "$NOISE_FOLDER/data" -name "*.npy" 2>/dev/null | wc -l)
            echo "=== Noise data loaded: $noise_count files ==="
        else
            echo "WARNING: Noise loading failed - continuing without noise experiments"
        fi
    else
        echo ""
        echo "============================================================"
        echo " WARNING: Noise data not available"
        echo "============================================================"
        echo "  Raw noise not found at: $NOISE_RAW"
        echo "  Processed noise not found at: $NOISE_FOLDER"
        echo ""
        echo "  Impact: Noise augmentation experiments will be SKIPPED:"
        echo "    - Noise intensity sweep (suite 3)"
        echo "    - Noise variety sweep (suite 4)"
        echo ""
        echo "  Continuing with normalization and DANN experiments only..."
        echo "============================================================"
        echo ""
    fi
else
    noise_count=$(find "$NOISE_FOLDER/data" -name "*.npy" 2>/dev/null | wc -l)
    echo ""
    echo "=== Noise data already exists: $noise_count files ==="
fi

# ============================================================
# PHASE 3: RUN ALL EXPERIMENTS
# ============================================================

echo ""
echo "============================================================"
echo " Running all experiments"
echo "============================================================"
if [ -d "$NOISE_FOLDER" ] && [ -f "$NOISE_FOLDER/labels.json" ]; then
    echo " The Python script will run:"
    echo "   - Normalization comparison (12 experiments)"
    echo "   - DANN domain adaptation (2 experiments)"
    echo "   - Noise intensity sweep (10 experiments)"
    echo "   - Noise variety sweep (10 experiments)"
    echo "   - Merged dataset experiments (2 experiments)"
    echo ""
    echo " Total: ~36 experiments"
else
    echo " The Python script will run:"
    echo "   - Normalization comparison (12 experiments)"
    echo "   - DANN domain adaptation (2 experiments)"
    echo "   - Merged dataset experiments (2 experiments)"
    echo ""
    echo " Total: ~16 experiments (noise experiments skipped)"
fi
echo ""
echo " Each experiment caches results automatically."
echo " To rerun: delete experiment folders in $RESULTS"
echo "============================================================"
echo ""

PYTHONPATH="$PWD" python3 src/experiments/run_cross_dataset_experiments.py \
    --avianz-train "$AVIANZ_TRAIN" \
    --avianz-test  "$AVIANZ_TEST" \
    --doc-train    "$DOC_TRAIN" \
    --doc-test     "$DOC_TEST" \
    --merged-train "$MERGED_TRAIN" \
    --noise-folder "$NOISE_FOLDER" \
    --output       "$RESULTS" \
    --results-dir  "$RESULTS_DIR" \
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
