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
# No flags. No skip logic. Just checks file existence.
# Want to rerun? Delete the folder.
#
# DEFAULTS (no arguments needed):
#   - Fixed-length spectrograms (1024 bins)
#   - Auto-detect GPUs and run in parallel
#   - Save results to ~/results
#
# Usage:
#   ./run_matched_experiments.sh                         # Use all defaults
#   ./run_matched_experiments.sh --no-fixed-length      # Variable-length mode
#   ./run_matched_experiments.sh --parallel 1           # Force single-threaded
#   ./run_matched_experiments.sh --results-dir ./output # Custom results location
#
# Multi-machine setup:
#   Just run on each machine - results automatically saved to ~/results (shared)
#   Large model files (.pt) stay in /local/scratch (machine-specific)
#   Small result files (JSON, CSV) copied to ~/results (shared across machines)
# ============================================================

# Default settings
FIXED_LENGTH_FLAG="--fixed-length"
PARALLEL_FLAG="--parallel 0"  # Auto-detect GPUs
RESULTS_DIR="$HOME/results"

# Parse arguments (to override defaults if needed)
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-fixed-length)
            FIXED_LENGTH_FLAG=""
            echo "Variable-length mode enabled"
            shift
            ;;
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
        --results-dir)
            RESULTS_DIR="$2"
            echo "Shared results directory: $RESULTS_DIR"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo " Configuration"
echo "============================================================"
echo "  Fixed-length mode: $([ -n "$FIXED_LENGTH_FLAG" ] && echo "YES" || echo "NO")"
echo "  Parallel mode    : ${PARALLEL_FLAG#--parallel }"
echo "  Results directory: $RESULTS_DIR"
echo "============================================================"
echo ""

# Paths
AVIANZ_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Joe_MoDone?"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"
OUTPUT_BASE="/local/scratch/freangi"

REVIEWED_CSV="doc_reviewed.csv"
MAPPING="DOC_bird_naming_map.csv"

MATCHED_BASE="${OUTPUT_BASE}/matched"
DOC_MATCHED="${MATCHED_BASE}/doc_matched"
AVIANZ_MATCHED="${MATCHED_BASE}/avianz_matched"  # Waitākere dataset

DOC_SPLIT_BASE="${MATCHED_BASE}/doc_split" F
fi

# Merge train datasets (DOC + Waitākere) for combined training experiments
if [ ! -d "$MERGED_TRAIN" ]; then
    echo ""
    echo "=== Merging training datasets (DOC + Waitākere) ==="
    python3 src/experiments/merge_datasets.py \
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
        python3 src/data/dataset_builder.py noise "$NOISE_RAW" "$NOISE_FOLDER" --samples $NUM_NOISE_SAMPLES
        
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

python3 src/experiments/run_cross_dataset_experiments.py \
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
