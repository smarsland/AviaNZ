#!/bin/bash
set -e

# ============================================================
# Domain shift experiments - test different normalizations
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

# Training config
EPOCHS=50
BATCH_SIZE=16
USE_MIXUP=1
MIXUP_ALPHA=0.25
TEST_SIZE=0.25

# Skip flags (set to 1 to skip a step)
SKIP_BUILD=1
SKIP_SPLIT=1
SKIP_NORMALIZATION_EXPERIMENTS=1  # Compare different spectrogram normalizations (no noise)
SKIP_NOISE_EXPERIMENTS=0           # Test effect of noise variety on robustness

# Force re-run experiments even if results exist (set to 1 to force)
FORCE_RERUN=0

# EXPERIMENT 1: Normalization strategies to test (no noise augmentation)
# Options: Log, PCEN, Box-Cox, None
NORMALIZATION_METHODS=("Log" "Log+normalize" "PCEN" "Box-Cox")

# EXPERIMENT 2: Noise augmentation config for testing variety hypothesis
NOISE_FOLDER="${OUTPUT_BASE}/noise"
NOISE_RATIO=0.5  # Fixed 50% noise mixing - we test variety, not amount
NOISE_LEVELS=(10 20 50 100 200 500 1000)  # Number of noise files to sample from

# Logging setup - capture all output to log file
LOG_FILE="${OUTPUT_BASE}/experiments_log.txt"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"

echo "============================================================"
echo " Domain Shift Experiments"
echo "  Raw AviaNZ : $AVIANZ_RAW"
echo "  Raw DOC    : $DOC_RAW"
echo "  Output     : $OUTPUT_BASE"
echo "============================================================"
echo ""

# ---- Step 1: Build matched datasets -------------------------
if [ $SKIP_BUILD -eq 0 ]; then
    echo "=== Step 1: Building matched datasets ==="
    python3 build_matched_datasets.py \
        --reviewed-csv "$REVIEWED_CSV" \
        --doc-raw      "$DOC_RAW" \
        --avianz-raw   "$AVIANZ_RAW" \
        --output       "$MATCHED_BASE" \
        --mapping      "$MAPPING"
else
    echo "=== Step 1: SKIPPED (SKIP_BUILD=1) ==="
fi

# ---- Step 2: Split each matched dataset into train / test ---
if [ $SKIP_SPLIT -eq 0 ]; then
    echo ""
    echo "=== Step 2: Splitting datasets ==="
    python3 split_dataset.py "$DOC_MATCHED"   "$DOC_SPLIT_BASE" \
        --test-ratio $TEST_SIZE \
        --overwrite

    python3 split_dataset.py "$AVIANZ_MATCHED" "$AVIANZ_SPLIT_BASE" \
        --test-ratio $TEST_SIZE \
        --overwrite
    
    echo ""
    echo "=== Validating split consistency ==="
    python3 validate_splits.py "$AVIANZ_TRAIN" "$AVIANZ_TEST" "$DOC_TRAIN" "$DOC_TEST"
else
    echo "=== Step 2: SKIPPED (SKIP_SPLIT=1) ==="
fi

# =============================================================================
# EXPERIMENT 1: NORMALIZATION COMPARISON
# Test different spectrogram normalization strategies for domain robustness
# =============================================================================

if [ $SKIP_NORMALIZATION_EXPERIMENTS -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo " EXPERIMENT 1: NORMALIZATION COMPARISON"
    echo "========================================================================"
    echo " Testing strategies: ${NORMALIZATION_METHODS[@]}"
    echo " Goal: Find which normalization reduces domain shift most"
    echo "========================================================================"
    
    RESULTS_DIR="${OUTPUT_BASE}/experiments_matched"
    
    MIXUP_ARGS=""
    if [ $USE_MIXUP -eq 1 ]; then
        MIXUP_ARGS="--mixup $MIXUP_ALPHA"
    fi
    
    FORCE_ARGS=""
    if [ $FORCE_RERUN -eq 1 ]; then
        FORCE_ARGS="--force"
    fi
    
    for METHOD in "${NORMALIZATION_METHODS[@]}"; do
        echo ""
        echo "--- Testing: $METHOD ---"
        
        # Parse method into spec-transform and normalize flag
        SPEC_TRANSFORM="Log"
        NORMALIZE_FLAG=""
        
        if [[ "$METHOD" == "Log+normalize" ]]; then
            SPEC_TRANSFORM="Log"
            NORMALIZE_FLAG="--normalize"
        elif [[ "$METHOD" == "PCEN" ]]; then
            SPEC_TRANSFORM="PCEN"
            NORMALIZE_FLAG=""
        elif [[ "$METHOD" == "Box-Cox" ]]; then
            SPEC_TRANSFORM="Box-Cox"
            NORMALIZE_FLAG=""
        elif [[ "$METHOD" == "None" ]]; then
            SPEC_TRANSFORM="None"
            NORMALIZE_FLAG=""
        fi
        
        python3 run_cross_dataset_experiments.py \
            --avianz-train "$AVIANZ_TRAIN" \
            --avianz-test  "$AVIANZ_TEST" \
            --doc-train    "$DOC_TRAIN" \
            --doc-test     "$DOC_TEST" \
            --output       "$RESULTS_DIR" \
            --epochs       $EPOCHS \
            --batch-size   $BATCH_SIZE \
            --spec-transform "$SPEC_TRANSFORM" \
            $NORMALIZE_FLAG \
            $MIXUP_ARGS \
            $FORCE_ARGS
        
        echo "✓ Completed: $METHOD"
    done
    
    echo ""
    echo "✓ NORMALIZATION EXPERIMENTS COMPLETE"
else
    echo "=== EXPERIMENT 1: SKIPPED (SKIP_NORMALIZATION_EXPERIMENTS=1) ==="
fi

# =============================================================================
# EXPERIMENT 2: NOISE AUGMENTATION
# Test hypothesis: More noise variety improves cross-domain robustness
# Uses best normalization from Experiment 1 (Log+normalize)
# =============================================================================

if [ $SKIP_NOISE_EXPERIMENTS -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo " EXPERIMENT 2: NOISE AUGMENTATION"
    echo "========================================================================"
    echo " Hypothesis: More noise VARIETY → better domain robustness"
    echo " Fixed noise ratio: $NOISE_RATIO (50% mixing)"
    echo " Testing: ${NOISE_LEVELS[@]} + all available noise files"
    echo " Using: Log + normalize (best from Experiment 1)"
    echo "========================================================================"
    
    if [ ! -d "$NOISE_FOLDER" ]; then
        echo "ERROR: Noise folder not found: $NOISE_FOLDER"
        echo "SKIPPING noise experiments"
    else
        TOTAL_NOISE=$(find "$NOISE_FOLDER/data" -name "*.npy" 2>/dev/null | wc -l)
        echo "Available noise files: $TOTAL_NOISE"
        
        if [ $TOTAL_NOISE -eq 0 ]; then
            echo "ERROR: No noise files in $NOISE_FOLDER/data/"
            echo "SKIPPING noise experiments"
        else
            # Add "all" to test levels
            ALL_LEVELS=("${NOISE_LEVELS[@]}" "$TOTAL_NOISE")
            
            RESULTS_DIR="${OUTPUT_BASE}/experiments_noise"
            
            MIXUP_ARGS=""
            if [ $USE_MIXUP -eq 1 ]; then
                MIXUP_ARGS="--mixup $MIXUP_ALPHA"
            fi
            
            for N_NOISE in "${ALL_LEVELS[@]}"; do
                echo ""
                echo "--- Testing: $N_NOISE noise files ---"
                
                if [ $N_NOISE -eq 0 ]; then
                    # No noise (baseline for comparison)
                    NOISE_ARGS=""
                elif [ $N_NOISE -eq $TOTAL_NOISE ]; then
                    # Use all available noise
                    NOISE_ARGS="--noise $NOISE_RATIO --noise-folder $NOISE_FOLDER --noise-mode both"
                else
                    # Create subset of N noise files
                    NOISE_SUBSET="${OUTPUT_BASE}/matched/noise_subset_${N_NOISE}"
                    
                    if [ ! -d "$NOISE_SUBSET" ] || [ $FORCE_RERUN -eq 1 ]; then
                        echo "  Creating subset: $N_NOISE files"
                        rm -rf "$NOISE_SUBSET"
                        mkdir -p "$NOISE_SUBSET/data"
                        find "$NOISE_FOLDER/data" -name "*.npy" | shuf -n $N_NOISE | while read f; do
                            cp "$f" "$NOISE_SUBSET/data/"
                        done
                        if [ -f "$NOISE_FOLDER/labels.json" ]; then
                            cp "$NOISE_FOLDER/labels.json" "$NOISE_SUBSET/"
                        fi
                    fi
                    
                    NOISE_ARGS="--noise $NOISE_RATIO --noise-folder $NOISE_SUBSET --noise-mode both"
                fi
                
                # Run experiment with this noise level
                # Add suffix to differentiate experiments by noise count
                python3 run_cross_dataset_experiments.py \
                    --avianz-train "$AVIANZ_TRAIN" \
                    --avianz-test  "$AVIANZ_TEST" \
                    --doc-train    "$DOC_TRAIN" \
                    --doc-test     "$DOC_TEST" \
                    --output       "$RESULTS_DIR" \
                    --epochs       $EPOCHS \
                    --batch-size   $BATCH_SIZE \
                    --spec-transform Log \
                    --normalize \
                    --experiment-suffix "_noise${N_NOISE}" \
                    $MIXUP_ARGS \
                    $NOISE_ARGS
                
                echo "✓ Completed: $N_NOISE noise files"
            done
            
            echo ""
            echo "✓ NOISE EXPERIMENTS COMPLETE"
        fi
    fi
else
    echo "=== EXPERIMENT 2: SKIPPED (SKIP_NOISE_EXPERIMENTS=1) ==="
fi

echo ""
echo "============================================================"
echo " ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo " Results:"
echo "   Normalization: ${OUTPUT_BASE}/experiments_matched"
echo "   Noise:         ${OUTPUT_BASE}/experiments_noise"
echo "============================================================"
