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
EPOCHS=10
BATCH_SIZE=16
USE_NOISE_AUG=0
NOISE_RATIO=0.0
NOISE_MODE="both"
BACKGROUND_PROB=0.0
USE_MIXUP=1
MIXUP_ALPHA=0.25

TEST_SIZE=0.25

# Skip flags (set to 1 to skip a step)
SKIP_BUILD=1
SKIP_SPLIT=1
SKIP_BASELINE=1
SKIP_NORMALIZATION_TESTS=0

# Force re-run experiments even if results exist (set to 1 to force)
FORCE_RERUN=0

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

# ---- Step 3: Baseline experiments (already done) ------------
if [ $SKIP_BASELINE -eq 0 ]; then
    echo ""
    echo "=== Step 3: Baseline experiments (Log transform only) ==="
    
    RESULTS_DIR="${OUTPUT_BASE}/experiments_matched"
    
    NOISE_ARGS=""
    if [ $USE_NOISE_AUG -eq 1 ] && [ -d "$NOISE_FOLDER" ]; then
        NOISE_ARGS="--noise $NOISE_RATIO --noise-folder $NOISE_FOLDER --noise-mode $NOISE_MODE --background-prob $BACKGROUND_PROB"
    fi

    MIXUP_ARGS=""
    if [ $USE_MIXUP -eq 1 ]; then
        MIXUP_ARGS="--mixup $MIXUP_ALPHA"
    fi

    FORCE_ARGS=""
    if [ $FORCE_RERUN -eq 1 ]; then
        FORCE_ARGS="--force"
    fi

    python3 run_cross_dataset_experiments.py \
        --avianz-train "$AVIANZ_TRAIN" \
        --avianz-test  "$AVIANZ_TEST" \
        --doc-train    "$DOC_TRAIN" \
        --doc-test     "$DOC_TEST" \
        --output       "$RESULTS_DIR" \
        --epochs       $EPOCHS \
        --batch-size   $BATCH_SIZE \
        --spec-transform Log \
        $NOISE_ARGS \
        $MIXUP_ARGS \
        $FORCE_ARGS
else
    echo "=== Step 3: SKIPPED (already ran baseline) ==="
fi

# ---- Step 4: Test normalization strategies ------------------
if [ $SKIP_NORMALIZATION_TESTS -eq 0 ]; then
    echo ""
    echo "=== Step 4: Testing normalization strategies ==="
    echo "  1. Log (baseline - already done)"
    echo "  2. Log + --normalize (log + background subtraction)"
    echo "  3. PCEN (alternative to log, already does normalization)"
    echo ""
    
    RESULTS_DIR="${OUTPUT_BASE}/experiments_matched"
    
    MIXUP_ARGS=""
    if [ $USE_MIXUP -eq 1 ]; then
        MIXUP_ARGS="--mixup $MIXUP_ALPHA"
    fi
    
    FORCE_ARGS=""
    if [ $FORCE_RERUN -eq 1 ]; then
        FORCE_ARGS="--force"
    fi
    
    # Test 2: Log + --normalize
    echo "--- Running: Log + --normalize ---"
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
        $MIXUP_ARGS \
        $FORCE_ARGS
    
    # Test 3: PCEN (no normalize - PCEN already does its own normalization)
    echo ""
    echo "--- Running: PCEN (no --normalize, PCEN does its own thing) ---"
    python3 run_cross_dataset_experiments.py \
        --avianz-train "$AVIANZ_TRAIN" \
        --avianz-test  "$AVIANZ_TEST" \
        --doc-train    "$DOC_TRAIN" \
        --doc-test     "$DOC_TEST" \
        --output       "$RESULTS_DIR" \
        --epochs       $EPOCHS \
        --batch-size   $BATCH_SIZE \
        --spec-transform PCEN \
        $MIXUP_ARGS \
        $FORCE_ARGS
else
    echo "=== Step 4: SKIPPED (SKIP_NORMALIZATION_TESTS=1) ==="
fi

echo ""
echo "============================================================"
echo " Done. All results in: $RESULTS_DIR"
echo "   - joe_mo_baseline_birdclef (Log baseline - already done)"
echo "   - joe_mo_baseline_birdclef_normalized (Log + normalize)"
echo "   - joe_mo_baseline_birdclef_pcen (PCEN - alternative to log)"
echo "   - doc_baseline_birdclef (Log baseline - already done)"
echo "   - doc_baseline_birdclef_normalized (Log + normalize)"
echo "   - doc_baseline_birdclef_pcen (PCEN - alternative to log)"
echo "============================================================"
