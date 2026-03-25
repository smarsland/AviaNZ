#!/bin/bash
set -e

# ============================================================
# End-to-end matched dataset experiments
# 1) Build matched DOC + AviaNZ datasets from raw audio
# 2) Split each into train / test
# 3) Train and evaluate (confusion matrices generated automatically)
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

RESULTS_DIR="${OUTPUT_BASE}/experiments_matched"
NOISE_FOLDER="${OUTPUT_BASE}/noise"

# Training config
EPOCHS=50
BATCH_SIZE=16
USE_NOISE_AUG=0
NOISE_RATIO=0.0
NOISE_MODE="both"
BACKGROUND_PROB=0.0
USE_MIXUP=1
MIXUP_ALPHA=0.25

TEST_SIZE=0.25

# Skip flags (set to 1 to skip a step)
SKIP_BUILD=0
SKIP_SPLIT=0
SKIP_EXPERIMENTS=0

echo "============================================================"
echo " Matched dataset experiments"
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
        --group-key  source_file \
        --overwrite

    python3 split_dataset.py "$AVIANZ_MATCHED" "$AVIANZ_SPLIT_BASE" \
        --test-ratio $TEST_SIZE \
        --group-key  source_file \
        --overwrite
else
    echo "=== Step 2: SKIPPED (SKIP_SPLIT=1) ==="
fi

# ---- Step 3: Run experiments (trains, evaluates, plots) -----
if [ $SKIP_EXPERIMENTS -eq 0 ]; then
    echo ""
    echo "=== Step 3: Running experiments ==="

    NOISE_ARGS=""
    if [ $USE_NOISE_AUG -eq 1 ] && [ -d "$NOISE_FOLDER" ]; then
        NOISE_ARGS="--noise $NOISE_RATIO --noise-folder $NOISE_FOLDER --noise-mode $NOISE_MODE --background-prob $BACKGROUND_PROB"
    fi

    MIXUP_ARGS=""
    if [ $USE_MIXUP -eq 1 ]; then
        MIXUP_ARGS="--mixup $MIXUP_ALPHA"
    fi

    python3 run_cross_dataset_experiments.py \
        --avianz-train "$AVIANZ_TRAIN" \
        --avianz-test  "$AVIANZ_TEST" \
        --doc-train    "$DOC_TRAIN" \
        --doc-test     "$DOC_TEST" \
        --output       "$RESULTS_DIR" \
        --epochs       $EPOCHS \
        --batch-size   $BATCH_SIZE \
        $NOISE_ARGS \
        $MIXUP_ARGS
else
    echo "=== Step 3: SKIPPED (SKIP_EXPERIMENTS=1) ==="
fi

echo ""
echo "============================================================"
echo " Done. Results: $RESULTS_DIR"
echo "============================================================"
