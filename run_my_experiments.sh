#!/bin/bash
# EXAMPLE - Edit paths and run
# This shows the exact commands for your setup

set -e

# =============================================================================
# EDIT THESE PATHS TO MATCH YOUR SETUP
# =============================================================================

# Where your raw data is located
AVIANZ_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Joe_MoDone?"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"

OUTPUT_BASE="/local/scratch/freangi"

# Set to 1 to skip data loading (if datasets already exist)
SKIP_LOAD=1

# Set to 1 to skip splitting (if splits already exist)
SKIP_SPLIT=0

# =============================================================================
# NO NEED TO EDIT BELOW THIS LINE
# =============================================================================

AVIANZ_FULL="${OUTPUT_BASE}/joe_mo"
AVIANZ_SPLIT_BASE="${OUTPUT_BASE}/joe_mo_split"
AVIANZ_TRAIN="${AVIANZ_SPLIT_BASE}/train"
AVIANZ_TEST="${AVIANZ_SPLIT_BASE}/test"

DOC_FULL="${OUTPUT_BASE}/doc"
DOC_SPLIT_BASE="${OUTPUT_BASE}/doc_split"
DOC_TRAIN="${DOC_SPLIT_BASE}/train"
DOC_TEST="${DOC_SPLIT_BASE}/test"

COMBINED_TRAIN="${OUTPUT_BASE}/combined_train"
RESULTS_DIR="${OUTPUT_BASE}/experiments_$(date +%Y%m%d_%H%M%S)"

SPECIES="nezfan1,silver3,comcha,nezbel1,eurbla,morepo2"
MAX_SAMPLES=100
TEST_SIZE=0.17

echo "=========================================="
echo "AviaNZ Cross-Dataset Training Pipeline"
echo "=========================================="
echo ""
echo "This will:"
if [ $SKIP_LOAD -eq 0 ]; then
    echo "  1. Create datasets (100 samples per species)"
else
    echo "  1. [SKIP] Create datasets"
fi
if [ $SKIP_SPLIT -eq 0 ]; then
    echo "  2. Split into train/test (17% test)"
else
    echo "  2. [SKIP] Split datasets"
fi
echo "  3. Merge training sets"
echo "  4. Run 6 training experiments"
echo "  5. Generate publication plots"
echo ""
echo "Paths:"
echo "  AviaNZ raw:     $AVIANZ_RAW"
echo "  DOC raw:        $DOC_RAW"
echo "  Output:         $OUTPUT_BASE"
echo "  Results:        $RESULTS_DIR"
echo ""
echo "Skip flags:"
echo "  SKIP_LOAD=$SKIP_LOAD  SKIP_SPLIT=$SKIP_SPLIT"
echo "  (Set to 1 at top of script to skip steps)"
echo ""

read -p "Press Enter to start (or Ctrl+C to cancel)..."

if [ $SKIP_LOAD -eq 0 ]; then
    # Step 1: Create datasets
    echo ""
    echo "=========================================="
    echo "Step 1/5: Creating datasets"
    echo "=========================================="

    echo "Creating AviaNZ dataset..."
    python3 data_loader.py avianz "$AVIANZ_RAW" "$AVIANZ_FULL" \
        --species "$SPECIES" \
        --max-samples $MAX_SAMPLES

    echo ""
    echo "Creating DOC dataset..."
    python3 data_loader.py doc "$DOC_RAW" "$DOC_FULL" \
        --species "$SPECIES" \
        --max-samples $MAX_SAMPLES
else
    echo ""
    echo "=========================================="
    echo "Step 1/5: SKIPPED (data already loaded)"
    echo "=========================================="
fi

if [ $SKIP_SPLIT -eq 0 ]; then
    # Step 2: Split datasets
    echo ""
    echo "=========================================="
    echo "Step 2/5: Splitting datasets"
    echo "=========================================="

    echo "Splitting AviaNZ..."
    python3 split_dataset.py "$AVIANZ_FULL" "$AVIANZ_SPLIT_BASE" \
        --test-ratio $TEST_SIZE \
        --overwrite

    echo ""
    echo "Splitting DOC..."
    python3 split_dataset.py "$DOC_FULL" "$DOC_SPLIT_BASE" \
        --test-ratio $TEST_SIZE \
        --overwrite
else
    echo ""
    echo "=========================================="
    echo "Step 2/5: SKIPPED (splits already exist)"
    echo "=========================================="
fi

# Step 3: Merge
echo ""
echo "=========================================="
echo "Step 3/5: Merging training sets"
echo "=========================================="

python3 merge_datasets.py "$AVIANZ_TRAIN" "$DOC_TRAIN" "$COMBINED_TRAIN"

# Step 4: Experiments
echo ""
echo "=========================================="
echo "Step 4/5: Running experiments"
echo "=========================================="
echo "This will take 2-3 hours..."
echo ""

python3 run_cross_dataset_experiments.py \
    --avianz-train "$AVIANZ_TRAIN" \
    --avianz-test "$AVIANZ_TEST" \
    --doc-train "$DOC_TRAIN" \
    --doc-test "$DOC_TEST" \
    --combined-train "$COMBINED_TRAIN" \
    --output "$RESULTS_DIR" \
    --epochs 10 \
    --batch-size 32

# Step 5: Summary
echo ""
echo "=========================================="
echo "Step 5/5: Summary"
echo "=========================================="
echo ""
cat "$RESULTS_DIR/report.txt"

echo ""
echo "=========================================="
echo "✓ COMPLETE!"
echo "=========================================="
echo ""
echo "Results: $RESULTS_DIR"
echo ""
echo "For your PDF, use these files:"
echo "  1. ${RESULTS_DIR}/test_accuracy_comparison.png"
echo "  2. ${RESULTS_DIR}/generalization_gap.png"
echo "  3. ${RESULTS_DIR}/training_curves.png"
echo "  4. ${RESULTS_DIR}/freeze_comparison.png"
echo "  5. ${RESULTS_DIR}/summary_table.csv"
echo ""
