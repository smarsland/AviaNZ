#!/bin/bash
# EXAMPLE - Edit paths and run
# This shows the exact commands for your setup

set -e

# =============================================================================
# EDIT THESE PATHS TO MATCH YOUR SETUP
# =============================================================================

# Where your raw data is located
AVIANZ_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Joe_MoDone\?"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"

OUTPUT_BASE="/local/scratch/freangi"

# =============================================================================
# NO NEED TO EDIT BELOW THIS LINE
# =============================================================================

AVIANZ_FULL="${OUTPUT_BASE}/joe_mo"
AVIANZ_TRAIN="${OUTPUT_BASE}/joe_mo_train"
AVIANZ_TEST="${OUTPUT_BASE}/joe_mo_test"

DOC_FULL="${OUTPUT_BASE}/doc"
DOC_TRAIN="${OUTPUT_BASE}/doc_train"
DOC_TEST="${OUTPUT_BASE}/doc_test"

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
echo "  1. Create datasets (100 samples per species)"
echo "  2. Split into train/test (17% test)"
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

read -p "Press Enter to start (or Ctrl+C to cancel)..."

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

# Step 2: Split datasets
echo ""
echo "=========================================="
echo "Step 2/5: Splitting datasets"
echo "=========================================="

echo "Splitting AviaNZ..."
python3 split_dataset.py "$AVIANZ_FULL" \
    --output-train "$AVIANZ_TRAIN" \
    --output-test "$AVIANZ_TEST" \
    --test-size $TEST_SIZE

echo ""
echo "Splitting DOC..."
python3 split_dataset.py "$DOC_FULL" \
    --output-train "$DOC_TRAIN" \
    --output-test "$DOC_TEST" \
    --test-size $TEST_SIZE

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
