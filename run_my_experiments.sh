#!/bin/bash
set -e

# Paths
AVIANZ_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Joe_MoDone?"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"
OUTPUT_BASE="/local/scratch/freangi"

# Noise folder for DANN (set this to your noise data path)
NOISE_FOLDER="${OUTPUT_BASE}/noise"

# Skip flags (set to 1 to skip)
SKIP_LOAD=1
SKIP_SPLIT=1
SKIP_BIRDNET=1

# Config

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
MAX_SAMPLES=120
TEST_SIZE=0.17

echo "Results will be saved to: $RESULTS_DIR"

if [ $SKIP_LOAD -eq 0 ]; then
    echo "Creating datasets..."
    python3 data_loader.py avianz "$AVIANZ_RAW" "$AVIANZ_FULL" \
        --species "$SPECIES" \
        --max-samples $MAX_SAMPLES \
        --ignore-multilabel \
        --with-audio \
        --overwrite

    python3 data_loader.py doc "$DOC_RAW" "$DOC_FULL" \
        --species "$SPECIES" \
        --max-samples $MAX_SAMPLES \
        --ignore-multilabel \
        --with-audio \
        --overwrite
fi

if [ $SKIP_SPLIT -eq 0 ]; then
    echo "Splitting datasets..."
    python3 split_dataset.py "$AVIANZ_FULL" "$AVIANZ_SPLIT_BASE" \
        --test-ratio $TEST_SIZE \
        --overwrite

    python3 split_dataset.py "$DOC_FULL" "$DOC_SPLIT_BASE" \
        --test-ratio $TEST_SIZE \
        --overwrite
fi

echo "Merging training sets..."
python3 merge_datasets.py "$AVIANZ_TRAIN" "$DOC_TRAIN" "$COMBINED_TRAIN"

echo "Running all experiments (12 tests: 2 model types × 6 configs)..."
python3 run_cross_dataset_experiments.py \
    --avianz-train "$AVIANZ_TRAIN" \
    --avianz-test "$AVIANZ_TEST" \
    --doc-train "$DOC_TRAIN" \
    --doc-test "$DOC_TEST" \
    --combined-train "$COMBINED_TRAIN" \
    --output "$RESULTS_DIR" \
    --epochs 50 \
    --batch-size 32

if [ $SKIP_BIRDNET -eq 0 ]; then
    echo ""
    echo "Running BirdNET evaluation on test sets..."
    BIRDNET_OUTPUT="${RESULTS_DIR}/birdnet_evaluation"
    python3 evaluate_birdnet.py \
        "$AVIANZ_TEST" "$DOC_TEST" \
        --output "$BIRDNET_OUTPUT" \
        --min-confidence 0.1
    echo "BirdNET results: $BIRDNET_OUTPUT"
fi

echo "Done. Results: $RESULTS_DIR"
