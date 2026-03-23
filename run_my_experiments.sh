#!/bin/bash
set -e

# Paths
AVIANZ_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Joe_MoDone?"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"
OUTPUT_BASE="/local/scratch/freangi"

# Noise folder for DANN (set this to your noise data path)
NOISE_FOLDER="${OUTPUT_BASE}/noise"

# Optional robustness training (set to 1 to enable)
USE_NOISE_AUG=1
NOISE_RATIO=0.25
NOISE_MODE="both"   # BirdClef only: full|background|both
BACKGROUND_PROB=0.05  # BirdClef only: replace samples with background and zero labels
USE_MIXUP=0
MIXUP_ALPHA=0.25
NOISE_AS_SAMPLES=0   # AST only: adds noise spectrograms as all-zero-label samples

# Skip flags (set to 1 to skip)
SKIP_LOAD=0
SKIP_SPLIT=0
SKIP_BIRDNET=1

# Freeze early layers? (set to 1 to freeze)
# BirdClef: freezes first 4 stages (stem,s1,s2,s3), trains only s4+classifier (~53% params)
# AST: freezes first 8 transformer layers, trains last 4 + classifier
FREEZE_BACKBONE=1

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
RESULTS_DIR="${OUTPUT_BASE}/experiments"

SPECIES="nezfan1,silver3,comcha,nezbel1,eurbla,morepo2"
MAX_SAMPLES=120
TEST_SIZE=0.17

echo "Results dir: $RESULTS_DIR"
echo "Completed experiments will be skipped automatically"
echo ""

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
        --group-key source_file \
        --overwrite

    python3 split_dataset.py "$DOC_FULL" "$DOC_SPLIT_BASE" \
        --test-ratio $TEST_SIZE \
        --group-key source_file \
        --overwrite
fi

echo "Merging training sets..."
python3 merge_datasets.py "$AVIANZ_TRAIN" "$DOC_TRAIN" "$COMBINED_TRAIN"

echo "Running all experiments (8 tests: 2 model types × 4 configs)..."
FREEZE_FLAG=""
if [ $FREEZE_BACKBONE -eq 1 ]; then
    FREEZE_FLAG="--freeze-backbone"
    echo "  Freeze strategy: Freeze early layers to reduce overfitting"
    echo "    - BirdClef: Freeze stem, s1, s2, s3 (train only s4 + classifier, ~53% params)"
    echo "    - AST: Freeze first 8 layers (train last 4 + classifier)"
else
    echo "  Freeze strategy: None (full fine-tuning)"
fi

echo ""
echo "Experiment types (per dataset pair):"
echo "  1. Baseline (no tricks)"
echo "  2. Normalize (background normalization)"
echo ""

python3 run_cross_dataset_experiments.py \
    --avianz-train "$AVIANZ_TRAIN" \
    --avianz-test "$AVIANZ_TEST" \
    --doc-train "$DOC_TRAIN" \
    --doc-test "$DOC_TEST" \
    --combined-train "$COMBINED_TRAIN" \
    --output "$RESULTS_DIR" \
    --epochs 50 \
    --batch-size 16 \
    $FREEZE_FLAG \
    $( [ $USE_NOISE_AUG -eq 1 ] && echo "--noise $NOISE_RATIO --noise-folder $NOISE_FOLDER --noise-mode $NOISE_MODE --background-prob $BACKGROUND_PROB" ) \
    $( [ $USE_MIXUP -eq 1 ] && echo "--mixup $MIXUP_ALPHA" ) \
    $( [ $NOISE_AS_SAMPLES -eq 1 ] && echo "--noise-as-samples" )

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
