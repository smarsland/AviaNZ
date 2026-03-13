#!/bin/bash
set -e

# Paths
AVIANZ_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Joe_MoDone?"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"
OUTPUT_BASE="/local/scratch/freangi"

# Skip flags (set to 1 to skip)
SKIP_LOAD=0
SKIP_SPLIT=0

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
        --max-samples $MAX_SAMPLES

    python3 data_loader.py doc "$DOC_RAW" "$DOC_FULL" \
        --species "$SPECIES" \
        --max-samples $MAX_SAMPLES
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

echo "Running standard fine-tuning experiments..."
python3 run_cross_dataset_experiments.py \
    --avianz-train "$AVIANZ_TRAIN" \
    --avianz-test "$AVIANZ_TEST" \
    --doc-train "$DOC_TRAIN" \
    --doc-test "$DOC_TEST" \
    --combined-train "$COMBINED_TRAIN" \
    --output "$RESULTS_DIR" \
    --epochs 100 \
    --batch-size 32

echo ""
echo "Running DANN (Domain Adaptation) experiments..."

# DANN: AviaNZ->DOC (full)
echo "DANN AviaNZ->DOC (full fine-tuning)..."
python3 train_domain_adaptation.py \
    "$AVIANZ_TRAIN" "$DOC_TRAIN" "${RESULTS_DIR}/dann_avianz_doc_full" \
    --architecture regnety_008 \
    --pretrained BirdClefModels/model_fold0.pth \
    --epochs 100 \
    --batch-size 32 \
    --lambda-domain 1.0 \
    --test-folder "$AVIANZ_TEST" \
    --test-folder2 "$DOC_TEST"

# DANN: AviaNZ->DOC (frozen)
echo "DANN AviaNZ->DOC (frozen backbone)..."
python3 train_domain_adaptation.py \
    "$AVIANZ_TRAIN" "$DOC_TRAIN" "${RESULTS_DIR}/dann_avianz_doc_frozen" \
    --architecture regnety_008 \
    --pretrained BirdClefModels/model_fold0.pth \
    --epochs 100 \
    --batch-size 32 \
    --lambda-domain 1.0 \
    --freeze-backbone \
    --test-folder "$AVIANZ_TEST" \
    --test-folder2 "$DOC_TEST"

# DANN: DOC->AviaNZ (full)
echo "DANN DOC->AviaNZ (full fine-tuning)..."
python3 train_domain_adaptation.py \
    "$DOC_TRAIN" "$AVIANZ_TRAIN" "${RESULTS_DIR}/dann_doc_avianz_full" \
    --architecture regnety_008 \
    --pretrained BirdClefModels/model_fold0.pth \
    --epochs 100 \
    --batch-size 32 \
    --lambda-domain 1.0 \
    --test-folder "$DOC_TEST" \
    --test-folder2 "$AVIANZ_TEST"

# DANN: DOC->AviaNZ (frozen)
echo "DANN DOC->AviaNZ (frozen backbone)..."
python3 train_domain_adaptation.py \
    "$DOC_TRAIN" "$AVIANZ_TRAIN" "${RESULTS_DIR}/dann_doc_avianz_frozen" \
    --architecture regnety_008 \
    --pretrained BirdClefModels/model_fold0.pth \
    --epochs 100 \
    --batch-size 32 \
    --lambda-domain 1.0 \
    --freeze-backbone \
    --test-folder "$DOC_TEST" \
    --test-folder2 "$AVIANZ_TEST"

echo "Done. Results: $RESULTS_DIR"
