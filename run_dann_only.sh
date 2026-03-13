#!/bin/bash
set -e

# Run the 2 DANN experiments and add them to existing results

AVIANZ_TEST="/local/scratch/freangi/joe_mo_split/test"
DOC_TEST="/local/scratch/freangi/doc_split/test"
AVIANZ_TRAIN="/local/scratch/freangi/joe_mo_split/train"
DOC_TRAIN="/local/scratch/freangi/doc_split/train"
RESULTS_DIR="/local/scratch/freangi/experiments_20260313_064819"

echo "Running 2 DANN experiments..."
echo "================================"

# DANN combined (full fine-tuning)
echo ""
echo "1/2: DANN combined (full fine-tuning)..."
python3 train_domain_adaptation.py \
    "$AVIANZ_TRAIN" \
    "$DOC_TRAIN" \
    "${RESULTS_DIR}/dann_combined_full" \
    --architecture regnety_008 \
    --pretrained BirdClefModels/model_fold0.pth \
    --epochs 100 \
    --batch-size 32 \
    --lambda-domain 1.0 \
    --test-folder "$AVIANZ_TEST" \
    --test-folder2 "$DOC_TEST"

# DANN combined (frozen backbone)
echo ""
echo "2/2: DANN combined (frozen backbone)..."
python3 train_domain_adaptation.py \
    "$AVIANZ_TRAIN" \
    "$DOC_TRAIN" \
    "${RESULTS_DIR}/dann_combined_frozen" \
    --architecture regnety_008 \
    --pretrained BirdClefModels/model_fold0.pth \
    --epochs 100 \
    --batch-size 32 \
    --lambda-domain 1.0 \
    --freeze-backbone \
    --test-folder "$AVIANZ_TEST" \
    --test-folder2 "$DOC_TEST"

echo ""
echo "================================"
echo "DANN experiments complete!"
echo ""
echo "Now adding DANN results to all_results.json..."
python3 add_dann_to_results.py "$RESULTS_DIR" "$AVIANZ_TEST" "$DOC_TEST"

echo ""
echo "Regenerating all plots..."
python3 regenerate_plots.py "$RESULTS_DIR"

echo ""
echo "================================"
echo "✓ Done! Check ${RESULTS_DIR}/ for updated results and plots."
