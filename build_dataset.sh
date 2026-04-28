#!/bin/bash
set -e

# Build matched DOC + AviaNZ datasets including audio files.
# Audio files are needed for Kaytoo and BirdNET evaluation.
#
# Usage:
#   ./build_dataset.sh
#   ./build_dataset.sh --freq-mask
#   ./build_dataset.sh --overwrite     # re-build even if datasets exist

FREQ_MASK_FLAG=""
OVERWRITE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --freq-mask) FREQ_MASK_FLAG="--freq-mask"; shift ;;
        --overwrite) OVERWRITE=true; shift ;;
        *) echo "Unknown option: $1"; echo "Valid options: --freq-mask, --overwrite"; exit 1 ;;
    esac
done

AVIANZ_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Joe_MoDone?"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"
OUTPUT_BASE="/local/scratch/freangi"

REVIEWED_CSV="data/doc_reviewed.csv"
MAPPING="data/DOC_bird_naming_map.csv"

MATCHED_BASE="${OUTPUT_BASE}/matched"
DOC_MATCHED="${MATCHED_BASE}/doc_matched"
AVIANZ_MATCHED="${MATCHED_BASE}/avianz_matched"

DOC_SPLIT_BASE="${MATCHED_BASE}/doc_split"
AVIANZ_SPLIT_BASE="${MATCHED_BASE}/avianz_split"

MERGED_TRAIN="${MATCHED_BASE}/merged_train"

TEST_SIZE=0.25

echo "============================================================"
echo " Build matched datasets"
echo "============================================================"
echo "  DOC raw    : $DOC_RAW"
echo "  AviaNZ raw : $AVIANZ_RAW"
echo "  Output     : $MATCHED_BASE"
echo "  With audio : yes"
echo "  Freq mask  : ${FREQ_MASK_FLAG:-no}"
echo "  Overwrite  : $OVERWRITE"
echo "============================================================"
echo ""

# Step 1: build matched datasets
if [ "$OVERWRITE" = true ] || [ ! -d "$DOC_MATCHED" ] || [ ! -f "$DOC_MATCHED/labels.json" ] || [ ! -d "$AVIANZ_MATCHED" ] || [ ! -f "$AVIANZ_MATCHED/labels.json" ]; then
    echo "=== Step 1: building matched datasets ==="
    PYTHONPATH="$PWD" python3 src/experiments/build_matched_datasets.py \
        --reviewed-csv "$REVIEWED_CSV" \
        --doc-raw      "$DOC_RAW" \
        --avianz-raw   "$AVIANZ_RAW" \
        --output       "$MATCHED_BASE" \
        --mapping      "$MAPPING" \
        --fixed-length \
        --with-audio \
        ${FREQ_MASK_FLAG:+$FREQ_MASK_FLAG}
else
    echo "=== Step 1: matched datasets already exist, skipping (use --overwrite to force) ==="
fi

# Step 2: split into train/test
SPLIT_MISSING=false
for d in "$DOC_SPLIT_BASE/train" "$DOC_SPLIT_BASE/test" "$AVIANZ_SPLIT_BASE/train" "$AVIANZ_SPLIT_BASE/test"; do
    [ ! -d "$d" ] || [ ! -f "$d/labels.json" ] && SPLIT_MISSING=true && break
done

if [ "$OVERWRITE" = true ] || [ "$SPLIT_MISSING" = true ]; then
    echo ""
    echo "=== Step 2: splitting datasets ==="
    PYTHONPATH="$PWD" python3 src/experiments/split_matched_datasets.py \
        "$AVIANZ_MATCHED" \
        "$DOC_MATCHED" \
        "$MATCHED_BASE" \
        --test-ratio $TEST_SIZE \
        --seed 42 \
        --overwrite
    echo ""
    echo "=== Validating splits ==="
    PYTHONPATH="$PWD" python3 src/experiments/validate_splits.py \
        "$AVIANZ_SPLIT_BASE/train" "$AVIANZ_SPLIT_BASE/test" \
        "$DOC_SPLIT_BASE/train"   "$DOC_SPLIT_BASE/test"
else
    echo ""
    echo "=== Step 2: splits already exist, skipping ==="
fi

# Step 3: merge train sets (spectrograms only; audio not needed for training)
if [ "$OVERWRITE" = true ] || [ ! -d "$MERGED_TRAIN" ] || [ ! -f "$MERGED_TRAIN/labels.json" ]; then
    echo ""
    echo "=== Step 3: merging training datasets ==="
    PYTHONPATH="$PWD" python3 src/experiments/merge_datasets.py \
        "$DOC_SPLIT_BASE/train" \
        "$AVIANZ_SPLIT_BASE/train" \
        "$MERGED_TRAIN" \
        --symlink \
        --no-audio
else
    echo ""
    echo "=== Step 3: merged training dataset already exists, skipping ==="
fi

echo ""
echo "============================================================"
echo " Done."
echo "  DOC      train/test : $DOC_SPLIT_BASE"
echo "  AviaNZ   train/test : $AVIANZ_SPLIT_BASE"
echo "  Merged   train      : $MERGED_TRAIN"
echo "============================================================"
