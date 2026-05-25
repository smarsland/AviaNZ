#!/bin/bash
set -e

# Build large (unmatched) DOC + AviaNZ datasets including audio files.
#
# Unlike build_dataset.sh this script:
#   - Takes up to 1000 samples per species from each dataset independently
#   - Does NOT do record-for-record matching or use the corrections CSV
#   - Uses the best-performing model's spectrogram settings:
#       Standard spectrogram, Hamming window, Mel frequency scale
#   - Includes audio so Kaytoo and BirdNET can be evaluated
#
# Usage:
#   ./build_large_dataset.sh
#   ./build_large_dataset.sh --overwrite
#   ./build_large_dataset.sh --max-per-species 500
#   ./build_large_dataset.sh --no-audio
#   ./build_large_dataset.sh --spec-type Standard --window-type Hann --sg-scale 'Mel Frequency'

OVERWRITE_FLAG=""
MAX_PER_SPECIES=300
NO_AUDIO_FLAG=""
SPEC_TYPE="Standard"
WINDOW_TYPE="Hamming"
SG_SCALE="Mel Frequency"
MIN_PER_CLASS=50
TEST_RATIO=0.25

while [[ $# -gt 0 ]]; do
    case $1 in
        --overwrite)            OVERWRITE_FLAG="--overwrite"; shift ;;
        --max-per-species)      MAX_PER_SPECIES="$2"; shift 2 ;;
        --min-per-class)        MIN_PER_CLASS="$2"; shift 2 ;;
        --test-ratio)           TEST_RATIO="$2"; shift 2 ;;
        --no-audio)             NO_AUDIO_FLAG="--no-audio"; shift ;;
        --spec-type)            SPEC_TYPE="$2"; shift 2 ;;
        --window-type)          WINDOW_TYPE="$2"; shift 2 ;;
        --sg-scale)             SG_SCALE="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Valid options: --overwrite, --max-per-species N, --min-per-class N,"
            echo "               --test-ratio F, --no-audio,"
            echo "               --spec-type {Standard,Multi-tapered,Reassigned,Bandpass},"
            echo "               --window-type {Hann,Hamming,Blackman,BlackmanHarris},"
            echo "               --sg-scale {Linear,'Mel Frequency','Bark Frequency'}"
            exit 1 ;;
    esac
done

AVIANZ_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Joe_MoDone?"
DOC_RAW="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/NZBirds"
OUTPUT_BASE="/local/scratch/freangi/large"
MAPPING="data/DOC_bird_naming_map.csv"

echo "============================================================"
echo " Build large unmatched datasets"
echo "============================================================"
echo "  DOC raw        : $DOC_RAW"
echo "  AviaNZ raw     : $AVIANZ_RAW"
echo "  Output         : $OUTPUT_BASE"
echo "  Max/species    : $MAX_PER_SPECIES"
echo "  Min/class      : $MIN_PER_CLASS"
echo "  Test ratio     : $TEST_RATIO"
echo "  With audio     : ${NO_AUDIO_FLAG:-yes}"
echo "  Spec type      : $SPEC_TYPE"
echo "  Window         : $WINDOW_TYPE"
echo "  Scale          : $SG_SCALE"
echo "  Overwrite      : ${OVERWRITE_FLAG:-no}"
echo "============================================================"
echo ""

PYTHONPATH="$PWD" python3 src/experiments/build_large_datasets.py \
    --doc-raw       "$DOC_RAW" \
    --avianz-raw    "$AVIANZ_RAW" \
    --output        "$OUTPUT_BASE" \
    --mapping       "$MAPPING" \
    --max-per-species "$MAX_PER_SPECIES" \
    --min-per-class   "$MIN_PER_CLASS" \
    --test-ratio      "$TEST_RATIO" \
    --spec-type     "$SPEC_TYPE" \
    --window-type   "$WINDOW_TYPE" \
    --sg-scale      "$SG_SCALE" \
    ${OVERWRITE_FLAG:+$OVERWRITE_FLAG} \
    ${NO_AUDIO_FLAG:+$NO_AUDIO_FLAG}

echo ""
echo "============================================================"
echo " Done."
echo "  Splits ready for training:"
echo "    DOC    train : $OUTPUT_BASE/doc_split/train"
echo "    DOC    test  : $OUTPUT_BASE/doc_split/test"
echo "    AviaNZ train : $OUTPUT_BASE/avianz_split/train"
echo "    AviaNZ test  : $OUTPUT_BASE/avianz_split/test"
echo ""
echo "  Evaluate with Kaytoo / BirdNET:"
echo "    ./run_kaytoo_eval.sh   (point at $OUTPUT_BASE/avianz_split/test  and  $OUTPUT_BASE/doc_split/test)"
echo "============================================================"
