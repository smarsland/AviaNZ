#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

BASE="${AVIA_NZ_BASE:-${AVIA_NZ_OUTPUT_ROOT:-/local/scratch/freangi}}"
KAYTOO_ROOT="${KAYTOO_ROOT:-$REPO_ROOT/../Kaytoo}"
DOC_RAW_DIR_OVERRIDE="${DOC_RAW_DIR:-}"
AVIANZ_RAW_DIR_OVERRIDE="${AVIANZ_RAW_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --doc-raw) DOC_RAW_DIR_OVERRIDE="$2"; shift 2 ;;
    --avianz-raw) AVIANZ_RAW_DIR_OVERRIDE="$2"; shift 2 ;;
    --kaytoo-root) KAYTOO_ROOT="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

MATCHED="${BASE}/matched"
AVIANZ_TEST="${MATCHED}/avianz_split/test"
DOC_TRAIN="${MATCHED}/doc_split/train"
DOC_TEST="${MATCHED}/doc_split/test"

OUT_KAYTOO="${BASE}/matched_tests/kaytoo_pretrained_seed0"
OUT_BIRDNET="${BASE}/matched_tests/birdnet_pretrained_seed0"
OUT_REGNET_DOC="${BASE}/matched_tests/regnet_on_doc_bgsub"
OUT_REGNET_COMBINED="${BASE}/combined_tests/regnet_combined_bgsubtract_seed0"
PRETRAINED_MODEL="${AVIA_NZ_PRETRAINED_PATH:-${BIRDCLEF_PRETRAINED_PATH:-BirdClefModels/model_fold0.pth}}"
COMBINED_DATASET="${BASE}/combined_dataset/combined_large"

mkdir -p "$OUT_KAYTOO" "$OUT_BIRDNET" "$OUT_REGNET_DOC" "$OUT_REGNET_COMBINED"

echo "Using output base: $BASE"
echo "Using matched data: $MATCHED"
echo "Using DOC raw: ${DOC_RAW_DIR_OVERRIDE:-<default>}"
echo "Using AviaNZ raw: ${AVIANZ_RAW_DIR_OVERRIDE:-<default>}"
echo "Using Kaytoo root: $KAYTOO_ROOT"
echo "Using pretrained checkpoint: $PRETRAINED_MODEL"
echo "Using combined dataset: $COMBINED_DATASET"

if [[ ! -f "$MATCHED/avianz_split/test/labels.json" || ! -f "$MATCHED/doc_split/test/labels.json" ]]; then
    echo "Building matched datasets..."
    AVIA_NZ_BASE="$BASE" DOC_RAW_DIR="${DOC_RAW_DIR_OVERRIDE}" AVIANZ_RAW_DIR="${AVIANZ_RAW_DIR_OVERRIDE}" bash build_dataset.sh --overwrite
fi

if [[ ! -f "$BASE/combined_dataset/combined_large/labels.json" ]]; then
    echo "Building combined dataset..."
    AVIA_NZ_BASE="$BASE" DOC_RAW_DIR="${DOC_RAW_DIR_OVERRIDE}" bash build_combined_dataset.sh --overwrite
fi

echo "Running: Kaytoo only"
python3 scripts/evaluate_kaytoo.py "$AVIANZ_TEST" "$DOC_TEST" \
  --kaytoo-root "$KAYTOO_ROOT" \
  --mapping "$SCRIPT_DIR/data/DOC_bird_naming_map.csv" \
  --output "$OUT_KAYTOO"

echo "Running: BirdNET pretrained baseline"
python3 scripts/evaluate_birdnet.py "$AVIANZ_TEST" "$DOC_TEST" \
  --output "$OUT_BIRDNET"

echo "Running: RegNet on DOC"
python3 train.py "$DOC_TRAIN" "$OUT_REGNET_DOC" \
  --model-type regnet \
  --pretrained "$PRETRAINED_MODEL" \
  --spec-transform Log \
  --bg-subtract \
  --kbird-prior 2.0 \
  --epochs 40 --patience 15 --seed 0 \
  --test-folder "$AVIANZ_TEST" \
  --test-folder2 "$DOC_TEST"

echo "Running: RegNet on full combined DOC+AviaNZ"
python3 train.py "$COMBINED_DATASET" "$OUT_REGNET_COMBINED" \
  --model-type regnet \
  --pretrained "$PRETRAINED_MODEL" \
  --spec-transform Log \
  --bg-subtract \
  --kbird-prior 2.0 \
  --epochs 40 --patience 15 --seed 0 \
  --test-folder "$AVIANZ_TEST" \
  --test-folder2 "$DOC_TEST"

echo "Done."
