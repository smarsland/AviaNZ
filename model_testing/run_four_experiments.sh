#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

BASE="/local/scratch/freangi"
MATCHED="${BASE}/matched"
AVIANZ_TEST="${MATCHED}/avianz_split/test"
DOC_TRAIN="${MATCHED}/doc_split/train"
DOC_TEST="${MATCHED}/doc_split/test"

OUT_KAYTOO="${BASE}/matched_tests/kaytoo_pretrained_seed0"
OUT_BIRDNET="${BASE}/matched_tests/birdnet_pretrained_seed0"
OUT_REGNET_DOC="${BASE}/matched_tests/regnet_on_doc_bgsub"
OUT_REGNET_COMBINED="${BASE}/combined_tests/regnet_combined_bgsubtract_seed0"

mkdir -p "$OUT_KAYTOO" "$OUT_BIRDNET" "$OUT_REGNET_DOC" "$OUT_REGNET_COMBINED"

echo "Running: Kaytoo only"
python3 scripts/evaluate_kaytoo.py "$AVIANZ_TEST" "$DOC_TEST" \
  --kaytoo-root "$REPO_ROOT/../Kaytoo" \
  --mapping "$SCRIPT_DIR/data/DOC_bird_naming_map.csv" \
  --output "$OUT_KAYTOO"

echo "Running: BirdNET only"
python3 scripts/evaluate_birdnet.py "$AVIANZ_TEST" "$DOC_TEST" \
  --output "$OUT_BIRDNET"

echo "Running: RegNet on DOC"
python3 train.py "$DOC_TRAIN" "$OUT_REGNET_DOC" \
  --model-type regnet \
  --spec-transform Log \
  --bg-subtract \
  --kbird-prior 2.0 \
  --epochs 40 --patience 15 --seed 0 \
  --test-folder "$AVIANZ_TEST" \
  --test-folder2 "$DOC_TEST"

echo "Running: RegNet on combined DOC+AviaNZ"
python3 train.py "$BASE/combined_dataset/combined_large" "$OUT_REGNET_COMBINED" \
  --model-type regnet \
  --pretrained BirdClefModels/model_fold0.pth \
  --spec-transform Log \
  --bg-subtract \
  --kbird-prior 2.0 \
  --epochs 40 --patience 15 --seed 0 \
  --test-folder "$AVIANZ_TEST" \
  --test-folder2 "$DOC_TEST"

echo "Done."
