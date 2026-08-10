#!/usr/bin/env bash
set -euo pipefail

# Run the five comparison experiments:
#   1. Kaytoo pretrained
#   2. BirdNET pretrained
#   3. RegNet + bg-subtract + kbird-prior 2, trained on DOC only
#   4. RegNet + bg-subtract + kbird-prior 2, trained on DOC + AviaNZ
#   5. RegNet + bg-subtract + kbird-prior 2 + apply-reverb, trained on DOC only
#
# Usage:
#   bash run_five_experiments.sh
#   bash run_five_experiments.sh --force
#   bash run_five_experiments.sh --rebuild-data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

BASE="${AVIA_NZ_BASE:-/local/scratch/freangi}"
KAYTOO_ROOT="${KAYTOO_ROOT:-$REPO_ROOT/../Kaytoo}"
DOC_RAW_DIR_OVERRIDE="${DOC_RAW_DIR:-}"
AVIANZ_RAW_DIR_OVERRIDE="${AVIANZ_RAW_DIR:-}"
KAYTOO_CORES="${KAYTOO_CORES:-4}"
FORCE=false
REBUILD_DATA=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --doc-raw) DOC_RAW_DIR_OVERRIDE="$2"; shift 2 ;;
    --avianz-raw) AVIANZ_RAW_DIR_OVERRIDE="$2"; shift 2 ;;
    --kaytoo-root) KAYTOO_ROOT="$2"; shift 2 ;;
    --kaytoo-cores) KAYTOO_CORES="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    --rebuild-data) REBUILD_DATA=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

MATCHED="${BASE}/matched"
AVIANZ_TEST="${MATCHED}/avianz_split/test"
DOC_TEST="${MATCHED}/doc_split/test"
COMBINED_DATASET="${BASE}/combined_dataset/merged_train"
COMBINED_DOC_HALF="${BASE}/combined_dataset/doc_large"
COMBINED_AVIANZ_TEST="${BASE}/combined_dataset/avianz_split/test"
COMBINED_DOC_TEST="${BASE}/combined_dataset/doc_split/test"

# All four evaluation datasets
FOUR_TEST_FOLDERS=(
  "$DOC_TEST"
  "$AVIANZ_TEST"
  "$COMBINED_DOC_TEST"
  "$COMBINED_AVIANZ_TEST"
)

OUT_ROOT="${BASE}/model_tests"
OUT_KAYTOO="${OUT_ROOT}/kaytoo_pretrained_seed0"
OUT_BIRDNET="${OUT_ROOT}/birdnet_pretrained_seed0"
OUT_REGNET_DOC="${OUT_ROOT}/regnet_on_doc_bgsub"
OUT_REGNET_COMBINED="${OUT_ROOT}/regnet_combined_bgsubtract_seed0"
OUT_REGNET_DOC_REVERB="${OUT_ROOT}/regnet_on_doc_bgsub_reverb"

PRETRAINED_MODEL="${BIRDCLEF_PRETRAINED_PATH:-BirdClefModels/model_fold0.pth}"

# Helper to get dataset key from folder path
get_dataset_key() {
  local folder="$1"
  echo "$(basename "$(dirname "$(dirname "$folder")")")__$(basename "$(dirname "$folder")")"
}

echo "============================================================"
echo " Output base   : $BASE"
echo " Experiments   : $OUT_ROOT"
echo " Matched data  : $MATCHED"
echo " Combined data : $COMBINED_DATASET"
echo " Kaytoo root   : $KAYTOO_ROOT"
echo " Pretrained    : $PRETRAINED_MODEL"
echo " Force rerun   : $FORCE"
echo "============================================================"

# ---------------------------------------------------------------- datasets
BUILD_FLAGS=()
[[ "$REBUILD_DATA" == true ]] && BUILD_FLAGS+=(--overwrite)

if [[ "$REBUILD_DATA" == true \
   || ! -f "$AVIANZ_TEST/labels.json" \
   || ! -f "$DOC_TEST/labels.json" ]]; then
  echo ""
  echo ">>> Matched datasets"
  AVIA_NZ_BASE="$BASE" \
  DOC_RAW_DIR="$DOC_RAW_DIR_OVERRIDE" \
  AVIANZ_RAW_DIR="$AVIANZ_RAW_DIR_OVERRIDE" \
    bash build_dataset.sh ${BUILD_FLAGS[@]+"${BUILD_FLAGS[@]}"}
else
  echo "--- matched datasets: present, skipping build"
fi

if [[ "$REBUILD_DATA" == true \
   || ! -f "$COMBINED_DATASET/labels.json" \
   || ! -f "$COMBINED_AVIANZ_TEST/labels.json" \
   || ! -f "$COMBINED_DOC_TEST/labels.json" ]]; then
  echo ""
  echo ">>> Combined dataset"
  AVIA_NZ_BASE="$BASE" \
  DOC_RAW_DIR="$DOC_RAW_DIR_OVERRIDE" \
    bash build_combined_dataset.sh ${BUILD_FLAGS[@]+"${BUILD_FLAGS[@]}"}
else
  echo "--- combined dataset: present, skipping build"
fi

# ------------------------------------------------------------- 1. Kaytoo
echo ""
echo ">>> 1/5 Kaytoo pretrained"
for folder in "${FOUR_TEST_FOLDERS[@]}"; do
  dataset_key="$(get_dataset_key "$folder")"
  marker="$OUT_KAYTOO/$dataset_key/done"
  
  if [[ "$FORCE" == false && -f "$marker" ]]; then
    echo "  Skipping $dataset_key (already evaluated)"
    continue
  fi
  
  mkdir -p "$(dirname "$marker")"
  python3 scripts/evaluate_kaytoo.py "$folder" \
    --kaytoo-root "$KAYTOO_ROOT" \
    --mapping "$SCRIPT_DIR/data/DOC_bird_naming_map.csv" \
    --cores "$KAYTOO_CORES" \
    --output "$(dirname "$marker")"
  touch "$marker"
done

# ------------------------------------------------------------ 2. BirdNET
echo ""
echo ">>> 2/5 BirdNET pretrained"
for folder in "${FOUR_TEST_FOLDERS[@]}"; do
  dataset_key="$(get_dataset_key "$folder")"
  marker="$OUT_BIRDNET/$dataset_key/done"
  
  if [[ "$FORCE" == false && -f "$marker" ]]; then
    echo "  Skipping $dataset_key (already evaluated)"
    continue
  fi
  
  mkdir -p "$(dirname "$marker")"
  python3 scripts/evaluate_birdnet.py "$folder" \
    --output "$(dirname "$marker")"
  touch "$marker"
done

# -------------------------------------------------------- 3. RegNet / DOC
echo ""
echo ">>> 3/5 RegNet + bgsub, DOC only"

# Training
training_marker="$OUT_REGNET_DOC/training_history.json"
if [[ "$FORCE" == false && -f "$training_marker" ]]; then
  echo "  Training already done, skipping"
else
  python3 train.py "$COMBINED_DOC_HALF" "$OUT_REGNET_DOC" \
    --model-type regnet \
    --pretrained "$PRETRAINED_MODEL" \
    --spec-transform Log \
    --bg-subtract \
    --kbird-prior 2.0 \
    --epochs 40 --patience 15 --seed 0
fi

# Evaluation
for folder in "${FOUR_TEST_FOLDERS[@]}"; do
  dataset_key="$(get_dataset_key "$folder")"
  marker="$OUT_REGNET_DOC/$dataset_key/done"
  
  if [[ "$FORCE" == false && -f "$marker" ]]; then
    echo "  Skipping $dataset_key (already evaluated)"
    continue
  fi
  
  mkdir -p "$(dirname "$marker")"
  python3 train.py "$COMBINED_DOC_HALF" "$OUT_REGNET_DOC" \
    --model-type regnet \
    --spec-transform Log \
    --bg-subtract \
    --kbird-prior 2.0 \
    --seed 0 \
    --eval-only --test-folder "$folder"
  touch "$marker"
done

# --------------------------------------------------- 4. RegNet / combined
echo ""
echo ">>> 4/5 RegNet + bgsub, combined DOC + AviaNZ"

# Training
training_marker="$OUT_REGNET_COMBINED/training_history.json"
if [[ "$FORCE" == false && -f "$training_marker" ]]; then
  echo "  Training already done, skipping"
else
  python3 train.py "$COMBINED_DATASET" "$OUT_REGNET_COMBINED" \
    --model-type regnet \
    --pretrained "$PRETRAINED_MODEL" \
    --spec-transform Log \
    --bg-subtract \
    --kbird-prior 2.0 \
    --epochs 40 --patience 15 --seed 0
fi

# Evaluation
for folder in "${FOUR_TEST_FOLDERS[@]}"; do
  dataset_key="$(get_dataset_key "$folder")"
  marker="$OUT_REGNET_COMBINED/$dataset_key/done"
  
  if [[ "$FORCE" == false && -f "$marker" ]]; then
    echo "  Skipping $dataset_key (already evaluated)"
    continue
  fi
  
  mkdir -p "$(dirname "$marker")"
  python3 train.py "$COMBINED_DATASET" "$OUT_REGNET_COMBINED" \
    --model-type regnet \
    --spec-transform Log \
    --bg-subtract \
    --kbird-prior 2.0 \
    --seed 0 \
    --eval-only --test-folder "$folder"
  touch "$marker"
done

# --------------------------------------------------- 5. RegNet / DOC + reverb
echo ""
echo ">>> 5/5 RegNet + bgsub + apply-reverb, DOC only"

# Training
training_marker="$OUT_REGNET_DOC_REVERB/training_history.json"
if [[ "$FORCE" == false && -f "$training_marker" ]]; then
  echo "  Training already done, skipping"
else
  python3 train.py "$COMBINED_DOC_HALF" "$OUT_REGNET_DOC_REVERB" \
    --model-type regnet \
    --pretrained "$PRETRAINED_MODEL" \
    --spec-transform Log \
    --bg-subtract \
    --kbird-prior 2.0 \
    --apply-reverb \
    --epochs 40 --patience 15 --seed 0
fi

# Evaluation
for folder in "${FOUR_TEST_FOLDERS[@]}"; do
  dataset_key="$(get_dataset_key "$folder")"
  marker="$OUT_REGNET_DOC_REVERB/$dataset_key/done"
  
  if [[ "$FORCE" == false && -f "$marker" ]]; then
    echo "  Skipping $dataset_key (already evaluated)"
    continue
  fi
  
  mkdir -p "$(dirname "$marker")"
  python3 train.py "$COMBINED_DOC_HALF" "$OUT_REGNET_DOC_REVERB" \
    --model-type regnet \
    --spec-transform Log \
    --bg-subtract \
    --kbird-prior 2.0 \
    --apply-reverb \
    --seed 0 \
    --eval-only --test-folder "$folder"
  touch "$marker"
done

echo ""
echo "All experiments complete!"