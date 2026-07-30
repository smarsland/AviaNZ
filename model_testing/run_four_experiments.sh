#!/usr/bin/env bash
set -euo pipefail

# Run the four comparison experiments:
#   1. Kaytoo pretrained, on its own
#   2. BirdNET pretrained, on its own
#   3. RegNet + bg-subtract + kbird-prior 2, trained on DOC only
#   4. RegNet + bg-subtract + kbird-prior 2, trained on combined DOC + AviaNZ
#
# Every stage is resumable: a stage that has already written its results is
# skipped. Nothing is rebuilt or retrained unless it is actually missing.
#
# Usage:
#   bash run_four_experiments.sh
#   bash run_four_experiments.sh --only kaytoo
#   bash run_four_experiments.sh --only regnet_doc,regnet_combined
#   bash run_four_experiments.sh --force            # redo everything
#   bash run_four_experiments.sh --rebuild-data     # rebuild datasets from raw

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
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --doc-raw) DOC_RAW_DIR_OVERRIDE="$2"; shift 2 ;;
    --avianz-raw) AVIANZ_RAW_DIR_OVERRIDE="$2"; shift 2 ;;
    --kaytoo-root) KAYTOO_ROOT="$2"; shift 2 ;;
    --kaytoo-cores) KAYTOO_CORES="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    --rebuild-data) REBUILD_DATA=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

MATCHED="${BASE}/matched"
AVIANZ_TEST="${MATCHED}/avianz_split/test"
DOC_TRAIN="${MATCHED}/doc_split/train"
DOC_TEST="${MATCHED}/doc_split/test"
COMBINED_DATASET="${BASE}/combined_dataset/combined_large"
COMBINED_DOC_HALF="${BASE}/combined_dataset/doc_large"

OUT_KAYTOO="${BASE}/matched_tests/kaytoo_pretrained_seed0"
OUT_BIRDNET="${BASE}/matched_tests/birdnet_pretrained_seed0"
OUT_REGNET_DOC="${BASE}/matched_tests/regnet_on_doc_bgsub"
OUT_REGNET_COMBINED="${BASE}/combined_tests/regnet_combined_bgsubtract_seed0"

PRETRAINED_MODEL="${BIRDCLEF_PRETRAINED_PATH:-BirdClefModels/model_fold0.pth}"

# A stage is done if its completion marker exists.
#   evaluate_kaytoo.py / evaluate_birdnet.py -> result.json
#   train.py                                 -> training_history.json
should_run() {
  local stage="$1" marker="$2"
  if [[ -n "$ONLY" && ",$ONLY," != *",$stage,"* ]]; then
    echo "--- $stage: not selected by --only, skipping"
    return 1
  fi
  if [[ "$FORCE" == false && -f "$marker" ]]; then
    echo "--- $stage: already done ($marker), skipping"
    return 1
  fi
  return 0
}

echo "============================================================"
echo " Output base   : $BASE"
echo " Matched data  : $MATCHED"
echo " Combined data : $COMBINED_DATASET"
echo " Kaytoo root   : $KAYTOO_ROOT"
echo " Pretrained    : $PRETRAINED_MODEL"
echo " Force rerun   : $FORCE"
echo "============================================================"

# ---------------------------------------------------------------- datasets
# build_dataset.sh / build_combined_dataset.sh already skip work that is done,
# so --overwrite is only passed when explicitly asked for. Passing it
# unconditionally is what caused full rebuilds from raw audio on every run.
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

if [[ "$REBUILD_DATA" == true || ! -f "$COMBINED_DATASET/labels.json" ]]; then
  echo ""
  echo ">>> Combined dataset"
  AVIA_NZ_BASE="$BASE" \
  DOC_RAW_DIR="$DOC_RAW_DIR_OVERRIDE" \
    bash build_combined_dataset.sh ${BUILD_FLAGS[@]+"${BUILD_FLAGS[@]}"}
else
  echo "--- combined dataset: present, skipping build"
fi

# ------------------------------------------------------------- 1. Kaytoo
if should_run kaytoo "$OUT_KAYTOO/result.json"; then
  echo ""
  echo ">>> 1/4 Kaytoo pretrained"
  mkdir -p "$OUT_KAYTOO"
  python3 scripts/evaluate_kaytoo.py "$AVIANZ_TEST" "$DOC_TEST" \
    --kaytoo-root "$KAYTOO_ROOT" \
    --mapping "$SCRIPT_DIR/data/DOC_bird_naming_map.csv" \
    --cores "$KAYTOO_CORES" \
    --output "$OUT_KAYTOO"
fi

# ------------------------------------------------------------ 2. BirdNET
if should_run birdnet "$OUT_BIRDNET/result.json"; then
  echo ""
  echo ">>> 2/4 BirdNET pretrained"
  mkdir -p "$OUT_BIRDNET"
  python3 scripts/evaluate_birdnet.py "$AVIANZ_TEST" "$DOC_TEST" \
    --output "$OUT_BIRDNET"
fi

# -------------------------------------------------------- 3. RegNet / DOC
if should_run regnet_doc "$OUT_REGNET_DOC/training_history.json"; then
  echo ""
  echo ">>> 3/4 RegNet + bgsub, DOC only"
  python3 train.py "$COMBINED_DOC_HALF" "$OUT_REGNET_DOC" \
    --model-type regnet \
    --pretrained "$PRETRAINED_MODEL" \
    --spec-transform Log \
    --bg-subtract \
    --kbird-prior 2.0 \
    --epochs 40 --patience 15 --seed 0 \
    --test-folder "$AVIANZ_TEST" \
    --test-folder2 "$DOC_TEST"
fi

# --------------------------------------------------- 4. RegNet / combined
if should_run regnet_combined "$OUT_REGNET_COMBINED/training_history.json"; then
  echo ""
  echo ">>> 4/4 RegNet + bgsub, combined DOC + AviaNZ"
  python3 train.py "$COMBINED_DATASET" "$OUT_REGNET_COMBINED" \
    --model-type regnet \
    --pretrained "$PRETRAINED_MODEL" \
    --spec-transform Log \
    --bg-subtract \
    --kbird-prior 2.0 \
    --epochs 40 --patience 15 --seed 0 \
    --test-folder "$AVIANZ_TEST" \
    --test-folder2 "$DOC_TEST"
fi

echo ""
echo "============================================================"
echo " All four experiments accounted for."
echo "   Kaytoo   : $OUT_KAYTOO"
echo "   BirdNET  : $OUT_BIRDNET"
echo "   RegNet   : $OUT_REGNET_DOC"
echo "   Combined : $OUT_REGNET_COMBINED"
echo ""
echo " Next:"
echo "   python3 scripts/analyze_all_results.py $BASE/matched_tests --output $BASE/matched_tests/analysis"
echo "   python3 scripts/analyze_all_results.py $BASE/combined_tests --output $BASE/combined_tests/analysis"
echo "   python3 scripts/make_summary_figure.py"
echo "============================================================"
