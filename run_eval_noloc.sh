#!/usr/bin/env bash
# run_eval_noloc.sh
#
# Recreate the matched dataset split WITHOUT DOC recording-location grouping,
# then evaluate every trained model on the new test sets.
#
# Training is skipped entirely — existing model checkpoints are reused.
# Only the split step and evaluation steps are run.
#
# The new test splits are written to:
#   ${MATCHED}/avianz_split_noloc/
#   ${MATCHED}/doc_split_noloc/
#
# Evaluation results land under *_noloc subdirectories, e.g.:
#   ${MATCHED_TESTS}/regnet_on_doc_baseline_noloc/
#
# Usage:
#   ./run_eval_noloc.sh                  # skip steps whose outputs already exist
#   ./run_eval_noloc.sh --overwrite      # force re-split and re-evaluate
#   ./run_eval_noloc.sh --dry-run        # print commands without executing

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
BASE="/local/scratch/freangi"
KAYTOO_ROOT="$(pwd)/../Kaytoo"
KAYTOO_PYTHON="${KAYTOO_ROOT}/venv_kay/bin/python"
MAPPING="data/DOC_bird_naming_map.csv"
SEED=0
NOLOC_SUFFIX="_noloc"

MATCHED="${BASE}/matched"
AVIANZ_MATCHED="${MATCHED}/avianz_matched"
DOC_MATCHED="${MATCHED}/doc_matched"

# New (noloc) test splits
AVIANZ_TEST="${MATCHED}/avianz_split${NOLOC_SUFFIX}/test"
DOC_TEST="${MATCHED}/doc_split${NOLOC_SUFFIX}/test"

# Training data folders — still needed for class-name lookup during eval
DOC_TRAIN_ORIG="${MATCHED}/doc_split/train"
FULL_DOC_TRAIN="${BASE}/combined_dataset/doc_large"
COMBINED_TRAIN="${BASE}/combined_dataset/combined_large"

# Existing model checkpoints
MATCHED_TESTS="${BASE}/matched_tests"
FULL_DOC_TESTS="${BASE}/full_doc_tests"
COMBINED_TESTS="${BASE}/combined_tests"

OVERWRITE=0
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        --overwrite) OVERWRITE=1 ;;
        --dry-run)   DRY_RUN=1 ;;
        *) echo "Unknown argument: $arg"; echo "Usage: $0 [--overwrite] [--dry-run]"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────
run() {
    echo "  \$ $*"
    [[ "$DRY_RUN" -eq 0 ]] && PYTHONPATH="$(pwd)" "$@"
}

skip_or_run() {
    local marker="$1"; shift
    if [[ "$OVERWRITE" -eq 0 && -e "$marker" ]]; then
        echo "  [skip] $(basename "$(dirname "$marker")") — output exists"
    else
        run "$@"
    fi
}

header() { echo ""; echo "════════════════════════════════════════════════"; echo " $*"; echo "════════════════════════════════════════════════"; }

[[ "$DRY_RUN" -eq 1 ]] && echo "DRY RUN — no commands will be executed"

# ── Step 1: Re-split without DOC location grouping ────────────────────────
header "Step 1 / 4 — Re-split matched data (no DOC location grouping)"

SPLIT_MARKER="${MATCHED}/doc_split${NOLOC_SUFFIX}/test/labels.json"
if [[ "$OVERWRITE" -eq 1 || ! -f "$SPLIT_MARKER" ]]; then
    run python3 src/experiments/split_matched_datasets.py \
        "$AVIANZ_MATCHED" \
        "$DOC_MATCHED" \
        "$MATCHED" \
        --test-ratio 0.25 \
        --seed 42 \
        --no-location-grouping \
        --suffix "$NOLOC_SUFFIX" \
        --overwrite
else
    echo "  [skip] noloc splits already exist"
fi

# ── Step 2: Matched experiments (eval only) ───────────────────────────────
header "Step 2 / 4 — Matched experiments (eval only, new test splits)"

# BirdNET pretrained — re-runs from scratch on new test set (no checkpoint)
skip_or_run "${MATCHED_TESTS}/birdnet_pretrained${NOLOC_SUFFIX}_seed${SEED}/result.json" \
    python3 scripts/evaluate_birdnet.py \
        "$AVIANZ_TEST" "$DOC_TEST" \
        --output "${MATCHED_TESTS}/birdnet_pretrained${NOLOC_SUFFIX}_seed${SEED}"

# Kaytoo pretrained — re-runs from scratch on new test set (no checkpoint)
skip_or_run "${MATCHED_TESTS}/kaytoo_pretrained${NOLOC_SUFFIX}_seed${SEED}/result.json" \
    "$KAYTOO_PYTHON" scripts/evaluate_kaytoo.py \
        "$AVIANZ_TEST" "$DOC_TEST" \
        --kaytoo-root "$KAYTOO_ROOT" \
        --mapping     "$MAPPING" \
        --output      "${MATCHED_TESTS}/kaytoo_pretrained${NOLOC_SUFFIX}_seed${SEED}"

# Kaytoo finetuned — eval-only using saved deploy/ from original finetune run
# First, ensure the deploy/ artefacts are in the noloc output directory
ORIG_FT_DIR="${MATCHED_TESTS}/kaytoo_finetuned_seed${SEED}"
NOLOC_FT_DIR="${MATCHED_TESTS}/kaytoo_finetuned${NOLOC_SUFFIX}_seed${SEED}"
if [[ "$DRY_RUN" -eq 0 && -d "$ORIG_FT_DIR/deploy" && ! -d "$NOLOC_FT_DIR/deploy" ]]; then
    echo "  Copying deploy/ from original finetuned run → noloc output dir"
    mkdir -p "$NOLOC_FT_DIR"
    cp -r "$ORIG_FT_DIR/deploy" "$NOLOC_FT_DIR/deploy"
fi
skip_or_run "${NOLOC_FT_DIR}/result.json" \
    "$KAYTOO_PYTHON" scripts/finetune_kaytoo.py \
        --avianz-test "$AVIANZ_TEST" \
        --doc-test    "$DOC_TEST" \
        --kaytoo-root "$KAYTOO_ROOT" \
        --mapping     "$MAPPING" \
        --output      "$NOLOC_FT_DIR" \
        --eval-only

# Copy the existing finetuned deploy/ artefacts so --eval-only can find them
# (the deploy/ folder lives under the original training output directory)
ORIG_FT_DIR="${MATCHED_TESTS}/kaytoo_finetuned_seed${SEED}"
NOLOC_FT_DIR="${MATCHED_TESTS}/kaytoo_finetuned${NOLOC_SUFFIX}_seed${SEED}"
if [[ "$DRY_RUN" -eq 0 && -d "$ORIG_FT_DIR/deploy" && ! -d "$NOLOC_FT_DIR/deploy" ]]; then
    echo "  Copying deploy/ from original finetuned run → noloc output dir"
    mkdir -p "$NOLOC_FT_DIR"
    cp -r "$ORIG_FT_DIR/deploy" "$NOLOC_FT_DIR/deploy"
fi

# RegNet baseline — eval-only
skip_or_run "${MATCHED_TESTS}/regnet_on_doc_baseline${NOLOC_SUFFIX}/regnet_multilabel_metrics.csv" \
    python3 train.py \
        "$DOC_TRAIN_ORIG" \
        "${MATCHED_TESTS}/regnet_on_doc_baseline${NOLOC_SUFFIX}" \
        --model-type  regnet \
        --checkpoint  "${MATCHED_TESTS}/regnet_on_doc_baseline/regnet_model.pt" \
        --spec-transform Log \
        --eval-only \
        --test-folder  "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST"

# RegNet +BgSub — eval-only
skip_or_run "${MATCHED_TESTS}/regnet_on_doc_bgsub${NOLOC_SUFFIX}/regnet_multilabel_metrics.csv" \
    python3 train.py \
        "$DOC_TRAIN_ORIG" \
        "${MATCHED_TESTS}/regnet_on_doc_bgsub${NOLOC_SUFFIX}" \
        --model-type  regnet \
        --checkpoint  "${MATCHED_TESTS}/regnet_on_doc_bgsub/regnet_model.pt" \
        --spec-transform Log \
        --bg-subtract \
        --eval-only \
        --test-folder  "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST"

# ── Step 3: Full-DOC and Combined experiments (eval only) ─────────────────
header "Step 3 / 4 — Full-DOC and Combined experiments (eval only)"

# RegNet +BgSub Full DOC — eval-only
skip_or_run "${FULL_DOC_TESTS}/regnet_full_doc_bgsubtract${NOLOC_SUFFIX}_seed${SEED}/regnet_multilabel_metrics.csv" \
    python3 train.py \
        "$FULL_DOC_TRAIN" \
        "${FULL_DOC_TESTS}/regnet_full_doc_bgsubtract${NOLOC_SUFFIX}_seed${SEED}" \
        --model-type  regnet \
        --checkpoint  "${FULL_DOC_TESTS}/regnet_full_doc_bgsubtract_seed${SEED}/regnet_model.pt" \
        --spec-transform Log \
        --bg-subtract \
        --eval-only \
        --test-folder  "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST"

# RegNet +BgSub Combined pre-train — eval-only
skip_or_run "${COMBINED_TESTS}/regnet_combined_bgsubtract${NOLOC_SUFFIX}_seed${SEED}/regnet_multilabel_metrics.csv" \
    python3 train.py \
        "$COMBINED_TRAIN" \
        "${COMBINED_TESTS}/regnet_combined_bgsubtract${NOLOC_SUFFIX}_seed${SEED}" \
        --model-type  regnet \
        --checkpoint  "${COMBINED_TESTS}/regnet_combined_bgsubtract_seed${SEED}/regnet_model_best.pt" \
        --spec-transform Log \
        --bg-subtract \
        --eval-only \
        --test-folder  "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST"

# RegNet +BgSub Combined finetune — eval-only
skip_or_run "${COMBINED_TESTS}/regnet_combined_bgsubtract_ft${NOLOC_SUFFIX}_seed${SEED}/regnet_multilabel_metrics.csv" \
    python3 train.py \
        "$DOC_TRAIN_ORIG" \
        "${COMBINED_TESTS}/regnet_combined_bgsubtract_ft${NOLOC_SUFFIX}_seed${SEED}" \
        --model-type  regnet \
        --checkpoint  "${COMBINED_TESTS}/regnet_combined_bgsubtract_ft_seed${SEED}/regnet_model.pt" \
        --spec-transform Log \
        --bg-subtract \
        --eval-only \
        --test-folder  "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST"

# ── Step 4: Analyse results ───────────────────────────────────────────────
header "Step 4 / 4 — Analyse results"
run python3 scripts/analyze_all_results.py \
        "${MATCHED_TESTS}" \
        --output "matched_tests/analysis${NOLOC_SUFFIX}"
run python3 scripts/analyze_all_results.py \
        "${FULL_DOC_TESTS}" \
        --output "full_doc_tests/analysis${NOLOC_SUFFIX}"
run python3 scripts/analyze_all_results.py \
        "${COMBINED_TESTS}" \
        --output "combined_tests/analysis${NOLOC_SUFFIX}"

echo ""
echo "════════════════════════════════════════════════"
echo " Done.  Results in *_noloc subdirectories."
echo "════════════════════════════════════════════════"
