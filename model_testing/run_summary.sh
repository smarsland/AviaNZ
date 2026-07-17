#!/usr/bin/env bash
# run_summary.sh
#
# Single script that builds all datasets and trains/evaluates every model
# required to produce the summary figure.  Results land under $BASE on the
# server; analysis CSVs are written back to the workspace so that
# make_summary_figure.py can be run locally.
#
# Models produced (in summary figure order):
#   1. BirdNET pretrained              (evaluation only)
#   2. RegNet Baseline                 (trained on matched DOC)
#   3. RegNet +BgSub                   (trained on matched DOC)
#   4. Kaytoo pretrained               (evaluation only)
#   5. Kaytoo finetuned                (finetuned on matched DOC)
#   6. RegNet +BgSub Full DOC          (trained on all DOC species)
#   7. RegNet +BgSub Combined          (trained on DOC + all AviaNZ)
#   8. RegNet +BgSub Combined (ft)     (combined pre-train → matched finetune)
#
# Usage:
#   ./run_summary.sh                   # skip steps whose outputs already exist
#   ./run_summary.sh --overwrite       # force rebuild of datasets + all models
#   ./run_summary.sh --dry-run         # print commands without executing

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
BASE="/local/scratch/freangi"
DRIVE="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02"
KAYTOO_ROOT="$(pwd)/../Kaytoo"
KAYTOO_PYTHON="${KAYTOO_ROOT}/venv_kay/bin/python"
PRETRAINED="BirdClefModels/model_fold0.pth"
MAPPING="data/DOC_bird_naming_map.csv"
SEED=0

MATCHED="${BASE}/matched"
AVIANZ_TEST="${MATCHED}/avianz_split/test"
DOC_TRAIN="${MATCHED}/doc_split/train"
DOC_TEST="${MATCHED}/doc_split/test"

# full-DOC training reuses the doc_large subset produced by build_combined_dataset.sh
FULL_DOC_TRAIN="${BASE}/combined_dataset/doc_large"
COMBINED_TRAIN="${BASE}/combined_dataset/combined_large"

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

# ── Step 1: Build matched dataset ─────────────────────────────────────────
header "Step 1 / 7 — Build matched dataset"
if [[ "$OVERWRITE" -eq 1 ]]; then
    run bash build_dataset.sh --overwrite
else
    run bash build_dataset.sh
fi

# ── Step 2: Build combined dataset (also produces doc_large as a subset) ─
header "Step 2 / 7 — Build combined dataset (produces doc_large subset too)"
COMBINED_MARKER="${BASE}/combined_dataset/combined_large/labels.json"
DOC_LARGE_MARKER="${BASE}/combined_dataset/doc_large/labels.json"
if [[ "$OVERWRITE" -eq 1 || ! -f "$COMBINED_MARKER" || ! -f "$DOC_LARGE_MARKER" ]]; then
    run bash build_combined_dataset.sh ${OVERWRITE:+--overwrite}
else
    echo "  [skip] combined dataset — already exists"
fi

# ── Step 3: Matched experiments ───────────────────────────────────────────
header "Step 3 / 7 — Matched experiments (baseline + bgsub + BirdNET + Kaytoo)"

# BirdNET (eval only — needs audio/)
skip_or_run "${MATCHED_TESTS}/birdnet_pretrained_seed${SEED}/result.json" \
    python3 scripts/evaluate_birdnet.py \
        "$AVIANZ_TEST" "$DOC_TEST" \
        --output "${MATCHED_TESTS}/birdnet_pretrained_seed${SEED}"

# Kaytoo pretrained (eval only — needs audio/)
skip_or_run "${MATCHED_TESTS}/kaytoo_pretrained_seed${SEED}/result.json" \
    "$KAYTOO_PYTHON" scripts/evaluate_kaytoo.py \
        "$AVIANZ_TEST" "$DOC_TEST" \
        --kaytoo-root "$KAYTOO_ROOT" \
        --mapping     "$MAPPING" \
        --output      "${MATCHED_TESTS}/kaytoo_pretrained_seed${SEED}"

# Kaytoo finetuned
skip_or_run "${MATCHED_TESTS}/kaytoo_finetuned_seed${SEED}/result.json" \
    "$KAYTOO_PYTHON" scripts/finetune_kaytoo.py \
        --avianz-test "$AVIANZ_TEST" \
        --doc-train   "$DOC_TRAIN" \
        --doc-test    "$DOC_TEST" \
        --kaytoo-root "$KAYTOO_ROOT" \
        --mapping     "$MAPPING" \
        --output      "${MATCHED_TESTS}/kaytoo_finetuned_seed${SEED}" \
        --epochs 10 --lr 1e-4 --batch-size 16 --num-workers 4

# RegNet baseline
skip_or_run "${MATCHED_TESTS}/regnet_on_doc_baseline/regnet_model.pt" \
    python3 train.py \
        "$DOC_TRAIN" \
        "${MATCHED_TESTS}/regnet_on_doc_baseline" \
        --test-folder  "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST" \
        --epochs 30 --patience 15 --mixup 0.25 \
        --model-type regnet \
        --spec-transform Log

# RegNet +BgSub
skip_or_run "${MATCHED_TESTS}/regnet_on_doc_bgsub/regnet_model.pt" \
    python3 train.py \
        "$DOC_TRAIN" \
        "${MATCHED_TESTS}/regnet_on_doc_bgsub" \
        --test-folder  "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST" \
        --epochs 30 --patience 15 --mixup 0.25 \
        --model-type regnet \
        --spec-transform Log \
        --bg-subtract

# ── Step 4: Full-DOC experiment ───────────────────────────────────────────
header "Step 4 / 7 — Full-DOC experiment (RegNet +BgSub, reuses combined/doc_large)"
skip_or_run "${FULL_DOC_TESTS}/regnet_full_doc_bgsubtract_seed${SEED}/regnet_model.pt" \
    python3 train.py \
        "$FULL_DOC_TRAIN" \
        "${FULL_DOC_TESTS}/regnet_full_doc_bgsubtract_seed${SEED}" \
        --model-type  regnet \
        --pretrained  "$PRETRAINED" \
        --spec-transform Log \
        --bg-subtract \
        --kbird-prior 2.0 \
        --epochs 40 --patience 15 \
        --seed "$SEED" \
        --test-folder  "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST"

# ── Step 5: Combined pre-train ────────────────────────────────────────────
header "Step 5 / 7 — Combined pre-train (RegNet +BgSub)"
skip_or_run "${COMBINED_TESTS}/regnet_combined_bgsubtract_seed${SEED}/regnet_model_best.pt" \
    python3 train.py \
        "$COMBINED_TRAIN" \
        "${COMBINED_TESTS}/regnet_combined_bgsubtract_seed${SEED}" \
        --model-type  regnet \
        --pretrained  "$PRETRAINED" \
        --spec-transform Log \
        --bg-subtract \
        --kbird-prior 2.0 \
        --epochs 40 --patience 15 \
        --seed "$SEED" \
        --test-folder  "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST"

# ── Step 6: Combined finetune ─────────────────────────────────────────────
header "Step 6 / 7 — Combined finetune (RegNet +BgSub → matched)"
COMBINED_CKPT="${COMBINED_TESTS}/regnet_combined_bgsubtract_seed${SEED}/regnet_model_best.pt"
if [[ ! -f "$COMBINED_CKPT" && "$DRY_RUN" -eq 0 ]]; then
    echo "  ERROR: combined checkpoint not found — step 6 may have failed"
    exit 1
fi
skip_or_run "${COMBINED_TESTS}/regnet_combined_bgsubtract_ft_seed${SEED}/regnet_model.pt" \
    python3 train.py \
        "$DOC_TRAIN" \
        "${COMBINED_TESTS}/regnet_combined_bgsubtract_ft_seed${SEED}" \
        --model-type  regnet \
        --pretrained  "$COMBINED_CKPT" \
        --spec-transform Log \
        --bg-subtract \
        --kbird-prior 2.0 \
        --freeze-backbone \
        --lr 1e-4 \
        --epochs 20 --patience 10 \
        --seed "$SEED" \
        --test-folder  "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST"

# ── Step 7: Analyse + generate figure ─────────────────────────────────────
header "Step 7 / 7 — Analyse results + generate summary figure"
run python3 scripts/analyze_all_results.py "$MATCHED_TESTS"  --output matched_tests/analysis
run python3 scripts/analyze_all_results.py "$FULL_DOC_TESTS" --output full_doc_tests/analysis
run python3 scripts/analyze_all_results.py "$COMBINED_TESTS" --output combined_tests/analysis
run python3 scripts/make_summary_figure.py

echo ""
echo "════════════════════════════════════════════════"
echo " Done.  Summary figure → summary_figure/"
echo "════════════════════════════════════════════════"
