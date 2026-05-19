#!/bin/bash
set -e

# Master script: build both datasets then run all experiments.
#
# Matched group  → results in /local/scratch/freangi/matched_tests/
#   - RegNet trained on matched DOC data
#   - RegNet trained on matched AviaNZ data (benchmark)
#   - AST trained on matched DOC data
#   - RegNet trained on full (large) DOC data, evaluated on matched test sets
#   - Kaytoo pretrained evaluation on matched test sets
#   - BirdNET pretrained evaluation on matched test sets
#
# Large group    → results in /local/scratch/freangi/large_tests/
#   - RegNet trained on full DOC dataset, evaluated on large test sets
#   - RegNet trained on full AviaNZ dataset, evaluated on large test sets
#   - Kaytoo pretrained evaluation on large test sets
#   - BirdNET pretrained evaluation on large test sets
#
# Usage:
#   ./run_all.sh
#   ./run_all.sh --skip-build     # skip dataset builds if already done

SKIP_BUILD=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build) SKIP_BUILD=true; shift ;;
        *) echo "Unknown option: $1"; echo "Valid options: --skip-build"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "############################################################"
echo "  run_all.sh — full experiment pipeline"
echo "############################################################"
echo ""

# ── Step 1: Build matched dataset ─────────────────────────────────────────────
if [ "$SKIP_BUILD" = false ]; then
    echo "============================================================"
    echo " [1/8] Building matched dataset"
    echo "============================================================"
    bash build_dataset.sh
else
    echo "[1/8] Skipping matched dataset build (--skip-build)"
fi

# # ── Step 2: Build large (full) dataset ────────────────────────────────────────
# if [ "$SKIP_BUILD" = false ]; then
#     echo ""
#     echo "============================================================"
#     echo " [2/8] Building large dataset"
#     echo "============================================================"
#     bash build_large_dataset.sh
# else
#     echo "[2/8] Skipping large dataset build (--skip-build)"
# fi

# ── Step 3: Matched group — training experiments ───────────────────────────────
echo ""
echo "============================================================"
echo " [3/8] Matched group — training experiments"
echo "   RegNet/matched-DOC, RegNet/matched-AviaNZ,"
echo "   AST/matched-DOC, RegNet/large-DOC → matched test sets"
echo "============================================================"
bash run_experiments.sh

# ── Step 4: Matched group — Kaytoo evaluation ─────────────────────────────────
echo ""
echo "============================================================"
echo " [4/8] Matched group — Kaytoo evaluation"
echo "============================================================"
bash run_kaytoo_eval.sh

# ── Step 5: Matched group — BirdNET evaluation ────────────────────────────────
echo ""
echo "============================================================"
echo " [5/8] Matched group — BirdNET evaluation"
echo "============================================================"
bash run_birdnet_eval.sh

# # ── Step 6: Large group — training experiments ────────────────────────────────
# echo ""
# echo "============================================================"
# echo " [6/8] Large group — training experiments"
# echo "   RegNet/large-DOC, RegNet/large-AviaNZ → large test sets"
# echo "============================================================"
# bash run_large_experiment.sh

# # ── Step 7: Large group — Kaytoo evaluation ───────────────────────────────────
# echo ""
# echo "============================================================"
# echo " [7/8] Large group — Kaytoo evaluation"
# echo "============================================================"
# bash run_kaytoo_eval.sh --large

# # ── Step 8: Large group — BirdNET evaluation ──────────────────────────────────
# echo ""
# echo "============================================================"
# echo " [8/8] Large group — BirdNET evaluation"
# echo "============================================================"
# bash run_birdnet_eval.sh --large

echo ""
echo "############################################################"
echo "  All done."
echo ""
echo "  Matched results : /local/scratch/freangi/matched_tests/"
echo "  Large results   : /local/scratch/freangi/large_tests/"
echo ""
echo "  Analyse with:"
echo "    python3 scripts/analyze_all_results.py"
echo "############################################################"
echo ""
