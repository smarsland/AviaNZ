#!/bin/bash
set -e
#
# Train RegNet models on the matched dataset and evaluate on both test sets.
# Run after build_dataset.sh.
#
# Ten runs (all on DOC matched train):
#    1. baseline             — Log transform
#    2. boxcox               — Box-Cox transform
#    3. kbird2               — baseline + k-bird prior 2
#    4. kbird4               — baseline + k-bird prior 4
#    5. bgsub                — baseline + background subtraction
#    6. bgmed                — baseline + background subtraction + median filter
#    7. no_background        — baseline + no background samples
#    8. delta                — baseline + delta + delta-delta channels
#    9. sed_head             — baseline + SED head
#   10. logminmax            — LogMinMax transform (Kaytoo-style)
#
# Results land in $OUTPUT and are picked up by scripts/analyze_all_results.py.
#
# Usage:
#   bash run_experiments.sh
#   bash run_experiments.sh --dry-run

set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

BASE="/local/scratch/freangi"
MATCHED="${BASE}/matched"
OUTPUT="${BASE}/matched_tests"

AVIANZ_TRAIN="${MATCHED}/avianz_split/train"
AVIANZ_TEST="${MATCHED}/avianz_split/test"
DOC_TRAIN="${MATCHED}/doc_split/train"
DOC_TEST="${MATCHED}/doc_split/test"

EPOCHS=30
PATIENCE=15
MIXUP=0.25
VIZ_SAMPLES=3

run_cmd() {
    echo "  \$ $*"
    if [ "$DRY_RUN" -eq 0 ]; then
        PYTHONPATH="$(pwd)" "$@"
    fi
}

run_experiment() {
    local model=$1
    local train_name=$2
    local train_dir=$3
    local transform_name=$4
    shift 4

    local out_dir="${OUTPUT}/${model}_on_${train_name}_${transform_name}"

    # Skip if already trained
    if [ -f "${out_dir}/${model}_model.pt" ] && [ "$DRY_RUN" -eq 0 ]; then
        echo "  [skip] ${model}_on_${train_name}_${transform_name} (already trained)"
        return
    fi

    echo ""
    echo "============================================================"
    echo " ${model}_on_${train_name}_${transform_name}"
    echo "============================================================"

    run_cmd python train.py \
        "$train_dir" \
        "$out_dir" \
        --test-folder  "$AVIANZ_TEST" \
        --test-folder2 "$DOC_TEST" \
        --visualize-attention \
        --viz-samples  $VIZ_SAMPLES \
        --epochs       $EPOCHS \
        --patience     $PATIENCE \
        --mixup        $MIXUP \
        --model-type   "$model" \
        "$@"
}

mkdir -p "$OUTPUT"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN — no commands will be executed"
fi

# # 1. Baseline — Log transform (default)
# run_experiment regnet doc "$DOC_TRAIN" baseline \
#     --spec-transform "Log"

# # 2. Alternative — Box-Cox transform
# run_experiment regnet doc "$DOC_TRAIN" boxcox \
#     --spec-transform "Box-Cox"

# # 3. Baseline + k-bird prior of 2
# run_experiment regnet doc "$DOC_TRAIN" kbird2 \
#     --spec-transform "Log" --kbird-prior 2.0

# # 4. Baseline + k-bird prior of 4
# run_experiment regnet doc "$DOC_TRAIN" kbird4 \
#     --spec-transform "Log" --kbird-prior 4.0

# # 5. Baseline + background subtraction
# run_experiment regnet doc "$DOC_TRAIN" bgsub \
#     --spec-transform "Log" --bg-subtract

# # 6. Baseline + background subtraction + median filter
# run_experiment regnet doc "$DOC_TRAIN" bgmed \
#     --spec-transform "Log" --bg-subtract --median-filter

# # 7. Baseline + no background samples
# run_experiment regnet doc "$DOC_TRAIN" no_background \
#     --spec-transform "Log" --no-background

# # 8. Baseline + delta + delta-delta channels
# run_experiment regnet doc "$DOC_TRAIN" delta \
#     --spec-transform "Log" --deltas

# # 9. Baseline + SED head
# run_experiment regnet doc "$DOC_TRAIN" sed_head \
#     --spec-transform "Log" --sed-head

# # 10. Baseline — LogMinMax transform (Kaytoo-style)
# run_experiment regnet doc "$DOC_TRAIN" logminmax \
#     --spec-transform "LogMinMax"

# # ======================================================================
# # EXTENDED EXPERIMENTS (11–100)
# #
# # Groups:
# #   A (11–25)  — bgsub combinations   (best single-flag on avianz)
# #   B (26–35)  — bgmed combinations   (joint-best single-flag on avianz)
# #   C (36–45)  — no-background combos (third-best single-flag)
# #   D (46–55)  — LogMinMax combos     (Kaytoo-style pipeline)
# #   E (56–62)  — PCEN transform       (never tested)
# #   F (63–71)  — loss-function variants on baseline
# #   G (72–76)  — CNN adapter
# #   H (77–80)  — additional k-bird prior values
# #   I (81–83)  — freeze-stages variants
# #   K (84–100) — large multi-flag combinations
# # ======================================================================

# # ─── Group A: bgsub combinations (11–25) ──────────────────────────────

# # 11. Background subtraction + delta channels
# run_experiment regnet doc "$DOC_TRAIN" bgsub_deltas \
#     --spec-transform "Log" --bg-subtract --deltas

# # 12. Background subtraction + no background samples
# run_experiment regnet doc "$DOC_TRAIN" bgsub_nobg \
#     --spec-transform "Log" --bg-subtract --no-background

# # 13. Background subtraction + SED head
# run_experiment regnet doc "$DOC_TRAIN" bgsub_sedhead \
#     --spec-transform "Log" --bg-subtract --sed-head

# # 14. Background subtraction + gated head
# run_experiment regnet doc "$DOC_TRAIN" bgsub_gated \
#     --spec-transform "Log" --bg-subtract --gated-head

# # 15. Background subtraction + Asymmetric Loss
# run_experiment regnet doc "$DOC_TRAIN" bgsub_asl \
#     --spec-transform "Log" --bg-subtract --use-asl

# # 16. Background subtraction + class weights
# run_experiment regnet doc "$DOC_TRAIN" bgsub_classw \
#     --spec-transform "Log" --bg-subtract --class-weights

# # 17. Background subtraction + k-bird prior 2
# run_experiment regnet doc "$DOC_TRAIN" bgsub_kbird2 \
#     --spec-transform "Log" --bg-subtract --kbird-prior 2.0

# # 18. Background subtraction + k-bird prior 4
# run_experiment regnet doc "$DOC_TRAIN" bgsub_kbird4 \
#     --spec-transform "Log" --bg-subtract --kbird-prior 4.0

# # 19. Background subtraction + CNN adapter
# run_experiment regnet doc "$DOC_TRAIN" bgsub_cnnadapter \
#     --spec-transform "Log" --bg-subtract --cnn-adapter

# # 20. Background subtraction + deltas + no background
# run_experiment regnet doc "$DOC_TRAIN" bgsub_deltas_nobg \
#     --spec-transform "Log" --bg-subtract --deltas --no-background

# # 21. Background subtraction + deltas + SED head
# run_experiment regnet doc "$DOC_TRAIN" bgsub_deltas_sedhead \
#     --spec-transform "Log" --bg-subtract --deltas --sed-head

# # 22. Background subtraction + deltas + ASL
# run_experiment regnet doc "$DOC_TRAIN" bgsub_deltas_asl \
#     --spec-transform "Log" --bg-subtract --deltas --use-asl

# # 23. Background subtraction + deltas + class weights
# run_experiment regnet doc "$DOC_TRAIN" bgsub_deltas_classw \
#     --spec-transform "Log" --bg-subtract --deltas --class-weights

# # 24. Background subtraction + deltas + k-bird prior 2
# run_experiment regnet doc "$DOC_TRAIN" bgsub_deltas_kbird2 \
#     --spec-transform "Log" --bg-subtract --deltas --kbird-prior 2.0

# # 25. Background subtraction + deltas + gated head
# run_experiment regnet doc "$DOC_TRAIN" bgsub_deltas_gated \
#     --spec-transform "Log" --bg-subtract --deltas --gated-head

# # ─── Group B: bgmed combinations (26–35) ──────────────────────────────

# # 26. bgmed + delta channels
# run_experiment regnet doc "$DOC_TRAIN" bgmed_deltas \
#     --spec-transform "Log" --bg-subtract --median-filter --deltas

# # 27. bgmed + no background
# run_experiment regnet doc "$DOC_TRAIN" bgmed_nobg \
#     --spec-transform "Log" --bg-subtract --median-filter --no-background

# # 28. bgmed + SED head
# run_experiment regnet doc "$DOC_TRAIN" bgmed_sedhead \
#     --spec-transform "Log" --bg-subtract --median-filter --sed-head

# # 29. bgmed + gated head
# run_experiment regnet doc "$DOC_TRAIN" bgmed_gated \
#     --spec-transform "Log" --bg-subtract --median-filter --gated-head

# # 30. bgmed + Asymmetric Loss
# run_experiment regnet doc "$DOC_TRAIN" bgmed_asl \
#     --spec-transform "Log" --bg-subtract --median-filter --use-asl

# # 31. bgmed + class weights
# run_experiment regnet doc "$DOC_TRAIN" bgmed_classw \
#     --spec-transform "Log" --bg-subtract --median-filter --class-weights

# # 32. bgmed + k-bird prior 2
# run_experiment regnet doc "$DOC_TRAIN" bgmed_kbird2 \
#     --spec-transform "Log" --bg-subtract --median-filter --kbird-prior 2.0

# # 33. bgmed + k-bird prior 4
# run_experiment regnet doc "$DOC_TRAIN" bgmed_kbird4 \
#     --spec-transform "Log" --bg-subtract --median-filter --kbird-prior 4.0

# # 34. bgmed + deltas + no background
# run_experiment regnet doc "$DOC_TRAIN" bgmed_deltas_nobg \
#     --spec-transform "Log" --bg-subtract --median-filter --deltas --no-background

# # 35. bgmed + deltas + ASL
# run_experiment regnet doc "$DOC_TRAIN" bgmed_deltas_asl \
#     --spec-transform "Log" --bg-subtract --median-filter --deltas --use-asl

# # ─── Group C: no-background combinations (36–45) ──────────────────────

# # 36. No background + delta channels
# run_experiment regnet doc "$DOC_TRAIN" nobg_deltas \
#     --spec-transform "Log" --no-background --deltas

# # 37. No background + SED head
# run_experiment regnet doc "$DOC_TRAIN" nobg_sedhead \
#     --spec-transform "Log" --no-background --sed-head

# # 38. No background + gated head
# run_experiment regnet doc "$DOC_TRAIN" nobg_gated \
#     --spec-transform "Log" --no-background --gated-head

# # 39. No background + Asymmetric Loss
# run_experiment regnet doc "$DOC_TRAIN" nobg_asl \
#     --spec-transform "Log" --no-background --use-asl

# # 40. No background + class weights
# run_experiment regnet doc "$DOC_TRAIN" nobg_classw \
#     --spec-transform "Log" --no-background --class-weights

# # 41. No background + k-bird prior 2
# run_experiment regnet doc "$DOC_TRAIN" nobg_kbird2 \
#     --spec-transform "Log" --no-background --kbird-prior 2.0

# # 42. No background + k-bird prior 4
# run_experiment regnet doc "$DOC_TRAIN" nobg_kbird4 \
#     --spec-transform "Log" --no-background --kbird-prior 4.0

# # 43. No background + deltas + SED head
# run_experiment regnet doc "$DOC_TRAIN" nobg_deltas_sedhead \
#     --spec-transform "Log" --no-background --deltas --sed-head

# # 44. No background + deltas + gated head
# run_experiment regnet doc "$DOC_TRAIN" nobg_deltas_gated \
#     --spec-transform "Log" --no-background --deltas --gated-head

# # 45. No background + deltas + ASL
# run_experiment regnet doc "$DOC_TRAIN" nobg_deltas_asl \
#     --spec-transform "Log" --no-background --deltas --use-asl

# # ─── Group D: LogMinMax (Kaytoo-style) combinations (46–55) ───────────

# # 46. LogMinMax + delta channels  (Kaytoo spectrogram pipeline)
# run_experiment regnet doc "$DOC_TRAIN" logminmax_deltas \
#     --spec-transform "LogMinMax" --deltas

# # 47. LogMinMax + deltas + SED head  ← closest to Kaytoo architecture
# run_experiment regnet doc "$DOC_TRAIN" logminmax_deltas_sedhead \
#     --spec-transform "LogMinMax" --deltas --sed-head

# # 48. LogMinMax + deltas + no background
# run_experiment regnet doc "$DOC_TRAIN" logminmax_deltas_nobg \
#     --spec-transform "LogMinMax" --deltas --no-background

# # 49. LogMinMax + deltas + gated head
# run_experiment regnet doc "$DOC_TRAIN" logminmax_deltas_gated \
#     --spec-transform "LogMinMax" --deltas --gated-head

# # 50. LogMinMax + deltas + ASL
# run_experiment regnet doc "$DOC_TRAIN" logminmax_deltas_asl \
#     --spec-transform "LogMinMax" --deltas --use-asl

# # 51. LogMinMax + no background
# run_experiment regnet doc "$DOC_TRAIN" logminmax_nobg \
#     --spec-transform "LogMinMax" --no-background

# # 52. LogMinMax + SED head
# run_experiment regnet doc "$DOC_TRAIN" logminmax_sedhead \
#     --spec-transform "LogMinMax" --sed-head

# # 53. LogMinMax + gated head
# run_experiment regnet doc "$DOC_TRAIN" logminmax_gated \
#     --spec-transform "LogMinMax" --gated-head

# # 54. LogMinMax + class weights
# run_experiment regnet doc "$DOC_TRAIN" logminmax_classw \
#     --spec-transform "LogMinMax" --class-weights

# # 55. LogMinMax + deltas + class weights
# run_experiment regnet doc "$DOC_TRAIN" logminmax_deltas_classw \
#     --spec-transform "LogMinMax" --deltas --class-weights

# # ─── Group E: PCEN transform (56–62) ──────────────────────────────────

# # 56. PCEN transform alone
# run_experiment regnet doc "$DOC_TRAIN" pcen \
#     --spec-transform "PCEN"

# # 57. PCEN + delta channels
# run_experiment regnet doc "$DOC_TRAIN" pcen_deltas \
#     --spec-transform "PCEN" --deltas

# # 58. PCEN + SED head
# run_experiment regnet doc "$DOC_TRAIN" pcen_sedhead \
#     --spec-transform "PCEN" --sed-head

# # 59. PCEN + gated head
# run_experiment regnet doc "$DOC_TRAIN" pcen_gated \
#     --spec-transform "PCEN" --gated-head

# # 60. PCEN + no background
# run_experiment regnet doc "$DOC_TRAIN" pcen_nobg \
#     --spec-transform "PCEN" --no-background

# # 61. PCEN + deltas + no background
# run_experiment regnet doc "$DOC_TRAIN" pcen_deltas_nobg \
#     --spec-transform "PCEN" --deltas --no-background

# # 62. PCEN + Asymmetric Loss
# run_experiment regnet doc "$DOC_TRAIN" pcen_asl \
#     --spec-transform "PCEN" --use-asl

# # ─── Group F: Loss function variants on Log baseline (63–71) ──────────

# # 63. Asymmetric Loss only
# run_experiment regnet doc "$DOC_TRAIN" asl \
#     --spec-transform "Log" --use-asl

# # 64. Class weights only
# run_experiment regnet doc "$DOC_TRAIN" classw \
#     --spec-transform "Log" --class-weights

# # 65. ASL + class weights
# run_experiment regnet doc "$DOC_TRAIN" asl_classw \
#     --spec-transform "Log" --use-asl --class-weights

# # 66. Deltas + ASL
# run_experiment regnet doc "$DOC_TRAIN" deltas_asl \
#     --spec-transform "Log" --deltas --use-asl

# # 67. Deltas + class weights
# run_experiment regnet doc "$DOC_TRAIN" deltas_classw \
#     --spec-transform "Log" --deltas --class-weights

# # 68. SED head + ASL
# run_experiment regnet doc "$DOC_TRAIN" sedhead_asl \
#     --spec-transform "Log" --sed-head --use-asl

# # 69. SED head + class weights
# run_experiment regnet doc "$DOC_TRAIN" sedhead_classw \
#     --spec-transform "Log" --sed-head --class-weights

# # 70. Gated head alone
# run_experiment regnet doc "$DOC_TRAIN" gated \
#     --spec-transform "Log" --gated-head

# # 71. Gated head + ASL
# run_experiment regnet doc "$DOC_TRAIN" gated_asl \
#     --spec-transform "Log" --gated-head --use-asl

# # ─── Group G: CNN adapter (72–76) ─────────────────────────────────────

# 72. CNN adapter alone
run_experiment regnet doc "$DOC_TRAIN" cnnadapter \
    --spec-transform "Log" --cnn-adapter

# 73. CNN adapter + delta channels
run_experiment regnet doc "$DOC_TRAIN" cnnadapter_deltas \
    --spec-transform "Log" --cnn-adapter --deltas

# 74. CNN adapter + no background
run_experiment regnet doc "$DOC_TRAIN" cnnadapter_nobg \
    --spec-transform "Log" --cnn-adapter --no-background

# 75. CNN adapter + ASL
run_experiment regnet doc "$DOC_TRAIN" cnnadapter_asl \
    --spec-transform "Log" --cnn-adapter --use-asl

# 76. CNN adapter + bgsub + deltas
run_experiment regnet doc "$DOC_TRAIN" cnnadapter_bgsub_deltas \
    --spec-transform "Log" --cnn-adapter --bg-subtract --deltas

# ─── Group H: Additional k-bird prior values (77–80) ──────────────────

# 77. k-bird prior 1  (very light constraint)
run_experiment regnet doc "$DOC_TRAIN" kbird1 \
    --spec-transform "Log" --kbird-prior 1.0

# 78. k-bird prior 6
run_experiment regnet doc "$DOC_TRAIN" kbird6 \
    --spec-transform "Log" --kbird-prior 6.0

# 79. k-bird prior 8
run_experiment regnet doc "$DOC_TRAIN" kbird8 \
    --spec-transform "Log" --kbird-prior 8.0

# 80. bgsub + k-bird prior 1
run_experiment regnet doc "$DOC_TRAIN" bgsub_kbird1 \
    --spec-transform "Log" --bg-subtract --kbird-prior 1.0

# ─── Group I: Freeze-stages variants (81–83) ──────────────────────────

# 81. Freeze first backbone stage
run_experiment regnet doc "$DOC_TRAIN" freeze1 \
    --spec-transform "Log" --freeze-stages 1

# 82. Freeze first two backbone stages
run_experiment regnet doc "$DOC_TRAIN" freeze2 \
    --spec-transform "Log" --freeze-stages 2

# 83. bgsub + freeze first backbone stage
run_experiment regnet doc "$DOC_TRAIN" bgsub_freeze1 \
    --spec-transform "Log" --bg-subtract --freeze-stages 1

# ─── Group K: Large multi-flag combinations (84–100) ──────────────────

# 84. bgsub + SED head + ASL
run_experiment regnet doc "$DOC_TRAIN" bgsub_sedhead_asl \
    --spec-transform "Log" --bg-subtract --sed-head --use-asl

# 85. bgsub + gated head + ASL
run_experiment regnet doc "$DOC_TRAIN" bgsub_gated_asl \
    --spec-transform "Log" --bg-subtract --gated-head --use-asl

# 86. bgmed + SED head + ASL
run_experiment regnet doc "$DOC_TRAIN" bgmed_sedhead_asl \
    --spec-transform "Log" --bg-subtract --median-filter --sed-head --use-asl

# 87. bgmed + gated head + ASL
run_experiment regnet doc "$DOC_TRAIN" bgmed_gated_asl \
    --spec-transform "Log" --bg-subtract --median-filter --gated-head --use-asl

# 88. bgsub + deltas + no background + ASL
run_experiment regnet doc "$DOC_TRAIN" bgsub_deltas_nobg_asl \
    --spec-transform "Log" --bg-subtract --deltas --no-background --use-asl

# 89. LogMinMax + deltas + no background + SED head  ← "full Kaytoo flags"
run_experiment regnet doc "$DOC_TRAIN" logminmax_deltas_nobg_sedhead \
    --spec-transform "LogMinMax" --deltas --no-background --sed-head

# 90. bgsub + deltas + no background + SED head
run_experiment regnet doc "$DOC_TRAIN" bgsub_deltas_nobg_sedhead \
    --spec-transform "Log" --bg-subtract --deltas --no-background --sed-head

# 91. bgmed + deltas + no background + SED head
run_experiment regnet doc "$DOC_TRAIN" bgmed_deltas_nobg_sedhead \
    --spec-transform "Log" --bg-subtract --median-filter --deltas --no-background --sed-head

# 92. bgsub + no background + ASL + class weights
run_experiment regnet doc "$DOC_TRAIN" bgsub_nobg_asl_classw \
    --spec-transform "Log" --bg-subtract --no-background --use-asl --class-weights

# 93. bgmed + no background + ASL
run_experiment regnet doc "$DOC_TRAIN" bgmed_nobg_asl \
    --spec-transform "Log" --bg-subtract --median-filter --no-background --use-asl

# 94. bgmed + no background + deltas + gated head
run_experiment regnet doc "$DOC_TRAIN" bgmed_nobg_deltas_gated \
    --spec-transform "Log" --bg-subtract --median-filter --no-background --deltas --gated-head

# 95. Box-Cox + delta channels
run_experiment regnet doc "$DOC_TRAIN" boxcox_deltas \
    --spec-transform "Box-Cox" --deltas

# 96. Box-Cox + background subtraction
run_experiment regnet doc "$DOC_TRAIN" boxcox_bgsub \
    --spec-transform "Box-Cox" --bg-subtract

# 97. Box-Cox + no background
run_experiment regnet doc "$DOC_TRAIN" boxcox_nobg \
    --spec-transform "Box-Cox" --no-background

# 98. bgsub + deltas + no background + gated head
run_experiment regnet doc "$DOC_TRAIN" bgsub_deltas_nobg_gated \
    --spec-transform "Log" --bg-subtract --deltas --no-background --gated-head

# 99. No background + deltas + ASL + class weights
run_experiment regnet doc "$DOC_TRAIN" nobg_deltas_asl_classw \
    --spec-transform "Log" --no-background --deltas --use-asl --class-weights

# 100. bgmed + deltas + no background + gated head
run_experiment regnet doc "$DOC_TRAIN" bgmed_deltas_nobg_gated \
    --spec-transform "Log" --bg-subtract --median-filter --deltas --no-background --gated-head

echo ""
echo "============================================================"
echo " All matched experiments complete."
echo " Results: $OUTPUT"
echo ""
echo " Next steps:"
echo "   ./run_kaytoo_eval.sh   — Kaytoo baseline on matched test sets"
echo "   ./run_birdnet_eval.sh  — BirdNET baseline on matched test sets"
echo "============================================================"
