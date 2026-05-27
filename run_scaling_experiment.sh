#!/usr/bin/env bash
# run_scaling_experiment.sh
#
# Data-scaling experiment: train RegNet on the large noisy DOC dataset
# (built by build_scaling_dataset.sh) at increasing sample budgets, and
# evaluate on the matched human-reviewed test sets.
#
# For each N in {100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000}:
#   - train on doc_large (3000/class) subsampled to N/class at train time
#   - evaluate on matched AviaNZ test and matched DOC test
#   - results written to ${SCALING_TESTS}/regnet_on_doc_scaling_N{N}/
#
# Prerequisites:
#   bash build_scaling_dataset.sh   # builds ${SCALING}/doc_large at 3000/class
#
# Usage:
#   bash run_scaling_experiment.sh
#   SEED=1 bash run_scaling_experiment.sh   # different seed

set -euo pipefail

BASE="/local/scratch/freangi"
SCALING="${BASE}/scaling"
SCALING_TRAIN="${SCALING}/doc_large"
MATCHED="${BASE}/matched"
MATCHED_AVIANZ_TEST="${MATCHED}/avianz_split/test"
MATCHED_DOC_TEST="${MATCHED}/doc_split/test"
SCALING_TESTS="${BASE}/scaling_tests"
PRETRAINED="BirdClefModels/model_fold0.pth"
SEED="${SEED:-0}"

N_VALUES=(100 200 300 500 750 1000 1500 2000 2500 3000)

# --- sanity checks ---
check_path() { [[ -d "$1" ]] || { echo "ERROR: directory not found: $1"; exit 1; }; }
check_path "${SCALING_TRAIN}"
check_path "${MATCHED_AVIANZ_TEST}"
check_path "${MATCHED_DOC_TEST}"
[[ -f "${PRETRAINED}" ]] || { echo "ERROR: pretrained weights not found: ${PRETRAINED}"; exit 1; }

mkdir -p "${SCALING_TESTS}"

echo "================================================================"
echo " Scaling experiment: DOC noisy labels → matched test"
echo "  Train data    : ${SCALING_TRAIN}"
echo "  AviaNZ test   : ${MATCHED_AVIANZ_TEST}"
echo "  DOC test      : ${MATCHED_DOC_TEST}"
echo "  Output root   : ${SCALING_TESTS}"
echo "  Seed          : ${SEED}"
echo "  N values      : ${N_VALUES[*]}"
echo "================================================================"

for N in "${N_VALUES[@]}"; do
    OUT="${SCALING_TESTS}/regnet_on_doc_scaling_N${N}_seed${SEED}"
    echo ""
    echo "------------------------------------------------------------"
    echo " N = ${N}  →  ${OUT}"
    echo "------------------------------------------------------------"

    PYTHONPATH=. python3 train.py \
        "${SCALING_TRAIN}" \
        "${OUT}" \
        --model-type  regnet \
        --pretrained  "${PRETRAINED}" \
        --spec-transform Log \
        --bg-subtract \
        --epochs      40 \
        --patience    15 \
        --seed        "${SEED}" \
        --max-samples-per-class "${N}" \
        --test-folder  "${MATCHED_AVIANZ_TEST}" \
        --test-folder2 "${MATCHED_DOC_TEST}"

    echo " Finished N=${N}"
done

echo ""
echo "================================================================"
echo " All scaling runs complete."
echo " Results in: ${SCALING_TESTS}/"
echo "================================================================"
