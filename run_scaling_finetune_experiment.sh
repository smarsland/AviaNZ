#!/usr/bin/env bash
# run_scaling_finetune_experiment.sh
#
# Fine-tuning variant of the data-scaling experiment: same setup as
# run_scaling_experiment.sh but with --freeze-backbone, mirroring how
# the Kaytoo fine-tuned model is trained (only the classifier head is
# updated, backbone weights are frozen).  This gives a fairer like-for-like
# comparison between the two approaches.
#
# For each N in N_VALUES:
#   - train on doc_large subsampled to N/class at train time
#   - freeze backbone, train classifier head only
#   - evaluate on matched AviaNZ test and matched DOC test
#   - results written to ${SCALING_TESTS}/regnet_on_doc_scaling_kbird2_bgsubtract_ft_N{N}_seed{SEED}/
#
# Prerequisites:
#   bash build_scaling_dataset.sh   # builds ${SCALING}/doc_large at 2000/class
#
# Usage:
#   bash run_scaling_finetune_experiment.sh
#   SEED=1 bash run_scaling_finetune_experiment.sh   # different seed

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

N_VALUES=(100 200 300 500 750 1000 1500 2000 2500 3000 4000 5000 6000 7000 8000)

# --- sanity checks ---
check_path() { [[ -d "$1" ]] || { echo "ERROR: directory not found: $1"; exit 1; }; }
check_path "${SCALING_TRAIN}"
check_path "${MATCHED_AVIANZ_TEST}"
check_path "${MATCHED_DOC_TEST}"
[[ -f "${PRETRAINED}" ]] || { echo "ERROR: pretrained weights not found: ${PRETRAINED}"; exit 1; }

mkdir -p "${SCALING_TESTS}"

echo "================================================================"
echo " Scaling fine-tune experiment: DOC noisy labels → matched test"
echo "  Train data    : ${SCALING_TRAIN}"
echo "  AviaNZ test   : ${MATCHED_AVIANZ_TEST}"
echo "  DOC test      : ${MATCHED_DOC_TEST}"
echo "  Output root   : ${SCALING_TESTS}"
echo "  Seed          : ${SEED}"
echo "  N values      : ${N_VALUES[*]}"
echo "  Mode          : freeze-backbone (head only)"
echo "================================================================"

for N in "${N_VALUES[@]}"; do
    OUT="${SCALING_TESTS}/regnet_on_doc_scaling_kbird2_bgsubtract_ft_N${N}_seed${SEED}"
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
        --kbird-prior 2.0 \
        --freeze-backbone \
        --lr          1e-4 \
        --epochs      20 \
        --patience    10 \
        --seed        "${SEED}" \
        --max-samples-per-class "${N}" \
        --test-folder  "${MATCHED_AVIANZ_TEST}" \
        --test-folder2 "${MATCHED_DOC_TEST}"

    echo " Finished N=${N}"
done

echo ""
echo "================================================================"
echo " All scaling fine-tune runs complete."
echo " Results in: ${SCALING_TESTS}/"
echo "================================================================"
