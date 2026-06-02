#!/usr/bin/env bash
# run_scaling_finetune_experiment.sh
#
# Fine-tuning variant of the data-scaling experiment: for each N, take the
# model already trained on DOC noisy data at that N (from
# run_scaling_experiment.sh) and fine-tune it on the corrected human-reviewed
# matched AviaNZ train set with a frozen backbone — matching the Kaytoo
# fine-tuning approach for a fair comparison.
#
# For each N in N_VALUES:
#   - load  ${SCALING_TESTS}/regnet_on_doc_scaling_kbird2_bgsubtract_N{N}_seed{SEED}/regnet_model_best.pt
#   - fine-tune on matched AviaNZ train (corrected labels), backbone frozen
#   - evaluate on matched AviaNZ test and matched DOC test
#   - results written to ${SCALING_TESTS}/regnet_on_doc_scaling_kbird2_bgsubtract_ft_N{N}_seed{SEED}/
#
# Prerequisites:
#   bash run_scaling_experiment.sh   # must have completed for all N values
#
# Usage:
#   bash run_scaling_finetune_experiment.sh
#   SEED=1 bash run_scaling_finetune_experiment.sh   # different seed

set -euo pipefail

BASE="/local/scratch/freangi"
MATCHED="${BASE}/matched"
MATCHED_MERGED_TRAIN="${MATCHED}/merged_train"
MATCHED_AVIANZ_TEST="${MATCHED}/avianz_split/test"
MATCHED_DOC_TEST="${MATCHED}/doc_split/test"
SCALING_TESTS="${BASE}/scaling_tests"
SEED="${SEED:-0}"

N_VALUES=(100 200 300 500 750 1000 1500 2000 2500 3000 4000 5000 6000 7000 8000)

# --- sanity checks ---
check_path() { [[ -d "$1" ]] || { echo "ERROR: directory not found: $1"; exit 1; }; }
check_path "${MATCHED_MERGED_TRAIN}"
check_path "${MATCHED_AVIANZ_TEST}"
check_path "${MATCHED_DOC_TEST}"

mkdir -p "${SCALING_TESTS}"

echo "================================================================"
echo " Scaling fine-tune experiment: DOC-trained → corrected labels"
echo "  Fine-tune train : ${MATCHED_MERGED_TRAIN}"
echo "  AviaNZ test     : ${MATCHED_AVIANZ_TEST}"
echo "  DOC test        : ${MATCHED_DOC_TEST}"
echo "  Output root     : ${SCALING_TESTS}"
echo "  Seed            : ${SEED}"
echo "  N values        : ${N_VALUES[*]}"
echo "  Mode            : load DOC-trained checkpoint, freeze backbone"
echo "================================================================"

for N in "${N_VALUES[@]}"; do
    DOC_RUN="${SCALING_TESTS}/regnet_on_doc_scaling_kbird2_bgsubtract_N${N}_seed${SEED}"
    CKPT="${DOC_RUN}/regnet_model_best.pt"
    OUT="${SCALING_TESTS}/regnet_on_doc_scaling_kbird2_bgsubtract_ft_N${N}_seed${SEED}"

    echo ""
    echo "------------------------------------------------------------"
    echo " N = ${N}  checkpoint: ${CKPT}"
    echo "          →  ${OUT}"
    echo "------------------------------------------------------------"

    if [[ ! -f "${CKPT}" ]]; then
        echo " SKIP: checkpoint not found (run_scaling_experiment.sh not finished for N=${N})"
        continue
    fi

    PYTHONPATH=. python3 train.py \
        "${MATCHED_MERGED_TRAIN}" \
        "${OUT}" \
        --model-type  regnet \
        --pretrained  "${CKPT}" \
        --spec-transform Log \
        --bg-subtract \
        --kbird-prior 2.0 \
        --freeze-backbone \
        --lr          1e-4 \
        --epochs      20 \
        --patience    10 \
        --seed        "${SEED}" \
        --test-folder  "${MATCHED_AVIANZ_TEST}" \
        --test-folder2 "${MATCHED_DOC_TEST}"

    echo " Finished N=${N}"
done

echo ""
echo "================================================================"
echo " All scaling fine-tune runs complete."
echo " Results in: ${SCALING_TESTS}/"
echo "================================================================"
