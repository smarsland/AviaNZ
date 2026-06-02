#!/usr/bin/env bash
# run_ast_scaling_finetune_experiment.sh
#
# Fine-tuning variant of the AST scaling experiment: for each N, take the
# AST model already trained on noisy DOC data at that N (from
# run_ast_scaling_experiment.sh) and fine-tune it on the corrected
# human-reviewed matched AviaNZ train set with the early transformer layers
# frozen — matching the approach used for RegNet in
# run_scaling_finetune_experiment.sh.
#
# For each N in N_VALUES:
#   - load  ${SCALING_TESTS}/ast_on_doc_scaling_N{N}_seed{SEED}/ast_model_best.pt
#   - fine-tune on matched AviaNZ train (corrected labels)
#   - freeze first 8 transformer encoder layers (out of 12), train the rest + head
#   - evaluate on matched AviaNZ test and matched DOC test
#   - results written to ${SCALING_TESTS}/ast_on_doc_scaling_ft_N{N}_seed{SEED}/
#
# Prerequisites:
#   bash run_ast_scaling_experiment.sh   # must have completed for all N values
#
# Usage:
#   bash run_ast_scaling_finetune_experiment.sh
#   SEED=1 bash run_ast_scaling_finetune_experiment.sh
#   N_VALUES="1000 2000 4000 8000" bash run_ast_scaling_finetune_experiment.sh

set -euo pipefail

BASE="/local/scratch/freangi"
MATCHED="${BASE}/matched"
MATCHED_MERGED_TRAIN="${MATCHED}/merged_train"
MATCHED_AVIANZ_TEST="${MATCHED}/avianz_split/test"
MATCHED_DOC_TEST="${MATCHED}/doc_split/test"
SCALING_TESTS="${BASE}/scaling_tests"
SEED="${SEED:-0}"

N_VALUES=(${N_VALUES:-8000})

# --- sanity checks ---
check_path() { [[ -d "$1" ]] || { echo "ERROR: directory not found: $1"; exit 1; }; }
check_path "${MATCHED_MERGED_TRAIN}"
check_path "${MATCHED_AVIANZ_TEST}"
check_path "${MATCHED_DOC_TEST}"

mkdir -p "${SCALING_TESTS}"

echo "================================================================"
echo " AST scaling fine-tune: DOC-trained → corrected labels"
echo "  Fine-tune train : ${MATCHED_MERGED_TRAIN}"
echo "  AviaNZ test     : ${MATCHED_AVIANZ_TEST}"
echo "  DOC test        : ${MATCHED_DOC_TEST}"
echo "  Output root     : ${SCALING_TESTS}"
echo "  Seed            : ${SEED}"
echo "  N values        : ${N_VALUES[*]}"
echo "  Mode            : load DOC-trained AST, freeze first 8 layers"
echo "================================================================"

for N in "${N_VALUES[@]}"; do
    DOC_RUN="${SCALING_TESTS}/ast_on_doc_scaling_N${N}_seed${SEED}"
    CKPT="${DOC_RUN}/ast_model_best.pt"
    OUT="${SCALING_TESTS}/ast_on_doc_scaling_ft_N${N}_seed${SEED}"

    echo ""
    echo "------------------------------------------------------------"
    echo " N = ${N}  checkpoint: ${CKPT}"
    echo "          →  ${OUT}"
    echo "------------------------------------------------------------"

    if [[ ! -f "${CKPT}" ]]; then
        echo " SKIP: checkpoint not found (run_ast_scaling_experiment.sh not finished for N=${N})"
        continue
    fi

    PYTHONPATH=. python3 train.py \
        "${MATCHED_MERGED_TRAIN}" \
        "${OUT}" \
        --model-type   ast \
        --pretrained   "${CKPT}" \
        --spec-transform Log \
        --bg-subtract \
        --kbird-prior  2.0 \
        --freeze-layers 8 \
        --lr           1e-4 \
        --epochs       20 \
        --patience     10 \
        --seed         "${SEED}" \
        --test-folder  "${MATCHED_AVIANZ_TEST}" \
        --test-folder2 "${MATCHED_DOC_TEST}"

    echo " Finished N=${N}"
done

echo ""
echo "================================================================"
echo " All AST scaling fine-tune runs complete."
echo " Results in: ${SCALING_TESTS}/"
echo "================================================================"
