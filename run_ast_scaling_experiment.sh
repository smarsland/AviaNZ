#!/usr/bin/env bash
# run_ast_scaling_experiment.sh
#
# Train an AST model on the large DOC scaling dataset (N=8000/class) and
# evaluate on the matched human-reviewed test sets.
#
# Mirrors the regnet scaling experiment but uses --model-type ast.
# The N=8000 value requires the subsample-before-split fix in model_trainer.py;
# without it N≥7000 silently trains on the same 6400-sample subset.
#
# Prerequisites:
#   bash build_scaling_dataset.sh   # builds ${SCALING}/doc_large at 8000/class
#
# Usage:
#   bash run_ast_scaling_experiment.sh
#   SEED=1 bash run_ast_scaling_experiment.sh
#   N=4000 bash run_ast_scaling_experiment.sh   # single N value

set -euo pipefail

BASE="/local/scratch/freangi"
SCALING="${BASE}/scaling"
SCALING_TRAIN="${SCALING}/doc_large"
MATCHED="${BASE}/matched"
MATCHED_AVIANZ_TEST="${MATCHED}/avianz_split/test"
MATCHED_DOC_TEST="${MATCHED}/doc_split/test"
SCALING_TESTS="${BASE}/scaling_tests"
SEED="${SEED:-0}"

# Default: single run at full data budget.
# Set N_VALUES externally to run a sweep, e.g.:
#   N_VALUES="100 500 1000 2000 4000 8000" bash run_ast_scaling_experiment.sh
N_VALUES=(${N_VALUES:-8000})

# --- sanity checks ---
check_path() { [[ -d "$1" ]] || { echo "ERROR: directory not found: $1"; exit 1; }; }
check_path "${SCALING_TRAIN}"
check_path "${MATCHED_AVIANZ_TEST}"
check_path "${MATCHED_DOC_TEST}"

mkdir -p "${SCALING_TESTS}"

echo "================================================================"
echo " AST scaling experiment: DOC noisy labels → matched test"
echo "  Train data    : ${SCALING_TRAIN}"
echo "  AviaNZ test   : ${MATCHED_AVIANZ_TEST}"
echo "  DOC test      : ${MATCHED_DOC_TEST}"
echo "  Output root   : ${SCALING_TESTS}"
echo "  Seed          : ${SEED}"
echo "  N values      : ${N_VALUES[*]}"
echo "================================================================"

for N in "${N_VALUES[@]}"; do
    OUT="${SCALING_TESTS}/ast_on_doc_scaling_N${N}_seed${SEED}"
    echo ""
    echo "------------------------------------------------------------"
    echo " N = ${N}  →  ${OUT}"
    echo "------------------------------------------------------------"

    PYTHONPATH=. python3 train.py \
        "${SCALING_TRAIN}" \
        "${OUT}" \
        --model-type   ast \
        --spec-transform Log \
        --bg-subtract \
        --kbird-prior  2.0 \
        --epochs       40 \
        --patience     15 \
        --seed         "${SEED}" \
        --max-samples-per-class "${N}" \
        --test-folder  "${MATCHED_AVIANZ_TEST}" \
        --test-folder2 "${MATCHED_DOC_TEST}"

    echo " Finished N=${N}"
done

echo ""
echo "================================================================"
echo " All AST scaling runs complete."
echo " Results in: ${SCALING_TESTS}/"
echo "================================================================"
