#!/bin/bash
set -e

echo "Re-evaluating AST experiments..."

cd /home/freangi/AviaNZ

# Experiment 1: avianz_ast_baseline_seed573
echo ""
echo "====================================================================="
echo "1/2: avianz_ast_baseline_seed573"
echo "====================================================================="
python3 reeval_ast.py \
    /local/scratch/freangi/experiments_matched/avianz_ast_baseline_seed573 \
    /local/scratch/freangi/data_matched/avianz_split/test \
    /local/scratch/freangi/data_matched/doc_split/test

# Experiment 2: doc_ast_baseline_seed573
echo ""
echo "====================================================================="
echo "2/2: doc_ast_baseline_seed573"
echo "====================================================================="
python3 reeval_ast.py \
    /local/scratch/freangi/experiments_matched/doc_ast_baseline_seed573 \
    /local/scratch/freangi/data_matched/doc_split/test \
    /local/scratch/freangi/data_matched/avianz_split/test

# Fix result.json files
echo ""
echo "====================================================================="
echo "Updating result.json files..."
echo "====================================================================="
python3 fix_ast_results.py /local/scratch/freangi/experiments_matched

echo ""
echo "✓✓✓ DONE ✓✓✓"
