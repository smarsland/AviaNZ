#!/bin/bash
set -euo pipefail

# Build domain classification datasets (AviaNZ vs DOC labels) from matched splits.
# Run after build_dataset.sh (requires avianz_split/ and doc_split/ to exist).
#
# Usage:
#   ./build_domain_dataset.sh
#   ./build_domain_dataset.sh --overwrite

BASE="/local/scratch/freangi"
MATCHED="${BASE}/matched"
DOMAIN="${BASE}/domain"

if [ ! -d "${MATCHED}/avianz_split/train" ] || [ ! -d "${MATCHED}/doc_split/train" ]; then
    echo "ERROR: Matched splits not found at ${MATCHED}. Run build_dataset.sh first."
    exit 1
fi

echo "============================================================"
echo " Build domain classification datasets"
echo "============================================================"
echo "  Matched base : $MATCHED"
echo "  Output       : $DOMAIN"
echo "============================================================"
echo ""

PYTHONPATH="$PWD" python3 src/experiments/build_domain_dataset.py \
    "$MATCHED" \
    "$DOMAIN" \
    "$@"
