#!/usr/bin/env bash
# build_avianz_all_species_dataset.sh
#
# Build an AviaNZ-only dataset from ALL annotated AviaNZ folders on the
# network drive, excluding:
#   - Joe_MoDone?   (already used as the matched-test AviaNZ source)
#   - NZBirds       (DOC audio, not AviaNZ annotated)
#   - BatTraining / BatsEglington2019 / SOIK_FiordlandBats  (bats / no data)
#
# Uses up to MAX_PER_SPECIES (2000) segments per species so the class
# distribution stays balanced.  Spectrogram settings match the all-species
# DOC model (Standard / Hamming / Mel Frequency) for direct comparison.
#
# Output: ${OUTPUT}/avianz_large/   (labels.json + data/)
# The Trainer does its own 80/20 train/val split internally.
#
# Usage:
#   bash build_avianz_all_species_dataset.sh
#   bash build_avianz_all_species_dataset.sh --overwrite

set -euo pipefail

BASE="/local/scratch/freangi"
DRIVE="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02"
OUTPUT="${BASE}/avianz_all_species"
MAX_PER_SPECIES=2000
MAPPING="data/DOC_bird_naming_map.csv"
OVERWRITE_FLAG=""

if [[ "${1:-}" == "--overwrite" ]]; then
    OVERWRITE_FLAG="--overwrite"
fi

# All annotated AviaNZ folders, excluding Joe_MoDone? and non-bird sources.
AVIANZ_FOLDERS=(
    "${DRIVE}/Remutaka20"
    "${DRIVE}/Bittern_Harry"
    "${DRIVE}/SOIK_FiordlandTokoeka"
    "${DRIVE}/Remutaka21"
    "${DRIVE}/RakiuraTokoeka"
    "${DRIVE}/MA_PILOT_2018_ARS"
    "${DRIVE}/Datasets_NP"
    "${DRIVE}/Rurudone"
    "${DRIVE}/Waiheke"
    "${DRIVE}/GSK"
    "${DRIVE}/Ruru_Tinakari1"
    "${DRIVE}/Nirosha"
    "${DRIVE}/Kakapo Recordings Andrew Digby"
    "${DRIVE}/Morepork Annotations"
    "${DRIVE}/AR4 kokako survey"
    "${DRIVE}/2017_Rawhiti playback experiment"
)

# Build --avianz-raw flags for each folder.
AVIANZ_ARGS=()
for FOLDER in "${AVIANZ_FOLDERS[@]}"; do
    AVIANZ_ARGS+=( "--avianz-raw" "${FOLDER}" )
done

echo "============================================================"
echo " Build all-AviaNZ-species dataset (${MAX_PER_SPECIES}/class max)"
echo "  Folders : ${#AVIANZ_FOLDERS[@]}"
echo "  Output  : ${OUTPUT}/avianz_large"
echo "  Mapping : ${MAPPING}"
echo "  Overwrite: ${OVERWRITE_FLAG:-no}"
echo "============================================================"
for F in "${AVIANZ_FOLDERS[@]}"; do echo "    $F"; done
echo ""

mkdir -p "${OUTPUT}"

PYTHONPATH=. python3 src/experiments/build_large_datasets.py \
    --avianz-only \
    "${AVIANZ_ARGS[@]}" \
    --output           "${OUTPUT}" \
    --mapping          "${MAPPING}" \
    --max-per-species  "${MAX_PER_SPECIES}" \
    --no-audio \
    --spec-type        Standard \
    --window-type      Hamming \
    --sg-scale         "Mel Frequency" \
    ${OVERWRITE_FLAG}

echo ""
echo "Done. Training data at: ${OUTPUT}/avianz_large"
echo "Run: bash run_avianz_all_species_experiment.sh"
