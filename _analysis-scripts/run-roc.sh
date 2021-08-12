#!/bin/bash
set -e

# ------------------------------------------------
# Run AviaNZ with the proposed CD+MFCC detector
# on full DCASE evaluation data
# over a range of thresholds to generate
# the precision-recall ROC
# ------------------------------------------------

ROOTDIR=~/Documents/audiodata/DCASE/ #train-cat/

AUDIODIR="$ROOTDIR/eval/"  # dir with audio files
REFFILE="filelists/eval.csv" # reference csv

ESTFILE="$ROOTDIR/output-dcase-tmp.csv"  # csv with estimated annotations in DCASE format
TMPSCOREFILE="$ROOTDIR/eval-CD-MFCC-tmp.log"  # final output w/ F-scores

FILTER="Cats_CD_MFCC"  # filter name, without .txt

ROCFILE="$ROOTDIR/rocdata-CD-MFCC-mean.csv"  # values for each threshold for plotting ROC
# To obtain the results for the quantile decision rule, uncomment the
# corresponding line (1299) in ../Segment.py

echo "Thr F1 Precision Recall" > $ROCFILE
for THR in -5 -2 -1 -0.5 0 0.5 1 1.5 2 3 5
do
    echo " ---------- Running with threshold $THR -------------"
    echo "Running AviaNZ detector ----------"
    cd ..;  python AviaNZ.py -c -b -d $AUDIODIR -r $FILTER -x $THR
    cd _analysis-scripts/
    
    echo "Converting annotations -----------"
    python convert-avianz-annots.py $AUDIODIR $ESTFILE
    
    echo "Calculating F-scores -------------"
    python evaluation_measures.py $REFFILE $ESTFILE > $TMPSCOREFILE
    
    tail $TMPSCOREFILE | awk -v THR=${THR} '$1~/Cat/{print THR, $6, $7, $8; exit}' >> $ROCFILE
    tail $TMPSCOREFILE
    echo " ----------------- Threshold $THR done -------------"
done

