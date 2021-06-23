#!/bin/bash
set -e

# ------------------------------------------------
# Run AviaNZ with trained filters and classifiers
# on full DCASE evaluation data
# ------------------------------------------------

ROOTDIR=~/Documents/audiodata/DCASE/train-cat/

AUDIODIR="$ROOTDIR/eval/"  # dir with audio files
REFFILE="filelists/eval.csv" # reference csv

ESTFILE="$ROOTDIR/output-dcase-tmp.csv"  # csv with estimated annotations in DCASE format


## ------------ REPEAT FOR EACH PRE-FILTER -----------
## FILTER 1
FILTER="Cats_WF"  # filter name, without .txt
OUTFILE="$ROOTDIR/eval-WFonly.log"  # final output w/ F-scores

echo "Running AviaNZ detector ----------"
cd ..; python AviaNZ.py -c -b -d $AUDIODIR -r $FILTER
cd _analysis-scripts/

echo "Converting annotations -----------"
python convert-avianz-annots.py $AUDIODIR $ESTFILE

echo "Calculating F-scores -------------"
python evaluation_measures.py $REFFILE $ESTFILE > $OUTFILE

tail $OUTFILE

## FILTER 2
FILTER="Cats_MC"  # filter name, without .txt
OUTFILE="$ROOTDIR/eval-MConly.log"  # final output w/ F-scores

echo "Running AviaNZ detector ----------"
cd ..; python AviaNZ.py -c -b -d $AUDIODIR -r $FILTER
cd _analysis-scripts/

echo "Converting annotations -----------"
python convert-avianz-annots.py $AUDIODIR $ESTFILE

echo "Calculating F-scores -------------"
python evaluation_measures.py $REFFILE $ESTFILE > $OUTFILE

tail $OUTFILE

## FILTER 3
FILTER="Cats_CD"  # filter name, without .txt
OUTFILE="$ROOTDIR/eval-CDonly.log"  # final output w/ F-scores

echo "Running AviaNZ detector ----------"
cd ..; python AviaNZ.py -c -b -d $AUDIODIR -r $FILTER
cd _analysis-scripts/

echo "Converting annotations -----------"
python convert-avianz-annots.py $AUDIODIR $ESTFILE

echo "Calculating F-scores -------------"
python evaluation_measures.py $REFFILE $ESTFILE > $OUTFILE

tail $OUTFILE

## ------------ REPEAT FOR EACH PRE-FILTER WITH MFCC CLASSIFIER -----------
## FILTER 1
FILTER="Cats_WF_MFCC"  # filter name, without .txt
OUTFILE="$ROOTDIR/eval-WF-mfcc.log"  # final output w/ F-scores

echo "Running AviaNZ detector ----------"
cd ..; python AviaNZ.py -c -b -d $AUDIODIR -r $FILTER
cd _analysis-scripts/

echo "Converting annotations -----------"
python convert-avianz-annots.py $AUDIODIR $ESTFILE

echo "Calculating F-scores -------------"
python evaluation_measures.py $REFFILE $ESTFILE > $OUTFILE

tail $OUTFILE

## FILTER 2
FILTER="Cats_MC_MFCC"  # filter name, without .txt
OUTFILE="$ROOTDIR/eval-MC-mfcc.log"  # final output w/ F-scores

echo "Running AviaNZ detector ----------"
cd ..; python AviaNZ.py -c -b -d $AUDIODIR -r $FILTER
cd _analysis-scripts/

echo "Converting annotations -----------"
python convert-avianz-annots.py $AUDIODIR $ESTFILE

echo "Calculating F-scores -------------"
python evaluation_measures.py $REFFILE $ESTFILE > $OUTFILE

tail $OUTFILE

## FILTER 3
FILTER="Cats_CD_MFCC"  # filter name, without .txt
OUTFILE="$ROOTDIR/eval-CD-mfcc.log"  # final output w/ F-scores

echo "Running AviaNZ detector ----------"
cd ..; python AviaNZ.py -c -b -d $AUDIODIR -r $FILTER
cd _analysis-scripts/

echo "Converting annotations -----------"
python convert-avianz-annots.py $AUDIODIR $ESTFILE

echo "Calculating F-scores -------------"
python evaluation_measures.py $REFFILE $ESTFILE > $OUTFILE

tail $OUTFILE

