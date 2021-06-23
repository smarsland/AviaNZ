#!/bin/bash
set -e

# ----------------------------
# Download, prepare data and train models
# ---------------------------

# in this directory, subdirs will be created for several downloaded datasets
# and all results & temp files will be stored here
ROOTDIR=~/Documents/audiodata/DCASE/train-cat/

# Download data
echo "Downloading data..."
python download_data.py $ROOTDIR

# Concatenate the WF training files into one
# (due to AviaNZ training specifics)
echo "Post-processing files..."
bash sox-combine.sh $ROOTDIR

# Here, the training & parameter tuning would happen.
# All four dirs (test/, train-onefile/, -wf/, and -neg/) must be annotated.
# Our annotations are provided in annotations/ directory and can be loaded with:
# cp -r annotations/* $ROOTDIR

# 1) Train the wavelet recognizer using AviaNZ GUI, on the $ROOTDIR/train-onefile/ directory.
# 2) Train the median clipping recognizer using AviaNZ GUI, same nodes and directory.
# 3) Tune the thresholds of both using AviaNZ "Test recogniser" function
#    on the $ROOTDIR/test/ directory, to around 90% recall.

if [ ! -f "$ROOTDIR/train-onefile/long.wav.data" ]; then
	echo "You must now annotate the files using AviaNZ."
	exit
fi

# extract features and ground truth for training a classifier
echo "Extracting features..."
mkdir -p "$ROOTDIR/tmp/"
python extract-mfcc.py "$ROOTDIR/train-wf/" "$ROOTDIR/tmp/"
python extract-mfcc.py "$ROOTDIR/test/" "$ROOTDIR/tmp/"
python extract-mfcc.py "$ROOTDIR/train-neg/" "$ROOTDIR/tmp/"

# load these into R to train a classifier
echo "Training..."
Rscript train-mfcc.R
# (the classifier will be stored in ~/.avianz/)

# proceed to run the evaluation:
# (change the ROOTDIR in that script as needed)
bash run-eval.sh
