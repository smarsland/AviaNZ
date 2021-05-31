set -e

# ----------------------------
# Download, prepare data and train models
# ---------------------------

# in this directory, subdirs will be created for several downloaded datasets
ROOTDIR=~/Documents/audiodata/DCASE/train-cat/clean/

# Download data
python download_data.py $ROOTDIR

# Concatenate the WF training files into one
# (due to AviaNZ training specifics)
bash sox-combine.sh $ROOTDIR

# Here, the training & parameter tuning would happen.
# All three dirs (train-onefile, -wf/, and -neg/) must be annotated.
if [ ! -f "$ROOTDIR/train-onefile/long.wav" ]; then
	echo "You must now annotate the files using AviaNZ."
	exit
fi

# extract features and ground truth for training a classifier
mkdir -p "$ROOTDIR/tmp/"
python extract-mfcc.py "$ROOTDIR/train-wf/" "$ROOTDIR/tmp/"
python extract-mfcc.py "$ROOTDIR/train-neg/" "$ROOTDIR/tmp/"
# Optional:
# extract MFCCs from the WF testing data to use for classifier testing
# python extract-mfcc.py "$ROOTDIR/test/" "$ROOTDIR/tmp/"

# load these into R to train a classifier
Rscript train-mfcc.R
# (the classifier will be stored in ~/.avianz/)
