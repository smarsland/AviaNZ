#!/bin/bash

## USAGE: bash sox-combine.sh AUDIODIR
## will process some files in AUDIODIR/train-wf/
## to create AUDIODIR/train-onefile/long.wav

echo "Combining training files in subdirs of $1"
cd $1

mkdir -p train-onefile

# these files had one channel and need to be converted to allow concat
sox -V3 train-wf/Y0DtJdRFPmS4_30.000_40.000.wav train-onefile/Y0Dtmp.wav channels 2
sox -V3 train-wf/Y0uab4-3d6MM_30.000_40.000.wav train-onefile/Y0uatmp.wav channels 2

# concat all files
sox -V3 train-wf/Y0E6Uaq_e6OA_20.000_30.000.wav \
	train-wf/Y0F40MJSfsDw_30.000_40.000.wav \
	train-onefile/Y0Dtmp.wav \
	train-wf/Y0632OqvXrwg_7.000_17.000.wav \
	train-onefile/Y0uatmp.wav \
	train-onefile/long.wav      # output
rm train-onefile/Y0Dtmp.wav
rm train-onefile/Y0uatmp.wav




