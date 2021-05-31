## Export ground truth and features for training a classifier

## USAGE: python train-mfcc.py INDIR OUTDIR
## where INDIR stores .wav files with .data annotations
## and OUTDIR will store exported files.
import sys
import os
sys.path.append('..')

import math
import librosa
import numpy as np
import Segment
import SignalProc

INDIR = sys.argv[1]  # "~/Documents/audiodata/DCASE/train-wf/" etc
OUTDIR = sys.argv[2] # "~/Documents/audiodata/DCASE/tmp/"

segments = Segment.SegmentList()
sp = SignalProc.SignalProc()

def generateMFCC(data, fs):
    """ Take data and extract MFCCs """
    print("Extracting MFCCs...")
    N_MFCC = 24
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=np.asarray(data), sr=fs, n_mfcc=N_MFCC)
    # (results in a matrix of N_MFCC x timewindows

    # extract ZCR
    zcr = librosa.feature.zero_crossing_rate(y=np.asarray(data))
    # drop the first MFCC and attach ZCR
    mfcc[0,:] = zcr

    return mfcc

for root, dirs, files in os.walk(INDIR):
    for filename in files:
        print("Working on", filename)
        if filename.endswith(".wav"):
            # load segments from the JSON
            segments.parseJSON(os.path.join(root, filename)+".data")
            # load audiodata from the wav
            sp.readWav(os.path.join(root,filename))
            if sp.sampleRate != 16000:
                sp.resample(16000)
            # extract mfccs
            mfcc = generateMFCC(sp.data, sp.sampleRate)
            print("extracted features", np.shape(mfcc))
            print(mfcc)
            # define which columns are 1/0
            win_len = segments.metadata["Duration"]/np.shape(mfcc)[1]
            gt = np.zeros((math.ceil(segments.metadata["Duration"]/win_len)))
            for seg in segments:
                gt[math.floor(seg[0]/win_len):math.ceil(seg[1]/win_len)] = 1
            print("Ground truth", gt)
            # Save the extracted info for training a  classifier
            np.savetxt(OUTDIR+filename+".txt", gt)
            np.savetxt(OUTDIR+filename+".mfcc", mfcc)

