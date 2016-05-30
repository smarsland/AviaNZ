# Version 0.1 30/5/16
# Author: Stephen Marsland

import numpy as np
import pywt
from scipy.io import wavfile
import pylab as pl
import matplotlib
import librosa

# TODO:
# Put some more stuff in here and use it!

class Features:
    # This class implements various feature extraction algorithms for the AviaNZ interface
    # Currently it's just MFCC. Has DTW in too.
    # TODO: test what there is so far!

    def __init__(self,data=[],sampleRate=0):
        self.data = data
        self.sampleRate = sampleRate

    def dtw(self,x,y,wantDistMatrix=False):
        # Compute the dynamic time warp between two 1D arrays
        # I've taught it to second years, should be easy!
        dist = np.zeros((len(x)+1,len(y)+1))
        dist[1:,:] = np.inf
        dist[:,1:] = np.inf
        for i in range(len(x)):
            for j in range(len(y)):
                dist[i+1,j+1] = np.abs(x[i]-y[j]) + min(dist[i,j+1],dist[i+1,j],dist[i,j])
        if wantDistMatrix:
            return dist
        else:
            return dist[-1,-1]

    def dtw_path(self,d):
        # Shortest path through DTW matrix
        i = np.shape(d)[0]-2
        j = np.shape(d)[1]-2
        xpath = [i]
        ypath = [j]
        while i>0 or j>0:
                next = np.argmin((d[i,j],d[i+1,j],d[i,j+1]))
                if next == 0:
                    i -= 1
                    j -= 1
                elif next == 1:
                    j -= 1
                else:
                    i -= 1
                xpath.insert(0,i)
                ypath.insert(0,j)
        return xpath, ypath

    def get_mfcc(self):
        # Use a library to get the MFCC coefficients
        mfcc = librosa.feature.mfcc(data, sampleRate)
        librosa.display.specshow(mfcc)

        # Normalise
        mfcc -= np.mean(mfcc,axis=0)
        mfcc /= np.max(np.abs(mfcc),axis=0)

        return mfcc

    def testDTW(self):
        x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
        y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]

        d = self.dtw(x,y,wantDistMatrix=True)
        print self.dtw_path(d)

def test():
    a = Features()
    a.testDTW()
    pl.show()
test()