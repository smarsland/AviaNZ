# Version 0.1 30/5/16
# Author: Stephen Marsland

import numpy as np
import pywt
from scipy.io import wavfile
import pylab as pl
import matplotlib

# TODO:
# Put some stuff in here!
# Should at least do power, wavelets, then think about more
# Add Nirosha's approach of simultaneous segmentation and recognition using wavelets
# Try onset_detect from librosa
# Want to take each second or so and say yes or no for presence
# Should compute SNR
# Use spectrogram instead/as well

class Segment:
    # This class implements various signal processing algorithms for the AviaNZ interface
    def __init__(self,data):
        self.data = data
        # This is the length of a window to average to get the power
        self.length = 100
        self.segments = []

    def segmentByAmplitude(self,threshold):
        self.seg = np.where(self.data>threshold,1,0)
        inSegment=False
        for i in range(len(self.data)):
            if self.seg[i] > 0:
                if inSegment:
                    pass
                else:
                    inSegment = True
                    start = i
            else:
                if inSegment:
                    self.segments.append([start, i])
                    inSegment = False
        return self.segments

    def segmentByWavelet(self,threshold):
        # Need to think about this. Basically should play with it (without the interface) and do some computations
        # and plot the wavelet packets
        pass

    def SnNR(self,startSignal,startNoise):
        pS = np.sum(self.data[startSignal:startSignal+self.length]**2)/self.length
        pN = np.sum(self.data[startNoise:startNoise+self.length]**2)/self.length
        return 10.*np.log10(pS/pN)