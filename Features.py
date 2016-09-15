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
# So what are good features? MFCC is a start, what else? Wavelets?
# **FINISH!!*** List from Raven:
    #1st Quartile Frequency Max Power
    #1st Quartile Time Max Time
    #3rd Quartile Frequency Min Amplitude
    #3rd Quartile Time Min Time
    #Average Power Peak Amplitude
    #Center Frequency Peak Correlation
    #Center Time Peak Frequency
    #Energy Peak Lag
    #filtered RMS Amplitude Peak Power
    #Frequency 5% Peak Time
    #Frequency 95% RMS Amplitude
    #Max Amplitude Time 5%
    #Max Bearing Time 95%
    #Max Frequency

# Add chroma features and Tonnetz
# Vibrato
# Prosodic features (pitch, duration, intensity)
# Spectral statistics
# Frequency modulation
# Linear Predictive Coding? -> from scikits.talkbox import lpc (see also audiolazy)
# Frechet distance for DTW?
# Pick things from spectrogram
# Correlation

# Add something that plots some of these to help playing, so that I can understand the librosa things, etc.

# And assemble a decent dataset of birdcalls to play with.
# Or in fact, two: kiwi, ruru, bittern, and then a much bigger one

class Features:
    # This class implements various feature extraction algorithms for the AviaNZ interface
    # In essence, it will be given a segment as a region of audiodata (between start and stop points)
    # Classifiers will then be called on the features
    # Currently it's just MFCC. Has DTW in too.
    # TODO: test what there is so far!

    def __init__(self,data=[],sampleRate=0):
        self.data = data
        self.sampleRate = sampleRate

    def setNewData(self,data,sg,fs):
        self.data = data
        self.sg = sg
        self.fs = fs

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
        # Use librosa to get the MFCC coefficients
        mfcc = librosa.feature.mfcc(self.data, self.sampleRate)
        librosa.display.specshow(mfcc)

        # Normalise
        mfcc -= np.mean(mfcc,axis=0)
        mfcc /= np.max(np.abs(mfcc),axis=0)

        return mfcc

    def get_chroma(self):
        # Use librosa to get the Chroma coefficients
        cstft = librosa.feature.chroma_stft(self.data,self.sampleRate)
        ccqt = librosa.feature.chroma_cqt(self.data,self.sampleRate)
        return[cstsft,ccqt]

    def get_tonnetz(self):
        tonnetz = librosa.feature.tonnetz(self.data,self.sampleRate)
        return tonnetz

    def get_spectral_features(self):
        s1 = librosa.feature.spectral_bandwidth(self.data,self.sampleRate)
        s2 = librosa.feature.spectral_centroid(self.data,self.sampleRate)
        s3 = librosa.feature.spectral_contrast(self.data,self.sampleRate)
        s4 = librosa.feature.spectral_rolloff(self.data,self.sampleRate)

        zcr = librosa.feature.zero_crossing_rate(self.data,self.sampleRate)
        return [s1,s2,s3,s4,zcr]

    def other_features(self):
        librosa.fft_frequencies(self.sampleRate)
        librosa.cqt_frequencies()
        librosa.audio.get_duration()

        # Estimate dominant frequency of STFT bins by parabolic interpolation
        librosa.piptrack()

        # Adaptive noise floor -> read up
        librosa.feature.logamplitude()

        librosa.onset.onset_detect()
        librosa.onset.onset_strength()

    def get_lpc(self):
        from scikits.talkbox import lpc
        lpc(data,order)

    def entropy(self,s):
        e = -s[np.nonzero(s)]**2 * np.log(s[np.nonzero(s)]**2)
        return np.sum(e)

    def energy(self,sg):
        return np.sum(10.0**(sg/10.0))*(self.sampleRate/self.config['window_width'])

    def wiener_entropy(self):
        return np.exp(1.0/len(self.data) * np.sum(np.log(self.data))) / (1.0/len(self.data) * np.sum(self.data))

    def get_spectrogram_measurements(self,t1,t2,f1,f2):
        # The first set of Raven features:
        # average power, delta power, energy, aggregate+average entropy, max+peak freq, max+peak power
        # These mostly assume that you have clipped the call in time and frequency
        #nbins = number of pixels = width*height
        avgPower = np.sum(sg[t1:t2,f1:f2])/nbins
        deltaPower = np.sum(sg[:,f2]) - np.sum(sg[:,f1])
        # For the energy, should this be log(sg)?
        energy = self.energy(sg[t1:t2,f1:f2])
        Ebin = self.energy(sg[t1:t2,f])
        aggEntropy = 0
        for f in range(f1,f2):
            aggEntropy += self.energy(sg[t1:t2,f]/Ebin)
        #nframes =
        avgEntropy = 0
        for t in range(t1,t2):
            avgEntropy += self.entropy(sg[t,f1:f2])
        avgEntropy /= nframes

        maxPower = np.max(sg[t1:t2,f1:f2])
        peakPower = np.max(np.abs(sg[t1:t2,f1:f2]))
        maxFreq = np.argmax(sg[t1:t2,f1:f2])[1]
        peakFreq = np.argmax(np.abs(sg[t1:t2,f1:f2]))[1]

        return (avgPower, deltaPower, energy, avgEntropy, maxPower, peakPower, maxFreq, peakFreq)

    def get_robust_measurements(self,t1,t2,f1,f2):
        # The second set of Raven features
        # 1st, 2nd, 3rd quartile, 5%, 95% frequency, inter-quartile range, bandwidth 90%
        # Ditto for time
        # Cumulative sums to get the quartile points for freq and time
        csf = np.cumsum(sg[f1:f2, t1:t2], axis=0)
        cst = np.cumsum(sg[f1:f2, t1:t2], axis=1)

        # List of the frequency points (5%, 25%, 50%, 75%, 95%)
        list = [.05, .25, .5, .75, .95]

        freqindices = []
        index = 0
        i = 0
        while i < (len(csf)) and index < len(list):
            if csf[i] > list[index] * csf[-1]:
                freqindices.extend(str(i))
                index += 1
            i += 1

        timeindices = []
        index = 0
        i = 0
        while i < (len(cst)) and index < len(list):
            if cst[i] > list[index] * cst[-1]:
                timeindices.extend(str(i))
                index += 1
            i += 1

        return (freqindices, freqindices[3] - freqindices[1], freqindices[4] - freqindices[0], timeindices, timeindices[3] - timeindices[1], timeindices[4] - timeindices[0])

    def get_waveform_measurements(self,t1,t2,f1,f2):
        # The third set of Raven features
        # Min, max, peak, RMS, filtered RMS amplitude (and times for the first 3), high, low, delta frequency, length of time
        mina = np.min(self.data[t1:t2])
        mint = np.argmin(self.data[t1,t2])
        maxa = np.max(self.data[t1:t2])
        maxt = np.argmax(self.data[t1:t2])
        peaka = np.max(np.abs(self.data[t1:t2]))
        peakt = np.argmax(np.abs(self.data[t1:t2]))
        npoints = t2-t1/self.config[]
        rmsa = np.sqrt(np.sum(self.data[t1:t2]**2)/npoints)
        # rmsfa = np.sqrt(np.sum(self.data[t1:t2] ** 2) / npoints)
        # Also? max bearing, peak correlation, peak lag
        return (mina, mint, maxa, maxt, peaka, peakt,rmsa)

    def computeCorrelation(self):
        scipy.signal.fftconvolve(a, b, mode='same')

    def testDTW(self):
        x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
        y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]

        d = self.dtw(x,y,wantDistMatrix=True)
        print self.dtw_path(d)

def test():
    a = Features()
    a.testDTW()
    pl.show()
#test()