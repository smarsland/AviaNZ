# Version 0.1 30/5/16
# Author: Stephen Marsland

import numpy as np
#import pywt
#from scipy.io import wavfile
import pylab as pl
#import matplotlib
import librosa

# TODO:
# (1) Raven features, (2) MFCC, (3) LPC, (4) Random stuff from sounds, (5) Anything else
# Geometric distance and other metrics should go somewhere
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
# Linear Predictive Coding -> from scikits.talkbox import lpc (see also audiolazy) Librosa?
# Compute the spectral derivatives??

# Add something that plots some of these to help playing, so that I can understand the librosa things, etc.

# And assemble a decent dataset of birdcalls to play with.
# Or in fact, two: kiwi, ruru, bittern, and then a much bigger one

class Features:
    # This class implements various feature extraction algorithms for the AviaNZ interface
    # Given a segment as a region of audiodata (between start and stop points)
    # Classifiers will then be called on the features
    # Currently it's just MFCC, Raven, some playing.
    # TODO: test what there is so far!

    def __init__(self,data=[],sampleRate=0):
        self.data = data
        self.sampleRate = sampleRate

    def setNewData(self,data,sg,fs):
        # To be called when a new sound file is loaded
        self.data = data
        self.sg = sg
        self.fs = fs

    def get_mfcc(self):
        # Use librosa to get the MFCC coefficients
        mfcc = librosa.feature.mfcc(self.data, self.sampleRate)
        librosa.display.specshow(mfcc)

        # Normalise
        mfcc -= np.mean(mfcc,axis=0)
        mfcc /= np.max(np.abs(mfcc),axis=0)

        return mfcc

    def get_chroma(self):
        # Use librosa to get the chroma coefficients
        # Short-time energy in the 12 pitch classes
        # CQT is constant-Q
        cstft = librosa.feature.chroma_stft(self.data,self.sampleRate)
        ccqt = librosa.feature.chroma_cqt(self.data,self.sampleRate)
        return[cstsft,ccqt]

    def get_tonnetz(self):
        # Use librosa to get the tonnetz coefficients
        # This are an alternative pitch representation to chroma
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

    def get_lpc(self,data,order=44):
        # Use talkbox to get the linear predictive coding
        from scikits.talkbox import lpc
        return lpc(data,order)

    def entropy(self,s):
        # Compute the Shannon entropy
        e = -s[np.nonzero(s)] * np.log2(s[np.nonzero(s)])
        return np.sum(e)

    def wiener_entropy(self):
        # Also known as spectral flatness, geometric mean divided by arithmetic mean of power
        return np.exp(1.0/len(self.data) * np.sum(np.log(self.data))) / (1.0/len(self.data) * np.sum(self.data))

    def morgan(self,sg):
        # Pitch (= fundamental frequency)
        s = source(f, sampleRate, 128)
        ff = pitch("yin", 256, 128, sampleRate)
        total_frames = 0
        pitches = []
        while True:
            samples, read = s()
            thispitch = ff(samples)[0]
            # print("%f %f %f" % (total_frames / float(samplerate), pitch, confidence))
            pitches += [thispitch]
            total_frames += read
            if read < 128: break
        features[6, whichfile] = np.mean(pitches)

        # Power in decibels
        sgd = 10.0 * np.log10(sg)
        sgd[np.isinf(sgd)] = 0
        features[7, whichfile] = np.sum(sgd) / (np.shape(sgd)[0] * np.shape(sgd)[1])
        features[8, whichfile] = np.max(sgd)
        index = np.argmax(sgd)

        # Energy
        features[9, whichfile] = np.sum(sg)

        # Frequency (I'm going to use freq at which max power occurs)
        features[10, whichfile] = sg.flatten()[index]

    # The Raven Features (27 of them)
    # Frequency: 5%, 25%, centre, 75%, 95%, peak, max
    # And their times, min time
    # Amplitude: min, peak, max, rms, filtered rms
    # Power: average, peak, max
    # Other: peak lag, max bearing, energy, peak correlation

    def get_spectrogram_measurements(self,sg,fs,window_width,f1,f2,t1,t2):
        # The first set of Raven features:
        # average power, delta power, energy, aggregate+average entropy, max+peak freq, max+peak power
        # These mostly assume that you have clipped the call in time and **frequency
        # Raven takes Fourier transform of signal and squares each coefficient to get power spectral density
        # Except this doesn't seem to be quite true, since many of them are in decibels!

        # t1, t2, f1, f2 are in pixels

        # Compute the energy before changing into decibels
        energy = 10.*np.log10(np.sum(sg[f1:f2,t1:t2])*fs/window_width)
        Ebin = np.sum(sg[f1:f2,t1:t2],axis=1)
        # Energy in each frequency bin over whole time
        Ebin /= np.sum(Ebin)
        # Entropy for each time slice
        # TODO: Unconvinced about this next one
        Fbin = 0
        for t in range(t2-t1):
            Fbin += self.entropy(sg[f1:f2,t+t1])
        aggEntropy = np.sum(self.entropy(Ebin))
        avgEntropy = Fbin/(t2-t1)

        # Convert spectrogram into decibels
        sg = -10.0*np.log10(sg)
        sg[np.isinf(sg)] = 0

        avgPower = np.sum(sg[f1:f2,t1:t2])/((f2-f1)*(t2-t1))
        # Should this have the sums in?
        deltaPower = (np.sum(sg[f2-1,:]) - np.sum(sg[f1,:]))/np.shape(sg)[1]

        maxPower = np.max(sg[f1:f2,t1:t2])
        maxFreq = (np.argmax(sg[f1:f2,t1:t2])+f1) * float(fs)/np.shape(sg)[0]

        return (avgPower, deltaPower, energy, aggEntropy, avgEntropy, maxPower, maxFreq)

    def get_robust_measurements(self,sg,length,fs,f1,f2,t1,t2):
        # The second set of Raven features
        # 1st, 2nd (centre), 3rd quartile, 5%, 95% frequency, inter-quartile range, bandwidth 90%
        # Ditto for time
        # Cumulative sums to get the quartile points for freq and time

        # t1, t2, f1, f2 are in pixels
        sg = -10.0*np.log10(sg)
        sg[np.isinf(sg)] = 0

        sgt = np.sum(sg[f1:f2,t1:t2], axis=0)
        cst = np.cumsum(sgt)

        sgf = np.sum(sg[f1:f2,t1:t2], axis=1)
        csf = np.cumsum(sgf)

        # List of the frequency points (5%, 25%, 50%, 75%, 95%)
        list = [.05, .25, .5, .75, .95]

        freqindices = np.zeros(5)
        index = 0
        i = 0
        while i < len(csf) and index < len(list):
            if csf[i] > list[index] * csf[-1]:
                freqindices[index] = i+1
                index += 1
            i += 1

        timeindices = np.zeros(5)
        index = 0
        i = 0
        while i < len(cst) and index < len(list):
            if cst[i] > list[index] * cst[-1]:
                timeindices[index] = i+1
                index += 1
            i += 1

        # Check that the centre time/freq are in the middle (look good)
        print np.sum(sg[f1:f2,t1:timeindices[2]]), np.sum(sg[f1:f2,timeindices[2]:t2])
        print np.sum(sg[f1:freqindices[2], t1:t2]), np.sum(sg[freqindices[2]:f2, t1:t2])

        freqindices = (freqindices+f1) * float(fs)/np.shape(sg)[0]
        timeindices = (timeindices+t1)/np.shape(sg)[1] * length
        return (freqindices, freqindices[3] - freqindices[1], freqindices[4] - freqindices[0], timeindices, timeindices[3] - timeindices[1], timeindices[4] - timeindices[0])

    def get_waveform_measurements(self,data,fs,t1,t2):
        # The third set of Raven features
        # Min, max, peak, RMS, filtered RMS amplitude (and times for the first 3), high, low, delta frequency, length of time

        # First, convert t1 and t2 into points in the amplitude plot
        t1 = float(t1) / np.shape(sg)[1] * len(data)
        t2 = float(t2) / np.shape(sg)[1] * len(data)

        mina = np.min(data[t1:t2])
        mint = float(np.argmin(data[t1:t2])+t1) / fs
        maxa = np.max(data[t1:t2])
        maxt = float(np.argmax(data[t1:t2])+t1) / fs
        peaka = np.max(np.abs(data[t1:t2]))
        peakt = float(np.argmax(np.abs(data[t1:t2]))+t1) / fs
        # TODO: check
        rmsa = np.sqrt(np.sum(data[t1:t2]**2)/len(data[t1:t2]))
        # Filtered rmsa (bandpass filtered first)
        # Also? max bearing, peak correlation, peak lag
        return (mina, mint, maxa, maxt, peaka, peakt,rmsa)

    def computeCorrelation(self):
        scipy.signal.fftconvolve(a, b, mode='same')



def test():
    a = Features()
    a.testDTW()
    pl.show()

def raven():
    #data, fs = librosa.load('Sound Files/tril1.wav',sr=None)
    #data, fs = librosa.load('Sound Files/kiwi.wav',sr=None)
    data, sampleRate = librosa.load('Sound Files/male1.wav',sr=None)

    import SignalProc
    sp = SignalProc.SignalProc()
    sg = sp.spectrogram(data,fs,multitaper=False)
    print np.shape(sg)
    f = Features()
    a = f.get_spectrogram_measurements(sg,fs,256,0,np.shape(sg)[0],0,np.shape(sg)[1])
    b = f.get_robust_measurements(sg,len(data)/fs,fs,0,np.shape(sg)[0],0,np.shape(sg)[1])
    c = f.get_waveform_measurements(data,fs,0,len(data))
    return a, b, c
#test()