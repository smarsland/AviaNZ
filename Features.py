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

# Wavelet energy

# Add chroma features and Tonnetz
# Vibrato
# Prosodic features (pitch, duration, intensity)
# Spectral statistics
# Frequency modulation
# Linear Predictive Coding -> from scikits.talkbox import lpc (see also audiolazy) Librosa?
# Compute the spectral derivatives??

# Fundamental frequency -- yin? (de Cheveigne and Kawahara 2002)
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
        return[cstft,ccqt]

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
        print np.sum(sg[f1:int(f2),t1:int(timeindices[2])]), np.sum(sg[f1:f2,int(timeindices[2]):int(t2)])
        print np.sum(sg[f1:int(freqindices[2]), t1:int(t2)]), np.sum(sg[int(freqindices[2]):int(f2), t1:int(t2)])

        freqindices = (freqindices+f1) * float(fs)/np.shape(sg)[0]
        timeindices = (timeindices+t1)/np.shape(sg)[1] * length
        return (freqindices, freqindices[3] - freqindices[1], freqindices[4] - freqindices[0], timeindices, timeindices[3] - timeindices[1], timeindices[4] - timeindices[0])

    def get_waveform_measurements(self,sg,data,fs,t1,t2):
        # The third set of Raven features
        # Min, max, peak, RMS, filtered RMS amplitude (and times for the first 3), high, low, delta frequency, length of time

        # First, convert t1 and t2 into points in the amplitude plot
        t1 = float(t1) / np.shape(sg)[1] * len(data)
        t2 = float(t2) / np.shape(sg)[1] * len(data)

        mina = np.min(data[int(t1):int(t2)])
        mint = float(np.argmin(data[int(t1):int(t2)])+int(t1)) / fs
        maxa = np.max(data[int(t1):int(t2)])
        maxt = float(np.argmax(data[int(t1):int(t2)])+int(t1)) / fs
        peaka = np.max(np.abs(data[int(t1):int(t2)]))
        peakt = float(np.argmax(np.abs(data[int(t1):int(t2)]))+int(t1)) / fs
        # TODO: check
        rmsa = np.sqrt(np.sum(data[int(t1):int(t2)]**2)/len(data[int(t1):int(t2)]))
        # Filtered rmsa (bandpass filtered first)
        # Also? max bearing, peak correlation, peak lag
        return (mina, mint, maxa, maxt, peaka, peakt,rmsa)

    def computeCorrelation(self):
        scipy.signal.fftconvolve(a, b, mode='same')

def raven():
    #data, fs = librosa.load('Sound Files/tril1.wav',sr=None)
    #data, fs = librosa.load('Sound Files/kiwi.wav',sr=None)
    data, fs = librosa.load('Sound Files/male1.wav',sr=None)
    # wavobj = wavio.read(self.filename, self.lenRead, self.startRead)

    import SignalProc
    sp = SignalProc.SignalProc()
    # sg = sp.spectrogram(data,fs,multitaper=False) # spectrogram(self,data,window_width=None,incr=None,window='Hann',mean_normalise=True,onesided=True,multitaper=False,need_even=False)
    sg = sp.spectrogram(data, multitaper=False)
    print np.shape(sg)
    f = Features()
    a = f.get_spectrogram_measurements(sg=sg,fs=fs,window_width=256,f1=0,f2=np.shape(sg)[0],t1=0,t2=np.shape(sg)[1])
    #get_spectrogram_measurements(self,sg,fs,window_width,f1,f2,t1,t2)
    b = f.get_robust_measurements(sg,len(data)/fs,fs,0,np.shape(sg)[0],0,np.shape(sg)[1])
    c = f.get_waveform_measurements(sg,data,fs,0,len(data))
    return a, b, c
raven()

def mfcc():
    import dtw
    import editdistance

    # Convert the data to mfcc:
    mfcc1 = librosa.feature.mfcc(y1, sr1,n_mfcc=20)
    mfcc2 = librosa.feature.mfcc(y2, sr2,n_mfcc=20)
    mfcc3 = librosa.feature.mfcc(y3, sr3,n_mfcc=20)
    mfccTest = librosa.feature.mfcc(yTest, srTest)

    # Remove mean and normalize each column of MFCC
    import copy
    def preprocess_mfcc(mfcc):
        mfcc_cp = copy.deepcopy(mfcc)
        for i in xrange(mfcc.shape[1]):
            mfcc_cp[:, i] = mfcc[:, i] - np.mean(mfcc[:, i])
            mfcc_cp[:, i] = mfcc_cp[:, i] / np.max(np.abs(mfcc_cp[:, i]))
        return mfcc_cp

    mfcc1 = preprocess_mfcc(mfcc1)
    mfcc2 = preprocess_mfcc(mfcc2)
    mfcc3 = preprocess_mfcc(mfcc3)
    mfccTest = preprocess_mfcc(mfccTest)

    window_size = mfcc1.shape[1]
    dists = np.zeros(mfccTest.shape[1] - window_size)

    for i in range(len(dists)):
        mfcci = mfccTest[:, i:i + window_size]
        dist1i = dtw(mfcc1.T, mfcci.T, dist=lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
        dist2i = dtw(mfcc2.T, mfcci.T, dist=lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
        dist3i = dtw(mfcc3.T, mfcci.T, dist=lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
        dists[i] = (dist1i + dist2i + dist3i) / 3
    plt.plot(dists)

    # select minimum distance window
    word_match_idx = dists.argmin()
    # convert MFCC to time domain
    word_match_idx_bnds = np.array([word_match_idx, np.ceil(word_match_idx + window_size)])
    samples_per_mfcc = 512
    word_samp_bounds = (2 / 2) + (word_match_idx_bnds * samples_per_mfcc)

    word = yTest[word_samp_bounds[0]:word_samp_bounds[1]]

    # Command to embed audio in IPython notebook :)
    #IPython.display.Audio(data=word, rate=sr1)

#def filters():
    # dct, mel, chroma, constant_q