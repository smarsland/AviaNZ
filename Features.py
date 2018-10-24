
# Features.py
#
# Code to extract various features from sound files
# Version 1.3 23/10/18
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis

#    AviaNZ birdsong analysis program
#    Copyright (C) 2017--2018

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import librosa
import SignalProc
import SupportClasses
import wavio


# TODO:
# First thing is to get my head around everything that is going on, which is:
    # Transforms of the data
        # Features extracted from those transforms

    # None (waveform)
        # Various amplitude measures (max, min, peak)
    # STFT (spectrogram)
        # Energy, entropy, power, spectral moments
    # DCT (related to real-values of FFT)
    # Mel
        # MFCC
    # Chroma and Constant-Q
        # Tonnetz
        # Chroma-CENS
    # LPC
    # Wavelets
        # Energy
        # Coefficients

    # Alignments that can be used to match up segments
        # DTW
        # Edit
        # LCS

    # Feature vectors for machine learning

    # Distance metrics for template matching methods (nearest neighbour, cross-correlation)
        # L1, L2
        # Geometric distance
        # SRVF for curves
        # Cross-correlation
        # Hausdorff (too slow)

# Get MFCC and LPC working -> per segment?
# Assemble the other features
# Put all into a vector, save

# Get smoothed fundamental freq
# Compute L1, L2, geometric and SRVF distances
# And cross-correlation

# Get DTW, LCS and edit distance alignment going

# Dominant frequency (highest energy)

# (1) Raven features, (2) MFCC, (3) LPC, (4) Random stuff from sounds, (5) Wavelets, (6) Anything else, (7) Fundamental freq
# Distances: L1, L2, Geometric, Cosine
# Alignment: DTW, LCS, Edit

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

# Fundamental frequency -- yin? (de Cheveigne and Kawahara 2002)
# Add something that plots some of these to help playing, so that I can understand the librosa things, etc.

# And assemble a decent dataset of birdcalls to play with.
# Or in fact, two: kiwi, ruru, bittern, and then a much bigger one

class Features:
    # This class implements various feature extraction algorithms for the AviaNZ interface
    # Given a segment as a region of audiodata (between start and stop points)
    # Classifiers will then be called on the features
    # Currently it's just MFCC, Raven, some playing.

    def __init__(self,data=[],sampleRate=0,window_width=256,incr=128):
        self.data = data
        self.sampleRate = sampleRate
        self.window_width=window_width
        self.incr = incr
        sp = SignalProc.SignalProc(sampleRate=self.sampleRate,window_width=self.window_width,incr=self.incr)
        # The next lines are to get a spectrogram that *should* precisely match the Raven one
        self.sg = sp.spectrogram(data, multitaper=False,window_width=self.window_width,incr=self.incr,window='Ones')
        self.sg = self.sg**2

    def setNewData(self,data,sampleRate):
        # To be called when a new sound file is loaded
        self.data = data
        self.sampleRate = sampleRate
        self.sg = sp.spectrogram(data, multitaper=False,window_width=self.window_width,incr=self.incr,window='Ones')

    def get_mfcc(self):
        # Use librosa to get the MFCC coefficients. These seem to have a window of 512, not changeable (NP: can change it, n_fft=2048, hop_length=512)
        mfcc = librosa.feature.mfcc(self.data, self.sampleRate)

        # Normalise
        mfcc -= np.mean(mfcc,axis=0)
        mfcc /= np.max(np.abs(mfcc),axis=0)

        return mfcc

    def get_chroma(self):
        # Use librosa to get the chroma coefficients
        # Short-time energy in the 12 pitch classes
        # CQT is constant-Q
        # Windows size is again 512
        cstft = librosa.feature.chroma_stft(self.data,self.sampleRate)
        ccqt = librosa.feature.chroma_cqt(self.data,self.sampleRate)
        cens = librosa.feature.chroma_cens(self.data,self.sampleRate)
        return[cstft,ccqt,cens]

    def get_tonnetz(self):
        # Use librosa to get the 6 tonnetz coefficients
        # This is an alternative pitch representation to chroma
        tonnetz = librosa.feature.tonnetz(self.data,self.sampleRate)
        return tonnetz

    def get_spectral_features(self):
        s1 = librosa.feature.spectral_bandwidth(self.data,self.sampleRate)
        s2 = librosa.feature.spectral_centroid(self.data,self.sampleRate)
        s3 = librosa.feature.spectral_contrast(self.data,self.sampleRate)
        s4 = librosa.feature.spectral_rolloff(self.data,self.sampleRate)

        zcr = librosa.feature.zero_crossing_rate(self.data,self.sampleRate)
        return [s1,s2,s3,s4,zcr]

    def get_lpc(self,data,order=44):
        # Use talkbox to get the linear predictive coding
        from scikits.talkbox import lpc
        coefs = lpc(data,order)
        return coefs[0]

    def wiener_entropy(self,data):
        # Also known as spectral flatness, geometric mean divided by arithmetic mean of power
        return np.exp(1.0/len(data) * np.sum(np.log(data))) / (1.0/len(data) * np.sum(data))

    # The Raven Features (27 of them)
    # Frequency: 5%, 25%, centre, 75%, 95%, peak, max
    # And their times, min time
    # Amplitude: min, peak, max, rms, filtered rms
    # Power: average, peak, max
    # Other: peak lag, max bearing, energy, peak correlation

    def get_Raven_spectrogram_measurements(self,sg,fs,window_width,f1,f2,t1,t2):
        """ The first set of Raven features.
        energy, aggregate+average entropy, average power, delta power, max+peak freq, max+peak power

        The function is given a spectrogram and 4 indices into it (t1, t2, f1, f2) in pixels, together with the sample rate and window width for the spectrogram.

        These features should match the Raven ones, but that's hard since their information is inconsistent.
        For example, they state (p 168) that the computation are based on the PSD, which they make by taking the Fourier transform of the signal and squaring
        each coefficient. This has no windowing function. Also, many of their computations are stated to be in decibels.

        The description of each computation is in the comments.
        """

        # Compute the energy
        # This is a little bit unclear. Eq (6.1) of Raven is the calculation below, but then it says it is in decibels, which this is not!
        energy = np.sum(sg[t1:t2,f1:f2])*float(fs)/window_width

        # Entropy of energy in each frequency bin over whole time
        Ebin = np.sum(sg[t1:t2,f1:f2],axis=0)
        Ebin /= np.sum(Ebin)
        aggEntropy = np.sum(-Ebin*np.log2(Ebin))

        # Entropy of each frame (time slice) averaged
        newsg = (sg.T/np.sum(sg,axis=1)).T
        avgEntropy = np.sum(-newsg*np.log2(newsg),axis=1)
        avgEntropy = np.mean(avgEntropy)

        # Convert spectrogram into decibels
        sg = np.abs(np.where(sg == 0, 0.0, 10.0 * np.log10(sg)))

        # Sum of PSD divided by number of pixels
        avgPower = np.sum(sg[t1:t2,f1:f2])/((f2-f1)*(t2-t1))

        # Power at the max frequency minus power at min freq. Unclear how it deals with the fact that there isn't one time in there!
        deltaPower = (np.sum(sg[:,f2-1]) - np.sum(sg[:,f1]))/np.shape(sg)[1]

        # Max power is the darkest pixel in the spectrogram
        maxPower = np.max(sg[t1:t2,f1:f2])

        # Max frequency is the frequency at which max power occurs
        maxFreq = (np.unravel_index(np.argmax(sg[t1:t2,f1:f2]), np.shape(sg[t1:t2,f1:f2]))[1] + f1) * float(fs/2.)/np.shape(sg)[1]
        return (avgPower, deltaPower, energy, aggEntropy, avgEntropy, maxPower, maxFreq)

    def get_Raven_robust_measurements(self,sg,fs,f1,f2,t1,t2):
        """ The second set of Raven features.
        1st, 2nd (centre), 3rd quartile, 5%, 95% frequency, inter-quartile range, bandwidth 90%
        Ditto for time

        t1, t2, f1, f2 are in pixels
        Compute the cumulative sums and then loop through looking for the 5 points where the left and right are in the correct proportion,
        i.e., the left is some percentage of the total.
        """

        sg = np.abs(np.where(sg == 0, 0.0, 10.0 * np.log10(sg)))

        # List of the match points (5%, 25%, 50%, 75%, 95%)
        list = [.05, .25, .5, .75, .95]

        sgf = np.sum(sg[t1:t2,f1:f2], axis=0)
        csf = np.cumsum(sgf)

        freqindices = np.zeros(5)
        index = 0
        i = 0
        while i < len(csf) and index < len(list):
            if csf[i] > list[index] * csf[-1]:
                freqindices[index] = i
                index += 1
            i += 1

        sgt = np.sum(sg[t1:t2,f1:f2], axis=1)
        cst = np.cumsum(sgt)

        timeindices = np.zeros(5)
        index = 0
        i = 0
        while i < len(cst) and index < len(list):
            if cst[i] > list[index] * cst[-1]:
                timeindices[index] = i
                index += 1
            i += 1

        # Turn into frequencies and times
        # Maxfreq in sg is fs/2, so float(fs/2.)/np.shape(sg)[1] is the value of 1 bin
        freqindices = (freqindices+f1) * float(fs/2.)/np.shape(sg)[1]
        # The time between two columns of the spectrogram is the increment divided by the sample rate
        timeindices = (timeindices+t1) * self.incr / fs
        return (freqindices, freqindices[3] - freqindices[1], freqindices[4] - freqindices[0], timeindices, timeindices[3] - timeindices[1], timeindices[4] - timeindices[0])

    def get_Raven_waveform_measurements(self,data,fs,t1,t2):
        """ The third set of Raven features. These are based on the waveform instead of the spectrogram.

        Min, max, peak, RMS, filtered RMS amplitude (and times for the first 3), high, low, delta frequency, length of time

        t1, t2 are in spectrogram pixels
        """

        # First, convert t1 and t2 into points in the amplitude plot
        t1 = t1 * self.incr
        t2 = t2 * self.incr

        mina = np.min(data[t1:t2])
        mint = float(np.argmin(data[t1:t2])+t1) / fs
        maxa = np.max(data[t1:t2])
        maxt = float(np.argmax(data[t1:t2])+t1) / fs
        peaka = np.max(np.abs(data[t1:t2]))
        peakt = float(np.argmax(np.abs(data[t1:t2]))+t1) / fs
        rmsa = np.sqrt(np.sum(data[t1:t2]**2)/len(data[t1:t2]))
        # Filtered rmsa (bandpass filtered first)
        # TODO
        # Also? max bearing, peak correlation, peak lag
        return (mina, mint, maxa, maxt, peaka, peakt,rmsa)

    def computeCorrelation(self):
        scipy.signal.fftconvolve(a, b, mode='same')

def testFeatures():
    import wavio
    wavobj = wavio.read('Sound Files/tril1.wav')
    fs = wavobj.rate
    data = wavobj.data

    if data.dtype is not 'float':
        data = data.astype('float') # / 32768.0

    if np.shape(np.shape(data))[0] > 1:
        data = data[:, 0]

    sp = SignalProc.SignalProc(sampleRate=fs, window_width=256, incr=128)
    # The next lines are to get a spectrogram that *should* precisely match the Raven one
    #sg = sp.spectrogram(data, multitaper=False, window_width=256, incr=128, window='Ones')
    #sg = sg ** 2
    sg = sp.spectrogram(data, multitaper=False, window_width=256, incr=128, window='Hann')

    f = Features(data,fs,256,128)

    features = []
    # Loop over the segments (and time slices within?)
    features.append([f.get_Raven_spectrogram_measurements(sg=sg,fs=fs,window_width=256,f1=0,f2=np.shape(sg)[1],t1=0,t2=np.shape(sg)[0]),f.get_Raven_robust_measurements(sg,fs,0,np.shape(sg)[1],0,np.shape(sg)[0]),f.get_Raven_waveform_measurements(data,fs,0,len(data)),f.weiner_entropy(data)])

    # Will need to think about feature vector length for the librosa features, since they are on fixed windows
    f.get_chroma()
    f.get_mfcc()
    f.get_tonnetz()
    f.get_spectral_features()
    f.get_lpc(data,order=44)
    # DCT


def mfcc(y1,y2,y3,sr1,sr2,sr3,yTest,srTest):
    # import dtw
    # import editdistance

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
        dist1i = librosa.dtw(mfcc1.T, mfcci.T, dist=lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
        dist2i = librosa.dtw(mfcc2.T, mfcci.T, dist=lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
        dist3i = librosa.dtw(mfcc3.T, mfcci.T, dist=lambda x, y: np.exp(np.linalg.norm(x - y, ord=1)))[0]
        dists[i] = (dist1i + dist2i + dist3i) / 3
    import matplotlib.pyplot as plt
    plt.plot(dists)

    # select minimum distance window
    word_match_idx = dists.argmin()
    # convert MFCC to time domain
    word_match_idx_bnds = np.array([word_match_idx, np.ceil(word_match_idx + window_size)])
    samples_per_mfcc = 512
    word_samp_bounds = (2 / 2) + (word_match_idx_bnds * samples_per_mfcc)

    word = yTest[word_samp_bounds[0]:word_samp_bounds[1]]

def mfcc_dtw(y, sr,yTest,srTest):
    # Convert the data to mfcc:
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=24,n_fft=2048, hop_length=512) # n_fft=10240, hop_length=2560
    mfccTest = librosa.feature.mfcc(yTest, srTest, n_mfcc=24, n_fft=2048, hop_length=512)
    # get delta mfccs
    mfcc_delta=librosa.feature.delta(mfcc)
    mfccTest_delta=librosa.feature.delta(mfccTest)
    # then merge
    mfcc=np.concatenate((mfcc,mfcc_delta),axis=0)
    mfccTest = np.concatenate((mfccTest, mfccTest_delta), axis=0)

    # mfcc = mfcc1.mean(1)
    # mfccTest = mfccTest.mean(1)

    # Remove mean and normalize each column of MFCC
    import copy
    def preprocess_mfcc(mfcc):
        mfcc_cp = copy.deepcopy(mfcc)
        for i in xrange(mfcc.shape[1]):
            mfcc_cp[:, i] = mfcc[:, i] - np.mean(mfcc[:, i])
            mfcc_cp[:, i] = mfcc_cp[:, i] / np.max(np.abs(mfcc_cp[:, i]))
        return mfcc_cp

    mfcc = preprocess_mfcc(mfcc)
    mfccTest = preprocess_mfcc(mfccTest)

    #average MFCC over all frames
    mfcc=mfcc.mean(1)
    mfccTest=mfccTest.mean(1)

    #Calculate the distances from the test signal
    d, wp = librosa.dtw(mfccTest, mfcc, metric='euclidean')

    return d[d.shape[0] - 1][d.shape[1] - 1]

#def filters():
    # dct, mel, chroma, constant_q

# Distance Functions

def lcs(s0, s1, distmatrix=None,s0ind=None,s1ind=None):
    if distmatrix is None:
        distmatrix = lcsDistanceMatrix(s0,s1)
    if s0ind is None:
        s0ind = len(s0)
    if s1ind is None:
        s1ind = len(s1)

    if distmatrix[s0ind][s1ind] == 0:
        return ""
    elif s0[s0ind-1] == s1[s1ind-1]:
        return lcs(s0, s1, distmatrix, s0ind-1, s1ind-1) + s0[s0ind-1]
    elif distmatrix[s0ind][s1ind-1] > distmatrix[s0ind-1][s1ind]:
        return lcs(s0, s1, distmatrix, s0ind, s1ind-1)
    else:
        return lcs(s0, s1, distmatrix, s0ind-1, s1ind)

def lcsDistanceMatrix(s0, s1):

    distMatrix = np.zeros((len(s0)+1,len(s1)+1))

    for i in range(len(s0)):
        for j in range(len(s1)):
            if s0[i] == s1[j]:
                distMatrix[i+1][j+1] = 1 + distMatrix[i][j]
            else:
                distMatrix[i+1][j+1] = max(distMatrix[i][j+1], distMatrix[i+1][j])
    return distMatrix

#--- testig
def loadFile(filename):
    wavobj = wavio.read(filename)
    sampleRate = wavobj.rate
    audiodata = wavobj.data

    # None of the following should be necessary for librosa
    if audiodata.dtype is not 'float':
        audiodata = audiodata.astype('float') #/ 32768.0
    if np.shape(np.shape(audiodata))[0]>1:
        audiodata = audiodata[:,0]

    # if sampleRate != 16000:
    #     audiodata = librosa.core.audio.resample(audiodata, sampleRate, 16000)
    #     sampleRate=16000

    # pre-process
    sc = SupportClasses.preProcess(audioData=audiodata, sampleRate=sampleRate, species='Kiwi', df=False)
    audiodata,sampleRate = sc.denoise_filter()
    return audiodata,sampleRate

#####
# yTest,srTest=loadFile('Sound Files/dtw_mfcc/kiwi/kiwifemale/bf10.wav')

def isKiwi_dtw_mfcc(dirName, yTest, srTest):
    '''
    given the set of kiwi templates (folder) and the test file
    :return: a binary value (kiwi or not)
    '''
    import os
    dList=[]
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.endswith('.wav'):
                filename = root + '/' + filename #[:-4]
                y, sr = loadFile(filename)
                d = mfcc_dtw(y=y, sr=sr, yTest=yTest, srTest=srTest)
                print(filename, d)
                dList.append(d)
    if sorted(dList)[1]<0.2:
        print('it is KIWI',sorted(dList)[1])
    else:
        print('it is NOT kiwi', sorted(dList)[1])
    return dList

# dList=isKiwi_dtw_mfcc("Sound Files/dtw_mfcc/kiwi/kiwimale", yTest,srTest)
#print dList
