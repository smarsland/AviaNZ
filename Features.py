
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
import WaveletSegment
import wavio
from scipy import signal
import math
import os
import re

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

# List from Raven:
    #1st Quartile Frequency Max Power Time Max Time
    #3rd Quartile Frequency Min Amplitude Quartile Time Min Time
    #Average Power Peak Amplitude
    #Center Frequency Peak Correlation
    #Center Time Peak Frequency
    #Energy Peak Lag
    #filtered RMS Amplitude Peak Power
    #Frequency 5% Peak Time 95% RMS Amplitude
    #Max Amplitude Time 5% #Max Bearing Time 95% #Max Frequency

# Wavelet energy

# Add chroma features and Tonnetz
# Vibrato
# Prosodic features (pitch, duration, intensity)
# Spectral statistics
# Frequency modulation
# Linear Predictive Coding -> from scikits.talkbox import lpc (see also audiolazy) Librosa?
# Spectral derivative

# Fundamental frequency -- yin? (de Cheveigne and Kawahara 2002)
# Add something that plots some of these to help playing, so that I can understand the librosa things, etc.

# And assemble a decent dataset of birdcalls to play with.
# Or in fact, two: kiwi, ruru, bittern, and then a much bigger one

class Features:
    # This class implements various feature extraction algorithms for the AviaNZ interface
    # Given a segment as a region of audiodata (between start and stop points)

    def __init__(self, data=[], sampleRate=0, window_width=256, incr=128):
        self.data = data
        self.sampleRate = sampleRate
        self.window_width=window_width
        self.incr = incr
        sp = SignalProc.SignalProc(window_width=self.window_width, incr=self.incr)
        sp.data = self.data
        sp.sampleRate = self.sampleRate
        # The next lines are to get a spectrogram that *should* precisely match the Raven one
        self.sg = sp.spectrogram(sgType='Standard',window_width=self.window_width,incr=self.incr,window='Ones')
        self.sg = self.sg**2

    def setNewData(self,data,sampleRate):
        # To be called when a new sound file is loaded
        self.data = data
        self.sampleRate = sampleRate
        self.sg = sp.spectrogram(data, sgType='Standard',window_width=self.window_width,incr=self.incr,window='Ones')

    def get_mfcc(self, n_mfcc=48, n_bins=32, delta=True):
        # Use librosa to get the MFCC coefficients.
        # n_bins = 8  # kakapo boom
        mfcc = librosa.feature.mfcc(y=self.data, sr=self.sampleRate, n_mfcc=n_mfcc, n_fft=2048,
                                        hop_length=512)  # n_fft=10240, hop_length=2560
        if delta:
            if n_bins == 8:
                mfcc_delta = librosa.feature.delta(mfcc, width=5)
            else:
                mfcc_delta = librosa.feature.delta(mfcc)
            mfcc = np.concatenate((mfcc, mfcc_delta), axis=0)

        # Normalise
        mfcc -= np.mean(mfcc,axis=0)
        mfcc /= np.max(np.abs(mfcc),axis=0)

        # # Or:
        # melBasis = librosa.filters.mel(self.sampleRate)
        # melSpec = np.dot(melBasis,self.data)

        return mfcc

    def get_WE(self, nlevels=5):
        """ Wavelet energies
        """
        ws = WaveletSegment.WaveletSegment(spInfo=[])
        WE = ws.computeWaveletEnergy(data=self.data, sampleRate=self.sampleRate, nlevels=nlevels, wpmode='new')

        return WE

    def get_chroma(self):
        # Use librosa to get the chroma coefficients
        # Short-time energy in the 12 pitch classes
        # CQT is constant-Q
        # Windows size is again 512
        cstft = librosa.feature.chroma_stft(self.data,self.sampleRate)
        ccqt = librosa.feature.chroma_cqt(self.data,self.sampleRate)
        cens = librosa.feature.chroma_cens(self.data,self.sampleRate)
        return[cstft, ccqt, cens]

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
        s5 = librosa.feature.spectral_flatness(self.data,self.sampleRate)

        zcr = librosa.feature.zero_crossing_rate(self.data,self.sampleRate)
        return [s1,s2,s3,s4,s5,zcr]

    def get_lpc(self,data,order=44):
        # Use talkbox to get the linear predictive coding
        from scikits.talkbox import lpc
        coefs = lpc(data,order)
        return coefs[0]

    # The Raven Features (27 of them)
    # Frequency: 5%, 25%, centre, 75%, 95%, peak, max (And their times, min time)
    # Amplitude: min, peak, max, rms, filtered rms
    # Power: average, peak, max
    # Other: peak lag, max bearing, energy, peak correlation

    def get_Raven_spectrogram_measurements(self, f1, f2):
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
        energy = np.sum(self.sg[:,f1:f2])*self.sampleRate/self.window_width

        # Entropy of energy in each frequency bin over whole time
        Ebin = np.sum(self.sg[:,f1:f2], axis=0)
        Ebin /= np.sum(Ebin)
        aggEntropy = np.sum(-Ebin*np.log2(Ebin))

        # Entropy of each frame (time slice) averaged
        newsg = (self.sg.T/np.sum(self.sg, axis=1)).T
        avgEntropy = np.sum(-newsg*np.log2(newsg), axis=1)
        avgEntropy = np.mean(avgEntropy)

        # Convert spectrogram into decibels
        sg = np.abs(np.where(self.sg == 0, 0.0, 10.0 * np.log10(self.sg)))

        # Sum of PSD divided by number of pixels
        # avgPower = np.sum(sg[t2-t1,f1:f2])/((f2-f1)*(t2-t1))
        avgPower = np.sum(sg[:,f1:f2])/((f2-f1)*(np.shape(sg)[0]))

        # Power at the max frequency minus power at min freq. Unclear how it deals with the fact that there isn't one time in there!
        deltaPower = (np.sum(sg[:, f2-1]) - np.sum(sg[:,f1]))/np.shape(sg)[1]

        # Max power is the darkest pixel in the spectrogram
        maxPower = np.max(sg[:, f1:f2])

        # Max frequency is the frequency at which max power occurs
        maxFreq = (np.unravel_index(np.argmax(sg[:, f1:f2]), np.shape(sg[:, f1:f2]))[1] + f1) * self.sampleRate/2 /np.shape(sg)[1]
        return avgPower, deltaPower, energy, aggEntropy, avgEntropy, maxPower, maxFreq

    def get_Raven_robust_measurements(self, f1, f2):
        """ The second set of Raven features.
        1st, 2nd (centre), 3rd quartile, 5%, 95% frequency, inter-quartile range, bandwidth 90%
        Ditto for time

        t1, t2, f1, f2 are in pixels
        Compute the cumulative sums and then loop through looking for the 5 points where the left and right are in the correct proportion,
        i.e., the left is some percentage of the total.
        """

        sg = np.abs(np.where(self.sg == 0, 0.0, 10.0 * np.log10(self.sg)))

        # List of the match points (5%, 25%, 50%, 75%, 95%)
        list = [.05, .25, .5, .75, .95]

        sgf = np.sum(sg[:,f1:f2], axis=0)
        csf = np.cumsum(sgf)

        freqindices = np.zeros(5)
        index = 0
        i = 0
        while i < len(csf) and index < len(list):
            if csf[i] > list[index] * csf[-1]:
                freqindices[index] = i
                index += 1
            i += 1

        sgt = np.sum(sg[:,f1:f2], axis=1)
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
        # Maxfreq in sg is fs/2, so fs/2 /np.shape(sg)[1] is the value of 1 bin
        freqindices = (freqindices+f1) * self.sampleRate/2 /np.shape(sg)[1]
        # The time between two columns of the spectrogram is the increment divided by the sample rate
        timeindices = (timeindices+0) * self.incr / self.sampleRate
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
        mint = np.argmin(data[t1:t2])+t1 / fs
        maxa = np.max(data[t1:t2])
        maxt = np.argmax(data[t1:t2])+t1 / fs
        peaka = np.max(np.abs(data[t1:t2]))
        peakt = np.argmax(np.abs(data[t1:t2]))+t1 / fs
        rmsa = np.sqrt(np.sum(data[t1:t2]**2)/len(data[t1:t2]))
        # Filtered rmsa (bandpass filtered first)
        # TODO
        # Also? max bearing, peak correlation, peak lag
        return (mina, mint, maxa, maxt, peaka, peakt,rmsa)

    def computeCorrelation(self, a, b):
        """ Convolve a and b using FFT
        """

        c = signal.fftconvolve(a, b, mode='same')

        return c

    def get_SAP_features(self,data,fs,window_width=256,incr=128,K=2):
        """ Compute the Sound Analysis Pro features, i.e., Wiener entropy, spectral derivative, and their variants.
        Most of the code is in SignalProc.py"""
        sp = SignalProc.SignalProc(sampleRate=fs, window_width=256, incr=128)
    
        spectral_deriv, sg, freq_mod, wiener_entropy, mean_freq, contours = sp.spectral_derivative(data, fs,
                                            window_width=window_width, incr=incr, K=2, threshold=0.5, returnAll=True)
    
        goodness_of_pitch = sp.goodness_of_pitch(spectral_deriv, sg)
    
        # Now compute the continuity over time, freq as mean duration of contours in window, mean frequency range
        # TODO
    
        return spectral_deriv, goodness_of_pitch, freq_mod, contours, wiener_entropy, mean_freq


def loadFile(filename):
    wavobj = wavio.read(filename)
    sampleRate = wavobj.rate
    audiodata = wavobj.data

    # None of the following should be necessary for librosa
    if audiodata.dtype is not 'float':
        audiodata = audiodata.astype('float')   #/ 32768.0
    if np.shape(np.shape(audiodata))[0]>1:
        audiodata = audiodata[:,0]

    # if sampleRate != 16000:
    #     audiodata = librosa.core.audio.resample(audiodata, sampleRate, 16000)
    #     sampleRate=16000

    # # pre-process
    # sc = SupportClasses.preProcess(audioData=audiodata, sampleRate=sampleRate, species='Kiwi', df=False)
    # audiodata,sampleRate = sc.denoise_filter()
    return audiodata, sampleRate

def genCluterData(dir, duration=1, sampRate=16000):
    # male, female kiwi syllables from denoising chapter. They are in different lengths ~.87 sec min, therefore get the
    # middle 0.8 sec only to make the features with fixed len.
    f1 = open(dir + '/' + "mfcc.tsv", "w")
    for root, dirs, files in os.walk(str(dir)):
        for filename in files:
            if filename.endswith('.wav'):
                filename = root + '/' + filename
                # determin call type from the folder name
                type = root.split("\\")[-1]
                if type == 'male':
                    tgt = 0
                elif type == 'female':
                    tgt = 1
                # elif type == 'trill':
                #     tgt = 2
                data, fs = loadFile(filename)
                # resample where necessary
                if fs != sampRate:
                    data = librosa.core.audio.resample(data, fs, sampRate)
                    fs = sampRate
                # get the middle 'duration' secs
                middle_duration = int(duration * fs)
                middleIndex = int((len(data) - 1) / 2)
                if middle_duration < len(data):
                    data = data[int(middleIndex - middle_duration/2): int(middleIndex + middle_duration/2)]

                # # Wavelet energy
                # ws = WaveletSegment.WaveletSegment(spInfo=[])
                # wc = ws.computeWaveletEnergy(data=data, sampleRate=fs, nlevels=5, wpmode='new')
                # wc = wc.tolist()
                # wc = [i for sublist in wc for i in sublist]
                # print(filename, np.shape(wc))
                # f1.write("%s\t" % (filename))
                # for i in wc:
                #     f1.write("%f\t" % (i))
                # f1.write("%d\n" % (tgt))

                # MFCC
                f = Features(data, fs, 256, 128)
                mfcc = f.get_mfcc(n_mfcc=24).tolist()    # 96x22 matrix

                m = [i for sublist in mfcc for i in sublist]
                print(filename, np.shape(mfcc))
                # print(filename, np.shape(m))
                f1.write("%s\t" % (filename))
                for i in m:
                    f1.write("%f\t" % (i))
                f1.write("%d\n" % (tgt))
    f1.close()

# genCluterData('D:\AviaNZ\Sound_Files\Denoising_paper_data\demo', duration=0.8)


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

def testFeatures():
    wavobj = wavio.read('D:\AviaNZ\Sound_Files\Denoising_paper_data\Primary_dataset\kiwi\male\male1.wav')
    fs = wavobj.rate
    data = wavobj.data

    if data.dtype is not 'float':
        data = data.astype('float')         # / 32768.0

    if np.shape(np.shape(data))[0] > 1:
        data = data[:, 0]

    sp = SignalProc.SignalProc(sampleRate=fs, window_width=256, incr=128)
    # The next lines are to get a spectrogram that *should* precisely match the Raven one
    #sg = sp.spectrogram(data, multitaper=False, window_width=256, incr=128, window='Ones')
    #sg = sg ** 2
    sg = sp.spectrogram(data, sgType='Standard', window_width=256, incr=128, window='Hann')

    f = Features(data, fs, 256, 128)

    features = []
    # Loop over the segments (and time slices within?)
    mfcc = f.get_mfcc().tolist()
    # features.append(mfcc.tolist())
    we = f.get_WE()
    we = we.transpose().tolist()
    # how to combine features with different resolution?

    features.append([f.get_Raven_spectrogram_measurements(sg=sg,fs=fs,window_width=256,f1=0,f2=np.shape(sg)[1],t1=0,t2=np.shape(sg)[0]),f.get_Raven_robust_measurements(sg,fs,0,np.shape(sg)[1],0,np.shape(sg)[0]),f.get_Raven_waveform_measurements(data,fs,0,len(data)),f.wiener_entropy(sg)])

    # Will need to think about feature vector length for the librosa features, since they are on fixed windows
    f.get_chroma()
    f.get_mfcc()
    f.get_tonnetz()
    f.get_spectral_features()
    f.get_lpc(data,order=44)
    # DCT

# testFeatures()

# ---
def generateDataset(dir_src, feature, species, filemode, wpmode, dir_out):
    '''
    Generates different data sets for ML - variations of WE and MFCC
    Can be continuous wav files or extracted segments
    Continuous files + GT annotations OR
    Extracted segments + tell if they are TPs or not
    :param dir_src: path to the directory with recordings + GT annotations
    :param feature: 'WEraw_all', 'WEbp_all', 'WEd_all', 'WEbpd_all', 'MFCCraw_all', 'MFCCbp_all', 'MFCCd_all',
                    'MFCCbpd_all',
                    'WE+MFCCraw_all', 'WE+MFCCbp_all', 'WE+MFCCd_all', 'WE+MFCCbpd_all'
    :param species: species name (should be able to find species filter in dir_src)
    :param filemode: 'long', 'segpos', 'segneg'
    :param wpmode: 'pywt' or 'new' or 'aa'
    :param dir_out: path to the output dir

    :return: saves the data file to out-dir
    '''

    annotation = []
    if 'WE' in feature:
        nlevels = 5
        waveletCoefs = np.array([]).reshape(2**(nlevels+1)-2, 0)
    if 'MFCC' in feature:
        n_mfcc = 48
        n_bins = 8  # kakapo boom
        # n_bins = 32   # others
        delta = True
        if delta:
            MFCC = np.array([]).reshape(0, n_mfcc * 2 * n_bins)
        else:
            MFCC = np.array([]).reshape(0, n_mfcc * n_bins)
    # Find the species filter
    speciesData = json.load(open(os.path.join(dir_src, species + '.txt')))

    for root, dirs, files in os.walk(str(dir_src)):
        for file in files:
            if file.endswith('.wav') and os.stat(root + '/' + file).st_size != 0 and file[:-4] + '-res1.0sec.txt' in files or file.endswith('.wav') and filemode!='long' and os.stat(root + '/' + file).st_size > 150:
                opstartingtime = time.time()
                wavFile = root + '/' + file[:-4]
                print(wavFile)
                data, currentannotation, sampleRate = loadData(wavFile, filemode)

                ws = WaveletSegment.WaveletSegment(data=data, sampleRate=sampleRate)
                if feature == 'WEraw_all' or feature == 'MFCCraw_all' or feature == 'WE+MFCCraw_all':
                    data = ws.preprocess(speciesData, d=False, f=False)
                elif feature == 'WEbp_all' or feature == 'MFCCbp_all' or feature == 'WE+MFCCbp_all':
                    data = ws.preprocess(speciesData, d=False, f=True)
                elif feature == 'WEd_all' or feature == 'MFCCd_all' or feature == 'WE+MFCCd_all':
                    data = ws.preprocess(speciesData, d=True, f=False)
                elif feature == 'WEbpd_all' or feature == 'MFCCbpd_all' or feature == 'WE+MFCCbpd_all':
                    data = ws.preprocess(speciesData, d=True, f=True)

                # Compute energy in each WP node and store
                if 'WE' in feature:
                    currWCs = ws.computeWaveletEnergy(data=data, sampleRate=ws.sampleRate, nlevels=nlevels,
                                                      wpmode=wpmode)
                    waveletCoefs = np.column_stack((waveletCoefs, currWCs))
                if 'MFCC' in feature:
                    currMFCC = computeMFCC(data=data, sampleRate=ws.sampleRate, n_mfcc=n_mfcc, n_bins=n_bins, delta=delta)
                    MFCC = np.concatenate((MFCC, currMFCC), axis=0)
                annotation.extend(currentannotation)
                print("file loaded in", time.time() - opstartingtime)

    annotation = np.array(annotation)
    ann = np.reshape(annotation, (len(annotation), 1))
    if 'WE' in feature and 'MFCC' not in feature:
        # Prepare WC data and annotation targets into a matrix for saving
        WC = np.transpose(waveletCoefs)
        # ann = np.reshape(annotation,(len(annotation),1))
        MLdata = np.append(WC, ann, axis=1)
    elif 'MFCC' in feature and 'WE' not in feature:
        # ann = np.reshape(annotation, (len(annotation), 1))
        MLdata = np.append(MFCC, ann, axis=1)
    elif 'WE' in feature and 'MFCC' in feature:
        WC = np.transpose(waveletCoefs)
        WE_MFCC = np.append(WC, MFCC, axis=1)
        MLdata = np.append(WE_MFCC, ann, axis=1)
    np.savetxt(os.path.join(dir_out, species + '_' + feature + '.tsv'), MLdata, delimiter="\t")
    print("Directory loaded. %d/%d presence blocks found.\n" % (np.sum(annotation), len(annotation)))


# def loadData(fName, filemode):
#     '''
#     Load wav and GT for ML data set generation
#     :param fName:
#     :param filemode: 'long' or 'segpos' or 'segneg'
#     :return: audio data, GT, sampleRate
#     '''
#     filename = fName+'.wav'
#     filenameAnnotation = fName+'-res1.0sec.txt'
#     try:
#         wavobj = wavio.read(filename)
#     except:
#         print("unsupported file: ", filename)
#         pass
#     sampleRate = wavobj.rate
#     data = wavobj.data
#     if data.dtype is not 'float':
#         data = data.astype('float') #/ 32768.0
#     if np.shape(np.shape(data))[0]>1:
#         data = np.squeeze(data[:,0])
#     n = math.ceil(len(data)/sampleRate)
#
#     if filemode=='long':
#         # GT from the txt file
#         fileAnnotation = []
#         with open(filenameAnnotation) as f:
#             reader = csv.reader(f, delimiter="\t")
#             d = list(reader)
#         if d[-1]==[]:
#             d = d[:-1]
#         if len(d) != n:
#             print("ERROR: annotation length %d does not match file duration %d!" %(len(d), n))
#             return
#         # for each second, store 0/1 presence:
#         sum = 0
#         for row in d:
#             fileAnnotation.append(int(row[1]))
#             sum += int(row[1])
#     elif filemode=='segpos':
#         fileAnnotation = np.ones((math.ceil(len(data) / sampleRate), 1))
#     elif filemode=='segneg':
#         fileAnnotation = np.zeros((math.ceil(len(data) / sampleRate), 1))
#     return data, np.array(fileAnnotation), sampleRate


def computeMFCC(data, sampleRate, n_mfcc, n_bins, delta):
    '''
    Compute MFCC for each second of data and return as a matrix
    :param data: audio data
    :param sampleRate: sample rate
    :param delta: True/False
    :return: MFCC metrix
    '''
    n = math.ceil(len(data) / sampleRate)
    if delta:
        mfcc = np.zeros((n, n_mfcc * 2 * n_bins))
    else:
        mfcc=np.zeros((n, n_mfcc*n_bins))
    i = 0
    for t in range(n):
        end = min(len(data), (t + 1) * sampleRate)
        if end == len(data) and len(data) % sampleRate != 0:
            continue
        mfcc1 = librosa.feature.mfcc(y=data[t * sampleRate:end], sr=sampleRate, n_mfcc=n_mfcc, n_fft=2048,
                                        hop_length=512)  # n_fft=10240, hop_length=2560
        if delta:
            if n_bins == 8:
                mfcc1_delta = librosa.feature.delta(mfcc1, width=5)
            else:
                mfcc1_delta = librosa.feature.delta(mfcc1)
            mfcc1 = np.concatenate((mfcc1, mfcc1_delta), axis=0)
        # Normalize
        mfcc1 -= np.mean(mfcc1, axis=0)
        mfcc1 /= np.max(np.abs(mfcc1), axis=0)
        mfcc1 = np.reshape(mfcc1, np.shape(mfcc1)[0] * np.shape(mfcc1)[1])
        mfcc1 = mfcc1.flatten()
        mfcc[i, :] = mfcc1
        i += 1
    return mfcc

