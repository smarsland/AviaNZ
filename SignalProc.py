# Version 0.3 20/7/16
# Author: Stephen Marsland

import numpy as np
import pywt
from scipy.io import wavfile
import pylab as pl
import matplotlib

# TODO:
# Denoising needs work
# Add in bandpass filtering
# Also downsampling (use librosa)
# Some tidying needed
# Test the different windows, play with threshold multiplier -> how to set? Look up log amplitude scaling
# Compute the spectral derivatives??
# What else should be added into here?

class SignalProc:
    # This class implements various signal processing algorithms for the AviaNZ interface
    def __init__(self,data=[],sampleRate=0,window_width=256,incr=128,maxSearchDepth=20,thresholdMultiplier=4.5):
        self.window_width=window_width
        self.incr=incr
        self.maxsearch=maxSearchDepth
        self.thresholdMultiplier = thresholdMultiplier
        if data != []:
            self.data = data
            self.sampleRate = sampleRate

    def setNewData(self,data,fs):
        self.data = data
        self.sampleRate = fs

    def set_width(self,window_width,incr):
        self.window_width = window_width
        self.incr = incr

    def spectrogram(self,t,window='Hanning',multitaper=False):
        # Compute the spectrogram from amplitude data
        from scipy.fftpack import fft

        if t is None:
            print ("Error")

        if multitaper:
            from spectrum import dpss, pmtm

        # Set of window options
        if window=='Hanning':
            # This is the Hanning window
            window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.window_width) / (self.window_width - 1)))
        elif window=='Parzen':
            # Parzen (self.window_width even)
            n = np.arange(self.window_width) - 0.5*self.window_width
            window = np.where(np.abs(n)<0.25*self.window_width,1 - 6*(n/(0.5*self.window_width))**2*(1-np.abs(n)/(0.5*self.window_width)), 2*(1-np.abs(n)/(0.5*self.window_width))**3)
        elif window=='Welch':
            # Welch
            window = 1.0 - ((np.arange(self.window_width) - 0.5*(self.window_width-1))/(0.5*(self.window_width-1)))**2
        elif window=='Hamming':
            # Hamming
            alpha = 0.54
            beta = 1.-alpha
            window = alpha - beta*np.cos(2 * np.pi * np.arange(self.window_width) / (self.window_width - 1))
        elif window=='Blackman':
            # Blackman
            alpha = 0.16
            a0 = 0.5*(1-alpha)
            a1 = 0.5
            a2 = 0.5*alpha
            window = a0 - a1*np.cos(2 * np.pi * np.arange(self.window_width) / (self.window_width - 1)) + a2*np.cos(4 * np.pi * np.arange(self.window_width) / (self.window_width - 1))
        elif window=='BlackmanHarris':
            # Blackman-Harris
            a0 = 0.358375
            a1 = 0.48829
            a2 = 0.14128
            a3 = 0.01168
            window = a0 - a1*np.cos(2 * np.pi * np.arange(self.window_width) / (self.window_width - 1)) + a2*np.cos(4 * np.pi * np.arange(self.window_width) / (self.window_width - 1)) - a3*np.cos(6 * np.pi * np.arange(self.window_width) / (self.window_width - 1))
        else:
            print "unknown window, using Hanning"
            window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.window_width) / (self.window_width - 1)))
        sg = np.zeros((self.window_width / 2, int(np.ceil(len(t) / self.incr))))
        counter = 0


        for start in range(0, len(t) - self.window_width+1, self.incr):
            # Multiply data with window function, take Fourier transform, multiply by complex conjugate, take real part
            if multitaper:
                S = pmtm(t[start:start+self.window_width], NW=2.5, k=4, show=False)
                sg[:, counter:counter+1] = S[self.window_width / 2:]
            else:
                windowedfn = window * t[start:start + self.window_width]
                ft = fft(windowedfn)
                ft = ft * np.conj(ft)
                sg[:, counter] = np.real(ft[self.window_width / 2:])
            counter += 1
        # Note that the last little bit (up to window_width) is lost. Can't just add it in since there are fewer points
        # Returns the fft. For plotting purposes, want it in decibels (10*np.log10(sg))
        #sg = 10.0 * np.log10(sg)
        return sg

    def ShannonEntropy(self,s):
        # Compute the Shannon entropy of data
        e = -s[np.nonzero(s)]**2 * np.log(s[np.nonzero(s)]**2)
        #e = np.where(s==0,0,-s**2*np.log(s**2))
        return np.sum(e)

    def BestLevel(self):
        # Compute the best level for the wavelet packet decomposition by using the Shannon entropy
        previouslevelmaxE = self.ShannonEntropy(self.data)
        self.wp = pywt.WaveletPacket(data=self.data, wavelet='dmey', mode='symmetric', maxlevel=self.maxsearch)
        level = 1
        currentlevelmaxE = np.max([self.ShannonEntropy(n.data) for n in self.wp.get_level(level, "freq")])
        while currentlevelmaxE < previouslevelmaxE and level<self.maxsearch:
            previouslevelmaxE = currentlevelmaxE
            level += 1
            currentlevelmaxE = np.max([self.ShannonEntropy(n.data) for n in self.wp.get_level(level, "freq")])

        return level-1

    def waveletDenoise(self,data=None,thresholdType='soft',threshold=None,maxlevel=None,bandpass=False,wavelet='dmey'):
        # Perform wavelet denoising. Can use soft or hard thresholding
        if data is None:
            data = self.data
        if maxlevel is None:
            self.maxlevel = self.BestLevel()
        else:
            self.maxlevel = maxlevel
        print self.maxlevel
        if threshold is not None:
            self.thresholdMultiplier = threshold

        wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric',maxlevel=self.maxlevel)

        # nlevels = self.maxsearch
        # while nlevels > self.maxlevel:
        #     for n in self.wp.get_leaf_nodes():
        #         del self.wp[n.path]
        #     nlevels -= 1

        det1 = wp['d'].data
        # Note magic conversion number
        sigma = np.median(np.abs(det1)) / 0.6745
        threshold = self.thresholdMultiplier*sigma
        for level in range(self.maxlevel):
            for n in wp.get_level(level, 'natural'):
                if thresholdType == 'hard':
                    # Hard thresholding
                    n.data = np.where(np.abs(n.data)<threshold,0.0,n.data)
                else:
                    # Soft thresholding
                    n.data = np.sign(n.data)*np.maximum((np.abs(n.data)-threshold),0.0)

        self.wData = wp.data
        #self.wp.reconstruct(update=False)

        return self.wData

    def bandpassFilter(self,data=None,start=500,end=8000):
        # Bandpass filter
        import scipy.signal as signal
        if data is None:
            data = self.data
        nyquist = self.sampleRate/2.0
        #ripple_db = 80.0
        #width = 1.0/nyquist
        #ntaps, beta = signal.kaiserord(ripple_db, width)
        ntaps = 128
        #taps = signal.firwin(ntaps,cutoff = [500/nyquist,8000/nyquist], window=('kaiser', beta),pass_zero=False)
        taps = signal.firwin(ntaps, cutoff=[start / nyquist, end / nyquist], window=('hamming'), pass_zero=False)
        fData = signal.lfilter(taps, 1.0, data)

        return fData

    def medianFilter(self,data=None,width=11):
        # Median Filtering
        # Uses smaller width windows at edges to remove edge effects
        # TODO: Use abs rather than pure median?
        if data is None:
            data = self.data
        mData = np.zeros(len(data))
        for i in range(width,len(data)-width):
            mData[i] = np.median(data[i-width:i+width])
        for i in range(len(data)):
            wid = min(i,len(data)-i,width)
            mData[i] = np.median(data[i - wid:i + wid])

        return mData

    def writeFile(self,name):
        # Save a sound file for after denoising
        # Need them to be 16 bit integers
        self.wData *= 32768.0
        self.wData = self.wData.astype('int16')
        wavfile.write(name,self.sampleRate, self.wData)

    def loadData(self,fileName):
        # Load a sound file and normalise it
        self.sampleRate, self.data = wavfile.read(fileName)
        # self.sampleRate, self.data = wavfile.read('../Birdsong/more1.wav')
        # self.sampleRate, self.data = wavfile.read('../Birdsong/Denoise/Primary dataset/kiwi/female/female1.wav')
        #self.sampleRate, self.data = wavfile.read('ruru.wav')
        #self.sampleRate, self.data = wavfile.read('tril1.wav')
        # self.sampleRate, self.data = wavfile.read('male1.wav')
        # The constant is for normalisation (2^15, as 16 bit numbers)
        self.data = self.data.astype('float') / 32768.0

def denoiseFile(fileName,thresholdMultiplier):
    sp = SignalProc(thresholdMultiplier=thresholdMultiplier)
    sp.loadData(fileName)
    sp.waveletDenoise()
    sp.writeFile(fileName[:-4]+'denoised'+str(sp.thresholdMultiplier)+fileName[-4:])

def medianClip(sg):
    rowmedians = np.median(sg, axis=1)
    colmedians = np.median(sg, axis=0)
    #print np.shape(rowmedians), np.shape(colmedians), np.shape(sg)
    clipped = np.zeros(np.shape(sg))
    for i in range(np.shape(sg)[0]):
        for j in range(np.shape(sg)[1]):
            if (sg[i, j] > 3 * rowmedians[i] and sg[i, j] > 3 * colmedians[j]):
                clipped[i, j] = 1.0
    return clipped

def test():
    #pl.ion()
    a = SignalProc()
    #a.splitFile5mins('ST0026.wav')

    a.loadData()
    #a.play()
    #a.testTree()
    sg = a.spectrogram(a.data)
    pl.figure()
    pl.imshow(10.0*np.log10(sg),cmap='gray')
    a.waveletDenoise()
    sgn = a.spectrogram(a.wData)
    pl.figure()
    pl.imshow(10.0*np.log10(sgn),cmap='gray')
    pl.figure()
    pl.plot(a.wData)
    #a.plot()
    #a.play()
    a.writefile('out.wav')
    pl.show()


def show():
    #pl.ion()
    a = SignalProc()
    #a.loadData('Sound Files/male1d.wav')
    a.loadData('Sound Files/kiwi.wav')
    a.data = a.data[:60000,0]
    sg = a.spectrogram(a.data)
    #pl.figure()
    #pl.plot(a.data)
    pl.figure()
    pl.imshow(10.0*np.log10(sg),cmap='gray_r')
    pl.figure()
    pl.imshow(10.0*np.log10(medianClip(sg)),cmap='gray')
    pl.show()

#show()
#pl.show()
#test()
#pl.show()

#pl.ion()

#denoiseFile('tril1.wav',1.5)
#denoiseFile('tril1.wav',2.5)
#denoiseFile('tril1.wav',3.5)
#denoiseFile('tril1.wav',4.0)
#denoiseFile('tril1.wav',4.5)
#denoiseFile('tril1.wav',5.0)
