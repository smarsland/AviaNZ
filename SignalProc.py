# Version 0.2 30/5/16
# Author: Stephen Marsland

import numpy as np
import pywt
from scipy.io import wavfile
import pylab as pl
import matplotlib

# TODO:
# Denoising needs work
# Add in bandpass filtering
# Some tidying needed
# What else should be added into here?

class SignalProc:
    # This class implements various signal processing algorithms for the AviaNZ interface
    def __init__(self,data=[],sampleRate=0,window_width=256,incr=128,maxSearchDepth=20):
        self.window_width=window_width
        self.incr=incr
        self.maxsearch=maxSearchDepth
        if data != []:
            self.data = data
            self.sampleRate = sampleRate

    def spectrogram(self,t,window='Hanning'):
        # Compute the spectrogram from amplitude data
        from scipy.fftpack import fft

        if t is None:
            print ("Error")

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

        sg = np.zeros((self.window_width / 2, np.ceil(len(t) / self.incr)))
        counter = 1

        for start in range(0, len(t) - self.window_width, self.incr):
            # Multiply data with window function, take Fourier transform and plot log10 version
            windowedfn = window * t[start:start + self.window_width]
            ft = fft(windowedfn)
            ft = ft * np.conj(ft)
            sg[:, counter] = np.real(ft[self.window_width / 2:])
            counter += 1
        # Note that the last little bit (up to window_width) is lost. Can't just add it in since there are fewer points

        sg = 10.0 * np.log10(sg)
        return sg

    def ShannonEntropy(self,s):
        # Compute the Shannon entropy of data
        # TODO: Work out why this sometimes has a log(0) error
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

    def denoise(self,threshold='soft'):
        # Perform wavelet denoising. Can use soft or hard thresholding
        level = 0
        self.maxlevel = self.BestLevel()
        print self.maxlevel

        # TODO: reuse previous tree instead of making new one!
        self.wp = pywt.WaveletPacket(data=self.data, wavelet='dmey', mode='symmetric',maxlevel=self.maxlevel)

        # nlevels = self.maxsearch
        # while nlevels > self.maxlevel:
        #     for n in self.wp.get_leaf_nodes():
        #         del self.wp[n.path]
        #     nlevels -= 1

        det1 = self.wp['d'].data
        # Note magic conversion number
        sigma = np.median(np.abs(det1)) / 0.6745
        threshold = 4.5*sigma
        for level in range(self.maxlevel):
            for n in self.wp.get_level(level, 'natural'):
                if threshold = 'hard':
                    # Hard thresholding
                    n.data = np.where(np.abs(n.data)<threshold,0.0,n.data)
                else:
                    # Soft thresholding
                    n.data = np.sign(n.data)*np.maximum((np.abs(n.data)-threshold),0.0)

        self.wData = self.wp.data
        #self.wp.reconstruct(update=False)

        # Commented out as I don't see the benefit. And don't know how to pick width
        # Bandpass filter
        # import scipy.signal as signal
        # nyquist = self.sampleRate/2.0
        # ripple_db = 80.0
        # width = 1.0/nyquist
        # ntaps, beta = signal.kaiserord(ripple_db, width)
        # taps = signal.firwin(ntaps,cutoff = [500/nyquist,8000/nyquist], window=('kaiser', beta),pass_zero=False)
        # self.fwData = signal.lfilter(taps, 1.0, self.wData)

        return self.wData

    def writefile(self,name):
        # Save a sound file for after denoising
        # Need them to be 16 bit integers
        self.wData *= 32768.0
        self.wData = self.wData.astype('int16')
        wavfile.write(name,self.sampleRate, self.wData)

    def loadData(self):
        # Load a sound file and normalise it
        # self.sampleRate, self.data = wavfile.read('../Birdsong/more1.wav')
        # self.sampleRate, self.data = wavfile.read('../Birdsong/Denoise/Primary dataset/kiwi/female/female1.wav')
        #self.sampleRate, self.data = wavfile.read('ruru.wav')
        self.sampleRate, self.data = wavfile.read('tril1.wav')
        # self.sampleRate, self.data = wavfile.read('male1.wav')
        # The constant is for normalisation (2^15, as 16 bit numbers)
        self.data = self.data.astype('float') / 32768.0

def test():
    #pl.ion()
    a = SignalProc()
    #a.splitFile5mins('ST0026.wav')

    a.loadData()
    #a.play()
    #a.testTree()
    sg = a.spectrogram(a.data)
    pl.figure()
    pl.imshow(sg,cmap='gray')
    a.denoise()
    sgn = a.spectrogram(a.wData)
    pl.figure()
    pl.imshow(sgn,cmap='gray')
    pl.figure()
    pl.plot(a.wData)
    #a.plot()
    #a.play()
    a.writefile('out.wav')
    pl.show()
#test()
#pl.show()

#pl.ion()