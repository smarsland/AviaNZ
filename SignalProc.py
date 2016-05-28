
import numpy as np
import pywt
from scipy.io import wavfile
import pylab as pl
import matplotlib

class SignalProc:

    def __init__(self,data=[],sampleRate=0,window_width=256,incr=128,maxSearchDepth=20):
        self.window_width=window_width
        self.incr=incr
        self.maxsearch=maxSearchDepth
        if data != []:
            self.data = data
            self.sampleRate = sampleRate

    def spectrogram(self,t):
        from scipy.fftpack import fft

        if t is None:
            print ("Error")

        # This is the Hanning window
        # TODO: add more windows
        hanning = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.window_width) / (self.window_width + 1)))

        sg = np.zeros((self.window_width / 2, np.ceil(len(t) / self.incr)))
        counter = 1

        for start in range(0, len(t) - self.window_width, self.incr):
            window = hanning * t[start:start + self.window_width]
            ft = fft(window)
            ft = ft * np.conj(ft)
            sg[:, counter] = np.real(ft[self.window_width / 2:])
            counter += 1
        # Note that the last little bit (up to window_width) is lost. Can't just add it in since there are fewer points

        sg = 10.0 * np.log10(sg)
        return sg

    def ShannonEntropy(self,s):
        # TODO: Work out why this sometimes has a log(0) error
        e = -s[np.nonzero(s)]**2 * np.log(s[np.nonzero(s)]**2)
        #e = np.where(s==0,0,-s**2*np.log(s**2))
        return np.sum(e)

    def BestLevel(self):
        previouslevelmaxE = self.ShannonEntropy(self.data)
        self.wp = pywt.WaveletPacket(data=self.data, wavelet='dmey', mode='symmetric', maxlevel=self.maxsearch)
        level = 1
        currentlevelmaxE = np.max([self.ShannonEntropy(n.data) for n in self.wp.get_level(level, "freq")])
        while currentlevelmaxE < previouslevelmaxE and level<self.maxsearch:
            previouslevelmaxE = currentlevelmaxE
            level += 1
            currentlevelmaxE = np.max([self.ShannonEntropy(n.data) for n in self.wp.get_level(level, "freq")])

        return level-1

    def denoise(self):
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
        threshold = 3.5*sigma
        for level in range(self.maxlevel):
            for n in self.wp.get_level(level, 'natural'):
                # Hard thresholding
                #n.data = np.where(np.abs(n.data)<threshold,0.0,n.data)
                # Soft thresholding
                n.data = np.sign(n.data)*np.maximum((np.abs(n.data)-threshold),0.0)

        self.wData = self.wp.reconstruct(update=False)

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
        # Need them to be 16 bit
        self.wData *= 32768.0
        self.wData = self.wData.astype('int16')
        wavfile.write(name,self.sampleRate, self.wData)

    def loadData(self):
        # self.sampleRate, self.data = wavfile.read('../Birdsong/more1.wav')
        # self.sampleRate, self.data = wavfile.read('../Birdsong/Denoise/Primary dataset/kiwi/female/female1.wav')
        self.sampleRate, self.data = wavfile.read('ruru.wav')
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
    a.denoise()
    a.plot()
    #a.play()
    #a.writefile('out.wav')
    pl.show()
#test()
#pl.show()