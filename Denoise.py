
import numpy as np
import pywt
from scipy.io import wavfile
import pylab as pl

# TODO:
# Check on more data

class Denoise:

    def __init__(self,data=[],sampleRate=0):
        self.maxsearch=10
        self.maxlevel = 4
        if data != []:
            self.data = data
            self.sampleRate = sampleRate

    def loadData(self):
        #self.sampleRate, self.data = wavfile.read('../Birdsong/more1.wav')
        self.sampleRate, self.data = wavfile.read('male1.wav')
        # The constant is for normalisation (2^15, as 16 bit numbers)
        self.data = self.data.astype('float')/32768.0

    # Need to store the tree this way
    class node():
        pass

    def ShannonEntropy(self,s):
        e = -s[np.nonzero(s)]**2 * np.log(s[np.nonzero(s)]**2)
        #e = np.where(s==0,0,-s**2*np.log(s**2))
        return np.sum(e)

    # Recurse down the tree until either maxsearch levels or sections are too short, or Entropy rule broken
    # Two versions -- with wavelets, or wavelet packets.
    def makeTree(self,data,level):
        newNode = Denoise.node()
        newNode.level=level
        newNode.Entropy = self.ShannonEntropy(data)
        newNode.A, newNode.D = pywt.dwt(data, 'dmey')
        #print np.shape(data)[0], self.ShannonEntropy(newNode.A) + self.ShannonEntropy(newNode.D), newNode.Entropy, level
        sA = self.ShannonEntropy(newNode.A)
        sD = self.ShannonEntropy(newNode.D)
        # Should this next line be max or sum?
        if (np.shape(newNode.A)[0]==1 or (sA + sD >= newNode.Entropy) or (level>self.maxsearch)):
            maxLevelL = level
            maxLevelR = level
        else:
            if sA > 0:
                newNode.left,maxLevelL = self.makeTree(newNode.A,level+1)
            else:
                newNode.left = None
                maxLevelL = level
            if sD > 0:
                newNode.right, maxLevelR = self.makeTree(newNode.D, level+1)
            else:
                newNode.right = None
                maxLevelR = level
        return newNode, max(maxLevelL,maxLevelR)


    def makeTree_wpt(self,data,level):
        newNode = Denoise.node()
        newNode.level=level
        newNode.Entropy = self.ShannonEntropy(data)
        wpt = pywt.WaveletPacket(data=self.data, wavelet='dmey', mode='symmetric',maxlevel=self.maxlevel)
        newNode.A = wpt['a'].data
        newNode.D = wpt['d'].data
        #print np.shape(data)[0], self.ShannonEntropy(newNode.A), self.ShannonEntropy(newNode.D), newNode.Entropy
        sA = self.ShannonEntropy(newNode.A)
        sD = self.ShannonEntropy(newNode.D)
        # Should this next line be max or sum?
        if np.shape(newNode.A)[0]==1 or (sA + sD >= newNode.Entropy) or level>self.maxsearch:
            stopFlag = True
            maxLevelL = level
            maxLevelR = level
        else:
            if sA > 0:
                newNode.left,maxLevelL = self.makeTree_wpt(newNode.A,level+1)
            else:
                newNode.left = None
                maxLevelL = level
            if sD > 0:
                newNode.right, maxLevelR = self.makeTree_wpt(newNode.D, level+1)
            else:
                newNode.right = None
                maxLevelR = level
        return newNode, max(maxLevelL,maxLevelR)

    def testTree(self):
        #maxsearch=14
        level = 0
        #maxLevel = 0
        root, maxLevel = self.makeTree(self.data,level)
        print(maxLevel)

        print "====="
        level = 0
        maxLevel = 0
        root, maxLevel = self.makeTree_wpt(self.data,level)
        print(maxLevel)
        print "====="

    def denoise(self):
        level = 0
        root, self.maxlevel = self.makeTree_wpt(self.data,level)
        # CHEAT!!!!
        self.maxlevel = max(self.maxlevel,4)

        # Use maxLevel (rather wasteful this!) to make a new wavelet packet decomposition
        wp = pywt.WaveletPacket(data=self.data, wavelet='dmey', mode='symmetric',maxlevel=self.maxlevel)
        out = wp.reconstruct(update=False)

        det1 = wp['d'].data
        # Note magic conversion number
        sigma = np.median(np.abs(det1)) / 0.6745
        threshold = 4.5*sigma
        for level in range(self.maxlevel):
            for n in wp.get_level(level, 'natural'):
                # Hard thresholding
                #n.data = np.where(np.abs(n.data)<threshold,0.0,n.data)
                # Soft thresholding
                n.data = np.sign(n.data)*np.maximum((np.abs(n.data)-threshold),0.0)

        wData = wp.reconstruct()
        #pl.plot(wp.reconstruct())
        #pl.show()

        # Bandpass filter
        import scipy.signal as signal
        ripple_db = 80.0
        width = 0.11
        ntaps, beta = signal.kaiserord(ripple_db, width)
        taps = signal.firwin(ntaps,cutoff = [500,8000], window=('kaiser', beta),pass_zero=False,nyq=self.sampleRate/2.0)
        self.fwData = signal.lfilter(taps, 1.0, wData)
        return self.fwData
        #pl.plot(self.data)
        #pl.plot(self.fwData)


    def plot(self):
        # Spectrogram plot
        fig2 = pl.figure()
        cmap = pl.cm.gray
        ax1 = fig2.add_subplot(211)
        ax1.specgram(self.data, NFFT=64, noverlap=32, cmap=cmap)
        ax2 = fig2.add_subplot(212)
        ax2.specgram(self.fwData, NFFT=64, noverlap=32, cmap=cmap)
        pl.show()

    def writefile(self):
        wavfile.write('male1d.wav',self.sampleRate, self.fwData)
    # Write audio

    def splitFile5mins(self, name):
        # Nirosha wants to split files that are long (15 mins) into 5 min segments
        self.sampleRate, self.audiodata = wavfile.read(name)
        nsamples = np.shape(self.audiodata)[0]
        lengthwanted = self.sampleRate * 60 * 5
        count = 0
        while (count + 1) * lengthwanted < nsamples:
            data = self.audiodata[count * lengthwanted:(count + 1) * lengthwanted]
            filename = name[:-4] + '_' +str(count) + name[-4:]
            print filename
            wavfile.write(filename, self.sampleRate, data)
            count += 1
        data = self.audiodata[(count) * lengthwanted:]
        filename = name[:-4] + '_' + str((count)) + name[-4:]
        print filename
        wavfile.write(filename, self.sampleRate, data)

#a = Denoise()
#a.splitFile5mins('ST0026.wav')

# a.loadData()
# # a.testTree()
# a.denoise()
# a.plot()
# a.writefile()