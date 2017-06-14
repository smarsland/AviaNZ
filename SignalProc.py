# Version 0.4 28/4/17
# Author: Stephen Marsland

import numpy as np
import pywt
# from scipy.io import wavfile
import wavio
import scipy.signal as signal
#import pylab as pl
import string

# TODO:
# Denoising needs work
# Bandpass filtering needs work
# Add downsampling (use librosa)
# Some tidying needed
# For spectrogram: Test the different windows, play with threshold multiplier -> how to set? Look up log amplitude scaling
# What else should be added into here?

class SignalProc:
    # This class implements various signal processing algorithms for the AviaNZ interface
    # Most important features are computing the spectrogram denoising with wavelets, and computing cross-correlation (findMatches)

    def __init__(self,data=[],sampleRate=0,window_width=256,incr=128,maxSearchDepth=20,thresholdMultiplier=4.5):
        self.window_width=window_width
        self.incr=incr
        self.maxsearch=maxSearchDepth
        self.thresholdMultiplier = thresholdMultiplier
        if data != []:
            self.data = data
            self.sampleRate = sampleRate

    def setNewData(self,data,sampleRate):
        # To be called when a new sound file is loaded
        self.data = data
        self.sampleRate = sampleRate

    def set_width(self,window_width,incr):
        self.window_width = window_width
        self.incr = incr

    def SnNR(self,startSignal,startNoise):
        pS = np.sum(self.data[startSignal:startSignal+self.length]**2)/self.length
        pN = np.sum(self.data[startNoise:startNoise+self.length]**2)/self.length
        return 10.*np.log10(pS/pN)

    def spectrogram(self,data,sampleRate=0,window='Hann',mean_normalise=True,onesided=True,multitaper=False,need_even=False):
        # Compute the spectrogram from amplitude data
        # Note that this returns the power spectrum (not the density) and without the log10.
        # Also, it's the absolute value of the FT, not FT*conj(FT), 'cos it seems to give better discimination
        # Can compute the multitaper version, but it's slow
        # Essential for median clipping, though
        # This version is faster than the default versions in pylab and scipy.signal
        # TODO: Note that using librosa to load files changes the values in the spectrogram, and this matters since they are normalised, so log makes things negative
        if data is None:
            print ("Error")

        # Set of window options
        if window=='Hann':
            # This is the Hann window
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
            print "unknown window, using Hann"
            window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.window_width) / (self.window_width - 1)))

        if mean_normalise:
            data -= data.mean()

        if multitaper:
            from spectrum import dpss, pmtm
            [tapers, eigen] = dpss(self.window_width, 2.5, 4)
            counter = 0
            sg = np.zeros((int(np.ceil(float(len(data)) / self.incr)),self.window_width / 2))
            for start in range(0, len(data) - self.window_width, self.incr):
                S = pmtm(data[start:start + self.window_width], e=tapers, v=eigen, show=False)
                sg[counter:counter + 1,:] = S[self.window_width / 2:].T
                counter += 1
            sg = np.fliplr(sg)
        else:
            starts = range(0, len(data) - self.window_width, self.incr)
            if need_even:
                starts = np.hstack((starts, np.zeros((self.window_width - len(data) % self.window_width))))

            ft = np.zeros((len(starts), self.window_width))
            for i in starts:
                ft[i / self.incr, :] = window * data[i:i + self.window_width]
            ft = np.fft.fft(ft)
            if onesided:
                #sg = np.absolute(ft[:, self.window_width / 2:]).T
                sg = np.absolute(ft[:, :self.window_width / 2])
            else:
                sg = np.absolute(ft)
            #sg = (ft*np.conj(ft))[:,self.window_width / 2:].T
        return sg

    def show_invS(self):
        print "Inverting spectrogam with window ", self.window_width, " and increment ", int(self.window_width/4.)
        oldIncr = self.incr
        self.incr = int(self.window_width/4.)
        sg = self.spectrogram(self.data)
        # sgi = self.invertSpectrogram(sg,self.window_width,self.incr)
        # self.incr = oldIncr
        # sg = self.spectrogram(sgi)
        # sgi = sgi.astype('int16')
        # wavfile.write('test.wav',self.sampleRate, sgi)
        # wavio.write('test.wav',sgi,self.sampleRate)
        return sg

    def invertSpectrogram(self,sg,window_width=256,incr=64,nits=10):
        # Assumes that this is the plain (not power) spectrogram
        import copy
        # Make the spectrogram two-sided and make the values small
        sg = np.concatenate([sg, sg[:, ::-1]], axis=1)

        sg_best = copy.deepcopy(sg)
        for i in range(nits):
            sgi = self.invert_spectrogram(sg_best, incr, calculate_offset=True,set_zero_phase=(i==0))
        est = self.spectrogram(sgi, onesided=False,need_even=True)
        phase = est / np.maximum(np.max(sg)/1E8, np.abs(est))

        sg_best = sg * phase[:len(sg)]
        sgi = self.invert_spectrogram(sg_best, incr, calculate_offset=True,set_zero_phase=False)
        return np.real(sgi)

    def invert_spectrogram(self,sg, incr, calculate_offset=True, set_zero_phase=True):
        """
        Under MSR-LA License
        Based on MATLAB implementation from Spectrogram Inversion Toolbox
        References
        ----------
        D. Griffin and J. Lim. Signal estimation from modified
        short-time Fourier transform. IEEE Trans. Acoust. Speech
        Signal Process., 32(2):236-243, 1984.
        Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
        Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
        Adelaide, 1994, II.77-80.
        Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
        Estimation from Modified Short-Time Fourier Transform
        Magnitude Spectra. IEEE Transactions on Audio Speech and
        Language Processing, 08/2007.
        """
        size = int(np.shape(sg)[1] // 2)
        wave = np.zeros((np.shape(sg)[0] * incr + size))
        # Getting overflow warnings with 32 bit...
        wave = wave.astype('float64')
        total_windowing_sum = np.zeros((np.shape(sg)[0] * incr + size))
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(size) / (size - 1)))

        est_start = int(size // 2) - 1
        est_end = est_start + size
        for i in range(sg.shape[0]):
            wave_start = int(incr * i)
            wave_end = wave_start + size
            if set_zero_phase:
                spectral_slice = sg[i].real + 0j
            else:
                # already complex
                spectral_slice = sg[i]

            # Don't need fftshift due to different impl.
            wave_est = np.real(np.fft.ifft(spectral_slice))[::-1]
            if calculate_offset and i > 0:
                offset_size = size - incr
                if offset_size <= 0:
                    print("WARNING: Large step size >50\% detected! "
                          "This code works best with high overlap - try "
                          "with 75% or greater")
                    offset_size = incr
                offset = self.xcorr_offset(wave[wave_start:wave_start + offset_size],
                                      wave_est[est_start:est_start + offset_size])
            else:
                offset = 0
            wave[wave_start:wave_end] += window * wave_est[
                est_start - offset:est_end - offset]
            total_windowing_sum[wave_start:wave_end] += window
        wave = np.real(wave) / (total_windowing_sum + 1E-6)
        return wave

    def xcorr_offset(self,x1, x2):
        """
        Under MSR-LA License
        Based on MATLAB implementation from Spectrogram Inversion Toolbox
        References
        ----------
        D. Griffin and J. Lim. Signal estimation from modified
        short-time Fourier transform. IEEE Trans. Acoust. Speech
        Signal Process., 32(2):236-243, 1984.
        Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
        Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
        Adelaide, 1994, II.77-80.
        Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
        Estimation from Modified Short-Time Fourier Transform
        Magnitude Spectra. IEEE Transactions on Audio Speech and
        Language Processing, 08/2007.
        """
        x1 = x1 - x1.mean()
        x2 = x2 - x2.mean()
        frame_size = len(x2)
        half = frame_size // 2
        corrs = np.convolve(x1.astype('float32'), x2[::-1].astype('float32'))
        corrs[:half] = -1E30
        corrs[-half:] = -1E30
        offset = corrs.argmax() - len(x1)
        return offset

    # Functions for denoising (wavelet and bandpass filtering)
    def ShannonEntropy(self,s):
        # Compute the Shannon entropy of data
        e = s[np.nonzero(s)]**2 * np.log(s[np.nonzero(s)]**2)
        #e = np.where(s==0,0,-s**2*np.log(s**2))
        return np.sum(e)

    def BestLevel(self,wavelet):
        # Compute the best level for the wavelet packet decomposition by using the Shannon entropy
        previouslevelmaxE = self.ShannonEntropy(self.data)
        #print previouslevelmaxE
        self.wp = pywt.WaveletPacket(data=self.data, wavelet=wavelet, mode='symmetric', maxlevel=self.maxsearch)
        level = 1
        currentlevelmaxE = np.max([self.ShannonEntropy(n.data) for n in self.wp.get_level(level, "freq")])
        #print currentlevelmaxE
        while currentlevelmaxE < previouslevelmaxE and level<self.maxsearch:
            previouslevelmaxE = currentlevelmaxE
            level += 1
            currentlevelmaxE = np.max([self.ShannonEntropy(n.data) for n in self.wp.get_level(level, "freq")])
            #print currentlevelmaxE
        return level

    def convert(self,i):
        level = int(np.floor(np.log2(i + 1)))
        first = 2 ** level - 1
        if i == 0:
            b = ''
        else:
            b = np.binary_repr(i - first, width=int(level))
            b = string.replace(b, '0', 'a', maxreplace=-1)
            b = string.replace(b, '1', 'd', maxreplace=-1)
        return b

    def BestTree(self,wp,threshold,costfn='threshold'):
        # Compute the `cost' of each node
        # ** Doesn't always exactly match matlab, but close
        nnodes = 2 ** (wp.maxlevel + 1) - 1
        cost = np.zeros(nnodes)
        count = 0
        for level in range(wp.maxlevel + 1):
            for n in wp.get_level(level, 'natural'):
                if costfn == 'threshold':
                    # Threshold
                    d = np.abs(n.data)
                    cost[count] = np.sum(d > threshold)
                elif costfn == 'entropy':
                    # Entropy
                    d = n.data ** 2
                    cost[count] = -np.sum(np.where(d != 0, d * np.log(d), 0))
                else:
                    # SURE
                    # TODO: Check this one ***
                    d = n.data ** 2
                    t2 = threshold * threshold
                    ds = np.sum(d > t2)
                    cost[count] = 2 * ds - len(n.data) + t2 * ds + np.sum(d * (d <= t2))

                count += 1

        # Compute the best tree using those cost values
        flags = 2 * np.ones(nnodes)
        flags[2 ** wp.maxlevel - 1:] = 1
        # Work up the tree from just above leaves
        inds = np.arange(2 ** wp.maxlevel - 1)
        inds = inds[-1::-1]
        for i in inds:
            # Get children
            children = (i + 1) * 2 + np.arange(2) - 1
            c = cost[children[0]] + cost[children[1]]
            if c < cost[i]:
                cost[i] = c
                flags[i] = 2
            else:
                flags[i] = flags[children[0]] + 2
                flags[children] = -flags[children]

        # Now get the new leaves of the tree. Anything below these nodes is deleted.
        newleaves = np.where(flags > 2)[0]

        # Make a list of the children of the newleaves, and recursively their children
        def getchildren(n):
            level = int(np.floor(np.log2(n + 1)))
            if level < wp.maxlevel:
                tbd.append((n + 1) * 2 - 1)
                tbd.append((n + 1) * 2)
                getchildren((n + 1) * 2 - 1)
                getchildren((n + 1) * 2)

        tbd = []
        for i in newleaves:
            getchildren(i)

        tbd = np.unique(tbd)

        # I wasn't happy that these were being deleted, so am going the other way round
        listnodes = np.arange(2 ** (wp.maxlevel + 1) - 1)
        listnodes = np.delete(listnodes, tbd)
        notleaves = np.intersect1d(newleaves, tbd)
        for i in notleaves:
            newleaves = np.delete(newleaves, np.where(newleaves == i))

        listleaves = np.intersect1d(np.arange(2 ** (wp.maxlevel) - 1, 2 ** (wp.maxlevel + 1) - 1), listnodes)
        listleaves = np.unique(np.concatenate((listleaves, newleaves)))

        return listleaves

    def reconstruct(self,new_wp,wavelet,listleaves):
        # Sort the list of leaves into order bottom-to-top, left-to-right
        # Note that it's a bit hack. Assumes that listleaves is top-to-bottom, so just reverses it
        working = listleaves.copy()
        working = working[-1::-1]

        level = int(np.floor(np.log2(working[0] + 1)))
        while level > 0:
            first = 2 ** level - 1
            while working[0] >= first:
                # Note this is Python2!
                # And also that it assumes that the whole list is backwards
                parent = (working[0] - 1) / 2
                p = self.convert(parent)
                #print self.convert(parent), working[0], working[1]
                # print working
                #print len(new_wp[self.convert(working[1])].data), len(new_wp[self.convert(working[0])].data), len(new_wp[self.convert(parent)].data), len(pywt.idwt(new_wp[self.convert(working[1])].data, new_wp[self.convert(working[0])].data, wavelet))
                new_wp[p].data = pywt.idwt(new_wp[self.convert(working[1])].data,new_wp[self.convert(working[0])].data, wavelet)[:len(new_wp[p].data)]
                # Delete these two nodes from working
                working = np.delete(working, 1)
                working = np.delete(working, 0)
                # print working
                # Insert parent into list of nodes at the next level
                ins = np.where(working > parent)
                if len(ins[0]) > 0:
                    ins = ins[0][-1] + 1
                else:
                    ins = 0
                working = np.insert(working, ins, parent)
            #print working
            level = int(np.floor(np.log2(working[0] + 1)))
            #print "next level: ", level
        return new_wp

    def waveletDenoise(self,data=None,thresholdType='soft',threshold=4.5,maxlevel=None,bandpass=False,wavelet='dmey2',costfn='threshold'):
        # Perform wavelet denoising. Can use soft or hard thresholding

        if data is None:
            data = self.data

        print data[:10]

        if wavelet == 'dmey2':
            [lowd, highd, lowr, highr] = np.loadtxt('dmey.txt')
            wavelet = pywt.Wavelet(filter_bank=[lowd, highd, lowr, highr])
            wavelet.orthogonal=True

        # print wavelet

        if maxlevel is None:
            self.maxlevel = self.BestLevel(wavelet)
            print "Best level is ", self.maxlevel
        else:
            self.maxlevel = maxlevel
            print "Using level ", self.maxlevel

        self.thresholdMultiplier = threshold

        wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=self.maxlevel)

        # Get the threshold
        det1 = wp['d'].data
        # Note magic conversion number
        sigma = np.median(np.abs(det1)) / 0.6745
        threshold = self.thresholdMultiplier * sigma

        bestleaves = self.BestTree(wp,threshold)
        print bestleaves

        # Make a new tree with these in
        new_wp = pywt.WaveletPacket(data=None, wavelet=wp.wavelet, mode='zero', maxlevel=wp.maxlevel)

        # There seems to be a bit of a bug to do with the size of the reconstucted nodes, so prime them
        # It's worse than that. pywavelet makes the whole tree. So if you don't give it blanks, it copies the details from wp even though it wasn't asked for. And reconstruction with the zeros is different to not reconstructing.
        for level in range(wp.maxlevel + 1):
            for n in new_wp.get_level(level, 'natural'):
                n.data = np.zeros(len(wp.get_level(level, 'natural')[0].data))

        # Copy thresholded versions of the leaves into the new wpt
        for l in bestleaves:
            ind = self.convert(l)
            if thresholdType == 'hard':
                # Hard thresholding
                new_wp[ind].data = np.where(np.abs(new_wp[ind].data) < threshold, 0.0, wp[ind].data)
            else:
                # Soft thresholding
                # n.data = np.sign(n.data) * np.maximum((np.abs(n.data) - threshold), 0.0)
                tmp = np.abs(wp[ind].data) - threshold
                tmp = (tmp + np.abs(tmp)) / 2.
                new_wp[ind].data = np.sign(wp[ind].data) * tmp

        # Reconstruct the internal nodes and the data
        new_wp = self.reconstruct(new_wp,wp.wavelet,bestleaves)
        print new_wp[''].data[:10]

        # wavio.write('testme.wav', new_wp[''].data, self.sampleRate, sampwidth=2)

        #np.savetxt('testme.txt',new_wp[''].data)
        #data = new_wp[''].data
        #data = float(2**16) * (data - np.min(data)) / (np.max(data)-np.min(data)) - 2**15
        return new_wp[''].data

    def bandpassFilter(self,data=None,start=1000,end=10000):
        # Bandpass filter
        if data is None:
            data = self.data
        nyquist = self.sampleRate/2.0
        #ripple_db = 80.0
        #width = 1.0/nyquist
        #ntaps, beta = signal.kaiserord(ripple_db, width)
        ntaps = 128
        #taps = signal.firwin(ntaps,cutoff = [500/nyquist,8000/nyquist], window=('kaiser', beta),pass_zero=False)
        taps = signal.firwin(ntaps, cutoff=[start / nyquist, end / nyquist], window=('hamming'), pass_zero=False)
        return signal.lfilter(taps, 1.0, data)

    def ButterworthBandpass(self,data,sampleRate,low=1000,high=5000,order=10):
        if data is None:
            data = self.data
            sampleRate = self.sampleRate
        nyquist = sampleRate/2.0

        lowPass = float(low)/nyquist
        highPass = float(high)/nyquist
        lowStop = float(low-50)/nyquist
        highStop = float(high+50)/nyquist
        # calculate the best order
        order,wN = signal.buttord([lowPass, highPass], [lowStop, highStop], 3, 50)
        print 'order=', order
        # print 'wN=', wN
        # print 'lowpass, highpass', lowPass,highPass
        if order>10:
            order=10
        b, a = signal.butter(order,[lowPass, highPass], btype='band')
        return signal.filtfilt(b, a, data)

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

    def loadData(self,fileName):
        # Load a sound file and normalise it
        # self.sampleRate, self.data = wavfile.read(fileName)
        wavobj = wavio.read(fileName)
        self.sampleRate = wavobj.rate
        self.data = wavobj.data
        # self.sampleRate, self.data = wavfile.read('../Birdsong/more1.wav')
        # self.sampleRate, self.data = wavfile.read('../Birdsong/Denoise/Primary dataset/kiwi/female/female1.wav')
        #self.sampleRate, self.data = wavfile.read('ruru.wav')
        #self.sampleRate, self.data = wavfile.read('tril1.wav')
        # self.sampleRate, self.data = wavfile.read('male1.wav')
        # The constant is for normalisation (2^15, as 16 bit numbers)
        if self.data.dtype is not 'float':
            self.data = self.data.astype('float') #/ 32768.0
        if np.shape(np.shape(self.data))[0]>1:
            self.data = self.data[:,0]
        # self.data = self.data.astype('float') / 32768.0

# Functions for loading and saving files -- largely unnecessary
def writeFile(data,name,fs):
    # Save a sound file (after denoising)
    # Need them to be 16 bit integers
    data *= 32768.0
    dData = data.astype('int16')
    # wavfile.write(name,self.sampleRate, self.wData)
    wavio.write(name,data,fs,sampwidth=2)


def denoiseFile(fileName,thresholdMultiplier):
    sp = SignalProc(thresholdMultiplier=thresholdMultiplier)
    sp.loadData(fileName)
    yd=sp.waveletDenoise()
    writeFile(yd,fileName[:-4]+'denoised'+str(sp.thresholdMultiplier)+fileName[-4:],sp.sampleRate)

# def test():
    # #pl.ion()
    # a = SignalProc()
    # #a.splitFile5mins('ST0026.wav')

    # a.loadData()
    # #a.play()
    # #a.testTree()
    # sg = a.spectrogram(a.data)
    # pl.figure()
    # pl.imshow(10.0*np.log10(sg),cmap='gray')
    # a.waveletDenoise()
    # sgn = a.spectrogram(a.wData)
    # pl.figure()
    # pl.imshow(10.0*np.log10(sgn),cmap='gray')
    # pl.figure()
    # pl.plot(a.wData)
    # #a.plot()
    # #a.play()
    # a.writefile('out.wav')
    # pl.show()

# def testCorr():
    # # This is an outmoded (as slower) version of cross-correlation
    # sp = SignalProc()
    # sp.loadData('Sound Files/tril1.wav')
    # sg = sp.spectrogram(sp.data,multitaper=True)
    # seg = sg[:,79:193]
    # indices = sp.findMatches(seg,sg,0.4)
    # #pl.figure()
    # #pl.plot(matches)
    # #for i in indices:
    # #    pl.plot(i,0.6,'x')
    # print indices

    # #print corr
    # fig = pl.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(sg)
    # for i in indices:
        # ax.add_patch(pl.Rectangle((i,0),114,128,alpha=0.3))
    # #pl.subplot(212), pl.imshow(corr)

    # #c1 = np.max(corr, axis=0)
    # #import heapq
    # #heapq.nlargest(20, range(len(c1)), c1.take)
    # # Peaks are at 990, 588, 135
    # return indices


# def show():
    # #pl.ion()
    # a = SignalProc()
    # #a.loadData('Sound Files/male1.wav')
    # a.loadData('Sound Files/tril1.wav')
    # #a.data = a.data[:60000,0]
    # sg = a.spectrogram(a.data)
    # #pl.figure()
    # #pl.plot(a.data)
    # pl.figure()
    # pl.imshow(10.0*np.log10(sg),cmap='gray_r')
    # pl.figure()
    # b = a.ButterworthBandpass(a.data,a.sampleRate)
    # print np.shape(a.data), np.shape(b)
    # pl.imshow(10.0*np.log10(a.spectrogram(a.ButterworthBandpass(a.data,a.sampleRate))),cmap='gray')
    # #pl.figure()
    # #pl.imshow(10.0*np.log10(a.spectrogram(a.bandpassFilter(a.data,a.sampleRate))),cmap='gray')
    # pl.show()

#show()
#pl.show()
#test()
#pl.show()

#pl.ion()

#denoiseFile('Sound Files/tril1.wav',4.5)
#denoiseFile('tril1.wav',3.5)
#denoiseFile('tril1.wav',4.0)
#denoiseFile('tril1.wav',4.5)
#denoiseFile('tril1.wav',5.0)
