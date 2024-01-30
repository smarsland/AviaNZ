
# Spectrogram.py
# The spectrogram class holds the audiodata and the spectrogram made from it
# Also holds functions that draw on the spectrogram 

# Version 3.0 14/09/20
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2020

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
import scipy.signal as signal
import scipy.fftpack as fft
from scipy.stats import boxcox
import wavio
import resampy
import copy
import gc
import SignalProc

from PyQt6.QtGui import QImage

QtMM = True
try:
    from PyQt6.QtMultimedia import QAudioFormat
except ImportError:
    print("No QtMM")
    QtMM = False

# for multitaper spec:
specExtra = True
try:
    from spectrum import dpss, pmtm
except ImportError:
    specExtra = False

# for fund freq
from scipy.signal import medfilt
# TODO: Needs some tidying up

class Spectrogram:
    """ This class reads and holds the audiodata and spectrogram, to be used in the main interface.
    Inverse, denoise, and other processing algorithms are provided here.
    Primary parameters are the width of a spectrogram window (window_width) and the shift between them (incr)
    """

    def __init__(self, window_width=256, incr=128, minFreqShow=0, maxFreqShow=float("inf")):
        # maxFreq = 0 means fall back to Fs/2 for any file.
        self.window_width=window_width
        self.incr=incr
        self.minFreqShow = minFreqShow
        self.maxFreqShow = maxFreqShow
        self.data = []

        # only accepting wav files of this format
        if QtMM:
            self.audioFormat = QAudioFormat()
            # TODO!!
            #self.audioFormat.setCodec("audio/pcm")
            #self.audioFormat.setByteOrder(QAudioFormat.LittleEndian)

    def readWav(self, file, duration=None, off=0, silent=False):
        """ Args the same as for wavio.read: filename, length in seconds, offset in seconds. """
        wavobj = wavio.read(file, duration, off)
        self.data = wavobj.data

        # take only left channel
        if np.shape(np.shape(self.data))[0] > 1:
            self.data = self.data[:, 0]
        if QtMM:
            self.audioFormat.setChannelCount(1)

        # force float type
        if self.data.dtype != 'float':
            self.data = self.data.astype('float')

        # total file length in s read from header (useful for paging)
        self.fileLength = wavobj.nseconds

        self.sampleRate = wavobj.rate

        if QtMM:
            self.audioFormat.setSampleRate(self.sampleRate)
            #self.audioFormat.setSampleSize(wavobj.sampwidth * 8)
            # Only 8-bit WAVs are unsigned:
            # TODO!! Int16/Int32
            if wavobj.sampwidth==1:
                self.audioFormat.setSampleFormat(QAudioFormat.SampleFormat.UInt8)
            elif wavobj.sampwidth==2:
                self.audioFormat.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            else:
                self.audioFormat.setSampleFormat(QAudioFormat.SampleFormat.Int32)
            #if wavobj.sampwidth==1:
                #self.audioFormat.setSampleType(QAudioFormat.UnSignedInt)
            #else:
                #self.audioFormat.setSampleType(QAudioFormat.SignedInt)

        # *Freq sets hard bounds, *Show can limit the spec display
        self.minFreq = 0
        self.maxFreq = self.sampleRate // 2
        self.minFreqShow = max(self.minFreq, self.minFreqShow)
        self.maxFreqShow = min(self.maxFreq, self.maxFreqShow)

        #print("a",self.sampleRate, self.fileLength, np.shape(self.data))

        if not silent:
            if QtMM:
                #print("Detected format: %d channels, %d Hz, ** bit samples" % (self.audioFormat.channelCount(), self.audioFormat.sampleRate()))
                sf = str(self.audioFormat.sampleFormat())
                print("Detected format: %d channels, %d Hz, %s format" % (self.audioFormat.channelCount(), self.audioFormat.sampleRate(), sf.split('.')[-1]))
                #print("Detected format: %d channels, %d Hz, %d bit samples" % (self.audioFormat.channelCount(), self.audioFormat.sampleRate(), self.audioFormat.sampleSize()))

    def readBmp(self, file, duration=None, off=0, silent=False, rotate=True, repeat=True):
        """ Reads DOC-standard bat recordings in 8x row-compressed BMP format.
            For similarity with readWav, accepts len and off args, in seconds.
            rotate: if True, rotates to match setImage and other spectrograms (rows=time)
                otherwise preserves normal orientation (cols=time)
        """
        # !! Important to set these, as they are used in other functions
        self.sampleRate = 176000
        # TODO: why was this here?
        #if not repeat:
            #self.incr = 512
        self.incr = 512

        img = QImage(file, "BMP")
        h = img.height()
        w = img.width()
        colc = img.colorCount()
        if h==0 or w==0:
            print("ERROR: image was not loaded")
            return(1)

        # Check color format and convert to grayscale
        if not silent and (not img.allGray() or colc>256):
            print("Warning: image provided not in 8-bit grayscale, information will be lost")
        img.convertTo(QImage.Format.Format_Grayscale8)

        # Convert to numpy
        # (remember that pyqtgraph images are column-major)
        ptr = img.constBits()
        ptr.setsize(h*w*1)
        img2 = np.array(ptr).reshape(h, w)

        # Determine if original image was rotated, based on expected num of freq bins and freq 0 being empty
        # We also used to check if np.median(img2[-1,:])==0,
        # but some files happen to have the bottom freq bin around 90, so we cannot rely on that.
        if h==64:
            # standard DoC format
            pass
        elif w==64:
            # seems like DoC format, rotated at -90*
            img2 = np.rot90(img2, 1, (1,0))
            w, h = h, w
        else:
            print("ERROR: image does not appear to be in DoC format!")
            print("Format details:")
            print(img2)
            print(h, w)
            print(min(img2[-1,:]), max(img2[-1,:]))
            print(np.sum(img2[-1,:]>0))
            print(np.median(img2[-1,:]))
            return(1)

        #print(np.shape(img2))
        # Could skip that for visual mode - maybe useful for establishing contrast?
        img2[-1, :] = 254  # lowest freq bin is 0, flip that
        img2 = 255 - img2  # reverse value having the black as the most intense
        img2 = img2/np.max(img2)  # normalization
        img2 = img2[:, 1:]  # Cutting first time bin because it only contains the scale and cutting last columns
        if repeat:
            img2 = np.repeat(img2, 8, axis=0)  # repeat freq bins 7 times to fit invertspectrogram
        #print(np.shape(img2))

        self.data = []
        self.fileLength = (w-2)*self.incr + self.window_width  # in samples
        # Alternatively:
        # self.fileLength = self.convertSpectoAmpl(h-1)*self.sampleRate

        # NOTE: conversions will use self.sampleRate and self.incr, so ensure those are already set!
        # trim to specified offset and length:
        if off>0 or duration is not None:
            # Convert offset from seconds to pixels
            off = int(self.convertAmpltoSpec(off))
            if duration is None:
                img2 = img2[:, off:]
            else:
                # Convert length from seconds to pixels:
                duration = int(self.convertAmpltoSpec(duration))
                img2 = img2[:, off:(off+duration)]

        if rotate:
            # rotate for display, b/c required spectrogram dimensions are:
            #  t increasing over rows, f increasing over cols
            # This will be enough if the original image was spectrogram-shape.
            img2 = np.rot90(img2, 1, (1,0))

        self.sg = img2

        if QtMM:
            self.audioFormat.setChannelCount(0)
            #self.audioFormat.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            #self.audioFormat.setSampleSize(0)
            self.audioFormat.setSampleRate(self.sampleRate)
        #else:
            #self.audioFormat['channelCount'] = 0
            #self.audioFormat['sampleFormat'] = 0
            #self.audioFormat['sampleRate'] = self.sampleRate

        self.minFreq = 0
        self.maxFreq = self.sampleRate //2
        self.minFreqShow = max(self.minFreq, self.minFreqShow)
        self.maxFreqShow = min(self.maxFreq, self.maxFreqShow)

        if not silent:
            print("Detected BMP format: %d x %d px, %d colours" % (w, h, colc))
        return(0)

    def resample(self, target, data=None):
        if data is None:
            data = self.data
        if len(data)==0:
            print("Warning: no data set to resample")
            return
        if target==self.sampleRate:
            print("No resampling needed")
            return

        data = resampy.resample(data, sr_orig=self.sampleRate, sr_new=target)

        self.sampleRate = target
        if QtMM:
            self.audioFormat.setSampleRate(target)
        #else:
            #self.audioFormat['sampleRate'] = target

        self.minFreq = 0
        self.maxFreq = self.sampleRate // 2

        self.fileLength = len(self.data)

    def convertAmpltoSpec(self, x):
        """ Unit conversion, for easier use wherever spectrograms are needed """
        return x*self.sampleRate/self.incr

    def convertSpectoAmpl(self,x):
        """ Unit conversion """
        return x*self.incr/self.sampleRate

    def convertFreqtoY(self,f):
        """ Unit conversion """
        sgy = np.shape(self.sg)[1]
        if f>self.maxFreqShow:
            return -100
        else:
            return (f-self.minFreqShow) * sgy / (self.maxFreqShow - self.minFreqShow)

    # SRM: TO TEST **
    def convertHztoMel(self,f):
        return 1125*np.log(1+f/700)
        #return 2595*np.log10(1+f/700)

    def convertMeltoHz(self,m):
        return 700*(np.exp(m/1125)-1)
        #return 700*(10**(m/2595)-1)

    def convertHztoBark(self,f):
        # TODO: Currently doesn't work on arrays
        b = (26.81*f)/(1960+f) -0.53
        if b<2:
            b += 0.15/(2-b)
        elif b>20.1:
            b += 0.22*(b-20.1)
        #inds = np.where(b<2)
        #print(inds)
        #b[inds] += 0.15/(2-b[inds])
        #inds = np.where(b>20.1)
        #b[inds] += 0.22*(b[inds]-20.1)
        return b

    def convertBarktoHz(self,b):
        inds = np.where(b<2)
        b[inds] = (b[inds]-0.3)/0.85
        inds = np.where(b>20.1)
        b[inds] = (b[inds]+4.422)/1.22
        return 1960*((b+0.53)/(26.28-b))

    def mel_filter(self,filter='mel',nfilters=40,minfreq=0,maxfreq=None,normalise=True):
        # Transform the spectrogram to mel or bark scale
        if maxfreq is None:
            maxfreq = self.sampleRate/2
        print(filter,nfilters,minfreq,maxfreq,normalise)

        if filter=='mel':
            filter_points = np.linspace(self.convertHztoMel(minfreq), self.convertHztoMel(maxfreq), nfilters + 2)  
            bins = self.convertMeltoHz(filter_points)
        elif filter=='bark':
            filter_points = np.linspace(self.convertHztoBark(minfreq), self.convertHztoBark(maxfreq), nfilters + 2)  
            bins = self.convertBarktoHz(filter_points)
        else:
            print("ERROR: filter not known",filter)
            return(1)

        nfft = np.shape(self.sg)[1]
        freq_points = np.linspace(minfreq,maxfreq,nfft)

        filterbank = np.zeros((nfft,nfilters))
        for m in range(nfilters):
            # Find points in first and second halves of the triangle
            inds1 = np.where((freq_points>=bins[m]) & (freq_points<=bins[m+1]))
            inds2 = np.where((freq_points>=bins[m+1]) & (freq_points<=bins[m+2]))
            # Compute their contributions
            filterbank[inds1,m] = (freq_points[inds1] - bins[m]) / (bins[m+1] - bins[m])   
            filterbank[inds2,m] = (bins[m+2] - freq_points[inds2]) / (bins[m+2] - bins[m+1])             

        if normalise:
            # Normalise to unit area if desired
            norm = filterbank.sum(axis=0)
            norm = np.where(norm==0,1,norm)
            filterbank /= norm

        return filterbank

    def convertToMel(self,filt='mel',nfilters=40,minfreq=0,maxfreq=None,normalise=True):
        filterbank = self.mel_filter(filt,nfilters,minfreq,maxfreq,normalise)
        # Single channel spectrograms will convert successfully. Exception is for Multi-tapered spectrograms.
        try:
            self.sg = np.dot(self.sg,filterbank)
        except:
            print("Mel conversion problems")
            placeholder = np.zeros(shape=(np.shape(self.sg)[0],np.shape(filterbank)[1],np.shape(self.sg)[2]))
            for i in range(np.shape(self.sg)[2]):
                placeholder[:,:,i] = np.dot(self.sg[:,:,i],filterbank)
            self.sg = placeholder
    # ====

    def setWidth(self,window_width,incr):
        # Does what it says. Called when the user modifies the spectrogram parameters
        self.window_width = window_width
        self.incr = incr

    def setData(self,audiodata,sampleRate=None):
        self.data = audiodata
        if sampleRate is not None:
            self.sampleRate = sampleRate

    def SnNR(self,startSignal,startNoise):
        # Compute the estimated signal-to-noise ratio
        pS = np.sum(self.data[startSignal:startSignal+self.length]**2)/self.length
        pN = np.sum(self.data[startNoise:startNoise+self.length]**2)/self.length
        return 10.*np.log10(pS/pN)

    def equalLoudness(self,data):
        # TODO: Assumes 16000 sampling rate, fix!
        # Basically, save a few more sets of filter coefficients...

        # Basic equal loudness curve. 
        # This is for humans, NOT birds (there is a paper that claims to have some, but I can't access it:
        # https://doi.org/10.1121/1.428951)

        # The filter weights were obtained from Matlab (using yulewalk) for the standard 80 dB ISO curve
        # for a sampling rate of 16000

        # 10 coefficient Yule-Walker fit for [0,120;20,113;30,103;40,97;50,93;60,91;70,89;80,87;90,86;100,85;200,78;300,76;400,76;500,76;600,76;700,77;800,78;900,79.5;1000,80;1500,79;2000,77;2500,74;3000,71.5;3700,70;4000,70.5;5000,74;6000,79;7000,84;8000,86]
        # Or at least, EL80(:,1)./(fs/2) and m=10.^((70-EL80(:,2))/20);

        ay = np.array([1.0000,-0.6282, 0.2966,-0.3726,0.0021,-0.4203,0.2220,0.0061, 0.0675, 0.0578,0.0322])
        by = np.array([0.4492,-0.1435,-0.2278,-0.0142,0.0408,-0.1240,0.0410,0.1048,-0.0186,-0.0319,0.0054])

        # Butterworth highpass
        ab = np.array([1.0000,-1.9167,0.9201])
        bb = np.array([0.9592,-1.9184,0.9592])

        data = signal.lfilter(by,ay,data)
        data = signal.lfilter(bb,ab,data)

        return data

    # from memory_profiler import profile
    # fp = open('memory_profiler_sp.log', 'w+')
    # @profile(stream=fp)
    def spectrogram(self,window_width=None,incr=None,window='Hann',sgType='Standard',sgScale='Linear',nfilters=128,equal_loudness=False,mean_normalise=True,onesided=True,need_even=False,start=None,stop=None,singleIm=True):
        """ Compute the spectrogram from amplitude data
        Returns the power spectrum, not the density -- compute 10.*log10(sg) 10.*log10(sg) before plotting.
        Uses absolute value of the FT, not FT*conj(FT), 'cos it seems to give better discrimination
        Options: multitaper version, but it's slow, mean normalised, even, one-sided.
        This version is faster than the default versions in pylab and scipy.signal
        Assumes that the values are not normalised.
        """
        if start is None:
            data = self.data
        else:
            # TODO: Error checking
            data = self.data[start:stop]
        if data is None or len(data)==0:
            print("ERROR: attempted to calculate spectrogram without audiodata")
            return

        #S = librosa.feature.melspectrogram(self.data, sr=self.sampleRate, power=1)
        #log_S = librosa.amplitude_to_db(S, ref=np.max)
        #self.sg = librosa.pcen(S * (2**31))
        #return self.sg.T
        if window_width is None:
            window_width = self.window_width
        if incr is None:
            incr = self.incr

        # clean handling of very short segments:
        if len(data) <= window_width:
            window_width = len(data) - 1

        self.sg = np.copy(data)
        if self.sg.dtype != 'float':
            self.sg = self.sg.astype('float')

        # Set of window options
        if window=='Hann':
            # This is the Hann window
            window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_width) / (window_width - 1)))
        elif window=='Parzen':
            # Parzen (window_width even)
            n = np.arange(window_width) - 0.5*window_width
            window = np.where(np.abs(n)<0.25*window_width,1 - 6*(n/(0.5*window_width))**2*(1-np.abs(n)/(0.5*window_width)), 2*(1-np.abs(n)/(0.5*window_width))**3)
        elif window=='Welch':
            # Welch
            window = 1.0 - ((np.arange(window_width) - 0.5*(window_width-1))/(0.5*(window_width-1)))**2
        elif window=='Hamming':
            # Hamming
            alpha = 0.54
            beta = 1.-alpha
            window = alpha - beta*np.cos(2 * np.pi * np.arange(window_width) / (window_width - 1))
        elif window=='Blackman':
            # Blackman
            alpha = 0.16
            a0 = 0.5*(1-alpha)
            a1 = 0.5
            a2 = 0.5*alpha
            window = a0 - a1*np.cos(2 * np.pi * np.arange(window_width) / (window_width - 1)) + a2*np.cos(4 * np.pi * np.arange(window_width) / (window_width - 1))
        elif window=='BlackmanHarris':
            # Blackman-Harris
            a0 = 0.358375
            a1 = 0.48829
            a2 = 0.14128
            a3 = 0.01168
            window = a0 - a1*np.cos(2 * np.pi * np.arange(window_width) / (window_width - 1)) + a2*np.cos(4 * np.pi * np.arange(window_width) / (window_width - 1)) - a3*np.cos(6 * np.pi * np.arange(window_width) / (window_width - 1))
        elif window=='Ones':
            window = np.ones(window_width)
        else:
            print("Unknown window, using Hann")
            window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_width) / (window_width - 1)))

        if equal_loudness:
            self.sg = self.equalLoudness(self.sg)

        if mean_normalise:
            self.sg -= self.sg.mean()

        starts = range(0, len(self.sg) - window_width, incr)
        # Returns either multiple channels or sums them and returns one
        if sgType=='Multi-tapered':
            # TODO: hard param -- 3 tapers
            if specExtra:
                [tapers, eigen] = dpss(window_width, 2.5, 3)
                counter = 0
                out = np.zeros(shape=(len(starts),window_width // 2,3))
                for start in starts:
                    Sk, weights, eigen = pmtm(self.sg[start:start + window_width], v=tapers, e=eigen, show=False)
                    Sk = abs(Sk)**2
                    #Sk = np.mean(Sk.T * weights, axis=1)
                    for taper in range(3):
                        out[:,:,taper][counter:counter + 1,:] = Sk[taper][window_width // 2:].T
                    counter += 1  
                if singleIm:
                    out = np.squeeze(np.sum(out,axis=2))
                self.sg = np.fliplr(out)
            else:
                print("Option not available")
        elif sgType=='Reassigned':
            ft = np.zeros((len(starts), window_width),dtype='complex')
            ft2 = np.zeros((len(starts), window_width),dtype='complex')
            for i in starts:
                winddata = window * self.sg[i:i + window_width]
                ft[i // incr, :] = fft.fft(winddata)[:window_width]
                winddata = window * np.roll(self.sg[i:i + window_width],1)
                ft2[i // incr, :] = fft.fft(winddata)[:window_width]

            # Approximate the derivative by finite differences and get the angle of the complex number
            CIF = np.mod(np.angle(ft*np.conj(ft2))/(2*np.pi),1.0)
            delay = (0.5 - np.mod(np.angle(ft*np.conj(np.roll(ft,1,axis=1)))/(2*np.pi),1.0))

            # Messiness. Need to work out where to put each pixel
            # I wish I could think of a way that didn't need a histogram
            times = np.tile(np.arange(0, (len(data) - window_width)/self.sampleRate, incr/self.sampleRate) + window_width/self.sampleRate/2,(np.shape(delay)[1],1)).T + delay*window_width/self.sampleRate
            self.sg,_,_ = np.histogram2d(times.flatten(),CIF.flatten(),weights=np.abs(ft).flatten(),bins=np.shape(ft))

            self.sg = np.absolute(self.sg[:, :window_width //2]) #+ 0.1

            print("SG range:", np.min(self.sg),np.max(self.sg))
        else:
            if need_even:
                starts = np.hstack((starts, np.zeros((window_width - len(self.sg) % window_width),dtype=int)))

            # this mode is optimized for speed, but reportedly sometimes
            # results in crashes when lots of large files are batch processed.
            # The FFTs here could be causing this, but I'm not sure.
            # hi_mem = False should switch FFTs to go over smaller vectors
            # and possibly use less caching, at the cost of 1.5x longer CPU time.
            hi_mem = True
            if hi_mem:
                ft = np.zeros((len(starts), window_width))
                for i in starts:
                    ft[i // incr, :] = self.sg[i:i + window_width]
                ft = np.multiply(window, ft)

                if onesided:
                    self.sg = np.absolute(fft.fft(ft)[:, :window_width //2])
                else:
                    self.sg = np.absolute(fft.fft(ft))
            else:
                if onesided:
                    ft = np.zeros((len(starts), window_width//2))
                    for i in starts:
                        winddata = window * self.sg[i:i + window_width]
                        ft[i // incr, :] = fft.fft(winddata)[:window_width//2]
                else:
                    ft = np.zeros((len(starts), window_width))
                    for i in starts:
                        winddata = window * self.sg[i:i + window_width]
                        ft[i // incr, :] = fft.fft(winddata)
                self.sg = np.absolute(ft)
            print(np.min(self.sg),np.max(self.sg))

            del ft
            gc.collect()
            #sg = (ft*np.conj(ft))[:,window_width // 2:].T

        if sgScale == 'Mel Frequency':
            self.convertToMel(filt='mel',nfilters=nfilters,minfreq=0,maxfreq=None,normalise=True)
        elif sgScale == 'Bark Frequency':
            self.convertToMel(filt='bark',nfilters=nfilters,minfreq=0,maxfreq=None,normalise=True)

        return self.sg

    def normalisedSpec(self, tr="Log"):
        """ Assumes the spectrogram was precomputed.
            Converts it to a scale appropriate for plotting
            tr: transform, "Log" or Box-Cox" or "Sigmoid" or "PCEN" or "Batmode".
            Latter sets a non-normalised log, useful for fixed-scale bat images.
        """
        LOG_OFFSET = 1e-7
        if tr=="Log":
            sg = self.sg + LOG_OFFSET
            minsg = np.min(sg)
            sg = 10*(np.log10(sg)-np.log10(minsg))
            sg = np.abs(sg)
            return sg
        elif tr=="Batmode":
            sg = self.sg + LOG_OFFSET
            sg = 10*np.log10(sg)
            sg = np.abs(sg)
            return sg
        elif tr=="Box-Cox":
            size = np.shape(self.sg)
            sg = self.sg + LOG_OFFSET
            sg = np.abs(sg.flatten())
            sg, lam = boxcox(sg)
            return np.reshape(sg, size)
        elif tr=="Sigmoid":
            # TODO!!!
            sig  = 1/(1+np.exp(1.2))
            return self.sg**sig
        elif tr=="PCEN":
            # Per Channel Energy Normalisation (non-trained version) arXiv 1607.05666, arXiv 1905.08352v2
            gain=0.8
            bias=10
            power=0.25
            t=0.060
            eps=1e-6
            s = 1 - np.exp( -self.incr / (t*self.sampleRate))
            M = signal.lfilter([s],[1,s-1],self.sg)
            smooth = (eps + M)**(-gain)
            return (self.sg*smooth+bias)**power - bias**power
        else:
            print("ERROR: unrecognized transformation", tr)

    def Stockwell(self):
        # Stockwell transform (Brown et al. version)
        # Need to get the starts etc. sorted

        width = len(self.audiodata) // 2

        # Gaussian window for frequencies
        f_half = np.arange(0, width + 1) / (2 * width)
        f = np.concatenate((f_half, np.flipud(-f_half[1:-1])))
        p = 2 * np.pi * np.outer(f, 1 / f_half[1:])
        window = np.exp(-p ** 2 / 2).T

        f_tran = fft.fft(self.audiodata, 2*width, overwrite_x=True)
        diag_con = np.linalg.toeplitz(np.conj(f_tran[:width + 1]), f_tran)
        # Remove zero freq line
        diag_con = diag_con[1:width + 1, :]  
        return np.flipud(fft.ifft(diag_con * window, axis=1))

    def wiener_entropy(self,sg):
        return np.sum(np.log(sg),1)/np.shape(sg)[1] - np.log(np.sum(sg,1)/np.shape(sg)[1])

    def mean_frequency(self,sampleRate,timederiv,freqderiv):
        freqs = sampleRate//2 / np.shape(timederiv)[1] * (np.arange(np.shape(timederiv)[1])+1)
        mfd = np.sum(timederiv**2 + freqderiv**2,axis=1)
        mfd = np.where(mfd==0,1,mfd)
        mf = np.sum(freqs * (timederiv**2 + freqderiv**2),axis=1)/mfd
        return freqs,mf

    def goodness_of_pitch(self,spectral_deriv,sg):
        return np.max(np.abs(fft.fft(spectral_deriv/sg, axis=0)),axis=0)

    def spectral_derivative(self, window_width, incr, K=2, threshold=0.5, returnAll=False):
        """ Compute the spectral derivative """
        if self.data is None or len(self.data)==0:
            print("ERROR: attempted to calculate spectrogram without audiodata")
            return
        if not specExtra:
            print("Option not available")
            return

        # Compute the set of multi-tapered spectrograms
        starts = range(0, len(self.data) - window_width, incr)
        [tapers, eigen] = dpss(window_width, 2.5, K)
        sg = np.zeros((len(starts), window_width, K), dtype=complex)
        for k in range(K):
            for i in starts:
                sg[i // incr, :, k] = tapers[:, k] * self.data[i:i + window_width]
            sg[:, :, k] = fft.fft(sg[:, :, k])
        sg = sg[:, window_width//2:, :]

        # Spectral derivative is the real part of exp(i \phi) \sum_ k s_k conj(s_{k+1}) where s_k is the k-th tapered spectrogram
        # and \phi is the direction of maximum change (tan inverse of the ratio of pure time and pure frequency components)
        S = np.sum(sg[:, :, :-1]*np.conj(sg[:, :, 1:]), axis=2)
        timederiv = np.real(S)
        freqderiv = np.imag(S)

        # Frequency modulation is the angle $\pi/2 - direction of max change$
        mfd = np.max(freqderiv**2, axis=0)
        mfd = np.where(mfd==0,1,mfd)
        fm = np.arctan(np.max(timederiv**2, axis=0) / mfd)
        spectral_deriv = -timederiv*np.sin(fm) + freqderiv*np.cos(fm)

        sg = np.sum(np.real(sg*np.conj(sg)), axis=2)
        sg /= np.max(sg)

        # Suppress the noise (spectral continuity)

        # Compute the zero crossings of the spectral derivative in all directions
        # Pixel is a contour pixel if it is at a zero crossing and both neighbouring pixels in that direction are > threshold
        sdt = spectral_deriv * np.roll(spectral_deriv, 1, 0)
        sdf = spectral_deriv * np.roll(spectral_deriv, 1, 1)
        sdtf = spectral_deriv * np.roll(spectral_deriv, 1, (0, 1))
        sdft = spectral_deriv * np.roll(spectral_deriv, (1, -1), (0, 1))
        indt, indf = np.where(((sdt < 0) | (sdf < 0) | (sdtf < 0) | (sdft < 0)) & (spectral_deriv < 0))

        # Noise reduction using a threshold
        we = np.abs(self.wiener_entropy(sg))
        freqs, mf = self.mean_frequency(self.sampleRate, timederiv, freqderiv)

        # Given a time and frequency bin
        contours = np.zeros(np.shape(spectral_deriv))
        for i in range(len(indf)):
            f = indf[i]
            t = indt[i]
            if (t > 0) & (t < (np.shape(sg)[0]-1)) & (f > 0) & (f < (np.shape(sg)[1]-1)):
                thr = threshold*we[t]/np.abs(freqs[f] - mf[t])
                if (sdt[t, f] < 0) & (sg[t-1, f] > thr) & (sg[t+1, f] > thr):
                    contours[t, f] = 1
                if (sdf[t, f] < 0) & (sg[t, f-1] > thr) & (sg[t, f+1] > thr):
                    contours[t, f] = 1
                if (sdtf[t, f] < 0) & (sg[t-1, f-1] > thr) & (sg[t+1, f+1] > thr):
                    contours[t, f] = 1
                if (sdft[t, f] < 0) & (sg[t-1, f+1] > thr) & (sg[t-1, f+1] > thr):
                    contours[t, f] = 1

        if returnAll:
            return spectral_deriv, sg, fm, we, mf, np.fliplr(contours)
        else:
            return np.fliplr(contours)

    def drawSpectralDeriv(self):
        # helper function to parse output for plotting spectral derivs.
        sd = self.spectral_derivative(self.window_width, self.incr, 2, 5.0)
        print(np.shape(sd))
        if sd is not None:
            x, y = np.where(sd > 0)
            #print(y)

            # remove points beyond frq range to show
            y1 = [i * self.sampleRate//2/np.shape(self.sg)[1] for i in y]
            y1 = np.asarray(y1)
            valminfrq = self.minFreqShow/(self.sampleRate//2/np.shape(self.sg)[1])
    
            inds = np.where((y1 >= self.minFreqShow) & (y1 <= self.maxFreqShow))
            x = x[inds]
            y = y[inds]
            y = [i - valminfrq for i in y]

            return x, y
        else:
            return None, None

    def drawFundFreq(self, seg):
        """ Produces marks of fundamental freq to be drawn on the spectrogram.
            Return is a list of (x, y) segments w/ x,y - lists in spec coords
        """
        import Shapes
        # Estimate fund freq, using windows of 2 spec FFT lengths (4 columns)
        # to make life easier:
        Wsamples = 4*self.incr
        # No set minfreq cutoff here, but warn of the lower limit for
        # reliable estimation (i.e max period such that 3 periods
        # fit in the F0 window):
        minReliableFreq = self.sampleRate / (Wsamples/3)
        print("Warning: F0 estimation below %d Hz will be unreliable" % minReliableFreq)
        # returns pitch in Hz for each window of Wsamples/2
        # over the entire data provided (so full page here)
        thr = 0.5
        pitchshape = Shapes.fundFreqShaper(self.data, Wsamples, thr, self.sampleRate)
        pitch = pitchshape.y  # pitch is a shape with y in Hz

        # find out which marks should be visible
        ind = np.logical_and(pitch > self.minFreqShow+50, pitch < self.maxFreqShow)
        if not np.any(ind):
            print("Warning: no fund. freq. identified in this page")
            return

        # ffreq is calculated over windows of size W
        # first, identify segments using that original scale:
        segs = seg.convert01(ind)
        segs = seg.deleteShort(segs, 2)
        segs = seg.joinGaps(segs, 2)
        # extra round to delete those which didn't merge with any longer segments
        segs = seg.deleteShort(segs, 4)

        yadjfact = 2/self.sampleRate*np.shape(self.sg)[1]

        # then create the x sequence (in spec coordinates)
        starts = np.arange(len(pitch)) * pitchshape.tunit + pitchshape.tstart # in seconds
        # (pitchshape.tstart should always be 0 here as it used full data)
        starts = starts * self.sampleRate / self.incr  # in spec columns

        # then convert segments back to positions in each array:
        out = []
        for s in segs:
            # convert [s, e] to [s s+1 ... e-1 e]
            ixs = np.arange(s[0], s[1])
            # retrieve all pitch and start positions corresponding to this segment
            pitchSeg = pitch[ixs]
            # Adjust pitch marks to the visible freq range on the spec
            y = ((pitchSeg-self.minFreqShow)*yadjfact).astype('int')
            # smooth the pitch lines
            medfiltsize = min((len(y)-1)//2*2+1, 15)
            y = medfilt(y, medfiltsize)
            # joinGaps can introduce no-pitch pixels, which cause
            # smoothed segments to have 0 ends. Trim those:
            trimst = 0
            while y[trimst]==0 and trimst<medfiltsize//2:
                trimst += 1
            trime = len(y)-1
            while y[trime]==0 and trime>len(y)-medfiltsize//2:
                trime -= 1
            y = y[trimst:trime]
            ixs = ixs[trimst:trime]

            out.append((starts[ixs], y))
        return out

    def drawFormants(self,ncoeff=None):

        ys = self.formants(ncoeff)
        x = []
        y = []

        step = self.window_width // self.incr
        starts = np.arange(0,np.shape(self.sg)[0],step)

        # remove points beyond frq range to show
        for t in range(len(ys)):
            for f in range(len(ys[t])):
                if (ys[t][f] >= self.minFreqShow) & (ys[t][f] <= self.maxFreqShow):
                    x.append(starts[t])
                    y.append(ys[t][f]/self.sampleRate*2*np.shape(self.sg)[1])

        valminfrq = self.minFreqShow/(self.sampleRate//2/np.shape(self.sg)[1])
        y = [i - valminfrq for i in y]

        return x, y

    # TODO: why is spectrogram passed in?
    def max_energy(self, sg,thr=1.2):
        # Remember that spectrogram is actually rotated!

        colmaxinds = np.argmax(sg,axis=1)

        points = np.zeros(np.shape(sg))

        # If one wants to show only some colmaxs:
        # sg = sg/np.max(sg)
        # colmedians = np.median(sg, axis=1)
        # colmax = np.max(sg,axis=1)
        # inds = np.where(colmax>thr*colmedians)
        # print(len(inds))
        # points[inds, colmaxinds[inds]] = 1

        # just mark the argmax position in each column
        points[range(points.shape[0]), colmaxinds] = 1

        x, y = np.where(points > 0)

        # convert points y coord from spec units to Hz
        yfr = [i * self.sampleRate//2/np.shape(self.sg)[1] for i in y]
        yfr = np.asarray(yfr)

        # remove points beyond frq range to show
        inds = np.where((yfr >= self.minFreqShow) & (yfr <= self.maxFreqShow))
        x = x[inds]
        y = y[inds]

        # adjust y pos for when spec doesn't start at 0
        specstarty = self.minFreqShow / (self.sampleRate // 2 / np.shape(self.sg)[1])
        y = [i - specstarty for i in y]

        return x, y

    def formants(self,ncoeff=None):
        # First look at formants. Snell and Milinazzo '93 method
        from LevinsonDurbanRecursion import LPC

        if ncoeff is None:
            # TODO
            ncoeff = 2 + self.sampleRate // 1000

        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.window_width) / (self.window_width - 1)))
        starts = range(0, len(self.data) - self.window_width, self.window_width)
        freqs = []
        for start in starts:
            x = self.data[start:start + self.window_width]*window
            # High-pass filter
            x = signal.lfilter([1], [1., 0.63], x)

            # LPC
            A, e, k = LPC(x, ncoeff)
            A = np.squeeze(A)

            # Extract roots, turn into angles
            roots = np.roots(A)
            roots = [r for r in roots if np.imag(r) >= 0]
            angles = np.arctan2(np.imag(roots), np.real(roots))

            freqs.append(sorted(angles / 2 / np.pi * self.sampleRate))

        return freqs

    # TODO: is anything below used?
    def clickSearch(self,thresh=3):
        """
        searches for clicks in the provided imspec, saves dataset
        returns click_label, dataset and count of detections
    
        The search is made on the spectrogram image that we know to be generated with parameters (1024,512)
        Click presence is assessed for each spectrogram column: if the mean in the
        frequency band [f0, f1] (*) is bigger than a threshold we have a click
        thr=mean(all_spec)+thresh*std(all_spec) (*)
    
        The clicks are discarded if longer than 0.05 sec
    
        imspec: unrotated spectrogram (rows=time)
        file: NOTE originally was basename, now full filename
        """
        import math
        imspec = self.sg[:,::8].T
        print('click',np.shape(imspec))
        df=self.sampleRate//2 /(np.shape(imspec)[0]+1)  # frequency increment
        # up_len=math.ceil(0.05/dt) #0.5 second lenth in indices divided by 11
        up_len=17
        # up_len=math.ceil((0.5/11)/dt)
    
        # Frequency band
        f0=24000
        index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
        f1=54000
        index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up
    
        # Mean in the frequency band
        mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0)
    
        # Threshold
        mean_spec_all=np.mean(imspec, axis=0)[2:]
        thr_spec=(np.mean(mean_spec_all)+thresh*np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))
    
        ## clickfinder
        # check when the mean is bigger than the threshold
        # clicks is an array which elements are equal to 1 only where the sum is bigger
        # than the mean, otherwise are equal to 0
        clicks = mean_spec>thr_spec
        inds = np.where(clicks>0)[0]
        if (len(inds)) > 0:
            # Have found something, now find first that isn't too long
            flag = False
            start = inds[0]
            while flag:
                i=1
                while inds[i]-inds[i-1] == 1:
                    i+=1
                end = i
                if end-start<up_len:
                    flag=True
                else:
                    start = inds[end+1]
    
            first = start

            # And last that isn't too long
            flag = False
            end = inds[-1]
            while flag:
                i=len(inds)-1
                while inds[i]-inds[i-1] == 1:
                    i-=1
                start = i
                if end-start<up_len:
                    flag=True
                else:
                    end = inds[start-1]
            last = end
            print(first,last)
            return [first,last]
        else:
            return None
    
    def denoiseImage(self,sg,thr=1.2):
        from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
        sigma_est = estimate_sigma(sg, multichannel=False, average_sigmas=True)
        sgnew = denoise_tv_chambolle(sg, weight=0.2, multichannel=False)
        #sgnew = denoise_bilateral(sg, sigma_color=0.05, sigma_spatial=15, multichannel=False)
        #sgnew = denoise_wavelet(sg, multichannel=False)

        return sgnew

    def denoiseImage2(self,sg,filterSize=5):
        # Filter size is odd
        [x,y] = np.shape(sg)
        width = filterSize//2
        
        sgnew = np.zeros(np.shape(sg))
        sgnew[0:width+1,:] = sg[0:width+1,:]
        sgnew[-width:,:] = sg[-width:,:]
        sgnew[:,0:width+1] = sg[:,0:width+1]
        sgnew[:,-width:] = sg[:,-width:]

        for i in range(width,x-width):
            for j in range(width,y-width):
               sgnew[i,j] = np.median(sg[i-width:i+width+1,j-width:j+width+1]) 

        print(sgnew)
        return sgnew

    def mark_rain(self, sg, thr=0.9):
        row, col = np.shape(sg.T)
        print(row, col)
        inds = np.where(sg > thr * np.max(sg))
        longest = np.zeros(col)
        start = np.zeros(col)
        for c in range(col):
            r = 0
            l = 0
            s = 0
            j = 0
            while inds[0][r] == c:
                if inds[1][r + 1] == inds[1][r] + 1:
                    l += 1
                else:
                    if l > longest[c]:
                        longest[c] = l
                        start[c] = s
                        l = 0
                        s = j + 1
                r += 1

        newsg = np.zeros(np.shape(sg))
        newsg = newsg.T
        for c in range(col):
            if longest[c] > 10:
                newsg[c, start[c]:start[c] + longest[c]] = 1
        print(longest)
        return newsg.T

    def denoise(self, alg, start=None, end=None, width=None):
        """ alg - string, algorithm type from the Denoise dialog
        start, end - filtering limits, from Denoise dialog
        width - median parameter, from Denoise dialog
        """
        if str(alg) == "Wavelets":
            print("Don't use this interface for wavelets")
            return
        elif str(alg) == "Bandpass":
            self.data = SignalProc.bandpassFilter(self.data,self.sampleRate, start=start, end=end)
        elif str(alg) == "Butterworth Bandpass":
            self.data = SignalProc.ButterworthBandpass(self.data, self.sampleRate, low=start, high=end)
        else:
            # Median Filter
            self.data = SignalProc.medianFilter(self.data,int(str(width)))

    def generateFeaturesCNN(self, seglen, real_spec_width, frame_size, frame_hop=None, CNNfRange=None):
        '''
        Prepare a syllable to input to the CNN model
        Returns the features (spectrogram for each frame)
        seglen: length of this segment (self.data), in s
        frame_size: length of each frame, in s
        real_spec_width: number of spectrogram columns in each frame
            (slightly differs from expected b/c of boundary effects,
             so passing w/ a precalculated adjustment)
        frame_hop: hop between frames, in s, or None to not overlap
            (i.e. hop by 1 frame_size)
        CNNfRange: frequency list [f1, f2], if not None, sets
            spectrogram pixels outside f1:f2 to 0
        '''
        # determine the number of frames:
        if frame_hop is None:
            n = seglen // frame_size
            frame_hop = frame_size
        else:
            n = (seglen-frame_size) // frame_hop + 1
        n = int(n)

        _ = self.spectrogram()

        # Mask out of band elements
        spec_height = np.shape(self.sg)[1]
        if CNNfRange is not None:
            bin_width = self.sampleRate / 2 / spec_height
            lb = int(np.ceil(CNNfRange[0] / bin_width))
            ub = int(np.floor(CNNfRange[1] / bin_width))
            self.sg[:, 0:lb] = 0.0
            self.sg[:, ub:] = 0.0

        # extract each frame:
        featuress = np.empty((n, spec_height, real_spec_width, 1), dtype=np.float32)
        for i in range(n):
            sgstart = int(frame_hop * i * self.sampleRate / self.incr)
            sgend = sgstart + real_spec_width
            # Skip the last bits if they don't comprise a full frame:
            if sgend > np.shape(self.sg)[0]:
                print("Warning: dropping frame at", sgend, n)
                # Alternatively could adjust:
                # sgstart = np.shape(sp.sg)[0] - real_spec_width
                # sgend = np.shape(sp.sg)[0]
                i = i-1
                break
            sgRaw = self.sg[sgstart:sgend, :, np.newaxis]

            # Standardize/rescale here.
            # NOTE the resulting features are on linear scale, not dB
            maxg = np.max(sgRaw)
            featuress[i, :, :, :] = np.rot90(sgRaw / maxg)

        # NOTE using i to account for possible loop break
        # this may be needed for dealing w/ boundary issues
        # which is maybe possible if the spec window is larger than the
        # CNN frame size, or due to inconsistent rounding
        featuress = featuress[:(i+1), :, :, :]
        return featuress

    def generateFeaturesCNN2(self, seglen, real_spec_width, frame_size, frame_hop=None):
        '''
        Prepare a syllable to input to the CNN model
        Returns the features (currently the spectrogram)
        '''
        # determine the number of frames:
        if frame_hop is None:
            n = seglen // frame_size
            frame_hop = frame_size
        else:
            n = (seglen-frame_size) // frame_hop + 1
        n = int(n)

        sgRaw1 = self.spectrogram(window='Hann')
        sgRaw2 = self.spectrogram(window='Hamming')
        sgRaw3 = self.spectrogram(window='Welch')

        spec_height = np.shape(self.sg)[1]

        # extract each frame:
        featuress = np.empty((n, spec_height, real_spec_width, 3))

        for i in range(n):
            sgstart = int(frame_hop * i * self.sampleRate / self.incr)
            sgend = sgstart + real_spec_width
            # Skip the last bits if they don't comprise a full frame:
            if sgend > np.shape(self.sg)[0]:
                print("Warning: dropping frame at", sgend, n)
                # Alternatively could adjust:
                # sgstart = np.shape(sp.sg)[0] - real_spec_width
                # sgend = np.shape(sp.sg)[0]
                break

            # Standardize/rescale here.
            # NOTE the resulting features are on linear scale, not dB
            sgRaw_i = np.empty((real_spec_width, spec_height, 3), dtype=np.float32)
            sgRaw_i[:, :, 0] = sgRaw1[sgstart:sgend, :] / np.max(sgRaw1[sgstart:sgend, :])
            sgRaw_i[:, :, 1] = sgRaw2[sgstart:sgend, :] / np.max(sgRaw2[sgstart:sgend, :])
            sgRaw_i[:, :, 2] = sgRaw3[sgstart:sgend, :] / np.max(sgRaw3[sgstart:sgend, :])
            featuress[i, :, :, :] = np.rot90(sgRaw_i)

        # NOTE using i to account for possible loop break
        # this may be needed for dealing w/ boundary issues
        # which is maybe possible if the spec window is larger than the
        # CNN frame size
        featuress = featuress[:i, :, :, :]
        return featuress

