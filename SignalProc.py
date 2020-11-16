
# SignalProc.py
# A variety of signal processing algorithms for AviaNZ.

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
import wavio
import librosa
import copy
import gc

from PyQt5.QtGui import QImage

QtMM = True
try:
    from PyQt5.QtMultimedia import QAudioFormat
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
# for impulse masking
from itertools import chain, repeat

class SignalProc:
    """ This class reads and holds the audiodata and spectrogram, to be used in the main interface.
    Inverse, denoise, and other processing algorithms are provided here.
    Also bandpass and Butterworth bandpass filters.
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
            self.audioFormat.setCodec("audio/pcm")
            self.audioFormat.setByteOrder(QAudioFormat.LittleEndian)
            self.audioFormat.setSampleType(QAudioFormat.SignedInt)
        #else:
            #self.audioFormat = {}

    def readWav(self, file, len=None, off=0, silent=False):
        """ Args the same as for wavio.read: filename, length in seconds, offset in seconds. """
        wavobj = wavio.read(file, len, off)
        self.data = wavobj.data

        # take only left channel
        if np.shape(np.shape(self.data))[0] > 1:
            self.data = self.data[:, 0]
        if QtMM:
            self.audioFormat.setChannelCount(1)
        #else:
            #self.audioFormat['channelCount'] = 1

        # force float type
        if self.data.dtype != 'float':
            self.data = self.data.astype('float')

        # total file length in s read from header (useful for paging)
        self.fileLength = wavobj.nframes

        self.sampleRate = wavobj.rate

        if QtMM:
            self.audioFormat.setSampleSize(wavobj.sampwidth * 8)
            self.audioFormat.setSampleRate(self.sampleRate)
        #else:
            #self.audioFormat['sampleSize'] = wavobj.sampwidth * 8
            #self.audioFormat['sampleRate'] = self.sampleRate

        # *Freq sets hard bounds, *Show can limit the spec display
        self.minFreq = 0
        self.maxFreq = self.sampleRate // 2
        self.minFreqShow = max(self.minFreq, self.minFreqShow)
        self.maxFreqShow = min(self.maxFreq, self.maxFreqShow)

        if not silent:
            if QtMM:
                print("Detected format: %d channels, %d Hz, %d bit samples" % (self.audioFormat.channelCount(), self.audioFormat.sampleRate(), self.audioFormat.sampleSize()))
            #else:
                #print("Detected format: %d channels, %d Hz, %d bit samples" % (self.audioFormat['channelCount'], self.audioFormat['sampleRate'], self.audioFormat['sampleSize']))

    def readBmp(self, file, len=None, off=0, silent=False, rotate=True):
        """ Reads DOC-standard bat recordings in 8x row-compressed BMP format.
            For similarity with readWav, accepts len and off args, in seconds.
            rotate: if True, rotates to match setImage and other spectrograms (rows=time)
                otherwise preserves normal orientation (cols=time)
        """
        # !! Important to set these, as they are used in other functions
        self.sampleRate = 176000
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
        img.convertTo(QImage.Format_Grayscale8)

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

        print(np.shape(img2))
        # Could skip that for visual mode - maybe useful for establishing contrast?
        img2[-1, :] = 254  # lowest freq bin is 0, flip that
        img2 = 255 - img2  # reverse value having the black as the most intense
        img2 = img2/np.max(img2)  # normalization
        img2 = img2[:, 1:]  # Cutting first time bin because it only contains the scale and cutting last columns
        img2 = np.repeat(img2, 8, axis=0)  # repeat freq bins 7 times to fit invertspectrogram
        print(np.shape(img2))

        self.data = []
        self.fileLength = (w-2)*self.incr + self.window_width  # in samples
        # Alternatively:
        # self.fileLength = self.convertSpectoAmpl(h-1)*self.sampleRate

        # NOTE: conversions will use self.sampleRate and self.incr, so ensure those are already set!
        # trim to specified offset and length:
        if off>0 or len is not None:
            # Convert offset from seconds to pixels
            off = int(self.convertAmpltoSpec(off))
            if len is None:
                img2 = img2[:, off:]
            else:
                # Convert length from seconds to pixels:
                len = int(self.convertAmpltoSpec(len))
                img2 = img2[:, off:(off+len)]

        if rotate:
            # rotate for display, b/c required spectrogram dimensions are:
            #  t increasing over rows, f increasing over cols
            # This will be enough if the original image was spectrogram-shape.
            img2 = np.rot90(img2, 1, (1,0))

        self.sg = img2

        if QtMM:
            self.audioFormat.setChannelCount(0)
            self.audioFormat.setSampleSize(0)
            self.audioFormat.setSampleRate(self.sampleRate)
        #else:
            #self.audioFormat['channelCount'] = 0
            #self.audioFormat['sampleSize'] = 0
            #self.audioFormat['sampleRate'] = self.sampleRate

        self.minFreq = 0
        self.maxFreq = self.sampleRate //2
        self.minFreqShow = max(self.minFreq, self.minFreqShow)
        self.maxFreqShow = min(self.maxFreq, self.maxFreqShow)

        if not silent:
            print("Detected BMP format: %d x %d px, %d colours" % (w, h, colc))
        return(0)

    def resample(self, target):
        if len(self.data)==0:
            print("Warning: no data set to resmample")
            return
        if target==self.sampleRate:
            print("No resampling needed")
            return

        self.data = librosa.core.audio.resample(self.data, self.sampleRate, target)

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
    def spectrogram(self,window_width=None,incr=None,window='Hann',sgType=None,equal_loudness=False,mean_normalise=True,onesided=True,need_even=False):
        """ Compute the spectrogram from amplitude data
        Returns the power spectrum, not the density -- compute 10.*log10(sg) 10.*log10(sg) before plotting.
        Uses absolute value of the FT, not FT*conj(FT), 'cos it seems to give better discrimination
        Options: multitaper version, but it's slow, mean normalised, even, one-sided.
        This version is faster than the default versions in pylab and scipy.signal
        Assumes that the values are not normalised.
        """
        if self.data is None or len(self.data)==0:
            print("ERROR: attempted to calculate spectrogram without audiodata")
            return

        #S = librosa.feature.melspectrogram(self.data, sr=self.sampleRate, power=1)
        #log_S = librosa.amplitude_to_db(S, ref=np.max)
        #self.sg = librosa.pcen(S * (2**31))
        #return self.sg.T
        if sgType is None:
            sgType = 'Standard'

        if window_width is None:
            window_width = self.window_width
        if incr is None:
            incr = self.incr

        # clean handling of very short segments:
        if len(self.data) <= window_width:
            window_width = len(self.data) - 1

        self.sg = np.copy(self.data)
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
        if sgType=='Multi-tapered':
            if specExtra:
                [tapers, eigen] = dpss(window_width, 2.5, 4)
                counter = 0
                out = np.zeros((len(starts),window_width // 2))
                for start in starts:
                    Sk, weights, eigen = pmtm(self.sg[start:start + window_width], v=tapers, e=eigen, show=False)
                    Sk = abs(Sk)**2
                    Sk = np.mean(Sk.T * weights, axis=1)
                    out[counter:counter + 1,:] = Sk[window_width // 2:].T
                    counter += 1
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
            times = np.tile(np.arange(0, (len(self.data) - window_width)/self.sampleRate, incr/self.sampleRate) + window_width/self.sampleRate/2,(np.shape(delay)[1],1)).T + delay*window_width/self.sampleRate
            self.sg,_,_ = np.histogram2d(times.flatten(),CIF.flatten(),weights=np.abs(ft).flatten(),bins=np.shape(ft))

            self.sg = np.absolute(self.sg[:, :window_width //2]) + 0.1

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
        return self.sg

    def scalogram(self,wavelet='morl'):
        # Compute the wavelet scalogram
        import pywt
        scalogram, freqs = pywt.cwt(self.audiodata, widths, wavelet)

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

    def bandpassFilter(self,data=None,sampleRate=None,start=0,end=None):
        """ FIR bandpass filter
        128 taps, Hamming window, very basic.
        """

        if data is None:
            data = self.data
        if sampleRate is None:
            sampleRate = self.sampleRate
        if end is None:
            end = sampleRate/2
        start = max(start,0)
        end = min(end,sampleRate/2)

        if start == 0 and end == sampleRate/2:
            print("No filter needed!")
            return data

        nyquist = sampleRate/2
        ntaps = 129

        if start == 0:
            # Low pass
            taps = signal.firwin(ntaps, cutoff=[end / nyquist], window=('hamming'), pass_zero=True)
        elif end == sampleRate/2:
            # High pass
            taps = signal.firwin(ntaps, cutoff=[start / nyquist], window=('hamming'), pass_zero=False)
        else:
            # Bandpass
            taps = signal.firwin(ntaps, cutoff=[start / nyquist, end / nyquist], window=('hamming'), pass_zero=False)
        #ntaps, beta = signal.kaiserord(ripple_db, width)
        #taps = signal.firwin(ntaps,cutoff = [500/nyquist,8000/nyquist], window=('kaiser', beta),pass_zero=False)
        return signal.lfilter(taps, 1.0, data)

    def ButterworthBandpass(self,data,sampleRate,low=0,high=None,band=0.005):
        """ Basic IIR bandpass filter.
            Identifies order of filter, max 10. If single-stage polynomial is unstable,
            switches to order 30, second-order filter.
            Args:
            1-2. data and sample rate.
            3-4. Low and high pass frequencies in Hz
            5. difference between stopband and passband, in fraction of Nyquist.
            Filter will lose no more than 3 dB in freqs [low,high], and attenuate
            at least 40 dB outside [low-band*Fn, high+band*Fn].

            Does double-pass filtering - slower, but keeps original phase.
        """

        if data is None:
            data = self.data
        if sampleRate is None:
            sampleRate = self.sampleRate
        nyquist = sampleRate/2

        if high is None:
            high = nyquist
        low = max(low,0)
        high = min(high,nyquist)

        # convert freqs to fractions of Nyquist:
        lowPass = low/nyquist
        highPass = high/nyquist
        lowStop = lowPass-band
        highStop = highPass+band
        # safety checks for values near edges
        if lowStop<=0:
            lowStop = lowPass/2
        if highStop>=1:
            highStop = (1+highPass)/2

        if lowPass == 0 and highPass == 1:
            print("No filter needed!")
            return data
        elif lowPass == 0:
            # Low pass
            # calculate the best order
            order,wN = signal.buttord(highPass, highStop, 3, 40)
            if order>10:
                order=10
            b, a = signal.butter(order,wN, btype='lowpass')
        elif highPass == 1:
            # High pass
            # calculate the best order
            order,wN = signal.buttord(lowPass, lowStop, 3, 40)
            if order>10:
                order=10
            b, a = signal.butter(order,wN, btype='highpass')
        else:
            # Band pass
            # calculate the best order
            order,wN = signal.buttord([lowPass, highPass], [lowStop, highStop], 3, 40)
            if order>10:
                order=10
            b, a = signal.butter(order,wN, btype='bandpass')

        # check if filter is stable
        filterUnstable = np.any(np.abs(np.roots(a))>1)
        if filterUnstable:
            # redesign to SOS and filter.
            # uses order=30 because why not
            print("single-stage filter unstable, switching to SOS filtering")
            if lowPass == 0:
                sos = signal.butter(30, wN, btype='lowpass', output='sos')
            elif highPass == 1:
                sos = signal.butter(30, wN, btype='highpass', output='sos')
            else:
                sos = signal.butter(30, wN, btype='bandpass', output='sos')

            # do the actual filtering
            data = signal.sosfiltfilt(sos, data)
        else:
            # do the actual filtering
            data = signal.filtfilt(b, a, data)

        return data

    def FastButterworthBandpass(self,data,low=0,high=None):
        """ Basic IIR bandpass filter.
            Streamlined to be fast - for use in antialiasing etc.
            Tries to construct a filter of order 7, with critical bands at +-0.002 Fn.
            This corresponds to +- 16 Hz or so.
            If single-stage polynomial is unstable,
            switches to order 30, second-order filter.
            Args:
            1-2. data and sample rate.
            3-4. Low and high pass frequencies in fraction of Nyquist

            Does single-pass filtering, so does not retain phase.
        """

        if data is None:
            data = self.data

        # convert freqs to fractions of Nyquist:
        lowPass = max(low-0.002, 0)
        highPass = min(high+0.002, 1)

        if lowPass == 0 and highPass == 1:
            print("No filter needed!")
            return data
        elif lowPass == 0:
            # Low pass
            b, a = signal.butter(7, highPass, btype='lowpass')
        elif highPass == 1:
            # High pass
            b, a = signal.butter(7, lowPass, btype='highpass')
        else:
            # Band pass
            b, a = signal.butter(7, [lowPass, highPass], btype='bandpass')

        # check if filter is stable
        filterUnstable = True
        try:
            filterUnstable = np.any(np.abs(np.roots(a))>1)
        except Exception as e:
            print("Warning:", e)
            filterUnstable = True
        if filterUnstable:
            # redesign to SOS and filter.
            # uses order=30 because why not
            print("single-stage filter unstable, switching to SOS filtering")
            if lowPass == 0:
                sos = signal.butter(30, highPass, btype='lowpass', output='sos')
            elif highPass == 1:
                sos = signal.butter(30, lowPass, btype='highpass', output='sos')
            else:
                sos = signal.butter(30, [lowPass, highPass], btype='bandpass', output='sos')

            # do the actual filtering
            data = signal.sosfilt(sos, data)
        else:
            data = signal.lfilter(b, a, data)

        return data

    # The next functions perform spectrogram inversion
    def invertSpectrogram(self,sg,window_width=256,incr=64,nits=10, window='Hann'):
        # Assumes that this is the plain (not power) spectrogram
        # Make the spectrogram two-sided and make the values small
        sg = np.concatenate([sg, sg[:, ::-1]], axis=1)

        sg_best = copy.deepcopy(sg)
        for i in range(nits):
            invertedSgram = self.inversion_iteration(sg_best, incr, calculate_offset=True,set_zero_phase=(i==0), window=window)
            self.setData(invertedSgram)
            est = self.spectrogram(window_width, incr, onesided=False,need_even=True, window=window)
            phase = est / np.maximum(np.max(sg)/1E8, np.abs(est))
            sg_best = sg * phase[:len(sg)]
        invertedSgram = self.inversion_iteration(sg_best, incr, calculate_offset=True,set_zero_phase=False, window=window)
        return np.real(invertedSgram)

    def inversion_iteration(self,sg, incr, calculate_offset=True, set_zero_phase=True, window='Hann'):
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
        wave = np.zeros((np.shape(sg)[0] * incr + size),dtype='float64')
        # Getting overflow warnings with 32 bit...
        #wave = wave.astype('float64')
        total_windowing_sum = np.zeros((np.shape(sg)[0] * incr + size))
        #Virginia: adding different windows

        
       # Set of window options
        if window=='Hann':
            # This is the Hann window
            window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(size) / (size - 1)))
        elif window=='Parzen':
            # Parzen (window_width even)
            n = np.arange(size) - 0.5*size
            window = np.where(np.abs(n)<0.25*size,1 - 6*(n/(0.5*size))**2*(1-np.abs(n)/(0.5*size)), 2*(1-np.abs(n)/(0.5*size))**3)
        elif window=='Welch':
            # Welch
            window = 1.0 - ((np.arange(size) - 0.5*(size-1))/(0.5*(size-1)))**2
        elif window=='Hamming':
            # Hamming
            alpha = 0.54
            beta = 1.-alpha
            window = alpha - beta*np.cos(2 * np.pi * np.arange(size) / (size - 1))
        elif window=='Blackman':
            # Blackman
            alpha = 0.16
            a0 = 0.5*(1-alpha)
            a1 = 0.5
            a2 = 0.5*alpha
            window = a0 - a1*np.cos(2 * np.pi * np.arange(size) / (size - 1)) + a2*np.cos(4 * np.pi * np.arange(size) / (size - 1))
        elif window=='BlackmanHarris':
            # Blackman-Harris
            a0 = 0.358375
            a1 = 0.48829
            a2 = 0.14128
            a3 = 0.01168
            window = a0 - a1*np.cos(2 * np.pi * np.arange(size) / (size - 1)) + a2*np.cos(4 * np.pi * np.arange(size) / (size - 1)) - a3*np.cos(6 * np.pi * np.arange(size) / (size - 1))
        elif window=='Ones':
            window = np.ones(size)
        else:
            print("Unknown window, using Hann")
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

            wave_est = np.real(fft.ifft(spectral_slice))[::-1]
            if calculate_offset and i > 0:
                offset_size = size - incr
                if offset_size <= 0:
                    #print("WARNING: Large step size >50\% detected! " "This code works best with high overlap - try " "with 75% or greater")
                    offset_size = incr
                offset = self.xcorr_offset(wave[wave_start:wave_start + offset_size], wave_est[est_start:est_start + offset_size])
            else:
                offset = 0
            wave[wave_start:wave_end] += window * wave_est[est_start - offset:est_end - offset]
            total_windowing_sum[wave_start:wave_end] += window**2 #Virginia: needed square
        wave = np.real(wave) / (total_windowing_sum + 1E-6)
        return wave

    def xcorr_offset(self,x1, x2):
        x1 = x1 - x1.mean()
        x2 = x2 - x2.mean()
        frame_size = len(x2)
        half = frame_size // 2
        corrs = np.convolve(x1.astype('float32'), x2[::-1].astype('float32'))
        corrs[:half] = -1E30
        corrs[-half:] = -1E30
        return corrs.argmax() - len(x1)

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

    # Could be either features of signal processing things. Anyway, they are here -- spectral derivatives and extensions
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

    def drawFundFreq(self, seg):
        # produces marks of fundamental freq to be drawn on the spectrogram.
        pitch, starts, _, W = seg.yin()
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

        # then map starts from samples to spec windows
        starts = starts / self.incr
        # then convert segments back to positions in each array:
        out = []
        for s in segs:
            # convert [s, e] to [s s+1 ... e-1 e]
            i = np.arange(s[0], s[1])
            # retrieve all pitch and start positions corresponding to this segment
            pitchSeg = pitch[i]
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
            i = i[trimst:trime]

            out.append((starts[i], y))
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
            self.data = self.bandpassFilter(self.data,self.sampleRate, start=start, end=end)
        elif str(alg) == "Butterworth Bandpass":
            self.data = self.ButterworthBandpass(self.data, self.sampleRate, low=start, high=end)
        else:
            # Median Filter
            self.data = self.medianFilter(self.data,int(str(width)))

    def impMask(self, engp=90, fp=0.75):
        """
        Impulse mask
        :param engp: energy percentile (for rows of the spectrogram)
        :param fp: frequency proportion to consider it as an impulse (cols of the spectrogram)
        :return: audiodata
        """
        print('Impulse masking...')
        imps = self.impulse_cal(fs=self.sampleRate, engp=engp, fp=fp)
        print('Samples to mask: ', len(self.data) - np.sum(imps))
        # Mask only the affected samples
        return np.multiply(self.data, imps)

    def impulse_cal(self, fs, engp=90, fp=0.75, blocksize=10):
        """
        Find sections where impulse sounds occur e.g. clicks
        window  -   window length (no overlap)
        engp    -   energy percentile (thr), the percentile of energy to inform that a section got high energy across
                    frequency bands
        fp      -   frequency percentage (thr), the percentage of frequency bands to have high energy to mark a section
                    as having impulse noise
        blocksize - max number of consecutive blocks, 10 consecutive blocks (~1/25 sec) is a good value, to not to mask
                    very close-range calls
        :return: a binary list of length len(data) indicating presence of impulsive noise (0) otherwise (1)
        """

        # Calculate window length
        w1 = np.floor(fs/250)      # Window length of 1/250 sec selected experimentally
        arr = [2 ** i for i in range(5, 11)]
        pos = np.abs(arr - w1).argmin()
        window = arr[pos]

        sp = SignalProc(window, window)     # No overlap
        sp.data = self.data
        sp.sampleRate = self.sampleRate
        sg = sp.spectrogram()

        # For each frq band get sections where energy exceeds some (90%) percentile, engp
        # and generate a binary spectrogram
        sgb = np.zeros((np.shape(sg)))
        ep = np.percentile(sg, engp, axis=0)    # note thr - 90% for energy percentile
        for y in range(np.shape(sg)[1]):
            ey = sg[:, y]
            sgb[np.where(ey > ep[y]), y] = 1

        # If lots of frq bands got 1 then predict a click
        # 1 - presence of impulse noise, 0 - otherwise here
        impulse = np.where(np.count_nonzero(sgb, axis=1) > np.shape(sgb)[1] * fp, 1, 0)     # Note thr fp

        # When an impulsive noise detected, it's better to check neighbours to make sure its not a bird call
        # very close to the microphone.
        imp_inds = np.where(impulse > 0)[0].tolist()
        imp = self.countConsecutive(imp_inds, len(impulse))

        impulse = []
        for item in imp:
            if item > blocksize or item == 0:        # Note threshold - blocksize, 10 consecutive blocks ~1/25 sec
                impulse.append(1)
            else:
                impulse.append(0)

        impulse = list(chain.from_iterable(repeat(e, window) for e in impulse))  # Make it same length as self.audioData

        if len(impulse) > len(self.data):      # Sanity check
            impulse = impulse[0:len(self.data)]
        elif len(impulse) < len(self.data):
            gap = len(self.data) - len(impulse)
            impulse = np.pad(impulse, (0, gap), 'constant')

        return impulse

    def countConsecutive(self, nums, length):
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        edges = list(zip(edges, edges))
        edges_reps = [item[1] - item[0] + 1 for item in edges]
        res = np.zeros((length)).tolist()
        t = 0
        for item in edges:
            for i in range(item[0], item[1]+1):
                res[i] = edges_reps[t]
            t += 1
        return res

