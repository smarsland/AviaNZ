
# SignalProc.py
# This file holds signal processing functions that don't use the full spectrogram or audio data
import scipy.signal as signal
import scipy.fftpack as fft
import numpy as np
import copy
import Spectrogram

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
        self.fileLength = wavobj.nframes

        self.sampleRate = wavobj.rate

        if QtMM:
            # TODO!!!
            self.audioFormat.setSampleSize(wavobj.sampwidth * 8)
            self.audioFormat.setSampleRate(self.sampleRate)
            # Only 8-bit WAVs are unsigned:
            # TODO!!
            if wavobj.sampwidth==1:
                self.audioFormat.setSampleType(QAudioFormat.UnSignedInt)
            else:
                self.audioFormat.setSampleType(QAudioFormat.SignedInt)

        # *Freq sets hard bounds, *Show can limit the spec display
        self.minFreq = 0
        self.maxFreq = self.sampleRate // 2
        self.minFreqShow = max(self.minFreq, self.minFreqShow)
        self.maxFreqShow = min(self.maxFreq, self.maxFreqShow)

        #print("a",self.sampleRate, self.fileLength, np.shape(self.data))

        if not silent:
            if QtMM:
                print("Detected format: %d channels, %d Hz, %d bit samples" % (self.audioFormat.channelCount(), self.audioFormat.sampleRate(), self.audioFormat.sampleSize()))

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
            #self.audioFormat.setSampleSize(0)
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
            print("Warning: no data set to resample")
            return
        if target==self.sampleRate:
            print("No resampling needed")
            return

        self.data = librosa.resample(self.data, orig_sr=self.sampleRate, target_sr=target)

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

def ButterworthBandpass(data,sampleRate,low=0,high=None,band=0.005):
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
        print("No data given")
        return
    if sampleRate is None:
        print("No sample rate given")
        return data
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

def FastButterworthBandpass(data,low=0,high=None):
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
        print("No data given")
        return

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

def bandpassFilter(data,sampleRate,start=0,end=-1):
    """ FIR bandpass filter
    128 taps, Hamming window, very basic.
    """

    if data is None:
        print("No data given")
        return
    if sampleRate is None:
        print("No sample rate given")
        return data
    if end==-1 or end is None:
        end = sampleRate/2
    print(start,end,sampleRate,len(data))

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
    #print("Taps:", taps)
    #ntaps, beta = signal.kaiserord(ripple_db, width)
    #taps = signal.firwin(ntaps,cutoff = [500/nyquist,8000/nyquist], window=('kaiser', beta),pass_zero=False)
    return signal.lfilter(taps, 1.0, data)

# TODO: Here or in spectrogram? Needs some work either way
# The next functions perform spectrogram inversion
def invertSpectrogram(sg,window_width=256,incr=64,nits=10, window='Hann'):
    # Assumes that this is the plain (not power) spectrogram
    # Make the spectrogram two-sided and make the values small
    sg = np.concatenate([sg, sg[:, ::-1]], axis=1)

    sg_best = copy.deepcopy(sg)
    for i in range(nits):
        invertedSgram = inversion_iteration(sg_best, incr, calculate_offset=True,set_zero_phase=(i==0), window=window)
        self.setData(invertedSgram)
        est = self.spectrogram(window_width, incr, onesided=False,need_even=True, window=window)
        phase = est / np.maximum(np.max(sg)/1E8, np.abs(est))
        sg_best = sg * phase[:len(sg)]
    invertedSgram = self.inversion_iteration(sg_best, incr, calculate_offset=True,set_zero_phase=False, window=window)
    return np.real(invertedSgram)

def inversion_iteration(sg, incr, calculate_offset=True, set_zero_phase=True, window='Hann'):
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
            print("Offset: ",offset_size)
            if offset_size <= 0:
                print("WARNING: Large step size >50\% detected! " "This code works best with high overlap - try " "with 75% or greater")
                offset_size = incr
            print(wave_start, est_start, offset_size, len(wave), len(wave_est), est_start+offset_size<len(wave_est))
            offset = self.xcorr_offset(wave[wave_start:wave_start + offset_size], wave_est[est_start:est_start + offset_size])
            print("New offset: ",offset)
        else:
            offset = 0
        print(wave_end-wave_start, len(window), est_end-est_start,len(wave_est), len(wave),est_start, offset)
        if est_end-offset >= size:
            offset+=(est_end-offset-size)
            wave_end-=(est_end-offset-size)
        wave[wave_start:wave_end] += window * wave_est[est_start - offset:est_end - offset]
        total_windowing_sum[wave_start:wave_end] += window**2 #Virginia: needed square
    wave = np.real(wave) / (total_windowing_sum + 1E-6)
    return wave

def xcorr_offset(x1, x2):
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    frame_size = len(x2)
    half = frame_size // 2
    corrs = np.convolve(x1.astype('float32'), x2[::-1].astype('float32'))
    corrs[:half] = -1E30
    corrs[-half:] = -1E30
    return corrs.argmax() - len(x1)

def medianFilter(data,width=11):
    # Median Filtering
    # Uses smaller width windows at edges to remove edge effects
    # TODO: Use abs rather than pure median?
    if data is None:
        print("No data")
        return
    mData = np.zeros(len(data))
    for i in range(width,len(data)-width):
        mData[i] = np.median(data[i-width:i+width])
    for i in range(len(data)):
        wid = min(i,len(data)-i,width)
        mData[i] = np.median(data[i - wid:i + wid])

    return mData

def wsola(x, s, win_type='hann', win_size=1024, syn_hop_size=512, tolerance=512):
    from scipy.interpolate import interp1d
    """Modify length of the audio sequence using WSOLA algorithm.
    This implementation is largely from pytsmod

    Parameters
    ----------

    start, stop : the part of the sound to play

    s : number > 0 [scalar] or numpy.ndarray [shape=(2, num_points)]
        the time stretching factor. Either a constant value (alpha)
        or an 2 x n array of anchor points which contains the sample points
        of the input signal in the first row
        and the sample points of the output signal in the second row.
    win_type : str
            type of the window function. hann and sin are available.
    win_size : int > 0 [scalar]
            size of the window function.
    syn_hop_size : int > 0 [scalar]
            hop size of the synthesis window.
            Usually half of the window size.
    tolerance : int >= 0 [scalar]
                number of samples the window positions
                in the input signal may be shifted
                to avoid phase discontinuities when overlap-adding them
                to form the output signal (given in samples).

    Returns
    -------

    y : numpy.ndarray [shape=(channel, num_samples) or (num_samples)]
        the modified output audio sequence.
    """

    x = np.expand_dims(x, 0)
    anc_points = np.array([[0, np.shape(x)[1] - 1], [0, np.ceil(s * np.shape(x)[1]) - 1]])
    #anc_points = np.array([[0, np.shape(x)[1] - 1], [0, np.ceil(s * np.shape(x)[1]) - 1]])
    n_chan = x.shape[0]
    output_length = int(anc_points[-1, -1]) + 1

    win = np.hanning(win_size)

    sw_pos = np.arange(0, output_length + win_size // 2, syn_hop_size)
    ana_interpolated = interp1d(anc_points[1, :], anc_points[0, :],
                                fill_value='extrapolate')
    aw_pos = np.round(ana_interpolated(sw_pos)).astype(int)
    ana_hop = np.insert(aw_pos[1:] - aw_pos[0: -1], 0, 0)
    
    y = np.zeros((n_chan, output_length))

    min_fac = np.min(syn_hop_size / ana_hop[1:])

    # padding the input audio sequence.
    left_pad = int(win_size // 2 + tolerance)
    right_pad = int(np.ceil(1 / min_fac) * win_size + tolerance)
    x_padded = np.pad(x, ((0, 0), (left_pad, right_pad)), 'constant')

    aw_pos = aw_pos + tolerance

    # Applying WSOLA to each channels
    for c, x_chan in enumerate(x_padded):
        y_chan = np.zeros(output_length + 2 * win_size)
        ow = np.zeros(output_length + 2 * win_size)

        delta = 0

        for i in range(len(aw_pos) - 1):
            x_adj = x_chan[aw_pos[i] + delta: aw_pos[i] + win_size + delta]
            y_chan[sw_pos[i]: sw_pos[i] + win_size] += x_adj * win
            ow[sw_pos[i]: sw_pos[i] + win_size] += win

            nat_prog = x_chan[aw_pos[i] + delta + syn_hop_size:
                            aw_pos[i] + delta + syn_hop_size + win_size]

            next_aw_range = np.arange(aw_pos[i+1] - tolerance,
                                    aw_pos[i+1] + win_size + tolerance)

            x_next = x_chan[next_aw_range]

            cross_corr = np.correlate(nat_prog, x_next)
            max_index = np.argmax(cross_corr)

            delta = tolerance - max_index

        # Calculate last frame
        x_adj = x_chan[aw_pos[-1] + delta: aw_pos[-1] + win_size + delta]
        y_chan[sw_pos[-1]: sw_pos[-1] + win_size] += x_adj * win
        ow[sw_pos[-1]: sw_pos[-1] + win_size] += + win

        ow[ow < 1e-3] = 1

        y_chan = y_chan / ow
        y_chan = y_chan[win_size // 2:]
        y_chan = y_chan[: output_length]

        y[c, :] = np.int_(y_chan)

    return y.squeeze()

def impMask(data,sampleRate,engp=90, fp=0.75):
    """
    Impulse mask
    :param engp: energy percentile (for rows of the spectrogram)
    :param fp: frequency proportion to consider it as an impulse (cols of the spectrogram)
    :return: audiodata
    """
    print('Impulse masking...')
    imps = impulse_cal(data,sampleRate, engp=engp, fp=fp)
    print('Samples to mask: ', len(data) - np.sum(imps))
    # Mask only the affected samples
    return np.multiply(data, imps)

def impulse_cal(data,sampleRate, engp=90, fp=0.75, blocksize=10):
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
    # for impulse masking
    from itertools import chain, repeat

    # Calculate window length
    w1 = np.floor(sampleRate/250)      # Window length of 1/250 sec selected experimentally
    arr = [2 ** i for i in range(5, 11)]
    pos = np.abs(arr - w1).argmin()
    window = arr[pos]

    sp = Spectrogram.Spectrogram(window, window)     # No overlap
    sp.data = data
    sp.sampleRate = sampleRate
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
    imp = countConsecutive(imp_inds, len(impulse))

    impulse = []
    for item in imp:
        if item > blocksize or item == 0:        # Note threshold - blocksize, 10 consecutive blocks ~1/25 sec
            impulse.append(1)
        else:
            impulse.append(0)

    impulse = list(chain.from_iterable(repeat(e, window) for e in impulse))  # Make it same length as self.audioData

    if len(impulse) > len(data):      # Sanity check
        impulse = impulse[0:len(data)]
    elif len(impulse) < len(data):
        gap = len(data) - len(impulse)
        impulse = np.pad(impulse, (0, gap), 'constant')

    return impulse

def countConsecutive(nums, length):
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
