# Version 0.5 10/7/17
# Author: Stephen Marsland

import numpy as np
import scipy.signal as signal
import spectrum

class SignalProc:
    """ This class implements various signal processing algorithms for the AviaNZ interface.
    Primary one is the spectrogram, together with its inverse.
    Also bandpass and Butterworth bandpass filters.
    Primary parameters are the width of a spectrogram window (window_width) and the shift between them (incr)
    """

    def __init__(self,data=[],sampleRate=0,window_width=256,incr=128):
        self.window_width=window_width
        self.incr=incr

        if data != []:
            self.data = data
            self.sampleRate = sampleRate

    def setNewData(self,data,sampleRate):
        # Does what it says. To be called when a new sound file is loaded
        self.data = data
        self.sampleRate = sampleRate

    def setWidth(self,window_width,incr):
        # Does what it says. Called when the user modifies the spectrogram parameters
        self.window_width = window_width
        self.incr = incr

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

    def spectrogram(self,data,window_width=None,incr=None,window='Hann',equal_loudness=False,mean_normalise=True,onesided=True,multitaper=False,need_even=False):
        """ Compute the spectrogram from amplitude data
        Returns the power spectrum, not the density -- compute 10.*log10(sg) before plotting.
        Uses absolute value of the FT, not FT*conj(FT), 'cos it seems to give better discrimination
        Options: multitaper version, but it's slow, mean normalised, even, one-sided.
        This version is faster than the default versions in pylab and scipy.signal
        Assumes that the values are not normalised.
        TODO: Makes a copy of the data and uses that to ensure can be inverted. This is memory wasteful. Is that a problem?
        """
        if data is None:
            print ("Error")

        datacopy = data.astype('float')

        if window_width is None:
            window_width = self.window_width
        if incr is None:
            incr = self.incr
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
            datacopy = self.equalLoudness(datacopy)

        if mean_normalise:
            datacopy -= datacopy.mean()

        starts = range(0, len(datacopy) - window_width, incr)
        if multitaper:
            from spectrum import dpss, pmtm
            [tapers, eigen] = dpss(window_width, 2.5, 4)
            counter = 0
            sg = np.zeros((len(starts),window_width // 2))
            for start in starts:
                Sk, weights, eigen = pmtm(datacopy[start:start + window_width], v=tapers, e=eigen, show=False)
                Sk = abs(Sk)**2
                Sk = np.mean(Sk.T * weights, axis=1)
                sg[counter:counter + 1,:] = Sk[window_width // 2:].T
                counter += 1
            sg = np.fliplr(sg)
        else:
            if need_even:
                starts = np.hstack((starts, np.zeros((window_width - len(datacopy) % window_width))))

            ft = np.zeros((len(starts), window_width))
            for i in starts:
                ft[i // incr, :] = window * datacopy[i:i + window_width]
            ft = np.fft.fft(ft)
            if onesided:
                sg = np.absolute(ft[:, :window_width // 2])
            else:
                sg = np.absolute(ft)
            #sg = (ft*np.conj(ft))[:,window_width // 2:].T
        return sg

    def bandpassFilter(self,data=None,sampleRate=None,start=0,end=None,minFreq=0,maxFreq=None):
        """ FIR bandpass filter
        128 taps, Hamming window, very basic.
        """
        if data is None:
            data = self.data
        if sampleRate is None:
            sampleRate = self.sampleRate
        if end is None:
            end = self.sampleRate/2
        if maxFreq is None:
            maxFreq = self.sampleRate/2
        start = max(start,0,minFreq)
        end = min(end,maxFreq,self.sampleRate/2)

        #print(start,end,minFreq,maxFreq)

        if start == minFreq and end == maxFreq:
            print("No filter needed!")
            return data

        nyquist = self.sampleRate/2
        ntaps = 128

        if start == minFreq:
            # Low pass
            #print("Low")
            taps = signal.firwin(ntaps, cutoff=[end / nyquist], window=('hamming'), pass_zero=True)
        elif end == maxFreq:
            # High pass
            #print("High")
            taps = signal.firwin(ntaps, cutoff=[start / nyquist], window=('hamming'), pass_zero=False)
        else:
            # Bandpass
            #print("Band")
            taps = signal.firwin(ntaps, cutoff=[start / nyquist, end / nyquist], window=('hamming'), pass_zero=False)
        #ntaps, beta = signal.kaiserord(ripple_db, width)
        #taps = signal.firwin(ntaps,cutoff = [500/nyquist,8000/nyquist], window=('kaiser', beta),pass_zero=False)
        return signal.lfilter(taps, 1.0, data)

    def ButterworthBandpass(self,data,sampleRate,low=0,high=None,minFreq=0,maxFreq=None,order=10,band=50):
        """ Basic IIR bandpass filter.
        Identifies order of filter, max 10.

        """
        if data is None:
            data = self.data
        if sampleRate is None:
            sampleRate = self.sampleRate
        if high is None:
            high = sampleRate/2
        if maxFreq is None:
            maxFreq = sampleRate/2
        low = max(low,0,minFreq)
        high = min(high,maxFreq,sampleRate/2)

        if low == minFreq and high == maxFreq:
            print("No filter needed!")
            return data

        nyquist = sampleRate/2

        if low == minFreq:
            # Low pass
            cut1 = high/nyquist
            cut2 = (high+band)/nyquist
            # calculate the best order
            order,wN = signal.buttord(cut1, cut2, 3, band)
            if order>10:
                order=10
            b, a = signal.butter(order,wN,btype='lowpass')
        elif high == maxFreq:
            # High pass
            cut1 = low/nyquist
            cut2 = (low-band)/nyquist
            # calculate the best order
            order,wN = signal.buttord(cut1, cut2, 3, band)
            if order>10:
                order=10
            b, a = signal.butter(order,wN, btype='highpass')
        else:
            # Band pass
            lowPass = low/nyquist
            highPass = high/nyquist
            lowStop = (low-band)/nyquist
            highStop = (high+band)/nyquist
            # calculate the best order
            order,wN = signal.buttord([lowPass, highPass], [lowStop, highStop], 3, band)
            if order>10:
                order=10
            b, a = signal.butter(order,wN, btype='bandpass')
            #b, a = signal.butter(order,[lowPass, highPass], btype='bandpass')

        return signal.filtfilt(b, a, data)

    # The next functions perform spectrogram inversion

    def show_invS(self):
        print("Inverting spectrogam with window ", self.window_width, " and increment ", int(self.window_width/4.))
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

    def spectralDerivatives(self):
        # Easy version -- compute horizontal and vertical derivatives
        sg = self.spectrogram(self.data)
        sgderivh = np.roll(sg,-1,axis=0)-sg
        sgderivh -= np.min(sgderivh)
        sgderivv = np.roll(sg,-1,axis=1)-sg
        sgderivv -= np.min(sgderivv)
        sgderivb = np.sqrt(sgderivh**2 + sgderivv**2)

        return sgderivh, sgderivv, sgderivb

