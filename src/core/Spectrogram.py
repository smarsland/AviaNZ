
# Spectrogram.py
# Version 3.4 18/12/24
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

import numpy as np
import scipy.signal as signal
import pyfftw as fft
from scipy.stats import boxcox
import resampy
import copy
import gc
from src.core import SignalProc
from src.core import AudioLoader
from PIL import Image
from scipy.signal import medfilt

specExtra = True

class Spectrogram:
    """Spectrogram computation and analysis for audio data.
    
    Key parameters: window_width (samples per window), incr (step size).
    FFT size is 2*window_width with zero-padding.
    """

    def __init__(self, window_width=256, incr=128, minFreqShow=0, maxFreqShow=float("inf")):
        self.window_width=window_width
        self.incr=incr
        self.minFreqShow = minFreqShow
        self.maxFreqShow = maxFreqShow
        self.audio_data = None
        self.audio_loader = AudioLoader.AudioLoader()

    def readSoundFile(self, filepath, duration=None, offset=0, silent=False, **kwargs):
        """Load audio file using AudioLoader or BMP file directly.
        
        Args:
            filepath: Path to audio or BMP file
            duration: Duration to read in seconds (None for all)
            offset: Offset to start reading in seconds  
            silent: Suppress output messages
            **kwargs: Format-specific arguments (rotate, repeat for BMP)
        """
        # Check if it's a BMP file - handle directly since it's spectrogram data
        if filepath.lower().endswith('.bmp'):
            return self.load_bmp(filepath, duration, offset, silent, **kwargs)
        
        # For audio files, use AudioLoader
        loaded_data = self.audio_loader.load_audio(filepath, duration, offset, silent)
        
        # Store reference to AudioData - it has all the format info built in
        self.audio_data = loaded_data
        self.fileLength = len(loaded_data.data) if loaded_data.data is not None else 0
        self.minFreq = 0
        self.maxFreq = loaded_data.sample_rate // 2
        
        # Update frequency bounds for display
        self.minFreqShow = max(self.minFreq, self.minFreqShow)
        self.maxFreqShow = min(self.maxFreq, self.maxFreqShow)
        
    def load_bmp(self, filepath, duration=None, offset=0, silent=False, **kwargs):
        """Load BMP file (DOC bat recording format) directly as spectrogram data."""
        rotate = kwargs.get('rotate', True)
        repeat = kwargs.get('repeat', True)
        
        import numpy as np
        # Load BMP using Pillow and convert to 8-bit grayscale numpy array
        pil = Image.open(filepath)
        if pil.mode != 'L':
            if not silent:
                print("Warning: image provided not in 8-bit grayscale, converting; information may be lost")
            pil = pil.convert('L')
        img2 = np.array(pil)
        h, w = img2.shape
        
        # Determine if original image was rotated
        if h == 64:
            # standard DoC format
            pass
        elif w == 64:
            # DoC format, rotated at -90*
            img2 = np.rot90(img2, 1, (1, 0))
            w, h = h, w
        else:
            raise ValueError("Image does not appear to be in DOC format!")
        
        # Process image
        img2[-1, :] = 254  # lowest freq bin is 0, flip that
        img2 = 255 - img2  # reverse value having the black as the most intense
        img2 = img2 / np.max(img2)  # normalization
        
        if repeat:
            img2 = np.repeat(img2, 8, axis=0)  # repeat freq bins 8 times

        # Calculate file length
        file_length = (w - 2) * 512 + 256  # incr=512, window_width=256 for BMP
        
        # Create AudioData for BMP (core use only)
        # BMP files don't have actual audio, so create dummy AudioData with no data array
        from src.core.AudioData import AudioData
        self.audio_data = AudioData(data=None, sample_rate=176000,
                                     sample_format='Int16', sample_size=16, channels=0)
        
        # Trim to specified offset and duration
        if offset > 0 or duration is not None:
            # Convert offset from seconds to pixels
            offset_px = int(offset * 176000 / 512)
            if duration is None:
                img2 = img2[:, offset_px:]
            else:
                # Convert length from seconds to pixels
                duration_px = int(duration * 176000 / 512)
                img2 = img2[:, offset_px:(offset_px + duration_px)]
        
        if rotate:
            # Rotate for display (t increasing over rows, f increasing over cols)
            img2 = np.rot90(img2, 1, (1, 0))
        
        if not silent:
            print(f"Detected BMP format: {w} x {h} px")
        
        # Store spectrogram data directly
        self.sg = img2
        self.fileLength = file_length
        self.minFreq = 0
        self.maxFreq = 88000  # 176000 / 2
        
        # Update frequency bounds for display
        self.minFreqShow = max(self.minFreq, self.minFreqShow)
        self.maxFreqShow = min(self.maxFreq, self.maxFreqShow)
        
        return 0  # BMP success indicator
        
    def readBmp(self, filepath, duration=None, offset=0, silent=False, rotate=True, repeat=True):
        """Load BMP file - delegates to readSoundFile."""
        return self.readSoundFile(filepath, duration, offset, silent, rotate=rotate, repeat=repeat)

    def resample(self, target):
        if self.audio_data.data is None or len(self.audio_data.data)==0:
            print("Warning: data is empty")
            return
        if target==self.audio_data.sample_rate:
            print("No resampling needed")
            return

        resampled_data = resampy.resample(self.audio_data.data, sr_orig=self.audio_data.sample_rate, sr_new=target)
        self.audio_data.data = resampled_data
        self.audio_data.sample_rate = target

        self.minFreq = 0
        self.maxFreq = self.audio_data.sample_rate // 2
        self.fileLength = len(resampled_data)

    def convertAmpltoSpec(self, x):
        """ Unit conversion, for easier use wherever spectrograms are needed """
        return x*self.audio_data.sample_rate/self.incr
        #return x*self.sampleRate/self.incr

    def convertSpectoAmpl(self,x):
        """ Unit conversion """
        return x*self.incr/self.audio_data.sample_rate
        #return x*self.incr/self.sampleRate

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
            maxfreq = self.audio_data.sample_rate/2
            #maxfreq = self.sampleRate/2
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
        if self.audio_data is None:
            from src.core.AudioData import AudioData
            if sampleRate is None:
                raise ValueError("sampleRate must be provided when creating new AudioData")
            self.audio_data = AudioData(
                data=audiodata,
                sample_rate=sampleRate,
                sample_format='float32',
                sample_size=32,
                channels=1
            )
        else:
            try:
                self.audio_data.data = audiodata
            except Exception:
                self.audio_data.data = audiodata
        if sampleRate is not None:
            self.audio_data.sample_rate = sampleRate

    def SnNR(self,startSignal,startNoise):
        # Compute the estimated signal-to-noise ratio
        pS = np.sum(self.audio_data.data[startSignal:startSignal+self.length]**2)/self.length
        pN = np.sum(self.audio_data.data[startNoise:startNoise+self.length]**2)/self.length
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

    def create_window(self, window_type, window_width):
        """Create windowing function."""
        n = np.arange(window_width)
        
        if window_type == 'Hann':
            return 0.5 * (1 - np.cos(2 * np.pi * n / (window_width - 1)))
        elif window_type == 'Hamming':
            # Hamming window
            alpha = 0.54
            beta = 1. - alpha
            return alpha - beta * np.cos(2 * np.pi * n / (window_width - 1))
        elif window_type == 'Blackman':
            # Blackman window
            alpha = 0.16
            a0 = 0.5 * (1 - alpha)
            a1 = 0.5
            a2 = 0.5 * alpha
            return (a0 - a1 * np.cos(2 * np.pi * n / (window_width - 1)) + 
                   a2 * np.cos(4 * np.pi * n / (window_width - 1)))
        elif window_type == 'BlackmanHarris':
            # Blackman-Harris window
            a0, a1, a2, a3 = 0.358375, 0.48829, 0.14128, 0.01168
            return (a0 - a1 * np.cos(2 * np.pi * n / (window_width - 1)) + 
                   a2 * np.cos(4 * np.pi * n / (window_width - 1)) - 
                   a3 * np.cos(6 * np.pi * n / (window_width - 1)))
        elif window_type == 'Welch':
            return 1.0 - ((n - 0.5 * (window_width - 1)) / (0.5 * (window_width - 1))) ** 2
        elif window_type == 'Parzen':
            n_centered = n - 0.5 * window_width
            quarter_width = 0.25 * window_width
            half_width = 0.5 * window_width
            
            # Two-piece definition based on distance from center
            condition = np.abs(n_centered) < quarter_width
            return np.where(condition,
                           1 - 6 * (n_centered / half_width) ** 2 * (1 - np.abs(n_centered) / half_width),
                           2 * (1 - np.abs(n_centered) / half_width) ** 3)
        elif window_type == 'Ones':
            return np.ones(window_width)
        else:
            return 0.5 * (1 - np.cos(2 * np.pi * n / (window_width - 1))) # Hann

    def compute_standard_spectrogram(self, data, window, window_width, incr, onesided=True, complex_values=False, need_even=False):
        """Standard STFT with zero-padding. FFT size = 2*window_width."""
        starts = range(0, len(data) - window_width + 1, incr)
        if need_even:
            starts = np.hstack((starts, np.zeros((window_width - len(data) % window_width), dtype=int)))

        sg = np.zeros((len(starts), 2*window_width), dtype=complex)
        fft_buffer = np.zeros(2 * window_width)
        
        for i, start_idx in enumerate(starts):
            fft_buffer.fill(0.0)
            center_start = window_width // 2
            fft_buffer[center_start:center_start + window_width] = window * data[start_idx:start_idx + window_width]
            fft_buffer = fft.interfaces.scipy_fft.fftshift(fft_buffer)
            fft_buffer = np.roll(fft_buffer, -1)
            sg[i, :] = fft.interfaces.scipy_fft.fft(fft_buffer)

        if onesided:
            sg = sg[:, :window_width]
        if not complex_values:
            sg = np.absolute(sg)
        return sg

    def compute_multitaper_spectrogram(self, data, window_width, incr, singleIm=True):
        """Multi-tapered spectrogram."""
        if not specExtra:
            print("Multi-taper option not available")
            return np.array([])
            
        try:
            import dpss
        except ImportError:
            print("dpss module not found")
            return np.array([])
            
        starts = range(0, len(data) - window_width + 1, incr)
        [tapers, eigen] = dpss.dpss(window_width, 2.5, 3)
        out = np.zeros(shape=(len(starts), window_width, 3))
        
        for i, start in enumerate(starts):
            Sk, weights, eigen = dpss.pmtm(data[start:start + window_width], v=tapers, e=eigen, show=False, NFFT=window_width)
            for taper in range(3):
                out[i, :, taper] = (abs(Sk[taper]) ** 2)[:window_width]
            
        return np.squeeze(np.sum(out, axis=2)) if singleIm else out

    def compute_reassigned_spectrogram(self, data, window, window_width, incr):
        """Reassigned spectrogram."""
        starts = range(0, len(data) - window_width + 1, incr)
        ft = np.zeros((len(starts), window_width), dtype='complex')
        ft2 = np.zeros((len(starts), window_width), dtype='complex')
        
        for i in starts:
            ft[i // incr, :] = fft.interfaces.scipy_fft.fft(window * data[i:i + window_width])[:window_width]
            ft2[i // incr, :] = fft.interfaces.scipy_fft.fft(window * np.roll(data[i:i + window_width], 1))[:window_width]

        CIF = np.mod(np.angle(ft * np.conj(ft2)) / (2 * np.pi), 1.0)
        delay = (0.5 - np.mod(np.angle(ft * np.conj(np.roll(ft, 1, axis=1))) / (2 * np.pi), 1.0))

        sample_rate = self.audio_data.sample_rate
        times = (np.tile(np.arange(0, (len(data) - window_width) / sample_rate, incr / sample_rate) + 
                        window_width / sample_rate / 2, (np.shape(delay)[1], 1)).T + 
                delay * window_width / sample_rate)

        sg, _, _ = np.histogram2d(times.flatten(), CIF.flatten(), 
                                weights=np.abs(ft).flatten(), bins=np.shape(ft))
        return np.absolute(sg[:, :window_width])

    # from memory_profiler import profile
    # fp = open('memory_profiler_sp.log', 'w+')
    # @profile(stream=fp)
    def spectrogram(self, window_width=None, incr=None, window='Hann', sgType='Standard', 
                   sgScale='Linear', nfilters=128, equal_loudness=False, mean_normalise=True, 
                   onesided=True, need_even=False, start=None, complex_values=False, 
                   stop=None, singleIm=True):
        """Compute spectrogram using STFT. FFT size = 2*window_width with zero-padding."""
        # Work on a copy so we don't mutate the stored AudioData buffer
        if start is None:
            data = self.audio_data.data.copy() if self.audio_data.data is not None else None
        else:
            # TODO: Error checking
            data = self.audio_data.data[start:stop].copy()
        if data is None or len(data)==0:
            print("ERROR: attempted to calculate spectrogram without audiodata")
            return

        if window_width is None:
            window_width = self.window_width
        if incr is None:
            incr = self.incr

        # clean handling of very short segments:
        if len(data) <= window_width:
            window_width = len(data) - 1

        # Always initialize as complex since FFT returns complex values
        self.sg = np.zeros((((len(data)-window_width)//incr)+1,2*window_width), dtype=complex)

        # Create the windowing function
        window = self.create_window(window, window_width)

        if equal_loudness:
            data = self.equalLoudness(data)

        # Do not mutate the underlying audio buffer when normalising
        if mean_normalise and data is not None:
            data = data - data.mean()

        # Compute the appropriate spectrogram type
        if sgType == 'Multi-tapered':
            self.sg = self.compute_multitaper_spectrogram(data, window_width, incr, singleIm)
        elif sgType == 'Reassigned':
            self.sg = self.compute_reassigned_spectrogram(data, window, window_width, incr)
        else:  # Standard spectrogram
            # Use the refactored standard FFT method
            self.sg = self.compute_standard_spectrogram(
                data, window, window_width, incr, 
                onesided=onesided, complex_values=complex_values, need_even=need_even
            )

        if sgScale == 'Mel Frequency':
            self.convertToMel(filt='mel',nfilters=nfilters,minfreq=0,maxfreq=None,normalise=True)
        elif sgScale == 'Bark Frequency':
            self.convertToMel(filt='bark',nfilters=nfilters,minfreq=0,maxfreq=None,normalise=True)

        print("SPECTROGRAM OUTPUT SHAPE:",self.sg.shape)
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
            s = 1 - np.exp( -self.incr / (t*self.audio_data.sample_rate))
            #s = 1 - np.exp( -self.incr / (t*self.sampleRate))
            M = signal.lfilter([s],[1,s-1],self.sg)
            smooth = (eps + M)**(-gain)
            return (self.sg*smooth+bias)**power - bias**power
        else:
            print("ERROR: unrecognized transformation", tr)

    def Stockwell(self):
        # Stockwell transform (Brown et al. version)
        # Need to get the starts etc. sorted

        width = len(self.audio_data.data) // 2

        # Gaussian window for frequencies
        f_half = np.arange(0, width + 1) / (2 * width)
        f = np.concatenate((f_half, np.flipud(-f_half[1:-1])))
        p = 2 * np.pi * np.outer(f, 1 / f_half[1:])
        window = np.exp(-p ** 2 / 2).T

        f_tran = fft.interfaces.scipy_fft.fft(self.audio_data.data, 2*width, overwrite_x=True)
        #f_tran = fft.fft(self.data, 2*width, overwrite_x=True)
        diag_con = np.linalg.toeplitz(np.conj(f_tran[:width + 1]), f_tran)
        # Remove zero freq line
        diag_con = diag_con[1:width + 1, :]  
        return np.flipud(fft.interfaces.scipy_fft.ifft(diag_con * window, axis=1))
        #return np.flipud(fft.ifft(diag_con * window, axis=1))

    def wiener_entropy(self,sg):
        return np.sum(np.log(sg),1)/np.shape(sg)[1] - np.log(np.sum(sg,1)/np.shape(sg)[1])

    def mean_frequency(self,sampleRate,timederiv,freqderiv):
        # TODO: samplerate
        freqs = sampleRate//2 / np.shape(timederiv)[1] * (np.arange(np.shape(timederiv)[1])+1)
        mfd = np.sum(timederiv**2 + freqderiv**2,axis=1)
        mfd = np.where(mfd==0,1,mfd)
        mf = np.sum(freqs * (timederiv**2 + freqderiv**2),axis=1)/mfd
        return freqs,mf

    def goodness_of_pitch(self,spectral_deriv,sg):
        return np.max(np.abs(fft.interfaces.scipy_fft.fft(spectral_deriv/sg, axis=0)),axis=0)
        #return np.max(np.abs(fft.fft(spectral_deriv/sg, axis=0)),axis=0)

    def spectral_derivative(self, window_width, incr, K=2, threshold=0.5, returnAll=False):
        """ Compute the spectral derivative """
        if self.audio_data.data is None or len(self.audio_data.data)==0:
            print("ERROR: attempted to calculate spectrogram without audiodata")
            return
        if not specExtra:
            print("Option not available")
            return

        # Compute the set of multi-tapered spectrograms
        starts = range(0, len(self.audio_data.data) - window_width, incr)
        import dpss
        [tapers, eigen] = dpss.dpss(window_width, 2.5, K)
        sg = np.zeros((len(starts), window_width, K), dtype=complex)
        for k in range(K):
            for i in starts:
                sg[i // incr, :, k] = tapers[:, k] * self.audio_data.data[i:i + window_width]
            sg[:, :, k] = fft.interfaces.scipy_fft.fft(sg[:, :, k])
            #sg[:, :, k] = fft.fft(sg[:, :, k])
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
        freqs, mf = self.mean_frequency(self.audio_data.sample_rate, timederiv, freqderiv)
        #freqs, mf = self.mean_frequency(self.sampleRate, timederiv, freqderiv)

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
            y1 = [i * self.audio_data.sample_rate//2/np.shape(self.sg)[1] for i in y]
            #y1 = [i * self.sampleRate//2/np.shape(self.sg)[1] for i in y]
            y1 = np.asarray(y1)
            valminfrq = self.minFreqShow/(self.audio_data.sample_rate//2/np.shape(self.sg)[1])
            #valminfrq = self.minFreqShow/(self.sampleRate//2/np.shape(self.sg)[1])
    
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
        from src.utils import Shapes
        # Estimate fund freq, using windows of 2 spec FFT lengths (4 columns)
        # to make life easier:
        Wsamples = 4*self.incr
        # No set minfreq cutoff here, but warn of the lower limit for
        # reliable estimation (i.e max period such that 3 periods
        # fit in the F0 window):
        minReliableFreq = self.audio_data.sample_rate / (Wsamples/3)
        #minReliableFreq = self.sampleRate / (Wsamples/3)
        print("Warning: F0 estimation below %d Hz will be unreliable" % minReliableFreq)
        # returns pitch in Hz for each window of Wsamples/2
        # over the entire data provided (so full page here)
        thr = 0.5
        pitchshape = Shapes.fundFreqShaper(self.audio_data.data, Wsamples, thr, self.audio_data.sample_rate)
        #pitchshape = Shapes.fundFreqShaper(self.data, Wsamples, thr, self.sampleRate)
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

        yadjfact = 2/self.audio_data.sample_rate*np.shape(self.sg)[1]
        #yadjfact = 2/self.sampleRate*np.shape(self.sg)[1]

        # then create the x sequence (in spec coordinates)
        starts = np.arange(len(pitch)) * pitchshape.tunit + pitchshape.tstart # in seconds
        # (pitchshape.tstart should always be 0 here as it used full data)
        starts = starts * self.audio_data.sample_rate / self.incr  # in spec columns
        #starts = starts * self.sampleRate / self.incr  # in spec columns

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
                    y.append(ys[t][f]/self.audio_data.sample_rate*2*np.shape(self.sg)[1])
                    #y.append(ys[t][f]/self.sampleRate*2*np.shape(self.sg)[1])

        valminfrq = self.minFreqShow/(self.audio_data.sample_rate//2/np.shape(self.sg)[1])
        #valminfrq = self.minFreqShow/(self.sampleRate//2/np.shape(self.sg)[1])
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
        yfr = [i * self.audio_data.sample_rate//2/np.shape(self.sg)[1] for i in y]
        #yfr = [i * self.sampleRate//2/np.shape(self.sg)[1] for i in y]
        yfr = np.asarray(yfr)

        # remove points beyond frq range to show
        inds = np.where((yfr >= self.minFreqShow) & (yfr <= self.maxFreqShow))
        x = x[inds]
        y = y[inds]

        # adjust y pos for when spec doesn't start at 0
        specstarty = self.minFreqShow / (self.audio_data.sample_rate // 2 / np.shape(self.sg)[1])
        #specstarty = self.minFreqShow / (self.sampleRate // 2 / np.shape(self.sg)[1])
        y = [i - specstarty for i in y]

        return x, y

    def formants(self,ncoeff=None):
        # First look at formants. Snell and Milinazzo '93 method
        from LevinsonDurbanRecursion import LPC

        if ncoeff is None:
            # TODO
            ncoeff = 2 + self.audio_data.sample_rate // 1000
            #ncoeff = 2 + self.sampleRate // 1000

        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.window_width) / (self.window_width - 1)))
        starts = range(0, len(self.audio_data.data) - self.window_width, self.window_width)
        freqs = []
        for start in starts:
            x = self.audio_data.data[start:start + self.window_width]*window
            # High-pass filter
            x = signal.lfilter([1], [1., 0.63], x)

            # LPC
            A, e, k = LPC(x, ncoeff)
            A = np.squeeze(A)

            # Extract roots, turn into angles
            roots = np.roots(A)
            roots = [r for r in roots if np.imag(r) >= 0]
            angles = np.arctan2(np.imag(roots), np.real(roots))

            freqs.append(sorted(angles / 2 / np.pi * self.audio_data.sample_rate))
            #freqs.append(sorted(angles / 2 / np.pi * self.sampleRate))

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
        df=self.audio_data.sample_rate//2 /(np.shape(imspec)[0]+1)  # frequency increment
        #df=self.sampleRate//2 /(np.shape(imspec)[0]+1)  # frequency increment
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
            self.audio_data.data = SignalProc.bandpassFilter(self.audio_data.data,self.audio_data.sample_rate, start=start, end=end)
            #self.data = SignalProc.bandpassFilter(self.data,self.sampleRate, start=start, end=end)
        elif str(alg) == "Butterworth Bandpass":
            self.audio_data.data = SignalProc.ButterworthBandpass(self.audio_data.data, self.audio_data.sample_rate, low=start, high=end)
            #self.data = SignalProc.ButterworthBandpass(self.data, self.sampleRate, low=start, high=end)
        else:
            # Median Filter
            self.audio_data.data = SignalProc.medianFilter(self.audio_data.data,int(str(width)))

    def generateFeaturesNN(self, seglen, real_spec_width, frame_size, frame_hop=None, NNfRange=None):
        '''
        Prepare a syllable to input to the NN model
        Returns the features (spectrogram for each frame)
        seglen: length of this segment (self.data), in s
        frame_size: length of each frame, in s
        real_spec_width: number of spectrogram columns in each frame
            (slightly differs from expected b/c of boundary effects,
             so passing w/ a precalculated adjustment)
        frame_hop: hop between frames, in s, or None to not overlap
            (i.e. hop by 1 frame_size)
        NNfRange: frequency list [f1, f2], if not None, sets
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
        if NNfRange is not None:
            bin_width = self.audio_data.sample_rate / 2 / spec_height
            #bin_width = self.sampleRate / 2 / spec_height
            lb = int(np.ceil(NNfRange[0] / bin_width))
            ub = int(np.floor(NNfRange[1] / bin_width))
            self.sg[:, 0:lb] = 0.0
            self.sg[:, ub:] = 0.0

        # extract each frame:
        featuress = np.empty((n, spec_height, real_spec_width, 1), dtype=np.float32)
        for i in range(n):
            sgstart = int(frame_hop * i * self.audio_data.sample_rate / self.incr)
            #sgstart = int(frame_hop * i * self.sampleRate / self.incr)
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
        # NN frame size, or due to inconsistent rounding
        featuress = featuress[:(i+1), :, :, :]
        return featuress

    def generateFeaturesNN2(self, seglen, real_spec_width, frame_size, frame_hop=None):
        '''
        Prepare a syllable to input to the NN model
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
            sgstart = int(frame_hop * i * self.audio_data.sample_rate / self.incr)
            #sgstart = int(frame_hop * i * self.sampleRate / self.incr)
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
        # NN frame size
        featuress = featuress[:i, :, :, :]
        return featuress

