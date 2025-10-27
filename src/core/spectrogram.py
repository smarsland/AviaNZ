
# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2025

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

# Contains the spectrogram object, optional reference to some AudioData, and some basic methods.

import numpy as np
import scipy.signal as signal
import pyfftw as fft
from scipy.stats import boxcox
import resampy
from PIL import Image

from src.core import signal_proc
from src.core import audio_loader
from src.core import audio_data

BAT_SPECTROGRAM_TIME_PER_PIXEL = 0.002909090909090909

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
        
        # For audio files, get total file duration first (before loading partial)
        loader = audio_loader.AudioLoader()
        file_info = loader.get_file_info(filepath)
        total_duration = file_info[1]  # (rate, nseconds, nchannels, sampwidth)
        
        # Now load the requested portion
        loaded_data = loader.load_audio(filepath, duration, offset, silent)
        
        # Store reference to AudioData - it has all the format info built in
        self.audio_data = loaded_data
        # fileLength is TOTAL file duration, not just the loaded portion
        self.fileLength = total_duration
        self.minFreq = 0
        self.maxFreq = loaded_data.sample_rate // 2
        
        # Update frequency bounds for display
        self.minFreqShow = max(self.minFreq, self.minFreqShow)
        self.maxFreqShow = min(self.maxFreq, self.maxFreqShow)
        
    def load_bmp(self, filepath, duration=None, offset=0, silent=False, **kwargs):
        """Load BMP file (DOC bat recording format) directly as spectrogram data.
        
        Private method - external code should use readSoundFile() which auto-detects format.
        """
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
        self.audio_data = audio_data.AudioData(data=None, sample_rate=176000,
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
        self.fileLength = file_length / 176000.0  # Convert samples to seconds
        self.minFreq = 0
        self.maxFreq = 88000  # 176000 / 2
        
        # Update frequency bounds for display
        self.minFreqShow = max(self.minFreq, self.minFreqShow)
        self.maxFreqShow = min(self.maxFreq, self.maxFreqShow)
        
        return 0  # BMP success indicator

    def get_duration(self):
        """Calculate duration of loaded audio or BMP spectrogram.
        
        Returns:
            float: Duration in seconds
        
        For audio files, delegates to AudioData.get_duration().
        For BMP files, calculates from spectrogram dimensions and time per pixel.
        """
        if self.audio_data is None:
            return 0.0
            
        if self.audio_data.data is not None:
            # Audio file - delegate to AudioData
            return self.audio_data.get_duration()
        else:
            # BMP file - calculate from spectrogram dimensions
            if hasattr(self, 'sg') and self.sg is not None:
                return self.sg.shape[1] * BAT_SPECTROGRAM_TIME_PER_PIXEL
            return 0.0

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
        self.fileLength = len(resampled_data) / float(target)  # Convert samples to seconds

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
            if sampleRate is None:
                raise ValueError("sampleRate must be provided when creating new AudioData")
            self.audio_data = audio_data.AudioData(
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
        """Multi-tapered spectrogram.
        
        Returns a one-sided spectrum (positive frequencies only) with shape 
        (num_frames, window_width) to match the standard spectrogram output.
        """
        try:
            from src.utils import dpss
        except ImportError:
            print("dpss module not found")
            return np.array([])
            
        starts = range(0, len(data) - window_width + 1, incr)
        [tapers, eigen] = dpss.dpss(window_width, 2.5, 3)
        # Use 2*window_width for NFFT to match standard spectrogram behavior
        nfft = 2 * window_width
        out = np.zeros(shape=(len(starts), window_width, 3))
        
        for i, start in enumerate(starts):
            # Compute with NFFT=2*window_width to get proper frequency resolution
            Sk, weights, eigen = dpss.pmtm(data[start:start + window_width], v=tapers, e=eigen, show=False, NFFT=nfft)
            for taper in range(3):
                # Take only positive frequencies (one-sided spectrum)
                out[i, :, taper] = (abs(Sk[taper]) ** 2)[:window_width]
            
        return np.squeeze(np.sum(out, axis=2)) if singleIm else out

    def compute_reassigned_spectrogram(self, data, window, window_width, incr):
        """Reassigned spectrogram using Auger-Flandrin method.
        
        Computes time-frequency reassignment by calculating STFTs with:
        - Standard window h
        - Time-weighted window (t*h) for time reassignment  
        - Window derivative (dh/dt) for frequency reassignment
        
        Returns a one-sided spectrum (positive frequencies only) with shape 
        (num_frames, window_width) to match the standard spectrogram output.
        """
        starts = range(0, len(data) - window_width + 1, incr)
        nfft = 2 * window_width
        num_frames = len(starts)
        
        # Compute time-weighted window and window derivative
        t = np.arange(window_width) - window_width // 2
        time_window = t * window
        
        # Window derivative (central differences)
        window_deriv = np.zeros(window_width)
        window_deriv[1:-1] = (window[2:] - window[:-2]) / 2.0
        window_deriv[0] = window[1] - window[0]
        window_deriv[-1] = window[-1] - window[-2]
        
        # Compute three STFTs
        stft_h = np.zeros((num_frames, nfft), dtype='complex')
        stft_th = np.zeros((num_frames, nfft), dtype='complex')
        stft_dh = np.zeros((num_frames, nfft), dtype='complex')
        
        padded = np.zeros(nfft)
        
        for idx, i in enumerate(starts):
            segment = data[i:i + window_width]
            
            # Standard STFT with window h
            padded.fill(0.0)
            center_start = window_width // 2
            padded[center_start:center_start + window_width] = window * segment
            padded = fft.interfaces.scipy_fft.fftshift(padded)
            padded = np.roll(padded, -1)
            stft_h[idx, :] = fft.interfaces.scipy_fft.fft(padded)
            
            # STFT with time-weighted window
            padded.fill(0.0)
            padded[center_start:center_start + window_width] = time_window * segment
            padded = fft.interfaces.scipy_fft.fftshift(padded)
            padded = np.roll(padded, -1)
            stft_th[idx, :] = fft.interfaces.scipy_fft.fft(padded)
            
            # STFT with derivative of window
            padded.fill(0.0)
            padded[center_start:center_start + window_width] = window_deriv * segment
            padded = fft.interfaces.scipy_fft.fftshift(padded)
            padded = np.roll(padded, -1)
            stft_dh[idx, :] = fft.interfaces.scipy_fft.fft(padded)
        
        # Take only positive frequencies
        stft_h = stft_h[:, :window_width]
        stft_th = stft_th[:, :window_width]
        stft_dh = stft_dh[:, :window_width]
        
        # Compute reassignment operators
        # Avoid division by zero
        eps = 1e-10
        magnitude = np.abs(stft_h)
        threshold = eps * np.max(magnitude)
        valid = magnitude > threshold
        
        # Time reassignment: real part of (t*h STFT) / (h STFT)
        time_reassign = np.zeros((num_frames, window_width))
        time_reassign[valid] = np.real(stft_th[valid] / stft_h[valid])
        
        # Frequency reassignment: imaginary part of (dh STFT) / (h STFT) / (2*pi)
        freq_reassign = np.zeros((num_frames, window_width))
        freq_reassign[valid] = np.imag(stft_dh[valid] / stft_h[valid]) / (2.0 * np.pi)
        
        # Convert reassignments to bin indices
        # Time: current frame index + time offset (in samples) / incr
        time_bins = np.tile(np.arange(num_frames), (window_width, 1)).T + time_reassign / incr
        
        # Frequency: current frequency bin + frequency offset (normalized) * window_width
        freq_bins = np.tile(np.arange(window_width), (num_frames, 1)) + freq_reassign * window_width
        
        # Clamp to valid ranges
        time_bins = np.clip(time_bins, 0, num_frames - 1)
        freq_bins = np.clip(freq_bins, 0, window_width - 1)
        
        # Create reassigned spectrogram using histogram
        sg, _, _ = np.histogram2d(
            time_bins.flatten(), 
            freq_bins.flatten(),
            weights=magnitude.flatten(),
            bins=[num_frames, window_width],
            range=[[0, num_frames], [0, window_width]]
        )
        
        return sg

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

        # Compute the set of multi-tapered spectrograms
        starts = range(0, len(self.audio_data.data) - window_width, incr)
        from src.utils import dpss
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
        from src.utils import levinson_durban_recursion

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
            A, e, k = levinson_durban_recursion.LPC(x, ncoeff)
            A = np.squeeze(A)

            # Extract roots, turn into angles
            roots = np.roots(A)
            roots = [r for r in roots if np.imag(r) >= 0]
            angles = np.arctan2(np.imag(roots), np.real(roots))

            freqs.append(sorted(angles / 2 / np.pi * self.audio_data.sample_rate))
            #freqs.append(sorted(angles / 2 / np.pi * self.sampleRate))

        return freqs

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
            self.audio_data.data = signal_proc.bandpass_filter(self.audio_data.data,self.audio_data.sample_rate, start=start, end=end)
            #self.data = SignalProc.bandpass_filter(self.data,self.sampleRate, start=start, end=end)
        elif str(alg) == "Butterworth Bandpass":
            self.audio_data.data = signal_proc.butterworth_bandpass(self.audio_data.data, self.audio_data.sample_rate, low=start, high=end)
            #self.data = SignalProc.butterworth_bandpass(self.data, self.sampleRate, low=start, high=end)
        else:
            # Median Filter
            self.audio_data.data = signal_proc.median_filter(self.audio_data.data,int(str(width)))

    def extractSpectrogramFrame(self, sgRaw, frame_idx, hop_seconds, spec_frame_width, 
                                   sample_rate, adjust_last=False):
        """Extract and normalize a single frame from spectrogram.
        
        Args:
            sgRaw: Full spectrogram array (time x frequency)
            frame_idx: Index of the frame to extract
            hop_seconds: Hop size in seconds between frames
            spec_frame_width: Width of each frame in spectrogram bins
            sample_rate: Audio sample rate
            adjust_last: If True, adjust the last frame to fit; if False, return None for incomplete frames
            
        Returns:
            Tuple of (normalized_rotated_frame, success) where success is True if frame was extracted
        """
        sgstart = int(hop_seconds * frame_idx * sample_rate / self.incr)
        sgend = sgstart + spec_frame_width
        
        if sgend > np.shape(sgRaw)[0]:
            if adjust_last and sgstart < np.shape(sgRaw)[0]:
                # Adjust to include the last frame
                sgend = np.shape(sgRaw)[0]
                sgstart = max(0, np.shape(sgRaw)[0] - spec_frame_width)
            else:
                return None, False
        
        sgRaw_frame = sgRaw[sgstart:sgend, :]
        
        # Normalize
        maxg = np.max(sgRaw_frame)
        if maxg > 0:
            sgRaw_frame = sgRaw_frame / maxg
        
        # Rotate for display convention (frequency on y-axis, time on x-axis)
        return np.rot90(sgRaw_frame), True

    def generateFeaturesNN(self, seglen, real_spec_width, frame_size, frame_hop=None, NNfRange=None):
        '''
        Prepare a syllable to input to the NN model for inference.
        Returns the features (spectrogram for each frame)
        
        Args:
            seglen: length of this segment (self.data), in s
            frame_size: length of each frame, in s
            real_spec_width: number of spectrogram columns in each frame
                (slightly differs from expected b/c of boundary effects,
                 so passing w/ a precalculated adjustment)
            frame_hop: hop between frames, in s, or None to not overlap
                (i.e. hop by 1 frame_size)
            NNfRange: frequency list [f1, f2], if not None, sets
                spectrogram pixels outside f1:f2 to 0
                
        Returns:
            4D numpy array of shape (n_frames, height, width, 1) ready for model input
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

        # Extract each frame using shared logic
        featuress = np.empty((n, spec_height, real_spec_width, 1), dtype=np.float32)
        frames_filled = 0  # Track how many frames were successfully filled
        
        for i in range(n):
            frame, success = self.extractSpectrogramFrame(
                self.sg, i, frame_hop, real_spec_width, 
                self.audio_data.sample_rate, adjust_last=False
            )
            
            if not success:
                print("Warning: dropping incomplete frame at index", i, "of", n)
                break
            
            # Add channel dimension for CNN input
            featuress[i, :, :, :] = frame[:, :, np.newaxis]
            frames_filled = i + 1

        # Return only the successfully filled frames
        # (may be needed for dealing w/ boundary issues when the spec window 
        # is larger than the NN frame size, or due to inconsistent rounding)
        return featuress[:frames_filled, :, :, :]
