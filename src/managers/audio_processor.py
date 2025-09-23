# Audio Processing Module for AviaNZ
# Handles coordinate conversions, signal processing, and audio denoising operations

import numpy as np
import time
from PyQt6.QtCore import QObject, pyqtSignal
from ..core import WaveletFunctions
from ..core import SignalProc

class AudioProcessor(QObject):
    """
    Handles audio signal processing operations including:
    - Coordinate conversions between amplitude/spectrogram/frequency domains
    - Audio denoising using wavelets
    - Signal processing utilities
    """
    
    # Signals
    processing_started = pyqtSignal(str)  # Processing operation name
    processing_completed = pyqtSignal()   # Processing finished
    coordinates_converted = pyqtSignal()  # Coordinate conversion completed
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        
        # These will be set by the main window when needed
        self.sp = None  # Spectrogram processor instance
        self.sg = None  # Spectrogram dataThat solves it
        self.batmode = False
        self.datalength = 0
        
    def process_audio_file(self, filename, start_read, batmode, cheatsheet, existing_sp=None):
        """Process audio file and create spectrogram.
        
        Args:
            filename: Path to audio file
            start_read: Starting position in seconds
            batmode: Whether in bat mode
            cheatsheet: Whether in cheatsheet mode
            existing_sp: Existing spectrogram processor to reuse (optional)
            
        Returns:
            bool: True if successful
        """
        from ..core import Spectrogram
        
        try:
            # Use existing spectrogram processor if provided, otherwise create new one
            if existing_sp is not None:
                self.sp = existing_sp
            elif not hasattr(self, 'sp') or self.sp is None:
                if cheatsheet:
                    self.sp = Spectrogram.Spectrogram(512, 256, 0, 0)
                else:
                    minFreqShow = self.config_manager.config['minFreq']
                    maxFreqShow = self.config_manager.config['maxFreq']
                    # Note: spectrogramDialog access would need to be passed from main window
                    self.sp = Spectrogram.Spectrogram(self.config_manager.config['window_width'], self.config_manager.config['incr'], minFreqShow, maxFreqShow)
            
            # Read audio file
            if batmode:
                self.sp.minFreqShow = self.config_manager.config['minFreqBats']
                self.sp.maxFreqShow = self.config_manager.config['maxFreqBats']
                successread = self.sp.readBmp(filename)
                if successread > 0:
                    print("ERROR: file not loaded")
                    return False
                self.datalength = self.sp.fileLength
            else:
                self.sp.minFreqShow = self.config_manager.config['minFreq']
                self.sp.maxFreqShow = self.config_manager.config['maxFreq']
                
                if start_read == 0:
                    lenRead = self.config_manager.config['maxFileShow'] + self.config_manager.config['fileOverlap']
                else:
                    lenRead = self.config_manager.config['maxFileShow'] + 2 * self.config_manager.config['fileOverlap']
                
                self.sp.readSoundFile(filename, lenRead, start_read)
                
                # Resample if needed
                if cheatsheet:
                    self.sp.resample(16000)
                    self.sp.maxFreqShow = 8000
                
                self.datalength = np.shape(self.sp.data)[0]
            
            # Create spectrogram if not in bat mode
            if not batmode:
                _ = self.sp.spectrogram(
                    window_width=self.config_manager.config['window_width'], 
                    incr=self.config_manager.config['incr'],
                    window=self.config_manager.config['windowType'],
                    sgType=self.config_manager.config['sgType'],
                    sgScale=self.config_manager.config['sgScale'],
                    nfilters=self.config_manager.config['nfilters'],
                    mean_normalise=self.config_manager.config['sgMeanNormalise'],
                    equal_loudness=self.config_manager.config['sgEqualLoudness'],
                    onesided=self.config_manager.config['sgOneSided']
                )
            
            self.batmode = batmode
            return True
            
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return False
    
    def get_spectrogram_processor(self):
        """Get the spectrogram processor instance."""
        return self.sp
    
    def get_data_length(self):
        """Get the current data length."""
        return self.datalength
        
    def set_audio_context(self, sp, sg, batmode=False):
        """Set the current audio context for processing operations"""
        self.sp = sp
        self.sg = sg
        self.batmode = batmode
        
    def convertAmpltoSpec(self, x):
        """Convert amplitude domain coordinate to spectrogram domain"""
        if self.batmode:
            incr = 512
        else:
            incr = self.config_manager.config['incr']
        
        if self.sp is None:
            return x  # Return unchanged if no audio context
            
        return x * self.sp.audioFormat.sampleRate() / incr
    
    def convertSpectoAmpl(self, x):
        """Convert spectrogram domain coordinate to amplitude domain"""
        if self.batmode:
            incr = 512
        else:
            incr = self.config_manager.config['incr']
            
        if self.sp is None:
            return x  # Return unchanged if no audio context
            
        return x * incr / self.sp.audioFormat.sampleRate()
    
    def convertMillisecs(self, millisecs):
        """Convert milliseconds to MM:SS format"""
        seconds = (millisecs / 1000) % 60
        minutes = (millisecs / (1000 * 60)) % 60
        return "%02d" % minutes + ":" + "%02d" % seconds
    
    def convertYtoFreq(self, y, sgy=None):
        """Convert Y coordinate to frequency"""
        if self.sp is None or self.sg is None:
            return y  # Return unchanged if no audio context
            
        if sgy is None:
            sgy = np.shape(self.sg)[1]
        return y * self.sp.audioFormat.sampleRate()//2 / sgy + self.sp.minFreqShow
    
    def convertFreqtoY(self, f):
        """Convert frequency to Y coordinate"""
        if self.sp is None or self.sg is None:
            return f  # Return unchanged if no audio context
            
        sgy = np.shape(self.sg)[1]
        return (f - self.sp.minFreqShow) * sgy / (self.sp.audioFormat.sampleRate()//2)
    
    def decompose_wavelet_packet(self, x=None):
        """
        Decompose audio data into wavelet packet representation
        
        Args:
            x: Audio data (optional, uses self.sp.data if not provided)
        """
        if self.sp is None:
            print("No audio context available for wavelet decomposition")
            return None
            
        self.processing_started.emit("Wavelet Packet Decomposition")
        
        print("Decomposing to WP...")
        ot = time.time()
        
        data = x if x is not None else self.sp.data
        
        WFinst = WaveletFunctions.WaveletFunctions(
            data=data, 
            wavelet="dmey2", 
            maxLevel=self.config_manager.config['maxSearchDepth'], 
            samplerate=self.sp.audioFormat.sampleRate()
        )
        
        maxLevel = 5
        allnodes = range(2 ** (maxLevel + 1) - 1)
        WFinst.WaveletPacket(allnodes, mode='symmetric', antialias=False)
        
        print("Done")
        print(time.time() - ot)
        
        self.processing_completed.emit()
        return WFinst
    
    def denoise_segment(self, start_sample, stop_sample):
        """
        Denoise a specific segment of audio data
        
        Args:
            start_sample: Start sample index
            stop_sample: Stop sample index
            
        Returns:
            Denoised audio segment
        """
        if self.sp is None:
            print("No audio context available for denoising")
            return None
            
        self.processing_started.emit("Segment Denoising")
        
        # Since there is no dialog menu, settings are preset constants here:
        noiseest = "ols"  # or qr, or const
        thrType = "soft"
        depth = 6   # can also use 0 to autoset
        wavelet = "dmey2"
        aaRec = False  # True if nicer spectrogram is needed - but it's not very clean either way
        aaWP = False
        thr = 2.0  # this one is difficult to set universally...
        
        opstartingtime = time.time()
        print("Denoising requested at " + time.strftime('%H:%M:%S', time.gmtime(opstartingtime)))
        
        # Extract the piece of audiodata under current segment
        segment_data = self.sp.data[start_sample:stop_sample].copy()
        
        WF = WaveletFunctions.WaveletFunctions(
            data=segment_data, 
            wavelet=wavelet, 
            maxLevel=self.config_manager.config['maxSearchDepth'], 
            samplerate=self.sp.audioFormat.sampleRate()
        )
        
        denoised = WF.waveletDenoise(
            thrType, thr, depth, 
            aaRec=aaRec, aaWP=aaWP, 
            noiseest=noiseest, costfn="fixed"
        )
        
        print("Denoising calculations completed in %.4f seconds" % (time.time() - opstartingtime))
        
        self.processing_completed.emit()
        return denoised
    
    def denoise_full_audio(self, alg, start=None, end=None, width=None, 
                          depth=None, thrType=None, thr=None, wavelet=None,
                          aaRec=None, aaWP=None, noiseest=None):
        """
        Denoise the full audio data using specified parameters
        
        Args:
            alg: Algorithm type ("Wavelets" or other)
            start: Start frequency for processing
            end: End frequency for processing  
            width: Width parameter
            depth: Wavelet decomposition depth
            thrType: Threshold type ("soft" or "hard")
            thr: Threshold value
            wavelet: Wavelet type
            aaRec: Anti-aliasing for reconstruction
            aaWP: Anti-aliasing for wavelet packet
            noiseest: Noise estimation method
            
        Returns:
            Denoised audio data
        """
        if self.sp is None:
            print("No audio context available for denoising")
            return None
            
        self.processing_started.emit("Full Audio Denoising")
        
        opstartingtime = time.time()
        print("Denoising requested at " + time.strftime('%H:%M:%S', time.gmtime(opstartingtime)))
        
        if str(alg) == "Wavelets":
            # Use default values if not provided
            wavelet = wavelet or "dmey2"
            start = start or self.sp.minFreqShow
            end = end or self.sp.maxFreqShow
            
            waveletDenoiser = WaveletFunctions.WaveletFunctions(
                data=self.sp.data, 
                wavelet=wavelet, 
                maxLevel=self.config_manager.config['maxSearchDepth'], 
                samplerate=self.sp.audioFormat.sampleRate()
            )
            
            if depth is not None and thrType is not None:
                # Use provided parameters
                denoised_data = waveletDenoiser.waveletDenoise(
                    thrType, float(str(thr)), depth, 
                    aaRec=aaRec, aaWP=aaWP, 
                    noiseest=noiseest, costfn="fixed"
                )
            else:
                # Use defaults (DOC mode)
                denoised_data = waveletDenoiser.waveletDenoise(
                    "soft", 3, aaRec=True, aaWP=False, 
                    costfn="fixed", noiseest="ols"
                )
        else:
            # Let Spectrogram handle denoising
            self.sp.denoise(alg, start=start, end=end, width=width)
            denoised_data = self.sp.data
        
        print("Denoising calculations completed in %.4f seconds" % (time.time() - opstartingtime))
        
        self.processing_completed.emit()
        return denoised_data
    
    def bandpass_filter_segment(self, data, bottom_freq, top_freq):
        """
        Apply bandpass filter to audio segment
        
        Args:
            data: Audio data to filter
            bottom_freq: Lower frequency bound
            top_freq: Upper frequency bound
            
        Returns:
            Filtered audio data
        """
        if self.sp is None:
            print("No audio context available for filtering")
            return data
            
        bottom = max(0.1, self.sp.minFreq, bottom_freq)
        top = min(top_freq, self.sp.maxFreq - 0.1)
        
        print("Extracting samples between %d-%d Hz" % (bottom, top))
        
        filtered_data = SignalProc.bandpassFilter(
            data, 
            sampleRate=self.sp.audioFormat.sampleRate(), 
            start=bottom, 
            end=top
        )
        
        return filtered_data
