# AudioLoader.py
# Centralized audio file loading for AviaNZ

import os
import tempfile
import shutil
import numpy as np
import soundfile as sf
import pyflac
from PyQt6.QtGui import QImage
from PyQt6.QtMultimedia import QAudioFormat
from src.utils import wavio


class AudioData:
    """Container for loaded audio data with metadata."""
    
    def __init__(self, data, sample_rate, file_length, audio_format, 
                 min_freq=0, max_freq=None, channels=1, sample_width=16):
        self.data = data  # numpy array of audio samples
        self.sample_rate = sample_rate
        self.file_length = file_length  # in samples
        self.audio_format = audio_format  # QAudioFormat object
        self.min_freq = min_freq
        self.max_freq = max_freq if max_freq is not None else sample_rate // 2
        self.channels = channels
        self.sample_width = sample_width


class AudioLoader:
    """Centralized audio file loading with format detection and validation."""
    
    SUPPORTED_FORMATS = {'.wav', '.flac'}
    
    def __init__(self):
        pass
    
    def load_audio(self, filepath, duration=None, offset=0, silent=False, **kwargs):
        """Load audio from file with automatic format detection.
        
        Args:
            filepath: Path to audio file
            duration: Duration to read in seconds (None for all)
            offset: Offset to start reading in seconds
            silent: Suppress output messages
            
        Returns:
            AudioData: Container with audio data and metadata
        """
        if not self.validate_file(filepath):
            raise ValueError(f"Invalid audio file: {filepath}")
        
        file_format = self.detect_format(filepath)
        
        if file_format == '.wav':
            return self._load_wav(filepath, duration, offset, silent)
        elif file_format == '.flac':
            return self._load_flac(filepath, duration, offset, silent)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def detect_format(self, filepath):
        """Auto-detect audio format from file extension."""
        ext = os.path.splitext(filepath.lower())[1]
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file extension: {ext}")
        return ext
    
    def validate_file(self, filepath):
        """Validate file exists, has content, and is readable."""
        if not os.path.exists(filepath):
            print(f"ERROR: file not found: {filepath}")
            return False
        
        if os.stat(filepath).st_size == 0:
            print(f"ERROR: file is empty: {filepath}")
            return False
        
        if os.stat(filepath).st_size < 1000:
            print(f"WARNING: file appears to have only header: {filepath}")
        
        return True
    
    def get_file_info(self, filepath):
        """Get audio file metadata without loading full content."""
        file_format = self.detect_format(filepath)
        
        if file_format == '.wav':
            return wavio.readFmt(filepath)
        elif file_format == '.flac':
            info = sf.info(filepath)
            return (info.samplerate, info.frames/info.samplerate, info.channels, 16)
        
    def _load_wav(self, filepath, duration, offset, silent):
        """Load WAV file using wavio."""
        wavobj = wavio.read(filepath, duration, offset)
        data = wavobj.data
        
        # Take only left channel
        if len(data.shape) > 1:
            data = data[:, 0]
        
        # Force float type
        if data.dtype != 'float':
            data = data.astype('float')
        
        # Get format info
        info = sf.info(filepath)
        sample_rate = info.samplerate
        file_length = len(data)
        
        # Create QAudioFormat
        audio_format = QAudioFormat()
        audio_format.setChannelCount(1)
        audio_format.setSampleRate(sample_rate)
        
        # Set sample format based on file
        sampwidth = info.subtype
        if sampwidth == "PCM_U8":
            audio_format.setSampleFormat(QAudioFormat.SampleFormat.UInt8)
        elif sampwidth == "PCM_16":
            audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        elif sampwidth == "PCM_S8":
            audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int8)
        elif sampwidth == "PCM_32":
            audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int32)
        else:
            print(f"Warning: Unsupported sample format {sampwidth}, using Int16")
            audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        
        if not silent:
            sf_name = str(audio_format.sampleFormat())
            print(f"Detected format: {audio_format.channelCount()} channels, {audio_format.sampleRate()} Hz, {sf_name.split('.')[-1]} format")
        
        return AudioData(
            data=data,
            sample_rate=sample_rate,
            file_length=file_length,
            audio_format=audio_format,
            channels=1
        )
    
    def _load_flac(self, filepath, duration, offset, silent):
        """Load FLAC file by converting to temporary WAV."""
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav:
            temp_wav_path = temp_wav.name
            temp_dir = os.path.dirname(temp_wav.name)
            estimated_wav_size = os.path.getsize(filepath) * 10
            total, used, free = shutil.disk_usage(temp_dir)
            
            if free < estimated_wav_size:
                raise IOError("Insufficient disk space for WAV conversion")
            
            pyf = pyflac.FileDecoder(filepath, temp_wav_path)
            pyf.process()
            return self._load_wav(temp_wav_path, duration, offset, silent)