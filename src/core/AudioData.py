# AudioData.py
#
# Holds array and formats

# Version 3.4 18/12/24
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2024

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

import os
import tempfile
import shutil
import numpy as np
import soundfile as sf
import pyflac
from src.utils import wavio


class AudioData:
    """Container for loaded audio data with metadata.
    
    Pure audio data - no PyQt6 dependencies.
    Access properties directly: .sample_rate, .sample_format, .sample_size, .channels
    """    
    def __init__(self, data, sample_rate, file_length, sample_format, sample_size,
                 min_freq=0, max_freq=None, channels=1):
        self.data = data  # numpy array of audio samples
        self.sample_rate = sample_rate
        self.file_length = file_length  # in samples
        self.sample_format = sample_format  # e.g. 'Int16', 'Int32', 'UInt8'
        self.sample_size = sample_size  # in bits: 8, 16, 32
        self.channels = channels
        self.min_freq = min_freq
        self.max_freq = max_freq if max_freq is not None else sample_rate // 2

    def replace_data(self, new_data, sample_rate=None, min_freq=None, max_freq=None):
        """Replace the audio buffer and update related metadata in one place.

        Args:
            new_data: numpy array with the new samples
            sample_rate: optional new sample rate (int)
            min_freq, max_freq: optional frequency bounds
        """
        self.data = new_data
        self.file_length = len(new_data) if new_data is not None else 0
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if min_freq is not None:
            self.min_freq = min_freq
        if max_freq is not None:
            self.max_freq = max_freq
        # Ensure max_freq has a sensible default
        if self.max_freq is None and hasattr(self, 'sample_rate'):
            self.max_freq = self.sample_rate // 2

    def extract_time_slice(self, start_sec, end_sec):
        """Extract a time slice of the audio data.
        
        Args:
            start_sec: Start time in seconds
            end_sec: End time in seconds
            
        Returns:
            AudioData: New AudioData object with the extracted slice
        """
        start_sample = int(start_sec * self.sample_rate)
        end_sample = int(end_sec * self.sample_rate)
        sliced_data = self.data[start_sample:end_sample]
        
        return AudioData(
            data=sliced_data,
            sample_rate=self.sample_rate,
            file_length=len(sliced_data),
            sample_format=self.sample_format,
            sample_size=self.sample_size,
            channels=self.channels,
            min_freq=self.min_freq,
            max_freq=self.max_freq
        )


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
            return self.load_wav(filepath, duration, offset, silent)
        elif file_format == '.flac':
            return self.load_flac(filepath, duration, offset, silent)
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
        
    def load_wav(self, filepath, duration, offset, silent):
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
        
        # Map soundfile subtype to (format_name, bit_size)
        FORMAT_MAP = {
            "PCM_U8": ("UInt8", 8),
            "PCM_S8": ("Int8", 8),
            "PCM_16": ("Int16", 16),
            "PCM_32": ("Int32", 32),
        }
        
        samp_fmt, samp_size = FORMAT_MAP.get(info.subtype, ("Int16", 16))
        if info.subtype not in FORMAT_MAP and not silent:
            print(f"Warning: Unsupported sample format {info.subtype}, using Int16")

        if not silent:
            print(f"Detected format: 1 channel, {sample_rate} Hz, {samp_fmt} format")

        return AudioData(
            data=data,
            sample_rate=sample_rate,
            file_length=file_length,
            sample_format=samp_fmt,
            sample_size=samp_size,
            channels=1
        )
    
    def load_flac(self, filepath, duration, offset, silent):
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
            return self.load_wav(temp_wav_path, duration, offset, silent)