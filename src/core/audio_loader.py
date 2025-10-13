
# Version 4.1 09/10/25
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

# Handles loading of wav, flac files into AudioData objects.

import os
import tempfile
import shutil
import soundfile as sf
import pyflac
from src.utils import wavio
from src.core import audio_data

class AudioLoader:
    """Centralized audio file loading with format detection and validation."""
    
    def load_audio(self, filepath, duration=None, offset=0, silent=False):
        """Load audio from file with automatic format detection"""
        # Validate file existence and size
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        file_size = os.stat(filepath).st_size
        if file_size == 0:
            raise ValueError(f"File is empty: {filepath}")
        
        if file_size < 1000 and not silent:
            print(f"WARNING: file appears to have only header: {filepath}")
        
        # Determine format and load
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.wav':
            return self.load_wav(filepath, duration, offset, silent)
        elif ext == '.flac':
            return self.load_flac(filepath, duration, offset, silent)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    def get_file_info(self, filepath):
        """Get audio file metadata without loading full content."""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.wav':
            return wavio.readFmt(filepath)
        elif ext == '.flac':
            info = sf.info(filepath)
            return (info.samplerate, info.frames/info.samplerate, info.channels, 16)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
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

        return audio_data.AudioData(
            data=data,
            sample_rate=sample_rate,
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