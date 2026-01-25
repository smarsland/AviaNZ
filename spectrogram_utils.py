"""
Shared utilities for spectrogram processing.

This module contains common functionality used across data loaders:
- SpectrogramProcessor: Generate spectrograms from audio files
- smart_overwrite_folder: Safely overwrite output folders
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import warnings
from PIL import Image
import config
import spectrogram


def smart_overwrite_folder(folder_path, preserve_noise=True):
    """
    Remove data and examples folders, preserving noise_data if it exists.
    
    Args:
        folder_path: Path to folder to overwrite
        preserve_noise: If True, preserve noise_data subfolder (default: True)
    """
    if not os.path.exists(folder_path):
        return
    
    data_folder = os.path.join(folder_path, "data")
    examples_folder = os.path.join(folder_path, "examples")
    labels_file = os.path.join(folder_path, "labels.json")
    
    if os.path.exists(data_folder):
        print(f"Removing {data_folder}")
        shutil.rmtree(data_folder)
    
    if os.path.exists(examples_folder):
        print(f"Removing {examples_folder}")
        shutil.rmtree(examples_folder)
    
    if os.path.exists(labels_file):
        print(f"Removing {labels_file}")
        os.remove(labels_file)
    
    if preserve_noise:
        noise_folder = os.path.join(folder_path, "noise_data")
        if os.path.exists(noise_folder):
            print(f"Preserving {noise_folder}")


class SpectrogramProcessor:
    def __init__(self, window_seconds, hop_seconds, freq_bins, fs, spec_params):
        """
        Initialize spectrogram processor with time-based parameters.
        
        Args:
            window_seconds: Window width in seconds (e.g., 0.025 for 25ms)
            hop_seconds: Hop length in seconds (e.g., 0.010 for 10ms)
            freq_bins: Target number of frequency bins in output
            fs: Sample rate in Hz
            spec_params: Dictionary of spectrogram parameters
        """
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.freq_bins = freq_bins
        self.fs = fs
        self.spec_params = spec_params
        
        self.window_width = int(window_seconds * fs)
        self.window_inc = int(hop_seconds * fs)
        
        self.sp = spectrogram.Spectrogram(window_width=self.window_width, incr=self.window_inc)

    def process_audio_file(self, sound_file):
        try:
            file_info = sf.info(sound_file)
            duration = file_info.frames / file_info.samplerate
            
            if duration > config.MAX_FILE_DURATION_SECONDS:
                raise ValueError(f"File {sound_file} is {duration:.1f} seconds ({duration/60:.1f} minutes) - longer than {config.MAX_FILE_DURATION_SECONDS/60:.0f} minutes")
            elif duration > config.WARNING_FILE_DURATION_SECONDS:
                warnings.warn(f"File {sound_file} is {duration:.1f} seconds ({duration/60:.1f} minutes) - longer than {config.WARNING_FILE_DURATION_SECONDS/60:.0f} minute")
            
            self.sp.readSoundFile(sound_file, silent=True)
            if self.sp.audio_data.sample_rate != self.fs:
                warnings.warn(f"File {sound_file} has sample rate {self.sp.audio_data.sample_rate} Hz, resampling to {self.fs} Hz")
                self.sp.resample(self.fs)

            _ = self.sp.spectrogram(
                window_width=self.window_width,
                incr=self.window_inc,
                window=self.spec_params['windowType'],
                sgType=self.spec_params['sgType'],
                sgScale=self.spec_params['sgScale'],
                nfilters=self.freq_bins,
                mean_normalise=self.spec_params['mean_normalise'],
                equal_loudness=self.spec_params['equal_loudness']
            )
            sg_raw = np.rot90(self.sp.sg)
            return sg_raw if not np.isnan(sg_raw).any() else None
        except Exception as e:
            print(f"Error processing {sound_file}: {e}")
            return None

    def process_audio_segment(self, sound_file, start_time, end_time):
        """
        Load and process a specific segment of an audio file.
        
        Args:
            sound_file: Path to audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Spectrogram array or None if processing failed
        """
        try:
            duration = end_time - start_time
            offset = start_time
            
            if duration > config.MAX_FILE_DURATION_SECONDS:
                raise ValueError(f"Segment is {duration:.1f} seconds ({duration/60:.1f} minutes) - longer than {config.MAX_FILE_DURATION_SECONDS/60:.0f} minutes")
            elif duration > config.WARNING_FILE_DURATION_SECONDS:
                warnings.warn(f"Segment is {duration:.1f} seconds ({duration/60:.1f} minutes) - longer than {config.WARNING_FILE_DURATION_SECONDS/60:.0f} minute")
            
            self.sp.readSoundFile(sound_file, offset=offset, duration=duration, silent=True)
            
            if self.sp.audio_data.sample_rate != self.fs:
                warnings.warn(f"File {sound_file} has sample rate {self.sp.audio_data.sample_rate} Hz, resampling to {self.fs} Hz")
                self.sp.resample(self.fs)

            _ = self.sp.spectrogram(
                window_width=self.window_width,
                incr=self.window_inc,
                window=self.spec_params['windowType'],
                sgType=self.spec_params['sgType'],
                sgScale=self.spec_params['sgScale'],
                nfilters=self.freq_bins,
                mean_normalise=self.spec_params['mean_normalise'],
                equal_loudness=self.spec_params['equal_loudness']
            )
            
            sg_raw = np.rot90(self.sp.sg)
            return sg_raw if not np.isnan(sg_raw).any() else None
            
        except Exception as e:
            print(f"Error processing segment [{start_time:.2f}-{end_time:.2f}] from {sound_file}: {e}")
            return None

    def save_spectrogram(self, sg_raw, output_folder, filename):
        np.save(os.path.join(output_folder, f"{filename}.npy"), sg_raw)

    def save_example_image(self, sg_raw, output_folder, filename, cmap_name='gray'):
        examples_folder = os.path.join(output_folder, "examples")
        os.makedirs(examples_folder, exist_ok=True)
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=np.nanmin(sg_raw), vmax=np.nanmax(sg_raw))
        colored = cmap(norm(sg_raw))
        img = Image.fromarray((colored[..., :3] * 255).astype(np.uint8))
        img.save(os.path.join(examples_folder, f"{filename}.png"))
