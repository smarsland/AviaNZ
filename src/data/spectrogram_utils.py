"""
Shared utilities for spectrogram processing.

This module contains common functionality used across data loaders:
- SpectrogramProcessor: Generate spectrograms from audio files
- CQTProcessor: Constant-Q Transform (log-spaced multiresolution) spectrograms
- smart_overwrite_folder: Safely overwrite output folders
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import warnings
from PIL import Image
from src.core import config
from . import spectrogram
import torch
import torchaudio
import torchaudio.compliance.kaldi


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

            audio = self.sp.audio_data.data
            rms = np.sqrt(np.mean(audio**2))
            if rms > 1e-8:
                self.sp.audio_data.data = audio / rms * 0.1

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

            audio = self.sp.audio_data.data
            rms = np.sqrt(np.mean(audio**2))
            if rms > 1e-8:
                self.sp.audio_data.data = audio / rms * 0.1

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
        np.save(os.path.join(output_folder, f"{filename}.npy"), np.asarray(sg_raw, dtype=np.float32))

    def save_example_image(self, sg_raw, output_folder, filename, cmap_name='gray'):
        examples_folder = os.path.join(output_folder, "examples")
        os.makedirs(examples_folder, exist_ok=True)
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=np.nanmin(sg_raw), vmax=np.nanmax(sg_raw))
        colored = cmap(norm(sg_raw))
        img = Image.fromarray((colored[..., :3] * 255).astype(np.uint8))
        img.save(os.path.join(examples_folder, f"{filename}.png"))


class AudioSetFbankProcessor:
    def __init__(self, target_sample_rate, frame_length_ms, frame_shift_ms, num_mel_bins):
        self.target_sample_rate = int(target_sample_rate)
        self.frame_length_ms = float(frame_length_ms)
        self.frame_shift_ms = float(frame_shift_ms)
        self.num_mel_bins = int(num_mel_bins)

    def process_audio_file(self, sound_file):
        return self._process(sound_file, start_time=None, end_time=None)

    def process_audio_segment(self, sound_file, start_time, end_time):
        return self._process(sound_file, start_time=float(start_time), end_time=float(end_time))

    def _process(self, sound_file, start_time, end_time):
        file_info = sf.info(sound_file)
        duration = file_info.frames / file_info.samplerate

        if duration > config.MAX_FILE_DURATION_SECONDS:
            raise ValueError(
                f"File {sound_file} is {duration:.1f} seconds ({duration/60:.1f} minutes) - longer than {config.MAX_FILE_DURATION_SECONDS/60:.0f} minutes"
            )
        elif duration > config.WARNING_FILE_DURATION_SECONDS:
            warnings.warn(
                f"File {sound_file} is {duration:.1f} seconds ({duration/60:.1f} minutes) - longer than {config.WARNING_FILE_DURATION_SECONDS/60:.0f} minute"
            )

        if start_time is None and end_time is None:
            audio, sr = sf.read(sound_file, dtype='float32', always_2d=True)
        else:
            sr = int(file_info.samplerate)
            start_sample = int(max(0.0, start_time) * sr)
            end_time = float(end_time)
            end_sample = int(min(end_time, duration) * sr)
            frames = max(0, end_sample - start_sample)
            audio, sr = sf.read(sound_file, start=start_sample, frames=frames, dtype='float32', always_2d=True)

        if audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        else:
            audio = audio[:, 0]

        rms = np.sqrt(np.mean(audio**2))
        if rms > 1e-8:
            audio = audio / rms * 0.1

        waveform = torch.from_numpy(audio).float().unsqueeze(0)

        if int(sr) != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, int(sr), self.target_sample_rate)

        feats = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=self.target_sample_rate,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length_ms,
            frame_shift=self.frame_shift_ms,
            dither=0.0,
            use_log_fbank=False,
        )

        feats = feats.transpose(0, 1).contiguous().cpu().numpy().astype(np.float32)

        if np.isnan(feats).any() or np.isinf(feats).any():
            return None
        return feats

    def save_spectrogram(self, sg_raw, output_folder, filename):
        np.save(os.path.join(output_folder, f"{filename}.npy"), np.asarray(sg_raw, dtype=np.float32))

    def save_example_image(self, sg_raw, output_folder, filename, cmap_name='gray'):
        examples_folder = os.path.join(output_folder, "examples")
        os.makedirs(examples_folder, exist_ok=True)
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=np.nanmin(sg_raw), vmax=np.nanmax(sg_raw))
        colored = cmap(norm(sg_raw))
        img = Image.fromarray((colored[..., :3] * 255).astype(np.uint8))
        img.save(os.path.join(examples_folder, f"{filename}.png"))


def apply_freq_mask(sg, freq_low, freq_high, fs):
    """
    Zero out spectrogram rows outside the annotated frequency range.

    sg is shaped (freq_bins, time_bins) where row 0 is the highest frequency
    and row freq_bins-1 is the lowest, matching the rot90 convention used by
    SpectrogramProcessor.  The mel-scale bin-to-Hz mapping is inverted to find
    which rows correspond to freq_low and freq_high.

    Masking is skipped when freq_high == 0 (AviaNZ convention for
    "full-bandwidth" annotations).

    Args:
        sg:        Spectrogram array of shape (freq_bins, time_bins).
        freq_low:  Lower frequency limit in Hz.
        freq_high: Upper frequency limit in Hz.
        fs:        Sample rate in Hz.

    Returns:
        A copy of sg with values outside [freq_low, freq_high] set to 0.
    """
    if freq_high == 0:
        return sg

    freq_bins = sg.shape[0]
    nyquist = fs / 2.0

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    mel_max = hz_to_mel(nyquist)
    mel_high = hz_to_mel(min(freq_high, nyquist))
    mel_low = hz_to_mel(max(freq_low, 0.0))

    # After rot90, row r corresponds to mel = (freq_bins-1-r) * mel_max / (freq_bins-1).
    # Solving for the row index that matches a given mel value:
    #   row = freq_bins - 1 - mel * (freq_bins - 1) / mel_max
    r_top = int(np.floor(freq_bins - 1 - mel_high * (freq_bins - 1) / mel_max))
    r_top = max(0, r_top)

    r_bottom = int(np.ceil(freq_bins - 1 - mel_low * (freq_bins - 1) / mel_max))
    r_bottom = min(freq_bins - 1, r_bottom)

    masked = sg.copy()
    if r_top > 0:
        masked[:r_top, :] = 0
    if r_bottom < freq_bins - 1:
        masked[r_bottom + 1:, :] = 0
    return masked


class CQTProcessor:
    """Constant-Q Transform spectrogram processor.

    Equivalent to a log-spaced bandpass filterbank (like bandpass_explore.py) but
    computed efficiently via FFT (orders of magnitude faster than time-domain FIR).

    The CQT gives better time resolution at high frequencies (wide bands) and better
    frequency resolution at low frequencies (narrow bands), which suits NZ bird
    vocalisations well.

    Output shape: (freq_bins, time_bins) with row 0 = highest frequency, matching
    the rot90 convention used by SpectrogramProcessor.

    Parameters
    ----------
    n_bins : int
        Number of frequency bins in the output (default: config.DEFAULT_FREQ_BINS = 224).
        If fmin × 2^(n_bins/bins_per_octave) would exceed Nyquist, n_bins is clipped
        automatically and the output is zero-padded back to this shape.
    hop_length : int
        Hop size in samples (default: 10 ms × fs).
    fs : int
        Target sample rate in Hz (audio is resampled to this before CQT).
    fmin : float
        Lowest frequency in Hz (default 32.7 Hz, C1, safely under all NZ bird species).
    bins_per_octave : int
        Frequency resolution per octave (default 24 = 2 bins per semitone).
    """

    def __init__(self, n_bins, hop_length, fs, fmin=32.7, bins_per_octave=24):
        self.hop_length = hop_length
        self.fs = fs
        self.fmin = fmin
        self.bins_per_octave = bins_per_octave
        self._output_bins = n_bins  # desired output rows (may include zero-pad)
        # Compute the maximum safe bin count for this fs (leave 5% Nyquist headroom)
        nyquist_safe = fs / 2 * 0.95
        max_safe = int(np.floor(np.log2(nyquist_safe / fmin) * bins_per_octave))
        self.n_bins = min(n_bins, max_safe)  # bins actually computed by librosa
        self._pad_rows = n_bins - self.n_bins  # zero-rows prepended (above Nyquist)

    def _load_audio(self, sound_file, start_time=None, end_time=None):
        """Load (optionally sliced) audio, resample to self.fs, and RMS-normalise."""
        file_info = sf.info(sound_file)
        sr = int(file_info.samplerate)
        duration = file_info.frames / sr

        if start_time is None:
            audio, _ = sf.read(sound_file, dtype='float32', always_2d=True)
        else:
            start_time = float(start_time)
            end_time = float(min(end_time, duration))
            start_sample = int(max(0.0, start_time) * sr)
            end_sample = int(end_time * sr)
            frames = max(0, end_sample - start_sample)
            audio, _ = sf.read(sound_file, start=start_sample, frames=frames,
                                dtype='float32', always_2d=True)

        if audio.shape[1] > 1:
            audio = audio.max(axis=1)
        else:
            audio = audio[:, 0]

        if sr != self.fs:
            import resampy
            audio = resampy.resample(audio, sr, self.fs)

        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-8:
            audio = audio / rms * 0.1

        return audio

    def _compute_cqt(self, audio):
        """Return CQT magnitude array shaped (_output_bins, time_bins), row 0 = highest freq."""
        import librosa
        C = librosa.cqt(
            audio,
            sr=self.fs,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            fmin=self.fmin,
        )
        mag = np.abs(C).astype(np.float32)
        # librosa CQT: row 0 = fmin (lowest). Flip so row 0 = highest, matching
        # the rot90 convention of SpectrogramProcessor.
        mag = np.flipud(mag)
        # Prepend zero rows for bins that would exceed Nyquist (rows represent
        # "no signal above Nyquist", consistent with what a real filter would give).
        if self._pad_rows > 0:
            pad = np.zeros((self._pad_rows, mag.shape[1]), dtype=np.float32)
            mag = np.vstack([pad, mag])
        return mag

    def process_audio_file(self, sound_file):
        try:
            audio = self._load_audio(sound_file)
            mag = self._compute_cqt(audio)
            return mag if not np.isnan(mag).any() else None
        except Exception as e:
            print(f"Error processing {sound_file}: {e}")
            return None

    def process_audio_segment(self, sound_file, start_time, end_time):
        try:
            audio = self._load_audio(sound_file, start_time, end_time)
            mag = self._compute_cqt(audio)
            return mag if not np.isnan(mag).any() else None
        except Exception as e:
            print(f"Error processing segment [{start_time:.2f}-{end_time:.2f}] from {sound_file}: {e}")
            return None

    def save_spectrogram(self, sg_raw, output_folder, filename):
        np.save(os.path.join(output_folder, f"{filename}.npy"), np.asarray(sg_raw, dtype=np.float32))

    def save_example_image(self, sg_raw, output_folder, filename, cmap_name='gray'):
        examples_folder = os.path.join(output_folder, "examples")
        os.makedirs(examples_folder, exist_ok=True)
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=np.nanmin(sg_raw), vmax=np.nanmax(sg_raw))
        colored = cmap(norm(sg_raw))
        img = Image.fromarray((colored[..., :3] * 255).astype(np.uint8))
        img.save(os.path.join(examples_folder, f"{filename}.png"))
