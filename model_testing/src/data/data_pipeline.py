"""
Unified data pipeline for bird sound classification.

This module provides common functionality for:
- Loading AviaNZ annotations (.data files)
- Processing audio segments
- Generating spectrograms
- Creating ground truth labels

All other scripts should use this module instead of duplicating code.
"""

import os
import json
import numpy as np
import warnings
import soundfile as sf
import subprocess
import tempfile


class Segment:
    """A single AviaNZ annotation."""
    
    def __init__(self, start_time, end_time, freq_low, freq_high, labels):
        if start_time < 0 or end_time < 0:
            raise ValueError("Segment times must be positive or 0")
        if freq_low < 0 or freq_high < 0:
            raise ValueError("Segment frequencies must be positive or 0")
        if not isinstance(labels, list):
            raise ValueError("Segment labels must be a list")

        for lab in labels:
            if not isinstance(lab, dict):
                raise ValueError("Segment label must be a dict")
            if "species" not in lab or not isinstance(lab["species"], str):
                raise ValueError("species bad or missing from label")
            if "certainty" not in lab or not isinstance(lab["certainty"], (int, float)):
                raise ValueError("certainty bad or missing from label")

        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.freq_low = int(freq_low)
        self.freq_high = int(freq_high)
        self.labels = labels
        self.keys = [lab['species'] for lab in self.labels]
        
        if len(self.keys) > len(set(self.keys)):
            raise ValueError("non-unique species detected")
    
    @staticmethod
    def migrate_label_format(label):
        """Convert an old-format label (a bare species string) to the new dict format.

        Mirrors AviaNZ core's migrateLabelFormat (src/core/annotation.py):
          - dict            -> returned unchanged
          - "Don't Know"    -> certainty 0
          - trailing '?'    -> certainty 50 (uncertain), '?' stripped
          - any other string-> certainty 100
        """
        if isinstance(label, dict):
            return label
        if label == "Don't Know":
            return {"species": "Don't Know", "certainty": 0}
        if isinstance(label, str) and label.endswith('?'):
            return {"species": label[:-1], "certainty": 50}
        return {"species": label, "certainty": 100}

    @classmethod
    def from_list(cls, data):
        """Create a Segment from legacy list format [start_time, end_time, freq_low, freq_high, labels].

        Tolerates old-format annotation labels where labels are bare strings
        (e.g. ["Kakapo(B)3"]) or a single string, migrating them to the new
        dict format before validation.
        """
        if not isinstance(data, (list, tuple)):
            raise ValueError("from_list expects a list or tuple")
        if len(data) != 5:
            raise ValueError(f"from_list requires 5 elements, got {len(data)}")

        labels = data[4]
        # Old format: a single bare species string instead of a list.
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(labels, list):
            labels = [cls.migrate_label_format(lab) for lab in labels]
            # Empty label list -> unknown, matching AviaNZ core behaviour.
            if len(labels) == 0:
                labels = [{"species": "Don't Know", "certainty": 0}]

        # Old annotations sometimes carry negative freq bounds (e.g. a low bound
        # of -1 meaning "full band"). Frequency is metadata only — extraction
        # uses the time bounds — so clamp negatives to 0 rather than discard an
        # otherwise-valid segment and its labels.
        freq_low = max(0, data[2]) if data[2] is not None else 0
        freq_high = max(0, data[3]) if data[3] is not None else 0

        return cls(start_time=data[0], end_time=data[1], freq_low=freq_low,
                   freq_high=freq_high, labels=labels)
    
    def __repr__(self):
        return f"Segment({self.start_time}, {self.end_time}, {self.freq_low}, {self.freq_high}, {len(self.labels)} labels)"
    
    def get_species(self, min_certainty=0):
        """Get list of species names meeting certainty threshold."""
        return [lab['species'] for lab in self.labels if lab['certainty'] >= min_certainty]


def load_avianz_annotations(data_file):
    """
    Load AviaNZ .data annotation file.
    
    Args:
        data_file: Path to .data file
        
    Returns:
        List of Segment objects
    """
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) < 1:
            return []
        
        segments = []
        for seg_data in data[1:]:
            try:
                segment = Segment.from_list(seg_data)
                segments.append(segment)
            except Exception as e:
                warnings.warn(f"Could not parse segment in {data_file}: {e}")
                continue
        
        return segments
        
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        warnings.warn(f"Could not parse JSON in {data_file}: {e}")
        return []
    except Exception as e:
        warnings.warn(f"Error loading {data_file}: {e}")
        return []


def find_audio_files(folder, extensions=('.wav', '.flac')):
    """
    Recursively find all audio files in a folder.
    
    Args:
        folder: Root folder to search
        extensions: Tuple of file extensions to search for
        
    Returns:
        List of paths to audio files
    """
    audio_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(extensions) and not file.endswith('.backup'):
                audio_files.append(os.path.join(root, file))
    return audio_files


def get_audio_duration(audio_path):
    """
    Get duration of audio file in seconds.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Duration in seconds, or None on error
    """
    try:
        info = sf.info(audio_path)
        return info.frames / info.samplerate
    except Exception as e:
        warnings.warn(f"Error reading {audio_path}: {e}")
        return None


def filter_segment_labels(segment, min_certainty=50, skip_species=None):
    """
    Filter segment labels by certainty and skip list.
    
    Args:
        segment: Segment object
        min_certainty: Minimum certainty threshold
        skip_species: List of species names to skip
        
    Returns:
        List of valid species names
    """
    skip_species = skip_species or []
    if "Don't Know" not in skip_species:
        skip_species = skip_species + ["Don't Know"]
    
    valid_labels = []
    for lab in segment.labels:
        species = lab['species']
        certainty = lab['certainty']
        
        if certainty < min_certainty:
            continue
        if species in skip_species:
            continue
        
        valid_labels.append(species)
    
    return valid_labels


def create_time_windows(duration, window_duration=5.0):
    """
    Create time windows for a given duration.
    
    Args:
        duration: Total duration in seconds
        window_duration: Window size in seconds
        
    Returns:
        List of (start_time, end_time) tuples
    """
    num_windows = int(duration / window_duration)
    windows = []
    
    for i in range(num_windows):
        start_time = i * window_duration
        end_time = (i + 1) * window_duration
        windows.append((start_time, end_time))
    
    return windows


def get_species_in_window(segments, window_start, window_end, min_certainty=50, skip_species=None):
    """
    Get all species present in a time window.
    
    Args:
        segments: List of Segment objects
        window_start: Window start time in seconds
        window_end: Window end time in seconds
        min_certainty: Minimum certainty threshold
        skip_species: List of species to skip
        
    Returns:
        Set of species names present in the window
    """
    species_in_window = set()
    
    for seg in segments:
        if seg.end_time > window_start and seg.start_time < window_end:
            valid_labels = filter_segment_labels(seg, min_certainty, skip_species)
            species_in_window.update(valid_labels)
    
    return species_in_window


def convert_flac_to_wav(flac_file):
    """
    Convert FLAC file to temporary WAV file using ffmpeg.
    
    Args:
        flac_file: Path to FLAC file
        
    Returns:
        Path to temporary WAV file, or None on failure
    """
    try:
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav_path = temp_wav.name
        temp_wav.close()
        
        cmd = ['ffmpeg', '-i', flac_file, '-y', temp_wav_path, '-loglevel', 'error']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error converting FLAC: {result.stderr}")
            os.unlink(temp_wav_path)
            return None
        
        return temp_wav_path
        
    except Exception as e:
        print(f"Error converting FLAC file {flac_file}: {e}")
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
        return None
