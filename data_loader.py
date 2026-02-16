import os
import sys
import json
import csv
import ast
import argparse
import random
import numpy as np
import subprocess
import tempfile
import shutil
import config
import wavio
from spectrogram_utils import SpectrogramProcessor, AudioSetFbankProcessor, smart_overwrite_folder
from data_pipeline import Segment
import soundfile as sf


class SegmentExtractor:
    def __init__(self, target_sr=None):
        self.target_sr = target_sr

    def convert_flac_to_temp_wav(self, flac_file):
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

    def extract_audio_segment(self, audio_file, start_time, end_time, output_path):
        temp_wav_path = None
        try:
            if audio_file.lower().endswith('.flac'):
                temp_wav_path = self.convert_flac_to_temp_wav(audio_file)
                if temp_wav_path is None:
                    return False
                file_to_read = temp_wav_path
            else:
                file_to_read = audio_file
            
            rate, nseconds, nchannels, sampwidth = wavio.readFmt(file_to_read)
            
            if end_time is None:
                end_time = nseconds
            
            duration = end_time - start_time
            if duration > config.MAX_FILE_DURATION_SECONDS:
                print(f"Info: Segment is {duration:.1f} seconds ({duration/60:.1f} minutes) - longer than {config.MAX_FILE_DURATION_SECONDS/60:.0f} minutes")
            
            if end_time > nseconds:
                end_time = nseconds
            
            segment_duration = end_time - start_time
            if segment_duration <= 0:
                print(f"Invalid segment duration: {segment_duration}")
                return False
            
            wav_data = wavio.read(file_to_read, nseconds=segment_duration, offset=start_time)
            
            if wav_data.data.shape[1] > 1:
                # Use max to preserve full amplitude for spectrograms
                audio_data = np.max(wav_data.data, axis=1).astype(wav_data.data.dtype)
            else:
                audio_data = wav_data.data[:, 0]
            
            if self.target_sr is not None and wav_data.rate != self.target_sr:
                import resampy
                if wav_data.sampwidth == 1:
                    audio_float = (audio_data.astype(np.float32) - 128) / 128.0
                elif wav_data.sampwidth == 2:
                    audio_float = audio_data.astype(np.float32) / 32768.0
                else:
                    audio_float = audio_data.astype(np.float32) / 2147483648.0
                
                audio_float = resampy.resample(audio_float, wav_data.rate, self.target_sr)
                
                audio_data = (audio_float * 32767).astype(np.int16)
                output_rate = self.target_sr
                output_sampwidth = 2
            else:
                output_rate = wav_data.rate
                output_sampwidth = wav_data.sampwidth
            
            wavio.write(output_path, audio_data, output_rate, sampwidth=output_sampwidth)
            
            return True
            
        except Exception as e:
            print(f"Error extracting segment [{start_time:.2f}-{end_time:.2f if end_time else 'end'}] from {audio_file}: {e}")
            return False
        finally:
            if temp_wav_path is not None and os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)


class BaseDataProcessor:
    def __init__(self, spec_processor=None, segment_extractor=None, output_format='spectrogram', with_audio=False):
        self.spec_processor = spec_processor
        self.segment_extractor = segment_extractor
        self.output_format = output_format
        self.with_audio = with_audio
        self.audio_output_dir = None
    
    def setup_audio_dir(self, output_folder):
        """Setup audio output directory if with_audio is enabled"""
        if self.with_audio:
            self.audio_output_dir = os.path.join(output_folder, "audio")
            os.makedirs(self.audio_output_dir, exist_ok=True)
            print(f"Will save audio files to {self.audio_output_dir}")
    
    def save_audio_segment(self, audio_file, start_time, end_time, output_filename):
        """Save audio segment to audio output directory"""
        if not self.audio_output_dir:
            return None
        
        try:
            output_path = os.path.join(self.audio_output_dir, output_filename)
            
            # If end_time is None, copy the whole file
            if end_time is None:
                # Convert if needed, otherwise just copy
                if audio_file.lower().endswith('.wav'):
                    shutil.copy2(audio_file, output_path)
                else:
                    # Convert to WAV using soundfile
                    data, sr = sf.read(audio_file)
                    sf.write(output_path, data, sr)
                return output_filename
            
            # Get file info to calculate sample positions
            info = sf.info(audio_file)
            sr = info.samplerate
            
            # Calculate sample positions
            start_sample = int(start_time * sr)
            num_samples = int((end_time - start_time) * sr)
            
            # Read ONLY the segment we need (efficient!)
            data = sf.read(audio_file, start=start_sample, frames=num_samples)[0]
            
            # Write segment
            sf.write(output_path, data, sr)
            return output_filename
        except Exception as e:
            print(f"Warning: Could not save audio segment: {e}")
            return None

    def save_labels(self, output_folder, files, categories, dataset_name, extra_metadata=None):
        labels_file = os.path.join(output_folder, "labels.json")
        metadata = {
            'files': files,
            'categories': categories,
            'num_classes': len(categories),
            'dataset': dataset_name
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        
        with open(labels_file, 'w') as f:
            json.dump(metadata, f, indent=2)


class AviaNZDataProcessor(BaseDataProcessor):
    def __init__(self, spec_processor=None, segment_extractor=None, output_format='spectrogram', with_audio=False, name_mapping=None):
        super().__init__(spec_processor, segment_extractor, output_format, with_audio)
        self.name_mapping = name_mapping
    
    def normalize_to_ebird(self, species_name):
        """Normalize species name to eBird code if mapping is available"""
        if not self.name_mapping:
            return species_name
        
        if species_name in ['Empty Sample', 'Tree Weta', 'Spy Bird', "Don't Know", None, '']:
            return species_name
        
        # Try direct match
        if species_name in self.name_mapping:
            return self.name_mapping[species_name]
        
        # Try case-insensitive match
        name_lower = species_name.lower()
        for key, value in self.name_mapping.items():
            if key.lower() == name_lower:
                return value
        
        # Try base name without parentheses
        if '(' in species_name:
            base_name = species_name.split('(')[0].strip()
            if base_name in self.name_mapping:
                return self.name_mapping[base_name]
        
        # Return original if no mapping found
        return species_name
    
    def load_annotation_file(self, data_file):
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON in {data_file}: {e}")
            return []
        except Exception as e:
            print(f"Warning: Failed to read {data_file}: {e}")
            return []
        
        if not isinstance(data, list) or len(data) < 1:
            print(f"Warning: Invalid annotation file format: {data_file}")
            return []
        
        segments = []
        for seg_data in data[1:]:
            try:
                segment = Segment.from_list(seg_data)
                segments.append(segment)
            except Exception as e:
                print(f"Warning: Failed to parse segment in {data_file}: {e}")
                continue
        
        return segments

    def find_wav_files(self, folder):
        wav_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith('.wav') and not file.endswith('.backup'):
                    wav_files.append(os.path.join(root, file))
        return wav_files

    def process(self, input_folder, output_folder, overwrite=False, min_certainty=50, skip_species=None, chunk_duration=None, max_segments=None, max_samples=None, specific_species=None, ignore_multilabel=False, max_species=None, min_examples=None):
        print(f"Loading AviaNZ data from {input_folder}")
        
        if chunk_duration:
            print(f"Chunking mode: splitting into {chunk_duration}s chunks with annotation mapping")
        
        if max_segments:
            print(f"Max segments limit (total): {max_segments}")
        
        if max_samples:
            print(f"Max samples limit (per species): {max_samples}")
        
        if specific_species:
            print(f"Filtering for specific species: {specific_species}")
        
        if os.path.exists(output_folder):
            if overwrite:
                smart_overwrite_folder(output_folder, preserve_noise=True)
            else:
                raise FileExistsError(f"Output folder {output_folder} already exists. Use overwrite=True to overwrite.")
        
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output folder: {output_folder}")
        
        skip_species = skip_species or []
        if "Don't Know" not in skip_species:
            skip_species.append("Don't Know")
        
        # Convert specific_species to a set for faster lookup
        # These should be eBird codes if name_mapping is provided
        specific_species_set = set(specific_species) if specific_species else None
        
        if specific_species_set and self.name_mapping:
            print(f"  Filtering for eBird codes: {sorted(specific_species_set)}")
            print(f"  Name mapping loaded: {len(self.name_mapping)} mappings")
        elif specific_species_set and not self.name_mapping:
            print(f"  WARNING: Species filter requested but NO NAME MAPPING LOADED!")
            print(f"  Filtering will NOT work correctly!")
        elif specific_species_set:
            print(f"  Filtering for species: {sorted(specific_species_set)}")
        
        # If no specific_species but max_species or min_examples specified, do first pass to count
        if not specific_species_set and (max_species or min_examples):
            print("\nFirst pass: counting species occurrences...")
            wav_files_temp = self.find_wav_files(input_folder)
            species_segment_counts = {}
            
            for wav_file in wav_files_temp:
                data_file = wav_file + ".data"
                if not os.path.exists(data_file):
                    continue
                segments = self.load_annotation_file(data_file)
                
                for seg in segments:
                    for lab in seg.labels:
                        species = lab['species']
                        certainty = lab['certainty']
                        if certainty < min_certainty or species in skip_species:
                            continue
                        species_normalized = self.normalize_to_ebird(species)
                        species_segment_counts[species_normalized] = species_segment_counts.get(species_normalized, 0) + 1
            
            # Filter by min_examples
            if min_examples:
                species_segment_counts = {sp: ct for sp, ct in species_segment_counts.items() if ct >= min_examples}
                print(f"Species with >= {min_examples} examples: {len(species_segment_counts)}")
            
            # Sort by count and take top max_species
            sorted_species = sorted(species_segment_counts.items(), key=lambda x: x[1], reverse=True)
            if max_species:
                sorted_species = sorted_species[:max_species]
                print(f"Selected top {max_species} species by count")
            
            # Use these as the specific species filter
            specific_species = [sp for sp, _ in sorted_species]
            specific_species_set = set(specific_species)
            print(f"Selected species: {specific_species}")
            for sp, ct in sorted_species:
                print(f"  {sp}: {ct} segments")
        
        wav_files = self.find_wav_files(input_folder)
        print(f"Found {len(wav_files)} .wav files")
        
        if len(wav_files) == 0:
            print("No .wav files found!")
            return 0
        
        data_folder = os.path.join(output_folder, "data")
        os.makedirs(data_folder, exist_ok=True)
        
        # Setup audio directory if needed
        self.setup_audio_dir(output_folder)
        
        labels = []
        file_count = 0
        segment_count = 0
        species_counts = {}
        species_file_counts = {}  # Track files saved per species for max_samples limit
        species_example_saved = set()
        
        if chunk_duration:
            for wav_idx, wav_file in enumerate(wav_files):
                data_file = wav_file + ".data"
                
                if not os.path.exists(data_file):
                    continue
                
                segments = self.load_annotation_file(data_file)
                
                if len(segments) == 0:
                    continue
                
                try:
                    file_info = sf.info(wav_file)
                    duration = file_info.frames / file_info.samplerate
                    num_chunks = int(np.ceil(duration / chunk_duration))
                    
                    relative_path = os.path.relpath(wav_file, input_folder)
                    print(f"Processing {relative_path} ({duration:.1f}s -> {num_chunks} chunks)")
                    
                    for chunk_idx in range(num_chunks):
                        if max_segments and file_count >= max_segments:
                            break
                        
                        start_time = chunk_idx * chunk_duration
                        end_time = min(start_time + chunk_duration, duration)
                        
                        chunk_labels = set()
                        for seg in segments:
                            if seg.start_time < end_time and seg.end_time > start_time:
                                for lab in seg.labels:
                                    species = lab['species']
                                    certainty = lab['certainty']
                                    
                                    # Skip if doesn't meet certainty or is in skip list
                                    if certainty < min_certainty or species in skip_species:
                                        continue
                                    
                                    # Normalize to eBird code if mapping available
                                    species_normalized = self.normalize_to_ebird(species)
                                    
                                    # If specific_species is set, only include those species (compare normalized)
                                    if specific_species_set and species_normalized not in specific_species_set:
                                        continue
                                    
                                    # Add normalized species to chunk labels
                                    chunk_labels.add(species_normalized)
                        
                        sg_raw = self.spec_processor.process_audio_segment(wav_file, start_time, end_time)
                        
                        if sg_raw is not None:
                            filename = f"file_{file_count:08d}"
                            self.spec_processor.save_spectrogram(sg_raw, data_folder, filename)
                            
                            # Save audio chunk if requested
                            audio_filename = None
                            if self.audio_output_dir:
                                audio_filename = self.save_audio_segment(wav_file, start_time, end_time, f"{filename}.wav")
                            
                            row_id_time = int((chunk_idx + 1) * chunk_duration)
                            chunk_labels_list = sorted(list(chunk_labels))
                            
                            label_entry = {
                                'filename': f"{filename}.npy",
                                'source_file': relative_path,
                                'row_id': f"{relative_path}_{row_id_time}",
                                'start_time': start_time,
                                'end_time': end_time,
                                'chunk_index': chunk_idx,
                                'class_names': chunk_labels_list
                            }
                            if audio_filename:
                                label_entry['audio_file'] = audio_filename
                            
                            labels.append(label_entry)
                            
                            for species in chunk_labels_list:
                                species_counts[species] = species_counts.get(species, 0) + 1
                            
                            file_count += 1
                            
                except Exception as e:
                    print(f"Error processing {wav_file}: {e}")
                    continue
                
                if max_segments and file_count >= max_segments:
                    print(f"Reached max_segments limit of {max_segments}")
                    break
        else:
            for wav_idx, wav_file in enumerate(wav_files):
                data_file = wav_file + ".data"
                
                if not os.path.exists(data_file):
                    continue
                
                segments = self.load_annotation_file(data_file)
                
                if len(segments) == 0:
                    continue
                
                for seg in segments:
                    if max_segments and file_count >= max_segments:
                        break
                    
                    segment_count += 1
                    
                    valid_labels = []
                    for lab in seg.labels:
                        species = lab['species']
                        certainty = lab['certainty']
                        
                        if certainty < min_certainty:
                            continue
                        if species in skip_species:
                            continue
                        
                        # Normalize to eBird code if mapping available
                        species_normalized = self.normalize_to_ebird(species)
                        
                        # If specific_species is set, only include those species (compare normalized)
                        if specific_species_set and species_normalized not in specific_species_set:
                            continue
                        
                        valid_labels.append(species)
                    
                    if len(valid_labels) == 0:
                        continue
                    
                    # Normalize all labels to eBird codes if mapping available
                    if self.name_mapping:
                        normalized_labels = [self.normalize_to_ebird(sp) for sp in valid_labels]
                        # Remove duplicates while preserving order
                        seen = set()
                        normalized_labels = [x for x in normalized_labels if not (x in seen or seen.add(x))]
                    else:
                        normalized_labels = valid_labels
                    
                    # Skip multi-label samples if flag is set
                    if ignore_multilabel and len(normalized_labels) > 1:
                        continue
                    
                    primary_species = normalized_labels[0]
                    
                    # Check max_samples AFTER collecting valid labels (check primary species)
                    if max_samples:
                        species_count = species_file_counts.get(primary_species, 0)
                        if species_count >= max_samples:
                            continue
                
                    if self.output_format == 'wav':
                        file_basename = f"file_{file_count:08d}"
                        output_path = os.path.join(data_folder, f"{file_basename}.wav")
                        
                        success = self.segment_extractor.extract_audio_segment(
                            wav_file, 
                            seg.start_time, 
                            seg.end_time,
                            output_path
                        )
                        
                        if success:
                            labels.append({
                                'filename': f"{file_basename}.wav",
                                'primary_class': primary_species,
                                'class_names': normalized_labels,
                                'source_file': wav_file,
                                'start_time': seg.start_time,
                                'end_time': seg.end_time,
                                'freq_low': seg.freq_low,
                                'freq_high': seg.freq_high
                            })
                            
                            file_count += 1
                            
                            for species in normalized_labels:
                                species_counts[species] = species_counts.get(species, 0) + 1
                            
                            # Track count for the primary species (for max_samples limit)
                            if max_samples:
                                species_file_counts[primary_species] = species_file_counts.get(primary_species, 0) + 1
                    else:
                        sg_raw = self.spec_processor.process_audio_segment(
                            wav_file, 
                            seg.start_time, 
                            seg.end_time
                        )
                        
                        if sg_raw is not None:
                            file_basename = f"file_{file_count:08d}"
                            self.spec_processor.save_spectrogram(sg_raw, data_folder, file_basename)
                            
                            # Save audio segment if requested
                            audio_filename = None
                            if self.audio_output_dir:
                                audio_filename = self.save_audio_segment(wav_file, seg.start_time, seg.end_time, f"{file_basename}.wav")
                            
                            label_entry = {
                                'filename': f"{file_basename}.npy",
                                'primary_class': primary_species,
                                'class_names': normalized_labels,
                                'source_file': wav_file,
                                'start_time': seg.start_time,
                                'end_time': seg.end_time,
                                'freq_low': seg.freq_low,
                                'freq_high': seg.freq_high
                            }
                            if audio_filename:
                                label_entry['audio_file'] = audio_filename
                            
                            labels.append(label_entry)
                            
                            file_count += 1
                            
                            for species in normalized_labels:
                                species_counts[species] = species_counts.get(species, 0) + 1
                            
                            # Track count for the primary species (for max_samples limit)
                            if max_samples:
                                species_file_counts[primary_species] = species_file_counts.get(primary_species, 0) + 1
                            
                            if primary_species not in species_example_saved:
                                safe_name = primary_species.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('\\', '_').replace(':', '_')
                                example_name = f"example_{safe_name}"
                                self.spec_processor.save_example_image(sg_raw, output_folder, example_name)
                                species_example_saved.add(primary_species)
                
                if segment_count % 50 == 0 or segment_count in [1, 10]:
                    output_type = "WAV files" if self.output_format == 'wav' else "spectrograms"
                    print(f"Processed {segment_count} segments from {wav_idx+1}/{len(wav_files)} files, "
                          f"saved {file_count} {output_type}")
            
                if (wav_idx + 1) % 10 == 0:
                    print(f"Processed {wav_idx+1}/{len(wav_files)} files...")
                
                if max_segments and file_count >= max_segments:
                    print(f"Reached max_segments limit of {max_segments}")
                    break
        
        all_species = sorted(species_counts.keys())
        
        dataset_name = 'AviaNZ' if not chunk_duration else 'AviaNZ_Chunked'
        self.save_labels(output_folder, labels, all_species, dataset_name, 
                        {'species_counts': species_counts, 'chunk_duration': chunk_duration})
        
        if chunk_duration:
            print(f"\nSaved {file_count} chunked spectrograms")
        else:
            output_type = "WAV files" if self.output_format == 'wav' else "spectrograms"
            print(f"\nSaved {file_count} {output_type} from {segment_count} total segments")
        
        print(f"Found {len(all_species)} unique species:")
        for species in sorted(species_counts.keys(), key=lambda x: species_counts[x], reverse=True):
            count_label = "chunks" if chunk_duration else "segments"
            print(f"  {species}: {species_counts[species]} {count_label}")
        
        return file_count


class DOCDataProcessor(BaseDataProcessor):
    def load_metadata(self, metadata_path):
        metadata = {}
        if not os.path.exists(metadata_path):
            print(f"Warning: Metadata file {metadata_path} not found")
            return metadata
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                primary_label = row['primary_label']
                
                secondary_labels = []
                if row['secondary_labels'] and row['secondary_labels'] != '[]':
                    secondary_labels = ast.literal_eval(row['secondary_labels'])
                    if not isinstance(secondary_labels, list):
                        secondary_labels = [secondary_labels]
                
                metadata[filename] = {
                    'primary_label': primary_label,
                    'secondary_labels': secondary_labels,
                    'all_labels': [primary_label] + secondary_labels
                }
        print(f"Loaded metadata for {len(metadata)} files from {metadata_path}")
        
        return metadata

    def load_bird_name_mapping(self, csv_path):
        mapping = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row['eBird']] = row['CommonName']
        return mapping

    def process(self, input_folder, output_folder, max_species=None, min_examples=None, 
                max_samples=None, specific_species=None, name_mapping=None, overwrite=False,
                ignore_multilabel=False, max_segments=None):
        print(f"Loading bird data from {input_folder}")
        
        if max_segments:
            print(f"Max segments limit: {max_segments}")
        
        if os.path.exists(output_folder):
            if overwrite:
                smart_overwrite_folder(output_folder, preserve_noise=True)
            else:
                print(f"Error: Output folder {output_folder} already exists. Use --overwrite to overwrite.")
                return 0
        
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output folder: {output_folder}")
        
        name_mapping = name_mapping or {}
        if name_mapping:
            print(f"Loaded {len(name_mapping)} bird name mappings")

        all_metadata = {}
        all_species = set()
        
        for big_folder in sorted(os.listdir(input_folder)):
            big_folder_path = os.path.join(input_folder, big_folder)
            if not os.path.isdir(big_folder_path):
                continue
                
            metadata_file = None
            
            nested_path = os.path.join(big_folder_path, big_folder)
            if os.path.isdir(nested_path):
                for file in sorted(os.listdir(nested_path)):
                    if file.endswith('_metadata.csv'):
                        metadata_file = os.path.join(nested_path, file)
                        break
            else:
                for file in sorted(os.listdir(big_folder_path)):
                    if file.endswith('_metadata.csv'):
                        metadata_file = os.path.join(big_folder_path, file)
                        break
            
            if metadata_file:
                folder_metadata = self.load_metadata(metadata_file)
                all_metadata.update(folder_metadata)
                
                for file_info in folder_metadata.values():
                    all_species.update(file_info['all_labels'])
        
        print(f"Found {len(all_species)} total species in metadata")
        print(f"Minimum examples per species: {min_examples}")
        
        # Count species, respecting ignore_multilabel filter
        species_counts = {}
        for species in all_species:
            count = 0
            for metadata in all_metadata.values():
                if species in metadata['all_labels']:
                    # If ignoring multilabel, only count single-label occurrences
                    if ignore_multilabel:
                        if len(metadata['all_labels']) == 1:
                            count += 1
                    else:
                        count += 1
            species_counts[species] = count
        
        if specific_species:
            print(f"Using specific species: {specific_species}")
            valid_species = []
            for species in specific_species:
                if species in species_counts:
                    if species_counts[species] >= min_examples:
                        valid_species.append(species)
                        print(f"  {species}: {species_counts[species]} examples ✓")
                    else:
                        print(f"  {species}: {species_counts[species]} examples (< {min_examples}) ❌")
                else:
                    print(f"  {species}: not found in metadata ❌")
            
            if not valid_species:
                print("❌ No valid species found from the specified list!")
                return 0
        else:
            valid_species = [species for species, count in species_counts.items() 
                             if count >= min_examples]
            
            print(f"Species with >= {min_examples} examples: {len(valid_species)}")
            
            valid_species = sorted(valid_species, key=lambda x: species_counts[x], reverse=True)[:max_species]
        
        print(f"Selected {len(valid_species)} species: {valid_species}")
        
        species_to_idx = {species: idx for idx, species in enumerate(valid_species)}
        valid_species_set = set(valid_species)
        
        species_to_files = {}
        for big_folder in sorted(os.listdir(input_folder)):
            big_folder_path = os.path.join(input_folder, big_folder)
            if not os.path.isdir(big_folder_path):
                continue
                
            nested_path = os.path.join(big_folder_path, big_folder)
            train_audio_path = os.path.join(nested_path, "train_audio") if os.path.isdir(nested_path) else os.path.join(big_folder_path, "train_audio")
            
            if os.path.isdir(train_audio_path):
                species_folders = sorted(os.listdir(train_audio_path))
                
                for species_folder in species_folders:
                    if species_folder in valid_species_set:
                        species_path = os.path.join(train_audio_path, species_folder)
                        if os.path.isdir(species_path):
                            species_to_files.setdefault(species_folder, [])
                            files = sorted(os.listdir(species_path))
                            species_to_files[species_folder] += [os.path.join(species_path, f) for f in files]
        
        print(f"Collected audio files for {len(species_to_files)} valid species")
        
        data_folder = os.path.join(output_folder, "data")
        os.makedirs(data_folder, exist_ok=True)
        
        # Setup audio directory if needed
        self.setup_audio_dir(output_folder)
        
        file_labels = []
        file_count = 0
        species_example_saved = set()
        # Don't pre-slice - we need to filter first, then limit
        processed_files = 0
        
        for species, files in species_to_files.items():
            species_count = 0
            for sound_file in files:
                # Check if we've saved enough for this species
                if max_samples and species_count >= max_samples:
                    break
                
                if max_segments and file_count >= max_segments:
                    break
                
                processed_files += 1
                
                if not sound_file.lower().endswith((".wav", ".flac")):
                    continue
                
                train_audio_idx = sound_file.find("train_audio")
                if train_audio_idx == -1:
                    continue
                    
                relative_path = sound_file[train_audio_idx + len("train_audio") + 1:]
                
                file_metadata = None
                if relative_path in all_metadata:
                    file_metadata = all_metadata[relative_path]
                else:
                    filename_only = os.path.basename(sound_file)
                    for metadata_key in all_metadata:
                        if filename_only in metadata_key:
                            file_metadata = all_metadata[metadata_key]
                            break
                
                if file_metadata is None:
                    file_labels_binary = np.zeros(len(valid_species))
                    if species in species_to_idx:
                        file_labels_binary[species_to_idx[species]] = 1
                else:
                    file_labels_binary = np.zeros(len(valid_species))
                    for label in file_metadata['all_labels']:
                        if label in species_to_idx:
                            file_labels_binary[species_to_idx[label]] = 1
                
                if np.sum(file_labels_binary) == 0:
                    continue
                
                # Check if we should skip multi-label samples (based on ORIGINAL metadata, not filtered)
                if ignore_multilabel:
                    if file_metadata and len(file_metadata['all_labels']) > 1:
                        continue
                    elif not file_metadata and len([species]) > 1:
                        continue
                
                if self.output_format == 'wav':
                    file_basename = f"file_{file_count:08d}"
                    output_path = os.path.join(data_folder, f"{file_basename}.wav")
                    
                    success = self.segment_extractor.extract_audio_segment(
                        sound_file, 
                        0,
                        None,
                        output_path
                    )
                    
                    if success:
                        file_labels.append({
                            'filename': f"{file_basename}.wav",
                            'class_names': file_metadata['all_labels'] if file_metadata else [species],
                            'primary_class': species,
                            'source_file': sound_file
                        })
                        
                        file_count += 1
                        species_count += 1
                else:
                    sg_raw = self.spec_processor.process_audio_file(sound_file)
                    if sg_raw is not None:
                        file_basename = f"file_{file_count:08d}"
                        self.spec_processor.save_spectrogram(sg_raw, data_folder, file_basename)
                        
                        # Save audio if requested
                        audio_filename = None
                        if self.audio_output_dir:
                            audio_filename = self.save_audio_segment(sound_file, 0, None, f"{file_basename}.wav")
                        
                        label_entry = {
                            'filename': f"{file_basename}.npy",
                            'class_names': file_metadata['all_labels'] if file_metadata else [species],
                            'primary_class': species,
                            'source_file': sound_file
                        }
                        if audio_filename:
                            label_entry['audio_file'] = audio_filename
                        
                        file_labels.append(label_entry)
                        
                        file_count += 1
                        species_count += 1
                        
                        if species not in species_example_saved:
                            self.spec_processor.save_example_image(sg_raw, output_folder, f"example_{species}")
                            species_example_saved.add(species)
                
                if processed_files % 50 == 0 or processed_files in [1, 10]:
                    output_type = "WAV files" if self.output_format == 'wav' else "spectrograms"
                    print(f"Checked {processed_files} files, saved {file_count} {output_type}")
            
            if species_count > 0:
                print(f"  {species}: saved {species_count} files")
            
            if max_segments and file_count >= max_segments:
                print(f"Reached max_segments limit of {max_segments}")
                break
        
        self.save_labels(output_folder, file_labels, valid_species, 'DOC')
            
        print(f"Saved {file_count} bird spectrogram files to {output_folder}")
        return file_count


class ESCDataProcessor(BaseDataProcessor):
    def load_esc_metadata(self, metadata_path):
        metadata = {}
        categories = set()
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file {metadata_path} not found")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                category = row['category']
                target = int(row['target'])
                
                metadata[filename] = {
                    'category': category,
                    'target': target
                }
                categories.add(category)
                
        print(f"Loaded metadata for {len(metadata)} files from {metadata_path}")
        print(f"Found {len(categories)} unique categories")
        return metadata, sorted(categories)

    def process(self, input_folder, output_folder, overwrite=False, max_segments=None, max_species=None, min_examples=None, specific_species=None, max_samples=None):
        print(f"Loading ESC-50 data from {input_folder}")
        
        if max_segments:
            print(f"Max segments limit: {max_segments}")
        
        audio_folder = os.path.join(input_folder, "audio")
        metadata_path = os.path.join(input_folder, "meta", "esc50.csv")
        
        if not os.path.exists(audio_folder):
            raise FileNotFoundError(f"Audio folder not found: {audio_folder}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        if os.path.exists(output_folder):
            if overwrite:
                smart_overwrite_folder(output_folder, preserve_noise=True)
            else:
                raise FileExistsError(f"Output folder {output_folder} already exists. Use overwrite=True to overwrite.")
        
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output folder: {output_folder}")
        
        metadata, all_categories = self.load_esc_metadata(metadata_path)
        
        # Count files per category
        category_counts = {}
        category_to_files_temp = {}
        for filename, file_metadata in metadata.items():
            category = file_metadata['category']
            audio_path = os.path.join(audio_folder, filename)
            if os.path.exists(audio_path):
                category_counts[category] = category_counts.get(category, 0) + 1
                if category not in category_to_files_temp:
                    category_to_files_temp[category] = []
                category_to_files_temp[category].append({'path': audio_path})
        
        # Filter by specific_species if provided
        if specific_species:
            selected_categories = [c for c in specific_species if c in category_counts]
            print(f"Using specific categories: {selected_categories}")
        else:
            # Filter by min_examples
            if min_examples:
                selected_categories = [c for c, ct in category_counts.items() if ct >= min_examples]
                print(f"Categories with >= {min_examples} examples: {len(selected_categories)}")
            else:
                selected_categories = list(category_counts.keys())
            
            # Sort by count and take top max_species
            selected_categories = sorted(selected_categories, key=lambda x: category_counts[x], reverse=True)
            if max_species:
                selected_categories = selected_categories[:max_species]
                print(f"Selected top {max_species} categories by count")
        
        print(f"Processing {len(selected_categories)} categories: {selected_categories}")
        for cat in selected_categories:
            print(f"  {cat}: {category_counts.get(cat, 0)} files")
        
        category_to_idx = {category: idx for idx, category in enumerate(selected_categories)}
        
        category_to_files = {category: category_to_files_temp.get(category, []) for category in selected_categories}
        
        data_folder = os.path.join(output_folder, "data")
        os.makedirs(data_folder, exist_ok=True)
        print("Processing all data into single folder")
        
        # Setup audio directory if needed
        self.setup_audio_dir(output_folder)
        
        labels = []
        file_count = 0
        categories_example_saved = set()
        
        total_files = sum(len(files) for files in category_to_files.values())
        processed_files = 0
        
        for category in selected_categories:
            files = category_to_files[category]
            category_count = 0
            
            # Limit files per category if max_samples specified
            if max_samples:
                files = files[:max_samples]
            
            for file_info in files:
                if max_segments and file_count >= max_segments:
                    break
                
                processed_files += 1
                audio_path = file_info['path']
                
                sg_raw = self.spec_processor.process_audio_file(audio_path)
                if sg_raw is not None:
                    file_basename = f"file_{file_count:08d}"
                    self.spec_processor.save_spectrogram(sg_raw, data_folder, file_basename)
                    
                    # Save audio if requested
                    audio_filename = None
                    if self.audio_output_dir:
                        audio_filename = self.save_audio_segment(audio_path, 0, None, f"{file_basename}.wav")
                    
                    label_entry = {
                        'filename': f"{file_basename}.npy",
                        'primary_class': category,
                        'class_names': [category],
                        'source_file': audio_path
                    }
                    if audio_filename:
                        label_entry['audio_file'] = audio_filename
                    
                    labels.append(label_entry)
                    file_count += 1
                    category_count += 1
                    
                    if category not in categories_example_saved:
                        example_name = f"example_{category}"
                        self.spec_processor.save_example_image(sg_raw, output_folder, example_name)
                        categories_example_saved.add(category)
                
                if processed_files % 50 == 0 or processed_files in [1, 10]:
                    print(f"Processed {processed_files}/{total_files} files ({100*processed_files/total_files:.1f}%), "
                          f"saved {file_count} spectrograms")
            
            if category_count > 0:
                print(f"  {category}: {category_count} files")
            
            if max_segments and file_count >= max_segments:
                print(f"Reached max_segments limit of {max_segments}")
                break
        
        self.save_labels(output_folder, labels, selected_categories, 'ESC-50')
            
        print(f"Saved {file_count} ESC spectrograms to {output_folder}")
        return file_count


class NoiseDataProcessor(BaseDataProcessor):
    def process(self, input_folder, output_folder, num_samples, max_segments=None):
        print(f"Loading noise files from {input_folder}")
        
        if max_segments:
            print(f"Max segments limit: {max_segments}")
            num_samples = min(num_samples, max_segments)
        
        data_folder = os.path.join(output_folder, "data")
        os.makedirs(data_folder, exist_ok=True)
        
        # Setup audio directory if needed
        self.setup_audio_dir(output_folder)

        # Check for zip files and extract if needed
        import zipfile
        import shutil
        temp_extract_dir = None
        
        zip_files = [f for f in os.listdir(input_folder) if f.endswith('.zip')]
        if zip_files:
            print(f"Found {len(zip_files)} zip files, extracting...")
            temp_extract_dir = tempfile.mkdtemp(prefix='noise_extract_')
            
            for zip_file in zip_files:
                zip_path = os.path.join(input_folder, zip_file)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_extract_dir)
                    print(f"  Extracted {zip_file}")
                except Exception as e:
                    print(f"  Warning: Failed to extract {zip_file}: {e}")
            
            search_folder = temp_extract_dir
        else:
            search_folder = input_folder

        try:
            noise_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(search_folder)
                for f in files if f.lower().endswith((".wav", ".flac", ".mp3"))
            ]

            print(f"Found {len(noise_files)} noise files")

            if len(noise_files) == 0:
                print("Warning: No noise files found!")
                return 0

            if len(noise_files) > num_samples:
                noise_files = random.sample(noise_files, num_samples)
                print(f"Randomly selected {num_samples} noise files")

            count = 0
            files_info = []
            for i, f in enumerate(noise_files):
                sg = self.spec_processor.process_audio_file(f)
                if sg is not None:
                    fname = f"noise_{count:08d}"
                    self.spec_processor.save_spectrogram(sg, data_folder, fname)
                    
                    # Save audio if requested
                    audio_filename = None
                    if self.audio_output_dir:
                        audio_filename = self.save_audio_segment(f, 0, None, f"{fname}.wav")
                    
                    file_entry = {'filename': f"{fname}.npy", 'source_file': f}
                    if audio_filename:
                        file_entry['audio_file'] = audio_filename
                    files_info.append(file_entry)
                    
                    if count < 3:
                        self.spec_processor.save_example_image(sg, output_folder, f"example_noise_{count}")
                    
                    count += 1
                    
                    if count % 50 == 0 or count in [1, 10]:
                        print(f"Processed {count}/{len(noise_files)} noise files ({100*count/len(noise_files):.1f}%)")

            self.save_labels(output_folder, files_info, ['noise'], 'Noise',
                            {'num_files': count, 'description': 'noise_spectrograms'})

            print(f"Saved {count} noise spectrograms to {output_folder}")
            return count
        
        finally:
            # Clean up temporary extraction directory
            if temp_extract_dir and os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir)
                print(f"Cleaned up temporary extraction directory")


class InferenceDataProcessor(BaseDataProcessor):
    def process(self, input_folder, output_folder, overwrite=False, chunk_duration=None, max_segments=None):
        print(f"Loading audio files for inference from {input_folder}")
        
        if chunk_duration:
            print(f"Splitting into {chunk_duration}s chunks")
        
        if max_segments:
            print(f"Max segments limit: {max_segments}")
        
        if os.path.exists(output_folder):
            if overwrite:
                smart_overwrite_folder(output_folder, preserve_noise=False)
            else:
                raise FileExistsError(f"Output folder {output_folder} already exists. Use overwrite=True to overwrite.")
        
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output folder: {output_folder}")
        
        audio_files = []
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith((".wav", ".flac", ".mp3")):
                    audio_files.append(os.path.join(root, file))
        
        print(f"Found {len(audio_files)} audio files")
        
        if len(audio_files) == 0:
            print("Warning: No audio files found!")
            return 0
        
        data_folder = os.path.join(output_folder, "data")
        os.makedirs(data_folder, exist_ok=True)
        
        # Setup audio directory if needed
        self.setup_audio_dir(output_folder)
        
        files_info = []
        file_count = 0
        
        if chunk_duration:
            for audio_file in audio_files:
                try:
                    file_info = sf.info(audio_file)
                    duration = file_info.frames / file_info.samplerate
                    num_chunks = int(np.ceil(duration / chunk_duration))
                    
                    relative_path = os.path.relpath(audio_file, input_folder)
                    print(f"Processing {relative_path} ({duration:.1f}s -> {num_chunks} chunks)")
                    
                    for chunk_idx in range(num_chunks):
                        if max_segments and file_count >= max_segments:
                            break
                        
                        start_time = chunk_idx * chunk_duration
                        end_time = min(start_time + chunk_duration, duration)
                        
                        sg_raw = self.spec_processor.process_audio_segment(audio_file, start_time, end_time)
                        
                        if sg_raw is not None:
                            filename = f"file_{file_count:08d}"
                            self.spec_processor.save_spectrogram(sg_raw, data_folder, filename)
                            
                            # Save audio chunk if requested
                            audio_filename = None
                            if self.audio_output_dir:
                                audio_filename = self.save_audio_segment(audio_file, start_time, end_time, f"{filename}.wav")
                            
                            row_id_time = int((chunk_idx + 1) * chunk_duration)
                            
                            file_entry = {
                                'filename': f"{filename}.npy",
                                'source_file': relative_path,
                                'row_id': f"{relative_path}_{row_id_time}",
                                'start_time': start_time,
                                'end_time': end_time,
                                'chunk_index': chunk_idx,
                                'class_names': []
                            }
                            if audio_filename:
                                file_entry['audio_file'] = audio_filename
                            
                            files_info.append(file_entry)
                            file_count += 1
                            
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
                
                if max_segments and file_count >= max_segments:
                    print(f"Reached max_segments limit of {max_segments}")
                    break
        else:
            for audio_file in audio_files:
                if max_segments and file_count >= max_segments:
                    print(f"Reached max_segments limit of {max_segments}")
                    break
                
                sg_raw = self.spec_processor.process_audio_file(audio_file)
                
                if sg_raw is not None:
                    file_basename = f"file_{file_count:08d}"
                    self.spec_processor.save_spectrogram(sg_raw, data_folder, file_basename)
                    
                    # Save audio if requested
                    audio_filename = None
                    if self.audio_output_dir:
                        audio_filename = self.save_audio_segment(audio_file, 0, None, f"{file_basename}.wav")
                    
                    relative_path = os.path.relpath(audio_file, input_folder)
                    
                    file_entry = {
                        'filename': f"{file_basename}.npy",
                        'source_file': relative_path,
                        'class_names': []
                    }
                    if audio_filename:
                        file_entry['audio_file'] = audio_filename
                    
                    files_info.append(file_entry)
                    
                    file_count += 1
                    
                    if file_count % 50 == 0 or file_count in [1, 10]:
                        print(f"Processed {file_count}/{len(audio_files)} files ({100*file_count/len(audio_files):.1f}%)")
        
        self.save_labels(output_folder, files_info, [], 'Inference',
                        {'num_files': file_count, 
                         'chunk_duration': chunk_duration,
                         'description': 'spectrograms_for_inference'})
        
        print(f"Saved {file_count} spectrograms to {output_folder}")
        return file_count


class LongAudioInferenceProcessor(BaseDataProcessor):
    def process(self, input_folder, output_folder, overwrite=False, chunk_duration=5.0):
        print(f"Loading long audio files for chunked inference from {input_folder}")
        print(f"Chunk duration: {chunk_duration} seconds")
        
        if os.path.exists(output_folder):
            if overwrite:
                smart_overwrite_folder(output_folder, preserve_noise=False)
            else:
                raise FileExistsError(f"Output folder {output_folder} already exists. Use overwrite=True to overwrite.")
        
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output folder: {output_folder}")
        
        audio_files = []
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith((".wav", ".flac")):
                    audio_files.append(os.path.join(root, file))
        
        print(f"Found {len(audio_files)} audio files")
        
        if len(audio_files) == 0:
            print("Warning: No audio files found!")
            return 0
        
        data_folder = os.path.join(output_folder, "data")
        os.makedirs(data_folder, exist_ok=True)
        
        files_info = []
        chunk_count = 0
        
        for audio_file in audio_files:
            try:
                file_info = sf.info(audio_file)
                duration = file_info.frames / file_info.samplerate
                num_chunks = int(np.ceil(duration / chunk_duration))
                
                relative_path = os.path.relpath(audio_file, input_folder)
                print(f"Processing {relative_path} ({duration:.1f}s -> {num_chunks} chunks)")
                
                for chunk_idx in range(num_chunks):
                    start_time = chunk_idx * chunk_duration
                    end_time = min(start_time + chunk_duration, duration)
                    
                    sg_raw = self.spec_processor.process_audio_segment(audio_file, start_time, end_time)
                    
                    if sg_raw is not None:
                        filename = f"file_{chunk_count:08d}"
                        self.spec_processor.save_spectrogram(sg_raw, data_folder, filename)
                        
                        row_id_time = int((chunk_idx + 1) * chunk_duration)
                        
                        files_info.append({
                            'filename': f"{filename}.npy",
                            'source_file': relative_path,
                            'row_id': f"{relative_path}_{row_id_time}",
                            'start_time': start_time,
                            'end_time': end_time,
                            'chunk_index': chunk_idx,
                            'class_names': []
                        })
                        chunk_count += 1
                        
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        self.save_labels(output_folder, files_info, [], 'LongAudioInference',
                        {'num_chunks': chunk_count, 
                         'chunk_duration': chunk_duration,
                         'description': 'spectrograms_from_long_audio_chunks'})
        
        print(f"Saved {file_count} spectrograms to {output_folder}")
        return file_count


def load_data(source_type, input_folder, output_folder, window_seconds=None, hop_seconds=None,
              freq_bins=None, fs=None, overwrite=False, output_format='spectrogram', 
              ignore_multilabel=False, with_audio=False, audioset_fbank=False, **kwargs):
    if window_seconds is None:
        window_seconds = config.DEFAULT_WINDOW_SECONDS
    if hop_seconds is None:
        hop_seconds = config.DEFAULT_HOP_SECONDS
    if freq_bins is None:
        freq_bins = config.SPECTROGRAM_PARAMS['nfilters']
    if fs is None:
        fs = config.DEFAULT_SAMPLE_RATE
    
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder {input_folder} does not exist")

    spec_processor = None
    segment_extractor = None
    
    if output_format == 'spectrogram':
        if audioset_fbank:
            spec_processor = AudioSetFbankProcessor(
                target_sample_rate=16000,
                frame_length_ms=25.0,
                frame_shift_ms=10.0,
                num_mel_bins=128,
            )
        else:
            spec_processor = SpectrogramProcessor(window_seconds, hop_seconds, freq_bins, 
                                                 fs, config.SPECTROGRAM_PARAMS)
    elif output_format == 'wav':
        segment_extractor = SegmentExtractor(target_sr=fs)
    else:
        raise ValueError(f"Unknown output format: {output_format}")

    print(f"\n{'='*50}")
    print(f"Processing {source_type.upper()} data...")
    print(f"Output format: {output_format.upper()}")
    if output_format == 'spectrogram':
        if audioset_fbank:
            print("Feature mode: AUDIOSET_FBANK (Kaldi fbank @ 16kHz, 25ms window, 10ms hop, 128 mel bins)")
        else:
            print(f"Window: {window_seconds*1000:.1f}ms, Hop: {hop_seconds*1000:.1f}ms, Freq bins: {freq_bins}")
    else:
        print(f"Sample rate: {fs} Hz")
    print(f"{'='*50}")
    
    if source_type == 'avianz':
        # Always load name mapping for consistent eBird code labels
        name_mapping = None
        mapping_file = kwargs.get('name_mapping')
        if mapping_file is None:
            # Look for mapping file in script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_mapping = os.path.join(script_dir, "DOC_bird_naming_map.csv")
            if os.path.exists(default_mapping):
                mapping_file = default_mapping
        
        if mapping_file and os.path.exists(mapping_file):
            print(f"Loading bird name mapping from {mapping_file}...")
            import csv
            import pandas as pd
            df = pd.read_csv(mapping_file)
            name_mapping = {}
            for _, row in df.iterrows():
                ebird_code = row['eBird']
                if pd.isna(ebird_code):
                    continue
                if pd.notna(row['CommonName']):
                    name_mapping[row['CommonName']] = ebird_code
                if pd.notna(row['ExtraName']):
                    name_mapping[row['ExtraName']] = ebird_code
                if pd.notna(row['ListDOCBirds']):
                    name_mapping[row['ListDOCBirds']] = ebird_code
                name_mapping[ebird_code] = ebird_code
            # Add hardcoded fixes for common mismatches
            name_mapping['Ruru'] = 'morepo2'
            name_mapping['Morepork'] = 'morepo2'
            name_mapping['Bellbird/Tui'] = 'nezbel1'
            name_mapping['Tomtit (Nth Is)'] = 'tomtit1'
            name_mapping['Fantail (Nth Is)'] = 'nezfan1'
            name_mapping['Fantail (spp)'] = 'nezfan1'
            name_mapping['Kaka (Nth Is)'] = 'nezkak1'
            name_mapping['Kaka (spp)'] = 'nezkak1'
            name_mapping['Tui (spp)'] = 'tui1'
            name_mapping['Robin (Nth Is)'] = 'nezrob2'
            name_mapping['Pigeon (NZ Kereru Kukupa)'] = 'nezpig2'
            name_mapping['Warbler (Grey)'] = 'gryger1'
            name_mapping['Magpie (Australian)'] = 'ausmag2'
            name_mapping['Myna (Indian)'] = 'commyn'
            name_mapping['Gull (Southern Black-backed)'] = 'kelgul'
            name_mapping['Plover (Spur-winged)'] = 'maslap1'
            name_mapping['Rosella (Eastern)'] = 'easros1'
            name_mapping['Cockatoo (Sulphur-crested)'] = 'succoc'
            name_mapping['Sparrow (House)'] = 'houspa'
            print(f"  Loaded {len(name_mapping)} name mappings")
        
        processor = AviaNZDataProcessor(spec_processor, segment_extractor, output_format, with_audio, name_mapping)
        file_count = processor.process(
            input_folder=input_folder,
            output_folder=output_folder,
            overwrite=overwrite,
            min_certainty=kwargs.get('min_certainty', 50),
            skip_species=kwargs.get('skip_species'),
            chunk_duration=kwargs.get('chunk_duration'),
            max_segments=kwargs.get('max_segments'),
            max_samples=kwargs.get('max_samples'),
            specific_species=kwargs.get('specific_species'),
            ignore_multilabel=ignore_multilabel,
            max_species=kwargs.get('max_species'),
            min_examples=kwargs.get('min_examples')
        )
    elif source_type == 'doc':
        processor = DOCDataProcessor(spec_processor, segment_extractor, output_format, with_audio)
        file_count = processor.process(
            input_folder=input_folder,
            output_folder=output_folder,
            max_species=kwargs.get('max_species', config.DEFAULT_MAX_SPECIES),
            min_examples=kwargs.get('min_examples', config.DEFAULT_MIN_EXAMPLES),
            max_samples=kwargs.get('max_samples', config.DEFAULT_MAX_SAMPLES),
            specific_species=kwargs.get('specific_species'),
            name_mapping=kwargs.get('name_mapping'),
            overwrite=overwrite,
            ignore_multilabel=ignore_multilabel,
            max_segments=kwargs.get('max_segments')
        )
    elif source_type == 'esc':
        if output_format == 'wav':
            raise ValueError("ESC dataset does not support WAV output format")
        processor = ESCDataProcessor(spec_processor, None, output_format, with_audio)
        file_count = processor.process(
            input_folder=input_folder,
            output_folder=output_folder,
            overwrite=overwrite,
            max_segments=kwargs.get('max_segments'),
            max_species=kwargs.get('max_species'),
            min_examples=kwargs.get('min_examples'),
            specific_species=kwargs.get('specific_species'),
            max_samples=kwargs.get('max_samples')
        )
    elif source_type == 'noise':
        if output_format == 'wav':
            raise ValueError("Noise dataset does not support WAV output format")
        processor = NoiseDataProcessor(spec_processor, None, output_format, with_audio)
        file_count = processor.process(
            input_folder=input_folder,
            output_folder=output_folder,
            num_samples=kwargs.get('num_samples', config.DEFAULT_NOISE_SAMPLES),
            max_segments=kwargs.get('max_segments')
        )
    elif source_type == 'inference':
        if output_format == 'wav':
            raise ValueError("Inference mode does not support WAV output format")
        processor = InferenceDataProcessor(spec_processor, None, output_format, with_audio)
        file_count = processor.process(
            input_folder=input_folder,
            output_folder=output_folder,
            overwrite=overwrite,
            chunk_duration=kwargs.get('chunk_duration'),
            max_segments=kwargs.get('max_segments')
        )
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    print(f"\n{'='*50}")
    print("Processing complete!")
    print(f"Total files: {file_count}")
    print(f"Output saved to: {output_folder}")
    print(f"{'='*50}")
    
    return file_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified data loader for converting audio data to spectrograms or extracting WAV segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Source Types:
  avianz     - AviaNZ annotated data (.wav.data files)
  doc        - DOC bird sound data with metadata
  esc        - ESC-50 environmental sound dataset
  noise      - Background noise audio files
  inference  - Audio files for inference

Output Formats:
  spectrogram - Convert to spectrograms (.npy files) [default]
  wav         - Extract as WAV files (avianz and doc only)

Common Arguments Work Across All Sources:
  --max-segments N    Limit to N total output files
  --max-samples N     Same as --max-segments (or per-species for DOC)
  --species "a,b,c"   Filter to specific species/categories
  --with-audio        Save audio files alongside spectrograms
  --chunk-duration X  Split into X-second chunks
  --overwrite         Overwrite existing output

Examples:
  # Process AviaNZ annotated data as spectrograms
  python data_loader.py avianz "Sound Files/GSK" "Sound Files/GSK_spec"
  
  # Process only first 1000 segments from AviaNZ data
  python data_loader.py avianz "Sound Files/GSK" "Sound Files/GSK_spec" --max-segments 1000
  
  # Process only specific species from AviaNZ data using eBird codes
  python data_loader.py avianz "Sound Files/GSK" "Sound Files/GSK_spec" --species "morepo2,tui1"
  
  # Extract AviaNZ annotated segments as WAV files
  python data_loader.py avianz "Sound Files/GSK" "Sound Files/GSK_wav" --format wav
  
  # Process DOC bird data as spectrograms with specific species
  python data_loader.py doc "Sound Files/NZ bird sounds" "Sound Files/DOC_spec" --species "tui1,bellbi1"
  
  # Process DOC bird data with audio files for listening
  python data_loader.py doc "Sound Files/NZ bird sounds" "Sound Files/DOC_spec" --with-audio
  
  # Extract DOC bird data as WAV files for testing
  python data_loader.py doc "Sound Files/NZ bird sounds" "Sound Files/DOC_wav" --format wav --max-samples 100
  
  # Process ESC-50 data (spectrograms only)
  python data_loader.py esc "Sound Files/ESC-50-master" "Sound Files/ESC_spec"
  
  # Process ESC-50 with audio for listening to samples
  python data_loader.py esc "Sound Files/ESC-50-master" "Sound Files/ESC_spec" --with-audio
  
  # Process noise files
  python data_loader.py noise "Sound Files/freefield" "Sound Files/Noise_spec" --samples 100
  
  # Process raw audio files for inference
  python data_loader.py inference "path/to/audio" "path/to/output_spec"
  
  # Process long audio files split into 5-second chunks (for comparison with kaytoo)
  python data_loader.py inference "Sound Files/NZ bird sounds" "Sound Files/Inference_spec" --chunk-duration 5.0
  
  # Custom spectrogram parameters
  python data_loader.py avianz "Sound Files/GSK" "Sound Files/GSK_spec" --window 0.025 --hop 0.010 --freq-bins 128
        """
    )
    
    parser.add_argument('source_type', type=str, choices=['avianz', 'doc', 'esc', 'noise', 'inference'],
                       help="Type of data source to process")
    parser.add_argument('input_folder', type=str,
                       help="Path to input folder containing audio data")
    parser.add_argument('output_folder', type=str,
                       help="Path to output folder for spectrograms or WAV files")
    
    parser.add_argument('--format', type=str, choices=['spectrogram', 'wav'], default='spectrogram',
                       help="Output format: 'spectrogram' for .npy files or 'wav' for audio files (default: spectrogram)")
    parser.add_argument('--window', type=float, default=config.DEFAULT_WINDOW_SECONDS,
                       help=f"[Spectrogram] Window width in seconds (default: {config.DEFAULT_WINDOW_SECONDS})")
    parser.add_argument('--hop', type=float, default=config.DEFAULT_HOP_SECONDS,
                       help=f"[Spectrogram] Hop length in seconds (default: {config.DEFAULT_HOP_SECONDS})")
    parser.add_argument('--freq-bins', type=int, default=config.SPECTROGRAM_PARAMS['nfilters'],
                       help=f"[Spectrogram] Number of frequency bins (default: {config.SPECTROGRAM_PARAMS['nfilters']})")
    parser.add_argument('--fs', type=int, default=config.DEFAULT_SAMPLE_RATE,
                       help=f"Sample rate in Hz (default: {config.DEFAULT_SAMPLE_RATE})")
    parser.add_argument('--overwrite', action='store_true',
                       help="Overwrite existing output folder")
    
    parser.add_argument('--min-certainty', type=int, default=50,
                       help="Minimum certainty threshold for labels (default: 50)")
    parser.add_argument('--skip-species', type=str,
                       help="Comma-separated list of species/categories to skip")
    parser.add_argument('--max-segments', type=int,
                       help="Maximum total output files/segments to process across all sources (default: no limit)")
    
    parser.add_argument('--max-species', type=int, default=config.DEFAULT_MAX_SPECIES,
                       help=f"Maximum number of species/categories to process (default: {config.DEFAULT_MAX_SPECIES})")
    parser.add_argument('--min-examples', type=int, default=config.DEFAULT_MIN_EXAMPLES,
                       help=f"Minimum examples per species/category to include it (default: {config.DEFAULT_MIN_EXAMPLES})")
    parser.add_argument('--max-samples', type=int, default=config.DEFAULT_MAX_SAMPLES,
                       help=f"Maximum samples per species/category (AviaNZ/DOC), OR total limit (ESC/Noise/Inference) (default: {config.DEFAULT_MAX_SAMPLES})")
    parser.add_argument('--species', type=str,
                       help="Comma-separated list of specific species/categories to include (e.g., 'morepo2,tui1' for birds, 'dog,cat' for ESC)")
    parser.add_argument('--mapping', type=str,
                       help="Path to bird name mapping CSV file (default: DOC_bird_naming_map.csv)")
    
    parser.add_argument('--samples', type=int, default=config.DEFAULT_NOISE_SAMPLES,
                       help=f"Maximum number of samples to process (default: {config.DEFAULT_NOISE_SAMPLES})")
    
    parser.add_argument('--chunk-duration', type=float, default=None,
                       help="Split audio into chunks of this duration in seconds (e.g., 5.0)")
    parser.add_argument('--ignore-multilabel', action='store_true',
                       help="Skip samples with multiple labels - only use single-label samples")
    parser.add_argument('--with-audio', action='store_true',
                       help="Also save audio segments to audio/ folder for listening")

    parser.add_argument('--audioset-fbank', action='store_true',
                       help="Generate AudioSet-style Kaldi fbank features for AST (forces 16kHz, 25ms window, 10ms hop, 128 mel bins). Saves linear fbank energies; training will still apply log transform as usual.")
    
    args = parser.parse_args()
    
    kwargs = {}
    
    if args.source_type == 'avianz':
        if args.skip_species:
            kwargs['skip_species'] = [s.strip() for s in args.skip_species.split(',')]
        if args.species:
            kwargs['specific_species'] = [s.strip() for s in args.species.split(',')]
        kwargs['min_certainty'] = args.min_certainty
        if args.chunk_duration:
            kwargs['chunk_duration'] = args.chunk_duration
        # For AviaNZ: max_segments = total limit, max_samples = per-species limit
        if args.max_segments:
            kwargs['max_segments'] = args.max_segments
        if args.max_samples:
            kwargs['max_samples'] = args.max_samples
        kwargs['max_species'] = args.max_species
        kwargs['min_examples'] = args.min_examples
    
    elif args.source_type == 'doc':
        if args.species:
            kwargs['specific_species'] = [s.strip() for s in args.species.split(',')]
        kwargs['max_species'] = args.max_species
        kwargs['min_examples'] = args.min_examples
        kwargs['max_samples'] = args.max_samples
        if args.max_segments:
            kwargs['max_segments'] = args.max_segments
        
        mapping_file = args.mapping
        if mapping_file is None:
            default_mapping = "DOC_bird_naming_map.csv"
            if os.path.exists(default_mapping):
                mapping_file = default_mapping
                print(f"Using default mapping file: {default_mapping}")
        
        if mapping_file:
            processor = DOCDataProcessor(None, None, 'spectrogram')
            kwargs['name_mapping'] = processor.load_bird_name_mapping(mapping_file)
    
    elif args.source_type == 'noise':
        kwargs['num_samples'] = args.samples
        # For Noise, both --max-segments and --max-samples mean the same thing
        if args.max_segments:
            kwargs['max_segments'] = args.max_segments
        elif args.max_samples:
            kwargs['max_segments'] = args.max_samples
    
    elif args.source_type == 'inference':
        if args.chunk_duration:
            kwargs['chunk_duration'] = args.chunk_duration
        # For Inference, both --max-segments and --max-samples mean the same thing
        if args.max_segments:
            kwargs['max_segments'] = args.max_segments
        elif args.max_samples:
            kwargs['max_segments'] = args.max_samples
    
    elif args.source_type == 'esc':
        # For ESC: support filtering by category/species
        if args.species:
            kwargs['specific_species'] = [s.strip() for s in args.species.split(',')]
        kwargs['max_species'] = args.max_species
        kwargs['min_examples'] = args.min_examples
        # max_segments = total limit, max_samples = per-category limit
        if args.max_segments:
            kwargs['max_segments'] = args.max_segments
        if args.max_samples:
            kwargs['max_samples'] = args.max_samples
    
    file_count = load_data(
        source_type=args.source_type,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        window_seconds=args.window,
        hop_seconds=args.hop,
        freq_bins=args.freq_bins,
        fs=args.fs,
        overwrite=args.overwrite,
        output_format=args.format,
        ignore_multilabel=args.ignore_multilabel,
        with_audio=args.with_audio,
        audioset_fbank=args.audioset_fbank,
        **kwargs
    )
    
    print(f"\n✓ Done! Processed {file_count} files.")
