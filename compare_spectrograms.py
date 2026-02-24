import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import subprocess
from normalizer import normalize_spectrogram


class SpectrogramComparer:
    def __init__(self, doc_root, joe_mo_root):
        self.doc_root = Path(doc_root)
        self.joe_mo_root = Path(joe_mo_root)
        
        with open(self.doc_root / 'labels.json') as f:
            self.doc_data = json.load(f)
        
        with open(self.joe_mo_root / 'labels.json') as f:
            self.joe_mo_data = json.load(f)
        
        self.doc_by_species = self.group_by_species(self.doc_data['files'])
        self.joe_mo_by_species = self.group_by_species(self.joe_mo_data['files'])
        
        self.common_species = sorted(set(self.doc_by_species.keys()) & set(self.joe_mo_by_species.keys()))
        
        if not self.common_species:
            raise ValueError("No common species found between datasets")
        
        self.current_species_idx = 0
        self.fig, self.axes = plt.subplots(2, 1, figsize=(14, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        self.playback_proc = None
        self.apply_normalizer = False
        self.doc_spec_raw = None
        self.joe_mo_spec_raw = None
        self.doc_filename = None
        self.joe_mo_filename = None
        
        self.load_new_pair()
        plt.tight_layout()
        plt.show()
    
    def group_by_species(self, files):
        by_species = {}
        for file_info in files:
            species = file_info['primary_class']
            if species not in by_species:
                by_species[species] = []
            by_species[species].append(file_info)
        return by_species
    
    def load_spectrogram(self, file_info, data_root):
        filename = file_info['filename']
        spec_path = data_root / 'data' / filename
        spec_linear = np.load(spec_path)
        spec_db = 10 * np.log10(spec_linear + 1e-10)
        return spec_db, filename

    def resolve_audio_path(self, file_info, data_root):
        audio_folder = data_root / 'audio'
        if 'audio_file' in file_info:
            return audio_folder / file_info['audio_file']
        filename = file_info['filename']
        base = Path(filename).stem
        return audio_folder / f"{base}.wav"

    def play_audio(self, file_info, data_root):
        audio_path = self.resolve_audio_path(file_info, data_root)
        if self.playback_proc:
            self.playback_proc.terminate()
            self.playback_proc.wait(timeout=1.0)
        self.playback_proc = subprocess.Popen(
            ['ffplay', '-nodisp', '-autoexit', '-af', 'pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1', str(audio_path)], 
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    
    def on_close(self, event):
        if self.playback_proc:
            self.playback_proc.terminate()
    
    def load_new_pair(self):
        current_species = self.common_species[self.current_species_idx]
        
        self.doc_file = random.choice(self.doc_by_species[current_species])
        self.joe_mo_file = random.choice(self.joe_mo_by_species[current_species])
        
        self.doc_spec_raw, self.doc_filename = self.load_spectrogram(self.doc_file, self.doc_root)
        self.joe_mo_spec_raw, self.joe_mo_filename = self.load_spectrogram(self.joe_mo_file, self.joe_mo_root)
        
        self.draw_spectrograms()
    
    def draw_spectrograms(self):
        doc_spec = normalize_spectrogram(self.doc_spec_raw) if self.apply_normalizer else self.doc_spec_raw
        joe_mo_spec = normalize_spectrogram(self.joe_mo_spec_raw) if self.apply_normalizer else self.joe_mo_spec_raw
        
        current_species = self.common_species[self.current_species_idx]
        
        max_width = max(doc_spec.shape[1], joe_mo_spec.shape[1])
        vmin = min(doc_spec.min(), joe_mo_spec.min())
        vmax = max(doc_spec.max(), joe_mo_spec.max())
        
        for ax in self.axes:
            ax.clear()
        
        self.axes[0].imshow(joe_mo_spec, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        self.axes[0].set_xlim(0, max_width)
        self.axes[0].set_title(f'Joe_mo: {current_species} - {self.joe_mo_filename} (shape: {joe_mo_spec.shape})')
        self.axes[0].set_ylabel('Frequency')
        self.axes[0].set_xlabel('Time bins')
        
        self.axes[1].imshow(doc_spec, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        self.axes[1].set_xlim(0, max_width)
        self.axes[1].set_title(f'DOC: {current_species} - {self.doc_filename} (shape: {doc_spec.shape})')
        self.axes[1].set_xlabel('Time bins')
        self.axes[1].set_ylabel('Frequency')
        
        species_info = f'Species {self.current_species_idx + 1}/{len(self.common_species)}: {current_species}'
        species_info += f' (Joe_mo: {len(self.joe_mo_by_species[current_species])}, DOC: {len(self.doc_by_species[current_species])})'
        norm_status = '[NORMALIZED]' if self.apply_normalizer else '[RAW]'
        self.fig.suptitle(
            f'{species_info} {norm_status}\nPress SPACE for new samples, ENTER for next species, '
            '1 to play top, 2 to play bottom, N to toggle normalizer, X to quit'
        )
        
        self.fig.canvas.draw()
    
    def next_species(self):
        self.current_species_idx = (self.current_species_idx + 1) % len(self.common_species)
        self.load_new_pair()
    
    def on_key(self, event):
        if event.key == ' ':
            self.load_new_pair()
        elif event.key == 'enter':
            self.next_species()
        elif event.key == '1':
            self.play_audio(self.joe_mo_file, self.joe_mo_root)
        elif event.key == '2':
            self.play_audio(self.doc_file, self.doc_root)
        elif event.key == 'n':
            self.apply_normalizer = not self.apply_normalizer
            self.draw_spectrograms()
        elif event.key == 'x':
            plt.close(self.fig)


if __name__ == '__main__':
    doc_root = 'test/doc_split/train'
    joe_mo_root = 'test/joe_mo_split/train'
    
    comparer = SpectrogramComparer(doc_root, joe_mo_root)
