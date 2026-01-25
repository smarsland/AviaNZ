"""
Shared audio processing functionality.
Consolidates common code between DOC and ESC data loaders.
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import config
import spectrogram


class AudioProcessor:
    """Shared audio file processing functionality."""
    
    def __init__(self, img_height, fs, spec_params):
        self.img_height = img_height
        self.fs = fs
        self.spec_params = spec_params
        self.window_width = img_height * 2
        self.window_inc = img_height
        self.sp = spectrogram.Spectrogram(window_width=self.window_width, incr=self.window_inc)

    def process_audio_file(self, sound_file):
        """Process single audio file to spectrogram."""
        try:
            file_info = sf.info(sound_file)
            duration = file_info.frames / file_info.samplerate
            
            if duration > config.MAX_FILE_DURATION_SECONDS:
                raise ValueError(f"File {sound_file} is {duration:.1f} seconds - longer than {config.MAX_FILE_DURATION_SECONDS/60:.0f} minutes")
            elif duration > config.WARNING_FILE_DURATION_SECONDS:
                warnings.warn(f"File {sound_file} is {duration:.1f} seconds - longer than {config.WARNING_FILE_DURATION_SECONDS/60:.0f} minute")
            
            self.sp.readSoundFile(sound_file, silent=True)
            if self.sp.audio_data.sample_rate != self.fs:
                self.sp.resample(self.fs)

            _ = self.sp.spectrogram(
                window_width=self.window_width,
                incr=self.window_inc,
                window=self.spec_params['windowType'],
                sgType=self.spec_params['sgType'],
                sgScale=self.spec_params['sgScale'],
                mean_normalise=self.spec_params['mean_normalise'],
                equal_loudness=self.spec_params['equal_loudness']
            )
            
            sg_raw = np.rot90(self.sp.sg)
            return sg_raw if not np.isnan(sg_raw).any() else None
            
        except Exception as e:
            print(f"Error processing {sound_file}: {e}")
            return None

    def save_spectrogram(self, sg_raw, output_folder, filename):
        """Save spectrogram as numpy array."""
        np.save(os.path.join(output_folder, f"{filename}.npy"), sg_raw)

    def save_example_image(self, sg_raw, output_folder, filename):
        """Save example spectrogram as image."""
        examples_folder = os.path.join(output_folder, "examples")
        os.makedirs(examples_folder, exist_ok=True)
        plt.imshow(sg_raw, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(examples_folder, f"{filename}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    @staticmethod
    def get_default_spec_params():
        """Get default spectrogram parameters."""
        return {
            'windowType': 'Hann', 
            'sgType': 'Standard', 
            'sgScale': 'Linear',
            'mean_normalise': False, 
            'equal_loudness': False
        }