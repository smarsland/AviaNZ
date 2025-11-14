# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2025

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

# Data augmentation functions for audio and spectrograms

import numpy as np

from src.core import spectrogram
from src.core import audio_data
from src.core import signal_proc


def add_noise(image, noise_image, noise_range=(0.2, 0.8)):
    """ Add random percentage of noise_image to image. """
    noise_factor = np.random.uniform(noise_range[0], noise_range[1])
    new_image = image + noise_image * noise_factor
    return new_image


def time_stretch(data, rate):
    """ Time stretch audio data by given rate (>1 speeds up, <1 slows down). """
    input_length = len(data)
    # wsola uses stretch factor (inverse of rate): larger values = longer/slower
    # rate: >1 speeds up (shorter), <1 slows down (longer)
    # so we pass 1/rate to wsola
    data = signal_proc.wsola(data, 1.0/rate)
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data


def change_speed_random(audiodata, fs, windowwidth, inc, mu=1.0, sigma=0.05):
    """ Change the speed of audio data randomly and generate spectrogram image. """
    s = np.random.normal(mu, sigma, 1000)
    rate = s[int(np.random.random() * 1000)]
    newdata = time_stretch(audiodata, rate)
    img = generate_spectrogram_image(newdata, fs, windowwidth, inc)
    return img


def generate_spectrogram_image(audiodata, fs, windowwidth, inc):
    """ Generate normalized and rotated spectrogram image from audio data. """
    sp = spectrogram.Spectrogram(windowwidth, inc)
    sp.audio_data = audio_data.AudioData(
        data=audiodata, 
        sample_rate=fs, 
        sample_format='float32', 
        sample_size=32, 
        channels=1
    )
    sgRaw = sp.spectrogram(windowwidth, inc)
    maxg = np.max(sgRaw)
    return np.rot90(sgRaw / maxg).tolist()


def augment_width_shift(images, num_needed, batch_size=32, shift_range=0.3):
    """ Apply horizontal shift augmentation to images. Mimics ImageDataGenerator width_shift. """
    augmented = []
    num_batches = int(np.ceil(num_needed / batch_size))
    
    for batch_idx in range(num_batches):
        batch = []
        for _ in range(batch_size):
            img = images[np.random.randint(0, len(images))]
            width = img.shape[1]
            shift = int(np.random.uniform(-shift_range, shift_range) * width)
            
            if shift > 0:
                shifted = np.pad(img, ((0, 0), (shift, 0), (0, 0)), mode='edge')[:, :width, :]
            elif shift < 0:
                shifted = np.pad(img, ((0, 0), (0, -shift), (0, 0)), mode='edge')[:, -width:, :]
            else:
                shifted = img
            
            batch.append(shifted)
        augmented.extend(batch)
    
    return np.array(augmented[:num_needed])


def generate_batch_noise(images, noise_pool, n, imageheight, imagewidth, noise_range=(0.2, 0.8)):
    """ Generate a batch of n new images by adding noise. """
    new_images = np.ndarray(shape=(n, imageheight, imagewidth, 1), dtype=float)
    for i in range(n):
        img = images[np.random.randint(0, len(images))]
        noise_img = noise_pool[np.random.randint(0, len(noise_pool))]
        new_images[i][:] = add_noise(img, noise_img, noise_range)
    return new_images


def generate_batch_noise_from_audio(audios, noise_pool, n, fs, length, windowwidth, inc, imageheight, imagewidth):
    """ Generate a batch of n new spectrogram images by adding audio noise. """
    nsamp = fs * length
    new_audios = np.ndarray(shape=(n, nsamp), dtype=float)
    for i in range(n):
        audio = audios[np.random.randint(0, len(audios))]
        noise = noise_pool[np.random.randint(0, len(noise_pool))]
        new_audios[i][:] = add_noise(audio, noise)
    
    new_images = np.ndarray(shape=(n, imageheight, imagewidth), dtype=float)
    for i in range(n):
        new_images[i][:] = generate_spectrogram_image(new_audios[i][:], fs, windowwidth, inc)
    
    return new_images.reshape(n, imageheight, imagewidth, 1)


def generate_batch_change_speed(audios, n, fs, windowwidth, inc, imageheight, imagewidth, mu=1.0, sigma=0.05):
    """ Generate a batch of n new spectrogram images by randomly changing audio speed. """
    new_images = np.ndarray(shape=(n, imageheight, imagewidth, 1), dtype=float)
    for i in range(n):
        audio = audios[np.random.randint(0, len(audios))]
        new_images[i][:] = change_speed_random(audio, fs, windowwidth, inc, mu, sigma)
    return new_images
