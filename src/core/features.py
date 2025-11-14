
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

# Code to extract various features from sound data

import numpy as np
import librosa
from src.core import spectrogram
from src.core import wavelet_segment
from src.core import audio_data


class Features:
    """
    This class implements feature extraction algorithms for the AviaNZ interface.
    Given a segment as a region of audiodata (between start and stop points).
    
    Currently used for:
    - Raven-compatible spectrogram measurements
    - MFCC extraction
    - Wavelet energy extraction
    - Sound Analysis Pro features
    """

    def __init__(self, data=[], sampleRate=0, window_width=256, incr=128):
        self.data = data
        self.sampleRate = sampleRate
        self.window_width = window_width
        self.incr = incr
        sp = spectrogram.Spectrogram(window_width=self.window_width, incr=self.incr)
        sp.audio_data = audio_data.AudioData(data=self.data, sample_rate=self.sampleRate, 
                                   sample_format='float32', sample_size=32, channels=1)
        self.sg = sp.spectrogram(sgType='Standard', window_width=self.window_width, incr=self.incr, window='Ones')
        self.sg = self.sg**2

    def get_WE(self, nlevels=5):
        """ Wavelet energies """
        ws = wavelet_segment.WaveletSegment(spInfo=[])
        WE = ws.computeWaveletEnergy(data=self.data, sampleRate=self.sampleRate, nlevels=nlevels, wpmode='new')
        return WE

    def get_Raven_spectrogram_measurements(self, f1, f2):
        """
        The first set of Raven features.
        energy, aggregate+average entropy, average power, delta power, max+peak freq, max+peak power

        The function is given a spectrogram and frequency indices (f1, f2) in pixels.
        
        Returns: avgPower, deltaPower, energy, aggEntropy, avgEntropy, maxPower, maxFreq
        """
        energy = np.sum(self.sg[:, f1:f2]) * self.sampleRate / self.window_width

        Ebin = np.sum(self.sg[:, f1:f2], axis=0)
        Ebin /= np.sum(Ebin)
        aggEntropy = np.sum(-Ebin * np.log2(Ebin))

        newsg = (self.sg.T / np.sum(self.sg, axis=1)).T
        avgEntropy = np.sum(-newsg * np.log2(newsg), axis=1)
        avgEntropy = np.mean(avgEntropy)

        sg = np.abs(np.where(self.sg == 0, 0.0, 10.0 * np.log10(self.sg)))

        avgPower = np.sum(sg[:, f1:f2]) / ((f2 - f1) * (np.shape(sg)[0]))

        deltaPower = (np.sum(sg[:, f2 - 1]) - np.sum(sg[:, f1])) / np.shape(sg)[1]

        maxPower = np.max(sg[:, f1:f2])

        maxFreq = (np.unravel_index(np.argmax(sg[:, f1:f2]), np.shape(sg[:, f1:f2]))[1] + f1) * self.sampleRate / 2 / np.shape(sg)[1]
        
        return avgPower, deltaPower, energy, aggEntropy, avgEntropy, maxPower, maxFreq

    def get_SAP_features(self, data, fs, window_width=256, incr=128, K=2):
        """
        Compute the Sound Analysis Pro features, i.e., Wiener entropy, spectral derivative, and their variants.
        Most of the code is in Spectrogram.py
        """
        sp = spectrogram.Spectrogram(sampleRate=fs, window_width=256, incr=128)
    
        spectral_deriv, sg, freq_mod, wiener_entropy, mean_freq, contours = sp.spectral_derivative(
            data, fs, window_width=window_width, incr=incr, K=2, threshold=0.5, returnAll=True
        )
    
        goodness_of_pitch = sp.goodness_of_pitch(spectral_deriv, sg)
    
        return spectral_deriv, goodness_of_pitch, freq_mod, contours, wiener_entropy, mean_freq
