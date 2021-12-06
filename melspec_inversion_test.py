"""
06/12/2021
Author: Virginia Listanti

This is a script to test how to invert a Melscale spectrogram
"""

import SignalProc
import IF as IFreq
import numpy as np
import wavio
import librosa

file_name = "C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy " + \
            "signals\\pure_tone\\Base_Dataset_2\\pure_tone_00.wav"

window_width=1024
incr=512
window= "hann"
nfilters= 218

sp = SignalProc.SignalProc(window_width, incr)
sp.readWav(file_name)
fs = sp.sampleRate

#evaluate spectrogram
TFR = sp.spectrogram(window_width, incr, window, sgType = 'Standard',sgScale = 'Mel Frequency', nfilters = nfilters)
TFR = TFR.T
#you need to call sp.scalogram
print("Spectrogram Dim =", np.shape(TFR))

#signal_inverted = sp.invertSpectrogram(TFR, window_width=window_width, incr=incr, window=window)
signal_inverted = librosa.feature.inverse.mel_to_audio(TFR, hop_length=incr, win_length=window_width, window=window)

save_file_path=file_name[:-4]+"_inv.wav"
wavio.write(save_file_path, signal_inverted, fs, sampwidth=2)