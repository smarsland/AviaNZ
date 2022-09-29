"""
08/07/2022
Author: Virginia Listanti

This script reads syllables file in a folder and saves a padded variant
"""

import numpy as np
import os
import wavio
import SignalProc


# syllable_folder = "C:\\Users\\Virginia\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\exemplars\\Models"
# syllable_folder = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID" \
#                   "\\exemplars\\Smaller_Dataset\\Original"
syllable_folder = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
                  "Smaller_Dataset1\\Original"
win_len = 512
hop = 128
for file in os.listdir(syllable_folder):
    if file.endswith('.wav'):
        syllable_path = os.path.join(syllable_folder, file)
        sp = SignalProc.SignalProc(win_len, hop)
        sp.readWav(syllable_path)
        signal = sp.data
        fs = sp.sampleRate
        T = sp.fileLength / fs
        # save padded version
        s2 = np.concatenate((np.zeros(int(2.5 * fs)), signal, np.zeros(int(2.5 * fs))))
        file_name2 = syllable_path[:-4] + '_padded.wav'
        wavio.write(file_name2, s2, fs, sampwidth=2)