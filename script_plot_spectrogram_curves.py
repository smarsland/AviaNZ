"""
28/09/22
Author: Virginia Listanti

This script plot the extracted IF in a subplot divided by classes
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import SignalProc
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as mticker

directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
            "Smaller_Dataset1\\Original"
save_dir = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
           "Smaller_Dataset1\\Classes_curves"

labels_lists = ["D", "E", "G", "K", "J", "L", "M", "O", "R", "Z"]
# labels_lists = ["D"]
win_len = 512
hop = 128
window_type = 'Hann'
spec_type = "Standard"
scale = 'Linear'
mel_num = None
norm_type = "Standard"
list_freq_labels = ['0','500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000', '5500', '6000',
                   '6500', '7000', '7500', '8000']

for label in labels_lists:
    # fig, ax = plt.subplots(2, 5, figsize=(300, 60), sharex = True, sharey = True)
    fig, ax = plt.subplots(2, 5, figsize=(300, 300))
    i = 0
    j = 0
    for file in os.listdir(directory):
        if file[0] == label:
            if file.endswith(".wav"):
                file_path = (directory + '\\' +file)
                sp = SignalProc.SignalProc(win_len, hop)
                sp.readWav(file_path)
                signal = sp.data
                fs = sp.sampleRate
                T = sp.fileLength / fs
                TFR = sp.spectrogram(win_len, hop, window_type, sgType=spec_type, sgScale=scale, nfilters=mel_num)
                TFR2 = TFR.T
                ax[i, j].set_title(file[:2], fontsize=320)
                ax[i, j].imshow(np.flipud(TFR2), extent=[0, np.shape(TFR2)[1], 0, np.shape(TFR2)[0]], aspect='auto')
                ax[i, j].yaxis.set_major_locator(LinearLocator(17))
                # ax[i,j].yaxis.set_ticks(np.arange(17))
                ax[i, j].set_yticklabels(list_freq_labels, size = 160)
                # ax[i, j].xaxis.set_major_locator(LinearLocator(6))
                # ax[i, j].set_xticklabels(list_time_labels)
                ax[i, j].set_xlabel('Time (seconds)', size = 200)
                ax[i, j].set_ylabel('Frequency (Hz)', size = 200)
                # fig.colorbar(im, ax = ax)
                if j == 4:
                    j = 0
                    i += 1
                else:
                    j += 1
    fig.suptitle('Class ' + label, fontsize=500)
    fig.tight_layout(pad=20)
    fig_name = save_dir + "\\spectrograms_"+label+"_class_curves.jpg"
    plt.savefig(fig_name)