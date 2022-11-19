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

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset1\\Original"
directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
            "Smaller_Dataset2\\Original"
# save_dir = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#            "Smaller_Dataset1\\Classes_curves"
# save_dir = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#            "Smaller_Dataset2\\Original\\Classes_curves"
save_dir = "C:\\Users\\Virginia\\Documents\\Work\\Thesis_images\\Chapter 5\\Classes_curves"

labels_lists = ["D", "E", "J", "K", "L", "M", "O", "R", "Z"]
# labels_lists = ["D"]
col_list = [7, 4, 5, 4, 6, 4, 4, 2, 4]

win_len = 512
hop = 256
window_type = 'Hann'
spec_type = "Standard"
scale = 'Linear'
mel_num = None
norm_type = "Standard"
# list_freq_labels = ['0','500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000', '5500', '6000',
#                    '6500', '7000', '7500', '8000']
list_freq_labels = ['0','500', '1000', '1500', '2000', '2500', '3000', '3500', '4000']

for label in labels_lists:
    col_max = int(col_list[labels_lists.index(label)])
    # fig, ax = plt.subplots(2, 5, figsize=(300, 60), sharex = True, sharey = True)
    fig, ax = plt.subplots(2, col_max +1, figsize=(15, 8))
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
                TFR2 = TFR2[:int(np.floor(np.shape(TFR2)[0]/2)), :]
                ax[i, j].set_title(file[:-4], fontsize=16)
                ax[i, j].imshow(np.flipud(TFR2), extent=[0, np.shape(TFR2)[1], 0, np.shape(TFR2)[0]], aspect='auto')
                ax[i, j].yaxis.set_major_locator(LinearLocator(9))
                # ax[i,j].yaxis.set_ticks(np.arange(17))
                ax[i, j].set_yticklabels(list_freq_labels, size = 8)
                ax[i, j].xaxis.set_major_locator(LinearLocator(5))
                time_stamps = [0, T/4, T/2, T*3/4, T]
                time_stamps = np.round(time_stamps, decimals=2)
                list_time_labels = [str(time_stamps[0]), str(time_stamps[1]), str(time_stamps[2]), str(time_stamps[3]),
                                    str(time_stamps[4])]
                ax[i, j].set_xticklabels(list_time_labels, size = 7)
                ax[i, j].set_xlabel('Time (seconds)', size = 10)
                ax[i, j].set_ylabel('Frequency (Hz)', size = 10)
                # fig.colorbar(im, ax = ax)
                if j == col_max:
                    j = 0
                    i += 1
                else:
                    j += 1
    fig.suptitle('Class ' + label, fontsize=30)
    # fig.tight_layout(pad=20)
    fig.tight_layout()
    fig_name = save_dir + "\\spectrograms_"+label+"_class_curves.jpg"
    plt.savefig(fig_name, dpi= 200)