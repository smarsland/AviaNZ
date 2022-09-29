"""
Date: 30/09/2022
Author: Virginia Listanti

This script prepare the dataset for blind manual classification.

- Reads files name from directory
- shuffles them
- for each one:
               - read .wav file
               - create spectrogram and saves it in new directory
               - saves padded signal
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import SignalProc
import random
from matplotlib.ticker import LinearLocator
import wavio
import csv

old_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
                "Smaller_Dataset1\\Original"
new_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
                "Manual_classification"

#read files

list_syllables =[]
for file in os.listdir(old_directory):
    if file.endswith('.wav'):
        list_syllables.append(file)

# randomise
random.shuffle(list_syllables)

win_len = 512
hop = 128
window_type = 'Hann'
spec_type = "Standard"
scale = 'Linear'
mel_num = None
norm_type = "Standard"
list_freq_labels = ['0','500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000', '5500', '6000',
                   '6500', '7000', '7500', '8000']

# for each one save spectrogram and padded signal
counter = 0
new_list_syllables = []
for syllable in list_syllables:
    file_path = (old_directory + '\\' + syllable)
    sp = SignalProc.SignalProc(win_len, hop)
    sp.readWav(file_path)
    signal = sp.data
    fs = sp.sampleRate
    T = sp.fileLength / fs
    TFR = sp.spectrogram(win_len, hop, window_type, sgType=spec_type, sgScale=scale, nfilters=mel_num)
    TFR2 = TFR.T
    #save spectrogram
    if counter <10:
        Syl_name = 'Syllable_0'+str(counter)
    else:
        Syl_name = 'Syllable_'+str(counter)
    new_list_syllables.append(Syl_name)
    figname = new_directory + '\\'+ Syl_name + '.jpg'
    fig, ax = plt.subplots(1, 1, figsize=(10, 40))
    im = ax.imshow(np.flipud(TFR2), extent=[0, np.shape(TFR2)[1], 0, np.shape(TFR2)[0]], aspect='auto')
    ax.yaxis.set_major_locator(LinearLocator(17))
    ax.set_yticklabels(list_freq_labels, size=20)
    ax.set_xlabel('Time (seconds)', size = 30)
    ax.set_ylabel('Frequency (Hz)', size = 30)
    fig.colorbar(im, ax = ax)
    fig.suptitle(Syl_name, size = 30)

    plt.savefig(figname)
    plt.close(fig)

    # padded signals
    s2 = np.concatenate((np.zeros(int(2.5 * fs)), signal, np.zeros(int(2.5 * fs))))
    file_name2 = new_directory+ '\\' + Syl_name +'.wav'
    wavio.write(file_name2, s2, fs, sampwidth=2)


    counter += 1


# save ground truth
fieldnames = ["New name", "Old Name"]
csv_path = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
           "Manual_classification\\Ground_truth\\GT_file.csv"
with open(csv_path, 'w', newline='') as csvfile:
    Writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    Writer.writeheader()

for i in range(len(new_list_syllables)):
    with open(csv_path, 'a', newline='') as csvfile:  # should be ok
        Writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # Writer.writerow({"Label": label_id, "Syllable id": file[:-4]})
        Writer.writerow({"New name": new_list_syllables[i], "Old Name": list_syllables[i][:-4]})

