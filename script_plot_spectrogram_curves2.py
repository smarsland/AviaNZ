"""
28/09/22
Updated 21/11/2022
Author: Virginia Listanti

This script plot the spectrogram of a selected number of syllables
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import SignalProc
import IF as IFreq
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

syllable_lists = ["D5", "D12", "E1", "E9", "J1", "J9", "K0", "K5", "L1", "L6", "M0", "M3", "O0", "O8", "R1", "R2",
                  "Z5", "Z6"]
# labels_lists = ["D"]

alpha = 1
beta = 0.5
win_len = 512
hop = 256
window_type = 'Hann'
spec_type = "Standard"
scale = 'Linear'
mel_num = None
norm_type = "Standard"

# list_freq_labels = ['0','500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000', '5500', '6000',
#                    '6500', '7000', '7500', '8000']
list_freq_labels = ['0', '1000', '2000', '3000', '4000', '5000', '6000']

i = 0
j = 0
fig, ax = plt.subplots(3, 6, figsize=(15, 8))
for syllable in syllable_lists:
    IF = IFreq.IF(method=2, pars=[alpha, beta])
    # col_max = int(col_list[syllables_lists.index(label)])
    # fig, ax = plt.subplots(2, 5, figsize=(300, 60), sharex = True, sharey = True)
    file = syllable + '.wav'
    print(file)
    file_path = (directory + '\\' +file)
    sp = SignalProc.SignalProc(win_len, hop)
    sp.readWav(file_path)
    signal = sp.data
    fs = sp.sampleRate
    T = sp.fileLength / fs
    TFR = sp.spectrogram(win_len, hop, window_type, sgType=spec_type, sgScale=scale, nfilters=mel_num)
    TFR2 = TFR.T
    t_step = T / np.shape(TFR2)[1]

    # appropriate frequency scale
    if scale == "Linear":
        fstep = (fs / 2) / np.shape(TFR2)[0]
        freqarr = np.arange(fstep, fs / 2 + fstep, fstep)
    else:
        # #mel freq axis
        nfilters = mel_num
        freqarr = np.linspace(sp.convertHztoMel(0), sp.convertHztoMel(fs / 2), nfilters + 1)
        freqarr = freqarr[1:]

    # bandpass spectrogram below 800 Hz
    f_band = 800
    band_index = int(np.ceil(f_band / fstep - 1))
    TFR2[0:band_index, :] = 0

    wopt = [fs, win_len]
    tfsupp, _, _ = IF.ecurve(TFR2, freqarr, wopt)
    TFR3 = np.copy(TFR2)

    # hardcoded check
    f_jumps = np.zeros((len(tfsupp[0, :], )))
    for k in range(len(tfsupp[0, :]) - 2):
        f_jumps[k] = tfsupp[0, k + 2] - tfsupp[0, k]
    freq_jump_boundary = 700
    # freq_jump_boundary = 500
    if np.amax(np.abs(f_jumps)) > freq_jump_boundary:
        del IF
        print("detected jump: correcting")
        IF = IFreq.IF(method=2, pars=[alpha, beta])
        jump_index = np.argmax(np.abs(f_jumps))
        print(jump_index)
        if f_jumps[jump_index] > 0:
            # if we are doing a step up we will focus on the first half
            f_min = np.amin(tfsupp[1, 0:jump_index + 1])
            f_max = np.amax(tfsupp[2, 0:jump_index + 1])
        else:
            f_min = np.amin(tfsupp[1, jump_index + 1:])
            f_max = np.amax(tfsupp[2, jump_index + 1:])
        min_index = int(np.floor(f_min / fstep - 1))
        max_index = int(np.ceil(f_max / fstep - 1))
        TFR3[0:min_index] = 0
        TFR3[max_index + 1:] = 0
        tfsupp2, _, _ = IF.ecurve(TFR3, freqarr, wopt)
        if_syllable = np.copy(tfsupp2[0, :])
        low_bound = np.copy(tfsupp2[1, :])
        high_bound = np.copy(tfsupp2[2, :])
    else:
        if_syllable = np.copy(tfsupp[0, :])
        low_bound = np.copy(tfsupp[1, :])
        high_bound = np.copy(tfsupp[2, :])

    # revert to Hz if Mel
    if scale == 'Mel Frequency':
        if_syllable = sp.convertMeltoHz(if_syllable)
        low_bound = sp.convertMeltoHz(low_bound)
        high_bound = sp.convertMeltoHz(high_bound)

    # START AND ENDING POLISHING
    # find spec_intensity array
    spec_intensity = np.zeros(np.shape(if_syllable)[0])
    for t_index in range(len(if_syllable)):
        f_index = int(np.ceil(if_syllable[t_index] / fstep - 1))
        spec_intensity[t_index] = TFR3[f_index, t_index]

    # borders check
    T2 = np.copy(T)
    if_syllable_copy = np.copy(if_syllable)
    threshold = np.amax(spec_intensity) * (5 / 100)
    start_index = int(np.floor(len(if_syllable) / 4))  # first index safe from check
    last_index = int(np.floor(len(if_syllable) * (3 / 4)))  # last index safe from check

    # check start syllable
    check_index = 0
    while (check_index < start_index) and (spec_intensity[check_index] < threshold):
        if_syllable[check_index] = np.nan  # update syllable
        T2 -= t_step  # update time_lenght
        check_index += 1

    # check end syllable
    # find first index where to start deleting
    del_index = len(if_syllable_copy) - 1
    for check_index2 in range(len(if_syllable_copy) - 1, last_index - 1, -1):
        if spec_intensity[check_index2] < threshold:
            del_index = check_index2
        else:
            break

    for canc_index in range(del_index, len(if_syllable_copy)):
        if_syllable[canc_index] = np.nan  # flag_index
        T2 -= t_step  # update time_lenght

    # update syllable
    index_list = np.argwhere(np.isnan(if_syllable))
    if_syllable = np.delete(if_syllable, index_list)
    TFR2 = np.delete(TFR2, index_list, 1)

    # array with temporal coordinates
    t_support = np.linspace(0, T2, np.shape(if_syllable)[0])
    t_support2 = np.linspace(0, T, np.shape(if_syllable_copy)[0])

    TFR2 = TFR2[:int(np.floor(np.shape(TFR2)[0]*3/4)), :]
    ax[i, j].set_title(file[:-4], fontsize=16)
    ax[i, j].imshow(np.flipud(TFR2), extent=[0, np.shape(TFR2)[1], 0, np.shape(TFR2)[0]], aspect='auto')
    x = np.array(range(np.shape(TFR2)[1]))
    ax[i, j].plot(x, if_syllable/ (fstep), linewidth=1, color='yellow', linestyle='-')
    ax[i, j].yaxis.set_major_locator(LinearLocator(7))
    # ax[i,j].yaxis.set_ticks(np.arange(17))
    ax[i, j].set_yticklabels(list_freq_labels, size = 8)
    ax[i, j].xaxis.set_major_locator(LinearLocator(5))
    time_stamps = [0, T2/4, T2/2, T2*3/4, T2]
    time_stamps = np.round(time_stamps, decimals=2)
    list_time_labels = [str(time_stamps[0]), str(time_stamps[1]), str(time_stamps[2]), str(time_stamps[3]),
                        str(time_stamps[4])]
    ax[i, j].set_xticklabels(list_time_labels, size = 7)
    ax[i, j].set_xlabel('Time (seconds)', size = 10)
    ax[i, j].set_ylabel('Frequency (Hz)', size = 10)
    # fig.colorbar(im, ax = ax)
    if j == 5:
        j = 0
        i += 1
    else:
        j += 1

    del IF
# fig.suptitle('Class ' + label, fontsize=30)
# fig.tight_layout(pad=20)
fig.tight_layout()
fig_name = save_dir + "\\spectrograms_class_curves.jpg"
plt.savefig(fig_name, dpi= 200)