"""
20/05/2022
Author: Virginia Listanti

This script reads syllables .wav files in a folder and extract the dominant frequency
We do a check to prevent jumps between harmonics and in case bandpass the spectrogram and rerun the IF extraction
algorithm
The fundamental frequency is then saved into a .csv file
We also generate a .jpg file for visualization
"""

import SignalProc
import IF as IFreq
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

# CHECK: change working directory
#analysis_folder = "/home/listanvirg/Documents/Individual_identification/extracted/"
# analysis_folder = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                   "extracted"
# analysis_folder = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Test_jump"
analysis_folder = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\extracted_test"
alpha = 1
beta = 0.5
win_len = 512
hop = 128
window_type = 'Hann'
spec_type = "Standard"
scale = 'Linear'
mel_num = None
norm_type = "Standard"


for file in os.listdir(analysis_folder):
    if file.endswith('wav'):
        syllable_path = os.path.join(analysis_folder, file)
        IF = IFreq.IF(method=2, pars=[alpha, beta])
        sp = SignalProc.SignalProc(win_len, hop)
        sp.readWav(syllable_path)
        signal = sp.data
        fs = sp.sampleRate
        T = sp.fileLength / fs

        TFR = sp.spectrogram(win_len, hop, window_type, sgType=spec_type, sgScale=scale, nfilters=mel_num)
        # spectrogram normalizations
        if norm_type != "Standard":
            sp.normalisedSpec(tr=norm_type)
            TFR = sp.sg
        TFR2 = TFR.T

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
        band_index = int(np.ceil(f_band/fstep -1))
        TFR2[0:band_index,:] = 0


        wopt = [fs, win_len]  # this neeeds review
        tfsupp, _, _ = IF.ecurve(TFR2, freqarr, wopt)
        TFR3 = np.copy(TFR2)

        # # hardcoded check
        # f_jumps = np.diff(tfsupp[0,:])
        # # f_jumps = np.diff(tfsupp[0,:], n=2)
        # # freq_jump_boundary = 800
        # freq_jump_boundary = 700
        # if np.amax(np.abs(f_jumps)) > freq_jump_boundary:
        #     del IF
        #     print("detected jump: correcting")
        #     IF = IFreq.IF(method=2, pars=[alpha, beta])
        #     jump_index = np.argmax(np.abs(f_jumps))
        #     print(jump_index)
        #     f_min = np.amin(tfsupp[1, 0:jump_index])
        #     min_index = int(np.floor(f_min / fstep - 1))
        #     f_max = np.amax(tfsupp[2, 0:jump_index])
        #     max_index = int(np.ceil(f_max / fstep - 1))
        #     TFR3[0:min_index] = 0
        #     TFR3[max_index+1:] = 0
        #     tfsupp2, _, _ = IF.ecurve(TFR3, freqarr, wopt)
        #     if_syllable = np.copy(tfsupp2[0, :])
        #     low_bound = np.copy(tfsupp2[1, :])
        #     high_bound = np.copy(tfsupp2[2, :])
        # else:
        #     if_syllable = np.copy(tfsupp[0, :])
        #     low_bound = np.copy(tfsupp[1, :])
        #     high_bound = np.copy(tfsupp[2, :])

        # hardcoded check
        #f_jumps = np.diff(tfsupp[0, :], 2)
        f_jumps = np.zeros((len(tfsupp[0,:],)))
        for k in range (len(tfsupp[0,:])-2):
            f_jumps[k] = tfsupp[0, k+2] - tfsupp[0, k]
        freq_jump_boundary = 700
        if np.amax(np.abs(f_jumps)) > freq_jump_boundary:
            del IF
            print("detected jump: correcting")
            IF = IFreq.IF(method=2, pars=[alpha, beta])
            jump_index = np.argmax(np.abs(f_jumps))
            print(jump_index)
            if f_jumps[jump_index]>0:
                # if we are doing a step up we will focus on the first half
                f_min = np.amin(tfsupp[1, 0:jump_index+1])
                f_max = np.amax(tfsupp[2, 0:jump_index+1])
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

        # array with temporal coordinates
        t_support = np.linspace(0, T, np.shape(if_syllable)[0])

        # SAVE IF into .csv
        csvfilename = syllable_path[:-4] + "_IF.csv"
        fieldnames = ['t', "IF"]
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(if_syllable)):
                writer.writerow({"t": t_support[i], "IF": if_syllable[i]})

        # plotting
        # save picture
        fig_name = syllable_path[:-4] + "_Fourier_ridges_1.jpg"
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots(1, 3, figsize=(10, 20), sharex=True)
        ax[0].imshow(np.flipud(TFR2), extent=[0, np.shape(TFR2)[1], 0, np.shape(TFR2)[0]], aspect='auto')
        x = np.array(range(np.shape(TFR2)[1]))
        ax[0].plot(x, if_syllable / fstep, linewidth=2, color='r')
        ax[0].plot(x, low_bound / fstep, linewidth=2, color='w')
        ax[0].plot(x, high_bound / fstep, linewidth=2, color='w')
        ax[1].imshow(np.flipud(TFR3), extent=[0, np.shape(TFR3)[1], 0, np.shape(TFR3)[0]], aspect='auto')
        x = np.array(range(np.shape(TFR3)[1]))
        ax[2].plot(x, if_syllable, color='r')
        ax[2].set_ylim([0, fs / 2])
        fig.suptitle(file[:-4]+' ridges')
        plt.savefig(fig_name)

        del IF






