"""
20/05/2022
Author: Virginia Listanti

This script reads syllables .wav files in a folder and extract the dominant frequency
The fundamental frequency is then saved into a .csv file
We also generate a .png file for visualization
"""

import SignalProc
import IF as IFreq
import numpy as np
from numpy.linalg import norm
import os
import csv
import imed
import speechmetrics as sm
from geodesic_copy import geod_sphere
import ast
import matplotlib.pyplot as plt

#analysis_folder = "/home/listanvirg/Documents/Individual_identification/extracted/"
analysis_folder = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\extracted_test"
alpha = 1
beta = 0.5
win_len = 512
hop =128
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
        
        # #bandpass outside [min tfsupp[1,:], max tfsupp[2,:]]
        # # f_min = np.amin(tfsupp[1,:])
        # # f_min = np.median(tfsupp[1, :])
        # f_min = np.amin(tfsupp[0,:])-np.median(np.abs(tfsupp[0,:]-tfsupp[1, :]))
        # # f_min = np.percentile(tfsupp[0, :], 75) - np.median(np.abs(tfsupp[0, :] - tfsupp[1, :]))
        # min_index = int(np.floor(f_min / fstep - 1))
        # # f_max = np.amax(tfsupp[2,:])
        # # f_max = np.median(tfsupp[2, :])
        # f_max = np.amax(tfsupp[0, :]) + np.median(np.abs(tfsupp[0, :] - tfsupp[2, :]))
        # # f_max = np.percentile(tfsupp[0, :], 75) + np.median(np.abs(tfsupp[0, :] - tfsupp[2, :]))
        # max_index = int(np.ceil(f_max / fstep - 1))
        #
        # TFR3 = np.copy(TFR2)
        # TFR3[0:min_index] = 0
        # TFR3[max_index+1:] = 0
        #
        # wopt = [fs, win_len]  # this neeeds review
        # tfsupp, _, _ = IF.ecurve(TFR3, freqarr, wopt)

        TFR3 = np.copy(TFR2)
        # hardcoded check
        f_jumps = np.diff(tfsupp[0,:])
        freq_jump_boundary = 800
        if np.abs(np.amax(f_jumps))> freq_jump_boundary:
            del IF
            IF = IFreq.IF(method=2, pars=[alpha, beta])
            jump_index = np.argmax(f_jumps)
            f_min = np.amin(tfsupp[1, 0:jump_index])
            min_index = int(np.floor(f_min / fstep - 1))
            f_max = np.amax(tfsupp[2, 0:jump_index])
            max_index = int(np.ceil(f_max / fstep - 1))
            TFR3[0:min_index] = 0
            TFR3[max_index+1:] = 0
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

        # HARMONICS EXTRACTIONS?
        # TFR2 = np.array(TFR.copy())
        #
        # for n in range(int(np.floor(np.amin(tfsupp[0, :] / fstep))), int(np.ceil(np.amax(tfsupp[0, :] / fstep))) + 1):
        #     TFR2[n, :] = 0
        #
        # IF = IFreq.IF(method=2, pars=[1, 1])
        # tfsupp2, _, _ = IF.ecurve(TFR2, freqarr, wopt)


        # SAVE IF into .csv

        csvfilename = syllable_path[:-4] + "_IF.csv"
        fieldnames = ['t', "IF"]
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(if_syllable)):
                writer.writerow({"t": t_support[i], "IF": if_syllable[i]})
                # writer.writerow({"IF": if_syllable[i]})
        # plotting
        # change fig_name with the path you want
        # save picture
        fig_name = syllable_path[:-4] + "_Fourier_ridges.jpg"
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots(1, 3, figsize=(10, 20), sharex=True)
        ax[0].imshow(np.flipud(TFR2), extent=[0, np.shape(TFR2)[1], 0, np.shape(TFR2)[0]], aspect='auto')
        x = np.array(range(np.shape(TFR2)[1]))
        # ax[0].plot(x, (ifreq / fstep).T, linewidth=1, color='red')
        ax[0].plot(x, if_syllable / fstep, linewidth=2, color='r')
        ax[0].plot(x, low_bound / fstep, linewidth=2, color='w')
        ax[0].plot(x, high_bound / fstep, linewidth=2, color='w')
        ax[1].imshow(np.flipud(TFR3), extent=[0, np.shape(TFR3)[1], 0, np.shape(TFR3)[0]], aspect='auto')
        x = np.array(range(np.shape(TFR3)[1]))
        # ax[1].plot(x, tfsupp2[0, :] / fstep, linewidth=2, color='g')
        # ax[2].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
        # ax[2].plot(ifreq, color='red')
        # ax[2].plot(x,tfsupp[0, :], color='green')
        ax[2].plot(x, if_syllable, color='r')
        ax[2].set_ylim([0, fs / 2])
        # ax[3].plot(x, tfsupp2[0, :], color='g')
        # ax[2].plot(x,sp.convertBarktoHz(tfsupp[0, :]), color='green')

        fig.suptitle(file[:-4]+' ridges')
        plt.savefig(fig_name)






