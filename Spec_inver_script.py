#5/10/2021
# Author: Virginia Listanti
#Script for spectrogram Inversion

import SignalProc
import IF as IFreq
import numpy as np
from numpy.linalg import norm
#sfrom scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import os
from scipy import optimize
import scipy.special as spec
import wavio
import csv
from scipy.special import kl_div
# import pytorch
# from pytorch import torchaudio
import torch
import torchaudio
import torchaudio.transforms as T


Test_List=["Test_03", "Test_04"]
samplerate = 8000
# A= np.iinfo(np.int16).max
test_dir="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals"
window="Hann"

# for sig_dir in os.listdir(test_dir):
#     if sig_dir=="Test_guide.txt":
#         continue
#
#     if sig_dir.startswith('pure'):
#         window_width=2048
#         incr=512
#     else:
#         window_width=1024
#         incr=256
#     for test_id in os.listdir(test_dir+"\\"+sig_dir):
#         if test_id=="Dataset" or test_id=="Test_01"or test_id=="Test_04":
#             continue
#         if test_id=="Test_03":
#             sgType = 'Reassigned'
#             sgScale = 'Linear'
#         else:
#             sgType = 'Standard'
#             sgScale = 'Linear'
#
#         for file in os.listdir(test_dir+"\\"+sig_dir+"\\"+test_id):
#             if file.endswith('.wav'):
#                 sp = SignalProc.SignalProc(window_width, incr)
#
#                 sp.readWav(test_dir+"\\"+sig_dir+"\\"+test_id +'\\' + file)
#                 samplerate = sp.sampleRate
#                 TFR = sp.spectrogram(window_width, incr, window, sgType=sgType, sgScale=sgScale)
#                 s1_inverted = sp.invertSpectrogram(TFR, window_width=window_width, incr=incr, window=window)
#                 save_file_path=test_dir+"\\"+sig_dir+"\\"+test_id +'\\' + file
#                 save_file_path=save_file_path[:-4]+"_inv.wav"
#                 wavio.write(save_file_path, s1_inverted, samplerate, sampwidth=2)

for sig_dir in os.listdir(test_dir):
    if sig_dir=="Test_guide.txt":
        continue
    if sig_dir=="fake_kiwi_syllables":
        continue
    if sig_dir.startswith('pure'):
        window_width=2048
    else:
        window_width=1024
    incr=int(np.floor(window_width*0.05))
    for test_id in os.listdir(test_dir+"\\"+sig_dir):
        if test_id=="Dataset" or test_id=="Test_01"or test_id=="Test_04":
            continue
        if test_id=="Test_03":
            sgType = 'Reassigned'
            sgScale = 'Linear'
        else:
            sgType = 'Standard'
            sgScale = 'Linear'

        for file in os.listdir(test_dir+"\\"+sig_dir+"\\"+test_id):
            if 'inv' in file:
                continue
            if file.endswith('.wav'):
                # #AviaNZ Signal Proc
                # sp = SignalProc.SignalProc(window_width, incr)
                #
                # sp.readWav(test_dir+"\\"+sig_dir+"\\"+test_id +'\\' + file)
                # samplerate = sp.sampleRate
                # TFR = sp.spectrogram(window_width, incr, window, sgType=sgType, sgScale=sgScale)
                # #s1_inverted = sp.invertSpectrogram(TFR, window_width=window_width, incr=incr, window=window)
                #
                # s1_inverted = s1_inverted[int(np.floor(window_width / 2 - (sp.fileLength - len(s1_inverted)))):-int(np.floor(window_width / 2))]
                # s1_inverted= s1_inverted / (np.ptp(s1_inverted) / np.ptp(sp.data))
                # save_file_path=test_dir+"\\"+sig_dir+"\\"+test_id +'\\' + file
                # save_file_path=save_file_path[:-4]+"_inv5.wav"
                # wavio.write(save_file_path, s1_inverted, samplerate, sampwidth=2)

                #torchaudio
                waveform, sample_rate = torchaudio.load(test_dir+"\\"+sig_dir+"\\"+test_id +'\\' + file)


                # define transformation
                spectrogram = T.Spectrogram(
                    n_fft=window_width,
                    win_length=None,
                    hop_length=incr,
                    )
                # Perform transformation
                spec = spectrogram(waveform)

                griffin_lim = T.GriffinLim(
                    n_fft=window_width,
                    win_length=None,
                    hop_length=incr,
                )
                waveform_inverted = griffin_lim(spec)
                #plot_waveform(waveform_inverted, sample_rate)
                save_file_path = test_dir + "\\" + sig_dir + "\\" + test_id + '\\' + file
                save_file_path=save_file_path[:-4]+"_inv5.wav"
                torchaudio.save(save_file_path, waveform_inverted , sample_rate, bits_per_sample=16)