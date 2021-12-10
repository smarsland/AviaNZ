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

fieldnames=["sp.data", "sp.filelenght", "torch tensor", "our tensor", "our spec", "torch spec1", "torch spec2", "inv sig", "torch inv1", "torch inv2"]
csvfilename="C:\\Users\\Virginia\\Documents\\GitHub\\Thesis\\Experiments\\torch_inv.csv"
with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
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
                sp = SignalProc.SignalProc(window_width, incr)

                sp.readWav(test_dir+"\\"+sig_dir+"\\"+test_id +'\\' + file)
                #wavform=wavio.read(test_dir+"\\"+sig_dir+"\\"+test_id +'\\' + file)

                sample_rate = sp.sampleRate
                TFR = sp.spectrogram(window_width, incr, window, sgType=sgType, sgScale=sgScale)
                s1_inverted = sp.invertSpectrogram(TFR, window_width=window_width, incr=incr, window=window)
                #
                # s1_inverted = s1_inverted[int(np.floor(window_width / 2 - (sp.fileLength - len(s1_inverted)))):-int(np.floor(window_width / 2))]
                # s1_inverted= s1_inverted / (np.ptp(s1_inverted) / np.ptp(sp.data))
                # save_file_path=test_dir+"\\"+sig_dir+"\\"+test_id +'\\' + file
                # save_file_path=save_file_path[:-4]+"_inv5.wav"
                # wavio.write(save_file_path, s1_inverted, samplerate, sampwidth=2)

                #torchaudio
                waveform, sample_rate = torchaudio.load(test_dir+"\\"+sig_dir+"\\"+test_id +'\\' + file)

                waveform2=torch.as_tensor(np.reshape(sp.data,(1,len(sp.data))))

                # define transformation
                spectrogram = T.Spectrogram(
                    n_fft=window_width,
                    win_length=window_width,
                    hop_length=incr,
                    )
                # Perform transformation
                spec = spectrogram(waveform)
                spec2= spectrogram(waveform2)
                griffin_lim = T.GriffinLim(
                    n_fft=window_width,
                    win_length=None,
                    hop_length=incr,
                )
                waveform_inverted = griffin_lim(spec)
                waveform_inverted2= griffin_lim(spec2)
                #waveform_inverted2 = griffin_lim(TFR)
                #plot_waveform(waveform_inverted, sample_rate)
                s2=waveform_inverted2.numpy()
                s2=np.reshape(s2,(np.shape(s2)[1],))
                save_file_path = test_dir + "\\" + sig_dir + "\\" + test_id + '\\' + file
                save_file_path=save_file_path[:-4]+"_inv6.wav"
                #torchaudio.save(save_file_path, waveform_inverted , sample_rate, bits_per_sample=16)
                with open(csvfilename, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({"sp.data":np.shape(sp.data), "sp.filelenght":sp.fileLength, "torch tensor":np.shape(waveform),
                                     "our tensor":np.shape(waveform2), "our spec":np.shape(TFR), "torch spec1":np.shape(spec),
                                     "torch spec2":np.shape(spec2), "inv sig":np.shape(s1_inverted), "torch inv1":np.shape(waveform_inverted),
                                     "torch inv2":np.shape(s2)})
                wavio.write(save_file_path, s1_inverted, samplerate, sampwidth=2)