"""
20/9/2022
Author: Virginia Listanti

This script creates images for Chapter 2 of my thesis
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import SignalProc
import wavio
from matplotlib.ticker import LinearLocator

def instant_freq(times, duration, f0, f1):
    """
    Instantaneous frequency function for hyperbolic chirp
    """
    # n = len(times)
    ifreq = (f0*f1*duration)/((f0-f1)*times + f1*duration)
    return ifreq

def instant_phase(times, duration, f0, f1):
    """
    Instantaneous phase function for hyperbolic chirp
    """
    # n = len(times)
    iphase = 2 * np.pi * ((f0*f1*duration)/(f0-f1)) * np.log(1 - ((f1-f0)/(f1*duration)) * times)
    return iphase

save_dir = "C:\\Users\\Virginia\\Documents\\Work\\Thesis_images\\Chapter2"

T=5 #signal duration
A=1 #amplitute
phi=0 #initial phase, usually 0
samplerate = 16000
t = np.linspace(0., T, samplerate*T,endpoint=False) #discretised time vector
freq0 = 1000
freq1 = 6000
s1=A*np.sin(phi+instant_phase(t, T, freq0, freq1))
s0=np.sin(phi+instant_phase(t, T, freq0, freq1))
# np.random.seed(19680801)
# At = np.random.uniform(0.1, 2, len(s0))
At = 0.0001 +t
s2 = At * s0
insta_freq = instant_freq(t, T, freq0, freq1)
save_path =save_dir+"\\hyperbolic_chirp.wav"
wavio.write(save_path, s1, samplerate, sampwidth=2)
save_path2 =save_dir+"\\hyperbolic_chirp_At.wav"
wavio.write(save_path2, s2, samplerate, sampwidth=2)

win_len = 512
hop = 128
window_type = 'Hann'
spec_type = "Standard"
scale = 'Linear'
mel_num = None
norm_type = "Standard"

sp = SignalProc.SignalProc(win_len, hop)
sp.readWav(save_path)
signal = sp.data
fs = sp.sampleRate
T = sp.fileLength / fs

TFR = sp.spectrogram(win_len, hop, window_type, sgType=spec_type, sgScale=scale, nfilters=mel_num)
sp.readWav(save_path2)
signal = sp.data
fs = sp.sampleRate
T = sp.fileLength / fs
TFR_3 = sp.spectrogram(win_len, hop, window_type, sgType=spec_type, sgScale=scale, nfilters=mel_num)
#
# # plot1: signal
# figname1 = save_dir + "\\chirp_signal_constant_A.jpg"
# plt.plot(t, s1)
# plt.xlim(0,0.05)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude')
# plt.savefig(figname1)
# plt.close

# # plot2: instantaneous frequency
# figname2 = save_dir + "\\chirp_signal_instantaneous_frequency.jpg"
# plt.plot(t, insta_freq)
# plt.ylim(0,8000)
# plt.xlabel('Time (seconds)')
# plt.ylabel(' Frequency (Hz)')
# plt.savefig(figname2)

# # plot 3: spectrogram
# TFR2 = TFR.T
# list_freq_labels = ['0','500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000', '5500', '6000',
#                    '6500', '7000', '7500', '8000']
# list_time_labels = ['0', '1', '2', '3', '4', '5']
# figname3 = save_dir + '\\chirp_spectrogram.jpg'
# # fig, ax = plt.subplots(1, 4, figsize=(10, 20), sharex=True)
# fig, ax = plt.subplots(1, 1)
# im = ax.imshow(np.flipud(TFR2), extent=[0, np.shape(TFR2)[1], 0, np.shape(TFR2)[0]], aspect='auto')
# ax.yaxis.set_major_locator(LinearLocator(17))
# ax.set_yticklabels(list_freq_labels)
# ax.xaxis.set_major_locator(LinearLocator(6))
# ax.set_xticklabels(list_time_labels)
# # ax.set_ylim(0, 8000)
# ax.set_xlabel('Time (seconds)')
# ax.set_ylabel('Frequency (Hz)')
# fig.colorbar(im, ax = ax)
# plt.savefig(figname3)

# # plot1 bis: signal
# figname1 = save_dir + "\\chirp_signal_constant_A_zoom.jpg"
# plt.plot(t, s1)
# plt.xlim(0,0.01)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude')
# plt.savefig(figname1)

# plot4: signal
figname1 = save_dir + "\\chirp_signal_At.jpg"
plt.plot(t, s2)
plt.xlim(0,0.1)
plt.ylim(-1,1)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.savefig(figname1)

# # plot 3: spectrogram s2
# TFR4 = TFR_3.T
# list_freq_labels = ['0','500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', '4500', '5000', '5500', '6000',
#                    '6500', '7000', '7500', '8000']
# list_time_labels = ['0', '1', '2', '3', '4', '5']
# figname3 = save_dir + '\\chirp_spectrogram_At.jpg'
# # fig, ax = plt.subplots(1, 4, figsize=(10, 20), sharex=True)
# fig, ax = plt.subplots(1, 1)
# im = ax.imshow(np.flipud(TFR4), extent=[0, np.shape(TFR4)[1], 0, np.shape(TFR4)[0]], aspect='auto')
# ax.yaxis.set_major_locator(LinearLocator(17))
# ax.set_yticklabels(list_freq_labels)
# ax.xaxis.set_major_locator(LinearLocator(6))
# ax.set_xticklabels(list_time_labels)
# # ax.set_ylim(0, 8000)
# ax.set_xlabel('Time (seconds)')
# ax.set_ylabel('Frequency (Hz)')
# fig.colorbar(im, ax = ax)
# plt.savefig(figname3)
