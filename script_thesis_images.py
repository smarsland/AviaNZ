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
insta_freq = instant_freq(t, T, freq0, freq1)
save_path =save_dir+"\\hyperbolic_chirp.wav"
wavio.write(save_path, s1, samplerate, sampwidth=2)

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

# # plot1: signal
# figname1 = save_dir + "\\chirp_signal_constant_A.jpg"
# plt.plot(t, s1)
# plt.xlim(0,0.05)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude')
# plt.savefig(figname1)

# # plot2: instantaneous frequency
# figname2 = save_dir + "\\chirp_signal_instantaneous_frequency.jpg"
# plt.plot(t, insta_freq)
# plt.ylim(0,8000)
# plt.xlabel('Time (seconds)')
# plt.ylabel(' Frequency (Hz)')
# plt.savefig(figname2)
