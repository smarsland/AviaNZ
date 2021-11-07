#29/10/2021
# Author: Virginia Listanti
#help script for TF tests

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

test_name="Standard_Mel"
file_name="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\exponential_downchirp\\Test_02\\exponential_downchirp_0.wav"

# file_name="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\pure_tone\\Test_02\\pure_tone_0_inv.wav"

#parameters
window_width=1024
incr=512
window= "Hann"
IF = IFreq.IF(method=2, pars=[1, 1])
sp = SignalProc.SignalProc(window_width, incr)
sp.readWav(file_name)
fs = sp.sampleRate

#evaluate spectrogram
TFR = sp.spectrogram(window_width, incr, window,sgType='Standard',sgScale='Mel Frequency',equal_loudness=False,mean_normalise=True)
TFR = TFR.T
print("Spectrogram Dim =", np.shape(TFR))

#Standard freq ax
# fstep = (fs / 2) / np.shape(TFR)[0]
# freqarr =sp.convertHztoMel(np.arange(fstep, fs / 2 + fstep, fstep))

#mel freq axis
nfilters=40
freqarr = np.linspace(sp.convertHztoMel(0), sp.convertHztoMel(fs/2), nfilters + 1)
freqarr=freqarr[1:]
fstep=np.mean(np.diff(freqarr))

#bark freq axis
# nfilters=40
# freqarr = np.linspace(sp.convertHztoBark(0), sp.convertHztoBark(fs/2), nfilters + 1)
# freqarr=freqarr[1:]
# fstep=np.mean(np.diff(freqarr))


wopt = [fs, window_width]
tfsupp,_,_=IF.ecurve(TFR,freqarr,wopt)

#save picture
fig_name="C:\\Users\\Virginia\\Documents\GitHub\\Thesis\\Experiments\\TFR_test_plot"+"\\"+test_name+".jpg"
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots(1, 3, figsize=(10, 20), sharex=True)
ax[0].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
x = np.array(range(np.shape(TFR)[1]))
#ax[0].plot(x, (ifreq / fstep).T, linewidth=1, color='red')
ax[0].plot(x, tfsupp[0, :] / fstep, linewidth=1, color='r')
ax[1].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
#ax[2].plot(ifreq, color='red')
#ax[2].plot(x,tfsupp[0, :], color='green')
ax[2].plot(x,sp.convertMeltoHz(tfsupp[0, :]), color='green')
# ax[2].plot(x,sp.convertBarktoHz(tfsupp[0, :]), color='green')
ax[2].set_ylim([0,fs/2])
plt.savefig(fig_name)
#plt.show()