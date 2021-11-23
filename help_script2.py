#29/10/2021
# Author: Virginia Listanti
#help script for TF tests

import SignalProc
import IF2 as IFreq
import numpy as np
from scipy import io
#sfrom scipy.io import loadmat, savemat #this can be useful to make python talk with MATLAB
import matplotlib.pyplot as plt

import WaveletFunctions
import WaveletSegment

test_name="test3" #change test name
file_name="C:\\Users\\Harvey\\Documents\\GitHub\\AviaNZ\\Toy signals\\exponential_downchirp_0.wav"

# file_name="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\pure_tone\\Test_02\\pure_tone_0_inv.wav"

#parameters (these are the parameters for a Fourier Transform)
window_width=1024
incr=32
window= "Hann"

#"calling IF class
#"method=2 is the second method of the paper
IF = IFreq.IF(method=2, pars=[0, 1])

#calling signal proc -> see if it change if you want to use it with wavelets
sp = SignalProc.SignalProc(window_width, incr)
#you need to read the wav file with signal proc
sp.readWav(file_name)
fs = sp.sampleRate

#wf = WaveletFunctions.WaveletFunctions(sp.data,'dmey2',None,fs)
ws = WaveletSegment.WaveletSegment()

#wf.maxLevel = 5
#allnodes = range(2 ** (wf.maxLevel + 1) - 1)
#wf.WaveletPacket(allnodes, 'symmetric')

#evaluate scalogram

TFR = ws.computeWaveletEnergy(sp.data, fs, window=0.25, inc=0.25)
TFR = np.log(TFR[30:62,:])

#io.savemat('export2.mat', {"sig": sp.data})
#io.savemat('export.mat', {"sig": TFR})

#you need to call sp.scalogram
print("Scalogram Dim =", np.shape(TFR))
#TFR = TFR.T

#Standard freq ax
#it is important to set freqarr
fstep = (fs / 2) / np.shape(TFR)[0]
freqarr =np.arange(fstep, fs / 2 + fstep, fstep)

#setting parametes for ecurve
wopt = [fs, window_width]
#calling ecurve
tfsupp,_,_=IF.ecurve(TFR,freqarr,wopt)

# plt.imshow(TFR)
# plt.show()
# change fig_name with the path you want
#save picture
fig_name="C:\\Users\\Harvey\\Desktop\\Uni\\avianz\\test plots"+"\\"+test_name+".jpg"
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots(1, 3, sharex=True)
ax[0].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
# ax[0].imshow(TFR, aspect='auto')
x = np.array(range(np.shape(TFR)[1]))
#ax[0].plot(x, (ifreq / fstep).T, linewidth=1, color='red')
ax[0].plot(x, tfsupp[0, :] / fstep, linewidth=1, color='r')
ax[1].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
#ax[2].plot(ifreq, color='red')
#ax[2].plot(x,tfsupp[0, :], color='green')
ax[2].plot(x,tfsupp[0, :], color='green')
ax[2].set_ylim([0,fs/2])
plt.savefig(fig_name)
#plt.show()