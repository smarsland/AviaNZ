#29/10/2021
# Author: Virginia Listanti
#help script for TF tests

import SignalProc
import IF2 as IFreq
import numpy as np
#from scipy import io
import matplotlib.pyplot as plt

import WaveletFunctions
#import WaveletSegment

test_name="test5" #change test name
file_name="C:\\Users\\Harvey\\Documents\\GitHub\\AviaNZ\\Toy signals\\exponential_downchirp_0.wav"

# file_name="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\pure_tone\\Test_02\\pure_tone_0_inv.wav"

#parameters (these are the parameters for a Fourier Transform)
window = 0.25
inc = 0.256

#"calling IF class
#"method=2 is the second method of the paper
IF = IFreq.IF(method=2, pars=[0, 1])

#calling signal proc -> see if it change if you want to use it with wavelets
sp = SignalProc.SignalProc(window, inc)
#you need to read the wav file with signal proc
sp.readWav(file_name)
fs = sp.sampleRate


wf = WaveletFunctions.WaveletFunctions(sp.data,'dmey2',None,fs)
#ws = WaveletSegment.WaveletSegment()

# number of samples in window
win_sr = int(np.ceil(window * fs))
# number of sample in increment
inc_sr = int(np.ceil(inc * fs))
wf.maxLevel = 5
# output columns dimension equal to number of sliding window
N = int(np.ceil(len(sp.data) / inc_sr))
coefs = np.zeros((2 ** (wf.maxLevel + 1) - 2, N))
allnodes = range(2 ** (wf.maxLevel + 1) - 1)
wf.WaveletPacket(allnodes, mode='symmetric', antialias=True, antialiasFilter=True)


#evaluate scalogram
for node in allnodes:
    nodeE = wf.extractE(node, window, wpantialias=True)
    # the wavelet energies may in theory have one more or less windows than annots
    # b/c they adjust the window size to use integer number of WCs.
    # If they differ by <=1, we allow that and just equalize them:
    if N == len(nodeE) + 1:
        coefs[node - 1, :-1] = nodeE
        coefs[node - 1, -1] = currWCs[node - 1, -2]  # repeat last element
    elif N == len(nodeE) - 1:
        # drop last WC
        coefs[node - 1, :] = nodeE[:-1]
    elif np.abs(N - len(nodeE)) > 1:
        print("ERROR: lengths of annotations and energies differ:", N, len(nodeE))
    else:
        coefs[node - 1, :] = nodeE

TFR = coefs
#TFR = ws.computeWaveletEnergy(sp.data, fs, window=0.25, inc=0.25)
#TFR = np.log(TFR[30:62,:])

print("Scalogram Dim =", np.shape(TFR))
#TFR = TFR.T

#Standard freq ax
#it is important to set freqarr
fstep = (fs / 2) / np.shape(TFR)[0]
freqarr =np.arange(fstep, fs / 2 + fstep, fstep)

#setting parametes for ecurve
wopt = [fs, window]
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