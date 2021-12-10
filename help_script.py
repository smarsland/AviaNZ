#29/10/2021
# Author: Virginia Listanti
#help script for TF tests
# For Harvey this is the script: see the comments

import SignalProc
import IF as IFreq
import numpy as np
#sfrom scipy.io import loadmat, savemat #this can be useful to make python talk with MATLAB
import matplotlib.pyplot as plt


test_name="Harmonics2" #change test name
#file_name -> path of the file you want to analise (note: somwntimes windows wants the double \\
#file_name="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\exponential_downchirp\\Test_02\\exponential_downchirp_0.wav"
file_name="C:\\Users\Virginia\\Documents\Work\\IF_extraction\\test_syllable.wav"
# file_name="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\pure_tone\\Test_02\\pure_tone_0_inv.wav"

#parameters (these are the parameters for a Fourier Transform)
window_width=1024
incr=512
window= "Hann"

#"callinf IF class
#"method=2 is the second method of the paper
IF = IFreq.IF(method=2, pars=[1, 1])

#calling signal proc -> see if it change if you want to use it with wavelets
sp = SignalProc.SignalProc(window_width, incr)
#you need to read the wav file with signal proc
sp.readWav(file_name)
fs = sp.sampleRate

#evaluate spectrogram
TFR = sp.spectrogram(window_width, incr, window,sgType='Standard',sgScale='Linear',equal_loudness=False,mean_normalise=True)
TFR = TFR.T
#you need to call sp.scalogram
print("Spectrogram Dim =", np.shape(TFR))

#Standard freq ax
#it is important to set freqarr
fstep = (fs / 2) / np.shape(TFR)[0]
freqarr =np.arange(fstep, fs / 2 + fstep, fstep)

# #mel freq axis
# nfilters=40
# freqarr = np.linspace(sp.convertHztoMel(0), sp.convertHztoMel(fs/2), nfilters + 1)
# freqarr=freqarr[1:]
# fstep=np.mean(np.diff(freqarr))

#bark freq axis
# nfilters=40
# freqarr = np.linspace(sp.convertHztoBark(0), sp.convertHztoBark(fs/2), nfilters + 1)
# freqarr=freqarr[1:]
# fstep=np.mean(np.diff(freqarr))

#setting parametes for ecurve
wopt = [fs, window_width]
#calling ecurve
tfsupp,_,_=IF.ecurve(TFR,freqarr,wopt)
del IF

TFR2=np.array(TFR.copy())

for n in range(int(np.floor(np.amin(tfsupp[0,:]/fstep))),int(np.ceil(np.amax(tfsupp[0,:]/fstep)))+1):
    TFR2[n,:]=0

IF = IFreq.IF(method=2, pars=[1, 1])
tfsupp2,_,_=IF.ecurve(TFR2,freqarr,wopt)

#plotting
# change fig_name with the path you want
#save picture
fig_name="C:\\Users\\Virginia\\Documents\GitHub\\Thesis\\Experiments\\TFR_test_plot"+"\\"+test_name+".jpg"
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots(1, 4, figsize=(10, 20), sharex=True)
ax[0].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
x = np.array(range(np.shape(TFR)[1]))
#ax[0].plot(x, (ifreq / fstep).T, linewidth=1, color='red')
ax[0].plot(x, tfsupp[0, :] / fstep, linewidth=2, color='r')
ax[1].imshow(np.flipud(TFR2), extent=[0, np.shape(TFR2)[1], 0, np.shape(TFR2)[0]], aspect='auto')
x = np.array(range(np.shape(TFR2)[1]))
ax[1].plot(x, tfsupp2[0, :] / fstep, linewidth=2, color='g')
ax[2].imshow(np.flipud(TFR2), extent=[0, np.shape(TFR2)[1], 0, np.shape(TFR2)[0]], aspect='auto')
#ax[2].plot(ifreq, color='red')
#ax[2].plot(x,tfsupp[0, :], color='green')
ax[3].plot(x,tfsupp[0, :], color='r')
ax[3].plot(x,tfsupp2[0, :], color='g')
# ax[2].plot(x,sp.convertBarktoHz(tfsupp[0, :]), color='green')
ax[3].set_ylim([0,fs/2])
plt.savefig(fig_name)
#plt.show()