#24/08/2021
# Author: Virginia Listanti
#Script for Instantaneous frequency estimation

#WHAT IS NEEDED TO CALL ECURVE?
#INPUT:
#      - TFR: matrix with spectrogram. Note: #columns=time, #rows=freq
#      - freqarray: array with discretized frequencies
#      - wopt: parameters needed byt the function. At the momenth this is fixed but need review

#TO Do: Define wopt (this requires REVIEW)


import SignalProc
import IF as IFreq
import numpy as np
#sfrom scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import os
from scipy import optimize
import scipy.special as spec

def int_function(x, L):
  if x<-1/L:
      y1,_=spec.sici(-np.pi*L*x)
      y2, _ = spec.sici(-(np.pi * L * x-np.pi))
      y3, _ = spec.sici(-(np.pi * L * x+np.pi))
      y=1/2 -(1/(2*np.pi))*y1-(1/(4*np.pi))*y2-(1/(4*np.pi))*y3
  elif  x>=-1/L and x<0:
      y1, _ = spec.sici(-np.pi * L * x)
      y2, _ = spec.sici(-(np.pi * L * x - np.pi))
      y3, _ = spec.sici(np.pi * L * x + np.pi)
      y = 1 / 2 - (1 / (2 * np.pi)) * y1 - (1 / (4 * np.pi)) * y2 + (1 / (4 * np.pi)) * y3
  elif  x>=0 and x<1/L:
      y1, _ = spec.sici(np.pi * L * x)
      y2, _ = spec.sici(-(np.pi * L * x - np.pi))
      y3, _ = spec.sici(np.pi * L * x + np.pi)
      y = 1 / 2 + (1 / (2 * np.pi)) * y1 - (1 / (4 * np.pi)) * y2 + (1 / (4 * np.pi)) * y3
  else:
      y1, _ = spec.sici(np.pi * L * x)
      y2, _ = spec.sici(np.pi * L * x - np.pi)
      y3, _ = spec.sici(np.pi * L * x + np.pi)
      y = y = 1 / 2 + (1 / (2 * np.pi)) * y1 + (1 / (4 * np.pi)) * y2 + (1 / (4 * np.pi)) * y3

  return y

window_width=1024
incr=256
window="Hann"


reassignment=False
sp=SignalProc.SignalProc(window_width,incr)
main_dir="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction"
test_fold="1024_256_Test2"

for f in os.listdir(main_dir):
    if f.endswith('.wav'):
        file_name = f
        data_file = main_dir + "\\" + file_name
        print(file_name)
        if 'song' in data_file:
            song_flag = True
        else:
            song_flag=False
        sp.readWav(data_file)
        fs = sp.sampleRate
        IF = IFreq.IF(method=1)
        #FOR JULIUS: check if window is Hann
        if reassignment:
            TFR = sp.spectrogram(window_width, incr, window, sgType='Reassigned')
        else:
            TFR=sp.spectrogram(window_width,incr,window)

        # REMEMBER: we need to transpose
        TFR=TFR.T
        #savemat'C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\test_signal.mat',{'TFR':TFR})
        fstep=(fs/2)/np.shape(TFR)[0]
        freqarr=np.arange(fstep,fs/2+fstep,fstep)


        wopt=[fs,window_width] #this neeeds review
        tfsupp,ecinfo, Skel=IF.ecurve(TFR,freqarr,wopt) # <= This is the function we need

        ########################## update wopt and wp

        wp=IFreq.Wp(incr,fs)
        wopt=IFreq.Wopt(fs,wp,0,fs/2)

        # function to reconstruct official Instantaneous Frequency
        #NOTE: at the moment this seems to not be needed
        iamp,iphi,ifreq = IF.rectfr(tfsupp,TFR,freqarr,wopt)

        fig_name=main_dir+"\\"+test_fold+"\\"+file_name[:-3]+"jpg"

        if song_flag:
            plt.rcParams["figure.autolayout"] = True
            fig, ax = plt.subplots(3, 1, figsize=(28,21), sharex=True)
            ax[0].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
            x = np.array(range(np.shape(TFR)[1]))
            ax[0].plot(x, (ifreq / fstep).T, linewidth=1, color='red')
            ax[0].plot(x, tfsupp[0, :] / fstep, linewidth=1, color='w')
            ax[1].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
            ax[2].plot(ifreq,color='red')
            ax[2].plot(tfsupp[0,:], color='green')
        else:
            plt.rcParams["figure.autolayout"] = True
            fig, ax = plt.subplots(1, 3, figsize=(28, 11.5), sharex=True)
            ax[0].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
            x = np.array(range(np.shape(TFR)[1]))
            ax[0].plot(x, (ifreq / fstep).T, linewidth=1, color='red')
            ax[0].plot(x, tfsupp[0, :] / fstep, linewidth=1, color='w')
            ax[1].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
            ax[2].plot(ifreq, color='red')
            ax[2].plot(tfsupp[0, :], color='green')
        plt.savefig(fig_name)
