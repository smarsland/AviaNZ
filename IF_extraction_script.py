#21/05/2021
# Author: Virginia Listanti
#Script for Instantaneous frequency exstration

#TO Do: Define wopt


import SignalProc 
import IF as IFreq
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import os


window_width=1024
incr=256
window="Hann"
reassignment=True
#sp=SignalProc.SignalProc(64,32)
sp=SignalProc.SignalProc(window_width,incr)


main_dir="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction"
test_fold="1024_256_Reassigned_1"

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
        IF = IFreq.IF()
        #check if window is Hann
        if reassignment:
            TFR = sp.spectrogram(window_width, incr, window, sgType='Reassigned')
        else:
            TFR=sp.spectrogram(window_width,incr,window)

        #transpose
        TFR=TFR.T
        #savemat'C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\test_signal.mat',{'TFR':TFR})
        #pl.imshow(TFR)(
        #pl.show()
        #print(np.shape(TFR))
        fstep=(fs/2)/np.shape(TFR)[0]
        freqarr=np.arange(fstep,fs/2+fstep,fstep)
        #print(freqarr)
        #print(np.shape(freqarr))
        wopt=[fs,1]
        tfsupp,ecinfo, Skel=IF.ecurve(TFR,freqarr,wopt)

        ########################## update wopt and wp

        wp=IFreq.Wp(incr,fs)
        wopt=IFreq.Wopt(fs,wp,0,fs/2)

        #official ifreq
        iamp,iphi,ifreq = IF.rectfr(tfsupp,TFR,freqarr,wopt)

        #print('ciao')
        #fig=plt.figure()

        #plt.plot(ifreq)
        #plt.plot(tfsupp[0,:],'r',linewidth=5)
        #plt.axis('off')
        #plt.ylim(0,fs/2)
        #pl.ylim(0,4000)
        #plt.imshow(np.flipud(TFR),zorder=0,extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]])
        #plt.plot(tfsupp[0,:],'r',linewidth=5, zorder=1, extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]])
        #plt.axis('off')
        #fig.savefig=("C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\aru_F1_TEST.png")

        fig_name=main_dir+"\\"+test_fold+"\\"+file_name[:-3]+"jpg"

        if song_flag:
            #plt.rcParams["figure.figsize"] = [14.00, 3.50]
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
            #plt.rcParams["figure.figsize"] = [3.50, 7.00]
            plt.rcParams["figure.autolayout"] = True
            fig, ax = plt.subplots(1, 3, figsize=(28, 11.5), sharex=True)
            ax[0].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
            x = np.array(range(np.shape(TFR)[1]))
            ax[0].plot(x, (ifreq / fstep).T, linewidth=1, color='red')
            ax[0].plot(x, tfsupp[0, :] / fstep, linewidth=1, color='w')
            ax[1].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
            ax[2].plot(ifreq, color='red')
            ax[2].plot(tfsupp[0, :], color='green')

        #plt.show()
        plt.savefig(fig_name)
#fig.savefig('C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\fake_kiwi_syl_1024_512.jpg')

#we need more wopt for signal recostruction


#freqarr=np.arange(np.shape(TFR)[0], step=10/fs)
#wp=IFreq.Wp(window_width,fs)
#wopt=IFreq.Wopt(fs,0,fs/2,wp)


#if calc_type == 1
#    [WT,freqarr,wopt]=wt(signal,fs,'fmin',interval1,'fmax',interval2,'CutEdges','off',...
#        'Preprocess',preprocess,'Wavelet',w_type);
#else
#    [WT,freqarr,wopt]=wft(signal,fs,'fmin',interval1,'fmax',interval2,'CutEdges','off',...
#        'Preprocess',preprocess,'Window',w_type);
#end

#tfsupp,ecinfo, ec, Skel= IF.ecurve(TFR,freqarr,wopt)

#[iamp,iphi,ifreq] = IF.rectfr(tfsupp,TFR,freqarr,wopt)

#recon = iamp.*cos(iphi);
#iphi = mod(iphi,2*pi);

#transform = WT;
#freq = freqarr;
