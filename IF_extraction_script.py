#21/05/2021
# Author: Virginia Listanti
#Script for Instantaneous frequency exstration

#TO Do: Define wopt


import SignalProc 
import IF as IFreq
import numpy as np
from scipy.io import loadmat, savemat
import pylab as pl


sp=SignalProc.SignalProc(512,256)
IF=IFreq.IF()

#data_file="C:\\Users\\Virginia\\Documents\\Work\\Data\\fakekiwi.wav"
data_file=loadmat("C:\\Users\\Virginia\\Documents\\GitHub\\AviaNZ\\1signal_10Hz.mat")
data_file=np.squeeze(data_file['y'])
print(np.shape(data_file))
fs=10

#spectrogram(self,window_width=None,incr=None,window='Hann',sgType=None,equal_loudness=False,mean_normalise=True,onesided=True,need_even=False):
window_width=64
incr=32
window="Hann"
sp.setData(data_file)
TFR=sp.spectrogram(window_width,incr,window)
#transpose
TFR=TFR.T
#savemat'C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\test_signal.mat',{'TFR':TFR})
#pl.imshow(TFR)(
#pl.show()
#print(np.shape(TFR))

freqarr=np.arange(5/32,5+5/32,5/32)
print(freqarr)
print(np.shape(freqarr))
wopt=[10,1]
tfsupp,ecinfo, Skel=IF.ecurve(TFR,freqarr,wopt);


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
