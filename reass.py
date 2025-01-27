
import numpy as np
import pylab as pl
import scipy.signal as signal
import scipy.fftpack as fft
#import wavio
import gc
import soundfile as sf

def spectrogram(data,window_width=None,incr=None,window='Hann',equal_loudness=False,mean_normalise=True,onesided=True,multitaper=False,need_even=False):
    sg = np.copy(data.squeeze())
    if sg.dtype != 'float':
        sg = sg.astype('float')

    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_width) / (window_width - 1)))
    starts = range(0, len(sg) - window_width, incr)
    ft = np.zeros((len(starts), window_width//2))
    for i in starts:
        winddata = window * sg[i:i + window_width]
        ft[i // incr, :] = fft.fft(winddata)[:window_width//2]
    sg = np.absolute(ft)

    #sg = (ft*np.conj(ft))[:,window_width // 2:].T
    pl.ion()
    pl.figure()
    pl.imshow(10.0*np.log10(sg.T),cmap='gray_r')
    pl.show()

data, fs = sf.read('Sound Files/lsk_1min.wav')
spectrogram(data,256,128)
