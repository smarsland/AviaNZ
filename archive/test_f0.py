
import numpy as np
import scipy.ndimage as spi
import pylab as pl
from scipy.io import wavfile

import scipy.signal
from spectrum import *

def equalLoudness():
    EL80 = [np.array([0, 20, 30, 40, 50 ,60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3700, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000, 20000,22050]), np.array([120,113,103,97,93,91,89,87,86,85,78,76,76,76,76,77,78,79.5,80,79,77,74,71.5,70,70.5,74,79,84,86,86,85,95,110,125,140])]

    # Muck about to get rid of ones about sampling frequency
    #np.where(EL80[0]>rate/2)

    # Scale
    f = EL80[0]/(rate/2.)
    m = 10**((70-EL80[1])/20.0)

    # Yule-Walker algorithm finds autoregressive coefficients (IIR filter)
    ar, variance, coeff_reflection = aryule(f, 10)
    print ar, variance, coeff_reflection

    #% Use a MATLAB utility to design a best bit IIR filter
    #[b1,a1]=yulewalk(10,f,m);
    
    #% Add a 2nd order high pass filter at 150Hz to finish the job
    #[b2,a2]=butter(2,(150/(fs/2)),'high');

    return ar

def extractPeaks(sg):
    # Returns the indices for each col of the spectrogram that is a local optimum and bigger than the mean of that col
    sg = np.abs(sg)
    points = (np.diff(np.sign(np.diff(sg,axis=0)),axis=0) < 0) & (sg > 2*np.mean(sg, axis=0))[2:, :]
    return np.where(points.T)

def istft(sg,fft_size,incr):
    # x array is far too long?!
    if np.remainder(fft_size,2)==0:
        w = fft_size+1
    halfwin = 0.5 * (1+np.cos(np.pi*np.arange(fft_size/2)/float(fft_size)/2))
    win = np.zeros(fft_size)
    win[fft_size/2:fft_size] = halfwin
    win[fft_size/2:0:-1] = halfwin
    # Assuming that the hops are 25% of window
    win = 2.0/3*win

    x = np.zeros(fft_size+(np.shape(sg)[1]-1)*incr)
    for b in range(0,np.shape(sg)[1]-1):
        ft = sg[:,b]
        ft = np.concatenate((ft, np.conj(ft[fft_size/2::-1])))
        px = np.real(np.fft.ifft(ft))
        x[b*incr:b*incr+fft_size] += px*win

    return x

def vocoder(sg,r,incr):
    # Basically, we just resample the spectrogram at a set of points
    # r is the speed up
    N = 2*(np.shape(sg)[0]-1)
    samples = np.arange(0,np.shape(sg)[1],r)
    newsg = np.zeros((np.shape(sg)[0],len(samples)),dtype=complex)
    dphi = np.zeros(N/2+1)
    dphi[1:-1] = 2.0*np.pi*incr/N/np.arange(1,N/2)

    # Phase accumulator
    ph_acc = np.angle(sg[:,0])

    col = 0
    for t in samples[:-1]:
        sgcols = sg[:,np.floor(t):np.floor(t)+2]
        tf = t - np.floor(t)
        sgmag = (1-tf)*np.abs(sgcols[:,0]) + tf*np.abs(sgcols[:,1])
        dp = np.angle(sgcols[:,1]) - np.angle(sgcols[:,0]) - dphi
        # Get the principal argument (so it is in [-pi:pi])
        dp -= 2.*np.pi*np.round(dp/2.*np.pi)
        newsg[:,col] = sgmag * np.exp(np.complex(0,1)*ph_acc)
        ph_acc += dphi+dp
        col += 1
    return dp, newsg

def corrections(peaks,rate,fft_size,incr):
    # Frequency and amplitude correction for the spectrogram bins
    kappa, newsg = vocoder(sg, 1, incr)
    kappa = fft_size / (2.0 * np.pi * incr) * kappa
    peakfreqs = np.zeros(np.shape(peaks))
    peakampls = np.zeros(np.shape(peaks))
    for i in range(np.shape(sg)[1]):
        inds = np.where(peaks[0] == i)
        peakfreqs[1][inds] = (peaks[1][inds] + kappa[peaks[1][inds]]) * rate / float(fft_size)
        peakampls[1][inds] = 0.5 * sg[peaks[1][inds], i] / (
                    0.5 * (1 + np.cos(np.pi * kappa[peaks[1][inds]] / float(fft_size) / 2)))

    return peakfreqs, peakampls

def saliance(freqs,ampls,maxh=5):
    B = lambda f: np.floor(120*np.log2(f/55.0)+1)

    # TODO: work out b (bin?) and put it all together
    def g(b,h,f,alpha):
        delta = lambda f, h: np.abs(B(f / h) - b) / 10.0

        d = delta(f,h)
        if d<=1:
            return np.cos(*np.pi/2)**2 * alpha**(h-1)
        else:
            return 0

    gamma = 0
    beta = 0
    alpha = 0

    maxa = max(ampls[1])
    e = np.where(20.0 * np.log(maxa / ampls) < gamma,1,0)

    salience = 0
    for h in range(1,maxh):
        salience += e*g*ampls[1]**beta






def test(fft_size = 2048,incr = 128):
    import SignalProc as sp

    a = sp.SignalProc(window_width=fft_size,incr=incr)
    rate, data = wavfile.read('Sound Files/tril1.wav')
    #rate, data = wavfile.read('../Sound Files/010816_191458_kiwi.wav')
    return a.spectrogram(data).T, rate

fft_size = 2048
incr = 128
sg, rate = test(fft_size,incr)


peaks = extractPeaks(sg)
freqs,ampls = corrections(peaks,rate,fft_size,incr)

#pl.figure(), pl.imshow(sg), pl.plot(peaks[0],peaks[1],'wo')

