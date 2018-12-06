import numpy as np
import pylab as pl
import pywt
import scipy.fftpack as fft

t = np.linspace(0,5+47/400,2048)

sines = np.sin(30*np.pi*t) + np.sin(60*np.pi*t) + np.sin(90*np.pi*t) + np.sin(120*np.pi*t) + np.sin(180*np.pi*t)
rest = 2*np.exp(-30*t) * np.sin(260*np.pi*t)
inds = np.where(t>0.5) and np.where(t<0.7475)
sines[inds] += rest[inds]
inds = np.where(t>0.125) and np.where(t<0.3725)
sines[inds] += rest[inds]
 
pl.ion()
#pl.figure()
#pl.plot(t[:300],sines[:300])

wp = pywt.WaveletPacket(data=sines, wavelet='db4',maxlevel=3)
l = len(wp.data)

fig,axes = pl.subplots(2,3)
new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['a'] = wp['a'].data
x = new_wp.reconstruct(update=False)
axes[0,0].plot(x[:l//8])
ft = fft.fft(x,l)
#axes[0,1].plot(np.abs(ft)[:l//2])
ft[l//4:3*l//4] = 0
new = np.real(fft.ifft(ft))
#axes[0,2].plot(new[:l//8])
wp2 = pywt.WaveletPacket(data=new, wavelet='db4',maxlevel=3)
axes[0,2].plot(wp2.reconstruct()[:l//8])
ft = fft.fft(wp2.reconstruct(),l)
axes[0,1].plot(np.abs(ft))


new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['d'] = wp['d'].data
x = new_wp.reconstruct(update=False)
axes[1,0].plot(x[:l//8])
ft = fft.fft(x,l)
#axes[1,1].plot(np.abs(ft)[:l//2])
ft[:l//4] = 0
ft[3*l//4:] = 0
axes[1,1].plot(np.abs(ft)[:l//2])
new2 = np.real(fft.ifft(ft))
axes[1,2].plot(new2[:l//8])

sines2 = new[0::2]
wp = pywt.WaveletPacket(data=sines2, wavelet='db4',maxlevel=3)
l = l //2

new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['a'] = wp['a'].data
x = new_wp.reconstruct(update=False)
axes[2,0].plot(x[:l//8])
ft = fft.fft(x,l)
axes[2,1].plot(np.abs(ft)[:l//2*2])
ft[l//4:3*l//4] = 0
#ft[l//8:7*l//8] = 0
#axes[2,1].plot(np.abs(ft)[:l//2*2])
new = np.real(fft.ifft(ft))
axes[2,2].plot(new[:l//8])

new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['d'] = wp['d'].data
x = new_wp.reconstruct(update=False)
axes[3,0].plot(x[:l//8])
ft = fft.fft(x,l)
axes[3,1].plot(np.abs(ft)[:l//2*2])
ft[:l//4] = 0
ft[3*l//4:] = 0
#ft[:l//8] = 0
#ft[2*l//8:] = 0
#axes[3,1].plot(np.abs(ft)[:l//2*2])
new3 = np.real(fft.ifft(ft))
axes[3,2].plot(new[:l//8])

sines3 = new3[0::2]
wp = pywt.WaveletPacket(data=sines3, wavelet='db4',maxlevel=3)
l = l //2

new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['a'] = wp['a'].data
x = new_wp.reconstruct(update=False)
axes[4,0].plot(x[:l//8])
ft = fft.fft(x,l)
axes[4,1].plot(np.abs(ft)[:l//2*4])
ft[l//4:3*l//4] = 0
#ft[l//16:15*l//16] = 0
#axes[4,1].plot(np.abs(ft)[:l//2*4])
new = np.real(fft.ifft(ft))
axes[4,2].plot(new[:l//8])

new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['d'] = wp['d'].data
x = new_wp.reconstruct(update=False)
axes[5,0].plot(x[:l//8])
ft = fft.fft(x,l)
axes[5,1].plot(np.abs(ft)[:l//2*4])
ft[:l//4] = 0
ft[3*l//4:] = 0
#ft[:l//16] = 0
#ft[2*l//16:] = 0
#axes[5,1].plot(np.abs(ft)[:l//2*4])
new = np.real(fft.ifft(ft))
axes[5,2].plot(new[:l//8])

#wp = pywt.WaveletPacket(data=sines, wavelet='db4', mode='symmetric', maxlevel=3)

import WaveletFunctions
importlib.reload(WaveletFunctions)
WF = WaveletFunctions.WaveletFunctions(data=None, wavelet='db4', maxLevel=1)
wp = WF.AntialiasWaveletPacket(data=sines, wavelet=WF.wavelet, mode='symmetric', maxlevel=1)
fig,axes = pl.subplots(1,2)
new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['a'] = wp['a'].data
x = new_wp.reconstruct(update=False)
axes[0].plot(x[:len(sines)//8])
ft = fft.fft(x)
axes[1].plot(np.abs(ft)[:len(sines)])

