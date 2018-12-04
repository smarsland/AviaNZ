import numpy as np
import pylab as pl
import pywt
import scipy.fftpack as fft

def ConvertWaveletNodeName(i):
    """ Convert from an integer to the 'ad' representations of the wavelet packets
    The root is 0 (''), the next level are 1 and 2 ('a' and 'd'), the next 3, 4, 5, 6 ('aa','ad','da','dd) and so on
    """
    level = int(np.floor(np.log2(i + 1)))
    first = 2 ** level - 1
    if i == 0:
        b = ''
    else:
        b = np.binary_repr(int(i) - first, width=int(level))
        b = b.replace('0', 'a')
        b = b.replace('1', 'd')
    return b

def reconstructWPT(new_wp,wavelet,listleaves):
    """ Create a new wavelet packet tree by copying in the data for the leaves and then performing
    the idwt up the tree to the root.
    Assumes that listleaves is top-to-bottom, so just reverses it.
    """
    # Sort the list of leaves into order bottom-to-top, left-to-right
    working = listleaves.copy()
    working = working[-1::-1]
    print(len(working), working, len(working)==1)
    if len(working) == 1:
        working = np.concatenate((working,np.array([working[0]-1])))
    print(len(working), working, len(working)==1)
    level = int(np.floor(np.log2(working[0] + 1)))
    while level > 0:
        first = 2 ** level - 1
        while working[0] >= first:
            # Note that it assumes that the whole list is backwards
            parent = (working[0] - 1) // 2
            p = ConvertWaveletNodeName(parent)
            print("ere",working)
            print("ere",working,working[0],working[1])
            names = [ConvertWaveletNodeName(working[0]), ConvertWaveletNodeName(working[1])]
            print("here")
            pywt.idwt(new_wp[names[1]].data, new_wp[names[0]].data, wavelet)
            print("there")
            new_wp[p].data = pywt.idwt(new_wp[names[1]].data, new_wp[names[0]].data, wavelet)[:len(new_wp[p].data)]
            # Delete these two nodes from working
            working = np.delete(working, 1)
            working = np.delete(working, 0)
            # Insert parent into list of nodes at the next level
            ins = np.where(working > parent)
            if len(ins[0]) > 0:
                ins = ins[0][-1] + 1
            else:
                ins = 0
            working = np.insert(working, ins, parent)
        level = int(np.floor(np.log2(working[0] + 1)))
    return new_wp

t = np.linspace(0,5+47/400,2048)

sines = np.sin(30*np.pi*t) + np.sin(60*np.pi*t) + np.sin(90*np.pi*t) + np.sin(120*np.pi*t) + np.sin(180*np.pi*t)
rest = 2*np.exp(-30*t) * np.sin(260*np.pi*t)
inds = np.where(t<0.125)
sines[inds] += rest[inds]
inds = np.where((t>0.3725) & (t<0.5))
sines[inds] += rest[inds]
inds = np.where((t>0.7475) & (t<5.1175))
sines[inds] += rest[inds]

pl.ion()
pl.figure()
pl.plot(t[:300],sines[:300])

wp = pywt.WaveletPacket(data=sines, wavelet='db4',maxlevel=3)
l = 400

fig,axes = pl.subplots(6,3)
new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['a'] = wp['a'].data
x = new_wp.reconstruct(update=False)
axes[0,0].plot(x[:300])
ft = fft.fft(x,l)
ft[l//4:3*l//4] = 0
axes[0,1].plot(np.abs(ft)[:200])
new = np.real(fft.ifft(ft))
axes[0,2].plot(new[:300])

new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['d'] = wp['d'].data
x = new_wp.reconstruct(update=False)
axes[1,0].plot(x[:300])
ft = fft.fft(x,l)
ft[:l//4] = 0
ft[3*l//4:] = 0
axes[1,1].plot(np.abs(ft)[:200])
new = np.real(fft.ifft(ft))
axes[1,2].plot(new[:300])

new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['aa'] = wp['aa'].data
x = new_wp.reconstruct(update=False)
axes[2,0].plot(x[:300])
ft = fft.fft(x,l)
ft[l//8:7*l//8] = 0
axes[2,1].plot(np.abs(ft)[:200])
new = np.real(fft.ifft(ft))
axes[2,2].plot(new[:300])

new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['ad'] = wp['ad'].data
x = new_wp.reconstruct(update=False)
axes[3,0].plot(x[:300])
ft = fft.fft(x,l)
ft[:l//8] = 0
ft[3*l//8:] = 0
axes[3,1].plot(np.abs(ft)[:200])
new = np.real(fft.ifft(ft))
axes[3,2].plot(new[:300])

new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['aaa'] = wp['aaa'].data
x = new_wp.reconstruct(update=False)
axes[4,0].plot(x[:300])
ft = fft.fft(x,l)
ft[l//16:15*l//16] = 0
axes[4,1].plot(np.abs(ft)[:200])
new = np.real(fft.ifft(ft))
axes[4,2].plot(new[:300])

new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['aad'] = wp['aad'].data
x = new_wp.reconstruct(update=False)
axes[5,0].plot(x[:300])
ft = fft.fft(x,l)
ft[:l//16] = 0
ft[3*l//16:] = 0
axes[5,1].plot(np.abs(ft)[:200])
new = np.real(fft.ifft(ft))
axes[5,2].plot(new[:300])
