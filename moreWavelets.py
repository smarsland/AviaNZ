import numpy as np
import pylab as pl
import pywt
import scipy.fftpack as fft

t = np.linspace(0,5+47/400,2048)

sines = np.sin(30*np.pi*t) + np.sin(60*np.pi*t) + np.sin(90*np.pi*t) + np.sin(120*np.pi*t) + np.sin(180*np.pi*t)
rest = 6*np.sin(260*np.pi*t) + exp(-10*t) #2*np.exp(-30*t) * np.sin(260*np.pi*t)
inds = np.where(t>0.5) and np.where(t<0.7475)
sines[inds] += rest[inds]
inds = np.where(t>0.125) and np.where(t<0.3725)
sines[inds] += rest[inds]
 
pl.ion()
pl.figure()
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

def ConvertWaveletNodeName(i):
    level = int(np.floor(np.log2(i + 1)))
    first = 2 ** level - 1
    if i == 0:
        b = ''
    else:
        b = np.binary_repr(int(i) - first, width=int(level))
        b = b.replace('0', 'a')
        b = b.replace('1', 'd')
    return b

fig,axes = pl.subplots(2,3)

# working
wp = pywt.WaveletPacket(data=sines, wavelet='db4',maxlevel=3)
l = len(wp.data)

new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=3)
new_wp['a'] = wp['a'].data
x = new_wp.reconstruct(update=False)
axes[1,0].plot(x[:l//8])
ft = fft.fft(x,l)
#axes[0,1].plot(np.abs(ft)[:l//2])
ft[l//4:3*l//4] = 0
new = np.real(fft.ifft(ft))
#axes[0,2].plot(new[:l//8])
wp2 = pywt.WaveletPacket(data=new, wavelet='db4',maxlevel=1)
print(wp2.reconstruct() - wp2[''].data)

new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=1)
new_wp['a'] = wp2['a'].data
x = new_wp.reconstruct(update=False)
axes[1,2].plot(x[:l//8])
ft = fft.fft(x,l) #not working
axes[0,1].plot(np.abs(ft))
ft = fft.fft(wp2.reconstruct(),l) # working
axes[1,1].plot(np.abs(ft))

# not working
def AntialiasWaveletPacket2(data,wavelet,mode,maxlevel):
    parent = 0
    print(parent, np.shape(wp_final[ConvertWaveletNodeName(parent)].data))
    wp_temp = pywt.WaveletPacket(data=wp_final[ConvertWaveletNodeName(parent)].data, wavelet=wavelet,mode=mode,maxlevel=1)
    l = len(wp_temp[''].data)
    # Get new approximation reconstruction
    new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet,mode=mode,maxlevel=1)
    # This doesn't help!
    #for level in range(new_wp.maxlevel + 1):
    #    for n in new_wp.get_level(level, 'natural'):
    #        n.data = np.zeros(len(wp_temp.get_level(level, 'natural')[0].data))
    new_wp['a'] = wp_temp['a'].data
    a = new_wp.reconstruct(update=False)
    ft = fft.fft(a,len(a))
    ft[l//4:3*l//4] = 0
    print(fft.fftfreq(len(a)))
    data = np.real(fft.ifft(ft))
    new_wp = pywt.WaveletPacket(data=data, wavelet=wavelet,mode=mode,maxlevel=1)
    wp_final[ConvertWaveletNodeName(parent)+'a'] = new_wp['a'].data
    # Get new detail reconstruction
    new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet,mode=mode,maxlevel=1)
    new_wp['d'] = wp_temp['d'].data
    d = new_wp.reconstruct(update=False)
    ft = fft.fft(d)
    ft[:l//4] = 0
    ft[3*l//4:] = 0
    data = np.real(fft.ifft(ft))
    new_wp = pywt.WaveletPacket(data=data, wavelet=wavelet,mode=mode,maxlevel=1)
    wp_final[ConvertWaveletNodeName(parent)+'d'] = new_wp['d'].data

import WaveletFunctions
l = len(sines)
WF = WaveletFunctions.WaveletFunctions(data=None, wavelet='db4', maxLevel=1)
wp_final = pywt.WaveletPacket(data=sines, wavelet=WF.wavelet,mode='symmetric',maxlevel=1)
wp_final[''] = sines
AntialiasWaveletPacket2(data=sines, wavelet=WF.wavelet, mode='symmetric', maxlevel=1)
#wp_final = WF.AntialiasWaveletPacket(data=sines, wavelet=WF.wavelet, mode='symmetric', maxlevel=1)


new_wp = pywt.WaveletPacket(data=None, wavelet='db4',maxlevel=1)
new_wp['a'] = wp_final['a'].data
#new_wp['d'] = wp_final['d'].data
x = new_wp.reconstruct(update=False)
axes[0,0].plot(x[:l//8])
ft = fft.fft(x, l)
axes[0,1].plot(np.abs(ft))


# manual
wv = pywt.Wavelet('db4')
fig,axes = pl.subplots(10,3)
goodtree = [sines]
# filter length for extension modes
flen = max(len(wv.dec_lo), len(wv.dec_hi), len(wv.rec_lo), len(wv.rec_hi))

for node in range(5):
    # retrieve parent node from J level
    data = goodtree[node]
    # downsample all non-root nodes because that wasn't done
    if node!=0:
        data = data[0::2]
    # symmetric mode
    data = np.concatenate((data[0:flen:-1], data, data[-flen:]))
    # zero-padding mode
    # data = np.concatenate((np.zeros(8), tree[node], np.zeros(8)))
    l = len(data)
    # make A_j+1 and D_j+1
    mana = np.convolve(data, wv.dec_lo, 'same')
    mand = np.convolve(data, wv.dec_hi, 'same')
    # antialias A_j+1
    ft = fft.fft(mana)
    axes[2*node,0].plot(np.abs(ft)[:l//2])
    ft[l//4 : 3*l//4] = 0
    nexta = np.real(fft.ifft(ft))
    # plot reconstruction
    x = np.convolve(nexta, wv.rec_lo)
    ft = fft.fft(x)
    axes[2*node,1].plot(x[:l//8])
    axes[2*node,2].plot(np.abs(ft)[:l//2])
    # store A before downsampling
    goodtree.append(nexta)
    # antialias D_j+1
    ft = fft.fft(mand)
    axes[2*node+1,0].plot(np.abs(ft)[:l//2])
    ft[:l//4] = 0
    ft[3*l//4:] = 0
    nextd = np.real(fft.ifft(ft))
    # plot reconstruction
    x = np.convolve(nextd, wv.rec_hi)
    ft = fft.fft(x)
    axes[2*node+1,1].plot(x[:l//8])
    axes[2*node+1,2].plot(np.abs(ft)[:l//2])
    # store D before downsampling
    goodtree.append(nextd)

cha = goodtree[9] #ada
chd = goodtree[10] #add
chdu = np.zeros(2*len(chd))
chau = np.zeros(2*len(cha))
chdu[0::2] = np.convolve(chd, wv.rec_hi, 'same') # d3->d2
chau[0::2] = np.convolve(cha, wv.rec_lo, 'same') # a3->d2
chduu = np.zeros(2*len(chdu))
chauu = np.zeros(2*len(chau))
chduu[0::2] = np.convolve(chdu, wv.rec_hi, 'same') # d2->a1
chauu[0::2] = np.convolve(chau, wv.rec_hi, 'same') # d2->a1
chduu = np.convolve(chduu, wv.rec_lo, 'same') # a1->a0
chauu = np.convolve(chauu, wv.rec_lo, 'same') # a1->a0
ft = fft.fft(chduu)
l = len(ft)
#ft[l//4 : 3*l//4] = 0 # 1/8 to 3/8 if recovering from aaa+aad
chduu = fft.ifft(ft)
x = chduu #+ chauu

cha = goodtree[9] #da
chd = goodtree[10] #dd
chdu = np.zeros(2*len(chd))
chau = np.zeros(2*len(cha))
chdu[0::2] = np.convolve(chd, wv.rec_hi, 'same') # d2->d1
chau[0::2] = np.convolve(cha, wv.rec_lo, 'same') # a2->d1
chdu = np.convolve(chdu, wv.rec_hi, 'same') # d1->a0
chau = np.convolve(chau, wv.rec_hi, 'same') # d1->a0
ft = fft.fft(chdu)
l = len(ft)
#ft[l//4 : 3*l//4] = 0 # 1/8 to 3/8 if recovering from aaa+aad
ft[:l//4] = 0
chdu = fft.ifft(ft)
x = chdu + chau

def graycode(n):
    ''' Return nth gray code. '''
    if n>0:
        return hipow(n) + graycode(2*hipow(n) - n - 1)
    return 0

def hipow(n):
    ''' Return the highest power of 2 within n. '''
    exp = 0
    while 2**exp <= n:
        exp += 1
    return 2**(exp-1)

def reconstruct(node, tree, wv):
    data = tree[node]
    lvl = math.floor(math.log2(node+1))
    # position of node in its level (0-based)
    nodepos = node - (2**lvl - 1)
    # Gray-permute node positions (cause wp is not in natural order)
    nodepos = graycode(nodepos)
    # positive freq is split into bands 0:1/2^lvl, 1:2/2^lvl,...
    # same for negative freq, so in total 2^lvl * 2 bands.
    numnodes = 2**(lvl+1)
    while lvl!=0:
        # convolve with rec filter
        if node%2 == 0:
            # node is detail
            data = np.convolve(data, wv.rec_hi, 'same')
        else:
            # node is approx
            data = np.convolve(data, wv.rec_lo, 'same')
        # upsample
        if lvl!=1:
            datau = np.zeros(2*len(data))
            datau[0::2] = data
            data = datau
        node = (node-1)//2
        lvl = lvl - 1
    # wipe images
    ft = fft.fft(data)
    l = len(ft)
    # to keep: [nodepos/numnodes : (nodepos+1)/numnodes] x Fs
    # (same for negative freqs)
    ft[ : l*nodepos//numnodes] = 0
    ft[l*(nodepos+1)//numnodes : -l*(nodepos+1)//numnodes] = 0
    # indexing [-0:] wipes everything
    if nodepos!=0:
        ft[-l*nodepos//numnodes : ] = 0
    data = np.real(fft.ifft(ft))
    return(data)

x = reconstruct(10, goodtree, wv)
ft = fft.fft(x)
l = len(ft)
axes[2,1].plot(x[:l//8])
axes[2,2].plot(np.abs(ft)[:l//2])
