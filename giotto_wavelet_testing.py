import numpy as np
import matplotlib.pyplot as plt
import wavio
import WaveletFunctions
from ext import ce_detect
import Wavelet

# file = "../birds.wav"
# wavobj = wavio.read(file)
# data = wavobj.data
# data = data[:,0]
# samplerate = wavobj.rate

# WF = WaveletFunctions.WaveletFunctions(data=data, wavelet="dmey2", maxLevel=20, samplerate=samplerate)

# allnodes = list(range(31, 63))

# WF.WaveletPacket(allnodes, mode='symmetric', antialias=False)

# E, realwindow = WF.extractE(33,0.05,wpantialias=True)
# print("E: ", E)
# plt.plot(E)
# plt.show()

# E, realwindow = WF.extractE(38,0.05,wpantialias=True)
# print("E: ", E)
# plt.plot(E)
# plt.show()

#realmaxlen = 1.3
#alpha = 2.3

x = np.linspace(-3,3,1000)
E = (x-1)**3 + 3 + np.random.normal(0,0.1,1000)
E = E - np.min(E) + 1

wavelet = Wavelet.Wavelet("dmey2")

plt.plot(E)
plt.plot(np.convolve(E, wavelet.dec_hi, 'same'))
plt.plot(np.convolve(E, wavelet.dec_lo, 'same'))
plt.show()

# segm1 = ce_detect.launchDetector2(E, 1, 1, alpha=2).astype('float')

# for line in segm1:
#     xstart = x[int(line[0])-1]
#     xend = x[int(line[1])-1]
#     plt.plot([xstart,xend],[0,0])
# plt.show()