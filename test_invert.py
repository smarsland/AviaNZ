
import numpy as np
import pylab as pl
import SignalProc
sp = SignalProc.SignalProc()
sp.readWav("/home/marslast/Downloads/linear_downchirp_00.wav")

sgRaw = sp.spectrogram(1024,428)
#pl.imshow(np.log10(sgRaw))
#pl.show()
newdata = sp.invertSpectrogram(sgRaw,1024,265)
#newdata = sp.invertSpectrogram(sgRaw,512,256)
#newdata = sp.invertSpectrogram(sgRaw,1024,765)
sgNew = sp.spectrogram()

pl.imshow(np.log10(sgNew))
pl.show()
