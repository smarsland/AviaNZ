import numpy as np
import pylab as pl
import SP as SignalProc

SP = SignalProc.SignalProc()
SP.readSoundFile('Sound Files/tril1.wav')
sg = SP.spectrogram()
pl.imshow(10*np.log10(np.flipud(sg.T)))
pl.show()
