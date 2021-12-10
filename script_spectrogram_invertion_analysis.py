"""
18/01/2020 Virginia Listanti

This is work script to analyze the efficience of the spectrogram inversion algorithm

TODO:
- real sound file
- generate spectrogram (normal, different parameters, reassigned)
- invert spectrogram
- compute distance from recovered signal or from spectrogram of recovered signal
- evaluate what happens with maxpooling

"""
import SignalProc
from numpy import linalg as LA

sp=SignalProc.SignalProc(1024,512)
wave1=sp.data
file_dir="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\BatWavs\\LT"
file_name="2020-01-22 05-12-41"
sp.readWav(file_dir+'\\'+file_name)
spec=sp.spectrogram(1024, 512,'Blackman')

#recovered wave
wave2 = sp.invertSpectrogram(spec, 1024, 512)

#metrics

#L2 norm
m1=LA.norm(wave2-wave1)

#KLD