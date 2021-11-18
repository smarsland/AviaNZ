# 16/11/2021
# Author Virginia Listanti

# Debug script to understand what happens with sisdr from speechmetrics

import speechmetrics as sm
import SignalProc

reference="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\pure_tone\\Base_Dataset\\pure_tone_0.wav"
test = "C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\pure_tone\\Base_Dataset\\pure_tone_1.wav"

metric=sm.load("relative.sisdr", window=None)
window_width = 2048
incr = 512
window = "Hann"
sp = SignalProc.SignalProc(window_width, incr)
sp.readWav(reference)

sample_rate=sp.sampleRate
ref_audio=sp.data

sp.readWav(test)
test_audio=sp.data

score=metric(ref_audio, test_audio, rate=sample_rate)
print(score)
print(score["sisdr"])