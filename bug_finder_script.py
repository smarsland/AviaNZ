"""
3/03/2022
Author: Virginia Listanti
"""

import SignalProc
import IF as IFreq
import numpy as np
import os

dir_path = "C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\pure_tone\\Dataset_2\\pure_tone_1"

for file in os.listdir(dir_path):

    signal_path = dir_path + '\\' +  file

    win_len = 256
    hop = 64
    window_type = "Parzen"
    mel_num = 256
    alpha = 0.5
    beta = 0

    spec_type = "Multi-tapered"
    scale = 'Mel Frequency'
    norm_type = "Standard"


    IF = IFreq.IF(method=2, pars=[alpha, beta])
    sp = SignalProc.SignalProc(win_len, hop)
    sp.readWav(signal_path)
    signal = sp.data
    fs = sp.sampleRate

    TFR = sp.spectrogram(win_len, hop, window_type, sgType=spec_type, sgScale=scale, nfilters=mel_num)
    # spectrogram normalizations
    if norm_type != "Standard":
        sp.normalisedSpec(tr=norm_type)
        TFR = sp.sg
    TFR2 = TFR.T

    # appropriate frequency scale
    if scale == "Linear":
        fstep = (fs / 2) / np.shape(TFR2)[0]
        freqarr = np.arange(fstep, fs / 2 + fstep, fstep)
    else:
        # #mel freq axis
        nfilters = mel_num
        freqarr = np.linspace(sp.convertHztoMel(0), sp.convertHztoMel(fs / 2), nfilters + 1)
        freqarr = freqarr[1:]

    wopt = [fs, "win_len"]  # this neeeds review
    tfsupp, _, _ = IF.ecurve(TFR2, freqarr, wopt)

    print('\n ', signal_path, ' works')

