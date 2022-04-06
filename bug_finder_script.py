"""
3/03/2022
Author: Virginia Listanti
"""

import SignalProc
import IF as IFreq
import numpy as np
import os
from geodesic_copy import geod_sphere

def Geodesic_curve_distance(x1, y1, x2, y2):
    """
    Code suggested by Arianna Salili-James
    This function computes the distance between two curves x and y using geodesic distance

    Input:
         - x1, y1 coordinates of the first curve
         - x2, y2 coordinates of the second curve
    """

    beta1 = np.column_stack([x1, y1]).T
    beta2 = np.column_stack([x2, y2]).T

    distance, _, _ = geod_sphere(np.array(beta1), np.array(beta2), rotation=False)

    return distance


def set_if_fun(sig_id, t_len):
    """
    Utility function to manage the instantaneous frequency function

    INPUT:
        - signal_id: str to recognise signal type

    OUTPUT:
        - lambda function for instantaneous frequency
    """
    if sig_id == "pure_tone":
        omega = 2000
        if_fun = lambda t: omega * np.ones((np.shape(t)))

    elif sig_id == "exponential_downchirp":
        omega_1 = 500
        omega_0 = 2000
        alpha = (omega_1 / omega_0) ** (1 / t_len)
        if_fun = lambda x: omega_0 * alpha ** x

    elif sig_id == "exponential_upchirp":
        omega_1 = 2000
        omega_0 = 500
        alpha = (omega_1 / omega_0) ** (1 / t_len)
        if_fun = lambda x: omega_0 * alpha ** x

    elif sig_id == "linear_downchirp":
        omega_1 = 500
        omega_0 = 2000
        alpha = (omega_1 - omega_0) / t_len
        if_fun = lambda x: omega_0 + alpha * x

    elif sig_id == "linear_upchirp":
        omega_1 = 2000
        omega_0 = 500
        alpha = (omega_1 - omega_0) / t_len
        if_fun = lambda x: omega_0 + alpha * x

    else:
        if_fun = []
        print("ERROR SIGNAL ID NOT CONSISTENT WITH THE IF WE CAN HANDLE")
    return if_fun

dir_path = "C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\pure_tone\\Dataset_2\\pure_tone_1"

for file in os.listdir(dir_path):

    print('Testing ', file)
    signal_path = dir_path + '\\' +  file

    win_len = 256
    hop = 64
    window_type = "Hann"
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
    T = sp.fileLength / fs
    # assign IF function
    inst_freq_fun = set_if_fun('pure_tone', T)

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

    # revert to Hz if Mel
    if scale == 'Mel Frequency':
        tfsupp[0, :] = sp.convertMeltoHz(tfsupp[0, :])

    # array with temporal coordinates
    t_support = np.linspace(0, T, np.shape(tfsupp[0, :])[0])
    inst_freq = inst_freq_fun(t_support)  # IF law

    GEODETIC = Geodesic_curve_distance(t_support, tfsupp[0, :], t_support, inst_freq)

    print('Geodetic distance = ', GEODETIC)

    print('\n ', signal_path, ' works')

