"""
06/12/2021
Author: Virginia Listanti

This is a script to test how to invert a Melscale spectrogram or Multitaper
"""

import SignalProc
import IF as IFreq
import numpy as np
import wavio
import librosa
import matplotlib.pyplot as plt


def mel_filterbank_maker(window_size, filter='mel', nfilters=40, minfreq=0, maxfreq=None, normalise=True):
    # Transform the spectrogram to mel or bark scale
    if maxfreq is None:
        maxfreq = sp.sampleRate / 2
    print(filter, nfilters, minfreq, maxfreq, normalise)

    if filter == 'mel':
        filter_points = np.linspace(sp.convertHztoMel(minfreq), sp.convertHztoMel(maxfreq), nfilters + 2)
        bins = sp.convertMeltoHz(filter_points)
    elif filter == 'bark':
        filter_points = np.linspace(sp.convertHztoBark(minfreq), sp.convertHztoBark(maxfreq), nfilters + 2)
        bins = sp.convertBarktoHz(filter_points)
    else:
        print("ERROR: filter not known", filter)
        return (1)

    nfft = int(window_size / 2)
    freq_points = np.linspace(minfreq, maxfreq, nfft)

    filterbank = np.zeros((nfft, nfilters))
    for m in range(nfilters):
        # Find points in first and second halves of the triangle
        inds1 = np.where((freq_points >= bins[m]) & (freq_points <= bins[m + 1]))
        inds2 = np.where((freq_points >= bins[m + 1]) & (freq_points <= bins[m + 2]))
        # Compute their contributions
        filterbank[inds1, m] = (freq_points[inds1] - bins[m]) / (bins[m + 1] - bins[m])
        filterbank[inds2, m] = (bins[m + 2] - freq_points[inds2]) / (bins[m + 2] - bins[m + 1])

    if normalise:
        # Normalise to unit area if desired
        norm = filterbank.sum(axis=0)
        norm = np.where(norm == 0, 1, norm)
        filterbank /= norm

    return filterbank

file_name = "C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy " + \
            "signals\\pure_tone\\Base_Dataset_2\\pure_tone_00.wav"

window_width=1024
incr=256
window= "Hann"
nfilters= 218

sp = SignalProc.SignalProc(window_width, incr)
sp.readWav(file_name)
fs = sp.sampleRate

# #evaluate spectrogram
# TFR = sp.spectrogram(window_width, incr, window, sgType = "Standard",sgScale = 'Mel Frequency',  nfilters = nfilters)
# TFR2 = sp.spectrogram(window_width, incr, window, sgType = 'Standard',sgScale = 'Linear')
# plt.imshow(TFR2)
# plt.show()
#you need to call sp.scalogram
#print("Spectrogram Dim =", np.shape(TFR))

#TEST 0 normal invertion
TFR = sp.spectrogram(window_width, incr, window, sgType = 'Standard',sgScale = 'Linear')
signal_inverted = sp.invertSpectrogram(TFR, window_width=window_width, incr=incr, window=window)

#TEST 1 using librosa library
#TFR = TFR.T (not sure about this)
#signal_inverted = librosa.feature.inverse.mel_to_audio(TFR, hop_length=incr, win_length=window_width, window=window)

# #TEST2 pseudoinverse
F = mel_filterbank_maker(window_width, 'mel', nfilters)
F_pseudo = np.linalg.pinv(F)
TFR_recovered = np.absolute(np.dot(TFR, F_pseudo)) #note: in signal proc we have self.sg = np.dot(self.sg,filterbank)
plt.imshow(np.absolute(TFR_recovered-TFR2))
plt.savefig(file_name[:-4]+"_mel_inversion_test_diff.jpg")
#TFR_recovered = TFR @ F_pseudo
signal_inverted = sp.invertSpectrogram(TFR_recovered, window_width=window_width, incr=incr, window=window)

# # TEST3 numpy.linalg least square
# F = mel_filterbank_maker(window_width, 'mel', nfilters)
# TFR_recovered=np.absolute(np.linalg.lstsq(F.T, TFR.T)[0])
# # plt.imshow(np.absolute(TFR_recovered.T-TFR2))
# # plt.savefig(file_name[:-4]+"_mel_inversion_test_diff2.jpg")
# plt.imshow(TFR_recovered)
# plt.savefig(file_name[:-4]+"_mel_inversion_test2.jpg")
# signal_inverted = sp.invertSpectrogram(TFR_recovered.T, window_width=window_width, incr=incr, window=window)


# # TEST4 &10 librosa.util non negative least square
# F = mel_filterbank_maker(window_width, 'mel', nfilters)
# power= 2.0 #default value in librosa mel_to_stft
# TFR_recovered = librosa.util.nnls(F.T, TFR.T)
# np.power(TFR_recovered, 1.0 / power, out=TFR_recovered)
# # plt.imshow(np.absolute(TFR_recovered.T-TFR2))
# # plt.savefig(file_name[:-4]+"_mel_inversion_test_diff2.jpg")
# plt.imshow(TFR_recovered)
# plt.savefig(file_name[:-4]+"_mel_inversion_test3.jpg")
# signal_inverted = sp.invertSpectrogram(TFR_recovered.T, window_width=window_width, incr=incr, window=window)

# #TEST 5 using librosa library but giving nfft
# TFR = TFR.T
# signal_inverted = librosa.feature.inverse.mel_to_audio(TFR, sr=fs, n_fft=window_width *2, hop_length=incr, win_length = window_width, window=window)
#

# #TEST 11 use librosa to recover spectrogram from TFR
# TFR_recovered = librosa.feature.inverse.mel_to_stft(TFR.T, sr=fs, n_fft= int(window_width-1))
# signal_inverted = sp.invertSpectrogram(TFR_recovered.T, window_width=window_width, incr=incr, window=window)
# save_file_path=file_name[:-4]+"_inv10.wav"
# wavio.write(save_file_path, signal_inverted, fs, sampwidth=2)

#TEST 12 MULTITAPERED SPECTROGRAM INVERSION
# TFR= sp.spectrogram(window_width, incr, window, sgType = "Multi-tapered",sgScale = 'Linear')
# signal_inverted = sp.invertSpectrogram(TFR, window_width=window_width, incr=incr, window=window)

save_file_path=file_name[:-4]+"_inv_00.wav"
wavio.write(save_file_path, signal_inverted, fs, sampwidth=2)