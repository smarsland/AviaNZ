"""
Aid script to test spectrogram inversion
"""
import SignalProc
import numpy as np
import wavio

file_name = "C:\\Users\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\linear_downchirp\\Base_Dataset_2\\linear_downchirp_00.wav"

opt_param = {'win_len': 2048, 'hop': 1024, 'window_type': 'Hann', 'mel_num': None, 'alpha': 10.0, 'beta': 10.0}
sg_type = 'Standard'
sg_scale = 'Linear'
sp = SignalProc.SignalProc(opt_param["win_len"], opt_param["hop"])

# read file
sp.readWav(file_name)
samplerate=sp.sampleRate
tfr = sp.spectrogram(opt_param["win_len"], opt_param["hop"], opt_param["window_type"], sgType=sg_type,
                         sgScale=sg_scale, nfilters=opt_param["mel_num"])


TFR_recovered = tfr
print('shape TFR rec ', np.shape(TFR_recovered), ' window width ', opt_param["win_len"], ' incr ', opt_param["hop"],
      ' window type ', opt_param["window_type"])
s1_inverted = sp.invertSpectrogram(TFR_recovered, window_width=opt_param["win_len"], incr=opt_param["hop"],
                                   window=opt_param["window_type"])


save_file_path=file_name[:-4]+"_inv_3.wav"
wavio.write(save_file_path, s1_inverted, samplerate, sampwidth=2)