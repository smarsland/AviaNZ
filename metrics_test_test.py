# 15/11/2021
# Author: Virginia Listanti
# Help script to test metrics before big test

#NOTE FOR SELF_ we just need TFR transposed (without flipud) because in this way
# the freq. axis is alligned with freqarray


import SignalProc
import IF as IFreq
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import os
import csv
import imed
import speechmetrics as sm



# define metrics functions

def Signal_to_noise_Ratio(signal, noise):
    # Signal-to-noise ratio
    # Handle the case with no noise as well
    if len(noise) == 0:
        snr = 0
    else:
        snr = 10 * np.log10((np.sum(signal** 2)/len(signal)) / (np.mean(noise ** 2)/len(noise)))
    return snr


def Renyi_Entropy(A, order=3):
    # Renyi entropy.
    # Default is order 3

    R_E = (1 / (1 - order)) * np.log2(np.sum(A ** order) / np.sum(A))
    return R_E


def Iatsenko_style(s1, s2):
    # This function implement error function as defined
    # in Iatsenko et al. IF paper
    # s1 is the reference signal
    try:
        error = np.mean((s1 - s2) ** 2) / np.mean((s1 - np.mean(s1)) ** 2)
    except:
        error=np.nan
    return error

def IMED_distance(A,B):
    # This function evaluate IMED distance between 2 matrix
    # 1) Rescale matrices to [0,1]
    # 2) call imed distance

    A2=(A-np.amin(A))/np.ptp(A)
    B2=(B-np.amin(B))/np.ptp(B)

    return imed.distance(A2,B2)


######################## MAIN ######################################################################

test_name = "Test_1"  # change test name
dataset_dir = "C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\exponential_upchirp\\Base_Dataset"
test_dir = "C:\\Users\\Virginia\\Documents\\GitHub\\Thesis\\Experiments\\Metrics_test_plot"
test_fold = test_dir + "\\" + test_name

#inizialization for sisdr score
metric=sm.load("relative.sisdr", window=None)

# check if test_fold exists
if not os.path.exists(test_fold):
    os.mkdir(test_fold)

# initialization
SNR = np.zeros((9,1))
RE = np.zeros((9,1))
L2 = np.zeros((9,1))
IAM = np.zeros((9,1)) #iatsenko metric
SDR_original = np.zeros((9,1))
SDR_noise= np.zeros((9,1))
IMED_original = np.zeros((9,1))
IMED_noise = np.zeros((9,1))

file_id = []

# TFR parameters
window_width = 2048
incr = 512
window = "Hann"

# IF law
# A=1
T = 5
#pure_tone
#omega_0=2000
#inst_freq_fun = lambda t: omega * np.ones((np.shape(t)))

#exponential down chirp
# omega_1=500
# omega_0=2000
# alpha=(omega_1/omega_0)**(1/T)
#
# inst_freq_fun=lambda x: omega_0*alpha**x

#exponential up-chirp
omega_1=2000
omega_0=500
alpha=(omega_1/omega_0)**(1/T)
inst_freq_fun=lambda x: omega_0*alpha**x

# #linear down_chirp
# omega_1=500
# omega_0=2000
# c=(omega_1-omega_0)/T
# inst_freq_fun=lambda x: omega_0+c*x

# #linear upchirp
# omega_1=2000
# omega_0=500
# c=(omega_1-omega_0)/T
# inst_freq_fun=lambda x: omega_0+c*x

k=0
for file in os.listdir(dataset_dir):
    if file.endswith(".wav"):
        file_id.append(file)
        IF = IFreq.IF(method=2, pars=[1, 1])
        sp = SignalProc.SignalProc(window_width, incr)
        sp.readWav(dataset_dir + '\\' + file)
        sig1 = sp.data
        fs = sp.sampleRate

        # evaluate TFR
        TFR = sp.spectrogram(window_width, incr, window)
        # plt.imshow(TFR)
        # plt.show()
        TFR2 = TFR.T
        # plt.imshow(TFR2)
        # plt.show()

        # extract IF
        fstep = (fs / 2) / np.shape(TFR2)[0]
        freqarr = np.arange(fstep, fs / 2 + fstep, fstep)

        wopt = [fs, window_width]  # this neeeds review
        tfsupp, _, _ = IF.ecurve(TFR2, freqarr, wopt)
        inst_freq = inst_freq_fun(np.linspace(0, T, np.shape(tfsupp[0, :])[0]))
        plt.plot(inst_freq)
        plt.show()

        # invert TFR
        s1_inverted = sp.invertSpectrogram(TFR, window_width=window_width, incr=incr, window=window)

        # spectrogram of inverted signal
        sp.data = s1_inverted
        TFR_inv = sp.spectrogram(window_width, incr, window)
        # plt.imshow(TFR_inv)
        # plt.show()
        TFR2_inv = TFR_inv.T
        # plt.imshow(TFR2_inv)
        # plt.show()

        #evaluate metrics

        #base metrics
        #snr
        if file.endswith("0.wav"):
           signal_original=sig1
           TFR_original=TFR2
           noise=[]
        else:
            noise=sig1-signal_original

        SNR[k,0]=Signal_to_noise_Ratio(sig1,noise)

        #Renyi Entropy
        RE[k,0]=Renyi_Entropy(TFR2)

        # IF metrics
        #L2 norm
        L2[k,0]=norm(tfsupp[0,:]-inst_freq)

        #IAM
        IAM[k,0]=Iatsenko_style(inst_freq,tfsupp[0,:])

        #Reconstructed signal metrics

        #sisdr
        score = metric(signal_original, s1_inverted, rate=fs)
        SDR_original[k,0]=score["sisdr"]
        score = metric(sig1, s1_inverted, rate=fs)
        SDR_noise[k, 0] = score["sisdr"]

        #imed
        IMED_original[k,0]=IMED_distance(TFR_original[:,1:-1], TFR2_inv)
        IMED_noise[k, 0] = IMED_distance(TFR2[:,1:-1], TFR2_inv)

        del tfsupp
        k+=1

    #save plots
fig_name=test_fold +'\\metrics_plot_exponential_upchirp.jpg'
#plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots(4, 2, figsize=(20,40))

ax[0, 0].plot(SNR, 'o')
ax[0, 0].set_title('Signal-to-noise ratio',fontsize='large')
ax[0,0].set_xticks(np.arange(0, 9))
ax[0,0].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
ax[0, 1].plot(RE, 'o')
ax[0, 1].set_title('Renyi Entropy')
ax[0,1].set_xticks(np.arange(0, 9))
ax[0,1].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
#ax[0, 2].boxplot(L2_inv_or_G)
#ax[0, 2].set_title('L2 inv. sound vs original')
#ax[0,2].set_xticks(np.arange(1, 9))
#ax[0,2].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
ax[1, 0].plot(L2, 'o')
ax[1, 0].set_title('L2 norm IF')
ax[1,0].set_xticks(np.arange(0, 9))
ax[1,0].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
ax[1, 1].plot(IAM,'o')
ax[1, 1].set_title('Iats. Error')
ax[1,1].set_xticks(np.arange(0, 9))
ax[1,1].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
#ax[1, 2].plot(KLD_inv_or_G)
#ax[1, 2].set_title('KLD inv. sound vs original')
#ax[1,2].set_xticks(np.arange(1, 9))
#ax[1,2].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
ax[2, 0].plot(SDR_original, 'o')
ax[2, 0].set_title('SDR original signal')
ax[2,0].set_xticks(np.arange(0, 9))
ax[2,0].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
ax[2, 1].plot(SDR_noise,'o')
ax[2, 1].set_title('SDR signal + noise')
ax[2, 1].set_xticks(np.arange(0, 9))
ax[2, 1].set_xticklabels(['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'], rotation=45)
ax[3, 0].plot(IMED_original, 'o')
ax[3, 0].set_title('IMED original signal')
ax[3, 0].set_xticks(np.arange(0, 9))
ax[3, 0].set_xticklabels(
    ['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],
    rotation=45)
ax[3, 1].plot(IMED_noise,'o')
ax[3, 1].set_title('IMED signal + noise')
ax[3, 1].set_xticks(np.arange(0, 9))
ax[3, 1].set_xticklabels(
    ['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],
    rotation=45)
fig.suptitle('Linear Downchirp', fontsize=30)
plt.savefig(fig_name)