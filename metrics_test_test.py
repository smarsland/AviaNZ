# 15/11/2021
# Author: Virginia Listanti
# Help script to test metrics before big test

# NOTE FOR SELF_ we just need TFR transposed (without flipud) because in this way
# the freq. axis is alligned with freqarray


import SignalProc
import IF as IFreq
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import os
#import csv
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

def set_if_fun(signal_id,T):
    """
    Utility function to manage the instantaneous frequency function
    """
    if signal_id=="pure_tone":
        omega=2000
        if_fun = lambda t: omega * np.ones((np.shape(t)))

    elif signal_id=="exponential_downchirp":
        omega_1=500
        omega_0=2000
        alpha=(omega_1/omega_0)**(1/T)
        if_fun=lambda x: omega_0*alpha**x

    elif signal_id=="exponential_upchirp":
        omega_1=2000
        omega_0=500
        alpha=(omega_1/omega_0)**(1/T)
        if_fun=lambda x: omega_0*alpha**x

    elif signal_id=="linear_downchirp":
        omega_1=500
        omega_0=2000
        c=(omega_1-omega_0)/T
        if_fun=lambda x: omega_0+c*x

    elif signal_id=="linear_upchirp":
        omega_1=2000
        omega_0=500
        c=(omega_1-omega_0)/T
        if_fun=lambda x: omega_0+c*x

    else:
        print("ERROR SIGNAL ID NOT CONSISTENT WITH THE IF WE CAN HANDLE")
    return if_fun

######################## MAIN ######################################################################

test_name = "Test_9"  # change test name
file_id="exponential_upchirp"
dataset_dir = "C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals\\"+file_id+"\\Base_Dataset"
test_dir = "C:\\Users\\Virginia\\Documents\\GitHub\\Thesis\\Experiments\\Metrics_test_plot"
test_fold = test_dir + "\\" + test_name

#inizialization for sisdr score
metrics=sm.load(['bsseval',"stoi",'sisdr'], window=None)

# check if test_fold exists
if not os.path.exists(test_fold):
    os.mkdir(test_fold)

# initialization
# SNR = np.zeros((9,1))
# RE = np.zeros((9,1))
# L2 = np.zeros((9,1))
# IAM = np.zeros((9,1)) #iatsenko metric
# SDR_original = np.zeros((9,1))
# SDR_noise= np.zeros((9,1))
# IMED_original = np.zeros((9,1))
# IMED_noise = np.zeros((9,1))
#
# NORM_S_1=np.zeros((9,1))
# NORM_S_2=np.zeros((9,1))
# RATIO_1=np.zeros((9,1))
# RATIO_2=np.zeros((9,1))
# NORM_DIFF_1= np.zeros((9,1))
# NORM_DIFF_2= np.zeros((9,1))

SDR_original=np.zeros((9, 1))
ISR_original=np.zeros((9, 1))
SAR_original=np.zeros((9, 1))
STOI_original=np.zeros((9,1))
SISDR_original = np.zeros((9, 1))
SDR_noise=np.zeros((9, 1))
ISR_noise=np.zeros((9, 1))
SAR_noise=np.zeros((9, 1))
STOI_noise=np.zeros((9,1))
SISDR_noise= np.zeros((9, 1))

# file_id = []

# TFR parameters
window_width = 2048
incr = 512
window = "Hann"

# IF law
# A=1
T = 5
inst_freq_fun=set_if_fun(file_id, T)

k=0
for file in os.listdir(dataset_dir):
    if file.endswith(".wav"):
        # file_id.append(file)
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
        # plt.plot(inst_freq)
        # plt.show()

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

        # #base metrics
        # #snr
        if file.endswith("0.wav"):
           signal_original=sig1
           TFR_original=TFR2
           noise=[]
        else:
            noise=sig1-signal_original
        #
        # SNR[k,0]=Signal_to_noise_Ratio(sig1,noise)
        #
        # #Renyi Entropy
        # RE[k,0]=Renyi_Entropy(TFR2)
        #
        # # IF metrics
        # #L2 norm
        # L2[k,0]=norm(tfsupp[0,:]-inst_freq)
        #
        # #IAM
        # IAM[k,0]=Iatsenko_style(inst_freq,tfsupp[0,:])
        #
        # #Reconstructed signal metrics
        #
        # #sisdr
        # # #cutting only reference and test  signal
        # # #respect to original signal
        # # len_diff=len(signal_original)-len(s1_inverted)
        # # score = metric(s1_inverted, signal_original[int(np.floor(len_diff/2)):-int(np.ceil(len_diff/2))],rate=fs)
        # # NORM_S_1[k, 0] = norm(signal_original)
        # # NORM_DIFF_1[k,0]=norm(signal_original[int(np.floor(len_diff/2)):-int(np.ceil(len_diff/2))]-s1_inverted)
        # # RATIO_1[k,0]=NORM_S_1[k,0]/NORM_DIFF_1[k,0]
        # # SDR_original[k,0]=score["sisdr"]
        # #
        # # #respect to signal +noise
        # # score = metric(s1_inverted, sig1[int(np.floor(len_diff/2)):-int(np.ceil(len_diff/2))], rate=fs)
        # # SDR_noise[k, 0] = score["sisdr"]
        # # NORM_S_2[k, 0] = norm(sig1)
        # # NORM_DIFF_2[k, 0] = norm(sig1[int(np.floor(len_diff/2)):-int(np.ceil(len_diff/2))] - s1_inverted)
        # # RATIO_2[k, 0] = NORM_S_2[k, 0] / NORM_DIFF_2[k, 0]
        #
        # # cutting both test and reference signal in order to have the same dim.
        #
        # # #test_4
        # # # respect to original signal
        # # len_diff = len(signal_original) - len(s1_inverted)
        # # s1_inverted = s1_inverted[int(np.floor(window_width/2-len_diff)):-int(np.floor(window_width/2))]
        # # score = metric(s1_inverted, signal_original[int(np.floor(window_width/2)):-int(np.floor(window_width/2))], rate=fs)
        # # NORM_S_1[k, 0] = norm(signal_original[int(np.floor(window_width/2)):-int(np.floor(window_width/2))])
        # # NORM_DIFF_1[k, 0] = norm(signal_original[int(np.floor(window_width/2)):-int(np.floor(window_width/2))] - s1_inverted)
        # # RATIO_1[k, 0] = NORM_S_1[k, 0] / NORM_DIFF_1[k, 0]
        # # SDR_original[k, 0] = score["sisdr"]
        # #
        # # # respect to signal +noise
        # # score = metric(s1_inverted, sig1[int(np.floor(window_width/2)):-int(np.floor(window_width/2))], rate=fs)
        # # SDR_noise[k, 0] = score["sisdr"]
        # # NORM_S_2[k, 0] = norm(sig1[int(np.floor(window_width/2)):-int(np.floor(window_width/2))])
        # # NORM_DIFF_2[k, 0] = norm(sig1[int(np.floor(window_width/2)):-int(np.floor(window_width/2))] - s1_inverted)
        # # RATIO_2[k, 0] = NORM_S_2[k, 0] / NORM_DIFF_2[k, 0]
        #
        # # # test_5
        # # #symmetric
        # # # respect to original signal
        # # len_diff = len(signal_original) - len(s1_inverted)
        # # s1_inverted = s1_inverted[int(np.ceil((window_width- len_diff)/2)):-int(np.floor((window_width -len_diff)/ 2))]
        # # score = metric(s1_inverted, signal_original[int(np.ceil(window_width / 2)):-int(np.floor(window_width / 2))],
        # #                rate=fs)
        # # NORM_S_1[k, 0] = norm(signal_original[int(np.ceil(window_width / 2)):-int(np.floor(window_width / 2))])
        # # NORM_DIFF_1[k, 0] = norm(signal_original[int(np.ceil(window_width / 2)):-int(np.floor(window_width / 2))] - s1_inverted)
        # # RATIO_1[k, 0] = NORM_S_1[k, 0] / NORM_DIFF_1[k, 0]
        # # SDR_original[k, 0] = score["sisdr"]
        # #
        # # # respect to signal +noise
        # # score = metric(s1_inverted, sig1[int(np.ceil(window_width / 2)):-int(np.floor(window_width / 2))], rate=fs)
        # # SDR_noise[k, 0] = score["sisdr"]
        # # NORM_S_2[k, 0] = norm(sig1[int(np.ceil(window_width / 2)):-int(np.floor(window_width / 2))])
        # # NORM_DIFF_2[k, 0] = norm(sig1[int(np.ceil(window_width / 2)):-int(np.floor(window_width / 2))] - s1_inverted)
        # # RATIO_2[k, 0] = NORM_S_2[k, 0] / NORM_DIFF_2[k, 0]
        #
        # # # test_6
        # # #as test 5but using incr instead of win len
        # # # respect to original signal
        # # len_diff = len(signal_original) - len(s1_inverted)
        # # s1_inverted = s1_inverted[int(np.ceil(incr - len_diff / 2)):-int(np.floor(incr - len_diff/ 2))]
        # # score = metric(s1_inverted, signal_original[int(np.ceil(incr )):-int(np.floor(incr))],
        # #                rate=fs)
        # # NORM_S_1[k, 0] = norm(signal_original[int(np.ceil(incr)):-int(np.floor(incr ))])
        # # NORM_DIFF_1[k, 0] = norm(
        # #     signal_original[int(np.ceil(incr )):-int(np.floor(incr ))] - s1_inverted)
        # # RATIO_1[k, 0] = NORM_S_1[k, 0] / NORM_DIFF_1[k, 0]
        # # SDR_original[k, 0] = score["sisdr"]
        # #
        # # # respect to signal +noise
        # # score = metric(s1_inverted, sig1[int(np.ceil(incr )):-int(np.floor(incr ))], rate=fs)
        # # SDR_noise[k, 0] = score["sisdr"]
        # # NORM_S_2[k, 0] = norm(sig1[int(np.ceil(incr)):-int(np.floor(incr ))])
        # # NORM_DIFF_2[k, 0] = norm(sig1[int(np.ceil(incr )):-int(np.floor(incr ))] - s1_inverted)
        # # RATIO_2[k, 0] = NORM_S_2[k, 0] / NORM_DIFF_2[k, 0]
        #
        # # test_7
        # # as test 6 + rescaling s1_inverted
        # # respect to original signal
        # len_diff = len(signal_original) - len(s1_inverted)
        # s1_inverted = s1_inverted[int(np.ceil(incr - len_diff / 2)):-int(np.floor(incr - len_diff / 2))]
        # s1_inverted_scaled_1 = s1_inverted / (np.ptp(s1_inverted) / np.ptp(signal_original))
        # score = metric(s1_inverted_scaled_1, signal_original[int(np.ceil(incr)):-int(np.floor(incr))],
        #                rate=fs)
        # NORM_S_1[k, 0] = norm(signal_original[int(np.ceil(incr)):-int(np.floor(incr))])
        # NORM_DIFF_1[k, 0] = norm(
        #     signal_original[int(np.ceil(incr)):-int(np.floor(incr))] - s1_inverted_scaled_1)
        # RATIO_1[k, 0] = NORM_S_1[k, 0] / NORM_DIFF_1[k, 0]
        # SDR_original[k, 0] = score["sisdr"]
        #
        # # respect to signal +noise
        # s1_inverted_scaled_2 = s1_inverted / (np.ptp(s1_inverted) / np.ptp(sig1))
        # score = metric(s1_inverted_scaled_2, sig1[int(np.ceil(incr)):-int(np.floor(incr))], rate=fs)
        # SDR_noise[k, 0] = score["sisdr"]
        # NORM_S_2[k, 0] = norm(sig1[int(np.ceil(incr)):-int(np.floor(incr))])
        # NORM_DIFF_2[k, 0] = norm(sig1[int(np.ceil(incr)):-int(np.floor(incr))] - s1_inverted_scaled_2)
        # RATIO_2[k, 0] = NORM_S_2[k, 0] / NORM_DIFF_2[k, 0]
        #
        # #imed
        # col_dif=np.shape(TFR_original)[1]-np.shape(TFR2_inv)[1]
        # IMED_original[k,0]=IMED_distance(TFR_original[:,int(np.floor(col_dif/2)):-int(np.ceil(col_dif/2))], TFR2_inv)
        # IMED_noise[k, 0] = IMED_distance(TFR2[:,int(np.floor(col_dif/2)):-int(np.ceil(col_dif/2))], TFR2_inv)


        # speech metrics comparison
        len_diff = len(signal_original) - len(s1_inverted)
        # [int(np.floor(len_diff/2)):-int(np.ceil(len_diff/2))]
        score_original = metrics(s1_inverted, signal_original,rate=fs)
        SDR_original[k, 0]=score_original['sdr']
        ISR_original[k, 0]=score_original['isr']
        SAR_original[k, 0]=score_original['sar']
        STOI_original[k,0]=score_original['stoi']
        SISDR_original[k, 0]=score_original['sisdr']

        score_noise = metrics(s1_inverted, sig1, rate=fs)
        SDR_noise[k, 0] = score_noise['sdr']
        ISR_noise[k, 0] = score_noise['isr']
        SAR_noise[k, 0] = score_noise['sar']
        STOI_noise[k, 0] = score_noise['stoi']
        SISDR_noise[k, 0] = score_noise['sisdr']
        del tfsupp
        k+=1


# #save metric plots
# fig_name=test_fold +"\\"+file_id+"_metrics_plot.jpg"
# #plt.rcParams["figure.autolayout"] = True
# fig, ax = plt.subplots(4, 2, figsize=(20,40))
#
# ax[0, 0].plot(SNR, 'o')
# ax[0, 0].set_title('Signal-to-noise ratio',fontsize='large')
# ax[0,0].set_xticks(np.arange(0, 9))
# ax[0,0].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
# ax[0, 1].plot(RE, 'o')
# ax[0, 1].set_title('Renyi Entropy')
# ax[0,1].set_xticks(np.arange(0, 9))
# ax[0,1].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
# #ax[0, 2].boxplot(L2_inv_or_G)
# #ax[0, 2].set_title('L2 inv. sound vs original')
# #ax[0,2].set_xticks(np.arange(1, 9))
# #ax[0,2].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
# ax[1, 0].plot(L2, 'o')
# ax[1, 0].set_title('L2 norm IF')
# ax[1,0].set_xticks(np.arange(0, 9))
# ax[1,0].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
# ax[1, 1].plot(IAM,'o')
# ax[1, 1].set_title('Iats. Error')
# ax[1,1].set_xticks(np.arange(0, 9))
# ax[1,1].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
# #ax[1, 2].plot(KLD_inv_or_G)
# #ax[1, 2].set_title('KLD inv. sound vs original')
# #ax[1,2].set_xticks(np.arange(1, 9))
# #ax[1,2].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
# ax[2, 0].plot(SDR_original, 'o')
# ax[2, 0].set_title('SDR original signal')
# ax[2,0].set_xticks(np.arange(0, 9))
# ax[2,0].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
# ax[2, 1].plot(SDR_noise,'o')
# ax[2, 1].set_title('SDR signal + noise')
# ax[2, 1].set_xticks(np.arange(0, 9))
# ax[2, 1].set_xticklabels(['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'], rotation=45)
# ax[3, 0].plot(IMED_original, 'o')
# ax[3, 0].set_title('IMED original signal')
# ax[3, 0].set_xticks(np.arange(0, 9))
# ax[3, 0].set_xticklabels(
#     ['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],
#     rotation=45)
# ax[3, 1].plot(IMED_noise,'o')
# ax[3, 1].set_title('IMED signal + noise')
# ax[3, 1].set_xticks(np.arange(0, 9))
# ax[3, 1].set_xticklabels(
#     ['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],
#     rotation=45)
# fig.suptitle(file_id+" metrics", fontsize=30)
# plt.savefig(fig_name)


# #save plots for sisdr check
# fig_name=test_fold +'\\'+file_id+'_sisdr.jpg'
# #plt.rcParams["figure.autolayout"] = True
# fig, ax = plt.subplots(2, 2, figsize=(20,40))
# ax[0,0].plot(NORM_S_1,'ro', label="norm orig.")
# ax[0,0].plot(NORM_DIFF_1,'bx', label="norm diff")
# ax[0,0].set_title("Norms sisdr1")
# ax[0, 0].set_xticks(np.arange(0, 9))
# ax[0, 0].set_xticklabels(
#     ['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],
#     rotation=45)
# ax[0,0].legend(loc="upper right")
# ax[0,1].plot(RATIO_1, 'bo', label="ratio")
# ax[0,1].plot(np.ones((9,1)), 'r-')
# ax[0,1].set_title("Ratio sisdr 1")
# ax[0, 1].set_xticks(np.arange(0, 9))
# ax[0, 1].set_xticklabels(
#     ['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],
#     rotation=45)
# ax[0,1].legend(loc="upper right")
# ax[1,0].plot(NORM_S_2,'ro', label="norm sign")
# ax[1,0].plot(NORM_DIFF_2,'bx', label="Norm diff.")
# ax[1,0].set_title("Norms sisdr2")
# ax[1, 0].set_xticks(np.arange(0, 9))
# ax[1, 0].set_xticklabels(
#     ['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],
#     rotation=45)
# ax[1,0].legend(loc="upper right")
# ax[1,1].plot(RATIO_2, 'bo', label="norm ratio")
# ax[1,1].plot(np.ones((9,1)), 'r-')
# ax[1,1].set_title("Ratio sisdr 2")
# ax[1, 1].set_xticks(np.arange(0, 9))
# ax[1, 1].set_xticklabels(
#     ['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],
#     rotation=45)
# ax[1,1].legend(loc="upper right")
# fig.suptitle(file_id+" sisdr", fontsize=30)
# plt.savefig(fig_name)

#save metric plots
fig_name=test_fold +"\\"+file_id+"_metrics_plot.jpg"
#plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots(5, 2, figsize=(20,40))

ax[0, 0].plot(SDR_original, 'o')
ax[0, 0].set_title('SDR original',fontsize='large')
ax[0,0].set_xticks(np.arange(0, 9))
ax[0,0].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
ax[0, 1].plot(SDR_noise, 'o')
ax[0, 1].set_title('SDR noise')
ax[0,1].set_xticks(np.arange(0, 9))
ax[0,1].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
ax[1, 0].plot(ISR_original, 'o')
ax[1, 0].set_title('ISR original')
ax[1,0].set_xticks(np.arange(0, 9))
ax[1,0].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
ax[1, 1].plot(ISR_noise, 'o')
ax[1, 1].set_title('ISR noise')
ax[1,1].set_xticks(np.arange(0, 9))
ax[1,1].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
ax[2, 0].plot(SISDR_original, 'o')
ax[2, 0].set_title('SISDR original signal')
ax[2,0].set_xticks(np.arange(0, 9))
ax[2,0].set_xticklabels(['Original','Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
ax[2, 1].plot(SISDR_noise, 'o')
ax[2, 1].set_title('SISDR signal + noise')
ax[2, 1].set_xticks(np.arange(0, 9))
ax[2, 1].set_xticklabels(['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'], rotation=45)
ax[3, 0].plot(SAR_original, 'o')
ax[3, 0].set_title('SAR original signal')
ax[3, 0].set_xticks(np.arange(0, 9))
ax[3, 0].set_xticklabels(
    ['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],
    rotation=45)
ax[3, 1].plot(SAR_noise, 'o')
ax[3, 1].set_title('SAR signal + noise')
ax[3, 1].set_xticks(np.arange(0, 9))
ax[3, 1].set_xticklabels(
    ['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],
    rotation=45)
ax[4, 0].plot(STOI_original, 'o')
ax[4, 0].set_title('STOI original signal')
ax[4, 0].set_xticks(np.arange(0, 9))
ax[4, 0].set_xticklabels(
    ['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],
    rotation=45)
ax[4, 1].plot(STOI_noise,'o')
ax[4, 1].set_title('STOI signal + noise')
ax[4, 1].set_xticks(np.arange(0, 9))
ax[4, 1].set_xticklabels(
    ['Original', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],
    rotation=45)
fig.suptitle(file_id+" metrics", fontsize=30)
plt.savefig(fig_name)