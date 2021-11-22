""""
22/11/2021
Author: Virginia Listanti

This script create a Dataset for the If experiment
"""

import SignalProc
import IF as IFreq
import numpy as np
from numpy.linalg import norm
#sfrom scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import os
from scipy import optimize
import scipy.special as spec
import wavio
import csv
from scipy.special import kl_div

def set_phase_fun(signal_id,T):
    """
    Utility function to manage the instantaneous phase function

    signal_id: type of signal
    T= lenght signal
    phi initial phase. DEFAULT phi=0

    """
    if signal_id=="pure_tone":
        omega=2000
        phi_t= lambda t:  2. * np.pi * omega * t

    elif signal_id=="exponential_downchirp":
        omega_1=500
        omega_0=2000
        alpha=(omega_1/omega_0)**(1/T)
        phi_t = lambda t: 2 * np.pi * omega_0 * ((alpha ** t - 1) / np.log(alpha))

    elif signal_id=="exponential_upchirp":
        omega_1=2000
        omega_0=500
        alpha=(omega_1/omega_0)**(1/T)
        phi_t = lambda t:  2 * np.pi * omega_0 * ((alpha ** t - 1) / np.log(alpha))


    elif signal_id=="linear_downchirp":
        omega_1=500
        omega_0=2000
        c=(omega_1-omega_0)/T
        phi_t = lambda t: 2 * np.pi * (0.5 * c * t ** 2 + omega_0 * t)

    elif signal_id=="linear_upchirp":
        omega_1=2000
        omega_0=500
        c=(omega_1-omega_0)/T
        phi_t = lambda t: 2 * np.pi * (0.5 * c * t ** 2 + omega_0 * t)


    else:
        print("ERROR SIGNAL ID NOT CONSISTENT WITH THE IF WE CAN HANDLE")
        print('signal_id: ', signal_id)
    return phi_t

##################################### MAIN ###################################################
main_dir="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals"
T=5 #signal duration
A=1 #amplitute
phi=0 #initial phase, usually 0
samplerate=16000
t = np.linspace(0., T, samplerate*T,endpoint=False) #discretised time vector



for sign_id in os.listdir(main_dir):
    #skipping the guide and fake kiwi syllables for now
    if sign_id.endswith(".txt"):
        continue
    if sign_id=="fake_kiwi_syllables":
        continue

    print("Generate dataset for ", sign_id)

    #set directroy path
    base_dir=main_dir+"\\"+sign_id+"\\Base_Dataset_2"
    dataset_dir=main_dir+"\\"+sign_id+"\\Dataset_2"

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    #set phase function
    phase_fun=set_phase_fun(sign_id, T)
    s1=A*np.sin(phi+phase_fun(t))
    std_sig=np.std(s1)
    coeff=np.array([0.0, 0.25, 0.5, 0.75,1.0,2.0,4.0,6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])*std_sig
    noise_levels=len(coeff)

    for k in range(noise_levels):
        mean = 0
        var=1
        w = np.std(s1) * np.random.normal(mean, var, (np.shape(t)))
        sig1 = s1 + coeff[k] * w

        #save sample in Base dataset
        if k<10:
            tag=str('0')+str(k)
        else:
            tag=str(k)
        save_path=base_dir+"\\"+sign_id+"_"+tag+".wav"
        wavio.write(save_path, sig1, samplerate, sampwidth=2)
        if k==0:
            continue
        level_dir = dataset_dir + "\\" + sign_id + "_" + str(k)

        if not os.path.exists(level_dir):
            os.mkdir(level_dir)

        #generate and save noise samples
        j=0
        while j<100:
            print('Sample ', j)

            if j<10:
                sample_tag=str(0)+str(j)
            else:
                sample_tag=str(j)
            aid_file=level_dir + "\\sample_"+sample_tag+".wav"

            wavio.write(aid_file, sig1, samplerate, sampwidth=2)

            #generate next sample
            w = np.random.normal(mean, var, (np.shape(t)))
            sig1 = s1 + coeff[k] * w

            j+=1


