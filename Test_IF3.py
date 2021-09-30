#24/08/2021
# Author: Virginia Listanti
#Script for Instantaneous frequency estimation

#WHAT IS NEEDED TO CALL ECURVE?
#INPUT:
#      - TFR: matrix with spectrogram. Note: #columns=time, #rows=freq
#      - freqarray: array with discretized frequencies
#      - wopt: parameters needed byt the function. At the momenth this is fixed but need review

#TO Do: Define wopt (this requires REVIEW)


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

# window_width=1024
# incr=256
# window="Hann"
#
#
# reassignment=False
# sp=SignalProc.SignalProc(window_width,incr)
# main_dir="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction"
# test_fold="signal_test"
#
# t=np.arange(1000)

def Renyi_Entropy(A, order=3):
    #A=np.abs(A)
    #[M,N]=np.shape(A)
    R_E= (1/(1-order))*np.log2(np.sum(A**order)/np.sum(A))
    return R_E

def Kul_Lieb_Div(x,y):

    if len(x)!=len(y):
        print("Dimensions not consistent for KLD")
        return
    x=x**2
    x/=np.linalg.norm(x)
    y = y ** 2
    y /= np.linalg.norm(y)
    #L=len(x)
    KLD=0
    for i in range(len(x)):
        if y[i]!=0 and x[i]!=0:
            KLD+=x[i]*np.log2(x[i]/y[i])
    return KLD

Test_List=["Test_03", "Test_04"]
samplerate = 8000
T = 5
# A= np.iinfo(np.int16).max
t = np.linspace(0., T, samplerate * T, endpoint=False)
test_dir="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Toy signals"
window="Hann"
for test_id in Test_List:
    if test_id=="Test_03":
        sgType = 'Reassigned'
        sgScale = 'Linear'
    else:
        sgType = 'Standard'
        sgScale = 'Mel Frequency'

    for test_fold in os.listdir(test_dir):
        if test_fold == "fake_kiwi_syllables":
            continue
        elif test_fold == "linear_downchirp":
            id_file="linear_downchirp"
            omega_1=500
            omega_0=2000
            c=(omega_1-omega_0)/T
            phi=0
            phi_t=2*np.pi*(0.5*c*t**2+omega_0*t)
            A=1
            inst_freq_fun=lambda x: omega_0+c*x
        elif test_fold == "linear_upchirp":
            id_file="linear_upchirp"
            omega_1=2000
            omega_0=500
            c=(omega_1-omega_0)/T
            phi=0
            A=1
            phi_t=phi+2*np.pi*(0.5*c*t**2+omega_0*t)
            inst_freq_fun=lambda x: omega_0+c*x
        elif test_fold == "pure_tone":
            id_file="pure_tone"
            A=1
            omega=1000
            phi=0
            phi_t= phi + 2. * np.pi * omega * t
            inst_freq_fun= lambda t: 100*np.ones((np.shape(t)))
        elif test_fold == "Test_guide":
            continue
        elif test_fold == "exponential_downchirp":
            # window_width = 1024
            # incr = 256
            id_file = "exponential_downchirp"
            omega_1 = 500
            omega_0 = 2000
            k = (omega_1 / omega_0) ** (1 / T)
            phi = 0
            phi_t = phi + 2 * np.pi * omega_0 * ((k ** t - 1) / np.log(k))
            A = 1
            inst_freq_fun = lambda x: omega_0 * k ** x
        elif test_fold == "exponential_upchirp":
            # window_width = 1024
            # incr = 256
            id_file = "exponential_upchirp"
            omega_1 = 2000
            omega_0 = 500
            k = (omega_1 / omega_0) ** (1 / T)
            phi = 0
            phi_t = phi + 2 * np.pi * omega_0 * ((k ** t - 1) / np.log(k))
            A = 1
            inst_freq_fun = lambda x: omega_0 * k ** x
        else:
            continue

        os.mkdir(test_dir+'\\'+test_fold+"\\"+test_id)
        s1 = A * np.sin(phi + phi_t)
        std_sig = np.std(s1)
        coeff = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0]) * std_sig

        ################################### SAVE SIGNALS WAVE FORM ##################################

        #noise
        mean=0
        var=1
        #var=np.std(s1)
        w=np.random.normal(mean, var, (np.shape(t)))
        #w=np.std(s1)*np.random.normal(mean, var, (np.shape(t)))
        for i in range(len(coeff)):
            #sig1 = A*np.sin(phi+phi_t + coeff[i]*w)
            sig1= s1+ coeff[i]*w
            aid_file=test_dir+"\\"+test_fold+"\\"+test_id+"\\"+id_file+"_"+str(i)+".wav"
            wavio.write(aid_file,sig1, samplerate,sampwidth=2)

################################# FIND OPTIMAL WINDOW AND OVERLAP PARM ######################################
#
        win=np.array([64,128,256,512,1024,2048])
        hop_perc=np.array([0.25, 0.5,0.75])
        file_name=id_file+"_0.wav"
        data_file = test_dir + "\\" + test_fold+"\\"+test_id +"\\"+file_name

        opt=np.Inf
        opt_param={"win_len":[], "hop":[]}

        #store values into .csv file
        #fieldnames=['window_width','incr','n. columns', 'measure']
        fieldnames=['window_width','incr','spec dim', 'measure']
        csv_filename=test_dir+'\\'+test_fold+"\\"+test_id + '\\find_optimal_parameters.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        for i in range(len(win)):
            for j in range(len(hop_perc)):
                window_width=int(win[i])
                incr=int(win[i]*hop_perc[j])
                print(window_width, incr)
                IF = IFreq.IF(method=2, pars=[1, 1])
                sp = SignalProc.SignalProc(window_width, incr)
                sp.readWav(data_file)
                fs = sp.sampleRate
                TFR = sp.spectrogram(window_width, incr, window,sgType=sgType,sgScale=sgScale)
                TFR = TFR.T
                print("spec dim", np.shape(TFR))
                # savemat'C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\test_signal.mat',{'TFR':TFR})
                fstep = (fs / 2) / np.shape(TFR)[0]
                freqarr = np.arange(fstep, fs / 2 + fstep, fstep)

                wopt = [fs, window_width]  # this neeeds review
                tfsupp, _,_ = IF.ecurve(TFR, freqarr, wopt)

                #inst_freq = omega * np.ones((np.shape(tfsupp[0,:])))
                inst_freq=inst_freq_fun(np.linspace(0,T,np.shape(tfsupp[0,:])[0]))
                #print("tfsupp", np.shape(tfsupp[0,:]))
                #print("inst_feq", np.shape(inst_freq))
                #measure2check=norm(tfsupp[0,:]-inst_freq, ord=2)/np.shape(tfsupp[0,:])[0]
                measure2check = norm(tfsupp[0, :] - inst_freq, ord=2) / (np.shape(TFR)[0]*np.shape(TFR)[1])

                with open(csv_filename, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(
                        {'window_width': window_width, 'incr': incr,'spec dim':np.shape(TFR)[0]*np.shape(TFR)[1], 'measure': measure2check})

                if measure2check<opt:
                    print("optimal parameters updated:", opt_param)
                    print(norm(tfsupp[0, :] - inst_freq, ord=2))
                    opt= measure2check
                    opt_param["win_len"]=window_width
                    opt_param["hop"] = incr


                del TFR, fstep, freqarr, wopt, tfsupp, window_width, incr, sp, IF, measure2check

        print("optimal parameters \n", opt_param)
        window_width=opt_param["win_len"]
        incr=opt_param["hop"]

# ######################################## SAVE BASELINE VALUES ################################################
        file_name=id_file+"_0.wav"
        data_file = test_dir + "\\" + test_fold+"\\"+test_id +"\\"+file_name

        IF = IFreq.IF(method=2, pars=[1, 1])
        sp = SignalProc.SignalProc(window_width, incr)
        sp.readWav(data_file)
        fs = sp.sampleRate
        TFR = sp.spectrogram(window_width, incr, window,sgType=sgType,sgScale=sgScale)
        TFR = TFR.T

        fstep = (fs / 2) / np.shape(TFR)[0]
        freqarr = np.arange(fstep, fs / 2 + fstep, fstep)

        wopt = [fs, window_width]  # this neeeds review
        tfsupp, _,_ = IF.ecurve(TFR, freqarr, wopt)
        inst_freq=inst_freq_fun(np.linspace(0,T,np.shape(tfsupp[0,:])[0]))
        #inst_freq = omega * np.ones((np.shape(tfsupp[0,:])))

        ##reconstruct signal from spectrogram
        # update wopt and wp

        wp=IFreq.Wp(incr,fs)
        wopt=IFreq.Wopt(fs,wp,0,fs/2)

        # function to reconstruct official Instantaneous Frequency

        #iamp,iphi,ifreq = IF.rectfr(tfsupp,TFR,freqarr,wopt,'ridge')

        #s1_recostructed=iamp*np.cos(iphi)
        #
        #invert spectrogram
        s1_data=sp.data
        s1_inverted=sp.invertSpectrogram(TFR.T,window_width=window_width,incr=incr,window=window)
        s1_inverted=s1_inverted[int(np.floor(window_width/2-(samplerate*T-len(s1_inverted)))):-int(np.floor(window_width/2))]
        s1_inverted_or=s1_inverted/(np.ptp(s1_inverted)/np.ptp(s1))
        s1_inverted_data=s1_inverted/(np.ptp(s1_inverted)/np.ptp(s1_data))

        #calculate baseline measures
        std_s1=np.std(s1)
        #IF
        L2_IF=norm(tfsupp[0,:]-inst_freq, ord=2)
        KLD_IF=Kul_Lieb_Div(inst_freq,tfsupp[0,:])

        L2_inv=norm(s1[int(np.floor(window_width/2)):-int(np.floor(window_width/2))]-s1_inverted, ord=2)
        KLD_inv=Kul_Lieb_Div(s1[int(np.floor(window_width/2)):-int(np.floor(window_width/2))],s1_inverted)
        L2_inv_data=norm(s1_data[int(np.floor(window_width/2)):-int(np.floor(window_width/2))]-s1_inverted_data, ord=2)
        KLD_inv_data=Kul_Lieb_Div(s1_data[int(np.floor(window_width/2)):-int(np.floor(window_width/2))],s1_inverted_data)
        #Renyi entropy
        RE_0=Renyi_Entropy(TFR)
        fieldnames=['std','L2 if', 'KLD if','L2 inverted','KLD inverted','L2 inverted data','KLD inverted data','Renyi Entropy']
        with open(test_dir+'\\'+test_fold+"\\"+test_id  + '\\baseline_values.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'std': std_s1, 'L2 if': L2_IF, 'KLD if': KLD_IF,
                             'L2 inverted':L2_inv,'KLD inverted':KLD_inv,
                             'L2 inverted data':L2_inv_data, 'KLD inverted data':KLD_inv_data,
                             'Renyi Entropy': RE_0 })

        del fieldnames, sp, IF,L2_inv_data, L2_inv, KLD_inv, KLD_inv_data, RE_0


# ####################################### NOISE LEVELS  MEASURE#################################


        fieldnames=['L2 if', 'KLD if','L2 inverted','KLD inverted','L2 inv or','KLD inv or','Renyi Entropy']
        file_name=id_file+"_0.wav"

        save_directory=test_dir + "\\" + test_fold+"\\Dataset"
        data_file = test_dir + "\\" + test_fold+'\\'+test_id+"\\"+file_name

        sp = SignalProc.SignalProc(window_width, incr)
        sp.readWav(data_file)
        s1=sp.data
        fs = sp.sampleRate
        TFR_0 = sp.spectrogram(window_width, incr, window, sgType=sgType,sgScale=sgScale)
        TFR_0 = TFR_0.T
        del sp
        mean=0
        var=1

        L2_IF_G=np.zeros((100,len(coeff)-1))
        KLD_IF_G=np.zeros((100,len(coeff)-1))
        L2_inv_G=np.zeros((100,len(coeff)-1))
        KLD_inv_G=np.zeros((100,len(coeff)-1))
        L2_inv_or_G=np.zeros((100,len(coeff)-1))
        KLD_inv_or_G=np.zeros((100,len(coeff)-1))
        RE_0_G=np.zeros((100,len(coeff)-1))


        for dir in os.listdir(save_directory):
            i=int(dir[-1])
            print('\n Noise level: ', i)
            csvfilename=test_dir + '\\' + test_fold + "\\"+test_id +'\\noise_level_'+str(dir[-1])+'.csv'
            with open(csvfilename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            # level_dir=test_dir + "\\" + test_fold + "\\"+id_file+"_" + str(i)
            # os.mkdir(level_dir)
            k=0
            for file in os.listdir(save_directory+'\\'+dir):
                print('Sample ', file)
                #w = np.random.normal(mean, var, (np.shape(t)))
                # aid_file=level_dir + "\\sample_"+str(k)+".wav"
                # sig1=s1+coeff[i]*w
                # wavio.write(aid_file, sig1, samplerate, sampwidth=2)
                IF = IFreq.IF(method=2, pars=[1, 1])
                sp = SignalProc.SignalProc(window_width, incr)
                sp.readWav(save_directory+'\\'+dir+'\\'+file)
                sig1 = sp.data
                fs = sp.sampleRate
                TFR = sp.spectrogram(window_width, incr, window,sgType=sgType,sgScale=sgScale)
                TFR2 = TFR.T
                fstep = (fs / 2) / np.shape(TFR2)[0]
                freqarr = np.arange(fstep, fs / 2 + fstep, fstep)

                wopt = [fs, window_width]  # this neeeds review
                tfsupp, _, _ = IF.ecurve(TFR2, freqarr, wopt)
                inst_freq = inst_freq_fun(np.linspace(0, T, np.shape(tfsupp[0, :])[0]))
                #inst_freq = omega * np.ones((np.shape(tfsupp[0, :])))

                s1_inverted = sp.invertSpectrogram(TFR, window_width=window_width, incr=incr, window=window)
                s1_inverted = s1_inverted[int(np.floor(window_width/2-(samplerate*T-len(s1_inverted)))):-int(np.floor(window_width/2))]
                s1_inverted_data=s1_inverted/(np.ptp(s1_inverted)/np.ptp(sig1))
                s1_inverted_or = s1_inverted / (np.ptp(s1_inverted) / np.ptp(s1))
                #s1_inverted_data=s1_inverted/(np.ptp(s1_inverted)/np.ptp(sig1_data))
                #t_inverted = np.linspace(0, T, len(s1_inverted))
                #s1_inverted = np.interp(t, t_inverted, s1_inverted)

                # IF
                L2_IF = norm(tfsupp[0, :] - inst_freq, ord=2)
                L2_IF_G[k,i-1]=L2_IF
                KLD_IF = Kul_Lieb_Div( inst_freq,tfsupp[0, :])
                KLD_IF_G[k,i-1]=KLD_IF

                # inverted signal
                L2_inv = norm(sig1[int(np.floor(window_width/2)):-int(np.floor(window_width/2))] - s1_inverted_data, ord=2)
                L2_inv_G[k,i-1]=L2_inv
                KLD_inv = Kul_Lieb_Div( sig1[int(np.floor(window_width/2)):-int(np.floor(window_width/2))],s1_inverted_data)
                KLD_inv_G[k,i-1]=KLD_inv

                #inverted signal VS original
                L2_inv_or = norm(s1[int(np.floor(window_width/2)):-int(np.floor(window_width/2))] - s1_inverted_or, ord=2)
                L2_inv_or_G[k, i - 1] = L2_inv_or
                KLD_inv_or = Kul_Lieb_Div(s1[int(np.floor(window_width/2)):-int(np.floor(window_width/2))], s1_inverted_or)
                KLD_inv_or_G[k, i - 1] = KLD_inv_or

                # Renyi entropy
                RE_0 = Renyi_Entropy(TFR2)
                RE_0_G[k,i-1]=RE_0

                with open(csvfilename, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'L2 if': L2_IF, 'KLD if': KLD_IF,
                                     'L2 inverted': L2_inv, 'KLD inverted': KLD_inv,
                                     'L2 inv or':L2_inv_or, 'KLD inv or':KLD_inv_or,
                                     'Renyi Entropy': RE_0})

                #os.remove(aid_file)
                k+=1
                del IF, sp,  fs, TFR, fstep, freqarr, wopt, tfsupp, s1_inverted, L2_inv, L2_IF,
                KLD_IF, KLD_inv, RE_0, inst_freq, KLD_inv_or, L2_inv_or

            del csvfilename

        #save a csvfile per metric

        fieldnames2=["Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7", "Level 8"]
        csvfilename=test_dir + '\\' + test_fold  +"\\"+test_id + '\\noise_level_L2_IF.csv'
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames2)
            writer.writeheader()
            for i in range(100):
                writer.writerow({"Level 1":L2_IF_G[i][0], "Level 2":L2_IF_G[i][1], "Level 3":L2_IF_G[i][2],
                                 "Level 4":L2_IF_G[i][3], "Level 5":L2_IF_G[i][4], "Level 6":L2_IF_G[i][5],
                                 "Level 7": L2_IF_G[i][6],  "Level 8":L2_IF_G[i][7] })

        csvfilename=test_dir + '\\' + test_fold  + "\\"+test_id +'\\noise_level_L2_inv.csv'
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames2)
            writer.writeheader()
            for i in range(100):
                writer.writerow({"Level 1":L2_inv_G[i][0], "Level 2":L2_inv_G[i][1], "Level 3":L2_inv_G[i][2],
                                 "Level 4":L2_inv_G[i][3], "Level 5":L2_inv_G[i][4], "Level 6":L2_inv_G[i][5],
                                 "Level 7":L2_inv_G[i][6], "Level 8":L2_inv_G[i][7]})

        csvfilename=test_dir + '\\' + test_fold  + "\\"+test_id +'\\noise_level_L2_inv_original.csv'
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames2)
            writer.writeheader()
            for i in range(100):
                writer.writerow({"Level 1":L2_inv_or_G[i][0], "Level 2":L2_inv_or_G[i][1], "Level 3":L2_inv_or_G[i][2],
                                 "Level 4":L2_inv_or_G[i][3], "Level 5":L2_inv_or_G[i][4], "Level 6":L2_inv_or_G[i][5],
                                 "Level 7":L2_inv_or_G[i][6],"Level 8":L2_inv_or_G[i][7]})

        csvfilename=test_dir + '\\' + test_fold  + "\\"+test_id +'\\noise_level_KLD_IF.csv'
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames2)
            writer.writeheader()
            for i in range(100):
                writer.writerow({"Level 1":KLD_IF_G[i][0], "Level 2":KLD_IF_G[i][1], "Level 3":KLD_IF_G[i][2],
                                 "Level 4":KLD_IF_G[i][3], "Level 5":KLD_IF_G[i][4], "Level 6":KLD_IF_G[i][5],
                                 "Level 7":KLD_IF_G[i][6],"Level 8":KLD_IF_G[i][7]})

        csvfilename=test_dir + '\\' + test_fold  + "\\"+test_id +'\\noise_level_KLD_inv.csv'
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames2)
            writer.writeheader()
            for i in range(100):
                writer.writerow({"Level 1":KLD_inv_G[i][0], "Level 2":KLD_inv_G[i][1], "Level 3":KLD_inv_G[i][2],
                                 "Level 4":KLD_inv_G[i][3], "Level 5":KLD_inv_G[i][4], "Level 6":KLD_inv_G[i][5],
                                 "Level 7":KLD_inv_G[i][6],"Level 8":KLD_inv_G[i][7]})


        csvfilename=test_dir + '\\' + test_fold  + "\\"+test_id +'\\noise_level_KLD_inv_original.csv'
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames2)
            writer.writeheader()
            for i in range(100):
                writer.writerow({"Level 1":KLD_inv_or_G[i][0], "Level 2":KLD_inv_or_G[i][1], "Level 3":KLD_inv_or_G[i][2],
                                 "Level 4":KLD_inv_or_G[i][3], "Level 5":KLD_inv_or_G[i][4], "Level 6":KLD_inv_or_G[i][5],
                                 "Level 7":KLD_inv_or_G[i][6],"Level 8":KLD_inv_or_G[i][7]})

        csvfilename=test_dir + '\\' + test_fold  + "\\"+test_id +'\\noise_level_Renyi_Entropy.csv'
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames2)
            writer.writeheader()
            for i in range(100):
                writer.writerow({"Level 1":RE_0_G[i][0], "Level 2":RE_0_G[i][1], "Level 3":RE_0_G[i][2],
                                 "Level 4":RE_0_G[i][3], "Level 5":RE_0_G[i][4], "Level 6":RE_0_G[i][5],
                                 "Level 7":RE_0_G[i][6],"Level 8":RE_0_G[i][7]})

        ####################################### ANALISE DATA ##########################################################à
        #read .csv
        csvfilename=test_dir + '\\' + test_fold+ '\\'+test_id + '\\noise_level_L2_IF.csv'
        L2_IF_G=[]
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                L2_IF_G.append(row)

        L2_IF_G=np.array(L2_IF_G[1:][:]).astype('float')

        csvfilename=test_dir + '\\' + test_fold  + '\\'+test_id +'\\noise_level_L2_inv.csv'
        L2_inv_G=[]
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                L2_inv_G.append(row)

        L2_inv_G=np.array(L2_inv_G[1:][:]).astype('float')

        csvfilename=test_dir + '\\' + test_fold  + '\\'+test_id +'\\noise_level_L2_inv_original.csv'
        L2_inv_or_G=[]
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                L2_inv_or_G.append(row)

        L2_inv_or_G=np.array(L2_inv_or_G[1:][:]).astype('float')

        csvfilename = test_dir + '\\' + test_fold + '\\'+test_id +'\\noise_level_KLD_IF.csv'
        KLD_IF_G = []
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                KLD_IF_G.append(row)
        KLD_IF_G=np.array(KLD_IF_G[1:][:]).astype('float')

        csvfilename = test_dir + '\\' + test_fold + '\\'+test_id +'\\noise_level_KLD_inv.csv'
        KLD_inv_G = []
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                KLD_inv_G.append(row)
        KLD_inv_G=np.array(KLD_inv_G[1:][:]).astype('float')

        csvfilename = test_dir + '\\' + test_fold + '\\'+test_id +'\\noise_level_KLD_inv_original.csv'
        KLD_inv_or_G = []
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                KLD_inv_or_G.append(row)
        KLD_inv_or_G=np.array(KLD_inv_or_G[1:][:]).astype('float')

        csvfilename = test_dir + '\\' + test_fold + '\\'+test_id +'\\noise_level_Renyi_Entropy.csv'
        RE_0_G= []
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                RE_0_G.append(row)
        RE_0_G=np.array(RE_0_G[1:][:]).astype('float')

        #save plots
        fig_name=test_dir + '\\' + test_fold  + '\\'+test_id +'\\metrics_plot.jpg'
        #plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots(3, 3, figsize=(20,40))

        ax[0, 0].boxplot(L2_IF_G)
        ax[0, 0].set_title('L2 inst. freq.',fontsize='large')
        ax[0,0].set_xticks(np.arange(1, 9))
        ax[0,0].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
        ax[0, 1].boxplot(L2_inv_G)
        ax[0, 1].set_title('L2 inv. sound')
        ax[0,1].set_xticks(np.arange(1, 9))
        ax[0,1].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
        ax[0, 2].boxplot(L2_inv_or_G)
        ax[0, 2].set_title('L2 inv. sound vs original')
        ax[0,2].set_xticks(np.arange(1, 9))
        ax[0,2].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
        ax[1, 0].boxplot(KLD_IF_G)
        ax[1, 0].set_title('KLD inst. freq.')
        ax[1,0].set_xticks(np.arange(1, 9))
        ax[1,0].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
        ax[1, 1].boxplot(KLD_inv_G)
        ax[1, 1].set_title('KLD inv. sound')
        ax[1,1].set_xticks(np.arange(1, 9))
        ax[1,1].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
        ax[1, 2].boxplot(KLD_inv_or_G)
        ax[1, 2].set_title('KLD inv. sound vs original')
        ax[1,2].set_xticks(np.arange(1, 9))
        ax[1,2].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
        ax[2, 0].boxplot(RE_0_G)
        ax[2, 0].set_title('Renyi entropy')
        ax[2,0].set_xticks(np.arange(1, 9))
        ax[2,0].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8'],rotation=45)
        fig.suptitle(test_fold, fontsize=30)
        plt.savefig(fig_name)

        #save plots without last
        fig_name=test_dir + '\\' + test_fold  + '\\'+test_id +'\\metrics_plot2.jpg'
        #plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots(3, 3, figsize=(20,40))

        ax[0, 0].boxplot(L2_IF_G[:,:-1])
        ax[0, 0].set_title('L2 inst. freq.',fontsize='large')
        ax[0,0].set_xticks(np.arange(1, 8))
        ax[0,0].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7'],rotation=45)
        ax[0, 1].boxplot(L2_inv_G[:,:-1])
        ax[0, 1].set_title('L2 inv. sound')
        ax[0,1].set_xticks(np.arange(1, 8))
        ax[0,1].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7'],rotation=45)
        ax[0, 2].boxplot(L2_inv_or_G[:,:-1])
        ax[0, 2].set_title('L2 inv. sound vs original')
        ax[0,2].set_xticks(np.arange(1, 8))
        ax[0,2].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7'],rotation=45)
        ax[1, 0].boxplot(KLD_IF_G[:,:-1])
        ax[1, 0].set_title('KLD inst. freq.')
        ax[1,0].set_xticks(np.arange(1, 8))
        ax[1,0].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7'],rotation=45)
        ax[1, 1].boxplot(KLD_inv_G[:,:-1])
        ax[1, 1].set_title('KLD inv. sound')
        ax[1,1].set_xticks(np.arange(1, 8))
        ax[1,1].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7'],rotation=45)
        ax[1, 2].boxplot(KLD_inv_or_G[:,:-1])
        ax[1, 2].set_title('KLD inv. sound vs original')
        ax[1,2].set_xticks(np.arange(1, 8))
        ax[1,2].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7'],rotation=45)
        ax[2, 0].boxplot(RE_0_G[:,:-1])
        ax[2, 0].set_title('Renyi entropy')
        ax[2,0].set_xticks(np.arange(1, 8))
        ax[2,0].set_xticklabels(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7'],rotation=45)
        fig.suptitle(test_fold, fontsize=30)
        plt.savefig(fig_name)




################################################################################################
# sig1=A*np.sin(2*np.pi*omega*t+phi +w) #check
# wavio.write(test_dir + "\\pure_tone_test"  + ".wav", sig1.astype(np.int16), samplerate, scale='dtype-limits',
#             sampwidth=2)
#
# plt.plot(t[0:1000],sig1[0:1000])
# plt.show()


# #noise
# mean=0
# var=1
# #var=np.std(s1)
# w=var=np.std(s2)*np.random.normal(mean, var, (np.shape(t)))
# for i in range(len(coeff)):
#     sig2=A*np.sin(phi_t)+coeff[i]*w
#     wavio.write(test_dir+"\\linear_upchirp_"+str(i)+".wav", sig2.astype(np.int16), samplerate,scale='dtype-limits',sampwidth=2)
#

#
# for i in range(len(coeff)):
#     sig3=A*np.sin(phi+2*np.pi*(0.5*c*t**2+omega_0*t))+coeff[i]*w
#     wavio.write(test_dir+"\\linear_downchirp_"+str(i)+".wav", sig3.astype(np.int16), samplerate,scale='dtype-limits',sampwidth=2)
#

#
# for i in range(len(coeff)):
#     sig4=A*np.sin(phi+2*np.pi*omega_0*((k**t-1)/np.log(k)))+coeff[i]*w
#     wavio.write(test_dir+"\\exponential_upchirp_"+str(i)+".wav", sig4.astype(np.int16), samplerate,scale='dtype-limits',sampwidth=2)
#
# #exponential down-chirp
# omega_1=500
# omega_0=2000
# k=(omega_1/omega_0)**(1/T)
# phi=0
#
# for i in range(len(coeff)):
#     sig5=A*np.sin(phi+2*np.pi*omega_0*((k**t-1)/np.log(k)))+coeff[i]*w
#     wavio.write(test_dir+"\\exponential_downchirp_"+str(i)+".wav", sig5.astype(np.int16), samplerate,scale='dtype-limits',sampwidth=2)

################################################ EXPERIMENT ######################################################
# window_width=1024
# incr=256
# window="Hann"
#
#
# reassignment=False
# sp=SignalProc.SignalProc(window_width,incr)
# main_dir="C:\\Users\\Virginia\\Documents\\Work\\IF_extraction"
# test_fold="signal_test"


# for f in os.listdir(main_dir):
#     if f.endswith('.wav'):
#         file_name = f
#         data_file = main_dir + "\\" + file_name
#         print(file_name)
#         if 'song' in data_file:
#             song_flag = True
#         else:
#             song_flag=False
#         sp.readWav(data_file)
#         fs = sp.sampleRate
#         IF = IFreq.IF(method=1)
#         #FOR JULIUS: check if window is Hann
#         if reassignment:
#             TFR = sp.spectrogram(window_width, incr, window, sgType='Reassigned')
#         else:
#             TFR=sp.spectrogram(window_width,incr,window)
#
#         # REMEMBER: we need to transpose
#         TFR=TFR.T
#         #savemat'C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\test_signal.mat',{'TFR':TFR})
#         fstep=(fs/2)/np.shape(TFR)[0]
#         freqarr=np.arange(fstep,fs/2+fstep,fstep)
#
#
#         wopt=[fs,window_width] #this neeeds review
#         tfsupp,ecinfo, Skel=IF.ecurve(TFR,freqarr,wopt) # <= This is the function we need
#
#         ########################## update wopt and wp
#
#         wp=IFreq.Wp(incr,fs)
#         wopt=IFreq.Wopt(fs,wp,0,fs/2)
#
#         # function to reconstruct official Instantaneous Frequency
#         #NOTE: at the moment this seems to not be needed
#         iamp,iphi,ifreq = IF.rectfr(tfsupp,TFR,freqarr,wopt)
#
#         fig_name=main_dir+"\\"+test_fold+"\\"+file_name[:-3]+"jpg"
#
#         if song_flag:
#             plt.rcParams["figure.autolayout"] = True
#             fig, ax = plt.subplots(3, 1, figsize=(28,21), sharex=True)
#             ax[0].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
#             x = np.array(range(np.shape(TFR)[1]))
#             ax[0].plot(x, (ifreq / fstep).T, linewidth=1, color='red')
#             ax[0].plot(x, tfsupp[0, :] / fstep, linewidth=1, color='w')
#             ax[1].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
#             ax[2].plot(ifreq,color='red')
#             ax[2].plot(tfsupp[0,:], color='green')
#         else:
#             plt.rcParams["figure.autolayout"] = True
#             fig, ax = plt.subplots(1, 3, figsize=(28, 11.5), sharex=True)
#             ax[0].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
#             x = np.array(range(np.shape(TFR)[1]))
#             ax[0].plot(x, (ifreq / fstep).T, linewidth=1, color='red')
#             ax[0].plot(x, tfsupp[0, :] / fstep, linewidth=1, color='w')
#             ax[1].imshow(np.flipud(TFR), extent=[0, np.shape(TFR)[1], 0, np.shape(TFR)[0]], aspect='auto')
#             ax[2].plot(ifreq, color='red')
#             ax[2].plot(tfsupp[0, :], color='green')
#         plt.savefig(fig_name)

# #save inverted sound
# window_width=1024
# incr=256
# window='Hann'
# for file in os.listdir(test_dir+'\\'+test_fold):
#     if file.endswith('.wav'):
#         data_file = test_dir+'\\'+test_fold + "\\" + file
#         sp = SignalProc.SignalProc(window_width, incr)
#         sp.readWav(data_file)
#         fs = sp.sampleRate
#         TFR = sp.spectrogram(window_width, incr, window)
#         s_inverted = sp.invertSpectrogram(TFR, window_width=window_width, incr=incr, window=window)
#         wavio.write(data_file[:-4]+'_inv.wav', s_inverted, samplerate, sampwidth=2)

