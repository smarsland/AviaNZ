"""
28/02/2022
Author: Virginia Listanti

This script purpose is to analyse data produced by IF experiment
Here we want to produce plot to compare results obtained with different optimizations techniques
"""

import SignalProc
import IF as IFreq
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import ast

def sort_test_list(s):
    "This script sort Test_list using nubers"

    test_num=[]
    for element in s:
        test_num.append(element[5:])

    test_num.sort(key=int)

    new_list =[]

    for element in test_num:
        new_list.append('Test_'+element)
    return new_list


def plot_parameters(t_list, t_result_dir, t_analysis_fold, s_id):
    """
    This function counts and plot the parameters used in t_list
    """

    # Spectrogram parameters
    win_list = {'256': 0, '512': 0, '1024': 0, '2048': 0}
    incr_list = {'25': 0, '51': 0, '64': 0, '102': 0, '128': 0, '192': 0, '204': 0, '230': 0, '256': 0, '384': 0,
                 '460': 0, '512': 0, '768': 0, '921': 0, '512': 0, '1024': 0, '1536': 0, '1843': 0}
    win_type_list = {'Hann': 0, 'Parzen': 0, 'Welch': 0, 'Hamming': 0, 'Blackman': 0}

    # If Extraction parameters
    alpha_list = {'0.0': 0, '0.5': 0, '1.0': 0, '2.5': 0, '5.0': 0, '7.5': 0, '10.0': 0}
    beta_list = {'0.0': 0, '0.5': 0, '1.0': 0, '2.5': 0, '5.0': 0, '7.5': 0, '10.0': 0}

    # mel bins options
    mel_bins = {'None': 0, '64': 0, '128': 0, '256': 0}  # only power of 2

    # cunting tests
    n = 0
    for test_id in t_list:
        # open TFR_inf file
        # save folder path
        folder_path = t_result_dir + '\\' + test_id

        if not s_id in os.listdir(folder_path):
            # if not test for signal_id continue
            continue

        # open TFR_info_path
        dictionary_path = folder_path + '\\' + s_id + '\\' + 'Optimal_parameters.txt'

        with open(dictionary_path) as f:
            data_param = f.read()

        # turn raw data into dictionary
        opt_param = ast.literal_eval(data_param)

        win_list[str(opt_param['win_len'])] += 1
        incr_list[str(opt_param['hop'])] += 1
        win_type_list[str(opt_param['window_type'])] += 1
        mel_bins[str(opt_param['mel_num'])] += 1
        alpha_list[str(opt_param['alpha'])] += 1
        beta_list[str(opt_param['beta'])] += 1

        n += 1

        del opt_param

    # save plots
    fig_name = t_analysis_fold + '\\' + 'optimal_parametes.jpg'
    # plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots(3, 2, figsize=(20, 20))

    # bar_width = 0.2

    index1 = np.arange(len(win_list))
    ax[0, 0].set_title('Win length', fontsize=25)
    ax[0, 0].bar(index1, win_list.values())
    ax[0, 0].set_xticks(index1, labels=win_list.keys(), fontsize=20)
    ax[0, 0].tick_params(axis='y', labelsize=20)
    # ax[0, 0].set_yticks(fontsize=20)
    # ax[0].set_xticks(index1)

    index2 = np.arange(len(incr_list))
    ax[0, 1].set_title('Incr length', fontsize=25)
    ax[0, 1].bar(index2, incr_list.values())
    ax[0, 1].set_xticks(index2, labels=incr_list.keys(), fontsize=10)
    ax[0, 1].tick_params(axis='y', labelsize=20)

    index3 = np.arange(len(win_type_list))
    ax[1, 0].set_title('Win type', fontsize=25)
    ax[1, 0].bar(index3, win_type_list.values())
    ax[1, 0].set_xticks(index3, labels=win_type_list.keys(), fontsize=18)
    ax[1, 0].tick_params(axis='y', labelsize=20)
    # ax[0].set_xticks(index1)

    index4 = np.arange(len(mel_bins))
    ax[1, 1].set_title('mel bins', fontsize=25)
    ax[1, 1].bar(index4, mel_bins.values())
    ax[1, 1].set_xticks(index4, labels=mel_bins.keys(), fontsize=20)
    ax[1, 1].tick_params(axis='y', labelsize=20)

    index5 = np.arange(len(alpha_list))
    ax[2, 0].set_title('Alpha', fontsize=25)
    ax[2, 0].bar(index5, alpha_list.values())
    ax[2, 0].set_xticks(index5, labels=alpha_list.keys(), fontsize=20)
    ax[2, 0].tick_params(axis='y', labelsize=20)
    # ax[0].set_xticks(index1)

    index6 = np.arange(len(beta_list))
    ax[2, 1].set_title('Beta', fontsize=25)
    ax[2, 1].bar(index6, beta_list.values())
    ax[2, 1].set_xticks(index6, labels=beta_list.keys(), fontsize=20)
    ax[2, 1].tick_params(axis='y', labelsize=20)

    fig.suptitle(signal_id + '\n test number = ' + str(n), fontsize=40)
    plt.savefig(fig_name)

    return

# signal type we are analysing
#signal_id='pure_tone'
#signal_id = 'linear_upchirp'
#signal_id = 'linear_downchirp'
# signal_id = 'exponential_upchirp'
signal_id = 'exponential_downchirp'
start_index = 0


#analysis for test folder
test_result_dir = "C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Test_Results"
test_analysis_dir= "C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Results analysis\\Optimization_methods\\Group1"

#create signal folder
test_analysis_fold = test_analysis_dir + '\\' + signal_id
if not os.path.exists(test_analysis_fold):
    os.mkdir(test_analysis_fold)
    
test_list = os.listdir(test_result_dir)

test_list = sort_test_list(test_list)[start_index:start_index+6]

# Count and plot parameters chosen
plot_parameters(test_list, test_result_dir, test_analysis_fold, signal_id)

#read baselive values for pure signal
#read baseline_values .csv
baseline=[]
for test_id in test_list:
    if not signal_id in os.listdir(test_result_dir+'\\'+test_id):
        # if not test for signal_id continue
        continue
    csvfilename = test_result_dir+ '\\'+test_id + '\\' + signal_id + '\\baseline_values.csv'
    with open(csvfilename) as csvfile:
        # open file as csv file
        csvReader = csv.reader(csvfile)
        # loop over rows
        aid_baseline = []
        for row in csvReader:
            aid_baseline.append(row)

    #aid_baseline=np.array(aid_baseline[1:][:]).astype('float')
    baseline.append(aid_baseline[1])

baseline=np.array(baseline[:][:]).astype('float')

level_list = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8', 'Level 9',
              'Level 10', 'Level 11', 'Level 12', 'Level 13', 'Level 14']

#signal baseline metrics
#save plots
fig_name1 = test_analysis_fold +'\\signal_baseline_metrics.jpg'

fig, ax = plt.subplots(2, 6, figsize=(100,40))
col_counter = 0
for test_id in test_list:
    if not signal_id in os.listdir(test_result_dir+'\\'+test_id):
        # if not test for signal_id continue
        continue
    #read SNR .csv
    csvfilename = test_result_dir+ '\\'+test_id + '\\' + signal_id + '\\noise_levels_SNR.csv'
    SNR_G=[]
    with open(csvfilename) as csvfile:
        # open file as csv file
        csvReader = csv.reader(csvfile)
        # loop over rows
        for row in csvReader:
            SNR_G.append(row)

    SNR_G=np.array(SNR_G[1:][:]).astype('float')

    # read RE .csv
    csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_RE.csv'
    RE_G = []
    with open(csvfilename) as csvfile:
        # open file as csv file
        csvReader = csv.reader(csvfile)
        # loop over rows
        for row in csvReader:
            RE_G.append(row)

    RE_G = np.array(RE_G[1:][:]).astype('float')

    #plot
    ax[0, col_counter].boxplot(SNR_G)
    ax[0, col_counter].set_title('SNR ' + test_id, fontsize=50)
    ax[0, col_counter].set_xticks(np.arange(1, len(level_list)+1))
    ax[0, col_counter].set_xticklabels(level_list, rotation=45, fontsize=25)
    ax[0, col_counter].tick_params(axis='y', labelsize=30)


    ax[1, col_counter].boxplot(RE_G)
    ax[1, col_counter].axhline(baseline[col_counter, 1], xmin=0, xmax=1, c='r', ls='--')
    ax[1, col_counter].set_title('RENYI ENTR. ' + test_id, fontsize=50)
    ax[1, col_counter].set_xticks(np.arange(1, len(level_list)+1))
    ax[1, col_counter].set_xticklabels(level_list, rotation=45, fontsize=25)
    ax[1, col_counter].tick_params(axis='y', labelsize=30)


    col_counter+=1

fig.suptitle('Signal baseline metrics', fontsize=100)
plt.savefig(fig_name1)


#IF_extraction metrics
#save plots
fig_name2 = test_analysis_fold +'\\IF_extraction_metrics.jpg'

if signal_id == 'pure_tone':
    fig, ax = plt.subplots(2, 6, figsize=(100,50))
else:
    fig, ax = plt.subplots(3, 6, figsize=(100, 60))

col_counter = 0
for test_id in test_list:
    if not signal_id in os.listdir(test_result_dir+'\\'+test_id):
        # if not test for signal_id continue
        continue

    #read L2 .csv
    csvfilename = test_result_dir+ '\\'+test_id + '\\' + signal_id + '\\noise_levels_L2.csv'
    L2_G=[]
    with open(csvfilename) as csvfile:
        # open file as csv file
        csvReader = csv.reader(csvfile)
        # loop over rows
        for row in csvReader:
            L2_G.append(row)

    L2_G=np.array(L2_G[1:][:]).astype('float')
    #L2_G=np.log(L2_G)

    # read Geodetic .csv
    csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_Geodetic.csv'
    Geod_G = []
    with open(csvfilename) as csvfile:
        # open file as csv file
        csvReader = csv.reader(csvfile)
        # loop over rows
        for row in csvReader:
            Geod_G.append(row)

    Geod_G = np.array(Geod_G[1:][:]).astype('float')
    #Geod_G = np.log(Geod_G)

    if signal_id != 'pure_tone':
        # read Iatsenko .csv
        csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_Iatsenko.csv'
        Iatsenko_G = []
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                Iatsenko_G.append(row)

        Iatsenko_G = np.array(Iatsenko_G[1:][:]).astype('float')
        #Iatsenko_G = np.log(Iatsenko_G)

    #plot
    ax[0, col_counter].boxplot(L2_G)
    #ax[0, col_counter].axhline(np.log(baseline[col_counter, 2]), xmin=0, xmax=1, c='r', ls='--')
    ax[0, col_counter].axhline(baseline[col_counter, 2], xmin=0, xmax=1, c='r', ls='--')
    ax[0, col_counter].set_title('L2 ' + test_id, fontsize=50)
    ax[0, col_counter].set_xticks(np.arange(1, len(level_list)+1))
    ax[0, col_counter].set_xticklabels(level_list, rotation=45, fontsize=25)
    ax[0, col_counter].tick_params(axis='y', labelsize=30)


    ax[1, col_counter].boxplot(Geod_G)
    if signal_id == 'pure_tone':
        #ax[1, col_counter].axhline(np.log(baseline[col_counter, 3]), xmin=0, xmax=1, c='r', ls='--')
        ax[1, col_counter].axhline(baseline[col_counter, 3], xmin=0, xmax=1, c='r', ls='--')
    else:
        #ax[1, col_counter].axhline(np.log(baseline[col_counter, 4]), xmin=0, xmax=1, c='r', ls='--')
        ax[1, col_counter].axhline(baseline[col_counter, 4], xmin=0, xmax=1, c='r', ls='--')
    ax[1, col_counter].set_title('Geodetic ' + test_id, fontsize=50)
    ax[1, col_counter].set_xticks(np.arange(1, len(level_list)+1))
    ax[1, col_counter].set_xticklabels(level_list, rotation=45, fontsize=25)
    ax[1, col_counter].tick_params(axis='y', labelsize=30)

    if signal_id != 'pure_tone':
        ax[2, col_counter].boxplot(Iatsenko_G)
        ax[2, col_counter].axhline(baseline[col_counter, 3], xmin=0, xmax=1, c='r', ls='--')
        ax[2, col_counter].set_title('Geodetic ' + test_id, fontsize=50)
        ax[2, col_counter].set_xticks(np.arange(1, len(level_list) + 1))
        ax[2, col_counter].set_xticklabels(level_list, rotation=45, fontsize=25)
        ax[2, col_counter].tick_params(axis='y', labelsize=30)


    col_counter+=1

fig.suptitle('Instantaneous frequency extraction metrics', fontsize=100)
plt.savefig(fig_name2)


#Signal inversion vs original signal
#save plots
fig_name3 = test_analysis_fold +'\\signal_inversion_vs_original_metrics.jpg'


fig, ax = plt.subplots(3, 6, figsize=(120, 60))

col_counter = 0
for test_id in test_list:
    if not signal_id in os.listdir(test_result_dir+'\\'+test_id):
        # if not test for signal_id continue
        continue

    #read IMED original .csv
    csvfilename = test_result_dir+ '\\'+test_id + '\\' + signal_id + '\\noise_levels_IMED_original.csv'
    IMED_G=[]
    with open(csvfilename) as csvfile:
        # open file as csv file
        csvReader = csv.reader(csvfile)
        # loop over rows
        for row in csvReader:
            IMED_G.append(row)

    IMED_G=np.array(IMED_G[1:][:]).astype('float')
    #L2_G=np.log(L2_G)

    # read Geodetic .csv
    csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_SISDR_original.csv'
    SISDR_G = []
    with open(csvfilename) as csvfile:
        # open file as csv file
        csvReader = csv.reader(csvfile)
        # loop over rows
        for row in csvReader:
            SISDR_G.append(row)

    SISDR_G = np.array(SISDR_G[1:][:]).astype('float')
    #Geod_G = np.log(Geod_G)


    # read Iatsenko .csv
    csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_STOI_original.csv'
    STOI_G = []
    with open(csvfilename) as csvfile:
        # open file as csv file
        csvReader = csv.reader(csvfile)
        # loop over rows
        for row in csvReader:
            STOI_G.append(row)

    STOI_G = np.array(STOI_G[1:][:]).astype('float')
    #Iatsenko_G = np.log(Iatsenko_G)

    #plot
    ax[0, col_counter].boxplot(IMED_G)
    #ax[0, col_counter].axhline(np.log(baseline[col_counter, 2]), xmin=0, xmax=1, c='r', ls='--')
    if signal_id != 'pure_tone':
        ax[0, col_counter].axhline(baseline[col_counter, 7], xmin=0, xmax=1, c='r', ls='--')
    else:
        ax[0, col_counter].axhline(baseline[col_counter, 6], xmin=0, xmax=1, c='r', ls='--')
    ax[0, col_counter].set_title('IMED ' + test_id, fontsize=50)
    ax[0, col_counter].set_xticks(np.arange(1, len(level_list)+1))
    ax[0, col_counter].set_xticklabels(level_list, rotation=45, fontsize=25)
    ax[0, col_counter].tick_params(axis='y', labelsize=30)


    ax[1, col_counter].boxplot(SISDR_G)
    if signal_id == 'pure_tone':
        #ax[1, col_counter].axhline(np.log(baseline[col_counter, 3]), xmin=0, xmax=1, c='r', ls='--')
        ax[1, col_counter].axhline(baseline[col_counter, 5], xmin=0, xmax=1, c='r', ls='--')
    else:
        #ax[1, col_counter].axhline(np.log(baseline[col_counter, 4]), xmin=0, xmax=1, c='r', ls='--')
        ax[1, col_counter].axhline(baseline[col_counter, 6], xmin=0, xmax=1, c='r', ls='--')
    ax[1, col_counter].set_title('SISDR ' + test_id, fontsize=50)
    ax[1, col_counter].set_xticks(np.arange(1, len(level_list)+1))
    ax[1, col_counter].set_xticklabels(level_list, rotation=45, fontsize=25)
    ax[1, col_counter].tick_params(axis='y', labelsize=30)


    ax[2, col_counter].boxplot(STOI_G)
    if signal_id != 'pure_tone':
        ax[2, col_counter].axhline(baseline[col_counter, 5], xmin=0, xmax=1, c='r', ls='--')
    else:
        ax[2, col_counter].axhline(baseline[col_counter, 4], xmin=0, xmax=1, c='r', ls='--')
    ax[2, col_counter].set_title('STOI ' + test_id, fontsize=50)
    ax[2, col_counter].set_xticks(np.arange(1, len(level_list) + 1))
    ax[2, col_counter].set_xticklabels(level_list, rotation=45, fontsize=25)
    ax[2, col_counter].tick_params(axis='y', labelsize=30)


    col_counter+=1

fig.suptitle('Signal inversion vs original metrics', fontsize=100)
plt.savefig(fig_name3)

#signal inversion metrics
#save plots
fig_name4 = test_analysis_fold +'\\signal_inversion_metrics.jpg'

fig, ax = plt.subplots(4, 6, figsize=(120, 70))

col_counter = 0
for test_id in test_list:
    if not signal_id in os.listdir(test_result_dir+'\\'+test_id):
        # if not test for signal_id continue
        continue

    #read IMED noise .csv
    csvfilename = test_result_dir+ '\\'+test_id + '\\' + signal_id + '\\noise_levels_IMED_noise.csv'
    IMED_G=[]
    with open(csvfilename) as csvfile:
        # open file as csv file
        csvReader = csv.reader(csvfile)
        # loop over rows
        for row in csvReader:
            IMED_G.append(row)

    IMED_G=np.array(IMED_G[1:][:]).astype('float')
    #L2_G=np.log(L2_G)

    # read SISDR noise .csv
    csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_SISDR_noise.csv'
    SISDR_G = []
    with open(csvfilename) as csvfile:
        # open file as csv file
        csvReader = csv.reader(csvfile)
        # loop over rows
        for row in csvReader:
            SISDR_G.append(row)

    SISDR_G = np.array(SISDR_G[1:][:]).astype('float')
    #Geod_G = np.log(Geod_G)


    # read STOI noise .csv
    csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_STOI_noise.csv'
    STOI_G = []
    with open(csvfilename) as csvfile:
        # open file as csv file
        csvReader = csv.reader(csvfile)
        # loop over rows
        for row in csvReader:
            STOI_G.append(row)

    STOI_G = np.array(STOI_G[1:][:]).astype('float')
    #Iatsenko_G = np.log(Iatsenko_G)

    # read RE inverted noise .csv
    csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_RE_inverted_noise.csv'
    RE_I_G = []
    with open(csvfilename) as csvfile:
        # open file as csv file
        csvReader = csv.reader(csvfile)
        # loop over rows
        for row in csvReader:
            RE_I_G.append(row)

    RE_I_G = np.array(RE_I_G[1:][:]).astype('float')

    #plot
    ax[0, col_counter].boxplot(IMED_G)
    #ax[0, col_counter].axhline(np.log(baseline[col_counter, 2]), xmin=0, xmax=1, c='r', ls='--')
    if signal_id != 'pure_tone':
        ax[0, col_counter].axhline(baseline[col_counter, 7], xmin=0, xmax=1, c='r', ls='--')
    else:
        ax[0, col_counter].axhline(baseline[col_counter, 6], xmin=0, xmax=1, c='r', ls='--')
    ax[0, col_counter].set_title('IMED ' + test_id, fontsize=50)
    ax[0, col_counter].set_xticks(np.arange(1, len(level_list)+1))
    ax[0, col_counter].set_xticklabels(level_list, rotation=45, fontsize=25)
    ax[0, col_counter].tick_params(axis='y', labelsize=30)


    ax[1, col_counter].boxplot(SISDR_G)
    if signal_id == 'pure_tone':
        #ax[1, col_counter].axhline(np.log(baseline[col_counter, 3]), xmin=0, xmax=1, c='r', ls='--')
        ax[1, col_counter].axhline(baseline[col_counter, 5], xmin=0, xmax=1, c='r', ls='--')
    else:
        #ax[1, col_counter].axhline(np.log(baseline[col_counter, 4]), xmin=0, xmax=1, c='r', ls='--')
        ax[1, col_counter].axhline(baseline[col_counter, 6], xmin=0, xmax=1, c='r', ls='--')
    ax[1, col_counter].set_title('SISDR ' + test_id, fontsize=50)
    ax[1, col_counter].set_xticks(np.arange(1, len(level_list)+1))
    ax[1, col_counter].set_xticklabels(level_list, rotation=45, fontsize=25)
    ax[1, col_counter].tick_params(axis='y', labelsize=30)


    ax[2, col_counter].boxplot(STOI_G)
    if signal_id != 'pure_tone':
        ax[2, col_counter].axhline(baseline[col_counter, 5], xmin=0, xmax=1, c='r', ls='--')
    else:
        ax[2, col_counter].axhline(baseline[col_counter, 4], xmin=0, xmax=1, c='r', ls='--')
    ax[2, col_counter].set_title('STOI ' + test_id, fontsize=50)
    ax[2, col_counter].set_xticks(np.arange(1, len(level_list) + 1))
    ax[2, col_counter].set_xticklabels(level_list, rotation=45, fontsize=25)
    ax[2, col_counter].tick_params(axis='y', labelsize=30)

    ax[3, col_counter].boxplot(RE_I_G)
    if signal_id != 'pure_tone':
        ax[3, col_counter].axhline(baseline[col_counter, 8], xmin=0, xmax=1, c='r', ls='--')
    else:
        ax[3, col_counter].axhline(baseline[col_counter, 7], xmin=0, xmax=1, c='r', ls='--')
    ax[3, col_counter].set_title('RE inverted sound ' + test_id, fontsize=50)
    ax[3, col_counter].set_xticks(np.arange(1, len(level_list) + 1))
    ax[3, col_counter].set_xticklabels(level_list, rotation=45, fontsize=25)
    ax[3, col_counter].tick_params(axis='y', labelsize=30)



    col_counter+=1

fig.suptitle('Signal inversion metrics', fontsize=100)
plt.savefig(fig_name4)