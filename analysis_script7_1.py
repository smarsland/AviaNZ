"""
08/11/2022
Reviewed: 12/11/2022
Author: Virginia Listanti

This script purpose is to analyse data produced by IF experiment
Here we want to produce plot to compare results obtained with different normalization
We will produce one plot for each normalisation method (No norm, log, Box-Cox, PCEN)
This is the global version: where we put all the signals together

Review: added heat map with p-values
For each metric we will have a plot for each noise level (using subplot). 
"""

# import SignalProc
# import IF as IFreq
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import ast
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import scikit_posthocs as spost

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



def plot_parameters(t_list, t_result_dir, t_analysis_dir, s_list):
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

        for s_id in s_list:

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

            # n += 1

            del opt_param

    # save plots
    fig_name = t_analysis_dir + '\\' + 'optimal_parametes.jpg'
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

    fig.suptitle('Optimal parameters', fontsize=40)
    plt.savefig(fig_name)

    return

# signal type we are analysing
Signal_list = ['pure_tone', 'linear_upchirp', 'linear_downchirp', 'exponential_upchirp', 'exponential_downchirp']


#analysis for test folder
test_result_dir = "C:\\Users\\Virginia\\Documents\Work\\IF_extraction\\Test_Results_new"
test_analysis_dir = "C:\\Users\\Virginia\\Documents\\Work\\IF_extraction\\Results analysis\\Normalization_comparison2\\" \
                    "MultiTapered-MelScale\\Global"

# test_list = ['Test_200', 'Test_201', 'Test_202', 'Test_204']
# test_list = ['Test_500', 'Test_501', 'Test_502', 'Test_504']
# test_list = ['Test_300', 'Test_301', 'Test_302', 'Test_304']
# test_list = ['Test_600', 'Test_601', 'Test_602', 'Test_604']
# test_list = ['Test_400', 'Test_401', 'Test_402', 'Test_404']
test_list = ['Test_700', 'Test_701', 'Test_702', 'Test_704']

#create test_list
# start_index = 5
# test_list = []
# for i in range(5):
#     test_list.append('Test_' + str(start_index + i*6))

# start_index = 35
# test_list = []
# for i in range(3):
#     test_list.append('Test_' + str(start_index + i*60))

# start_index = 150
# test_list = []
# for i in range(6):
#     test_list.append('Test_' + str(start_index + i))

# plot_titles = [' Linear scale ', ' Mel scale']
plot_titles = ['No Norm.', 'Log', 'Box-Cox', 'PCEN']
# plot_titles = ["L2 pure file", "L2 noise samples", "Iats. pure file", "Iats. noise samples", "Geod. pure file", "Geod. noise samples"]
# for signal_id in Signal_list:
#     print('analysing ', signal_id)


    # # create signal folder
    # test_analysis_fold = test_analysis_dir + '\\' + signal_id
    # if not os.path.exists(test_analysis_fold):
    #     os.mkdir(test_analysis_fold)

    # Count and plot parameters chosen
plot_parameters(test_list, test_result_dir, test_analysis_dir, Signal_list)


test_number = 0
#recover baseline
baseline_median = []
for test_id in test_list:
    baseline = []
    for signal_id in Signal_list:
        if not signal_id in os.listdir(test_result_dir+'\\'+test_id):
            # if not test for signal_id continue
            continue

        csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\baseline_values.csv'
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            aid_baseline = []
            for row in csvReader:
                aid_baseline.append(row)
        aid_baseline = aid_baseline[1]
        aid_baseline = np.array(aid_baseline[:][:]).astype('float')
        if signal_id == 'pure_tone':
            baseline_new = np.zeros((1,9))
            baseline_new[0,:3] = aid_baseline[:3]
            baseline_new[0,3] = 0
            baseline_new[0,4:] = aid_baseline[3:]
        else:
            baseline_new = np.zeros((1, 9))
            baseline_new[0,:] = aid_baseline

        if np.size(baseline)==0:
            baseline = baseline_new
        else:
            baseline = np.vstack((baseline, baseline_new))

    # update median baseline
    if np.size(baseline_median)==0:
        baseline_median = np.median(baseline, axis=0)
    else:
        baseline_median=np.vstack((baseline_median, np.median(baseline, axis=0)))
    test_number += 1

# if test_number ==0:
#     continue
# baseline=np.array(baseline[:][:]).astype('float')

level_list = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8', 'Level 9',
              'Level 10', 'Level 11', 'Level 12', 'Level 13', 'Level 14']


#save plots
fig_name1 = test_analysis_dir +'\\signal_baseline_metrics.jpg'

fig, ax = plt.subplots(2, test_number, figsize=(100,40))
col_counter = 0
test_counter = 0
sample_counter = 0

for test_id in test_list:
    # SNR_G = np.zeros((100, 14, 4))
    RE_G = np.zeros((500, 14, 4))
    for signal_id in Signal_list:
        if not signal_id in os.listdir(test_result_dir+'\\'+test_id):
            # if not test for signal_id continue
            continue
            #read SNR .csv

        # read RE .csv
        csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_RE.csv'

        RE_copy = []
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                RE_copy.append(row)

        if np.size(RE_G)==0:
            RE_G[sample_counter, :, test_counter] = np.array(RE_copy[1:][:]).astype('float')
        else:
            RE_copy = np.array(RE_copy[1:][:]).astype('float')
            RE_G[:,:, test_counter] = np.vstack((RE_G,RE_copy))

df = sa.datasets.get_rdataset(RE_G).data


