"""
21/11/2022

Author: Virginia Listanti

This script purpose is to analyse data produced by IF experiment
Here we want to produce plot to compare results obtained with different ofrequency_scale
This is the global version: where we put all the signals together

We are also adding statistical analysis

"""

import SignalProc
import IF as IFreq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import os
import csv
import ast
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import scikit_posthocs as spost
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

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



def plot_parameters(t_list, t_result_dir, t_analysis_dir, s_list, meth_id):
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

    bar_color = "navy"

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
    fig_name = t_analysis_dir + '\\' + 'optimal_parametes_method'+str(meth_id)+'.jpg'
    # plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots(3, 2, figsize=(10, 10))

    # bar_width = 0.2

    index1 = np.arange(len(win_list))
    ax[0, 0].set_title('Win length', fontsize=14)
    ax[0, 0].bar(index1, win_list.values(), color=bar_color)
    ax[0, 0].set_xticks(index1, labels=win_list.keys(), fontsize=10)
    ax[0, 0].tick_params(axis='y', labelsize=10)
    # ax[0, 0].set_yticks(fontsize=20)
    # ax[0].set_xticks(index1)

    index2 = np.arange(len(incr_list))
    ax[0, 1].set_title('Incr length', fontsize=14)
    ax[0, 1].bar(index2, incr_list.values(), color=bar_color)
    ax[0, 1].set_xticks(index2, labels=incr_list.keys(), fontsize=6)
    ax[0, 1].tick_params(axis='y', labelsize=10)

    index3 = np.arange(len(win_type_list))
    ax[1, 0].set_title('Win type', fontsize=12)
    ax[1, 0].bar(index3, win_type_list.values(), color=bar_color)
    ax[1, 0].set_xticks(index3, labels=win_type_list.keys(), fontsize=10)
    ax[1, 0].tick_params(axis='y', labelsize=10)
    # ax[0].set_xticks(index1)

    index4 = np.arange(len(mel_bins))
    ax[1, 1].set_title('mel bins', fontsize=14)
    ax[1, 1].bar(index4, mel_bins.values(), color=bar_color)
    ax[1, 1].set_xticks(index4, labels=mel_bins.keys(), fontsize=10)
    ax[1, 1].tick_params(axis='y', labelsize=10)

    index5 = np.arange(len(alpha_list))
    ax[2, 0].set_title('Alpha', fontsize=14)
    ax[2, 0].bar(index5, alpha_list.values(), color=bar_color)
    ax[2, 0].set_xticks(index5, labels=alpha_list.keys(), fontsize=10)
    ax[2, 0].tick_params(axis='y', labelsize=10)
    # ax[0].set_xticks(index1)

    index6 = np.arange(len(beta_list))
    ax[2, 1].set_title('Beta', fontsize=14)
    ax[2, 1].bar(index6, beta_list.values(), color=bar_color)
    ax[2, 1].set_xticks(index6, labels=beta_list.keys(), fontsize=10)
    ax[2, 1].tick_params(axis='y', labelsize=10)

    # fig.suptitle('Optimal parameters Method '+ str(meth_id), fontsize=14)
    fig.tight_layout()
    plt.savefig(fig_name, dpi=200)

    return

def level_significance_plot(dataset, save_fig_name, label_list, plot_t):
    """
    This function produce the significative subplots for each level
    dataset is a tuple NXLXM
    N = number of samples
    L = number of levels
    M = number of tests
    """

    [N, L, M] = np.shape(dataset)
    # diagonal, non-significant, p<0.001, p<0.01, p<0.05
    cmap = ['white', 'crimson', 'darkgreen', 'mediumseagreen', 'lightgreen']

    i = 0
    j= 0
    fig, ax = plt.subplots(2, 7, figsize=(16,6))
    for level in range(L):
    # Level1
        dataframe = pd.DataFrame(dataset[:, level, :], columns = ['Linear', 'Mel'])
        pc = spost.posthoc_conover(dataframe, val_col = 'Linear', group_col ='Mel', p_adjust = 'holm')
    #     pc = spost.posthoc_conover(dataset[:, level, :], p_adjust='holm')
        heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True,
                        'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.5],
                        'ax': ax[i, j]}
        ax[i, j], cbar = spost.sign_plot(pc, **heatmap_args)
        ax[i, j].set_xticks(np.arange(0.5, len(label_list) + 0.5), labels=label_list)
        ax[i, j].set_yticks(np.arange(0.5, len(label_list) + 0.5), labels=label_list)
        ax[i, j].set_title('Level '+ str(level+1))

        if j<6:
            j+=1
        else:
            i+=1
            j=0

    # fig.delaxes(ax[-1,-1])
    fig.suptitle(plot_t, fontsize=20)
    fig.tight_layout()
    fig.savefig(save_fig_name, dpi=200)

    return



# signal type we are analysing
Signal_list = ['pure_tone', 'linear_upchirp', 'linear_downchirp', 'exponential_upchirp', 'exponential_downchirp']


#analysis for test folder
# test_result_dir = "C:\\Users\\Virginia\\Documents\Work\\IF_extraction\\Test_Results_new"
test_result_dir = "C:\\Users\\Virginia\\Documents\Work\\IF_extraction\\Test_Results"
# test_analysis_dir = "C:\\Users\\Virginia\\Documents\\Work\\Thesis_images\\Chapter_4\\Optimal_parameters_method\\"
# test_analysis_dir = "C:\\Users\\Virginia\\Documents\\Work\\Thesis_images\\Chapter_4\\Frequency_scale\\Global"
test_analysis_dir = "C:\\Users\\Virginia\\Documents\\Work\\Thesis_images\\Chapter_4\\Frequency_scale\\STandard"
method_number = 2

test_list0= ['Test_5', 'Test_35']
test_list1= ['Test_65', 'Test_95']
test_list2= ['Test_125', 'Test_155']

# global_test_list = [test_list0, test_list1, test_list2, test_list3, test_list4, test_list5]
# global_test_list = [test_list0, test_list1, test_list2]
global_test_list = [test_list0]

# plot_titles = [' Linear scale ', ' Mel scale']
# plot_titles = ['No Norm.', 'Log', 'Box-Cox', 'PCEN']
# plot_titles = ["L2 pure file", "L2 noise samples", "Iats. pure file", "Iats. noise samples", "Geod. pure file", "Geod. noise samples"]
# for signal_id in Signal_list:
#     print('analysing ', signal_id)
# plot_titles = ['Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5', 'Method 6']
plot_titles = ['Linear', 'Mel']


    # Count and plot parameters chosen
for i in range(method_number):
    test_list = []
    for list in global_test_list:
        test_list.append(list[i])
    method_id = i
    plot_parameters(test_list, test_result_dir, test_analysis_dir, Signal_list, method_id)

#recover baseline
baseline_median = np.zeros((6,9))
for i in range(method_number):
    for list in global_test_list:
        test_id = list[i]
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


        baseline_median[i] = np.median(baseline, axis=0)


# if test_number ==0:
#     continue
# baseline=np.array(baseline[:][:]).astype('float')

level_list = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5', 'Level 6', 'Level 7', 'Level 8', 'Level 9',
              'Level 10', 'Level 11', 'Level 12', 'Level 13', 'Level 14']


#save plots
fig_name1 = test_analysis_dir +'\\signal_baseline_metrics.jpg'

fig, ax = plt.subplots(2, method_number, figsize=(10,10))
boxprops = dict(color="navy")
flierprops = dict(markeredgecolor="navy")
whiskerprops = dict(color="navy")
capprops = dict(color="navy")
medianprops = dict(color="navy")
median_color = 'orange'

# col_counter = 0
RE_tot = []
for i in range(method_number):
    SNR_G = []
    RE_G = []
    RE_t = []
    for list in global_test_list:
        test_id = list[i]
        # print(test_id)
        for signal_id in Signal_list:
            if not signal_id in os.listdir(test_result_dir+'\\'+test_id):
                # if not test for signal_id continue
                continue
                #read SNR .csv
            csvfilename = test_result_dir+ '\\'+test_id + '\\' + signal_id + '\\noise_levels_SNR.csv'

            SNR_copy = []
            with open(csvfilename) as csvfile:
                # open file as csv file
                csvReader = csv.reader(csvfile)
                # loop over rows
                for row in csvReader:
                    SNR_copy.append(row)

            if np.size(SNR_G)==0:
                SNR_G = np.array(SNR_copy[1:][:]).astype('float')
            else:
                SNR_copy = np.array(SNR_copy[1:][:]).astype('float')
                SNR_G = np.vstack((SNR_G,SNR_copy))

            # read RE .csv
            csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_RE.csv'

            RE_copy = []
            with open(csvfilename) as csvfile:
                # open file as csv file
                csvReader = csv.reader(csvfile)
                # loop over rows<
                for row in csvReader:
                    RE_copy.append(row)

            if np.size(RE_G)==0:
                RE_G = np.array(RE_copy[1:][:]).astype('float')
            else:
                RE_copy = np.array(RE_copy[1:][:]).astype('float')
                RE_G = np.vstack((RE_G,RE_copy))

            if np.size(RE_t)==0 and signal_id != 'pure_tone':
                RE_t = np.array(RE_copy[1:][:]).astype('float')
            elif signal_id != 'pure_tone':
                RE_copy = np.array(RE_copy[1:][:]).astype('float')
                RE_t = np.vstack((RE_t,RE_copy))

    if i==0:
        RE_tot = RE_t
    else:
        row_check=np.shape(RE_tot)[0]
        RE_tot = np.dstack((RE_tot, RE_t[:row_check, :]))

    print('RE-T dim', np.shape(RE_t))
    print('dimensions', np.shape(RE_tot))
    ax[0, i].boxplot(SNR_G, boxprops=boxprops, flierprops= flierprops, whiskerprops= whiskerprops,
                               capprops= capprops, medianprops = medianprops)
    ax[0, i].set_title('SNR ' + plot_titles[i], fontsize=12)
    ax[0, i].set_xticks(np.arange(1, len(level_list)+1))
    ax[0, i].set_xticklabels(level_list, rotation=45, fontsize=8)
    ax[0, i].tick_params(axis='y', labelsize=7)


    ax[1, i].boxplot(RE_G, boxprops= boxprops, flierprops= flierprops, whiskerprops= whiskerprops,
                               capprops= capprops, medianprops = medianprops)
    ax[1, i].axhline(baseline_median[i, 1], xmin=0, xmax=1, c=median_color, ls='--')
    ax[1, i].set_title('RENYI ENTR. ' +plot_titles[i], fontsize=12)
    ax[1, i].set_xticks(np.arange(1, len(level_list)+1))
    ax[1, i].set_xticklabels(level_list, rotation=45, fontsize=8)
    ax[1, i].tick_params(axis='y', labelsize=7)


fig.suptitle('Signal baseline metrics', fontsize=25)
fig.tight_layout()
plt.savefig(fig_name1, dpi =200)
plt.close(fig)

fig_name1_s = test_analysis_dir + '\\RE_significative_tests.jpg'
lab_list = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
plot_title = 'Renyi Entropy Conover Test'
level_significance_plot(RE_tot, fig_name1_s, lab_list, plot_title)




#IF_extraction metrics
#save plots
fig_name2 = test_analysis_dir +'\\IF_extraction_metrics.jpg'

fig, ax = plt.subplots(3, method_number, figsize=(20, 30))
boxprops = dict(color="navy")
flierprops = dict(markeredgecolor="navy")
whiskerprops = dict(color="navy")
capprops = dict(color="navy")
medianprops = dict(color="navy")
median_color = 'orange'

L2_tot = []
Geod_tot = []
Iatsenko_tot = []
for i in range(method_number):
    L2_G = []
    Geod_G = []
    Iatsenko_G = []
    L2_t = []
    Geod_t = []
    Iatsenko_t = []
    for list in global_test_list:
        test_id = list[i]
        # print(test_id)
        for signal_id in Signal_list:
            if not signal_id in os.listdir(test_result_dir+'\\'+test_id):
                # if not test for signal_id continue
                print('skipping')
                continue

            #read L2 .csv
            csvfilename = test_result_dir+ '\\'+test_id + '\\' + signal_id + '\\noise_levels_L2.csv'
            L2_copy=[]
            with open(csvfilename) as csvfile:
                # open file as csv file
                csvReader = csv.reader(csvfile)
                # loop over rows
                for row in csvReader:
                    L2_copy.append(row)

            L2_copy=np.array(L2_copy[1:][:]).astype('float')
            if np.size(L2_G)==0:
                L2_G = np.copy(L2_copy)
            else:
                L2_G = np.vstack((L2_G,L2_copy))

            if np.size(L2_t) == 0 and signal_id != 'pure_tone':
                L2_t = L2_copy
            elif signal_id != 'pure_tone':
                L2_t = np.vstack((L2_t, L2_copy))
            #L2_G=np.log(L2_G)
            #from here
            # read Geodetic .csv
            csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_Geodetic.csv'
            Geod_copy = []
            with open(csvfilename) as csvfile:
                # open file as csv file
                csvReader = csv.reader(csvfile)
                # loop over rows
                for row in csvReader:
                    Geod_copy.append(row)

            Geod_copy = np.array(Geod_copy[1:][:]).astype('float')
            if np.size(Geod_G)==0:
                Geod_G = np.copy(Geod_copy)
            else:
                Geod_G = np.vstack((Geod_G,Geod_copy))
            #Geod_G = np.log(Geod_G)

            if np.size(Geod_t) == 0 and signal_id != 'pure_tone':
                Geod_t = Geod_copy
            elif signal_id != 'pure_tone':
                Geod_t = np.vstack((Geod_t, Geod_copy))

            if signal_id != 'pure_tone':
                # read Iatsenko .csv
                csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_Iatsenko.csv'
                Iatsenko_copy = []
                with open(csvfilename) as csvfile:
                    # open file as csv file
                    csvReader = csv.reader(csvfile)
                    # loop over rows
                    for row in csvReader:
                        Iatsenko_copy.append(row)

                Iatsenko_copy = np.array(Iatsenko_copy[1:][:]).astype('float')
                if np.size(Iatsenko_G)==0:
                    Iatsenko_G = np.copy(Iatsenko_copy)
                else:
                    Iatsenko_G = np.vstack((Iatsenko_G,Iatsenko_copy))

                if np.size(Iatsenko_t) == 0:
                    Iatsenko_t = Iatsenko_copy
                else:
                    Iatsenko_t = np.vstack((Iatsenko_t, Iatsenko_copy))



    if i==0:
        L2_tot = L2_t
        Geod_tot = Geod_t
        Iatsenko_tot = Iatsenko_t
    else:
        row_check1=np.shape(L2_tot)[0]
        L2_tot= np.dstack((L2_tot, L2_t[:row_check1, :]))
        row_check2 = np.shape(Geod_tot)[0]
        Geod_tot = np.dstack((Geod_tot, Geod_t[:row_check2, :]))
        row_check3 = np.shape(Iatsenko_tot)[0]
        Iatsenko_tot = np.dstack((Iatsenko_tot, Iatsenko_t[:row_check3, :]))
    #plot
    ax[0, i].boxplot(L2_G, boxprops=boxprops, flierprops= flierprops, whiskerprops= whiskerprops,
                               capprops= capprops, medianprops = medianprops)
    #ax[0, col_counter].axhline(np.log(baseline[col_counter, 2]), xmin=0, xmax=1, c='r', ls='--')
    ax[0, i].axhline(baseline_median[i, 2], xmin=0, xmax=1, c=median_color, ls='--')
    ax[0, i].set_title('L2 ' + plot_titles[i], fontsize=12)
    ax[0, i].set_xticks(np.arange(1, len(level_list)+1))
    ax[0, i].set_xticklabels(level_list, rotation=45, fontsize=8)
    ax[0, i].tick_params(axis='y', labelsize=7)


    ax[1, i].boxplot(Geod_G, boxprops=boxprops, flierprops= flierprops, whiskerprops= whiskerprops,
                               capprops= capprops, medianprops = medianprops)
    ax[1, i].axhline(baseline_median[i, 4], xmin=0, xmax=1, c=median_color, ls='--')
    ax[1, i].set_title('SRVF ' + plot_titles[i], fontsize=12)
    ax[1, i].set_xticks(np.arange(1, len(level_list)+1))
    ax[1, i].set_xticklabels(level_list, rotation=45, fontsize=8)
    ax[1, i].tick_params(axis='y', labelsize=7)

    ax[2, i].boxplot(Iatsenko_G, boxprops=boxprops, flierprops= flierprops, whiskerprops= whiskerprops,
                               capprops= capprops, medianprops = medianprops)
    ax[2, i].axhline(baseline_median[i, 3], xmin=0, xmax=1, c=median_color, ls='--')
    ax[2, i].set_title('Iatsenko ' + plot_titles[i], fontsize=12)
    ax[2, i].set_xticks(np.arange(1, len(level_list) + 1))
    ax[2, i].set_xticklabels(level_list, rotation=45, fontsize=8)
    ax[2, i].tick_params(axis='y', labelsize=7)


        # col_counter+=1

fig.suptitle('Instantaneous frequency extraction metrics', fontsize=25)
fig.tight_layout()
plt.savefig(fig_name2, dpi = 200)
plt.close(fig)

lab_list = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
fig_name2_s = test_analysis_dir + '\\L2_significative_tests.jpg'
plot_title = 'L2 Conover Test'
level_significance_plot(L2_tot, fig_name2_s, lab_list, plot_title)

fig_name2_s1 = test_analysis_dir + '\\SRVF_significative_tests.jpg'
plot_title = 'SRVF Conover Test'
level_significance_plot(Geod_tot, fig_name2_s1, lab_list, plot_title)

fig_name2_s2 = test_analysis_dir + '\\Iatsenko_significative_tests.jpg'
plot_title = 'Iatsenko Conover Test'
level_significance_plot(Iatsenko_tot, fig_name2_s2, lab_list, plot_title)


#Signal inversion vs original signal
#save plots
fig_name3 = test_analysis_dir +'\\signal_inversion_vs_original_metrics.jpg'


fig, ax = plt.subplots(3, method_number, figsize=(20, 30))
boxprops = dict(color="navy")
flierprops = dict(markeredgecolor="navy")
whiskerprops = dict(color="navy")
capprops = dict(color="navy")
medianprops = dict(color="navy")
median_color = 'orange'

IMED_tot = []
SISDR_tot = []
STOI_tot = []
for i in range(method_number):
    IMED_G = []
    SISDR_G = []
    STOI_G = []
    IMED_t = []
    SISDR_t = []
    STOI_t = []

    for list in global_test_list:
        test_id = list[i]
        for signal_id in Signal_list:
            if not signal_id in os.listdir(test_result_dir+'\\'+test_id):
                # if not test for signal_id continue
                continue

            #read IMED original .csv
            csvfilename = test_result_dir+ '\\'+test_id + '\\' + signal_id + '\\noise_levels_IMED_original.csv'
            IMED_copy=[]
            with open(csvfilename) as csvfile:
                # open file as csv file
                csvReader = csv.reader(csvfile)
                # loop over rows
                for row in csvReader:
                    IMED_copy.append(row)

            IMED_copy=np.array(IMED_copy[1:][:]).astype('float')
            if np.size(IMED_G)==0:
                IMED_G = np.copy(IMED_copy)
            else:
                IMED_G = np.vstack((IMED_G,IMED_copy))

            if np.size(IMED_t) == 0 and signal_id != 'pure_tone':
                IMED_t = IMED_copy
            elif signal_id != 'pure_tone':
                IMED_t = np.vstack((IMED_t, IMED_copy))

            # read Geodetic .csv
            csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_SISDR_original.csv'
            SISDR_copy = []
            with open(csvfilename) as csvfile:
                # open file as csv file
                csvReader = csv.reader(csvfile)
                # loop over rows
                for row in csvReader:
                    SISDR_copy.append(row)

            SISDR_copy = np.array(SISDR_copy[1:][:]).astype('float')
            if np.size(SISDR_G)==0:
                SISDR_G = np.copy(SISDR_copy)
            else:
                SISDR_G = np.vstack((SISDR_G,SISDR_copy))

            if np.size(SISDR_t) == 0 and signal_id != 'pure_tone':
                SISDR_t = SISDR_copy
            elif signal_id != 'pure_tone':
                SISDR_t = np.vstack((SISDR_t, SISDR_copy))

            # read Iatsenko .csv
            csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_STOI_original.csv'
            STOI_copy = []
            with open(csvfilename) as csvfile:
                # open file as csv file
                csvReader = csv.reader(csvfile)
                # loop over rows
                for row in csvReader:
                    STOI_copy.append(row)

            STOI_copy = np.array(STOI_copy[1:][:]).astype('float')
            if np.size(STOI_G)==0:
                STOI_G = np.copy(STOI_copy)
            else:
                STOI_G = np.vstack((STOI_G, STOI_copy))

            if np.size(STOI_t) == 0 and signal_id != 'pure_tone':
                STOI_t = STOI_copy
            elif signal_id != 'pure_tone':
                STOI_t = np.vstack((STOI_t, STOI_copy))

    if i==0:
        IMED_tot = IMED_t
        SISDR_tot = SISDR_t
        STOI_tot = STOI_t
    else:
        row_check1=np.shape(IMED_tot)[0]
        IMED_tot= np.dstack((IMED_tot, IMED_t[:row_check1, :]))
        row_check2 = np.shape(SISDR_tot)[0]
        SISDR_tot = np.dstack((SISDR_tot, SISDR_t[:row_check2, :]))
        row_check3 = np.shape(STOI_tot)[0]
        STOI_tot = np.dstack((STOI_tot, STOI_t[:row_check3, :]))
    #plot
    ax[0, i].boxplot(IMED_G, boxprops=boxprops, flierprops= flierprops, whiskerprops= whiskerprops,
                               capprops= capprops, medianprops = medianprops)
    #ax[0, col_counter].axhline(np.log(baseline[col_counter, 2]), xmin=0, xmax=1, c='r', ls='--')
    ax[0, i].axhline(baseline_median[i, 7], xmin=0, xmax=1, c=median_color, ls='--')
    ax[0, i].set_title('IMED ' + plot_titles[i], fontsize=12)
    ax[0, i].set_xticks(np.arange(1, len(level_list)+1))
    ax[0, i].set_xticklabels(level_list, rotation=45, fontsize=8)
    ax[0, i].tick_params(axis='y', labelsize=8)


    ax[1, i].boxplot(SISDR_G, boxprops=boxprops, flierprops= flierprops, whiskerprops= whiskerprops,
                               capprops= capprops, medianprops = medianprops)
    ax[1, i].axhline(baseline_median[i, 6], xmin=0, xmax=1, c=median_color, ls='--')
    ax[1, i].set_title('SISDR ' + plot_titles[i], fontsize=12)
    ax[1, i].set_xticks(np.arange(1, len(level_list)+1))
    ax[1, i].set_xticklabels(level_list, rotation=45, fontsize=8)
    ax[1, i].tick_params(axis='y', labelsize=8)


    ax[2, i].boxplot(STOI_G, boxprops=boxprops, flierprops= flierprops, whiskerprops= whiskerprops,
                               capprops= capprops, medianprops = medianprops)
    ax[2, i].axhline(baseline_median[i, 5], xmin=0, xmax=1, c=median_color, ls='--')
    ax[2, i].set_title('STOI ' + plot_titles[i], fontsize=12)
    ax[2, i].set_xticks(np.arange(1, len(level_list) + 1))
    ax[2, i].set_xticklabels(level_list, rotation=45, fontsize=8)
    ax[2, i].tick_params(axis='y', labelsize=8)


fig.suptitle('Signal inversion vs original metrics', fontsize=25)
fig.tight_layout()
plt.savefig(fig_name3, dpi=200)
plt.close(fig)

lab_list = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
fig_name3_s = test_analysis_dir + '\\IMED_significative_tests.jpg'
plot_title = 'IMED inverted vs original signal Conover Test'
level_significance_plot(IMED_tot, fig_name3_s, lab_list, plot_title)

fig_name3_s1 = test_analysis_dir + '\\SISDR_significative_tests.jpg'
plot_title = 'SISDR inverted vs original signal Conover Test'
level_significance_plot(SISDR_tot, fig_name3_s1, lab_list, plot_title)

fig_name3_s2 = test_analysis_dir + '\\STOI_significative_tests.jpg'
plot_title = 'STOI inverted vs original signal Conover Test'
level_significance_plot(STOI_tot, fig_name3_s2, lab_list, plot_title)
#
#signal inversion metrics
#save plots
fig_name4 = test_analysis_dir +'\\signal_inversion_metrics.jpg'

fig, ax = plt.subplots(4, method_number, figsize=(20, 40))
boxprops = dict(color="navy")
flierprops = dict(markeredgecolor="navy")
whiskerprops = dict(color="navy")
capprops = dict(color="navy")
medianprops = dict(color="navy")
median_color = 'orange'

IMED_tot = []
SISDR_tot = []
STOI_tot = []
RE_I_tot = []
for i in range(method_number):
    IMED_G = []
    SISDR_G = []
    STOI_G = []
    RE_I_G = []
    IMED_t = []
    SISDR_t = []
    STOI_t = []
    RE_I_t = []

    for list in global_test_list:
        test_id = list[i]
        if not signal_id in os.listdir(test_result_dir+'\\'+test_id):
            # if not test for signal_id continue
            continue

        #read IMED noise .csv
        csvfilename = test_result_dir+ '\\'+test_id + '\\' + signal_id + '\\noise_levels_IMED_noise.csv'
        IMED_copy=[]
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                IMED_copy.append(row)

        IMED_copy=np.array(IMED_copy[1:][:]).astype('float')
        if np.size(IMED_G) == 0:
            IMED_G = np.copy(IMED_copy)
        else:
            IMED_G = np.vstack((IMED_G, IMED_copy))

        if np.size(IMED_t) == 0 and signal_id != 'pure_tone':
            IMED_t = IMED_copy
        elif signal_id != 'pure_tone':
            IMED_t = np.vstack((IMED_t, IMED_copy))

        # read SISDR noise .csv
        csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_SISDR_noise.csv'
        SISDR_copy = []
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                SISDR_copy.append(row)

        SISDR_copy = np.array(SISDR_copy[1:][:]).astype('float')
        if np.size(SISDR_G) == 0:
            SISDR_G = np.copy(SISDR_copy)
        else:
            SISDR_G = np.vstack((SISDR_G, SISDR_copy))

        if np.size(SISDR_t) == 0 and signal_id != 'pure_tone':
            SISDR_t = SISDR_copy
        elif signal_id != 'pure_tone':
            SISDR_t = np.vstack((SISDR_t, SISDR_copy))


        # read STOI noise .csv
        csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_STOI_noise.csv'
        STOI_copy = []
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                STOI_copy.append(row)

        STOI_copy = np.array(STOI_copy[1:][:]).astype('float')
        if np.size(STOI_G) == 0:
            STOI_G = np.copy(STOI_copy)
        else:
            STOI_G = np.vstack((STOI_G, STOI_copy))

        if np.size(STOI_t) == 0 and signal_id != 'pure_tone':
            STOI_t = STOI_copy
        elif signal_id != 'pure_tone':
            STOI_t = np.vstack((STOI_t, STOI_copy))

        # read RE inverted noise .csv
        csvfilename = test_result_dir + '\\' + test_id + '\\' + signal_id + '\\noise_levels_RE_inverted_noise.csv'
        RE_I_copy = []
        with open(csvfilename) as csvfile:
            # open file as csv file
            csvReader = csv.reader(csvfile)
            # loop over rows
            for row in csvReader:
                RE_I_copy.append(row)

        RE_I_copy = np.array(RE_I_copy[1:][:]).astype('float')
        if np.size(RE_I_G) == 0:
            RE_I_G = RE_I_copy
        else:
            RE_I_G = np.vstack((RE_I_G, RE_I_copy))

        if np.size(RE_I_t) == 0 and signal_id != 'pure_tone':
            RE_I_t = RE_I_copy
        elif signal_id != 'pure_tone':
            RE_I_t = np.vstack((RE_I_t, RE_I_copy))

    if i==0:
        IMED_tot = IMED_t
        SISDR_tot = SISDR_t
        STOI_tot = STOI_t
        RE_I_tot = RE_I_t
    else:
        row_check1=np.shape(IMED_tot)[0]
        IMED_tot= np.dstack((IMED_tot, IMED_t[:row_check1, :]))
        row_check2 = np.shape(SISDR_tot)[0]
        SISDR_tot = np.dstack((SISDR_tot, SISDR_t[:row_check2, :]))
        row_check3 = np.shape(STOI_tot)[0]
        STOI_tot = np.dstack((STOI_tot, STOI_t[:row_check3, :]))
        row_check4 = np.shape(RE_I_tot)[0]
        RE_I_tot = np.dstack((RE_I_tot, RE_I_t[:row_check4, :]))
    #plot
    ax[0, i].boxplot(IMED_G, boxprops=boxprops, flierprops= flierprops, whiskerprops= whiskerprops,
                               capprops= capprops, medianprops = medianprops)
    #ax[0, col_counter].axhline(np.log(baseline[col_counter, 2]), xmin=0, xmax=1, c='r', ls='--')
    ax[0, i].axhline(baseline_median[i, 7], xmin=0, xmax=1, c=median_color, ls='--')
    ax[0, i].set_title('IMED ' + plot_titles[i], fontsize=12)
    ax[0, i].set_xticks(np.arange(1, len(level_list)+1))
    ax[0, i].set_xticklabels(level_list, rotation=45, fontsize=8)
    ax[0, i].tick_params(axis='y', labelsize=8)


    ax[1, i].boxplot(SISDR_G, boxprops=boxprops, flierprops= flierprops, whiskerprops= whiskerprops,
                               capprops= capprops, medianprops = medianprops)
    ax[1, i].axhline(baseline_median[i, 6], xmin=0, xmax=1, c=median_color, ls='--')
    ax[1, i].set_title('SISDR ' + plot_titles[i], fontsize=12)
    ax[1, i].set_xticks(np.arange(1, len(level_list)+1))
    ax[1, i].set_xticklabels(level_list, rotation=45, fontsize=8)
    ax[1, i].tick_params(axis='y', labelsize=8)


    ax[2, i].boxplot(STOI_G, boxprops=boxprops, flierprops= flierprops, whiskerprops= whiskerprops,
                               capprops= capprops, medianprops = medianprops)
    ax[2, i].axhline(baseline_median[i, 5], xmin=0, xmax=1, c=median_color, ls='--')
    ax[2, i].set_title('STOI ' + plot_titles[i], fontsize=12)
    ax[2, i].set_xticks(np.arange(1, len(level_list) + 1))
    ax[2, i].set_xticklabels(level_list, rotation=45, fontsize=8)
    ax[2, i].tick_params(axis='y', labelsize=8)

    ax[3, i].boxplot(RE_I_G, boxprops=boxprops, flierprops= flierprops, whiskerprops= whiskerprops,
                               capprops= capprops, medianprops = medianprops)
    ax[3, i].axhline(baseline_median[i, 8], xmin=0, xmax=1, c=median_color, ls='--')
    ax[3, i].set_title('RE inverted sound ' + plot_titles[i], fontsize=12)
    ax[3, i].set_xticks(np.arange(1, len(level_list) + 1))
    ax[3, i].set_xticklabels(level_list, rotation=45, fontsize=8)
    ax[3, i].tick_params(axis='y', labelsize=8)

fig.suptitle('Signal inversion metrics', fontsize=18)
fig.tight_layout()
plt.savefig(fig_name4)
plt.close(fig)

lab_list = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
fig_name4_s = test_analysis_dir + '\\IMED_inversion_significative_tests.jpg'
plot_title = 'IMED inverted signal Conover Test'
level_significance_plot(IMED_tot, fig_name4_s, lab_list, plot_title)

fig_name4_s1 = test_analysis_dir + '\\SISDR_inversion_significative_tests.jpg'
plot_title = 'SISDR inverted signal Conover Test'
level_significance_plot(SISDR_tot, fig_name4_s1, lab_list, plot_title)

fig_name4_s2 = test_analysis_dir + '\\STOI_inversion_significative_tests.jpg'
plot_title = 'STOI inverted signal Conover Test'
level_significance_plot(STOI_tot, fig_name4_s2, lab_list, plot_title)

fig_name4_s3 = test_analysis_dir + '\\RE_inverted_signal_significative_tests.jpg'
lab_list = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
plot_title = 'Renyi Entropy inverted signal Conover Test'
level_significance_plot(RE_I_tot, fig_name4_s3, lab_list, plot_title)

print('Plots done')