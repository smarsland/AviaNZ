"""
23/11/2021
Author: Virginia Listanti

This program implements the experiment pipeline for the IF extraction

This program is built to run on the CUDA machines

PIPELINE:
- Iterate over:
       * possible TFRs & co.
            - TFR: Short-time Fourier Transform
            - Post-processing: Reassigned, Multitapered
            - Frequency scale: Linear, Mel
            - Spectrogram Normalization: Standard, Log, Box-Cox, Sigmoid, PCEN

       * parameter optimization metric:
            - L2
            - Iatsenko (no pure-tone)
            - curve registration

       * optimization technique:
            - only using "pure signal"
            - using all baseline datase


For each Test:
    - Create dir (local Virginia) where to store the result metrics
    - Store TEST INFO: TFR  + parameter optimization metric
    - Navigate Dataset subdirectory (aka signal-type)

    * For each subdirectory in the dataset directory (aka signal-type):
        - create subdirectory in the test result directory
        - Find optimal parameters using optimization metric: window_lenght, incr, window_type, alpha, beta
        - Store optimal paramters info
        - Evaluate & store as .csv Baseline metrics for "pure" signal
        - Initialize general metrics (for signal type): 1 column for each noise level
        - Navigate subdirectory with noise-level samples

        * For each noise level
            - initialize local metrics (1 column for each metric)
            - evaluate metrics for each samples ans store both in local and general
            - save local metrics in .csv

        - Save general metrics as .csv
        - Update Test id

Metrics we are going to measure:
    * Baseline metrics:
        - Signal-to-noise ratio
        - Renyi Entropy of the spectrogram

    * Metrics on IF extraction (between correct IF and extracted IF)
        - l2 norm
        - Iatsenko error
        - Geodetic distance

    * Metric on spectrogram inversion
        - SISDR (between signal obtained via spectrogram inversion and original signal without noise) &
                (between signal obtained via spectrogram inversion and original signal)
        - STOI  (between signal obtained via spectrogram inversion and original signal without noise) &
                (between signal obtained via spectrogram inversion and original signal)
        - IMED (Between spectrogram of inverted signal and original spectrogram without noise)
                (Between spectrogram of inverted signal and original spectrogram without noise)

"""

import SignalProc
import IF as IFreq
import numpy as np
from numpy.linalg import norm
# from scipy.io import loadmat, savemat
import os
import csv
import imed
import speechmetrics as sm
from fdasrsf.geodesic import geod_sphere
#from librosa.feature.inverse import mel_to_audio
#from numba import cuda


########################## Utility functions ###########################################################################

def Signal_to_noise_Ratio(signal_sample, noise_sample):
    # Signal-to-noise ratio
    # Handle the case with no noise as well
    if len(noise_sample) == 0:
        snr = 0
    else:
        snr = 10 * np.log10((np.sum(signal_sample ** 2) / len(signal_sample)) /
                            (np.mean(noise_sample ** 2) / len(noise_sample)))
    return snr


def Renyi_Entropy(a, order=3):
    # Renyi entropy.
    # Default is order 3

    r_e = (1 / (1 - order)) * np.log2(np.sum(a ** order) / np.sum(a))
    return r_e


def Iatsenko_style(s1, s2):
    # This function implement error function as defined
    # in Iatsenko et al. IF paper
    # s1 is the reference signal

    # Removing try because we can't enter this function with pure_tone

    error = np.mean((s1 - s2) ** 2) / np.mean((s1 - np.mean(s1)) ** 2)

    return error


def IMED_distance(a, b):
    # This function evaluate IMED distance between 2 matrix
    # 1) Rescale matrices to [0,1]
    # 2) call imed distance

    a2 = (a - np.amin(a)) / np.ptp(a)
    b2 = (b - np.amin(b)) / np.ptp(b)

    #print('shape a2', np.shape(a2), 'shape b2 ', np.shape(b2))

    return imed.distance(a2, b2)


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

def find_optimal_spec_IF_parameters(test_param, op_param, op_measure, file_dir, wav_file_list, csv_path, f_names,
                                    sign_id, spectrogram_type, freq_scale, normal_type, op_metric, op_option="Original"):
    """
    This function computes the optimization metrics OP_METRIC for the set of parameters TEST_PARAM.
    It the metric is smaller than OP_MEASURE, it updates OP_PARAM.
    Furthermore, it updates the csv_files stored at CVS_PATH
    """

    print("\nTESTING with window lenght= ", test_param["win_len"], " and increment = ", test_param["hop"],
          "window_type = ", test_param["window_type"], " mel bins = ", test_param["mel_num"], " alpha = ",
          test_param["alpha"], "beta = ", test_param["beta"], "\n")

    if op_option == "Original":
        n = 1
    else:
        # only two possible options
        n = len(wav_file_list)

    measure2check = 0
    for signal_file in wav_file_list:
        IF = IFreq.IF(method=2, pars=[test_param["alpha"], test_param["beta"]])
        sp = SignalProc.SignalProc(test_param["win_len"], test_param["hop"])

        # read signal
        signal_path = file_dir + "/" + signal_file
        sp.readWav(signal_path)
        print('Using ', signal_path)
        sample_rate = sp.sampleRate
        file_len = sp.fileLength / sample_rate
        instant_freq_fun = set_if_fun(sign_id, file_len)

        tfr = sp.spectrogram(test_param["win_len"], test_param["hop"], test_param["window_type"], sgType=spectrogram_type, sgScale=freq_scale,
                             nfilters=test_param["mel_num"])
        print("spec dims", np.shape(tfr))

        # spectrogram normalizations
        if normal_type != "Standard":
            sp.normalisedSpec(tr=normal_type)
            tfr = sp.sg

        tfr = tfr.T
        [num_row, num_col] = np.shape(tfr)

        # appropriate frequency scale
        if freq_scale == "Linear":
            f_step = (sample_rate / 2) / np.shape(tfr)[0]
            freq_arr = np.arange(f_step, sample_rate / 2 + f_step, f_step)
        else:
            # #mel freq axis
            n_filters = test_param["mel_num"]
            freq_arr = np.linspace(sp.convertHztoMel(0), sp.convertHztoMel(fs / 2),
                                   n_filters + 1)
            freq_arr = freq_arr[1:]

        w_opt = [sample_rate, test_param["win_len"]]  # this neeeds review
        tf_supp, _, _ = IF.ecurve(tfr, freq_arr, w_opt)
        # try:
        #     tf_supp, _, _ = IF.ecurve(tfr, freq_arr, w_opt)
        # except:
        #     print('ERROR IN CURVE EXTRACTION')
        #     measure2check += np.nan
        #     break

        # revert to Hz if Mel
        if freq_scale == 'Mel Frequency':
            tf_supp[0, :] = sp.convertMeltoHz(tfsupp[0, :])

        # calculate
        instant_freq = instant_freq_fun(np.linspace(0, file_len, np.shape(tf_supp[0, :])[0]))
        # line checked

        if op_metric == "L2":
            measure2check += norm(tf_supp[0, :] - instant_freq, ord=2) / (num_row * num_col)
        elif op_metric == "Iatsenko":
            measure2check += Iatsenko_style(instant_freq, tf_supp[0, :])
        else:
            time_support = np.linspace(0, file_len, np.shape(tf_supp[0, :])[0])
            measure2check += Geodesic_curve_distance(time_support, tf_supp[0, :], time_support,
                                                     instant_freq)

        # safety chack cleaning
        del tfr, f_step, freq_arr, w_opt, tf_supp, sp, IF

    measure2check /= n  # mean over samples
    with open(csv_path, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=f_names)
        writer.writerow({'window_width': test_param["win_len"], 'incr': test_param["hop"], 'window type': test_param["window_type"],
                         "mel bins": test_param["mel_num"], 'alpha': test_param["alpha"], 'beta': test_param["beta"],
                         'spec dim': num_row * num_col, 'measure': measure2check})

    if (not np.isnan(measure2check)) and (measure2check < op_measure):

        op_measure = measure2check
        op_param = test_param
        print("optimal parameters updated:", op_param)

    return op_param, op_measure


def find_optimal_spec_IF_parameters_handle(base_dir, save_dir, sign_id, spectrogram_type, freq_scale, normal_type,
                                    optim_metric, optim_option="Original"):
    """
    This function find optimal parameters for the spectrogram and the frequency extraction algorithm in order
    to minimize the distance optim_metric between the extracted IF and the "original" ones

    Input:
        base_dir: directory where are stores samples files
        save_dir: directory where to save log and parameters
        sign id: signal type usefule to retrieve the inst_freq_fun
        Spectrogram_type= selected spectrogram type
        freq_scale= selected frequency scale
        normal_type = selected spectrogram normalisation o
        optim_metric = selected optimization metric
        optim_option =selected optimizarion techinique
                    "Original" testing only on pure signal
                    "Directory" testing over all samples (one per each noise level) [ considering mean]

    Output
        window_length_opt
        incr_opt
        window_type_opt
        alpha_opt
        beta_opt
    """

    # Spectrogram parameters
    win = np.array([256, 512, 1024, 2048])
    hop_perc = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    win_type = ['Hann', 'Parzen', 'Welch', 'Hamming', 'Blackman']

    # If Extraction parameters
    alpha_list = np.array([0, 0.5, 1, 2.5, 5, 10])
    beta_list = np.array([0, 0.5, 1, 2.5, 5, 10])

    # mel bins options
    if freq_scale == 'Mel Frequency':
        mel_bins = np.array([64, 128, 256])#only power of 2
    else:
        mel_bins = [None]

    opt = np.Inf

    #inizialize opt_param with standard values
    opt_param = {"win_len": [], "hop": [], "window_type": [], "mel_num": [], "alpha": [], "beta": []}
    test_param = {"win_len": [], "hop": 128, "window_type": 'Hann', "mel_num": None, "alpha": 1, "beta": 1}

    if freq_scale == 'Mel Frequency':
        opt_param['mel_num'] = 64

    if optim_option== "Original":
        file_list = [sign_id + "_00.wav"]
    else:
        file_list = os.listdir(base_dir)

    # store values into .csv file
    fieldnames = ['window_width', 'incr', 'window type', "mel bins", 'alpha', 'beta', 'spec dim', 'measure']
    csv_filename = save_dir + '/find_optimal_parameters_log.csv'
    with open(csv_filename, 'w', newline='') as csv_save_file:
        writer = csv.DictWriter(csv_save_file, fieldnames=fieldnames)
        writer.writeheader()

    #find optimal window
    for win_len in win:
        # loop on possible window_length
        test_param["win_len"] = int(win_len)
        [opt_param, opt] = find_optimal_spec_IF_parameters(test_param, opt_param, opt, base_dir, file_list, csv_filename,
                                                         fieldnames, sign_id, spectrogram_type, freq_scale, normal_type,
                                                         optim_metric, op_option=optim_option)

    test_param = opt_param
    for hop in hop_perc:
        # loop on possible hop
        test_param["hop"] = int(test_param["win_len"]*hop)
        [opt_param, opt] = find_optimal_spec_IF_parameters(test_param, opt_param, opt, base_dir, file_list,
                                                           csv_filename,fieldnames, sign_id, spectrogram_type,
                                                           freq_scale, normal_type, optim_metric,
                                                           op_option=optim_option)

    test_param = opt_param
    for window_type in win_type:
        # loop on possible window_types
        test_param["window_type"] = window_type
        [opt_param, opt] = find_optimal_spec_IF_parameters(test_param, opt_param, opt, base_dir, file_list,
                                                           csv_filename, fieldnames, sign_id, spectrogram_type,
                                                           freq_scale, normal_type, optim_metric,
                                                           op_option=optim_option)

    if freq_scale == 'Mel Frequency':
        # do the loop only if we are testing mel spectrogram
        test_param = opt_param
        for num_bin in mel_bins:
            # loop over possible numbers of bins. If None this is just one loop
            test_param["mel_num"] = num_bin
            [opt_param, opt] = find_optimal_spec_IF_parameters(test_param, opt_param, opt, base_dir, file_list,
                                                               csv_filename, fieldnames, sign_id, spectrogram_type,
                                                               freq_scale, normal_type, optim_metric,
                                                               op_option=optim_option)


    # optimize paremeters needed for IF extraction: these are dipendent
    test_param = opt_param
    for alpha in alpha_list:
        # loop on alpha
        for beta in beta_list:
            # loop on beta

            test_param["alpha"] = alpha
            test_param["beta"] = beta
            [opt_param, opt] = find_optimal_spec_IF_parameters(test_param, opt_param, opt, base_dir, file_list,
                                                               csv_filename, fieldnames, sign_id, spectrogram_type,
                                                               freq_scale, normal_type, optim_metric,
                                                               op_option=optim_option)


    print("\n Optimal parameters \n", opt_param)
    return opt_param


def save_test_info(file_path, spectrogram_type, freq_scale, norm_spec, optim_metric, optim_option):
    """
    This function stores TFR info into a .txt file

    INPUT:
        - file_path: path where to store our .txt file
        - spectrogram_type: str with spectrogram type information
        - freq_scale: str with frequency scale information
        - norm_spec: str with spectrogram normalization information
        - optim_metric: str with info on metric utilised to select spectrogram parameters
        - optim_option: approach to find optimal parameters
    """

    file_txt = open(file_path, 'w')

    l0 = [" Test info Log\n"]
    l1 = ["\n Spectrogram type used: " + spectrogram_type]
    l2 = ["\n Frequency scale used: " + freq_scale]
    l3 = ["\n Spectrogram normalization technique used: " + norm_spec]
    l4 = ["\n Distance used for path optimization: " + optim_metric]
    l5 = ["\n Method used for path optimization: " + optim_option]
    file_txt.writelines(np.concatenate((l0, l1, l2, l3, l4, l5)))
    file_txt.close()
    return


def save_optima_parameters(dir_path, opt_par):
    """
    This function store optimal parameters opt_par into the directory folder_path

    INPUT:
        - dir_path location where we want to store the parameters
        - opt_par dictionary where we are storing the parameters
    """

    save_path = dir_path + "/Optimal_parameters.txt"

    f = open(save_path,'w')
    f.write(str(opt_par))
    f.close()

    return


def calculate_metrics_original_signal(signal_dir, save_dir, sign_id, sg_type, sg_scale, sg_norm, opt_param):
    """
    This function calculate metrics for the signal without noise
    """

    file_name = signal_dir + "/" + sign_id + "_00.wav"
    # data_file = test_dir + "\\" + test_fold + "\\" + test_id + "\\" + file_name

    # inizialization for sisdr e stoi score
    sm_metrics = sm.load(["stoi", 'sisdr'], window=None)

    # define classes
    IF = IFreq.IF(method=2, pars=[opt_param["alpha"], opt_param["beta"]])
    sp = SignalProc.SignalProc(opt_param["win_len"], opt_param["hop"])

    # read file
    sp.readWav(file_name)
    sample_rate = sp.sampleRate
    file_len = sp.fileLength / sample_rate

    # set IF
    instant_freq_fun = set_if_fun(sign_id, file_len)

    # Evaluate TFR
    tfr = sp.spectrogram(opt_param["win_len"], opt_param["hop"], opt_param["window_type"], sgType=sg_type,
                         sgScale=sg_scale, nfilters=opt_param["mel_num"])
    # spectrogram normalizations
    if sg_norm != "Standard":
        sp.normalisedSpec(tr=sg_norm)
        tfr = sp.sg

    tfr2 = tfr.T

    # appropriate frequency scale
    if sg_scale == "Linear":
        f_step = (sample_rate / 2) / np.shape(tfr2)[0]
        freq_arr = np.arange(f_step, sample_rate / 2 + f_step, f_step)
    else:
        # #mel freq axis
        n_filters = 40
        freq_arr = np.linspace(sp.convertHztoMel(0), sp.convertHztoMel(sample_rate / 2), n_filters + 1)
        freq_arr = freq_arr[1:]

    # extract IF
    w_opt = [sample_rate, opt_param["win_len"]]  # this neeeds review
    tf_supp, _, _ = IF.ecurve(tfr2, freq_arr, w_opt)

    # revert to Hz if Mel
    if sg_scale == 'Mel Frequency':
        tf_supp[0, :] = sp.convertMeltoHz(tfsupp[0, :])

    # reference IF
    time_support = np.linspace(0, file_len, np.shape(tf_supp[0, :])[0])  # array with temporal coordinates
    inst_freq_law = instant_freq_fun(time_support)

    # invert spectrogram
    sign_original = sp.data
    if sg_scale == 'Mel Frequency':
        # Pathched
        F = mel_filterbank_maker(opt_param["win_len"], 'mel', nfilters)
        F_pseudo = np.linalg.pinv(F)
        TFR_recovered = np.absolute(np.dot(TFR, F_pseudo))

    else:
        TFR_recovered=tfr

    print('shape TFR rec ', np.shape(TFR_recovered), ' window width ',opt_param["win_len"], ' incr ', opt_param["hop"],
          ' window type ', opt_param["window_type"])
    s1_inverted = sp.invertSpectrogram(TFR_recovered, window_width=opt_param["win_len"], incr=opt_param["hop"],
                                       window=opt_param["window_type"])

    # spectrogram of inverted signal
    sp.data = s1_inverted
    tfr_inv = sp.spectrogram(opt_param["win_len"], opt_param["hop"], opt_param["window_type"], sgType=sg_type,
                             sgScale=sg_scale, nfilters=opt_param["mel_num"])

    # spectrogram normalizations
    if sg_norm != "Standard":
        sp.normalisedSpec(tr=sg_norm)
        tfr_inv = sp.sg

    tfr2_inv = tfr_inv.T

    # store metrics into a dictionary
    # initialise dictionary with: standard dev, Renyi Entropy and L2 norm (IF)
    original_sound_metrics = {"std": np.std(sign_original), 'Renyi Entropy': Renyi_Entropy(tfr),
                              'L2 norm': norm(tf_supp[0, :] - inst_freq_law)}

    # IAM
    if sign_id != "pure_tone":
        original_sound_metrics['Iatsenko err.'] = Iatsenko_style(inst_freq_law, tf_supp[0, :])
    # CURVE DISTANCE
    original_sound_metrics["Geodetic"] = Geodesic_curve_distance(time_support, tf_supp[0, :], time_support,
                                                                 inst_freq_law)

    # Reconstructed signal metrics
    # speech metrics
    l_diff = len(sign_original) - len(s1_inverted)
    score_orig = sm_metrics(s1_inverted, sign_original[int(np.floor(l_diff / 2)):-int(np.ceil(l_diff / 2))],
                            rate=sample_rate)
    original_sound_metrics['STOI'] = score_orig['stoi']
    original_sound_metrics["SISDR"] = score_orig['sisdr']
    # imed
    col_diff = np.shape(tfr2)[1] - np.shape(tfr2_inv)[1]
    print('col:diff ',col_diff)
    if col_diff==0:
        original_sound_metrics["IMED"] = IMED_distance(tfr2, tfr2_inv)
    else:
        original_sound_metrics["IMED"] = IMED_distance(tfr2[:, int(np.floor(col_diff / 2)):-int(np.ceil(col_diff / 2))],
                                                   tfr2_inv)
    original_sound_metrics["Renyi Entropy inv. spec."] = Renyi_Entropy(tfr2_inv)
    # store fieldnames
    field_names = []
    for element in original_sound_metrics:
        field_names.append(element)

    with open(save_dir + '/baseline_values.csv', 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerow(original_sound_metrics)

    return


def save_metric_csv(csv_filename, fieldnames, metric_matrix):
    """
    This functions save the values stored into metric_matrix to csvfilename using the fieldnames indicated by fieldnames
    """
    m, _ = np.shape(metric_matrix)
    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for h in range(m):
            row = {}
            for j in range(len(fieldnames)):
                row[fieldnames[j]] = metric_matrix[h, j]
            writer.writerow(row)
    return


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


########################################################################################################################
########################################################   MAIN ########################################################
########################################################################################################################


# directory where to find test dataset files
dataset_dir = "/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_03/Virginia_IF_experiment/Toy_Signals"
# directory to store test result
main_results_dir = '/am/state-opera/home1/listanvirg/Documents/IF_experiment_Results'

# signals list
signals_list = os.listdir(dataset_dir)

# TFR options
# sgtypes
spectrogram_types = ["Standard", "Reassigned", "Multi-tapered"]
# sgscale
freq_scales = ['Linear', 'Mel Frequency']
# spectrogram normalization functions
spec_normalizations = ["Standard", "Log", "Box-Cox", "Sigmoid", "PCEN"]
# optimization metrics
optimization_metrics = ['L2', 'Iatsenko', 'Curve_registration']
# optimization options
optimization_options = ["Original", "Directory"]

# metrics from sm
metrics = sm.load(["stoi", 'sisdr'], window=None)

# Inizialization
Test_id = 0

# start loop

for spec_type in spectrogram_types:
    # loop over different spectrogrma types

    for scale in freq_scales:
        # loop over possible scales

        for norm_type in spec_normalizations:
            # loop over spectrogram normalizations techniques

            for opt_metric in optimization_metrics:
                # loop over optimization metrics
                for opt_option in optimization_options:
                    # loop over optimization options

                    # if Test_id<1:
                    #     continue

                    print("Starting test: ", Test_id)
                    # create test result directory
                    test_result_dir = main_results_dir + '/Test_' + str(Test_id)

                    if not os.path.exists(test_result_dir):
                        os.mkdir(test_result_dir)

                    # store Test info
                    test_info_file_path = test_result_dir + '/TFR_info.txt'
                    save_test_info(test_info_file_path, spec_type, scale, norm_type, opt_metric, opt_option)

                    for signal_id in os.listdir(dataset_dir):
                        # looping over signal_directories
                        # CHECK TO SKIP IF OPT_METRIC==IATSENKO and pure_tone signal
                        if opt_metric == 'Iatsenko' and signal_id == 'pure_tone':
                            continue

                        folder_path = dataset_dir + '/' + signal_id
                        print("Analysing folder: ", folder_path)
                        # create test_dir for signal type
                        test_result_subfolder = test_result_dir + '/' + signal_id
                        if not os.path.exists(test_result_subfolder):
                            os.mkdir(test_result_subfolder)

                        # inst.frequency law
                        baseline_dir = folder_path + '/Base_Dataset_2'

                        # find optima parameters and store them
                        optima_parameters = find_optimal_spec_IF_parameters_handle(baseline_dir, test_result_subfolder,
                                                                                   signal_id, spec_type, scale,
                                                                                   norm_type, opt_metric, opt_option)
                        save_optima_parameters(test_result_subfolder, optima_parameters)

                        # evaluate metrics for "original signal" and store them
                        calculate_metrics_original_signal(baseline_dir, test_result_subfolder, signal_id, spec_type,
                                                          scale, norm_type, optima_parameters)

                        # ####################################### NOISE LEVELS  MEASURE#################################

                        # fieldnames = ['L2 if', 'KLD if', 'L2 inverted', 'KLD inverted', 'L2 inv or', 'KLD inv or',
                        #              'Renyi Entropy']
                        original_signal_name = signal_id + "_00.wav"

                        # save_directory = test_dir + "\\" + test_fold + "\\Dataset"

                        data_file = baseline_dir + "/" + original_signal_name

                        sp = SignalProc.SignalProc(optima_parameters["win_len"], optima_parameters["hop"])
                        sp.readWav(data_file)
                        # store_reference signal
                        signal_original = sp.data
                        fs = sp.sampleRate
                        T = sp.fileLength / fs
                        # assign IF function
                        inst_freq_fun = set_if_fun(signal_id, T)
                        #
                        TFR_original = sp.spectrogram(optima_parameters["win_len"], optima_parameters["hop"],
                                                      optima_parameters["window_type"], sgType=spec_type, sgScale=scale,
                                                      nfilters=optima_parameters["mel_num"])
                        # spectrogram normalizations
                        if norm_type != "Standard":
                            sp.normalisedSpec(tr=norm_type)
                            TFR_0 = sp.sg
                        TFR_original = TFR_original.T
                        del sp

                        sig_dataset_path = folder_path+'/Dataset_2'
                        noise_levels_folders = os.listdir(sig_dataset_path)
                        num_levels = len(noise_levels_folders)

                        # Fieldnames for general metrics csv
                        general_csv_fieldnames = []
                        for k in range(num_levels):
                            general_csv_fieldnames.append("Level " + str(k + 1))
                        del k

                        if signal_id == "pure_tone":
                            level_csv_fieldnames = ["SNR", "Renyi Entropy", "L2", "Geodetic", "SISDR orig.",
                                                    "SISDR noise",
                                                    "STOI orig.", "STOI noise", "IMED orig.", "IMED noise",
                                                    "Renyi Entropy inv. sound"]
                        else:
                            level_csv_fieldnames = ["SNR", "Renyi Entropy", "L2", "Iats. err.", "Geodetic",
                                                    "SISDR orig.", "SISDR noise", "STOI orig.", "STOI noise",
                                                    "IMED orig.", "IMED noise", "Renyi Entropy inv. sound"]
                        # initialise general metrics
                        SNR_G = np.zeros((100, num_levels))
                        RE_G = np.zeros((100, num_levels))
                        L2_G = np.zeros((100, num_levels))
                        if signal_id != "pure_tone":
                            IAT_ERR_G = np.zeros((100, num_levels))
                        GEODETIC_G = np.zeros((100, num_levels))
                        SISDR_original_G = np.zeros((100, num_levels))
                        SISDR_noise_G = np.zeros((100, num_levels))
                        STOI_noise_G = np.zeros((100, num_levels))
                        STOI_original_G = np.zeros((100, num_levels))
                        IMED_original_G = np.zeros((100, num_levels))
                        IMED_noise_G = np.zeros((100, num_levels))
                        RE_inv_G = np.zeros((100, num_levels))

                        # aid variable initialization
                        i = 0  # counter for noise level
                        for noise_dir in noise_levels_folders:
                            i += 1
                            print('\n Noise level: ', i)
                            csvfilename_noise_level = test_result_subfolder + '/noise_level_' + str(i) + '.csv'
                            with open(csvfilename_noise_level, 'w', newline='') as csvfile:
                                Writer = csv.DictWriter(csvfile, fieldnames=level_csv_fieldnames)
                                Writer.writeheader()

                            # aid variable initialization
                            k = 0  # counter for sample
                            for file in os.listdir(sig_dataset_path + "/" + noise_dir):
                                print('Sample ', file)
                                IF = IFreq.IF(method=2, pars=[optima_parameters["alpha"], optima_parameters["beta"]])
                                sp = SignalProc.SignalProc(optima_parameters["win_len"], optima_parameters["hop"])
                                sp.readWav(sig_dataset_path + "/" + noise_dir + '/' + file)
                                signal = sp.data
                                fs = sp.sampleRate

                                TFR = sp.spectrogram(optima_parameters["win_len"], optima_parameters["hop"],
                                                     optima_parameters["window_type"], sgType=spec_type,
                                                     sgScale=scale, nfilters=optima_parameters["mel_num"])
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
                                    nfilters = 40
                                    freqarr = np.linspace(sp.convertHztoMel(0), sp.convertHztoMel(fs / 2), nfilters + 1)
                                    freqarr = freqarr[1:]

                                wopt = [fs, optima_parameters["win_len"]]  # this neeeds review
                                tfsupp, _, _ = IF.ecurve(TFR2, freqarr, wopt)

                                # revert to Hz if Mel
                                if scale == 'Mel Frequency':
                                    tfsupp[0, :] = sp.convertMeltoHz(tfsupp[0, :])

                                # array with temporal coordinates
                                t_support = np.linspace(0, T, np.shape(tfsupp[0, :])[0])
                                inst_freq = inst_freq_fun(t_support)  # IF law

                                #spectrogram of inverted signal
                                if scale == 'Mel Frequency':
                                    # Patched
                                    F = mel_filterbank_maker(optima_parameters["win_len"], 'mel',
                                                             optima_parameters["mel_num"])
                                    F_pseudo = np.linalg.pinv(F)
                                    TFR_recovered = np.absolute(np.dot(TFR, F_pseudo))
                                    signal_inverted = sp.invertSpectrogram(TFR_recovered,
                                                                           window_width=optima_parameters["win_len"],
                                                                           incr=optima_parameters["hop"],
                                                                           window=optima_parameters["window_type"])

                                else:
                                    signal_inverted = sp.invertSpectrogram(TFR,
                                                                           window_width=optima_parameters["win_len"],
                                                                           incr=optima_parameters["hop"],
                                                                           window=optima_parameters["window_type"])

                                # generate spectrogram of inverted sound
                                sp.data = signal_inverted
                                TFR_inv = sp.spectrogram(optima_parameters["win_len"], optima_parameters["hop"],
                                                         optima_parameters["window_type"], sgType=spec_type,
                                                         sgScale=scale, nfilters=optima_parameters["mel_num"])
                                # spectrogram normalizations
                                if norm_type != "Standard":
                                    sp.normalisedSpec(tr=norm_type)
                                    TFR_inv = sp.sg

                                TFR2_inv = TFR_inv.T
                                ### evaluate metrics

                                # #base metrics
                                # snr
                                noise = signal - signal_original
                                SNR = Signal_to_noise_Ratio(signal, noise)
                                SNR_G[k, i - 1] = SNR
                                # Renyi Entropy
                                RE = Renyi_Entropy(TFR2)
                                RE_G[k, i - 1] = RE

                                # IF metrics
                                # L2 norm
                                L2 = norm(tfsupp[0, :] - inst_freq)
                                L2_G[k, i - 1] = L2
                                # IAM
                                if signal_id != "pure_tone":
                                    IAT_ERR = Iatsenko_style(inst_freq, tfsupp[0, :])
                                    IAT_ERR_G[k, i - 1] = IAT_ERR
                                # CURVE DISTANCE (Geodesic)
                                GEODETIC = Geodesic_curve_distance(t_support, tfsupp[0, :], t_support, inst_freq)
                                GEODETIC_G[k, i - 1] = GEODETIC

                                # Reconstructed signal metrics
                                # speech metrics comparison
                                len_diff = len(signal_original) - len(signal_inverted)
                                # [int(np.floor(len_diff/2)):-int(np.ceil(len_diff/2))]
                                score_original = metrics(signal_inverted,
                                                         signal_original[int(np.floor(len_diff / 2)):
                                                                         -int(np.ceil(len_diff / 2))], rate=fs)
                                STOI_original = score_original['stoi']
                                STOI_original_G[k, i - 1] = STOI_original
                                SISDR_original = score_original['sisdr']
                                SISDR_original_G[k, i - 1] = SISDR_original

                                score_noise = metrics(signal_inverted, signal, rate=fs)
                                STOI_noise = score_noise['stoi']
                                STOI_noise_G[k, i - 1] = STOI_noise
                                SISDR_noise = score_noise['sisdr']
                                SISDR_noise_G[k, i - 1] = SISDR_noise

                                # imed
                                col_dif = np.shape(TFR_original)[1] - np.shape(TFR2_inv)[1]
                                if col_dif == 0:
                                    IMED_original = IMED_distance(TFR_original, TFR2_inv)
                                    IMED_noise = IMED_distance(TFR2, TFR2_inv)
                                else:
                                    IMED_original = IMED_distance(TFR_original[:,
                                                                  int(np.floor(col_dif / 2)):-int(np.ceil(col_dif / 2))],
                                                                TFR2_inv)
                                    IMED_noise = IMED_distance( TFR2[:,
                                                                int(np.floor(col_dif / 2)):-int(np.ceil(col_dif / 2))],
                                                                TFR2_inv)
                                IMED_original_G[k, i - 1] = IMED_original

                                IMED_noise_G[k, i - 1] = IMED_noise

                                # Renyi entropy spectrogram inverted sound
                                RE_inv = Renyi_Entropy(TFR2_inv)
                                RE_inv_G[k, i - 1] = RE_inv

                                with open(csvfilename_noise_level, 'a', newline='') as csvfile:  # should be ok
                                    Writer = csv.DictWriter(csvfile, fieldnames=level_csv_fieldnames)
                                    if signal_id == "pure_tone":
                                        Writer.writerow(
                                            {"SNR": SNR, "Renyi Entropy": RE, "L2": L2, "Geodetic": GEODETIC,
                                             "SISDR orig.": SISDR_original, "SISDR noise": SISDR_noise,
                                             "STOI orig.": STOI_original, "STOI noise": STOI_noise,
                                             "IMED orig.": IMED_original, "IMED noise": IMED_noise,
                                             "Renyi Entropy inv. sound": RE_inv})
                                    else:
                                        Writer.writerow(
                                            {"SNR": SNR, "Renyi Entropy": RE, "L2": L2, "Iats. err.": IAT_ERR,
                                             "Geodetic": GEODETIC, "SISDR orig.": SISDR_original,
                                             "SISDR noise": SISDR_noise, "STOI orig.": STOI_original,
                                             "STOI noise": STOI_noise, "IMED orig.": IMED_original,
                                             "IMED noise": IMED_noise, "Renyi Entropy inv. sound": RE_inv})

                                # os.remove(aid_file)
                                k += 1
                                del IF, sp, fs, TFR, fstep, freqarr, wopt, tfsupp, signal, signal_inverted, inst_freq
                                del SNR, RE, L2, GEODETIC, SISDR_original, SISDR_noise, STOI_original, STOI_noise
                                del IMED_original, IMED_noise, RE_inv
                                if signal_id != "pure_tone":
                                    del IAT_ERR
                            del csvfilename_noise_level

                        # save a csvfile per metric
                        # SNR
                        csvfilename = test_result_subfolder + '/noise_levels_SNR.csv'
                        save_metric_csv(csvfilename, general_csv_fieldnames, SNR_G)
                        # RENYI ENTROPY
                        csvfilename = test_result_subfolder + '/noise_levels_RE.csv'
                        save_metric_csv(csvfilename, general_csv_fieldnames, RE_G)
                        # L2
                        csvfilename = test_result_subfolder + '/noise_levels_L2.csv'
                        save_metric_csv(csvfilename, general_csv_fieldnames, L2_G)
                        # IATSENKO ERROR
                        if signal_id != "pure_tone":
                            csvfilename = test_result_subfolder + '/noise_levels_Iatsenko.csv'
                            save_metric_csv(csvfilename, general_csv_fieldnames, IAT_ERR_G)
                        # GEODETIC DISTANCE
                        csvfilename = test_result_subfolder + '/noise_levels_Geodetic.csv'
                        save_metric_csv(csvfilename, general_csv_fieldnames, GEODETIC_G)
                        # SISDR respect to original signal
                        csvfilename = test_result_subfolder + '/noise_levels_SISDR_original.csv'
                        save_metric_csv(csvfilename, general_csv_fieldnames, SISDR_original_G)
                        # SISDR respect to signal+noise
                        csvfilename = test_result_subfolder + '/noise_levels_SISDR_noise.csv'
                        save_metric_csv(csvfilename, general_csv_fieldnames, SISDR_noise_G)
                        # STOI respect to original signal
                        csvfilename = test_result_subfolder + '/noise_levels_STOI_original.csv'
                        save_metric_csv(csvfilename, general_csv_fieldnames, STOI_original_G)
                        # STOI respect to signal+noise
                        csvfilename = test_result_subfolder + '/noise_levels_STOI_noise.csv'
                        save_metric_csv(csvfilename, general_csv_fieldnames, STOI_noise_G)
                        # IMED respect to original signal
                        csvfilename = test_result_subfolder + '/noise_levels_IMED_original.csv'
                        save_metric_csv(csvfilename, general_csv_fieldnames, IMED_original_G)
                        # IMED respect to signal+noise
                        csvfilename = test_result_subfolder + '/noise_levels_IMED_noise.csv'
                        save_metric_csv(csvfilename, general_csv_fieldnames, IMED_noise_G)
                        # Renyi entropy inverted sound
                        csvfilename = test_result_subfolder + '/noise_levels_RE_inverted_noise.csv'
                        save_metric_csv(csvfilename, general_csv_fieldnames, RE_inv_G)

                        del signal_original, TFR_original

                        # test over: update Test id
                    print("\n Test ", Test_id, " is over")
                    Test_id += 1
