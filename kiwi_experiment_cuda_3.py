"""
10/11/2022
Author: Virginia Listanti

This is an experiment script testing different parameter for a standard spectrogram with mel scale

1. Extract IF profiles from spectrogram of .wav files
2. Prepare curves for classification
3. Perfom classification tasks
4. Calculate accuray and save
"""

import SignalProc
import IF as IFreq
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import distances
import DTW_functions as DTW


def assign_label(D, list2, label_list, true_label_list):
    """
    This function assign label by symmetry breaking given the distance matrix D

    D distance matrix
    LIST2 is the list of test data
    LABEL_LIST list of labels for train data
    TRUE_LABEL_LIST list of true label fro LIST2
    """
    # N1 = len(list1)
    N2 = len(list2)
    accuracy = 0
    alg_labels = []
    best3_list = []

    for k in range(N2):
        distances = D[k, :]
        indices = np.argsort(distances)[:3]
        best_match = [label_list[indices[0]], label_list[indices[1]], label_list[indices[2]]]
        best3_list.append(best_match)

        if best_match[0] == best_match[1]:
            label = best_match[0]
        elif best_match[2] == best_match[1]:
            label = best_match[2]
        elif best_match[2] == best_match[0]:
            label = best_match[2]
        else:
            label = best_match[0]

        alg_labels.append(label)
        if label == true_label_list[k]:
            accuracy += 1

        del best_match

    accuracy /= N2


    return alg_labels, best3_list, accuracy


def find_best(D, list2, label_list):
    """
    This function find best 3 matches from a distance matrix D

    D distance matrix
    LIST2 is the list of test data
    LABEL_LIST list of labels for train data
    TRUE_LABEL_LIST list of true label fro LIST2
    """
    # N1 = len(list1)
    N2 = len(list2)
    best3_list = []

    for k in range(N2):
        distances = D[k, :]
        indices = np.argsort(distances)[:3]
        best_match = [label_list[indices[0]], label_list[indices[1]], label_list[indices[2]]]
        best3_list.append(best_match)

    return best3_list

def assign_label2(list2, best3_list1, best3_list2, true_label_list):
    """
    This function assign label by majority from best3_list1 and best3_list2

    LIST2 is the list of test data
    BEST3_LIST1 best 3 list from distance 1
    BEST3_LIST2 best 3 list from distance 2
    TRUE_LABEL_LIST list of true label fro LIST2
    """
    # N1 = len(list1)
    N2 = len(list2)
    accuracy = 0
    alg_labels = []

    for k in range(N2):
        freq_count = CountFrequency2(best3_list1[k], best3_list2[k])
        label = max(freq_count, key=freq_count.get)
        alg_labels.append(label)
        if label == true_label_list[k]:
            accuracy += 1
    accuracy /= N2

    return alg_labels, accuracy

def assign_label3(list2, best3_list1, best3_list2, best3_list3, true_label_list):
    """
    This function assign label by majority from best3_list1 and best3_list2

    LIST2 is the list of test data
    BEST3_LIST1 best 3 list from distance 1
    BEST3_LIST2 best 3 list from distance 2
    BEST3_LIST3 best 3 list from distance 3
    TRUE_LABEL_LIST list of true label fro LIST2
    """
    # N1 = len(list1)
    N2 = len(list2)
    accuracy = 0
    alg_labels = []

    for k in range(N2):
        freq_count = CountFrequency3(best3_list1[k], best3_list2[k], best3_list3[k])
        label = max(freq_count, key=freq_count.get)
        alg_labels.append(label)
        if label == true_label_list[k]:
            accuracy += 1
    accuracy /= N2

    return alg_labels, accuracy

def CountFrequency2(my_list1, my_list2):

    """
    This function counts the frequencies in two lists using a dictionary
    """
    # Creating an empty dictionary
    freq = {}
    for item in my_list1:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    for item in my_list2:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    return freq

def CountFrequency3(my_list1, my_list2, my_list3):

    """
    This function counts the frequencies in three lists using a dictionary
    """
    # Creating an empty dictionary
    freq = {}
    for item in my_list1:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    for item in my_list2:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    for item in my_list3:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    return freq

def normalise_matrix(A):
    """
    This function normalise a Matrix A
    """

    range_A = np.abs(np.max(A) - np.min(A))


    return (A-np.min(A))/range_A

def extract_IF(syllable_path, spec_par, file_id, Save_dir):
    "This function extract IF with given TFR parameters"

    IF = IFreq.IF(method=2, pars=[spec_par['alpha'], spec_par['beta']])
    sp = SignalProc.SignalProc(spec_par['win_len'], spec_par['hop'])
    sp.readWav(syllable_path)
    # signal = sp.data
    fs = sp.sampleRate
    T = sp.fileLength / fs

    TFR = sp.spectrogram(spec_par['win_len'], spec_par['hop'], spec_par['window_type'], sgType=spec_par['spec_type'],
                         sgScale=spec_par['scale'], nfilters=spec_par['mel_num'])
    # spectrogram normalizations
    if spec_par['norm_type'] != "Standard":
        sp.normalisedSpec(tr=spec_par['norm_type'])
        TFR = sp.sg
    TFR2 = TFR.T
    t_step = T / np.shape(TFR2)[1]

    # appropriate frequency scale
    if spec_par['scale'] == "Linear":
        fstep = (fs / 2) / np.shape(TFR2)[0]
        freqarr = np.arange(fstep, fs / 2 + fstep, fstep)
        f_band = 800
        band_index = int(np.ceil(f_band / fstep - 1))
    else:
        # #mel freq axis
        nfilters = spec_par['mel_num']
        freqarr = np.linspace(sp.convertHztoMel(0), sp.convertHztoMel(fs / 2), nfilters + 1)
        freqarr = freqarr[1:]
        f_band = sp.convertMeltoHz(800)
        band_index = 0
        while freqarr[band_index]<=f_band and band_index<len(freqarr):
            band_index += 1

    # bandpass spectrogram below 800 Hz
    TFR2[0:band_index, :] = 0

    wopt = [fs, spec_par['win_len']]
    tfsupp, _, _ = IF.ecurve(TFR2, freqarr, wopt)
    TFR3 = np.copy(TFR2)
    # revert to Hz if Mel
    if spec_par['scale'] == 'Mel Frequency':
        tfsupp[:, :] = sp.convertMeltoHz(tfsupp[0, :])

    # hardcoded check
    f_jumps = np.zeros((len(tfsupp[0, :], )))
    for k in range(len(tfsupp[0, :]) - 2):
        f_jumps[k] = tfsupp[0, k + 2] - tfsupp[0, k]
    freq_jump_boundary = 700
    if np.amax(np.abs(f_jumps)) > freq_jump_boundary:
        del IF
        print("detected jump: correcting")
        IF = IFreq.IF(method=2, pars=[spec_par['alpha'], spec_par['beta']])
        jump_index = np.argmax(np.abs(f_jumps))
        print(jump_index)
        if f_jumps[jump_index] > 0:
            # if we are doing a step up we will focus on the first half
            f_min = np.amin(tfsupp[1, 0:jump_index + 1])
            f_max = np.amax(tfsupp[2, 0:jump_index + 1])
        else:
            f_min = np.amin(tfsupp[1, jump_index + 1:])
            f_max = np.amax(tfsupp[2, jump_index + 1:])
        if spec_par['scale'] == 'Mel Frequency':
            f_min = sp.convertHztoMel(f_min)
            min_index = 0
            while freqarr[min_index] <= f_min and min_index < len(freqarr): #here
                min_index += 1

            f_max = sp.convertHztoMel(f_max)
            max_index = 0
            while freqarr[max_index] <= f_max and max_index < len(freqarr):  # here
                max_index += 1
        else:
            min_index = int(np.floor(f_min / fstep - 1))
            max_index = int(np.ceil(f_max / fstep - 1))
        TFR3[0:min_index] = 0
        TFR3[max_index + 1:] = 0
        tfsupp2, _, _ = IF.ecurve(TFR3, freqarr, wopt)
        if_syllable = np.copy(tfsupp2[0, :])
        low_bound = np.copy(tfsupp2[1, :])
        high_bound = np.copy(tfsupp2[2, :])
        # revert to Hz if Mel
        if spec_par['scale'] == 'Mel Frequency':
            if_syllable = sp.convertMeltoHz(if_syllable)
            low_bound = sp.convertMeltoHz(low_bound)
            high_bound = sp.convertMeltoHz(high_bound)
    else:
        #for Mel-scale already converted
        if_syllable = np.copy(tfsupp[0, :])
        low_bound = np.copy(tfsupp[1, :])
        high_bound = np.copy(tfsupp[2, :])



    # START AND ENDING POLISHING
    # find spec_intensity array
    spec_intensity = np.zeros(np.shape(if_syllable)[0])
    for t_index in range(len(if_syllable)):
        if spec_par['scale'] == 'Mel Frequency':
            freq_check = sp.convertHztoMel(if_syllable[t_index])
            f_index = 0
            while freqarr[f_index] <= freq_check and f_index < len(freqarr): #here
                f_index += 1
        else:
            f_index = int(np.ceil(if_syllable[t_index] / fstep - 1))
        spec_intensity[t_index] = TFR3[f_index, t_index]

    # borders check
    T2 = np.copy(T)
    if_syllable_copy = np.copy(if_syllable)
    threshold = np.amax(spec_intensity) * (5 / 100)
    start_index = int(np.floor(len(if_syllable) / 4))  # first index safe from check
    last_index = int(np.floor(len(if_syllable) * (3 / 4)))  # last index safe from check

    # check start syllable
    check_index = 0
    while (check_index < start_index) and (spec_intensity[check_index] < threshold):
        if_syllable[check_index] = np.nan  # update syllable
        T2 -= t_step  # update time_lenght
        check_index += 1

    # check end syllable
    # find first index where to start deleting
    del_index = len(if_syllable_copy) - 1
    for check_index2 in range(len(if_syllable_copy) - 1, last_index - 1, -1):
        if spec_intensity[check_index2] < threshold:
            del_index = check_index2
        else:
            break

    for canc_index in range(del_index, len(if_syllable_copy)):
        if_syllable[canc_index] = np.nan  # flag_index
        T2 -= t_step  # update time_lenght

    # update syllable
    index_list = np.argwhere(np.isnan(if_syllable))
    if_syllable = np.delete(if_syllable, index_list)
    TFR2 = np.delete(TFR2, index_list, 1)

    # array with temporal coordinates
    t_support = np.linspace(0, T2, np.shape(if_syllable)[0])

    IF_syllable = np.zeros((np.shape(if_syllable)[0], 2))
    IF_syllable[:, 0] = t_support
    IF_syllable[:, 1] = if_syllable

    # SAVE IF into .csv
    csvfilename = Save_dir+ '/'+ file_id + "_extracted_IF.csv"
    fieldnames = ['t', "IF"]
    with open(csvfilename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(if_syllable)):
            writer.writerow({"t": t_support[i], "IF": if_syllable[i]})

    # plotting
    #if Mel scale translate
    if spec_par['scale'] == 'Mel Frequency':
        syl_len = len(if_syllable)
        if_syl_discr = np.zeros(np.shape(if_syllable))
        for i in range(syl_len):
            f_index = 0
            freq_check = sp.convertHztoMel(if_syllable[i])
            while freqarr[f_index] <= freq_check and f_index < len(freqarr): #here
                f_index += 1
            if_syl_discr[i] =f_index
    # save picture
    fig_name = Save_dir+ '/'+ file_id  + "_Fourier_ridges_1.jpg"
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(1, 4, figsize=(10, 20), sharex=True)
    ax[0].imshow(np.flipud(TFR2), extent=[0, np.shape(TFR2)[1], 0, np.shape(TFR2)[0]], aspect='auto')
    x = np.array(range(np.shape(TFR2)[1]))
    if spec_par['scale'] == 'Mel Frequency':
        ax[0].plot(x, if_syl_discr, linewidth=2, color='r')
    else:
        ax[0].plot(x, if_syllable / fstep, linewidth=2, color='r')
    ax[1].imshow(np.flipud(TFR3), extent=[0, np.shape(TFR3)[1], 0, np.shape(TFR3)[0]], aspect='auto')
    ax[2].plot(x, if_syllable, color='r')
    ax[2].set_ylim([0, fs / 2])
    x = np.array(range(np.shape(TFR3)[1]))
    ax[3].plot(x, if_syllable_copy, color='r')
    ax[3].set_ylim([0, fs / 2])
    fig.suptitle(file[:-4] + ' ridges')
    plt.savefig(fig_name)
    # del IF


    return IF_syllable


def prepare_curves(extracted_curves, reference_curve, len_list, min_len):
    "This function prepare the curves for classification aligning them to reference_curve"

    M = np.shape(extracted_curves)[0]
    #new curves
    new_curves = np.zeros((M, min_len, 2))
    # new points
    new_times = np.linspace(0, 1, min_len)

    for I in range(M):
        # dynamic time warping
        target_curve = extracted_curves[I, : len_list[I], 1]
        m = DTW.dtw(target_curve, reference_curve, wantDistMatrix=True)
        x, y = DTW.dtw_path(m)
        aligned_times = np.linspace(0, 1, len(x))
        aligned_curve = target_curve[x]
        # subratct average
        aligned_curve -= np.mean(aligned_curve)
        # resample
        new_curve = np.interp(new_times, aligned_times, aligned_curve)
        new_curves[I,:,0] = new_times
        new_curves[I, :, 1] = new_curve

    return new_curves

def save_curve(newdirectory, curves, list_files):
    "This function saves the frequency information of if curves on a .csv file"
    M = np.shape(curves)[0]
    fieldnames = ['t', "IF"]
    for i in range(M):
        csvfilename = newdirectory + "/" + list_files[i]+'.csv'
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for j in range(np.shape(curves[i])[0]):
                writer.writerow({"t": curves[i,j,0], "IF": curves[i,j,1]})
    return

def calculate_distances(prep_curves1, prep_curves2, extracted_curves1, extracted_curves2, lenghts1, lengths2):
    "This function evaluate the distance matrices"

    N1 = np.shape(prep_curves1)[0]
    N2= np.shape(prep_curves2)[0]
    # pre-allocate distance_matrices
    ssd_m = np.zeros((N2, N1))
    geod_m = np.zeros((N2, N1))
    pca_m = np.zeros((N2, N1))
    dtw_m = np.zeros((N2, N1))
    df_m = np.zeros((N2, N1))  # matrix of difference of mean frequencies

    for I in range(N2):
        # prepare curves row by row
        new_reference_curve = np.copy(prep_curves2[I, :, :])
        new_reference_freq_curve = np.copy(extracted_curves2[I, :lengths2[I], :])
        new_curves = prep_curves1

        # # evaluate PCA distance vector
        pca_m[I, :] = distances.pca_distance_vector(new_curves[:, :, 1], new_reference_curve[:, 1])

        for J in range(N1):
            ssd_m[I, ] = distances.ssd(new_reference_curve[:, 1], new_curves[J, :, 1])
            geod_m[I, J] = distances.Geodesic_curve_distance(new_reference_curve[:, 0], new_reference_curve[:, 1],
                                                                  new_curves[J, :, 0], new_curves[J, :, 1])
            dtw_m[I, J] = distances.dtw(new_reference_curve[:, 1], new_curves[J, :, 1])
            df_m[I, J] = np.abs(np.mean(new_reference_freq_curve[:, 1]) -
                                     np.mean(extracted_curves1[J, :lenghts1[J], 1]))

    return ssd_m, geod_m, pca_m, dtw_m, df_m


################################################# MAIN ################################################################

dataset_folder = "/am/state-opera/home1/listanvirg/Documents/Individual_identification/Kiwi_syllable_dataset"
train_dataset_path  = "/am/state-opera/home1/listanvirg/Documents/Individual_identification/Dataset_list"
result_dir = '/am/state-opera/home1/listanvirg/Documents/Individual_identification/Test_results'

#read classes from train dataset
list_labels = ["D", "E", "J", "K", "L", "M", "O", "R", "Z"]

#recover train and test list
test_dataset_list_path =  train_dataset_path + '/Test_file_list.csv'

test_list = np.loadtxt(test_dataset_list_path, skiprows=1, delimiter=',', dtype=str)[:, 0]
list_true_labels = np.loadtxt(test_dataset_list_path, skiprows=1, delimiter=',', dtype=str)[:, 1]


mel_list = [None]
norm_list = ["Standard", 'PCEN']
spectrogram_parameters = {'spec_type': "Reassigned", 'scale': 'Mel Frequency', 'norm_type': None, 'win_len': None, 'hop': None,
                          'window_type': None, 'mel_num': None, 'alpha': None, 'beta': None}

Test_id = 700

for norm_type in norm_list:
    spectrogram_parameters['norm_type'] = norm_type
    print(norm_type)
    if norm_type == "Standard":
        window_parameters_list = [[512, 460], [256, 230], [256, 192], [128, 96]]
        Iatsenko_parameters_list = [[0.5, 0], [2.5, 0]]
        window_type_list = ['Hann', "Welch", 'Blackman']
        mel_list = [64, 128, 256]
        # print('check')
    elif norm_type == 'PCEN':
        window_parameters_list = [[512, 460], [512, 384], [256, 230], [256, 192], [128, 115]]
        Iatsenko_parameters_list = [[0.5, 0], [2.5, 0], [5, 0], [7.5, 0]]
        window_type_list = ['Hann', "Welch", 'Blackman']
        mel_list = [64, 128, 256]

    for window_type in window_type_list:
        spectrogram_parameters['window_type'] = window_type
        for window_parameters in window_parameters_list:
            spectrogram_parameters['win_len'] = window_parameters[0]
            spectrogram_parameters['hop'] = window_parameters[1]

            for mel_num in mel_list:
                spectrogram_parameters['mel_num'] = mel_num
                for Iatsenko_parameters in Iatsenko_parameters_list:
                    spectrogram_parameters['alpha'] = Iatsenko_parameters[0]
                    spectrogram_parameters['beta'] = Iatsenko_parameters[1]

                    print('\n\n Starting Test ', Test_id)
                    print(spectrogram_parameters)


                    save_dir = result_dir + '/Test_' + str(Test_id)

                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)

                    save_path_par = save_dir + "/Test_parameters.txt"

                    f = open(save_path_par, 'w')
                    f.write(str(spectrogram_parameters))
                    f.close()

                    save_dir_IF = save_dir + "/Dataset"
                    if not os.path.exists(save_dir_IF):
                        os.mkdir(save_dir_IF)

                    list_train_syllables_path = []
                    # list_train_path = []
                    list_train_labels = []
                    len_list1 = []
                    len_freq_list1 = []
                    list_train_files =[]
                    list_train_freq_path =[]
                    train_extracted_IF = np.zeros((46, 40000, 2)) # we already know that we have 46 train file,
                    # file lenght maximised

                    i = 0
                    #read files and extract IF
                    for label in list_labels:
                        train_dataset_list_path = train_dataset_path + "/Class_"+label+".csv"
                        train_list = np.loadtxt(train_dataset_list_path, skiprows=1, delimiter=',', dtype=str)
                        for file in train_list:
                            file_wav = file[:-7] +'.wav'
                            file_path = dataset_folder + '/' + file_wav
                            list_train_files.append(file[:-7])
                            list_train_syllables_path.append(file_path)
                            list_train_labels.append(label)
                            syllable_curve = extract_IF(file_path, spectrogram_parameters, file[:-7],
                                                        save_dir_IF)
                            len_syl = np.shape(syllable_curve)[0]
                            len_list1.append(len_syl)
                            train_extracted_IF[i,:len_syl,:] = syllable_curve
                            i += 1
                    n1 = len(list_train_syllables_path)

                    # read label for test dataset
                    list_syllables = []
                    list_syllables_path = []
                    len_list2 = []
                    len_freq_list2 = []
                    test_extracted_IF = np.zeros((49, 40000, 2)) # we already know that we have 46 train file,
                    i = 0
                    for file in test_list:
                        file_wav = file[:-7] + '.wav'
                        file_path = dataset_folder + '/' + file_wav
                        list_syllables.append(file[:-7])
                        list_syllables_path.append(file_path)
                        syllable_curve = extract_IF(file_path, spectrogram_parameters,file[:-7],
                                                        save_dir_IF)
                        len_syl = np.shape(syllable_curve)[0]
                        len_list2.append(len_syl)
                        test_extracted_IF[i, :len_syl, :] = syllable_curve
                        i += 1

                    n2 = len(list_syllables_path)
                    len_max = max(np.max(len_list1), np.max(len_list2))
                    len_min = min(np.min(len_list1), np.min(len_list2))
                    # len_max_freq = max(np.max(len_freq_list1), np.max(len_freq_list2))

                    # number of files
                    n = n1 + n2

                    reference_curve = train_extracted_IF[0, :len_list1[0], 1]
                    #prepare train curves
                    prepared_train_curves = prepare_curves(train_extracted_IF, reference_curve, len_list1, len_min)
                    save_curve(save_dir_IF, prepared_train_curves, list_train_files)

                    # prepare test curves
                    prepared_test_curves = prepare_curves(test_extracted_IF, reference_curve, len_list2, len_min)
                    save_curve(save_dir_IF, prepared_test_curves, list_syllables)

                    # iniziatlization
                    num_label = len(list_labels)
                    # accuracy


                    ssd_matrix, geod_matrix, pca_matrix, dtw_matrix, df_matrix = calculate_distances(prepared_train_curves,
                                                                                                     prepared_test_curves,
                                                                                                     train_extracted_IF,
                                                                                                     test_extracted_IF,
                                                                                                     len_list1,
                                                                                                     len_list2)
                    # normalise matrices

                    df_matrix = normalise_matrix(df_matrix)
                    ssd_matrix_1 = normalise_matrix(ssd_matrix) + df_matrix
                    pca_matrix_1 = normalise_matrix(pca_matrix) + df_matrix
                    dtw_matrix_1 = normalise_matrix(dtw_matrix) + df_matrix
                    geod_matrix_1 = normalise_matrix(geod_matrix) + df_matrix

                    list_assigned_labels_ssd, best3_list_ssd, accuracy_ssd = assign_label(ssd_matrix, list_syllables,
                                                                                          list_train_labels,
                                                                                          list_true_labels)
                    list_assigned_labels_pca, best3_list_pca, accuracy_pca = assign_label(pca_matrix, list_syllables,
                                                                                          list_train_labels,
                                                                                          list_true_labels)
                    list_assigned_labels_dtw, best3_list_dtw, accuracy_dtw = assign_label(dtw_matrix, list_syllables,
                                                                                          list_train_labels,
                                                                                          list_true_labels)
                    list_assigned_labels_geod, best3_list_geod, accuracy_geod = assign_label(geod_matrix,
                                                                                             list_syllables,
                                                                                             list_train_labels,
                                                                                             list_true_labels)

                    list_assigned_labels_ssd_1, best3_list_ssd_1, accuracy_ssd_1 = assign_label(ssd_matrix_1, list_syllables,
                                                                                          list_train_labels,
                                                                                          list_true_labels)
                    list_assigned_labels_pca_1, best3_list_pca_1, accuracy_pca_1 = assign_label(pca_matrix_1, list_syllables,
                                                                                          list_train_labels,
                                                                                          list_true_labels)
                    list_assigned_labels_dtw_1, best3_list_dtw_1, accuracy_dtw_1 = assign_label(dtw_matrix_1, list_syllables,
                                                                                          list_train_labels,
                                                                                          list_true_labels)
                    list_assigned_labels_geod_1, best3_list_geod_1, accuracy_geod_1 = assign_label(geod_matrix_1,
                                                                                             list_syllables,
                                                                                             list_train_labels,
                                                                                             list_true_labels)

                    # combine 2

                    list_assigned_labels_ssd_dtw, accuracy_ssd_dtw = assign_label2(list_syllables, best3_list_ssd,
                                                                                  best3_list_dtw,
                                                                                  list_true_labels)
                    list_assigned_labels_ssd_pca, accuracy_ssd_pca = assign_label2(list_syllables, best3_list_ssd,
                                                                                  best3_list_pca,
                                                                                  list_true_labels)
                    list_assigned_labels_ssd_geo, accuracy_ssd_geo = assign_label2(list_syllables, best3_list_ssd,
                                                                                  best3_list_geod,
                                                                                  list_true_labels)
                    list_assigned_labels_dtw_pca, accuracy_dtw_pca = assign_label2(list_syllables, best3_list_dtw,
                                                                                  best3_list_pca,
                                                                                  list_true_labels)
                    list_assigned_labels_dtw_geo, accuracy_dtw_geo = assign_label2(list_syllables, best3_list_dtw,
                                                                                  best3_list_geod,
                                                                                  list_true_labels)
                    list_assigned_labels_pca_geo, accuracy_pca_geo = assign_label2(list_syllables, best3_list_pca,
                                                                                  best3_list_geod,
                                                                                  list_true_labels)

                    # combine 2 with frequency

                    list_assigned_labels_ssd_dtw_1, accuracy_ssd_dtw_1 = assign_label2(list_syllables, best3_list_ssd_1,
                                                                                   best3_list_dtw_1,
                                                                                   list_true_labels)
                    list_assigned_labels_ssd_pca_1, accuracy_ssd_pca_1 = assign_label2(list_syllables, best3_list_ssd_1,
                                                                                   best3_list_pca_1,
                                                                                   list_true_labels)
                    list_assigned_labels_ssd_geo_1, accuracy_ssd_geo_1 = assign_label2(list_syllables, best3_list_ssd_1,
                                                                                   best3_list_geod_1,
                                                                                   list_true_labels)
                    list_assigned_labels_dtw_pca_1, accuracy_dtw_pca_1 = assign_label2(list_syllables, best3_list_dtw_1,
                                                                                   best3_list_pca_1,
                                                                                   list_true_labels)
                    list_assigned_labels_dtw_geo_1, accuracy_dtw_geo_1 = assign_label2(list_syllables, best3_list_dtw_1,
                                                                                   best3_list_geod_1,
                                                                                   list_true_labels)
                    list_assigned_labels_pca_geo_1, accuracy_pca_geo_1 = assign_label2(list_syllables, best3_list_pca_1,
                                                                                   best3_list_geod_1,
                                                                                   list_true_labels)

                    #combine 3

                    list_assigned_labels_ssd_dtw_pca, accuracy_ssd_dtw_pca = assign_label3(list_syllables,
                                                                                          best3_list_ssd,
                                                                                          best3_list_dtw,
                                                                                          best3_list_pca,
                                                                                          list_true_labels)
                    list_assigned_labels_ssd_dtw_geo, accuracy_ssd_dtw_geo = assign_label3(list_syllables,
                                                                                          best3_list_ssd,
                                                                                          best3_list_dtw,
                                                                                          best3_list_geod,
                                                                                          list_true_labels)
                    list_assigned_labels_ssd_pca_geo, accuracy_ssd_pca_geo = assign_label3(list_syllables,
                                                                                          best3_list_ssd,
                                                                                          best3_list_pca,
                                                                                          best3_list_geod,
                                                                                          list_true_labels)
                    list_assigned_labels_dtw_pca_geo, accuracy_dtw_pca_geo = assign_label3(list_syllables,
                                                                                          best3_list_dtw,
                                                                                          best3_list_pca,
                                                                                          best3_list_geod,
                                                                                          list_true_labels)

                    # combine 3 with frequency

                    list_assigned_labels_ssd_dtw_pca_1, accuracy_ssd_dtw_pca_1 = assign_label3(list_syllables,
                                                                                           best3_list_ssd_1,
                                                                                           best3_list_dtw_1,
                                                                                           best3_list_pca_1,
                                                                                           list_true_labels)
                    list_assigned_labels_ssd_dtw_geo_1, accuracy_ssd_dtw_geo_1 = assign_label3(list_syllables,
                                                                                           best3_list_ssd_1,
                                                                                           best3_list_dtw_1,
                                                                                           best3_list_geod_1,
                                                                                           list_true_labels)
                    list_assigned_labels_ssd_pca_geo_1, accuracy_ssd_pca_geo_1 = assign_label3(list_syllables,
                                                                                           best3_list_ssd_1,
                                                                                           best3_list_pca_1,
                                                                                           best3_list_geod_1,
                                                                                           list_true_labels)
                    list_assigned_labels_dtw_pca_geo_1, accuracy_dtw_pca_geo_1 = assign_label3(list_syllables,
                                                                                           best3_list_dtw_1,
                                                                                           best3_list_pca_1,
                                                                                           best3_list_geod_1,
                                                                                           list_true_labels)

                    csvfilename = save_dir + "/" + "Labels_comparison_pure.csv"
                    fieldnames = ['Syllable', 'True Label', 'SSD label', 'PCA label', 'DTW label', 'GEO label']
                    with open(csvfilename, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                        for i in range(n2):
                            dictionary = {'Syllable': list_syllables[i], 'True Label': list_true_labels[i],
                                          'SSD label': list_assigned_labels_ssd[i],
                                          'PCA label': list_assigned_labels_pca[i],
                                          'DTW label': list_assigned_labels_dtw[i],
                                          'GEO label': list_assigned_labels_geod[i]}

                            writer.writerow(dictionary)
                            del dictionary

                    csvfilename = save_dir + "/" + "Labels_comparison_df.csv"
                    fieldnames = ['Syllable', 'True Label', 'SSD label', 'PCA label', 'DTW label', 'GEO label']
                    with open(csvfilename, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                        for i in range(n2):
                            dictionary = {'Syllable': list_syllables[i], 'True Label': list_true_labels[i],
                                          'SSD label': list_assigned_labels_ssd_1[i],
                                          'PCA label': list_assigned_labels_pca_1[i],
                                          'DTW label': list_assigned_labels_dtw_1[i],
                                          'GEO label': list_assigned_labels_geod_1[i]}

                            writer.writerow(dictionary)
                            del dictionary

                    csvfilename = save_dir + "/" + "Labels_comparison_combine2.csv"
                    fieldnames = ['Syllable', 'True Label', 'SSD + DTW label', 'SSD + PCA label', 'SSD + GEO label',
                                  'DTW + PCA label',
                                  'DTW + GEO label', 'PCA + GEO label']
                    with open(csvfilename, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                        for i in range(n2):
                            dictionary = {'Syllable': list_syllables[i], 'True Label': list_true_labels[i],
                                          'SSD + DTW label': list_assigned_labels_ssd_dtw[i], 'SSD + PCA label':
                                              list_assigned_labels_ssd_pca[i],
                                          'SSD + GEO label': list_assigned_labels_ssd_geo[i],
                                          'DTW + PCA label': list_assigned_labels_dtw_pca[i], 'DTW + GEO label':
                                              list_assigned_labels_dtw_geo[i],
                                          'PCA + GEO label': list_assigned_labels_pca_geo[i]}

                            writer.writerow(dictionary)
                            del dictionary

                    csvfilename = save_dir + "/" + "Labels_comparison_combine2_df.csv"
                    fieldnames = ['Syllable', 'True Label', 'SSD + DTW label', 'SSD + PCA label', 'SSD + GEO label',
                                  'DTW + PCA label',
                                  'DTW + GEO label', 'PCA + GEO label']
                    with open(csvfilename, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                        for i in range(n2):
                            dictionary = {'Syllable': list_syllables[i], 'True Label': list_true_labels[i],
                                          'SSD + DTW label': list_assigned_labels_ssd_dtw_1[i], 'SSD + PCA label':
                                              list_assigned_labels_ssd_pca_1[i],
                                          'SSD + GEO label': list_assigned_labels_ssd_geo_1[i],
                                          'DTW + PCA label': list_assigned_labels_dtw_pca_1[i], 'DTW + GEO label':
                                              list_assigned_labels_dtw_geo_1[i],
                                          'PCA + GEO label': list_assigned_labels_pca_geo_1[i]}

                            writer.writerow(dictionary)
                            del dictionary

                    csvfilename = save_dir + "/" + "Labels_comparison_combine3.csv"
                    fieldnames = ['Syllable', 'True Label', 'SSD + DTW + PCA label', 'SSD + DTW + GEO label',
                                  'SSD + PCA +GEO label',
                                  'DTW + PCA + GEO label']
                    with open(csvfilename, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                        for i in range(n2):
                            dictionary = {'Syllable': list_syllables[i], 'True Label': list_true_labels[i],
                                          'SSD + DTW + PCA label': list_assigned_labels_ssd_dtw_pca[i],
                                          'SSD + DTW + GEO label':
                                              list_assigned_labels_ssd_dtw_geo[i],
                                          'SSD + PCA +GEO label': list_assigned_labels_ssd_pca_geo[i],
                                          'DTW + PCA + GEO label': list_assigned_labels_dtw_pca_geo[i]}

                            writer.writerow(dictionary)
                            del dictionary

                    csvfilename = save_dir + "/" + "Labels_comparison_combine3_df.csv"
                    fieldnames = ['Syllable', 'True Label', 'SSD + DTW + PCA label', 'SSD + DTW + GEO label',
                                  'SSD + PCA +GEO label',
                                  'DTW + PCA + GEO label']
                    with open(csvfilename, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                        for i in range(n2):
                            dictionary = {'Syllable': list_syllables[i], 'True Label': list_true_labels[i],
                                          'SSD + DTW + PCA label': list_assigned_labels_ssd_dtw_pca_1[i],
                                          'SSD + DTW + GEO label':
                                              list_assigned_labels_ssd_dtw_geo_1[i],
                                          'SSD + PCA +GEO label': list_assigned_labels_ssd_pca_geo_1[i],
                                          'DTW + PCA + GEO label': list_assigned_labels_dtw_pca_geo_1[i]}

                            writer.writerow(dictionary)
                            del dictionary



                    # save best3 match
                    csvfilename = save_dir + "/" + "Best3_comparison.csv"
                    fieldnames = ['Syllable', 'SSD', 'PCA', 'DTW', 'GEO']
                    with open(csvfilename, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                        for i in range(n2):
                            dictionary = {'Syllable': list_syllables[i], 'SSD': best3_list_ssd[i],
                                          'PCA': best3_list_pca[i],
                                          'DTW': best3_list_dtw[i], 'GEO': best3_list_geod[i]}

                            writer.writerow(dictionary)
                            del dictionary

                    # save best3 match df
                    csvfilename = save_dir + "/" + "Best3_comparison_df.csv"
                    fieldnames = ['Syllable', 'SSD', 'PCA', 'DTW', 'GEO']
                    with open(csvfilename, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                        for i in range(n2):
                            dictionary = {'Syllable': list_syllables[i], 'SSD': best3_list_ssd_1[i],
                                          'PCA': best3_list_pca_1[i],
                                          'DTW': best3_list_dtw_1[i], 'GEO': best3_list_geod_1[i]}

                            writer.writerow(dictionary)
                            del dictionary



                    # print accuracies in txt files
                    file_path = save_dir + '/Accuracy_results.txt'
                    file_txt = open(file_path, 'w')
                    l0 = [" Accuracy results \n"]
                    l1 = ["\n SSD Accuracy: " + str(accuracy_ssd)]
                    l2 = ["\n PCA Accuracy: " + str(accuracy_pca)]
                    l3 = ["\n DTW Accuracy: " + str(accuracy_dtw)]
                    l4 = ["\n Geodesic Accuracy: " + str(accuracy_geod)]
                    l5 = ["\n\n SSD + df Accuracy: " + str(accuracy_ssd_1)]
                    l6 = ["\n PCA + df Accuracy: " + str(accuracy_pca_1)]
                    l7 = ["\n DTW + df  Accuracy: " + str(accuracy_dtw_1)]
                    l8 = ["\n Geodesic + dfAccuracy: " + str(accuracy_geod_1)]
                    l9 = ["\n\n SSD + DTW Accuracy: " + str(accuracy_ssd_dtw)]
                    l10 = ["\n SSD + PCA Accuracy: " + str(accuracy_ssd_pca)]
                    l11 = ["\n SSD + GEO Accuracy: " + str(accuracy_ssd_geo)]
                    l12 = ["\n DTW + PCA Accuracy: " + str(accuracy_dtw_pca)]
                    l13 = ["\n DTW + GEO Accuracy: " + str(accuracy_dtw_geo)]
                    l14 = ["\n PCA + GEO Accuracy: " + str(accuracy_pca_geo)]
                    l15 = ["\n\n SSD + DTW Accuracy(with df): " + str(accuracy_ssd_dtw_1)]
                    l16 = ["\n SSD + PCA Accuracy(with df): " + str(accuracy_ssd_pca_1)]
                    l17 = ["\n SSD + GEO Accuracy(with df): " + str(accuracy_ssd_geo_1)]
                    l18 = ["\n DTW + PCA Accuracy(with df): " + str(accuracy_dtw_pca_1)]
                    l19 = ["\n DTW + GEO Accuracy(with df): " + str(accuracy_dtw_geo_1)]
                    l20 = ["\n PCA + GEO Accuracy(with df): " + str(accuracy_pca_geo_1)]
                    l21 = ["\n\n SSD + DTW + PCA Accuracy: " + str(accuracy_ssd_dtw_pca)]
                    l22 = ["\n SSD + DTW + GEO Accuracy: " + str(accuracy_ssd_dtw_geo)]
                    l23 = ["\n SSD + PCA + GEO Accuracy: " + str(accuracy_ssd_pca_geo)]
                    l24 = ["\n DTW + PCA + GEO Accuracy: " + str(accuracy_dtw_pca_geo)]
                    l25 = ["\n\n SSD + DTW + PCA Accuracy(with df): " + str(accuracy_ssd_dtw_pca_1)]
                    l26 = ["\n SSD + DTW + GEO Accuracy(with df): " + str(accuracy_ssd_dtw_geo_1)]
                    l27 = ["\n SSD + PCA + GEO Accuracy(with df): " + str(accuracy_ssd_pca_geo_1)]
                    l28 = ["\n DTW + PCA + GEO Accuracy(with df): " + str(accuracy_dtw_pca_geo_1)]
                    file_txt.writelines(np.concatenate((l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14,
                                                        l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27,
                                                        l28)))
                    file_txt.close()

                    # save matrices
                    np.savetxt(save_dir + "/SSD.txt", ssd_matrix, fmt='%s')
                    np.savetxt(save_dir + "/Geodesic.txt", geod_matrix, fmt='%s')
                    np.savetxt(save_dir + "/PCA.txt", pca_matrix, fmt='%s')
                    np.savetxt(save_dir + "/DTW.txt", dtw_matrix, fmt='%s')
                    np.savetxt(save_dir + "/Duration.txt", df_matrix, fmt='%s')

                    Test_id +=1


