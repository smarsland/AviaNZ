"""
28/10/2022
Reviewed: 25/11/2022
Author: Virginia Listanti

This script manage the experiment to find the best metric to perform the classification task on
 kiwi syllables

In this script we test the use of the difference of median averages fro original frequency profile

The experiment run on 10 classes

This experiments evaluate the metrics on 5 different train-test samples

Run on prep curves
run on cuda machines
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import distances
import csv


def moving_average(s, win_len):
    """
    This function smooths the signal s with a moving average filter
    """
    N = len(s)
    half_win = int(np.floor(win_len / 2))
    new_s = []

    for I in range(half_win):
        new_s.append(np.mean(s[:I + half_win + 1]))

    for I in range(half_win, N - (half_win - 1)):
        new_s.append(np.mean(s[I - half_win: I + half_win + 1]))

    for I in range(N - (half_win - 1), N):
        new_s.append(np.mean(s[I - half_win:]))

    return np.array(new_s)


def prepare_curves(curves_old, ref_curve, length_list, M):
    """
    This function prepare curves alignind them respect to reference curves

    Alignment using dynamic time-warping
    Resampling with min_lenght

    """
    N = np.shape(curves_old)[0]
    curves_new = np.zeros((N, M, 2))
    new_times = np.linspace(0, 1, M)
    new_ref_curve = np.zeros((M,2))

    for i in range(N):
        # dynamic time warping
        target_curve = curves_old[i, :length_list[i], 1]
        m = distances.dtw(target_curve, ref_curve[:, 1], wantDistMatrix=True)
        x, y = distances.dtw_path(m)
        aligned_times = np.linspace(0, 1, len(x))
        aligned_curve = target_curve[x]
        # subratct average
        aligned_curve -= np.mean(aligned_curve)
        # resample
        curves_new[i, :, 0] = new_times
        curves_new[i, :, 1] = np.interp(new_times, aligned_times, aligned_curve)

    new_ref_curve[:, 0] = new_times
    new_ref_curve[:,1] = np.interp(new_times, ref_curve[:, 0], ref_curve[:, 1])

    return curves_new, new_ref_curve

def symmetrize_matrix(A):
    """
    This function symmetrize a square matrix
    """

    return (A + A.T) / 2

def assign_label(D, list2, label_list, true_label_list):
    """
    This function assign label by symmetry breaking given the distance matrix D

    LIST1 is the list of train data
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

def normalise_matrix(A):
    """
    This function normalise a Matrix A
    """

    range_A = np.abs(np.max(A) - np.min(A))


    return (A-np.min(A))/range_A

#################################################        MAIN             ################################
pipeline_list = ["Original_prep", "Smoothed_prep", "Cutted_prep", "Cutted_smoothed_prep"]

dataset_main = "/home/listanvirg/Documents/Individual_identification/Kiwi_syllable_dataset"

# save labels
results_directory = "/home/listanvirg/Documents/Individual_identification/Test_results/Bootstrap_tests_6"

#read classes from train dataset
list_labels = ["D", "E", "J", "K", "L", "M", "O", "R", "Z"]

for pipeline in pipeline_list:
    print('Analysing pipeline ', pipeline)
    dataset_path = dataset_main + '/' + pipeline

    result_folder = results_directory + "/" + pipeline
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    freq_path = dataset_path[:-5]
    #initialise metrics vector
    SSD_v = np.zeros((5,1))
    PCA_v = np.zeros((5, 1))
    DTW_v = np.zeros((5, 1))
    GEO_v = np.zeros((5, 1))

    for k in range(5):
        print('Boot strap test ', k)
        datasets_list_path ="/home/listanvirg/Documents/Individual_identification/Dataset_lists" + "/Test_50_50_"+str(k)
        # print(datasets_list_path)
        train_dataset_path  = datasets_list_path


        #recover train and test list
        test_dataset_list_path = datasets_list_path + "/Test_file_list.csv"
        test_list = np.loadtxt(test_dataset_list_path, skiprows=1, delimiter=',', dtype=str)[:, 0]
        list_true_labels = np.loadtxt(test_dataset_list_path, skiprows=1, delimiter=',', dtype=str)[:, 1]

        list_train_syllables_path = []
        # list_train_path = []
        list_train_labels = []
        len_list1 = []
        list_train_files =[]
        len_freq_list1 = []
        list_train_freq_path = []
        for label in list_labels:
            train_dataset_list_path = train_dataset_path + "/Class_"+label+".csv"
            train_list = np.loadtxt(train_dataset_list_path, skiprows=1, delimiter=',', dtype=str)
            for file in train_list:
                file_path = dataset_path + '/' + file
                freq_file_path = freq_path + "/" + file
                list_train_files.append(file[:-7])
                list_train_syllables_path.append(file_path)
                list_train_freq_path.append(freq_file_path)
                list_train_labels.append(label)
                len_syl = len(np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1])
                len_freq_list1.append(len(np.loadtxt(freq_file_path, skiprows=1, delimiter=',')[:, 1]))
                len_list1.append(len_syl)
        n1 = len(list_train_syllables_path)
        # print(n1)

        #read label for test dataset
        list_syllables = []
        list_syllables_path = []
        list_freq_path = []
        list_assigned_labels_ssd = []
        list_assigned_labels_pca = []
        list_assigned_labels_dtw = []
        list_assigned_labels_crosscorr = []
        list_assigned_labels_geod = []
        len_list2= []
        len_freq_list2 = []

        for file in test_list:
            list_syllables.append(file[:-7])
            file_path = dataset_path+'/'+file
            freq_file_path = freq_path + '/' + file
            list_syllables_path.append(file_path)
            list_freq_path.append(freq_file_path)
            len_syl = len(np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1])
            len_list2.append(len_syl)
            len_freq_list2.append(len(np.loadtxt(freq_file_path, skiprows=1, delimiter=',')[:, 1]))

        n2 = len(list_syllables_path)
        # print(n2)
        len_max = max(np.max(len_list1),np.max(len_list2))
        len_min = min(np.min(len_list1),np.min(len_list2))
        len_max_freq = max(np.max(len_freq_list1), np.max(len_freq_list2))

        #number of files
        n = n1+n2

        # save train curves
        train_curves = np.zeros((n1, len_max, 2))
        train_freq_curves = np.zeros((n1, len_max_freq, 2))  # note I am saving the original curve (t,f)
        for i in range(n1):
            train_curves[i,:len_list1[i],:] = np.loadtxt(open(list_train_syllables_path[i], "rb"), delimiter=",", skiprows=1)
            train_freq_curves[i, :len_freq_list1[i], :] = np.loadtxt(open(list_train_freq_path[i], "rb"), delimiter=",",
                                                                     skiprows=1)

        del list_train_syllables_path, list_train_freq_path

        # print('Loaded train curves')
        # save test curves
        test_curves = np.zeros((n2, len_max, 2))
        test_freq_curves = np.zeros((n2, len_max_freq, 2))  # note I am saving the original curve (t,f)

        for i in range(n2):
            test_curves[i,:len_list2[i],:] = np.loadtxt(open(list_syllables_path[i], "rb"), delimiter=",", skiprows=1)
            test_freq_curves[i, :len_freq_list2[i], :] = np.loadtxt(open(list_freq_path[i], "rb"), delimiter=",", skiprows=1)

        del list_syllables_path, list_freq_path
        # print('Loaded test curves')
        num_label = len(list_labels)
        # accuracy
        accuracy_ssd = 0
        accuracy_pca = 0
        accuracy_dtw = 0
        accuracy_crosscorr = 0
        accuracy_geod = 0

        #pre-allocate distance_matrices
        ssd_matrix = np.zeros((n2,n1))
        geod_matrix = np.zeros((n2,n1))
        pca_matrix = np.zeros((n2,n1))
        dtw_matrix = np.zeros((n2,n1))
        df_matrix = np.zeros((n2, n1))  # matrix of difference of mean frequencies

        print('Evaluating metrics ')
        # print('Train curves ', np.shape(train_curves))
        # print('Train curves ', np.shape(test_curves))
        for i in range(n2):
            new_reference_curve = np.copy(test_curves[i, :len_list2[i], :])
            new_curves = train_curves
            new_reference_freq_curve = np.copy(test_freq_curves[i, :len_freq_list2[i], :])
            # # evaluate PCA distance vector
            pca_matrix[i,:] = distances.pca_distance_vector(new_curves[:, :, 1], new_reference_curve[:,1])

            for j in range(n1):
                # print(j)
                ssd_matrix[i, j] = distances.ssd(new_reference_curve[:,1], new_curves[j, :, 1])
                geod_matrix[i, j] = distances.Geodesic_curve_distance( new_reference_curve[:,0], new_reference_curve[:,1],
                                                                      new_curves[j, :, 0], new_curves[j, :, 1])
                dtw_matrix[i, j] = distances.dtw(new_reference_curve[:,1], new_curves[j, :, 1])
                df_matrix[i, j] = np.abs(np.median(new_reference_freq_curve[:, 1]) -
                                         np.median(train_freq_curves[j, :len_freq_list1[j], 1]))



        del new_curves, train_curves, test_curves, train_freq_curves, test_freq_curves

        # normalise matrices

        df_matrix = normalise_matrix(df_matrix)
        ssd_matrix = normalise_matrix(ssd_matrix) + df_matrix
        pca_matrix = normalise_matrix(pca_matrix) + df_matrix
        dtw_matrix = normalise_matrix(dtw_matrix) + df_matrix
        geod_matrix = normalise_matrix(geod_matrix) + df_matrix

        list_assigned_labels_ssd, best3_list_ssd, accuracy_ssd = assign_label(ssd_matrix, list_syllables, list_train_labels,
                                                                              list_true_labels)
        list_assigned_labels_pca, best3_list_pca, accuracy_pca = assign_label(pca_matrix, list_syllables, list_train_labels,
                                                                              list_true_labels)
        list_assigned_labels_dtw, best3_list_dtw, accuracy_dtw = assign_label(dtw_matrix, list_syllables, list_train_labels,
                                                                              list_true_labels)
        list_assigned_labels_geod, best3_list_geod, accuracy_geod = assign_label(geod_matrix, list_syllables, list_train_labels,
                                                                              list_true_labels)

        SSD_v[k] = accuracy_ssd
        PCA_v[k] = accuracy_pca
        DTW_v[k] = accuracy_dtw
        GEO_v[k] = accuracy_geod

        print('Saving metrics')
        csvfilename = result_folder + "/" + "Labels_comparison"+ str(k)+".csv"
        fieldnames = ['Syllable', 'True Label', 'SSD label', 'PCA label', 'DTW label', 'GEO label']
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(n2):
                dictionary = {'Syllable': list_syllables[i], 'True Label': list_true_labels[i],
                              'SSD label': list_assigned_labels_ssd[i], 'PCA label': list_assigned_labels_pca[i],
                              'DTW label': list_assigned_labels_dtw[i],
                              'GEO label': list_assigned_labels_geod[i]}

                writer.writerow(dictionary)
                del dictionary

        del list_true_labels, list_assigned_labels_ssd, list_assigned_labels_pca
        del list_assigned_labels_dtw, list_assigned_labels_crosscorr, list_assigned_labels_geod
        # save best3 match
        csvfilename = result_folder + "/" + "Best3_comparison"+ str(k)+".csv"
        fieldnames = ['Syllable', 'SSD', 'PCA', 'DTW', 'GEO']
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(n2):
                dictionary = {'Syllable': list_syllables[i], 'SSD': best3_list_ssd[i], 'PCA': best3_list_pca[i],
                              'DTW': best3_list_dtw[i], 'GEO': best3_list_geod[i]}

                writer.writerow(dictionary)
                del dictionary

            del list_syllables,  best3_list_ssd, best3_list_pca, best3_list_dtw, best3_list_geod

        # print accuracies in txt files
        file_path = result_folder + '/Accuracy_results'+ str(k)+".txt"
        file_txt = open(file_path, 'w')
        l0 = [" Accuracy results \n"]
        l1 = ["\n SSD Accuracy: " + str(accuracy_ssd)]
        l2 = ["\n PCA Accuracy: " + str(accuracy_pca)]
        l3 = ["\n DTW Accuracy: " + str(accuracy_dtw)]
        l4 = ["\n Crosscorr Accuracy: " + str(accuracy_crosscorr)]
        l5 = ["\n Geodesic Accuracy: " + str(accuracy_geod)]
        file_txt.writelines(np.concatenate((l0, l1, l2, l3, l4, l5)))
        file_txt.close()

        del accuracy_dtw, accuracy_pca, accuracy_ssd, accuracy_crosscorr, accuracy_geod

        #save matrices
        np.savetxt(result_folder +"/SSD"+ str(k)+".txt", ssd_matrix, fmt='%s')
        np.savetxt(result_folder+"/Geodesic"+ str(k)+".txt", geod_matrix, fmt='%s')
        np.savetxt(result_folder+"/PCA"+ str(k)+".txt", pca_matrix, fmt='%s')
        np.savetxt(result_folder +"/DTW"+ str(k)+".txt", dtw_matrix, fmt='%s')
        np.savetxt(result_folder + "/MedianFreq.txt", df_matrix, fmt='%s')

        del ssd_matrix, df_matrix, geod_matrix, pca_matrix, dtw_matrix

    # save_metrics

    csvfilename = result_folder + "/metric_bootstrap.csv"
    fieldnames = ['SSD', 'PCA', 'DTW', 'GEO']
    with open(csvfilename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(5):
            dictionary = {'SSD': SSD_v[i], 'PCA': PCA_v[i], 'DTW': DTW_v[i],
                          'GEO': GEO_v[i]}

            writer.writerow(dictionary)
            del dictionary

    #evaluate bootstrap confidence interval
    conf_interval_ssd = np.percentile(SSD_v, [2.5, 97.5])
    conf_interval_pca = np.percentile(PCA_v, [2.5, 97.5])
    conf_interval_dtw = np.percentile(DTW_v, [2.5, 97.5])
    conf_interval_geo = np.percentile(GEO_v, [2.5, 97.5])

    # print accuracies in txt files
    print('Final bootstrap accuracies')
    file_path = result_folder + '/Accuracy_results_bootstrap_'+pipeline+ ".txt"
    file_txt = open(file_path, 'w')
    l0 = [" Accuracy results \n"]
    l1 = ["\n SSD Confidence interval: " + str(conf_interval_ssd)]
    l2 = ["\n SSD Range: [" + str(np.amin(SSD_v))+", "+ str(np.amax(SSD_v))+"]"]
    l3 = ["\n PCA Confidence interval: " + str(conf_interval_pca)]
    l4 = ["\n PCA Range: [" + str(np.amin(PCA_v)) + ", " + str(np.amax(PCA_v)) + "]"]
    l5 = ["\n DTW Confidence interval: " + str(conf_interval_dtw)]
    l6 = ["\n DTW Range: [" + str(np.amin(DTW_v)) + ", " + str(np.amax(DTW_v)) + "]"]
    l7 = ["\n Geodesic Confidence interval: " + str(conf_interval_geo)]
    l8 = ["\n Geodesic Range: [" + str(np.amin(GEO_v)) + ", " + str(np.amax(GEO_v)) + "]"]
    file_txt.writelines(np.concatenate((l0, l1, l2, l3, l4, l5, l6, l7, l8)))
    file_txt.close()

    del SSD_v, GEO_v, PCA_v, DTW_v, conf_interval_geo, conf_interval_dtw, conf_interval_pca, conf_interval_ssd






