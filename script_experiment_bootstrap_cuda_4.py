"""
28/10/2022
Reviewed: 25/11/2022
Author: Virginia Listanti

This script manage the experiment to find the best metric to perform the classification task on
 kiwi syllables

 combination of four metrics

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

def assign_label4(list2, best3_list1, best3_list2, best3_list3, best3_list4, true_label_list):
    """
    This function assign label by majority from best3_list1 and best3_list2

    LIST2 is the list of test data
    BEST3_LIST1 best 3 list from distance 1
    BEST3_LIST2 best 3 list from distance 2
    BEST3_LIST3 best 3 list from distance 3
    BEST3_LIST4 best 3 list from distance 4
    TRUE_LABEL_LIST list of true label fro LIST2
    """
    # N1 = len(list1)
    N2 = len(list2)
    accuracy = 0
    alg_labels = []

    for k in range(N2):
        freq_count = CountFrequency4(best3_list1[k], best3_list2[k], best3_list3[k], best3_list4[k])
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

def CountFrequency3(my_list1, my_list2, my_list3, my_list4):

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

    for item in my_list4:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    return freq

def CountFrequency4(my_list1, my_list2, my_list3):

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

#################################################        MAIN             ################################
pipeline_list = ["Original_prep", "Smoothed_prep", "Cutted_prep", "Cutted_smoothed_prep"]

dataset_main = "/home/listanvirg/Documents/Individual_identification/Kiwi_syllable_dataset"

# save labels
results_directory = "/home/listanvirg/Documents/Individual_identification/Test_results/Bootstrap_tests_3"

#read classes from train dataset
list_labels = ["D", "E", "J", "K", "L", "M", "O", "R", "Z"]

for pipeline in pipeline_list:
    print('Analysing pipeline ', pipeline)
    dataset_path = dataset_main + '/' + pipeline

    result_folder = results_directory + "/" + pipeline
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    #initialise metrics vector
    SSD_DTW_PCA_v = np.zeros((5,1))
    SSD_DTW_GEO_v = np.zeros((5, 1))
    SSD_PCA_GEO_v = np.zeros((5, 1))
    DTW_PCA_GEO_v = np.zeros((5, 1))


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
        for label in list_labels:
            train_dataset_list_path = train_dataset_path + "/Class_"+label+".csv"
            train_list = np.loadtxt(train_dataset_list_path, skiprows=1, delimiter=',', dtype=str)
            for file in train_list:
                file_path = dataset_path + '/' + file
                list_train_files.append(file[:-7])
                list_train_syllables_path.append(file_path)
                list_train_labels.append(label)
                len_syl = len(np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1])
                len_list1.append(len_syl)
        n1 = len(list_train_syllables_path)
        # print(n1)

        #read label for test dataset
        list_syllables = []
        list_syllables_path = []
        list_assigned_labels_ssd = []
        list_assigned_labels_pca = []
        list_assigned_labels_dtw = []
        list_assigned_labels_crosscorr = []
        list_assigned_labels_geod = []
        len_list2= []


        for file in test_list:
            list_syllables.append(file[:-7])
            file_path = dataset_path+'/'+file
            list_syllables_path.append(file_path)
            len_syl = len(np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1])
            len_list2.append(len_syl)
                    #how to do comparison?.

        n2 = len(list_syllables_path)
        # print(n2)
        len_max = max(np.max(len_list1),np.max(len_list2))
        len_min = min(np.min(len_list1),np.min(len_list2))

        #number of files
        n = n1+n2

        # save train curves
        train_curves = np.zeros((n1, len_max, 2))
        for i in range(n1):
            train_curves[i,:len_list1[i],:] = np.loadtxt(open(list_train_syllables_path[i], "rb"), delimiter=",", skiprows=1)
            #subtract mean
            train_curves[i, :len_list1[i], :] -= np.mean(train_curves[i, :len_list1[i], :] )

        del list_train_syllables_path

        # print('Loaded train curves')
        # save test curves
        test_curves = np.zeros((n2, len_max, 2))
        for i in range(n2):
            test_curves[i,:len_list2[i],:] = np.loadtxt(open(list_syllables_path[i], "rb"), delimiter=",", skiprows=1)
            #subtract mean
            test_curves[i, :len_list2[i], :] -= np.mean(test_curves[i, :len_list2[i], :] )

        del list_syllables_path
        # print('Loaded test curves')
        num_label = len(list_labels)
        # accuracy
        accuracy_ssd_dtw_pca = 0
        accuracy_ssd_dtw_geo = 0
        accuracy_ssd_pca_geo = 0
        accuracy_dtw_pca_geo = 0

        #pre-allocate distance_matrices
        ssd_matrix = np.zeros((n2,n1))
        crosscorr_matrix = np.zeros((n2,n1))
        geod_matrix = np.zeros((n2,n1))
        pca_matrix = np.zeros((n2,n1))
        dtw_matrix = np.zeros((n2,n1))

        print('Evaluating metrics ')
        # print('Train curves ', np.shape(train_curves))
        # print('Train curves ', np.shape(test_curves))
        for i in range(n2):
            new_reference_curve = np.copy(test_curves[i, :len_list2[i], :])
            new_curves = train_curves
            # # evaluate PCA distance vector
            pca_matrix[i,:] = distances.pca_distance_vector(new_curves[:, :, 1], new_reference_curve[:,1])

            for j in range(n1):
                # print(j)
                ssd_matrix[i, j] = distances.ssd(new_reference_curve[:,1], new_curves[j, :, 1])
                geod_matrix[i, j] = distances.Geodesic_curve_distance( new_reference_curve[:,0], new_reference_curve[:,1],
                                                                      new_curves[j, :, 0], new_curves[j, :, 1])
                dtw_matrix[i, j] = distances.dtw(new_reference_curve[:,1], new_curves[j, :, 1])



        del new_curves, train_curves, test_curves

        best3_list_ssd = find_best(ssd_matrix, list_syllables, list_train_labels)
        best3_list_pca = find_best(pca_matrix, list_syllables, list_train_labels)
        best3_list_dtw = find_best(dtw_matrix, list_syllables, list_train_labels)
        best3_list_geo = find_best(geod_matrix, list_syllables, list_train_labels)

        # combine 3

        list_assigned_labels_ssd_dtw_pca, accuracy_ssd_dtw_pca = assign_label3(list_syllables,
                                                                               best3_list_ssd,
                                                                               best3_list_dtw,
                                                                               best3_list_pca,
                                                                               list_true_labels)
        list_assigned_labels_ssd_dtw_geo, accuracy_ssd_dtw_geo = assign_label3(list_syllables,
                                                                               best3_list_ssd,
                                                                               best3_list_dtw,
                                                                               best3_list_geo,
                                                                               list_true_labels)
        list_assigned_labels_ssd_pca_geo, accuracy_ssd_pca_geo = assign_label3(list_syllables,
                                                                               best3_list_ssd,
                                                                               best3_list_pca,
                                                                               best3_list_geo,
                                                                               list_true_labels)
        list_assigned_labels_dtw_pca_geo, accuracy_dtw_pca_geo = assign_label3(list_syllables,
                                                                               best3_list_dtw,
                                                                               best3_list_pca,
                                                                               best3_list_geo,
                                                                               list_true_labels)

        SSD_DTW_PCA_v[k] = accuracy_ssd_dtw_pca
        SSD_DTW_GEO_v[k] = accuracy_ssd_dtw_geo
        SSD_PCA_GEO_v[k] = accuracy_ssd_pca_geo
        DTW_PCA_GEO_v[k] = accuracy_dtw_pca_geo


        print('Saving metrics')
        csvfilename = result_folder + "/" + "Labels_comparison"+ str(k)+".csv"
        fieldnames = ['Syllable', 'True Label', 'SSD + DTW + PCA label', 'SSD + DTW + GEO label',
                      'SSD + PCA +GEO label',
                      'DTW + PCA + GEO label']
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(n2):
                dictionary = {'Syllable': list_syllables[i], 'True Label': list_true_labels[i],
                              'SSD + DTW + PCA label': list_assigned_labels_ssd_dtw_pca[i], 'SSD + DTW + GEO label':
                                  list_assigned_labels_ssd_dtw_geo[i],
                              'SSD + PCA +GEO label': list_assigned_labels_ssd_pca_geo[i],
                              'DTW + PCA + GEO label': list_assigned_labels_dtw_pca_geo[i]}

                writer.writerow(dictionary)
                del dictionary

        del list_true_labels, list_assigned_labels_ssd_dtw_pca, list_assigned_labels_ssd_dtw_geo,
        del list_assigned_labels_ssd_pca_geo, list_assigned_labels_dtw_pca_geo
        # save best3 match
        csvfilename = result_folder + "/" + "Best3_comparison"+ str(k)+".csv"
        fieldnames = ['Syllable', 'SSD', 'PCA', 'DTW', 'GEO']
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(n2):
                dictionary = {'Syllable': list_syllables[i], 'SSD': best3_list_ssd[i], 'PCA': best3_list_pca[i],
                              'DTW': best3_list_dtw[i], 'GEO': best3_list_geo[i]}

                writer.writerow(dictionary)
                del dictionary

            del list_syllables,  best3_list_ssd, best3_list_pca, best3_list_dtw, best3_list_geo

        # print accuracies in txt files
        file_path = result_folder + '/Accuracy_results'+ str(k)+".txt"
        file_txt = open(file_path, 'w')
        l0 = [" Accuracy results \n"]
        l1 = ["\n SSD + DTW + PCA Accuracy: " + str(accuracy_ssd_dtw_pca)]
        l2 = ["\n SSD + DTW + GEO Accuracy: " + str(accuracy_ssd_dtw_geo)]
        l3 = ["\n SSD + PCA + GEO Accuracy: " + str(accuracy_ssd_pca_geo)]
        l4 = ["\n DTW + PCA + GEO Accuracy: " + str(accuracy_dtw_pca_geo)]
        file_txt.writelines(np.concatenate((l0, l1, l2, l3, l4)))
        file_txt.close()

        del accuracy_ssd_dtw_pca, accuracy_ssd_dtw_geo, accuracy_ssd_pca_geo,accuracy_dtw_pca_geo

        #save matrices
        np.savetxt(result_folder +"/SSD"+ str(k)+".txt", ssd_matrix, fmt='%s')
        np.savetxt(result_folder +"/cross-correlation"+ str(k)+".txt", crosscorr_matrix, fmt='%s')
        np.savetxt(result_folder+"/Geodesic"+ str(k)+".txt", geod_matrix, fmt='%s')
        np.savetxt(result_folder+"/PCA"+ str(k)+".txt", pca_matrix, fmt='%s')
        np.savetxt(result_folder +"/DTW"+ str(k)+".txt", dtw_matrix, fmt='%s')

        del ssd_matrix, crosscorr_matrix, geod_matrix, pca_matrix, dtw_matrix

    # save_metrics
    csvfilename = result_folder + "/metric_bootstrap.csv"
    fieldnames = ['SSD+DTW+PCA', 'SSD+DTW+GEO', 'SSD+PCA+GEO', 'DTW+PCA+GEO']
    with open(csvfilename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        SSD_DTW_PCA_v[k] = accuracy_ssd_dtw_pca
        SSD_DTW_GEO_v[k] = accuracy_ssd_dtw_geo
        SSD_PCA_GEO_v[k] = accuracy_ssd_pca_geo
        DTW_PCA_GEO_v[k] = accuracy_dtw_pca_geo

        for i in range(5):
            dictionary = {'SSD+DTW+PCA': SSD_DTW_PCA_v[i], 'SSD+DTW+GEO': SSD_DTW_GEO_v[i], 'SSD+PCA+GEO':
                SSD_PCA_GEO_v[i], 'DTW+PCA+GEO':DTW_PCA_GEO_v[i]}

            writer.writerow(dictionary)
            del dictionary

    #evaluate bootstrap confidence interval
    conf_interval_ssd_dtw_pca = np.percentile(SSD_DTW_PCA_v, [2.5, 97.5])
    conf_interval_ssd_dtw_geo = np.percentile(SSD_DTW_GEO_v, [2.5, 97.5])
    conf_interval_ssd_pca_geo = np.percentile(SSD_PCA_GEO_v, [2.5, 97.5])
    conf_interval_dtw_pca_geo = np.percentile(DTW_PCA_GEO_v, [2.5, 97.5])


    # print accuracies in txt files
    print('Final bootstrap accuracies')
    file_path = result_folder + '/Accuracy_results_bootstrap_'+pipeline+ ".txt"
    file_txt = open(file_path, 'w')
    l0 = [" Accuracy results \n"]
    l1 = ["\n SSD + DTW + PCA Confidence interval: " + str(conf_interval_ssd_dtw_pca)]
    l2 = ["\n SSD + DTW + PCA Range: [" + str(np.amin(SSD_DTW_PCA_v))+", "+ str(np.amax(SSD_DTW_PCA_v))+"]"]
    l3 = ["\n SSD + DTW + GEO Confidence interval: " + str(conf_interval_ssd_dtw_geo)]
    l4 = ["\n SSD + DTW + GEO Range: [" + str(np.amin(SSD_DTW_GEO_v)) + ", " + str(np.amax(SSD_DTW_GEO_v)) + "]"]
    l5 = ["\n SSD + PCA + GEO Confidence interval: " + str(conf_interval_ssd_pca_geo)]
    l6 = ["\n SSD + PCA + GEO Range: [" + str(np.amin(SSD_PCA_GEO_v)) + ", " + str(np.amax(SSD_PCA_GEO_v)) + "]"]
    l7 = ["\n DTW + PCA + GEO Confidence interval: " + str(conf_interval_dtw_pca_geo)]
    l8 = ["\n DTW + PCA + GEO Range: [" + str(np.amin(DTW_PCA_GEO_v)) + ", " + str(np.amax(DTW_PCA_GEO_v)) + "]"]

    file_txt.writelines(np.concatenate((l0, l1, l2, l3, l4, l5, l6, l7, l8)))
    file_txt.close()

    del SSD_DTW_PCA_v, SSD_DTW_GEO_v, SSD_PCA_GEO_v, DTW_PCA_GEO_v
    del conf_interval_ssd_dtw_pca, conf_interval_ssd_dtw_geo, conf_interval_ssd_pca_geo, conf_interval_dtw_pca_geo







