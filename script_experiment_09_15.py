"""
15/09/2022
Author: Virginia Listanti

This script manage the experiment to find the best metric to perform the classification task of kiwi syllables
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import distances


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

    for i in range(N):
        # dynamic time warping
        target_curve = moving_average(curves_old[i, :length_list[i], 1], 21)
        m = distances.dtw(target_curve, ref_curve[:, 1], wantDistMatrix=True)
        x, y = distances.dtw_path(m)
        aligned_times = np.linspace(0, 1, len(x))
        aligned_curve = target_curve[x]
        # # subratct average
        # aligned_curve -= np.mean(aligned_curve)
        # resample
        curves_new[i, :, 0] = new_times
        curves_new[i, :, 1] = np.interp(new_times, aligned_times, aligned_curve)

    new_ref_curve = np.interp(new_times, ref_curve[:, 0], ref_curve[:, 1])

    return curves_new, new_ref_curve

def symmetrize_matrix(A):
    """
    This function symmetrize a square matrix
    """

    return (A + A.T) / 2

train_dataset = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID" \
                "\\exemplars\\Smaller_Dataset\\Metrics_experiment\\Train"

test_dataset = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID" \
               "\\exemplars\\Smaller_Dataset\\Metrics_experiment\\Test"

#read classes from train dataset
list_labels = []
len_list1 = []
list_train_syllables_path = []
for folder in os.listdir(train_dataset):
    list_labels.append(folder)
    for file in os.listdir(train_dataset + '\\' + folder):
        if file.endswith("_IF.csv"):
            file_path = train_dataset + '\\' + folder + + '\\' +file
            list_train_syllables_path.append(file_path)
            len_syl = len(np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1])
            len_list1.append(len_syl)
n1 = len(list_train_syllables_path)

#read label for test dataset
list_syllables = []
list_syllables_path = []
list_true_labels = []
list_assigned_labels = []
len_list2= []

for label in list_labels:
    for file in os.listdir(test_dataset+'\\'+label):
        if file.endswith('_IF.csv'):
            list_syllables.append(file[:-7])
            list_true_labels.append(label)
            file_path = test_dataset+'\\'+label+'\\'+file
            list_syllables_path.append(file_path)
            len_syl = len(np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1])
            len_list2.append(len_syl)
            #how to do comparison?.

n2 = len(list_syllables_path)
len_max = np.max(np.max(len_list1),np.max(len_list2))
len_min = np.min(np.min(len_list1),np.min(len_list2))

#number of files
n = n1+n2

# save train curves
train_curves = np.zeros((n1, len_max, 2))
for i in range(n1):
    train_curves[i,:len_list1[i],:] = np.loadtxt(open(list_train_syllables_path[i], "rb"), delimiter=",", skiprows=1)
    #subtract mean
    train_curves[i, :len_list1[i], :] -= np.mean(train_curves[i, :len_list1[i], :] )

# save test curves
test_curves = np.zeros((n2, len_max, 2))
for i in range(n2):
    test_curves[i,:len_list2[i],:] = np.loadtxt(open(list_syllables_path[i], "rb"), delimiter=",", skiprows=1)
    #subtract mean
    test_curves[i, :len_list2[i], :] -= np.mean(test_curves[i, :len_list2[i], :] )

#pre-allocate distance_vectors
num_label = len(list_labels)
ssd_vector = np.zeros((num_label))
crosscorr_vector = np.zeros((num_label))
geod_vector = np.zeros((num_label))
pca_vector = np.zeros((num_label))
dtw_vector = np.zeros((num_label))


for i in range(n):
    # prepare curves row by row
    reference_curve = np.copy(test_curves[i,:,:])
    reference_curve[:,1] = moving_average(reference_curve[:,1], 21)
    new_curves, new_reference_curve = prepare_curves(train_curves, reference_curve, len_list1, len_min)
    # # evaluate PCA matrix
    # pca_matrix[i, :] = distances.pca_distance(new_curves[:, :, 1])[i, :]
    for j in range(0, n1, 2):
        ssd_vector[int(j/2)] = (distances.ssd(new_reference_curve, new_curves[j, :, 1]) +
                                distances.ssd(new_reference_curve, new_curves[j+1, :, 1]))/2
        crosscorr_vector[int(j/2)] = (distances.cross_corr(new_reference_curve, new_curves[j, :, 1]) +
                                      distances.cross_corr(new_reference_curve, new_curves[j, :, 1]))/2
        geod_vector[int(j/2)] = (distances.Geodesic_curve_distance(new_curves[j, :, 0], new_reference_curve,
                                                              new_curves[j, :, 0], new_curves[j, :, 1]) +
                                 distances.Geodesic_curve_distance(new_curves[j+1, :, 0], new_reference_curve,
                                                                   new_curves[j+1, :, 0], new_curves[j+1, :, 1]))/2
        dtw_vector[int(j/2)] = (distances.dtw(new_reference_curve, new_curves[j, :, 1]) +
                                distances.dtw(new_reference_curve, new_curves[j+1, :, 1]))/2

    list_assigned_labels.append(list_labels[np.argmin(ssd_vector)])
# analyse syllables

# accuracy