"""
07/11/2022
Author: Virginia Listanti

This script manage the experiment to find the best pipeline and metric to perform the classification task on
 kiwi syllables

The experiment run on 10 classes

We combine the metrics with the difference of the frequencies medians

using other dataset Test_50_50_3

Part 2: we combine the results from SSD, DTW, PCA, Geo in trios
 - SSD + DTW +PCA
 - SSD + DTW + Geo
 - SSD + PCA + Geo
 - DTW + PCA + Geo
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
        # # subratct average
        # aligned_curve -= np.mean(aligned_curve)
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

def assign_label(list2, best3_list1, best3_list2, best3_list3, true_label_list):
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
        freq_count = CountFrequency(best3_list1[k], best3_list2[k], best3_list3[k])
        label = max(freq_count, key=freq_count.get)
        alg_labels.append(label)
        if label == true_label_list[k]:
            accuracy += 1
    accuracy /= N2

    return alg_labels, accuracy

def CountFrequency(my_list1, my_list2, my_list3):

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
# dataset_path = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#                "Smaller_Dataset2\\Original_prep"
# dataset_path = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#                "Smaller_Dataset2\\Smoothed_prep"
dataset_path = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
               "Smaller_Dataset2\\Cutted_prep"
# dataset_path = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#                "Smaller_Dataset2\\Cutted_smoothed_prep"

train_dataset_path  = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
                      "Smaller_Dataset2\\Test_50_50_3"
Test_id = 3

#read classes from train dataset
list_labels = ["D", "E", "J", "K", "L", "M", "O", "R", "Z"]

#recover train and test list
test_dataset_list_path = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\" \
                         "exemplars\\Smaller_Dataset2\\Test_50_50_3\\Test_file_list.csv"
test_list = np.loadtxt(test_dataset_list_path, skiprows=1, delimiter=',', dtype=str)[:, 0]
list_true_labels = np.loadtxt(test_dataset_list_path, skiprows=1, delimiter=',', dtype=str)[:, 1]

list_train_syllables_path = []
# list_train_path = []
list_train_labels = []
len_list1 = []
list_train_files =[]
for label in list_labels:
    train_dataset_list_path = train_dataset_path + "\\Class_"+label+".csv"
    train_list = np.loadtxt(train_dataset_list_path, skiprows=1, delimiter=',', dtype=str)
    for file in train_list:
        file_path = dataset_path + '\\' + file
        list_train_files.append(file[:-7])
        list_train_syllables_path.append(file_path)
        list_train_labels.append(label)
        len_syl = len(np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1])
        len_list1.append(len_syl)
n1 = len(list_train_syllables_path)

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
    file_path = dataset_path+'\\'+file
    list_syllables_path.append(file_path)
    len_syl = len(np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1])
    len_list2.append(len_syl)
            #how to do comparison?.

n2 = len(list_syllables_path)
len_max = max(np.max(len_list1),np.max(len_list2))
len_min = min(np.min(len_list1),np.min(len_list2))

#number of files
n = n1+n2

# save train curves
train_curves = np.zeros((n1, len_max, 2))
for i in range(n1):
    train_curves[i,:len_list1[i],:] = np.loadtxt(open(list_train_syllables_path[i], "rb"), delimiter=",", skiprows=1)

# save test curves
test_curves = np.zeros((n2, len_max, 2))
for i in range(n2):
    test_curves[i,:len_list2[i],:] = np.loadtxt(open(list_syllables_path[i], "rb"), delimiter=",", skiprows=1)

num_label = len(list_labels)
# accuracy
accuracy_ssd_dtw_pca = 0
accuracy_ssd_dtw_geo = 0
accuracy_ssd_pca_geo = 0
accuracy_dtw_pca_geo = 0

#pre-allocate distance_matrices
ssd_matrix = np.zeros((n2,n1))
geod_matrix = np.zeros((n2,n1))
pca_matrix = np.zeros((n2,n1))
dtw_matrix = np.zeros((n2,n1))


for i in range(n2):
    # prepare curves row by row
    new_reference_curve = np.copy(test_curves[i,:len_list2[i],:])
    new_curves = train_curves

    # # evaluate PCA distance vector
    pca_matrix[i,:] = distances.pca_distance_vector(new_curves[:, :, 1], new_reference_curve[:,1])

    for j in range(n1):
        ssd_matrix[i, j] = distances.ssd(new_reference_curve[:,1], new_curves[j, :, 1])
        # crosscorr_matrix[i, j] = distances.cross_corr(new_reference_curve[:,1], new_curves[j, :, 1])
        geod_matrix[i, j] = distances.Geodesic_curve_distance( new_reference_curve[:,0], new_reference_curve[:,1],
                                                              new_curves[j, :, 0], new_curves[j, :, 1])
        dtw_matrix[i, j] = distances.dtw(new_reference_curve[:,1], new_curves[j, :, 1])


#symmetrize matrices

# crosscorr_matrix = symmetrize_matrix(crosscorr_matrix)
# ssd_matrix = symmetrize_matrix(ssd_matrix)
# pca_matrix = symmetrize_matrix(pca_matrix)
# dtw_matrix = symmetrize_matrix(dtw_matrix)
# geod_matrix = symmetrize_matrix(geod_matrix)

# prepare corosscorellation matrix for clustering
# NOTE: we don't need this but I am doing for having it for the next step
# crosscorr_matrix = np.max(crosscorr_matrix) - crosscorr_matrix

# find best 3 match for each metric
best3_list_ssd = find_best(ssd_matrix, list_syllables, list_train_labels)
best3_list_pca = find_best(pca_matrix, list_syllables, list_train_labels)
best3_list_dtw = find_best(dtw_matrix, list_syllables, list_train_labels)
best3_list_geo = find_best(geod_matrix, list_syllables, list_train_labels)

list_assigned_labels_ssd_dtw_pca, accuracy_ssd_dtw_pca = assign_label(list_syllables, best3_list_ssd, best3_list_dtw,
                                                                      best3_list_pca, list_true_labels)
list_assigned_labels_ssd_dtw_geo, accuracy_ssd_dtw_geo = assign_label(list_syllables, best3_list_ssd, best3_list_dtw,
                                                                      best3_list_geo, list_true_labels)
list_assigned_labels_ssd_pca_geo, accuracy_ssd_pca_geo = assign_label(list_syllables, best3_list_ssd, best3_list_pca,
                                                                      best3_list_geo, list_true_labels)
list_assigned_labels_dtw_pca_geo, accuracy_dtw_pca_geo = assign_label(list_syllables, best3_list_dtw, best3_list_pca,
                                                                      best3_list_geo, list_true_labels)



# save labels
results_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
                    "Smaller_Dataset2\\Tests_metrics\\Test_metrics_16"
result_folder = results_directory + "\\"+ "Test_"+ str(Test_id)
if not "Test_" + str(Test_id) in os.listdir(results_directory):
    os.mkdir(result_folder)

csvfilename = result_folder + "\\" + "Labels_comparison.csv"
fieldnames = ['Syllable', 'True Label', 'SSD + DTW + PCA label', 'SSD + DTW + GEO label', 'SSD + PCA +GEO label',
              'DTW + PCA + GEO label']
with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(n2):
        dictionary = {'Syllable': list_syllables[i], 'True Label': list_true_labels[i],
                      'SSD + DTW + PCA label': list_assigned_labels_ssd_dtw_pca[i], 'SSD + DTW + GEO label':
                      list_assigned_labels_ssd_dtw_geo[i], 'SSD + PCA +GEO label': list_assigned_labels_ssd_pca_geo[i],
                      'DTW + PCA + GEO label': list_assigned_labels_dtw_pca_geo[i]}

        writer.writerow(dictionary)
        del dictionary

# save best3 match
csvfilename = result_folder + "\\" + "Best3_comparison.csv"
fieldnames = ['Syllable', 'SSD', 'PCA', 'DTW', 'GEO']
with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(n2):
        dictionary = {'Syllable': list_syllables[i], 'SSD': best3_list_ssd[i], 'PCA': best3_list_pca[i],
                      'DTW': best3_list_dtw[i], 'GEO': best3_list_geo[i]}

        writer.writerow(dictionary)
        del dictionary

# print accuracies in txt files
file_path = result_folder + '\\Accuracy_results.txt'
file_txt = open(file_path, 'w')
l0 = [" Accuracy results \n"]
l1 = ["\n SSD + DTW + PCA Accuracy: " + str(accuracy_ssd_dtw_pca)]
l2 = ["\n SSD + DTW + GEO Accuracy: " + str(accuracy_ssd_dtw_geo)]
l3 = ["\n SSD + PCA + GEO Accuracy: " + str(accuracy_ssd_pca_geo)]
l4 = ["\n DTW + PCA + GEO Accuracy: " + str(accuracy_dtw_pca_geo)]
file_txt.writelines(np.concatenate((l0, l1, l2, l3, l4)))
file_txt.close()

#save matrices
np.savetxt(result_folder +"\\SSD.txt", ssd_matrix, fmt='%s')
np.savetxt(result_folder+"\\Geodesic.txt", geod_matrix, fmt='%s')
np.savetxt(result_folder+"\\PCA.txt", pca_matrix, fmt='%s')
np.savetxt(result_folder +"\\DTW.txt", dtw_matrix, fmt='%s')

# Plot the matrices

list_labels2 = list_train_files
list_labels1 = list_syllables

fig, ax = plt.subplots(2,2, figsize=(80, 80))

ax[0,0].imshow(ssd_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,0].set_xticks(np.arange(n1), labels=list_labels2, fontsize=40)
ax[0,0].set_yticks(np.arange(n2), labels=list_labels1, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 0].set_title("SSD distance", fontsize=80)

ax[0, 1].imshow(pca_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0, 1].set_xticks(np.arange(n1), labels=list_labels2, fontsize=40)
ax[0, 1].set_yticks(np.arange(n2), labels=list_labels1, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 1].set_title("PCA distance", fontsize=80)

ax[1,0].imshow(geod_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[1,0].set_xticks(np.arange(n1), labels=list_labels2, fontsize=40)
ax[1,0].set_yticks(np.arange(n2), labels=list_labels1, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[1, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[1, 0].set_title("Geodesic distance", fontsize=80)

ax[1,1].imshow(dtw_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[1,1].set_xticks(np.arange(n1), labels=list_labels2, fontsize=40)
ax[1,1].set_yticks(np.arange(n2), labels=list_labels1, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[1, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[1, 1].set_title("DTW distance", fontsize=80)

# fig.suptitle('Models Test 1', fontsize=120)
fig.suptitle('Distance matrices', fontsize=120)

fig_name =  result_folder + "\\Distance_matrices.jpg"
plt.savefig(fig_name)

