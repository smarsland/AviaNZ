"""
15/09/2022
Author: Virginia Listanti

This script manage the experiment to find the best metric to perform the classification task of kiwi syllables
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


def prepare_curves(curves_old, ref_curve, length_list, M, s_points):
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
        target_curve = moving_average(curves_old[i, :length_list[i], 1], s_points)
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

train_dataset = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID" \
                "\\exemplars\\Smaller_Dataset\\Metrics_experiment\\Train"

test_dataset = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID" \
               "\\exemplars\\Smaller_Dataset\\Metrics_experiment\\Test"

smoothing_points = 5
#read classes from train dataset
list_labels = []
len_list1 = []
list_train_syllables_path = []
list_train_files = []
list_train_labels = []
for folder in os.listdir(train_dataset):
    list_labels.append(folder)
    for file in os.listdir(train_dataset + '\\' + folder):
        if file.endswith("_IF.csv"):
            file_path = train_dataset + '\\' + folder + '\\' + file
            list_train_files.append(file[:-7])
            list_train_syllables_path.append(file_path)
            list_train_labels.append(folder)
            len_syl = len(np.loadtxt(file_path, skiprows=1, delimiter=',')[:, 1])
            len_list1.append(len_syl)
n1 = len(list_train_syllables_path)

#read label for test dataset
list_syllables = []
list_syllables_path = []
list_true_labels = []
list_assigned_labels_ssd = []
list_assigned_labels_pca = []
list_assigned_labels_dtw = []
list_assigned_labels_crosscorr = []
list_assigned_labels_geod = []
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

# save test curves
test_curves = np.zeros((n2, len_max, 2))
for i in range(n2):
    test_curves[i,:len_list2[i],:] = np.loadtxt(open(list_syllables_path[i], "rb"), delimiter=",", skiprows=1)
    #subtract mean
    test_curves[i, :len_list2[i], :] -= np.mean(test_curves[i, :len_list2[i], :] )

num_label = len(list_labels)
# accuracy
accuracy_ssd = 0
accuracy_pca = 0
accuracy_dtw = 0
accuracy_crosscorr = 0
accuracy_geod = 0

#pre-allocate distance_matrices
ssd_matrix = np.zeros((n2,n1))
crosscorr_matrix = np.zeros((n2,n1))
geod_matrix = np.zeros((n2,n1))
pca_matrix = np.zeros((n2,n1))
dtw_matrix = np.zeros((n2,n1))


for i in range(n2):
    # prepare curves row by row
    reference_curve = np.copy(test_curves[i,:len_list2[i],:])
    reference_curve[:,1] = moving_average(reference_curve[:,1], smoothing_points)
    new_curves, new_reference_curve = prepare_curves(train_curves, reference_curve, len_list1, len_min, smoothing_points)
    # # evaluate PCA distance vector
    pca_matrix[i,:] = distances.pca_distance_vector(new_curves[:, :, 1], new_reference_curve[:,1])

    for j in range(n1):
        ssd_matrix[i, j] = distances.ssd(new_reference_curve[:,1], new_curves[j, :, 1])
        crosscorr_matrix[i, j] = distances.cross_corr(new_reference_curve[:,1], new_curves[j, :, 1])
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
crosscorr_matrix = np.max(crosscorr_matrix) - crosscorr_matrix

list_assigned_labels_ssd, best3_list_ssd, accuracy_ssd = assign_label(ssd_matrix, list_syllables, list_train_labels,
                                                                      list_true_labels)
list_assigned_labels_pca, best3_list_pca, accuracy_pca = assign_label(pca_matrix, list_syllables, list_train_labels,
                                                                      list_true_labels)
list_assigned_labels_dtw, best3_list_dtw, accuracy_dtw = assign_label(dtw_matrix, list_syllables, list_train_labels,
                                                                      list_true_labels)
list_assigned_labels_geod, best3_list_geod, accuracy_geod = assign_label(geod_matrix, list_syllables, list_train_labels,
                                                                      list_true_labels)
list_assigned_labels_crosscorr, best3_list_crosscorr, accuracy_crosscorr = assign_label(crosscorr_matrix, list_syllables,
                                                                                        list_train_labels,
                                                                                        list_true_labels)

# save labels
results_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
                   "exemplars\\Smaller_Dataset\\Metrics_experiment\\Results_test2"
csvfilename = results_directory + "\\" + "Labels_comparison.csv"
fieldnames = ['Syllable', 'True Label', 'SSD label', 'PCA label', 'DTW label', 'Crosscorr label', 'GEO label']
with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(n2):
        dictionary = {'Syllable': list_syllables[i], 'True Label': list_true_labels[i],
                      'SSD label': list_assigned_labels_ssd[i], 'PCA label': list_assigned_labels_pca[i],
                      'DTW label': list_assigned_labels_dtw[i], 'Crosscorr label': list_assigned_labels_crosscorr[i],
                      'GEO label': list_assigned_labels_geod[i]}

        writer.writerow(dictionary)
        del dictionary

# save best3 match
csvfilename = results_directory + "\\" + "Best3_comparison.csv"
fieldnames = ['Syllable', 'SSD', 'PCA', 'DTW', 'Crosscorr', 'GEO']
with open(csvfilename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(n2):
        dictionary = {'Syllable': list_syllables[i], 'SSD': best3_list_ssd[i], 'PCA': best3_list_pca[i],
                      'DTW': best3_list_dtw[i], 'Crosscorr': best3_list_crosscorr[i], 'GEO': best3_list_geod[i]}

        writer.writerow(dictionary)
        del dictionary

# print accuracies in txt files
file_path = results_directory + '\\Accuracy_results.txt'
file_txt = open(file_path, 'w')
l0 = [" Accuracy results \n"]
l1 = ["\n SSD Accuracy: " + str(accuracy_ssd)]
l2 = ["\n PCA Accuracy: " + str(accuracy_pca)]
l3 = ["\n DTW Accuracy: " + str(accuracy_dtw)]
l4 = ["\n Crosscorr Accuracy: " + str(accuracy_crosscorr)]
l5 = ["\n Geodesic Accuracy: " + str(accuracy_geod)]
file_txt.writelines(np.concatenate((l0, l1, l2, l3, l4, l5)))
file_txt.close()

#save matrices
np.savetxt(results_directory +"\\SSD.txt", ssd_matrix, fmt='%s')
np.savetxt(results_directory +"\\cross-correlation.txt", crosscorr_matrix, fmt='%s')
np.savetxt(results_directory +"\\Geodesic.txt", geod_matrix, fmt='%s')
# np.savetxt(save_directory+"\\PCA.txt", pca_matrix, fmt='%s')
np.savetxt(results_directory +"\\DTW.txt", dtw_matrix, fmt='%s')

# Plot the matrices

list_labels2 = list_train_files
list_labels1 = list_syllables

fig, ax = plt.subplots(3,2, figsize=(80, 80))

ax[0,0].imshow(ssd_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,0].set_xticks(np.arange(n1), labels=list_labels2, fontsize=40)
ax[0,0].set_yticks(np.arange(n2), labels=list_labels1, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 0].set_title("SSD distance", fontsize=80)

ax[0,1].imshow(crosscorr_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,1].set_xticks(np.arange(n1), labels=list_labels2, fontsize=40)
ax[0,1].set_yticks(np.arange(n2), labels=list_labels1, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 1].set_title("Cross-correlation", fontsize=80)

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

ax[2,0].imshow(pca_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[2,0].set_xticks(np.arange(n1), labels=list_labels2, fontsize=40)
ax[2,0].set_yticks(np.arange(n2), labels=list_labels1, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[2, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[2, 0].set_title("PCA distance", fontsize=80)

# fig.suptitle('Models Test 1', fontsize=120)
fig.suptitle('Distance matrices', fontsize=120)

fig_name =  results_directory + "\\Distance_matrices.jpg"
plt.savefig(fig_name)

