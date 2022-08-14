"""
5/8/2022
Author Virginia Listanti

This script evaluate the distances among syllables stored in DIRECTORY. And, then, store the distances in distances
matrices which plot is then saved into SAVE_DIR

DISTANCES:
* dtw
* ssd
* srfv
* pca
* cross-correlation


IF TEST 2:
the code prepare the curves aligning them row by row
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import distances

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
        target_curve = curves_old[i, :length_list[i], 1]
        m = distances.dtw(target_curve, ref_curve[:,1], wantDistMatrix=True)
        x, y = distances.dtw_path(m)
        aligned_times = np.linspace(0, 1, len(x))
        aligned_curve = target_curve[x]
        # # subratct average
        # aligned_curve -= np.mean(aligned_curve)
        # resample
        curves_new[i, :, 0] = new_times
        curves_new[i, :, 1] = np.interp(new_times, aligned_times, aligned_curve)

    new_ref_curve = np.interp(new_times, ref_curve[:,0], ref_curve[:,1])

    return curves_new, new_ref_curve

def symmetrize_matrix(A):
    """
    This function symmetrize a square matrix
    """

    return (A+A.T)/2



directory1 = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
            "exemplars\\Models\\Exemplars_Ridges"

directory2 = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
            "exemplars\\Models\\Models_Ridges"

# save syllables list and find number of syllables and max lenght for both sets

# NOTE: max lenght can be different: we will have to resample

list_syllables1 = []
len_list1 = []
for file in os.listdir(directory1):
    if file.endswith("_IF.csv"):
        list_syllables1.append(file)
        len_syl = len(np.loadtxt(directory1 + "//" + file, skiprows=1, delimiter=',')[:,1])
        len_list1.append(len_syl)

n = len(list_syllables1)
len_max1 = np.max(len_list1)

list_syllables2 = []
len_list2 = []
for file in os.listdir(directory2):
    if file.endswith("_IF.csv"):
        list_syllables2.append(file)
        len_syl = len(np.loadtxt(directory2 + "//" + file, skiprows=1, delimiter=',')[:,1])
        len_list2.append(len_syl)

len_max2 = np.max(len_list2)

len_min = min(np.min(len_list1), np.min(len_list2))

# load curves
curves1 = np.zeros((n, len_max1, 2))
curves2 = np.zeros((n, len_max2, 2))

for i in range(n):
    curves1[i,:len_list1[i], :] = np.loadtxt(open(directory1 + "\\" + list_syllables1[i], "rb"), delimiter=",",
                                             skiprows=1)
    # subtract mean
    curves1[i, :len_list1[i], :] -= np.mean(curves1[i, :len_list1[i], :])

    curves2[i, :len_list2[i], :] = np.loadtxt(open(directory2 + "\\" + list_syllables2[i], "rb"), delimiter=",",
                                              skiprows=1)
    # subtract mean
    curves2[i, :len_list2[i], :] -= np.mean(curves2[i, :len_list2[i], :])


#pre-allocate distance_matrices
ssd_matrix = np.zeros((n,n))
crosscorr_matrix = np.zeros((n,n))
geod_matrix = np.zeros((n,n))
# pca_matrix = np.zeros((n,n))
dtw_matrix = np.zeros((n,n))


for i in range(n):
    # prepare curves row by row
    reference_curve = curves1[i,:len_list1[i],:]
    new_curves, new_reference_curve = prepare_curves(curves2, reference_curve, len_list2, len_min)
    # # evaluate PCA matrix
    # pca_matrix[i, :] = distances.pca_distance(new_curves[:, :, 1])[i, :]
    for j in range(n):
        ssd_matrix[i, j] = distances.ssd(new_reference_curve, new_curves[j, :, 1])
        crosscorr_matrix[i, j] = distances.cross_corr(new_reference_curve, new_curves[j, :, 1])
        geod_matrix[i, j] = distances.Geodesic_curve_distance(new_curves[i, :, 0], new_reference_curve,
                                                              new_curves[j, :, 0], new_curves[j, :, 1])
        dtw_matrix[i,j] = distances.dtw(new_reference_curve, new_curves[j, :, 1])

# prepare corosscorellation matrix for clustering
# NOTE: we don't need this but I am doing for having it for the next step
crosscorr_matrix = np.max(crosscorr_matrix) - crosscorr_matrix

#symmetrize matrices

crosscorr_matrix = symmetrize_matrix(crosscorr_matrix)
ssd_matrix = symmetrize_matrix(ssd_matrix)
# pca_matrix = symmetrize_matrix(pca_matrix)
dtw_matrix = symmetrize_matrix(dtw_matrix)
geod_matrix = symmetrize_matrix(geod_matrix)



# plot distance matrix

save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
                 "exemplars\\Models\\Distance_ matrices\\Exemplars_vs_Models\\Test2"

#save matrices
np.savetxt(save_directory+"\\SSD.txt", ssd_matrix, fmt='%s')
np.savetxt(save_directory+"\\cross-correlation.txt", crosscorr_matrix, fmt='%s')
np.savetxt(save_directory+"\\Geodesic.txt", geod_matrix, fmt='%s')
# np.savetxt(save_directory+"\\PCA.txt", pca_matrix, fmt='%s')
np.savetxt(save_directory+"\\DTW.txt", dtw_matrix, fmt='%s')

# Plot the matrices

list_labels1 = []
list_labels2 = []
for i in range(n):
    list_labels1.append(list_syllables1[i][:-7])
    list_labels2.append(list_syllables2[i][:-7])

fig, ax = plt.subplots(2,2, figsize=(80, 80))

ax[0,0].imshow(ssd_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,0].set_xticks(np.arange(n), labels=list_labels2, fontsize=40)
ax[0,0].set_yticks(np.arange(n), labels=list_labels1, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 0].set_title("SSD distance", fontsize=80)

ax[0,1].imshow(crosscorr_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,1].set_xticks(np.arange(n), labels=list_labels2, fontsize=40)
ax[0,1].set_yticks(np.arange(n), labels=list_labels1, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 1].set_title("Cross-correlation", fontsize=80)

ax[1,0].imshow(geod_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[1,0].set_xticks(np.arange(n), labels=list_labels2, fontsize=40)
ax[1,0].set_yticks(np.arange(n), labels=list_labels1, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[1, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[1, 0].set_title("Geodesic distance", fontsize=80)

ax[1,1].imshow(dtw_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[1,1].set_xticks(np.arange(n), labels=list_labels2, fontsize=40)
ax[1,1].set_yticks(np.arange(n), labels=list_labels1, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[1, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[1, 1].set_title("DTW distance", fontsize=80)

# fig.suptitle('Models Test 1', fontsize=120)
fig.suptitle('Exemplars vs Models Test 2', fontsize=120)

fig_name = save_directory+"\\exemplars_vs_models_test2.jpg"
plt.savefig(fig_name)

