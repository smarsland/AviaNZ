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

def prepare_curves(curves_old, ref_curve, length_list):
    """
    This function prepare curves alignind them respect to reference curves
    
    Alignment using dynamic time-warping
    Resampling with min_lenght
    
    """
    N = np.shape(curves_old)[0]
    M = np.min(length_list)
    curves_new = np.zeros((N, M, 2))
    new_times = np.linspace(0, 1, M)

    for i in range(N):
        # dynamic time warping
        target_curve = curves_old[i, :length_list[i], 1]
        m = distances.dtw(target_curve, ref_curve, wantDistMatrix=True)
        x, y = distances.dtw_path(m)
        aligned_times = np.linspace(0, 1, len(x))
        aligned_curve = target_curve[x]
        # # subratct average
        # aligned_curve -= np.mean(aligned_curve)
        # resample
        curves_new[i, :, 0] = new_times
        curves_new[i, :, 1] = np.interp(new_times, aligned_times, aligned_curve)

    return curves_new

def symmetrize_matrix(A):
    """
    This function symmetrize a square matrix
    """

    return (A+A.T)/2

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Exemplars_Ridges"

directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
            "exemplars\\Models\\Models_Ridges"

# save syllables list and find number of syllables and max lenght
# NOTE: we actually know n = 26 and len_max = 108 but kept to not lose generality (especially for when we will have to
# adapt for all our syllable set
list_syllables = []
len_list = []
for file in os.listdir(directory):
    if file.endswith("_IF.csv"):
        list_syllables.append(file)
        len_syl = len(np.loadtxt(directory + "//" + file, skiprows=1, delimiter=',')[:,1])
        len_list.append(len_syl)
n = len(list_syllables)
len_max = np.max(len_list)
len_min = np.min(len_list)

# load curves
curves = np.zeros((n, len_max, 2))
for i in range(n):
    curves[i,:len_list[i],:] = np.loadtxt(open(directory + "\\" + list_syllables[i], "rb"), delimiter=",", skiprows=1)
    #subtract mean
    curves[i, :len_list[i], :] -= np.mean(curves[i, :len_list[i], :] )

#pre-allocate distance_matrices
ssd_matrix = np.zeros((n,n))
crosscorr_matrix = np.zeros((n,n))
geod_matrix = np.zeros((n,n))
pca_matrix = np.zeros((n,n))
dtw_matrix = np.zeros((n,n))


for i in range(n):
    # prepare curves row by row
    reference_curve = curves[i,:,:]
    new_curves = prepare_curves(curves, reference_curve[:,1], len_list)
    # evaluate PCA matrix
    pca_matrix[i, :] = distances.pca_distance(new_curves[:, :, 1])[i, :]
    for j in range(n):
        ssd_matrix[i, j] = distances.ssd(new_curves[i, :, 1], new_curves[j, :, 1])
        crosscorr_matrix[i, j] = distances.cross_corr(new_curves[i, :, 1], new_curves[j, :, 1])
        geod_matrix[i, j] = distances.Geodesic_curve_distance(new_curves[i, :, 0], new_curves[i, :, 1],
                                                              new_curves[j, :, 0], new_curves[j, :, 1])
        dtw_matrix[i,j] = distances.dtw(new_curves[i, :, 1], new_curves[j, :, 1])

# prepare corosscorellation matrix for clustering
# NOTE: we don't need this but I am doing for having it for the next step
crosscorr_matrix = np.max(crosscorr_matrix) - crosscorr_matrix

#symmetrize matrices

crosscorr_matrix = symmetrize_matrix(crosscorr_matrix)
ssd_matrix = symmetrize_matrix(ssd_matrix)
pca_matrix = symmetrize_matrix(pca_matrix)
dtw_matrix = symmetrize_matrix(dtw_matrix)
geod_matrix = symmetrize_matrix(geod_matrix)



# plot distance matrix

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                  "exemplars\\Models\\Distance_ matrices\\Exemplars\\Test2"

save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
                 "exemplars\\Models\\Distance_ matrices\\Models\\Test2"

#save matrices
np.savetxt(save_directory+"\\SSD.txt", ssd_matrix, fmt='%s')
np.savetxt(save_directory+"\\cross-correlation.txt", crosscorr_matrix, fmt='%s')
np.savetxt(save_directory+"\\Geodesic.txt", geod_matrix, fmt='%s')
np.savetxt(save_directory+"\\PCA.txt", pca_matrix, fmt='%s')
np.savetxt(save_directory+"\\DTW.txt", dtw_matrix, fmt='%s')

# Plot the matrices

list_labels = []
for element in list_syllables:
    list_labels.append(element[:-7])

fig, ax = plt.subplots(2,3, figsize=(80, 80))

ax[0,0].imshow(ssd_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,0].set_xticks(np.arange(n), labels=list_labels, fontsize=40)
ax[0,0].set_yticks(np.arange(n), labels=list_labels, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 0].set_title("SSD distance", fontsize=80)

ax[0,1].imshow(np.log(crosscorr_matrix+1), cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,1].set_xticks(np.arange(n), labels=list_labels, fontsize=40)
ax[0,1].set_yticks(np.arange(n), labels=list_labels, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 1].set_title("Cross-correlation", fontsize=80)

ax[0,2].imshow(dtw_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,2].set_xticks(np.arange(n), labels=list_labels, fontsize=40)
ax[0,2].set_yticks(np.arange(n), labels=list_labels, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 2].set_title("Dynamic Time-Warping", fontsize=80)

ax[1,0].imshow(geod_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[1,0].set_xticks(np.arange(n), labels=list_labels, fontsize=40)
ax[1,0].set_yticks(np.arange(n), labels=list_labels, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[1, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[1, 0].set_title("Geodesic distance", fontsize=80)

ax[1,1].imshow(pca_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[1,1].set_xticks(np.arange(n), labels=list_labels, fontsize=40)
ax[1,1].set_yticks(np.arange(n), labels=list_labels, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[1, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[1, 1].set_title("PCA distance", fontsize=80)

fig.suptitle('Models Test 2', fontsize=120)
# fig.suptitle('Exemplars Test 2', fontsize=120)

# fig.tight_layout()

fig_name = save_directory+"\\models_test2_log.jpg"
# fig_name = save_directory+"\\exemplars_test2.jpg"
plt.savefig(fig_name)

