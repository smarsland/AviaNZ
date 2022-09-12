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

IF TEST 1:
the code works on pre-prepared curves


"""

import os
import numpy as np
import matplotlib.pyplot as plt
import distances

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Exemplars_Ridges_new"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                "exemplars\\Models\\Exemplars_Ridges_cutted_new"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                 "exemplars\\Models\\Exemplars_Ridges_smooth1"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                 "exemplars\\Models\\Exemplars_Ridges_smooth2"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                 "exemplars\\Models\\Exemplars_Ridges_smooth3"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Models_Ridges_new"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Smaller_Dataset\\Original_prep"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Smaller_Dataset\\Cutted_prep"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Smaller_Dataset\\Smooth_prep"

directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
                "exemplars\\Smaller_Dataset\\Cutted_smooth_prep"

# save syllables list and find number of syllables and max lenght
# NOTE: we actually know n = 26 and len_max = 108 but kept to not lose generality (especially for when we will have to
# adapt for all our syllable set
list_syllables = []
len_max = 0
for file in os.listdir(directory):
    if file.endswith("_newIF.csv"):
        list_syllables.append(file)
        len_syl = len(np.loadtxt(directory + "//" + file, skiprows=1, delimiter=',')[:,1])
        if len_max < len_syl:
            len_max = np.copy(len_syl)
n = len(list_syllables)

# load curves
curves = np.zeros((n, len_max, 2))

for i in range(n):
    curves[i,:,:] = np.loadtxt(open(directory + "\\" + list_syllables[i], "rb"), delimiter=",", skiprows=1)

#pre-allocate distance_matrices
ssd_matrix = np.zeros((n,n))
crosscorr_matrix = np.zeros((n,n))
geod_matrix = np.zeros((n,n))
pca_matrix = np.zeros((n,n))


for i in range(n):
    for j in range(i, n):
        ssd_matrix[i, j] = distances.ssd(curves[i, :, 1], curves[j, :, 1])
        crosscorr_matrix[i, j] = distances.cross_corr(curves[i, :, 1], curves[j, :, 1])
        geod_matrix[i, j] = distances.Geodesic_curve_distance(curves[i, :, 0], curves[i, :, 1], curves[j, :, 0],
                                                             curves[j, :, 1])
        if i!=j:
            ssd_matrix[j, i] = ssd_matrix[i, j]
            crosscorr_matrix[j, i] = crosscorr_matrix[i, j]
            geod_matrix[j, i] = geod_matrix[i, j]

# prepare corosscorellation matrix for clustering
# NOTE: we don't need this but I am doing for having it for the next step
crosscorr_matrix = np.max(crosscorr_matrix) - crosscorr_matrix

# evaluate PCA matrix
pca_matrix = distances.pca_distance(curves[:, :,1])

# plot distance matrix

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                  "exemplars\\Models\\Distance_ matrices\\Exemplars\\Test1"

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                  "exemplars\\Models\\Distance_matrices\\Exemplars\\Test9"

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                  "exemplars\\Models\\Distance_ matrices\\Models\\Test1"

save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
                 "exemplars\\Smaller_Dataset\\Tests_results\\Test7"
#save matrices
np.savetxt(save_directory+"\\SSD.txt", ssd_matrix, fmt='%s')
np.savetxt(save_directory+"\\cross-correlation.txt", crosscorr_matrix, fmt='%s')
np.savetxt(save_directory+"\\Geodesic.txt", geod_matrix, fmt='%s')
np.savetxt(save_directory+"\\PCA.txt", pca_matrix, fmt='%s')

# Plot the matrices

list_labels = []
for element in list_syllables:
    list_labels.append(element[:-10])

fig, ax = plt.subplots(2,2, figsize=(80, 80))

ax[0,0].imshow(ssd_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,0].set_xticks(np.arange(n), labels=list_labels, fontsize=40)
ax[0,0].set_yticks(np.arange(n), labels=list_labels, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 0].set_title("SSD distance", fontsize=80)

ax[0,1].imshow(crosscorr_matrix, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,1].set_xticks(np.arange(n), labels=list_labels, fontsize=40)
ax[0,1].set_yticks(np.arange(n), labels=list_labels, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 1].set_title("Cross-correlation", fontsize=80)

ax[1,0].imshow(np.log(geod_matrix+1), cmap="Purples")
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

# fig.suptitle('Models Test 1', fontsize=120)
fig.suptitle('Test 7', fontsize=120)

# fig.tight_layout()

# fig_name = save_directory+"\\models_test1.jpg"
# fig_name = save_directory+"\\exemplars_test1.jpg"
# fig_name = save_directory+"\\exemplars_test9.jpg"
fig_name = save_directory + "\\distance_matrices_test7.jpg"
plt.savefig(fig_name)

