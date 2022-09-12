"""
11/8/2022
Author Virginia Listanti

This script evaluate the distances among syllables stored in DIRECTORY1 with syllables stored in DIRECTORY2. And, then,
store the distances in distances matrices which plot is then saved into SAVE_DIR.

DISTANCES:
* dtw
* ssd
* srfv
* pca
* cross-correlation

IF TEST 1:
the code works on pre-prepared curves

IF TEST 2:
the code prepare the curves aligning them row by row
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import distances

# directory1 = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Exemplars_Ridges_new"

# directory1 = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                "exemplars\\Models\\Exemplars_Ridges_cutted_new"

# directory1 = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                 "exemplars\\Models\\Exemplars_Ridges_smooth1"

directory1 = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
                "exemplars\\Models\\Exemplars_Ridges_smooth3"

# directory2 = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Models_Ridges_new"

directory2 = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
            "exemplars\\Models\\Models_Ridges_smooth2"

# save syllables list and find number of syllables and max lenght for both sets

# NOTE: max lenght can be different: we will have to resample

list_syllables1 = []
len_max1 = 0
for file in os.listdir(directory1):
    if file.endswith("_newIF.csv"):
        list_syllables1.append(file)
        len_syl = len(np.loadtxt(directory1 + "//" + file, skiprows=1, delimiter=',')[:,1])
        if len_max1 < len_syl:
            len_max1 = np.copy(len_syl)
n = len(list_syllables1)

list_syllables2 = []
len_max2 = 0
for file in os.listdir(directory2):
    if file.endswith("_newIF.csv"):
        list_syllables2.append(file)
        len_syl = len(np.loadtxt(directory2 + "//" + file, skiprows=1, delimiter=',')[:,1])
        if len_max2 < len_syl:
            len_max2 = np.copy(len_syl)

# load curves
curves1 = np.zeros((n, len_max1, 2))
curves2 = np.zeros((n, len_max2, 2))

for i in range(n):
    curves1[i,:,:] = np.loadtxt(open(directory1 + "\\" + list_syllables1[i], "rb"), delimiter=",", skiprows=1)
    curves2[i, :, :] = np.loadtxt(open(directory2 + "\\" + list_syllables2[i], "rb"), delimiter=",", skiprows=1)


# resample if max length is not the same

if len_max1 > len_max2:
    newcurves2 = np.copy(curves2)
    newcurves1 = np.zeros((n, len_max2, 2))
    new_times = np.linspace(0, 1, len_max2)
    for i in range(n):
        newcurves1[i,:,0] = new_times
        newcurves1[i, :, 1] = np.interp(new_times, curves1[i, :, 0], curves1[i, :, 1])
elif len_max2 > len_max1:
    newcurves1 = np.copy(curves1)
    newcurves2 = np.zeros((n, len_max1, 2))
    new_times = np.linspace(0, 1, len_max1)
    for i in range(n):
        newcurves2[i,:,0] = new_times
        newcurves2[i, :, 1] = np.interp(new_times, curves2[i, :, 0], curves2[i, :, 1])
else:
    newcurves1 = np.copy(curves1)
    del curves1
    newcurves2 = np.copy(curves2)
    del curves2



#pre-allocate distance_matrices
ssd_matrix = np.zeros((n,n))
crosscorr_matrix = np.zeros((n,n))
geod_matrix = np.zeros((n,n))
dtw_matrix = np.zeros((n,n))


for i in range(n):
    for j in range(i, n):
        ssd_matrix[i, j] = distances.ssd(newcurves1[i, :, 1], newcurves2[j, :, 1])
        crosscorr_matrix[i, j] = distances.cross_corr(newcurves1[i, :, 1], newcurves2[j, :, 1])
        geod_matrix[i, j] = distances.Geodesic_curve_distance(newcurves1[i, :, 0], newcurves1[i, :, 1],
                                                              newcurves2[j, :, 0], newcurves2[j, :, 1])
        dtw_matrix[i,j] = distances.dtw(newcurves1[i, :, 1], newcurves2[j, :, 1])
        if i!=j:
            ssd_matrix[j, i] = ssd_matrix[i, j]
            crosscorr_matrix[j, i] = crosscorr_matrix[i, j]
            geod_matrix[j, i] = geod_matrix[i, j]
            dtw_matrix[j, i] = dtw_matrix[i,j]

# prepare corosscorellation matrix for clustering
# NOTE: we don't need this but I am doing for having it for the next step
crosscorr_matrix = np.max(crosscorr_matrix) - crosscorr_matrix



# plot distance matrix

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                  "exemplars\\Models\\Distance_ matrices\\Exemplars_vs_Models\\Test1"

save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
                 "exemplars\\Models\\Distance_matrices\\Exemplars_vs_Models\\Test9"

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                  "exemplars\\Models\\Distance_ matrices\\Models\\Test1"

#save matrices
np.savetxt(save_directory+"\\SSD.txt", ssd_matrix, fmt='%s')
np.savetxt(save_directory+"\\cross-correlation.txt", crosscorr_matrix, fmt='%s')
np.savetxt(save_directory+"\\Geodesic.txt", geod_matrix, fmt='%s')
np.savetxt(save_directory+"\\DTW.txt", dtw_matrix, fmt='%s')

# Plot the matrices

list_labels1 = []
list_labels2 = []
for i in range(n):
    list_labels1.append(list_syllables1[i][:-10])
    list_labels2.append(list_syllables2[i][:-10])


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

ax[1,0].imshow(np.log(geod_matrix+1), cmap="Purples")
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
fig.suptitle('Exemplars vs Models Test 9', fontsize=120)

# fig.tight_layout()

# fig_name = save_directory+"\\models_test1.jpg"
fig_name = save_directory+"\\exemplars_vs_models_test9.jpg"
plt.savefig(fig_name)