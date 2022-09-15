"""
14/09/2022

Author: Virginia Listanti

Small script to compare geodesic distance matrices
"""

# lybraries

import os
import numpy as np
import matplotlib.pyplot as plt

# Geodesic Distance Matrices paths
geo1 = np.loadtxt("C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\exemplars\\Smaller_Dataset\\Tests_results\\Test1\\Geodesic.txt")
geo2 = np.loadtxt("C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\exemplars\\Smaller_Dataset\\Tests_results\\Test2\\Geodesic.txt")
geo3 = np.loadtxt("C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\exemplars\\Smaller_Dataset\\Tests_results\\Test3\\Geodesic.txt")
geo4 = np.loadtxt("C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\exemplars\\Smaller_Dataset\\Tests_results\\Test4\\Geodesic.txt")
geo5 = np.loadtxt("C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\exemplars\\Smaller_Dataset\\Tests_results\\Test5\\Geodesic.txt")
geo6 = np.loadtxt("C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\exemplars\\Smaller_Dataset\\Tests_results\\Test6\\Geodesic.txt")
geo7 = np.loadtxt("C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\exemplars\\Smaller_Dataset\\Tests_results\\Test7\\Geodesic.txt")
geo8 = np.loadtxt("C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\exemplars\\Smaller_Dataset\\Tests_results\\Test8\\Geodesic.txt")

# file labels
file_label = []
dir_file = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\exemplars\\Smaller_Dataset\\Original"

for file in os.listdir(dir_file):
    if file.endswith('.wav'):
        file_label.append(file[:-4])

n = len(file_label)

# plot

fig, ax = plt.subplots(2,4, figsize=(100, 50))

ax[0,0].imshow(geo1, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,0].set_xticks(np.arange(n), labels=file_label, fontsize=40)
ax[0,0].set_yticks(np.arange(n), labels=file_label, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 0].set_title("Pre-aligned", fontsize=80)

ax[0,1].imshow(geo3, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,1].set_xticks(np.arange(n), labels=file_label, fontsize=40)
ax[0,1].set_yticks(np.arange(n), labels=file_label, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 1].set_title(" Cutted Pre-aligned", fontsize=80)

ax[0,2].imshow(geo5, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,2].set_xticks(np.arange(n), labels=file_label, fontsize=40)
ax[0,2].set_yticks(np.arange(n), labels=file_label, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 2].set_title("Smoothed Pre-aligned", fontsize=80)

ax[0,3].imshow(geo7, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[0,3].set_xticks(np.arange(n), labels=file_label, fontsize=40)
ax[0,3].set_yticks(np.arange(n), labels=file_label, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[0, 3].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[0, 3].set_title("Smoothed Cutted Pre-aligned", fontsize=80)

ax[1,0].imshow(geo2, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[1,0].set_xticks(np.arange(n), labels=file_label, fontsize=40)
ax[1,0].set_yticks(np.arange(n), labels=file_label, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[1, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[1, 0].set_title("Pairwise alignment", fontsize=80)

ax[1,1].imshow(geo4, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[1,1].set_xticks(np.arange(n), labels=file_label, fontsize=40)
ax[1,1].set_yticks(np.arange(n), labels=file_label, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[1, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[1, 1].set_title(" Cutted Pairwise alignment", fontsize=80)

ax[1,2].imshow(geo6, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[1,2].set_xticks(np.arange(n), labels=file_label, fontsize=40)
ax[1,2].set_yticks(np.arange(n), labels=file_label, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[1, 2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[1, 2].set_title("Smoothed Pairwise alignment", fontsize=80)

ax[1,3].imshow(geo8, cmap="Purples")
# Show all ticks and label them with the respective list entries
ax[1,3].set_xticks(np.arange(n), labels=file_label, fontsize=40)
ax[1,3].set_yticks(np.arange(n), labels=file_label, fontsize=40)
# Rotate the tick labels and set their alignment.
plt.setp(ax[1, 3].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax[1, 3].set_title("Smoothed Cutted Pairwise alignment", fontsize=80)


fig.suptitle('Geodesic matrices comparison', fontsize=120)

save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
                 "exemplars\\Smaller_Dataset"
fig_name = save_directory+"\\Geodesic_distance_matrices_comparison.jpg"
plt.savefig(fig_name)
