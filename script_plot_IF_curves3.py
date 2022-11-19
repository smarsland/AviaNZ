"""
28/09/22
Checked: 28/10/2022
Author: Virginia Listanti

This script plot the extracted IF in a subplot divided by classes
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset1\\Original"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset1\\Original_prep"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset1\\Smoothed"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset1\\Smoothed_prep"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset1\\Cutted"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset1\\Cutted_prep"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset1\\Cutted_smoothed"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset1\\Cutted_smoothed_prep"

directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
            "Smaller_Dataset2\\Original"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset2\\Original_prep"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset2\\Smoothed"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset2\\Smoothed_prep"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset2\\Cutted"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset2\\Cutted_prep"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset2\\Cutted_smoothed"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset2\\Cutted_smoothed_prep"

save_fold = "Classes_curves"
# save_dir = directory +'\\'+ save_fold
save_dir = "C:\\Users\\Virginia\\Documents\\Work\\Thesis_images\\Chapter 5\\Classes_curves"
if not save_fold in os.listdir(directory):
    os.makedirs(save_dir)


labels_lists = ["D", "E", "J", "K", "L", "M", "O", "R", "Z"]
# labels_lists = ["D"]
col_list = [7, 4, 5, 4, 6, 4, 4, 2, 4]

for label in labels_lists:
    col_max = int(col_list[labels_lists.index(label)])
    fig, ax = plt.subplots(2, col_max +1, figsize=(30, 6), sharex = True, sharey = True)
    i = 0
    j = 0
    for file in os.listdir(directory):
        if file[0] == label:
            if file.endswith("_IF.csv"):
                curve = np.loadtxt(directory + "//" + file, skiprows=1, delimiter=',')[:, 1]
                times = np.loadtxt(directory + "//" + file, skiprows=1, delimiter=',')[:, 0]
                ax[i, j].set_title(file[:-7], fontsize=12)
                ax[i, j].plot(times, curve)
                # ax[i,j].yaxis.set_tick_params(which='major', labelcolor='green',
                #                          labelleft=False, labelright=True)
                ax[i, j].yaxis.set_tick_params(labelsize = 10)
                ax[i, j].xaxis.set_tick_params(labelsize = 10)
                if j == col_max:
                    j = 0
                    i += 1
                else:
                    j += 1

    fig.suptitle('Class ' + label, fontsize=16)
    fig_name = save_dir + "\\class"+label+"_IFcurves.jpg"
    fig.tight_layout()
    plt.savefig(fig_name, dpi = 200)


