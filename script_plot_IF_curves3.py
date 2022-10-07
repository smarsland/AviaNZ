"""
28/09/22
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
directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
            "Smaller_Dataset1\\Cutted_smoothed_prep"
save_fold = "Classes_curves"
save_dir = directory +'\\'+ save_fold
if not save_fold in os.listdir(directory):
    os.makedirs(save_dir)


labels_lists = ["D", "E", "G", "K", "J", "L", "M", "O", "R", "Z"]
# labels_lists = ["D"]

for label in labels_lists:
    fig, ax = plt.subplots(2, 5, figsize=(300, 60), sharex = True, sharey = True)
    i = 0
    j = 0
    for file in os.listdir(directory):
        if file[0] == label:
            if file.endswith("_IF.csv"):
                curve = np.loadtxt(directory + "//" + file, skiprows=1, delimiter=',')[:, 1]
                times = np.loadtxt(directory + "//" + file, skiprows=1, delimiter=',')[:, 0]
                ax[i, j].set_title(file[:2], fontsize=80)
                ax[i, j].plot(times, curve)
                # ax[i,j].yaxis.set_tick_params(which='major', labelcolor='green',
                #                          labelleft=False, labelright=True)
                ax[i, j].yaxis.set_tick_params(labelsize = 50)
                ax[i, j].xaxis.set_tick_params(labelsize = 50)
                if j == 4:
                    j = 0
                    i += 1
                else:
                    j += 1
    fig.suptitle('Class ' + label, fontsize=120)
    fig_name = save_dir + "\\class"+label+"_IFcurves.jpg"
    plt.savefig(fig_name)


