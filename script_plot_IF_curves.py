"""
12/08/22
Author: Virginia Listanti

This script plot the extracted IF in a subplot
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Exemplars_Ridges_new"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Exemplars_Ridges"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Models_Ridges_new"

directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
            "exemplars\\Models\\Models_Ridges"

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                  "exemplars\\Models\\Distance_ matrices\\Exemplars\\Test1"

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                  "exemplars\\Models\\Distance_ matrices\\Exemplars\\Test2"

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                  "exemplars\\Models\\Distance_ matrices\\Models\\Test1"

save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
                 "exemplars\\Models\\Distance_ matrices\\Models\\Test2"

fig, ax = plt.subplots(3,9, figsize=(300, 60))
i = 0
j = 0
for file in os.listdir(directory):
    if file.endswith("_IF.csv"):
        curve = np.loadtxt(directory + "//" + file, skiprows=1, delimiter=',')[:,1]
        ax[i, j].set_title(file[:-7], fontsize=80)
        ax[i, j].plot(curve)
        if j == 8:
            j = 0
            i += 1
        else:
            j += 1

fig_name = save_directory+"\\models_curves.jpg"
# fig_name = save_directory+"\\exemplars_curves.jpg"
plt.savefig(fig_name)
