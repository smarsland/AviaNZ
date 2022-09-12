"""
12/08/22
Author: Virginia Listanti

This script plot the extracted IF in a subplot
"""

import os
import numpy as np
import matplotlib.pyplot as plt

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

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Exemplars_Ridges_new"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Exemplars_Ridges"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Exemplars_Ridges_cutted_new"
#
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Exemplars_Ridges_cutted"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Smaller_Dataset\\Original"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Smaller_Dataset\\Original_prep"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Smaller_Dataset\\Cutted_prep"

directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
            "exemplars\\Smaller_Dataset\\Cutted"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Smaller_Dataset\\Smooth_prep"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Smaller_Dataset\\Smooth_prep"


# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Models_Ridges_new"

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Models_Ridges"

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                  "exemplars\\Models\\Distance_ matrices\\Exemplars\\Test1"

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                  "exemplars\\Models\\Distance_ matrices\\Exemplars\\Test2"

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID" \
#                  "\\exemplars\\Models\\Distance_ matrices\\Exemplars\\Test3"

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID" \
#                  "\\exemplars\\Models\\Distance_ matrices\\Exemplars\\Test4"

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                  "exemplars\\Models\\Distance_ matrices\\Models\\Test1"

# save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#                  "exemplars\\Models\\Distance_ matrices\\Models\\Test2"

save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
                 "exemplars\\Smaller_Dataset\\Tests_results\\Test8"

# fig, ax = plt.subplots(3,9, figsize=(300, 60))
fig, ax = plt.subplots(4,10, figsize=(300, 60))
i = 0
j = 0
for file in os.listdir(directory):
    if file.endswith("IF.csv"):
        curve = moving_average(np.loadtxt(directory + "//" + file, skiprows=1, delimiter=',')[:,1], 21)
        ax[i, j].set_title(file[:-7], fontsize=80)
        ax[i, j].plot(curve)
        if j == 9:
            j = 0
            i += 1
        else:
            j += 1

# fig_name = save_directory+"\\models_curves.jpg"
# fig_name = save_directory+"\\exemplars_curves.jpg"
fig_name = save_directory+"\\IF_curves.jpg"
plt.savefig(fig_name)
