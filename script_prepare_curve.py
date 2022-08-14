"""
3/8/2022
Author: Virginia Listanti

This script adapt Harvey Barons's script RIDGEDISTANCES.PY to prepare the  kiwi syllables curves for distances' test.

Process:
        - read extracted IF from .csv
        - DTW (dinamic time-warping) in time
        - subtract average frequency
        - resampling to minimum number of points to


Then: it saves a .jpg image with all the curves

The syllables are stored in DIRECTORY new curves are stored in NEWDIRECTORY and the .jpg image will be stored in
SAVEDIRECTORY

NOTE: WORK IN PROGRESS

"""

import os
import SignalProc
import numpy as np
import wavio
import matplotlib.pyplot as plt
import scipy
from geodesic_copy import geod_sphere
import Linear
import matplotlib.pyplot as plt
import csv
import DTW_functions as DTW


#################################################################################

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Exemplars Ridges"
directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
            "exemplars\\Models\\Models_Ridges"
# newdirectory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Exemplars_Ridges_new"
newdirectory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
            "exemplars\\Models\\Models_Ridges_new"


list_files = []
list_length = []
list_max_freq = []
for file in os.listdir(directory):
    if file.endswith("IF.csv"):
        list_files.append(file)
        curve = np.loadtxt(open(directory + "\\" + file, "rb"), delimiter=",", skiprows=1)[:,1]
        list_length.append(len(curve))
        list_max_freq.append(np.max(curve)-np.mean(curve))

n = len(list_files) #number of curves
# print(list_files)
min_len = np.min(list_length) #min. curves lenght
# max_len = np.max(list_length) #max. curves length
max_freq = np.max(list_max_freq)

# reference curve for DTW
reference_curve = np.loadtxt(open(directory + "\\" + list_files[0], "rb"), delimiter=",", skiprows=1)[:, 1]

#new points
new_times = np.linspace(0, 1, min_len)
# fig_name = newdirectory + "\\freq_plots_3_unstacked.jpg"
# fig, ax = plt.subplots()
fieldnames = ['t', "IF"]

for i in range(n):
    # dynamic time warping
    target_curve = np.loadtxt(open(directory + "\\" + list_files[i], "rb"), delimiter=",", skiprows=1)[:, 1]
    m = DTW.dtw(target_curve, reference_curve, wantDistMatrix=True)
    x, y = DTW.dtw_path(m)
    aligned_times = np.linspace(0,1, len(x))
    aligned_curve = target_curve[x]
    # subratct average
    aligned_curve -= np.mean(aligned_curve)
    #resample
    new_curve = np.interp(new_times, aligned_times, aligned_curve)

    #save new curve
    csvfilename = newdirectory + "\\" + list_files[i][:-7] + "_newIF.csv"
    with open(csvfilename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for j in range(len(new_curve)):
            writer.writerow({"t":new_times[j], "IF": new_curve[j]})

    #plot
    # ax.plot(new_times, new_curve+i*2*max_freq)
    # ax.plot(new_times, new_curve)


# fig.suptitle("Unstacked syllables")
# plt.savefig(fig_name)





