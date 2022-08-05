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
import DTW_functions


#################################################################################

directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
            "exemplars\\Models\\Exemplars Ridges"
newdirectory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
            "exemplars\\Models\\Exemplars_Ridges_new"


list_files = []
list_length = []
for file in os.listdir(directory):
    if file.endswith("IF.csv"):
        list_files.append(file)
        curve = np.loadtxt(open(directory + "\\" + file, "rb"), delimiter=",", skiprows=1)[:,1]
        list_length.append(len(curve))

n = len(list_files) #number of curves
# print(list_files)
min_len = np.min(list_length) #min. curves lenght
max_len = np.max(list_length) #max. curves length

#pre-allocate curvespece
old_curves = np.zeros((n, max_len, 2))
new_curves = np.zeros((n, min_len, 2))

#store curves
for i in range(2):
    csvfilename = directory + "\\" + list_files[i]
    curve = []
    with open(csvfilename) as csvfile:
        # open file as csv file
        csvReader = csv.reader(csvfile)
        # loop over rows
        for row in csvReader:
            curve.append(row)

    curve = np.array(curve[1:][:]).astype('float')
    old_curves[i, :list_length[i], :] = curve

# plt.plot(old_curves[0,:list_length[0], 1], 'r')
# plt.plot(old_curves[1,:list_length[1], 1], 'k')
# plt.show()

# modify curves
# initialise for first curve
reference_curve = np.copy(old_curves[0, :list_length[0], :])
target_curve = np.copy(old_curves[1, :, :])
m = dtw(reference_curve[:, 1], target_curve[:list_length[1]][1], wantDistMatrix=True)
x, y = dtw_path(m)
aligned_times = np.arange(len(y))/len(y)
aligned_curve = target_curve[y, 1]
plt.plot(old_curves[0,:list_length[0], 1], 'r')
plt.plot(old_curves[1,:list_length[1], 1], 'k')
plt.plot(reference_curve[:, 1], 'y')
plt.plot(aligned_curve, 'b')
plt.show()
old_curves[0, :, 1] -= np.mean(old_curves[0][:list_length[0]][1])
points = np.linspace(0, 1, min_len)
new_curves[0, :, 0] = points
new_curves[0, :, 1] = np.interp(points, old_curves[0, :list_length[0], 0], old_curves[0, :list_length[0], 1])


# for i in range(n):
#     # dynamic time warping
#     m = dtw(old_curves[i][:list_length[i]][1], reference_curve[:, 1], wantDistMatrix=True)
#     x, y = dtw_path(m)
#     aligned_times = np.arange(len(x))/len(x)
#     aligned_curve = old_curves[i][x][1]
#     # subratct average
#     aligned_curve -= np.mean(aligned_curve)
#     #resample
#     new_curves[i, :, 0] = points
#     new_curves[i, :, 1] = np.interp(points, aligned_times, aligned_times)


#plot and save

fig_name = newdirectory + "\\2freq_plots.jpg"
fig, ax = plt.subplots()
fieldnames = ['t', "IF"]
max_freq = np.max(new_curves[:][:][1])
for i in range(2,5):
    #save new if
    csvfilename = newdirectory + "\\" + list_files[i][:-7] + "_newIF.csv"
    with open(csvfilename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for j in range(min_len):
            writer.writerow({"t": new_curves[i][j][0], "IF": new_curves[i][j][1]})

    # plot
    #ax.plot(new_curves[i][:][1]+i*max_freq)
    ax.plot(new_curves[i][:][1])

fig.suptitle("2 Stacked syllables")
plt.savefig(fig_name)





