"""
06/10/2022
Checked: 28/10/2022
Author: Virginia Listanti

This script smooth a set of curves using moving averages

Process:
        - read extracted IF from .csv
        - smoothing
        - seves new IF

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


#################################################################################

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset1\\Original"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset1\\Cutted"
# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#             "Smaller_Dataset2\\Original"
directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
            "Smaller_Dataset2\\Cutted"

# newdirectory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#                "Smaller_Dataset1\\Smoothed"
# newdirectory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#                "Smaller_Dataset1\\Cutted_smoothed"
# newdirectory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
#                "Smaller_Dataset2\\Smoothed"
newdirectory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\exemplars\\" \
               "Smaller_Dataset2\\Cutted_smoothed"


fieldnames = ['t', "IF"]
for file in os.listdir(directory):
    if file.endswith("IF.csv"):
        curve = np.loadtxt(open(directory + "\\" + file, "rb"), delimiter=",", skiprows=1)[:, 1]
        times = np.loadtxt(open(directory + "\\" + file, "rb"), delimiter=",", skiprows=1)[:, 0]
        newcurve = moving_average(curve, 21)

        # save new curve
        csvfilename = newdirectory + "\\" + file
        with open(csvfilename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for j in range(len(newcurve)):
                writer.writerow({"t": times[j], "IF": newcurve[j]})

