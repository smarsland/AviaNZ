"""
29/7/2022
Author: Virginia Listanti

This script adapt Harvey Barons's script RIDGEDISTANCES.PY to create the intra-group distance matrix of 2 sets of
syllables.

The syllables are stored in DIRECTORY1 and  DIRECTORY2 and the .jpg distance matrix will be stored in SAVEDIRECTORY

NOTE: WORK IN PROGRESS

"""


import os
import SignalProc
import numpy as np
import wavio
import matplotlib.pyplot as plt
import scipy
#from geodesic_copy import geod_sphere
from fdasrsf.geodesic import geod_sphere
import Linear
import matplotlib.pyplot as plt

directory1 = "../exemplars/Models/Exemplars-Ridges" 
directory2 = "../exemplars/Models/Models-Ridges" 
#directory1 = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
            #"exemplars\\Models\\Exemplars Ridges"
#directory2 = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
            #"exemplars\\Models\\Models-ridges"
# pairwise distance calculations
#originals
listridges1 = []
for root, dirs, files in os.walk(directory1):
    for filename in files:
        if filename.endswith("IF.csv"):
            listridges1.append(filename.replace('_IF.csv', ''))
n1 = len(listridges1)
print(listridges1)
# get length of longest
lengthmax = 0
for i in range(0, n1):
    curve = np.transpose(np.loadtxt(open(directory1 + "/" + listridges1[i] + "_IF.csv", "rb"), delimiter=",", skiprows=1))
    #print(np.shape(curve)[1])
    if np.shape(curve)[1] > lengthmax:
        lengthmax = np.shape(curve)[1]

print(lengthmax)
curves1 = np.zeros((n1, lengthmax, 2))
for i in range(0,n1):
    for j in range(0,lengthmax):
        try:
            curves1[i, j,:] = np.loadtxt(open(directory1 + "/" + listridges1[i] + "_IF.csv", "rb"), delimiter=",", skiprows=1)[j,:]
        except IndexError:
            curves1[i, j,0] = 0
            curves1[i, j, 1] = 0

rscurves1 = Linear.resample(curves1,53)[0]

#models
listridges2 = []
for root, dirs, files in os.walk(directory2):
    for filename in files:
        if filename.endswith("IF.csv"):
            listridges2.append(filename.replace('_IF.csv', ''))
n2 = len(listridges2)
print(listridges2)
# get length of longest
lengthmax2 =0
for i in range(0, n2):
    curve = np.transpose(np.loadtxt(open(directory2 + "/" + listridges2[i] + "_IF.csv", "rb"), delimiter=",", skiprows=1))
    #print(np.shape(curve)[1])
    if np.shape(curve)[1] > lengthmax2:
        lengthmax2= np.shape(curve)[1]

print(lengthmax2)
curves2 = np.zeros((n2, lengthmax2, 2))
for i in range(0,n2):
    for j in range(0,lengthmax2):
        try:
            curves2[i, j,:] = np.loadtxt(open(directory2 + "/" + listridges2[i] + "_IF.csv", "rb"), delimiter=",", skiprows=1)[j,:]
        except IndexError:
            curves2[i, j,0] = 0
            curves2[i, j, 1] = 0

rscurves2 = Linear.resample(curves2,53)[0]
#
# print(rscurves[0])
print("here")
distances = np.zeros((n2, n1))
for i in range(0,n2):
    for j in range(i, n1):
        d, _, _, = geod_sphere(np.vstack((np.arange(53),rscurves2[i,:,1])), np.vstack((np.arange(53),rscurves1[j,:,1])))
        distances[i, j] = d
        distances[j, i] = d
    print(i)
# print(listridges)
# print(distances)

#save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
                 #"exemplars\\Models\\Distance matrices"
save_directory = "."

# Plot the matrix
fig, ax = plt.subplots()
im = ax.imshow(distances, cmap="Purples")

# Show all ticks and label them with the respective list entries
#ax.set_xticks(np.arange(len(listridges1)), labels=listridges1)
#ax.set_yticks(np.arange(len(listridges2)), labels=listridges2)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

ax.set_title("Distance matrix original vs models")
fig.tight_layout()

fig_name = save_directory+"\\original_vs_models.jpg"
plt.savefig(fig_name)
# np.savetxt(directory+"/spectrogramridges_2706.txt", listridges, fmt='%s')
# np.savetxt(directory+"/spectrogramdistances_2706.txt", distances, fmt='%s')
# np.savetxt(directory+"/dsspectrogramridges2.txt", np.array(listridges))
# np.savetxt(directory+"/dsspectrogramdistances2.txt", np.array(distances))
