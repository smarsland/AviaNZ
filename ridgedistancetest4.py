import os
import SignalProc
import numpy as np
import wavio
import matplotlib.pyplot as plt
import scipy
from geodesic_copy import geod_sphere
import Linear

directory = "/am/state-opera/home1/listanvirg/Documents/Individual_identification/extracted"

# pairwise distance calculations
listridges = []
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith(".csv"):
            listridges.append(filename.replace('.csv', ''))
n = len(listridges)
print(listridges)
# get length of longest
lengthmax = 0
for i in range(0,n):
    curve = np.transpose(np.loadtxt(open(directory + "/" + listridges[i] + ".csv", "rb"), delimiter=",", skiprows=1))
    print(np.shape(curve)[1])
    if np.shape(curve)[1] > lengthmax:
        lengthmax = np.shape(curve)[1]

print(lengthmax)
curves = np.zeros((n, lengthmax, 2))
for i in range(0,n):
    for j in range(0,lengthmax):
        try:
            curves[i, j,:] = np.loadtxt(open(directory + "/" + listridges[i] + ".csv", "rb"), delimiter=",", skiprows=1)[j,:]
        except IndexError:
            curves[i, j,0] = 0
            curves[i, j, 1] = 0

rscurves = Linear.resample(curves,53)[0]

print(rscurves[0])
distances = np.zeros((n, n))
for i in range(0,n):
    for j in range(i+1, n):
        d, _, _, = geod_sphere(np.vstack((np.arange(53),rscurves[i,:,1])), np.vstack((np.arange(53),rscurves[j,:,1])))
        distances[i, j] = d
        distances[j, i] = d
    print(i)
print(listridges)
print(distances)
save_directory = "/am/state-opera/home1/listanvirg/Documents/Harvey_results"
np.savetxt(directory+"/spectrogramridges_2706.txt", listridges, fmt='%s')
np.savetxt(directory+"/spectrogramdistances_2706.txt", distances, fmt='%s')
# np.savetxt(directory+"/dsspectrogramridges2.txt", np.array(listridges))
# np.savetxt(directory+"/dsspectrogramdistances2.txt", np.array(distances))