import os
import SignalProc
import numpy as np
import wavio
import matplotlib.pyplot as plt
import scipy
from fdasrsf.geodesic import geod_sphere

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
curves = np.zeros((n, lengthmax))
resampledcurves = np.zeros((n, lengthmax))
for i in range(0,n):
    for j in range(0,lengthmax):
        try:
            curves[i, j] = np.transpose(np.loadtxt(open(directory + "/" + listridges[i] + ".csv", "rb"), delimiter=",", skiprows=1))[1, j]
        except IndexError:
            curves[i, j] = 0
    resampledcurves[i,:] = scipy.signal.cspline1d_eval(scipy.signal.cspline1d(np.transpose(np.loadtxt(open(directory + "/" + listridges[i] + ".csv", "rb"), delimiter=",", skiprows=1))[1, :]), lengthmax)
print(resampledcurves)
distances = np.zeros((n, n))
for i in range(0,n):
    for j in range(i+1, n):
        print(np.column_stack((curves[i,:],np.arange(53))))
        d, _, _, = geod_sphere(np.vstack((np.arange(lengthmax),curves[i,:])), np.vstack((np.arange(lengthmax),curves[j,:])))
        distances[i, j] = d
        distances[j, i] = d

print(listridges)
print(distances)
save_directory = "/am/state-opera/home1/listanvirg/Documents/Harvey_results"
np.savetxt(directory+"/spectrogramridges.txt", listridges, fmt='%s')
np.savetxt(directory+"/spectrogramdistances.txt", distances, fmt='%s')
# np.savetxt(directory+"/spectrogramridges2.txt", np.array(listridges))
# np.savetxt(directory+"/spectrogramdistances2.txt", np.array(distances))
