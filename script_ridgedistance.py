import os
import SignalProc
import numpy as np
import wavio
import matplotlib.pyplot as plt
import scipy
from geodesic_copy import geod_sphere
import Linear
import matplotlib.pyplot as plt

# directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
#             "exemplars\\Models\\Exemplars Ridges"
directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
            "exemplars\\Models\\Models-ridges"
# pairwise distance calculations
listridges = []
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith("IF.csv"):
            listridges.append(filename.replace('_IF.csv', ''))
n = len(listridges)
print(listridges)
# get length of longest
lengthmax = 0
for i in range(0,n):
    curve = np.transpose(np.loadtxt(open(directory + "\\" + listridges[i] + "_IF.csv", "rb"), delimiter=",", skiprows=1))
    print(np.shape(curve)[1])
    if np.shape(curve)[1] > lengthmax:
        lengthmax = np.shape(curve)[1]

print(lengthmax)
curves = np.zeros((n, lengthmax, 2))
for i in range(0,n):
    for j in range(0,lengthmax):
        try:
            curves[i, j,:] = np.loadtxt(open(directory + "/" + listridges[i] + "_IF.csv", "rb"), delimiter=",", skiprows=1)[j,:]
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

save_directory = "C:\\Users\\Virginia\\Documents\\Work\\Individual recognition\\Kiwi_IndividualID\\Kiwi_IndividualID\\" \
                 "exemplars\\Models\\Distance matrices"
# Plot the matrix
# plt.matshow(distances, cmap="Purples")
fig, ax = plt.subplots()
im = ax.imshow(distances, cmap="Purples")

# OLd example
# ax = plt.gca()

# # Set the plot labels
# xlabels = listridges
# ylabels = listridges
# ax.set_xticklabels(xlabels)
# ax.set_yticklabels(ylabels)

# #Add text to the plot showing the values at that point
# for i in range(n):
#     for j in range(n):
#         plt.text(j, i, distances[i,j], horizontalalignment='center', verticalalignment='center')

# plt.show()

# New example
# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(listridges)), labels=listridges)
ax.set_yticks(np.arange(len(listridges)), labels=listridges)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax.set_title("Distance matrix models")
fig.tight_layout()

fig_name = save_directory+"\\models.jpg"
plt.savefig(fig_name)
# np.savetxt(directory+"/spectrogramridges_2706.txt", listridges, fmt='%s')
# np.savetxt(directory+"/spectrogramdistances_2706.txt", distances, fmt='%s')
# np.savetxt(directory+"/dsspectrogramridges2.txt", np.array(listridges))
# np.savetxt(directory+"/dsspectrogramdistances2.txt", np.array(distances))
