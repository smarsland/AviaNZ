import numpy as np
import pylab as pl
from scipy import stats
#import umap
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

d1 = np.loadtxt("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\spectrogramdistances.txt")
n1 = np.loadtxt("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\spectrogramridges.txt",dtype = 'str')
n2 = np.loadtxt("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\scalogramridges.txt",dtype = 'str')
d2 = np.loadtxt("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\scalogramdistances.txt")

d3 = np.loadtxt("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\PCAspectrogramdistancesfixed.txt")
d4 = np.loadtxt("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\PCAscalogramdistancesfixed.txt")
d5 = np.loadtxt("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\downsampledspectrogramdistances.txt")


# model = AgglomerativeClustering(affinity='precomputed',distance_threshold=0, n_clusters=None,linkage="average")
#
# model = model.fit(d5)
# plt.title("Hierarchical Clustering Dendrogram")
# # plot the top three levels of the dendrogram
# plot_dendrogram(model, truncate_mode="level", p=3)
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.savefig("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\"+"dendrogramdownsampledspectrogramsrvf.jpg")
print("1: ")
print(np.std(d1))
print(np.mean(d1))
print("2: ")
print(np.std(d2))
print(np.mean(d2))
print("3: ")
print(np.std(d3))
print(np.mean(d3))
print("4: ")
print(np.std(d4))
print(np.mean(d4))
print("5: ")
print(np.std(d5))
print(np.mean(d5))

print("diff srvf:")
print(np.mean(np.abs(d1/np.max(d1)-d2/np.max(d2))))
print("diff srvf ds:")
print(np.mean(np.abs(d5/np.max(d5)-d2/np.max(d2))))
print("diff PCA:")
print(np.mean(np.abs(d3/np.max(d3)-d4/np.max(d4))))

k = 4
ind = np.argsort(d5, axis=1)[:,:k+1]

knn = np.zeros((1002,k+1), dtype='object')
birdknn = np.zeros((1002,k+1), dtype='int')
birdyearknn = np.zeros((1002,k+1), dtype='int')
callknn = np.zeros((1002,k+1), dtype='int')

for i in np.arange(0,1002):
    for j in np.arange(0,k+1):
        knn[i,j] = n2[ind[i,j],0].replace(" ridges","").split("_")
        birdknn[i,j] = int(knn[i,j][0])
        birdyearknn[i,j] = int(knn[i,j][0]+knn[i,j][1])
        callknn[i,j] = int(knn[i,j][0]+knn[i,j][1]+knn[i,j][2])
  #  plt.plot(np.sort(d4[:,i]),'r',alpha=0.1)
#plt.savefig("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\"+"PCAscalogramdistancesortedfixed.jpg")

mode = np.zeros((1002,7),dtype='object')
for i in np.arange(0,1002):
    mode[i,0] = n2[i]
    mode[i,1] = int(stats.mode(birdknn[i,1:])[0])
    mode[i,2] = int(stats.mode(birdyearknn[i, 1:])[0])
    mode[i,3] = int(stats.mode(callknn[i, 1:])[0])
    mode[i,4] = birdknn[i,0]==mode[i,1]
    mode[i,5] = birdyearknn[i,0]==mode[i,2]
    mode[i,6] = callknn[i,0]==mode[i,3]

print(np.count_nonzero(mode[:,4]) / 1002)
print(np.count_nonzero(mode[:,6]) / 1002)
#np.savetxt("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\"+"knnPCAscalogramfixed.csv", np.array(mode),fmt='%s')
