
# Compute all distances we want between a pair of contours
    # cross-correlation
    # DWT
    # SSD

    # geodesic distance in done, but add here
    # basic PCA is done
    # add Macleod PCA
import numpy as np

def dtw(x, y, wantDistMatrix=False):
    # Compute the dynamic time warp between two 1D arrays
    dist = np.zeros((len(x)+1,len(y)+1))
    dist[1:, :] = np.inf
    dist[:, 1:] = np.inf
    for i in range(len(x)):
        for j in range(len(y)):
            dist[i+1, j+1] = np.abs(x[i]-y[j]) + min(dist[i, j+1], dist[i+1, j], dist[i, j])
    if wantDistMatrix:
        return dist
    else:
        return dist[-1, -1]

s1 = np.loadtxt('/home/marslast/Dropbox/Kiwi_IndividualID/extracted/04_20_001_013_IF.csv',skiprows=1,delimiter=',')[:,1]
s2 = np.loadtxt('/home/marslast/Dropbox/Kiwi_IndividualID/extracted/04_20_001_013_IF.csv',skiprows=1,delimiter=',')[:,1]

# Cross-correlation
# This is not a distance -- we want the biggest score
# This will need some kind of normalisation for length (for now, just divide by max of the two)
from scipy.signal import correlate
xc = correlate(s1,s2)
xcscores = max(xc)/max(len(s1),len(s2))
print(xcscores)

# Now make 0 the best
#maxx = np.max(xcscores)
#xcscores = maxx - xcscores

# DTW
dt = dtw(s1,s2)
print(dt)

# SSD
# Pad to same length
l = np.argmax([len(s1),len(s2)])
if l==0:
    s2n = np.zeros(len(s1))
    s2n[:len(s2)] = s2
    s2 = s2n
else:
    s1n = np.zeros(len(s2))
    s1n[:len(s1)] = s1
    s1 = s1n

print(np.sum((s2-s1)**2)/len(s1))

