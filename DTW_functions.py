"""
Class for dtw functions written by Stephen Marsland
"""
import numpy as np

# dtw functions
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

def findDTWMatches(seg, data):
    # TODO: This is slow and crap. Note all the same length, for a start, and the fact that it takes forever!
    # Use MFCC first?
    d = np.zeros(len(data))
    for i in range(len(data)):
        d[i] = dtw(seg, data[i:i+len(seg)])
    return d

def dtw_path(d):
    # Shortest path through DTW matrix
    i = np.shape(d)[0]-2
    j = np.shape(d)[1]-2
    xpath = [i]
    ypath = [j]
    while i>0 or j>0:
            next = np.argmin((d[i,j],d[i+1,j],d[i,j+1]))
            if next == 0:
                i -= 1
                j -= 1
            elif next == 1:
                j -= 1
            else:
                i -= 1
            xpath.insert(0,i)
            ypath.insert(0,j)
    return xpath, ypath
