"""
5/8/2022
Class with all the required distance functions.
Reviewed by: Virginia Listanti

# Compute all distances we want between a pair of contours
    # cross-correlation
    # DWT
    # SSD
    # geodesic distance
    # basic PCA functions

TO DO:

    # add Macleod PCA
"""

import numpy as np
from scipy.signal import correlate
from geodesic_copy import geod_sphere

def dtw(x, y, wantDistMatrix=False):
    """
    Compute the dynamic time warp between two 1D arrays
    OK
    """
    dist = np.zeros((len(x) + 1, len(y) + 1))
    dist[1:, :] = np.inf
    dist[:, 1:] = np.inf
    for i in range(len(x)):
        for j in range(len(y)):
            dist[i + 1, j + 1] = np.abs(x[i] - y[j]) + min(dist[i, j + 1], dist[i + 1, j], dist[i, j])
    if wantDistMatrix:
        return dist
    else:
        return dist[-1, -1]

def dtw_path(d):
    # Shortest path through DTW matrix
    i = np.shape(d)[0] - 2
    j = np.shape(d)[1] - 2
    xpath = [i]
    ypath = [j]
    while i > 0 or j > 0:
        next = np.argmin((d[i, j], d[i + 1, j], d[i, j + 1]))
        if next == 0:
            i -= 1
            j -= 1
        elif next == 1:
            j -= 1
        else:
            i -= 1
        xpath.insert(0, i)
        ypath.insert(0, j)
    return xpath, ypath


def ssd(s1, s2):
    """
    Sum-squared distance (l2) bewtween two 1-d signals: S1, S2
    NOTE:  we assume that len(s1) = len(s2)
    OK
    """

    return np.sum((s2-s1)**2)/len(s1)


def cross_corr(s1, s2):
    """
    NORMALISED Cross correlation: ok
     We take the max of the correlation vector and divide by std of both signals

     NOTE: This cannot be use for clustering directly!!
    """
    xc = correlate(s1, s2)

    # just make the max end interpret it correctly. We can assume all have the same lenght.
    xscore = max(xc)

    # # Divide by both signals
    # xscore /= np.std(s1)
    # xscore /= np.std(s2)

    return xscore

def Geodesic_curve_distance(x1, y1, x2, y2):
    """
    Code suggested by Arianna Salili-James
    This function computes the distance between two curves x and y using geodesic distance

    Input:
         - x1, y1 coordinates of the first curve
         - x2, y2 coordinates of the second curve
    """

    beta1 = np.column_stack([x1, y1]).T
    beta2 = np.column_stack([x2, y2]).T

    distance, _, _ = geod_sphere(np.array(beta1), np.array(beta2), rotation=False)

    # distance, _, _ = geod_sphere(np.array(beta1), np.array(beta2))

    return distance

def asm(curves):
    """
    This function calculate eigenvalues and eigenvectors of the covariance matrix of CURVES


    """
    #standardise
    means = np.mean(curves, axis=0)
    sd = np.std(curves, axis=0)
    curves = curves - means
    curves = (curves - means)/sd

    # Compute the covariance matrix
    C = np.cov(curves.T) #ok I guess

    # Get the eigenvalues and eigenvectors
    evals, evecs = np.linalg.eig(C)

    # Now need to sort them into descending order
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:, indices]
    evals = evals[indices]
    evecs = np.real(evecs)
    evals = np.real(evals)

    return means, evals, evecs

def pca_distance(data, ps_dim = 4):
    """
    This function evaluate the distance matrix using PCA over a PS_DIM projection space of DATA

    If not given PS_DIM = 4
    we evaluated it in the experiment reported in the notebook Test_distance

    Note: Data is assumed to have the dimensions (number os data, number of times)
    """

    n = np.shape(data)[0]

    # calculate data eigenvectors
    # NOTE: m and evals not needed
    m, evals, evecs = asm(data)

    # projection in a lower dim space
    newdata = np.dot(evecs[:, :ps_dim].T, data.T).T

    # inizialise matrix
    D_PCA = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            D_PCA[i, j] = np.linalg.norm(newdata[i, :] - newdata[j, :], 2)
            D_PCA[j, i] = D_PCA[i, j]

    return D_PCA


