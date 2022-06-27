import numpy as np
# Resampling, Procrustes alignment, basic ASM/Eigenshape

def resample(points,npoints):
    ncurves = np.shape(points)[0]
    newcurves = np.zeros((ncurves,npoints,2))
#    newcurves = np.zeros((ncurves,2*npoints-2,2))
    for i in range(ncurves):
        ind = np.max(np.where(points[i,:,0] != 0)[0])
        p = np.squeeze(points[i,:ind+1,:])
        dp = np.diff(p,axis=0)
        pts = np.zeros(len(dp)+1)
        pts[1:] = np.cumsum(np.sqrt(np.sum(dp*dp,axis=1)))
        newpts = np.linspace(0,pts[-1],npoints)
        newcurves[i,:,0] = np.interp(newpts,pts,p[:,0])
        newcurves[i,:,1] = np.interp(newpts,pts,p[:,1])
        newcurves[i,:npoints,:] = np.copy(newcurves[i,:,:]) - newcurves[i,0,:]
        # extra = np.squeeze(np.copy(newcurves[i,-2:0:-1,:])- newcurves[i,0,:])
        # extra[:,0] = -extra[:,0]
        # newcurves[i,npoints:,:] = np.copy(extra)
    return newcurves, newcurves

# Procrustes alignment 
from scipy.spatial import procrustes

def procrustes_align(curves):
    ncurves = np.shape(curves)[0]
    newcurves = np.zeros(np.shape(curves))
    #disparity = np.zeros((ncurves,ncurves))
    for j in range(ncurves):
        #mtx1, newcurves[i+1,:,:], disparity[i,j] = procrustes(curves[i,:,:],curves[j,:,:])
        mtx1, newcurves[j,:,:], _ = procrustes(curves[0,:,:],curves[j,:,:])
    newcurves[0,:,:] = mtx1
    # Scale to -1:1
    newcurves[:,:,0] /= np.max(newcurves[:,:,0])
    newcurves[:,:,1] /= np.max(newcurves[:,:,1])
    return newcurves

# Simple Active Shape Model (Eigenfaces)

def asm(file):
    curves = np.loadtxt(file,delimiter=' ')
    npoints = np.shape(curves)[1]
    # Compute the covariance matrix
    C = np.cov(curves.T)

    # Get the eigenvalues and eigenvectors
    evals,evecs = np.linalg.eig(C)

    # Now need to sort them into descending order
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]
    evecs = np.real(evecs)  
    evals = np.real(evals)  
    m = np.mean(curves,axis=0)
    #evecs = evecs.reshape((npoints,2,npoints))
    return m, evals, evecs

# def asm(file):
#     curves = np.loadtxt(file,delimiter='/')
#     npoints = np.shape(curves)[1]//2
#     # Compute the covariance matrix
#     C = np.cov(curves.T)
#
#     # Get the eigenvalues and eigenvectors
#     evals,evecs = np.linalg.eig(C)
#
#     # Now need to sort them into descending order
#     indices = np.argsort(evals)
#     indices = indices[::-1]
#     evecs = evecs[:,indices]
#     evals = evals[indices]
#     evecs = np.real(evecs)
#     evals = np.real(evals)
#     m = np.mean(curves,axis=0).reshape((npoints,2))
#     evecs = evecs.reshape((npoints,2,npoints*2))
#     return m, evals, evecs

