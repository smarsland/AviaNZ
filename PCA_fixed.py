import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def resample(points,npoints):
    ncurves = np.shape(points)[0]
    newcurves = np.zeros((ncurves,npoints,2))
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
    return newcurves

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


def normalise(curves):
    npoints = np.shape(curves)[1]

    # Make them all unit length
    for i in range(np.shape(curves)[0]):
        curves[i, ::2] = curves[i, ::2] / curves[i, npoints - 2]

    # Scale the frequencies
    # mins = np.min(curves[:,1::2],axis=0)
    # maxs = np.max(curves[:,1::2],axis=0)
    curves[:, 1::2] = curves[:, 1::2] - np.min(curves[:, 1::2]) / (np.max(curves[:, 1::2]) - np.min(curves[:, 1::2]))
    return curves

def asm(curves):
    # curves = np.loadtxt(file,delimiter=' ')
    npoints = np.shape(curves)[1]
    print(npoints)
    mean = np.mean(curves, axis=0)
    curves = curves - mean

    # Compute the covariance matrix
    C = np.cov(curves.T)

    # Get the eigenvalues and eigenvectors
    evals, evecs = np.linalg.eig(C)

    # Now need to sort them into descending order
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:, indices]
    evals = evals[indices]
    evecs = np.real(evecs)
    evals = np.real(evals)
    # m = np.mean(curves,axis=0)
    # evecs = evecs.reshape((npoints,2,npoints))
    return mean, evals, evecs

# Oh dear this code is horrible, let's run it once and hope nobody notices
directory = "./extracted"

listridges = []
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith(".txt"):
            listridges.append(filename.replace('.txt', ''))
n = len(listridges)
print(n)

lengthmax = 0
for i in range(0,n):
    #curve = np.transpose(np.loadtxt(open(os.path.join(directory, listridges[i] + ".csv"), "rb"), delimiter=",", skiprows=1))
    curve = np.loadtxt(open(os.path.join(directory, listridges[i] + ".txt"), "rb"), delimiter=" ")
    #print(np.shape(curve))
    if np.shape(curve)[1] > lengthmax:
        lengthmax = np.shape(curve)[1]

curves = np.zeros((n, lengthmax,2))
for i in range(0,n):
    #file = np.loadtxt(open(os.path.join(directory, listridges[i] + ".csv"), "rb"), delimiter=",", skiprows=1)
    file = np.transpose(np.loadtxt(open(os.path.join(directory, listridges[i] + ".txt"), "rb"), delimiter=" "))
    maxl = min(lengthmax,np.shape(file)[0])
    curves[i,:maxl,:] = file[:maxl,:]

#getting good curves
newcurves = resample(curves,53)
newcurves = procrustes_align(newcurves)
newcurves = newcurves.reshape(np.shape(newcurves)[0],np.shape(newcurves)[1]*2)

print(np.shape(newcurves))
#np.savetxt(os.path.join("Results","scalogramcurvesforpca1.txt"),newcurves)

#Scree
freqs = newcurves[:,1::2]
times = newcurves[:,0]
m, evals, evecs = asm(freqs)
m1, evals1, evecs1 = asm(newcurves)
#pl.plot(np.arange(1,11),np.cumsum(evals)[:10])
plt.plot(np.arange(1,11),evals1[:10])

#PCA
npc = 5
newdata = np.dot(evecs[:,:npc].T,freqs.T).T
y=np.transpose(np.dot(evecs[:,:npc],newdata.T))+m
newdata1 = np.dot(evecs1[:,:npc].T,newcurves.T).T
y1=np.transpose(np.dot(evecs1[:,:npc],newdata1.T))+m1
print(np.shape(newdata1))

distances = np.zeros((1002,1002))
for i in range(0,1002):
    for j in range(i+1,1002):
        distances[i,j] = np.linalg.norm(newdata1[i,:]-newdata1[j,:],2)
        distances[j,i] = distances[i,j]

#np.savetxt("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\"+"PCAscalogramdistancesfixed.txt",distances)
#plt.imshow(np.log(distances))
#plt.savefig("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\"+"PCAscalogramscreefixed.jpg")

# for 3d
fig = plt.figure()
ax = plt.axes(projection='3d')
n1 = np.loadtxt(os.path.join("Results","spectrogramridges.txt"), dtype="str")
for i in range(0,1002):

    if n1[i].startswith('01'):
        plt.plot(newdata1[i, 0], newdata1[i, 1], newdata1[i, 2],'.', color="blue")
    if n1[i].startswith('02'):
            plt.plot(newdata1[i, 0], newdata1[i, 1], newdata1[i, 2], '.', color="red")
    if n1[i].startswith('03'):
            plt.plot(newdata1[i, 0], newdata1[i, 1], newdata1[i, 2], '.', color="green")
    if n1[i].startswith('04'):
            plt.plot(newdata1[i, 0], newdata1[i, 1], newdata1[i, 2], '.', color="orange")
    if n1[i].startswith('05'):
        plt.plot(newdata1[i, 0], newdata1[i, 1], newdata1[i, 2],'.', color="purple")
    if n1[i].startswith('06'):
            plt.plot(newdata1[i, 0], newdata1[i, 1], newdata1[i, 2], '.', color="darkblue")
    if n1[i].startswith('07'):
            plt.plot(newdata1[i, 0], newdata1[i, 1], newdata1[i, 2], '.', color="teal")

#plt.savefig("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\"+"PCAspectrogramprojection3dfixed.jpg")
