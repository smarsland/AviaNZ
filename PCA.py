import os
import numpy as np
import matplotlib.pyplot as plt
import Linear
from mpl_toolkits import mplot3d

directory = "./extracted"
#directory = "C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\extracted"

listridges = []
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith(".csv"):
            listridges.append(filename.replace('.csv', ''))
n = len(listridges)
print(n)

lengthmax = 0
for i in range(0,n):
    curve = np.transpose(np.loadtxt(open(os.path.join(directory, listridges[i] + ".csv"), "rb"), delimiter=",", skiprows=1))
    #print(np.shape(curve))
    if np.shape(curve)[1] > lengthmax:
        lengthmax = np.shape(curve)[1]

#print(lengthmax)

curves = np.zeros((n, lengthmax,2))
for i in range(0,n):
    file = np.loadtxt(open(os.path.join(directory, listridges[i] + ".csv"), "rb"), delimiter=",", skiprows=1)
    maxl = min(lengthmax,np.shape(file)[0])
    curves[i,:maxl,:] = file[:maxl,:]
    #print(curves[i,:,:])

rescurves = Linear.resample(curves,lengthmax)[0]
#procurves = Linear.procrustes_align(rescurves)
# ???? !!!!
procurves = rescurves.copy()
procurves = procurves.reshape(n,lengthmax*2)
np.savetxt(os.path.join("Results","spectrogramcurvesforpca.txt"),procurves)

#np.savetxt("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\"+"spectrogramcurvesforpca.txt", procurves[:,:,1])

data = os.path.join("Results","spectrogramcurvesforpca.txt")

m, _, _= Linear.asm(data)
#C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\"+"spectrogramcurvesforpca.txt")
curves = np.loadtxt(data,delimiter=' ')
n1 = np.loadtxt(os.path.join("Results","spectrogramridges.txt"), dtype="str")
#"C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\spectrogramridges.txt", dtype='str')

#print(curves[1,:])
for i in range(0,1002):
    curves[i,:] = curves[i,:]-m[:].T

np.savetxt(os.path.join("Results","spectrogramcurvesforpcaminusmean.txt"),curves)


plt.plot(curves[230,0::2],curves[230,1::2])

#np.savetxt("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\"+"spectrogramcurvesforpcaminusmean.txt",curves)
data = os.path.join("Results","spectrogramcurvesforpcaminusmean.txt")
m, evals, evecs = Linear.asm(data)
b = np.dot(curves.T,evecs[:,:10].T).T
#b = curves @ evecs[:,:10]

#y=np.transpose(np.dot(evecs[:,:10],b.T))+m

#distances = np.zeros((1002,1002))
#for i in range(0,1002):
    #for j in range(i+1,1002):
        #distances[i,j] = np.linalg.norm(b[i,:]-b[j,:],2)
        #distances[j,i] = distances[i,j]

#np.savetxt("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\"+"PCAscalogramdistances.txt",distances)
#plt.imshow(np.log(distances))
#plt.savefig("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\"+"PCAscalogramdistancematrix.jpg")

# for 3d
# fig = plt.figure()
# ax = plt.axes(projection='3d')
"""
for i in range(0,1002):
        if n1[i].startswith('01'):
            plt.plot(b[i, 1], b[i, 2],'.', color="blue")
        if n1[i].startswith('02'):
                plt.plot(b[i, 1], b[i, 2], '.', color="red")
        if n1[i].startswith('03'):
                plt.plot(b[i, 1], b[i, 2], '.', color="green")
        if n1[i].startswith('04'):
                plt.plot(b[i, 1], b[i, 2], '.', color="orange")
        if n1[i].startswith('05'):
            plt.plot(b[i, 1], b[i, 2], '.', color="purple")
        if n1[i].startswith('06'):
                plt.plot(b[i, 1], b[i, 2], '.', color="darkblue")
        if n1[i].startswith('07'):
                plt.plot(b[i, 1], b[i, 2], '.', color="teal")
"""
#plt.savefig("C:\\Users\\Harvey\\Dropbox\\Kiwi_IndividualID\\Results\\"+"PCAspectrogramprojection3.jpg")
