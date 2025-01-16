

import numpy as np
import pylab as pl
from numpy.fft import fft, ifft

def LPC(x, ncoeffs):
    dims = np.shape(x)
    nrows = dims[0]
    if len(dims) == 1:
        ncols = 1
        x = (x*np.ones((1,nrows))).T
    else:
        ncols = dims[1]
    
    # Autocorrelation
    #R = np.zeros((ncols,ncoeffs+1))
    R = np.real(ifft(np.abs(fft(x.T, n=len(x)) ** 2)))
    R = R[:,:ncoeffs+1]

    # Least-squares minimisation of a Toeplitz matrix
    # by Levinson-Durban recursion
    A = np.zeros((ncols,ncoeffs+1))
    A[:,0] = np.ones(ncols)
    #print(A)
    E = np.zeros(ncols)
    for col in range(ncols):
        # Crapper autocorrelation
        #for i in range(ncoeffs+1):
            #for j in range(nrows-i):
                #R[col,i] += x[j,col] * x[j+i,col] 
        
        e = R[col,0]

        for k in range(ncoeffs):
            l = 0
            for j in range(k+1):
                l -= A[col,j]*R[col,k+1-j]
            l /= e

            for n in range((k+1)//2 + 1):
                temp = A[col,k+1-n] + l*A[col,n]
                A[col,n] += l * A[col,k+1-n]
                A[col,k+1-n] = temp
            e *= 1 - l*l

        E[col] = e
    
    return A, E, R 

def test(ncoeffs=4):
    o = np.zeros(128)
    for i in range(1,129):
        o[i-1] = np.sin(0.01*i) + 0.75*np.sin(0.03*i) + 0.5*np.sin(0.05*i) + 0.25 * np.sin(0.11*i)

    #print(o)
    A, E, R = LPC(o,ncoeffs)
    #print(np.shape(A))

    p = np.zeros(128)
    for i in range(ncoeffs,len(o)):
        for j in range(ncoeffs):
            p[i] -= A[0,j+1] * o[i-1-j]

    e = 0
    for i in range(ncoeffs,len(o)):
        delta = o[i] - p[i]
        e += delta*delta
    print(e)
    print(np.shape(A))

    A = np.squeeze(A)
    roots = np.roots(A)
    roots = [r for r in roots if np.imag(r) >= 0]
    angles = np.arctan2(np.imag(roots), np.real(roots))

    freqs = []
    freqs.append(sorted(angles * (128 / (2 * np.pi))))
    print(freqs)

def test2(ncoeffs=4):
    o = np.zeros((128,2))
    for i in range(1,129):
        o[i-1,0] = np.sin(0.01*i) + 0.75*np.sin(0.03*i) + 0.5*np.sin(0.05*i) + 0.25 * np.sin(0.11*i)
        o[i-1,1] = np.sin(0.01*i) + 0.75*np.sin(0.03*i) + 0.5*np.sin(0.05*i) + 0.25 * np.sin(0.11*i)

    #print(o)
    A, E, R = LPC(o,ncoeffs)
    print(np.shape(A))
    print(A)

    p = np.zeros(128)
    for i in range(ncoeffs,len(o)):
        for j in range(ncoeffs):
            p[i] -= A[0,j+1] * o[i-1-j,0]

    e = 0
    for i in range(ncoeffs,len(o)):
        delta = o[i] - p[i]
        e += delta*delta
    print(e)

    pl.ion()
    pl.plot(o,'ro')
    pl.plot(p,'k+')
    pl.show()

#test(4)
#test2(4)
