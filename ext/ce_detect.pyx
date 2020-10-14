cimport numpy as np
import numpy as np

cdef extern from "detector.h":
    int alg1_mean(double xs[], size_t n, double sd, double penalty);
cdef extern from "detector.h":
    int alg1_var(double xs[], size_t n, double mu0, double penalty, int outstarts[], int outends[], char outtypes[]);
cdef extern from "detector.h":
    int alg2_var(double xs[], size_t nn, size_t maxlb, double sigma0, double penalty_s, double penalty_n, int outstarts[], int outends[], char outtypes[]);

def launchDetector1(infile, startfrom=None, upto=None):
    xl = []
    with open(infile) as f:
        for line in f:
            xl.extend([float(x) for x in line.strip().split(' ')])
    cdef np.ndarray xs = np.asarray(xl, dtype='float64')
    print("read %d datapoints" % len(xs))
    if startfrom is not None:
        xs = xs[startfrom:]
    if upto is not None:
        xs = xs[:(upto+1)]
    print("using %d datapoints" % len(xs))

    cdef double sd = 1
    cdef int n = len(xs)
    cdef double penalty = 1.1*3*np.log(n)
    cdef double mu0 = 0

    # outputs:
    cdef np.ndarray[int] outst = np.zeros(n, dtype=np.intc)
    cdef np.ndarray[int] oute = np.zeros(n, dtype=np.intc)
    cdef np.ndarray[np.uint8_t] outt = np.zeros(n, dtype='uint8')
    alg1_var(<double*> np.PyArray_DATA(xs), n, mu0, penalty, <int*> np.PyArray_DATA(outst), <int*> np.PyArray_DATA(oute), <char*> np.PyArray_DATA(outt))
    print(outt)
    outnum = outt.tolist().index(0)
    print(outnum)
    res = np.vstack((outst[:outnum], oute[:outnum], [chr(t) for t in outt[:outnum]]))
    return(res.T)

def launchDetector(infile, startfrom=None, upto=None):
    xl = []
    with open(infile) as f:
        for line in f:
            xl.extend([float(x) for x in line.strip().split(' ')])
    cdef np.ndarray xs = np.asarray(xl, dtype='float64')
    print("read %d datapoints" % len(xs))
    if startfrom is not None:
        xs = xs[startfrom:]
    if upto is not None:
        xs = xs[:(upto+1)]
    print("using %d datapoints" % len(xs))

    cdef double sigma2 = 1
    cdef int n = len(xs)
    cdef double penalty = 1.1*3*np.log(n)
    cdef double mu0 = 0
    cdef int maxlookback = int(0.3*len(xs))

    # outputs:
    cdef np.ndarray[int] outst = np.zeros(n, dtype=np.intc)
    cdef np.ndarray[int] oute = np.zeros(n, dtype=np.intc)
    cdef np.ndarray[np.uint8_t] outt = np.zeros(n, dtype='uint8')
    succ = alg2_var(<double*> np.PyArray_DATA(xs), n, maxlookback, sigma2, penalty, penalty, <int*> np.PyArray_DATA(outst), <int*> np.PyArray_DATA(oute), <char*> np.PyArray_DATA(outt))
    if succ>0:
        print("ERROR: C detector failure")
        return
    outnum = outt.tolist().index(0)
    res = np.vstack((outst[:outnum], oute[:outnum], [chr(t) for t in outt[:outnum]]))
    return(res.T)

def compareBoth(infile):
    r1 = launchDetector1(infile)
    r2 = launchDetector(infile)
    print("Alg1:")
    print(r1)
    print("Alg2:")
    print(r2)
