cimport numpy as np
import numpy as np

cdef extern from "detector.h":
    int alg1_mean(double xs[], size_t n, double sd, double penalty);
cdef extern from "detector.h":
    int alg1_var(double xs[], size_t n, double mu0, double penalty);
cdef extern from "detector.h":
    int alg2_var(double xs[], size_t nn, double sigma0, double penalty_s, double penalty_n);

def launchDetector1(infile):
    xl = []
    with open(infile) as f:
        for line in f:
            xl.extend([float(x) for x in line.strip().split(' ')])
    cdef np.ndarray xs = np.asarray(xl, dtype='float64')
    print("read %d datapoints" % len(xs))

    cdef double sd = 1
    cdef int n = len(xs)
    cdef double penalty = 1.1*3*np.log(n)
    cdef double mu0 = 0
    return alg1_var(<double*> np.PyArray_DATA(xs), n, mu0, penalty)

def launchDetector(infile):
    xl = []
    with open(infile) as f:
        for line in f:
            xl.extend([float(x) for x in line.strip().split(' ')])
    cdef np.ndarray xs = np.asarray(xl, dtype='float64')
    print("read %d datapoints" % len(xs))

    cdef double sigma2 = 1
    cdef int n = len(xs)
    cdef double penalty = 1.1*3*np.log(n)
    cdef double mu0 = 0
    return alg2_var(<double*> np.PyArray_DATA(xs), n, sigma2, penalty, penalty)
