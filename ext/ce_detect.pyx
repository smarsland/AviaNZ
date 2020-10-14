cimport numpy as np
import numpy as np

cdef extern from "detector.h":
    int alg1(double xs[], size_t n, double sd, double penalty);

def launchDetector(infile):
    xl = []
    with open(infile) as f:
        for line in f:
            xl.extend([float(x) for x in line.strip().split(' ')])
    cdef np.ndarray xs = np.asarray(xl, dtype='float64')
    print("read %d datapoints" % len(xs))

    cdef double sd = 1
    cdef int n = len(xs)
    cdef double penalty = 1.1*3*np.log(n)
    return alg1(<double*> np.PyArray_DATA(xs), n, sd, penalty)
