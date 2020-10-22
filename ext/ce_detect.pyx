cimport numpy as np
import numpy as np

cdef extern from "detector.h":
    int alg1_mean(double xs[], size_t n, double sd, double penalty);
cdef extern from "detector.h":
    int alg1_var(double xs[], size_t n, double mu0, double penalty, int outstarts[], int outends[], char outtypes[]);
cdef extern from "detector.h":
    int alg2_var(double xs[], size_t nn, size_t maxlb, double sigma0, double penalty_s, double penalty_n, int outstarts[], int outends[], char outtypes[]);


def analyzeFile1(infile, startfrom=None, upto=None):
    """ For use in simulations: reads x from an external file
        and passes the selected range of them to launchDetector1. """
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

    res = launchDetector1(xs, alpha=3)
    return(res)

def analyzeFile2(infile, startfrom=None, upto=None):
    """ For use in simulations: reads x from an external file
        and passes the selected range of them to launchDetector2. """
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

    res = launchDetector2(xs, alpha=3)
    return(res)

def launchDetector1(xs, float alpha=3):
    xs = np.asarray(xs, dtype='float64')
    print("using %d datapoints" % len(xs))

    cdef double sd = 1
    cdef int n = len(xs)
    cdef double penalty = 1.1*alpha*np.log(n)
    cdef double mu0 = 0
    if n>10000:
        print("ERROR: n=%d exceeds max permitted series size" % n)
        return

    # outputs:
    cdef np.ndarray[int] outst = np.zeros(n, dtype=np.intc)
    cdef np.ndarray[int] oute = np.zeros(n, dtype=np.intc)
    cdef np.ndarray[np.uint8_t] outt = np.zeros(n, dtype='uint8')
    succ = alg1_var(<double*> np.PyArray_DATA(xs), n, mu0, penalty, <int*> np.PyArray_DATA(outst), <int*> np.PyArray_DATA(oute), <char*> np.PyArray_DATA(outt))
    if succ>0:
        print("ERROR: C detector failure")
        return
    # print(outt)
    outnum = outt.tolist().index(0)
    # print(outnum)
    # NOTE: a segment s:e is reported as a list [s,e+1,...]
    res = np.vstack((outst[:outnum], oute[:outnum]+1, outt[:outnum]))
    return(res.T)

def launchDetector2(xs, float alpha=3):
    xs = np.asarray(xs, dtype='float64')
    print("using %d datapoints" % len(xs))

    cdef double sigma2 = 1
    cdef int n = len(xs)
    cdef double penalty = 1.1*alpha*np.log(n)
    cdef double mu0 = 0
    cdef int maxlookback = int(0.15*len(xs))

    if n>10000:
        print("ERROR: n=%d exceeds max permitted series size" % n)
        return

    # outputs:
    cdef np.ndarray[int] outst = np.zeros(n, dtype=np.intc)
    cdef np.ndarray[int] oute = np.zeros(n, dtype=np.intc)
    cdef np.ndarray[np.uint8_t] outt = np.zeros(n, dtype='uint8')
    succ = alg2_var(<double*> np.PyArray_DATA(xs), n, maxlookback, sigma2, penalty, penalty, <int*> np.PyArray_DATA(outst), <int*> np.PyArray_DATA(oute), <char*> np.PyArray_DATA(outt))
    if succ>0:
        print("ERROR: C detector failure")
        return
    # print(outst)
    # print(oute)
    # print(outt)
    outnum = outt.tolist().index(0)
    # NOTE: a segment s:e is reported as a list [s,e+1,...]
    res = np.vstack((outst[:outnum], oute[:outnum]+1, outt[:outnum]))
    return(res.T)
