cimport numpy as np
import numpy as np

cdef extern from "detector.h":
    int alg1_mean(double xs[], size_t n, double sd, double penalty);
cdef extern from "detector.h":
    int alg1_var(double xs[], size_t n, size_t maxlb, double mu0, double penalty, int outstarts[], int outends[], char outtypes[]);
cdef extern from "detector.h":
    int alg2_var(double xs[], size_t nn, size_t maxlb, double sigma0, double penalty_s, double penalty_n, int outstarts[], int outends[], char outtypes[], size_t printing);


def launchDetector1(xs, int maxlookback, float alpha):
    xs = np.asarray(xs, dtype='float64')
    print("using %d datapoints" % len(xs))

    cdef double sd = 1
    cdef int n = len(xs)
    cdef double penalty = 1.1*alpha*np.log(n)
    cdef double mu0 = 0

    assert np.min(xs)>0, "ERROR: all datapoints must be strictly > 0 for variance detection"
    if n>10000:
        print("ERROR: n=%d exceeds max permitted series size" % n)
        return

    # outputs:
    cdef np.ndarray[int] outst = np.zeros(n, dtype=np.intc)
    cdef np.ndarray[int] oute = np.zeros(n, dtype=np.intc)
    cdef np.ndarray[np.uint8_t] outt = np.zeros(n, dtype='uint8')
    succ = alg1_var(<double*> np.PyArray_DATA(xs), n, maxlookback, mu0, penalty, <int*> np.PyArray_DATA(outst), <int*> np.PyArray_DATA(oute), <char*> np.PyArray_DATA(outt))
    if succ>0:
        print("ERROR: C detector failure")
        return
    # print(outt)
    outnum = outt.tolist().index(0)
    # print(outnum)
    # NOTE: detector.c outputs 0-indexed window positions, inclusive.
    # This conversion here makes the interval [s,e).
    # true timestamp can be obtained simply by multiply it by window size
    res = np.vstack((outst[:outnum], oute[:outnum]+1, outt[:outnum]))
    return(res.T)

def launchDetector2(xs, float sigma2, int maxlookback, float alpha, int printing=1):
    # printing=1 means that more details will be printed, 0="silent"
    xs = np.asarray(xs, dtype='float64')
    print("using %d datapoints" % len(xs))

    # Type conversion, not sure if needed
    cdef double csigma2 = sigma2
    cdef int n = len(xs)
    cdef double penalty = 1.1*alpha*np.log(n)

    assert np.min(xs)>0, "ERROR: all datapoints must be strictly > 0 for variance detection"
    if n>10000:
        print("ERROR: n=%d exceeds max permitted series size" % n)
        return

    # outputs:
    cdef np.ndarray[int] outst = np.zeros(n, dtype=np.intc)
    cdef np.ndarray[int] oute = np.zeros(n, dtype=np.intc)
    cdef np.ndarray[np.uint8_t] outt = np.zeros(n, dtype='uint8')
    succ = alg2_var(<double*> np.PyArray_DATA(xs), n, maxlookback, csigma2, penalty, penalty, <int*> np.PyArray_DATA(outst), <int*> np.PyArray_DATA(oute), <char*> np.PyArray_DATA(outt), printing)
    if succ>0:
        print("ERROR: C detector failure")
        return
    # print(outst)
    # print(oute)
    # print(outt)
    outnum = outt.tolist().index(0)
    # NOTE: detector.c outputs 0-indexed window positions, inclusive.
    # This conversion here makes the interval [s,e).
    # true timestamp can be obtained simply by multiply it by window size
    res = np.vstack((outst[:outnum], oute[:outnum]+1, outt[:outnum]))
    return(res.T)
