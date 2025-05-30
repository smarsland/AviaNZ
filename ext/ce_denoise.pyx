cimport numpy as np
import numpy as np

import time

cdef extern from "math.h":
        double log(double arg)

cdef extern from "ce_functions.h":
        double ce_getcost(double *in_array, int size, double threshold, char costfn, int step)

cdef extern from "ce_functions.h":
        double ce_thresnode(double *in_array, double *out_array, int size, double threshold, char type)

cdef extern from "ce_functions.h":
        int ce_thresnode2(double *in_array, int size, double threshold, int type)

cdef extern from "ce_functions.h":
        int ce_thresnode2_block(double *in_array, int size, int blocklen, double *threshold, int type)

cdef extern from "ce_functions.h":
        void ce_energycurve(double *arrE, double *arrC, int N, int M)

cdef extern from "ce_functions.h":
        void ce_sumsquares(double *arr, const size_t arrs, const int W, double *besttau, const double thr)

cdef extern from "ce_functions.h":
        int upsampling_convolution_valid_sf(const double * const input, const size_t N,
                const double * const filter, const size_t F,
                double * const output, const size_t O)


# Simplified caller to the cost calculator. Useful for testing purposes
def JustCost(np.ndarray array, threshold, costfn):
         if array.dtype != 'float64':
                 array = array.astype('float64')
         if costfn == 'threshold':
                 # Threshold
                 cost = ce_getcost(<double*> np.PyArray_DATA(array), array.shape[0], threshold, 't', 1)
         elif costfn == 'entropy':
                 # Entropy
                 cost = ce_getcost(<double*> np.PyArray_DATA(array), array.shape[0], threshold, 'e', 1)
         else:
                 cost = ce_getcost(<double*> np.PyArray_DATA(array), array.shape[0], threshold, '*', 1)

         return(cost)


def BestTree2(wp,threshold,costfn='threshold'):
        """ Compute the best wavelet tree using one of three cost functions: threshold, entropy, or SURE.
        Scores each node and uses those scores to identify new leaves of the tree by working up the tree.
        Returns the list of new leaves of the tree.

        This version works on our custom WPs (ndarrays), not pywt trees.
        """
        nnodes = len(wp)
        cost = np.zeros(nnodes)
        count = 0
        step = 1
        opstartingtime = time.time()

        # Get costs. Thr cost is +1 for each WC that exceeds t, Entr cost is -p*logp
        for n in range(nnodes):
                node = wp[n]
                if node.dtype != 'float64':
                        node = node.astype('float64')
                # downsample non-root nodes to keep compatible w/ pywt WCs:
                if n!=0:
                        step = 2
                if costfn == 'threshold':
                        # Threshold
                        cost[count] = ce_getcost(<double*> np.PyArray_DATA(node), node.shape[0], threshold, 't', step)
                elif costfn == 'entropy':
                        # Entropy
                        cost[count] = ce_getcost(<double*> np.PyArray_DATA(node), node.shape[0], threshold, 'e', step)
                else:
                        cost[count] = ce_getcost(<double*> np.PyArray_DATA(node), node.shape[0], threshold, '*', step)

                count += 1
        print("Best basis selected in %.5f s" % (time.time() - opstartingtime))

        # Compute the best tree using those cost values

        flags = np.zeros(nnodes)  # initiate w/ 2 for all nodes
        flags[ nnodes//2 :] = 1  # leaves. nnodes//2 starts from 2^maxlevel-1

        # Work up the tree from just above leaves
        inds = np.arange(nnodes//2)    # for every ind except leaves
        inds = inds[-1::-1]    # reverse
        for i in inds:
                # Get children of this ind
                children = np.array([2*i+1, 2*i+2])
                # Get total cost of both children
                c = cost[children[0]] + cost[children[1]]
                if c < cost[i]:
                        # if children more negative (less entropy), will decompose
                        # flags are set to decompose by default,
                        # so just update parent's cost:
                        cost[i] = c
                else:
                        # if children have more entropy, will keep parent:
                        flags[i] = 1
                        flags[children] = 0

        # Now get the new leaves of the tree. Anything below these nodes is deleted.
        # Go through the tree from the top, and stop at "no-decompose" nodes:
        def decompose(n):
            if flags[n]==1:
                # threshold says to keep parent
                return [n]
            else:
                # Get children of this ind
                child1 = 2*n+1
                child2 = 2*n+2
                if(child2 >= nnodes):
                    # nowhere to decompose, return parent
                    return [n]
                else:
                    # collect best node IDs from children
                    return np.concatenate((decompose(child1), decompose(child2)))
        newleaves = decompose(0)

        return newleaves


# simplified version of ThresholdNodes for testing.
def JustThreshold(self, np.ndarray indata, threshold, type):
        # then keep & threshold (inplace)
        indata = np.ascontiguousarray(indata)
        length = indata.shape[0]

        if(type=='hard'):
                ce_thresnode2(<double*> np.PyArray_DATA(indata), length, threshold, 'h')
        else:
                ce_thresnode2(<double*> np.PyArray_DATA(indata), length, threshold, 's')

        # note: no return value b/c indata is edited inplace.
        return indata


def ThresholdNodes2(list oldtree, list bestleaves, threshold, str thrtype, int blocklen=0):
    """ Thresholds nodes of our ndarray-type WPs
        Uses inplace thresholding, so use with care! (i.e. arg oldtree will be overwritten)
        Args:
        1. oldtree - custom WP, a list of 2^(J+1)-1 ndarrays
        2. bestleaves - list of N nodes to be thresholded
        3. threshold - if int or ndarray, will apply this thr over all nodes & times.
            Otherwise if Nx1 ndarray, thresholding will be node-specific, but constant over time.
            Otherwise if NxT ndarray, thresholding will be node- and time-specific (for each of T blocks).
            IMPORTANT: assumes that row i matches SORTED bestleaves[i]!
        4. blocklen - int, in samples. Required if threshold is NxT. T*blocklen must be greater or equal to datalength.
    """
    bestleavesset = set(bestleaves)
    N = len(bestleavesset)
    if list(bestleavesset) != bestleaves:
        print("Warning: best leaves were not sorted, make sure threshold order is the same")
        # could return an error

    # Input checks
    if blocklen!=0:
        # Will split data into T time blocks.
        # TODO: using floor division, so last sub-block piece will be denoised
        # with some random threshold. This matters very little as the blocks
        # are small and usually divide the datalen perfectly, but should
        # fix at some point.
        T = len(oldtree[0]) // blocklen
        if T<1:
            print("ERROR: data shorter than the block size")
            return 1
    else:
        # will keep data in a single block
        T = 1

    if np.ndim(threshold)==0:
        threshold = threshold * np.ones((N,T))
        print("Applying constant threshold over nodes and time")
    elif type(threshold) is np.ndarray:
        # checking for both 1D and 2D arrays to allow simple scripts outside for prep
        if np.shape(threshold)==(1,) or np.shape(threshold)==(1,1):
            threshold = threshold * np.ones((N,T))
            print("Applying constant threshold over nodes and time")
        elif np.shape(threshold)==(N,) or np.shape(threshold)==(N,1):
            print(np.shape(threshold))
            threshold = np.transpose(threshold * np.ones((T,N)))
            print(np.shape(threshold))
            print("Applying node-specific, time-constant threshold")
        elif np.shape(threshold)==(N,T):
            if blocklen==0:
                print("ERROR: blocklen must be provided for NxT thresholding")
                return 1
            else:
                print("will use blocks of", blocklen, "samples")
            print("Applying node- and time-specific threshold over blocks of %d samples" % blocklen)
        else:
            print("ERROR: threshold shape %d x %d unrecognized" % (N, T))
            return 1
    else:
        print("ERROR: wrong type of threshold provided")
        return 1

    # a dumb check, but important
    if np.shape(threshold)!=(N,T):
        print("ERROR: something went wrong in denoising")
        print(np.shape(threshold))
        return 1

    thrtype_ce = -1
    if thrtype=="soft":
        thrtype_ce = 1
    elif thrtype=="hard":
        thrtype_ce = 2
    else:
        print("ERROR: type of threshold not recognized")
        return 1

    # Main loop
    #print("Bestleaves", bestleaves)
    #print("thresholds", threshold[:,0])
    for node in range(len(oldtree)):
        if node in bestleavesset:
            # then keep & threshold (inplace)
            length = oldtree[node].shape[0]
            oldtree[node] = np.ascontiguousarray(oldtree[node])
            nodeix = list(bestleavesset).index(node)
            if blocklen==0:
                ce_thresnode2(<double*> np.PyArray_DATA(oldtree[node]), length, threshold[nodeix,0], thrtype_ce)
            else:
                # adjust blocklength for the wavelet downsampling
                # (assuming last level is not downsampled)
                if node>0:
                    nodelvl = np.floor(np.log2(node+1))
                    blocklen_adj = blocklen // 2**(nodelvl-1)
                thresarray = np.ascontiguousarray(threshold[nodeix,:])
                # ce_thresnode2(<double*> np.PyArray_DATA(oldtree[node]), length, threshold[nodeix,0], thrtype_ce)
                ce_thresnode2_block(<double*> np.PyArray_DATA(oldtree[node]), length, blocklen_adj, <double*> np.PyArray_DATA(thresarray), thrtype_ce)
        else:
            # zero-out all the other nodes
            # NOT USED because current reconstruction already assumes all other nodes are 0.
            oldtree[node] = np.zeros(len(oldtree[node]))

    # note: no useful return b/c oldtree is edited inplace.
    return 0

def EnergyCurve(np.ndarray C, M):
        assert C.dtype==np.float64
        assert len(C)>2*M+1
        # Args: 1. wav data 2. M (int), expansion in samples
        N = len(C)
        E = np.zeros(N)
        E[M] = np.sum(C[:2*M+1])
        ce_energycurve(<double*> np.PyArray_DATA(E), <double*> np.PyArray_DATA(C), N, M)
        return E

def FundFreqYin(np.ndarray data, int W, double thr, double fs):
        assert data.dtype==np.float64
        assert thr>0
        cdef int arrs = len(data)
        starts = range(0, len(data) - 2*W, W//2)
        assert len(starts)>0
        pitch = np.zeros(len(starts) + 1)
        besttau = -1 * np.ones(len(starts))

        # Compute sum of squared diff (autocorrelation)
        # C code will return array of best tau for each window start
        ce_sumsquares(<double*> np.PyArray_DATA(data), arrs, W, <double*> np.PyArray_DATA(besttau), thr)

        for i in range(len(starts)):
            # -1 is an error code for no ff found / correlation too weak / numeric error
            if besttau[i] == -1:
                pitch[i] = -1
            else:
                pitch[i] = float(fs)/besttau[i]
        return pitch

def reconstruct(np.ndarray data, int node, np.ndarray wv_rec_hi, np.ndarray wv_rec_lo, int lvl):
    assert data.dtype==np.float64
    print(len(data), lvl, node) 
    cdef np.ndarray datau = np.zeros(int(2**(lvl-1) * len(data)), dtype=np.float64)
    cdef int datau_len, wv_hi_len, wv_lo_len, data_len

    if lvl==0:
        print("Warning: reconstruction from level 0 requested")
        return data
    elif lvl<0:
        print("ERROR: suggested level %d < 0" % lvl)
        return

    wv_hi_len = len(wv_rec_hi)
    wv_lo_len = len(wv_rec_lo)

    # Jth level doesn't need upsampling b/c WCs were non-downsampled on the last level
    if node % 2 == 0:
        data = np.convolve(data, wv_rec_hi, 'same')
    else:
        data = np.convolve(data, wv_rec_lo, 'same')
    # cut ends because the WP process produces too long WC vectors
    # data = data[wv_hi_len//2 : -wv_lo_len//2]

    # and then proceed with standard upsampling o convolution, J-1 times
    node = (node - 1)//2
    lvl = lvl - 1
    extlen = (wv_hi_len+wv_lo_len)//8

    while lvl != 0:
        # extend the ends symmetrically
        data = np.concatenate((data[extlen::-1], data, data[-1:-extlen:-1]))

        # allocate mem for upsampled output
        data_len = len(data)
        if node % 2 == 0:
            datau_len = 2*data_len - wv_hi_len + 2
        else:
            datau_len = 2*data_len - wv_lo_len + 2
        datau = np.zeros(datau_len, dtype=np.float64)
        
        # pray to gods all arrays are C_CONTIGUOUS
        # and upsample o convolve:
        if node % 2 == 0:
            c_exit_code = upsampling_convolution_valid_sf(<double*> np.PyArray_DATA(data), data_len,
                <double*> np.PyArray_DATA(wv_rec_hi), wv_hi_len,
                <double*> np.PyArray_DATA(datau), datau_len)
        else:
            c_exit_code = upsampling_convolution_valid_sf(<double*> np.PyArray_DATA(data), data_len,
                <double*> np.PyArray_DATA(wv_rec_lo), wv_lo_len,
                <double*> np.PyArray_DATA(datau), datau_len)

        if c_exit_code!=0:
            print("ERROR: Cythonized convolution failed")
            return
        
        # move output back on top of input for next loop
        data = datau

        #data = data[wv_hi_len//2-1 : -(wv_lo_len//2-1)]

        node = (node - 1)//2
        lvl = lvl - 1

    return data
