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
        double ce_thresnode2(double *in_array, int size, double threshold, char type)

cdef extern from "ce_functions.h":
        void ce_energycurve(double *arrE, double *arrC, int N, int M)

cdef extern from "ce_functions.h":
        void ce_sumsquares(double *arr, int W, double *out)

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


def BestTree(wp,threshold,costfn='threshold'):
        """ Compute the best wavelet tree using one of three cost functions: threshold, entropy, or SURE.
        Scores each node and uses those scores to identify new leaves of the tree by working up the tree.
        Returns the list of new leaves of the tree.

        Rewritten in Cython 27/07/18 by JJ.
        """
        nnodes = 2 ** (wp.maxlevel + 1) - 1
        cost = np.zeros(nnodes)
        count = 0
        opstartingtime = time.time()

        print("Checkpoint 1ba, %.5f" % (time.time() - opstartingtime))
        for level in range(wp.maxlevel + 1):
                for n in wp.get_level(level, 'natural'):
                        # Might be needed, tho pywt always outputs float, hopefully:
                        # if n.data.dtype != 'float64':
                        #        n.data = n.data.astype('float64')
                        if costfn == 'threshold':
                                # Threshold
                                cost[count] = ce_getcost(<double*> np.PyArray_DATA(n.data), n.data.shape[0], threshold, 't', 1)
                        elif costfn == 'entropy':
                                # Entropy
                                cost[count] = ce_getcost(<double*> np.PyArray_DATA(n.data), n.data.shape[0], threshold, 'e', 1)
                        else:
                                cost[count] = ce_getcost(<double*> np.PyArray_DATA(n.data), n.data.shape[0], threshold, '*', 1)

                        count += 1
        print("Checkpoint 1bb, %.5f" % (time.time() - opstartingtime))

        # Compute the best tree using those cost values
        flags = 2 * np.ones(nnodes)
        flags[2 ** wp.maxlevel - 1:] = 1
        # Work up the tree from just above leaves
        inds = np.arange(2 ** wp.maxlevel - 1)
        inds = inds[-1::-1]
        for i in inds:
                # Get children
                children = (i + 1) * 2 + np.arange(2) - 1
                c = cost[children[0]] + cost[children[1]]
                if c < cost[i]:
                        cost[i] = c
                        flags[i] = 2
                else:
                        flags[i] = flags[children[0]] + 2
                        flags[children] = -flags[children]

        # Now get the new leaves of the tree. Anything below these nodes is deleted.
        newleaves = np.where(flags > 2)[0]

        # Make a list of the children of the newleaves, and recursively their children
        def getchildren(n):
                level = int(np.floor(np.log2(n + 1)))
                if level < wp.maxlevel:
                        tbd.append((n + 1) * 2 - 1)
                        tbd.append((n + 1) * 2)
                        getchildren((n + 1) * 2 - 1)
                        getchildren((n + 1) * 2)

        tbd = []
        for i in newleaves:
                getchildren(i)

        tbd = np.unique(tbd)

        # I wasn't happy that these were being deleted, so am going the other way round
        listnodes = np.arange(2 ** (wp.maxlevel + 1) - 1)
        listnodes = np.delete(listnodes, tbd)
        notleaves = np.intersect1d(newleaves, tbd)
        for i in notleaves:
                newleaves = np.delete(newleaves, np.where(newleaves == i))

        listleaves = np.intersect1d(np.arange(2 ** (wp.maxlevel) - 1, 2 ** (wp.maxlevel + 1) - 1), listnodes)
        listleaves = np.unique(np.concatenate((listleaves, newleaves)))

        return listleaves

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

        print("Checkpoint 1ba, %.5f" % (time.time() - opstartingtime))
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
        print("Checkpoint 1bb, %.5f" % (time.time() - opstartingtime))

        # Compute the best tree using those cost values

        flags = 2 * np.ones(nnodes)  # initiate w/ 2 for all nodes
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
                        # if children more negative (less entropy), will keep children:
                        cost[i] = c     # its cost will be that of children
                        flags[i] = 2    # changes nothing???
                else:
                        # if children have more entropy:
                        flags[i] = flags[children[0]] + 2  # flag > 2: keep
                        flags[children] = -flags[children] # can be deleted

        # Now get the new leaves of the tree. Anything below these nodes is deleted.
        newleaves = np.where(flags > 2)[0] # does not include current leaves (flags=2)

        # Make a list of the children of the newleaves, and recursively their children
        def getchildren(n):
                level = int(np.floor(np.log2(n + 1)))
                if level < maxlevel:
                        tbd.append((n + 1) * 2 - 1)
                        tbd.append((n + 1) * 2)
                        getchildren((n + 1) * 2 - 1)
                        getchildren((n + 1) * 2)

        tbd = []
        maxlevel = int(np.floor(np.log2(nnodes+1)))
        for i in newleaves:
                getchildren(i)  # find all nodes below TOKEEP nodes

        tbd = np.unique(tbd)

        # I wasn't happy that these were being deleted, so am going the other way round
        listnodes = np.arange(nnodes)
        listnodes = np.delete(listnodes, tbd)  # delete all nodes below TOKEEP nodes

        notleaves = np.intersect1d(newleaves, tbd) # ??? how is this not empty?
        for i in notleaves:
                newleaves = np.delete(newleaves, np.where(newleaves == i)) # so delete notleaves?
                # so just delete tbd from newleaves???

        listleaves = np.intersect1d(np.arange(nnodes//2, nnodes), listnodes) # current leaves to keep
        listleaves = np.unique(np.concatenate((listleaves, newleaves))) # current u new leaves

        return listleaves

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


def ThresholdNodes(self, list oldtree, bestleaves, threshold, str type):
        newtree = oldtree
        bestleavesset = set(bestleaves)
        for l in range(0, 2**(oldtree.maxlevel +1) - 1 ):
                ind = self.ConvertWaveletNodeName(l)
                if(l in bestleavesset):
                        # then keep & threshold
                        length = oldtree[ind].data.shape[0]

                        if(type=='hard'):
                                ce_thresnode(<double*> np.PyArray_DATA(oldtree[ind].data), <double*> np.PyArray_DATA(newtree[ind].data), length, threshold, 'h')
                        else:
                                ce_thresnode(<double*> np.PyArray_DATA(oldtree[ind].data), <double*> np.PyArray_DATA(newtree[ind].data), length, threshold, 's')
                else:
                        newtree[ind].data = np.zeros(len(oldtree[ind].data))

        return newtree

def ThresholdNodes2(self, list oldtree, bestleaves, threshold, str type):
        # Alternative version for custom (ndarray-type) WPs
        # Uses inplace thresholding, so use with care!
        # (i.e. arg oldtree will be overwritten)
        bestleavesset = set(bestleaves)
        for ind in range(len(oldtree)):
                if(ind in bestleavesset):
                        # then keep & threshold (inplace)
                        length = oldtree[ind].shape[0]
                        oldtree[ind] = np.ascontiguousarray(oldtree[ind])

                        if(type=='hard'):
                                ce_thresnode2(<double*> np.PyArray_DATA(oldtree[ind]), length, threshold, 'h')
                        else:
                                ce_thresnode2(<double*> np.PyArray_DATA(oldtree[ind]), length, threshold, 's')
                else:
                        oldtree[ind] = np.zeros(len(oldtree[ind]))

        # note: no return value b/c oldtree is edited inplace.

def EnergyCurve(np.ndarray C, M):
        assert C.dtype==np.float64
        # Args: 1. wav data 2. M (int), expansion in samples
        N = len(C)
        E = np.zeros(N)
        E[M] = np.sum(C[:2*M+1])
        ce_energycurve(<double*> np.PyArray_DATA(E), <double*> np.PyArray_DATA(C), N, M)
        return E

def FundFreqYin(np.ndarray data, W, i, ints):
        assert data.dtype==np.float64
        sd = np.zeros(W)
        data = data[i:]
        # Compute sum of squared diff (autocorrelation)
        ce_sumsquares(<double*> np.PyArray_DATA(data), W, <double*> np.PyArray_DATA(sd))

        # If not using window, instead:
        # for tau in range(1, W): 
            # if i>0:
            # for tau in range(1,W):
            # sd[tau] -= np.sum((data[i-1] - data[i-1+tau])**2)
            # sd[tau] += np.sum((data[i+W] - data[i+W+tau])**2)

        # Compute cumulative mean of normalised diff
        d = np.zeros(W)
        d[0] = 1 
        # TODO: sometimes all np.cumsum(sd[1;]) == 0 ??
        d[1:] = sd[1:] * ints / np.cumsum(sd[1:])

        return d
