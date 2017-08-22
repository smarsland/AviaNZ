# Version 0.3 14/8/17
# Author: Stephen Marsland

import numpy as np
import pywt

class WaveletFunctions:
    """ This class contains the wavelet specific methods.
    It is based on pywavelets (pywt), but has extra functions that are required
    to work with the wavelet packet tree.
    As far as possible it matches Matlab.
    dmey2 is in a file, and is an exact match for the Matlab dmeyer wavelet.

    Implements:
        waveletDenoise
        reconstructWPT
        waveletLeafCoeffs

    and helper functions:
        ShannonEntropy
        BestLevel
        BestTree
        ConvertWaveletNodeName
    """

    def __init__(self,data,wavelet,maxLevel):
        """ Gets the data and makes the wavelet, loading dmey2 (an exact match to Matlab's dmey) from a file.
        """
        self.data = data
        self.maxLevel = maxLevel

        if wavelet == 'dmey2':
            [lowd, highd, lowr, highr] = np.loadtxt('dmey.txt')
            self.wavelet = pywt.Wavelet(filter_bank=[lowd, highd, lowr, highr])
            self.wavelet.orthogonal=True
        else:
            self.wavelet = wavelet


    def ShannonEntropy(self,s):
        """ Compute the Shannon entropy of data
        """
        e = s[np.nonzero(s)]**2 * np.log(s[np.nonzero(s)]**2)
        return np.sum(e)

    def BestLevel(self,wavelet=None,maxLevel=None):
        """ Compute the best level for the wavelet packet decomposition by using the Shannon entropy.
        Iteratively add a new depth of tree until either the maxLevel level is found, or the entropy drops.
        """

        if wavelet is None:
            wavelet = self.wavelet
        if maxLevel is None:
            maxLevel = self.maxLevel

        previouslevelmaxE = self.ShannonEntropy(self.data)
        self.wp = pywt.WaveletPacket(data=self.data, wavelet=wavelet, mode='symmetric', maxlevel=maxLevel)
        level = 1
        currentlevelmaxE = np.max([self.ShannonEntropy(n.data) for n in self.wp.get_level(level, "freq")])
        while currentlevelmaxE < previouslevelmaxE and level<maxLevel:
            previouslevelmaxE = currentlevelmaxE
            level += 1
            currentlevelmaxE = np.max([self.ShannonEntropy(n.data) for n in self.wp.get_level(level, "freq")])
        return level

    def ConvertWaveletNodeName(self,i):
        """ Convert from an integer to the 'ad' representations of the wavelet packets
        The root is 0 (''), the next level are 1 and 2 ('a' and 'd'), the next 3, 4, 5, 6 ('aa','ad','da','dd) and so on
        """
        import string
        level = int(np.floor(np.log2(i + 1)))
        first = 2 ** level - 1
        if i == 0:
            b = ''
        else:
            b = np.binary_repr(i - first, width=int(level))
            b = string.replace(b, '0', 'a', maxreplace=-1)
            b = string.replace(b, '1', 'd', maxreplace=-1)
        return b

    def BestTree(self,wp,threshold,costfn='threshold'):
        """ Compute the best wavelet tree using one of three cost functions: threshold, entropy, or SURE.
        Scores each node and uses those scores to identify new leaves of the tree by working up the tree.
        Returns the list of new leaves of the tree.
        """
        nnodes = 2 ** (wp.maxlevel + 1) - 1
        cost = np.zeros(nnodes)
        count = 0
        for level in range(wp.maxlevel + 1):
            for n in wp.get_level(level, 'natural'):
                if costfn == 'threshold':
                    # Threshold
                    d = np.abs(n.data)
                    cost[count] = np.sum(d > threshold)
                elif costfn == 'entropy':
                    # Entropy
                    d = n.data ** 2
                    cost[count] = -np.sum(np.where(d != 0, d * np.log(d), 0))
                else:
                    # SURE
                    d = n.data ** 2
                    t2 = threshold * threshold
                    ds = np.sum(d > t2)
                    cost[count] = 2 * ds - len(n.data) + t2 * ds + np.sum(d * (d <= t2))

                count += 1

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

    def reconstructWPT(self,new_wp,wavelet,listleaves):
        """ Create a new wavelet packet tree by copying in the data for the leaves and then performing
        the idwt up the tree to the root.
        Assumes that listleaves is top-to-bottom, so just reverses it.
        """
        # Sort the list of leaves into order bottom-to-top, left-to-right
        working = listleaves.copy()
        working = working[-1::-1]

        level = int(np.floor(np.log2(working[0] + 1)))
        while level > 0:
            first = 2 ** level - 1
            while working[0] >= first:
                # Note this is Python2!
                # And also that it assumes that the whole list is backwards
                parent = (working[0] - 1) / 2
                p = self.ConvertWaveletNodeName(parent)
                new_wp[p].data = pywt.idwt(new_wp[self.ConvertWaveletNodeName(working[1])].data,new_wp[self.ConvertWaveletNodeName(working[0])].data, wavelet)[:len(new_wp[p].data)]
                # Delete these two nodes from working
                working = np.delete(working, 1)
                working = np.delete(working, 0)
                # Insert parent into list of nodes at the next level
                ins = np.where(working > parent)
                if len(ins[0]) > 0:
                    ins = ins[0][-1] + 1
                else:
                    ins = 0
                working = np.insert(working, ins, parent)
            level = int(np.floor(np.log2(working[0] + 1)))
        return new_wp

    def waveletDenoise(self,data=None,thresholdType='soft',threshold=4.5,maxLevel=5,bandpass=False,wavelet='dmey2',costfn='threshold'):
        """ Perform wavelet denoising.
        Constructs the wavelet tree to max depth (either specified or found), constructs the best tree, and then
        thresholds the coefficients (soft or hard thresholding), reconstructs the data and returns the data at the
        root.
        """

        if data is None:
            data = self.data

        if wavelet == 'dmey2':
            [lowd, highd, lowr, highr] = np.loadtxt('dmey.txt')
            wavelet = pywt.Wavelet(filter_bank=[lowd, highd, lowr, highr])
            wavelet.orthogonal=True

        if maxLevel is None:
            self.maxLevel = self.BestLevel(wavelet)
            print "Best level is ", self.maxLevel
        else:
            self.maxLevel = maxLevel

        self.thresholdMultiplier = threshold

        wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=self.maxLevel)

        # Get the threshold
        det1 = wp['d'].data
        # Note magic conversion number
        sigma = np.median(np.abs(det1)) / 0.6745
        threshold = self.thresholdMultiplier * sigma

        bestleaves = self.BestTree(wp,threshold)

        # Make a new tree with these in
        new_wp = pywt.WaveletPacket(data=None, wavelet=wp.wavelet, mode='zero', maxlevel=wp.maxlevel)

        # pywavelet makes the whole tree. So if you don't give it blanks from places where you don't want the values in
        # the original tree, it copies the details from wp even though it wasn't asked for them.
        # Reconstruction with the zeros is different to not reconstructing.
        for level in range(wp.maxlevel + 1):
            for n in new_wp.get_level(level, 'natural'):
                n.data = np.zeros(len(wp.get_level(level, 'natural')[0].data))

        # Copy thresholded versions of the leaves into the new wpt
        for l in bestleaves:
            ind = self.ConvertWaveletNodeName(l)
            if thresholdType == 'hard':
                # Hard thresholding
                new_wp[ind].data = np.where(np.abs(wp[ind].data) < threshold, 0.0, wp[ind].data)
            else:
                # Soft thresholding
                # n.data = np.sign(n.data) * np.maximum((np.abs(n.data) - threshold), 0.0)
                tmp = np.abs(wp[ind].data) - threshold
                tmp = (tmp + np.abs(tmp)) / 2.
                new_wp[ind].data = np.sign(wp[ind].data) * tmp

        # Reconstruct the internal nodes and the data
        new_wp = self.reconstructWPT(new_wp,wp.wavelet,bestleaves)

        return new_wp[''].data

    def waveletLeafCoeffs(self,data=None,maxLevel=None,wavelet='dmey2'):
        """ Return the wavelet coefficients of the leaf nodes.
        """
        if data is None:
            data = self.data

        if wavelet == 'dmey2':
            [lowd, highd, lowr, highr] = np.loadtxt('dmey.txt')
            wavelet = pywt.Wavelet(filter_bank=[lowd, highd, lowr, highr])
            wavelet.orthogonal=True

        if maxLevel is None:
            maxLevel = self.BestLevel(wavelet)
            print "Best level is ", self.maxLevel
        else:
            maxLevel = maxLevel

        wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxLevel)
        leafNodes=wp.get_leaf_nodes(decompose=False)
        # make the matrix 64*n
        leaves=[node.path for node in wp.get_level(maxLevel, 'natural')]
        mat=np.zeros((len(leaves),len(leafNodes[0][leaves[0]].data)))
        j=0
        for leaf in leaves:
            mat[j]=leafNodes[0][leaf].data
            j=j+1
        return mat

