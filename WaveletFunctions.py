
# WaveletFunctions.py
#
# Class containing wavelet specific methods

# Version 1.3 23/10/18
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis

#    AviaNZ birdsong analysis program
#    Copyright (C) 2017--2018

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
import pywt
import scipy.fftpack as fft
from ext import ce_denoise as ce

class WaveletFunctions:
    """ This class contains the wavelet specific methods.
    It is based on pywavelets (pywt), but has extra functions that are required
    to work with the wavelet packet tree.
    As far as possible it matches Matlab.
    dmey2 is in a file, and is an exact match for the Matlab dmeyer wavelet. It's the one to use.

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
        level = int(np.floor(np.log2(i + 1)))
        first = 2 ** level - 1
        if i == 0:
            b = ''
        else:
            b = np.binary_repr(int(i) - first, width=int(level))
            b = b.replace('0', 'a')
            b = b.replace('1', 'd')
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
                # Note that it assumes that the whole list is backwards
                parent = (working[0] - 1) // 2
                p = self.ConvertWaveletNodeName(parent)
                names = [self.ConvertWaveletNodeName(working[0]), self.ConvertWaveletNodeName(working[1])]

                new_wp[p].data = pywt.idwt(new_wp[names[1]].data, new_wp[names[0]].data, wavelet)[:len(new_wp[p].data)]

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

        print("Wavelet Denoising requested, with the following parameters: type %s, threshold %f, maxLevel %d, bandpass %s, wavelet %s, costfn %s" % (thresholdType, threshold, maxLevel, bandpass, wavelet, costfn))
        import time
        opstartingtime = time.time()
        if data is None:
            data = self.data

        if wavelet == 'dmey2':
            [lowd, highd, lowr, highr] = np.loadtxt('dmey.txt')
            wavelet = pywt.Wavelet(filter_bank=[lowd, highd, lowr, highr])
            wavelet.orthogonal=True

        if maxLevel == 0:
            self.maxLevel = self.BestLevel(wavelet)
            print("Best level is %d" % self.maxLevel)
        else:
            self.maxLevel = maxLevel

        self.thresholdMultiplier = threshold

        wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=self.maxLevel)
        # print("Checkpoint 1, %.5f" % (time.time() - opstartingtime))

        # Get the threshold
        det1 = wp['d'].data
        # Note magic conversion number
        sigma = np.median(np.abs(det1)) / 0.6745
        threshold = self.thresholdMultiplier * sigma

        # print("Checkpoint 1b, %.5f" % (time.time() - opstartingtime))
        bestleaves = ce.BestTree(wp,threshold,costfn)

        # Make a new tree with these in
        # pywavelet makes the whole tree. So if you don't give it blanks from places where you don't want the values in
        # the original tree, it copies the details from wp even though it wasn't asked for them.
        # Reconstruction with the zeros is different to not reconstructing.

        # Copy thresholded versions of the leaves into the new wpt
        new_wp = ce.ThresholdNodes(self, wp, bestleaves, threshold, thresholdType)

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
            print("Best level is ", self.maxLevel)
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

    def AntialiasWaveletPacket(self,data,wavelet,mode,maxlevel):
        wp_final = pywt.WaveletPacket(data=None, wavelet=wavelet,mode=mode,maxlevel=maxlevel)
        wp_final[''] = data

        for parent in range(2**maxlevel-1):
            print(parent, np.shape(wp_final[self.ConvertWaveletNodeName(parent)].data))
            wp_temp = pywt.WaveletPacket(data=wp_final[self.ConvertWaveletNodeName(parent)].data, wavelet=wavelet,mode=mode,maxlevel=1)
            l = len(wp_temp[''].data)

            # Get new approximation reconstruction
            new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet,mode=mode,maxlevel=1)
            # This doesn't help!
            #for level in range(new_wp.maxlevel + 1):
                #for n in new_wp.get_level(level, 'natural'):
                    #n.data = np.zeros(len(wp_temp.get_level(level, 'natural')[0].data))

            new_wp['a'] = wp_temp['a'].data
            a = new_wp.reconstruct(update=False)
            #print('a')
            ft = fft.fft(a,len(a))
            #print('b',len(a),l,len(new_wp['a'].data))
            ft[l//8:] = 0
            #ft[l//4:3*l//4] = 0
            data = np.real(fft.ifft(ft))
            #data = data[0::2]
            #print('c', self.ConvertWaveletNodeName(parent)+'a')
            new_wp = pywt.WaveletPacket(data=data, wavelet=wavelet,mode=mode,maxlevel=1)
            wp_final[self.ConvertWaveletNodeName(parent)+'a'] = new_wp['a'].data
           
            # Get new detail reconstruction
            new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet,mode=mode,maxlevel=1)
            new_wp['d'] = wp_temp['d'].data
            d = new_wp.reconstruct(update=False)
            ft = fft.fft(d)
            #ft[:l//4] = 0
            #ft[3*l//4:] = 0
            ft[:] = 0
            data = np.real(fft.ifft(ft))
            #data = data[0::2]
            new_wp = pywt.WaveletPacket(data=data, wavelet=wavelet,mode=mode,maxlevel=1)
            wp_final[self.ConvertWaveletNodeName(parent)+'d'] = new_wp['d'].data

        return wp_final

