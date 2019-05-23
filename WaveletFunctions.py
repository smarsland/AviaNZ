
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
import math
import scipy.fftpack as fft
from scipy import signal
import pyfftw
from ext import ce_denoise as ce
import time
import Wavelet

class WaveletFunctions:
    """ This class contains the wavelet specific methods.
    It is based on pywavelets (pywt), but has extra functions that are required
    to work with the wavelet packet tree.
    As far as possible it matches Matlab.
    dmey2 is created from the Matlab dmeyer wavelet. It's the one to use.
    Other wavelets are created from pywt.Wavelet filter banks.

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

    def __init__(self,data,wavelet,maxLevel,samplerate):
        """ Gets the data and makes the wavelet, loading dmey2 (an exact match to Matlab's dmey) from a file.
            Stores some basic properties of the data (samplerate).
        """
        if data is None:
            print("ERROR: data must be provided")
            return
        if wavelet is None:
            print("ERROR: wavelet must be provided")
            return

        self.data = data
        self.maxLevel = maxLevel
        self.tree = None
        self.treefs = samplerate

        self.wavelet = Wavelet.Wavelet(name=wavelet)

    def ShannonEntropy(self,s):
        """ Compute the Shannon entropy of data
        """
        e = s[np.nonzero(s)]**2 * np.log(s[np.nonzero(s)]**2)
        return np.sum(e)

    def BestLevel(self,maxLevel=None):
        """ Compute the best level for the wavelet packet decomposition by using the Shannon entropy.
        Iteratively add a new depth of tree until either the maxLevel level is found, or the entropy drops.
        """

        if maxLevel is None:
            maxLevel = self.maxLevel

        previouslevelmaxE = self.ShannonEntropy(self.data)
        self.WaveletPacket(maxLevel, 'symmetric', aaWP=False, antialiasFilter=True)

        level = 1
        currentlevelmaxE = np.max([self.ShannonEntropy(self.tree[n]) for n in range(1,3)])
        while currentlevelmaxE < previouslevelmaxE and level<maxLevel:
            previouslevelmaxE = currentlevelmaxE
            level += 1
            currentlevelmaxE = np.max([self.ShannonEntropy(self.tree[n]) for n in range(2**level-1, 2**(level+1)-1)])
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

    def graycode(self, n):
        """ Returns a MODIFIED Gray permutation of n -
            which corresponds to the frequency band of position n.
            Input and output are integer ranks indicating position within level."""
        # convert number to binary repr string:
        n = bin(n)[2:]
        out = ''
        # never flip first bit
        toflip = False
        while n!='':
            # store leftmost bit or its complement to output
            if toflip:
                out = out + str(1-int(n[0]))
            else:
                out = out + n[0]
            # strip leftmost bit
            n = n[1:]
            # if this bit was 1, flip next bit
            if out[-1]=='1':
                toflip = True
            else:
                toflip = False

        return(int(out, 2))

    # from memory_profiler import profile
    # fp = open('memory_profiler_wp.log', 'w+')
    # @profile(stream=fp)
    def WaveletPacket(self, maxlevel, mode='symmetric', antialias=False, antialiasFilter=True):
        """ Reimplementation of pywt.WaveletPacket, but allowing for antialias
            following Strang & Nguyen (1996) or
            An anti-aliasing algorithm for discrete wavelet transform. Jianguo Yang & S.T. Park (2003) or
            An Anti-aliasing and De-noising Hybrid Algorithm for Wavelet Transform. Yuding Cui, Caihua Xiong, and Ronglei Sun (2013)

            Data and wavelet are taken from current instance of WF. Therefore, ALWAYS use this together with WF, unless you're sure what you're doing.

            Args:
            3. maxlevel - integer, mandatory!
            4. mode - symmetric by default, as in pywt.WaveletPacket
            5. antialias - on/off switch
            6. antialiasFilter - switches between using filters or fft zeroing

            Return: none - sets self.tree.
        """
        if len(self.data) > 910*16000 and antialias:
            print("ERROR: processing files larger than 15 min in slow antialiasing mode is disabled. Enable this only if you are ready to wait.")
            return

        # object with dec_lo, dec_hi, rec_lo, rec_hi properties. Can be pywt.Wavelet or WF.wavelet
        wavelet = self.wavelet

        # filter length for extension modes
        flen = max(len(wavelet.dec_lo), len(wavelet.dec_hi), len(wavelet.rec_lo), len(wavelet.rec_hi))
        # this tree will store non-downsampled coefs for reconstruction
        self.tree = [self.data]
        if mode != 'symmetric':
            print("ERROR: only symmetric WP mode implemented so far")
            return

        # optional filtering instead of FFT squashing.
        # see reconstructWP2 for more detailed explanation
        # manually confirmed that this filter is stable hence no SOS option.
        if antialiasFilter:
            low = 0.5
            hb,ha = signal.butter(20, low, btype='highpass')
            lb,la = signal.butter(20, low, btype='lowpass')

        # loop over possible parent nodes
        for node in range(2**maxlevel-1):
            # retrieve parent node from J level
            data = self.tree[node]
            # downsample all non-root nodes because that wasn't done
            if node != 0:
                data = data[0::2]

            # symmetric mode
            data = np.concatenate((data[0:flen:-1], data, data[-flen:]))
            # zero-padding mode
            # data = np.concatenate((np.zeros(8), tree[node], np.zeros(8)))

            ll = len(data)
            # make A_j+1 and D_j+1 (of length l)
            # fftconvolve seems slower and the caching results in high RAM usage
            # nexta = signal.fftconvolve(data, wavelet.dec_lo, 'same')[1:-1]
            # nextd = signal.fftconvolve(data, wavelet.dec_hi, 'same')[1:-1]
            nexta = np.convolve(data, wavelet.dec_lo, 'same')[1:-1]
            nextd = np.convolve(data, wavelet.dec_hi, 'same')[1:-1]

            # antialias A_j+1
            if antialias:
                if antialiasFilter:
                    nexta = signal.lfilter(lb, la, nexta)
                else:
                    ft = pyfftw.interfaces.scipy_fftpack.fft(nexta)
                    ft[ll//4 : 3*ll//4] = 0
                    nexta = np.real(pyfftw.interfaces.scipy_fftpack.ifft(ft))
            # store A before downsampling
            self.tree.append(nexta)

            # antialias D_j+1
            if antialias:
                if antialiasFilter:
                    nextd = signal.lfilter(hb, ha, nextd)
                else:
                    ft = pyfftw.interfaces.scipy_fftpack.fft(nextd)
                    ft[:ll//4] = 0
                    ft[3*ll//4:] = 0
                    nextd = np.real(pyfftw.interfaces.scipy_fftpack.ifft(ft))
            # store D before downsampling
            self.tree.append(nextd)

            # explicit garbage collection - it helps somehow:
            del nexta
            del nextd

            if antialias:
                print("Node ", node, " complete.")

        # Note: no return value, as it sets a tree on the WF object.

    def getWCFreq(self, node, sampleRate):
        """ Gets true frequencies of a wavelet node, based on sampling rate sampleRate."""

        # find node's scale
        lvl = math.floor(math.log2(node+1))
        # position of node in its level (0-based)
        nodepos = node - (2**lvl - 1)
        # Gray-permute node positions (cause wp is not in natural order)
        nodepos = self.graycode(nodepos)
        # get number of nodes in this level
        numnodes = 2**lvl

        freqmin = nodepos*sampleRate/2/numnodes
        freqmax = (nodepos+1)*sampleRate/2/numnodes
        return((freqmin, freqmax))


    def reconstructWP2(self, node, antialias=False, antialiasFilter=False):
        """ Inverse of WaveletPacket: returns the signal from a single node.
            Expects our homebrew (non-downsampled) WP.
            Takes Data and Wavelet from current WF instance.
            Antialias option controls freq squashing in final step.

            Return: the reconstructed signal, ndarray.
        """
        opstt = time.time()
        wv = self.wavelet
        data = self.tree[node]

        lvl = math.floor(math.log2(node+1))
        # position of node in its level (0-based)
        nodepos = node - (2**lvl - 1)
        # Gray-permute node positions (cause wp is not in natural order)
        nodepos = self.graycode(nodepos)
        # positive freq is split into bands 0:1/2^lvl, 1:2/2^lvl,...
        # same for negative freq, so in total 2^lvl * 2 bands.
        numnodes = 2**(lvl+1)

        # do the actual convolutions + upsampling
        data = ce.reconstruct(data, node, np.array(wv.rec_hi), np.array(wv.rec_lo), lvl)
        print("rec ch 1", time.time() - opstt)

        if antialias:
            if len(data) > 910*16000 and not antialiasFilter:
                print("Size of signal to be reconstructed is", len(data))
                print("ERROR: processing of big data chunks is currently disabled. Recommend splitting files to below 15 min chunks. Enable this only if you know what you're doing.")
                return

            if antialiasFilter:
                # BETTER METHOD for antialiasing
                # essentially same as SignalProc.ButterworthBandpass,
                # just stripped to minimum for speed.
                low = nodepos / numnodes*2
                high = (nodepos+1) / numnodes*2
                print("antialising by filtering between %.3f-%.3f FN" %(low, high))

                # Small buffer bands of 0.001 extend critical bands by 16 Hz at 32 kHz sampling
                # (i.e. critical bands will be 16 Hz wider than passbands in each direction)
                # Otherwise could use signal.buttord to calculate the critical bands.
                if low==0 and high==1:
                    return data
                if low==0:
                    b,a = signal.butter(7, high+0.002, btype='lowpass')
                elif high==1:
                    b,a = signal.butter(7, low-0.002, btype='highpass')
                else:
                    b,a = signal.butter(7, [low-0.002, high+0.002], btype='bandpass')

                # Check filter stability
                # (smarter methods exist, but this should be fine for short filters)
                filterUnstable = np.any(np.abs(np.roots(a))>1)

                # NOTE: can use SOS instead of (b,a) representation to improve stability at high order
                # (needed for steep transitions).
                if filterUnstable:
                    print("single-stage filter unstable, switching to SOS filtering")
                    if low==0:
                        sos = signal.butter(30, high+0.002, btype='lowpass', output='sos')
                    elif high==1:
                        sos = signal.butter(30, low-0.002, btype='highpass', output='sos')
                    else:
                        sos = signal.butter(30, [low-0.002, high+0.002], btype='bandpass', output='sos')
                    data = signal.sosfilt(sos, data)
                else:
                    # if filter appears stable, run it on full data
                    data = signal.lfilter(b, a, data)

            else:
                # OLD METHOD for antialiasing
                # just setting image frequencies to 0
                print("antialiasing via FFT")
                ft = pyfftw.interfaces.scipy_fftpack.fft(data)
                ll = len(ft)
                # to keep: [nodepos/numnodes : (nodepos+1)/numnodes] x Fs
                # (same for negative freqs)
                ft[ : ll*nodepos//numnodes] = 0
                ft[ll*(nodepos+1)//numnodes : -ll*(nodepos+1)//numnodes] = 0
                # indexing [-0:] wipes everything
                if nodepos!=0:
                    ft[-ll*nodepos//numnodes : ] = 0
                data = np.real(pyfftw.interfaces.scipy_fftpack.ifft(ft))
        print("rec ch 2", time.time() - opstt)

        return data


    def waveletDenoise(self,thresholdType='soft',threshold=4.5,maxLevel=5,bandpass=False, costfn='threshold', aaRec=False, aaWP=False, thrfun="c"):
        """ Perform wavelet denoising.
        Constructs the wavelet tree to max depth (either specified or found), constructs the best tree, and then
        thresholds the coefficients (soft or hard thresholding), reconstructs the data and returns the data at the root.
        Data and wavelet are taken from WF object's self.
        Args:
          1. threshold type ('soft'/'hard')
          2-5. obvious parameters
          6. antialias while reconstructing (T/F)
          7. antialias while building the WP ('full'), (T/F)
        Return: reconstructed signal (ndarray)
        """

        print("Wavelet Denoising-Modified requested, with the following parameters: type %s, threshold %f, maxLevel %d, bandpass %s, costfn %s" % (thresholdType, threshold, maxLevel, bandpass, costfn))
        opstartingtime = time.time()

        if maxLevel == 0:
            self.maxLevel = self.BestLevel()
            print("Best level is %d" % self.maxLevel)
        else:
            self.maxLevel = maxLevel

        self.thresholdMultiplier = threshold

        # Create wavelet decomposition. Note: using full AA here
        self.WaveletPacket(self.maxLevel, 'symmetric', aaWP, antialiasFilter=True)
        print("Checkpoint 1, %.5f" % (time.time() - opstartingtime))

        # Get the threshold
        det1 = self.tree[2]
        # Note magic conversion number
        sigma = np.median(np.abs(det1)) / 0.6745
        threshold = self.thresholdMultiplier * sigma

        print("Checkpoint 2, %.5f" % (time.time() - opstartingtime))
        # NOTE: node order is not the same
        # NOTE: threshold isn't needed for Entropy cost fn
        bestleaves = ce.BestTree2(self.tree,threshold,costfn)
        print("leaves to keep:", bestleaves)

        # Make a new tree with these in
        # pywavelet makes the whole tree. So if you don't give it blanks from places where you don't want the values in
        # the original tree, it copies the details from wp even though it wasn't asked for them.
        # Reconstruction with the zeros is different to not reconstructing.

        # Copy thresholded versions of the leaves into the new wpt
        # NOTE: this version overwrites the provided wp
        if thrfun == "c":
            # constant threshold across all levels, nodes and times
            exit_code = ce.ThresholdNodes2(self, self.tree, bestleaves, threshold, thresholdType)
        elif thrfun == "l":
            # threshold level-specific, constant across nodes and times
            exit_code = ce.ThresholdNodes2(self, self.tree, bestleaves, threshold, thresholdType)
            # TODO
        elif thrfun == "n":
            # threshold node-specific, constant across times
            # Get the threshold
            threshold = np.zeros(len(bestleaves))
            bestleaves_sort = list(set(bestleaves))
            # NOTE: IMPORTANT: bestleaves must be in set-order!!
            for leavenum in range(len(bestleaves_sort)):
                node = bestleaves_sort[leavenum]
                det1 = self.tree[node]
                # Note magic conversion number
                sigma = np.median(np.abs(det1)) / 0.6745
                threshold[leavenum] = self.thresholdMultiplier * sigma
            exit_code = ce.ThresholdNodes2(self, self.tree, bestleaves, threshold, thresholdType)
        else:
            print("ERROR: unknown threshold type ", thrfun)
            return
        if exit_code != 0:
            print("ERROR: ThresholdNodes2 exited with exit code ", exit_code)
            return

        # Reconstruct the internal nodes and the data
        print("Checkpoint 3, %.5f" % (time.time() - opstartingtime))
        data = self.tree[0]
        new_wp = np.zeros(len(data))
        for i in bestleaves:
            tmp = self.reconstructWP2(i, aaRec, True)[0:len(data)]
            new_wp = new_wp + tmp
        print("Checkpoint 4, %.5f" % (time.time() - opstartingtime))

        return new_wp
