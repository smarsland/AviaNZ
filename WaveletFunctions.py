
# WaveletFunctions.py
# Class containing wavelet specific methods

# Version 3.4 18/12/24
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2024

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
# import scipy.fftpack as fft
from scipy import signal
import pyfftw
from ext import ce_denoise as ce
import time
import Wavelet
import Spectrogram
import SignalProc

# A pair of helper functions that are often useful:
def graycode(n):
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
        toflip = bool(out[-1]=='1')

    return(int(out, 2))

def getWCFreq(node, sampleRate):
    """ Gets true frequencies of a wavelet node, based on sampling rate sampleRate."""
    # find node's scale
    lvl = math.floor(math.log2(node+1))
    # position of node in its level (0-based)
    nodepos = node - (2**lvl - 1)
    # Gray-permute node positions (cause wp is not in natural order)
    nodepos = graycode(nodepos)
    # get number of nodes in this level
    numnodes = 2**lvl

    freqmin = nodepos*sampleRate/2/numnodes
    freqmax = (nodepos+1)*sampleRate/2/numnodes
    return((freqmin, freqmax))

def adjustNodes(nodes, change):
    """ Fast remapping of node numbers which can be used
        instead of resampling by 2x.
        Change: "down2" or "up2", indicating what kind of
        resampling should be emulated this way.
    """
    adjnodes = []
    for node in nodes:
        lvl = math.floor(math.log2(node+1))
        numnodes = 2**lvl
        nodepos = node - (2**lvl - 1)

        # if you want the lower half subtree ("downsampling")
        if change=="down2":
            # remove nodes that are on the right side of the tree
            # (the only case when numnodes is odd is lvl=0 and that needs to go as well)
            if nodepos >= numnodes // 2:
                continue

            # else, renumber starting with a level lower
            node = 2**(lvl-1) - 1 + nodepos
            if node<0:
                print("Warning: weird node produced, skipping:", node)
            else:
                adjnodes.append(node)
        # if you want to change coords to one level higher ("upsampling")
        elif change=="up2":
            # renumber starting with a level higher
            node = 2**(lvl+1) - 1 + nodepos
            adjnodes.append(node)
        else:
            print("ERROR: unrecognised change", change)
    return adjnodes


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
        e = -s[np.nonzero(s)]**2 * np.log(s[np.nonzero(s)]**2)
        return np.sum(e)

    def BestLevel(self,maxLevel=None):
        """ Compute the best level for the wavelet packet decomposition by using the Shannon entropy.
        Iteratively add a new depth of tree until either the maxLevel level is found, or the entropy drops.
        """

        if maxLevel is None:
            maxLevel = self.maxLevel
        allnodes = range(2 ** (maxLevel + 1) - 1)

        previouslevelmaxE = self.ShannonEntropy(self.data)
        self.WaveletPacket(allnodes, 'symmetric', antialias=False, antialiasFilter=True)

        level = 1
        currentlevelmaxE = np.min([self.ShannonEntropy(self.tree[n][::2]) for n in range(1,3)])
        while currentlevelmaxE < previouslevelmaxE and level<maxLevel:
            previouslevelmaxE = currentlevelmaxE
            level += 1
            currentlevelmaxE = np.min([self.ShannonEntropy(self.tree[n][::2]) for n in range(2**level-1, 2**(level+1)-1)])
        return level

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

    # from memory_profiler import profile
    # fp = open('memory_profiler_wp.log', 'w+')
    # @profile(stream=fp)
    def WaveletPacket(self, nodes, mode='symmetric', antialias=False, antialiasFilter=True):
        """ Reimplementation of pywt.WaveletPacket, but allowing for antialias
            following Strang & Nguyen (1996) or
            An anti-aliasing algorithm for discrete wavelet transform. Jianguo Yang & S.T. Park (2003) or
            An Anti-aliasing and De-noising Hybrid Algorithm for Wavelet Transform. Yuding Cui, Caihua Xiong, and Ronglei Sun (2013)

            Data and wavelet are taken from current instance of WF. Therefore, ALWAYS use this together with WF, unless you're sure what you're doing.

            Args:
            1. nodes - list of integers, mandatory! will determine decomposition level from it
            2. mode - symmetric by default, as in pywt.WaveletPacket
            3. antialias - on/off switch
            4. antialiasFilter - switches between using filters or fft zeroing

            Return: none - sets self.tree.
        """
        if len(self.data) > 910*16000 and antialias:
            print("ERROR: processing files larger than 15 min in slow antialiasing mode is disabled. Enable this only if you are ready to wait.")
            return
        if len(nodes)==0 or not isinstance(nodes[0], int):
            print("ERROR: must provide a list of integer node IDs")
            return

        # identify max decomposition level
        maxlevel = math.floor(math.log2(max(nodes)+1))
        if maxlevel>10:
            print("ERROR: got level above 10, probably the nodes are specified badly")
            return

        # determine which nodes need to be produced (all parents of provided nodes)
        nodes = list(nodes)
        for child in nodes:
            parent = (child - 1) // 2
            if parent not in nodes and parent>=0:
                nodes.append(parent)
        nodes.sort()

        # object with dec_lo, dec_hi, rec_lo, rec_hi properties. Can be pywt.Wavelet or WF.wavelet
        wavelet = self.wavelet

        # filter length for extension modes
        flen = max(len(wavelet.dec_lo), len(wavelet.dec_hi), len(wavelet.rec_lo), len(wavelet.rec_hi))//2
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

        # loop over possible parent nodes (so down to leaf level-1)
        for node in range(2**maxlevel-1):
            childa = node*2 + 1
            childd = node*2 + 2
            # if this node is irrelevant, just put empty children to
            # keep tree order compatible with freq/filters
            if childa not in nodes and childd not in nodes:
                self.tree.append(np.array([]))
                self.tree.append(np.array([]))
                continue

            # retrieve parent node from J level
            data = self.tree[node]
            # downsample all non-root nodes because that wasn't done
            if node != 0:
                data = data[0::2]

            # symmetric mode
            data = np.concatenate((data[flen::-1], data, data[-1:-flen:-1]))
            # zero-padding mode
            # data = np.concatenate((np.zeros(8), tree[node], np.zeros(8)))

            ll = len(data)
            # make A_j+1 and D_j+1 (of length l)
            if childa in nodes:
                # fftconvolve seems slower and the caching results in high RAM usage
                # nexta = signal.fftconvolve(data, wavelet.dec_lo, 'same')[1:-1]
                nexta = np.convolve(data, wavelet.dec_lo, 'same')[flen:-flen]
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
                # explicit garbage collection - it helps somehow:
                del nexta
            else:
                self.tree.append(np.array([]))

            if childd in nodes:
                nextd = np.convolve(data, wavelet.dec_hi, 'same')[flen:-flen]
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
                del nextd
            else:
                self.tree.append(np.array([]))

            if antialias:
                print("Node ", node, " complete.")

        # Note: no return value, as it sets a tree on the WF object.

    def extractE(self, node, winsize, wpantialias=True):
        """ Extracts mean energies of node over windows of size winsize (s).
            Winsize will be adjusted to obtain integer number of WCs in this node.
            wpantialias - True for antialiased (non-decimated) tree
            Returns:
              np array of length nwins = datalength/winsize
              actual window size (in s) that was used
        """
        # wpantialias=True doubles the expected number of coefficients.
        # Turn it on when storing non-decimated WCs in a tree - 
        # this is currently true for all packet modes but NEVER for the root node
        # (as it's never made longer than the data length).
        if node==0:
            if wpantialias:
                print("Warning: you assumed antialias for a root node, this is probably not intended and will be reset now")
            wpantialias = False

        # ratio of current WC size to data ("how many samples went into one WC")
        level = math.floor(math.log2(node+1))
        dsratio = 2**level
        # (theoretical) sampling rate at this node ("how many WCs go into one second")
        nodefs = self.treefs / dsratio

        # or WCperWindow = math.ceil(WCperWindowFull / dsratio)
        WCperWindow = math.ceil(winsize * nodefs)
        # print("Node %d: %d WCs per window" %(node, WCperWindow))

        # realized window size in s - may differ from the requested one if it is not a multiple of 2^j samples
        realwindow = WCperWindow / nodefs

        # or nwindows = math.floor(datalengthSec / realwindow)
        if wpantialias:
            nwindows = math.floor(len(self.tree[node])/2 / WCperWindow)
        else:
            nwindows = math.floor(len(self.tree[node]) / WCperWindow)
        maxnumwcs = nwindows * WCperWindow

        # Sanity check for empty node:
        if nwindows <= 0:
            print("ERROR: data length %d shorter than window size %d s" %(len(self.tree[node]), winsize))
            return

        # WC from test node(s), trimmed to non-padded size
        if wpantialias:
            C = self.tree[node][:maxnumwcs*2:2]
        else:
            C = self.tree[node][:maxnumwcs]

        # Sanity check for all zero cases:
        if not any(C):
            print("Warning: tree empty at node %d" % node)
            return np.ndarray()

        # Might be useful to track any DC offset
        # print("DC offset = %.3f" % np.mean(C))

        # convert into a matrix (seconds x wcs in sec), and get the energy of each row (second)
        E = (C**2).reshape((nwindows, WCperWindow)).mean(axis=1)

        # cleanup
        C = None
        del C
        return E, realwindow

    def reconstructWP2(self, node, antialias=False, antialiasFilter=False):
        """ Inverse of WaveletPacket: returns the signal from a single node.
            Expects our homebrew (non-downsampled) WP.
            Takes Data and Wavelet from current WF instance.
            Antialias option controls freq squashing in final step.

            Return: the reconstructed signal, ndarray.
        """
        wv = self.wavelet
        data = self.tree[node]
        sp = Spectrogram.Spectrogram()

        lvl = math.floor(math.log2(node+1))
        # position of node in its level (0-based)
        nodepos = node - (2**lvl - 1)
        # Gray-permute node positions (cause wp is not in natural order)
        nodepos = graycode(nodepos)
        # positive freq is split into bands 0:1/2^lvl, 1:2/2^lvl,...
        # same for negative freq, so in total 2^lvl * 2 bands.
        numnodes = 2**(lvl+1)

        # do the actual convolutions + upsampling
        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype='float64')
        data = ce.reconstruct(data, node, np.array(wv.rec_hi), np.array(wv.rec_lo), lvl)

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
                print("antialiasing by filtering between %.3f-%.3f FN" %(low, high))
                data = SignalProc.FastButterworthBandpass(data, low, high)
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

        return data


    def waveletDenoise(self,thresholdType='soft',thrMultiplier=4.5,maxLevel=5, costfn='threshold', aaRec=False, aaWP=False, noiseest="const"):
        """ Perform wavelet denoising.
        Constructs the wavelet tree to max depth (either specified or found), constructs the best tree, and then
        thresholds the coefficients (soft or hard thresholding), reconstructs the data and returns the data at the root.
        Data and wavelet are taken from WF object's self.
        Args:
          1. threshold type ('soft'/'hard')
          2. threshold multiplier in sigmas
          3. max level (best basis up to this depth will be chosen)
          4. cost func for selecting best tree, or "fixed" to use maxLevel leaves
          6. antialias while reconstructing (T/F)
          7. antialias while building the WP ('full'), (T/F)
          8. noise energy estimation ("const"/"ols"/"qr")
        Return: reconstructed signal (ndarray)
        """
        print("Wavelet Denoising-Modified requested, with the following parameters: type %s, threshold %f, maxLevel %d, costfn %s, noiseest %s" % (thresholdType, thrMultiplier, maxLevel, costfn, noiseest))
        opstartingtime = time.time()

        ADJBLOCKLEN = 0.15  # block length in s to be used when estimating adj

        if maxLevel == 0:
            self.maxLevel = self.BestLevel()
            print("Best level is %d" % self.maxLevel)
        else:
            self.maxLevel = maxLevel

        # Create wavelet decomposition. Note: recommend full AA here
        allnodes = range(2 ** (self.maxLevel + 1) - 1)
        self.WaveletPacket(allnodes, 'symmetric', aaWP, antialiasFilter=True)
        print("Checkpoint 1, %.5f" % (time.time() - opstartingtime))

        datalen = len(self.tree[0])

        # Determine the best basis, or use all leaves ("fixed")
        # NOTE: nodes must be sorted here, very important!
        if costfn=="fixed":
            bestleaves = list(range(2**self.maxLevel-1,2**(self.maxLevel+1)-1))[:-1] # exclude the top node
        else:
            # NOTE: using same MAD threshold for basis selection.
            # it isn't even needed if entropy costfn is used here
            det1 = self.tree[2]
            basisThres = thrMultiplier * np.median(np.abs(det1)) / 0.6745
            bestleaves = ce.BestTree2(self.tree,basisThres,costfn)
            bestleaves = list(set(bestleaves))
            print("leaves to keep:", bestleaves)
        print("Checkpoint 2, %.5f" % (time.time() - opstartingtime))

        # Estimate the threshold (for each node)
        if noiseest == "const":
            # Constant threshold across all levels, nodes and times.
            # Estimate sd by MAD median of lvl 1 detail coefs.
            # Note magic conversion number for Gaussian MAD->SD
            det1 = self.tree[2]
            sigma = np.median(np.abs(det1)) / 0.6745
            threshold = thrMultiplier * sigma
            blocklen = 0
        elif noiseest == "n":
            # threshold node-specific, constant across times
            # Estimate the threshold by MAD for each node separately
            threshold = np.zeros(len(bestleaves))
            for leavenum in range(len(bestleaves)):
                node = bestleaves[leavenum]
                det1 = self.tree[node]
                sigma = np.median(np.abs(det1)) / 0.6745
                threshold[leavenum] = thrMultiplier * sigma
            blocklen = 0
        elif noiseest == "ols" or noiseest == "qr":
            # Thr is varying over time blocks, so need to supply block size.
            # Here we round it to obtain integer number of WCs:
            minwin = 32/self.treefs
            blocklen = round(ADJBLOCKLEN/minwin)*32  # in samples
            blocklen_s = blocklen / self.treefs  # in s

            # Estimate the thr for each node x block
            numblocks = math.floor(datalen/blocklen)
            threshold = np.zeros((len(bestleaves), numblocks))

            # Regression X: Extract log center freqs of appropriate nodes
            # (all 5th lvl leaves except top one which has filter edge effects):
            wind_nodes = list(range(31, 63))
            wind_nodes.remove(47)
            windnodecenters = [sum(getWCFreq(n, self.treefs))/2 for n in wind_nodes]
            regx = np.log(windnodecenters)

            # Regression Y: Extract log energies from the same nodes
            print("extracting node energy...")
            windE = np.zeros((numblocks, len(windnodecenters)))
            for node_ix in range(len(windnodecenters)):
                node = wind_nodes[node_ix]
                windE[:, node_ix], _ = self.extractE(node, blocklen_s)
            windE = np.log(windE)

            # X positions to interpolate at (can be non-5th lvl nodes)
            bestleafcenters = [sum(getWCFreq(n, self.treefs))/2 for n in bestleaves]
            interpx = np.log(bestleafcenters)

            # Will fit the log energies at log center freqs of each node
            # w/ a smooth interpolator, and then retrieve the smoothed values.
            if noiseest == "ols":
                # Fill the thr array w/ OLS estimates
                for t in range(numblocks):
                    regy = windE[t, :]
                    pol = np.polynomial.polynomial.Polynomial.fit(regx, regy, 3)
                    for node_ix in range(len(interpx)):
                        threshold[node_ix, t] = pol(interpx[node_ix])
            elif noiseest == "qr":
                # Create the polynomial features manually
                regx_poly = np.column_stack((np.ones(len(regx)), regx, regx**2, regx**3))
                # Fill the thr array w/ QR estimates
                for t in range(numblocks):
                    regy = windE[t, :]
                    pol = QuantReg(regy, regx_poly, q=0.20, max_iter=250, p_tol=1e-3)
                    for node_ix in range(len(interpx)):
                        threshold[node_ix, t] = pol(interpx[node_ix])
            # Threshold so far contains the predicted log-energies
            threshold = np.sqrt(np.exp(threshold))

            # for the highest freq node, just use the default MAD estimator
            # b/c filtering effects cause deviations from smooth models there
            # (hardcoded for top nodes in levels 3-7)
            for topnode in [11, 23, 47, 95, 191]:
                if topnode in bestleaves:
                    threshold[bestleaves.index(topnode), :] = np.median(np.abs(self.tree[topnode])) / 0.6745

            threshold *= thrMultiplier
        else:
            print("ERROR: unknown noise energy estimator ", noiseest)
            return
        print("thr shape", np.shape(threshold))

        # Overwrite the WPT with thresholded versions of the leaves
        exit_code = ce.ThresholdNodes2(self.tree, bestleaves, threshold=threshold, thrtype=thresholdType, blocklen=blocklen)
        if exit_code != 0:
            print("ERROR: ThresholdNodes2 exited with exit code ", exit_code)
            return
        print("Checkpoint 3, %.5f" % (time.time() - opstartingtime))

        # Reconstruct the internal nodes and the data
        data = self.tree[0]
        new_signal = np.zeros(len(data))
        for i in bestleaves:
            tmp = self.reconstructWP2(i, aaRec, True)[0:len(data)]
            new_signal = new_signal + tmp
        print("Checkpoint 4, %.5f" % (time.time() - opstartingtime))

        return new_signal


# Quantile regression model
#
# Model parameters are estimated using iterated reweighted least squares.
# Simplified version of statsmodels.regression.quantile_Regression
# (removed vcov matrix estimation etc.), as well as made compatible
# with numpy.polynomial callable API.
#
# Original author: Vincent Arel-Bundock
# License: BSD-3
# Created: 2013-03-19
class QuantReg():
    def __init__(self, endog, exog, q=.5, max_iter=500, p_tol=1e-5):
        """
        Estimate a quantile regression model using iterative reweighted least
        squares.

        Parameters
        ----------
        endog : array or dataframe
            endogenous/response variable
        exog : array or dataframe
            exogenous/explanatory variable(s)
        q : float
            Quantile must be strictly between 0 and 1
        """
        # Very much a hardcoded normalization, knowing that X is polynomial features
        exog = exog * np.asarray([1000, 100, 10, 1])

        # Ignoring rank check as we know X were created by non-linear transf.
        # exog_rank = np.linalg.matrix_rank(exog)
        exog_rank = 4
        n_iter = 0
        xstar = exog

        beta = np.ones(exog_rank)

        diff = 10
        while n_iter < max_iter and diff > p_tol:
            n_iter += 1
            beta0 = beta
            xtx = np.dot(xstar.T, exog)
            xty = np.dot(xstar.T, endog)
            beta = np.dot(np.linalg.pinv(xtx), xty)
            resid = endog - np.dot(exog, beta)

            mask = np.abs(resid) < .000001
            resid[mask] = ((resid[mask] >= 0) * 2 - 1) * .000001
            resid = np.where(resid < 0, q * resid, (1-q) * resid)
            resid = np.abs(resid)
            xstar = exog / resid[:, np.newaxis]
            diff = np.max(np.abs(beta - beta0))

        if n_iter == max_iter:
            print("Warning: maximum number of iterations (" + str(max_iter) + ") reached.")

        # un-transform the betas to allow predicting w/o normalizing
        self.beta = beta * np.asarray([1000, 100, 10, 1])

    def __call__(self, x):
        """ Predicts for the x value using a polynomial model and self.beta.
            Really hardcoded to our situation - assumes that .fit() was called previously
            and reads off model polynomial order based on beta length.
        """
        return sum([self.beta[i] * (x**i) for i in range(len(self.beta))])
