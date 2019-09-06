# WaveletSegment.py
#
# Wavelet Segmentation

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
import WaveletFunctions
import wavio, librosa
import numpy as np
import json, time, os, math, csv, gc
import SignalProc
import SupportClasses
import Segment
from ext import ce_denoise as ce
import psutil
import copy, pickle, tempfile

# Nirosha's approach of simultaneous segmentation and recognition using wavelets
# (0) Bandpass filter with different parameters for each species
# (1) 5 level wavelet packet decomposition
# (2) Sort nodes of (1) into order by point-biserial correlation with training labels
# This is based on energy
# (3) Retain top nodes (up to 20)
# (4) Re-sort to favour child nodes
# (5) Reduce number using F_2 score
# This is based on thresholded reconstruction
# (6) Classify as call if OR of (5) is true

# Virginia: added window overlap
# NOTE: inc is supposed to be a "fair" fraction of window

# TODO: Inconsisient about symmlots of or zeros for the wavelet packet
# TODO: This still needs lots of tidying up

class WaveletSegment:
    # This class implements wavelet segmentation for the AviaNZ interface

    def __init__(self, spInfo, wavelet='dmey2', annotation=[], mingap=0.3, minlength=0.2):
        self.annotation = annotation
        self.wavelet = wavelet
        self.spInfo = spInfo
        if not spInfo == {}:
            # for now, we default to the first subfilter:
            print("Detected %d subfilters in this filter" % len(spInfo["Filters"]))

        self.sp = SignalProc.SignalProc([], 0, 256, 128)
        self.segmenter = Segment.Segmenter(None, None, self.sp, 0, window_width=256, incr=128, mingap=mingap, minlength=minlength)

    def waveletSegment(self, data, sampleRate, d, wpmode="new"):
        """ Main analysis wrapper (segmentation in batch mode).
            Args:
            1. data to be segmented, ndarray
            2. sampleRate of the data, int
            3. d - turn on denoising before calling?
            4. wpmode: old/new/aa to indicate no/partial/full antialias

            Returns: list of lists of segments found (over each subfilter)
        """
        if data is None or data == [] or len(data) == 0:
            print("ERROR: data must be provided for WS")
            return

        opst = time.time()
        denoisedData = self.preprocess(data, sampleRate, d=d)

        # Generate a full 5 level wavelet packet decomposition (stored in WF.tree)
        self.WF = WaveletFunctions.WaveletFunctions(data=denoisedData, wavelet=self.wavelet, maxLevel=20, samplerate=sampleRate)
        if wpmode == "pywt":
            print("ERROR: pywt wpmode is deprecated, use new or aa")
            return
        if wpmode == "new" or wpmode == "old":
            self.WF.WaveletPacket(mode='symmetric', maxlevel=5, antialias=False)
        if wpmode == "aa":
            self.WF.WaveletPacket(mode='symmetric', maxlevel=5, antialias=True, antialiasFilter=True)

        # For memory concerns, could reset some nodes to 0:
        # for ni in range(len(self.WF.tree)):
        #     # note that we don't reset node 0 as it's good to keep original data
        #     if ni not in goodnodes and ni!=0:
        #         self.WF.tree[ni] = [0]

        ### Now, find segments with each subfilter separately
        detected_allsubf = []
        for subfilter in self.spInfo["Filters"]:
            print("Identifying calls using subfilter", subfilter["calltype"])

            # Segment detection and neighbour merging
            goodnodes = subfilter['WaveletParams'][2]
            detected = self.detectCalls(self.WF, nodelist=goodnodes, samplerate=self.spInfo["SampleRate"], subfilter=subfilter, rf=True, aa=wpmode!="old")

            # merge neighbours in order to convert the detections into segments
            # note: detected np[0 1 1 1] becomes [[1,3]]
            detected = np.where(detected > 0)
            if np.shape(detected)[1] > 1:
                detected = self.identifySegments(np.squeeze(detected))
            elif np.shape(detected)[1] == 1:
                detected = np.array(detected).flatten().tolist()
                detected = self.identifySegments(detected)
            else:
                detected = []
            detected = self.mergeSeg(detected)
            detected_allsubf.extend(detected)
        print("Wavelet segmenting completed in", time.time() - opst)
        return detected_allsubf

    def waveletSegment_train(self, dirName, thrList, MList, d=False, rf=True, learnMode='recaa', window=1,
                             inc=None):
        """ Entry point to use during training, called from AviaNZ.py.
            Switches between various training methods, orders data loading etc.,
            then just passes the arguments to the right training method and returns the results.

            Input: path to directory with wav & wav.data files.
            window is window length in sec.
            inc is increment length in sec.
            Argument _learnMode_ will determine which detectCalls function is used:
            learnMode=="ethr":
            "1" - get wavelet node energies, threshold over those
            learnMode=="recsep":
            "3" - reconstruct signal from each node individually, threshold over that
            ("the old way")
            learnMode=="recmulti":
            "2" - reconstruct signal from all selected nodes, threshold over that
            learnMode=="recaa":
            reconstruct signal from each node individually,
            using homebrew NOT antialiased WPs, and antialiased reconstruction.
            learnMode=="recaafull":
            reconstruct signal from each node individually,
            using homebrew antialiased WPs (SLOW), and antialiased reconstruction.
            Return: tuple of arrays (nodes, tp, fp, tn, fn)
        """
        # 1. read wavs and annotations into self.annotation, self.audioList
        self.loadDirectory(dirName=dirName, denoise=d, window=window, inc=inc)

        # 2. find top nodes for each file (self.nodeCorrs)
        nlevels = 5
        # recommend using wpmode="new", because it is fast and almost alias-free.
        wpmode = "new"
        self.nodeCorrs = []
        self.bestNodes = []
        self.maxEs = []
        inc = 1
        for filenum in range(len(self.audioList)):
            print("Computing wavelet node correlations in file", filenum+1)
            currWCs = self.computeWaveletEnergy(self.audioList[filenum], self.spInfo['SampleRate'], nlevels, wpmode, window=window, inc=inc)
            # Compute all WC-annot correlations
            nodeCorr = self.compute_r(self.annotation[filenum], currWCs)
            self.nodeCorrs.append(nodeCorr)
            # find best nodes
            bestnodes, _ = self.listTopNodes(filenum)
            self.bestNodes.append(bestnodes)

        # 3. generate WPs for each file and store the max energies
        for filenum in range(len(self.audioList)):
            print("Extracting energies from file", filenum+1)
            data = self.audioList[filenum]
            self.WF = WaveletFunctions.WaveletFunctions(data=data, wavelet=self.wavelet, maxLevel=20, samplerate=self.spInfo['SampleRate'])
            # Generate a full 5 level wavelet packet decomposition
            if learnMode == "recaa" or learnMode == "recold":
                self.WF.WaveletPacket(mode='symmetric', maxlevel=nlevels, antialias=False)
            elif learnMode == "recaafull":
                self.WF.WaveletPacket(mode='symmetric', maxlevel=nlevels, antialias=True, antialiasFilter=True)
            else:
                print("ERROR: learnMode unrecognized")
                return

            # find E peaks over possible M (returns [MxTxN])
            maxEsFile = self.extractE(self.WF, self.bestNodes[filenum], MList, aa=learnMode!="recold", window=window, inc=inc, annotation=self.annotation[filenum])
            self.maxEs.append(maxEsFile)
        # self.maxEs now is a list of [files][M][TxN] ndarrays

        # 4. mark calls and learn threshold
        res = self.gridSearch(self.maxEs, thrList, MList, rf, learnMode, window, inc)

        return res

    def waveletSegment_test(self, dirName, listnodes=None, d=False, rf=True, learnMode='recaa',
                            savedetections=False, window=1, inc=None):
        """ Wrapper for segmentation to be used when testing a new filter
            (called at the end of training from AviaNZ.py).
            Basically a simplified gridSearch.
        """
        # Virginia changes
        # Added window and inc input
        # window -> window length in seconds
        # Inc -> increment length in seconds
        # Resol is the base of the annotations

        # Load the relevant list of nodes
        if listnodes is None:
            self.nodes = self.spInfo['WaveletParams'][2]
        else:
            self.nodes = listnodes

        # clear storage for multifile processing
        self.annotation = []
        self.audioList = []
        self.filelengths = []
        self.filenames = []
        detected = np.array([])

        # Loads all audio data to memory
        # and downsamples to self.spInfo['SampleRate']
        self.loadDirectory(dirName=dirName, denoise=d, filter=f, window=window, inc=inc)

        # remember to convert main structures to np arrays
        self.annotation = np.array(self.annotation)
        print("Testing with %s positive and %s negative annotations" % (np.sum(self.annotation == 1),
                                                                        np.sum(self.annotation == 0)))

        # wavelet decomposition and call detection
        for fileId in range(len(self.audioList)):
            print('Processing file # ', fileId + 1)

            if learnMode == "recsep" or learnMode == "recmulti":
                print("Warning: recsep and recmulti modes deprecated, defaulting to recaa")
                learnMode = "recaa"

            data = self.audioList[fileId]
            # Generate a full 5 level wavelet packet decomposition and detect calls
            self.WF = WaveletFunctions.WaveletFunctions(data=data, wavelet=self.wavelet, maxLevel=20,
                                                        samplerate=self.spInfo['SampleRate'])
            if learnMode == "recaa" or learnMode =="recold":
                self.WF.WaveletPacket(mode='symmetric', maxlevel=5, antialias=False)
                detected_c = self.detectCalls(self.WF, nodelist=self.nodes, spInfo=self.spInfo, rf=rf, window=1,
                                              inc=None)
            elif learnMode == "recaafull":
                self.WF.WaveletPacket(mode='symmetric', maxlevel=5, antialias=True, antialiasFilter=True)
                detected_c = self.detectCalls(self.WF, nodelist=self.nodes, spInfo=self.spInfo, rf=rf, window=1,
                                              inc=None)
            else:
                print("ERROR: the specified learning mode is not implemented in this function yet")
                return

            detected = np.concatenate((detected, detected_c))
            # Generate .data for this file
            # Merge neighbours in order to convert the detections into segments
            # Note: detected np[0 1 1 1] becomes [[1,3]]
            # TODO currently disabled to avoid conflicts with new format
            # if savedetections:
            #     detected_c = np.where(detected_c > 0)
            #     if np.shape(detected_c)[1] > 1:
            #         detected_c = self.identifySegments(np.squeeze(detected_c))
            #     elif np.shape(detected_c)[1] == 1:
            #         detected_c = np.array(detected_c).flatten().tolist()
            #         detected_c = self.identifySegments(detected_c)
            #     else:
            #         detected_c = []
            #     detected_c = self.mergeSeg(detected_c)
            #     for item in detected_c:
            #         item[0] = int(item[0])
            #         item[1] = int(item[1])
            #         item = item.append(self.spInfo['FreqRange'][0])
            #     for item in detected_c:
            #         item = item.append(self.spInfo['FreqRange'][1])
            #     for item in detected_c:
            #         item = item.append(self.spInfo['Name'])
            #     file = open(str(self.filenames[fileId]) + '.wav.data', 'w')
            #     json.dump(detected_c, file)

            # memory cleanup:
            del self.WF
            gc.collect()

        self.annotation = np.concatenate(self.annotation, axis=0)
        fB, recall, TP, FP, TN, FN = self.fBetaScore(self.annotation, detected)
        return detected, TP, FP, TN, FN


    def computeWaveletEnergy(self, data, sampleRate, nlevels=5, wpmode="new", window=1, inc=1, resol=1):
        """ Computes the energy of the nodes in the wavelet packet decomposition
        Args:
        1. data (waveform)
        2. sample rate
        3. max levels for WP decomposition
        4. WP style ("new"-our non-downsampled, "aa"-our fully AA'd)
        There are 62 coefficients up to level 5 of the wavelet tree (without root!!), and 300 seconds [N sliding window] in 5 mins
        Hence returned coefs would then be a 62*300 matrix [62*N matrix]
        The energy is the sum of the squares of the data in each node divided by the total in that level of the tree as a percentage.
        """

        # Virginia changes:
        # Added window and inc input
        # Window is window length in sec.
        # Inc is increment length in sec.
        # Energy is calculated on sliding windows
        # the window is a "centered" window

        if data is None or sampleRate is None:
            print("ERROR: data and Fs need to be specified")
            return

        #Virginia: number of samples in window
        win_sr = int(math.ceil(window*sampleRate))
        # half-window length in samples
        #win_sr2=int(math.ceil(win_sr/2))
        #Virginia: number of sample in increment
        inc_sr = math.ceil(inc*sampleRate)
        #Virginia: number of samples in resolution
        resol_sr = math.ceil(resol * sampleRate)
        #Virginia: needed to generate coef of the same size of annotations
        step = int(inc/resol)

        #Virginia:number of windows = number of sliding window at resol distance
        N = int(math.ceil(len(data)/resol_sr))

        #Virginia: changed columns dimension -> must be equal to number of sliding window
        coefs = np.zeros((2 ** (nlevels + 1) - 2, N))

        #Virginia-> for each sliding window:
        # start is the sample start of a window
        #end is the sample end of a window
        #We are working with sliding windows starting from the file start
        start = 0 #inizialization
        #Virginia: the loop works on the resolution scale to adjust with annotations
        for t in range(0, N, step):
            E = []
            end = min(len(data), start+win_sr)
            # generate a WP
            WF = WaveletFunctions.WaveletFunctions(data=data[start:end], wavelet=self.wavelet, maxLevel=20, samplerate=sampleRate)
            if wpmode == "pywt":
                print("ERROR: pywt mode deprecated, use new or aa")
                return
            if wpmode == "new":
                WF.WaveletPacket(mode='symmetric', maxlevel=nlevels, antialias=False)
            if wpmode == "aa":
                WF.WaveletPacket(mode='symmetric', maxlevel=nlevels, antialias=True, antialiasFilter=True)

            # Calculate energies
            for level in range(1, nlevels + 1):
                lvlnodes = WF.tree[2 ** level - 1:2 ** (level + 1) - 1]
                e = np.array([np.sum(n ** 2) for n in lvlnodes])
                if np.sum(e) > 0:
                    e = 100.0 * e / np.sum(e)
                E = np.concatenate((E, e), axis=0)


            #Virginia:update start
            start += inc_sr     # Virginia: corrected
            for T in range(t, t+step):
                coefs[:, T] = E
        return coefs


    def fBetaScore(self, annotation, predicted, beta=2):
        """ Computes the beta scores given two sets of predictions """
        annotation = np.array(annotation)
        predicted = np.array(predicted)
        TP = np.sum(np.where((annotation == 1) & (predicted == 1), 1, 0))
        T = np.sum(annotation)
        P = np.sum(predicted)
        if T != 0:
            recall = float(TP) / T  # TruePositive/#True
        else:
            recall = None
        if P != 0:
            precision = float(TP) / P  # TruePositive/#Positive
        else:
            precision = None
        if recall is not None and precision is not None and not (recall == 0 and precision == 0):
            fB = ((1. + beta ** 2) * recall * precision) / (recall + beta ** 2 * precision)
        else:
            fB = None
        if recall is None and precision is None:
            print("TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%s \tPrecision=%s \tfB=%s" % (
            TP, P - TP, len(annotation) - (P + T - TP), T - TP, recall, precision, fB))
        elif recall is None:
            print("TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%s \tPrecision=%0.2f \tfB=%s" % (
            TP, P - TP, len(annotation) - (P + T - TP), T - TP, recall, precision, fB))
        elif precision is None:
            print("TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%0.2f \tPrecision=%s \tfB=%s" % (
            TP, P - TP, len(annotation) - (P + T - TP), T - TP, recall, precision, fB))
        elif fB is None:
            print("TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%0.2f \tPrecision=%0.2f \tfB=%s" % (
            TP, P - TP, len(annotation) - (P + T - TP), T - TP, recall, precision, fB))
        else:
            print("TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%0.2f \tPrecision=%0.2f \tfB=%0.2f" % (
            TP, P - TP, len(annotation) - (P + T - TP), T - TP, recall, precision, fB))
        # print TP, int(T), int(P), recall, precision, ((1.+beta**2)*recall*precision)/(recall + beta**2*precision)
        return fB, recall, TP, P - TP, len(annotation) - (P + T - TP), T - TP  # fB, recall, TP, FP, TN, FN


    def compute_r(self, annotation, waveletCoefs):
        """ Computes the point-biserial correlations for a set of labels and a set of wavelet coefficients.
            r = (M_p - M_q) / S * sqrt(p*q), M_p = mean for those that are 0, S = std dev overall, p = proportion that are 0.
            Inputs:
            1. annotation - np.array of length n, where n - number of blocks (with resolution length) in file
            2. waveletCoefs - np.array of DxN, where D - number of nodes in WP (62 for lvl 5) N= number of sliding windows
        """
        w0 = np.where(annotation == 0)[0]
        w1 = np.where(annotation == 1)[0]

        r = np.zeros(np.shape(waveletCoefs)[0])
        for node in range(len(r)):
            r[node] = (np.mean(waveletCoefs[(node, w1)]) - np.mean(waveletCoefs[(node, w0)])) / np.std(
                waveletCoefs[node, :]) * np.sqrt(len(w0) * len(w1)) / len(annotation)

        return r

    def sortListByChild(self, order):
        """ Inputs is a list sorted into order of correlation.
        This functions resort so that any children of the current node that are in the list go first.
        Assumes that there are five levels in the tree (easy to extend, though)
        """
        # TODO: simplify and make flexible for any size list
        newlist = []
        currentIndex = 0
        # Need to keep track of where each level of the tree starts
        # Note that there is no root to the tree, hence the 0 then 2
        starts = [0, 2, 6, 14, 30, 62]
        while len(order) > 0:
            if order[0] < 30:
                # It could have children lower down the list
                # Build a list of the children of the first element of order
                level = int(np.log2(order[0] + 2))
                nc = 2
                first = order[0]
                for l in range(level + 1, 6):
                    children = []
                    current = currentIndex
                    for i in range(nc):
                        children.append(starts[l] + 2 * (first - starts[l - 1]) + i)
                    nc *= 2
                    first = starts[l] + 2 * (first - starts[l - 1])
                    # Have to do it this annoying way since Python seems to ignore the next element if you delete one while iterating over the list
                    i = 0
                    order_sub = []
                    while i < len(children):
                        if children[i] not in order:
                            del (children[i])
                        else:
                            order_sub.append(order.index(children[i]))
                            i += 1

                    # Sort into order
                    children = [x for (y, x) in sorted(zip(order_sub, children), key=lambda pair: pair[0])]

                    for a in children:
                        # If so, remove and insert at the current location in the new list
                        newlist.insert(current, a)
                        current += 1
                        order.remove(a)

            # Finally, add the first element
            newlist.append(order[0])
            currentIndex = newlist.index(order[0]) + 1
            del (order[0])

        return newlist


    def extractE(self, wf, nodelist, MList, rf=True, annotation=None, window=1, inc=None, aa=True):
        """
        Regenerates the signal from each of nodes and finds max standardized E.
        Args:
        1. wf - WaveletFunctions with a homebrew wavelet tree (list of ndarray nodes)
        2. nodelist - will reconstruct signal and run detections on each of these nodes separately
        3. MList - passed here to allow multiple Ms to be tested from one reconstruction
        4. rf - bandpass to species freq range?
        5. annotation - for calculating noise properties during training
        6-7. window, inc - window / increment length, seconds
        8. antialias - True/False
        Return: ndarrays of MxTxN energies, for each of M values, T windows and N nodes
        """

        if inc is None:
            inc = window
            resol = window
        else:
            resol = (math.gcd(int(100 * window), int(100 * inc))) / 100
        annotation = np.array(annotation)

        duration = len(wf.tree[0])

        # Window length in samples
        win_sr = math.ceil(window * self.spInfo['SampleRate'])
        # Increment length in samples
        inc_sr = math.ceil(inc * self.spInfo['SampleRate'])
        # Resolution length in samples
        resol_sr = math.ceil(resol * self.spInfo['SampleRate'])

        # number of windows of length inc
        nw = int(np.ceil(duration / inc_sr))

        nodenum = 0
        maxE = np.zeros((len(MList), nw, len(nodelist)))
        for node in nodelist:
            useWCenergies = False
            # Option 1: use wavelet coef energies directly
            if useWCenergies:
                # how many samples went into one WC?
                samples_wc = 2**math.floor(math.log2(node+1))
                duration = int(duration/samples_wc)
                # put WC from test node(s) on the new tree
                C = wf.tree[node][0::2]
            # Option 2: reconstruct from the WCs, as before
            else:
                samples_wc = 1
                C = wf.reconstructWP2(node, antialias=aa, antialiasFilter=True)

            # Sanity check for all zero case
            if not any(C):
                continue

            if len(C) > duration:
                C = C[:duration]

            C = np.abs(C)
            N = len(C)

            # Compute threshold using mean & sd from non-call sections
            if annotation is not None:
                noiseSamples = np.repeat(annotation == 0, resol_sr/samples_wc)
                noiseSamples = noiseSamples[:len(C)]
            else:
                print("Warning: no annotations detected in file")
                noiseSamples = np.full(len(C), True)
            meanC = np.mean(np.log(C[noiseSamples]))
            stdC = np.std(np.log(C[noiseSamples]))

            # Compute the energy curve (a la Jinnai et al. 2012)
            # using different M values, for a single node.
            for indexM in range(len(MList)):
                # Compute the number of samples in a window -- species specific
                # Convert M to number of WCs
                M = int(MList[indexM] * win_sr/samples_wc)
                E = ce.EnergyCurve(C, M)
                # for each sliding window, find largest E
                start = 0
                for j in range(nw):
                    end = min(N, int(start + win_sr/samples_wc))
                    # NOTE: here we determine the statistic (mean/max...) for detecting calls
                    maxE[indexM, j, nodenum] = (np.log(np.mean(E[start:end])) - meanC) / stdC
                    start += int(inc_sr/samples_wc)
            nodenum += 1

        del C
        del E
        gc.collect()
        return maxE

    def detectCalls(self, wf, nodelist, samplerate, subfilter, rf=True, annotation=None, window=1, inc=None, aa=True):
        """
        For both TRAIN and NON_TRAIN modes
        Regenerates the signal from the node and threshold.
        Args:
        1. wf - WaveletFunctions with a homebrew wavelet tree (list of ndarray nodes)
        2. nodelist - will reconstruct signal and run detections on each of these nodes separately
        3. samplerate
        4. subfilter - used to pass thr, M, and other parameters
        5. rf - bandpass to species freq range?
        6. annotation - for calculating noise properties during training
        7-8. window, inc
        9. antialias - True/False

        Return: ndarray of 1/0 annotations for each of T windows
        """

        # Virginia: this function now works with OVERLAPPING sliding windows
        # Added window and increment input.
        # Window is window length in seconds
        # inc is increment length in seconds
        # Changed detection to work with sliding overlapping window
        # It compute energy in "centered" window.

        # Virginia: if no increment I set it equal to window
        if inc is None:
            inc = window
            resol = window
        else:
            resol = (math.gcd(int(100 * window), int(100 * inc))) / 100

        duration = len(wf.tree[0])

        # Virginia: added window sample rate
        win_sr = math.ceil(window * samplerate)
        # Increment length in samples
        inc_sr = math.ceil(inc * samplerate)
        # Resolution length in samples
        resol_sr = math.ceil(resol * samplerate)

        thr = subfilter['WaveletParams'][0]
        # Compute the number of samples in a window -- species specific
        # Virginia: changed sampleRate with win_sr
        M = int(subfilter['WaveletParams'][1] * win_sr)
        nw = int(np.ceil(duration / inc_sr))
        detected = np.zeros((nw, len(nodelist)))
        count = 0
        for node in nodelist:
            # put WC from test node(s) on the new tree
            C = wf.reconstructWP2(node, antialias=aa, antialiasFilter=True)
            # Sanity check for all zero case
            if not any(C):
                continue    # return np.zeros(nw)

            if len(C) > duration:
                C = C[:duration]

            # Filter
            if rf:
                C = self.sp.ButterworthBandpass(C, win_sr, low=subfilter['FreqRange'][0],
                                                high=subfilter['FreqRange'][1])
            C = np.abs(C)
            N = len(C)
            # Virginia: number of segments = number of centers of length inc
            # nw=int(np.ceil(N / inc_sr))
            # detected = np.zeros(nw)

            # Compute the energy curve (a la Jinnai et al. 2012)
            E = ce.EnergyCurve(C, M)
            # Compute threshold using mean & sd from non-call sections
            # Virginia: changed the base. I'm using resol_sr as a base. Cause I'm looking for detections on windows.
            #This step is not so clear for me
            if annotation is not None:
                noiseSamples = np.repeat(annotation == 0, resol_sr)
                noiseSamples = noiseSamples[:len(C)]
                C = C[noiseSamples]
            C = np.log(C)
            threshold = np.exp(np.mean(C) + np.std(C) * thr)

            # If there is a call anywhere in the window, report it as a call
            # Virginia-> for each sliding window:
            # start is the sample start of a window
            # end is the sample end of a window
            # The window are sliding windows: starting from data start
            #center = int(math.ceil(inc_sr/2)) #keeped if neede in future
            start = 0
            for j in range(nw):
                #start = max(0, center - win_sr2) keeped if needed in future
                end = min(N, start + win_sr)
                # max/ mean/median
                # detected[j, count] = np.any(E[start:end] > threshold)
                # mean
                # inds2use = np.intersect1d(np.arange(start, end), indstouse)
                if np.mean(E[start:end]) > threshold:
                    detected[j, count] = 1
                # if len(inds2use)== 0:
                #     detected[j, count] = 0
                # elif np.mean(E[inds2use]) > threshold:
                #     detected[j, count] = 1
                # # median
                # if np.median(E[start:end]) > threshold:
                #     detected[j, count] = 1
                start += inc_sr  # Virginia: corrected
            count += 1

        detected = np.max(detected, axis=1)

        # Virginia: caution. Annotation are 1-sec segments and also detections
        # Otherwise no comparison make sense
        # So I have to put them on a one second scale

        if window != 1 or inc != window:
            N = int(math.ceil(duration/ samplerate))  # numbers of seconds
            N = int(math.ceil(duration/ spInfo['SampleRate']))  # numbers of seconds
            detect_ann = np.zeros(N)
            start = 0
            # follow the windows checking in what second they start or end
            for i in range(nw):
                if detected[i]==1:
                    end= min(math.ceil(start + 1), N)
                    detect_ann[int(math.floor(start)):end] = 1
                start += inc
            detected = detect_ann

        del C
        del E
        gc.collect()
        return detected

    def gridSearch(self, E, thrList, MList, rf=True, learnMode=None, window=1, inc=None, wind=False):
        """ Take list of energy peaks of dimensions:
            [files] [MListxTxN ndarrays],
            perform grid search over thr and M parameters,
            do a stepwise search for best nodes for detecting calls.
            In turn, calls are detected when the peaks exceed thrList (provided peaks can be max, mean...)
            Output structure:
            1. 2d list of [nodes]
                (1st d runs over M, 2nd d runs over thr)
            2-5. 2d np arrays of TP/FP/TN/FN
        """
        # Virginia changes
        # Added window and increment input
        # Window = window length in seconds
        # inc= increment length in seconds
        # If there is inc I call other functions to work with overlapping windows

        shape = (len(MList), len(thrList))
        tpa = np.zeros(shape)
        fpa = np.zeros(shape)
        tna = np.zeros(shape)
        fna = np.zeros(shape)
        finalnodes = []
        negative_nodes = []
        top_nodes = []

        # Grid search over M x thr x Files
        for indexM in range(len(MList)):
            finalnodesT = []
            for indext in range(len(thrList)):
                # Accumulate nodes for the set of files for this M and thr
                finalnodesMT = []
                detected_all = []
                annot_all = []
                # loop over files:
                for indexF in range(len(E)):
                    # Best nodes found within this file:
                    finalnodesF = []
                    bestBetaScore = 0
                    bestRecall = 0

                    EfileM = E[indexF][indexM,:,:]
                    nodesToTest = self.bestNodes[indexF]
                    # load the annotations for this file
                    if (inc is not None and inc!=1) or window!=1:
                        annot = self.annotation2[indexF]
                    else:
                        annot = self.annotation[indexF]

                    ### STEPWISE SEARCH for best node combination:
                    # (try to detect using thr, add node if it improves F2)
                    print("Starting stepwise search. Possible nodes:", nodesToTest)
                    detect_best = np.zeros(len(EfileM[:,0]))
                    for nodenum in range(len(nodesToTest)):
                        print("Testing node ", nodesToTest[nodenum])
                        detect_onenode = EfileM[:,nodenum] > thrList[indext]

                        if window != 1 or inc is not None:
                            if inc is None:
                                inc2 = window
                            else:
                                inc2 = inc
                            # detected length is equal to the number of windows.
                            N = len(annot)
                            detect_ann = np.zeros(N)
                            start = 0
                            # map detect_onenode to non-standard annotation windows
                            for i in range(len(detect_onenode)):
                                if detect_onenode[i] == 1:
                                    end = int(min(math.ceil(start + 1), N))
                                    detect_ann[int(math.floor(start)):end] = 1
                                start += inc2
                            detect_onenode = detect_ann

                        # exclude detections in noisy sections
                        if wind:
                            detect_onenode = np.minimum.reduce([detect_onenode, self.noiseList[indexF]])

                        # What do we detect if we add this node to currently best detections?
                        detect_allnodes = np.maximum.reduce([detect_best, detect_onenode])
                        fB, recall, tp, fp, tn, fn = self.fBetaScore(annot, detect_allnodes)

                        # If this node improved fB,
                        # store it and update fB, recall, best detections, and optimum nodes
                        if fB is not None and fB > bestBetaScore:
                            bestBetaScore = fB
                            bestRecall = recall
                            detect_best = detect_allnodes
                            finalnodesF.append(nodesToTest[nodenum])
                        # Adding more nodes will not reduce FPs, so this is sufficient to stop:
                        # Stopping a bit earlier to have fewer nodes and fewer FPs:
                        if bestBetaScore == 0.95 or bestRecall == 0.95:
                            break

                    # Store the best nodes for this file
                    finalnodesMT.append(finalnodesF)
                    print("Iteration f %d/%d complete" % (indexF + 1, len(self.audioList)))

                    ##### TODO clean this once confirmed that it works
                    nodesToTest, worstnodes = self.listTopNodes(indexF)
                    if np.sum(annot) > 0:
                        top_nodes.extend(nodesToTest[0:2])
                    negative_nodes.extend(worstnodes)
                    print("Adding to negative nodes", worstnodes)

                    # Memory cleanup:
                    gc.collect()


                    # build long vectors of detections and annotations
                    detected_all.extend(detect_best)
                    annot_all.extend(annot)

                # One iteration done, store results
                finalnodesMT = [y for x in finalnodesMT for y in x]
                finalnodesMT = list(set(finalnodesMT))
                finalnodesT.append(finalnodesMT)
                # Get the measures with the selected node set for this threshold and M over the set of files
                # TODO check if this needs fixing for non-standard window and inc (used to use self.annotation2?)
                fB, recall, tp, fp, tn, fn = self.fBetaScore(annot_all, detected_all)
                tpa[indexM, indext] = tp
                fpa[indexM, indext] = fp
                tna[indexM, indext] = tn
                fna[indexM, indext] = fn
                print("Iteration t %d/%d complete\t thr=%f\n---------------- " % (indext + 1, len(thrList), thrList[indext]))
            # One row done, store nodes
            finalnodes.append(finalnodesT)
            print("Iteration M %d/%d complete\t M=%f\n-----------------------------------------------------------------"
                  "\n-----------------------------------------------------------------" % (indexM + 1, len(MList), MList[indexM]))
        # remove duplicates
        negative_nodes = set(negative_nodes)
        negative_nodes = list(negative_nodes)
        # Remove any top nodes from negative list
        negative_nodes = [i for i in negative_nodes if i not in top_nodes]
        # Convert negative correlated nodes
        negative_nodes = [n + 1 for n in negative_nodes]
        # Remove any negatively correlated nodes
        print("Final nodes before neg. node removal:", finalnodes)
        print("Negative nodes:", negative_nodes)
        finalnodes = [[[item for item in sublst if item not in negative_nodes] for sublst in lst] for lst in finalnodes]
        return finalnodes, tpa, fpa, tna, fna


    def listTopNodes(self, filenum):
        """ Selects top 10 or so nodes to be tested for this file,
            using correlations stored in nodeCorrs, and provided file index.

            Return: tuple of lists (bestnodes, worstnodes)
        """

        # Retrieve stored node correlations
        nodeCorrs = self.nodeCorrs[filenum]
        nodes1 = np.flip(np.argsort(nodeCorrs)[:], 0).tolist()
        bestnodes = []

        # avoid low-level nodes
        low_level_nodes = list(range(15))
        for item in nodes1:
            if item not in low_level_nodes:
                bestnodes.append(item)

        # Find 10 most positively correlated nodes
        bestnodes = bestnodes[0:10]

        # Keep track of negative correlated nodes
        worstnodes = np.argsort(nodeCorrs)[:10]
        # # Avoid having any node in the first half of the positive nodes as a neg node
        # negative_nodes = [i for i in negative_nodes if i not in nodes[0:5]]

        # Sort the nodes, put any of its children (and their children, iteratively) that are in the list in front of it
        bestnodes = self.sortListByChild(bestnodes)

        # These nodes refer to the un-rooted tree, so add 1 to get the real WP indices
        bestnodes = [n + 1 for n in bestnodes]

        # TODO disabled until we review node ranges
        # # Lastly remove any node beyond the target frq range, last two levels, a sanity check
        # r1 = np.linspace(0, self.spInfo['SampleRate'] / 2, 16)
        # node_low = (np.abs(r1 - self.spInfo['FreqRange'][0])).argmin() + 15
        # node_high = (np.abs(r1 - self.spInfo['FreqRange'][1])).argmin() + 15
        # nr1 = np.arange(node_low, node_high).tolist()
        # r2 = np.linspace(0, self.spInfo['SampleRate']/2, 32)
        # node_low = (np.abs(r2 - self.spInfo['FreqRange'][0])).argmin() + 31
        # node_high = (np.abs(r2 - self.spInfo['FreqRange'][1])).argmin() + 31
        # nr2 = np.arange(node_low, node_high).tolist()
        # node_range = nr1 + nr2
        # bestnodes = [x for x in bestnodes if x in node_range]

        return (bestnodes, worstnodes)

    def preprocess(self, data, sampleRate, d=False, fastRes=True):
        """ Downsamples, denoises, and filters the data.
            sampleRate - actual sample rate of the input. Will be resampled based on spInfo.
            d - boolean, perform denoising?
            fastRes - use kaiser_fast instead of best. Twice faster but pretty similar output.
        """
        # set target sample rate:
        fsOut = self.spInfo['SampleRate']

        if sampleRate != fsOut:
            print("Resampling from", sampleRate, "to", fsOut)
            if fastRes:
                data = librosa.core.audio.resample(data, sampleRate, fsOut, res_type='kaiser_fast')
            else:
                data = librosa.core.audio.resample(data, sampleRate, fsOut, res_type='kaiser_best')
            sampleRate = fsOut

        # Get the five level wavelet decomposition
        if d:
            WF = WaveletFunctions.WaveletFunctions(data=data, wavelet=self.wavelet, maxLevel=20, samplerate=fsOut)
            denoisedData = WF.waveletDenoise(thresholdType='soft', maxLevel=5)
        else:
            denoisedData = data  # this is to avoid washing out very fade calls during the denoising

        WF = []
        del WF
        return denoisedData

    def loadDirectory(self, dirName, denoise, window=1, inc=None):
        """
            Finds and reads wavs from directory dirName.
            Denoise arg is passed to preprocessing.
            wpmode selects WP decomposition function ("new"-our but not AA'd, "aa"-our AA'd)

            Results: self.annotation, self.audioList, self.noiseList arrays.
        """
        #Virginia: if no inc I set resol equal to window, otherwise it is equal to inc
        if inc is None:
            inc = window
            resol = window
        else:
            resol = (math.gcd(int(100 * window), int(100 * inc))) / 100

        self.annotation = []
        self.audioList = []

        for root, dirs, files in os.walk(str(dirName)):
            for file in files:
                if file.endswith('.wav') and os.stat(root + '/' + file).st_size != 0 and file[:-4] + '-' + self.spInfo["Filters"][0]["calltype"] + '-res'+str(float(resol))+'sec.txt' in files:
                    opstartingtime = time.time()
                    wavFile = os.path.join(root, file[:-4])

                    # adds to self.annotation array
                    data, sampleRate = self.loadData(wavFile, window, inc, resol)

                    # denoise and store actual audio data:
                    # note: preprocessing is a side effect on data
                    # (preprocess only reads target SampleRate from spInfo)
                    denoisedData = self.preprocess(data, sampleRate, d=denoise)
                    self.audioList.append(denoisedData)

                    print("file loaded in", time.time() - opstartingtime)

        if len(self.annotation) == 0 or len(self.audioList) == 0:
            print("ERROR: no files loaded!")
            return

        # Prepare WC data and annotation targets into a matrix for saving
        # WC = np.transpose(self.waveletCoefs)
        # ann = np.reshape(self.annotation, (len(self.annotation), 1))
        # MLdata = np.append(WC, ann, axis=1)
        # np.savetxt(os.path.join(dirName, "energies.tsv"), MLdata, delimiter="\t")
        del data
        del denoisedData
        totalcalls = sum([sum(a) for a in self.annotation])
        totalblocks = sum([len(a) for a in self.annotation])
        print("Directory loaded. %d/%d presence blocks found.\n" % (totalcalls, totalblocks))
        print(np.shape(self.annotation))


    def loadData(self, fName, window, inc, resol):
        """ Loads a single file.
            Input: fName - filestem for wav and annotation files
            Output: fills self.annotation, returns data, samplerate
        """
        # Virginia chamges
        # Added resol input as basic unit for read annotation file
        filename = fName + '.wav'
        print('\n\n', filename)
        # Virginia: added resol for identify annotation txt
        filenameAnnotation = fName + '-' + self.spInfo["Filters"][0]["calltype"] + '-res' + str(float(resol)) + 'sec.txt'
        # filenameAnnotation = fName + '-res'+str(float(resol))+'sec.txt'
        try:
            wavobj = wavio.read(filename)
        except Exception as e:
            print("unsupported file: ", filename)
            print("encountered exception: ", e)
            pass

        sampleRate = wavobj.rate
        data = wavobj.data
        if data.dtype != 'float':
            data = data.astype('float')  # / 32768.0
        if np.shape(np.shape(data))[0] > 1:
            data = np.squeeze(data[:, 0])
        #Virginia-> number of entries in annotation file: built on resol scale
        n = math.ceil((len(data) / sampleRate)/resol)

        # # Impulse masking
        # postp = SupportClasses.postProcess(audioData=data, sampleRate=sampleRate, segments=[], spInfo={})
        # imps = postp.impulse_cal(fs=sampleRate)    # 0 - presence of impulse noise, 1 - otherwise
        #
        # # Mask only the affected samples
        # if n - np.sum(imps) > 0:
        #     print('impulse detected: ', n - np.sum(imps), ' samples')
        # data = np.multiply(data, imps)

        fileAnnotations = []
        # Get the segmentation from the txt file
        with open(filenameAnnotation) as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)
        if d[-1] == []:
            d = d[:-1]
        if len(d) != n:
            print("ERROR: annotation length %d does not match file duration %d!" % (len(d), n))
            self.annotation = None
            return

        # for each second, store 0/1 presence:
        presblocks = 0
        for row in d:
            fileAnnotations.append(int(row[1]))
            presblocks += int(row[1])

        self.annotation.append(np.array(fileAnnotations))

        # #Virginia: if overlapping window or window!=1sec I save in annotation2 segments o length 1 sec to make useful comparison
        # if window != 1 or inc != window:
        #     N = int(math.ceil(len(data)/sampleRate)) # of seconds
        #     annotation_sec = np.zeros(N)
        #     sec_step = int(math.ceil(1/resol)) # window length in resolution scale
        #     #inc_step = int(math.ceil(inc / resol)) #increment length in resolution scale
        #     start = 0
        #     for i in range(N):
        #         end=int(min(start+sec_step,n))
        #         if np.count_nonzero(fileAnnotations[start:end])!=0:
        #             annotation_sec[i]=1
        #         start += sec_step
        #     self.annotation2.append(np.array(annotation_sec))

        totalblocks = sum([len(a) for a in self.annotation])
        print( "%d blocks read, %d presence blocks found. %d blocks stored so far.\n" % (n, presblocks, totalblocks))
        return data, sampleRate

    def identifySegments(self, detection):  # , maxgap=1, minlength=1):
        """ Turn binary detection to segments """
        # TODO: *** Replace with segmenter.checkSegmentLength(self,segs, mingap=0, minlength=0, maxlength=5.0)
        segments = []
        # print seg, type(seg)
        if len(detection) > 0:
            for s in detection:
                segments.append([s, s + 1])
        return segments

    def mergeSeg(self, segments):
        """ Merge the neighbouring segments """
        # **** Replace with segmenter.identifySegments(self, seg, maxgap=1, minlength=1,notSpec=False):
        # but note the order of deleting short segments and merging matters
        indx = []
        for i in range(len(segments) - 1):
            if segments[i][1] == segments[i + 1][0]:
                indx.append(i)
        indx.reverse()
        for i in indx:
            segments[i][1] = segments[i + 1][1]
            del (segments[i + 1])
        return segments

