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
        self.sampleRate = 0

        self.sp = SignalProc.SignalProc([], 0, 256, 128)
        self.segmenter = Segment.Segment(None, None, self.sp, 0, window_width=256, incr=128, mingap=mingap, minlength=minlength)

    def waveletSegment(self, data, sampleRate, d, f, wpmode="new"):
        """ Main analysis wrapper (segmentation in batch mode).
            Args:
            1. data to be segmented, ndarray
            2. sampleRate of the data, int
            3-4. d,f - turn on denoising/filtering before calling?
            5. wpmode: old/new/aa to indicate no/partial/full antialias

            Returns: list of segments found
        """
        if data is None or data==[] or len(data)==0:
            print("ERROR: data must be provided for WS")
            return

        opst = time.time()
        self.sampleRate = sampleRate
        filteredDenoisedData = self.preprocess(data, d=d, f=f)

        # Generate a full 5 level wavelet packet decomposition (stored in WF.tree)
        self.WF = WaveletFunctions.WaveletFunctions(data=filteredDenoisedData, wavelet=self.wavelet, maxLevel=20, samplerate=sampleRate)
        if wpmode == "pywt":
            print("ERROR: pywt wpmode is deprecated, use new or aa")
            return
        if wpmode == "new" or wpmode == "old":
            self.WF.WaveletPacket(mode='symmetric', maxlevel=5, antialias=False)
        if wpmode == "aa":
            self.WF.WaveletPacket(mode='symmetric', maxlevel=5, antialias=True, antialiasFilter=True)
        # No need to store everything:
        goodnodes = self.spInfo['WaveletParams'][2]
        print("ws  ch1", time.time() - opst)

        # set other nodes to 0
        for ni in range(len(self.WF.tree)):
            # note that we don't reset node 0 as it's good to keep original data
            if ni not in goodnodes and ni!=0:
                self.WF.tree[ni] = [0]

        # Segment detection and neighbour merging
        detected = self.detectCalls(self.WF, nodelist=goodnodes, spInfo=self.spInfo, rf=True, aa=wpmode!="old")
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
        print("ws  ch2", time.time() - opst)
        return detected

    #Virginia: I noticed you added the input wavelet but it is not used in the function and it is not consistent with the
    #function call in AviaNZ.py. Thus i removed it!
    def waveletSegment_train(self, dirName, thrList, MList, d=False, f=False, rf=True, learnMode='recaa', window=1, inc=None):
        """ Entry point to use during training, called from AviaNZ.py.
            Switches between various training methods, orders data loading etc.,
            then just passes the arguments to the right training method and returns the results.

            Input: path to directory with wav & wav.data files.
            Return: tuple of arrays (nodes, tp, fp, tn, fn)
        """
        spInfo = self.spInfo

        # Virginia changes
        # input change: added variables window and inc
        # window is window length in sec.
        # inc is increment length in sec.
        # Default values set to window=1 and inc=None

        # for reconstructing filters, all audio currently is stored in RAM
        # ("high memory" mode)
        keepaudio = (learnMode=="recsep" or learnMode=="recmulti" or learnMode=="recaa" or learnMode=="recaafull")
        # recommend using wpmode="new", because it is fast and almost alias-free.
        # Virginia: added window and inc input
        self.loadDirectory(dirName=dirName, denoise=d, filter=f, keepaudio=keepaudio, wpmode="new", savedetections=False, train=True, window=window, inc=inc)

        # Argument _learnMode_ will determine which detectCalls function is used:
        # learnMode=="ethr":
        # "1" - get wavelet node energies, threshold over those
        # learnMode=="recsep":
        # "3" - reconstruct signal from each node individually, threshold over that
        # ("the old way")
        # learnMode=="recmulti":
        # "2" - reconstruct signal from all selected nodes, threshold over that
        # learnMode=="recaa":
        # reconstruct signal from each node individually,
        # using homebrew NOT antialiased WPs, and antialiased reconstruction.
        # learnMode=="recaafull":
        # reconstruct signal from each node individually,
        # using homebrew antialiased WPs (SLOW), and antialiased reconstruction.

        # Start training by generating and storing WPs for all files
        # not sure if this is useful for other modes, but definitely needed for recaafull
        if learnMode == "recold":
            self.tempfiles = self.generateWPs(maxlevel=5, wpmode="old", train=True)
        elif learnMode == "recaa":
            self.tempfiles = self.generateWPs(maxlevel=5, wpmode="new", train=True)
        elif learnMode == "recaafull":
            self.tempfiles = self.generateWPs(maxlevel=5, wpmode="aa", train=True)
        else:
            print("ERROR: learning mode unrecognized")
            return

        # energies are stored in self.waveletCoefs,
        # Or can be read-in from the export file.
        # Virginia: added window and inc input
        res = self.gridSearch(thrList, MList, spInfo, rf, learnMode, window, inc)

        # Release disk space
        for f in self.tempfiles:
            os.remove(f)
        return res

    def waveletSegment_test(self, dirName, listnodes=None, spInfo={}, d=False, f=False, rf=True, withzeros=True, learnMode='recaa', savedetections=False, window=1, inc=None):
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
            self.nodes = spInfo['WaveletParams'][2]
        else:
            self.nodes = listnodes

        # clear storage for multifile processing
        self.annotation = []
        self.audioList = []
        self.filelengths = []
        self.filenames = []
        detected = np.array([])

        self.loadDirectory(dirName=dirName, denoise=d, filter=f, keepaudio=True, wpmode="new",
                           savedetections=savedetections, train=False, window=window, inc=inc)

        # stores WPs as a temp file. Not saving any work here, but cleaner this way
        # (because there's no iterating over these files)
        if learnMode == "recaa":
            self.tempfiles = self.generateWPs(maxlevel=5, wpmode="new", train=False)
        if learnMode == "recaafull":
            self.tempfiles = self.generateWPs(maxlevel=5, wpmode="aa", train=False)

        # remember to convert main structures to np arrays
        self.annotation = np.array(self.annotation)
        print("Testing with %s positive and %s negative annotations" % (np.sum(self.annotation == 1), np.sum(self.annotation == 0)))

        # wavelet decomposition and call detection
        for fileId in range(len(self.audioList)):
            print('Processing file # ', fileId + 1)

            if learnMode == "recsep" or learnMode == "recmulti":
                print("Warning: recsep and recmulti modes deprecated, defaulting to recaa")
                learnMode = "recaa"

            # Read a full 5 level packet decomposition from antialiased results
            print("reading WP from file", self.tempfiles[fileId])
            file = open(self.tempfiles[fileId], 'rb')
            self.WF = pickle.load(file)
            file.close()

            # Using new-style WPs, detect calls:
            if learnMode == 'recaa'or learnMode == "recaafull":
                detected_c = self.detectCalls(self.WF, nodelist=self.nodes, spInfo=spInfo, rf=rf,
                                              annotation=None, window=1, inc=None)
            else:
                print("ERROR: the specified learning mode is not implemented in this function yet")
                return

            detected = np.concatenate((detected, detected_c))
            # Generate .data for this file
            # Merge neighbours in order to convert the detections into segments
            # Note: detected np[0 1 1 1] becomes [[1,3]]
            if savedetections:
                detected_c = np.where(detected_c > 0)
                if np.shape(detected_c)[1] > 1:
                    detected_c = self.identifySegments(np.squeeze(detected_c))
                elif np.shape(detected_c)[1] == 1:
                    detected_c = np.array(detected_c).flatten().tolist()
                    detected_c = self.identifySegments(detected_c)
                else:
                    detected_c = []
                detected_c = self.mergeSeg(detected_c)
                for item in detected_c:
                    item[0] = int(item[0])
                    item[1] = int(item[1])
                    item = item.append(spInfo['FreqRange'][0])
                for item in detected_c:
                    item = item.append(spInfo['FreqRange'][1])
                for item in detected_c:
                    item = item.append(spInfo['Name'])
                file = open(str(self.filenames[fileId]) + '.data', 'w')
                json.dump(detected_c, file)

            # memory cleanup:
            gc.collect()

        fB, recall, TP, FP, TN, FN = self.fBetaScore(self.annotation, detected)
        print("Releasing disk space")
        for f in self.tempfiles:
            os.remove(f)
        return detected, TP, FP, TN, FN

    # Virginia: this function to work with sliding windows
    def computeWaveletEnergy(self, data=None, sampleRate=0, nlevels=5, wpmode="new", window=1, inc=None, resol=1):
        """ Computes the energy of the nodes in the wavelet packet decomposition
        Args:
        1. data (waveform)
        2. sample rate
        3. max levels for WP decomposition
        4. WP style ("new"-our non-downsampled, "aa"-our fully AA'd)
        There are 62 coefficients up to level 5 of the wavelet tree (without root), and 300 seconds [N sliding window] in 5 mins
        Hence coefs would then be a 62*300 matrix [62*N matrix]
        The energy is the sum of the squares of the data in each node divided by the total in that level of the tree as a percentage.
        """

        # Virginia changes:
        # Added window and inc input
        # Window is window length in sec.
        # Inc is increment length in sec.
        # Energy is calculated on sliding windows
        # the window is a "centered" window

        #sanity check if called from AviaNZ.py
        if inc==None and window==1:
            inc=window
            resol=window

        if data is None:
            print("ERROR: data needs to be specified")
            return

        #Virginia: number of samples in window
        win_sr=int(math.ceil(window*sampleRate))
        # half-window length in samples
        #win_sr2=int(math.ceil(win_sr/2))
        #Virginia: number of sample in increment
        inc_sr=math.ceil(inc*sampleRate)
        #Virginia: number of samples in resolution
        resol_sr = math.ceil(resol * sampleRate)
        #Virginia: needed to generate coef of the same size of annotations
        step=int(inc/resol)

        #Virginia:number of windows = number of sliding window at resol distance
        N=int(math.ceil(len(data)/resol_sr))

        #Virginia: changed columns dimension -> must be equal to number of sliding window
        coefs = np.zeros((2 ** (nlevels + 1) - 2, N))

        #Virginia-> for each sliding window:
        # start is the sample start of a window
        #end is the sample end of a window
        #We are working with sliding windows starting from the file start
        start=0 #inizialization
        #Virginia: the loop works on the resolution scale to adjust with annotations
        for t in range(0,N,step):
            E = []
            end = min(len(data), start+win_sr)
            # generate a WP
            WF = WaveletFunctions.WaveletFunctions(data=data[start:end], wavelet=self.wavelet, maxLevel=20, samplerate=sampleRate)
            if wpmode == "pywt":
                print("ERROR: pywt mode deprecated, use new or aa")
                return
            if wpmode == "new":
                wp = WF.WaveletPacket(mode='symmetric', maxlevel=nlevels, antialias=False)
            if wpmode == "aa":
                wp = WF.WaveletPacket(mode='symmetric', maxlevel=nlevels, antialias=True, antialiasFilter=True)

            # Calculate energies
            for level in range(1, nlevels + 1):
                lvlnodes = wp[2 ** level - 1:2 ** (level + 1) - 1]
                e = np.array([np.sum(n ** 2) for n in lvlnodes])
                if np.sum(e) > 0:
                    e = 100.0 * e / np.sum(e)
                E = np.concatenate((E, e), axis=0)
            #Virginia:update start
            start+=inc_sr # Virginia: corrected
            for T in range(t,t+step):
                coefs[:, T] = E
        return coefs


    def fBetaScore(self, annotation, predicted, beta=2):
        """ Computes the beta scores given two sets of predictions """
        #print('fBetaScore')
        #print((len(annotation),len(predicted)))
        print('Annotation length ', len(annotation))
        print('Predicted length ', len(predicted))
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


    def sortListByChild2(self, order):
        # Virginia's Version
        #it still needs test
        # It uses only the father to find the childre
        """ Inputs is a list sorted into order of correlation.
        This functions resort so that any children of the current node that are in the list go first.
        Assumes that there are five levels in the tree (easy to extend, though)
        """
        newlist = []
        currentIndex = 0
        # Need to keep track of where each level of the tree starts
        # Note that there is no root to the tree, hence the 0 then 2
        #starts = [0, 2, 6, 14, 30, 62]
        while len(order) > 0:
            if order[0] < 30:
                # It could have children lower down the list
                # Build a list of the children of the first element of order
                level = int(np.log2(order[0] + 2))
                nc = 2
                first = order[0]
                for l in range(level, 5):
                    children = []
                    current = currentIndex
                    for i in range(nc):
                        children.append(2*(first)+ i) #Virginia: find children from father
                    nc *= 2
                    first = 2*first+1 #Update father
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

    def detectCalls(self, wf, nodelist, spInfo, rf=True, annotation=None, window=1, inc=None, aa=True):
        """
        For both TRAIN and NON_TRAIN modes
        Regenerates the signal from the node and threshold.
        Args:
        1. wf - WaveletFunctions with a homebrew wavelet tree (list of ndarray nodes)
        2. nodelist - will reconstruct signal and run detections on each of these nodes separately
        3. spInfo - used to pass thr, M, and other parameters
        4. rf - ??
        5. annotation - for calculating noise properties during training
        6-7. window, inc - ??
        8. antialias - True/False

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
        win_sr = math.ceil(window * self.sampleRate)
        # Increment length in samples
        inc_sr = math.ceil(inc * self.sampleRate)
        # Resolution length in samples
        resol_sr = math.ceil(resol * self.sampleRate)

        thr = spInfo['WaveletParams'][0]
        # Compute the number of samples in a window -- species specific
        # Virginia: changed sampleRate with win_sr
        M = int(spInfo['WaveletParams'][1] * win_sr)
        nw = int(np.ceil(duration / inc_sr))
        detected = np.zeros((nw, len(nodelist)))

        count = 0
        for node in nodelist:
            # put WC from test node(s) on the new tree
            C = wf.reconstructWP2(node,antialias=aa, antialiasFilter=True)
            # Sanity check for all zero case
            if not any(C):
                continue    # return np.zeros(nw)

            if len(C) > duration:
                C = C[:duration]

            # Filter
            if rf:
                C = self.sp.ButterworthBandpass(C, self.sampleRate, low=spInfo['FreqRange'][0],
                                                high=spInfo['FreqRange'][1], order=10)
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
                C = C[np.repeat(annotation == 0, resol_sr)]
            C = np.log(C)
            threshold = np.exp(np.mean(C) + np.std(C) * thr)

            # If there is a call anywhere in the window, report it as a call
            # Virginia-> for each sliding window:
            # start is the sample start of a window
            # end is the sample end of a window
            # The window are sliding windows: starting from data start
            #center = int(math.ceil(inc_sr/2)) #keeped if neede in future
            start = 0 #inizializzation
            for j in range(nw):
                #start = max(0, center - win_sr2) keeped if needed in future
                end = min(N, start + win_sr)
                detected[j, count] = np.any(E[start:end] > threshold)
                start += inc_sr  # Virginia: corrected
            count += 1

        detected = np.max(detected, axis=1)

        # Virginia: caution. Annotation are 1-sec segments and also detections
        # Otherwise no comparison make sense
        # So I have to put them on a one second scale

        if window != 1 or inc != window:
            N = int(math.ceil(duration/ self.sampleRate))  # numbers of seconds
            detect_ann = np.zeros(N)
            start=0
            #I follow the windows checking in what second they start or end
            for i in range(nw):
                if detected[i]==1:
                    end= min(math.ceil(start + 1), N)
                    detect_ann[int(math.floor(start)):end] = 1
                start+=inc
            detected=detect_ann

        del C
        del E
        gc.collect()
        return detected

    def gridSearch(self, thrList, MList, rf=True, learnMode=None, window=1, inc=None):

        """ Take list of files and other parameters,
             load files, compute wavelet coefficients and reuse them in each (M, thr) combination,
             perform grid search over thr and M parameters,
             do a stepwise search for best nodes.
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
        # avoid low-level nodes
        low_level_nodes = list(range(15))

        spInfo = self.spInfo

        # Grid search over M x thr x Files
        for indexM in range(len(MList)):
            finalnodesT = []
            M = MList[indexM]
            for indext in range(len(thrList)):
                thr = thrList[indext]
                spInfo['WaveletParams'] = [thr, M]
                # Accumulate nodes for the set of files for this M and thr
                nodesacc = []
                detected_all = []
                # loop over files:
                # Virginia: needs to distinguish filelength in both cases
                if inc!=None or window!=1:
                    file_lengths=self.filelengths2
                else:
                    file_lengths=self.filelengths
                print('File Lengths', file_lengths)

                for indexF in range(len(file_lengths)):
                    # load the annotation and WCs for this file
                    if inc!=None or window!=1:
                        annotation2 = self.annotation2[int(np.sum(self.filelengths2[0:indexF])):int(np.sum(self.filelengths2[0:indexF + 1]))]
                    annotation = self.annotation[int(np.sum(self.filelengths[0:indexF])):int(np.sum(self.filelengths[0:indexF + 1]))]

                    # Find 10 most positively correlated nodes
                    nodeCorrs = self.nodeCorrs[:, indexF]
                    nodes1 = np.flip(np.argsort(nodeCorrs)[:], 0).tolist()
                    nodes = []
                    for item in nodes1:
                        if item not in low_level_nodes:
                            nodes.append(item)
                    nodes = nodes[0:10]

                    # Keep track of negative correlated nodes
                    negative_nodes.extend(np.argsort(nodeCorrs)[:10])
                    # # Avoid having any node in the first half of the positive nodes as a neg node
                    # negative_nodes = [i for i in negative_nodes if i not in nodes[0:5]]
                    if np.sum(annotation) > 0:
                        top_nodes.extend(nodes[0:2])

                    # Sort the nodes, put any of its children (and their children, iteratively) that are in the list in front of it
                    nodes = self.sortListByChild(nodes)

                    # These nodes refer to the un-rooted tree, so add 1 to get the real indices
                    nodes = [n + 1 for n in nodes]

                    # Now check the F2 values and add node if it improves F2
                    listnodes = []
                    bestBetaScore = 0
                    bestRecall = 0

                    #detected = np.zeros(self.filelengths[indexF])
                    detected = np.zeros(file_lengths[indexF])

                    ### GET WPs for reconstructing detectors:
                    #Changed to read convenient filelength
                    #wp = []

                    # Read a full 5 level packet decomposition from antialiased results
                    if learnMode == "recaafull" or learnMode == "recaa":
                        print("reading WP from file", self.tempfiles[indexF])
                        file = open(self.tempfiles[indexF], 'rb')
                        self.WF = pickle.load(file)
                        file.close()
                    elif learnMode == "recsep" or learnMode =="recmulti":
                        print("ERROR: pywt-based modes deprecated, use recaa or recaafull")
                        return
                    elif learnMode == "ethr" or learnMode == "elearn":
                        pass
                    else:
                        print("ERROR: unrecognized learning mode", learnMode)
                        return

                    ### STEPWISE SEARCH for best node combination:
                    for node in nodes:
                        testlist = listnodes[:]
                        testlist.append(node)
                        print("Test list: ", testlist)

                        # Detect calls, using signal reconstructed from current node
                        # with antialias (freq squashing + non-downsampled tree, recaa...),
                        # or no reconstruction, just energy-based detection (ethr...)

                        if learnMode == "recaa" or learnMode == "recaafull":
                            detected_c = self.detectCalls(self.WF, nodelist=[node], spInfo=spInfo,
                                                          rf=rf, annotation=annotation, window=window, inc=inc)
                        elif learnMode == "ethr" or learnMode == "elearn":
                            print("ERROR: learning mode ethr not implemented yet")
                            return
                            # TODO
                            # Non-reconstructing detectors:
                            # detected_c = self.detectCalls_en(self.audioList[indexF], self.sampleRate, nodes=testlist)
                        else:
                            print("ERROR: learning mode not recognized")
                            return

                        if window != 1 or inc != None:
                            print('Changing annotations')
                            if inc == None:
                                inc2 = window
                            else:
                                inc2 = inc
                            N = len(annotation2)  # numbers of seconds:  must be equal to annotation length
                            detect_ann = np.zeros(N)
                            start = 0
                            # I follow the windows checking in what second they start or end
                            # detected length is equal to the number of windows. I follow the sliding windows to reconstruct useful annotaions
                            for i in range(len(detected_c)):
                                if detected_c[i] == 1:
                                    end = int(min(math.ceil(start + 1), N))
                                    # Virginia: remember start and end must be integers
                                    detect_ann[int(math.floor(start)):end] = 1
                                start += inc2
                            detected_c = detect_ann

                        #Virginia: I'm supposing that detection are of the same length of annaotation on a window base
                        # adjust for rounding errors:
                        if len(detected_c) < len(detected):
                            detected_c = np.append(detected_c, [0])
                        if len(detected_c) > len(detected):
                            detected_c = detected_c[:len(detected)]

                        # Merge the detections from current node with those from previous nodes
                        detections = np.maximum.reduce([detected, detected_c])

                        #NB: I need to use correct annotation
                        if inc!=None or window!=1:
                            fB, recall, tp, fp, tn, fn = self.fBetaScore(annotation2, detections)
                        else:
                            fB, recall, tp, fp, tn, fn = self.fBetaScore(annotation, detections)
                        if fB is not None and fB > bestBetaScore:  # Keep this node and update fB, recall, detected, and optimum nodes
                            bestBetaScore = fB
                            bestRecall = recall
                            detected = detections
                            listnodes.append(node)
                        if bestBetaScore == 1 or bestRecall == 1:
                            break

                    # Memory cleanup:
                    gc.collect()

                    detected_all = np.concatenate((detected_all, detected))
                    nodesacc.append(listnodes)
                    print("Iteration f %d/%d complete" % (indexF + 1, len(self.filelengths)))

                # One iteration done, store results
                nodesacc = [y for x in nodesacc for y in x]
                nodesacc = list(set(nodesacc))
                finalnodesT.append(nodesacc)
                # Get the measures with the selected node set for this threshold and M over the set of files
                #Virginia: call the appropriate annotation variable
                if window!=1 or inc!= None:
                    fB, recall, tp, fp, tn, fn = self.fBetaScore(self.annotation2, detected_all)
                else:    
                    fB, recall, tp, fp, tn, fn = self.fBetaScore(self.annotation, detected_all)
                tpa[indexM, indext] = tp
                fpa[indexM, indext] = fp
                tna[indexM, indext] = tn
                fna[indexM, indext] = fn
                print("Iteration t %d/%d complete\t thr=%f\n---------------- " % (indext + 1, len(thrList), thr))
            # One row done, store nodes
            finalnodes.append(finalnodesT)
            print("Iteration M %d/%d complete\t M=%f\n----------------\n---------------- " % (indexM + 1, len(MList), M))
        # remove duplicates
        negative_nodes = set(negative_nodes)
        negative_nodes = list(negative_nodes)
        # Remove any top nodes from negative list
        negative_nodes = [i for i in negative_nodes if i not in top_nodes]
        # Convert negative correlated nodes
        negative_nodes = [n + 1 for n in negative_nodes]
        # print("Negative nodes:", negative_nodes)
        # Remove any negatively correlated nodes
        finalnodes = [[[item for item in sublst if item not in negative_nodes] for sublst in lst] for lst in finalnodes]
        return finalnodes, tpa, fpa, tna, fna


    def generateWPs(self, maxlevel, wpmode, train):

        """ Stores WPs of selected nodes for all loaded files.
            Useful for disk-caching when WP decomp is slow.

            Args:
            1. wavelet object
            2. maxlevel
            3. wpmode ("old", "new", "aa")
            Returns a list of file paths
        """
        # For each file:
        files = list()
        for indexF in range(len(self.filelengths)):
            data = self.audioList[indexF]
            self.WF = WaveletFunctions.WaveletFunctions(data=data, wavelet=self.wavelet, maxLevel=20, samplerate=self.sampleRate)
            # Generate a full 5 level wavelet packet decomposition
            if wpmode == "pywt":
                print("ERROR: pywt objects deprecated, cannot store")
                return
            if wpmode == "new" or wpmode =="old":
                self.WF.WaveletPacket(mode='symmetric', maxlevel=maxlevel, antialias=False)
            if wpmode == "aa":
                self.WF.WaveletPacket(mode='symmetric', maxlevel=maxlevel, antialias=True, antialiasFilter=True)

            # No need to store everything:
            # Find 10 most positively correlated nodes in train mode and the optimum nodes from filter in test mode
            # TODO cleanup this argument
            if train:
                nodeCorrs = self.nodeCorrs[:, indexF]
                goodnodes = np.flip(np.argsort(nodeCorrs)[-10:], 0)
                goodnodes = [n + 1 for n in goodnodes]
            else:
                goodnodes = self.nodes

            # set other nodes to 0
            for ni in range(len(self.WF.tree)):
                if ni not in goodnodes:
                    self.WF.tree[ni] = [0]

            # save:
            files.append(os.path.join(tempfile.gettempdir(), "avianz_wp" + str(os.getpid()) + "_" + str(indexF)))
            file = open(files[indexF], 'w+b')
            pickle.dump(self.WF, file)
            file.flush()
            file.close()
            self.WF = []
            print("saved WP to file", files[indexF])

        return (files)

    def preprocess(self, data, d=False, f=False):
        """ Downsamples, denoises, and filters the data.
            d/f - Bools to perform denoise/filtering.
        """
        # set target sample rate:
        fs = self.spInfo['SampleRate']

        if self.sampleRate != fs:
            print("Resampling from", self.sampleRate, "to", fs)
            data = librosa.core.audio.resample(data, self.sampleRate, fs)
            self.sampleRate = fs

        # Get the five level wavelet decomposition
        if d:
            WF = WaveletFunctions.WaveletFunctions(data=data, wavelet=self.wavelet, maxLevel=20, samplerate=fs)
            denoisedData = WF.waveletDenoise(thresholdType='soft', maxLevel=5)
        else:
            denoisedData = data  # this is to avoid washing out very fade calls during the denoising

        if f:
            filteredDenoisedData = self.sp.ButterworthBandpass(denoisedData, self.sampleRate,
                                                               low=self.spInfo['FreqRange'][0], high=self.spInfo['FreqRange'][1])
        else:
            filteredDenoisedData = denoisedData
        WF = []
        del WF
        return filteredDenoisedData

    def loadDirectory(self, dirName, denoise, filter, keepaudio, wpmode,savedetections, train, window=1, inc=None):
        """ (moved out from individual training functions)
            Finds and reads wavs from directory dirName.
            Computes a WP and stores the node energies for each second.
            Computes and stores the WC-annotation correlations.
            Denoise and Filter args are passed to preprocessing.
            keepaudio arg controls whether audio is stored (needed for reconstructing signal).
                Otherwise only energies (matrix of 62 x duration in s) will be stored.
            wpmode selects WP decomposition function ("new"-our but not AA'd, "aa"-our AA'd)

            Results: self.annotation, filelengths, [audioList,] waveletCoefs, nodeCorrs arrays.
            waveletCoefs also exported to a file.

            For filter training and testing, therefore (correlated) nodes have two versions: save correlated nodes in train and use optimum nodes from the filter in test mode
            """
        # Virginia changes:
        # input changed: added window and inc for window's and increment's length in sec.
        # Default values setted as window=1 and inc=None

        #Virginia: if no inc I set resol equal to window, otherwise it is equal to inc
        if inc is None:
            inc=window
            resol = window
        else:
            resol = (math.gcd(int(100 * window), int(100 * inc))) / 100

        nlevels = 5
        self.annotation = []
        self.filelengths = []
        self.audioList = []
        self.waveletCoefs = np.array([]).reshape(2 ** (nlevels + 1) - 2, 0)
        if train:
            self.nodeCorrs = np.array([]).reshape(2 ** (nlevels + 1) - 2, 0)

        for root, dirs, files in os.walk(str(dirName)):
            for file in files:
                if file.endswith('.wav') and os.stat(root + '/' + file).st_size != 0 and file[:-4] + '-res'+str(float(resol))+'sec.txt' in files:
                    opstartingtime = time.time()
                    wavFile = root + '/' + file[:-4]
                    # adds to annotation and filelength arrays, sets self.data:
                    # Virginia: added resol input
                    self.loadData(wavFile, window, inc, resol, savedetections=savedetections, trainPerFile=False)

                    # denoise and store actual audio data:
                    # note: preprocessing is a side effect on self.data
                    # (preprocess only requires SampleRate and FreqRange from spInfo)
                    filteredDenoisedData = self.preprocess(self.data, d=denoise, f=filter)
                    if keepaudio:
                        self.audioList.append(filteredDenoisedData)

                    # Compute energy in each WP node and store
                    # Virginia: added window and inc input
                    if train:
                        currWCs = self.computeWaveletEnergy(filteredDenoisedData, self.sampleRate, 5, wpmode, window=window, inc=inc)
                        self.waveletCoefs = np.column_stack((self.waveletCoefs, currWCs))
                        # Compute all WC-annot correlations and store
                        currAnnot = np.array(self.annotation[-self.filelengths[-1]:])
                        self.nodeCorrs = np.column_stack((self.nodeCorrs, self.compute_r(currAnnot, currWCs)))

                    print("file loaded in", time.time() - opstartingtime)

        if len(self.annotation) == 0:
            print("ERROR: no files loaded!")
            return

        self.annotation = np.array(self.annotation)
        # Prepare WC data and annotation targets into a matrix for saving
        # WC = np.transpose(self.waveletCoefs)
        # ann = np.reshape(self.annotation, (len(self.annotation), 1))
        # MLdata = np.append(WC, ann, axis=1)
        # np.savetxt(os.path.join(dirName, "energies.tsv"), MLdata, delimiter="\t")
        print("Directory loaded. %d/%d presence blocks found.\n" % (np.sum(self.annotation), len(self.annotation)))


    def loadData(self, fName, window, inc, resol, trainPerFile=False, wavOnly=False, savedetections=False):
        """ Loads a single file.
            Output: fills self.annotation, filelengths, filenames, produces self.data
        """
        # Virginia chamges
        # Added resol input as basic unit for read annotation file
        filename = fName + '.wav'  # 'train/kiwi/train1.wav'
        #filename = dName+'/'+fName + '.wav'
        # Virginia: added resol for identify annotation txt
        filenameAnnotation = fName + '-res'+str(float(resol))+'sec.txt'  # 'train/kiwi/train1-res1sec.txt'
        #filenameAnnotation = filename[:-4] + '-res'+str(float(resol))+'sec.txt'  # 'train/kiwi/train1-res1sec.txt'
        try:

            wavobj = wavio.read(filename)
        except Exception as e:
            print("unsupported file: ", filename)
            print("encountered exception: ", e)
            pass

        self.sampleRate = wavobj.rate
        self.data = wavobj.data
        if self.data.dtype != 'float':
            self.data = self.data.astype('float')  # / 32768.0
        if np.shape(np.shape(self.data))[0] > 1:
            self.data = np.squeeze(self.data[:, 0])
        #Virginia-> number of entries in annotation file: built on resol scale
        n = math.ceil((len(self.data) / self.sampleRate)/resol)

        if not wavOnly:
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
            sum = 0
            for row in d:
                fileAnnotations.append(int(row[1]))
                sum += int(row[1])


            # TWO VERSIONS FOR COMPATIBILITY WITH BOTH TRAINING LOOPS:
            if trainPerFile:
                self.annotation = np.array(fileAnnotations)

            else:
                self.annotation.extend(fileAnnotations)
                self.filelengths.append(n)

            #Virginia: if overlapping window or window!=1sec I save in annotation2 segments o length 1 sec to make useful comparison

            if window!=1 or inc!=window:
                print('Storing annotation2')
                N=int(math.ceil(len(self.data)/self.sampleRate)) # of seconds
                annotation_sec=np.zeros(N)
                sec_step=int(math.ceil(1/resol)) # window length in resolution scale
                #inc_step = int(math.ceil(inc / resol)) #increment length in resolution scale
                start=0
                for i in range (N):
                    end=int(min(start+sec_step,n))
                    if np.count_nonzero(fileAnnotations[start:end])!=0:
                        annotation_sec[i]=1
                    start+=sec_step
                if trainPerFile:
                    self.annotation2 = np.array(annotation_sec)

                else:
                    self.annotation2.extend(annotation_sec)
                    self.filelengths2.append(N)

            if savedetections:
                self.filenames.append(fName)
            print( "%d blocks read, %d presence blocks found. %d blocks stored so far.\n" % (n, sum, len(self.annotation)))

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



                #change before push
                #new_dir='D:\Desktop\Documents\Work\Filter Experiment\RURU\Part1\Test10C'
                #new_dir='/home/listanvirg/FilterTest/Ruru/Test10D'
                #new_dir='/home/listanvirg/FilterTest/Kiwi/New/Test14D'
                #new_dir = 'D:\Desktop\Documents\Work\Filter Experiment\KIWI\Ponui\Test2F'
                #new_filename=new_dir+ '/' +fName
                #self.filenames.append(new_filename)
                #self.filenames.append(filename)
            #Virginia:change this?
            #if window!=1 or inc!=window:
                #print("%d blocks read, %d presence blocks found. %d blocks stored so far.\n" % (N, sum, len(self.annotation2)))
            #else:
                #print( "%d blocks read, %d presence blocks found. %d blocks stored so far.\n" % (n, sum, len(self.annotation)))




