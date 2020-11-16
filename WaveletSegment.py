# WaveletSegment.py
# Wavelet Segmentation

# Version 3.0 14/09/20
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2020

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
import librosa
import copy
import numpy as np
import time, os, math, csv, gc
import SignalProc
import Segment
from ext import ce_denoise as ce


class WaveletSegment:
    # This class implements wavelet segmentation for the AviaNZ interface

    def __init__(self, spInfo={}, wavelet='dmey2'):
        self.wavelet = wavelet
        self.spInfo = spInfo
        self.currentSR = 0
        if not spInfo == {}:
            # for now, we default to the first subfilter:
            print("Detected %d subfilters in this filter" % len(spInfo["Filters"]))

        self.sp = SignalProc.SignalProc(256, 128)

    def readBatch(self, data, sampleRate, d, spInfo, wpmode="new"):
        """ File (or page) loading for batch mode. Must be followed by self.waveletSegment.
            Args:
            1. data to be segmented, ndarray
            2. sampleRate of the data, int
            3. d - turn on denoising before calling?
            4. spInfo - List of filters to determine which nodes are needed & target sample rate
            5. wpmode: old/new/aa to indicate no/partial/full antialias
        """
        if data is None or data == [] or len(data) == 0:
            print("ERROR: data must be provided for WS")
            return

        opst = time.time()

        # resample or adjust nodes if needed
        self.spInfo = spInfo

        # target sample rates should be equal over all requested species
        fsOut = set([filt["SampleRate"] for filt in spInfo])
        if len(fsOut)>1:
            print("ERROR: sample rates must match in all selected filters")
            return
        fsOut = fsOut.pop()

        # copy the filters so that the originals wouldn't be affected between files
        # WARNING: this set self.spInfo to be a list [Filters],
        # while usually it is just a single filter. I'm sorry.
        self.spInfo = copy.deepcopy(spInfo)

        # in batch mode, it's worth trying some tricks to avoid resampling
        if fsOut == 2*sampleRate:
            print("Adjusting nodes for upsampling to", fsOut)
            WF = WaveletFunctions.WaveletFunctions(data=[], wavelet='dmey2', maxLevel=1, samplerate=1)
            for filter in self.spInfo:
                for subfilter in filter["Filters"]:
                    subfilter["WaveletParams"]['nodes'] = WF.adjustNodes(subfilter["WaveletParams"]['nodes'], "down2")
            # Don't want to resample again, so fsTarget = fsIn
            fsOut = sampleRate
        elif fsOut == 4*sampleRate:
            print("Adjusting nodes for upsampling to", fsOut)
            # same. Wouldn't recommend repeating for larger ratios than 4x
            WF = WaveletFunctions.WaveletFunctions(data=[], wavelet='dmey2', maxLevel=1, samplerate=1)
            for filter in self.spInfo:
                for subfilter in filter["Filters"]:
                    downsampled2x = WF.adjustNodes(subfilter["WaveletParams"]['nodes'], "down2")
                    subfilter["WaveletParams"]['nodes'] = WF.adjustNodes(downsampled2x, "down2")
            # Don't want to resample again, so fsTarget = fsIn
            fsOut = sampleRate
        # Could also similarly "downsample" by adding an extra convolution, but it's way slower
        # elif sampleRate == 2*fsOut:
        #     # don't actually downsample audio, just "upsample" the nodes needed
        #     WF = WaveletFunctions.WaveletFunctions(data=[], wavelet='dmey2', maxLevel=1, samplerate=1)
        #     for subfilter in self.spInfo["Filters"]:
        #         subfilter["WaveletParams"]['nodes'] = WF.adjustNodes(subfilter["WaveletParams"]['nodes'], "up2")
        #     print("upsampled nodes")
        #     self.spInfo["SampleRate"] = sampleRate

        denoisedData = self.preprocess(data, sampleRate, fsOut, d=d, fastRes=True)

        # Find out which nodes will be needed:
        allnodes = []
        for filt in self.spInfo:
            for subfilter in filt["Filters"]:
                allnodes.extend(subfilter["WaveletParams"]["nodes"])

        # Generate a full 5 level wavelet packet decomposition (stored in WF.tree)
        self.WF = WaveletFunctions.WaveletFunctions(data=denoisedData, wavelet=self.wavelet, maxLevel=20, samplerate=fsOut)
        if wpmode == "pywt":
            print("ERROR: pywt wpmode is deprecated, use new or aa")
            return
        if wpmode == "new" or wpmode == "old":
            self.WF.WaveletPacket(allnodes, mode='symmetric', antialias=False)
        if wpmode == "aa":
            self.WF.WaveletPacket(allnodes, mode='symmetric', antialias=True, antialiasFilter=True)
        print("File loaded in", time.time() - opst)

        # no return, just preloaded self.WF

    def waveletSegment(self, filtnum, wpmode="new"):
        """ Main analysis wrapper (segmentation in batch mode).
            Also do species-specific post-processing.
            Reads data pre-loaded onto self.WF.tree by self.readBatch.
            Args:
            1. filtnum: index of the current filter in self.spInfo (which is a list of filters...)
            2. wpmode: old/new/aa to indicate no/partial/full antialias
            Returns: list of lists of segments found (over each subfilter)-->[[sub-filter1 segments], [sub-filter2 segments]]
        """
        opst = time.time()

        # No resampling here. Will read nodes from self.spInfo, which may already be adjusted

        ### find segments with each subfilter separately
        detected_allsubf = []
        for subfilter in self.spInfo[filtnum]["Filters"]:
            print("Identifying calls using subfilter", subfilter["calltype"])
            goodnodes = subfilter['WaveletParams']["nodes"]

            detected = self.detectCalls(self.WF, nodelist=goodnodes, subfilter=subfilter, rf=True, aa=wpmode!="old")

            # merge neighbours in order to convert the detections into segments
            # note: detected np[0 1 1 1] becomes [[1,3]]
            segmenter = Segment.Segmenter()
            detected = segmenter.convert01(detected)
            detected = segmenter.joinGaps(detected, maxgap=0)
            detected_allsubf.append(detected)
        print("Wavelet segmenting completed in", time.time() - opst)
        return detected_allsubf

    def waveletSegment_train(self, dirName, thrList, MList, d=False, rf=True, learnMode='recaa', window=1,
                             inc=None):
        """ Entry point to use during training, called from DialogsTraining.py.
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
        self.filenames = []
        self.loadDirectory(dirName=dirName, denoise=d, window=window, inc=inc)
        if len(self.annotation) == 0:
            print("ERROR: no files loaded!")
            return

        # 2. find top nodes for each file (self.nodeCorrs)
        nlevels = 5
        # recommend using wpmode="new", because it is fast and almost alias-free.
        wpmode = "new"
        self.nodeCorrs = []
        self.bestNodes = []
        self.worstNodes = []
        self.maxEs = []
        inc = 1

        if len(self.spInfo["Filters"])>1:
            print("ERROR: must provide only 1 subfilter at a time!")
            return
        else:
            subfilter = self.spInfo["Filters"][0]

        # 2a. prefilter audio to species freq range
        for filenum in range(len(self.audioList)):
            self.audioList[filenum] = self.sp.bandpassFilter(self.audioList[filenum],
                                            self.spInfo['SampleRate'],
                                            start=subfilter['FreqRange'][0],
                                            end=subfilter['FreqRange'][1])

        # 2b. actually compute correlations
        for filenum in range(len(self.audioList)):
            print("Computing wavelet node correlations in file", filenum+1)
            currWCs = self.computeWaveletEnergy(self.audioList[filenum], self.spInfo['SampleRate'], nlevels, wpmode, window=window, inc=inc)
            # Compute all WC-annot correlations
            nodeCorr = self.compute_r(self.annotation[filenum], currWCs)
            self.nodeCorrs.append(nodeCorr)
            # find best nodes
            bestnodes, worstnodes = self.listTopNodes(filenum)
            self.bestNodes.append(bestnodes)
            self.worstNodes.append(worstnodes)
            print("Adding to negative nodes", worstnodes)

        # 3. generate WPs for each file and store the max energies
        for filenum in range(len(self.audioList)):
            print("Extracting energies from file", filenum+1)
            data = self.audioList[filenum]

            self.WF = WaveletFunctions.WaveletFunctions(data=data, wavelet=self.wavelet, maxLevel=20, samplerate=self.spInfo['SampleRate'])

            # Generate a full 5 level wavelet packet decomposition
            if learnMode == "recaa" or learnMode == "recold":
                self.WF.WaveletPacket(self.bestNodes[filenum], mode='symmetric', antialias=False)
            elif learnMode == "recaafull":
                self.WF.WaveletPacket(self.bestNodes[filenum], mode='symmetric', antialias=True, antialiasFilter=True)
            else:
                print("ERROR: learnMode unrecognised")
                return

            # find E peaks over possible M (returns [MxTxN])
            maxEsFile = self.extractE(self.WF, self.bestNodes[filenum], MList, aa=learnMode!="recold", window=window, inc=inc, annotation=self.annotation[filenum])
            self.maxEs.append(maxEsFile)
        # self.maxEs now is a list of [files][M][TxN] ndarrays

        # 4. mark calls and learn threshold
        res = self.gridSearch(self.maxEs, thrList, MList, rf, learnMode, window, inc)

        return res

    def waveletSegment_cnn(self, dirName, filter):
        """ Wrapper for segmentation to be used when generating cnn data.
            Should be identical to processing the files in batch mode,
            + returns annotations.
            Does not do any processing besides basic conversion 0/1 -> [s,e].
            Uses 15 min pages, if files are larger than that.

            Args:
            1. directory to process (recursively)
            2. a filter with a single subfilter.

            Return values:
            1. list of (filename, [segments]) over all files and pages
        """

        # constant - files longer than this will be processed in pages
        samplesInPage = 900*16000

        # clear storage for multifile processing
        detected_out = []
        filenames = []
        self.annotation = []

        # find audio files with 0/1 annotations:
        resol = 1.0
        for root, dirs, files in os.walk(dirName):
            for file in files:
                if file.lower().endswith('.wav') and os.stat(os.path.join(root, file)).st_size != 0 and file[
                                                                                                        :-4] + '-res' + str(
                        float(resol)) + 'sec.txt' in files:
                    filenames.append(os.path.join(root, file))
        if len(filenames) < 1:
            print("ERROR: no suitable files")
            return

        for filename in filenames:
            # similar to _batch: loadFile(self.species)
            # sets self.sp.data, self.sp.sampleRate, appends self.annotation
            succ = self.loadData(filename)
            if not succ:
                print("ERROR: failed to load file", filename)
                return

            # (ceil division for large integers)
            numPages = (len(self.sp.data) - 1) // samplesInPage + 1

            for page in range(numPages):
                print("Processing page %d / %d" % (page+1, numPages))
                start = page*samplesInPage
                end = min(start+samplesInPage, len(self.sp.data))
                filelen = math.ceil((end-start)/self.sp.sampleRate)
                if filelen < 2:
                    print("Warning: can't process short file ends (%.2f s)" % filelen)
                    continue

                # read in page and resample as needed
                # will also set self.spInfo with ADJUSTED nodes if resampling!
                self.readBatch(self.sp.data[start:end], self.sp.sampleRate, d=False, spInfo=[filter], wpmode="new")

                # segmentation, same as in batch mode. returns [[sub-filter1 segments]]
                detected_segs = self.waveletSegment(0, wpmode="new")
                if len(filter["Filters"]) > 1:
                    out = []
                    for subfilterdet in detected_segs:
                        for seg in subfilterdet:
                            out.append(seg)
                    detected_out.append((filename, out))
                else:
                    detected_out.append((filename, detected_segs[0]))

        return detected_out

    def computeWaveletEnergy(self, data, sampleRate, nlevels=5, wpmode="new", window=1, inc=1):
        """ Computes the energy of the nodes in the wavelet packet decomposition
        Args:
        1. data (waveform)
        2. sample rate
        3. max levels for WP decomposition
        4. WP style ("new"-our non-downsampled, "aa"-our fully AA'd)
        5-6. window and inc in seconds, as in other functions. NOTE: this does NOT take into account annotation length
        There are 62 coefficients up to level 5 of the wavelet tree (without root!!), and 300 seconds [N sliding window] in 5 mins
        Hence returned coefs would then be a 62*300 matrix [62*N matrix]
        For smaller windows (controlled by window and inc args), returns the energy within each window, so the return has len/inc columns.
        The energy is the sum of the squares of the data in each node divided by the total in that level of the tree as a percentage.
        """

        if data is None or sampleRate is None:
            print("ERROR: data and Fs need to be specified")
            return

        # number of samples in window
        win_sr = int(math.ceil(window*sampleRate))
        # number of sample in increment
        inc_sr = math.ceil(inc*sampleRate)

        # output columns dimension equal to number of sliding window
        N = int(math.ceil(len(data)/inc_sr))
        coefs = np.zeros((2 ** (nlevels + 1) - 2, N))

        # for each sliding window:
        # start is the sample start of a window
        # end is the sample end of a window
        # We are working with sliding windows starting from the file start
        start = 0
        for t in range(N):
            E = []
            end = min(len(data), start+win_sr)
            # generate a WP
            WF = WaveletFunctions.WaveletFunctions(data=data[start:end], wavelet=self.wavelet, maxLevel=20, samplerate=sampleRate)
            if wpmode == "pywt":
                print("ERROR: pywt mode deprecated, use new or aa")
                return
            if wpmode == "new":
                allnodes = range(2 ** (nlevels + 1) - 1)
                WF.WaveletPacket(allnodes, mode='symmetric', antialias=False)
            if wpmode == "aa":
                allnodes = range(2 ** (nlevels + 1) - 1)
                WF.WaveletPacket(allnodes, mode='symmetric', antialias=True, antialiasFilter=True)

            # Calculate energies
            for level in range(1, nlevels + 1):
                lvlnodes = WF.tree[2 ** level - 1:2 ** (level + 1) - 1]
                e = np.array([np.sum(n ** 2) for n in lvlnodes])
                if np.sum(e) > 0:
                    e = 100.0 * e / np.sum(e)
                E = np.concatenate((E, e), axis=0)

            start += inc_sr
            coefs[:,t] = E
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
        # unrooted:
        starts = [0, 2, 6, 14, 30, 62]
        # rooted:
        # starts = [1, 3, 7, 15, 31, 63]
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

        # Window, inrement, and resolution are converted to true seconds based on the tree samplerate
        samplerate = wf.treefs
        win_sr = math.ceil(window * samplerate)
        inc_sr = math.ceil(inc * samplerate)
        resol_sr = math.ceil(resol * samplerate)

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

        C = None
        E = None
        del C
        del E
        gc.collect()
        return maxE

    def detectCalls(self, wf, nodelist, subfilter, rf=True, annotation=None, window=1, inc=None, aa=True):
        """
        For wavelet TESTING and general SEGMENTATION
        Regenerates the signal from the node and threshold.
        Args:
        1. wf - WaveletFunctions with a homebrew wavelet tree (list of ndarray nodes)
        2. nodelist - will reconstruct signal and run detections on each of these nodes separately
        3. subfilter - used to pass thr, M, and other parameters
        4. rf - bandpass to species freq range?
        5. annotation - for calculating noise properties during training
        6-7. window, inc
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

        # To convert window / increment / resolution to samples,
        # we use the actual sampling rate of the tree
        # i.e. "wf.treefs" samples of wf.tree[0] will always correspond to 1 s
        win_sr = math.ceil(window * wf.treefs)
        inc_sr = math.ceil(inc * wf.treefs)
        resol_sr = math.ceil(resol * wf.treefs)

        thr = subfilter['WaveletParams']['thr']
        # Compute the number of samples in a window -- species specific
        # Virginia: changed sampleRate with win_sr
        M = int(subfilter['WaveletParams']['M'] * win_sr)
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
                C = self.sp.bandpassFilter(C, win_sr, subfilter['FreqRange'][0], subfilter['FreqRange'][1])

            C = np.abs(C)
            N = len(C)
            # Virginia: number of segments = number of centers of length inc
            # nw=int(np.ceil(N / inc_sr))
            # detected = np.zeros(nw)

            if len(C) > 2*M+1:
                # Compute the energy curve (a la Jinnai et al. 2012)
                E = ce.EnergyCurve(C, M)
            else:
                break
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
            detect_ann = np.zeros(N)
            start = 0
            # follow the windows checking in what second they start or end
            for i in range(nw):
                if detected[i]==1:
                    end= min(math.ceil(start + 1), N)
                    detect_ann[int(math.floor(start)):end] = 1
                start += inc
            detected = detect_ann

        C = None
        E = None
        del C
        del E
        gc.collect()
        return detected

    def gridSearch(self, E, thrList, MList, rf=True, learnMode=None, window=1, inc=None):
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

                    # In addition to the correlation, re-order nodes according to fB. The order of nodes seems really
                    # important.
                    thisfile_fBs = []
                    for nodenum in range(len(nodesToTest)):
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

                        thisnode_fB, _, _, _, _, _ = self.fBetaScore(annot, detect_onenode)
                        if thisnode_fB:
                            thisfile_fBs.append(thisnode_fB)
                        else:
                            thisfile_fBs.append(0.0)
                    thisfile_node_ix = np.argsort(np.array(thisfile_fBs)).tolist()[::-1]
                    print('thisfile_fBs:%s, nodes:%s' % (str(thisfile_fBs), str(nodesToTest)))

                    ### STEPWISE SEARCH for best node combination:
                    # (try to detect using thr, add node if it improves F2)
                    print("Starting stepwise search. Possible nodes:", list(np.array(nodesToTest)[thisfile_node_ix]))
                    detect_best = np.zeros(len(EfileM[:,0]))
                    for nodenum in thisfile_node_ix:
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

                    # fill top node lists
                    if np.sum(annot) > 0:
                        top_nodes.extend(nodesToTest[0:2])

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
        negative_nodes = np.unique(self.worstNodes)
        # Remove any top nodes from negative list
        negative_nodes = [i for i in negative_nodes if i not in top_nodes]
        # Convert negative correlated nodes
        negative_nodes = [n + 1 for n in negative_nodes]
        # Remove any negatively correlated nodes
        print("Final nodes before neg. node removal:", finalnodes)
        print("Negative nodes:", negative_nodes)
        finalnodes2 = [[[item for item in sublst if item not in negative_nodes] for sublst in lst] for lst in finalnodes]
        # Sanity check
        for i in range(len(finalnodes2)):
            for j in range(len(finalnodes2[i])):
                if len(finalnodes2[i][j]) == 0:
                    finalnodes2[i][j] = finalnodes[i][j]
        return finalnodes2, tpa, fpa, tna, fna


    def listTopNodes(self, filenum):
        """ Selects top 10 or so nodes to be tested for this file,
            using correlations stored in nodeCorrs, and provided file index.

            Return: tuple of lists (bestnodes, worstnodes)
        """
        # Retrieve stored node correlations
        nodeCorrs = self.nodeCorrs[filenum]
        nodes1 = np.flip(np.argsort(nodeCorrs)[:], 0).tolist()
        bestnodes = []

        # filter nodes that are outside target species freq range
        WF = WaveletFunctions.WaveletFunctions(data=[], wavelet=self.wavelet, maxLevel=1, samplerate=self.spInfo["SampleRate"])
        freqrange = [subf["FreqRange"] for subf in self.spInfo["Filters"]]
        freqrange = (np.min(freqrange), np.max(freqrange))

        # avoid low-level nodes
        low_level_nodes = list(range(14))
        for item in nodes1:
            itemfrl, itemfru = WF.getWCFreq(item, self.spInfo["SampleRate"])
            if item not in low_level_nodes and itemfrl < freqrange[1] and itemfru > freqrange[0]:
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
        worstnodes = [n + 1 for n in worstnodes]

        return (bestnodes, worstnodes)

    def preprocess(self, data, sampleRate, fsOut, d=False, fastRes=False):
        """ Downsamples, denoises, and filters the data.
            sampleRate - actual sample rate of the input. Will be resampled based on spInfo.
            fsOut - target sample rate
            d - boolean, perform denoising?
            fastRes - use node-adjusting, or kaiser_fast instead of best. Twice faster but pretty similar output. To use only in batch mode! Otherwise need to deal with returned nodes properly.
        """
        # resample (implies this hasn't been done by node adjustment before)
        if sampleRate != fsOut:
            print("Resampling from", sampleRate, "to", fsOut)
            if not fastRes:
                # actually up/down-sample
                data = librosa.core.audio.resample(data, sampleRate, fsOut, res_type='kaiser_best')
            else:
                data = librosa.core.audio.resample(data, sampleRate, fsOut, res_type='kaiser_fast')

        # Get the five level wavelet decomposition
        if d:
            WF = WaveletFunctions.WaveletFunctions(data=data, wavelet=self.wavelet, maxLevel=20, samplerate=fsOut)
            denoisedData = WF.waveletDenoise(thresholdType='soft', maxLevel=5)
        else:
            denoisedData = data  # this is to avoid washing out very fade calls during the denoising

        WF = []
        del WF
        gc.collect()
        return denoisedData

    def loadDirectory(self, dirName, denoise, window=1, inc=None):
        """
            Finds and reads wavs from directory dirName.
            Denoise arg is passed to preprocessing.
            wpmode selects WP decomposition function ("new"-our but not AA'd, "aa"-our AA'd)
            Used in training to load an entire dir of wavs into memory.

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
                if file.lower().endswith('.wav') and os.stat(os.path.join(root, file)).st_size != 0 and file[:-4] + '-res'+str(float(resol))+'sec.txt' in files:
                    opstartingtime = time.time()
                    wavFile = os.path.join(root, file)
                    self.filenames.append(wavFile)

                    # adds to self.annotation array, also sets self.sp data and sampleRate
                    succ = self.loadData(wavFile)
                    if not succ:
                        print("ERROR: failed to load file", wavFile)
                        return

                    # denoise and store actual audio data:
                    # note: preprocessing is a side effect on data
                    # (preprocess only reads target nodes from spInfo)
                    denoisedData = self.preprocess(self.sp.data, self.sp.sampleRate, self.spInfo['SampleRate'], d=denoise)
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
        denoisedData = None
        del denoisedData
        gc.collect()
        totalcalls = sum([sum(a) for a in self.annotation])
        totalblocks = sum([len(a) for a in self.annotation])
        print("Directory loaded. %d/%d presence blocks found.\n" % (totalcalls, totalblocks))

    def loadData(self, filename):
        """ Loads a single WAV file and corresponding 0/1 annotations.
            Input: filename - wav file name
            Output: fills self.annotation, sets self.sp data, samplerate
            Returns True if read without errors - important to
            catch this and immediately stop the process otherwise
        """
        # In case we want flexible-size windows again:
        # Added resol input as basic unit for read annotation file
        resol = 1.0
        print('\nLoading:', filename)
        filenameAnnotation = filename[:-4] + '-res' + str(float(resol)) + 'sec.txt'

        self.sp.readWav(filename)

        n = math.ceil((len(self.sp.data) / self.sp.sampleRate)/resol)

        # Do impulse masking by default
        self.sp.data = self.sp.impMask()

        fileAnnotations = []
        # Get the segmentation from the txt file
        with open(filenameAnnotation) as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)
        if d[-1] == []:
            d = d[:-1]
        if len(d) != n:
            print("ERROR: annotation length %d does not match file duration %d!" % (len(d), n))
            self.annotation = []
            return False

        # for each second, store 0/1 presence:
        presblocks = 0
        for row in d:
            fileAnnotations.append(int(row[1]))
            presblocks += int(row[1])

        self.annotation.append(np.array(fileAnnotations))

        totalblocks = sum([len(a) for a in self.annotation])
        print("%d blocks read, %d presence blocks found. %d blocks stored so far.\n" % (n, presblocks, totalblocks))
        return True
