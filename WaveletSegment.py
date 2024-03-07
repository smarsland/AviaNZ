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
from ext import ce_detect
from itertools import combinations


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

    def readBatch(self, data, sampleRate, d, spInfo, wpmode="new", wind=False):
        """ File (or page) loading for batch mode. Must be followed by self.waveletSegment.
            Args:
            1. data to be segmented, ndarray
            2. sampleRate of the data, int
            3. d - turn on denoising before calling?
            4. spInfo - List of filters to determine which nodes are needed & target sample rate
            5. wpmode - old/new/aa to indicate no/partial/full antialias
            6. wind - if True, will produce a WP with all nodes to be used in de-winding
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
            for filter in self.spInfo:
                for subfilter in filter["Filters"]:
                    subfilter["WaveletParams"]['nodes'] = WaveletFunctions.adjustNodes(subfilter["WaveletParams"]['nodes'], "down2")
            # Don't want to resample again, so fsTarget = fsIn
            fsOut = sampleRate
        elif fsOut == 4*sampleRate:
            print("Adjusting nodes for upsampling to", fsOut)
            # same. Wouldn't recommend repeating for larger ratios than 4x
            for filter in self.spInfo:
                for subfilter in filter["Filters"]:
                    downsampled2x = WaveletFunctions.adjustNodes(subfilter["WaveletParams"]['nodes'], "down2")
                    subfilter["WaveletParams"]['nodes'] = WaveletFunctions.adjustNodes(downsampled2x, "down2")
            # Don't want to resample again, so fsTarget = fsIn
            fsOut = sampleRate
        # Could also similarly "downsample" by adding an extra convolution, but it's way slower
        # elif sampleRate == 2*fsOut:
        #     # don't actually downsample audio, just "upsample" the nodes needed
        #     for subfilter in self.spInfo["Filters"]:
        #         subfilter["WaveletParams"]['nodes'] = WaveletFunctions.adjustNodes(subfilter["WaveletParams"]['nodes'], "up2")
        #     print("upsampled nodes")
        #     self.spInfo["SampleRate"] = sampleRate

        # After upsampling, there will be a sharp drop in energy
        # at the original Fs. This will really distort the polynomial
        # fit used for wind prediction, so do not allow it.
        # (If the nodes were adjusted by the mechanism above, the fit
        # will use a continuous block of lower freqs and mostly be fine.)
        if wind and fsOut>sampleRate:
            print("ERROR: upsampling will cause problems for wind removal. Either turn off the wind filter, or retrain your recognizer to match the sampling rate of these files.")
            return

        denoisedData = self.preprocess(data, sampleRate, fsOut, d=d, fastRes=True)

        # Find out which nodes will be needed:
        allnodes = []
        if wind:
            allnodes = list(range(31, 63))
        for filt in self.spInfo:
            for subfilter in filt["Filters"]:
                allnodes.extend(subfilter["WaveletParams"]["nodes"])
        allnodes = list(set(allnodes))

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
            print("-- Identifying calls using subfilter %s --" % subfilter["calltype"])
            goodnodes = subfilter['WaveletParams']["nodes"]

            detected = self.detectCalls(self.WF, nodelist=goodnodes, subfilter=subfilter, rf=True, aa=wpmode!="old")

            # merge neighbours in order to convert the detections into segments
            # note: detected np[0 1 1 1] becomes [[1,3]]
            segmenter = Segment.Segmenter()
            detected = segmenter.convert01(detected)
            detected = segmenter.joinGaps(detected, maxgap=0)
            detected_allsubf.append(detected)
        print("--- Wavelet segmenting completed in %.3f s ---" % (time.time() - opst))
        return detected_allsubf

    def waveletSegmentChp(self, filtnum, alg, alpha=None, window=None, maxlen=None, silent=True, wind=0):
        """ Main analysis wrapper, similar to waveletSegment,
            but uses changepoint detection for postprocessing.
            Args:
            1. filtnum: index of the current filter in self.spInfo (which is a list of filters...)
            2. alg: 1 - standard epidemic detector, 2 - nuisance-signal detector
            3. alpha: penalty strength for the detector
            4. window: wavelets will be merged in groups of this size (s) before analysis
            5. maxlen: maximum allowed length (s) of signal segments
              3-5 can be None, in which case they are retrieved from self.spInfo.
            6. silent: silent (True) or verbose (False) mode
            7. adjust for wind? 0=no, 1=interpolate by OLS, 2=interpolate by quantreg
            Returns: list of lists of segments found (over each subfilter)-->[[sub-filter1 segments], [sub-filter2 segments]]
        """
        opst = time.time()

        if silent:
            printing=0
        else:
            printing=1

        # No resampling here. Will read nodes from self.spInfo, which may already be adjusted.

        ### find segments with each subfilter separately
        detected_allsubf = []
        for subfilter in self.spInfo[filtnum]["Filters"]:
            print("-- Identifying calls using subfilter %s --" % subfilter["calltype"])
            goodnodes = subfilter['WaveletParams']["nodes"]
            if alpha is None:
                alpha = subfilter["WaveletParams"]["thr"]
            if window is None:
                window = subfilter["WaveletParams"]["win"]
            if maxlen is None:
                maxlen = subfilter["TimeRange"][1]

            detected = self.detectCallsChp(self.WF, nodelist=goodnodes, alpha=alpha, window=window, maxlen=maxlen, alg=alg, printing=printing, wind=wind)

            detected_allsubf.append(detected)
        print("--- WV changepoint segmenting completed in %.3f s ---" % (time.time() - opst))
        return detected_allsubf

    def waveletSegment_train(self, dirName, thrList, MList, d=False, learnMode='recaa', window=1,
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
        self.loadDirectory(dirName=dirName, denoise=d)
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
        res = self.gridSearch(self.maxEs, thrList, MList, learnMode, window, inc)

        return res

    def waveletSegment_trainChp(self, dirName, thrList, window, maxlen):
        """ Entry point to use during training, called from DialogsTraining.py.
            Switches between various training methods, orders data loading etc.,
            then just passes the arguments to the right training method and returns the results.

            Input: path to directory with wav & wav.data files.
            thrList: list of possible alphas to test.
            window: window length in sec (for averaging energies in detection).
            maxlen: maximum signal length in sec.
            Return: tuple of arrays (nodes, tp, fp, tn, fn)
        """
        if len(self.spInfo["Filters"])>1:
            print("ERROR: must provide only 1 subfilter at a time!")
            return
        else:
            subfilter = self.spInfo["Filters"][0]

        # verify that the provided window will result in an integer
        # number of WCs at any level <=5
        estWCperWindow = math.ceil(window * self.spInfo['SampleRate']/32)
        estrealwindow = estWCperWindow / self.spInfo['SampleRate']*32
        if estrealwindow!=window:
            print("ERROR: provided window (%f s) will not produce an integer number of WCs. This is currently disabled for safety." % window)
            return

        # 1. read wavs and annotations into self.annotation, self.audioList
        self.filenames = []
        self.loadDirectoryChp(dirName=dirName, window=window)
        if len(self.annotation) == 0:
            print("ERROR: no files loaded!")
            return

        nwins = [len(annot) for annot in self.annotation]

        # --------------------
        # 2. determine what nodes can be meaningfully tested, given freq limits
        nodeList = []
        freqrange = subfilter['FreqRange']

        # levels: 0-1, 2-5, 6-13, 14-29, 30-61 (unrooted tree)
        # so this will take the last three levels:
        for node_un in range(14, 62):
            # corresponding node in a rooted tree (as used by WF)
            node = node_un + 1
            nodefrl, nodefru = WaveletFunctions.getWCFreq(node, self.spInfo["SampleRate"])
            if nodefrl < freqrange[1] and nodefru > freqrange[0]:
                # node has some overlap with the target range, so can be tested
                nodeList.append(node)
        # nodeList now stores ROOTED numbers of nodes, to match WF.tree (which has tree[0] = data)

        # --------------------
        # 3. extract node energies for all files and get node correlations:
        # TODO see what can go to loadData
        opstartingtime = time.time()
        print("--- Starting correlation stage ---")
        # TODO convert to a matrix maybe?
        allEs = []
        allwindows = np.zeros((62, len(self.audioList)))
        nodeCorrs = np.zeros((62, len(self.audioList)))
        for indexF in range(len(self.audioList)):
            print("-- Computing wavelet correlations in file %d / %d --" %(indexF+1, len(self.audioList)))
            filenwins = nwins[indexF]
            # foffs = sum(nwins[:indexF])  # start of this file's pieces in allEs

            # extract energies
            currWCs = np.zeros((62, filenwins))
            # Generate a full 5 level wavelet packet decomposition
            # (this will not be downsampled. antialias=False means no post-filtering)
            self.WF = WaveletFunctions.WaveletFunctions(data=self.audioList[indexF], wavelet=self.wavelet, maxLevel=5, samplerate=self.spInfo['SampleRate'])
            self.WF.WaveletPacket(nodeList, mode='symmetric', antialias=False)
            for node in nodeList:
                nodeE, noderealwindow = self.WF.extractE(node, window, wpantialias=True)
                allwindows[node-1, indexF] = noderealwindow
                # the wavelet energies may in theory have one more or less windows than annots
                # b/c they adjust the window size to use integer number of WCs.
                # If they differ by <=1, we allow that and just equalize them:
                if filenwins==len(nodeE)+1:
                    currWCs[node-1,:-1] = nodeE
                    currWCs[node-1,-1] = currWCs[node-1,-2] # repeat last element
                elif filenwins==len(nodeE)-1:
                    # drop last WC
                    currWCs[node-1,:] = nodeE[:-1]
                elif np.abs(filenwins-len(nodeE))>1:
                    print("ERROR: lengths of annotations and energies differ:", filenwins, len(nodeE))
                    return
                else:
                    currWCs[node-1,:] = nodeE

            allEs.append(currWCs)
            # note that currWCs and nodeCorrs are UNROOTED

            # Compute all WC-annot correlations for this file
            nodeCorrs[:,indexF] = self.compute_r(self.annotation[indexF], currWCs) * len(self.annotation[indexF])

        # get a single "correlation" value for each node:
        # Note: averaged correlation is not the same as calculating correlation over
        # all files, but should be meaningful enough for this.
        nodeCorrs = np.abs(np.sum(nodeCorrs, axis=1) / sum(nwins))
        print(nodeCorrs)
        print("Correlations completed in", time.time() - opstartingtime)

        # find best nodes (+1 b/c nodeCorrs are unrooted indices, and nodeList is rooted)
        # this will likely include nodes in top levels as well (0-14), just keep in mind.
        bestnodes = np.argsort(nodeCorrs)[-15:]+1
        print("Best nodes: ", bestnodes)
        print("Before filtering: ", nodeList)
        nodeList = [node for node in nodeList if node in bestnodes]

        # --------------------
        # 4. run the detector for each setting (node x thr x filepieces)
        opstartingtime = time.time()
        print("--- Starting detection stage ---")
        alldetections = np.zeros((len(nodeList), len(thrList), sum(nwins)))
        for indexF in range(len(self.audioList)):
            print("-- Extracting energies from file %d / %d --" %(indexF+1, len(self.audioList)))
            # Could bandpass, but I wasn't doing it in the paper
            # audio = self.sp.ButterworthBandpass(self.audioList[filenum],
            #                                     self.spInfo['SampleRate'],
            #                                     low=freqrange[0], high=freqrange[1])
            filenwins = nwins[indexF]
            foffs = sum(nwins[:indexF])  # start of this file's pieces in alldetections

            for indexn in range(len(nodeList)):
                node = nodeList[indexn]
                print("Analysing node", node)
                # extract node from file num, and average over windows of set size
                # (wpantialias=True specifies the non-downsampled WP)
                nodeE = allEs[indexF][node-1,:]
                noderealwindow = allwindows[node-1, indexF]
                nodesigma2 = np.percentile(nodeE, 10)
                # NOTE: we're providing points on the original scale (non-squared) for the C part
                nodeE = np.sqrt(nodeE)

                # Convert max segment length from s to realized windows
                # (segments exceeding this length will be marked as 'n')
                realmaxlen = math.ceil(maxlen / noderealwindow)

                print("node prepared for detection")

                for indext in range(len(thrList)):
                    print("Detecting with alpha=", thrList[indext])
                    # run detector with thr on nodeE
                    thrdet = ce_detect.launchDetector2(nodeE, nodesigma2, realmaxlen, alpha=thrList[indext], printing=0).astype('float')

                    # keep only S and their positions:
                    if np.shape(thrdet)[0]>0:
                        thrdet = thrdet[np.logical_or(thrdet[:,2]==ord('s'), thrdet[:,2]==ord('o')), 0:2]

                    # convert detections from the window scale into actual seconds
                    # and then back into 0/1 over some windows for computing F1 score later
                    # (the original binary detections are in realised windows which may differ over nodes)
                    thrdet[:,:2] = thrdet[:,:2] * noderealwindow / window
                    thrdetbin = np.zeros(filenwins, dtype=np.uint8)
                    for i in range(np.shape(thrdet)[0]):
                        start = math.floor(thrdet[i,0])
                        end = min(filenwins, math.ceil(thrdet[i,1]))
                        thrdetbin[start:end] = 1

                    # store the detections
                    alldetections[indexn, indext, foffs:(foffs+filenwins)] = thrdetbin
        # alldetections is now a 3d np.array of 0/1 over nodes x thr x windows over all files
        print("Detections completed in", time.time() - opstartingtime)

        # for Fbeta score, concat 0/1 annotations over all files into a long vector:
        allannots = np.concatenate(self.annotation)

        # --------------------
        # 5. find top nodes by Fbeta-score, for each thr
        # enumerate all possible subsets of up to k nodes:
        opstartingtime = time.time()
        print("--- Starting best-subset search ---")
        MAXK = 6
        print("Enumerating subsets up to K=", MAXK)
        ksubsets = []
        for k in range(1,MAXK+1):
            ksubsets.extend(list(combinations(range(len(nodeList)), k)))
        subsetfbs = np.zeros((len(ksubsets), len(thrList)))

        NODEPEN = 0.01
        total_positives = np.count_nonzero(allannots)   # precalculated for speed
        for indext in range(len(thrList)):
            # Best-subset part
            # find the best set of k nodes to start with:
            print("-- thr = ", thrList[indext])
            for nodeset_ix in range(len(ksubsets)):
                # evaluate F1 score of this nodeset over all files
                # (each nodeset is a list of indices to nodes in nodeList)
                nodeset = list(ksubsets[nodeset_ix])
                # print("evaluating subset", [nodeList[nn] for nn in nodeset])
                detect_allnodes = np.logical_or.reduce(alldetections[nodeset, indext, :])
                fB = self.fBetaScore_fast(allannots, detect_allnodes, total_positives)
                # Add a penalty for the number of nodes
                subsetfbs[nodeset_ix, indext] = fB - len(nodeset)*NODEPEN
        print("Best-subset search completed in", time.time() - opstartingtime)

        # output arrays
        # (extra dimension for compatibility w/ multiple Ms)
        tpa = np.zeros((1, len(thrList)))
        fpa = np.zeros((1, len(thrList)))
        tna = np.zeros((1, len(thrList)))
        fna = np.zeros((1, len(thrList)))
        finalnodes = []

        # TODO this part can be entirely replaced with larger K
        # b/c 15 nodes produce 32k subsets over all K.
        # STEPWISE part, with top N best subsets used to initialise N runs
        opstartingtime = time.time()
        print("--- Starting stepwise search ---")
        NTOPSETS = 1   # how many different initialisations to use
        for indext in range(len(thrList)):
            print("-- Optimising with t %d/%d (thr=%f)" % (indext + 1, len(thrList), thrList[indext]))
            # best nodesets for this t
            bestix = np.argsort(subsetfbs[:,indext])[-NTOPSETS:]

            # take a good subset, and continue from there:
            # (repeat with NTOPSETS different initializations for each threshold)
            fB_out = 0
            nodes_out = []
            tp_out = 0
            fp_out = 1  # init the bad stats to 1, so that nothing breaks
            tn_out = 0  # even if the detector fails entirely
            fn_out = 1
            for i in range(np.shape(bestix)[0]):
                b = bestix[i]
                top_subset_nodes = [nodeList[nix] for nix in ksubsets[b]]
                print("Top subset:", top_subset_nodes)

                # detections with the nodes in the current subset
                # (note that alldetections is indexed by node position in nodeList,
                #  not by actual node number!)
                detect_best = np.maximum.reduce(alldetections[list(ksubsets[b]), indext, :])

                # recalculate statistics for these nodes (should be same as subsetfbs)
                bestfB, bestRecall, tp, fp, tn, fn = self.fBetaScore(allannots, detect_best)
                # Add a penalty for the number of nodes
                if bestfB is not None:
                    bestfB = bestfB - NODEPEN*len(top_subset_nodes)
                else:
                    bestfB = 0

                print("starting nodes", top_subset_nodes)
                print("starting fb", bestfB, "recall", bestRecall)

                # nodes still available for the stepwise search:
                # reversed to start with lower-level nodes
                for new_node_ix in reversed(range(len(nodeList))):
                    new_node = nodeList[new_node_ix]
                    if new_node in top_subset_nodes:
                        continue
                    print("testing node", new_node)

                    # try adding the new node
                    detect_withnew = np.maximum.reduce([detect_best, alldetections[new_node_ix, indext, :]])
                    fB, recall, tp, fp, tn, fn = self.fBetaScore(allannots, detect_withnew)

                    if fB is None:
                        continue

                    # Add a penalty for the number of nodes
                    fB = fB - NODEPEN*(len(top_subset_nodes)+1)

                    # If this node improved fB,
                    # store it and update fB, recall, best detections, and optimum nodes
                    if fB > bestfB:
                        print("Adding node", new_node)
                        print("new fb", fB, "recall", recall)
                        top_subset_nodes.append(new_node)
                        detect_best = detect_withnew
                        bestfB = fB
                        bestRecall = recall

                    # Adding more nodes will not reduce FPs, so this is sufficient to stop:
                    # Stopping a bit earlier to have fewer nodes and fewer FPs:
                    if bestfB >= 0.95 or bestRecall >= 0.95:
                        break
                # store this if this is the best initialisation
                if bestfB>fB_out:
                    fB_out = bestfB
                    nodes_out = top_subset_nodes
                    tp_out = tp
                    fp_out = fp
                    tn_out = tn
                    fn_out = fn

            # populate output
            tpa[0, indext] = tp_out
            fpa[0, indext] = fp_out
            tna[0, indext] = tn_out
            fna[0, indext] = fn_out
            finalnodes.append(nodes_out)
            print("-- Run t %d/%d complete\n---------------- " % (indext + 1, len(thrList)))
        print("Stepwise search completed in", time.time() - opstartingtime)

        return [finalnodes], tpa, fpa, tna, fna

    def waveletSegment_cnn(self, dirName, filt):
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
        for root, dirs, files in os.walk(dirName):
            for file in files:
                if file.lower().endswith('.wav') and os.stat(os.path.join(root, file)).st_size != 0 and file[:-4] + '-GT.txt' in files:
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
                self.readBatch(self.sp.data[start:end], self.sp.sampleRate, d=False, spInfo=[filt], wpmode="new", wind=False)
                # TODO not sure if wind removal should be done here.
                # Maybe not, to generate more noise examples?

                # segmentation, same as in batch mode. returns [[sub-filter1 segments]]
                if "method" not in filt or filt["method"]=="wv":
                    detected_segs = self.waveletSegment(0, wpmode="new")
                elif filt["method"]=="chp":
                    detected_segs = self.waveletSegmentChp(0, alg=2, wind=False)

                # flatten over the call types and store
                out = []
                for subfilterdet in detected_segs:
                    out.extend(subfilterdet)
                detected_out.append((filename, out))

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
        inc_sr = int(math.ceil(inc*sampleRate))

        # output columns dimension equal to number of sliding window
        N = int(math.ceil(len(data)/inc_sr))
        coefs = np.zeros((2 ** (nlevels + 1) - 2, N))

        # generate a WP on all of the data
        WF = WaveletFunctions.WaveletFunctions(data, wavelet=self.wavelet, maxLevel=20, samplerate=sampleRate)
        if wpmode == "pywt":
            print("ERROR: pywt mode deprecated, use new or aa")
            return
        elif wpmode == "new":
            allnodes = range(2 ** (nlevels + 1) - 1)
            WF.WaveletPacket(allnodes, mode='symmetric', antialias=False)
        elif wpmode == "aa":
            allnodes = range(2 ** (nlevels + 1) - 1)
            WF.WaveletPacket(allnodes, mode='symmetric', antialias=True, antialiasFilter=True)

        # TODO this nonsense could be replaced w/ WF.extractE for consistency

        # for each sliding window:
        # start,end are its coordinates (in samples)
        start = 0
        for t in range(N):
            E = []
            end = min(len(data), start+win_sr)

            # Calculate energies of all nodes EXCEPT ROOT - from 1 to 2^(nlevel+1)-1
            for level in range(1, nlevels + 1):
                # Calculate the window position in WC coordinates
                dsratio = 2**level
                WCperWindow = math.ceil(win_sr/dsratio)
                if wpmode=="aa" or wpmode=="new": # account for non-downsampled tree
                    WCperWindow = 2*WCperWindow
                # (root would not require this, but is skipped here anyway)

                startwc = t*WCperWindow
                endwc = startwc+WCperWindow

                # Extract the energy
                lvlnodes = WF.tree[2 ** level - 1:2 ** (level + 1) - 1]
                e = np.array([np.sum(n[startwc:endwc] ** 2) for n in lvlnodes])
                if np.sum(e) > 0:
                    e = 100.0 * e / np.sum(e)  # normalize per-level
                E = np.concatenate((E, e), axis=0)

            start += inc_sr
            # so now 0-1 is the first level, 2-5 the second etc.
            coefs[:,t] = E
        return coefs

    def fBetaScore_fast(self, annotation, predicted, T, beta=2):
        """ Computes the beta scores given two sets of redictions.
            Simplified by dropping printouts and some safety checks.
            (Assumes logical or int 1/0 input.)
            Outputs 0 when the score is undefined. """
        TP = np.count_nonzero(annotation & predicted)
        # T = np.count_nonzero(annotation)   # precalculated and passed in for speed
        P = np.count_nonzero(predicted)
        if T==0 or P==0 or TP==0:
            return 0
        recall = float(TP) / T  # TruePositive/#True
        precision = float(TP) / P  # TruePositive/#Positive
        fB = ((1. + beta ** 2) * recall * precision) / (recall + beta ** 2 * precision)
        return fB

    def fBetaScore(self, annotation, predicted, beta=2):
        """ Computes the beta scores given two sets of predictions """
        annotation = np.array(annotation)
        predicted = np.array(predicted)
        TP = float(np.sum(np.where((annotation == 1) & (predicted == 1), 1, 0)))
        T = float(np.sum(annotation))  # to force all divisions to float
        P = float(np.sum(predicted))
        if T != 0:
            recall = TP / T  # TruePositive/#True
        else:
            recall = None
        if P != 0:
            precision = TP / P  # TruePositive/#Positive
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
        if len(annotation)!=np.shape(waveletCoefs)[1]:
            print("ERROR: wavelet and annotation lengths must match")
            #return

        w0 = np.where(annotation == 0)[0]
        w1 = np.where(annotation == 1)[0]

        r = np.zeros(np.shape(waveletCoefs)[0])
        for node in range(len(r)):
            # for safety e.g. when an array was filled with a const and SD=0
            if np.all(waveletCoefs[node,:]==waveletCoefs[node,0]):
                r[node] = 0
                continue

            r[node] = (np.mean(waveletCoefs[(node, w1)]) - np.mean(waveletCoefs[(node, w0)])) / np.std(waveletCoefs[node, :]) * np.sqrt(len(w0) * len(w1)) / len(annotation)

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


    def extractE(self, wf, nodelist, MList, annotation=None, window=1, inc=None, aa=True):
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

        #print(nw, np.shape(detected))
        if np.shape(detected)[1]>0:
            detected = np.max(detected, axis=1)
        else:
            detected = np.zeros(nw)
        #print(detected)

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

    def detectCallsChp(self, wf, nodelist, alpha, maxlen, window=1, alg=1, printing=1, wind=0):
        """
        For wavelet TESTING and general SEGMENTATION using changepoint detection
        (non-reconstructing)
        Args:
        wf - WaveletFunctions with a homebrew wavelet tree (list of ndarray nodes)
        nodelist - will reconstruct signal and run detections on each of these nodes separately
        alpha - penalty strength for the detector
        maxlen - maximum allowed signal segment length, in s
        window - energy will be calculated over these windows, in s
        alg - standard (1) or with nuisance segments (2)
        printing - run silent (0) or verbose (1)
        wind - adjust for wind? 0=no, 1=interpolate by OLS, 2=interpolate by QR

        Return: ndarray of 1/0 annotations for each of T windows
        """

        # Verify that the provided window will result in an integer
        # number of WCs at any level <=5.
        # This isn't necessary for detection, but makes life easier.
        # Currently, only needed if using wind adjustment
        # with nodes at different levels (not clear how to adjust then).
        # (I.e. for any filters w/ any 4th lvl nodes, as wind adj is
        # hardcoded to use 5th lvl nodes anyway).
        if wind:
            # all nodes will be needed for wind adjustment
            dsratio = 2**5
            # for not-wind: dsratio = 2**math.floor(math.log2(max(nodelist)+1))
            nodefs = wf.treefs/dsratio
            estWCperWindow = math.ceil(window * nodefs)
            estrealwindow = estWCperWindow / nodefs
            if estrealwindow!=window:
                print("ERROR: provided window (%f s) will not produce an integer number of WCs. This is currently disabled for safety." % window)
                raise

        # Estimate wind noise levels for each window x target node.
        if wind:
            # identify wind nodes, and calculate
            # regression x - node freq centers
            print("identifying wind nodes...")
            wind_nodes = []
            windnodecenters = []
            for node in range(31, 63):  # only use leaf nodes when estimating wind
                if node in nodelist:  # skip target nodes, obviously
                    continue
                # target node can be 1 level higher than this node, so check for that too
                if (node-1)//2 in nodelist:
                    continue
                if node==31 or node==47:  # skip extreme nodes with filtering artifacts
                    continue
                nodecenter = sum(WaveletFunctions.getWCFreq(node, wf.treefs))/2
                if nodecenter>=6000:  # skip high freqs when estimating wind
                    continue
                wind_nodes.append(node)
                windnodecenters.append(nodecenter)

            # Regression y: extract energies from all nodes
            print("extracting wind node energy...")
            datalen = math.floor(len(wf.tree[0])/wf.treefs/window)
            windE = np.zeros((datalen, len(wind_nodes)))
            for node_ix in range(len(wind_nodes)):
                node = wind_nodes[node_ix]
                windE[:, node_ix], _ = wf.extractE(node, window)

            # For oversubtraction, roughly estimate background level
            # from 10% quietest frames in each node:
            # TODO optimize all this to avoid re-extracting same nodes
            OVERSUBALPHA = 1.0
            print("Will oversubtract with alpha=", OVERSUBALPHA)
            rootE, _ = wf.extractE(0, window, wpantialias=False)
            numframes = round(0.1*len(rootE))
            quietframes = np.argpartition(rootE, numframes)[:numframes]
            bgpow = np.zeros(len(nodelist))
            for node_ix in range(len(nodelist)):
                E, _ = wf.extractE(nodelist[node_ix], window, wpantialias=True)
                bgpow[node_ix] = np.mean(np.log(E[quietframes]))

            # for each window, interpolate wind (log) energy in each target node:
            pred = np.zeros((datalen, len(nodelist)))
            regx = np.log(windnodecenters)  # NOTE that here and further centers are in log(freq)!
            qrbiasadjust = 0
            # need to prepare polynomial features manually for non-OLS methods
            # and also add an adjustment factor to QR, calculated assuming
            # quantile 0.2 and roughly 0.1-0.2 s windows
            if wind==2:
                regx = np.column_stack((np.ones(len(regx)), regx, regx**2, regx**3))
                # ideally this should be based on the actual number of WCs
                # in the window and the gamma function (see paper),
                # but generally is negligible except for v small windows and low SRs
                if window<=0.1 and wf.treefs<16000:
                    qrbiasadjust = 0.4

            tgtnodecenters = np.log([sum(WaveletFunctions.getWCFreq(node, wf.treefs))/2 for node in nodelist])
            windE = np.log(windE)
            for w in range(datalen):
                regy = windE[w, :]
                # ---- REGRESSION IS DONE HERE ----
                if wind==1:
                    pol = np.polynomial.polynomial.Polynomial.fit(regx,regy,3)
                elif wind==2:
                    # TODO sklearn will add quantreg in v1.0, see if it is any better
                    pol = WaveletFunctions.QuantReg(regy, regx, q=0.2, max_iter=250, p_tol=1e-3)
                else:
                    print("ERROR: unrecognized wind adjustment %s" % wind)
                    raise

                # Interpolate using the fitted model:
                for node_ix in range(len(nodelist)):
                    # for higher level nodes, need to (linearly) average the nearest predictions:
                    if nodelist[node_ix] in range(15, 31):
                        delta = wf.treefs/128   # half width of a leaf node band = Fs/2/numnodes/2
                        f1 = np.exp(tgtnodecenters[node_ix]) - delta
                        f2 = np.exp(tgtnodecenters[node_ix]) + delta
                        pred1 = pol(np.log(f1))
                        pred2 = pol(np.log(f2))
                        # oversubtraction:
                        pred1 = (pred1 - bgpow[node_ix])*OVERSUBALPHA + bgpow[node_ix]
                        pred2 = (pred2 - bgpow[node_ix])*OVERSUBALPHA + bgpow[node_ix]
                        pred[w, node_ix] = np.log((np.exp(pred1) + np.exp(pred2))/2)
                    else:
                        # Straightforward for 5th lvl nodes
                        pred[w, node_ix] = pol(tgtnodecenters[node_ix])
                        # Oversubtraction:
                        pred[w, node_ix] = (pred[w, node_ix] - bgpow[node_ix])*OVERSUBALPHA + bgpow[node_ix]
                # print("Predictions (log): ", pred[w,:])
            # TODO would probably be faster to predict all nodes and then average
            # to obtain upper level nodes, but difficult to keep track of nodes then.

            # convert back to (linear) energies:
            pred = np.exp(pred+qrbiasadjust)

        # Compute the number of samples in a window -- species specific
        detected = np.empty((0,3))
        for node_ix in range(len(nodelist)):
            node = nodelist[node_ix]
            # Extracts energies (i.e. integral of square magnitudes) over windows.
            # Window size will be adjusted to realwindow (in s), b/c it needs to
            # correspond to an integer number of WCs at this node,
            # Returns E: vector of energies
            E, realwindow = wf.extractE(node, window, wpantialias=True)

            # Convert max segment length from s to realized windows
            # (segments exceeding this length will be marked as 'n')
            realmaxlen = math.ceil(maxlen / realwindow)

            if np.max(E)>1e-2:  # (checking so that the hardcoded epsilon would be relatively small)
                E = E + 1e-5 # add epsilon in case there is a short quiet period
                # NOTE: any non-negligible adjustments need to be applied to pred too

            # Estimate of the global background for this file/page
            sigma2 = np.percentile(E, 10)
            print("Global var: %.1f, range of E: %.1f-%.1f, Q10: %.1f" % (np.mean(E), np.min(E), np.max(E), sigma2))

            if wind:
                # ---- LOG SP SUB ----
                # retrieve and adjust for the predicted wind strength
                print("Wind strength summary: mean %.2f, median %.2f" % (np.mean(pred[:, node_ix]), np.median(pred[:, node_ix])))
                E = np.maximum(1, E / pred[:, node_ix])
                # This implicitly normalizes to sigma2=1
            else:
                # just normalize, same as passing sigma2 to the detectors
                E = E / sigma2

            # sqrt because the detector squares the data itself (sign not important)
            # Note that no transformation is applied otherwise (i.e. linear, not log scale)
            E = np.sqrt(E)

            # analyze by our algorithms.
            # returns a matrix of n x [s, e, type]
            # type is an int corresponding to 'n'=NUIS, 's'=SEG, 'o'=SEGonNUIS
            if alg==1:
                segm1 = ce_detect.launchDetector1(E, realmaxlen, alpha=alpha).astype('float')
            else:
                segm1 = ce_detect.launchDetector2(E, 1, realmaxlen, alpha=alpha, printing=printing).astype('float')

            # here's how you would extract segment means:
            # for seg in segm1:
            #     print(seg, round(np.mean(E[int(seg[0]):int(seg[1])]**2)))

            # convert from the window scale into actual seconds
            segm1[:,:2] = segm1[:,:2] * realwindow

            detected = np.vstack((detected, np.asarray(segm1)))

        # keep only S and their positions:
        if np.shape(detected)[0]>0:
            detected = detected[np.logical_or(detected[:,2]==ord('s'), detected[:,2]==ord('o')), 0:2]

        # now, need to go over the segments and find any overlapping ones (i.e. combine across nodes).
        # NOTE: will sort them
        s = Segment.Segmenter()
        outsegs = s.checkSegmentOverlap(detected)
        print("After merge:", outsegs)

        E = None
        del E
        gc.collect()
        return outsegs

    def gridSearch(self, E, thrList, MList, learnMode=None, window=1, inc=None):
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
        freqrange = [subf["FreqRange"] for subf in self.spInfo["Filters"]]
        freqrange = (np.min(freqrange), np.max(freqrange))

        # avoid low-level nodes
        low_level_nodes = list(range(14))
        for item in nodes1:
            itemfrl, itemfru = WaveletFunctions.getWCFreq(item, self.spInfo["SampleRate"])
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
            fastRes - use kaiser_fast instead of best. Twice faster but pretty similar output.
        """
        # resample (implies this hasn't been done by node adjustment before)
        if sampleRate != fsOut:
            print("Resampling from", sampleRate, "to", fsOut)
            if not fastRes:
                data = librosa.resample(data, orig_sr=sampleRate, target_sr=fsOut, res_type='kaiser_best')
            else:
                data = librosa.resample(data, orig_sr=sampleRate, target_sr=fsOut, res_type='kaiser_fast')

        # Get the five level wavelet decomposition
        if d:
            WF = WaveletFunctions.WaveletFunctions(data=data, wavelet=self.wavelet, maxLevel=20, samplerate=fsOut)
            denoisedData = WF.waveletDenoise(thresholdType='soft', maxLevel=5)
            del WF
        else:
            denoisedData = data  # this is to avoid washing out very fade calls during the denoising

        gc.collect()
        return denoisedData

    def loadDirectory(self, dirName, denoise, impMask=True):
        """
            Finds and reads wavs from directory dirName.
            Denoise arg is passed to preprocessing.
            wpmode selects WP decomposition function ("new"-our but not AA'd, "aa"-our AA'd)
            Used in training to load an entire dir of wavs into memory.
            impMask: impulse masking on audiodata. Off for changepoints to avoid distorting the mean

            Results: self.annotation, self.audioList, self.noiseList arrays.
        """
        # When reading annotations, it is important to know that the
        # stored ones match the annotation set in this analysis by window and inc.
        # One could use something like this:
        # resol = (math.gcd(int(100 * window), int(100 * inc))) / 100
        # to check this match with one variable, but no good way to store it
        # in the file, as this is a float and might need more than 2 decimals.

        self.annotation = []
        self.audioList = []

        for root, dirs, files in os.walk(str(dirName)):
            for file in files:
                if file.lower().endswith('.wav') and os.stat(os.path.join(root, file)).st_size != 0 and file[:-4] + '-GT.txt' in files:
                    opstartingtime = time.time()
                    wavFile = os.path.join(root, file)
                    self.filenames.append(wavFile)

                    # adds to self.annotation array, also sets self.sp data and sampleRate
                    succ = self.loadData(wavFile, impMask=impMask)
                    if not succ:
                        print("ERROR: failed to load file", wavFile)
                        return

                    # resample, denoise and store the resulting audio data:
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

    def loadDirectoryChp(self, dirName, window):
        """
            Finds and reads wavs from directory dirName.
            Used in training to load an entire dir of wavs into memory.
            window: changepoint analysis window

            Results: self.annotation, self.audioList, self.noiseList arrays.
        """
        self.annotation = []
        self.audioList = []
        print("Loading data from dir", dirName)

        for root, dirs, files in os.walk(str(dirName)):
            for file in files:
                if file.lower().endswith('.wav') and os.stat(os.path.join(root, file)).st_size != 0 and file[:-4] + '-GT.txt' in files:
                    opstartingtime = time.time()
                    wavFile = os.path.join(root, file)
                    self.filenames.append(wavFile)

                    # adds to self.annotation array, also sets self.sp data and sampleRate
                    self.loadDataChp(wavFile, window)

                    # resample, denoise and store the resulting audio data:
                    # note: preprocessing is a side effect on data
                    # (preprocess only reads target nodes from spInfo)
                    denoisedData = self.preprocess(self.sp.data, self.sp.sampleRate, self.spInfo['SampleRate'], d=False)
                    self.audioList.append(denoisedData)

                    print("file loaded in %.3f s" % (time.time() - opstartingtime))

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

    def loadData(self, filename, impMask=True):
        """ Loads a single WAV file and corresponding 0/1 annotations.
            Input: filename - wav file name
            Output: fills self.annotation, sets self.sp data, samplerate
            Returns True if read without errors - important to
            catch this and immediately stop the process otherwise
        """
        print('\nLoading:', filename)
        filenameAnnotation = filename[:-4] + '-GT.txt'

        self.sp.readWav(filename)

        # Do impulse masking by default
        if impMask:
            self.sp.data = self.sp.impMask()

        fileAnnotations = []
        # Get the segmentation from the txt file
        with open(filenameAnnotation) as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)
        if d[-1] == []:
            d = d[:-1]

        # Hardcoded resolution of the GT file:
        # (only used for a sanity check now)
        resol = 1.0
        n = math.ceil((len(self.sp.data) / self.sp.sampleRate)/resol)
        if len(d) != n:
            print("ERROR: annotation length %d does not match file duration %d! %s" % (len(d), n, filename))
            self.annotation = []
            #return False

        # for each second, store 0/1 presence:
        presblocks = 0
        for row in d:
            fileAnnotations.append(int(row[1]))
            presblocks += int(row[1])

        self.annotation.append(np.array(fileAnnotations))

        totalblocks = sum([len(a) for a in self.annotation])
        print("%d blocks read, %d presence blocks found. %d blocks stored so far.\n" % (n, presblocks, totalblocks))
        return True

    def loadDataChp(self, filename, window):
        """ Loads a single WAV file and 0/1 annotations.
            Input: filename - wav file name
            Output: fills self.annotation, sets self.sp data, samplerate
            Returns True if read without errors - important to
            catch this and immediately stop the process otherwise
        """
        print('Loading file:', filename)
        filenameAnnotation = filename[:-4] + '-GT.txt'

        # Read data. No impulse masking
        self.sp.readWav(filename)

        # Get the segmentation from the txt file
        with open(filenameAnnotation) as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)
        if d[-1] == []:
            d = d[:-1]

        # A sanity check as the durations will be used in F1 score
        nwins = math.ceil(len(self.sp.data) / self.sp.sampleRate / window)
        if len(d) != nwins:
            print("ERROR: annotation length %d does not match file duration %d!" % (len(d), nwins))
            self.annotation = []
            return

        # for each window, store 0/1 presence
        fileAnnotations = np.array([int(row[1]) for row in d])
        self.annotation.append(fileAnnotations)

        presblocks = sum(fileAnnotations)
        totalblocks = sum([len(a) for a in self.annotation])
        print("%d blocks read, %d presence blocks found. %d blocks stored so far.\n" % (nwins, presblocks, totalblocks))
