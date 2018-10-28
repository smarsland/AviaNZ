
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
import pywt
import WaveletFunctions
import wavio, librosa
import numpy as np
import json, time, os, math, csv, gc
import SignalProc
import Segment
from ext import ce_denoise as ce

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

# TODO: Inconsisient about symmlots of or zeros for the wavelet packet
# TODO: This still needs lots of tidying up

class WaveletSegment:
    # This class implements wavelet segmentation for the AviaNZ interface

    def __init__(self,data=[],sampleRate=0,wavelet='dmey2',annotation=[],mingap=0.3,minlength=0.2):
        self.annotation = annotation
        self.waveletCoefs = []
        self.audioList = []
        if data != []:
            self.data = data
            self.sampleRate = sampleRate
            if self.data.dtype is not 'float':
                self.data = self.data.astype('float') / 32768.0

        self.sp = SignalProc.SignalProc([],0,256,128)
        self.WaveletFunctions = WaveletFunctions.WaveletFunctions(data=data, wavelet=wavelet,maxLevel=20)
        self.segmenter = Segment.Segment(data, None, self.sp, sampleRate, window_width=256, incr=128, mingap=mingap, minlength=minlength)

    def computeWaveletEnergy(self,data=None,sampleRate=0,nlevels=5):
        """ Computes the energy of the nodes in the wavelet packet decomposition
        # There are 62 coefficients up to level 5 of the wavelet tree (without root), and 300 seconds in 5 mins
        # Hence coefs would then be a 62*300 matrix
        # The energy is the sum of the squares of the data in each node divided by the total in that level of the tree as a percentage.
        """

        if data is None:
            data = self.data
            sampleRate = self.sampleRate

        n = int(len(data)/sampleRate)

        coefs = np.zeros((2**(nlevels+1)-2, n))
        for t in range(n):
            E = []
            for level in range(1,nlevels+1):
                wp = pywt.WaveletPacket(data=data[t * sampleRate:(t + 1) * sampleRate], wavelet=self.WaveletFunctions.wavelet, mode='symmetric', maxlevel=level)
                e = np.array([np.sum(n.data**2) for n in wp.get_level(level, "natural")])
                if np.sum(e)>0:
                    e = 100.0*e/np.sum(e)
                E = np.concatenate((E, e),axis=0)
            coefs[:, t] = E
        return coefs

    def fBetaScore(self,annotation, predicted,beta=2):
        """ Computes the beta scores given two sets of predictions """
        TP = np.sum(np.where((annotation==1)&(predicted==1),1,0))
        T = np.sum(annotation)
        P = np.sum(predicted)
        if T != 0:
            recall = float(TP)/T #TruePositive/#True
        else:
            recall = None
        if P!=0:
            precision = float(TP)/P #TruePositive/#Positive
        else:
            precision = None
        if recall != None and precision != None and not (recall==0 and precision==0):
            fB=((1.+beta**2)*recall*precision)/(recall + beta**2*precision)
        else:
            fB=None
        if recall==None and precision==None:
            print("TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%s \tPrecision=%s \tfB=%s" %(TP,P-TP,len(annotation)-(P+T-TP),T-TP,recall,precision,fB))
        elif recall==None:
            print("TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%s \tPrecision=%0.2f \tfB=%s" %(TP,P-TP,len(annotation)-(P+T-TP),T-TP,recall,precision,fB))
        elif precision==None:
            print("TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%0.2f \tPrecision=%s \tfB=%s" %(TP,P-TP,len(annotation)-(P+T-TP),T-TP,recall,precision,fB))
        elif fB==None:
            print("TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%0.2f \tPrecision=%0.2f \tfB=%s" %(TP,P-TP,len(annotation)-(P+T-TP),T-TP,recall,precision,fB))
        else:
            print("TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%0.2f \tPrecision=%0.2f \tfB=%0.2f" %(TP,P-TP,len(annotation)-(P+T-TP),T-TP,recall,precision,fB))
        #print TP, int(T), int(P), recall, precision, ((1.+beta**2)*recall*precision)/(recall + beta**2*precision)
        return fB,recall, TP,P-TP,len(annotation)-(P+T-TP), T-TP    # fB, recall, TP, FP, TN, FN

    def compute_r(self,annotation,waveletCoefs,nNodes=10):
        """ Computes the point-biserial correlations for a set of labels and a set of wavelet coefficients.
        r = (M_p - M_q) / S * sqrt(p*q), M_p = mean for those that are 0, S = std dev overall, p = proportion that are 0.
        """
        w0 = np.where(annotation==0)[0]
        w1 = np.where(annotation==1)[0]

        r = np.zeros(62)
        for count in range(62):
            r[count] = (np.mean(waveletCoefs[count,w1]) - np.mean(waveletCoefs[count,w0]))/np.std(waveletCoefs[count,:]) * np.sqrt(len(w0)*len(w1))/len(annotation)

        order = np.argsort(r)
        order = order[-1:-nNodes-1:-1]

        return order

    def sortListByChild(self,order):
        """ Inputs is a list sorted into order of correlation.
        This functions resort so that any children of the current node that are in the list go first.
        Assumes that there are five levels in the tree (easy to extend, though)
        """

        newlist = []
        currentIndex = 0
        # Need to keep track of where each level of the tree starts
        # Note that there is no root to the tree, hence the 0 then 2
        starts = [0, 2, 6, 14,30,62]
        while len(order)>0:
            if order[0]<30:
                # It could have children lower down the list
                # Build a list of the children of the first element of order
                level = int(np.log2(order[0]+2))
                nc = 2
                first = order[0]
                for l in range(level+1,6):
                    children = []
                    current = currentIndex
                    for i in range(nc):
                        children.append(starts[l] + 2*(first-starts[l-1])+i)
                    nc*=2
                    first = starts[l] + 2*(first-starts[l-1])
                    # Have to do it this annoying way since Python seems to ignore the next element if you delete one while iterating over the list
                    i=0
                    order_sub = []
                    while i < len(children):
                        if children[i] not in order:
                            del(children[i])
                        else:
                            order_sub.append(order.index(children[i]))
                            i+=1

                    # Sort into order
                    children = [x for (y, x) in sorted(zip(order_sub, children), key=lambda pair: pair[0])]

                    for a in children:
                        # If so, remove and insert at the current location in the new list
                        newlist.insert(current,a)
                        current += 1
                        order.remove(a)

            # Finally, add the first element
            newlist.append(order[0])
            currentIndex = newlist.index(order[0])+1
            del(order[0])

        return newlist

    # def detectCalls(self,wp,node,sampleRate=0,n=300,species='Kiwi',thr=0):
    #     """ For a given node of the wavelet tree, make the tree, detect calls """
    #     # TODO: Pass in (or load) a species-specific threshold and use that
    #
    #     if sampleRate==0:
    #         sampleRate=self.sampleRate
    #
    #     if thr==0:
    #         if species.title() == 'Sipo':
    #             thr = 0.25
    #         else:
    #             thr = 1.0
    #
    #     # Add relevant nodes to the wavelet packet tree and then reconstruct the data
    #     new_wp = pywt.WaveletPacket(data=None, wavelet=self.WaveletFunctions.wavelet, mode='symmetric')
    #
    #     for level in range(wp.maxlevel + 1):
    #         for n in new_wp.get_level(level, 'natural'):
    #             n.data = np.zeros(len(wp.get_level(level, 'natural')[0].data))
    #
    #     bin = self.WaveletFunctions.ConvertWaveletNodeName(node)
    #     new_wp[bin] = wp[bin].data
    #
    #     # Get the coefficients
    #     C = np.abs(new_wp.reconstruct(update=True))
    #     N = len(C)
    #
    #     # Compute the number of samples in a window -- species specific
    #     # TODO: set M in relation to the call length, M can make a big difference in results
    #     if species.title()=='Sipo':
    #         M=int(0.2*sampleRate/2.0)
    #     else:
    #         M = int(0.8*sampleRate/2.0)
    #     #print M
    #
    #     # Compute the energy curve (a la Jinnai et al. 2012)
    #     E = np.zeros(N)
    #     E[M] = np.sum(C[:2 * M+1])
    #     for i in range(M + 1, N - M):
    #         E[i] = E[i - 1] - C[i - M - 1] + C[i + M]
    #     E = E / (2. * M)
    #
    #     threshold = np.mean(C) + np.std(C)*thr
    #
    #     # bittern
    #     # TODO: test
    #     #thresholds[np.where(waveletCoefs<=32)] = 0
    #     #thresholds[np.where(waveletCoefs>32)] = 0.3936 + 0.1829*np.log2(np.where(waveletCoefs>32))
    #
    #     # If there is a call anywhere in the window, report it as a call
    #     E = np.where(E<threshold, 0, 1)
    #     detected = np.zeros(n)
    #     j = 0
    #     for i in range(0,N-sampleRate,sampleRate):
    #         detected[j] = np.max(E[i:i+sampleRate])
    #         j+=1
    #
    #     return detected

    def detectCalls(self, wp, sampleRate, listnodes=[], thr=None, M=None, spInfo={},trainTest=False):
        """ Given a parameter combination (thr x M) and a wavelet packet tree:
            take relevant nodes from the wavelet packet tree, reconstruct the data from it,
            identify detections based on thr and M.
            Return value: 1D vector of detections over time blocks (seconds), for this file.
        """
        st = time.time()
        if sampleRate==0:
            sampleRate=self.sampleRate
        if thr is None:
            thr = spInfo['WaveletParams'][0]
        if M is None:
            M = spInfo['WaveletParams'][1]

        # Reconstruct data from a limited WP tree
        new_wp = pywt.WaveletPacket(data=None, wavelet=wp.wavelet, mode='symmetric', maxlevel=wp.maxlevel)
        print("ch detectcalls 1 prebuilt tree", time.time() - st)
        for index in listnodes:
            binNodeId = self.WaveletFunctions.ConvertWaveletNodeName(index)
            new_wp[binNodeId] = wp[binNodeId].data

        print("ch detectcalls 2 prep for rec", time.time() - st)
        # Get the coefficients
        C = new_wp.reconstruct(update=True)
        # filter
        C = self.sp.ButterworthBandpass(C, self.sampleRate, low=spInfo['FreqRange'][0],high=spInfo['FreqRange'][1],order=10)
        C = np.abs(C)
        N = len(C)
        print("ch detectcalls 3 wp rec for file", time.time() - st)

        # Compute the number of samples in a window -- species specific
        M = int(M * sampleRate / 2.0)

        # Compute the energy curve (a la Jinnai et al. 2012)
        E = ce.EnergyCurve(C, M)

        # thrLevel=len(bin)
        # if thrLevel==1 or thrLevel==2 or thrLevel==3:
        #     threshold = np.mean(C) + np.std(C)*thr
        # elif thrLevel == 4:
        #     threshold = np.mean(C) + np.std(C) * thr/2
        # elif thrLevel == 5:
        #     threshold = np.mean(C) + np.std(C) * thr/4
        threshold = np.mean(C) + np.std(C) * thr
        # if species.title() == 'Bittern':
        #     threshold = 0.3936 + 0.1829 * (math.log(len(C),10) / math.log(2,10));

        # bittern
        # TODO: test
        #thresholds[np.where(waveletCoefs<=32)] = 0
        #thresholds[np.where(waveletCoefs>32)] = 0.3936 + 0.1829*np.log2(np.where(waveletCoefs>32))

        # If there is a call anywhere in the window, report it as a call
        detected = np.zeros(int(N/sampleRate))
        j = 0
        for i in range(0,N-sampleRate,sampleRate):
            detected[j] = np.any(E[i:max(i+sampleRate, N)]>threshold)
            j+=1
        print("ch in 5", time.time() - st)

        # detected[0]=0       # to avoid two FPs usually occur at the start and end of the recording
        # detected[-1]=0
        return detected

    def identifySegments(self, seg): #, maxgap=1, minlength=1):
    # TODO: *** Replace with segmenter.checkSegmentLength(self,segs, mingap=0, minlength=0, maxlength=5.0)
        segments = []
        # print seg, type(seg)
        if len(seg)>0:
            for s in seg:
                segments.append([s, s+1])
        return segments

    # Usage functions
    def preprocess(self, spInfo, df=False):
        # set df=True to perform both denoise and filter
        # df=False to skip denoise
        fs = spInfo['SampleRate']

        if self.sampleRate != fs:
            self.data = librosa.core.audio.resample(self.data, self.sampleRate, fs)
            self.sampleRate = fs

        # Get the five level wavelet decomposition
        if df == True:
            denoisedData = self.WaveletFunctions.waveletDenoise(self.data, thresholdType='soft', wavelet=self.WaveletFunctions.wavelet,maxLevel=5)
        else:
            denoisedData=self.data  # this is to avoid washing out very fade calls during the denoising

        filteredDenoisedData = self.sp.ButterworthBandpass(denoisedData, self.sampleRate, low=spInfo['FreqRange'][0], high=spInfo['FreqRange'][1])
        return filteredDenoisedData

    def waveletSegment_train(self, dirName, thrList, MList, spInfo={}, df=False):
        """ Take list of files and other parameters,
            load files, compute correlation for wavelets,
            perform grid search over thr and M parameters,
            do a stepwise search for best nodes.

            Let df=true (denoise during preprocess) for bittern, df=false for others
            A lot of stuff is stored in memory currently, but that's OK:
            1 min wav ~= 3.7 MB, 1h ~= 220 MB, so storing ~10 h isn't a problem on any desktop.
        """
        opstartingtime = time.time()
        # 1. Load each file, generate point-biserial correlations for the nodes.
        for root, dirs, files in os.walk(str(dirName)):
            for file in files:
                if file.endswith('.wav') and os.stat(root + '/' + file).st_size != 0 and file[:-4] + '-sec.txt' in files and file + '.data' in files:
                    wavFile = root + '/' + file[:-4]
                    self.loadData(wavFile)
                    # Load data and annotation
                    # (preprocess only requires SampleRate and FreqRange from spInfo)
                    filteredDenoisedData = self.preprocess(spInfo,df=df)
                    self.audioList.append(filteredDenoisedData)
                    # Compute energy in each WP node and store
                    self.waveletCoefs.extend(self.computeWaveletEnergy(filteredDenoisedData, self.sampleRate))
                    print("ch 1, loading completed", time.time() - opstartingtime)

        # Compute point-biserial correlations and sort wrt it, return top nNodes
        # (limit number of nodes to 10 and avoid getting in low level nodes)
        self.annotation = np.array(self.annotation)
        self.waveletCoefs = np.array(self.waveletCoefs)
        nodes = self.compute_r(self.annotation, self.waveletCoefs, nNodes=10)

        # Now for Nirosha's sorting
        # Basically, for each node, put any of its children (and their children, iteratively) that are in the list in front of it
        nodes = self.sortListByChild(np.ndarray.tolist(nodes))

        # These nodes refer to the unrooted tree, so add 1 to get the real indices
        nodes = [n + 1 for n in nodes]
        print("ch 2, found r2", time.time() - opstartingtime)

        # Now check the F2 values and add node if it improves F2
        m = len(self.annotation)
        # output structure: 1. 2d list of [nodes]
        # (1st d runs over M, 2nd d runs over thr)
        # 2-5. 2d np arrays of TP/FP/TN/FN
        shape = (len(MList), len(thrList))
        tpa = np.zeros(shape)
        fpa = np.zeros(shape)
        tna = np.zeros(shape)
        fna = np.zeros(shape)
        finalnodes = []

        # Grid search over M x thr
        for indexM in range(len(MList)):
            finalnodesT = []
            for indext in range(len(thrList)):
                # Stepwise search over nodes:
                # test with node 0, then nodes 0,1, then nodes 0,1,2 ...
                # (here 0,1,2,...,9 - 10 best nodes from above)
                oldDetections = np.zeros(m)
                listnodes = []
                bestBetaScore = 0
                bestRecall=0
                for node in nodes:
                    testlist = listnodes[:]
                    testlist.append(node)
                    print("Test list: ", testlist)
                    # detectCalls returns 1d vector for each file, over s
                    newDetections = np.array([])
                    for fileId in range(len(self.audioList)):
                        wp = pywt.WaveletPacket(data=self.audioList[fileId], wavelet=self.WaveletFunctions.wavelet, mode='symmetric', maxlevel=5)
                        detected_c = self.detectCalls(wp, self.sampleRate, listnodes=testlist, thr=thrList[indext], M=MList[indexM], spInfo=spInfo, trainTest=True)
                        newDetections = np.concatenate((newDetections, detected_c))
                        # memory cleanup:
                        wp = []
                        del wp
                        gc.collect()

                    # OR over old and new detections
                    if np.shape(newDetections) != np.shape(oldDetections):
                        print("ERROR: detection result dimensions do not match: ", np.shape(oldDetections), np.shape(newDetections))
                        break
                    newDetections = np.maximum.reduce([oldDetections, newDetections])
                    fB,recall,tp,fp,tn,fn = self.fBetaScore(self.annotation, newDetections)
                    print("fB, recall: ", fB,recall)


                    # if current node improves score, append it to list and update maximums:
                    if fB is not None and fB > bestBetaScore:
                        bestBetaScore = fB
                        bestRecall = recall
                        oldDetections = newDetections
                        listnodes.append(node)
                    # if perfect score reached, no need to continue with other nodes:
                    if bestBetaScore == 1 or bestRecall == 1:
                        break
                    print("ch evaluation done", time.time() - opstartingtime)
                # one iteration done, store results
                finalnodesT.append(listnodes)
                tpa[indexM, indext] = tp
                fpa[indexM, indext] = fp
                tna[indexM, indext] = tn
                fna[indexM, indext] = fn
                print("Iteration t %d/%d, M %d/%d complete\n" %(indext+1, len(thrList), indexM+1, len(MList)))
            # one row done, store nodes
            finalnodes.append(finalnodesT)
        return finalnodes, tpa, fpa, tna, fna

    def waveletSegment_test(self,dirName=None, data=None, sampleRate=None, listnodes = None, spInfo={}, trainTest=False, df=False):
        print("Wavelet testing with annotation")
        print(np.sum(self.annotation==1), np.sum(self.annotation==0))

        # Load the relevant list of nodes
        if listnodes is None:
            nodes = spInfo['WaveletParams'][2]
        else:
            nodes = listnodes

        if dirName != None:
            # clear storage for multifile processing
            self.annotation = []
            self.audioList = []

            # populate storage
            for root, dirs, files in os.walk(str(dirName)):
                for file in files:
                    if file.endswith('.wav') and os.stat(root + '/' + file).st_size != 0 and file[:-4] + '-sec.txt' in files and file + '.data' in files:
                        wavFile = root + '/' + file[:-4]
                        # Load data and annotation
                        # (preprocess only requires SampleRate and FreqRange from spInfo)
                        self.loadData(wavFile, trainTest)
                        filteredDenoisedData = self.preprocess(spInfo,df=df)
                        self.audioList.append(filteredDenoisedData)
        else:
            self.data = data
            self.sampleRate = sampleRate
            filteredDenoisedData = self.preprocess(spInfo=spInfo, df=df)

        wpFull = pywt.WaveletPacket(data=filteredDenoisedData, wavelet=self.WaveletFunctions.wavelet, mode='symmetric', maxlevel=5)
        detected = self.detectCalls(wpFull, self.sampleRate, listnodes=nodes, spInfo=spInfo, trainTest=trainTest)

        # Todo: remove clicks

        if trainTest:
            if np.all(self.annotation==0):
                return detected, 0, 0, 0, 0
            else:
                fB, recall, TP, FP, TN, FN = self.fBetaScore(self.annotation, detected)

        # merge neighbours in order to convert the detections into segments
        detected = np.where(detected > 0)
        if np.shape(detected)[1] > 1:
            detected = self.identifySegments(np.squeeze(detected))
        elif np.shape(detected)[1] == 1:
            detected = np.array(detected).flatten().tolist()
            detected = self.identifySegments(detected)
        else:
            detected = []
        detected = self.mergeSeg(detected)

        if trainTest==True:
            return detected, TP, FP, TN, FN
        else:
            return detected

    def mergeSeg(self,detected):
        # Merge the neighbours, for now wavelet segments
        #     # **** Replace with segmenter.identifySegments(self, seg, maxgap=1, minlength=1,notSpec=False):
        indx = []
        for i in range(len(detected) - 1):
            if detected[i][1] == detected[i + 1][0]:
                indx.append(i)
        indx.reverse()
        for i in indx:
            detected[i][1] = detected[i + 1][1]
            del (detected[i + 1])
        return detected

    def loadData(self,fName,trainTest=True):
        # Load data
        filename = fName+'.wav' #'train/kiwi/train1.wav'
        filenameAnnotation = fName+'-sec.txt'#'train/kiwi/train1-sec.txt'
        try:
            wavobj = wavio.read(filename)
        except:
            print("unsupported file: ", filename)
            pass
        self.sampleRate = wavobj.rate
        self.data = wavobj.data
        if self.data.dtype is not 'float':
            self.data = self.data.astype('float') #/ 32768.0
        if np.shape(np.shape(self.data))[0]>1:
            self.data = np.squeeze(self.data[:,0])
        n=int(len(self.data)/self.sampleRate)

        fileAnnotations = []
        if trainTest==True:     #survey data don't have annotations
            # Get the segmentation from the txt file
            with open(filenameAnnotation) as f:
                reader = csv.reader(f, delimiter="\t")
                d = list(reader)
            d = d[:-1]
            if len(d) != n:
                print("ERROR: annotation length %d does not match file duration %d!" %(len(d), n))
                self.annotation = None
                return
            # for each second, store 0/1 presence:
            sum = 0
            for row in d:
                fileAnnotations.append(int(row[1]))
                sum += int(row[1])
            # annotations are stored in two formats: per-file and all
            self.annotation.extend(fileAnnotations)
            print("%d blocks read, %d presence blocks found. %d blocks stored so far.\n" % (n, sum, len(self.annotation)))
