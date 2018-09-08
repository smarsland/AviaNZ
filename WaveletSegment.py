# Version 0.3 14/8/17
# Author: Stephen Marsland

import pywt
import WaveletFunctions
import wavio
import numpy as np
import json
import SignalProc
import Segment
import librosa

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

# TODO: Inconsisient about symmetric or zeros for the wavelet packet
# TODO: This still needs some tidying up
# TODO: Make a dictionary and json it with species params in (read-only)

class WaveletSegment:
    # This class implements wavelet segmentation for the AviaNZ interface

    def __init__(self,data=[],sampleRate=0,species=None,wavelet='dmey2',annotation=None,mingap=0.3,minlength=0.2):
        self.species=species
        self.annotation=annotation
        if data != []:
            self.data = data
            self.sampleRate = sampleRate
            if self.data.dtype is not 'float':
                self.data = self.data.astype('float') / 32768.0

        # TODO: What else should be in there? mingap, minlength?
        # TODO: Weights for learning alg?
        #speciesdata = json.load(open('species.data'))
        #self.listnodes = speciesdata[species][0]
        #self.listbands = speciesdata[species][1]

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

        n=int(len(data)/sampleRate)

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

    def compute_r(self,annotation,waveletCoefs,nNodes=20):
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

    def detectCalls(self,wp,sampleRate, listnodes=[], species=[],trainTest=False):
        #for test recordings given the set of nodes
        # Add relevant nodes to the wavelet packet tree and then reconstruct the data
        import math
        if sampleRate==0:
            sampleRate=self.sampleRate
        thr = species[7]
        detected = np.zeros((int(len(wp.data)/sampleRate),len(listnodes)))
        count = 0

        for index in listnodes:
            new_wp = pywt.WaveletPacket(data=None, wavelet=wp.wavelet, mode='symmetric', maxlevel=wp.maxlevel)

            # TODO: This primes the tree with zeros, which was necessary for denoising.
            # for level in range(wp.maxlevel + 1):
            #    for n in new_wp.get_level(level, 'natural'):
            #        n.data = np.zeros(len(wp.get_level(level, 'natural')[0].data))

            bin = self.WaveletFunctions.ConvertWaveletNodeName(index)
            new_wp[bin] = wp[bin].data

            # # Get the coefficients
            # C = np.abs(new_wp.reconstruct(update=True))
            # get the coefficients
            C = new_wp.reconstruct(update=True)
            # filter
            C = self.sp.ButterworthBandpass(C, self.sampleRate, low=self.species[2],high=self.species[3],order=10)
            C = np.abs(C)
            N = len(C)

            # Compute the number of samples in a window -- species specific
            M = int(species[8] * sampleRate / 2.0)
            # Compute the energy curve (a la Jinnai et al. 2012)
            E = np.zeros(N)
            E[M] = np.sum(C[:2 * M+1])
            for i in range(M + 1, N - M):
                E[i] = E[i - 1] - C[i - M - 1] + C[i + M]
            E = E / (2. * M)

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
            E = np.where(E<threshold, 0, 1)
            j = 0
            for i in range(0,N-sampleRate,sampleRate):
                detected[j,count] = np.max(E[i:i+sampleRate])
                j+=1
            count += 1

        detected= np.max(detected,axis=1)
        # detected[0]=0       # to avoid two FPs usually occur at the start and end of the recording
        # detected[-1]=0
        return detected
        # if trainTest==True:
        #     return detected
        # else:
        #     detected=np.where(detected>0)
        #     # print "det",detected
        #     if np.shape(detected)[1]>1:
        #         return self.identifySegments(np.squeeze(detected))
        #     elif np.shape(detected)[1]==1:
        #         return self.identifySegments(detected)
        #     else:
        #         return []

    def identifySegments(self, seg): #, maxgap=1, minlength=1):
    # TODO: *** Replace with segmenter.checkSegmentLength(self,segs, mingap=0, minlength=0, maxlength=5.0)
        segments = []
        # print seg, type(seg)
        if len(seg)>0:
            for s in seg:
                segments.append([s, s+1])
        return segments

    # def mergeSeg(self,segments):
    #     # **** Replace with segmenter.identifySegments(self, seg, maxgap=1, minlength=1,notSpec=False):
    #     """ Combines segments from the wavelet segmenter."""
    #     indx = []
    #     for i in range(len(segments) - 1):
    #         if segments[i][1] == segments[i + 1][0]:
    #             indx.append(i)
    #     indx.reverse()
    #     for i in indx:
    #         segments[i][1] = segments[i + 1][1]
    #         del (segments[i + 1])
    #     return segments

    # Usage functions
    def preprocess(self, species, df=False):
        # set df=True to perform both denoise and filter
        # df=False to skip denoise
        f1 = species[2]
        f2 = species[3]
        fs = species[4]

        if self.sampleRate != fs:
            self.data = librosa.core.audio.resample(self.data, self.sampleRate, fs)
            self.sampleRate = fs

        # Get the five level wavelet decomposition
        if df == True:
            denoisedData = self.WaveletFunctions.waveletDenoise(self.data, thresholdType='soft', wavelet=self.WaveletFunctions.wavelet,maxLevel=5)
        else:
            denoisedData=self.data  # this is to avoid washing out very fade calls during the denoising

        # # Denoise each 10 secs and merge
        # denoisedData = []
        # n = len(self.data)
        # dLen=10*self.sampleRate
        # for i in range(0,n,dLen):
        #     temp = self.WaveletFunctions.waveletDenoise(self.data[i:i+dLen], thresholdType='soft', wavelet=self.WaveletFunctions.wavelet,maxLevel=5)
        #     denoisedData.append(temp)
        # import itertools
        # denoisedData = list(itertools.chain(*denoisedData))
        # denoisedData = np.asarray(denoisedData)
        # wavio.write('../Sound Files/Kiwi/test/Tier1/test/test/test/test_whole.wav', denoisedData, self.sampleRate, sampwidth=2)
        # librosa.output.write_wav('Sound Files/Kiwi/test/Tier1/test/test/test', denoisedData, self.sampleRate, norm=False)

        filteredDenoisedData = self.sp.ButterworthBandpass(denoisedData, self.sampleRate, low=species[2], high=species[3])
        return filteredDenoisedData

    def waveletSegment_train(self,fName, species=[], df=False):
        # Let df=true (denoise during preprocess) for bittern, df=false for others
        # Load data and annotation
        self.loadData(fName)
        # print(self.annotation)
        filteredDenoisedData = self.preprocess(species,df=df)    # skip denoising
        # print("denoising completed")
        # print("inside waveletSegment_train fs= ", self.sampleRate)
        waveletCoefs = self.computeWaveletEnergy(filteredDenoisedData, self.sampleRate)

        # Compute point-biserial correlations and sort wrt it, return top nNodes
        nodes = self.compute_r(self.annotation, waveletCoefs)
        # print(nodes)

        # Now for Nirosha's sorting
        # Basically, for each node, put any of its children (and their children, iteratively) that are in the list in front of it
        nodes = self.sortListByChild(np.ndarray.tolist(nodes))

        # These nodes refer to the unrooted tree, so add 1 to get the real indices
        nodes = [n + 1 for n in nodes]
        # print(nodes)

        # Generate a full 5 level wavelet packet decomposition
        wpFull = pywt.WaveletPacket(data=filteredDenoisedData, wavelet=self.WaveletFunctions.wavelet, mode='symmetric', maxlevel=5)

        # Now check the F2 values and add node if it improves F2
        listnodes = []
        bestBetaScore = 0
        bestRecall=0
        m = int(len(filteredDenoisedData) / self.sampleRate)
        detected = np.zeros(m)

        for node in nodes:
            testlist = listnodes[:]
            testlist.append(node)
            print(testlist)
            detected_c = self.detectCalls(wpFull, self.sampleRate, listnodes=testlist, species=species,trainTest=True)

            # update the detections
            detections = np.maximum.reduce([detected, detected_c])
            fB,recall,tp,fp,tn,fn = self.fBetaScore(self.annotation, detections)
            # print("Node,", node)
            # print("fB, recall: ", fB,recall)
            if fB > bestBetaScore:
                bestBetaScore = fB
                bestRecall=recall
                # now apend the detections of node c to detected
                detected = detections
                listnodes.append(node)
            if bestBetaScore == 1 or bestRecall == 1:
                break

        # TODO: json.dump('species.data', open('species.data', 'wb'))
        return listnodes

    def waveletSegment_test(self,fName=None, data=None, sampleRate=None, listnodes = None, spInfo=[], trainTest=False, df=False):
        # Load the relevant list of nodes
        # TODO: Put these into a file along with other relevant parameters (frequency, length, etc.)
        if listnodes is None:
            nodes = spInfo[9]
        else:
            nodes = listnodes

        if fName != None:
            self.loadData(fName, trainTest)
        else:
            self.data = data
            self.sampleRate = sampleRate

        filteredDenoisedData = self.preprocess(species=spInfo, df=df)

        wpFull = pywt.WaveletPacket(data=filteredDenoisedData, wavelet=self.WaveletFunctions.wavelet, mode='symmetric', maxlevel=5)

        detected = self.detectCalls(wpFull, self.sampleRate, listnodes=nodes, species=spInfo, trainTest=trainTest)

        # Todo: remove clicks

        if trainTest == True:
            # print fName
            fB, recall, TP, FP, TN, FN = self.fBetaScore(self.annotation, detected)

        # merge neighbours in order to convert the detections into segments
        detected = np.where(detected > 0)
        # print "det",detected
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
        filenameAnnotation = fName+'-sec.txt'#'train/kiwi/train1-sec.xlsx'
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

        if trainTest==True:     #survey data don't have annotations
            # Get the segmentation from the txt file
            import csv
            self.annotation = np.zeros(n)
            count = 0
            with open(filenameAnnotation) as f:
                reader = csv.reader(f, delimiter="\t")
                d = list(reader)
            for row in range(0,n):
                self.annotation[count]=d[row][1]
                count += 1

def batch(dirName,species,ws,listnodes,train=False,df=False):
    import os
    nodeList=[]
    TP=FP=TN=FN=0
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.endswith('.wav'):
                filename = root + '/' + filename[:-4]
                if not train:
                    print("***", filename)
                    det, tp, fp, tn, fn = ws.waveletSegment_test(filename, listnodes=listnodes, species=species, trainTest=True,df=df)
                    TP+=tp
                    FP+=fp
                    TN+=tn
                    FN+=fn
                else:
                    print("***", filename)
                    nodes = ws.waveletSegment_train(filename, species=species,df=df)
                    print(nodes)
                    nodeList=np.union1d(nodeList, nodes)
    if train:
        print('----- wavelet nodes for the species', nodeList)
    else:
        print("-----TP   FP  TN  FN")
        print(TP, FP, TN, FN)

ws=WaveletSegment(wavelet='dmey')
# batch('Sound Files/test/test', ws, None)
#train bittern
# batch('Sound Files/Bittern/thesis-Hatuma/train','Bittern',ws,listnodes=None,train=True)
# bittern_nodes=[41,43,44,45,46]
# bittern_nodes=[4, 21,43, 44, 45, 46]
bittern_nodes=[10,39, 40, 41, 42, 43, 44, 45, 46]
# batch('E:/AviaNZ/Sound Files/Bittern/kessel/KA13_Oct 17-24_down','Bittern',ws,listnodes=bittern_nodes,train=False,df=True)
# batch('E:/Employ/Halema/Survey2/Card 1/newTrain','Kiwi',ws,listnodes=None,train=True,df=False)
# batch('E:/Employ/Halema/Survey2/Card 1/newTrain','Kiwi',ws,listnodes=None,train=False,df=False)

# detect(dirName='E:/Employ/Halema/Survey2/Card 1/rerun', trainTest=False, species='Kiwi')


## Testing e-ratio
# ws=WaveletSegment(wavelet='dmey')
# segments=ws.waveletSegment_test('Sound Files\Kiwi\\test\Tier1\\xx\\xx\BX23_BIRA_150107_225906', trainTest=True)
#
# sp = SignalProc.SignalProc(ws.data, ws.sampleRate, 256, 128)
# ws.sg = sp.spectrogram(ws.data)
#
# f1 = 1100
# f2 = 4000
# print "eRatio2"
# for seg in segments:
#     e = np.sum(ws.sg[seg[0] * ws.sampleRate / 128:seg[1] * ws.sampleRate / 128, :]) /128     # whole frequency range
#     nBand = 128  # number of frequency bands
#     #e = np.sum(ws.sg[seg[0] * ws.sampleRate / 128:seg[1] * ws.sampleRate / 128,
#     #           f2 * 128 / (ws.sampleRate / 2):])  # f2:
#     #nBand = 128 - f2 * 128 / (ws.sampleRate / 2)  # number of frequency bands
#     e = e / nBand  # per band power
#
#     eBand = np.sum(ws.sg[seg[0] * ws.sampleRate / 128:seg[1] * ws.sampleRate / 128,
#                    f1 * 128 / (ws.sampleRate / 2):f2 * 128 / (ws.sampleRate / 2)])  # f1:f2
#     nBand = f2 * 128 / (ws.sampleRate / 2) - f1 * 128 / (ws.sampleRate / 2)
#     eBand = eBand / nBand
#     r = eBand / e
#     print seg, r
#
# print "eRatio1"
# for seg in segments:
#     #e = np.sum(ws.sg[seg[0] * ws.sampleRate / 128:seg[1] * ws.sampleRate / 128, :]) / 128  # whole frequency range
#     #nBand = 128  # number of frequency bands
#     e = np.sum(ws.sg[seg[0] * ws.sampleRate / 128:seg[1] * ws.sampleRate / 128, f2 * 128 / (ws.sampleRate / 2):])  # f2:
#     nBand = 128 - f2 * 128 / (ws.sampleRate / 2)  # number of frequency bands
#     e = e / nBand  # per band power
#
#     eBand = np.sum(ws.sg[seg[0] * ws.sampleRate / 128:seg[1] * ws.sampleRate / 128,
#                    f1 * 128 / (ws.sampleRate / 2):f2 * 128 / (ws.sampleRate / 2)])  # f1:f2
#     nBand = f2 * 128 / (ws.sampleRate / 2) - f1 * 128 / (ws.sampleRate / 2)
#     eBand = eBand / nBand
#     r = eBand / e
#     print seg, r


    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test2',listnodes=listnodes1,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test2',listnodes=listnodes2,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test2',listnodes=listnodes3,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test2',listnodes=listnodes4,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test2',listnodes=listnodes5,trainTest=True)
    #
    #
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test3',listnodes=listnodes1,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test3',listnodes=listnodes2,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test3',listnodes=listnodes3,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test3',listnodes=listnodes4,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test3',listnodes=listnodes5,trainTest=True)
    #
    #
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test4',listnodes=listnodes1,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test4',listnodes=listnodes2,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test4',listnodes=listnodes3,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test4',listnodes=listnodes4,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test4',listnodes=listnodes5,trainTest=True)
    #
    #
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test5',listnodes=listnodes1,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test5',listnodes=listnodes2,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test5',listnodes=listnodes3,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test5',listnodes=listnodes4,trainTest=True)
    # dummy = ws.waveletSegment_test('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/test/kiwi-test5',listnodes=listnodes5,trainTest=True)

def test2():
    ws=WaveletSegment(wavelet='dmey')
    listnodes1 = ws.waveletSegment_train('/Users/srmarsla/Projects/AviaNZ/Wavelet Segmentation/kiwi/train/train1')
    print(listnodes1)

def test():
    ws=WaveletSegment()
    # listnodes2 = ws.waveletSegment_train('Sound Files/Kiwi/train/Ponui/train2')
    # print "***", listnodes2
    # listnodes3 = ws.waveletSegment_train('Sound Files/Kiwi/train/Ponui/train3')
    # print "***", listnodes3
    # listnodes4 = ws.waveletSegment_train('Sound Files/Kiwi/train/Ponui/train4')
    # print "***", listnodes4
    # listnodes6 = ws.waveletSegment_train('Sound Files/Kiwi/train/Ponui/train6')
    # print "***", listnodes6
    # listnodes8 = ws.waveletSegment_train('Sound Files/Kiwi/train/Ponui/train8')
    # print "***", listnodes8
    # listnodes9 = ws.waveletSegment_train('Sound Files/Kiwi/train/Taranaki/Omoana_230515_185924')
    # # print "***", listnodes9
    # listnodes = ws.waveletSegment_train('Sound Files/Kiwi/train/Taranaki/LK_020816_202935_p3')
    # print "***", listnodes
    # listnodes10 = ws.waveletSegment_train('Sound Files/Kiwi/train/Jason/110617_180001')
    # print "***", listnodes10
    # listnodes11 = ws.waveletSegment_train('Sound Files/Kiwi/train/Tier1/5min/CJ68_BIRD_141107_221420') # Kiwi(M)3
    # print "***", listnodes11
    # listnodes12 = ws.waveletSegment_train('Sound Files/Kiwi/train/Tier1/5min/CJ68_BIRM_141107_221420') # Kiwi(M)1
    # print "***", listnodes12
    # listnodes13 = ws.waveletSegment_train('Sound Files/Kiwi/train/Tier1/5min/CJ68_BIRX_141107_221420') # Kiwi(M)3
    # print "***", listnodes13
    # listnodes14 = ws.waveletSegment_train('Sound Files/Kiwi/train/Tier1/5min/CL78_BIRX_141121_022801') # Kiwi(M)2
    # print "***", listnodes14
    # listnodes15 = ws.waveletSegment_train('Sound Files/Kiwi/train/Tier1/CH66_BIRA_151125_012820') # Kiwi(F)3
    # print "***", listnodes15
    # listnodes16 = ws.waveletSegment_train('Sound Files/Kiwi/train/Tier1/CG68_BIRA_151214_004334') # Kiwi(M)4
    # print "***", listnodes16

    ## batch(ws, listnodes2, listnodes3, listnodes4, listnodes6, listnodes8)

def test_listmerge():
    p1 = [46, 45, 43, 38]           # Ponui train2 (female)
    p2 = [43, 44, 35, 36, 55]       # Ponui train3
    p3 = [45, 46, 42, 50]           # Ponui train4 (female)
    p4 = [35, 36, 17, 43, 40, 20]   # Ponui train6
    p5 = [35, 36]                   # Ponui train8
    h1 = [43, 36, 44, 38, 22, 40]       # Taranaki (female)
    h2 = [55, 56, 35, 43, 40, 17]       # Taranaki
    # j1 = [] # Jason
    t1 = [35, 43]      # Tier 1 (15 min)
    t2 = [35]          # Tier 1
    t3 = [35, 43]      # Tier 1
    t4 = [44, 43, 46]  # Tier 1
    t5 = [44, 48]      # Tier 1 (female-fade)
    t6 = [35, 36]      # Tier 1

    # listnodes=np.union1d(l1,np.union1d(l2,np.union1d(l3,np.union1d(l4,np.union1d(l5,np.union1d(l6,np.union1d(l7,np.union1d(l8,np.union1d(l9,np.union1d(l10,l11))))))))))
            # [17.0, 20.0, 22.0, 35.0, 36.0, 38.0, 40.0, 42.0, 43.0, 44.0, 45.0, 46.0, 50.0, 55.0]
    listnodes=np.union1d(p1,np.union1d(p2,np.union1d(p3,np.union1d(p4,np.union1d(p5,np.union1d(h1,np.union1d(h2,np.union1d(t1,np.union1d(t2,np.union1d(t3,np.union1d(t4,np.union1d(t5,t6))))))))))))
    # [17 20 22 35 36 38 40 42 43 44 45 46 48 50 55 56]
    ws = WaveletSegment()
    # batch(ws, listnodes.astype(int), 'Sound Files\Kiwi\\test\Tier1')
    batch(ws, listnodes.astype(int), 'Sound Files\Kiwi\\test\Tier1')

    # listnodes1 = np.union1d(l1,np.union1d(l2,np.union1d(l3,np.union1d(l4,l5))))
    # # Note no l4 below -> empty set
    # listnodes2 = np.intersect1d(l1,np.intersect1d(l2,np.intersect1d(l3,l5)))
    # listnodes3 = np.union1d(l1,np.union1d(l2,l3))
    # listnodes4 = np.intersect1d(l1,np.intersect1d(l2,l3))
    # a = np.arange(35,55)
    # np.random.shuffle(a)
    # listnodes5 = a[:6]
    #
    # ws = WaveletSegment()
    # batch(ws, listnodes1, listnodes2, listnodes3, listnodes4, listnodes5)

# test_listmerge()

# ws = WaveletSegment()
# dummy = ws.waveletSegment_test('Sound Files/Kiwi/train/Tier1/CG68_BIRA_151214_004334', trainTest=True)
# dummy = ws.waveletSegment_test('Sound Files/Kiwi/train/Tier1/CH66_BIRA_151125_012820', trainTest=True)
# dummy = ws.waveletSegment_test('Sound Files/Kiwi/train/Tier1/CJ68_BIRD_141107_221420', trainTest=True)
# dummy = ws.waveletSegment_test('Sound Files/Kiwi/train/Tier1/CJ68_BIRM_141107_221420', trainTest=True)
# dummy = ws.waveletSegment_test('Sound Files/Kiwi/train/Tier1/CJ68_BIRX_141107_221420', trainTest=True)
# dummy = ws.waveletSegment_test('Sound Files/Kiwi/train/Tier1/CL78_BIRX_141121_022801', trainTest=True)

def waveletSegment_train_learning(fName,species='Kiwi'):
    ws=WaveletSegment()
    f = np.genfromtxt("Sound Files\MLdata\wE.data",delimiter=',',dtype=None)
    ld = len(f[0])
    data = np.zeros((len(f),ld))

    names = []

    for i in range(len(f)):
        for j in range(ld-2):
            data[i,j] = f[i][j]
        data[i,ld-1] = f[i][ld-1]
        if not f[i][ld-2] in names:
            names.append(f[i][ld-2])
            data[i,ld-2] = len(names)
        else:
            data[i,ld-2] = names.index(f[i][ld-2])

    # Decide on a class to be the 1 to detect
    # It is choosing male kiwi as the positive class (10) here
    data[:,63] = 0
    ind = np.where(data[:,62] == 10)
    # ind = np.where(data[:,62] == 4)
    data[ind,63] = 1

    # Compute point-biserial correlations and sort wrt it, return top nNodes
    nodes = ws.compute_r(data[:,62],data[:,:62].transpose())

    # Now for Nirosha's sorting
    # Basically, for each node, put any of its children (and their children, iteratively) that are in the list in front of it
    nodes = ws.sortListByChild(np.ndarray.tolist(nodes))

    # These nodes refer to the unrooted tree, so add 1 to get the real indices
    nodes = [n + 1 for n in nodes]

    # **** We actually need the real data :(
    # Generate a full 5 level wavelet packet decomposition
    # **** load newdata, species
    f = np.genfromtxt("Sound Files\MLdata\data-1s.data",delimiter=',',dtype=None)
    f = np.squeeze(np.reshape(f,(np.shape(f)[0]*np.shape(f)[1],1)))
    #g = np.genfromtxt("Sound Files\MLdata\label-1s",delimiter=',',dtype=None)

    return f, data, nodes


def moretest():
    ws=WaveletSegment()
    wpFull = pywt.WaveletPacket(data=f, wavelet=ws.WaveletFunctions.wavelet, mode='symmetric', maxlevel=5)
    # Now check the F2 values and add node if it improves F2
    listnodes = []
    bestBetaScore = 0
    detected = np.zeros(len(data))

    for node in nodes:
        testlist = listnodes[:]
        testlist.append(node)
        print(testlist)
        detected_c = ws.detectCalls(wpFull,node,16000,n=len(data))
        #update the detections
        det=np.maximum.reduce([detected,detected_c])
        fB = ws.fBetaScore(data[:,63], det)
        if fB > bestBetaScore:
            bestBetaScore = fB
            #now apend the detections of node c to detected
            detected=det
            listnodes.append(node)
        if bestBetaScore == 1:
            break
    # listnodes_all=listnodes_all.append(listnodes)
    return listnodes

# This is a basic learner using xgboost
# It didn't seem to work well, but needs more testing
# def waveletSegment_learn(fName=None,data=None, sampleRate=None, species='kiwi',trainTest=False):
#     # import xgboost as xgb
#     # from sklearn.externals import joblib
#
#     if species == 'Kiwi (M)':
#         clf = clf_maleKiwi
#     elif species == 'Kiwi (F)':
#         clf = clf_femaleKiwi
#     elif species == 'Ruru':
#         clf = clf_ruru
#
#     # Second by second, run through the data file and compute the wavelet energy, then classify them
#     segs = []
#     for i in range(0,len(data),sampleRate):
#         currentSec = data[i:(i+1)*sampleRate]
#         # Compute wavelet energy for this second
#         E = computeWaveletEnergy_1s(currentSec, 'dmey2')    # always calculate E on row data not denoised or bp
#         E = np.ones((1,len(E))) * E
#         #segs.append(int(clf.predict(E)[0]))
#         print clf.predict(E)[0]
#         # if int(clf.predict(E)[0]) == 1:
#         #     segs.append([float(i)/sampleRate,float(i+sampleRate)/sampleRate])
#         segs.append(int(clf.predict(E)[0]))
#     print segs
#     return segs
