# Wavelet Segmentation.py

# import numpy as np
import pywt
# from scipy.io import wavfile
import wavio
import numpy as np
import librosa
import os,json
import glob
import string
import SignalProc

# Nirosha's approach of simultaneous segmentation and recognition using wavelets

# Nirosha's version:
    # (0) Bandpass filter with different parameters for each species
    # (1) 5 level wavelet packet decomposition
    # (2) Sort nodes of (1) into order by point-biserial correlation with training labels
        # This is based on energy
    # (3) Retain top nodes (up to 20)
    # (4) Re-sort to favour child nodes
    # (5) Reduce number using F_2 score
        # This is based on thresholded reconstruction
    # (6) Classify as call if OR of (5) is true
# Stephen: Think about (4), fix (0), (5), (6) -> learning!
    # MLP or decision tree
    # DT: most informative nodes?

# TODO: more testing, proper code to run experiment, work out desired output, work on parameters for each species

# a few gotchas:
    # make sure bandpass max is BELOW nyquist freq -Done
    # this is like the Matlab code, one wavelet packet at a time, unfortunately
    # resampling is in librosa - Librosa is working on Windows

class WaveletSeg:
    # This class implements wavelet segmentation for the AviaNZ interface

    def __init__(self,data=[],sampleRate=0,spp='kiwi',annotation=None):
        self.spp=spp
        self.annotation=annotation
        if data != []:
            self.data = data
            self.sampleRate = sampleRate
            if self.data.dtype is not 'float':
                self.data = self.data.astype('float') / 32768.0

        [lowd,highd,lowr,highr] = np.loadtxt('dmey.txt')
        self.wavelet = pywt.Wavelet(name="mydmey",filter_bank=[lowd,highd,lowr,highr])
        self.wavelet.orthogonal=True

        self.sp = SignalProc.SignalProc(data,sampleRate)

    def computeWaveletEnergy(self,fwData,sampleRate):
        # Get the energy of the nodes in the wavelet packet decomposition
        # There are 62 coefficients up to level 5 of the wavelet tree (without root), and 300 seconds in 5 mins
        # The energy is the sum of the squares of the data in each node divided by the total in the tree
        coefs = np.zeros((62, 300))
        for t in range(300):
            E = []
            for level in range(1,6):
                wp = pywt.WaveletPacket(data=fwData[t * sampleRate:(t + 1) * sampleRate], wavelet=self.wavelet, mode='symmetric', maxlevel=level)
                e = np.array([np.sum(n.data**2) for n in wp.get_level(level, "natural")])
                if np.sum(e)>0:
                    e = 100.0*e/np.sum(e)
                E = np.concatenate((E, e),axis=0)
            coefs[:, t] = E
        return coefs

    def fBetaScore(self,annotation, predicted,beta=2):
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
        if recall != None and precision != None:
            fB=((1.+beta**2)*recall*precision)/(recall + beta**2*precision)
        else:
            fB=None
        if recall==None and precision==None:
            print "TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%s \tPrecision=%s \tfB=%s" %(TP,P-TP,300-(P+T-TP),T-TP,recall,precision,fB)
        elif recall==None:
            print "TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%s \tPrecision=%0.2f \tfB=%s" %(TP,P-TP,300-(P+T-TP),T-TP,recall,precision,fB)
        elif precision==None:
            print "TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%0.2f \tPrecision=%s \tfB=%s" %(TP,P-TP,300-(P+T-TP),T-TP,recall,precision,fB)
        else:
            print "TP=%d \tFP=%d \tTN=%d \tFN=%d \tRecall=%0.2f \tPrecision=%0.2f \tfB=%0.2f" %(TP,P-TP,300-(P+T-TP),T-TP,recall,precision,fB)
        #print TP, int(T), int(P), recall, precision, ((1.+beta**2)*recall*precision)/(recall + beta**2*precision)
        return fB

    def compute_r(self,annotation,waveletCoefs,nNodes=20):
        # Find the correlations (point-biserial)
        # r = (M_p - M_q) / S * sqrt(p*q), M_p = mean for those that are 0, S = std dev overall, p = proportion that are 0.
        w0 = np.where(annotation==0)[0]
        w1 = np.where(annotation==1)[0]

        r = np.zeros(62)
        for count in range(62):
            r[count] = (np.mean(waveletCoefs[count,w1]) - np.mean(waveletCoefs[count,w0]))/np.std(waveletCoefs[count,:]) * np.sqrt(len(w0)*len(w1))/len(annotation)

        order = np.argsort(r)
        order = order[-1:-nNodes-1:-1]

        return order

    def sortListByChild(self,order):
        # Have a list sorted into order of correlation
        # Want to resort so that any children of the current node that are in the list go first
        # Assumes that there are five levels in the tree (easy to extend, though)

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

    def detectCalls(self,wp,node,sampleRate=0):
        if sampleRate==0:
            sampleRate=self.sampleRate
        import string
        # Add relevant nodes to the wavelet packet tree and then reconstruct the data
        new_wp = pywt.WaveletPacket(data=None, wavelet=self.wavelet, mode='symmetric')
        # First, turn the index into a leaf name.
        level = np.floor(np.log2(node))
        first = int(2**level-1)
        bin = np.binary_repr(node-first,width=int(level))
        bin = string.replace(bin,'0','a',maxreplace=-1)
        bin = string.replace(bin,'1','d',maxreplace=-1)
        #print index+1, bin
        new_wp[bin] = wp[bin].data

        # Get the coefficients
        C = np.abs(new_wp.reconstruct(update=True))
        N = len(C)

        # Compute the number of samples in a window -- species specific
        M = int(0.8*sampleRate/2.0)
        #print M

        # Compute the energy curve (a la Jinnai et al. 2012)
        E = np.zeros(N)
        E[M] = np.sum(C[:2 * M+1])
        for i in range(M + 1, N - M):
            E[i] = E[i - 1] - C[i - M - 1] + C[i + M]
        E = E / (2. * M)

        threshold = np.mean(C) + np.std(C)
        # bittern
        # TODO: test
        #thresholds[np.where(waveletCoefs<=32)] = 0
        #thresholds[np.where(waveletCoefs>32)] = 0.3936 + 0.1829*np.log2(np.where(waveletCoefs>32))

        # If there is a call anywhere in the window, report it as a call
        E = np.where(E<threshold, 0, 1)
        detected = np.zeros(300)
        j = 0
        for i in range(0,N-sampleRate,sampleRate):
            detected[j] = np.max(E[i:i+sampleRate])
            j+=1

        return detected

    def detectCalls_test(self,wp,sampleRate, listnodes=[34,35,36,38,40,41,42,43,44,45,46,55],trainTest=False): #default kiwi
        #for test recordings given the set of nodes
        import string
        # Add relevant nodes to the wavelet packet tree and then reconstruct the data
        # detected = np.zeros((300,len(listnodes)))
        detected = np.zeros((int(len(wp.data)/sampleRate),len(listnodes)))
        count = 0
        for index in listnodes:
            new_wp = pywt.WaveletPacket(data=None, wavelet=self.wavelet, mode='symmetric')
            # First, turn the index into a leaf name.
            level = np.floor(np.log2(index))
            first = int(2**level-1)
            bin = np.binary_repr(index-first,width=int(level))
            bin = string.replace(bin,'0','a',maxreplace=-1)
            bin = string.replace(bin,'1','d',maxreplace=-1)
            #print index+1, bin
            new_wp[bin] = wp[bin].data

            # Get the coefficients
            C = np.abs(new_wp.reconstruct(update=True))
            N = len(C)

            # Compute the number of samples in a window -- species specific
            M = int(0.8*sampleRate/2.0)
            #print M

            # Compute the energy curve (a la Jinnai et al. 2012)
            E = np.zeros(N)
            E[M] = np.sum(C[:2 * M+1])
            for i in range(M + 1, N - M):
                E[i] = E[i - 1] - C[i - M - 1] + C[i + M]
            E = E / (2. * M)

            threshold = np.mean(C) + np.std(C)
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
        if trainTest==True:
            return detected
        else:
            detected=np.where(detected>0)
            # print "det",detected
            return self.identifySegments(np.squeeze(detected))

    def detectCalls1(self,wp,listnodes,sampleRate):
        # This way should be the best -- reconstruct from setting all of the relevant nodes. But it gives an error message
        # about different size coefficient arrays most of the time.
        import string
        # Add relevant nodes to the wavelet packet tree and then reconstruct the data
        new_wp = pywt.WaveletPacket(data=None, wavelet=self.wavelet, mode='symmetric')

        for index in listnodes:
            # First, turn the index into a leaf name.
            # TODO: This needs checking -- are they the right nodes?
            level = np.floor(np.log2(index))
            first = 2**level-1
            bin = np.binary_repr(index-first,width=int(level))
            bin = string.replace(bin,'0','a',maxreplace=-1)
            bin = string.replace(bin,'1','d',maxreplace=-1)
            #print index+1, bin
            new_wp[bin] = wp[bin].data

        # Get the coefficients
        C = np.abs(new_wp.reconstruct(update=True))
        N = len(C)

        # Compute the number of samples in a window -- species specific
        M = int(0.8*sampleRate/2.0)
        #print M

        # Compute the energy curve (a la Jinnai et al. 2012)
        E = np.zeros(N)
        E[M] = np.sum(C[:2 * M+1])
        for i in range(M + 1, N - M):
            E[i] = E[i - 1] - C[i - M - 1] + C[i + M]
        E = E / (2. * M)

        threshold = np.mean(C) + np.std(C)
        # bittern
        # TODO: test
        #thresholds[np.where(waveletCoefs<=32)] = 0
        #thresholds[np.where(waveletCoefs>32)] = 0.3936 + 0.1829*np.log2(np.where(waveletCoefs>32))

        # If there is a call anywhere in the window, report it as a call
        E = np.where(E<threshold, 0, 1)
        detected = np.zeros(300)
        j = 0
        for i in range(0,N-sampleRate,sampleRate):
            detected[j] = np.max(E[i:i+sampleRate])
            j+=1

        return detected

    def identifySegments(self, seg): #, maxgap=1, minlength=1):
        segments = []
        if len(seg)>0:
            for s in seg:
                segments.append([s, s+1])
        return segments

    def loadData(self,fName,trainTest=True):
        # Load data
        filename = fName+'.wav' #'train/kiwi/train1.wav'
        filenameAnnotation = fName+'-sec.xlsx'#'train/kiwi/train1-sec.xlsx'
        # self.sampleRate, self.data = wavfile.read(filename)
        # if self.data.dtype is not 'float':
        #     self.data = self.data.astype('float') / 32768.0
        wavobj = wavio.read(filename)
        self.sampleRate = wavobj.rate
        self.data = wavobj.data
        if self.data.dtype is not 'float':
            self.data = self.data.astype('float') #/ 32768.0
        if np.shape(np.shape(self.data))[0]>1:
            self.data = np.squeeze(self.data[:,0])

        self.sp.setNewData(self.data,self.sampleRate)

        if trainTest==True:     #survey data don't have annotations
            # Get the segmentation from the excel file
            self.annotation = np.zeros(300)
            count = 0
            import xlrd
            wb=xlrd.open_workbook(filename = filenameAnnotation)
            ws=wb.sheet_by_index(0)
            col=ws.col(1)
            for row in range(1,301):
                self.annotation[count]=col[row].value
                count += 1
        #return self.data, self.sampleRate, self.annotation

    # def splitAudio(self,folder_to_process='Sound Files/survey'):
    #     #Split audio into 5-min to a subfolder '5min'
    #     #Alternative: read the first, second, third 5 mins from 15 min recs - in this way we dont duplicate recs
    #     os.makedirs(folder_to_process+'/5min/')
    #     for filename in glob.glob(os.path.join(folder_to_process,'*.wav')):
    #         fs,data=wavfile.read(filename) #faster than librosa #fs, data = librosa.load(filename)
    #         if data.dtype is not 'float':
    #             data = data.astype('float') / 32768.0
    #         if np.shape(np.shape(data))[0]>1:
    #             data = data[:,0]
    #         if fs!=16000:
    #             data = librosa.core.audio.resample(data,fs,16000)
    #             fs=16000
    #         file=str(filename).split('\\')[-1:][0]
    #         i=1
    #         for start in range(0,len(data),fs*60*5):
    #             fName = folder_to_process+'/5min/'+file[:-4]+'_'+str(i) + '.wav'
    #             x=data[start:start+fs*60*5]
    #             x *= 32768.0
    #             x = x.astype('int16')
    #             wavfile.write(fName,fs, x)
    #             #librosa.output.write_wav(fName, data[start:start+fs*60*5], fs)
    #             i+=1

def findCalls_train(fName,species='kiwi'):
    # Load data and annotation
    ws=WaveletSeg()
    ws.loadData(fName)
    if species=='boom':
        fs=1000
    else:
        fs = 16000
    if ws.sampleRate != fs:
        ws.data = librosa.core.audio.resample(ws.data,ws.sampleRate,fs)
        ws.sampleRate=fs

    # Get the five level wavelet decomposition
    wData = ws.sp.waveletDenoise(ws.data, thresholdType='soft', wavelet='dmey',maxlevel=5)  # wavelet='dmey2' ??
    print np.min(wData), np.max(wData)

    #librosa.output.write_wav('train/kiwi/D/', wData, sampleRate, norm=False)

    # Bandpass filter
    #fwData = bandpass(wData,sampleRate)
    # TODO: Params in here!
    # bittern
    #fwData = ButterworthBandpass(wData,sampleRate,low=100,high=400)
    # kiwi
    fwData = ws.sp.ButterworthBandpass(wData,ws.sampleRate,low=1100,high=7500)
    print fwData

    #fwData = data
    waveletCoefs = ws.computeWaveletEnergy(fwData, ws.sampleRate)

    # Compute point-biserial correlations and sort wrt it, return top nNodes
    nodes = ws.compute_r(ws.annotation,waveletCoefs)
    print nodes

    # Now for Nirosha's sorting
    # Basically, for each node, put any of its children (and their children, iteratively) that are in the list in front of it
    nodes = ws.sortListByChild(np.ndarray.tolist(nodes))

    # These nodes refer to the unrooted tree, so add 1 to get the real indices
    nodes = [n + 1 for n in nodes]
    print nodes

    # Generate a full 5 level wavelet packet decomposition
    wpFull = pywt.WaveletPacket(data=fwData, wavelet=ws.wavelet, mode='symmetric', maxlevel=5)

    # Now check the F2 values and add node if it improves F2
    listnodes = []
    bestBetaScore = 0
    detected = np.zeros(300)

    for node in nodes:
        testlist = listnodes[:]
        testlist.append(node)
        print testlist
        detected_c = ws.detectCalls(wpFull,node,ws.sampleRate)
        #update the detections
        det=np.maximum.reduce([detected,detected_c])
        fB = ws.fBetaScore(ws.annotation, det)
        if fB > bestBetaScore:
            bestBetaScore = fB
            #now apend the detections of node c to detected
            detected=det
            listnodes.append(node)
        if bestBetaScore == 1:
            break
    return listnodes

def findCalls_test(fName=None,data=None, sampleRate=None, species='kiwi',trainTest=False):
    #data, sampleRate_o, annotation = loadData(fName)
    ws=WaveletSeg()
    if fName!=None:
        ws.loadData(fName,trainTest)
    else:
        ws.data=data
        ws.sampleRate=sampleRate
    if species=='boom':
        fs=1000
    else:
        fs = 16000
    if ws.sampleRate != fs:
        ws.data = librosa.core.audio.resample(ws.data,ws.sampleRate,fs)
        ws.sampleRate=fs
    wData = ws.sp.waveletDenoise(ws.data, thresholdType='soft', maxlevel=5)
    fwData = ws.sp.ButterworthBandpass(wData,ws.sampleRate,low=1000,high=7000)
    wpFull = pywt.WaveletPacket(data=fwData, wavelet=ws.wavelet, mode='symmetric', maxlevel=5)
    detected = ws.detectCalls_test(wpFull, ws.sampleRate,trainTest=trainTest) #detect based on a previously defined nodeset, default for kiwi
    if trainTest==True:
        print fName
        ws.fBetaScore(ws.annotation, detected)
    return detected

def processFolder(folder_to_process = 'Sound Files/survey/5min', species='kiwi'):
    #process survey recordings
    nfiles=len(glob.glob(os.path.join(folder_to_process,'*.wav')))
    detected=np.zeros((nfiles,300))
    i=0
    for filename in glob.glob(os.path.join(folder_to_process,'*.wav')):
        ws=WaveletSeg()
        ws.loadData(filename[:-4],trainTest=False)
        wData = ws.waveletDenoise(ws.data, thresholdType='soft', maxlevel=5)
        fwData = ws.sp.ButterworthBandpass(wData,ws.sampleRate,low=1000,high=7000)
        wpFull = pywt.WaveletPacket(data=fwData, wavelet=self.wavelet, mode='symmetric', maxlevel=5)
        detected[i,:] = ws.detectCalls_test(wpFull, ws.sampleRate, nodelist_kiwi) #detect based on a previously defined nodeset
    return detected

def processFolder_train(folder_to_process = 'E:/SONGSCAPE/MakeExecutable/AviaNZ_12thJune/Sound Files/MLPdata/train', species='kiwi'):
    #Trainig on a set of files
    nfiles=len(glob.glob(os.path.join(folder_to_process,'*.wav')))
    for filename in glob.glob(os.path.join(folder_to_process,'*.wav')):
        nodes=findCalls_train(filename[:-4])
        print filename
        print "Node list:", nodes
        print "**********************************"

def genReport(folder_to_process,detected):
    #generate the report from the detections (yes-no)
    #ToDO: detailed report
    fnames=["" for x in range(np.shape(detected)[0])]
    presenceAbsence=["" for x in range(np.shape(detected)[0])]
    i=0
    for filename in glob.glob(os.path.join(folder_to_process,'*.wav')):
        fnames[i]=str(filename).split('\\')[-1:][0]
    for i in range(np.shape(detected)[0]):
        if sum(detected[i,:])>0:
            presenceAbsence[i]='Yes'
        else:
            presenceAbsence[i]='-'

    col_format = "{:<10}" + "," + "{:<3}" + "\n"
    with open(folder_to_process+"/presenceAbs.csv", 'w') as of:
        for x in zip(fnames,presenceAbsence):
            of.write(col_format.format(*x))

#Test
# nodelist_kiwi = [20, 31, 34, 35, 36, 38, 40, 41, 43, 44, 45, 46] # python
# nodelist_kiwi=[1,15,20,34,35,36,38,40,41,42,43,44,45,46,55] # python with new implimentation
nodelist_kiwi=[34,35,36,38,40,41,42,43,44,45,46,55] # removed first three nodes from python with new implimentation
#nodelist_kiwi = [34, 35, 36, 38, 40, 41, 42, 43, 44, 45, 46, 55] # matlab
#[36, 35, 43, 41, 38, 45, 44, 55]
#[36, 35, 43, 41, 38, 45, 44, 39, 31, 17, 21, 18, 20, 15, 8, 10, 3, 4, 1, 55]

# def test(nodelist):
#     for filename in glob.glob(os.path.join('E:/SONGSCAPE/birdscapeConda2/Sound Files/test','*.wav')):
#         findCalls_test(nodelist,filename[:-4])
#
# test(nodelist_kiwi) # TESTING

# #Survey data processing
# #First split audio into 5-min to a subfolder '5min'
# ws=WaveletSeg()
# ws.splitAudio(folder_to_process='Sound Files/survey')
# print "5-min splitting done"
# Now to process survey data
# detected=processFolder(folder_to_process='Sound Files/survey/5min', species='kiwi')
# genReport(folder_to_process='Sound Files/survey/5min',detected=detected)
# print detected

#print findCalls_train('Wavelet Segmentation/kiwi/train/train1',species='kiwi')

#ws = WaveletSeg()
#ws.loadData('Wavelet Segmentation/kiwi/train/train1')

#fs = 16000
#if ws.sampleRate != fs:
    #ws.data = librosa.core.audio.resample(ws.data, ws.sampleRate, fs)
    #ws.sampleRate = fs

# Get the five level wavelet decomposition
#wData = ws.denoise(ws.data, thresholdType='soft', maxlevel=5)
# librosa.output.write_wav('train/kiwi/D/', wData, sampleRate, norm=False)

# Bandpass filter
# fwData = bandpass(wData,sampleRate)
# TODO: Params in here!
# bittern
# fwData = ButterworthBandpass(wData,sampleRate,low=100,high=400)
# kiwi
#fwData = ws.ButterworthBandpass(wData, ws.sampleRate, low=1100, high=7500)
#print fwData

#fwData = ws.data
# fwData = data
#waveletCoefs = ws.computeWaveletEnergy(fwData, ws.sampleRate)

#np.savetxt('waveout.txt',waveletCoefs)

# findCalls_train('E:/SONGSCAPE/birdscapeConda2/Sound Files/train/kiwi/train1')

# processFolder_train()  #TRAINING


# **********************************************PREPARE DATA SET (using 5 min recordings) FOR ML************************

def computeWaveletEnergy_1s(data,wavelet):
    # Generate wavelet energy (all 62 nodes) given 1 sec data
    E=[]
    for level in range(1,6):
        if wavelet == 'dmey2':
            [lowd, highd, lowr, highr] = np.loadtxt('dmey.txt')
            wavelet = pywt.Wavelet(filter_bank=[lowd, highd, lowr, highr])
            wavelet.orthogonal=True
        wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=level)
        e = np.array([np.sum(n.data**2) for n in wp.get_level(level, "natural")])
        if np.sum(e)>0:
            e = 100.0*e/np.sum(e)
        E = np.concatenate((E, e),axis=0)
    return E

# def genWEnergy(fName,species='kiwi'):      # origin 'findCalls_train'
#     # Given the 5 min recording generates the 62x300 (level 5) wavelet energy(all) matrix
#     # Load data and annotation
#     ws=WaveletSeg()
#     ws.loadData(fName)
#     if species=='boom':
#         fs=1000
#     else:
#         fs = 16000
#     if ws.sampleRate != fs:
#         ws.data = librosa.core.audio.resample(ws.data,ws.sampleRate,fs)
#         ws.sampleRate=fs
#
#     # Get the five level wavelet decomposition
#     wData = ws.sp.waveletDenoise(ws.data, thresholdType='soft', wavelet='dmey',maxlevel=5)
#     print np.min(wData), np.max(wData)
#
#     # Bandpass filter
#     # bittern
#     #fwData = ButterworthBandpass(wData,sampleRate,low=100,high=400)
#     if species=='kiwi':
#         low=1100
#         high=7500
#     elif species=='ruru':
#         low=0
#         high=0
#     elif species=='bittern':
#         low=100
#         high=400
#     fwData = ws.sp.ButterworthBandpass(wData,ws.sampleRate,low=low,high=high)
#     print fwData
#
#     #fwData = data
#     waveletCoefs = ws.computeWaveletEnergy(fwData, ws.sampleRate)
#     return waveletCoefs
#
# def genWEnergyAllNodes(folder= 'E:/SONGSCAPE/MakeExecutable/AviaNZ_12thJune/Sound Files/MLPdata/train', species='kiwi'):
#     #Generate the wavelet energy (all nodes)
#     nfiles=len(glob.glob(os.path.join(folder,'*.wav')))
#     for filename in glob.glob(os.path.join(folder,'*.wav')):
#         E=genWEnergy(filename,species=species)
#         np.savetxt(str(filename[:-4])+'_'+ str(species)+'_DF_E_all.dat', E, delimiter=',')
#
# # Save data per 5 min
# # genWEnergyAllNodes(folder= 'E:/SONGSCAPE/MakeExecutable/AviaNZ_12thJune/Sound Files/MLPdata/train', species='kiwi')

#***************************************************************************************
# Ready DATA SET (using 5 min recordings) FOR ML

# def read_tgt(file):
#     annotation = np.zeros(300)
#     count = 0
#     import xlrd
#     wb=xlrd.open_workbook(filename = file)
#     ws=wb.sheet_by_index(0)
#     col=ws.col(1)
#     for row in range(1,301):
#         annotation[count]=col[row].value
#         count += 1
#     return annotation
#
# data=np.zeros((8*300,63))    #8 rec =8*300=2400 data points, 62 nodes + target (0/1)
# # train1
# d = np.loadtxt("Sound Files/MLPdata/train/train1_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train1-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[0:300,0:62]=d.transpose()
# data[0:300,62]=t
# # train2
# d = np.loadtxt("Sound Files/MLPdata/train/train2_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train2-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[300:600,0:62]=d.transpose()
# data[300:600,62]=t
# # train3
# d = np.loadtxt("Sound Files/MLPdata/train/train3_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train3-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[600:900,0:62]=d.transpose()
# data[600:900,62]=t
# # train4
# d = np.loadtxt("Sound Files/MLPdata/train/train4_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train4-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[900:1200,0:62]=d.transpose()
# data[900:1200,62]=t
# # train5
# d = np.loadtxt("Sound Files/MLPdata/train/train5_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train5-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[1200:1500,0:62]=d.transpose()
# data[1200:1500,62]=t
# # train6
# d = np.loadtxt("Sound Files/MLPdata/train/train6_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train6-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[1500:1800,0:62]=d.transpose()
# data[1500:1800,62]=t
# # train7
# d = np.loadtxt("Sound Files/MLPdata/train/train7_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train7-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[1800:2100,0:62]=d.transpose()
# data[1800:2100,62]=t
# # train8
# d = np.loadtxt("Sound Files/MLPdata/train/train8_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train8-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[2100:2400,0:62]=d.transpose()
# data[2100:2400,62]=t
#
# # Save it as 2400 sec *63 [62 nodes + tgt]
# np.savetxt('Sound Files/MLPdata/train/DF_E_all.dat', data, delimiter=',')


#***************************************************************************************
# Create DATA SET FOR ML - using annotations made with AviaNZ
def CreateDataSet(directory,species='kiwi'):
    #Generate the wavelet energy (all nodes)give the directory with sound and annotation
    ws=WaveletSeg()
    f2=open('Sound Files/MLdata/test.data','a')
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.wav'):
                if not os.path.isfile(root+'/'+filename+'.data'): # if no AviaNZ annotation then skip
                    continue
                ws.loadData(root+'/'+filename[:-4],trainTest=False)
                if species !='boom' and ws.sampleRate!=16000:
                    ws.data=librosa.core.audio.resample(ws.data,ws.sampleRate,16000)
                    ws.sampleRate=16000
                # Read the AviaNZ annotation
                with open(root+'/'+filename+".data") as f:
                    segments = json.load(f)
                for seg in segments:
                    # If the length of a segment is less than 1 sec make it 1sec
                    if seg[1]-seg[0]<1:
                        seg[1]=seg[0]+1
                    # Discard the tail (<1 sec)
                    seg[1]=seg[0]+np.floor(seg[1]-seg[0])
                    n=int(seg[1]-seg[0])
                    for i in range(n):
                        current=ws.data[(int(seg[0])+i)*ws.sampleRate:(int(seg[0])+(i+1))*ws.sampleRate]
                        # Compute wavelet energy for this second
                        E=computeWaveletEnergy_1s(current,'dmey2')
                        # E=genWEnergy(filename[:-4],species=species) # try later with bp filter e.g when trainig for kiwi male use 1200-7500
                        E=E.tolist()
                        spp=str(seg[4])
                        if 'Noise' in spp:
                            target=0
                        else:
                            target=1
                        E.append(target)
                        E.append(spp)
                        f2.write(str(E)[1:-1]+"\n")
    f2.close()

CreateDataSet(directory= 'E:/SONGSCAPE/AviaNZ/Sound Files/MLdata')
