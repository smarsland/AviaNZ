
import numpy as np
import pywt
from scipy.io import wavfile
import pylab as pl
import librosa

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

# TODO: more testing, proper code to run experiment, work out desired output, work on parameters for each species


# a few gotchas:
    # make sure bandpass max is BELOW nyquist freq
    # this is like the Matlab code, one wavelet packet at a time, unfortunately
    # resampling is in librosa


def denoise(data,thresholdType='soft', maxlevel=5):
    # Perform wavelet denoising. Can use soft or hard thresholding
    wp = pywt.WaveletPacket(data=data, wavelet='dmey', mode='symmetric', maxlevel=maxlevel)

    det1 = wp['d'].data
    # Note magic conversion number
    sigma = np.median(np.abs(det1)) / 0.6745
    threshold = 4.5 * sigma
    for level in range(maxlevel):
        for n in wp.get_level(level, 'natural'):
            if thresholdType == 'hard':
                # Hard thresholding
                n.data = np.where(np.abs(n.data) < threshold, 0.0, n.data)
            else:
                # Soft thresholding
                n.data = np.sign(n.data) * np.maximum((np.abs(n.data) - threshold), 0.0)

    return wp.data

def loadData(train=True):
    # Load data
    if train:
        filename = 'Wavelet Segmentation/kiwi/train/train1.wav'
        filenameAnnotation = 'Wavelet Segmentation/kiwi/train/train1-sec.xlsx'
    else:
        filename = 'Wavelet Segmentation/kiwi/test/kiwi-test1.wav'
        filenameAnnotation = 'Wavelet Segmentation/kiwi/test/kiwi-test1-sec.xlsx'

    sampleRate, audiodata = wavfile.read(filename)
    if audiodata.dtype is not 'float':
        audiodata = audiodata.astype('float') / 32768.0
    if np.shape(np.shape(audiodata))[0]>1:
        audiodata = audiodata[:,0]

    # Get the segmentation from the excel file
    annotation = np.zeros(300)
    count = 0
    import openpyxl as op
    wb = op.load_workbook(filename = filenameAnnotation)
    ws = wb.active
    for row in ws.iter_rows('B2:B301'):
        annotation[count] = row[0].value
        count += 1

    return audiodata, sampleRate, annotation

def computeWaveletEnergy(fwData,sampleRate):
    # Get the energy of the nodes in the wavelet packet decomposition
    # There are 62 coefficients up to level 5 of the wavelet tree (without root), and 300 seconds in 5 mins
    # The energy is the sum of the squares of the data in each node divided by the total in the tree
    coefs = np.zeros((62, 300))
    for t in range(300):
        E = []
        for level in range(1,6):
            wp = pywt.WaveletPacket(data=fwData[t * sampleRate:(t + 1) * sampleRate], wavelet='dmey', mode='symmetric', maxlevel=level)
            e = np.array([np.sum(n.data**2) for n in wp.get_level(level, "natural")])
            if np.sum(e)>0:
                e = 100.0*e/np.sum(e)
            E = np.concatenate((E, e),axis=0)
        coefs[:, t] = E
    return coefs

def fBetaScore(annotation, predicted,beta=2):
    TP = np.sum(np.where((annotation==1)&(predicted==1),1,0))
    T = np.sum(annotation)
    P = np.sum(predicted)
    recall = float(TP)/T #TruePositive/#True
    precision = float(TP)/P #TruePositive/#Positive
    print TP, int(T), int(P), recall, precision, ((1.+beta**2)*recall*precision)/(recall + beta**2*precision)
    return ((1.+beta**2)*recall*precision)/(recall + beta**2*precision)

def compute_r(annotation,waveletCoefs,nNodes=20):
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

def sortListByChild(order):
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

def ButterworthBandpass(data,sampleRate,order=10,low=1000,high=7000):
    import scipy.signal as signal
    nyquist = sampleRate/2.0

    low = float(low)/nyquist
    high = float(high)/nyquist
    #print nyquist, low, high
    b, a = signal.butter(order, [low, high], btype='band')
    # apply filter
    return signal.filtfilt(b, a, data)

def detectCalls(wp,listnodes,sampleRate):
    import string
    # Add relevant nodes to the wavelet packet tree and then reconstruct the data
    detected = np.zeros((300,len(listnodes)))
    count = 0
    for index in listnodes:
        new_wp = pywt.WaveletPacket(data=None, wavelet='dmey', mode='symmetric')
        # First, turn the index into a leaf name.
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
        j = 0
        for i in range(0,N-sampleRate,sampleRate):
            detected[j,count] = np.max(E[i:i+sampleRate])
            j+=1
        count += 1

    return np.max(detected,axis=1)

def detectCalls1(wp,listnodes,sampleRate):
    # This way should be the best -- reconstruct from setting all of the relevant nodes. But it gives an error message
    # about different size coefficient arrays most of the time.
    import string
    # Add relevant nodes to the wavelet packet tree and then reconstruct the data
    new_wp = pywt.WaveletPacket(data=None, wavelet='dmey', mode='symmetric')


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

def findCalls_train():
    # Nirosha's ugly version

    # Load data and annotation
    data, sampleRate_o, annotation = loadData(True)
    # Resample the data
    # TOOD: Species specific
    # bittern
    #sampleRate = 1000
    # rest
    sampleRate = 16000
    sdata = librosa.core.audio.resample(data,sampleRate_o,sampleRate)

    # Get the five level wavelet decomposition
    wData = denoise(sdata, thresholdType='soft', maxlevel=5)

    # Bandpass filter
    #fwData = bandpass(wData,sampleRate)
    # TODO: Params in here!
    # bittern
    #fwData = ButterworthBandpass(wData,sampleRate,low=100,high=400)
    # kiwi
    fwData = ButterworthBandpass(wData,sampleRate,low=1000,high=7000)
    print fwData

    #fwData = data
    waveletCoefs = computeWaveletEnergy(fwData, sampleRate)

    # Compute point-biserial correlations and sort wrt it, return top nNodes
    nodes = compute_r(annotation,waveletCoefs)
    print nodes

    # Now for Nirosha's weird sorting
    # Basically, for each node, put any of its children (and their children, iteratively) that are in the list in front of it
    nodes = sortListByChild(np.ndarray.tolist(nodes))

    # These nodes refer to the unrooted tree, so add 1 to get the real indices
    nodes = [n + 1 for n in nodes]
    print nodes

    # Generate a full 5 level wavelet packet decomposition
    wpFull = pywt.WaveletPacket(data=fwData, wavelet='dmey', mode='symmetric', maxlevel=5)

    # Now check the F2 values and add node if it improves F2
    #nodes = nodes[]
    listnodes = []
    bestBetaScore = 0
    for c in nodes:
        testlist = listnodes[:]
        testlist.append(c)
        print testlist
        detected = detectCalls(wpFull,testlist,sampleRate)
        fB = fBetaScore(annotation, detected)
        if fB > bestBetaScore:
            bestBetaScore = fB
            listnodes.append(c)
    return listnodes

def findCalls_test(listnodes):

    data, sampleRate_o, annotation = loadData(train=False)
    sampleRate = 16000
    sdata = librosa.core.audio.resample(data,sampleRate_o,sampleRate)
    wData = denoise(sdata, thresholdType='soft', maxlevel=5)
    fwData = ButterworthBandpass(wData,sampleRate,low=1000,high=7000)
    wpFull = pywt.WaveletPacket(data=fwData, wavelet='dmey', mode='symmetric', maxlevel=5)

    detected = detectCalls(wpFull, listnodes, sampleRate)
    print fBetaScore(annotation,detected)
    #print annotation
    #print detected


#D = np.array([0.4364,-0.5044,  0.1021,  1.1963,   0.1203,  -1.0368,  -0.8571,  -0.1699,  -0.1917, -0.8658,  0.1807,   1.2665,  -0.2512,  -0.2046, -2.2015, -0.7745, -1.3933, -0.3862, 0.5256,  1.5233,  1.7985, -0.1169,  -0.3202,   0.8175,  0.4902,  0.7653,  0.7783,  -1.4803, 0.5404, -0.0915])
