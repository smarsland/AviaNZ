
import numpy as np
import pywt
from scipy.io import wavfile
import pylab as pl
import librosa

# Nirosha's approach of simultaneous segmentation and recognition using wavelets

# Nirosha's version:
    # (0) Bandpass filter with hacks
    # (1) 5 level wavelet packet decomposition
    # (2) Sort nodes of (1) into order by point-biserial correlation with training labels
    # (3) Retain top nodes (up to 20)
    # (4) Re-sort to favour child nodes
    # (5) Reduce number using F_2 score
    # (6) Classify as call if OR of (5) is true
# Stephen: Think about (4), fix (0), (5), (6) -> learning!

def denoise(data,thresholdType='soft', maxlevel=5):
    # Perform wavelet denoising. Can use soft or hard thresholding
    # TODO: Reconstruction didn't work too well. Why not?
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

    wData = wp.data # self.wp.reconstruct(update=False)

    return wData

def bandpass(wData,sampleRate):
    # Bandpass filter
    # There are some ugly parameters in here
    # TODO: need to work on this more
    import scipy.signal as signal
    nyquist = sampleRate/2.0
    ntaps = 128
    #taps = signal.firwin(ntaps,cutoff = [1100/nyquist,7500/nyquist], window=('hamming'),pass_zero=False)
    taps = signal.firwin(ntaps,cutoff = [50/nyquist,400/nyquist], window=('hamming'),pass_zero=False)
    fwData = signal.lfilter(taps, 1.0, wData)

    # width = 1.6
    # atten = signal.kaiser_atten(ntaps, width / nyquist)
    # beta = signal.kaiser_beta(atten)
    # taps = signal.firwin(ntaps,cutoff = [1100/nyquist,7500/nyquist], window=('kaiser',beta),pass_zero=False)
    # fwData = signal.lfilter(taps, 1.0, wData)

    return fwData

def ShannonEntropy(s):
    # Compute the Shannon entropy of data
    e = -s[np.nonzero(s)] ** 2 * np.log(s[np.nonzero(s)] ** 2)
    # e = np.where(s==0,0,-s**2*np.log(s**2))
    return np.sum(e)

def load():
    # Load data
    filename = 'Wavelet Segmentation/bittern/train/train1.wav'
    sampleRate, audiodata = wavfile.read(filename)
    if audiodata.dtype is not 'float':
        audiodata = audiodata.astype('float') / 32768.0
    if np.shape(np.shape(audiodata))[0]>1:
        audiodata = audiodata[:,0]

    # Get the segmentation from the excel file
    annotation = np.zeros(300)
    count = 0
    import openpyxl as op
    wb = op.load_workbook(filename = 'Wavelet Segmentation/bittern/train/train1-sec.xlsx')
    ws = wb.active
    for row in ws.iter_rows('B2:B301'):
        annotation[count] = row[0].value
        count += 1

    return audiodata, sampleRate, annotation

def wpd(fwData,sampleRate):
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
    precision = float(T)/P #TruePositive/#Positive
    return ((1.+beta**2)*recall*precision)/(recall + beta**2*precision)

def compute_r(annotation,waveletCoefs,nNodes=20):
    # TODO: Compare with Nirosha's matlab code
    # Find the correlations (point-biserial)
    # r = (M_p - M_q) / S * sqrt(p*q), M_p = mean for those that are 0, S = std dev overall, p = proportion that are 0.
    w0 = np.where(annotation==0)[0]
    w1 = np.where(annotation==1)[0]

    r = np.zeros(62)
    for count in range(62):
        r[count] = (np.mean(waveletCoefs[count,w0]) - np.mean(waveletCoefs[count,w1]))/np.std(waveletCoefs[count,:]) * np.sqrt(len(w0)*len(w1))/len(annotation)

    order = np.argsort(r)
    print r[order]
    print order
    order = order[-1:-nNodes-1:-1]

    return order

def sortListByChild(order):
    # Have a list sorted into order of correlation
    # Want to resort so that any children of the current node that are in the list go first
    # Assumes that there are five levels in the tree

    #order = [3, 9, 43, 45, 21, 20, 40, 19, 8, 44]
    #output = [43, 45, 40, 44, 21, 20, 19, 9, 8, 3]

    #order = [42, 34, 20, 2, 16, 8, 7, 40, 19, 1]
    #output = [42, 34, 40, 20, 16, 7, 2, 19, 8, 1]
    #output = [42, 34, 20, 16, 7, 2, 40, 19, 8, 1]

    newlist = []
    currentIndex = 0
    # Need to keep track of where each level of the tree starts
    starts = [0, 2, 6, 14,30,62]
    while len(order)>0:
        if order[0]<30:
            #print order[0]
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

def ButterworthBandpass(data,sampleRate,order=10,low=1000,high=10000):
    import scipy.signal as signal
    nyquist = sampleRate/2.0

    low = float(low)/nyquist
    high = float(high)/nyquist
    #print nyquist, low, high
    b, a = signal.butter(order, [low, high], btype='band')
    # apply filter
    return signal.filtfilt(b, a, data)

def detectCalls(waveletCoefs, listnodes,thresholds):
    # TODO: thresholds can be on the coefficients or on the reconstructed sound

    # Reconstruct signal from each node separately (or just the list)
    # Choose nodes to keep (listnodes)
    # wp.reconstruct(update=False)
    # Generate the energy curve
    # Threshold

    # Raw wavelets
    predicted = np.zeros(np.shape(waveletCoefs)[1])

    for node in listnodes:
        times = np.squeeze(np.where(waveletCoefs[node,:] > thresholds[node]))
        # OR of the individual node outputs
        if np.shape(times) > 0:
            predicted[times] = 1
    return predicted

def findCalls_train():
    # Nirosha's ugly version
    # Load data and annotation
    data, sampleRate_o, annotation = load()

    # Resample the data
    sampleRate = 1000
    data = librosa.core.audio.resample(data,sampleRate_o,sampleRate)
    wData = denoise(data, thresholdType='soft', maxlevel=5)

    # Bandpass filter
    #fwData = bandpass(wData,sampleRate)
    # TODO: Params in here!
    fwData = ButterworthBandpass(wData,sampleRate,low=100,high=400)

    # Compute wavelet packet decomposition
    waveletCoefs = wpd(fwData, sampleRate)

    # TODO: thresholds can be on the coefficients
    thresholds = np.mean(waveletCoefs,1) + np.std(waveletCoefs,1)
    # TODO: or on the reconstructed sound
    # Generate the reconstructed signal based on each individual node
    # Then threshold is mean + sd of the reconstruction

    # Compute point-biserial correlations and sort wrt it, return top nNodes
    nodes = compute_r(annotation,waveletCoefs)
    print nodes

    # Now for Nirosha's weird sorting
    # Basically, for each node, put any of its children (and their children, iteratively) that are in the list in front of it
    nodes = sortListByChild(np.ndarray.tolist(nodes))
    print nodes

    # Now check the F2 values and add node if it improves F2
    listnodes = []
    bestBetaScore = 0
    for c in nodes:
        print "testing ",c
        testlist = listnodes[:]
        testlist.append(c)
        # Make prediction **** What exactly does this do?
        predicted = detectCalls(waveletCoefs, testlist, thresholds)
        fB = fBetaScore(annotation, predicted)
        if fB > bestBetaScore:
            bestBetaScore = fB
            listnodes.append(c)
            print "adding ", c
    return listnodes

def findCalls_test(listnodes):
    data,  sampleRate, annotation = load()
    waveletCoefs = wpd(fwData, sampleRate)
    predicted = predicted or findCalls(data, listnodes)







def test_filters(sampleRate):
    pl.figure()
    pl.subplot(611),pl.plot(data)
    pl.subplot(612),pl.specgram(audiodata, NFFT=256, sampleRate=sampleRate, noverlap=128,cmap=pl.cm.gray_r)

    wData = denoise(data, thresholdType='soft', maxlevel=5)
    pl.subplot(613),pl.plot(wData)
    pl.subplot(614),pl.specgram(wData, NFFT=256, sampleRate=sampleRate, noverlap=128,cmap=pl.cm.gray_r)

    fwData = bandpass(wData,sampleRate)
    pl.subplot(615),pl.plot(fwData)
    pl.subplot(616),pl.specgram(fwData, NFFT=256, sampleRate=sampleRate, noverlap=128,cmap=pl.cm.gray_r)

    pl.show()

#data, sampleRate, annotation = WaveletSegment.load()
#r, coefsampleRate = WaveletSegment.compute_r(annotation,fwData,sampleRate)
#WaveletSegment.findCalls(coefsampleRate,data)
