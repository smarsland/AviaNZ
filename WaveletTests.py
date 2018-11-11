import pywt
import wavio
import numpy as np
import string
import os, json
#import SignalProc
import WaveletSegment2  # the original version
import WaveletSegment   # the current version
import matplotlib.markers as mks
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from PyQt5.QtWidgets import QMessageBox
import time

def showEnergies():
    import pylab as pl
    pl.ion()

    #filename = 'Sound Files/tril1_d1.wav'
    filename = 'Sound Files/tril1.wav'
    #filename = 'Sound Files/090811_184501.wav'
    #filename = 'Sound Files/kiwi_1min.wav'
    wavobj = wavio.read(filename)
    sampleRate = wavobj.rate
    data = wavobj.data
    if data.dtype is not 'float':
        data = data.astype('float')  # / 32768.0
    if np.shape(np.shape(data))[0] > 1:
        data = np.squeeze(data[:, 0])

    if os.path.isfile(filename + '.data'):
        file = open(filename + '.data', 'r')
        segments = json.load(file)
        file.close()
        if len(segments) > 0:
            if segments[0][0] == -1:
                del segments[0]

    data1 = data[int(segments[0][0]*sampleRate):int(segments[0][1]*sampleRate)]
    data2 = data[int(segments[1][0]*sampleRate):int(segments[1][1]*sampleRate)]
    data3 = data[int(segments[2][0]*sampleRate):int(segments[2][1]*sampleRate)]
    data4 = data[int(segments[3][0]*sampleRate):int(segments[3][1]*sampleRate)]
    data5 = data[int(segments[4][0]*sampleRate):int(segments[4][1]*sampleRate)]

    import SignalProc
    sp = SignalProc.SignalProc(data5, sampleRate)
    pl.figure()
    pl.subplot(5, 1, 1)
    sg = sp.spectrogram(data1,sampleRate)
    pl.imshow(10.*np.log10(sg))
    pl.subplot(5, 1, 2)
    sg = sp.spectrogram(data2,sampleRate)
    pl.imshow(10.*np.log10(sg))
    pl.subplot(5, 1, 3)
    sg = sp.spectrogram(data3,sampleRate)
    pl.imshow(10.*np.log10(sg))
    pl.subplot(5, 1, 4)
    sg = sp.spectrogram(data4,sampleRate)
    pl.imshow(10.*np.log10(sg))
    pl.subplot(5, 1, 5)
    sg = sp.spectrogram(data5,sampleRate)
    pl.imshow(10.*np.log10(sg))

    pl.figure()

    e1 = WaveletSegment.computeWaveletEnergy_1s(data1,'dmey2')
    pl.subplot(5,1,1)
    pl.plot(e1)
    e2 = WaveletSegment.computeWaveletEnergy_1s(data2,'dmey2')
    pl.subplot(5,1,2)
    pl.plot(e2)
    e3 = WaveletSegment.computeWaveletEnergy_1s(data3,'dmey2')
    pl.subplot(5,1,3)
    pl.plot(e3)
    e4 = WaveletSegment.computeWaveletEnergy_1s(data4,'dmey2')
    pl.subplot(5,1,4)
    pl.plot(e4)
    e5 = WaveletSegment.computeWaveletEnergy_1s(data5,'dmey2')
    pl.subplot(5,1,5)
    pl.plot(e5)

    pl.figure()
    pl.plot(e1)
    pl.plot(e2)
    pl.plot(e3)
    pl.plot(e4)
    pl.plot(e5)

    #return e2
    pl.show()


def showNoiseEnergies():
    import pylab as pl
    import SignalProc
    #sp = SignalProc.SignalProc(data5, sampleRate)
    pl.ion()
    tbd = [0, 1, 3, 7, 15, 31]
    #tbd = np.concatenate([np.arange(30),np.arange(50,63)])
    #tbd = np.arange(50)
    listnodes = np.arange(63)
    listnodes = np.delete(listnodes, tbd)

    for root, dirs, files in os.walk(str('Sound Files/Noise examples/Noise_10s')):
        for filename in files:
            if filename.endswith('.wav'):
                filename = root + '/' + filename
                wavobj = wavio.read(filename)
                sampleRate = wavobj.rate
                data = wavobj.data
                if data.dtype is not 'float':
                    data = data.astype('float')  # / 32768.0
                if np.shape(np.shape(data))[0] > 1:
                    data = np.squeeze(data[:, 0])

                pl.figure()
                e1 = WaveletSegment.computeWaveletEnergy_1s(data,'dmey2')
                pl.plot(e1[listnodes])
                pl.title(filename)

    #pl.show()

def convert(i):
    level = int(np.floor(np.log2(i + 1)))
    first = 2 ** level - 1
    if i == 0:
        b = ''
    else:
        b = np.binary_repr(i - first, width=int(level))
        b = string.replace(b, '0', 'a', maxreplace=-1)
        b = string.replace(b, '1', 'd', maxreplace=-1)
    return b

def reconWPT():
    import pylab as pl
    pl.ion()

    filename = 'Sound Files/tril1.wav'
    #filename = 'Sound Files/090811_184501.wav'
    #filename = 'Sound Files/kiwi_1min.wav'
    wavobj = wavio.read(filename)
    sampleRate = wavobj.rate
    data = wavobj.data
    if data.dtype is not 'float':
        data = data.astype('float')  # / 32768.0
    if np.shape(np.shape(data))[0] > 1:
        data = np.squeeze(data[:, 0])

    tbd = [0, 1, 3, 7, 15, 31]
    #tbd = np.concatenate([np.arange(30),np.arange(50,63)])
    #tbd = np.arange(50)
    listnodes = np.arange(63)
    listnodes = np.delete(listnodes, tbd)

    [lowd, highd, lowr, highr] = np.loadtxt('dmey.txt')
    wavelet = pywt.Wavelet(filter_bank=[lowd, highd, lowr, highr])
    wavelet.orthogonal = True
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=5)
    # Make a new tree with these in
    new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='zero', maxlevel=5)

    # There seems to be a bit of a bug to do with the size of the reconstucted nodes, so prime them
    # It's worse than that. pywavelet makes the whole tree. So if you don't give it blanks, it copies the details from wp even though it wasn't asked for. And reconstruction with the zeros is different to not reconstructing.
    for level in range(6):
        for n in new_wp.get_level(level, 'natural'):
            n.data = np.zeros(len(wp.get_level(level, 'natural')[0].data))

    # Copy thresholded versions of the leaves into the new wpt
    for l in listnodes:
        ind = convert(l)
        new_wp[ind].data = wp[ind].data

    newdata = new_wp.reconstruct(update=False)
    import SignalProc
    sp = SignalProc.SignalProc(newdata, sampleRate)
    pl.figure()
    pl.subplot(3,1,1)
    sg = sp.spectrogram(data,sampleRate)
    pl.imshow(10.*np.log10(sg).T)
    pl.subplot(3,1,2)
    sg = sp.spectrogram(newdata,sampleRate)
    pl.imshow(10.*np.log10(sg).T)

    #wavio.write('tril1_d1.wav', data.astype('int16'), sampleRate, scale='dtype-limits', sampwidth=2)

    wavobj = wavio.read('Sound Files/tril1_d.wav')
    data = wavobj.data
    if data.dtype is not 'float':
        data = data.astype('float')  # / 32768.0
    if np.shape(np.shape(data))[0] > 1:
        data = np.squeeze(data[:, 0])
    pl.subplot(3,1,3)
    sg = sp.spectrogram(data,sampleRate)
    pl.imshow(10.*np.log10(sg).T)

# Test previous code with and without zeroing the tree, save the filters and compare
def testTrainers2(dName, withzeros):
    # Hard code meta data
    species = "Kiwi (Little Spotted)"
    fs = 16000
    minLen = 6
    maxLen = 32
    minFrq = 1200
    maxFrq = 8000
    wind = True
    rain = True
    ff = False
    f0_low = 0
    f0_high = 0

    # Define the depth of grid
    M_range = np.linspace(0.25, 1.5, num=3)
    thr_range = np.linspace(0, 1, num=5)

    optimumNodes_M = []
    TPR_M = []
    FPR_M = []
    iterNum = 1
    ws = WaveletSegment2.WaveletSegment()   # refers old version

    opstartingtime = time.time()
    for M in M_range:
        print('------ Now M:', M)
        # Find wavelet nodes for different thresholds
        optimumNodes = []
        TPR = []
        FPR = []
        for thr in thr_range:
            speciesData = {'Name': species, 'SampleRate': fs, 'TimeRange': [minLen, maxLen],
                           'FreqRange': [minFrq, maxFrq], 'WaveletParams': [thr, M]}
            optimumNodes_thr = []
            TP = FP = TN = FN = 0
            for root, dirs, files in os.walk(str(dName)):
                for file in files:
                    if file.endswith('.wav') and os.stat(root + '/' + file).st_size != 0 and file[:-4] + '-sec.txt' in files:
                        wavFile = root + '/' + file[:-4]
                        nodes, stats = ws.waveletSegment_train(wavFile, spInfo=speciesData, df=False, withzeros=withzeros)
                        TP += stats[0]
                        FP += stats[1]
                        TN += stats[2]
                        FN += stats[3]
                        print('Current:', wavFile)
                        print("Parameters: M, Thr ", M, thr)
                        print("Filtered nodes for current file: ", nodes)
                        print("Iteration %d/%d" % (iterNum, len(M_range) * len(thr_range)))
                        for node in nodes:
                            if node not in optimumNodes_thr:
                                optimumNodes_thr.append(node)
            TPR_thr = TP / (TP + FN)
            FPR_thr = 1 - TN / (FP + TN)
            TPR.append(TPR_thr)
            FPR.append(FPR_thr)
            optimumNodes.append(optimumNodes_thr)
            iterNum += 1
        TPR_M.append(TPR)
        FPR_M.append(FPR)
        optimumNodes_M.append(optimumNodes)
    print("TRAINING COMPLETED IN ", time.time() - opstartingtime)
    print(TPR_M)
    print(FPR_M)
    print(optimumNodes_M)

    # Plot AUC and let the user to choose threshold and M
    plt.style.use('ggplot')
    valid_markers = ([item[0] for item in mks.MarkerStyle.markers.items() if
                      item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith(
                          'caret')])
    markers = np.random.choice(valid_markers, len(M_range) * len(thr_range), replace=False)
    fig, ax = plt.subplots()
    for i in range(len(M_range)):
        ax.plot(FPR_M[i], TPR_M[i], marker=markers[i], label='M=' + str(M_range[i]))
    ax.set_title('Double click and set Tolerance')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    fig.canvas.set_window_title('ROC Curve - %s' % (species))
    ax.set_ybound(0, 1)
    ax.set_xbound(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax.legend()
    def onclick(event):
        if event.dblclick:
            fpr_cl = event.xdata
            tpr_cl = event.ydata
            print("fpr_cl, tpr_cl: ", fpr_cl, tpr_cl)
            # TODO: Interpolate?, currently get the closest point
            TPRmin_M = []
            for i in range(len(M_range)):
                TPRmin = [np.abs(x - tpr_cl) for x in TPR_M[i]]
                TPRmin_M.append(TPRmin)
            # Choose M
            M_min = [np.min(x) for x in TPRmin_M]
            ind = np.argmin(M_min)
            M = M_range[ind]
            # Choose threshold
            ind_thr = np.argmin(TPRmin_M[ind])
            thr = thr_range[ind_thr]
            optimumNodesSel = optimumNodes_M[ind][ind_thr]
            plt.close()
            speciesData['Wind'] = wind
            speciesData['Rain'] = rain
            speciesData['F0'] = ff
            if ff:
                speciesData['F0Range'] = [f0_low, f0_high]
            speciesData['WaveletParams'].clear()
            speciesData['WaveletParams'].append(thr)
            speciesData['WaveletParams'].append(M)
            speciesData['WaveletParams'].append(optimumNodesSel)
            # Save it
            filename = dName + '\\' + species + '.txt'
            print("Saving new filter to ", filename)
            f = open(filename, 'w')
            f.write(json.dumps(speciesData))
            f.close()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

# testTrainers2('E:\AviaNZ\Sound Files\LSK\\train', withzeros=True)

# Test the new version of training
def testTrainers(dName, trainPerFile=True, withzeros=False):
    # Hard code meta data
    species = "Kiwi (Little Spotted)"
    fs = 16000
    minLen = 6
    maxLen = 32
    minFrq = 1200
    maxFrq = 8000
    wind = True
    rain = True
    ff = False
    f0_low = 0
    f0_high = 0

    opstartingtime = time.time()
    speciesData = {'Name': species, 'SampleRate': fs, 'TimeRange': [minLen, maxLen],
                   'FreqRange': [minFrq, maxFrq]}  # last params are thr, M
    # returns 2d lists of nodes over M x thr, or stats over M x thr
    thrList = np.linspace(0, 1, 5)
    MList = np.linspace(0.25, 1.5, 3)
    ws = WaveletSegment.WaveletSegment()
    nodes, TP, FP, TN, FN = ws.waveletSegment_train(dName, thrList, MList, spInfo=speciesData, df=False, trainPerFile=trainPerFile, withzeros=withzeros)
    print("Filtered nodes: ", nodes)

    TPR = TP / (TP + FN)
    FPR = 1 - TN / (FP + TN)
    print("TP rate: ", TPR)
    print("FP rate: ", FPR)
    print("TRAINING COMPLETED IN ", time.time() - opstartingtime)
    # Plot AUC and let the user to choose threshold and M
    plt.style.use('ggplot')
    valid_markers = ([item[0] for item in mks.MarkerStyle.markers.items() if
                      item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith(
                          'caret')])
    markers = np.random.choice(valid_markers, len(MList) * len(thrList), replace=False)
    fig, ax = plt.subplots()
    for i in range(len(MList)):
        # each line - different M (rows of result arrays)
        ax.plot(FPR[i], TPR[i], marker=markers[i], label='M=' + str(MList[i]))
    ax.set_title('Double click to choose TPR and FPR and set tolerance')
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    fig.canvas.set_window_title('ROC Curve - %s' % (species))
    ax.set_ybound(0, 1)
    ax.set_xbound(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax.legend()
    def onclick(event):
        if event.dblclick:
            fpr_cl = event.xdata
            tpr_cl = event.ydata
            print("fpr_cl, tpr_cl: ", fpr_cl, tpr_cl)
            # TODO: Interpolate?, currently get the closest point
            # get M and thr for closest point
            distarr = (tpr_cl - TPR) ** 2 + (fpr_cl - FPR) ** 2
            M_min_ind, thr_min_ind = np.unravel_index(np.argmin(distarr), distarr.shape)
            M = MList[M_min_ind]
            thr = thrList[thr_min_ind]
            # Get nodes for closest point
            optimumNodesSel = nodes[M_min_ind][thr_min_ind]
            plt.close()
            speciesData['Wind'] = wind
            speciesData['Rain'] = rain
            speciesData['F0'] = ff
            if ff:
                speciesData['F0Range'] = [f0_low, f0_high]
            speciesData['WaveletParams'].clear()
            speciesData['WaveletParams'].append(thr)
            speciesData['WaveletParams'].append(M)
            speciesData['WaveletParams'].append(optimumNodesSel)
            # Save it
            filename = dName + '\\' + species + '.txt'
            print("Saving new filter to ", filename)
            f = open(filename, 'w')
            f.write(json.dumps(speciesData))
            f.close()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

# testTrainers('E:\AviaNZ\Sound Files\LSK\\train', trainPerFile=False, withzeros=True)