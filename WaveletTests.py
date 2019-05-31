import pywt
import wavio
import numpy as np
import string
import os, json
#import SignalProc
import WaveletSegment   # the current version
import matplotlib.markers as mks
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from PyQt5.QtWidgets import QMessageBox
import time
import math

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


def y(t):
    sines = math.sin(30*math.pi*t) + math.sin(60*math.pi*t) + math.sin(90*math.pi*t) + math.sin(120*math.pi*t) + math.sin(180*math.pi*t)
    rest = 2*math.exp(-30*t) * math.sin(260*math.pi*t)
    if (t>0 and t<0.125) or (t>0.3725 and t<0.5) or (t>0.7475 and t<5.1175):
        return sines
    else:
        return sines+rest

def sampley(fs=400, nump=2048):
    out = np.zeros(nump)
    for p in range(nump):
        out[p] = y(p/400)
    return out

import matplotlib
import matplotlib.pyplot as plt
def ploty(y, name):
    xs = np.arange(0, y.size)/400

    ws = np.fft.fft(y)
    fbins = np.fft.fftfreq(y.size, d=1/400)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(xs,y)
    ws[np.where(ws<0)] = 0
    ax[1].plot(fbins, ws)
    ax[0].set(xlabel='time, s', ylabel='signal', title=name)
    ax[0].set(xlabel='freq, Hz', ylabel='signal')

import pywt
def gety():
    ys = sampley()
    ploty(ys, "original")

    wp = pywt.WaveletPacket(data=ys, wavelet='db4')

    new_wp = pywt.WaveletPacket(data=None, wavelet='db4')
    new_wp['a'] = wp['a'].data
    recy = new_wp.reconstruct()
    afilt = np.fft.fft(recy)
    abins = np.fft.fftfreq(len(wp['a'].data), d=1/400)
    todrop = np.where(abins < len(wp['a'].data)/4) or np.where(abins > 3*len(wp['a'].data)/4)
    afilt[todrop] = 0
    recy = np.fft.ifft(afilt)
    ploty(recy, "a")

    new_wp = pywt.WaveletPacket(data=None, wavelet='db4')
    #new_wp['a'] = np.zeros(len(wp['a'].data))
    new_wp['d'] = wp['d'].data
    recy = new_wp.reconstruct()
    afilt = np.fft.fft(recy)
    abins = np.fft.fftfreq(len(wp['d'].data), d=1/400)
    todrop = np.where(abins < len(wp['d'].data)/4) or np.where(abins > 3*len(wp['d'].data)/4)
    afilt[todrop] = 0
    recy = np.fft.ifft(afilt)
    ploty(recy, "d")

    plt.show()


############ Test wavelet filters
def testTrainers(dName, species, f1, f2, fs, thrList, MList, d, f, rf, feature='recaa', window=1, inc=None):
    # Hard-code extra information (they are not used but to be compatible with main program)
    minLen = 6
    maxLen = 32
    minFrq = f1
    maxFrq = f2

    wind = False
    rain = False

    ff = False
    f0_low = 0
    f0_high = 0

    opstartingtime = time.time()
    speciesData = {'Name': species, 'SampleRate': fs, 'TimeRange': [minLen, maxLen],
                   'FreqRange': [minFrq, maxFrq]}  # last params are thr, M
    # returns 2d lists of nodes over M x thr, or stats over M x thr

    print("Thr: ", thrList)
    print("M: ", MList)
    #Virginia: added window and increment
    ws = WaveletSegment.WaveletSegment(speciesData)

    nodes, TP, FP, TN, FN = ws.waveletSegment_train(dName, thrList, MList, d=d, f=f, rf=rf, learnMode=feature,window=window, inc=inc)

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
    markers = np.random.choice(valid_markers, len(MList), replace=False)
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
    print('check1')
    def onclick(event):
        if event.dblclick:
            print('check2')
            fpr_cl = event.xdata
            tpr_cl = event.ydata
            print("fpr_cl, tpr_cl: ", fpr_cl, tpr_cl)
            # TODO: Interpolate?, currently get the closest point
            # get M and thr for closest point
            distarr = (tpr_cl - TPR) ** 2 + (fpr_cl - FPR) ** 2
            print(distarr)
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
            speciesData['WaveletParams'] = []
            speciesData['WaveletParams'].clear()
            speciesData['WaveletParams'].append(thr)
            speciesData['WaveletParams'].append(M)
            speciesData['WaveletParams'].append(optimumNodesSel)
            # Save it
            #dName_train = '/home/listanvirg/FilterTest/Ruru/Test1'
            #dName_train = 'D:\Desktop\Documents\Work\Filter Experiment\RURU\Part1\Test10C'
            #dName_train = '/home/listanvirg/FilterTest/Ruru/Test10D'
            dName_train = '/home/listanvirg/FilterTest/Kiwi/New/Code'
            filename = dName_train + '/' + species + '.txt'
            print(filename)
            print("Saving new filter to ", filename)
            f = open(filename, 'w')
            f.write(json.dumps(speciesData))
            f.close()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


# Test the trained filter on test data
def testWavelet(dName, species, savedetections, feature, d, f, rf, window=1, inc=None):
    print(dName)
    print(os.path.join(dName, species + '.txt'))
    #dName_train = '/home/listanvirg/FilterTest/Ruru/Test1'
    #dName_train = 'D:\Desktop\Documents\Work\Filter Experiment\RURU\Part1\Test10C'
    #dName_train = '/home/listanvirg/FilterTest/Ruru/Test1F'
    dName_train='/home/listanvirg/FilterTest/Kiwi/New/Code'
    speciesData = json.load(open(os.path.join(dName_train, species + '.txt')))
    # speciesData = json.load(open(os.path.join(dName, species + '.txt')))
    #speciesData = json.load(open(os.path.join(dName, species + '.txt')))
    # speciesData = json.load(open("D:\WaveletDetection\DATASETS\Morepork\Test-5min\Morepork.txt"))
    opstartingtime = time.time()
    ws = WaveletSegment.WaveletSegment(speciesData)
    #Virginia: added window and incremnt

    #window= 1
    #inc=None
    Segments, TP, FP, TN, FN = ws.waveletSegment_test(dirName=dName, d=d, f=f, rf=rf, withzeros=True, learnMode=feature, savedetections=savedetections, window=window, inc=inc)


    print("TESTING COMPLETED IN ", time.time() - opstartingtime)
    print('--Test summary--\n%d %d %d %d' %(TP, FP, TN, FN))
    if TP+FN != 0:
        recall = TP/(TP+FN)
    else:
        recall = 0
    if TP+FP != 0:
        precision = TP/(TP+FP)
    else:
        precision = 0
    if TN+FP != 0:
        specificity = TN/(TN+FP)
    else:
        specificity = 0
    if TP+FP+TN+FN != 0:
        accuracy = (TP+TN)/(TP+FP+TN+FN)
    print(' Detection summary:TPR:%.2f%% -- FPR:%.2f%%\n\t\t  Recall:%.2f%%\n\t\t  Precision:%.2f%%\n\t\t  Specificity:%.2f%%\n\t\t  Accuracy:%.2f%%' % (recall*100, 100-specificity*100, recall*100, precision*100, specificity*100, accuracy*100))


#BAT (LONG TAILED) PARAMETERS
bat_LT_M = [0.01, 0.02, 0.06, 0.12]    # syllable length 4ms=0.04 sec; syl_length/8, syl_length/2, syl_length*2 etc.
# kiwi_M = [0.25, 0.75, 1.25, 1.75]
bat_LT_thr = [0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1, 1.25]   #[0.25, 0.5, 0.75, 1, 1.25]
bat_LT_fs = 176000    # fs
bat_LT_f1 = 800   # f low
bat_LT_f2 = 88000  # f high
bat_LT_d = False  # denoise
bat_LT_f = True   # filter
bat_LT_rf = True  # filter the reconstructed signal

#BAT (LONG TAILED) PARAMETERS
bat_LT_M = [0.00375, 0.015, 0.06, 0.12]    # syllable length 2ms=0.02 sec; syl_length/8, syl_length/2, syl_length*2 etc.
# kiwi_M = [0.25, 0.75, 1.25, 1.75]
bat_LT_thr = [0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1, 1.25]   #[0.25, 0.5, 0.75, 1, 1.25]
bat_LT_fs = 176000    # fs
bat_LT_f1 = 800   # f low
bat_LT_f2 = 88000  # f high
bat_LT_d = False  # denoise
bat_LT_f = True   # filter
bat_LT_rf = True  # filter the reconstructed signal


#KIWI PARAMETERS
kiwi_M = [0.125, 0.5, 2.0, 8.0]    # syllable length =1 sec; syl_length/8, syl_length/2, syl_length*2 etc.
# kiwi_M = [0.25, 0.75, 1.25, 1.75]
kiwi_thr = [0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1, 1.25]   #[0.25, 0.5, 0.75, 1, 1.25]
kiwi_fs = 16000     # fs
kiwi_f1 = 800   # f low
kiwi_f2 = 8000  # f high
kiwi_d = False  # denoise
kiwi_f = True   # filter
kiwi_rf = True  # filter the reconstructed signal

# morepork_M = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
morepork_M = [0.0625, 0.25, 1.0, 4]     # syllable length =0.5 sec; syl_length/8, syl_length/2, syl_length*2 etc.
morepork_thr = [0.1, 0.2, 0.4, 0.8, 1.6]
morepork_fs = 16000
morepork_f1 = 600
morepork_f2 = 7000
morepork_d = False
morepork_f = True
morepork_rf = True

robin_M = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]
robin_thr = [0.1, 0.2, 0.4, 0.8, 1.6]
robin_fs = 16000
robin_f1 = 1500
robin_f2 = 8000
robin_d = False
robin_f = True
robin_rf = True

# kakapoB_M = [0.125, 0.25, 1, 1.5, 2]
kakapoB_M = [0.0625, 0.25, 1.0, 4]  # syllable length ~ 0.5 sec
kakapoB_thr = [0.1, 0.2, 0.4, 0.8, 1.6]
kakapoB_fs = 4000
kakapoB_f1 = 50
kakapoB_f2 = 500
kakapoB_d = False
kakapoB_f = False
kakapoB_rf = False

kakapoC_M = [0.125, 0.25, 1, 1.5, 2]
kakapoC_thr = [0.1, 0.2, 0.4, 0.8, 1.6]
kakapoC_fs = 16000
kakapoC_f1 = 1100
kakapoC_f2 = 8000
kakapoC_d = False
kakapoC_f = True
kakapoC_rf = True

bittern_M = [0.125, 0.25, 1, 1.5, 2]
bittern_thr = [0.1, 0.2, 0.4, 0.8, 1.6]
bittern_fs = 4000
bittern_f1 = 100
bittern_f2 = 200
bittern_d = False
bittern_f = False
bittern_rf = False

# testTrainers('D:\WaveletDetection\DATASETS\\NIbrownkiwi\Train_5min', "Kiwi", f1=kiwi_f1, f2=kiwi_f2, fs=kiwi_fs, thrList=kiwi_thr, MList=kiwi_M, d=kiwi_d, f=kiwi_f, rf=kiwi_rf, feature="recaa")
# testWavelet('D:\\Nirosha\WaveletDetection\DATASETS\\NIbrownkiwi\Test_5min', "Kiwi", savedetections=True, feature='recaa', d=kiwi_d, f=kiwi_f, rf=kiwi_rf, window=1, inc=None)


# testTrainers('D:\WaveletDetection\DATASETS\Kakapo\KakapoB\\train-5min', "KakapoB", f1=kakapoB_f1, f2=kakapoB_f2, fs=kakapoB_fs, thrList=kakapoB_thr, MList=kakapoB_M, d=kakapoB_d, f=kakapoB_f, rf=kakapoB_rf, feature="recaa")
# testWavelet('D:\WaveletDetection\DATASETS\Kakapo\KakapoB\\test-5min\\New folder', "KakapoB", savedetections=True, feature='recaa', d=kakapoB_d, f=kakapoB_f, rf=kakapoB_rf, window=1, inc=None)

# testTrainers('D:\WaveletDetection\DATASETS\RMBL\\train', "Robin", f1=robin_f1, f2=robin_f2, fs=robin_fs, thrList=robin_thr, MList=robin_M, d=robin_d, f=robin_f, rf=robin_rf, feature="recaafull")
# testWavelet('D:\WaveletDetection\DATASETS\RMBL-Robin all\\robin\\test', "Robin", savedetections=True, feature='recaafull', d=robin_d, f=robin_f, rf=robin_rf, window=1, inc=None)
# testWavelet('D:\WaveletDetection\DATASETS\RMBL\\aa\\recaa', "Robin", savedetections=True, feature='recaa', d=robin_d, f=robin_f, rf=robin_rf, window=1, inc=None)

# testTrainers('D:\\Nirosha\WaveletDetection\DATASETS\Bittern\\train', "Bittern", f1=bittern_f1, f2=bittern_f2, fs=bittern_fs, thrList=bittern_thr, MList=bittern_M, d=bittern_d, f=bittern_f, rf=bittern_rf, feature="recaa")
# testWavelet('D:\\Nirosha\WaveletDetection\\bittern\\train_national\Kessel', "Bittern", savedetections=True, feature='recaa', d=bittern_d, f=bittern_f, rf=bittern_rf, window=1, inc=None)

# testTrainers('D:\\Nirosha\WaveletDetection\DATASETS\Morepork\Train-5min', "Morepork", f1=morepork_f1, f2=morepork_f2, fs=morepork_fs, thrList=morepork_thr, MList=morepork_M, d=morepork_d, f=morepork_f, rf=morepork_rf, feature="recaa")
# testWavelet('D:\WaveletDetection\DATASETS\Morepork\Test-5min', "Morepork", savedetections=True, feature='recaa', d=morepork_d, f=morepork_f, rf=morepork_rf, window=1, inc=None)

#VIRGINIA VERSION
window=1
inc= None
# MOREPORK
#testTrainers('/home/listanvirg/Data/Filter experiment/BKiwi/Ponui/Ponui-train', "Kiwi", f1=morepork_f1, f2=morepork_f2, fs=morepork_fs, thrList=morepork_thr, MList=morepork_M, d=morepork_d, f=morepork_f, rf=morepork_rf, feature="recaa")
#testTrainers('D:\Desktop\Documents\Work\Data\Filter experiment\Ruru\Train-5min', "Morepork", f1=morepork_f1, f2=morepork_f2, fs=morepork_fs, thrList=morepork_thr, MList=morepork_M, d=morepork_d, f=morepork_f, rf=morepork_rf, feature="recaa",window=window, inc=inc)
#testWavelet('/home/listanvirg/Data/Filter experiment/Ruru/Test-5min', "Morepork", savedetections=True, feature='recaa', d=morepork_d, f=morepork_f, rf=morepork_rf, window=1, inc=None)
#testWavelet('D:\Desktop\Documents\Work\Data\Filter experiment\Ruru\Test-5min', "Morepork", savedetections=True, feature='recaa', d=morepork_d, f=morepork_f, rf=morepork_rf, window=window, inc=inc)

#KiWI
# testTrainers('D:\\Nirosha\WaveletDetection\DATASETS\\NIbrownkiwi\Train_5min', "Kiwi", f1=kiwi_f1, f2=kiwi_f2, fs=kiwi_fs, thrList=kiwi_thr, MList=kiwi_M, d=kiwi_d, f=kiwi_f, rf=kiwi_rf, feature="recaa")
# testWavelet('D:\\Nirosha\WaveletDetection\DATASETS\\NIbrownkiwi\Test_5min', "Kiwi", savedetections=True, feature='recaa', d=kiwi_d, f=kiwi_f, rf=kiwi_rf, window=window, inc=inc)
testTrainers('/home/listanvirg/Data/Filter experiment/BKiwi/Ponui/Ponui_train', "Kiwi", f1=kiwi_f1, f2=kiwi_f2, fs=kiwi_fs, thrList=kiwi_thr, MList=kiwi_M, d=kiwi_d, f=kiwi_f, rf=kiwi_rf, feature="recaa", window=window, inc=inc)
testWavelet('/home/listanvirg/Data/Filter experiment/BKiwi/Ponui/Ponui_test', "Kiwi", savedetections=True, feature='recaa', d=kiwi_d, f=kiwi_f, rf=kiwi_rf, window=window, inc=inc)


#Virginia: keeped as a track
#testTrainers('D:\Desktop\Documents\Work\Data\Filter experiment\Ruru\Train-5min', "Morepork", f1=1000, f2=8000, fs=16000, cleanNodelist=True, feature="recaa")
#testTrainers('/home/listanvirg/Data/Filter experiment/BKiwi/Train', "Kiwi (Nth Is Brown)", f1=1000, f2=8000, fs=16000, cleanNodelist=True, feature="recaa")
#testWavelet('D:\Desktop\Documents\Work\Data\Filter experiment\Ruru\Test-5min', "Morepork", withzeros=True, savedetections=False)
#testWavelet('/home/listanvirg/Data/Filter experiment/BKiwi/Test', "Kiwi (Nth Is Brown)", withzeros=True, savedetections=True)




# testWavelet('D:\WaveletDetection\Figure4\mp', "Morepork", savedetections=True, feature='recaa', d=morepork_d, f=morepork_f, rf=morepork_rf, window=1, inc=None)

# testWavelet('D:\AviaNZ\Sound Files\Fiordland kiwi\Dataset\\Negative\\New folder', "Kiwi(Tokoeka Fiordland)", savedetections=True, feature='recaa', d=kiwi_d, f=kiwi_f, rf=kiwi_rf, window=1, inc=None)
