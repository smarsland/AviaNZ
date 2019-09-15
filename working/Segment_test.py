
# All the testing of Segment.py goes here

import Segment
import numpy as np

def convertAmpltoSpec(x,fs,inc):
    """ Unit conversion """
    return x*fs/inc

def testMC():
    import wavio
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui

    #wavobj = wavio.read('Sound Files/kiwi_1min.wav')
    wavobj = wavio.read('Sound Files/tril1.wav')
    fs = wavobj.rate
    data = wavobj.data#[:20*fs]

    if data.dtype is not 'float':
        data = data.astype('float')  #/ 32768.0

    if np.shape(np.shape(data))[0] > 1:
        data = data[:, 0]

    import SignalProc
    sp = SignalProc.SignalProc(data,fs,256,128)
    sg = sp.spectrogram(data=data,window_width=256,incr=128,window='Hann',mean_normalise=True,onesided=True,multitaper=False,need_even=False)
    s = Segment.Segmenter(data,sg,sp,fs)

    #print np.shape(sg)

    #s1 = s.medianClip()
    s1,p,t = s.yin(returnSegs=True)
    app = QtGui.QApplication([])

    mw = QtGui.QMainWindow()
    mw.show()
    mw.resize(800, 600)

    win = pg.GraphicsLayoutWidget()
    mw.setCentralWidget(win)
    vb1 = win.addViewBox(enableMouse=False, enableMenu=False, row=0, col=0)
    im1 = pg.ImageItem(enableMouse=False)
    vb1.addItem(im1)
    im1.setImage(10.*np.log10(sg))

    # vb2 = win.addViewBox(enableMouse=False, enableMenu=False, row=1, col=0)
    # im2 = pg.ImageItem(enableMouse=False)
    # vb2.addItem(im2)
    # im2.setImage(c)

    vb3 = win.addViewBox(enableMouse=False, enableMenu=False, row=1, col=0)
    im3 = pg.ImageItem(enableMouse=False)
    vb3.addItem(im3)
    im3.setImage(10.*np.log10(sg))

    vb4 = win.addViewBox(enableMouse=False, enableMenu=False, row=2, col=0)
    im4 = pg.PlotDataItem(enableMouse=False)
    vb4.addItem(im4)
    im4.setData(data)

    for seg in s1:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        #a.setRegion([seg[0],seg[1]])
        vb3.addItem(a, ignoreBounds=True)

    QtGui.QApplication.instance().exec_()


def showSegs():
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    import wavio
    import WaveletSegment
    from time import time

    #wavobj = wavio.read('Sound Files/tril1.wav')
    #wavobj = wavio.read('Sound Files/010816_202935_p1.wav')
    #wavobj = wavio.read('Sound Files/20170515_223004 piping.wav')
    wavobj = wavio.read('Sound Files/kiwi_1min.wav')
    fs = wavobj.rate
    data = wavobj.data#[:20*fs]

    if data.dtype is not 'float':
        data = data.astype('float') # / 32768.0

    if np.shape(np.shape(data))[0] > 1:
        data = data[:, 0]

    import SignalProc
    sp = SignalProc.SignalProc(data,fs,256,128)
    sg = sp.spectrogram(data,multitaper=False)
    s = Segment(data,sg,sp,fs,50)

    # FIR: threshold doesn't matter much, but low is better (0.01).
    # Amplitude: not great, will have to work on width and abs if want to use it (threshold about 0.6)
    # Power: OK, but threshold matters (0.5)
    # Median clipping: OK, threshold of 3 fine.
    # Onsets: Threshold of 4.0 was fine, lower not. Still no offsets!
    # Yin: Threshold 0.9 is pretty good
    # Energy: Not great, but thr 1.0
    ts = time()
    s1=s.checkSegmentLength(s.segmentByFIR(0.1))
    s2=s.checkSegmentLength(s.segmentByFIR(0.01))
    s3= s.checkSegmentLength(s.medianClip(3.0))
    s4= s.checkSegmentLength(s.medianClip(2.0))
    s5,p,t=s.yin(100, thr=0.5,returnSegs=True)
    s5 = s.checkSegmentLength(s5)
    s6=s.mergeSegments(s2,s4)
    ws = WaveletSegment.WaveletSegment()
    s7= ws.waveletSegment_test(None, data, fs, None, 'Kiwi', False)
    #print('Took {}s'.format(time() - ts))
    #s7 = s.mergeSegments(s1,s.mergeSegments(s3,s4))

    #s4, samp = s.segmentByFIR(0.4)
    #s4 = s.checkSegmentLength(s4)
    #s2 = s.segmentByAmplitude1(0.6)
    #s5 = s.checkSegmentLength(s.segmentByPower(0.3))
    #s6, samp = s.segmentByFIR(0.6)
    #s6 = s.checkSegmentLength(s6)
    #s7 = []
    #s5 = s.onsets(3.0)
    #s6 = s.segmentByEnergy(1.0,500)

    #s5 = s.Harma(5.0,0.8)
    #s4 = s.Harma(10.0,0.8)
    #s7 = s.Harma(15.0,0.8)

    #s2 = s.segmentByAmplitude1(0.7)
    #s3 = s.segmentByPower(1.)
    #s4 = s.medianClip(3.0)
    #s5 = s.onsets(3.0)
    #s6, p, t = s.yin(100,thr=0.5,returnSegs=True)
    #s7 = s.Harma(10.0,0.8)

    app = QtGui.QApplication([])

    mw = QtGui.QMainWindow()
    mw.show()
    mw.resize(800, 600)

    win = pg.GraphicsLayoutWidget()
    mw.setCentralWidget(win)
    vb1 = win.addViewBox(enableMouse=False, enableMenu=False, row=0, col=0)
    im1 = pg.ImageItem(enableMouse=False)
    vb1.addItem(im1)
    im1.setImage(10.*np.log10(sg))

    vb2 = win.addViewBox(enableMouse=False, enableMenu=False, row=1, col=0)
    im2 = pg.ImageItem(enableMouse=False)
    vb2.addItem(im2)
    im2.setImage(10.*np.log10(sg))

    vb3 = win.addViewBox(enableMouse=False, enableMenu=False, row=2, col=0)
    im3 = pg.ImageItem(enableMouse=False)
    vb3.addItem(im3)
    im3.setImage(10.*np.log10(sg))

    vb4 = win.addViewBox(enableMouse=False, enableMenu=False, row=3, col=0)
    im4 = pg.ImageItem(enableMouse=False)
    vb4.addItem(im4)
    im4.setImage(10.*np.log10(sg))

    vb5 = win.addViewBox(enableMouse=False, enableMenu=False, row=4, col=0)
    im5 = pg.ImageItem(enableMouse=False)
    vb5.addItem(im5)
    im5.setImage(10.*np.log10(sg))

    vb6 = win.addViewBox(enableMouse=False, enableMenu=False, row=5, col=0)
    im6 = pg.ImageItem(enableMouse=False)
    vb6.addItem(im6)
    im6.setImage(10.*np.log10(sg))

    vb7 = win.addViewBox(enableMouse=False, enableMenu=False, row=6, col=0)
    im7 = pg.ImageItem(enableMouse=False)
    vb7.addItem(im7)
    im7.setImage(10.*np.log10(sg))

    print("====")
    print(s1)
    for seg in s1:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb1.addItem(a, ignoreBounds=True)

    print(s2)
    for seg in s2:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb2.addItem(a, ignoreBounds=True)

    print(s3)
    for seg in s3:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb3.addItem(a, ignoreBounds=True)

    print(s4)
    for seg in s4:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb4.addItem(a, ignoreBounds=True)

    print(s5)
    for seg in s5:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb5.addItem(a, ignoreBounds=True)

    print(s6)
    for seg in s6:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb6.addItem(a, ignoreBounds=True)

    print(s7)
    for seg in s7:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb7.addItem(a, ignoreBounds=True)

    QtGui.QApplication.instance().exec_()

def showSpecDerivs():
    import SignalProc
    reload(SignalProc)
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    import wavio

    #wavobj = wavio.read('Sound Files/tril1.wav')
    #wavobj = wavio.read('Sound Files/010816_202935_p1.wav')
    #wavobj = wavio.read('Sound Files/20170515_223004 piping.wav')
    wavobj = wavio.read('Sound Files/kiwi_1min.wav')
    fs = wavobj.rate
    data = wavobj.data[:20*fs]

    if data.dtype is not 'float':
        data = data.astype('float')     # / 32768.0

    if np.shape(np.shape(data))[0] > 1:
        data = data[:, 0]

    import SignalProc
    sp = SignalProc.SignalProc(data, fs, 256, 128)
    sg = sp.spectrogram(data, multitaper=False)

    h,v,b = sp.spectralDerivatives()
    h = np.abs(np.where(h == 0, 0.0, 10.0 * np.log10(h)))
    v = np.abs(np.where(v == 0, 0.0, 10.0 * np.log10(v)))
    b = np.abs(np.where(b == 0, 0.0, 10.0 * np.log10(b)))
    s = Segment(data, sg, sp, fs, 50)

    hm = np.max(h[:, 10:], axis=1)
    inds = np.squeeze(np.where(hm > (np.mean(h[:,10:]+2.5*np.std(h[:, 10:])))))
    segmentsh = s.identifySegments(inds, minlength=10)

    vm = np.max(v[:, 10:], axis=1)
    inds = np.squeeze(np.where(vm > (np.mean(v[:, 10:]+2.5*np.std(v[:, 10:])))))
    segmentsv = s.identifySegments(inds, minlength=10)

    bm = np.max(b[:, 10:], axis=1)
    segs = np.squeeze(np.where(bm > (np.mean(b[:, 10:]+2.5*np.std(b[:, 10:])))))
    segmentsb = s.identifySegments(segs, minlength=10)
    #print np.mean(h), np.max(h)
    #print np.where(h>np.mean(h)+np.std(h))

    app = QtGui.QApplication([])

    mw = QtGui.QMainWindow()
    mw.show()
    mw.resize(800, 600)

    win = pg.GraphicsLayoutWidget()
    mw.setCentralWidget(win)
    vb1 = win.addViewBox(enableMouse=False, enableMenu=False, row=0, col=0)
    im1 = pg.ImageItem(enableMouse=False)
    vb1.addItem(im1)
    im1.setImage(10.*np.log10(sg))

    vb2 = win.addViewBox(enableMouse=False, enableMenu=False, row=1, col=0)
    im2 = pg.ImageItem(enableMouse=False)
    vb2.addItem(im2)
    im2.setImage(h)
    for seg in segmentsh:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0], fs, 128), convertAmpltoSpec(seg[1], fs, 128)])
        vb2.addItem(a, ignoreBounds=True)

    vb3 = win.addViewBox(enableMouse=False, enableMenu=False, row=2, col=0)
    im3 = pg.ImageItem(enableMouse=False)
    vb3.addItem(im3)
    im3.setImage(v)
    for seg in segmentsv:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0], fs, 128), convertAmpltoSpec(seg[1], fs, 128)])
        vb3.addItem(a, ignoreBounds=True)

    vb4 = win.addViewBox(enableMouse=False, enableMenu=False, row=3, col=0)
    im4 = pg.ImageItem(enableMouse=False)
    vb4.addItem(im4)
    im4.setImage(b)
    for seg in segmentsb:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0], fs, 128), convertAmpltoSpec(seg[1], fs, 128)])
        vb4.addItem(a, ignoreBounds=True)
    QtGui.QApplication.instance().exec_()

def detectClicks():
    import SignalProc
    # reload(SignalProc)
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    import wavio
    from scipy.signal import medfilt

    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\Clicks\\1ex\Lake_Thompson__01052018_SOUTH1047849_01052018_High_20180509_'
    #                     '20180509_183506.wav')  # close kiwi and rain
    wavobj = wavio.read('D:\AviaNZ\Sound_Files\Clicks\Lake_Thompson__01052018_SOUTH1047849_01052018_High_20180508_'
                        '20180508_200506.wav')  # very close kiwi with steady wind
    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\Clicks\\1ex\Murchison_Kelper_Heli_25042018_SOUTH7881_25042018_High_'
    #                     '20180405_20180405_211007.wav')
    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\\Noise examples\\Noise_10s\Rain_010.wav')
    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\Clicks\Ponui_SR2_Jono_20130911_021920.wav')   #
    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\Clicks\CL78_BIRM_141120_212934.wav')   #
    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\Clicks\CL78_BIRD_141120_212934.wav')   # Loud click
    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\Tier1\Tier1 dataset\positive\DE66_BIRD_141011_005829.wav')   # close kiwi
    # wavobj = wavio.read('Sound Files/010816_202935_p1.wav')
    #wavobj = wavio.read('Sound Files/20170515_223004 piping.wav')
    # wavobj = wavio.read('Sound Files/test/DE66_BIRD_141011_005829.wav')
    #wavobj = wavio.read('/Users/srmarsla/DE66_BIRD_141011_005829_wb.wav')
    #wavobj = wavio.read('/Users/srmarsla/ex1.wav')
    #wavobj = wavio.read('/Users/srmarsla/ex2.wav')
    fs = wavobj.rate
    data = wavobj.data #[:20*fs]

    if data.dtype is not 'float':
        data = data.astype('float')     # / 32768.0

    if np.shape(np.shape(data))[0] > 1:
        data = data[:, 0]

    import SignalProc
    sp = SignalProc.SignalProc(data, fs, 128, 128)
    sg = sp.spectrogram(data, multitaper=False)
    s = Segment(data, sg, sp, fs, 128)

    # for each frq band get sections where energy exceeds some (90%) percentile
    # and generate a binary spectrogram
    sgb = np.zeros((np.shape(sg)))
    for y in range(np.shape(sg)[1]):
        ey = sg[:, y]
        # em = medfilt(ey, 15)
        ep = np.percentile(ey, 90)
        sgb[np.where(ey > ep), y] = 1

    # If lots of frq bands got 1 then predict a click
    clicks = []
    for x in range(np.shape(sg)[0]):
        if np.sum(sgb[x, :]) > np.shape(sgb)[1]*0.75:
            clicks.append(x)

    app = QtGui.QApplication([])

    mw = QtGui.QMainWindow()
    mw.show()
    mw.resize(1200, 500)

    win = pg.GraphicsLayoutWidget()
    mw.setCentralWidget(win)
    vb1 = win.addViewBox(enableMouse=False, enableMenu=False, row=0, col=0)
    im1 = pg.ImageItem(enableMouse=False)
    vb1.addItem(im1)
    im1.setImage(sgb)

    if len(clicks) > 0:
        clicks = s.identifySegments(clicks, minlength=1)

    for seg in clicks:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0], fs, 128), convertAmpltoSpec(seg[1], fs, 128)])
        vb1.addItem(a, ignoreBounds=True)

    QtGui.QApplication.instance().exec_()


    # energy = np.sum(sg, axis=1)
    # energy = medfilt(energy, 15)
    # e2 = np.percentile(energy, 50)*2
    # # Step 1: clicks have high energy
    # clicks = np.squeeze(np.where(energy > e2))
    # print(clicks)
    # if len(clicks) > 0:
    #     clicks = s.identifySegments(clicks, minlength=1)
    #
    # app = QtGui.QApplication([])
    #
    # mw = QtGui.QMainWindow()
    # mw.show()
    # mw.resize(800, 600)
    #
    # win = pg.GraphicsLayoutWidget()
    # mw.setCentralWidget(win)
    # vb1 = win.addViewBox(enableMouse=False, enableMenu=False, row=0, col=0)
    # im1 = pg.ImageItem(enableMouse=False)
    # vb1.addItem(im1)
    # im1.setImage(10.*np.log10(sg))
    #
    # for seg in clicks:
    #     a = pg.LinearRegionItem()
    #     a.setRegion([convertAmpltoSpec(seg[0], fs, 128), convertAmpltoSpec(seg[1],fs,128)])
    #     vb1.addItem(a, ignoreBounds=True)
    #
    # QtGui.QApplication.instance().exec_()

# detectClicks()
