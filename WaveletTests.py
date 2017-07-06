
import pywt
import wavio
import numpy as np
import SignalProc
import WaveletSegment

def showEnergies():
    import pylab as pl
    pl.ion()

    filename = 'Sound Files/tril1_d1.wav'
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

    e1 = computeWaveletEnergy_1s(data1,'dmey2')
    pl.subplot(5,1,1)
    pl.plot(e1)
    e2 = computeWaveletEnergy_1s(data2,'dmey2')
    pl.subplot(5,1,2)
    pl.plot(e2)
    e3 = computeWaveletEnergy_1s(data3,'dmey2')
    pl.subplot(5,1,3)
    pl.plot(e3)
    e4 = computeWaveletEnergy_1s(data4,'dmey2')
    pl.subplot(5,1,4)
    pl.plot(e4)
    e5 = computeWaveletEnergy_1s(data5,'dmey2')
    pl.subplot(5,1,5)
    pl.plot(e5)

    pl.figure()
    pl.plot(e1)
    pl.plot(e2)
    pl.plot(e3)
    pl.plot(e4)
    pl.plot(e5)

    return e2
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

    #tbd = [1, 3, 7, 15, 31]
    tbd = np.concatenate([np.arange(30),np.arange(50,63)])

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