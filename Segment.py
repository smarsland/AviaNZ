# Version 0.2 20/7/16
# Author: Stephen Marsland

import numpy as np
import pywt
import scipy.ndimage as spi
import pylab as pl
import matplotlib
import librosa
#import cv2

# TODO:
# Median clipping: play with the various parameters, get one that I'm happy with
#   -> size of diamond, med filter, min size of blob
# Should at least do power, wavelets, then think about more
# Add Nirosha's approach of simultaneous segmentation and recognition using wavelets
# Try onset_detect from librosa
#   -> doesn't do offset :)
# Want to take each second or so and say yes or no for presence (?)
# Should compute SNR
# Use spectrogram as well -> object detection, turn into shape

class Segment:
    # This class implements various signal processing algorithms for the AviaNZ interface
    def __init__(self,data,sg,sp,fs):
        self.data = data
        # This is the length of a window to average to get the power
        self.length = 100
        self.segments = []
        self.fs = fs
        self.maxlevel = 5
        # This is the spectrogram
        self.sg = sg
        # This is the reference to SignalProc
        self.sp = sp

    def setNewData(self,data,sg,fs):
        self.data = data
        self.sg = sg
        self.fs = fs

    def segmentByAmplitude(self,threshold):
        self.seg = np.where(np.abs(self.data)>threshold,1,0)
        inSegment=False
        segments = []
        for i in range(len(self.data)):
            if self.seg[i] > 0:
                if inSegment:
                    pass
                else:
                    inSegment = True
                    start = i
            else:
                if inSegment:
                    # If segment is long enough to be worth bothering with
                    if i-start>1:
                        segments.append([float(start)/self.fs, float(i)/self.fs])
                    inSegment = False
        return segments

    def findCorrelations(self):
        self.data = librosa.core.audio.resample(self.fs)
        self.wData = self.sp.denoise(self.data,thresholdType='soft',maxlevel=5)

        # Bandpass filter
        import scipy.signal as signal
        nyquist = self.fs/2.0
        ripple_db = 80.0
        width = 1.0/nyquist
        ntaps, beta = signal.kaiserord(ripple_db, width)
        taps = signal.firwin(ntaps,cutoff = [1100,7500], window=('kaiser', beta),pass_zero=False)
        self.fwData = signal.lfilter(taps, 1.0, self.wData)

        pl.plot(self.data)

        # Get the correlations between marked up example and this

        #self.wp = pywt.WaveletPacket(data=self.fwData, wavelet='dmey', mode='symmetric',maxlevel=5)

    def segmentByWavelet(self,thresholdType='soft',threshold=None,maxlevel=5,bandpass=True,wavelet='dmey5',sampleRate=1000,start=500,end=800,learning=True):
        # Need to think about this. Basically should play with it (without the interface) and do some computations
        # and plot the wavelet packets

        data = librosa.core.audio.resample(audiodata, self.sampleRate, sampleRate)
        wData = self.sp.denoise(data, thresholdType,threshold,maxlevel,bandpass,wavelet)
        fwData = self.sp.bandpassFilter(wData, start,end)

        if learning:
            # Get the segmentation from the excel file
            annotation = np.zeros(300)
            count = 0
            from openpyxl import load_workbook
            wb = load_workbook(filename='train1-boom-sec.xlsx')
            ws = wb.active
            for row in ws.iter_rows('B2:B301'):
                annotation[count] = row[0].value
                count += 1

        r, coeffs = self.compute_r(annotation,fwData,sampleRate)
        print r
        return coeffs


    def compute_r(self,annotation, fwData, fs):
        # Get the wavelet packet data
        # There are 62 coefficients up to level 5 of the wavelet tree (without root), and 300 seconds in 5 mins
        # The energy is the sum of the squares of the data in each leaf divided by the total in the tree
        coeffs = np.zeros((62, 300))
        for t in range(300):
            E = []
            for level in range(1, 6):
                wp = pywt.WaveletPacket(data=fwData[t * fs:(t + 1) * fs], wavelet='dmey', mode='symmetric',
                                        maxlevel=level)
                e = np.array([np.sum(n.data ** 2) for n in wp.get_level(level, "natural")])
                if np.sum(e) > 0:
                    e = 100.0 * e / np.sum(e)
                E = np.concatenate((E, e), axis=0)
            coeffs[:, t] = E

        # Find the correlations (point-biserial)
        # r = (M_p - M_q) / S * sqrt(p*q), M_p = mean for those that are 0, S = std dev overall, p = proportion that are 0.

        w0 = np.where(annotation == 0)[0]
        w1 = np.where(annotation == 1)[0]

        r = np.zeros(62)
        for count in range(62):
            r[count] = (np.mean(coeffs[count, w0]) - np.mean(coeffs[count, w1])) / np.std(
                coeffs[count, :]) * np.sqrt(len(w0) * len(w1) / 90000.0)

        order = np.argsort(r)
        print order[-1::-1]
        return r, coeffs

    def SnNR(self,startSignal,startNoise):
        pS = np.sum(self.data[startSignal:startSignal+self.length]**2)/self.length
        pN = np.sum(self.data[startNoise:startNoise+self.length]**2)/self.length
        return 10.*np.log10(pS/pN)

    def medianClip(self,threshold):
        # TODO: median filter and dilation; also see below
        segments = []
        #np.savetxt('sp.txt',self.sg)
        self.sg = self.sg/np.max(self.sg)
        self.sg = self.sg[4:232, :]

        rowmedians = np.median(self.sg, axis=1)
        colmedians = np.median(self.sg, axis=0)
        clipped = np.zeros(np.shape(self.sg),dtype=int)
        for i in range(np.shape(self.sg)[0]):
            for j in range(np.shape(self.sg)[1]):
                if (self.sg[i, j] > threshold * rowmedians[i] and self.sg[i, j] > threshold * colmedians[j]):
                    clipped[i, j] = 1

        print np.shape(clipped)
        # This is the stencil for the closing and dilation
        diamond = np.zeros((5,5),dtype=int)
        diamond[2,:] = 1
        diamond[:,2] = 1
        diamond[1,1] = diamond[1,3] = diamond[3,1] = diamond[3,3] = 1

        clipped = spi.binary_closing(clipped,structure=diamond).astype(int)
        clipped = spi.binary_dilation(clipped,structure=diamond).astype(int)
        clipped = spi.median_filter(clipped,size=5)

        import skimage.measure as skm
        blobs = skm.regionprops(skm.label(clipped.astype(int)))

        for i in blobs:
            if i.filled_area < 10:
                blobs.remove(i)

        return clipped, blobs


    def lasseck(self):
        #import SignalProc
        # Exactly the description in Lasseck's work
        data, fs = librosa.load('Sound Files/male1.wav')
        sp = SignalProc.SignalProc(data, fs,512,128)
        self.sg = sp.spectrogram(data)
        self.sg = self.sg/np.max(self.sg)

        self.sg = self.sg[4:232,:]

        threshold = 3

        rowmedians = np.median(self.sg, axis=1)
        colmedians = np.median(self.sg, axis=0)
        clipped = np.zeros(np.shape(self.sg))
        for i in range(np.shape(self.sg)[0]):
            for j in range(np.shape(self.sg)[1]):
                if (self.sg[i, j] > threshold * rowmedians[i] and self.sg[i, j] > threshold * colmedians[j]):
                    clipped[i, j] = 1
        pl.ion()
        pl.figure()
        pl.subplot(5,1,1), pl.imshow(self.sg)
        pl.subplot(5,1,2), pl.imshow(clipped)
        #np.savetxt('out.txt', clipped)
        # This is the stencil for the closing and dilation
        diamond = np.zeros((5,5),dtype=int)
        diamond[2,:] = 1
        diamond[:,2] = 1
        diamond[1,1] = diamond[1,3] = diamond[3,1] = diamond[3,3] = 1

        d1 = spi.binary_closing(clipped,structure=diamond).astype(int)
        pl.subplot(5,1,3), pl.imshow(d1)
        d2 = spi.binary_dilation(d1,structure=diamond).astype(int)
        pl.subplot(5,1,4), pl.imshow(d2)
        d3 = spi.median_filter(d2,size=5)
        pl.subplot(5,1,5), pl.imshow(d3)

        pl.figure()
        pl.subplot(5,1,1), pl.imshow(self.sg)
        pl.subplot(5,1,2), pl.imshow(clipped)
        diamond = np.zeros((5, 5), dtype=int)
        diamond[2, 1:4] = 1
        diamond[1:4, 2] = 1
        d1 = spi.binary_closing(clipped,structure=diamond).astype(int)
        pl.subplot(5,1,3), pl.imshow(d1)
        d2 = spi.binary_dilation(d1,structure=diamond).astype(int)
        pl.subplot(5,1,4), pl.imshow(d2)
        d3 = spi.median_filter(d2,size=5)
        pl.subplot(5,1,5), pl.imshow(d3)

        import skimage.measure as skm
        blobs = skm.regionprops(skm.label(out.astype(int)))

        for i in blobs:
            if i.filled_area < 10:
                blobs.remove(i)


        return clipped, blobs

    def onsets(self):
        onsets = librosa.onset.onset_detect(data)
        onset_times = librosa.frames_to_time(onsets)

    def testMedianClip(self,sg,minsize):
        # TODO: median filter and dilation; also see below
        sg = sg / np.max(sg)
        sg = sg[4:232, :]

        rowmedians = np.median(sg, axis=1)
        colmedians = np.median(sg, axis=0)
        clipped = np.zeros(np.shape(sg), dtype=int)
        for i in range(np.shape(sg)[0]):
            for j in range(np.shape(sg)[1]):
                if (sg[i, j] > 3 * rowmedians[i] and sg[i, j] > 3 * colmedians[j]):
                    clipped[i, j] = 1

        print np.shape(clipped)
        # This is the stencil for the closing and dilation
        diamond = np.zeros((5, 5), dtype=int)
        diamond[2, :] = 1
        diamond[:, 2] = 1
        diamond[1, 1] = diamond[1, 3] = diamond[3, 1] = diamond[3, 3] = 1

        a1 = spi.binary_closing(clipped, structure=diamond).astype(int)
        a2 = spi.binary_dilation(a1, structure=diamond).astype(int)
        a3 = spi.binary_erosion(a2, structure=diamond).astype(int)
        a4 = spi.median_filter(a3, size=5)

        # Delete if too skinny, not just too small?
        # Need to delete small blocks inside large blocks
        # And also split very large blocks that contain many syllables?
        # Erosion?

        import skimage.measure as skm
        blobsa1 = skm.regionprops(skm.label(a1.astype(int)))
        blobsa2 = skm.regionprops(skm.label(a2.astype(int)))
        blobsa3 = skm.regionprops(skm.label(a3.astype(int)))
        blobsa4 = skm.regionprops(skm.label(a4.astype(int)))

        for i in blobsa1:
            if i.filled_area < minsize or i.minor_axis_length<5:
                blobsa1.remove(i)
        for i in blobsa2:
            if i.filled_area < minsize or i.minor_axis_length<5:
                blobsa2.remove(i)
        for i in blobsa3:
            if i.filled_area < minsize or i.minor_axis_length<5:
                blobsa3.remove(i)
        for i in blobsa4:
            if i.filled_area < minsize or i.minor_axis_length < 5:
                blobsa4.remove(i)

        diamond = np.zeros((5, 5), dtype=int)
        diamond[2, 1:4] = 1
        diamond[1:4, 2] = 1

        b1 = spi.binary_closing(clipped, structure=diamond).astype(int)
        b2 = spi.binary_dilation(b1, structure=diamond).astype(int)
        b3 = spi.binary_erosion(b2, structure=diamond).astype(int)
        b4 = spi.median_filter(b3, size=5)

        blobsb1 = skm.regionprops(skm.label(b1.astype(int)))
        blobsb2 = skm.regionprops(skm.label(b2.astype(int)))
        blobsb3 = skm.regionprops(skm.label(b3.astype(int)))
        blobsb4 = skm.regionprops(skm.label(b4.astype(int)))

        for i in blobsb1:
            if i.filled_area < minsize or i.minor_axis_length<5:
                blobsb1.remove(i)
        for i in blobsb2:
            if i.filled_area < minsize or i.minor_axis_length<5:
                blobsb2.remove(i)
        for i in blobsb3:
            if i.filled_area < minsize or i.minor_axis_length<5:
                blobsb3.remove(i)
        for i in blobsb4:
            if i.filled_area < minsize or i.minor_axis_length<5:
                blobsb4.remove(i)

        pl.figure()
        #pl.subplot(5, 1, 1), pl.imshow(sg,aspect='auto')
        pl.subplot(5, 1, 1), pl.imshow(clipped,aspect='auto')
        pl.subplot(5, 1, 2), pl.imshow(a1,aspect='auto')
        for i in blobsa1:
            pl.gca().add_patch(pl.Rectangle((i.bbox[1],i.bbox[0]),i.bbox[3]-i.bbox[1],i.bbox[2]-i.bbox[0],facecolor='w',alpha=0.5))
        pl.subplot(5, 1, 3), pl.imshow(a2,aspect='auto')
        for i in blobsa2:
            pl.gca().add_patch(pl.Rectangle((i.bbox[1],i.bbox[0]),i.bbox[3]-i.bbox[1],i.bbox[2]-i.bbox[0],facecolor='w',alpha=0.5))
        pl.subplot(5, 1, 4), pl.imshow(a3,aspect='auto')
        for i in blobsa3:
            pl.gca().add_patch(pl.Rectangle((i.bbox[1],i.bbox[0]),i.bbox[3]-i.bbox[1],i.bbox[2]-i.bbox[0],facecolor='w',alpha=0.5))
        pl.subplot(5, 1, 5), pl.imshow(a5,aspect='auto')
        for i in blobsa4:
            pl.gca().add_patch(pl.Rectangle((i.bbox[1],i.bbox[0]),i.bbox[3]-i.bbox[1],i.bbox[2]-i.bbox[0],facecolor='w',alpha=0.5))

        pl.figure()
        #pl.subplot(5, 1, 1), pl.imshow(sg,aspect='auto')
        pl.subplot(5, 1, 1), pl.imshow(clipped,aspect='auto')
        pl.subplot(5, 1, 2), pl.imshow(b1,aspect='auto')
        for i in blobsb1:
            pl.gca().add_patch(
                pl.Rectangle((i.bbox[1], i.bbox[0]), i.bbox[3] - i.bbox[1], i.bbox[2] - i.bbox[0], facecolor='w',
                             alpha=0.5))
        pl.subplot(5, 1, 3), pl.imshow(b2,aspect='auto')
        for i in blobsb2:
            pl.gca().add_patch(
                pl.Rectangle((i.bbox[1], i.bbox[0]), i.bbox[3] - i.bbox[1], i.bbox[2] - i.bbox[0], facecolor='w',
                             alpha=0.5))
        pl.subplot(5, 1, 4), pl.imshow(b3,aspect='auto')
        for i in blobsb3:
            pl.gca().add_patch(
                pl.Rectangle((i.bbox[1], i.bbox[0]), i.bbox[3] - i.bbox[1], i.bbox[2] - i.bbox[0], facecolor='w',
                             alpha=0.5))
        pl.subplot(5, 1, 5), pl.imshow(b4,aspect='auto')
        for i in blobsb4:
            pl.gca().add_patch(
                pl.Rectangle((i.bbox[1], i.bbox[0]), i.bbox[3] - i.bbox[1], i.bbox[2] - i.bbox[0], facecolor='w',
                             alpha=0.5))
        return clipped, a1, a2, a3, b1, b2, b3, blobsa1, blobsa2, blobsa3, blobsb1, blobsb2, blobsb3
# pl.show()
# pl.ion()
# a = Segment([],[],None,None)
# c = a.lasseck()
# #a.plot_lasseck(c)
# pl.show()
# pl.ion()
# pl.show()