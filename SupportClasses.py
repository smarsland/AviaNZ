# Version 0.2 10/7/17
# Author: Stephen Marsland

# Support classes for the AviaNZ program
# Mostly subclassed from pyqtgraph
#     from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QAbstractButton
from PyQt5.QtCore import QTime, QFile, QIODevice, QBuffer, QByteArray
from PyQt5.QtMultimedia import QAudio, QAudioOutput, QAudioFormat

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.functions as fn

from openpyxl import load_workbook, Workbook
from openpyxl.styles import colors
from openpyxl.styles import Font, Color

from scipy import signal
from scipy.signal import medfilt
import SignalProc
import WaveletFunctions
import Segment

from time import sleep

import librosa

import math
import numpy as np
import os, json
import copy

import pywt
import wavio

import io

# import WaveletSegment

class preProcess:
    """ This class implements few pre processing methods to avoid noise
    """
    # todo: remove duplicate preprocess in 'Wavelet Segments'

    def __init__(self,audioData=None, sampleRate=0, species='Kiwi', df=False, wavelet='dmey2'):
        self.audioData=audioData
        self.sampleRate=sampleRate
        self.species=species
        self.df=df
        if wavelet == 'dmey2':
            [lowd, highd, lowr, highr] = np.loadtxt('dmey.txt')
            self.wavelet = pywt.Wavelet(filter_bank=[lowd, highd, lowr, highr])
            self.wavelet.orthogonal=True
        else:
            self.wavelet = wavelet
        self.sp = SignalProc.SignalProc([], 0, 256, 128)
        self.WaveletFunctions = WaveletFunctions.WaveletFunctions(data=self.audioData, wavelet=self.wavelet, maxLevel=20)

    def denoise_filter(self, level=5):
        # set df=True to perform both denoise and filter
        # df=False to skip denoise
        if self.species == 'Kiwi':
            f1 = 1100
            f2 = 7000
            fs = 16000
        elif self.species == 'Ruru':
            f1 = 500
            f2 = 7000
            fs = 16000
        elif self.species == 'Bittern':
            f1 = 100
            f2 = 200
            fs = 1000
        elif self.species == 'Sipo':
            f1 = 1200
            f2 = 3800
            fs = 8000
        else:
            fs = 8000

        if self.sampleRate != fs:
            self.audioData = librosa.core.audio.resample(self.audioData, self.sampleRate, fs)
            self.sampleRate = fs

        # Get the five level wavelet decomposition
        if self.df == True:
            denoisedData = self.WaveletFunctions.waveletDenoise(self.audioData, thresholdType='soft', wavelet=self.wavelet,maxLevel=level)
        else:
            denoisedData=self.audioData  # this is to avoid washing out very fade calls during the denoising

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

        if self.species in ['Kiwi', 'Ruru', 'Bittern', 'Sipo']:
            filteredDenoisedData = self.sp.ButterworthBandpass(denoisedData, self.sampleRate, low=f1, high=f2)
            # filteredDenoisedData = self.sp.bandpassFilter(denoisedData, start=f1, end=f2, sampleRate=self.sampleRate)
        # elif species == 'Ruru':
        #     filteredDenoisedData = self.sp.ButterworthBandpass(denoisedData, self.sampleRate, low=f1, high=7000)
        # elif species == 'Sipo':
        #     filteredDenoisedData = self.sp.ButterworthBandpass(denoisedData, self.sampleRate, low=1200, high=3800)
        else:
            filteredDenoisedData = denoisedData

        return filteredDenoisedData, self.sampleRate

class postProcess:
    """ This class implements few post processing methods to avoid false positives

    segments:   detected segments in form of [[s1,e1], [s2,e2],...]
    species:    species to consider
    minLen:     minimum length for the species # min length for kiwi is 5 secs
    """

    def __init__(self,audioData=None, sampleRate=0, segments=[], species='Kiwi', minLen=0):
        self.audioData = audioData
        self.sampleRate = sampleRate
        self.segments = segments
        self.species = species
        self.minLen = minLen
        if self.minLen == 0 and self.species =='Kiwi':
            self.minLen = 10
        # self.confirmedSegments = []  # post processed detections with confidence TP
        # self.segmentstoCheck = []  # need more testing to confirm

    def short(self):
        """
        This will delete segments < minLen/2 secs
        """
        newSegments = []
        for seg in self.segments:
            if seg[0] == -1:
                newSegments.append(seg)
            elif seg[1] - seg[0] >= self.minLen/2:
                newSegments.append(seg)
            else:
                continue
        self.segments = newSegments

    def wind(self, Tmean_wind = 1e-8):
        """
        delete wind corrupted segments (targeting moderate wind and above) if no sign of kiwi (check len)
        Automatic Identification of Rainfall in Acoustic Recordings by Carol Bedoya, Claudia Isaza, Juan M.Daza, and Jose D.Lopez
        """
        newSegments = copy.deepcopy(self.segments)
        for seg in self.segments:
            if seg[0] == -1:
                continue
            else:  # read the sound segment and check for wind
                secs = seg[1] - seg[0]
                data = self.audioData[seg[0]*self.sampleRate:seg[1]*self.sampleRate]

                wind_lower = 2.0 * 100 / self.sampleRate
                wind_upper = 2.0 * 250 / self.sampleRate

                f, p = signal.welch(data, fs=self.sampleRate, window='hamming', nperseg=512, detrend=False)

                # check wind
                limite_inf = int(
                    round(p.__len__() * wind_lower))  # minimum frequency of the rainfall frequency band 0.00625(in
                # normalized frequency); in Hz = 0.00625 * (44100 / 2) = 100 Hz
                limite_sup = int(
                    round(p.__len__() * wind_upper))  # maximum frequency of the rainfall frequency band 0.03125(in
                # normalized frequency); in Hz = 0.03125 * (44100 / 2) = 250 Hz
                a_wind = p[
                         limite_inf:limite_sup]  # section of interest of the power spectral density.Step 2 in Algorithm 2.1

                mean_a_wind = np.mean(
                    a_wind)  # mean of the PSD in the frequency band of interest.Upper part of the step 3 in Algorithm 2.1
                # std_a_wind = np.std(a_wind)  # standar deviation of the PSD in the frequency band of the interest. Lower part of the step 3 in Algorithm 2.1
                if mean_a_wind > Tmean_wind:
                    if secs > self.minLen:  # just check duration
                        continue
                    else:
                        print(file, seg, "--> windy")
                        newSegments.remove(seg)
        self.segments = newSegments

    def rainClick(self):
        """
        delete random clicks e.g. rain. Check for sign of kiwi (len)
        """
        newSegments = copy.deepcopy(self.segments)
        if newSegments.__len__() > 1:
            mfcc = librosa.feature.mfcc(self.audioData, self.sampleRate)
            # Normalise
            mfcc -= np.mean(mfcc, axis=0)
            mfcc /= np.max(np.abs(mfcc), axis=0)
            mean = np.mean(mfcc[1, :])
            std = np.std(mfcc[1, :])
            thr = mean - 2 * std  # mfcc1 thr for the recording

            for seg in self.segments:
                if seg[0] == -1:
                    continue
                else:
                    secs = seg[1] - seg[0]
                    data = self.audioData[seg[0]*self.sampleRate:seg[1]*self.sampleRate]
                mfcc = librosa.feature.mfcc(data, self.sampleRate)
                # Normalise
                mfcc -= np.mean(mfcc, axis=0)
                mfcc /= np.max(np.abs(mfcc), axis=0)
                mfcc1 = mfcc[1, :]  # mfcc1 of the segment
                if np.min(mfcc1) < thr:
                    if secs > self.minLen:  # just check duration>10 sec
                        continue
                    else:
                        newSegments.remove(seg)
        self.segments = newSegments

    def fundamentalFrq(self ):
        '''
        Check for fundamental frequency of the segments, discard the segments that does not indicate the species.
        '''
        newSegments = copy.deepcopy(self.segments)
        for seg in self.segments:
            if seg[0] == -1:
                continue
            else:
                # read the sound segment and check fundamental frq.
                secs = seg[1] - seg[0]
                data = self.audioData[seg[0]*self.sampleRate:seg[1]*self.sampleRate]

                # bring the segment into 16000
                if self.sampleRate != 16000:
                    data = librosa.core.audio.resample(data, self.sampleRate, 16000)
                    sampleRate = 16000
                else:
                    sampleRate = self.sampleRate
                # denoise before fundamental frq. extraction
                sc = preProcess(audioData=data, sampleRate=sampleRate, species='', df=True)  # species left empty to avoid bandpass filter
                data, sampleRate = sc.denoise_filter(level=10)

                sp = SignalProc.SignalProc([], 0, 512, 256)
                sgRaw = sp.spectrogram(data, 512, 256, mean_normalise=True, onesided=True, multitaper=False)
                segment = Segment.Segment(data, sgRaw, sp, sampleRate, 512, 256)
                pitch, y, minfreq, W = segment.yin(minfreq=100)
                ind = np.squeeze(np.where(pitch > minfreq))
                pitch = pitch[ind]
                if pitch.size == 0:
                    print(file, 'segment ', seg, ' *++ no fundamental freq detected, could be faded kiwi or noise')
                    newSegments.remove(seg)
                    continue
                ind = ind * W / 512
                x = (pitch * 2. / sampleRate * np.shape(sgRaw)[1]).astype('int')

                from scipy.signal import medfilt
                x = medfilt(pitch, 15)

                if ind.size < 2:
                    if pitch > 1200 and pitch < 4200:
                        print(file, 'segment ', seg, round(pitch), ' *##kiwi found')
                    else:
                        print(file, 'segment ', seg, round(
                            pitch), ' *-- fundamental freq is out of kiwi region, could be noise')
                        newSegments.remove(seg)
                else:
                    # Get the individual pieces
                    segs = segment.identifySegments(ind, maxgap=10, minlength=5)
                    count = 0
                    if segs == []:
                        if np.mean(pitch) > 1200 and np.mean(pitch) < 4000:
                            print(file, 'segment ', seg, round(np.mean(pitch)), ' *## kiwi found ')
                        else:
                            print(file, 'segment ', seg, round(
                                np.mean(pitch)), ' *-- fundamental freq is out of kiwi region, could be noise')
                            newSegments.remove(seg)
                            continue
                    flag = False
                    for s in segs:
                        count += 1
                        s[0] = s[0] * sampleRate / float(256)
                        s[1] = s[1] * sampleRate / float(256)
                        i = np.where((ind > s[0]) & (ind < s[1]))
                        if np.mean(x[i]) > 1200 and np.mean(x[i]) < 4000:
                            print(file, 'segment ', seg, round(np.mean(x[i])), ' *## kiwi found ##')
                            flag = True
                            break
                    if not flag:
                        newSegments.remove(seg)
        self.segments = newSegments

    # ***no use of the rest of the functions in this class for the moment.
    def eRatioConfd(self, seg, AviaNZ_extra = False):
        '''
        This is a post processor to introduce some confidence level
        high ratio --> classes 1-3 'good' calls
        low ratio --> classes 4-5 'weak' calls
        ratio = energy in band/energy above the band
        The problem with this simple classifier is that the ratio is relatively low when the
        calls are having most of the harmonics (close range)
        Mostly works
        '''
        # TODO: Check range -- species specific of course!
        # Also recording range specific -- 16KHz will be different -- resample?
        # import WaveletSegment
        # ws = WaveletSegment.WaveletSegment()
        # detected = np.where(self.detections > 0)
        # # print "det",detected
        # if np.shape(detected)[1] > 1:
        #     detected = ws.identifySegments(np.squeeze(detected))
        # elif np.shape(detected)[1] == 1:
        #     detected = ws.identifySegments(detected)
        # else:
        #     detected=[]
        if seg: # going through segments
            sp = SignalProc.SignalProc(self.audioData[int(seg[0])*self.sampleRate:int(seg[1])*self.sampleRate], self.sampleRate, 256, 128)
            self.sg = sp.spectrogram(self.audioData[int(seg[0])*self.sampleRate:int(seg[1])*self.sampleRate])
        else: # eRatio of the whole file e.g. the extracted segments
            sp = SignalProc.SignalProc(self.audioData, self.sampleRate, 256, 128)
            self.sg = sp.spectrogram(self.audioData)

        f1 = 1500
        f2 = 4000
        F1 = f1 * np.shape(self.sg)[1] / (self.sampleRate / 2.)
        F2 = f2 * np.shape(self.sg)[1] / (self.sampleRate / 2.)

        e = np.sum(self.sg[:,int(F2):],axis=1)
        eband = np.sum(self.sg[:,int(F1):int(F2)],axis=1)
        if AviaNZ_extra:
            return eband/e, 1
        else:
            return np.mean(eband/e)

    def eRatioConfdV2(self, seg):
            '''
            This is a post processor to introduce some confidence level
            testing a variation of eratio = energy in band within segment/energy in band 10sec before or after the segment
            '''
            # TODO: Check range -- species specific of course!
            # Also recording range specific -- 16KHz will be different -- resample?
            if seg:  # going through segments
                sp = SignalProc.SignalProc(self.audioData[int(seg[0]) * self.sampleRate:int(seg[1]) * self.sampleRate],
                                           self.sampleRate, 256, 128)
                self.sg = sp.spectrogram(self.audioData[int(seg[0]) * self.sampleRate:int(seg[1]) * self.sampleRate])
                # get neighbour
                if seg[0] >= 0: #10 sec before
                    sp_nbr = SignalProc.SignalProc(self.audioData[int(seg[0]-10) * self.sampleRate:int(seg[0]) * self.sampleRate],
                                           self.sampleRate, 256, 128)
                    sg_nbr = sp_nbr.spectrogram(self.audioData[int(seg[0]-10) * self.sampleRate:int(seg[0]) * self.sampleRate])
                else: # 10 sec after
                    sp_nbr = SignalProc.SignalProc(
                        self.audioData[int(seg[1]) * self.sampleRate:int(seg[1]+10) * self.sampleRate],
                        self.sampleRate, 256, 128)
                    sg_nbr = sp_nbr.spectrogram(
                        self.audioData[int(seg[1]) * self.sampleRate:int(seg[1]+10) * self.sampleRate])

            f1 = 1500
            f2 = 7000
            F1 = f1 * np.shape(self.sg)[1] / (self.sampleRate / 2.)
            F2 = f2 * np.shape(self.sg)[1] / (self.sampleRate / 2.)

            # e = np.sum(self.sg[:, int(F2):], axis=1)
            eband = np.sum(self.sg[:, int(F1):int(F2)], axis=1)
            enbr = np.sum(sg_nbr[:, int(F1):int(F2)], axis=1)
            return (np.mean(eband) / np.mean(enbr))

    def eRatioConfd2(self, thr=2.5):
        '''
        Same as above but it checks all segments (delete after Tier1)
        This is a post processor to introduce some confidence level
        high ratio --> classes 1-3 'good' calls
        low ratio --> classes 4-5 'weak' calls
        ratio = energy in band/energy above the band
        The problem with this simple classifier is that the ratio is relatively low when the
        calls are having most of the harmonics (close range)
        Mostly works
        '''
        # TODO: Check range -- species specific of course!
        # Also recording range specific -- 16KHz will be different -- resample?
        # import WaveletSegment
        # ws = WaveletSegment.WaveletSegment()
        # detected = np.where(self.detections > 0)
        # # print "det",detected
        # if np.shape(detected)[1] > 1:
        #     detected = ws.identifySegments(np.squeeze(detected))
        # elif np.shape(detected)[1] == 1:
        #     detected = ws.identifySegments(detected)
        # else:
        #     detected=[]

        sp = SignalProc.SignalProc(self.audioData, self.sampleRate, 256, 128)
        self.sg = sp.spectrogram(self.audioData)

        # f1 = 1500
        # f2 = 4000
        # F1 = f1 * np.shape(self.sg)[1] / (self.sampleRate / 2.)
        # F2 = f2 * np.shape(self.sg)[1] / (self.sampleRate / 2.)
        #
        # e = np.sum(self.sg[:,F2:],axis=1)
        # eband = np.sum(self.sg[:,F1:F2],axis=1)
        #
        # return eband/e, 1
        f1 = 1100
        f2 = 4000
        for seg in self.segments:
            # e = np.sum(self.sg[seg[0] * self.sampleRate / 128:seg[1] * self.sampleRate / 128, :]) /128     # whole frequency range
            # nBand = 128  # number of frequency bands
            e = np.sum(self.sg[seg[0] * self.sampleRate / 128:seg[1] * self.sampleRate / 128, f2 * 128 / (self.sampleRate / 2):])  # f2:
            nBand = 128 - f2 * 128 / (self.sampleRate / 2)    # number of frequency bands
            e=e/nBand   # per band power

            eBand = np.sum(self.sg[seg[0] * self.sampleRate / 128:seg[1] * self.sampleRate / 128, f1 * 128 / (self.sampleRate / 2):f2 * 128 / (self.sampleRate / 2)]) # f1:f2
            nBand = f2 * 128 / (self.sampleRate / 2) - f1 * 128 / (self.sampleRate / 2)
            eBand = eBand / nBand
            r = eBand/e
            # print seg, r
            if r>thr:
                self.confirmedSegments.append(seg)
            else:
                self.segmentstoCheck.append(seg)

    def detectClicks(self,sg=None):
        '''
        This function finds 'click' sounds that normally pick up by any detector as false positives.
        Remove those from the output.
        '''
        # TODO: this also tends to delete true positives! Try looking back and forth to see if its longer than 1 sec
        #fs = self.sampleRate
        #data = self.audioData

        if sg == None:
            sp = SignalProc.SignalProc(self.audioData, self.sampleRate, 256, 128)
            self.sg = sp.spectrogram(self.audioData)
        else:
            self.sg = sg
        # s = Segment(data, sg, sp, fs, 50)

        energy = np.sum(self.sg, axis=1)
        energy = medfilt(energy, 15)
        e2 = np.percentile(energy, 90) * 2
        # Step 1: clicks have high energy
        clicks = np.squeeze(np.where(energy > e2))
        # Step 2: clicks are short!

        # clicks = s.identifySegments(clicks, minlength=1)
        clicks = clicks * 128 / self.sampleRate  # convert frame numbers to seconds
        #c = list(set(clicks))
        #for i in c:
        #    self.detections[i] = 0        # remove clicks
        return energy, e2

class exportSegments:
    """ This class saves the batch detection results(Find Species) and also current annotations (AviaNZ interface)
        in three different formats: time stamps, presence/absence, and per second presence/absence
        in an excel workbook. It makes the workbook if necessary.

        TODO: Ask for what species if not specified
        TODO: Add a presence/absence at minute (or 5 minute) resolution
        TODO: Save the annotation files for batch processing

        Inputs
            segments:   detected segments in form of [[s1,e1], [s2,e2],...] # excel is still based on this, to be fixed later using next two.
                segmentstoCheck     : segments without confidence in form of [[s1,e1], [s2,e2],...]
                confirmedSegments   : segments with confidence
            species:    e.g. 'Kiwi'. Default is 'all'
            startTime:  start time of the recording (in DoC format). Default is 0
            dirName:    directory name
            filename:   file name e.g.
            datalength: number of data points in the recording
            sampleRate: sample rate
            method:     e.g. 'Wavelets'. Default is 'Default'
            resolution: output resolution on excel (sheet 3) in seconds. Default is 1
            trainTest:  is it for training/testing (=True) or real use (=False)
            withConf:   is it with some level of confidence? e.g. after post-processing (e ratio). Default is 'False'
            seg_pos:    possible segments are needed apart from the segments when withConf is True. This is just to
                        generate the annotation including the segments with conf (kiwi) and without confidence (kiwi?).
            minLen: minimum length of a segment in secs

    """

    def __init__(self, segments=[], confirmedSegments=[], segmentstoCheck=[], species='all', startTime=0, dirName='', filename='',datalength=0,sampleRate=0, method="Default", resolution=1, trainTest=False, withConf=False, seg_pos=[], operator='', reviewer='', minLen=0):
        self.segments=segments
        self.confirmedSegments = confirmedSegments
        self.segmentstoCheck = segmentstoCheck
        self.species=species
        self.startTime=startTime
        self.dirName=dirName
        self.filename=filename
        self.datalength=datalength
        self.sampleRate=sampleRate
        self.method=method
        if resolution>math.ceil(float(self.datalength)/self.sampleRate):
            self.resolution=int(math.ceil(float(self.datalength)/self.sampleRate))
        else:
            self.resolution=resolution
        self.trainTest = trainTest
        self.withConf=withConf  # todo: remove
        self.seg_pos=seg_pos #segmentstoCheck
        self.operator = operator
        self.reviewer = reviewer
        self.minLen = minLen

    def excel(self):
        """ This saves the detections in three different formats: time stamps, presence/absence, and per second presence/absence
        in an excel workbook. It makes the workbook if necessary.
        TODO: Ask for what species if not specified
        TODO: Add a presence/absence at minute (or 5 minute) resolution
        """
        def makeNewWorkbook():
            wb = Workbook()
            wb.create_sheet(title='Time Stamps', index=1)
            wb.create_sheet(title='Presence Absence', index=2)
            wb.create_sheet(title='Per second', index=3)

            ws = wb['Time Stamps']
            ws.cell(row=1, column=1, value="File Name")
            ws.cell(row=1, column=2, value="start (hh:mm:ss)")
            ws.cell(row=1, column=3, value="end (hh:mm:ss)")

            # Second sheet
            ws = wb['Presence Absence']
            ws.cell(row=1, column=1, value="File Name")
            ws.cell(row=1, column=2, value="Presence/Absence")

            # Third sheet
            ws = wb['Per second']
            ws.cell(row=1, column=1, value="File Name")
            ws.cell(row=1, column=2, value="Presence=1, Absence=0")

            # TODO: Per minute sheet?

            # Hack to delete original sheet
            wb.remove_sheet(wb['Sheet'])
            return wb

        def writeToExcelp1():
            ws = wb['Time Stamps']
            r = ws.max_row + 1
            # Print the filename
            ws.cell(row=r, column=1, value=str(relfname))
            # Loop over the segments
            for seg in self.segments:
                if int(seg[1]-seg[0]) < self.minLen: # skip very short segments
                    continue
                ws.cell(row=r, column=2, value=str(QTime().addSecs(seg[0]+self.startTime).toString('hh:mm:ss')))
                ws.cell(row=r, column=3, value=str(QTime().addSecs(seg[1]+self.startTime).toString('hh:mm:ss')))
                r += 1

        def writeToExcelp2():
            ws = wb['Presence Absence']
            r = ws.max_row + 1
            ws.cell(row=r, column=1, value=str(relfname))
            ws.cell(row=r, column=2, value='_')
            for seg in self.segments:
                if seg[1]-seg[0] > self.minLen: # skip very short segments
                    ws.cell(row=r, column=2, value='Yes')
                    break

        def writeToExcelp3():
            # todo: use minLen
            ws = wb['Per second']
            r = ws.max_row + 1
            ws.cell(row=r, column=1, value= str(self.resolution) + ' secs resolution')
            ft = Font(color=colors.DARKYELLOW)
            ws.cell(row=r, column=1).font=ft
            c = 2
            for i in range(0,len(detected), self.resolution):
                if i+self.resolution > self.datalength/self.sampleRate:
                    ws.cell(row=r, column=c, value=str(i) + '-' + str(int(math.ceil(float(self.datalength)/self.sampleRate))))
                    ws.cell(row=r, column=c).font = ft
                else:
                    ws.cell(row=r, column=c, value=str(i) + '-' + str(i+self.resolution))
                    ws.cell(row=r, column=c).font = ft
                c += 1
            r += 1
            ws.cell(row=r, column=1, value=str(relfname))
            c = 2
            for i in range(0, len(detected), self.resolution):
                j=1 if np.sum(detected[i:i+self.resolution])>0 else 0
                ws.cell(row=r, column=c, value=j)
                c += 1

        # method=self.algs.currentText()
        if self.withConf:
            eFile = self.dirName + '/DetectionSummary_withConf_' + self.species + '.xlsx'
        else:
            eFile = self.dirName + '/DetectionSummary_' + self.species + '.xlsx'
        relfname = os.path.relpath(str(self.filename), str(self.dirName))
        #print eFile, relfname

        if os.path.isfile(eFile):
            try:
                wb = load_workbook(str(eFile))
            except:
                print("Unable to open file")  # Does not exist OR no read permissions
                return
        else:
            wb = makeNewWorkbook()

        # Now write the data out

        # if self.method == "Wavelets": # and self.trainTest == False:
        #     detected = np.where(self.annotation > 0)
        #     # print "det",detected
        #     if np.shape(detected)[1] > 1:
        #         self.annotation = self.identifySegments(np.squeeze(detected))
        #     elif np.shape(detected)[1] == 1:
        #         self.annotation = self.identifySegments(detected)
        #     else:
        #         self.annotation = []
        #     self.mergeSeg()
        writeToExcelp1()
        writeToExcelp2()

        # Generate per second binary output
        n = math.ceil(float(self.datalength) / self.sampleRate)
        detected = np.zeros(int(n))
        for seg in self.segments:
            for a in range(len(detected)):
                if math.floor(seg[0]) <= a and a < math.ceil(seg[1]):
                    detected[a] = 1
        writeToExcelp3()

        # Save the file
        wb.save(str(eFile))

    # def mergeSeg(self):
    #     # Merge the neighbours, for now wavelet segments
    #     indx = []
    #     for i in range(len(self.annotation) - 1):
    #         # print "index:", i
    #         # print np.shape(self.annotation)
    #         # print self.annotation
    #         if self.annotation[i][1] == self.annotation[i + 1][0]:
    #             indx.append(i)
    #     indx.reverse()
    #     for i in indx:
    #         self.annotation[i][1] = self.annotation[i + 1][1]
    #         del (self.annotation[i + 1])
    #         # return self.annotation

    # def identifySegments(self, seg): #, maxgap=1, minlength=1):
    # # TODO: *** Replace with segmenter.checkSegmentLength(self,segs, mingap=0, minlength=0, maxlength=5.0)
    #     segments = []
    #     # print seg, type(seg)
    #     if len(seg)>0:
    #         for s in seg:
    #             segments.append([s, s+1])
    #     return segments

    def saveAnnotation(self):
        # Save annotations - batch processing
        annotation = []
        # self.startTime = int(self.startTime[:2]) * 3600 + int(self.startTime[2:4]) * 60 + int(self.startTime[4:6])
        annotation.append([-1, str(QTime().addSecs(self.startTime).toString('hh:mm:ss')), "Nirosha", "Stephen", -1])
        # if len(self.segments) > 0 or len(self.seg_pos) > 0:
        if len(self.confirmedSegments) > 0 or len(self.segmentstoCheck) > 0:
            if self.method == "Wavelets":
                # if self.withConf:
                for seg in self.confirmedSegments:
                    # if seg in self.segments:
                    annotation.append([float(seg[0]), float(seg[1]), 0, 0, self.species])
                for seg in self.segmentstoCheck:
                    annotation.append([float(seg[0]), float(seg[1]), 0, 0, self.species + '?'])
                # else:
                #     for seg in self.segments:
                #         annotation.append([float(seg[0]), float(seg[1]), 0, 0, self.species + '?'])
            else:
                for seg in self.segments:
                    annotation.append([float(seg[0]), float(seg[1]), 0, 0, "Don't know"])

        if isinstance(self.filename, str):
            file = open(self.filename + '.data', 'w')
        else:
            file = open(str(self.filename) + '.data', 'w')
        json.dump(annotation, file)

class TimeAxisHour(pg.AxisItem):
    # Time axis (at bottom of spectrogram)
    # Writes the time as hh:mm:ss, and can add an offset
    def __init__(self, *args, **kwargs):
        super(TimeAxisHour, self).__init__(*args, **kwargs)
        self.offset = 0
        self.setLabel('Time', units='hh:mm:ss')

    def tickStrings(self, values, scale, spacing):
        # Overwrite the axis tick code
        return [QTime(0,0,0).addSecs(value+self.offset).toString('hh:mm:ss') for value in values]

    def setOffset(self,offset):
        self.offset = offset
        #self.update()

class TimeAxisMin(pg.AxisItem):
    # Time axis (at bottom of spectrogram)
    # Writes the time as mm:ss, and can add an offset
    def __init__(self, *args, **kwargs):
        super(TimeAxisMin, self).__init__(*args, **kwargs)
        self.offset = 0
        self.setLabel('Time', units='mm:ss')

    def tickStrings(self, values, scale, spacing):
        # Overwrite the axis tick code
        return [QTime(0,0,0).addSecs(value+self.offset).toString('mm:ss') for value in values]

    def setOffset(self,offset):
        self.offset = offset
        self.update()

class ShadedROI(pg.ROI):
    # A region of interest that is shaded, for marking segments
    def paint(self, p, opt, widget):
        #brush = QtGui.QBrush(QtGui.QColor(0, 0, 255, 50))
        if not hasattr(self, 'currentBrush'):
            self.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 50)))
        if not hasattr(self, 'currentPen'):
            self.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0, 255)))
        p.save()
        r = self.boundingRect()
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.currentPen)
        p.setBrush(self.currentBrush)
        p.translate(r.left(), r.top())
        p.scale(r.width(), r.height())
        p.drawRect(0, 0, 1, 1)
        p.restore()

    def setMovable(self,value):
        self.translatable = value

    def setBrush(self, *br, **kargs):
        """Set the brush that fills the region. Can have any arguments that are valid
        for :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.brush = fn.mkBrush(*br, **kargs)
        self.currentBrush = self.brush

    def setPen(self, *br, **kargs):
        self.pen = fn.mkPen(*br, **kargs)
        self.currentPen = self.pen


class ShadedRectROI(ShadedROI):
    # A rectangular ROI that it shaded, for marking segments
    def __init__(self, pos, size, centered=False, sideScalers=False, **args):
        #QtGui.QGraphicsRectItem.__init__(self, 0, 0, size[0], size[1])
        pg.ROI.__init__(self, pos, size, **args)
        if centered:
            center = [0.5, 0.5]
        else:
            center = [0, 0]

        #self.addTranslateHandle(center)
        self.addScaleHandle([1, 1], center)
        if sideScalers:
            self.addScaleHandle([1, 0.5], [center[0], 0.5])
            self.addScaleHandle([0.5, 1], [0.5, center[1]])

    # this allows compatibility with LinearRegions:
    def setHoverBrush(self, *br, **args):
        pass

class DragViewBox(pg.ViewBox):
    # A normal ViewBox, but with ability to drag the segments
    # and also processes keypress events
    sigMouseDragged = QtCore.Signal(object,object,object)
    keyPressed = QtCore.Signal(int)

    def __init__(self, enableDrag, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.enableDrag = enableDrag

    def mouseDragEvent(self, ev):
        if self.enableDrag:
            ## if axis is specified, event will only affect that axis.
            ev.accept()
            if self.state['mouseMode'] != pg.ViewBox.RectMode or ev.button() == QtCore.Qt.RightButton:
                ev.ignore()

            if ev.isFinish():  ## This is the final move in the drag; draw the actual box
                self.rbScaleBox.hide()
                self.sigMouseDragged.emit(ev.buttonDownScenePos(ev.button()),ev.scenePos(),ev.screenPos())
            else:
                ## update shape of scale box
                self.updateScaleBox(ev.buttonDownPos(), ev.pos())
        else:
            pass

    def keyPressEvent(self,ev):
        # This catches the keypresses and sends out a signal
        #self.emit(SIGNAL("keyPressed"),ev)
        super(DragViewBox, self).keyPressEvent(ev)
        self.keyPressed.emit(ev.key())

class ChildInfoViewBox(pg.ViewBox):
    # Normal ViewBox, but with ability to pass a message back from a child
    sigChildMessage = QtCore.Signal(object)

    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)

    def resend(self,x):
        self.sigChildMessage.emit(x)

class PicButton(QAbstractButton):
    # Class for HumanClassify dialogs to put spectrograms on buttons
    def __init__(self, index, im1, im2, parent=None):
        super(PicButton, self).__init__(parent)
        self.index = index
        self.im1 = im1
        self.im2 = im2
        self.buttonClicked = False
        self.clicked.connect(self.changePic)

    def paintEvent(self, event):
        im = self.im2 if self.buttonClicked else self.im1

        if type(event) is not bool:
            painter = QPainter(self)
            painter.drawImage(event.rect(), im)

    def sizeHint(self):
        return self.im1.size()

    def changePic(self,event):
        self.buttonClicked = not(self.buttonClicked)
        self.paintEvent(event)
        self.update()

class ClickableRectItem(QtGui.QGraphicsRectItem):
    # QGraphicsItem doesn't include signals, hence this mess
    def __init__(self, *args, **kwds):
        QtGui.QGraphicsRectItem.__init__(self, *args, **kwds)

    def mousePressEvent(self, ev):
        super(ClickableRectItem, self).mousePressEvent(ev)
        self.parentWidget().resend(self.mapRectToParent(self.boundingRect()).x())

class ControllableAudio(QAudioOutput):
    # This links all the PyQt5 audio playback things -
    # QAudioOutput, QFile, and input from main interfaces
    format = QAudioFormat()
    format.setChannelCount(2)
    format.setSampleRate(48000)
    format.setSampleSize(16)
    format.setCodec("audio/pcm")
    format.setByteOrder(QAudioFormat.LittleEndian)
    format.setSampleType(QAudioFormat.SignedInt)

    def __init__(self):
        super(ControllableAudio, self).__init__(self.format)
        # on this notify, move slider (connected in main file)
        self.setNotifyInterval(20)
        self.stateChanged.connect(self.endListener)
        self.soundFile = QFile()
        self.tempin = QBuffer()
        self.setBufferSize(1000000)
        self.startpos = 0

    def load(self, soundFileName):
        if self.soundFile.isOpen():
            self.soundFile.close()
        self.startpos = 0

        self.soundFile.setFileName(soundFileName)
        try:
            self.soundFile.open(QIODevice.ReadOnly)
        except Exception as e:
            print("ERROR opening file: %s" % e)

    def isPlaying(self):
        return(self.state() == QAudio.ActiveState)

    def endListener(self):
        if self.state() == QAudio.IdleState:
            # give some time for GUI to catch up and stop
            sleep(0.2)
            self.notify.emit()
            self.stop()

    def pressedPlay(self):
        sleep(0.1)
        # save starting position in bytes
        self.startpos = self.soundFile.pos()
        self.start(self.soundFile)

    def pressedStop(self):
        self.stop()
        if self.tempin.isOpen():
            self.tempin.close()
        self.soundFile.seek(self.startpos)

    def filterBand(self, start, stop, lo, hi, audiodata, sp):
        # takes start-end in ms
        start = start * self.format.sampleRate() // 1000
        stop = stop * self.format.sampleRate() // 1000
        segment = audiodata[start:stop]
        segment = sp.bandpassFilter(segment, lo, hi)
        # segment = self.sp.ButterworthBandpass(segment, self.sampleRate, bottom, top,order=5)
        self.loadArray(segment)

    def loadArray(self, audiodata):
        # loads an array from memory into an audio buffer

        audiodata = audiodata.astype('int16') # 16 corresponds to sampwidth=2
        # double mono sound to get two channels - simplifies reading
        audiodata = np.column_stack((audiodata, audiodata))

        # write filtered output to a BytesIO buffer
        self.tempout = io.BytesIO()
        wavio.write(self.tempout, audiodata, self.format.sampleRate(), scale='dtype-limits', sampwidth=2)

        # copy BytesIO@write to QBuffer@read for playing
        self.temparr = QByteArray(self.tempout.getvalue())
        # self.tempout.close()
        if self.tempin.isOpen():
            self.tempin.close()
        self.tempin.setBuffer(self.temparr)
        self.tempin.open(QIODevice.ReadOnly)

        sleep(0.1)
        self.start(self.tempin)

    def seekToMs(self, ms):
        # note: important to specify format correctly!
        self.soundFile.seek(self.format.bytesForDuration(ms*1000))

    def applyVolSlider(self, value):
        # passes UI volume nonlinearly
        # value = QAudio.convertVolume(value / 100, QAudio.LogarithmicVolumeScale, QAudio.LinearVolumeScale)
        value = (math.exp(value/50)-1)/(math.exp(2)-1)
        self.setVolume(value)

class FlowLayout(QtGui.QLayout):
    # This is the flow layout which lays out a set of spectrogram pictures on buttons (for HumanClassify2) as
    # nicely as possible
    # From https://gist.github.com/Cysu/7461066
    def __init__(self, parent=None, margin=0, spacing=-1):
        super(FlowLayout, self).__init__(parent)

        if parent is not None:
            self.setMargin(margin)

        self.setSpacing(spacing)

        self.itemList = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList[index]

        return None

    def takeAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList.pop(index)

        return None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self._doLayout(QtCore.QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self._doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()

        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())

        size += QtCore.QSize(2 * self.margin(), 2 * self.margin())
        return size

    def _doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.spacing() + wid.style().layoutSpacing(
                QtGui.QSizePolicy.PushButton,
                QtGui.QSizePolicy.PushButton,
                QtCore.Qt.Horizontal)

            spaceY = self.spacing() + wid.style().layoutSpacing(
                QtGui.QSizePolicy.PushButton,
                QtGui.QSizePolicy.PushButton,
                QtCore.Qt.Vertical)

            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(
                    QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()


#+++++++++++++++++++++++++++++++
# Helper functions
def splitFile5mins(self, name):
    # Nirosha wants to split files that are long (15 mins) into 5 min segments
    # Could be used when loading long files :)
    try:
        self.audiodata, self.sampleRate = lr.load(name,sr=None)
    except:
        print("Error: try another file")
    nsamples = np.shape(self.audiodata)[0]
    lengthwanted = self.sampleRate * 60 * 5
    count = 0
    while (count + 1) * lengthwanted < nsamples:
        data = self.audiodata[count * lengthwanted:(count + 1) * lengthwanted]
        filename = name[:-4] + '_' +str(count) + name[-4:]
        lr.output.write_wav(filename, data, self.sampleRate)
        count += 1
    data = self.audiodata[(count) * lengthwanted:]
    filename = name[:-4] + '_' + str((count)) + name[-4:]
    lr.output.write_wav(filename,data,self.sampleRate)

    # ###########
    # #just testing
    # # fName='Sound Files/Kiwi/test/Tier1/CL78_BIRM_141120_212934'
    # fName='Sound Files/Kiwi/test/Tier1/BV21_BIRD_141206_234353'
    # # fName='Sound Files/Kiwi/test/Ponui/kiwi-test2'
    # ws1=WaveletSegment()
    # ws1.loadData(fName, trainTest=False)
    # det = ws1.waveletSegment_test(fName=None, data=ws1.data, sampleRate=ws1.sampleRate, species=ws1.species,
    #                                          trainTest=False)
    # # det = np.ones(900)
    # if sum(det)>0:
    #     post=postProcess(ws1.data, ws1.sampleRate, det)
    #     # post.detectClicks()
    #     post.eRatioConfd()

