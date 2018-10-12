# Version 0.2 10/7/17
# Author: Stephen Marsland

# Support classes for the AviaNZ program
# Mostly subclassed from pyqtgraph
#     from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QAbstractButton
from PyQt5.QtCore import QTime, QFile, QIODevice, QBuffer, QByteArray
from PyQt5.QtMultimedia import QAudio, QAudioOutput
from PyQt5.QtGui import QPainter

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
import time

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

    def __init__(self,audioData=None, sampleRate=0, spInfo={}, df=False, wavelet='dmey2'):
        self.audioData=audioData
        self.sampleRate=sampleRate
        self.spInfo=spInfo
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
        if self.spInfo == {}:
            fs = 8000
            f1 = None
            f2 = None
        else:
            f1 = self.spInfo['FreqRange'][0]
            f2 = self.spInfo['FreqRange'][1]
            fs = self.spInfo['SampleRate']

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

        if f1 and f2:
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
    """

    def __init__(self,audioData=None, sampleRate=0, segments=[], spInfo={}):
        self.audioData = audioData
        self.sampleRate = sampleRate
        self.segments = segments
        if spInfo != {}:
            self.minLen = spInfo['TimeRange'][0]
            self.F0 = spInfo['F0Range']
        else:
            self.minLen = 0
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
                data = self.audioData[int(seg[0]*self.sampleRate):int(seg[1]*self.sampleRate)]

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
                    data = self.audioData[int(seg[0]*self.sampleRate):int(seg[1]*self.sampleRate)]
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

    def fundamentalFrq(self):
        '''
        Check for fundamental frequency of the segments, discard the segments that does not indicate the species.
        '''
        newSegments = copy.deepcopy(self.segments)
        for seg in self.segments:
            if seg[0] == -1:
                continue
            else:
                # read the sound segment and check fundamental frq.
                data = self.audioData[int(seg[0]*self.sampleRate):int(seg[1]*self.sampleRate)]

                # denoise before fundamental frq. extraction
                sc = preProcess(audioData=data, sampleRate=self.sampleRate, spInfo={}, df=True)  # species left empty to avoid bandpass filter
                data, sampleRate = sc.denoise_filter(level=10)

                sp = SignalProc.SignalProc([], 0, 512, 256)
                sgRaw = sp.spectrogram(data, 512, 256, mean_normalise=True, onesided=True, multitaper=False)
                segment = Segment.Segment(data, sgRaw, sp, sampleRate, 512, 256)
                pitch, y, minfreq, W = segment.yin(minfreq=100)
                ind = np.squeeze(np.where(pitch > minfreq))
                pitch = pitch[ind]
                if pitch.size == 0:
                    print('segment ', seg, ' *++ no fundamental freq detected, could be faded call or noise')
                    # newSegments.remove(seg) # for now keep it
                    continue    # continue to the next seg
                ind = ind * W / 512
                x = (pitch * 2. / sampleRate * np.shape(sgRaw)[1]).astype('int')
                from scipy.signal import medfilt
                x = medfilt(pitch, 15)
                if ind.size < 2:
                    if (pitch > self.F0[0]) and (pitch < self.F0[1]):
                        print("match with F0 of bird, ", pitch)
                        continue    # print file, 'segment ', seg, round(pitch), ' *##kiwi found'
                    else:
                        print('segment ', seg, round(pitch), ' *-- fundamental freq is out of range, could be noise')
                        newSegments.remove(seg)
                else:
                    if (np.mean(pitch) > self.F0[0]) and (np.mean(pitch) < self.F0[1]):
                        # print file, 'segment ', seg, round(np.mean(pitch)), ' *## kiwi found '
                        continue
                    else:
                        print('segment ', seg, round(np.mean(pitch)),
                              ' *-- fundamental freq is out of range, could be noise')
                        newSegments.remove(seg)
                        continue
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

        if sg is None:
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

        Inputs
            segments:   detected segments in form of [[s1,e1], [s2,e2],...]
                        OR in format [[s1, e1, fs1, fe1, sp1], [s2, e2, fs2, fe2, sp2], ...]
                segmentstoCheck     : segments without confidence in form of [[s1,e1], [s2,e2],...]
                confirmedSegments   : segments with confidence
            species:    default species. e.g. 'Kiwi'. Default is 'all'
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

    def __init__(self, segments, confirmedSegments=[], segmentstoCheck=[], species=["Don't Know"], startTime=0, dirName='', filename='',datalength=0,sampleRate=0, method="Default", resolution=1, trainTest=False, withConf=False, seg_pos=[], operator='', reviewer='', minLen=0, numpages=1, batch=False):

        self.species=species
        # convert 2-col lists to 5-col lists, if needed
        self.segments = self.correctSegFormat(segments, [])
        self.confirmedSegments = self.correctSegFormat(confirmedSegments, species)
        if species==[]:
            self.segmentstoCheck = self.correctSegFormat(segmentstoCheck, ["Don't Know"])
        else:
            self.segmentstoCheck = self.correctSegFormat(segmentstoCheck, [species[0] + "?"])

        self.numpages=numpages
        self.startTime=startTime
        self.dirName=dirName
        self.filename=filename
        self.datalength=datalength
        self.sampleRate=sampleRate
        self.method=method
        self.resolution = resolution
        self.trainTest = trainTest
        self.withConf=withConf  # todo: remove
        self.seg_pos=seg_pos #segmentstoCheck
        self.operator = operator
        self.reviewer = reviewer
        self.minLen = minLen
        self.batch = batch

    def correctSegFormat(self, seglist, species):
        # Checks and if needed corrects 2-col segments to 5-col segments.
        # segments can be provided as confirmed/toCheck lists,
        # while everything from segments list is exported as-is.
        if len(seglist)>0:
            if len(seglist[0])==2:
                print("using old format segment list")
                # convert to new format
                for seg in seglist:
                    seg.append(0)
                    seg.append(0)
                    seg.append(species)
                return(seglist)
            elif len(seglist[0])==5:
                print("using new format segment list")
                return(seglist)
            else:
                print("ERROR: incorrect segment format")
                return
        else:
            return([])

    def makeNewWorkbook(self, species):
        self.wb = Workbook()
        self.wb.create_sheet(title='Time Stamps', index=1)
        self.wb.create_sheet(title='Presence Absence', index=2)
        self.wb.create_sheet(title='Per second', index=3)

        ws = self.wb['Time Stamps']
        ws.cell(row=1, column=1, value="File Name")
        ws.cell(row=1, column=2, value="start (hh:mm:ss)")
        ws.cell(row=1, column=3, value="end (hh:mm:ss)")
        ws.cell(row=1, column=4, value="min freq., Hz")
        ws.cell(row=1, column=5, value="max freq., Hz")
        if species=="All species":
            ws.cell(row=1, column=6, value="species")

        # Second sheet
        ws = self.wb['Presence Absence']
        ws.cell(row=1, column=1, value="File Name")
        ws.cell(row=1, column=2, value="Presence/Absence")

        # Third sheet
        ws = self.wb['Per second']
        ws.cell(row=1, column=1, value="File Name_Page")
        ws.cell(row=1, column=2, value="Presence=1, Absence=0")

        # Hack to delete original sheet
        del self.wb['Sheet']
        return self.wb

    def excel(self):
        """ This saves the detections in three different formats: time stamps, presence/absence, and per second presence/absence in an excel workbook. It makes the workbook if necessary.
        Saves each species into a separate workbook,
        + an extra workbook for all species (to function as a readable segment printout).
        """
        # identify all unique species
        speciesList = set()
        for sp in self.species:
            speciesList.add(sp)
        for seg in self.segments:
            for birdName in seg[4]:
                segmentSpecies = birdName
                if birdName.endswith('?'):
                    segmentSpecies = segmentSpecies[:-1]
                speciesList.add(segmentSpecies)
        speciesList.add("All species")
        print("The following species were detected for export:")
        print(speciesList)

        def writeToExcelp1(segments):
            ws = wb['Time Stamps']
            r = ws.max_row + 1
            # Print the filename
            ws.cell(row=r, column=1, value=str(relfname))
            # Loop over the segments
            for seg in segments:
                # if int(seg[1]-seg[0]) < self.minLen: # skip very short segments
                #     continue
                # deleting short segments already done during post processing
                ws.cell(row=r, column=2, value=str(QTime(0,0,0).addSecs(seg[0]+self.startTime).toString('hh:mm:ss')))
                ws.cell(row=r, column=3, value=str(QTime(0,0,0).addSecs(seg[1]+self.startTime).toString('hh:mm:ss')))
                if seg[3]!=0:
                    ws.cell(row=r, column=4, value=int(seg[2]))
                    ws.cell(row=r, column=5, value=int(seg[3]))
                if species=="All species":
                    ws.cell(row=r, column=6, value=", ".join(seg[4]))
                r += 1

        def writeToExcelp2(segments):
            ws = wb['Presence Absence']
            r = ws.max_row + 1
            ws.cell(row=r, column=1, value=str(relfname))
            ws.cell(row=r, column=2, value='_')
            if len(segments)>0:
                # if seg[1]-seg[0] > self.minLen: # skip very short segments
                ws.cell(row=r, column=2, value='Yes')
                # break
            else:
                ws.cell(row=r, column=2, value='No')

        def writeToExcelp3(detected, starttime=0):
            # todo: use minLen
            need_reset = False
            if self.resolution > math.ceil(float(self.datalength) / self.sampleRate):
                resolution_before = self.resolution
                need_reset = True
                self.resolution = int(math.ceil(float(self.datalength) / self.sampleRate))
            ws = wb['Per second']
            r = ws.max_row + 1
            ws.cell(row=r, column=1, value= str(self.resolution) + ' secs resolution')
            ft = Font(color=colors.DARKYELLOW)
            ws.cell(row=r, column=1).font=ft
            c = 2
            for i in range(starttime,starttime+len(detected), self.resolution):
                endtime = min(i+self.resolution, int(math.ceil(self.datalength * self.numpages / self.sampleRate)))
                ws.cell(row=r, column=c, value=str(i) + '-' + str(endtime))
                ws.cell(row=r, column=c).font = ft
                c += 1
            r += 1
            pagesize = int(starttime/self.datalength * self.sampleRate)
            ws.cell(row=r, column=1, value=str(relfname)+ '_p' + str(pagesize))
            c = 2
            for i in range(0, len(detected), self.resolution):
                j=1 if np.sum(detected[i:i+self.resolution])>0 else 0
                ws.cell(row=r, column=c, value=j)
                c += 1
            # reset resolution
            if need_reset:
                self.resolution = resolution_before

        # now, generate the actual files, SEPARATELY FOR EACH SPECIES:
        for species in speciesList:
            print("Exporting species %s" % species)
            # setup output files:
            # if an Excel exists, append (so multiple files go into one worksheet)
            # if not, create new

            if self.withConf:
                if self.batch:
                    self.eFile = self.dirName + '/DetectionSummary_withConf_' + species + '.xlsx'
                else:
                    self.eFile = self.filename + '_withConf_' + species + '.xlsx'
            else:
                if self.batch:
                    self.eFile = self.dirName + '/DetectionSummary_' + species + '.xlsx'
                else:
                    self.eFile = self.filename + '_' + species + '.xlsx'

            if os.path.isfile(self.eFile):
                try:
                    wb = load_workbook(str(self.eFile))
                except:
                    print("Unable to open file")  # Does not exist OR no read permissions
                    return
            else:
                wb = self.makeNewWorkbook(species)
            relfname = os.path.relpath(str(self.filename), str(self.dirName))
            # extract SINGLE-SPECIES ONLY segments,
            # incl. potential assignments ('Kiwi?').
            # if species=="All", take ALL segments.
            segmentsWPossible = []
            for seg in self.segments + self.confirmedSegments + self.segmentstoCheck:
                if len(seg) == 2:
                    seg.append(0)
                    seg.append(0)
                    seg.append(species)
                if species in seg[4] or species+'?' in seg[4] or species == "All species":
                    segmentsWPossible.append(seg)
            # if len(segmentsWPossible)==0:
            #     print("Warning: no segments found for species %s" % species)
            #     continue

            # export segments
            writeToExcelp1(segmentsWPossible)
            # export presence/absence
            writeToExcelp2(segmentsWPossible)

            # Generate per second binary output
            n = math.ceil(float(self.datalength) / self.sampleRate)
            for p in range(0, self.numpages):
                detected = np.zeros(n)
                print("exporting page %d" % p)
                for seg in segmentsWPossible:
                    for t in range(n):
                        truet = t + p*n
                        if math.floor(seg[0]) <= truet and truet < math.ceil(seg[1]):
                            detected[t] = 1
                writeToExcelp3(detected, p*n)

            # Save the file
            wb.save(str(self.eFile))

    def saveAnnotation(self):
        # Save annotations - batch processing
        annotation = []
        annotation.append([-1, str(QTime(0,0,0).addSecs(self.startTime).toString('hh:mm:ss')), self.operator, self.reviewer, -1])
        for seg in self.confirmedSegments:
            annotation.append([float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3]), seg[4]])
        for seg in self.segmentstoCheck:
            annotation.append([float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3]), seg[4]])
        for seg in self.segments:
            annotation.append([float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3]), seg[4]])

        if isinstance(self.filename, str):
            file = open(self.filename + '.data', 'w')
        else:
            file = open(str(self.filename) + '.data', 'w')

        json.dump(annotation, file)
        file.write("\n")

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

class FixedLineROI(pg.LineSegmentROI):
    def clearHandles(self):
        self.scene().removeItem(self.handles[0]['item'])
        self.scene().removeItem(self.handles[1]['item'])
        #while len(self.handles) > 0:
        #    self.removeHandle(self.handles[0]['item'])

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

def mouseDragEventFlexible(self, ev):
    if ev.button() == self.rois[0].parent.MouseDrawingButton:
        return
    ev.accept()
    
    ## Inform ROIs that a drag is happening 
    ##  note: the ROI is informed that the handle has moved using ROI.movePoint
    ##  this is for other (more nefarious) purposes.
    #for r in self.roi:
        #r[0].pointDragEvent(r[1], ev)
        
    if ev.isFinish():
        if self.isMoving:
            for r in self.rois:
                r.stateChangeFinished()
        self.isMoving = False
    elif ev.isStart():
        for r in self.rois:
            r.handleMoveStarted()
        self.isMoving = True
        self.startPos = self.scenePos()
        self.cursorOffset = self.scenePos() - ev.buttonDownScenePos()
        
    if self.isMoving:  ## note: isMoving may become False in mid-drag due to right-click.
        pos = ev.scenePos() + self.cursorOffset
        self.movePoint(pos, ev.modifiers(), finish=False)

def mouseDragEventFlexibleLine(self, ev):
    if self.movable and ev.button() != self.btn:
        if ev.isStart():
            self.moving = True
            self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
            self.startPosition = self.pos()
        ev.accept()

        if not self.moving:
            return

        self.setPos(self.cursorOffset + self.mapToParent(ev.pos()))
        self.sigDragged.emit(self)
        if ev.isFinish():
            self.moving = False
            self.sigPositionChangeFinished.emit(self)

class ShadedRectROI(ShadedROI):
    # A rectangular ROI that it shaded, for marking segments
    def __init__(self, pos, size, centered=False, sideScalers=False, parent=None, **args):
        #QtGui.QGraphicsRectItem.__init__(self, 0, 0, size[0], size[1])
        pg.ROI.__init__(self, pos, size, **args)
        self.parent = parent
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

    def mouseDragEvent(self, ev):
        if ev.isStart():
            if ev.button() != self.parent.MouseDrawingButton:
                self.setSelected(True)
                if self.translatable:
                    self.isMoving = True
                    self.preMoveState = self.getState()
                    self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                    self.sigRegionChangeStarted.emit(self)
                    ev.accept()
                else:
                    ev.ignore()

        elif ev.isFinish():
            if self.translatable:
                if self.isMoving:
                    self.stateChangeFinished()
                self.isMoving = False
            return

        if self.translatable and self.isMoving and ev.buttons() != self.parent.MouseDrawingButton:
            snap = True if (ev.modifiers() & QtCore.Qt.ControlModifier) else None
            newPos = self.mapToParent(ev.pos()) + self.cursorOffset
            self.translate(newPos - self.pos(), snap=snap, finish=False)

pg.graphicsItems.ROI.Handle.mouseDragEvent = mouseDragEventFlexible
pg.graphicsItems.InfiniteLine.InfiniteLine.mouseDragEvent = mouseDragEventFlexibleLine

class LinearRegionItem2(pg.LinearRegionItem):
    def __init__(self, parent, *args, **kwds):
        pg.LinearRegionItem.__init__(self, *args, **kwds)
        self.parent = parent
        self.lines[0].btn = self.parent.MouseDrawingButton
        self.lines[1].btn = self.parent.MouseDrawingButton

    def mouseDragEvent(self, ev):
        if not self.movable or ev.button()==self.parent.MouseDrawingButton:
            return
        ev.accept()
        
        if ev.isStart():
            bdp = ev.buttonDownPos()
            self.cursorOffsets = [l.pos() - bdp for l in self.lines]
            self.startPositions = [l.pos() for l in self.lines]
            self.moving = True
            
        if not self.moving:
            return
            
        self.lines[0].blockSignals(True)  # only want to update once
        for i, l in enumerate(self.lines):
            l.setPos(self.cursorOffsets[i] + ev.pos())
        self.lines[0].blockSignals(False)
        self.prepareGeometryChange()
        
        if ev.isFinish():
            self.moving = False
            self.sigRegionChangeFinished.emit(self)
        else:
            self.sigRegionChanged.emit(self)

class DragViewBox(pg.ViewBox):
    # A normal ViewBox, but with ability to drag the segments
    # and also processes keypress events
    sigMouseDragged = QtCore.Signal(object,object,object)
    keyPressed = QtCore.Signal(int)

    def __init__(self, parent, enableDrag, thisIsAmpl, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.enableDrag = enableDrag
        self.parent = parent
        self.thisIsAmpl = thisIsAmpl

    def mouseDragEvent(self, ev):
        print("uncaptured drag event")
        # if self.enableDrag:
        #     ## if axis is specified, event will only affect that axis.
        #     ev.accept()
        #     if self.state['mouseMode'] != pg.ViewBox.RectMode or ev.button() == QtCore.Qt.RightButton:
        #         ev.ignore()

        #     if ev.isFinish():  ## This is the final move in the drag; draw the actual box
        #         print("dragging done")
        #         self.rbScaleBox.hide()
        #         self.sigMouseDragged.emit(ev.buttonDownScenePos(ev.button()),ev.scenePos(),ev.screenPos())
        #     else:
        #         ## update shape of scale box
        #         self.updateScaleBox(ev.buttonDownPos(), ev.pos())
        # else:
        #     pass

    def mousePressEvent(self, ev):
        if self.enableDrag and ev.button() == self.parent.MouseDrawingButton:
            if self.thisIsAmpl:
                self.parent.mouseClicked_ampl(ev)
            else:
                self.parent.mouseClicked_spec(ev)
            ev.accept()
        else:
            ev.ignore()

    def mouseReleaseEvent(self, ev):
        if self.enableDrag and ev.button() == self.parent.MouseDrawingButton:
            if self.thisIsAmpl:
                self.parent.mouseClicked_ampl(ev)
            else:
                self.parent.mouseClicked_spec(ev)
            ev.accept()
        else:
            ev.ignore()

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
    # Also includes playback capability.
    def __init__(self, index, im1, im2, audiodata, format, duration, parent=None):
        super(PicButton, self).__init__(parent)
        self.index = index
        self.im1 = im1
        self.im2 = im2
        self.buttonClicked = False
        self.clicked.connect(self.changePic)

        # playback things
        self.media_obj = ControllableAudio(format)
        self.media_obj.notify.connect(self.endListener)
        self.audiodata = audiodata
        self.duration = duration * 1000 # in ms

        self.playButton = QtGui.QToolButton(self)
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.playImage)
        self.playButton.hide()

    def paintEvent(self, event):
        im = self.im2 if self.buttonClicked else self.im1

        if type(event) is not bool:
            painter = QPainter(self)
            painter.drawImage(event.rect(), im)

    def enterEvent(self, QEvent):
        self.playButton.show()

    def leaveEvent(self, QEvent):
        if not self.media_obj.isPlaying():
            self.playButton.hide()

    def playImage(self):
        if self.media_obj.isPlaying():
            self.stopPlayback()
        else:
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
            self.media_obj.loadArray(self.audiodata)

    def endListener(self):
        time = self.media_obj.elapsedUSecs() // 1000
        if time > self.duration:
            self.stopPlayback()

    def stopPlayback(self):
        self.media_obj.pressedStop()
        self.playButton.hide()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))

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

    def __init__(self, format):
        super(ControllableAudio, self).__init__(format)
        # on this notify, move slider (connected in main file)
        self.setNotifyInterval(30)
        self.stateChanged.connect(self.endListener)
        self.tempin = QBuffer()
        self.startpos = 0
        self.timeoffset = 0
        self.keepSlider = False
        self.format = format
        # set buffer size to 100 ms
        self.setBufferSize(int(self.format.sampleSize() * self.format.sampleRate()/10 * self.format.channelCount()))

    def isPlaying(self):
        return(self.state() == QAudio.ActiveState)

    def endListener(self):
        # this should only be called if there's some misalignment between GUI and Audio
        if self.state() == QAudio.IdleState:
            # give some time for GUI to catch up and stop
            while(self.state() != QAudio.StoppedState):
                sleep(0.03)
                self.notify.emit()
            self.keepSlider=False
            self.stop()

    def pressedPlay(self, resetPause=False, start=0, stop=0, audiodata=None):
        if not resetPause and self.state() == QAudio.SuspendedState:
            print("resuming at: %d" % self.pauseoffset)
            self.sttime = time.time() - self.pauseoffset/1000
            self.resume()
        else:
            if not self.keepSlider or resetPause:
                self.pressedStop()

            print("starting at: %d" % self.tempin.pos())
            sleep(0.2)
            # in case bar was moved under pause, we need this:
            pos = self.tempin.pos() # bytes
            pos = self.format.durationForBytes(pos) / 1000 # convert to ms
            pos = pos + start
            print("pos: %d start: %d stop %d" %(pos, start, stop))
            self.filterSeg(pos, stop, audiodata)

    def pressedPause(self):
        self.keepSlider=True # a flag to avoid jumping the slider back to 0
        pos = self.tempin.pos() # bytes
        pos = self.format.durationForBytes(pos) / 1000 # convert to ms
        # store offset, relative to the start of played segment
        self.pauseoffset = pos + self.timeoffset
        self.suspend()

    def pressedStop(self):
        # stop and reset to window/segment start
        self.keepSlider=False
        self.stop()
        if self.tempin.isOpen():
            self.tempin.close()

    def filterBand(self, start, stop, low, high, audiodata, sp):
        # takes start-end in ms, relative to file start
        self.timeoffset = max(0, start)
        start = max(0, start * self.format.sampleRate() // 1000)
        stop = min(stop * self.format.sampleRate() // 1000, len(audiodata))
        segment = audiodata[int(start):int(stop)]
        segment = sp.bandpassFilter(segment,sampleRate=None, start=low, end=high)
        # segment = self.sp.ButterworthBandpass(segment, self.sampleRate, bottom, top,order=5)
        self.loadArray(segment)

    def filterSeg(self, start, stop, audiodata):
        # takes start-end in ms
        self.timeoffset = max(0, start)
        start = max(0, int(start * self.format.sampleRate() // 1000))
        stop = min(int(stop * self.format.sampleRate() // 1000), len(audiodata))
        segment = audiodata[start:stop]
        self.loadArray(segment)

    def loadArray(self, audiodata):
        # loads an array from memory into an audio buffer
        if self.format.sampleSize() == 16:
            audiodata = audiodata.astype('int16') # 16 corresponds to sampwidth=2
        elif self.format.sampleSize() == 32:
            audiodata = audiodata.astype('int32')
        elif self.format.sampleSize() == 24:
            audiodata = audiodata.astype('int32')
            print("Warning: 24-bit sample playback currently not supported")
        else:
            print("ERROR: sampleSize %d not supported" % self.format.sampleSize())
            return
        # double mono sound to get two channels - simplifies reading
        if self.format.channelCount()==2:
            audiodata = np.column_stack((audiodata, audiodata))

        # write filtered output to a BytesIO buffer
        self.tempout = io.BytesIO()
        wavio.write(self.tempout, audiodata, self.format.sampleRate(), scale='none', sampwidth=self.format.sampleSize() // 8)

        # copy BytesIO@write to QBuffer@read for playing
        self.temparr = QByteArray(self.tempout.getvalue()[44:])
        # self.tempout.close()
        if self.tempin.isOpen():
            self.tempin.close()
        self.tempin.setBuffer(self.temparr)
        self.tempin.open(QIODevice.ReadOnly)

        # actual timer is launched here, with time offset set asynchronously
        sleep(0.2)
        self.sttime = time.time() - self.timeoffset/1000
        self.start(self.tempin)

    def seekToMs(self, ms, start):
        print("seeking to %d ms" % ms)
        # start is an offset for the current view start, as it is position 0 in extracted file
        self.reset()
        self.tempin.seek(self.format.bytesForDuration((ms-start)*1000))
        self.timeoffset = ms

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

        self.margin = margin

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

    # def minimumSize(self):
    #     size = QtCore.QSize()
    #
    #     for item in self.itemList:
    #         size = size.expandedTo(item.minimumSize())
    #
    #     size += QtCore.QSize(2 * self.margin(), 2 * self.margin())
    #     return size

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

class Log(object):
    """ Used for logging info during batch processing.
        Stores most recent analysis for each species, to stay in sync w/ data files.
        Arguments:
        1. path to log file
        2. species
        3. list of other settings of the current analysis

        LOG FORMAT, for each analysis:
        #freetext line
        species
        settings line
        files, multiple lines
    """

    def __init__(self, path, species, settings):
        # in order to append, the previous log must:
        # 1. exist
        # 2. be writeable
        # 3. match current analysis
        # On init, we parse the existing log to see if appending is possible.
        # Actual append/create happens later.
        self.possibleAppend = False
        self.file = path
        self.species = species
        self.settings = ','.join(map(str, settings))
        self.oldAnalyses = []
        self.filesDone = []
        self.currentHeader = ""
        allans = []

        # now, check if the specified log can be resumed:
        if os.path.isfile(path):
            try:
                f = open(path, 'r+')
                print("Found log file at %s" % path)

                lines = [line.rstrip('\n') for line in f]
                f.close()
                lstart = 0
                lend = 1
                # parse to separate each analysis into
                # [freetext, species, settings, [files]]
                # (basically I'm parsing txt into json because I'm dumb)
                while lend<len(lines):
                    print(lines[lend])
                    if lines[lend][0] == "#":
                        allans.append([lines[lstart], lines[lstart+1], lines[lstart+2],
                                        lines[lstart+3 : lend]])
                        lstart = lend
                    lend += 1
                allans.append([lines[lstart], lines[lstart+1], lines[lstart+2],
                                lines[lstart+3 : lend]])

                # parse the log thusly:
                # if current species analysis found, store parameters
                # and compare to check if it can be resumed.
                # store all other analyses for re-printing.
                for a in allans:
                    print(a)
                    if a[1]==self.species:
                        print("resumable analysis found")
                        # do not reprint this in log
                        if a[2]==self.settings:
                            self.currentHeader = a[0]
                            # (a1 and a2 match species & settings anyway)
                            self.filesDone = a[3]
                            self.possibleAppend = True
                    else:
                        # store this for re-printing to log
                        self.oldAnalyses.append(a)

            except IOError:
                # bad error: lacking permissions?
                print("ERROR: could not open log at %s" % path)

    def appendFile(self, filename):
        print('appending %s to log' % filename)
        # attach file path to end of log
        self.file.write(filename)
        self.file.write("\n")
        self.file.flush()

    def appendHeader(self, header, species, settings):
        if header is None:
            header = "#Analysis started on " + time.strftime("%Y %m %d, %H:%M:%S") + ":"
        self.file.write(header)
        self.file.write("\n")
        self.file.write(species)
        self.file.write("\n")
        if type(settings) is list:
            settings = ','.join(settings)
        self.file.write(settings)
        self.file.write("\n")
        self.file.flush()

    def reprintOld(self):
        # push everything from oldAnalyses to log
        # To be called once starting a new log is confirmed
        for a in self.oldAnalyses:
            self.appendHeader(a[0], a[1], a[2])
            for f in a[3]:
                self.appendFile(f)

