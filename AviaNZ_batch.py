
# AviaNZ_batch.py
#
# This is the proceesing class for the batch AviaNZ interface
# Version 1.5 05/08/19
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis

#    AviaNZ birdsong analysis program
#    Copyright (C) 2017--2019

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
import os, re, platform, fnmatch, sys

from PyQt5.QtGui import QIcon, QPixmap, QApplication
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QLabel, QPlainTextEdit, QPushButton, QTimeEdit, QSpinBox, QListWidget, QDesktopWidget, QApplication, QComboBox, QLineEdit, QSlider, QListWidgetItem
from PyQt5.QtMultimedia import QAudioFormat
from PyQt5.QtCore import Qt, QDir

import wavio
import librosa
import numpy as np
import math
import statistics
from itertools import chain, repeat

from pyqtgraph.Qt import QtGui
from pyqtgraph.dockarea import *
import pyqtgraph as pg

import SignalProc
import Segment
import WaveletSegment
import SupportClasses
import Dialogs

import json, copy, time


class AviaNZ_batchProcess(QMainWindow):
    # Main class for batch processing

    def __init__(self, root=None, configdir='', minSegment=50):
        # Allow the user to browse a folder and push a button to process that folder to find a target species
        # and sets up the window.
        super(AviaNZ_batchProcess, self).__init__()
        self.root = root
        self.dirName=[]

        # read config and filters from user location
        self.configfile = os.path.join(configdir, "AviaNZconfig.txt")
        self.ConfigLoader = SupportClasses.ConfigLoader()
        self.config = self.ConfigLoader.config(self.configfile)
        self.saveConfig = True

        self.filtersDir = os.path.join(configdir, self.config['FiltersDir'])
        self.FilterDicts = self.ConfigLoader.filters(self.filtersDir)

        # Make the window and associated widgets
        QMainWindow.__init__(self, root)

        self.statusBar().showMessage("Processing file Current/Total")

        self.setWindowTitle('AviaNZ - Batch Processing')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.createMenu()
        self.createFrame()
        self.center()

    def createFrame(self):
        # Make the window and set its size
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.setFixedSize(870,550)

        # Make the docks
        self.d_detection = Dock("Automatic Detection",size=(600,550))
        self.d_files = Dock("File list", size=(270, 550))

        self.area.addDock(self.d_detection,'right')
        self.area.addDock(self.d_files, 'left')

        self.w_browse = QPushButton("  &Browse Folder")
        self.w_browse.setToolTip("Can select a folder with sub folders to process")
        self.w_browse.setFixedHeight(50)
        self.w_browse.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')
        self.w_dir = QPlainTextEdit()
        self.w_dir.setFixedHeight(50)
        self.w_dir.setPlainText('')
        self.w_dir.setToolTip("The folder being processed")
        self.d_detection.addWidget(self.w_dir,row=0,col=1,colspan=2)
        self.d_detection.addWidget(self.w_browse,row=0,col=0)

        self.w_speLabel1 = QLabel("  Select Species")
        self.d_detection.addWidget(self.w_speLabel1,row=1,col=0)
        self.w_spe1 = QComboBox()
        # read filter list, replace subsp marks with brackets
        spp = list(self.FilterDicts.keys())
        for sp in spp:
            ind = sp.find('>')
            if ind > -1:
                sp = sp[:ind] + ' (' + sp[ind+1:] + ')'
        spp.insert(0, "All species")
        self.w_spe1.addItems(spp)
        self.d_detection.addWidget(self.w_spe1,row=1,col=1,colspan=2)

        self.w_resLabel = QLabel("  Time Resolution in Excel Output (secs)")
        self.d_detection.addWidget(self.w_resLabel, row=2, col=0)
        self.w_res = QSpinBox()
        self.w_res.setRange(1,600)
        self.w_res.setSingleStep(5)
        self.w_res.setValue(60)
        self.d_detection.addWidget(self.w_res, row=2, col=1, colspan=2)

        self.w_timeWindow = QLabel("  Choose Time Window (from-to)")
        self.d_detection.addWidget(self.w_timeWindow, row=4, col=0)
        self.w_timeStart = QTimeEdit()
        self.w_timeStart.setDisplayFormat('hh:mm:ss')
        self.d_detection.addWidget(self.w_timeStart, row=4, col=1)
        self.w_timeEnd = QTimeEdit()
        self.w_timeEnd.setDisplayFormat('hh:mm:ss')
        self.d_detection.addWidget(self.w_timeEnd, row=4, col=2)

        self.w_processButton = QPushButton("&Process Folder")
        self.w_processButton.clicked.connect(self.detect)
        self.d_detection.addWidget(self.w_processButton,row=11,col=2)
        self.w_processButton.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')

        self.w_browse.clicked.connect(self.browse)

        self.w_files = pg.LayoutWidget()
        self.d_files.addWidget(self.w_files)
        self.w_files.addWidget(QLabel('View Only'), row=0, col=0)
        self.w_files.addWidget(QLabel('use Browse Folder to choose data for processing'), row=1, col=0)
        # self.w_files.addWidget(QLabel(''), row=2, col=0)
        # List to hold the list of files
        self.listFiles = QListWidget()
        self.listFiles.setMinimumWidth(150)
        self.listFiles.itemDoubleClicked.connect(self.listLoadFile)
        self.w_files.addWidget(self.listFiles, row=2, col=0)

        self.show()

    def createMenu(self):
        """ Create the basic menu.
        """

        helpMenu = self.menuBar().addMenu("&Help")
        helpMenu.addAction("Help", self.showHelp,"Ctrl+H")
        aboutMenu = self.menuBar().addMenu("&About")
        aboutMenu.addAction("About", self.showAbout,"Ctrl+A")
        aboutMenu = self.menuBar().addMenu("&Quit")
        aboutMenu.addAction("Quit", self.quitPro,"Ctrl+Q")

    def showAbout(self):
        """ Create the About Message Box. Text is set in SupportClasses.MessagePopup"""
        msg = SupportClasses.MessagePopup("a", "About", ".")
        msg.exec_()
        return

    def showHelp(self):
        """ Show the user manual (a pdf file)"""
        # TODO: manual is not distributed as pdf now
        import webbrowser
        # webbrowser.open_new(r'file://' + os.path.realpath('./Docs/AviaNZManual.pdf'))
        webbrowser.open_new(r'http://avianz.net/docs/AviaNZManual.pdf')

    def quitPro(self):
        """ quit program
        """
        QApplication.quit()

    def center(self):
        # geometry of the main window
        qr = self.frameGeometry()
        # center point of screen
        cp = QDesktopWidget().availableGeometry().center()
        # move rectangle's center point to screen's center point
        qr.moveCenter(cp)
        # top left of rectangle becomes top left of window centering it
        self.move(qr.topLeft())

    def cleanStatus(self):
        self.statusBar().showMessage("Processing file Current/Total")

    def browse(self):
        if self.dirName:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirName))
        else:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process')
        #print("Dir:", self.dirName)
        self.w_dir.setPlainText(self.dirName)
        self.w_dir.setReadOnly(True)
        self.fillFileList(self.dirName)


    # from memory_profiler import profile
    # fp = open('memory_profiler_wp.log', 'w+')
    # @profile(stream=fp)
    def detect(self):
        # check if folder was selected:
        if not self.dirName:
            msg = SupportClasses.MessagePopup("w", "Select Folder", "Please select a folder to process!")
            msg.exec_()
            return

        self.species=self.w_spe1.currentText()
        if self.species == "All species":
            self.method = "Default"
        else:
            self.speciesData = self.FilterDicts[self.species]
            # Defaulting to first subfilter for now
            self.speciesSubf = self.speciesData["Filters"][0]
            self.method = "Wavelets"

        # Parse the user-set time window to process
        timeWindow_s = self.w_timeStart.time().hour() * 3600 + self.w_timeStart.time().minute() * 60 + self.w_timeStart.time().second()
        timeWindow_e = self.w_timeEnd.time().hour() * 3600 + self.w_timeEnd.time().minute() * 60 + self.w_timeEnd.time().second()

        # LIST ALL WAV files that will be processed
        allwavs = []
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                if filename.endswith('.wav'):
                    allwavs.append(os.path.join(root, filename))
        total = len(allwavs)

        # LOG FILE is read here
        # note: important to log all analysis settings here
        self.log = SupportClasses.Log(os.path.join(self.dirName, 'LastAnalysisLog.txt'),
                                self.species, [self.method, self.w_res.value(), timeWindow_s, timeWindow_e])

        # Ask for RESUME CONFIRMATION here
        confirmedResume = QMessageBox.Cancel
        if self.log.possibleAppend:
            filesExistAndDone = set(self.log.filesDone).intersection(set(allwavs))
            if len(filesExistAndDone) < total:
                text = "Previous analysis found in this folder (analyzed " + str(len(filesExistAndDone)) + " out of " + str(total) + " files in this folder).\nWould you like to resume that analysis?"
                msg = SupportClasses.MessagePopup("t", "Resume previous batch analysis?", text)
                msg.setStandardButtons(QMessageBox.No | QMessageBox.Yes)
                confirmedResume = msg.exec_()
            else:
                print("All files appear to have previous analysis results")
                msg = SupportClasses.MessagePopup("d", "Already processed", "All files have previous analysis results")
                msg.exec_()
        else:
            confirmedResume = QMessageBox.No

        if confirmedResume == QMessageBox.Cancel:
            # catch unclean (Esc) exits
            return
        elif confirmedResume == QMessageBox.No:
            # work on all files
            self.filesDone = []
        elif confirmedResume == QMessageBox.Yes:
            # ignore files in log
            self.filesDone = filesExistAndDone

        # Ask for FINAL USER CONFIRMATION here
        cnt = len(self.filesDone)
        confirmedLaunch = QMessageBox.Cancel

        text = "Species: " + self.species + ", resolution: "+ str(self.w_res.value()) + ", method: " + self.method + ".\nNumber of files to analyze: " + str(total) + ", " + str(cnt) + " done so far.\n"
        text += "Output stored in " + self.dirName + "/DetectionSummary_*.xlsx.\n"
        text += "Log file stored in " + self.dirName + "/LastAnalysisLog.txt.\n"
        text = "Analysis will be launched with these settings:\n" + text + "\nConfirm?"

        msg = SupportClasses.MessagePopup("t", "Launch batch analysis", text)
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        confirmedLaunch = msg.exec_()

        if confirmedLaunch == QMessageBox.Cancel:
            print("Analysis cancelled")
            return

        # update log: delete everything (by opening in overwrite mode),
        # reprint old headers,
        # print current header (or old if resuming),
        # print old file list if resuming.
        self.log.file = open(self.log.file, 'w')
        if self.species != "All species":
            self.log.reprintOld()
            # else single-sp runs should be deleted anyway
        if confirmedResume == QMessageBox.No:
            self.log.appendHeader(header=None, species=self.log.species, settings=self.log.settings)
        elif confirmedResume == QMessageBox.Yes:
            self.log.appendHeader(self.log.currentHeader, self.log.species, self.log.settings)
            for f in self.log.filesDone:
                self.log.appendFile(f)

        # MAIN PROCESSING starts here
        processingTime = 0
        cleanexit = 0
        cnt = 0
        with pg.BusyCursor():
            for filename in allwavs:
                processingTimeStart = time.time()
                self.filename = filename
                self.segments = Segment.SegmentList()
                newSegments = []
                # get remaining run time in min
                hh,mm = divmod(processingTime * (total-cnt) / 60, 60)
                cnt = cnt+1
                print("*** Processing file %d / %d : %s ***" % (cnt, total, filename))
                self.statusBar().showMessage("Processing file %d / %d. Time remaining: %d h %.2f min" % (cnt, total, hh, mm))
                QtGui.QApplication.processEvents()

                # if it was processed previously (stored in log)
                if filename in self.filesDone:
                    # skip the processing:
                    print("File %s processed previously, skipping" % filename)
                    continue

                # check if file not empty
                if os.stat(filename).st_size < 100:
                    print("File %s empty, skipping" % filename)
                    self.log.appendFile(filename)
                    continue

                # test the selected time window if it is a doc recording
                inWindow = False

                DOCRecording = re.search('(\d{6})_(\d{6})', os.path.basename(filename))
                if DOCRecording:
                    startTime = DOCRecording.group(2)
                    sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
                    if timeWindow_s == timeWindow_e:
                        inWindow = True
                    elif timeWindow_s < timeWindow_e:
                        if sTime >= timeWindow_s and sTime <= timeWindow_e:
                            inWindow = True
                        else:
                            inWindow = False
                    else:
                        if sTime >= timeWindow_s or sTime <= timeWindow_e:
                            inWindow = True
                        else:
                            inWindow = False
                else:
                    inWindow = True

                if DOCRecording and not inWindow:
                    print("Skipping out-of-time-window recording")
                    self.log.appendFile(filename)
                    continue

                # ALL SYSTEMS GO: process this file
                print("Loading file...")
                self.loadFile(wipe=(self.species == "All species"))

                newSegments = []
                print("Segmenting...")
                # clean up old segments:
                if self.species != 'All species':
                    # wipe same species:
                    self.ws = WaveletSegment.WaveletSegment(self.speciesData, 'dmey2')
                    oldsegs = self.segments.getSpecies(self.species)
                    for i in reversed(oldsegs):
                        wipeAll = self.segments[i].wipeSpecies(self.species)
                        if wipeAll:
                            del self.segments[i]
                else:
                    # wipe all segments:
                    self.segments.clear()

                # Segment over pages separately, to allow dealing with large files smoothly:
                # page size fixed for now
                samplesInPage = 900*16000
                # (ceil division for large integers)
                numPages = (len(self.audiodata) - 1) // samplesInPage + 1

                # Actual segmentation happens here:
                for page in range(numPages):
                    print("Segmenting page %d / %d" % (page+1, numPages))
                    start = page*samplesInPage
                    end = min(start+samplesInPage, len(self.audiodata))
                    thisPageLen = (end-start) / self.sampleRate

                    if thisPageLen < 2:
                        print("Warning: can't process short file ends (%0.2f s)" % thisPageLen)
                        continue

                    if self.species != 'All species':
                        # note: using 'recaa' mode = partial anitalias
                        thisPageSegs = self.ws.waveletSegment(data=self.audiodata[start:end], sampleRate=self.sampleRate, d=False, f=True, wpmode="new")
                    else:
                        # Create spectrogram for median clipping etc
                        self.sgRaw = self.sp.spectrogram(self.audiodata[start:end], window_width=self.config['window_width'], incr=self.config['incr'], window='Hann', mean_normalise=True, onesided=True, multitaper=False, need_even=False)
                        self.seg = Segment.Segmenter(self.audiodata[start:end], self.sgRaw, self.sp, self.sampleRate)
                        thisPageSegs = self.seg.bestSegments()

                    if start != 0 :
                        thisPageSegs = [[seg[0]+start/self.sampleRate, seg[1]+start/self.sampleRate] for seg in thisPageSegs]

                    # combine segments from each page into a single list
                    newSegments.extend(thisPageSegs)
                print("Segmentation complete. %d new segments marked" % len(newSegments))

                # post process to remove short segments, wind, rain, and use F0 check.
                print("Post processing...")
                if self.species == 'All species':
                    post = SupportClasses.postProcess(audioData=self.audiodata, sampleRate=self.sampleRate,
                                                      segments=newSegments, spInfo={})
                    post.wind()
                    print('After wind: ', post.segments)
                else:
                    post = SupportClasses.postProcess(audioData=self.audiodata, sampleRate=self.sampleRate,
                                                      segments=newSegments, spInfo=self.speciesData)
                    print('self.speciesData: ', self.speciesData)
                    if 'Wind' in self.speciesData:
                        if self.speciesData['Wind']:
                            print('Deleting wind...')
                            post.wind(windT=2.5, fn_peak=0.35)
                            print('After wind: ', post.segments)
                    if 'F0' in self.speciesData:
                        if self.speciesData['F0']:
                            pass
                        # post.fundamentalFrq(self.filename, self.speciesData)
                        # print('After ff: ', post.segments)
                    if 'Kiwi' in self.speciesData['species']:  # Not sure if it is appropriate for other species
                        print('Merging neighbours...')
                        post.mergeneighbours()
                        print('After merge neighbours: ', post.segments)
                        print('Deleting short...')
                        post.short(minLen=2)  # TODO: keep 'deleteShort' in filter file?
                        print('After del short: ', post.segments)
                newSegments = post.segments
                print('Segments after post pro: ', newSegments)

                # export segments
                cleanexit = self.saveAnnotation(newSegments)
                if cleanexit != 1:
                    print("Warning: could not save segments!")
                # Log success for this file
                self.log.appendFile(self.filename)

                # track how long it took to process one file:
                processingTime = time.time() - processingTimeStart
                print("File processed in", processingTime)
            # END of audio batch processing

            # delete old results (xlsx)
            # ! WARNING: any Detection...xlsx files will be DELETED,
            # ! ANYWHERE INSIDE the specified dir, recursively
            for root, dirs, files in os.walk(str(self.dirName)):
                for filename in files:
                    filenamef = os.path.join(root, filename)
                    if fnmatch.fnmatch(filenamef, '*DetectionSummary_*.xlsx'):
                        print("Removing excel file %s" % filenamef)
                        os.remove(filenamef)

            # Determine all species detected in at least one file
            # (two loops ensure that all files will have pres/abs xlsx for all species.
            # Ugly, but more readable this way)
            spList = set([self.species])
            for filename in allwavs:
                if not os.path.isfile(filename + '.data'):
                    continue

                segments = Segment.SegmentList()
                segments.parseJSON(filename + '.data')
                print(filename,len(segments))

                for seg in segments:
                    spList.update([lab["species"] for lab in seg[4]])

            # Save the new excels
            print("Exporting to Excel ...")
            self.statusBar().showMessage("Exporting to Excel ...")
            for filename in allwavs:
                if not os.path.isfile(filename + '.data'):
                    continue

                segments = Segment.SegmentList()
                segments.parseJSON(filename + '.data')

                # This will be incompatible with old .data files that didn't store file size!
                datalen = np.ceil(segments.metadata["Duration"])

                success = segments.exportExcel(self.dirName, filename, "append", datalen, resolution=self.w_res.value(), speciesList=list(spList))
                print(success)

        # END of processing and exporting. Final cleanup
        self.log.file.close()
        self.statusBar().showMessage("Processed all %d files" % total)
        msg = SupportClasses.MessagePopup("d", "Finished", "Finished processing. Would you like to return to the start screen?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        reply = msg.exec_()
        if reply == QMessageBox.Yes:
            QApplication.exit(1)

    def saveAnnotation(self, segmentsNew):
        """ Generates default batch-mode metadata,
            and saves that and segmentsNew to a .data file. """

        self.segments.metadata["Operator"] = "Auto"
        self.segments.metadata["Reviewer"] = ""
        self.segments.metadata["Duration"] = float(self.datalength)/self.sampleRate
        self.segments.metadata["noiseLevel"] = None
        self.segments.metadata["noiseTypes"] = []

        # These parameters will be set for the new segments:
        if self.species != 'All species':
            y1 = self.speciesSubf['FreqRange'][0]
            y2 = min(self.sampleRate//2, self.speciesSubf['FreqRange'][1])
            species = self.species
            cert = 50
        else:
            y1 = 0
            y2 = 0
            species = "Don't Know"
            cert = 0

        self.segments.addBasicSegments(segmentsNew, [y1, y2], species=species, certainty=cert)
        self.segments.saveJSON(str(self.filename) + '.data')

        return 1


    def fillFileList(self,fileName):
        """ Generates the list of files for the file listbox.
        fileName - currently opened file (marks it in the list).
        Most of the work is to deal with directories in that list.
        It only sees *.wav files. Picks up *.data and *_1.wav files, the first to make the filenames
        red in the list, and the second to know if the files are long."""

        if not os.path.isdir(self.dirName):
            print("ERROR: directory %s doesn't exist" % self.soundFileDir)
            return

        # clear file listbox
        self.listFiles.clearSelection()
        self.listFiles.clearFocus()
        self.listFiles.clear()

        self.listOfFiles = QDir(self.dirName).entryInfoList(['..','*.wav'],filters=QDir.AllDirs|QDir.NoDot|QDir.Files,sort=QDir.DirsFirst)
        listOfDataFiles = QDir(self.dirName).entryList(['*.data'])
        for file in self.listOfFiles:
            # If there is a .data version, colour the name red to show it has been labelled
            item = QListWidgetItem(self.listFiles)
            self.listitemtype = type(item)
            if file.isDir():
                item.setText(file.fileName() + "/")
            else:
                item.setText(file.fileName())
            if file.fileName()+'.data' in listOfDataFiles:
                item.setForeground(Qt.red)
        # mark the current file
        if fileName:
            index = self.listFiles.findItems(fileName+"\/?", Qt.MatchRegExp)
            if len(index)>0:
                self.listFiles.setCurrentItem(index[0])
            else:
                self.listFiles.setCurrentRow(0)

    def listLoadFile(self,current):
        """ Listener for when the user clicks on an item in filelist
        """

        # Need name of file
        if type(current) is self.listitemtype:
            current = current.text()
            current = re.sub('\/.*', '', current)

        self.previousFile = current

        # Update the file list to show the right one
        i=0
        while i<len(self.listOfFiles)-1 and self.listOfFiles[i].fileName() != current:
            i+=1
        if self.listOfFiles[i].isDir() or (i == len(self.listOfFiles)-1 and self.listOfFiles[i].fileName() != current):
            dir = QDir(self.dirName)
            dir.cd(self.listOfFiles[i].fileName())
            # Now repopulate the listbox
            self.dirName=str(dir.absolutePath())
            self.previousFile = None
            if (i == len(self.listOfFiles)-1) and (self.listOfFiles[i].fileName() != current):
                self.loadFile(current)
            self.fillFileList(current)
        return(0)

    def loadFile(self, wipe=True):
        print(self.filename)
        wavobj = wavio.read(self.filename)
        self.sampleRate = wavobj.rate
        self.audiodata = wavobj.data

        # None of the following should be necessary for librosa
        if np.shape(np.shape(self.audiodata))[0] > 1:
            self.audiodata = np.squeeze(self.audiodata[:, 0])
        if self.audiodata.dtype != 'float':
            self.audiodata = self.audiodata.astype('float') #/ 32768.0
            # self.audiodata = self.audiodata[:, 0]
        self.datalength = np.shape(self.audiodata)[0]
        print("Read %d samples, %f s at %d Hz" % (len(self.audiodata), float(self.datalength)/self.sampleRate, self.sampleRate))

        # Create an instance of the Signal Processing class
        if not hasattr(self, 'sp'):
            self.sp = SignalProc.SignalProc()

        # Read in stored segments (useful when doing multi-species)
        self.segments = Segment.SegmentList()
        if wipe or not os.path.isfile(self.filename + '.data'):
            # Initialize default metadata values
            self.segments.metadata = dict()
            self.segments.metadata["Operator"] = "Auto"
            self.segments.metadata["Reviewer"] = ""
            self.segments.metadata["Duration"] = float(self.datalength)/self.sampleRate
        else:
            self.segments.parseJSON(self.filename+'.data')

            # wipe segments if running species-specific analysis:
            oldsegs = self.segments.getSpecies(self.species)
            # deleting from the end, to avoid shifting IDs:
            for si in reversed(oldsegs):
                # remove labels or drop the segment if it's the only species
                wipedAll = self.segments[si].wipeSpecies(self.species)
                if wipedAll:
                    del self.segments[si]

            print("%d segments loaded from .data file" % len(self.segments))

        # Do impulse masking by default
        self.impMask()

        # Update the data that is seen by the other classes
        self.sp.setNewData(self.audiodata, self.sampleRate)

    def impMask(self, engp=90, fp=0.75):
        """
        Impulse mask
        :param engp: energy percentile (for rows of the spectrogram)
        :param fp: frequency proportion to consider it as an impulse (cols of the spectrogram)
        :return: None, but reset audiodata
        """
        print('Impulse masking...')
        postp = SupportClasses.postProcess(audioData=self.audiodata, sampleRate=self.sampleRate, segments=[], spInfo={})
        imps = postp.impulse_cal(fs=self.sampleRate, engp=engp, fp=fp)    # 0 - presence of impulse noise
        print('Samples to mask: ', len(self.audiodata) - np.sum(imps))
        # Mask only the affected samples
        self.audiodata = np.multiply(self.audiodata, imps)

    def countConsecutive(self, nums, length):
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        edges = list(zip(edges, edges))
        edges_reps = [item[1] - item[0] + 1 for item in edges]
        res = np.zeros((length)).tolist()
        t = 0
        for item in edges:
            for i in range(item[0], item[1]+1):
                res[i] = edges_reps[t]
            t += 1
        return res


class AviaNZ_reviewAll(QMainWindow):
    # Main class for reviewing batch processing results
    # Should call HumanClassify1 somehow

    def __init__(self,root=None,configdir='',minSegment=50):
        # Allow the user to browse a folder and push a button to process that folder to find a target species
        # and sets up the window.
        super(AviaNZ_reviewAll, self).__init__()
        self.root = root
        self.dirName=""
        self.configdir = configdir

        # At this point, the main config file should already be ensured to exist.
        self.configfile = os.path.join(configdir, "AviaNZconfig.txt")
        self.ConfigLoader = SupportClasses.ConfigLoader()
        self.config = self.ConfigLoader.config(self.configfile)
        self.saveConfig = True

        # audio things
        self.audioFormat = QAudioFormat()
        self.audioFormat.setCodec("audio/pcm")
        self.audioFormat.setByteOrder(QAudioFormat.LittleEndian)
        self.audioFormat.setSampleType(QAudioFormat.SignedInt)

        # Make the window and associated widgets
        QMainWindow.__init__(self, root)

        self.statusBar().showMessage("Reviewing file Current/Total")

        self.setWindowTitle('AviaNZ - Review Batch Results')
        self.createFrame()
        self.createMenu()
        self.center()

    def createFrame(self):
        # Make the window and set its size
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.setFixedSize(800, 500)
        self.setWindowIcon(QIcon('img/Avianz.ico'))

        # Make the docks
        self.d_detection = Dock("Review",size=(500,500))
        # self.d_detection.hideTitleBar()
        self.d_files = Dock("File list", size=(270, 500))

        self.area.addDock(self.d_detection, 'right')
        self.area.addDock(self.d_files, 'left')

        self.w_revLabel = QLabel("  Reviewer")
        self.w_reviewer = QLineEdit()
        self.d_detection.addWidget(self.w_revLabel, row=0, col=0)
        self.d_detection.addWidget(self.w_reviewer, row=0, col=1, colspan=2)
        self.w_browse = QPushButton("  &Browse Folder")
        self.w_browse.setToolTip("Can select a folder with sub folders to process")
        self.w_browse.setFixedHeight(50)
        self.w_browse.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')
        self.w_dir = QPlainTextEdit()
        self.w_dir.setFixedHeight(50)
        self.w_dir.setPlainText('')
        self.w_dir.setToolTip("The folder being processed")
        self.d_detection.addWidget(self.w_dir,row=1,col=1,colspan=2)
        self.d_detection.addWidget(self.w_browse,row=1,col=0)

        self.w_speLabel1 = QLabel("  Select Species")
        self.d_detection.addWidget(self.w_speLabel1,row=2,col=0)
        self.w_spe1 = QComboBox()
        self.spList = ['All species']
        self.w_spe1.addItems(self.spList)
        self.d_detection.addWidget(self.w_spe1,row=2,col=1,colspan=2)

        self.w_resLabel = QLabel("  Time Resolution in Excel Output (s)")
        self.d_detection.addWidget(self.w_resLabel, row=3, col=0)
        self.w_res = QSpinBox()
        self.w_res.setRange(1,600)
        self.w_res.setSingleStep(5)
        self.w_res.setValue(60)
        self.d_detection.addWidget(self.w_res, row=3, col=1, colspan=2)

        # sliders to select min/max frequencies for ALL SPECIES only
        self.fLow = QSlider(Qt.Horizontal)
        self.fLow.setTickPosition(QSlider.TicksBelow)
        self.fLow.setTickInterval(500)
        self.fLow.setRange(0, 5000)
        self.fLow.setSingleStep(100)
        self.fLowtext = QLabel('  Show freq. above (Hz)')
        self.fLowvalue = QLabel('0')
        receiverL = lambda value: self.fLowvalue.setText(str(value))
        self.fLow.valueChanged.connect(receiverL)
        self.fHigh = QSlider(Qt.Horizontal)
        self.fHigh.setTickPosition(QSlider.TicksBelow)
        self.fHigh.setTickInterval(1000)
        self.fHigh.setRange(4000, 32000)
        self.fHigh.setSingleStep(250)
        self.fHigh.setValue(8000)
        self.fHightext = QLabel('  Show freq. below (Hz)')
        self.fHighvalue = QLabel('8000')
        receiverH = lambda value: self.fHighvalue.setText(str(value))
        self.fHigh.valueChanged.connect(receiverH)
        # add sliders to dock
        self.d_detection.addWidget(self.fLowtext, row=4, col=0)
        self.d_detection.addWidget(self.fLow, row=4, col=1)
        self.d_detection.addWidget(self.fLowvalue, row=4, col=2)
        self.d_detection.addWidget(self.fHightext, row=5, col=0)
        self.d_detection.addWidget(self.fHigh, row=5, col=1)
        self.d_detection.addWidget(self.fHighvalue, row=5, col=2)

        self.w_processButton = QPushButton("&Review Folder")
        self.w_processButton.clicked.connect(self.review)
        self.d_detection.addWidget(self.w_processButton,row=11,col=2)
        self.w_processButton.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')

        self.w_browse.clicked.connect(self.browse)
        # print("spList after browse: ", self.spList)

        self.w_files = pg.LayoutWidget()
        self.d_files.addWidget(self.w_files)
        self.w_files.addWidget(QLabel('View Only'), row=0, col=0)
        self.w_files.addWidget(QLabel('use Browse Folder to choose data for processing'), row=1, col=0)
        # self.w_files.addWidget(QLabel(''), row=2, col=0)
        # List to hold the list of files
        self.listFiles = QListWidget()
        self.listFiles.setMinimumWidth(150)
        self.listFiles.itemDoubleClicked.connect(self.listLoadFile)
        self.w_files.addWidget(self.listFiles, row=2, col=0)

        self.show()

    def createMenu(self):
        """ Create the basic menu.
        """

        helpMenu = self.menuBar().addMenu("&Help")
        helpMenu.addAction("Help", self.showHelp,"Ctrl+H")
        aboutMenu = self.menuBar().addMenu("&About")
        aboutMenu.addAction("About", self.showAbout,"Ctrl+A")
        aboutMenu = self.menuBar().addMenu("&Quit")
        aboutMenu.addAction("Quit", self.quitPro,"Ctrl+Q")

    def showAbout(self):
        """ Create the About Message Box. Text is set in SupportClasses.MessagePopup"""
        msg = SupportClasses.MessagePopup("a", "About", ".")
        msg.exec_()
        return

    def showHelp(self):
        """ Show the user manual (a pdf file)"""
        # TODO: manual is not distributed as pdf now
        import webbrowser
        # webbrowser.open_new(r'file://' + os.path.realpath('./Docs/AviaNZManual.pdf'))
        webbrowser.open_new(r'http://avianz.net/docs/AviaNZManual.pdf')

    def quitPro(self):
        """ quit program
        """
        QApplication.quit()

    def center(self):
        # geometry of the main window
        qr = self.frameGeometry()
        # center point of screen
        cp = QDesktopWidget().availableGeometry().center()
        # move rectangle's center point to screen's center point
        qr.moveCenter(cp)
        # top left of rectangle becomes top left of window centering it
        self.move(qr.topLeft())

    def cleanStatus(self):
        self.statusBar().showMessage("Processing file Current/Total")

    def browse(self):
        # self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',"Wav files (*.wav)")
        if self.dirName:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirName))
        else:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process')
        #print("Dir:", self.dirName)
        self.w_dir.setPlainText(self.dirName)
        self.spList = set()
        # find species names from the annotations
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                if filename.endswith('.wav') and filename+'.data' in files:
                    f = os.path.join(root, filename+'.data')
                    segments = Segment.SegmentList()
                    segments.parseJSON(f)
                    for seg in segments:
                        self.spList.update([lab["species"] for lab in seg[4]])
        self.spList = list(self.spList)
        # Can't review only "Don't Knows". Ideally this should call AllSpecies dialog tho
        try:
            self.spList.remove("Don't Know")
        except Exception:
            pass
        self.spList.insert(0, 'All species')
        self.w_spe1.clear()
        self.w_spe1.addItems(self.spList)
        self.fillFileList(self.dirName)

    def review(self):
        self.species = self.w_spe1.currentText()
        self.reviewer = self.w_reviewer.text()
        print("Reviewer: ", self.reviewer)
        if self.reviewer == '':
            msg = SupportClasses.MessagePopup("w", "Enter Reviewer", "Please enter reviewer name")
            msg.exec_()
            return

        if self.dirName == '':
            msg = SupportClasses.MessagePopup("w", "Select Folder", "Please select a folder to process!")
            msg.exec_()
            return

        # LIST ALL WAV + DATA pairs that can be processed
        allwavs = []
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                filenamef = os.path.join(root, filename)
                if filename.endswith('.wav') and os.path.isfile(filenamef + '.data'):
                    allwavs.append(filenamef)
        total = len(allwavs)
        print(total, "files found")

        # main file review loop
        cnt = 0
        filesuccess = 1
        for filename in allwavs:
            self.filename = filename

            cnt=cnt+1
            print("*** Reviewing file %d / %d : %s ***" % (cnt, total, filename))
            self.statusBar().showMessage("Reviewing file " + str(cnt) + "/" + str(total) + "...")
            QtGui.QApplication.processEvents()

            if not os.path.isfile(filename + '.data'):
                print("Warning: .data file lost for file", filename)
                continue

            if os.stat(filename).st_size < 100:
                print("File %s empty, skipping" % filename)
                continue

            DOCRecording = re.search('(\d{6})_(\d{6})', os.path.basename(filename))
            if DOCRecording:
                startTime = DOCRecording.group(2)
                sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
            else:
                sTime = 0

            # load segments
            self.segments = Segment.SegmentList()
            self.segments.parseJSON(filename+'.data')

            if len(self.segments)==0:
                # skip review dialog, but save the name into excel
                print("No segments found in file %s" % filename)
                filesuccess = 1
            # file has segments, so call the right review dialog:
            # (they will update self.segments and store corrections)
            elif self.species == 'All species':
                self.loadFile(filename)
                filesuccess = self.review_all(sTime)
            else:
                # check if there are any segments for this single species
                if len(self.segments.getSpecies(self.species))==0:
                    print("No segments found in file %s" % filename)
                else:
                    # thus, we can be sure that >=1 relevant segment exists
                    # if this dialog is called.
                    self.loadFile(filename)
                    filesuccess = self.review_single(sTime)
            # break out of review loop if Esc detected
            # (return value will be 1 for correct close, 0 for Esc)
            if filesuccess == 0:
                print("Review stopped")
                break
            # otherwise save the corrected segment JSON
            cleanexit = self.segments.saveJSON(filename+'.data')
            if cleanexit != 1:
                print("Warning: could not save segments!")
        # END of main review loop

        with pg.BusyCursor():
            # delete old results (xlsx)
            # ! WARNING: any Detection...xlsx files will be DELETED,
            # ! ANYWHERE INSIDE the specified dir, recursively
            for root, dirs, files in os.walk(str(self.dirName)):
                for filename in files:
                    filenamef = os.path.join(root, filename)
                    if fnmatch.fnmatch(filenamef, '*DetectionSummary_*.xlsx'):
                        print("Removing excel file %s" % filenamef)
                        os.remove(filenamef)

            # Determine all species detected in at least one file
            # (two loops ensure that all files will have pres/abs xlsx for all species.
            # Ugly, but more readable this way)
            spList = set([self.species])
            for filename in allwavs:
                if not os.path.isfile(filename + '.data'):
                    continue

                segments = Segment.SegmentList()
                segments.parseJSON(filename + '.data')
                print(filename,len(segments))

                for seg in segments:
                    spList.update([lab["species"] for lab in seg[4]])

            # Collect all .data contents to an Excel file (no matter if review dialog exit was clean)
            print("Exporting to Excel ...")
            self.statusBar().showMessage("Exporting to Excel ...")
            for filename in allwavs:
                if not os.path.isfile(filename + '.data'):
                    continue

                segments = Segment.SegmentList()
                segments.parseJSON(filename + '.data')

                # This will be incompatible with old .data files that didn't store file size!
                datalen = np.ceil(segments.metadata["Duration"])

                success = segments.exportExcel(self.dirName, filename, "append", datalen, resolution=self.w_res.value(), speciesList=list(spList))
                print(success)

        # END of review and exporting. Final cleanup
        self.statusBar().showMessage("Reviewed files " + str(cnt) + "/" + str(total))
        if filesuccess == 1:
            msg = SupportClasses.MessagePopup("d", "Finished", "All files checked. Would you like to return to the start screen?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            reply = msg.exec_()
            if reply == QMessageBox.Yes:
                QApplication.exit(1)
        else:
            msg = SupportClasses.MessagePopup("w", "Review stopped", "Review stopped at file %s of %s.\nWould you like to return to the start screen?" % (cnt, total))
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            reply = msg.exec_()
            if reply == QMessageBox.Yes:
                QApplication.exit(1)

    def review_single(self, sTime):
        """ Initializes single species dialog, based on self.species
            (thus we don't need the small species choice dialog here).
            Updates self.segments as a side effect.
            Returns 1 for clean completion, 0 for Esc press or other dirty exit.
        """
        # Initialize the dialog for this file
        self.humanClassifyDialog2 = Dialogs.HumanClassify2(self.sg, self.audiodata, self.segments,
                                                           self.species, self.sampleRate, self.audioFormat,
                                                           self.config['incr'], self.lut, self.colourStart,
                                                           self.colourEnd, self.config['invertColourMap'],
                                                           self.config['brightness'], self.config['contrast'], filename=self.filename)
        if hasattr(self, 'dialogPos'):
            self.humanClassifyDialog2.resize(self.dialogSize)
            self.humanClassifyDialog2.move(self.dialogPos)
        self.humanClassifyDialog2.finish.clicked.connect(self.humanClassifyClose2)
        success = self.humanClassifyDialog2.exec_()

        # capture Esc press or other "dirty" exit:
        if success == 0:
            return(0)
        else:
            return(1)

    def humanClassifyClose2(self):
        self.segmentsToSave = True
        todelete = []
        # initialize correction file. All "downgraded" segments will be stored
        outputErrors = []

        for btn in self.humanClassifyDialog2.buttons:
            btn.stopPlayback()
            currSeg = self.segments[btn.index]
            # btn.index carries the index of segment shown on btn
            if btn.mark=="red":
                outputErrors.append(currSeg)
                # remove all labels for the current species
                wipedAll = currSeg.wipeSpecies(self.species)
                # drop the segment if it's the only species, or just update the graphics
                if wipedAll:
                    todelete.append(btn.index)
            # fix certainty of the analyzed species
            elif btn.mark=="yellow":
                for lbindex in range(len(currSeg[4])):
                    label = currSeg[4][lbindex]
                    # find "greens", swap to "yellows"
                    if label["species"]==self.species and label["certainty"]==100:
                        outputErrors.append(currSeg)
                        label["certainty"] = 50
                        currSeg.keys[lbindex] = (label["species"], label["certainty"])
                # update the graphics
                self.updateText(btn.index)
                self.updateColour(btn.index)
            elif btn.mark=="green":
                # find "yellows", swap to "greens"
                currSeg.confirmLabels(self.species)

        # store position to popup the next one in there
        self.dialogSize = self.humanClassifyDialog2.size()
        self.dialogPos = self.humanClassifyDialog2.pos()
        self.humanClassifyDialog2.done(1)

        # Save the errors in a file
        if self.config['saveCorrections'] and len(outputErrors)>0:
            speciesClean = re.sub(r'\W', "_", self.species)
            file = open(self.filename + '.corrections_' + speciesClean, 'a')
            json.dump(outputErrors, file,indent=1)
            file.close()

        # reverse loop to allow deleting segments
        for dl in reversed(todelete):
            del self.segments[dl]
        # done - the segments will be saved by the main loop
        return

    def review_all(self, sTime, minLen=5):
        """ Initializes all species dialog.
            Updates self.segments as a side effect.
            Returns 1 for clean completion, 0 for Esc press or other dirty exit.
        """
        # Load the birdlists:
        # short list is necessary, long list can be None
        self.shortBirdList = self.ConfigLoader.shortbl(self.config['BirdListShort'], self.configdir)
        if self.shortBirdList is None:
            sys.exit()

        # Will be None if fails to load or filename was "None"
        self.longBirdList = self.ConfigLoader.longbl(self.config['BirdListLong'], self.configdir)
        if self.config['BirdListLong'] is None:
            # If don't have a long bird list,
            # check the length of the short bird list is OK, and otherwise split it
            # 40 is a bit random, but 20 in a list is long enough!
            if len(self.shortBirdList) > 40:
                self.longBirdList = self.shortBirdList.copy()
                self.shortBirdList = self.shortBirdList[:40]
            else:
                self.longBirdList = None

        self.humanClassifyDialog1 = Dialogs.HumanClassify1(self.lut,self.colourStart,self.colourEnd,self.config['invertColourMap'], self.config['brightness'], self.config['contrast'], self.shortBirdList, self.longBirdList, self.config['MultipleSpecies'], self)
        self.box1id = 0
        if hasattr(self, 'dialogPos'):
            self.humanClassifyDialog1.resize(self.dialogSize)
            self.humanClassifyDialog1.move(self.dialogPos)
        self.humanClassifyDialog1.setWindowTitle("AviaNZ - reviewing " + self.filename)
        self.humanClassifyNextImage1()
        # connect listeners
        self.humanClassifyDialog1.correct.clicked.connect(self.humanClassifyCorrect1)
        self.humanClassifyDialog1.delete.clicked.connect(self.humanClassifyDelete1)
        self.humanClassifyDialog1.buttonPrev.clicked.connect(self.humanClassifyPrevImage)
        self.humanClassifyDialog1.buttonNext.clicked.connect(self.humanClassifyNextImage1)
        success = self.humanClassifyDialog1.exec_() # 1 on clean exit

        if success == 0:
            self.humanClassifyDialog1.stopPlayback()
            return(0)

        return(1)

    def loadFile(self, filename):
        with pg.BusyCursor():
            wavobj = wavio.read(filename)
            self.sampleRate = wavobj.rate
            self.audiodata = wavobj.data
            self.audioFormat.setChannelCount(np.shape(self.audiodata)[1])
            self.audioFormat.setSampleRate(self.sampleRate)
            self.audioFormat.setSampleSize(wavobj.sampwidth*8)
            print("Detected format: %d channels, %d Hz, %d bit samples" % (self.audioFormat.channelCount(), self.audioFormat.sampleRate(), self.audioFormat.sampleSize()))

            # None of the following should be necessary for librosa
            if self.audiodata.dtype!='float':
                self.audiodata = self.audiodata.astype('float')
            if np.shape(np.shape(self.audiodata))[0]>1:
                self.audiodata = self.audiodata[:,0]
            self.datalength = np.shape(self.audiodata)[0]
            print("Length of file is ",len(self.audiodata),float(self.datalength)/self.sampleRate,self.sampleRate)
            # self.w_dir.setPlainText(self.filename)

            # Create an instance of the Signal Processing class
            if not hasattr(self,'sp'):
                self.sp = SignalProc.SignalProc()

            # Filter the audiodata based on initial sliders
            minFreq = max(self.fLow.value(), 0)
            maxFreq = min(self.fHigh.value(), self.sampleRate//2)
            if maxFreq - minFreq < 100:
                print("ERROR: less than 100 Hz band set for spectrogram")
                return
            print("Filtering samples to %d - %d Hz" % (minFreq, maxFreq))
            self.audiodata = self.sp.ButterworthBandpass(self.audiodata, self.sampleRate, minFreq, maxFreq)

            # Get the data for the spectrogram
            self.sgRaw = self.sp.spectrogram(self.audiodata, window_width=self.config['window_width'], incr=self.config['incr'], window='Hann', mean_normalise=True, onesided=True,multitaper=False, need_even=False)
            maxsg = np.min(self.sgRaw)
            self.sg = np.abs(np.where(self.sgRaw==0, 0.0, 10.0 * np.log10(self.sgRaw/maxsg)))
            self.setColourMap()

            # trim the spectrogram
            # TODO: could actually skip filtering above
            height = self.sampleRate//2 / np.shape(self.sg)[1]
            pixelstart = int(minFreq/height)
            pixelend = int(maxFreq/height)
            self.sg = self.sg[:,pixelstart:pixelend]

            # Update the data that is seen by the other classes
            self.sp.setNewData(self.audiodata,self.sampleRate)

    def humanClassifyNextImage1(self):
        # Get the next image
        if self.box1id < len(self.segments)-1:
            self.box1id += 1
            # update "done/to go" numbers:
            self.humanClassifyDialog1.setSegNumbers(self.box1id, len(self.segments))
            # Check if have moved to next segment, and if so load it
            # If there was a section without segments this would be a bit inefficient, actually no, it was wrong!

            # Show the next segment
            seg = self.segments[self.box1id]

            # get a list of all species names present
            specnames = []
            for lab in seg[4]:
                if 0<lab["certainty"]<100:
                    specnames.append(lab["species"]+'?')
                else:
                    specnames.append(lab["species"])
            specnames = list(set(specnames))

            x1nob = seg[0]
            x2nob = seg[1]
            x1 = int(self.convertAmpltoSpec(x1nob - self.config['reviewSpecBuffer']))
            x1 = max(x1, 0)
            x2 = int(self.convertAmpltoSpec(x2nob + self.config['reviewSpecBuffer']))
            x2 = min(x2, len(self.sg))
            x3 = int((x1nob - self.config['reviewSpecBuffer']) * self.sampleRate)
            x3 = max(x3, 0)
            x4 = int((x2nob + self.config['reviewSpecBuffer']) * self.sampleRate)
            x4 = min(x4, len(self.audiodata))
            # these pass the axis limits set by slider
            minFreq = max(self.fLow.value(), 0)
            maxFreq = min(self.fHigh.value(), self.sampleRate//2)
            self.humanClassifyDialog1.setImage(self.sg[x1:x2, :], self.audiodata[x3:x4], self.sampleRate, self.config['incr'],
                                           specnames, self.convertAmpltoSpec(x1nob)-x1, self.convertAmpltoSpec(x2nob)-x1,
                                           seg[0], seg[1], minFreq, maxFreq)

        else:
            msg = SupportClasses.MessagePopup("d", "Finished", "All segments in this file checked")
            msg.exec_()

            # store position to popup the next one in there
            self.dialogSize = self.humanClassifyDialog1.size()
            self.dialogPos = self.humanClassifyDialog1.pos()
            self.humanClassifyDialog1.done(1)

    def humanClassifyPrevImage(self):
        """ Go back one image by changing boxid and calling NextImage.
        Note: won't undo deleted segments."""
        if self.box1id>0:
            self.box1id -= 2
            self.humanClassifyNextImage1()

    def humanClassifyCorrect1(self):
        """ Correct segment labels, save the old ones if necessary """
        currSeg = self.segments[self.box1id]

        self.humanClassifyDialog1.stopPlayback()
        label, self.saveConfig, checkText = self.humanClassifyDialog1.getValues()

        # is this needed? this does not exist in AviaNZ.py?
        # if len(checkText) > 0:
        #     if label != checkText:
        #         label = str(checkText)
        #         self.humanClassifyDialog1.birdTextEntered()

        # deal with manual bird entries under "Other"
        if len(checkText) > 0:
            if checkText in self.longBirdList:
                pass
            else:
                self.longBirdList.append(checkText)
                self.longBirdList = sorted(self.longBirdList, key=str.lower)
                self.longBirdList.remove('Unidentifiable')
                self.longBirdList.append('Unidentifiable')
                self.ConfigLoader.blwrite(self.longBirdList, self.config['BirdListLong'], self.configdir)

        if label != [lab["species"] for lab in currSeg[4]]:
            if self.config['saveCorrections']:
                # Save the correction
                outputError = [currSeg, label]
                file = open(self.filename + '.corrections', 'a')
                json.dump(outputError, file, indent=1)
                file.close()

            # Create new segment label, assigning certainty 100 for each species:
            newlabel = []
            for species in label:
                if species == "Don't Know":
                    newlabel.append({"species": "Don't Know", "certainty": 0})
                else:
                    newlabel.append({"species": species, "certainty": 100})
            self.segments[self.box1id] = Segment.Segment([currSeg[0], currSeg[1], currSeg[2], currSeg[3], newlabel])

        elif 0 < min([lab["certainty"] for lab in currSeg[4]]) < 100:
            # If all species remained the same, just raise certainty to 100
            currSeg.confirmLabels()
        else:
            # segment info matches, so don't do anything
            pass

        self.humanClassifyDialog1.tbox.setText('')
        self.humanClassifyDialog1.tbox.setEnabled(False)
        self.humanClassifyNextImage1()

    def humanClassifyDelete1(self):
        # Delete a segment
        # (no need to update counter then)
        id = self.box1id
        del self.segments[id]

        self.box1id = id-1
        self.segmentsToSave = True
        self.humanClassifyNextImage1()

    def closeDialog(self, ev):
        # (actually a poorly named listener for the Esc key)
        if ev == Qt.Key_Escape and hasattr(self, 'humanClassifyDialog1'):
            self.humanClassifyDialog1.done(0)

    def convertAmpltoSpec(self,x):
        """ Unit conversion """
        return x*self.sampleRate/self.config['incr']

    def setColourMap(self):
        """ Listener for the menu item that chooses a colour map.
        Loads them from the file as appropriate and sets the lookup table.
        """
        cmap = self.config['cmap']

        import colourMaps
        pos, colour, mode = colourMaps.colourMaps(cmap)

        cmap = pg.ColorMap(pos, colour,mode)
        self.lut = cmap.getLookupTable(0.0, 1.0, 256)
        minsg = np.min(self.sg)
        maxsg = np.max(self.sg)
        self.colourStart = (self.config['brightness'] / 100.0 * self.config['contrast'] / 100.0) * (maxsg - minsg) + minsg
        self.colourEnd = (maxsg - minsg) * (1.0 - self.config['contrast'] / 100.0) + self.colourStart


    def fillFileList(self,fileName):
        """ Generates the list of files for the file listbox.
        fileName - currently opened file (marks it in the list).
        Most of the work is to deal with directories in that list.
        It only sees *.wav files. Picks up *.data and *_1.wav files, the first to make the filenames
        red in the list, and the second to know if the files are long."""

        if not os.path.isdir(self.dirName):
            print("ERROR: directory %s doesn't exist" % self.soundFileDir)
            return

        # clear file listbox
        self.listFiles.clearSelection()
        self.listFiles.clearFocus()
        self.listFiles.clear()

        self.listOfFiles = QDir(self.dirName).entryInfoList(['..','*.wav'],filters=QDir.AllDirs|QDir.NoDot|QDir.Files,sort=QDir.DirsFirst)
        listOfDataFiles = QDir(self.dirName).entryList(['*.data'])
        for file in self.listOfFiles:
            # If there is a .data version, colour the name red to show it has been labelled
            item = QListWidgetItem(self.listFiles)
            self.listitemtype = type(item)
            if file.isDir():
                item.setText(file.fileName() + "/")
            else:
                item.setText(file.fileName())
            if file.fileName()+'.data' in listOfDataFiles:
                item.setForeground(Qt.red)

    def listLoadFile(self,current):
        """ Listener for when the user clicks on an item in filelist
        """

        # Need name of file
        if type(current) is self.listitemtype:
            current = current.text()
            current = re.sub('\/.*', '', current)

        self.previousFile = current

        # Update the file list to show the right one
        i=0
        while i<len(self.listOfFiles)-1 and self.listOfFiles[i].fileName() != current:
            i+=1
        if self.listOfFiles[i].isDir() or (i == len(self.listOfFiles)-1 and self.listOfFiles[i].fileName() != current):
            dir = QDir(self.dirName)
            dir.cd(self.listOfFiles[i].fileName())
            # Now repopulate the listbox
            self.dirName=str(dir.absolutePath())
            self.listFiles.clearSelection()
            self.listFiles.clearFocus()
            self.listFiles.clear()
            self.previousFile = None
            if (i == len(self.listOfFiles)-1) and (self.listOfFiles[i].fileName() != current):
                self.loadFile(current)
            self.fillFileList(current)
            # Show the selected file
            index = self.listFiles.findItems(os.path.basename(current), Qt.MatchExactly)
            if len(index) > 0:
                self.listFiles.setCurrentItem(index[0])
        return(0)
