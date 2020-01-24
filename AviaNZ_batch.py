
# AviaNZ_batch.py
#
# This is the proceesing class for the batch AviaNZ interface
# Version 2.0 18/11/19
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis

#    AviaNZ bioacoustic analysis program
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
import os, re, fnmatch, sys, gc

from PyQt5.QtGui import QIcon, QPixmap, QApplication, QFont
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QLabel, QPlainTextEdit, QPushButton, QTimeEdit, QSpinBox, QListWidget, QDesktopWidget, QApplication, QComboBox, QLineEdit, QSlider, QListWidgetItem, QCheckBox, QGroupBox, QFormLayout, QGridLayout, QHBoxLayout, QVBoxLayout
from PyQt5.QtMultimedia import QAudioFormat
from PyQt5.QtCore import Qt, QDir

import numpy as np

from pyqtgraph.Qt import QtGui
from pyqtgraph.dockarea import *
import pyqtgraph as pg

import SignalProc
import Segment
import WaveletSegment
import SupportClasses
import Dialogs
import colourMaps

import webbrowser
import json, time


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
        self.statusBar().showMessage("Ready for processing")

        self.setWindowTitle('AviaNZ - Batch Processing')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.createMenu()
        self.createFrame()
        self.center()

    def createFrame(self):
        # Make the window and set its size
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.setMinimumSize(850, 720)

        # Make the docks
        self.d_detection = Dock("Automatic Detection",size=(550, 700))
        self.d_files = Dock("File list", size=(300, 700))

        self.area.addDock(self.d_detection, 'right')
        self.area.addDock(self.d_files, 'left')

        self.w_browse = QPushButton("&Browse Folder")
        self.w_browse.setToolTip("Can select a folder with sub folders to process")
        self.w_browse.setFixedSize(150, 50)
        self.w_browse.setStyleSheet('QPushButton {font-weight: bold; font-size:14px}')
        self.w_dir = QLineEdit()
        self.w_dir.setFixedHeight(50)
        self.w_dir.setReadOnly(True)
        self.w_dir.setText('')
        self.w_dir.setToolTip("The folder being processed")
        self.w_dir.setStyleSheet("color : #808080;")

        w_speLabel1 = QLabel("Select one or more recognisers to use:")
        self.w_spe1 = QComboBox()
        self.speCombos = [self.w_spe1]

        # populate this box (always show all filters here)
        spp = list(self.FilterDicts.keys())
        self.w_spe1.addItems(spp)
        self.w_spe1.addItem("Any sound")
        self.w_spe1.currentTextChanged.connect(self.fillSpeciesBoxes)
        self.addSp = QPushButton("Add another recogniser")
        self.addSp.clicked.connect(self.addSpeciesBox)

        w_resLabel = QLabel("Set size of presence/absence blocks in Excel output\n(Sheet 3)")
        self.w_res = QSpinBox()
        self.w_res.setRange(1, 600)
        self.w_res.setSingleStep(5)
        self.w_res.setValue(60)
        w_timeLabel = QLabel("Want to process a subset of recordings only e.g. dawn or dusk?\nThen select the time window, otherwise skip")
        self.w_timeStart = QTimeEdit()
        self.w_timeStart.setDisplayFormat('hh:mm:ss')
        self.w_timeEnd = QTimeEdit()
        self.w_timeEnd.setDisplayFormat('hh:mm:ss')

        self.w_wind = QCheckBox("")
        self.w_mergect = QCheckBox("")

        # Sliders for minlen and maxgap are in ms scale
        self.minlen = QSlider(Qt.Horizontal)
        self.minlen.setTickPosition(QSlider.TicksBelow)
        self.minlen.setTickInterval(0.5*1000)
        self.minlen.setRange(0.25*1000, 10*1000)
        self.minlen.setSingleStep(1*1000)
        self.minlen.setValue(0.5*1000)
        self.minlen.valueChanged.connect(self.minLenChange)
        self.minlenlbl = QLabel("Minimum segment length: 0.5 sec")

        self.maxlen = QSlider(Qt.Horizontal)
        self.maxlen.setTickPosition(QSlider.TicksBelow)
        self.maxlen.setTickInterval(5*1000)
        self.maxlen.setRange(5*1000, 120*1000)
        self.maxlen.setSingleStep(5*1000)
        self.maxlen.setValue(10*1000)
        self.maxlen.valueChanged.connect(self.maxLenChange)
        self.maxlenlbl = QLabel("Maximum segment length: 10 sec")

        self.maxgap = QSlider(Qt.Horizontal)
        self.maxgap.setTickPosition(QSlider.TicksBelow)
        self.maxgap.setTickInterval(0.5*1000)
        self.maxgap.setRange(0.25*1000, 10*1000)
        self.maxgap.setSingleStep(0.5*1000)
        self.maxgap.setValue(1*1000)
        self.maxgap.valueChanged.connect(self.maxGapChange)
        self.maxgaplbl = QLabel("Maximum gap between syllables: 1 sec")

        self.w_processButton = QPushButton("&Process Folder")
        self.w_processButton.clicked.connect(self.detect)
        self.w_processButton.setStyleSheet('QPushButton {font-weight: bold; font-size:14px}')
        self.w_processButton.setFixedSize(150, 50)
        self.w_browse.clicked.connect(self.browse)

        self.d_detection.addWidget(self.w_dir, row=0, col=0, colspan=2)
        self.d_detection.addWidget(self.w_browse, row=0, col=2)
        self.d_detection.addWidget(w_speLabel1, row=1, col=0)

        # Filter selection group
        self.boxSp = QGroupBox("")
        self.formSp = QVBoxLayout()
        self.formSp.addWidget(w_speLabel1)
        self.formSp.addWidget(self.w_spe1)
        self.formSp.addWidget(self.addSp)
        self.boxSp.setLayout(self.formSp)
        self.d_detection.addWidget(self.boxSp, row=1, col=0, colspan=3)

        # Time Settings group
        boxTime = QGroupBox()
        formTime = QGridLayout()
        formTime.addWidget(w_timeLabel, 0, 0, 1, 2)
        formTime.addWidget(QLabel("Start time (hh:mm:ss)"), 1, 0)
        formTime.addWidget(self.w_timeStart, 1, 1)
        formTime.addWidget(QLabel("End time (hh:mm:ss)"), 2, 0)
        formTime.addWidget(self.w_timeEnd, 2, 1)
        boxTime.setLayout(formTime)
        self.d_detection.addWidget(boxTime, row=2, col=0, colspan=3)

        # Post Proc checkbox group
        boxPost = QGroupBox("Post processing")
        formPost = QGridLayout()
        formPost.addWidget(QLabel("Add wind filter"), 0, 0)
        formPost.addWidget(self.w_wind, 0, 1)
        self.mergectlbl = QLabel("Merge different call types")
        formPost.addWidget(self.mergectlbl, 2, 0)
        formPost.addWidget(self.w_mergect, 2, 1)
        formPost.addWidget(self.maxgaplbl, 3, 0)
        formPost.addWidget(self.maxgap, 3, 1)
        formPost.addWidget(self.minlenlbl, 4, 0)
        formPost.addWidget(self.minlen, 4, 1)
        formPost.addWidget(self.maxlenlbl, 5, 0)
        formPost.addWidget(self.maxlen, 5, 1)
        boxPost.setLayout(formPost)
        self.d_detection.addWidget(boxPost, row=3, col=0, colspan=3)
        if len(spp) > 0:
            self.maxgaplbl.hide()
            self.maxgap.hide()
            self.minlenlbl.hide()
            self.minlen.hide()
            self.maxlenlbl.hide()
            self.maxlen.hide()

        self.d_detection.addWidget(w_resLabel, row=4, col=0)
        self.d_detection.addWidget(self.w_res, row=4, col=1)
        self.d_detection.addWidget(QLabel("(seconds)"), row=4, col=2)
        self.d_detection.addWidget(self.w_processButton, row=5, col=2)

        self.w_files = pg.LayoutWidget()
        self.d_files.addWidget(self.w_files)
        # List to hold the list of files
        self.listFiles = QListWidget()
        self.listFiles.setMinimumWidth(150)
        self.listFiles.itemDoubleClicked.connect(self.listLoadFile)

        #self.w_files.addWidget(QLabel('Double click to select a folder'), row=0, col=0)
        self.w_files.addWidget(QLabel('Red files have annotations'), row=1, col=0)
        self.w_files.addWidget(self.listFiles, row=2, col=0)

        self.d_detection.layout.setContentsMargins(20, 20, 20, 20)
        self.d_detection.layout.setSpacing(20)
        self.d_files.layout.setContentsMargins(10, 10, 10, 10)
        self.d_files.layout.setSpacing(10)
        self.show()

    def minLenChange(self, value):
        self.minlenlbl.setText("Minimum segment length: %s sec" % str(round(int(value)/1000, 2)))

    def maxLenChange(self, value):
        self.maxlenlbl.setText("Maximum segment length: %s sec" % str(round(int(value)/1000, 2)))

    def maxGapChange(self, value):
        self.maxgaplbl.setText("Maximum gap between syllables: %s sec" % str(round(int(value)/1000, 2)))

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

    def browse(self):
        if self.dirName:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirName))
        else:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process')
        #print("Dir:", self.dirName)
        self.w_dir.setText(self.dirName)
        self.w_dir.setReadOnly(True)
        self.fillFileList(self.dirName)

    def addSpeciesBox(self):
        """ Deals with adding and moving species comboboxes """
        # create a new combobox
        newSpBox = QComboBox()
        self.speCombos.append(newSpBox)

        # populate it with possible species (that have same Fs)
        self.fillSpeciesBoxes()

        # create a "delete" button for it
        delSpBtn = QPushButton("X")
        delSpBtn.speciesbox = newSpBox
        delSpBtn.setFixedWidth(30)

        # connect the listener for deleting
        delSpBtn.clicked.connect(self.removeSpeciesBox)

        # insert those just above Add button
        btncombo = QHBoxLayout()
        delSpBtn.layout = btncombo
        btncombo.addWidget(newSpBox)
        btncombo.addWidget(delSpBtn)
        self.formSp.insertLayout(len(self.speCombos), btncombo)

        self.boxSp.setMinimumHeight(30*len(self.speCombos)+90)
        self.setMinimumHeight(610+30*len(self.speCombos))
        self.boxSp.updateGeometry()

    def removeSpeciesBox(self):
        """ Deals with removing and moving species comboboxes """
        # identify the clicked button
        called = self.sender()
        lay = called.layout

        # delete the corresponding combobox and button from their HBox
        self.speCombos.remove(called.speciesbox)
        lay.removeWidget(called.speciesbox)
        called.speciesbox.deleteLater()
        lay.removeWidget(called)
        called.deleteLater()

        # remove the empty HBox
        self.formSp.removeItem(lay)
        lay.deleteLater()

        self.boxSp.setMinimumHeight(30*len(self.speCombos)+90)
        self.setMinimumHeight(610+30*len(self.speCombos))
        self.boxSp.updateGeometry()

    def fillSpeciesBoxes(self):
        # select filters with Fs matching box 1 selection
        # and show/hide minlen maxgap sliders
        spp = []
        currname = self.w_spe1.currentText()
        if currname != "Any sound":
            currfilt = self.FilterDicts[currname]
            # (can't use AllSp with any other filter)
            # Also don't add the same name again
            for name, filter in self.FilterDicts.items():
                if filter["SampleRate"]==currfilt["SampleRate"] and name!=currname:
                    spp.append(name)
            self.minlen.hide()
            self.minlenlbl.hide()
            self.maxlen.hide()
            self.maxlenlbl.hide()
            self.maxgap.hide()
            self.maxgaplbl.hide()
            self.w_mergect.show()
            self.mergectlbl.show()
        else:
            self.minlen.show()
            self.minlenlbl.show()
            self.maxlen.show()
            self.maxlenlbl.show()
            self.maxgap.show()
            self.maxgaplbl.show()
            self.w_mergect.hide()
            self.mergectlbl.hide()

        # (skip first box which is fixed)
        for box in self.speCombos[1:]:
            # clear old items:
            for i in reversed(range(box.count())):
                box.removeItem(i)
            box.setCurrentIndex(-1)
            box.setCurrentText("")

            box.addItems(spp)

    # from memory_profiler import profile
    # fp = open('memory_profiler_batch.log', 'w+')
    # @profile(stream=fp)
    def detect(self):
        # check if folder was selected:
        if not self.dirName:
            msg = SupportClasses.MessagePopup("w", "Select Folder", "Please select a folder to process!")
            msg.exec_()
            return

        # retrieve selected filter(s)
        self.species = set()
        for box in self.speCombos:
            if box.currentText() != "":
                self.species.add(box.currentText())
        self.species = list(self.species)
        print("Species:", self.species)

        if "Any sound" in self.species:
            self.method = "Default"
            speciesStr = "Any sound"
        else:
            self.method = "Wavelets"

            # double-check that all Fs are equal
            filters = [self.FilterDicts[name] for name in self.species]
            samplerate = set([filt["SampleRate"] for filt in filters])
            if len(samplerate)>1:
                print("ERROR: multiple sample rates found in selected recognisers, change selection")
                return

            # convert list to string
            speciesStr = " & ".join(self.species)

        # Parse the user-set time window to process
        timeWindow_s = self.w_timeStart.time().hour() * 3600 + self.w_timeStart.time().minute() * 60 + self.w_timeStart.time().second()
        timeWindow_e = self.w_timeEnd.time().hour() * 3600 + self.w_timeEnd.time().minute() * 60 + self.w_timeEnd.time().second()

        # LIST ALL WAV files that will be processed
        allwavs = []
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                if filename.lower().endswith('.wav'):
                    allwavs.append(os.path.join(root, filename))
        total = len(allwavs)

        # LOG FILE is read here
        # note: important to log all analysis settings here
        settings = [self.method, self.w_res.value(), timeWindow_s, timeWindow_e,
                    self.w_wind.isChecked(), self.w_mergect.isChecked()]
        self.log = SupportClasses.Log(os.path.join(self.dirName, 'LastAnalysisLog.txt'), speciesStr, settings)

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

        text = "Species: " + speciesStr + ", resolution: "+ str(self.w_res.value()) + ", method: " + self.method + ".\nNumber of files to analyze: " + str(total) + ", " + str(cnt) + " done so far.\n"
        text += "Output stored in " + self.dirName + "/DetectionSummary_*.xlsx.\n"
        text += "Log file stored in " + self.dirName + "/LastAnalysisLog.txt.\n"
        if speciesStr=="Any sound":
            text += "\nWarning: any previous annotations in these files will be deleted!\n"
        else:
            text += "\nWarning: any previous annotations for the selected species in these files will be deleted!\n"
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
        if speciesStr != "Any sound":
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
        # clean up the UI before entering the long loop
        self.update()
        self.repaint()
        QtGui.QApplication.processEvents()
        with pg.BusyCursor():
            for filename in allwavs:
                processingTimeStart = time.time()
                self.filename = filename
                self.segments = Segment.SegmentList()
                # get remaining run time in min
                hh,mm = divmod(processingTime * (total-cnt) / 60, 60)
                cnt = cnt+1
                print("*** Processing file %d / %d : %s ***" % (cnt, total, filename))
                self.statusBar().showMessage("Processing file %d / %d. Time remaining: %d h %.2f min" % (cnt, total, hh, mm))
                self.update()
                self.repaint()

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
                # load audiodata and clean up old segments:
                self.loadFile(species=self.species, anysound=(speciesStr == "Any sound"))
                # Segment over pages separately, to allow dealing with large files smoothly:
                # page size fixed for now
                samplesInPage = 900*16000
                # (ceil division for large integers)
                numPages = (len(self.audiodata) - 1) // samplesInPage + 1

                print("Segmenting...")
                self.ws = WaveletSegment.WaveletSegment(wavelet='dmey2')

                # Actual segmentation happens here:
                for page in range(numPages):
                    print("Segmenting page %d / %d" % (page+1, numPages))
                    start = page*samplesInPage
                    end = min(start+samplesInPage, len(self.audiodata))
                    thisPageLen = (end-start) / self.sampleRate

                    if thisPageLen < 2:
                        print("Warning: can't process short file ends (%.2f s)" % thisPageLen)
                        continue

                    # Process
                    if speciesStr == "Any sound":
                        # Create spectrogram for median clipping etc
                        if not hasattr(self, 'sp'):
                            self.sp = SignalProc.SignalProc(self.config['window_width'], self.config['incr'])
                        self.sp.data = self.audiodata[start:end]
                        self.sp.sampleRate = self.sampleRate
                        _ = self.sp.spectrogram(window='Hann', mean_normalise=True, onesided=True, multitaper=False, need_even=False)
                        self.seg = Segment.Segmenter(self.sp, self.sampleRate)
                        # thisPageSegs = self.seg.bestSegments()
                        thisPageSegs = self.seg.medianClip(thr=3.5)
                        # Post-process
                        # 1. Delete windy segments
                        # 2. Delete rainy segments
                        # 3. Check fundamental frq
                        # 4. Merge neighbours
                        # 5. Delete short segments
                        print("Segments detected: ", len(thisPageSegs))
                        print("Post-processing...")
                        maxgap = int(self.maxgap.value())/1000
                        minlen = int(self.minlen.value())/1000
                        maxlen = int(self.maxlen.value())/1000
                        post = Segment.PostProcess(audioData=self.audiodata[start:end], sampleRate=self.sampleRate,
                                                   segments=thisPageSegs, subfilter={})
                        if self.w_wind.isChecked():
                            post.wind()
                            print('After wind segments: ', len(post.segments))
                        post.segments = self.seg.joinGaps(post.segments, maxgap=maxgap)
                        post.segments = self.seg.deleteShort(post.segments, minlength=minlen)
                        print('Segments after merge (<=%d secs) and delete short (<%.2f secs): %d' % (maxgap, minlen, len(post.segments)))
                        # avoid extra long segments (for Isabel)
                        post.segments = self.seg.splitLong(post.segments, maxlen = maxlen)
                        print('Segments after splitting long segments (>%.2f secs): %d' % (maxlen, len(post.segments)))

                        # adjust segment starts for 15min "pages"
                        if start != 0:
                            for seg in post.segments:
                                seg[0] += start/self.sampleRate
                                seg[1] += start/self.sampleRate
                        # attach mandatory "Don't Know"s etc and put on self.segments
                        self.makeSegments(post.segments)
                        del self.seg
                        gc.collect()
                    else:
                        # read in the page and resample as needed
                        self.ws.readBatch(self.audiodata[start:end], self.sampleRate, d=False, spInfo=filters, wpmode="new")

                        allCtSegs = []
                        for speciesix in range(len(filters)):
                            print("Working with recogniser:", filters[speciesix])
                            # note: using 'recaa' mode = partial antialias
                            thisPageSegs = self.ws.waveletSegment(speciesix, wpmode="new")

                            # Post-process
                            # 1. Delete windy segments
                            # 2. Delete rainy segments
                            # 3. Check fundamental frq
                            # 4. Merge neighbours
                            # 5. Delete short segments
                            print("Segments detected: ", len(thisPageSegs))
                            print("Post-processing...")
                            # postProcess currently operates on single-level list of segments,
                            # so we run it over subfilters for wavelets:
                            spInfo = filters[speciesix]
                            for filtix in range(len(spInfo['Filters'])):
                                post = Segment.PostProcess(audioData=self.audiodata[start:end], sampleRate=self.sampleRate, segments=thisPageSegs[filtix], subfilter=spInfo['Filters'][filtix])
                                if self.w_wind.isChecked():
                                    post.wind()
                                    print('After wind: segments: ', len(post.segments))
                                if 'F0' in spInfo['Filters'][filtix] and 'F0Range' in spInfo['Filters'][filtix]:
                                    if spInfo['Filters'][filtix]["F0"]:
                                        print("Checking for fundamental frequency...")
                                        post.fundamentalFrq()
                                        print("After FF segments:", len(post.segments))
                                segmenter = Segment.Segmenter()
                                post.segments = segmenter.joinGaps(post.segments, maxgap=spInfo['Filters'][filtix]['TimeRange'][3])
                                post.segments = segmenter.deleteShort(post.segments, minlength=spInfo['Filters'][filtix]['TimeRange'][0])
                                print('Segments after merge (<=%d secs) and delete short (<%.4f): %d' %(spInfo['Filters'][filtix]['TimeRange'][3], spInfo['Filters'][filtix]['TimeRange'][0], len(post.segments)))

                                # adjust segment starts for 15min "pages"
                                if start != 0:
                                    for seg in post.segments:
                                        seg[0] += start/self.sampleRate
                                        seg[1] += start/self.sampleRate

                                if self.w_mergect.isChecked():
                                    # collect segments from all call types
                                    allCtSegs.extend(post.segments)
                                else:
                                    # attach filter info and put on self.segments:
                                    self.makeSegments(post.segments, self.species[speciesix], spInfo["species"], spInfo['Filters'][filtix])

                            if self.w_mergect.isChecked():
                                # merge different call type segments
                                segmenter = Segment.Segmenter()
                                segs = segmenter.checkSegmentOverlap(allCtSegs)
                                print('allCtSegs:', allCtSegs)
                                print('segs:', segs)
                                # also merge neighbours (segments from different call types)
                                segs = segmenter.joinGaps(segs, maxgap=max([subf['TimeRange'][3] for subf in spInfo["Filters"]]))
                                # construct "Any call" info to place on the segments
                                flow = min([subf["FreqRange"][0] for subf in spInfo["Filters"]])
                                fhigh = max([subf["FreqRange"][1] for subf in spInfo["Filters"]])
                                ctinfo = {"calltype": "(Other)", "FreqRange": [flow, fhigh]}
                                print('self.species[speciesix]:', self.species[speciesix])
                                print('spInfo["species"]:', spInfo["species"])
                                self.makeSegments(segs, self.species[speciesix], spInfo["species"], ctinfo)

                print('Segments in this file: ', self.segments)
                print("Segmentation complete. %d new segments marked" % len(self.segments))

                # export segments
                cleanexit = self.saveAnnotation()
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
            self.statusBar().showMessage("Preparing Excel output, almost done...")
            self.update()
            self.repaint()
            for root, dirs, files in os.walk(str(self.dirName)):
                for filename in files:
                    filenamef = os.path.join(root, filename)
                    if fnmatch.fnmatch(filenamef, '*DetectionSummary_*.xlsx'):
                        print("Removing excel file %s" % filenamef)
                        os.remove(filenamef)

            # Determine all species detected in at least one file
            # (two loops ensure that all files will have pres/abs xlsx for all species.
            # Ugly, but more readable this way)
            if speciesStr != "Any sound":
                spList = set([filter["species"] for filter in filters])
            else:
                spList = set()
            for filename in allwavs:
                if not os.path.isfile(filename + '.data'):
                    continue

                segments = Segment.SegmentList()
                segments.parseJSON(filename + '.data')
                print(filename)

                for seg in segments:
                    spList.update([lab["species"] for lab in seg[4]])

            # Save the new excels
            print("Exporting to Excel ...")
            self.statusBar().showMessage("Exporting to Excel ...")
            self.update()
            self.repaint()
            for filename in allwavs:
                if not os.path.isfile(filename + '.data'):
                    continue

                segments = Segment.SegmentList()
                segments.parseJSON(filename + '.data')

                # This will be incompatible with old .data files that didn't store file size!
                datalen = np.ceil(segments.metadata["Duration"])

                # sort by time and save
                segments.orderTime()
                success = segments.exportExcel(self.dirName, filename, "append", datalen, resolution=self.w_res.value(), speciesList=list(spList))
                if success!=1:
                    print("Warning: failed to export Excel output!")

        # END of processing and exporting. Final cleanup
        self.log.file.close()
        self.statusBar().showMessage("Processed all %d files" % total)
        msg = SupportClasses.MessagePopup("d", "Finished", "Finished processing. Would you like to return to the start screen?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        reply = msg.exec_()
        if reply == QMessageBox.Yes:
            QApplication.exit(1)

    def makeSegments(self, segmentsNew, filtName=None, species=None, subfilter=None):
        """ Adds segments to self.segments """
        # for wavelet segments: (same as self.species!="Any sound")
        if subfilter is not None:
            y1 = subfilter["FreqRange"][0]
            y2 = min(subfilter["FreqRange"][1], self.sampleRate//2)
            species = [{"species": species, "certainty": 50, "filter": filtName, "calltype": subfilter["calltype"]}]
            for s in segmentsNew:
                segment = Segment.Segment([s[0], s[1], y1, y2, species])
                self.segments.addSegment(segment)
        # for generic all-species segments:
        else:
            y1 = 0
            y2 = 0
            species = "Don't Know"
            cert = 0
            self.segments.addBasicSegments(segmentsNew, [y1, y2], species=species, certainty=cert)

    def saveAnnotation(self):
        """ Generates default batch-mode metadata,
            and saves the current self.segments to a .data file. """

        self.segments.metadata["Operator"] = "Auto"
        self.segments.metadata["Reviewer"] = ""
        self.segments.metadata["Duration"] = float(self.datalength)/self.sampleRate
        self.segments.metadata["noiseLevel"] = None
        self.segments.metadata["noiseTypes"] = []

        self.segments.saveJSON(str(self.filename) + '.data')

        return 1

    def fillFileList(self,fileName):
        """ Generates the list of files for the file listbox.
        fileName - currently opened file (marks it in the list).
        Most of the work is to deal with directories in that list.
        It only sees *.wav files. Picks up *.data files, to make the filenames
        red in the list."""

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

        # update the "Browse" field text
        self.w_dir.setText(self.dirName)

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
            self.fillFileList(current)
        return(0)

    def loadFile(self, species, anysound=False):
        print(self.filename)
        # Create an instance of the Signal Processing class
        if not hasattr(self,'sp'):
            self.sp = SignalProc.SignalProc(self.config['window_width'], self.config['incr'])
        self.sp.readWav(self.filename)
        self.sampleRate = self.sp.sampleRate
        self.audiodata = self.sp.data

        self.datalength = np.shape(self.audiodata)[0]
        print("Read %d samples, %f s at %d Hz" % (len(self.audiodata), float(self.datalength)/self.sampleRate, self.sampleRate))

        # Read in stored segments (useful when doing multi-species)
        self.segments = Segment.SegmentList()
        if species==["Any sound"] or not os.path.isfile(self.filename + '.data'):
            # Initialize default metadata values
            self.segments.metadata = dict()
            self.segments.metadata["Operator"] = "Auto"
            self.segments.metadata["Reviewer"] = ""
            self.segments.metadata["Duration"] = float(self.datalength)/self.sampleRate
            # wipe all segments:
            print("Wiping all previous segments")
            self.segments.clear()
        else:
            self.segments.parseJSON(self.filename+'.data', float(self.datalength)/self.sampleRate)
            # wipe same species:
            for filt in self.FilterDicts.values():
                if filt["species"] in species:
                    print("Wiping species", filt["species"])
                    oldsegs = self.segments.getSpecies(filt["species"])
                    for i in reversed(oldsegs):
                        wipeAll = self.segments[i].wipeSpecies(filt["species"])
                        if wipeAll:
                            del self.segments[i]
            print("%d segments loaded from .data file" % len(self.segments))

        # Do impulse masking by default
        sg = Segment.Segmenter(sp=self.sp, fs=self.sampleRate)
        if anysound:
            self.sp.data = sg.impMask(engp=70, fp=0.50)
        else:
            self.sp.data = sg.impMask()
        self.audiodata = self.sp.data
        del self.sp
        gc.collect()

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

        # Make the window and associated widgets
        QMainWindow.__init__(self, root)

        self.statusBar().showMessage("Ready to review")

        self.setWindowTitle('AviaNZ - Review Batch Results')
        self.createFrame()
        self.createMenu()
        self.center()

    def createFrame(self):
        # Make the window and set its size
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.setFixedSize(1000, 600)
        self.setWindowIcon(QIcon('img/Avianz.ico'))

        # Make the docks
        self.d_detection = Dock("Review",size=(700, 600))
        # self.d_detection.hideTitleBar()
        self.d_files = Dock("File list", size=(300, 600))

        self.area.addDock(self.d_detection, 'right')
        self.area.addDock(self.d_files, 'left')

        self.w_revLabel = QLabel("Reviewer")
        self.w_reviewer = QLineEdit()
        self.d_detection.addWidget(self.w_revLabel, row=0, col=0)
        self.d_detection.addWidget(self.w_reviewer, row=0, col=1, colspan=2)
        self.w_browse = QPushButton("&Browse Folder")
        self.w_browse.setToolTip("Can select a folder with sub folders to process")
        self.w_browse.setFixedHeight(50)
        self.w_browse.setStyleSheet('QPushButton {font-weight: bold; font-size:14px}')
        self.w_dir = QPlainTextEdit()
        self.w_dir.setFixedHeight(50)
        self.w_dir.setPlainText('')
        self.w_dir.setToolTip("The folder being processed")
        self.d_detection.addWidget(self.w_dir, row=1,col=1,colspan=2)
        self.d_detection.addWidget(self.w_browse, row=1,col=0)

        self.w_speLabel1 = QLabel("Select Species")
        self.d_detection.addWidget(self.w_speLabel1,row=2,col=0)
        self.w_spe1 = QComboBox()
        self.spList = ['Any sound']
        self.w_spe1.addItems(self.spList)
        self.d_detection.addWidget(self.w_spe1,row=2,col=1,colspan=2)

        minCertLab = QLabel("Skip if certainty above:")
        self.d_detection.addWidget(minCertLab, row=3, col=0)
        self.certBox = QSpinBox()
        self.certBox.setRange(0,100)
        self.certBox.setSingleStep(10)
        self.certBox.setValue(90)
        self.d_detection.addWidget(self.certBox, row=3, col=1)

        self.w_resLabel = QLabel("Set size of presence/absence blocks\nin Excel output (Sheet 3)")
        self.d_detection.addWidget(self.w_resLabel, row=4, col=0)
        self.w_res = QSpinBox()
        self.w_res.setRange(1,600)
        self.w_res.setSingleStep(5)
        self.w_res.setValue(60)
        self.d_detection.addWidget(self.w_res, row=4, col=1)

        # sliders to select min/max frequencies for ALL SPECIES only
        self.fLow = QSlider(Qt.Horizontal)
        self.fLow.setTickPosition(QSlider.TicksBelow)
        self.fLow.setTickInterval(500)
        self.fLow.setRange(0, 5000)
        self.fLow.setSingleStep(100)
        self.fLowtext = QLabel('Show freq. above (Hz)')
        self.fLowvalue = QLabel('0')
        receiverL = lambda value: self.fLowvalue.setText(str(value))
        self.fLow.valueChanged.connect(receiverL)
        self.fHigh = QSlider(Qt.Horizontal)
        self.fHigh.setTickPosition(QSlider.TicksBelow)
        self.fHigh.setTickInterval(1000)
        self.fHigh.setRange(4000, 32000)
        self.fHigh.setSingleStep(250)
        self.fHigh.setValue(8000)
        self.fHightext = QLabel('Show freq. below (Hz)')
        self.fHighvalue = QLabel('8000')
        receiverH = lambda value: self.fHighvalue.setText(str(value))
        self.fHigh.valueChanged.connect(receiverH)
        # add sliders to dock
        self.d_detection.addWidget(self.fLowtext, row=5, col=0)
        self.d_detection.addWidget(self.fLow, row=5, col=1)
        self.d_detection.addWidget(self.fLowvalue, row=5, col=2)
        self.d_detection.addWidget(self.fHightext, row=6, col=0)
        self.d_detection.addWidget(self.fHigh, row=6, col=1)
        self.d_detection.addWidget(self.fHighvalue, row=6, col=2)

        self.w_processButton = QPushButton("&Review Folder")
        self.w_processButton.setFixedHeight(50)
        self.w_processButton.clicked.connect(self.review)
        self.d_detection.addWidget(self.w_processButton,row=11,col=2)
        self.w_processButton.setStyleSheet('QPushButton {font-weight: bold; font-size:14px}')

        self.w_browse.clicked.connect(self.browse)
        # print("spList after browse: ", self.spList)

        self.w_files = pg.LayoutWidget()
        self.d_files.addWidget(self.w_files)
        self.w_files.addWidget(QLabel('View Only'), row=0, col=0)
        self.w_files.addWidget(QLabel('Use Browse Folder button to choose data for processing'), row=1, col=0)
        # self.w_files.addWidget(QLabel(''), row=2, col=0)
        # List to hold the list of files
        self.listFiles = QListWidget()
        self.listFiles.setMinimumWidth(150)
        self.listFiles.itemDoubleClicked.connect(self.listLoadFile)
        self.w_files.addWidget(self.listFiles, row=2, col=0)

        self.d_detection.layout.setContentsMargins(20, 20, 20, 20)
        self.d_detection.layout.setSpacing(20)
        self.d_files.layout.setContentsMargins(10, 10, 10, 10)
        self.d_files.layout.setSpacing(10)
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

    def browse(self):
        # self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',"Wav files (*.wav)")
        if self.dirName:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirName))
        else:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process')
        self.w_dir.setPlainText(self.dirName)
        self.spList = set()
        # find species names from the annotations
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                if filename.lower().endswith('.wav') and filename+'.data' in files:
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
        self.spList.insert(0, 'Any sound')
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
                if filename.lower().endswith('.wav') and os.path.isfile(filenamef + '.data'):
                    allwavs.append(filenamef)
        total = len(allwavs)
        print(total, "files found")

        # main file review loop
        cnt = 0
        filesuccess = 1
        self.update()
        self.repaint()

        for filename in allwavs:
            self.filename = filename

            cnt=cnt+1
            print("*** Reviewing file %d / %d : %s ***" % (cnt, total, filename))
            self.statusBar().showMessage("Reviewing file " + str(cnt) + "/" + str(total) + "...")
            self.update()
            self.repaint()

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
            with pg.BusyCursor():
                self.segments = Segment.SegmentList()
                self.segments.parseJSON(filename+'.data')
                # separate out segments which do not need review
                self.goodsegments = []
                for seg in reversed(self.segments):
                    goodenough = True
                    for lab in seg[4]:
                        if lab["certainty"] <= self.certBox.value():
                            goodenough = False
                    if goodenough:
                        self.goodsegments.append(seg)
                        self.segments.remove(seg)

            if len(self.segments)==0:
                # skip review dialog, but save the name into excel
                print("No segments found in file %s" % filename)
                filesuccess = 1
            # file has segments, so call the right review dialog:
            # (they will update self.segments and store corrections)
            elif self.species == 'Any sound':
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
            # otherwise re-add the segments that were good enough to skip review,
            # and save the corrected segment JSON
            self.segments.extend(self.goodsegments)
            cleanexit = self.segments.saveJSON(filename+'.data', self.reviewer)
            if cleanexit != 1:
                print("Warning: could not save segments!")
        # END of main review loop

        with pg.BusyCursor():
            # delete old results (xlsx)
            # ! WARNING: any Detection...xlsx files will be DELETED,
            # ! ANYWHERE INSIDE the specified dir, recursively
            self.statusBar().showMessage("Preparing Excel output, almost done...")
            self.update()
            self.repaint()
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
                print(filename)

                for seg in segments:
                    spList.update([lab["species"] for lab in seg[4]])

            # Collect all .data contents to an Excel file (no matter if review dialog exit was clean)
            print("Exporting to Excel ...")
            self.statusBar().showMessage("Exporting to Excel ...")
            self.update()
            self.repaint()
            for filename in allwavs:
                if not os.path.isfile(filename + '.data'):
                    continue

                segments = Segment.SegmentList()
                segments.parseJSON(filename + '.data')

                # This will be incompatible with old .data files that didn't store file size!
                datalen = np.ceil(segments.metadata["Duration"])

                # sort by time and save
                segments.orderTime()
                success = segments.exportExcel(self.dirName, filename, "append", datalen, resolution=self.w_res.value(), speciesList=list(spList))
                if success!=1:
                    print("Warning: failed to save Excel output")

        # END of review and exporting. Final cleanup
        self.statusBar().showMessage("Reviewed files " + str(cnt) + "/" + str(total))
        self.update()
        self.repaint()
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
                                                           self.species, self.sampleRate, self.sp.audioFormat,
                                                           self.config['incr'], self.lut, self.colourStart,
                                                           self.colourEnd, self.config['invertColourMap'],
                                                           self.config['brightness'], self.config['contrast'], filename=self.filename)
        if hasattr(self, 'dialogPos'):
            self.humanClassifyDialog2.resize(self.dialogSize)
            self.humanClassifyDialog2.move(self.dialogPos)
        self.humanClassifyDialog2.finish.clicked.connect(self.humanClassifyClose2)
        self.humanClassifyDialog2.setModal(True)
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
        # (on load, shortBirdList is copied over from config, and if that fails - can't start anything)
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
        self.box1id = -1
        if hasattr(self, 'dialogPos'):
            self.humanClassifyDialog1.resize(self.dialogSize)
            self.humanClassifyDialog1.move(self.dialogPos)
        self.humanClassifyDialog1.setWindowTitle("AviaNZ - reviewing " + self.filename)
        self.humanClassifyNextImage1()
        # connect listeners
        self.humanClassifyDialog1.correct.clicked.connect(self.humanClassifyCorrect1)
        self.humanClassifyDialog1.delete.clicked.connect(self.humanClassifyDelete1)
        self.humanClassifyDialog1.buttonPrev.clicked.connect(self.humanClassifyPrevImage)
        success = self.humanClassifyDialog1.exec_() # 1 on clean exit

        if success == 0:
            self.humanClassifyDialog1.stopPlayback()
            return(0)

        return(1)

    def loadFile(self, filename):
        with pg.BusyCursor():
            with pg.ProgressDialog("Loading file...", 0, 4) as dlg:
                dlg.setCancelButton(None)
                dlg.setWindowIcon(QIcon('img/Avianz.ico'))
                dlg.setWindowTitle('AviaNZ')
                dlg.setFixedSize(350, 100)
                dlg.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
                dlg.update()
                dlg.repaint()
                dlg.show()
                # Create an instance of the Signal Processing class
                if not hasattr(self,'sp'):
                    self.sp = SignalProc.SignalProc(self.config['window_width'], self.config['incr'])
                self.sp.readWav(filename)
                dlg += 1
                dlg.update()
                dlg.repaint()

                self.sampleRate = self.sp.sampleRate
                self.audiodata = self.sp.data

                self.datalength = np.shape(self.audiodata)[0]
                print("Length of file is ",len(self.audiodata),float(self.datalength)/self.sampleRate,self.sampleRate)

                # Filter the audiodata based on initial sliders
                minFreq = max(self.fLow.value(), 0)
                maxFreq = min(self.fHigh.value(), self.sampleRate//2)
                if maxFreq - minFreq < 100:
                    print("ERROR: less than 100 Hz band set for spectrogram")
                    return
                print("Filtering samples to %d - %d Hz" % (minFreq, maxFreq))
                self.sp.data = self.sp.ButterworthBandpass(self.audiodata, self.sampleRate, minFreq, maxFreq)
                self.audiodata = self.sp.data
                dlg += 1
                dlg.update()
                dlg.repaint()

                # Get the data for the spectrogram
                self.sgRaw = self.sp.spectrogram(window='Hann', mean_normalise=True, onesided=True,multitaper=False, need_even=False)
                dlg += 1
                dlg.update()
                dlg.repaint()
                maxsg = np.min(self.sgRaw)
                self.sg = np.abs(np.where(self.sgRaw==0, 0.0, 10.0 * np.log10(self.sgRaw/maxsg)))
                self.setColourMap()

                # trim the spectrogram
                height = self.sampleRate//2 / np.shape(self.sg)[1]
                pixelstart = int(minFreq/height)
                pixelend = int(maxFreq/height)
                self.sg = self.sg[:,pixelstart:pixelend]
                dlg += 1
                dlg.update()
                dlg.repaint()

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
        self.humanClassifyDialog1.stopPlayback()

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
