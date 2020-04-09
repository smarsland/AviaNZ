
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
import os, re, fnmatch, sys, gc, math

from PyQt5.QtGui import QIcon, QPixmap, QColor
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QLabel, QPlainTextEdit, QPushButton, QTimeEdit, QSpinBox, QListWidget, QDesktopWidget, QApplication, QComboBox, QLineEdit, QSlider, QListWidgetItem, QCheckBox, QGroupBox, QFormLayout, QGridLayout, QHBoxLayout, QVBoxLayout, QFrame
from PyQt5.QtCore import Qt, QDir

import numpy as np
import wavio

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
import copy


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

        self.w_browse = QPushButton("  &Browse Folder")
        self.w_browse.setToolTip("Can select a folder with sub folders to process")
        self.w_browse.setFixedSize(165, 50)
        self.w_browse.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogOpenButton))
        self.w_browse.setStyleSheet('QPushButton {font-weight: bold; font-size:14px; padding: 3px 3px 3px 3px}')
        self.w_dir = QPlainTextEdit()
        self.w_dir.setFixedHeight(50)
        self.w_dir.setReadOnly(True)
        self.w_dir.setPlainText('')
        self.w_dir.setToolTip("The folder being processed")
        self.w_dir.setStyleSheet("color : #808080;")

        w_speLabel1 = QLabel("Select one or more recognisers to use:")
        self.w_spe1 = QComboBox()
        self.speCombos = [self.w_spe1]

        # populate this box (always show all filters here)
        spp = list(self.FilterDicts.keys())
        self.w_spe1.addItems(spp)
        self.w_spe1.addItem("Any sound")
        self.w_spe1.addItem("Any sound (Intermittent sampling)")
        self.w_spe1.currentTextChanged.connect(self.fillSpeciesBoxes)
        self.addSp = QPushButton("Add another recogniser")
        self.addSp.clicked.connect(self.addSpeciesBox)

        self.w_resLabel = QLabel("Set size of presence/absence blocks in Excel output\n(Sheet 3)")
        self.w_res = QSpinBox()
        self.w_res.setRange(1, 600)
        self.w_res.setSingleStep(5)
        self.w_res.setValue(60)
        w_timeLabel = QLabel("Want to process a subset of recordings only e.g. dawn or dusk?\nThen select the time window, otherwise skip")
        self.w_timeStart = QTimeEdit()
        self.w_timeStart.setDisplayFormat('hh:mm:ss')
        self.w_timeEnd = QTimeEdit()
        self.w_timeEnd.setDisplayFormat('hh:mm:ss')

        self.w_wind = QCheckBox("Add wind filter")
        self.w_mergect = QCheckBox("Merge different call types")

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

        self.w_processButton = QPushButton("  &Process Folder")
        self.w_processButton.setStyleSheet('QPushButton {font-weight: bold; font-size:14px; padding: 2px 2px 2px 8px}')
        self.w_processButton.setIcon(QIcon(QPixmap('img/process.png')))
        self.w_processButton.clicked.connect(self.detect)
        self.w_processButton.setFixedSize(165, 50)
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
        self.boxTime = QGroupBox()
        formTime = QGridLayout()
        formTime.addWidget(w_timeLabel, 0, 0, 1, 2)
        formTime.addWidget(QLabel("Start time (hh:mm:ss)"), 1, 0)
        formTime.addWidget(self.w_timeStart, 1, 1)
        formTime.addWidget(QLabel("End time (hh:mm:ss)"), 2, 0)
        formTime.addWidget(self.w_timeEnd, 2, 1)
        self.boxTime.setLayout(formTime)
        self.d_detection.addWidget(self.boxTime, row=2, col=0, colspan=3)

        self.warning = QLabel("Warning!\nThe chosen \"Any sound\" mode will delete ALL the existing annotations\nin the above selected folder")
        self.warning.setStyleSheet('QLabel {font-size:14px; color:red;}')
        self.d_detection.addWidget(self.warning, row=3, col=0, colspan=3)
        self.warning.hide()

        # Post Proc checkbox group
        self.boxPost = QGroupBox("Post processing")
        formPost = QGridLayout()
        formPost.addWidget(self.w_wind, 0, 1)
        formPost.addWidget(self.w_mergect, 2, 1)
        formPost.addWidget(self.maxgaplbl, 3, 0)
        formPost.addWidget(self.maxgap, 3, 1)
        formPost.addWidget(self.minlenlbl, 4, 0)
        formPost.addWidget(self.minlen, 4, 1)
        formPost.addWidget(self.maxlenlbl, 5, 0)
        formPost.addWidget(self.maxlen, 5, 1)
        self.boxPost.setLayout(formPost)
        self.d_detection.addWidget(self.boxPost, row=4, col=0, colspan=3)
        if len(spp) > 0:
            self.maxgaplbl.hide()
            self.maxgap.hide()
            self.minlenlbl.hide()
            self.minlen.hide()
            self.maxlenlbl.hide()
            self.maxlen.hide()

        self.lblsec = QLabel("(seconds)")
        self.d_detection.addWidget(self.w_resLabel, row=5, col=0)
        self.d_detection.addWidget(self.w_res, row=5, col=1)
        self.d_detection.addWidget(self.lblsec, row=5, col=2)
        self.d_detection.addWidget(self.w_processButton, row=6, col=2)

        self.w_files = pg.LayoutWidget()
        self.d_files.addWidget(self.w_files)

        # List to hold the list of files
        colourNone = QColor(self.config['ColourNone'][0], self.config['ColourNone'][1], self.config['ColourNone'][2], self.config['ColourNone'][3])
        colourPossibleDark = QColor(self.config['ColourPossible'][0], self.config['ColourPossible'][1], self.config['ColourPossible'][2], 255)
        colourNamed = QColor(self.config['ColourNamed'][0], self.config['ColourNamed'][1], self.config['ColourNamed'][2], self.config['ColourNamed'][3])
        self.listFiles = SupportClasses.LightedFileList(colourNone, colourPossibleDark, colourNamed)
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
        self.w_dir.setPlainText(self.dirName)
        self.w_dir.setReadOnly(True)
        self.fillFileList()

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
        if currname not in ["Any sound", "Any sound (Intermittent sampling)"]:
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
            self.boxPost.show()
            self.boxTime.show()
            self.addSp.show()
            self.warning.hide()
            self.w_resLabel.hide()
            self.w_res.hide()
            self.lblsec.hide()
        elif currname != "Any sound (Intermittent sampling)":
            self.minlen.show()
            self.minlenlbl.show()
            self.maxlen.show()
            self.maxlenlbl.show()
            self.maxgap.show()
            self.maxgaplbl.show()
            self.w_mergect.hide()
            self.boxPost.show()
            self.boxTime.show()
            self.addSp.hide()
            self.warning.show()
            self.w_resLabel.hide()
            self.w_res.hide()
            self.lblsec.hide()
        else:
            self.boxPost.hide()
            self.boxTime.hide()
            self.addSp.hide()
            self.warning.show()
            self.w_resLabel.hide()
            self.w_res.hide()
            self.lblsec.hide()

        # (skip first box which is fixed)
        for box in self.speCombos[1:]:
            # clear old items:
            for i in reversed(range(box.count())):
                box.removeItem(i)
            box.setCurrentIndex(-1)
            box.setCurrentText("")

            box.addItems(spp)

    def addRegularSegments(self, wav):
        """ Perform the Hartley bodge: add 10s segments every minute. """
        # if wav.data exists get the duration
        self.filename = wav
        (rate, nseconds, nchannels, sampwidth) = wavio.readFmt(self.filename)
        self.segments = Segment.SegmentList()
        self.segments.metadata = dict()
        self.segments.metadata["Operator"] = "Auto"
        self.segments.metadata["Reviewer"] = ""
        self.segments.metadata["Duration"] = nseconds
        i = 0
        segments = []
        print("Adding segments (%d s every %d s) to %s" %(self.config['protocolSize'], self.config['protocolInterval'], str(self.filename)))
        while i < nseconds:
            segments.append([i, i + self.config['protocolSize']])
            i += self.config['protocolInterval']
        post = Segment.PostProcess(audioData=None, sampleRate=0, segments=segments, subfilter={}, cert=0)
        self.makeSegments(post.segments)

    def useWindF(self, flow, fhigh):
        """
        Check if the wind filter is appropriate for this species/call type.
        Return true if wind filter target band 50-500 Hz does not overlap with flow-fhigh Hz.
        """
        if 50 < fhigh and 500 > flow:
            print('Skipping wind filter...')
            return False
        else:
            return True

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
        elif "Any sound (Intermittent sampling)" in self.species:
            self.method = "Intermittent sampling"
            speciesStr = "Intermittent sampling"
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

            # load target CNN models (currently storing in the same dir as filters)
            # format: {filtername: [model, win, inputdim, output]}
            self.CNNDicts = self.ConfigLoader.CNNmodels(self.FilterDicts, self.filtersDir, self.species)

        # LIST ALL WAV files that will be processed
        allwavs = []
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                if filename.lower().endswith('.wav'):
                    allwavs.append(os.path.join(root, filename))
        total = len(allwavs)
        if self.method != "Intermittent sampling":
            # Parse the user-set time window to process
            timeWindow_s = self.w_timeStart.time().hour() * 3600 + self.w_timeStart.time().minute() * 60 + self.w_timeStart.time().second()
            timeWindow_e = self.w_timeEnd.time().hour() * 3600 + self.w_timeEnd.time().minute() * 60 + self.w_timeEnd.time().second()
            # LOG FILE is read here
            # note: important to log all analysis settings here
            settings = [self.method, self.w_res.value(), timeWindow_s, timeWindow_e,
                        self.w_wind.isChecked(), self.w_mergect.isChecked()]
        else:
            settings = [self.method, self.config["protocolSize"], self.config["protocolInterval"]]

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
        if self.method == "Intermittent sampling":
            text = "Method: " + self.method + ".\nNumber of files to analyze: " + str(total) + "\n"
        else:
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
        if speciesStr not in ["Any sound", "Intermittent sampling"]:
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
        msgtext = ""
        # clean up the UI before entering the long loop
        self.update()
        self.repaint()
        QtGui.QApplication.processEvents()
        if self.method == "Intermittent sampling":
            with pg.BusyCursor():
                for filename in allwavs:
                    # get remaining run time in min
                    processingTimeStart = time.time()
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

                    # check if file is formatted correctly
                    with open(filename, 'br') as f:
                        if f.read(4) != b'RIFF':
                            print("Warning: file %s not formatted correctly, skipping" % filename)
                            continue

                    # actual processing
                    self.addRegularSegments(filename)

                    print("%d intermittent segments marked" % len(self.segments))

                    # export segments
                    cleanexit = self.saveAnnotation()
                    if cleanexit != 1:
                        print("Warning: could not save segments!")
                    # Log success for this file
                    self.log.appendFile(filename)

                    # track how long it took to process one file:
                    processingTime = time.time() - processingTimeStart
                    print("File processed in", processingTime)
        else:
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

                    # check if file is formatted correctly
                    with open(filename, 'br') as f:
                        if f.read(4) != b'RIFF':
                            print("Warning: file %s not formatted correctly, skipping" % filename)
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
                            post = Segment.PostProcess(audioData=self.audiodata[start:end], sampleRate=self.sampleRate, segments=thisPageSegs, subfilter={}, cert=0)
                            if self.w_wind.isChecked():
                                post.wind()
                            post.joinGaps(maxgap)
                            post.deleteShort(minlen)
                            # avoid extra long segments (for Isabel)
                            post.splitLong(maxlen)

                            # adjust segment starts for 15min "pages"
                            if start != 0:
                                for seg in post.segments:
                                    seg[0][0] += start/self.sampleRate
                                    seg[0][1] += start/self.sampleRate
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
                                print("Segments detected (all subfilters): ", thisPageSegs)
                                print("Post-processing...")
                                # postProcess currently operates on single-level list of segments,
                                # so we run it over subfilters for wavelets:
                                spInfo = filters[speciesix]
                                for filtix in range(len(spInfo['Filters'])):
                                    CNNmodel = None
                                    if spInfo['species'] in self.CNNDicts.keys():
                                        CNNmodel = self.CNNDicts[spInfo['species']]
                                    post = Segment.PostProcess(audioData=self.audiodata[start:end], sampleRate=self.sampleRate, tgtsampleRate=spInfo["SampleRate"], segments=thisPageSegs[filtix], subfilter=spInfo['Filters'][filtix], CNNmodel=CNNmodel, cert=50)
                                    print("Segments detected after WF: ", len(thisPageSegs[filtix]))
                                    if self.w_wind.isChecked() and self.useWindF(spInfo['Filters'][filtix]['FreqRange'][0], spInfo['Filters'][filtix]['FreqRange'][1]):
                                        post.wind()
                                    if CNNmodel:
                                        print('Post-processing with CNN')
                                        post.CNN()
                                    if 'F0' in spInfo['Filters'][filtix] and 'F0Range' in spInfo['Filters'][filtix]:
                                        if spInfo['Filters'][filtix]["F0"]:
                                            print("Checking for fundamental frequency...")
                                            post.fundamentalFrq()

                                    post.joinGaps(maxgap=spInfo['Filters'][filtix]['TimeRange'][3])
                                    post.deleteShort(minlength=spInfo['Filters'][filtix]['TimeRange'][0])

                                    # adjust segment starts for 15min "pages"
                                    if start != 0:
                                        for seg in post.segments:
                                            seg[0][0] += start/self.sampleRate
                                            seg[0][1] += start/self.sampleRate

                                    if self.w_mergect.isChecked():
                                        # collect segments from all call types
                                        allCtSegs.extend(post.segments)
                                    else:
                                        # attach filter info and put on self.segments:
                                        self.makeSegments(post.segments, self.species[speciesix], spInfo["species"], spInfo['Filters'][filtix])

                                if self.w_mergect.isChecked():
                                    # merge different call type segments
                                    post.segments = allCtSegs
                                    post.checkSegmentOverlap()

                                    # also merge neighbours (segments from different call types)
                                    post.joinGaps(maxgap=max([subf['TimeRange'][3] for subf in spInfo["Filters"]]))
                                    # construct "Any call" info to place on the segments
                                    flow = min([subf["FreqRange"][0] for subf in spInfo["Filters"]])
                                    fhigh = max([subf["FreqRange"][1] for subf in spInfo["Filters"]])
                                    ctinfo = {"calltype": "(Other)", "FreqRange": [flow, fhigh]}
                                    print('self.species[speciesix]:', self.species[speciesix])
                                    print('spInfo["species"]:', spInfo["species"])
                                    self.makeSegments(post.segments, self.species[speciesix], spInfo["species"], ctinfo)

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
                self.statusBar().showMessage("Removing old Excel files, almost done...")
                self.update()
                self.repaint()
                for root, dirs, files in os.walk(str(self.dirName)):
                    for filename in files:
                        filenamef = os.path.join(root, filename)
                        if fnmatch.fnmatch(filenamef, '*DetectionSummary_*.xlsx'):
                            print("Removing excel file %s" % filenamef)
                            os.remove(filenamef)
                # We currently do not export any excels automatically
                # in this mode. We only delete old excels, and let
                # user generate new ones through Batch Review.

        # END of processing and exporting. Final cleanup
        self.log.file.close()
        self.statusBar().showMessage("Processed all %d files" % total)
        msgtext = "Finished processing.\nWould you like to return to the start screen?"
        msg = SupportClasses.MessagePopup("d", "Finished", msgtext)
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
            for s in segmentsNew:
                segment = Segment.Segment([s[0][0], s[0][1], y1, y2, [{"species": species, "certainty": s[1], "filter": filtName, "calltype": subfilter["calltype"]}]])
                self.segments.addSegment(segment)
        # for generic all-species segments:
        else:
            y1 = 0
            y2 = 0
            species = "Don't Know"
            cert = 0.0
            self.segments.addBasicSegments(segmentsNew, [y1, y2], species=species, certainty=cert)

    def saveAnnotation(self):
        """ Generates default batch-mode metadata,
            and saves the current self.segments to a .data file. """

        self.segments.metadata["Operator"] = "Auto"
        self.segments.metadata["Reviewer"] = ""
        if self.method != "Intermittent sampling":
            self.segments.metadata["Duration"] = float(self.datalength)/self.sampleRate
        self.segments.metadata["noiseLevel"] = None
        self.segments.metadata["noiseTypes"] = []

        self.segments.saveJSON(str(self.filename) + '.data')

        return 1

    def fillFileList(self, fileName=None):
        """ Generates the list of files for the file listbox.
        Most of the work is to deal with directories in that list.
        It only sees *.wav files. Picks up *.data files, to make the filenames
        red in the list."""

        if not os.path.isdir(self.dirName):
            print("ERROR: directory %s doesn't exist" % self.soundFileDir)
            return

        self.listFiles.fill(self.dirName, fileName)

        # update the "Browse" field text
        self.w_dir.setPlainText(self.dirName)

    def listLoadFile(self,current):
        """ Listener for when the user clicks on an item in filelist
        """

        # Need name of file
        if type(current) is QListWidgetItem:
            current = current.text()
            current = re.sub('\/.*', '', current)

        self.previousFile = current

        # Update the file list to show the right one
        i=0
        lof = self.listFiles.listOfFiles
        while i<len(lof)-1 and lof[i].fileName() != current:
            i+=1
        if lof[i].isDir() or (i == len(lof)-1 and lof[i].fileName() != current):
            dir = QDir(self.dirName)
            dir.cd(lof[i].fileName())
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
        if anysound:
            self.sp.data = self.sp.impMask(engp=70, fp=0.50)
        else:
            self.sp.data = self.sp.impMask()
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
        self.setFixedSize(900, 600)
        self.setWindowIcon(QIcon('img/Avianz.ico'))

        # Make the docks
        self.d_detection = Dock("Review",size=(600, 600))
        # self.d_detection.hideTitleBar()
        self.d_files = Dock("File list", size=(300, 600))

        self.area.addDock(self.d_detection, 'right')
        self.area.addDock(self.d_files, 'left')

        self.w_revLabel = QLabel("Reviewer")
        self.w_reviewer = QLineEdit()
        self.d_detection.addWidget(self.w_revLabel, row=0, col=0)
        self.d_detection.addWidget(self.w_reviewer, row=0, col=1, colspan=2)
        self.w_browse = QPushButton("  &Browse Folder")
        self.w_browse.setToolTip("Can select a folder with sub folders to process")
        self.w_browse.setFixedHeight(50)
        self.w_browse.setStyleSheet('QPushButton {font-weight: bold; font-size:14px}')
        self.w_browse.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogOpenButton))
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
        self.d_detection.addWidget(self.fLowtext, row=4, col=0)
        self.d_detection.addWidget(self.fLow, row=4, col=1)
        self.d_detection.addWidget(self.fLowvalue, row=4, col=2)
        self.d_detection.addWidget(self.fHightext, row=5, col=0)
        self.d_detection.addWidget(self.fHigh, row=5, col=1)
        self.d_detection.addWidget(self.fHighvalue, row=5, col=2)

        self.w_processButton = QPushButton("  &Review Folder")
        self.w_processButton.setStyleSheet('QPushButton {font-weight: bold; font-size:14px; padding: 2px 2px 2px 8px}')
        self.w_processButton.setFixedHeight(45)
        self.w_processButton.setFixedHeight(45)
        self.w_processButton.setIcon(QIcon(QPixmap('img/review.png')))
        self.w_processButton.clicked.connect(self.review)
        self.d_detection.addWidget(self.w_processButton, row=10, col=2)

        # Excel export section
        linesep = QFrame()
        linesep.setFrameShape(QFrame.HLine)
        linesep.setFrameShadow(QFrame.Sunken)
        self.d_detection.addWidget(linesep, row=11, col=0, colspan=3)
        self.w_resLabel = QLabel("Size (s) of presence/absence\nwindows in the output")
        self.d_detection.addWidget(self.w_resLabel, row=13, col=0)
        self.w_res = QSpinBox()
        self.w_res.setRange(1,600)
        self.w_res.setSingleStep(5)
        self.w_res.setValue(60)
        self.d_detection.addWidget(self.w_res, row=13, col=1)

        w_excelButton = QPushButton(" Generate Excel  ")
        w_excelButton.setStyleSheet('QPushButton {font-weight: bold; font-size:14px; padding: 2px 2px 2px 8px}')
        w_excelButton.setFixedHeight(45)
        w_excelButton.setIcon(QIcon(QPixmap('img/excel.png')))
        w_excelButton.clicked.connect(self.exportExcel)
        self.d_detection.addWidget(w_excelButton, row=13, col=2)

        self.w_browse.clicked.connect(self.browse)
        # print("spList after browse: ", self.spList)

        self.w_files = pg.LayoutWidget()
        self.d_files.addWidget(self.w_files)
        self.w_files.addWidget(QLabel('Double click to select a folder'), row=0, col=0)
        self.w_files.addWidget(QLabel('Red files have annotations'), row=1, col=0)

        # List to hold the list of files
        colourNone = QColor(self.config['ColourNone'][0], self.config['ColourNone'][1], self.config['ColourNone'][2], self.config['ColourNone'][3])
        colourPossibleDark = QColor(self.config['ColourPossible'][0], self.config['ColourPossible'][1], self.config['ColourPossible'][2], 255)
        colourNamed = QColor(self.config['ColourNamed'][0], self.config['ColourNamed'][1], self.config['ColourNamed'][2], self.config['ColourNamed'][3])
        self.listFiles = SupportClasses.LightedFileList(colourNone, colourPossibleDark, colourNamed)
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
        if self.dirName:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirName))
        else:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process')
        self.w_dir.setPlainText(self.dirName)
        self.w_dir.setReadOnly(True)
        # this will also collect some info about the dir
        self.fillFileList()

        # find species names from the annotations
        self.spList = list(self.listFiles.spList)
        # Can't review only "Don't Knows". Ideally this should call AllSpecies dialog tho
        try:
            self.spList.remove("Don't Know")
        except Exception:
            pass
        self.spList.insert(0, 'Any sound')
        self.w_spe1.clear()
        self.w_spe1.addItems(self.spList)

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
        self.sps = []
        msgtext = ""
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

            # check if file is formatted correctly
            with open(filename, 'br') as f:
                if f.read(4) != b'RIFF':
                    print("Warning: file %s not formatted correctly, skipping" % filename)
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

            # skip review dialog if there's no segments passing relevant criteria
            # (self.segments will have all species even if only one is being reviewed)
            if len(self.segments)==0 or self.species!='Any sound' and len(self.segments.getSpecies(self.species))==0:
                print("No segments found in file %s" % filename)
                filesuccess = 1
                continue

            # file has >=1 segments to review,
            # so call the right dialog:
            # (they will update self.segments and store corrections)
            if self.species == 'Any sound':
                filesuccess = self.review_all(filename, sTime)
            else:
                # split long segments for single species review
                self.segments.splitLongSeg(species=self.species)
                _ = self.segments.orderTime()
                print(self.segments)
                filesuccess = self.review_single(filename, sTime)

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
            self.statusBar().showMessage("Removing old Excel files, almost done...")
            self.update()
            self.repaint()
            for root, dirs, files in os.walk(str(self.dirName)):
                for filename in files:
                    filenamef = os.path.join(root, filename)
                    if fnmatch.fnmatch(filenamef, '*DetectionSummary_*.xlsx'):
                        print("Removing excel file %s" % filenamef)
                        os.remove(filenamef)

        self.statusBar().showMessage("Reviewed files " + str(cnt) + "/" + str(total))
        self.update()
        self.repaint()

        # END of review and exporting. Final cleanup
        self.ConfigLoader.configwrite(self.config, self.configfile)
        if filesuccess == 1:
            msgtext = "All files checked.\nWould you like to return to the start screen?"
            msg = SupportClasses.MessagePopup("d", "Finished", msgtext)
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            reply = msg.exec_()
            if reply == QMessageBox.Yes:
                QApplication.exit(1)
        else:
            msgtext = "Review stopped at file %s of %s.\nWould you like to return to the start screen?" % (cnt, total)
            msg = SupportClasses.MessagePopup("w", "Review stopped", msgtext)
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            reply = msg.exec_()
            if reply == QMessageBox.Yes:
                QApplication.exit(1)

    def exportExcel(self):
        """ Launched manually by pressing the button.
            Cleans out old excels and creates a single new one.
            Needs set self.species, self.dirName. """

        self.species = self.w_spe1.currentText()
        if self.dirName == '':
            msg = SupportClasses.MessagePopup("w", "Select Folder", "Please select a folder to process!")
            msg.exec_()
            return

        with pg.BusyCursor():
            # delete old results (xlsx)
            # ! WARNING: any Detection...xlsx files will be DELETED,
            # ! ANYWHERE INSIDE the specified dir, recursively
            self.statusBar().showMessage("Removing old Excel files...")
            self.update()
            self.repaint()
            for root, dirs, files in os.walk(str(self.dirName)):
                for filename in files:
                    filenamef = os.path.join(root, filename)
                    if fnmatch.fnmatch(filenamef, '*DetectionSummary_*.xlsx'):
                        print("Removing excel file %s" % filenamef)
                        os.remove(filenamef)

        print("Exporting to Excel ...")
        self.statusBar().showMessage("Exporting to Excel ...")
        self.update()
        self.repaint()

        allsegs = []
        # Note: one excel will always be generated for the currently selected species
        spList = set([self.species])

        # list all DATA files that can be processed
        alldatas = []
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                print(filename)
                if filename.endswith('.data'):
                    print("Appending" ,filename)
                    filenamef = os.path.join(root, filename)
                    alldatas.append(filenamef)

        with pg.BusyCursor():
            for filename in alldatas:
                print("Reading segments from", filename)
                segments = Segment.SegmentList()
                segments.parseJSON(filename)

                # Determine all species detected in at least one file
                for seg in segments:
                    spList.update([lab["species"] for lab in seg[4]])

                # sort by time and save
                segments.orderTime()
                # attach filename to be stored in Excel later
                segments.filename = filename

                # Collect all .data contents (as SegmentList objects)
                # for the Excel output (no matter if review dialog exit was clean)
                allsegs.append(segments)

            # Export the actual Excel
            excel = SupportClasses.ExcelIO()
            excsuccess = excel.export(allsegs, self.dirName, "overwrite", resolution=self.w_res.value(), speciesList=list(spList))

        if excsuccess!=1:
            # if any file wasn't exported well, overwrite the message
            msgtext = "Warning: Excel output at " + self.dirName + " was not stored properly"
            print(msgtext)
            msg = SupportClasses.MessagePopup("w", "Failed to export Excel file", msgtext)
        else:
            msgtext = "Excel output is stored in " + os.path.join(self.dirName, "DetectionSummary_*.xlsx")
            msg = SupportClasses.MessagePopup("d", "Excel output produced", msgtext)
        msg.exec_()

    def review_single(self, filename, sTime):
        """ Initializes single species dialog, based on self.species
            (thus we don't need the small species choice dialog here).
            Updates self.segments as a side effect.
            Returns 1 for clean completion, 0 for Esc press or other dirty exit.
        """
        # Initialize the dialog for this file
        self.loadFile(filename, self.species)
        self.humanClassifyDialog2 = Dialogs.HumanClassify2(self.sps, self.segments, self.indices2show,
                                                           self.species, self.lut, self.colourStart,
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

    def mergeSplitSeg(self):
        # After segments are split, put them back if all are still there
        # Really simple -- assumes they are in order
        # SRM
        todelete = []
        last = [0,0,0,0,0]
        count=0
        for seg in self.segments:
            #print(math.isclose(seg[0],last[1]))
            # Merge the two segments if they abut and have the same species
            print(seg, seg[0], last[4])
            if math.isclose(seg[0],last[1]) and seg[4] == last[4]:
                last[1] = seg[1]
                todelete.append(count)
            else:
                last = seg
            count+=1

        print(todelete)
        for dl in reversed(todelete):
            del self.segments[dl]

        print(self.segments)

    def species2clean(self, species):
        """ Returns True when the species name got a special character"""
        search = re.compile(r'[^A-Za-z0-9()-]').search
        return bool(search(species))

    def cleanSpecies(self):
        """ Returns cleaned species name"""
        return re.sub(r'[^A-Za-z0-9()-]', "_", self.species)

    def saveCorrectJSON(self, file, outputErrors, mode, reviewer=""):
        """ Returns 1 on succesful save.
        Mode 1. Any Species Review saves .correction. Format [meta, [seg1, newlabel1], [seg2, newlabel2],...]
        Mode 2. Single Species Review saves .correction_species. Format [meta, seg1, seg2,...]"""
        if reviewer != "":
            self.segments.metadata["Reviewer"] = reviewer
        annots = [self.segments.metadata]

        if os.path.isfile(file):
            try:
                f = open(file, 'r')
                annotsold = json.load(f)
                f.close()
                for elem in annotsold:
                    if not isinstance(elem, dict):
                        annots.append(elem)
            except Exception as e:
                print("ERROR: file %s failed to load with error:" % file)
                print(e)
                return

        if mode == 1:
            if outputErrors[0] not in annots:
                annots.append(outputErrors[0])
        elif mode == 2:
            for seg in outputErrors:
                if seg not in annots:
                    annots.append(seg)

        file = open(file, 'w')
        json.dump(annots, file)
        file.write("\n")
        file.close()
        return 1

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
                cSeg = copy.deepcopy(currSeg)
                outputErrors.append(cSeg)
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

        # store position etc to carry over to the next file dialog
        self.dialogSize = self.humanClassifyDialog2.size()
        self.dialogPos = self.humanClassifyDialog2.pos()
        self.config['brightness'] = self.humanClassifyDialog2.brightnessSlider.value()
        self.config['contrast'] = self.humanClassifyDialog2.contrastSlider.value()
        if not self.config['invertColourMap']:
            self.config['brightness'] = 100-self.config['brightness']
        self.humanClassifyDialog2.done(1)

        # Save the errors in a file
        if self.config['saveCorrections'] and len(outputErrors)>0:
            if self.species2clean(self.species):
                speciesClean = self.cleanSpecies()
            else:
                speciesClean = self.species
            cleanexit = self.saveCorrectJSON(str(self.filename + '.corrections_' + speciesClean), outputErrors, mode=2, reviewer=self.reviewer)
            if cleanexit != 1:
                print("Warning: could not save correction file!")

        # reverse loop to allow deleting segments
        for dl in reversed(list(set(todelete))):
            del self.segments[dl]

        self.mergeSplitSeg()
        # done - the segments will be saved by the main loop
        return

    def review_all(self, filename, sTime, minLen=5):
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

        self.loadFile(filename)
        # HumanClassify1 reads audioFormat from parent.sp.audioFormat, so need this:
        self.humanClassifyDialog1 = Dialogs.HumanClassify1(self.lut,self.colourStart,self.colourEnd,self.config['invertColourMap'], self.config['brightness'], self.config['contrast'], self.shortBirdList, self.longBirdList, self.config['MultipleSpecies'], self.sps[0].audioFormat, self)
        self.box1id = -1
        # if there was a previous dialog, try to recreate its settings
        if hasattr(self, 'dialogPos'):
            self.humanClassifyDialog1.resize(self.dialogSize)
            self.humanClassifyDialog1.move(self.dialogPos)
        if hasattr(self, 'dialogPlotAspect'):
            self.humanClassifyDialog1.plotAspect = self.dialogPlotAspect
            self.humanClassifyDialog1.pPlot.setAspectLocked(ratio=self.dialogPlotAspect)
        self.humanClassifyDialog1.setWindowTitle("AviaNZ - reviewing " + self.filename)
        self.humanClassifyNextImage1()
        # connect listeners
        self.humanClassifyDialog1.correct.clicked.connect(self.humanClassifyCorrect1)
        self.humanClassifyDialog1.delete.clicked.connect(self.humanClassifyDelete1)
        self.humanClassifyDialog1.buttonPrev.clicked.connect(self.humanClassifyPrevImage)
        self.humanClassifyDialog1.buttonNext.clicked.connect(self.humanClassifyQuestion)
        success = self.humanClassifyDialog1.exec_() # 1 on clean exit

        if success == 0:
            self.humanClassifyDialog1.stopPlayback()
            return(0)

        return(1)

    def loadFile(self, filename, species=None):
        """ Needs to generate spectrograms and audiodatas
            for each segment in self.segments.
            The SignalProcs containing these are loaded into self.sps.
        """
        with pg.BusyCursor():
            # delete old instances to force release memory
            for sp in reversed(range(len(self.sps))):
                del self.sps[sp]
            minsg = 1
            maxsg = 1
            gc.collect()

            with pg.ProgressDialog("Loading file...", 0, len(self.segments)) as dlg:
                dlg.setCancelButton(None)
                dlg.setWindowIcon(QIcon('img/Avianz.ico'))
                dlg.setWindowTitle('AviaNZ')
                dlg.setFixedSize(350, 100)
                dlg.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
                dlg.update()
                dlg.repaint()
                dlg.show()

                # Determine the sample rate and set some file-level parameters
                samplerate, duration, _, _ = wavio.readFmt(filename)
                minFreq = max(self.fLow.value(), 0)
                maxFreq = min(self.fHigh.value(), samplerate//2)
                if maxFreq - minFreq < 100:
                    print("ERROR: less than 100 Hz band set for spectrogram")
                    return
                print("Filtering samples to %d - %d Hz" % (minFreq, maxFreq))

                # For single sp, no need to load all segments, but don't want to edit self.segments
                if species is not None:
                    self.indices2show = self.segments.getSpecies(species)
                else:
                    self.indices2show = range(len(self.segments))

                # Load data into a list of SignalProcs (with spectrograms) for each segment
                for segix in range(len(self.segments)):
                    if segix in self.indices2show:
                        seg = self.segments[segix]
                        # note that sp also stores the range of shown freqs
                        sp = SignalProc.SignalProc(self.config['window_width'], self.config['incr'], minFreq, maxFreq)

                        if species is not None:
                            mid = (seg[0]+seg[1])/2
                            # buffered limits in audiodata (sec) = display limits
                            x1 = max(0, mid-5)
                            x2 = min(duration, mid+5)

                            # unbuffered limits in audiodata
                            x1nob = max(seg[0], x1)
                            x2nob = min(seg[1], x2)
                        else:
                            # unbuffered limits in audiodata
                            x1nob = seg[0]
                            x2nob = seg[1]

                            # buffered limits in audiodata (sec) = display limits
                            x1 = max(x1nob - self.config['reviewSpecBuffer'], 0)
                            x2 = min(x2nob + self.config['reviewSpecBuffer'], duration)

                        # Load. segix>1 to print the format details only once for each file
                        sp.readWav(filename, off=x1, len=x2-x1, silent=segix>1)

                        # Filter the audiodata based on initial sliders
                        sp.data = sp.ButterworthBandpass(sp.data, sp.sampleRate, minFreq, maxFreq)

                        # need to also store unbuffered limits in spec units
                        # (relative to start of segment)
                        sp.x1nobspec = sp.convertAmpltoSpec(x1nob-x1)
                        sp.x2nobspec = sp.convertAmpltoSpec(x2nob-x1)

                        # Generate the spectrogram
                        _ = sp.spectrogram(window='Hann', mean_normalise=True, onesided=True,multitaper=False, need_even=False)

                        # collect min and max values for final colour scale
                        minsg = min(np.min(sp.sg), minsg)
                        maxsg = max(np.max(sp.sg), maxsg)
                        sp.sg = np.abs(np.where(sp.sg==0, 0.0, 10.0 * np.log10(sp.sg/minsg)))

                        # trim the spectrogram
                        height = sp.sampleRate//2 / np.shape(sp.sg)[1]
                        pixelstart = int(minFreq/height)
                        pixelend = int(maxFreq/height)
                        sp.sg = sp.sg[:,pixelstart:pixelend]
                    else:
                        sp = None

                    self.sps.append(sp)

                    dlg += 1
                    dlg.update()
                    dlg.repaint()

            # sets the color map, based on the extremes of all segment spectrograms
            cmap = self.config['cmap']
            pos, colour, mode = colourMaps.colourMaps(cmap)
            cmap = pg.ColorMap(pos, colour,mode)

            self.lut = cmap.getLookupTable(0.0, 1.0, 256)
            self.colourStart = (self.config['brightness'] / 100.0 * self.config['contrast'] / 100.0) * (maxsg - minsg) + minsg
            self.colourEnd = (maxsg - minsg) * (1.0 - self.config['contrast'] / 100.0) + self.colourStart

        # END of file loading

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

            # select the SignalProc with relevant data
            sp = self.sps[self.box1id]

            # get a list of all species names present
            specnames = []
            for lab in seg[4]:
                if 0<lab["certainty"]<100:
                    specnames.append(lab["species"]+'?')
                else:
                    specnames.append(lab["species"])
            specnames = list(set(specnames))

            # these pass the axis limits set by slider
            minFreq = max(self.fLow.value(), 0)
            maxFreq = min(self.fHigh.value(), sp.sampleRate//2)

            # specnames, then unbufstart in spec units rel to start, unbufend,
            # then true time to display start, end,
            self.humanClassifyDialog1.setImage(sp.sg, sp.data, sp.sampleRate, sp.incr,
                                               specnames, sp.x1nobspec, sp.x2nobspec,
                                               seg[0], seg[1], minFreq, maxFreq)
        else:
            # store dialog properties such as position for the next file
            self.dialogSize = self.humanClassifyDialog1.size()
            self.dialogPos = self.humanClassifyDialog1.pos()
            self.dialogPlotAspect = self.humanClassifyDialog1.plotAspect
            self.config['brightness'] = self.humanClassifyDialog1.brightnessSlider.value()
            self.config['contrast'] = self.humanClassifyDialog1.contrastSlider.value()
            if not self.config['invertColourMap']:
                self.config['brightness'] = 100-self.config['brightness']
            self.humanClassifyDialog1.done(1)

    def humanClassifyPrevImage(self):
        """ Go back one image by changing boxid and calling NextImage.
        Note: won't undo deleted segments."""
        if self.box1id>0:
            self.box1id -= 2
            self.humanClassifyNextImage1()

    def humanClassifyQuestion(self):
        """ Go to next image, keeping this one as it was found
            (so any changes made to it will be discarded, and cert kept) """
        self.humanClassifyDialog1.stopPlayback()
        self.segmentsToSave = True
        currSeg = self.segments[self.box1id]

        # update the actual segment.
        label, self.saveConfig, _ = self.humanClassifyDialog1.getValues()
        print("working on ", self.box1id, currSeg)

        # if any species names were changed,
        # save the correction file
        if label != [lab["species"] for lab in currSeg[4]]:
            if self.config['saveCorrections']:
                outputError = [[currSeg, label]]
                cleanexit = self.saveCorrectJSON(str(self.filename + '.corrections'), outputError, mode=1,
                                                 reviewer=self.reviewer)
                if cleanexit != 1:
                    print("Warning: could not save correction file!")

        # Then, just recreate the label with certainty 50 for all currently selected species:
        # (not very neat but safer)

        # Create new segment label, assigning certainty 50 for each species:
        newlabel = []
        for species in label:
            if species == "Don't Know":
                newlabel.append({"species": "Don't Know", "certainty": 0})
            else:
                newlabel.append({"species": species, "certainty": 50})
        self.segments[self.box1id] = Segment.Segment([currSeg[0], currSeg[1], currSeg[2], currSeg[3], newlabel])

        self.humanClassifyDialog1.tbox.setText('')
        self.humanClassifyDialog1.tbox.setEnabled(False)
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
                outputError = [[currSeg, label]]
                cleanexit = self.saveCorrectJSON(str(self.filename + '.corrections'), outputError, mode=1, reviewer=self.reviewer)
                if cleanexit != 1:
                    print("Warning: could not save correction file!")

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
        del self.sps[id]
        # self.indicestoshow then becomes incorrect, but we don't use that in here anyway

        self.box1id = id-1
        self.segmentsToSave = True
        self.humanClassifyNextImage1()

    def closeDialog(self, ev):
        # (actually a poorly named listener for the Esc key)
        if ev == Qt.Key_Escape and hasattr(self, 'humanClassifyDialog1'):
            self.humanClassifyDialog1.done(0)

    def fillFileList(self,fileName=None):
        """ Generates the list of files for the file listbox.
            fileName - currently opened file (marks it in the list).
        """

        if not os.path.isdir(self.dirName):
            print("ERROR: directory %s doesn't exist" % self.soundFileDir)
            return

        self.listFiles.fill(self.dirName, fileName)

        # update the "Browse" field text
        self.w_dir.setPlainText(self.dirName)

    def listLoadFile(self,current):
        """ Listener for when the user clicks on an item in filelist """

        # Need name of file
        if type(current) is QListWidgetItem:
            current = current.text()
            current = re.sub('\/.*', '', current)

        self.previousFile = current

        # Update the file list to show the right one
        i=0
        lof = self.listFiles.listOfFiles
        while i<len(lof)-1 and lof[i].fileName() != current:
            i+=1
        if lof[i].isDir() or (i == len(lof)-1 and lof[i].fileName() != current):
            dir = QDir(self.dirName)
            dir.cd(lof[i].fileName())
            # Now repopulate the listbox
            self.dirName=str(dir.absolutePath())
            self.previousFile = None
            self.fillFileList(current)
            # Show the selected file
            index = self.listFiles.findItems(os.path.basename(current), Qt.MatchExactly)
            if len(index) > 0:
                self.listFiles.setCurrentItem(index[0])
        return(0)
