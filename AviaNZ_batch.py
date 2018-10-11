import os, re, platform, fnmatch

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import QAudioFormat
from PyQt5.QtCore import Qt, QDir

import wavio
import librosa
import numpy as np

from pyqtgraph.Qt import QtGui
from pyqtgraph.dockarea import *
import pyqtgraph as pg

import SignalProc
import Segment
import WaveletSegment
import SupportClasses
import Dialogs

import json, copy

# sppInfo = {
#             # spp: [min_len, max_len, flow, fhigh, fs, f0_low, f0_high, wavelet_thr, wavelet_M, wavelet_nodes]
#             'Kiwi': [10, 30, 1100, 7000, 16000, 1200, 4200, 0.5, 0.6, [17, 20, 22, 35, 36, 38, 40, 42, 43, 44, 45, 46, 48, 50, 55, 56]],
#             'Gsk': [6, 25, 900, 7000, 16000, 1200, 4200, 0.25, 0.6, [35, 38, 43, 44, 52, 54]],
#             'Lsk': [10, 30, 1200, 7000, 16000, 1200, 4200,  0.25, 0.6, []], # todo: find len, f0, nodes
#             'Ruru': [1, 30, 500, 7000, 16000, 600, 1300,  0.25, 0.5, [33, 37, 38]], # find M
#             'SIPO': [1, 5, 1200, 3800, 8000, 1200, 3800,  0.25, 0.2, [61, 59, 54, 51, 60, 58, 49, 47]],  # find len, f0
#             'Bittern': [1, 5, 100, 200, 1000, 100, 200, 0.75, 0.2, [10,21,22,43,44,45,46]],  # find len, f0, confirm nodes
# }

class AviaNZ_batchProcess(QMainWindow):
    # Main class for batch processing

    def __init__(self,root=None,minSegment=50):
        # Allow the user to browse a folder and push a button to process that folder to find a target species
        # and sets up the window.
        super(AviaNZ_batchProcess, self).__init__()
        self.root = root
        self.dirName=[]

        try:
            self.FilterFiles = [f[:-4] for f in os.listdir('Filters') if os.path.isfile(os.path.join('Filters', f))]
        except:
            "Folder not found, no filters loaded"
            self.FilterFiles = None

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
        self.setFixedSize(800,500)

        # Make the docks
        self.d_detection = Dock("Automatic Detection",size=(500,500))
        # self.d_detection.hideTitleBar()

        self.d_files = Dock("File list", size=(270, 500))

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
        # print(self.sppInfo)


        spp = [*self.FilterFiles]
        # spp = []
        spp.insert(0, "All species")
        self.w_spe1.addItems(spp)
        # self.w_spe1.addItems(["Kiwi", "Ruru", "Bittern", "all"])
        self.d_detection.addWidget(self.w_spe1,row=1,col=1,colspan=2)

        self.w_resLabel = QLabel("  Output Resolution (secs)")
        self.d_detection.addWidget(self.w_resLabel, row=2, col=0)
        self.w_res = QSpinBox()
        self.w_res.setRange(1,600)
        self.w_res.setSingleStep(5)
        self.w_res.setValue(60)
        self.d_detection.addWidget(self.w_res, row=2, col=1, colspan=2)

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
        """ Create the About Message Box"""
        msg = QMessageBox()
        msg.setIconPixmap(QPixmap("img\AviaNZ.png"))
        msg.setWindowIcon(QIcon('img/Avianz.ico'))
        msg.setText("The AviaNZ Program, v1.1 (August 2018)")
        msg.setInformativeText("By Stephen Marsland, Victoria University of Wellington. With code by Nirosha Priyadarshani and Julius Juodakis, and input from Isabel Castro, Moira Pryde, Stuart Cockburn, Rebecca Stirnemann, Sumudu Purage, Virginia Listanti, and Rebecca Huistra. \n stephen.marsland@vuw.ac.nz")
        msg.setWindowTitle("About")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        return

    def showHelp(self):
        """ Show the user manual (a pdf file)"""
        # TODO: manual is not distributed as pdf now
        import webbrowser
        # webbrowser.open_new(r'file://' + os.path.realpath('./Docs/AviaNZManual.pdf'))
        webbrowser.open_new(r'http://avianz.net/docs/AviaNZManual_v1.1.pdf')

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

    def detect(self, minLen=5):
        # check if folder was selected:
        if not self.dirName:
            msg = QMessageBox()
            msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setText("Please select a folder to process!")
            msg.setWindowTitle("Select Folder")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        
        self.species=self.w_spe1.currentText()
        if self.species == "All species":
            self.method = "Default"
        else:
            self.method = "Wavelets"
        
        # directory found, so find any .wav files
        total=0
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                if filename.endswith('.wav'):
                    total=total+1

        # LOG FILE is read here
        # note: important to log all analysis settings here
        self.log = SupportClasses.Log(os.path.join(self.dirName, 'LastAnalysisLog.txt'),
                                self.species, [self.method, self.w_res.value()])

        # Ask for RESUME CONFIRMATION here
        confirmedResume = QMessageBox.Cancel
        if self.log.possibleAppend:
            if len(self.log.filesDone) < total:
                msg = QMessageBox()
                msg.setIconPixmap(QPixmap("img/Owl_thinking.png"))
                msg.setWindowIcon(QIcon('img/Avianz.ico'))
                msg.setWindowTitle("Resume previous batch analysis?")
                msg.setStandardButtons(QMessageBox.No | QMessageBox.Yes)
                text = "Previous analysis found in this folder (analyzed " + str(len(self.log.filesDone)) + " out of " + str(total) + " files in this folder).\nWould you like to resume that analysis?"
                msg.setText(text)
                confirmedResume = msg.exec_()
            else:
                print("All files appear to have previous analysis results")
                msg = QMessageBox()
                msg.setIconPixmap(QPixmap("img/Owl_done.png"))
                msg.setWindowIcon(QIcon('img/Avianz.ico'))
                msg.setText("All files have previous analysis results")
                msg.setWindowTitle("Already processed")
                msg.setStandardButtons(QMessageBox.Ok)
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
            self.filesDone = self.log.filesDone

        # Ask for FINAL USER CONFIRMATION here
        cnt = len(self.filesDone)
        confirmedLaunch = QMessageBox.Cancel
        msg = QMessageBox()
        msg.setIconPixmap(QPixmap("img/Owl_thinking.png"))
        msg.setWindowIcon(QIcon('img/Avianz.ico'))
        text = "Species: " + self.species + ", resolution: "+ str(self.w_res.value()) + ", method: " + self.method + ".\nNumber of files to analyze: " + str(total) + ", " + str(cnt) + " done so far.\n"
        text += "Output stored in " + self.dirName + "/DetectionSummary_*.xlsx.\n"
        text += "Log file stored in " + self.dirName + "/LastAnalysisLog.txt.\n"
        msg.setText("Analysis will be launched with these settings:\n" + text + "\nConfirm?")
        msg.setWindowTitle("Launch batch analysis")
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
        if self.species!="All species":
            self.log.reprintOld()
            # else single-sp runs should be deleted anyway
        if confirmedResume == QMessageBox.No:
            self.log.appendHeader(header=None, species=self.log.species, settings=self.log.settings)
        elif confirmedResume == QMessageBox.Yes:
            self.log.appendHeader(self.log.currentHeader, self.log.species, self.log.settings)
            for f in self.log.filesDone:
                self.log.appendFile(f)

        # delete old results (xlsx)
        # ! WARNING: any Detection...xlsx files will be DELETED,
        # ! ANYWHERE INSIDE the specified dir, recursively
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                if fnmatch.fnmatch(filename, '*DetectionSummary_*.xlsx'):
                    print("Removing excel file %s" % filename)
                    os.remove(os.path.join(root, filename))

        # MAIN PROCESSING starts here
        with pg.BusyCursor():
            for root, dirs, files in os.walk(str(self.dirName)):
                for filename in files:
                    self.filename = os.path.join(root, filename)
                    self.segments = []
                    newSegments = []
                    if self.filename in self.filesDone:
                        # skip the processing, but still need to update excel:
                        print("File %s processed previously, skipping" % filename)
                        # TODO: check the following line, if skip no need to load .wav (except for getting file length for sheet3?)
                        # TODO: Instead can we keep length of the recording as part of the meta info in [-1 ... -1]
                        self.loadFile(wipe = (self.species=="All species"))
                        DOCRecording = re.search('(\d{6})_(\d{6})', filename)
                        if DOCRecording:
                            startTime = DOCRecording.group(2)
                            sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
                        else:
                            sTime = 0
                        if self.species == 'All species':
                            out = SupportClasses.exportSegments(segments=self.segments, species=[], startTime=sTime, dirName=self.dirName, filename=self.filename, datalength=self.datalength, sampleRate=self.sampleRate,method=self.method, resolution=self.w_res.value(), operator="Auto", batch=True)
                        else:
                            out = SupportClasses.exportSegments(segments=self.segments, species=[self.species], startTime=sTime, dirName=self.dirName, filename=self.filename, datalength=self.datalength, sampleRate=self.sampleRate,method=self.method, resolution=self.w_res.value(), operator="Auto", batch=True)
                        out.excel()
                        continue

                    if filename.endswith('.wav'):
                        cnt=cnt+1
                        # check if file not empty                            
                        print("Opening file %s" % filename)
                        self.statusBar().showMessage("Processing file " + str(cnt) + "/" + str(total))
                        if os.stat(self.filename).st_size < 100:
                            print("Skipping empty file")
                            self.log.appendFile(self.filename)
                            continue

                        # test day/night if it is a doc recording
                        Night = False

                        DOCRecording = re.search('(\d{6})_(\d{6})', filename)
                        if DOCRecording:
                            startTime = DOCRecording.group(2)
                            if int(startTime[:2]) > 18 or int(startTime[:2]) < 6:  # 6pm to 6am as night
                                Night = True
                                sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
                            else:
                                Night = False
                                sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
                        else:
                            sTime=0

                        if DOCRecording and self.species in ['Kiwi', 'Ruru'] and not Night:
                            print("Skipping daytime recording")
                            self.log.appendFile(self.filename)
                            continue
                        
                        # ALL SYSTEMS GO: process this file
                        self.loadFile(wipe = (self.species=="All species"))

                        # print self.algs.itemText(self.algs.currentIndex())
                        # if self.algs.currentText() == "Amplitude":
                        #     newSegments = self.seg.segmentByAmplitude(float(str(self.ampThr.text())))
                        # elif self.algs.currentText() == "Median Clipping":
                        #     newSegments = self.seg.medianClip(float(str(self.medThr.text())))
                        #     #print newSegments
                        # elif self.algs.currentText() == "Harma":
                        #     newSegments = self.seg.Harma(float(str(self.HarmaThr1.text())),float(str(self.HarmaThr2.text())))
                        # elif self.algs.currentText() == "Power":
                        #     newSegments = self.seg.segmentByPower(float(str(self.PowerThr.text())))
                        # elif self.algs.currentText() == "Onsets":
                        #     newSegments = self.seg.onsets()
                        #     #print newSegments
                        # elif self.algs.currentText() == "Fundamental Frequency":
                        #     newSegments, pitch, times = self.seg.yin(int(str(self.Fundminfreq.text())),int(str(self.Fundminperiods.text())),float(str(self.Fundthr.text())),int(str(self.Fundwindow.text())),returnSegs=True)
                        #     print newSegments
                        # elif self.algs.currentText() == "FIR":
                        #     print float(str(self.FIRThr1.text()))
                        #     # newSegments = self.seg.segmentByFIR(0.1)
                        #     newSegments = self.seg.segmentByFIR(float(str(self.FIRThr1.text())))
                        #     # print newSegments
                        # elif self.algs.currentText()=='Wavelets':
                        # print("Species: ", self.species)
                        if self.species!='All species':
                            # wipe same species:
                            self.segments[:] = [s for s in self.segments if self.species not in s[4] and self.species+'?' not in s[4]]
                            ws = WaveletSegment.WaveletSegment()
                            speciesData = json.load(open(os.path.join('Filters', self.species+'.txt')))
                            newSegments = ws.waveletSegment_test(fName=None, data=self.audiodata, sampleRate= self.sampleRate, spInfo=speciesData, trainTest=False)
                        else:
                            # wipe all segments:
                            self.segments = []
                            self.seg = Segment.Segment(self.audiodata, self.sgRaw, self.sp, self.sampleRate)
                            newSegments=self.seg.bestSegments()

                        # post process to remove short segments, wind, rain, and use F0 check.
                        if self.species == "Bittern":
                            post = SupportClasses.postProcess(audioData=self.audiodata,
                                                              sampleRate=self.sampleRate,
                                                              segments=newSegments, spInfo=speciesData)
                        elif self.species == "All species":
                            post = SupportClasses.postProcess(audioData=self.audiodata,
                                                              sampleRate=self.sampleRate,
                                                              segments=newSegments, spInfo={})
                            post.wind()
                            post.rainClick()
                        else:
                            post = SupportClasses.postProcess(audioData=self.audiodata,
                                                              sampleRate=self.sampleRate,
                                                              segments=newSegments,
                                                              spInfo=speciesData)
                            # print ("After wavelets: ", post.segments)
                            post.short()  # species specific
                            # print ("After short: ", post.segments)
                            post.wind()
                            # print ("After wind: ", post.segments)
                            post.rainClick()
                            print ("After rain: ", post.segments)
                            post.fundamentalFrq()  # species specific
                            print ("After ff: ", post.segments)
                        newSegments = post.segments

                        # Save the excel
                        if self.species == 'All species':
                            out = SupportClasses.exportSegments(segments=[], segmentstoCheck=newSegments, species=[], startTime=sTime, dirName=self.dirName, filename=self.filename, datalength=self.datalength, sampleRate=self.sampleRate,method=self.method, resolution=self.w_res.value(), operator="Auto", batch=True)
                        else:
                            out = SupportClasses.exportSegments(segments=self.segments, segmentstoCheck=newSegments, species=[self.species], startTime=sTime, dirName=self.dirName, filename=self.filename, datalength=self.datalength, sampleRate=self.sampleRate,method=self.method, resolution=self.w_res.value(), operator="Auto", batch=True)
                        out.excel()
                        # Save the annotation
                        out.saveAnnotation()
                        # Log success for this file
                        self.log.appendFile(self.filename)
            self.log.file.close()
            self.statusBar().showMessage("Processed all %d files" % total)
            msg = QMessageBox()
            msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setText("Finished processing. Would you like to return to the start screen?")
            msg.setWindowTitle("Finished")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            reply = msg.exec_()
            if reply == QMessageBox.Yes: 
                QApplication.exit(1)

    def fillFileList(self,fileName):
        """ Generates the list of files for the file listbox.
        fileName - currently opened file (marks it in the list).
        Most of the work is to deal with directories in that list.
        It only sees *.wav files. Picks up *.data and *_1.wav files, the first to make the filenames
        red in the list, and the second to know if the files are long."""

        # if not os.path.isdir(self.dirName):
        #     print("Directory doesn't exist: making it")
        #     os.makedirs(self.dirName)

        self.listFiles.clear()
        self.listOfFiles = QDir(self.dirName).entryInfoList(['..','*.wav'],filters=QDir.AllDirs|QDir.NoDot|QDir.Files,sort=QDir.DirsFirst)
        listOfDataFiles = QDir(self.dirName).entryList(['*.data'])
        listOfLongFiles = QDir(self.dirName).entryList(['*_1.wav'])
        for file in self.listOfFiles:
            if file.fileName()[:-4]+'_1.wav' in listOfLongFiles:
                # Ignore this entry
                pass
            else:
                # If there is a .data version, colour the name red to show it has been labelled
                item = QListWidgetItem(self.listFiles)
                self.listitemtype = type(item)
                item.setText(file.fileName())
                if file.fileName()+'.data' in listOfDataFiles:
                    item.setForeground(Qt.red)

    def listLoadFile(self,current):
        """ Listener for when the user clicks on an item in filelist
        """

        # Need name of file
        if type(current) is self.listitemtype:
            current = current.text()

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

    def loadFile(self, wipe=True):
        print(self.filename)
        wavobj = wavio.read(self.filename)
        self.sampleRate = wavobj.rate
        self.audiodata = wavobj.data

        # None of the following should be necessary for librosa
        if self.audiodata.dtype is not 'float':
            self.audiodata = self.audiodata.astype('float') #/ 32768.0
        if np.shape(np.shape(self.audiodata))[0]>1:
            self.audiodata = self.audiodata[:,0]
        self.datalength = np.shape(self.audiodata)[0]
        print("Read %d samples, %f s at %d Hz" %(len(self.audiodata),float(self.datalength)/self.sampleRate,self.sampleRate))

        if (self.species=='Kiwi' or self.species=='Ruru') and self.sampleRate!=16000:
            self.audiodata = librosa.core.audio.resample(self.audiodata,self.sampleRate,16000)
            self.sampleRate=16000
            # self.audioFormat.setSampleRate(self.sampleRate)
            self.datalength = np.shape(self.audiodata)[0]
            print("File was downsampled to %d" % self.sampleRate)

        # Create an instance of the Signal Processing class
        if not hasattr(self,'sp'):
            self.sp = SignalProc.SignalProc()

        # Get the data for the spectrogram
        self.sgRaw = self.sp.spectrogram(self.audiodata, window_width=256, incr=128, window='Hann', mean_normalise=True, onesided=True,multitaper=False, need_even=False)
        maxsg = np.min(self.sgRaw)
        self.sg = np.abs(np.where(self.sgRaw==0,0.0,10.0 * np.log10(self.sgRaw/maxsg)))

        # Read in stored segments (useful when doing multi-species)
        if wipe or not os.path.isfile(self.filename + '.data'):
            self.segments = []
        else:
            file = open(self.filename + '.data', 'r')
            self.segments = json.load(file)
            file.close()
            if len(self.segments) > 0:
                if self.segments[0][0] == -1:
                    del self.segments[0]
            if len(self.segments) > 0:
                for s in self.segments:
                    if 0 < s[2] < 1.1 and 0 < s[3] < 1.1:
                        # *** Potential for major cockups here. First version didn't normalise the segmen     t data for dragged boxes.
                        # The second version did, storing them as values between 0 and 1. It modified the      original versions by assuming that the spectrogram was 128 pixels high (256 width window).
                        # This version does what it should have done in the first place, which is to reco     rd actual frequencies
                        # The .1 is to take care of rounding errors
                        # TODO: Because of this change (23/8/18) I run a backup on the datafiles in the i     nit
                        s[2] = self.convertYtoFreq(s[2])
                        s[3] = self.convertYtoFreq(s[3])
                        self.segmentsToSave = True

                    # convert single-species IDs to [species]
                    if type(s[4]) is not list:
                        s[4] = [s[4]]

                    # wipe segments if running species-specific analysis:
                    if s[4] == [self.species]:
                        self.segments.remove(s)

            print("%d segments loaded from .data file" % len(self.segments))

        # Update the data that is seen by the other classes
        # TODO: keep an eye on this to add other classes as required
        if hasattr(self,'seg'):
            self.seg.setNewData(self.audiodata,self.sgRaw,self.sampleRate,256,128)
        else:
            self.seg = Segment.Segment(self.audiodata, self.sgRaw, self.sp, self.sampleRate)
        self.sp.setNewData(self.audiodata,self.sampleRate)


class AviaNZ_reviewAll(QMainWindow):
    # Main class for reviewing batch processing results
    # Should call HumanClassify1 somehow

    def __init__(self,root=None,configdir='',minSegment=50):
        # Allow the user to browse a folder and push a button to process that folder to find a target species
        # and sets up the window.
        super(AviaNZ_reviewAll, self).__init__()
        self.root = root
        self.dirName=""

        # At this point, the main config file should already be ensured to exist.
        self.configfile = os.path.join(configdir, "AviaNZconfig.txt")
        print("Loading configs from file %s" % self.configfile)
        self.config = json.load(open(self.configfile))
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

        self.w_resLabel = QLabel("  Output Resolution (secs)")
        self.d_detection.addWidget(self.w_resLabel, row=3, col=0)
        self.w_res = QSpinBox()
        self.w_res.setRange(1,600)
        self.w_res.setSingleStep(5)
        self.w_res.setValue(60)
        self.d_detection.addWidget(self.w_res, row=3, col=1, colspan=2)

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
        """ Create the About Message Box"""
        msg = QMessageBox()
        msg.setIconPixmap(QPixmap("img\AviaNZ.png"))
        msg.setWindowIcon(QIcon('img/Avianz.ico'))
        msg.setText("The AviaNZ Program, v1.1 (August 2018)")
        msg.setInformativeText("By Stephen Marsland, Victoria University of Wellington. With code by Nirosha Priyadarshani and Julius Juodakis, and input from Isabel Castro, Moira Pryde, Stuart Cockburn, Rebecca Stirnemann, Sumudu Purage, Virginia Listanti, and Rebecca Huistra. \n stephen.marsland@vuw.ac.nz")
        msg.setWindowTitle("About")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        return

    def showHelp(self):
        """ Show the user manual (a pdf file)"""
        # TODO: manual is not distributed as pdf now
        import webbrowser
        # webbrowser.open_new(r'file://' + os.path.realpath('./Docs/AviaNZManual.pdf'))
        webbrowser.open_new(r'http://avianz.net/docs/AviaNZManual_v1.1.pdf')

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
        self.spList = ['All species']
        # find species names from the annotations
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                if filename.endswith('.data'):
                    datFile = root + '/' + filename
                    if os.path.isfile(datFile):
                        with open(datFile) as f:
                            segments = json.load(f)
                            for seg in segments:
                                if seg[0] == -1:
                                    continue
                                elif len(seg[4])>0:
                                    for birdName in seg[4]:
                                        if len(birdName)>0 and birdName[-1] == '?':
                                            if birdName[:-1] not in self.spList:
                                                self.spList.append(birdName[:-1])
                                        elif birdName not in self.spList:
                                            self.spList.append(birdName)
        self.w_spe1.clear()
        self.w_spe1.addItems(self.spList)
        self.fillFileList(self.dirName)

    def review(self):
        self.species = self.w_spe1.currentText()
        self.reviewer = self.w_reviewer.text()
        print("Reviewer: ", self.reviewer)
        if self.reviewer == '':
            msg = QMessageBox()
            msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setText("Please enter reviewer name")
            msg.setWindowTitle("Reviewer")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        if self.dirName is "":
            msg = QMessageBox()
            msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setText("Please select a folder to process!")
            msg.setWindowTitle("Select Folder")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        # directory found, reviewer provided, so start review
        # 1. find any .wav+.data files
        # 2. delete old results (xlsx)
        # ! WARNING: any Detection...xlsx files will be DELETED,
        # ! ANYWHERE INSIDE the specified dir, recursively
        total = 0
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                filename = os.path.join(root, filename)

                if fnmatch.fnmatch(filename, '*DetectionSummary_*.xlsx'):
                    print("Removing excel file %s" % filename)
                    os.remove(filename)

                if filename.endswith('.wav') and os.path.isfile(filename + '.data'):
                    total = total + 1


        # main file review loop
        cnt = 0
        filesuccess = 1
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                DOCRecording = re.search('(\d{6})_(\d{6})', filename)
                filename = os.path.join(root, filename)
                self.filename = filename
                filesuccess = 1
                if filename.endswith('.wav') and os.path.isfile(filename + '.data'):
                    print("Opening file %s" % filename)
                    cnt=cnt+1
                    if os.stat(filename).st_size < 100:
                        print("Skipping empty file")
                        continue

                    # test day/night if it is a doc recording
                    Night = False
                    if DOCRecording:
                        startTime = DOCRecording.group(2)
                        if int(startTime[:2]) > 17 or int(startTime[:2]) < 7:  # 6pm to 6am as night
                            Night = True
                        sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
                    else:
                        sTime = 0

                    if DOCRecording and self.species in ['Kiwi', 'Ruru'] and not Night:
                        continue

                    self.statusBar().showMessage("Reviewing file " + str(cnt) + "/" + str(total) + "...")
                    # load segments
                    self.segments = json.load(open(filename + '.data'))
                    # read in operator from first "segment"
                    if len(self.segments)>0 and self.segments[0][0] == -1:
                        self.operator = self.segments[0][2]
                        del self.segments[0]
                    else:
                        self.operator = "None"

                    self.loadFile()
                    if len(self.segments) == 0:
                        # and skip review dialog, but save the name into excel
                        print("No segments found in file %s" % filename)
                    # file has segments, so call the right review dialog:
                    elif self.species == 'All species':
                        filesuccess = self.review_all(sTime)
                    else:
                        filesuccess = self.review_single(sTime)
                        print("File success: ", filesuccess)

                    # Store the output to an Excel file (no matter if review dialog exit was clean)
                    out = SupportClasses.exportSegments(segments=self.segments, startTime=sTime, dirName=self.dirName, filename=self.filename, datalength=self.datalength, sampleRate=self.sampleRate, resolution=self.w_res.value(), operator=self.operator, reviewer=self.reviewer, species=[self.species], batch=True)
                    out.excel()
                    # Save the corrected segment JSON
                    out.saveAnnotation()

                    # break out of both loops if Esc detected
                    # (return value will be 1 for correct close, 0 for Esc)
                    if filesuccess == 0:
                        break

            # after the loop, check if file wasn't Esc-broken
            if filesuccess == 0:
                break

        # loop complete, all files checked
        # save the excel at the end
        self.statusBar().showMessage("Reviewed files " + str(cnt) + "/" + str(total))
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowIcon(QIcon('img/Avianz.ico'))
        msg.setStandardButtons(QMessageBox.Ok)
        if filesuccess == 1:
            msg.setIconPixmap(QPixmap("img/Owl_done.png"))
            msg.setText("All files checked")
            msg.setWindowTitle("Finished")
        else:
            msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
            msg.setText("Review stopped at file %s of %s" % (cnt, total))
            msg.setWindowTitle("Review stopped")
        msg.exec_()

    def review_single(self, sTime):
        """ Initializes all species dialog.
            Updates self.segments as a side effect.
            Returns 1 for clean completion, 0 for Esc press or other dirty exit.
        """
        # self.segments_other = []
        self.segments_sp = []
        for seg in self.segments:
            for birdName in seg[4]:
                if len(birdName)>0 and birdName[-1] == '?':
                    if self.species == birdName[:-1]:
                        self.segments_sp.append(seg)
                        break
                elif self.species == birdName:
                    self.segments_sp.append(seg)
                    break

        segments = copy.deepcopy(self.segments)
        errorInds = []
        # Initialize the dialog for this file
        if len(self.segments_sp) > 0:
            self.humanClassifyDialog2 = Dialogs.HumanClassify2(self.sg, self.audiodata, self.segments_sp,
                                           self.species, self.sampleRate, self.audioFormat,
                                           self.config['incr'], self.lut, self.colourStart,
                                           self.colourEnd, self.config['invertColourMap'], self.filename)

            success = self.humanClassifyDialog2.exec_()
            # capture Esc press or other "dirty" exit:
            if success == 0:
                 return(0)
            errorInds = self.humanClassifyDialog2.getValues()
            print("Errors: ", errorInds, len(errorInds))

        outputErrors = []
        if len(errorInds) > 0:
            # print(self.segments)
            for ind in errorInds:
                outputErrors.append(self.segments[ind])
                # self.deleteSegment(id=ids[ind], hr=True)
                # ids = [x - 1 for x in ids]
            self.segmentsToSave = True
            if self.config['saveCorrections']:
                # Save the errors in a file
                file = open(self.filename + '.corrections_' + str(self.species), 'a')
                json.dump(outputErrors, file)
                file.close()

        # Produce segments:
        for seg in outputErrors:
            if seg in self.segments:
                segments.remove(seg)
        # remove '?'
        for seg in segments:
            for sp in seg[4]:
                if sp[:-1] == self.species and sp[-1] == '?':
                    sp = sp[:-1]

        self.segments = segments
        return(1)

    def review_all(self, sTime, minLen=5):
       """ Initializes all species dialog.
           Updates self.segments as a side effect.
           Returns 1 for clean completion, 0 for Esc press or other dirty exit.
       """
       # Initialize the dialog for this file
       shortBirdList = json.load(open(self.config['BirdListShort']))
       if self.config['BirdListLong'] is not None and self.config['BirdListLong'] != "None":
            longBirdList = json.load(open(self.config['BirdListLong']))
       else:
            longBirdList = None
       self.humanClassifyDialog1 = Dialogs.HumanClassify1(self.lut,self.colourStart,self.colourEnd,self.config['invertColourMap'], shortBirdList, longBirdList, self)
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
       success = self.humanClassifyDialog1.exec_() # 1 on clean exit

       if success == 0:
           self.humanClassifyDialog1.stopPlayback()
           return(0)

       return(1)

    def loadFile(self):
        wavobj = wavio.read(self.filename)
        self.sampleRate = wavobj.rate
        self.audiodata = wavobj.data
        self.audioFormat.setChannelCount(np.shape(self.audiodata)[1])
        self.audioFormat.setSampleRate(self.sampleRate)
        self.audioFormat.setSampleSize(wavobj.sampwidth*8)
        print("Detected format: %d channels, %d Hz, %d bit samples" % (self.audioFormat.channelCount(), self.audioFormat.sampleRate(), self.audioFormat.sampleSize()))

        # None of the following should be necessary for librosa
        if self.audiodata.dtype is not 'float':
            self.audiodata = self.audiodata.astype('float') #/ 32768.0
        if np.shape(np.shape(self.audiodata))[0]>1:
            self.audiodata = self.audiodata[:,0]
        self.datalength = np.shape(self.audiodata)[0]
        print("Length of file is ",len(self.audiodata),float(self.datalength)/self.sampleRate,self.sampleRate)
        # self.w_dir.setPlainText(self.filename)

        if (self.species=='Kiwi' or self.species=='Ruru') and self.sampleRate!=16000:
            self.audiodata = librosa.core.audio.resample(self.audiodata,self.sampleRate,16000)
            self.sampleRate=16000
            self.audioFormat.setSampleRate(self.sampleRate)
            self.datalength = np.shape(self.audiodata)[0]
            print("File was downsampled to %d" % self.sampleRate)

        # Create an instance of the Signal Processing class
        if not hasattr(self,'sp'):
            self.sp = SignalProc.SignalProc()

        # Get the data for the spectrogram
        self.sgRaw = self.sp.spectrogram(self.audiodata, window_width=256, incr=128, window='Hann', mean_normalise=True, onesided=True,multitaper=False, need_even=False)
        maxsg = np.min(self.sgRaw)
        self.sg = np.abs(np.where(self.sgRaw==0,0.0,10.0 * np.log10(self.sgRaw/maxsg)))
        self.setColourMap()

        # Update the data that is seen by the other classes
        # TODO: keep an eye on this to add other classes as required
        # self.seg.setNewData(self.audiodata,self.sgRaw,self.sampleRate,256,128)
        self.sp.setNewData(self.audiodata,self.sampleRate)

    def humanClassifyNextImage1(self):
        # Get the next image
        if self.box1id < len(self.segments):
            # update "done/to go" numbers:
            self.humanClassifyDialog1.setSegNumbers(self.box1id, len(self.segments))
            # Check if have moved to next segment, and if so load it
            # If there was a section without segments this would be a bit inefficient, actually no, it was wrong!

            # Show the next segment
            #print(self.segments[self.box1id])
            x1nob = self.segments[self.box1id][0]
            x2nob = self.segments[self.box1id][1]
            x1 = int(self.convertAmpltoSpec(x1nob - self.config['reviewSpecBuffer']))
            x1 = max(x1, 0)
            x2 = int(self.convertAmpltoSpec(x2nob + self.config['reviewSpecBuffer']))
            x2 = min(x2, len(self.sg))
            x3 = int((x1nob - self.config['reviewSpecBuffer']) * self.sampleRate)
            x3 = max(x3, 0)
            x4 = int((x2nob + self.config['reviewSpecBuffer']) * self.sampleRate)
            x4 = min(x4, len(self.audiodata))
            self.humanClassifyDialog1.setImage(self.sg[x1:x2, :], self.audiodata[x3:x4], self.sampleRate, self.config['incr'],
                                           self.segments[self.box1id][4], self.convertAmpltoSpec(x1nob)-x1, self.convertAmpltoSpec(x2nob)-x1,
                                           self.segments[self.box1id][0], self.segments[self.box1id][1])

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setIconPixmap(QPixmap("img/Owl_done.png"))
            msg.setText("All segments in this file checked")
            msg.setWindowTitle("Finished")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

            # store position to popup the next one in there
            self.dialogSize = self.humanClassifyDialog1.size()
            self.dialogPos = self.humanClassifyDialog1.pos()
            self.humanClassifyDialog1.done(1)

    def humanClassifyPrevImage(self):
        """ Go back one image by changing boxid and calling NextImage.
        Note: won't undo deleted segments."""
        if self.box1id>0:
            self.box1id -= 1
            self.humanClassifyNextImage1()

    def humanClassifyCorrect1(self):
        """ Correct segment labels, save the old ones if necessary """
        label, self.saveConfig, checkText = self.humanClassifyDialog1.getValues()
        if len(checkText) > 0:
            if label != checkText:
                label = str(checkText)
                self.humanClassifyDialog1.birdTextEntered()
                # self.saveConfig = True
            #self.humanClassifyDialog1.tbox.setText('')

        if label != self.segments[self.box1id][4]:
            if self.config['saveCorrections']:
                # Save the correction
                outputError = [self.segments[self.box1id], label]
                file = open(self.filename + '.corrections', 'a')
                json.dump(outputError, file)
                file.close()

            # Update the label on the box if it is in the current page
            self.segments[self.box1id][4] = label

            if self.saveConfig:
                self.config['BirdList'].append(label)
        elif len(label)>0 and label[-1] == '?':
            # Remove the question mark, since the user has agreed
            self.segments[self.box1id][4] = label[:-1]

        self.humanClassifyDialog1.tbox.setText('')
        self.humanClassifyDialog1.tbox.setEnabled(False)
        # counter updated here
        self.box1id += 1
        self.humanClassifyNextImage1()

    def humanClassifyDelete1(self):
        # Delete a segment
        # (no need to update counter then)
        id = self.box1id
        del self.segments[id]
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

        # if not os.path.isdir(self.dirName):
        #     print("Directory doesn't exist: making it")
        #     os.makedirs(self.dirName)

        self.listFiles.clear()
        self.listOfFiles = QDir(self.dirName).entryInfoList(['..','*.wav'],filters=QDir.AllDirs|QDir.NoDot|QDir.Files,sort=QDir.DirsFirst)
        listOfDataFiles = QDir(self.dirName).entryList(['*.data'])
        listOfLongFiles = QDir(self.dirName).entryList(['*_1.wav'])
        for file in self.listOfFiles:
            if file.fileName()[:-4]+'_1.wav' in listOfLongFiles:
                # Ignore this entry
                pass
            else:
                # If there is a .data version, colour the name red to show it has been labelled
                item = QListWidgetItem(self.listFiles)
                self.listitemtype = type(item)
                item.setText(file.fileName())
                if file.fileName()+'.data' in listOfDataFiles:
                    item.setForeground(Qt.red)

    def listLoadFile(self,current):
        """ Listener for when the user clicks on an item in filelist
        """

        # Need name of file
        if type(current) is self.listitemtype:
            current = current.text()

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
