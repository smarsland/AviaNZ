import os, re, platform

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

import json

sppInfo = {
            # spp: [min len, max len, flow, fhigh, fs, f0_low, f0_high, wavelet_thr, wavelet_M, wavelet_nodes]
            'Kiwi': [10, 30, 1100, 7000, 16000, 1200, 4200, 0.25, 0.6, [17, 20, 22, 35, 36, 38, 40, 42, 43, 44, 45, 46, 48, 50, 55, 56]],
            'Gsk': [6, 25, 900, 7000, 16000, 1200, 4200, 0.25, 0.6, [35, 38, 43, 44, 52, 54]],
            'Lsk': [10, 30, 1200, 7000, 16000, 1200, 4200,  0.25, 0.6, []], # todo: find len, f0, nodes
            'Ruru': [1, 30, 500, 7000, 16000, 600, 1300,  0.25, 0.5, [33, 37, 38]], # find M
            'SIPO': [1, 5, 1200, 3800, 8000, 1200, 3800,  0.25, 0.2, [61, 59, 54, 51, 60, 58, 49, 47]],  # find len, f0
            'Bittern': [1, 5, 100, 200, 1000, 100, 200, 0.75, 0.2, [10,21,22,43,44,45,46]],  # find len, f0, confirm nodes
}

class AviaNZ_batchProcess(QMainWindow):
    # Main class for batch processing

    def __init__(self,root=None,minSegment=50, DOC=True):
        # Allow the user to browse a folder and push a button to process that folder to find a target species
        # and sets up the window.
        super(AviaNZ_batchProcess, self).__init__()
        self.root = root
        self.dirName=[]
        self.DOC=DOC

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
        self.setMinimumSize(600,400)

        # Make the docks
        self.d_detection = Dock("Automatic Detection",size=(350,100))
        # self.d_detection.hideTitleBar()

        self.d_files = Dock("File list", size=(150, 100))


        self.area.addDock(self.d_detection,'right')
        self.area.addDock(self.d_files, 'left')

        self.w_browse1 = QPushButton("  &Browse Folder")
        self.w_browse1.setToolTip("Can select a folder with sub folders to process")
        self.w_browse1.setFixedHeight(50)
        self.w_browse1.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')
        self.w_dir1 = QPlainTextEdit()
        self.w_dir1.setFixedHeight(50)
        self.w_dir1.setPlainText('')
        self.w_dir1.setToolTip("The folder being processed")
        self.d_detection.addWidget(self.w_dir1,row=0,col=1,colspan=2)
        self.d_detection.addWidget(self.w_browse1,row=0,col=0)

        self.w_speLabel1 = QLabel("  Select Species")
        self.d_detection.addWidget(self.w_speLabel1,row=1,col=0)
        self.w_spe1 = QComboBox()
        self.w_spe1.addItems(["Kiwi", "Ruru", "Bittern", "all"])
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

        self.w_browse1.clicked.connect(self.browse_detect)

        self.w_files = pg.LayoutWidget()
        self.d_files.addWidget(self.w_files)
        self.w_files.addWidget(QLabel('You chose to process'), row=0, col=0)
        self.w_files.addWidget(QLabel('# sound files'), row=1, col=0)
        # List to hold the list of files
        self.listFiles = QListWidget()
        self.listFiles.setMinimumWidth(150)
        # self.listFiles.connect(self.listFiles, SIGNAL('itemDoubleClicked(QListWidgetItem*)'), self.listLoadFile)
        # self.listFiles.itemDoubleClicked.connect(self.listLoadFile)
        self.w_files.addWidget(self.listFiles, row=2, col=0)

        self.show()

    def createMenu(self):
        """ Create the basic menu.
        """

        helpMenu = self.menuBar().addMenu("&Help")
        helpMenu.addAction("Help", self.showHelp,"Ctrl+H")
        aboutMenu = self.menuBar().addMenu("&About")
        aboutMenu.addAction("About", self.showAbout,"Ctrl+A")

    def showAbout(self):
        """ Create the About Message Box"""
        msg = QMessageBox()
        msg.setIconPixmap(QPixmap("img\AviaNZ.png"))
        msg.setWindowIcon(QIcon('img/Avianz.ico'))
        msg.setText("The AviaNZ Program, v0.10 (June 2017)")
        msg.setInformativeText("By Stephen Marsland, Massey University (2016--2017). With code by Nirosha Priyadarshani and input from Isabel Castro, Moira Pryde, Stuart Cockburn, Rebecca Stirnemann, Sumudu Manic Purage. \ns.r.marsland@massey.ac.nz; n.p.priyadarshani@massey.ac.nz")
        msg.setWindowTitle("About")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        return

    def showHelp(self):
        """ Show the user manual (a pdf file)"""
        # TODO: manual is not distributed as pdf now
        import webbrowser
        # webbrowser.open_new(r'file://' + os.path.realpath('./Docs/AviaNZManual.pdf'))
        webbrowser.open_new(r'http://avianz.net/docs/AviaNZManual.pdf')

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

    def browse_detect(self):
        self.browse(d=True)

    def browse_denoise(self):
        self.browse(d=False)

    def browse(self,d=True):
        # self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',"Wav files (*.wav)")
        if self.dirName:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirName))
        else:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process')
        print("Dir:", self.dirName)
        if d:
            self.w_dir1.setPlainText(self.dirName)
        else:
            self.w_dir2.setPlainText(self.dirName)

        self.fillFileList(self.dirName)

    def detect(self, minLen=5):
        with pg.BusyCursor():
            if self.dirName:
                # self.statusLeft.setText("Processing...")
                i=self.w_spe1.currentIndex()
                if i==0:
                    self.species="Kiwi"
                elif i==1:
                    self.species="Ruru"
                elif i==2:
                    self.species ="Bittern"
                else: # All
                    self.species="all"
                total=0
                for root, dirs, files in os.walk(str(self.dirName)):
                    for filename in files:
                        if filename.endswith('.wav'):
                            total=total+1
                cnt=0   # processed number of files

                for root, dirs, files in os.walk(str(self.dirName)):
                    for filename in files:
                        if filename.endswith('.wav'):
                            # test day/night if it is a doc recording
                            print(filename)
                            Night = False
                            DOCRecording = re.search('(\d{6})_(\d{6})', filename)
                            if DOCRecording:
                                startTime = DOCRecording.group(2)
                                if int(startTime[:2]) > 18 or int(startTime[:2]) < 6:  # 6pm to 6am as night
                                    Night = True
                                    sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
                            else:
                                sTime=0

                            if DOCRecording and self.species in ['Kiwi', 'Ruru'] and not Night:
                                continue
                            else:
                                # if not os.path.isfile(root+'/'+filename+'.data'): # if already processed then skip?
                                #     continue
                                cnt=cnt+1
                                # self.statusRight.setText("Processing file " + str(cnt) + "/" + str(total))
                                self.statusBar().showMessage("Processing file " + str(cnt) + "/" + str(total) + "...")
                                self.filename=root+'/'+filename
                                self.loadFile()
                                # self.seg = Segment.Segment(self.audiodata, self.sgRaw, self.sp, self.sampleRate)
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
                                if self.species!='all':
                                    self.method = "Wavelets"
                                    ws = WaveletSegment.WaveletSegment(species=sppInfo[self.species])
                                    print ("sppInfo: ", sppInfo[self.species])
                                    newSegments = ws.waveletSegment_test(fName=None,data=self.audiodata, sampleRate=self.sampleRate, spInfo=sppInfo[self.species],trainTest=False)
                                    print("in batch", newSegments)
                                else:
                                    self.method = "Default"
                                    self.seg = Segment.Segment(self.audiodata, self.sgRaw, self.sp, self.sampleRate)
                                    newSegments=self.seg.bestSegments()
                                    # print newSegments

                                # post process to remove short segments, wind, rain, and use F0 check.
                                if self.species == "all":
                                    post = SupportClasses.postProcess(audioData=self.audiodata,
                                                                      sampleRate=self.sampleRate,
                                                                      segments=newSegments, species=[])
                                    post.wind()
                                    post.rainClick()
                                else:
                                    post = SupportClasses.postProcess(audioData=self.audiodata,
                                                                      sampleRate=self.sampleRate,
                                                                      segments=newSegments,
                                                                      species=sppInfo[self.species])
                                    post.short()  # species specific
                                    post.wind()
                                    post.rainClick()
                                    post.fundamentalFrq()  # species specific
                                newSegments = post.segments
                                # Save output
                                out = SupportClasses.exportSegments(segments=newSegments, confirmedSegments=[], segmentstoCheck=post.segments, species=self.species, startTime=sTime, dirName=self.dirName, filename=self.filename, datalength=self.datalength, sampleRate=self.sampleRate,method=self.method, resolution=self.w_res.value())
                                out.excel()
                                # Save the annotation
                                out.saveAnnotation()
                # self.statusLeft.setText("Ready")
                self.statusBar().showMessage("Processed files " + str(cnt) + "/" + str(total))
            else:
                msg = QMessageBox()
                msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
                msg.setWindowIcon(QIcon('img/Avianz.ico'))
                msg.setText("Please select a folder to process!")
                msg.setWindowTitle("Select Folder")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

    def fillFileList(self,fileName):
        """ Generates the list of files for the file listbox.
        fileName - currently opened file (marks it in the list).
        Most of the work is to deal with directories in that list.
        It only sees *.wav files. Picks up *.data and *_1.wav files, the first to make the filenames
        red in the list, and the second to know if the files are long."""

        if not os.path.isdir(self.dirName):
            print("Directory doesn't exist: making it")
            os.makedirs(self.dirName)

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
        if fileName:
            index = self.listFiles.findItems(fileName,Qt.MatchExactly)
            if len(index)>0:
                self.listFiles.setCurrentItem(index[0])
            else:
                index = self.listFiles.findItems(self.listOfFiles[0].fileName(),Qt.MatchExactly)
                self.listFiles.setCurrentItem(index[0])

    def loadFile(self):
        print(self.filename)
        wavobj = wavio.read(self.filename)
        self.sampleRate = wavobj.rate
        self.audiodata = wavobj.data
        print(np.shape(self.audiodata))

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
            self.datalength = np.shape(self.audiodata)[0]
        print(self.sampleRate)

        # Create an instance of the Signal Processing class
        if not hasattr(self,'sp'):
            # self.sp = SignalProc.SignalProc(self.audiodata, self.sampleRate)
            self.sp = SignalProc.SignalProc()

        # Get the data for the spectrogram
        self.sgRaw = self.sp.spectrogram(self.audiodata, window_width=256, incr=128, window='Hann', mean_normalise=True, onesided=True,multitaper=False, need_even=False)
        maxsg = np.min(self.sgRaw)
        self.sg = np.abs(np.where(self.sgRaw==0,0.0,10.0 * np.log10(self.sgRaw/maxsg)))

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

    def __init__(self,root=None,configfile='',minSegment=50, DOC=True):
        # Allow the user to browse a folder and push a button to process that folder to find a target species
        # and sets up the window.
        super(AviaNZ_reviewAll, self).__init__()
        self.root = root
        self.dirName=[]
        self.DOC=DOC

        # read config file
        try:
            print("Loading configs from file %s" % configfile)
            self.config = json.load(open(configfile))
            self.saveConfig = True
        except:
            print("Failed to load config file, using defaults")
            self.config = json.load(open('AviaNZconfig.txt'))
            self.saveConfig = True # TODO: revise this with user permissions in mind
        self.configfile = configfile

        # audio things
        self.audioFormat = QAudioFormat()
        self.audioFormat.setCodec("audio/pcm")
        self.audioFormat.setByteOrder(QAudioFormat.LittleEndian)
        self.audioFormat.setSampleType(QAudioFormat.SignedInt)

        # Make the window and associated widgets
        QMainWindow.__init__(self, root)

        self.statusBar().showMessage("Processing file Current/Total")

        self.setWindowTitle('AviaNZ - Reviewing Batch Results')
        self.createFrame()
        self.center()

    def createFrame(self):
        # Make the window and set its size
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.setFixedSize(500,300)

        # Make the docks
        self.d_detection = Dock("Automatic Detection",size=(350,100))
        self.d_detection.hideTitleBar()

        self.area.addDock(self.d_detection,'right')

        self.w_browse1 = QPushButton("  &Browse Folder")
        self.w_browse1.setToolTip("Can select a folder with sub folders to process")
        self.w_browse1.setFixedHeight(50)
        self.w_browse1.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')
        self.w_dir1 = QPlainTextEdit()
        self.w_dir1.setFixedHeight(50)
        self.w_dir1.setPlainText('')
        self.w_dir1.setToolTip("The folder being processed")
        self.d_detection.addWidget(self.w_dir1,row=0,col=1,colspan=2)
        self.d_detection.addWidget(self.w_browse1,row=0,col=0)

        self.w_speLabel1 = QLabel("  Select Species")
        self.d_detection.addWidget(self.w_speLabel1,row=1,col=0)
        self.w_spe1 = QComboBox()
        self.w_spe1.addItems(["all"])
        self.d_detection.addWidget(self.w_spe1,row=1,col=1,colspan=2)

        self.w_resLabel = QLabel("  Output Resolution (secs)")
        self.d_detection.addWidget(self.w_resLabel, row=2, col=0)
        self.w_res = QSpinBox()
        self.w_res.setRange(1,600)
        self.w_res.setSingleStep(5)
        self.w_res.setValue(60)
        self.d_detection.addWidget(self.w_res, row=2, col=1, colspan=2)

        self.w_processButton = QPushButton("&Review Folder")
        self.w_processButton.clicked.connect(self.detect)
        self.d_detection.addWidget(self.w_processButton,row=11,col=2)
        self.w_processButton.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')

        self.w_browse1.clicked.connect(self.browse_detect)

        self.show()

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

    def browse_detect(self):
        self.browse(d=True)

    def browse_denoise(self):
        self.browse(d=False)

    def browse(self,d=True):
        # self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',"Wav files (*.wav)")
        if self.dirName:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirName))
        else:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process')
        print("Dir:", self.dirName)
        if d:
            self.w_dir1.setPlainText(self.dirName)
        else:
            self.w_dir2.setPlainText(self.dirName)

    def detect(self, minLen=5):
        with pg.BusyCursor():
            if self.dirName:
                # self.statusLeft.setText("Processing...")
                i=self.w_spe1.currentIndex()
                self.species="all"
                total=0
                for root, dirs, files in os.walk(str(self.dirName)):
                    for filename in files:
                        filename = os.path.join(root, filename)
                        if filename.endswith('.wav') and os.path.isfile(filename + '.data'):
                            print(filename)
                            total=total+1
                cnt=0   # processed number of files

                for root, dirs, files in os.walk(str(self.dirName)):
                    for filename in files:
                        DOCRecording = re.search('(\d{6})_(\d{6})', filename)
                        filename = os.path.join(root, filename)
                        self.filename = filename
                        if filename.endswith('.wav') and os.path.isfile(filename + '.data'):
                            # test day/night if it is a doc recording
                            print("Opening file %s" % filename)

                            Night = False
                            if DOCRecording:
                                startTime = DOCRecording.group(2)
                                if int(startTime[:2]) > 18 or int(startTime[:2]) < 6:  # 6pm to 6am as night
                                    Night = True
                                    sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
                            else:
                                sTime=0

                            if DOCRecording and self.species in ['Kiwi', 'Ruru'] and not Night:
                                continue
                            else:
                                cnt=cnt+1
                                self.statusBar().showMessage("Processing file " + str(cnt) + "/" + str(total) + "...")
                                # load segments
                                self.segments = json.load(open(filename + '.data'))
                                if self.segments[0][0] == -1:
                                    self.operator = self.segments[0][2]
                                    self.reviewer = self.segments[0][3]
                                    del self.segments[0]
                                if len(self.segments)==0:
                                    # no segments, skip
                                    print("no segments found in file %s" % filename)
                                    continue

                                self.loadFile()

                                # Initialize the dialog for this file
                                self.humanClassifyDialog1 = Dialogs.HumanClassify1(self.lut,self.colourStart,self.colourEnd,self.config['invertColourMap'], self.config['BirdList'], self)
                                self.box1id = 0
                                self.humanClassifyNextImage1()
                                self.humanClassifyDialog1.activateWindow()
                                self.humanClassifyDialog1.correct.clicked.connect(self.humanClassifyCorrect1)
                                self.humanClassifyDialog1.delete.clicked.connect(self.humanClassifyDelete1)
                                self.humanClassifyDialog1.exec_()

                                # (this is resumed after each file is done)
                                # Append this file's info to the worksheet:
                                out = SupportClasses.exportSegments(segments=self.segments, startTime=sTime, dirName=self.dirName, filename=self.filename, datalength=self.datalength, sampleRate=self.sampleRate, resolution=self.w_res.value())
                                out.excel()
                                # Save the corrected segment JSON
                                out.saveAnnotation()

                # loop complete, all files checked
                self.statusBar().showMessage("Processed files " + str(cnt) + "/" + str(total))
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowIcon(QIcon('img/Avianz.ico'))
                msg.setIconPixmap(QPixmap("img/Owl_done.png"))
                msg.setText("All files checked")
                msg.setWindowTitle("Finished")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                self.humanClassifyDialog1.done(1)
            else:
                msg = QMessageBox()
                msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
                msg.setWindowIcon(QIcon('img/Avianz.ico'))
                msg.setText("Please select a folder to process!")
                msg.setWindowTitle("Select Folder")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()


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
            self.datalength = np.shape(self.audiodata)[0]
        print(self.sampleRate)

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
            print(self.segments[self.box1id])
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
            self.humanClassifyDialog1.setImage(self.sg[x1:x2, :], self.audiodata[x3:x4], self.sampleRate,
                                           self.segments[self.box1id][4], self.convertAmpltoSpec(x1nob)-x1, self.convertAmpltoSpec(x2nob)-x1)

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setIconPixmap(QPixmap("img/Owl_done.png"))
            msg.setText("All segments in this file checked")
            msg.setWindowTitle("Finished")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            self.humanClassifyDialog1.done(1)

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
        elif label[-1] == '?':
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


