import os, json
from PyQt4.QtCore import *
from PyQt4.QtGui import *

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

class AviaNZFindSpeciesInterface(QMainWindow):
    # Main class for batch processing

    def __init__(self,root=None,minSegment=50):
        # Allow the user to browse a folder and push a button to process that folder to find a target species
        # and sets up the window.
        super(AviaNZFindSpeciesInterface, self).__init__()
        self.root = root
        self.dirName=[]
        # self.minSegment=minSegment

        # Make the window and associated widgets
        QMainWindow.__init__(self, root)

        # add statusbar
        self.statusLeft = QLabel("Ready")
        self.statusLeft.setFrameStyle(QFrame.Panel)
        self.statusRight = QLabel("Processing file Current/Total")
        self.statusRight.setAlignment(Qt.AlignRight)
        self.statusRight.setFrameStyle(QFrame.Panel)
        statusStyle='QLabel {border:transparent}'
        self.statusLeft.setStyleSheet(statusStyle)
        self.statusRight.setStyleSheet(statusStyle)
        self.statusBar().addPermanentWidget(self.statusLeft,1)
        self.statusBar().addPermanentWidget(self.statusRight,1)

        # Set the message in the status bar
        self.statusLeft.setText("Ready")
        self.statusRight.setText("Processing file Current/Total")

        self.setWindowTitle('AviaNZ - Automatic Detection')
        self.createFrame()

    def createFrame(self):
        # Make the window and set its size
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.setFixedSize(500,300)

        # Make the docks
        self.d_detection = Dock("Automatic Detection",size=(350,100))
        self.d_detection.hideTitleBar()

        # self.area.addDock(self.d_fileList,'left')
        self.area.addDock(self.d_detection,'right')

        self.w_browse = QPushButton("  &Browse Folder")
        self.connect(self.w_browse, SIGNAL('clicked()'), self.browse)
        self.w_browse.setToolTip("Can select a folder with sub folders to process")
        self.w_browse.setFixedHeight(50)
        self.w_browse.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')
        self.w_dir = QPlainTextEdit()
        self.w_dir.setFixedHeight(50)
        self.w_dir.setPlainText('')
        self.w_dir.setToolTip("The folder being processed")
        self.d_detection.addWidget(self.w_dir,row=0,col=1,colspan=2)
        self.d_detection.addWidget(self.w_browse,row=0,col=0)

        self.w_speLabel = QLabel("  Select Species")
        self.d_detection.addWidget(self.w_speLabel,row=1,col=0)
        self.w_spe = QComboBox()
        self.w_spe.addItems(["Kiwi", "Ruru","all"])
        self.d_detection.addWidget(self.w_spe,row=1,col=1,colspan=2)

        self.w_processButton = QPushButton("&Process Folder")
        self.connect(self.w_processButton, SIGNAL('clicked()'), self.detect)
        self.d_detection.addWidget(self.w_processButton,row=10,col=2)
        self.w_processButton.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')

        self.statusLeft.setText("Ready")

        # Plot everything
        self.show()

    def browse(self):
        # self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',"Wav files (*.wav)")
        if self.dirName:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirName))
        else:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process')
        print "Dir:", self.dirName
        self.w_dir.setPlainText(self.dirName)

    def detect(self):
        with pg.BusyCursor():
            if self.dirName:
                self.statusLeft.setText("Processing...")
                i=self.w_spe.currentIndex()
                if i==0:
                    self.species="Kiwi"
                elif i==1:
                    self.species="Ruru"
                else: # All
                    self.species="all"

                total=0
                for root, dirs, files in os.walk(str(self.dirName)):
                    for filename in files:
                        if filename.endswith('.wav'):
                            total=total+1
                cnt=0   # processed number of files
                self.statusRight.setText("Processing file " + "0/" + str(total))

                for root, dirs, files in os.walk(str(self.dirName)):
                    for filename in files:
                        if filename.endswith('.wav'):
                            # if not os.path.isfile(root+'/'+filename+'.data'): # if already processed then skip?
                            #     continue
                            cnt=cnt+1
                            self.statusRight.setText("Processing file " + str(cnt) + "/" + str(total))
                            self.filename=root+'/'+filename
                            self.loadFile()
                            self.seg = Segment.Segment(self.audiodata, self.sgRaw, self.sp, self.sampleRate)
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
                                ws = WaveletSegment.WaveletSegment(species=self.species)
                                newSegments = ws.waveletSegment_test(fName=None,data=self.audiodata, sampleRate=self.sampleRate, species=self.species,trainTest=False)
                            else:
                                self.method = "Default"
                                newSegments=self.seg.bestSegments()

                            # Save the excel file
                            out = SupportClasses.exportSegments(annotation=newSegments, species=self.species,
                                                                dirName=self.dirName, filename=self.filename,
                                                                datalength=self.datalength, sampleRate=self.sampleRate,method=self.method)
                            out.excel()
                            # Save the annotation
                            out.saveAnnotation()
                self.statusLeft.setText("Ready")
            else:
                msg = QMessageBox()
                msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
                msg.setWindowIcon(QIcon('img/Avianz.ico'))
                msg.setText("Please select a folder to process!")
                msg.setWindowTitle("Select Folder")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

    def loadFile(self):
        print self.filename
        wavobj = wavio.read(self.filename)
        self.sampleRate = wavobj.rate
        self.audiodata = wavobj.data
        print np.shape(self.audiodata)

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
        print self.sampleRate

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
