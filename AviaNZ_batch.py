import os, re

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

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
                                out = SupportClasses.exportSegments(segments=newSegments, confirmedSegments=[], segmentstoCheck=post.segments, species=self.species, startTime=sTime,
                                                                    dirName=self.dirName, filename=self.filename,
                                                                    datalength=self.datalength, sampleRate=self.sampleRate,method=self.method, resolution=self.w_res.value())
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
