import os, json, re
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
import pywt
import WaveletFunctions
import WaveletSegment
import SupportClasses

class AviaNZFindSpeciesInterface(QMainWindow):
    # Main class for batch processing

    def __init__(self,root=None,minSegment=50, DOC=True):
        # Allow the user to browse a folder and push a button to process that folder to find a target species
        # and sets up the window.
        super(AviaNZFindSpeciesInterface, self).__init__()
        self.root = root
        self.dirName=[]
        # self.minSegment=minSegment
        self.DOC=DOC

        # Make the window and associated widgets
        QMainWindow.__init__(self, root)

        # add statusbar
        # self.statusLeft = QLabel("Ready")
        # self.statusLeft.setFrameStyle(QFrame.Panel)
        # self.statusRight = QLabel("Processing file Current/Total")
        # self.statusRight.setAlignment(Qt.AlignRight)
        # self.statusRight.setFrameStyle(QFrame.Panel)
        # statusStyle='QLabel {border:transparent}'
        # self.statusLeft.setStyleSheet(statusStyle)
        # self.statusRight.setStyleSheet(statusStyle)
        # self.statusBar().addPermanentWidget(self.statusLeft,1)
        # self.statusBar().addPermanentWidget(self.statusRight,1)

        # # Set the message in the status bar
        # self.statusLeft.setText("Ready")
        # # self.statusRight.setText("Processing file Current/Total")
        self.statusBar().showMessage("Processing file Current/Total")

        self.setWindowTitle('AviaNZ - Automatic Detection')
        self.createFrame()

    def createFrame(self):
        # Make the window and set its size
        self.area = DockArea()
        self.setCentralWidget(self.area)
        if self.DOC==True:
            self.setFixedSize(500,300)
        else:
            self.setFixedSize(1000, 400)

        # Make the docks
        self.d_detection = Dock("Automatic Detection",size=(350,100))
        self.d_detection.hideTitleBar()

        # self.area.addDock(self.d_fileList,'left')
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
        self.w_spe1.selected.connect(self.cleanStatus)
        #self.connect(self.w_spe1, SIGNAL('selected()'), self.cleanStatus)

        self.w_resLabel = QLabel("  Output Resolution (secs)")
        self.d_detection.addWidget(self.w_resLabel, row=2, col=0)
        self.w_res = QSpinBox()
        self.w_res.setRange(1,600)
        self.w_res.setSingleStep(5)
        self.w_res.setValue(60)
        self.d_detection.addWidget(self.w_res, row=2, col=1, colspan=2)

        self.w_processButton = QPushButton("&Process Folder")
        self.w_processButton.clicked.connect(self.detect)
        #self.connect(self.w_processButton, SIGNAL('clicked()'), self.detect)
        self.d_detection.addWidget(self.w_processButton,row=11,col=2)
        self.w_processButton.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')

        if not self.DOC:
            self.d_denoise = Dock("Denoising", size=(350, 100))
            self.area.addDock(self.d_denoise, 'right')
            self.w_denoiseButton = QPushButton("&Wavelet Denoise (d+f)")
            self.w_denoiseButton.clicked.connect(self.denoise_df)
            #self.connect(self.w_denoiseButton, SIGNAL('clicked()'), self.denoise_df)
            self.d_denoise.addWidget(self.w_denoiseButton,row=12,col=2)
            #self.w_denoiseButton.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')

            self.w_filterButton = QPushButton("&Filter (f)")
            self.w_filterButton.clicked.connect(self.denoise_f)
            #self.connect(self.w_filterButton, SIGNAL('clicked()'), self.denoise_f)
            self.d_denoise.addWidget(self.w_filterButton, row=13, col=2)

            self.w_downButton = QPushButton("&DownSample)")
            self.w_downButton.clickedconnect(self.downSample)
            #self.connect(self.w_downButton, SIGNAL('clicked()'), self.downSample)
            self.d_denoise.addWidget(self.w_downButton, row=14, col=2)

            self.w_browse2 = QPushButton("  &Browse Folder")
            # self.connect(self.w_browse2, SIGNAL('clicked()'), self.browse_denoise)
            self.w_browse2.setToolTip("Can select a folder with sub folders to process")
            self.w_browse2.setFixedHeight(50)
            self.w_browse2.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')
            self.w_dir2 = QPlainTextEdit()
            self.w_dir2.setFixedHeight(50)
            self.w_dir2.setPlainText('')
            self.w_dir2.setToolTip("The folder being processed")
            self.d_denoise.addWidget(self.w_dir2, row=0, col=1, colspan=2)
            self.d_denoise.addWidget(self.w_browse2, row=0, col=0)

            self.w_speLabel2 = QLabel("  Select Species")
            self.d_detection.addWidget(self.w_speLabel2, row=1, col=0)
            self.w_spe2 = QComboBox()
            self.w_spe2.addItems(["Kiwi", "Ruru", "Bittern", "all"])
            self.w_spe2.selected.connect(self.cleanStatus)
            #self.connect(self.w_spe2, SIGNAL('selected()'),self.cleanStatus)
            self.d_denoise.addWidget(self.w_spe2, row=1, col=1, colspan=2)

            self.d_detection.showTitleBar()
        # self.statusLeft.setText("Ready")

        self.w_browse1.clicked.connect(self.browse_detect)
        #self.connect(self.w_browse1, SIGNAL('clicked()'), self.browse_detect)
        self.w_browse2.clicked.connect(self.browse_denoise)
        #self.connect(self.w_browse2, SIGNAL('clicked()'), self.browse_denoise)


        # Plot everything
        self.show()

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
        print "Dir:", self.dirName
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
                            print filename
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
                                    ws = WaveletSegment.WaveletSegment(species=self.species)
                                    newSegments = ws.waveletSegment_test(fName=None,data=self.audiodata, sampleRate=self.sampleRate, species=self.species,trainTest=False)
                                    print "in batch", newSegments
                                else:
                                    self.method = "Default"
                                    self.seg = Segment.Segment(self.audiodata, self.sgRaw, self.sp, self.sampleRate)
                                    newSegments=self.seg.bestSegments()
                                    # print newSegments

                                # Postprocess to remove FPs
                                # delete short segments
                                postProc = SupportClasses.postProcess(audioData=self.audiodata, sampleRate=self.sampleRate, segments=newSegments, species=self.species)
                                postProc.deleteShort()
                                # postProc.deleteWindRain(windTest=True, rainTest=False, T_ERatio=1.5)
                                # print "after postProc", postProc.segmentstoCheck
                                # Save the excel file
                                out = SupportClasses.exportSegments(segments=newSegments, confirmedSegments=postProc.confirmedSegments, segmentstoCheck=postProc.segmentstoCheck, species=self.species, startTime=sTime,
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

    def denoise_df(self):
        self.denoise(d=True)

    def denoise_f(self):
        self.denoise(d=False)

    def denoise(self, d=False):
        with pg.BusyCursor():
            if self.dirName:
                self.w_processButton.setDisabled(True)
                # self.statusLeft.setText("Processing...")
                i = self.w_spe2.currentIndex()
                if i == 0:
                    self.species = "Kiwi"
                    fs = 16000
                    f1 = 1100
                    f2 = 7000
                elif i == 1:
                    self.species = "Ruru"
                    fs = 16000
                    f1 = 500
                    f2 = 7000
                elif i == 2:
                    self.species = "Bittern"
                    fs = 1000
                    f1 = 10
                    f2 = 200
                else:  # All
                    self.species = "all"

                total = 0
                for root, dirs, files in os.walk(str(self.dirName)):
                    for filename in files:
                        if filename.endswith('.wav'):
                            total = total + 1
                cnt = 0  # processed number of files

                for root, dirs, files in os.walk(str(self.dirName)):
                    for filename in files:
                        if filename.endswith('.wav'):
                            cnt = cnt + 1
                            self.statusBar().showMessage("Denoising file " + str(cnt) + "/" + str(total) + "...")
                            self.filename = root + '/' + filename
                            self.loadFile()

                            if self.species != 'all':
                                if self.sampleRate != fs:
                                    self.audiodata = librosa.core.audio.resample(self.audiodata, self.sampleRate, fs)
                                    self.sampleRate = fs
                                if d:
                                    wf = WaveletFunctions.WaveletFunctions(data=self.audiodata, wavelet='dmey2',maxLevel=5)
                                    denoisedData= wf.waveletDenoise()
                                else:
                                    denoisedData = self.audiodata   # skip denoising
                                sp = SignalProc.SignalProc(data=denoisedData, sampleRate=self.sampleRate)
                                denoisedData = sp.bandpassFilter(start=f1,end=f2)
                                # denoisedData = sp.ButterworthBandpass(data=denoisedData, sampleRate=self.sampleRate, low=100, high=200, order=10)

                                # ws = WaveletSegment.WaveletSegment(data=self.audiodata, sampleRate=self.sampleRate, species=self.species)
                                # denoisedData = ws.preprocess(species=self.species)
                                # save denoised audio
                                if d:
                                    filename = self.filename[:-4] + '_df' + self.filename[-4:]
                                else:
                                    filename = self.filename[:-4] + '_f' + self.filename[-4:]
                                wavio.write(filename, denoisedData, fs, sampwidth=2)
                self.statusBar().showMessage("Denoised files " + str(cnt) + "/" + str(total))
            else:
                msg = QMessageBox()
                msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
                msg.setWindowIcon(QIcon('img/Avianz.ico'))
                msg.setText("Please select your folder to denoise!")
                msg.setWindowTitle("Select Folder")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            self.w_processButton.setDisabled(False)

    def denoiseBittern(self):
        '''
        reconstruct recording using tw noodes (10 and 4)
        '''
        with pg.BusyCursor():
            if self.dirName:
                self.w_processButton.setDisabled(True)
                # self.statusLeft.setText("Processing...")
                i = self.w_spe2.currentIndex()
                if i == 2:
                    self.species = "Bittern"
                    fs = 1000
                    f1 = 100
                    f2 = 200
                else:  # All
                    self.species = "all"

                total = 0
                for root, dirs, files in os.walk(str(self.dirName)):
                    for filename in files:
                        if filename.endswith('.wav'):
                            total = total + 1
                cnt = 0  # processed number of files

                for root, dirs, files in os.walk(str(self.dirName)):
                    for filename in files:
                        if filename.endswith('.wav'):
                            cnt = cnt + 1
                            self.statusBar().showMessage("Denoising file " + str(cnt) + "/" + str(total) + "...")
                            self.filename = root + '/' + filename
                            self.loadFile()

                            if self.species != 'all':
                                if self.sampleRate != fs:
                                    self.audiodata = librosa.core.audio.resample(self.audiodata, self.sampleRate, fs)
                                    self.sampleRate = fs
                                wf = WaveletFunctions.WaveletFunctions(data=self.audiodata, wavelet='dmey2',maxLevel=5)
                                # # I want to save node 10 output
                                # denoisedData= wf.waveletDenoise()
                                # sp = SignalProc.SignalProc(data=denoisedData, sampleRate=self.sampleRate)
                                # denoisedData = sp.bandpassFilter(start=f1,end=f2)
                                # # denoisedData = sp.ButterworthBandpass(data=denoisedData, sampleRate=self.sampleRate, low=100, high=200, order=10)
                                ws = WaveletSegment.WaveletSegment(data=self.audiodata, sampleRate=self.sampleRate, species=self.species)
                                denoisedData = ws.preprocess(species=self.species)
                                wp = pywt.WaveletPacket(data=denoisedData, wavelet=wf.wavelet, mode='symmetric', maxlevel=5)
                                new_wp=pywt.WaveletPacket(data=None, wavelet=wp.wavelet, mode='symmetric', maxlevel=wp.maxlevel)
                                bin = wf.ConvertWaveletNodeName(10)
                                new_wp[bin] = wp[bin].data
                                bin = wf.ConvertWaveletNodeName(4)
                                new_wp[bin] = wp[bin].data

                                # Get the coefficients
                                node4_10 = np.abs(new_wp.reconstruct(update=False))

                                # ws = WaveletSegment.WaveletSegment(data=self.audiodata, sampleRate=self.sampleRate, species=self.species)
                                # denoisedData = ws.preprocess(species=self.species)
                                # save denoised audio
                                filename = self.filename[:-4] + '_node4_10' + self.filename[-4:]
                                wavio.write(filename, node4_10, fs, sampwidth=2)
                self.statusBar().showMessage("Denoised files " + str(cnt) + "/" + str(total))
            else:
                msg = QMessageBox()
                msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
                msg.setWindowIcon(QIcon('img/Avianz.ico'))
                msg.setText("Please select your folder to denoise!")
                msg.setWindowTitle("Select Folder")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            self.w_processButton.setDisabled(False)

    def downSample(self):
        '''
        downsample for bittern
        '''
        with pg.BusyCursor():
            if self.dirName:
                self.w_processButton.setDisabled(True)
                # self.statusLeft.setText("Processing...")
                i = self.w_spe2.currentIndex()
                if i == 2:
                    self.species = "Bittern"
                    fs = 2000
                else:  # All
                    self.species = "all"

                total = 0
                for root, dirs, files in os.walk(str(self.dirName)):
                    for filename in files:
                        if filename.endswith('.wav'):
                            total = total + 1
                cnt = 0  # processed number of files

                for root, dirs, files in os.walk(str(self.dirName)):
                    for filename in files:
                        if filename.endswith('.wav'):
                            cnt = cnt + 1
                            self.statusBar().showMessage("Downsampling file " + str(cnt) + "/" + str(total) + "...")
                            self.filename = root + '/' + filename
                            self.loadFile()

                            if self.species != 'all':
                                if self.sampleRate != fs:
                                    self.audiodata = librosa.core.audio.resample(self.audiodata, self.sampleRate, fs)
                                    self.sampleRate = fs
                                # save downsampled audio
                                filename = root + '/down/' + filename
                                # filename = self.filename[:-4] + '_down' + self.filename[-4:]
                                wavio.write(filename, self.audiodata, fs, sampwidth=2)
                self.statusBar().showMessage("downsampled files " + str(cnt) + "/" + str(total))
            else:
                msg = QMessageBox()
                msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
                msg.setWindowIcon(QIcon('img/Avianz.ico'))
                msg.setText("Please select your folder to downsample!")
                msg.setWindowTitle("Select Folder")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            self.w_processButton.setDisabled(False)

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
