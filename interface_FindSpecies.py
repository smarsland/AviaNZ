import sys, os, json,glob
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import math
import pywt

import wavio
#from scipy.io import wavfile
# from scipy import signal
import librosa
import numpy as np

from openpyxl import load_workbook, Workbook

import pyqtgraph as pg
# pg.setConfigOption('background','w')
# pg.setConfigOption('foreground','k')
# pg.setConfigOption('antialias',True)
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *
import pyqtgraph.functions as fn

# import SupportClasses as SupportClasses
# import Dialogs as Dialogs
# from Dialogs import *

import SignalProc
import Segment
import WaveletSegment

class AviaNZFindSpeciesInterface(QMainWindow):
    # Main class for the interface, which contains most of the user interface and plotting code

    def __init__(self,root=None,configfile=None):
        # Basically allow the user to browse a folder and push a button to process that folder to find a target species
        # and sets up the window.
        super(AviaNZFindSpeciesInterface, self).__init__()
        self.root = root
        self.dirName=[]
        if configfile is not None:
            try:
                self.config = json.load(open(configfile))
                self.saveConfig = False
            except:
                print("Failed to load config file")
                self.genConfigFile()
                self.saveConfig = True
            self.configfile = configfile
        else:
            self.genConfigFile()
            self.saveConfig=True
            self.configfile = 'AviaNZconfig.txt'

        # Make the window and associated widgets
        QMainWindow.__init__(self, root)

        self.setWindowTitle('AviaNZ - Automatic Segmentation')
        self.createFrame()

    def createFrame(self):
        # Make the window and set its size
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.resize(1240,600)
        # self.move(100,50)

        # Make the docks
        self.d_fileList = Dock("Folder",size=(300,150))
        self.d_detection = Dock("Automatic Detection",size=(600,150))

        self.area.addDock(self.d_fileList,'left')
        self.area.addDock(self.d_detection,'right')

        # Put content widgets in the docks
        self.w_fileList = QListWidget(self)
        self.w_fileList.setMinimumWidth(100)
        # self.fileList.connect(self.fileList, SIGNAL('itemDoubleClicked(QListWidgetItem*)'), self.listLoadFile)
        self.d_fileList.addWidget(self.w_fileList)

        self.w_browse = QPushButton("  &Browse Folder to Process")
        self.connect(self.w_browse, SIGNAL('clicked()'), self.browse)
        self.w_browse.setToolTip("Can select a folder with sub folders to process")
        self.w_dir = QLineEdit()
        self.w_dir.setText('')
        self.d_detection.addWidget(self.w_dir,row=0,col=1,colspan=2)
        self.d_detection.addWidget(self.w_browse,row=0,col=0)

        self.w_speLabel = QLabel("  Select Species")
        self.d_detection.addWidget(self.w_speLabel,row=1,col=0)
        self.w_spe = QComboBox()
        self.w_spe.addItems(["Kiwi", "Ruru"])
        self.d_detection.addWidget(self.w_spe,row=1,col=1,colspan=2)

        self.w_methodLabel = QLabel("  Select Method")
        self.d_detection.addWidget(self.w_methodLabel,row=2,col=0)

        # Dialogs.AutoSegmentation(1000,self.d_detection)  #maxv=?

        self.algs = QComboBox()
        self.algs.addItems(["Wavelets", "Amplitude","Harma","Power","Median Clipping","Onsets","Fundamental Frequency","FIR"])
        self.d_detection.addWidget(self.algs,row=2,col=1,colspan=2)
        self.prevAlg = "FIR"
        self.algs.setCurrentIndex(7)
        a = self.algs.currentIndex()
        item = self.algs.itemText(a)
        # print str(item)=="Wavelets"
        # self.algs.currentIndexChanged.connect(self.changeBoxes(item))
        self.algs.currentIndexChanged[QString].connect(self.changeBoxes)

        # Define the whole set of possible options for the dialog box here, just to have them together.
        # Then hide and show them as required as the algorithm chosen changes.

        maxv=3000
        #amplitude threshold
        self.ampThr = QDoubleSpinBox()
        self.ampThr.setRange(0.0,1.0)
        self.ampThr.setSingleStep(0.1)
        self.ampThr.setDecimals(1)
        self.ampThr.setValue(0.5)

        self.amplabel = QLabel("  Set threshold (%) amplitude")
        self.d_detection.addWidget(self.amplabel,row=3,col=0)
        self.amplabel.hide()
        self.d_detection.addWidget(self.ampThr,row=3,col=1,colspan=2)
        self.ampThr.hide()

        self.HarmaThr1 = QSpinBox()
        self.HarmaThr1.setRange(10,50)
        self.HarmaThr1.setSingleStep(1)
        self.HarmaThr1.setValue(10)
        self.HarmaThr2 = QDoubleSpinBox()
        self.HarmaThr2.setRange(0.1,0.95)
        self.HarmaThr2.setSingleStep(0.05)
        self.HarmaThr2.setDecimals(2)
        self.HarmaThr2.setValue(0.9)

        self.Harmalabel = QLabel("  Set decibal threshold")
        self.d_detection.addWidget(self.Harmalabel,row=3,col=0)
        self.Harmalabel.hide()
        self.d_detection.addWidget(self.HarmaThr1,row=3,col=1,colspan=2)
        self.d_detection.addWidget(self.HarmaThr2,row=4, col=1,colspan=2)
        self.HarmaThr1.hide()
        self.HarmaThr2.hide()

        self.PowerThr = QDoubleSpinBox()
        self.PowerThr.setRange(0.0,2.0)
        self.PowerThr.setSingleStep(0.1)
        self.PowerThr.setValue(1.0)

        self.PowerThrLabel = QLabel("  Threshold")
        self.d_detection.addWidget(self.PowerThrLabel,row=3, col=0)
        self.PowerThrLabel.hide()
        self.d_detection.addWidget(self.PowerThr,row=3,col=1,colspan=2)
        self.PowerThr.hide()

        self.Fundminfreqlabel = QLabel("  Min Frequency")
        self.Fundminfreq = QLineEdit()
        self.Fundminfreq.setText('100')
        self.Fundminperiodslabel = QLabel("  Min Number of periods")
        self.Fundminperiods = QSpinBox()
        self.Fundminperiods.setRange(1,10)
        self.Fundminperiods.setValue(3)
        self.Fundthrlabel = QLabel("  Threshold")
        self.Fundthr = QDoubleSpinBox()
        self.Fundthr.setRange(0.1,1.0)
        self.Fundthr.setDecimals(1)
        self.Fundthr.setValue(0.5)
        self.Fundwindowlabel = QLabel("  Window size (will be rounded up as appropriate)")
        self.Fundwindow = QSpinBox()
        self.Fundwindow.setRange(300,5000)
        self.Fundwindow.setSingleStep(500)
        self.Fundwindow.setValue(1000)

        self.d_detection.addWidget(self.Fundminfreqlabel, row=3,col=0)
        self.Fundminfreqlabel.hide()
        self.d_detection.addWidget(self.Fundminfreq,row=3,col=1,colspan=2)
        self.Fundminfreq.hide()
        self.d_detection.addWidget(self.Fundminperiodslabel,row=4,col=0)
        self.Fundminperiodslabel.hide()
        self.d_detection.addWidget(self.Fundminperiods,row=4,col=1,colspan=2)
        self.Fundminperiods.hide()
        self.d_detection.addWidget(self.Fundthrlabel,row=5,col=0)
        self.Fundthrlabel.hide()
        self.d_detection.addWidget(self.Fundthr,row=5,col=1,colspan=2)
        self.Fundthr.hide()
        self.d_detection.addWidget(self.Fundwindowlabel,row=6,col=0)
        self.Fundwindowlabel.hide()
        self.d_detection.addWidget(self.Fundwindow,row=6,col=1,colspan=2)
        self.Fundwindow.hide()

        self.medThr = QDoubleSpinBox()
        self.medThr.setRange(0.2,6)
        self.medThr.setSingleStep(1)
        self.medThr.setDecimals(1)
        self.medThr.setValue(3)

        self.medlabel = QLabel("  Set median threshold")
        self.d_detection.addWidget(self.medlabel,row=3,col=0)
        self.medlabel.hide()
        self.d_detection.addWidget(self.medThr, row=3,col=1,colspan=2)
        self.medThr.hide()

        self.ecThr = QDoubleSpinBox()
        self.ecThr.setRange(0.001,6)
        self.ecThr.setSingleStep(1)
        self.ecThr.setDecimals(3)
        self.ecThr.setValue(1)

        self.eclabel = QLabel("  Set energy curve threshold")
        self.d_detection.addWidget(self.eclabel,row=3,col=0)
        self.eclabel.hide()
        self.ecthrtype = [QRadioButton("N standard deviations"), QRadioButton("Threshold")]
        for i in range(len(self.ecthrtype)):
            self.d_detection.addWidget(self.ecthrtype[i])
            self.ecthrtype[i].hide()
        self.d_detection.addWidget(self.ecThr)
        self.ecThr.hide()

        self.FIRThr1 = QDoubleSpinBox()
        self.FIRThr1.setRange(0.0,2.0) #setRange(0.0,1.0)
        self.FIRThr1.setSingleStep(0.05)
        self.FIRThr1.setValue(0.1)
        self.FIRthrlabel=QLabel("  Set threshold")

        self.d_detection.addWidget(self.FIRthrlabel,row=3,col=0)
        self.d_detection.addWidget(self.FIRThr1, row=3,col=1,colspan=2)
        # self.FIRThr1.hide()

        # self.Onsetslabel = QLabel("  Onsets: No parameters")
        # self.d_detection.addWidget(self.Onsetslabel,row=3,col=1)
        # self.Onsetslabel.hide()

        self.depthlabel = QLabel("  Depth of wavelet packet decomposition")
        #self.depthchoice = QCheckBox()
        #self.connect(self.depthchoice, SIGNAL('clicked()'), self.depthclicked)
        self.depth = QSpinBox()
        self.depth.setRange(1,10)
        self.depth.setSingleStep(1)
        self.depth.setValue(5)

        self.thrtypelabel = QLabel("  Type of thresholding")
        self.thrtype = [QRadioButton("Soft"), QRadioButton("Hard")]
        self.thrtype[0].setChecked(True)

        self.thrlabel = QLabel("  Multiplier of std dev for threshold")
        self.thr = QDoubleSpinBox()
        self.thr.setRange(1,10)
        self.thr.setSingleStep(0.5)
        self.thr.setValue(4.5)

        self.waveletlabel = QLabel("  Type of wavelet")
        self.wavelet = QComboBox()
        self.wavelet.addItems(["dmey2","dmey","db2","db5","haar"])
        self.wavelet.setCurrentIndex(0)

        self.blabel = QLabel("  Start and end points for bandpass filter")
        self.start = QLineEdit()
        self.start.setText('1000')
        self.end = QLineEdit()
        self.end.setText('7500')
        self.blabel2 = QLabel("  Check if not using bandpass")
        self.bandchoice = QCheckBox()
        # self.connect(self.bandchoice, SIGNAL('clicked()'), self.bandclicked)

        self.d_detection.addWidget(self.depthlabel,row=3,col=0)
        self.depthlabel.hide()
        #Box.addWidget(self.depthchoice)
        #self.depthchoice.hide()
        self.d_detection.addWidget(self.depth, row=3,col=1,colspan=2)
        self.depth.hide()

        self.d_detection.addWidget(self.thrtypelabel,row=4,col=0)
        self.thrtypelabel.hide()
        self.d_detection.addWidget(self.thrtype[0],row=4,col=1)
        self.thrtype[0].hide()
        self.d_detection.addWidget(self.thrtype[1],row=4,col=2)
        self.thrtype[1].hide()

        self.d_detection.addWidget(self.thrlabel,row=5,col=0)
        self.thrlabel.hide()
        self.d_detection.addWidget(self.thr,row=5,col=1,colspan=2)
        self.thr.hide()

        self.d_detection.addWidget(self.waveletlabel,row=6,col=0)
        self.waveletlabel.hide()
        self.d_detection.addWidget(self.wavelet,row=6,col=1,colspan=2)
        self.wavelet.hide()

        self.d_detection.addWidget(self.blabel,row=7,col=0)
        self.blabel.hide()
        self.d_detection.addWidget(self.start,row=7,col=1,colspan=2)
        self.start.hide()
        self.d_detection.addWidget(self.end,row=8,col=1,colspan=2)
        self.end.hide()
        self.d_detection.addWidget(self.blabel2,row=9,col=0)
        self.blabel2.hide()
        self.d_detection.addWidget(self.bandchoice,row=9,col=1,colspan=2)
        self.bandchoice.hide()

        self.w_processButton = QPushButton("&Process Folder")
        self.connect(self.w_processButton, SIGNAL('clicked()'), self.detect)
        self.d_detection.addWidget(self.w_processButton,row=10,col=2)
        self.w_processButton.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')

        self.statusBar().showMessage("Ready")

        # Store the state of the docks
        self.state = self.area.saveState()

        # Plot everything
        self.show()

    def changeBoxes(self,alg):
        # This does the hiding and showing of the options as the algorithm changes
        # print self.prevAlg
        if self.prevAlg == "Amplitude":
            self.amplabel.hide()
            self.ampThr.hide()
        elif self.prevAlg == "Energy Curve":
            self.eclabel.hide()
            self.ecThr.hide()
            for i in range(len(self.ecthrtype)):
                self.ecthrtype[i].hide()
            #self.ecThr.hide()
        elif self.prevAlg == "Harma":
            self.Harmalabel.hide()
            self.HarmaThr1.hide()
            self.HarmaThr2.hide()
        elif self.prevAlg == "Power":
            self.PowerThr.hide()
            self.PowerThrLabel.hide()
        elif self.prevAlg == "Median Clipping":
            self.medlabel.hide()
            self.medThr.hide()
        elif self.prevAlg == "Fundamental Frequency":
            self.Fundminfreq.hide()
            self.Fundminperiods.hide()
            self.Fundthr.hide()
            self.Fundwindow.hide()
            self.Fundminfreqlabel.hide()
            self.Fundminperiodslabel.hide()
            self.Fundthrlabel.hide()
            self.Fundwindowlabel.hide()
        elif self.prevAlg == "Onsets":
            # self.Onsetslabel.hide()
            pass
        elif self.prevAlg == "FIR":
            self.FIRThr1.hide()
            self.FIRthrlabel.hide()
        else:
            # self.wavlabel.hide()
            self.depthlabel.hide()
            self.depth.hide()
            #self.depthchoice.hide()
            self.thrtypelabel.hide()
            self.thrtype[0].hide()
            self.thrtype[1].hide()
            self.thrlabel.hide()
            self.thr.hide()
            self.waveletlabel.hide()
            self.wavelet.hide()
            self.blabel.hide()
            self.start.hide()
            self.end.hide()
            self.blabel2.hide()
            self.bandchoice.hide()
        self.prevAlg = str(alg)

        if str(alg) == "Amplitude":
            self.amplabel.show()
            self.ampThr.show()
        elif str(alg) == "Energy Curve":
            self.eclabel.show()
            self.ecThr.show()
            for i in range(len(self.ecthrtype)):
                self.ecthrtype[i].show()
            self.ecThr.show()
            self.w_processButton.show()
        elif str(alg) == "Harma":
            self.Harmalabel.show()
            self.HarmaThr1.show()
            self.HarmaThr2.show()
        elif str(alg) == "Power":
            self.PowerThr.show()
            self.PowerThrLabel.show()
        elif str(alg) == "Median Clipping":
            self.medlabel.show()
            self.medThr.show()
        elif str(alg) == "Fundamental Frequency":
            self.Fundminfreq.show()
            self.Fundminperiods.show()
            self.Fundthr.show()
            self.Fundwindow.show()
            self.Fundminfreqlabel.show()
            self.Fundminperiodslabel.show()
            self.Fundthrlabel.show()
            self.Fundwindowlabel.show()
        elif str(alg) == "Onsets":
            # self.Onsetslabel.show()
            pass
        elif str(alg) == "FIR":
            self.FIRThr1.show()
            self.FIRthrlabel.show()
        else:
            #"Wavelets"
            # self.wavlabel.show()
            # self.depthlabel.show()
            #self.depthchoice.show()
            # self.depth.show()
            # self.thrtypelabel.show()
            # self.thrtype[0].show()
            # self.thrtype[1].show()
            self.thrlabel.show()
            self.thr.show()
            # self.waveletlabel.show()
            # self.wavelet.show()
            # self.blabel.show()
            # self.start.show()
            # self.end.show()
            # self.blabel2.show()
            # self.bandchoice.show()

    # def getValues(self):
    #     return [self.algs.currentText(),self.ampThr.text(),self.medThr.text(),self.HarmaThr1.text(),self.HarmaThr2.text(),self.PowerThr.text(),self.Fundminfreq.text(),self.Fundminperiods.text(),self.Fundthr.text(),self.Fundwindow.text(),self.FIRThr1.text(),self.depth.text(),self.thrtype[0].isChecked(),self.thr.text(),self.wavelet.currentText(),self.bandchoice.isChecked(),self.start.text(),self.end.text()]


    def browse(self):
        # self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',"Wav files (*.wav)")
        if self.dirName:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirName))
        else:
            self.dirName = QtGui.QFileDialog.getExistingDirectory(self,'Choose Folder to Process')
        print "Dir:", self.dirName
        self.setWindowTitle('AviaNZ - '+ self.dirName)
        self.w_dir.setText(self.dirName)

        files=[]
        self.w_fileList.clear()
        for f in os.listdir(self.dirName):
            # if f.endswith(".wav"): #os.path.isfile(f) and
            self.w_fileList.addItem(f)
            files.append(f)
        print files

    def detect(self):
        # print self.dirName
        # print self.dirName[:-1]
        # nfiles=len(glob.glob(os.path.join(str(self.dirName[:-1]),'*.wav')))
        # print nfiles
        # print os.path.join(self.dirName[:-1],'*.wav')
        if self.dirName:
            self.statusBar().showMessage("Processing...")
            i=self.w_spe.currentIndex()
            if i==0:
                self.species="Kiwi"
            elif i==1:
                self.species="Ruru"

            for root, dirs, files in os.walk(str(self.dirName)):
                for filename in files:
                    if filename.endswith('.wav'):
                        # if not os.path.isfile(root+'/'+filename+'.data'): # if already processed then skip
                        #     continue
                        self.filename=root+'/'+filename
                        self.loadFile()

                        self.seg = Segment.Segment(self.audiodata, self.sgRaw, self.sp, self.sampleRate, self.config['minSegment'],self.config['window_width'], self.config['incr'])
                        # print self.algs.itemText(self.algs.currentIndex())
                        if self.algs.currentText() == "Amplitude":
                            newSegments = self.seg.segmentByAmplitude(float(str(self.ampThr.text())))
                        elif self.algs.currentText() == "Median Clipping":
                            newSegments = self.seg.medianClip(float(str(self.medThr.text())))
                            #print newSegments
                        elif self.algs.currentText() == "Harma":
                            newSegments = self.seg.Harma(float(str(self.HarmaThr1.text())),float(str(self.HarmaThr2.text())))
                        elif self.algs.currentText() == "Power":
                            newSegments = self.seg.segmentByPower(float(str(self.PowerThr.text())))
                        elif self.algs.currentText() == "Onsets":
                            newSegments = self.seg.onsets()
                            #print newSegments
                        elif self.algs.currentText() == "Fundamental Frequency":
                            newSegments, pitch, times = self.seg.yin(int(str(self.Fundminfreq.text())),int(str(self.Fundminperiods.text())),float(str(self.Fundthr.text())),int(str(self.Fundwindow.text())),returnSegs=True)
                            print newSegments
                        elif self.algs.currentText() == "FIR":
                            print float(str(self.FIRThr1.text()))
                            # newSegments = self.seg.segmentByFIR(0.1)
                            newSegments = self.seg.segmentByFIR(float(str(self.FIRThr1.text())))
                            # print newSegments
                        elif self.algs.currentText()=='Wavelets':
                            newSegments = WaveletSegment.findCalls_test(fName=None,data=self.audiodata, sampleRate=self.sampleRate, species=self.species,trainTest=False)
                            # TODO: remove short segments?

                        # Generate Binary output ('Binary)
                        n=math.ceil(float(self.datalength)/self.sampleRate)
                        detected=np.zeros(int(n))
                        for seg in newSegments:
                            for a in range(len(detected)):
                                if math.floor(seg[0])<=a and a<math.ceil(seg[1]):
                                    detected[a]=1
                        self.saveSegments(detected, mode='Binary') # append

                        # Generate annotation friendly output ('Annotation')
                        # print "Generate annotation friendly output", newSegments
                        annotation=[]
                        if len(newSegments)>0:
                            if self.algs.currentText()=='Wavelets':
                                mergedSeg=self.mergeSeg(newSegments)
                                if len(mergedSeg)>0:
                                    for seg in mergedSeg:
                                        annotation.append([float(seg[0]),float(seg[1]),0,0,self.species+'?'])
                            elif len(newSegments)>0:
                                for seg in newSegments:
                                    annotation.append([float(seg[0]),float(seg[1]),0,0,self.species+'?'])
                                # print annotation
                            self.saveSegments(annotation, mode='Annotation')

                        # Generate excel summary - time stamps [start(mm:ss) end(mm:ss)]
                        annotation=[]
                        for seg in newSegments:
                            annotation.append([self.convertMillisecs(seg[0]*1000),self.convertMillisecs(seg[1]*1000)])
                        # print annotation
                        self.saveSegments(annotation, mode='Excel') # append
            self.statusBar().showMessage("Ready")
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setText("Please select a folder to process!")
            msg.setWindowTitle("Select Folder")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        # return newSegments

    def mergeSeg(self,segments):
        indx=[]
        for i in range(len(segments)-1):
            if segments[i][1]==segments[i+1][0]:
                indx.append(i)
        indx.reverse()
        for i in indx:
            segments[i][1]=segments[i+1][1]
            del(segments[i+1])
        return segments

    def saveSegments(self,annotation,mode):
        # This saves the detections into three different formats: annotation, excel, and binary
        # print("Saving detections to "+self.filename)

        method=self.algs.currentText()
        relfname=os.path.relpath(str(self.filename),str(self.dirName))

        if mode=='Annotation':
            if isinstance(self.filename, str):
                file = open(self.filename + '.data', 'w')
            else:
                file = open(str(self.filename) + '.data', 'w')
            json.dump(annotation,file)
        elif mode=='Excel':
            eFile=self.dirName+'\AutoDet_'+self.species+'_'+method+'.xlsx'

            if os.path.isfile(eFile):   #if the file is already there
                try:
                    wb = load_workbook(str(eFile))
                    ws=wb.get_sheet_by_name('Sheet')
                    c=1
                    r=ws.max_row+1 # TODO: get last row number from existing file
                    ws.cell(row=r,column=1,value=str(relfname))
                    for seg in annotation:
                        ws.cell(row=r,column=c+1,value=seg[0]+'-'+seg[1])
                        c=c+1
                    # ws.append([relfname, str(annotation)])    # Append this filename and its detections
                    wb.save(str(eFile))
                except:
                    print "Unable to open file"           #Does not exist OR no read permissions
            else:
                wb = Workbook()
                ws = wb.active
                # print "Here Annotation:",type(annotation), annotation
                ws.append(["File Name","Detections [start(mm:ss),end(mm:ss)]"])
                ws.cell(row=1,column=1, value="File Name")
                ws.cell(row=1,column=2, value="Detections [start(mm:ss),end(mm:ss)]")
                c=1
                r=2
                ws.cell(row=r,column=c,value=str(relfname))
                for seg in annotation:
                    ws.cell(row=r,column=c+1,value=seg[0]+'-'+seg[1])
                    c=c+1
                # ws.append([relfname, str(annotation)])   # Append this filename and its detections
                wb.save(str(eFile))
        else:   # mode=='Binary'
            eFile=self.dirName+'\AutoDet_'+self.species+'_'+method+'_sec.xlsx'
            if os.path.isfile(eFile):   #if the file is already there
                try:
                    wb = load_workbook(str(eFile))
                    ws=wb.get_sheet_by_name('Sheet')
                    ws.append([relfname, str(annotation)])    # Append this filename and its detections
                    wb.save(str(eFile))
                except:
                    print "Unable to open file"           #Does not exist OR no read permissions
            else:
                wb = Workbook()
                ws = wb.active
                ws.append([relfname, str(annotation)])   # Append this filename and its detections
                wb.save(str(eFile))

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

        if (self.species=='Kiwi' or self.species=='Ruru') and self.sampleRate!=16000:
            self.audiodata = librosa.core.audio.resample(self.audiodata,self.sampleRate,16000)
            self.sampleRate=16000
            self.datalength = np.shape(self.audiodata)[0]
        print self.sampleRate
        self.minFreq = 0
        self.maxFreq = self.sampleRate/2.

        # Create an instance of the Signal Processing class
        if not hasattr(self,'sp'):
            self.sp = SignalProc.SignalProc(self.audiodata, self.sampleRate,self.config['window_width'],self.config['incr'])

        # Get the data for the spectrogram
        self.sgRaw = self.sp.spectrogram(self.audiodata,self.sampleRate,mean_normalise=True,onesided=True,multitaper=False)
        maxsg = np.min(self.sgRaw)
        self.sg = np.abs(np.where(self.sgRaw==0,0.0,10.0 * np.log10(self.sgRaw/maxsg)))

        # Update the data that is seen by the other classes
        # TODO: keep an eye on this to add other classes as required
        if hasattr(self,'seg'):
            self.seg.setNewData(self.audiodata,self.sgRaw,self.sampleRate,self.config['window_width'],self.config['incr'])
        else:
            self.seg = Segment.Segment(self.audiodata, self.sgRaw, self.sp, self.sampleRate, self.config['minSegment'],
                                       self.config['window_width'], self.config['incr'])
        self.sp.setNewData(self.audiodata,self.sampleRate)

        self.setWindowTitle('AviaNZ - ' + self.filename)

    def convertMillisecs(self,millisecs):
        seconds = (millisecs / 1000) % 60
        minutes = (millisecs / (1000 * 60)) % 60
        return "%02d" % minutes+":"+"%02d" % seconds
