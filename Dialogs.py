# Version 0.2 10/7/17
# Author: Stephen Marsland

# Dialogs used by the AviaNZ program
# Since most of them just get user selections, they are mostly just a mess of UI things
import sys,os
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.functions as fn
import numpy as np
import SupportClasses as SupportClasses
import json
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

#======
class StartScreen(QDialog):
    def __init__(self, parent=None,DOC=True):
        QDialog.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle('AviaNZ - Choose Task')
        self.setAutoFillBackground(False)
        self.setFixedSize(681, 315)
        self.setStyleSheet("background-image: url(img/AviaNZ_SW_Large.jpg);")
        self.activateWindow()

        self.DOC=DOC

        btn_style='QPushButton {background-color: #A3C1DA; color: white; font-size:20px; font-weight: bold; font-family: "Arial"}'
        # btn_style2='QPushButton {background-color: #A3C1DA; color: grey; font-size:16px}'
        b1 = QPushButton(" Manual Segmentation ")
        b2 = QPushButton("      Find a Species      ")
        #b3 = QPushButton("Denoise a folder")
        l1 = QLabel("-------")
        b1.setStyleSheet(btn_style)
        b2.setStyleSheet(btn_style)
        #b3.setStyleSheet(btn_style2)
        l1.setStyleSheet('QLabel {color:transparent}')

        hbox = QHBoxLayout()
        # hbox.addStretch(1)
        hbox.addWidget(l1)
        hbox.addWidget(b1)
        hbox.addWidget(l1)
        hbox.addWidget(b2)
        hbox.addWidget(l1)
        #hbox.addWidget(b3)
        # b3.setEnabled(False)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addWidget(l1)

        self.setLayout(vbox)

        # self.setGeometry(300, 300, 430, 210)

        self.connect(b1, SIGNAL('clicked()'), self.manualSeg)
        # if DOC==False:
        self.connect(b2, SIGNAL('clicked()'), self.findSpecies)
        # self.connect(b3, SIGNAL('clicked()'), self.denoise)

        # vbox = QVBoxLayout()
        # for w in [b1, b2, b3]:
        #         vbox.addWidget(w)
        #
        # self.setLayout(vbox)
        self.task = -1

    def manualSeg(self):
        self.task = 1
        self.accept()

    def findSpecies(self):
        self.task = 2
        self.accept()

    def denoise(self):
        self.task = 3
        self.accept()

    def getValues(self):
        return self.task

#======
class FileDataDialog(QDialog):
    def __init__(self, name, date, time, parent=None):
        super(FileDataDialog, self).__init__(parent)

        layout = QVBoxLayout(self)

        l1 = QLabel("Annotator")
        self.name = QLineEdit(self)
        self.name.setText(name)

        l2 = QLabel("Data recorded: " + date)
        l3 = QLabel("Time recorded: " + time)

        layout.addWidget(l1)
        layout.addWidget(self.name)
        layout.addWidget(l2)
        layout.addWidget(l3)

        button = QPushButton("OK")
        button.clicked.connect(self.accept)

        layout.addWidget(button)

    def getData(self):
        return

#======
class Spectrogram(QDialog):
    # Class for the spectrogram dialog box
    # TODO: Steal the graph from Raven (View/Configure Brightness)
    def __init__(self, width, incr, minFreq, maxFreq, sampleRate, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Spectrogram Options')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setMinimumWidth(300)

        self.windowType = QComboBox()
        self.windowType.addItems(['Hann','Parzen','Welch','Hamming','Blackman','BlackmanHarris'])

        self.mean_normalise = QCheckBox()
        self.mean_normalise.setChecked(True)
        self.multitaper = QCheckBox()

        self.low = QSpinBox()
        self.low.setRange(minFreq,sampleRate/2)
        self.low.setSingleStep(100)
        self.low.setValue(0)

        self.high = QSpinBox()
        self.high.setRange(minFreq,sampleRate/2)
        self.high.setSingleStep(100)
        self.high.setValue(maxFreq)

        self.activate = QPushButton("Update Spectrogram")

        self.window_width = QLineEdit(self)
        self.window_width.setText(str(width))
        self.incr = QLineEdit(self)
        self.incr.setText(str(incr))

        Box = QVBoxLayout()
        Box.addWidget(self.windowType)
        Box.addWidget(QLabel('Mean normalise'))
        Box.addWidget(self.mean_normalise)
        Box.addWidget(QLabel('Multitapering'))
        Box.addWidget(self.multitaper)
        Box.addWidget(QLabel('Window Width'))
        Box.addWidget(self.window_width)
        Box.addWidget(QLabel('Hop'))
        Box.addWidget(self.incr)

        Box.addWidget(QLabel('Frequency range to show'))
        Box.addWidget(QLabel('Lowest frequency'))
        Box.addWidget(self.low)
        Box.addWidget(QLabel('Highest frequency'))
        Box.addWidget(self.high)

        Box.addWidget(self.activate)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        return [self.windowType.currentText(),self.mean_normalise.checkState(),self.multitaper.checkState(),self.window_width.text(),self.incr.text(),self.low.value(),self.high.value()]

    # def closeEvent(self, event):
    #     msg = QMessageBox()
    #     msg.setIcon(QMessageBox.Question)
    #     msg.setText("Do you want to keep the new values?")
    #     msg.setWindowTitle("Closing Spectrogram Dialog")
    #     msg.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
    #     msg.buttonClicked.connect(self.resetValues)
    #     msg.exec_()
    #     return

    # def resetValues(self,button):
    #     print button.text()

#======
class OperatorReviewer(QDialog):
    # Class for the set operator dialog box
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Set Operator/Reviewer')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setMinimumWidth(320)

        self.operatorlabel = QLabel("Operator")
        self.name1 = QLineEdit(self)
        self.reviewerlabel = QLabel("Reviewer")
        self.name2 = QLineEdit(self)
        self.activate = QPushButton("Set")

        Box = QVBoxLayout()
        Box.addWidget(self.operatorlabel)
        Box.addWidget(self.name1)
        Box.addWidget(self.reviewerlabel)
        Box.addWidget(self.name2)
        Box.addWidget(self.activate)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        return [self.name1.text(),self.name2.text()]

#======
class Segmentation(QDialog):
    # Class for the segmentation dialog box
    # TODO: add the wavelet params
    # TODO: work out how to return varying size of params, also process them
    # TODO: test and play
    def __init__(self, maxv, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Segmentation Options')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setMinimumWidth(350)

        self.algs = QComboBox()
        #self.algs.addItems(["Amplitude","Energy Curve","Harma","Median Clipping","Wavelets"])
        self.algs.addItems(["Default","Median Clipping","Fundamental Frequency","FIR","Wavelets","Harma","Power","Cross-Correlation"])
        self.algs.currentIndexChanged[QString].connect(self.changeBoxes)
        self.prevAlg = "Default"
        self.undo = QPushButton("Undo")
        self.activate = QPushButton("Segment")
        #self.save = QPushButton("Save segments")

        # Define the whole set of possible options for the dialog box here, just to have them together.
        # Then hide and show them as required as the algorithm chosen changes.

        # Spin box for amplitude threshold
        self.ampThr = QDoubleSpinBox()
        self.ampThr.setRange(0.001,maxv+0.001)
        self.ampThr.setSingleStep(0.002)
        self.ampThr.setDecimals(4)
        self.ampThr.setValue(maxv+0.001)

        self.HarmaThr1 = QSpinBox()
        self.HarmaThr1.setRange(10,90)
        self.HarmaThr1.setSingleStep(1)
        self.HarmaThr1.setValue(10)
        self.HarmaThr2 = QDoubleSpinBox()
        self.HarmaThr2.setRange(0.1,0.95)
        self.HarmaThr2.setSingleStep(0.05)
        self.HarmaThr2.setDecimals(2)
        self.HarmaThr2.setValue(0.9)

        self.PowerThr = QDoubleSpinBox()
        self.PowerThr.setRange(0.0,2.0)
        self.PowerThr.setSingleStep(0.1)
        self.PowerThr.setValue(1.0)

        self.Fundminfreqlabel = QLabel("Min Frequency")
        self.Fundminfreq = QLineEdit()
        self.Fundminfreq.setText('100')
        self.Fundminperiodslabel = QLabel("Min Number of periods")
        self.Fundminperiods = QSpinBox()
        self.Fundminperiods.setRange(1,10)
        self.Fundminperiods.setValue(3)
        self.Fundthrlabel = QLabel("Threshold")
        self.Fundthr = QDoubleSpinBox()
        self.Fundthr.setRange(0.1,1.0)
        self.Fundthr.setDecimals(1)
        self.Fundthr.setValue(0.5)
        self.Fundwindowlabel = QLabel("Window size (will be rounded up as appropriate)")
        self.Fundwindow = QSpinBox()
        self.Fundwindow.setRange(300,5000)
        self.Fundwindow.setSingleStep(500)
        self.Fundwindow.setValue(1000)

        self.medThr = QDoubleSpinBox()
        self.medThr.setRange(0.2,6)
        self.medThr.setSingleStep(1)
        self.medThr.setDecimals(1)
        self.medThr.setValue(3)

        self.ecThr = QDoubleSpinBox()
        self.ecThr.setRange(0.001,6)
        self.ecThr.setSingleStep(1)
        self.ecThr.setDecimals(3)
        self.ecThr.setValue(1)

        self.FIRThr1 = QDoubleSpinBox()
        self.FIRThr1.setRange(0.0,2.0) #setRange(0.0,1.0)
        self.FIRThr1.setSingleStep(0.05)
        self.FIRThr1.setValue(0.1)

        self.CCThr1 = QDoubleSpinBox()
        self.CCThr1.setRange(0.0,2.0) #setRange(0.0,1.0)
        self.CCThr1.setSingleStep(0.1)
        self.CCThr1.setValue(0.4)

        Box = QVBoxLayout()
        Box.addWidget(self.algs)
        # Labels
        #self.amplabel = QLabel("Set threshold amplitude")
        #Box.addWidget(self.amplabel)

        self.Harmalabel = QLabel("Set decibal threshold")
        Box.addWidget(self.Harmalabel)
        self.Harmalabel.hide()

        #self.Onsetslabel = QLabel("Onsets: No parameters")
        #Box.addWidget(self.Onsetslabel)
        #self.Onsetslabel.hide()

        self.medlabel = QLabel("Set median threshold")
        Box.addWidget(self.medlabel)
        self.medlabel.hide()

        self.eclabel = QLabel("Set energy curve threshold")
        Box.addWidget(self.eclabel)
        self.eclabel.hide()
        self.ecthrtype = [QRadioButton("N standard deviations"), QRadioButton("Threshold")]

        self.specieslabel = QLabel("Species")
        self.species=QComboBox()
        # self.species.addItems(["Kiwi (M)", "Kiwi (F)", "Ruru"])
        self.species.addItems(["Choose species...","Kiwi","Ruru"])
        self.species.currentIndexChanged[QString].connect(self.changeBoxes)

        Box.addWidget(self.specieslabel)
        self.specieslabel.hide()
        Box.addWidget(self.species)
        self.species.hide()

        Box.addWidget(self.HarmaThr1)
        Box.addWidget(self.HarmaThr2)
        self.HarmaThr1.hide()
        self.HarmaThr2.hide()
        Box.addWidget(self.PowerThr)
        self.PowerThr.hide()

        Box.addWidget(self.medThr)
        self.medThr.hide()
        for i in range(len(self.ecthrtype)):
            Box.addWidget(self.ecthrtype[i])
            self.ecthrtype[i].hide()
        Box.addWidget(self.ecThr)
        self.ecThr.hide()

        Box.addWidget(self.FIRThr1)
        self.FIRThr1.hide()

        Box.addWidget(self.Fundminfreqlabel)
        self.Fundminfreqlabel.hide()
        Box.addWidget(self.Fundminfreq)
        self.Fundminfreq.hide()
        Box.addWidget(self.Fundminperiodslabel)
        self.Fundminperiodslabel.hide()
        Box.addWidget(self.Fundminperiods)
        self.Fundminperiods.hide()
        Box.addWidget(self.Fundthrlabel)
        self.Fundthrlabel.hide()
        Box.addWidget(self.Fundthr)
        self.Fundthr.hide()
        Box.addWidget(self.Fundwindowlabel)
        self.Fundwindowlabel.hide()
        Box.addWidget(self.Fundwindow)
        self.Fundwindow.hide()

        Box.addWidget(self.CCThr1)
        self.CCThr1.hide()

        Box.addWidget(self.undo)
        self.undo.setEnabled(False)
        Box.addWidget(self.activate)
        #Box.addWidget(self.save)

        # Now put everything into the frame
        self.setLayout(Box)

    def changeBoxes(self,alg):
        # This does the hiding and showing of the options as the algorithm changes
        if self.prevAlg == "Default":
            pass
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
        elif self.prevAlg == "Cross-Correlation":
            self.CCThr1.hide()
        #elif self.prevAlg == "Onsets":
        #    self.Onsetslabel.hide()
        elif self.prevAlg == "FIR":
            self.FIRThr1.hide()
        else:
            self.specieslabel.hide()
            self.species.hide()
            #self.depthlabel.hide()
            #self.depth.hide()
            ##self.depthchoice.hide()
            #self.thrtypelabel.hide()
            #self.thrtype[0].hide()
            #self.thrtype[1].hide()
            #self.thrlabel.hide()
            #self.thr.hide()
            #self.waveletlabel.hide()
            #self.wavelet.hide()
            #self.blabel.hide()
            #self.start.hide()
            #self.end.hide()
            #self.blabel2.hide()
            #self.bandchoice.hide()
        self.prevAlg = str(alg)

        if str(alg) == "Default":
            pass
        elif str(alg) == "Energy Curve":
            self.eclabel.show()
            self.ecThr.show()
            for i in range(len(self.ecthrtype)):
                self.ecthrtype[i].show()
            self.ecThr.show()
        elif str(alg) == "Harma":
            self.Harmalabel.show()
            self.HarmaThr1.show()
            self.HarmaThr2.show()
        elif str(alg) == "Power":
            self.PowerThr.show()
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
        #elif str(alg) == "Onsets":
        #    self.Onsetslabel.show()
        elif str(alg) == "FIR":
            self.FIRThr1.show()
        #elif str(alg) == "Best":
        #    pass
        elif self.prevAlg == "Cross-Correlation":
            self.CCThr1.show()
        else:
            #"Wavelets"
            self.specieslabel.show()
            self.species.show()

    def bandclicked(self):
        # TODO: Can they be grayed out?
        self.start.setEnabled(not self.start.isEnabled())
        self.end.setEnabled(not self.end.isEnabled())

    def getValues(self):
        return [self.algs.currentText(),self.medThr.text(),self.HarmaThr1.text(),self.HarmaThr2.text(),self.PowerThr.text(),self.Fundminfreq.text(),self.Fundminperiods.text(),self.Fundthr.text(),self.Fundwindow.text(),self.FIRThr1.text(),self.CCThr1.text(),self.species.currentText()]
        #return [self.algs.currentText(),self.ampThr.text(),self.medThr.text(),self.HarmaThr1.text(),self.HarmaThr2.text(),self.PowerThr.text(),self.Fundminfreq.text(),self.Fundminperiods.text(),self.Fundthr.text(),self.Fundwindow.text(),self.FIRThr1.text(),self.depth.text(),self.thrtype[0].isChecked(),self.thr.text(),self.wavelet.currentText(),self.bandchoice.isChecked(),self.start.text(),self.end.text(),self.species.currentText()]

#======
class Denoise(QDialog):
    # Class for the denoising dialog box
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Denoising Options')
        self.setWindowIcon(QIcon('img/Avianz.ico'))

        self.algs = QComboBox()
        self.algs.addItems(["Wavelets","Bandpass","Butterworth Bandpass" ,"Wavelets --> Bandpass","Bandpass --> Wavelets","Median Filter"])
        self.algs.currentIndexChanged[QString].connect(self.changeBoxes)
        self.prevAlg = "Wavelets"

        # Wavelet: Depth of tree, threshold type, threshold multiplier, wavelet
        self.wavlabel = QLabel("Wavelets")
        self.depthlabel = QLabel("Depth of wavelet packet decomposition (or tick box to use best)")
        self.depthchoice = QCheckBox()
        self.connect(self.depthchoice, SIGNAL('clicked()'), self.depthclicked)
        self.depth = QSpinBox()
        self.depth.setRange(1,10)
        self.depth.setSingleStep(1)
        self.depth.setValue(5)

        self.thrtypelabel = QLabel("Type of thresholding")
        self.thrtype = [QRadioButton("Soft"), QRadioButton("Hard")]
        self.thrtype[0].setChecked(True)

        self.thrlabel = QLabel("Multiplier of std dev for threshold")
        self.thr = QDoubleSpinBox()
        self.thr.setRange(1,10)
        self.thr.setSingleStep(0.5)
        self.thr.setValue(4.5)

        self.waveletlabel = QLabel("Type of wavelet")
        self.wavelet = QComboBox()
        self.wavelet.addItems(["dmey2","db2","db5","haar"])
        self.wavelet.setCurrentIndex(0)

        # Median: width of filter
        self.medlabel = QLabel("Median Filter")
        self.widthlabel = QLabel("Half width of median filter")
        self.width = QSpinBox()
        self.width.setRange(1,101)
        self.width.setSingleStep(1)
        self.width.setValue(11)

        # Bandpass: start and end
        self.bandlabel = QLabel("Bandpass Filter")
        self.wblabel = QLabel("Wavelets and Bandpass Filter")
        self.blabel = QLabel("Start and end points of the band")
        self.start = QLineEdit(self)
        self.start.setText('1000')
        self.end = QLineEdit(self)
        self.end.setText('7500')

        self.trimlabel = QLabel("Make frequency axis tight")
        self.trimaxis = QCheckBox()

        # Want combinations of these too!

        self.activate = QPushButton("Denoise")
        self.undo = QPushButton("Undo")
        self.save = QPushButton("Save Denoised Sound")
        #self.connect(self.undo, SIGNAL('clicked()'), self.undo)
        Box = QVBoxLayout()
        Box.addWidget(self.algs)

        Box.addWidget(self.wavlabel)
        Box.addWidget(self.depthlabel)
        Box.addWidget(self.depthchoice)
        Box.addWidget(self.depth)

        Box.addWidget(self.thrtypelabel)
        Box.addWidget(self.thrtype[0])
        Box.addWidget(self.thrtype[1])

        Box.addWidget(self.thrlabel)
        Box.addWidget(self.thr)

        Box.addWidget(self.waveletlabel)
        Box.addWidget(self.wavelet)

        # Median: width of filter
        Box.addWidget(self.medlabel)
        self.medlabel.hide()
        Box.addWidget(self.widthlabel)
        self.widthlabel.hide()
        Box.addWidget(self.width)
        self.width.hide()

        # Bandpass: start and end
        Box.addWidget(self.bandlabel)
        self.bandlabel.hide()
        Box.addWidget(self.wblabel)
        self.wblabel.hide()
        Box.addWidget(self.blabel)
        self.blabel.hide()
        Box.addWidget(self.start)
        self.start.hide()
        Box.addWidget(self.end)
        self.end.hide()

        Box.addWidget(self.trimlabel)
        self.trimlabel.hide()
        Box.addWidget(self.trimaxis)
        self.trimaxis.hide()

        Box.addWidget(self.activate)
        Box.addWidget(self.undo)
        Box.addWidget(self.save)

        # Now put everything into the frame
        self.setLayout(Box)

    def changeBoxes(self,alg):
        # This does the hiding and showing of the options as the algorithm changes
        if self.prevAlg == "Wavelets":
            self.wavlabel.hide()
            self.depthlabel.hide()
            self.depth.hide()
            self.depthchoice.hide()
            self.thrtypelabel.hide()
            self.thrtype[0].hide()
            self.thrtype[1].hide()
            self.thrlabel.hide()
            self.thr.hide()
            self.waveletlabel.hide()
            self.wavelet.hide()
        elif self.prevAlg == "Bandpass --> Wavelets":
            self.wblabel.hide()
            self.depthlabel.hide()
            self.depth.hide()
            self.depthchoice.hide()
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
            self.trimlabel.hide()
            self.trimaxis.hide()
            self.medlabel.hide()
            self.widthlabel.hide()
            self.width.hide()
        elif self.prevAlg == "Wavelets --> Bandpass":
            self.wblabel.hide()
            self.depthlabel.hide()
            self.depth.hide()
            self.depthchoice.hide()
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
            self.trimlabel.hide()
            self.trimaxis.hide()
            self.medlabel.hide()
            self.widthlabel.hide()
            self.width.hide()
        elif self.prevAlg == "Bandpass" or self.prevAlg == "Butterworth Bandpass":
            self.bandlabel.hide()
            self.blabel.hide()
            self.start.hide()
            self.end.hide()
            self.trimlabel.hide()
            self.trimaxis.hide()
        else:
            # Median filter
            self.medlabel.hide()
            self.widthlabel.hide()
            self.width.hide()

        self.prevAlg = str(alg)
        if str(alg) == "Wavelets":
            self.wavlabel.show()
            self.depthlabel.show()
            self.depthchoice.show()
            self.depth.show()
            self.thrtypelabel.show()
            self.thrtype[0].show()
            self.thrtype[1].show()
            self.thrlabel.show()
            self.thr.show()
            self.waveletlabel.show()
            self.wavelet.show()
        elif str(alg) == "Wavelets --> Bandpass":
            # self.wblabel.show()
            self.depthlabel.show()
            self.depthchoice.show()
            self.depth.show()
            self.thrtypelabel.show()
            self.thrtype[0].show()
            self.thrtype[1].show()
            self.thrlabel.show()
            self.thr.show()
            self.waveletlabel.show()
            self.wavelet.show()
            self.blabel.show()
            self.start.show()
            self.end.show()
            self.trimlabel.show()
            self.trimaxis.show()
        elif str(alg) == "Bandpass --> Wavelets":
            # self.wblabel.show()
            self.depthlabel.show()
            self.depthchoice.show()
            self.depth.show()
            self.thrtypelabel.show()
            self.thrtype[0].show()
            self.thrtype[1].show()
            self.thrlabel.show()
            self.thr.show()
            self.waveletlabel.show()
            self.wavelet.show()
            self.blabel.show()
            self.start.show()
            self.end.show()
            self.trimlabel.show()
            self.trimaxis.show()
        elif str(alg) == "Bandpass" or str(alg) == "Butterworth Bandpass":
            self.bandlabel.show()
            self.start.show()
            self.end.show()
            self.trimlabel.show()
            self.trimaxis.show()
        else:
            #"Median filter"
            self.medlabel.show()
            self.widthlabel.show()
            self.width.show()

    def depthclicked(self):
        self.depth.setEnabled(not self.depth.isEnabled())

    def getValues(self):
        return [self.algs.currentText(),self.depthchoice.isChecked(),self.depth.text(),self.thrtype[0].isChecked(),self.thr.text(),self.wavelet.currentText(),self.start.text(),self.end.text(),self.width.text(),self.trimaxis.isChecked()]

#======
class HumanClassify1(QDialog):
    # This dialog allows the checking of classifications for segments.
    # It shows a single segment at a time, working through all the segments.
    # TODO: Delete segment button

    def __init__(self, sg, audiodata, sampleRate, label, lut, colourStart, colourEnd, cmapInverted, birdList, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Check Classifications')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.frame = QWidget()

        self.lut = lut
        self.colourStart = colourStart
        self.colourEnd = colourEnd
        self.cmapInverted = cmapInverted
        self.label = label
        self.birdList = birdList
        self.saveConfig = False

        # Set up the plot window, then the right and wrong buttons, and a close button
        self.wPlot = pg.GraphicsLayoutWidget()
        self.pPlot = self.wPlot.addViewBox(enableMouse=False, row=0, col=1)
        self.plot = pg.ImageItem()
        self.pPlot.addItem(self.plot)

        self.species = QLabel(self.label)
        font = self.species.font()
        font.setPointSize(24)
        font.setBold(True)
        self.species.setFont(font)

        # The buttons to move through the overview
        self.correct = QtGui.QToolButton()
        self.correct.setIcon(QtGui.QIcon('img/tick.jpg'))
        iconSize = QtCore.QSize(50, 50)
        self.correct.setIconSize(iconSize)

        self.delete = QtGui.QToolButton()
        self.delete.setIcon(QtGui.QIcon('img/delete.jpg'))
        iconSize = QtCore.QSize(50, 50)
        self.delete.setIconSize(iconSize)

        # self.wrong = QtGui.QToolButton()
        # self.wrong.setIcon(QtGui.QIcon('Resources/cross.png'))

        # An array of radio buttons and a list and a text entry box
        # Create an array of radio buttons for the most common birds (2 columns of 10 choices)
        self.birds1 = []
        for item in self.birdList[:10]:
            self.birds1.append(QRadioButton(item))
        self.birds2 = []
        for item in self.birdList[10:19]:
            self.birds2.append(QRadioButton(item))
        self.birds2.append(QRadioButton('Other'))

        for i in xrange(len(self.birds1)):
            self.birds1[i].setEnabled(True)
            self.connect(self.birds1[i], SIGNAL("clicked()"), self.radioBirdsClicked)
        for i in xrange(len(self.birds2)):
            self.birds2[i].setEnabled(True)
            self.connect(self.birds2[i], SIGNAL("clicked()"), self.radioBirdsClicked)

        # The list of less common birds
        self.birds3 = QListWidget(self)
        self.birds3.setMaximumWidth(150)
        for item in self.birdList[19:]:
            self.birds3.addItem(item)
        #self.birds3.sortItems()
        # Explicitly add "Other" option in
        self.birds3.insertItem(0, 'Other')

        self.connect(self.birds3, SIGNAL("itemClicked(QListWidgetItem*)"), self.listBirdsClicked)
        self.birds3.setEnabled(False)

        # This is the text box for missing birds
        self.tbox = QLineEdit(self)
        self.tbox.setMaximumWidth(150)
        self.connect(self.tbox, SIGNAL('returnPressed()'), self.birdTextEntered)
        #self.connect(self.tbox, SIGNAL('textChanged(QString*)'), self.birdTextEntered)
        self.tbox.setEnabled(False)

        #self.close = QPushButton("Done")
        #self.connect(self.close, SIGNAL("clicked()"), self.accept)

        # The layouts
        birds1Layout = QVBoxLayout()
        for i in xrange(len(self.birds1)):
            birds1Layout.addWidget(self.birds1[i])

        birds2Layout = QVBoxLayout()
        for i in xrange(len(self.birds2)):
            birds2Layout.addWidget(self.birds2[i])

        birdListLayout = QVBoxLayout()
        birdListLayout.addWidget(self.birds3)
        birdListLayout.addWidget(QLabel("If bird isn't in list, select Other"))
        birdListLayout.addWidget(QLabel("Type below, Return at end"))
        birdListLayout.addWidget(self.tbox)

        hboxBirds = QHBoxLayout()
        hboxBirds.addLayout(birds1Layout)
        hboxBirds.addLayout(birds2Layout)
        hboxBirds.addLayout(birdListLayout)

        # The layouts
        vboxButtons = QVBoxLayout()
        vboxButtons.addWidget(self.correct)
        vboxButtons.addWidget(self.delete)
        # hboxButtons.addWidget(self.wrong)
        #hboxButtons.addWidget(self.close)

        self.playButton = QtGui.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(iconSize)

        vboxLabel = QVBoxLayout()
        vboxLabel.addWidget(self.species)
        vboxLabel.addWidget(self.playButton)
        self.connect(self.playButton, SIGNAL('clicked()'), self.playSeg)

        vboxFull = QHBoxLayout()
        vboxFull.addWidget(self.wPlot)
        vboxFull.addLayout(vboxLabel)
        vboxFull.addLayout(hboxBirds)
        vboxFull.addLayout(vboxButtons)
        #vboxFull.addWidget(self.close)

        self.setLayout(vboxFull)
        # print seg
        self.setImage(sg,audiodata,sampleRate,self.label)

    def playSeg(self):  #This is not the right place though
        import wavio
        import platform
        if platform.system() == 'Darwin':
            filename = 'temp.wav'
        else:
            import tempfile
            f = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
            filename = f.name
        self.audiodata = self.audiodata.astype('int16')
        wavio.write(filename,self.audiodata,self.sampleRate,scale='dtype-limits',sampwidth=2)
        import PyQt4.phonon as phonon
        # Create a media object
        media_obj = phonon.Phonon.MediaObject(self)
        audio_output = phonon.Phonon.AudioOutput(phonon.Phonon.MusicCategory, self)
        phonon.Phonon.createPath(media_obj, audio_output)
        media_obj.setTickInterval(20)
        media_obj.setCurrentSource(phonon.Phonon.MediaSource(filename))
        media_obj.seek(0)
        media_obj.play()

    def setImage(self, sg, audiodata, sampleRate, label):

        self.audiodata = audiodata
        self.sampleRate = sampleRate

        sg2 = sg
        if np.shape(sg)[0] < 100:
            sg2 = 255*np.ones((100,np.shape(sg)[1]))
            sg2[:np.shape(sg)[0],:np.shape(sg)[1]] = sg
        if  np.shape(sg)[1]<100:
            sg2 = 255 * np.ones((np.shape(sg)[0], 100))
            sg2[:np.shape(sg)[0],:np.shape(sg)[1]] = sg

        self.plot.setImage(sg2)
        self.plot.setLookupTable(self.lut)

        if self.cmapInverted:
            self.plot.setLevels([self.colourEnd, self.colourStart])
        else:
            self.plot.setLevels([self.colourStart, self.colourEnd])

        # Make one of the options be selected
        self.species.setText(label)
        self.label=label
        if label[-1]=='?':
            label = label[:-1]
        ind = self.birdList.index(label)
        if ind < 10:
            self.birds1[ind].setChecked(True)
        elif ind < 19:
            self.birds2[ind-10].setChecked(True)
        else:
            self.birds2[9].setChecked(True)
            self.birds3.setEnabled(True)
            self.birds3.setCurrentRow(ind-18)

    def radioBirdsClicked(self):
        # Listener for when the user selects a radio button
        # Update the text and store the data
        for button in self.birds1 + self.birds2:
            if button.isChecked():
                if button.text() == "Other":
                    #pass
                    self.birds3.setEnabled(True)
                else:
                    self.birds3.setEnabled(False)
                    self.label = str(button.text())
                    self.species.setText(self.label)

    def listBirdsClicked(self, item):
        # Listener for clicks in the listbox of birds
        if (item.text() == "Other"):
            self.tbox.setEnabled(True)
        else:
            # Save the entry
            self.tbox.setEnabled(False)
            self.label = str(item.text())
            self.species.setText(self.label)

    def birdTextEntered(self):
        # Listener for the text entry in the bird list
        # Check text isn't already in the listbox, and add if not
        # Doesn't sort the list, but will when dialog is reopened
        textitem = self.tbox.text()
        item = self.birds3.findItems(textitem, Qt.MatchExactly)
        if item:
            pass
        else:
            self.birds3.addItem(textitem)
        self.label = str(textitem)
        self.species.setText(self.label)
        self.saveConfig = True

    def getValues(self):
        return [self.label, self.saveConfig, self.tbox.text()]

#======
class HumanClassify2(QDialog):
    # This dialog is different to the others. The aim is to check (or ask for) classifications for segments.
    # This version gets *12* at a time, and put them all out together on buttons, and their labels.
    # It could be all the same species, or the ones that it is unsure about, or whatever.

    # TODO: Work out how big the spect plots are, and make the right number of cols. Also have a min size?
    def __init__(self, sg, segments, label, part, nParts, sampleRate, incr, lut, colourStart, colourEnd, cmapInverted, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Check Classifications')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.frame = QWidget()

        self.sampleRate = sampleRate
        self.incr = incr
        self.lut = lut
        self.colourStart = colourStart
        self.colourEnd = colourEnd
        self.cmapInverted = cmapInverted

        # Seems that image is backwards?
        self.sg = np.fliplr(sg)
        self.segments = segments
        self.firstSegment = 0
        self.errors = []

        #self.indices = []
        #self.segments = []

        #self.segments = [item for item in self.segments if item[4] == label or item[4][:-1] == label]
        #print len(self.segments)
        next = QPushButton("Next/Finish")
        self.connect(next, SIGNAL("clicked()"), self.nextPage)

        if len(self.segments) > 0:

            species = QLabel(label)
            if nParts>1:
                partLabel = QLabel("Part "+str(part+1)+" of " + str(nParts))
            else:
                partLabel = QLabel("")

            # Check that width is at least max seg width, or there is a problem!
            self.width = 0
            for ind in range(len(self.segments)):
                x1 = int(self.convertAmpltoSpec(self.segments[ind][0]))
                x2 = int(self.convertAmpltoSpec(self.segments[ind][1]))
                if x2 - x1 > self.width:
                    self.width = x2-x1
            self.width = max(800,self.width+10)
            #print self.width
            self.h = 4
            self.flowLayout = SupportClasses.FlowLayout()
            self.makeButtons()

            self.vboxFull = QVBoxLayout()
            self.vboxFull.addWidget(QLabel('Click on the images that are incorrectly labelled'))
            self.vboxFull.addWidget(species)
            self.vboxFull.addWidget(partLabel)
            self.vboxFull.addLayout(self.flowLayout)
            self.vboxFull.addWidget(next)
        else:
            self.vboxFull = QVBoxLayout()
            self.vboxFull.addWidget(QLabel('No images to show'))
            self.vboxFull.addWidget(next)

        self.setLayout(self.vboxFull)


    def makeButtons(self):
        segRemain = len(self.segments) - self.firstSegment
        width = 0
        col = 0

        ind = self.firstSegment
        self.buttons = []

        while segRemain>0 and col<self.h:
            x1 = int(self.convertAmpltoSpec(self.segments[ind][0]))
            x2 = int(self.convertAmpltoSpec(self.segments[ind][1]))
            im = self.setImage(self.sg[x1:x2, :])
            segRemain -= 1
            ind += 1
            if width + x2-x1 < self.width:
                width = width + x2-x1
                self.buttons.append(SupportClasses.PicButton(0,im[0], im[1]))
                self.flowLayout.addWidget(self.buttons[-1])
            else:
                col += 1
                width = 0

    def convertAmpltoSpec(self, x):
        return x * self.sampleRate / self.incr

    def setImage(self, seg):
        if self.cmapInverted:
            im, alpha = fn.makeARGB(seg, lut=self.lut, levels=[self.colourEnd, self.colourStart])
        else:
            im, alpha = fn.makeARGB(seg, lut=self.lut, levels=[self.colourStart, self.colourEnd])
        im1 = fn.makeQImage(im, alpha)

        if self.cmapInverted:
            im, alpha = fn.makeARGB(seg, lut=self.lut, levels=[self.colourStart, self.colourEnd])
        else:
            im, alpha = fn.makeARGB(seg, lut=self.lut, levels=[self.colourEnd, self.colourStart])

        im2 = fn.makeQImage(im, alpha)

        return [im1, im2]

    def nextPage(self):
        # Find out which buttons have been clicked (so are not correct)
        for i in range(len(self.buttons)):
            if self.buttons[i].buttonClicked:
                self.errors.append(i+self.firstSegment)
        print self.errors

        # Now find out if there are more segments to check, and remake the buttons, otherwise close
        if len(self.segments) > 0:
            self.firstSegment += len(self.buttons)
            if self.firstSegment != len(self.segments):
                for btn in reversed(self.buttons):
                    self.flowLayout.removeWidget(btn)
                    # remove it from the gui
                    btn.setParent(None)
                self.makeButtons()
            else:
                self.close()
        else:
            self.close()

    def getValues(self):
        return self.errors

# ======
# class HumanClassify3(QDialog):
#     # TODO: Delete when other version works!
#     # This dialog is different to the others. The aim is to check (or ask for) classifications for segments.
#     # This version gets *12* at a time, and put them all out together on buttons, and their labels.
#     # It could be all the same species, or the ones that it is unsure about, or whatever.
#
#     # TODO: Work out how big the spect plots are, and make the right number of cols. Also have a min size?
#
#     def __init__(self, sg, segments, label, sampleRate, incr, lut, colourStart, colourEnd, cmapInverted,
#                  parent=None):
#         QDialog.__init__(self, parent)
#         self.setWindowTitle('Check Classifications')
#         self.setWindowIcon(QIcon('img/Avianz.ico'))
#         self.frame = QWidget()
#
#         self.sampleRate = sampleRate
#         self.incr = incr
#         self.lut = lut
#         self.colourStart = colourStart
#         self.colourEnd = colourEnd
#         self.cmapInverted = cmapInverted
#
#         # Seems that image is backwards?
#         self.sg = np.fliplr(sg)
#         self.segments = segments
#         self.firstSegment = 0
#         self.errors = []
#
#         self.segments = [item for item in self.segments if item[4] == label or item[4][:-1] == label]
#         # print len(self.segments)
#         next = QPushButton("Next/Finish")
#         self.connect(next, SIGNAL("clicked()"), self.next)
#
#         if len(self.segments) > 0:
#
#             species = QLabel(label)
#
#             # TODO: Decide on these sizes
#             self.w = 3
#             self.h = 4
#             self.grid = QGridLayout()
#
#             self.makeButtons()
#
#             vboxFull = QVBoxLayout()
#             vboxFull.addWidget(QLabel('Click on the images that are incorrectly labelled'))
#             vboxFull.addWidget(species)
#             vboxFull.addLayout(self.grid)
#             vboxFull.addWidget(next)
#         else:
#             vboxFull = QVBoxLayout()
#             vboxFull.addWidget(QLabel('No images to show'))
#             vboxFull.addWidget(next)
#
#         self.setLayout(vboxFull)
#
#     def makeButtons(self):
#         positions = [(i, j) for i in range(self.h) for j in range(self.w)]
#         images = []
#         segRemain = len(self.segments) - self.firstSegment
#         # print segRemain, self.firstSegment
#
#         if segRemain < self.w * self.h:
#             for i in range(segRemain):
#                 ind = i + self.firstSegment
#                 x1 = int(self.convertAmpltoSpec(self.segments[ind][0]))
#                 x2 = int(self.convertAmpltoSpec(self.segments[ind][1]))
#                 images.append(self.setImage(self.sg[x1:x2, :]))
#             for i in range(segRemain, self.w * self.h):
#                 images.append([None, None])
#         else:
#             for i in range(self.w * self.h):
#                 ind = i + self.firstSegment
#                 x1 = int(self.convertAmpltoSpec(self.segments[ind][0]))
#                 x2 = int(self.convertAmpltoSpec(self.segments[ind][1]))
#                 images.append(self.setImage(self.sg[x1:x2, :]))
#         self.buttons = []
#         for position, im in zip(positions, images):
#             if im[0] is not None:
#                 self.buttons.append(SupportClasses.PicButton(position[0] * self.w + position[1], im[0], im[1]))
#                 self.grid.addWidget(self.buttons[-1], *position)
#
#     def convertAmpltoSpec(self, x):
#         return x * self.sampleRate / self.incr
#
#     def setImage(self, seg):
#         if self.cmapInverted:
#             im, alpha = fn.makeARGB(seg, lut=self.lut, levels=[self.colourEnd, self.colourStart])
#         else:
#             im, alpha = fn.makeARGB(seg, lut=self.lut, levels=[self.colourStart, self.colourEnd])
#         im1 = fn.makeQImage(im, alpha)
#
#         if self.cmapInverted:
#             im, alpha = fn.makeARGB(seg, lut=self.lut, levels=[self.colourStart, self.colourEnd])
#         else:
#             im, alpha = fn.makeARGB(seg, lut=self.lut, levels=[self.colourEnd, self.colourStart])
#
#         im2 = fn.makeQImage(im, alpha)
#
#         return [im1, im2]
#
#     def next(self):
#         # Find out which buttons have been clicked (so are not correct)
#
#         for i in range(len(self.buttons)):
#             # print self.buttons[i].buttonClicked
#             if self.buttons[i].buttonClicked:
#                 self.errors.append(i + self.firstSegment)
#         # print self.errors
#
#         # Now find out if there are more segments to check, and remake the buttons, otherwise close
#         if len(self.segments) > 0:
#             if (len(self.segments) - self.firstSegment) < self.w * self.h:
#                 self.close()
#             else:
#                 self.firstSegment += self.w * self.h
#                 if self.firstSegment != len(self.segments):
#                     for i in range(self.w * self.h):
#                         self.grid.removeWidget(self.buttons[i])
#                     self.makeButtons()
#                 else:
#                     self.close()
#         else:
#             self.close()
#
#     def getValues(self):
#         return self.errors

# ======
class HumanClassify2a(QDialog):
    def __init__(self, birdlist,parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Check Classification')
        self.setWindowIcon(QIcon('img/Avianz.ico'))

        self.birds = QListWidget(self)
        self.birds.setMaximumWidth(150)
        #self.birds.addItem('All calls')
        #self.birds.addItem('Uncertain calls')
        for item in birdlist:
            self.birds.addItem(item)
        #self.birds.setCurrentRow(0)
        self.connect(self.birds, SIGNAL('itemDoubleClicked(QListWidgetItem*)'), self.dbl)

        ok = QPushButton('OK')
        cancel = QPushButton('Cancel')
        self.connect(ok, SIGNAL('clicked()'), self.ok)
        self.connect(cancel,SIGNAL('clicked()'), self.cancel)

        layout = QVBoxLayout()
        layout.addWidget(QLabel('Choose the bird you wish to see classification of:'))
        layout.addWidget(self.birds)
        layout.addWidget(ok)
        layout.addWidget(cancel)

        # Now put everything into the frame
        self.setLayout(layout)

    def dbl(self,item):
        self.birds.setCurrentItem(item)
        self.accept()

    def ok(self):
        #self.chosen = self.birds.selectedItems()
        self.accept()

    def cancel(self):
        #self.chosen = None
        self.reject()

    def getValues(self):
        return self.birds.currentItem().text()

# ======
class InterfaceSettings2(QDialog):
    def __init__(self, parent = None):
      super(InterfaceSettings2, self).__init__(parent)

      self.tabWidget = QTabWidget()
      self.tabWidget.tab1 = QWidget()
      self.tabWidget.tab2 = QWidget()
      self.tabWidget.tab3 = QWidget()

      self.tabWidget.addTab(self.tabWidget.tab1,"Tab 1")
      self.tabWidget.addTab(self.tabWidget.tab2,"Tab 2")
      self.tabWidget.addTab(self.tabWidget.tab3,"Tab 3")
      self.tab1UI()
      self.tab2UI()
      self.tab3UI()
      self.setWindowTitle("Interface Settings")
      self.setWindowIcon(QIcon('img/Avianz.ico'))
      self.setMinimumWidth(400)

      mainLayout = QVBoxLayout()
      mainLayout.addWidget(self.tabWidget)
      self.setLayout(mainLayout)

    def tab1UI(self):
      layout = QFormLayout()
      colorNamed=pg.ColorButton()
      colorNamed.setColor(color='FF0000')
      colorPossible = pg.ColorButton()
      colorPossible.setColor(color='FFFF00')
      colorNone=pg.ColorButton()
      colorNone.setColor(color='0000FF')
      colorSelected = pg.ColorButton()
      colorSelected.setColor(color='00FF00')
      layout.addRow(QLabel("Confirmed"),colorNamed)
      layout.addRow(QLabel("Possible"), colorPossible)
      layout.addRow(QLabel("Don't Know"),colorNone)
      layout.addRow(QLabel("Currently Selected"), colorSelected)
      self.tabWidget.setTabText(0, "Segment Colours")
      self.tabWidget.tab1.setLayout(layout)

    def tab2UI(self):
      layout = QFormLayout()
      sex = QHBoxLayout()
      sex.addWidget(QRadioButton("Male"))
      sex.addWidget(QRadioButton("Female"))
      layout.addRow(QLabel("Sex"), sex)
      layout.addRow("Date of Birth", QLineEdit())
      self.tabWidget.setTabText(1, "Bird Names")
      self.tabWidget.tab2.setLayout(layout)

    def tab3UI(self):
      layout = QHBoxLayout()
      layout.addWidget(QLabel("subjects"))
      layout.addWidget(QCheckBox("Physics"))
      layout.addWidget(QCheckBox("Maths"))
      self.tabWidget.setTabText(2, "Spectrogram Settings")
      self.tabWidget.tab3.setLayout(layout)

