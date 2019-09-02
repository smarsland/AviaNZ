
#
# This is part of the AviaNZ interface
# Holds most of the code for the various dialog boxes
# Version 1.3 23/10/18
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis

#    AviaNZ birdsong analysis program
#    Copyright (C) 2017--2018

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

# Dialogs used by the AviaNZ program
# Since most of them just get user selections, they are mostly just a mess of UI things
import sys,os

from PyQt5.QtGui import *
from PyQt5.QtWidgets import QDialog, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QSlider, QCheckBox, QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QDir, QPointF, QTime, Qt, QLineF

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.functions as fn
import numpy as np
import SupportClasses as SupportClasses
import SignalProc
import Segment


class StartScreen(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle('AviaNZ - Choose Task')
        self.setAutoFillBackground(False)
        self.setFixedSize(900, 350)
        self.setStyleSheet("background-image: url(img/AviaNZ_SW_V2.jpg);")
        self.activateWindow()

        btn_style='QPushButton {background-color: #A3C1DA; color: white; font-size:20px; font-weight: bold; font-family: "Arial"}'
        # btn_style2='QPushButton {background-color: #A3C1DA; color: grey; font-size:16px}'
        b1 = QPushButton(" Manual Segmentation ")
        b2 = QPushButton("      Batch Processing      ")
        b3 = QPushButton(" Review Batch Results ")
        l1 = QLabel("-------")
        l2 = QLabel("---")
        b1.setStyleSheet(btn_style)
        b2.setStyleSheet(btn_style)
        b3.setStyleSheet(btn_style)
        l1.setStyleSheet('QLabel {color:transparent}')

        hbox = QHBoxLayout()
        hbox.addWidget(l1)
        hbox.addWidget(b1)
        hbox.addWidget(l1)
        hbox.addWidget(b2)
        hbox.addWidget(l1)
        hbox.addWidget(b3)
        hbox.addWidget(l2)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addWidget(l1)

        self.setLayout(vbox)

        b1.clicked.connect(self.manualSeg)
        b2.clicked.connect(self.findSpecies)
        b3.clicked.connect(self.reviewSeg)

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

    def reviewSeg(self):
        self.task = 4
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
    def __init__(self, width, incr, minFreq, maxFreq, minFreqShow, maxFreqShow, DOC=True, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Spectrogram Options')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setMinimumWidth(300)
        self.DOC = DOC

        self.windowType = QComboBox()
        self.windowType.addItems(['Hann','Parzen','Welch','Hamming','Blackman','BlackmanHarris'])

        self.mean_normalise = QCheckBox()
        self.mean_normalise.setChecked(True)

        self.equal_loudness = QCheckBox()
        self.equal_loudness.setChecked(False)

        self.multitaper = QCheckBox()
        self.multitaper.setChecked(False)

        self.low = QSlider(Qt.Horizontal)
        self.low.setTickPosition(QSlider.TicksBelow)
        self.low.setTickInterval(1000)
        self.low.setRange(minFreq,maxFreq)
        self.low.setSingleStep(100)
        self.low.setValue(minFreqShow)
        self.low.valueChanged.connect(self.lowChange)
        self.lowtext = QLabel(str(self.low.value()))

        self.high = QSlider(Qt.Horizontal)
        self.high.setTickPosition(QSlider.TicksBelow)
        self.high.setTickInterval(1000)
        self.high.setRange(minFreq,maxFreq)
        self.high.setSingleStep(100)
        self.high.setValue(maxFreqShow)
        self.high.valueChanged.connect(self.highChange)
        self.hightext = QLabel(str(self.high.value()))

        self.activate = QPushButton("Update Spectrogram")

        self.window_width = QLineEdit(self)
        self.window_width.setText(str(width))
        self.incr = QLineEdit(self)
        self.incr.setText(str(incr))

        Box = QVBoxLayout()
        Box.addWidget(self.windowType)
        Box.addWidget(QLabel('Mean normalise'))
        Box.addWidget(self.mean_normalise)
        Box.addWidget(QLabel('Equal loudness'))
        Box.addWidget(self.equal_loudness)
        if not self.DOC:
            Box.addWidget(QLabel('Multitapering'))
            Box.addWidget(self.multitaper)
        Box.addWidget(QLabel('Window Width'))
        Box.addWidget(self.window_width)
        Box.addWidget(QLabel('Hop'))
        Box.addWidget(self.incr)

        Box.addWidget(QLabel('Frequency range to show'))
        Box.addWidget(QLabel('Lowest frequency'))
        Box.addWidget(self.lowtext)
        Box.addWidget(self.low)
        Box.addWidget(QLabel('Highest frequency'))
        Box.addWidget(self.high)
        Box.addWidget(self.hightext)

        Box.addWidget(self.activate)

        # Now put everything into the frame
        self.setLayout(Box)

    def setValues(self,minFreq,maxFreq,minFreqShow,maxFreqShow):
        self.low.setRange(minFreq,maxFreq)
        self.low.setValue(minFreqShow)
        self.high.setRange(minFreq,maxFreq)
        self.high.setValue(maxFreqShow)

    def getValues(self):
        return [self.windowType.currentText(),self.mean_normalise.checkState(),self.equal_loudness.checkState(),self.multitaper.checkState(),self.window_width.text(),self.incr.text(),self.low.value(),self.high.value()]

    def lowChange(self,value):
        self.lowtext.setText(str(value))

    def highChange(self,value):
        self.hightext.setText(str(value))

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
    def __init__(self, operator='', reviewer='', parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Set Operator/Reviewer')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setMinimumWidth(320)

        self.operatorlabel = QLabel("Operator")
        self.name1 = QLineEdit(self)
        self.name1.setText(operator)
        self.reviewerlabel = QLabel("Reviewer")
        self.name2 = QLineEdit(self)
        self.name2.setText(reviewer)
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
        #print(self.name1.text(),self.name2.text())
        return [self.name1.text(),self.name2.text()]

#======
class addNoiseData(QDialog):
    # Class for the noise data dialog box
    # TODO: Options are hard-coded for now. Does it matter?
    def __init__(self, noiseLevel, noiseTypes, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Noise Information')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setMinimumWidth(320)

        print(noiseLevel,noiseTypes)
        HBox1 = QVBoxLayout()
        levelLabel = QLabel("Level of Noise")
        HBox1.addWidget(levelLabel)
        self.level = QButtonGroup()
        self.btnLow = QRadioButton('Low')
        self.level.addButton(self.btnLow)
        if noiseLevel == 'Low':
            self.btnLow.setChecked(True)
        HBox1.addWidget(self.btnLow)
        self.btnMed = QRadioButton('Medium')
        self.level.addButton(self.btnMed)
        if noiseLevel == 'Medium':
            self.btnLow.setChecked(True)
        HBox1.addWidget(self.btnMed)
        self.btnHigh = QRadioButton('High')
        self.level.addButton(self.btnHigh)
        if noiseLevel == 'High':
            self.btnLow.setChecked(True)
        HBox1.addWidget(self.btnHigh)
        self.btnTerrible = QRadioButton('Terrible')
        self.level.addButton(self.btnTerrible)
        HBox1.addWidget(self.btnTerrible)
        if noiseLevel == 'Terrible':
            self.btnLow.setChecked(True)

        self.activate = QPushButton("Set")
        HBox1.addWidget(self.activate)

        HBox2 = QVBoxLayout()
        typesLabel = QLabel("Types of Noise")
        HBox2.addWidget(typesLabel)
        self.types = QButtonGroup()
        self.types.setExclusive(False)
        self.btns = []

        self.btns.append(QCheckBox('Rain'))
        self.types.addButton(self.btns[0])
        HBox2.addWidget(self.btns[0])
        if 'Rain' in noiseTypes:
            self.btns[0].setChecked(True)

        self.btns.append(QCheckBox('Wind'))
        self.types.addButton(self.btns[1])
        HBox2.addWidget(self.btns[1])
        if 'Wind' in noiseTypes:
            self.btns[1].setChecked(True)

        self.btns.append(QCheckBox('Wind Gusts'))
        self.types.addButton(self.btns[2])
        HBox2.addWidget(self.btns[2])
        if 'Wind Gusts' in noiseTypes:
            self.btns[2].setChecked(True)

        self.btns.append(QCheckBox('Waves/water'))
        self.types.addButton(self.btns[3])
        HBox2.addWidget(self.btns[3])
        if 'Waves/water' in noiseTypes:
            self.btns[3].setChecked(True)

        self.btns.append(QCheckBox('Insects'))
        self.types.addButton(self.btns[4])
        HBox2.addWidget(self.btns[4])
        if 'Insects' in noiseTypes:
            self.btns[4].setChecked(True)

        self.btns.append(QCheckBox('People'))
        self.types.addButton(self.btns[5])
        HBox2.addWidget(self.btns[5])
        if 'People' in noiseTypes:
            self.btns[5].setChecked(True)

        self.btns.append(QCheckBox('Other'))
        self.types.addButton(self.btns[6])
        HBox2.addWidget(self.btns[6])
        if 'Other' in noiseTypes:
            self.btns[6].setChecked(True)

        Box = QHBoxLayout()
        Box.setAlignment(Qt.AlignTop)
        Box.addLayout(HBox1)
        Box.addLayout(HBox2)

        # Now put everything into the frame
        self.setLayout(Box)

    def getNoiseData(self):
        #print(self.name1.text(),self.name2.text())
        if self.level.checkedButton() is None:
            self.btnLow.setChecked(True)
        types = []
        for btn in self.btns:
            if btn.isChecked():
                types.append(btn.text())

        return [self.level.checkedButton().text(),types]

#======
class Diagnostic(QDialog):
    # Class for the diagnostic dialog box
    def __init__(self, filters, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Diagnostic Plot Options')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setMinimumWidth(300)

        # species / filter
        self.filterLabel = QLabel("Select filter to use")
        self.filter = QComboBox()
        # add filter file names to combobox
        self.filter.addItems(list(filters.keys()))

        # antialiasing
        self.aaLabel = QLabel("Select antialiasing type:")
        self.aaGroup = QButtonGroup()
        aaButtons = [QRadioButton("No AA"), QRadioButton("Fast partial AA"), QRadioButton("Full AA")]
        aaButtons[0].setChecked(True)
        for a in aaButtons:
            self.aaGroup.addButton(a)

        # spec or energy plot
        # self.plotLabel = QLabel("Select plot type:")
        # self.plotGroup = QButtonGroup()
        # plotButtons = [QRadioButton("Filter band energy"), QRadioButton("Reconstructed spectrogram")]
        # plotButtons[0].setChecked(True)
        # for a in plotButtons:
        #     self.plotGroup.addButton(a)

        # mark calls on spectrogram?
        self.mark = QCheckBox("Mark calls on spectrogram")
        self.mark.setChecked(True)

        # buttons
        self.activate = QPushButton("Make plots")
        self.clear = QPushButton("Clear plots")

        # layout
        Box = QVBoxLayout()
        Box.addWidget(self.filterLabel)
        Box.addWidget(self.filter)

        Box.addWidget(self.aaLabel)
        for a in aaButtons:
            Box.addWidget(a)

        # Box.addWidget(self.plotLabel)
        # for a in plotButtons:
        #     Box.addWidget(a)

        Box.addWidget(self.mark)

        Box.addWidget(self.activate)
        Box.addWidget(self.clear)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        return [self.filter.currentText(), self.aaGroup.checkedId(), self.mark.isChecked()]

#======
class WaveletTrain(QDialog):
    # Class for the segmentation dialog box
    # TODO: add the wavelet params
    # TODO: work out how to return varying size of params, also process them
    # TODO: test and play
    def __init__(self, maxv, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Train a Species Detector')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setMinimumWidth(475)
        self.setMinimumHeight(500)

        # Step1 - prepare for training
        self.layout_step1 = QVBoxLayout()
        self.form1_step1 = QFormLayout()
        self.form2_step1 = QFormLayout()
        self.hBox_step1 = QHBoxLayout()
        self.blank = QLabel('           ')
        self.Step1Label1 = QLabel('Step 1: Prepare')
        self.Step1Label1.setFont(QtGui.QFont('TimesNewRoman', 12))
        self.browse = QPushButton('Browse')
        self.form1_step1.addRow('Load data', self.browse)
        self.listFiles = QListWidget()
        self.listFiles.setMinimumWidth(150)
        self.listFiles.setMinimumHeight(275)
        self.form1_step1.addRow(' ', self.listFiles)
        self.species = QComboBox()  # fill during browse
        self.species.addItems(['Choose species...'])
        self.form2_step1.addRow('Species   ', self.species)
        self.genGT = QPushButton('Prepare for Training')
        self.genGT.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')
        self.genGT.setEnabled(False)
        self.hBox_step1.addStretch(1)
        self.hBox_step1.addWidget(self.genGT)
        # Step1 layout
        self.layout_step1.addWidget( self.Step1Label1)
        self.layout_step1.addWidget(self.blank)
        self.layout_step1.addLayout(self.form1_step1)
        self.layout_step1.addWidget(self.blank)
        self.layout_step1.addLayout(self.form2_step1)
        self.layout_step1.addStretch(1)
        self.layout_step1.addLayout(self.hBox_step1)

        # Step2 - actual training
        self.layout_step2 = QVBoxLayout()
        self.form1_step2 = QFormLayout()
        self.hBox_step2 = QHBoxLayout()
        self.Step1Label2 = QLabel('Step 2: Train')
        self.Step1Label2.setFont(QtGui.QFont('TimesNewRoman', 12))
        self.minlen = QLineEdit(self)
        self.minlen.setText('')
        self.minlen.setEnabled(False)
        self.form1_step2.addRow('Min call length (secs)', self.minlen)
        self.maxlen = QLineEdit(self)
        self.maxlen.setText('')
        self.maxlen.setEnabled(False)
        self.form1_step2.addRow('Max call length (secs)', self.maxlen)
        self.fLow = QSlider(Qt.Horizontal)
        self.fLow.setTickPosition(QSlider.TicksBelow)
        self.fLow.setTickInterval(2000)
        self.fLow.setRange(0, 32000)
        self.fLow.setSingleStep(100)
        self.fLow.setEnabled(False)
        self.fLow.valueChanged.connect(self.fLowChange)
        self.fLowtext = QLabel('')
        self.form1_step2.addRow('', self.fLowtext)
        self.form1_step2.addRow('Lower frq. (Hz)', self.fLow)
        self.fHigh = QSlider(Qt.Horizontal)
        self.fHigh.setTickPosition(QSlider.TicksBelow)
        self.fHigh.setTickInterval(2000)
        self.fHigh.setRange(0, 32000)
        self.fHigh.setSingleStep(100)
        self.fHigh.setEnabled(False)
        self.fHigh.valueChanged.connect(self.fHighChange)
        self.fHightext = QLabel('')
        self.form1_step2.addRow('', self.fHightext)
        self.form1_step2.addRow('Higher frq. (Hz)', self.fHigh)
        self.fs = QSlider(Qt.Horizontal)
        self.fs.setTickPosition(QSlider.TicksBelow)
        self.fs.setTickInterval(2000)
        self.fs.setRange(0, 32000)
        self.fs.setSingleStep(2000)
        self.fs.valueChanged.connect(self.fsChange)
        self.fs.setEnabled(False)
        self.fstext = QLabel('')
        self.form1_step2.addRow('', self.fstext)
        self.form1_step2.addRow('Preferred sampling frq. (Hz)', self.fs)
        self.hBox2_step2 = QHBoxLayout()
        self.wind = QCheckBox()
        self.wind_label = QLabel('Wind')
        self.wind.setChecked(False)
        self.rain = QCheckBox()
        self.rain_label = QLabel('Rain')
        self.rain.setChecked(False)
        self.ff = QCheckBox()
        self.ff_label = QLabel('Fundamental frequency     ')
        self.ff.setChecked(False)
        self.wind.setEnabled(False)
        self.rain.setEnabled(False)
        self.ff.setEnabled(False)
        self.hBox2_step2.addWidget(self.wind_label)
        self.hBox2_step2.addWidget(self.wind)
        self.hBox2_step2.addWidget(self.rain_label)
        self.hBox2_step2.addWidget(self.rain)
        self.hBox2_step2.addWidget(self.ff_label)
        self.hBox2_step2.addWidget(self.ff)

        # thr and M spinboxes
        self.hBox3_step2 = QHBoxLayout()
        self.setthr = QSpinBox()
        self.setthr_label = QLabel("thr iterations")
        self.setM = QSpinBox()
        self.setM_label = QLabel("M iterations")
        self.setthr.setMinimum(3)
        self.setM.setMinimum(1)
        self.setM.setValue(3)
        self.setthr.setMaximum(20)
        self.setM.setMaximum(10)
        self.setM.setEnabled(False)
        self.setthr.setEnabled(False)
        self.hBox3_step2.addWidget(self.setthr_label)
        self.hBox3_step2.addWidget(self.setthr)
        self.hBox3_step2.addWidget(self.setM_label)
        self.hBox3_step2.addWidget(self.setM)

        self.train = QPushButton('Train')
        self.train.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')
        self.train.setEnabled(False)
        self.hBox_step2.addStretch(1)
        self.hBox_step2.addWidget(self.train)
        # Step2 layout
        self.layout_step2.addWidget(self.Step1Label2)
        self.layout_step2.addWidget(self.blank)
        self.layout_step2.addLayout(self.form1_step2)
        self.note_step2 = QLabel(' ')
        self.note_step2.setFont(QtGui.QFont('TimesNewRoman', 9))
        self.layout_step2.addWidget(self.blank)
        self.layout_step2.addWidget(self.note_step2)
        self.layout_step2.addWidget(self.blank)
        self.layout_step2.addLayout(self.hBox2_step2)
        self.layout_step2.addLayout(self.hBox3_step2)
        self.layout_step2.addStretch(1)
        self.layout_step2.addLayout(self.hBox_step2)

        # Step3 - test
        self.layout_step3 = QVBoxLayout()
        self.form1_step3 = QFormLayout()
        self.hBox_step3 = QHBoxLayout()
        self.Step1Label3 = QLabel('Step 3: Test')
        self.Step1Label3.setFont(QtGui.QFont('TimesNewRoman', 12))
        self.browseTest = QPushButton('Browse')
        self.browseTest.setEnabled(False)
        self.form1_step3.addRow('Load data', self.browseTest)
        self.listFilesTest = QListWidget()
        self.listFilesTest.setMinimumWidth(150)
        self.listFilesTest.setMinimumHeight(225)
        self.form1_step3.addRow(' ', self.listFilesTest)
        self.test = QPushButton('Test')
        self.test.setStyleSheet('QPushButton {background-color: #A3C1DA; font-weight: bold; font-size:14px}')
        self.test.setEnabled(False)
        self.hBox_step3.addStretch(1)
        self.hBox_step3.addWidget(self.test)
        # Step3 layout
        self.layout_step3.addWidget(self.Step1Label3)
        self.layout_step3.addWidget(self.blank)
        self.layout_step3.addLayout(self.form1_step3)
        self.layout_step3.addWidget(self.blank)
        self.note_step3 = QLabel(' ')
        self.note_step3.setFont(QtGui.QFont('TimesNewRoman', 9))
        self.layout_step3.addWidget(self.note_step3)
        self.layout_step3.addStretch(1)
        self.layout_step3.addLayout(self.hBox_step3)

        # Put together
        self.layout = QHBoxLayout()
        self.layout.addLayout(self.layout_step1)
        self.layout.addWidget(self.blank)
        self.layout.addLayout(self.layout_step2)
        self.layout.addWidget(self.blank)
        self.layout.addLayout(self.layout_step3)
        self.setLayout(self.layout)

        self.rain_label.setHidden(True)
        self.rain.setHidden(True)

    def fLowChange(self, value):
        value = value - (value % 10)
        if value < 50:
            value = 50
        self.fLowtext.setText(str(value))

    def fHighChange(self, value):
        value = value - (value % 10)
        if value < 100:
            value = 100
        self.fHightext.setText(str(value))

    def fsChange(self, value):
        value = value - (value % 1000)
        if value < 1000:
            value = 1000
        self.fstext.setText(str(value))


    def fillFileList(self, dirName, train=True):
        """ Generates the list of files for the file listbox.
        fileName - currently opened file (marks it in the list).
        Most of the work is to deal with directories in that list.
        It only sees *.wav files. Picks up *.data and *_1.wav files, the first to make the filenames
        red in the list, and the second to know if the files are long."""

        if not os.path.isdir(dirName):
            print("ERROR: Directory doesn't exist")
            return

        if train:
            self.listFiles.clear()
            spList = set()
            # collect possible species from annotations:
            for root, dirs, files in os.walk(dirName):
                for filename in files:
                    if filename.endswith('.wav') and filename+'.data' in files:
                        # this wav has data, so see what species are in there
                        segments = Segment.SegmentList()
                        segments.parseJSON(os.path.join(root, filename+'.data'))
                        spList.update([lab["species"] for seg in segments for lab in seg[4]])
            spList = list(spList)
            spList.insert(0, 'Choose species...')
            self.species.clear()
            self.species.addItems(spList)
        else:
            self.listFilesTest.clear()

        listOfFiles = QDir(dirName).entryInfoList(['*.wav'],filters=QDir.AllDirs|QDir.NoDotAndDotDot|QDir.Files,sort=QDir.DirsFirst)
        listOfDataFiles = QDir(dirName).entryList(['*.wav.data'])
        for file in listOfFiles:
            # Add the filename to the right list
            if train:
                item = QListWidgetItem(self.listFiles)
            else:
                item = QListWidgetItem(self.listFilesTest)
            # count wavs in directories:
            if file.isDir():
                numwavs = 0
                for root, dirs, files in os.walk(file.filePath()):
                    numwavs += sum(f.endswith('.wav') for f in files)
                item.setText("%s/\t\t(%d wav files)" % (file.fileName(), numwavs))
            else:
                item.setText(file.fileName())
            # If there is a .data version, colour the name red to show it has been labelled
            if file.fileName()+'.data' in listOfDataFiles:
                item.setForeground(Qt.red)

#======
class Segmentation(QDialog):
    # Class for the segmentation dialog box
    # TODO: add the wavelet params
    # TODO: work out how to return varying size of params, also process them
    # TODO: test and play
    def __init__(self, maxv, DOC=False, species=None, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Segmentation Options')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setMinimumWidth(350)

        self.algs = QComboBox()
        if DOC:
            self.algs.addItems(["Wavelets", "FIR", "Median Clipping"])
        else:
            self.algs.addItems(["Default","Median Clipping","Fundamental Frequency","FIR","Wavelets","Harma","Power","Cross-Correlation"])
        self.algs.currentIndexChanged[str].connect(self.changeBoxes)
        self.undo = QPushButton("Undo")
        self.resLabel = QLabel("Time Resolution in Excel Output (secs)")
        self.res = QSpinBox()
        self.res.setRange(1, 600)
        self.res.setSingleStep(5)
        self.res.setValue(60)
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

        # set min seg size for median clipping
        self.medSize = QSlider(Qt.Horizontal)
        self.medSize.setTickPosition(QSlider.TicksBelow)
        self.medSize.setTickInterval(100)
        self.medSize.setRange(100,2000)
        self.medSize.setSingleStep(100)
        self.medSize.setValue(1000)
        self.medSize.valueChanged.connect(self.medSizeChange)
        self.medSizeText = QLabel("Minimum length: 1000 ms")

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

        self.Harmalabel = QLabel("Set decibel threshold")
        Box.addWidget(self.Harmalabel)

        #self.Onsetslabel = QLabel("Onsets: No parameters")
        #Box.addWidget(self.Onsetslabel)

        self.medlabel = QLabel("Set median threshold")
        Box.addWidget(self.medlabel)
        self.medlabel.show()

        self.eclabel = QLabel("Set energy curve threshold")
        Box.addWidget(self.eclabel)
        self.ecthrtype = [QRadioButton("N standard deviations"), QRadioButton("Threshold")]

        self.specieslabel = QLabel("Species")
        self.species=QComboBox()

        # extra hiding step - less widgets to draw on initial load
        for w in range(Box.count()):
            Box.itemAt(w).widget().hide()

        self.specieslabel_cc = QLabel("Species")
        self.species_cc = QComboBox()
        self.species_cc.addItems(["Choose species...", "Bittern"])

        spp = list(species.keys())
        spp.insert(0,"Choose species...")
        self.species.addItems(spp)

        Box.addWidget(self.specieslabel)
        Box.addWidget(self.species)

        Box.addWidget(self.specieslabel_cc)
        Box.addWidget(self.species_cc)

        Box.addWidget(self.HarmaThr1)
        Box.addWidget(self.HarmaThr2)
        Box.addWidget(self.PowerThr)

        Box.addWidget(self.medThr)
        Box.addWidget(self.medSizeText)
        Box.addWidget(self.medSize)

        for i in range(len(self.ecthrtype)):
            Box.addWidget(self.ecthrtype[i])
        Box.addWidget(self.ecThr)

        Box.addWidget(self.FIRThr1)

        Box.addWidget(self.Fundminfreqlabel)
        Box.addWidget(self.Fundminfreq)
        Box.addWidget(self.Fundminperiodslabel)
        Box.addWidget(self.Fundminperiods)
        Box.addWidget(self.Fundthrlabel)
        Box.addWidget(self.Fundthr)
        Box.addWidget(self.Fundwindowlabel)
        Box.addWidget(self.Fundwindow)

        Box.addWidget(self.CCThr1)

        Box.addWidget(self.resLabel)
        Box.addWidget(self.res)
        Box.addWidget(self.undo)
        self.undo.setEnabled(False)
        Box.addWidget(self.activate)
        #Box.addWidget(self.save)

        # Now put everything into the frame,
        # hide and reopen the default
        for w in range(Box.count()):
            Box.itemAt(w).widget().hide()
        self.setLayout(Box)
        self.algs.show()
        self.undo.show()
        self.res.show()
        self.resLabel.show()
        self.activate.show()
        if DOC:
            self.changeBoxes("Wavelets")
        else:
            self.changeBoxes("Default")

    def changeBoxes(self,alg):
        # This does the hiding and showing of the options as the algorithm changes
        # hide and reopen the default
        for w in range(self.layout().count()):
            self.layout().itemAt(w).widget().hide()
        self.algs.show()
        self.undo.show()
        self.res.show()
        self.resLabel.show()
        self.activate.show()

        if alg == "Default":
            pass
        elif alg == "Energy Curve":
            self.eclabel.show()
            self.ecThr.show()
            for i in range(len(self.ecthrtype)):
                self.ecthrtype[i].show()
            self.ecThr.show()
        elif alg == "Harma":
            self.Harmalabel.show()
            self.HarmaThr1.show()
            self.HarmaThr2.show()
        elif alg == "Power":
            self.PowerThr.show()
        elif alg == "Median Clipping":
            self.medlabel.show()
            self.medThr.show()
            self.medSize.show()
            self.medSizeText.show()
        elif alg == "Fundamental Frequency":
            self.Fundminfreq.show()
            self.Fundminperiods.show()
            self.Fundthr.show()
            self.Fundwindow.show()
            self.Fundminfreqlabel.show()
            self.Fundminperiodslabel.show()
            self.Fundthrlabel.show()
            self.Fundwindowlabel.show()
        #elif alg == "Onsets":
        #    self.Onsetslabel.show()
        elif alg == "FIR":
            self.FIRThr1.show()
        #elif alg == "Best":
        #    pass
        elif alg == "Cross-Correlation":
            self.CCThr1.show()
            self.specieslabel_cc.show()
            self.species_cc.show()
        else:
            #"Wavelets"
            self.specieslabel.show()
            self.species.show()

    def medSizeChange(self,value):
        self.medSizeText.setText("Minimum length: %s ms" % value)

    def getValues(self):
        return [self.algs.currentText(), self.medThr.text(), self.medSize.value(), self.HarmaThr1.text(),self.HarmaThr2.text(),self.PowerThr.text(),self.Fundminfreq.text(),self.Fundminperiods.text(),self.Fundthr.text(),self.Fundwindow.text(),self.FIRThr1.text(),self.CCThr1.text(),self.species.currentText(), self.res.value(), self.species_cc.currentText()]

#======
class Denoise(QDialog):
    # Class for the denoising dialog box
    def __init__(self, parent=None,DOC=True,minFreq=0,maxFreq=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Denoising Options')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setMinimumWidth(300)
        self.setMinimumHeight(250)
        self.DOC=DOC
        self.minFreq=minFreq
        self.maxFreq=maxFreq

        self.algs = QComboBox()
        # self.algs.addItems(["Wavelets","Bandpass","Butterworth Bandpass" ,"Wavelets --> Bandpass","Bandpass --> Wavelets","Median Filter"])
        if not self.DOC:
            self.algs.addItems(["Wavelets", "Bandpass", "Butterworth Bandpass", "Median Filter"])
        else:
            self.algs.addItems(["Wavelets", "Bandpass", "Butterworth Bandpass"])
        self.algs.currentIndexChanged[str].connect(self.changeBoxes)
        self.prevAlg = "Wavelets"

        # Wavelet: Depth of tree, threshold type, threshold multiplier, wavelet
        # self.wavlabel = QLabel("Wavelets")
        if not self.DOC:
            self.depthlabel = QLabel("Depth of wavelet packet decomposition (or tick box to use best)")
            self.depthchoice = QCheckBox()
            #self.connect(self.depthchoice, SIGNAL('clicked()'), self.depthclicked)
            self.depthchoice.clicked.connect(self.depthclicked)
            self.depth = QSpinBox()
            self.depth.setRange(1,12)
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

            self.aabox1 = QCheckBox("Antialias reconstruction")
            self.aabox2 = QCheckBox("Antialias WP build")

            self.waveletlabel = QLabel("Type of wavelet")
            self.wavelet = QComboBox()
            self.wavelet.addItems(["dmey2","db2","db5","db10","haar","coif5","coif15", "sym2","sym8","sym18"])
            self.wavelet.setCurrentIndex(0)

        # Median: width of filter
        self.medlabel = QLabel("Median Filter")
        self.widthlabel = QLabel("Half width of median filter")
        self.width = QSpinBox()
        self.width.setRange(1,101)
        self.width.setSingleStep(1)
        self.width.setValue(11)

        # Bandpass: high and low
        self.bandlabel = QLabel("Bandpass Filter")
        self.wblabel = QLabel("Wavelets and Bandpass Filter")
        self.blabel = QLabel("Start and end points of the band")
        self.low = QSlider(Qt.Horizontal)
        self.low.setTickPosition(QSlider.TicksBelow)
        self.low.setTickInterval(1000)
        self.low.setRange(minFreq,maxFreq)
        self.low.setSingleStep(100)
        self.low.setValue(minFreq)
        self.low.valueChanged.connect(self.lowChange)
        self.lowtext = QLabel(str(self.low.value()))
        self.high = QSlider(Qt.Horizontal)
        self.high.setTickPosition(QSlider.TicksBelow)
        self.high.setTickInterval(1000)
        self.high.setRange(minFreq,maxFreq)
        self.high.setSingleStep(100)
        self.high.setValue(maxFreq)
        self.high.valueChanged.connect(self.highChange)
        self.hightext = QLabel(str(self.high.value()))

        #self.trimlabel = QLabel("Make frequency axis tight")
        #self.trimaxis = QCheckBox()
        #self.trimaxis.setChecked(False)

        # Want combinations of these too!

        self.activate = QPushButton("Denoise")
        self.undo = QPushButton("Undo")
        self.save = QPushButton("Save Denoised Sound")
        #self.connect(self.undo, SIGNAL('clicked()'), self.undo)
        Box = QVBoxLayout()
        Box.addWidget(self.algs)

        if not self.DOC:
            Box.addWidget(self.depthlabel)
            Box.addWidget(self.depthchoice)
            Box.addWidget(self.depth)

            Box.addWidget(self.thrtypelabel)
            Box.addWidget(self.thrtype[0])
            Box.addWidget(self.thrtype[1])

            Box.addWidget(self.thrlabel)
            Box.addWidget(self.thr)

            Box.addWidget(self.aabox1)
            Box.addWidget(self.aabox2)

            Box.addWidget(self.waveletlabel)
            Box.addWidget(self.wavelet)

            # Median: width of filter
            Box.addWidget(self.medlabel)
            self.medlabel.hide()
            Box.addWidget(self.widthlabel)
            self.widthlabel.hide()
            Box.addWidget(self.width)
            self.width.hide()

        # Bandpass: high and low
        Box.addWidget(self.bandlabel)
        self.bandlabel.hide()
        Box.addWidget(self.wblabel)
        self.wblabel.hide()
        Box.addWidget(self.blabel)
        self.blabel.hide()
        Box.addWidget(self.lowtext)
        self.lowtext.hide()
        Box.addWidget(self.low)
        self.low.hide()
        Box.addWidget(self.high)
        self.high.hide()
        Box.addWidget(self.hightext)
        self.hightext.hide()

        #Box.addWidget(self.trimlabel)
        #self.trimlabel.hide()
        #Box.addWidget(self.trimaxis)
        #self.trimaxis.hide()

        Box.addWidget(self.activate)
        Box.addWidget(self.undo)
        Box.addWidget(self.save)

        # Now put everything into the frame
        self.setLayout(Box)

    def setValues(self,minFreq,maxFreq):
        self.minFreq = minFreq
        self.maxFreq = maxFreq
        self.low.setMinimum(self.minFreq)
        self.low.setMaximum(self.maxFreq)
        self.high.setMinimum(self.minFreq)
        self.high.setMaximum(self.maxFreq)

    def changeBoxes(self,alg):
        # This does the hiding and showing of the options as the algorithm changes
        if self.prevAlg == "Wavelets" and not self.DOC:
            # self.wavlabel.hide()
            self.depthlabel.hide()
            self.depth.hide()
            self.depthchoice.hide()
            self.thrtypelabel.hide()
            self.thrtype[0].hide()
            self.thrtype[1].hide()
            self.thrlabel.hide()
            self.thr.hide()
            self.aabox1.hide()
            self.aabox2.hide()
            self.waveletlabel.hide()
            self.wavelet.hide()
        elif (self.prevAlg == "Bandpass --> Wavelets" or self.prevAlg == "Wavelets --> Bandpass") and not self.DOC:
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
            self.low.hide()
            self.lowtext.hide()
            self.high.hide()
            self.hightext.hide()
            #self.trimlabel.hide()
            #self.trimaxis.hide()
            #self.trimaxis.setChecked(False)
            self.medlabel.hide()
            self.widthlabel.hide()
            self.width.hide()
        elif self.prevAlg == "Bandpass" or self.prevAlg == "Butterworth Bandpass":
            self.bandlabel.hide()
            self.blabel.hide()
            self.low.hide()
            self.lowtext.hide()
            self.high.hide()
            self.hightext.hide()
            #self.trimlabel.hide()
            #self.trimaxis.hide()
            #self.trimaxis.setChecked(False)
        else:
            # Median filter
            self.medlabel.hide()
            self.widthlabel.hide()
            self.width.hide()

        self.prevAlg = str(alg)
        if str(alg) == "Wavelets" and not self.DOC:
            # TEST OPTION: boxes are currently same as for Wavelets
            # self.wavlabel.show()
            self.depthlabel.show()
            self.depthchoice.show()
            self.depth.show()
            self.thrtypelabel.show()
            self.thrtype[0].show()
            self.thrtype[1].show()
            self.thrlabel.show()
            self.thr.show()
            self.aabox1.show()
            self.aabox2.show()
            self.waveletlabel.show()
            self.wavelet.show()
        elif str(alg) == "Wavelets --> Bandpass" and not self.DOC:
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
            self.low.show()
            self.lowtext.show()
            self.high.show()
            self.hightext.show()
            #self.trimlabel.show()
            #self.trimaxis.show()
        elif str(alg) == "Bandpass --> Wavelets" and not self.DOC:
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
            self.low.show()
            self.lowtext.show()
            self.high.show()
            self.hightext.show()
            #self.trimlabel.show()
            #self.trimaxis.show()
        elif str(alg) == "Bandpass" or str(alg) == "Butterworth Bandpass":
            self.bandlabel.show()
            self.low.show()
            self.lowtext.show()
            self.high.show()
            self.hightext.show()
            #self.trimlabel.show()
            #self.trimaxis.show()
        # else:
        #     #"Median filter"
        #     self.medlabel.show()
        #     self.widthlabel.show()
        #     self.width.show()

    def depthclicked(self):
        self.depth.setEnabled(not self.depth.isEnabled())

    def getValues(self):
        if not self.DOC:
            # some preprocessing of dialog options before returning
            if self.thrtype[0].isChecked() is True:
                thrType = 'soft'
            else:
                thrType = 'hard'
            
            if self.depthchoice.isChecked():
                depth = 0 # "please auto-find best"
            else:
                depth = int(str(self.depth.text()))
            return [self.algs.currentText(), depth, thrType, self.thr.text(),self.wavelet.currentText(),self.low.value(),self.high.value(),self.width.text(), self.aabox1.isChecked(), self.aabox2.isChecked()]
        else:
            return [self.algs.currentText(),self.low.value(),self.high.value(),self.width.text()]#,self.trimaxis.isChecked()]

    def lowChange(self,value):
        self.lowtext.setText(str(value))

    def highChange(self,value):
        self.hightext.setText(str(value))

#======
class HumanClassify1(QDialog):
    # This dialog allows the checking of classifications for segments.
    # It shows a single segment at a time, working through all the segments.

    def __init__(self, lut, colourStart, colourEnd, cmapInverted, brightness, contrast, shortBirdList, longBirdList, multipleBirds, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Check Classifications')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.frame = QWidget()

        self.lut = lut
        self.label = []
        self.colourStart = colourStart
        self.colourEnd = colourEnd
        self.cmapInverted = cmapInverted
        self.shortBirdList = shortBirdList
        self.longBirdList = longBirdList
        self.multipleBirds = multipleBirds
        self.saveConfig = False
        # exec_ forces the cursor into waiting
        self.activateWindow()
        pg.QtGui.QApplication.setOverrideCursor(Qt.ArrowCursor)

        # Set up the plot window, then the right and wrong buttons, and a close button
        self.wPlot = pg.GraphicsLayoutWidget()
        self.pPlot = self.wPlot.addViewBox(enableMouse=False, row=0, col=1)
        self.plot = pg.ImageItem()
        self.pPlot.addItem(self.plot)
        self.plotAspect = 0.2
        self.pPlot.setAspectLocked(ratio=self.plotAspect)
        self.pPlot.disableAutoRange()
        self.pPlot.setLimits(xMin=0, yMin=-5)
        self.sg_axis = pg.AxisItem(orientation='left')
        #self.sg_axis2 = pg.AxisItem(orientation='right')
        self.wPlot.addItem(self.sg_axis, row=0, col=0)
        #self.wPlot.addItem(self.sg_axis2, row=0, col=2)

        self.sg_axis.linkToView(self.pPlot)
        #self.sg_axis2.linkToView(self.pPlot)

        # prepare the lines for marking true segment boundaries
        self.line1 = pg.InfiniteLine(angle=90, pen={'color': 'g'})
        self.line2 = pg.InfiniteLine(angle=90, pen={'color': 'g'})
        self.pPlot.addItem(self.line1)
        self.pPlot.addItem(self.line2)

        # time texts to go along these two lines
        self.segTimeText1 = pg.TextItem(color=(50,205,50), anchor=(0,1.10))
        self.segTimeText2 = pg.TextItem(color=(50,205,50), anchor=(0,0.75))
        self.pPlot.addItem(self.segTimeText1)
        self.pPlot.addItem(self.segTimeText2)

        # playback line
        self.bar = pg.InfiniteLine(angle=90, movable=False, pen={'color':'c', 'width': 3})
        self.bar.btn = QtCore.Qt.RightButton
        self.bar.setValue(0)
        self.pPlot.addItem(self.bar)

        # label for current segment assignment
        self.speciesTop = QLabel("Currently:")
        self.species = QLabel()
        font = self.species.font()
        font.setPointSize(24)
        font.setBold(True)
        self.species.setFont(font)
        self.parent = parent

        # The buttons to move through the overview
        self.numberDone = QLabel()
        self.numberLeft = QLabel()
        self.numberDone.setAlignment(QtCore.Qt.AlignCenter)
        self.numberLeft.setAlignment(QtCore.Qt.AlignCenter)

        iconSize = QtCore.QSize(50, 50)
        self.buttonPrev = QtGui.QToolButton()
        self.buttonPrev.setIcon(self.style().standardIcon(QtGui.QStyle.SP_ArrowBack))
        self.buttonPrev.setIconSize(iconSize)

        self.correct = QtGui.QToolButton()
        self.correct.setIcon(QtGui.QIcon('img/tick.jpg'))
        self.correct.setIconSize(iconSize)

        self.delete = QtGui.QToolButton()
        self.delete.setIcon(QtGui.QIcon('img/delete.jpg'))
        self.delete.setIconSize(iconSize)

        self.buttonNext = QtGui.QToolButton()
        self.buttonNext.setIcon(self.style().standardIcon(QtGui.QStyle.SP_ArrowForward))
        self.buttonNext.setIconSize(iconSize)

        # The list of less common birds
        self.birds3 = QListWidget(self)
        if self.longBirdList is not None and self.longBirdList != 'None':
            for item in self.longBirdList:
                if '>' in item:
                    ind = item.index('>')
                    item = item[:ind] + " (" + item[ind+1:] + ")"
                self.birds3.addItem(item)
        # Explicitly add "Other" option in
        self.birds3.addItem('Other')
        self.birds3.setMaximumWidth(400)
        self.birds3.itemClicked.connect(self.listBirdsClicked)

        # An array of check boxes and a list and a text entry box
        # Create an array of check bixes for the most common birds (2 columns of 10 choices)
        self.birds = QButtonGroup()
        self.birdbtns = []
        if self.multipleBirds:
            self.birds.setExclusive(False)
            for item in self.shortBirdList[:29]:
                self.birdbtns.append(QCheckBox(item))
                self.birds.addButton(self.birdbtns[-1],len(self.birdbtns)-1)
                self.birdbtns[-1].clicked.connect(self.tickBirdsClicked)
            self.birdbtns.append(QCheckBox('Other')),
            self.birds.addButton(self.birdbtns[-1],len(self.birdbtns)-1)
            self.birdbtns[-1].clicked.connect(self.tickBirdsClicked)
            self.birds3.setSelectionMode(QAbstractItemView.MultiSelection)
        else:
            self.birds.setExclusive(True)
            for item in self.shortBirdList[:29]:
                btn = QRadioButton(item)
                self.birdbtns.append(btn)
                self.birds.addButton(btn,len(self.birdbtns)-1)
                btn.clicked.connect(self.radioBirdsClicked)
            self.birdbtns.append(QRadioButton('Other')),
            self.birds.addButton(self.birdbtns[-1],len(self.birdbtns)-1)
            self.birdbtns[-1].clicked.connect(self.radioBirdsClicked)
            self.birds3.setSelectionMode(QAbstractItemView.SingleSelection)

        self.birds3.setEnabled(False)

        # This is the text box for missing birds
        self.tbox = QLineEdit(self)
        self.tbox.setMaximumWidth(150)
        self.tbox.returnPressed.connect(self.birdTextEntered)
        self.tbox.setEnabled(False)

        # Audio playback object
        self.media_obj2 = SupportClasses.ControllableAudio(self.parent.audioFormat)
        self.media_obj2.notify.connect(self.endListener)

        # The layouts
        birds1Layout = QVBoxLayout()
        birds2Layout = QVBoxLayout()
        birds3Layout = QVBoxLayout()
        count = 0
        for btn in self.birdbtns:
            if count<10: 
                birds1Layout.addWidget(btn)
            elif count<20:
                birds2Layout.addWidget(btn)
            else:
                birds3Layout.addWidget(btn)
            count += 1

        birdListLayout = QVBoxLayout()
        birdListLayout.addWidget(self.birds3)
        birdListLayout.addWidget(QLabel("If bird isn't in list, select Other"))
        birdListLayout.addWidget(QLabel("Type below, Return at end"))
        birdListLayout.addWidget(self.tbox)

        hboxBirds = QHBoxLayout()
        hboxBirds.addLayout(birds1Layout)
        hboxBirds.addLayout(birds2Layout)
        hboxBirds.addLayout(birds3Layout)
        hboxBirds.addLayout(birdListLayout)

        # The layouts
        hboxNextPrev = QHBoxLayout()
        hboxNextPrev.addWidget(self.numberDone)
        hboxNextPrev.addWidget(self.buttonPrev)
        hboxNextPrev.addWidget(self.correct)
        hboxNextPrev.addWidget(self.delete)
        hboxNextPrev.addWidget(self.buttonNext)
        hboxNextPrev.addWidget(self.numberLeft)

        self.playButton = QtGui.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(QtCore.QSize(40, 40))
        self.playButton.clicked.connect(self.playSeg)

        self.scroll = QtGui.QScrollArea()
        self.scroll.setWidget(self.wPlot)
        self.scroll.setWidgetResizable(True)

        # Volume control
        self.volSlider = QSlider(Qt.Horizontal)
        self.volSlider.valueChanged.connect(self.volSliderMoved)
        self.volSlider.setRange(0,100)
        self.volSlider.setValue(50)
        self.volIcon = QLabel()
        self.volIcon.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.volIcon.setPixmap(self.style().standardIcon(QtGui.QStyle.SP_MediaVolume).pixmap(32))

        # Brightness, and contrast sliders
        self.brightnessSlider = QSlider(Qt.Horizontal)
        self.brightnessSlider.setMinimum(0)
        self.brightnessSlider.setMaximum(100)
        self.brightnessSlider.setValue(brightness)
        self.brightnessSlider.setTickInterval(1)
        self.brightnessSlider.valueChanged.connect(self.setColourLevels)

        self.contrastSlider = QSlider(Qt.Horizontal)
        self.contrastSlider.setMinimum(0)
        self.contrastSlider.setMaximum(100)
        self.contrastSlider.setValue(contrast)
        self.contrastSlider.setTickInterval(1)
        self.contrastSlider.valueChanged.connect(self.setColourLevels)

        # zoom buttons
        self.zoomInBtn = QPushButton("+")
        self.zoomOutBtn = QPushButton("-")
        self.zoomInBtn.clicked.connect(self.zoomIn)
        self.zoomOutBtn.clicked.connect(self.zoomOut)

        vboxSpecContr = pg.LayoutWidget()
        vboxSpecContr.addWidget(self.speciesTop, row=0, col=0, colspan=2)
        vboxSpecContr.addWidget(self.species, row=0, col=2, colspan=8)
        vboxSpecContr.addWidget(self.scroll, row=1, col=0, colspan=12)
        vboxSpecContr.addWidget(self.playButton, row=2, col=0)
        vboxSpecContr.addWidget(self.volIcon, row=2, col=1)
        vboxSpecContr.addWidget(self.volSlider, row=2, col=2, colspan=2)
        labelBr = QLabel("Bright.")
        labelBr.setAlignment(QtCore.Qt.AlignRight)
        vboxSpecContr.addWidget(labelBr, row=2, col=4)
        vboxSpecContr.addWidget(self.brightnessSlider, row=2, col=5, colspan=2)
        labelCo = QLabel("Contr.")
        labelCo.setAlignment(QtCore.Qt.AlignRight)
        vboxSpecContr.addWidget(labelCo, row=2, col=7)
        vboxSpecContr.addWidget(self.contrastSlider, row=2, col=8, colspan=2)
        vboxSpecContr.addWidget(self.zoomInBtn, row=2, col=10)
        vboxSpecContr.addWidget(self.zoomOutBtn, row=2, col=11)


        vboxFull = QVBoxLayout()
        vboxFull.addWidget(vboxSpecContr)
        vboxFull.addLayout(hboxBirds)
        vboxFull.addLayout(hboxNextPrev)

        self.setLayout(vboxFull)
        # print seg
        # self.setImage(self.sg,audiodata,sampleRate,self.label, unbufStart, unbufStop)

    def playSeg(self):
        if self.media_obj2.isPlaying():
            self.stopPlayback()
        else:
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
            self.playButton.setIconSize(QtCore.QSize(40, 40))
            self.media_obj2.loadArray(self.audiodata)

    def stopPlayback(self):
        self.media_obj2.pressedStop()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(QtCore.QSize(40, 40))

    def volSliderMoved(self, value):
        self.media_obj2.applyVolSlider(value)

    def endListener(self):
        """ Listener to check for playback end.
        Also hijacked to move the playback bar."""
        time = self.media_obj2.elapsedUSecs() // 1000
        if time > self.duration:
            self.stopPlayback()
        else:
            barx = time / 1000 * self.sampleRate / self.incr
            self.bar.setValue(barx)
            self.bar.update()
            # QApplication.processEvents()

    def zoomIn(self):
        self.plotAspect = self.plotAspect * 1.5
        self.pPlot.setAspectLocked(ratio=self.plotAspect)
        xyratio = np.shape(self.sg)
        xyratio = xyratio[0] / xyratio[1]
        self.wPlot.setMaximumSize(max(500, xyratio*250*self.plotAspect*0.9), 250)
        self.wPlot.setMinimumSize(max(500, xyratio*250*self.plotAspect*0.9), 250)

    def zoomOut(self):
        self.plotAspect = self.plotAspect / 1.5
        #self.plot.setImage(self.sg)
        self.pPlot.setAspectLocked(ratio=self.plotAspect)
        xyratio = np.shape(self.sg)
        xyratio = xyratio[0] / xyratio[1]
        self.wPlot.setMaximumSize(max(500, xyratio*250*self.plotAspect*0.9), 250)
        self.wPlot.setMinimumSize(max(500, xyratio*250*self.plotAspect*0.9), 250)

    def updateButtonList(self):
        # refreshes bird button names
        # to be used when bird list updates
        for i in range(len(self.birdbtns)-1):
            self.birdbtns[i].setChecked(False)
            self.birdbtns[i].setText(self.shortBirdList[i])
        # "other" button
        self.birdbtns[-1].setChecked(False)
        self.birds3.setEnabled(False)

    def setSegNumbers(self, done, total):
        text1 = "calls reviewed: " + str(done)
        text2 = str(total - done) + " to go"
        self.numberDone.setText(text1)
        self.numberLeft.setText(text2)

    def setImage(self, sg, audiodata, sampleRate, incr, label, unbufStart, unbufStop, time1, time2, minFreq=0, maxFreq=0):
        """ label - list of species in the current segment.
            Currently, we ignore the certainty and just display the species.
            During review, this updates self.label.
        """
        self.audiodata = audiodata
        self.sg = sg
        self.sampleRate = sampleRate
        self.incr = incr
        self.bar.setValue(0)
        if maxFreq==0:
            maxFreq = sampleRate / 2
        self.duration = len(audiodata) / sampleRate * 1000 # in ms

        #print("Parent species:" , self.parent.segments[self.parent.box1id][4])

        # fill up a rectangle with dark grey to act as background if the segment is small
        sg2 = sg
        # sg2 = 40 * np.ones((max(1000, np.shape(sg)[0]), max(100, np.shape(sg)[1])))
        # sg2[:np.shape(sg)[0], :np.shape(sg)[1]] = sg

        # add axis
        self.plot.setImage(sg2)
        self.plot.setLookupTable(self.lut)
        self.setColourLevels()
        self.scroll.horizontalScrollBar().setValue(0)

        FreqRange = (maxFreq-minFreq)/1000.
        SgSize = np.shape(sg2)[1]
        ticks = [[(0,minFreq/1000.), (SgSize/4, minFreq/1000.+FreqRange/4.), (SgSize/2, minFreq/1000.+FreqRange/2.), (3*SgSize/4, minFreq/1000.+3*FreqRange/4.), (SgSize,minFreq/1000.+FreqRange)]]
        self.sg_axis.setTicks(ticks)
        self.sg_axis.setLabel('kHz')
        #self.sg_axis2.setTicks(ticks)
        #self.sg_axis2.setLabel('kHz')

        self.show()

        self.pPlot.setYRange(0, SgSize, padding=0.02)
        self.pPlot.setRange(xRange=(0, np.shape(sg2)[0]), yRange=(0, SgSize))
        xyratio = np.shape(sg2)
        xyratio = xyratio[0] / xyratio[1]
        # self.plotAspect = 0.2 for x/y pixel aspect ratio
        # 0.9 for padding
        # TODO: ***Issues here
        self.wPlot.setMaximumSize(max(500, xyratio*250*self.plotAspect*0.9), 250)
        self.wPlot.setMinimumSize(max(500, xyratio*250*self.plotAspect*0.9), 250)

        # add marks to separate actual segment from buffer zone
        # Note: need to use view coordinates to add items to pPlot
        try:
            self.stopPlayback()
        except Exception as e:
            print(e)
            pass
        startV = self.pPlot.mapFromItemToView(self.plot, QPointF(unbufStart, 0)).x()
        stopV = self.pPlot.mapFromItemToView(self.plot, QPointF(unbufStop, 0)).x()
        self.line1.setPos(startV)
        self.line2.setPos(stopV)
        # add time markers next to the lines
        time1 = QTime(0,0,0).addSecs(time1).toString('hh:mm:ss')
        time2 = QTime(0,0,0).addSecs(time2).toString('hh:mm:ss')
        self.segTimeText1.setText(time1)
        self.segTimeText2.setText(time2)
        self.segTimeText1.setPos(startV, SgSize)
        self.segTimeText2.setPos(stopV, SgSize)

        if self.cmapInverted:
            self.plot.setLevels([self.colourEnd, self.colourStart])
        else:
            self.plot.setLevels([self.colourStart, self.colourEnd])

        # DEAL WITH SPECIES NAMES

        # currently, we ignore the certainty and only display species:
        self.species.setText(','.join(label))
        # Select the right species tickboxes / buttons
        self.birds3.clearSelection()
        self.updateButtonList()
        for lsp in label:
            # question marks are displayed on the first pass,
            # but any clicking sets certainty to 100 in effect.
            if lsp.endswith('?'):
                lsp = lsp[:-1]
            if lsp in self.shortBirdList[:29]:
                ind = self.shortBirdList.index(lsp)
                self.birdbtns[ind].setChecked(True)
                print(ind,lsp)
            else:
                self.birdbtns[29].setChecked(True)
                self.birds3.setEnabled(True)
                if lsp not in self.longBirdList:
                    if '(' in lsp:
                        ind = lsp.index('(')
                        lsp = lsp[:ind-1] + ">" + lsp[ind+1:-1]
                if lsp not in self.longBirdList:
                    self.longBirdList.append(lsp)
                    self.saveConfig = True
                ind = self.longBirdList.index(lsp)
                self.birds3.item(ind).setSelected(True)
        self.label = label

    def tickBirdsClicked(self):
        # Listener for when the user selects a bird tick box
        # Update the text and store the data
        # This makes it easy to switch out of DontKnow-only segments
        checkedButton = None
        dontknowButton = None
        for button in self.birds.buttons():
            if button.text() == "Other":
                # just toggle the long list box
                if button.isChecked():
                    self.birds3.setEnabled(True)
                else:
                    self.birds3.setEnabled(False)
            else:
                if button.text() == "Don't Know":
                    dontknowButton = button
                # figure out which one was changed now
                if button.isChecked():
                    if button.text() not in self.label:
                        # this was just ticked ON
                        checkedButton = button
                else:
                    if button.text() in self.label:
                        # this was just ticked OFF
                        checkedButton = button
        if checkedButton is None:
            print("Warning: unrecognized check event")
            return
        if checkedButton.isChecked():
            # if label was empty, just change from DontKnow:
            if self.label == ["Don't Know"]:
                self.label = [checkedButton.text()]
                if dontknowButton is not None:
                    dontknowButton.setChecked(False)
            else:
                self.label.append(checkedButton.text())
        else:
            # a button was unchecked:
            if checkedButton.text() in self.label:
                self.label.remove(checkedButton.text())
            # if this erased everything, revert to don't know:
            if self.label == []:
                self.label = ["Don't Know"]
                if dontknowButton is not None:
                    dontknowButton.setChecked(True)
        self.species.setText(','.join(self.label))

    def radioBirdsClicked(self):
        # Listener for when the user selects a radio button
        # Update the text and store the data
        for button in self.birdbtns:
            if button.isChecked():
                if button.text() == "Other":
                    self.birds3.setEnabled(True)
                else:
                    self.birds3.setEnabled(False)
                    self.label = [button.text()]
                    self.species.setText(button.text())

    def listBirdsClicked(self, item):
        # Listener for clicks in the listbox of birds
        if (item.text() == "Other"):
            self.tbox.setEnabled(True)
        else:
            #TODO: Check if selected or not
            # Save the entry
            self.tbox.setEnabled(False)
            if self.multipleBirds:
                if item.isSelected() and item.text() not in self.label:
                    self.label.append(str(item.text()))
                if not item.isSelected() and item.text in self.label:
                    self.label.remove(str(item.text()))
            else:
                self.label = [str(item.text())]
            self.species.setText(','.join(self.label))

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
        if self.multipleBirds:
            self.label.append(str(textitem))
        else:
            self.label = [str(textitem)]
        self.species.setText(','.join(self.label))
        self.saveConfig = True

    def setColourLevels(self):
        """ Listener for the brightness and contrast sliders being changed. Also called when spectrograms are loaded, etc.
        Translates the brightness and contrast values into appropriate image levels.
        Calculation is simple.
        """
        try:
            self.stopPlayback()
        except Exception:
            pass
        minsg = np.min(self.sg)
        maxsg = np.max(self.sg)
        if self.cmapInverted:
            brightness = self.brightnessSlider.value()
        else:
            brightness = 100-self.brightnessSlider.value()
        contrast = self.contrastSlider.value()

        self.colourStart = (brightness / 100.0 * contrast / 100.0) * (maxsg - minsg) + minsg
        self.colourEnd = (maxsg - minsg) * (1.0 - contrast / 100.0) + self.colourStart
        if self.cmapInverted:
            self.plot.setLevels([self.colourEnd, self.colourStart])
        else:
            self.plot.setLevels([self.colourStart, self.colourEnd])

    def getValues(self):
        return [self.label, self.saveConfig, self.tbox.text()]


class HumanClassify2a(QDialog):
    # This is a small popup dialog for selecting species to review in Classify2
    def __init__(self, birdlist,parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Human review')
        self.setWindowIcon(QIcon('img/Avianz.ico'))

        self.birds = QListWidget(self)
        self.birds.setMaximumWidth(350)
        #self.birds.addItem('All calls')
        #self.birds.addItem('Uncertain calls')
        for item in birdlist:
            self.birds.addItem(item)
        #self.birds.setCurrentRow(0)
        self.birds.itemDoubleClicked.connect(self.dbl)

        ok = QPushButton('OK')
        cancel = QPushButton('Cancel')
        ok.clicked.connect(self.ok)
        cancel.clicked.connect(self.cancel)

        layout = QVBoxLayout()
        layout.addWidget(QLabel('Choose species/call type to review:'))
        layout.addWidget(self.birds)
        layout.addWidget(ok)
        layout.addWidget(cancel)

        # Now put everything into the frame
        self.setLayout(layout)

    def dbl(self,item):
        self.birds.setCurrentItem(item)
        self.accept()

    def ok(self):
        self.accept()

    def cancel(self):
        self.reject()

    def getValues(self):
        return self.birds.currentItem().text()

class HumanClassify2(QDialog):
    """ Single Species review main dialog.
        Puts all segments of a certain species together on buttons, and their labels.
        Allows quick confirm/leave/delete check over many segments.

        Construction:
        1-3. spectrogram, audiodata, segments. Just provide full versions of these,
          and this dialog will select the needed parts/segments.
        4. name of the species that we are reviewing
        5-6. sampleRate, audioFormat for playback
        7. increment which was used for the spectrogram
        8-13. spec color parameters
        14. page start, in seconds - to convert segment-time to spec-time
        15. ???
    """

    def __init__(self, sg, audiodata, segments, label, sampleRate, audioFormat, incr, lut, colourStart, colourEnd, cmapInverted, brightness, contrast, filename=None, startRead=0):
        QDialog.__init__(self)

        if len(segments)==0:
            print("No segments provided")
            return

        if filename:
            self.setWindowTitle('Human review - ' + filename)
        else:
            self.setWindowTitle('Human review')

        self.setWindowIcon(QIcon('img/Avianz.ico'))

        # let the user quit without bothering rest of it
        self.setWindowFlags(self.windowFlags() & QtCore.Qt.WindowCloseButtonHint)

        self.sampleRate = sampleRate
        self.audiodata = audiodata
        self.audioFormat = audioFormat
        self.incr = incr
        self.lut = lut
        self.colourStart = colourStart
        self.colourEnd = colourEnd
        self.cmapInverted = cmapInverted
        self.startRead = startRead

        # Seems that image is backwards?
        self.sg = np.fliplr(sg)

        # filter segments for the requested species
        self.segments = segments
        self.indices2show = self.segments.getSpecies(label)
        for i in reversed(self.indices2show):
            # show segments which have midpoint in this page (ensures all are shown only once)
            mid = (segments[i][0] + segments[i][1]) / 2
            if mid < startRead or mid > startRead + len(audiodata)//sampleRate:
                del self.indices2show[i]

        self.errors = []

        # Volume control
        self.volSlider = QSlider(Qt.Horizontal)
        self.volSlider.valueChanged.connect(self.volSliderMoved)
        self.volSlider.setRange(0,100)
        self.volSlider.setValue(50)
        self.volIcon = QLabel()
        self.volIcon.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.volIcon.setPixmap(self.style().standardIcon(QtGui.QStyle.SP_MediaVolume).pixmap(32))

        # Brightness, and contrast sliders
        self.brightnessSlider = QSlider(Qt.Horizontal)
        self.brightnessSlider.setMinimum(0)
        self.brightnessSlider.setMaximum(100)
        self.brightnessSlider.setValue(brightness)
        self.brightnessSlider.setTickInterval(1)
        self.brightnessSlider.valueChanged.connect(self.setColourLevels)

        self.contrastSlider = QSlider(Qt.Horizontal)
        self.contrastSlider.setMinimum(0)
        self.contrastSlider.setMaximum(100)
        self.contrastSlider.setValue(contrast)
        self.contrastSlider.setTickInterval(1)
        self.contrastSlider.valueChanged.connect(self.setColourLevels)

        hboxSpecContr = QHBoxLayout()
        hboxSpecContr.addWidget(self.volIcon)
        hboxSpecContr.addWidget(self.volSlider)
        labelBr = QLabel("Bright.")
        hboxSpecContr.addWidget(labelBr)
        hboxSpecContr.addWidget(self.brightnessSlider)
        labelCo = QLabel("Contr.")
        hboxSpecContr.addWidget(labelCo)
        hboxSpecContr.addWidget(self.contrastSlider)

        label1 = QLabel('Click on the images that are incorrectly labelled.')
        label1.setFont(QtGui.QFont('SansSerif', 10))
        species = QLabel(label)
        font = QtGui.QFont('SansSerif', 12)
        font.setBold(True)
        species.setFont(font)

        # species label and sliders
        vboxTop = QVBoxLayout()
        vboxTop.addWidget(label1)
        vboxTop.addWidget(species)
        vboxTop.addLayout(hboxSpecContr)

        # Controls at the bottom
        self.buttonPrev = QtGui.QToolButton()
        self.buttonPrev.setArrowType(Qt.LeftArrow)
        self.buttonPrev.setIconSize(QtCore.QSize(30,30))
        self.buttonPrev.clicked.connect(self.prevPage)

        self.buttonNext = QtGui.QToolButton()
        self.buttonNext.setArrowType(Qt.RightArrow)
        self.buttonNext.setIconSize(QtCore.QSize(30,30))
        self.buttonNext.clicked.connect(self.nextPage)
        self.pageLabel = QLabel()

        self.none = QPushButton("Toggle all")
        self.none.setSizePolicy(QSizePolicy(5,5))
        self.none.setMaximumSize(250, 30)
        self.none.clicked.connect(self.toggleAll)

        self.finish = QPushButton("Confirm and close")
        self.finish.setSizePolicy(QSizePolicy(5,5))
        self.finish.setMaximumSize(250, 30)

        # movement buttons and page numbers
        vboxBot = QHBoxLayout()
        vboxBot.addWidget(self.buttonPrev)
        vboxBot.addWidget(self.buttonNext)
        vboxBot.addSpacing(20)
        vboxBot.addWidget(self.pageLabel)
        vboxBot.addSpacing(20)
        vboxBot.addWidget(self.none)
        vboxBot.addWidget(self.finish)

        # set up the middle section of images
        self.numPicsV = 0
        self.numPicsH = 0
        # create the button objects, and we'll show them as needed
        # (fills self.buttons)
        # self.flowLayout = QGridLayout()
        self.flowLayout = pg.LayoutWidget()
        # these sizes ensure at least one image fits:
        self.specV = 0
        self.specH = 0
        self.createButtons()

        # sets a lot of self properties needed before showing anything
        self.butStart = 0
        self.countPages()

        self.flowLayout.setMinimumSize(self.specH+20, self.specV+20)

        # set overall layout of the dialog
        self.vboxFull = QVBoxLayout()
        # self.vboxFull.setSpacing(0)
        self.vboxFull.addLayout(vboxTop)
        self.vboxSpacer = QSpacerItem(1,1, 5, 5)
        self.vboxFull.addItem(self.vboxSpacer)
        self.vboxFull.addWidget(self.flowLayout)
        self.vboxFull.addLayout(vboxBot)
        # must be fixed size!
        vboxTop.setSizeConstraint(QLayout.SetFixedSize)
        # must be fixed size!
        vboxBot.setSizeConstraint(QLayout.SetFixedSize)

        # we need to know the true size of space available for flowLayout.
        # the idea is that spacer absorbs all height changes
        self.setSizePolicy(1,1)
        self.setMinimumSize(self.specH+100, self.specV+100)
        self.setLayout(self.vboxFull)
        self.vboxFull.setStretch(1, 100)
        # plan B could be to measure the sizes of the top/bottom boxes and subtract them
        # self.boxSpaceAdjustment = vboxTop.sizeHint().height() + vboxBot.sizeHint().height()

    def createButtons(self):
        """ Create the button objects, add audio, calculate spec, etc.
            So that when users flips through pages, we only need to
            retrieve the right ones from resizeEvent.
            No return, fills out self.buttons.
        """
        self.buttons = []
        self.marked = []
        for i in self.indices2show:
            seg = self.segments[i]

            # select 10 s region around segment center
            mid = (seg[0] + seg[1])/2 - self.startRead
            tstart = min(len(self.audiodata)/self.sampleRate-1, max(0, mid-5))
            tend = min(len(self.audiodata)/self.sampleRate, mid+5)

            # find the right boundaries from the audiodata
            x1a = int(tstart * self.sampleRate)
            x2a = int(tend * self.sampleRate)
            # get the right slice from the spectrogram:
            x1 = int(self.convertAmpltoSpec(tstart))
            x2 = int(self.convertAmpltoSpec(tend))

            # boundaries of raw segment, in spec units, relative to start of seg:
            unbufStart = self.convertAmpltoSpec(seg[0]-self.startRead) - x1
            unbufStop = self.convertAmpltoSpec(seg[1]-self.startRead) - x1

            # create the button:
            newButton = PicButton(i, self.sg[x1:x2, :], self.audiodata[x1a:x2a], self.audioFormat, tend-tstart, unbufStart, unbufStop, self.lut, self.colourStart, self.colourEnd, self.cmapInverted)
            if newButton.im1.size().width() > self.specH:
                self.specH = newButton.im1.size().width()
            if newButton.im1.size().height() > self.specV:
                self.specV = newButton.im1.size().height()

            #newButton.setMinimumSize(self.specH, self.specV//2)

            self.buttons.append(newButton)
            self.buttons[-1].buttonClicked=False
            self.marked.append(False)
        # set volume and brightness on these new buttons
        self.volSliderMoved(self.volSlider.value())
        self.setColourLevels()

    def resizeEvent(self, ev):
        """ On this event, choose which (and how many) buttons to display
            from self.buttons. It is also called on initialization.
        """
        QDialog.resizeEvent(self, ev)

        # space for remaining widgets:
        # width is just dialog width with small buffer for now
        spaceV = self.flowLayout.size().height() + self.vboxSpacer.geometry().height()
        spaceH = self.size().width() - 20

        # this is the grid that fits in dialog, and self.* is the current grid
        # Let's say we leave 10px for spacing
        numPicsV = spaceV // (self.specV+10)
        numPicsH = spaceH // (self.specH+10)
        # only need to redraw pics if the grid has changed
        if numPicsV!=self.numPicsV or numPicsH!=self.numPicsH:
            # clear
            self.clearButtons()
            # draw new
            self.numPicsV = numPicsV
            self.numPicsH = numPicsH
            self.redrawButtons()
            self.countPages()
        return

    def redrawButtons(self):
        butNum = 0
        for row in range(self.numPicsV):
            for col in range(self.numPicsH):
                # resizing shouldn't change which segments are displayed,
                # so we use a fixed start point for counting shown buttons.
                self.flowLayout.addWidget(self.buttons[self.butStart+butNum], row, col)
                # just in case, update the bounds of grid on every redraw
                self.flowLayout.layout.setColumnMinimumWidth(col, self.specH+10)
                self.flowLayout.layout.setRowMinimumHeight(row, self.specV+10)
                self.buttons[self.butStart+butNum].show()
                butNum += 1
                # stop if we are out of segments
                if self.butStart+butNum==len(self.buttons):
                    return

    def volSliderMoved(self, value):
        # try/pass to avoid race situations when smth is not initialized
        try:
            for btn in self.buttons:
                btn.media_obj.applyVolSlider(value)
        except Exception:
            pass

    def convertAmpltoSpec(self, x):
        return x * self.sampleRate / self.incr

    def countPages(self):
        """ Counts the total number of pages,
            finds where we are, how many remain, etc.
            Called on resize, so does not update current button position.
            Updates next/prev arrow states.
        """
        buttonsPerPage = self.numPicsV * self.numPicsH
        if buttonsPerPage == 0:
            # dialog still initializing or too small to show segments
            self.buttonPrev.setEnabled(False)
            self.buttonNext.setEnabled(False)
            return
        self.totalPages = np.ceil(len(self.buttons) / buttonsPerPage)
        # basically, count how many segments are "before" the current
        # top-lef one, and see how many pages we need to fit them.
        currpage = np.ceil(self.butStart / buttonsPerPage)+1
        self.pageLabel.setText("Page %d out of %d" % (currpage, self.totalPages))
        if currpage==1:
            self.buttonPrev.setEnabled(False)
        else:
            self.buttonPrev.setEnabled(True)
        if currpage==self.totalPages:
            self.buttonNext.setEnabled(False)
        else:
            self.buttonNext.setEnabled(True)

    def nextPage(self):
        """ Called on arrow button clicks.
            Updates current segment position, and calls other functions
            to deal with actual page recount/redraw.
        """
        buttonsPerPage = self.numPicsV * self.numPicsH
        # clear buttons while self.butStart is still old:
        self.clearButtons()
        self.butStart = min(len(self.buttons), self.butStart+buttonsPerPage)
        self.countPages()
        # redraw buttons:
        self.redrawButtons()

    def prevPage(self):
        """ Called on arrow button clicks.
            Updates current segment position, and calls other functions
            to deal with actual page recount/redraw.
        """
        buttonsPerPage = self.numPicsV * self.numPicsH
        # clear buttons while self.butStart is still old:
        self.clearButtons()
        self.butStart = max(0, self.butStart-buttonsPerPage)
        self.countPages()
        # redraw buttons:
        self.redrawButtons()

    def clearButtons(self):
        for btnum in reversed(range(self.flowLayout.layout.count())):
            item = self.flowLayout.layout.itemAt(btnum)
            if item is not None:
                self.flowLayout.layout.removeItem(item)
                r,c = self.flowLayout.items[item.widget()]
                self.flowLayout.layout.setColumnMinimumWidth(c, 1)
                self.flowLayout.layout.setRowMinimumHeight(r, 1)
                del self.flowLayout.rows[r][c]
                item.widget().hide()
        self.flowLayout.update()

    def toggleAll(self):
            for btn in self.buttons:
                btn.changePic(False)

    def setColourLevels(self):
        """ Listener for the brightness and contrast sliders being changed. Also called when spectrograms are loaded, etc.
        Translates the brightness and contrast values into appropriate image levels.
        Calculation is simple.
        """
        minsg = np.min(self.sg)
        maxsg = np.max(self.sg)
        brightness = self.brightnessSlider.value()
        contrast = self.contrastSlider.value()
        colourStart = (brightness / 100.0 * contrast / 100.0) * (maxsg - minsg) + minsg
        colourEnd = (maxsg - minsg) * (1.0 - contrast / 100.0) + colourStart
        for btn in self.buttons:
            btn.stopPlayback()
            btn.setImage(self.lut, colourStart, colourEnd, self.cmapInverted)
            btn.update()


class PicButton(QAbstractButton):
    # Class for HumanClassify dialogs to put spectrograms on buttons
    # Also includes playback capability.
    def __init__(self, index, spec, audiodata, format, duration, unbufStart, unbufStop, lut, colStart, colEnd, cmapInv, parent=None, cluster=False):
        super(PicButton, self).__init__(parent)
        self.index = index
        self.mark = "green"
        self.spec = spec
        self.unbufStart = unbufStart
        self.unbufStop = unbufStop
        self.cluster = cluster
        self.setMouseTracking(True)
        # setImage reads some properties from self, to allow easy update
        # when color map changes
        self.setImage(lut, colStart, colEnd, cmapInv)

        self.buttonClicked = False
        if not self.cluster:
            self.clicked.connect(self.changePic)
        # fixed size
        self.setSizePolicy(0,0)
        self.setMinimumSize(self.im1.size())

        # playback things
        self.media_obj = SupportClasses.ControllableAudio(format)
        self.media_obj.notify.connect(self.endListener)
        self.audiodata = audiodata
        self.duration = duration * 1000 # in ms

        self.playButton = QtGui.QToolButton(self)
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.playImage)
        self.playButton.hide()

    def setImage(self, lut, colStart, colEnd, cmapInv):
        # takes in a piece of spectrogram and produces a pair of images
        if cmapInv:
            im, alpha = fn.makeARGB(self.spec, lut=lut, levels=[colEnd, colStart])
        else:
            im, alpha = fn.makeARGB(self.spec, lut=lut, levels=[colStart, colEnd])
        im1 = fn.makeQImage(im, alpha)
        if im1.size().width() == 0:
            print("ERROR: button not shown, likely bad spectrogram coordinates")
            return

        # hardcode all image sizes
        if self.cluster:
            self.im1 = im1.scaled(200, 150)
        else:
            self.specReductionFact = im1.size().width()/500
            self.im1 = im1.scaled(500, im1.size().height())

        # draw lines
        if not self.cluster:
            unbufStartAdj = self.unbufStart / self.specReductionFact
            unbufStopAdj = self.unbufStop / self.specReductionFact
            self.line1 = QLineF(unbufStartAdj, 0, unbufStartAdj, im1.size().height())
            self.line2 = QLineF(unbufStopAdj, 0, unbufStopAdj, im1.size().height())

    def paintEvent(self, event):
        if type(event) is not bool:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(80,255,80), 2))
            if self.mark == "yellow":
                painter.setOpacity(0.8)
            elif self.mark == "red":
                painter.setOpacity(0.5)
            painter.drawImage(event.rect(), self.im1)
            if not self.cluster:
                painter.drawLine(self.line1)
                painter.drawLine(self.line2)

            # draw decision mark
            fontsize = int(self.im1.size().height() * 0.65)
            if self.mark == "green":
                pass
            elif self.mark == "yellow":
                painter.setOpacity(0.9)
                painter.setPen(QPen(QColor(220,220,0)))
                painter.setFont(QFont("Helvetica", fontsize))
                painter.drawText(self.im1.rect(), Qt.AlignHCenter | Qt.AlignVCenter, "?")
            elif self.mark == "red":
                painter.setOpacity(0.8)
                painter.setPen(QPen(QColor(220,0,0)))
                painter.setFont(QFont("Helvetica", fontsize))
                painter.drawText(self.im1.rect(), Qt.AlignHCenter | Qt.AlignVCenter, "X")
            else:
                print("ERROR: unrecognized segment mark")
                return

    def enterEvent(self, QEvent):
        # to reset the icon if it didn't stop cleanly
        if not self.media_obj.isPlaying():
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.show()

    def leaveEvent(self, QEvent):
        if not self.media_obj.isPlaying():
            self.playButton.hide()

    def playImage(self):
        if self.media_obj.isPlaying():
            self.stopPlayback()
        else:
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
            self.media_obj.loadArray(self.audiodata)

    def endListener(self):
        time = self.media_obj.elapsedUSecs() // 1000
        if time > self.duration:
            self.stopPlayback()

    def stopPlayback(self):
        self.media_obj.pressedStop()
        self.playButton.hide()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))

    def sizeHint(self):
        return self.im1.size()

    def minimumSizeHint(self):
        return self.im1.size()

    def changePic(self,event):
        # cycle through CONFIRM / DELETE / RECHECK marks
        if self.mark == "green":
            self.mark = "red"
        elif self.mark == "red":
            self.mark = "yellow"
        elif self.mark == "yellow":
            self.mark = "green"
        self.paintEvent(event)
        #self.update()
        self.repaint()
        pg.QtGui.QApplication.processEvents()


class Cluster(QDialog):
    """ Cluster dialog.
        Shows clustered segments
        Allows to merge clusters and move segments from one class to another.
    """

    def __init__(self, segments, sampleRate, classes, config, parent=None):
        QDialog.__init__(self, parent)

        if len(segments) == 0:
            print("No segments provided")
            return

        self.setWindowTitle('Clusters')

        self.setWindowIcon(QIcon('img/Avianz.ico'))

        # let the user quit without bothering rest of it
        # self.setWindowFlags(self.windowFlags() & QtCore.Qt.WindowCloseButtonHint)

        self.sampleRate = sampleRate
        self.segments = segments
        self.nclasses = classes
        self.config = config

        # Merge button
        self.btnMerge = QPushButton('Merge Clusters')
        self.btnMerge.clicked.connect(self.merge)

        # Volume control
        self.volSlider = QSlider(Qt.Horizontal)
        self.volSlider.valueChanged.connect(self.volSliderMoved)
        self.volSlider.setRange(0, 100)
        self.volSlider.setValue(50)
        self.volIcon = QLabel()
        self.volIcon.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.volIcon.setPixmap(self.style().standardIcon(QtGui.QStyle.SP_MediaVolume).pixmap(32))

        # Brightness, and contrast sliders
        self.brightnessSlider = QSlider(Qt.Horizontal)
        self.brightnessSlider.setMinimum(0)
        self.brightnessSlider.setMaximum(100)
        self.brightnessSlider.setValue(20)
        self.brightnessSlider.setTickInterval(1)
        self.brightnessSlider.valueChanged.connect(self.setColourLevels)

        self.contrastSlider = QSlider(Qt.Horizontal)
        self.contrastSlider.setMinimum(0)
        self.contrastSlider.setMaximum(100)
        self.contrastSlider.setValue(20)
        self.contrastSlider.setTickInterval(1)
        self.contrastSlider.valueChanged.connect(self.setColourLevels)

        hboxSpecContr = QHBoxLayout()
        hboxSpecContr.addWidget(self.btnMerge)
        labelBr = QLabel(" Bright.")
        hboxSpecContr.addWidget(labelBr)
        hboxSpecContr.addWidget(self.brightnessSlider)
        labelCo = QLabel("Contr.")
        hboxSpecContr.addWidget(labelCo)
        hboxSpecContr.addWidget(self.contrastSlider)
        labelVl = QLabel("Vol.")
        hboxSpecContr.addWidget(labelVl)
        hboxSpecContr.addWidget(self.volSlider)
        label1 = QLabel('Adjust clusters if required')
        label1.setFont(QtGui.QFont('SansSerif', 10))

        # top part
        vboxTop = QVBoxLayout()
        vboxTop.addWidget(label1)
        vboxTop.addLayout(hboxSpecContr)
        # must be fixed size!
        vboxTop.setSizeConstraint(QLayout.SetFixedSize)

        # set up the images
        self.flowLayout = pg.LayoutWidget()
        self.flowLayout.setGeometry(QtCore.QRect(0, 0, 380, 247))

        # Add the clusters to rows
        self.addButtons()

        self.scrollArea = QtGui.QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.flowLayout)

        # set overall layout of the dialog
        self.vboxFull = QVBoxLayout()
        # self.vboxFull.setSpacing(0)
        self.vboxFull.addLayout(vboxTop)
        # self.vboxSpacer = QSpacerItem(1, 1, 5, 5)
        # self.vboxFull.addItem(self.vboxSpacer)
        self.vboxFull.addWidget(self.scrollArea)

        self.setLayout(self.vboxFull)

    def merge(self):
        """ Listner for the merge button. Merge the rows (clusters) checked into one cluster.
        """
        # Find which clusters/rows to merge
        tomerge = []
        i = 0
        for cbox in self.cboxes:
            if cbox.checkState() != 0:
                tomerge.append(i)
            i += 1
        # print('rows/clusters to merge are:', tomerge)
        if len(tomerge) < 2:
            return

        # Generate new class labels
        nclasses = self.nclasses - len(tomerge) + 1
        max_label = nclasses - 1
        labels = []
        c = self.nclasses - 1
        while c > -1:
            if c in tomerge:
                labels.append((c, 0))
            else:
                labels.append((c, max_label))
                max_label -= 1
            c -= 1

        self.nclasses = nclasses

        # print('[old, new] labels')
        labels = dict(labels)
        print(labels)

        # update the segments
        for seg in self.segments:
            seg[3] = labels[seg[3]]

        # Clean and redraw
        self.clearButtons()
        # print('cleaned')

        self.updateButtons()
        print('updated')

    def addButtons(self):
        """ Make the buttons and display them
        """
        self.cboxes = []
        for r in range(self.nclasses):
            c = 0
            cbox = QCheckBox("")
            self.cboxes.append(cbox)
            self.flowLayout.addWidget(self.cboxes[-1], r, c)
            c += 1
            # Find the segments under this class, create buttons, and show them
            for seg in self.segments:
                if seg[2] == r:
                    sg, audiodata, audioFormat = self.loadFile(seg[0], seg[1][1]-seg[1][0], seg[1][0])
                    newButton = PicButton(1, np.fliplr(sg), audiodata, audioFormat, seg[1][1]-seg[1][0], 0, seg[1][1], self.lut, self.colourStart,
                                          self.colourEnd, False, cluster=True)
                    seg.insert(2, newButton)
                    self.flowLayout.addWidget(seg[2], r, c)
                    c += 1
            print('*', r, c-1)

    def updateButtons(self):
        """ Update the existing buttons, call when merging clusters
        """
        self.cboxes = []
        for r in range(self.nclasses):
            c = 0
            cbox = QCheckBox("")
            self.cboxes.append(cbox)
            self.flowLayout.addWidget(self.cboxes[-1], r, c)
            c += 1
            # Find the segments under this class and show them
            for seg in self.segments:
                if seg[3] == r:
                    self.flowLayout.addWidget(seg[2], r, c)
                    c += 1
            # print(r, c-1)
        self.flowLayout.update()

    def clearButtons(self):
        """ Remove existing buttons, call when merging clusters
        """
        for ch in self.cboxes:
            ch.hide()
        for btnum in reversed(range(self.flowLayout.layout.count())):
            item = self.flowLayout.layout.itemAt(btnum)
            if item is not None:
                self.flowLayout.layout.removeItem(item)
                r, c = self.flowLayout.items[item.widget()]
                del self.flowLayout.rows[r][c]
        self.flowLayout.update()


    def setColourLevels(self):
        """ Listener for the brightness and contrast sliders being changed. Also called when spectrograms are loaded, etc.
        Translates the brightness and contrast values into appropriate image levels.
        """
        minsg = np.min(self.sg)
        maxsg = np.max(self.sg)
        brightness = self.brightnessSlider.value()
        contrast = self.contrastSlider.value()
        colourStart = (brightness / 100.0 * contrast / 100.0) * (maxsg - minsg) + minsg
        colourEnd = (maxsg - minsg) * (1.0 - contrast / 100.0) + colourStart
        for seg in self.segments:
            seg[2].stopPlayback()
            seg[2].setImage(self.lut, colourStart, colourEnd, False)
            seg[2].update()


    def volSliderMoved(self, value):
        # try/pass to avoid race situations when smth is not initialized
        try:
            for seg in self.segments:
                seg[2].media_obj.applyVolSlider(value)
        except Exception:
            pass


    def loadFile(self, filename, duration=0, offset=0, fs=0):
        """
        Read audio file
        """
        # TODO: Move out of Dialogs
        import wavio
        import librosa
        from PyQt5.QtMultimedia import QAudioFormat

        if offset == 0 and duration == 0:
            wavobj = wavio.read(filename)
        else:
            wavobj = wavio.read(filename, duration, offset)
        sampleRate = wavobj.rate
        audiodata = wavobj.data

        audioFormat = QAudioFormat()
        audioFormat.setCodec("audio/pcm")
        audioFormat.setByteOrder(QAudioFormat.LittleEndian)
        audioFormat.setSampleType(QAudioFormat.SignedInt)

        audioFormat.setChannelCount(np.shape(audiodata)[1])
        audioFormat.setSampleRate(sampleRate)
        audioFormat.setSampleSize(wavobj.sampwidth * 8)

        if audiodata.dtype is not 'float':
            audiodata = audiodata.astype('float')  # / 32768.0
        if np.shape(np.shape(audiodata))[0] > 1:
            audiodata = audiodata[:, 0]

        if fs != 0 and sampleRate != fs:
            audiodata = librosa.core.audio.resample(audiodata, sampleRate, fs)
            sampleRate = fs
            audioFormat.setSampleRate(sampleRate)

        sp = SignalProc.SignalProc()
        sgRaw = sp.spectrogram(audiodata, window_width=512,
                                         incr=256, window='Hann', mean_normalise=True, onesided=True,
                                         multitaper=False, need_even=False)
        maxsg = np.min(sgRaw)
        self.sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))
        self.setColourMap()

        return self.sg, audiodata, audioFormat


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


    def resizeEvent(self, ev):
        """ On this event, choose which (and how many) buttons to display
            from self.buttons. It is also called on initialization.
        """
        pass


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

