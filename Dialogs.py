
# This is part of the AviaNZ interface
# Holds most of the code for the various dialog boxes

# Version 3.4 18/12/24
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2024

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
import os
import shutil

from PyQt6 import QtCore, QtGui
from PyQt6.QtGui import *
from PyQt6.QtWidgets import QLabel, QDialog, QComboBox, QCheckBox, QPushButton, QLineEdit, QSlider, QFileDialog, QHBoxLayout, QVBoxLayout, QFormLayout, QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox, QToolButton, QStyle, QScrollArea # listing some explicitly to make syntax checks lighter
from PyQt6.QtWidgets import *
from PyQt6.QtCore import QPoint, QPointF, QTime, Qt, QSize, pyqtSignal, pyqtSlot, QDir, QTimer

import pyqtgraph as pg

import numpy as np
import colourMaps
import SupportClasses_GUI
import Spectrogram
import SupportClasses
import openpyxl
import json
from scipy.stats import boxcox

import copy

import re

pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class StartScreen(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowIcon(QIcon('img/AviaNZ.ico'))
        self.setWindowTitle('AviaNZ - Choose Task')
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowCloseButtonHint)
        self.setAutoFillBackground(False)
        self.setMinimumSize(860, 350)
        self.setStyleSheet("QDialog {background-image: url(img/AviaNZ_SW_V2.jpg); background-repeat: no-repeat; background-color: #242021; background-position: top center;}")
        self.activateWindow()

        # #242021 for the bgcolor of that image
        # btn_style='QPushButton {background-color: #A3C1DA; color: white; font-size:20px; font-weight: bold; font-family: "Arial"}'
        btn_style=""" QAbstractButton {background-color: #242021;
                    border-color: #b2c8da; border-width:2px; border-style: outset;
                    color: white; font-size:21px; font-weight: bold; font-family: "Arial"; padding: 3px;}
                    QAbstractButton:pressed {border-style: inset;}
                    """
        b1 = QPushButton("   Manual Processing   ")
        b2 = QPushButton("     Batch Processing     ")
        b3 = QPushButton("  Review Batch Results  ")
        b1.setStyleSheet(btn_style)
        b2.setStyleSheet(btn_style)
        b3.setStyleSheet(btn_style)
        bclose = QToolButton()
        bclose.setIcon(QtGui.QIcon('img/close.png'))
        bclose.setIconSize(QSize(40, 40))
        bclose.setToolTip("Close")
        bclose.setStyleSheet(btn_style)
        bclose.clicked.connect(self.reject)

        hboxclose = QHBoxLayout()
        hboxclose.addWidget(bclose, alignment=Qt.AlignmentFlag.AlignRight)

        hbox = QHBoxLayout()
        hbox.addStretch(5)
        hbox.addWidget(b1)
        hbox.addStretch(4)
        hbox.addWidget(b2)
        hbox.addStretch(4)
        hbox.addWidget(b3)
        hbox.addStretch(5)

        vbox = QVBoxLayout()
        vbox.addLayout(hboxclose)
        vbox.addSpacing(180)
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addStretch(1)

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

    def reviewSeg(self):
        self.task = 3
        self.accept()

    #def utilities(self):
        #self.task = 4
        #self.accept()

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
class SpectrogramDialog(QDialog):
    # Class for the spectrogram dialog box
    # TODO: Reproduce the graph from Raven (View/Configure Brightness)
    def __init__(self, width, incr, minFreq, maxFreq, minFreqShow, maxFreqShow, window, sgtype='Standard', sgnorm='Log', sgscale='Linear', nfilters=128, batmode=False, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Spectrogram Options')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(300)

        self.windowType = QComboBox()
        self.windowType.addItems(['Hann','Parzen','Welch','Hamming','Blackman','BlackmanHarris'])
        self.windowType.setCurrentText(window)

        self.sgType = QComboBox()
        self.sgType.addItems(['Standard','Multi-tapered','Reassigned'])
        self.sgType.setCurrentText(sgtype)

        self.sgNorm = QComboBox()
        self.sgNorm.addItems(['Log','Box-Cox','Sigmoid','PCEN'])
        self.sgNorm.setCurrentText(sgnorm)

        self.sgScale = QComboBox()
        self.sgScale.addItems(['Linear','Mel Frequency','Bark Frequency'])
        self.sgScale.setCurrentText(sgscale)

        self.nfilters = QLineEdit(self)
        self.nfilters.setValidator(QIntValidator(8, 256))
        self.nfilters.setText(str(nfilters))

        self.mean_normalise = QCheckBox()
        self.mean_normalise.setChecked(True)

        self.equal_loudness = QCheckBox()
        self.equal_loudness.setChecked(False)

        self.low = QSlider(Qt.Orientation.Horizontal)
        self.low.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.low.setTickInterval(1000)
        self.low.setSingleStep(100)
        self.low.valueChanged.connect(self.lowChange)
        self.lowtext = QLabel()
        self.lowtext.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.lowChange(minFreqShow)

        self.high = QSlider(Qt.Orientation.Horizontal)
        self.high.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.high.setTickInterval(1000)
        self.high.setSingleStep(100)
        self.high.valueChanged.connect(self.highChange)
        self.hightext = QLabel(str(self.high.value()) + ' Hz')
        self.hightext.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.highChange(maxFreqShow)

        self.labelMinF = QLabel()
        self.labelMaxF = QLabel()
        self.labelMaxF.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.setValues(minFreq, maxFreq, minFreqShow, maxFreqShow)
        self.restore = QPushButton("Restore Defaults && Update")
        self.restore.clicked.connect(self.resetValues)
        self.activate = QPushButton("Update Spectrogram")

        self.window_width = QLineEdit(self)
        self.window_width.setValidator(QIntValidator(1, 128000))
        self.window_width.setText(str(width))
        self.incr = QLineEdit(self)
        self.incr.setValidator(QIntValidator(1, 128000))
        self.incr.setText(str(incr))

        Box = QVBoxLayout()
        form = QFormLayout()
        form.addRow('Window', self.windowType)
        form.addRow('Spectrogram type', self.sgType)
        form.addRow('Spectrogram normalisation', self.sgNorm)
        form.addRow('Mean normalise', self.mean_normalise)
        form.addRow('Equal loudness', self.equal_loudness)
        #form.addRow('Multitapering', self.multitaper)
        #form.addRow('Reassignment', self.reassigned)
        form.addRow('Window width', self.window_width)
        form.addRow('Hop', self.incr)
        form.addRow('Frequency scaling', self.sgScale)
        form.addRow('Number of filters', self.nfilters)
        form.setVerticalSpacing(15)

        # Most of the settings can't be changed when using BMPs:
        if batmode:
            for i in range(form.count()):
                form.itemAt(i).widget().setEnabled(False)

        form2 = pg.LayoutWidget()
        form2.addWidget(QLabel('Lowest frequency'), row=0, col=0)
        form2.addWidget(self.lowtext, row=0, col=1)
        form2.addWidget(self.low, row=1, col=0, colspan=2)
        form2.addWidget(QLabel('Highest frequency'), row=2, col=0)
        form2.addWidget(self.hightext, row=2, col=1)
        form2.addWidget(self.high, row=3, col=0, colspan=2)
        form2.addWidget(self.labelMinF, row=4, col=0)
        form2.addWidget(self.labelMaxF, row=4, col=1)

        Box.addLayout(form)
        Box.addSpacing(15)
        Box.addWidget(QLabel('Frequency range to show:'))
        Box.addWidget(form2)

        Box.addWidget(self.activate)
        Box.addWidget(self.restore)

        # Now put everything into the frame
        self.setLayout(Box)

    def setValues(self,minFreq,maxFreq,minFreqShow,maxFreqShow):
        self.low.setRange(minFreq,maxFreq)
        self.low.setValue(minFreqShow)
        self.high.setRange(minFreq,maxFreq)
        self.high.setValue(maxFreqShow)
        self.labelMinF.setText(str(minFreq))
        self.labelMaxF.setText(str(maxFreq))

    def getValues(self):
        if not self.incr.hasAcceptableInput() or not self.window_width.hasAcceptableInput():
            print("ERROR: bad window parameters specified, overriding")
            self.incr.setText('128')
            self.window_width.setText('256')
        low = int(self.low.value() // 100 *100)
        high = int(self.high.value() // 100 *100)
        return [self.windowType.currentText(),self.sgType.currentText(),self.sgNorm.currentText(),self.mean_normalise.isChecked(),self.equal_loudness.isChecked(),int(self.window_width.text()),int(self.incr.text()),int(low),int(high),self.sgScale.currentText(),int(self.nfilters.text())]

    def lowChange(self,value):
        # NOTE returned values should also use this rounding
        value = value // 100 * 100
        self.lowtext.setText(str(value)+' Hz')

    def highChange(self,value):
        value = value // 100 * 100
        self.hightext.setText(str(value)+' Hz')

    def resetValues(self):
        self.windowType.setCurrentText('Hann')
        self.sgType.setCurrentText('Standard')
        self.sgNorm.setCurrentText('Log')
        self.mean_normalise.setChecked(True)
        self.equal_loudness.setChecked(False)
        self.setValues(self.low.minimum(), self.low.maximum(), self.low.minimum(), self.high.maximum())
        self.window_width.setText('256')
        self.incr.setText('128')
        self.sgScale.setCurrentText('Linear')
        self.nfilters.setText('128')
        
        self.activate.clicked.emit()

    # def closeEvent(self, event):
    #     msg = QMessageBox()
    #     msg.setIcon(QMessageBox.Question)
    #     msg.setText("Do you want to keep the new values?")
    #     msg.setWindowTitle("Closing Spectrogram Dialog")
    #     msg.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
    #     msg.buttonClicked.connect(self.resetValues)
    #     msg.exec_()
    #     return

#======
class Excel2Annotation(QDialog):
    # Class for Excel to AviaNZ annotation
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Generate annotations from Excel')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(700)

        self.txtExcel = QLineEdit()
        self.txtExcel.setMinimumWidth(400)
        self.txtExcel.setText('')
        self.btnBrowseExcel = QPushButton("&Choose Excel file")
        self.btnBrowseExcel.setFixedWidth(220)
        self.btnBrowseExcel.clicked.connect(self.browseExcel)

        lblHeader = QLabel('Choose Columns')
        lblHeader.setFixedWidth(220)
        lblHeader.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.comboStart = QComboBox()
        self.comboEnd = QComboBox()
        self.comboLow = QComboBox()
        self.comboHigh = QComboBox()

        self.txtAudio = QLineEdit()
        self.txtAudio.setMinimumWidth(400)
        self.txtAudio.setText('')
        self.btnBrowseAudio = QPushButton("Choose Corresponding Audio")
        self.btnBrowseAudio.setFixedWidth(220)
        self.btnBrowseAudio.setToolTip("Select corresponding .wav")
        self.btnBrowseAudio.clicked.connect(self.browseAudio)

        self.txtSpecies = QLineEdit()
        self.txtSpecies.setMinimumWidth(400)
        self.txtSpecies.setText('')
        lblSpecies = QLabel("Species Name")
        lblSpecies.setFixedWidth(220)
        lblSpecies.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.txtCalltype = QLineEdit()
        self.txtCalltype.setMinimumWidth(400)
        self.txtCalltype.setText('')
        lblCalltype = QLabel("Calltype (optional)")
        lblCalltype.setFixedWidth(220)
        lblCalltype.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btnGenerateAnnot = SupportClasses_GUI.MainPushButton("Generate AviaNZ Annotation")

        # Show a template
        tableWidget = QTableWidget()
        tableWidget.setRowCount(4)
        tableWidget.setColumnCount(4)
        tableWidget.setHorizontalHeaderLabels("A;B;C;D".split(";"))
        tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tableWidget.setItem(0, 0, QTableWidgetItem("Start time (sec)"))
        tableWidget.setItem(0, 1, QTableWidgetItem("End time (sec)"))
        tableWidget.setItem(0, 2, QTableWidgetItem("Lower frequency (Hz)"))
        tableWidget.setItem(0, 3, QTableWidgetItem("Upper frequency (Hz)"))
        tableWidget.setItem(1, 0, QTableWidgetItem("42.15"))
        tableWidget.setItem(1, 1, QTableWidgetItem("48.24"))
        tableWidget.setItem(1, 2, QTableWidgetItem("546.26"))
        tableWidget.setItem(1, 3, QTableWidgetItem("7492.35"))
        tableWidget.setItem(2, 0, QTableWidgetItem("88.54"))
        tableWidget.setItem(2, 1, QTableWidgetItem("95.25"))
        tableWidget.setItem(2, 2, QTableWidgetItem("550.74"))
        tableWidget.setItem(2, 3, QTableWidgetItem("7505.25"))
        tableWidget.setItem(3, 0, QTableWidgetItem("684.15"))
        tableWidget.setItem(3, 1, QTableWidgetItem("699.74"))
        tableWidget.setItem(3, 2, QTableWidgetItem("560.25"))
        tableWidget.setItem(3, 3, QTableWidgetItem("8000.30"))
        tableWidget.setMinimumWidth(700)
        tableWidget.setStyleSheet("QTableWidget { color : #808080; }")
        tableWidget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        Box = QVBoxLayout()
        Box.addWidget(QLabel('Sample Excel:'))
        Box.addWidget(tableWidget)
        Box.addWidget(QLabel())
        Box1 = QHBoxLayout()
        Box1.addWidget(self.btnBrowseExcel)
        Box1.addWidget(self.txtExcel)
        Box10 = QHBoxLayout()
        Box11 = QVBoxLayout()
        Box11.addWidget(QLabel('Start time'))
        Box11.addWidget(self.comboStart)
        Box12 = QVBoxLayout()
        Box12.addWidget(QLabel('End time'))
        Box12.addWidget(self.comboEnd)
        Box13 = QVBoxLayout()
        Box13.addWidget(QLabel('Lower frequency'))
        Box13.addWidget(self.comboLow)
        Box14 = QVBoxLayout()
        Box14.addWidget(QLabel('Higher frequency'))
        Box14.addWidget(self.comboHigh)
        Box10.addWidget(lblHeader)
        Box10.addLayout(Box11)
        Box10.addLayout(Box12)
        Box10.addLayout(Box13)
        Box10.addLayout(Box14)

        Box2 = QHBoxLayout()
        Box2.addWidget(self.btnBrowseAudio)
        Box2.addWidget(self.txtAudio)
        Box3 = QHBoxLayout()
        Box3.addWidget(lblSpecies)
        Box3.addWidget(self.txtSpecies)
        Box3.addWidget(lblCalltype)
        Box3.addWidget(self.txtCalltype)
        Box.addLayout(Box1)
        Box.addLayout(Box10)
        Box.addLayout(Box2)
        Box.addLayout(Box3)
        Box.addWidget(QLabel())
        Box.addWidget(self.btnGenerateAnnot)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        if self.txtSpecies.text() and self.txtExcel.text() and self.txtAudio.text():
            return [self.txtExcel.text(), self.txtAudio.text(), self.txtSpecies.text(), self.txtCalltype.text(), self.headers[self.comboStart.currentIndex()], self.headers[self.comboEnd.currentIndex()], self.headers[self.comboLow.currentIndex()], self.headers[self.comboHigh.currentIndex()]]
        else:
            msg = SupportClasses_GUI.MessagePopup("t", "All fields are required ", "All fields are required ")
            msg.exec()
            return []

    def browseExcel(self):
        try:
            if not self.txtAudio.text():
                userDir = os.path.expanduser("~")
            else:
                userDir, _ = os.path.split(self.txtAudio.text())
            excelfile, _ = QFileDialog.getOpenFileName(self, 'Open file', userDir, "Excel (*.xlsx *.xls)")
            self.txtExcel.setText(excelfile)
            self.txtExcel.setReadOnly(True)
            # Read the excel to get the headers
            book = openpyxl.load_workbook(excelfile)
            sheet = book.active
            headers = [value for value in sheet.iter_rows(min_row=1, max_row=1)]
            headers = [h for h in headers[0]]
            self.headers = [h.column for h in headers]
            values = [h.value for h in headers]
            self.comboStart.addItems(values)
            self.comboStart.setCurrentText(values[0])
            self.comboEnd.addItems(values)
            self.comboEnd.setCurrentText(values[1])
            self.comboLow.addItems(values)
            self.comboLow.setCurrentText(values[2])
            self.comboHigh.addItems(values)
            self.comboHigh.setCurrentText(values[3])
        except Exception as e:
            print("ERROR: failed with error:")
            print(e)
            return

    def browseAudio(self):
        try:
            if not self.txtExcel.text():
                userDir = os.path.expanduser("~")
            else:
                userDir, _ = os.path.split(self.txtExcel.text())
            audiofile, _ = QFileDialog.getOpenFileName(self, 'Open file', userDir, "Audio (*.wav)")
            self.txtAudio.setText(audiofile)
            self.txtAudio.setReadOnly(True)
        except Exception as e:
            print("ERROR: failed with error:")
            print(e)
            return

#======
class Tag2Annotation(QDialog):
    # Class for XML Tag to AviaNZ annotation
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Generate annotations from XML (Freebird)')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(700)

        self.txtSession = QLineEdit()
        self.txtSession.setMinimumWidth(400)
        self.txtSession.setText('')
        self.btnBrowseSession = QPushButton("&Choose Session")
        self.btnBrowseSession.setFixedWidth(220)
        self.btnBrowseSession.clicked.connect(self.browseSession)

        #self.txtDuration = QLineEdit()
        #self.txtDuration.setMinimumWidth(400)
        #self.txtDuration.setText('')
        #lblDuration = QLabel("Duration (sec) of a recording")
        #lblDuration.setFixedWidth(220)
        #lblDuration.setAlignment(Qt.AlignCenter)

        self.btnGenerateAnnot = SupportClasses_GUI.MainPushButton("Generate AviaNZ Annotation")

        Box = QVBoxLayout()
        Box.addWidget(QLabel())
        Box1 = QHBoxLayout()
        Box1.addWidget(self.btnBrowseSession)
        Box1.addWidget(self.txtSession)
        #Box2 = QHBoxLayout()
        #Box2.addWidget(lblDuration)
        #Box2.addWidget(self.txtDuration)
        Box.addLayout(Box1)
        #Box.addLayout(Box2)
        Box.addWidget(QLabel())
        Box.addWidget(self.btnGenerateAnnot)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        if self.txtSession.text(): # and self.txtDuration.text():
            return self.txtSession.text()
            #return [self.txtSession.text(), self.txtDuration.text()]
        else:
            msg = SupportClasses_GUI.MessagePopup("w", "Folder ", "Need a folder of Freebird tags.")
            msg.exec()
            return []

    def browseSession(self):
        #dirName = QFileDialog.getExistingDirectory(self, 'Choose .session folder with .tag and .setting')
        d = QFileDialog(self)
        d.setFilter(QDir.Filter.AllDirs | QDir.Filter.Hidden | QDir.Filter.NoDotAndDotDot)
        d.setFileMode(QFileDialog.FileMode.Directory)
        if(d.exec()):
            dirName = d.selectedFiles()[0]
            self.txtSession.setText(dirName)

#======
class BackupAnnotation(QDialog):
    # Backup AviaNZ annotations into another folder. Simply copies all of the .data files with directory hierarchy preserved
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Backup annotations')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(700)

        self.txtSrc = QLineEdit()
        self.txtSrc.setMinimumWidth(400)
        self.txtSrc.setText('')
        self.btnBrowseSrc = QPushButton("&Choose Source Directory")
        self.btnBrowseSrc.setFixedWidth(220)
        self.btnBrowseSrc.clicked.connect(self.browseSrc)

        self.txtDst = QLineEdit()
        self.txtDst.setMinimumWidth(400)
        self.txtDst.setText('')
        self.btnBrowseDst = QPushButton("&Choose Destination Directory")
        self.btnBrowseDst.setFixedWidth(220)
        self.btnBrowseDst.clicked.connect(self.browseDst)

        self.btnCopyAnnot = SupportClasses_GUI.MainPushButton("Copy Annotations")

        Box = QVBoxLayout()
        Box.addWidget(QLabel('This allows you to get a copy of your annotations while preserving the directory hierarchy, only copy the .data files.\nSelect the directory you want to backup the annotations from and create a destination directory to copy the annotations'))
        Box.addWidget(QLabel())
        Box1 = QHBoxLayout()
        Box1.addWidget(self.btnBrowseSrc)
        Box1.addWidget(self.txtSrc)
        Box2 = QHBoxLayout()
        Box2.addWidget(self.btnBrowseDst)
        Box2.addWidget(self.txtDst)
        Box.addLayout(Box1)
        Box.addLayout(Box2)
        Box.addWidget(QLabel())
        Box.addWidget(self.btnCopyAnnot)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        if self.txtSrc.text() and self.txtDst.text():
            return [self.txtSrc.text(), self.txtDst.text()]
        else:
            msg = SupportClasses_GUI.MessagePopup("t", "Need both source and target directories", "Need both source and target directories")
            msg.exec()
            return []

    def browseSrc(self):
        dirName = QFileDialog.getExistingDirectory(self, 'Choose the source folder to backup')
        self.txtSrc.setText(dirName)

    def browseDst(self):
        dirName = QFileDialog.getExistingDirectory(self, 'Choose the destination folder')
        self.txtDst.setText(dirName)

class OperatorReviewer(QDialog):
    # Class for the set operator dialog box
    def __init__(self, operator='', reviewer='', parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Set Operator/Reviewer')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint))
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
        return [self.name1.text(),self.name2.text()]

#======
class addNoiseData(QDialog):
    # Class for the noise data dialog box
    def __init__(self, noiseLevel, noiseTypes, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Noise Information')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint))
        self.setMinimumWidth(320)

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
            self.btnMed.setChecked(True)
        HBox1.addWidget(self.btnMed)
        self.btnHigh = QRadioButton('High')
        self.level.addButton(self.btnHigh)
        if noiseLevel == 'High':
            self.btnHigh.setChecked(True)
        HBox1.addWidget(self.btnHigh)
        self.btnTerrible = QRadioButton('Terrible')
        self.level.addButton(self.btnTerrible)
        HBox1.addWidget(self.btnTerrible)
        if noiseLevel == 'Terrible':
            self.btnTerrible.setChecked(True)

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
        Box.setAlignment(Qt.AlignmentFlag.AlignTop)
        Box.addLayout(HBox1)
        Box.addLayout(HBox2)

        # Now put everything into the frame
        self.setLayout(Box)

    def getNoiseData(self):
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
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)

        # species / filter
        self.filterLabel = QLabel("Select recogniser to use")
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

class DiagnosticNN(QDialog):
    # Class for the diagnostic dialog box - NN
    def __init__(self, filters, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('NN Diagnostic Plot Options')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setMinimumWidth(300)
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)

        # species / filter
        self.filterLabel = QLabel("Select recogniser to use")
        self.filter = QComboBox()
        # add filter file names to combobox
        self.filter.addItems(list(filters.keys()))

        # select call types to plot
        self.ctbox = QHBoxLayout()
        self.chkboxes = []

        # buttons
        self.activate = QPushButton("Make plots")
        self.clear = QPushButton("Clear plots")

        # layout
        Box = QVBoxLayout()
        Box.addWidget(self.filterLabel)
        Box.addWidget(self.filter)
        Box.addLayout(self.ctbox)

        Box.addWidget(self.activate)
        Box.addWidget(self.clear)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        return [self.filter.currentText(), [cb.isChecked() for cb in self.chkboxes]]

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
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(350)

        self.algs = QComboBox()
        if DOC:
            self.algs.addItems(["WV Changepoint", "Wavelet Filter", "FIR", "Median Clipping"])
        else:
            self.algs.addItems(["Default","Median Clipping","Fundamental Frequency","FIR","Harma","Power","WV Changepoint","Wavelet Filter","Cross-Correlation"])
        self.algs.currentTextChanged.connect(self.changeBoxes)
        #self.algs.currentIndexChanged.connect(self.changeBoxes)
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
        self.medThr.setDecimals(2)
        self.medThr.setValue(3)

        # set min seg size for median clipping
        self.medSize = QSlider(Qt.Orientation.Horizontal)
        self.medSize.setTickPosition(QSlider.TickPosition.TicksBelow)
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

        self.FIRThr1text = QLabel("Set threshold")
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
        Box.addStretch(1)

        # Labels
        #self.amplabel = QLabel("Set threshold amplitude")
        #Box.addWidget(self.amplabel)

        self.Harmalabel = QLabel("Set decibel threshold")
        Box.addWidget(self.Harmalabel)

        #self.Onsetslabel = QLabel("Onsets: No parameters")
        #Box.addWidget(self.Onsetslabel)

        self.medlabel = QLabel("Set median threshold")
        self.medlabel.show()

        self.eclabel = QLabel("Set energy curve threshold")
        self.ecthrtype = [QRadioButton("N standard deviations"), QRadioButton("Threshold")]

        self.specieslabel = QLabel("Species")
        self.species_wv = QComboBox()
        self.species_chp = QComboBox()

        self.specieslabel_cc = QLabel("Species")
        self.species_cc = QComboBox()
        self.species_cc.addItems(["Choose species...", "Bittern"])

        # parse the provided filter list into wavelet and changepoint boxes
        self.filters = species
        spp_chp = []
        spp_wv = []
        for key, item in species.items():
            if item.get("method")=="chp":
                spp_chp.append(key)
            else:
                spp_wv.append(key)
        spp_chp = sorted(spp_chp)
        spp_wv = sorted(spp_wv)
        spp_chp.insert(0, "Choose species...")
        spp_wv.insert(0, "Choose species...")
        self.species_wv.addItems(spp_wv)
        self.species_chp.addItems(spp_chp)
        self.species_chp.currentTextChanged.connect(self.filterChange)

        Box.addWidget(self.specieslabel)
        Box.addWidget(self.species_wv)
        Box.addWidget(self.species_chp)

        Box.addWidget(self.specieslabel_cc)
        Box.addWidget(self.species_cc)

        Box.addWidget(self.HarmaThr1)
        Box.addWidget(self.HarmaThr2)
        Box.addWidget(self.PowerThr)

        Box.addWidget(self.medlabel)
        Box.addWidget(self.medThr)
        Box.addWidget(self.medSizeText)
        Box.addWidget(self.medSize)

        Box.addWidget(self.eclabel)
        for i in range(len(self.ecthrtype)):
            Box.addWidget(self.ecthrtype[i])
        Box.addWidget(self.ecThr)

        Box.addWidget(self.FIRThr1text)
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

        self.medSize = QSlider(Qt.Orientation.Horizontal)
        self.medSize.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.medSize.setTickInterval(100)
        self.medSize.setRange(100,2000)
        self.medSize.setSingleStep(100)
        self.medSize.setValue(1000)
        self.medSize.valueChanged.connect(self.medSizeChange)

        # Parameter selectors for changepoint methods
        self.chpalpha = QDoubleSpinBox()
        self.chpalpha.setRange(0.1, 20)
        self.chpalpha.setValue(3)

        self.chpwin = QDoubleSpinBox()
        self.chpwin.setDecimals(3)
        self.chpwin.setRange(0.005, 3)
        self.chpwin.setValue(0.5)

        self.maxlen = QDoubleSpinBox()
        self.maxlen.setDecimals(3)
        self.maxlen.setRange(0.05, 100)
        self.maxlen.setValue(10)

        # Sliders for minlen and maxgap are in ms scale
        self.minlen = QSlider(Qt.Orientation.Horizontal)
        self.minlen.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.minlen.setTickInterval(500)
        self.minlen.setRange(100, 10000)
        self.minlen.setSingleStep(100)
        self.minlen.setValue(500)
        self.minlen.valueChanged.connect(self.minLenChange)
        self.minlenlbl = QLabel("Minimum segment length: 0.5 sec")

        self.maxgap = QSlider(Qt.Orientation.Horizontal)
        self.maxgap.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.maxgap.setTickInterval(250)
        self.maxgap.setRange(50, 4000)
        self.maxgap.setSingleStep(50)
        self.maxgap.setValue(1000)
        self.maxgap.valueChanged.connect(self.maxGapChange)
        self.maxgaplbl = QLabel("Maximum gap between syllables: 1 sec")

        self.wind = QComboBox()
        self.wind.addItems(["OLS wind filter (recommended)", "Robust wind filter (experimental, slow)", "None"])

        self.chpLayout = QFormLayout()
        self.chpLayout.addRow("Threshold:", self.chpalpha)
        self.chpLayout.addRow("Window size (s):", self.chpwin)
        self.chpLayout.addRow("Max length (s):", self.maxlen)
        self.chpLayout.addRow("Wind denoising:", self.wind)

        self.rain = QCheckBox("Remove rain")
        Box.addWidget(self.rain)
        Box.addWidget(self.maxgaplbl)
        Box.addWidget(self.maxgap)
        Box.addWidget(self.minlenlbl)
        Box.addWidget(self.minlen)
        Box.addLayout(self.chpLayout)
        Box.addStretch(2)
        Box.addWidget(self.undo)
        self.undo.setEnabled(False)
        Box.addWidget(self.activate)
        #Box.addWidget(self.save)

        # Now put everything into the frame,
        # hide and reopen the default
        self.setLayout(Box)
        self.hideAll()
        self.algs.show()
        self.undo.show()
        self.activate.show()
        if DOC:
            self.changeBoxes("WV Changepoint")
        else:
            self.changeBoxes("Default")

    def hideAll(self):
        for w in range(self.layout().count()):
            item = self.layout().itemAt(w)
            if item.widget() is not None:
                # it is a widget
                item.widget().hide()
            elif item.layout() is not None:
                # it is a layout, so loop again:
                for ww in range(item.layout().count()):
                    item.layout().itemAt(ww).widget().hide()
            # pure items (stretch/spacer) get skipped


    def changeBoxes(self,alg):
        # This does the hiding and showing of the options as the algorithm changes
        self.hideAll()
        self.algs.show()
        # self.rain.show()
        self.minlenlbl.show()
        self.minlen.show()
        self.maxgaplbl.show()
        self.maxgap.show()
        self.undo.show()
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
            # self.medSize.show()
            # self.medSizeText.show()
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
            self.FIRThr1text.show()
            self.FIRThr1.show()
        elif alg == "Cross-Correlation":
            self.CCThr1.show()
            self.specieslabel_cc.show()
            self.species_cc.show()
        else:
            #"Wavelet Filter" or "WV Changepoint"
            self.specieslabel.show()
            if alg == "WV Changepoint":
                for ww in range(self.chpLayout.count()):
                    self.chpLayout.itemAt(ww).widget().show()
                self.species_chp.show()
            else:
                self.species_wv.show()
            self.maxgaplbl.hide()
            self.maxgap.hide()
            self.minlenlbl.hide()
            self.minlen.hide()

    def filterChange(self, species):
        """ Override UI with parameters from the requested filter. """
        subfilt = self.filters[species]["Filters"][0]
        self.chpalpha.setValue(subfilt["WaveletParams"]["thr"])
        self.medThr.setValue(subfilt["WaveletParams"]["thr"])
        self.chpwin.setValue(subfilt["WaveletParams"]["win"])
        self.maxlen.setValue(subfilt["TimeRange"][1])

    def medSizeChange(self,value):
        self.medSizeText.setText("Minimum length: %s ms" % value)

    def minLenChange(self,value):
        self.minlenlbl.setText("Minimum segment length: %s sec" % str(round(int(value)/1000, 2)))

    def maxGapChange(self,value):
        self.maxgaplbl.setText("Maximum gap between syllables: %s sec" % str(round(int(value)/1000, 2)))

    def getValues(self):
        # TODO: check: self.medSize.value() is not used, should we keep it?
        alg = self.algs.currentText()
        if alg=="Wavelet Filter":
            filtname = self.species_wv.currentText()
        elif alg=="WV Changepoint":
            filtname = self.species_chp.currentText()
        elif alg=="Cross-Correlation":
            filtname = self.species_cc.currentText()
        else:
            filtname = None
        settings = {"medThr": self.medThr.value(), "medSize": self.medSize.value(), "HarmaThr1": self.HarmaThr1.text(), "HarmaThr2": self.HarmaThr2.text(), "PowerThr": self.PowerThr.text(),
                    "FFminfreq": self.Fundminfreq.text(), "FFminperiods": self.Fundminperiods.text(), "Yinthr": self.Fundthr.text(), "FFwindow": self.Fundwindow.text(), "FIRThr1": self.FIRThr1.text(),
                    "CCThr1": self.CCThr1.text(), "filtname": filtname, "rain": self.rain.isChecked(),
                    "maxgap": int(self.maxgap.value())/1000, "minlen": int(self.minlen.value())/1000, "chpalpha": self.chpalpha.value(), "chpwindow": self.chpwin.value(), "maxlen": self.maxlen.value(),
                    "wind": self.wind.currentText()}
        return(alg, settings)

#======
class Denoise(QDialog):
    # Class for the denoising dialog box
    def __init__(self, parent=None,DOC=True,minFreq=0,maxFreq=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Denoising Options')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)

        self.setMinimumWidth(300)
        self.setMinimumHeight(250)
        self.DOC=DOC
        self.minFreq=minFreq
        self.maxFreq=maxFreq

        self.algs = QComboBox()
        # self.algs.addItems(["Wavelets","Bandpass","Butterworth Bandpass" ,"Wavelets --> Bandpass","Bandpass --> Wavelets","Median Filter"])
        if not self.DOC:
            self.alg_items = ["Wavelets", "Bandpass", "Butterworth Bandpass", "Median Filter"]
        else:
            self.alg_items = ["Wavelets", "Bandpass", "Butterworth Bandpass"]
        self.algs.addItems(self.alg_items)
        self.algs.currentIndexChanged.connect(self.changeBoxes)
        self.prevAlg = "Wavelets"

        # Wavelet: Depth of tree, threshold type, threshold multiplier, wavelet
        if not self.DOC:
            self.noiseest = QComboBox()
            self.noiseest.addItems(["Constant SD", "SD estimated by OLS fit", "SD estimated by QR fit"])

            self.depthlabel = QLabel("Depth of wavelet packet decomposition (or tick box to use best)")
            self.depthchoice = QCheckBox()
            self.depthchoice.clicked.connect(self.depthclicked)
            self.depth = QSpinBox()
            self.depth.setRange(1,12)
            self.depth.setSingleStep(1)
            self.depth.setValue(5)

            self.thrtypelabel = QLabel("Type of thresholding")
            self.thrtype = [QRadioButton("Soft"), QRadioButton("Hard")]
            self.thrtype[0].setChecked(True)

            self.thrlabel = QLabel("Multiplier of SD for threshold")
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
        self.low = QSlider(Qt.Orientation.Horizontal)
        self.low.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.low.setTickInterval(1000)
        self.low.setRange(minFreq,maxFreq)
        self.low.setSingleStep(100)
        self.low.setValue(minFreq)
        self.low.valueChanged.connect(self.lowChange)
        self.lowtext = QLabel(str(self.low.value()))
        self.high = QSlider(Qt.Orientation.Horizontal)
        self.high.setTickPosition(QSlider.TickPosition.TicksBelow)
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
        Box = QVBoxLayout()
        Box.addWidget(self.algs)

        if not self.DOC:
            Box.addWidget(self.noiseest)
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

    def changeBoxes(self,algIndex):
        alg = self.alg_items[algIndex]
        print("changing from", self.prevAlg, " to", alg)
        # This does the hiding and showing of the options as the algorithm changes
        if self.prevAlg == "Wavelets" and not self.DOC:
            self.noiseest.hide()
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
        elif self.prevAlg == "Bandpass" or self.prevAlg == "Butterworth Bandpass":
            self.bandlabel.hide()
            self.blabel.hide()
            self.low.hide()
            self.lowtext.hide()
            self.high.hide()
            self.hightext.hide()
        else:
            # Median filter
            self.medlabel.hide()
            self.widthlabel.hide()
            self.width.hide()

        self.prevAlg = str(alg)
        if str(alg) == "Wavelets" and not self.DOC:
            self.noiseest.show()
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

            if self.noiseest.currentText()=="Constant SD":
                noiseest = "const"
            elif self.noiseest.currentText()=="SD estimated by OLS fit":
                noiseest = "ols"
            elif self.noiseest.currentText()=="SD estimated by QR fit":
                noiseest = "qr"
            return [self.algs.currentText(), depth, thrType, self.thr.text(),self.wavelet.currentText(),self.low.value(),self.high.value(),self.width.text(), self.aabox1.isChecked(), self.aabox2.isChecked(), noiseest]
        else:
            return [self.algs.currentText(),self.low.value(),self.high.value(),self.width.text()]#,self.trimaxis.isChecked()]

    def lowChange(self,value):
        self.lowtext.setText(str(value))

    def highChange(self,value):
        self.hightext.setText(str(value))


#======
# The list of less common birds
class SearchableBirdWidget(QWidget):
    add_requested = pyqtSignal(str)

    def __init__(self, allow_multi, parent=None):
        super(SearchableBirdWidget, self).__init__(parent)
        self.header = QLabel("Type to search")
        self.addSpBtn = QPushButton(QIcon('img/add.png'), " Add new")
        self.addSpBtn.setIconSize(QSize(14,14))
        self.addSpBtn.clicked.connect(self.clickedAdd)
        self.addSpBtn.setEnabled(False)
        self.searchline = QLineEdit()
        self.searchline.textChanged.connect(self.searched)
        self.fulllist = QListWidget()

        if allow_multi:
            self.fulllist.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        else:
            self.fulllist.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # layout
        self.headerBox = QHBoxLayout()
        self.headerBox.addWidget(self.header)
        self.headerBox.addWidget(self.addSpBtn)
        self.headerBox.setStretch(0,3)
        self.headerBox.setStretch(1,1)
        self.layout = QVBoxLayout()
        self.layout.addLayout(self.headerBox)
        self.layout.addWidget(self.searchline)
        self.layout.addWidget(self.fulllist)
        self.setLayout(self.layout)

    def searched(self, text):
        # will scroll to the first item matching TEXT
        if text=="":
            self.addSpBtn.setEnabled(False)
            return
        self.addSpBtn.setEnabled(True)
        hit = self.fulllist.findItems(text, Qt.MatchFlag.MatchContains)
        if len(hit)>0:
            self.fulllist.scrollToItem(hit[0], QAbstractItemView.SelectionMode.PositionAtTop)
            # also check if the exact species is already present
            exacthit = self.fulllist.findItems(text, Qt.MatchFlag.MatchFixedString)
            if len(exacthit)==1:
                self.addSpBtn.setEnabled(False)

    def addBird(self, sp):
        self.fulllist.addItem(sp)

    def clearSelection(self):
        self.fulllist.clearSelection()

    def selectBird(self, index):
        # Takes index (int) or label (str).
        if type(index) is str:
            item = self.fulllist.findItems(index, Qt.MatchFlag.MatchExactly)
            if len(item)!=1:
                print("Warning: could not find bird", index)
                return
            item = item[0]
        else:
            item = self.fulllist.item(index)
        item.setSelected(not item.isSelected())

    def clickedAdd(self):
        # Listener for the text entry in the bird list
        # Check text isn't already in the listbox, and add if not
        # Then calls the usual handler for listbox selections
        species = self.searchline.text()
        msg = SupportClasses_GUI.MessagePopup("t", "Adding new species", 'Species "%s" will be added to the full bird list. Are you sure?' % species)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if msg.exec()==QMessageBox.StandardButton.Yes:
            if species.lower()=="don't know" or species.lower()=="other":
                print("ERROR: provided name %s is reserved, cannot create" % species)
                return
            if "?" in species:
                print("ERROR: provided name %s contains reserved symbol '?'" % species)
                return
            if len(species)==0 or len(species)>150:
                print("ERROR: provided name appears to be too short or too long")
                return

            print("Species name appears OK, will add")
            self.addBird(species)
            self.selectBird(species)
            newitem = self.fulllist.item(self.fulllist.count()-1)  # replace this w/ findItems if needed
            self.fulllist.scrollToItem(newitem)
            self.searchline.clear()
            # this will deal with updating the label and buttons
            self.fulllist.itemClicked.emit(newitem)
            self.add_requested.emit(species)

class HumanClassify1(QDialog):
    # This dialog allows the checking of classifications for segments.
    # It shows a single segment at a time, working through all the segments.

    def __init__(self, lut, cmapInverted, brightness, contrast, shortBirdList, longBirdList, knownCalls, batList, multipleBirds, audioFormat, guidecols, plotAspect=2, loop=False, autoplay=False, parent=None, reorderShortList=False):
        # plotAspect: initial stretch factor in the X direction
        QDialog.__init__(self, parent)
        self.setWindowTitle('Check Classifications')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint)

        self.setModal(True)
        self.frame = QWidget()

        self.parent = parent
        self.batmode = self.parent.batmode
        self.lut = lut
        self.label = []
        self.cmapInverted = cmapInverted
        self.shortBirdList = shortBirdList
        self.longBirdList = longBirdList
        self.knownCalls = knownCalls
        self.batList = batList
        self.multipleBirds = multipleBirds
        self.saveBirdList = False
        self.viewingct = False
        self.reorderShortList = reorderShortList
        self.needToSaveConfig = False
        # exec_ forces the cursor into waiting

        # Set up the plot window, then the right and wrong buttons, and a close button
        # wPlot: white area around the spectrogram
        self.wPlot = SupportClasses_GUI.PartlyResizableGLW()
        self.pPlot = self.wPlot.addViewBox(enableMouse=False, row=0, col=1)
        self.plot = pg.ImageItem()

        # TODO: Useful?
        #self.blurEffect = QGraphicsBlurEffect(blurRadius=1.1)
        #self.plot.setGraphicsEffect(self.blurEffect)

        self.pPlot.addItem(self.plot)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.wPlot)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(270)

        # Fix the aspect ratio to a preset number. Initial view box
        # will be about 2:1, so aspect ratio of 2 means
        # that a square spectrogram (e.g. 512x512) will fill it
        self.plotAspect = plotAspect
        self.wPlot.setMinimumHeight(250)
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

        # prepare guides for marking true segment boundaries

        self.guidelines = [0]*len(guidecols)
        for gi in range(len(guidecols)):
            self.guidelines[gi] = pg.InfiniteLine(angle=0, pen={'color': guidecols[gi], 'width': 2})
            self.pPlot.addItem(self.guidelines[gi])

        # time texts to go along these two lines
        self.segTimeText1 = pg.TextItem(color=(50,205,50), anchor=(0,1.10))
        self.segTimeText2 = pg.TextItem(color=(50,205,50), anchor=(0,0.75))
        self.pPlot.addItem(self.segTimeText1)
        self.pPlot.addItem(self.segTimeText2)

        # playback line
        self.bar = pg.InfiniteLine(angle=90, movable=False, pen={'color':'c', 'width': 3})
        self.bar.btn = Qt.MouseButton.RightButton
        self.bar.setValue(0)
        self.pPlot.addItem(self.bar)

        # label for current segment assignment
        self.speciesTop = QLabel("Currently:")
        self.species = QLabel()
        self.species.setStyleSheet("QLabel { font-size:22pt; font-weight: bold}")

        # The buttons to move through the overview
        self.numberDone = QLabel()
        self.numberLeft = QLabel()
        self.numberDone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.numberLeft.setAlignment(Qt.AlignmentFlag.AlignCenter)

        iconSize = QSize(45, 45)
        self.buttonPrev = QToolButton()
        self.buttonPrev.setIcon(QtGui.QIcon('img/undo.png'))
        self.buttonPrev.setIconSize(iconSize)
        self.buttonPrev.setStyleSheet("padding: 5px 5px 5px 5px")

        self.buttonNext = QToolButton()
        self.buttonNext.setIcon(QtGui.QIcon('img/questionL.png'))
        self.buttonNext.setIconSize(iconSize)
        self.buttonNext.setStyleSheet("padding: 5px 5px 5px 5px")

        self.correct = QToolButton()
        self.correct.setIcon(QtGui.QIcon('img/check-mark2.png'))
        self.correct.setIconSize(iconSize)
        self.correct.setStyleSheet("padding: 5px 5px 5px 5px")

        self.delete = QToolButton()
        self.delete.setIcon(QtGui.QIcon('img/deleteL.png'))
        self.delete.setIconSize(iconSize)
        self.delete.setStyleSheet("padding: 5px 5px 5px 5px")

        # TODO: Icon
        self.buttonPlus = QToolButton()
        self.buttonPlus.setIcon(QtGui.QIcon('img/add.png'))
        self.buttonPlus.setIconSize(iconSize)
        #self.buttonPlus.setIcon(QtGui.QIcon('img/iconplus.png'))
        #self.buttonPlus.setText('+')
        self.buttonPlus.setStyleSheet("padding: 5px 5px 5px 5px")

        self.ctLabel = QLabel("")
        self.ctLabel.setStyleSheet("QLabel { font-size:16pt; font-weight: bold}")
        self.ctLabel.hide()

        # Audio playback object
        self.media_obj = SupportClasses_GUI.ControllableAudio(sp=None,audioFormat=audioFormat,useBar=True)
        # TODO: make own timer
        self.NotifyTimer = QTimer(self)
        self.NotifyTimer.timeout.connect(self.movePlaySlider)
        #self.media_obj.NotifyTimer.timeout.connect(self.movePlaySlider)
        #self.media_obj.NotifyTimer.timeout.connect(self.endListener)
        self.media_obj.loop = loop
        self.autoplay = autoplay

        # The layouts
        hboxNextPrev = QHBoxLayout()
        hboxNextPrev.addWidget(self.numberDone)
        hboxNextPrev.addWidget(self.buttonPrev)
        hboxNextPrev.addWidget(self.correct)
        hboxNextPrev.addWidget(self.buttonNext)
        hboxNextPrev.addWidget(self.delete)
        hboxNextPrev.addWidget(self.buttonPlus)
        hboxNextPrev.addWidget(self.numberLeft)

        self.playButton = QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playButton.setIconSize(QSize(40, 40))
        self.playButton.clicked.connect(self.playSeg)

        # Volume, brightness and contrast sliders.
        # Need to pass true (config) values to set up correct initial positions
        self.specControls = SupportClasses_GUI.BrightContrVol(brightness, contrast, self.cmapInverted)
        self.specControls.colChanged.connect(self.setColourLevels)
        self.specControls.volChanged.connect(self.volSliderMoved)

        # zoom buttons
        self.zoomInBtn = QToolButton()
        self.zoomOutBtn = QToolButton()
        self.zoomInBtn.setIcon(QtGui.QIcon('img/zoom-in.png'))
        self.zoomOutBtn.setIcon(QtGui.QIcon('img/search.png'))
        self.zoomInBtn.setIconSize(QSize(24, 24))
        self.zoomOutBtn.setIconSize(QSize(24, 24))
        self.zoomInBtn.clicked.connect(self.zoomIn)
        self.zoomOutBtn.clicked.connect(self.zoomOut)
        self.zoomInBtn.setStyleSheet("padding: 4px 4px 4px 4px")
        self.zoomOutBtn.setStyleSheet("padding: 4px 4px 4px 4px")

        spNameBox = QHBoxLayout()
        spNameBox.addWidget(self.speciesTop)
        spNameBox.addWidget(self.species)
        spNameBox.addWidget(self.ctLabel)
        spNameBox.setStretch(0, 1)
        spNameBox.setStretch(1, 7)
        spNameBox.setStretch(2, 2)

        vboxSpecContr = pg.LayoutWidget()
        vboxSpecContr.addWidget(self.scroll, row=1, col=0, colspan=4)
        vboxSpecContr.addWidget(self.playButton, row=2, col=0)

        vboxSpecContr.addWidget(self.specControls, row=2, col=1)

        vboxSpecContr.addWidget(self.zoomInBtn, row=2, col=2)
        vboxSpecContr.addWidget(self.zoomOutBtn, row=2, col=3)

        vboxSpecContr.layout.setColumnStretch(1, 6) # specControls

        changeLabelsLayout = QHBoxLayout()
        self.menuBirdButton = QPushButton("Change species / calltypes")
        self.menuBirdButton.clicked.connect(self.showBirdMenu)
        changeLabelsLayout.addWidget(self.menuBirdButton)

        vboxFull = QVBoxLayout()
        vboxFull.addLayout(spNameBox)
        vboxFull.addWidget(vboxSpecContr)
        vboxFull.addLayout(changeLabelsLayout)
        vboxFull.addSpacing(7)
        vboxFull.addLayout(hboxNextPrev)

        self.setLayout(vboxFull)
        # print seg
        # self.setImage(self.sg,audiodata,sampleRate,self.label, unbufStart, unbufStop)
    
    def showBirdMenu(self):
        button_pos = self.menuBirdButton.mapToGlobal(self.menuBirdButton.rect().topLeft())
        self.updateSelectionMenu()
        self.menuSpeciesSelection.popup(button_pos)

    def playSeg(self):
        if self.media_obj.isPlayingorPaused():
            self.stopPlayback()
        else:
            self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
            self.playButton.setIconSize(QSize(40, 40))
            self.media_obj.loadArray(self.audiodata)
            self.NotifyTimer.start(30)

    def stopPlayback(self):
        self.media_obj.pressedStop()
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playButton.setIconSize(QSize(40, 40))
        self.NotifyTimer.stop()

    def volSliderMoved(self, value):
        self.media_obj.applyVolSlider(value)

    def movePlaySlider(self):
        """ Move the playback bar. And hijacked to detect the end of playback. """

        time = self.media_obj.elapsedUSecs() // 1000
        if time > self.duration:
            if self.media_obj.loop:
                self.media_obj.pressedStop()
                self.media_obj.loadArray(self.audiodata)
            else:
                self.stopPlayback()
        else:
            # TODO: Check
            barx = time / 1000 * self.sampleRate / self.incr
            self.bar.setValue(barx)
            self.bar.update()
            # QApplication.processEvents()

    def zoomIn(self):
        # resize the ViewBox with spec, lines, axis
        self.wPlot.zoomIn()

    def zoomOut(self):
        self.wPlot.zoomOut()

    def setSegNumbers(self, accepted, deleted, total):
        #print(accepted,deleted,total)
        text1 = "calls accepted: " + str(accepted) + ", deleted: " + str(deleted)
        text2 = str(total - accepted - deleted) + " to go"
        self.numberDone.setText(text1)
        self.numberLeft.setText(text2)
        # based on these, update "previous" arrow status
        self.buttonPrev.setEnabled((accepted+deleted)>0)
        self.update()
        QApplication.processEvents()
    
    def reorderShortBirdList(self, segment):
        # reorder list based on the existing segment
        if self.reorderShortList and not segment is None:
            for species in segment.getKeys():
                if species in self.shortBirdList:
                    self.shortBirdList.remove(species)
                else:
                    del self.shortBirdList[-1]
                self.shortBirdList.insert(0,species)
                        
        # move any blanks to the end, and 'Don't Know' to the start.
        self.shortBirdList = [x for x in self.shortBirdList if x != ""] + [x for x in self.shortBirdList if x == ""]
        self.shortBirdList = ["Don't Know"] + [x for x in self.shortBirdList if x != "Don't Know"][:29]
    
    def batLabelsUpdated(self,new_labels,species_changed,new_certainty):
        self.currentSegment[4] = new_labels        
                
        # Put the selected bird name at the top of the list.
        if self.reorderShortList:
            if species_changed in self.batList:
                self.batList.remove(species_changed)
            else:
                del self.batList[-1]
            self.batList.insert(0,species_changed)
    
    def birdLabelsUpdated(self,new_labels,species_changed,callname_changed,new_certainty):
        self.currentSegment[4] = new_labels
        
        # Put the selected bird name at the top of the list.
        if self.reorderShortList:
            if species_changed in self.shortBirdList:
                self.shortBirdList.remove(species_changed)
            else:
                del self.shortBirdList[-1]
            self.shortBirdList.insert(0,species_changed)
        
        self.updateTitle()

    def addBirdSpecies(self,certainty):
        # Ask the user for the new name, and save it
        species, ok = QInputDialog.getText(self, 'Bird name', 'Enter the bird name as genus (species)')
        if not ok:
            return

        species = str(species).title()
        # splits "A (B)", with B optional, into groups A and B
        match = re.fullmatch(r'(.*?)(?: \((.*)\))?', species)
        if not match:
            print("ERROR: provided name %s does not match format requirements" % species)
            return

        if species.lower()=="don't know" or species.lower()=="other" or species.lower()=="(other)":
            print("ERROR: provided name %s is reserved, cannot create" % species)
            return

        if "?" in species:
            print("ERROR: provided name %s contains reserved symbol '?'" % species)
            return

        if len(species)==0 or len(species)>150:
            print("ERROR: provided name appears to be too short or too long")
            return

        twolevelname = '>'.join(match.groups(default=''))
        if species in self.longBirdList or twolevelname in self.longBirdList:
            # bird is already listed
            print("Warning: not adding species %s as it is already present" % species)
            return

        # update the main list:
        nametostore = species
        pattern = r"^(.*) \((.*)\)$"
        match = re.match(pattern, species)
        if match:
            A = match.group(1)
            B = match.group(2)
            nametostore = A + '>' + B
        
        self.longBirdList.append(nametostore)
        self.longBirdList = sorted(self.longBirdList, key=str.lower)
        self.knownCalls[species] = []

        labels = copy.deepcopy(self.currentSegment[4])
        labels.append({"species": species, "certainty": certainty})
        if "Don't Know" in [x["species"] for x in labels]:
            labels = [x for x in labels if x["species"]!="Don't Know"]
        self.birdLabelsUpdated(labels,species,None,certainty)
        self.needToSaveConfig = True
    
    def addBirdCallname(self,species,certainty):
        # Ask the user for the new name, and save it
        callname, ok = QInputDialog.getText(self, 'Call type', 'Enter a label for this call type ')
        if not ok:
            return

        callname = str(callname).title()
        # splits "A (B)", with B optional, into groups A and B
        match = re.fullmatch(r'(.*?)(?: \((.*)\))?', callname)
        if not match:
            print("ERROR: provided name %s does not match format requirements" % callname)
            return

        if callname.lower()=="don't know" or callname.lower()=="other" or callname.lower()=="(other)":
            print("ERROR: provided name %s is reserved, cannot create" % callname)
            return

        if "?" in callname:
            print("ERROR: provided name %s contains reserved symbol '?'" % callname)
            return

        if len(callname)==0 or len(callname)>150:
            print("ERROR: provided name appears to be too short or too long")
            return
        
        if not species in self.knownCalls: self.knownCalls[species] = []

        if species in self.knownCalls:
            if callname in self.knownCalls[species]:
                print("Warning: not adding call type %s as it is already present" % callname)
                return

        self.knownCalls[species].append(callname)

        labels = copy.deepcopy(self.currentSegment[4])
        if species in [x["species"] for x in labels]:
            labels = [x for x in labels if x["species"]!=species]
        labels.append({"species": species, "certainty": certainty, "calltype": callname})
        if "Don't Know" in [x["species"] for x in labels]:
            labels = [x for x in labels if x["species"]!="Don't Know"]
        self.birdLabelsUpdated(labels,species,None,certainty)
        self.needToSaveConfig = True

    def setImage(self, sg, audiodata, sampleRate, incr, segment, unbufStart, unbufStop, guides=None, minFreq=0, maxFreq=0):
        """ Be careful not to edit it, as it is NOT a deep copy!!
            Used for extracting current species and calltype.
            During review, this updates self.label and self.ctLabel.
        """
        self.audiodata = audiodata
        self.sg = sg
        self.sgMinimum = np.min(self.sg)
        self.sgMaximum = np.max(self.sg)
        self.sampleRate = sampleRate
        self.incr = incr
        self.bar.setValue(0)
        self.currentSegment = segment
        if maxFreq==0:
            maxFreq = sampleRate / 2
        self.duration = len(audiodata) / sampleRate * 1000  # in ms
        self.segment = segment

        # Update UI if no audio (e.g. batmode)
        self.playButton.setEnabled(len(audiodata))
        self.specControls.volIcon.setEnabled(len(audiodata))
        self.specControls.volSlider.setEnabled(len(audiodata))

        self.updateSelectionMenu()
        
        # Fill up a rectangle with dark grey to act as background if the segment is small
        sg2 = sg
        # sg2 = 40 * np.ones((max(1000, np.shape(sg)[0]), max(100, np.shape(sg)[1])))
        # sg2[:np.shape(sg)[0], :np.shape(sg)[1]] = sg

        # add axis
        self.plot.setImage(sg2)
        self.plot.setLookupTable(self.lut)
        self.specControls.emitCol()
        self.scroll.horizontalScrollBar().setValue(0)

        self.wPlot.forceResize()

        FreqRange = (maxFreq-minFreq)/1000.
        SgSize = np.shape(sg2)[1]
        ticks = [(0,minFreq/1000.), (SgSize/4, minFreq/1000.+FreqRange/4.), (SgSize/2, minFreq/1000.+FreqRange/2.), (3*SgSize/4, minFreq/1000.+3*FreqRange/4.), (SgSize,minFreq/1000.+FreqRange)]
        ticks = [[(tick[0], "%.1f" % tick[1] ) for tick in ticks]]
        self.sg_axis.setTicks(ticks)
        self.sg_axis.setLabel('kHz')
        #self.sg_axis2.setTicks(ticks)
        #self.sg_axis2.setLabel('kHz')

        self.show()

        # self.pPlot.setYRange(0, SgSize, padding=0.02)
        self.pPlot.setRange(xRange=(0, np.shape(sg2)[0]), yRange=(0, SgSize))
        
        # Add marks to separate actual segment from buffer zone
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
        # Add time markers next to the lines
        time1 = QTime(0,0,0).addSecs(int(segment[0])).toString('hh:mm:ss')
        time2 = QTime(0,0,0).addSecs(int(segment[1])).toString('hh:mm:ss')
        self.segTimeText1.setText(time1)
        self.segTimeText2.setText(time2)
        self.segTimeText1.setPos(startV, SgSize)
        self.segTimeText2.setPos(stopV, SgSize)

        # Bat mode freq guides
        if guides is not None:
            for i in range(len(self.guidelines)):
                self.guidelines[i].setPos(guides[i])
        else:
            for i in range(len(self.guidelines)):
                self.guidelines[i].setPos(-100)

        self.specControls.emitAll()

        # DEAL WITH SPECIES NAMES
        # Extract a string of current species names
        self.updateTitle()

        # Extract the call type of the (first) species
        labels = self.segment[4]
        if "calltype" in labels[0]:
            self.ctLabel.setText(labels[0]["calltype"])
        else:
            self.ctLabel.setText("")

        if self.autoplay:
            self.playSeg()
    
    def updateTitle(self):
        specnames = []
        labels = self.segment[4]
        for lab in labels:
            specnames.append(lab["species"]+" ["+lab["calltype"]+"]" if "calltype" in lab else lab["species"])
        specnames = list(set(specnames))
        self.species.setText(','.join(specnames))

    def updateSelectionMenu(self):
        self.reorderShortBirdList(self.segment)
        currentLabels = self.segment[4]
        if self.batmode:
            self.menuSpeciesSelection = SupportClasses_GUI.BatSelectionMenu(
                batList=self.batList, 
                currentLabels=currentLabels, 
                parent=self, 
                unsure=False,
                multipleBirds=self.multipleBirds
            )
            self.menuSpeciesSelection.labelsUpdated.connect(self.batLabelsUpdated)
        else:
            self.menuSpeciesSelection = SupportClasses_GUI.BirdSelectionMenu(
                shortBirdList=self.shortBirdList, 
                longBirdList=self.longBirdList,
                knownCalls=self.knownCalls, 
                currentLabels=currentLabels, 
                parent=self, 
                unsure=False,
                multipleBirds=self.multipleBirds
            )
            self.menuSpeciesSelection.addSpecies.connect(self.addBirdSpecies)
            self.menuSpeciesSelection.addCallname.connect(self.addBirdCallname)
            self.menuSpeciesSelection.labelsUpdated.connect(self.birdLabelsUpdated)

    def setColourLevels(self, brightness, contrast):
        """ Listener for the brightness and contrast sliders being changed. Also called when spectrograms are loaded, etc.
        Translates the brightness and contrast values into appropriate image levels.
        Calculation is simple.
        """
        try:
            self.stopPlayback()
        except Exception:
            pass

        if not self.cmapInverted:
            brightness = 100-brightness

        colRange = colourMaps.getColourRange(self.sgMinimum, self.sgMaximum, brightness, contrast, self.cmapInverted)
        self.plot.setLevels(colRange)
    
    def checkIfNeedToSaveConfig(self):
        return self.needToSaveConfig


class HumanClassify2(QDialog):
    """ Single Species review main dialog.
        Puts all segments of a certain species together on buttons, and their labels.
        Allows quick confirm/leave/delete check over many segments.

        Construction:
        1. a list of Spectrogram containing spectrograms for ALL the segments in arg2
        2. SegmentList. Just provide full versions of this,
          and this dialog will select the needed segments.
        3. indices of segments to show (i.e. the selected species and current page)
        4. name of the species that we are reviewing
        5-8. spec color parameters
        9-10. guide positions and colors for batmode
        11. Loop playback or not?
        12. Filename - just for setting the window title
    """

    def __init__(self, sps, sgs, segments, indicestoshow, label, lut, cmapInverted, brightness, contrast, guidefreq=None, guidecol=None, loop=False, filename=None):
        QDialog.__init__(self)

        if len(segments)==0:
            print("No segments provided")
            return

        if filename:
            self.setWindowTitle('Human review - ' + filename)
        else:
            self.setWindowTitle('Human review')

        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint)
        # Let the user quit without bothering rest of it

        self.sps = sps
        self.sgs = sgs
        # Check if playback is possible (e.g. for batmode):
        haveaudio = all(len(sp.data)>0 for sp in sps if sp is not None)

        self.lut = lut
        self.cmapInverted = cmapInverted

        # Filter segments for the requested species
        self.segments = segments
        self.indices2show = indicestoshow

        self.errors = []

        self.loop = loop

        # Volume, brightness and contrast sliders.
        # Need to pass true (config) values to set up correct initial positions
        self.specControls = SupportClasses_GUI.BrightContrVol(brightness, contrast, self.cmapInverted)
        self.specControls.colChanged.connect(self.setColourLevels)
        self.specControls.volChanged.connect(self.volSliderMoved)
        self.specControls.layout().addStretch(3) # add a big stretchable outer margin

        # Batmode customizations:
        self.guidefreq = guidefreq
        self.guidecol = guidecol
        if not haveaudio:
            self.specControls.volSlider.setEnabled(False)
            self.specControls.volIcon.setEnabled(False)

        label1 = QLabel('Click on the images that are incorrectly labelled or need some change.')
        label1.setFont(QtGui.QFont('SansSerif', 10))
        species = QLabel(label)
        species.setStyleSheet("padding: 2px 0px 5px 0px")
        font = QtGui.QFont('SansSerif', 12)
        font.setBold(True)
        species.setFont(font)

        # Species label and sliders
        vboxTop = QVBoxLayout()
        vboxTop.addWidget(label1)
        vboxTop.addWidget(species)
        vboxTop.addWidget(self.specControls)

        # Controls at the bottom
        # self.buttonPrev = QtGui.QToolButton()
        # self.buttonPrev.setArrowType(Qt.LeftArrow)
        # self.buttonPrev.setIconSize(QSize(30,30))
        # self.buttonPrev.clicked.connect(self.prevPage)

        # TODO: Is this useful?
        self.pageLabel = QLabel()

        self.none = QPushButton("Toggle all")
        #self.none.setSizePolicy(QSizePolicy(5,5))
        self.none.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.none.setMaximumSize(250, 30)
        self.none.clicked.connect(self.toggleAll)

        # Either the next or finish button is visible. They have different internal
        # functionality, but look the same to the user
        self.next = QPushButton("Next")
        #self.next.setSizePolicy(QSizePolicy(5,5))
        self.next.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.next.setMaximumSize(250, 30)
        self.next.clicked.connect(self.nextPage)

        self.finish = QPushButton("Next")
        #self.finish.setSizePolicy(QSizePolicy(5,5))
        self.finish.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.finish.setMaximumSize(250, 30)

        # Movement buttons and page numbers
        self.vboxBot = QHBoxLayout()
        # vboxBot.addWidget(self.buttonPrev)
        # vboxBot.addSpacing(20)
        self.vboxBot.addWidget(self.pageLabel)
        self.vboxBot.addSpacing(20)
        self.vboxBot.addWidget(self.none)
        self.vboxBot.addWidget(self.next)

        # Create the button objects, and we'll show them as needed
        # (fills self.buttons)
        self.flowContainer = QWidget()
        self.flowLayout = QGridLayout()
        self.flowLayout.setSpacing(10)
        self.flowContainer.setLayout(self.flowLayout)
        self.flowContainer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.flowLayout.setRowStretch(0, 100)
        self.flowLayout.setColumnStretch(0, 100)

        # this is the starting index of the first button, which we change when we flip pages
        self.butStart = 0
        self.maxCols = 10
        self.maxRows = 10

        self.createButtons()
        self.specControls.emitAll()  # applies initial colour, volume levels

        self.countPages()

        # Set overall layout of the dialog
        self.vboxFull = QVBoxLayout()
        # self.vboxFull.setSpacing(0)
        self.vboxFull.addLayout(vboxTop)
        self.vboxSpacer = QSpacerItem(1,1, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.vboxFull.addItem(self.vboxSpacer)
        self.vboxFull.addWidget(self.flowContainer)
        self.vboxFull.addLayout(self.vboxBot)
        # Must be fixed size!
        vboxTop.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        # Must be fixed size!
        self.vboxBot.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        # We need to know the true size of space available for flowLayout.
        # The idea is that spacer absorbs all height changes
        #self.setSizePolicy(1,1)
        self.setSizePolicy(QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Minimum)
        self.setLayout(self.vboxFull)
        self.vboxFull.setStretch(1, 100)
        # Plan B could be to measure the sizes of the top/bottom boxes and subtract them
        # self.boxSpaceAdjustment = vboxTop.sizeHint().height() + vboxBot.sizeHint().height()

    def createButtons(self):
        """ Create the button objects, add audio, calculate spec, etc.
            So that when users flips through pages, we only need to
            retrieve the right ones from resizeEvent.
            No return, fills out self.buttons.
        """
        self.buttons = []
        self.marked = []
        self.minsg = 1
        self.maxsg = 1
        for i in self.indices2show:
            # This will contain pre-made slices of spec and audio
            sp = self.sps[i]
            duration = len(sp.data)/sp.audioFormat.sampleRate()

            sg = self.sgs[i]
            
            # Seems that image is backwards?
            sg = np.fliplr(sg)

            self.minsg = min(self.minsg, np.min(sg))
            self.maxsg = max(self.maxsg, np.max(sg))

            # Batmode guides, in y of this particular spectrogram:
            if self.guidefreq is not None:
                gy = [0]*len(self.guidefreq)
                for gix in range(len(self.guidefreq)):
                    gy[gix] = sp.convertFreqtoY(self.guidefreq[gix])
            else:
                gy = None

            # Create the button:
            # args: index, sp, audio, format, duration, ubstart, ubstop (in spec units)

            newButton = SupportClasses_GUI.PicButton(i, sg, sp.data, sp.audioFormat, duration, sp.x1nobspec, sp.x2nobspec, self.lut, guides=gy, guidecol=self.guidecol, loop=self.loop, scaleToButton=True)
            newButton.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
            newButton.setMinimumSize(10, 10)
            self.buttons.append(newButton)
            self.buttons[-1].buttonClicked=False
            self.marked.append(False)
        self.redrawButtons()

    def redrawButtons(self):
        # create one frequency axis
        # (all of them are identical b/c only 1 file shown at each time)
        exampleSP = self.sps[self.indices2show[0]]
        minFreq = exampleSP.minFreqShow
        maxFreq = exampleSP.maxFreqShow
        if maxFreq==0:
            maxFreq = exampleSP.audioFormat.sampleRate() // 2
        if len(exampleSP.data)>0:
            duration = len(exampleSP.data)/exampleSP.audioFormat.sampleRate()
        else:
            duration = exampleSP.convertSpectoAmpl(np.shape(exampleSP.sg)[0])

        butNum = 0

        numRows = min(self.maxCols,int(np.ceil(np.sqrt(len(self.buttons)))))
        numCols = min(self.maxRows,int(np.ceil(len(self.buttons)/numRows)))

        for row in range(self.maxRows):
            if row<numRows:
                self.flowLayout.setRowStretch(row, 1)
            else:
                self.flowLayout.setRowStretch(row, 0)
        for col in range(numCols):
            if col<numCols:
                self.flowLayout.setColumnStretch(col, 1)
            else:
                self.flowLayout.setColumnStretch(col, 0)

        for row in range(numRows):
            for col in range(numCols):
                self.flowLayout.addWidget(self.buttons[self.butStart+butNum], row, col)
                self.buttons[self.butStart+butNum].show()
                butNum += 1
                if self.butStart+butNum==len(self.buttons):
                    # stop if we are out of segments
                    break
            if self.butStart+butNum==len(self.buttons):
                # stop if we are out of segments
                break

        self.repaint()
        QApplication.processEvents()

    def volSliderMoved(self, value):
        # try/pass to avoid race situations when smth is not initialized
        try:
            for btn in self.buttons:
                btn.media_obj.applyVolSlider(value)
        except Exception:
            pass

    def countPages(self):
        """ Counts the total number of pages,
            finds where we are, how many remain, etc.
            Called on resize, so does not update current button position.
            Updates next/prev arrow states.
        """
        buttonsPerPage = self.maxRows * self.maxCols
        if buttonsPerPage == 0:
            # dialog still initializing or too small to show segments
            #self.buttonPrev.setEnabled(False)
            return
        # basically, count how many segments are "before" the current
        # top-lef one, and see how many pages we need to fit them.
        currpage = int(np.ceil(self.butStart / buttonsPerPage)+1)
        self.totalPages = max(int(np.ceil(len(self.buttons) / buttonsPerPage)),currpage)
        self.pageLabel.setText("Page %d out of %d" % (currpage, self.totalPages))

        if currpage == self.totalPages:
            try:
                self.vboxBot.removeWidget(self.next)
                self.next.setVisible(False)
                self.vboxBot.addWidget(self.finish)
                self.finish.setVisible(True)
            except:
                pass
        else:
            if self.finish.isVisible():
                try:
                    self.vboxBot.removeWidget(self.finish)
                    self.finish.setVisible(False)
                    self.vboxBot.addWidget(self.next)
                    self.next.setVisible(True)
                except:
                    pass

        self.repaint()

        #if currpage==1:
            #self.buttonPrev.setEnabled(False)
        #else:
            #self.buttonPrev.setEnabled(True)

    def nextPage(self):
        """ Called on arrow button clicks.
            Updates current segment position, and calls other functions
            to deal with actual page recount/redraw.
        """
        buttonsPerPage = self.maxRows * self.maxCols
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
        buttonsPerPage = self.maxRows * self.maxCols
        # clear buttons while self.butStart is still old:
        self.clearButtons()
        self.butStart = max(0, self.butStart-buttonsPerPage)
        self.countPages()
        # redraw buttons:
        self.redrawButtons()

    def clearButtons(self):
        for btn in self.buttons:
            btn.stopPlayback()
        # clear pic buttons
        for i in reversed(range(self.flowLayout.count())):
            item = self.flowLayout.itemAt(i)
            widget = item.widget()
            if widget:
                self.flowLayout.removeWidget(widget)
                widget.hide()
                widget.setParent(None)

    def toggleAll(self):
        buttonsPerPage = self.maxRows * self.maxCols
        for butNum in range(self.butStart,min(self.butStart+buttonsPerPage,len(self.buttons))):
            self.buttons[butNum].changePic(False)
        #self.update()
        self.repaint()
        QApplication.processEvents()

    def setColourLevels(self, brightness, contrast):
        """ Listener for the brightness and contrast sliders being changed. Also called when spectrograms are loaded, etc.
        Translates the brightness and contrast values into appropriate image levels.
        """
        if not self.cmapInverted:
            brightness = 100-brightness

        colRange = colourMaps.getColourRange(self.minsg, self.maxsg, brightness, contrast, self.cmapInverted)

        for btn in self.buttons:
            btn.stopPlayback()
            btn.setImage(colRange)
            btn.update()


class FilterManager(QDialog):
    def __init__(self, filtdir, parent=None):
        super(FilterManager, self).__init__(parent)
        self.setWindowTitle("Manage recognisers")
        self.setWindowIcon(QIcon('img/Avianz.ico'))

        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.filtdir = filtdir

        # filter dir name
        labDirName = QLineEdit()
        labDirName.setText(filtdir)
        labDirName.setReadOnly(True)
        labDirName.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        labDirName.setStyleSheet("background-color: #e0e0e0")

        # filter dir contents
        self.listFiles = QListWidget()
        self.listFiles.setMinimumHeight(275)
        self.listFiles.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        self.readContents()

        # rename a filter
        self.enterFiltName = QLineEdit()

        class FiltValidator(QValidator):
            def validate(self, input, pos):
                if not input.endswith('.txt'):
                    input = input+'.txt'
                if input==".txt" or input=="":
                    return(QValidator.State.Intermediate, input, pos)
                if self.listFiles.findItems(input, Qt.MatchFlag.MatchExactly):
                    print("duplicated input", input)
                    return(QValidator.State.Intermediate, input, pos)
                else:
                    return(QValidator.State.Acceptable, input, pos)

        renameFiltValid = FiltValidator()
        renameFiltValid.listFiles = self.listFiles
        self.enterFiltName.setValidator(renameFiltValid)

        self.renameBtn = QPushButton("Rename")
        self.renameBtn.clicked.connect(self.rename)

        # delete a filter
        self.deleteBtn = QPushButton("Delete")
        self.deleteBtn.clicked.connect(self.delete)

        # export a filter for upload
        self.uploadBtn = QPushButton("Export")
        self.uploadBtn.clicked.connect(self.upload)

        # import downloaded filters
        self.downloadBtn = QPushButton("Import")
        self.downloadBtn.clicked.connect(self.download)

        # make button state respond to selection + name entry
        self.refreshButtons()
        self.listFiles.itemSelectionChanged.connect(self.refreshButtons)
        self.enterFiltName.textChanged.connect(self.refreshButtons)

        # layouts
        box_rename = QHBoxLayout()
        box_rename.addWidget(self.enterFiltName)
        box_rename.addWidget(self.renameBtn)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Recognisers are stored in:"))
        layout.addWidget(labDirName)
        layout.addWidget(QLabel("The following recognisers are present:"))
        layout.addWidget(self.listFiles)
        layout.addWidget(QLabel("To rename a recogniser, select one and enter a new (unique) name below:"))
        layout.addLayout(box_rename)

        layout.addWidget(self.deleteBtn)
        layout.addWidget(self.uploadBtn)
        layout.addWidget(self.downloadBtn)
        self.setLayout(layout)

    def readContents(self):
        self.listFiles.clear()
        cl = SupportClasses.ConfigLoader()
        self.FilterDict = cl.filters(self.filtdir, bats=True)
        for file in self.FilterDict:
            item = QListWidgetItem(self.listFiles)
            item.setText(file)

    def refreshButtons(self):
        if len(self.listFiles.selectedItems())==0:
            self.deleteBtn.setEnabled(False)
            self.uploadBtn.setEnabled(False)
            self.enterFiltName.setEnabled(False)
            self.renameBtn.setEnabled(False)
        else:
            self.deleteBtn.setEnabled(True)
            self.uploadBtn.setEnabled(True)
            self.enterFiltName.setEnabled(True)
            if self.enterFiltName.hasAcceptableInput():
                self.renameBtn.setEnabled(True)
            else:
                self.renameBtn.setEnabled(False)

    def rename(self):
        """ move the filter file. """
        source = self.listFiles.currentItem().text()
        source = os.path.join(self.filtdir, source + '.txt')
        target = self.enterFiltName.text()
        target = os.path.join(self.filtdir, target)
        # figured we should have our own gentle error handling
        # before trying to force move with shutil
        if os.path.isfile(target) or not target.endswith(".txt"):
            print("ERROR: unable to rename, bad target", target)
            return
        if not os.path.isfile(source):
            print("ERROR: unable to rename, bad source", source)
            return
        try:
            os.rename(source, target)
            self.readContents()
            self.enterFiltName.setText("")
        except Exception as e:
            print("ERROR: could not rename:", e)

    def delete(self):
        """ confirm and delete the file/s. """
        sources = []
        fn = self.listFiles.currentItem().text()
        currfilt = self.FilterDict[fn]
        sources.append(os.path.join(self.filtdir, fn + '.txt'))

        # ROCs
        if "ROCWF" in currfilt:
            sources.append(os.path.join(self.filtdir, currfilt["ROCWF"] + ".json"))
        if "RONN" in currfilt:
            sources.append(os.path.join(self.filtdir, currfilt["RONN"] + ".json"))

        if "NN" in currfilt:
            if os.path.isfile(os.path.join(self.filtdir, currfilt["NN"]["NN_name"] + ".h5")):
                sources.append(os.path.join(self.filtdir, currfilt["NN"]["NN_name"] + ".h5"))
            elif os.path.isfile(os.path.join(self.filtdir, currfilt["NN"]["NN_name"] + ".weights.h5")):
                sources.append(os.path.join(self.filtdir, currfilt["NN"]["NN_name"] + ".weights.h5"))
            # bat filters do not have jsons:
            if os.path.isfile(os.path.join(self.filtdir, currfilt["NN"]["NN_name"] + ".json")):
                sources.append(os.path.join(self.filtdir, currfilt["NN"]["NN_name"] + ".json"))

        for src in sources:
            if not os.path.isfile(src):
                print("ERROR: unable to delete, bad source", src)
                return

        msg = SupportClasses_GUI.MessagePopup("w", "Confirm delete", "Warning: you are about to permanently delete recogniser %s.\nAre you sure?" % sources[0])
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        reply = msg.exec()
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            print("removing", sources)
            for src in sources:
                os.remove(src)
            self.readContents()
        except Exception as e:
            print("ERROR: could not delete:", e)

    def download(self):
        # Also import corresponding NN files if any
        sources = []
        targets = []
        source, _ = QFileDialog.getOpenFileName(self, 'Select the downloaded recogniser file', os.path.expanduser("~"), "Text files (*.txt)")
        sources.append(source)
        targets.append(os.path.join(self.filtdir, os.path.basename(source)))
        try:
            ff = open(source)
            filt = json.load(ff)
            ff.close()

            # skip this filter if it looks fishy:
            if not isinstance(filt, dict) or "species" not in filt or "SampleRate" not in filt or "Filters" not in filt or len(filt["Filters"]) < 1:
                raise ValueError("Filter JSON format wrong, skipping")
            for subfilt in filt["Filters"]:
                if not isinstance(subfilt, dict) or "calltype" not in subfilt or "WaveletParams" not in subfilt or "TimeRange" not in subfilt:
                    raise ValueError("Subfilter JSON format wrong, skipping")
                if "thr" not in subfilt["WaveletParams"] or "nodes" not in subfilt["WaveletParams"] or len(
                        subfilt["TimeRange"]) < 4:
                    raise ValueError("Subfilter JSON format wrong (details), skipping")
            # wavelet ROC if exists
            JSONsource = os.path.join(os.path.dirname(source), filt["ROCWF"] + ".json")
            if os.path.isfile(JSONsource):
                sources.append(JSONsource)
                targets.append(os.path.join(self.filtdir, filt["ROCWF"] + ".json"))
            if "NN" in filt:
                if os.path.isfile(os.path.join(os.path.dirname(source), filt["NN"]["NN_name"] + ".h5")): # old tensorflow version of saving weights
                    sources.append(os.path.join(os.path.dirname(source), filt["NN"]["NN_name"] + ".h5"))
                    targets.append(os.path.join(self.filtdir, filt["NN"]["NN_name"] + ".h5"))
                elif os.path.isfile(os.path.join(os.path.dirname(source), filt["NN"]["NN_name"] + ".weights.h5")): # new
                    sources.append(os.path.join(os.path.dirname(source), filt["NN"]["NN_name"] + ".weights.h5"))
                    targets.append(os.path.join(self.filtdir, filt["NN"]["NN_name"] + ".weights.h5"))
                # bat filters do not have jsons:
                JSONsource = os.path.join(os.path.dirname(source), filt["NN"]["NN_name"] + ".json")
                if os.path.isfile(JSONsource):
                    sources.append(JSONsource)
                    targets.append(os.path.join(self.filtdir, filt["NN"]["NN_name"] + ".json"))
                # NN ROC if exists
                JSONsource = os.path.join(os.path.dirname(source), filt["RONN"] + ".json")
                if os.path.isfile(JSONsource):
                    sources.append(JSONsource)
                    targets.append(os.path.join(self.filtdir, filt["RONN"] + ".json"))
        except Exception as e:
            print("Could not load filter:", source, e)
            return

        try:

            for i in range(len(sources)):
                if not os.path.isfile(sources[i]):
                    print("ERROR: unable to import, bad source %s" % sources[i])
                    return
                # Don't risk replacing NN files (i.e. no overwriting)
                reply = 0
                if os.path.isfile(targets[i]):
                    print("Warning: target file %s exists" % targets[i])
                    msg = SupportClasses_GUI.MessagePopup("t", "Import error"," A file %s already exists. Overwrite or skip?" % targets[i])
                    msg.setStandardButtons(QMessageBox.StandardButton.NoButton)
                    msg.addButton("Overwrite", QMessageBox.ButtonRole.YesRole)
                    msg.addButton("Skip", QMessageBox.ButtonRole.RejectRole)
                    reply = msg.exec()
                #print(reply,QMessageBox.ButtonRole.YesRole.value,QMessageBox.ButtonRole.RejectRole.value)
                if reply==QMessageBox.ButtonRole.YesRole.value:
                    # no problems, or chose to overwrite
                    print("Copying", sources[i], "->", targets[i])
                    shutil.copy2(sources[i], targets[i])
                elif reply==QMessageBox.ButtonRole.RejectRole.value: #4194304:
                    # cancelled the entire copy
                    print("Cancelled")
                    return
            msg = SupportClasses_GUI.MessagePopup("d", "Successfully imported","Import complete. Now you can use the recogniser %s" % os.path.basename(targets[0]))
            msg.exec()
            self.readContents()
        except Exception as e:
            print("ERROR: failed to import")
            print(e)
            return

    def upload(self):
        # Also export corresponding NN files if any
        fn = self.listFiles.currentItem().text()
        currfilt = self.FilterDict[fn]
        sources = []
        sources.append(fn + '.txt')

        # ROCs
        if "ROCWF" in currfilt:
            sources.append(currfilt["ROCWF"] + ".json")
        if "RONN" in currfilt:
            sources.append(currfilt["RONN"] + ".json")

        if "NN" in currfilt:
            if os.path.isfile(os.path.join(self.filtdir, currfilt["NN"]["NN_name"] + ".h5")): # old tensorflow version of saving weights
                sources.append(currfilt["NN"]["NN_name"] + ".h5")
            elif os.path.isfile(os.path.join(self.filtdir, currfilt["NN"]["NN_name"] + ".weights.h5")): # new
                sources.append(currfilt["NN"]["NN_name"] + ".weights.h5")
            # bat filters do not have jsons:
            if os.path.isfile(os.path.join(self.filtdir, currfilt["NN"]["NN_name"] + ".json")):
                sources.append(currfilt["NN"]["NN_name"] + ".json")

        target = QFileDialog.getExistingDirectory(self, 'Choose where to save the recogniser')
        if target != "":
            targets = []
            for src in sources:
                targets.append(os.path.join(target, src))
            sources = [os.path.join(self.filtdir, src) for src in sources]

            print("Exporting from %s to %s" % (sources, targets))
            try:
                for i in range(len(sources)):
                    if not os.path.isfile(sources[i]):
                        print("ERROR: unable to export, bad source %s" % sources[i])
                        return
                    if os.path.isfile(targets[i]):
                        print("ERROR: target file %s exists" % targets[i])
                        return
                    shutil.copy2(sources[i], targets[i])
                msg = SupportClasses_GUI.MessagePopup("d", "Successfully exported", "Export successful. Now you can share the recogniser file(s) in %s" % target)
                msg.exec()
            except Exception as e:
                print("ERROR: failed to export")
                print(e)
                return

class Cluster(QDialog):
    def __init__(self, segments, sampleRate, classes, config, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Clustered segments')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)

        if len(segments) == 0:
            print("No segments provided")
            return

        self.sampleRate = sampleRate
        self.segments = segments
        self.nclasses = classes
        self.config = config

        # Volume, brightness and contrast sliders.
        # Need to pass true (config) values to set up correct initial positions
        self.specControls = SupportClasses_GUI.BrightContrVol(80, 20, False)
        self.specControls.colChanged.connect(self.setColourLevels)
        self.specControls.volChanged.connect(self.volSliderMoved)

        # Colour map
        self.lut = colourMaps.getLookupTable(self.config['cmap'])

        # set up the images
        self.flowLayout = pg.LayoutWidget()
        self.flowLayout.setGeometry(QtCore.QRect(0, 0, 380, 247))

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.flowLayout)

        # set overall layout of the dialog
        self.vboxFull = QVBoxLayout()
        self.vboxFull.addWidget(self.specControls)
        self.vboxFull.addWidget(self.scrollArea)
        self.setLayout(self.vboxFull)

        # Add the clusters to rows
        self.addButtons()
        self.updateButtons()

    def addButtons(self):
        """ Only makes the PicButtons and self.clusters dict
        TODO: Get the parameters for the spectrogram from the config
        """
        self.clusters = []
        self.picbuttons = []
        for i in range(self.nclasses):
            self.clusters.append((i, 'Type_' + str(i)))
        self.clusters = dict(self.clusters)  # Dictionary of {ID: cluster_name}

        # Create the buttons for each segment
        self.minsg = 1
        self.maxsg = 1
        for seg in self.segments:
            sp = Spectrogram.Spectrogram(self.config['window_width'],self.config['incr'])
            sp.readSoundFile(seg[0], seg[1][1] - seg[1][0], seg[1][0])
            #_ = sp.spectrogram(window='Hann', sgType='Standard',mean_normalise=True, onesided=True, need_even=False)
            self.sg = self.sp.spectrogram(window_width=self.config['window_width'], incr=self.config['incr'],window=self.config['windowType'],sgType=self.config['sgType'],sgScale=self.config['sgScale'],nfilters=self.config['nfilters'],mean_normalise=self.config['sgMeanNormalise'],equal_loudness=self.config['sgEqualLoudness'],onesided=self.config['sgOneSided'])
            #self.sg = sp.normalisedSpec("Log")

            self.minsg = min(self.minsg, np.min(self.sg))
            self.maxsg = max(self.maxsg, np.max(self.sg))

            newButton = SupportClasses_GUI.PicButton(1, np.fliplr(sg), sp.data, sp.audioFormat, seg[1][1] - seg[1][0], 0, seg[1][1], self.lut, cluster=True)
            self.picbuttons.append(newButton)
        # (updateButtons will place them in layouts and show them)

    def updateButtons(self):
        """ Draw the existing buttons, and create check- and text-boxes.
        Called when merging clusters or initializing the page. """
        self.tboxes = []    # Corresponding list of text boxes
        for r in range(self.nclasses):
            c = 0
            tbox = QLineEdit(self.clusters[r])
            tbox.setMinimumWidth(80)
            tbox.setMaximumHeight(150)
            tbox.setStyleSheet("border: none;")
            tbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tboxes.append(tbox)
            self.flowLayout.addWidget(self.tboxes[-1], r, c)
            c += 1
            # Find the segments under this class and show them
            for segix in range(len(self.segments)):
                if self.segments[segix][-1] == r:
                    self.flowLayout.addWidget(self.picbuttons[segix], r, c)
                    c += 1
                    self.picbuttons[segix].show()
        self.flowLayout.update()
        self.specControls.emitAll()  # applies initial colour, volume levels

    def setColourLevels(self, brightness, contrast):
        """ Listener for the brightness and contrast sliders being changed. Also called when spectrograms are loaded, etc.
        Translates the brightness and contrast values into appropriate image levels.
        """
        brightness = 100-brightness
        colRange = colourMaps.getColourRange(self.minsg, self.maxsg, brightness, contrast, False)
        for btn in self.picbuttons:
            btn.stopPlayback()
            btn.setImage(colRange)
            btn.update()

    def volSliderMoved(self, value):
        # try/pass to avoid race situations when smth is not initialized
        try:
            for btn in self.picbuttons:
                btn.media_obj.applyVolSlider(value)
        except Exception:
            pass


class ExportBats(QDialog):
    def __init__(self,filename,observer,easting="",northing="",recorder=""):
        QDialog.__init__(self)
        self.setWindowTitle('Export Results?')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)

        l1 = QLabel('Do you want to export an entry for the National Bat Database?\n(It will be saved as '+filename+'.\nYou will need to email this file yourself.\nFields with a * are mandatory)\n')

        forml = QFormLayout()
        self.data = QLineEdit(self)
        forml.addRow('* Data source:\n(e.g., your community group)', self.data)
        self.observer = QLineEdit(self)
        self.observer.setText(observer)
        forml.addRow('* Your name: ', self.observer)
        self.method = QLineEdit(self)
        forml.addRow('Method: ', self.method)
        self.detector = QLineEdit(self)
        self.detector.setText('ABM')
        forml.addRow('Detector Type: ', self.detector)
        self.notes = QLineEdit(self)
        forml.addRow('Any notes: ', self.notes)

        self.easting = QLineEdit(self)
        self.easting.setText(easting)
        self.easting.setValidator(QIntValidator())
        self.northing = QLineEdit(self)
        self.northing.setText(northing)
        self.northing.setValidator(QIntValidator())
        forml.addRow('* Easting: ', self.easting)
        forml.addRow('* Northing: ', self.northing)
        self.site = QLineEdit(self)
        self.site.setText(recorder)
        forml.addRow('* Site where data collected: ', self.site)
        self.region = QLineEdit(self)
        forml.addRow('Region where data collected: ', self.region)

        # mandatory fields emit signal to check if "Accept" should be enabled
        self.data.textEdited.connect(self.checkInputs)
        self.observer.textEdited.connect(self.checkInputs)
        self.easting.textEdited.connect(self.checkInputs)
        self.northing.textEdited.connect(self.checkInputs)
        self.site.textEdited.connect(self.checkInputs)

        # buttons
        hbox9 = QHBoxLayout()
        self.yesbtn = QPushButton('Export')
        no = QPushButton('Skip')
        self.yesbtn.clicked.connect(self.accept)
        no.clicked.connect(self.reject)
        hbox9.addWidget(self.yesbtn)
        self.yesbtn.setEnabled(False)
        self.yesbtn.setToolTip("You need to fill in the mandatory fields")
        hbox9.addWidget(no)

        vbox = QVBoxLayout()
        vbox.addWidget(l1)
        vbox.addLayout(forml)
        vbox.addLayout(hbox9)

        self.setLayout(vbox)

    def checkInputs(self):
        allgood = len(self.data.text()) > 0 and len(self.observer.text()) > 0 and len(self.easting.text()) > 0 and len(self.northing.text()) > 0 and len(self.site.text())>0
        if allgood:
            self.yesbtn.setEnabled(True)
        else:
            self.yesbtn.setEnabled(False)
            self.yesbtn.setToolTip("You need to fill in the mandatory fields")

    def getValues(self):
        return [self.data.text(), self.observer.text(),self.method.text(),self.detector.text(),self.notes.text(), self.easting.text(),self.northing.text(), self.site.text(), self.region.text()]

class Shapes(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Shape analysis')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(300)

        # if len(segments) == 0:
        #     print("No segments to analyse!")
        #     return

        # self.sampleRate = sampleRate
        # self.config = config
        # self.segments = segments

        self.detectorCombo = QComboBox()
        self.detectorCombo.addItems(['stupidShaper', 'fundFreqShaper', 'instantShaper1', 'instantShaper2'])
        self.detectorCombo.currentIndexChanged.connect(self.changeBoxes)

        self.activate = QPushButton('Detect shapes in segments')

        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("Choose shape detection method"))
        vbox.addWidget(self.detectorCombo)

        #add parameter selection for instan frequency methods

        self.IF1alpha = QDoubleSpinBox()
        self.IF1alpha.setRange(0.0, 2.0)
        self.IF1alpha.setSingleStep(0.05)
        self.IF1alpha.setValue(1)

        self.IF2alpha = QDoubleSpinBox()
        self.IF2alpha.setRange(0.0, 2.0)
        self.IF2alpha.setSingleStep(0.05)
        self.IF2alpha.setValue(1)

        self.IF2beta = QDoubleSpinBox()
        self.IF2beta.setRange(0.0, 2.0)
        self.IF2beta.setSingleStep(0.05)
        self.IF2beta.setValue(1)

        self.IF1Layout = QFormLayout()
        self.IF1Layout.addRow("Alpha:", self.IF1alpha)

        self.IF2Layout = QFormLayout()
        self.IF2Layout.addRow("Alpha:", self.IF2alpha)
        self.IF2Layout.addRow("Beta:", self.IF2beta)


        vbox.addLayout(self.IF1Layout)
        vbox.addLayout(self.IF2Layout)
        vbox.addWidget(self.activate)

        # Now put everything into the frame,
        # hide and reopen the default
        for w in range(vbox.count()):
            item = vbox.itemAt(w)
            if item.widget() is not None:
                item.widget().hide()
            else:
                # it is a layout, so loop again:
                for ww in range(item.layout().count()):
                    item.layout().itemAt(ww).widget().hide()
        self.setLayout(vbox)
        self.detectorCombo.show()
        self.activate.show()

    def changeBoxes(self, method):
        # This does the hiding and showing of the options as the algorithm changes
        # hide and reopen the default
        for w in range(self.layout().count()):
            item = self.layout().itemAt(w)
            if item.widget() is not None:
                item.widget().hide()
            else:
            # it is a layout, so loop again:
                for ww in range(item.layout().count()):
                    item.layout().itemAt(ww).widget().hide()
        self.detectorCombo.show()
        self.activate.show()

        if method == "instantShaper1":
            for ww in range(self.IF1Layout.count()):
                self.IF1Layout.itemAt(ww).widget().show()

        elif method == "instantShaper2":
            for ww in range(self.IF2Layout.count()):
                self.IF2Layout.itemAt(ww).widget().show()


    def getValues(self):
        method = self.detectorCombo.currentText()
        if method=="instantShaper1":
            pars = [self.IF1alpha.value()]
        elif method=="instantShaper2":
            pars = [self.IF2alpha.value(),self.IF2beta.value()]
        else:
            pars=[]
        print('pars', pars)
        return(method, pars)

class getNumberCopiesPlus(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Multiple Calls')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(300)

        self.numCopies = QSpinBox()
        self.numCopies.setRange(0,10)
        self.numCopies.setSingleStep(1)
        self.numCopies.setValue(1)

        button = QPushButton("OK")
        button.clicked.connect(self.accept)

        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("How many calls in this segment?"))
        vbox.addWidget(self.numCopies)
        vbox.addWidget(button)

        self.setLayout(vbox)

    def getValues(self):
        return self.numCopies.value()


