
# This is part of the AviaNZ interface
# Holds most of the code for the various dialog boxes

# Version 3.0 14/09/20
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2020

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

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QPointF, QTime, Qt, QSize

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import colourMaps
import SupportClasses_GUI
import SignalProc
import SupportClasses
import openpyxl
import json

pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class StartScreen(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowIcon(QIcon('img/AviaNZ.ico'))
        self.setWindowTitle('AviaNZ - Choose Task')
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.FramelessWindowHint | Qt.WindowCloseButtonHint)
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
        bclose = QtGui.QToolButton()
        bclose.setIcon(QtGui.QIcon('img/close.png'))
        bclose.setIconSize(QSize(40, 40))
        bclose.setToolTip("Close")
        bclose.setStyleSheet(btn_style)
        bclose.clicked.connect(self.reject)

        hboxclose = QHBoxLayout()
        hboxclose.addWidget(bclose, alignment=Qt.AlignRight)

        hbox = QHBoxLayout()
        hbox.addStretch(5)
        hbox.addWidget(b1)
        hbox.addStretch(4)
        hbox.addWidget(b2)
        hbox.addStretch(4)
        hbox.addWidget(b3)
        hbox.addStretch(5)
        #hbox.addWidget(b4)

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
        #b4.clicked.connect(self.utilities)

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
class Spectrogram(QDialog):
    # Class for the spectrogram dialog box
    # TODO: Steal the graph from Raven (View/Configure Brightness)
    def __init__(self, width, incr, minFreq, maxFreq, minFreqShow, maxFreqShow, window, sgtype='Standard', batmode=False, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Spectrogram Options')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)
        self.setMinimumWidth(300)

        self.windowType = QComboBox()
        self.windowType.addItems(['Hann','Parzen','Welch','Hamming','Blackman','BlackmanHarris'])
        self.windowType.setCurrentText(window)

        self.sgType = QComboBox()
        self.sgType.addItems(['Standard','Multi-tapered','Reassigned'])
        self.sgType.setCurrentText(sgtype)

        self.mean_normalise = QCheckBox()
        self.mean_normalise.setChecked(True)

        self.equal_loudness = QCheckBox()
        self.equal_loudness.setChecked(False)

        self.low = QSlider(Qt.Horizontal)
        self.low.setTickPosition(QSlider.TicksBelow)
        self.low.setTickInterval(1000)
        self.low.setSingleStep(100)
        self.low.valueChanged.connect(self.lowChange)
        self.lowtext = QLabel()
        self.lowtext.setAlignment(Qt.AlignRight)
        self.lowChange(minFreqShow)

        self.high = QSlider(Qt.Horizontal)
        self.high.setTickPosition(QSlider.TicksBelow)
        self.high.setTickInterval(1000)
        self.high.setSingleStep(100)
        self.high.valueChanged.connect(self.highChange)
        self.hightext = QLabel(str(self.high.value()) + ' Hz')
        self.hightext.setAlignment(Qt.AlignRight)
        self.highChange(maxFreqShow)

        self.labelMinF = QLabel()
        self.labelMaxF = QLabel()
        self.labelMaxF.setAlignment(Qt.AlignRight)

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
        form.addRow('Mean normalise', self.mean_normalise)
        form.addRow('Equal loudness', self.equal_loudness)
        #form.addRow('Multitapering', self.multitaper)
        #form.addRow('Reassignment', self.reassigned)
        form.addRow('Window width', self.window_width)
        form.addRow('Hop', self.incr)
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
        return [self.windowType.currentText(),self.sgType.currentText(),self.mean_normalise.checkState(),self.equal_loudness.checkState(),self.window_width.text(),self.incr.text(),low,high]

    def lowChange(self,value):
        # NOTE returned values should also use this rounding
        value = value // 100 * 100
        self.lowtext.setText(str(value)+' Hz')

    def highChange(self,value):
        value = value // 100 * 100
        self.hightext.setText(str(value)+' Hz')

    def resetValues(self):
        self.windowType.setCurrentText('Hann')

        self.mean_normalise.setChecked(True)
        self.equal_loudness.setChecked(False)
        #self.multitaper.setChecked(False)
        #self.reassigned.setChecked(False)

        self.setValues(self.low.minimum(), self.low.maximum(), self.low.minimum(), self.high.maximum())

        self.window_width.setText('256')
        self.incr.setText('128')
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
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)
        self.setMinimumWidth(700)

        self.txtExcel = QLineEdit()
        self.txtExcel.setMinimumWidth(400)
        self.txtExcel.setText('')
        self.btnBrowseExcel = QPushButton("&Browse Excel")
        self.btnBrowseExcel.setFixedWidth(220)
        self.btnBrowseExcel.clicked.connect(self.browseExcel)

        lblHeader = QLabel('Choose Columns')
        lblHeader.setFixedWidth(220)
        lblHeader.setAlignment(Qt.AlignCenter)
        self.comboStart = QComboBox()
        self.comboEnd = QComboBox()
        self.comboLow = QComboBox()
        self.comboHigh = QComboBox()

        self.txtAudio = QLineEdit()
        self.txtAudio.setMinimumWidth(400)
        self.txtAudio.setText('')
        self.btnBrowseAudio = QPushButton("Browse Corresponding Audio")
        self.btnBrowseAudio.setFixedWidth(220)
        self.btnBrowseAudio.setToolTip("Select corresponding .wav")
        self.btnBrowseAudio.clicked.connect(self.browseAudio)

        self.txtSpecies = QLineEdit()
        self.txtSpecies.setMinimumWidth(400)
        self.txtSpecies.setText('')
        lblSpecies = QLabel("Species Name")
        lblSpecies.setFixedWidth(220)
        lblSpecies.setAlignment(Qt.AlignCenter)

        self.btnGenerateAnnot = QPushButton("Generate AviaNZ Annotation")
        self.btnGenerateAnnot.setFixedHeight(50)
        self.btnGenerateAnnot.setStyleSheet('QPushButton {font-weight: bold; font-size:14px; padding: 2px 2px 2px 8px}')

        # Show a template
        tableWidget = QTableWidget()
        tableWidget.setRowCount(4)
        tableWidget.setColumnCount(4)
        tableWidget.setHorizontalHeaderLabels("A;B;C;D".split(";"))
        tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
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
        tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

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
            return [self.txtExcel.text(), self.txtAudio.text(), self.txtSpecies.text(), self.headers[self.comboStart.currentIndex()], self.headers[self.comboEnd.currentIndex()], self.headers[self.comboLow.currentIndex()], self.headers[self.comboHigh.currentIndex()]]
        else:
            msg = SupportClasses_GUI.MessagePopup("t", "All fields are Mandatory ", "All fields are Mandatory ")
            msg.exec_()
            return []

    def browseExcel(self):
        try:
            if not self.txtAudio.text():
                userDir = os.path.expanduser("~")
            else:
                userDir, _ = os.path.split(self.txtAudio.text())
            excelfile, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', userDir, "Excel (*.xlsx *.xls)")
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
            audiofile, _ = QtGui.QFileDialog.getOpenFileName(self, 'Open file', userDir, "Audio (*.wav)")
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
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)
        self.setMinimumWidth(700)

        self.txtSession = QLineEdit()
        self.txtSession.setMinimumWidth(400)
        self.txtSession.setText('')
        self.btnBrowseSession = QPushButton("&Browse Session")
        self.btnBrowseSession.setFixedWidth(220)
        self.btnBrowseSession.clicked.connect(self.browseSession)

        self.txtDuration = QLineEdit()
        self.txtDuration.setMinimumWidth(400)
        self.txtDuration.setText('')
        lblDuration = QLabel("Duration (sec) of a recording")
        lblDuration.setFixedWidth(220)
        lblDuration.setAlignment(Qt.AlignCenter)

        self.btnGenerateAnnot = QPushButton("Generate AviaNZ Annotation")
        self.btnGenerateAnnot.setFixedHeight(50)
        self.btnGenerateAnnot.setStyleSheet('QPushButton {font-weight: bold; font-size:14px; padding: 2px 2px 2px 8px}')

        Box = QVBoxLayout()
        Box.addWidget(QLabel())
        Box1 = QHBoxLayout()
        Box1.addWidget(self.btnBrowseSession)
        Box1.addWidget(self.txtSession)
        Box2 = QHBoxLayout()
        Box2.addWidget(lblDuration)
        Box2.addWidget(self.txtDuration)
        Box.addLayout(Box1)
        Box.addLayout(Box2)
        Box.addWidget(QLabel())
        Box.addWidget(self.btnGenerateAnnot)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        if self.txtDuration.text() and self.txtSession.text():
            return [self.txtSession.text(), self.txtDuration.text()]
        else:
            msg = SupportClasses_GUI.MessagePopup("t", "All fields are Mandatory ", "All fields are Mandatory ")
            msg.exec_()
            return []

    def browseSession(self):
        dirName = QFileDialog.getExistingDirectory(self, 'Choose .session folder with .tag and .setting')
        self.txtSession.setText(dirName)

#======
class BackupAnnotation(QDialog):
    # Class for XML Tag to AviaNZ annotation
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Backup annotations')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)
        self.setMinimumWidth(700)

        self.txtSrc = QLineEdit()
        self.txtSrc.setMinimumWidth(400)
        self.txtSrc.setText('')
        self.btnBrowseSrc = QPushButton("&Browse Source Directory")
        self.btnBrowseSrc.setFixedWidth(220)
        self.btnBrowseSrc.clicked.connect(self.browseSrc)

        self.txtDst = QLineEdit()
        self.txtDst.setMinimumWidth(400)
        self.txtDst.setText('')
        self.btnBrowseDst = QPushButton("&Browse Destination Directory")
        self.btnBrowseDst.setFixedWidth(220)
        self.btnBrowseDst.clicked.connect(self.browseDst)

        self.btnCopyAnnot = QPushButton("Copy Annotations")
        self.btnCopyAnnot.setFixedHeight(50)
        self.btnCopyAnnot.setStyleSheet('QPushButton {font-weight: bold; font-size:14px; padding: 2px 2px 2px 8px}')

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
            msg = SupportClasses_GUI.MessagePopup("t", "All fields are Mandatory ", "All fields are Mandatory ")
            msg.exec_()
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
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint))
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
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint))
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
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)

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

class DiagnosticCNN(QDialog):
    # Class for the diagnostic dialog box - CNN
    def __init__(self, filters, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('CNN Diagnostic Plot Options')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setMinimumWidth(300)
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)

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
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)
        self.setMinimumWidth(350)

        self.algs = QComboBox()
        if DOC:
            self.algs.addItems(["Wavelets", "FIR", "Median Clipping"])
        else:
            self.algs.addItems(["Default","Median Clipping","Fundamental Frequency","FIR","Wavelets","Harma","Power","Cross-Correlation"])
        self.algs.currentIndexChanged[str].connect(self.changeBoxes)
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
        spp.insert(0, "Choose species...")
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

        self.medSize = QSlider(Qt.Horizontal)
        self.medSize.setTickPosition(QSlider.TicksBelow)
        self.medSize.setTickInterval(100)
        self.medSize.setRange(100,2000)
        self.medSize.setSingleStep(100)
        self.medSize.setValue(1000)
        self.medSize.valueChanged.connect(self.medSizeChange)

        # Sliders for minlen and maxgap are in ms scale
        self.minlen = QSlider(Qt.Horizontal)
        self.minlen.setTickPosition(QSlider.TicksBelow)
        self.minlen.setTickInterval(0.25*1000)
        self.minlen.setRange(0.25*1000, 10*1000)
        self.minlen.setSingleStep(0.25*1000)
        self.minlen.setValue(0.5*1000)
        self.minlen.valueChanged.connect(self.minLenChange)
        self.minlenlbl = QLabel("Minimum segment length: 0.5 sec")

        self.maxgap = QSlider(Qt.Horizontal)
        self.maxgap.setTickPosition(QSlider.TicksBelow)
        self.maxgap.setTickInterval(0.25*1000)
        self.maxgap.setRange(0.25*1000, 10*1000)
        self.maxgap.setSingleStep(0.5*1000)
        self.maxgap.setValue(1*1000)
        self.maxgap.valueChanged.connect(self.maxGapChange)
        self.maxgaplbl = QLabel("Maximum gap between syllables: 1 sec")

        self.wind = QCheckBox("Remove wind")
        self.rain = QCheckBox("Remove rain")
        Box.addWidget(self.wind)
        Box.addWidget(self.rain)
        Box.addWidget(self.maxgaplbl)
        Box.addWidget(self.maxgap)
        Box.addWidget(self.minlenlbl)
        Box.addWidget(self.minlen)
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
        self.wind.show()
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
            self.maxgaplbl.hide()
            self.maxgap.hide()
            self.minlenlbl.hide()
            self.minlen.hide()

    def medSizeChange(self,value):
        self.medSizeText.setText("Minimum length: %s ms" % value)

    def minLenChange(self,value):
        self.minlenlbl.setText("Minimum segment length: %s sec" % str(round(int(value)/1000, 2)))

    def maxGapChange(self,value):
        self.maxgaplbl.setText("Maximum gap between syllables: %s sec" % str(round(int(value)/1000, 2)))

    def getValues(self):
        # TODO: check: self.medSize.value() is not used, should we keep it?
        return [self.algs.currentText(), self.medThr.text(), self.medSize.value(), self.HarmaThr1.text(),self.HarmaThr2.text(),self.PowerThr.text(),self.Fundminfreq.text(),self.Fundminperiods.text(),self.Fundthr.text(),self.Fundwindow.text(),self.FIRThr1.text(),self.CCThr1.text(),self.species.currentText(), self.species_cc.currentText(), self.wind.isChecked(), self.rain.isChecked(), int(self.maxgap.value())/1000, int(self.minlen.value())/1000]

#======
class Denoise(QDialog):
    # Class for the denoising dialog box
    def __init__(self, parent=None,DOC=True,minFreq=0,maxFreq=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Denoising Options')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)

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

    def __init__(self, lut, colourStart, colourEnd, cmapInverted, brightness, contrast, shortBirdList, longBirdList, batList, multipleBirds, audioFormat, plotAspect=2, parent=None):
        # plotAspect: initial stretch factor in the X direction
        QDialog.__init__(self, parent)
        self.setWindowTitle('Check Classifications')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

        self.setModal(True)
        self.frame = QWidget()

        self.parent = parent
        self.batmode = self.parent.batmode
        self.lut = lut
        self.label = []
        self.colourStart = colourStart
        self.colourEnd = colourEnd
        self.cmapInverted = cmapInverted
        self.shortBirdList = shortBirdList
        self.longBirdList = longBirdList
        self.batList = batList
        self.multipleBirds = multipleBirds
        self.saveConfig = False
        self.viewingct = False
        # exec_ forces the cursor into waiting

        # Set up the plot window, then the right and wrong buttons, and a close button
        # wPlot: white area around the spectrogram
        self.wPlot = SupportClasses_GUI.PartlyResizableGLW()
        self.pPlot = self.wPlot.addViewBox(enableMouse=False, row=0, col=1)
        self.plot = pg.ImageItem()
        self.pPlot.addItem(self.plot)
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
        self.guidelines = [0]*4
        self.guidelines[0] = pg.InfiniteLine(angle=0, pen={'color': (255,232,140), 'width': 2})
        self.guidelines[1] = pg.InfiniteLine(angle=0, pen={'color': (239,189,124), 'width': 2})
        self.guidelines[2] = pg.InfiniteLine(angle=0, pen={'color': (239,189,124), 'width': 2})
        self.guidelines[3] = pg.InfiniteLine(angle=0, pen={'color': (255,232,140), 'width': 2})
        for g in self.guidelines:
            self.pPlot.addItem(g)

        # time texts to go along these two lines
        self.segTimeText1 = pg.TextItem(color=(50,205,50), anchor=(0,1.10))
        self.segTimeText2 = pg.TextItem(color=(50,205,50), anchor=(0,0.75))
        self.pPlot.addItem(self.segTimeText1)
        self.pPlot.addItem(self.segTimeText2)

        # playback line
        self.bar = pg.InfiniteLine(angle=90, movable=False, pen={'color':'c', 'width': 3})
        self.bar.btn = Qt.RightButton
        self.bar.setValue(0)
        self.pPlot.addItem(self.bar)

        # label for current segment assignment
        self.speciesTop = QLabel("Currently:")
        self.species = QLabel()
        self.species.setStyleSheet("QLabel { font-size:22pt; font-weight: bold}")

        # The buttons to move through the overview
        self.numberDone = QLabel()
        self.numberLeft = QLabel()
        self.numberDone.setAlignment(Qt.AlignCenter)
        self.numberLeft.setAlignment(Qt.AlignCenter)

        iconSize = QSize(45, 45)
        self.buttonPrev = QtGui.QToolButton()
        self.buttonPrev.setIcon(QtGui.QIcon('img/undo.png'))
        self.buttonPrev.setIconSize(iconSize)
        self.buttonPrev.setStyleSheet("padding: 5px 5px 5px 5px")

        self.buttonNext = QtGui.QToolButton()
        self.buttonNext.setIcon(QtGui.QIcon('img/questionL.png'))
        self.buttonNext.setIconSize(iconSize)
        self.buttonNext.setStyleSheet("padding: 5px 5px 5px 5px")

        self.correct = QtGui.QToolButton()
        self.correct.setIcon(QtGui.QIcon('img/check-mark2.png'))
        self.correct.setIconSize(iconSize)
        self.correct.setStyleSheet("padding: 5px 5px 5px 5px")

        self.delete = QtGui.QToolButton()
        self.delete.setIcon(QtGui.QIcon('img/deleteL.png'))
        self.delete.setIconSize(iconSize)
        self.delete.setStyleSheet("padding: 5px 5px 5px 5px")

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
        # Create an array of check boxes for the most common birds
        self.birds = QButtonGroup()
        self.birdbtns = []
        if self.multipleBirds:
            self.birds.setExclusive(False)
            if self.batmode:
                for item in self.batList:
                    self.birdbtns.append(QCheckBox(item))
                    self.birds.addButton(self.birdbtns[-1],len(self.birdbtns)-1)
                    self.birdbtns[-1].clicked.connect(self.tickBirdsClicked)
            else:
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
            if self.batmode:
                for item in self.batList:
                    btn = QRadioButton(item)
                    self.birdbtns.append(btn)
                    self.birds.addButton(btn,len(self.birdbtns)-1)
                    btn.clicked.connect(self.radioBirdsClicked)
            else:
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

        # Call type label
        self.ctLabel = QLabel("")
        self.ctLabel.setStyleSheet("QLabel { font-size:16pt; font-weight: bold}")
        self.ctLabel.hide()

        # Call type buttons
        self.ctbtns = []
        for ct in range(10):
            btn = QRadioButton("")
            btn.clicked.connect(self.radioCtClicked)
            btn.hide()
            self.ctbtns.append(btn)

        # This is the text box for missing birds
        self.tbox = QLineEdit(self)
        self.tbox.setMaximumWidth(150)
        self.tbox.returnPressed.connect(self.birdTextEntered)
        self.tbox.setEnabled(False)
        self.tboxLabel1 = QLabel("If bird isn't in list, select Other")
        self.tboxLabel2 = QLabel("Type below, Return at end")

        # button to switch to call type view
        self.viewSpButton = QtGui.QToolButton()
        self.viewSpButton.setIcon(QIcon('img/splarge-ct.png'))
        self.viewSpButton.setIconSize(QSize(42, 25))
        self.viewSpButton.setToolTip("Toggle between species/calltype views")
        self.viewSpButton.clicked.connect(lambda: self.refreshCtUI(not self.viewingct))

        # Audio playback object
        self.media_obj2 = SupportClasses_GUI.ControllableAudio(audioFormat)
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

        # Also add call types to columns 1-2 (either those or SP btns will be shown)
        for btn in self.ctbtns[:5]:
            btn.hide()
            birds1Layout.addWidget(btn)
        for btn in self.ctbtns[5:]:
            btn.hide()
            birds2Layout.addWidget(btn)

        hboxBirds = QHBoxLayout()
        hboxBirds.addLayout(birds1Layout)
        hboxBirds.addLayout(birds2Layout)
        hboxBirds.addLayout(birds3Layout)
        # this hides the long list and "Add" options in batmode
        if self.batmode:
            self.birds3.hide()
            self.tboxLabel1.hide()
            self.tboxLabel2.hide()
            self.tbox.hide()
            self.viewSpButton.hide()
        else:
            birdListLayout = QGridLayout()
            birdListLayout.setRowStretch(0, 10)
            birdListLayout.setRowStretch(1, 0)
            birdListLayout.setRowStretch(2, 0)
            birdListLayout.addWidget(self.birds3, 0, 0, 1, 3)
            birdListLayout.addWidget(self.tboxLabel1, 1, 0, 1, 3)
            birdListLayout.addWidget(self.tboxLabel2, 2, 0, 1, 3)
            birdListLayout.addWidget(self.tbox, 3, 0, 1, 2)
            birdListLayout.addWidget(self.viewSpButton, 3, 2, 1, 1)
            hboxBirds.addLayout(birdListLayout)

        # The layouts
        hboxNextPrev = QHBoxLayout()
        hboxNextPrev.addWidget(self.numberDone)
        hboxNextPrev.addWidget(self.buttonPrev)
        hboxNextPrev.addWidget(self.correct)
        hboxNextPrev.addWidget(self.buttonNext)
        hboxNextPrev.addWidget(self.delete)
        hboxNextPrev.addWidget(self.numberLeft)

        self.playButton = QtGui.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(QSize(40, 40))
        self.playButton.clicked.connect(self.playSeg)

        self.scroll = QtGui.QScrollArea()
        self.scroll.setWidget(self.wPlot)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(270)

        # Volume control
        self.volSlider = QSlider(Qt.Horizontal)
        self.volSlider.valueChanged.connect(self.volSliderMoved)
        self.volSlider.setRange(0,100)
        self.volSlider.setValue(50)
        self.volIcon = QLabel()
        self.volIcon.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.volIcon.setPixmap(QPixmap('img/volume.png').scaled(18, 18, transformMode=1))

        # Brightness and contrast sliders. Need to pass true (config) values of these as args
        self.brightnessSlider = QSlider(Qt.Horizontal)
        self.brightnessSlider.setMinimum(0)
        self.brightnessSlider.setMaximum(100)
        if self.cmapInverted:
            self.brightnessSlider.setValue(brightness)
        else:
            self.brightnessSlider.setValue(100-brightness)
        self.brightnessSlider.setTickInterval(1)
        self.brightnessSlider.valueChanged.connect(self.setColourLevels)

        self.contrastSlider = QSlider(Qt.Horizontal)
        self.contrastSlider.setMinimum(0)
        self.contrastSlider.setMaximum(100)
        self.contrastSlider.setValue(contrast)
        self.contrastSlider.setTickInterval(1)
        self.contrastSlider.valueChanged.connect(self.setColourLevels)

        # zoom buttons
        self.zoomInBtn = QtGui.QToolButton()
        self.zoomOutBtn = QtGui.QToolButton()
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
        vboxSpecContr.addWidget(self.scroll, row=1, col=0, colspan=13)
        vboxSpecContr.addWidget(self.playButton, row=2, col=0)
        vboxSpecContr.addWidget(self.volIcon, row=2, col=1)
        vboxSpecContr.addWidget(self.volSlider, row=2, col=2, colspan=2)
        labelBr = QLabel()
        labelBr.setPixmap(QPixmap('img/brightstr24.png').scaled(18, 18, transformMode=1))
        labelBr.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        vboxSpecContr.addWidget(labelBr, row=2, col=4)
        vboxSpecContr.addWidget(self.brightnessSlider, row=2, col=5, colspan=2)
        labelCo = QLabel()
        labelCo.setPixmap(QPixmap('img/contrstr24.png').scaled(18, 18, transformMode=1))
        labelCo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        vboxSpecContr.addWidget(labelCo, row=2, col=7)
        vboxSpecContr.addWidget(self.contrastSlider, row=2, col=8, colspan=2)
        #spacer = QSpacerItem(1,1)
        #vboxSpecContr.layout.addWidget(spacer, row=2, col=10)
        vboxSpecContr.addWidget(self.zoomInBtn, row=2, col=11)
        vboxSpecContr.addWidget(self.zoomOutBtn, row=2, col=12)

        vboxSpecContr.layout.setColumnStretch(1, 1)
        vboxSpecContr.layout.setColumnStretch(2, 2)
        vboxSpecContr.layout.setColumnStretch(4, 1)
        vboxSpecContr.layout.setColumnStretch(5, 2)
        vboxSpecContr.layout.setColumnStretch(7, 1)
        vboxSpecContr.layout.setColumnStretch(8, 2)
        vboxSpecContr.layout.setColumnStretch(10, 1)

        vboxFull = QVBoxLayout()
        vboxFull.addLayout(spNameBox)
        vboxFull.addWidget(vboxSpecContr)
        vboxFull.addLayout(hboxBirds)
        vboxFull.addSpacing(7)
        vboxFull.addLayout(hboxNextPrev)

        self.setLayout(vboxFull)
        # print seg
        # self.setImage(self.sg,audiodata,sampleRate,self.label, unbufStart, unbufStop)

    def playSeg(self):
        if self.media_obj2.isPlaying():
            self.stopPlayback()
        else:
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
            self.playButton.setIconSize(QSize(40, 40))
            self.media_obj2.loadArray(self.audiodata)

    def stopPlayback(self):
        self.media_obj2.pressedStop()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(QSize(40, 40))

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
        # resize the ViewBox with spec, lines, axis
        self.plotAspect = self.plotAspect * 1.5
        # self.pPlot.setAspectLocked(ratio=self.plotAspect)
        xyratio = np.shape(self.sg)
        # self.pPlot.setYRange(0, xyratio[1], padding=0.02)
        xyratio = xyratio[0] / xyratio[1]
        # resize the white area around the spectrogram if it's under 500
        self.wPlot.plotAspect = self.plotAspect * xyratio
        self.wPlot.forceResize()

    def zoomOut(self):
        self.plotAspect = self.plotAspect / 1.5
        # self.pPlot.setAspectLocked(ratio=self.plotAspect)
        xyratio = np.shape(self.sg)
        # self.pPlot.setYRange(0, xyratio[1], padding=0.02)
        xyratio = xyratio[0] / xyratio[1]
        # resize the white area around the spectrogram if it's under 500
        self.wPlot.plotAspect = self.plotAspect * xyratio
        self.wPlot.forceResize()

    def updateButtonList(self):
        # refreshes bird button names
        # to be used when bird list updates
        if self.batmode:
            for i in range(len(self.birdbtns)):
                self.birdbtns[i].setChecked(False)
                self.birdbtns[i].setText(self.batList[i])
        else:
            for i in range(len(self.birdbtns)-1):
                self.birdbtns[i].setChecked(False)
                self.birdbtns[i].setText(self.shortBirdList[i])
        # "other" button
        self.birdbtns[-1].setChecked(False)
        self.birds3.setEnabled(False)

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

    def setImage(self, sg, audiodata, sampleRate, incr, labels, unbufStart, unbufStop, time1, time2, guides=None, minFreq=0, maxFreq=0):
        """ labels - simply seg[4] of the current segment.
            Be careful not to edit it, as it is NOT a deep copy!!
            Used for extracting current species and calltype.
            During review, this updates self.label and self.ctLabel.
        """
        self.audiodata = audiodata
        self.sg = sg
        self.sampleRate = sampleRate
        self.incr = incr
        self.bar.setValue(0)
        if maxFreq==0:
            maxFreq = sampleRate / 2
        self.duration = len(audiodata) / sampleRate * 1000  # in ms

        # Update UI if no audio (e.g. batmode)
        self.playButton.setEnabled(len(audiodata))
        self.volIcon.setEnabled(len(audiodata))
        self.volSlider.setEnabled(len(audiodata))

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
        ticks = [(0,minFreq/1000.), (SgSize/4, minFreq/1000.+FreqRange/4.), (SgSize/2, minFreq/1000.+FreqRange/2.), (3*SgSize/4, minFreq/1000.+3*FreqRange/4.), (SgSize,minFreq/1000.+FreqRange)]
        ticks = [[(tick[0], "%.1f" % tick[1] ) for tick in ticks]]
        self.sg_axis.setTicks(ticks)
        self.sg_axis.setLabel('kHz')
        #self.sg_axis2.setTicks(ticks)
        #self.sg_axis2.setLabel('kHz')

        self.show()

        # self.pPlot.setYRange(0, SgSize, padding=0.02)
        self.pPlot.setRange(xRange=(0, np.shape(sg2)[0]), yRange=(0, SgSize))
        xyratio = np.shape(sg2)
        xyratio = xyratio[0] / xyratio[1]
        # self.plotAspect = 2 for x/y pixel aspect ratio
        self.wPlot.plotAspect = self.plotAspect * xyratio
        self.wPlot.forceResize()

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

        # bat mode freq guides
        if guides is not None:
            for i in range(len(self.guidelines)):
                self.guidelines[i].setPos(guides[i])
        else:
            for i in range(len(self.guidelines)):
                self.guidelines[i].setPos(-100)

        if self.cmapInverted:
            self.plot.setLevels([self.colourEnd, self.colourStart])
        else:
            self.plot.setLevels([self.colourStart, self.colourEnd])

        # DEAL WITH SPECIES NAMES
        # extract a string of current species names
        specnames = []
        for lab in labels:
            if 0<lab["certainty"]<100:
                specnames.append(lab["species"]+'?')
            else:
                specnames.append(lab["species"])
        specnames = list(set(specnames))

        # Update the "currently" labels
        self.species.setText(','.join(specnames))

        # extract the call type of the (first) species
        if "calltype" in labels[0]:
            self.ctLabel.setText(labels[0]["calltype"])
        else:
            self.ctLabel.setText("")

        # question marks are displayed on the first pass,
        # but any clicking sets certainty to 100 in effect.
        for lsp_ix in range(len(specnames)):
            if specnames[lsp_ix] != "-To Be Deleted-":
                if specnames[lsp_ix].endswith('?'):
                    specnames[lsp_ix] = specnames[lsp_ix][:-1]
                # move the label to the top of the list
                if self.parent.config['ReorderList']:
                    if self.batmode:
                        if specnames[lsp_ix] in self.batList:
                            self.batList.remove(specnames[lsp_ix])
                        else:
                            del self.batList[-1]
                        self.batList.insert(0, specnames[lsp_ix])
                    else:
                        if specnames[lsp_ix] in self.shortBirdList:
                            self.shortBirdList.remove(specnames[lsp_ix])
                        else:
                            del self.shortBirdList[-1]
                        self.shortBirdList.insert(0, specnames[lsp_ix])

        # clear selection
        self.birds3.clearSelection()
        self.updateButtonList()
        # Select the right species tickboxes / buttons
        for lsp in specnames:
            if lsp != "-To Be Deleted-":
                # add ticks to the right checkboxes
                if self.batmode:
                    ind = self.batList.index(lsp)
                    self.birdbtns[ind].setChecked(True)

                    # since there is no long list or birds3 box, we ignore those parts.
                else:
                    if lsp in self.shortBirdList[:29]:
                        ind = self.shortBirdList.index(lsp)
                        self.birdbtns[ind].setChecked(True)
                    else:
                        self.birdbtns[29].setChecked(True)
                        self.birds3.setEnabled(True)

                    # mark this species in the long list box
                    if lsp not in self.longBirdList:
                        # try genus>species instead of genus (species)
                        if '(' in lsp:
                            ind = lsp.index('(')
                            lsp = lsp[:ind-1] + ">" + lsp[ind+1:-1]
                        # add to long bird list then
                        if lsp not in self.longBirdList:
                            print("Species", lsp, "not found in long bird list, adding")
                            self.longBirdList.append(lsp)
                            cc = self.birds3.count()
                            self.birds3.insertItem(cc-1, lsp)
                            self.saveConfig = True

                    # all species by now are in the long bird list
                    if self.longBirdList is not None:
                        ind = self.longBirdList.index(lsp)
                        self.birds3.item(ind).setSelected(True)

        self.label = specnames

        # Determine if we can review call types based on selected annotations
        self.checkCallTypes()

    def checkCallTypes(self):
        # parses current annotations to determine if call type review allowed
        # should be allowed, updates GUI accordingly
        spWithCalltypes = [filt["species"] for filt in self.parent.FilterDicts.values()]
        self.viewSpButton.setEnabled(False)
        showCt = False
        if len(self.label)>1:
            self.viewSpButton.setToolTip("Cannot review call types when >1 species marked")
        elif self.label[0]=="Don't Know" or self.label[0]=="-To Be Deleted-":
            self.viewSpButton.setToolTip("No call types possible without species marked")
        elif self.label[0] not in spWithCalltypes:
            self.viewSpButton.setToolTip("Cannot review call types as this species has no recogniser")
        else:
            self.viewSpButton.setToolTip("Toggle between species/calltype views")
            self.viewSpButton.setEnabled(True)
            # keep UI in the same SP/CT mode as it was before
            showCt = self.viewingct

        # this part actually refreshes the buttons (even if mode is the same, names
        # or number of cts may have changed)
        self.refreshCtUI(showCt)

    def refreshCtUI(self, showCt):
        # Hides/shows species/calltype checkboxes
        # and updates calltype button lists (in case sp changed)
        # will also update self.viewingct to reflect current state
        if not (self.viewingct or showCt):
            # everything was hidden and stays hidden
            return
        elif showCt:
            # show calltype, hide species boxes

            # extract all possible call types for selected species
            possibleCts = set()
            for filt in self.parent.FilterDicts.values():
                if filt["species"]==self.label[0]:
                    ctsinthisfilt = [subf["calltype"] for subf in filt["Filters"]]
                    possibleCts.update(ctsinthisfilt)
            possibleCts = list(possibleCts)

            # show and update call type names to match selected species
            for cti in range(len(possibleCts)):
                btn = self.ctbtns[cti]
                btn.show()
                btn.setText(possibleCts[cti])
                # (temp disable so that we could have all buttons unchecked if no CT selected)
                btn.setAutoExclusive(False)
                # mark current call type:
                if possibleCts[cti]==self.ctLabel.text():
                    btn.setChecked(True)
                else:
                    btn.setChecked(False)
                btn.setAutoExclusive(True)
            self.ctLabel.show()
            # hide remaining CT buttons in case more were shown before
            for ctbtn in self.ctbtns[len(possibleCts):]:
                ctbtn.hide()
                ctbtn.setChecked(False)
            for spbtn in self.birdbtns:
                spbtn.hide()
            self.birds3.hide()
            self.tbox.hide()
            self.tboxLabel1.hide()
            self.tboxLabel2.hide()
            self.viewSpButton.setIcon(QIcon('img/sp-ctlarge.png'))
        else:
            # show species, hide call type boxes
            for ctbtn in self.ctbtns:
                ctbtn.hide()
            self.ctLabel.hide()
            for spbtn in self.birdbtns:
                spbtn.show()
            if not self.batmode:
                self.birds3.show()
                self.tbox.show()
                self.tboxLabel1.show()
                self.tboxLabel2.show()
                self.viewSpButton.setIcon(QIcon('img/splarge-ct.png'))

        self.viewingct = showCt

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
            print("Warning: unrecognised check event")
            return
        if checkedButton.isChecked():
            # if label was empty, just change from DontKnow:
            if self.label == ["Don't Know"] or self.label == ["-To Be Deleted-"]:
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

        # enable/disable CT mode button
        self.checkCallTypes()

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

        # enable/disable CT mode button
        self.checkCallTypes()

    def radioCtClicked(self):
        # Listener for when the user selects a call type
        # Update the label and store the new CT as a property on self
        # to be read when self...Correct is called
        for btn in self.ctbtns:
            if btn.isChecked():
                print("Call type selected:", btn.text())
                self.ctLabel.setText(btn.text())
                break

    def listBirdsClicked(self, item):
        # Listener for clicks in the listbox of birds
        # check for a corresponding button in the short list
        # - if the user is silly enough to click here when it's there as well
        checkedButton = None
        dontknowButton = None
        for button in self.birds.buttons():
            if button.text() == item.text() and button.text() != "Other":
                checkedButton = button
            if button.text() == "Don't Know":
                dontknowButton = button

        if (item.text() == "Other"):
            self.tbox.setEnabled(True)
        else:
            # Save the entry
            self.tbox.setEnabled(False)
            if self.multipleBirds:
                # mark this
                if item.isSelected() and item.text() not in self.label:
                    # if label was empty, just change from DontKnow:
                    if self.label == ["Don't Know"]:
                        self.label = [str(item.text())]
                        if dontknowButton is not None:
                            dontknowButton.setChecked(False)
                    else:
                        self.label.append(str(item.text()))
                        if checkedButton is not None:
                            checkedButton.setChecked(True)

                # unmark this
                if not item.isSelected() and item.text() in self.label:
                    self.label.remove(str(item.text()))
                    if checkedButton is not None:
                        checkedButton.setChecked(False)
                # if this erased everything, revert to don't know:
                if self.label == []:
                    self.label = ["Don't Know"]
                    if dontknowButton is not None:
                        dontknowButton.setChecked(True)
            else:
                self.label = [str(item.text())]
                # for radio buttons, only "Other" will be selected already

            self.species.setText(','.join(self.label))

        # enable/disable CT mode button
        self.checkCallTypes()

    def birdTextEntered(self):
        # Listener for the text entry in the bird list
        # Check text isn't already in the listbox, and add if not
        # Then calls the usual handler for listbox selections
        textitem = self.tbox.text()
        if textitem.lower()=="don't know" or textitem.lower()=="other":
            print("ERROR: provided name %s is reserved, cannot create" % textitem)
            return
        if "?" in textitem:
            print("ERROR: provided name %s contains reserved symbol '?'" % textitem)
            return
        if len(textitem)==0 or len(textitem)>150:
            print("ERROR: provided name appears to be too short or too long")
            return

        item = self.birds3.findItems(textitem, Qt.MatchExactly)

        if not item:
            self.birds3.addItem(textitem)
            item = self.birds3.findItems(textitem, Qt.MatchExactly)

        item[0].setSelected(True)
        # this will deal with updating the label and buttons
        self.listBirdsClicked(item[0])

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
        # Note: we are reading off the calltypes from the label
        # and not from radio buttons, b/c when no ct is present,
        # radio buttons cannot be all disabled.
        return [self.label, self.saveConfig, self.tbox.text(), self.ctLabel.text()]


class HumanClassify2(QDialog):
    """ Single Species review main dialog.
        Puts all segments of a certain species together on buttons, and their labels.
        Allows quick confirm/leave/delete check over many segments.

        Construction:
        1. a list of SignalProcs containing spectrograms for ALL the segments in arg2
        2. SegmentList. Just provide full versions of this,
          and this dialog will select the needed segments.
        3. indices of segments to show (i.e. the selected species and current page)
        4. name of the species that we are reviewing
        5-10. spec color parameters
        11-12. guide positions for batmode
        13. Filename - just for setting the window title
    """

    def __init__(self, sps, segments, indicestoshow, label, lut, colourStart, colourEnd, cmapInverted, brightness, contrast, guidefreq=None, filename=None):
        QDialog.__init__(self)

        if len(segments)==0:
            print("No segments provided")
            return

        if filename:
            self.setWindowTitle('Human review - ' + filename)
        else:
            self.setWindowTitle('Human review')

        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        # let the user quit without bothering rest of it

        self.sps = sps
        # Check if playback is possible (e.g. for batmode):
        haveaudio = all([len(sp.data)>0 for sp in sps if sp is not None])

        self.lut = lut
        self.colourStart = colourStart
        self.colourEnd = colourEnd
        self.cmapInverted = cmapInverted

        # filter segments for the requested species
        self.segments = segments
        self.indices2show = indicestoshow
        # CHECK: do we need this?
        # self.sampleRate = sampleRate
        # self.audiodata = audiodata
        # for i in reversed(self.indices2show):
        #     # show segments which have midpoint in this page (ensures all are shown only once)
        #     mid = (segments[i][0] + segments[i][1]) / 2
        #     if mid < startRead or mid > startRead + len(audiodata)//sampleRate:
        #         self.indices2show.remove(i)

        self.errors = []

        # Volume control
        self.volSlider = QSlider(Qt.Horizontal)
        self.volSlider.valueChanged.connect(self.volSliderMoved)
        self.volSlider.setRange(0,100)
        self.volSlider.setValue(50)
        self.volIcon = QLabel()
        self.volIcon.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.volIcon.setPixmap(QPixmap('img/volume.png').scaled(18, 18, transformMode=1))
        #self.volIcon.setStyleSheet("padding: 0px 1px 0px 8px")

        # batmode customizations:
        self.guidefreq = guidefreq
        if not haveaudio:
            self.volSlider.setEnabled(False)
            self.volIcon.setEnabled(False)

        # Brightness and contrast sliders - need to pass true (config) values of these as args
        self.brightnessSlider = QSlider(Qt.Horizontal)
        self.brightnessSlider.setMinimum(0)
        self.brightnessSlider.setMaximum(100)
        if self.cmapInverted:
            self.brightnessSlider.setValue(brightness)
        else:
            self.brightnessSlider.setValue(100-brightness)
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
        labelBr = QLabel()
        labelBr.setPixmap(QPixmap('img/brightstr24.png').scaled(18, 18, transformMode=1))
        labelBr.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        #labelBr.setStyleSheet("padding: 0px 1px 0px 12px")
        hboxSpecContr.addWidget(labelBr)
        hboxSpecContr.addWidget(self.brightnessSlider)
        labelCo = QLabel()
        labelCo.setPixmap(QPixmap('img/contrstr24.png').scaled(18, 18, transformMode=1))
        labelCo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        #labelCo.setStyleSheet("padding: 0px 1px 0px 12px")
        hboxSpecContr.addWidget(labelCo)
        hboxSpecContr.addWidget(self.contrastSlider)

        hboxSpecContr.setStretch(0, 1)
        hboxSpecContr.setStretch(1, 2)
        hboxSpecContr.setStretch(2, 1)
        hboxSpecContr.setStretch(3, 2)
        hboxSpecContr.setStretch(4, 1)
        hboxSpecContr.setStretch(5, 2)
        hboxSpecContr.addStretch(3)

        label1 = QLabel('Click on the images that are incorrectly labelled.')
        label1.setFont(QtGui.QFont('SansSerif', 10))
        species = QLabel(label)
        species.setStyleSheet("padding: 2px 0px 5px 0px")
        font = QtGui.QFont('SansSerif', 12)
        font.setBold(True)
        species.setFont(font)

        # species label and sliders
        vboxTop = QVBoxLayout()
        vboxTop.addWidget(label1)
        vboxTop.addWidget(species)
        vboxTop.addLayout(hboxSpecContr)

        # Controls at the bottom
        # self.buttonPrev = QtGui.QToolButton()
        # self.buttonPrev.setArrowType(Qt.LeftArrow)
        # self.buttonPrev.setIconSize(QSize(30,30))
        # self.buttonPrev.clicked.connect(self.prevPage)

        # TODO: Is this useful?
        self.pageLabel = QLabel()

        self.none = QPushButton("Toggle all")
        self.none.setSizePolicy(QSizePolicy(5,5))
        self.none.setMaximumSize(250, 30)
        self.none.clicked.connect(self.toggleAll)

        # Either the next or finish button is visible. They have different internal
        # functionality, but look the same to the user
        self.next = QPushButton("Next")
        self.next.setSizePolicy(QSizePolicy(5,5))
        self.next.setMaximumSize(250, 30)
        self.next.clicked.connect(self.nextPage)

        self.finish = QPushButton("Next")
        self.finish.setSizePolicy(QSizePolicy(5,5))
        self.finish.setMaximumSize(250, 30)

        # movement buttons and page numbers
        self.vboxBot = QHBoxLayout()
        # vboxBot.addWidget(self.buttonPrev)
        # vboxBot.addSpacing(20)
        self.vboxBot.addWidget(self.pageLabel)
        self.vboxBot.addSpacing(20)
        self.vboxBot.addWidget(self.none)
        self.vboxBot.addWidget(self.next)

        # set up the middle section of images
        self.numPicsV = 0
        self.numPicsH = 0
        # create the button objects, and we'll show them as needed
        # (fills self.buttons)
        # self.flowLayout = QGridLayout()
        self.flowLayout = pg.LayoutWidget()
        self.flowLayout.layout.setAlignment(Qt.AlignLeft | Qt.AlignBottom)

        # these sizes ensure at least one image fits:
        self.specV = 0
        self.specH = 0
        self.createButtons()

        # sets a lot of self properties needed before showing anything
        self.butStart = 0
        self.countPages()

        self.flowLayout.setMinimumSize(self.specH+20, self.specV+20)

        # Freq axes
        self.flowAxes = pg.LayoutWidget()
        self.flowAxes.setMinimumSize(70, self.specV+20)
        self.flowAxes.setSizePolicy(0, 5)
        # Time axes
        self.flowAxesT = pg.LayoutWidget()
        self.flowAxesT.setMinimumSize(self.specH+20, 40)
        self.flowAxesT.setSizePolicy(5, 0)
        self.flowAxesT.layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        gridFlowAndAxes = QGridLayout()
        gridFlowAndAxes.addWidget(self.flowAxes, 0, 0)
        gridFlowAndAxes.addWidget(self.flowLayout, 0, 1)
        gridFlowAndAxes.addWidget(self.flowAxesT, 1, 1)
        gridFlowAndAxes.setRowStretch(0, 10)
        gridFlowAndAxes.setRowStretch(1, 0)

        # set overall layout of the dialog
        self.vboxFull = QVBoxLayout()
        # self.vboxFull.setSpacing(0)
        self.vboxFull.addLayout(vboxTop)
        self.vboxSpacer = QSpacerItem(1,1, 5, 5)
        self.vboxFull.addItem(self.vboxSpacer)
        self.vboxFull.addLayout(gridFlowAndAxes)
        self.vboxFull.addLayout(self.vboxBot)
        # must be fixed size!
        vboxTop.setSizeConstraint(QLayout.SetFixedSize)
        # must be fixed size!
        self.vboxBot.setSizeConstraint(QLayout.SetFixedSize)

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
            # This will contain pre-made slices of spec and audio
            sp = self.sps[i]
            duration = len(sp.data)/sp.sampleRate

            # Seems that image is backwards?
            sp.sg = np.fliplr(sp.sg)
            self.minsg = 1
            self.maxsg = 1
            self.minsg = min(self.minsg, np.min(sp.sg))
            self.maxsg = max(self.maxsg, np.max(sp.sg))

            # batmode guides, in y of this particular spectrogram:
            if self.guidefreq is not None:
                gy = [0]*len(self.guidefreq)
                for gix in range(len(self.guidefreq)):
                    gy[gix] = sp.convertFreqtoY(self.guidefreq[gix])
            else:
                gy = None

            # create the button:
            # args: index, sp, audio, format, duration, ubstart, ubstop (in spec units)
            newButton = SupportClasses_GUI.PicButton(i, sp.sg, sp.data, sp.audioFormat, duration, sp.x1nobspec, sp.x2nobspec, self.lut, self.colourStart, self.colourEnd, self.cmapInverted, guides=gy)
            if newButton.im1.size().width() > self.specH:
                self.specH = newButton.im1.size().width()
            if newButton.im1.size().height() > self.specV:
                self.specV = newButton.im1.size().height()

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
        spaceH = self.flowLayout.size().width()

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
        # create one frequency axis
        # (all of them are identical b/c only 1 file shown at each time)
        exampleSP = self.sps[self.indices2show[0]]
        minFreq = exampleSP.minFreqShow
        maxFreq = exampleSP.maxFreqShow
        if maxFreq==0:
            maxFreq = exampleSP.sampleRate // 2
        #SgSize = np.shape(exampleSP.sg)[1]  # in spec units
        SgSize = self.specV
        if len(exampleSP.data)>0:
            duration = len(exampleSP.data)/exampleSP.sampleRate
        else:
            duration = exampleSP.convertSpectoAmpl(np.shape(exampleSP.sg)[0])
        print("Found duration", duration, "px", self.specH)

        butNum = 0
        for row in range(self.numPicsV):
            # add a frequency axis
            # args: spectrogram height in spec units, min and max frq in kHz for axis ticks
            # print(self.numPicsV)
            sg_axis = SupportClasses_GUI.AxisWidget(SgSize, minFreq/1000, maxFreq/1000)
            self.flowAxes.addWidget(sg_axis, row, 0)
            self.flowAxes.layout.setRowMinimumHeight(row, self.specV+10)

            # draw a row of buttons
            for col in range(1, self.numPicsH+1):
                if row==0:
                    time_axis = SupportClasses_GUI.TimeAxisWidget(self.specH, duration)
                    self.flowAxesT.addWidget(time_axis, 0, col)
                    self.flowAxesT.layout.setColumnMinimumWidth(col, self.specH+10)
                    time_axis.show()

                # resizing shouldn't change which segments are displayed,
                # so we use a fixed start point for counting shown buttons.
                self.flowLayout.addWidget(self.buttons[self.butStart+butNum], row, col)
                # just in case, update the bounds of grid on every redraw
                self.flowLayout.layout.setColumnMinimumWidth(col, self.specH+10)
                self.flowLayout.layout.setRowMinimumHeight(row, self.specV+10)
                self.buttons[self.butStart+butNum].show()
                butNum += 1

                if self.butStart+butNum==len(self.buttons):
                    # stop if we are out of segments
                    break
            if self.butStart+butNum==len(self.buttons):
                # stop if we are out of segments
                break

        self.repaint()
        pg.QtGui.QApplication.processEvents()

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
        buttonsPerPage = self.numPicsV * self.numPicsH
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
        for btn in self.buttons:
            btn.stopPlayback()
        # clear pic buttons
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

        # clear freq axes
        for axnum in reversed(range(self.flowAxes.layout.count())):
            item = self.flowAxes.layout.itemAt(axnum)
            if item is not None:
                self.flowAxes.layout.removeItem(item)
                r,c = self.flowAxes.items[item.widget()]
                self.flowAxes.layout.setRowMinimumHeight(r, 1)
                del self.flowAxes.rows[r][c]
                item.widget().hide()
        self.flowAxes.update()

        # clear time axes
        for axnum in reversed(range(self.flowAxesT.layout.count())):
            item = self.flowAxesT.layout.itemAt(axnum)
            if item is not None:
                self.flowAxesT.layout.removeItem(item)
                r,c = self.flowAxesT.items[item.widget()]
                self.flowAxesT.layout.setColumnMinimumWidth(r, 1)
                del self.flowAxesT.rows[r][c]
                item.widget().hide()
        self.flowAxesT.update()

    def toggleAll(self):
        buttonsPerPage = self.numPicsV * self.numPicsH
        for butNum in range(self.butStart,min(self.butStart+buttonsPerPage,len(self.buttons))):
            self.buttons[butNum].changePic(False)
        #self.update()
        self.repaint()
        pg.QtGui.QApplication.processEvents()

    def setColourLevels(self):
        """ Listener for the brightness and contrast sliders being changed. Also called when spectrograms are loaded, etc.
        Translates the brightness and contrast values into appropriate image levels.
        Calculation is simple.
        """
        if self.cmapInverted:
            brightness = self.brightnessSlider.value()
        else:
            brightness = 100-self.brightnessSlider.value()
        contrast = self.contrastSlider.value()
        colourStart = (brightness / 100.0 * contrast / 100.0) * (self.maxsg - self.minsg) + self.minsg
        colourEnd = (self.maxsg - self.minsg) * (1.0 - contrast / 100.0) + colourStart
        for btn in self.buttons:
            btn.stopPlayback()
            btn.setImage(self.lut, colourStart, colourEnd, self.cmapInverted)
            btn.update()


class FilterManager(QDialog):
    def __init__(self, filtdir, parent=None):
        super(FilterManager, self).__init__(parent)
        self.setWindowTitle("Manage recognisers")
        self.setWindowIcon(QIcon('img/Avianz.ico'))

        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)
        self.filtdir = filtdir

        # filter dir name
        labDirName = QLineEdit()
        labDirName.setText(filtdir)
        labDirName.setReadOnly(True)
        labDirName.setFocusPolicy(Qt.NoFocus)
        labDirName.setStyleSheet("background-color: #e0e0e0")

        # filter dir contents
        self.listFiles = QListWidget()
        self.listFiles.setMinimumWidth(150)
        self.listFiles.setMinimumHeight(275)
        self.listFiles.setSelectionMode(QAbstractItemView.SingleSelection)

        self.readContents()

        # rename a filter
        self.enterFiltName = QLineEdit()

        class FiltValidator(QValidator):
            def validate(self, input, pos):
                if not input.endswith('.txt'):
                    input = input+'.txt'
                if input==".txt" or input=="":
                    return(QValidator.Intermediate, input, pos)
                if self.listFiles.findItems(input, Qt.MatchExactly):
                    print("duplicated input", input)
                    return(QValidator.Intermediate, input, pos)
                else:
                    return(QValidator.Acceptable, input, pos)

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
        if "CNN" in currfilt:
            sources.append(os.path.join(self.filtdir, currfilt["CNN"]["CNN_name"] + ".h5"))
            # bat filters do not have jsons:
            if os.path.isfile(os.path.join(self.filtdir, currfilt["CNN"]["CNN_name"] + ".json")):
                sources.append(os.path.join(self.filtdir, currfilt["CNN"]["CNN_name"] + ".json"))

        for src in sources:
            if not os.path.isfile(src):
                print("ERROR: unable to delete, bad source", src)
                return

        msg = SupportClasses_GUI.MessagePopup("w", "Confirm delete", "Warning: you are about to permanently delete recogniser %s.\nAre you sure?" % sources[0])
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        reply = msg.exec_()
        if reply != QMessageBox.Yes:
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
        source, _ = QtGui.QFileDialog.getOpenFileName(self, 'Select the downloaded recogniser file', os.path.expanduser("~"), "Text files (*.txt)")
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
            if  "CNN" in filt:
                sources.append(os.path.join(os.path.dirname(source), filt["CNN"]["CNN_name"] + ".h5"))
                targets.append(os.path.join(self.filtdir, filt["CNN"]["CNN_name"] + ".h5"))
                # bat filters do not have jsons:
                JSONsource = os.path.join(os.path.dirname(source), filt["CNN"]["CNN_name"] + ".json")
                if os.path.isfile(JSONsource):
                    sources.append(JSONsource)
                    targets.append(os.path.join(self.filtdir, filt["CNN"]["CNN_name"] + ".json"))
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
                    msg.setStandardButtons(QMessageBox.NoButton)
                    msg.addButton("Overwrite", QMessageBox.YesRole)
                    msg.addButton("Skip", QMessageBox.RejectRole)
                    reply = msg.exec_()
                if reply==0:
                    # no problems, or chose to overwrite
                    print("Copying", sources[i], "->", targets[i])
                    shutil.copy2(sources[i], targets[i])
                elif reply==4194304:
                    # cancelled the entire copy
                    return
            msg = SupportClasses_GUI.MessagePopup("d", "Successfully imported","Import complete. Now you can use the recogniser %s" % os.path.basename(targets[0]))
            msg.exec_()
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
        if "CNN" in currfilt:
            sources.append(currfilt["CNN"]["CNN_name"] + ".h5")
            # bat filters do not have jsons:
            if os.path.isfile(currfilt["CNN"]["CNN_name"] + ".json"):
                sources.append(currfilt["CNN"]["CNN_name"] + ".json")

        target = QtGui.QFileDialog.getExistingDirectory(self, 'Choose where to save the recogniser')
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
                msg.exec_()
            except Exception as e:
                print("ERROR: failed to export")
                print(e)
                return

class Cluster(QDialog):
    def __init__(self, segments, sampleRate, classes, config, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Clustered segments')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)

        if len(segments) == 0:
            print("No segments provided")
            return

        self.sampleRate = sampleRate
        self.segments = segments
        self.nclasses = classes
        self.config = config

        # Volume control
        self.volSlider = QSlider(Qt.Horizontal)
        self.volSlider.valueChanged.connect(self.volSliderMoved)
        self.volSlider.setRange(0, 100)
        self.volSlider.setValue(50)
        volIcon = QLabel()
        volIcon.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        volIcon.setPixmap(self.style().standardIcon(QtGui.QStyle.SP_MediaVolume).pixmap(32))

        # Brightness, and contrast sliders
        labelBr = QLabel(" Bright.")
        self.brightnessSlider = QSlider(Qt.Horizontal)
        self.brightnessSlider.setMinimum(0)
        self.brightnessSlider.setMaximum(100)
        self.brightnessSlider.setValue(20)
        self.brightnessSlider.setTickInterval(1)
        self.brightnessSlider.valueChanged.connect(self.setColourLevels)

        labelCo = QLabel("Contr.")
        self.contrastSlider = QSlider(Qt.Horizontal)
        self.contrastSlider.setMinimum(0)
        self.contrastSlider.setMaximum(100)
        self.contrastSlider.setValue(20)
        self.contrastSlider.setTickInterval(1)
        self.contrastSlider.valueChanged.connect(self.setColourLevels)

        hboxSpecContr = QHBoxLayout()
        hboxSpecContr.addWidget(labelBr)
        hboxSpecContr.addWidget(self.brightnessSlider)
        hboxSpecContr.addWidget(labelCo)
        hboxSpecContr.addWidget(self.contrastSlider)
        hboxSpecContr.addWidget(volIcon)
        hboxSpecContr.addWidget(self.volSlider)

        # set up the images
        self.flowLayout = pg.LayoutWidget()
        self.flowLayout.setGeometry(QtCore.QRect(0, 0, 380, 247))

        self.scrollArea = QtGui.QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.flowLayout)

        # set overall layout of the dialog
        self.vboxFull = QVBoxLayout()
        self.vboxFull.addLayout(hboxSpecContr)
        self.vboxFull.addWidget(self.scrollArea)
        self.setLayout(self.vboxFull)

        # Add the clusters to rows
        self.addButtons()
        self.updateButtons()

    def addButtons(self):
        """ Only makes the PicButtons and self.clusters dict
        """
        self.clusters = []
        self.picbuttons = []
        for i in range(self.nclasses):
            self.clusters.append((i, 'Type_' + str(i)))
        self.clusters = dict(self.clusters)  # Dictionary of {ID: cluster_name}

        # Create the buttons for each segment
        for seg in self.segments:
            sp = SignalProc.SignalProc(512, 256)
            sp.readWav(seg[0], seg[1][1] - seg[1][0], seg[1][0])
            sgRaw = sp.spectrogram(window='Hann', sgType='Standard',mean_normalise=True, onesided=True, need_even=False)
            maxsg = np.min(sgRaw)
            self.sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))
            self.setColourMap()

            sg = self.sg

            newButton = SupportClasses_GUI.PicButton(1, np.fliplr(sg), sp.data, sp.audioFormat, seg[1][1] - seg[1][0], 0, seg[1][1],
                                          self.lut, self.colourStart, self.colourEnd, False, cluster=True)
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
            tbox.setAlignment(Qt.AlignCenter)
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
        for btn in self.picbuttons:
            btn.stopPlayback()
            btn.setImage(self.lut, colourStart, colourEnd, False)
            btn.update()

    def volSliderMoved(self, value):
        # try/pass to avoid race situations when smth is not initialized
        try:
            for btn in self.picbuttons:
                btn.media_obj.applyVolSlider(value)
        except Exception:
            pass

    def setColourMap(self):
        """ Listener for the menu item that chooses a colour map.
        Loads them from the file as appropriate and sets the lookup table.
        """
        cmap = self.config['cmap']

        pos, colour, mode = colourMaps.colourMaps(cmap)

        cmap = pg.ColorMap(pos, colour,mode)
        self.lut = cmap.getLookupTable(0.0, 1.0, 256)
        minsg = np.min(self.sg)
        maxsg = np.max(self.sg)
        self.colourStart = (self.config['brightness'] / 100.0 * self.config['contrast'] / 100.0) * (maxsg - minsg) + minsg
        self.colourEnd = (maxsg - minsg) * (1.0 - self.config['contrast'] / 100.0) + self.colourStart

class ExportBats(QDialog):
    def __init__(self,observer):
        QDialog.__init__(self)
        self.setWindowTitle('Export Results?')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)

        l1 = QLabel('Do you want to export an entry for the National Bat Database?\n(It will be saved at the top level of the folder with the recordings in as BatDB.csv, you will need to email it yourself\nFields with a * are mandatory)\n')
        l2 = QLabel('*Data source (e.g., your community group): ')
        self.data = QLineEdit(self)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(l2)
        hbox1.addWidget(self.data)
        l3 = QLabel('*Your name: ')
        self.observer = QLineEdit(self)
        self.observer.setText(observer)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(l3)
        hbox2.addWidget(self.observer)
        l4a = QLabel('Method: ')
        self.method = QLineEdit(self)
        l4 = QLabel('Detector Type: ')
        self.detector = QLineEdit(self)
        self.detector.setText('ABM')
        hbox3 = QHBoxLayout()
        hbox3.addWidget(l4a)
        hbox3.addWidget(self.method)
        hbox3.addWidget(l4)
        hbox3.addWidget(self.detector)
        l5 = QLabel('Any notes: ')
        self.notes = QLineEdit(self)
        hbox4 = QHBoxLayout()
        hbox4.addWidget(l5)
        hbox4.addWidget(self.notes)
        l7 = QLabel('*Easting: ')
        self.easting = QLineEdit(self)
        l8 = QLabel('*Northing: ')
        self.northing = QLineEdit(self)
        hbox6 = QHBoxLayout()
        hbox6.addWidget(l7)
        hbox6.addWidget(self.easting)
        hbox6.addWidget(l8)
        hbox6.addWidget(self.northing)
        l6 = QLabel('*Site where data collected: ')
        self.site = QLineEdit(self)
        l9 = QLabel('Region where data collected: ')
        self.region = QLineEdit(self)
        hbox7 = QHBoxLayout()
        hbox7.addWidget(l6)
        hbox7.addWidget(self.site)
        hbox7.addWidget(l9)
        hbox7.addWidget(self.region)

        hbox9 = QHBoxLayout()
        yes = QPushButton('Yes')
        no = QPushButton('No')
        yes.clicked.connect(self.returnYes)
        no.clicked.connect(self.returnNo)
        hbox9.addWidget(yes)
        hbox9.addWidget(no)

        vbox = QVBoxLayout()
        vbox.addWidget(l1)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addLayout(hbox7)
        vbox.addLayout(hbox6)
        vbox.addLayout(hbox9)

        self.setLayout(vbox)

    def returnYes(self):
        if len(self.data.text()) > 0 and len(self.observer.text()) > 0 and len(self.easting.text()) > 0 and len(self.northing.text()) > 0 and len(self.site.text())>0:
            self.accept()
        else:
            msg = SupportClasses_GUI.MessagePopup("t", "Mandatory fields missing", "You need at least a data source, name, easting, northing, and site name")
            msg.exec_()
            

    def returnNo(self):
        self.reject()

    def getValues(self):
        return [self.data.text(), self.observer.text(),self.method.text(),self.detector.text(),self.notes.text(), self.easting.text(),self.northing.text(), self.site.text(), self.region.text()]

