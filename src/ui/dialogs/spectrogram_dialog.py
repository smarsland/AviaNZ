
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
from src.ui.colourMaps import colourMaps
from src.ui.components.audio_player import ControllableAudio
from src.ui.components.buttons_and_controls import BrightContrVol, MainPushButton, PicButton
from src.ui.components.popups import MessagePopup
from src.ui.components.layout_widgets import PartlyResizableGLW
from src.ui.components.species_menus import BatSelectionMenu, BirdSelectionMenu
from src.core import Spectrogram
from src.core import SupportClasses
import openpyxl
import json
from scipy.stats import boxcox

import copy

import re

pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class SpectrogramDialog(QDialog):
    # Class for the spectrogram dialog box
    # TODO: Reproduce the graph from Raven (View/Configure Brightness)
    def __init__(self, width, incr, minFreq, maxFreq, minFreqShow, maxFreqShow, window, sgtype='Standard', sgnorm='Log', sgscale='Linear', nfilters=128, batmode=False, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Spectrogram Options')
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
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