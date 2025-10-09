
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

class ExportBats(QDialog):
    def __init__(self,filename,observer,easting="",northing="",recorder=""):
        QDialog.__init__(self)
        self.setWindowTitle('Export Results?')
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
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
