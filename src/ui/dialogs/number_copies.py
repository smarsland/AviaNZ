
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

class getNumberCopiesPlus(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Multiple Calls')
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
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
