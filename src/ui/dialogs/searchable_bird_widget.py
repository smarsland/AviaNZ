
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
from src.ui.components.dialogs_and_popups import MessagePopup
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

class SearchableBirdWidget(QWidget):
    add_requested = pyqtSignal(str)

    def __init__(self, allow_multi, parent=None):
        super(SearchableBirdWidget, self).__init__(parent)
        self.header = QLabel("Type to search")
        self.addSpBtn = QPushButton(QIcon('src/resources/images/add.png'), " Add new")
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
        msg = MessagePopup("t", "Adding new species", 'Species "%s" will be added to the full bird list. Are you sure?' % species)
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
