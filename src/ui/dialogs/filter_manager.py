
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

class FilterManager(QDialog):
    def __init__(self, filtdir, parent=None):
        super(FilterManager, self).__init__(parent)
        self.setWindowTitle("Manage recognisers")
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))

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

        msg = MessagePopup("w", "Confirm delete", "Warning: you are about to permanently delete recogniser %s.\nAre you sure?" % sources[0])
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
                    msg = MessagePopup("t", "Import error"," A file %s already exists. Overwrite or skip?" % targets[i])
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
            msg = MessagePopup("d", "Successfully imported","Import complete. Now you can use the recogniser %s" % os.path.basename(targets[0]))
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
                msg = MessagePopup("d", "Successfully exported", "Export successful. Now you can share the recogniser file(s) in %s" % target)
                msg.exec()
            except Exception as e:
                print("ERROR: failed to export")
                print(e)
                return
