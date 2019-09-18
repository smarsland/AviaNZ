
#
# This is part of the AviaNZ interface
# Holds most of the code for the various dialog boxes
# Version 1.5.1 13/09/2019
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis

#    AviaNZ birdsong analysis program
#    Copyright (C) 2017--2019

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

# Dialogs used for filter training / testing.
# These are relatively complicated wizards which also do file I/O

import os
import time
import platform
import wavio
import json
import copy

from PyQt5.QtGui import QIcon, QValidator, QAbstractItemView, QPixmap
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtWidgets import QLabel, QSlider, QPushButton, QListWidget, QListWidgetItem, QComboBox, QWizard, QWizardPage, QLineEdit, QSizePolicy, QFormLayout, QVBoxLayout, QHBoxLayout, QCheckBox
from PyQt5.QtMultimedia import QAudioFormat

import matplotlib.markers as mks
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import colourMaps
import SupportClasses as SupportClasses
import SignalProc
import WaveletSegment
import Segment
import Clustering
import Dialogs


class BuildRecAdvWizard(QWizard):
    # page 1 - select training data
    class WPageData(QWizardPage):
        def __init__(self, parent=None):
            super(BuildRecAdvWizard.WPageData, self).__init__(parent)
            self.setTitle('Training data')
            self.setSubTitle('To start training, you need labelled calls from your species as training data (use the manual interface to create this). Select the folder where this data is located. Then select the  species.')

            self.setMinimumSize(250, 150)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.trainDirName = QLineEdit()
            self.trainDirName.setReadOnly(True)
            self.btnBrowse = QPushButton('Browse')
            self.btnBrowse.clicked.connect(self.browseTrainData)

            self.listFiles = QListWidget()
            self.listFiles.setMinimumWidth(150)
            self.listFiles.setMinimumHeight(275)
            self.listFiles.setSelectionMode(QAbstractItemView.NoSelection)

            selectSpLabel = QLabel("Choose the species for which you want to build the recogniser")
            self.species = QComboBox()  # fill during browse
            self.species.addItems(['Choose species...'])

            space = QLabel()
            space.setFixedHeight(20)

            # SampleRate parameter
            self.fs = QSlider(Qt.Horizontal)
            self.fs.setTickPosition(QSlider.TicksBelow)
            self.fs.setTickInterval(2000)
            self.fs.setRange(0, 32000)
            self.fs.setValue(0)
            self.fs.valueChanged.connect(self.fsChange)
            self.fstext = QLabel('')
            form1 = QFormLayout()
            form1.addRow('', self.fstext)
            form1.addRow('Preferred sampling rate (Hz)', self.fs)

            # training page layout
            layout1 = QHBoxLayout()
            layout1.addWidget(self.trainDirName)
            layout1.addWidget(self.btnBrowse)
            layout = QVBoxLayout()
            layout.addWidget(space)
            layout.addLayout(layout1)
            layout.addWidget(self.listFiles)
            layout.addWidget(space)
            layout.addWidget(selectSpLabel)
            layout.addWidget(self.species)
            layout.addLayout(form1)
            layout.setAlignment(Qt.AlignVCenter)
            self.setLayout(layout)

        def browseTrainData(self):
            trainDir = QtGui.QFileDialog.getExistingDirectory(self, 'Choose folder for training')
            self.trainDirName.setText(trainDir)
            self.fillFileList(trainDir)

        def fsChange(self, value):
            value = value // 4000 * 4000
            if value < 4000:
                value = 4000
            self.fstext.setText(str(value))

        def fillFileList(self, dirName):
            """ Generates the list of files for a file listbox. """
            if not os.path.isdir(dirName):
                print("Warning: directory doesn't exist")
                return

            self.listFiles.clear()
            spList = set()
            fs = []
            # collect possible species from annotations:
            for root, dirs, files in os.walk(dirName):
                for filename in files:
                    if filename.endswith('.wav') and filename+'.data' in files:
                        # this wav has data, so see what species are in there
                        segments = Segment.SegmentList()
                        segments.parseJSON(os.path.join(root, filename+'.data'))
                        spList.update([lab["species"] for seg in segments for lab in seg[4]])

                        # also retrieve its sample rate
                        samplerate = wavio.read(os.path.join(root, filename), 1).rate
                        fs.append(samplerate)

            # might need better limits on selectable sample rate here
            self.fs.setValue(int(np.min(fs)))
            self.fs.setRange(4000, int(np.max(fs)))
            self.fs.setSingleStep(4000)
            self.fs.setTickInterval(4000)

            spList = list(spList)
            spList.insert(0, 'Choose species...')
            self.species.clear()
            self.species.addItems(spList)

            listOfFiles = QDir(dirName).entryInfoList(['*.wav'], filters=QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files,sort=QDir.DirsFirst)
            listOfDataFiles = QDir(dirName).entryList(['*.wav.data'])
            for file in listOfFiles:
                # Add the filename to the right list
                item = QListWidgetItem(self.listFiles)
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

    # page 2 - precluster
    class WPagePrecluster(QWizardPage):
        def __init__(self, parent=None):
            super(BuildRecAdvWizard.WPagePrecluster, self).__init__(parent)
            self.setTitle('Confirm data input')
            self.setSubTitle('When ready, press \"Cluster\" to start clustering. The process may take a long time.')
            self.setMinimumSize(250, 150)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            revtop = QLabel("The following parameters were set:")
            self.params = QLabel("")
            self.params.setStyleSheet("QLabel { color : #808080; }")
            self.warnLabel = QLabel("")
            self.warnLabel.setStyleSheet("QLabel { color : #800000; }")

            layout2 = QVBoxLayout()
            layout2.addWidget(revtop)
            layout2.addWidget(self.params)
            layout2.addWidget(self.warnLabel)
            self.setLayout(layout2)
            self.setButtonText(QWizard.NextButton, 'Cluster >')

        def initializePage(self):
            self.wizard().saveTestBtn.setVisible(False)
            # parse some params
            fs = int(self.field("fs"))//4000*4000
            if fs not in [8000, 16000, 24000, 32000, 36000, 48000]:
                self.warnLabel.setText("Warning: unusual sampling rate selected, make sure it is intended.")
            else:
                self.warnLabel.setText("")
            self.params.setText("Species: %s\nTraining data: %s\nSampling rate: %d\n" % (self.field("species"), self.field("trainDir"), fs))

    # page 3 - calculate and adjust clusters
    class WPageCluster(QWizardPage):
        def __init__(self, config, parent=None):
            super(BuildRecAdvWizard.WPageCluster, self).__init__(parent)
            self.setTitle('Cluster similar looking calls')
            self.setSubTitle('AviaNZ has tried to identify similar calls in your dataset. Please merge clusters, and transfer any errors as appropriate. You might also want to name each type of call.')
            self.setMinimumSize(800, 500)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
            self.adjustSize()

            self.sampleRate = 0
            self.segments = []
            self.clusters = {}
            self.clustercentres = {}
            self.duration = 0
            self.feature = 'we'
            self.picbuttons = []
            self.cboxes = []
            self.tboxes = []
            self.nclasses = 0
            self.config = config
            self.segsChanged = False

            self.lblSpecies = QLabel()
            self.lblSpecies.setStyleSheet("QLabel { color : #808080; }")

            # Volume control
            self.volSlider = QSlider(Qt.Horizontal)
            self.volSlider.valueChanged.connect(self.volSliderMoved)
            self.volSlider.setRange(0, 100)
            self.volSlider.setValue(50)
            volIcon = QLabel()
            volIcon.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
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

            self.btnMerge = QPushButton('Merge Clusters')
            self.btnMerge.clicked.connect(self.merge)
            self.btnMerge.setEnabled(False)
            lb = QLabel('Move Selected Segment/s to Cluster')
            lb.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.cmbUpdateSeg = QComboBox()
            self.btnUpdateSeg = QPushButton('Apply')
            self.btnUpdateSeg.clicked.connect(self.moveSelectedSegs)
            self.btnDeleteSeg = QPushButton("Remove selected segment/s")
            self.btnDeleteSeg.setFixedWidth(200)
            self.btnDeleteSeg.clicked.connect(self.deleteSelectedSegs)

            # page 2 layout
            layout1 = QVBoxLayout()
            layout1.addWidget(self.lblSpecies)
            hboxSpecContr = QHBoxLayout()
            hboxSpecContr.addWidget(labelBr)
            hboxSpecContr.addWidget(self.brightnessSlider)
            hboxSpecContr.addWidget(labelCo)
            hboxSpecContr.addWidget(self.contrastSlider)
            hboxSpecContr.addWidget(volIcon)
            hboxSpecContr.addWidget(self.volSlider)

            hboxBtns = QHBoxLayout()
            hboxBtns.addWidget(self.btnMerge)
            hboxBtns.addWidget(lb)
            hboxBtns.addWidget(self.cmbUpdateSeg)
            hboxBtns.addWidget(self.btnUpdateSeg)

            # top part
            vboxTop = QVBoxLayout()
            vboxTop.addLayout(hboxSpecContr)
            vboxTop.addLayout(hboxBtns)

            # set up the images
            self.flowLayout = pg.LayoutWidget()
            self.flowLayout.setGeometry(QtCore.QRect(0, 0, 380, 247))

            self.scrollArea = QtGui.QScrollArea(self)
            self.scrollArea.setWidgetResizable(True)
            self.scrollArea.setWidget(self.flowLayout)

            # set overall layout of the dialog
            self.vboxFull = QVBoxLayout()
            self.vboxFull.addLayout(layout1)
            self.vboxFull.addLayout(vboxTop)
            self.vboxFull.addWidget(self.scrollArea)
            self.vboxFull.addWidget(self.btnDeleteSeg, alignment=Qt.AlignRight)
            self.setLayout(self.vboxFull)

        def initializePage(self):
            self.wizard().saveTestBtn.setVisible(False)
            # parse field shared by all subfilters
            fs = int(self.field("fs"))//4000*4000
            self.wizard().speciesData = {"species": self.field("species"), "SampleRate": fs, "Filters": []}

            with pg.BusyCursor():
                print("Processing. Please wait...")
                # return format:
                # self.segments: [parent_audio_file, [segment], [syllables], [features], class_label]
                # fs: sampling freq
                # self.nclasses: number of class_labels
                self.cluster = Clustering.Clustering([], [])
                self.segments, fs, self.nclasses, self.duration = self.cluster.cluster(self.field("trainDir"),
                                                                                       self.field("species"),
                                                                                        feature=self.feature,
                                                                                        n_clusters=5)
                # self.segments, fs, self.nclasses = Clustering.cluster_by_dist(self.dName, feature='we', max_clusters=5, single=True)

                # Create and show the buttons
                self.clearButtons()
                self.addButtons()
                self.updateButtons()
                self.segsChanged = True
                self.completeChanged.emit()

        def isComplete(self):
            # empty cluster names?
            if len(self.clusters)==0:
                return False
            # duplicate cluster names aren't updated:

            for ID in range(self.nclasses):
                if self.clusters[ID] != self.tboxes[ID].text():
                    return False
            # no segments at all?
            if len(self.segments)==0:
                return False

            # if all good, then check if we need to redo the pages.
            # segsChanged should be updated by any user changes!
            if self.segsChanged:
                # update the cluster centres
                # TODO: safely update cluster centres after all wavelet trainings are done, required frq range etc. are editable by the user
                # clustercentres = {}
                # for c in range(self.nclasses):
                #     clustercentres[c] = self.getClusterCenter(self.segments, c, self.field("fs"), f1, f2, self.feature, self.duration)
                self.segsChanged = False
                self.wizard().redoTrainPages()
            return True

        def merge(self):
            """ Listner for the merge button. Merge the rows (clusters) checked into one cluster.
            """
            # Find which clusters/rows to merge
            self.segsChanged = True
            tomerge = []
            i = 0
            for cbox in self.cboxes:
                if cbox.checkState() != 0:
                    tomerge.append(i)
                i += 1
            print('rows/clusters to merge are:', tomerge)
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

            # print('[old, new] labels')
            labels = dict(labels)
            # print(labels)

            keys = [i for i in range(self.nclasses) if i not in tomerge]        # the old keys those didn't merge
            # print('old keys left: ', keys)

            # update clusters dictionary {ID: cluster_name}
            clusters = {0: self.clusters[tomerge[0]]}
            for i in keys:
                clusters.update({labels[i]: self.clusters[i]})

            print('before update: ', self.clusters)
            self.clusters = clusters
            print('after update: ', self.clusters)

            self.nclasses = nclasses

            # update the segments
            for seg in self.segments:
                seg[-1] = labels[seg[-1]]

            # update the cluster combobox
            self.cmbUpdateSeg.clear()
            for x in self.clusters:
                self.cmbUpdateSeg.addItem(self.clusters[x])

            # Clean and redraw
            self.clearButtons()
            self.updateButtons()
            self.completeChanged.emit()
            # print('updated')

        def moveSelectedSegs(self):
            """ Listner for Apply button to move the selected segments to another cluster.
                Change the cluster ID of those selected buttons and redraw all the clusters.
            """
            self.segsChanged = True
            moveto = self.cmbUpdateSeg.currentText()
            # find the clusterID from name
            for key in self.clusters.keys():
                if moveto == self.clusters[key]:
                    movetoID = key
                    break
            # print(moveto, movetoID)

            for ix in range(len(self.picbuttons)):
                if self.picbuttons[ix].mark == 'yellow':
                    self.segments[ix][-1] = movetoID
                    self.picbuttons[ix].mark = 'green'

            # update self.clusters, delete clusters with no members
            todelete = []
            for ID, label in self.clusters.items():
                empty = True
                for seg in self.segments:
                    if seg[-1] == ID:
                        empty = False
                        break
                if empty:
                    todelete.append(ID)

            self.clearButtons()

            # Generate new class labels
            if len(todelete) > 0:
                keys = [i for i in range(self.nclasses) if i not in todelete]        # the old keys those didn't delete
                # print('old keys left: ', keys)

                nclasses = self.nclasses - len(todelete)
                max_label = nclasses - 1
                labels = []
                c = self.nclasses - 1
                while c > -1:
                    if c in keys:
                        labels.append((c, max_label))
                        max_label -= 1
                    c -= 1

                # print('[old, new] labels')
                labels = dict(labels)
                print(labels)

                # update clusters dictionary {ID: cluster_name}
                clusters = {}
                for i in keys:
                    clusters.update({labels[i]: self.clusters[i]})

                print('before move: ', self.clusters)
                self.clusters = clusters
                print('after move: ', self.clusters)

                # update the segments
                for seg in self.segments:
                    seg[-1] = labels[seg[-1]]

                self.nclasses = nclasses

            # redraw the buttons
            self.updateButtons()
            self.completeChanged.emit()

        def deleteSelectedSegs(self):
            """ Listner for Delete button to delete the selected segments completely.
            """
            inds = []
            for ix in range(len(self.picbuttons)):
                if self.picbuttons[ix].mark == 'yellow':
                    inds.append(ix)

            if len(inds) > 0:
                self.segsChanged = True
                segments = []
                picbuttons = []
                for ix in range(len(self.picbuttons)):
                    if ix not in inds:
                        segments.append(self.segments[ix])
                        picbuttons.append(self.picbuttons[ix])
                self.segments = segments
                self.picbuttons = picbuttons

                # update self.clusters, delete clusters with no members
                todelete = []
                for ID, label in self.clusters.items():
                    empty = True
                    for seg in self.segments:
                        if seg[-1] == ID:
                            empty = False
                            break
                    if empty:
                        todelete.append(ID)

                self.clearButtons()

                # Generate new class labels
                if len(todelete) > 0:
                    keys = [i for i in range(self.nclasses) if i not in todelete]        # the old keys those didn't delete
                    # print('old keys left: ', keys)

                    nclasses = self.nclasses - len(todelete)
                    max_label = nclasses - 1
                    labels = []
                    c = self.nclasses - 1
                    while c > -1:
                        if c in keys:
                            labels.append((c, max_label))
                            max_label -= 1
                        c -= 1

                    # print('[old, new] labels')
                    labels = dict(labels)
                    # print(labels)

                    # update clusters dictionary {ID: cluster_name}
                    clusters = {}
                    for i in keys:
                        clusters.update({labels[i]: self.clusters[i]})

                    print('before delete: ', self.clusters)
                    self.clusters = clusters
                    print('after delete: ', self.clusters)

                    # update the segments
                    for seg in self.segments:
                        seg[-1] = labels[seg[-1]]

                    self.nclasses = nclasses
                    self.cmbUpdateSeg.clear()
                    for x in self.clusters:
                        self.cmbUpdateSeg.addItem(self.clusters[x])

                # redraw the buttons
                self.updateButtons()
                self.completeChanged.emit()

        def updateClusterNames(self):
            # Check duplicate names
            self.segsChanged = True
            names = [self.tboxes[ID].text() for ID in range(self.nclasses)]
            if len(names) != len(set(names)):
                msg = SupportClasses.MessagePopup("w", "Name error", "Duplicate cluster names! \nTry again")
                msg.exec_()
                self.completeChanged.emit()
                return

            for ID in range(self.nclasses):
                self.clusters[ID] = self.tboxes[ID].text()

            self.cmbUpdateSeg.clear()
            for x in self.clusters:
                self.cmbUpdateSeg.addItem(self.clusters[x])
            self.completeChanged.emit()
            print('updated clusters: ', self.clusters)

        def addButtons(self):
            """ Only makes the PicButtons and self.clusters dict
            """
            self.clusters = []
            self.picbuttons = []
            for i in range(self.nclasses):
                self.clusters.append((i, 'Cluster_' + str(i)))
            self.clusters = dict(self.clusters)     # Dictionary of {ID: cluster_name}

            self.cmbUpdateSeg.clear()
            for x in self.clusters:
                self.cmbUpdateSeg.addItem(self.clusters[x])

            # Create the buttons for each segment
            for seg in self.segments:
                sg, audiodata, audioFormat = self.loadFile(seg[0], seg[1][1]-seg[1][0], seg[1][0])
                newButton = Dialogs.PicButton(1, np.fliplr(sg), audiodata, audioFormat, seg[1][1]-seg[1][0], 0, seg[1][1], self.lut, self.colourStart, self.colourEnd, False, cluster=True)
                self.picbuttons.append(newButton)
            # (updateButtons will place them in layouts and show them)

        def checkMerge(self):
            """ Updates merge buttons state when two clusters are selected """
            sumCh = sum([box.isChecked() for box in self.cboxes])
            if sumCh>=2:
                self.btnMerge.setEnabled(True)
            else:
                self.btnMerge.setEnabled(False)

        def updateButtons(self):
            """ Draw the existing buttons, and create check- and text-boxes.
            Called when merging clusters or initializing the page. """
            self.cboxes = []    # List of check boxes
            self.tboxes = []    # Corresponding list of text boxes
            for r in range(self.nclasses):
                c = 0
                # print('**', self.clusters[r])
                tbox = QLineEdit(self.clusters[r])
                tbox.setMinimumWidth(80)
                tbox.setMaximumHeight(150)
                # tbox.setStyleSheet("border: none; background: rgba(0,0,0,0%); text-align: center; vertical-align: middle;")
                tbox.setStyleSheet("border: none;")
                tbox.setAlignment(QtCore.Qt.AlignCenter)
                tbox.textChanged.connect(self.updateClusterNames)
                self.tboxes.append(tbox)
                self.flowLayout.addWidget(self.tboxes[-1], r, c)
                c += 1
                cbox = QCheckBox("")
                cbox.clicked.connect(self.checkMerge)
                self.cboxes.append(cbox)
                self.flowLayout.addWidget(self.cboxes[-1], r, c)
                c += 1
                # Find the segments under this class and show them
                for segix in range(len(self.segments)):
                    if self.segments[segix][-1] == r:
                        self.flowLayout.addWidget(self.picbuttons[segix], r, c)
                        c += 1
                        self.picbuttons[segix].show()
            self.flowLayout.update()

        def clearButtons(self):
            """ Remove existing buttons, call when merging clusters
            """
            for ch in self.cboxes:
                ch.hide()
            for tbx in self.tboxes:
                tbx.hide()
            for btnum in reversed(range(self.flowLayout.layout.count())):
                item = self.flowLayout.layout.itemAt(btnum)
                if item is not None:
                    self.flowLayout.layout.removeItem(item)
                    r, c = self.flowLayout.items[item.widget()]
                    del self.flowLayout.items[item.widget()]
                    del self.flowLayout.rows[r][c]
                    item.widget().hide()
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

        def loadFile(self, filename, duration=0, offset=0):
            if duration == 0:
                duration = None
            sp = SignalProc.SignalProc(512, 256)
            sp.readWav(filename, duration, offset)

            sgRaw = sp.spectrogram(window='Hann', mean_normalise=True, onesided=True,
                                          multitaper=False, need_even=False)
            maxsg = np.min(sgRaw)
            self.sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))
            self.setColourMap()

            return self.sg, sp.data, sp.audioFormat

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

    # page 4 - set params for training
    class WPageParams(QWizardPage):
        def __init__(self, cluster, segments, picbtn, parent=None):
            super(BuildRecAdvWizard.WPageParams, self).__init__(parent)
            self.setTitle("Training parameters: %s" % cluster)
            self.setSubTitle("These fields were completed using the training data. Adjust if required.\nWhen ready, "
                             "press \"Train\". The process may take a long time.")
            self.setMinimumSize(250, 350)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.lblSpecies = QLabel("")
            self.lblSpecies.setStyleSheet("QLabel { color : #808080; }")
            self.numSegs = QLabel("")
            self.numSegs.setStyleSheet("QLabel { color : #808080; }")
            self.segments = segments
            lblCluster = QLabel(cluster)
            lblCluster.setStyleSheet("QLabel { color : #808080; }")

            # small image of the cluster and other info
            calldescr = QFormLayout()
            calldescr.addRow('Species:', self.lblSpecies)
            calldescr.addRow('Call type:', lblCluster)
            calldescr.addRow('Number of segments:', self.numSegs)
            imgCluster = QLabel()
            imgCluster.setPixmap(QPixmap.fromImage(picbtn.im1))

            # TimeRange parameters
            form1_step4 = QFormLayout()
            self.minlen = QLineEdit(self)
            self.minlen.setText('')
            form1_step4.addRow('Min call length (secs)', self.minlen)
            self.maxlen = QLineEdit(self)
            self.maxlen.setText('')
            form1_step4.addRow('Max call length (secs)', self.maxlen)

            # FreqRange parameters
            self.fLow = QSlider(Qt.Horizontal)
            self.fLow.setTickPosition(QSlider.TicksBelow)
            self.fLow.setTickInterval(2000)
            self.fLow.setRange(0, 32000)
            self.fLow.setSingleStep(100)
            self.fLow.valueChanged.connect(self.fLowChange)
            self.fLowtext = QLabel('')
            form1_step4.addRow('', self.fLowtext)
            form1_step4.addRow('Lower frq. limit (Hz)', self.fLow)
            self.fHigh = QSlider(Qt.Horizontal)
            self.fHigh.setTickPosition(QSlider.TicksBelow)
            self.fHigh.setTickInterval(2000)
            self.fHigh.setRange(0, 32000)
            self.fHigh.setSingleStep(100)
            self.fHigh.valueChanged.connect(self.fHighChange)
            self.fHightext = QLabel('')
            form1_step4.addRow('', self.fHightext)
            form1_step4.addRow('Upper frq. limit (Hz)', self.fHigh)

            # thr, M parameters
            thrLabel = QLabel('ROC curve points per line (thr)')
            MLabel = QLabel('ROC curve lines (M)')
            self.cbxThr = QComboBox()
            self.cbxThr.addItems(['4', '5', '6', '7', '8', '9', '10'])
            self.cbxM = QComboBox()
            self.cbxM.addItems(['2', '3', '4', '5'])
            form2_step4 = QFormLayout()
            form2_step4.addRow(thrLabel, self.cbxThr)
            form2_step4.addRow(MLabel, self.cbxM)

            ### Step4 layout
            hboxTop = QHBoxLayout()
            hboxTop.addLayout(calldescr)
            hboxTop.addSpacing(30)
            hboxTop.addWidget(imgCluster)
            layout_step4 = QVBoxLayout()
            layout_step4.setSpacing(10)
            layout_step4.addWidget(QLabel("<b>Current call type</b>"))
            layout_step4.addLayout(hboxTop)
            layout_step4.addWidget(QLabel("<b>Call parameters</b>"))
            layout_step4.addLayout(form1_step4)
            layout_step4.addWidget(QLabel("<b>Training parameters</b>"))
            layout_step4.addLayout(form2_step4)
            self.setLayout(layout_step4)

            self.setButtonText(QWizard.NextButton, 'Train >')

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

        def initializePage(self):
            self.wizard().saveTestBtn.setVisible(False)
            # populates values based on training files
            f_low = []
            f_high = []
            len_min = []
            len_max = []
            fs = int(self.field("fs")) // 4000 * 4000

            # self.segments is already selected to be this cluster only
            pageSegs = Segment.SegmentList()
            for longseg in self.segments:
                # long seg has format: [file [segment] clusternum]
                pageSegs.addSegment(longseg[1])
            len_min, len_max, f_low, f_high = pageSegs.getSummaries()

            self.minlen.setText(str(round(np.min(len_min),2)))
            self.maxlen.setText(str(round(np.max(len_max),2)))
            self.fLow.setRange(0, fs/2)
            self.fLow.setValue(max(0,int(np.min(f_low))))
            self.fHigh.setRange(0, fs/2)
            self.fHigh.setValue(min(fs/2,int(np.max(f_high))))

    # page 5 - run training, show ROC
    class WPageTrain(QWizardPage):
        def __init__(self, id, clustID, clustname, segments, parent=None):
            super(BuildRecAdvWizard.WPageTrain, self).__init__(parent)
            self.setTitle('Training results')
            self.setMinimumSize(600, 500)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.segments = segments
            self.clust = clustname
            self.clusterID = clustID
            # this ID links it to the parameter fields
            self.pageId = id

            self.lblTrainDir = QLabel()
            self.lblTrainDir.setStyleSheet("QLabel { color : #808080; }")
            self.lblSpecies = QLabel()
            self.lblSpecies.setStyleSheet("QLabel { color : #808080; }")
            self.lblCluster = QLabel()
            self.lblCluster.setStyleSheet("QLabel { color : #808080; }")
            space = QLabel()
            space.setFixedHeight(25)
            spaceH = QLabel()
            spaceH.setFixedWidth(30)

            self.lblUpdate = QLabel()

            # These are connected to fields and actually control the wizard's flow
            self.bestM = QLineEdit()
            self.bestThr = QLineEdit()
            self.bestNodes = QLineEdit()
            self.bestM.setReadOnly(True)
            self.bestThr.setReadOnly(True)
            self.bestNodes.setReadOnly(True)
            self.filtSummary = QFormLayout()
            self.filtSummary.addRow("Current M:", self.bestM)
            self.filtSummary.addRow("Current thr:", self.bestThr)
            self.filtSummary.addRow("Current nodes:", self.bestNodes)

            # this is the Canvas Widget that displays the plot
            self.figCanvas = ROCCanvas(self)
            self.figCanvas.plotme()
            self.marker = self.figCanvas.ax.plot([0,1], [0,1], marker='o', color='black', linestyle='dotted')[0]

            # figure click handler
            def onclick(event):
                fpr_cl = event.xdata
                tpr_cl = event.ydata
                if tpr_cl is None or fpr_cl is None:
                    return

                # get M and thr for closest point
                distarr = (tpr_cl - self.TPR) ** 2 + (fpr_cl - self.FPR) ** 2
                M_min_ind, thr_min_ind = np.unravel_index(np.argmin(distarr), distarr.shape)
                tpr_near = self.TPR[M_min_ind, thr_min_ind]
                fpr_near = self.FPR[M_min_ind, thr_min_ind]
                self.marker.set_visible(False)
                self.figCanvas.draw()
                self.marker.set_xdata([fpr_cl, fpr_near])
                self.marker.set_ydata([tpr_cl, tpr_near])
                self.marker.set_visible(True)
                self.figCanvas.ax.draw_artist(self.marker)
                self.figCanvas.update()

                print("fpr_cl, tpr_cl: ", fpr_near, tpr_near)

                # update sidebar
                self.lblUpdate.setText('DETECTION SUMMARY\n\nTPR:\t' + str(round(tpr_near * 100, 2)) + '%'
                                              '\nFPR:\t' + str(round(fpr_near * 100, 2)) + '%\n\nClick "Next" to proceed.')

                # this will save the best parameters to the global fields
                self.bestM.setText("%.4f" % self.MList[M_min_ind])
                self.bestThr.setText("%.4f" % self.thrList[thr_min_ind])
                # Get nodes for closest point
                self.bestNodes.setText(str(self.nodes[M_min_ind][thr_min_ind]))
                for itemnum in range(self.filtSummary.count()):
                    self.filtSummary.itemAt(itemnum).widget().show()

            self.figCanvas.figure.canvas.mpl_connect('button_press_event', onclick)


            vboxHead = QFormLayout()
            vboxHead.addRow("Training data:", self.lblTrainDir)
            vboxHead.addRow("Target species:", self.lblSpecies)
            vboxHead.addRow("Target calltype:", self.lblCluster)
            vboxHead.addWidget(space)

            hbox2 = QHBoxLayout()
            hbox2.addWidget(self.figCanvas)

            hbox3 = QHBoxLayout()
            hbox3.addLayout(self.filtSummary)
            hbox3.addWidget(spaceH)
            hbox3.addWidget(self.lblUpdate)

            vbox = QVBoxLayout()
            vbox.addLayout(vboxHead)
            vbox.addLayout(hbox2)
            vbox.addSpacing(10)
            vbox.addLayout(hbox3)

            self.setLayout(vbox)

        # ACTUAL TRAINING IS DONE HERE
        def initializePage(self):
            self.lblTrainDir.setText(self.field("trainDir"))
            self.lblSpecies.setText(self.field("species"))
            self.wizard().saveTestBtn.setVisible(False)
            self.lblCluster.setText(self.clust)
            for itemnum in range(self.filtSummary.count()):
                self.filtSummary.itemAt(itemnum).widget().hide()

            # parse fields specific to this subfilter
            minlen = float(self.field("minlen"+str(self.pageId)))
            maxlen = float(self.field("maxlen"+str(self.pageId)))
            fLow = int(self.field("fLow"+str(self.pageId)))
            fHigh = int(self.field("fHigh"+str(self.pageId)))
            numthr = int(self.field("thr"+str(self.pageId)))
            numM = int(self.field("M"+str(self.pageId)))
            # note: for each page we reset the filter to contain 1 calltype
            self.wizard().speciesData["Filters"] = [{'calltype': self.clust, 'TimeRange': [minlen, maxlen], 'FreqRange': [fLow, fHigh]}]

            # export 1/0 ground truth
            window = 1
            inc = None
            with pg.BusyCursor():
                for root, dirs, files in os.walk(self.field("trainDir")):
                    for file in files:
                        wavFile = os.path.join(root, file)
                        if file.endswith('.wav') and os.stat(wavFile).st_size != 0 and file + '.data' in files:
                            pageSegs = Segment.SegmentList()
                            pageSegs.parseJSON(wavFile + '.data')

                            # CLUSTERS COME IN HERE:
                            # replace segments with the current cluster
                            # (self.segments is already selected to be this cluster only)
                            pageSegs.clear()
                            for longseg in self.segments:
                                # long seg has format: [file [segment] clusternum]
                                if longseg[0] == wavFile:
                                    pageSegs.addSegment(longseg[1])

                            # So, each page will overwrite a file with the 0/1 annots,
                            # and recalculate the stats for that cluster.

                            # exports 0/1 annotations and retrieves segment time, freq bounds
                            pageSegs.exportGT(wavFile, self.field("species"), window=window, inc=inc)

            # calculate cluster centres
            # (self.segments is already selected to be this cluster only)
            with pg.BusyCursor():
                cl = Clustering.Clustering([], [])
                self.clustercentre = cl.getClusterCenter(self.segments, self.field("fs"), fLow, fHigh, self.wizard().clusterPage.feature, self.wizard().clusterPage.duration)

            # Get detection measures over all M,thr combinations
            print("starting wavelet training")
            with pg.BusyCursor():
                opstartingtime = time.time()
                ws = WaveletSegment.WaveletSegment(self.wizard().speciesData)
                # returns 2d lists of nodes over M x thr, or stats over M x thr
                self.thrList = np.linspace(0.2, 1, num=numthr)
                self.MList = np.linspace(0.25, 1.5, num=numM)  # TODO determine from syllable length
                # options for training are:
                #  recold - no antialias, recaa - partial AA, recaafull - full AA
                #  Window and inc - in seconds
                window = 1
                inc = None
                self.nodes, TP, FP, TN, FN = ws.waveletSegment_train(self.field("trainDir"),
                                                                self.thrList, self.MList,
                                                                d=False, rf=True,
                                                                learnMode="recaa", window=window, inc=inc)
                print("Filtered nodes: ", self.nodes)
                print("TRAINING COMPLETED IN ", time.time() - opstartingtime)
                self.TPR = TP/(TP+FN)
                self.FPR = 1 - TN/(FP+TN)
                print("TP rate: ", self.TPR)
                print("FP rate: ", self.FPR)

                self.marker.set_visible(False)
                self.figCanvas.plotmeagain(self.TPR, self.FPR)

    # page 6 - fundamental frequency calculation
    class WFFPage(QWizardPage):
        def __init__(self, clust, picbuttons, parent=None):
            super(BuildRecAdvWizard.WFFPage, self).__init__(parent)
            self.setTitle('Post-processing')
            self.setSubTitle('Set the post-processing options available below.')
            self.setMinimumSize(250, 350)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.picbuttons = picbuttons
            self.clust = clust

            self.lblTrainDir = QLabel()
            self.lblTrainDir.setStyleSheet("QLabel { color : #808080; }")
            self.lblSpecies = QLabel()
            self.lblSpecies.setStyleSheet("QLabel { color : #808080; }")
            self.lblCluster = QLabel()
            self.lblCluster.setStyleSheet("QLabel { color : #808080; }")

            # fund freq checkbox
            self.hadF0label = QLabel("")
            self.hadNoF0label = QLabel("")
            self.f0_label = QLabel('Fundamental frequency')
            self.ckbF0 = QCheckBox()
            self.ckbF0.setChecked(False)
            self.ckbF0.toggled.connect(self.toggleF0)
            formFFinfo = QFormLayout()
            formFFinfo.addRow("Training segments with detected fund. freq.:", self.hadF0label)
            formFFinfo.addRow("Training segments without fund. freq.:", self.hadNoF0label)

            # fund freq range
            # FreqRange parameters
            form1_step6 = QFormLayout()
            self.F0low = QSlider(Qt.Horizontal)
            self.F0low.setTickPosition(QSlider.TicksBelow)
            self.F0low.setTickInterval(2000)
            self.F0low.setRange(0, 16000)
            self.F0low.setSingleStep(100)
            self.F0low.valueChanged.connect(self.F0lowChange)
            self.F0lowtext = QLabel('')
            form1_step6.addRow('', self.F0lowtext)
            form1_step6.addRow('Lower F0 limit (Hz)    ', self.F0low)
            self.F0high = QSlider(Qt.Horizontal)
            self.F0high.setTickPosition(QSlider.TicksBelow)
            self.F0high.setTickInterval(2000)
            self.F0high.setRange(0, 16000)
            self.F0high.setSingleStep(100)
            self.F0high.valueChanged.connect(self.F0highChange)
            self.F0hightext = QLabel('')
            form1_step6.addRow('', self.F0hightext)
            form1_step6.addRow('Upper F0 limit (Hz)    ', self.F0high)

            # post-proc page layout
            vboxHead = QFormLayout()
            vboxHead.addRow("Training data:", self.lblTrainDir)
            vboxHead.addRow("Target species:", self.lblSpecies)
            vboxHead.addRow("Target calltype:", self.lblCluster)

            hBox = QHBoxLayout()
            hBox.addWidget(self.f0_label)
            hBox.addWidget(self.ckbF0)

            vbox = QVBoxLayout()
            vbox.addLayout(vboxHead)
            vbox.addLayout(formFFinfo)
            vbox.addLayout(hBox)
            vbox.addSpacing(20)
            vbox.addLayout(form1_step6)

            self.setLayout(vbox)

        def F0lowChange(self, value):
            value = value - (value % 10)
            if value < 50:
                value = 50
            self.F0lowtext.setText(str(value))

        def F0highChange(self, value):
            value = value - (value % 10)
            if value < 100:
                value = 100
            self.F0hightext.setText(str(value))

        def toggleF0(self, checked):
            if checked:
                self.F0low.setEnabled(True)
                self.F0lowtext.setEnabled(True)
                self.F0high.setEnabled(True)
                self.F0hightext.setEnabled(True)
            else:
                self.F0low.setEnabled(False)
                self.F0lowtext.setEnabled(False)
                self.F0high.setEnabled(False)
                self.F0hightext.setEnabled(False)

        def initializePage(self):
            self.lblTrainDir.setText(self.field("trainDir"))
            self.lblSpecies.setText(self.field("species"))
            self.lblCluster.setText(self.clust)
            self.wizard().saveTestBtn.setVisible(False)
            # obtain fundamental frequency from each segment
            with pg.BusyCursor():
                print("measuring fundamental frequency range...")
                f0_low = []  # could add a field to input these
                f0_high = []
                # for each segment:
                for picbtn in self.picbuttons:
                    f0_l, f0_h = self.getFundFreq(picbtn.audiodata, picbtn.media_obj.format.sampleRate())
                    # we use NaNs to represent "no F0 found"
                    f0_low.append(f0_l)
                    f0_high.append(f0_h)

            # how many had F0?
            hadNoF0 = sum(np.isnan(f0_low))
            hadF0 = sum(np.invert(np.isnan(f0_high)))
            self.hadF0label.setText("%d (%d %%)" % (hadF0, hadF0/len(f0_low)*100))
            self.hadNoF0label.setText("%d (%d %%)" % (hadNoF0, hadNoF0/len(f0_low)*100))
            if hadF0 == 0:
                print("Warning: no F0 found in the training segments")
                self.ckbF0.setChecked(False)
            else:
                f0_low = round(np.nanmin(f0_low))
                f0_high = round(np.nanmax(f0_high))

                # update the actual fields
                self.ckbF0.setChecked(True)
                self.F0low.setValue(f0_low)
                self.F0high.setValue(f0_high)
                self.F0lowtext.setText(str(f0_low))
                self.F0hightext.setText(str(f0_high))
                print("Determined ff bounds:", f0_low, f0_high)

        def getFundFreq(self, data, sampleRate):
            """ Extracts fund freq range from audiodata """
            sp = SignalProc.SignalProc(256, 128)
            sp.data = data
            sp.sampleRate = sampleRate
            # spectrogram is not necessary if we're not returning segments
            segment = Segment.Segmenter(sp, sampleRate)
            pitch, y, minfreq, W = segment.yin(minfreq=100, returnSegs=False)
            ind = np.squeeze(np.where(pitch > minfreq))
            pitch = pitch[ind]
            # we use NaNs to represent "no F0 found"
            if pitch.size == 0:
                return float("nan"), float("nan")
            if ind.size < 2:
                f0 = pitch
                return f0, f0
            else:
                return round(np.min(pitch)), round(np.max(pitch))

    # page 7 - save the filter
    class WLastPage(QWizardPage):
        def __init__(self, filtdir, parent=None):
            super(BuildRecAdvWizard.WLastPage, self).__init__(parent)
            self.setTitle('Save recogniser')
            self.setSubTitle('Check the overall call detection summary and save the recogniser.')
            self.setMinimumSize(400, 500)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.lblTrainDir = QLabel()
            self.lblTrainDir.setStyleSheet("QLabel { color : #808080; }")
            self.lblSpecies = QLabel()
            self.lblSpecies.setStyleSheet("QLabel { color : #808080; }")
            space = QLabel()
            space.setFixedHeight(25)
            spaceH = QLabel()
            spaceH.setFixedWidth(30)

            self.lblFilter = QLabel('')
            self.lblFilter.setWordWrap(True)
            self.lblFilter.setStyleSheet("QLabel { color : #808080; border: 1px solid black }")

            # filter dir listbox
            self.listFiles = QListWidget()
            self.listFiles.setSelectionMode(QAbstractItemView.NoSelection)
            self.listFiles.setMinimumWidth(150)
            self.listFiles.setMinimumHeight(250)
            filtdir = QDir(filtdir).entryList(filters=QDir.NoDotAndDotDot | QDir.Files)
            for file in filtdir:
                item = QListWidgetItem(self.listFiles)
                item.setText(file)

            # filter file name
            self.enterFiltName = QLineEdit()

            class FiltValidator(QValidator):
                def validate(self, input, pos):
                    if not input.endswith('.txt'):
                        input = input+'.txt'
                    if self.listFiles.findItems(input, Qt.MatchExactly):
                        print("duplicated input", input)
                        return(QValidator.Intermediate, input, pos)
                    else:
                        return(QValidator.Acceptable, input, pos)

            trainFiltValid = FiltValidator()
            trainFiltValid.listFiles = self.listFiles
            self.enterFiltName.setValidator(trainFiltValid)

            # layouts
            vboxHead = QFormLayout()
            vboxHead.addRow("Training data:", self.lblTrainDir)
            vboxHead.addRow("Target species:", self.lblSpecies)
            vboxHead.addWidget(space)

            layout = QVBoxLayout()
            layout.addLayout(vboxHead)
            layout.addWidget(space)
            layout.addWidget(QLabel("The following filter was produced:"))
            layout.addWidget(self.lblFilter)
            layout.addWidget(QLabel("Currently available filters"))
            layout.addWidget(self.listFiles)
            layout.addWidget(space)
            layout.addWidget(QLabel("Enter file name (must be unique)"))
            layout.addWidget(self.enterFiltName)

            self.setButtonText(QWizard.FinishButton, 'Save and Finish')
            self.setLayout(layout)

        def initializePage(self):
            self.lblTrainDir.setText(self.field("trainDir"))
            self.lblSpecies.setText(self.field("species"))

            self.wizard().speciesData["Filters"] = []

            # collect parameters from training pages (except this)
            for pageId in self.wizard().trainpages[:-1]:
                minlen = float(self.field("minlen"+str(pageId)))
                maxlen = float(self.field("maxlen"+str(pageId)))
                fLow = int(self.field("fLow"+str(pageId)))
                fHigh = int(self.field("fHigh"+str(pageId)))
                thr = float(self.field("bestThr"+str(pageId)))
                M = float(self.field("bestM"+str(pageId)))
                nodes = eval(self.field("bestNodes"+str(pageId)))

                # post parameters
                F0 = self.field("F0"+str(pageId))
                F0low = int(self.field("F0low"+str(pageId)))
                F0high = int(self.field("F0high"+str(pageId)))

                newSubfilt = {'calltype': self.wizard().page(pageId+1).clust, 'TimeRange': [minlen, maxlen], 'FreqRange': [fLow, fHigh], 'WaveletParams': {"thr": thr, "M": M, "nodes": nodes}, 'ClusterCentre': list(self.wizard().page(pageId+1).clustercentre), 'Feature': self.wizard().clusterPage.feature}

                if F0:
                    newSubfilt["F0"] = True
                    newSubfilt["F0Range"] = [F0low, F0high]
                print(newSubfilt)
                self.wizard().speciesData["Filters"].append(newSubfilt)

            speciesDataText = copy.deepcopy(self.wizard().speciesData)
            for f in speciesDataText["Filters"]:
                f["ClusterCentre"] = "(...)"
                f["WaveletParams"] = "(...)"

            self.lblFilter.setText(str(speciesDataText))
            self.wizard().saveTestBtn.setVisible(True)
            self.wizard().saveTestBtn.setEnabled(False)
            try:
                self.completeChanged.connect(self.refreshCustomBtn)
            except Exception:
                pass

        def refreshCustomBtn(self):
            if self.isComplete():
                self.wizard().saveTestBtn.setEnabled(True)
            else:
                self.wizard().saveTestBtn.setEnabled(False)

        def validatePage(self):
            # actually write out the filter
            try:
                filename = os.path.join(self.wizard().filtersDir, self.field("filtfile"))
                f = open(filename, 'w')
                f.write(json.dumps(self.wizard().speciesData))
                f.close()
                # prompt the user
                print("Saving new filter to ", filename)
                # Add it to the Filter list
                msg = SupportClasses.MessagePopup("d", "Training completed!", 'Training completed!\nWe recommend to test it on a separate dataset before actual use.')
                msg.exec_()
                return True
            except Exception as e:
                print("ERROR: could not save filter because:", e)
                return False

    # Main init of the training wizard
    def __init__(self, filtdir, config, parent=None):
        super(BuildRecAdvWizard, self).__init__()
        self.setWindowTitle("Build Recogniser")
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        if platform.system() == 'Linux':
            self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        else:
            self.setWindowFlags((self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint) & QtCore.Qt.WindowCloseButtonHint)
        self.setWizardStyle(QWizard.ModernStyle)

        # add the Save & Test button
        self.saveTestBtn = QPushButton("Save and Test")
        self.setButton(QWizard.CustomButton1, self.saveTestBtn)
        self.setButtonLayout([QWizard.Stretch, QWizard.BackButton, QWizard.NextButton, QWizard.CustomButton1, QWizard.FinishButton, QWizard.CancelButton])
        self.setOptions(QWizard.HaveCustomButton1)

        self.filtersDir = filtdir

        # page 1: select training data
        browsedataPage = BuildRecAdvWizard.WPageData()
        browsedataPage.registerField("trainDir*", browsedataPage.trainDirName)
        browsedataPage.registerField("species*", browsedataPage.species, "currentText", browsedataPage.species.currentTextChanged)
        browsedataPage.registerField("fs*", browsedataPage.fs)
        self.addPage(browsedataPage)

        # page 2
        self.preclusterPage = BuildRecAdvWizard.WPagePrecluster()
        self.addPage(self.preclusterPage)

        # page 3: clustering results
        # clusters are created as self.clusterPage.clusters
        self.clusterPage = BuildRecAdvWizard.WPageCluster(config)
        self.addPage(self.clusterPage)
        self.trainpages = []
        self.speciesData = {}
        # then a pair of pages for each calltype will be created by redoTrainPages.

        # Size adjustment between pages:
        self.saveTestBtn.setVisible(False)
        self.currentIdChanged.connect(self.pageChangeResize)

    def redoTrainPages(self):
        self.speciesData["Filters"] = []
        for page in self.trainpages:
            # for each calltype, remove params, ROC, FF pages
            self.removePage(page)
            self.removePage(page+1)
            self.removePage(page+2)
        self.trainpages = []

        for key, value in self.clusterPage.clusters.items():
            print("adding pages for ", key, value)
            # retrieve the segments for this cluster:
            newsegs = []
            newbtns = []
            for segix in range(len(self.clusterPage.segments)):
                seg = self.clusterPage.segments[segix]
                if seg[-1] == key:
                    # save source file, actual segment, and cluster ID
                    newsegs.append(seg)
                    # save the pic button for sound/spec, to be used in post
                    newbtns.append(self.clusterPage.picbuttons[segix])

            # page 4: set training params
            page4 = BuildRecAdvWizard.WPageParams(value, newsegs, newbtns[0])
            page4.lblSpecies.setText(self.field("species"))
            page4.numSegs.setText(str(len(newsegs)))
            pageid = self.addPage(page4)
            self.trainpages.append(pageid)

            # Note: these need to be unique
            page4.registerField("minlen"+str(pageid), page4.minlen)
            page4.registerField("maxlen"+str(pageid), page4.maxlen)
            page4.registerField("fLow"+str(pageid), page4.fLow)
            page4.registerField("fHigh"+str(pageid), page4.fHigh)
            page4.registerField("thr"+str(pageid), page4.cbxThr, "currentText", page4.cbxThr.currentTextChanged)
            page4.registerField("M"+str(pageid), page4.cbxM, "currentText", page4.cbxM.currentTextChanged)

            # page 5: get training results
            page5 = BuildRecAdvWizard.WPageTrain(pageid, key, value, newsegs)
            self.addPage(page5)

            # note: pageid is the same for both page fields
            page5.registerField("bestThr"+str(pageid)+"*", page5.bestThr)
            page5.registerField("bestM"+str(pageid)+"*", page5.bestM)
            page5.registerField("bestNodes"+str(pageid)+"*", page5.bestNodes)

            # page 6: post process
            page6 = BuildRecAdvWizard.WFFPage(value, newbtns)
            self.addPage(page6)

            page6.registerField("F0low"+str(pageid), page6.F0low)
            page6.registerField("F0high"+str(pageid), page6.F0high)
            page6.registerField("F0"+str(pageid), page6.ckbF0)

        # page 7: confirm the results & save
        page7 = BuildRecAdvWizard.WLastPage(self.filtersDir)
        pageid = self.addPage(page7)
        # (store this as well, so that we could wipe it without worrying about page order)
        self.trainpages.append(pageid)
        page7.registerField("filtfile*", page7.enterFiltName)

        self.clusterPage.setFinalPage(False)
        self.clusterPage.completeChanged.emit()

    def pageChangeResize(self, pageid):
        try:
            if self.page(pageid) is not None:
                newsize = self.page(pageid).sizeHint()
                self.setMinimumSize(newsize)
                self.adjustSize()
        except Exception as e:
            print(e)


class TestRecWizard(QWizard):
    class WPageData(QWizardPage):
        def __init__(self, parent=None):
            super(TestRecWizard.WPageData, self).__init__(parent)
            self.setTitle('Testing data')
            self.setSubTitle('Select the folder with testing data, then choose species')

            self.setMinimumSize(250, 150)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.testDirName = QLineEdit()
            self.testDirName.setReadOnly(True)
            self.btnBrowse = QPushButton('Browse')
            self.btnBrowse.clicked.connect(self.browseTestData)

            self.listFiles = QListWidget()
            self.listFiles.setMinimumWidth(150)
            self.listFiles.setMinimumHeight(275)
            self.listFiles.setSelectionMode(QAbstractItemView.NoSelection)

            selectSpLabel = QLabel("Choose the filter that you want to test")
            self.species = QComboBox()  # fill during browse
            self.species.addItems(['Choose filter...'])

            space = QLabel()
            space.setFixedHeight(20)

            # data selection page layout
            layout1 = QHBoxLayout()
            layout1.addWidget(self.testDirName)
            layout1.addWidget(self.btnBrowse)
            layout = QVBoxLayout()
            layout.addWidget(space)
            layout.addLayout(layout1)
            layout.addWidget(self.listFiles)
            layout.addWidget(space)
            layout.addWidget(selectSpLabel)
            layout.addWidget(self.species)
            layout.setAlignment(Qt.AlignVCenter)
            self.setLayout(layout)
            self.setButtonText(QWizard.NextButton, 'Test >')

        def initializePage(self):
            filternames = [key + ".txt" for key in self.wizard().filterlist.keys()]
            self.species.addItems(filternames)
            try:
                self.species.currentTextChanged.connect(self.wizard().redoTestPages)
            except Exception:
                pass

        def browseTestData(self):
            dirName = QtGui.QFileDialog.getExistingDirectory(self, 'Choose folder for testing')
            self.testDirName.setText(dirName)

            self.listFiles.clear()
            listOfFiles = QDir(dirName).entryInfoList(['*.wav'], filters=QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files,sort=QDir.DirsFirst)
            listOfDataFiles = QDir(dirName).entryList(['*.wav.data'])
            for file in listOfFiles:
                # Add the filename to the right list
                item = QListWidgetItem(self.listFiles)
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

    class WPageTest(QWizardPage):
        def __init__(self, clustnum, parent=None):
            super(TestRecWizard.WPageTest, self).__init__(parent)
            self.setTitle('Testing results')
            #self.setSubTitle('Testing results are shown here')

            self.setMinimumSize(250, 200)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.lblTestDir = QLabel()
            self.lblTestDir.setStyleSheet("QLabel { color : #808080; }")
            self.lblSpecies = QLabel()
            self.lblSpecies.setStyleSheet("QLabel { color : #808080; }")
            self.lblCluster = QLabel()
            self.lblCluster.setStyleSheet("QLabel { color : #808080; }")
            space = QLabel()
            space.setFixedHeight(25)

            self.manSegs = QLabel()
            self.manTime = QLabel()
            self.autoSegs = QLabel()
            self.autoTime = QLabel()
            self.sensTime = QLabel()

            # same, after post-processing
            self.manSegsP = QLabel()
            self.manTimeP = QLabel()
            self.autoSegsP = QLabel()
            self.autoTimeP = QLabel()
            self.sensTimeP = QLabel()

            # NOTE: this is not the subfilter name, but the number!
            self.clustnum = clustnum

            # testing results page layout
            vboxHead = QFormLayout()
            vboxHead.addRow("Testing data:", self.lblTestDir)
            vboxHead.addRow("Filter name:", self.lblSpecies)
            vboxHead.addRow("Target calltype:", self.lblCluster)
            vboxHead.addWidget(space)

            form2 = QFormLayout()
            form2.addRow("Manually labelled segments:", self.manSegs)
            form2.addRow("\ttotal seconds:", self.manTime)
            form2.addRow("Segments detected:", self.autoSegs)
            form2.addRow("\ttotal seconds:", self.autoTime)
            form2.addRow("Recall (sensitivity) in 1 s resolution:", self.sensTime)

            form_post = QFormLayout()
            form_post.addRow("Segments detected:", self.autoSegsP)
            form_post.addRow("\ttotal seconds:", self.autoTimeP)
            form_post.addRow("Recall (sensitivity) in 1 s resolution:", self.sensTimeP)

            vbox = QVBoxLayout()
            vbox.addLayout(vboxHead)
            vbox.addWidget(QLabel("<b>Detection summary</b>"))
            vbox.addLayout(form2)
            vbox.addWidget(QLabel("<b>After post-processing</b>"))
            vbox.addLayout(form_post)

            self.setLayout(vbox)

        # Actual testing is done here
        def initializePage(self):
            speciesData = self.wizard().filterlist[self.field("species")[:-4]]
            subfilter = speciesData["Filters"][self.clustnum]

            self.lblTestDir.setText(self.field("testDir"))
            self.lblSpecies.setText(speciesData["species"])
            self.lblCluster.setText(subfilter["calltype"])

            with pg.BusyCursor():
                species = speciesData["species"]
                # not sure if this is needed?
                ind = species.find('>')
                if ind != -1:
                    species = species.replace('>', '(')
                    species = species + ')'

                ws = WaveletSegment.WaveletSegment(speciesData, 'dmey2')
                manSegNum = 0
                window = 1
                inc = None
                # Generate GT files from annotations in test folder
                print('Generating GT...')
                for root, dirs, files in os.walk(self.field("testDir")):
                    for file in files:
                        wavFile = os.path.join(root, file)
                        if file.endswith('.wav') and os.stat(wavFile).st_size != 0 and file + '.data' in files:
                            segments = Segment.SegmentList()
                            segments.parseJSON(wavFile + '.data')
                            manSegNum += len(segments)

                            # Currently, we ignore call types here and just
                            # look for all calls for the target species.
                            # export 0/1 annotations for this calltype
                            # (next page will overwrite the same files)
                            segments.exportGT(wavFile, species, window=window, inc=inc)

                # first return value: single array of 0/1 detections over all files
                # second return value: list of tuples ([segments], filename, filelen) for each file
                detected01, detectedS = ws.waveletSegment_test(self.field("testDir"),
                                               subfilter, d=False, rf=True, learnMode='recaa',
                                               savedetections=False, window=window, inc=inc)
                # save the 0/1 annotations as well
                self.wizard().annotations = copy.deepcopy(ws.annotation)

                fB, recall, TP, FP, TN, FN = ws.fBetaScore(ws.annotation, detected01)
                print('--Test summary--\n%d %d %d %d' %(TP, FP, TN, FN))
                total = TP+FP+TN+FN
                if total == 0:
                    print("ERROR: failed to find any testing data")
                    return

                totallen = sum([len(detfile[0]) for detfile in detectedS])

                # update fields
                self.manSegs.setText(str(manSegNum))
                self.manTime.setText("%.1f" % (TP+FN))
                self.autoSegs.setText(str(totallen))
                self.autoTime.setText("%.1f" % (TP+FP))
                self.sensTime.setText("%d %%" % (TP/(TP+FN)*100))

                # Post process:
                print("Post-processing...")
                detectedSpost = []
                self.detected01post = []
                for detfile in detectedS:
                    post = Segment.PostProcess(segments=detfile[0], subfilter=subfilter)
                    print("got segments", len(post.segments))
                    post.fundamentalFrq(fileName=detfile[1])
                    detectedSpost.extend(post.segments)
                    print("kept segments", len(post.segments))

                    # back-convert to 0/1:
                    det01post = np.zeros(detfile[2])
                    for seg in post.segments:
                        det01post[int(seg[0]):int(seg[1])] = 1
                    self.detected01post.extend(det01post)

                # now, detectedS and detectedSpost contain lists of segments before/after post
                # and detected01 and self.detected01post - corresponding pres/abs marks

                # update fields
                _, _, TP, FP, TN, FN = ws.fBetaScore(ws.annotation, self.detected01post)
                print('--Post-processing summary--\n%d %d %d %d' %(TP, FP, TN, FN))
                self.autoSegsP.setText(str(len(detectedSpost)))
                self.autoTimeP.setText("%.1f" % (TP+FP))
                self.sensTimeP.setText("%d %%" % (TP/(TP+FN)*100))

    class WPageLast(QWizardPage):
        def __init__(self, parent=None):
            super(TestRecWizard.WPageLast, self).__init__(parent)
            self.setTitle('Overall testing results')

            self.setMinimumSize(250, 200)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.lblTestDir = QLabel()
            self.lblTestDir.setStyleSheet("QLabel { color : #808080; }")
            self.lblSpecies = QLabel()
            self.lblSpecies.setStyleSheet("QLabel { color : #808080; }")
            space = QLabel()
            space.setFixedHeight(25)

            # final overall results:
            self.labTP = QLabel()
            self.labFP = QLabel()
            self.labTN = QLabel()
            self.labFN = QLabel()
            self.spec = QLabel()
            self.sens = QLabel()
            self.FPR = QLabel()
            self.prec = QLabel()
            self.acc = QLabel()

            # overall results page layout
            vboxHead = QFormLayout()
            vboxHead.addRow("Testing data:", self.lblTestDir)
            vboxHead.addRow("Filter name:", self.lblSpecies)
            vboxHead.addWidget(space)

            form2 = QFormLayout()
            form2.addRow("True positives:", self.labTP)
            form2.addRow("False positives:", self.labFP)
            form2.addRow("True negatives:", self.labTN)
            form2.addRow("False negatives:", self.labFN)
            form2.addWidget(space)
            form2.addRow("Specificity:", self.spec)
            form2.addRow("Recall (sensitivity):", self.sens)
            form2.addRow("FPR:", self.FPR)
            form2.addRow("Precision (PPV):", self.prec)
            form2.addRow("Accuracy:", self.acc)

            vbox = QVBoxLayout()
            vbox.addLayout(vboxHead)
            # vbox.addWidget(QLabel("<b>Detection summary</b>"))
            vbox.addLayout(form2)

            self.setLayout(vbox)

        def initializePage(self):
            speciesData = self.wizard().filterlist[self.field("species")[:-4]]

            self.lblTestDir.setText(self.field("testDir"))
            self.lblSpecies.setText(speciesData["species"])

            ws = WaveletSegment.WaveletSegment(speciesData, 'dmey2')

            # "OR"-combine detection results from each calltype
            first = True
            for pageId in self.wizard().testpages[:-1]:
                page = self.wizard().page(pageId)
                if first:
                    self.detections = page.detected01post
                    first = False
                else:
                    self.detections = np.maximum.reduce([page.detected01post, self.detections])

            # get and parse the agreement metrics
            fB, recall, TP, FP, TN, FN = ws.fBetaScore(self.wizard().annotations, self.detections)
            total = TP+FP+TN+FN
            if total == 0:
                print("ERROR: failed to find any testing data")
                return

            if TP+FN != 0:
                recall = TP/(TP+FN)
            else:
                recall = 0
            if TP+FP != 0:
                precision = TP/(TP+FP)
            else:
                precision = 0
            if TN+FP != 0:
                specificity = TN/(TN+FP)
            else:
                specificity = 0

            accuracy = (TP+TN)/(TP+FP+TN+FN)

            self.labTP.setText("%.1f %%" % (TP*100/total))
            self.labFP.setText("%.1f %%" % (FP*100/total))
            self.labTN.setText("%.1f %%" % (TN*100/total))
            self.labFN.setText("%.1f %%" % (FN*100/total))

            self.spec.setText("%.1f %%" % (specificity*100))
            self.sens.setText("%.1f %%" % (recall*100))
            self.FPR.setText("%.1f %%" % (100-specificity*100))
            self.prec.setText("%.1f %%" % (precision*100))
            self.acc.setText("%.1f %%" % (accuracy*100))

    # Main init of the testing wizard
    def __init__(self, filtdir, parent=None):
        super(TestRecWizard, self).__init__()
        self.setWindowTitle("Test Recogniser")
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        if platform.system() == 'Linux':
            self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        else:
            self.setWindowFlags((self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint) & QtCore.Qt.WindowCloseButtonHint)
        self.setWizardStyle(QWizard.ModernStyle)

        cl = SupportClasses.ConfigLoader()
        self.filterlist = cl.filters(filtdir)
        browsedataPage = TestRecWizard.WPageData()
        browsedataPage.registerField("testDir", browsedataPage.testDirName)
        browsedataPage.registerField("species*", browsedataPage.species, "currentText", browsedataPage.species.currentTextChanged)
        self.addPage(browsedataPage)
        self.testpages = []

    def redoTestPages(self):
        if self.field("species")=="Choose filter...":
            return

        speciesData = self.filterlist[self.field("species")[:-4]]
        print("using filter", speciesData)
        # remove old test pages
        for page in self.testpages:
            self.removePage(page)
        self.testpages = []

        # add new
        for subf in range(len(speciesData["Filters"])):
            testRes = TestRecWizard.WPageTest(subf)
            pageid = self.addPage(testRes)
            self.testpages.append(pageid)

        pageLast = TestRecWizard.WPageLast()
        pageid = self.addPage(pageLast)
        self.testpages.append(pageid)
        self.currentPage().setFinalPage(False)


class ROCCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=6, dpi=100):
        plt.style.use('ggplot')
        self.MList = []
        self.thrList = []
        self.TPR = []
        self.FPR = []
        self.fpr_cl = None
        self.tpr_cl = None
        self.parent = parent

        self.lines = None
        self.plotLines = []

        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

    def plotme(self):
        valid_markers = ([item[0] for item in mks.MarkerStyle.markers.items() if
                          item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])
        markers = np.random.choice(valid_markers, 5, replace=False)

        self.ax = self.figure.subplots()
        for i in range(5):
            self.lines, = self.ax.plot([], [], marker=markers[i])
            self.plotLines.append(self.lines)
        self.ax.set_title('ROC curve')
        self.ax.set_xlabel('False Positive Rate (FPR)')
        self.ax.set_ylabel('True Positive Rate (TPR)')
        # fig.canvas.set_window_title('ROC Curve')
        self.ax.set_ybound(0, 1)
        self.ax.set_xbound(0, 1)
        self.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
        self.ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
        # ax.legend()

    def plotmeagain(self, TPR, FPR):
        # Update data (with the new _and_ the old points)
        for i in range(np.shape(TPR)[0]):
            self.plotLines[i].set_xdata(FPR[i])
            self.plotLines[i].set_ydata(TPR[i])

        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


