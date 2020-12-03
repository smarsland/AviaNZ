
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

# Dialogs used for filter training / testing.
# These are relatively complicated wizards which also do file I/O

import os
import time
import platform
import copy
from shutil import copyfile

from PyQt5.QtGui import QIcon, QValidator, QAbstractItemView, QPixmap, QColor, QFileDialog, QScrollArea
from PyQt5.QtCore import QDir, Qt, QEvent, QSize
from PyQt5.QtWidgets import QLabel, QSlider, QPushButton, QListWidget, QListWidgetItem, QComboBox, QWizard, QWizardPage, QLineEdit, QTextEdit, QSizePolicy, QFormLayout, QVBoxLayout, QHBoxLayout, QCheckBox, QLayout, QApplication, QRadioButton, QGridLayout

import matplotlib.markers as mks
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg

import numpy as np
import colourMaps
import SupportClasses, SupportClasses_GUI
import SignalProc
import WaveletSegment
import WaveletFunctions
import Segment
import Clustering
import Training

class BuildRecAdvWizard(QWizard):
    # page 1 - select training data
    class WPageData(QWizardPage):
        def __init__(self, config, parent=None):
            super(BuildRecAdvWizard.WPageData, self).__init__(parent)
            self.setTitle('Training data')
            self.setSubTitle('To start training, you need labelled calls from your species as training data (see the manual). Select the folder where this data is located. Then select the species.')

            self.setMinimumSize(600, 150)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.trainDirName = QLineEdit()
            self.trainDirName.setReadOnly(True)
            self.btnBrowse = QPushButton('Browse')
            self.btnBrowse.clicked.connect(self.browseTrainData)

            colourNone = QColor(config['ColourNone'][0], config['ColourNone'][1], config['ColourNone'][2], config['ColourNone'][3])
            colourPossibleDark = QColor(config['ColourPossible'][0], config['ColourPossible'][1], config['ColourPossible'][2], 255)
            colourNamed = QColor(config['ColourNamed'][0], config['ColourNamed'][1], config['ColourNamed'][2], config['ColourNamed'][3])
            self.listFiles = SupportClasses_GUI.LightedFileList(colourNone, colourPossibleDark, colourNamed)
            self.listFiles.setMinimumWidth(150)
            self.listFiles.setMinimumHeight(225)
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
            trainDir = QFileDialog.getExistingDirectory(self, 'Choose folder for training')
            self.trainDirName.setText(trainDir)
            self.fillFileList(trainDir)

        def fsChange(self, value):
            value = value // 4000 * 4000
            if value < 4000:
                value = 4000
            self.fstext.setText(str(value))
            self.fs.setValue(value)

        def fillFileList(self, dirName):
            """ Generates the list of files for a file listbox. """
            if not os.path.isdir(dirName):
                print("Warning: directory doesn't exist")
                return

            self.listFiles.fill(dirName, fileName=None, readFmt=True, addWavNum=True, recursive=True)

            # while reading the file, we also collected a list of species present there
            spList = list(self.listFiles.spList)
            # and sample rate info
            fs = list(self.listFiles.fsList)

            if len(fs)==0:
                print("Warning: no suitable files found")
                return

            # might need better limits on selectable sample rate here
            self.fs.setValue(int(np.min(fs)))
            self.fs.setRange(4000, int(np.max(fs)))
            self.fs.setSingleStep(4000)
            self.fs.setTickInterval(4000)

            spList.insert(0, 'Choose species...')
            self.species.clear()
            self.species.addItems(spList)
            if len(spList)==2:
                self.species.setCurrentIndex(1)

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
            self.wizard().button(QWizard.NextButton).setDefault(False)
            self.wizard().saveTestBtn.setVisible(False)
            # parse some params
            fs = int(self.field("fs"))//4000*4000
            if fs not in [8000, 16000, 24000, 32000, 36000, 48000]:
                self.warnLabel.setText("Warning: unusual sampling rate selected, make sure it is intended.")
            else:
                self.warnLabel.setText("")
            self.params.setText("Species: %s\nTraining data: %s\nSampling rate: %d Hz\n" % (self.field("species"), self.field("trainDir"), fs))

    # page 3 - calculate and adjust clusters
    class WPageCluster(QWizardPage):
        def __init__(self, config, parent=None):
            super(BuildRecAdvWizard.WPageCluster, self).__init__(parent)
            self.setTitle('Cluster similar looking calls')
            self.setSubTitle('AviaNZ has tried to identify similar calls in your dataset. Please check the output, and move calls as appropriate.')
            # start larger than minimumSize, but not bigger than the screen:
            screenresol = QApplication.primaryScreen().availableSize()
            self.manualSizeHint = QSize(min(800, 0.9*screenresol.width()), min(600, 0.9*screenresol.height()))
            self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
            self.adjustSize()

            instr = QLabel("To move one call, just drag it with the mouse. To move more, click on them so they are marked with a tick and drag any of them. To merge two types, select all of one group by clicking the empty box next to the name, and then drag any of them. You might also want to name each type of call.")
            instr.setWordWrap(True)

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
            self.hasCTannotations = True

            self.lblSpecies = QLabel()
            self.lblSpecies.setStyleSheet("QLabel { color : #808080; }")

            # Volume control
            self.volSlider = QSlider(Qt.Horizontal)
            self.volSlider.valueChanged.connect(self.volSliderMoved)
            self.volSlider.setRange(0, 100)
            self.volSlider.setValue(50)
            volIcon = QLabel()
            volIcon.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            volIcon.setPixmap(QPixmap('img/volume.png').scaled(18, 18, transformMode=1))

            # Brightness, and contrast sliders
            labelBr = QLabel()
            labelBr.setPixmap(QPixmap('img/brightstr24.png').scaled(18, 18, transformMode=1))
            labelBr.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.brightnessSlider = QSlider(Qt.Horizontal)
            self.brightnessSlider.setMinimum(0)
            self.brightnessSlider.setMaximum(100)
            self.brightnessSlider.setValue(20)
            self.brightnessSlider.setTickInterval(1)
            self.brightnessSlider.valueChanged.connect(self.setColourLevels)

            labelCo = QLabel()
            labelCo.setPixmap(QPixmap('img/contrstr24.png').scaled(18, 18, transformMode=1))
            labelCo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.contrastSlider = QSlider(Qt.Horizontal)
            self.contrastSlider.setMinimum(0)
            self.contrastSlider.setMaximum(100)
            self.contrastSlider.setValue(20)
            self.contrastSlider.setTickInterval(1)
            self.contrastSlider.valueChanged.connect(self.setColourLevels)

            self.btnCreateNewCluster = QPushButton('Create cluster')
            self.btnCreateNewCluster.setFixedWidth(150)
            self.btnCreateNewCluster.clicked.connect(self.createNewcluster)
            self.btnDeleteSeg = QPushButton('Remove selected segment/s')
            self.btnDeleteSeg.setFixedWidth(200)
            self.btnDeleteSeg.clicked.connect(self.deleteSelectedSegs)

            # page 3 layout
            layout1 = QVBoxLayout()
            layout1.addWidget(instr)
            layout1.addWidget(self.lblSpecies)
            hboxSpecContr = QHBoxLayout()
            hboxSpecContr.addWidget(volIcon)
            hboxSpecContr.addWidget(self.volSlider)
            hboxSpecContr.addWidget(labelBr)
            hboxSpecContr.addWidget(self.brightnessSlider)
            hboxSpecContr.addWidget(labelCo)
            hboxSpecContr.addWidget(self.contrastSlider)
            hboxSpecContr.setContentsMargins(20, 0, 20, 10)

            hboxSpecContr.setStretch(0, 1)
            hboxSpecContr.setStretch(1, 3)
            hboxSpecContr.setStretch(2, 1)
            hboxSpecContr.setStretch(3, 3)
            hboxSpecContr.setStretch(4, 1)
            hboxSpecContr.setStretch(5, 3)
            hboxSpecContr.addStretch(2)

            hboxBtns2 = QHBoxLayout()
            hboxBtns2.addWidget(self.btnCreateNewCluster)
            hboxBtns2.addWidget(self.btnDeleteSeg)
            hboxBtns2.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            hboxBtns = QHBoxLayout()
            hboxBtns.addLayout(hboxBtns2)

            # top part
            vboxTop = QVBoxLayout()
            vboxTop.addLayout(hboxSpecContr)
            vboxTop.addLayout(hboxBtns)

            # set up the images
            self.flowLayout = SupportClasses_GUI.Layout()
            self.flowLayout.setMinimumSize(380, 247)
            self.flowLayout.buttonDragged.connect(self.moveSelectedSegs)
            self.flowLayout.layout.setSizeConstraint(QLayout.SetMinimumSize)

            self.scrollArea = QScrollArea(self)
            #self.scrollArea.setWidgetResizable(True)
            self.scrollArea.setWidget(self.flowLayout)
            self.scrollArea.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

            # set overall layout of the dialog
            self.vboxFull = QVBoxLayout()
            self.vboxFull.addLayout(layout1)
            self.vboxFull.addLayout(vboxTop)
            self.vboxFull.addWidget(self.scrollArea)
            self.setLayout(self.vboxFull)

        def initializePage(self):
            self.wizard().saveTestBtn.setVisible(False)
            # parse field shared by all subfilters
            fs = int(self.field("fs"))//4000*4000
            self.wizard().speciesData = {"species": self.field("species"), "SampleRate": fs, "Filters": []}

            with pg.BusyCursor():
                print("Processing. Please wait...")
                # Check if the annotations come with call type labels, if so skip auto clustering
                self.CTannotations()
                if self.hasCTannotations:
                    # self.segments: [parent_audio_file, [segment], class_label]
                    self.segments, self.nclasses, self.duration = self.getClustersGT()
                    self.setSubTitle('AviaNZ found call type annotations in your dataset. You can still make corrections by moving calls as appropriate.')
                else:
                    # return format:
                    # self.segments: [parent_audio_file, [segment], [syllables], [features], class_label]
                    # self.nclasses: number of class_labels
                    # duration: median length of segments
                    self.cluster = Clustering.Clustering([], [], 5)
                    self.segments, self.nclasses, self.duration = self.cluster.cluster(self.field("trainDir"), self.field("fs"), self.field("species"), feature=self.feature)
                    # segments format: [[file1, seg1, [syl1, syl2], [features1, features2], predict], ...]
                    # self.segments, fs, self.nclasses, self.duration = self.cluster.cluster_by_dist(self.field("trainDir"),
                    #                                                                              self.field("species"),
                    #                                                                              feature=self.feature,
                    #                                                                              max_clusters=5,
                    #                                                                              single=True)
                    self.setSubTitle('AviaNZ has tried to identify similar calls in your dataset. Please check the output, and move calls as appropriate.')

                # Create and show the buttons
                self.clearButtons()
                self.addButtons()
                self.updateButtons()
                print("buttons added")
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
                self.segsChanged = False
                self.wizard().redoTrainPages()
            return True

        def validatePage(self):
            self.updateAnnotations()
            return True

        def cleanupPage(self):
            self.clusters = {}

        def CTannotations(self):
            """ Check if all the segments from target species has call type annotations"""
            self.hasCTannotations = True
            listOfDataFiles = QDir(self.field("trainDir")).entryList(['*.data'])
            for file in listOfDataFiles:
                # Read the annotation
                segments = Segment.SegmentList()
                segments.parseJSON(os.path.join(self.field("trainDir"), file))
                SpSegs = segments.getSpecies(self.field("species"))
                for segix in SpSegs:
                    seg = segments[segix]
                    for label in seg[4]:
                        if label["species"] == self.field("species") and "calltype" not in label:
                            self.hasCTannotations = False
                            break

        def getClustersGT(self):
            """ Gets call type clusters from annotations
             returns [parent_audio_file, [segment], [syllables], class_label], number of clusters, median duration
            """
            ctTexts = []
            CTsegments = []
            duration = []
            cl = Clustering.Clustering([], [], 5)

            listOfDataFiles = QDir(self.field("trainDir")).entryList(['*.data'])
            listOfWavFiles = QDir(self.field("trainDir")).entryList(['*.wav'])
            for file in listOfDataFiles:
                if file[:-5] in listOfWavFiles:
                    # Read the annotation
                    segments = Segment.SegmentList()
                    segments.parseJSON(os.path.join(self.field("trainDir"), file))
                    SpSegs = segments.getSpecies(self.field("species"))
                    for segix in SpSegs:
                        seg = segments[segix]
                        for label in seg[4]:
                            if label["species"] == self.field("species") and "calltype" in label:
                                if label["calltype"] not in ctTexts:
                                    ctTexts.append(label["calltype"])
            for i in range(len(ctTexts)):
                self.clusters[i] = ctTexts[i]

            for file in listOfDataFiles:
                if file[:-5] in listOfWavFiles:
                    # Read the annotation
                    segments = Segment.SegmentList()
                    wavfile = os.path.join(self.field("trainDir"), file[:-5])
                    segments.parseJSON(os.path.join(self.field("trainDir"), file))
                    SpSegs = segments.getSpecies(self.field("species"))
                    for segix in SpSegs:
                        seg = segments[segix]
                        for label in seg[4]:
                            if label["species"] == self.field("species") and "calltype" in label:
                                # Find the syllables inside this segment
                                # TODO: Filter all the hardcoded parameters into a .txt in config (minlen=0.2, denoise=False)
                                if seg[2] == seg[3] == 0:
                                    syls = cl.findSyllablesSeg(file=wavfile, seg=seg, fs=self.field("fs"), f1=seg[2], f2=seg[3], denoise=False, minlen=0.2)
                                else:
                                    syls = cl.findSyllablesSeg(file=wavfile, seg=seg, fs=self.field("fs"), f1=0, f2=0, denoise=False, minlen=0.2)
                                CTsegments.append([wavfile, seg, syls, list(self.clusters.keys())[list(self.clusters.values()).index(label["calltype"])]])
                                duration.append(seg[1]-seg[0])
            return CTsegments, len(self.clusters), np.median(duration)

        def backupDatafiles(self):
            # Backup original data fiels before updating them
            print("Backing up files ", self.field("trainDir"))
            listOfDataFiles = QDir(self.field("trainDir")).entryList(['*.data'])
            for file in listOfDataFiles:
                source = self.field("trainDir") + '/' + file
                destination = source[:-5] + ".backup"
                if os.path.isfile(destination):
                    pass
                else:
                    copyfile(source, destination)

        def updateAnnotations(self):
            """ Update annotation files. Assign call types suggested by clusters and remove any segment deleted in the
            clustering. Keep a backup of the original .data."""
            self.backupDatafiles()
            print("Updating annotation files ", self.field("trainDir"))
            listOfDataFiles = QDir(self.field("trainDir")).entryList(['*.data'])
            for file in listOfDataFiles:
                # Read the annotation
                segments = Segment.SegmentList()
                newsegments = Segment.SegmentList()
                segments.parseJSON(os.path.join(self.field("trainDir"), file))
                allSpSegs = np.arange(len(segments)).tolist()
                newsegments.metadata = segments.metadata
                for segix in allSpSegs:
                    seg = segments[segix]
                    if self.field("species") not in [fil["species"] for fil in seg[4]]:
                        newsegments.addSegment(seg) # leave non-target segments unchanged
                    else:
                        for seg2 in self.segments:
                            if seg2[1] == seg:
                                # find the index of target sp and update call type
                                seg[4][[fil["species"] for fil in seg[4]].index(self.field("species"))]["calltype"] = self.clusters[seg2[-1]]
                                newsegments.addSegment(seg)
                newsegments.saveJSON(os.path.join(self.field("trainDir"), file))

        def merge(self):
            """ Listener for the merge button. Merge the rows (clusters) checked into one cluster.
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
            #self.cmbUpdateSeg.clear()
            #for x in self.clusters:
                #self.cmbUpdateSeg.addItem(self.clusters[x])

            # Clean and redraw
            self.clearButtons()
            self.updateButtons()
            self.completeChanged.emit()

        def moveSelectedSegs(self,dragPosy,source):
            """ Listener for Apply button to move the selected segments to another cluster.
                Change the cluster ID of those selected buttons and redraw all the clusters.
            """
            # TODO: check: I think the dict is always in descending order down screen?
            self.segsChanged = True
            # The first line seemed neater, but the verticalSpacing() doesn't update when you rescale the window
            #movetoID = dragPosy//(self.picbuttons[0].size().height()+self.flowLayout.layout.verticalSpacing())
            movetoID = dragPosy//(self.flowLayout.layout.geometry().height()//self.nclasses)

            # drags which start and end in the same cluster most likely were just long clicks:
            for ix in range(len(self.picbuttons)):
                if self.picbuttons[ix] == source:
                    if self.segments[ix][-1] == movetoID:
                        source.clicked.emit()
                        return

            # Even if the button that was dragged isn't highlighted, make it so
            source.mark = 'yellow'

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
            self.updateClusterNames()
            self.completeChanged.emit()

        def createNewcluster(self):
            """ Listener for Create cluster button to move the selected segments to a new cluster.
                Change the cluster ID of those selected buttons and redraw all the clusters.
            """
            self.segsChanged = True

            # There should be at least one segment selected to proceed
            proceed = False
            for ix in range(len(self.picbuttons)):
                if self.picbuttons[ix].mark == 'yellow':
                    proceed = True
                    break

            if proceed:
                # User to enter new cluster name
                #newLabel, ok = QInputDialog.getText(self, 'Cluster name', 'Enter unique Cluster Name\t\t\t')
                #if not ok:
                    #self.completeChanged.emit()
                    #return
                names = [self.tboxes[ID].text() for ID in range(self.nclasses)]
                nextNumber = 0
                newLabel = 'Cluster_'+str(nextNumber)
                names.append(newLabel)
                while len(names) != len(set(names)):
                    del(names[-1])
                    nextNumber += 1
                    newLabel = 'Cluster_'+str(nextNumber)
                    names.append(newLabel)

                # create new cluster ID, label
                newID = len(self.clusters)
                self.clusters[newID] = newLabel
                self.nclasses += 1
                print('after adding new cluster: ', self.clusters)

                for ix in range(len(self.picbuttons)):
                    if self.picbuttons[ix].mark == 'yellow':
                        self.segments[ix][-1] = newID
                        self.picbuttons[ix].mark = 'green'

                # Delete clusters with no members left and update self.clusters before adding the new cluster
                todelete = []
                for ID, label in self.clusters.items():
                    empty = True
                    for seg in self.segments:
                        if seg[-1] == ID:
                            empty = False
                            break
                    if empty:
                        todelete.append(ID)

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

                    print('before: ', self.clusters)
                    self.clusters = clusters
                    self.nclasses = nclasses
                    print('after: ', self.clusters)

                    # update the segments
                    for seg in self.segments:
                        seg[-1] = labels[seg[-1]]
                # redraw the buttons
                self.clearButtons()
                self.updateButtons()
                #self.cmbUpdateSeg.addItem(newLabel)
                self.completeChanged.emit()
            else:
                msg = SupportClasses_GUI.MessagePopup("t", "Select", "Select calls to make the new cluster")
                msg.exec_()
                self.completeChanged.emit()
                return

        def deleteSelectedSegs(self):
            """ Listener for Delete button to delete the selected segments completely.
            """
            inds = []
            for ix in range(len(self.picbuttons)):
                if self.picbuttons[ix].mark == 'yellow':
                    inds.append(ix)

            if len(inds)==0:
                print("No segments selected")
                return

            self.segsChanged = True
            for ix in reversed(inds):
                del self.segments[ix]
                del self.picbuttons[ix]

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

            # redraw the buttons
            self.updateButtons()
            self.completeChanged.emit()

        def updateClusterNames(self):
            # Check duplicate names
            self.segsChanged = True
            names = [self.tboxes[ID].text() for ID in range(self.nclasses)]
            if len(names) != len(set(names)):
                msg = SupportClasses_GUI.MessagePopup("w", "Name error", "Duplicate cluster names! \nTry again")
                msg.exec_()
                self.completeChanged.emit()
                return

            if "(Other)" in names:
                msg = SupportClasses_GUI.MessagePopup("w", "Name error", "Name \"(Other)\" is reserved! \nTry again")
                msg.exec_()
                self.completeChanged.emit()
                return

            for ID in range(self.nclasses):
                self.clusters[ID] = self.tboxes[ID].text()

            self.completeChanged.emit()
            print('updated clusters: ', self.clusters)

        def addButtons(self):
            """ Only makes the PicButtons and self.clusters dict
            """
            self.picbuttons = []
            if not self.hasCTannotations:
                self.clusters = []
                for i in range(self.nclasses):
                    self.clusters.append((i, 'Cluster_' + str(i)))
                self.clusters = dict(self.clusters)     # Dictionary of {ID: cluster_name}

            # largest spec will be this wide
            maxspecsize = max([seg[1][1]-seg[1][0] for seg in self.segments]) * self.field("fs") // 256

            # Create the buttons for each segment
            for seg in self.segments:
                sp = SignalProc.SignalProc(512, 256)
                sp.readWav(seg[0], seg[1][1]-seg[1][0], seg[1][0], silent=True)

                # set increment to depend on Fs to have a constant scale of 256/tgt seconds/px of spec
                incr = 256 * sp.sampleRate // self.field("fs")
                sgRaw = sp.spectrogram(window='Hann', sgType='Standard',incr=incr, mean_normalise=True, onesided=True, need_even=False)
                maxsg = np.min(sgRaw)
                self.sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))
                self.setColourMap()

                # buffer the image to largest spec size, so that the resulting buttons would have equal scale
                if self.sg.shape[0]<maxspecsize:
                    padlen = int(maxspecsize - self.sg.shape[0])//2
                    sg = np.pad(self.sg, ((padlen, padlen), (0,0)), 'constant', constant_values=np.quantile(self.sg, 0.1))
                else:
                    sg = self.sg

                newButton = SupportClasses_GUI.PicButton(1, np.fliplr(sg), sp.data, sp.audioFormat, seg[1][1]-seg[1][0], 0, seg[1][1], self.lut, self.colourStart, self.colourEnd, False, cluster=True)
                self.picbuttons.append(newButton)
            # (updateButtons will place them in layouts and show them)

        def selectAll(self):
            """ Tick all buttons in the row and vise versa"""
            for ID in range(len(self.cboxes)):
                if self.cboxes[ID].isChecked():
                    for ix in range(len(self.segments)):
                        if self.segments[ix][-1] == ID:
                            self.picbuttons[ix].mark = 'yellow'
                            self.picbuttons[ix].buttonClicked = True
                            self.picbuttons[ix].setChecked(True)
                            self.picbuttons[ix].repaint()
                else:
                    for ix in range(len(self.segments)):
                        if self.segments[ix][-1] == ID:
                            self.picbuttons[ix].mark = 'green'
                            self.picbuttons[ix].buttonClicked = False
                            self.picbuttons[ix].setChecked(False)
                            self.picbuttons[ix].repaint()

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
                tbox.setStyleSheet("border: none;")
                tbox.setAlignment(Qt.AlignCenter)
                tbox.textChanged.connect(self.updateClusterNames)
                self.tboxes.append(tbox)
                self.flowLayout.addWidget(self.tboxes[-1], r, c)
                c += 1
                cbox = QCheckBox("")
                cbox.clicked.connect(self.selectAll)
                self.cboxes.append(cbox)
                self.flowLayout.addWidget(self.cboxes[-1], r, c)
                c += 1
                # Find the segments under this class and show them
                for segix in range(len(self.segments)):
                    if self.segments[segix][-1] == r:
                        self.flowLayout.addWidget(self.picbuttons[segix], r, c)
                        c += 1
                        self.picbuttons[segix].show()
            self.flowLayout.adjustSize()
            self.flowLayout.update()
            self.setColourLevels()

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
            #self.setMinimumSize(350, 430)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
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
            imgCluster.setFixedHeight(100)
            picimg = QPixmap.fromImage(picbtn.im1)
            imgCluster.setPixmap(picimg.scaledToHeight(100))

            # TimeRange parameters
            form1_step4 = QFormLayout()
            self.minlen = QLineEdit(self)
            self.minlen.setText('')
            form1_step4.addRow('Min call length (sec)', self.minlen)
            self.maxlen = QLineEdit(self)
            self.maxlen.setText('')
            form1_step4.addRow('Max call length (sec)', self.maxlen)
            self.avgslen = QLineEdit(self)
            self.avgslen.setText('')
            form1_step4.addRow('Avg syllable length (sec)', self.avgslen)
            self.maxgap = QLineEdit(self)
            self.maxgap.setText('')
            form1_step4.addRow('Max gap between syllables (sec)', self.maxgap)

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
            self.setLayout(layout_step4)

            self.setButtonText(QWizard.NextButton, 'Train >')

        def fLowChange(self, value):
            value = value//10*10
            if value < 0:
                value = 0
            self.fLow.setValue(value)
            self.fLowtext.setText(str(value))

        def fHighChange(self, value):
            value = value//10*10
            if value < 100:
                value = 100
            self.fHigh.setValue(value)
            self.fHightext.setText(str(value))

        def initializePage(self):
            self.wizard().saveTestBtn.setVisible(False)
            # populates values based on training files
            fs = int(self.field("fs")) // 4000 * 4000

            # self.segments is already selected to be this cluster only
            pageSegs = Segment.SegmentList()
            for longseg in self.segments:
                # long seg has format: [file [segment] clusternum]
                pageSegs.addSegment(longseg[1])
            len_min, len_max, f_low, f_high = pageSegs.getSummaries()
            # Get max inter syllable gap
            if not self.wizard().clusterPage.hasCTannotations:
                gaps = []
                syllen = []
                for longseg in self.segments:
                    if len(longseg[2]) > 1:
                        for i in range(len(longseg[2]) - 1):
                            gaps.append(longseg[2][i + 1][0] - longseg[2][i][1])
                if len(gaps) > 0:
                    maxgap = max(gaps)
                else:
                    maxgap = 0

                for longseg in self.segments:
                    for i in range(len(longseg[2])):
                        syllen.append(longseg[2][i][1] - longseg[2][i][0])
                avgslen = np.mean(syllen)
            else:
                maxgap = 0.5    # TODO
                avgslen = 0.5

            self.minlen.setText(str(round(np.min(len_min), 2)))
            self.maxlen.setText(str(round(np.max(len_max), 2)))
            self.maxgap.setText(str(round(maxgap, 2)))
            self.avgslen.setText(str(round(avgslen, 2)))
            self.fLow.setRange(0, fs/2)
            self.fLow.setValue(max(0, int(np.min(f_low))))
            self.fHigh.setRange(0, fs/2)
            if np.max(f_high) == 0:
                # happens when no segments have y limits
                f_high = fs/2
            self.fHigh.setValue(min(fs/2,int(np.max(f_high))))
            self.adjustSize()

    # page 5 - run training, show ROC
    class WPageTrain(QWizardPage):
        def __init__(self, id, clustID, clustname, segments, parent=None):
            super(BuildRecAdvWizard.WPageTrain, self).__init__(parent)
            self.setTitle('Training results')
            self.setSubTitle('Click on the graph at the point where you would like the classifier to trade-off false positives with false negatives. Points closest to the top-left are best.')
            self.setMinimumSize(520, 440)
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
            space.setFixedHeight(20)
            spaceH = QLabel()
            spaceH.setFixedWidth(30)

            self.lblUpdate = QLabel()
            self.lblUpdate.setStyleSheet("QLabel { font-weight: bold; }")

            # These are connected to fields and actually control the wizard's flow
            self.bestM = QLineEdit()
            self.bestThr = QLineEdit()
            self.bestNodes = QLineEdit()
            self.bestFrqBands = QLineEdit()
            self.bestM.setReadOnly(True)
            self.bestThr.setReadOnly(True)
            self.bestNodes.setReadOnly(True)
            self.bestFrqBands.setReadOnly(True)
            self.bestM.setStyleSheet("QLineEdit { color : #808080; }")
            self.bestThr.setStyleSheet("QLineEdit { color : #808080; }")
            self.bestNodes.setStyleSheet("QLineEdit { color : #808080; }")
            self.bestFrqBands.setStyleSheet("QLineEdit { color : #808080; }")
            self.filtSummary = QFormLayout()
            self.filtSummary.addRow("Current M:", self.bestM)
            self.filtSummary.addRow("Threshold:", self.bestThr)
            self.filtSummary.addRow("Wavelet nodes:", self.bestNodes)
            self.filtSummary.addRow("Frequency bands (Hz):", self.bestFrqBands)

            self.selectedTPR = QLineEdit()
            self.selectedFPR = QLineEdit()
            self.saveStat = QCheckBox("Save TPR, FPR to the recogniser")
            self.saveStat.setVisible(False)

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
                self.tpr_near = self.TPR[M_min_ind, thr_min_ind]
                self.fpr_near = self.FPR[M_min_ind, thr_min_ind]
                self.marker.set_visible(False)
                self.figCanvas.draw()
                self.marker.set_xdata([fpr_cl, self.fpr_near])
                self.marker.set_ydata([tpr_cl, self.tpr_near])
                self.marker.set_visible(True)
                self.figCanvas.ax.draw_artist(self.marker)
                self.figCanvas.update()

                print("fpr_cl, tpr_cl: ", self.fpr_near, self.tpr_near)

                # update sidebar
                self.lblUpdate.setText('Detection Summary\n\nTPR:\t' + str(round(self.tpr_near * 100, 2)) +
                                       '%\nFPR:\t' + str(round(self.fpr_near * 100, 2)) + '%')

                # this will save the best parameters to the global fields
                self.bestM.setText("%.4f" % self.MList[M_min_ind])
                self.bestThr.setText("%.4f" % self.thrList[thr_min_ind])
                # Get nodes for closest point
                optimumNodesSel = self.nodes[M_min_ind][thr_min_ind]
                self.bestNodes.setText(str(optimumNodesSel))
                # corresponding frequency bands
                optimumNodesBand = self.getFrqBands(optimumNodesSel)
                self.bestFrqBands.setText(str(optimumNodesBand))
                self.saveStat.setVisible(True)
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

            vboxStats = QVBoxLayout()
            vboxStats.addWidget(self.lblUpdate)
            vboxStats.addWidget(self.saveStat)

            hbox3 = QHBoxLayout()
            hbox3.addLayout(self.filtSummary)
            hbox3.addWidget(spaceH)
            hbox3.addLayout(vboxStats)

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
            self.tpr_near = -1
            self.fpr_near = -1

            # parse fields specific to this subfilter
            minlen = float(self.field("minlen"+str(self.pageId)))
            maxlen = float(self.field("maxlen"+str(self.pageId)))
            maxgap = float(self.field("maxgap" + str(self.pageId)))
            avgslen = float(self.field("avgslen" + str(self.pageId)))
            fLow = int(self.field("fLow"+str(self.pageId)))
            fHigh = int(self.field("fHigh"+str(self.pageId)))
            numthr = 50
            # note: for each page we reset the filter to contain 1 calltype
            self.wizard().speciesData["Filters"] = [{'calltype': self.clust, 'TimeRange': [minlen, maxlen, avgslen, maxgap], 'FreqRange': [fLow, fHigh]}]

            # export 1/0 ground truth
            window = 1
            inc = None
            with pg.BusyCursor():
                for root, dirs, files in os.walk(self.field("trainDir")):
                    for file in files:
                        wavFile = os.path.join(root, file)
                        if file.lower().endswith('.wav') and os.stat(wavFile).st_size != 0 and file + '.data' in files:
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
                cl = Clustering.Clustering([], [], 5)
                self.clustercentre = cl.getClusterCenter(self.segments, self.field("fs"), fLow, fHigh, self.wizard().clusterPage.feature, self.wizard().clusterPage.duration)

            # Get detection measures over all M,thr combinations
            print("starting wavelet training")
            with pg.BusyCursor():
                opstartingtime = time.time()
                ws = WaveletSegment.WaveletSegment(self.wizard().speciesData)
                # returns 2d lists of nodes over M x thr, or stats over M x thr
                self.thrList = np.linspace(0.2, 1, num=numthr)
                # self.MList = np.linspace(0.25, 1.5, num=numM)
                self.MList = np.linspace(avgslen, avgslen, num=1)
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

        def getFrqBands(self, nodes):
            WF = WaveletFunctions.WaveletFunctions(data=[], wavelet='dmey2', maxLevel=1, samplerate=self.field("fs"))
            fRanges = []
            for node in nodes:
                f1, f2 = WF.getWCFreq(node, self.field("fs"))
                print(node, f1, f2)
                fRanges.append([f1, f2])
            return fRanges

        def cleanupPage(self):
            self.lblUpdate.setText('')

        def isComplete(self):
            if self.tpr_near == self.fpr_near == -1:
                return False
            else:
                return True

        def validatePage(self):
            if self.saveStat.isChecked():
                self.selectedTPR.setText(str(round(self.tpr_near * 100, 2)))
                self.selectedFPR.setText(str(round(self.fpr_near * 100, 2)))
            else:
                self.selectedTPR.setText(str(-1))
                self.selectedFPR.setText(str(-1))
            return True

    # page 6 - save the filter
    class WLastPage(QWizardPage):
        def __init__(self, filtdir, parent=None):
            super(BuildRecAdvWizard.WLastPage, self).__init__(parent)
            self.setTitle('Save recogniser')
            self.setSubTitle('If you are happy with the overall call detection summary, save the recogniser. \n You should now test it.')
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
            self.listFiles.setMinimumHeight(200)
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
                    if input=="M.txt":
                        print("filter name \"M\" reserved for manual annotations")
                        return(QValidator.Intermediate, input, pos)
                    elif self.listFiles.findItems(input, Qt.MatchExactly):
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

            layout = QVBoxLayout()
            layout.addLayout(vboxHead)
            layout.addWidget(space)
            layout.addWidget(QLabel("The following recogniser was produced:"))
            layout.addWidget(self.lblFilter)
            layout.addWidget(QLabel("Currently available recognisers"))
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
                maxgap = float(self.field("maxgap" + str(pageId)))
                avgslen = float(self.field("avgslen" + str(pageId)))
                fLow = int(self.field("fLow"+str(pageId)))
                fHigh = int(self.field("fHigh"+str(pageId)))
                thr = float(self.field("bestThr"+str(pageId)))
                M = float(self.field("bestM"+str(pageId)))
                nodes = eval(self.field("bestNodes"+str(pageId)))
                tpr = float(self.field("TPR" + str(pageId)))
                fpr = float(self.field("FPR" + str(pageId)))

                if tpr != -1:
                    newSubfilt = {'calltype': self.wizard().page(pageId + 1).clust,
                                  'TimeRange': [minlen, maxlen, avgslen, maxgap], 'FreqRange': [fLow, fHigh],
                                  'WaveletParams': {"thr": thr, "M": M, "nodes": nodes},
                                  'ClusterCentre': list(self.wizard().page(pageId + 1).clustercentre),
                                  'Feature': self.wizard().clusterPage.feature, 'TPR, FPR': [tpr, fpr]}

                else:
                    newSubfilt = {'calltype': self.wizard().page(pageId + 1).clust,
                                  'TimeRange': [minlen, maxlen, avgslen, maxgap], 'FreqRange': [fLow, fHigh],
                                  'WaveletParams': {"thr": thr, "M": M, "nodes": nodes},
                                  'ClusterCentre': list(self.wizard().page(pageId + 1).clustercentre),
                                  'Feature': self.wizard().clusterPage.feature}

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

        def cleanupPage(self):
            self.wizard().saveTestBtn.setVisible(False)
            super(BuildRecAdvWizard.WLastPage, self).cleanupPage()

    # Main init of the training wizard
    def __init__(self, filtdir, config, parent=None):
        super(BuildRecAdvWizard, self).__init__()
        self.setWindowTitle("Build Recogniser")
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        if platform.system() == 'Linux':
            self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        else:
            self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setWizardStyle(QWizard.ModernStyle)

        # add the Save & Test button
        self.saveTestBtn = QPushButton("Save and Test")
        self.setButton(QWizard.CustomButton1, self.saveTestBtn)
        self.setButtonLayout([QWizard.Stretch, QWizard.BackButton, QWizard.NextButton, QWizard.CustomButton1, QWizard.FinishButton, QWizard.CancelButton])
        self.setOptions(QWizard.NoBackButtonOnStartPage | QWizard.HaveCustomButton1)

        self.filtersDir = filtdir

        # page 1: select training data
        browsedataPage = BuildRecAdvWizard.WPageData(config)
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
        # try to deal with buttons catching Enter presses
        self.buttons = [self.button(t) for t in (1, 3, 6)]
        for btn in self.buttons:
            btn.installEventFilter(self)

    def redoTrainPages(self):
        self.speciesData["Filters"] = []
        for page in self.trainpages:
            # for each calltype, remove params, ROC, FF pages
            self.removePage(page)
            self.removePage(page+1)
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
            page4.registerField("maxgap" + str(pageid), page4.maxgap)
            page4.registerField("avgslen" + str(pageid), page4.avgslen)
            page4.registerField("fLow"+str(pageid), page4.fLow)
            page4.registerField("fHigh"+str(pageid), page4.fHigh)

            # page 5: get training results
            page5 = BuildRecAdvWizard.WPageTrain(pageid, key, value, newsegs)
            self.addPage(page5)

            # note: pageid is the same for both page fields
            page5.registerField("bestThr"+str(pageid)+"*", page5.bestThr)
            page5.registerField("bestM"+str(pageid)+"*", page5.bestM)
            page5.registerField("bestNodes"+str(pageid)+"*", page5.bestNodes)
            page5.registerField("TPR" + str(pageid) + "*", page5.selectedTPR)
            page5.registerField("FPR" + str(pageid) + "*", page5.selectedFPR)

        # page 6: confirm the results & save
        page6 = BuildRecAdvWizard.WLastPage(self.filtersDir)
        pageid = self.addPage(page6)
        # (store this as well, so that we could wipe it without worrying about page order)
        self.trainpages.append(pageid)
        page6.registerField("filtfile*", page6.enterFiltName)

        self.clusterPage.setFinalPage(False)
        self.clusterPage.completeChanged.emit()

    def pageChangeResize(self, pageid):
        # wizard dialog size needs to refresh when pages are flipped
        try:
            if self.page(pageid) is not None:
                # do not minimize the clustering page
                if pageid==2:
                    self.resize(self.page(pageid).manualSizeHint)
                else:
                    newsize = self.page(pageid).sizeHint()
                    # need tiny adjustment for parameter pages
                    if pageid in self.trainpages:
                        newsize.setHeight(newsize.height()+80)
                    elif pageid-1 in self.trainpages:
                        newsize.setWidth(newsize.width()+100)
                        newsize.setHeight(newsize.height()+135)
                    elif pageid-2 in self.trainpages:
                        newsize.setHeight(newsize.height()+170)
                    # print("Resizing to", newsize)
                    self.setMinimumSize(newsize)
                    self.adjustSize()
                    # print("Current size", self.size())
        except Exception as e:
            print(e)

    def eventFilter(self, obj, event):
        # disable accidentally pressing Enter
        if obj in self.buttons and event.type() == QEvent.Show:
            obj.setDefault(False)
        return super(BuildRecAdvWizard, self).eventFilter(obj, event)

class TestRecWizard(QWizard):
    class WPageData(QWizardPage):
        def __init__(self, config, filter=None, parent=None):
            super(TestRecWizard.WPageData, self).__init__(parent)
            self.setTitle('Testing data')
            self.setSubTitle('Select the folder with testing data, then choose species')

            self.setMinimumSize(250, 150)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            # the combobox will default to this filter initially if provided
            self.initialFilter = filter

            self.testDirName = QLineEdit()
            self.testDirName.setReadOnly(True)
            self.btnBrowse = QPushButton('Browse')
            self.btnBrowse.clicked.connect(self.browseTestData)

            colourNone = QColor(config['ColourNone'][0], config['ColourNone'][1], config['ColourNone'][2], config['ColourNone'][3])
            colourPossibleDark = QColor(config['ColourPossible'][0], config['ColourPossible'][1], config['ColourPossible'][2], 255)
            colourNamed = QColor(config['ColourNamed'][0], config['ColourNamed'][1], config['ColourNamed'][2], config['ColourNamed'][3])
            self.listFiles = SupportClasses_GUI.LightedFileList(colourNone, colourPossibleDark, colourNamed)
            self.listFiles.setMinimumWidth(150)
            self.listFiles.setMinimumHeight(275)
            self.listFiles.setSelectionMode(QAbstractItemView.NoSelection)

            selectSpLabel = QLabel("Choose the recogniser that you want to test")
            self.species = QComboBox()  # fill during browse
            self.species.addItems(['Choose recogniser...'])

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
            if self.initialFilter is not None:
                self.species.setCurrentText(self.initialFilter)

        def browseTestData(self):
            dirName = QFileDialog.getExistingDirectory(self, 'Choose folder for testing')
            self.testDirName.setText(dirName)

            self.listFiles.fill(dirName, fileName=None, readFmt=False, addWavNum=True, recursive=True)

    class WPageMain(QWizardPage):
        def __init__(self, configdir, filterdir, parent=None):
            super(TestRecWizard.WPageMain, self).__init__(parent)
            self.setTitle('Summary of testing results')

            self.setMinimumSize(300, 300)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()
            self.configdir = configdir
            self.filterdir = filterdir

            self.lblTestDir = QLabel()
            self.lblTestDir.setStyleSheet("QLabel { color : #808080; }")
            self.lblTestFilter = QLabel()
            self.lblTestFilter.setStyleSheet("QLabel { color : #808080; }")
            self.lblSpecies = QLabel()
            self.lblSpecies.setStyleSheet("QLabel { color : #808080; }")
            space = QLabel()
            space.setFixedHeight(25)

            self.lblWFsummary = QLabel()
            # self.lblWFCNNsummary = QLabel()
            # self.lblWFsummary.setStyleSheet("QLabel { color : #808080; }")
            # self.lblWFCNNsummary.setStyleSheet("QLabel { color : #808080; }")
            self.lblOutfile = QLabel()

            # page layout
            vboxHead = QFormLayout()
            vboxHead.addRow("Testing data:", self.lblTestDir)
            vboxHead.addRow("Filter name:", self.lblTestFilter)
            vboxHead.addRow("Species name:", self.lblSpecies)
            vboxHead.addWidget(space)
            vbox = QVBoxLayout()
            vbox.addLayout(vboxHead)
            vbox.addWidget(self.lblWFsummary)
            # vbox.addWidget(self.lblWFCNNsummary)
            vbox.addWidget(self.lblOutfile)
            self.setLayout(vbox)

        def initializePage(self):
            # Testing results will be stored there
            #testresfile = os.path.join(self.field("testDir"), "test-results.txt")
            # Run the actual testing here:
            with pg.BusyCursor():
                self.currfilt = self.wizard().filterlist[self.field("species")[:-4]]

                self.lblTestDir.setText(self.field("testDir"))
                self.lblTestFilter.setText(self.field("species"))
                self.lblSpecies.setText(self.currfilt['species'])

                test = Training.CNNtest(self.field("testDir"), self.currfilt, self.field("species")[:-4], self.configdir,self.filterdir)
                text = test.getOutput()

            if text == 0:
                self.lblWFsummary.setText("No segments for species \'%s\' found!" % self.field("species")[:-4])
                return

            self.lblWFsummary.setText(text)
            resfile = os.path.join(self.field("testDir"), "test-results.txt")
            self.lblOutfile.setText("The detailed results have been saved in file\n%s" % resfile)

        def cleanupPage(self):
            self.lblWFsummary.setText('')
            self.lblWFCNNsummary.setText('')
            self.lblSpecies.setText('')
            self.lblTestDir.setText('')
            self.lblTestFilter.setText('')

        def validatePage(self):
            # Clean tmpdata
            for root, dirs, files in os.walk(self.field("testDir")):
                for file in files:
                    if file.endswith('.tmpdata'):
                        os.remove(os.path.join(root, file))
            return True

    # extra page to display the full results?
    # class WPageFull(QWizardPage):
    #     def __init__(self, parent=None):
    #         super(TestRecWizard.WPageFull, self).__init__(parent)
    #         self.setTitle('Detailed testing results')

    #         self.setMinimumSize(300, 300)
    #         self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    #         self.adjustSize()

    #         self.results = QTextEdit()
    #         self.results.setReadOnly(True)

    #         vbox = QVBoxLayout()
    #         vbox.addWidget(self.results)
    #         self.setLayout(vbox)

    #     def initializePage(self):
    #         resfile = os.path.join(self.field("testDir"), "test-results.txt")
    #         resstream = open(resfile, 'r')
    #         self.results.setPlainText(resstream.read())
    #         resstream.close()

    #     def cleanupPage(self):
    #         self.results.setPlainText('')

    # Main init of the testing wizard
    def __init__(self, filtdir, configdir, filter=None, parent=None):
        super(TestRecWizard, self).__init__()
        self.setWindowTitle("Test Recogniser")
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        if platform.system() == 'Linux':
            self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        else:
            self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOptions(QWizard.NoBackButtonOnStartPage)

        cl = SupportClasses.ConfigLoader()
        self.filterlist = cl.filters(filtdir, bats=False)
        configfile = os.path.join(configdir, "AviaNZconfig.txt")
        ConfigLoader = SupportClasses.ConfigLoader()
        config = ConfigLoader.config(configfile)
        browsedataPage = TestRecWizard.WPageData(config, filter=filter)
        browsedataPage.registerField("testDir*", browsedataPage.testDirName)
        browsedataPage.registerField("species*", browsedataPage.species, "currentText", browsedataPage.species.currentTextChanged)
        self.addPage(browsedataPage)

        pageMain = TestRecWizard.WPageMain(configdir, filtdir)
        self.addPage(pageMain)

        # extra page to show more details
        # self.addPage(TestRecWizard.WPageFull())

class ROCCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=6, dpi=100):
        # reduce size on low-res monitors
        aspectratio = width/height
        height = min(height, 0.6*(QApplication.primaryScreen().availableSize().height()-150)/dpi)
        width = height*aspectratio
        # print("resetting to", height, width)
        plt.style.use('ggplot')
        self.parent = parent

        self.lines = None
        self.plotLines = []

        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.set_tight_layout(True)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

    def plotme(self):
        # valid_markers = ([item[0] for item in mks.MarkerStyle.markers.items() if
        #                   item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])
        # markers = np.random.choice(valid_markers, 5, replace=False)

        self.ax = self.figure.subplots()
        for i in range(5):
            self.lines, = self.ax.plot([], [], marker='o')
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

    def plotmeagain(self, TPR, FPR, CNN=False):
        # Update data (with the new _and_ the old points)
        if CNN:
            self.lines, = self.ax.plot(FPR, TPR, marker='o')
            self.plotLines.append(self.lines)
        else:
            for i in range(np.shape(TPR)[0]):
                self.plotLines[i].set_xdata(FPR[i])
                self.plotLines[i].set_ydata(TPR[i])

        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

class BuildCNNWizard(QWizard):
    # Main init of the CNN training wizard
    def __init__(self, filtdir, config, configdir, parent=None):
        super(BuildCNNWizard, self).__init__()
        self.setWindowTitle("Train CNN")
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        if platform.system() == 'Linux':
            self.setWindowFlags(self.windowFlags() ^ Qt.WindowContextHelpButtonHint)
        else:
            self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOptions(QWizard.NoBackButtonOnStartPage)

        self.rocpages = []

        self.cnntrain = Training.CNNtrain(configdir, filtdir)

        # P1
        self.browsedataPage = BuildCNNWizard.WPageData(self.cnntrain, config)
        self.browsedataPage.registerField("trainDir1*", self.browsedataPage.trainDirName1)
        self.browsedataPage.registerField("trainDir2*", self.browsedataPage.trainDirName2)
        self.browsedataPage.registerField("filter*", self.browsedataPage.speciesCombo, "currentText", self.browsedataPage.speciesCombo.currentTextChanged)
        self.addPage(self.browsedataPage)

        # P2
        self.confirminputPage = BuildCNNWizard.WPageConfirminput(self.cnntrain, configdir)
        self.addPage(self.confirminputPage)

        # P3
        self.parameterPage = BuildCNNWizard.WPageParameters(self.cnntrain, config)
        self.addPage(self.parameterPage)

        # add the Save & Test button
        self.saveTestBtn = QPushButton("Save and Test")
        self.setButton(QWizard.CustomButton1, self.saveTestBtn)
        self.setButtonLayout( [QWizard.Stretch, QWizard.BackButton, QWizard.NextButton, QWizard.CustomButton1, QWizard.FinishButton, QWizard.CancelButton])
        self.setOptions(QWizard.NoBackButtonOnStartPage | QWizard.HaveCustomButton1)
        self.saveTestBtn.setVisible(False)

    # page 1 - select train data
    class WPageData(QWizardPage):
        def __init__(self, cnntrain, config, parent=None):
            super(BuildCNNWizard.WPageData, self).__init__(parent)
            self.setTitle('Select data')
            self.setSubTitle('Choose the recogniser that you want to extend with CNN, then select training data.')

            self.setMinimumSize(250, 200)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.cnntrain = cnntrain

            self.splist1 = []
            self.splist2 = []
            self.anntlevel = "Some"

            self.trainDirName1 = QLineEdit()
            self.trainDirName1.setReadOnly(True)
            self.btnBrowseTrain1 = QPushButton('Browse')
            self.btnBrowseTrain1.clicked.connect(self.browseTrainData1)
            self.trainDirName2 = QLineEdit()
            self.trainDirName2.setReadOnly(True)
            self.btnBrowseTrain2 = QPushButton('Browse')
            self.btnBrowseTrain2.clicked.connect(self.browseTrainData2)

            colourNone = QColor(config['ColourNone'][0], config['ColourNone'][1], config['ColourNone'][2], config['ColourNone'][3])
            colourPossibleDark = QColor(config['ColourPossible'][0], config['ColourPossible'][1], config['ColourPossible'][2], 255)
            colourNamed = QColor(config['ColourNamed'][0], config['ColourNamed'][1], config['ColourNamed'][2], config['ColourNamed'][3])
            self.listFilesTrain2 = SupportClasses_GUI.LightedFileList(colourNone, colourPossibleDark, colourNamed)
            self.listFilesTrain2.setMinimumWidth(350)
            self.listFilesTrain2.setMinimumHeight(275)
            self.listFilesTrain2.setSelectionMode(QAbstractItemView.NoSelection)
            self.listFilesTrain1 = SupportClasses_GUI.LightedFileList(colourNone, colourPossibleDark, colourNamed)
            self.listFilesTrain1.setMinimumWidth(350)
            self.listFilesTrain1.setMinimumHeight(275)
            self.listFilesTrain1.setSelectionMode(QAbstractItemView.NoSelection)
            self.listFilesTest = SupportClasses_GUI.LightedFileList(colourNone, colourPossibleDark, colourNamed)
            self.listFilesTest.setMinimumWidth(150)
            self.listFilesTest.setMinimumHeight(275)
            self.listFilesTest.setSelectionMode(QAbstractItemView.NoSelection)

            self.speciesCombo = QComboBox()  # fill during browse
            self.speciesCombo.addItems(['Choose recogniser...'])

            self.rbtn1 = QRadioButton('Annotated some calls')
            self.rbtn1.setChecked(True)
            self.rbtn1.annt = "Some"
            self.rbtn1.toggled.connect(self.onClicked)
            self.rbtn2 = QRadioButton('Annotated all calls')
            self.rbtn2.annt = "All"
            self.rbtn2.toggled.connect(self.onClicked)

            space = QLabel()
            space.setFixedHeight(10)
            space.setFixedWidth(40)

            # page layout
            layout = QGridLayout()
            layout.addWidget(QLabel('<b>Recogniser</b>'), 0, 0)
            layout.addWidget(QLabel("Recogniser that you want to train CNN for"), 1, 0)
            layout.addWidget(self.speciesCombo, 1, 1)
            layout.addWidget(space, 2, 0)
            layout.addWidget(QLabel('<b>TRAINING data</b>'), 3, 0)
            layout.addWidget(QLabel('<i>Manually annotated</i>'), 4, 0)
            layout.addWidget(self.btnBrowseTrain1, 5, 0)
            layout.addWidget(self.trainDirName1, 6, 0)
            layout.addWidget(self.listFilesTrain1, 7, 0)
            layout.addWidget(QLabel('<i>Auto processed & Batch reviewed</i>'), 4, 1)
            layout.addWidget(self.btnBrowseTrain2, 5, 1)
            layout.addWidget(self.trainDirName2, 6, 1)
            layout.addWidget(self.listFilesTrain2, 7, 1)
            layout.addWidget(QLabel('How is your manual annotation?'), 8, 0)
            layout.addWidget(self.rbtn1, 9, 0)
            layout.addWidget(self.rbtn2, 10, 0)
            layout.addWidget(space, 3, 2)
            self.setLayout(layout)

        def initializePage(self):
            filternames = [key + ".txt" for key in self.cnntrain.FilterDict.keys()]
            self.speciesCombo.addItems(filternames)

        def browseTrainData2(self):
            dirName = QFileDialog.getExistingDirectory(self, 'Choose folder with auto-processed and reviewed training data')
            self.trainDirName2.setText(dirName)

            self.listFilesTrain2.fill(dirName, fileName=None, readFmt=False, addWavNum=True, recursive=True)
            # while reading the file, we also collected a list of species present there
            self.splist2 = list(self.listFilesTrain2.spList)
            self.completeChanged.emit()

        def browseTrainData1(self):
            dirName = QFileDialog.getExistingDirectory(self, 'Choose folder with manually annotated training data')
            self.trainDirName1.setText(dirName)

            self.listFilesTrain1.fill(dirName, fileName=None, readFmt=False, addWavNum=True, recursive=True)
            # while reading the file, we also collected a list of species present there
            self.splist1 = list(self.listFilesTrain1.spList)
            self.completeChanged.emit()

        def onClicked(self):
            radioBtn = self.sender()
            if radioBtn.isChecked():
                self.anntlevel = radioBtn.annt
            self.completeChanged.emit()

        def isComplete(self):
            if self.speciesCombo.currentText() != "Choose recogniser..." and (self.trainDirName1.text() or self.trainDirName2.text()):
                self.cnntrain.setP1(self.trainDirName1.text(),self.trainDirName2.text(),self.speciesCombo.currentText(),self.rbtn2.isChecked())
                return True
            else:
                return False

    # page 2 - data confirm page
    class WPageConfirminput(QWizardPage):
        def __init__(self, cnntrain, configdir, parent=None):
            super(BuildCNNWizard.WPageConfirminput, self).__init__(parent)
            self.setTitle('Confirm data input')
            self.setSubTitle('When ready, press \"Next\" to start preparing images and train the CNN.')
            self.setMinimumSize(350, 275)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.cnntrain = cnntrain
            self.certainty1 = True
            self.certainty2 = True
            self.hasant1 = False
            self.hasant2 = False
            cl = SupportClasses.ConfigLoader()
            self.LearningDict = cl.learningParams(os.path.join(configdir, "LearningParams.txt"))

            self.msgmdir = QLabel("")
            self.msgmdir.setFixedWidth(600)
            self.msgmdir.setWordWrap(True)
            self.msgmdir.setStyleSheet("QLabel { color : #808080; }")
            self.warnnoannt1 = QLabel("")
            self.warnnoannt1.setStyleSheet("QLabel { color : #800000; }")
            self.msgadir = QLabel("")
            self.msgadir.setFixedWidth(600)
            self.msgadir.setWordWrap(True)
            self.msgadir.setStyleSheet("QLabel { color : #808080; }")
            self.warnnoannt2 = QLabel("")
            self.warnnoannt2.setStyleSheet("QLabel { color : #800000; }")

            self.msgrecfilter = QLabel("")
            self.msgrecfilter.setStyleSheet("QLabel { color : #808080; }")
            self.msgrecspp = QLabel("")
            self.msgrecspp.setStyleSheet("QLabel { color : #808080; }")
            self.msgreccts = QLabel("")
            self.msgreccts.setStyleSheet("QLabel { color : #808080; }")
            self.msgrecclens = QLabel("")
            self.msgrecclens.setStyleSheet("QLabel { color : #808080; }")
            self.msgrecfs = QLabel("")
            self.msgrecfs.setStyleSheet("QLabel { color : #808080; }")
            self.warnLabel = QLabel("")
            self.warnLabel.setStyleSheet("QLabel { color : #800000; }")
            self.warnoise = QLabel("")
            self.warnoise.setStyleSheet("QLabel { color : #800000; }")
            self.msgseg = QLabel("")
            self.msgseg.setFixedWidth(600)
            self.msgseg.setWordWrap(True)
            self.msgseg.setStyleSheet("QLabel { color : #808080; }")
            lblmsgseg = QLabel("<b>Segments detected<b>")
            lblmsgseg.setStyleSheet("QLabel { color : #808080; }")
            self.warnseg = QLabel("")
            self.warnseg.setStyleSheet("QLabel { color : #800000; }")
            space = QLabel()
            space.setFixedHeight(20)
            space.setFixedWidth(10)

            # page layout
            layout = QGridLayout()
            layout.addWidget(QLabel("<b>Selected TRAINING data</b>"), 0, 0)
            layout.addWidget(self.msgadir, 0, 2)
            layout.addWidget(self.warnnoannt2, 1, 2)
            layout.addWidget(self.msgmdir, 2, 2)
            layout.addWidget(self.warnnoannt1, 3, 2)
            layout.addWidget(space, 4, 0)
            layout.addWidget(self.warnoise, 5, 2)
            layout.addWidget(lblmsgseg, 6, 2)
            layout.addWidget(self.msgseg, 7, 2)
            layout.addWidget(self.warnseg, 8, 2)
            layout.addWidget(space, 9, 0)
            layout.addWidget(space, 12, 0)
            layout.addWidget(QLabel("<b>Selected Recogniser</b>"), 13, 0)
            layout.addWidget(self.msgrecfilter, 13, 2)
            layout.addWidget(self.msgrecspp, 14, 2)
            layout.addWidget(self.msgreccts, 15, 2)
            layout.addWidget(self.msgrecclens, 16, 2)
            layout.addWidget(self.msgrecfs, 17, 2)
            layout.addWidget(self.warnLabel, 18, 2)
            self.setLayout(layout)

        def initializePage(self):
            self.certainty1 = True
            self.certainty2 = True

            with pg.BusyCursor():
                self.cnntrain.readFilter()

            # Error checking

            # Check if it already got a CNN model
            if "CNN" in self.cnntrain.currfilt:
                self.warnLabel.setText("Warning: This recogniser already has a CNN.")
            else:
                self.warnLabel.setText("")

            warn = ""
            # Check the annotation certainty
            if self.field("trainDir1"):
                self.certainty1 = self.getCertainty(self.field("trainDir1"))
                if not self.certainty1:
                    warn += "Warning: Detected uncertain segments\n"

            # Check if there are annotations from the target species at all
            if self.field("trainDir1"):
                if self.cnntrain.species not in self.wizard().browsedataPage.splist1:
                    warn += "Warning: No annotations of " + self.cnntrain.species + " detected\n"
                    self.hasant1 = False
                else:
                    self.hasant1 = True

            self.warnnoannt1.setText(warn)

            warn = ""
            # Check the annotation certainty
            if self.field("trainDir2"):
                self.certainty2 = self.getCertainty(self.field("trainDir2"))
                if not self.certainty2:
                    warn += "Warning: Detected uncertain segments\n"

            # Check if there are annotations from the target species at all
            if self.field("trainDir2"):
                if self.cnntrain.species not in self.wizard().browsedataPage.splist2:
                    warn += "Warning: No annotations of " + self.cnntrain.species + " detected\n"
                    self.hasant2 = False
                else:
                    self.hasant2 = True

            self.warnnoannt2.setText(warn)

            if self.field("trainDir1"):
                self.msgmdir.setText("<b>Manually annotated:</b> %s" % (self.field("trainDir1")))
            if self.field("trainDir2"):
                self.msgadir.setText("\n<b>Auto processed and reviewed:</b> %s" % (self.field("trainDir2")))

            # Get training data 
            with pg.BusyCursor():
                self.cnntrain.genSegmentDataset(self.hasant1)

            self.msgrecfilter.setText("<b>Recogniser:</b> %s" % (self.field("filter")))
            self.msgrecspp.setText("<b>Species:</b> %s" % (self.cnntrain.species))
            self.msgreccts.setText("<b>Call types:</b> %s" % (self.cnntrain.calltypes))
            self.msgrecclens.setText("<b>Call length:</b> %.2f - %.2f sec" % (self.cnntrain.mincallength, self.cnntrain.maxcallength))
            self.msgrecfs.setText("<b>Sample rate:</b> %d Hz" % (self.cnntrain.fs))

            for i in range(len(self.cnntrain.calltypes)):
                self.msgseg.setText("%s:\t%d\t" % (self.msgseg.text() + self.cnntrain.calltypes[i], self.cnntrain.trainN[i]))
            self.msgseg.setText("%s:\t%d" % (self.msgseg.text() + "Noise", self.cnntrain.trainN[-1]))

            # We need at least some number of segments from each class to proceed
            if min(self.cnntrain.trainN) < self.LearningDict['minPerClass']:    
                print('Warning: Need at least %d segments from each class to train CNN' % self.LearningDict['minPerClass'])
                self.warnseg.setText('<b>Warning: Need at least %d segments from each class to train CNN\n\n</b>' % self.LearningDict['minPerClass'])

            if not self.cnntrain.correction and self.wizard().browsedataPage.anntlevel == 'Some':
                self.warnoise.setText('Warning: No segments found for Noise class\n(no correction segments/fully (manual) annotations)')
            
            freeGB,totalbytes = self.cnntrain.checkDisk()

            if freeGB < 10:
                self.imgDirwarn.setText('Warning: Free space in the user directory is %.2f GB/ %.2f GB, you may run out of space' % (freeGB, totalbytes))

        def getCertainty(self, dirname):
            minCertainty = 100
            for root, dirs, files in os.walk(dirname):
                for file in files:
                    wavFile = os.path.join(root, file)
                    if file.lower().endswith('.wav') and os.stat(wavFile).st_size != 0 and file + '.data' in files:
                        segments = Segment.SegmentList()
                        segments.parseJSON(wavFile + '.data')
                        cert = [lab["certainty"] if lab["species"] == self.cnntrain.species else 100 for seg in segments for lab in seg[4]]
                        if cert:
                            mincert = min(cert)
                            if minCertainty > mincert:
                                minCertainty = mincert
            if minCertainty < 100:
                return False
            else:
                return True

        def cleanupPage(self):
            pass
            self.msgmdir.setText('')
            self.msgadir.setText('')
            self.warnnoannt1.setText('')
            self.warnLabel.setText('')
            self.warnnoannt2.setText('')
            self.warnoise.setText('')
            self.msgseg.setText('')
            self.warnseg.setText('')
            self.msgrecfilter.setText('')
            self.msgrecspp.setText('')
            self.msgreccts.setText('')
            self.msgrecclens.setText('')
            self.msgrecfs.setText('')

        def isComplete(self):
            if (self.hasant1 or self.hasant2) and min(self.cnntrain.trainN) >= self.LearningDict['minPerClass']:
                return True
            else:
                return False

    # page 3 - set parameters, generate data and train
    class WPageParameters(QWizardPage):
        def __init__(self, cnntrain, config,parent=None):
            super(BuildCNNWizard.WPageParameters, self).__init__(parent)
            self.setTitle('Choose call length')
            self.setSubTitle('When ready, press \"Generate CNN images and Train\" to start preparing data for CNN and training.\nThe process may take a long time.')

            self.setMinimumSize(350, 200)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.cnntrain = cnntrain
            self.config = config
            self.indx = np.ndarray(0)

            # Make pages to plot ROC for each call type OR automatically select thresholds
            self.redopages = True

            # Parameter/s
            self.imgsec = QSlider(Qt.Horizontal)
            self.imgsec.setTickPosition(QSlider.TicksBelow)
            self.imgsec.setTickInterval(25)
            self.imgsec.setRange(0, 600)  # 0-6 sec
            self.imgsec.setValue(25)
            self.imgsec.valueChanged.connect(self.imglenChange)
            self.imgtext = QLabel('0.25 sec')

            space = QLabel()
            space.setFixedSize(10, 30)
            msglayout = QVBoxLayout()
            self.msgspp = QLabel('')
            self.msgspp.setStyleSheet("QLabel { color : #808080; }")
            self.msgtrain1 = QLabel('')
            self.msgtrain1.setStyleSheet("QLabel { color : #808080; }")
            self.msgtrain2 = QLabel('')
            self.msgtrain2.setStyleSheet("QLabel { color : #808080; }")
            self.msgtest1 = QLabel('')
            self.msgtest1.setStyleSheet("QLabel { color : #808080; }")
            msglayout.addWidget(self.msgspp)
            msglayout.addWidget(self.msgtrain2)
            msglayout.addWidget(self.msgtrain1)
            msglayout.addWidget(self.msgtest1)

            layout0 = QVBoxLayout()
            layout0.addLayout(msglayout)
            layout0.addWidget(space)
            layout0.addWidget(QLabel('<b>Choose call length (sec) you want to show to CNN</b>'))
            layout0.addWidget(QLabel('Make sure an image covers at least couple of syllables when appropriate'))
            layout0.addWidget(space)
            layout0.addWidget(self.imgtext)
            layout0.addWidget(self.imgsec)

            layout2 = QVBoxLayout()
            layout2.addWidget(QLabel('<i>Example images from your dataset</i>'))
            self.flowLayout = QHBoxLayout()
            self.img1 = QLabel()
            self.img1.setFixedHeight(175)
            self.img2 = QLabel()
            self.img2.setFixedHeight(175)
            self.img3 = QLabel()
            self.img3.setFixedHeight(175)
            self.flowLayout.addWidget(self.img1)
            self.flowLayout.addWidget(self.img2)
            self.flowLayout.addWidget(self.img3)
            layout2.addLayout(self.flowLayout)

            self.imgDirwarn = QLabel('')
            self.imgDirwarn.setStyleSheet("QLabel { color : #800000; }")

            self.cbAutoThr = QCheckBox("Tick if you want AviaNZ to decide threshold/s")
            self.cbAutoThr.setStyleSheet("QCheckBox { font-weight: bold; }")
            self.cbAutoThr.toggled.connect(self.onClicked)

            layout1 = QVBoxLayout()
            layout1.addLayout(layout0)
            layout1.addLayout(layout2)
            layout1.addWidget(self.imgDirwarn)
            layout1.addWidget(self.cbAutoThr)
            self.setLayout(layout1)
            self.setButtonText(QWizard.NextButton, 'Generate CNN images and Train>')

        def initializePage(self):
            self.wizard().button(QWizard.NextButton).setDefault(False)
            self.msgspp.setText("<b>Species:</b> %s" % (self.cnntrain.species))

            if self.field("trainDir1"):
                self.msgtrain1.setText("<b>Training data (Manually annotated):</b> %s" % (self.field("trainDir1")))
            if self.field("trainDir2"):
                self.msgtrain2.setText("<b>Training data (Auto processed and reviewed):</b> %s" % (self.field("trainDir2")))

            # Ideally, the image length should be bigger than the max gap between syllables
            if np.max(self.cnntrain.maxgaps) * 2 <= 6:
                self.imgtext.setText(str(np.max(self.cnntrain.maxgaps) * 2) + ' sec')
                self.imgsec.setValue(np.max(self.cnntrain.maxgaps) * 2 * 100)
            elif np.max(self.cnntrain.maxgaps) * 1.5 <= 6:
                self.imgtext.setText(str(np.max(self.cnntrain.maxgaps) * 1.5) + ' sec')
                self.imgsec.setValue(np.max(self.cnntrain.maxgaps) * 1.5 * 100)
            elif np.max(self.cnntrain.mincallength) <= 6:
                self.imgtext.setText(str(np.max(self.cnntrain.mincallength)) + ' sec')
                self.imgsec.setValue(np.max(self.cnntrain.mincallength) * 100)
            self.cnntrain.imgWidth = self.imgsec.value() / 100

            self.setWindowInc()
            self.showimg()
            self.completeChanged.emit()

        def onClicked(self):
            cbutton = self.sender()
            if cbutton.isChecked():
                self.cnntrain.autoThr = True
            else:
                self.cnntrain.autoThr = False
            self.redopages = True
            self.completeChanged.emit()

        def loadFile(self, filename, duration=0, offset=0, fs=0):
            """
            Read audio file.
            """
            if duration == 0:
                duration = None

            self.cnntrain.sp.readWav(filename, duration, offset)
            self.cnntrain.sp.resample(fs)

            return self.cnntrain.sp.data

        def showimg(self, indices=[]):
            ''' Show example spectrogram (random ct segments from train dataset)
            '''
            i = 0
            # SM
            #trainsegments = self.cnntrain.trainsegments
            if len(indices) == 0:
                target = [rec[-1] for rec in self.cnntrain.traindata]
                indxs = [list(np.where(np.array(target) == i)[0]) for i in range(len(self.cnntrain.calltypes))]
                indxs = [i for sublist in indxs for i in sublist]
                self.indx = np.random.choice(indxs, 3, replace=False)
            else:
                self.indx = indices
            for ind in self.indx:
                audiodata = self.loadFile(filename=self.cnntrain.traindata[ind][0], duration=self.imgsec.value()/100, offset=self.cnntrain.traindata[ind][1][0], fs=self.cnntrain.fs)
                self.cnntrain.sp.data = audiodata
                self.cnntrain.sp.sampleRate = self.cnntrain.fs
                sgRaw = self.cnntrain.sp.spectrogram(window_width=self.cnntrain.windowWidth, incr=self.cnntrain.windowInc)
                maxsg = np.min(sgRaw)
                self.sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))
                self.setColourMap()
                picbtn = SupportClasses_GUI.PicButton(1, np.fliplr(self.sg), self.cnntrain.sp.data, self.cnntrain.sp.audioFormat, self.imgsec.value(), 0, 0, self.lut, self.colourStart, self.colourEnd, False,
                                                  cluster=True)
                if i == 0:
                    pic = QPixmap.fromImage(picbtn.im1)
                    self.img1.setPixmap(pic.scaledToHeight(175))
                    self.flowLayout.update()
                    i += 1
                elif i == 1:
                    pic = QPixmap.fromImage(picbtn.im1)
                    self.img2.setPixmap(pic.scaledToHeight(175))
                    self.flowLayout.update()
                    i += 1
                elif i == 2:
                    pic = QPixmap.fromImage(picbtn.im1)
                    self.img3.setPixmap(pic.scaledToHeight(175))
                    self.flowLayout.update()
                    i += 1
                else:
                    break
            if i == 0:
                self.img1.setText('<no image to show>')
                self.img2.setText('')
                self.img3.setText('')
                self.flowLayout.update()

        def setColourMap(self):
            """ Listener for the menu item that chooses a colour map.
            Loads them from the file as appropriate and sets the lookup table.
            """
            cmap = self.config['cmap']

            pos, colour, mode = colourMaps.colourMaps(cmap)

            cmap = pg.ColorMap(pos, colour, mode)
            self.lut = cmap.getLookupTable(0.0, 1.0, 256)
            minsg = np.min(self.sg)
            maxsg = np.max(self.sg)
            self.colourStart = (self.config['brightness'] / 100.0 * self.config['contrast'] / 100.0) * (
                        maxsg - minsg) + minsg
            self.colourEnd = (maxsg - minsg) * (1.0 - self.config['contrast'] / 100.0) + self.colourStart

        def setWindowInc(self):
            self.cnntrain.windowWidth = self.cnntrain.imgsize[0] * 2
            self.cnntrain.windowInc = int(np.ceil(self.imgsec.value() * self.cnntrain.fs / (self.cnntrain.imgsize[1] - 1)) / 100)
            print('window and increment set: ', self.cnntrain.windowWidth, self.cnntrain.windowInc)

        def imglenChange(self, value):
            if value < 10:
                self.imgsec.setValue(10)
                self.imgtext.setText('0.1 sec')
            else:
                self.imgtext.setText(str(value / 100) + ' sec')
            self.cnntrain.imgWidth = self.imgsec.value()/100
            self.setWindowInc()
            self.showimg(self.indx)

        def cleanupPage(self):
            self.imgDirwarn.setText('')
            self.img1.setText('')
            self.img2.setText('')
            self.img3.setText('')

        def validatePage(self):
            with pg.BusyCursor():
                self.cnntrain.train()
            return True

        def isComplete(self):
            if self.img1.text() == '<no image to show>':
                return False
            if self.redopages:
                self.redopages = False
                self.wizard().redoROCPages()
            return True

    # page 4 - ROC curve
    class WPageROC(QWizardPage):
        def __init__(self, cnntrain, ct, parent=None):
            super(BuildCNNWizard.WPageROC, self).__init__(parent)
            self.setTitle('Training results')
            self.setSubTitle('Click on the graph at the point where you would like the classifier to trade-off false positives with false negatives. Points closest to the top-left are best.')
            self.setMinimumSize(350, 200)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.cnntrain = cnntrain
            self.ct = ct

            self.lblSpecies = QLabel()
            self.lblSpecies.setStyleSheet("QLabel { color : #808080; }")
            self.lblCalltype = QLabel()
            self.lblCalltype.setStyleSheet("QLabel { color : #808080; }")
            self.lblUpdate = QLabel()

            self.layout = QGridLayout()
            self.layout.addWidget(self.lblSpecies, 0, 0)
            self.layout.addWidget(self.lblCalltype, 1, 0)
            self.setLayout(self.layout)

        def initializePage(self):
            self.thrs = self.cnntrain.Thrs
            self.TPR = self.cnntrain.TPRs[self.ct]
            self.FPR = self.cnntrain.FPRs[self.ct]
            self.Precision = self.cnntrain.Precisions[self.ct]
            self.Acc = self.cnntrain.Accs[self.ct]

            # This is the Canvas Widget that displays the plot
            self.figCanvas = ROCCanvas(self)
            self.figCanvas.plotme()

            self.marker = self.figCanvas.ax.plot([0, 1], [0, 1], marker='o', color='black', linestyle='dotted')[0]
            # self.marker.set_visible(False)
            self.figCanvas.plotmeagain(self.TPR, self.FPR, CNN=True)

            if self.ct == len(self.cnntrain.calltypes):
                self.lblCalltype.setText('Noise (treat same as call types)')
            else:
                self.lblCalltype.setText('Call type: ' + self.cnntrain.calltypes[self.ct])
            self.lblSpecies.setText('Species: ' + self.cnntrain.species)

            # Figure click handler
            def onclick(event):
                fpr_cl = event.xdata
                tpr_cl = event.ydata
                if tpr_cl is None or fpr_cl is None:
                    return

                # Get thr for closest point
                distarr = (tpr_cl - self.TPR) ** 2 + (fpr_cl - self.FPR) ** 2
                thr_min_ind = np.unravel_index(np.argmin(distarr), distarr.shape)[0]
                tpr_near = self.TPR[thr_min_ind]
                fpr_near = self.FPR[thr_min_ind]
                self.marker.set_visible(False)
                self.figCanvas.draw()
                self.marker.set_xdata([fpr_cl, fpr_near])
                self.marker.set_ydata([tpr_cl, tpr_near])
                self.marker.set_visible(True)
                self.figCanvas.ax.draw_artist(self.marker)
                self.figCanvas.update()

                print("fpr_cl, tpr_cl: ", fpr_near, tpr_near)

                # Update sidebar info
                self.lblUpdate.setText(
                    '\tTrue Positive Rate: %.2f\n\tFalse Positive Rate: %.2f\n\tPrecision: %.2f\n\tAccuracy: %.2f' % (
                        self.TPR[thr_min_ind], self.FPR[thr_min_ind], self.Precision[thr_min_ind], self.Acc[thr_min_ind]))

                # This will save the best lower thr
                self.cnntrain.bestThr[self.ct][0] = self.cnntrain.thrs[thr_min_ind]
                self.cnntrain.bestThrInd[self.ct] = thr_min_ind

                self.completeChanged.emit()

            self.figCanvas.figure.canvas.mpl_connect('button_press_event', onclick)

            self.layout.addWidget(self.figCanvas, 2, 0)
            self.layout.addWidget(self.lblUpdate, 2, 1)

        def cleanupPage(self):
            pass
            #try:
                #self.wizard().parameterPage.tmpdir1.cleanup()
                #self.wizard().parameterPage.tmpdir2.cleanup()
            #except:
                #pass

        def isComplete(self):
            if self.lblUpdate.text() == '':
                return False
            else:
                return True

    # page 5 - Summary
    class WPageSummary(QWizardPage):
        def __init__(self, cnntrain, parent=None):
            super(BuildCNNWizard.WPageSummary, self).__init__(parent)
            self.setTitle('Training Summary')
            self.setSubTitle('If you are happy with the CNN performance, press \"Save the Recogniser.\"')
            self.setMinimumSize(250, 150)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.cnntrain = cnntrain

            self.space = QLabel('').setFixedSize(20, 20)
            self.msgfilter = QLabel('')
            self.msgfilter.setStyleSheet("QLabel { color : #808080; }")
            self.msgspp = QLabel('')
            self.msgspp.setStyleSheet("QLabel { color : #808080; }")

            # page layout
            self.layout = QGridLayout()
            self.layout.addWidget(self.msgfilter, 0, 0)
            self.layout.addWidget(self.msgspp, 1, 0)
            self.layout.addWidget(self.space, 2, 0)

            self.setButtonText(QWizard.NextButton, 'Save the Recogniser>')
            self.setLayout(self.layout)

        def initializePage(self):
            self.msgfilter.setText("<b>Current recogniser:</b> %s" % (self.field("filter")))
            self.msgspp.setText("<b>Species:</b> %s" % (self.cnntrain.species))

            row = 3
            for ct in range(len(self.cnntrain.calltypes)):
                lblct = QLabel('Call type: ' + self.cnntrain.calltypes[ct])
                lblct.setStyleSheet("QLabel { color : #808080; font-weight: bold; }")
                self.layout.addWidget(lblct, row, 0, alignment=Qt.AlignTop)
                lblctsumy = QLabel('True Positive Rate: %.2f\nFalse Positive Rate: %.2f\nPrecision: %.2f\nAccuracy: %.2f'
                                   % (self.cnntrain.TPRs[ct][self.cnntrain.bestThrInd[ct]],
                                      self.cnntrain.FPRs[ct][self.cnntrain.bestThrInd[ct]],
                                      self.cnntrain.Precisions[ct][self.cnntrain.bestThrInd[ct]],
                                      self.cnntrain.Accs[ct][self.cnntrain.bestThrInd[ct]]))
                lblctsumy.setStyleSheet("QLabel { color : #808080; }")
                self.layout.addWidget(self.space, row, 1)
                self.layout.addWidget(lblctsumy, row, 2)
                row += 1
            self.layout.update()

        def cleanupPage(self):
            wgts = []
            for ct in range(len(self.cnntrain.calltypes) ):
                if self.layout.itemAtPosition(ct+3, 0):
                    wgts.append(self.layout.itemAtPosition(ct+3, 0).widget())
                if self.layout.itemAtPosition(ct+3, 1):
                    wgts.append(self.layout.itemAtPosition(ct+3, 1).widget())
                if self.layout.itemAtPosition(ct+3, 2):
                    wgts.append(self.layout.itemAtPosition(ct+3, 2).widget())

            for i in reversed(range(len(wgts))):
                self.layout.removeWidget(wgts[i])
                wgts[i].deleteLater()
                del wgts[i]

    # page 6 - Save Filter
    class WPageSave(QWizardPage):
        def __init__(self, cnntrain, parent=None):
            super(BuildCNNWizard.WPageSave, self).__init__(parent)
            self.setTitle('Save Recogniser')
            self.setSubTitle('If you are happy with the CNN performance, save the recogniser.')
            self.setMinimumSize(250, 150)
            self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
            self.adjustSize()

            self.cnntrain = cnntrain
            self.filterfile = ''
            self.saveoption = 'New'

            # filter dir listbox
            self.listFiles = QListWidget()
            self.listFiles.setSelectionMode(QAbstractItemView.NoSelection)
            self.listFiles.setMinimumWidth(150)
            self.listFiles.setMinimumHeight(200)
            filtdir = QDir(self.cnntrain.filterdir).entryList(filters=QDir.NoDotAndDotDot | QDir.Files)
            for file in filtdir:
                item = QListWidgetItem(self.listFiles)
                item.setText(file)
            # filter file name
            self.enterFiltName = QLineEdit()
            self.enterFiltName.textChanged.connect(self.textChanged)

            class FiltValidator(QValidator):
                def validate(self, input, pos):
                    if not input.endswith('.txt'):
                        input = input + '.txt'
                    if input == "M.txt":
                        print("filter name \"M\" reserved for manual annotations")
                        return (QValidator.Intermediate, input, pos)
                    elif self.listFiles.findItems(input, Qt.MatchExactly):
                        print("duplicated input", input)
                        return (QValidator.Intermediate, input, pos)
                    else:
                        return (QValidator.Acceptable, input, pos)

            trainFiltValid = FiltValidator()
            trainFiltValid.listFiles = self.listFiles
            self.enterFiltName.setValidator(trainFiltValid)
            space = QLabel('').setFixedSize(30, 50)

            self.msgfilter = QLabel('')
            self.msgfilter.setStyleSheet("QLabel { color : #808080; }")
            self.warnfilter = QLabel('')
            self.warnfilter.setStyleSheet("QLabel { color : #800000; }")
            self.msgspp = QLabel('')
            self.msgspp.setStyleSheet("QLabel { color : #808080; }")

            self.rbtn1 = QRadioButton('New recogniser (enter name below)')
            self.rbtn1.setChecked(True)
            self.rbtn1.val = "New"
            self.rbtn1.toggled.connect(self.onClicked)
            self.rbtn2 = QRadioButton('Update existing')
            self.rbtn2.val = "Update"
            self.rbtn2.toggled.connect(self.onClicked)

            # page layout
            layout = QGridLayout()
            layout.addWidget(self.msgfilter, 0, 0)
            layout.addWidget(self.warnfilter, 0, 1)
            layout.addWidget(self.msgspp, 1, 0)
            layout.addWidget(space, 2, 0)
            layout.addWidget(QLabel('<b>How do you want to save it?</b>'), 3, 0)
            layout.addWidget(self.rbtn1, 4, 1)
            layout.addWidget(self.rbtn2, 5, 1)
            layout.addWidget(space, 6, 0)
            layout.addWidget(QLabel("<i>Currently available recognisers</i>"), 7, 0)
            layout.addWidget(self.listFiles, 8, 0, 1, 2)
            layout.addWidget(space, 9, 0)
            layout.addWidget(QLabel("Enter file name if you choose to save the recogniser as a new file (must be unique)"), 10, 0, 1, 2)
            layout.addWidget(self.enterFiltName, 12, 0, 1, 2)

            self.setButtonText(QWizard.FinishButton, 'Save and Finish')
            self.setLayout(layout)

        def initializePage(self):
            self.msgfilter.setText("<b>Current recogniser:</b> %s" % (self.field("filter")))
            if "CNN" in self.cnntrain.currfilt:
                self.warnfilter.setText("Warning: The recogniser already has a CNN.")
            self.msgspp.setText("<b>Species:</b> %s" % (self.cnntrain.species))
            self.rbtn2.setText('Update existing (' + self.field("filter") + ')')

            self.cnntrain.addCNNFilter()

            self.wizard().saveTestBtn.setVisible(True)
            self.wizard().saveTestBtn.setEnabled(False)
            self.completeChanged.emit()

        def refreshCustomBtn(self):
            if self.isComplete():
                self.wizard().saveTestBtn.setEnabled(True)
            else:
                self.wizard().saveTestBtn.setEnabled(False)
            self.completeChanged.emit()

        def onClicked(self):
            radioBtn = self.sender()
            if radioBtn.isChecked():
                self.saveoption = radioBtn.val
            self.refreshCustomBtn()
            self.completeChanged.emit()

        def textChanged(self, text):
            self.refreshCustomBtn()
            self.completeChanged.emit()

        # def validatePage(self):
        #     with pg.BusyCursor():
        #         self.cnntrain.saveFilter()
        #     return True

        def cleanupPage(self):
            self.wizard().saveTestBtn.setEnabled(False)
            self.enterFiltName.setText('')
            self.rbtn1.setChecked(True)
            self.saveoption = "New"
            self.wizard().saveTestBtn.setVisible(False)

        def isComplete(self):
            if self.saveoption == 'New' and self.enterFiltName.text() != '' and self.enterFiltName.text() != '.txt':
                self.cnntrain.setP6(self.enterFiltName.text())
                return True
            elif self.saveoption == "Update":
                # SM
                self.cnntrain.setP6(self.enterFiltName.text())
                return True
            else:
                return False

    def redoROCPages(self):
        # clean any existing pages
        for page in self.rocpages:
            # for each calltype, remove roc page
            self.removePage(page)
        self.rocpages = []

        if not self.cnntrain.autoThr:
            for i in range(len(self.cnntrain.calltypes)):
                print("adding ROC page for class:", self.cnntrain.calltypes[i])
                page4 = BuildCNNWizard.WPageROC(self.cnntrain,i)
                pageid = self.addPage(page4)
                self.rocpages.append(pageid)

        self.summaryPage = BuildCNNWizard.WPageSummary(self.cnntrain)
        pageid = self.addPage(self.summaryPage)
        self.rocpages.append(pageid)

        self.savePage = BuildCNNWizard.WPageSave(self.cnntrain)
        pageid = self.addPage(self.savePage)
        self.rocpages.append(pageid)

        self.parameterPage.setFinalPage(False)
        self.parameterPage.completeChanged.emit()

    def undoROCPages(self):
        # clean any existing pages
        for page in self.rocpages:
            # for each calltype, remove roc page
            self.removePage(page)
        self.rocpages = []

        self.summaryPage = BuildCNNWizard.WPageSummary(self.cnntrain)
        self.addPage(self.summaryPage)

        self.savePage = BuildCNNWizard.WPageSave(self.cnntrain)
        self.addPage(self.savePage)

        self.parameterPage.setFinalPage(False)
        self.parameterPage.completeChanged.emit()

