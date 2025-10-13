
# Version 4.1 09/10/25
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

import os
import time
import platform
import copy
from shutil import copyfile

from PyQt6.QtGui import QIcon, QValidator, QPixmap, QColor
from PyQt6.QtCore import QDir, Qt, QEvent, QSize

from src.ui.components.validators import FiltValidator
from PyQt6.QtWidgets import QLabel, QSlider, QPushButton, QListWidget, QListWidgetItem, QComboBox, QWizard, QWizardPage, QLineEdit, QSizePolicy, QFormLayout, QVBoxLayout, QHBoxLayout, QCheckBox, QLayout, QApplication, QFileDialog, QScrollArea, QAbstractItemView

import pyqtgraph as pg

import numpy as np
from src.ui.colourMaps import colourMaps
from src.ui.components.buttons_and_controls import BrightContrVol, PicButton
from src.ui.components.popups import MessagePopup
from src.ui.components.file_list import LightedFileList
from src.ui.components.layout_widgets import Layout
from src.ui.dialogs_training.roc_canvas import ROCCanvas
from src.core import spectrogram
from src.core import wavelet_segment
from src.core import wavelet_functions
from src.core import annotation
from src.core import clustering
from src.core import audio_data

import math


class BuildRecAdvWizard(QWizard):
    # page 1 - select training data
    class WPageData(QWizardPage):
        def __init__(self, config, parent=None):
            super(BuildRecAdvWizard.WPageData, self).__init__(parent)
            self.setTitle('Training data')
            self.setSubTitle('To start training, you need labelled calls from your species as training data (see the manual). Select the folder where this data is located. Then select the species.')

            self.setMinimumSize(600, 150)
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            self.adjustSize()

            self.trainDirName = QLineEdit()
            self.trainDirName.setReadOnly(True)
            self.btnBrowse = QPushButton('Browse')
            self.btnBrowse.clicked.connect(self.browseTrainData)

            colourNone = QColor(config['ColourNone'][0], config['ColourNone'][1], config['ColourNone'][2], config['ColourNone'][3])
            colourPossibleDark = QColor(config['ColourPossible'][0], config['ColourPossible'][1], config['ColourPossible'][2], 255)
            colourNamed = QColor(config['ColourNamed'][0], config['ColourNamed'][1], config['ColourNamed'][2], config['ColourNamed'][3])
            self.listFiles = LightedFileList(colourNone, colourPossibleDark, colourNamed)
            self.listFiles.setMinimumHeight(225)
            self.listFiles.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

            selectSpLabel = QLabel("Choose the species for which you want to build the recogniser")
            self.species = QComboBox()  # fill during browse
            self.species.addItems(['Choose species...'])

            space = QLabel()
            space.setFixedHeight(20)

            # SampleRate parameter
            self.fs = QSlider(Qt.Orientation.Horizontal)
            self.fs.setTickPosition(QSlider.TickPosition.TicksBelow)
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
            layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
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
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
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
            self.setButtonText(QWizard.WizardButton.NextButton, 'Cluster >')

        def initializePage(self):
            self.wizard().button(QWizard.WizardButton.NextButton).setDefault(False)
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
            self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
            self.adjustSize()

            # TODO: SRM: resample
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

            # Volume, brightness and contrast sliders.
            # Config values are overwritten with fixed bright/contr/no inversion.
            self.specControls = BrightContrVol(80, 20, False)
            self.specControls.colChanged.connect(self.setColourLevels)
            self.specControls.volChanged.connect(self.volSliderMoved)
            self.specControls.layout().setContentsMargins(20, 0, 20, 10)

            self.btnCreateNewCluster = QPushButton('Create cluster')
            self.btnCreateNewCluster.setFixedWidth(150)
            self.btnCreateNewCluster.clicked.connect(self.createNewcluster)
            self.btnDeleteSeg = QPushButton('Remove selected segment/s')
            self.btnDeleteSeg.setFixedWidth(200)
            self.btnDeleteSeg.clicked.connect(self.deleteSelectedSegs)

            # Colour map
            self.lut = colourMaps.getLookupTable(self.config['cmap'])

            # page 3 layout
            layout1 = QVBoxLayout()
            layout1.addWidget(instr)
            layout1.addWidget(self.lblSpecies)

            hboxBtns2 = QHBoxLayout()
            hboxBtns2.addWidget(self.btnCreateNewCluster)
            hboxBtns2.addWidget(self.btnDeleteSeg)
            hboxBtns2.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            # top part
            vboxTop = QVBoxLayout()
            vboxTop.addWidget(self.specControls)
            vboxTop.addLayout(hboxBtns2)

            # set up the images
            self.flowLayout = Layout()
            self.flowLayout.setMinimumSize(380, 247)
            self.flowLayout.buttonDragged.connect(self.moveSelectedSegs)
            self.flowLayout.layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

            self.scrollArea = QScrollArea(self)
            #self.scrollArea.setWidgetResizable(True)
            self.scrollArea.setWidget(self.flowLayout)
            self.scrollArea.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)

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
            self.wizard().speciesData = {"species": self.field("species"), "method": self.wizard().method, "SampleRate": fs, "Filters": []}

            with pg.BusyCursor():
                print("Processing. Please wait...")
                # Check if the annotations come with call type labels, if so skip auto clustering
                self.hasCTannotations = True if len(self.CTannotations())>0 else False
                if self.hasCTannotations:
                    # self.segments: [parent_audio_file, [segment], class_label]
                    self.segments, self.nclasses, self.duration = self.getClustersGT()
                    self.setSubTitle('AviaNZ found call type annotations in your dataset. You can still make corrections by moving calls as appropriate.')
                else:
                    # return format:
                    # self.segments: [parent_audio_file, [segment], [syllables], [features], class_label]
                    # self.nclasses: number of class_labels
                    # duration: median length of segments
                    self.cluster = clustering.Clustering([], [], 5)
                    f1,f2 = self.cluster.getFrqRange(self.field("trainDir"),self.field("species"),self.field("fs"))
                    dataset = self.cluster.findSyllables(self.field("trainDir"),self.field("species"),0.2,self.field("fs"),f1,f2,False)
                    self.segments, self.nclasses, self.duration = self.cluster.cluster(dataset, self.field("trainDir"), self.field("fs"), self.field("species"), feature=self.feature)
                    # segments format: [[file1, seg1, [syl1, syl2], [features1, features2], predict], ...]
                    # self.segments, fs, self.nclasses, self.duration = self.cluster.cluster_by_dist(self.field("trainDir"),
                    #                                                                              self.field("species"),
                    #                                                                              feature=self.feature,
                    #                                                                              max_clusters=5,
                    #                                                                              single=True)
                    self.setSubTitle('AviaNZ has tried to identify similar calls in your dataset. Please check the output, and move calls as appropriate.')

                # Start of better code from here at bottom of file

                # Create and show the buttons
                # TODO: !
                #self.nclasses=1

                self.clearButtons()
                self.addButtons()
                #self.addButtons(callsgs,audios,callIDs,sp)
                self.updateButtons()
                self.segsChanged = True
                self.completeChanged.emit()

        def isComplete(self):
            # empty cluster names?
            if len(self.clusters)==0:
                return False

            # Duplicate cluster names aren't updated:
            print("nclasses: ", self.nclasses, len(self.clusters), len(self.tboxes))
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
            """ Collect any calltype annotations that are present """

            listOfDataFiles = []
            for root, dirs, files in os.walk(self.field("trainDir")):
                for file in files:
                    if file[-5:].lower() == '.data':
                        listOfDataFiles.append(os.path.join(root, file))

            calltypes = {}
            for file in listOfDataFiles:
                # Read the annotation
                segments = annotation.SegmentList()
                segments.parseJSON(file)
                SpSegs = segments.getSpecies(self.field("species"))
                for segix in SpSegs:
                    seg = segments[segix]
                    for label in seg.labels:
                        if label["species"] == self.field("species") and "calltype" in label:
                            if label["calltype"] in calltypes:
                                calltypes.update({label["calltype"]:calltypes[label["calltype"] ] + 1})
                            else:
                                calltypes.update({label["calltype"]:1})
            print(calltypes)                     
            return calltypes

        def getClustersGT(self):
            """ Gets call type clusters from annotations
             returns [parent_audio_file, [segment], [syllables], class_label], number of clusters, median duration
            """
            # Should be in Clustering...
            ctTexts = []
            CTsegments = []
            duration = []
            cl = clustering.Clustering([], [], 5)

            listOfDataFiles = []
            listOfSoundFiles = []
            for root, dirs, files in os.walk(self.field("trainDir")):
                for file in files:
                    if file.lower().endswith('.data'):
                        listOfDataFiles.append(os.path.join(root, file))
                    elif file.lower().endswith('.wav') or file.lower().endswith('.flac'):
                        listOfSoundFiles.append(os.path.join(root, file))

            for file in listOfDataFiles:
                if file[:-5] in listOfSoundFiles:
                    # Read the annotation
                    segments = annotation.SegmentList()
                    segments.parseJSON(file)
                    SpSegs = segments.getSpecies(self.field("species"))
                    for segix in SpSegs:
                        seg = segments[segix]
                        for label in seg.labels:
                            if label["species"] == self.field("species") and "calltype" in label:
                                if label["calltype"] not in ctTexts:
                                    ctTexts.append(label["calltype"])
            for i in range(len(ctTexts)):
                self.clusters[i] = ctTexts[i]

            for file in listOfDataFiles:
                if file[:-5] in listOfSoundFiles:
                    # Read the annotation
                    segments = annotation.SegmentList()
                    soundfile = os.path.join(self.field("trainDir"), file[:-5])
                    segments.parseJSON(os.path.join(self.field("trainDir"), file))
                    SpSegs = segments.getSpecies(self.field("species"))
                    for segix in SpSegs:
                        seg = segments[segix]
                        for label in seg.labels:
                            if label["species"] == self.field("species") and "calltype" in label:
                                # Find the syllables inside this segment
                                # TODO: Filter all the hardcoded parameters into a .txt in config (minlen=0.2, denoise=False)
                                syls = cl.findSyllablesSeg(soundfile, seg, fs=self.field("fs"), denoise=False, minlen=0.2)
                                CTsegments.append([soundfile, seg, syls, list(self.clusters.keys())[list(self.clusters.values()).index(label["calltype"])]])
                                duration.append(seg.end_time - seg.start_time)
            return CTsegments, len(self.clusters), np.median(duration)

        def backupDatafiles(self):
            # Backup original data files before updating them
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
            print("=== UPDATE ANNOTATIONS ===")
            print(f"Valid clusters: {self.clusters}")
            print(f"Total segments in wizard: {len(self.segments)}")
            
            # Check for invalid cluster IDs
            invalid_count = 0
            for i, seg in enumerate(self.segments):
                if seg[-1] not in self.clusters:
                    print(f"  WARNING: Segment {i} has invalid cluster ID {seg[-1]}")
                    invalid_count += 1
            print(f"Segments with invalid cluster IDs: {invalid_count}")
            
            # Group segments by filename
            segments_by_file = {}
            for seg_data in self.segments:
                # seg_data format: [filename, Segment_object, syls, cluster_ID]
                filename = seg_data[0]
                if filename not in segments_by_file:
                    segments_by_file[filename] = []
                segments_by_file[filename].append(seg_data)
            
            print(f"Updating annotation files for {len(segments_by_file)} files")
            
            # Update each file
            for filename, file_segments in segments_by_file.items():
                datafile = filename + ".data"
                print(f"\nProcessing: {os.path.basename(datafile)}")
                
                # Load original segments
                original_segments = annotation.SegmentList()
                original_segments.parseJSON(datafile)
                print(f"  Original segments: {len(original_segments)}")
                
                # Create new segment list
                newsegments = annotation.SegmentList()
                newsegments.metadata = original_segments.metadata
                
                # Keep non-target species segments unchanged
                for seg in original_segments:
                    if self.field("species") not in [fil["species"] for fil in seg.labels]:
                        newsegments.addSegment(seg)
                
                # Add target species segments with updated call types
                for seg_data in file_segments:
                    seg = seg_data[1]  # The Segment object
                    cluster_id = seg_data[-1]
                    
                    if cluster_id in self.clusters:
                        # Update the calltype label
                        species_idx = [fil["species"] for fil in seg.labels].index(self.field("species"))
                        seg.labels[species_idx]["calltype"] = self.clusters[cluster_id]
                        newsegments.addSegment(seg)
                        print(f"  Updated segment {seg.start_time:.1f}-{seg.end_time:.1f} -> {self.clusters[cluster_id]}")
                    else:
                        print(f"  WARNING: Segment has invalid cluster ID {cluster_id}, skipping")
                
                print(f"  Saving {len(newsegments)} segments")
                newsegments.saveJSON(datafile)
            
            print("=== UPDATE ANNOTATIONS DONE ===\n")

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
            self.segsChanged = True
            
            # Find which row was dropped on by checking the Y positions of widgets in each row
            # Look at the textboxes (first widget in each row) to determine row boundaries
            movetoID = 0
            minDistance = float('inf')
            
            for r in range(self.nclasses):
                if r < len(self.tboxes):
                    # Get the Y position of this row's label
                    rowY = self.tboxes[r].y()
                    rowHeight = self.tboxes[r].height()
                    rowCenter = rowY + rowHeight / 2
                    
                    # Find the closest row center
                    distance = abs(dragPosy - rowCenter)
                    if distance < minDistance:
                        minDistance = distance
                        movetoID = r
            
            print(f"dragPosy={dragPosy}, calculated movetoID={movetoID}")
            
            # Clamp to valid cluster range just in case
            movetoID = max(0, min(movetoID, self.nclasses - 1))

            # drags which start and end in the same cluster most likely were just long clicks:
            for ix in range(len(self.picbuttons)):
                if self.picbuttons[ix] == source:
                    if self.segments[ix][-1] == movetoID:
                        source.clicked.emit()
                        return

            # Even if the button that was dragged isn't highlighted, make it so
            source.mark = 'selected'

            print(f"\n=== MOVING SEGMENTS ===")
            print(f"Target cluster ID: {movetoID}")
            print(f"Row positions:")
            for r in range(self.nclasses):
                if r < len(self.tboxes):
                    print(f"  Row {r}: y={self.tboxes[r].y()}, height={self.tboxes[r].height()}")
            
            moved_count = 0
            for ix in range(len(self.picbuttons)):
                if self.picbuttons[ix].mark == 'selected':
                    old_cluster = self.segments[ix][-1]
                    self.segments[ix][-1] = movetoID
                    self.picbuttons[ix].mark = 'none'
                    moved_count += 1
                    print(f"  Moved segment {ix} from cluster {old_cluster} to {movetoID}")
            print(f"Total moved: {moved_count}")

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
            
            print(f"Empty clusters to delete: {todelete}")
            print(f"BEFORE clearButtons: {len(self.picbuttons)} buttons")
            self.clearButtons()
            print(f"AFTER clearButtons: {len(self.picbuttons)} buttons")

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

                # update the segments - only those that need relabeling
                for seg in self.segments:
                    if seg[-1] in labels:
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
                if self.picbuttons[ix].mark == 'selected':
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
                    if self.picbuttons[ix].mark == 'selected':
                        self.segments[ix][-1] = newID
                        self.picbuttons[ix].mark = 'none'

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
                msg = MessagePopup("t", "Select", "Select calls to make the new cluster")
                msg.exec()
                self.completeChanged.emit()
                return

        def deleteSelectedSegs(self):
            """ Listener for Delete button to delete the selected segments completely.
            """
            inds = []
            for ix in range(len(self.picbuttons)):
                if self.picbuttons[ix].mark == 'selected':
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
                msg = MessagePopup("w", "Name error", "Duplicate cluster names! \nTry again")
                msg.exec()
                self.completeChanged.emit()
                return

            if "(Other)" in names:
                msg = MessagePopup("w", "Name error", "Name \"(Other)\" is reserved! \nTry again")
                msg.exec()
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

            # # largest spec will be this wide
            # if len(self.segments)<=1:
            #     return
            print(len(self.segments))
            print(self.segments)

            maxspecsize = max([seg[1].end_time-seg[1].start_time for seg in self.segments]) * self.field("fs") // 256

            # Create the buttons for each segment
            self.minsg = 1
            self.maxsg = 1
            for seg in self.segments:
                sp = spectrogram.Spectrogram(512, 256)
                sp.readSoundFile(seg[0], seg[1].end_time-seg[1].start_time, seg[1].start_time, silent=True)

                # set increment to depend on Fs to have a constant scale of 256/tgt seconds/px of spec
                incr = 256 * sp.audio_data.sample_rate // self.field("fs")
                #_ = sp.spectrogram(window='Hann', sgType='Standard',incr=incr, mean_normalise=True, onesided=True, need_even=False)
                sg = sp.spectrogram(window_width=self.config['window_width'], incr=self.config['incr'],window=self.config['windowType'],sgType=self.config['sgType'],sgScale=self.config['sgScale'],nfilters=self.config['nfilters'],mean_normalise=self.config['sgMeanNormalise'],equal_loudness=self.config['sgEqualLoudness'],onesided=self.config['sgOneSided'])
                #sg = sp.normalisedSpec("Log")
                sg = sp.normalisedSpec(self.config['sgNormMode'])
                
                # buffer the image to largest spec size, so that the resulting buttons would have equal scale
                if sg.shape[0]<maxspecsize:
                    padlen = int(maxspecsize - sg.shape[0])//2
                    sg = np.pad(sg, ((padlen, padlen), (0,0)), 'constant', constant_values=np.quantile(sg, 0.1))

                self.minsg = min(self.minsg, np.min(sg))
                self.maxsg = max(self.maxsg, np.max(sg))

                newButton = PicButton(1, np.fliplr(sg), sp, sp.audio_data, seg[1].end_time-seg[1].start_time, 0, seg[1].end_time, self.lut, cluster=True)
                self.picbuttons.append(newButton)
            # (updateButtons will place them in layouts and show them)

        def addButtons_TBD(self,ims=None,calls=None,calltypes=None,sp=None):
            """ Only makes the PicButtons and self.clusters dict
            """
            self.picbuttons = []
            # TODO: Here
            #print(len(ims))
            #print(len(ims[0]))
            #print(np.shape(calls[0][0]))
            self.minsg = 1
            self.maxsg = 1
            if ims is not None:
                for i in range(len(ims)):
                    for j in range(len(ims[i])):
                        self.minsg = min(self.minsg, np.min(ims[i][j]))
                        self.maxsg = max(self.maxsg, np.max(ims[i][j]))
                        # TODO: get length right
    #def __init__(self, index, spec, audiodata, audioFormat, duration, unbufStart, unbufStop, lut, guides=None, guidecol=None, loop=False, parent=None, cluster=False):
                        # Create temporary AudioData object for this specific audio segment
                        temp_audio_data = audio_data.AudioData(
                            data=calls[i][j],
                            sample_rate=sp.audio_data.sample_rate,
                            file_length=len(calls[i][j]),
                            audio_format=sp.audio_data
                        )
                        newButton = PicButton(1, np.fliplr(ims[i][j]), temp_audio_data, sp.audio_data, len(calls[i][j])/sp.audio_data.sample_rate, 0, len(calls[i][j]), self.lut, cluster=True)
                        #newButton = PicButton(1, np.fliplr(ims[i][j]), sp.data, sp.audio_data, calls[1][1]-calls[1][0], 0, seg[1][1], self.lut, cluster=True)
                        self.picbuttons.append(newButton)
                        self.clusters = calltypes
            else:
                self.clusters = []
                for i in range(self.nclasses):
                    self.clusters.append((i, 'Cluster_' + str(i)))
                self.clusters = dict(self.clusters)     # Dictionary of {ID: cluster_name}

                # largest spec will be this wide
                maxspecsize = max([seg[1].end_time-seg[1].start_time for seg in self.segments]) * self.field("fs") // 256

                # Create the buttons for each segment
                self.minsg = 1
                self.maxsg = 1
                for seg in self.segments:
                    sp = spectrogram.Spectrogram(512, 256)
                    sp.readSoundFile(seg[0], seg[1].end_time-seg[1].start_time, seg[1].start_time, silent=True)
    
                    # set increment to depend on Fs to have a constant scale of 256/tgt seconds/px of spec
                    incr = 256 * sp.audio_data.sample_rate // self.field("fs")
                    #_ = sp.spectrogram(window='Hann', sgType='Standard',incr=incr, mean_normalise=True, onesided=True, need_even=False)
                    sg = sp.spectrogram(window_width=self.config['window_width'], incr=self.config['incr'],window=self.config['windowType'],sgType=self.config['sgType'],sgScale=self.config['sgScale'],nfilters=self.config['nfilters'],mean_normalise=self.config['sgMeanNormalise'],equal_loudness=self.config['sgEqualLoudness'],onesided=self.config['sgOneSided'])
                    #sg = sp.normalisedSpec("Log")

                    # buffer the image to largest spec size, so that the resulting buttons would have equal scale
                    if sg.shape[0]<maxspecsize:
                        padlen = int(maxspecsize - sg.shape[0])//2
                        sg = np.pad(sg, ((padlen, padlen), (0,0)), 'constant', constant_values=np.quantile(sg, 0.1))
    
                    self.minsg = min(self.minsg, np.min(sg))
                    self.maxsg = max(self.maxsg, np.max(sg))

                    newButton = PicButton(1, np.fliplr(sg), sp, sp.audio_data, seg[1].end_time-seg[1].start_time, 0, seg[1].end_time, self.lut, cluster=True)
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
                            self.picbuttons[ix].mark = 'none'
                            self.picbuttons[ix].buttonClicked = False
                            self.picbuttons[ix].setChecked(False)
                            self.picbuttons[ix].repaint()

        def updateButtons(self):
            """ Draw the existing buttons, and create check- and text-boxes.
            Called when merging clusters or initializing the page. """
            print(f"=== updateButtons called ===")
            print(f"Total segments: {len(self.segments)}")
            print(f"Total picbuttons: {len(self.picbuttons)}")
            print(f"nclasses: {self.nclasses}")
            
            # Count segments per cluster
            for r in range(self.nclasses):
                count = sum(1 for seg in self.segments if seg[-1] == r)
                print(f"  Cluster {r} ({self.clusters[r]}): {count} segments")
            
            self.cboxes = []    # List of check boxes
            self.tboxes = []    # Corresponding list of text boxes
            
            for r in range(self.nclasses):
                c = 0
                tbox = QLineEdit(self.clusters[r])
                tbox.setMinimumWidth(80)
                tbox.setMaximumHeight(150)
                tbox.setStyleSheet("border: none;")
                tbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
                # Keep them in their original order from the segments list
                buttons_added = 0
                for segix in range(len(self.segments)):
                    if self.segments[segix][-1] == r:
                        print(f"  Adding button {segix} to row {r}, col {c}")
                        self.flowLayout.addWidget(self.picbuttons[segix], r, c)
                        self.picbuttons[segix].show()
                        c += 1
                        buttons_added += 1
                print(f"  Row {r}: added {buttons_added} buttons")
            
            self.flowLayout.adjustSize()
            self.flowLayout.update()
            print(f"=== updateButtons done ===\n")
            # Apply colour and volume levels
            self.specControls.emitAll()

        def clearButtons(self):
            """ Remove existing buttons, call when merging clusters
            """
            print(f"=== clearButtons called ===")
            print(f"Layout item count before clear: {self.flowLayout.layout.count()}")
            for ch in self.cboxes:
                ch.hide()
            for tbx in self.tboxes:
                tbx.hide()
            items_removed = 0
            for btnum in reversed(range(self.flowLayout.layout.count())):
                item = self.flowLayout.layout.itemAt(btnum)
                if item is not None:
                    self.flowLayout.layout.removeItem(item)
                    r, c = self.flowLayout.items[item.widget()]
                    del self.flowLayout.items[item.widget()]
                    del self.flowLayout.rows[r][c]
                    item.widget().hide()
                    items_removed += 1
            print(f"Items removed from layout: {items_removed}")
            print(f"Layout item count after clear: {self.flowLayout.layout.count()}")
            self.flowLayout.update()
            print(f"=== clearButtons done ===\n")

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

    # page 4 - set params for training
    class WPageParams(QWizardPage):
        def __init__(self, method, cluster, segments, picbtn, parent=None):
            super(BuildRecAdvWizard.WPageParams, self).__init__(parent)
            self.setTitle("Training parameters: %s" % cluster)
            self.setSubTitle("These fields were completed using the training data. Adjust if required.\nWhen ready, "
                             "press \"Train\". The process may take a long time.")
            #self.setMinimumSize(350, 430)
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.MinimumExpanding)
            self.adjustSize()

            self.method = method

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
            if method=="wv":
                self.minlen = QLineEdit(self)
                form1_step4.addRow('Min call length (sec)', self.minlen)
                self.maxlen = QLineEdit(self)
                form1_step4.addRow('Max call length (sec)', self.maxlen)
                self.avgslen = QLineEdit(self)
                form1_step4.addRow('Avg syllable length (sec)', self.avgslen)
                self.maxgap = QLineEdit(self)
                form1_step4.addRow('Max gap between syllables (sec)', self.maxgap)
            elif method=="chp":
                self.chpwin = QLineEdit(self)
                form1_step4.addRow('Window size (sec)', self.chpwin)
                self.minlen = QLineEdit(self)
                form1_step4.addRow('Min call length (sec)', self.minlen)
                self.maxlen = QLineEdit(self)
                form1_step4.addRow('Max call length (sec)', self.maxlen)
            else:
                print("ERROR: unrecognized method", method)
                return

            # FreqRange parameters
            self.fLow = QSlider(Qt.Orientation.Horizontal)
            self.fLow.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.fLow.setTickInterval(2000)
            self.fLow.setRange(0, 32000)
            self.fLow.setSingleStep(100)
            self.fLow.valueChanged.connect(self.fLowChange)
            self.fLowtext = QLabel('')
            form1_step4.addRow('', self.fLowtext)
            form1_step4.addRow('Lower frq. limit (Hz)', self.fLow)
            self.fHigh = QSlider(Qt.Orientation.Horizontal)
            self.fHigh.setTickPosition(QSlider.TickPosition.TicksBelow)
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

            self.setButtonText(QWizard.WizardButton.NextButton, 'Train >')

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
            pageSegs = annotation.SegmentList()
            for longseg in self.segments:
                # long seg has format: [file [segment] clusternum]
                pageSegs.addSegment(longseg[1])
            len_min, len_max, f_low, f_high = pageSegs.getSummaries()
            self.maxlen.setText(str(round(np.max(len_max),2)))

            self.fLow.setRange(0, int(fs/2))
            self.fLow.setValue(max(0, int(np.min(f_low))))
            self.fHigh.setRange(0, int(fs/2))
            if np.max(f_high) == 0:
                # happens when no segments have y limits
                f_high = fs/2
            self.fHigh.setValue(min(fs/2,int(np.max(f_high))))

            # this is just the minimum call length:
            self.minlen.setText(str(round(np.min(len_min),2)))

            # need some more properties for the older methods
            if self.method=="wv":
                if not self.wizard().clusterPage.hasCTannotations:
                    # Get max inter syllable gap
                    gaps = []
                    maxgap = 0
                    for longseg in self.segments:
                        if len(longseg[2]) > 1:
                            for i in range(len(longseg[2]) - 1):
                                gaps.append(longseg[2][i + 1][0] - longseg[2][i][1])
                    if len(gaps) > 0:
                        maxgap = max(gaps)
                    else:
                        maxgap = 0

                    # get average syllable length
                    syllen = []
                    for longseg in self.segments:
                        for i in range(len(longseg[2])):
                            syllen.append(longseg[2][i][1] - longseg[2][i][0])

                    avgslen = np.mean(syllen)
                else:
                    maxgap = 0.5    # TODO
                    avgslen = 0.5  # TODO

                self.maxgap.setText(str(round(maxgap,2)))
                self.avgslen.setText(str(round(avgslen,2)))

            elif self.method=="chp":
                # this is window size:
                # let's say, 10 % of the min call length
                self.chpwin.setText(str(round(np.min(len_min/10),2)))

            self.adjustSize()

    # page 5 - run training, show ROC
    class WPageTrain(QWizardPage):
        def __init__(self, method, id, clustID, clustname, segments, parent=None):
            super(BuildRecAdvWizard.WPageTrain, self).__init__(parent)
            self.setTitle('Training results')
            self.setSubTitle('Click on the graph at the point where you would like the classifier to trade-off false positives with false negatives. Points closest to the top-left are best.')
            self.setMinimumSize(520, 440)
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            self.adjustSize()

            self.segments = segments
            self.clust = clustname
            self.clusterID = clustID
            # this ID links it to the parameter fields
            self.pageId = id

            self.method = method

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
            self.bestThr = QLineEdit()
            self.bestNodes = QLineEdit()
            self.bestFrqBands = QLineEdit()
            self.bestThr.setReadOnly(True)
            self.bestNodes.setReadOnly(True)
            self.bestFrqBands.setReadOnly(True)
            self.bestThr.setStyleSheet("QLineEdit { color : #808080; }")
            self.bestNodes.setStyleSheet("QLineEdit { color : #808080; }")
            self.bestFrqBands.setStyleSheet("QLineEdit { color : #808080; }")
            self.filtSummary = QFormLayout()

            if self.method=="wv":
                self.bestM = QLineEdit()
                self.bestM.setReadOnly(True)
                self.bestM.setStyleSheet("QLineEdit { color : #808080; }")
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
                if self.method == "wv":
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
                self.completeChanged.emit()

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
            fLow = int(self.field("fLow"+str(self.pageId)))
            fHigh = int(self.field("fHigh"+str(self.pageId)))
            if self.method=="wv":
                minlen = float(self.field("minlen"+str(self.pageId)))
                maxlen = float(self.field("maxlen"+str(self.pageId)))
                maxgap = float(self.field("maxgap" + str(self.pageId)))
                avgslen = float(self.field("avgslen" + str(self.pageId)))
                # note: for each page we reset the filter to contain 1 calltype
                self.wizard().speciesData["Filters"] = [{'calltype': self.clust, 'TimeRange': [minlen, maxlen, avgslen, maxgap], 'FreqRange': [fLow, fHigh]}]
            elif self.method=="chp":
                minlen = float(self.field("minlen"+str(self.pageId)))
                maxlen = float(self.field("maxlen"+str(self.pageId)))
                chpwin = float(self.field("chpwin"+str(self.pageId)))
                # Important: chpwin is rounded to nearest multiple of 32/Fs
                # to ensure that this window corresponds to integer number of wavelet coefs.
                # Not reading from the field to avoid rounding errors.
                # But any change here must be reflected in the training as well!
                MINCHPWIN = 32/self.wizard().speciesData['SampleRate']
                chpwin = math.ceil(chpwin/MINCHPWIN)*MINCHPWIN
                print("Changepoint window was rounded to", chpwin)

                self.wizard().speciesData["Filters"] = [{'calltype': self.clust, 'TimeRange': [minlen, maxlen, 0.0, 0.0], 'FreqRange': [fLow, fHigh]}]

            # export 1/0 ground truth
            window = 1
            inc = None
            with pg.BusyCursor():
                for root, dirs, files in os.walk(self.field("trainDir")):
                    for file in files:
                        soundFile = os.path.join(root, file)
                        if (file.lower().endswith('.wav') or file.lower().endswith('.flac')) and os.stat(soundFile).st_size != 0 and file + '.data' in files:
                            pageSegs = annotation.SegmentList()
                            pageSegs.parseJSON(soundFile + '.data')

                            # CLUSTERS COME IN HERE:
                            # replace segments with the current cluster
                            # (self.segments is already selected to be this cluster only)
                            pageSegs.clear()
                            for longseg in self.segments:
                                # long seg has format: [file [segment] clusternum]
                                if longseg[0] == soundFile:
                                    pageSegs.addSegment(longseg[1])

                            # So, each page will overwrite a file with the 0/1 annots,
                            # and recalculate the stats for that cluster.

                            # exports 0/1 annotations
                            if self.method=="wv":
                                pageSegs.exportGT(soundFile, self.field("species"), resolution=1.0)
                            elif self.method=="chp":
                                pageSegs.exportGT(soundFile, self.field("species"), resolution=chpwin)


            # calculate cluster centres
            # (self.segments is already selected to be this cluster only)
            with pg.BusyCursor():
                cl = clustering.Clustering([], [], 5)
                self.clustercentre = cl.getClusterCenter(self.segments, self.field("fs"), fLow, fHigh, self.wizard().clusterPage.feature, self.wizard().clusterPage.duration)

            # Get detection measures over all M,thr combinations
            print("starting wavelet training")
            with pg.BusyCursor():
                opstartingtime = time.time()
                ws = wavelet_segment.WaveletSegment(self.wizard().speciesData)
                if self.method=="wv":
                    # returns 2d lists of nodes over M x thr, or stats over M x thr
                    numthr = 50
                    self.thrList = np.linspace(0.2, 1, num=numthr)
                    self.MList = np.linspace(avgslen, avgslen, num=1)
                    # options for training are:
                    #  recold - no antialias, recaa - partial AA, recaafull - full AA
                    #  Window and inc - in seconds
                    self.nodes, TP, FP, TN, FN = ws.waveletSegment_train(self.field("trainDir"),
                                                                    self.thrList, self.MList,
                                                                    d=False,
                                                                    learnMode="recaa", window=window, inc=inc)
                elif self.method=="chp":
                    # Note: using energies averaged over window size set before
                    numthr = 9
                    self.thrList = np.geomspace(0.03, 10, num=numthr)
                    print("trainDir: ", self.field("trainDir"))
                    print("thrList: ", self.thrList)
                    print("maxlen: ", maxlen)
                    print("chpwin: ", chpwin)
                    self.nodes, TP, FP, TN, FN = ws.waveletSegment_trainChp(self.field("trainDir"),
                                                                    self.thrList,
                                                                    maxlen=maxlen, window=chpwin)

                print("Filtered nodes: ", self.nodes)
                print("TRAINING COMPLETED IN ", time.time() - opstartingtime)
                self.TPR = TP/(TP+FN)
                self.FPR = 1 - TN/(FP+TN)
                print("TP rate: ", self.TPR)
                print("FP rate: ", self.FPR)

                self.marker.set_visible(False)
                self.figCanvas.plotmeagain(self.TPR, self.FPR)

        def getFrqBands(self, nodes):
            fRanges = []
            for node in nodes:
                f1, f2 = wavelet_functions.getWCFreq(node, self.field("fs"))
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
            self.setMinimumSize(430, 300)
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            self.adjustSize()

            self.lblTrainDir = QLabel()
            self.lblTrainDir.setStyleSheet("QLabel { color : #808080; }")
            self.lblSpecies = QLabel()
            self.lblSpecies.setStyleSheet("QLabel { color : #808080; }")
            space = QLabel()
            space.setFixedHeight(20)
            spaceH = QLabel()
            spaceH.setFixedWidth(30)

            self.lblFilter = QLabel('')
            self.lblFilter.setWordWrap(True)
            self.lblFilter.setStyleSheet("QLabel { color : #808080; }")

            # filter dir listbox
            self.listFiles = QListWidget()
            self.listFiles.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            self.listFiles.setMinimumHeight(200)
            filtdir = QDir(filtdir).entryList(filters=QDir.Filter.NoDotAndDotDot | QDir.Filter.Files)
            for file in filtdir:
                if file.endswith('.txt'):
                    item = QListWidgetItem(self.listFiles)
                    item.setText(file)

            # filter file name
            self.enterFiltName = QLineEdit()

            trainFiltValid = FiltValidator(self.listFiles, check_reserved_m=True)
            self.enterFiltName.setValidator(trainFiltValid)

            # layouts
            vboxHead = QFormLayout()
            vboxHead.addRow("Training data:", self.lblTrainDir)
            vboxHead.addRow("Target species:", self.lblSpecies)
            
            scrollFilter = QScrollArea()
            scrollFilter.setWidgetResizable(True)
            scrollFilter.setWidget(self.lblFilter)
            scrollFilter.setMinimumHeight(30)

            layout = QVBoxLayout()
            layout.addLayout(vboxHead)
            layout.addWidget(space)
            layout.addWidget(QLabel("The following recogniser was produced:"))
            layout.addWidget(scrollFilter)
            layout.addWidget(QLabel("Currently available recognisers"))
            layout.addWidget(self.listFiles)
            layout.addWidget(space)
            layout.addWidget(QLabel("Enter file name (must be unique)"))
            layout.addWidget(self.enterFiltName)

            self.setButtonText(QWizard.WizardButton.FinishButton, 'Save and Finish')
            self.setLayout(layout)

        def initializePage(self):
            self.lblTrainDir.setText(self.field("trainDir"))
            self.lblSpecies.setText(self.field("species"))

            self.wizard().speciesData["Filters"] = []

            # collect parameters from training pages (except this)
            for pageId in self.wizard().trainpages[:-1]:
                ct = self.wizard().page(pageId + 1).clust
                # main parameters, depending on the method:
                if self.wizard().method=="wv":
                    minlen = float(self.field("minlen"+str(pageId)))
                    maxlen = float(self.field("maxlen"+str(pageId)))
                    maxgap = float(self.field("maxgap" + str(pageId)))
                    avgslen = float(self.field("avgslen" + str(pageId)))
                    fLow = int(self.field("fLow"+str(pageId)))
                    fHigh = int(self.field("fHigh"+str(pageId)))
                    thr = float(self.field("bestThr"+str(pageId)))
                    M = float(self.field("bestM"+str(pageId)))
                    nodes = eval(self.field("bestNodes"+str(pageId)))

                    newSubfilt = {'calltype': ct, 'TimeRange': [minlen, maxlen, avgslen, maxgap], 'FreqRange': [fLow, fHigh], 'WaveletParams': {"thr": thr, "M": M, "nodes": nodes}, 'ClusterCentre': list(self.wizard().page(pageId+1).clustercentre), 'Feature': self.wizard().clusterPage.feature}
                elif self.wizard().method=="chp":
                    chpwin = float(self.field("chpwin"+str(pageId)))
                    minlen = float(self.field("minlen"+str(pageId)))
                    maxlen = float(self.field("maxlen"+str(pageId)))
                    fLow = int(self.field("fLow"+str(pageId)))
                    fHigh = int(self.field("fHigh"+str(pageId)))
                    thr = float(self.field("bestThr"+str(pageId)))
                    nodes = eval(self.field("bestNodes"+str(pageId)))

                    # Important: chpwin is rounded to nearest multiple of 32/Fs
                    # to ensure that this window corresponds to integer number of wavelet coefs.
                    # Not reading from the field to avoid rounding errors.
                    # But any change here must be reflected in the training as well!
                    MINCHPWIN = 32/self.wizard().speciesData['SampleRate']
                    chpwin = round(chpwin/MINCHPWIN)*MINCHPWIN

                    newSubfilt = {'calltype': ct, 'TimeRange': [minlen, maxlen, 0.0, 0.0], 'FreqRange': [fLow, fHigh], 'WaveletParams': {"thr": thr, "nodes": nodes, "win": chpwin}, 'ClusterCentre': list(self.wizard().page(pageId+1).clustercentre), 'Feature': self.wizard().clusterPage.feature}
                else:
                    print("ERROR: unrecognized method %s" % self.wizard().method)
                    return

                # optionally, attach TPR/FPR:
                tpr = float(self.field("TPR" + str(pageId)))
                fpr = float(self.field("FPR" + str(pageId)))
                if tpr != -1:
                    newSubfilt["TPR, FPR"] = [tpr, fpr]

                print(newSubfilt)
                self.wizard().speciesData["Filters"].append(newSubfilt)
                # collate ROC data
                self.wizard().ROCData[ct] = [self.wizard().page(pageId + 1).TPR.tolist()[0], self.wizard().page(pageId + 1).FPR.tolist()[0], self.wizard().page(pageId + 1).nodes[0]]
                self.wizard().ROCData["thr"] = self.wizard().page(pageId + 1).thrList.tolist()

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
    def __init__(self, filtdir, config, method, parent=None):
        # method: "wv" or "chp" to easily switch between old wavelet filter
        # and the new changepoint detection
        super(BuildRecAdvWizard, self).__init__()
        self.setWindowTitle("Build Recogniser")
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        if platform.system() == 'Linux':
            self.setWindowFlags(self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint)
        else:
            self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)

        # add the Save & Test button
        self.saveTestBtn = QPushButton("Save and Test")
        self.setButton(QWizard.WizardButton.CustomButton1, self.saveTestBtn)
        self.setButtonLayout([QWizard.WizardButton.Stretch, QWizard.WizardButton.BackButton, QWizard.WizardButton.NextButton, QWizard.WizardButton.CustomButton1, QWizard.WizardButton.FinishButton, QWizard.WizardButton.CancelButton])
        self.setOptions(QWizard.WizardOption.NoBackButtonOnStartPage | QWizard.WizardOption.HaveCustomButton1)

        self.filtersDir = filtdir

        self.method = method

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
        self.ROCData = {}
        # then a pair of pages for each calltype will be created by redoTrainPages.

        # Size adjustment between pages:
        self.saveTestBtn.setVisible(False)
        self.currentIdChanged.connect(self.pageChangeResize)
        # try to deal with buttons catching Enter presses
        self.buttons = [self.button(t) for t in (QWizard.WizardButton.NextButton, QWizard.WizardButton.FinishButton, QWizard.WizardButton.CustomButton1)]
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
            page4 = BuildRecAdvWizard.WPageParams(self.method, value, newsegs, newbtns[0])
            page4.lblSpecies.setText(self.field("species"))
            page4.numSegs.setText(str(len(newsegs)))
            pageid = self.addPage(page4)
            self.trainpages.append(pageid)

            # page 5: get training results
            page5 = BuildRecAdvWizard.WPageTrain(self.method, pageid, key, value, newsegs)
            self.addPage(page5)

            if self.method=="wv":
                # Note: these need to be unique hence attaching the number
                page4.registerField("minlen"+str(pageid), page4.minlen)
                page4.registerField("maxlen"+str(pageid), page4.maxlen)
                page4.registerField("maxgap" + str(pageid), page4.maxgap)
                page4.registerField("avgslen" + str(pageid), page4.avgslen)
                page4.registerField("fLow"+str(pageid), page4.fLow)
                page4.registerField("fHigh"+str(pageid), page4.fHigh)

                # note: pageid is the same for both page fields
                page5.registerField("bestThr"+str(pageid)+"*", page5.bestThr)
                page5.registerField("bestM"+str(pageid)+"*", page5.bestM)
                page5.registerField("bestNodes"+str(pageid)+"*", page5.bestNodes)
            elif self.method=="chp":
                page4.registerField("chpwin"+str(pageid), page4.chpwin)
                page4.registerField("minlen"+str(pageid), page4.minlen)
                page4.registerField("maxlen"+str(pageid), page4.maxlen)
                page4.registerField("fLow"+str(pageid), page4.fLow)
                page4.registerField("fHigh"+str(pageid), page4.fHigh)

                # note: pageid is the same for both page fields
                page5.registerField("bestThr"+str(pageid)+"*", page5.bestThr)
                # While this stores the output nodes from ROC, which in principle may be different from page4
                page5.registerField("bestNodes"+str(pageid)+"*", page5.bestNodes)
            else:
                print("ERROR: unrecognized method %s" % self.method)
                return

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
        if obj in self.buttons and event.type() == QEvent.Type.Show:
            obj.setDefault(False)
        return super(BuildRecAdvWizard, self).eventFilter(obj, event)
