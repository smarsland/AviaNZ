
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

from PyQt6 import QtGui
from PyQt6.QtGui import *
from PyQt6.QtWidgets import QLabel, QDialog, QPushButton, QHBoxLayout, QVBoxLayout, QToolButton, QStyle, QScrollArea # listing some explicitly to make syntax checks lighter
from PyQt6.QtWidgets import *
from PyQt6.QtCore import QPointF, QTime, Qt, QSize, QTimer

import pyqtgraph as pg

import numpy as np
from src.ui.colourMaps import colourMaps
from src.ui.components.audio_player import ControllableAudio
from src.ui.components.buttons_and_controls import BrightContrVol
from src.ui.components.layout_widgets import PartlyResizableGLW
from src.ui.components.species_menus import BatSelectionMenu, BirdSelectionMenu

import copy

import re

pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class HumanClassify1(QDialog):
    # This dialog allows the checking of classifications for segments.
    # It shows a single segment at a time, working through all the segments.

    def __init__(self, lut, cmapInverted, brightness, contrast, shortBirdList, longBirdList, knownCalls, batList, multipleBirds, audioFormat, guidecols, plotAspect=2, loop=False, autoplay=False, parent=None, reorderShortList=False):
        # plotAspect: initial stretch factor in the X direction
        QDialog.__init__(self, parent)
        self.setWindowTitle('Check Classifications')
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
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
        self.wPlot = PartlyResizableGLW()
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
        self.buttonPrev.setIcon(QtGui.QIcon('src/resources/images/undo.png'))
        self.buttonPrev.setIconSize(iconSize)
        self.buttonPrev.setStyleSheet("padding: 5px 5px 5px 5px")

        self.buttonNext = QToolButton()
        self.buttonNext.setIcon(QtGui.QIcon('src/resources/images/questionL.png'))
        self.buttonNext.setIconSize(iconSize)
        self.buttonNext.setStyleSheet("padding: 5px 5px 5px 5px")

        self.correct = QToolButton()
        self.correct.setIcon(QtGui.QIcon('src/resources/images/check-mark2.png'))
        self.correct.setIconSize(iconSize)
        self.correct.setStyleSheet("padding: 5px 5px 5px 5px")

        self.delete = QToolButton()
        self.delete.setIcon(QtGui.QIcon('src/resources/images/deleteL.png'))
        self.delete.setIconSize(iconSize)
        self.delete.setStyleSheet("padding: 5px 5px 5px 5px")

        # TODO: Icon
        self.buttonPlus = QToolButton()
        self.buttonPlus.setIcon(QtGui.QIcon('src/resources/images/add.png'))
        self.buttonPlus.setIconSize(iconSize)
        #self.buttonPlus.setIcon(QtGui.QIcon('src/resources/images/iconplus.png'))
        #self.buttonPlus.setText('+')
        self.buttonPlus.setStyleSheet("padding: 5px 5px 5px 5px")

        self.ctLabel = QLabel("")
        self.ctLabel.setStyleSheet("QLabel { font-size:16pt; font-weight: bold}")
        self.ctLabel.hide()

        # Audio playback object
        self.media_obj = ControllableAudio(sp=None,audioFormat=audioFormat,useBar=True)
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
        self.specControls = BrightContrVol(brightness, contrast, self.cmapInverted)
        self.specControls.colChanged.connect(self.setColourLevels)
        self.specControls.volChanged.connect(self.volSliderMoved)

        # zoom buttons
        self.zoomInBtn = QToolButton()
        self.zoomOutBtn = QToolButton()
        self.zoomInBtn.setIcon(QtGui.QIcon('src/resources/images/zoom-in.png'))
        self.zoomOutBtn.setIcon(QtGui.QIcon('src/resources/images/search.png'))
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

    def setSegNumbers(self, accepted, deleted, questioned, total):
        #print(accepted,deleted,questioned,total)
        text1 = "calls accepted: " + str(accepted) + ", deleted: " + str(deleted) + ", questioned: " + str(questioned)
        text2 = str(total - accepted - deleted - questioned) + " to go"
        self.numberDone.setText(text1)
        self.numberLeft.setText(text2)
        # based on these, update "previous" arrow status
        self.buttonPrev.setEnabled((accepted+deleted+questioned)>0)
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
        self.currentSegment.labels = new_labels        
                
        # Put the selected bird name at the top of the list.
        if self.reorderShortList:
            if species_changed in self.batList:
                self.batList.remove(species_changed)
            else:
                del self.batList[-1]
            self.batList.insert(0,species_changed)
    
    def birdLabelsUpdated(self,new_labels,species_changed,callname_changed,new_certainty):
        self.currentSegment.labels = new_labels
        
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

        labels = copy.deepcopy(self.currentSegment.labels)
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

        labels = copy.deepcopy(self.currentSegment.labels)
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
        time1 = QTime(0,0,0).addSecs(int(segment.start_time)).toString('hh:mm:ss')
        time2 = QTime(0,0,0).addSecs(int(segment.end_time)).toString('hh:mm:ss')
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
        labels = self.segment.labels
        if "calltype" in labels[0]:
            self.ctLabel.setText(labels[0]["calltype"])
        else:
            self.ctLabel.setText("")

        if self.autoplay:
            self.playSeg()
    
    def updateTitle(self):
        specnames = []
        labels = self.segment.labels
        for lab in labels:
            specnames.append(lab["species"]+" ["+lab["calltype"]+"]" if "calltype" in lab else lab["species"])
        specnames = list(set(specnames))
        self.species.setText(','.join(specnames))

    def updateSelectionMenu(self):
        self.reorderShortBirdList(self.segment)
        currentLabels = self.segment.labels
        if self.batmode:
            self.menuSpeciesSelection = BatSelectionMenu(
                batList=self.batList, 
                currentLabels=currentLabels, 
                parent=self, 
                unsure=False,
                multipleBirds=self.multipleBirds
            )
            self.menuSpeciesSelection.labelsUpdated.connect(self.batLabelsUpdated)
        else:
            self.menuSpeciesSelection = BirdSelectionMenu(
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
