# Interface.py
#
# This is the main class for the AviaNZ interface
# It's fairly simple, but seems to work OK
# Version 0.9 16/04/17
# Author: Stephen Marsland, with input from Nirosha Priyadarshani

#     <one line to give the program's name and a brief idea of what it does.>
#    Copyright (C) <year>  <name of author>

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

import sys, os, json  #glob
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.phonon as phonon

from scipy.io import wavfile
import numpy as np

import pyqtgraph as pg
pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *
import pyqtgraph.functions as fn

import SignalProc
import Segment
#import Features
#import Learning
# ==============
# TODO

# Need to make the window shrinkable and get the sizing right for the screen

# Finish implementation for button to show individual segments to user and ask for feedback and the other feedback dialogs
# Ditto lots of segments at once

# Would it be good to smooth the image?

# Modify the user manual to include shift-click, backspace key, etc.

# Scrollbar below the spectrogram to move through the file -> where are the buttons?

# Some way of allowing marking of 'possible' or 'maybe' or 'unsure' classifications -> menu option?
# Challenge here is, can you make a menu option that is selectable and has an arrow outwards?

# Colormaps
    # Find some not-horrible ones!
    # HistogramLUTItem
# Pause on segment play (and also on bandpass limited play)

# Mouse location printing -> Is it correct?

# Dynamically updating context menu -> done, should it be an option?

# Keyboard input
    # Hot key to say 'same bird' -> done, use shift-click.

# Allow non-integer thresholds for eg wavelets

# Tiny segments can appear that are (virtually) impossible to delete -> work out how they are made, stop it

# Enable zooming of the spectrogram in the y-axis, and do automatically after bandpass filtering

# Decide on license

# Add a minimum length of time for a segment, debug the segmentation algorithms
# Finish sorting out parameters for median clipping segmentation, energy segmentation
# Finish cross-correlation to pick out similar bits of spectrogram -> and what other methods?

# Add in the wavelet segmentation for kiwi, ruru
# Think about nice ways to train them

# The ruru file is a good one to play with for now



# Show a mini picture of spectrogram images by the bird names in the file, or a cheat sheet or similar

# Try yaapt or bana as well as yin for fundamental frequency
# Get the line rather than the points, (with filtering?) and add shape metric

# Is there something weird with spectrogram and denoising? Why are there spikes?
# Should load in the new sound file after denoising and play that

# Have a busy bar when computing

# Finish the raven features

# How to set bandpass params? -> is there a useful plot to help? -> function of sampleRate

# Look into ParameterTree for saving the config stuff in particular
# Better loading of files -> paging, not computing whole spectrogram (how to deal with overview? -> coarser spec?)
    # Maybe: check length of file. If > 5 mins, load first 5 only (? how to move to next 5?)

# Overall layout -> buttons on the left in a column, or with tabs? Add menu?

# Implement something for the Classify button:
    # Take the segments that have been given and try to classify them in lots of ways:
    # Cross-correlation, DTW, shape metric, features and learning

# Testing data
# Documentation
# Licensing

# Some bug in denoising? -> tril1
# More features, add learning!

# Needs decent testing

# Option to turn button menu on/off?

# Minor:
# Turn stereo sound into mono using librosa, consider always resampling to 22050Hz (except when it's less in file :) )
# Font size to match segment size -> make it smaller, could also move it up or down as appropriate
# Use intensity of colour to encode certainty?
# Is play all useful? Would need to move the plots as appropriate
# If don't select something in context menu get error -> not critical

# Things to consider:
    # Second spectrogram (currently use right button for interleaving)? My current choice is no as it takes up space
    # Put the labelling (and loading) in a dialog to free up space? -> bigger plots

# Look at raven and praat and luscinia -> what else is actually useful? Other annotations on graphs?

# Given files > 5 mins, split them into 5 mins versions anyway (code is there, make it part of workflow)
# Don't really want to load the whole thing, just 5 mins, and then move through with arrows -> how?
# This is sometimes called paging, I think. (y, sr = librosa.load(filename, offset=15.0, duration=5.0) might help. Doesn't do much for the overview through)

# As well as this version with pictures, will need to be able to call various bits to work offline
# denoising, segmentation, etc. that will then just show some pictures

# Things to remember
    # When adding new classes, make sure to pass new data to them in undoing and loading

# This version has the selection of birds using a context menu and then has removed the radio buttons
# Code is still there, though, just commented out. Add as an option?

# Get suggestions from the others

# Diane:
    # menu
    # for matching, show matched segment, and extended 'guess' (with some overlap)
    # Something to show nesting of segments, such as a number of segments in the top bit
    # Find similar segments in other files -- other birds
    # Group files by species

# Rebecca:
    # x colour spectrogram
    # add a marker on the overview to show where you have marked segments, with different colours for unknown, possible
    # x reorder the list dynamically by amount of use -> done, but maybe it should be an option?
    # Maybe include day or night differently in the context menu
    # x have a hot key to add the same bird repeatedly
    # Change the visible window width (or just add) magnify/shrink buttons
    # x Add date, time, person, location, would be good to have weather info calculated automatically (wind, rain -> don't bother), broken sound recorder

    # pull out any bird call (useful for if you don't know what a bird sounds like (Fiji petrel) or wind farm monitoring)
    # do bats!
    # Look up David Bryden (kokako data) looking at male, female, juvenile
    # Get all calls of a species
    # Look up freebird and raven and also BatSearch
# ===============

class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        # Overwrite the axis tick code
        return [QTime().addSecs(value).toString('mm:ss') for value in values]

class ShadedROI(pg.ROI):
    def paint(self, p, opt, widget):
        #brush = QtGui.QBrush(QtGui.QColor(0, 0, 255, 50))
        if not hasattr(self, 'currentBrush'):
            self.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 50)))

        p.save()
        r = self.boundingRect()
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(fn.mkPen(None))
        p.setBrush(self.currentBrush)
        p.translate(r.left(), r.top())
        p.scale(r.width(), r.height())
        p.drawRect(0, 0, 1, 1)
        p.restore()

    def setBrush(self, *br, **kargs):
        """Set the brush that fills the region. Can have any arguments that are valid
        for :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.brush = fn.mkBrush(*br, **kargs)
        self.currentBrush = self.brush

class ShadedRectROI(ShadedROI):
    def __init__(self, pos, size, centered=False, sideScalers=False, **args):
        #QtGui.QGraphicsRectItem.__init__(self, 0, 0, size[0], size[1])
        pg.ROI.__init__(self, pos, size, **args)
        if centered:
            center = [0.5, 0.5]
        else:
            center = [0, 0]

        #self.addTranslateHandle(center)
        self.addScaleHandle([1, 1], center)
        if sideScalers:
            self.addScaleHandle([1, 0.5], [center[0], 0.5])
            self.addScaleHandle([0.5, 1], [0.5, center[1]])

class AviaNZInterface(QMainWindow):
    # Main class for the interface, which contains most of the user interface and plotting code

    def __init__(self,root=None,configfile=None):
        # Main part of the initialisation is loading a configuration file, or creating a new one if it doesn't
        # exist. Also loads an initial file (specified explicitly) and sets up the window.
        # TODO: better way to choose initial file (or don't choose one at all)
        super(AviaNZInterface, self).__init__()
        self.root = root
        if configfile is not None:
            try:
                self.config = json.load(open(configfile))
                self.saveConfig = False
            except:
                print("Failed to load config file")
                self.genConfigFile()
                self.saveConfig = True
            self.configfile = configfile
        else:
            self.genConfigFile()
            self.saveConfig=True
            self.configfile = 'AviaNZconfig.txt'

        # The data structures for the segments
        self.listLabels = []
        self.listRectanglesa1 = []
        self.listRectanglesa2 = []
        self.box1id = -1

        self.colourList = ['Grey','Viridis', 'Inferno', 'Plasma', 'Autumn', 'Cool', 'Bone', 'Copper', 'Hot', 'Jet','Thermal','Flame','Yellowy','Bipolar','Spectrum']
        self.lastSpecies = 'None'
        self.resetStorageArrays()

        self.dirName = self.config['dirpath']
        self.previousFile = None
        self.focusRegion = None

        # Make the window and associated widgets
        QMainWindow.__init__(self, root)
        self.setWindowTitle('AviaNZ')

        # Make life easier for now: preload a birdsong
        self.firstFile = 'tril1.wav' #'male1.wav' # 'kiwi.wav'#'
        #self.firstFile = 'kiwi.wav'

        self.createMenu()
        self.createFrame()

        # Some safety checking for paths and files
        if not os.path.isdir(self.dirName):
            print("Directory doesn't exist: making it")
            os.makedirs(self.dirName)
        if not os.path.isfile(self.dirName+'/'+self.firstFile):
            fileName = QtGui.QFileDialog.getOpenFileName(self, 'Choose File', self.dirName, "Wav files (*.wav)")
            if fileName:
                self.firstFile = fileName
        self.loadFile(self.firstFile)

        # Save the segments every minute
        self.timer = QTimer()
        QObject.connect(self.timer, SIGNAL("timeout()"), self.saveSegments)
        self.timer.start(self.config['secsSave']*1000)

    def createMenu(self):
        # Create the menu entries at the top of the screen.

        fileMenu = self.menuBar().addMenu("&File")
        fileMenu.addAction("&Open sound file", self.openFile, "Ctrl+O")
        fileMenu.addSeparator()
        fileMenu.addAction("&Delete all segments", self.deleteAll, "Ctrl+D")
        fileMenu.addAction("Quit",self.quit,"Ctrl+Q")
        fileMenu.addAction("&Quit",self.quit,"Ctrl+Q")

        specMenu = self.menuBar().addMenu("&Interface")

        self.useAmplitudeTick = specMenu.addAction("Show amplitude plot", self.useAmplitudeCheck)
        self.useAmplitudeTick.setCheckable(True)
        self.useAmplitudeTick.setChecked(True)
        self.useAmplitude = True

        self.useFilesTick = specMenu.addAction("Show list of files", self.useFilesCheck)
        self.useFilesTick.setCheckable(True)
        self.useFilesTick.setChecked(True)
        self.useFiles = True

        self.dragRectangles = specMenu.addAction("Drag boxes in spectrogram", self.dragRectanglesCheck)
        self.dragRectangles.setCheckable(True)
        self.dragRectangles.setChecked(False)

        self.showFundamental = specMenu.addAction("Show fundamental frequency", self.showFundamentalFreq)
        self.showFundamental.setCheckable(True)
        self.showFundamental.setChecked(False)

        colMenu = specMenu.addMenu("&Choose colour map")
        colGroup = QActionGroup(self)
        for colour in self.colourList:
            cm = colMenu.addAction(colour)
            cm.setCheckable(True)
            if colour==self.colourList[0]:
                cm.setChecked(True)
            receiver = lambda cmap=colour: self.setColourMap(cmap)
            self.connect(cm, SIGNAL("triggered()"), receiver)
            colGroup.addAction(cm)
        specMenu.addAction("Invert colour map",self.invertColourMap)
        self.cmapInverted = False

        specMenu.addSeparator()
        specMenu.addAction("Change spectrogram parameters",self.showSpectrogramDialog)

        actionMenu = self.menuBar().addMenu("&Actions")
        actionMenu.addAction("Denoise",self.denoiseDialog)
        actionMenu.addAction("Segment",self.segmentationDialog)
        actionMenu.addAction("Find matches",self.findMatches)
        actionMenu.addAction("Put docks back",self.dockReplace)
        actionMenu.addSeparator()
        actionMenu.addAction("Check segments 1",self.humanClassifyDialog1)
        actionMenu.addAction("Check segments 2",self.humanClassifyDialog2)

        helpMenu = self.menuBar().addMenu("&Help")
        #aboutAction = QAction("About")
        helpMenu.addAction("About",self.showAbout)
        helpMenu.addAction("Help",self.showHelp)

        #quitAction = QAction("&Quit", self)
        #self.connect(quitAction, SIGNAL("triggered()"), self.quit)
        #self.fileMenu.addAction(quitAction)

    def showAbout(self):
        return

    def showHelp(self):
        return

    def genConfigFile(self):
        # Generates a configuration file with default values for parameters
        # These are quite hard to change currently (edit text file, or delete the file and make another)
        # TODO: enable parameter changing
        print("Generating new config file")
        self.config = {
            # Params for spectrogram
            'window_width': 256,
            'incr': 128,

            # Params for denoising
            'maxSearchDepth': 20,

            # Params for segmentation
            'minSegment': 50,
            'dirpath': './Sound Files',
            'secsSave': 60,

            # Param for width in seconds of the main representation
            'windowWidth': 10.0,

            # These are the contrast parameters for the spectrogram
            #'colourStart': 0.25,
            #'colourEnd': 0.75,
            'brightness': 50,
            'contrast': 50,
            'coloursInverted': False,

            # Params for cross-correlation and related
            'corrThr': 0.4,
            # Amount of overlap for 2 segments to be counted as the same
            'overlap_allowed': 5,

            'BirdButtons1': ["Bellbird", "Bittern", "Cuckoo", "Fantail", "Hihi", "Kakapo", "Kereru", "Kiwi (F)", "Kiwi (M)",
                             "Petrel"],
            'BirdButtons2': ["Rifleman", "Ruru", "Saddleback", "Silvereye", "Tomtit", "Tui", "Warbler", "Not Bird",
                             "Don't Know", "Other"],
            'ListBirdsEntries': ['Albatross', 'Avocet', 'Blackbird', 'Bunting', 'Chaffinch', 'Egret', 'Gannet', 'Godwit',
                                 'Gull', 'Kahu', 'Kaka', 'Kea', 'Kingfisher', 'Kokako', 'Lark', 'Magpie', 'Plover',
                                 'Pukeko', "Rooster" 'Rook', 'Thrush', 'Warbler', 'Whio'],
            'BirdList': ["Bellbird", "Bittern", "Cuckoo", "Fantail", "Hihi", "Kakapo", "Kereru", "Kiwi (F)", "Kiwi (M)","Petrel","Rifleman", "Ruru", "Saddleback", "Silvereye", "Tomtit", "Tui", "Warbler", "Not Bird", "Don't Know",'Albatross', 'Avocet', 'Blackbird', 'Bunting', 'Chaffinch', 'Egret', 'Gannet', 'Godwit','Gull', 'Kahu', 'Kaka', 'Kea', 'Kingfisher', 'Kokako', 'Lark', 'Magpie', 'Plover','Pukeko', "Rooster" 'Rook', 'Thrush', 'Warbler', 'Whio'],

            # The colours for the segment boxes
            'ColourNone': (0, 0, 225, 50), # Blue
            'ColourSelected': (0, 225, 0, 50), # Green
            'ColourNamed': (225, 0, 0, 50) # Red
        }

    def createFrame(self):
        # This creates the actual interface. Some Qt, some PyQtGraph

        # Make the window and set its size
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.resize(1200,950)
        self.move(100,50)

        # Make the docks
        self.d_overview = Dock("Overview",size = (1200,120))
        self.d_ampl = Dock("Amplitude",size=(1200,150))
        self.d_spec = Dock("Spectrogram",size=(1200,400))
        self.d_controls = Dock("Controls",size=(800,150))
        self.d_files = Dock("Files",size=(400,250))
        #self.d_buttons = Dock("Buttons",size=(800,100))

        self.area.addDock(self.d_overview,'top')
        self.area.addDock(self.d_ampl,'bottom',self.d_overview)
        self.area.addDock(self.d_spec,'bottom',self.d_ampl)
        self.area.addDock(self.d_controls,'bottom',self.d_spec)
        self.area.addDock(self.d_files,'left',self.d_controls)
        #self.area.addDock(self.d_buttons,'bottom',self.d_controls)

        # Put content widgets in the docks
        self.w_overview = pg.LayoutWidget()
        self.d_overview.addWidget(self.w_overview)
        self.w_overview1 = pg.GraphicsLayoutWidget()
        self.w_overview.addWidget(self.w_overview1)
        self.p_overview = self.w_overview1.addViewBox(enableMouse=False,enableMenu=False,row=0,col=0)
        self.p_overview2 = self.w_overview1.addViewBox(enableMouse=False, enableMenu=False, row=1, col=0)

        self.w_ampl = pg.GraphicsLayoutWidget()
        self.p_ampl = self.w_ampl.addViewBox(enableMouse=False,enableMenu=False)
        self.w_ampl.addItem(self.p_ampl,row=0,col=1)
        self.d_ampl.addWidget(self.w_ampl)

        # The axes
        self.timeaxis = TimeAxis(orientation='bottom')
        self.timeaxis.linkToView(self.p_ampl)
        self.timeaxis.setLabel('Time',units='mm:ss')

        self.ampaxis = pg.AxisItem(orientation='left')
        self.w_ampl.addItem(self.ampaxis,row=0,col=0)
        self.ampaxis.linkToView(self.p_ampl)

        self.w_spec = pg.GraphicsLayoutWidget()
        self.p_spec = CustomViewBox(enableMouse=False,enableMenu=False)
        self.w_spec.addItem(self.p_spec,row=0,col=1)
        self.w_spec.addItem(self.timeaxis,row=1,col=1)

        #self.plot_spec = pg.PlotItem(enableMouse=False,enableMenu=False)
        #self.w_spec.addItem(self.plot_spec,row=0,col=1)
        #self.annot_spec = pg.ScatterPlotItem(size=10,pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255),enableMouse=False,enableMenu=False)
        #self.annot_spec.addPoints([0,0.1,20,30],[0,0.1,20,30])
        #self.plot_spec.addItem(self.annot_spec)
        self.d_spec.addWidget(self.w_spec)
        #proxy = pg.SignalProxy(self.p_spec.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.p_spec.scene().sigMouseMoved.connect(self.mouseMoved)

        self.pointData = pg.TextItem(color=(255,0,0),anchor=(0,0))
        self.p_spec.addItem(self.pointData)

        self.specaxis = pg.AxisItem(orientation='left')
        self.w_spec.addItem(self.specaxis,row=0,col=0)
        self.specaxis.linkToView(self.p_spec)

        self.overviewImage = pg.ImageItem(enableMouse=False)
        self.p_overview.addItem(self.overviewImage)

        #self.amplPlot = pg.PlotItem()
        self.amplPlot = pg.PlotDataItem()

        self.p_ampl.addItem(self.amplPlot)

        self.specPlot = pg.ImageItem()
        self.p_spec.addItem(self.specPlot)

        # Connect up the listeners
        #self.p_overview.scene().sigMouseClicked.connect(self.mouseClicked_overview)
        self.p_ampl.scene().sigMouseClicked.connect(self.mouseClicked_ampl)
        self.p_spec.scene().sigMouseClicked.connect(self.mouseClicked_spec)
        self.p_spec.sigMouseDragged.connect(self.mouseDragged_spec)

        self.w_controls = pg.LayoutWidget()
        self.d_controls.addWidget(self.w_controls)

        self.w_files = pg.LayoutWidget()
        self.d_files.addWidget(self.w_files)

        #self.w_buttons = pg.LayoutWidget()
        #self.d_buttons.addWidget(self.w_buttons)

        # # The buttons to move through the overview
        self.leftBtn = QToolButton()
        self.leftBtn.setArrowType(Qt.LeftArrow)
        self.connect(self.leftBtn, SIGNAL('clicked()'), self.moveLeft)
        self.w_overview.addWidget(self.leftBtn)
        self.rightBtn = QToolButton()
        self.rightBtn.setArrowType(Qt.RightArrow)
        self.connect(self.rightBtn, SIGNAL('clicked()'), self.moveRight)
        self.w_overview.addWidget(self.rightBtn)

        # The instructions and buttons below figMain
        # playButton = QPushButton(QIcon(":/Resources/play.svg"),"&Play Window")

        self.playButton = QtGui.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        #self.playButton = QPushButton("&Play")
        self.connect(self.playButton, SIGNAL('clicked()'), self.playSegment)
        self.timePlayed = QLabel()
        #self.resetButton = QPushButton("&Reset")
        #self.connect(self.resetButton, SIGNAL('clicked()'), self.resetSegment)

        self.playSegButton = QtGui.QToolButton()
        self.playSegButton.setIcon(QtGui.QIcon('img/playsegment.png'))
        self.playSegButton.setIconSize(QtCore.QSize(20, 20))
        self.connect(self.playSegButton, SIGNAL('clicked()'), self.playSelectedSegment)
        self.playSegButton.setEnabled(False)

        self.playBandLimitedSegButton = QtGui.QToolButton()
        self.playBandLimitedSegButton.setIcon(QtGui.QIcon('img/playsegment.png'))
        self.playBandLimitedSegButton.setIconSize(QtCore.QSize(20, 20))
        self.connect(self.playBandLimitedSegButton, SIGNAL('clicked()'), self.playBandLimitedSegment)
        self.playBandLimitedSegButton.setEnabled(False)

        # Checkbox for whether or not user is drawing boxes around song in the spectrogram (defaults to clicks not drags)
        #self.dragRectangles = QCheckBox('Drag boxes in spectrogram')
        #self.dragRectangles.stateChanged[int].connect(self.dragRectanglesCheck)
        #self.useAmplitudeTick = QCheckBox('Show amplitude plot')
        #self.useAmplitudeTick.stateChanged[int].connect(self.useAmplitudeCheck)
        #self.useAmplitudeTick.setChecked(True)
        #self.showFundamental = QCheckBox('Show fundamental frequency')
        #self.showFundamental.stateChanged[int].connect(self.showFundamentalFreq)
        #self.showFundamental.setChecked(False)

        # A slider to show playback position
        # This is hidden, but controls the moving bar
        self.playSlider = QSlider(Qt.Horizontal)
        self.connect(self.playSlider,SIGNAL('sliderReleased()'),self.sliderMoved)
        self.playSlider.setVisible(False)
        self.d_spec.addWidget(self.playSlider)
        #self.w_controls.addWidget(self.playSlider,row=1,col=0,colspan=4)

        # A slider to move through the file easily
        # TODO: Why doesn't this have the forward/backward arrows?
        self.scrollSlider = QScrollBar(Qt.Horizontal)
        self.scrollSlider.valueChanged.connect(self.scroll)
        self.d_spec.addWidget(self.scrollSlider)

        self.w_controls.addWidget(QLabel('Slide top box to move through recording, click to start and end a segment, click on segment to edit or label. Right click to interleave.'),row=0,col=0,colspan=3)
        self.w_controls.addWidget(self.playButton,row=1,col=0)
        #self.w_controls.addWidget(self.playSegButton,row=1,col=1)
        self.w_controls.addWidget(self.playBandLimitedSegButton,row=1,col=1)
        self.w_controls.addWidget(self.timePlayed,row=1,col=2)
        #self.w_controls.addWidget(self.resetButton,row=2,col=1)
        #self.w_controls.addWidget(self.dragRectangles,row=0,col=3)
        #self.w_controls.addWidget(self.useAmplitudeTick,row=1,col=3)
        #self.w_controls.addWidget(self.showFundamental,row=0,col=4)

        # The spinbox for changing the width shown in figMain
        self.widthWindow = QDoubleSpinBox()
        self.widthWindow.setSingleStep(1.0)
        self.widthWindow.setDecimals(2)
        self.widthWindow.setValue(self.config['windowWidth'])
        self.w_controls.addWidget(QLabel('Visible window width (seconds)'),row=2,col=3)
        self.w_controls.addWidget(self.widthWindow,row=3,col=3)#,colspan=2)
        self.widthWindow.valueChanged[float].connect(self.changeWidth)

        # Brightness, contrast and colour reverse options
        self.brightnessSlider = QSlider(Qt.Horizontal)
        self.brightnessSlider.setMinimum(0)
        self.brightnessSlider.setMaximum(100)
        self.brightnessSlider.setValue(self.config['brightness'])
        self.brightnessSlider.setTickInterval(1)
        self.brightnessSlider.valueChanged.connect(self.colourChange)

        self.contrastSlider = QSlider(Qt.Horizontal)
        self.contrastSlider.setMinimum(0)
        self.contrastSlider.setMaximum(100)
        self.contrastSlider.setValue(self.config['contrast'])
        self.contrastSlider.setTickInterval(1)
        self.contrastSlider.valueChanged.connect(self.colourChange)

        deleteButton = QPushButton("&Delete Current Segment")
        self.connect(deleteButton, SIGNAL('clicked()'), self.deleteSegment)
        #self.swapBW = QCheckBox()
        #self.swapBW.setChecked(self.config['coloursInverted'])
        #self.swapBW.stateChanged.connect(self.colourChange)
        #self.swapBW.stateChanged.connect(self.swappedBW)

        #self.w_controls.addWidget(QLabel("Swap B/W"),row=2,col=0)
        #self.w_controls.addWidget(self.swapBW,row=3,col=0)
        self.w_controls.addWidget(QLabel("Brightness"),row=2,col=0)
        self.w_controls.addWidget(self.brightnessSlider,row=3,col=0)
        self.w_controls.addWidget(QLabel("Contrast"),row=2,col=1)
        self.w_controls.addWidget(self.contrastSlider,row=3,col=1)
        self.w_controls.addWidget(deleteButton,row=4,col=1)


        # List to hold the list of files
        self.listFiles = QListWidget(self)
        self.listFiles.setFixedWidth(150)
        self.fillFileList()
        self.listFiles.connect(self.listFiles, SIGNAL('itemDoubleClicked(QListWidgetItem*)'), self.listLoadFile)

        self.w_files.addWidget(QLabel('Double click to select'),row=0,col=0)
        self.w_files.addWidget(QLabel('Red names have segments'),row=1,col=0)
        self.w_files.addWidget(self.listFiles,row=2,col=0)

        # These are the main buttons, on the bottom right
        #quitButton = QPushButton("&Quit")
        #self.connect(quitButton, SIGNAL('clicked()'), self.quit)
        #spectrogramButton = QPushButton("Spectrogram &Params")
        #self.connect(spectrogramButton, SIGNAL('clicked()'), self.showSpectrogramDialog)
        #segmentButton = QPushButton("&Segment")
        #self.connect(segmentButton, SIGNAL('clicked()'), self.segmentationDialog)
        #denoiseButton = QPushButton("&Denoise")
        #self.connect(denoiseButton, SIGNAL('clicked()'), self.denoiseDialog)
        #recogniseButton = QPushButton("&Recognise")
        #self.connect(recogniseButton, SIGNAL('clicked()'), self.recognise)
        #deleteAllButton = QPushButton("&Delete All Segments")
        #self.connect(deleteAllButton, SIGNAL('clicked()'), self.deleteAll)
        #findMatchButton = QPushButton("&Find Matches")
        #self.connect(findMatchButton, SIGNAL('clicked()'), self.findMatches)
        #checkButton1 = QPushButton("&Check Segments 1")
        #self.connect(checkButton1, SIGNAL('clicked()'), self.humanClassifyDialog1)
        #checkButton2 = QPushButton("&Check Segments 2")
        #self.connect(checkButton2, SIGNAL('clicked()'), self.humanClassifyDialog2)
        #dockButton = QPushButton("&Put Docks Back")
        #self.connect(dockButton, SIGNAL('clicked()'), self.dockReplace)
        #loadButton = QPushButton("&Load File")
        #self.connect(loadButton, SIGNAL('clicked()'), self.openFile)
        #playSegButton = QPushButton("&Play Segment")
        #self.connect(playSegButton, SIGNAL('clicked()'), self.playSelectedSegment)

        #self.w_buttons.addWidget(loadButton,row=0,col=0)
        #self.w_buttons.addWidget(deleteAllButton,row=0,col=4)
        #self.w_buttons.addWidget(denoiseButton,row=0,col=2)
        #self.w_buttons.addWidget(spectrogramButton,row=0,col=3)
        #self.w_buttons.addWidget(checkButton1,row=1,col=4)
        #self.w_buttons.addWidget(checkButton2,row=1,col=5)

        #self.w_buttons.addWidget(segmentButton,row=1,col=0)
        #self.w_buttons.addWidget(findMatchButton,row=1,col=1)
        #self.w_buttons.addWidget(dockButton,row=1,col=2)
        #self.w_buttons.addWidget(quitButton,row=1,col=3)
        #for w in [deleteButton, deleteAllButton, spectrogramButton, segmentButton, findMatchButton,
        #              checkButton1, checkButton2, dockButton, quitButton]:

        # The context menu (drops down on mouse click) to select birds
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.menuBirdList = QMenu()
        self.menuBird2 = self.menuBirdList.addMenu('Other')
        self.fillBirdList()

        # # An array of radio buttons and a list and a text entry box
        # # Create an array of radio buttons for the most common birds (2 columns of 10 choices)
        # # self.birds1 = []
        # # for item in self.config['BirdButtons1']:
        # #     self.birds1.append(QRadioButton(item))
        # # self.birds2 = []
        # # for item in self.config['BirdButtons2']:
        # #     self.birds2.append(QRadioButton(item))
        # #
        # # for i in xrange(len(self.birds1)):
        # #     self.birds1[i].setEnabled(False)
        # #     self.connect(self.birds1[i], SIGNAL("clicked()"), self.radioBirdsClicked)
        # # for i in xrange(len(self.birds2)):
        # #     self.birds2[i].setEnabled(False)
        # #     self.connect(self.birds2[i], SIGNAL("clicked()"), self.radioBirdsClicked)
        #
        # # The list of less common birds
        # # self.birdList = QListWidget(self)
        # # self.birdList.setMaximumWidth(150)
        # # for item in self.config['ListBirdsEntries']:
        # #     self.birdList.addItem(item)
        # # self.birdList.sortItems()
        # # # Explicitly add "Other" option in
        # # self.birdList.insertItem(0,'Other')
        #
        # # self.connect(self.birdList, SIGNAL("itemClicked(QListWidgetItem*)"), self.listBirdsClicked)
        # # self.birdList.setEnabled(False)
        #
        # # This is the text box for missing birds
        # # self.tbox = QLineEdit(self)
        # # self.tbox.setMaximumWidth(150)
        # # self.connect(self.tbox, SIGNAL('editingFinished()'), self.birdTextEntered)
        # # self.tbox.setEnabled(False)
        #
        # # birds1Layout = QVBoxLayout()
        # # for i in xrange(len(self.birds1)):
        # #     birds1Layout.addWidget(self.birds1[i])
        # #
        # # birds2Layout = QVBoxLayout()
        # # for i in xrange(len(self.birds2)):
        # #     birds2Layout.addWidget(self.birds2[i])
        # #
        # # birdListLayout = QVBoxLayout()
        # # birdListLayout.addWidget(self.birdList)
        # # birdListLayout.addWidget(QLabel("If bird isn't in list, select Other"))
        # # birdListLayout.addWidget(QLabel("Type below, Return at end"))
        # # birdListLayout.addWidget(self.tbox)

        # Instantiate a Qt media object and prepare it (for audio playback)
        self.media_obj = phonon.Phonon.MediaObject(self)
        self.audio_output = phonon.Phonon.AudioOutput(phonon.Phonon.MusicCategory, self)
        phonon.Phonon.createPath(self.media_obj, self.audio_output)
        self.media_obj.setTickInterval(20)
        #self.media_obj.tick.connect(self.movePlaySlider)
        self.media_obj.finished.connect(self.playFinished)
        # TODO: Check the next line out!
        #self.media_obj.totalTimeChanged.connect(self.setSliderLimits)

        # Make the colours
        self.ColourSelected = QtGui.QBrush(QtGui.QColor(self.config['ColourSelected'][0], self.config['ColourSelected'][1], self.config['ColourSelected'][2], self.config['ColourSelected'][3]))
        self.ColourNamed = QtGui.QBrush(QtGui.QColor(self.config['ColourNamed'][0], self.config['ColourNamed'][1], self.config['ColourNamed'][2], self.config['ColourNamed'][3]))
        self.ColourNone = QtGui.QBrush(QtGui.QColor(self.config['ColourNone'][0], self.config['ColourNone'][1], self.config['ColourNone'][2], self.config['ColourNone'][3]))

        # Hack to get the type of an ROI
        p_spec_r = ShadedRectROI(0, 0)
        self.ROItype = type(p_spec_r)

        # Listener for key presses
        self.connect(self.p_spec, SIGNAL("keyPressed"),self.handleKey)
        # Store the state of the docks
        self.state = self.area.saveState()

        # Plot everything
        self.show()

    def handleKey(self,ev):
        if ev.key() == Qt.Key_Backspace:
            self.deleteSegment()

    def fillBirdList(self):
        # Need the lambda function to connect all menu events to same trigger and know which was selected
        # TODO: Work out how to make the menu item be selectable when it has 'Possible' with it
        self.menuBirdList.clear()
        self.menuBird2.clear()
        for item in self.config['BirdList'][:20]:
            bird = self.menuBirdList.addAction(item)
            #birdp = self.menuBirdList.addMenu(item)
            #birdp.addAction('Possible '+item)
            receiver = lambda birdname=item: self.birdSelected(birdname)
            self.connect(bird, SIGNAL("triggered()"), receiver)
            self.menuBirdList.addAction(bird)
        self.menuBird2 = self.menuBirdList.addMenu('Other')
        for item in self.config['BirdList'][20:]+['Other']:
            bird = self.menuBird2.addAction(item)
            receiver = lambda birdname=item: self.birdSelected(birdname)
            self.connect(bird, SIGNAL("triggered()"), receiver)
            self.menuBird2.addAction(bird)

    def fillFileList(self):
        # Generates the list of files for the listbox on lower left
        # Most of the work is to deal with directories in that list
        if not os.path.isdir(self.dirName):
            print("Directory doesn't exist: making it")
            os.makedirs(self.dirName)
        self.listOfFiles = QDir(self.dirName).entryInfoList(['..','*.wav'],filters=QDir.AllDirs|QDir.NoDot|QDir.Files,sort=QDir.DirsFirst)
        listOfDataFiles = QDir(self.dirName).entryList(['*.data'])
        listOfLongFiles = QDir(self.dirName).entryList(['*_1.wav'])
        for file in self.listOfFiles:
            if file.fileName()[:-4]+'_1.wav' in listOfLongFiles:
                # Ignore this entry
                pass
            else:
                # If there is a .data version, colour the name red to show it has been labelled
                item = QListWidgetItem(self.listFiles)
                item.setText(file.fileName())
                if file.fileName()+'.data' in listOfDataFiles:
                    item.setTextColor(Qt.red)
        index = self.listFiles.findItems(self.firstFile,Qt.MatchExactly)
        if len(index)>0:
            self.listFiles.setCurrentItem(index[0])
        else:
            index = self.listFiles.findItems(self.listOfFiles[0].fileName(),Qt.MatchExactly)
            self.listFiles.setCurrentItem(index[0])

    def resetStorageArrays(self):
        # These variables hold the data to be saved and/or plotted
        # Used when new files are loaded

        # Remove the segments
        self.deleteAll(True)
        if hasattr(self, 'overviewImageRegion'):
            self.p_overview.removeItem(self.overviewImageRegion)

        # This is a flag to say if the next thing that they click on should be a start or a stop for segmentation
        self.started = False

        # Keep track of start points and selected buttons
        #self.start_a = 0
        self.windowStart = 0
        self.playPosition = self.windowStart
        self.prevBoxCol = self.config['ColourNone']
        #self.line = None

        #self.recta1 = None
        #self.recta2 = None
        #self.focusRegionSelected = False
        #self.figMainSegment1 = None
        #self.figMainSegment2 = None
        #self.figMainSegmenting = False

        #self.playbar1 = None
        #self.isPlaying = False

    def openFile(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, 'Choose File', self.dirName,"Wav files (*.wav)")

        # Find the '/' in the fileName
        i=len(fileName)-1
        while fileName[i] != '/' and i>0:
            i = i-1
        self.dirName = fileName[:i+1]
        if self.previousFile is not None:
            if self.segments != [] or self.hasSegments:
                self.saveSegments()
        self.previousFile = fileName
        self.resetStorageArrays()
        self.loadFile(fileName)

    def listLoadFile(self,current):
        # Listener for when the user clicks on a filename
        # Saves segments of current file, resets flags and calls loader
        #self.p_ampl.clear()
        if self.previousFile is not None:
            if self.segments != [] or self.hasSegments:
                self.saveSegments()
                self.previousFile.setTextColor(Qt.red)
        self.previousFile = current
        self.resetStorageArrays()

        i=0
        while self.listOfFiles[i].fileName() != current.text():
            i+=1
        # Slightly dangerous, but the file REALLY should exist
        if self.listOfFiles[i].isDir():
            dir = QDir(self.dirName)
            dir.cd(self.listOfFiles[i].fileName())
            # Now repopulate the listbox
            #print "Now in "+self.listOfFiles[i].fileName()
            self.dirName=dir.absolutePath()
            self.listFiles.clearSelection()
            self.listFiles.clearFocus()
            self.listFiles.clear()
            self.previousFile = None
            self.fillFileList()
        else:
            self.loadFile(current)

    def loadFile(self,name):
        # This does the work of loading a file
        # One magic constant, which normalises the data
        # TODO: Note that if load normalised the data, this buggers up the spectrogram for e.g., Harma

        # Create a modal dialog to get the name of the user and show some file info
        # TODO: do something with the stuff below
        # TODO: at least: add username to metadata for segments
        #name = 'xxx'
        #date = 'xy'
        #time = '12:00'
        #fdd = FileDataDialog(name,date,time)
        #fdd.exec_()
        #username = fdd.getData()

        if isinstance(name,str):
            self.filename = self.dirName+'/'+name
        elif isinstance(name,QString):
            self.filename = name
        else:
            self.filename = self.dirName+'/'+str(name.text())
        #self.audiodata, self.sampleRate = lr.load(self.filename,sr=None)
        self.sampleRate, self.audiodata = wavfile.read(self.filename)
        # None of the following should be necessary for librosa
        if self.audiodata.dtype is not 'float':
            self.audiodata = self.audiodata.astype('float') #/ 32768.0
        if np.shape(np.shape(self.audiodata))[0]>1:
            self.audiodata = self.audiodata[:,0]
        self.datalength = np.shape(self.audiodata)[0]
        self.setWindowTitle('AviaNZ - ' + self.filename)
        print("Length of file is ",len(self.audiodata),float(self.datalength)/self.sampleRate)

        # Create an instance of the Signal Processing class
        if not hasattr(self,'sp'):
            self.sp = SignalProc.SignalProc(self.audiodata, self.sampleRate,self.config['window_width'],self.config['incr'])

        # Get the data for the spectrogram
        self.sgRaw = self.sp.spectrogram(self.audiodata,self.sampleRate,multitaper=False)
        self.sg = np.abs(np.where(self.sgRaw==0,0.0,10.0 * np.log10(self.sgRaw)))

        # Colour scaling for the spectrograms
        # TODO: Sort this so that doesn't necessarily reinitialise
        #self.colourStart = self.config['colourStart'] * (maxsg-minsg)
        #self.colourEnd = self.config['colourEnd'] * (maxsg-minsg)

        # Load any previous segments stored
        if os.path.isfile(self.filename+'.data'):
            file = open(self.filename+'.data', 'r')
            self.segments = json.load(file)
            file.close()
            self.hasSegments = True
        else:
            self.hasSegments = False

        # Update the data that is seen by the other classes
        # TODO: keep an eye on this to add other classes as required
        if hasattr(self,'seg'):
            self.seg.setNewData(self.audiodata,self.sgRaw,self.sampleRate)
        self.sp.setNewData(self.audiodata,self.sampleRate)

        # Delete any denoising backups from the previous one
        if hasattr(self,'audiodata_backup'):
            self.audiodata_backup = None

        # Set the values for the segmentation thresholds
        # self.ampThr.setRange(0.001,np.max(self.audiodata)+0.001)
        # self.ampThr.setSingleStep(0.001)
        # self.ampThr.setDecimals(4)
        # self.ampThr.setValue(np.max(self.audiodata)+0.001)

        # Set the window size
        self.windowSize = self.config['windowWidth']
        self.widthWindow.setRange(0.5, float(len(self.audiodata))/self.sampleRate)

        # Reset it if the file is shorter than the window
        if float(len(self.audiodata))/self.sampleRate < self.windowSize:
            self.windowSize = float(len(self.audiodata))/self.sampleRate
        self.widthWindow.setValue(self.windowSize)

        # Load the file for playback as well, and connect up the listeners for it
        self.media_obj.setCurrentSource(phonon.Phonon.MediaSource(self.filename))
        self.totalTime = self.convertMillisecs(self.media_obj.totalTime())
        print self.media_obj.totalTime(), self.totalTime
        self.media_obj.tick.connect(self.movePlaySlider)

        # Set the length of the scrollbar
        self.scrollSlider.setRange(0,np.shape(self.sg)[1]-self.convertAmpltoSpec(self.widthWindow.value()))
        self.scrollSlider.setValue(0)

        # Get the height of the amplitude for plotting the box
        self.minampl = np.min(self.audiodata)+0.1*(np.max(self.audiodata)+np.abs(np.min(self.audiodata)))
        #self.plotheight = np.abs(self.minampl) + np.max(self.audiodata)
        self.drawOverview()
        self.drawfigMain()

    # def openFile(self):
    #     # If have an open file option this will deal with it via a file dialog
    #     # Currently unused
    #     Formats = "Wav file (*.wav)"
    #     filename = QFileDialog.getOpenFileName(self, 'Open File', '/Users/srmarsla/Projects/AviaNZ', Formats)
    #     if filename != None:
    #         self.loadFile(filename)

    def dragRectanglesCheck(self):
        # The checkbox that says if the user is dragging rectangles or clicking on the spectrogram has changed state
        if self.dragRectangles.isChecked():
            #print "Checked"
            self.p_spec.setMouseMode(pg.ViewBox.RectMode)
        else:
            #print "Unchecked"
            self.p_spec.setMouseMode(pg.ViewBox.PanMode)

    def useAmplitudeCheck(self):
        # Note that this doesn't remove the dock, just hide it. So it's all still live and easy to replace :)
        # Also move all the labels
        if self.useAmplitudeTick.isChecked():
            #if hasattr(self,'useAmplitude'):
            #    self.w_spec.removeItem(self.timeaxis)
            self.useAmplitude = True
            #self.w_spec.addItem(self.timeaxis, row=1, col=1)
            #self.timeaxis.linkToView(self.p_ampl)
            for r in self.listLabels:
                self.p_spec.removeItem(r)
                self.p_ampl.addItem(r)
                r.setPos(self.convertSpectoAmpl(r.x()),self.minampl)
            self.d_ampl.show()
        else:
            self.useAmplitude = False
            #self.w_ampl.removeItem(self.timeaxis)
            #self.w_spec.addItem(self.timeaxis, row=1, col=1)
            for r in self.listLabels:
                self.p_ampl.removeItem(r)
                self.p_spec.addItem(r)
                r.setPos(self.convertAmpltoSpec(r.x()),0.1)
            self.d_ampl.hide()

    def useFilesCheck(self):
        if self.useFilesTick.isChecked():
            self.d_files.show()
        else:
            self.d_files.hide()

    def showFundamentalFreq(self):
        # Draw the fundamental frequency
        if self.showFundamental.isChecked():
            if not hasattr(self,'seg'):
                self.seg = Segment.Segment(self.audiodata,self.sgRaw,self.sp,self.sampleRate,self.config['minSegment'])
            pitch, y, minfreq, W = self.seg.yin()
            ind = np.squeeze(np.where(pitch>minfreq))
            pitch = pitch[ind]
            ind = ind*W/self.config['window_width']
            x = (pitch*2./self.sampleRate*np.shape(self.sg)[0]).astype('int')
            self.yinRois = []
            for r in range(len(x)):
                self.yinRois.append(pg.CircleROI([ind[r],x[r]], [2,2], pen=(4, 9),movable=False))
            for r in self.yinRois:
                self.p_spec.addItem(r)

            # TODO: Fit a spline and draw it
            #from scipy.interpolate import interp1d
            #f = interp1d(x, ind, kind='cubic')
            #self.sg[x,ind] = 1
        else:
            for r in self.yinRois:
                self.p_spec.removeItem(r)
        #self.specPlot.setImage(np.fliplr(self.sg.T))

    # ==============
# Code for drawing and using the main figure

    def convertAmpltoSpec(self,x):
        return x*self.sampleRate/self.config['incr']

    def convertSpectoAmpl(self,x):
        return x*self.config['incr']/self.sampleRate

    def convertMillisecs(self,millisecs):
        seconds = (millisecs / 1000) % 60
        minutes = (millisecs / (1000 * 60)) % 60
        return "%02d" % minutes+":"+"%02d" % seconds

    def drawOverview(self):
        # On loading a new file, update the overview figure to show where you are up to in the file
        self.overviewImage.setImage(np.fliplr(self.sg.T))
        self.overviewImageRegion = pg.LinearRegionItem()
        self.p_overview.addItem(self.overviewImageRegion, ignoreBounds=True)
        self.overviewImageRegion.setRegion([0, self.convertAmpltoSpec(self.widthWindow.value())])
        self.overviewImageRegion.sigRegionChanged.connect(self.updateOverview)
        #self.overviewImageRegion.sigRegionChangeFinished.connect(self.updateOverview)

        # Make the rectangle that summarises segments
        self.rect1 = pg.QtGui.QGraphicsRectItem(0,0,np.shape(self.sg)[1],0.5)
        self.rect1.setPen(pg.mkPen(None))
        self.rect1.setBrush(pg.mkBrush('r'))
        self.p_overview2.addItem(self.rect1)

    def updateOverview(self):
        # Listener for when the overview box is changed
        # Does the work of keeping all the plots in the right range
        # TODO: Why does this update the other plots so slowly?
        minX, maxX = self.overviewImageRegion.getRegion()
        #print "updating overview", minX, maxX, self.convertSpectoAmpl(maxX)
        self.widthWindow.setValue(self.convertSpectoAmpl(maxX-minX))
        self.p_ampl.setXRange(self.convertSpectoAmpl(minX), self.convertSpectoAmpl(maxX), padding=0)
        self.p_spec.setXRange(minX, maxX, padding=0)
        self.p_ampl.setXRange(self.convertSpectoAmpl(minX), self.convertSpectoAmpl(maxX), padding=0)
        self.p_spec.setXRange(minX, maxX, padding=0)
        #print "Slider:", self.convertSpectoAmpl(minX),self.convertSpectoAmpl(maxX)
        self.setSliderLimits(1000.0*self.convertSpectoAmpl(minX),1000.0*self.convertSpectoAmpl(maxX))
        self.scrollSlider.setValue(minX)
        self.pointData.setPos(minX,0)
        pg.QtGui.QApplication.processEvents()

    def drawfigMain(self):
        # This draws the main amplitude and spectrogram plots

        self.amplPlot.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=self.datalength,endpoint=True),self.audiodata)

        self.specPlot.setImage(np.fliplr(self.sg.T))
        # The constants here are divide by 1000 to get kHz, and then remember the top is sampleRate/2
        self.specaxis.setTicks([[(0,0),(np.shape(self.sg)[0]/4,self.sampleRate/8000),(np.shape(self.sg)[0]/2,self.sampleRate/4000),(3*np.shape(self.sg)[0]/4,3*self.sampleRate/8000),(np.shape(self.sg)[0],self.sampleRate/2000)]])
        self.specaxis.setLabel('kHz')
        #self.specaxis.tickSpacing(0,self.sampleRate/2,self.sampleRate/8)
        self.updateOverview()

        self.setColourLevels()

        # If there are segments, show them
        for count in range(len(self.segments)):
            self.addSegment(self.segments[count][0], self.segments[count][1],self.segments[count][2],self.segments[count][3],self.segments[count][4],False)

        # This is the moving bar for the playback
        if not hasattr(self,'bar'):
            self.bar = pg.InfiniteLine(angle=90, movable=True, pen={'color': 'c', 'width': 3})
        self.p_spec.addItem(self.bar, ignoreBounds=True)
        self.bar.sigPositionChangeFinished.connect(self.barMoved)

    def updateRegion_spec(self):
        # This is the listener for when a segment box is changed
        # It updates the position of the matching box, and also the text within it
        sender = self.sender()
        i = 0
        while self.listRectanglesa2[i] != sender and i<len(self.listRectanglesa2):
            i = i+1
        if i>len(self.listRectanglesa2):
            print "segment not found!"
        else:
            if type(sender) == self.ROItype:
                x1 = self.convertSpectoAmpl(sender.pos()[0])
                x2 = self.convertSpectoAmpl(sender.pos()[0]+sender.size()[0])
            else:
                x1 = self.convertSpectoAmpl(sender.getRegion()[0])
                x2 = self.convertSpectoAmpl(sender.getRegion()[1])
            self.listRectanglesa1[i].setRegion([x1,x2])
            self.listLabels[i].setPos(x1,self.minampl)
            self.segments[i][0] = x1
            self.segments[i][1] = x2

    def updateRegion_ampl(self):
        # This is the listener for when a segment box is changed
        # It updates the position of the matching box, and also the text within it
        sender = self.sender()
        i = 0
        while self.listRectanglesa1[i] != sender and i<len(self.listRectanglesa1):
            i = i+1
        if i>len(self.listRectanglesa1):
            print "segment not found!"
        else:
            x1 = self.convertAmpltoSpec(sender.getRegion()[0])
            x2 = self.convertAmpltoSpec(sender.getRegion()[1])

            if type(self.listRectanglesa2[i]) == self.ROItype:
                y1 = self.listRectanglesa2[i].pos().y()
                y2 = self.listRectanglesa2[i].size().y()
                self.listRectanglesa2[i].setPos(pg.Point(x1,y1))
                self.listRectanglesa2[i].setSize(pg.Point(x2-x1,y2))
            else:
                self.listRectanglesa2[i].setRegion([x1,x2])
            self.listLabels[i].setPos(sender.getRegion()[0],self.minampl)
            self.segments[i][0] = sender.getRegion()[0]
            self.segments[i][1] = sender.getRegion()[1]

    # def mouseClicked_overview(self,evt):
    #     pos = evt.pos()
    #     if self.p_overview.sceneBoundingRect().contains(pos):
    #         mousePoint = self.p_overview.mapSceneToView(pos)

    def addSegment(self,startpoint,endpoint,y1=0,y2=0,species=None,saveSeg=True):
        # Create a new segment and all the associated stuff
        # x, y in amplitude coordinates

        # Get the name and colour sorted
        if species is None:
            species = 'None'

        if species != "None" and species != "Don't Know":
            brush = self.ColourNamed
            self.prevBoxCol = brush
        else:
            brush = self.ColourNone
            self.prevBoxCol = brush

        if startpoint > endpoint:
            temp = startpoint
            startpoint = endpoint
            endpoint = temp
        if y1 > y2:
            temp = y1
            y1 = y2
            y2 = temp

        # Add the segments, connect up the listeners
        p_ampl_r = pg.LinearRegionItem(brush=brush)

        self.p_ampl.addItem(p_ampl_r, ignoreBounds=True)
        p_ampl_r.setRegion([startpoint, endpoint])
        p_ampl_r.sigRegionChangeFinished.connect(self.updateRegion_ampl)

        if y1==0 and y2==0:
            p_spec_r = pg.LinearRegionItem(brush = brush)
            #self.p_spec.addItem(p_spec_r, ignoreBounds=True)
            p_spec_r.setRegion([self.convertAmpltoSpec(startpoint), self.convertAmpltoSpec(endpoint)])
        else:
            startpointS = QPointF(self.convertAmpltoSpec(startpoint),y1)
            endpointS = QPointF(self.convertAmpltoSpec(endpoint),y2)
            p_spec_r = ShadedRectROI(startpointS, endpointS - startpointS, pen='r')
            #self.p_spec.addItem(p_spec_r, ignoreBounds=True)
            #p_spec_r.sigRegionChangeFinished.connect(self.updateRegion_spec)
        self.p_spec.addItem(p_spec_r, ignoreBounds=True)
        p_spec_r.sigRegionChangeFinished.connect(self.updateRegion_spec)

        # Put the text into the box
        label = pg.TextItem(text=species, color='k')
        if self.useAmplitude:
            self.p_ampl.addItem(label)
            label.setPos(startpoint, self.minampl)
        else:
            self.p_spec.addItem(label)
            label.setPos(self.convertAmpltoSpec(startpoint), 1)

        # Add the segments to the relevent lists
        self.listRectanglesa1.append(p_ampl_r)
        self.listRectanglesa2.append(p_spec_r)
        self.listLabels.append(label)

        if saveSeg:
            # Add the segment to the data
            self.segments.append([startpoint, endpoint, y1, y2, species])

    def mouseClicked_ampl(self,evt):
        pos = evt.scenePos()
        #print pos.x(), self.p_ampl.mapSceneToView(pos).x()

        if self.box1id>-1:
            self.listRectanglesa1[self.box1id].setBrush(self.prevBoxCol)
            self.listRectanglesa1[self.box1id].update()
            self.listRectanglesa2[self.box1id].setBrush(self.prevBoxCol)
            self.listRectanglesa2[self.box1id].update()

        if self.p_ampl.sceneBoundingRect().contains(pos):
            mousePoint = self.p_ampl.mapSceneToView(pos)

            if self.started:
                # This is the second click, so should pay attention and close the segment
                # Stop the mouse motion connection, remove the drawing boxes
                if self.started_window=='a':
                    self.p_ampl.scene().sigMouseMoved.disconnect()
                    self.p_ampl.removeItem(self.vLine_a)
                else:
                    self.p_spec.scene().sigMouseMoved.disconnect()
                    # Add the other mouse move listener back
                    self.p_spec.scene().sigMouseMoved.connect(self.mouseMoved)
                    self.p_spec.removeItem(self.vLine_s)

                #self.p_ampl.scene().sigMouseMoved.disconnect()
                #self.p_ampl.removeItem(self.vLine_a)
                self.p_ampl.removeItem(self.drawingBox_ampl)
                self.p_spec.removeItem(self.drawingBox_spec)
                # If the user has pressed shift, copy the last species and don't use the context menu
                modifiers = QtGui.QApplication.keyboardModifiers()
                if modifiers == QtCore.Qt.ShiftModifier:
                    print "here"
                    self.addSegment(self.start_location, mousePoint.x(),species=self.lastSpecies)
                else:
                    print "there"
                    self.addSegment(self.start_location,mousePoint.x())
                    # Context menu
                    self.box1id = len(self.segments) - 1
                    self.fillBirdList()
                    self.menuBirdList.popup(QPoint(evt.screenPos().x(), evt.screenPos().y()))

                self.playSegButton.setEnabled(True)
                self.playBandLimitedSegButton.setEnabled(True)

                self.listRectanglesa1[self.box1id].setBrush(fn.mkBrush(self.ColourSelected))
                self.listRectanglesa1[self.box1id].update()
                self.listRectanglesa2[self.box1id].setBrush(fn.mkBrush(self.ColourSelected))
                self.listRectanglesa2[self.box1id].update()

                self.started = not(self.started)
            else:
                # Check if the user has clicked in a box
                # Note: Returns the first one it finds
                box1id = -1
                for count in range(len(self.listRectanglesa1)):
                    x1, x2 = self.listRectanglesa1[count].getRegion()
                    if x1 <= mousePoint.x() and x2 >= mousePoint.x():
                        box1id = count
                    #print box1id, len(self.listRectanglesa1)

                if box1id > -1 and not evt.button() == QtCore.Qt.RightButton:
                    # User clicked in a box (with the left button)
                    # Change colour, store the old colour
                    self.box1id = box1id
                    self.prevBoxCol = self.listRectanglesa1[box1id].brush.color()
                    self.listRectanglesa1[box1id].setBrush(fn.mkBrush(self.ColourSelected))
                    self.listRectanglesa1[box1id].update()
                    self.playSegButton.setEnabled(True)
                    self.playBandLimitedSegButton.setEnabled(True)
                    self.listRectanglesa2[box1id].setBrush(fn.mkBrush(self.ColourSelected))
                    self.listRectanglesa2[box1id].update()

                    self.fillBirdList()
                    self.menuBirdList.popup(QPoint(evt.screenPos().x(), evt.screenPos().y()))
                else:
                    # User hasn't clicked in a box (or used the right button), so start a new segment
                    self.start_location = mousePoint.x()
                    self.vLine_a = pg.InfiniteLine(angle=90, movable=False,pen={'color': 'r', 'width': 3})
                    self.p_ampl.addItem(self.vLine_a, ignoreBounds=True)
                    self.vLine_a.setPos(self.start_location)

                    self.playSegButton.setEnabled(False)
                    self.playBandLimitedSegButton.setEnabled(True)
                    brush = self.ColourNone
                    self.drawingBox_ampl = pg.LinearRegionItem(brush=brush)
                    self.p_ampl.addItem(self.drawingBox_ampl, ignoreBounds=True)
                    self.drawingBox_ampl.setRegion([self.start_location, self.start_location])
                    self.drawingBox_spec = pg.LinearRegionItem(brush=brush)
                    self.p_spec.addItem(self.drawingBox_spec, ignoreBounds=True)
                    self.drawingBox_spec.setRegion([self.convertAmpltoSpec(self.start_location), self.convertAmpltoSpec(self.start_location)])
                    self.p_ampl.scene().sigMouseMoved.connect(self.GrowBox_ampl)
                    self.started_window = 'a'

                    self.started = not (self.started)

    def mouseClicked_spec(self,evt):

        pos = evt.scenePos()
        #print pos, self.p_spec.mapSceneToView(pos)

        if self.box1id>-1:
            self.listRectanglesa1[self.box1id].setBrush(self.prevBoxCol)
            self.listRectanglesa1[self.box1id].update()
            self.listRectanglesa2[self.box1id].setBrush(self.prevBoxCol)
            self.listRectanglesa2[self.box1id].update()

        if self.p_spec.sceneBoundingRect().contains(pos):
            mousePoint = self.p_spec.mapSceneToView(pos)

            if self.started:
                # This is the second click, so should pay attention and close the segment
                # Stop the mouse motion connection, remove the drawing boxes
                if self.dragRectangles.isChecked():
                    return
                else:
                    if self.started_window == 's':
                        self.p_spec.scene().sigMouseMoved.disconnect()
                        self.p_spec.scene().sigMouseMoved.connect(self.mouseMoved)
                        self.p_spec.removeItem(self.vLine_s)
                    else:
                        self.p_ampl.scene().sigMouseMoved.disconnect()
                        self.p_ampl.removeItem(self.vLine_a)
                    self.p_ampl.removeItem(self.drawingBox_ampl)
                    self.p_spec.removeItem(self.drawingBox_spec)
                    #self.addSegment(self.start_location,self.convertSpectoAmpl(mousePoint.x()))
                    # If the user has pressed shift, copy the last species and don't use the context menu
                    modifiers = QtGui.QApplication.keyboardModifiers()
                    if modifiers == QtCore.Qt.ShiftModifier:
                        print "here"
                        self.addSegment(self.start_location, self.convertSpectoAmpl(mousePoint.x()), species=self.lastSpecies)
                    else:
                        print "there"
                        self.addSegment(self.start_location, self.convertSpectoAmpl(mousePoint.x()))
                        # Context menu
                        self.box1id = len(self.segments) - 1
                        self.fillBirdList()
                        self.menuBirdList.popup(QPoint(evt.screenPos().x(), evt.screenPos().y()))

                    self.playSegButton.setEnabled(True)
                    self.playBandLimitedSegButton.setEnabled(True)

                    # Context menu
                    #self.box1id = len(self.segments)-1
                    #self.menuBirdList.popup(QPoint(evt.screenPos().x(),evt.screenPos().y()))

                    self.listRectanglesa1[self.box1id].setBrush(fn.mkBrush(self.ColourSelected))
                    self.listRectanglesa1[self.box1id].update()
                    self.listRectanglesa2[self.box1id].setBrush(fn.mkBrush(self.ColourSelected))
                    self.listRectanglesa2[self.box1id].update()

                    self.started = not(self.started)
            else:
                # Check if the user has clicked in a box
                # Note: Returns the first one it finds
                box1id = -1
                for count in range(len(self.listRectanglesa2)):
                    if type(self.listRectanglesa2[count]) == self.ROItype:
                        x1 = self.listRectanglesa2[count].pos().x()
                        y1 = self.listRectanglesa2[count].pos().y()
                        x2 = x1 + self.listRectanglesa2[count].size().x()
                        y2 = y1 + self.listRectanglesa2[count].size().y()
                        if x1 <= mousePoint.x() and x2 >= mousePoint.x() and y1 <= mousePoint.y() and y2 >= mousePoint.y():
                            box1id = count
                    else:
                        x1, x2 = self.listRectanglesa2[count].getRegion()
                        if x1 <= mousePoint.x() and x2 >= mousePoint.x():
                            box1id = count

                if box1id > -1 and not evt.button() == QtCore.Qt.RightButton:
                    # User clicked in a box (with the left button)
                    self.box1id = box1id
                    self.prevBoxCol = self.listRectanglesa1[box1id].brush.color()
                    self.listRectanglesa1[box1id].setBrush(fn.mkBrush(self.ColourSelected))
                    self.listRectanglesa1[box1id].update()
                    self.playSegButton.setEnabled(True)
                    self.playBandLimitedSegButton.setEnabled(True)
                    self.listRectanglesa2[box1id].setBrush(fn.mkBrush(self.ColourSelected))
                    self.listRectanglesa2[box1id].update()

                    self.fillBirdList()
                    self.menuBirdList.popup(QPoint(evt.screenPos().x(), evt.screenPos().y()))
                else:
                    # User hasn't clicked in a box (or used the right button), so start a new segment
                    # Note that need to click in the same plot both times.
                    if self.dragRectangles.isChecked():
                        return
                    else:
                        self.start_location = self.convertSpectoAmpl(mousePoint.x())
                        self.vLine_s = pg.InfiniteLine(angle=90, movable=False,pen={'color': 'r', 'width': 3})
                        self.p_spec.addItem(self.vLine_s, ignoreBounds=True)
                        self.vLine_s.setPos(mousePoint.x())
                        self.playSegButton.setEnabled(False)
                        self.playBandLimitedSegButton.setEnabled(True)

                        brush = self.ColourNone
                        self.drawingBox_ampl = pg.LinearRegionItem(brush=brush)
                        self.p_ampl.addItem(self.drawingBox_ampl, ignoreBounds=True)
                        self.drawingBox_ampl.setRegion([self.start_location, self.start_location])
                        self.drawingBox_spec = pg.LinearRegionItem(brush=brush)
                        self.p_spec.addItem(self.drawingBox_spec, ignoreBounds=True)
                        self.drawingBox_spec.setRegion([mousePoint.x(),mousePoint.x()])
                        self.p_spec.scene().sigMouseMoved.connect(self.GrowBox_spec)
                        self.started_window = 's'

                        self.started = not (self.started)

    def mouseDragged_spec(self, evt1, evt2, evt3):
        if self.box1id>-1:
            self.listRectanglesa1[self.box1id].setBrush(self.prevBoxCol)
            self.listRectanglesa1[self.box1id].update()
            self.listRectanglesa2[self.box1id].setBrush(self.prevBoxCol)
            self.listRectanglesa2[self.box1id].update()

        if self.dragRectangles.isChecked():
            evt1 = self.p_spec.mapSceneToView(evt1)
            evt2 = self.p_spec.mapSceneToView(evt2)

            # If the user has pressed shift, copy the last species and don't use the context menu
            modifiers = QtGui.QApplication.keyboardModifiers()
            if modifiers == QtCore.Qt.ShiftModifier:
                self.addSegment(self.convertSpectoAmpl(evt1.x()), self.convertSpectoAmpl(evt2.x()), evt1.y(), evt2.y(),self.lastSpecies)
            else:
                self.addSegment(self.convertSpectoAmpl(evt1.x()), self.convertSpectoAmpl(evt2.x()), evt1.y(), evt2.y())
                # Context menu
                self.box1id = len(self.segments) - 1
                self.fillBirdList()
                self.menuBirdList.popup(QPoint(evt3.x(), evt3.y()))

            self.playSegButton.setEnabled(True)
            self.playBandLimitedSegButton.setEnabled(True)

            self.listRectanglesa1[self.box1id].setBrush(fn.mkBrush(self.ColourSelected))
            self.listRectanglesa1[self.box1id].update()
            self.listRectanglesa2[self.box1id].setBrush(fn.mkBrush(self.ColourSelected))
            self.listRectanglesa2[self.box1id].update()
        else:
            return

    def GrowBox_ampl(self,evt):
        pos = evt
        if self.p_ampl.sceneBoundingRect().contains(pos):
            mousePoint = self.p_ampl.mapSceneToView(pos)
            self.drawingBox_ampl.setRegion([self.start_location, mousePoint.x()])
            self.drawingBox_spec.setRegion([self.convertAmpltoSpec(self.start_location), self.convertAmpltoSpec(mousePoint.x())])

    def GrowBox_spec(self, evt):
        pos = evt
        if self.p_spec.sceneBoundingRect().contains(pos):
            mousePoint = self.p_spec.mapSceneToView(pos)
            self.drawingBox_ampl.setRegion([self.start_location, self.convertSpectoAmpl(mousePoint.x())])
            self.drawingBox_spec.setRegion([self.convertAmpltoSpec(self.start_location), mousePoint.x()])

    def birdSelected(self,birdname):
        # This collects the label for a bird from the context menu and processes it
        #print birdname, self.box1id
        if birdname is not 'Other':
            self.updateText(birdname)
            # Put the selected bird name at the top of the list
            self.config['BirdList'].remove(birdname)
            self.config['BirdList'].insert(0,birdname)

        else:
            text, ok = QInputDialog.getText(self, 'Bird name', 'Enter the bird name:')
            if ok:
                text = str(text).title()
                self.updateText(text)

                if text in self.config['BirdList']:
                    pass
                else:
                    # Add the new bird name.
                    #count = 0
                    #while self.config['BirdList'][count] < text and count < len(self.config['BirdList'])-1:
                    #    count += 1
                    #self.config['BirdList'].insert(count-1,text)
                    self.config['BirdList'].insert(0,text)
                    self.saveConfig = True

                    #bird = self.menuBird2.addAction(text)
                    #receiver = lambda birdname=text: self.birdSelected(birdname)
                    #self.connect(bird, SIGNAL("triggered()"), receiver)
                    #self.menuBird2.addAction(bird)

    def mouseMoved(self,evt):
        # Print the time, frequency, power for mouse location in the spectrogram
        # TODO: Format the time string, check the others
        # TODO: Position?
        if self.p_spec.sceneBoundingRect().contains(evt):
            mousePoint = self.p_spec.mapSceneToView(evt)
            indexx = int(mousePoint.x())
            indexy = int(mousePoint.y())
            if indexx > 0 and indexx < np.shape(self.sg)[1] and indexy > 0 and indexy < np.shape(self.sg)[0]:
                self.pointData.setText('time=%0.1f (s), freq=%0.1f (Hz),power=%0.1f (dB)' % (self.convertSpectoAmpl(mousePoint.x()), mousePoint.y() * self.sampleRate / 2. / np.shape(self.sg)[0], self.sg[int(mousePoint.y()), int(mousePoint.x())]))

    # def activateRadioButtons(self):
        # Make the radio buttons selectable
    #     found = False
    #     for i in range(len(self.birds1)):
    #         self.birds1[i].setEnabled(True)
    #         if str(self.a1text[self.box1id].get_text()) == self.birds1[i].text():
    #             self.birds1[i].setChecked(True)
    #             found = True
    #     for i in range(len(self.birds2)):
    #         self.birds2[i].setEnabled(True)
    #         if str(self.a1text[self.box1id].get_text()) == self.birds2[i].text():
    #             self.birds2[i].setChecked(True)
    #             found = True
    #
    #     if not found:
    #         # Select 'Other' in radio buttons (last item) and activate the listwidget
    #         self.birds2[-1].setChecked(True)
    #         self.birdList.setEnabled(True)
    #         items = self.birdList.findItems(str(self.a1text[self.box1id].get_text()), Qt.MatchExactly)
    #         for item in items:
    #             self.birdList.setCurrentItem(item)

    def dockReplace(self):
        self.area.restoreState(self.state)

    def updateText(self, text):
        # When the user sets or changes the name in a segment, update the text
        self.segments[self.box1id][4] = text
        self.listLabels[self.box1id].setText(text,'k')

        # Update the colour
        if text != "Don't Know":
            self.prevBoxCol = self.ColourNamed
        else:
            self.prevBoxCol = self.ColourNone

        self.lastSpecies = text

    # def radioBirdsClicked(self):
    #     # Listener for when the user selects a radio button
    #     # Update the text and store the data
    #     for button in self.birds1 + self.birds2:
    #         if button.isChecked():
    #             if button.text() == "Other":
    #                 self.birdList.setEnabled(True)
    #             else:
    #                 self.birdList.setEnabled(False)
    #                 self.updateText(str(button.text()))
    #
    # def listBirdsClicked(self, item):
    #     # Listener for clicks in the listbox of birds
    #     if (item.text() == "Other"):
    #         self.tbox.setEnabled(True)
    #     else:
    #         # Save the entry
    #         self.updateText(str(item.text()))
    #         # self.segments[self.box1id][4] = str(item.text())
    #         # self.a1text[self.box1id].set_text(str(item.text()))
    #         # # The font size is a pain because the window is in pixels, so have to transform it
    #         # # Force a redraw to make bounding box available
    #         # self.canvasMain.draw()
    #         # # fs = a1t.get_fontsize()
    #         # width = self.a1text[self.box1id].get_window_extent().inverse_transformed(self.a1.transData).width
    #         # if width > self.segments[self.box1id][1] - self.segments[self.box1id][0]:
    #         #     self.a1text[self.box1id].set_fontsize(8)
    #         #
    #         # self.listRectanglesa1[self.box1id].set_facecolor('b')
    #         # self.listRectanglesa2[self.box1id].set_facecolor('b')
    #         # self.topBoxCol = self.listRectanglesa1[self.box1id].get_facecolor()
    #         # self.canvasMain.draw()
    #
    # def birdTextEntered(self):
    #     # Listener for the text entry in the bird list
    #     # Check text isn't already in the listbox, and add if not
    #     # Doesn't sort the list, but will when program is closed
    #     item = self.birdList.findItems(self.tbox.text(), Qt.MatchExactly)
    #     if item:
    #         pass
    #     else:
    #         self.birdList.addItem(self.tbox.text())
    #         self.config['ListBirdsEntries'].append(str(self.tbox.text()))
    #     self.updateText(str(self.tbox.text()))
    #     # self.segments[self.box1id][4] = str(self.tbox.text())
    #     # self.a1text[self.box1id].set_text(str(self.tbox.text()))
    #     # # The font size is a pain because the window is in pixels, so have to transform it
    #     # # Force a redraw to make bounding box available
    #     # self.canvasMain.draw()
    #     # # fs = self.a1text[self.box1id].get_fontsize()
    #     # width = self.a1text[self.box1id].get_window_extent().inverse_transformed(self.a1.transData).width
    #     # if width > self.segments[self.box1id][1] - self.segments[self.box1id][0]:
    #     #     self.a1text[self.box1id].set_fontsize(8)
    #     #
    #     # self.listRectanglesa1[self.box1id].set_facecolor('b')
    #     # self.listRectanglesa2[self.box1id].set_facecolor('b')
    #     # self.topBoxCol = self.listRectanglesa1[self.box1id].get_facecolor()
    #     # self.canvasMain.draw()
    #     self.saveConfig = True
    #     # self.tbox.setEnabled(False)

    def setColourMap(self,cmap):
        # Colours taken from GradientEditorItem
        #self.colourList = ['Grey','Viridis', 'Inferno', 'Plasma', 'Autumn', 'Cool', 'Bone', 'Copper', 'Hot', 'Jet','Thermal','Flame','Yellowy','Bipolar','Spectrum']

        if cmap == 'Grey':
            pos = np.array([0., 1.])
            colour = np.array([[0, 0, 0, 255],[255, 255, 255, 255]])
            mode = 'rgb'
        # viridis
        elif cmap == 'Viridis':
            pos=np.array( [ 0.          ,0.00392157  ,0.00784314  ,0.01176471  ,0.01568627  ,0.01960784
          ,0.02352941  ,0.02745098  ,0.03137255  ,0.03529412  ,0.03921569  ,0.04313725
          ,0.04705882  ,0.05098039  ,0.05490196  ,0.05882353  ,0.0627451   ,0.06666667
          ,0.07058824  ,0.0745098   ,0.07843137  ,0.08235294  ,0.08627451  ,0.09019608
          ,0.09411765  ,0.09803922  ,0.10196078  ,0.10588235  ,0.10980392  ,0.11372549
          ,0.11764706  ,0.12156863  ,0.1254902   ,0.12941176  ,0.13333333  ,0.1372549
          ,0.14117647  ,0.14509804  ,0.14901961  ,0.15294118  ,0.15686275  ,0.16078431
          ,0.16470588  ,0.16862745  ,0.17254902  ,0.17647059  ,0.18039216  ,0.18431373
          ,0.18823529  ,0.19215686  ,0.19607843  ,0.2         ,0.20392157  ,0.20784314
          ,0.21176471  ,0.21568627  ,0.21960784  ,0.22352941  ,0.22745098  ,0.23137255
          ,0.23529412  ,0.23921569  ,0.24313725  ,0.24705882  ,0.25098039  ,0.25490196
          ,0.25882353  ,0.2627451   ,0.26666667  ,0.27058824  ,0.2745098   ,0.27843137
          ,0.28235294  ,0.28627451  ,0.29019608  ,0.29411765  ,0.29803922  ,0.30196078
          ,0.30588235  ,0.30980392  ,0.31372549  ,0.31764706  ,0.32156863  ,0.3254902
          ,0.32941176  ,0.33333333  ,0.3372549   ,0.34117647  ,0.34509804  ,0.34901961
          ,0.35294118  ,0.35686275  ,0.36078431  ,0.36470588  ,0.36862745  ,0.37254902
          ,0.37647059  ,0.38039216  ,0.38431373  ,0.38823529  ,0.39215686  ,0.39607843
          ,0.4         ,0.40392157  ,0.40784314  ,0.41176471  ,0.41568627  ,0.41960784
          ,0.42352941  ,0.42745098  ,0.43137255  ,0.43529412  ,0.43921569  ,0.44313725
          ,0.44705882  ,0.45098039  ,0.45490196  ,0.45882353  ,0.4627451   ,0.46666667
          ,0.47058824  ,0.4745098   ,0.47843137  ,0.48235294  ,0.48627451  ,0.49019608
          ,0.49411765  ,0.49803922  ,0.50196078  ,0.50588235  ,0.50980392  ,0.51372549
          ,0.51764706  ,0.52156863  ,0.5254902   ,0.52941176  ,0.53333333  ,0.5372549
          ,0.54117647  ,0.54509804  ,0.54901961  ,0.55294118  ,0.55686275  ,0.56078431
          ,0.56470588  ,0.56862745  ,0.57254902  ,0.57647059  ,0.58039216  ,0.58431373
          ,0.58823529  ,0.59215686  ,0.59607843  ,0.6         ,0.60392157  ,0.60784314
          ,0.61176471  ,0.61568627  ,0.61960784  ,0.62352941  ,0.62745098  ,0.63137255
          ,0.63529412  ,0.63921569  ,0.64313725  ,0.64705882  ,0.65098039  ,0.65490196
          ,0.65882353  ,0.6627451   ,0.66666667  ,0.67058824  ,0.6745098   ,0.67843137
          ,0.68235294  ,0.68627451  ,0.69019608  ,0.69411765  ,0.69803922  ,0.70196078
          ,0.70588235  ,0.70980392  ,0.71372549  ,0.71764706  ,0.72156863  ,0.7254902
          ,0.72941176  ,0.73333333  ,0.7372549   ,0.74117647  ,0.74509804  ,0.74901961
          ,0.75294118  ,0.75686275  ,0.76078431  ,0.76470588  ,0.76862745  ,0.77254902
          ,0.77647059  ,0.78039216  ,0.78431373  ,0.78823529  ,0.79215686  ,0.79607843
          ,0.8         ,0.80392157  ,0.80784314  ,0.81176471  ,0.81568627  ,0.81960784
          ,0.82352941  ,0.82745098  ,0.83137255  ,0.83529412  ,0.83921569  ,0.84313725
          ,0.84705882  ,0.85098039  ,0.85490196  ,0.85882353  ,0.8627451   ,0.86666667
          ,0.87058824  ,0.8745098   ,0.87843137  ,0.88235294  ,0.88627451  ,0.89019608
          ,0.89411765  ,0.89803922  ,0.90196078  ,0.90588235  ,0.90980392  ,0.91372549
          ,0.91764706  ,0.92156863  ,0.9254902   ,0.92941176  ,0.93333333  ,0.9372549
          ,0.94117647  ,0.94509804  ,0.94901961  ,0.95294118  ,0.95686275  ,0.96078431
          ,0.96470588  ,0.96862745  ,0.97254902  ,0.97647059  ,0.98039216  ,0.98431373
          ,0.98823529  ,0.99215686  ,0.99607843  ,1.        ] )
            colour=np.array( [[68.08602, 1.24287, 84.000825,255], [68.47005, 2.449275, 85.533885,255], [68.83572000000001, 3.729375, 87.051645,255], [69.182775, 5.08521, 88.553595,255], [69.51147, 6.518565, 90.038715,255], [69.821295, 8.031735, 91.507515,255], [70.11276, 9.62676, 92.958465,255], [70.38561, 11.262585, 94.39182,255], [70.63959, 12.83772, 95.807325,255], [70.874955, 14.36262, 97.203705,255], [71.091705, 15.846975, 98.58096,255], [71.28932999999999, 17.29818, 99.938835,255], [71.468085, 18.721335, 101.27656499999999,255], [71.62796999999999, 20.121285, 102.593895,255], [71.76872999999999, 21.5016, 103.89057,255], [71.89062, 22.864829999999998, 105.165825,255], [71.993385, 24.213525, 106.419405,255], [72.07728, 25.549979999999998, 107.65079999999999,255], [72.14205, 26.875215, 108.86001,255], [72.188205, 28.191015, 110.04626999999999,255], [72.21523499999999, 29.4984, 111.20932499999999,255], [72.223395, 30.798135, 112.34891999999999,255], [72.21268500000001, 32.09124, 113.46480000000001,255], [72.18336, 33.378225, 114.556455,255], [72.13542000000001, 34.659600000000005, 115.623885,255], [72.068865, 35.93613, 116.666835,255], [71.98395, 37.207559999999994, 117.68504999999999,255], [71.881185, 38.474655, 118.678275,255], [71.76006, 39.73767, 119.646255,255], [71.62134, 40.996605, 120.589245,255], [71.465025, 42.251715000000004, 121.50698999999999,255], [71.29137, 43.502745, 122.399235,255], [71.10063000000001, 44.74995, 123.26623500000001,255], [70.89305999999999, 45.993585, 124.10773499999999,255], [70.66917, 47.23314, 124.92399,255], [70.42947, 48.468869999999995, 125.715255,255], [70.173705, 49.700775, 126.481275,255], [69.90263999999999, 50.928855000000006, 127.22230499999999,255], [69.61653000000001, 52.1526, 127.93885499999999,255], [69.31614, 53.372265, 128.63067,255], [69.001725, 54.587595, 129.29826,255], [68.67354, 55.798590000000004, 129.94213499999998,255], [68.33184, 57.004995, 130.56204,255], [67.97789999999999, 58.20681, 131.15899499999998,255], [67.611975, 59.40378, 131.732745,255], [67.234065, 60.595905, 132.28430999999998,255], [66.84519, 61.78293, 132.813435,255], [66.445605, 62.96511, 133.32113999999999,255], [66.036075, 64.141935, 133.80768,255], [65.61711, 65.31315000000001, 134.273565,255], [65.189475, 66.479265, 134.71956,255], [64.75342500000001, 67.63977, 135.145665,255], [64.30946999999999, 68.794665, 135.552645,255], [63.858375, 69.94395, 135.941265,255], [63.400394999999996, 71.087625, 136.31178,255], [62.936805, 72.225435, 136.664955,255], [62.46786, 73.35712500000001, 137.0013,255], [61.993815, 74.48346000000001, 137.32158,255], [61.515435000000004, 75.603675, 137.625795,255], [61.03323, 76.718025, 137.91522,255], [60.54745500000001, 77.82651, 138.189855,255], [60.05913, 78.929385, 138.45072,255], [59.568765, 80.02614, 138.69807,255], [59.07687, 81.11703, 138.93267,255], [58.583445, 82.202055, 139.15503,255], [58.089510000000004, 83.28147, 139.36566,255], [57.595065000000005, 84.355275, 139.56507,255], [57.100875, 85.42347000000001, 139.753515,255], [56.607195, 86.486055, 139.93176,255], [56.114535000000004, 87.543285, 140.10031500000002,255], [55.623149999999995, 88.59516, 140.25969,255], [55.13355000000001, 89.641425, 140.409885,255], [54.64599, 90.682845, 140.55192,255], [54.160725, 91.71916499999999, 140.68605,255], [53.678264999999996, 92.75038500000001, 140.81252999999998,255], [53.198865, 93.77676000000001, 140.932125,255], [52.72278, 94.79829, 141.044835,255], [52.250265, 95.81523, 141.15091500000003,255], [51.781065, 96.82758, 141.250875,255], [51.315945, 97.83585000000001, 141.34497,255], [50.85465, 98.83978499999999, 141.43371,255], [50.39718, 99.83964, 141.517095,255], [49.9443, 100.835415, 141.59538,255], [49.4955, 101.827365, 141.669075,255], [49.051035, 102.81574499999999, 141.73818,255], [48.610904999999995, 103.800555, 141.80269500000003,255], [48.175365, 104.78205, 141.86312999999998,255], [47.743905000000005, 105.76023, 141.919485,255], [47.31678, 106.73535, 141.97201500000003,255], [46.89399, 107.707665, 142.02072,255], [46.47528, 108.67692, 142.0656,255], [46.060395, 109.643625, 142.10691000000003,255], [45.649845000000006, 110.60777999999999, 142.14464999999998,255], [45.242865, 111.569385, 142.17907499999998,255], [44.839455, 112.52895000000001, 142.209675,255], [44.439870000000006, 113.48622, 142.23695999999998,255], [44.043345, 114.441705, 142.260675,255], [43.649879999999996, 115.39515, 142.28107500000002,255], [43.25973, 116.34681, 142.29765,255], [42.87213, 117.29694, 142.31090999999998,255], [42.487334999999995, 118.24554, 142.320345,255], [42.104835, 119.192865, 142.325955,255], [41.724374999999995, 120.13891500000001, 142.32774,255], [41.34621, 121.08368999999999, 142.32569999999998,255], [40.969575, 122.02770000000001, 142.31932500000002,255], [40.59447, 122.97043500000001, 142.308615,255], [40.220895, 123.91265999999999, 142.293315,255], [39.84885, 124.85412, 142.27367999999998,255], [39.477825, 125.794815, 142.2492,255], [39.10782, 126.735, 142.21962,255], [38.73909, 127.67467500000001, 142.184685,255], [38.37138, 128.614095, 142.14464999999998,255], [38.004945, 129.553005, 142.09875,255], [37.639784999999996, 130.491915, 142.047495,255], [37.2759, 131.430315, 141.98986499999998,255], [36.913545, 132.368715, 141.92586,255], [36.552465, 133.307115, 141.855225,255], [36.193425000000005, 134.24551499999998, 141.777705,255], [35.83668, 135.18366, 141.693045,255], [35.482485, 136.12205999999998, 141.60099,255], [35.13135, 137.06046, 141.50103000000001,255], [34.78404, 137.99911500000002, 141.39316499999998,255], [34.441829999999996, 138.93751500000002, 141.27739499999998,255], [34.104465, 139.876425, 141.152955,255], [33.77322, 140.81508000000002, 141.01959,255], [33.44886, 141.754245, 140.877045,255], [33.132915, 142.69341, 140.72532,255], [32.825895, 143.632575, 140.56339499999999,255], [32.52984, 144.57199500000002, 140.39178,255], [32.245515000000005, 145.511415, 140.20945500000002,255], [31.97547, 146.45109, 140.01693,255], [31.720725, 147.39051, 139.813185,255], [31.483065, 148.330185, 139.59847499999998,255], [31.26453, 149.26960499999998, 139.37203499999998,255], [31.066905, 150.209025, 139.133865,255], [30.89274, 151.148445, 138.883455,255], [30.744075000000002, 152.08761, 138.620805,255], [30.62346, 153.02652, 138.34515,255], [30.533189999999998, 153.96517500000002, 138.057,255], [30.475559999999998, 154.90332, 137.75558999999998,255], [30.452865, 155.840955, 137.44040999999999,255], [30.468165000000003, 156.778335, 137.11146,255], [30.523245, 157.71495, 136.768485,255], [30.620655, 158.65105499999999, 136.41123000000002,255], [30.76269, 159.58614, 136.03943999999998,255], [30.951900000000002, 160.52046, 135.653115,255], [31.18956, 161.454015, 135.25149000000002,255], [31.47822, 162.386295, 134.834565,255], [31.8189, 163.317555, 134.40233999999998,255], [32.21313, 164.247285, 133.954305,255], [32.662185, 165.175995, 133.490205,255], [33.167085, 166.10291999999998, 133.01004,255], [33.728339999999996, 167.02857, 132.513555,255], [34.34646, 167.95218, 132.000495,255], [35.021445, 168.87426, 131.470605,255], [35.75355, 169.79404499999998, 130.92388499999998,255], [36.542265, 170.71204500000002, 130.359825,255], [37.38708, 171.62775000000002, 129.77868,255], [38.28774, 172.540905, 129.180195,255], [39.24297, 173.451765, 128.56385999999998,255], [40.252005, 174.360075, 127.92992999999998,255], [41.31408, 175.26558, 127.277895,255], [42.427665, 176.16828, 126.60801,255], [43.591739999999994, 177.06792, 125.919765,255], [44.805285, 177.9645, 125.213415,255], [46.066515, 178.85751, 124.48819499999999,255], [47.374665, 179.747205, 123.744615,255], [48.72795, 180.63333, 122.98241999999999,255], [50.125605, 181.515885, 122.201355,255], [51.565845, 182.39436, 121.40142,255], [53.04765, 183.268755, 120.58261499999999,255], [54.57, 184.13907, 119.74494,255], [56.13162, 185.004795, 118.88762999999999,255], [57.731235, 185.86643999999998, 118.011195,255], [59.367824999999996, 186.722985, 117.115635,255], [61.04037, 187.57494, 116.20044,255], [62.74785, 188.42204999999998, 115.26612,255], [64.489245, 189.263805, 114.31242,255], [66.263535, 190.10046, 113.339085,255], [68.069955, 190.93150500000002, 112.346115,255], [69.907995, 191.75694, 111.33325500000001,255], [71.776635, 192.576765, 110.30076,255], [73.674855, 193.39047, 109.24862999999999,255], [75.602145, 194.198055, 108.176865,255], [77.55774, 194.99952000000002, 107.085465,255], [79.540875, 195.79461, 105.97443,255], [81.551295, 196.58307, 104.84376,255], [83.58797999999999, 197.3649, 103.6932,255], [85.65067499999999, 198.13959, 102.52249499999999,255], [87.73886999999999, 198.90739499999998, 101.332155,255], [89.8518, 199.66780500000002, 100.12218,255], [91.98895499999999, 200.42082, 98.89256999999999,255], [94.14957, 201.16644000000002, 97.64307,255], [96.33364499999999, 201.904155, 96.37444500000001,255], [98.54041500000001, 202.63422, 95.08593,255], [100.76937000000001, 203.35612500000002, 93.778035,255], [103.020255, 204.070125, 92.45076,255], [105.29281499999999, 204.775455, 91.103595,255], [107.58654, 205.47236999999998, 89.73705,255], [109.900665, 206.160615, 88.35138,255], [112.23493500000001, 206.84019, 86.946585,255], [114.58883999999999, 207.51084, 85.52292,255], [116.96187, 208.172565, 84.08038499999999,255], [119.353515, 208.824855, 82.61949,255], [121.76352, 209.46821999999997, 81.139725,255], [124.19163, 210.101895, 79.641855,255], [126.63682499999999, 210.72588, 78.126135,255], [129.099105, 211.34043, 76.59231,255], [131.57796, 211.94529, 75.041145,255], [134.07288, 212.540205, 73.472385,255], [136.583355, 213.125175, 71.88654,255], [139.10862, 213.69994499999999, 70.28462999999999,255], [141.64842, 214.26477, 68.66665499999999,255], [144.20199, 214.81965, 67.033635,255], [146.76856500000002, 215.36433000000002, 65.385825,255], [149.34789, 215.898555, 63.723735000000005,255], [151.938945, 216.42283500000002, 62.048894999999995,255], [154.541475, 216.936915, 60.361560000000004,255], [157.15471499999998, 217.440795, 58.66326,255], [159.777645, 217.934475, 56.955014999999996,255], [162.41001, 218.41821000000002, 55.2381,255], [165.050535, 218.89200000000002, 53.514555,255], [167.69870999999998, 219.355845, 51.78591,255], [170.35377, 219.809745, 50.054715,255], [173.014695, 220.25421, 48.323265,255], [175.68072, 220.68924, 46.594875,255], [178.35082500000001, 221.114835, 44.872605,255], [181.02399, 221.531505, 43.160534999999996,255], [183.699705, 221.93925, 41.463765,255], [186.376695, 222.33858, 39.787395000000004,255], [189.05394, 222.72949500000001, 38.138055,255], [191.73042, 223.112505, 36.52314,255], [194.405115, 223.48812, 34.951319999999996,255], [197.07726, 223.85634, 33.432795,255], [199.745325, 224.21767499999999, 31.978274999999996,255], [202.4088, 224.57289, 30.601275,255], [205.06641, 224.92173, 29.316074999999998,255], [207.71688, 225.26521499999998, 28.138485,255], [210.3597, 225.6036, 27.085335,255], [212.99384999999998, 225.93739499999998, 26.17473,255], [215.618055, 226.26711, 25.42401,255], [218.23155, 226.593255, 24.85026,255], [220.833315, 226.91634, 24.468014999999998,255], [223.42283999999998, 227.236875, 24.28875,255], [225.99910500000001, 227.55537, 24.32037,255], [228.5616, 227.87207999999998, 24.565425,255], [231.109305, 228.18802499999998, 25.021875,255], [233.64171, 228.50320499999998, 25.682835,255], [236.15703, 228.81914999999998, 26.538104999999998,255], [238.65552, 229.13535, 27.573405,255], [241.13718, 229.45282500000002, 28.77369,255], [243.60150000000002, 229.771575, 30.12264,255], [246.04797000000002, 230.092365, 31.604955,255], [248.476335, 230.41545, 33.204825,255], [250.88634, 230.741085, 34.908735,255], [253.27824, 231.070035, 36.703680000000006,255]] ,dtype=np.ubyte)
            mode = 'rgb'
        # inferno
        elif cmap == 'Inferno':
            pos=np.array( [ 0.          ,0.00392157  ,0.00784314  ,0.01176471  ,0.01568627  ,0.01960784
          ,0.02352941  ,0.02745098  ,0.03137255  ,0.03529412  ,0.03921569  ,0.04313725
          ,0.04705882  ,0.05098039  ,0.05490196  ,0.05882353  ,0.0627451   ,0.06666667
          ,0.07058824  ,0.0745098   ,0.07843137  ,0.08235294  ,0.08627451  ,0.09019608
          ,0.09411765  ,0.09803922  ,0.10196078  ,0.10588235  ,0.10980392  ,0.11372549
          ,0.11764706  ,0.12156863  ,0.1254902   ,0.12941176  ,0.13333333  ,0.1372549
          ,0.14117647  ,0.14509804  ,0.14901961  ,0.15294118  ,0.15686275  ,0.16078431
          ,0.16470588  ,0.16862745  ,0.17254902  ,0.17647059  ,0.18039216  ,0.18431373
          ,0.18823529  ,0.19215686  ,0.19607843  ,0.2         ,0.20392157  ,0.20784314
          ,0.21176471  ,0.21568627  ,0.21960784  ,0.22352941  ,0.22745098  ,0.23137255
          ,0.23529412  ,0.23921569  ,0.24313725  ,0.24705882  ,0.25098039  ,0.25490196
          ,0.25882353  ,0.2627451   ,0.26666667  ,0.27058824  ,0.2745098   ,0.27843137
          ,0.28235294  ,0.28627451  ,0.29019608  ,0.29411765  ,0.29803922  ,0.30196078
          ,0.30588235  ,0.30980392  ,0.31372549  ,0.31764706  ,0.32156863  ,0.3254902
          ,0.32941176  ,0.33333333  ,0.3372549   ,0.34117647  ,0.34509804  ,0.34901961
          ,0.35294118  ,0.35686275  ,0.36078431  ,0.36470588  ,0.36862745  ,0.37254902
          ,0.37647059  ,0.38039216  ,0.38431373  ,0.38823529  ,0.39215686  ,0.39607843
          ,0.4         ,0.40392157  ,0.40784314  ,0.41176471  ,0.41568627  ,0.41960784
          ,0.42352941  ,0.42745098  ,0.43137255  ,0.43529412  ,0.43921569  ,0.44313725
          ,0.44705882  ,0.45098039  ,0.45490196  ,0.45882353  ,0.4627451   ,0.46666667
          ,0.47058824  ,0.4745098   ,0.47843137  ,0.48235294  ,0.48627451  ,0.49019608
          ,0.49411765  ,0.49803922  ,0.50196078  ,0.50588235  ,0.50980392  ,0.51372549
          ,0.51764706  ,0.52156863  ,0.5254902   ,0.52941176  ,0.53333333  ,0.5372549
          ,0.54117647  ,0.54509804  ,0.54901961  ,0.55294118  ,0.55686275  ,0.56078431
          ,0.56470588  ,0.56862745  ,0.57254902  ,0.57647059  ,0.58039216  ,0.58431373
          ,0.58823529  ,0.59215686  ,0.59607843  ,0.6         ,0.60392157  ,0.60784314
          ,0.61176471  ,0.61568627  ,0.61960784  ,0.62352941  ,0.62745098  ,0.63137255
          ,0.63529412  ,0.63921569  ,0.64313725  ,0.64705882  ,0.65098039  ,0.65490196
          ,0.65882353  ,0.6627451   ,0.66666667  ,0.67058824  ,0.6745098   ,0.67843137
          ,0.68235294  ,0.68627451  ,0.69019608  ,0.69411765  ,0.69803922  ,0.70196078
          ,0.70588235  ,0.70980392  ,0.71372549  ,0.71764706  ,0.72156863  ,0.7254902
          ,0.72941176  ,0.73333333  ,0.7372549   ,0.74117647  ,0.74509804  ,0.74901961
          ,0.75294118  ,0.75686275  ,0.76078431  ,0.76470588  ,0.76862745  ,0.77254902
          ,0.77647059  ,0.78039216  ,0.78431373  ,0.78823529  ,0.79215686  ,0.79607843
          ,0.8         ,0.80392157  ,0.80784314  ,0.81176471  ,0.81568627  ,0.81960784
          ,0.82352941  ,0.82745098  ,0.83137255  ,0.83529412  ,0.83921569  ,0.84313725
          ,0.84705882  ,0.85098039  ,0.85490196  ,0.85882353  ,0.8627451   ,0.86666667
          ,0.87058824  ,0.8745098   ,0.87843137  ,0.88235294  ,0.88627451  ,0.89019608
          ,0.89411765  ,0.89803922  ,0.90196078  ,0.90588235  ,0.90980392  ,0.91372549
          ,0.91764706  ,0.92156863  ,0.9254902   ,0.92941176  ,0.93333333  ,0.9372549
          ,0.94117647  ,0.94509804  ,0.94901961  ,0.95294118  ,0.95686275  ,0.96078431
          ,0.96470588  ,0.96862745  ,0.97254902  ,0.97647059  ,0.98039216  ,0.98431373
          ,0.98823529  ,0.99215686  ,0.99607843  ,1.        ] )
            colour=np.array( [[0.37281, 0.11883, 3.53583,255], [0.578085, 0.32385, 4.7353499999999995,255], [0.8412449999999999, 0.5734950000000001, 6.180945,255], [1.1594849999999999, 0.8649600000000001, 7.881794999999999,255], [1.53153, 1.19646, 9.83229,255], [1.95738, 1.56468, 11.94318,255], [2.438055, 1.966815, 14.061465,255], [2.974065, 2.401335, 16.1823,255], [3.568725, 2.862375, 18.32481,255], [4.223055, 3.34968, 20.47191,255], [4.9401150000000005, 3.858915, 22.635585,255], [5.723985000000001, 4.385745, 24.818385,255], [6.577215, 4.929405, 27.01215,255], [7.50516, 5.483265, 29.228355,255], [8.513175, 6.04401, 31.466235,255], [9.60534, 6.609855, 33.719159999999995,255], [10.774515, 7.175445, 35.990955,255], [11.963325, 7.73262, 38.29182,255], [13.169220000000001, 8.28087, 40.609770000000005,255], [14.394495, 8.815095000000001, 42.945570000000004,255], [15.6417, 9.330449999999999, 45.29871,255], [16.914405, 9.81852, 47.675309999999996,255], [18.214395000000003, 10.274970000000001, 50.07027,255], [19.542434999999998, 10.685775, 52.478745,255], [20.900309999999998, 11.048639999999999, 54.898695000000004,255], [22.289805, 11.36178, 57.327315000000006,255], [23.71245, 11.623664999999999, 59.76129,255], [25.16901, 11.83251, 62.19552,255], [26.660505, 11.98704, 64.62465,255], [28.18668, 12.086744999999999, 67.04256,255], [29.74728, 12.13137, 69.44185499999999,255], [31.341540000000002, 12.121680000000001, 71.81412,255], [32.967675, 12.059715, 74.15093999999999,255], [34.62339, 11.94828, 76.44288,255], [36.30639, 11.79171, 78.681015,255], [38.013615, 11.59434, 80.856675,255], [39.741749999999996, 11.362545, 82.96119,255], [41.485695, 11.10627, 84.985635,255], [43.241625, 10.834695, 86.92287,255], [45.005715, 10.55751, 88.768305,255], [46.774395000000005, 10.283895, 90.51760499999999,255], [48.543585, 10.023795, 92.168985,255], [50.310735, 9.792, 93.721425,255], [52.073295, 9.59616, 95.17569,255], [53.829225, 9.44265, 96.533565,255], [55.576995000000004, 9.336825000000001, 97.79811,255], [57.314564999999995, 9.283275, 98.972895,255], [59.04219, 9.283275, 100.06200000000001,255], [60.759615000000004, 9.338355, 101.070015,255], [62.466584999999995, 9.449024999999999, 102.001785,255], [64.1631, 9.614775, 102.86139,255], [65.84967, 9.835605000000001, 103.65367499999999,255], [67.52655, 10.109985, 104.382975,255], [69.193485, 10.43511, 105.05388,255], [70.85175, 10.800015, 105.66995999999999,255], [72.50185499999999, 11.202915, 106.23504,255], [74.144565, 11.63922, 106.75243499999999,255], [75.78039, 12.104849999999999, 107.225205,255], [77.40984, 12.59598, 107.65641,255], [79.03342500000001, 13.108785000000001, 108.048855,255], [80.65191, 13.63995, 108.40458,255], [82.26555, 14.186670000000001, 108.726135,255], [83.87485500000001, 14.745885000000001, 109.015305,255], [85.480335, 15.3153, 109.27362000000001,255], [87.08250000000001, 15.892875, 109.503375,255], [88.681605, 16.47708, 109.705335,255], [90.27816, 17.065875, 109.88103000000001,255], [91.87242, 17.657985, 110.03173500000001,255], [93.464895, 18.252645, 110.15847,255], [95.05584, 18.848325, 110.262,255], [96.64525499999999, 19.444515, 110.343345,255], [98.23314, 20.040705, 110.40352499999999,255], [99.820515, 20.636385, 110.442795,255], [101.40687000000001, 21.230535, 110.461665,255], [102.99297, 21.8229, 110.460645,255], [104.578815, 22.41348, 110.43999,255], [106.164405, 23.001765000000002, 110.40046500000001,255], [107.749995, 23.587755, 110.34206999999999,255], [109.33583999999999, 24.17145, 110.26506,255], [110.921685, 24.752595, 110.169945,255], [112.507785, 25.33119, 110.05646999999999,255], [114.09414, 25.907235, 109.92540000000001,255], [115.68100500000001, 26.48124, 109.77699,255], [117.268125, 27.052695, 109.61073,255], [118.8555, 27.62211, 109.426875,255], [120.44364, 28.189485, 109.22516999999999,255], [122.03228999999999, 28.754820000000002, 109.006125,255], [123.621195, 29.31837, 108.76974,255], [125.21061, 29.880645, 108.51576,255], [126.800535, 30.441645, 108.24444,255], [128.390715, 31.001625, 107.95578,255], [129.98115, 31.561095, 107.64977999999999,255], [131.571585, 32.1198, 107.32618500000001,255], [133.16252999999998, 32.67825, 106.984995,255], [134.75322, 33.236955, 106.62621,255], [136.344165, 33.796170000000004, 106.250085,255], [137.9346, 34.355895, 105.85636500000001,255], [139.525035, 34.916895, 105.445305,255], [141.11496, 35.47917, 105.016395,255], [142.70412, 36.04323, 104.56989,255], [144.29277, 36.609585, 104.10579,255], [145.880655, 37.178235, 103.624095,255], [147.46752, 37.749945000000004, 103.12480500000001,255], [149.052855, 38.32497, 102.608175,255], [150.63717, 38.903565, 102.07395,255], [152.21970000000002, 39.48624, 101.52187500000001,255], [153.800445, 40.073505000000004, 100.95220499999999,255], [155.37915, 40.66587, 100.365195,255], [156.955815, 41.263335, 99.76084499999999,255], [158.529675, 41.86692, 99.139155,255], [160.100985, 42.476625, 98.50038,255], [161.66949, 43.09296, 97.84452,255], [163.23442500000002, 43.71669, 97.17157499999999,255], [164.79629999999997, 44.34807, 96.481545,255], [166.354095, 44.987355, 95.77443,255], [167.908065, 45.635310000000004, 95.05074,255], [169.45770000000002, 46.292445, 94.31073,255], [171.00274499999998, 46.959015, 93.554145,255], [172.54269, 47.635785, 92.78149499999999,255], [174.07728, 48.322755, 91.99303499999999,255], [175.606515, 49.020945, 91.188765,255], [177.129885, 49.730355, 90.36894,255], [178.64687999999998, 50.452005, 89.533815,255], [180.1575, 51.18564, 88.68313500000001,255], [181.66098, 51.93228, 87.817665,255], [183.15732, 52.69217999999999, 86.937405,255], [184.646265, 53.465849999999996, 86.04312,255], [186.12679500000002, 54.253545, 85.134555,255], [187.599165, 55.05603, 84.212475,255], [189.06286500000002, 55.87356, 83.27687999999999,255], [190.517385, 56.70639, 82.32827999999999,255], [191.96247, 57.555029999999995, 81.366675,255], [193.39761000000001, 58.419734999999996, 80.39283,255], [194.82254999999998, 59.30127, 79.406745,255], [196.23678, 60.199635, 78.408675,255], [197.64004500000001, 61.115085, 77.39913,255], [199.031835, 62.048384999999996, 76.378365,255], [200.411895, 62.99928, 75.34663499999999,255], [201.779715, 63.96828000000001, 74.30444999999999,255], [203.134785, 64.95564, 73.25232000000001,255], [204.477105, 65.96187, 72.190245,255], [205.80590999999998, 66.98646, 71.11899,255], [207.120945, 68.03043000000001, 70.038555,255], [208.421955, 69.09326999999999, 68.94945,255], [209.70843, 70.175235, 67.851675,255], [210.97986, 71.276835, 66.74624999999999,255], [212.236245, 72.39781500000001, 65.63266499999999,255], [213.477075, 73.538175, 64.51194,255], [214.70209499999999, 74.697915, 63.38382,255], [215.910795, 75.877545, 62.248815,255], [217.10292, 77.07629999999999, 61.10718,255], [218.27796, 78.29468999999999, 59.958915000000005,255], [219.435915, 79.53246, 58.80453,255], [220.57653000000002, 80.78961, 57.644025,255], [221.699295, 82.065885, 56.47791,255], [222.803955, 83.36103, 55.30593,255], [223.890255, 84.67530000000001, 54.12834,255], [224.95793999999998, 86.008185, 52.94514,255], [226.00701, 87.35943, 51.756840000000004,255], [227.036955, 88.72903500000001, 50.562929999999994,255], [228.047775, 90.11674500000001, 49.36392,255], [229.03896, 91.52230499999999, 48.1593,255], [230.010765, 92.94546, 46.94958,255], [230.962425, 94.3857, 45.73425,255], [231.89445, 95.84328000000001, 44.513565,255], [232.80633, 97.31718, 43.287524999999995,255], [233.69781, 98.80765500000001, 42.05562,255], [234.569145, 100.314195, 40.81785,255], [235.419825, 101.836545, 39.574215,255], [236.24985, 103.374195, 38.32446,255], [237.05922, 104.927145, 37.068585,255], [237.847935, 106.49488500000001, 35.806335,255], [238.615485, 108.076905, 34.5372,255], [239.36212500000002, 109.673205, 33.26169,255], [240.08785500000002, 111.283275, 31.979294999999997,255], [240.792675, 112.90686, 30.69027,255], [241.47607499999998, 114.543705, 29.39436,255], [242.13831000000002, 116.19330000000001, 28.09182,255], [242.779125, 117.85539, 26.782905,255], [243.39902999999998, 119.52972, 25.46787,255], [243.99726, 121.21578, 24.147225,255], [244.57407, 122.91356999999999, 22.822245,255], [245.12971499999998, 124.62258, 21.493695000000002,255], [245.663685, 126.34281, 20.163615,255], [246.176235, 128.07349499999998, 18.834045,255], [246.66711, 129.81489000000002, 17.508045,255], [247.136565, 131.56623, 16.18944,255], [247.58434499999998, 133.327515, 14.883585,255], [248.01045, 135.09849, 13.597620000000001,255], [248.41488, 136.87890000000002, 12.33996,255], [248.797635, 138.66849, 11.122589999999999,255], [249.15846, 140.46675, 9.95775,255], [249.49761, 142.273935, 8.907404999999999,255], [249.81483, 144.089535, 8.009295,255], [250.11012, 145.913295, 7.269539999999999,255], [250.38322499999998, 147.74496, 6.69375,255], [250.634655, 149.58453, 6.288555,255], [250.863645, 151.43149499999998, 6.06135,255], [251.070705, 153.28611, 6.01953,255], [251.255325, 155.14761000000001, 6.1715100000000005,255], [251.41776000000002, 157.01625, 6.52596,255], [251.55801, 158.891775, 7.092569999999999,255], [251.67582, 160.773675, 7.88154,255], [251.770935, 162.66195, 8.90358,255], [251.84361, 164.5566, 10.17093,255], [251.893845, 166.45711500000002, 11.623155,255], [251.92113, 168.36375, 13.19625,255], [251.925975, 170.27574, 14.873895,255], [251.90787, 172.193085, 16.640535,255], [251.86706999999998, 174.11578500000002, 18.484695,255], [251.80332, 176.04333, 20.397450000000003,255], [251.71662, 177.97572, 22.371405,255], [251.60697, 179.91269999999997, 24.40197,255], [251.474625, 181.854015, 26.485065,255], [251.31933, 183.79941, 28.618395,255], [251.14057499999998, 185.748885, 30.800175,255], [250.93912500000002, 187.70218500000001, 33.029385,255], [250.71498, 189.65829, 35.305515,255], [250.46814, 191.61771000000002, 37.629075,255], [250.19911499999998, 193.57942500000001, 40.000065,255], [249.90816, 195.543435, 42.420015,255], [249.59553, 197.50897500000002, 44.889435,255], [249.261735, 199.47579, 47.410365,255], [248.90753999999998, 201.44337, 49.98459,255], [248.53269, 203.41145999999998, 52.61465999999999,255], [248.13744, 205.379295, 55.303635,255], [247.72434, 207.34611, 58.05279,255], [247.29466499999998, 209.310375, 60.86493,255], [246.850455, 211.271325, 63.74286,255], [246.391965, 213.228705, 66.69117,255], [245.92047, 215.18124, 69.714705,255], [245.441835, 217.12638, 72.81423000000001,255], [244.95963, 219.062595, 75.99255,255], [244.4736, 220.98911999999999, 79.25909999999999,255], [243.99267, 222.902895, 82.61336999999999,255], [243.524235, 224.800095, 86.05612500000001,255], [243.069825, 226.68021000000002, 89.59909499999999,255], [242.64423, 228.53762999999998, 93.23488499999999,255], [242.25459, 230.369295, 96.96910500000001,255], [241.91416500000003, 232.170615, 100.798695,255], [241.63647, 233.936745, 104.719575,255], [241.436295, 235.66284, 108.725115,255], [241.32996, 237.344055, 112.803585,255], [241.332765, 238.97554499999998, 116.94096,255], [241.46026500000002, 240.55374, 121.11735,255], [241.723935, 242.07609, 125.31362999999999,255], [242.133975, 243.541065, 129.5043,255], [242.6937, 244.949685, 133.671765,255], [243.40489499999998, 246.30348, 137.792055,255], [244.26348, 247.605765, 141.850125,255], [245.26206, 248.86062, 145.840875,255], [246.393495, 250.07289, 149.73753,255], [247.64631, 251.24690999999999, 153.54926999999998,255], [249.01030500000002, 252.387015, 157.2738,255], [250.475535, 253.497795, 160.909335,255], [252.03231, 254.58282, 164.45562,255]] ,dtype=np.ubyte)
            mode = 'rgb'
        # plasma
        elif cmap == 'Plasma':
            pos=np.array( [ 0.          ,0.00392157  ,0.00784314  ,0.01176471  ,0.01568627  ,0.01960784
          ,0.02352941  ,0.02745098  ,0.03137255  ,0.03529412  ,0.03921569  ,0.04313725
          ,0.04705882  ,0.05098039  ,0.05490196  ,0.05882353  ,0.0627451   ,0.06666667
          ,0.07058824  ,0.0745098   ,0.07843137  ,0.08235294  ,0.08627451  ,0.09019608
          ,0.09411765  ,0.09803922  ,0.10196078  ,0.10588235  ,0.10980392  ,0.11372549
          ,0.11764706  ,0.12156863  ,0.1254902   ,0.12941176  ,0.13333333  ,0.1372549
          ,0.14117647  ,0.14509804  ,0.14901961  ,0.15294118  ,0.15686275  ,0.16078431
          ,0.16470588  ,0.16862745  ,0.17254902  ,0.17647059  ,0.18039216  ,0.18431373
          ,0.18823529  ,0.19215686  ,0.19607843  ,0.2         ,0.20392157  ,0.20784314
          ,0.21176471  ,0.21568627  ,0.21960784  ,0.22352941  ,0.22745098  ,0.23137255
          ,0.23529412  ,0.23921569  ,0.24313725  ,0.24705882  ,0.25098039  ,0.25490196
          ,0.25882353  ,0.2627451   ,0.26666667  ,0.27058824  ,0.2745098   ,0.27843137
          ,0.28235294  ,0.28627451  ,0.29019608  ,0.29411765  ,0.29803922  ,0.30196078
          ,0.30588235  ,0.30980392  ,0.31372549  ,0.31764706  ,0.32156863  ,0.3254902
          ,0.32941176  ,0.33333333  ,0.3372549   ,0.34117647  ,0.34509804  ,0.34901961
          ,0.35294118  ,0.35686275  ,0.36078431  ,0.36470588  ,0.36862745  ,0.37254902
          ,0.37647059  ,0.38039216  ,0.38431373  ,0.38823529  ,0.39215686  ,0.39607843
          ,0.4         ,0.40392157  ,0.40784314  ,0.41176471  ,0.41568627  ,0.41960784
          ,0.42352941  ,0.42745098  ,0.43137255  ,0.43529412  ,0.43921569  ,0.44313725
          ,0.44705882  ,0.45098039  ,0.45490196  ,0.45882353  ,0.4627451   ,0.46666667
          ,0.47058824  ,0.4745098   ,0.47843137  ,0.48235294  ,0.48627451  ,0.49019608
          ,0.49411765  ,0.49803922  ,0.50196078  ,0.50588235  ,0.50980392  ,0.51372549
          ,0.51764706  ,0.52156863  ,0.5254902   ,0.52941176  ,0.53333333  ,0.5372549
          ,0.54117647  ,0.54509804  ,0.54901961  ,0.55294118  ,0.55686275  ,0.56078431
          ,0.56470588  ,0.56862745  ,0.57254902  ,0.57647059  ,0.58039216  ,0.58431373
          ,0.58823529  ,0.59215686  ,0.59607843  ,0.6         ,0.60392157  ,0.60784314
          ,0.61176471  ,0.61568627  ,0.61960784  ,0.62352941  ,0.62745098  ,0.63137255
          ,0.63529412  ,0.63921569  ,0.64313725  ,0.64705882  ,0.65098039  ,0.65490196
          ,0.65882353  ,0.6627451   ,0.66666667  ,0.67058824  ,0.6745098   ,0.67843137
          ,0.68235294  ,0.68627451  ,0.69019608  ,0.69411765  ,0.69803922  ,0.70196078
          ,0.70588235  ,0.70980392  ,0.71372549  ,0.71764706  ,0.72156863  ,0.7254902
          ,0.72941176  ,0.73333333  ,0.7372549   ,0.74117647  ,0.74509804  ,0.74901961
          ,0.75294118  ,0.75686275  ,0.76078431  ,0.76470588  ,0.76862745  ,0.77254902
          ,0.77647059  ,0.78039216  ,0.78431373  ,0.78823529  ,0.79215686  ,0.79607843
          ,0.8         ,0.80392157  ,0.80784314  ,0.81176471  ,0.81568627  ,0.81960784
          ,0.82352941  ,0.82745098  ,0.83137255  ,0.83529412  ,0.83921569  ,0.84313725
          ,0.84705882  ,0.85098039  ,0.85490196  ,0.85882353  ,0.8627451   ,0.86666667
          ,0.87058824  ,0.8745098   ,0.87843137  ,0.88235294  ,0.88627451  ,0.89019608
          ,0.89411765  ,0.89803922  ,0.90196078  ,0.90588235  ,0.90980392  ,0.91372549
          ,0.91764706  ,0.92156863  ,0.9254902   ,0.92941176  ,0.93333333  ,0.9372549
          ,0.94117647  ,0.94509804  ,0.94901961  ,0.95294118  ,0.95686275  ,0.96078431
          ,0.96470588  ,0.96862745  ,0.97254902  ,0.97647059  ,0.98039216  ,0.98431373
          ,0.98823529  ,0.99215686  ,0.99607843  ,1.        ] )
            colour=np.array( [[12.847665, 7.599765, 134.633625,255], [16.20168, 7.24863, 135.94662000000002,255], [19.215015, 6.937530000000001, 137.191785,255], [21.98661, 6.661874999999999, 138.37779,255], [24.576645000000003, 6.417075, 139.511265,255], [27.024900000000002, 6.1987950000000005, 140.59884,255], [29.35662, 6.00678, 141.64434,255], [31.595265, 5.833889999999999, 142.652865,255], [33.757155, 5.67579, 143.62875,255], [35.853765, 5.530185, 144.574545,255], [37.894785, 5.39427, 145.49331,255], [39.887355, 5.266005, 146.38657500000002,255], [41.837849999999996, 5.143605, 147.25689,255], [43.75137, 5.02503, 148.10553000000002,255], [45.63225, 4.90926, 148.93376999999998,255], [47.484314999999995, 4.794765, 149.74313999999998,255], [49.31037, 4.680269999999999, 150.53415,255], [51.113475, 4.56501, 151.30782,255], [52.895925000000005, 4.44771, 152.064915,255], [54.65925, 4.3281149999999995, 152.80594499999998,255], [56.405235000000005, 4.206735, 153.53116500000002,255], [58.135664999999996, 4.081785, 154.241085,255], [59.852325, 3.95301, 154.93596,255], [61.55598, 3.819645, 155.61604499999999,255], [63.24816, 3.6819450000000002, 156.28134,255], [64.929885, 3.53991, 156.931845,255], [66.601665, 3.3935400000000002, 157.567305,255], [68.26426500000001, 3.24258, 158.18822999999998,255], [69.918705, 3.087795, 158.79411,255], [71.56524, 2.92944, 159.38469,255], [73.20438, 2.768025, 159.960225,255], [74.83689000000001, 2.604315, 160.51995,255], [76.463025, 2.438055, 161.06412,255], [78.08355, 2.27001, 161.59197,255], [79.698465, 2.100945, 162.1035,255], [81.30828, 1.93188, 162.5982,255], [82.91325, 1.763325, 163.07556,255], [84.51363, 1.596555, 163.53558,255], [86.109165, 1.43259, 163.977495,255], [87.700875, 1.272705, 164.40105,255], [89.28825, 1.1174099999999998, 164.80599,255], [90.871545, 0.9684900000000001, 165.19155,255], [92.451015, 0.826965, 165.55747499999998,255], [94.02691499999999, 0.69462, 165.903255,255], [95.59873499999999, 0.572475, 166.22838000000002,255], [97.16698500000001, 0.46257000000000004, 166.53234,255], [98.731665, 0.36567, 166.815135,255], [100.29252, 0.28407, 167.07574499999998,255], [101.849805, 0.219045, 167.31391499999998,255], [103.403265, 0.17289, 167.529135,255], [104.9529, 0.14713500000000002, 167.72115000000002,255], [106.49871, 0.14382, 167.88945,255], [108.040695, 0.16473, 168.03378,255], [109.578345, 0.211905, 168.153375,255], [111.11217, 0.287385, 168.248235,255], [112.64166, 0.3927, 168.317595,255], [114.16707, 0.5304, 168.36120000000003,255], [115.687635, 0.7025250000000001, 168.37904999999998,255], [117.20386500000001, 0.91137, 168.370635,255], [118.71525000000001, 1.158975, 168.33544500000002,255], [120.221535, 1.44789, 168.273735,255], [121.72272, 1.7799, 168.18499500000001,255], [123.21855, 2.1573, 168.069225,255], [124.70902500000001, 2.582385, 167.92616999999998,255], [126.193635, 3.0574500000000002, 167.75557500000002,255], [127.67289, 3.584025, 167.55744,255], [129.14577, 4.164915, 167.33150999999998,255], [130.61253000000002, 4.802415, 167.078295,255], [132.072915, 5.498564999999999, 166.797795,255], [133.52641500000001, 6.255660000000001, 166.489755,255], [134.97303000000002, 7.0754850000000005, 166.15443,255], [136.41276, 7.960335000000001, 165.792075,255], [137.84535, 8.91225, 165.4032,255], [139.270035, 9.93327, 164.98755,255], [140.687325, 10.99968, 164.545635,255], [142.096965, 12.069405, 164.077965,255], [143.49819, 13.143975, 163.58479499999999,255], [144.891255, 14.22339, 163.066635,255], [146.27616, 15.307139999999999, 162.523995,255], [147.652395, 16.395480000000003, 161.95712999999998,255], [149.019705, 17.487645, 161.36706,255], [150.378345, 18.58389, 160.75404,255], [151.727805, 19.683449999999997, 160.118835,255], [153.06783, 20.78658, 159.46221,255], [154.39867500000003, 21.89277, 158.78493,255], [155.72008499999998, 23.00202, 158.08750500000002,255], [157.03206, 24.11382, 157.3707,255], [158.334345, 25.22817, 156.635535,255], [159.62668499999998, 26.34456, 155.882775,255], [160.909335, 27.463245, 155.11318500000002,255], [162.18204, 28.58346, 154.327275,255], [163.44454499999998, 29.70546, 153.52657499999998,255], [164.69736, 30.82899, 152.711085,255], [165.94023, 31.953795, 151.88233499999998,255], [167.17290000000003, 33.079875, 151.040835,255], [168.39537, 34.206720000000004, 150.18760500000002,255], [169.60789499999998, 35.33433, 149.32341000000002,255], [170.810475, 36.46296, 148.44926999999998,255], [172.00311, 37.591845, 147.56544,255], [173.1858, 38.72124, 146.673195,255], [174.35828999999998, 39.85089, 145.77329999999998,255], [175.52109, 40.980795, 144.866265,255], [176.6742, 42.110955000000004, 143.95310999999998,255], [177.81762, 43.241115, 143.03434499999997,255], [178.95109499999998, 44.371275, 142.11048,255], [180.07539, 45.501435, 141.182535,255], [181.189995, 46.63134, 140.25102,255], [182.29516500000003, 47.761244999999995, 139.31619,255], [183.391155, 48.890895, 138.379065,255], [184.47822, 50.02029, 137.440155,255], [185.55585000000002, 51.149429999999995, 136.499715,255], [186.62481, 52.278315, 135.558255,255], [187.684845, 53.40694499999999, 134.61654000000001,255], [188.736465, 54.53532, 133.67508,255], [189.77916, 55.66344, 132.73362,255], [190.813695, 56.791305, 131.79267000000002,255], [191.83956, 57.918915, 130.852995,255], [192.85752, 59.046525, 129.91434,255], [193.86732, 60.17388, 128.97746999999998,255], [194.869215, 61.30098, 128.04213,255], [195.86295, 62.428335000000004, 127.108575,255], [196.84929, 63.555434999999996, 126.17731500000001,255], [197.82798, 64.68279, 125.24860500000001,255], [198.79901999999998, 65.80989, 124.322445,255], [199.76266500000003, 66.9375, 123.39909,255], [200.71891499999998, 68.06511, 122.478285,255], [201.668025, 69.192975, 121.56003000000001,255], [202.609995, 70.32135000000001, 120.644835,255], [203.54508, 71.45023499999999, 119.73219,255], [204.473025, 72.57963, 118.82260500000001,255], [205.39408500000002, 73.709535, 117.91582500000001,255], [206.30826, 74.840205, 117.01185,255], [207.21606, 75.97164000000001, 116.11119000000001,255], [208.11672, 77.10384, 115.21308,255], [209.011005, 78.23706, 114.31803,255], [209.89866, 79.371555, 113.42553,255], [210.77994, 80.50707, 112.53558,255], [211.65459, 81.64386, 111.64818,255], [212.52261, 82.781925, 110.76333,255], [213.384255, 83.921775, 109.880775,255], [214.239525, 85.0629, 109.001025,255], [215.08842, 86.20581, 108.12331499999999,255], [215.93094, 87.350505, 107.24764499999999,255], [216.76683, 88.49724, 106.374015,255], [217.596345, 89.646015, 105.50216999999999,255], [218.41948499999998, 90.79683, 104.63211000000001,255], [219.23625, 91.94994, 103.763835,255], [220.046385, 93.105345, 102.897345,255], [220.84989000000002, 94.2633, 102.03213,255], [221.646765, 95.42406, 101.16819,255], [222.43726500000002, 96.58736999999999, 100.305525,255], [223.22088000000002, 97.753485, 99.44388,255], [223.997865, 98.92266, 98.583,255], [224.767965, 100.09489500000001, 97.723395,255], [225.53118, 101.27044500000001, 96.8643,255], [226.28751, 102.44931, 96.00597,255], [227.0367, 103.63149, 95.14815,255], [227.77875, 104.81724000000001, 94.29084,255], [228.513405, 106.00656000000001, 93.433785,255], [229.24092, 107.19995999999999, 92.57698500000001,255], [229.96078500000002, 108.397185, 91.72044,255], [230.67325499999998, 109.598235, 90.863895,255], [231.378075, 110.80362000000001, 90.00735,255], [232.07498999999999, 112.01334, 89.15055,255], [232.76399999999998, 113.227395, 88.294005,255], [233.445105, 114.445785, 87.43695,255], [234.117795, 115.668765, 86.57989500000001,255], [234.78207, 116.89633500000001, 85.72233,255], [235.438185, 118.129005, 84.864255,255], [236.085375, 119.366265, 84.00592499999999,255], [236.723895, 120.60862499999999, 83.147085,255], [237.35349, 121.856085, 82.287735,255], [237.97415999999998, 123.10889999999999, 81.427875,255], [238.58565, 124.36655999999999, 80.56776,255], [239.18744999999998, 125.63008500000001, 79.706625,255], [239.77981499999999, 126.89871, 78.845235,255], [240.36249, 128.172945, 77.98308,255], [240.93522000000002, 129.45279000000002, 77.12041500000001,255], [241.498005, 130.738245, 76.257495,255], [242.050335, 132.029565, 75.39381,255], [242.59271999999999, 133.32675, 74.530125,255], [243.12414, 134.6298, 73.665165,255], [243.64485000000002, 135.938715, 72.79995000000001,255], [244.154595, 137.25375, 71.93448000000001,255], [244.65312, 138.574905, 71.068755,255], [245.14068, 139.90218000000002, 70.202775,255], [245.61676500000002, 141.235575, 69.33679500000001,255], [246.08112, 142.57509, 68.470815,255], [246.53349, 143.92098000000001, 67.60509,255], [246.97413, 145.27349999999998, 66.738855,255], [247.402275, 146.63214, 65.87287500000001,255], [247.817925, 147.99741, 65.007405,255], [248.22108, 149.369055, 64.14269999999999,255], [248.61148500000002, 150.74707500000002, 63.278505,255], [248.98914, 152.131725, 62.415585,255], [249.35327999999998, 153.523005, 61.553684999999994,255], [249.704415, 154.92066, 60.693315,255], [250.04178, 156.324945, 59.83473,255], [250.36562999999998, 157.73586, 58.978184999999996,255], [250.675455, 159.153405, 58.123935,255], [250.97074500000002, 160.57809, 57.271724999999996,255], [251.251755, 162.00914999999998, 56.422574999999995,255], [251.517975, 163.447095, 55.57674,255], [251.76966, 164.891415, 54.735240000000005,255], [252.0063, 166.34287500000002, 53.897819999999996,255], [252.22764, 167.80096500000002, 53.0655,255], [252.433425, 169.265685, 52.239045000000004,255], [252.623655, 170.73729, 51.41871,255], [252.798075, 172.215525, 50.605515,255], [252.956175, 173.700645, 49.800225,255], [253.097955, 175.19265000000001, 49.003350000000005,255], [253.22316, 176.69128500000002, 48.21642,255], [253.33128, 178.19655, 47.440455,255], [253.42257, 179.70895499999997, 46.675965000000005,255], [253.496265, 181.22799, 45.924735,255], [253.55262, 182.753655, 45.18804,255], [253.59087, 184.286205, 44.467155000000005,255], [253.611015, 185.82564000000002, 43.76361,255], [253.613055, 187.371705, 43.079190000000004,255], [253.596225, 188.9244, 42.415425000000006,255], [253.56052499999998, 190.483725, 41.774355,255], [253.505955, 192.04993499999998, 41.15802,255], [253.432005, 193.62252, 40.56846,255], [253.33791, 195.202245, 40.007205,255], [253.22341500000002, 196.7886, 39.47604,255], [253.088775, 198.381585, 38.978024999999995,255], [252.933735, 199.980945, 38.515710000000006,255], [252.758295, 201.586935, 38.091135,255], [252.56194499999998, 203.19904499999998, 37.70685,255], [252.344685, 204.817275, 37.364895,255], [252.10523999999998, 206.442645, 37.066035,255], [251.843355, 208.07439, 36.812565,255], [251.55979499999998, 209.712255, 36.607034999999996,255], [251.25507000000002, 211.35573, 36.450975,255], [250.927905, 213.005325, 36.34464,255], [250.576515, 214.66206, 36.287265000000005,255], [250.20345, 216.323895, 36.281144999999995,255], [249.80921999999998, 217.99083000000002, 36.325514999999996,255], [249.388725, 219.66516, 36.416039999999995,255], [248.947575, 221.34408, 36.554505,255], [248.48296499999998, 223.02861000000001, 36.735555,255], [247.99515, 224.71875, 36.955365,255], [247.485915, 226.41348, 37.209345,255], [246.952965, 228.11382, 37.48857,255], [246.399105, 229.81849499999998, 37.7859,255], [245.825355, 231.52725, 38.08935,255], [245.228655, 233.24136000000001, 38.3826,255], [244.61538000000002, 234.958785, 38.64933,255], [243.98604, 236.67875999999998, 38.864295,255], [243.343185, 238.40153999999998, 38.994855,255], [242.69012999999998, 240.126105, 38.995875,255], [242.033505, 241.85092500000002, 38.80539,255], [241.38351, 243.57345, 38.333639999999995,255], [240.75876, 245.28858, 37.449555,255], [240.18347999999997, 246.99044999999998, 35.94378,255], [239.70382500000002, 248.66529, 33.48813,255]] ,dtype=np.ubyte)
            mode = 'rbg'
        # autumn
        elif cmap == 'Autumn':
            pos=np.array( [0.0,1.0] )
            colour=np.array( [[255,0,0,255], [255, 255,0,255]], dtype=np.ubyte)
            mode = 'rbg'
        # cool
        elif cmap == 'Cool':
            pos=np.array( [0.0,1.0] )
            colour=np.array( [[0.0, 255.0, 255.0, 255], [255.0,0.0, 255.0,255]], dtype=np.ubyte)
            mode = 'rbg'
        # bone
        elif cmap == 'Bone':
            pos=np.array( [0.0,0.365079,0.746032,1.0] )
            colour=np.array( [[0.0,0.0,0.0,255], [166.45838999999998, 198.33338999999998, 198.33337725002008,255], [81.45825187499999, 81.45822, 113.33322,255], [255.0, 255.0, 255.0,255]], dtype=np.ubyte)
            mode='rbg'
        # copper
        elif cmap == 'Copper':
            pos=np.array( [0.0,0.809524,1.0] )
            colour=np.array( [[0.0,0.0,0.0,255], [255.0, 161.262037944, 102.69823845,255], [255.0, 199.206, 126.8625,255]], dtype=np.ubyte)
            mode = 'rbg'
        # hot
        elif cmap == 'Hot':
            pos=np.array( [0.0,0.365079,0.746032,1.0] )
            colour=np.array( [[10.607999999999999,0.0,0.0,255], [255.0, 255.0, 255.0,255], [255.0,0.0,0.0,255], [255.0, 255.0,0.0,255]], dtype=np.ubyte)
            mode = 'rbg'
        # jet
        elif cmap == 'Jet':
            pos=np.array( [0.0,0.11,0.125,0.34,0.35,0.375,0.64,0.65,0.66,0.89,0.91, 1] )
            colour=np.array( [[0, 0, 127.5,255], [0.0, 0, 255.0,255], [20.56451612903227, 255, 226.20967741935488,255], [127.5, 0, 0,255], [0.0,0.0, 255,255], [255, 236.1111111111111,0.0,255], [0.0, 219.3, 255,255], [231.81818181818178, 0,0.0,255], [246.7741935483871, 245.55555555555554, 0,255], [238.54838709677418, 255, 8.225806451612923,255], [0, 229.49999999999997, 246.77419354838713,255], [255, 18.888888888888918, 0.0,255]], dtype=np.ubyte)
            mode = 'rbg'
        elif cmap == 'Thermal':
            pos = np.array([0., 1., 0.3333, 0.6666])
            colour = np.array([[0, 0, 0, 255], [255, 255, 255, 255], [185, 0, 0, 255], [255, 220, 0, 255]], dtype=np.ubyte)
            mode = 'rbg'
        elif cmap == 'Flame':
            pos = np.array([0., 1., 0.2, 0.5, 0.8])
            colour = np.array([[0, 0, 0, 255], [255, 255, 255, 255], [7, 0, 220, 255], [236, 0, 134, 255], [246, 246, 0, 255]],
                             dtype=np.ubyte)
            mode = 'rbg'
        elif cmap == 'Yellowy':
            # Yellowy
            pos = np.array([0., 1., 0.2328863796753704, 0.8362738179251941, 0.5257586450247])
            colour = np.array([[0, 0, 0, 255], [255, 255, 255, 255], [32, 0, 129, 255], [255, 255, 0, 255], [115, 15, 255, 255]],
                             dtype=np.ubyte)
            mode = 'rbg'
        elif cmap == 'Bipolar':
            pos = np.array([0., 1., 0.5, 0.25, 0.75])
            colour = np.array([[0, 255, 255, 255], [255, 255, 0, 255], [0, 0, 0, 255], [0, 0, 255, 255], [255, 0, 0, 255]],
                             dtype=np.ubyte)
            mode = 'rbg'
        elif cmap == 'Spectrum':
            # Spectrum
            pos = np.array([0., 1.])
            colour = np.array([[255, 0, 0, 255], [255, 0, 255, 255]],
                             dtype=np.ubyte)
            mode = 'hsv'
        else:
            print "No such colour map"

        cmap = pg.ColorMap(pos, colour,mode)
        lut = cmap.getLookupTable(0.0, 1.0, 256)

        self.specPlot.setLookupTable(lut)
        self.overviewImage.setLookupTable(lut)

    def invertColourMap(self):
        self.cmapInverted = not self.cmapInverted
        self.setColourLevels()

    def setColourLevels(self):
        # Choose the black and white levels
        minsg = np.min(self.sg)
        maxsg = np.max(self.sg)
        brightness = self.brightnessSlider.value()
        contrast = self.contrastSlider.value()
        colourStart = (brightness / 100.0 * contrast / 100.0) * (maxsg - minsg) + minsg
        colourEnd = (maxsg - minsg) * (1.0 - contrast / 100.0) + colourStart

        if self.cmapInverted:
            self.overviewImage.setLevels([colourEnd, colourStart])
            self.specPlot.setLevels([colourEnd, colourStart])
        else:
            self.overviewImage.setLevels([colourStart, colourEnd])
            self.specPlot.setLevels([colourStart, colourEnd])
# ===========
# Code for things at the top of the screen (overview figure and left/right buttons)
    def moveLeft(self):
        # When the left button is pressed (next to the overview plot), move everything along
        # Note the parameter to all a 10% overlap
        minX, maxX = self.overviewImageRegion.getRegion()
        newminX = max(0,minX-(maxX-minX)*0.9)
        self.overviewImageRegion.setRegion([newminX, newminX+maxX-minX])
        self.updateOverview()
        self.playPosition = int(self.convertSpectoAmpl(newminX)*1000.0)
        #print "Slider:", self.convertSpectoAmpl(newminX),self.convertSpectoAmpl(maxX)
        #self.setSliderLimits(1000*self.convertSpectoAmpl(newminX),1000*self.convertSpectoAmpl(maxX))

    def moveRight(self):
        # When the right button is pressed (next to the overview plot), move everything along
        # Note the parameter to allow a 10% overlap
        minX, maxX = self.overviewImageRegion.getRegion()
        newminX = min(np.shape(self.sg)[1]-(maxX-minX),minX+(maxX-minX)*0.9)
        self.overviewImageRegion.setRegion([newminX, newminX+maxX-minX])
        self.updateOverview()
        self.playPosition = int(self.convertSpectoAmpl(newminX)*1000.0)
        #print "Slider:", self.convertSpectoAmpl(newminX),self.convertSpectoAmpl(maxX)
        #self.setSliderLimits(1000*self.convertSpectoAmpl(newminX),1000*self.convertSpectoAmpl(maxX))

    def scroll(self):
        # When the slider is moved, change the position of the plot
        newminX = self.scrollSlider.value()
        print newminX
        #print newminX, self.scrollSlider.minimum(), self.scrollSlider.maximum()
        minX, maxX = self.overviewImageRegion.getRegion()
        self.overviewImageRegion.setRegion([newminX, newminX+maxX-minX])
        self.updateOverview()
        self.playPosition = int(self.convertSpectoAmpl(newminX)*1000.0)

    # def showSegments(self,seglen=0):
    #     # This plots the segments that are returned from any of the segmenters and adds them to the set of segments
    #     # If there are segments, show them
    #     for count in range(seglen,len(self.segments)):
    #         if self.segments[count][4] == 'None' or self.segments[count][4] == "Don't Know":
    #             facecolour = 'r'
    #         else:
    #             facecolour = 'b'
    #         a1R = self.a1.add_patch(pl.Rectangle((self.segments[count][0], np.min(self.audiodata)),
    #                                              self.segments[count][1] - self.segments[count][0],
    #                                              self.plotheight,
    #                                              facecolor=facecolour,
    #                                              alpha=0.5))
    #         a2R = self.a2.add_patch(pl.Rectangle((self.segments[count][0]*self.sampleRate / self.config['incr'], self.segments[count][2]),
    #                                              self.segments[count][1]*self.sampleRate / self.config['incr'] - self.segments[count][
    #                                                  0]*self.sampleRate / self.config['incr'],
    #                                              self.segments[count][3], facecolor=facecolour,
    #                                              alpha=0.5))
    #
    #         self.listRectanglesa1.append(a1R)
    #         self.listRectanglesa2.append(a2R)
    #         a1t = self.a1.text(self.segments[count][0], np.min(self.audiodata), self.segments[count][4])
    #         # The font size is a pain because the window is in pixels, so have to transform it
    #         # Force a redraw to make bounding box available
    #         self.canvasMain.draw()
    #         # fs = a1t.get_fontsize()
    #         width = a1t.get_window_extent().inverse_transformed(self.a1.transData).width
    #         #print width, self.segments[count][1] - self.segments[count][0]
    #         if width > self.segments[count][1] - self.segments[count][0]:
    #             a1t.set_fontsize(8)
    #
    #         self.a1text.append(a1t)
    #
    #     self.canvasMain.draw()

    def changeWidth(self, value):
        # This is the listener for the spinbox that decides the width of the main window.
        # It updates the top figure plots as the window width is changed.
        # Slightly annoyingly, it gets called when the value gets reset, hence the first line
        if not hasattr(self,'overviewImageRegion'):
            return
        self.windowSize = value

        #self.a1.set_xlim(self.windowStart, self.windowStart+self.windowSize)
        #self.a2.set_xlim(self.windowStart*self.sampleRate / self.config['incr'], (self.windowStart + self.windowSize)*self.sampleRate/self.config['incr'])

        # Redraw the highlight in the overview figure appropriately
        minX, maxX = self.overviewImageRegion.getRegion()
        newmaxX = self.convertAmpltoSpec(value)+minX
        self.overviewImageRegion.setRegion([minX, newmaxX])
        self.scrollSlider.setMaximum(np.shape(self.sg)[1]-self.convertAmpltoSpec(self.widthWindow.value()))
        #print "Slider:", self.convertSpectoAmpl(minX),self.convertSpectoAmpl(maxX)
        #self.setSliderLimits(1000*self.convertSpectoAmpl(minX),1000*self.convertSpectoAmpl(maxX))
        self.updateOverview()

# ===============
# Generate the various dialogs that match the buttons

    def humanClassifyDialog1(self):
        # Create the dialog that shows calls to the user for verification
        # Currently assumes that there is a selected box (later, use the first!)
        self.currentSegment = 0
        x1,x2 = self.listRectanglesa2[self.currentSegment].getRegion()
        x1 = int(x1)
        x2 = int(x2)
        self.humanClassifyDialog1 = HumanClassify1(self.sg[:,x1:x2],self.segments[self.currentSegment][4])
        self.humanClassifyDialog1.show()
        self.humanClassifyDialog1.activateWindow()
        self.humanClassifyDialog1.close.clicked.connect(self.humanClassifyClose1)
        self.humanClassifyDialog1.correct.clicked.connect(self.humanClassifyCorrect1)
        self.humanClassifyDialog1.wrong.clicked.connect(self.humanClassifyWrong1)

    def humanClassifyClose1(self):
        # Listener for the human verification dialog.
        self.humanClassifyDialog1.done(1)

    def humanClassifyNextImage1(self):
        # Get the next image
        # TODO: Ends rather suddenly
        if self.currentSegment != len(self.listRectanglesa2)-1:
            self.currentSegment += 1
            x1, x2 = self.listRectanglesa2[self.currentSegment].getRegion()
            self.humanClassifyDialog1.setImage(self.sg[:,x1:x2],self.segments[self.currentSegment][4])
        else:
            print "Last image"
            self.humanClassifyClose1()

    def humanClassifyCorrect1(self):
        self.humanClassifyNextImage1()

    def humanClassifyWrong1(self):
        # First get the correct classification (by producing a new modal dialog) and update the text, then show the next image
        # TODO: Test, particularly that new birds are added
        # TODO: update the listRects
        x1, x2 = self.listRectanglesa2[self.currentSegment].getRegion()
        self.correctClassifyDialog1 = CorrectHumanClassify1(self.sg[:,x1:x2],self.config['BirdButtons1'],self.config['BirdButtons2'], self.config['ListBirdsEntries'])
        label, self.saveConfig = self.correctClassifyDialog1.getValues(self.sg[:,x1:x2],self.config['BirdButtons1'],self.config['BirdButtons2'], self.config['ListBirdsEntries'])
        self.updateText(label,self.currentSegment)
        if self.saveConfig:
            self.config['ListBirdsEntries'].append(label)
        self.humanClassifyNextImage1()

    def humanClassifyDialog2(self):
        # Create the dialog that shows calls to the user for verification
        # Currently assumes that there is a selected box (later, use the first!)
        self.currentSegment = 0
        x1,x2 = self.listRectanglesa2[self.currentSegment].getRegion()
        x1 = int(x1)
        x2 = int(x2)
        self.humanClassifyDialog2 = HumanClassify2(self.sg[:,x1:x2],self.segments[self.currentSegment][4])
        self.humanClassifyDialog2.show()
        self.humanClassifyDialog2.activateWindow()

    def showSpectrogramDialog(self):
        # Create the spectrogram dialog when the button is pressed
        if not hasattr(self,'spectrogramDialog'):
            self.spectrogramDialog = Spectrogram()
        self.spectrogramDialog.show()
        self.spectrogramDialog.activateWindow()
        self.spectrogramDialog.activate.clicked.connect(self.spectrogram)
        # TODO: next line
        # self.connect(self.spectrogramDialog, SIGNAL("changed"), self.spectrogram)

    def spectrogram(self):
        # Listener for the spectrogram dialog.
        [alg, multitaper, window_width, incr] = self.spectrogramDialog.getValues()

        self.sp.set_width(int(str(window_width)), int(str(incr)))
        self.sgRaw = self.sp.spectrogram(self.audiodata,str(alg),multitaper=multitaper)
        self.sg = np.abs(np.where(self.sgRaw==0,0.0,10.0 * np.log10(self.sgRaw)))
        self.overviewImage.setImage(np.fliplr(self.sg.T))
        self.specPlot.setImage(np.fliplr(self.sg.T))

        # If the size of the spectrogram has changed, need to update the positions of things
        if int(str(incr)) != self.config['incr']:
            self.config['incr'] = int(str(incr))
            self.changeWidth(self.widthWindow.value())
            # Update the positions of the segments
            for s in range(len(self.listRectanglesa2)):
                x1 = self.convertAmpltoSpec(self.listRectanglesa1[s].getRegion()[0])
                x2 = self.convertAmpltoSpec(self.listRectanglesa1[s].getRegion()[1])
                self.listRectanglesa2[s].setRegion([x1, x2])
        if int(str(window_width)) != self.config['window_width']:
            self.config['window_width'] = int(str(window_width))
            # Update the axis
            self.specaxis.setTicks([[(0, 0), (np.shape(self.sg)[0] / 4, self.sampleRate / 8000),
                                     (np.shape(self.sg)[0] / 2, self.sampleRate / 4000),
                                     (3 * np.shape(self.sg)[0] / 4, 3 * self.sampleRate / 8000),
                                     (np.shape(self.sg)[0], self.sampleRate / 2000)]])


    def denoiseDialog(self):
        # Create the denoising dialog when the relevant button is pressed
        # TODO: Anything to help setting bandpass levels?
        self.denoiseDialog = Denoise()
        self.denoiseDialog.show()
        self.denoiseDialog.activateWindow()
        self.denoiseDialog.activate.clicked.connect(self.denoise)
        self.denoiseDialog.undo.clicked.connect(self.denoise_undo)
        self.denoiseDialog.save.clicked.connect(self.denoise_save)

    def backup(self):
        if hasattr(self, 'audiodata_backup'):
            if self.audiodata_backup is not None:
                audiodata_backup_new = np.empty(
                    (np.shape(self.audiodata_backup)[0], np.shape(self.audiodata_backup)[1] + 1))
                audiodata_backup_new[:, :-1] = np.copy(self.audiodata_backup)
                audiodata_backup_new[:, -1] = np.copy(self.audiodata)
                self.audiodata_backup = audiodata_backup_new
            else:
                self.audiodata_backup = np.empty((np.shape(self.audiodata)[0], 1))
                self.audiodata_backup[:, 0] = np.copy(self.audiodata)
        else:
            self.audiodata_backup = np.empty((np.shape(self.audiodata)[0], 1))
            self.audiodata_backup[:, 0] = np.copy(self.audiodata)

    def denoise(self):
        # Listener for the denoising dialog.
        # Calls the denoiser and then plots the updated data
        # TODO: should it be saved automatically, or a button added?
        [alg,depthchoice,depth,thrType,thr,wavelet,start,end,width] = self.denoiseDialog.getValues()
        # TODO: deal with these!
        # TODO: Undo needs testing
        self.backup()
        if str(alg) == "Wavelets":
            if thrType is True:
                type = 'Soft'
            else:
                type = 'Hard'
            if depthchoice:
                depth = None
            else:
                depth = int(str(depth))
            self.audiodata = self.sp.waveletDenoise(self.audiodata,type,float(str(thr)),depth,str(wavelet))
        elif str(alg) == "Bandpass --> Wavelets":
            if thrType is True:
                type = 'soft'
            else:
                type = 'hard'
            if depthchoice:
                depth = None
            else:
                depth = int(str(depth))
            self.audiodata = self.sp.bandpassFilter(self.audiodata,int(str(start)),int(str(end)))
            self.audiodata = self.sp.waveletDenoise(self.audiodata,type,float(str(thr)),depth,str(wavelet))
        elif str(alg) == "Wavelets --> Bandpass":
            if thrType is True:
                type = 'soft'
            else:
                type = 'hard'
            if depthchoice:
                depth = None
            else:
                depth = int(str(depth))
            self.audiodata = self.sp.waveletDenoise(self.audiodata,type,float(str(thr)),depth,str(wavelet))
            self.audiodata = self.sp.bandpassFilter(self.audiodata,int(str(start)),int(str(end)))
        #elif str(alg) == "Wavelets + Bandpass":
            #if thrType is True:
                #type = 'soft'
            #else:
                #type = 'hard'
            #if depthchoice:
                #depth = None
            #else:
                #depth = int(str(depth))
            #self.audiodata = self.sp.waveletDenoise(self.audiodata,float(str(thr)),int(str(depth)),str(wavelet))
            #self.audiodata = self.sp.bandpassFilter(self.audiodata,int(str(start)),int(str(end)))
        elif str(alg) == "Bandpass":
            self.audiodata = self.sp.bandpassFilter(self.audiodata, int(str(start)), int(str(end)))
        elif str(alg) == "Butterworth Bandpass":
            self.audiodata = self.sp.ButterworthBandpass(self.audiodata, self.sampleRate, low=int(str(start)), high=int(str(end)))
        else:
            #"Median Filter"
            self.audiodata = self.sp.medianFilter(self.audiodata,int(str(width)))

        self.sgRaw = self.sp.spectrogram(self.audiodata,self.sampleRate)
        self.sg = np.abs(np.where(self.sgRaw==0,0.0,10.0 * np.log10(self.sgRaw)))
        self.overviewImage.setImage(np.fliplr(self.sg.T))
        self.specPlot.setImage(np.fliplr(self.sg.T))
        self.amplPlot.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate*1000.0,num=self.datalength,endpoint=True),self.audiodata)

        self.setColourLevels()

    def denoise_undo(self):
        # Listener for undo button in denoising dialog
        # TODO: Can I actually delete something from an object?
        print("Undoing",np.shape(self.audiodata_backup))
        if hasattr(self,'audiodata_backup'):
            if self.audiodata_backup is not None:
                if np.shape(self.audiodata_backup)[1]>0:
                    self.audiodata = np.copy(self.audiodata_backup[:,-1])
                    self.audiodata_backup = self.audiodata_backup[:,:-1]
                    self.sp.setNewData(self.audiodata,self.sampleRate)
                    self.sgRaw = self.sp.spectrogram(self.audiodata,self.sampleRate)
                    self.sg = np.abs(np.where(self.sgRaw == 0, 0.0, 10.0 * np.log10(self.sgRaw)))
                    self.overviewImage.setImage(np.fliplr(self.sg.T))
                    self.specPlot.setImage(np.fliplr(self.sg.T))
                    self.amplPlot.setData(
                        np.linspace(0.0, float(self.datalength) / self.sampleRate, num=self.datalength, endpoint=True),
                        self.audiodata)
                    if hasattr(self,'seg'):
                        self.seg.setNewData(self.audiodata,self.sgRaw,self.sampleRate)

                    self.setColourLevels()

                    #self.drawfigMain()

    def denoise_save(self):
        # Listener for save button in denoising dialog
        # Save denoised data
        # Other players need them to be 16 bit, which is this magic number
        #self.audiodata *= 32768.0
        #self.audiodata = self.audiodata.astype('int16')
        #import soundfile as sf
        filename = self.filename[:-4] + '_d' + self.filename[-4:]
        self.audiodata = self.audiodata.astype('int16')
        wavfile.write(filename,self.sampleRate, self.audiodata)
        #sf.write(filename,self.audiodata,self.sampleRate,subtype='PCM_16')

    def segmentationDialog(self):
        # Create the segmentation dialog when the relevant button is pressed
        self.segmentDialog = Segmentation(np.max(self.audiodata))
        self.segmentDialog.show()
        self.segmentDialog.activateWindow()
        self.segmentDialog.activate.clicked.connect(self.segment)
        #self.segmentDialog.save.clicked.connect(self.segments_save)

    def segment(self):
        # Listener for the segmentation dialog
        # TODO: Currently just gives them all the label 'None'
        # TODO: Add in the wavelet one
        # TODO: More testing of the algorithms, parameters, etc.
        seglen = len(self.segments)
        [alg, ampThr, medThr,HarmaThr1,HarmaThr2,minfreq,minperiods,Yinthr,window,depth,thrType,thr,wavelet,bandchoice,start,end] = self.segmentDialog.getValues()
        if not hasattr(self,'seg'):
            self.seg = Segment.Segment(self.audiodata,self.sgRaw,self.sp,self.sampleRate,self.config['minSegment'])
        if str(alg) == "Amplitude":
            newSegments = self.seg.segmentByAmplitude(float(str(ampThr)))
            # TODO: *** Next few lines need updating
            #if hasattr(self, 'line'):
            #    if self.line is not None:
            #        self.line.remove()
            #self.line = self.a1.add_patch(pl.Rectangle((0,float(str(ampThr))),len(self.audiodata),0,facecolor='r'))
        elif str(alg) == "Median Clipping":
            newSegments = self.seg.medianClip(float(str(medThr)))
            #print newSegments
        elif str(alg) == "Harma":
            newSegments = self.seg.Harma(float(str(HarmaThr1)),float(str(HarmaThr2)))
        elif str(alg) == "Onsets":
            newSegments = self.seg.onsets()
            #print newSegments
        elif str(alg) == "Fundamental Frequency":
            newSegments, pitch, times = self.seg.yin(int(str(minfreq)),int(str(minperiods)),float(str(Yinthr)),int(str(window)),returnSegs=True)
            #print newSegments
        else:
            #"Wavelets"
            # TODO!!
            if thrType is True:
                type = 'soft'
            else:
                type = 'hard'
            if bandchoice:
                start = None
                end = None
            else:
                start = int(str(start))
                end = int(str(end))
            # TODO: needs learning and samplerate
                #newSegments = self.seg.segmentByWavelet(thrType,float(str(thr)), int(str(depth)), wavelet,sampleRate,bandchoice,start,end,learning,)
        for seg in newSegments:
            self.addSegment(seg[0],seg[1],0,0,'None',True)

    def findMatches(self):
        # Calls the cross-correlation function to find matches like the currently highlighted box
        # TODO: Other methods apart from c-c?
        # So there needs to be a currently highlighted box
        # TODO: Tell user if there isn't a box highlighted, or grey out the button
        #print self.box1id
        if not hasattr(self,'seg'):
            self.seg = Segment.Segment(self.audiodata,self.sgRaw,self.sp,self.sampleRate,self.config['minSegment'])

        if self.box1id is None or self.box1id == -1:
            print "No box selected"
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No segment selected to match")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        else:
            #[alg, thr] = self.matchDialog.getValues()
            # Only want to draw new segments, so find out how many there are now
            seglen = len(self.segments)
            # Get the segment -- note that takes the full y range
            # TODO: is this correct?
            if type(self.listRectanglesa2[self.box1id]) == self.ROItype:
                x1 = self.listRectanglesa2[self.box1id].pos().x()
                x2 = x1 + self.listRectanglesa2[self.box1id].size().x()
            else:
                x1, x2 = self.listRectanglesa2[self.box1id].getRegion()
            print x1, x2
            segment = self.sgRaw[:,int(x1):int(x2)]
            len_seg = (x2-x1) * self.config['incr'] / self.sampleRate
            indices = self.seg.findCCMatches(segment,self.sgRaw,self.config['corrThr'])
            # indices are in spectrogram pixels, need to turn into times
            for i in indices:
                # Miss out the one selected: note the hack parameter
                if np.abs(i-x1) > self.config['overlap_allowed']:
                    time = float(i)*self.config['incr'] / self.sampleRate
                    self.addSegment(time, time+len_seg,0,0,self.segments[self.box1id][4])

    def recognise(self):
        # This will eventually call methods to do automatic recognition
        # Actually, will produce a dialog to ask which species, etc.
        # TODO
        pass

# ===============
# Code for playing sounds
    # These functions are the phonon playing code
    # Note that if want to play e.g. denoised one, will have to save it and then reload it
    def playSegment(self):
        self.segmentStop = self.playSlider.maximum()
        if self.media_obj.state() == phonon.Phonon.PlayingState:
            self.media_obj.pause()
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
            #self.playButton.setText("Play")
        elif self.media_obj.state() == phonon.Phonon.PausedState or self.media_obj.state() == phonon.Phonon.StoppedState:
            self.media_obj.play()
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
            #self.playButton.setText("Pause")

    def playFinished(self):
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.bar.setValue(0)
        #self.playButton.setText("Play")

    def sliderMoved(self):
        # When the slider is moved, change the position of playback
        self.media_obj.seek(self.playSlider.value())
        # playSlider.value() is in ms, need to convert this into spectrogram pixels
        self.bar.setValue(self.convertAmpltoSpec(self.playSlider.value()/1000.0))

    def barMoved(self,evt):
        self.playSlider.setValue(self.convertSpectoAmpl(evt.x())*1000)
        self.media_obj.seek(self.convertSpectoAmpl(evt.x())*1000)

    def movePlaySlider(self, time):
        if not self.playSlider.isSliderDown():
            self.playSlider.setValue(time)
        self.timePlayed.setText(self.convertMillisecs(time)+"/"+self.totalTime)
        if time > min(self.playSlider.maximum(),self.segmentStop):
            self.media_obj.stop()
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
            self.media_obj.seek(self.playSlider.minimum())
            #val = 60.*(time / (1000 * 60)) % 60 + (time / 1000) % 60 + time/1000.
        self.bar.setValue(self.convertAmpltoSpec(self.playSlider.value()/1000.0))

    def setSliderLimits(self, start,end):
        self.playSlider.setRange(start, end)
        self.playSlider.setValue(start)
        self.segmentStop = self.playSlider.maximum()
        self.media_obj.seek(start)

    def playSelectedSegment(self):
        # Get selected segment start and end (or return if no segment selected)
        # TODO: check if has been made pauseable
        if self.box1id > -1:
            start = self.listRectanglesa1[self.box1id].getRegion()[0]*1000
            self.segmentStop = self.listRectanglesa1[self.box1id].getRegion()[1]*1000
            self.media_obj.seek(start)
            #self.media_obj.play()
            #self.segmentStop = self.playSlider.maximum()
            if self.media_obj.state() == phonon.Phonon.PlayingState:
                self.media_obj.pause()
                self.playButton.setIcon(QtGui.QIcon('img/playsegment.png'))
                # self.playButton.setText("Play")
            elif self.media_obj.state() == phonon.Phonon.PausedState or self.media_obj.state() == phonon.Phonon.StoppedState:
                self.media_obj.play()
                self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))

    def playBandLimitedSegment(self):
        # Get the band limits of the segment, bandpass filter, then play that
        # TODO: This version uses sounddevice to play it back because the phonon needed to save it and then still wouldn't actually
        # play it. Does it matter? You can't see the bar moving.
        import sounddevice as sd
        start = int(self.listRectanglesa1[self.box1id].getRegion()[0]*self.sampleRate)
        stop = int(self.listRectanglesa1[self.box1id].getRegion()[1]*self.sampleRate)
        bottom = int(self.segments[self.box1id][2]*self.sampleRate/2./np.shape(self.sg)[0])
        top = int(self.segments[self.box1id][3]*self.sampleRate/2./np.shape(self.sg)[0])
        data = self.audiodata[start:stop]
        data = self.sp.bandpassFilter(data, bottom, top)

        sd.play(data,self.sampleRate)
        # TODO!! Why won't this actually play back? The file exists, and it will play if you load it
        # So there is something odd about the media_obj
        # Kludge: save the file, load, play, put it back
        # Save
        #filename = 'temp.wav'
        #data = data.astype('int16')
        #wavfile.write(filename,self.sampleRate, data)

        #sr, data = wavfile.read(filename)

        #self.media_obj.setCurrentSource(phonon.Phonon.MediaSource(filename))
        #self.media_obj.tick.connect(self.movePlaySlider)

        #print "got here"
        #print self.convertMillisecs(self.media_obj.totalTime())
        #self.media_obj.play()

        #self.media_obj.setCurrentSource(phonon.Phonon.MediaSource(self.filename))
        #self.media_obj.tick.connect(self.movePlaySlider)

    #def changeBrightness(self, brightness):
    #    self.colourChange(brightness=brightness)

    #def changeContrast(self, contrast):
    #    self.colourChange(contrast=contrast)

    #def swappedBW(self):
    #    self.config['coloursInverted'] = self.swapBW.checkState()
    #    self.colourChange()

    def colourChange(self):
        # These are brightness and contrast
        # Changing brightness moves the blackpoint and whitepoint by the same amount
        # Changing contrast widens the gap between blackpoint and whitepoint
        brightness = self.brightnessSlider.value()
        contrast = self.contrastSlider.value()

        self.setColourLevels()

                # ============
# Code for the buttons
    def deleteSegment(self):
        # Listener for delete segment button, or backspace key
        # Deletes segment if one is selected, otherwise does nothing
        if self.box1id>-1:
            self.p_ampl.removeItem(self.listRectanglesa1[self.box1id])
            self.p_spec.removeItem(self.listRectanglesa2[self.box1id])
            if self.useAmplitude:
                self.p_ampl.removeItem(self.listLabels[self.box1id])
            else:
                self.p_spec.removeItem(self.listLabels[self.box1id])
            del self.listLabels[self.box1id]
            del self.segments[self.box1id]
            del self.listRectanglesa1[self.box1id]
            del self.listRectanglesa2[self.box1id]
            self.box1id = -1

    def deleteAll(self,force=False):
        # Listener for delete all button
        if not force:
            reply = QMessageBox.question(self,"Delete All Segments","Are you sure you want to delete all segments?",    QMessageBox.Yes | QMessageBox.No)
        else:
            reply = QMessageBox.Yes
        if reply==QMessageBox.Yes:
            self.segments=[]
            for r in self.listLabels:
                if self.useAmplitude:
                    self.p_ampl.removeItem(r)
                else:
                    self.p_spec.removeItem(r)
            for r in self.listRectanglesa1:
                self.p_ampl.removeItem(r)
            for r in self.listRectanglesa2:
                self.p_spec.removeItem(r)
            self.listRectanglesa1 = []
            self.listRectanglesa2 = []
            self.listLabels = []
            self.box1id = -1

    def saveSegments(self):
        # This saves the segmentation data as a json file
        if len(self.segments)>0 or self.hasSegments:
            print("Saving segments to "+self.filename)
            if isinstance(self.filename, str):
                file = open(self.filename + '.data', 'w')
            else:
                file = open(str(self.filename) + '.data', 'w')
            json.dump(self.segments,file)

    def closeEvent(self, event):
        # Catch the user closing the window by clicking the Close button instead of quitting
        self.quit()

    def quit(self):
        # Listener for the quit button
        print("Quitting")
        self.saveSegments()
        if self.saveConfig == True:
            print "Saving config file"
            json.dump(self.config, open(self.configfile, 'wb'))
        QApplication.quit()

# =============
# Helper functions

    def splitFile5mins(self, name):
        # Nirosha wants to split files that are long (15 mins) into 5 min segments
        # Could be used when loading long files :)
        # TODO: put paging in here, possible in librosa
        try:
            self.audiodata, self.sampleRate = lr.load(name,sr=None)
        except:
            print("Error: try another file")
        nsamples = np.shape(self.audiodata)[0]
        lengthwanted = self.sampleRate * 60 * 5
        count = 0
        while (count + 1) * lengthwanted < nsamples:
            data = self.audiodata[count * lengthwanted:(count + 1) * lengthwanted]
            filename = name[:-4] + '_' +str(count) + name[-4:]
            lr.output.write_wav(filename, data, self.sampleRate)
            count += 1
        data = self.audiodata[(count) * lengthwanted:]
        filename = name[:-4] + '_' + str((count)) + name[-4:]
        lr.output.write_wav(filename,data,self.sampleRate)

# ===============
# Classes for the dialog boxes. Since most of them just get user selections, they are mostly just a mess of UI things
class Spectrogram(QDialog):
    # Class for the spectrogram dialog box
    # TODO: Steal the graph from Raven (View/Configure Brightness)
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Spectrogram Options')

        self.algs = QComboBox()
        self.algs.addItems(['Hann','Parzen','Welch','Hamming','Blackman','BlackmanHarris'])

        self.multitaper = QCheckBox()

        self.activate = QPushButton("Update Spectrogram")

        self.window_width = QLineEdit(self)
        self.window_width.setText('256')
        self.incr = QLineEdit(self)
        self.incr.setText('128')

        Box = QVBoxLayout()
        Box.addWidget(self.algs)
        Box.addWidget(QLabel('Multitapering'))
        Box.addWidget(self.multitaper)
        Box.addWidget(QLabel('Window Width'))
        Box.addWidget(self.window_width)
        Box.addWidget(QLabel('Hop'))
        Box.addWidget(self.incr)
        Box.addWidget(self.activate)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        return [self.algs.currentText(),self.multitaper.checkState(),self.window_width.text(),self.incr.text()]

    # def closeEvent(self, event):
    #     msg = QMessageBox()
    #     msg.setIcon(QMessageBox.Question)
    #     msg.setText("Do you want to keep the new values?")
    #     msg.setWindowTitle("Closing Spectrogram Dialog")
    #     msg.setStandardButtons(QMessageBox.Yes|QMessageBox.No)
    #     msg.buttonClicked.connect(self.resetValues)
    #     msg.exec_()
    #     return

    # def resetValues(self,button):
    #     print button.text()

class Segmentation(QDialog):
    # Class for the segmentation dialog box
    # TODO: add the wavelet params
    # TODO: work out how to return varying size of params, also process them
    # TODO: test and play
    def __init__(self, maxv, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Segmentation Options')

        self.algs = QComboBox()
        #self.algs.addItems(["Amplitude","Energy Curve","Harma","Median Clipping","Wavelets"])
        self.algs.addItems(["Amplitude","Harma","Median Clipping","Onsets","Fundamental Frequency"])
        self.algs.currentIndexChanged[QString].connect(self.changeBoxes)
        self.prevAlg = "Amplitude"
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
        self.HarmaThr1.setRange(10,50)
        self.HarmaThr1.setSingleStep(1)
        self.HarmaThr1.setValue(30)
        self.HarmaThr2 = QDoubleSpinBox()
        self.HarmaThr2.setRange(0.1,0.95)
        self.HarmaThr2.setSingleStep(0.05)
        self.HarmaThr2.setDecimals(2)
        self.HarmaThr2.setValue(0.9)

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

        self.ecThr = QDoubleSpinBox()
        self.ecThr.setRange(0.001,6)
        self.ecThr.setSingleStep(1)
        self.ecThr.setDecimals(3)
        self.ecThr.setValue(1)

        Box = QVBoxLayout()
        Box.addWidget(self.algs)
        # Labels
        self.amplabel = QLabel("Set threshold amplitude")
        Box.addWidget(self.amplabel)

        self.Harmalabel = QLabel("Set decibal threshold")
        Box.addWidget(self.Harmalabel)
        self.Harmalabel.hide()

        self.Onsetslabel = QLabel("Onsets: No parameters")
        Box.addWidget(self.Onsetslabel)
        self.Onsetslabel.hide()

        self.medlabel = QLabel("Set median threshold")
        Box.addWidget(self.medlabel)
        self.medlabel.hide()

        self.eclabel = QLabel("Set energy curve threshold")
        Box.addWidget(self.eclabel)
        self.eclabel.hide()
        self.ecthrtype = [QRadioButton("N standard deviations"), QRadioButton("Threshold")]

        self.wavlabel = QLabel("Wavelets")
        self.depthlabel = QLabel("Depth of wavelet packet decomposition")
        #self.depthchoice = QCheckBox()
        #self.connect(self.depthchoice, SIGNAL('clicked()'), self.depthclicked)
        self.depth = QSpinBox()
        self.depth.setRange(1,10)
        self.depth.setSingleStep(1)
        self.depth.setValue(5)

        self.thrtypelabel = QLabel("Type of thresholding")
        self.thrtype = [QRadioButton("Soft"), QRadioButton("Hard")]
        self.thrtype[0].setChecked(True)

        self.thrlabel = QLabel("Multiplier of std dev for threshold")
        self.thr = QSpinBox()
        self.thr.setRange(1,10)
        self.thr.setSingleStep(1)
        self.thr.setValue(5)

        self.waveletlabel = QLabel("Type of wavelet")
        self.wavelet = QComboBox()
        self.wavelet.addItems(["dmey","db2","db5","haar"])
        self.wavelet.setCurrentIndex(0)

        self.blabel = QLabel("Start and end points of the band for bandpass filter")
        self.start = QLineEdit()
        self.start.setText('1000')
        self.end = QLineEdit()
        self.end.setText('7500')
        self.blabel2 = QLabel("Check if not using bandpass")
        self.bandchoice = QCheckBox()
        self.connect(self.bandchoice, SIGNAL('clicked()'), self.bandclicked)


        Box.addWidget(self.wavlabel)
        self.wavlabel.hide()
        Box.addWidget(self.depthlabel)
        self.depthlabel.hide()
        #Box.addWidget(self.depthchoice)
        #self.depthchoice.hide()
        Box.addWidget(self.depth)
        self.depth.hide()

        Box.addWidget(self.thrtypelabel)
        self.thrtypelabel.hide()
        Box.addWidget(self.thrtype[0])
        self.thrtype[0].hide()
        Box.addWidget(self.thrtype[1])
        self.thrtype[1].hide()

        Box.addWidget(self.thrlabel)
        self.thrlabel.hide()
        Box.addWidget(self.thr)
        self.thr.hide()

        Box.addWidget(self.waveletlabel)
        self.waveletlabel.hide()
        Box.addWidget(self.wavelet)
        self.wavelet.hide()

        Box.addWidget(self.blabel)
        self.blabel.hide()
        Box.addWidget(self.start)
        self.start.hide()
        Box.addWidget(self.end)
        self.end.hide()
        Box.addWidget(self.blabel2)
        self.blabel2.hide()
        Box.addWidget(self.bandchoice)
        self.bandchoice.hide()

        Box.addWidget(self.ampThr)
        Box.addWidget(self.HarmaThr1)
        Box.addWidget(self.HarmaThr2)
        self.HarmaThr1.hide()
        self.HarmaThr2.hide()
        Box.addWidget(self.medThr)
        self.medThr.hide()
        for i in range(len(self.ecthrtype)):
            Box.addWidget(self.ecthrtype[i])
            self.ecthrtype[i].hide()
        Box.addWidget(self.ecThr)
        self.ecThr.hide()

        Box.addWidget(self.Fundminfreqlabel)
        self.Fundminfreqlabel.hide()
        Box.addWidget(self.Fundminfreq)
        self.Fundminfreq.hide()
        Box.addWidget(self.Fundminperiodslabel)
        self.Fundminperiodslabel.hide()
        Box.addWidget(self.Fundminperiods)
        self.Fundminperiods.hide()
        Box.addWidget(self.Fundthrlabel)
        self.Fundthrlabel.hide()
        Box.addWidget(self.Fundthr)
        self.Fundthr.hide()
        Box.addWidget(self.Fundwindowlabel)
        self.Fundwindowlabel.hide()
        Box.addWidget(self.Fundwindow)
        self.Fundwindow.hide()

        Box.addWidget(self.activate)
        #Box.addWidget(self.save)

        # Now put everything into the frame
        self.setLayout(Box)

    def changeBoxes(self,alg):
        # This does the hiding and showing of the options as the algorithm changes
        if self.prevAlg == "Amplitude":
            self.amplabel.hide()
            self.ampThr.hide()
        elif self.prevAlg == "Energy Curve":
            self.eclabel.hide()
            self.ecThr.hide()
            for i in range(len(self.ecthrtype)):
                self.ecthrtype[i].hide()
            #self.ecThr.hide()
        elif self.prevAlg == "Harma":
            self.Harmalabel.hide()
            self.HarmaThr1.hide()
            self.HarmaThr2.hide()
        elif self.prevAlg == "Median Clipping":
            self.medlabel.hide()
            self.medThr.hide()
        elif self.prevAlg == "Fundamental Frequency":
            self.Fundminfreq.hide()
            self.Fundminperiods.hide()
            self.Fundthr.hide()
            self.Fundwindow.hide()
            self.Fundminfreqlabel.hide()
            self.Fundminperiodslabel.hide()
            self.Fundthrlabel.hide()
            self.Fundwindowlabel.hide()
        elif self.prevAlg == "Onsets":
            # Don't need to do anything
            self.Onsetslabel.hide()
        else:
            self.wavlabel.hide()
            self.depthlabel.hide()
            self.depth.hide()
            #self.depthchoice.hide()
            self.thrtypelabel.hide()
            self.thrtype[0].hide()
            self.thrtype[1].hide()
            self.thrlabel.hide()
            self.thr.hide()
            self.waveletlabel.hide()
            self.wavelet.hide()
            self.blabel.hide()
            self.start.hide()
            self.end.hide()
            self.blabel2.hide()
            self.bandchoice.hide()
        self.prevAlg = str(alg)
        if str(alg) == "Amplitude":
            self.amplabel.show()
            self.ampThr.show()
        elif str(alg) == "Energy Curve":
            self.eclabel.show()
            self.ecThr.show()
            for i in range(len(self.ecthrtype)):
                self.ecthrtype[i].show()
            self.ecThr.show()
        elif str(alg) == "Harma":
            self.Harmalabel.show()
            self.HarmaThr1.show()
            self.HarmaThr2.show()
        elif str(alg) == "Median Clipping":
            self.medlabel.show()
            self.medThr.show()
        elif str(alg) == "Fundamental Frequency":
            self.Fundminfreq.show()
            self.Fundminperiods.show()
            self.Fundthr.show()
            self.Fundwindow.show()
            self.Fundminfreqlabel.show()
            self.Fundminperiodslabel.show()
            self.Fundthrlabel.show()
            self.Fundwindowlabel.show()
        elif str(alg) == "Onsets":
            self.Onsetslabel.show()
        else:
            #"Wavelets"
            self.wavlabel.show()
            self.depthlabel.show()
            #self.depthchoice.show()
            self.depth.show()
            self.thrtypelabel.show()
            self.thrtype[0].show()
            self.thrtype[1].show()
            self.thrlabel.show()
            self.thr.show()
            self.waveletlabel.show()
            self.wavelet.show()
            self.blabel.show()
            self.start.show()
            self.end.show()
            self.blabel2.show()
            self.bandchoice.show()

    def bandclicked(self):
        # TODO: Can they be grayed out?
        self.start.setEnabled(not self.start.isEnabled())
        self.end.setEnabled(not self.end.isEnabled())

    def getValues(self):
        return [self.algs.currentText(),self.ampThr.text(),self.medThr.text(),self.HarmaThr1.text(),self.HarmaThr2.text(),self.Fundminfreq.text(),self.Fundminperiods.text(),self.Fundthr.text(),self.Fundwindow.text(),self.depth.text(),self.thrtype[0].isChecked(),self.thr.text(),self.wavelet.currentText(),self.bandchoice.isChecked(),self.start.text(),self.end.text()]

class Denoise(QDialog):
    # Class for the denoising dialog box
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Denoising Options')

        self.algs = QComboBox()
        self.algs.addItems(["Wavelets","Bandpass", "Wavelets --> Bandpass","Bandpass --> Wavelets","Median Filter"])
        self.algs.currentIndexChanged[QString].connect(self.changeBoxes)
        self.prevAlg = "Wavelets"

        # Wavelet: Depth of tree, threshold type, threshold multiplier, wavelet
        self.wavlabel = QLabel("Wavelets")
        self.depthlabel = QLabel("Depth of wavelet packet decomposition (or tick box to use best)")
        self.depthchoice = QCheckBox()
        self.connect(self.depthchoice, SIGNAL('clicked()'), self.depthclicked)
        self.depth = QSpinBox()
        self.depth.setRange(1,10)
        self.depth.setSingleStep(1)
        self.depth.setValue(5)

        self.thrtypelabel = QLabel("Type of thresholding")
        self.thrtype = [QRadioButton("Soft"), QRadioButton("Hard")]
        self.thrtype[0].setChecked(True)

        self.thrlabel = QLabel("Multiplier of std dev for threshold")
        self.thr = QSpinBox()
        self.thr.setRange(1,10)
        self.thr.setSingleStep(1)
        self.thr.setValue(5)

        self.waveletlabel = QLabel("Type of wavelet")
        self.wavelet = QComboBox()
        self.wavelet.addItems(["dmey","db2","db5","haar"])
        self.wavelet.setCurrentIndex(0)

        # Median: width of filter
        self.medlabel = QLabel("Median Filter")
        self.widthlabel = QLabel("Half width of median filter")
        self.width = QSpinBox()
        self.width.setRange(1,101)
        self.width.setSingleStep(1)
        self.width.setValue(11)

        # Bandpass: start and end
        self.bandlabel = QLabel("Bandpass Filter")
        self.wblabel = QLabel("Wavelets and Bandpass Filter")
        self.blabel = QLabel("Start and end points of the band")
        self.start = QLineEdit(self)
        self.start.setText('1000')
        self.end = QLineEdit(self)
        self.end.setText('7500')

        # Want combinations of these too!

        self.activate = QPushButton("Denoise")
        self.undo = QPushButton("Undo")
        self.save = QPushButton("Save Denoised Sound")
        #self.connect(self.undo, SIGNAL('clicked()'), self.undo)
        Box = QVBoxLayout()
        Box.addWidget(self.algs)

        Box.addWidget(self.wavlabel)
        Box.addWidget(self.depthlabel)
        Box.addWidget(self.depthchoice)
        Box.addWidget(self.depth)

        Box.addWidget(self.thrtypelabel)
        Box.addWidget(self.thrtype[0])
        Box.addWidget(self.thrtype[1])

        Box.addWidget(self.thrlabel)
        Box.addWidget(self.thr)

        Box.addWidget(self.waveletlabel)
        Box.addWidget(self.wavelet)

        # Median: width of filter
        Box.addWidget(self.medlabel)
        self.medlabel.hide()
        Box.addWidget(self.widthlabel)
        self.widthlabel.hide()
        Box.addWidget(self.width)
        self.width.hide()

        # Bandpass: start and end
        Box.addWidget(self.bandlabel)
        self.bandlabel.hide()
        Box.addWidget(self.wblabel)
        self.wblabel.hide()
        Box.addWidget(self.blabel)
        self.blabel.hide()
        Box.addWidget(self.start)
        self.start.hide()
        Box.addWidget(self.end)
        self.end.hide()

        Box.addWidget(self.activate)
        Box.addWidget(self.undo)
        Box.addWidget(self.save)

        # Now put everything into the frame
        self.setLayout(Box)

    def changeBoxes(self,alg):
        # This does the hiding and showing of the options as the algorithm changes
        if self.prevAlg == "Wavelets":
            self.wavlabel.hide()
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
        elif self.prevAlg == "Bandpass --> Wavelets":
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
            self.start.hide()
            self.end.hide()
            self.medlabel.hide()
            self.widthlabel.hide()
            self.width.hide()
        elif self.prevAlg == "Wavelets --> Bandpass":
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
            self.start.hide()
            self.end.hide()
            self.medlabel.hide()
            self.widthlabel.hide()
            self.width.hide()
        elif self.prevAlg == "Bandpass" or self.prevAlg == "Butterworth Bandpass":
            self.bandlabel.hide()
            self.blabel.hide()
            self.start.hide()
            self.end.hide()
        else:
            # Median filter
            self.medlabel.hide()
            self.widthlabel.hide()
            self.width.hide()

        self.prevAlg = str(alg)
        if str(alg) == "Wavelets":
            self.wavlabel.show()
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
        elif str(alg) == "Wavelets --> Bandpass":
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
            self.start.show()
            self.end.show()
        elif str(alg) == "Bandpass --> Wavelets":
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
            self.start.show()
            self.end.show()
        elif str(alg) == "Bandpass" or str(alg) == "Butterworth Bandpass":
            self.bandlabel.show()
            self.start.show()
            self.end.show()
        else:
            #"Median filter"
            self.medlabel.show()
            self.widthlabel.show()
            self.width.show()

    def depthclicked(self):
        self.depth.setEnabled(not self.depth.isEnabled())

    def getValues(self):
        return [self.algs.currentText(),self.depthchoice.isChecked(),self.depth.text(),self.thrtype[0].isChecked(),self.thr.text(),self.wavelet.currentText(),self.start.text(),self.end.text(),self.width.text()]

class HumanClassify1(QDialog):
    # This dialog is different to the others. The aim is to check (or ask for) classifications for segments.
    # This version shows a single segment at a time, working through all the segments.
    # So it needs to show a segment, and its current label
    # Has tick and cross, and the tick loads the next one, cross asks for new one
    # TODO: Could have the label options in here and preselect it, then have the tick or change it.
    # Better for things that aren't labelled already?

    def __init__(self, seg, label, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Check Classifications')
        self.frame = QWidget()
        #self.cmap_grey = cmap_grey

        # Set up the plot windows, then the right and wrong buttons, and a close button

        self.wPlot = pg.GraphicsLayoutWidget()
        self.pPlot = self.wPlot.addViewBox(enableMouse=False,row=0,col=1)
        self.plot = pg.ImageItem()
        self.pPlot.addItem(self.plot)

        self.species = QLabel(label)

        # The buttons to move through the overview
        # Pics are in twice since I'm reusing the PicButton class lazily
        self.correct = PicButton(0,QPixmap("Resources/tick.png"),QPixmap("Resources/tick.png"))
        self.wrong = PicButton(0,QPixmap("Resources/cross.png"),QPixmap("Resources/cross.png"))

        self.close = QPushButton("Close")

        # The layouts
        hboxButtons = QHBoxLayout()
        hboxButtons.addWidget(self.correct)
        hboxButtons.addWidget(self.wrong)
        hboxButtons.addWidget(self.close)

        vboxFull = QVBoxLayout()
        vboxFull.addWidget(self.wPlot)
        vboxFull.addWidget(self.species)
        vboxFull.addLayout(hboxButtons)

        self.setLayout(vboxFull)
        self.makefig(seg)

    def makefig(self,seg):
        self.plot.setImage(np.fliplr(seg.T))

    def getValues(self):
        # TODO
        return True

    def setImage(self,seg,label):
        self.plot.setImage(np.fliplr(seg.T))
        self.species.setText(label)
        self.canvasPlot.draw()

class CorrectHumanClassify(QDialog):
    # This is to correct the classification of those that the program got wrong
    def __init__(self, seg, bb1, bb2, bb3, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Correct Classification')
        self.frame = QWidget()

        self.saveConfig = False

        # Set up the plot windows, then the forward and backward buttons
        # TODO: Replace with pyqtgraph
        #self.plot = Figure()
        #self.plot.set_size_inches(10.0, 2.0, forward=True)
        #self.canvasPlot = FigureCanvas(self.plot)
        #self.canvasPlot.setParent(self.frame)

        self.wPlot = pg.GraphicsLayoutWidget()
        self.pPlot = self.wPlot.addViewBox(enableMouse=False,row=0,col=1)
        self.plot = pg.ImageItem()
        self.pPlot.addItem(self.plot)

        # An array of radio buttons and a list and a text entry box
        # Create an array of radio buttons for the most common birds (2 columns of 10 choices)
        self.birds1 = []
        for item in bb1:
            self.birds1.append(QRadioButton(item))
        self.birds2 = []
        for item in bb2:
            self.birds2.append(QRadioButton(item))

        for i in xrange(len(self.birds1)):
            self.birds1[i].setEnabled(True)
            self.connect(self.birds1[i], SIGNAL("clicked()"), self.radioBirdsClicked)
        for i in xrange(len(self.birds2)):
            self.birds2[i].setEnabled(True)
            self.connect(self.birds2[i], SIGNAL("clicked()"), self.radioBirdsClicked)

        # The list of less common birds
        self.birdList = QListWidget(self)
        self.birdList.setMaximumWidth(150)
        for item in bb3:
            self.birdList.addItem(item)
        self.birdList.sortItems()
        # Explicitly add "Other" option in
        self.birdList.insertItem(0,'Other')

        self.connect(self.birdList, SIGNAL("itemClicked(QListWidgetItem*)"), self.listBirdsClicked)
        self.birdList.setEnabled(False)

        # This is the text box for missing birds
        self.tbox = QLineEdit(self)
        self.tbox.setMaximumWidth(150)
        self.connect(self.tbox, SIGNAL('editingFinished()'), self.birdTextEntered)
        self.tbox.setEnabled(False)

        self.close = QPushButton("Done")
        self.connect(self.close, SIGNAL("clicked()"), self.accept)

        # The layouts
        birds1Layout = QVBoxLayout()
        for i in xrange(len(self.birds1)):
            birds1Layout.addWidget(self.birds1[i])

        birds2Layout = QVBoxLayout()
        for i in xrange(len(self.birds2)):
            birds2Layout.addWidget(self.birds2[i])

        birdListLayout = QVBoxLayout()
        birdListLayout.addWidget(self.birdList)
        birdListLayout.addWidget(QLabel("If bird isn't in list, select Other"))
        birdListLayout.addWidget(QLabel("Type below, Return at end"))
        birdListLayout.addWidget(self.tbox)

        hbox = QHBoxLayout()
        hbox.addLayout(birds1Layout)
        hbox.addLayout(birds2Layout)
        hbox.addLayout(birdListLayout)

        vboxFull = QVBoxLayout()
        vboxFull.addWidget(self.plot)
        vboxFull.addLayout(hbox)
        vboxFull.addWidget(self.close)

        self.setLayout(vboxFull)
        self.makefig(seg)

    def makefig(self,seg):
        self.plot.setImage(np.fliplr(seg.T))

    def radioBirdsClicked(self):
        # Listener for when the user selects a radio button
        # Update the text and store the data
        for button in self.birds1+self.birds2:
            if button.isChecked():
                if button.text()=="Other":
                    self.birdList.setEnabled(True)
                else:
                    self.birdList.setEnabled(False)
                    self.label = str(button.text())

    def listBirdsClicked(self, item):
        # Listener for clicks in the listbox of birds
        if (item.text() == "Other"):
            self.tbox.setEnabled(True)
        else:
            # Save the entry
            self.label = str(item.text())

    def birdTextEntered(self):
        # Listener for the text entry in the bird list
        # Check text isn't already in the listbox, and add if not
        # Doesn't sort the list, but will when program is closed
        item = self.birdList.findItems(self.tbox.text(),Qt.MatchExactly)
        if item:
            pass
        else:
            self.birdList.addItem(self.tbox.text())
        self.label = str(self.tbox.text())
        self.saveConfig=True
        self.tbox.setEnabled(False)

    # static method to create the dialog and return the correct label
    @staticmethod
    def getValues(seg, cmap_grey, bb1, bb2, bb3,parent=None):
        dialog = CorrectHumanClassify(seg, cmap_grey, bb1, bb2, bb3,parent)
        result = dialog.exec_()
        return dialog.label, dialog.saveConfig

    def setImage(self,seg,label):
        self.plot.setImage(np.fliplr(seg.T))

class PicButton(QAbstractButton):
    # Class for HumanClassify2 to put spectrograms on buttons
    def __init__(self, index, pixmap1, pixmap2, parent=None):
        super(PicButton, self).__init__(parent)
        self.index = index
        self.pixmap1 = pixmap1
        self.pixmap2 = pixmap2
        self.buttonClicked = False
        self.clicked.connect(self.update)

    def paintEvent(self, event):
        pix = self.pixmap2 if self.buttonClicked else self.pixmap1

        if type(event) is not bool:
            painter = QPainter(self)
            painter.drawPixmap(event.rect(), pix)

    def sizeHint(self):
        return self.pixmap1.size()

    def update(self,event):
        print "Button " + str(self.index) + " clicked"
        self.buttonClicked = not(self.buttonClicked)
        self.paintEvent(event)

class HumanClassify2(QDialog):
    # This dialog is different to the others. The aim is to check (or ask for) classifications for segments.
    # This version gets *12* at a time, and put them all out together on buttons, and their labels.
    # It could be all the same species, or the ones that it is unsure about, or whatever.

    def __init__(self, seg, label, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Check Classifications')
        self.frame = QWidget()

        # TODO: Add a label with instructions
        # TODO: Add button to finish and/or get more
        # TODO: Add a label with the species

        # TODO: Decide on these sizes
        self.width = 3
        self.height = 4
        grid = QGridLayout(self.frame)
        self.setLayout(grid)

        positions = [(i, j) for i in range(self.height) for j in range(self.width)]

        # TODO: Next line needs to go!
        segs = [1,2,3,4,5,6,7]
        images = []
        # TODO: Turn this into different images
        if len(segs) < self.width*self.height:
            for i in range(len(segs)):
                images.append(self.setImage(seg))
            for i in range(len(segs),self.width*self.height):
                images.append([None,None])
        else:
            for i in range(self.width*self.height):
                images.append(self.setImage(seg))


        for position, im in zip(positions, images):
            if im is not None:
                button = PicButton(position[0] * self.width + position[1], im[0], im[1])
                grid.addWidget(button, *position)

        self.setLayout(grid)

    def setImage(self,seg):
        # TODO: interesting bug in making one of the images sometimes!
        self.image = pg.ImageItem()
        self.image.setImage(np.fliplr(seg.T))
        im1 = self.image.getPixmap()

        self.image.setImage(np.fliplr(-seg.T))
        im2 = self.image.getPixmap()

        return [im1, im2]

    def activate(self):
        return True

class CustomViewBox(pg.ViewBox):
    # Normal ViewBox, but with ability to drag the segments
    sigMouseDragged = QtCore.Signal(object,object,object)

    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)

    def mouseDragEvent(self, ev):
        ## if axis is specified, event will only affect that axis.
        ev.accept()  ## we accept all buttons
        if self.state['mouseMode'] != pg.ViewBox.RectMode or ev.button() == QtCore.Qt.RightButton:
            ev.ignore()

        if ev.isFinish():  ## This is the final move in the drag; draw the actual box
            self.rbScaleBox.hide()
            self.sigMouseDragged.emit(ev.buttonDownScenePos(ev.button()),ev.scenePos(),ev.screenPos())
        else:
            ## update shape of scale box
            self.updateScaleBox(ev.buttonDownPos(), ev.pos())

    def keyPressEvent(self,ev):
        # TODO: This catches the keypresses and sends out a signal
        #print ev.key(), ev.text()
        self.emit(SIGNAL("keyPressed"),ev)

class FileDataDialog(QDialog):
    def __init__(self, name,date,time,parent=None):
        super(FileDataDialog, self).__init__(parent)

        layout = QVBoxLayout(self)

        l1 = QLabel("Annotator")
        self.name = QLineEdit(self)
        self.name.setText(name)

        l2 = QLabel("Data recorded: "+date)
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

class StartScreen(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Choose Task')
        self.activateWindow()

        #b1 = QPushButton(QIcon(":/Resources/play.svg"), "&Play Window")
        b1 = QPushButton("Manual Segmentation")
        b2 = QPushButton("Find a species")
        b3 = QPushButton("Denoise a folder")

        self.connect(b1, SIGNAL('clicked()'), self.manualSeg)
        self.connect(b2, SIGNAL('clicked()'), self.findSpecies)
        self.connect(b3, SIGNAL('clicked()'), self.denoise)

        vbox = QVBoxLayout()
        for w in [b1, b2, b3]:
                vbox.addWidget(w)

        self.setLayout(vbox)
        self.task = -1

    def manualSeg(self):
        self.task = 0
        self.accept()

    def findSpecies(self):
        self.task = 1
        self.accept()

    def denoise(self):
        self.task = 2
        self.accept()

    def getValues(self):
        return self.task


# Start the application
app = QApplication(sys.argv)

# This screen asks what you want to do, then gets the response
# TODO: Why don't the buttons appear at once?
#first = StartScreen()
#first.exec_()

#task = first.getValues()

#if task == 0:
avianz = AviaNZInterface(configfile='AviaNZconfig.txt')
avianz.show()
app.exec_()
