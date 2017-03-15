# Interface.py
#
# This is the main class for the AviaNZ interface
# It's fairly simple, but seems to work OK
# Now with pyqtgraph for speed
# Version 0.8 28/02/17
# Author: Stephen Marsland

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

import sys, os, json, datetime  #glob
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.phonon as phonon

import librosa as lr
from scipy.io import wavfile
import numpy as np
import pylab as pl

#import matplotlib
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.figure import Figure
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

# Folder to look in, what if file doesn't exist?
# Moving bar!!

# Add in the wavelet segmentation
# Finish sorting out parameters for median clipping segmentation, energy segmentation
# Finish the raven features
# Finish cross-correlation to pick out similar bits of spectrogram -> and what else?
# Finish denoising

#   Colour code of boxes by label (add info somewhere)
#   Finish implementing the drag box on the spectrogram --> how to shade it? Then finish the 'add me' bits
#  If don't select something in context menu get error -> not critical

#   The remote plotting should speed things up
#   Look into ParameterTree for saving the config stuff in particular
# Better loading of files -> paging, not computing whole spectrogram (how to deal with overview? -> coarser spec?)
    # Maybe: check length of file. If > 5 mins, load first 5 only (? how to move to next 5?)
# How to set bandpass params? -> is there a useful plot to help? -> function of sampleRate

# Sound playback:
    # make it only play back the visible section
    # replace slider, or at least debug it! finish

# Overall layout -> buttons on the left in a column, or with tabs?
# Test the init part about if file or directory doesn't exist
# Finish implementation for button to show individual segments to user and ask for feedback and the other feedback dialogs
# Ditto lots of segments at once

# Implement something for the Classify button:
    # Take the segments that have been given and try to classify them in lots of ways:
    # Cross-correlation, DTW, shape metric, features and learning

# Testing data
# Documentation
# Licensing

# Text resizing

# Use intensity of colour to encode certainty?

# Some bug in denoising? -> tril1
# More features, add learning!
# Pitch tracking, fundamental frequency
# Other parameters for dialogs?
#      multitaper for spectrogram

# Needs decent testing

# Option to turn button menu on/off?

# Minor:
# Librosa isn't in windows!
# Turn stereo sound into mono using librosa, consider always resampling to 22050Hz (except when it's less in file :) )
# Font size to match segment size -> make it smaller, could also move it up or down as appropriate
# Would be nice to put the axis label on the right

# Things to consider:
    # Second spectrogram (currently use right button for interleaving)? My current choice is no as it takes up space
    # Put the labelling (and loading) in a dialog to free up space? -> bigger plots
    # Useful to have a go to start button as well as forward and backward?

# Look at raven and praat and luscinia -> what else is actually useful? Other annotations on graphs?

# Given files > 5 mins, split them into 5 mins versions anyway (code is there, make it part of workflow)
# Don't really want to load the whole thing, just 5 mins, and then move through with arrows -> how?
# This is sometimes called paging, I think. (y, sr = librosa.load(filename, offset=15.0, duration=5.0) might help. Doesn't do much for the overview through)

# As well as this version with pictures, will need to be able to call various bits to work offline
# denoising, segmentation, etc. that will then just show some pictures

# Get suggestions from the others

# Things to remember
# When adding new classes, make sure to pass new data to them in undoing and loading

# This version has the selection of birds using a context menu and then has removed the radio buttons
# Code is still there, though, just commented out. Add as an option?

# Diane:
    # menu
    # for matching, show matched segment, and extended 'guess' (with some overlap)
    # Something to show nesting of segments, such as a number of segments in the top bit
    # Find similar segments in other files -- other birds
    # Group files by call type
# ===============

class Interface(QMainWindow):
    # Main class for the interface, which contains most of the user interface and plotting code

    def __init__(self,root=None,configfile=None):
        # Main part of the initialisation is loading a configuration file, or creating a new one if it doesn't
        # exist. Also loads an initial file (specified explicitly) and sets up the window.
        # TODO: better way to choose initial file (or don't choose one at all)
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

        #self.createMenu()
        self.createFrame()

        # Some safety checking for paths and files
        if not os.path.isdir(self.dirName):
            print("Directory doesn't exist: making it")
            os.makedirs(self.dirName)
        if not os.path.isfile(self.dirName+'/'+self.firstFile):
            fileName = QtGui.QFileDialog.getOpenFileName(self, 'Chose File', self.dirName,
                                                         selectedFilter='*.wav')
            if fileName:
                self.firstFile = fileName
        self.loadFile(self.firstFile)

    # def createMenu(self):
    #     # Create the menu entries at the top of the screen. Not really needed, and hence commented out currently
    #     self.fileMenu = self.menuBar().addMenu("&File")
    #
    #     openFileAction = QAction("&Open wave file", self)
    #     self.connect(openFileAction,SIGNAL("triggered()"),self.openFile)
    #     self.fileMenu.addAction(openFileAction)
    #
    #     # This seems to only work if it is there twice ?!
    #     quitAction = QAction("&Quit", self)
    #     self.connect(quitAction,SIGNAL("triggered()"),self.quit)
    #     self.fileMenu.addAction(quitAction)
    #
    #     quitAction = QAction("&Quit", self)
    #     self.connect(quitAction, SIGNAL("triggered()"), self.quit)
    #     self.fileMenu.addAction(quitAction)

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
            'dirpath': './Sound Files',

            # Param for width in seconds of the main representation
            'windowWidth': 10.0,

            # These are the contrast parameters for the spectrogram
            'colourStart': 0.4,
            'colourEnd': 1.0,

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
        self.d_overview = Dock("Overview",size = (1200,100))
        self.d_ampl = Dock("Amplitude",size=(1200,300))
        self.d_spec = Dock("Spectrogram",size=(1200,300))
        self.d_controls = Dock("Controls",size=(800,150))
        self.d_files = Dock("Files",size=(400,250))
        self.d_buttons = Dock("Buttons",size=(800,100))

        self.area.addDock(self.d_overview,'top')
        self.area.addDock(self.d_ampl,'bottom',self.d_overview)
        self.area.addDock(self.d_spec,'bottom',self.d_ampl)
        self.area.addDock(self.d_controls,'bottom',self.d_spec)
        self.area.addDock(self.d_files,'left',self.d_controls)
        self.area.addDock(self.d_buttons,'bottom',self.d_controls)

        # Put content widgets in the docks
        self.w_overview = pg.LayoutWidget()
        self.d_overview.addWidget(self.w_overview)
        self.w_overview1 = pg.GraphicsLayoutWidget()
        self.w_overview.addWidget(self.w_overview1)
        self.p_overview = self.w_overview1.addViewBox(enableMouse=False,enableMenu=False,row=1,col=0)

        self.w_ampl = pg.GraphicsLayoutWidget()
        self.p_ampl = self.w_ampl.addViewBox(enableMouse=False,enableMenu=False)
        self.w_ampl.addItem(self.p_ampl,row=0,col=1)
        self.d_ampl.addWidget(self.w_ampl)

        # The axes
        self.timeaxis = pg.AxisItem(orientation='bottom')
        self.w_ampl.addItem(self.timeaxis,row=1,col=1)
        self.timeaxis.linkToView(self.p_ampl)
        self.timeaxis.setLabel('Time',units='s')

        self.ampaxis = pg.AxisItem(orientation='left')
        self.w_ampl.addItem(self.ampaxis,row=0,col=0)
        self.ampaxis.linkToView(self.p_ampl)

        self.w_spec = pg.GraphicsLayoutWidget()
        self.p_spec = CustomViewBox(enableMouse=False,enableMenu=False)
        #self.p_spec = pg.ViewBox(enableMouse=False,enableMenu=False)
        self.w_spec.addItem(self.p_spec,row=0,col=1)
        #self.p_spec = self.w_spec.addViewBox(enableMouse=False,enableMenu=False,row=0,col=1)
        self.d_spec.addWidget(self.w_spec)

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

        self.w_buttons = pg.LayoutWidget()
        self.d_buttons.addWidget(self.w_buttons)

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

        # Checkbox for whether or not user is drawing boxes around song in the spectrogram (defaults to clicks not drags)
        self.dragRectangles = QCheckBox('Drag boxes in spectrogram')
        self.dragRectangles.stateChanged[int].connect(self.dragRectanglesCheck)

        # A slider to show playback position
        # TODO: Experiment with other options -- bar in the window
        self.playSlider = QSlider(Qt.Horizontal, self)
        self.connect(self.playSlider,SIGNAL('sliderReleased()'),self.sliderMoved)

        self.w_controls.addWidget(QLabel('Slide top box to move through recording, click to start and end a segment, click on segment to edit or label. Right click to interleave.'),row=0,col=0,colspan=4)
        self.w_controls.addWidget(self.playSlider,row=1,col=0,colspan=4)
        self.w_controls.addWidget(self.playButton,row=2,col=0)
        self.w_controls.addWidget(self.timePlayed,row=2,col=1)
        #self.w_controls.addWidget(self.resetButton,row=2,col=1)
        self.w_controls.addWidget(self.dragRectangles,row=2,col=2)

        # The spinbox for changing the width shown in figMain
        self.widthWindow = QDoubleSpinBox()
        self.widthWindow.setSingleStep(1.0)
        self.widthWindow.setDecimals(2)
        self.widthWindow.setValue(self.config['windowWidth'])
        self.w_controls.addWidget(QLabel('Visible window width (seconds)'),row=2,col=3)
        self.w_controls.addWidget(self.widthWindow,row=3,col=2,colspan=2)
        self.widthWindow.valueChanged[float].connect(self.changeWidth)

        # Next is the bottom part of the screen

        # List to hold the list of files
        self.listFiles = QListWidget(self)
        self.listFiles.setFixedWidth(150)
        self.fillList()
        self.listFiles.connect(self.listFiles, SIGNAL('itemDoubleClicked(QListWidgetItem*)'), self.listLoadFile)

        self.w_files.addWidget(QLabel('Double click to select'),row=0,col=0)
        self.w_files.addWidget(QLabel('Red names have segments'),row=1,col=0)
        self.w_files.addWidget(self.listFiles,row=2,col=0)

        # These are the main buttons, on the bottom right
        quitButton = QPushButton("&Quit")
        self.connect(quitButton, SIGNAL('clicked()'), self.quit)
        spectrogramButton = QPushButton("Spectrogram &Params")
        self.connect(spectrogramButton, SIGNAL('clicked()'), self.spectrogramDialog)
        segmentButton = QPushButton("&Segment")
        self.connect(segmentButton, SIGNAL('clicked()'), self.segmentationDialog)
        denoiseButton = QPushButton("&Denoise")
        self.connect(denoiseButton, SIGNAL('clicked()'), self.denoiseDialog)
        recogniseButton = QPushButton("&Recognise")
        self.connect(recogniseButton, SIGNAL('clicked()'), self.recognise)
        deleteButton = QPushButton("&Delete Current Segment")
        self.connect(deleteButton, SIGNAL('clicked()'), self.deleteSegment)
        deleteAllButton = QPushButton("&Delete All Segments")
        self.connect(deleteAllButton, SIGNAL('clicked()'), self.deleteAll)
        findMatchButton = QPushButton("&Find Matches")
        self.connect(findMatchButton, SIGNAL('clicked()'), self.findMatches)
        checkButton = QPushButton("&Check Segments")
        self.connect(checkButton, SIGNAL('clicked()'), self.humanClassifyDialog)
        dockButton = QPushButton("&Put Docks Back")
        self.connect(dockButton, SIGNAL('clicked()'), self.dockReplace)

        # vboxButtons2 = QVBoxLayout()
        for w in [deleteButton,deleteAllButton,spectrogramButton,denoiseButton, segmentButton, findMatchButton, quitButton, checkButton, dockButton]:
            self.w_buttons.addWidget(w)

        # The context menu (drops down on mouse click) to select birds
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.menuBirdList = QMenu()
        # Need the lambda function to connect all menu events to same trigger and know which was selected
        for item in self.config['BirdButtons1'] + self.config['BirdButtons2'][:-1]:
            bird = self.menuBirdList.addAction(item)
            receiver = lambda birdname=item: self.birdSelected(birdname)
            self.connect(bird, SIGNAL("triggered()"), receiver)
            self.menuBirdList.addAction(bird)
        self.menuBird2 = self.menuBirdList.addMenu('Other')
        for item in self.config['ListBirdsEntries']+['Other']:
            bird = self.menuBird2.addAction(item)
            receiver = lambda birdname=item: self.birdSelected(birdname)
            self.connect(bird, SIGNAL("triggered()"), receiver)
            self.menuBird2.addAction(bird)

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
        self.media_obj.tick.connect(self.movePlaySlider)
        self.media_obj.finished.connect(self.playFinished)
        # TODO: Check the next line out!
        #self.media_obj.totalTimeChanged.connect(self.setSliderLimits)

        # Make the colours
        self.ColourSelected = QtGui.QBrush(QtGui.QColor(self.config['ColourSelected'][0], self.config['ColourSelected'][1], self.config['ColourSelected'][2], self.config['ColourSelected'][3]))
        self.ColourNamed = QtGui.QBrush(QtGui.QColor(self.config['ColourNamed'][0], self.config['ColourNamed'][1], self.config['ColourNamed'][2], self.config['ColourNamed'][3]))
        self.ColourNone = QtGui.QBrush(QtGui.QColor(self.config['ColourNone'][0], self.config['ColourNone'][1], self.config['ColourNone'][2], self.config['ColourNone'][3]))

        # Hack to get the type of an ROI
        p_spec_r = pg.RectROI(0, 0)
        self.ROItype = type(p_spec_r)

        # Store the state of the docks
        self.state = self.area.saveState()

        # Plot everything
        self.show()

    def fillList(self):
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
        self.deleteAll()
        if hasattr(self, 'overviewImageRegion'):
            self.p_overview.removeItem(self.overviewImageRegion)

        # This is a flag to say if the next thing that they click on should be a start or a stop for segmentation
        self.started_a = False
        self.started_s = False

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
            self.fillList()
        else:
            self.loadFile(current)

    def loadFile(self,name):
        # This does the work of loading a file
        # One magic constant, which normalises the data
        # TODO: moved to librosa instead of wavfile. Put both in and see if it is slower!
        # TODO: Note that librosa normalised things, which buggers up the spectrogram for e.g., Harma

        if isinstance(name,str):
            self.filename = self.dirName+'/'+name
        else:
            self.filename = self.dirName+'/'+str(name.text())
        #self.audiodata, self.sampleRate = lr.load(self.filename,sr=None)
        self.sampleRate, self.audiodata = wavfile.read(self.filename)
        # None of the following should be necessary for librosa
        if self.audiodata.dtype is not 'float':
            self.audiodata = self.audiodata.astype('float') / 32768.0
        if np.shape(np.shape(self.audiodata))[0]>1:
            self.audiodata = self.audiodata[:,0]
        self.datalength = np.shape(self.audiodata)[0]
        print("Length of file is ",len(self.audiodata),float(self.datalength)/self.sampleRate)

        # Create an instance of the Signal Processing class
        if not hasattr(self,'sp'):
            self.sp = SignalProc.SignalProc(self.audiodata, self.sampleRate,self.config['window_width'],self.config['incr'])

        # Get the data for the spectrogram
        # TODO: put a button for multitapering somewhere
        self.sgRaw = self.sp.spectrogram(self.audiodata,self.sampleRate,multitaper=False)
        #print np.min(self.sgRaw), np.max(self.sgRaw)
        self.sg = np.abs(np.where(self.sgRaw==0,0.0,10.0 * np.log10(self.sgRaw)))

        # Colour scaling for the spectrograms
        #print np.shape(self.sg), np.max(self.sg), np.min(self.sg)
        self.overviewImage.setLevels([self.config['colourStart']*np.max(self.sg), self.config['colourEnd']*np.max(self.sg)])
        self.specPlot.setLevels([self.config['colourStart']*np.max(self.sg), self.config['colourEnd']*np.max(self.sg)])

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

        # Decide on the length of the playback bit for the slider
        self.setSliderLimits(0,self.media_obj.totalTime())
        self.totalTime = self.convertMillisecs(self.media_obj.totalTime())

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

    def dragRectanglesCheck(self,check):
        # The checkbox that says if the user is dragging rectangles or clicking on the spectrogram has changed state
        if self.dragRectangles.isChecked():
            print self.p_spec.state['mouseMode'], self.p_ampl.state['mouseMode']
            print "Checked"
            self.p_spec.setMouseMode(pg.ViewBox.RectMode)
            print self.p_spec.state['mouseMode'], self.p_ampl.state['mouseMode']
        else:
            print "Unchecked"
            self.p_spec.setMouseMode(pg.ViewBox.PanMode)

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

    def updateOverview(self):
        # Listener for when the overview box is changed
        # Does the work of keeping all the plots in the right range
        minX, maxX = self.overviewImageRegion.getRegion()
        #print "updating overview", minX, maxX, self.convertSpectoAmpl(maxX)
        self.widthWindow.setValue(self.convertSpectoAmpl(maxX-minX))
        self.p_ampl.setXRange(self.convertSpectoAmpl(minX), self.convertSpectoAmpl(maxX), padding=0)
        self.p_spec.setXRange(minX, maxX, padding=0)
        self.p_ampl.setXRange(self.convertSpectoAmpl(minX), self.convertSpectoAmpl(maxX), padding=0)
        self.p_spec.setXRange(minX, maxX, padding=0)

    def drawfigMain(self):
        # This draws the main amplitude and spectrogram plots
        # TODO: RemoteGraphicsView: parallel

        self.amplPlot.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=self.datalength,endpoint=True),self.audiodata)

        self.specPlot.setImage(np.fliplr(self.sg.T))
        self.specaxis.setTicks([[(0,0),(np.shape(self.sg)[0],self.sampleRate)]])

        self.updateOverview()

        # If there are segments, show them
        for count in range(len(self.segments)):
            self.addSegment(self.segments[count][0], self.segments[count][1],self.segments[count][2],self.segments[count][3],self.segments[count][4],False)

        # Another go at a moving bar
        self.bar = pg.InfiniteLine(angle=90, movable=False, pen={'color': 'r', 'width': 3})
        self.p_ampl.addItem(self.bar, ignoreBounds=True)
        self.bar.setValue(0.0)

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

        # Add the segments, connect up the listeners
        p_ampl_r = pg.LinearRegionItem(brush=brush)

        self.p_ampl.addItem(p_ampl_r, ignoreBounds=True)
        p_ampl_r.setRegion([startpoint, endpoint])
        p_ampl_r.sigRegionChangeFinished.connect(self.updateRegion_ampl)

        if y1==0 and y2==0:
            p_spec_r = pg.LinearRegionItem(brush = brush)
            self.p_spec.addItem(p_spec_r, ignoreBounds=True)
            p_spec_r.setRegion([self.convertAmpltoSpec(startpoint), self.convertAmpltoSpec(endpoint)])
        else:
            startpointS = QPointF(self.convertAmpltoSpec(startpoint),y1)
            endpointS = QPointF(self.convertAmpltoSpec(endpoint),y2)
            p_spec_r = pg.RectROI(startpointS, endpointS - startpointS, pen='r')
            self.p_spec.addItem(p_spec_r, ignoreBounds=True)
            p_spec_r.sigRegionChangeFinished.connect(self.updateRegion_spec)

        p_spec_r.sigRegionChangeFinished.connect(self.updateRegion_spec)

        # Put the text into the box
        label = pg.TextItem(text=species, color='k')
        self.p_ampl.addItem(label)
        label.setPos(min(startpoint, endpoint),self.minampl)

        # Add the segments to the relevent lists
        self.listRectanglesa1.append(p_ampl_r)
        self.listRectanglesa2.append(p_spec_r)
        self.listLabels.append(label)

        if saveSeg:
            # Add the segment to the data
            self.segments.append([min(startpoint, endpoint), max(startpoint, endpoint), y1, y2, species])

    def mouseClicked_ampl(self,evt):
        pos = evt.scenePos()
        #print pos.x(), self.p_ampl.mapSceneToView(pos).x()

        if self.box1id>-1:
            self.listRectanglesa1[self.box1id].setBrush(self.prevBoxCol)
            self.listRectanglesa1[self.box1id].update()
            if type(self.listRectanglesa2[self.box1id]) == self.ROItype:
                print "add me!"
                # TODO:
            else:
                self.listRectanglesa2[self.box1id].setBrush(self.prevBoxCol)
                self.listRectanglesa2[self.box1id].update()

        if self.p_ampl.sceneBoundingRect().contains(pos):
            mousePoint = self.p_ampl.mapSceneToView(pos)

            if self.started_a:
                # This is the second click, so should pay attention and close the segment
                # Stop the mouse motion connection, remove the drawing boxes
                self.p_ampl.scene().sigMouseMoved.disconnect()
                self.p_ampl.removeItem(self.vLine_a)
                self.p_ampl.removeItem(self.drawingBox_ampl)
                self.p_spec.removeItem(self.drawingBox_spec)
                self.addSegment(self.start_location_a,mousePoint.x())

                # Context menu
                self.box1id = len(self.segments)-1
                self.menuBirdList.popup(QPoint(evt.screenPos().x(),evt.screenPos().y()))

                self.listRectanglesa1[self.box1id].setBrush(fn.mkBrush(self.ColourSelected))
                self.listRectanglesa1[self.box1id].update()
                self.listRectanglesa2[self.box1id].setBrush(fn.mkBrush(self.ColourSelected))
                self.listRectanglesa2[self.box1id].update()

                self.started_a = not(self.started_a)
            else:
                # Check if the user has clicked in a box
                # Note: Returns the first one it finds
                box1id = -1
                for count in range(len(self.listRectanglesa1)):
                    x1, x2 = self.listRectanglesa1[count].getRegion()
                    if x1 <= mousePoint.x() and x2 >= mousePoint.x():
                        box1id = count

                if box1id > -1 and not evt.button() == QtCore.Qt.RightButton:
                    # User clicked in a box (with the left button)
                    # Change colour, store the old colour
                    self.box1id = box1id
                    self.prevBoxCol = self.listRectanglesa1[box1id].brush.color()
                    self.listRectanglesa1[box1id].setBrush(fn.mkBrush(self.ColourSelected))
                    self.listRectanglesa1[box1id].update()
                    if type(self.listRectanglesa2[self.box1id]) == self.ROItype:
                        print "add me!"
                        # TODO
                    else:
                        self.listRectanglesa2[box1id].setBrush(fn.mkBrush(self.ColourSelected))
                        self.listRectanglesa2[box1id].update()

                    self.menuBirdList.popup(QPoint(evt.screenPos().x(), evt.screenPos().y()))
                else:
                    # User hasn't clicked in a box (or used the right button), so start a new segment
                    # Note that need to click in the same plot both times.
                    self.start_location_a = mousePoint.x()
                    self.vLine_a = pg.InfiniteLine(angle=90, movable=False,pen={'color': 'r', 'width': 3})
                    self.p_ampl.addItem(self.vLine_a, ignoreBounds=True)
                    self.vLine_a.setPos(self.start_location_a)

                    brush = self.ColourNone
                    self.drawingBox_ampl = pg.LinearRegionItem(brush=brush)
                    self.p_ampl.addItem(self.drawingBox_ampl, ignoreBounds=True)
                    self.drawingBox_ampl.setRegion([self.start_location_a, self.start_location_a])
                    self.drawingBox_spec = pg.LinearRegionItem(brush=brush)
                    self.p_spec.addItem(self.drawingBox_spec, ignoreBounds=True)
                    self.drawingBox_spec.setRegion([self.convertAmpltoSpec(self.start_location_a), self.convertAmpltoSpec(self.start_location_a)])
                    self.p_ampl.scene().sigMouseMoved.connect(self.GrowBox_ampl)

                    self.started_a = not (self.started_a)

    def mouseClicked_spec(self,evt):
        if self.dragRectangles.isChecked():
            return
        else:
            pos = evt.scenePos()
            #print pos, self.p_spec.mapSceneToView(pos)

            if self.box1id>-1:
                self.listRectanglesa1[self.box1id].setBrush(self.prevBoxCol)
                self.listRectanglesa1[self.box1id].update()
                if type(self.listRectanglesa2[self.box1id]) == self.ROItype:
                    print "add me!"
                    # TODO:
                else:
                    self.listRectanglesa2[self.box1id].setBrush(self.prevBoxCol)
                    self.listRectanglesa2[self.box1id].update()

            if self.p_spec.sceneBoundingRect().contains(pos):
                mousePoint = self.p_spec.mapSceneToView(pos)

                if self.started_s:
                    # This is the second click, so should pay attention and close the segment
                    # Stop the mouse motion connection, remove the drawing boxes
                    self.p_spec.scene().sigMouseMoved.disconnect()
                    self.p_spec.removeItem(self.vLine_s)
                    self.p_ampl.removeItem(self.drawingBox_ampl)
                    self.p_spec.removeItem(self.drawingBox_spec)
                    self.addSegment(self.convertSpectoAmpl(self.start_location_s),self.convertSpectoAmpl(mousePoint.x()))

                    # Context menu
                    self.box1id = len(self.segments)-1
                    self.menuBirdList.popup(QPoint(evt.screenPos().x(),evt.screenPos().y()))

                    self.listRectanglesa1[self.box1id].setBrush(fn.mkBrush(self.ColourSelected))
                    self.listRectanglesa1[self.box1id].update()
                    self.listRectanglesa2[self.box1id].setBrush(fn.mkBrush(self.ColourSelected))
                    self.listRectanglesa2[self.box1id].update()

                    self.started_s = not(self.started_s)
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
                        if type(self.listRectanglesa2[self.box1id]) == self.ROItype:
                            print "add me"
                            # TODO!!
                        else:
                            self.listRectanglesa2[box1id].setBrush(fn.mkBrush(self.ColourSelected))
                            self.listRectanglesa2[box1id].update()

                        self.menuBirdList.popup(QPoint(evt.screenPos().x(), evt.screenPos().y()))
                    else:
                        # User hasn't clicked in a box (or used the right button), so start a new segment
                        # Note that need to click in the same plot both times.
                        self.start_location_s = mousePoint.x()
                        self.vLine_s = pg.InfiniteLine(angle=90, movable=False,pen={'color': 'r', 'width': 3})
                        self.p_spec.addItem(self.vLine_s, ignoreBounds=True)
                        self.vLine_s.setPos(self.start_location_s)

                        brush = self.ColourNone
                        self.drawingBox_ampl = pg.LinearRegionItem(brush=brush)
                        self.p_ampl.addItem(self.drawingBox_ampl, ignoreBounds=True)
                        self.drawingBox_ampl.setRegion([self.convertSpectoAmpl(self.start_location_s), self.convertSpectoAmpl(self.start_location_s)])
                        self.drawingBox_spec = pg.LinearRegionItem(brush=brush)
                        self.p_spec.addItem(self.drawingBox_spec, ignoreBounds=True)
                        self.drawingBox_spec.setRegion([self.start_location_s, self.start_location_s])
                        self.p_spec.scene().sigMouseMoved.connect(self.GrowBox_spec)

                        self.started_s = not (self.started_s)

    def mouseDragged_spec(self, evt1, evt2,evt3):
        # TODO: *** Finish this -- how to shade?
        if self.dragRectangles.isChecked():
            evt1 = self.p_spec.mapSceneToView(evt1)
            evt2 = self.p_spec.mapSceneToView(evt2)
            p_spec_r = pg.RectROI(evt1, evt2 - evt1, pen='r')
            self.p_spec.addItem(p_spec_r, ignoreBounds=True)
            p_spec_r.sigRegionChangeFinished.connect(self.updateRegion_spec)

            # Add the segment to the amplitude graph, connect up the listener
            startpoint = self.convertSpectoAmpl(evt1.x())
            endpoint = self.convertSpectoAmpl(evt2.x())
            p_ampl_r = pg.LinearRegionItem(brush=self.ColourNone)
            self.p_ampl.addItem(p_ampl_r, ignoreBounds=True)
            p_ampl_r.setRegion([startpoint,endpoint])
            p_ampl_r.sigRegionChangeFinished.connect(self.updateRegion_ampl)

            # Put the text into the box
            label = pg.TextItem(text='None', color='k')
            self.p_ampl.addItem(label)
            label.setPos(min(startpoint, endpoint), self.minampl)

            # Add the segments to the relevent lists
            self.listRectanglesa1.append(p_ampl_r)
            self.listRectanglesa2.append(p_spec_r)
            self.listLabels.append(label)

            # Add the segment to the data
            self.segments.append([min(startpoint, endpoint), max(startpoint, endpoint), min(evt1.y(),evt2.y()), max(evt1.y(),evt2.y()),'None'])

            # Context menu
            self.box1id = len(self.segments) - 1
            self.menuBirdList.popup(QPoint(evt3.x(),evt3.y()))
        else:
            return

    def GrowBox_ampl(self,evt):
        pos = evt
        if self.p_ampl.sceneBoundingRect().contains(pos):
            mousePoint = self.p_ampl.mapSceneToView(pos)
            self.drawingBox_ampl.setRegion([self.start_location_a, mousePoint.x()])
            self.drawingBox_spec.setRegion([self.convertAmpltoSpec(self.start_location_a), self.convertAmpltoSpec(mousePoint.x())])

    def GrowBox_spec(self, evt):
        pos = evt
        if self.p_spec.sceneBoundingRect().contains(pos):
            mousePoint = self.p_spec.mapSceneToView(pos)
            self.drawingBox_ampl.setRegion([self.convertSpectoAmpl(self.start_location_s), self.convertSpectoAmpl(mousePoint.x())])
            self.drawingBox_spec.setRegion([self.start_location_s, mousePoint.x()])

    def birdSelected(self,birdname):
        # This collects the label for a bird from the context menu and processes it
        #print birdname, self.box1id
        if birdname is not 'Other':
            self.updateText(birdname)
        else:
            text, ok = QInputDialog.getText(self, 'Bird name', 'Enter the bird name:')
            if ok:
                text = str(text).title()
                self.updateText(text)

                if text in self.config['ListBirdsEntries']:
                    pass
                else:
                    # Add the new bird name. Will appear in alpha order next time, but at the end for now
                    count = 0
                    while self.config['ListBirdsEntries'][count] < text and count < len(self.config['ListBirdsEntries'])-1:
                        count += 1
                    self.config['ListBirdsEntries'].insert(count-1,text)
                    self.saveConfig = True

                    bird = self.menuBird2.addAction(text)
                    receiver = lambda birdname=text: self.birdSelected(birdname)
                    self.connect(bird, SIGNAL("triggered()"), receiver)
                    self.menuBird2.addAction(bird)

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

    def figMainKeypress(self,event):
        # TODO: Connect up?
        # Listener for any key presses when focus is on figMain
        # Currently just allows deleting
        # TODO: anything else?
        if event.key == 'backspace':
            self.deleteSegment()

    def updateText(self, text):
        # When the user changes the name in a segment, update the text
        self.segments[self.box1id][4] = text
        self.listLabels[self.box1id].setText(text,'k')

        # Also update the colour
        if text != "Don't Know":
            self.prevBoxCol = self.ColourNamed
        else:
            self.prevBoxCol = self.ColourNone

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

# ===========
# Code for things at the top of the screen (overview figure and left/right buttons)
    def moveLeft(self):
        # When the left button is pressed (next to the overview plot), move everything along
        # Note the parameter to all a 10% overlap
        # TODO: These change the playposition, but not the slider position?
        minX, maxX = self.overviewImageRegion.getRegion()
        newminX = max(0,minX-(maxX-minX)*0.9)
        self.overviewImageRegion.setRegion([newminX, newminX+maxX-minX])
        self.updateOverview()
        self.playPosition = int(self.convertSpectoAmpl(newminX)*1000.0)

    def moveRight(self):
        # When the right button is pressed (next to the overview plot), move everything along
        # Note the parameter to allow a 10% overlap
        # TODO: These change the playposition, but not the slider position?
        minX, maxX = self.overviewImageRegion.getRegion()
        newminX = min(np.shape(self.sg)[1]-(maxX-minX),minX+(maxX-minX)*0.9)
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
        self.updateOverview()

# ===============
# Generate the various dialogs that match the buttons

    def humanClassifyDialog(self):
        # Create the dialog that shows calls to the user for verification
        # Currently assumes that there is a selected box (later, use the first!)
        self.currentSegment = 0
        x1,x2 = self.listRectanglesa2[self.currentSegment].getRegion()
        x1 = int(x1)
        x2 = int(x2)
        self.humanClassifyDialog = HumanClassify2(self.sg[:,x1:x2],self.segments[self.currentSegment][4])
        self.humanClassifyDialog.show()
        self.humanClassifyDialog.activateWindow()
        self.humanClassifyDialog.close.clicked.connect(self.humanClassifyClose)
        self.humanClassifyDialog.correct.clicked.connect(self.humanClassifyCorrect)
        self.humanClassifyDialog.wrong.clicked.connect(self.humanClassifyWrong)

    def humanClassifyClose(self):
        # Listener for the human verification dialog.
        self.humanClassifyDialog.done(1)

    def humanClassifyNextImage(self):
        # Get the next image
        # TODO: Ends rather suddenly
        if self.currentSegment != len(self.listRectanglesa2)-1:
            self.currentSegment += 1
            self.humanClassifyDialog.setImage(self.sg[:,int(self.listRectanglesa2[self.currentSegment].get_x()):int(self.listRectanglesa2[self.currentSegment].get_x()+self.listRectanglesa2[self.currentSegment].get_width())],self.segments[self.currentSegment][4])
        else:
            print "Last image"
            self.humanClassifyClose()

    def humanClassifyCorrect(self):
        self.humanClassifyNextImage()

    def humanClassifyWrong(self):
        # First get the correct classification (by producing a new modal dialog) and update the text, then show the next image
        # TODO: Test, particularly that new birds are added
        # TODO: update the listRects
        self.correctClassifyDialog = CorrectHumanClassify(self.sg[:,int(self.listRectanglesa2[self.currentSegment].get_x()):int(self.listRectanglesa2[self.currentSegment].get_x()+self.listRectanglesa2[self.currentSegment].get_width())],self.cmap_grey,self.config['BirdButtons1'],self.config['BirdButtons2'], self.config['ListBirdsEntries'])
        label, self.saveConfig = self.correctClassifyDialog.getValues(self.sg[:,int(self.listRectanglesa2[self.currentSegment].get_x()):int(self.listRectanglesa2[self.currentSegment].get_x()+self.listRectanglesa2[self.currentSegment].get_width())],self.cmap_grey,self.config['BirdButtons1'],self.config['BirdButtons2'], self.config['ListBirdsEntries'])
        self.updateText(label,self.currentSegment)
        if self.saveConfig:
            self.config['ListBirdsEntries'].append(label)
        self.humanClassifyNextImage()

    def spectrogramDialog(self):
        # Create the spectrogram dialog when the relevant button is pressed
        self.spectrogramDialog = Spectrogram()
        self.spectrogramDialog.show()
        self.spectrogramDialog.activateWindow()
        self.spectrogramDialog.activate.clicked.connect(self.spectrogram)

    def spectrogram(self):
        # Listener for the spectrogram dialog.
        [alg, colourStart, colourEnd, window_width, incr] = self.spectrogramDialog.getValues()
        self.sp.set_width(int(str(window_width)), int(str(incr)))
        self.sgRaw = self.sp.spectrogram(self.audiodata,str(alg))
        self.sg = np.abs(np.where(self.sgRaw==0,0.0,10.0 * np.log10(self.sgRaw)))
        print np.min(self.sg), np.max(self.sg)
        self.overviewImage.setImage(np.fliplr(self.sg.T))
        self.specPlot.setImage(np.fliplr(self.sg.T))
        # Colour scaling for the spectrograms
        self.overviewImage.setLevels([float(str(colourStart))*np.max(self.sg), float(str(colourEnd))*np.max(self.sg)])
        self.specPlot.setLevels([float(str(colourStart))*np.max(self.sg), float(str(colourEnd))*np.max(self.sg)])
        #self.drawfigMain()

    def denoiseDialog(self):
        # Create the denoising dialog when the relevant button is pressed
        # TODO: Anything to help setting bandpass levels?
        self.denoiseDialog = Denoise()
        self.denoiseDialog.show()
        self.denoiseDialog.activateWindow()
        self.denoiseDialog.activate.clicked.connect(self.denoise)
        self.denoiseDialog.undo.clicked.connect(self.denoise_undo)
        self.denoiseDialog.save.clicked.connect(self.denoise_save)

    def denoise(self):
        # Listener for the denoising dialog.
        # Calls the denoiser and then plots the updated data
        # TODO: should it be saved automatically, or a button added?
        [alg,depthchoice,depth,thrType,thr,wavelet,start,end,width] = self.denoiseDialog.getValues()
        # TODO: deal with these!
        # TODO: Undo needs testing

        if str(alg) == "Wavelets":
            if thrType is True:
                type = 'Soft'
            else:
                type = 'Hard'
            if depthchoice:
                depth = None
            else:
                depth = int(str(depth))
            if hasattr(self,'audiodata_backup'):
                if self.audiodata_backup is not None:
                    audiodata_backup_new = np.empty((np.shape(self.audiodata_backup)[0],np.shape(self.audiodata_backup)[1]+1))
                    audiodata_backup_new[:,:-1] = np.copy(self.audiodata_backup)
                    audiodata_backup_new[:,-1] = np.copy(self.audiodata)
                    self.audiodata_backup = audiodata_backup_new
                else:
                    self.audiodata_backup = np.empty((np.shape(self.audiodata)[0], 1))
                    self.audiodata_backup[:, 0] = np.copy(self.audiodata)
            else:
                self.audiodata_backup = np.empty((np.shape(self.audiodata)[0],1))
                self.audiodata_backup[:,0] = np.copy(self.audiodata)
            self.audiodata = self.sp.waveletDenoise(self.audiodata,type,float(str(thr)),depth,str(wavelet))
        elif str(alg) == "Wavelets + Bandpass":
            if thrType is True:
                type = 'soft'
            else:
                type = 'hard'
            if depthchoice:
                depth = None
            else:
                depth = int(str(depth))
            self.audiodata = self.sp.waveletDenoise(self.audiodata,float(str(thr)),int(str(depth)),str(wavelet))
            self.audiodata = self.sp.bandpassFilter(self.audiodata,int(str(start)),int(str(end)))
        elif str(alg) == "Bandpass":
            self.audiodata = self.sp.bandpassFilter(self.audiodata, int(str(start)), int(str(end)))
        else:
            #"Median Filter"
            self.audiodata = self.sp.medianFilter(self.audiodata,int(str(width)))

        print "Denoising"
        self.audiodata = self.sp.waveletDenoise()
        print "Done"
        self.sgRaw = self.sp.spectrogram(self.audiodata,self.sampleRate)
        self.sg = np.abs(np.where(self.sgRaw==0,0.0,10.0 * np.log10(self.sgRaw)))
        self.overviewImage.setImage(np.fliplr(self.sg.T))
        self.specPlot.setImage(np.fliplr(self.sg.T))


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
                    if hasattr(self,'seg'):
                        self.seg.setNewData(self.audiodata,self.sgRaw,self.sampleRate)
                    #self.drawfigMain()

    def denoise_save(self):
        # Listener for save button in denoising dialog
        # Save denoised data
        # Other players need them to be 16 bit, which is this magic number
        # TODO: with librosa, probably don't need magic number, but check
        #self.audiodata *= 32768.0
        #self.audiodata = self.audiodata.astype('int16')
        import soundfile as sf
        filename = self.filename[:-4] + '_d' + self.filename[-4:]
        sf.write(filename,self.audiodata,self.sampleRate,subtype='PCM_16')

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
        [alg, ampThr, medThr,HarmaThr1,HarmaThr2,depth,thrType,thr,wavelet,bandchoice,start,end] = self.segmentDialog.getValues()
        if not hasattr(self,'seg'):
            self.seg = Segment.Segment(self.audiodata,self.sgRaw,self.sp,self.sampleRate)
        if str(alg) == "Amplitude":
            newSegments = self.seg.segmentByAmplitude(float(str(ampThr)))
            # TODO: *** Next few lines need updating
            if hasattr(self, 'line'):
                if self.line is not None:
                    self.line.remove()
            self.line = self.a1.add_patch(pl.Rectangle((0,float(str(ampThr))),len(self.audiodata),0,facecolor='r'))
        elif str(alg) == "Median Clipping":
            newSegments = self.seg.medianClip(float(str(medThr)))
            print newSegments
        elif str(alg) == "Harma":
            print "here"
            newSegments = self.seg.Harma(float(str(HarmaThr1)),float(str(HarmaThr2)))
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
        # TODO: Tell user if there isn't a box highlighted
        #print self.box1id
        if not hasattr(self,'seg'):
            self.seg = Segment.Segment(self.audiodata,self.sgRaw,self.sp,self.sampleRate)

        if self.box1id is None or self.box1id == -1:
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
        #self.playButton.setText("Play")

    def sliderMoved(self):
        # When the slider is moved, change the position of playback
        print self.playSlider.value()
        self.media_obj.seek(self.playSlider.value())

    def movePlaySlider(self, time):
        if not self.playSlider.isSliderDown():
            self.playSlider.setValue(time)
        self.timePlayed.setText(self.convertMillisecs(time)+"/"+self.totalTime)
        val = 60.*(time / (1000 * 60)) % 60 + (time / 1000) % 60 + time/1000.
        self.bar.setValue(val)

    def setSliderLimits(self, start,end):
        self.playSlider.setRange(start, end)
        self.playSlider.setValue(start)
        self.media_obj.seek(start)

# ============
# Code for the buttons
    def deleteSegment(self):
        # Listener for delete segment button, or backspace key
        # Deletes segment if one is selected, otherwise does nothing
        if self.box1id>-1:
            self.p_ampl.removeItem(self.listRectanglesa1[self.box1id])
            #print "deleted", self.box1id
            self.p_spec.removeItem(self.listRectanglesa2[self.box1id])
            #self.listRectanglesa1.remove(self.listRectanglesa1[self.box1id])
            #self.listRectanglesa2.remove(self.listRectanglesa2[self.box1id])
            self.p_ampl.removeItem(self.listLabels[self.box1id])
            #self.a1text.remove(self.a1text[self.box1id])
            self.segments.remove(self.segments[self.box1id])
            self.box1id = -1

    def deleteAll(self):
        # Listener for delete all button
        self.segments=[]
        for r in self.listLabels:
            self.p_ampl.removeItem(r)
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
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Signal Processing Options')

        self.algs = QComboBox()
        self.algs.addItems(['Hann','Parzen','Welch','Hamming','Blackman','BlackmanHarris'])

        self.activate = QPushButton("Update Spectrogram")
        # TODO: Add option for multitapering
        # TODO: Make these two into sliders
        self.colourStart = QLineEdit(self)
        self.colourStart.setText('0.4')
        self.colourEnd = QLineEdit(self)
        self.colourEnd.setText('1.0')
        self.window_width = QLineEdit(self)
        self.window_width.setText('256')
        self.incr = QLineEdit(self)
        self.incr.setText('128')

        #self.tbox.setMaximumWidth(150)

        Box = QVBoxLayout()
        Box.addWidget(self.algs)
        Box.addWidget(QLabel('Colour Start'))
        Box.addWidget(self.colourStart)
        Box.addWidget(QLabel('Colour End'))
        Box.addWidget(self.colourEnd)
        Box.addWidget(QLabel('Window Width'))
        Box.addWidget(self.window_width)
        Box.addWidget(QLabel('Hop'))
        Box.addWidget(self.incr)
        #Box.addWidget(self.ampThr)
        Box.addWidget(self.activate)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        return [self.algs.currentText(),self.colourStart.text(),self.colourEnd.text(),self.window_width.text(),self.incr.text()]

class Segmentation(QDialog):
    # Class for the segmentation dialog box
    # TODO: add the wavelet params
    # TODO: work out how to return varying size of params, also process them
    # TODO: test and play
    def __init__(self, maxv, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Segmentation Options')

        self.algs = QComboBox()
        self.algs.addItems(["Amplitude","Energy Curve","Harma","Median Clipping","Wavelets"])
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
        self.start = QLineEdit(self)
        self.start.setText('400')
        self.end = QLineEdit(self)
        self.end.setText('1000')
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
        return [self.algs.currentText(),self.ampThr.text(),self.medThr.text(),self.HarmaThr1.text(),self.HarmaThr2.text(),self.depth.text(),self.thrtype[0].isChecked(),self.thr.text(),self.wavelet.currentText(),self.bandchoice.isChecked(),self.start.text(),self.end.text()]

class Denoise(QDialog):
    # Class for the denoising dialog box
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Denoising Options')

        self.algs = QComboBox()
        self.algs.addItems(["Wavelets","Wavelets + Bandpass","Bandpass","Median Filter"])
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
        self.start.setText('400')
        self.end = QLineEdit(self)
        self.end.setText('1000')

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
        elif self.prevAlg == "Wavelets + Bandpass":
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
        elif self.prevAlg == "Bandpass":
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
        elif str(alg) == "Wavelets + Bandpass":
            self.wblabel.show()
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
        elif str(alg) == "Bandpass":
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
    def __init__(self, seg, cmap_grey, bb1, bb2, bb3, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Correct Classification')
        self.frame = QWidget()
        self.cmap_grey = cmap_grey

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

# class OpeningLinearRegionItem(pg.LinearRegionItem):
# This might be useful for picking out when a box is clicked, but it isn't perfect since there will be multiple signals sent
#    sigMouseMoved = QtCore.Signal(object)
#    def __init__(self):
#        pg.LinearRegionItem.__init__(self)

#    def mouseMovedEvent(self, ev):
#        print "it works!"

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


# Start the application
app = QApplication(sys.argv)
form = Interface(configfile='AviaNZconfig.txt')
form.show()
app.exec_()
