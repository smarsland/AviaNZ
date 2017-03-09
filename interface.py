# Interface.py
#
# This is the main class for the AviaNZ interface
# It's fairly simple, but seems to work OK
# Version 0.7 5/01/17
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

import sys, os, glob, json, datetime
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import PyQt4.phonon as phonon

import librosa as lr
import numpy as np
import pylab as pl
from threading import Thread

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import sounddevice as sd

import SignalProc
import Segment
import Features
#import Learning
# ==============
# TODO

# Test the init part about if file or directory doesn't exist
# Finish implementation for button to show individual segments to user and ask for feedback
# Ditto lots of segments at once
# Finish sorting out parameters for median clipping segmentation, energy segmentation
# Finish the raven features
# ** Sort out the wavelet segmentation
# Finish cross-correlation to pick out similar bits of spectrogram
# Sound playback:
    # make it only play back the visible section
    # replace slider?, finish
# Implement something for the Classify button:
    # Take the segments that have been given and try to classify them in lots of ways:
    # Cross-correlation, DTW, shape metric, features and learning

# Testing data
# Documentation
# Licensing

# Some things need a mini method, e.g. text resizing

# Decide on a colour code for segmentation -> add text somewhere!
# Use intensity of colour to encode certainty?

# Some bug in denoising? -> tril1
# How to set bandpass params? -> is there a useful plot to help?
# More features, add learning!
# Pitch tracking, fundamental frequency
# Other parameters for dialogs?
#      multitaper for spectrogram

# Size of zoom window?

# Needs decent testing

# Option to turn button menu on/off

# Speed is an issue, unfortunately
    # I think that is largely matplotlib
    # Would Pyqtgraph help? What can be actually done in it?
    # What can be speeded up? Spectrogram in C? It probably is. Denoising!

# Minor:
# Turn stereo sound into mono using librosa, consider always resampling to 22050Hz (except when it's less in file :) )
# Font size to match segment size -> currently make it smaller, could also move it up or down as appropriate

# Things to consider:
    # Second spectrogram (currently use right button for interleaving)? My current choice is no as it takes up space
    # Put the labelling (and loading) in a dialog to free up space? -> bigger plots
    # Useful to have a go to start button as well as forward and backward?
    # How to show a moving point through the file as it plays? Sound has been a pain in the arse to date!

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

# ===============

class Interface(QMainWindow):
    # Main class for the interface, which contains most of the user interface and plotting code
    # All plotting is done in Matplotlib inside a PyQt window

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

        self.resetStorageArrays()
        #self.secondSpectrogram = False

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

            # Params for the width of the lines to draw for segmentation in the two plots
            'linewidtha1': 0.005,
            'minbar_thickness': 200,
            'linewidtha2': 0.03,

            # Params for the extra padding in the second figure, and how much keys move bars by
            # All as percentage of width
            'padding': 0.15,
            'move_amount': 0.01,

            # Param for width in seconds of the main representation
            'windowWidth': 10.0,  #

            # These are the contrast parameters for the spectrogram
            'colourStart': 0.4,
            'colourEnd': 1.0,

            # Params for cross-correlation and related
            'corrThr': 0.4,

            'dpi': 100,
            'BirdButtons1': ["Bellbird", "Bittern", "Cuckoo", "Fantail", "Hihi", "Kakapo", "Kereru", "Kiwi (F)", "Kiwi (M)",
                             "Petrel"],
            'BirdButtons2': ["Rifleman", "Ruru", "Saddleback", "Silvereye", "Tomtit", "Tui", "Warbler", "Not Bird",
                             "Don't Know", "Other"],
            'ListBirdsEntries': ['Albatross', 'Avocet', 'Blackbird', 'Bunting', 'Chaffinch', 'Egret', 'Gannet', 'Godwit',
                                 'Gull', 'Kahu', 'Kaka', 'Kea', 'Kingfisher', 'Kokako', 'Lark', 'Magpie', 'Plover',
                                 'Pukeko', "Rooster" 'Rook', 'Thrush', 'Warbler', 'Whio'],
        }

    def createFrame(self):
        # This creates the actual interface. A bit of a mess of Qt widgets, the connector calls, and layouts.

        # Overview                  < >
        # Main plot
        # Instructions, Buttons     Width
        # Files     Zoom            Buttons
        
        self.defineColourmap()
        self.frame = QWidget()

        # This is the overview picture of the whole input, enabling the user to move around it
        self.figOverview = Figure((22.0, 2.0), dpi=self.config['dpi'])
        self.canvasOverview = FigureCanvas(self.figOverview)
        self.canvasOverview.setParent(self.frame)
        self.canvasOverview.mpl_connect('button_press_event', self.figOverviewClick)
        self.canvasOverview.mpl_connect('motion_notify_event', self.figOverviewDrag)

        # The buttons to move through the overview
        self.leftBtn = QToolButton()
        self.leftBtn.setArrowType(Qt.LeftArrow)
        self.connect(self.leftBtn, SIGNAL('clicked()'), self.moveLeft)
        self.rightBtn = QToolButton()
        self.rightBtn.setArrowType(Qt.RightArrow)
        self.connect(self.rightBtn, SIGNAL('clicked()'), self.moveRight)

        # The layout for the overview
        hboxOverview = QHBoxLayout()
        hboxOverview.addWidget(self.canvasOverview)
        hboxOverview.addWidget(self.leftBtn)
        hboxOverview.addWidget(self.rightBtn)

        # This is the main figure, where the user selects things
        self.figMain = Figure(dpi=self.config['dpi'])
        self.figMain.set_size_inches(22.0, 4.0, forward=True)
        self.canvasMain = FigureCanvas(self.figMain)
        self.canvasMain.setParent(self.frame)

        # Connectors for the click for start and end bars and other interactions on figMain
        self.canvasMain.mpl_connect('button_press_event', self.figMainClick)
        self.canvasMain.setFocusPolicy(Qt.ClickFocus)
        self.canvasMain.setFocus()
        #self.mpl_toolbar = NavigationToolbar(self.canvas, self.frame)

        # The instructions and buttons below figMain
        # playButton1 = QPushButton(QIcon(":/Resources/play.svg"),"&Play Window")
        self.playButton1 = QPushButton("&Play")
        self.connect(self.playButton1, SIGNAL('clicked()'), self.playSegment)
        resetButton1 = QPushButton("&Reset")
        self.connect(resetButton1, SIGNAL('clicked()'), self.resetSegment)

        # Checkbox for whether or not user is drawing boxes around song in the spectrogram (defaults to clicks not drags)
        self.dragRectangles = QCheckBox('Drag boxes in spectrogram')
        self.dragRectangles.stateChanged[int].connect(self.dragRectanglesCheck)

        # A slider to show playback position
        # TODO: Experiment with other options -- bar in the window
        self.playSlider = QSlider(Qt.Horizontal, self)
        self.connect(self.playSlider,SIGNAL('sliderReleased()'),self.sliderMoved)

        hboxButtons1 = QHBoxLayout()
        hboxButtons1.addWidget(QLabel(
            'Slide top box to move through recording, click to start and end a segment, click on segment to edit or label. Right click to interleave.'))
        hboxButtons1.addWidget(self.playButton1)
        hboxButtons1.addWidget(resetButton1)
        hboxButtons1.addWidget(self.dragRectangles)

        # The spinbox for changing the width shown in figMain
        self.widthWindow = QDoubleSpinBox()
        self.widthWindow.setRange(0.5, 900.0)
        self.widthWindow.setSingleStep(1.0)
        self.widthWindow.setDecimals(2)
        self.widthWindow.setValue(self.config['windowWidth'])
        self.widthWindow.valueChanged[float].connect(self.changeWidth)

        vboxWidthWindow = QVBoxLayout()
        vboxWidthWindow.addWidget(QLabel('Visible window width (seconds)'))
        vboxWidthWindow.addWidget(self.widthWindow)

        # The layouts for the top half of the screen
        hboxBelowMain = QHBoxLayout()
        hboxBelowMain.addLayout(hboxButtons1)
        hboxBelowMain.addLayout(vboxWidthWindow)

        vboxTopHalf = QVBoxLayout()
        vboxTopHalf.addLayout(hboxOverview)
        vboxTopHalf.addWidget(self.canvasMain)
        vboxTopHalf.addLayout(hboxBelowMain)
        vboxTopHalf.addWidget(self.playSlider)

        #vboxTopHalf.addWidget(self.mpl_toolbar)

        # Next is the bottom part of the screen

        # List to hold the list of files
        self.listFiles = QListWidget(self)
        self.listFiles.setFixedWidth(150)
        self.fillList()
        self.listFiles.connect(self.listFiles, SIGNAL('itemDoubleClicked(QListWidgetItem*)'), self.listLoadFile)

        vboxListFiles = QVBoxLayout()
        vboxListFiles.addWidget(self.listFiles)
        vboxListFiles.addWidget(QLabel('Double click to select'))
        vboxListFiles.addWidget(QLabel('Red names have segments'))

        # This is the zoomed-in figure
        self.figZoom = Figure((6.0, 4.0), dpi=self.config['dpi'])
        self.canvasZoom = FigureCanvas(self.figZoom)
        self.canvasZoom.setParent(self.frame)
        self.canvasZoom.setFocusPolicy(Qt.ClickFocus)
        self.canvasZoom.setFocus()

        # The buttons that go below figZoom
        playButton2 = QPushButton("&Play")
        self.connect(playButton2, SIGNAL('clicked()'), self.play)
        self.addButton = QPushButton("&Add segment")
        self.connect(self.addButton, SIGNAL('clicked()'), self.addSegmentClick)

        hboxButtons3 = QHBoxLayout()
        for w in [playButton2,self.addButton]:
            hboxButtons3.addWidget(w)
            hboxButtons3.setAlignment(w, Qt.AlignVCenter)

        # Layout for FigZoom
        vboxFigZoom = QVBoxLayout()
        vboxFigZoom.addWidget(self.canvasZoom)
        vboxFigZoom.addWidget(QLabel('Click on a start/end to select, click again to move'))
        vboxFigZoom.addWidget(QLabel('Or use arrow keys. Press Backspace to delete'))
        vboxFigZoom.addLayout(hboxButtons3)

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

        vboxButtons2 = QVBoxLayout()
        for w in [deleteButton,deleteAllButton,spectrogramButton,denoiseButton, segmentButton, findMatchButton, quitButton, checkButton]:
            vboxButtons2.addWidget(w)
            #vboxButtons2.setAlignment(w, Qt.AlignHCenter)

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

        # An array of radio buttons and a list and a text entry box
        # Create an array of radio buttons for the most common birds (2 columns of 10 choices)
        # self.birds1 = []
        # for item in self.config['BirdButtons1']:
        #     self.birds1.append(QRadioButton(item))
        # self.birds2 = []
        # for item in self.config['BirdButtons2']:
        #     self.birds2.append(QRadioButton(item))
        #
        # for i in xrange(len(self.birds1)):
        #     self.birds1[i].setEnabled(False)
        #     self.connect(self.birds1[i], SIGNAL("clicked()"), self.radioBirdsClicked)
        # for i in xrange(len(self.birds2)):
        #     self.birds2[i].setEnabled(False)
        #     self.connect(self.birds2[i], SIGNAL("clicked()"), self.radioBirdsClicked)

        # The list of less common birds
        # self.birdList = QListWidget(self)
        # self.birdList.setMaximumWidth(150)
        # for item in self.config['ListBirdsEntries']:
        #     self.birdList.addItem(item)
        # self.birdList.sortItems()
        # # Explicitly add "Other" option in
        # self.birdList.insertItem(0,'Other')

        # self.connect(self.birdList, SIGNAL("itemClicked(QListWidgetItem*)"), self.listBirdsClicked)
        # self.birdList.setEnabled(False)

        # This is the text box for missing birds
        # self.tbox = QLineEdit(self)
        # self.tbox.setMaximumWidth(150)
        # self.connect(self.tbox, SIGNAL('editingFinished()'), self.birdTextEntered)
        # self.tbox.setEnabled(False)

        # birds1Layout = QVBoxLayout()
        # for i in xrange(len(self.birds1)):
        #     birds1Layout.addWidget(self.birds1[i])
        #
        # birds2Layout = QVBoxLayout()
        # for i in xrange(len(self.birds2)):
        #     birds2Layout.addWidget(self.birds2[i])
        #
        # birdListLayout = QVBoxLayout()
        # birdListLayout.addWidget(self.birdList)
        # birdListLayout.addWidget(QLabel("If bird isn't in list, select Other"))
        # birdListLayout.addWidget(QLabel("Type below, Return at end"))
        # birdListLayout.addWidget(self.tbox)

        hboxBottomHalf = QHBoxLayout()
        hboxBottomHalf.addLayout(vboxListFiles)
        #hboxBottomHalf.addLayout(birds1Layout)
        #hboxBottomHalf.addLayout(birds2Layout)
        #hboxBottomHalf.addLayout(birdListLayout)
        hboxBottomHalf.addLayout(vboxFigZoom)
        hboxBottomHalf.addLayout(vboxButtons2)

        # And the layout for the whole thing
        vboxFull = QVBoxLayout()
        vboxFull.addLayout(vboxTopHalf)
        vboxFull.addLayout(hboxBottomHalf)

        # Now put everything into the frame
        self.frame.setLayout(vboxFull)
        self.setCentralWidget(self.frame)

        # Instantiate a Qt media object and prepare it
        self.media_obj = phonon.Phonon.MediaObject(self)
        self.audio_output = phonon.Phonon.AudioOutput(phonon.Phonon.MusicCategory, self)
        phonon.Phonon.createPath(self.media_obj, self.audio_output)
        self.media_obj.setTickInterval(20)
        self.media_obj.tick.connect(self.movePlaySlider)
        self.media_obj.finished.connect(self.playFinished)

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
        self.segments=[]
        self.listRectanglesa1 = []
        self.listRectanglesa2 = []
        self.listRectanglesb1 = []
        self.listRectanglesb2 = []
        self.a1text = []

        # This is a flag to say if the next thing that they click on should be a start or a stop for segmentation
        self.start_stop = False
        self.start_stop2 = False

        # Keep track of start points and selected buttons
        self.start_a = 0
        self.windowStart = 0
        self.playPosition = self.windowStart
        self.box1id = None
        self.buttonID = None

        self.line = None

        self.recta1 = None
        self.recta2 = None
        self.focusRegionSelected = False
        self.figMainSegment1 = None
        self.figMainSegment2 = None
        self.figMainSegmenting = False

        self.playbar1 = None
        self.isPlaying = False

    def listLoadFile(self,current):
        # Listener for when the user clicks on a filename
        # Saves segments of current file, resets flags and calls loader
        self.a1.clear()
        self.a2.clear()
        if self.previousFile is not None:
            if self.segments != []:
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
        #if len(self.segments)>0:
        #    self.saveSegments()
        #self.segments = []

        if isinstance(name,str):
            self.filename = self.dirName+'/'+name
        else:
            self.filename = self.dirName+'/'+str(name.text())
        self.audiodata, self.sampleRate = lr.load(self.filename,sr=None)
        #self.sampleRate, self.audiodata = wavfile.read(self.filename)
        # None of the following should be necessary for librosa
        #if self.audiodata.dtype is not 'float':
        #    self.audiodata = self.audiodata.astype('float') / 32768.0
        #if np.shape(np.shape(self.audiodata))[0]>1:
        #    self.audiodata = self.audiodata[:,0]
        self.datalength = np.shape(self.audiodata)[0]
        print("Length of file is ",len(self.audiodata),float(self.datalength)/self.sampleRate)

        # Create an instance of the Signal Processing class
        if not hasattr(self,'sp'):
            self.sp = SignalProc.SignalProc(self.audiodata, self.sampleRate,self.config['window_width'],self.config['incr'])

        # Get the data for the spectrogram
        # TODO: put a button for multitapering somewhere
        self.sg = self.sp.spectrogram(self.audiodata,self.sampleRate,multitaper=False)

        # Load any previous segments stored
        if os.path.isfile(self.filename+'.data'):
            file = open(self.filename+'.data', 'r')
            self.segments = json.load(file)
            file.close()

        # Update the data that is seen by the other classes
        # TODO: keep an eye on this to add other classes as required
        if hasattr(self,'seg'):
            self.seg.setNewData(self.audiodata,self.sg,self.sampleRate)
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

        # Reset it if the file is shorter than the window
        if float(len(self.audiodata))/self.sampleRate < self.windowSize:
            self.windowSize = float(len(self.audiodata))/self.sampleRate
        self.widthWindow.setValue(self.windowSize)

        # Set the width of the segment marker
        # This is the config width as a percentage of the window width, which is in seconds
        self.linewidtha1 = float(self.config['linewidtha1'])*self.windowSize

        # Load the file for playback as well, and connect up the listeners for it
        self.media_obj.setCurrentSource(phonon.Phonon.MediaSource(self.filename))

        # Decide on the length of the playback bit for the slider
        self.setSliderLimits(0,self.media_obj.totalTime())

        # Get the height of the amplitude for plotting the box
        self.plotheight = np.abs(np.min(self.audiodata)) + np.max(self.audiodata)
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
            self.figMainmotion = self.canvasMain.mpl_connect('motion_notify_event', self.figMainDrag)
            self.figMainendmotion = self.canvasMain.mpl_connect('button_release_event', self.figMainDragEnd)
        else:
            self.canvasMain.mpl_disconnect(self.figMainmotion)
            self.canvasMain.mpl_disconnect(self.figMainendmotion)

# ==============
# Code for drawing and using the main figure
    def drawfigMain(self):
        # This draws the main figure, amplitude and spectrogram plots
        # Also the overview figure to show where you are up to in the file

        self.a1 = self.figMain.add_subplot(211)
        self.a1.clear()
        self.a1.set_xlim(self.windowStart, self.windowSize)
        self.a1.plot(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=self.datalength,endpoint=True),self.audiodata)
        #self.a1.axis('off')

        self.a2 = self.figMain.add_subplot(212)
        self.a2.clear()
        self.a2.imshow(10.0*np.log10(self.sg), cmap=self.cmap_grey, aspect='auto')

        # Sort out the check marks for the x axis
        l = [str(0),str(self.sampleRate)]
        for i in range(len(self.a2.axes.get_yticklabels())-4):
            l.append('')
        l.append(str(0))
        self.a2.axes.set_yticklabels(l)
        self.a2.set_xlim(self.windowStart*self.sampleRate / self.config['incr'], self.windowSize*self.sampleRate / self.config['incr'])

        # If there are segments, show them
        for count in range(len(self.segments)):
            if self.segments[count][4] == 'None' or self.segments[count][4] == "Don't Know":
                facecolour = 'r'
            else:
                facecolour = 'b'
            a1R = self.a1.add_patch(pl.Rectangle((self.segments[count][0], np.min(self.audiodata)),
                                                 self.segments[count][1] - self.segments[count][0],
                                                 self.plotheight,
                                                 facecolor=facecolour,
                                                 alpha=0.5))
            a2R = self.a2.add_patch(pl.Rectangle((self.segments[count][0]*self.sampleRate / self.config['incr'], self.segments[count][2]),
                                                 self.segments[count][1]*self.sampleRate / self.config['incr'] - self.segments[count][
                                                     0]*self.sampleRate / self.config['incr'],
                                                 self.segments[count][3], facecolor=facecolour,
                                                 alpha=0.5))
            # These store the ids for the rectangles
            self.listRectanglesa1.append(a1R)
            self.listRectanglesa2.append(a2R)
            a1t = self.a1.text(self.segments[count][0], np.min(self.audiodata), self.segments[count][4])

            # The font size is a pain because the window is in pixels, so have to transform it
            # Force a redraw to make bounding box available
            self.canvasMain.draw()
            # fs = a1t.get_fontsize()
            width = a1t.get_window_extent().inverse_transformed(self.a1.transData).width
            #print width, self.segments[count][1] - self.segments[count][0]
            if width > self.segments[count][1] - self.segments[count][0]:
                a1t.set_fontsize(8)

            self.a1text.append(a1t)

            # These are alternative ways to print the annotations that would enable putting it not inside the axes
            # But they are relative coordinates, so would be fiddly and have to move with the slider -> annoying!
            # self.a1.annotate()
            # self.figMain.text(0.3,0.5,'xxx')
        self.figMain.subplots_adjust(0.05,0.02,0.99,0.98)
        self.canvasMain.draw()

        # Draw the top (overview) figure
        self.topfig = self.figOverview.add_axes((0.03, 0.01, 0.99, 0.98))
        self.topfig.imshow(10.0*np.log10(self.sg), cmap=self.cmap_grey,aspect='auto')
        self.topfig.axis('off')
        if self.focusRegion is not None:
            self.focusRegion.remove()
        self.focusRegion = self.topfig.add_patch(pl.Rectangle((self.windowStart*self.sampleRate/self.config['incr'], 0),
                                             self.windowSize*self.sampleRate / self.config['incr'],
                                             self.config['window_width'] / 2,
                                             facecolor='r',
                                             alpha=0.5))
        self.canvasOverview.draw()
        self.canvasMain.mpl_connect('key_press_event', self.figMainKeypress)

    def figMainDrag(self,event):
        # This is the start listener for the user dragging boxes on the spectrogram
        # Just gets coordinates and draws a box outline as the mouse moves
        if self.recta2 is None:
            return
        if event.inaxes is None:
            return
        x0,y0 = self.recta2.get_xy()
        a1 = str(self.a1)
        a1ind = float(a1[a1.index(',') + 1:a1.index(';')])
        a = str(event.inaxes)
        aind = float(a[a.index(',') + 1:a.index(';')])
        if aind == a1ind:
            return

        self.recta2.set_width(event.xdata - x0)
        self.recta2.set_height(event.ydata - y0)
        self.canvasMain.draw()

    def figMainDragEnd(self,event):
        # This is the end listener for the user dragging boxes on the spectrogram
        # Deletes the outline box and replaces it with a Rectangle patch, adds the segment
        if event.inaxes is None:
            return
        if self.recta2 is None:
            return
        a1 = str(self.a1)
        a1ind = float(a1[a1.index(',') + 1:a1.index(';')])
        a = str(event.inaxes)
        aind = float(a[a.index(',') + 1:a.index(';')])
        if aind == a1ind:
            return

        ampl_x0 = self.recta2.get_x()*self.config['incr']/self.sampleRate
        ampl_x1 = self.recta2.get_x()*self.config['incr']/self.sampleRate+self.recta2.get_width()*self.config['incr']/self.sampleRate

        a1R = self.a1.add_patch(pl.Rectangle((min(ampl_x0,ampl_x1),self.a1.get_ylim()[0]), self.recta2.get_width()*self.config['incr']/self.sampleRate, self.plotheight, alpha=0.5, facecolor='g'))
        a2R = self.a2.add_patch(pl.Rectangle(self.recta2.get_xy(), self.recta2.get_width(), self.recta2.get_height(), alpha=0.5, facecolor='g'))

        self.listRectanglesa1.append(a1R)
        self.listRectanglesa2.append(a2R)

        self.segments.append([min(ampl_x0,ampl_x1), max(ampl_x0,ampl_x1), self.recta2.get_y(), self.recta2.get_height(),'None'])
        a1t = self.a1.text(min(ampl_x0,ampl_x1), np.min(self.audiodata), 'None')
        # The font size is a pain because the window is in pixels, so have to transform it
        # Force a redraw to make bounding box available
        self.canvasMain.draw()
        # fs = a1t.get_fontsize()
        width = a1t.get_window_extent().inverse_transformed(self.a1.transData).width
        if width > self.segments[-1][1] - self.segments[-1][0]:
            a1t.set_fontsize(8)

        self.a1text.append(a1t)
        self.topBoxCol = 'r'
        # Show it in the zoom window
        self.zoomstart = a1R.get_x()
        self.zoomend = a1R.get_x() + a1R.get_width()
        self.box1id = len(self.segments) - 1
        self.drawfigZoom()
        # Activate the radio buttons for labelling, selecting one from label if necessary
        #self.activateRadioButtons()
        self.recta2.remove()
        self.recta2 = None
        self.canvasMain.draw()

    def birdSelected(self,birdname):
        # This collects the label for a bird from the context menu and processes it
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

    def figMainClick(self, event):
        # When user clicks in figMain there are several options:
        # Right click always defines a new box (first marking the start, then the end (add box, save segment)
        # Otherwise, if click in a box that is already made, select it, show zoom-in version, prepare to label
        # There is a checkbox to allow dragging of a box on the spectrogram
        # Note that a smaller box could become unreachable in interleaving.

        # If there was a previous box, turn it the right colour again
        if self.box1id>-1:
            self.listRectanglesa1[self.box1id].set_facecolor(self.topBoxCol)
            self.listRectanglesa2[self.box1id].set_facecolor(self.topBoxCol)

        # Deactivate the radio buttons and listbox
        # for i in xrange(len(self.birds1)):
        #     self.birds1[i].setChecked(False)
        #     self.birds1[i].setEnabled(False)
        # for i in xrange(len(self.birds2)):
        #     self.birds2[i].setChecked(False)
        #     self.birds2[i].setEnabled(False)
        # self.birdList.setEnabled(False)
        # self.tbox.setEnabled(False)

        # Check if the user has clicked in a box
        box1id = -1
        for count in range(len(self.listRectanglesa1)):
            if self.listRectanglesa1[count].xy[0] <= event.xdata and self.listRectanglesa1[count].xy[0]+self.listRectanglesa1[count].get_width() >= event.xdata:
                box1id = count
            if self.listRectanglesa2[count].xy[0] <= event.xdata and self.listRectanglesa2[count].xy[0]+self.listRectanglesa2[count].get_width() >= event.xdata:
                box1id = count

        # If they have clicked in a box, store its colour for later resetting
        # Show the zoomed-in version and enable the radio buttons
        if box1id>-1 and event.button==1:
            self.box1id = box1id
            self.topBoxCol = self.listRectanglesa1[box1id].get_facecolor()
            self.listRectanglesa1[box1id].set_facecolor('green')
            self.listRectanglesa2[box1id].set_facecolor('green')
            self.zoomstart = self.listRectanglesa1[box1id].get_x()
            self.zoomend = self.listRectanglesa1[box1id].get_x()+self.listRectanglesa1[box1id].get_width()
            self.drawfigZoom()

            # Put the context menu at the right point. The event.x and y are in pixels relative to the bottom left-hand corner of canvasMain (self.figMain)
            # Need to be converted into pixels relative to top left-hand corner of the window!
            self.menuBirdList.popup(QPoint(event.x + event.canvas.x() + self.frame.x() + self.x(),
                                           event.canvas.height() - event.y + event.canvas.y() + self.frame.y() + self.y()))

            # Activate the radio buttons for labelling, selecting one from label if necessary
            # self.activateRadioButtons()
        else:
            # User is doing segmentation. Check if the checkbox for dragging on spectrogram is checked.
            # Work out which axes they have clicked in. s holds the point they have clicked on
            # in coordinates of the top plot (amplitude)
            # TODO: Should the bottom figure show the amplitude box, and be able to change it?
            if event.inaxes is None:
                return
            # Work out which axes have been clicked on
            a1 = str(self.a1)
            a1ind = float(a1[a1.index(',') + 1:a1.index(';')])
            a = str(event.inaxes)
            aind = float(a[a.index(',') + 1:a.index(';')])

            if aind != a1ind and self.dragRectangles.isChecked():
                # User is selecting a region **on the spectrogram only**
                # The region is drawn as full size on the amplitude plot
                if event.xdata > 0 and event.xdata < float(self.datalength) / self.config['incr']:
                    self.recta2 = self.a2.add_patch(pl.Rectangle((event.xdata, event.ydata), 0.0, 0.0, alpha=1, facecolor='none'))
            else:
                s = event.xdata
                if aind != a1ind:
                    s = s*self.config['incr']/self.sampleRate
                if s>0 and s<float(self.datalength)/self.sampleRate:
                    if not self.start_stop:
                        # This is the start of a segment, draw a green line
                        self.markstarta1 = self.a1.add_patch(pl.Rectangle((s, np.min(self.audiodata)), self.linewidtha1, self.plotheight, facecolor='g', edgecolor='None',alpha=0.8))
                        self.markstarta2 = self.a2.add_patch(pl.Rectangle((s*self.sampleRate / self.config['incr'] , 0), self.linewidtha1/self.config['incr']*self.sampleRate, self.config['window_width']/2, facecolor='g', edgecolor='None',alpha=0.8))
                        self.start_a = s
                        self.segmenting = self.canvasMain.mpl_connect('motion_notify_event', self.figMainSelecting)
                        self.figMainSegment1 = self.a1.add_patch(pl.Rectangle((s+self.linewidtha1, np.min(self.audiodata)), 0, self.plotheight, facecolor='r', edgecolor='None',alpha=0.4))
                        self.figMainSegment2 = self.a2.add_patch(pl.Rectangle(((s+self.linewidtha1)*self.sampleRate / self.config['incr'], 0), 0, self.config['window_width']/2, facecolor='r', edgecolor='None', alpha=0.4))
                    else:
                        # This is the end, draw the box, save the data, update the text, make radio buttons available
                        # And show it in the zoom window
                        self.markstarta1.remove()
                        self.markstarta2.remove()

                        # Check if the ends have been drawn backwards
                        if self.start_a > s:
                            start = s
                            width = self.start_a - s
                        else:
                            start = self.start_a
                            width = s - self.start_a

                        # Draw the segments, add them to the list
                        a1R = self.a1.add_patch(pl.Rectangle((start, np.min(self.audiodata)), width, self.plotheight,facecolor='g', alpha=0.5))
                        a2R = self.a2.add_patch(pl.Rectangle((start*self.sampleRate / self.config['incr'], 0), width*self.sampleRate / self.config['incr'], self.config['window_width']/2,facecolor='g', alpha=0.5))
                        self.listRectanglesa1.append(a1R)
                        self.listRectanglesa2.append(a2R)
                        self.segments.append([start,max(self.start_a,s),0.0,self.sampleRate/2.,'None'])
                        a1t = self.a1.text(start, np.min(self.audiodata), 'None')
                        self.a1text.append(a1t)

                        # The font size is a pain because the window is in pixels, so have to transform it
                        # Force a redraw to make bounding box available
                        self.canvasMain.draw()
                        #fs = a1t.get_fontsize()
                        win_width = a1t.get_window_extent().inverse_transformed(self.a1.transData).width
                        #print width, win_width
                        if win_width > width:
                            a1t.set_fontsize(8)

                        # Delete the box and listener for the shading
                        self.canvasMain.mpl_disconnect(self.segmenting)
                        self.figMainSegment1.remove()
                        self.figMainSegment2.remove()

                        # Show it in the zoom window
                        self.zoomstart = a1R.xy[0]
                        self.zoomend = a1R.xy[0] + a1R.get_width()
                        self.box1id = len(self.segments)-1
                        self.drawfigZoom()
                        self.topBoxCol = 'r'
                        self.menuBirdList.popup(QPoint(event.x + event.canvas.x() + self.frame.x() + self.x(),
                                                       event.canvas.height() - event.y + event.canvas.y() + self.frame.y() + self.y()))
                        # Activate the radio buttons for labelling, selecting one from label if necessary
                        #self.activateRadioButtons()
                    # Switch to know if start or end or segment
                    self.start_stop = not self.start_stop

        self.canvasMain.draw()

    def figMainSelecting(self,event):
        # When the user is drawing a box in the main plot, this makes the red box they are drawing move as they move the mouse
        if not self.start_stop:
            return
        if event.inaxes is None:
            return

        # Work out which axes have been clicked on
        a1 = str(self.a1)
        a1ind = float(a1[a1.index(',') + 1:a1.index(';')])
        a = str(event.inaxes)
        aind = float(a[a.index(',') + 1:a.index(';')])

        if aind == a1ind:
            width = event.xdata - self.figMainSegment1.get_x()
        else:
            width = (event.xdata - self.figMainSegment2.get_x()) / self.sampleRate * self.config['incr']

        self.figMainSegment1.set_width(width)
        self.figMainSegment2.set_width(width * self.sampleRate / self.config['incr'])
        self.canvasMain.draw()

    def figMainKeypress(self,event):
        # Listener for any key presses when focus is on figMain
        # Currently just allows deleting
        # TODO: anything else?
        if event.key == 'backspace':
            self.deleteSegment()

    def figOverviewClick(self,event):
        # Listener 1 for the overview figure
        # Move the top box by first clicking on it, and then clicking again at the start of where you want it
        # Ignores other clicks
        if event.inaxes is None:
            return
        if self.focusRegionSelected == False:
            if self.focusRegion.get_x() <= event.xdata and self.focusRegion.get_x() + self.focusRegion.get_width() >= event.xdata:
                self.focusRegionSelected = True
                self.focusRegionPoint = event.xdata-self.focusRegion.get_x()
                self.focusRegion.set_facecolor('b')
                self.canvasOverview.draw()
        else:
            self.windowStart = (event.xdata-self.focusRegionPoint)/self.sampleRate*self.config['incr']
            if self.windowStart < 0:
                self.windowStart = 0
            elif self.windowStart + self.windowSize  > float(self.datalength) / self.sampleRate:
                self.windowStart = float(self.datalength) / self.sampleRate - self.windowSize
            self.playPosition = self.windowStart
            self.focusRegionSelected = False
            self.focusRegion.set_facecolor('r')
            self.updateMainWindow()

    def figOverviewDrag(self,event):
        # Listener 2 for the overview figure
        # Lets you drag the highlight box
        if event.inaxes is None:
            return
        if event.button!=1:
            return
        self.windowStart = event.xdata/self.sampleRate*self.config['incr']
        self.playPosition = self.windowStart
        self.updateMainWindow()

    def updateMainWindow(self):
        # Updates the main figure when the overview highlight box is moved
        self.a1.set_xlim(self.windowStart, self.windowStart+self.windowSize)
        self.a2.set_xlim(self.windowStart*self.sampleRate/self.config['incr'], (self.windowStart + self.windowSize)*self.sampleRate/self.config['incr'])

        self.focusRegion.set_x(self.windowStart*self.sampleRate/self.config['incr'])
        self.focusRegion.set_width(self.windowSize*self.sampleRate / self.config['incr'])
        #self.focusRegion.remove()
        #self.focusRegion = self.topfig.add_patch(pl.Rectangle((self.windowStart*self.sampleRate/self.config['incr'], 0),
        #                                                      self.windowSize*self.sampleRate / self.config['incr'],
        #                                                      self.config['window_width'] / 2,
        #                                                      facecolor='r',
        #                                                      alpha=0.5))
        self.canvasMain.draw()
        self.canvasOverview.draw()

    def updateText(self, text, boxid=None):
        if boxid is None:
            boxid = self.box1id
        # When the user changes the name in a segment, update the text
        self.segments[boxid][4] = text
        self.a1text[boxid].set_text(text)
        # The font size is a pain because the window is in pixels, so have to transform it
        # Force a redraw to make bounding box available
        self.canvasMain.draw()
        # fs = self.a1text[boxid].get_fontsize()
        width = self.a1text[boxid].get_window_extent().inverse_transformed(self.a1.transData).width
        if width > self.segments[boxid][1] - self.segments[boxid][0]:
            self.a1text[boxid].set_fontsize(8)
        if self.segments[boxid][4] != "Don't Know":
            facecolour = 'b'
        else:
            facecolour = 'r'
        self.listRectanglesa1[boxid].set_facecolor(facecolour)
        self.listRectanglesa2[boxid].set_facecolor(facecolour)
        self.topBoxCol = self.listRectanglesa1[boxid].get_facecolor()
        self.canvasMain.draw()

    def radioBirdsClicked(self):
        # Listener for when the user selects a radio button
        # Update the text and store the data
        for button in self.birds1 + self.birds2:
            if button.isChecked():
                if button.text() == "Other":
                    self.birdList.setEnabled(True)
                else:
                    self.birdList.setEnabled(False)
                    self.updateText(str(button.text()))

    def listBirdsClicked(self, item):
        # Listener for clicks in the listbox of birds
        if (item.text() == "Other"):
            self.tbox.setEnabled(True)
        else:
            # Save the entry
            self.updateText(str(item.text()))
            # self.segments[self.box1id][4] = str(item.text())
            # self.a1text[self.box1id].set_text(str(item.text()))
            # # The font size is a pain because the window is in pixels, so have to transform it
            # # Force a redraw to make bounding box available
            # self.canvasMain.draw()
            # # fs = a1t.get_fontsize()
            # width = self.a1text[self.box1id].get_window_extent().inverse_transformed(self.a1.transData).width
            # if width > self.segments[self.box1id][1] - self.segments[self.box1id][0]:
            #     self.a1text[self.box1id].set_fontsize(8)
            #
            # self.listRectanglesa1[self.box1id].set_facecolor('b')
            # self.listRectanglesa2[self.box1id].set_facecolor('b')
            # self.topBoxCol = self.listRectanglesa1[self.box1id].get_facecolor()
            # self.canvasMain.draw()

    def birdTextEntered(self):
        # Listener for the text entry in the bird list
        # Check text isn't already in the listbox, and add if not
        # Doesn't sort the list, but will when program is closed
        item = self.birdList.findItems(self.tbox.text(), Qt.MatchExactly)
        if item:
            pass
        else:
            self.birdList.addItem(self.tbox.text())
            self.config['ListBirdsEntries'].append(str(self.tbox.text()))
        self.updateText(str(self.tbox.text()))
        # self.segments[self.box1id][4] = str(self.tbox.text())
        # self.a1text[self.box1id].set_text(str(self.tbox.text()))
        # # The font size is a pain because the window is in pixels, so have to transform it
        # # Force a redraw to make bounding box available
        # self.canvasMain.draw()
        # # fs = self.a1text[self.box1id].get_fontsize()
        # width = self.a1text[self.box1id].get_window_extent().inverse_transformed(self.a1.transData).width
        # if width > self.segments[self.box1id][1] - self.segments[self.box1id][0]:
        #     self.a1text[self.box1id].set_fontsize(8)
        #
        # self.listRectanglesa1[self.box1id].set_facecolor('b')
        # self.listRectanglesa2[self.box1id].set_facecolor('b')
        # self.topBoxCol = self.listRectanglesa1[self.box1id].get_facecolor()
        # self.canvasMain.draw()
        self.saveConfig = True
        # self.tbox.setEnabled(False)

# ===========
# Code for things at the top of the screen (overview figure and left/right buttons)
    def moveLeft(self):
        # When the left button is pressed (next to the overview plot), move everything along
        # Note the parameter to all a 10% overlap
        self.windowStart = max(0,self.windowStart-self.windowSize*0.9)
        self.playPosition = self.windowStart
        self.updateMainWindow()

    def moveRight(self):
        # When the right button is pressed (next to the overview plot), move everything along
        # Note the parameter to all a 10% overlap
        self.windowStart = min(float(self.datalength)/self.sampleRate-self.windowSize,self.windowStart+self.windowSize*0.9)
        self.playPosition = self.windowStart
        self.updateMainWindow()

    def showSegments(self,seglen=0):
        # This plots the segments that are returned from any of the segmenters and adds them to the set of segments

        # If there are segments, show them
        for count in range(seglen,len(self.segments)):
            if self.segments[count][4] == 'None' or self.segments[count][4] == "Don't Know":
                facecolour = 'r'
            else:
                facecolour = 'b'
            a1R = self.a1.add_patch(pl.Rectangle((self.segments[count][0], np.min(self.audiodata)),
                                                 self.segments[count][1] - self.segments[count][0],
                                                 self.plotheight,
                                                 facecolor=facecolour,
                                                 alpha=0.5))
            a2R = self.a2.add_patch(pl.Rectangle((self.segments[count][0]*self.sampleRate / self.config['incr'], self.segments[count][2]),
                                                 self.segments[count][1]*self.sampleRate / self.config['incr'] - self.segments[count][
                                                     0]*self.sampleRate / self.config['incr'],
                                                 self.segments[count][3], facecolor=facecolour,
                                                 alpha=0.5))

            self.listRectanglesa1.append(a1R)
            self.listRectanglesa2.append(a2R)
            a1t = self.a1.text(self.segments[count][0], np.min(self.audiodata), self.segments[count][4])
            # The font size is a pain because the window is in pixels, so have to transform it
            # Force a redraw to make bounding box available
            self.canvasMain.draw()
            # fs = a1t.get_fontsize()
            width = a1t.get_window_extent().inverse_transformed(self.a1.transData).width
            #print width, self.segments[count][1] - self.segments[count][0]
            if width > self.segments[count][1] - self.segments[count][0]:
                a1t.set_fontsize(8)

            self.a1text.append(a1t)

        self.canvasMain.draw()

    def changeWidth(self, value):
        # This is the listener for the spinbox that decides the width of the main window.
        # It updates the top figure plots as the window width is changed.
        # Slightly annoyingly, it gets called when the value gets reset, hence the first line
        if not hasattr(self,'a1'):
            return
        self.windowSize = value
        self.a1.set_xlim(self.windowStart, self.windowStart+self.windowSize)
        self.a2.set_xlim(self.windowStart*self.sampleRate / self.config['incr'], (self.windowStart + self.windowSize)*self.sampleRate/self.config['incr'])

        # Reset the width of the segment marker
        self.linewidtha1 = float(self.config['linewidtha1'])*self.windowSize

        # Redraw the highlight in the overview figure appropriately
        self.focusRegion.remove()
        self.focusRegion = self.topfig.add_patch(pl.Rectangle((self.windowStart*self.sampleRate / self.config['incr'], 0),
                                                              self.windowSize*self.sampleRate / self.config['incr'],
                                                              self.config['window_width'] / 2,
                                                              facecolor='r',
                                                              alpha=0.5))
        self.canvasMain.draw()
        self.canvasOverview.draw()

# ===============
# Generate the various dialogs that match the buttons

    def humanClassifyDialog(self):
        # Create the dialog that shows calls to the user for verification
        # Currently assumes that there is a selected box (later, use the first!)
        self.currentSegment = 0
        self.humanClassifyDialog = HumanClassify2(self.sg[:,int(self.listRectanglesa2[self.currentSegment].get_x()):int(self.listRectanglesa2[self.currentSegment].get_x()+self.listRectanglesa2[self.currentSegment].get_width())],self.segments[self.currentSegment][4],self.cmap_grey,self.cmap_grey_r)
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
        self.sg = self.sp.spectrogram(self.audiodata,str(alg))
        self.config['colourStart'] = float(str(colourStart))
        self.config['colourEnd'] = float(str(colourEnd))
        self.defineColourmap()
        self.drawfigMain()

    def denoiseDialog(self):
        # Create the denoising dialog when the relevant button is pressed
        # And add listeners for the undo and save buttons
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

        #print "Denoising"
        #self.audiodata = self.sp.waveletDenoise()
        #print "Done"
        self.sg = self.sp.spectrogram(self.audiodata,self.sampleRate)
        self.drawfigMain()

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
                    self.sg = self.sp.spectrogram(self.audiodata,self.sampleRate)
                    if hasattr(self,'seg'):
                        self.seg.setNewData(self.audiodata,self.sg,self.sampleRate)
                    self.drawfigMain()

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
        [alg, ampThr, medThr,depth,thrType,thr,wavelet,bandchoice,start,end] = self.segmentDialog.getValues()
        if not hasattr(self,'seg'):
            self.seg = Segment.Segment(self.audiodata,self.sg,self.sp,self.sampleRate)
        if str(alg) == "Amplitude":
            newSegments = self.seg.segmentByAmplitude(float(str(ampThr)))
            if hasattr(self, 'line'):
                if self.line is not None:
                    self.line.remove()
            self.line = self.a1.add_patch(pl.Rectangle((0,float(str(ampThr))),len(self.audiodata),0,facecolor='r'))
        elif str(alg) == "Median Clipping":
            newSegments = self.seg.medianClip(float(str(medThr)))
            print newSegments
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
                newSegments = self.seg.segmentByWavelet(thrType,float(str(thr)), int(str(depth)), wavelet,sampleRate,bandchoice,start,end,learning,)
        for seg in newSegments:
            self.segments.append([seg[0],seg[1],0.0, self.sampleRate / 2., 'None'])

        self.showSegments(seglen)

    def findMatches(self):
        # Calls the cross-correlation function to find matches like the currently highlighted box
        # TODO: Other methods apart from c-c?
        # TODO: Should give them the same label as the currently highlighted box
        # So there needs to be a currently highlighted box
        #print self.box1id
        if not hasattr(self,'seg'):
            self.seg = Segment.Segment(self.audiodata,self.sg,self.sp,self.sampleRate)

        if self.box1id is None or self.box1id == -1:
            return
        else:
            #[alg, thr] = self.matchDialog.getValues()
            # Only want to draw new segments, so find out how many there are now
            seglen = len(self.segments)
            # Get the segment
            #print self.listRectanglesa2[self.box1id].get_x()
            segment = self.sg[:,self.listRectanglesa2[self.box1id].get_x():self.listRectanglesa2[self.box1id].get_x()+self.listRectanglesa2[self.box1id].get_width()]
            len_seg = float(self.listRectanglesa2[self.box1id].get_width()) *self.config['incr'] / self.sampleRate
            indices = self.seg.findCCMatches(segment,self.sg,self.config['corrThr'])
            # indices are in spectrogram pixels, need to turn into times
            for i in indices:
                # Miss out the one selected: note the hack parameter
                if np.abs(i-self.listRectanglesa2[self.box1id].get_x()) > 5:
                    time = float(i)*self.config['incr'] / self.sampleRate
                    self.segments.append([time, time+len_seg, 0.0, self.sampleRate / 2., 'None'])
            self.showSegments(seglen)

    def recognise(self):
        # This will eventually call methods to do automatic recognition
        # Actually, will produce a dialog to ask which species, etc.
        # TODO
        pass

# ===============
# Code for the zoom figure
    def drawfigZoom(self):
        # This produces the plots for the zoomed-in window
        # Has to process the start and end times and process them, which is a pain
        self.figZoom.subplots_adjust(0.0, 0.0, 1.0,1.0)
        start = int(np.round(self.zoomstart*self.sampleRate))
        end = int(np.round(self.zoomend*self.sampleRate))
        self.listRectanglesb1 = []
        self.listRectanglesb2 = []

        # Make the start and end bands be big and draggable
        # Subtract off the thickness of the bars
        # When they are moved, update the plotting
        padding = int(np.round(self.config['padding']*(end-start)))
        # Draw the two charts
        self.a3 = self.figZoom.add_subplot(211)
        self.a3.clear()
        xstart = max(start - padding, 0)
        if xstart != 0:
            start = padding
        xend = min(end + padding, len(self.audiodata))
        if xend != end+padding:
            end = xend-end

        self.a3.plot(self.audiodata[xstart:xend])
        self.a3.axis('off')

        self.a4 = self.figZoom.add_subplot(212)
        self.a4.clear()
        newsg = self.sp.spectrogram(self.audiodata[xstart:xend],self.sampleRate)
        self.a4.imshow(10.0*np.log10(newsg), cmap=self.cmap_grey, aspect='auto')
        self.a4.axis('off')

        self.figZoomwidth = xend - xstart
        self.bar_thickness = self.config['linewidtha2'] * self.figZoomwidth
        self.bar_thickness = max(self.bar_thickness, self.config['minbar_thickness'])

        self.End1 = self.a3.add_patch(pl.Rectangle((start - self.bar_thickness, np.min(self.audiodata)), self.bar_thickness,
                                                   self.plotheight,
                                                   facecolor='g',
                                                   edgecolor='None', alpha=1.0, picker=10))
        self.End2 = self.a4.add_patch(
            pl.Rectangle(((start - self.bar_thickness) / self.config['incr'], 0), self.bar_thickness / self.config['incr'],
                         self.config['window_width'] / 2, facecolor='g',
                         edgecolor='None', alpha=1.0, picker=10))
        self.End3 = self.a3.add_patch(pl.Rectangle((end-xstart, np.min(self.audiodata)), self.bar_thickness,
                                                   self.plotheight, facecolor='r',
                         edgecolor='None', alpha=1.0, picker=10))
        self.End4 = self.a4.add_patch(pl.Rectangle(((end-xstart) / self.config['incr'], 0),
                                                   self.bar_thickness / self.config['incr'],
                                                   self.config['window_width'] / 2, facecolor='r',
                                                   edgecolor='None', alpha=1.0, picker=10))
        # self.End3 = self.a3.add_patch(
        #     pl.Rectangle((xend - self.zoomstart*self.sampleRate + self.bar_thickness, np.min(self.audiodata)), self.bar_thickness,
        #                  self.plotheight, facecolor='r',
        #                  edgecolor='None', alpha=1.0, picker=10))
        # self.End4 = self.a4.add_patch(pl.Rectangle(((xend - self.zoomstart*self.sampleRate + self.bar_thickness) / self.config['incr'], 0),
        #                                            self.bar_thickness / self.config['incr'],
        #                                            self.config['window_width'] / 2, facecolor='r',
        #                                            edgecolor='None', alpha=1.0, picker=10))
        self.listRectanglesb1.append(self.End1)
        self.listRectanglesb2.append(self.End2)
        self.listRectanglesb1.append(self.End3)
        self.listRectanglesb2.append(self.End4)
        # Keep track of where you are in figMain
        self.offset_figMain = self.zoomstart*self.sampleRate - start

        self.addButton.setEnabled(True)
        # Start the listener for picking the bars
        self.picker = self.canvasZoom.mpl_connect('pick_event', self.segmentfigZoomPicked)
        self.canvasZoom.draw()

    def segmentfigZoomPicked(self,event):
        # Listener for when the user clicks on one of the segment ends in figure 2
        # Turn off the listeners
        if self.buttonID is not None:
            self.canvasZoom.mpl_disconnect(self.buttonID)
            self.canvasZoom.mpl_disconnect(self.keypress)
            self.buttonID = None
            self.keypress = None
        # Put the colours back for any other segment
        for box in self.listRectanglesb1:
            if box.get_facecolor() == matplotlib.colors.colorConverter.to_rgba('k'):
                # It's black, put it back
                box.set_facecolor(self.Boxcol)
        for box in self.listRectanglesb2:
            if box.get_facecolor() == matplotlib.colors.colorConverter.to_rgba('k'):
                box.set_facecolor(self.Boxcol)

        self.Box1 = event.artist
        self.box2id = -1
        # Work out which axes you are in
        for count in range(len(self.listRectanglesb1)):
            if self.listRectanglesb1[count].xy[0] == self.Box1.xy[0]:
                self.box2id = count
                self.Box2 = self.listRectanglesb2[self.box2id]
                self.axes = 3
        if self.box2id == -1:
            # Must be in spectrogram axes
            for count in range(len(self.listRectanglesb2)):
                if self.listRectanglesb2[count].xy[0] == self.Box1.xy[0]:
                    self.box2id = count
                    self.Box2 = self.Box1
                    self.Box1 = self.listRectanglesb1[self.box2id]
                    self.axes = 4

        # Set the bar to black and restart the listeners
        self.Boxcol = self.Box1.get_facecolor()
        self.Box1.set_facecolor('black')
        self.Box2.set_facecolor('black')
        self.buttonID = self.canvasZoom.mpl_connect('button_press_event', self.moveSegmentEnd)
        self.keypress = self.canvasZoom.mpl_connect('key_press_event', self.keypressEnd)
        self.canvasZoom.draw()

    def moveSegmentEnd(self,event):
        # After selecting a segment end in Figure 2, this is what happens when the user clicks again
        if self.Box1 is None: return
        current_x1 = self.Box1.get_x()
        current_x2 = self.Box2.get_x()
        current_width1 = self.Box1.get_width()
        current_width2 = self.Box2.get_width()
        if (current_x1<=event.xdata and (current_x1+current_width1)>=event.xdata) or (current_x2<=event.xdata and (current_x2+current_width2)>=event.xdata):
            # Clicked on the same box, don't move it
            pass
        else:
            # Both of these work out where to put the updated box and update the data
            if self.axes==3:
                self.Box1.set_x(event.xdata)
                self.Box2.set_x(event.xdata/self.config['incr'])
                if self.Boxcol == matplotlib.colors.colorConverter.to_rgba('g'):
                    # Green -> left
                    self.listRectanglesa1[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x() + (event.xdata - current_x1)/self.sampleRate)
                    self.listRectanglesa1[self.box1id].set_width(self.listRectanglesa1[self.box1id].get_width() - (event.xdata - current_x1)/self.sampleRate)
                    self.listRectanglesa2[self.box1id].set_x(self.listRectanglesa2[self.box1id].get_x() + (event.xdata/self.config['incr'] - current_x2))
                    self.listRectanglesa2[self.box1id].set_width(self.listRectanglesa2[self.box1id].get_width() - (event.xdata/self.config['incr'] - current_x2))
                    self.a1text[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x())
                else:
                    self.listRectanglesa1[self.box1id].set_width(self.listRectanglesa1[self.box1id].get_width() + (event.xdata - current_x1)/self.sampleRate)
                    self.listRectanglesa2[self.box1id].set_width(self.listRectanglesa2[self.box1id].get_width() + (event.xdata/self.config['incr'] - current_x2))
            else:
                self.Box2.set_x(event.xdata)
                self.Box1.set_x(event.xdata*self.config['incr'])
                if self.Boxcol == matplotlib.colors.colorConverter.to_rgba('g'):
                    # Green -> left
                    self.listRectanglesa1[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x() + (event.xdata*self.config['incr'] - current_x1)/self.sampleRate)
                    self.listRectanglesa1[self.box1id].set_width(self.listRectanglesa1[self.box1id].get_width() - (event.xdata*self.config['incr'] - current_x1)/self.sampleRate)
                    self.listRectanglesa2[self.box1id].set_x(self.listRectanglesa2[self.box1id].get_x() + (event.xdata - current_x2))
                    self.listRectanglesa2[self.box1id].set_width(self.listRectanglesa2[self.box1id].get_width() - (event.xdata - current_x2))
                    self.a1text[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x())
                else:
                    self.listRectanglesa2[self.box1id].set_width(self.listRectanglesa2[self.box1id].get_width() + (event.xdata - current_x2))
                    self.listRectanglesa1[self.box1id].set_width(self.listRectanglesa1[self.box1id].get_width() + (event.xdata*self.config['incr'] - current_x1)/self.sampleRate)
        self.segments[self.box1id][0] = min(self.listRectanglesa1[self.box1id].get_x(),self.listRectanglesa1[self.box1id].get_x() + self.listRectanglesa1[self.box1id].get_width())
        self.segments[self.box1id][1] = max(self.listRectanglesa1[self.box1id].get_x(),self.listRectanglesa1[self.box1id].get_x() + self.listRectanglesa1[self.box1id].get_width())
        # Put the colours back, disconnect the listeners and ready everything for another click
        self.Box1.set_facecolor(self.Boxcol)
        self.Box2.set_facecolor(self.Boxcol)
        self.Box1 = None
        self.Box2 = None
        self.canvasZoom.mpl_disconnect(self.buttonID)
        self.canvasZoom.mpl_disconnect(self.keypress)
        self.buttonID = None
        self.keypress = None
        self.canvasZoom.draw()
        self.canvasMain.draw()

    def keypressEnd(self,event):
        # If the user has selected a segment in figZoom and then presses a key, this deals with it
        if self.Box1 is None: return
        move_amount = self.config['move_amount']*self.figZoomwidth
        # TODO: how to remove the listener callback?
        if event.key == 'left':
            self.Box1.set_x(self.Box1.get_x() - move_amount)
            self.Box2.set_x(self.Box2.get_x() - move_amount / self.config['incr'])
            if self.Boxcol == matplotlib.colors.colorConverter.to_rgba('g'):
                self.listRectanglesa1[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x() - move_amount/self.sampleRate)
                self.listRectanglesa1[self.box1id].set_width(self.listRectanglesa1[self.box1id].get_width() + move_amount/self.sampleRate)
                self.listRectanglesa2[self.box1id].set_x(self.listRectanglesa2[self.box1id].get_x() - move_amount/self.config['incr'])
                self.listRectanglesa2[self.box1id].set_width(self.listRectanglesa2[self.box1id].get_width() + move_amount/self.config['incr'])
                self.a1text[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x())
            else:
                self.listRectanglesa1[self.box1id].set_width(
                    self.listRectanglesa1[self.box1id].get_width() - move_amount/self.sampleRate)
                self.listRectanglesa2[self.box1id].set_width(
                    self.listRectanglesa2[self.box1id].get_width() - move_amount / self.config['incr'])
            self.segments[self.box1id][0] = min(self.listRectanglesa1[self.box1id].get_x(),self.listRectanglesa1[self.box1id].get_x() + self.listRectanglesa1[self.box1id].get_width())
            self.segments[self.box1id][1] = max(self.listRectanglesa1[self.box1id].get_x(),self.listRectanglesa1[self.box1id].get_x() + self.listRectanglesa1[self.box1id].get_width())

        elif event.key == 'right':
            self.Box1.set_x(self.Box1.get_x() + move_amount)
            self.Box2.set_x(self.Box2.get_x() + move_amount / self.config['incr'])
            if self.Boxcol == matplotlib.colors.colorConverter.to_rgba('g'):
                self.listRectanglesa1[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x() + move_amount/self.sampleRate)
                self.listRectanglesa1[self.box1id].set_width(
                    self.listRectanglesa1[self.box1id].get_width() - move_amount/self.sampleRate)
                self.listRectanglesa2[self.box1id].set_x(
                    self.listRectanglesa2[self.box1id].get_x() + move_amount/self.config['incr'])
                self.listRectanglesa2[self.box1id].set_width(
                    self.listRectanglesa2[self.box1id].get_width() - move_amount/self.config['incr'])
                self.a1text[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x())
            else:
                self.listRectanglesa1[self.box1id].set_width(
                    self.listRectanglesa1[self.box1id].get_width() + move_amount/self.sampleRate)
                self.listRectanglesa2[self.box1id].set_width(
                    self.listRectanglesa2[self.box1id].get_width() + move_amount/self.config['incr'])
            self.segments[self.box1id][0] = min(self.listRectanglesa1[self.box1id].get_x(),self.listRectanglesa1[self.box1id].get_x() + self.listRectanglesa1[self.box1id].get_width())
            self.segments[self.box1id][1] = max(self.listRectanglesa1[self.box1id].get_x(),self.listRectanglesa1[self.box1id].get_x() + self.listRectanglesa1[self.box1id].get_width())

        elif event.key == 'backspace':
            self.listRectanglesb1[0].remove()
            self.listRectanglesb2[0].remove()
            self.listRectanglesb1[1].remove()
            self.listRectanglesb2[1].remove()
            self.listRectanglesa1[self.box1id].remove()
            self.listRectanglesa2[self.box1id].remove()
            self.listRectanglesa1.remove(self.listRectanglesa1[self.box1id])
            self.listRectanglesa2.remove(self.listRectanglesa2[self.box1id])
            self.a1text[self.box1id].remove()
            self.a1text.remove(self.a1text[self.box1id])
            self.segments.remove(self.segments[self.box1id])
            self.box1id = -1

        # TODO: When to stop and turn them back to right colour? -> mouse moves out of window?
        self.canvasZoom.draw()
        self.canvasMain.draw()

    def addSegmentClick(self):
        # When the user clicks the button to add a segment in figZoom, this deals with it
        # Turn the labelling off to avoid confusion
        # for i in xrange(len(self.birds1)):
        #     self.birds1[i].setEnabled(False)
        # for i in xrange(len(self.birds2)):
        #     self.birds2[i].setEnabled(False)
        # self.birdList.setEnabled(False)
        # self.tbox.setEnabled(False)

        # Should check for and stop the listener for the moving of segment ends
        if hasattr(self,'keypress'):
            if self.keypress is not None:
                self.Box1.set_facecolor(self.Boxcol)
                self.Box2.set_facecolor(self.Boxcol)
                self.Box1 = None
                self.Box2 = None
                self.canvasZoom.mpl_disconnect(self.buttonID)
                self.canvasZoom.mpl_disconnect(self.keypress)
                self.keypress = None
                self.buttonID = None
                self.canvasZoom.draw()

        # Make a new listener to check for the button clicks
        self.buttonAdd = self.canvasZoom.mpl_connect('button_press_event', self.addSegmentfigZoom)

    def addSegmentfigZoom(self,event):
        # This deals with the mouse events after choosing to add a new segment
        # First click should be the END of the first segment, second should be START of second
        # This should work logically providing person moves left to right

        # Work out which axes you are in, and update all of them
        a3 = str(self.a3)
        a3ind = float(a3[a3.index(',') + 1:a3.index(';')])

        if event.inaxes is not None:
            s = event.xdata
            a = str(event.inaxes)
            aind = float(a[a.index(',') + 1:a.index(';')])
            if aind == a3ind:
                s2 = np.round(s / self.config['incr'])
            else:
                s2 = s
                s = np.round(s * self.config['incr'])

            if not self.start_stop2:
                # This is the end of the first segment, draw a red line
                # TODO: At the moment you can't pick these, 'cos it's a pain. Does it matter?
                markstarta3 = self.a3.add_patch(pl.Rectangle((s, np.min(self.audiodata)), self.bar_thickness,
                                                             self.plotheight, facecolor='r', edgecolor='None',
                                                                  alpha=0.8)) #,picker=10))
                markstarta4 = self.a4.add_patch(pl.Rectangle((s2, 0), self.bar_thickness/self.config['incr'],
                                                                  self.config['window_width']/2, facecolor='r', edgecolor='None',
                                                                  alpha=0.8)) #,picker=10))
                self.markstarta1 = self.a1.add_patch(pl.Rectangle(((s + self.offset_figMain)/self.sampleRate, np.min(self.audiodata)), self.linewidtha1 ,
                                                                  self.plotheight, facecolor='r', edgecolor='None',
                                                                  alpha=0.8))
                self.markstarta2 = self.a2.add_patch(pl.Rectangle((s2 + self.offset_figMain/self.config['incr'], 0), self.linewidtha1 /self.config['incr']*self.sampleRate, self.config['window_width'], facecolor='r', edgecolor='None',alpha=0.8))
                self.listRectanglesb1.append(markstarta3)
                self.listRectanglesb2.append(markstarta4)
                self.end_a2 = s
                self.end_s2 = s2
            else:
                # This is the start of the second segment, draw green lines on figZoom, then redraw the boxes on the top figure and save
                b1e = self.a3.add_patch(pl.Rectangle((s, np.min(self.audiodata)), self.bar_thickness, self.plotheight, facecolor='g', edgecolor='None',alpha=0.8,picker=10))
                b2e = self.a4.add_patch(pl.Rectangle((s2, 0), self.bar_thickness/self.config['incr'], self.config['window_width']/2, facecolor='g', edgecolor='None',alpha=0.8,picker=10))
                self.markstarta1.remove()
                self.markstarta2.remove()
                # Delete the boxes
                self.listRectanglesa1[self.box1id].remove()
                self.listRectanglesa2[self.box1id].remove()
                # Delete the references to the boxes
                self.listRectanglesa1.remove(self.listRectanglesa1[self.box1id])
                self.listRectanglesa2.remove(self.listRectanglesa2[self.box1id])
                self.listRectanglesb1.append(b1e)
                self.listRectanglesb2.append(b2e)

                # Update the segments; remove first to keep the indexing consistent
                old = self.segments[self.box1id]
                self.segments.remove(self.segments[self.box1id])
                self.segments.append([old[0],(self.end_a2 + self.offset_figMain)/self.sampleRate,old[2],old[3],old[4]])
                self.segments.append([(s + self.offset_figMain)/self.sampleRate, old[1], old[2],old[3],old[4]])
                # Delete the old rectangles and text
                self.a1text[self.box1id].remove()
                self.a1text.remove(self.a1text[self.box1id])

                # Add the two new ones
                a1R = self.a1.add_patch(pl.Rectangle((self.segments[-2][0], np.min(self.audiodata)), self.segments[-2][1] - self.segments[-2][0],
                                                     self.plotheight,
                                                     facecolor='r', alpha=0.5))
                a2R = self.a2.add_patch(pl.Rectangle((self.segments[-2][0]*self.sampleRate/self.config['incr'], self.segments[-2][2]), (self.segments[-2][1] - self.segments[-2][0])*self.sampleRate/self.config['incr'],
                                                     self.segments[-2][3],
                                                     facecolor='r', alpha=0.5))
                self.listRectanglesa1.append(a1R)
                self.listRectanglesa2.append(a2R)
                a1t = self.a1.text(self.segments[-2][0], np.min(self.audiodata), self.segments[-2][4])
                # The font size is a pain because the window is in pixels, so have to transform it
                # Force a redraw to make bounding box available
                self.canvasMain.draw()
                # fs = a1t.get_fontsize()
                width = a1t.get_window_extent().inverse_transformed(self.a1.transData).width
                #print width, self.segments[-2][1] - self.segments[-2][0]
                if width > self.segments[-2][1] - self.segments[-2][0]:
                    a1t.set_fontsize(8)
                self.a1text.append(a1t)

                a3R = self.a1.add_patch(pl.Rectangle((self.segments[-1][0], np.min(self.audiodata)), self.segments[-1][1] - self.segments[-1][0],
                                                     self.plotheight,
                                                     facecolor='r', alpha=0.5))
                a4R = self.a2.add_patch(pl.Rectangle((self.segments[-1][0]*self.sampleRate/self.config['incr'], self.segments[-1][2]), (self.segments[-1][1]-self.segments[-1][0])*self.sampleRate/self.config['incr'],
                                                     self.segments[-1][3],
                                                 facecolor='r', alpha=0.5))
                self.listRectanglesa1.append(a3R)
                self.listRectanglesa2.append(a4R)
                a2t = self.a1.text(self.segments[-1][0], np.min(self.audiodata), self.segments[-1][4])
                self.a1text.append(a2t)
                # The font size is a pain because the window is in pixels, so have to transform it
                # Force a redraw to make bounding box available
                self.canvasMain.draw()
                # fs = a1t.get_fontsize()
                width = a1t.get_window_extent().inverse_transformed(self.a1.transData).width
                #print width, self.segments[-1][1] - self.segments[-1][0]
                if width > self.segments[-1][1] - self.segments[-1][0]:
                    a1t.set_fontsize(8)

                # Stop the listener
                self.canvasZoom.mpl_disconnect(self.buttonAdd)

                # Make the addSegments button unselectable
                self.addButton.setEnabled(False)
                self.box1id = None
        self.canvasZoom.draw()
        self.canvas.draw()
        self.start_stop2 = not self.start_stop2
        # For now at least, stop the canvasZoom listener
        self.canvasZoom.mpl_disconnect(self.picker)

# ===============
# Code for playing sounds
    # What follows is the sound playing things. It's all a mess.
    # There were attempts to make it a separate thread so that I could make a bar move, etc. None of it is nice.
    def runthread(self,start,end,sampleRate):
        sd.play(self.audiodata[start:end],sampleRate)

    def runthread2(self,start_time,end_time):
        if self.playbar1 is not None:
            self.playbar1.remove() 
            self.playbar2.remove() 
            self.playbar1 = None
        self.playbar1 = self.a1.add_patch(
             pl.Rectangle((self.playPosition, np.min(self.audiodata)), self.linewidtha1, self.plotheight, facecolor='k',
                          edgecolor='None', alpha=0.8))
        self.playbar2 = self.a2.add_patch(
             pl.Rectangle((self.playPosition*self.sampleRate/self.config['incr'], 0), self.linewidtha1/self.config['incr']*self.sampleRate, self.config['window_width'], facecolor='k', edgecolor='None', alpha=0.8))
        
        dummy = self.a1.add_patch(
             pl.Rectangle((self.playPosition, np.min(self.audiodata)), self.linewidtha1, self.plotheight, facecolor='k',
                          edgecolor='None', alpha=0.8))
        self.canvasMain.draw()
        current_time = start_time
        step = 0
        while (current_time - start_time).total_seconds() < end_time and not self.stopRequest:
             now = datetime.datetime.now()
             timedelta = (now - current_time).total_seconds()
             step += timedelta
             self.playbar1.set_x(self.playbar1.get_x() + timedelta)
             self.playbar2.set_x(self.playbar2.get_x() + timedelta/self.config['incr']*self.sampleRate)
             # For reasons unknown, the next one is needed to stop the plot vanishing
             dummy.remove()
             dummy = self.a1.add_patch( pl.Rectangle((step, np.min(self.audiodata)), self.linewidtha1, self.plotheight, facecolor='w', edgecolor='None', alpha=0.0))
             self.canvasMain.draw()
             current_time = now

    def resetSegment(self):
        self.playPosition = self.windowStart
        if self.playbar1 is not None:
            self.playbar1.remove() 
            self.playbar2.remove() 
            self.playbar1 = None
            self.canvasMain.draw()

    # These functions are the phonon playing code
    # Note that if want to play e.g. denoised one, will have to save it and then reload it
    def playSegment(self):
        if self.media_obj.state() == phonon.Phonon.PlayingState:
            self.media_obj.pause()
            self.playButton1.setText("Play")
        elif self.media_obj.state() == phonon.Phonon.PausedState or self.media_obj.state() == phonon.Phonon.StoppedState:
            self.media_obj.play()
            self.playButton1.setText("Pause")

    def playFinished(self):
        self.playButton1.setText("Play")

    def sliderMoved(self):
        # When the slider is moved, change the position of playback
        print self.playSlider.value()
        self.media_obj.seek(self.playSlider.value())

    def movePlaySlider(self, time):
        #print time
        #print self.media_obj.state()
        if not self.playSlider.isSliderDown():
            self.playSlider.setValue(time)

    def setSliderLimits(self, start,end):
        self.playSlider.setRange(start, end)
        self.playSlider.setValue(start)
        self.media_obj.seek(start)

    def playSegment_old(self):
        # This is the listener for the play button. A very simple wave file player
        # Move a marker through in real time?!

        if not self.isPlaying:
            self.isPlaying = True
            print int(self.playPosition*self.sampleRate),int(self.playPosition*self.sampleRate+self.windowSize*self.sampleRate)
            #sd.play(self.audiodata[int(self.playPosition*self.sampleRate):int(self.playPosition*self.sampleRate+self.windowSize*self.sampleRate)], self.sampleRate)
            #t = Thread(target=self.runthread, args=(int(self.playPosition*self.sampleRate),int(self.playPosition*self.sampleRate+self.windowSize*self.sampleRate),self.sampleRate))
            #t.start()

            end_time = self.windowStart + self.windowSize
            start_time = datetime.datetime.now()
            print "Start: ",start_time, end_time
            self.stopRequest = False
            #t2 = Thread(target=self.runthread2, args=(start_time,end_time))
            #t2.start()
        else:
            #print self.playbar1.get_x()
            #self.playPosition = self.playbar1.get_x()
            self.stopRequest = True
            sd.stop()
            self.isPlaying = False

    def play(self):
        # This is the listener for the play button. 
        #playThread().start()
        t = Thread(target=run, args=(self.zoomstart,self.zoomend))
        t.start()

# ============
# Code for the buttons
    def deleteSegment(self):
        # Listener for delete segment button, or backspace key
        # Deletes segment if one is selected, otherwise does nothing
        if self.box1id>-1:
            self.listRectanglesa1[self.box1id].remove()
            self.listRectanglesa2[self.box1id].remove()
            self.listRectanglesa1.remove(self.listRectanglesa1[self.box1id])
            self.listRectanglesa2.remove(self.listRectanglesa2[self.box1id])
            for r in self.listRectanglesb1:
                r.remove()
            for r in self.listRectanglesb2:
                r.remove()
            self.listRectanglesb1 = []
            self.listRectanglesb2 = []
            self.a1text[self.box1id].remove()
            self.a1text.remove(self.a1text[self.box1id])
            self.segments.remove(self.segments[self.box1id])
            self.box1id = -1
            self.canvasMain.draw()
            # Make canvasZoom blank
            self.a3.clear()
            self.a3.axis('off')
            self.a4.clear()
            self.canvasZoom.draw()

    def deleteAll(self):
        # Listener for delete all button
        self.segments=[]
        for r in self.a1text:
            r.remove()
        for r in self.listRectanglesa1:
            r.remove()
        for r in self.listRectanglesa2:
            r.remove()
        for r in self.listRectanglesb1:
            r.remove()
        for r in self.listRectanglesb2:
            r.remove()
        self.listRectanglesa1 = []
        self.listRectanglesa2 = []
        self.listRectanglesb1 = []
        self.listRectanglesb2 = []
        self.a1text = []
        self.box1id = -1

        self.canvasMain.draw()
        self.canvasZoom.draw()
        self.canvasOverview.draw()

    def saveSegments(self):
        # This saves the segmentation data as a json file
        if len(self.segments)>0:
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

    def defineColourmap(self):
        # This makes the spectrograms look better. It defines a colourmap that hides a lot of the black
        # We want a colourmap that goes from white to black in greys, but has limited contrast
        # First is sorted by keeping all 3 colours the same, second by squeezing the range
        cdict = {
            'blue': ((0, 1, 1), (self.config['colourStart'], 1, 1), (self.config['colourEnd'], 0, 0), (1, 0, 0)),
            'green': ((0, 1, 1), (self.config['colourStart'], 1, 1), (self.config['colourEnd'], 0, 0), (1, 0, 0)),
            'red': ((0, 1, 1), (self.config['colourStart'], 1, 1), (self.config['colourEnd'], 0, 0), (1, 0, 0))
        }
        self.cmap_grey = matplotlib.colors.LinearSegmentedColormap('cmap_grey', cdict, 256)

        # Reverse each colour
        cdict = {
            'blue': ((0, 0, 0), (self.config['colourStart'], 0, 0), (self.config['colourEnd'], 1, 1), (1, 1, 1)),
            'green': ((0, 0, 0), (self.config['colourStart'], 0, 0), (self.config['colourEnd'], 1, 1), (1, 1, 1)),
            'red': ((0, 0, 0), (self.config['colourStart'], 0, 0), (self.config['colourEnd'], 1, 1), (1, 1, 1))
        }
        self.cmap_grey_r = matplotlib.colors.LinearSegmentedColormap('cmap_grey_r', cdict, 256)

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
        self.algs.addItems(["Amplitude","Energy Curve","Median Clipping","Wavelets"])
        self.algs.currentIndexChanged[QString].connect(self.changeBoxes)
        self.prevAlg = "Amplitude"
        self.activate = QPushButton("Segment")
        self.save = QPushButton("Save segments")

        # Define the whole set of possible options for the dialog box here, just to have them together.
        # Then hide and show them as required as the algorithm chosen changes.

        # Spin box for amplitude threshold
        self.ampThr = QDoubleSpinBox()
        self.ampThr.setRange(0.001,maxv+0.001)
        self.ampThr.setSingleStep(0.002)
        self.ampThr.setDecimals(4)
        self.ampThr.setValue(maxv+0.001)

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
        Box.addWidget(self.medThr)
        for i in range(len(self.ecthrtype)):
            Box.addWidget(self.ecthrtype[i])
            self.ecthrtype[i].hide()
        Box.addWidget(self.ecThr)
        self.medThr.hide()
        #self.ecthrtype.hide()
        self.ecThr.hide()

        Box.addWidget(self.activate)
        Box.addWidget(self.save)

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
        return [self.algs.currentText(),self.ampThr.text(),self.medThr.text(),self.depth.text(),self.thrtype[0].isChecked(),self.thr.text(),self.wavelet.currentText(),self.bandchoice.isChecked(),self.start.text(),self.end.text()]

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

    def __init__(self, seg, label, cmap_grey, cmap_grey_r, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Check Classifications')
        self.frame = QWidget()
        self.cmap_grey = cmap_grey

        # Set up the plot windows, then the right and wrong buttons, and a close button
        self.plot = Figure()
        self.plot.set_size_inches(10.0, 2.0, forward=True)
        self.canvasPlot = FigureCanvas(self.plot)
        self.canvasPlot.setParent(self.frame)

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
        vboxFull.addWidget(self.canvasPlot)
        vboxFull.addWidget(self.species)
        vboxFull.addLayout(hboxButtons)

        self.setLayout(vboxFull)
        self.makefig(seg)

    def makefig(self,seg):
        self.a = self.plot.add_subplot(111)
        self.a.imshow(10.0*np.log10(seg), cmap=self.cmap_grey, aspect='auto')
        self.a.axis('off')

    def getValues(self):
        # TODO
        return True

    def setImage(self,seg,label):
        self.a.imshow(10.0 * np.log10(seg), cmap=self.cmap_grey, aspect='auto')
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
        self.plot = Figure()
        self.plot.set_size_inches(10.0, 2.0, forward=True)
        self.canvasPlot = FigureCanvas(self.plot)
        self.canvasPlot.setParent(self.frame)

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
        vboxFull.addWidget(self.canvasPlot)
        vboxFull.addLayout(hbox)
        vboxFull.addWidget(self.close)

        self.setLayout(vboxFull)
        self.makefig(seg)

    def makefig(self,seg):
        self.a = self.plot.add_subplot(111)
        self.a.imshow(10.0*np.log10(seg), cmap=self.cmap_grey, aspect='auto')
        self.a.axis('off')

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
        self.a.imshow(10.0 * np.log10(seg), cmap=self.cmap_grey, aspect='auto')
        self.species.setText(label)
        self.canvasPlot.draw()

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

    def __init__(self, seg, label, cmap_grey, cmap_grey_r, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Check Classifications')
        self.frame = QWidget()
        self.cmap_grey = cmap_grey
        self.cmap_grey_r = cmap_grey_r

        # TODO: Add a label with instructions
        # TODO: Add button to finish and/or get more
        # TODO: Add a label with the species

        # TODO: Decide on these sizes
        width = 3
        height = 4
        grid = QGridLayout(self.frame)
        self.setLayout(grid)

        # TODO: Next line needs to go!
        segs = [1,2,3,4,5,6,7]
        images = []
        # TODO: Turn this into different images
        if len(segs) < width*height:
            for i in range(len(segs)):
                images.append(self.setImage(seg))
            for i in range(len(segs),width*height):
                images.append([None,None])
        else:
            for i in range(width*height):
                images.append(self.setImage(seg))

        positions = [(i, j) for i in range(height) for j in range(width)]

        for position, image in zip(positions, images):
            if image[0] is not None:
                button = PicButton(position[0]*width + position[1],QPixmap(QImage(image[0].buffer_rgba(), image[0].size().width(), image[0].size().height(), QImage.Format_ARGB32)),
                                QPixmap(QImage(image[1].buffer_rgba(), image[1].size().width(), image[1].size().height(), QImage.Format_ARGB32)))
                grid.addWidget(button, *position)


        self.setLayout(grid)

    def setImage(self,seg):
        fig = Figure((2.5, 1.2))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(10.*np.log10(seg), cmap = self.cmap_grey, aspect='auto')
        ax.set_axis_off()
        canvas.draw()

        canvas2 = FigureCanvas(fig)
        ax.imshow(10.*np.log10(seg), cmap = self.cmap_grey_r, aspect='auto')
        canvas2.draw()
        return [canvas, canvas2]

    def activate(self):
        return True

# Start the application
app = QApplication(sys.argv)
form = Interface(configfile='AviaNZconfig.txt')
form.show()
app.exec_()
