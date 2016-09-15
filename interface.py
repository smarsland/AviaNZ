# Interface.py
#
# This is the main class for the AviaNZ interface
# It's fairly simple, but seems to work OK
# Version 0.6 27/8/16
# Author: Stephen Marsland

import sys, os, glob, json, datetime
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from scipy.io import wavfile
import numpy as np
import pylab as pl

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import sounddevice as sd

import SignalProc
import Segment
#import Features
#import Learning
# ==============
# TODO

# Finish choosing parameters for median clipping segmentation
# Finish the raven features
# Sort out the wavelet segmentation

# Some bug in denoising?
# How to set bandpass params? -> is there a useful plot to help?
# More features, more segmentations
# Other parameters for dialogs?
# Needs decent testing
# Some learning algs

# Pyqtgraph?
# Size of zoom window?

# Pitch tracking, fundamental frequency

# Minor:
# Turn stereo sound into mono using librosa, consider always resampling to 22050Hz?
# Font size to match segment size -> currently make it smaller, could also move it up or down as appropriate

# Things to consider:
    # Second spectrogram (currently use right button for interleaving)? My current choice is no as it takes up space
    # Put the labelling (and loading) in a dialog to free up space? -> bigger plots
    # Useful to have a go to start button?
    # Use to show a moving point through the file as it plays? Sound has been a pain in the arse to date!

# Look at raven and praat and luscinia -> what else is actually useful? Other annotations on graphs?

# Need to have another looking at wavelet denoising and work out what is happening, e.g., the tril1 call??

# Given files > 5 mins, split them into 5 mins versions anyway (code is there, make it part of workflow)
# Don't really want to load the whole thing, just 5 mins, and then move through with arrows -> how?
# This is sometimes called paging, I think.

# As well as this version with pictures, will need to be able to call various bits to work offline
# denoising, segmentation, etc. that will then just show some pictures

# What can be speeded up? Spectrogram in C? Denoising!
# Look into cython

# Get suggestions from the others

# Things to remember
# When adding new classes, make sure to pass new data to them in undoing and loading

# This version has the selection of birds using a context menu and then has removed the radio buttons
# Code is still there, though, just commented out. Add as an option?

# ===============

class Interface(QMainWindow):

    def __init__(self,root=None,configfile=None):
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

        self.firstFile = 'male1.wav' # 'kiwi.wav'#'
        self.dirName = self.config['dirpath']
        self.previousFile = None
        self.focusRegion = None

        # Make the window and associated widgets
        QMainWindow.__init__(self, root)
        self.setWindowTitle('AviaNZ')
        #self.createMenu()
        self.createFrame()

        # Make life easier for now: preload a birdsong
        #self.loadFile('../Birdsong/more1.wav')
        #self.loadFile('male1.wav')
        #self.loadFile('ST0026_1.wav')
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
        # This creates the actual interface. A bit of a mess of Qt widgets, and the connector calls
        self.defineColourmap()
        self.frame = QWidget()

        # This is the brief picture of the whole input figure for the zoomed-in version
        self.fig3 = Figure((22.0, 2.0), dpi=self.config['dpi'])
        self.canvas3 = FigureCanvas(self.fig3)
        self.canvas3.setParent(self.frame)
        self.canvas3.mpl_connect('button_press_event', self.fig3Click)
        self.canvas3.mpl_connect('motion_notify_event', self.fig3Drag)

        # This is the top figure
        self.fig = Figure(dpi=self.config['dpi'])
        self.fig.set_size_inches(22.0, 4.0, forward=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.frame)
        # This is for the click for start and end bars and other interactions on fig1
        self.canvas.mpl_connect('button_press_event', self.fig1Click)

        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()
        #self.mpl_toolbar = NavigationToolbar(self.canvas, self.frame)

        # Holds the list of files
        self.listFiles = QListWidget(self)
        self.listFiles.setFixedWidth(150)
        self.fillList()
        self.listFiles.connect(self.listFiles, SIGNAL('itemDoubleClicked(QListWidgetItem*)'), self.listLoadFile)

        # Whether or not user is drawing boxes around song in the spectrogram
        self.dragRectangles = QCheckBox('Drag boxes in spectrogram')
        self.dragRectangles.stateChanged[int].connect(self.dragRectanglesCheck)
        # The buttons on the right hand side, and also the spinbox for the window width

        self.leftBtn = QToolButton()
        self.leftBtn.setArrowType(Qt.LeftArrow)
        self.connect(self.leftBtn,SIGNAL('clicked()'),self.moveLeft)
        self.rightBtn = QToolButton()
        self.rightBtn.setArrowType(Qt.RightArrow)
        self.connect(self.rightBtn, SIGNAL('clicked()'), self.moveRight)
        # TODO: what else should be there?
        playButton1 = QPushButton("&Play Window")
        self.connect(playButton1, SIGNAL('clicked()'), self.playSegment)
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
        self.widthWindow = QDoubleSpinBox()
        self.widthWindow.setRange(0.5,900.0)
        self.widthWindow.setSingleStep(1.0)
        self.widthWindow.setDecimals(2)
        self.widthWindow.setValue(self.config['windowWidth'])
        self.widthWindow.valueChanged[float].connect(self.changeWidth)

        # This was a slider to run through a file
        #sld = QSlider(Qt.Horizontal, self)
        #sld.setFocusPolicy(Qt.NoFocus)
        #sld.valueChanged[int].connect(self.sliderMoved)

        # A context menu to select birds
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

        # This was an array of radio buttons and a list and a text entry box
        # Removed for now, replaced by the right button menu
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

        # This is the list of less common birds
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

        # This is the lower figure for the zoomed-in version
        self.fig2 = Figure((6.0, 4.0), dpi=self.config['dpi'])
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setParent(self.frame)
        self.canvas2.setFocusPolicy(Qt.ClickFocus)
        self.canvas2.setFocus()

        # The buttons that go below that figure
        playButton2 = QPushButton("&Play")
        self.connect(playButton2, SIGNAL('clicked()'), self.play)
        self.addButton = QPushButton("&Add segment")
        self.connect(self.addButton, SIGNAL('clicked()'), self.addSegmentClick)

        # This is the set of layout instructions. It's a bit of a mess, but basically looks like this:
        # { vbox1 }  -> hbox1
        #
        # {vbox0 | birds1layout birds2layout | birdListLayout | vbox3 | vboxbuttons} -> selectorLayout

        vbox0 = QVBoxLayout()
        vbox0.addWidget(self.listFiles)
        vbox0.addWidget(QLabel('Double click to select'))
        vbox0.addWidget(QLabel('Red names have segments'))

        vbox4a = QVBoxLayout()
        vbox4a.addWidget(QLabel('Visible window width (seconds)'))
        vbox4a.addWidget(self.widthWindow)

        hbox4a = QHBoxLayout()
        hbox4a.addWidget(QLabel('Slide top box to move through recording, click to start and end a segment, click on segment to edit or label. Right click to interleave.'))
        hbox4a.addWidget(playButton1)
        hbox4a.addWidget(self.dragRectangles)

        vbox4b = QVBoxLayout()
        vbox4b.addLayout(hbox4a)

        #vbox4b.addWidget(sld)

        hbox2 = QHBoxLayout()
        hbox2.addLayout(vbox4b)
        hbox2.addLayout(vbox4a)

        hbox00 = QHBoxLayout()
        hbox00.addWidget(self.canvas3)
        hbox00.addWidget(self.leftBtn)
        hbox00.addWidget(self.rightBtn)

        vbox1 = QVBoxLayout()
        vbox1.addLayout(hbox00)
        vbox1.addWidget(self.canvas)
        vbox1.addLayout(hbox2)
        #vbox1.addWidget(self.mpl_toolbar)

        vboxbuttons = QVBoxLayout()
        for w in [deleteButton,deleteAllButton,spectrogramButton,denoiseButton, segmentButton, quitButton]:
            vboxbuttons.addWidget(w)
            #vboxbuttons.setAlignment(w, Qt.AlignHCenter)

        hbox1 = QHBoxLayout()
        hbox1.addLayout(vbox1)

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

        hbox4 = QHBoxLayout()
        for w in [playButton2,self.addButton]:
            hbox4.addWidget(w)
            hbox4.setAlignment(w, Qt.AlignVCenter)

        vbox3 = QVBoxLayout()
        vbox3.addWidget(self.canvas2)
        vbox3.addWidget(QLabel('Click on a start/end to select, click again to move'))
        vbox3.addWidget(QLabel('Or use arrow keys. Press Backspace to delete'))
        vbox3.addLayout(hbox4)

        selectorLayout = QHBoxLayout()
        selectorLayout.addLayout(vbox0)
        #selectorLayout.addLayout(birds1Layout)
        #selectorLayout.addLayout(birds2Layout)
        #selectorLayout.addLayout(birdListLayout)
        selectorLayout.addLayout(vbox3)
        selectorLayout.addLayout(vboxbuttons)

        vbox2 = QVBoxLayout()
        vbox2.addLayout(hbox1)
        vbox2.addLayout(selectorLayout)

        # Now put everything into the frame
        self.frame.setLayout(vbox2)
        self.setCentralWidget(self.frame)

    def fillList(self):
        # Generates the list of files for the listbox on top left
        # Most of the work is to deal with directories in that list
        self.listOfFiles = QDir(self.dirName).entryInfoList(['..','*.wav'],filters=QDir.AllDirs|QDir.NoDot|QDir.Files,sort=QDir.DirsFirst)
        listOfDataFiles = QDir(self.dirName).entryList(['*.data'])
        listOfLongFiles = QDir(self.dirName).entryList(['*_1.wav'])
        for file in self.listOfFiles:
            if file.fileName()[:-4]+'_1.wav' in listOfLongFiles:
                # Ignore this entry
                pass
            else:
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
        self.box1id = None
        self.buttonID = None

        self.line = None
        self.segsRectsa1 = []
        self.segsRectsa2 = []
        self.segtext = []

        self.recta1 = None
        self.recta2 = None
        self.focusRegionSelected = False
        self.fig1Segment1 = None
        self.fig1Segment2 = None
        self.fig1Segmenting = False

    def listLoadFile(self,current):
        # Listener for when the user clicks on a filename
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
        # TODO: currently just takes the first channel of 2 -> what to do with the other?
        #if len(self.segments)>0:
        #    self.saveSegments()
        #self.segments = []

        if isinstance(name,str):
            self.filename = self.dirName+'/'+name
        else:
            self.filename = self.dirName+'/'+str(name.text())
        self.sampleRate, self.audiodata = wavfile.read(self.filename)
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
        self.sg = self.sp.spectrogram(self.audiodata,multitaper=True)

        # Load any previous segments stored
        if os.path.isfile(self.filename+'.data'):
            file = open(self.filename+'.data', 'r')
            self.segments = json.load(file)
            file.close()

        # Update the data that is seen by the other classes
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

        self.windowSize = self.config['windowWidth']

        # Reset this if the file is shorter than the window
        if float(len(self.audiodata))/self.sampleRate < self.windowSize:
            self.windowSize = float(len(self.audiodata))/self.sampleRate
        self.widthWindow.setValue(self.windowSize)

        # Set the width of the segment marker
        # This is the config width as a percentage of the window width, which is in seconds
        self.linewidtha1 = float(self.config['linewidtha1'])*self.windowSize

        # Get the height of the amplitude for plotting the box
        self.plotheight = np.abs(np.min(self.audiodata)) + np.max(self.audiodata)
        self.drawFig1()

    def openFile(self):
        # If have an open file option this will deal with it via a file dialog
        Formats = "Wav file (*.wav)"
        filename = QFileDialog.getOpenFileName(self, 'Open File', '/Users/srmarsla/Projects/AviaNZ', Formats)
        if filename != None:
            self.loadFile(filename)

    def dragRectanglesCheck(self,check):
        # The checkbox that says if the user is dragging rectangles or clicking on the spectrogram has changed state
        if self.dragRectangles.isChecked():
            #print "Listeners for drag Enabled"
            self.fig1motion = self.canvas.mpl_connect('motion_notify_event', self.fig1Drag)
            self.fig1endmotion = self.canvas.mpl_connect('button_release_event', self.fig1DragEnd)
        else:
            #print "Listeners for drag Disabled"
            self.canvas.mpl_disconnect(self.fig1motion)
            self.canvas.mpl_disconnect(self.fig1endmotion)

    def drawFig1(self):
        # This draws figure 1, amplitude and spectrogram plots
        # Also the topmost figure to show where you are up to in the file

        self.a1 = self.fig.add_subplot(211)
        self.a1.clear()
        self.a1.set_xlim(self.windowStart, self.windowSize)
        self.a1.plot(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=self.datalength,endpoint=True),self.audiodata)
        #self.a1.axis('off')

        self.a2 = self.fig.add_subplot(212)
        self.a2.clear()
        self.a2.imshow(10.0*np.log10(self.sg), cmap=self.cmap_grey, aspect='auto')

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

            self.listRectanglesa1.append(a1R)
            self.listRectanglesa2.append(a2R)
            a1t = self.a1.text(self.segments[count][0], np.min(self.audiodata), self.segments[count][4])
            # The font size is a pain because the window is in pixels, so have to transform it
            # Force a redraw to make bounding box available
            self.canvas.draw()
            # fs = a1t.get_fontsize()
            width = a1t.get_window_extent().inverse_transformed(self.a1.transData).width
            #print width, self.segments[count][1] - self.segments[count][0]
            if width > self.segments[count][1] - self.segments[count][0]:
                a1t.set_fontsize(8)

            self.a1text.append(a1t)

            # These are alternative ways to print the annotations that would enable putting it not inside the axes
            # But they are relative coordinates, so would be fiddly and have to move with the slider -> annoying!
            # self.a1.annotate()
            # self.fig.text(0.3,0.5,'xxx')
        self.fig.subplots_adjust(0.05,0.02,0.99,0.98)
        self.canvas.draw()

        self.topfig = self.fig3.add_axes((0.03, 0.01, 0.99, 0.98))
        self.topfig.imshow(10.0*np.log10(self.sg), cmap=self.cmap_grey,aspect='auto')
        self.topfig.axis('off')
        if self.focusRegion is not None:
            self.focusRegion.remove()
        self.focusRegion = self.topfig.add_patch(pl.Rectangle((self.windowStart*self.sampleRate/self.config['incr'], 0),
                                             self.windowSize*self.sampleRate / self.config['incr'],
                                             self.config['window_width'] / 2,
                                             facecolor='r',
                                             alpha=0.5))
        self.canvas3.draw()
        self.canvas.mpl_connect('key_press_event', self.fig1Keypress)

    def fig1DragEnd(self,event):
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
        self.canvas.draw()
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
        self.drawFig2()
        # Activate the radio buttons for labelling, selecting one from label if necessary
        self.activateRadioButtons()
        self.recta2.remove()
        self.recta2 = None
        self.canvas.draw()

    def fig1Drag(self,event):
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
        self.canvas.draw()

    def birdSelected(self,birdname):
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

    def fig1Click(self, event):
        # When user clicks in Fig 1 there are several options:
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
            self.drawFig2()

            # Put the context menu at the right point. The event.x and y are in pixels relative to the bottom left-hand corner of canvas (self.fig)
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
                # Use is selecting a region **on the spectrogram only**
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
                        self.segmenting = self.canvas.mpl_connect('motion_notify_event', self.fig1Selecting)
                        self.fig1Segment1 = self.a1.add_patch(pl.Rectangle((s+self.linewidtha1, np.min(self.audiodata)), 0, self.plotheight, facecolor='r', edgecolor='None',alpha=0.4))
                        self.fig1Segment2 = self.a2.add_patch(pl.Rectangle(((s+self.linewidtha1)*self.sampleRate / self.config['incr'], 0), 0, self.config['window_width']/2, facecolor='r', edgecolor='None', alpha=0.4))
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

                        a1R = self.a1.add_patch(pl.Rectangle((start, np.min(self.audiodata)), width, self.plotheight,facecolor='g', alpha=0.5))
                        a2R = self.a2.add_patch(pl.Rectangle((start*self.sampleRate / self.config['incr'], 0), width*self.sampleRate / self.config['incr'], self.config['window_width']/2,facecolor='g', alpha=0.5))
                        self.listRectanglesa1.append(a1R)
                        self.listRectanglesa2.append(a2R)
                        self.segments.append([start,max(self.start_a,s),0.0,self.sampleRate/2.,'None'])
                        a1t = self.a1.text(start, np.min(self.audiodata), 'None')
                        self.a1text.append(a1t)

                        # The font size is a pain because the window is in pixels, so have to transform it
                        # Force a redraw to make bounding box available
                        self.canvas.draw()
                        #fs = a1t.get_fontsize()
                        win_width = a1t.get_window_extent().inverse_transformed(self.a1.transData).width
                        #print width, win_width
                        if win_width > width:
                            a1t.set_fontsize(8)

                        # Delete the box and listener for the shading
                        self.canvas.mpl_disconnect(self.segmenting)
                        self.fig1Segment1.remove()
                        self.fig1Segment2.remove()

                        # Show it in the zoom window
                        self.zoomstart = a1R.xy[0]
                        self.zoomend = a1R.xy[0] + a1R.get_width()
                        self.box1id = len(self.segments)-1
                        self.drawFig2()
                        self.topBoxCol = 'r'
                        self.menuBirdList.popup(QPoint(event.x + event.canvas.x() + self.frame.x() + self.x(),
                                                       event.canvas.height() - event.y + event.canvas.y() + self.frame.y() + self.y()))
                        # Activate the radio buttons for labelling, selecting one from label if necessary
                        #self.activateRadioButtons()
                    # Switch to know if start or end or segment
                    self.start_stop = not self.start_stop

        self.canvas.draw()

    def fig1Selecting(self,event):
        # Update the red box showing the highlighted section
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
            width = event.xdata - self.fig1Segment1.get_x()
        else:
            width = (event.xdata - self.fig1Segment2.get_x()) / self.sampleRate * self.config['incr']

        self.fig1Segment1.set_width(width)
        self.fig1Segment2.set_width(width * self.sampleRate / self.config['incr'])
        self.canvas.draw()

    def fig1Keypress(self,event):
        # Listener for any key presses when focus is on Fig1
        if event.key == 'backspace':
            self.deleteSegment()

    def fig3Click(self,event):
        # Move the top box by first clicking on it, and then clicking again at the start of where you want it
        # Ignores other clicks
        if event.inaxes is None:
            return
        if self.focusRegionSelected == False:
            if self.focusRegion.get_x() <= event.xdata and self.focusRegion.get_x() + self.focusRegion.get_width() >= event.xdata:
                self.focusRegionSelected = True
                self.focusRegionPoint = event.xdata-self.focusRegion.get_x()
                self.focusRegion.set_facecolor('b')
                self.canvas3.draw()
        else:
            self.windowStart = (event.xdata-self.focusRegionPoint)/self.sampleRate*self.config['incr']
            if self.windowStart < 0:
                self.windowStart = 0
            elif self.windowStart + self.windowSize  > float(self.datalength) / self.sampleRate:
                self.windowStart = float(self.datalength) / self.sampleRate - self.windowSize
            self.focusRegionSelected = False
            self.focusRegion.set_facecolor('r')
            self.updateWindow()

    def fig3Drag(self,event):
        if event.inaxes is None:
            return
        if event.button!=1:
            return
        self.windowStart = event.xdata/self.sampleRate*self.config['incr']
        self.updateWindow()

    def updateWindow(self):
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
        self.canvas.draw()
        self.canvas3.draw()

    def moveLeft(self):
        # When the left button is pressed (on the top right of the screen, move everything along
        self.windowStart = max(0,self.windowStart-self.windowSize*0.9)
        self.updateWindow()

    def moveRight(self):
        # When the right button is pressed (on the top right of the screen, move everything along
        self.windowStart = min(float(self.datalength)/self.sampleRate-self.windowSize,self.windowStart+self.windowSize*0.9)
        self.updateWindow()

    #def sliderMoved(self,value):
    #    # When the slider is moved, update the top figures to reflect the new position in the file
    #    self.windowStart = (float(self.datalength)/self.sampleRate - self.windowSize)/100.0*value
    #    self.updateWindow()

    def showSegments(self):
        # This plots the segments that are returned from any of the segmenters.
        # It currently also deletes the previous ones. TODO: is that the best option?
        # TODO: also need to save the segments at some point

        # Delete the old segmentation
        for i in range(len(self.segsRectsa1)):
            self.segsRectsa1[i].remove()
            self.segsRectsa2[i].remove()
            self.segtext[i].remove()
        self.segtext = []
        self.segsRectsa1 = []
        self.segsRectsa2 = []
        for i in range(len(self.segs)):
            a1R = self.a1.add_patch(pl.Rectangle((self.segs[i][0], np.min(self.audiodata)),
                                                 self.segs[i][1] - self.segs[i][0],
                                                 self.plotheight,
                                           facecolor='r',alpha=0.5))
            a2R = self.a2.add_patch(pl.Rectangle((self.segs[i][0]* self.sampleRate / self.config['incr'], 0),
                                                 (self.segs[i][1] - self.segs[i][0])* self.sampleRate / self.config['incr'],
                                                 self.config['window_width'] / 2,
                                                 facecolor='r', alpha=0.5))
            self.segsRectsa1.append(a1R)
            self.segsRectsa2.append(a2R)
            a1t = self.a1.text(self.segs[i][0], np.min(self.audiodata), self.segs[i][2])
            # The font size is a pain because the window is in pixels, so have to transform it
            # Force a redraw to make bounding box available
            self.canvas.draw()
            # fs = a1t.get_fontsize()
            width = a1t.get_window_extent().inverse_transformed(self.a1.transData).width
            if width > self.segs[i][1] - self.segs[i][0]:
                a1t.set_fontsize(8)
            self.segtext.append(a1t)
        self.canvas.draw()

    def changeWidth(self, value):
        # This is the listener for the spinbox that decides the width of the main window. It updates the top figure plots as the window width is changed.
        # Slightly annoyingly, it gets called when the value gets reset, hence the first line
        if not hasattr(self,'a1'):
            return
        self.windowSize = value
        self.a1.set_xlim(self.windowStart, self.windowStart+self.windowSize)
        self.a2.set_xlim(self.windowStart*self.sampleRate / self.config['incr'], (self.windowStart + self.windowSize)*self.sampleRate/self.config['incr'])

        # Reset the width of the segment marker
        self.linewidtha1 = float(self.config['linewidtha1'])*self.windowSize

        self.focusRegion.remove()
        self.focusRegion = self.topfig.add_patch(pl.Rectangle((self.windowStart*self.sampleRate / self.config['incr'], 0),
                                                              self.windowSize*self.sampleRate / self.config['incr'],
                                                              self.config['window_width'] / 2,
                                                              facecolor='r',
                                                              alpha=0.5))
        self.canvas.draw()
        self.canvas3.draw()

    def spectrogramDialog(self):
        self.spectrogramDialog = Spectrogram()
        self.spectrogramDialog.show()
        self.spectrogramDialog.activateWindow()
        self.spectrogramDialog.activate.clicked.connect(self.spectrogram)

    def spectrogram(self):
        [alg, colourStart, colourEnd, window_width, incr] = self.spectrogramDialog.getValues()
        self.sp.set_width(int(str(window_width)), int(str(incr)))
        self.sg = self.sp.spectrogram(self.audiodata,str(alg))
        self.config['colourStart'] = float(str(colourStart))
        self.config['colourEnd'] = float(str(colourEnd))
        self.defineColourmap()
        self.drawFig1()

    def denoiseDialog(self):
        self.denoiseDialog = Denoise()
        self.denoiseDialog.show()
        self.denoiseDialog.activateWindow()
        self.denoiseDialog.activate.clicked.connect(self.denoise)
        self.denoiseDialog.undo.clicked.connect(self.denoise_undo)
        self.denoiseDialog.save.clicked.connect(self.denoise_save)

    def denoise(self):
        # This calls the denoiser and then plots the updated data
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
        self.sg = self.sp.spectrogram(self.audiodata)
        self.drawFig1()

    def denoise_undo(self):
        # TODO: Can I actually delete something from an object?
        print("Undoing",np.shape(self.audiodata_backup))
        if hasattr(self,'audiodata_backup'):
            if self.audiodata_backup is not None:
                if np.shape(self.audiodata_backup)[1]>0:
                    self.audiodata = np.copy(self.audiodata_backup[:,-1])
                    self.audiodata_backup = self.audiodata_backup[:,:-1]
                    self.sp.setNewData(self.audiodata,self.sampleRate)
                    self.sg = self.sp.spectrogram(self.audiodata)
                    if hasattr(self,'seg'):
                        self.seg.setNewData(self.audiodata,self.sg,self.sampleRate)
                    self.drawFig1()

    def denoise_save(self):
        # Save denoised data
        # Other players need them to be 16 bit, which is this magic number
        self.audiodata *= 32768.0
        self.audiodata = self.audiodata.astype('int16')
        filename = self.filename[:-4] + '_d' + self.filename[-4:]
        wavfile.write(filename, self.sampleRate, self.audiodata)

    def segmentationDialog(self):
        self.segmentDialog = Segmentation(np.max(self.audiodata))
        self.segmentDialog.show()
        self.segmentDialog.activateWindow()
        self.segmentDialog.activate.clicked.connect(self.segment)
        self.segmentDialog.save.clicked.connect(self.segments_save)

    def segment(self):
        [alg, ampThr, medThr,depth,thrType,thr,wavelet,bandchoice,start,end] = self.segmentDialog.getValues()
        if not hasattr(self,'seg'):
            self.seg = Segment.Segment(self.audiodata,self.sg,self.sp,self.sampleRate)
        if str(alg) == "Amplitude":
            self.segs = self.seg.segmentByAmplitude(float(str(ampThr)))
            if hasattr(self, 'line'):
                if self.line is not None:
                    self.line.remove()
            self.line = self.a1.add_patch(pl.Rectangle((0,float(str(ampThr))),len(self.audiodata),0,facecolor='r'))
        elif str(alg) == "Median Clipping":
            self.clip,blobs = self.seg.medianClip(float(str(medThr)))
            print np.shape(self.clip), np.shape(blobs)
            #print "here", np.shape(self.segs)
            self.a2.imshow(self.clip, cmap=self.cmap_grey, aspect='auto')
            indices = np.where(blobs[:, 2] > 3)

            # Seems to be offset by 1 for some reason
            for i in range(len(blobs)):
                self.a2.add_patch(pl.Rectangle((blobs[i].bbox[1]-1, blobs[i].bbox[0]-1),blobs[i].bbox[3]-blobs[i].bbox[1],blobs[i].bbox[2]-blobs[i].bbox[0],facecolor='g',alpha=0.3))

            self.canvas.draw()
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
            self.segs = self.seg.segmentByWavelet(thrType,float(str(thr)), int(str(depth)), wavelet,sampleRate,bandchoice,start,end,learning,)
        #self.showSegments()

        #sender().text()

    def segments_save(self):
        # This just appends those in the segslist to the segments list
        # TODO: Check for matching segments and remove duplicates
        # TODO: Then delete? Note that have to be saved before being annotated
        self.segments.extend(self.segments,self.segs)

    def recognise(self):
        pass

    def drawFig2(self):
        # This produces the plots for the zoomed-in window
        self.fig2.subplots_adjust(0.0, 0.0, 1.0,1.0)
        start = int(np.round(self.zoomstart*self.sampleRate))
        end = int(np.round(self.zoomend*self.sampleRate))
        self.listRectanglesb1 = []
        self.listRectanglesb2 = []
        # Make the start and end bands be big and draggable
        # Subtract off the thickness of the bars
        # When they are moved, update the plotting

        padding = int(np.round(self.config['padding']*(end-start)))
        # Draw the two charts
        self.a3 = self.fig2.add_subplot(211)
        self.a3.clear()
        xstart = max(start - padding, 0)
        if xstart != 0:
            start = padding
        xend = min(end + padding, len(self.audiodata))
        if xend != end+padding:
            end = xend-end

        self.a3.plot(self.audiodata[xstart:xend])
        self.a3.axis('off')

        self.a4 = self.fig2.add_subplot(212)
        self.a4.clear()
        newsg = self.sp.spectrogram(self.audiodata[xstart:xend])
        self.a4.imshow(10.0*np.log10(newsg), cmap=self.cmap_grey, aspect='auto')
        self.a4.axis('off')

        self.fig2width = xend - xstart
        self.bar_thickness = self.config['linewidtha2'] * self.fig2width
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
        # Keep track of where you are in Fig1
        self.offset_fig1 = self.zoomstart*self.sampleRate - start

        self.addButton.setEnabled(True)
        # Start the listener for picking the bars
        self.picker = self.canvas2.mpl_connect('pick_event', self.segmentFig2Picked)
        self.canvas2.draw()

    def segmentFig2Picked(self,event):
        # This is when the user clicks on one of the segment ends in figure 2
        if self.buttonID is not None:
            self.canvas2.mpl_disconnect(self.buttonID)
            self.canvas2.mpl_disconnect(self.keypress)
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

        # Set the bar to black and start the listeners
        self.Boxcol = self.Box1.get_facecolor()
        self.Box1.set_facecolor('black')
        self.Box2.set_facecolor('black')
        self.buttonID = self.canvas2.mpl_connect('button_press_event', self.moveSegmentEnd)
        self.keypress = self.canvas2.mpl_connect('key_press_event', self.keypressEnd)
        self.canvas2.draw()

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
        self.canvas2.mpl_disconnect(self.buttonID)
        self.canvas2.mpl_disconnect(self.keypress)
        self.buttonID = None
        self.keypress = None
        self.canvas2.draw()
        self.canvas.draw()

    def keypressEnd(self,event):
        # If the user has selected a segment in Fig2 and then presses a key, this deals with it
        if self.Box1 is None: return
        move_amount = self.config['move_amount']*self.fig2width
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
        self.canvas2.draw()
        self.canvas.draw()

    def addSegmentClick(self):
        # When the user clicks the button to add a segment in Fig2, this deals with it
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
                self.canvas2.mpl_disconnect(self.buttonID)
                self.canvas2.mpl_disconnect(self.keypress)
                self.keypress = None
                self.buttonID = None
                self.canvas2.draw()

        # Make a new listener to check for the button clicks
        self.buttonAdd = self.canvas2.mpl_connect('button_press_event', self.addSegmentFig2)

    def addSegmentFig2(self,event):
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
                self.markstarta1 = self.a1.add_patch(pl.Rectangle(((s + self.offset_fig1)/self.sampleRate, np.min(self.audiodata)), self.linewidtha1 ,
                                                                  self.plotheight, facecolor='r', edgecolor='None',
                                                                  alpha=0.8))
                self.markstarta2 = self.a2.add_patch(pl.Rectangle((s2 + self.offset_fig1/self.config['incr'], 0), self.linewidtha1 /self.config['incr']*self.sampleRate, self.config['window_width'], facecolor='r', edgecolor='None',alpha=0.8))
                self.listRectanglesb1.append(markstarta3)
                self.listRectanglesb2.append(markstarta4)
                self.end_a2 = s
                self.end_s2 = s2
            else:
                # This is the start of the second segment, draw green lines on fig2, then redraw the boxes on the top figure and save
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
                self.segments.append([old[0],(self.end_a2 + self.offset_fig1)/self.sampleRate,old[2],old[3],old[4]])
                self.segments.append([(s + self.offset_fig1)/self.sampleRate, old[1], old[2],old[3],old[4]])
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
                self.canvas.draw()
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
                self.canvas.draw()
                # fs = a1t.get_fontsize()
                width = a1t.get_window_extent().inverse_transformed(self.a1.transData).width
                #print width, self.segments[-1][1] - self.segments[-1][0]
                if width > self.segments[-1][1] - self.segments[-1][0]:
                    a1t.set_fontsize(8)

                # Stop the listener
                self.canvas2.mpl_disconnect(self.buttonAdd)

                # Make the addSegments button unselectable
                self.addButton.setEnabled(False)
                self.box1id = None
        self.canvas2.draw()
        self.canvas.draw()
        self.start_stop2 = not self.start_stop2
        # For now at least, stop the canvas2 listener
        self.canvas2.mpl_disconnect(self.picker)

    def playSegment(self):
        # This is the listener for the play button. A very simple wave file player

        #print(self.windowStart, self.sampleRate, self.windowSize)
        # Move a marker through in real time?!
        # self.playbar = self.a1.add_patch(
        #     pl.Rectangle((self.windowStart+0.1, np.min(self.audiodata)), self.linewidtha1, self.plotheight, facecolor='k',
        #                  edgecolor='None', alpha=0.8))
        # self.canvas.draw()

        sd.play(self.audiodata[int(self.windowStart*self.sampleRate):int(self.windowStart*self.sampleRate+self.windowSize*self.sampleRate)],self.sampleRate)

        # end_time = self.windowStart + self.windowSize
        # start_time = datetime.datetime.now()
        # current_time = start_time
        # step = 0
        # while (current_time - start_time).total_seconds() < end_time:
        #     now = datetime.datetime.now()
        #     timedelta = (now - current_time).total_seconds()
        #     step += timedelta
        #     #self.playbar.set_x(self.playbar.get_x() + 0.05)
        #     self.a1.add_patch(
        #         pl.Rectangle((step, np.min(self.audiodata)), self.linewidtha1, self.plotheight,
        #                      facecolor='k',
        #                      edgecolor='None', alpha=0.8))
        #     #print self.playbar.get_x()
        #     current_time = now
        #     self.canvas.draw()

        # blocksize = 16
        # s = Stream(samplerate=self.sampleRate, blocksize=blocksize)
        # s.start()
        # s.write(self.audiodata[int(self.windowStart*self.sampleRate):int(self.windowStart*self.sampleRate+self.windowSize*self.sampleRate)])
        # s.stop()

    def play(self):
        # This is the listener for the play button. A very simple wave file player
        sd.play(self.audiodata[int(self.zoomstart*self.sampleRate):int(self.zoomend*self.sampleRate)],self.sampleRate)
        # blocksize = 16
        # s = Stream(samplerate=self.sampleRate, blocksize=blocksize)
        # s.start()
        # s.write(self.audiodata[int(self.zoomstart*self.sampleRate):int(self.zoomend*self.sampleRate)])
        # s.stop()

    def updateText(self,text):
        self.segments[self.box1id][4] = text
        self.a1text[self.box1id].set_text(text)
        # The font size is a pain because the window is in pixels, so have to transform it
        # Force a redraw to make bounding box available
        self.canvas.draw()
        # fs = self.a1text[self.box1id].get_fontsize()
        width = self.a1text[self.box1id].get_window_extent().inverse_transformed(self.a1.transData).width
        if width > self.segments[self.box1id][1] - self.segments[self.box1id][0]:
            self.a1text[self.box1id].set_fontsize(8)
        if self.segments[self.box1id][4] != "Don't Know":
            facecolour = 'b'
        else:
            facecolour = 'r'
        self.listRectanglesa1[self.box1id].set_facecolor(facecolour)
        self.listRectanglesa2[self.box1id].set_facecolor(facecolour)
        self.topBoxCol = self.listRectanglesa1[self.box1id].get_facecolor()
        self.canvas.draw()

    def radioBirdsClicked(self):
        # Listener for when the user selects a radio button
        # Update the text and store the data
        for button in self.birds1+self.birds2:
            if button.isChecked():
                if button.text()=="Other":
                    self.birdList.setEnabled(True)
                else:
                    self.birdList.setEnabled(False)
                    self.updateText(str(button.text()))

    def listBirdsClicked(self, item):
        # Listener for the listbox of birds
        if (item.text() == "Other"):
            self.tbox.setEnabled(True)
        else:
            # Save the entry
            self.updateText(str(item.text()))
            # self.segments[self.box1id][4] = str(item.text())
            # self.a1text[self.box1id].set_text(str(item.text()))
            # # The font size is a pain because the window is in pixels, so have to transform it
            # # Force a redraw to make bounding box available
            # self.canvas.draw()
            # # fs = a1t.get_fontsize()
            # width = self.a1text[self.box1id].get_window_extent().inverse_transformed(self.a1.transData).width
            # if width > self.segments[self.box1id][1] - self.segments[self.box1id][0]:
            #     self.a1text[self.box1id].set_fontsize(8)
            #
            # self.listRectanglesa1[self.box1id].set_facecolor('b')
            # self.listRectanglesa2[self.box1id].set_facecolor('b')
            # self.topBoxCol = self.listRectanglesa1[self.box1id].get_facecolor()
            # self.canvas.draw()

    def birdTextEntered(self):
        # Listener for the text entry
        # Check text isn't already in the listbox, and add if not
        # Doesn't sort the list, but will when program is closed
        item = self.birdList.findItems(self.tbox.text(),Qt.MatchExactly)
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
        # self.canvas.draw()
        # # fs = self.a1text[self.box1id].get_fontsize()
        # width = self.a1text[self.box1id].get_window_extent().inverse_transformed(self.a1.transData).width
        # if width > self.segments[self.box1id][1] - self.segments[self.box1id][0]:
        #     self.a1text[self.box1id].set_fontsize(8)
        #
        # self.listRectanglesa1[self.box1id].set_facecolor('b')
        # self.listRectanglesa2[self.box1id].set_facecolor('b')
        # self.topBoxCol = self.listRectanglesa1[self.box1id].get_facecolor()
        # self.canvas.draw()
        self.saveConfig=True
        #self.tbox.setEnabled(False)


    def deleteSegment(self):
        # Delete segment if one is selected
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
            self.canvas.draw()
            # Make canvas2 blank
            self.a3.clear()
            self.a3.axis('off')
            self.a4.clear()
            self.canvas2.draw()

    def deleteAll(self):
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

        self.canvas.draw()
        self.canvas2.draw()
        self.canvas3.draw()

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
        # This is to catch the user closing the window by clicking the Close button instead of quitting
        self.quit()

    def quit(self):
        # Listener for the quit button
        print("Quitting")
        self.saveSegments()
        if self.saveConfig == True:
            print "Saving config file"
            json.dump(self.config, open(self.configfile, 'wb'))
        QApplication.quit()

    def splitFile5mins(self, name):
        # Nirosha wants to split files that are long (15 mins) into 5 min segments
        # Could be used when loading long files :)
        try:
            self.sampleRate, self.audiodata = wavfile.read(name)
        except:
            print("Error: try another file")
        nsamples = np.shape(self.audiodata)[0]
        lengthwanted = self.sampleRate * 60 * 5
        count = 0
        while (count + 1) * lengthwanted < nsamples:
            data = self.audiodata[count * lengthwanted:(count + 1) * lengthwanted]
            filename = name[:-4] + '_' +str(count) + name[-4:]
            wavfile.write(filename, self.sampleRate, data)
            count += 1
        data = self.audiodata[(count) * lengthwanted:]
        filename = name[:-4] + '_' + str((count)) + name[-4:]
        wavfile.write(filename, self.sampleRate, data)

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


class Spectrogram(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Signal Processing Options')

        self.algs = QComboBox()
        self.algs.addItems(['Hanning','Parzen','Welch','Hamming','Blackman','BlackmanHarris'])

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
    # TODO: add the wavelet params
    # TODO: work out how to return varying size of params, also process them
    # TODO: test and play
    def __init__(self, maxv, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Segmentation Options')

        self.algs = QComboBox()
        self.algs.addItems(["Amplitude","Median Clipping","Wavelets"])
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

        Box = QVBoxLayout()
        Box.addWidget(self.algs)
        # Labels
        self.amplabel = QLabel("Set threshold amplitude")
        Box.addWidget(self.amplabel)
        self.medlabel = QLabel("Set median threshold")
        Box.addWidget(self.medlabel)
        self.medlabel.hide()

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
        self.medThr.hide()

        Box.addWidget(self.activate)
        Box.addWidget(self.save)

        # Now put everything into the frame
        self.setLayout(Box)

    def changeBoxes(self,alg):
        # This does the hiding and showing of the options as the algorithm changes
        if self.prevAlg == "Amplitude":
            self.amplabel.hide()
            self.ampThr.hide()
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

# Start the application
app = QApplication(sys.argv)
form = Interface(configfile='AviaNZconfig.txt')
form.show()
app.exec_()
