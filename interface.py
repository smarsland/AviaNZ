# Interface.py
#
# This is currently the base class for the AviaNZ interface
# It's fairly simplistic, but hopefully works
# Version 0.4 30/5/16
# Author: Stephen Marsland

import sys, os, glob, json
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
import Features
import Learning
# ==============
# TODO

# Needs decent testing -> suggestions from Nirosha
    # Use new forms for denoise/segment to allow parameter selection
# For overlapping bird calls, need two spectrograms
# Font size to match segment size

# Make threshold only update when press go
# Size of zoom window?

# Make a 'final' version for Kim -> list of 10 files and that's all in the list box
#   -> it should make a new folder with understandable name, copy them in, start, record times as well, remove .. as option
# Time is datetime.datetime.now().time()
# How to deploy it for her?
# 1 page user manual

# For second plot, how wide should padding be, and how much should the move_amount be? -> needs work, relating to windowWidth

# Look at raven and praat -> what else is actually useful? Other annotations on graphs?

# Need to have another looking at denoising and work out what is happening, e.g., the tril1 call??
# For segmenter, work out how to delete the old threshold lines m
# Make it plot the segments

# Given files > 5 mins, split them into 5 mins versions anyway (code is there, make it part of workflow)
# Don't really want to load the whole thing, just 5 mins, and then move through with arrows -> how?

# Clear interface to plug in segmenters, labellers, etc. -> just use time in seconds?

# As well as this version with pictures, will need to be able to call various bits to work offline
# denoising, segmentation, etc. that will then just show some pictures

# Add option somewhere to change windowing for spectrogram, and any other params -> easier for me to play!

# Turn stereo sound into mono using librosa, consider always resampling to 22050Hz?

# Get suggestions from the others
# ===============

class Interface(QMainWindow):

    def __init__(self,root=None,configfile=None):
        self.root = root
        if configfile is not None:
            self.config = json.load(open(configfile))
            self.configfile=configfile
        else:
            self.config = {
            # Params for spectrogram
            'window_width':256,
            'incr':128,

            # Params for denoising
            'maxSearchDepth':20,
            'dirpath':'.',

            # Params for the width of the lines to draw for segmentation in the two plots
            'linewidtha1':1,
            'minbar_thickness':200,
            'a2barthickness':0.02,

            # Params for the extra padding in the second figure, and how much keys move bars by
            'padding':3000,
            'move_amount':100,

            # Param for width in seconds of the main representation
            'windowWidth':10.0,  #

            # These are the contrast parameters for the spectrogram
            'colourStart':0.4,
            'colourEnd':1.0,

            'dpi':100,
            'ListBirdsEntries':['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            }
            self.configfile = 'AviaNZconfig.txt'

        self.resetStorageArrays()

        self.firstFile = 'ruru.wav'
        self.dirName = self.config['dirpath']
        self.previousFile = None

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

    def createFrame(self):
        # This creates the actual interface. A bit of a mess of Qt widgets, and the connector calls
        self.defineColourmap()
        self.frame = QWidget()

        # This is the brief picture of the whole input figure for the zoomed-in version
        self.fig3 = Figure((22.0, 2.0), dpi=self.config['dpi'])
        self.canvas3 = FigureCanvas(self.fig3)
        self.canvas3.setParent(self.frame)

        # This is the top figure
        self.fig = Figure(dpi=self.config['dpi'])
        self.fig.set_size_inches(22.0, 4.0, forward=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.frame)
        self.canvas.mpl_connect('button_press_event', self.fig1Click)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()
        #self.mpl_toolbar = NavigationToolbar(self.canvas, self.frame)

        # Holds the list of files
        self.listFiles = QListWidget(self)
        self.listFiles.setFixedWidth(150)
        self.fillList()
        self.listFiles.connect(self.listFiles, SIGNAL('itemDoubleClicked(QListWidgetItem*)'), self.listLoadFile)

        # The buttons on the right hand side, and also the spinbox for the window width
        # TODO: what else should be there?
        playButton1 = QPushButton("&Play Window")
        self.connect(playButton1, SIGNAL('clicked()'), self.playSegment)
        quitButton = QPushButton("&Quit")
        self.connect(quitButton, SIGNAL('clicked()'), self.quit)
        segmentButton = QPushButton("&Segment")
        self.connect(segmentButton, SIGNAL('clicked()'), self.segment)
        denoiseButton = QPushButton("&Denoise")
        self.connect(denoiseButton, SIGNAL('clicked()'), self.denoise)
        recogniseButton = QPushButton("&Recognise")
        self.connect(recogniseButton, SIGNAL('clicked()'), self.recognise)
        deleteButton = QPushButton("&Delete Current Segment")
        self.connect(deleteButton, SIGNAL('clicked()'), self.deleteSegment)
        deleteAllButton = QPushButton("&Delete All Segments")
        self.connect(deleteAllButton, SIGNAL('clicked()'), self.deleteAll)
        widthWindow = QDoubleSpinBox()
        widthWindow.setRange(0.5,900.0)
        widthWindow.setSingleStep(1.0)
        widthWindow.setDecimals(2)
        widthWindow.setValue(self.config['windowWidth'])
        widthWindow.valueChanged[float].connect(self.changeWidth)

        # Not necessarily permanent: spin box for amplitude threshold
        self.ampThr = QDoubleSpinBox()
        self.ampThr.setRange(0.001,0.3)
        self.ampThr.setSingleStep(0.002)
        self.ampThr.setDecimals(4)
        self.ampThr.setValue(0.02)
        self.ampThr.valueChanged[float].connect(self.changeAmpThr)

        # This is the slider to run through a file
        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.valueChanged[int].connect(self.sliderMoved)

        # Create an array of radio buttons for the most common birds (2 columns of 10 choices)
        # TODO: make this part of the config
        self.birds1 = [QRadioButton("Female Kiwi"), QRadioButton("Male Kiwi"), QRadioButton("Ruru"), QRadioButton("Hihi"),
                  QRadioButton("Bittern"), QRadioButton("Petrel"), QRadioButton("Robin"), QRadioButton("Tomtit"), QRadioButton("Cuckoo"),QRadioButton("Kereru")]
        self.birds2 = [QRadioButton("Tui"), QRadioButton("Bellbird"), QRadioButton("Fantail"), QRadioButton("Saddleback"), QRadioButton("Silvereye"), QRadioButton("Rifleman"), QRadioButton("Warbler"),
                  QRadioButton("Not Bird"), QRadioButton("Don't Know"), QRadioButton("Other")]

        for i in xrange(len(self.birds1)):
            self.birds1[i].setEnabled(False)
            self.connect(self.birds1[i], SIGNAL("clicked()"), self.radioBirdsClicked)
        for i in xrange(len(self.birds2)):
            self.birds2[i].setEnabled(False)
            self.connect(self.birds2[i], SIGNAL("clicked()"), self.radioBirdsClicked)

        # This is the list of less common birds
        self.birdList = QListWidget(self)
        self.birdList.setMaximumWidth(150)
        for item in self.config['ListBirdsEntries']:
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
        vbox4a.addWidget(widthWindow)

        hbox4a = QHBoxLayout()
        hbox4a.addWidget(QLabel('Slide to move through recording, click to start and end a segment, click on segment to edit or label'))
        hbox4a.addWidget(playButton1)

        vbox4b = QVBoxLayout()
        vbox4b.addLayout(hbox4a)
        vbox4b.addWidget(sld)

        hbox2 = QHBoxLayout()
        hbox2.addLayout(vbox4b)
        hbox2.addLayout(vbox4a)

        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.canvas3)
        vbox1.addWidget(self.canvas)
        vbox1.addLayout(hbox2)
        #vbox1.addWidget(self.mpl_toolbar)

        vboxbuttons = QVBoxLayout()
        for w in [deleteButton,deleteAllButton,denoiseButton, segmentButton, quitButton]:
            vboxbuttons.addWidget(w)
            #vboxbuttons.setAlignment(w, Qt.AlignHCenter)

        vboxbuttons.addWidget(QLabel('Amplitude threshold'))
        vboxbuttons.addWidget(self.ampThr)

        hbox1 = QHBoxLayout()
        hbox1.addLayout(vbox1)

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
        selectorLayout.addLayout(birds1Layout)
        selectorLayout.addLayout(birds2Layout)
        selectorLayout.addLayout(birdListLayout)
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
        self.start_stop = 0
        self.start_stop2 = 0

        # Keep track of start points and selected buttons
        self.start_a = 0
        self.windowStart = 0
        self.box1id = None
        self.buttonID = None

        self.segsRects = []

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
        self.datamax = np.shape(self.audiodata)[0]

        # Create an instance of the Signal Processing class
        if not hasattr(self,'sp'):
            self.sp = SignalProc.SignalProc(self.audiodata, self.sampleRate,self.config['window_width'],self.config['incr'])

        # Get the data for the spectrogram
        self.sg = self.sp.spectrogram(self.audiodata)

        # Load any previous segments stored
        if os.path.isfile(self.filename+'.data'):
            file = open(self.filename+'.data', 'r')
            self.segments = json.load(file)
            file.close()

        # Set the values for the segmentation thresholds
        self.ampThr.setRange(0.001,np.max(self.audiodata)+0.001)
        self.ampThr.setSingleStep(0.001)
        self.ampThr.setDecimals(4)
        self.ampThr.setValue(np.max(self.audiodata)+0.001)

        self.windowSize = self.config['windowWidth']
        self.drawFig1()

    def openFile(self):
        # If have an open file option this will deal with it via a file dialog
        Formats = "Wav file (*.wav)"
        filename = QFileDialog.getOpenFileName(self, 'Open File', '/Users/srmarsla/Projects/AviaNZ', Formats)
        if filename != None:
            self.loadFile(filename)

    def drawFig1(self):
        self.linewidtha1 = float(self.config['linewidtha1'])/self.windowSize
        # This draws figure 1, amplitude and spectrogram plots
        # Also the topmost figure to show where you are up to in the file
        self.a1 = self.fig.add_subplot(211)
        self.a1.clear()
        self.a1.set_xlim(self.windowStart, self.windowSize)
        self.a1.plot(np.linspace(0.0,float(self.datamax)/self.sampleRate,num=self.datamax,endpoint=True),self.audiodata)
        #self.a1.axis('off')

        self.a2 = self.fig.add_subplot(212)
        self.a2.clear()
        self.a2.imshow(self.sg, cmap=self.cmap_grey, aspect='auto')
        l = [str(0),str(self.sampleRate/2)]
        for i in range(len(self.a2.axes.get_yticklabels())-4):
            l.append('')
        l.append(str(0))
        self.a2.axes.set_yticklabels(l)
        self.a2.set_xlim(self.windowStart*self.sampleRate / self.config['incr'], self.windowSize*self.sampleRate / self.config['incr'])

        # If there are segments, show them
        for count in range(len(self.segments)):
            if self.segments[count][2] == 'None' or self.segments[count][2] == "Don't Know":
                facecolour = 'r'
            else:
                facecolour = 'b'
            a1R = self.a1.add_patch(pl.Rectangle((self.segments[count][0], np.min(self.audiodata)),
                                                 self.segments[count][1] - self.segments[count][0],
                                                 np.abs(np.min(self.audiodata)) + np.max(self.audiodata),
                                                 facecolor=facecolour,
                                                 alpha=0.5))
            a2R = self.a2.add_patch(pl.Rectangle((self.segments[count][0]*self.sampleRate / self.config['incr'], 0),
                                                 self.segments[count][1]*self.sampleRate / self.config['incr'] - self.segments[count][
                                                     0]*self.sampleRate / self.config['incr'],
                                                 self.config['window_width'] / 2, facecolor=facecolour,
                                                 alpha=0.5))
            self.listRectanglesa1.append(a1R)
            self.listRectanglesa2.append(a2R)
            a1t = self.a1.text(self.segments[count][0], np.min(self.audiodata), self.segments[count][2])
            self.a1text.append(a1t)

            # These are alternative ways to print the annotations that would enable putting it not inside the axes
            # But they are relative coordinates, so would be fiddly and have to move with the slider -> annoying!
            # self.a1.annotate()
            # self.fig.text(0.3,0.5,'xxx')
        self.fig.subplots_adjust(0.05,0.02,0.99,0.98)
        self.canvas.draw()

        self.topfig = self.fig3.add_axes((0.03, 0.01, 0.99, 0.98))
        self.topfig.imshow(self.sg, cmap=self.cmap_grey,aspect='auto')
        self.topfig.axis('off')
        self.focusRegion = self.topfig.add_patch(pl.Rectangle((self.windowStart*self.sampleRate/ self.config['incr'], 0),
                                             self.windowSize*self.sampleRate / self.config['incr'],
                                             self.config['window_width'] / 2,
                                             facecolor='r',
                                             alpha=0.5))
        self.canvas3.draw()
        self.canvas.mpl_connect('key_press_event', self.fig1Keypress)

    def fig1Click(self, event):
        # When user clicks in Fig 1 there are two options depending whether you click in
        # a box that is already made (select it, show the zoomed-in version, and prepare to label it)
        # or anywhere else (segmenting, either the start (mark it) or the end (add the box, save the segment)

        # If there was a previous box, turn it the right colour again
        if self.box1id>-1:
            self.listRectanglesa1[self.box1id].set_facecolor(self.topBoxCol)
            self.listRectanglesa2[self.box1id].set_facecolor(self.topBoxCol)

        # Deactivate the radio buttons and listbox
        for i in xrange(len(self.birds1)):
            self.birds1[i].setChecked(False)
            self.birds1[i].setEnabled(False)
        for i in xrange(len(self.birds2)):
            self.birds2[i].setChecked(False)
            self.birds2[i].setEnabled(False)
        self.birdList.setEnabled(False)
        self.tbox.setEnabled(False)

        # Check if the user has clicked in a box
        box1id = -1
        for count in range(len(self.listRectanglesa1)):
            if self.listRectanglesa1[count].xy[0] <= event.xdata and self.listRectanglesa1[count].xy[0]+self.listRectanglesa1[count].get_width() >= event.xdata:
                box1id = count
            if self.listRectanglesa2[count].xy[0] <= event.xdata and self.listRectanglesa2[count].xy[0]+self.listRectanglesa2[count].get_width() >= event.xdata:
                box1id = count

        # If they have clicked in a box, store its colour for later resetting
        # Show the zoomed-in version and enable the radio buttons
        if box1id>-1:
            self.box1id = box1id
            self.topBoxCol = self.listRectanglesa1[box1id].get_facecolor()
            self.listRectanglesa1[box1id].set_facecolor('green')
            self.listRectanglesa2[box1id].set_facecolor('green')
            self.zoomstart = self.listRectanglesa1[box1id].xy[0]
            self.zoomend = self.listRectanglesa1[box1id].xy[0]+self.listRectanglesa1[box1id].get_width()
            self.drawFig2()

            # Activate the radio buttons for labelling, selecting one from label if necessary
            found = False
            for i in range(len(self.birds1)):
                self.birds1[i].setEnabled(True)
                if str(self.a1text[box1id].get_text()) == self.birds1[i].text():
                    self.birds1[i].setChecked(True)
                    found = True
            for i in range(len(self.birds2)):
                self.birds2[i].setEnabled(True)
                if str(self.a1text[box1id].get_text()) == self.birds2[i].text():
                    self.birds2[i].setChecked(True)
                    found = True

            if not found:
                # Select 'Other' in radio buttons (last item) and activate the listwidget
                self.birds2[-1].setChecked(True)
                self.birdList.setEnabled(True)
                items = self.birdList.findItems(str(self.a1text[box1id].get_text()), Qt.MatchExactly)
                for item in items:
                    self.birdList.setCurrentItem(item)
        else:
            # User is doing segmentation
            # Work out which axes they have clicked in. s holds the point they have clicked on
            # in coordinates of the top plot (amplitude)
            a1 = str(self.a1)
            a1ind = float(a1[a1.index(',') + 1:a1.index(';')])

            if event.inaxes is not None:
                s = event.xdata

                a = str(event.inaxes)
                aind = float(a[a.index(',') + 1:a.index(';')])
                if aind != a1ind:
                    s = s*self.config['incr']/self.sampleRate

                if s>0 and s<self.datamax/self.sampleRate:
                    if self.start_stop==0:
                        # This is the start of a segment, draw a green line
                        self.markstarta1 = self.a1.add_patch(pl.Rectangle((s, np.min(self.audiodata)), self.linewidtha1, np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='g', edgecolor='None',alpha=0.8))
                        self.markstarta2 = self.a2.add_patch(pl.Rectangle((s*self.sampleRate / self.config['incr'] , 0), self.linewidtha1/self.config['incr']*self.sampleRate, self.config['window_width']/2, facecolor='g', edgecolor='None',alpha=0.8))
                        self.start_a = s
                    else:
                        # This is the end, draw the box, save the data, update the text
                        # And show it in the zoom window
                        # TODO: make the radio buttons available too?
                        self.markstarta1.remove()
                        self.markstarta2.remove()
                        a1R = self.a1.add_patch(pl.Rectangle((self.start_a, np.min(self.audiodata)), s - self.start_a, np.abs(np.min(self.audiodata)) + np.max(self.audiodata),facecolor='r', alpha=0.5))
                        a2R = self.a2.add_patch(pl.Rectangle((self.start_a*self.sampleRate / self.config['incr'], 0), (s - self.start_a)*self.sampleRate / self.config['incr'], self.config['window_width']/2,facecolor='r', alpha=0.5))
                        self.listRectanglesa1.append(a1R)
                        self.listRectanglesa2.append(a2R)
                        self.segments.append([min(self.start_a,s),max(self.start_a,s),'None'])
                        a1t = self.a1.text(self.start_a, np.min(self.audiodata), 'None')
                        self.a1text.append(a1t)
                        self.topBoxCol = self.listRectanglesa1[box1id].get_facecolor()
                        # Show it in the zoom window
                        self.zoomstart = a1R.xy[0]
                        self.zoomend = a1R.xy[0] + a1R.get_width()
                        self.box1id = len(self.segments)-1
                        self.drawFig2()
                    # Switch to know if start or end or segment
                    self.start_stop = 1 - self.start_stop

        self.canvas.draw()

    def fig1Keypress(self,event):
        # Listener for any key presses when focus is on Fig1
        if event.key == 'backspace':
            self.deleteSegment()

    def sliderMoved(self,value):
        # When the slider is moved, update the top figures to reflect the new position in the file
        self.windowStart = (self.datamax/self.sampleRate - self.windowSize)/100.0*value
        self.a1.set_xlim(self.windowStart, self.windowStart+self.windowSize)
        self.a2.set_xlim(self.windowStart*self.sampleRate/self.config['incr'], (self.windowStart + self.windowSize)*self.sampleRate/self.config['incr'])

        self.focusRegion.remove()
        self.focusRegion = self.topfig.add_patch(pl.Rectangle((self.windowStart*self.sampleRate/self.config['incr'], 0),
                                                              self.windowSize*self.sampleRate / self.config['incr'],
                                                              self.config['window_width'] / 2,
                                                              facecolor='r',
                                                              alpha=0.5))
        self.canvas.draw()
        self.canvas3.draw()

    def changeAmpThr(self,value):
        seg = Segment.Segment(self.audiodata)
        segs = seg.segmentByAmplitude(value)
        if hasattr(self,'line'):
            # TODO How to delete the line?
            #self.line.remove()
            pass
        self.line = self.a1.plot(np.arange(len(self.audiodata)), np.ones(len(self.audiodata)) * value)
        for i in range(len(self.segsRects)):
            self.segsRects[i].remove()
        self.segsRects = []
        for i in range(len(segs)):
            self.segsRects.append(self.a1.add_patch(pl.Rectangle((segs[i][0], np.min(self.audiodata)),
                                           segs[i][1] - segs[i][0],
                                           np.abs(np.min(self.audiodata)) + np.max(self.audiodata),
                                           facecolor='k',alpha=0.5)))
        self.canvas.draw()

    def changeWidth(self, value):
        # This is the listener for the spinbox. It updates the top figure plots as the window width is changed
        self.windowSize = value
        self.a1.set_xlim(self.windowStart, self.windowStart+self.windowSize)
        self.a2.set_xlim(self.windowStart*self.sampleRate / self.config['incr'], (self.windowStart + self.windowSize)*self.sampleRate/self.config['incr'])

        self.focusRegion.remove()
        self.focusRegion = self.topfig.add_patch(pl.Rectangle((self.windowStart*self.sampleRate / self.config['incr'], 0),
                                                              self.windowSize*self.sampleRate / self.config['incr'],
                                                              self.config['window_width'] / 2,
                                                              facecolor='r',
                                                              alpha=0.5))
        self.canvas.draw()
        self.canvas3.draw()

    def denoise(self):
        # This calls the denoiser and then plots the updated data
        # TODO: should it be saved automatically, or a button added?
        print "Denoising"
        self.audiodata = self.sp.denoise()
        print "Done"
        self.sg = self.sp.spectrogram(self.audiodata)
        self.drawFig1()

    def segment(self):
        pass

    def recognise(self):
        pass

    def drawFig2(self):
        # This produces the plots for the zoomed-in window
        self.fig2.subplots_adjust(0.0, 0.0, 1.0,1.0)
        start = self.zoomstart*self.sampleRate
        end = self.zoomend*self.sampleRate
        self.listRectanglesb1 = []
        self.listRectanglesb2 = []
        # Make the start and end bands be big and draggable
        # Subtract off the thickness of the bars
        # When they are moved, update the plotting
        self.bar_thickness = self.config['a2barthickness'] * (end - start)
        self.bar_thickness = max(self.bar_thickness, self.config['minbar_thickness'])
        # Draw the two charts
        self.a3 = self.fig2.add_subplot(211)
        self.a3.clear()
        xstart = max(start - self.config['padding'], 0)
        if xstart != 0:
            start = self.config['padding']
        xend = min(end + self.config['padding'], len(self.audiodata))
        self.a3.plot(self.audiodata[xstart:xend])
        self.a3.axis('off')

        self.a4 = self.fig2.add_subplot(212)
        self.a4.clear()
        newsg = self.sp.spectrogram(self.audiodata[xstart:xend])
        self.a4.imshow(newsg, cmap=self.cmap_grey, aspect='auto')
        self.a4.axis('off')

        self.End1 = self.a3.add_patch(pl.Rectangle((start - self.bar_thickness, np.min(self.audiodata)), self.bar_thickness,
                                                   np.abs(np.min(self.audiodata)) + np.max(self.audiodata),
                                                   facecolor='g',
                                                   edgecolor='None', alpha=1.0, picker=10))
        self.End2 = self.a4.add_patch(
            pl.Rectangle(((start - self.bar_thickness) / self.config['incr'], 0), self.bar_thickness / self.config['incr'],
                         self.config['window_width'] / 2, facecolor='g',
                         edgecolor='None', alpha=1.0, picker=10))
        self.End3 = self.a3.add_patch(
            pl.Rectangle((xend - self.zoomstart*self.sampleRate + self.bar_thickness, np.min(self.audiodata)), self.bar_thickness,
                         np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r',
                         edgecolor='None', alpha=1.0, picker=10))
        self.End4 = self.a4.add_patch(pl.Rectangle(((xend - self.zoomstart*self.sampleRate + self.bar_thickness) / self.config['incr'], 0),
                                                   self.bar_thickness / self.config['incr'],
                                                   self.config['window_width'] / 2, facecolor='r',
                                                   edgecolor='None', alpha=1.0, picker=10))
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
        # TODO: how to remove the listener callback?
        if event.key == 'left':
            self.Box1.set_x(self.Box1.get_x() - self.config['move_amount'])
            self.Box2.set_x(self.Box2.get_x() - float(self.config['move_amount']) / self.config['incr'])
            if self.Boxcol == matplotlib.colors.colorConverter.to_rgba('g'):
                self.listRectanglesa1[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x() - float(self.config['move_amount'])/self.sampleRate)
                self.listRectanglesa1[self.box1id].set_width(self.listRectanglesa1[self.box1id].get_width() + float(self.config['move_amount'])/self.sampleRate)
                self.listRectanglesa2[self.box1id].set_x(self.listRectanglesa2[self.box1id].get_x() - float(self.config['move_amount'])/self.config['incr'])
                self.listRectanglesa2[self.box1id].set_width(self.listRectanglesa2[self.box1id].get_width() + float(self.config['move_amount'])/self.config['incr'])
                self.a1text[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x())
            else:
                self.listRectanglesa1[self.box1id].set_width(
                    self.listRectanglesa1[self.box1id].get_width() - float(self.config['move_amount'])/self.sampleRate)
                self.listRectanglesa2[self.box1id].set_width(
                    self.listRectanglesa2[self.box1id].get_width() - float(self.config['move_amount']) / self.config['incr'])
            self.segments[self.box1id][0] = min(self.listRectanglesa1[self.box1id].get_x(),self.listRectanglesa1[self.box1id].get_x() + self.listRectanglesa1[self.box1id].get_width())
            self.segments[self.box1id][1] = max(self.listRectanglesa1[self.box1id].get_x(),self.listRectanglesa1[self.box1id].get_x() + self.listRectanglesa1[self.box1id].get_width())

        elif event.key == 'right':
            self.Box1.set_x(self.Box1.get_x() + self.config['move_amount'])
            self.Box2.set_x(self.Box2.get_x() + float(self.config['move_amount']) / self.config['incr'])
            if self.Boxcol == matplotlib.colors.colorConverter.to_rgba('g'):
                self.listRectanglesa1[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x() + float(self.config['move_amount'])/self.sampleRate)
                self.listRectanglesa1[self.box1id].set_width(
                    self.listRectanglesa1[self.box1id].get_width() - float(self.config['move_amount'])/self.sampleRate)
                self.listRectanglesa2[self.box1id].set_x(
                    self.listRectanglesa2[self.box1id].get_x() + float(self.config['move_amount']) / self.config['incr'])
                self.listRectanglesa2[self.box1id].set_width(
                    self.listRectanglesa2[self.box1id].get_width() - float(self.config['move_amount']) / self.config['incr'])
                self.a1text[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x())
            else:
                self.listRectanglesa1[self.box1id].set_width(
                    self.listRectanglesa1[self.box1id].get_width() + float(self.config['move_amount'])/self.sampleRate)
                self.listRectanglesa2[self.box1id].set_width(
                    self.listRectanglesa2[self.box1id].get_width() + float(self.config['move_amount']) / self.config['incr'])
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
        for i in xrange(len(self.birds1)):
            self.birds1[i].setEnabled(False)
        for i in xrange(len(self.birds2)):
            self.birds2[i].setEnabled(False)
        self.birdList.setEnabled(False)
        self.tbox.setEnabled(False)

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

            print s, s2
            if self.start_stop2 == 0:
                # This is the end of the first segment, draw a red line
                # TODO: At the moment you can't pick these, 'cos it's a pain. Does it matter?
                markstarta3 = self.a3.add_patch(pl.Rectangle((s, np.min(self.audiodata)), self.bar_thickness,
                                                                  np.abs(np.min(self.audiodata)) + np.max(
                                                                      self.audiodata), facecolor='r', edgecolor='None',
                                                                  alpha=0.8)) #,picker=10))
                markstarta4 = self.a4.add_patch(pl.Rectangle((s2, 0), self.bar_thickness/self.config['incr'],
                                                                  self.config['window_width']/2, facecolor='r', edgecolor='None',
                                                                  alpha=0.8)) #,picker=10))
                print s + self.offset_fig1
                self.markstarta1 = self.a1.add_patch(pl.Rectangle(((s + self.offset_fig1)/self.sampleRate, np.min(self.audiodata)), self.linewidtha1 ,
                                                                  np.abs(np.min(self.audiodata)) + np.max(
                                                                      self.audiodata), facecolor='r', edgecolor='None',
                                                                  alpha=0.8))
                self.markstarta2 = self.a2.add_patch(pl.Rectangle((s2 + self.offset_fig1/self.config['incr'], 0), self.linewidtha1 /self.config['incr']*self.sampleRate, self.config['window_width'], facecolor='r', edgecolor='None',alpha=0.8))
                self.listRectanglesb1.append(markstarta3)
                self.listRectanglesb2.append(markstarta4)
                self.end_a2 = s
                self.end_s2 = s2
            else:
                # This is the start of the second segment, draw green lines on fig2, then redraw the boxes on the top figure and save
                b1e = self.a3.add_patch(pl.Rectangle((s, np.min(self.audiodata)), self.bar_thickness, np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='g', edgecolor='None',alpha=0.8,picker=10))
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
                self.segments.append([old[0],(self.end_a2 + self.offset_fig1)/self.sampleRate,old[2]])
                self.segments.append([(s + self.offset_fig1)/self.sampleRate, old[1], old[2]])
                # Delete the old rectangles and text
                self.a1text[self.box1id].remove()
                self.a1text.remove(self.a1text[self.box1id])

                # Add the two new ones
                a1R = self.a1.add_patch(pl.Rectangle((self.segments[-2][0], np.min(self.audiodata)), self.segments[-2][1] - self.segments[-2][0],
                                                     np.abs(np.min(self.audiodata)) + np.max(self.audiodata),
                                                     facecolor='r', alpha=0.5))
                a2R = self.a2.add_patch(pl.Rectangle((self.segments[-2][0]*self.sampleRate/self.config['incr'], 0), (self.segments[-2][1] - self.segments[-2][0])*self.sampleRate/self.config['incr'],
                                                     self.config['window_width']/2,
                                                     facecolor='r', alpha=0.5))
                self.listRectanglesa1.append(a1R)
                self.listRectanglesa2.append(a2R)
                a1t = self.a1.text(self.segments[-2][0], np.min(self.audiodata), self.segments[-2][2])
                self.a1text.append(a1t)

                a3R = self.a1.add_patch(pl.Rectangle((self.segments[-1][0], np.min(self.audiodata)), self.segments[-1][1] - self.segments[-1][0],
                                                     np.abs(np.min(self.audiodata)) + np.max(self.audiodata),
                                                     facecolor='r', alpha=0.5))
                a4R = self.a2.add_patch(pl.Rectangle((self.segments[-1][0]*self.sampleRate/self.config['incr'], 0), (self.segments[-1][1]-self.segments[-1][0])*self.sampleRate/self.config['incr'],
                                                 self.config['window_width']/2,
                                                 facecolor='r', alpha=0.5))
                self.listRectanglesa1.append(a3R)
                self.listRectanglesa2.append(a4R)
                a2t = self.a1.text(self.segments[-1][0], np.min(self.audiodata), self.segments[-1][2])
                self.a1text.append(a2t)

                # Stop the listener
                self.canvas2.mpl_disconnect(self.buttonAdd)

                # Make the addSegments button unselectable
                self.addButton.setEnabled(False)
                self.box1id = None
        self.canvas2.draw()
        self.canvas.draw()
        self.start_stop2 = 1 - self.start_stop2
        # For now at least, stop the canvas2 listener
        self.canvas2.mpl_disconnect(self.picker)

    def playSegment(self):
        # This is the listener for the play button. A very simple wave file player
        sd.play(self.audiodata[self.windowStart:self.windowStart+self.windowSize],self.sampleRate)

    def play(self):
        # This is the listener for the play button. A very simple wave file player
        sd.play(self.audiodata[self.zoomstart:self.zoomend],self.sampleRate)

    def radioBirdsClicked(self):
        # Listener for when the user selects a radio button
        # Update the text and store the data
        for button in self.birds1+self.birds2:
            if button.isChecked():
                if button.text()=="Other":
                    self.birdList.setEnabled(True)
                else:
                    self.birdList.setEnabled(False)
                    self.segments[self.box1id][2] = str(button.text())
                    self.a1text[self.box1id].set_text(str(button.text()))
                    if self.segments[self.box1id][2] != "Don't Know":
                        facecolour = 'b'
                    else:
                        facecolour = 'r'
                    self.listRectanglesa1[self.box1id].set_facecolor(facecolour)
                    self.listRectanglesa2[self.box1id].set_facecolor(facecolour)
                    self.topBoxCol = self.listRectanglesa1[self.box1id].get_facecolor()
                    self.canvas.draw()

    def listBirdsClicked(self, item):
        # Listener for the listbox of birds
        if (item.text() == "Other"):
            self.tbox.setEnabled(True)
        else:
            # Save the entry
            self.segments[self.box1id][2] = str(item.text())
            self.a1text[self.box1id].set_text(str(item.text()))
            self.listRectanglesa1[self.box1id].set_facecolor('b')
            self.listRectanglesa2[self.box1id].set_facecolor('b')
            self.topBoxCol = self.listRectanglesa1[self.box1id].get_facecolor()
            self.canvas.draw()

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
        self.segments[self.box1id][2] = str(self.tbox.text())
        self.a1text[self.box1id].set_text(str(self.tbox.text()))
        self.listRectanglesa1[self.box1id].set_facecolor('b')
        self.listRectanglesa2[self.box1id].set_facecolor('b')
        self.topBoxCol = self.listRectanglesa1[self.box1id].get_facecolor()
        self.canvas.draw()
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
            print "Saving segments to "+self.filename
            if isinstance(self.filename, str):
                file = open(self.filename + '.data', 'w')
            else:
                file = open(str(self.filename) + '.data', 'w')
            json.dump(self.segments,file)

    def writefile(self):
        # Assuming that you want to save denoised data at some point, this will do it
        # Other players need them to be 16 bit, which is this magic number
        self.audiodata *= 32768.0
        self.audiodata = self.audiodata.astype('int16')
        filename = self.filename[:-4] + '_clear' + self.filename[-4:]
        wavfile.write(filename,self.sampleRate, self.audiodata)

    def quit(self):
        # Listener for the quit button
        self.saveSegments()
        json.dump(self.config, open(self.configfile, 'wb'))
        QApplication.quit()

    def splitFile5mins(self, name):
        # Nirosha wants to split files that are long (15 mins) into 5 min segments
        # Could be used when loading long files :)
        self.sampleRate, self.audiodata = wavfile.read(name)
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

# Start the application
app = QApplication(sys.argv)
form = Interface(configfile='AviaNZconfig.txt')
form.show()
app.exec_()
