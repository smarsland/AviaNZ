# Manual Segmentation.py
#
# This is a modification of the AviaNZ interface to perform segmentation
# as simply as possible and collect data about usage.
# Originally written for Kim's experiment

# Version 0.2 27/8/16
# Author: Stephen Marsland

# Description
# A folder is specified. The program randomly selects 5 of these that this user will process
# It makes a new folder to store the data, which will be the segments made by this user, together with
# various meta-data

# Installation
# Install anaconda (Python 2.7)  (http://continuum.io/downloads)
# Install Qt (conda install -c anaconda pyqt=4.11.4 from a command line)
# Install sounddevice (pip install sounddevice --user)
# at command line: python manualsegmentation.py
# Hopefully that's it!

import sys, os, json, time
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from scipy.io import wavfile
import numpy as np
import pylab as pl
from threading import Thread

import datetime
#import pyaudio
import sounddevice as sd
#from pysoundcard import Stream, continue_flag

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import SignalProc

# ==============
# TODO
# Put all the questions for the surveys in
# 1 page user manual
# ** Testing
# How to deploy it for her? Docker?


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


        #self.firstFile = 'male1.wav' # 'kiwi.wav'#'
        #self.dirName = self.config['dirpath']
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

        self.startQnDialog()
        self.timestamps=[]
        self.genData()

    def genData(self):
        now = datetime.datetime.now()
        self.dirName = str("{:0>2d}".format(now.day))+str("{:0>2d}".format(now.month))+str("{:0>2d}".format(now.hour))+str("{:0>2d}".format(now.minute))
        if os.path.isdir(self.dirName):
            self.dirName = self.dirName+'a'
        os.mkdir(self.dirName)

        self.dir = 'Sound Files'
        dirlist = QDir('Sound Files/')
        # 2s are because . and .. are there
        order = np.random.permutation(len(dirlist)-2)+2
        self.filelist = []
        for i in range(5):
            self.filelist.append(str(dirlist[order[i]]))
        print self.filelist
        self.inFile = 0
        self.loadFile(self.filelist[self.inFile])
        self.timestamps.append(["Loading ",self.filelist[self.inFile],[now.hour,now.minute,now.second,now.microsecond]])

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

            'UseConfigMenu': False,
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
        #self.canvas3.mpl_connect('motion_notify_event', self.fig3Drag)

        # This is the top figure
        self.fig = Figure(dpi=self.config['dpi'])
        self.fig.set_size_inches(22.0, 4.0, forward=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.frame)
        # This is for the click for start and end bars and other interactions on fig1
        self.canvas.mpl_connect('button_press_event', self.fig1Click)

        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()

        # Whether or not user is drawing boxes around song in the spectrogram
        #self.dragRectangles = QCheckBox('Drag boxes in spectrogram')
        #self.dragRectangles.stateChanged[int].connect(self.dragRectanglesCheck)

        # The buttons on the right hand side, and also the spinbox for the window width
        self.leftBtn = QToolButton()
        self.leftBtn.setArrowType(Qt.LeftArrow)
        self.connect(self.leftBtn,SIGNAL('clicked()'),self.moveLeft)
        self.rightBtn = QToolButton()
        self.rightBtn.setArrowType(Qt.RightArrow)
        self.connect(self.rightBtn, SIGNAL('clicked()'), self.moveRight)
        # TODO: what else should be there?
        #playButton1 = QPushButton(QIcon(":/Resources/play.svg"),"&Play Window")
        playButton1 = QPushButton("&Play/Pause")
        self.connect(playButton1, SIGNAL('clicked()'), self.playSegment)
        resetButton1 = QPushButton("&Reset")
        self.connect(resetButton1, SIGNAL('clicked()'), self.resetSegment)
        self.quitButton = QPushButton("&Quit")
        self.connect(self.quitButton, SIGNAL('clicked()'), self.quit)
        deleteButton = QPushButton("&Delete Current Segment")
        self.connect(deleteButton, SIGNAL('clicked()'), self.deleteSegment)
        deleteAllButton = QPushButton("&Delete All Segments")
        self.connect(deleteAllButton, SIGNAL('clicked()'), self.deleteAll)
        self.nextButton = QPushButton("Next File")
        self.connect(self.nextButton, SIGNAL('clicked()'), self.next)
        self.previousButton = QPushButton("&Previous File")
        self.connect(self.previousButton, SIGNAL('clicked()'), self.previous)
        self.previousButton.setEnabled(False)

        self.widthWindow = QDoubleSpinBox()
        self.widthWindow.setRange(0.5,900.0)
        self.widthWindow.setSingleStep(1.0)
        self.widthWindow.setDecimals(2)
        self.widthWindow.setValue(self.config['windowWidth'])
        self.widthWindow.valueChanged[float].connect(self.changeWidth)

        # Do we actually want labels?
        self.birds1 = []
        for item in self.config['BirdButtons1']:
            self.birds1.append(QRadioButton(item))
        self.birds2 = []
        for item in self.config['BirdButtons2']:
            self.birds2.append(QRadioButton(item))

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

        # A context menu to select birds
        if self.config['UseConfigMenu']:
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

        #b = self.menuBirdList.addAction('b')
        #self.connect(b, SIGNAL("triggered()"),self.birdSelected)

        # This is the set of layout instructions. It's a bit of a mess, but basically looks like this:
        # { vbox1 }  -> hbox1
        #
        # {vbox0 | birds1layout birds2layout | birdListLayout | vbox3 | vboxbuttons} -> selectorLayout

        vbox4a = QVBoxLayout()
        vbox4a.addWidget(QLabel('Visible window width (seconds)'))
        vbox4a.addWidget(self.widthWindow)

        hbox4a = QHBoxLayout()
        hbox4a.addWidget(QLabel('Slide top box to move through recording, click to start and end a segment, click on segment to edit or label. Right click to interleave.'))
        hbox4a.addWidget(playButton1)
        hbox4a.addWidget(resetButton1)
        #hbox4a.addWidget(self.dragRectangles)

        vbox4b = QVBoxLayout()
        vbox4b.addLayout(hbox4a)

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

        vboxbuttons = QVBoxLayout()
        for w in [deleteButton,deleteAllButton,self.quitButton,self.previousButton, self.nextButton]:
        #for w in [deleteButton, deleteAllButton, self.previousButton, self.nextButton]:
            vboxbuttons.addWidget(w)
            #vboxbuttons.setAlignment(w, Qt.AlignHCenter)
        self.quitButton.setEnabled(False)

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

    def previous(self):
        self.inFile -= 1
        self.nextButton.setEnabled(True)
        if self.inFile == 0:
            self.previousButton.setEnabled(False)
        print self.filelist[self.inFile]
        self.loadFile(self.filelist[self.inFile])
        now = datetime.datetime.now()
        self.timestamps.append(["Loading ",self.filelist[self.inFile],[now.hour,now.minute,now.second,now.microsecond]])


    def next(self):
        self.inFile += 1
        self.previousButton.setEnabled(True)
        if self.inFile == 4:
            self.nextButton.setEnabled(False)
            self.quitButton.setEnabled(True)
        print self.filelist[self.inFile]
        self.loadFile(self.filelist[self.inFile])
        now = datetime.datetime.now()
        self.timestamps.append(["Loading ",self.filelist[self.inFile],[now.hour,now.minute,now.second,now.microsecond]])

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
        self.playPosition = self.windowStart
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

        self.playbar1 = None
        self.isPlaying = False

    def loadFile(self,name):
        # This does the work of loading a file
        # One magic constant, which normalises the data
        # TODO: currently just takes the first channel of 2 -> what to do with the other?
        #if len(self.segments)>0:
        #    self.saveSegments()
        #self.segments = []
        if self.previousFile is not None:
            if self.segments != []:
                self.saveSegments()
        self.resetStorageArrays()

        if isinstance(name,str):
            self.filename = name
        else:
            self.filename = str(name)
        self.previousFile = self.filename

        self.sampleRate, self.audiodata = wavfile.read(self.dir+'/'+self.filename)

        #self.wf = wave.open(self.dir+'/'+self.filename, 'rb')
        #self.sampleRate = self.wf.getframerate()

        # read data
        #self.audiodata = np.zeros(self.wf.getnframes())
        #fmt = '<' + 'h' * self.wf.getnchannels()

        #for i in range(self.wf.getnframes()):
        #    frame = self.wf.readframes(1)
        #    self.audiodata[i] = struct.unpack(fmt, frame)[0]

        #self.audiodata = self.audiodata.astype('float')/32768.8

        if self.audiodata.dtype is not 'float':
            self.audiodata = self.audiodata.astype('float') / 32768.0
        if np.shape(np.shape(self.audiodata))[0]>1:
            self.audiodata = self.audiodata[:,0]
        self.datalength = np.shape(self.audiodata)[0]
        print("Length of file is ",len(self.audiodata),float(self.datalength)/self.sampleRate)

        # Create an instance of the Signal Processing class
        if not hasattr(self,'sp'):
            self.sp = SignalProc.SignalProc(self.audiodata, self.sampleRate,self.config['window_width'],self.config['incr'])
        # Create an instance of the pyaudio player
        #if not hasattr(self,'player'):
        #    # instantiate PyAudio
        #    self.player = pyaudio.PyAudio()
        # open stream 
        #self.stream = self.player.open(format=self.player.get_format_from_width(self.wf.getsampwidth()), channels=self.wf.getnchannels(), rate=self.wf.getframerate(), output=True)
        # Get the data for the spectrogram
        self.sg = self.sp.spectrogram(self.audiodata)

        # Load any previous segments stored
        if os.path.isfile(self.dirName+'/'+self.filename+'.data'):
            file = open(self.dirName+'/'+self.filename+'.data', 'r')
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


    #def dragRectanglesCheck(self,check):
        # The checkbox that says if the user is dragging rectangles or clicking on the spectrogram has changed state
        #if self.dragRectangles.isChecked():
            #print "Listeners for drag Enabled"
            #self.fig1motion = self.canvas.mpl_connect('motion_notify_event', self.fig1Drag)
            #self.fig1endmotion = self.canvas.mpl_connect('button_release_event', self.fig1DragEnd)
        #else:
            #print "Listeners for drag Disabled"
            #self.canvas.mpl_disconnect(self.fig1motion)
            #self.canvas.mpl_disconnect(self.fig1endmotion)

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

                    if self.config['UseConfigMenu']:
                        bird = self.menuBird2.addAction(text)
                        receiver = lambda birdname=text: self.birdSelected(birdname)
                        self.connect(bird, SIGNAL("triggered()"), receiver)
                        self.menuBird2.addAction(bird)

    def activateRadioButtons(self):
        found = False
        for i in range(len(self.birds1)):
            self.birds1[i].setEnabled(True)
            if str(self.a1text[self.box1id].get_text()) == self.birds1[i].text():
                self.birds1[i].setChecked(True)
                found = True
        for i in range(len(self.birds2)):
            self.birds2[i].setEnabled(True)
            if str(self.a1text[self.box1id].get_text()) == self.birds2[i].text():
                self.birds2[i].setChecked(True)
                found = True

        if not found:
            # Select 'Other' in radio buttons (last item) and activate the listwidget
            self.birds2[-1].setChecked(True)
            self.birdList.setEnabled(True)
            items = self.birdList.findItems(str(self.a1text[self.box1id].get_text()), Qt.MatchExactly)
            for item in items:
                self.birdList.setCurrentItem(item)

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
        if box1id>-1 and event.button==1:
            self.box1id = box1id
            self.topBoxCol = self.listRectanglesa1[box1id].get_facecolor()
            self.listRectanglesa1[box1id].set_facecolor('green')
            self.listRectanglesa2[box1id].set_facecolor('green')
            self.zoomstart = self.listRectanglesa1[box1id].get_x()
            self.zoomend = self.listRectanglesa1[box1id].get_x()+self.listRectanglesa1[box1id].get_width()
            self.drawFig2()

            if self.config['UseConfigMenu']:
                # Put the context menu at the right point. The event.x and y are in pixels relative to the bottom left-hand corner of canvas (self.fig)
                # Need to be converted into pixels relative to top left-hand corner of the window!
                self.menuBirdList.popup(QPoint(event.x + event.canvas.x() + self.frame.x() + self.x(), event.canvas.height() - event.y + event.canvas.y() + self.frame.y() + self.y()))
            # Activate the radio buttons for labelling, selecting one from label if necessary
            self.activateRadioButtons()

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

            #if aind != a1ind: #and self.dragRectangles.isChecked():
                # Use is selecting a region **on the spectrogram only**
                # The region is drawn as full size on the amplitude plot
                #if event.xdata > 0 and event.xdata < float(self.datalength) / self.config['incr']:
                #    self.recta2 = self.a2.add_patch(pl.Rectangle((event.xdata, event.ydata), 0.0, 0.0, alpha=1, facecolor='none'))
            #else:
            s = event.xdata
            if aind != a1ind:
                s = s*self.config['incr']/self.sampleRate
            if s>0 and s<float(self.datalength)/self.sampleRate:
                if not self.start_stop:
                    # This is the start of a segment, draw a green line
                    self.markstarta1 = self.a1.add_patch(pl.Rectangle((s, np.min(self.audiodata)), self.linewidtha1, self.plotheight, facecolor='g', edgecolor='None',alpha=0.8))
                    self.markstarta2 = self.a2.add_patch(pl.Rectangle((s*self.sampleRate / self.config['incr'] , 0), self.linewidtha1/self.config['incr']*self.sampleRate, self.config['window_width']/2, facecolor='g', edgecolor='None',alpha=0.8))
                    self.start_a = s
                    now = datetime.datetime.now()
                    self.then = [now.hour,now.minute,now.second,now.microsecond]
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
                    now = datetime.datetime.now()
                    self.segments.append([start,max(self.start_a,s),0.0,self.sampleRate/2.,'None',self.then,[now.hour,now.minute,now.second,now.microsecond]])
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
                    if self.config['UseConfigMenu']:
                        self.menuBirdList.popup(QPoint(event.x + event.canvas.x() + self.frame.x() + self.x(), event.canvas.height() - event.y + event.canvas.y() + self.frame.y() + self.y()))
                    # Activate the radio buttons for labelling, selecting one from label if necessary
                    self.activateRadioButtons()
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

    def showMenu(self,pos):
        self.menuBirdList.popup(self.mapToGlobal(pos))

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
            self.playPosition = self.windowStart
            self.focusRegionSelected = False
            self.focusRegion.set_facecolor('r')
            self.updateWindow()

    def fig3Drag(self,event):
        if event.inaxes is None:
            return
        if event.button!=1:
            return
        self.windowStart = event.xdata/self.sampleRate*self.config['incr']
        self.playPosition = self.windowStart
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
        self.playPosition = self.windowStart
        self.updateWindow()

    def moveRight(self):
        # When the right button is pressed (on the top right of the screen, move everything along
        self.windowStart = min(float(self.datalength)/self.sampleRate-self.windowSize,self.windowStart+self.windowSize*0.9)
        self.playPosition = self.windowStart
        self.updateWindow()

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
                now = datetime.datetime.now()
                self.segments.append([old[0],(self.end_a2 + self.offset_fig1)/self.sampleRate,old[2],old[3],old[4],old[5],[now.hour,now.minute,now.second,now.microsecond]])
                self.segments.append([(s + self.offset_fig1)/self.sampleRate, old[1], old[2],old[3],old[4],[now.hour,now.minute,now.second,now.microsecond],old[6]])
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

    def runthread(self,start,end,sampleRate):
        sd.read(self.audiodata[start:end],sampleRate)

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
        self.canvas.draw()
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
             self.canvas.draw()
             current_time = now

    def resetSegment(self):
        self.playPosition = self.windowStart
        if self.playbar1 is not None:
            self.playbar1.remove() 
            self.playbar2.remove() 
            self.playbar1 = None
            self.canvas.draw()

    def playSegment(self):
        # This is the listener for the play button. A very simple wave file player
        # Move a marker through in real time?!
        sd.play(self.audiodata[int(self.windowStart*self.sampleRate):int((self.windowStart+self.windowSize)*self.sampleRate)], self.sampleRate,blocking=True)
        # if not self.isPlaying:
        #     self.isPlaying = True
        #
        #     #t = Thread(target=self.runthread, args=(int(self.playPosition*self.sampleRate),int(self.playPosition*self.sampleRate+self.windowSize*self.sampleRate),self.sampleRate))
        #     #t.start()
        #
        #     end_time = self.windowStart + self.windowSize
        #     start_time = datetime.datetime.now()
        #     print "Start: ",start_time
        #     self.stopRequest = False
        #     #t2 = Thread(target=self.runthread2, args=(start_time,end_time))
        #     #t2.start()
        # else:
        #     print self.playbar1.get_x()
        #     self.playPosition = self.playbar1.get_x()
        #     self.stopRequest = True
        #     sd.stop()
        #     self.isPlaying = False

    def play(self):
        # This is the listener for the play button. 
        #playThread().start()
        sd.play(self.audiodata[self.zoomstart:self.zoomend], self.sampleRate,blocking=True)

        #t = Thread(target=run, args=(self.zoomstart,self.zoomend))
        #t.start()


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
            self.filename = str(self.filename)
            print("Saving segments for "+self.filename)
            if isinstance(self.filename, str):
                file1 = open(self.dirName+'/'+self.filename + '.data', 'w')
            else:
                file1 = open(self.dirName+'/'+str(self.filename) + '.data', 'w')
            json.dump(self.segments, file1)

    def closeEvent(self, event):
        # This is to catch the user closing the window by clicking the Close button instead of quitting
        self.quit()

    def quit(self):
        # Listener for the quit button
        print("Quitting")
        now = datetime.datetime.now()
        self.timestamps.append(["Quitting ","",[now.hour,now.minute,now.second,now.microsecond]])

        # close PyAudio 
        #self.stream.close()
        #self.player.terminate()
        #self.wf.close()

        self.endQnDialog()

        self.saveSegments()
        if self.saveConfig == True:
            print "Saving config file"
            json.dump(self.config, open(self.configfile, 'wb'))
        file2 = open(self.dirName + '/meta.txt', 'w')
        json.dump([self.filelist, self.timestamps, self.ansS, self.ansE], file2)
        QApplication.quit()

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


    def startQnDialog(self):
        print "here"
        ans = startQuestions.startQns()
        print ans

    def startQnDialog(self):
        self.ansS = startQuestions.startQs()
        #self.startDialog = startQuestions()
        #self.startDialog.setModal(True)
        #self.startDialog.show()
        #self.startDialog.activate.clicked.connect(self.getAnswersStart)

    def getAnswersStart(self):
        self.ansS = self.startDialog.getAnswers()
        print "s",self.ansS
        self.startDialog.destroy()

    def endQnDialog(self):
        self.ansE = endQuestions.endQs()
        #self.endDialog = endQuestions()
        #self.endDialog.setModal(True)
        #self.endDialog.show()
        #self.endDialog.activate.clicked.connect(self.getAnswersEnd)

    def getAnswersEnd(self):
        self.ansE = self.endDialog.getAnswers()
        print "e",self.ansE
        self.endDialog.destroy()

class startQuestions(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Initial Survey')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.resize(800,700)
        #self.setMinimumSize()
        layout = QVBoxLayout(self)
        layout0 = QHBoxLayout(self)

        layout1 = QVBoxLayout(self)
        layout1.addWidget(QLabel('How much knowledge do you have about birds?'))
        self.Q1 = QButtonGroup()
        self.Q1.setExclusive(True)
        self.Q1a = QRadioButton('None')
        self.Q1b = QRadioButton('Some')
        self.Q1c = QRadioButton('Moderate')
        self.Q1d = QRadioButton('A lot')
        layout1.addWidget(self.Q1a)
        layout1.addWidget(self.Q1b)
        layout1.addWidget(self.Q1c)
        layout1.addWidget(self.Q1d)
        self.Q1.addButton(self.Q1a)
        self.Q1.addButton(self.Q1b)
        self.Q1.addButton(self.Q1c)
        self.Q1.addButton(self.Q1d)

        layout1.addWidget(QLabel('How much experience do you have in using audio software?'))
        self.Q2 = QButtonGroup()
        self.Q2.setExclusive(True)
        self.Q2a = QRadioButton('None')
        self.Q2b = QRadioButton('Some')
        self.Q2c = QRadioButton('Moderate')
        self.Q2d = QRadioButton('A lot')
        layout1.addWidget(self.Q2a)
        layout1.addWidget(self.Q2b)
        layout1.addWidget(self.Q2c)
        layout1.addWidget(self.Q2d)
        self.Q2.addButton(self.Q2a)
        self.Q2.addButton(self.Q2b)
        self.Q2.addButton(self.Q2c)
        self.Q2.addButton(self.Q2d)

        layout1.addWidget(QLabel('Which programs have you used?'))
        self.Q3 = QLineEdit(self)
        layout1.addWidget(self.Q3)

        layout1.addWidget(QLabel('Have you ever used audio software for bird song?')) 
        self.Q4 = QButtonGroup()
        self.Q4.setExclusive(True)
        self.Q4a = QRadioButton('Yes')
        self.Q4b = QRadioButton('No')
        layout1.addWidget(self.Q4a)
        layout1.addWidget(self.Q4b)
        self.Q4.addButton(self.Q4a)
        self.Q4.addButton(self.Q4a)

        layout1.addWidget(QLabel('If yes, which programs have you used?'))
        self.Q5 = QLineEdit(self)
        layout1.addWidget(self.Q5)

        layout2 = QVBoxLayout(self)
        layout2.addWidget(QLabel('How old are you?'))
        self.Q6 = QButtonGroup()
        self.Q6.setExclusive(True)
        self.Q6a = QRadioButton('Under 25')
        self.Q6b = QRadioButton('25--50')
        self.Q6c = QRadioButton('Over 50')
        layout2.addWidget(self.Q6a)
        layout2.addWidget(self.Q6b)
        layout2.addWidget(self.Q6c)
        self.Q6.addButton(self.Q6a)
        self.Q6.addButton(self.Q6b)
        self.Q6.addButton(self.Q6c)

        layout2.addWidget(QLabel('Do you have any hearing impediments?'))
        self.Q7 = QButtonGroup()
        self.Q7.setExclusive(True)
        self.Q7a = QRadioButton('No')
        self.Q7b = QRadioButton('Yes')
        layout2.addWidget(self.Q7a)
        layout2.addWidget(self.Q7b)
        self.Q7.addButton(self.Q7a)
        self.Q7.addButton(self.Q7b)

        layout2.addWidget(QLabel('What is your gender?'))
        self.Q8 = QButtonGroup()
        self.Q8.setExclusive(True)
        self.Q8a = QRadioButton('Male')
        self.Q8b = QRadioButton('Female')
        self.Q8c = QRadioButton('Not prepared to say')
        layout2.addWidget(self.Q8a)
        layout2.addWidget(self.Q8b)
        layout2.addWidget(self.Q8c)
        self.Q8.addButton(self.Q8a)
        self.Q8.addButton(self.Q8b)
        self.Q8.addButton(self.Q8c)

        button = QDialogButtonBox(
            QDialogButtonBox.Ok,
            Qt.Horizontal, self)
        button.accepted.connect(self.accept)

        layout0.addLayout(layout1)
        layout0.addLayout(layout2)
        layout.addLayout(layout0)
        layout.addWidget(button)

    def getAnswers(self):
        return [self.Q1a.isChecked(),self.Q1b.isChecked(),self.Q1c.isChecked(),self.Q1d.isChecked(),self.Q2a.isChecked(),self.Q2b.isChecked(),self.Q2c.isChecked(),self.Q2d.isChecked(),str(self.Q3.text()),self.Q4a.isChecked(),self.Q4b.isChecked(),str(self.Q5.text()),self.Q6a.isChecked(),self.Q6b.isChecked(),self.Q6c.isChecked(),self.Q7a.isChecked(),self.Q7b.isChecked(),self.Q8a.isChecked(),self.Q8b.isChecked(),self.Q8c.isChecked(),]

    # static method to create the dialog and return the data
    @staticmethod
    def startQs(parent=None):
        dialog = startQuestions()
        dialog.exec_()
        return (dialog.getAnswers())

class endQuestions(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Final Survey')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.resize(800,700)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel('How difficult was this to do?'))
        self.aQ1a = QRadioButton('Easy')
        self.aQ1b = QRadioButton('Medium')
        self.aQ1c = QRadioButton('Hard')
        layout.addWidget(self.aQ1a)
        layout.addWidget(self.aQ1b)
        layout.addWidget(self.aQ1c)

        layout.addWidget(QLabel('What did you find easy?'))
        self.aQ2 = QTextEdit(self)
        layout.addWidget(self.aQ2)

        layout.addWidget(QLabel('What did you find difficult?'))
        self.aQ3 = QTextEdit(self)
        layout.addWidget(self.aQ3)

        layout.addWidget(QLabel('Do you have any suggestions for improvements?'))
        self.aQ4 = QTextEdit(self)
        layout.addWidget(self.aQ4)

        button = QDialogButtonBox(
            QDialogButtonBox.Ok,
            Qt.Horizontal, self)
        button.accepted.connect(self.accept)

        layout.addWidget(button)

    def getAnswers(self):
        return [self.aQ1a.isChecked(), self.aQ1b.isChecked(), self.aQ1c.isChecked(),str(self.aQ2.toPlainText()),str(self.aQ3.toPlainText()),str(self.aQ4.toPlainText())]

    # static method to create the dialog and return the data
    @staticmethod
    def endQs(parent=None):
        dialog = endQuestions()
        dialog.exec_()
        return (dialog.getAnswers())

# Start the application
app = QApplication(sys.argv)
form = Interface(configfile='AviaNZconfig.txt')
form.show()
app.exec_()
