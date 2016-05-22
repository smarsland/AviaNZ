# Interface,py
#
# This is currently the base class for the AviaNZ interface
# It's fairly simplistic, but hopefully works
# Version 0.3 18/5/16
# Author: Stephen Marsland

import sys, os, glob, json
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from scipy.io import wavfile
from scipy.fftpack import fft
import numpy as np
import pylab as pl

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
# implement the default mpl key bindings
# from matplotlib.backend_bases import key_press_handler

import Denoise
# ==============
# TODO

# Finish manual segmentation
#   In Zoom window, allow click to move segmentation ends ->
#       get the size of bars, move right
# Error check -> can't move green to right of red

# Denoise the ruru call??

# Save additions to the Listwidget make a config dictionary for everything
# Given files > 5 mins, split them into 5 mins versions anyway (code is there, make it part of workflow)
# List of files in the box on the left -> directories, .., etc.

# Tidy the code (change denoising to signalproc, move spectrogram in there, etc.)
# Clear interface to plug in segmenters, labellers, etc.

# Add other windowing functions

# Changes to interface?
# Make things that are 'correct' blue instead of red?
# Print text above parts that have been recognised -> put this inbetween the two graphs
# Does it need play and pause buttons? Plus a marker for where up to in playback -> needs another player
# Related: should you be able to select place to play from in fig1?
# Automatic segmentation -> power, wavelets, ...
# Start to consider some learning interfaces
# Can I process spectrogram directly?
# Get suggestions from the others
# ===============

def spectrogram(t):
    if t is None:
        print ("Error")

    window_width = 256
    incr = 128
    # This is the Hanning window
    hanning = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_width) / (window_width + 1)))

    sg = np.zeros((window_width / 2, np.ceil(len(t) / incr)))
    counter = 1

    for start in range(0, len(t) - window_width, incr):
        window = hanning * t[start:start + window_width]
        ft = fft(window)
        ft = ft * np.conj(ft)
        sg[:, counter] = np.real(ft[window_width / 2:])
        counter += 1
    # Note that the last little bit (up to window_width) is lost. Can't just add it in since there are fewer points

    sg = 10.0 * np.log10(sg)
    return sg

class Interface(QMainWindow):


    def __init__(self,root=None):

        self.root = root

        # This is a flag to say if the next thing that they click on should be a start or a stop for segmentation
        # Second variables hold a start point for the amplitude and spectrogram respectively
        self.start_stop = 0
        self.start_stop2 = 0
        self.start_a = 0
        self.start_s = 0

        self.Boxx = None

        # Params for spectrogram
        self.window_width = 256
        self.incr = 128

        # Params for the width of the lines to draw for segmentation in the two plots
        self.linewidtha1 = 100
        self.linewidtha2 = self.linewidtha1/self.incr
        self.dirpath = '.'

        # Params for amount to plot in window
        self.windowWidth = 2.0 # width in seconds of the main representation
        self.windowStart = 0

        # This hold the actual data for now, and also the rectangular patches
        self.segments = []
        self.listRectanglesa1 = []
        self.listRectanglesa2 = []
        self.listRectanglesb1 = []
        self.listRectanglesb2 = []
        self.a1text = []
        self.box1id = None
        self.buttonID = None

        QMainWindow.__init__(self, root)
        self.setWindowTitle('AviaNZ')

        self.createMenu()
        self.createFrame()

        # Make life easier for now: preload a birdsong
        #self.loadFile('../Birdsong/more1.wav')
        #self.loadFile('male1.wav')
        #self.loadFile('ST0026_1.wav')
        self.loadFile('ruru.wav')

        #self.sampleRate, self.audiodata = wavfile.read('kiwi.wav')
        #self.sampleRate, self.audiodata = wavfile.read('/Users/srmarsla/Students/Nirosha/bittern/ST0026.wav')

    def createMenu(self):
        self.fileMenu = self.menuBar().addMenu("&File")

        openFileAction = QAction("&Open wave file", self)
        self.connect(openFileAction,SIGNAL("triggered()"),self.openFile)
        self.fileMenu.addAction(openFileAction)

        # This seems to only work if it is there twice ?!
        quitAction = QAction("&Quit", self)
        self.connect(quitAction,SIGNAL("triggered()"),self.quit)
        self.fileMenu.addAction(quitAction)

        quitAction = QAction("&Quit", self)
        self.connect(quitAction, SIGNAL("triggered()"), self.quit)
        self.fileMenu.addAction(quitAction)

    def createFrame(self):

        # These are the contrast parameters for the spectrogram
        self.colourStart = 0.6
        self.colourEnd = 1.0
        self.defineColourmap()

        self.frame = QWidget()
        self.dpi = 100

        self.fig = Figure((8.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.frame)

        self.canvas.mpl_connect('button_press_event', self.onClick)

        #self.mpl_toolbar = NavigationToolbar(self.canvas, self.frame)

        # Needs a bit of sorting -> subdirectories, etc.
        listFiles = QListWidget(self)
        listOfFiles = []
        #listOfFiles.extend('..')
        for extension in ['wav','WAV']:
            pattern = os.path.join(self.dirpath,'*.%s' % extension)
            listOfFiles.extend(glob.glob(pattern))
        for file in listOfFiles:
            item = QListWidgetItem(listFiles)
            item.setText(file)
        listFiles.connect(listFiles, SIGNAL('itemClicked(QListWidgetItem*)'), self.listLoadFile)

        #self.playButton1 = QPushButton("&Play")
        #self.connect(self.playButton1, SIGNAL('clicked()'), self.play)
        quitButton = QPushButton("&Quit")
        self.connect(quitButton, SIGNAL('clicked()'), self.quit)
        segmentButton = QPushButton("&Segment")
        self.connect(segmentButton, SIGNAL('clicked()'), self.segment)
        denoiseButton = QPushButton("&Denoise")
        self.connect(denoiseButton, SIGNAL('clicked()'), self.denoise)
        recogniseButton = QPushButton("&Recognise")
        self.connect(recogniseButton, SIGNAL('clicked()'), self.recognise)
        widthWindow = QDoubleSpinBox()
        widthWindow.setRange(0.5,20.0)
        widthWindow.setSingleStep(1.0)
        widthWindow.setDecimals(2)
        widthWindow.setValue(2.0)
        widthWindow.valueChanged[float ].connect(self.changeWidth)

        sld = QSlider(Qt.Horizontal, self)
        sld.setFocusPolicy(Qt.NoFocus)
        sld.valueChanged[int].connect(self.sliderMoved)

        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.canvas)
        vbox1.addWidget(QLabel('Slide to move through recording, click to start and end a segment, click on segment to edit or label'))
        vbox1.addWidget(sld)
        #vbox1.addWidget(self.mpl_toolbar)

        vbox0a = QVBoxLayout()
        vbox0a.addWidget(QLabel('Select another file to work on here'))
        vbox0a.addWidget(listFiles)

        vboxbuttons = QVBoxLayout()
        for w in [denoiseButton, segmentButton, quitButton]:
            vboxbuttons.addWidget(w)
            #vboxbuttons.setAlignment(w, Qt.AlignHCenter)
        vboxbuttons.addWidget(QLabel('Visible window width (seconds)'))
        vboxbuttons.addWidget(widthWindow)

        hbox1 = QHBoxLayout()
        hbox1.addLayout(vbox0a)
        hbox1.addLayout(vbox1)
        hbox1.addLayout(vboxbuttons)

        selectorLayout = QHBoxLayout()
        # Create an array of radio buttons
        self.birds1 = [QRadioButton("Female Kiwi"), QRadioButton("Male Kiwi"), QRadioButton("Ruru"), QRadioButton("Hihi"),
                  QRadioButton("Not Bird"), QRadioButton("Don't Know"), QRadioButton("Other")]
        birds1Layout = QVBoxLayout()
        birdButtonGroup = QButtonGroup()

        for i in xrange(len(self.birds1)):
            birds1Layout.addWidget(self.birds1[i])
            birdButtonGroup.addButton(self.birds1[i], i)
            self.birds1[i].setEnabled(False)
            self.connect(self.birds1[i], SIGNAL("clicked()"), self.birds1Clicked)
        selectorLayout.addLayout(birds1Layout)

        self.birdList = QListWidget(self)
        self.items = ['Other','a', 'b', 'c', 'd', 'e', 'f', 'g']
        for item in self.items:
            self.birdList.addItem(item)
        self.connect(self.birdList, SIGNAL("itemClicked(QListWidgetItem*)"), self.birds2Clicked)
        birdListLayout = QVBoxLayout()
        birdListLayout.addWidget(self.birdList)
        selectorLayout.addLayout(birdListLayout)
        birdListLayout.addWidget(QLabel('If bird is not in list, select Other, and type into box, pressing Return at end'))
        self.birdList.setEnabled(False)

        self.tbox = QLineEdit(self)
        birdListLayout.addWidget(self.tbox)
        self.connect(self.tbox, SIGNAL('editingFinished()'), self.getTextEntry)
        self.tbox.setEnabled(False)

        vbox3 = QVBoxLayout()
        self.fig2 = Figure((4.0, 4.0), dpi=self.dpi)
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setParent(self.frame)
        self.canvas2.setFocusPolicy(Qt.ClickFocus)
        self.canvas2.setFocus()
        hbox4 = QHBoxLayout()
        playButton2 = QPushButton("&Play")
        self.connect(playButton2, SIGNAL('clicked()'), self.play)
        addButton = QPushButton("&Add segment")
        self.connect(addButton, SIGNAL('clicked()'), self.addSegmentClick)

        for w in [playButton2,addButton]:
            hbox4.addWidget(w)
            hbox4.setAlignment(w, Qt.AlignVCenter)
        vbox3.addWidget(self.canvas2)
        vbox3.addWidget(QLabel('Click on a start/end to select, click again to move'))
        vbox3.addWidget(QLabel('Or use arrow keys. Press Backspace to delete'))
        vbox3.addLayout(hbox4)
        selectorLayout.addLayout(vbox3)

        vbox2 = QVBoxLayout()
        vbox2.addLayout(hbox1)
        vbox2.addLayout(selectorLayout)

        self.frame.setLayout(vbox2)
        self.setCentralWidget(self.frame)

    def sliderMoved(self,value):
        # Get the scaling sorted for the slider
        totalRange = self.datamax - self.windowSize
        self.a1.set_xlim(totalRange/100.*value, totalRange/100.*value+self.windowSize)
        self.a2.set_xlim(totalRange/100.*value/self.incr, totalRange/100.*value/self.incr + self.windowSize/self.incr)
        self.canvas.draw()

    def onClick(self, event):
        # Different behaviour according to if you click in an already created box, or are making one
        # if self.box1id>-1:
        #     self.listRectanglesa1[self.box1id].set_facecolor('red')
        #     self.listRectanglesa2[self.box1id].set_facecolor('red')
        # # Deactivate the radio buttons and listbox
        # for i in xrange(len(self.birds1)):
        #     self.birds1[i].setChecked(False)
        #     self.birds1[i].setEnabled(False)
        # self.birdList.setEnabled(False)
        # # What about the textbox?

        # Check if the user has clicked in a box
        box1id = -1
        for count in range(len(self.listRectanglesa1)):
            if self.listRectanglesa1[count].xy[0] <= event.xdata and self.listRectanglesa1[count].xy[0]+self.listRectanglesa1[count].get_width() >= event.xdata:
                box1id = count
            if self.listRectanglesa2[count].xy[0] <= event.xdata and self.listRectanglesa2[count].xy[0]+self.listRectanglesa2[count].get_width() >= event.xdata:
                box1id = count
        print "box " + str(box1id)

        if box1id>-1:
            # User has clicked on a box
            # Store it in order to reset colour later
            self.box1id = box1id
            self.listRectanglesa1[box1id].set_facecolor('green')
            self.listRectanglesa2[box1id].set_facecolor('green')
            self.zoomstart = self.listRectanglesa1[box1id].xy[0]
            self.zoomend = self.listRectanglesa1[box1id].xy[0]+self.listRectanglesa1[box1id].get_width()
            self.showPlotZoom()

            # Activate the radio buttons for labelling
            found = False
            for i in range(len(self.birds1)):
                self.birds1[i].setEnabled(True)
                if str(self.a1text[box1id].get_text()) == self.birds1[i].text():
                    self.birds1[i].setChecked(True)
                    found = True

            if not found:
                self.birds1[i].setChecked(True)
                self.birdList.setEnabled(True)
                items = self.birdList.findItems(str(self.a1text[box1id].get_text()), Qt.MatchExactly)
                for item in items:
                    self.birdList.setCurrentItem(item)
        else:
            # User is doing segmentation
            a1 = str(self.a1)
            #a2 = str(self.a2)
            a1ind = float(a1[a1.index(',') + 1:a1.index(';') - 1])
            #a2ind = float(a2[a2.index(',') + 1:a2.index(';') - 1])

            if event.inaxes is not None:
                # Work out which axes you are in
                # Then make it appear in both
                s = event.xdata
                a = str(event.inaxes)
                aind = float(a[a.index(',') + 1:a.index(';') - 1])
                if aind == a1ind:
                    s2 = np.round(float(s) / self.incr)
                else:
                    s2 = s
                    s = np.round(s*self.incr)

                if self.start_stop==0:
                    # This is the start of a segment, draw a green line
                    self.markstarta1 = self.a1.add_patch(pl.Rectangle((s, np.min(self.audiodata)), self.linewidtha1, np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='g', edgecolor='None',alpha=0.8))
                    #self.markstarta2 =self.a2.add_patch(pl.Rectangle((s2, np.min(self.audiodata)), self.linewidtha2, np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='g', edgecolor='None',alpha=0.8))
                    self.markstarta2 = self.a2.add_patch(pl.Rectangle((s2, 0), self.linewidtha2, self.window_width/2, facecolor='g', edgecolor='None',alpha=0.8))
                    self.start_a = s
                    self.start_s = s2
                else:
                    # This is the end, draw the box
                    self.markstarta1.remove()
                    self.markstarta2.remove()
                    #self.a1.add_patch(pl.Rectangle((s, np.min(self.audiodata)), self.linewidtha1, np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r', edgecolor='None',alpha=0.8))
                    #self.a2.add_patch(pl.Rectangle((s2, np.min(self.audiodata)), self.linewidtha2, np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r', edgecolor='None',alpha=0.8))
                    a1R = self.a1.add_patch(pl.Rectangle((self.start_a, np.min(self.audiodata)), s - self.start_a, np.abs(np.min(self.audiodata)) + np.max(self.audiodata),facecolor='r', alpha=0.5))
                    #a2R = self.a2.add_patch(pl.Rectangle((self.start_s, np.min(self.audiodata)), s2 - self.start_s, np.abs(np.min(self.audiodata)) + np.max(self.audiodata),facecolor='r', alpha=0.5))
                    a2R = self.a2.add_patch(pl.Rectangle((self.start_s, 0), s2 - self.start_s, self.window_width/2,facecolor='r', alpha=0.5))
                    self.listRectanglesa1.append(a1R)
                    self.listRectanglesa2.append(a2R)
                    self.segments.append([min(self.start_a,s),max(self.start_a,s),'None'])
                    a1t = self.a1.text(self.start_a, np.min(self.audiodata), 'None')
                    self.a1text.append(a1t)

                    # Show it in the zoom window
                    #self.listRectanglesb1 = []
                    #self.listRectanglesb2 = []
                    self.zoomstart = a1R.xy[0]
                    self.zoomend = a1R.xy[0] + a1R.get_width()
                    self.box1id = len(self.segments)-1
                    self.showPlotZoom()

                self.start_stop = 1 - self.start_stop
                print len(self.listRectanglesa1)

            else:
                print 'Clicked ouside axes bounds but inside plot window'

        self.canvas.draw()

    def chooseEnd(self,event):
        # Note: if you select an end, and then don't deselect it before clicking on the other there is a bug
        # -> it moves a bit -> need to enable deselection (click on again?)
        if self.buttonID is not None:
            self.canvas2.mpl_disconnect(self.buttonID)
        # Put the colours back
        for box in self.listRectanglesb1:
            if box.get_facecolor() == matplotlib.colors.colorConverter.to_rgba('k'):
                # It's black, put it back
                box.set_facecolor(self.Boxcol)
        for box in self.listRectanglesb2:
            if box.get_facecolor() == matplotlib.colors.colorConverter.to_rgba('k'):
                box.set_facecolor(self.Boxcol)

        self.Box1 = event.artist

        self.box2id = -1
        for count in range(len(self.listRectanglesb1)):
            if self.listRectanglesb1[count].xy[0] == self.Box1.xy[0]:
                self.box2id = count
                self.Box2 = self.listRectanglesb2[self.box2id]
                self.axes = 3
        if self.box2id == -1:
            # Must be in other axes
            for count in range(len(self.listRectanglesb2)):
                if self.listRectanglesb2[count].xy[0] == self.Box1.xy[0]:
                    self.box2id = count
                    self.Box2 = self.Box1
                    self.Box1 = self.listRectanglesb1[self.box2id]
                    self.axes = 4

        self.Boxcol = self.Box1.get_facecolor()
        self.Box1.set_facecolor('black')
        self.Box2.set_facecolor('black')
        self.buttonID = self.canvas2.mpl_connect('button_press_event', self.moveEnd)
        self.keypress = self.canvas2.mpl_connect('key_press_event', self.deleteEnd)
        self.canvas2.draw()

    def moveEnd(self,event):
        if self.Box1 is None: return
        current_x1 = self.Box1.get_x()
        current_x2 = self.Box2.get_x()
        current_width1 = self.Box1.get_width()
        current_width2 = self.Box2.get_width()
        if (current_x1<=event.xdata and (current_x1+current_width1)>=event.xdata) or (current_x2<=event.xdata and (current_x2+current_width2)>=event.xdata):
            # Clicked on the same box, don't move it
            pass
        else:
            if self.axes==3:
                self.Box1.set_x(event.xdata)
                self.Box2.set_x(event.xdata/self.incr)
                if self.Boxcol == matplotlib.colors.colorConverter.to_rgba('g'):
                    # Green -> left
                    self.listRectanglesa1[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x() + (event.xdata - current_x1))
                    self.listRectanglesa1[self.box1id].set_width(self.listRectanglesa1[self.box1id].get_width() - (event.xdata - current_x1))
                    self.listRectanglesa2[self.box1id].set_x(self.listRectanglesa2[self.box1id].get_x() + (event.xdata/self.incr - current_x2))
                    self.listRectanglesa2[self.box1id].set_width(self.listRectanglesa2[self.box1id].get_width() - (event.xdata/self.incr - current_x2))
                    self.a1text[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x())
                else:
                    self.listRectanglesa1[self.box1id].set_width(self.listRectanglesa1[self.box1id].get_width() + (event.xdata - current_x1))
                    self.listRectanglesa2[self.box1id].set_width(self.listRectanglesa2[self.box1id].get_width() + (event.xdata/self.incr - current_x2))
            else:
                self.Box2.set_x(event.xdata)
                self.Box1.set_x(event.xdata*self.incr)
                if self.Boxcol == matplotlib.colors.colorConverter.to_rgba('g'):
                    # Green -> left
                    self.listRectanglesa1[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x() + (event.xdata*self.incr - current_x1))
                    self.listRectanglesa1[self.box1id].set_width(self.listRectanglesa1[self.box1id].get_width() - (event.xdata*self.incr - current_x1))
                    self.listRectanglesa2[self.box1id].set_x(self.listRectanglesa2[self.box1id].get_x() + (event.xdata - current_x2))
                    self.listRectanglesa2[self.box1id].set_width(self.listRectanglesa2[self.box1id].get_width() - (event.xdata - current_x2))
                    self.a1text[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x())
                else:
                    self.listRectanglesa2[self.box1id].set_width(self.listRectanglesa2[self.box1id].get_width() + (event.xdata - current_x2))
                    self.listRectanglesa1[self.box1id].set_width(self.listRectanglesa1[self.box1id].get_width() + (event.xdata*self.incr - current_x1))
        self.segments[self.box1id][0] = self.listRectanglesa1[self.box1id].get_x()
        self.segments[self.box1id][1] = self.listRectanglesa1[self.box1id].get_x() + self.listRectanglesa1[
            self.box1id].get_width()
        self.Box1.set_facecolor(self.Boxcol)
        self.Box2.set_facecolor(self.Boxcol)
        self.Box1 = None
        self.Box2 = None
        self.canvas2.mpl_disconnect(self.buttonID)
        self.canvas2.mpl_disconnect(self.keypress)
        self.canvas2.draw()
        self.canvas.draw()

    def deleteEnd(self,event):
        if self.Box1 is None: return
        print event.key
        # TODO: how to set amount to move?
        # TODO: how to remove the listener callback?
        move_amount = 100.0
        if event.key == 'left':
            self.Box1.set_x(self.Box1.get_x() - move_amount)
            self.Box2.set_x(self.Box2.get_x() - (move_amount / self.incr))
            # Point is wrong!
            if self.Boxcol == matplotlib.colors.colorConverter.to_rgba('g'):
                self.listRectanglesa1[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x() - move_amount)
                self.listRectanglesa1[self.box1id].set_width(self.listRectanglesa1[self.box1id].get_width() + move_amount)
                self.listRectanglesa2[self.box1id].set_x(self.listRectanglesa2[self.box1id].get_x() - move_amount/self.incr)
                self.listRectanglesa2[self.box1id].set_width(self.listRectanglesa2[self.box1id].get_width() + move_amount/self.incr)
                self.a1text[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x())
            else:
                self.listRectanglesa1[self.box1id].set_width(
                    self.listRectanglesa1[self.box1id].get_width() - move_amount)
                self.listRectanglesa2[self.box1id].set_width(
                    self.listRectanglesa2[self.box1id].get_width() - (move_amount / self.incr))
            self.segments[self.box1id][0] = self.listRectanglesa1[self.box1id].get_x()
            self.segments[self.box1id][1] = self.listRectanglesa1[self.box1id].get_x() + self.listRectanglesa1[self.box1id].get_width()
        elif event.key == 'right':
            self.Box1.set_x(self.Box1.get_x() + move_amount)
            self.Box2.set_x(self.Box2.get_x() + (move_amount / self.incr))
            # Point is wrong!
            if self.Boxcol == matplotlib.colors.colorConverter.to_rgba('g'):
                self.listRectanglesa1[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x() + move_amount)
                self.listRectanglesa1[self.box1id].set_width(
                    self.listRectanglesa1[self.box1id].get_width() - move_amount)
                self.listRectanglesa2[self.box1id].set_x(
                    self.listRectanglesa2[self.box1id].get_x() + move_amount / self.incr)
                self.listRectanglesa2[self.box1id].set_width(
                    self.listRectanglesa2[self.box1id].get_width() - move_amount / self.incr)
                self.a1text[self.box1id].set_x(self.listRectanglesa1[self.box1id].get_x())
            else:
                self.listRectanglesa1[self.box1id].set_width(
                    self.listRectanglesa1[self.box1id].get_width() + move_amount)
                self.listRectanglesa2[self.box1id].set_width(
                    self.listRectanglesa2[self.box1id].get_width() + (move_amount / self.incr))
            self.segments[self.box1id][0] = self.listRectanglesa1[self.box1id].get_x()
            self.segments[self.box1id][1] = self.listRectanglesa1[self.box1id].get_x() + self.listRectanglesa1[
                self.box1id].get_width()
        elif event.key == 'backspace':
            if len(self.listRectanglesb1) == 2:
                # Why can't these be deleted?
                self.listRectanglesb1[0].remove()
                self.listRectanglesb2[0].remove()
                self.listRectanglesb1[1].remove()
                self.listRectanglesb2[1].remove()
                self.listRectanglesa1[self.box1id].remove()
                self.listRectanglesa2[self.box1id].remove()
                self.listRectanglesa1.remove(listRectanglesa1[self.box1id])
                self.listRectanglesa2.remove(listRectanglesa2[self.box1id])
                self.a1text[self.box1id].remove()
                self.a1text.remove(self.a1text[box1id])
                self.segments.remove(self.segments[self.box1id])

        # TODO: When to stop and turn them back to right colour? -> mouse moves out of window?
        self.canvas2.draw()
        self.canvas.draw()

    def addSegmentClick(self):
        # Turn the labelling off to avoid confusion
        for i in xrange(len(self.birds1)):
            self.birds1[i].setEnabled(False)
        self.birdList.setEnabled(False)
        self.tbox.setEnabled(False)

        # Should check for and stop the listener for the moving of segment ends
        if hasattr(self,'keypress'):
            print "stopping"
            self.Box1.set_facecolor(self.Boxcol)
            self.Box2.set_facecolor(self.Boxcol)
            self.Box1 = None
            self.Box2 = None
            self.canvas2.mpl_disconnect(self.buttonID)
            self.canvas2.mpl_disconnect(self.keypress)
            self.canvas2.draw()

        self.buttonAdd = self.canvas2.mpl_connect('button_press_event', self.addSegment)

    def addSegment(self,event):
        # First click should be the END of the first segment, second should be START of second
        # This should work logically providing person moves left to right

        a3 = str(self.a3)
        #a4 = str(self.a4)
        a3ind = float(a3[a3.index(',') + 1:a3.index(';') - 1])
        #a4ind = float(a4[a4.index(',') + 1:a4.index(';') - 1])

        if event.inaxes is not None:
            # Work out which axes you are in
            # Then make it appear in all of them
            s = event.xdata
            a = str(event.inaxes)
            aind = float(a[a.index(',') + 1:a.index(';') - 1])
            if aind == a3ind:
                s2 = np.round(s / self.incr)
            else:
                s2 = s
                s = np.round(s * self.incr)

            if self.start_stop2 == 0:
                # This is the end of the first segment, draw a red line
                # TODO: At the moment you can't pick these, 'cos it's a pain. Does it matter?
                markstarta3 = self.a3.add_patch(pl.Rectangle((s, np.min(self.audiodata)), self.bar_thickness,
                                                                  np.abs(np.min(self.audiodata)) + np.max(
                                                                      self.audiodata), facecolor='r', edgecolor='None',
                                                                  alpha=0.8)) #,picker=10))
                markstarta4 = self.a4.add_patch(pl.Rectangle((s2, 0), self.bar_thickness/self.incr,
                                                                  self.window_width/2, facecolor='r', edgecolor='None',
                                                                  alpha=0.8)) #,picker=10))
                self.markstarta1 = self.a1.add_patch(pl.Rectangle((s + self.offset_fig1, np.min(self.audiodata)), self.bar_thickness,
                                                                  np.abs(np.min(self.audiodata)) + np.max(
                                                                      self.audiodata), facecolor='r', edgecolor='None',
                                                                  alpha=0.8))
                self.markstarta2 = self.a2.add_patch(pl.Rectangle((s2 + self.offset_fig1/self.incr, 0), self.bar_thickness/self.incr, self.window_width, facecolor='r', edgecolor='None',alpha=0.8))
                self.listRectanglesb1.append(markstarta3)
                self.listRectanglesb2.append(markstarta4)
                self.end_a2 = s
                self.end_s2 = s2
            else:
                # This is the start of the second segment, draw red lines on fig2, then redraw the boxes on the top figure and save
                b1e = self.a3.add_patch(pl.Rectangle((s, np.min(self.audiodata)), self.bar_thickness, np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='g', edgecolor='None',alpha=0.8,picker=10))
                b2e = self.a4.add_patch(pl.Rectangle((s2, 0), self.bar_thickness/self.incr, self.window_width/2, facecolor='g', edgecolor='None',alpha=0.8,picker=10))
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
                self.segments.append([old[0],self.end_a2 + self.offset_fig1,old[2]])
                self.segments.append([s + self.offset_fig1, old[1], old[2]])
                #self.segments[self.box1id][1] = self.end_a2 + self.offset_fig1
                # Delete the old rectangles and text

                self.a1text[self.box1id].remove()
                self.a1text.remove(self.a1text[self.box1id])

                # Add the two new ones
                a1R = self.a1.add_patch(pl.Rectangle((self.segments[-2][0], np.min(self.audiodata)), self.segments[-2][1] - self.segments[self.box1id][0],
                                                     np.abs(np.min(self.audiodata)) + np.max(self.audiodata),
                                                     facecolor='r', alpha=0.5))
                a2R = self.a2.add_patch(pl.Rectangle((self.segments[-2][0]/self.incr, 0), (self.segments[-2][1] - self.segments[self.box1id][0])/self.incr,
                                                     self.window_width/2,
                                                     facecolor='r', alpha=0.5))

                self.listRectanglesa1.append(a1R)
                self.listRectanglesa2.append(a2R)

                a1t = self.a1.text(self.segments[-2][0], np.min(self.audiodata), self.segments[-2][2])
                self.a1text.append(a1t)

                a3R = self.a1.add_patch(pl.Rectangle((self.segments[-1][0], np.min(self.audiodata)), self.segments[-1][1] - self.segments[-1][0],
                                                     np.abs(np.min(self.audiodata)) + np.max(self.audiodata),
                                                     facecolor='r', alpha=0.5))
                a4R = self.a2.add_patch(pl.Rectangle((self.segments[-1][0]/self.incr, 0), (self.segments[-1][0]-self.segments[-1][1])/self.incr,
                                                 self.window_width/2,
                                                 facecolor='r', alpha=0.5))
                self.listRectanglesa1.append(a3R)
                self.listRectanglesa2.append(a4R)
                a2t = self.a1.text(self.segments[-1][0], np.min(self.audiodata), self.segments[-1][2])
                self.a1text.append(a2t)

                # Stop the listener
                self.canvas2.mpl_disconnect(self.buttonAdd)
                self.box1id = None
        self.canvas2.draw()
        self.canvas.draw()
        self.start_stop2 = 1 - self.start_stop2
        # For now at least, stop the canvas2 listener
        self.canvas2.mpl_connect('pick_event', self.chooseEnd)

    def denoise(self):
        print "Denoising"
        den = Denoise.Denoise(self.audiodata,self.sampleRate)
        self.audiodata = den.denoise()
        print "Done"
        self.sg = spectrogram(self.audiodata)
        self.showPlot()

    def segment(self):
        pass

    def recognise(self):
        pass

    def listLoadFile(self,name):
        self.loadFile(name.text())

    def loadFile(self,name):
        if len(self.segments)>0:
            self.saveSegments()
        self.segments = []

        self.sampleRate, self.audiodata = wavfile.read(name)
        self.filename = name
        #self.sampleRate, self.audiodata = wavfile.read(name.text())
        self.audiodata = self.audiodata.astype('float') / 32768.0
        if np.shape(np.shape(self.audiodata))[0]>1:
            self.audiodata = self.audiodata[:,0]
        self.datamax = np.shape(self.audiodata)[0]

        self.sg = spectrogram(self.audiodata)

        if os.path.isfile(name+'.data'):
            file = open(name+'.data', 'r')
            self.segments = json.load(file)
            file.close()

        self.windowSize = self.sampleRate*self.windowWidth

        self.showPlot()

    def openFile(self):
        Formats = "Wav file (*.wav)"
        filename = QFileDialog.getOpenFileName(self, 'Open File', '/Users/srmarsla/Projects/AviaNZ', Formats)
        if filename != None:
            loadFile(filename)

        self.sg = self.spectrogram()
        self.showPlot()

    def showPlot(self):
        # Draw the two charts
        self.a1 = self.fig.add_subplot(211)
        self.a1.clear()
        self.a1.set_xlim(self.windowStart,self.windowSize)
        self.a1.plot(self.audiodata)
        self.a1.axis('off')

        self.a2 = self.fig.add_subplot(212)
        self.a2.clear()
        self.a2.imshow(self.sg,cmap=self.cmap_grey,aspect='auto')
        self.a2.axis('off')
        self.a2.set_xlim(self.windowStart/self.incr,self.windowSize/self.incr)

        # If there were segments already made, show them
        for count in range(len(self.segments)):
            #self.a1.add_patch(pl.Rectangle((self.segments[count][0], np.min(self.audiodata)), self.linewidtha1,np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='g',edgecolor='None', alpha=0.8))
            #self.a2.add_patch(pl.Rectangle((self.segments[count][0]/self.incr, 0), self.linewidtha2,self.window_width/2, facecolor='g', edgecolor='None', alpha=0.8))
            #self.a1.add_patch(pl.Rectangle((self.segments[count][1], np.min(self.audiodata)), self.linewidtha1,np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r',edgecolor='None', alpha=0.8))
            #self.a2.add_patch(pl.Rectangle((self.segments[count][1]/self.incr, 0), self.linewidtha2,self.window_width/2, facecolor='r',edgecolor='None', alpha=0.8))
            a1R = self.a1.add_patch(pl.Rectangle((self.segments[count][0], np.min(self.audiodata)),
                                           self.segments[count][1] - self.segments[count][0],
                                           np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r',
                                           alpha=0.5))
            a2R = self.a2.add_patch(pl.Rectangle((self.segments[count][0] / self.incr, 0),
                                           self.segments[count][1] / self.incr - self.segments[count][0] / self.incr,
                                           self.window_width / 2, facecolor='r',
                                           alpha=0.5))
            self.listRectanglesa1.append(a1R)
            self.listRectanglesa2.append(a2R)
            a1t = self.a1.text(self.segments[count][0], np.min(self.audiodata),self.segments[count][2])
            self.a1text.append(a1t)

            # Next 2 ways would be better, but a bit of a pain to work out!
            # Coords are relative, first index needs fiddling, and then would have to make it move with slider
            # self.a1.annotate()
            # self.fig.text(0.3,0.5,'xxx')

        self.canvas.draw()

    def showPlotZoom(self):
        # This is for the zoomed-in window
        start = self.zoomstart
        end = self.zoomend
        self.width_param = 0.01
        self.padding = 1000
        self.listRectanglesb1 = []
        self.listRectanglesb2 = []
        # Make the start and end bands be big and draggable
        # TODO: Size needs to be relative, and the ends should be clear
        # Choose the widths of the bars
        # Subtract off the thickness of the bars
        # When they are moved, update the plotting
        self.bar_thickness = self.width_param*(end-start)
        self.bar_thickness = max(self.bar_thickness,150)
        # Draw the two charts
        self.a3 = self.fig2.add_subplot(211)
        self.a3.clear()
        xstart = max(start-self.padding,0)
        if xstart != 0:
            start = self.padding
        xend = min(end+self.padding,len(self.audiodata))
        self.a3.plot(self.audiodata[xstart:xend])
        self.a3.axis('off')

        self.a4 = self.fig2.add_subplot(212)
        self.a4.clear()
        newsg = spectrogram(self.audiodata[xstart:xend])
        self.a4.imshow(newsg, cmap=self.cmap_grey, aspect='auto')
        self.a4.axis('off')

        self.End1 = self.a3.add_patch(pl.Rectangle((start-self.bar_thickness, np.min(self.audiodata)), self.bar_thickness,
                                       np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='g',
                                       edgecolor='None', alpha=1.0, picker=10))
        self.End2 = self.a4.add_patch(pl.Rectangle(((start-self.bar_thickness)/self.incr, 0), self.bar_thickness/self.incr,
                                       self.window_width / 2, facecolor='g',
                                       edgecolor='None', alpha=1.0, picker=10))
        self.End3 = self.a3.add_patch(pl.Rectangle((xend-self.zoomstart+self.bar_thickness, np.min(self.audiodata)), self.bar_thickness,
                                       np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r',
                                       edgecolor='None', alpha=1.0, picker=10))
        self.End4 = self.a4.add_patch(pl.Rectangle(((xend-self.zoomstart+self.bar_thickness)/self.incr, 0), self.bar_thickness/self.incr,
                                       self.window_width / 2, facecolor='r',
                                       edgecolor='None', alpha=1.0, picker=10))
        self.listRectanglesb1.append(self.End1)
        self.listRectanglesb2.append(self.End2)
        self.listRectanglesb1.append(self.End3)
        self.listRectanglesb2.append(self.End4)
        self.offset_fig1 = self.zoomstart - start
        self.canvas2.mpl_connect('pick_event', self.chooseEnd)
        self.canvas2.draw()

    def changeWidth(self,value):
        # Updates the plot as the window width is changed
        self.windowWidth = value
        self.windowSize = self.sampleRate*self.windowWidth
        self.a1.set_xlim(self.windowStart, self.windowSize)
        self.a2.set_xlim(self.windowStart / self.incr, self.windowSize / self.incr)
        self.canvas.draw()

    def writefile(self,name):
        # Need them to be 16 bit
        self.fwData *= 32768.0
        self.fwData = self.fwData.astype('int16')
        wavfile.write(name,self.sampleRate, self.fwData)

    def play(self):
        import sounddevice as sd
        sd.play(self.audiodata[self.zoomstart:self.zoomend],self.sampleRate)

    def birds1Clicked(self):
        for button in self.birds1:
            if button.isChecked():
                if button.text()=="Other":
                    self.birdList.setEnabled(True)
                else:
                    self.birdList.setEnabled(False)
                    self.segments[self.box1id][2] = str(button.text())
                    self.a1text[self.box1id].set_text(str(button.text()))
                    self.canvas.draw()

    def birds2Clicked(self, item):
        if (item.text() == "Other"):
            self.tbox.setEnabled(True)
        else:
            # Save the entry
            # TODO: Think about this -> is it always correct?
            self.segments[self.box1id][2] = str(item.text())
            self.a1text[self.box1id].set_text(str(item.text()))
            self.canvas.draw()

    def getTextEntry(self):
        # Check text isn't there
        # If not, add it
        item = self.birdList.findItems(self.tbox.text(),Qt.MatchExactly)
        if item:
            pass
        else:
            self.birdList.addItem(self.tbox.text())
        # TODO: Should this list be alpha sorted?
        self.segments[self.box1id][2] = str(self.tbox.text())
        self.a1text[self.box1id].set_text(str(self.tbox.text()))
        self.canvas.draw()
        self.tbox.setEnabled(False)

    def saveSegments(self):
        file = open(self.filename + '.data', 'w')
        json.dump(self.segments,file)

    def quit(self):
        self.saveSegments()
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
        # We want a colormap that goes from white to black in greys, but has limited contrast
        # First is sorted by keeping all 3 colours the same, second by squeezing the range
        cdict = {
            'blue': ((0, 1, 1), (self.colourStart, 1, 1), (self.colourEnd, 0, 0), (1, 0, 0)),
            'green': ((0, 1, 1), (self.colourStart, 1, 1), (self.colourEnd, 0, 0), (1, 0, 0)),
            'red': ((0, 1, 1), (self.colourStart, 1, 1), (self.colourEnd, 0, 0), (1, 0, 0))
        }
        self.cmap_grey = matplotlib.colors.LinearSegmentedColormap('cmap_grey', cdict, 256)

app = QApplication(sys.argv)
form = Interface()
form.show()
app.exec_()