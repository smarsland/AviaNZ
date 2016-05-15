# Interface,py
#
# This is currently the base class for the AviaNZ interface
# It's fairly simplistic, but hopefully works
# Version 0.2 15/5/16
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
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
# implement the default mpl key bindings
# from matplotlib.backend_bases import key_press_handler

import Denoise
# ==============
# TODO
# Still some debugging in the clicking on boxes bit -> last box doesn't go green
# Get the listWidget to select
# Debug the denoiser! Check other sample rates for the bandpass filter
# Finish manual segmentation
# In Zoom window, fix spectrogram and allow click to move segmentation ends
# Tidy the code
# List of files in the box on the left -> directories, .., etc.
# Full interface -> see drawing
# Print text above parts that have been recognised -> put this inbetween the two graphs
# Play and pause buttons, plus a marker for where up to in playback -> needs another player
# Automatic segmentation -> power, wavelets, ...
# Put the calls to the labeller in a loop
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
        self.start_a = 0
        self.start_s = 0

        # Params for amount to plot in window
        self.windowSize = 22050 #44100
        self.windowStart = 0

        # Params for spectrogram
        self.window_width = 256
        self.incr = 128

        # Params for the width of the lines to draw for segmentation in the two plots
        self.linewidtha1 = 100
        self.linewidtha2 = self.linewidtha1/self.incr
        self.dirpath = '.'

        # This hold the actual data for now, and also the rectangular patches
        self.segments = []
        self.listRectanglesa1 = []
        self.listRectanglesa2 = []
        self.a1text = []
        self.boxid = -1

        QMainWindow.__init__(self, root)
        self.setWindowTitle('AviaNZ')

        self.createMenu()
        self.createFrame()

        # Make life easier for now: preload a birdsong
        self.loadFile('../Birdsong/more1.wav')

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

        self.frame = QWidget()
        self.dpi = 100

        self.fig = Figure((8.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.frame)

        #self.canvas.mpl_connect('pick_event', self.onClick)
        self.canvas.mpl_connect('button_press_event', self.onClick)

        self.mpl_toolbar = NavigationToolbar(self.canvas, self.frame)

        # Needs a bit of sorting -> subdirectories, etc.
        self.listFiles = QListWidget(self)
        listOfFiles = []
        #listOfFiles.extend('..')
        for extension in ['wav','WAV']:
            pattern = os.path.join(self.dirpath,'*.%s' % extension)
            listOfFiles.extend(glob.glob(pattern))
        for file in listOfFiles:
            item = QListWidgetItem(self.listFiles)
            item.setText(file)
        self.listFiles.connect(self.listFiles, SIGNAL('itemClicked(QListWidgetItem*)'), self.listLoadFile)

        #self.classifyButton = QPushButton("&Classify")
        #self.connect(self.classifyButton, SIGNAL('clicked()'), self.classify)
        self.playButton  = QPushButton("&Play")
        self.connect(self.playButton, SIGNAL('clicked()'), self.play)
        self.quitButton = QPushButton("&Quit")
        self.connect(self.quitButton, SIGNAL('clicked()'), self.quit)
        self.denoiseButton = QPushButton("&Denoise")
        self.connect(self.denoiseButton, SIGNAL('clicked()'), self.denoise)

        self.sld = QSlider(Qt.Horizontal, self)
        self.sld.setFocusPolicy(Qt.NoFocus)
        self.sld.valueChanged[int].connect(self.sliderMoved)

        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.canvas)
        vbox1.addWidget(self.sld)
        vbox1.addWidget(self.mpl_toolbar)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.listFiles)
        hbox1.addLayout(vbox1)

        hbox2 = QHBoxLayout()
        for w in [self.playButton,self.denoiseButton,self.quitButton]:
            hbox2.addWidget(w)
            hbox2.setAlignment(w, Qt.AlignVCenter)

        self.selectorLayout = QHBoxLayout()
        # Create an array of radio buttons
        self.birds1 = [QRadioButton("Female Kiwi"), QRadioButton("Male Kiwi"), QRadioButton("Ruru"), QRadioButton("Hihi"),
                  QRadioButton("Not Bird"), QRadioButton("Don't Know"), QRadioButton("Other")]
        birds1Layout = QVBoxLayout()
        # Create a button group for radio buttons
        self.birdButtonGroup = QButtonGroup()

        for i in xrange(len(self.birds1)):
            # Add each radio button to the button layout
            birds1Layout.addWidget(self.birds1[i])
            # Add each radio button to the button group & give it an ID of i
            self.birdButtonGroup.addButton(self.birds1[i], i)
            self.birds1[i].setEnabled(False)
            self.connect(self.birds1[i], SIGNAL("clicked()"), self.birds1Clicked)
        # Only need to run anything extra if the last one is clicked ("Other")
        #self.connect(birds1[len(birds1)-1], SIGNAL("clicked()"), self.birds1OtherClicked)
        self.selectorLayout.addLayout(birds1Layout)

        self.birdList = QListWidget(self)
        items = ['Other','a', 'b', 'c', 'd', 'e', 'f', 'g']
        for item in items:
            self.birdList.addItem(item)
        self.connect(self.birdList, SIGNAL("itemClicked(QListWidgetItem*)"), self.birds2Clicked)
        self.birdListLayout = QVBoxLayout()
        self.birdListLayout.addWidget(self.birdList)
        self.selectorLayout.addLayout(self.birdListLayout)
        #self.frame.setLayout(self.vbox2)
        self.birdList.setEnabled(False)
        # self.setLayout(birds1Layout)

        self.tbox = QLineEdit(self)
        self.birdListLayout.addWidget(self.tbox)
        #self.frame.setLayout(self.vbox2)
        self.connect(self.tbox, SIGNAL('editingFinished()'), self.getTextEntry)
        self.tbox.setEnabled(False)

        #self.frame.setLayout(self.vbox)
        self.fig2 = Figure((4.0, 4.0), dpi=self.dpi)
        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.setParent(self.frame)

        self.vbox2 = QVBoxLayout()
        self.vbox2.addLayout(hbox1)
        self.vbox2.addLayout(hbox2)
        self.vbox2.addLayout(self.selectorLayout)

        self.selectorLayout.addWidget(self.canvas2)

        self.frame.setLayout(self.vbox2)
        self.setCentralWidget(self.frame)

    def sliderMoved(self,value):
        # Get the scaling sorted for the slider
        totalRange = self.datamax - self.windowSize
        self.a1.set_xlim(totalRange/100.*value, totalRange/100.*value+self.windowSize)
        self.a2.set_xlim(totalRange/100.*value/self.incr, totalRange/100.*value/self.incr + self.windowSize/self.incr)
        self.canvas.draw()

    def onClick(self, event):
        # Could/should add some error checking: start before end
        # Different behaviour according to if you click in an already created box, or are making one
        if self.boxid>-1:
            self.listRectanglesa1[self.boxid].set_facecolor('red')
            self.listRectanglesa2[self.boxid].set_facecolor('red')
        # Deactivate the radio buttons and listbox
        for i in xrange(len(self.birds1)):
            self.birds1[i].setChecked(False)
            self.birds1[i].setEnabled(False)
        self.birdList.setEnabled(False)
        # What about the textbox?

        boxid = -1
        for count in range(len(self.listRectanglesa1)):
            if self.listRectanglesa1[count].xy[0] <= event.xdata and self.listRectanglesa1[count].xy[0]+self.listRectanglesa1[count].get_width() >= event.xdata:
                boxid = count
            if self.listRectanglesa2[count].xy[0] <= event.xdata and self.listRectanglesa2[count].xy[0]+self.listRectanglesa2[count].get_width() >= event.xdata:
                boxid = count

        if boxid>-1:
            # User has clicked on a box
            # Store it in order to reset colour later
            self.boxid = boxid
            self.listRectanglesa1[boxid].set_facecolor('green')
            self.listRectanglesa2[boxid].set_facecolor('green')
            self.showPlotZoom(self.listRectanglesa1[boxid].xy[0],self.listRectanglesa1[boxid].xy[0]+self.listRectanglesa1[boxid].get_width())



            # Activate the radio buttons for labelling
            # TODO: Would be nice to set the correct radiobox/listbox item
            found = False
            for i in range(len(self.birds1)):
                self.birds1[i].setEnabled(True)
                if str(self.a1text[boxid].get_text()) == self.birds1[i].text():
                    self.birds1[i].setChecked(True)
                    found = True

            if not found:
                item = self.birdList.findItems(str(self.a1text[boxid].get_text()), Qt.MatchExactly)
                print item
                if item:
                    self.birdList.setCurrentItem(item)
        else:
            print "still segmenting"
            # User is doing segmentation
            a1 = str(self.a1)
            a2 = str(self.a2)
            a1ind = float(a1[a1.index(',') + 1:a1.index(';') - 1])
            a2ind = float(a2[a2.index(',') + 1:a2.index(';') - 1])

            if event.inaxes is not None:
                # Work out which axes you are in
                # Then make it appear in both
                s = event.xdata
                a = str(event.inaxes)
                aind = float(a[a.index(',') + 1:a.index(';') - 1])
                if aind == a1ind:
                    s2 = np.round(s / self.incr)
                else:
                    s2 = s
                    s = np.round(s*self.incr)

                if self.start_stop==0:
                    # This is the start of a segment, draw a green line
                    self.a1.add_patch(pl.Rectangle((s, np.min(self.audiodata)), self.linewidtha1, np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='g', edgecolor='None',alpha=0.8, picker=1))
                    self.a2.add_patch(pl.Rectangle((s2, np.min(self.audiodata)), self.linewidtha2, np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='g', edgecolor='None',alpha=0.8, picker=1))
                    self.start_a = s
                    self.start_s = s2
                else:
                    # This is the end, draw the box
                    self.a1.add_patch(pl.Rectangle((s, np.min(self.audiodata)), self.linewidtha1, np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r', edgecolor='None',alpha=0.8,picker=1))
                    self.a2.add_patch(pl.Rectangle((s2, np.min(self.audiodata)), self.linewidtha2, np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r', edgecolor='None',alpha=0.8,picker=1))
                    a1R = self.a1.add_patch(pl.Rectangle((self.start_a, np.min(self.audiodata)), s - self.start_a, np.abs(np.min(self.audiodata)) + np.max(self.audiodata),facecolor='r', alpha=0.5,picker=1))
                    a2R = self.a2.add_patch(pl.Rectangle((self.start_s, np.min(self.audiodata)), s2 - self.start_s, np.abs(np.min(self.audiodata)) + np.max(self.audiodata),facecolor='r', alpha=0.5,picker=1))
                    self.listRectanglesa1.append(a1R)
                    self.listRectanglesa2.append(a2R)
                    print "New Rectangle. Current list of Rectangles"
                    print self.listRectanglesa1
                    print len(self.listRectanglesa1)
                    self.segments.append([min(self.start_a,s),max(self.start_a,s),'None'])
                    a1t = self.a1.text(self.start_a, np.min(self.audiodata), 'None')
                    self.a1text.append(a1t)

                self.start_stop = 1 - self.start_stop
                #print event.xdata, event.ydata, event.inaxes
            else:
                print 'Clicked ouside axes bounds but inside plot window'

        self.canvas.draw()

    def denoise(self):
        den = Denoise.Denoise(self.audiodata,self.sampleRate)
        self.audiodata = den.denoise()
        self.sg = spectrogram(self.audiodata)
        self.showPlot()

    def listLoadFile(self,name):
        self.loadFile(name.text())

    def loadFile(self,name):
        if len(self.segments)>0:
            self.saveSegments()
        self.segments = []

        self.sampleRate, self.audiodata = wavfile.read(name)
        self.filename = name
        #self.sampleRate, self.audiodata = wavfile.read(name.text())
        self.audiodata.astype('float') / 32768.0
        if np.shape(np.shape(self.audiodata))[0]>1:
            self.audiodata = self.audiodata[:,0]
        self.datamax = np.shape(self.audiodata)[0]

        self.sg = spectrogram(self.audiodata)

        if os.path.isfile(name+'.data'):
            file = open(name+'.data', 'r')
            self.segments = json.load(file)
            file.close()

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
        #self.a1.set_ylim([np.min(self.audiodata),np.max(self.audiodata)])
        self.a1.plot(self.audiodata)
        self.a1.axis('off')

        self.a2 = self.fig.add_subplot(212)
        self.a2.clear()
        self.a2.imshow(self.sg,cmap='gray_r',aspect='auto')
        self.a2.axis('off')
        self.a2.set_xlim(self.windowStart/self.incr,self.windowSize/self.incr)

        # If there were segments already made, show them
        print len(self.segments)
        for count in range(len(self.segments)):
            self.a1.add_patch(pl.Rectangle((self.segments[count][0], np.min(self.audiodata)), self.linewidtha1,np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='g',edgecolor='None', alpha=0.8))
            self.a2.add_patch(pl.Rectangle((self.segments[count][0]/self.incr, np.min(self.audiodata)), self.linewidtha2,np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='g', edgecolor='None', alpha=0.8))
            self.a1.add_patch(pl.Rectangle((self.segments[count][1], np.min(self.audiodata)), self.linewidtha1,np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r',edgecolor='None', alpha=0.8))
            self.a2.add_patch(pl.Rectangle((self.segments[count][1]/self.incr, np.min(self.audiodata)), self.linewidtha2,np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r',edgecolor='None', alpha=0.8))
            a1R = self.a1.add_patch(pl.Rectangle((self.segments[count][0], np.min(self.audiodata)),
                                           self.segments[count][1] - self.segments[count][0],
                                           np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r',
                                           alpha=0.5))
            a2R = self.a2.add_patch(pl.Rectangle((self.segments[count][0] / self.incr, np.min(self.audiodata)),
                                           self.segments[count][1] / self.incr - self.segments[count][0] / self.incr,
                                           np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r',
                                           alpha=0.5))
            self.listRectanglesa1.append(a1R)
            self.listRectanglesa2.append(a2R)
            print "Loading Rectangle. Current list of Rectangles"
            print self.listRectanglesa1
            print len(self.listRectanglesa1)
            a1t = self.a1.text(self.segments[count][0]+500, np.min(self.audiodata),self.segments[count][2])
            self.a1text.append(a1t)

            # The next way would be better, but a bit of a pain to work out!
            # Coords are relative, first index needs fiddling
            #self.fig.text(0.3,0.5,'xxx')

        self.canvas.draw()

    def showPlotZoom(self,xstart,xend):
        # This is for the zoomed-in window
        # ***** SPECTROGRAM NOT CURRENTLY CORRECT
        # Make the start and end bands be big and draggable
        # Draw the two charts
        self.a3 = self.fig2.add_subplot(211)
        self.a3.clear()
        xstart = max(xstart-500,0)
        xend = min(xend+500,len(self.audiodata))
        self.a3.plot(self.audiodata[xstart:xend])
        self.a3.axis('off')

        self.a4 = self.fig2.add_subplot(212)
        self.a4.clear()
        self.a4.imshow(self.sg[xstart/self.incr:xend/self.incr], cmap='gray_r', aspect='auto')
        self.a4.axis('off')

        self.a3.add_patch(pl.Rectangle((xstart, np.min(self.audiodata)), self.linewidtha1/10,
                                       np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r',
                                       edgecolor='None', alpha=0.1, picker=10))
        # self.a4.add_patch(pl.Rectangle((xstart/self.incr, np.min(self.audiodata)), self.linewidtha2,
        #                                np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='r',
        #                                edgecolor='None', alpha=0.8, picker=10))
        self.a3.add_patch(pl.Rectangle((xend, np.min(self.audiodata)), self.linewidtha1/10,
                                       np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='g',
                                       edgecolor='None', alpha=0.1, picker=10))
        # self.a4.add_patch(pl.Rectangle((xend/self.incr, np.min(self.audiodata)), self.linewidtha2,
        #                                np.abs(np.min(self.audiodata)) + np.max(self.audiodata), facecolor='g',
        #                                edgecolor='None', alpha=0.8, picker=10))

        self.canvas2.draw()

    def move(self,event):
        print event

    def play(self):
        import sounddevice as sd
        sd.play(self.audiodata)

    def birds1Clicked(self):
        for button in self.birds1:
            if button.isChecked():
                if button.text()=="Other":
                    self.birdList.setEnabled(True)
                else:
                    self.birdList.setEnabled(False)
                    self.segments[self.boxid][2] = str(button.text())
                    self.a1text[self.boxid].set_text(str(button.text()))
                    self.canvas.draw()

    def birds2Clicked(self, item):
        if (item.text() == "Other"):
            self.tbox.setEnabled(True)
        else:
            # Save the entry
            # TODO: Think about this -> is it always correct?
            self.segments[self.boxid][2] = str(item.text())
            self.a1text[self.boxid].set_text(str(item.text()))
            self.canvas.draw()
            #for count in range(len(self.segments)):
            #   if self.segments[count][0] <= event.xdata and self.segments[count][1] >= event.xdata:
            #       # Inside a box: store it
            #       boxid = count

                # Update the printing on the figure

    def getTextEntry(self):
        # Check if it's already in the list
        # If not, add to list
        # And save (see birds2Clicked)
        #print self.tbox.text()
        #self.birdList.setEnabled(True)

        # Check text isn't there
        # If not, add it
        item = self.birdList.findItems(self.tbox.text(),Qt.MatchExactly)
        if item:
            pass
        else:
            self.birdList.addItem(self.tbox.text())
        # TODO: Should this list be alpha sorted?
        self.segments[self.boxid][2] = str(self.tbox.text())
        self.a1text[self.boxid].set_text(str(self.tbox.text()))
        self.canvas.draw()
        self.tbox.setEnabled(False)


    def saveSegments(self):
        file = open(self.filename + '.data', 'w')
        json.dump(self.segments,file)

    def quit(self):
        self.saveSegments()
        QApplication.quit()



app = QApplication(sys.argv)
form = Interface()
form.show()
app.exec_()