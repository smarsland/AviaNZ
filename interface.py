import sys, os
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
from matplotlib.backend_bases import key_press_handler

# ==============
# TODO
# Store the values selected in the second window -> design some form of data storage
# Turn the second window into a QDialog
# Put the calls to the second window in a loop
# Do the segmentation so that the red boxes make sense
# Add the denoising -> make pywt work!
# Get suggestions from the others
# Use the clicks on the first page to let people select other regions where there are calls

# Other version -> manual segmentation
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

        QMainWindow.__init__(self, root)
        self.setWindowTitle('AviaNZ')

        self.createMenu()
        self.createFrame()

        # Make life easier for now: preload a birdsong
        fp, self.t = wavfile.read('../Birdsong/more1.wav')
        #fp, self.t = wavfile.read('kiwi.wav')
        #self.t = self.t[:,0]
        #fp, self.t = wavfile.read('/Users/srmarsla/Students/Nirosha/bittern/ST0026.wav')

        self.sg = spectrogram(self.t)
        self.drawSpec()

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

        self.fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.frame)

        self.canvas.mpl_connect('button_press_event', self.onClick)
        self.canvas.mpl_connect('pick_event', self.onPick)

        self.mpl_toolbar = NavigationToolbar(self.canvas, self.frame)

        self.classifyButton = QPushButton("&Classify")
        self.connect(self.classifyButton, SIGNAL('clicked()'), self.classify)
        self.playButton  = QPushButton("&Play")
        self.connect(self.playButton, SIGNAL('clicked()'), self.play)
        self.quitButton = QPushButton("&Quit")
        self.connect(self.quitButton, SIGNAL('clicked()'), self.quit)

        hbox = QHBoxLayout()
        for w in [self.classifyButton,self.playButton,self.quitButton]:
            hbox.addWidget(w)
            hbox.setAlignment(w, Qt.AlignVCenter)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox)

        self.frame.setLayout(vbox)
        self.setCentralWidget(self.frame)

    def onClick(self, event):

        # This is for segmentation
        a1 = str(self.a1)
        a2 = str(self.a2)
        a1ind = float(a1[a1.index(',') + 1:a1.index(';') - 1])
        a2ind = float(a2[a2.index(',') + 1:a2.index(';') - 1])
        incr = 128

        if event.inaxes is not None:
            # Work out which axes you are in
            # Then make it appear in both
            s = event.xdata
            a = str(event.inaxes)
            aind = float(a[a.index(',') + 1:a.index(';') - 1])
            if aind == a1ind:
                s2 = np.round(s / incr)
            else:
                s2 = s
                s = np.round(s*incr)
            width1 = 100
            width2 = 2
            if self.start_stop==0:
                self.a1.add_patch(pl.Rectangle((s, np.min(self.t)), width1, np.abs(np.min(self.t)) + np.max(self.t), facecolor='g', edgecolor='None',alpha=0.8, picker=1))
                self.a2.add_patch(pl.Rectangle((s2, np.min(self.t)), width2, np.abs(np.min(self.t)) + np.max(self.t), facecolor='g', edgecolor='None',alpha=0.8, picker=1))
                self.start_a = s
                self.start_s = s2
            else:
                self.a1.add_patch(pl.Rectangle((s, np.min(self.t)), width1, np.abs(np.min(self.t)) + np.max(self.t), facecolor='r', edgecolor='None',alpha=0.8,picker=1))
                self.a2.add_patch(pl.Rectangle((s2, np.min(self.t)), width2, np.abs(np.min(self.t)) + np.max(self.t), facecolor='r', edgecolor='None',alpha=0.8,picker=1))
                self.a1.add_patch(pl.Rectangle((self.start_a, np.min(self.t)), s - self.start_a, np.abs(np.min(self.t)) + np.max(self.t),facecolor='r', alpha=0.5))
                self.a2.add_patch(pl.Rectangle((self.start_s, np.min(self.t)), s2 - self.start_s, np.abs(np.min(self.t)) + np.max(self.t),facecolor='r', alpha=0.5))
            self.canvas.draw()
            self.start_stop = 1 - self.start_stop
            #print event.xdata, event.ydata, event.inaxes
        else:
            print 'Clicked ouside axes bounds but inside plot window'

    def onPick(self,event):
        # Two things: use this to let the user move a set start
        # And stop it being a click as well!
        box_points = event.artist.get_bbox().get_points()
        print box_points

    def openFile(self):
        Formats = "Wav file (*.wav)"
        filename = QFileDialog.getOpenFileName(self, 'Open File', '/Users/srmarsla/Projects/AviaNZ', Formats)
        if filename != None:
            fp, self.t = wavfile.read(filename)
        self.sg = self.sp().spectrogram()
        self.drawSpec()

    def drawSpec(self):
        start = 10000
        end = 12000
        incr = 128

        self.a1 = self.fig.add_subplot(211)
        print self.a1
        self.a1.clear()
        self.a1.plot(self.t)
        self.a1.axis('off')
        self.a2 = self.fig.add_subplot(212)
        self.a2.clear()
        self.a2.imshow(self.sg,cmap='gray',aspect='auto')
        self.a2.axis('off')

        self.canvas.draw()

    def move(self,event):
        print event

    def play(self):
        import sounddevice as sd
        sd.play(self.t)

    def classify(self):
        self.childWindow = Classify(self.t[10000:12000])
        # Which is the right thing: pass the signal or the fft, or both?
        # This is just the signal, and then recomputes the spectrogram -> wasteful
        #self.classify = Classify(self.t[10000:12000])

    def quit(self):
        QApplication.quit()

class Classify(QMainWindow):
    # TO DO: Sort out inheritance and get this properly structured
    def __init__(self,t):

        QMainWindow.__init__(self)
        self.setWindowTitle('Classify')

        self.t = t

        self.createFrame()
        self.show()

    def createFrame(self):
        self.frame = QWidget()
        self.dpi = 100

        self.fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.frame)

        #self.canvas.mpl_connect('pick_event', self.on_pick)
        #self.mpl_toolbar = NavigationToolbar(self.canvas, self.frame)

        self.playButton = QPushButton("&Play")
        self.connect(self.playButton, SIGNAL('clicked()'), self.play)

        self.dontKnowButton = QPushButton("&Don't Know")
        self.connect(self.dontKnowButton, SIGNAL('clicked()'), self.dontKnow)

        self.quitButton = QPushButton("&Close")
        self.connect(self.quitButton, SIGNAL('clicked()'), self.exit)

        hbox1 = QHBoxLayout()
        for w in [self.playButton, self.dontKnowButton, self.quitButton]:
            hbox1.addWidget(w)
            hbox1.setAlignment(w, Qt.AlignVCenter)

        # Create an array of radio buttons
        birds1 = [QRadioButton("Female Kiwi"), QRadioButton("Male Kiwi"), QRadioButton("Ruru") , QRadioButton("Hihi"), QRadioButton("Not Bird"), QRadioButton("Don't Know"), QRadioButton("Other")]

        self.selectorLayout = QHBoxLayout()

        birds1Layout = QVBoxLayout()

        # Create a button group for radio buttons
        self.birdButtonGroup = QButtonGroup()

        for i in xrange(len(birds1)):
            # Add each radio button to the button layout
            birds1Layout.addWidget(birds1[i])
            # Add each radio button to the button group & give it an ID of i
            self.birdButtonGroup.addButton(birds1[i], i)
        # Only need to run anything if the last one is clicked ("Other")
        self.connect(birds1[i], SIGNAL("clicked()"), self.birds1Clicked)

        self.selectorLayout.addLayout(birds1Layout)
        #self.setLayout(birds1Layout)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.canvas)
        #vbox.addWidget(self.mpl_toolbar)
        self.vbox.addLayout(hbox1)
        self.vbox.addLayout(self.selectorLayout)

        self.frame.setLayout(self.vbox)
        self.setCentralWidget(self.frame)

        self.sg = spectrogram(self.t)
        self.drawSpec()


    def birds1Clicked(self):

        self.list = QListWidget(self)
        items = ['a','b','c','d','e','f','g','Other']
        for item in items:
            self.list.addItem(item)
        self.connect(self.list, SIGNAL("itemClicked(QListWidgetItem*)"), self.birds2Clicked)
        self.selectorLayout.addWidget(self.list)
        self.frame.setLayout(self.vbox)


    def birds2Clicked(self,item):
        print item.text()
        if (item.text()=='Other'):
            self.tbox = QLineEdit(self)
            self.selectorLayout.addWidget(self.tbox)
            self.frame.setLayout(self.vbox)

    def drawSpec(self):
        a = self.fig.add_subplot(211)
        a.plot(self.t)
        a.axis('off')
        a = self.fig.add_subplot(212)
        a.imshow(self.sg, cmap='gray',aspect='auto')
        a.axis('off')
        self.canvas.draw()

    def exit(self):
        # Find out which button, and if necessary which list item
        #if self.e is not None:
        #    self.species = self.e.get()
        #print self.species
        self.close()

    def dontKnow(self):
        self.species="dk"
        self.close()

    def play(self):
        import sounddevice as sd
        sd.play(self.t)



app = QApplication(sys.argv)
form = Interface()
form.show()
app.exec_()