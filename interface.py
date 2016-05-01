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

        QMainWindow.__init__(self, root)
        self.setWindowTitle('AviaNZ')

        self.createMenu()
        self.createFrame()

        # Make life easier for now: preload a birdsong
        fp, self.t = wavfile.read('../Birdsong/more1.wav')
        #fp, self.t = wavfile.read('tril1.wav')
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
        if event.inaxes is not None:
            print event.xdata, event.ydata
        else:
            print 'Clicked ouside axes bounds but inside plot window'

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

        a1 = self.fig.add_subplot(211)
        a1.clear()
        a1.plot(self.t)
        a1.add_patch(pl.Rectangle((10000,np.min(self.t)),end-start,np.abs(np.min(self.t))+np.max(self.t),facecolor='r',alpha=0.5))
        a1.axis('off')
        a2 = self.fig.add_subplot(212)
        a2.clear()
        a2.imshow(self.sg,cmap='gray',aspect='auto')
        a2.axis('off')
        a2.add_patch(pl.Rectangle((np.round(start/incr),0),np.round((end-start)/incr),np.shape(self.sg)[1],facecolor='r',alpha=0.2))

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