# AviaNZ.py
#
# This is the main class for the AviaNZ interface
# Version 0.11 10/07/17
# Author: Stephen Marsland, with input from Nirosha Priyadarshani

#    AviaNZ birdsong analysis program
#    Copyright (C) 2017

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

import sys, os, json, platform, re

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QFileDialog, QMainWindow, QActionGroup, QToolButton, QLabel, QSlider, QScrollBar, QDoubleSpinBox, QPushButton, QListWidget, QListWidgetItem, QMenu, QFrame, QMessageBox
from PyQt5.QtCore import Qt, QDir, QTime, QTimer, QPoint, QPointF, QLocale, QFile, QIODevice
from PyQt5.QtMultimedia import QAudio, QAudioOutput, QAudioFormat

import wavio
import numpy as np

import pyqtgraph as pg
pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *
import pyqtgraph.functions as fn

import SupportClasses as SupportClasses
import Dialogs as Dialogs
import SignalProc
import Segment
import WaveletSegment
import WaveletFunctions
#import Features
#import Learning
import interface_FindSpecies
#import math

from openpyxl import load_workbook, Workbook

from pyqtgraph.parametertree import Parameter, ParameterTree #, ParameterItem, registerParameterType

import locale, time

import click

print("Package import complete.")

# ==============
# TODO

# (4) pysox

# Finish segmentation
#   Mostly there, need to test them
#   Add a minimum length of time for a segment -> make this a parameter
#   Finish sorting out parameters for median clipping segmentation, energy segmentation
#   Finish cross-correlation to pick out similar bits of spectrogram -> and what other methods?
#   Add something that aggregates them -> needs planning

# Integrate the wavelet segmentation
    # Remove the code that is in SignalProc and use that one

# At times the program does not respond and ask to repair/close (e.g. when move the overview slider fast or something like that).
# Need to work on memory management!

# Interface -> inverted spectrogram does not work - spec and amp do not synchronize

# Actions -> Denoise -> median filter check
# Make the median filter on the spectrogram have params and a dialog. Other options?

# Fundamental frequency
#   Smoothing?
#   Add shape metric
#   Try the harvest f0
#   Try yaapt or bana (tried yaapt, awful)

# Finish the raven features

# Add in the wavelet segmentation for kiwi, ruru
# Think about nice ways to train them

# Would it be good to smooth the image? Actually, lots of ideas here! Might be nice way to denoise?
    # Median filter, smoothing, consider also grab-cut
    # Continue to play with inverting spectrogram

# Colourmaps
    # HistogramLUTItem

# Make the scrollbar be the same size as the spectrogram -> hard!

# Context menu different for day and night birds?

# The ruru file is a good one to play with for now

# Look into ParameterTree for saving the config stuff in particular

# Needs decent testing

# Minor:
# Consider always resampling to 22050Hz (except when it's less in file :) )?
# Font size to match segment size -> make it smaller, could also move it up or down as appropriate
# Where should label be written?
# Use intensity of colour to encode certainty?
# If don't select something in context menu get error -> not critical
# Colours of the segments to be visible with different colourmaps? Not important!

# Look at raven and praat and luscinia -> what else is actually useful? Other annotations on graphs?

# Given files > 5 mins, split them into 5 mins versions anyway (code is there, make it part of workflow)
# Don't really want to load the whole thing, just 5 mins, and then move through with arrows -> how?
# This is sometimes called paging, I think. (y, sr = librosa.load(filename, offset=15.0, duration=5.0) might help. Doesn't do much for the overview through)
# I tried saving a figure of the overview, but they are too big in general

# Things to remember
    # When adding new classes, make sure to pass new data to them in undoing and loading

# This version has the selection of birds using a context menu and then has removed the radio buttons
# Code is still there, though, just commented out. Add as an option?

# Diane:
    # menu
    # for matching, show matched segment, and extended 'guess' (with some overlap)
    # Something to show nesting of segments, such as a number of segments in the top bit
    # Find similar segments in other files -- other birds
    # Group files by species

# Rebecca:
    # x colour spectrogram
    # x add a marker on the overview to show where you have marked segments, with different colours for unknown, possible
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

class AviaNZ(QMainWindow):
    """Main class for the user interface.
    Contains most of the user interface and plotting code"""

    def __init__(self,root=None,configfile=None,DOC=True,CLI=False,firstFile='', imageFile='', command=''):
        """Initialisation of the class. Load a configuration file, or create a new one if it doesn't
        exist. Also initialises the data structures and loads an initial file (specified explicitly)
        and sets up the window.
        One interesting configuration point is the DOC setting, which hides the more 'research' functions."""
        print("Starting AviaNZ...")
        super(AviaNZ, self).__init__()
        self.root = root
        self.extra=True

        self.CLI = CLI
        try:
            print("loading configs from file %s" % configfile)
            self.config = json.load(open(configfile))
            self.saveConfig = True
        except:
            print("Failed to load config file, using defaults")
            configfile = 'AviaNZconfig.txt'
            self.config = json.load(open(configfile))
            self.saveConfig = False # TODO: revise this with user permissions in mind
        self.configfile = configfile

        # avoid comma/point problem in number parsing
        QLocale.setDefault(QLocale(QLocale.English, QLocale.NewZealand))
        print('Locale is set to ' + QLocale().name())

        # The data structures for the segments
        self.listLabels = []
        self.listRectanglesa1 = []
        self.listRectanglesa2 = []
        self.SegmentRects = []
        self.segmentPlots=[]
        self.box1id = -1
        self.DOC = DOC
        self.started = False
        self.startedInAmpl = False
        self.startTime = 0
        self.segmentsToSave = False

        self.lastSpecies = "Don't Know"

        self.dirName = self.config['dirpath']
        self.previousFile = None
        self.focusRegion = None
        self.operator = self.config['operator']
        self.reviewer = self.config['reviewer']

        # audio things
        self.audioFormat = QAudioFormat()
        self.audioFormat.setCodec("audio/pcm")
        self.audioFormat.setByteOrder(QAudioFormat.LittleEndian)
        self.audioFormat.setSampleType(QAudioFormat.SignedInt)

        # working directory
        if not os.path.isdir(self.dirName):
            print("Directory doesn't exist: making it")
            os.makedirs(self.dirName)

        # INPUT FILE LOADING
        # search order: infile -> firstFile -> dialog
        # Make life easier for now: preload a birdsong
        if not os.path.isfile(firstFile):
            # firstFile = self.dirName + '/' + 'tril1.wav' #'male1.wav' # 'kiwi.wav'
            firstFile = "/home/julius/Documents/kiwis/rec/birds1.wav"

        if not os.path.isfile(firstFile):
            if self.CLI:
                print("file %s not found, exiting" % firstFile)
                sys.exit()
            else:
                # pop up a dialog to select file
                firstFile, drop = QtGui.QFileDialog.getOpenFileName(self, 'Choose File', self.dirName, "Wav files (*.wav)")
                while firstFile == '':
                    msg = QMessageBox()
                    msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
                    msg.setWindowIcon(QIcon('img/Avianz.ico'))
                    msg.setText("Choose a sound file to proceed.\nDo you want to continue?")
                    msg.setWindowTitle("Select Sound File")
                    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    reply = msg.exec_()
                    if reply == QMessageBox.Yes:
                        firstFile, drop = QFileDialog.getOpenFileName(self, 'Choose File', self.dirName, "Wav files (*.wav)")
                    else:
                        sys.exit()

        # parse firstFile to dir and file parts
        self.dirName = os.path.dirname(firstFile)
        firstFile = os.path.basename(firstFile)
        print("Working dir set to %s" % self.dirName)
        print("Opening file %s" % firstFile)

        # to keep code simpler, graphic options are created even in CLI mode
        # they're just not shown because QMainWindow.__init__ is skipped
        if not self.CLI:
            QMainWindow.__init__(self, root)

        self.createMenu()
        self.createFrame()
        self.resetStorageArrays()
        if self.CLI:
            self.loadFile(firstFile)
            while command!=():
                c = command[0]
                command = command[1:]
                print("next command to execute is %s" % c)
                if c=="denoise":
                    self.denoise()
                elif c=="segment":
                    self.segment()
                else:
                    print("ERROR: %s is not a valid command" % c)
                    sys.exit()
            if imageFile!='':
                # reset images to show full width if in CLI:
                self.widthWindow.setValue(self.datalengthSec)
                # looks unnecessary:
                # self.p_spec.setXRange(0, self.convertAmpltoSpec(self.datalengthSec), update=True, padding=0)
                self.saveImage(imageFile)
        else:
            # Make the window and associated widgets
            self.setWindowTitle('AviaNZ')
            keyPressed = QtCore.Signal(int)

            if self.DOC:
                self.setOperatorReviewerDialog()

            # parse mouse settings
            if self.config['drawingRightBtn']:
                self.MouseDrawingButton = QtCore.Qt.RightButton
            else:
                self.MouseDrawingButton = QtCore.Qt.LeftButton
            # Save the segments every minute
            self.timer = QTimer()
            #QObject.connect(self.timer, SIGNAL("timeout()"), self.saveSegments)
            self.timer.timeout.connect(self.saveSegments)
            self.timer.start(self.config['secsSave']*1000)
            
            self.fillFileList(firstFile)
            self.listLoadFile(firstFile)
            self.previousFile = firstFile


    def createMenu(self):
        """ Create the menu entries at the top of the screen and link them as appropriate.
        Some of them are initialised according to the data in the configuration file."""

        fileMenu = self.menuBar().addMenu("&File")
        fileMenu.addAction("&Open sound file", self.openFile, "Ctrl+O")
        # fileMenu.addAction("&Change Directory", self.chDir)
        fileMenu.addAction("&Set Operator/Reviewer (Current File)", self.setOperatorReviewerDialog)
        fileMenu.addSeparator()
        fileMenu.addAction("Quit",self.quit,"Ctrl+Q")
        # This is a very bad way to do this, but I haven't worked anything else out (setMenuRole() didn't work)
        # Add it a second time, then it appears!
        if platform.system() == 'Darwin':
            fileMenu.addAction("&Quit",self.quit,"Ctrl+Q")

        specMenu = self.menuBar().addMenu("&Appearance")

        self.useAmplitudeTick = specMenu.addAction("Show amplitude plot", self.useAmplitudeCheck)
        self.useAmplitudeTick.setCheckable(True)
        self.useAmplitudeTick.setChecked(self.config['showAmplitudePlot'])
        self.useAmplitude = True

        self.useFilesTick = specMenu.addAction("Show file list", self.useFilesCheck)
        self.useFilesTick.setCheckable(True)
        self.useFilesTick.setChecked(self.config['showListofFiles'])

        # this can go under "Change interface settings"
        self.showOverviewSegsTick = specMenu.addAction("Show annotation overview", self.showOverviewSegsCheck)
        self.showOverviewSegsTick.setCheckable(True)
        self.showOverviewSegsTick.setChecked(self.config['showAnnotationOverview'])

        self.showPointerDetails = specMenu.addAction("Show pointer details in spectrogram", self.showPointerDetailsCheck)
        self.showPointerDetails.setCheckable(True)
        self.showPointerDetails.setChecked(self.config['showPointerDetails'])

        specMenu.addSeparator()

        self.dragRectangles = specMenu.addAction("Mark boxes in spectrogram", self.dragRectanglesCheck)
        self.dragRectangles.setCheckable(True)
        self.dragRectangles.setChecked(self.config['dragBoxes'])

        self.dragRectTransparent = specMenu.addAction("Make dragged boxes transparent", self.dragRectsTransparent)
        self.dragRectTransparent.setCheckable(True)
        self.dragRectTransparent.setChecked(self.config['transparentBoxes'])

        specMenu.addSeparator()

        colMenu = specMenu.addMenu("&Choose colour map")
        colGroup = QActionGroup(self)
        for colour in self.config['ColourList']:
            cm = colMenu.addAction(colour)
            cm.setCheckable(True)
            if colour==self.config['cmap']:
                cm.setChecked(True)
            receiver = lambda checked, cmap=colour: self.setColourMap(cmap)
            #self.connect(cm, SIGNAL("triggered()"), receiver)
            cm.triggered.connect(receiver)
            colGroup.addAction(cm)
        self.invertcm = specMenu.addAction("Invert colour map",self.invertColourMap)
        self.invertcm.setCheckable(True)
        self.invertcm.setChecked(self.config['invertColourMap'])

        #if self.DOC==False:
        #    self.showFundamental2 = specMenu.addAction("Show fundamental frequency2", self.showFundamentalFreq2)
        #    self.showFundamental2.setCheckable(True)
        #    self.showFundamental2.setChecked(False)

        if self.DOC==False:
            self.showInvSpec = specMenu.addAction("Show inverted spectrogram", self.showInvertedSpectrogram)
            self.showInvSpec.setCheckable(True)
            self.showInvSpec.setChecked(False)

        # if self.DOC==False:
        # self.redoaxis = specMenu.addAction("Make frequency axis tight", self.redoFreqAxis)

        # specMenu.addSeparator()
        specMenu.addAction("Change spectrogram parameters",self.showSpectrogramDialog)

        self.showFundamental = specMenu.addAction("Show fundamental frequency", self.showFundamentalFreq,"Ctrl+F")
        self.showFundamental.setCheckable(True)
        self.showFundamental.setChecked(False)

        specMenu.addSeparator()
        self.readonly = specMenu.addAction("Make read only",self.makeReadOnly,"Ctrl+R")
        self.readonly.setCheckable(True)
        self.readonly.setChecked(self.config['readOnly'])
        
        specMenu.addSeparator()
        specMenu.addAction("Interface settings", self.changeSettings)

        actionMenu = self.menuBar().addMenu("&Actions")
        actionMenu.addAction("&Delete all segments", self.deleteAll, "Ctrl+D")
        actionMenu.addSeparator()
        actionMenu.addAction("Denoise",self.denoiseDialog,"Ctrl+N")
        #actionMenu.addAction("Find matches",self.findMatches)
        if self.DOC==False:
            actionMenu.addAction("Filter spectrogram",self.medianFilterSpec)
            actionMenu.addAction("Denoise spectrogram",self.denoiseImage)
        actionMenu.addSeparator()
        actionMenu.addAction("Segment",self.segmentationDialog,"Ctrl+S")
        if self.DOC == False:
            actionMenu.addAction("Classify segments",self.classifySegments,"Ctrl+C")
        actionMenu.addSeparator()
        #self.showAllTick = actionMenu.addAction("Show all pages", self.showAllCheck)
        #self.showAllTick.setCheckable(True)
        #self.showAllTick.setChecked(self.config['showAllPages'])
        actionMenu.addAction("Check segments [All segments]",self.humanClassifyDialog1,"Ctrl+1")
        actionMenu.addAction("Check segments [Choose species]",self.humanClassifyDialog2,"Ctrl+2")
        actionMenu.addAction("Human Review [Choose species]", self.humanReview2)
        actionMenu.addSeparator()
        actionMenu.addAction("Export segments to Excel",self.exportSeg)
        actionMenu.addSeparator()
        actionMenu.addAction("Save as image",self.saveImage,"Ctrl+I")
        actionMenu.addAction("Save selected sound", self.save_selected_sound)
        actionMenu.addSeparator()
        actionMenu.addAction("Put docks back",self.dockReplace)

        helpMenu = self.menuBar().addMenu("&Help")
        #aboutAction = QAction("About")
        helpMenu.addAction("Help",self.showHelp,"Ctrl+H")
        helpMenu.addAction("Cheat Sheet", self.showCheatSheet)
        helpMenu.addSeparator()
        helpMenu.addAction("About",self.showAbout,"Ctrl+A")
        if platform.system() == 'Darwin':
            helpMenu.addAction("About",self.showAbout,"Ctrl+A")

    def showAbout(self):
        """ Create the About Message Box"""
        msg = QMessageBox()
        msg.setIconPixmap(QPixmap("img\AviaNZ.png"))
        msg.setWindowIcon(QIcon('img/Avianz.ico'))
        msg.setText("The AviaNZ Program, v0.10 (June 2017)")
        msg.setInformativeText("By Stephen Marsland, Massey University (2016--2017). With code by Nirosha Priyadarshani and input from Isabel Castro, Moira Pryde, Stuart Cockburn, Rebecca Stirnemann, Sumudu Manic Purage. \ns.r.marsland@massey.ac.nz; n.p.priyadarshani@massey.ac.nz")
        msg.setWindowTitle("About")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        return

    def showHelp(self):
        """ Show the user manual (a pdf file)"""
        # TODO: manual is not distributed as pdf now
        import webbrowser
        # webbrowser.open_new(r'file://' + os.path.realpath('./Docs/AviaNZManual.pdf'))
        webbrowser.open_new(r'http://avianz.net/docs/AviaNZManual.pdf')

    def showCheatSheet(self):
        """ Show the cheat sheet of sample spectrograms (a pdf file)"""
        import webbrowser
        webbrowser.open_new(r'file://' + os.path.realpath('./Docs/CheatSheet.pdf'))

    def createFrame(self):
        """ Creates the main window.
        This consists of a set of pyqtgraph docks with widgets in.
         d_ for docks, w_ for widgets, p_ for plots"""

        # Make the window and set its size
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.resize(1240,600)
        self.move(100,50)

        # Make the docks and lay them out
        self.d_overview = Dock("Overview",size = (1200,150))
        self.d_ampl = Dock("Amplitude",size=(1200,150))
        self.d_spec = Dock("Spectrogram",size=(1200,300))
        self.d_controls = Dock("Controls",size=(40,100))
        self.d_files = Dock("Files",size=(40,200))
        if self.extra:
            self.d_plot = Dock("Plots",size=(1200,150))

        self.area.addDock(self.d_files,'left')
        self.area.addDock(self.d_overview,'right',self.d_files)
        self.area.addDock(self.d_ampl,'bottom',self.d_overview)
        self.area.addDock(self.d_spec,'bottom',self.d_ampl)
        self.area.addDock(self.d_controls,'bottom',self.d_files)
        if self.extra:
            self.area.addDock(self.d_plot,'bottom',self.d_spec)

        # Put content widgets in the docks
        self.w_overview = pg.LayoutWidget()
        self.d_overview.addWidget(self.w_overview)
        self.w_overview1 = pg.GraphicsLayoutWidget()
        self.w_overview1.ci.layout.setContentsMargins(0.5, 1, 0.5, 1)
        self.w_overview.addWidget(self.w_overview1,row=0, col=2,rowspan=3)

        self.p_overview = self.w_overview1.addViewBox(enableMouse=False,enableMenu=False,row=0,col=0)
        self.p_overview2 = SupportClasses.ChildInfoViewBox(enableMouse=False, enableMenu=False)
        self.w_overview1.addItem(self.p_overview2,row=1,col=0)
        self.p_overview2.setXLink(self.p_overview)

        self.w_ampl = pg.GraphicsLayoutWidget()
        self.p_ampl = SupportClasses.DragViewBox(self, enableMouse=False,enableMenu=False,enableDrag=self.config['requireDrag'], thisIsAmpl=True)
        self.w_ampl.addItem(self.p_ampl,row=0,col=1)
        self.d_ampl.addWidget(self.w_ampl)

        self.w_spec = pg.GraphicsLayoutWidget()
        self.p_spec = SupportClasses.DragViewBox(self, enableMouse=False,enableMenu=False,enableDrag=self.config['requireDrag'], thisIsAmpl=False)
        self.w_spec.addItem(self.p_spec,row=0,col=1)
        self.d_spec.addWidget(self.w_spec)

        if self.extra:
            self.w_plot = pg.GraphicsLayoutWidget()
            self.p_plot = self.w_plot.addViewBox(enableMouse=False,enableMenu=False)
            self.w_plot.addItem(self.p_plot,row=0,col=1)
            self.d_plot.addWidget(self.w_plot)

        # The axes
        # Time axis has to go separately in loadFile

        self.ampaxis = pg.AxisItem(orientation='left')
        self.w_ampl.addItem(self.ampaxis,row=0,col=0)
        self.ampaxis.linkToView(self.p_ampl)
        self.ampaxis.setWidth(w=65)
        self.ampaxis.setLabel('')

        self.specaxis = pg.AxisItem(orientation='left')
        self.w_spec.addItem(self.specaxis,row=0,col=0)
        self.specaxis.linkToView(self.p_spec)
        self.specaxis.setWidth(w=65)

        if self.extra:
            # Plot window also needs an axis to make them line up
            self.plotaxis = pg.AxisItem(orientation='left')
            self.w_plot.addItem(self.plotaxis,row=0,col=0)
            self.plotaxis.linkToView(self.p_plot)
            self.plotaxis.setWidth(w=65)
            self.plotaxis.setLabel('')

        # The print out at the bottom of the spectrogram with data in
        self.pointData = pg.TextItem(color=(255,0,0),anchor=(0,0))
        self.p_spec.addItem(self.pointData)

        # The various plots
        self.overviewImage = pg.ImageItem(enableMouse=False)
        self.p_overview.addItem(self.overviewImage)
        self.amplPlot = pg.PlotDataItem()
        self.p_ampl.addItem(self.amplPlot)
        self.specPlot = pg.ImageItem()
        self.p_spec.addItem(self.specPlot)

        if self.extra:
            self.plotPlot = pg.PlotDataItem()
            self.p_plot.addItem(self.plotPlot)
            self.plotPlot2 = pg.PlotDataItem()
            self.p_plot.addItem(self.plotPlot2)
            self.plotPlot3 = pg.PlotDataItem()
            self.p_plot.addItem(self.plotPlot3)
            self.plotPlot4 = pg.PlotDataItem()
            self.p_plot.addItem(self.plotPlot4)
            self.plotPlot5 = pg.PlotDataItem()
            self.p_plot.addItem(self.plotPlot5)
            self.plotPlot6 = pg.PlotDataItem()
            self.p_plot.addItem(self.plotPlot6)
            self.plotPlot7 = pg.PlotDataItem()
            self.p_plot.addItem(self.plotPlot7)
            self.plotPlot8 = pg.PlotDataItem()
            self.p_plot.addItem(self.plotPlot8)
            self.plotPlot9 = pg.PlotDataItem()
            self.p_plot.addItem(self.plotPlot9)
            self.plotPlot10 = pg.PlotDataItem()
            self.p_plot.addItem(self.plotPlot10)
            self.plotPlot11 = pg.PlotDataItem()
            self.p_plot.addItem(self.plotPlot11)

        # Connect up the listeners
        self.p_ampl.scene().sigMouseClicked.connect(self.mouseClicked_ampl)
        # self.p_ampl.sigMouseButtonPress.connect(self.mousePressed_ampl)
        # self.p_ampl.sigMouseButtonRelease.connect(self.mouseReleased_ampl)
        #self.p_spec.sigMouseDragged.connect(self.mouseDragged_spec)
        self.p_spec.scene().sigMouseClicked.connect(self.mouseClicked_spec)
        self.p_spec.scene().sigMouseMoved.connect(self.mouseMoved)

        # The content of the other two docks
        self.w_controls = pg.LayoutWidget()
        self.d_controls.addWidget(self.w_controls)
        self.w_files = pg.LayoutWidget()
        self.d_files.addWidget(self.w_files)

        # The buttons to move through the overview
        self.leftBtn = QToolButton()
        self.leftBtn.setArrowType(Qt.LeftArrow)
        #self.connect(self.leftBtn, SIGNAL('clicked()'), self.moveLeft)
        self.leftBtn.clicked.connect(self.moveLeft)
        self.w_overview.addWidget(self.leftBtn,row=0,col=0)
        self.rightBtn = QToolButton()
        self.rightBtn.setArrowType(Qt.RightArrow)
        #self.connect(self.rightBtn, SIGNAL('clicked()'), self.moveRight)
        self.rightBtn.clicked.connect(self.moveRight)
        self.w_overview.addWidget(self.rightBtn,row=0,col=1)

        # Button to move to the next file in the list
        self.nextFileBtn=QToolButton()
        self.nextFileBtn.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaSkipForward))
        #self.connect(self.nextFileBtn, SIGNAL('clicked()'), self.openNextFile)
        self.nextFileBtn.clicked.connect(self.openNextFile)
        self.nextFileBtn.setToolTip("Open next file")
        self.w_files.addWidget(self.nextFileBtn,row=0,col=1)
        #self.w_overview.addWidget(self.nextFileBtn,row=1,colspan=2)

        # Buttons to move to next/previous five minutes
        self.prev5mins=QToolButton()
        self.prev5mins.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaSeekBackward))
        self.prev5mins.setToolTip("Previous page")
        #self.connect(self.prev5mins, SIGNAL('clicked()'), self.movePrev5mins)
        self.prev5mins.clicked.connect(self.movePrev5mins)
        self.w_overview.addWidget(self.prev5mins,row=2,col=0)
        self.next5mins=QToolButton()
        self.next5mins.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaSeekForward))
        self.next5mins.setToolTip("Next page")
        #self.connect(self.next5mins, SIGNAL('clicked()'), self.moveNext5mins)
        self.next5mins.clicked.connect(self.moveNext5mins)
        self.w_overview.addWidget(self.next5mins,row=2,col=1)
        self.placeInFileLabel = QLabel('')
        self.w_overview.addWidget(self.placeInFileLabel,row=1,colspan=2)

        # The buttons inside the controls dock
        self.playButton = QtGui.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(QtCore.QSize(20, 20))
        self.playButton.setToolTip("Play visible")
        self.playButton.clicked.connect(self.playVisible)

        self.stopButton = QtGui.QToolButton()
        self.stopButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
        self.stopButton.setIconSize(QtCore.QSize(20, 20))
        self.stopButton.setToolTip("Stop playback")
        self.stopButton.clicked.connect(self.stopPlayback)

        self.playSegButton = QtGui.QToolButton()
        self.playSegButton.setIcon(QtGui.QIcon('img/playsegment.png'))
        self.playSegButton.setIconSize(QtCore.QSize(20, 20))
        self.playSegButton.setToolTip("Play selected")
        self.playSegButton.clicked.connect(self.playSelectedSegment)
        self.playSegButton.setEnabled(False)

        self.playBandLimitedSegButton = QtGui.QToolButton()
        self.playBandLimitedSegButton.setIcon(QtGui.QIcon('img/playBandLimited.png'))
        self.playBandLimitedSegButton.setIconSize(QtCore.QSize(20, 20))
        self.playBandLimitedSegButton.setToolTip("Play selected-band limited")
        self.playBandLimitedSegButton.clicked.connect(self.playBandLimitedSegment)
        self.playBandLimitedSegButton.setEnabled(False)

        self.timePlayed = QLabel()

        # Volume control
        self.volSlider = QSlider(Qt.Horizontal)
        self.volSlider.sliderMoved.connect(self.volSliderMoved)
        self.volSlider.setRange(0,100)
        self.volSlider.setValue(50)
        self.volIcon = QLabel()
        self.volIcon.setPixmap(self.style().standardIcon(QtGui.QStyle.SP_MediaVolume).pixmap(32))

        # Brightness, and contrast sliders
        self.brightnessSlider = QSlider(Qt.Horizontal)
        self.brightnessSlider.setMinimum(0)
        self.brightnessSlider.setMaximum(100)
        self.brightnessSlider.setValue(self.config['brightness'])
        self.brightnessSlider.setTickInterval(1)
        self.brightnessSlider.valueChanged.connect(self.setColourLevels)

        self.contrastSlider = QSlider(Qt.Horizontal)
        self.contrastSlider.setMinimum(0)
        self.contrastSlider.setMaximum(100)
        self.contrastSlider.setValue(self.config['contrast'])
        self.contrastSlider.setTickInterval(1)
        self.contrastSlider.valueChanged.connect(self.setColourLevels)

        # Delete segment button
        deleteButton = QPushButton("&Delete Current Segment")
        deleteButton.clicked.connect(self.deleteSegment)

        # The spinbox for changing the width shown in the controls dock
        self.widthWindow = QDoubleSpinBox()
        self.widthWindow.setSingleStep(1.0)
        self.widthWindow.setDecimals(2)
        self.widthWindow.setValue(self.config['windowWidth'])
        self.widthWindow.valueChanged[float].connect(self.changeWidth)

        # Place all these widgets in the Controls dock
        self.w_controls.addWidget(self.playButton,row=0,col=0)
        self.w_controls.addWidget(self.stopButton,row=0,col=1)
        self.w_controls.addWidget(self.playSegButton,row=0,col=2)
        self.w_controls.addWidget(self.playBandLimitedSegButton,row=0,col=3)
        self.w_controls.addWidget(self.timePlayed,row=1,col=0, colspan=3)
        self.w_controls.addWidget(self.volIcon, row=2, col=0)
        self.w_controls.addWidget(self.volSlider, row=2, col=1, colspan=2)
        self.w_controls.addWidget(QLabel("Brightness"),row=3,col=0,colspan=3)
        self.w_controls.addWidget(self.brightnessSlider,row=4,col=0,colspan=3)
        self.w_controls.addWidget(QLabel("Contrast"),row=5,col=0,colspan=3)
        self.w_controls.addWidget(self.contrastSlider,row=6,col=0,colspan=3)
        self.w_controls.addWidget(deleteButton,row=7,col=0,colspan=3)
        self.w_controls.addWidget(QLabel('Visible window (seconds)'),row=8,col=0,colspan=3)
        self.w_controls.addWidget(self.widthWindow,row=9,col=0,colspan=3)


        # The slider to show playback position
        # This is hidden, but controls the moving bar
        self.playSlider = QSlider(Qt.Horizontal)
        # self.playSlider.sliderReleased.connect(self.playSliderMoved)
        self.playSlider.setVisible(False)
        self.d_spec.addWidget(self.playSlider)
        self.bar = pg.InfiniteLine(angle=90, movable=True, pen={'color': 'c', 'width': 3})

        # A slider to move through the file easily
        self.scrollSlider = QScrollBar(Qt.Horizontal)
        self.scrollSlider.valueChanged.connect(self.scroll)
        self.d_spec.addWidget(self.scrollSlider)

        # List to hold the list of files
        self.listFiles = QListWidget()
        self.listFiles.setMinimumWidth(150)
        #self.listFiles.connect(self.listFiles, SIGNAL('itemDoubleClicked(QListWidgetItem*)'), self.listLoadFile)
        self.listFiles.itemDoubleClicked.connect(self.listLoadFile)

        self.w_files.addWidget(QLabel('Double click to open'),row=0,col=0)
        self.w_files.addWidget(QLabel('Red names have segments'),row=1,col=0)
        self.w_files.addWidget(self.listFiles,row=2,colspan=2)

        # The context menu (drops down on mouse click) to select birds
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.menuBirdList = QMenu()
        self.menuBird2 = self.menuBirdList.addMenu('Other')
        self.fillBirdList()

        # Make the colours that are used in the interface
        # The dark ones are to draw lines instead of boxes
        self.ColourSelected = QtGui.QColor(self.config['ColourSelected'][0], self.config['ColourSelected'][1], self.config['ColourSelected'][2], self.config['ColourSelected'][3])
        self.ColourNamed = QtGui.QColor(self.config['ColourNamed'][0], self.config['ColourNamed'][1], self.config['ColourNamed'][2], self.config['ColourNamed'][3])
        self.ColourNone = QtGui.QColor(self.config['ColourNone'][0], self.config['ColourNone'][1], self.config['ColourNone'][2], self.config['ColourNone'][3])
        self.ColourPossible = QtGui.QColor(self.config['ColourPossible'][0], self.config['ColourPossible'][1], self.config['ColourPossible'][2], self.config['ColourPossible'][3])

        self.ColourSelectedDark = QtGui.QColor(self.config['ColourSelected'][0], self.config['ColourSelected'][1], self.config['ColourSelected'][2], 255)
        self.ColourNamedDark = QtGui.QColor(self.config['ColourNamed'][0], self.config['ColourNamed'][1], self.config['ColourNamed'][2], 255)
        self.ColourNoneDark = QtGui.QColor(self.config['ColourNone'][0], self.config['ColourNone'][1], self.config['ColourNone'][2], 255)
        self.ColourPossibleDark = QtGui.QColor(self.config['ColourPossible'][0], self.config['ColourPossible'][1], self.config['ColourPossible'][2], 255)

        # Hack to get the type of an ROI
        p_spec_r = SupportClasses.ShadedRectROI(0, 0)
        self.ROItype = type(p_spec_r)

        # Listener for key presses
        self.p_ampl.keyPressed.connect(self.handleKey)
        self.p_spec.keyPressed.connect(self.handleKey)

        # Store the state of the docks in case the user wants to reset it
        self.state = self.area.saveState()

        # Function calls to check if should show various parts of the interface, whether dragging boxes or not
        self.useAmplitudeCheck()
        self.useFilesCheck()
        self.showOverviewSegsCheck()
        self.dragRectsTransparent()
        self.showPointerDetailsCheck()
        self.dragRectanglesCheck()

        # add statusbar
        self.statusLeft = QLabel("Left")
        self.statusLeft.setFrameStyle(QFrame.Panel) #,QFrame.Sunken)
        self.statusMid = QLabel("????")
        self.statusMid.setFrameStyle(QFrame.Panel) #,QFrame.Sunken)
        self.statusRight = QLabel("")
        self.statusRight.setAlignment(Qt.AlignRight)
        self.statusRight.setFrameStyle(QFrame.Panel) #,QFrame.Sunken)
        # Style
        statusStyle='QLabel {border:transparent}'
        self.statusLeft.setStyleSheet(statusStyle)
        # self.statusMid.setStyleSheet(statusStyle)
        self.statusRight.setStyleSheet(statusStyle)
        self.statusBar().addPermanentWidget(self.statusLeft,1)
        # self.statusBar().addPermanentWidget(self.statusMid,1)
        self.statusBar().addPermanentWidget(self.statusRight,1)

        # Set the message in the status bar
        self.statusLeft.setText("Ready")

        # Plot everything
        if not self.CLI:
            self.show()

    #def keyPressEvent(self,ev):
    #    """ Listener to handle keypresses and emit a keypress event, which is dealt with by handleKey()"""
    #    #self.emit(SIGNAL("keyPressed"),ev)
    #    print "here"
    #    print ev.key()
    #    self.keyPressed.emit(ev)

    def handleKey(self,ev):
        """ Handle keys pressed during program use.
        These are:
            backspace to delete a segment
            escape to pause playback """
        if ev == Qt.Key_Backspace:
            self.deleteSegment()
        elif ev == Qt.Key_Escape and self.media_obj.isPlaying():
            self.stopPlayback()

    def fillBirdList(self,unsure=False):
        """ Sets the contents of the context menu.
        The first 20 items are in the first menu, the next in a second menu.
        This is called a lot because the order of birds in the list changes since the last choice
        is moved to the top of the list. """
        self.menuBirdList.clear()
        self.menuBird2.clear()
        for item in self.config['BirdList'][:20]:
            if unsure and item != "Don't Know":
                item = item+'?'
            bird = self.menuBirdList.addAction(item)
            receiver = lambda checked, birdname=item: self.birdSelected(birdname)
            #self.connect(bird, SIGNAL("triggered()"), receiver)
            bird.triggered.connect(receiver)
            self.menuBirdList.addAction(bird)
        self.menuBird2 = self.menuBirdList.addMenu('Other')
        for item in self.config['BirdList'][20:]+['Other']:
            if unsure and item != "Don't Know" and item != "Other":
                item = item+'?'
            bird = self.menuBird2.addAction(item)
            receiver = lambda checked, birdname=item: self.birdSelected(birdname)
            #self.connect(bird, SIGNAL("triggered()"), receiver)
            bird.triggered.connect(receiver)
            self.menuBird2.addAction(bird)

    def fillFileList(self,fileName):
        """ Generates the list of files for the file listbox.
        Most of the work is to deal with directories in that list.
        It only sees *.wav files. Picks up *.data and *_1.wav files, the first to make the filenames
        red in the list, and the second to know if the files are long."""
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
                self.listitemtype = type(item)
                item.setText(file.fileName())
                if file.fileName()+'.data' in listOfDataFiles:
                    item.setForeground(Qt.red)
        if fileName:
            index = self.listFiles.findItems(fileName,Qt.MatchExactly)
            if len(index)>0:
                self.listFiles.setCurrentItem(index[0])
            else:
                index = self.listFiles.findItems(self.listOfFiles[0].fileName(),Qt.MatchExactly)
                self.listFiles.setCurrentItem(index[0])

    def resetStorageArrays(self):
        """ Called when new files are loaded.
        Resets the variables that hold the data to be saved and/or plotted. """

        # Remove the segments
        self.removeSegments()
        if hasattr(self, 'overviewImageRegion'):
            self.p_overview.removeItem(self.overviewImageRegion)

        # This is a flag to say if the next thing that the user clicks on should be a start or a stop for segmentation
        if self.started:
            # This is the second click, so should pay attention and close the segment
            # Stop the mouse motion connection, remove the drawing boxes
            if self.started_window=='a':
                try:
                    self.p_ampl.scene().sigMouseMoved.disconnect()
                except Exception:
                    pass
                self.p_ampl.removeItem(self.vLine_a)
            else:
                try:
                    self.p_spec.scene().sigMouseMoved.disconnect()
                except Exception:
                    pass
                # Add the other mouse move listener back
                self.p_spec.scene().sigMouseMoved.connect(self.mouseMoved)
                self.p_spec.removeItem(self.vLine_s)
            self.p_ampl.removeItem(self.drawingBox_ampl)
            self.p_spec.removeItem(self.drawingBox_spec)
        self.started = False
        self.startedInAmpl = False
        self.segmentsToSave = False

        # Keep track of start points and selected buttons
        self.windowStart = 0
        self.playPosition = self.windowStart
        self.prevBoxCol = self.config['ColourNone']
        self.bar.setValue(0)

        # reset playback buttons
        self.playSegButton.setEnabled(False)
        self.playBandLimitedSegButton.setEnabled(False)

        # Delete the overview segments
        for r in self.SegmentRects:
            self.p_overview2.removeItem(r)
        self.SegmentRects = []

        # Remove any fundamental frequencies drawn
        for r in self.segmentPlots:
            self.p_spec.removeItem(r)
        self.segmentPlots=[]

    def openFile(self):
        """ This handles the menu item for opening a file.
        Splits the directory name and filename out, and then passes the filename to the loader."""
        fileName, drop = QtGui.QFileDialog.getOpenFileName(self, 'Choose File', self.dirName,"Wav files (*.wav)")
        self.dirName = os.path.dirname(fileName)
        print("opening file %s" % fileName)
        self.listLoadFile(os.path.basename(fileName))

    def listLoadFile(self,current):
        """ Listener for when the user clicks on a filename (also called by openFile() )
        Prepares the program for a new file.
        Saves the segments of the current file, resets flags and calls loadFile() """

        # If there was a previous file, make sure the type of its name is OK. This is because you can get these
        # names from the file listwidget, or from the openFile dialog.
        if self.previousFile is not None:
            if type(self.previousFile) is not self.listitemtype:
                self.previousFile = self.listFiles.findItems(os.path.basename(str(self.previousFile)), Qt.MatchExactly)
                if len(self.previousFile)>0:
                    self.previousFile = self.previousFile[0]

            # if len(self.segments) > 0 or self.hasSegments:
            #     if len(self.segments) > 0:
            #         if self.segments[0][0] > -1:
            #             self.segments.insert(0, [-1, str(QTime().addSecs(self.startTime).toString('hh:mm:ss')),
            #                                      self.operator, self.reviewer, -1])
            #     else:
            #         self.segments.insert(0,
            #                              [-1, str(QTime().addSecs(self.startTime).toString('hh:mm:ss')), self.operator,
            #                               self.reviewer, -1])
            self.saveSegments()
            # self.previousFile.setForeground(Qt.red)
        self.previousFile = current
        if type(current) is self.listitemtype:
            current = current.text()
        self.resetStorageArrays()

        # Reset the media player
        self.media_obj = SupportClasses.ControllableAudio(self.audioFormat) # protects on first load
        if self.media_obj.isPlaying():
            self.media_obj.stop()
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        # Update the file list to show the right one
        i=0
        while self.listOfFiles[i].fileName() != current and i<len(self.listOfFiles)-1:
            i+=1
        if self.listOfFiles[i].isDir() or (i == len(self.listOfFiles)-1 and self.listOfFiles[i].fileName() != current):
            dir = QDir(self.dirName)
            dir.cd(self.listOfFiles[i].fileName())
            # Now repopulate the listbox
            self.dirName=str(dir.absolutePath())
            self.listFiles.clearSelection()
            self.listFiles.clearFocus()
            self.listFiles.clear()
            self.previousFile = None
            if (i == len(self.listOfFiles)-1) and (self.listOfFiles[i].fileName() != current):
                self.loadFile(current)
            self.fillFileList(current)
            # Show the selected file
            index = self.listFiles.findItems(os.path.basename(current), Qt.MatchExactly)
            if len(index) > 0:
                self.listFiles.setCurrentItem(index[0])
        else:
            self.loadFile(current)

    def loadFile(self,name=None):
        """ This does the work of loading a file.
        We are using wavio to do the reading. We turn the data into a float, but do not normalise it (/2^(15)).
        For 2 channels, just take the first one.
        Normalisation can cause problems for some segmentations, e.g. Harma.

        If no name is specified, loads the next section of the current file

        This method also gets the spectrogram to plot it, loads the segments from a *.data file, and
        passes the new data to any of the other classes that need it.
        Then it sets up the audio player and fills in the appropriate time data in the window, and makes
        the scroll bar and overview the appropriate lengths.
        """

        with pg.ProgressDialog("Loading..", 0, 7) as dlg:
            dlg.setCancelButton(None)
            dlg.setWindowIcon(QIcon('img/Avianz.ico'))
            dlg.setWindowTitle('AviaNZ')
            if name is not None:
                self.filename = self.dirName+'/'+name
                dlg += 1

                # Create an instance of the Signal Processing class
                if not hasattr(self, 'sp'):
                    self.sp = SignalProc.SignalProc([],0,self.config['window_width'],self.config['incr'])

                self.currentFileSection = 0

                if hasattr(self, 'timeaxis'):
                    self.w_spec.removeItem(self.timeaxis)

                # Check if the filename is in standard DOC format
                # Which is xxxxxx_xxxxxx.wav or ccxx_cccc_xxxxxx_xxxxxx.wav (c=char, x=0-9), could have _ afterward
                # So this checks for the 6 ints _ 6 ints part anywhere in string
                DOCRecording = re.search('(\d{6})_(\d{6})',name[-17:-4])

                if DOCRecording:
                    self.startTime = DOCRecording.group(2)

                    #if int(self.startTime[:2]) > 8 or int(self.startTime[:2]) < 8:
                    if int(self.startTime[:2]) > 18 or int(self.startTime[:2]) < 6: # 6pm to 6am
                        print("Night time DOC recording")
                    else:
                        print("Day time DOC recording")
                        # TODO: And modify the order of the bird list
                    self.startTime = int(self.startTime[:2]) * 3600 + int(self.startTime[2:4]) * 60 + int(self.startTime[4:6])
                    self.timeaxis = SupportClasses.TimeAxisHour(orientation='bottom',linkView=self.p_ampl)
                else:
                    self.startTime = 0

                self.timeaxis = SupportClasses.TimeAxisMin(orientation='bottom',linkView=self.p_ampl)
                self.w_spec.addItem(self.timeaxis, row=1, col=1)
                # This next line is a hack to make the axis update
                self.changeWidth(self.widthWindow.value())

                dlg += 1
            else:
                dlg += 2

            # Read in the file and make the spectrogram
            self.startRead = max(0,self.currentFileSection*self.config['maxFileShow']-self.config['fileOverlap'])
            if self.startRead == 0:
                self.lenRead = self.config['maxFileShow']+self.config['fileOverlap']
            else:
                self.lenRead = self.config['maxFileShow'] + 2*self.config['fileOverlap']
            if os.stat(self.filename).st_size != 0: # avoid files with no data (Tier 1 has 0Kb .wavs)
                wavobj = wavio.read(self.filename,self.lenRead,self.startRead)
                #wavobj = wavio.read(self.filename)

                # Parse wav format details based on file header:
                self.sampleRate = wavobj.rate
                self.audiodata = wavobj.data
                self.minFreq = 0
                self.maxFreq = self.sampleRate / 2.
                self.fileLength = wavobj.nframes
                self.audioFormat.setChannelCount(np.shape(self.audiodata)[1])
                self.audioFormat.setSampleRate(self.sampleRate)
                self.audioFormat.setSampleSize(wavobj.sampwidth*8)
                print("Detected format: %d channels, %d Hz, %d bit samples" % (self.audioFormat.channelCount(), self.audioFormat.sampleRate(), self.audioFormat.sampleSize()))

                dlg += 1

                if self.audiodata.dtype is not 'float':
                    self.audiodata = self.audiodata.astype('float')  # / 32768.0

                if np.shape(np.shape(self.audiodata))[0] > 1:
                    self.audiodata = self.audiodata[:, 0]
                self.datalength = np.shape(self.audiodata)[0]
                self.datalengthSec = float(self.datalength) / self.sampleRate
                print("Length of file is ", self.datalengthSec, " seconds (", self.datalength, "samples) loaded from ", float(self.fileLength) / self.sampleRate, "seconds (", self.fileLength, " samples) with sample rate ",self.sampleRate, " Hz.")

                if name is not None: # i.e. starting a new file, not next section
                    if self.datalength != self.fileLength:
                        print("not all of file loaded")
                        self.nFileSections = int(np.ceil(float(self.fileLength)/self.datalength))
                        self.prev5mins.setEnabled(False)
                        self.next5mins.setEnabled(True)
                    else:
                        self.nFileSections = 1
                        self.prev5mins.setEnabled(False)
                        self.next5mins.setEnabled(False)

                if self.nFileSections == 1:
                    self.placeInFileLabel.setText('')
                else:
                    self.placeInFileLabel.setText("Part "+ str(self.currentFileSection+1) + " of " + str(self.nFileSections))

                # Get the data for the main spectrogram
                sgRaw = self.sp.spectrogram(self.audiodata, self.config['window_width'],
                                            self.config['incr'], mean_normalise=True, onesided=True,
                                            multitaper=False)
                maxsg = np.min(sgRaw)
                self.sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))

                # Load any previous segments stored
                if os.path.isfile(self.filename + '.data'):
                    file = open(self.filename + '.data', 'r')
                    self.segments = json.load(file)
                    file.close()
                    if len(self.segments) > 0:
                        if self.segments[0][0] == -1:
                            self.operator = self.segments[0][2]
                            self.reviewer = self.segments[0][3]
                            del self.segments[0]
                    if len(self.segments) > 0:
                        if self.segments[0][2] > 1.5 and self.segments[0][3] > 1.5:
                            # Legacy version didn't normalise the segment data for dragged boxes
                            # This fixes it, assuming that the spectrogram was 128 pixels high (256 width window)
                            # The .5 is to take care of rounding errors
                            print("Old segments, normalising")
                            for s in self.segments:
                                s[2] = s[2]/128
                                s[3] = s[3]/128
                        self.segmentsToSave = True

                self.statusRight.setText("Operator: " + str(self.operator) + ", Reviewer: " + str(self.reviewer))

                # Update the data that is seen by the other classes
                if hasattr(self,'seg'):
                    self.seg.setNewData(self.audiodata,sgRaw,self.sampleRate,self.config['window_width'],self.config['incr'])
                else:
                    self.seg = Segment.Segment(self.audiodata, sgRaw, self.sp, self.sampleRate,
                                               self.config['window_width'], self.config['incr'])
                self.sp.setNewData(self.audiodata,self.sampleRate)

                # Delete any denoising backups from the previous file
                if hasattr(self,'audiodata_backup'):
                    self.audiodata_backup = None
                self.showFundamental.setChecked(False)
                if self.DOC == False:
                    self.showInvSpec.setChecked(False)

                self.timeaxis.setOffset(self.startRead+self.startTime)

                # Set the window size
                self.windowSize = self.config['windowWidth']
                self.widthWindow.setRange(0.5, self.datalengthSec)
    
                # Reset it if the file is shorter than the window
                if self.datalengthSec < self.windowSize:
                    self.windowSize = self.datalengthSec
                self.widthWindow.setValue(self.windowSize)
    
                self.totalTime = self.convertMillisecs(1000*self.datalengthSec)
                self.timePlayed.setText(self.convertMillisecs(0) + "/" + self.totalTime)
    
                # Load the file for playback
                self.media_obj = SupportClasses.ControllableAudio(self.audioFormat)
                self.media_obj.load(self.filename)
                # this responds to audio output timer
                self.media_obj.notify.connect(self.movePlaySlider)
    
                # Set the length of the scrollbar.
                self.scrollSlider.setRange(0,np.shape(self.sg)[0] - self.convertAmpltoSpec(self.widthWindow.value()))
                self.scrollSlider.setValue(0)
    
                # Get the height of the amplitude for plotting the box
                self.minampl = np.min(self.audiodata)+0.1*(np.max(self.audiodata)+np.abs(np.min(self.audiodata)))
                self.drawOverview()
                dlg += 1
                self.drawfigMain()
                self.setWindowTitle('AviaNZ - ' + self.filename)
                dlg += 1
                self.statusLeft.setText("Ready")

    def openNextFile(self):
        """ Listener for next file >> button.
        Get the next file in the list and call the loader. """
        i=self.listFiles.currentRow()
        if i+1<len(self.listFiles):
            self.listFiles.setCurrentRow(i+1)
            self.listLoadFile(self.listFiles.currentItem())
        else:
            # Tell the user they've finished
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("You've finished processing the folder")
            msg.setIconPixmap(QPixmap("img/Owl_done.png"))
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setWindowTitle("Last file")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def showPointerDetailsCheck(self):
        """ Listener for the menuitem that sets if detailed info should be shown when hovering over spectrogram.
        Turning this off saves lots of CPU performance."""
        self.config['showPointerDetails'] = self.showPointerDetails.isChecked()
        if self.showPointerDetails.isChecked():
            self.p_spec.scene().sigMouseMoved.connect(self.mouseMoved)
        else:
            self.p_spec.scene().sigMouseMoved.disconnect()
            self.pointData.setText("")

    def dragRectanglesCheck(self):
        """ Listener for the menuitem that says if the user is dragging rectangles or clicking on the spectrogram has
        changed state.
        Changes the pyqtgraph MouseMode.
        Also swaps the listeners. """
        self.config['dragBoxes'] = self.dragRectangles.isChecked()

    def dragRectsTransparent(self):
        """ Listener for the check menu item that decides if the user wants the dragged rectangles to have colour or not.
        It's a switch from Brush to Pen or vice versa.
        """
        if self.dragRectTransparent.isChecked():
            for box in self.listRectanglesa2:
                if type(box) == self.ROItype:
                    col = box.brush.color()
                    col.setAlpha(255)
                    box.setBrush(pg.mkBrush(None))
                    box.setPen(pg.mkPen(col,width=1))
                    box.update()
        else:
            for box in self.listRectanglesa2:
                if type(box) == self.ROItype:
                    col = box.pen.color()
                    col.setAlpha(self.ColourNamed.alpha())
                    box.setBrush(pg.mkBrush(col))
                    box.setPen(pg.mkPen(None))
                    box.update()
        self.config['transparentBoxes'] = self.dragRectTransparent.isChecked()

    def useAmplitudeCheck(self):
        """ Listener for the check menu item saying if the user wants to see the waveform.
        Does not remove the dock, just hide it. It's therefore easy to replace, but could have some performance overhead.
        """
        if self.useAmplitudeTick.isChecked():
            self.useAmplitude = True
            self.d_ampl.show()
        else:
            self.useAmplitude = False
            self.d_ampl.hide()
        self.config['showAmplitudePlot'] = self.useAmplitudeTick.isChecked()

    def useFilesCheck(self):
        """ Listener to process if the user swaps the check menu item to see the file list. """
        if self.useFilesTick.isChecked():
            self.d_files.show()
        else:
            self.d_files.hide()
        self.config['showListofFiles'] = self.useFilesTick.isChecked()

    def showOverviewSegsCheck(self):
        """ Listener to process if the user swaps the check menu item to see the overview segment boxes. """
        if self.showOverviewSegsTick.isChecked():
            self.p_overview2.show()
        else:
            self.p_overview2.hide()
        self.config['showAnnotationOverview'] = self.showOverviewSegsTick.isChecked()

    def makeReadOnly(self):
        """ Listener to process the check menu item to make the plots read only.
        Turns off the listeners for the amplitude and spectrogram plots.
        Also has to go through all of the segments, turn off the listeners, and make them unmovable.
        """
        self.config['readOnly']=self.readonly.isChecked()
        if self.readonly.isChecked():
            try:
                self.p_ampl.scene().sigMouseClicked.disconnect()
                self.p_spec.scene().sigMouseClicked.disconnect()
                self.p_spec.sigMouseDragged.disconnect()
            except Exception as e:
                print(e)
                pass
            try:
                self.p_spec.scene().sigMouseMoved.disconnect()
            except Exception:
                pass
            for rect in self.listRectanglesa1:
                if rect is not None:
                    try:
                        rect.sigRegionChangeFinished.disconnect()
                    except Exception:
                        pass
                    rect.setMovable(False)
            for rect in self.listRectanglesa2:
                if rect is not None:
                    try:
                        rect.sigRegionChangeFinished.disconnect()
                    except Exception:
                        pass
                    rect.setMovable(False)
        else:
            self.p_ampl.scene().sigMouseClicked.connect(self.mouseClicked_ampl)
            self.p_spec.scene().sigMouseClicked.connect(self.mouseClicked_spec)
            self.p_spec.scene().sigMouseMoved.connect(self.mouseMoved)
            for rect in self.listRectanglesa1:
                if rect is not None:
                    rect.sigRegionChangeFinished.connect(self.updateRegion_ampl)
                    rect.setMovable(True)
            for rect in self.listRectanglesa2:
                if rect is not None:
                    rect.sigRegionChangeFinished.connect(self.updateRegion_spec)
                    rect.setMovable(True)

    def dockReplace(self):
        """ Listener for if the docks should be replaced menu item. """
        self.area.restoreState(self.state)

    def showFundamentalFreq(self):
        """ Calls the SignalProc class to compute, and then draws the fundamental frequency.
        Uses the yin algorithm. """
        with pg.BusyCursor():
            if self.showFundamental.isChecked():
                self.statusLeft.setText("Drawing fundamental frequency...")
                pitch, y, minfreq, W = self.seg.yin()
                ind = np.squeeze(np.where(pitch>minfreq))
                pitch = pitch[ind]
                ind = ind*W/(self.config['window_width'])
                x = (pitch*2./self.sampleRate*np.shape(self.sg)[1]).astype('int')

                from scipy.signal import medfilt
                x = medfilt(x,15)

                # Get the individual pieces
                segs = self.seg.identifySegments(ind,maxgap=10,minlength=5)
                count = 0
                self.segmentPlots = []
                for s in segs:
                    count += 1
                    s[0] = s[0] * self.sampleRate / float(self.config['incr'])
                    s[1] = s[1] * self.sampleRate / float(self.config['incr'])
                    i = np.where((ind>s[0]) & (ind<s[1]))
                    self.segmentPlots.append(pg.PlotDataItem())
                    self.segmentPlots[-1].setData(ind[i], x[i], pen=pg.mkPen('r', width=2))
                    self.p_spec.addItem(self.segmentPlots[-1])
            else:
                self.statusLeft.setText("Removing fundamental frequency...")
                for r in self.segmentPlots:
                    self.p_spec.removeItem(r)
            self.statusLeft.setText("Ready")

    # def showFundamentalFreq2(self):
    #     # This and the next function are to check whether or not yaapt or harvest are any good. They aren't.
    #     import pYAAPT
    #     import basic_tools
    #     # Actually this is a pain, since it either gives back a value for each amplitude sample, or for it's own weird windows
    #     if self.showFundamental2.isChecked():
    #         y = basic_tools.SignalObj(self.filename)
    #         x = pYAAPT.yaapt(y)
    #         self.yinRois = []
    #         for r in range(len(x)):
    #             self.yinRois.append(pg.CircleROI([ind[r],x[r]], [2,2], pen=(4, 9),movable=False))
    #         for r in self.yinRois:
    #             self.p_spec.addItem(r)
    #     else:
    #         for r in self.yinRois:
    #             self.p_spec.removeItem(r)
    #
    # def showFundamentalFreq3(self):
    #     # Harvest
    #     import audio_tools
    #     if self.showFundamental2.isChecked():
    #         p, f, t, fa = audio_tools.harvest(self.audiodata,self.sampleRate)
    #         ind = f/self.config['window_width']
    #         x = (p*2./self.sampleRate*np.shape(self.sg)[1]).astype('int')
    #
    #         self.yinRois = []
    #         for r in range(len(x)):
    #             self.yinRois.append(pg.CircleROI([ind[r],x[r]], [2,2], pen=(4, 9),movable=False))
    #         for r in self.yinRois:
    #             self.p_spec.addItem(r)
    #     else:
    #         for r in self.yinRois:
    #             self.p_spec.removeItem(r)

    def showInvertedSpectrogram(self):
        """ Listener for the menu item that draws the spectrogram of the waveform of the inverted spectrogram."""
        if self.showInvSpec.isChecked():
            sgRaw = self.sp.show_invS()
        else:
            sgRaw = self.sp.spectrogram(self.audiodata, mean_normalise=True, onesided=True,
                                         multitaper=False)
        maxsg = np.min(sgRaw)
        self.sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))
        self.overviewImage.setImage(self.sg)
        self.specPlot.setImage(self.sg)

    def medianFilterSpec(self):
        """ Median filter the spectrogram. To be used in conjunction with spectrogram inversion. """
        # TODO: Play with this
        with pg.BusyCursor():
            self.statusLeft.setText("Filtering...")
            from scipy.ndimage.filters import median_filter
            median_filter(self.sg,size=(100,20))
            self.specPlot.setImage(self.sg)
            self.statusLeft.setText("Ready")

    def denoiseImage(self):
        """ Denoise the spectrogram. To be used in conjunction with spectrogram inversion. """
        #from cv2 import fastNlMeansDenoising
        #sg = np.array(self.sg/np.max(self.sg)*255,dtype = np.uint8)
        #sg = fastNlMeansDenoising(sg,10,7,21)
        #self.specPlot.setImage(sg)
# ==============
# Code for drawing and using the main figure

    def convertAmpltoSpec(self,x):
        """ Unit conversion """
        return x*self.sampleRate/float(self.config['incr'])

    def convertSpectoAmpl(self,x):
        """ Unit conversion """
        return x*self.config['incr']/float(self.sampleRate)

    def convertMillisecs(self,millisecs):
        """ Unit conversion """
        seconds = (millisecs / 1000) % 60
        minutes = (millisecs / (1000 * 60)) % 60
        return "%02d" % minutes+":"+"%02d" % seconds

    def drawOverview(self):
        """ On loading a new file, update the overview figure to show where you are up to in the file.
        Also, compute the new segments for the overview, and make sure that the listeners are connected
        for clicks on them. """
        self.overviewImage.setImage(self.sg)
        self.overviewImageRegion = pg.LinearRegionItem()
        self.p_overview.addItem(self.overviewImageRegion, ignoreBounds=True)
        self.overviewImageRegion.setRegion([0, self.convertAmpltoSpec(self.widthWindow.value())])
        self.overviewImageRegion.sigRegionChangeFinished.connect(self.updateOverview)

        # Three y values are No. not known, No. known, No. possible
        # widthOverviewSegment is in seconds
        numSegments = int(np.ceil(np.shape(self.sg)[0]/self.convertAmpltoSpec(self.config['widthOverviewSegment'])))
        self.widthOverviewSegment = int(float(np.shape(self.sg)[0])/numSegments)

        self.overviewSegments = np.zeros((numSegments,3))
        for i in range(numSegments):
            r = SupportClasses.ClickableRectItem(i*self.widthOverviewSegment, 0, self.widthOverviewSegment, 0.5)
            r.setPen(pg.mkPen('k'))
            r.setBrush(pg.mkBrush('w'))
            self.SegmentRects.append(r)
            self.p_overview2.addItem(r)
        self.p_overview2.sigChildMessage.connect(self.overviewSegmentClicked)

    def overviewSegmentClicked(self,x):
        """ Listener for an overview segment being clicked on.
        Work out which one, and move the region appropriately. Calls updateOverview to do the work. """
        minX, maxX = self.overviewImageRegion.getRegion()
        self.overviewImageRegion.setRegion([x, x+maxX-minX])
        self.updateOverview()
        self.playPosition = int(self.convertSpectoAmpl(x)*1000.0)

    def updateOverview(self):
        """ Listener for when the overview box is changed. Also called by overviewSegmentClicked().
        Does the work of keeping all the plots in the right place as the overview moves.
        It sometimes updates a bit slowly. """

        #3/4/18: Want to stop it moving past either end
        # Need to disconnect the listener and reconnect it to avoid a recursive call
        minX, maxX = self.overviewImageRegion.getRegion()
        #print minX, maxX
        if minX<0:
            l = maxX-minX
            minX=0.0
            maxX=minX+l
            self.overviewImageRegion.sigRegionChangeFinished.disconnect()
            self.overviewImageRegion.setRegion([minX,maxX])
            self.overviewImageRegion.sigRegionChangeFinished.connect(self.updateOverview)
        if maxX>len(self.sg):
            l = maxX-minX
            maxX=float(len(self.sg))
            minX=maxX-l
            self.overviewImageRegion.sigRegionChangeFinished.disconnect()
            self.overviewImageRegion.setRegion([minX,maxX])
            self.overviewImageRegion.sigRegionChangeFinished.connect(self.updateOverview)

        self.widthWindow.setValue(self.convertSpectoAmpl(maxX-minX))
        self.p_ampl.setXRange(self.convertSpectoAmpl(minX), self.convertSpectoAmpl(maxX), update=True, padding=0)
        self.p_spec.setXRange(minX, maxX, update=True, padding=0)

        # I know the next two lines SHOULD be unnecessary. But they aren't!
        self.p_ampl.setXRange(self.convertSpectoAmpl(minX), self.convertSpectoAmpl(maxX), padding=0)
        self.p_spec.setXRange(minX, maxX, padding=0)

        if self.extra:
            self.p_plot.setXRange(self.convertSpectoAmpl(minX), self.convertSpectoAmpl(maxX), padding=0)
        # self.setPlaySliderLimits(1000.0*self.convertSpectoAmpl(minX),1000.0*self.convertSpectoAmpl(maxX))
        self.scrollSlider.setValue(minX)
        self.pointData.setPos(minX,0)
        self.config['windowWidth'] = self.convertSpectoAmpl(maxX-minX)
        # self.saveConfig = True
        self.timeaxis.update()
        pg.QtGui.QApplication.processEvents()

    def drawfigMain(self):
        """ Draws the main amplitude and spectrogram plots and any segments on them.
        Has to do some work to get the axis labels correct.
        """
        self.amplPlot.setData(np.linspace(0.0,self.datalengthSec,num=self.datalength,endpoint=True),self.audiodata)
        self.timeaxis.setLabel('')
        self.specPlot.setImage(self.sg)
        self.setColourMap(self.config['cmap'])
        self.setColourLevels()

        # Sort out the spectrogram frequency axis
        # The constants here are divided by 1000 to get kHz, and then remember the top is sampleRate/2
        FreqRange = (self.maxFreq-self.minFreq)/1000.
        self.specaxis.setTicks([[(0,self.minFreq/1000.),(np.shape(self.sg)[1]/4,self.minFreq/1000.+FreqRange/4.),(np.shape(self.sg)[1]/2,self.minFreq/1000.+FreqRange/2.),(3*np.shape(self.sg)[1]/4,self.minFreq/1000.+3*FreqRange/4.),(np.shape(self.sg)[1],self.minFreq/1000.+FreqRange)]])
        self.specaxis.setLabel('kHz')

        self.updateOverview()
        self.textpos = np.shape(self.sg)[1] + self.config['textoffset']

        # If there are segments, show them
        for count in range(len(self.segments)):
            self.addSegment(self.segments[count][0], self.segments[count][1],self.segments[count][2],self.segments[count][3],self.segments[count][4],False,count)

        # This is the moving bar for the playback
        if not hasattr(self,'bar'):
            self.bar = pg.InfiniteLine(angle=90, movable=True, pen={'color': 'c', 'width': 3})
        self.p_spec.addItem(self.bar, ignoreBounds=True)
        self.bar.sigPositionChangeFinished.connect(self.barMoved)

        if self.extra:
            # Extra stuff to show test plots
            #self.plotPlot.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=self.datalength,endpoint=True),self.audiodata)
            pproc = SupportClasses.postProcess(self.audiodata,self.sampleRate)
            #energy, e = pproc.detectClicks()
            #energy, e = pproc.eRatioConfd()
            #if len(clicks)>0:
            #self.plotPlot.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(self.sg)[0],endpoint=True),energy)
            #self.plotPlot2.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(self.sg)[0],endpoint=True),e*np.ones(np.shape(self.sg)[0]))
            #self.plotPlot2.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(self.sg)[0],endpoint=True),e2)

            #ws = WaveletSegment.WaveletSegment(species='kiwi')
            #e = ws.computeWaveletEnergy(self.audiodata,self.sampleRate)

            # # Call MFCC in Features and plot some of them :)
            # ff = Features.Features(self.audiodata,self.sampleRate)
            # e = ff.get_mfcc()
            # print np.shape(e)
            # print np.sum(e, axis=0)
            # print e[0,:]
            #
            # # self.plotPlot.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),np.sum(e,axis=0))
            # # self.plotPlot.setPen(fn.mkPen('k'))
            # # self.plotPlot2.setData(np.linspace(0.0, float(self.datalength) / self.sampleRate, num=np.shape(e)[1], endpoint=True), e[0,:])
            # # self.plotPlot2.setPen(fn.mkPen('c'))
            # e1 = e[1,:]
            # e1 = (e[1,:]- np.mean(e[1,:]))/np.std(e[1,:])
            # self.plotPlot2.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e1)
            # self.plotPlot2.setPen(fn.mkPen('r'))
            # mean = np.mean(e1)
            # std = np.std(e1)
            # thr = mean - 2 * std
            # thr = np.ones((1, 100)) * thr
            # self.plotPlot7.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=100, endpoint=True), thr[0,:])
            # self.plotPlot7.setPen(fn.mkPen('c'))
            # # self.plotPlot3.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e[2,:])
            # # self.plotPlot3.setPen(fn.mkPen('c'))
            # # self.plotPlot4.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e[3,:])
            # # self.plotPlot4.setPen(fn.mkPen('r'))
            # # self.plotPlot5.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e[4,:])
            # # self.plotPlot5.setPen(fn.mkPen('g'))
            # # self.plotPlot6.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e[5,:])
            # # self.plotPlot6.setPen(fn.mkPen('g'))
            # # self.plotPlot7.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e[6,:])
            # # self.plotPlot7.setPen(fn.mkPen('g'))
            # # self.plotPlot8.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e[7,:])
            # # self.plotPlot8.setPen(fn.mkPen('g'))
            # # self.plotPlot9.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e[8,:])
            # # self.plotPlot9.setPen(fn.mkPen('g'))
            # # self.plotPlot10.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e[9,:])
            # # self.plotPlot10.setPen(fn.mkPen('g'))
            # # self.plotPlot11.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e[10,:])
            # # self.plotPlot11.setPen(fn.mkPen('c'))

            # # plot eRatio
            post = SupportClasses.postProcess(self.audiodata, self.sampleRate, [])
            # e = post.eRatioConfd([], AviaNZ_extra=True)
            # # print np.shape(e)
            # # print e[0]
            # # print np.shape(e[0])[0]
            # self.plotPlot.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e[0])[0],endpoint=True),e[0])
            # self.plotPlot.setPen(fn.mkPen('b'))

            # # plot wind/rain
            # wind, rain, mean_rain = post.wind()
            # wind = np.ones((1,100))*wind
            # # rain = np.ones((1,100))*rain    # rain is SNR
            # # thr = np.ones((1,100))*3.5      # rain SNR thr is 3.5
            # # mean_rain = np.ones((1, 100)) * mean_rain
            # # mean_rain_thr = np.ones((1,100)) * 1e-6     # rain mean thr is 1e-6
            # wind_thr = np.ones((1, 100)) * 1e-8
            # # print np.shape(wind)
            # # print rain[0,:]
            # self.plotPlot3.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=100,endpoint=True), wind[0,:])
            # self.plotPlot3.setPen(fn.mkPen('r'))
            # self.plotPlot4.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=100,endpoint=True), wind_thr[0,:])
            # self.plotPlot4.setPen(fn.mkPen('k'))
            # # self.plotPlot5.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=100,endpoint=True), mean_rain[0,:])
            # # self.plotPlot5.setPen(fn.mkPen('b'))
            # # self.plotPlot6.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=100,endpoint=True), mean_rain_thr[0,:])
            # # self.plotPlot6.setPen(fn.mkPen('g'))

            # # plot wavelet
            # ws = WaveletSegment.WaveletSegment(self.audiodata, self.sampleRate)
            # e = ws.computeWaveletEnergy(self.audiodata, self.sampleRate)
            # print np.shape(e)
            # print np.shape(e)[1]
            # self.plotPlot3.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e[1,:])
            # self.plotPlot3.setPen(fn.mkPen('r'))
            # mean = np.mean(e[1,:])
            # std = np.std(e[1,:])
            # thr = mean + 2.5 * std
            # thr = np.ones((1, 100)) * thr
            # self.plotPlot7.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=100, endpoint=True), thr[0,:])
            # self.plotPlot7.setPen(fn.mkPen('c'))
            #
            # # self.plotPlot4.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e[2,:])
            # # self.plotPlot4.setPen(fn.mkPen('g'))
            # # self.plotPlot5.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e[0,:])
            # # self.plotPlot5.setPen(fn.mkPen('b'))
            # # self.plotPlot6.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=np.shape(e)[1],endpoint=True),e[14,:])
            # # self.plotPlot6.setPen(fn.mkPen('k'))

        QApplication.processEvents()

    def updateRegion_spec(self):
        """ This is the listener for when a segment box is changed in the spectrogram.
        It updates the position of the matching box, and also the text within it.
        """
        sender = self.sender()
        i = 0
        while self.listRectanglesa2[i] != sender and i<len(self.listRectanglesa2):
            i = i+1
        if i==len(self.listRectanglesa2):
            print("segment not found!")
        else:
            if type(sender) == self.ROItype:
                # update the box visual
                x1 = self.convertSpectoAmpl(sender.pos()[0])
                x2 = self.convertSpectoAmpl(sender.pos()[0]+sender.size()[0])
                self.segments[i][2] = sender.pos()[1]/np.shape(self.sg)[1]
                self.segments[i][3] = (sender.pos()[1]+sender.size()[1])/np.shape(self.sg)[1]
                self.listLabels[i].setPos(sender.pos()[0], self.textpos)
            else:
                # update the segment visual
                x1 = self.convertSpectoAmpl(sender.getRegion()[0])
                x2 = self.convertSpectoAmpl(sender.getRegion()[1])
                self.listLabels[i].setPos(sender.getRegion()[0], self.textpos)
            # update the amplitude visual
            self.listRectanglesa1[i].setRegion([x1,x2])
            self.segmentsToSave = True

            # update the actual segment list and overview boxes
            startold = self.segments[i][0]
            endold = self.segments[i][1]
            species = self.segments[i][4]
            self.refreshOverviewWith(startold, endold, species, delete=True)

            self.segments[i][0] = x1 + self.startRead
            self.segments[i][1] = x2 + self.startRead
            self.refreshOverviewWith(self.segments[i][0], self.segments[i][1], species)

    def updateRegion_ampl(self):
        """ This is the listener for when a segment box is changed in the waveform plot.
        It updates the position of the matching box, and also the text within it.
        """
        sender = self.sender()
        i = 0
        while self.listRectanglesa1[i] != sender and i<len(self.listRectanglesa1):
            i = i+1
        if i==len(self.listRectanglesa1):
            print("segment not found!")
        else:
            x1 = self.convertAmpltoSpec(sender.getRegion()[0])
            x2 = self.convertAmpltoSpec(sender.getRegion()[1])

            # if self.listRectanglesa2[i] is not None: - this shouldn't happen anyway
            if type(self.listRectanglesa2[i]) == self.ROItype:
                # update the box visual
                y1 = self.listRectanglesa2[i].pos().y()
                y2 = self.listRectanglesa2[i].size().y()
                self.listRectanglesa2[i].setPos(pg.Point(x1,y1))
                self.listRectanglesa2[i].setSize(pg.Point(x2-x1,y2))
            else:
                # update the segment visual
                self.listRectanglesa2[i].setRegion([x1,x2])
            self.listLabels[i].setPos(x1,self.textpos)
            self.segmentsToSave = True
            # how does this update amplitude visual??

            # update the actual segment list and overview boxes
            startold = self.segments[i][0]
            endold = self.segments[i][1]
            species = self.segments[i][4]
            self.refreshOverviewWith(startold, endold, species, delete=True)

            self.segments[i][0] = sender.getRegion()[0] + self.startRead
            self.segments[i][1] = sender.getRegion()[1] + self.startRead
            self.refreshOverviewWith(self.segments[i][0], self.segments[i][1], species)

    def refreshOverviewWith(self, startpoint, endpoint, species, delete=False):
        """Recalculates the overview box colors and refreshes their display.
        To be used when segments are added, deleted or moved."""
        # Work out which overview segment this segment is in (could be more than one)
        # min is to remove possible rounding error
        inds = int(float(self.convertAmpltoSpec(startpoint)) / self.widthOverviewSegment)
        inde = min(int(float(self.convertAmpltoSpec(endpoint)) / self.widthOverviewSegment),len(self.overviewSegments)-1)
        if species == "Don't Know" or type(species) is int:
            brush = self.ColourNone
            if delete:
                self.overviewSegments[inds:inde+1,0] -= 1
            else:
                self.overviewSegments[inds:inde+1,0] += 1

        if species[-1] == '?':
            brush = self.ColourPossible
            if delete:
                self.overviewSegments[inds:inde + 1, 2] -= 1
            else:
                self.overviewSegments[inds:inde + 1, 2] += 1
        else:
            brush = self.ColourNamed
            if delete:
                self.overviewSegments[inds:inde + 1, 1] -= 1
            else:
                self.overviewSegments[inds:inde + 1, 1] += 1

        # set the colour of these boxes in the overview
        for box in range(inds, inde + 1):
            if self.overviewSegments[box,0] > 0:
                self.SegmentRects[box].setBrush(self.ColourNone)
            elif self.overviewSegments[box,2] > 0:
                self.SegmentRects[box].setBrush(self.ColourPossible)
            elif self.overviewSegments[box,1] > 0:
                self.SegmentRects[box].setBrush(self.ColourNamed)
            else:
                # boxes w/o segments
                self.SegmentRects[box].setBrush(pg.mkBrush('w'))
            self.SegmentRects[box].update()

    def addSegment(self,startpoint,endpoint,y1=0,y2=0,species=None,saveSeg=True,index=-1):
        """ When a new segment is created, does the work of creating it and connecting its
        listeners. Also updates the relevant overview segment.
        startpoint, endpoint are in amplitude coordinates, while y1, y2 should be between 0 and 1, meaning 0 and np.shape(sg)[1].
        saveSeg means that we are drawing the saved ones. Need to check that those ones fit into
        the current window, can assume the other do, but have to save their times correctly.
        If a segment is too long for the current section, truncates it.
        """
        print("segment added at %d-%d" % (startpoint, endpoint))
        if not saveSeg:
            timeRangeStart = self.startRead
            timeRangeEnd = min(self.startRead + self.lenRead, float(self.fileLength) / self.sampleRate)

            if startpoint >= timeRangeStart and endpoint <= timeRangeEnd:
                show = True
                # Put the startpoint and endpoint in the right range
                startpoint = startpoint - timeRangeStart
                endpoint = endpoint - timeRangeStart
            elif startpoint >= timeRangeStart and endpoint > timeRangeEnd:
                startpoint = startpoint - timeRangeStart
                endpoint = timeRangeEnd - timeRangeStart
                show = True
            elif startpoint < timeRangeStart and endpoint >= timeRangeEnd:
                startpoint = 0
                endpoint = endpoint - timeRangeStart
                show = True
            else:
                # not sure why these shouldn't be shown?
                print("Warning: a segment was not shown")
                show = False
        else:
            self.segmentsToSave = True
            show = True

        if show:
            # This is one we want to show
            # Get the name and colour sorted
            if species is None or species=="Don't Know":
                species = "Don't Know"
                brush = self.ColourNone
            elif species[-1]=='?':
                brush = self.ColourPossible
            else:
                brush = self.ColourNamed

            self.refreshOverviewWith(startpoint, endpoint, species)
            self.prevBoxCol = brush

            # Make sure startpoint and endpoint are in the right order
            if startpoint > endpoint:
                temp = startpoint
                startpoint = endpoint
                endpoint = temp
            if y1 > y2:
                temp = y1
                y1 = y2
                y2 = temp

            # Add the segment in both plots and connect up the listeners
            p_ampl_r = pg.LinearRegionItem(brush=brush)
            self.p_ampl.addItem(p_ampl_r, ignoreBounds=True)
            p_ampl_r.setRegion([startpoint, endpoint])
            p_ampl_r.sigRegionChangeFinished.connect(self.updateRegion_ampl)

            if y1==0 and y2==0:
                p_spec_r = pg.LinearRegionItem(brush = brush)
                p_spec_r.setRegion([self.convertAmpltoSpec(startpoint), self.convertAmpltoSpec(endpoint)])
            else:
                startpointS = QPointF(self.convertAmpltoSpec(startpoint),y1*np.shape(self.sg)[1])
                endpointS = QPointF(self.convertAmpltoSpec(endpoint),y2*np.shape(self.sg)[1])
                p_spec_r = SupportClasses.ShadedRectROI(startpointS, endpointS - startpointS)
                if self.dragRectTransparent.isChecked():
                    col = self.prevBoxCol.rgb()
                    col = QtGui.QColor(col)
                    col.setAlpha(255)
                    p_spec_r.setBrush(None)
                    p_spec_r.setPen(pg.mkPen(col,width=1))
                else:
                    p_spec_r.setBrush(pg.mkBrush(self.prevBoxCol))
                    p_spec_r.setPen(pg.mkPen(None))
            self.p_spec.addItem(p_spec_r, ignoreBounds=True)
            p_spec_r.sigRegionChangeFinished.connect(self.updateRegion_spec)

            # Put the text into the box
            label = pg.TextItem(text=species, color='k')
            self.p_spec.addItem(label)
            label.setPos(self.convertAmpltoSpec(startpoint), self.textpos)

            # Add the segments to the relevent lists
            self.listRectanglesa1.append(p_ampl_r)
            self.listRectanglesa2.append(p_spec_r)
            self.listLabels.append(label)

            if saveSeg:
                # Add the segment to the data
                # Increment the time to be correct for the current section of the file
                self.segments.append([startpoint+self.startRead, endpoint+self.startRead, y1, y2, species])
            # mark this as the current segment
            self.box1id = len(self.segments) - 1
        else:
            # Add a None element into the array so that the correct boxids work
            self.listRectanglesa1.append(None)
            self.listRectanglesa2.append(None)
            self.listLabels.append(None)

    def deleteSegment(self,id=-1,hr=False):
        """ Listener for delete segment button, or backspace key. Also called when segments are deleted by the
        human classify dialogs.
        Deletes the segment that is selected, otherwise does nothing.
        Updates the overview segments as well.
        """

        if not hr and (id<0 or not id):
            id = self.box1id

        #if id<0 or not id:
            # delete selected
            #id = self.box1id

        if id>-1:
            print("segment %d deleted" % id)
            startpoint = self.segments[id][0]-self.startRead
            endpoint = self.segments[id][1]-self.startRead
            species = self.segments[id][4]

            self.refreshOverviewWith(startpoint, endpoint, species, delete=True)

            if self.listRectanglesa1[id] is not None:
                self.p_ampl.removeItem(self.listRectanglesa1[id])
                self.p_spec.removeItem(self.listRectanglesa2[id])
                self.p_spec.removeItem(self.listLabels[id])
            del self.listLabels[id]
            del self.segments[id]
            del self.listRectanglesa1[id]
            del self.listRectanglesa2[id]
            self.segmentsToSave = True

            # reset segment playback buttons
            self.playSegButton.setEnabled(False)
            self.playBandLimitedSegButton.setEnabled(False)
            self.box1id = -1

    def selectSegment(self, boxid):
        """ Changes the segment colors and enables playback buttons."""
        self.playSegButton.setEnabled(True)
        self.box1id = boxid
        print("selected %d" % self.box1id)

        brush = fn.mkBrush(self.ColourSelected)
        self.listRectanglesa1[boxid].setBrush(brush)
        self.listRectanglesa2[boxid].setBrush(brush)
        self.listRectanglesa1[boxid].setHoverBrush(brush)
        self.listRectanglesa2[boxid].setHoverBrush(brush)
        # self.listRectanglesa2[boxid].setPen(fn.mkPen(self.ColourSelectedDark,width=1))
        # if it's a rectangle:
        if type(self.listRectanglesa2[boxid]) == self.ROItype:
            self.playBandLimitedSegButton.setEnabled(True)

        self.listRectanglesa1[boxid].update()
        self.listRectanglesa2[boxid].update()

    def deselectSegment(self, boxid):
        """ Restores the segment colors and disables playback buttons."""
        self.playSegButton.setEnabled(False)
        self.playBandLimitedSegButton.setEnabled(False)
        self.box1id = -1

        col = self.prevBoxCol
        col.setAlpha(100)
        self.listRectanglesa1[boxid].setBrush(fn.mkBrush(col))
        self.listRectanglesa2[boxid].setBrush(fn.mkBrush(col))
        col.setAlpha(200)
        self.listRectanglesa1[boxid].setHoverBrush(fn.mkBrush(col))
        self.listRectanglesa2[boxid].setHoverBrush(fn.mkBrush(col))
        col.setAlpha(100)
        if self.dragRectTransparent.isChecked() and type(self.listRectanglesa2[boxid]) == self.ROItype:
            col = self.prevBoxCol.rgb()
            col = QtGui.QColor(col)
            col.setAlpha(255)
            self.listRectanglesa2[boxid].setBrush(pg.mkBrush(None))
            self.listRectanglesa2[boxid].setPen(col,width=1)

        self.listRectanglesa1[boxid].update()
        self.listRectanglesa2[boxid].update()

### mouse management

    def mouseMoved(self,evt):
        """ Listener for mouse moves.
        If the user moves the mouse in the spectrogram, print the time, frequency, power for the mouse location. """
        if not self.showPointerDetails:
            return
        elif self.p_spec.sceneBoundingRect().contains(evt):
            mousePoint = self.p_spec.mapSceneToView(evt)
            indexx = int(mousePoint.x())
            indexy = int(mousePoint.y())
            if indexx > 0 and indexx < np.shape(self.sg)[0] and indexy > 0 and indexy < np.shape(self.sg)[1]:
                time = self.convertSpectoAmpl(mousePoint.x()) + self.currentFileSection * self.config['maxFileShow'] - (self.currentFileSection>0)*self.config['fileOverlap'] + self.startTime
                seconds = time % 60
                minutes = int((time/60) % 60)
                hours = int((time/3600) % 24)
                if hours>0:
                    self.pointData.setText('time=%.2d:%.2d:%05.2f (hh:mm:ss.ms), freq=%0.1f (Hz),power=%0.1f (dB)' % (hours,minutes,seconds, mousePoint.y() * self.sampleRate / 2. / np.shape(self.sg)[1] + self.minFreq, self.sg[indexx, indexy]))
                else:
                    self.pointData.setText('time=%.2d:%05.2f (mm:ss.ms), freq=%0.1f (Hz),power=%0.1f (dB)' % (minutes,seconds, mousePoint.y() * self.sampleRate / 2. / np.shape(self.sg)[1] + self.minFreq, self.sg[indexx, indexy]))

    def mouseClicked_ampl(self,evt):
        """ Listener for if the user clicks on the amplitude plot.
        If there is a box selected, get its colour.
        If the user has clicked inside the scene, they could be
        (1) clicking in an already existing box -> select it
        (2) clicking anywhere else -> start a box
        (3) clicking a second time to finish a box -> create the segment
        """
        pos = evt.scenePos()

        # if any box is selected, deselect (wherever clicked)
        if self.box1id>-1:
            self.deselectSegment(self.box1id)

        # if clicked inside scene:
        if self.p_ampl.sceneBoundingRect().contains(pos):
            mousePoint = self.p_ampl.mapSceneToView(pos)

            # if this is the second click and not a box, close the segment
            if self.started:
                # can't finish boxes in ampl plot
                if self.dragRectangles.isChecked():
                    if self.startedInAmpl:
                        # started in ampl and finish in ampl,
                        # so continue as usual to draw a segment
                        pass
                    else:
                        # started in spec so ignore this bullshit
                        return

                # remove the drawing box:
                self.p_spec.removeItem(self.vLine_s)
                self.p_ampl.removeItem(self.vLine_a)
                self.p_ampl.removeItem(self.drawingBox_ampl)
                self.p_spec.removeItem(self.drawingBox_spec)
                # disconnect GrowBox listeners, leave the position listener
                self.p_ampl.scene().sigMouseMoved.disconnect()
                self.p_spec.scene().sigMouseMoved.disconnect()
                if self.showPointerDetails.isChecked():
                    self.p_spec.scene().sigMouseMoved.connect(self.mouseMoved)

                # If the user has pressed shift, copy the last species and don't use the context menu
                # If they pressed Control, add ? to the names
                # note: Ctrl+Shift combo doesn't have a Qt modifier and is ignored.
                modifiers = QtGui.QApplication.keyboardModifiers()
                if modifiers == QtCore.Qt.ShiftModifier:
                    self.addSegment(self.start_ampl_loc, max(mousePoint.x(),0.0),species=self.lastSpecies)
                elif modifiers == QtCore.Qt.ControlModifier:
                    self.addSegment(self.start_ampl_loc,max(mousePoint.x(),0.0))
                    # Context menu
                    self.fillBirdList(unsure=True)
                    self.menuBirdList.popup(QPoint(evt.screenPos().x(), evt.screenPos().y()))
                else:
                    self.addSegment(self.start_ampl_loc,max(mousePoint.x(),0.0))
                    # Context menu
                    self.fillBirdList()
                    self.menuBirdList.popup(QPoint(evt.screenPos().x(), evt.screenPos().y()))

                # the new segment is now selected and can be played
                self.selectSegment(self.box1id)
                self.started = not(self.started)
                self.startedInAmpl = False

            # if this is the first click:
            else:
                # if this is right click (drawing mode):
                # (or whatever you want)
                if evt.button() == self.MouseDrawingButton:
                    # this would prevent starting boxes in ampl plot
                    # if self.dragRectangles.isChecked():
                    #    return

                    nonebrush = self.ColourNone
                    self.start_ampl_loc = mousePoint.x()

                    # spectrogram plot bar and mouse followers:
                    self.vLine_s = pg.InfiniteLine(angle=90, movable=False,pen={'color': 'r', 'width': 3})
                    self.p_spec.addItem(self.vLine_s, ignoreBounds=True)
                    self.vLine_s.setPos(self.convertAmpltoSpec(self.start_ampl_loc))

                    self.drawingBox_spec = pg.LinearRegionItem(brush=nonebrush)
                    self.p_spec.addItem(self.drawingBox_spec, ignoreBounds=True)
                    self.drawingBox_spec.setRegion([self.convertAmpltoSpec(self.start_ampl_loc), self.convertAmpltoSpec(self.start_ampl_loc)])
                    self.p_spec.scene().sigMouseMoved.connect(self.GrowBox_spec)

                    # amplitude plot bar and mouse followers:
                    self.vLine_a = pg.InfiniteLine(angle=90, movable=False,pen={'color': 'r', 'width': 3})
                    self.p_ampl.addItem(self.vLine_a, ignoreBounds=True)
                    self.vLine_a.setPos(self.start_ampl_loc)

                    self.drawingBox_ampl = pg.LinearRegionItem(brush=nonebrush)
                    self.p_ampl.addItem(self.drawingBox_ampl, ignoreBounds=True)
                    self.drawingBox_ampl.setRegion([self.start_ampl_loc, self.start_ampl_loc])
                    self.p_ampl.scene().sigMouseMoved.connect(self.GrowBox_ampl)

                    self.started = not (self.started)
                    self.startedInAmpl = True

                # if this is left click (selection mode):
                else:
                    # Check if the user has clicked in a box
                    # Note: Returns the first one it finds, i.e. the newest
                    box1id = -1
                    for count in range(len(self.listRectanglesa1)):
                        if self.listRectanglesa1[count] is not None:
                            x1, x2 = self.listRectanglesa1[count].getRegion()
                            if x1 <= mousePoint.x() and x2 >= mousePoint.x():
                                box1id = count
                                break

                    # User clicked in a segment:
                    if box1id > -1:
                        # select the segment:
                        self.prevBoxCol = self.listRectanglesa1[box1id].brush.color()
                        self.selectSegment(box1id)

                        # popup dialog
                        modifiers = QtGui.QApplication.keyboardModifiers()
                        if modifiers == QtCore.Qt.ControlModifier:
                            self.fillBirdList(unsure=True)
                        else:
                            self.fillBirdList()
                        self.menuBirdList.popup(QPoint(evt.screenPos().x(), evt.screenPos().y()))
                    else:
                        # TODO: pan the view
                        pass

    def mouseClicked_spec(self,evt):
        """ Listener for if the user clicks on the spectrogram plot.
        See the amplitude version (mouseClicked_ampl()) for details. Although much of the code is a repeat,
        it is separated for clarity.
        """
        pos = evt.scenePos()

        # if any box is selected, deselect (wherever clicked)
        if self.box1id>-1:
            self.deselectSegment(self.box1id)

        # if clicked inside scene:
        if self.p_spec.sceneBoundingRect().contains(pos):
            mousePoint = self.p_spec.mapSceneToView(pos)

            # if this is the second click, close the segment/box
            # note: can finish segment with either left or right click
            if self.started:
                if self.dragRectangles.isChecked() and self.startedInAmpl:
                    # started in ampl, and spec is used for boxes, so can't continue here
                    return

                # remove the drawing box:
                if not self.dragRectangles.isChecked():
                    self.p_spec.removeItem(self.vLine_s)
                    self.p_ampl.scene().sigMouseMoved.disconnect()
                self.p_ampl.removeItem(self.vLine_a)
                self.p_ampl.removeItem(self.drawingBox_ampl)
                self.p_spec.removeItem(self.drawingBox_spec)
                # disconnect GrowBox listeners, leave the position listener
                self.p_spec.scene().sigMouseMoved.disconnect()
                if self.showPointerDetails.isChecked():
                    self.p_spec.scene().sigMouseMoved.connect(self.mouseMoved)

                # Pass either default y coords or box limits:
                x1 = self.start_ampl_loc
                x2 = self.convertSpectoAmpl(max(mousePoint.x(), 0.0))
                # Could add this check if right edge seems dangerous:
                # endx = min(x2, np.shape(self.sg)[0]+1)
                if self.dragRectangles.isChecked():
                    y1 = self.start_spec_y
                    y2 = mousePoint.y() / np.shape(self.sg)[1]
                else:
                    y1 = 0
                    y2 = 0
                # If the user has pressed shift, copy the last species and don't use the context menu
                # If they pressed Control, add ? to the names
                # note: Ctrl+Shift combo doesn't have a Qt modifier and is ignored.
                modifiers = QtGui.QApplication.keyboardModifiers()
                if modifiers == QtCore.Qt.ShiftModifier:
                    self.addSegment(x1, x2, y1, y2, species=self.lastSpecies)
                elif modifiers == QtCore.Qt.ControlModifier:
                    self.addSegment(x1, x2, y1, y2)
                    # Context menu
                    self.fillBirdList(unsure=True)
                    self.menuBirdList.popup(QPoint(evt.screenPos().x(), evt.screenPos().y()))
                else:
                    self.addSegment(x1, x2, y1, y2)
                    # Context menu
                    self.fillBirdList()
                    self.menuBirdList.popup(QPoint(evt.screenPos().x(), evt.screenPos().y()))

                # select the new segment/box
                self.selectSegment(self.box1id)
                self.started = not(self.started)
                self.startedInAmpl = False

            # if this is the first click:
            else:
                # if this is right click (drawing mode):
                if evt.button() == self.MouseDrawingButton:
                    nonebrush = self.ColourNone
                    self.start_ampl_loc = self.convertSpectoAmpl(mousePoint.x())
                    self.start_spec_y = mousePoint.y() / np.shape(self.sg)[1]

                    # start a new box:
                    if self.dragRectangles.isChecked():
                        # spectrogram mouse follower box:
                        startpointS = QPointF(mousePoint.x(), mousePoint.y())
                        endpointS = QPointF(mousePoint.x(), mousePoint.y())

                        self.drawingBox_spec = SupportClasses.ShadedRectROI(startpointS, endpointS - startpointS, invertible=True)
                        self.drawingBox_spec.setBrush(nonebrush)
                        self.p_spec.addItem(self.drawingBox_spec, ignoreBounds=True)
                        self.p_spec.scene().sigMouseMoved.connect(self.GrowBox_spec)
                    # start a new segment:
                    else:
                        # spectrogram bar and mouse follower:
                        self.vLine_s = pg.InfiniteLine(angle=90, movable=False,pen={'color': 'r', 'width': 3})
                        self.p_spec.addItem(self.vLine_s, ignoreBounds=True)
                        self.vLine_s.setPos(mousePoint.x())

                        self.drawingBox_spec = pg.LinearRegionItem(brush=nonebrush)
                        self.p_spec.addItem(self.drawingBox_spec, ignoreBounds=True)
                        self.drawingBox_spec.setRegion([mousePoint.x(),mousePoint.x()])
                        self.p_spec.scene().sigMouseMoved.connect(self.GrowBox_spec)
                        # note - only in segment mode react to movement over ampl plot:
                        self.p_ampl.scene().sigMouseMoved.connect(self.GrowBox_ampl)

                    # for box and segment - amplitude plot bar:
                    self.vLine_a = pg.InfiniteLine(angle=90, movable=False,pen={'color': 'r', 'width': 3})
                    self.p_ampl.addItem(self.vLine_a, ignoreBounds=True)
                    self.vLine_a.setPos(self.start_ampl_loc)

                    self.drawingBox_ampl = pg.LinearRegionItem(brush=nonebrush)
                    self.p_ampl.addItem(self.drawingBox_ampl, ignoreBounds=True)
                    self.drawingBox_ampl.setRegion([self.start_ampl_loc, self.start_ampl_loc])

                    self.started = not (self.started)
                    self.startedInAmpl = False

                # if this is left click (selection mode):
                else:
                    # Check if the user has clicked in a box
                    # Note: Returns the first one it finds, i.e. the newest
                    box1id = -1
                    for count in range(len(self.listRectanglesa2)):
                        if type(self.listRectanglesa2[count]) == self.ROItype:
                            x1 = self.listRectanglesa2[count].pos().x()
                            y1 = self.listRectanglesa2[count].pos().y()
                            x2 = x1 + self.listRectanglesa2[count].size().x()
                            y2 = y1 + self.listRectanglesa2[count].size().y()
                            if x1 <= mousePoint.x() and x2 >= mousePoint.x() and y1 <= mousePoint.y() and y2 >= mousePoint.y():
                                box1id = count
                                break
                        elif self.listRectanglesa2[count] is not None:
                            x1, x2 = self.listRectanglesa2[count].getRegion()
                            if x1 <= mousePoint.x() and x2 >= mousePoint.x():
                                box1id = count
                                break

                    # User clicked in a segment:
                    if box1id > -1:
                        # select the segment:
                        self.prevBoxCol = self.listRectanglesa1[box1id].brush.color()
                        self.selectSegment(box1id)

                        modifiers = QtGui.QApplication.keyboardModifiers()
                        if modifiers == QtCore.Qt.ControlModifier:
                            self.fillBirdList(unsure=True)
                        else:
                            self.fillBirdList()
                        self.menuBirdList.popup(QPoint(evt.screenPos().x(), evt.screenPos().y()))
                    else:
                        # TODO: pan the view
                        pass

# TODO: if in rect mode you start in ampl plot -> get undefined ys

    def GrowBox_ampl(self,pos):
        """ Listener for when a segment is being made in the amplitude plot.
        Makes the blue box that follows the mouse change size. """
        if self.p_ampl.sceneBoundingRect().contains(pos):
            mousePoint = self.p_ampl.mapSceneToView(pos)
            self.drawingBox_ampl.setRegion([self.start_ampl_loc, mousePoint.x()])
            self.drawingBox_spec.setRegion([self.convertAmpltoSpec(self.start_ampl_loc), self.convertAmpltoSpec(mousePoint.x())])

    def GrowBox_spec(self, pos):
        """ Listener for when a segment is being made in the spectrogram plot.
        Makes the blue box that follows the mouse change size. """
        if self.p_spec.sceneBoundingRect().contains(pos):
            mousePoint = self.p_spec.mapSceneToView(pos)
            self.drawingBox_ampl.setRegion([self.start_ampl_loc, self.convertSpectoAmpl(mousePoint.x())])
            if self.dragRectangles.isChecked() and not self.startedInAmpl:
                # making a box
                posY = mousePoint.y() - self.start_spec_y * np.shape(self.sg)[1]
                self.drawingBox_spec.setSize([mousePoint.x()-self.convertAmpltoSpec(self.start_ampl_loc), posY])
            else:
                # making a segment
                self.drawingBox_spec.setRegion([self.convertAmpltoSpec(self.start_ampl_loc), mousePoint.x()])

    def birdSelected(self,birdname,update=True):
        """ Collects the label for a bird from the context menu and processes it.
        Has to update the overview segments in case their colour should change.
        Also handles getting the name through a message box if necessary.
        """
        startpoint = self.segments[self.box1id][0]-self.startRead
        endpoint = self.segments[self.box1id][1]-self.startRead
        oldname = self.segments[self.box1id][4]

        self.refreshOverviewWith(startpoint, endpoint, oldname, delete=True)
        self.refreshOverviewWith(startpoint, endpoint, birdname)

        # Now update the text
        if birdname is not 'Other':
            self.updateText(birdname)
            if update:
                # Put the selected bird name at the top of the list
                if birdname[-1] == '?':
                    birdname = birdname[:-1]
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
                    if update:
                        self.config['BirdList'].insert(0,text)
                    else:
                        self.config['BirdList'].append(text)
                    # self.saveConfig = True

    def updateText(self, text,segID=None):
        """ When the user sets or changes the name in a segment, update the text and the colour. """
        if segID is None:
            segID = self.box1id
        #print segID, len(self.segments), len(self.listRectanglesa1)
        self.segments[segID][4] = text
        print(segID, len(self.listLabels))
        self.listLabels[segID].setText(text,'k')

        # Update the colour
        if text != "Don't Know":
            if text[-1] == '?':
                self.prevBoxCol = self.ColourPossible
            else:
                self.prevBoxCol = self.ColourNamed
        else:
            self.prevBoxCol = self.ColourNone

        # Store the species in case the user wants it for the next segment
        self.lastSpecies = text
        self.segmentsToSave = True

    def setColourMap(self,cmap):
        """ Listener for the menu item that chooses a colour map.
        Loads them from the file as appropriate and sets the lookup table.
        """
        self.config['cmap'] = cmap

        import colourMaps
        pos, colour, mode = colourMaps.colourMaps(cmap)

        cmap = pg.ColorMap(pos, colour,mode)
        self.lut = cmap.getLookupTable(0.0, 1.0, 256)

        self.specPlot.setLookupTable(self.lut)
        self.overviewImage.setLookupTable(self.lut)

    def invertColourMap(self):
        """ Listener for the menu item that converts the colour map"""
        # self.config['invertColourMap'] = not self.config['invertColourMap']
        self.config['invertColourMap'] = self.invertcm.isChecked()
        self.setColourLevels()

    def setColourLevels(self):
        """ Listener for the brightness and contrast sliders being changed. Also called when spectrograms are loaded, etc.
        Translates the brightness and contrast values into appropriate image levels.
        Calculation is simple.
        """
        minsg = np.min(self.sg)
        maxsg = np.max(self.sg)
        self.config['brightness'] = self.brightnessSlider.value()
        self.config['contrast'] = self.contrastSlider.value()
        self.colourStart = (self.config['brightness'] / 100.0 * self.config['contrast'] / 100.0) * (maxsg - minsg) + minsg
        self.colourEnd = (maxsg - minsg) * (1.0 - self.config['contrast'] / 100.0) + self.colourStart

        if self.config['invertColourMap']:
            self.overviewImage.setLevels([self.colourEnd, self.colourStart])
            self.specPlot.setLevels([self.colourEnd, self.colourStart])
        else:
            self.overviewImage.setLevels([self.colourStart, self.colourEnd])
            self.specPlot.setLevels([self.colourStart, self.colourEnd])

    def moveLeft(self):
        """ When the left button is pressed (next to the overview plot), move everything along
        Allows a 10% overlap """
        minX, maxX = self.overviewImageRegion.getRegion()
        newminX = max(0,minX-(maxX-minX)*0.9)
        self.overviewImageRegion.setRegion([newminX, newminX+maxX-minX])
        self.updateOverview()
        self.playPosition = int(self.convertSpectoAmpl(newminX)*1000.0)

    def moveRight(self):
        """ When the right button is pressed (next to the overview plot), move everything along
        Allows a 10% overlap """
        minX, maxX = self.overviewImageRegion.getRegion()
        newminX = min(np.shape(self.sg)[0]-(maxX-minX),minX+(maxX-minX)*0.9)
        self.overviewImageRegion.setRegion([newminX, newminX+maxX-minX])
        self.updateOverview()
        self.playPosition = int(self.convertSpectoAmpl(newminX)*1000.0)

    def prepare5minMove(self):
        #if len(self.segments) > 0 or self.hasSegments:
            # if len(self.segments) > 0:
            #     if self.segments[0][0] > -1:
            #         self.segments.insert(0,
            #                              [-1, str(QTime().addSecs(self.startTime).toString('hh:mm:ss')), self.operator,
            #                               self.reviewer, -1])
            # else:
            #     self.segments.insert(0, [-1, str(QTime().addSecs(self.startTime).toString('hh:mm:ss')), self.operator,
            #                              self.reviewer, -1])
        self.saveSegments()
        self.resetStorageArrays()
        # Reset the media player
        if self.media_obj.isPlaying():
            self.media_obj.stop()
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.loadFile()

    def movePrev5mins(self):
        """ When the button to move to the next 5 minutes is pressed, enable that.
        Have to check if the buttons should be disabled or not,
        save the segments and reset the arrays, then call loadFile.
        """
        self.currentFileSection -= 1
        self.next5mins.setEnabled(True)
        if self.currentFileSection <= 0:
            self.prev5mins.setEnabled(False)
        self.prepare5minMove()


    def moveNext5mins(self):
        """ When the button to move to the previous 5 minutes is pressed, enable that.
        Have to check if the buttons should be disabled or not,
        save the segments and reset the arrays, then call loadFile.
        """
        self.currentFileSection += 1
        self.prev5mins.setEnabled(True)
        if self.currentFileSection >= self.nFileSections-1:
            self.next5mins.setEnabled(False)
        self.prepare5minMove()

    def scroll(self):
        """ When the slider at the bottom of the screen is moved, move everything along. """
        newminX = self.scrollSlider.value()
        minX, maxX = self.overviewImageRegion.getRegion()
        self.overviewImageRegion.setRegion([newminX, newminX+maxX-minX])
        self.updateOverview()
        self.playPosition = int(self.convertSpectoAmpl(newminX)*1000.0)

    def changeWidth(self, value):
        """ Listener for the spinbox that decides the width of the main window.
        It updates the top figure plots as the window width is changed.
        Slightly annoyingly, it gets called when the value gets reset, hence the first line. """
        if not hasattr(self,'overviewImageRegion'):
            return
        self.windowSize = value

        # Redraw the highlight in the overview figure appropriately
        minX, maxX = self.overviewImageRegion.getRegion()
        newmaxX = self.convertAmpltoSpec(value)+minX
        self.overviewImageRegion.setRegion([minX, newmaxX])
        self.scrollSlider.setMaximum(np.shape(self.sg)[0]-self.convertAmpltoSpec(self.widthWindow.value()))
        self.updateOverview()

# ===============
# Generate the various dialogs that match the menu items

    def loadSegment(self):
        # Loads a segment for the HumanClassify dialogs
        wavobj = wavio.read(self.filename, self.config['maxFileShow'], self.startRead)
        self.audiodata = wavobj.data

        if self.audiodata.dtype is not 'float':
            self.audiodata = self.audiodata.astype('float')  # / 32768.0

        if np.shape(np.shape(self.audiodata))[0] > 1:
            self.audiodata = self.audiodata[:, 0]

        # Get the data for the spectrogram
        sgRaw = self.sp.spectrogram(self.audiodata, self.config['window_width'],
                                    self.config['incr'], mean_normalise=True, onesided=True,
                                    multitaper=False)
        maxsg = np.min(sgRaw)
        self.sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))

    def showFirstPage(self):
        # After the HumanClassify dialogs have closed, need to show the correct data on the screen
        # Returns to the page user started with
        if self.config['maxFileShow']<self.datalengthSec:
            self.currentFileSection = self.currentPage
            self.prepare5minMove()
            self.next5mins.setEnabled(True)
            self.prev5mins.setEnabled(False)

    def humanClassifyDialog1(self):
        """ Create the dialog that shows calls to the user for verification.
        There are two versions in here depending on whether you wish to show them from all the pages (5 min sections), the default, or not.
        The only difference is that that version requires sorting the segments and loading of new file sections and making the spectrogram intermittently.
        """

        # Store the current page to return to
        self.currentPage = self.currentFileSection
        self.segmentsDone = 0
        # Check there are segments to show on this page
        if not self.config['showAllPages']:
            if len(self.segments)>0:
                self.box1id = 0
                while self.box1id<len(self.segments) and self.listRectanglesa2[self.box1id] is None:
                    self.box1id += 1
        else:
            self.box1id = 0

        if (self.config['showAllPages'] and len(self.segments)==0) or (not self.config['showAllPages'] and (self.box1id == len(self.segments) or len(self.listRectanglesa2)==0)):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No segments to check")
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
            msg.setWindowTitle("No segments")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        else:
            if not self.config['showAllPages']:
                # Different calls for the two types of region
                if type(self.listRectanglesa2[self.box1id]) == self.ROItype:
                    x1 = self.listRectanglesa2[self.box1id].pos()[0]
                    x2 = x1 + self.listRectanglesa2[self.box1id].size()[0]
                else:
                    x1, x2 = self.listRectanglesa2[self.box1id].getRegion()
                x1 = int(x1)
                x2 = int(x2)
                x3 = int(self.listRectanglesa1[self.box1id].getRegion()[0] * self.sampleRate)
                x4 = int(self.listRectanglesa1[self.box1id].getRegion()[1] * self.sampleRate)
            else:
                # Showing them on all pages is a bit more of a pain
                # Sort the segments into increasing time order, apply same order to listRects and labels
                sortOrder = sorted(range(len(self.segments)), key=self.segments.__getitem__)
                self.segments = [self.segments[i] for i in sortOrder]
                self.listRectanglesa1 = [self.listRectanglesa1[i] for i in sortOrder]
                self.listRectanglesa2 = [self.listRectanglesa2[i] for i in sortOrder]
                self.listLabels = [self.listLabels[i] for i in sortOrder]

                # Check which page is first to have segments on
                self.currentFileSection = 0
                while self.segments[0][0] > (self.currentFileSection+1)*self.config['maxFileShow']:
                    self.currentFileSection += 1

                # Load the first segment
                self.startRead = self.currentFileSection * self.config['maxFileShow']
                self.loadSegment()

                # And show it
                x1 = int(self.convertAmpltoSpec(self.segments[self.box1id][0]-self.startRead))
                x2 = int(self.convertAmpltoSpec(self.segments[self.box1id][1]-self.startRead))
                x3 = int((self.segments[self.box1id][0]-self.startRead)*self.sampleRate)
                x4 = int((self.segments[self.box1id][1]-self.startRead)*self.sampleRate)

            self.humanClassifyDialog1 = Dialogs.HumanClassify1(self.sg[x1:x2,:],self.audiodata[x3:x4],self.sampleRate,self.segments[self.box1id][4],self.lut,self.colourStart,self.colourEnd,self.config['invertColourMap'], self.config['BirdList'], self)
            self.humanClassifyDialog1.setSegNumbers(0, len(self.segments))
            self.humanClassifyDialog1.show()
            self.humanClassifyDialog1.activateWindow()
            #self.humanClassifyDialog1.close.clicked.connect(self.humanClassifyClose1)
            self.humanClassifyDialog1.correct.clicked.connect(self.humanClassifyCorrect1)
            self.humanClassifyDialog1.delete.clicked.connect(self.humanClassifyDelete1)
            # self.statusLeft.setText("Ready")

    def humanClassifyClose1(self):
        # Listener for the human verification dialog.
        # Listener for the human verification dialog.
        self.humanClassifyDialog1.done(1)
        # Want to show a page at the end, so make it the first one
        if self.config['showAllPages']:
            self.showFirstPage()

    def humanClassifyNextImage1(self):
        # Get the next image
        if self.box1id < len(self.segments)-1:
            self.box1id += 1
            # update "done/to go" numbers:
            self.humanClassifyDialog1.setSegNumbers(self.box1id, len(self.segments))
            if not self.config['showAllPages']:
                # Different calls for the two types of region
                if self.listRectanglesa2[self.box1id] is not None:
                    if type(self.listRectanglesa2[self.box1id]) == self.ROItype:
                        x1 = self.listRectanglesa2[self.box1id].pos()[0]
                        x2 = x1 + self.listRectanglesa2[self.box1id].size()[0]
                    else:
                        x1, x2 = self.listRectanglesa2[self.box1id].getRegion()
                    x1 = int(x1)
                    x2 = int(x2)
                    x3 = int(self.listRectanglesa1[self.box1id].getRegion()[0] * self.sampleRate)
                    x4 = int(self.listRectanglesa1[self.box1id].getRegion()[1] * self.sampleRate)
                    self.humanClassifyDialog1.setImage(self.sg[x1:x2, :], self.audiodata[x3:x4], self.sampleRate,
                                                       self.segments[self.box1id][4])
            else:
                # Check if have moved to next segment, and if so load it
                # If there was a section without segments this would be a bit inefficient, actually no, it was wrong!
                if self.segments[self.box1id][0] > (self.currentFileSection+1)*self.config['maxFileShow']:
                    while self.segments[self.box1id][0] > (self.currentFileSection+1)*self.config['maxFileShow']:
                        self.currentFileSection += 1
                    self.startRead = self.currentFileSection * self.config['maxFileShow']
                    print("Loading next page", self.currentFileSection)
                    self.loadSegment()

                # Show the next segment
                x1 = int(self.convertAmpltoSpec(self.segments[self.box1id][0] - self.startRead))
                x2 = int(self.convertAmpltoSpec(self.segments[self.box1id][1] - self.startRead))
                x3 = int((self.segments[self.box1id][0] - self.startRead) * self.sampleRate)
                x4 = int((self.segments[self.box1id][1] - self.startRead) * self.sampleRate)
                self.humanClassifyDialog1.setImage(self.sg[x1:x2, :], self.audiodata[x3:x4], self.sampleRate,
                                                   self.segments[self.box1id][4])

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setIconPixmap(QPixmap("img/Owl_done.png"))
            msg.setText("All segmentations checked")
            msg.setWindowTitle("Finished")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            self.humanClassifyClose1()

    def updateLabel(self,label):
        """ Update the label on a segment that is currently shown in the display. """
        self.birdSelected(label, update=False)

        if self.listRectanglesa2[self.box1id] is not None:
            self.listRectanglesa1[self.box1id].setBrush(self.prevBoxCol)
            self.listRectanglesa1[self.box1id].update()
            if self.dragRectTransparent.isChecked() and type(self.listRectanglesa2[self.box1id]) == self.ROItype:
                col = self.prevBoxCol.rgb()
                col = QtGui.QColor(col)
                col.setAlpha(255)
                self.listRectanglesa2[self.box1id].setPen(col, width=1)
            else:
                self.listRectanglesa2[self.box1id].setBrush(self.prevBoxCol)

            self.listRectanglesa2[self.box1id].update()
            self.segmentsToSave = True

    def humanClassifyCorrect1(self):
        """ Correct segment labels, save the old ones if necessary """
        label, self.saveConfig, checkText = self.humanClassifyDialog1.getValues()
        self.segmentsDone += 1
        if len(checkText) > 0:
            if label != checkText:
                label = str(checkText)
                self.humanClassifyDialog1.birdTextEntered()
                # self.saveConfig = True
            #self.humanClassifyDialog1.tbox.setText('')

        if label != self.segments[self.box1id][4]:
            if self.config['saveCorrections']:
                # Save the correction
                outputError = [self.segments[self.box1id], label]
                file = open(self.filename + '.corrections', 'a')
                json.dump(outputError, file)
                file.close()

            # Update the label on the box if it is in the current page
            self.updateLabel(label)
            # if self.listRectanglesa2[self.box1id] is not None:
            #     self.birdSelected(label,update=False)
            #
            #     self.listRectanglesa1[self.box1id].setBrush(self.prevBoxCol)
            #     self.listRectanglesa1[self.box1id].update()
            #     if self.dragRectTransparent.isChecked() and type(self.listRectanglesa2[self.box1id]) == self.ROItype:
            #         col = self.prevBoxCol.rgb()
            #         col = QtGui.QColor(col)
            #         col.setAlpha(255)
            #         self.listRectanglesa2[self.box1id].setPen(col,width=1)
            #     else:
            #         self.listRectanglesa2[self.box1id].setBrush(self.prevBoxCol)
            #
            #     self.listRectanglesa2[self.box1id].update()

            if self.saveConfig:
                self.config['BirdList'].append(label)
        elif label[-1] == '?':
            # Remove the question mark, since the user has agreed
            self.updateLabel(label[:-1])
            #self.segments[self.box1id][4] = label[:-1]

        self.humanClassifyDialog1.tbox.setText('')
        self.humanClassifyDialog1.tbox.setEnabled(False)
        self.humanClassifyNextImage1()

    def humanClassifyDelete1(self):
        # Delete a segment
        id = self.box1id
        self.deleteSegment(self.box1id)
        self.box1id = id-1
        self.segmentsToSave = True
        self.humanClassifyNextImage1()
        self.segmentsDone += 1

    def humanReview2(self):
        """ Create the dialog that shows sets of calls to the user for verification.
        """
        # Check there are segments to show on this page
        if not self.config['showAllPages']:
            if len(self.segments)>0:
                self.box1id = 0
                while self.box1id<len(self.segments) and self.listRectanglesa2[self.box1id] is None:
                    self.box1id += 1
        if (self.config['showAllPages'] and len(self.segments)==0) or (not self.config['showAllPages'] and (self.box1id == len(self.segments) or len(self.listRectanglesa2)==0)):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No segments to check")
            msg.setIconPixmap(QPixmap('img/Owl_warning.png'))
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setWindowTitle("No segment")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        self.statusLeft.setText("Checking...")

        if not self.config['showAllPages']:
            # Get the segments that are in the current page
            segs = []
            indices = []
            for i in range(len(self.segments)):
                if self.listRectanglesa2[i] is not None:
                    segs.append([self.segments[i][0]-self.startRead,self.segments[i][1]-self.startRead,-1,-1,self.segments[i][4]])
                    indices.append(i)
            names = [item[4] for item in segs]
            names = [n if n[-1] != '?' else n[:-1] for n in names]
        else:
            # Get all of them
            names = [item[4] for item in self.segments]
            names = [n if n[-1] != '?' else n[:-1] for n in names]
        # Make them unique
        keys = {}
        for n in names:
            keys[n] = 1
        names = keys.keys()
        self.humanClassifyDialog2a = Dialogs.HumanClassify2a(names)

        if self.humanClassifyDialog2a.exec_() == 1:
            label = self.humanClassifyDialog2a.getValues()
            self.indices = []
            segments = []
            if not self.config['showAllPages']:
                # Find the segments that have the right label
                for ind in range(len(segs)):
                    if self.segments[ind][4] == label or self.segments[ind][4][:-1] == label:
                        self.indices.append(ind)
                        segments.append([segs[ind][0] - self.startRead, segs[i][1] - self.startRead, -1, -1,
                                     segs[i][4]])

                # Pass those segments to the dialog
                self.humanReviewDialog2 = Dialogs.HumanReview2Tab(label=label)
                self.humanReviewDialog2.show()
                errors = self.humanReviewDialog2.getValues()
                # If there are errors, get their correct indices and process them
                if len(errors) > 0:
                    # Turn these numbers back into indices into self.segments
                    # The worst naming ever, sorry. There are two sets of indices -- self.indices keeps track of those with the chosen label, while indices is those on the current page!
                    inderr = []
                    for error in errors:
                        inderr.append(indices[self.indices[error]])
                    outputErrors = []
                    for error in inderr[-1::-1]:
                        outputErrors.append(self.segments[error])
                        self.deleteSegment(error)
                    self.segmentsToSave = True
                    if self.config['saveCorrections']:
                        # Save the errors in a file
                        file = open(self.filename + '.corrections_' + str(label), 'a')
                        json.dump(outputErrors, file)
                        file.close()
            else:
                # Sort everything into order
                sortOrder = sorted(range(len(self.segments)), key=self.segments.__getitem__)
                self.segments = [self.segments[i] for i in sortOrder]
                self.listRectanglesa1 = [self.listRectanglesa1[i] for i in sortOrder]
                self.listRectanglesa2 = [self.listRectanglesa2[i] for i in sortOrder]
                self.listLabels = [self.listLabels[i] for i in sortOrder]
                # Loop over the pages of the file
                for self.currentFileSection in range(self.nFileSections):
                    self.startRead = self.currentFileSection * self.config['maxFileShow']
                    # Get the index of the first and last segments on this page
                    firstSeg = 0
                    while firstSeg < len(self.segments) and self.segments[firstSeg][0] < self.startRead:
                        firstSeg += 1
                    lastSeg = firstSeg
                    while lastSeg < len(self.segments) and self.segments[lastSeg][0] < (self.currentFileSection + 1) * self.config['maxFileShow']:
                        lastSeg += 1
                    self.indices = []
                    segments = []
                    # Loop over the segments on this page and find those with the correct label
                    for ind in range(firstSeg,lastSeg):
                        if self.segments[ind][4] == label or self.segments[ind][4][:-1] == label:
                            self.indices.append(ind)
                            segments.append([self.segments[ind][0]-self.startRead,self.segments[ind][1]-self.startRead,-1,-1,self.segments[ind][4]])
                    errors = []
                    # if there are segments on the next page, load it
                    if len(segments)>0:
                        self.loadSegment()
                        self.humanClassifyDialog2 = Dialogs.HumanClassify2(self.sg,segments,label,self.currentFileSection,self.nFileSections,self.sampleRate, self.config['incr'], self.lut,self.colourStart,self.colourEnd,self.config['invertColourMap'])
                        self.humanClassifyDialog2.exec_()
                        errors = self.humanClassifyDialog2.getValues()
                        if len(errors)>0:
                            inderr = []
                            for error in errors:
                                inderr.append(self.indices[error])
                            outputErrors = []
                            for error in inderr[-1::-1]:
                                outputErrors.append(self.segments[error])
                                self.deleteSegment(id=error)
                            self.segmentsToSave = True
                            if self.config['saveCorrections']:
                                # Save the errors in a file
                                file = open(self.filename + '.corrections_' + str(label), 'a')
                                json.dump(outputErrors, file)
                                file.close()
        # Want to show a page at the end, so make it the first one
        # self.showFirstPage()
        self.statusLeft.setText("Ready") #Todo: why updated segments doesn't save when no paging? (short files)

    def humanClassifyDialog2(self):
        """ Create the dialog that shows sets of calls to the user for verification.
        This is complicated because of the pages.
        Basic idea is to keep track of which page you are on, and load them as required. Unfortunately, it gets convoluted.
        """
        # Check there are segments to show on this page
        if not self.config['showAllPages']:
            if len(self.segments)>0:
                self.box1id = 0
                while self.box1id<len(self.segments) and self.listRectanglesa2[self.box1id] is None:
                    self.box1id += 1
        if (self.config['showAllPages'] and len(self.segments)==0) or (not self.config['showAllPages'] and (self.box1id == len(self.segments) or len(self.listRectanglesa2)==0)):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No segments to check")
            msg.setIconPixmap(QPixmap('img/Owl_warning.png'))
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setWindowTitle("No segment")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        self.statusLeft.setText("Checking...")

        if not self.config['showAllPages']:
            # Get the segments that are in the current page
            segs = []
            indices = []
            for i in range(len(self.segments)):
                if self.listRectanglesa2[i] is not None:
                    segs.append([self.segments[i][0]-self.startRead,self.segments[i][1]-self.startRead,-1,-1,self.segments[i][4]])
                    indices.append(i)
            names = [item[4] for item in segs]
            names = [n if n[-1] != '?' else n[:-1] for n in names]
        else:
            # Get all of them
            names = [item[4] for item in self.segments]
            names = [n if n[-1] != '?' else n[:-1] for n in names]
        # Make them unique
        keys = {}
        for n in names:
            keys[n] = 1
        names = keys.keys()
        self.humanClassifyDialog2a = Dialogs.HumanClassify2a(names)

        if self.humanClassifyDialog2a.exec_() == 1:
            label = self.humanClassifyDialog2a.getValues()
            self.indices = []
            segments = []
            if not self.config['showAllPages']:
                # Find the segments that have the right label
                for ind in range(len(segs)):
                    if self.segments[ind][4] == label or self.segments[ind][4][:-1] == label:
                        self.indices.append(ind)
                        segments.append([segs[ind][0] - self.startRead, segs[i][1] - self.startRead, -1, -1,
                                     segs[i][4]])

                # Pass those segments to the dialog
                self.humanClassifyDialog2 = Dialogs.HumanClassify2(self.sg,segments,label,1,1,self.sampleRate, self.config['incr'], self.lut,self.colourStart,self.colourEnd,self.config['invertColourMap'])
                self.humanClassifyDialog2.exec_()
                errors = self.humanClassifyDialog2.getValues()
                # If there are errors, get their correct indices and process them
                if len(errors) > 0:
                    # Turn these numbers back into indices into self.segments
                    # The worst naming ever, sorry. There are two sets of indices -- self.indices keeps track of those with the chosen label, while indices is those on the current page!
                    inderr = []
                    for error in errors:
                        inderr.append(indices[self.indices[error]])
                    outputErrors = []
                    for error in inderr[-1::-1]:
                        outputErrors.append(self.segments[error])
                        self.deleteSegment(error,hr=True)
                    self.segmentsToSave = True
                    if self.config['saveCorrections']:
                        # Save the errors in a file
                        file = open(self.filename + '.corrections_' + str(label), 'a')
                        json.dump(outputErrors, file)
                        file.close()
            else:
                # Sort everything into order
                sortOrder = sorted(range(len(self.segments)), key=self.segments.__getitem__)
                self.segments = [self.segments[i] for i in sortOrder]
                self.listRectanglesa1 = [self.listRectanglesa1[i] for i in sortOrder]
                self.listRectanglesa2 = [self.listRectanglesa2[i] for i in sortOrder]
                self.listLabels = [self.listLabels[i] for i in sortOrder]
                # Loop over the pages of the file
                for self.currentFileSection in range(self.nFileSections):
                    self.startRead = self.currentFileSection * self.config['maxFileShow']
                    # Get the index of the first and last segments on this page
                    firstSeg = 0
                    while firstSeg < len(self.segments) and self.segments[firstSeg][0] < self.startRead:
                        firstSeg += 1
                    lastSeg = firstSeg
                    while lastSeg < len(self.segments) and self.segments[lastSeg][0] < (self.currentFileSection + 1) * self.config['maxFileShow']:
                        lastSeg += 1
                    self.indices = []
                    segments = []
                    # Loop over the segments on this page and find those with the correct label
                    for ind in range(firstSeg,lastSeg):
                        if self.segments[ind][4] == label or self.segments[ind][4][:-1] == label:
                            self.indices.append(ind)
                            segments.append([self.segments[ind][0]-self.startRead,self.segments[ind][1]-self.startRead,-1,-1,self.segments[ind][4]])
                    errors = []
                    # if there are segments on the next page, load it
                    if len(segments)>0:
                        self.loadSegment()
                        self.humanClassifyDialog2 = Dialogs.HumanClassify2(self.sg,segments,label,self.currentFileSection,self.nFileSections,self.sampleRate, self.config['incr'], self.lut,self.colourStart,self.colourEnd,self.config['invertColourMap'])
                        self.humanClassifyDialog2.exec_()
                        errors = self.humanClassifyDialog2.getValues()
                        if len(errors)>0:
                            inderr = []
                            for error in errors:
                                inderr.append(self.indices[error])
                            outputErrors = []
                            for error in inderr[-1::-1]:
                                outputErrors.append(self.segments[error])
                                self.deleteSegment(id=error)
                            self.segmentsToSave = True
                            if self.config['saveCorrections']:
                                # Save the errors in a file
                                file = open(self.filename + '.corrections_' + str(label), 'a')
                                json.dump(outputErrors, file)
                                file.close()
        # Want to show a page at the end, so make it the first one
        # self.showFirstPage()
        self.statusLeft.setText("Ready") #Todo: why updated segments doesn't save when no paging? (short files)

    def showSpectrogramDialog(self):
        """ Create the spectrogram dialog when the button is pressed.
        """
        if not hasattr(self,'spectrogramDialog'):
            self.spectrogramDialog = Dialogs.Spectrogram(self.config['window_width'],self.config['incr'],self.minFreq,self.maxFreq, self.sampleRate)
        self.spectrogramDialog.show()
        self.spectrogramDialog.activateWindow()
        self.spectrogramDialog.activate.clicked.connect(self.spectrogram)
        # This next line was for dynamic update.
        # self.connect(self.spectrogramDialog, SIGNAL("changed"), self.spectrogram)

    def spectrogram(self):
        """ Listener for the spectrogram dialog.
        Has to do quite a bit of work to make sure segments are in the correct place, etc."""
        [windowType, mean_normalise, multitaper, window_width, incr, minFreq, maxFreq] = self.spectrogramDialog.getValues()
        if (minFreq >= maxFreq):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Incorrect frequency range")
            msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        with pg.BusyCursor():
            self.statusLeft.setText("Updating the spectrogram...")
            self.sp.setWidth(int(str(window_width)), int(str(incr)))
            oldSpecy = np.shape(self.sg)[1]
            sgRaw = self.sp.spectrogram(self.audiodata,window=str(windowType),mean_normalise=mean_normalise,onesided=True,multitaper=False)
            maxsg = np.min(sgRaw)
            self.sg = np.abs(np.where(sgRaw==0,0.0,10.0 * np.log10(sgRaw/maxsg)))

            # If the size of the spectrogram has changed, need to update the positions of things
            if int(str(incr)) != self.config['incr'] or int(str(window_width)) != self.config['window_width']:
                self.config['incr'] = int(str(incr))
                self.config['window_width'] = int(str(window_width))
                self.changeWidth(self.widthWindow.value())
                # Update the positions of the segments
                self.textpos = np.shape(self.sg)[1] + self.config['textoffset']
                for s in range(len(self.listRectanglesa2)):
                    if self.listRectanglesa1[s] is not None:
                        x1 = self.convertAmpltoSpec(self.listRectanglesa1[s].getRegion()[0])
                        x2 = self.convertAmpltoSpec(self.listRectanglesa1[s].getRegion()[1])
                        if type(self.listRectanglesa2[s]) == self.ROItype:
                            y1 = self.listRectanglesa2[s].pos().y()/oldSpecy
                            y2 = self.listRectanglesa2[s].size().y()/oldSpecy
                            self.listRectanglesa2[s].setPos(pg.Point(x1, y1*np.shape(self.sg)[1]))
                            self.listRectanglesa2[s].setSize(pg.Point(x2 - x1, y2*np.shape(self.sg)[1]))
                        else:
                            self.listRectanglesa2[s].setRegion([x1, x2])
                        self.listLabels[s].setPos(x1,self.textpos)

                # Update the axis
                #FreqRange = (self.maxFreq - self.minFreq)/1000.
                #self.specaxis.setTicks([[(0, self.minFreq / 1000.),
                #                         (np.shape(self.sg)[1] / 4, self.minFreq / 1000. + FreqRange / 4.),
                #                         (np.shape(self.sg)[1] / 2, self.minFreq / 1000. + FreqRange / 2.),
                #                         (3 * np.shape(self.sg)[1] / 4, self.minFreq / 1000. + 3 * FreqRange / 4.),
                #                         (np.shape(self.sg)[1], self.minFreq / 1000. + FreqRange)]])

                # Redraw everything and redraw it
                self.removeSegments(delete=False)
                for r in self.SegmentRects:
                    self.p_overview2.removeItem(r)
                self.SegmentRects = []
                self.p_overview.removeItem(self.overviewImageRegion)

                self.drawOverview()
                self.drawfigMain()
                if hasattr(self, 'seg'):
                    self.seg.setNewData(self.audiodata, sgRaw, self.sampleRate, self.config['window_width'],
                                        self.config['incr'])

            self.redoFreqAxis(minFreq,maxFreq)

            self.statusLeft.setText("Ready")

    def denoiseDialog(self):
        """ Create the denoising dialog when the relevant button is pressed.
        """
        self.denoiseDialog = Dialogs.Denoise(DOC=self.DOC,sampleRate=self.sampleRate)
        self.denoiseDialog.show()
        self.denoiseDialog.activateWindow()
        self.denoiseDialog.activate.clicked.connect(self.denoise)
        self.denoiseDialog.undo.clicked.connect(self.denoise_undo)
        self.denoiseDialog.save.clicked.connect(self.denoise_save)

    def backup(self):
        """ Enables denoising to be undone. """
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
        """ Listener for the denoising dialog.
        Calls the denoiser and then plots the updated data.
        """
        # TODO: should it be saved automatically, or a button added?
        if self.CLI:
            # in CLI mode, default values will be retrieved from dialogs.
            self.denoiseDialog = Dialogs.Denoise(DOC=self.DOC,sampleRate=self.sampleRate)
            # values can be passed here explicitly, e.g.:
            # self.denoiseDialog.depth.setValue(10)
            # or could add an argument to pass custom defaults, e.g.:
            # self.denoiseDialog = Dialogs.Denoise(defaults=("wt", 1, 2, 'a')
        with pg.BusyCursor():
            opstartingtime = time.time()
            print("Denoising requested at " + time.strftime('%H:%M:%S', time.gmtime(opstartingtime)))
            self.statusLeft.setText("Denoising...")
            if self.DOC==False:
                [alg,depthchoice,depth,thrType,thr,wavelet,start,end,width,trimaxis] = self.denoiseDialog.getValues()
            else:
                [alg, start, end, width, trimaxis] = self.denoiseDialog.getValues()
            self.backup()
            if not hasattr(self, 'waveletDenoiser'):
                self.waveletDenoiser = WaveletFunctions.WaveletFunctions(data=self.audiodata,wavelet=None,maxLevel=self.config['maxSearchDepth'])

            if str(alg) == "Wavelets" and self.DOC==False:
                if thrType is True:
                    type = 'Soft'
                else:
                    type = 'Hard'
                if depthchoice:
                    depth = None
                else:
                    depth = int(str(depth))
                self.audiodata = self.waveletDenoiser.waveletDenoise(self.audiodata,type,float(str(thr)),depth,wavelet=str(wavelet))
            elif str(alg) == "Wavelets" and self.DOC==True:
                self.audiodata = self.waveletDenoiser.waveletDenoise(self.audiodata)

            elif str(alg) == "Bandpass --> Wavelets" and self.DOC==False:
                if thrType is True:
                    type = 'soft'
                else:
                    type = 'hard'
                if depthchoice:
                    depth = None
                else:
                    depth = int(str(depth))
                self.audiodata = self.sp.bandpassFilter(self.audiodata,int(str(start)),int(str(end)))
                self.audiodata = self.waveletDenoiser.waveletDenoise(self.audiodata,type,float(str(thr)),depth,wavelet=str(wavelet))
            elif str(alg) == "Wavelets --> Bandpass" and self.DOC==False:
                if thrType is True:
                    type = 'soft'
                else:
                    type = 'hard'
                if depthchoice:
                    depth = None
                else:
                    depth = int(str(depth))
                self.audiodata = self.waveletDenoiser.waveletDenoise(self.audiodata,type,float(str(thr)),depth,wavelet=str(wavelet))
                self.audiodata = self.sp.bandpassFilter(self.audiodata,int(str(start)),int(str(end)))

            elif str(alg) == "Bandpass":
                self.audiodata = self.sp.bandpassFilter(self.audiodata, int(str(start)), int(str(end)))
                #self.audiodata = self.sp.ButterworthBandpass(self.audiodata, self.sampleRate, low=int(str(start)), high=int(str(end)))
                if trimaxis:
                    self.redoFreqAxis(int(str(start)), int(str(end)))
            elif str(alg) == "Butterworth Bandpass":
                self.audiodata = self.sp.ButterworthBandpass(self.audiodata, self.sampleRate, low=int(str(start)), high=int(str(end)))
            else:
                #"Median Filter"
                self.audiodata = self.sp.medianFilter(self.audiodata,int(str(width)))

            print("Denoising w/o temp file completed in %.4f seconds" % (time.time() - opstartingtime))

            ## Creating temp files
            # The temp files work properly on a Mac, not on Windows
            if platform.system() == 'Darwin':
                filename = 'temp.wav'
            else:
                import tempfile
                f = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
                filename = f.name

            wavio.write(filename,self.audiodata.astype('int16'),self.sampleRate,scale='dtype-limits',sampwidth=2)

            #open the temp file
            wavobj = wavio.read(filename)
            self.sampleRate = wavobj.rate
            self.audiodata = wavobj.data
            if self.audiodata.dtype is not 'float':
                self.audiodata = self.audiodata.astype('float') #/ 32768.0

            self.audiodata=self.audiodata[:,0]

            sgRaw = self.sp.spectrogram(self.audiodata,mean_normalise=True,onesided=True,multitaper=False)
            maxsg = np.min(sgRaw)
            self.sg = np.abs(np.where(sgRaw==0,0.0,10.0 * np.log10(sgRaw/maxsg)))
            self.overviewImage.setImage(self.sg)

            self.specPlot.setImage(self.sg)
            self.amplPlot.setData(np.linspace(0.0,float(self.datalength)/self.sampleRate,num=self.datalength,endpoint=True),self.audiodata)
            self.minFreq = int(str(start))
            self.maxFreq = int(str(end))

            if trimaxis:
                self.redoFreqAxis()

            self.setColourLevels()

            print("Denoising completed in %s seconds" % round(time.time() - opstartingtime, 4))
            self.statusLeft.setText("Ready")

        # TODO: Make the temp file playable
        # # media_obj = phonon.Phonon.MediaObject(self)
        # self.media_obj.setCurrentSource(phonon.Phonon.MediaSource(filename))
        # audio_output = phonon.Phonon.AudioOutput(phonon.Phonon.MusicCategory, self)
        # phonon.Phonon.createPath(self.media_obj, audio_output)
        # self.media_obj.tick.connect(self.movePlaySlider)
        # self.media_obj.setTickInterval(20)
        # self.media_obj.tick.connect(self.movePlaySlider)
        # self.media_obj.finished.connect(self.playFinished)

    def denoise_undo(self):
        """ Listener for undo button in denoising dialog.
        """
        # TODO: Can I actually delete something from an object?
        print("Undoing",np.shape(self.audiodata_backup))
        if hasattr(self,'audiodata_backup'):
            if self.audiodata_backup is not None:
                if np.shape(self.audiodata_backup)[1]>0:
                    self.audiodata = np.copy(self.audiodata_backup[:,-1])
                    self.audiodata_backup = self.audiodata_backup[:,:-1]
                    self.sp.setNewData(self.audiodata,self.sampleRate)
                    sgRaw = self.sp.spectrogram(self.audiodata,mean_normalise=True,onesided=True,multitaper=False)
                    maxsg = np.min(sgRaw)
                    self.sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))
                    self.overviewImage.setImage(self.sg)
                    self.specPlot.setImage(self.sg)
                    self.amplPlot.setData(
                        np.linspace(0.0, self.datalengthSec, num=self.datalength, endpoint=True),
                        self.audiodata)
                    if hasattr(self,'seg'):
                        self.seg.setNewData(self.audiodata,sgRaw,self.sampleRate,self.config['window_width'],self.config['incr'])

                    self.redoFreqAxis(0,self.sampleRate/2)
                    self.setColourLevels()

    def denoise_save(self):
        """ Listener for save button in denoising dialog.
        Adds _d to the filename and saves it as a new sound file.
        """
        filename = self.filename[:-4] + '_d' + self.filename[-4:]
        wavio.write(filename,self.audiodata.astype('int16'),self.sampleRate,scale='dtype-limits', sampwidth=2)
        self.statusLeft.setText("Saved")

    def save_selected_sound(self, id=-1):
        """ Listener for 'Save selected sound' menu item.
        choose destination and give it a name
        """
        import math
        if self.box1id is None or self.box1id == -1:
            print("No box selected")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No sound selected to save")
            msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setWindowTitle("No segment")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        else:
            if type(self.listRectanglesa2[self.box1id]) == self.ROItype:
                x1 = self.listRectanglesa2[self.box1id].pos().x()
                x2 = x1 + self.listRectanglesa2[self.box1id].size().x()
            else:
                x1, x2 = self.listRectanglesa2[self.box1id].getRegion()
            x1 = math.floor(float(x1) * self.config['incr']) #/ self.sampleRate
            x2 = math.floor(float(x2) * self.config['incr']) #/ self.sampleRate
            #print x1, x2
            # filename = self.filename[:-4] + '_selected' + self.filename[-4:]
            filename, drop = QtGui.QFileDialog.getSaveFileName(self, 'Save File as', self.dirName, selectedFilter='*.wav')
            if filename:
                wavio.write(str(filename), self.audiodata[int(x1):int(x2)].astype('int16'), self.sampleRate, scale='dtype-limits', sampwidth=2)

    def redoFreqAxis(self,start=None,end=None):
        """ This is the listener for the menu option to make the frequency axis tight (after bandpass filtering)
        """
        if start is None:
            start = self.minFreq
        if end is None:
            end = self.maxFreq

        height = self.sampleRate / 2. / np.shape(self.sg)[1]

        self.overviewImage.setImage(self.sg[:,int(float(start)/height):int(float(end)/height)])
        self.specPlot.setImage(self.sg[:,int(float(start)/height):int(float(end)/height)])

        FreqRange = end - start
        SpecRange = FreqRange/height
        self.specaxis.setTicks([[(0,(start/1000.)),(SpecRange/4,(start/1000.+FreqRange/4000.)),(SpecRange/2,(start/1000.+FreqRange/2000.)),(3*SpecRange/4,(start/1000.+3*FreqRange/4000.)),(SpecRange,(start/1000.+FreqRange/1000.))]])

        self.textpos = int(float(end-start)/height) + self.config['textoffset']
        for i in range(len(self.segments)):
            if self.segments[i][0] >= self.startRead and self.segments[i][1] <= min(self.startRead + self.lenRead, float(self.fileLength) / self.sampleRate):
                self.listLabels[i].setPos(self.listLabels[i].pos()[0], self.textpos)

    def segmentationDialog(self):
        """ Create the segmentation dialog when the relevant button is pressed.
        """
        self.segmentDialog = Dialogs.Segmentation(np.max(self.audiodata))
        self.segmentDialog.show()
        self.segmentDialog.activateWindow()
        self.segmentDialog.undo.clicked.connect(self.segment_undo)
        self.segmentDialog.activate.clicked.connect(self.segment)

    def segment(self):
        """ Listener for the segmentation dialog. Calls the relevant segmenter.
        """
        if self.CLI:
            self.segmentDialog = Dialogs.Segmentation(np.max(self.audiodata))

        opstartingtime = time.time()
        print("Segmenting requested at " + time.strftime('%H:%M:%S', time.gmtime(opstartingtime)))

        # TODO: Currently just gives them all the label "Don't Know"
        seglen = len(self.segments)
        [alg, medThr,HarmaThr1,HarmaThr2,PowerThr,minfreq,minperiods,Yinthr,window,FIRThr1,CCThr1,species,resolution] = self.segmentDialog.getValues()

        #[alg, ampThr, medThr,HarmaThr1,HarmaThr2,PowerThr,minfreq,minperiods,Yinthr,window,FIRThr1,depth,thrType,thr,wavelet,bandchoice,start,end,species] = self.segmentDialog.getValues()
        with pg.BusyCursor():
            species = str(species)
            if species=='Choose species...':
                species='all'
            #if not hasattr(self,'seg'):
            #    self.seg = Segment.Segment(self.audiodata,sgRaw,self.sp,self.sampleRate,self.config['minSegment'],self.config['window_width'],self.config['incr'])
            self.statusLeft.setText("Segmenting...")
            if str(alg) == "Default":
                newSegments = self.seg.bestSegments()
            elif str(alg) == "Median Clipping":
                newSegments = self.seg.medianClip(float(str(medThr)),self.config['minSegment'])
            elif str(alg) == "Harma":
                newSegments = self.seg.Harma(float(str(HarmaThr1)),float(str(HarmaThr2)),self.config['minSegment'])
            elif str(alg) == "Power":
                newSegments = self.seg.segmentByPower(float(str(PowerThr)))
            elif str(alg) == "Onsets":
                newSegments = self.seg.onsets()
            elif str(alg) == "Fundamental Frequency":
                newSegments, pitch, times = self.seg.yin(int(str(minfreq)),int(str(minperiods)),float(str(Yinthr)),int(str(window)),returnSegs=True)
            elif str(alg) == "FIR":
                newSegments = self.seg.segmentByFIR(float(str(FIRThr1)))
            elif str(alg)=="Wavelets":
                if species == 'all':    # Ask the species
                    msg = QMessageBox()
                    msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
                    msg.setWindowIcon(QIcon('img/Avianz.ico'))
                    msg.setText("Please select your species!")
                    msg.setWindowTitle("Select Species")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return
                else:
                    ws = WaveletSegment.WaveletSegment(species=str(species))
                    newSegments = ws.waveletSegment_test(fName=None,data=self.audiodata, sampleRate=self.sampleRate, species=species,trainTest=False)
            elif str(alg)=="Cross-Correlation":
                self.findMatches(float(str(CCThr1)))
                newSegments = []

            # print "to excel", newSegments
                # # Here the idea is to use both ML and wavelets then label AND as definite and XOR as possible just for wavelets
                # # but ML is extremely slow and crappy. So I decided to use just the wavelets
                # newSegmentsML = WaveletSegment.findCalls_learn(fName=None,data=self.audiodata, sampleRate=self.sampleRate, species=species,trainTest=False)
                # print np.shape(newSegmentsML),type(newSegmentsML), newSegmentsML
                #
                # newSegments = WaveletSegment.findCalls_test(fName=None,data=self.audiodata, sampleRate=self.sampleRate, species='kiwi',trainTest=False)
                # # print type(newSegments),newSegments
                # import itertools
                # newSegments=list(itertools.chain.from_iterable(newSegments))
                # temp=np.zeros(len(newSegmentsML))
                # for i in newSegments:
                #     temp[i]=1
                # newSegments=temp.astype(int)
                # newSegments=newSegments.tolist()
                # print np.shape(newSegments), type(newSegments), newSegments
                #
                # newSegmentsDef=np.minimum.reduce([newSegmentsML,newSegments])
                # newSegmentsDef=newSegmentsDef.tolist()
                # print "newSegmentsDef:", np.shape(newSegmentsDef), type(newSegmentsDef), newSegmentsDef
                # C=[(a and not b) or (not a and b) for a,b in zip(newSegmentsML,newSegments)]
                # newSegmentsPb=[int(c) for c in C]
                # print "newSegmentsPosi:", np.shape(newSegmentsPb), type(newSegmentsPb), newSegmentsPb
                #
                # # convert these segments to [start,end] format
                # newSegmentsDef=self.binary2seg(newSegmentsDef)
                # newSegmentsPb=self.binary2seg(newSegmentsPb)

            # post process to remove short segments, wind, rain, and use F0 check.
            post = SupportClasses.postProcess(audioData=self.audiodata, sampleRate=self.sampleRate,
                                              segments=newSegments, species=species)
            if species == "Kiwi":
                post.short()    # species specific
                post.wind()
                post.rainClick()
                post.fundamentalFrq()   # species specific
            else: # do
                post.wind()
                post.rainClick()
            newSegments = post.segments
            # Save the excel file
            out = SupportClasses.exportSegments(species=species, startTime=self.startTime, segments=newSegments,dirName=self.dirName, filename=self.filename, datalength=self.datalength,sampleRate=self.sampleRate, method=str(alg),resolution=resolution)
            if generateExcel:
                out.excel()
            # self.exportSegments(newSegments,species=species)

            # Generate annotation friendly output.
            if str(alg)=="Wavelets":
                 if len(out.segments)>0:
                    for seg in newSegments:
                        self.addSegment(float(seg[0]), float(seg[1]), 0, 0,
                                        species.title() + "?",index=-1)
                        self.segmentsToSave = True

            else:
                if len(newSegments)>0:
                    for seg in newSegments:
                        self.addSegment(float(seg[0]),float(seg[1]))
                        self.segmentsToSave = True

            self.lenNewSegments = len(newSegments)
            self.segmentDialog.undo.setEnabled(True)
            self.statusLeft.setText("Ready")

    def segment_undo(self):
        """ Listener for undo button in segmentation dialog.
        This is very cheap: the segments were appended, so delete the last len of them (from the end)
        """
        end = len(self.segments)
        for seg in range(end-1,end-self.lenNewSegments-1,-1):
            self.deleteSegment(seg)
        self.segmentDialog.undo.setEnabled(False)

    def exportSeg(self, annotation=None, species='all'):
        out = SupportClasses.exportSegments(startTime=self.startTime, segments=self.segments, dirName=self.dirName, filename=self.filename,
                                               datalength=self.datalength, sampleRate=self.sampleRate)
        out.excel()

    def findMatches(self,thr=0.4):
        """ Calls the cross-correlation function to find matches like the currently highlighted box.
        """
        if self.box1id is None or self.box1id == -1:
            print("No box selected")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No segment selected to match")
            msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setWindowTitle("No segment")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        else:
            self.statusLeft.setText("Finding matches...")
            # Only want to draw new segments, so find out how many there are now
            seglen = len(self.segments)
            # Get the segment -- note that takes the full y range
            if type(self.listRectanglesa2[self.box1id]) == self.ROItype:
                x1 = self.listRectanglesa2[self.box1id].pos().x()
                x2 = x1 + self.listRectanglesa2[self.box1id].size().x()
            else:
                x1, x2 = self.listRectanglesa2[self.box1id].getRegion()
            # Get the data for the spectrogram
            sgRaw = self.sp.spectrogram(self.audiodata,mean_normalise=True,onesided=True,multitaper=False)
            segment = sgRaw[int(x1):int(x2),:]
            len_seg = (x2-x1) * self.config['incr'] / float(self.sampleRate)
            indices = self.seg.findCCMatches(segment,sgRaw,thr)
            # indices are in spectrogram pixels, need to turn into times
            for i in indices:
                # Miss out the one selected: note the hack parameter
                if np.abs(i-x1) > self.config['overlap_allowed']:
                    time = float(i)*self.config['incr'] / float(self.sampleRate)
                    self.addSegment(time, time+len_seg,0,0,self.segments[self.box1id][4])
            self.statusLeft.setText("Ready")

    def classifySegments(self):
        # TODO: Finish this
        # Note that this still works on 1 second -- species-specific parameter eventually (here twice: as 1 and in sec loop)
        if self.segments is None or len(self.segments) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No segments to recognise")
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
            msg.setWindowTitle("No segments")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        else:
            with pg.BusyCursor():
                # TODO: Ask for species; brown kiwi for now
                # TODO: ***** TIDY UP WAVELET SEG, USE THIS!
                for i in range(len(self.segments)):
                    seglength = np.abs(self.segments[i][1] - self.segments[i][0])
                    if seglength <= 1:
                        # Recognise as is
                        label = WaveletSegment.computeWaveletEnergy(self.audiodata[self.segments[i][0]:self.segments[i][1]],wavelet='dmey2')
                        self.updateText(label,i)
                    else:
                        for sec in range(np.ceil(seglength)):
                            label = WaveletSegment.computeWaveletEnergy(self.audiodata[sec*self.sampleRate+self.segments[i][0]:(sec+1)*self.sampleRate+self.segments[i][0]],wavelet='dmey2')
                            # TODO: Check if the labels match, decide what to do if not
                        self.updateText(label,i)

    def recognise(self):
        # This will eventually call methods to do automatic recognition
        # Actually, will produce a dialog to ask which species, etc.
        # TODO
        pass

# ===============
# Code for playing sounds
    def playVisible(self):
        """ Listener for button to play the visible area.
        On PLAY, turns to PAUSE and two other buttons turn to STOPs.
        """
        if self.media_obj.isPlaying():
            self.pausePlayback()
        else:
            if self.media_obj.state() != QAudio.SuspendedState and not self.media_obj.keepSlider:
                # restart playback
                range = self.p_ampl.viewRange()[0]
                self.setPlaySliderLimits(range[0]*1000, range[1]*1000)
                # (else keep play slider range from before)
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
            self.playSegButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
            self.playBandLimitedSegButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
            self.media_obj.pressedPlay(start=self.segmentStart, stop=self.segmentStop, audiodata=self.audiodata)

    def playSelectedSegment(self):
        """ Listener for PlaySegment button.
        Get selected segment start and end (or return if no segment selected).
        On PLAY, all three buttons turn to STOPs.
        """
        if self.media_obj.isPlaying():
            self.stopPlayback()
        else:
            if self.box1id > -1:
                self.stopPlayback()

                start = self.listRectanglesa1[self.box1id].getRegion()[0] * 1000 + self.startRead * 1000
                stop = self.listRectanglesa1[self.box1id].getRegion()[1] * 1000 + self.startRead * 1000

                self.setPlaySliderLimits(start, stop)
                self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
                self.playSegButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
                self.playBandLimitedSegButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
                self.media_obj.filterSeg(start, stop, self.audiodata)
            else:
                print("Can't play, no segment selected")

    def playBandLimitedSegment(self):
        """ Listener for PlayBandlimitedSegment button.
        Gets the band limits of the segment, bandpass filters, then plays that.
        Currently uses FIR bandpass filter -- Butterworth is commented out.
        On PLAY, all three buttons turn to STOPs.
        """
        if self.media_obj.isPlaying():
            self.stopPlayback()
        else:
            if self.box1id > -1:
                self.stopPlayback()
                # check frequency limits, + small buffer bands
                bottom = max(0.1, self.minFreq, self.segments[self.box1id][2] * self.sampleRate / 2.)
                top = min(self.segments[self.box1id][3] * self.sampleRate / 2., self.maxFreq-0.1)

                print("extracting samples between %d-%d Hz" % (bottom, top))
                # set segment limits as usual, in ms
                start = self.listRectanglesa1[self.box1id].getRegion()[0] * 1000
                stop = self.listRectanglesa1[self.box1id].getRegion()[1] * 1000
                self.setPlaySliderLimits(start, stop)
                self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
                self.playSegButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
                self.playBandLimitedSegButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))

                # filter the data into a temporary file or buffer
                self.media_obj.filterBand(self.segmentStart, self.segmentStop, bottom, top, self.audiodata, self.sp)
            else:
                print("Can't play, no segment selected")

    def pausePlayback(self):
        """ Restores the PLAY buttons, calls media_obj to pause playing."""
        self.media_obj.pressedPause()

        # Reset all button icons:
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playSegButton.setIcon(QtGui.QIcon('img/playsegment.png'))
        self.playBandLimitedSegButton.setIcon(QtGui.QIcon('img/playBandLimited.png'))

    def stopPlayback(self):
        """ Restores the PLAY buttons, slider, text, calls media_obj to stop playing."""
        self.media_obj.pressedStop()
        if not hasattr(self, 'segmentStart') or self.segmentStart is None:
            self.segmentStart = 0
        self.playSlider.setValue(-1000)
        self.bar.setValue(-1000)
        self.timePlayed.setText(self.convertMillisecs(self.segmentStart) + "/" + self.totalTime)

        # Reset all button icons:
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playSegButton.setIcon(QtGui.QIcon('img/playsegment.png'))
        self.playBandLimitedSegButton.setIcon(QtGui.QIcon('img/playBandLimited.png'))

    def movePlaySlider(self):
        """ Listener called on sound notify (every 20 ms).
        Controls the slider, text timer, and listens for playback finish.
        """
        self.media_obj.time = self.media_obj.time + 20 # in ms. TODO: not hardcode notifyInterval

        # listener for playback finish. Note small buffer for catching up
        if self.media_obj.time > (self.segmentStop-10):
            print("stopped at %d ms" % self.media_obj.time)
            self.stopPlayback()
        else:
            self.playSlider.setValue(self.media_obj.time)
            self.timePlayed.setText(self.convertMillisecs(self.media_obj.time) + "/" + self.totalTime)
            # playSlider.value() is in ms, need to convert this into spectrogram pixels
            self.bar.setValue(self.convertAmpltoSpec(self.media_obj.time / 1000.0 - self.startRead))

    def setPlaySliderLimits(self, start, end):
        """ Uses start/end in ms, does what it says, and also seeks file position marker.
        """
        self.playSlider.setRange(start + 1000.0 * self.startRead, end + 1000.0 * self.startRead)
        self.segmentStart = self.playSlider.minimum()
        self.segmentStop = self.playSlider.maximum()

    def volSliderMoved(self, value):
        self.media_obj.applyVolSlider(value)

    def barMoved(self, evt):
        """ Listener for when the bar showing playback position moves.
        """
        self.playSlider.setValue(self.convertSpectoAmpl(evt.x()) * 1000 + self.startRead * 1000)
        self.media_obj.seekToMs(self.convertSpectoAmpl(evt.x()) * 1000 + self.startRead * 1000, self.segmentStart)

    def setOperatorReviewerDialog(self):
        """ Listener for Set Operator/Reviewer menu item.
        """
        if hasattr(self, 'operator') and hasattr(self, 'reviewer') :
            self.setOperatorReviewerDialog = Dialogs.OperatorReviewer(operator=self.operator,reviewer=self.reviewer)
        else:
            self.setOperatorReviewerDialog = Dialogs.OperatorReviewer(operator='', reviewer='')
        self.setOperatorReviewerDialog.show()
        self.setOperatorReviewerDialog.activateWindow()
        self.setOperatorReviewerDialog.activate.clicked.connect(self.changeOperator)

    def changeOperator(self):
        """ Listener for the operator/reviewer dialog.
        """
        name1, name2 = self.setOperatorReviewerDialog.getValues()
        self.operator = str(name1)
        self.reviewer = str(name2)
        self.statusRight.setText("Operator: " + self.operator + ", Reviewer: "+self.reviewer)
        self.setOperatorReviewerDialog.close()
        #self.segmentsToSave = True

    def saveImage(self, imageFile=''):
        import pyqtgraph.exporters as pge
        exporter = pge.ImageExporter(self.w_spec.scene())

        if imageFile=='':
            imageFile, drop = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.xpm *.jpg)");
        try:
            # works but requires devel (>=0.11) version of pyqtgraph:
            exporter.export(imageFile)
            print("Exporting spectrogram to file %s" % imageFile)
        except:
            print("Failed to save image")

    def changeSettings(self):
        """ Create the parameter tree when the Interface settings menu is pressed.
        """
        # first save the annotations
        # self.saveSegments()

        birdList = [str(item) for item in self.config['BirdList']]
        bl = ""
        for i in range(len(birdList)):
            bl += birdList[i]
            bl += '\n'

        params = [
            {'name': 'Mouse settings', 'type' : 'group', 'children': [
                {'name': 'Invert mouse', 'type': 'bool', 'tip': 'If true, segments are drawn with right clicking.',
                 'value': self.config['drawingRightBtn']},
                {'name': 'Mark by dragging', 'type': 'bool', 'tip': 'If false, mark by clicking',
                 'value': self.config['requireDrag']}
            ]},
            {'name': 'Paging', 'type': 'group', 'children': [
                {'name': 'Page size', 'type': 'int', 'value': self.config['maxFileShow'], 'limits': (5, 900),
                 'step': 5,
                 'suffix': ' sec'},
                {'name': 'Page overlap', 'type': 'int', 'value': self.config['fileOverlap'], 'limits': (0, 20),
                 'step': 2,
                 'suffix': ' sec'},
            ]},

            {'name': 'Annotation', 'type': 'group', 'children': [
                {'name': 'Auto save segments every', 'type': 'float', 'value': self.config['secsSave'], 'step': 5,
                 'limits': (5, 900),
                 'suffix': ' sec'},
                {'name': 'Annotation overview cell length', 'type': 'float',
                 'value': self.config['widthOverviewSegment'],
                 'limits': (5, 300), 'step': 5,
                 'suffix': ' sec'},
                {'name': 'Segment colours', 'type': 'group', 'children': [
                    {'name': 'Confirmed segments', 'type': 'color', 'value': self.config['ColourNamed'],
                     'tip': "Correctly labeled segments"},
                    {'name': 'Possible', 'type': 'color', 'value': self.config['ColourPossible'],
                     'tip': "Segments that need further approval"},
                    {'name': "Don't know", 'type': 'color', 'value': self.config['ColourNone'],
                     'tip': "Segments that are not labelled"},
                    {'name': 'Currently selected', 'type': 'color', 'value': self.config['ColourSelected'],
                     'tip': "Currently delected segment"},
                ]},
            ]},

            {'name': 'Human classify', 'type': 'group', 'children': [
                {'name': 'Save corrections', 'type': 'bool', 'value': self.config['saveCorrections'],
                 'tip': "This helps the developers"},
            ]},

            {'name': 'Output parameters', 'type': 'group', 'children': [
                {'name': 'Show all pages', 'type': 'bool', 'value': self.config['showAllPages'],
                 'tip': "Where to show segments from when looking at outputs"},
            ]},

            {'name': 'User', 'type': 'group', 'children': [
                {'name': 'Operator', 'type': 'str', 'value': self.config['operator'],
                 'tip': "Person name"},

                {'name': 'Reviewer', 'type': 'str', 'value': self.config['reviewer'],
                 'tip': "Person name"},
            ]},

            {'name': 'Bird List', 'type': 'group', 'children': [
                {'name': 'Add/Remove/Modify', 'type': 'text', 'value': bl}
                ]},
        ]

        ## Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', children=params)
        self.p.sigTreeStateChanged.connect(self.changeParams)
        ## Create ParameterTree widget
        self.t = ParameterTree()
        self.t.setParameters(self.p, showTop=False)
        self.t.show()
        self.t.setWindowTitle('AviaNZ - Interface Settings')
        self.t.setWindowIcon(QIcon('img/Avianz.ico'))
        self.t.setFixedSize(420, 550)

    def changeParams(self,param, changes):
        """ Update the config and the interface if anything changes in the tree
        """
        # first save the annotations
        self.saveSegments()

        for param, change, data in changes:
            path = self.p.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()

            if childName=='Annotation.Auto save segments every':
                self.config['secsSave']=data
            elif childName=='Annotation.Annotation overview cell length':
                self.config['widthOverviewSegment']=data
            elif childName == 'Mouse settings.Invert mouse':
                self.config['drawingRightBtn'] = data
                if self.config['drawingRightBtn']:
                    self.MouseDrawingButton = QtCore.Qt.RightButton
                else:
                    self.MouseDrawingButton = QtCore.Qt.LeftButton
            elif childName == 'Mouse settings.Mark by dragging':
                self.config['requireDrag'] = data
                self.p_ampl.enableDrag = self.config['requireDrag']
                self.p_spec.enableDrag = self.config['requireDrag']
            elif childName == 'Paging.Page size':
                self.config['maxFileShow'] = data
            elif childName=='Paging.Page overlap':
                self.config['fileOverlap']=data
            elif childName=='Human classify.Save corrections':
                self.config['saveCorrections'] = data
            elif childName=='Bird List.Add/Remove/Modify':
                self.config['BirdList'] = data.split('\n')
            elif childName=='Annotation.Segment colours.Confirmed segments':
                rgbaNamed = list(data.getRgb())
                if rgbaNamed[3] > 100:
                    rgbaNamed[3] = 100
                self.config['ColourNamed'] = rgbaNamed
                self.ColourNamed = QtGui.QColor(self.config['ColourNamed'][0], self.config['ColourNamed'][1],
                                                self.config['ColourNamed'][2], self.config['ColourNamed'][3])
                self.ColourNamedDark = QtGui.QColor(self.config['ColourNamed'][0], self.config['ColourNamed'][1],
                                                    self.config['ColourNamed'][2], 255)
            elif childName=='Annotation.Segment colours.Possible':
                rgbaVal = list(data.getRgb())
                if rgbaVal[3] > 100:
                    rgbaVal[3] = 100
                self.config['ColourPossible'] = rgbaVal
                self.ColourPossible = QtGui.QColor(self.config['ColourPossible'][0], self.config['ColourPossible'][1],
                                                   self.config['ColourPossible'][2], self.config['ColourPossible'][3])
                self.ColourPossibleDark = QtGui.QColor(self.config['ColourPossible'][0],
                                                       self.config['ColourPossible'][1],
                                                       self.config['ColourPossible'][2], 255)
            elif childName=="Annotation.Segment colours.Don't know":
                rgbaVal = list(data.getRgb())
                if rgbaVal[3] > 100:
                    rgbaVal[3] = 100
                self.config['ColourNone'] = rgbaVal
                self.ColourNone = QtGui.QColor(self.config['ColourNone'][0], self.config['ColourNone'][1],
                                               self.config['ColourNone'][2], self.config['ColourNone'][3])
                self.ColourNoneDark = QtGui.QColor(self.config['ColourNone'][0], self.config['ColourNone'][1],
                                                   self.config['ColourNone'][2], 255)
            elif childName=='Annotation.Segment colours.Currently selected':
                rgbaVal = list(data.getRgb())
                if rgbaVal[3] > 100:
                    rgbaVal[3] = 100
                self.config['ColourSelected'] = rgbaVal
                # update the interface
                self.ColourSelected = QtGui.QColor(self.config['ColourSelected'][0], self.config['ColourSelected'][1],
                                                   self.config['ColourSelected'][2], self.config['ColourSelected'][3])
                self.ColourSelectedDark = QtGui.QColor(self.config['ColourSelected'][0], self.config['ColourSelected'][1],
                                                   self.config['ColourSelected'][2], 255)
            elif childName=='Output parameters.Show all pages':
                self.config['showAllPages'] = data
            elif childName=='User.Operator':
                self.config['operator'] = data
                self.operator = data
                self.statusRight.setText("Operator: " + str(self.operator) + ", Reviewer: " + str(self.reviewer))

            elif childName == 'User.Reviewer':
                self.config['reviewer'] = data
                self.reviewer = data
                self.statusRight.setText("Operator: " + str(self.operator) + ", Reviewer: " + str(self.reviewer))

        # # Reload the file to make these changes take effect
        # self.resetStorageArrays()
        # # Reset the media player
        # if self.media_obj.state() == phonon.Phonon.PlayingState:
        #    self.media_obj.pause()
        #    self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))

        # Find the '/' in the fileName
        i=len(self.filename)-1
        while self.filename[i] != '/' and i>0:
            i = i-1
        #print self.filename[i:], type(self.filename)

        #if len(self.segments) > 0 or self.hasSegments:
            # if len(self.segments) > 0:
            #     if self.segments[0][0] > -1:
            #         self.segments.insert(0,
            #                              [-1, str(QTime().addSecs(self.startTime).toString('hh:mm:ss')), self.operator,
            #                               self.reviewer, -1])
            # else:
            #     self.segments.insert(0, [-1, str(QTime().addSecs(self.startTime).toString('hh:mm:ss')), self.operator,
            #                              self.reviewer, -1])
        self.saveSegments()

        self.resetStorageArrays()
        self.loadFile(self.filename[i+1:])

# ============
# Various actions: deleting segments, saving, quitting
    #def deleteSegment(self,id=-1,hr=False):
        #""" Listener for delete segment button, or backspace key. Also called when segments are deleted by the
        #human classify dialogs.
        #Deletes the segment that is selected, otherwise does nothing.
        #Updates the overview segments as well.
        #"""
        ## print id, self.box1id, not id
        #if not hr and (id<0 or not id):
            #id = self.box1id
#
        #if id>-1:
            ## Work out which overview segment this segment is in (could be more than one) and update it
            #inds = int(float(self.convertAmpltoSpec(self.segments[id][0]-self.startRead))/self.widthOverviewSegment)
            ## print type(int(float(self.convertAmpltoSpec(self.segments[id][1]-self.startRead))/self.widthOverviewSegment)), type(len(self.overviewSegments) - 1)
            #inde = min(int(float(self.convertAmpltoSpec(self.segments[id][1]-self.startRead))/self.widthOverviewSegment),len(self.overviewSegments) - 1)
            ## print "inde", inde

            #if self.segments[id][4] == "Don't Know":
                #self.overviewSegments[inds:inde+1,0] -= 1
            #elif self.segments[id][4][-1] == '?':
                #self.overviewSegments[inds:inde + 1, 2] -= 1
            #else:
                #self.overviewSegments[inds:inde + 1, 1] -= 1
            #for box in range(inds, inde + 1):
                #if self.overviewSegments[box,0] > 0:
                    #self.SegmentRects[box].setBrush(self.ColourNone)
                #elif self.overviewSegments[box,2] > 0:
                    #self.SegmentRects[box].setBrush(self.ColourPossible)
                #elif self.overviewSegments[box,1] > 0:
                    #self.SegmentRects[box].setBrush(self.ColourNamed)
                #else:
                    #self.SegmentRects[box].setBrush(pg.mkBrush('w'))
#
            #if self.listRectanglesa1[id] is not None:
                #self.p_ampl.removeItem(self.listRectanglesa1[id])
                #self.p_spec.removeItem(self.listRectanglesa2[id])
                #self.p_spec.removeItem(self.listLabels[id])
            #del self.listLabels[id]
            #del self.segments[id]
            #del self.listRectanglesa1[id]
            #del self.listRectanglesa2[id]
            #self.segmentsToSave = True
            #self.box1id = -1

    def deleteAll(self):
        """ Listener for delete all button.
        Checks if the user meant to do it, then calls removeSegments()
        """
        if len(self.segments) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No segments to delete")
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setIconPixmap(QPixmap("img/Owl_warning.png"))
            msg.setWindowTitle("No segments")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        else:
            msg = QMessageBox()
            msg.setIconPixmap(QPixmap("img/Owl_thinking.png"))
            msg.setWindowIcon(QIcon('img/Avianz.ico'))
            msg.setText("Are you sure you want to delete all segments?")
            msg.setWindowTitle("Delete All Segments")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            reply = msg.exec_()
            if reply == QMessageBox.Yes:
                self.removeSegments()
                self.segmentsToSave = True

            # reset segment playback buttons
            self.playSegButton.setEnabled(False)
            self.playBandLimitedSegButton.setEnabled(False)

    def removeSegments(self,delete=True):
        """ Remove all the segments in response to the menu selection, or when a new file is loaded. """
        for r in self.listLabels:
            if r is not None:
                self.p_spec.removeItem(r)
        for r in self.listRectanglesa1:
            if r is not None:
                self.p_ampl.removeItem(r)
        for r in self.listRectanglesa2:
            if r is not None:
                self.p_spec.removeItem(r)
        for r in self.SegmentRects:
            r.setBrush(pg.mkBrush('w'))
            r.update()

        if delete:
            self.segments=[]
            self.listRectanglesa1 = []
            self.listRectanglesa2 = []
            self.listLabels = []
            self.box1id = -1

    def saveSegments(self):
        # TODO: Fix this up to include quitting stuff
        """ Save the segmentation data as a json file.
        Name of the file is the name of the wave file + .data"""

        # def checkSave():
        #     msg = QMessageBox()
        #     msg.setIcon(QMessageBox.Information)
        #     msg.setText("Do you want to save?")
        #     msg.setInformativeText("You didn't identify any segments, are you sure you want to save this annotation?")
        #     msg.setWindowTitle("No segments")
        #     msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        #     msg.buttonClicked.connect(msgbtn)
        #     retval = msg.exec_()
        #     print "value of pressed message box button:", retval
        #     return retval

        if self.segmentsToSave:
            print("Saving segments to " + self.filename)
            if len(self.segments) > 0:
                if self.segments[0][0] > -1:
                    self.segments.insert(0,
                                         [-1, str(QTime().addSecs(self.startTime).toString('hh:mm:ss')),
                                          self.operator,
                                          self.reviewer, -1])
            else:
                self.segments.insert(0,
                                     [-1, str(QTime().addSecs(self.startTime).toString('hh:mm:ss')), self.operator,
                                      self.reviewer, -1])

            if isinstance(self.filename, str):
                file = open(self.filename + '.data', 'w')
            else:
                file = open(str(self.filename) + '.data', 'w')
            json.dump(self.segments,file)
            self.segmentsToSave = False
            del self.segments[0]
        else:
            print("Nothing to save")

    def closeEvent(self, event):
        """ Catch the user closing the window by clicking the Close button instead of quitting. """
        self.quit()

    def quit(self):
        """ Listener for the quit button, also called by closeEvent().
        Add in the operator and reviewer at the top, and then save the segments and the config file.
        """

        print("Quitting")
        # if len(self.segments) > 0:
        #     if self.segments[0][0] > -1:
        #         self.segments.insert(0, [-1, str(QTime().addSecs(self.startTime).toString('hh:mm:ss')), self.operator,
        #                                  self.reviewer, -1])
        #     #else:
        #     #    retval = checkSave()
        # else:
        #     # TODO: This means that a file is always created. Is that a bug? Option: ask user -> wording?
        #     #retval = checkSave()
        #     if self.segments[0][0] > -1:
        #         self.segments.insert(0, [-1, str(QTime().addSecs(self.startTime).toString('hh:mm:ss')), self.operator,
        #                              self.reviewer, -1])
        self.saveSegments()
        if self.saveConfig == True:
            print("Saving config file")
            json.dump(self.config, open(self.configfile, 'w'))
        QApplication.quit()

# =============

@click.command()
@click.option('-c', '--cli', is_flag=True, help='Run in command-line mode')
@click.option('-f', '--infile', type=click.Path(), help='Input wav file (mandatory in CLI mode)')
@click.option('-o', '--imagefile', type=click.Path(), help='If specified, a spectrogram will be saved to this file')
@click.argument('command', nargs=-1)
def mainlauncher(cli, infile, imagefile, command):
    if cli:
        print("Starting AviaNZ in CLI mode")
        if not isinstance(infile, str):
            print("ERROR: valid input file (-f) is mandatory in CLI mode!")
            sys.exit()
        avianz = AviaNZ(configfile='AviaNZconfig.txt',DOC=DOC,CLI=True,firstFile=infile, imageFile=imagefile, command=command)
        print("Analysis complete, closing AviaNZ")
    else:
        print("Starting AviaNZ in GUI mode")
        # This screen asks what you want to do, then processes the response
        first = Dialogs.StartScreen(DOC=DOC)
        first.setWindowIcon(QtGui.QIcon('img/AviaNZ.ico'))
        first.show()
        app.exec_()
        
        task = first.getValues()

        if task == 1:
            avianz = AviaNZ(DOC=DOC)
            avianz.setWindowIcon(QtGui.QIcon('img/AviaNZ.ico'))
        elif task==2:
            avianz = interface_FindSpecies.AviaNZFindSpeciesInterface(DOC=DOC)
            avianz.setWindowIcon(QtGui.QIcon('img/AviaNZ.ico'))
            avianz.show()
            app.exec_()

        avianz.show()
        app.exec_()

DOC=False    # only DOC features or all
generateExcel=False

# Start the application
app = QApplication(sys.argv)
mainlauncher()
