
# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2025

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

from PyQt6 import QtGui
from PyQt6.QtGui import *
from PyQt6.QtWidgets import QLabel, QDialog, QPushButton, QHBoxLayout, QVBoxLayout # listing some explicitly to make syntax checks lighter
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt

import pyqtgraph as pg

import numpy as np
from src.ui.colourMaps import colourMaps
from src.ui.components.buttons_and_controls import BrightContrVol, PicButton



pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class HumanClassify2(QDialog):
    """ Single Species review main dialog.
        Puts all segments of a certain species together on buttons, and their labels.
        Allows quick confirm/leave/delete check over many segments.

        Construction:
        1. a list of Spectrogram containing spectrograms for ALL the segments in arg2
        2. SegmentList. Just provide full versions of this,
          and this dialog will select the needed segments.
        3. indices of segments to show (i.e. the selected species and current page)
        4. name of the species that we are reviewing
        5-8. spec color parameters
        9-10. guide positions and colors for batmode
        11. Loop playback or not?
        12. Filename - just for setting the window title
    """

    def __init__(self, sps, sgs, segments, indicestoshow, label, lut, cmapInverted, brightness, contrast, guidefreq=None, guidecol=None, loop=False, filename=None, lazyLoadParams=None, saveCallback=None):
        QDialog.__init__(self)

        if len(segments)==0:
            print("No segments provided")
            return

        if filename:
            self.setWindowTitle('Human review - ' + filename)
        else:
            self.setWindowTitle('Human review')

        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint)

        self.lazyLoadParams = lazyLoadParams
        self.lazyLoadEnabled = lazyLoadParams is not None
        self.saveCallback = saveCallback
        
        if self.lazyLoadEnabled:
            self.sps = [None] * len(indicestoshow)
            self.sgs = [None] * len(indicestoshow)
            self.loadedPages = set()
        else:
            self.sps = sps
            self.sgs = sgs
            haveaudio = all(len(sp.audio_data.data)>0 for sp in sps if sp is not None)

        self.lut = lut
        self.cmapInverted = cmapInverted

        self.segments = segments
        self.indices2show = indicestoshow

        self.errors = []

        self.loop = loop

        # Volume, brightness and contrast sliders.
        # Need to pass true (config) values to set up correct initial positions
        self.specControls = BrightContrVol(brightness, contrast, self.cmapInverted)
        self.specControls.colChanged.connect(self.setColourLevels)
        self.specControls.volChanged.connect(self.volSliderMoved)
        self.specControls.layout().addStretch(3) # add a big stretchable outer margin

        # Batmode customizations:
        self.guidefreq = guidefreq
        self.guidecol = guidecol
        if self.lazyLoadEnabled:
            haveaudio = not lazyLoadParams.get('batmode', False)
        if not haveaudio:
            self.specControls.volSlider.setEnabled(False)
            self.specControls.volIcon.setEnabled(False)

        label1 = QLabel('Click on the images that are incorrectly labelled or need some change.')
        label1.setFont(QtGui.QFont('SansSerif', 10))
        species = QLabel(label)
        species.setStyleSheet("padding: 2px 0px 5px 0px")
        font = QtGui.QFont('SansSerif', 12)
        font.setBold(True)
        species.setFont(font)

        # Species label and sliders
        vboxTop = QVBoxLayout()
        vboxTop.addWidget(label1)
        vboxTop.addWidget(species)
        vboxTop.addWidget(self.specControls)

        # Controls at the bottom
        # self.buttonPrev = QtGui.QToolButton()
        # self.buttonPrev.setArrowType(Qt.LeftArrow)
        # self.buttonPrev.setIconSize(QSize(30,30))
        # self.buttonPrev.clicked.connect(self.prevPage)

        # TODO: Is this useful?
        self.pageLabel = QLabel()

        self.none = QPushButton("Toggle all")
        #self.none.setSizePolicy(QSizePolicy(5,5))
        self.none.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.none.setMaximumSize(250, 30)
        self.none.clicked.connect(self.toggleAll)

        # Either the next or finish button is visible. They have different internal
        # functionality, but look the same to the user
        self.next = QPushButton("Next")
        #self.next.setSizePolicy(QSizePolicy(5,5))
        self.next.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.next.setMaximumSize(250, 30)
        self.next.clicked.connect(self.nextPage)

        self.finish = QPushButton("Next")
        #self.finish.setSizePolicy(QSizePolicy(5,5))
        self.finish.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.finish.setMaximumSize(250, 30)

        # Movement buttons and page numbers
        self.vboxBot = QHBoxLayout()
        # vboxBot.addWidget(self.buttonPrev)
        # vboxBot.addSpacing(20)
        self.vboxBot.addWidget(self.pageLabel)
        self.vboxBot.addSpacing(20)
        self.vboxBot.addWidget(self.none)
        self.vboxBot.addWidget(self.next)

        # Create the button objects, and we'll show them as needed
        # (fills self.buttons)
        self.flowContainer = QWidget()
        self.flowLayout = QGridLayout()
        self.flowLayout.setSpacing(10)
        self.flowContainer.setLayout(self.flowLayout)
        self.flowContainer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.flowLayout.setRowStretch(0, 100)
        self.flowLayout.setColumnStretch(0, 100)

        # this is the starting index of the first button, which we change when we flip pages
        self.butStart = 0
        self.maxCols = 5
        self.maxRows = 5

        self.createButtons()
        self.specControls.emitAll()  # applies initial colour, volume levels

        self.countPages()

        # Set overall layout of the dialog
        self.vboxFull = QVBoxLayout()
        # self.vboxFull.setSpacing(0)
        self.vboxFull.addLayout(vboxTop)
        self.vboxSpacer = QSpacerItem(1,1, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.vboxFull.addItem(self.vboxSpacer)
        self.vboxFull.addWidget(self.flowContainer)
        self.vboxFull.addLayout(self.vboxBot)
        # Must be fixed size!
        vboxTop.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        # Must be fixed size!
        self.vboxBot.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        # We need to know the true size of space available for flowLayout.
        # The idea is that spacer absorbs all height changes
        #self.setSizePolicy(1,1)
        self.setSizePolicy(QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Minimum)
        self.setLayout(self.vboxFull)
        self.vboxFull.setStretch(1, 100)
        # Plan B could be to measure the sizes of the top/bottom boxes and subtract them
        # self.boxSpaceAdjustment = vboxTop.sizeHint().height() + vboxBot.sizeHint().height()

    def closeEvent(self, event):
        if self.saveCallback:
            self.saveCallback(self)
        event.accept()

    def loadSpectrogramsForPage(self, pageNum):
        if not self.lazyLoadEnabled:
            return
        
        if pageNum in self.loadedPages:
            return
            
        from src.core import spectrogram
        from src.core import signal_proc
        
        buttonsPerPage = self.maxRows * self.maxCols
        startIdx = pageNum * buttonsPerPage
        endIdx = min(startIdx + buttonsPerPage, len(self.indices2show))
        
        params = self.lazyLoadParams
        config = params['config']
        
        for localIdx in range(startIdx, endIdx):
            globalIdx = self.indices2show[localIdx]
            
            if self.sps[localIdx] is not None:
                continue
                
            segData = params['segmentData'][globalIdx]
            filename = segData['filename']
            seg = segData['segment']
            chunksize = segData['chunksize']
            batmode = segData['batmode']
            duration = segData['duration']
            samplerate = segData['samplerate']
            minFreq = segData['minFreq']
            maxFreq = segData['maxFreq']
            
            sp = spectrogram.Spectrogram(config['window_width'], config['incr'], minFreq, maxFreq)
            
            if chunksize > 0:
                halfChunk = 1.1/2 * chunksize
                mid = (seg.start_time + seg.end_time)/2
                x1 = max(0, mid-halfChunk)
                x2 = min(duration, mid+halfChunk)
                x1nob = max(seg.start_time, x1)
                x2nob = min(seg.end_time, x2)
            else:
                x1nob = seg.start_time
                x2nob = seg.end_time
                x1 = max(x1nob - config['reviewSpecBuffer'], 0)
                x2 = min(x2nob + config['reviewSpecBuffer'], duration)
            
            if batmode:
                sp.readSoundFile(filename, offset=x1, duration=x2-x1, silent=True)
                sp.sg = sp.normalisedSpec("Batmode")
            else:
                sp.readSoundFile(filename, offset=x1, duration=x2-x1, silent=True)
                sp.audio_data.data = signal_proc.bandpass_filter(sp.audio_data.data, sp.audio_data.sample_rate, minFreq, maxFreq)
                sp.sg = sp.spectrogram(window_width=config['window_width'], 
                                     incr=config['incr'],
                                     window=config['windowType'],
                                     sgType=config['sgType'],
                                     sgScale=config['sgScale'],
                                     nfilters=config['nfilters'],
                                     mean_normalise=config['sgMeanNormalise'],
                                     equal_loudness=config['sgEqualLoudness'],
                                     onesided=config['sgOneSided'])
                
                height = sp.audio_data.sample_rate//2 / np.shape(sp.sg)[1]
                pixelstart = int(minFreq/height)
                pixelend = int(maxFreq/height)
                sp.sg = sp.sg[:,pixelstart:pixelend]
            
            sp.x1nobspec = sp.convertAmpltoSpec(x1nob-x1)
            sp.x2nobspec = sp.convertAmpltoSpec(x2nob-x1)
            
            self.sps[localIdx] = sp
            if batmode:
                self.sgs[localIdx] = sp.sg
            else:
                self.sgs[localIdx] = sp.normalisedSpec(config['sgNormMode'])
        
        self.loadedPages.add(pageNum)

    def loadAndCreateButtonsForPage(self, pageNum):
        if not self.lazyLoadEnabled:
            return
        
        if pageNum in self.loadedPages:
            return
            
        self.loadSpectrogramsForPage(pageNum)
        
        buttonsPerPage = self.maxRows * self.maxCols
        startIdx = pageNum * buttonsPerPage
        endIdx = min(startIdx + buttonsPerPage, len(self.indices2show))
        
        for localIdx in range(startIdx, endIdx):
            if self.buttons[localIdx] is not None:
                continue
                
            sp = self.sps[localIdx]
            if sp is None:
                continue
                
            globalIdx = self.indices2show[localIdx]
            duration = len(sp.audio_data.data)/sp.audio_data.sample_rate
            sg = self.sgs[localIdx]
            sg = np.fliplr(sg)

            self.minsg = min(self.minsg, np.min(sg))
            self.maxsg = max(self.maxsg, np.max(sg))

            if self.guidefreq is not None:
                gy = [0]*len(self.guidefreq)
                for gix in range(len(self.guidefreq)):
                    gy[gix] = sp.convertFreqtoY(self.guidefreq[gix])
            else:
                gy = None

            newButton = PicButton(globalIdx, sg, sp, sp.audio_data, duration, sp.x1nobspec, sp.x2nobspec, self.lut, guides=gy, guidecol=self.guidecol, loop=self.loop, scaleToButton=True)
            newButton.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
            newButton.setMinimumSize(10, 10)
            newButton.buttonClicked = False
            self.buttons[localIdx] = newButton

    def createButtons(self):
        """ Create the button objects, add audio, calculate spec, etc.
            So that when users flips through pages, we only need to
            retrieve the right ones from resizeEvent.
            No return, fills out self.buttons.
        """
        if self.lazyLoadEnabled:
            self.buttons = [None] * len(self.indices2show)
            self.marked = [False] * len(self.indices2show)
            self.minsg = 1
            self.maxsg = 1
            self.loadAndCreateButtonsForPage(0)
        else:
            self.buttons = []
            self.marked = []
            self.minsg = 1
            self.maxsg = 1
            for localIdx, globalIdx in enumerate(self.indices2show):
                sp = self.sps[localIdx]
                duration = len(sp.audio_data.data)/sp.audio_data.sample_rate
                sg = self.sgs[localIdx]
                sg = np.fliplr(sg)

                self.minsg = min(self.minsg, np.min(sg))
                self.maxsg = max(self.maxsg, np.max(sg))

                if self.guidefreq is not None:
                    gy = [0]*len(self.guidefreq)
                    for gix in range(len(self.guidefreq)):
                        gy[gix] = sp.convertFreqtoY(self.guidefreq[gix])
                else:
                    gy = None

                newButton = PicButton(globalIdx, sg, sp, sp.audio_data, duration, sp.x1nobspec, sp.x2nobspec, self.lut, guides=gy, guidecol=self.guidecol, loop=self.loop, scaleToButton=True)
                newButton.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
                newButton.setMinimumSize(10, 10)
                self.buttons.append(newButton)
                self.buttons[-1].buttonClicked=False
                self.marked.append(False)
        self.redrawButtons()

    def redrawButtons(self):
        if len(self.buttons) == 0:
            return
            
        exampleSP = None
        for sp in self.sps:
            if sp is not None:
                exampleSP = sp
                break
        
        if exampleSP is None:
            return
            
        minFreq = exampleSP.minFreqShow
        maxFreq = exampleSP.maxFreqShow
        if maxFreq==0:
            maxFreq = exampleSP.audio_data.sample_rate // 2
        if len(exampleSP.audio_data.data)>0:
            duration = len(exampleSP.audio_data.data)/exampleSP.audio_data.sample_rate
        else:
            duration = exampleSP.convertSpectoAmpl(np.shape(exampleSP.sg)[0])

        if self.lazyLoadEnabled:
            buttonsPerPage = self.maxRows * self.maxCols
            endPos = min(self.butStart + buttonsPerPage, len(self.buttons))
            buttonsToShow = [b for b in self.buttons[self.butStart:endPos] if b is not None]
            numButtonsToShow = len(buttonsToShow)
        else:
            numButtonsToShow = min(self.maxRows * self.maxCols, len(self.buttons) - self.butStart)

        numRows = min(self.maxRows, int(np.ceil(np.sqrt(numButtonsToShow))))
        numCols = min(self.maxCols, int(np.ceil(numButtonsToShow / numRows))) if numRows > 0 else 0

        for row in range(self.maxRows):
            if row < numRows:
                self.flowLayout.setRowStretch(row, 1)
            else:
                self.flowLayout.setRowStretch(row, 0)
        for col in range(self.maxCols):
            if col < numCols:
                self.flowLayout.setColumnStretch(col, 1)
            else:
                self.flowLayout.setColumnStretch(col, 0)

        butNum = 0
        for row in range(numRows):
            for col in range(numCols):
                actualIdx = self.butStart + butNum
                if actualIdx >= len(self.buttons):
                    break
                
                btn = self.buttons[actualIdx]
                if btn is not None:
                    self.flowLayout.addWidget(btn, row, col)
                    btn.show()
                
                butNum += 1
                if actualIdx + 1 >= len(self.buttons):
                    break
            if actualIdx + 1 >= len(self.buttons):
                break

        self.repaint()
        QApplication.processEvents()

    def volSliderMoved(self, value):
        try:
            for btn in self.buttons:
                if btn is not None:
                    btn.media_obj.applyVolSlider(value)
        except Exception:
            pass

    def countPages(self):
        """ Counts the total number of pages,
            finds where we are, how many remain, etc.
            Called on resize, so does not update current button position.
            Updates next/prev arrow states.
        """
        buttonsPerPage = self.maxRows * self.maxCols
        if buttonsPerPage == 0:
            # dialog still initializing or too small to show segments
            #self.buttonPrev.setEnabled(False)
            return
        # basically, count how many segments are "before" the current
        # top-lef one, and see how many pages we need to fit them.
        currpage = int(np.ceil(self.butStart / buttonsPerPage)+1)
        self.totalPages = max(int(np.ceil(len(self.buttons) / buttonsPerPage)),currpage)
        self.pageLabel.setText("Page %d out of %d" % (currpage, self.totalPages))

        if currpage == self.totalPages:
            try:
                self.vboxBot.removeWidget(self.next)
                self.next.setVisible(False)
                self.vboxBot.addWidget(self.finish)
                self.finish.setVisible(True)
            except:
                pass
        else:
            if self.finish.isVisible():
                try:
                    self.vboxBot.removeWidget(self.finish)
                    self.finish.setVisible(False)
                    self.vboxBot.addWidget(self.next)
                    self.next.setVisible(True)
                except:
                    pass

        self.repaint()

        #if currpage==1:
            #self.buttonPrev.setEnabled(False)
        #else:
            #self.buttonPrev.setEnabled(True)

    def nextPage(self):
        """ Called on arrow button clicks.
            Updates current segment position, and calls other functions
            to deal with actual page recount/redraw.
        """
        if self.saveCallback:
            self.saveCallback(self)
        
        buttonsPerPage = self.maxRows * self.maxCols
        self.clearButtons()
        self.butStart = min(len(self.buttons), self.butStart+buttonsPerPage)
        
        if self.lazyLoadEnabled:
            pageNum = self.butStart // buttonsPerPage
            with pg.BusyCursor():
                self.loadAndCreateButtonsForPage(pageNum)
        
        self.countPages()
        self.redrawButtons()

    def prevPage(self):
        """ Called on arrow button clicks.
            Updates current segment position, and calls other functions
            to deal with actual page recount/redraw.
        """
        if self.saveCallback:
            self.saveCallback(self)
        
        buttonsPerPage = self.maxRows * self.maxCols
        self.clearButtons()
        self.butStart = max(0, self.butStart-buttonsPerPage)
        
        if self.lazyLoadEnabled:
            pageNum = self.butStart // buttonsPerPage
            with pg.BusyCursor():
                self.loadAndCreateButtonsForPage(pageNum)
        
        self.countPages()
        self.redrawButtons()

    def clearButtons(self):
        for btn in self.buttons:
            if btn is not None:
                btn.stopPlayback()
        for i in reversed(range(self.flowLayout.count())):
            item = self.flowLayout.itemAt(i)
            widget = item.widget()
            if widget:
                self.flowLayout.removeWidget(widget)
                widget.hide()
                widget.setParent(None)

    def toggleAll(self):
        buttonsPerPage = self.maxRows * self.maxCols
        for butNum in range(self.butStart, min(self.butStart+buttonsPerPage, len(self.buttons))):
            btn = self.buttons[butNum]
            if btn is not None:
                btn.changePic(False)
        self.repaint()
        QApplication.processEvents()

    def setColourLevels(self, brightness, contrast):
        """ Listener for the brightness and contrast sliders being changed. Also called when spectrograms are loaded, etc.
        Translates the brightness and contrast values into appropriate image levels.
        """
        if not self.cmapInverted:
            brightness = 100-brightness

        colRange = colourMaps.getColourRange(self.minsg, self.maxsg, brightness, contrast, self.cmapInverted)

        for btn in self.buttons:
            if btn is not None:
                btn.stopPlayback()
                btn.setImage(colRange)
                btn.update()
