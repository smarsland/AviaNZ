
# Version 4.1 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2024

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

from PyQt6 import QtCore
from PyQt6.QtGui import *
from PyQt6.QtWidgets import QDialog, QLineEdit, QVBoxLayout, QScrollArea # listing some explicitly to make syntax checks lighter
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt

import pyqtgraph as pg

import numpy as np
from src.ui.colourMaps import colourMaps
from src.ui.components.buttons_and_controls import BrightContrVol, PicButton
from src.core import spectrogram

pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class Cluster(QDialog):
    def __init__(self, segments, sampleRate, classes, config, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Clustered segments')
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)

        if len(segments) == 0:
            print("No segments provided")
            return

        self.sampleRate = sampleRate
        self.segments = segments
        self.nclasses = classes
        self.config = config

        # Volume, brightness and contrast sliders.
        # Need to pass true (config) values to set up correct initial positions
        self.specControls = BrightContrVol(80, 20, False)
        self.specControls.colChanged.connect(self.setColourLevels)
        self.specControls.volChanged.connect(self.volSliderMoved)

        # Colour map
        self.lut = colourMaps.getLookupTable(self.config['cmap'])

        # set up the images
        self.flowLayout = pg.LayoutWidget()
        self.flowLayout.setGeometry(QtCore.QRect(0, 0, 380, 247))

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.flowLayout)

        # set overall layout of the dialog
        self.vboxFull = QVBoxLayout()
        self.vboxFull.addWidget(self.specControls)
        self.vboxFull.addWidget(self.scrollArea)
        self.setLayout(self.vboxFull)

        # Add the clusters to rows
        self.addButtons()
        self.updateButtons()

    def addButtons(self):
        """ Only makes the PicButtons and self.clusters dict
        TODO: Get the parameters for the spectrogram from the config
        """
        self.clusters = []
        self.picbuttons = []
        for i in range(self.nclasses):
            self.clusters.append((i, 'Type_' + str(i)))
        self.clusters = dict(self.clusters)  # Dictionary of {ID: cluster_name}

        # Create the buttons for each segment
        self.minsg = 1
        self.maxsg = 1
        for seg in self.segments:
            sp = spectrogram.Spectrogram(self.config['window_width'],self.config['incr'])
            sp.readSoundFile(seg[0], seg[1].end_time - seg[1].start_time, seg[1].start_time)
            #_ = sp.spectrogram(window='Hann', sgType='Standard',mean_normalise=True, onesided=True, need_even=False)
            self.sg = sp.spectrogram(window_width=self.config['window_width'], incr=self.config['incr'],window=self.config['windowType'],sgType=self.config['sgType'],sgScale=self.config['sgScale'],nfilters=self.config['nfilters'],mean_normalise=self.config['sgMeanNormalise'],equal_loudness=self.config['sgEqualLoudness'],onesided=self.config['sgOneSided'])
            self.sg = sp.normalisedSpec(self.config['sgNormMode'])

            self.minsg = min(self.minsg, np.min(self.sg))
            self.maxsg = max(self.maxsg, np.max(self.sg))

            newButton = PicButton(1, np.fliplr(self.sg), sp, sp.audio_data, seg[1].end_time - seg[1].start_time, 0, seg[1].end_time, self.lut, cluster=True)
            self.picbuttons.append(newButton)
        # (updateButtons will place them in layouts and show them)

    def updateButtons(self):
        """ Draw the existing buttons, and create check- and text-boxes.
        Called when merging clusters or initializing the page. """
        self.tboxes = []    # Corresponding list of text boxes
        for r in range(self.nclasses):
            c = 0
            tbox = QLineEdit(self.clusters[r])
            tbox.setMinimumWidth(80)
            tbox.setMaximumHeight(150)
            tbox.setStyleSheet("border: none;")
            tbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tboxes.append(tbox)
            self.flowLayout.addWidget(self.tboxes[-1], r, c)
            c += 1
            # Find the segments under this class and show them
            for segix in range(len(self.segments)):
                if self.segments[segix][-1] == r:
                    self.flowLayout.addWidget(self.picbuttons[segix], r, c)
                    c += 1
                    self.picbuttons[segix].show()
        self.flowLayout.update()
        self.specControls.emitAll()  # applies initial colour, volume levels

    def setColourLevels(self, brightness, contrast):
        """ Listener for the brightness and contrast sliders being changed. Also called when spectrograms are loaded, etc.
        Translates the brightness and contrast values into appropriate image levels.
        """
        brightness = 100-brightness
        colRange = colourMaps.getColourRange(self.minsg, self.maxsg, brightness, contrast, False)
        for btn in self.picbuttons:
            btn.stopPlayback()
            btn.setImage(colRange)
            btn.update()

    def volSliderMoved(self, value):
        # try/pass to avoid race situations when smth is not initialized
        try:
            for btn in self.picbuttons:
                btn.media_obj.applyVolSlider(value)
        except Exception:
            pass
