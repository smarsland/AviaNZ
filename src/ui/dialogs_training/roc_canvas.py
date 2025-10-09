
# This is part of the AviaNZ interface
# Holds most of the code for the various dialog boxes

# Version 3.4 18/12/24
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

# Dialogs used for filter training / testing.
# These are relatively complicated wizards which also do file I/O

import os
import time
import platform
import copy
from shutil import copyfile
import json

from PyQt6.QtGui import QIcon, QValidator, QPixmap, QColor
from PyQt6.QtCore import QDir, Qt, QEvent, QSize, pyqtSignal
from PyQt6.QtWidgets import QLabel, QSlider, QPushButton, QListWidget, QListWidgetItem, QComboBox, QDialog, QWizard, QWizardPage, QLineEdit, QSizePolicy, QFormLayout, QVBoxLayout, QHBoxLayout, QCheckBox, QLayout, QApplication, QRadioButton, QGridLayout, QFileDialog, QScrollArea, QWidget, QAbstractItemView

import matplotlib.markers as mks
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg

import numpy as np
from src.ui.colourMaps import colourMaps
from src.ui.components.buttons_and_controls import BrightContrVol, CustomSlider, PicButton
from src.ui.components.popups import MessagePopup
from src.ui.components.file_list import LightedFileList
from src.ui.components.layout_widgets import Layout
from src.core import SupportClasses
from src.core import Spectrogram
from src.core import WaveletSegment
from src.core import WaveletFunctions
from src.core import Segment
from src.core import Clustering
from src.core import Training

from src.models import NNModels

import math

class ROCCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=6, dpi=100):
        # reduce size on low-res monitors
        aspectratio = width/height
        height = min(height, 0.6*(QApplication.primaryScreen().availableSize().height()-150)/dpi)
        width = height*aspectratio
        # print("resetting to", height, width)
        plt.style.use('ggplot')
        self.parent = parent

        self.lines = None
        self.plotLines = []

        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.set_tight_layout(True)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

    def plotme(self):
        # valid_markers = ([item[0] for item in mks.MarkerStyle.markers.items() if
        #                   item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])
        # markers = np.random.choice(valid_markers, 5, replace=False)

        self.ax = self.figure.subplots()
        for i in range(5):
            self.lines, = self.ax.plot([], [], marker='o')
            self.plotLines.append(self.lines)
        self.ax.set_title('ROC curve')
        self.ax.set_xlabel('False Positive Rate (FPR)')
        self.ax.set_ylabel('True Positive Rate (TPR)')
        # fig.canvas.set_window_title('ROC Curve')
        self.ax.set_ybound(0, 1)
        self.ax.set_xbound(0, 1)
        self.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
        self.ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
        # ax.legend()

    def plotmeagain(self, TPR, FPR, NN=False):
        # Update data (with the new _and_ the old points)
        if NN:
            self.lines, = self.ax.plot(FPR, TPR, marker='o')
            self.plotLines.append(self.lines)
        else:
            for i in range(np.shape(TPR)[0]):
                self.plotLines[i].set_xdata(FPR[i])
                self.plotLines[i].set_ydata(TPR[i])

        # Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def Clear(self):
        if hasattr(self, 'ax'):
            self.figure.clf()
            self.ax.clear()
            # self.draw()
