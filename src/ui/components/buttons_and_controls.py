# coding=latin-1

# buttons_and_controls.py
# Interactive controls for the AviaNZ program


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

import numpy as np
import pyqtgraph as pg
import pyqtgraph.functions as fn
from PyQt6.QtWidgets import QAbstractButton, QPushButton, QSlider, QLabel, QHBoxLayout, QGridLayout, QWidget, QSizePolicy, QToolButton, QStyle, QApplication
from PyQt6.QtCore import Qt, QTimer, QMimeData, pyqtSignal, QEvent
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QDrag
from .audio_player import ControllableAudio


class PicButton(QAbstractButton):
    # Class for HumanClassify dialogs to put spectrograms on buttons
    # Also includes playback capability.
    def __init__(self, index, spec, data_source, audioFormat, duration, unbufStart, unbufStop, lut, guides=None, guidecol=None, loop=False, parent=None, cluster=False, scaleToButton=False):
        super(PicButton, self).__init__(parent)
        self.index = index
        self.mark = "green"
        self.spec = spec
        self.unbufStart = unbufStart
        self.unbufStop = unbufStop
        self.cluster = cluster
        self.scaleToButton = scaleToButton
        self.setMouseTracking(True)

        self.playButton = QToolButton(self)
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playButton.hide()

        self.mouseIn = False

        # what to set the range to if a resize happens. This is overwritten in setImage
        self.colRangForResize = [np.min(self.spec), np.max(self.spec)]

        # batmode frequency guides (in Y positions 0-1)
        self.guides = guides
        if guides is not None:
            self.guidelines = [0]*len(self.guides)
            self.guidecol = [QColor(*col) for col in guidecol]

        # Store reference to data source instead of copying data
        self.data_source = data_source
        
        # check if playback possible (e.g. batmode)
        if data_source is not None and hasattr(data_source, 'data') and data_source.data is not None and len(data_source.data)>0:
            self.noaudio = False
            self.playButton.clicked.connect(self.playImage)
        else:
            self.noaudio = True

        # setImage reads some properties from self, to allow easy update
        # when color map changes. Initialize with full colour scale,
        # then we expect to call setImage soon again to update.
        self.lut = lut
        self.setImage([np.min(self.spec), np.max(self.spec)])

        self.buttonClicked = False
        self.clicked.connect(self.changePic)
        # fixed size
        self.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Fixed)
        self.setMinimumSize(self.im1.size())

        # playback things
        self.media_obj = ControllableAudio(None,audioFormat=audioFormat,useBar=True)
        self.media_obj.loop = loop
        self.duration = duration * 1000  # in ms
        self.media_obj.NotifyTimer.timeout.connect(self.endListener)

    def setImage(self, colRange):
        # takes in a piece of spectrogram and produces a pair of images
        # colRange: list [colStart, colEnd]
        self.colRangeForResize = colRange

        im, alpha = fn.makeARGB(self.spec, lut=self.lut, levels=colRange)
        im1 = fn.makeQImage(im, alpha)
        if im1.size().width() == 0:
            print("ERROR: button not shown, likely bad spectrogram coordinates")
            return

        # hardcode all image sizes
        if self.cluster:
            self.im1 = im1.scaled(200, 150)
        else: # just scaling the x axis basically
            if self.scaleToButton:
                self.im1 = im1.scaled(self.size())
            else:
                if im1.size().height() > 200:
                    self.im1 = im1.scaledToHeight(200)
                else:
                    self.im1 = im1
                    
        self.im2 = self.im1.copy()

        # add frequency guides if this is a bat image
        if self.guides is not None:
            painter = QPainter(self.im2)
            painter.setPen(QPen(QColor(180, 180, 180), 1))
            y1 = 0
            y2 = self.im1.size().height()
            for guide in self.guides:
                painter.setPen(QPen(self.guidecol[self.guides.index(guide)], 1))
                pos = int(self.im1.size().height() - guide * self.im1.size().height())
                painter.drawLine(0, pos, self.im1.size().width(), pos)
            painter.end()

        # start with "?" mark
        self.setImage_cluster() if self.cluster else self.setImage_overview()

    def setImage_cluster(self):
        painter = QPainter(self.im2)
        painter.fillRect(0, 0, 40, 40, QColor(255, 255, 255, 200))
        painter.setPen(QPen(QColor(0, 0, 0), 4))
        painter.setFont(QFont("Helvetica", 20))
        painter.drawText(12, 30, "?")
        painter.end()

    def setImage_overview(self):
        painter = QPainter(self.im2)
        painter.fillRect(160, 0, 40, 25, QColor(255, 255, 255, 200))
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.setFont(QFont("Helvetica", 14))
        painter.drawText(175, 18, "?")
        painter.end()
        
    def resizeEvent(self, event):
        if self.scaleToButton:
            self.setImage(self.colRangForResize)
        else:
            self.setFixedSize(self.im1.size())

    def paintEvent(self, event):
        if type(event) is not bool:
            painter = QPainter(self)
            painter.drawImage(event.rect(), self.im2)

            # Add coloured frame to indicate the mark:
            # green = confirmed, red = delete, yellow = uncertain
            pen = QPen()
            if self.mark == "red":
                pen.setColor(QColor(255, 0, 0))
            elif self.mark == "yellow":
                pen.setColor(QColor(255, 255, 0))
            elif self.mark == "green":
                pen.setColor(QColor(0, 255, 0))
            else:
                pen.setColor(QColor(255, 255, 255))
            pen.setWidth(4)
            painter.setPen(pen)

            painter.drawRect(event.rect())

            # add playback button on top if hovering over the button
            if self.mouseIn:
                self.playButton.show()
                if not self.noaudio:
                    # Position the playback button
                    iconSize = min(self.width(), self.height()) // 4
                    self.playButton.resize(iconSize, iconSize)
                    self.playButton.move((self.width() - iconSize) // 2, (self.height() - iconSize) // 2)

    def enterEvent(self, QEvent):
        # to reset the icon if it didn't stop cleanly
        self.mouseIn = True
        if self.noaudio:
            return
        if not self.media_obj.isPlaying():
            self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playButton.show()

    def leaveEvent(self, QEvent):
        self.mouseIn = False
        if self.noaudio:
            return
        if not self.media_obj.isPlaying():
            self.playButton.hide()

    def mouseMoveEvent(self, ev):
        if ev.buttons() != Qt.MouseButton.LeftButton:
            return

        mimeData = QMimeData()

        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setPixmap(QPixmap("./img/Owl_thinking.png"))
        dropAction = drag.exec(Qt.DropAction.MoveAction)

    def playImage(self):
        if self.media_obj.isPlayingorPaused():
            self.stopPlayback()
        else:
            self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
            if self.data_source and hasattr(self.data_source, 'data') and self.data_source.data is not None:
                self.media_obj.loadArray(self.data_source.data)

    def endListener(self):
        timeel = self.media_obj.elapsedUSecs() // 1000
        if timeel > self.duration:
            self.stopPlayback()

    def stopPlayback(self):
        self.media_obj.pressedStop()
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def sizeHint(self):
        return self.im1.size()

    def minimumSizeHint(self):
        return self.im1.size()

    def changePic(self,ev):
        # cycle through CONFIRM / DELETE / RECHECK marks

        if self.cluster:
            if self.mark == "green":
                self.mark = "red"
            else:
                self.mark = "green"
        else:
            if self.mark == "green":
                self.mark = "red"
            elif self.mark == "red":
                self.mark = "yellow"
            else:
                self.mark = "green"
        self.paintEvent(ev)
        self.repaint()
        QApplication.processEvents()


class MainPushButton(QPushButton):
    """ QPushButton with a standard styling """
    def __init__(self, *args, **kwargs):
        super(MainPushButton, self).__init__(*args, **kwargs)
        self.setStyleSheet("""
          MainPushButton { font-weight: bold; font-size: 14px; padding: 3px 3px 3px 7px; }
        """)
        self.setFixedHeight(45)


class BrightContrVol(QWidget):
    """ Widget containing brightness, contrast, volume control sliders
        and icons. On bright./contr. change, emits a colChanged signal
        with (brightness, contrast) values. On vol. change, emits a volChanged
        signal with (volume) value.
        All values are ints on 0-100 scale.
    """
    # Initialize with values to accurately set up slider positions
    # horizontal: bool, True for e.g. review modes, False for manual
    #  (adjusts layout accordingly)
    colChanged = pyqtSignal(int, int)
    volChanged = pyqtSignal(int)
    def __init__(self, brightness, contrast, inverted, horizontal=True, parent=None, **kwargs):
        super(BrightContrVol, self).__init__(parent, **kwargs)

        # Sliders and signals
        self.brightSlider = CustomSlider(Qt.Orientation.Horizontal)
        self.brightSlider.setMinimum(0)
        self.brightSlider.setMaximum(100)
        if inverted:
            self.brightSlider.setValue(100-brightness)
        else:
            self.brightSlider.setValue(brightness)
        self.brightSlider.setTickInterval(1)
        self.brightSlider.sliderClicked.connect(self.emitCol)
        self.brightSlider.sliderReleased.connect(self.emitCol)

        self.contrSlider = CustomSlider(Qt.Orientation.Horizontal)
        self.contrSlider.setMinimum(0)
        self.contrSlider.setMaximum(100)
        self.contrSlider.setValue(contrast)
        self.contrSlider.setTickInterval(1)
        self.contrSlider.sliderClicked.connect(self.emitCol)
        self.contrSlider.sliderReleased.connect(self.emitCol)

        # Volume control
        self.volSlider = CustomSlider(Qt.Orientation.Horizontal)
        self.volSlider.setRange(0,100)
        self.volSlider.setValue(50)
        self.volSlider.sliderClicked.connect(self.emitVol)
        self.volSlider.sliderReleased.connect(self.emitVol)

        # static labels
        labelBr = QLabel()
        labelBr.setPixmap(QPixmap('src/resources/images/brightstr24.png').scaled(18, 18, transformMode=Qt.TransformationMode.SmoothTransformation))

        labelCo = QLabel()
        labelCo.setPixmap(QPixmap('src/resources/images/contrstr24.png').scaled(18, 18, transformMode=Qt.TransformationMode.SmoothTransformation))

        self.volIcon = QLabel()
        self.volIcon.setPixmap(QPixmap('src/resources/images/volume.png').scaled(18, 18, transformMode=Qt.TransformationMode.SmoothTransformation))
        
        # Layout
        if horizontal:
            box = QHBoxLayout()
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(labelBr)
            box.addWidget(self.brightSlider)
            box.addWidget(labelCo)
            box.addWidget(self.contrSlider)
            # for vol icon: will be overwritten by signal connections
            box.addWidget(self.volIcon)
            box.addWidget(self.volSlider)
            self.volSlider.valueChanged.connect(
                lambda: self.volIcon.setPixmap(
                    QPixmap('src/resources/images/volume-mute.png').scaled(18, 18, transformMode=Qt.TransformationMode.SmoothTransformation)
                    if self.volSlider.value() == 0
                    else QPixmap('src/resources/images/volume.png').scaled(18, 18, transformMode=Qt.TransformationMode.SmoothTransformation)
                )
            )
        else:
            box = QGridLayout()
            box.setContentsMargins(5, 5, 5, 5)
            box.addWidget(labelBr, 0, 0)
            box.addWidget(self.brightSlider, 0, 1)
            box.addWidget(labelCo, 1, 0)
            box.addWidget(self.contrSlider, 1, 1)
            box.addWidget(self.volIcon, 2, 0)
            box.addWidget(self.volSlider, 2, 1)
            # for vol icon: will be overwritten by signal connections
            self.volSlider.valueChanged.connect(
                lambda: self.volIcon.setPixmap(
                    QPixmap('src/resources/images/volume-mute.png').scaled(18, 18, transformMode=Qt.TransformationMode.SmoothTransformation)
                    if self.volSlider.value() == 0
                    else QPixmap('src/resources/images/volume.png').scaled(18, 18, transformMode=Qt.TransformationMode.SmoothTransformation)
                )
            )

        self.setLayout(box)

    def emitCol(self):
        """ Emit the colour signal (to be triggered by valueChanged or
            programmatically, when a colour refresh is needed)
        """
        self.colChanged.emit(self.brightSlider.value(), self.contrSlider.value())
    
    def emitVol(self):
        self.volChanged.emit(self.volSlider.value())

    def emitAll(self):
        """ Emit both colour and volume signals (useful for initialization)
        """
        self.emitCol()
        self.emitVol()


class CustomSlider(QSlider):
    sliderClicked = pyqtSignal()

    def __init__(self,*args):
        super().__init__(*args)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.sliderClicked.emit()
        super().mouseReleaseEvent(event)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Jump the slider to the mouse position
            value = self.minimum() + (self.maximum() - self.minimum()) * event.position().x() / self.width()
            self.setValue(int(value))
            self.sliderClicked.emit()
        super().mousePressEvent(event)