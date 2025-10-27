
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

# Interactive controls for the AviaNZ program

import numpy as np
import pyqtgraph as pg
import pyqtgraph.functions as fn
from PyQt6.QtWidgets import QAbstractButton, QPushButton, QSlider, QLabel, QHBoxLayout, QGridLayout, QWidget, QSizePolicy, QToolButton, QStyle, QApplication
from PyQt6.QtCore import Qt, QTimer, QMimeData, pyqtSignal, QEvent
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QDrag
from src.ui.components.audio_player import ControllableAudio


class PicButton(QAbstractButton):
    # Class for HumanClassify dialogs to put spectrograms on buttons
    # Also includes playback capability.
    def __init__(self, index, spec, data_source, audioFormat, duration, unbufStart, unbufStop, lut, guides=None, guidecol=None, loop=False, parent=None, cluster=False, scaleToButton=False):
        super(PicButton, self).__init__(parent)
        self.index = index
        # In cluster mode, start with no mark; in review mode, start with "green" (uncertain/to review)
        self.mark = "none" if cluster else "green"
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

        # Store reference to data source (Spectrogram object) instead of copying data
        self.data_source = data_source
        
        # check if playback possible (e.g. batmode - check if audio_data has actual samples)
        if data_source is not None and hasattr(data_source, 'audio_data'):
            audio_data = data_source.audio_data
            if audio_data and hasattr(audio_data, 'data') and audio_data.data is not None and len(audio_data.data)>0:
                self.noaudio = False
                self.playButton.clicked.connect(self.playImage)
            else:
                self.noaudio = True
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

        # Note: don't paint "?" here - it will be painted in paintEvent if needed
        
    def resizeEvent(self, event):
        if self.scaleToButton:
            self.setImage(self.colRangForResize)
        else:
            self.setFixedSize(self.im1.size())

    def paintEvent(self, event):
        if type(event) is not bool:
            painter = QPainter(self)
            painter.drawImage(event.rect(), self.im2)

            # Add markers based on mark state
            if self.cluster:
                # Cluster mode: selected items get a blue overlay
                if self.mark == "selected":
                    painter.fillRect(0, 0, self.width(), self.height(), QColor(0, 150, 255, 80))
            else:
                # Overview/quick mode: show large symbols covering the image
                if self.mark == "green":
                    # Confirmed/reviewed - show nothing (blank/tick state)
                    pass
                elif self.mark == "red":
                    # Marked for deletion - show large "X" covering image
                    painter.fillRect(0, 0, self.width(), self.height(), QColor(255, 255, 255, 180))
                    painter.setPen(QPen(QColor(255, 0, 0), 6))
                    painter.setFont(QFont("Helvetica", 80, QFont.Weight.Bold))
                    # Center the X
                    fm = painter.fontMetrics()
                    x = (self.width() - fm.horizontalAdvance("X")) // 2
                    y = (self.height() + fm.ascent() - fm.descent()) // 2
                    painter.drawText(x, y, "X")
                elif self.mark == "yellow":
                    # Questioned/uncertain - show large "?" covering image
                    painter.fillRect(0, 0, self.width(), self.height(), QColor(255, 255, 255, 180))
                    painter.setPen(QPen(QColor(200, 150, 0), 6))
                    painter.setFont(QFont("Helvetica", 80, QFont.Weight.Bold))
                    # Center the ?
                    fm = painter.fontMetrics()
                    x = (self.width() - fm.horizontalAdvance("?")) // 2
                    y = (self.height() + fm.ascent() - fm.descent()) // 2
                    painter.drawText(x, y, "?")

            # Add coloured frame to indicate the mark:
            pen = QPen()
            if self.cluster:
                # Cluster mode: selected = bright color, not selected = white
                if self.mark == "selected":
                    pen.setColor(QColor(0, 200, 255))  # Bright blue for selected
                else:
                    pen.setColor(QColor(200, 200, 200))  # Light gray for unselected
            else:
                # Review mode: green = confirmed, red = delete, yellow = uncertain
                if self.mark == "red":
                    pen.setColor(QColor(255, 0, 0))
                elif self.mark == "yellow":
                    pen.setColor(QColor(255, 255, 0))
                elif self.mark == "green":
                    pen.setColor(QColor(0, 255, 0))
                else:
                    pen.setColor(QColor(255, 255, 255))
            # Scale border width based on button size (minimum 3, maximum 8)
            borderWidth = max(3, min(8, min(self.width(), self.height()) // 50))
            pen.setWidth(borderWidth)
            painter.setPen(pen)

            painter.drawRect(event.rect())

    def enterEvent(self, QEvent):
        # to reset the icon if it didn't stop cleanly
        self.mouseIn = True
        if self.noaudio:
            return
        if not self.media_obj.isPlaying():
            self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        
        # Position and size the playback button in top-left corner
        # Scale between 30-60 pixels based on button size
        iconSize = max(30, min(60, min(self.width(), self.height()) // 4))
        self.playButton.resize(iconSize, iconSize)
        self.playButton.move(0, 0)  # Exactly in top-left corner
        self.playButton.show()
        self.update()  # Force repaint to show play button

    def leaveEvent(self, QEvent):
        self.mouseIn = False
        if self.noaudio:
            return
        if not self.media_obj.isPlaying():
            self.playButton.hide()
        self.update()  # Force repaint to hide play button

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
            # data_source is a Spectrogram, so access audio_data.data
            if self.data_source and hasattr(self.data_source, 'audio_data'):
                audio_data = self.data_source.audio_data
                if audio_data and hasattr(audio_data, 'data') and audio_data.data is not None and len(audio_data.data) > 0:
                    # loadArray() loads and starts playback automatically
                    self.media_obj.loadArray(audio_data.data)
                else:
                    print("DEBUG: No audio data to play")
            else:
                print("DEBUG: data_source doesn't have audio_data attribute")

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
            # In cluster mode, toggle between selected and not selected
            if self.mark == "selected":
                self.mark = "none"
            else:
                self.mark = "selected"
        else:
            if self.mark == "green":
                self.mark = "red"
            elif self.mark == "red":
                self.mark = "yellow"
            else:
                self.mark = "green"
        # Don't call paintEvent directly, just trigger a repaint
        self.update()
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