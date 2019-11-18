
# SupportClasses.py
#
# Support classes for the AviaNZ program
# Mostly subclassed from pyqtgraph

# Version 2.0 18/11/19
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2019

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
#     from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QMessageBox, QAbstractButton, QWidget
from PyQt5.QtCore import Qt, QTime, QIODevice, QBuffer, QByteArray, QMimeData, QEvent, QLineF
from PyQt5.QtMultimedia import QAudio, QAudioOutput, QAudioFormat
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen, QColor, QFont, QDrag

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.functions as fn

import wavio
from time import sleep
import time
import math
import numpy as np
import os, json
import sys
import io

class TimeAxisHour(pg.AxisItem):
    # Time axis (at bottom of spectrogram)
    # Writes the time as hh:mm:ss, and can add an offset
    def __init__(self, *args, **kwargs):
        super(TimeAxisHour, self).__init__(*args, **kwargs)
        self.offset = 0
        self.setLabel('Time', units='hh:mm:ss')

    def tickStrings(self, values, scale, spacing):
        # Overwrite the axis tick code
        return [QTime(0,0,0).addSecs(value+self.offset).toString('hh:mm:ss') for value in values]

    def setOffset(self,offset):
        self.offset = offset
        #self.update()


class TimeAxisMin(pg.AxisItem):
    # Time axis (at bottom of spectrogram)
    # Writes the time as mm:ss, and can add an offset
    def __init__(self, *args, **kwargs):
        super(TimeAxisMin, self).__init__(*args, **kwargs)
        self.offset = 0
        self.setLabel('Time', units='mm:ss')

    def tickStrings(self, values, scale, spacing):
        # Overwrite the axis tick code
        return [QTime(0,0,0).addSecs(value+self.offset).toString('mm:ss') for value in values]

    def setOffset(self,offset):
        self.offset = offset
        self.update()


class ShadedROI(pg.ROI):
    # A region of interest that is shaded, for marking segments
    def paint(self, p, opt, widget):
        #brush = QtGui.QBrush(QtGui.QColor(0, 0, 255, 50))
        if not hasattr(self, 'currentBrush'):
            self.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 50)))
        if not hasattr(self, 'currentPen'):
            self.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0, 255)))
        p.save()
        r = self.boundingRect()
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(self.currentPen)
        p.setBrush(self.currentBrush)

        p.translate(r.left(), r.top())
        p.scale(r.width(), r.height())
        p.drawRect(0, 0, 1, 1)
        p.restore()

    def setBrush(self, *br, **kargs):
        """Set the brush that fills the region. Can have any arguments that are valid
        for :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        self.brush = fn.mkBrush(*br, **kargs)
        self.currentBrush = self.brush

    # this allows compatibility with LinearRegions:
    def setHoverBrush(self, *br, **kargs):
        self.hoverBrush = fn.mkBrush(*br, **kargs)

    def setPen(self, *br, **kargs):
        self.pen = fn.mkPen(*br, **kargs)
        self.currentPen = self.pen

    def hoverEvent(self, ev):
        if self.transparent:
            return
        if not ev.isExit():
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)

    def setMouseHover(self, hover):
        # for ignoring when ReadOnly enabled:
        if not self.translatable:
            return
        # don't waste time if state isn't changing:
        if self.mouseHovering == hover:
            return
        self.mouseHovering = hover
        if hover:
            self.currentBrush = self.hoverBrush
        else:
            self.currentBrush = self.brush
        self.update()


def mouseDragEventFlexible(self, ev):
    if ev.button() == self.rois[0].parent.MouseDrawingButton:
        return
    ev.accept()
    
    ## Inform ROIs that a drag is happening 
    ##  note: the ROI is informed that the handle has moved using ROI.movePoint
    ##  this is for other (more nefarious) purposes.
    #for r in self.roi:
        #r[0].pointDragEvent(r[1], ev)
        
    if ev.isFinish():
        if self.isMoving:
            for r in self.rois:
                r.stateChangeFinished()
        self.isMoving = False
    elif ev.isStart():
        for r in self.rois:
            r.handleMoveStarted()
        self.isMoving = True
        self.startPos = self.scenePos()
        self.cursorOffset = self.scenePos() - ev.buttonDownScenePos()
        
    if self.isMoving:  ## note: isMoving may become False in mid-drag due to right-click.
        pos = ev.scenePos() + self.cursorOffset
        self.movePoint(pos, ev.modifiers(), finish=False)


def mouseDragEventFlexibleLine(self, ev):
    if self.movable and ev.button() != self.btn:
        if ev.isStart():
            self.moving = True
            self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
            self.startPosition = self.pos()
        ev.accept()

        if not self.moving:
            return

        self.setPos(self.cursorOffset + self.mapToParent(ev.pos()))
        self.sigDragged.emit(self)
        if ev.isFinish():
            self.moving = False
            self.sigPositionChangeFinished.emit(self)


class ShadedRectROI(ShadedROI):
    # A rectangular ROI that it shaded, for marking segments
    def __init__(self, pos, size, centered=False, movable=True, sideScalers=True, parent=None, **args):
        #QtGui.QGraphicsRectItem.__init__(self, 0, 0, size[0], size[1])
        pg.ROI.__init__(self, pos, size, movable=movable, **args)
        self.parent = parent
        self.mouseHovering = False
        self.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 50)))
        self.setHoverBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 100)))
        self.transparent = True

        #self.addTranslateHandle(center)
        if self.translatable:
            self.addScaleHandle([1, 1], [0, 0]) # top right
            self.addScaleHandle([1, 0], [0, 1]) # bottom right
            self.addScaleHandle([0, 1], [1, 0]) # top left
            self.addScaleHandle([0, 0], [1, 1]) # bottom left

    def setMovable(self,value):
        self.resizable = value
        self.translatable = value

    def mouseDragEvent(self, ev):
        if ev.isStart():
            if ev.button() != self.parent.MouseDrawingButton:
                self.setSelected(True)
                if self.translatable:
                    self.isMoving = True
                    self.preMoveState = self.getState()
                    self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                    self.sigRegionChangeStarted.emit(self)
                    ev.accept()
                else:
                    ev.ignore()
        elif ev.isFinish():
            if self.translatable:
                if self.isMoving:
                    self.stateChangeFinished()
                self.isMoving = False
            return

        if self.translatable and self.isMoving and ev.buttons() != self.parent.MouseDrawingButton:
            snap = True if (ev.modifiers() & QtCore.Qt.ControlModifier) else None
            newPos = self.mapToParent(ev.pos()) + self.cursorOffset
            self.translate(newPos - self.pos(), snap=snap, finish=False)

pg.graphicsItems.ROI.Handle.mouseDragEvent = mouseDragEventFlexible
pg.graphicsItems.InfiniteLine.InfiniteLine.mouseDragEvent = mouseDragEventFlexibleLine


class LinearRegionItem2(pg.LinearRegionItem):
    def __init__(self, parent, bounds=None, *args, **kwds):
        pg.LinearRegionItem.__init__(self, bounds, *args, **kwds)
        self.parent = parent
        self.bounds = bounds
        self.lines[0].btn = self.parent.MouseDrawingButton
        self.lines[1].btn = self.parent.MouseDrawingButton

    def mouseDragEvent(self, ev):
        if not self.movable or ev.button()==self.parent.MouseDrawingButton:
            return
        ev.accept()

        if ev.isStart():
            bdp = ev.buttonDownPos()
            self.cursorOffsets = [l.pos() - bdp for l in self.lines]
            self.startPositions = [l.pos() for l in self.lines]
            self.moving = True

        if not self.moving:
            return

        self.lines[0].blockSignals(True)  # only want to update once
        newcenter = ev.pos()
        # added this to bound its dragging, as in ROI.
        # first, adjust center position to avoid dragging too far:
        for i, l in enumerate(self.lines):
            tomove = self.cursorOffsets[i] + newcenter
            if self.bounds is not None:
                # stop center from moving too far left
                if tomove.x() < self.bounds[0]:
                    newcenter.setX(-self.cursorOffsets[i].x() + self.bounds[0])
                # stop center from moving too far right
                if tomove.x() > self.bounds[1]:
                    newcenter.setX(-self.cursorOffsets[i].x() + self.bounds[1])

        # update lines based on adjusted center
        for i, l in enumerate(self.lines):
            tomove = self.cursorOffsets[i] + newcenter
            l.setPos(tomove)

        self.lines[0].blockSignals(False)
        self.prepareGeometryChange()

        if ev.isFinish():
            self.moving = False
            self.sigRegionChangeFinished.emit(self)
        else:
            self.sigRegionChanged.emit(self)


class DragViewBox(pg.ViewBox):
    # A normal ViewBox, but with the ability to capture drag.
    # Effectively, if "dragging" is enabled, it captures press & release signals.
    # Otherwise it ignores the event, which then goes to the scene(),
    # which only captures click events.
    sigMouseDragged = QtCore.Signal(object,object,object)
    keyPressed = QtCore.Signal(int)

    def __init__(self, parent, enableDrag, thisIsAmpl, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.enableDrag = enableDrag
        self.parent = parent
        self.thisIsAmpl = thisIsAmpl

    def mouseDragEvent(self, ev):
        print("Uncaptured drag event")
        # if self.enableDrag:
        #     ## if axis is specified, event will only affect that axis.
        #     ev.accept()
        #     if self.state['mouseMode'] != pg.ViewBox.RectMode or ev.button() == QtCore.Qt.RightButton:
        #         ev.ignore()

        #     if ev.isFinish():  ## This is the final move in the drag; draw the actual box
        #         print("dragging done")
        #         self.rbScaleBox.hide()
        #         self.sigMouseDragged.emit(ev.buttonDownScenePos(ev.button()),ev.scenePos(),ev.screenPos())
        #     else:
        #         ## update shape of scale box
        #         self.updateScaleBox(ev.buttonDownPos(), ev.pos())
        # else:
        #     pass

    def mousePressEvent(self, ev):
        if self.enableDrag and ev.button() == self.parent.MouseDrawingButton:
            if self.thisIsAmpl:
                self.parent.mouseClicked_ampl(ev)
            else:
                self.parent.mouseClicked_spec(ev)
            ev.accept()
        else:
            ev.ignore()

    def mouseReleaseEvent(self, ev):
        if self.enableDrag and ev.button() == self.parent.MouseDrawingButton:
            if self.thisIsAmpl:
                self.parent.mouseClicked_ampl(ev)
            else:
                self.parent.mouseClicked_spec(ev)
            ev.accept()
        else:
            ev.ignore()

    def keyPressEvent(self,ev):
        # This catches the keypresses and sends out a signal
        #self.emit(SIGNAL("keyPressed"),ev)
        super(DragViewBox, self).keyPressEvent(ev)
        self.keyPressed.emit(ev.key())


class ChildInfoViewBox(pg.ViewBox):
    # Normal ViewBox, but with ability to pass a message back from a child
    sigChildMessage = QtCore.Signal(object)

    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)

    def resend(self,x):
        self.sigChildMessage.emit(x)


class ClickableRectItem(QtGui.QGraphicsRectItem):
    # QGraphicsItem doesn't include signals, hence this mess
    def __init__(self, *args, **kwds):
        QtGui.QGraphicsRectItem.__init__(self, *args, **kwds)

    def mousePressEvent(self, ev):
        super(ClickableRectItem, self).mousePressEvent(ev)
        self.parentWidget().resend(self.mapRectToParent(self.boundingRect()).x())


class ControllableAudio(QAudioOutput):
    # This links all the PyQt5 audio playback things -
    # QAudioOutput, QFile, and input from main interfaces

    def __init__(self, format):
        super(ControllableAudio, self).__init__(format)
        # on this notify, move slider (connected in main file)
        self.setNotifyInterval(30)
        self.stateChanged.connect(self.endListener)
        self.tempin = QBuffer()
        self.startpos = 0
        self.timeoffset = 0
        self.keepSlider = False
        #self.format = format
        # set small buffer (10 ms) and use processed time
        self.setBufferSize(int(self.format().sampleSize() * self.format().sampleRate()/100 * self.format().channelCount()))

    def isPlaying(self):
        return(self.state() == QAudio.ActiveState)

    def endListener(self):
        # this should only be called if there's some misalignment between GUI and Audio
        if self.state() == QAudio.IdleState:
            # give some time for GUI to catch up and stop
            sleepCycles = 0
            while(self.state() != QAudio.StoppedState and sleepCycles < 30):
                sleep(0.03)
                sleepCycles += 1
                # This loop stops when timeoffset+processedtime > designated stop position.
                # By adding this offset, we ensure the loop stops even if
                # processed audio timer breaks somehow.
                self.timeoffset += 30
                self.notify.emit()
            self.pressedStop()

    def pressedPlay(self, resetPause=False, start=0, stop=0, audiodata=None):
        if not resetPause and self.state() == QAudio.SuspendedState:
            print("Resuming at: %d" % self.pauseoffset)
            self.sttime = time.time() - self.pauseoffset/1000
            self.resume()
        else:
            if not self.keepSlider or resetPause:
                self.pressedStop()

            print("Starting at: %d" % self.tempin.pos())
            sleep(0.2)
            # in case bar was moved under pause, we need this:
            pos = self.tempin.pos() # bytes
            pos = self.format().durationForBytes(pos) / 1000 # convert to ms
            pos = pos + start
            print("Pos: %d start: %d stop %d" %(pos, start, stop))
            self.filterSeg(pos, stop, audiodata)

    def pressedPause(self):
        self.keepSlider=True # a flag to avoid jumping the slider back to 0
        pos = self.tempin.pos() # bytes
        pos = self.format().durationForBytes(pos) / 1000 # convert to ms
        # store offset, relative to the start of played segment
        self.pauseoffset = pos + self.timeoffset
        self.suspend()

    def pressedStop(self):
        # stop and reset to window/segment start
        self.keepSlider=False
        self.stop()
        if self.tempin.isOpen():
            self.tempin.close()

    def filterBand(self, start, stop, low, high, audiodata, sp):
        # takes start-end in ms, relative to file start
        self.timeoffset = max(0, start)
        start = max(0, start * self.format().sampleRate() // 1000)
        stop = min(stop * self.format().sampleRate() // 1000, len(audiodata))
        segment = audiodata[int(start):int(stop)]
        segment = sp.bandpassFilter(segment,sampleRate=None, start=low, end=high)
        # segment = self.sp.ButterworthBandpass(segment, self.sampleRate, bottom, top,order=5)
        self.loadArray(segment)

    def filterSeg(self, start, stop, audiodata):
        # takes start-end in ms
        self.timeoffset = max(0, start)
        start = max(0, int(start * self.format().sampleRate() // 1000))
        stop = min(int(stop * self.format().sampleRate() // 1000), len(audiodata))
        segment = audiodata[start:stop]
        self.loadArray(segment)

    def loadArray(self, audiodata):
        # loads an array from memory into an audio buffer
        if self.format().sampleSize() == 16:
            audiodata = audiodata.astype('int16')  # 16 corresponds to sampwidth=2
        elif self.format().sampleSize() == 32:
            audiodata = audiodata.astype('int32')
        elif self.format().sampleSize() == 24:
            audiodata = audiodata.astype('int32')
            print("Warning: 24-bit sample playback currently not supported")
        elif self.format().sampleSize() == 8:
            audiodata = audiodata.astype('uint8')
        else:
            print("ERROR: sampleSize %d not supported" % self.format().sampleSize())
            return
        # double mono sound to get two channels - simplifies reading
        if self.format().channelCount()==2:
            audiodata = np.column_stack((audiodata, audiodata))

        # write filtered output to a BytesIO buffer
        self.tempout = io.BytesIO()
        # NOTE: scale=None rescales using data minimum/max. This can cause clipping. Use scale="none" if this causes weird playback sound issues.
        # in particular for 8bit samples, we need more scaling:
        if self.format().sampleSize() == 8:
            scale = (audiodata.min()/2, audiodata.max()*2)
        else:
            scale = None
        wavio.write(self.tempout, audiodata, self.format().sampleRate(), scale=scale, sampwidth=self.format().sampleSize() // 8)

        # copy BytesIO@write to QBuffer@read for playing
        self.temparr = QByteArray(self.tempout.getvalue()[44:])
        # self.tempout.close()
        if self.tempin.isOpen():
            self.tempin.close()
        self.tempin.setBuffer(self.temparr)
        self.tempin.open(QIODevice.ReadOnly)

        # actual timer is launched here, with time offset set asynchronously
        sleep(0.2)
        self.sttime = time.time() - self.timeoffset/1000
        self.start(self.tempin)

    def seekToMs(self, ms, start):
        print("Seeking to %d ms" % ms)
        # start is an offset for the current view start, as it is position 0 in extracted file
        self.reset()
        self.tempin.seek(self.format().bytesForDuration((ms-start)*1000))
        self.timeoffset = ms

    def applyVolSlider(self, value):
        # passes UI volume nonlinearly
        # value = QAudio.convertVolume(value / 100, QAudio.LogarithmicVolumeScale, QAudio.LinearVolumeScale)
        value = (math.exp(value/50)-1)/(math.exp(2)-1)
        self.setVolume(value)


class FlowLayout(QtGui.QLayout):
    # This is the flow layout which lays out a set of spectrogram pictures on buttons (for HumanClassify2) as
    # nicely as possible
    # From https://gist.github.com/Cysu/7461066
    def __init__(self, parent=None, margin=0, spacing=-1):
        super(FlowLayout, self).__init__(parent)

        if parent is not None:
            self.setMargin(margin)

        self.setSpacing(spacing)

        self.itemList = []

        self.margin = margin

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList[index]

        return None

    def takeAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList.pop(index)

        return None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self._doLayout(QtCore.QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self._doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    # def minimumSize(self):
    #     size = QtCore.QSize()
    #
    #     for item in self.itemList:
    #         size = size.expandedTo(item.minimumSize())
    #
    #     size += QtCore.QSize(2 * self.margin(), 2 * self.margin())
    #     return size

    def _doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.spacing() + wid.style().layoutSpacing(
                QtGui.QSizePolicy.PushButton,
                QtGui.QSizePolicy.PushButton,
                QtCore.Qt.Horizontal)

            spaceY = self.spacing() + wid.style().layoutSpacing(
                QtGui.QSizePolicy.PushButton,
                QtGui.QSizePolicy.PushButton,
                QtCore.Qt.Vertical)

            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(
                    QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()


class Log(object):
    """ Used for logging info during batch processing.
        Stores most recent analysis for each species, to stay in sync w/ data files.
        Arguments:
        1. path to log file
        2. species
        3. list of other settings of the current analysis

        LOG FORMAT, for each analysis:
        #freetext line
        species
        settings line
        files, multiple lines
    """

    def __init__(self, path, species, settings):
        # in order to append, the previous log must:
        # 1. exist
        # 2. be writeable
        # 3. match current analysis
        # On init, we parse the existing log to see if appending is possible.
        # Actual append/create happens later.
        self.possibleAppend = False
        self.file = path
        self.species = species
        self.settings = ','.join(map(str, settings))
        self.oldAnalyses = []
        self.filesDone = []
        self.currentHeader = ""
        allans = []

        # now, check if the specified log can be resumed:
        if os.path.isfile(path):
            try:
                f = open(path, 'r+')
                print("Found log file at %s" % path)

                lines = [line.rstrip('\n') for line in f]
                f.close()
                lstart = 0
                lend = 1
                # parse to separate each analysis into
                # [freetext, species, settings, [files]]
                # (basically I'm parsing txt into json because I'm dumb)
                while lend<len(lines):
                    #print(lines[lend])
                    if lines[lend][0] == "#":
                        allans.append([lines[lstart], lines[lstart+1], lines[lstart+2],
                                        lines[lstart+3 : lend]])
                        lstart = lend
                    lend += 1
                allans.append([lines[lstart], lines[lstart+1], lines[lstart+2],
                                lines[lstart+3 : lend]])

                # parse the log thusly:
                # if current species analysis found, store parameters
                # and compare to check if it can be resumed.
                # store all other analyses for re-printing.
                for a in allans:
                    #print(a)
                    if a[1]==self.species:
                        print("Resumable analysis found")
                        # do not reprint this in log
                        if a[2]==self.settings:
                            self.currentHeader = a[0]
                            # (a1 and a2 match species & settings anyway)
                            self.filesDone = a[3]
                            self.possibleAppend = True
                    else:
                        # store this for re-printing to log
                        self.oldAnalyses.append(a)

            except IOError:
                # bad error: lacking permissions?
                print("ERROR: could not open log at %s" % path)

    def appendFile(self, filename):
        print('Appending %s to log' % filename)
        # attach file path to end of log
        self.file.write(filename)
        self.file.write("\n")
        self.file.flush()

    def appendHeader(self, header, species, settings):
        if header is None:
            header = "#Analysis started on " + time.strftime("%Y %m %d, %H:%M:%S") + ":"
        self.file.write(header)
        self.file.write("\n")
        self.file.write(species)
        self.file.write("\n")
        if type(settings) is list:
            settings = ','.join(settings)
        self.file.write(settings)
        self.file.write("\n")
        self.file.flush()

    def reprintOld(self):
        # push everything from oldAnalyses to log
        # To be called once starting a new log is confirmed
        for a in self.oldAnalyses:
            self.appendHeader(a[0], a[1], a[2])
            for f in a[3]:
                self.appendFile(f)


class MessagePopup(QMessageBox):
    """ Convenience wrapper around QMessageBox.
        TYPES, based on main icon:
        w - warning
        d - done (successful completion)
        t - thinking (questions)
        o - other
        a - about
    """
    def __init__(self, type, title, text):
        super(QMessageBox, self).__init__()

        self.setText(text)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        if (type=="w"):
            self.setIconPixmap(QPixmap("img/Owl_warning.png"))
        elif (type=="d"):
            self.setIcon(QMessageBox.Information)
            self.setIconPixmap(QPixmap("img/Owl_done.png"))
        elif (type=="t"):
            self.setIcon(QMessageBox.Information)
            self.setIconPixmap(QPixmap("img/Owl_thinking.png"))
        elif (type=="a"):
            # Easy way to set ABOUT text here:
            self.setIconPixmap(QPixmap("img/AviaNZ.png"))
            self.setText("The AviaNZ Program, v2.0 (November 2019)")
            self.setInformativeText("By Stephen Marsland, Victoria University of Wellington. With code by Nirosha Priyadarshani and Julius Juodakis, and input from Isabel Castro, Moira Pryde, Stuart Cockburn, Rebecca Stirnemann, Sumudu Purage, Virginia Listanti, and Rebecca Huistra. \n stephen.marsland@vuw.ac.nz")
        elif (type=="o"):
            self.setIconPixmap(QPixmap("img/AviaNZ.png"))

        self.setWindowIcon(QIcon("img/Avianz.ico"))

        # by default, adding OK button. Can easily be overwritten after creating
        self.setStandardButtons(QMessageBox.Ok)


class ConfigLoader(object):
    """ This deals with reading main config files.
        Not much functionality, but lots of exception handling,
        so moved it out separately.

        Most of these functions return the contents of a corresponding JSON file.
    """

    def config(self, file):
        # At this point, the main config file should already be ensured to exist.
        # It will always be in user configdir, otherwise it would be impossible to find.
        print("Loading software settings from file %s" % file)
        try:
            config = json.load(open(file))
            return config
        except ValueError as e:
            # if JSON looks corrupt, quit:
            print(e)
            msg = MessagePopup("w", "Bad config file", "ERROR: file " + file + " corrupt, delete it to restore default")
            msg.exec_()
            sys.exit()

    def filters(self, dir):
        """ Returns a dict of filter JSONs,
            named after the corresponding file names. """
        print("Loading call filters from folder %s" % dir)
        try:
            filters = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        except Exception:
            print("Folder %s not found, no filters loaded" % dir)
            return None

        goodfilters = dict()
        for filtfile in filters:
            try:
                filt = json.load(open(os.path.join(dir, filtfile)))

                # skip this filter if it looks fishy:
                if not isinstance(filt, dict) or "species" not in filt or "SampleRate" not in filt or "Filters" not in filt or len(filt["Filters"])<1:
                    raise ValueError("Filter JSON format wrong, skipping")
                for subfilt in filt["Filters"]:
                    if not isinstance(subfilt, dict) or "calltype" not in subfilt or "WaveletParams" not in subfilt or "TimeRange" not in subfilt:
                        raise ValueError("Subfilter JSON format wrong, skipping")
                    if "thr" not in subfilt["WaveletParams"] or "nodes" not in subfilt["WaveletParams"] or len(subfilt["TimeRange"])<4:
                        raise ValueError("Subfilter JSON format wrong (details), skipping")

                # if filter passed checks, store it,
                # using filename (without extension) as the key
                goodfilters[filtfile[:-4]] = filt
            except Exception as e:
                print("Could not load filter:", filtfile, e)
        print("Loaded filters:", list(goodfilters.keys()))
        return goodfilters

    def shortbl(self, file, configdir):
        # A fallback shortlist will be confirmed to exist in configdir.
        # This list is necessary
        print("Loading short species list from file %s" % file)
        try:
            if os.path.isabs(file):
                # user-picked files will have absolute paths
                shortblfile = file
            else:
                # initial file will have relative path,
                # to allow looking it up in various OSes.
                shortblfile = os.path.join(configdir, file)
            if not os.path.isfile(shortblfile):
                print("Warning: file %s not found, falling back to default" % shortblfile)
                shortblfile = os.path.join(configdir, "ListCommonBirds.txt")

            try:
                readlist = json.load(open(shortblfile))
                if len(readlist)>29:
                    print("Warning: short species list has %s entries, truncating to 30" % len(readlist))
                    readlist = readlist[:29]
                return readlist
            except ValueError as e:
                # if JSON looks corrupt, quit and suggest deleting:
                print(e)
                msg = MessagePopup("w", "Bad species list", "ERROR: file " + shortblfile + " corrupt, delete it to restore default. Reverting to default.")
                msg.exec_()
                return None

        except Exception as e:
            # if file is not found at all, quit, user must recreate the file or change path
            print(e)
            msg = MessagePopup("w", "Bad species list", "ERROR: Failed to load short species list from " + file + ". Reverting to default.")
            msg.exec_()
            return None

    def longbl(self, file, configdir):

        print("Loading long species list from file %s" % file)
        try:
            if os.path.isabs(file):
                # user-picked files will have absolute paths
                longblfile = file
            else:
                # initial file will have relative path,
                # to allow looking it up in various OSes.
                longblfile = os.path.join(configdir, file)
            if not os.path.isfile(longblfile):
                print("Warning: file %s not found, falling back to default" % longblfile)
                longblfile = os.path.join(configdir, "ListDOCBirds.txt")

            try:
                readlist = json.load(open(longblfile))
                return readlist
            except ValueError as e:
                print(e)
                msg = MessagePopup("w", "Bad species list", "Warning: file " + longblfile + " corrupt, delete it to restore default. Reverting to default.")
                msg.exec_()
                return None

        except Exception as e:
            print(e)
            msg = MessagePopup("w", "Bad species list", "Warning: Failed to load long species list from " + file + ". Reverting to default.")
            msg.exec_()
            return None

    # Dumps the provided JSON array to the corresponding bird file.
    def blwrite(self, content, file, configdir):
        print("Updating species list in file %s" % file)
        try:
            if os.path.isabs(file):
                # user-picked files will have absolute paths
                file = file
            else:
                # initial file will have relative path,
                # to allow looking it up in various OSes.
                file = os.path.join(configdir, file)

            # no fallback in case file not found - don't want to write to random places.
            json.dump(content, open(file, 'w'), indent=1)

        except Exception as e:
            print(e)
            msg = MessagePopup("w", "Unwriteable species list", "Warning: Failed to write species list to " + file)
            msg.exec_()

    # Dumps the provided JSON array to the corresponding config file.
    def configwrite(self, content, file):
        print("Saving config to file %s" % file)
        try:
            # will always be an absolute path to the user configdir.
            json.dump(content, open(file, 'w'), indent=1)

        except Exception as e:
            print("ERROR while saving config file:")
            print(e)

class PicButton(QAbstractButton):
    # Class for HumanClassify dialogs to put spectrograms on buttons
    # Also includes playback capability.
    def __init__(self, index, spec, audiodata, format, duration, unbufStart, unbufStop, lut, colStart, colEnd, cmapInv, parent=None, cluster=False):
        super(PicButton, self).__init__(parent)
        self.index = index
        self.mark = "green"
        self.spec = spec
        self.unbufStart = unbufStart
        self.unbufStop = unbufStop
        self.cluster = cluster
        self.setMouseTracking(True)
        # setImage reads some properties from self, to allow easy update
        # when color map changes
        self.setImage(lut, colStart, colEnd, cmapInv)

        self.buttonClicked = False
        # if not self.cluster:
        self.clicked.connect(self.changePic)
        # fixed size
        self.setSizePolicy(0,0)
        self.setMinimumSize(self.im1.size())

        # playback things
        self.media_obj = ControllableAudio(format)
        self.media_obj.notify.connect(self.endListener)
        self.audiodata = audiodata
        self.duration = duration * 1000 # in ms

        self.playButton = QtGui.QToolButton(self)
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.playImage)
        self.playButton.hide()

    def setImage(self, lut, colStart, colEnd, cmapInv):
        # takes in a piece of spectrogram and produces a pair of images
        if cmapInv:
            im, alpha = fn.makeARGB(self.spec, lut=lut, levels=[colEnd, colStart])
        else:
            im, alpha = fn.makeARGB(self.spec, lut=lut, levels=[colStart, colEnd])
        im1 = fn.makeQImage(im, alpha)
        if im1.size().width() == 0:
            print("ERROR: button not shown, likely bad spectrogram coordinates")
            return

        # hardcode all image sizes
        if self.cluster:
            self.im1 = im1.scaled(200, 150)
        else:
            self.specReductionFact = im1.size().width()/500
            self.im1 = im1.scaled(500, im1.size().height())

        # draw lines
        if not self.cluster:
            unbufStartAdj = self.unbufStart / self.specReductionFact
            unbufStopAdj = self.unbufStop / self.specReductionFact
            self.line1 = QLineF(unbufStartAdj, 0, unbufStartAdj, im1.size().height())
            self.line2 = QLineF(unbufStopAdj, 0, unbufStopAdj, im1.size().height())

    def paintEvent(self, event):
        if type(event) is not bool:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(80,255,80), 2))
            if self.cluster:
                if self.mark == "yellow":
                    painter.setOpacity(0.5)
            else:
                if self.mark == "yellow":
                    painter.setOpacity(0.8)
                elif self.mark == "red":
                    painter.setOpacity(0.5)
            painter.drawImage(event.rect(), self.im1)
            if not self.cluster:
                painter.drawLine(self.line1)
                painter.drawLine(self.line2)

            # draw decision mark
            fontsize = int(self.im1.size().height() * 0.65)
            if self.mark == "green":
                pass
            elif self.mark == "yellow" and not self.cluster:
                painter.setOpacity(0.9)
                painter.setPen(QPen(QColor(220,220,0)))
                painter.setFont(QFont("Helvetica", fontsize))
                painter.drawText(self.im1.rect(), Qt.AlignHCenter | Qt.AlignVCenter, "?")
            elif self.mark == "yellow" and self.cluster:
                painter.setOpacity(0.9)
                painter.setPen(QPen(QColor(220, 220, 0)))
                painter.setFont(QFont("Helvetica", fontsize))
                painter.drawText(self.im1.rect(), Qt.AlignHCenter | Qt.AlignVCenter, "√")
            elif self.mark == "red":
                painter.setOpacity(0.8)
                painter.setPen(QPen(QColor(220,0,0)))
                painter.setFont(QFont("Helvetica", fontsize))
                painter.drawText(self.im1.rect(), Qt.AlignHCenter | Qt.AlignVCenter, "X")
            else:
                print("ERROR: unrecognised segment mark")
                return

    def enterEvent(self, QEvent):
        # to reset the icon if it didn't stop cleanly
        if not self.media_obj.isPlaying():
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.show()

    def leaveEvent(self, QEvent):
        if not self.media_obj.isPlaying():
            self.playButton.hide()

    def mouseMoveEvent(self, ev):
        if ev.buttons() != Qt.LeftButton:
            return

        mimeData = QMimeData()

        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setPixmap(QPixmap("./img/Owl_thinking.png"))
        dropAction = drag.exec_(Qt.MoveAction)

    def playImage(self):
        if self.media_obj.isPlaying():
            self.stopPlayback()
        else:
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaStop))
            self.media_obj.loadArray(self.audiodata)

    def endListener(self):
        timeel = self.media_obj.elapsedUSecs() // 1000
        if timeel > self.duration:
            self.stopPlayback()

    def stopPlayback(self):
        self.media_obj.pressedStop()
        self.playButton.hide()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))

    def sizeHint(self):
        return self.im1.size()

    def minimumSizeHint(self):
        return self.im1.size()

    def changePic(self,ev):
        # cycle through CONFIRM / DELETE / RECHECK marks

        if self.cluster:
            if self.mark == "green":
                self.mark = "yellow"
            elif self.mark == "yellow":
                self.mark = "green"
        else:
            if self.mark == "green":
                self.mark = "red"
            elif self.mark == "red":
                self.mark = "yellow"
            elif self.mark == "yellow":
                self.mark = "green"
        self.paintEvent(ev)
        #self.update()
        self.repaint()
        pg.QtGui.QApplication.processEvents()

class Layout(pg.LayoutWidget):
    # Layout for the clustering that allows drag and drop
    buttonDragged = QtCore.Signal(int,object)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, ev):
        ev.accept()

    def dropEvent(self, ev):
        self.buttonDragged.emit(ev.pos().y(),ev.source())
        ev.setDropAction(Qt.MoveAction)
        ev.accept()
