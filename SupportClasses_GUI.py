
# SupportClasses_GUI.py
# Support classes for the AviaNZ program
# Mostly subclassed from pyqtgraph

# Version 3.0 14/09/20
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2020

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

from PyQt5.QtWidgets import QMessageBox, QAbstractButton, QListWidget, QListWidgetItem
from PyQt5.QtCore import Qt, QTime, QIODevice, QBuffer, QByteArray, QMimeData, QLineF, QLine, QPoint, QSize, QDir
from PyQt5.QtMultimedia import QAudio, QAudioOutput
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen, QColor, QFont, QDrag

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.functions as fn

import Segment

import wavio
from time import sleep
import time
import math
import numpy as np
import os
import io

class TimeAxisHour(pg.AxisItem):
    # Time axis (at bottom of spectrogram)
    # Writes the time as hh:mm:ss, and can add an offset
    def __init__(self, *args, **kwargs):
        super(TimeAxisHour, self).__init__(*args, **kwargs)
        self.offset = 0
        self.setLabel('Time', units='hh:mm:ss')
        self.showMS = False

    def setShowMS(self,value):
        self.showMS = value

    def tickStrings(self, values, scale, spacing):
        # Overwrite the axis tick code
        if self.showMS:
            self.setLabel('Time', units='hh:mm:ss.ms')
            return [QTime(0,0,0).addMSecs((value+self.offset)*1000).toString('hh:mm:ss.z') for value in values]
        else:
            self.setLabel('Time', units='hh:mm:ss')
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
        self.setLabel('Time', units='mm:ss.z')
        self.showMS = False

    def setShowMS(self,value):
        self.showMS = value

    def tickStrings(self, values, scale, spacing):
        # Overwrite the axis tick code
        # First, get absolute time ('values' are relative to page start)
        if len(values)==0:
            return []
        vs = [value + self.offset for value in values]
        if self.showMS:
            self.setLabel('Time', units='mm:ss.ms')
            vstr1 = [QTime(0,0,0).addMSecs(value*1000).toString('mm:ss.z') for value in vs]
            # check if we need to add hours:
            if vs[-1]>=3600:
                self.setLabel('Time', units='h:mm:ss.ms')
                for i in range(len(vs)):
                    if vs[i]>=3600:
                        vstr1[i] = QTime(0,0,0).addMSecs(vs[i]*1000).toString('h:mm:ss.z')
            return vstr1
        else:
            self.setLabel('Time', units='mm:ss')
            vstr1 = [QTime(0,0,0).addSecs(value).toString('mm:ss') for value in vs]
            # check if we need to add hours:
            if vs[-1]>=3600:
                self.setLabel('Time', units='h:mm:ss')
                for i in range(len(vs)):
                    if vs[i]>=3600:
                        vstr1[i] = QTime(0,0,0).addSecs(vs[i]).toString('h:mm:ss')
            return vstr1

    def setOffset(self,offset):
        self.offset = offset
        self.update()


class AxisWidget(QAbstractButton):
    # Axis shown along the side of Single Sp buttons
    def __init__(self, sgsize, minFreq, maxFreq, parent=None):
        super(AxisWidget, self).__init__(parent)
        self.minFreq = minFreq
        self.maxFreq = maxFreq
        self.sgsize = sgsize

        # fixed size
        self.setSizePolicy(0,0)
        self.setMinimumSize(70, sgsize)
        self.fontsize = min(max(int(math.sqrt(sgsize-30)*0.8), 9), 13)

    def paintEvent(self, event):
        if type(event) is not bool:
            painter = QPainter(self)
            # actual axis line painting
            bottomR = event.rect().bottomRight()
            bottomR.setX(bottomR.x()-12)
            topR = event.rect().topRight()
            topR.setX(topR.x()-12)
            painter.setPen(QPen(QColor(20,20,20), 1))
            painter.drawLine(bottomR, topR)

            painter.setFont(QFont("Helvetica", self.fontsize))

            # draw tickmarks and numbers
            currFrq = self.minFreq
            fontOffset = 5 + 2.6*self.fontsize
            tickmark = QLine(bottomR, QPoint(bottomR.x()+6, bottomR.y()))
            painter.drawLine(tickmark)
            painter.drawText(tickmark.x2()-fontOffset, tickmark.y2()+1, "%.1f" % currFrq)
            for ticknum in range(3):
                currFrq += (self.maxFreq - self.minFreq)/4
                tickmark.translate(0, -event.rect().height()//4)
                painter.drawLine(tickmark)
                painter.drawText(tickmark.x2()-fontOffset, tickmark.y2()+self.fontsize//2, "%.1f" % currFrq)
            tickmark.translate(0, -tickmark.y2())
            painter.drawLine(tickmark)
            painter.drawText(tickmark.x2()-fontOffset, tickmark.y2()+self.fontsize+1, "%.1f" % self.maxFreq)

            painter.save()
            painter.translate(self.fontsize//2, event.rect().height()//2)
            painter.rotate(-90)
            painter.drawText(-12, 8, "kHz")
            painter.restore()

    def sizeHint(self):
        return QSize(60, self.sgsize)

    def minimumSizeHint(self):
        return QSize(60, self.sgsize)

class TimeAxisWidget(QAbstractButton):
    # Class for HumanClassify dialogs to put spectrograms on buttons
    # Also includes playback capability.
    def __init__(self, sgsize, maxTime, parent=None):
        super(TimeAxisWidget, self).__init__(parent)
        self.sgsize = sgsize
        self.maxTime = maxTime

        # fixed size
        self.setSizePolicy(0,0)
        self.setMinimumSize(sgsize, 40)
        self.setMaximumSize(sgsize, 50)
        self.fontsize = min(max(int(math.sqrt(sgsize)*0.55), 9), 13)

    def paintEvent(self, event):
        if type(event) is not bool:
            painter = QPainter(self)
            # actual axis line painting
            bottomL = event.rect().bottomLeft()
            bottomR = event.rect().bottomRight()
            top = event.rect().top()
            painter.setPen(QPen(QColor(20,20,20), 1))

            painter.setFont(QFont("Helvetica", self.fontsize))

            # draw tickmarks and numbers
            currTime = 0
            fontOffset = 5+1.5*self.fontsize
            if self.maxTime>=10:
                timeFormat = "%d"
            else:
                timeFormat = "%.1f"

            painter.drawLine(bottomL.x(), top+6, bottomR.x(), top+6)

            tickmark = QLine(bottomL.x(), top+6, bottomL.x(), top)
            painter.drawLine(tickmark)
            painter.drawText(tickmark.x1(), tickmark.y1()+fontOffset, timeFormat % currTime)
            for ticknum in range(4):
                currTime += self.maxTime/5
                tickmark.translate(event.rect().width()//5,0)
                painter.drawLine(tickmark)
                painter.drawText(tickmark.x1()-fontOffset//4, tickmark.y1()+fontOffset, timeFormat % currTime)
            tickmark.translate(event.rect().width()//5-2,0)
            painter.drawLine(tickmark)
            painter.drawText(tickmark.x2()-fontOffset*0.7, tickmark.y1()+fontOffset, timeFormat % self.maxTime)

            painter.save()
            painter.drawText((bottomR.x() - bottomL.x())//2, bottomL.y(), "s")
            painter.restore()

    def sizeHint(self):
        return QSize(self.sgsize,60)

    def minimumSizeHint(self):
        return QSize(self.sgsize,60)

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


class DemousedViewBox(pg.ViewBox):
    # A version of ViewBox with no mouse events.
    # Dramatically reduces CPU usage when such events are not needed.
    def keyPressEvent(self, ev):
        return

    def mouseDragEvent(self, ev, axis=None):
        return

    def mouseClickEvent(self, ev):
        return

    def mouseMoveEvent(self, ev):
        return

    def wheelEvent(self, ev, axis=None):
        return


# Two subclasses of LinearRegionItem, that account for spectrogram bounds when resizing
# and use boundary caching to reduce CPU load e.g. when detecting mouse hover
class LinearRegionItem2(pg.LinearRegionItem):
    def __init__(self, parent, bounds=None, *args, **kwds):
        pg.LinearRegionItem.__init__(self, bounds, *args, **kwds)
        self.parent = parent
        self.bounds = bounds
        self.useCachedView = None
        # we don't provide parent, and therefore don't switch buttons,
        # when using this for overview
        if self.parent is not None:
            self.lines[0].btn = self.parent.MouseDrawingButton
            self.lines[1].btn = self.parent.MouseDrawingButton
        self.setHoverBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 100)))

    def setHoverBrush(self, *br, **kargs):
        self.hoverBrush = fn.mkBrush(*br, **kargs)

    def setPen(self, *pen, **kargs):
        self.lines[0].setPen(*pen, **kargs)
        self.lines[1].setPen(*pen, **kargs)

    def viewRect(self):
        """ Return the visible bounds of this item's ViewBox or GraphicsWidget,
            in the local coordinate system of the item.
            Overwritten to use caching. """
        if self.useCachedView is not None:
            return self.useCachedView

        view = self.getViewBox()
        if view is None:
            return None
        bounds = view.viewRect()
        bounds = self.mapRectFromView(bounds)
        if bounds is None:
            return None

        bounds = bounds.normalized()

        # For debugging cache misses:
        # if self.useCachedView is not None:
        #     if self.useCachedView.top()!=bounds.top() or self.useCachedView.bottom()!=bounds.bottom():
        #         import traceback
        #         traceback.print_stack()
        #         print("cached:", self.useCachedView)
        #         print(bounds)

        self.useCachedView = bounds
        return bounds

    def viewTransformChanged(self):
        # Clear cache
        self.useCachedView = None

    # def boundingRect(self):
    #     # because we react to hover, this is called frequently

    #     # ORIGINAL:
    #     br = self.viewRect()  # bounds of containing ViewBox mapped to local coords.

    #     rng = self.getRegion()
    #     br.setLeft(rng[0])
    #     br.setRight(rng[1])
    #     length = br.height()
    #     br.setBottom(br.top() + length * self.span[1])
    #     br.setTop(br.top() + length * self.span[0])

    #     br = br.normalized()

    #     if self._bounds != br:
    #         print("Preparing geom")
    #         self._bounds = br
    #         self.prepareGeometryChange()

    #     return br

    def mouseDragEvent(self, ev):
        if not self.movable or (self.parent is not None and ev.button()==self.parent.MouseDrawingButton):
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


# Just another slight optimization - immediately dropping unneeded mouse events
class LinearRegionItemO(LinearRegionItem2):
    def __init__(self, *args, **kwds):
        LinearRegionItem2.__init__(self, parent=None, bounds=[0,100], *args, **kwds)

    def setRegion(self, rgn):
        """Set the values for the edges of the region.
        ==============   ==============================================
        **Arguments:**
        rgn              A list or tuple of the lower and upper values.
        bounds           A tuple indicating allowed x range
        ==============   ==============================================
        """
        if self.lines[0].value() == rgn[0] and self.lines[1].value() == rgn[1]:
            return
        # shift the requested length to fit within bounds:
        if self.bounds[0] is not None:
            if rgn[0]<self.bounds[0]:
                ll = rgn[1]-rgn[0]
                rgn[0] = self.bounds[0]
                rgn[1] = rgn[0]+ll
        if self.bounds[1] is not None:
            if rgn[1]>self.bounds[1]:
                ll = rgn[1]-rgn[0]
                rgn[1] = self.bounds[1]
                rgn[0] = max(0, rgn[1]-ll)
        self.blockLineSignal = True
        self.lines[0].setValue(rgn[0])
        self.lines[1].setValue(rgn[1])
        self.blockLineSignal = False
        # self.lineMoved(0)
        # self.lineMoved(1)
        self.lineMoveFinished()

    def setBounds(self, bounds):
        self.bounds = bounds
        super(LinearRegionItemO, self).setBounds(bounds)

    # identical to original, just w/o debugger
    def paint(self, p, *args):
        p.setBrush(self.currentBrush)
        p.setPen(fn.mkPen(None))
        p.drawRect(self.boundingRect())

    # Immediate rejects on all unneeded events:
    def keyPressEvent(self, ev):
        return

    def mouseClickEvent(self, ev):
        ev.accept()
        return

    def wheelEvent(self, ev):
        ev.accept()
        return

    # Other events could be dropped too:
    # def lineMoved(self, i):
    #     return
    # def lineMoveFinished(self):
    #     return
    # def setMouseHover(self, hover):
    #     return
    # def hoverEvent(self, ev):
    #     return


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
        # send the position of this rectangle in ViewBox coords
        # left corner:
        # x = self.mapRectToParent(self.boundingRect()).x()
        # or center:
        x = self.mapRectToParent(self.boundingRect()).center().x()
        self.parentWidget().resend(x)


class PartlyResizableGLW(pg.GraphicsLayoutWidget):
    # a widget which has a fixed aspect ratio, set by height.
    # useful for horizontal scroll areas.
    def __init__(self):
        self.plotAspect = 5
        # to prevent infinite loops:
        self.alreadyResizing = False
        super(PartlyResizableGLW, self).__init__()

    def forceResize(self):
        # this should be doable by postEvent(QResizeEvent),
        # but somehow doesn't always work.
        self.alreadyResizing = False
        self.setMinimumWidth(self.height()*self.plotAspect-10)
        self.setMaximumWidth(self.height()*self.plotAspect+10)
        self.adjustSize()

    def resizeEvent(self, e):
        if e is not None:
            # break any infinite loops,
            # and also processes every second event:
            if self.alreadyResizing:
                self.alreadyResizing = False
                return

            self.alreadyResizing = True
            # Some buffer for flexibility, so that it could adjust itself
            # and avoid infinite loops
            self.setMinimumWidth(e.size().height()*self.plotAspect-10)
            self.setMaximumWidth(e.size().height()*self.plotAspect+10)

            pg.GraphicsLayoutWidget.resizeEvent(self, e)


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
            self.setText("The AviaNZ Program, v3.1.2 (November 2020)")
            self.setInformativeText("By Stephen Marsland, Victoria University of Wellington. With code by Nirosha Priyadarshani, Julius Juodakis, and Virginia Listanti. Input from Isabel Castro, Moira Pryde, Stuart Cockburn, Rebecca Stirnemann, Sumudu Purage, and Rebecca Huistra. \n stephen.marsland@vuw.ac.nz")
        elif (type=="o"):
            self.setIconPixmap(QPixmap("img/AviaNZ.png"))

        self.setWindowIcon(QIcon("img/Avianz.ico"))

        # by default, adding OK button. Can easily be overwritten after creating
        self.setStandardButtons(QMessageBox.Ok)

class PicButton(QAbstractButton):
    # Class for HumanClassify dialogs to put spectrograms on buttons
    # Also includes playback capability.
    def __init__(self, index, spec, audiodata, format, duration, unbufStart, unbufStop, lut, colStart, colEnd, cmapInv, guides=None, parent=None, cluster=False):
        super(PicButton, self).__init__(parent)
        self.index = index
        self.mark = "green"
        self.spec = spec
        self.unbufStart = unbufStart
        self.unbufStop = unbufStop
        self.cluster = cluster
        self.setMouseTracking(True)

        self.playButton = QtGui.QToolButton(self)
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.hide()
        # check if playback possible (e.g. batmode)
        if len(audiodata)>0:
            self.noaudio = False
            self.playButton.clicked.connect(self.playImage)
        else:
            self.noaudio = True
            # batmode frequency guides (in Y positions 0-1)
            if guides is not None:
                self.guides = guides
                self.guidelines = [0]*len(self.guides)

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
        self.duration = duration * 1000  # in ms

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

        # Output image width - larger for batmode:
        if self.noaudio:
            targwidth = 750
        else:
            targwidth = 500

        # hardcode all image sizes
        if self.cluster:
            self.im1 = im1.scaled(200, 150)
        else:
            self.specReductionFact = im1.size().width()/targwidth
            # use original height if it is not extreme
            prefheight = max(192, min(im1.size().height(), 512))
            self.im1 = im1.scaled(targwidth, prefheight)

            heightRedFact = im1.size().height()/prefheight

            # draw lines marking true segment position
            unbufStartAdj = self.unbufStart / self.specReductionFact
            unbufStopAdj = self.unbufStop / self.specReductionFact
            self.line1 = QLineF(unbufStartAdj, 0, unbufStartAdj, self.im1.size().height())
            self.line2 = QLineF(unbufStopAdj, 0, unbufStopAdj, self.im1.size().height())

            # create guides for batmode
            if self.noaudio:
                for i in range(len(self.guides)):
                    self.guidelines[i] = QLineF(0, self.im1.height() - self.guides[i]/heightRedFact, targwidth, self.im1.height() - self.guides[i]/heightRedFact)

    def paintEvent(self, event):
        if type(event) is not bool:
            # this will repaint the entire widget, rather than
            # trying to squish the image into the current viewport
            rect = self.im1.rect()

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
            painter.drawImage(rect, self.im1)
            if not self.cluster:
                painter.drawLine(self.line1)
                painter.drawLine(self.line2)

            if self.noaudio:
                painter.setPen(QPen(QColor(255,232,140), 2))
                painter.drawLine(self.guidelines[0])
                painter.drawLine(self.guidelines[3])
                painter.setPen(QPen(QColor(239,189,124), 2))
                painter.drawLine(self.guidelines[1])
                painter.drawLine(self.guidelines[2])

            # draw decision mark
            fontsize = int(self.im1.size().height() * 0.65)
            if self.mark == "green":
                pass
            elif self.mark == "yellow" and not self.cluster:
                painter.setOpacity(0.9)
                painter.setPen(QPen(QColor(220,220,0)))
                painter.setFont(QFont("Helvetica", fontsize))
                painter.drawText(rect, Qt.AlignHCenter | Qt.AlignVCenter, "?")
            elif self.mark == "yellow" and self.cluster:
                painter.setOpacity(0.9)
                painter.setPen(QPen(QColor(220, 220, 0)))
                painter.setFont(QFont("Helvetica", fontsize))
                painter.drawText(rect, Qt.AlignHCenter | Qt.AlignVCenter, "√")
            elif self.mark == "red":
                painter.setOpacity(0.8)
                painter.setPen(QPen(QColor(220,0,0)))
                painter.setFont(QFont("Helvetica", fontsize))
                painter.drawText(rect, Qt.AlignHCenter | Qt.AlignVCenter, "X")
            else:
                print("ERROR: unrecognised segment mark")
                return

    def enterEvent(self, QEvent):
        # to reset the icon if it didn't stop cleanly
        if self.noaudio:
            return
        if not self.media_obj.isPlaying():
            self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.show()

    def leaveEvent(self, QEvent):
        if self.noaudio:
            return
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


class LightedFileList(QListWidget):
    """ File list with traffic light icons.
        On init (or after any change), pass the red, darkyellow, and green colors.
    """
    def __init__(self, ColourNone, ColourPossibleDark, ColourNamed):
        super().__init__()
        self.ColourNone = ColourNone
        self.ColourPossibleDark = ColourPossibleDark
        self.ColourNamed = ColourNamed
        self.soundDir = None
        self.spList = set()
        self.fsList = set()
        self.listOfFiles = []
        self.minCertainty = 100

        # for the traffic light icons
        self.pixmap = QPixmap(10, 10)
        self.blackpen = fn.mkPen(color=(160,160,160,255), width=2)
        self.tempsl = Segment.SegmentList()

    def fill(self, soundDir, fileName, recursive=False, readFmt=False, addWavNum=False):
        """ read folder contents, populate the list widget.
            soundDir: current dir
            fileName: file which should be selected, or None
            recursive: should we read the species list/format info from subdirs as well?
            readFmt: should we read the wav header as well?
            addWavNum: add extra info to the end of dir names
        """
        # clear current listbox
        self.clearSelection()
        self.clearFocus()
        self.clear()
        # collect some additional info about the current dir
        self.spList = set()
        self.fsList = set()
        self.listOfFiles = []
        self.minCertainty = 100     # TODO: not used in training, can remove?

        with pg.BusyCursor():
            # Read contents of current dir
            self.listOfFiles = QDir(soundDir).entryInfoList(['..','*.wav','*.bmp'],filters=QDir.AllDirs | QDir.NoDot | QDir.Files,sort=QDir.DirsFirst)
            self.soundDir = soundDir

            for file in self.listOfFiles:
                # add entry to the list
                item = QListWidgetItem(self)

                if file.isDir():
                    if file.fileName()=="..":
                        item.setText(file.fileName() + "/")
                        continue

                    # detailed dir view can be used for non-clickable instances
                    if addWavNum:
                        # count wavs in this dir:
                        numbmps = 0
                        numwavs = 0
                        for root, dirs, files in os.walk(file.filePath()):
                            numwavs += sum(f.lower().endswith('.wav') for f in files)
                            numbmps += sum(f.lower().endswith('.bmp') for f in files)
                        # keep these strings as short as possible
                        if numbmps==0:
                            item.setText("%s/\t\t(%d wav files)" % (file.fileName(), numwavs))
                        elif numwavs==0:
                            item.setText("%s/\t\t(%d bmp files)" % (file.fileName(), numbmps))
                        else:
                            item.setText("%s/\t\t(%d wav, %d bmp files)" % (file.fileName(), numwavs, numbmps))
                    else:
                        item.setText(file.fileName() + "/")

                    # We still might need to walk the subfolders for sp lists and wav formats!
                    if not recursive:
                        continue
                    for root, dirs, files in os.walk(file.filePath()):
                        for filename in files:
                            filenamef = os.path.join(root, filename)
                            if filename.lower().endswith('.wav') or filename.lower().endswith('.bmp'):
                                if readFmt:
                                    if filename.lower().endswith('.wav'):
                                        try:
                                            samplerate = wavio.readFmt(filenamef)[0]
                                            self.fsList.add(samplerate)
                                        except Exception as e:
                                            print("Warning: could not parse format of WAV file", filenamef)
                                            print(e)
                                    else:
                                        # For bitmaps, using hardcoded samplerate as there's no readFmt
                                        self.fsList.add(176000)

                                # Data files can accompany either wavs or bmps
                                dataf = filenamef + '.data'
                                if os.path.isfile(dataf):
                                    try:
                                        self.tempsl.parseJSON(dataf, silent=True)
                                        if len(self.tempsl)>0:
                                            # collect any species present
                                            filesp = [lab["species"] for seg in self.tempsl for lab in seg[4]]
                                            self.spList.update(filesp)
                                            # min certainty
                                            cert = [lab["certainty"] for seg in self.tempsl for lab in seg[4]]
                                            if cert:
                                                mincert = min(cert)
                                                if self.minCertainty > mincert:
                                                    self.minCertainty = mincert
                                    except Exception as e:
                                        # .data exists, but unreadable
                                        print("Could not read DATA file", dataf)
                                        print(e)
                else:
                    item.setText(file.fileName())

                    # check for a data file here and color this entry based on that
                    fullname = os.path.join(soundDir, file.fileName())
                    # (also updates the directory info sets, and minCertainty)
                    self.paintItem(item, fullname+'.data')
                    # format collection only implemented for WAVs currently
                    if readFmt:
                        if file.fileName().lower().endswith('.wav'):
                            try:
                                samplerate = wavio.readFmt(fullname)[0]
                                self.fsList.add(samplerate)
                            except Exception as e:
                                print("Warning: could not parse format of WAV file", fullname)
                                print(e)
                        if file.fileName().lower().endswith('.bmp'):
                            # For bitmaps, using hardcoded samplerate as there's no readFmt
                            self.fsList.add(176000)

        if readFmt:
            print("Found the following Fs:", self.fsList)

        # mark the current file or first row (..), if not found
        if fileName:
            # for matching dirs:
            # index = self.findItems(fileName+"\/",Qt.MatchExactly)
            index = self.findItems(fileName,Qt.MatchExactly)
            if len(index)>0:
                self.setCurrentItem(index[0])
            else:
                self.setCurrentRow(0)

    def refreshFile(self, fileName, cert):
        """ Repaint a single file icon with the provided certainty.
            fileName: file stem (dir will be read from self)
            cert:     0-100, or -1 if no annotations
        """
        # for matching dirs - not sure if needed:
        # index = self.findItems(fileName+"\/",Qt.MatchExactly)
        index = self.findItems(fileName,Qt.MatchExactly)
        if len(index)==0:
            return

        curritem = index[0]
        # Repainting identical to paintItem
        if cert == -1:
            # .data exists, but no annotations
            self.pixmap.fill(QColor(255,255,255,0))
            painter = QPainter(self.pixmap)
            painter.setPen(self.blackpen)
            painter.drawRect(self.pixmap.rect())
            painter.end()
            curritem.setIcon(QIcon(self.pixmap))
            # no change to self.minCertainty
        elif cert == 0:
            self.pixmap.fill(self.ColourNone)
            curritem.setIcon(QIcon(self.pixmap))
            self.minCertainty = 0
        elif cert < 100:
            self.pixmap.fill(self.ColourPossibleDark)
            curritem.setIcon(QIcon(self.pixmap))
            self.minCertainty = min(self.minCertainty, cert)
        else:
            self.pixmap.fill(self.ColourNamed)
            curritem.setIcon(QIcon(self.pixmap))
            # self.minCertainty cannot be changed by a cert=100 segment

    def paintItem(self, item, datafile):
        """ Read the JSON and draw the traffic light for a single item """
        filesp = []
        if os.path.isfile(datafile):
            # Try loading the segments to get min certainty
            try:
                self.tempsl.parseJSON(datafile, silent=True)
                if len(self.tempsl)==0:
                    # .data exists, but empty - "file was looked at"
                    mincert = -1
                else:
                    cert = [lab["certainty"] for seg in self.tempsl for lab in seg[4]]
                    if cert:
                        mincert = min(cert)
                    else:
                        mincert = -1
                    # also collect any species present
                    filesp = [lab["species"] for seg in self.tempsl for lab in seg[4]]
            except Exception as e:
                # .data exists, but unreadable
                print("Could not determine certainty for file", datafile)
                print(e)
                mincert = -1

            if mincert == -1:
                # .data exists, but no annotations
                self.pixmap.fill(QColor(255,255,255,0))
                painter = QPainter(self.pixmap)
                painter.setPen(self.blackpen)
                painter.drawRect(self.pixmap.rect())
                painter.end()
                item.setIcon(QIcon(self.pixmap))

                # no change to self.minCertainty
            elif mincert == 0:
                self.pixmap.fill(self.ColourNone)
                item.setIcon(QIcon(self.pixmap))
                self.minCertainty = 0
            elif mincert < 100:
                self.pixmap.fill(self.ColourPossibleDark)
                item.setIcon(QIcon(self.pixmap))
                self.minCertainty = min(self.minCertainty, mincert)
            else:
                self.pixmap.fill(self.ColourNamed)
                item.setIcon(QIcon(self.pixmap))
                # self.minCertainty cannot be changed by a cert=100 segment
        else:
            # no .data for this sound file
            self.pixmap.fill(QColor(255,255,255,0))
            item.setIcon(QIcon(self.pixmap))

        # collect some extra info about this file as we've read it anyway
        self.spList.update(filesp)
