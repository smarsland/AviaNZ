
# coding=latin-1

# SupportClasses_GUI.py
# Support classes for the AviaNZ program
# Mostly subclassed from pyqtgraph

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

from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox, QAbstractButton, QListWidget, QListWidgetItem, QPushButton, QSlider, QLabel, QHBoxLayout, QGridLayout, QWidget, QGraphicsRectItem, QLayout, QToolButton, QStyle, QSizePolicy, QMenu
from PyQt6.QtCore import Qt, QTime, QTimer, QIODevice, QBuffer, QByteArray, QMimeData, QLineF, QLine, QPoint, QSize, QDir, pyqtSignal, pyqtSlot, QThread, QEvent
from PyQt6.QtMultimedia import QAudio, QAudioOutput, QAudioSink, QAudioFormat, QMediaDevices
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QPen, QColor, QFont, QDrag

import pyqtgraph as pg
import pyqtgraph.functions as fn

import Segment
import SignalProc

from time import sleep
import time
import math
import numpy as np
import os
import io
import Spectrogram

from functools import partial
import soundfile as sf
import threading
import re

import copy

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
        # Overwrite the axis \u2714 code
        if self.showMS:
            self.setLabel('Time', units='hh:mm:ss.ms')
            return [QTime(0,0,0).addMSecs(int(value+self.offset)*1000).toString('hh:mm:ss.z') for value in values]
        else:
            self.setLabel('Time', units='hh:mm:ss')
            return [QTime(0,0,0).addSecs(int(value+self.offset)).toString('hh:mm:ss') for value in values]

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
        # Overwrite the axis \u2714 code
        # First, get absolute time ('values' are relative to page start)
        if len(values)==0:
            return []
        vs = [value + self.offset for value in values]
        if self.showMS:
            self.setLabel('Time', units='mm:ss.ms')
            vstr1 = [QTime(0,0,0).addMSecs(int(value*1000)).toString('mm:ss.z') for value in vs]
            # check if we need to add hours:
            if vs[-1]>=3600:
                self.setLabel('Time', units='h:mm:ss.ms')
                for i in range(len(vs)):
                    if vs[i]>=3600:
                        vstr1[i] = QTime(0,0,0).addMSecs(int(vs[i]*1000)).toString('h:mm:ss.z')
            return vstr1
        else:
            self.setLabel('Time', units='mm:ss')
            # SRM: bug? (int)
            vstr1 = [QTime(0,0,0).addSecs(int(value)).toString('mm:ss') for value in vs]
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
        self.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Fixed)
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
            painter.drawText(int(tickmark.x2()-fontOffset), int(tickmark.y2()+1), "%.1f" % currFrq)
            for ticknum in range(3):
                currFrq += (self.maxFreq - self.minFreq)/4
                tickmark.translate(0, -event.rect().height()//4)
                painter.drawLine(tickmark)
                painter.drawText(int(tickmark.x2()-fontOffset), int(tickmark.y2()+self.fontsize//2), "%.1f" % currFrq)
            tickmark.translate(0, -tickmark.y2())
            painter.drawLine(tickmark)
            painter.drawText(int(tickmark.x2()-fontOffset), int(tickmark.y2()+self.fontsize+1), "%.1f" % self.maxFreq)

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
        self.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Fixed)
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
            painter.drawText(int(tickmark.x1()), int(tickmark.y1()+fontOffset), timeFormat % currTime)
            for ticknum in range(4):
                currTime += self.maxTime/5
                tickmark.translate(event.rect().width()//5,0)
                painter.drawLine(tickmark)
                painter.drawText(int(tickmark.x1()-fontOffset//4), int(tickmark.y1()+fontOffset), timeFormat % currTime)
            tickmark.translate(event.rect().width()//5-2,0)
            painter.drawLine(tickmark)
            painter.drawText(int(tickmark.x2()-fontOffset*0.7), int(tickmark.y1()+fontOffset), timeFormat % self.maxTime)

            painter.save()
            painter.drawText(int((bottomR.x() - bottomL.x())//2), int(bottomL.y()), "s")
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
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
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
    if self.movable and hasattr(self, 'btn') and ev.button() != self.btn:
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
            snap = True if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier) else None
            #snap = True if (ev.modifiers() & QtCore.Qt.ControlModifier) else None
            newPos = self.mapToParent(ev.pos()) + self.cursorOffset
            self.translate(newPos - self.pos(), snap=snap, finish=False)

pg.graphicsItems.ROI.Handle.mouseDragEvent = mouseDragEventFlexible
pg.graphicsItems.InfiniteLine.InfiniteLine.mouseDragEvent = mouseDragEventFlexibleLine


class DemousedViewBox(pg.ViewBox):
    # A version of ViewBox with no mouse events.
    # Dramatically reduces CPU usage when such events are not needed.
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
    sigMouseDragged = QtCore.pyqtSignal(object,object,object)
    keyPressed = QtCore.pyqtSignal(int)

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
        #         self.updateScaleBox(ev.buttonDownPos(), ev.position())
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
    sigChildMessage = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)

    def resend(self,x):
        self.sigChildMessage.emit(x)


class ClickableRectItem(QGraphicsRectItem):
    # QGraphicsItem doesn't include signals, hence this mess
    def __init__(self, *args, **kwds):
        QGraphicsRectItem.__init__(self, *args, **kwds)

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
        super().__init__()
        self.plotAspect = 5
        super(PartlyResizableGLW, self).__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def forceResize(self):
        self.setMinimumWidth(0)
        self.setMaximumWidth(9999)
        self.adjustSize()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.plotAspect = self.width() / max(1, self.height())
        # Optionally enforce aspect ratio on the inner widget
        if hasattr(self, 'wPlot'):
            self.wPlot.resize(self.width(), self.height())

    def zoomIn(self):
        # Increase the width by a factor (e.g., 1.1 for 10% zoom in)
        new_width = int(self.width() * 1.1)
        self.setFixedWidth(new_width)  # Force the width to change
        self.updateGeometry()  # Notify the layout system of the change

    def zoomOut(self):
        # Decrease the width by a factor (e.g., 0.9 for 10% zoom out)
        new_width = int(self.width() * 0.9)
        self.setFixedWidth(new_width)  # Force the width to change
        self.updateGeometry()  # Notify the layout system of the change

class ControllableAudio(QAudioSink):
    # This carries out the audio playback 
    # Pass in either an audioFormat or a ref to Spectrogram
    # If called by main interface, starts a timer for the moving bar
    #failed = pyqtSignal(str)
    #need_msg = pyqtSignal(str, str)
    #need_clean_UI = pyqtSignal(int, int)

    def __init__(self, sp=None, loop=False, audioFormat=None,useBar=False):
        # Note the order here is audioFormat passed, otherwise sp.audioFormat
        #print(self.audioFormat.sampleFormat(), self.audioFormat.sampleRate(), self.audioFormat.bytesPerSample(), self.audioFormat.channelCount())

        if audioFormat is not None:
            self.audioFormat = audioFormat
        else:
            if sp is None:
                print("Error!: audio format needed: no sound can be played")
                return
            self.audioFormat = sp.audioFormat

        if self.audioFormat.sampleFormat() == QAudioFormat.SampleFormat.Int16:
            self.sampwidth = 2
        elif self.audioFormat.sampleFormat() == QAudioFormat.SampleFormat.Int32:
            self.sampwidth = 4
        elif self.audioFormat.sampleFormat() == QAudioFormat.SampleFormat.UInt8:
            self.sampwidth = 1
        else:
            print("ERROR: sampleSize %d not supported" % self.audioFormat.sampleSize())
        super(ControllableAudio, self).__init__(QMediaDevices.defaultAudioOutput(), format=self.audioFormat)
        #self.setBufferSize(int(sampwidth*8 * self.audioFormat.sampleRate()/100 * self.audioFormat.channelCount()))
        self.bytesPerSecond = int(self.sampwidth * self.audioFormat.sampleRate() * self.audioFormat.channelCount())
        # TODO: or the size of the data if < 4 secs
        self.setBufferSize(int(self.bytesPerSecond/0.25)) # 4 s buffer
        #print("buffer: ",int(sampwidth*8 * self.audioFormat.sampleRate()/100 * self.audioFormat.channelCount()))

        #super(ControllableAudio, self).__init__(format=self.audioFormat)

        #self.media_thread = QThread()
        #self.moveToThread(self.media_thread)

        # This is a timer for the moving bar. 
        # On this notify, move slider (connected where called)
        self.useBar = useBar
        #self.useBar = False
        if self.useBar:
            self.NotifyTimer = QTimer(self)
            #self.NotifyTimer.timeout.connect(self.NotifyTimerTick)
        #self.stateChanged.connect(self.endListener)

        self.timeoffset = 0  # start time of the played audio, in ms, relative to page start
        #self.loop = loop
        self.sp = sp
        self.audioThread = None
        self.audioThreadLoading = False
        self.audioThreadPaused = False
        self.playbackSpeed = 1.0
        self.bytesWritten = 0
        # set small buffer (10 ms) and use processed time
        # sampwidth*8 is the sampleSize
        #self.setBufferSize(int(sampwidth*8 * self.audioFormat.sampleRate()/100 * self.audioFormat.channelCount()))
        #print("Buffer size: ",int(sampwidth*8 * self.audioFormat.sampleRate()/100 * self.audioFormat.channelCount()),self.bufferSize())
        #print("Buffer size: ",self.bufferSize())
        #self.setBufferSize(int(self.format().sampleSize() * self.format().sampleRate()/100 * self.format().channelCount()))
    
    def setSpeed(self, speed):
        self.playbackSpeed = speed

    @pyqtSlot()
    def isPlaying(self):
        return(self.state() == QAudio.State.ActiveState)

    @pyqtSlot()
    def isPlayingorPaused(self):
        return(self.state() == QAudio.State.ActiveState or self.state() == QAudio.State.SuspendedState)

    def endListener(self):
        # this should only be called if there's some misalignment between GUI and Audio
        # Should deal with underrun errors somehow
        # TODO: eventually, remove
        print(self.bufferSize(),self.bytesFree(), self.elapsedUSecs())
        print("endlistener",self.state(),self.error())
        # This is to catch when things finish
        #if self.state() == QAudio.State.StoppedState:
            #self.pressedStop()
            #self.reset()
        # NOTE: code below is under return!
        #elif self.state() == QAudio.State.IdleState and self.error() == QAudio.Error.NoError:
            #print("ended",self.loop)
            #if self.loop:
                #self.restart()
            #else:
            #self.pressedStop()
            #self.reset()
        return
        # give some time for GUI to catch up and stop
        # this should only be called if there's some misalignment between GUI and Audio
        if self.state() == QAudio.State.IdleState:
            # give some time for GUI to catch up and stop
            sleepCycles = 0
            while(self.state() != QAudio.State.StoppedState and sleepCycles < 30):
                sleep(0.03)
                sleepCycles += 1
                # This loop stops when timeoffset+processedtime > designated stop position.
                # By adding this offset, we ensure the loop stops even if
                # processed audio timer breaks somehow.
                self.timeoffset += 30
                #self.notify.emit()
            self.pressedStop()

        #print(self.error(), QAudio.Error.UnderrunError, self.error() == QAudio.Error.UnderrunError)
        #if self.error() == QAudio.Error.UnderrunError:
            #print("yes")
        #sleepCycles = 0
        #while(self.state() != QAudio.State.StoppedState and sleepCycles < 30):
            #print("sleeping")
            #sleep(0.03)
            #sleepCycles += 1
            # This loop stops when timeoffset+processedtime > designated stop position.
            # By adding this offset, we ensure the loop stops even if
            # processed audio timer breaks somehow.
            #self.timeoffset += 30
            #self.notify.emit()
        #self.pressedStop()

    @pyqtSlot()
    def pressedPlay(self, start=0, stop=0):#, audiodata=None):
        # If playback bar has not moved, this can use resume() to continue from the same spot.
        # Otherwise assumes that the QAudioOutput was stopped/reset. In that case the updated 
        # position is passed as start, and playing starts anew from there.
        print("---", self.state(),start)
        if self.state() == QAudio.State.SuspendedState:
            self.audioThreadPaused = False
            self.resume()
            if self.useBar:
                self.NotifyTimer.start(30)
        else:
            self.pressedStop()
            self.playSeg(start, stop) 

    @pyqtSlot()
    def pressedPause(self):
        self.audioThreadPaused = True
        self.suspend()
        if self.useBar:
            self.NotifyTimer.stop()

    @pyqtSlot()
    def pressedStop(self):
        # stop and reset to window/segment start

        # finish the threads
        self.audioThreadLoading = False
        if not self.audioThread is None:
            self.audioThread.join() # finish the thread
            self.audioThread = None
        
        # note if the audio was paused
        audio_was_paused = True if self.state() == QAudio.State.SuspendedState else False

        # do the reset
        self.reset()
        
        # Now if we were paused we resume. We couldn't do this before the reset, or it would play a short sound.
        if audio_was_paused:
            self.audioThreadPaused = False
            self.resume()

        if self.useBar:
            self.NotifyTimer.stop()

    @pyqtSlot()
    def playSeg(self, start, stop, speed=1.0, audiodata=None, low=None, high=None):
        # Selects the data between start-stop ms, relative to file start
        # and plays it, optionally at a different speed and after bandpassing

        self.timeoffset = max(0, start)
        start = max(0, int(start * self.audioFormat.sampleRate() // 1000))

        if audiodata is None:
            stop = min(int(stop * self.audioFormat.sampleRate() // 1000), len(self.sp.data))
            segment = self.sp.data[start:stop]
        else:
            stop = min(int(stop * self.audioFormat.sampleRate() // 1000), len(audiodata))
            segment = audiodata[start:stop]

        if low is not None:
            segment = SignalProc.bandpassFilter(segment, sampleRate=self.audioFormat.sampleRate(), start=low, end=high)

        if self.playbackSpeed != 1.0:
            segment = SignalProc.wsola(segment,self.playbackSpeed) 

        print("Play starting ",start)
        self.loadArray(segment)

    def loadArray(self, audiodata):
        # Plays the entire audiodata 
        # Gets the format, then puts the data in a buffer
        # and then starts the QAudioOutput from that buffer
        if self.audioFormat.sampleFormat() == QAudioFormat.SampleFormat.Int16:
            audiodata = audiodata.astype('int16')  
        elif self.audioFormat.sampleFormat() == QAudioFormat.SampleFormat.Int32:
            audiodata = audiodata.astype('int32')  
        elif self.audioFormat.sampleFormat() == QAudioFormat.SampleFormat.UInt8:
            audiodata = audiodata.astype('uint8')  
        else:
            print("ERROR: sampleFormat %s not supported" % self.audioFormat.sampleFormat())
        #print(type(audiodata),audiodata.dtype,self.audioFormat.sampleFormat())

        # double mono sound to get two channels -- simplifies reading
        if self.audioFormat.channelCount()==2:
            audiodata = np.column_stack((audiodata, audiodata))

        # write filtered output to a BytesIO buffer
        #self.audioBuffer = io.BytesIO()
        # NOTE: scale=None rescales using data minimum/max. This can cause clipping. Use scale="none" if this causes weird playback sound issues.
        # in particular for 8bit samples, we need more scaling:
        #if self.audioFormat.sampleFormat() == QAudioFormat.SampleFormat.UInt8:
            #scale = (audiodata.min()/2, audiodata.max()*2)
        #else:
            #scale = None

        #print(self.audioFormat.sampleRate(),sampwidth,len(audiodata))
        #wavio.write(self.audioBuffer, audiodata, self.audioFormat.sampleRate(), scale=scale, sampwidth=sampwidth)

        # copy BytesIO@write to QBuffer@read for playing
        #self.audioByteArray = QByteArray(self.audioBuffer.getvalue()[44:])
        self.audioByteArray = QByteArray(audiodata.tobytes())
        #print("QByteArray: ",self.audioByteArray.size())
        # Won't actually be closed yet
        #self.audioBuffer.close()
        #if self.InBuffer.isOpen():
            #self.InBuffer.close()
        self.InBuffer = QBuffer(self.audioByteArray)
        #print("QBuffer: ",self.InBuffer.size())
        #self.InBuffer.setBuffer(self.audioByteArray)
        self.InBuffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self.bytesWritten = 0
        sleep(0.2)
        #self.start(self.InBuffer)
        self.audioThreadLoading = True
        self.audioBuffer = self.start()
        #print("Actual size: ",self.bufferSize(), self.bytesFree())
        self.audioThread = threading.Thread(target=self.fillBuffer)
        self.audioThread.start()

        if self.useBar:
            self.NotifyTimer.start(30)
        """
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
        self.start(self.tempin)
        """

    def fillBuffer(self):
        while self.InBuffer.bytesAvailable() > 0 and self.audioThreadLoading:
            if self.bytesFree() > 0 and not self.audioThreadPaused:
                data = self.InBuffer.read(self.bytesFree())
                if data:
                    self.audioBuffer.write(data)
                    self.bytesWritten += len(data)

    #def restart(self):
        #self.InBuffer.seek(0)
        #self.start(self.InBuffer)
        #if self.useBar:
            #self.NotifyTimer.start(30)

    @pyqtSlot()
    def applyVolSlider(self, value):
        # passes UI volume nonlinearly
        value = QAudio.convertVolume(value / 100, QAudio.VolumeScale.LogarithmicVolumeScale, QAudio.VolumeScale.LinearVolumeScale)
        # value = (math.exp(value/50)-1)/(math.exp(2)-1)
        self.setVolume(value)

class FlowLayout(QLayout):
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
                QtCore.Qt.Orientation.Horizontal)

            spaceY = self.spacing() + wid.style().layoutSpacing(
                QtGui.QSizePolicy.PushButton,
                QtGui.QSizePolicy.PushButton,
                QtCore.Qt.Orientation.Vertical)

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
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        if (type=="w"):
            self.setIconPixmap(QPixmap("img/Owl_warning.png"))
        elif (type=="d"):
            self.setIcon(QMessageBox.Icon.Information)
            self.setIconPixmap(QPixmap("img/Owl_done.png"))
        elif (type=="t"):
            self.setIcon(QMessageBox.Icon.Information)
            self.setIconPixmap(QPixmap("img/Owl_thinking.png"))
        elif (type=="a"):
            # Easy way to set ABOUT text here:
            self.setIconPixmap(QPixmap("img/AviaNZ.png"))
            self.setText("The AviaNZ Program, v3.4-devel (December 2024)")
            self.setInformativeText("By Stephen Marsland, Victoria University of Wellington. With code by Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, and Giotto Frean. Input from Isabel Castro, Moira Pryde, Stuart Cockburn, Rebecca Stirnemann, Sumudu Purage, and Rebecca Huistra. \n stephen.marsland@vuw.ac.nz")
        elif (type=="o"):
            self.setIconPixmap(QPixmap("img/AviaNZ.png"))

        self.setWindowIcon(QIcon("img/Avianz.ico"))

        # by default, adding OK button. Can easily be overwritten after creating
        self.setStandardButtons(QMessageBox.StandardButton.Ok)

class PicButton(QAbstractButton):
    # Class for HumanClassify dialogs to put spectrograms on buttons
    # Also includes playback capability.
    def __init__(self, index, spec, audiodata, audioFormat, duration, unbufStart, unbufStop, lut, guides=None, guidecol=None, loop=False, parent=None, cluster=False):
        super(PicButton, self).__init__(parent)
        self.index = index
        self.mark = "green"
        self.spec = spec
        self.unbufStart = unbufStart
        self.unbufStop = unbufStop
        self.cluster = cluster
        self.setMouseTracking(True)

        self.playButton = QToolButton(self)
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.playButton.hide()

        self.mouseIn = False

        # SRM: Decided this isn't the right idea
        #self.plusButton = QToolButton(self)
        #self.plusButton.setText('+')
        #self.plusButton.hide()
        #self.plusButton.clicked.connect(self.plusSegment)

        # batmode frequency guides (in Y positions 0-1)
        self.guides = guides
        if guides is not None:
            self.guidelines = [0]*len(self.guides)
            self.guidecol = [QColor(*col) for col in guidecol]

        # check if playback possible (e.g. batmode)
        if len(audiodata)>0:
            self.noaudio = False
            self.playButton.clicked.connect(self.playImage)
        else:
            self.noaudio = True

        # setImage reads some properties from self, to allow easy update
        # when color map changes. Initialize with full colour scale,
        # then we expect to call setImage soon again to update.
        self.lut = lut
        #print(np.shape(self.spec))
        #print(np.min(self.spec), np.max(self.spec))
        self.setImage([np.min(self.spec), np.max(self.spec)])

        self.buttonClicked = False
        # if not self.cluster:
        self.clicked.connect(self.changePic)
        # fixed size
        self.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Fixed)
        self.setMinimumSize(self.im1.size())

        # playback things
        self.media_obj = ControllableAudio(None,audioFormat=audioFormat,useBar=True)
        # TODO: SRM: update -- do we want the moving bar?
        #self.media_obj.NotifyTimer.timeout.connect(self.endListener)
        self.media_obj.loop = loop
        self.audiodata = audiodata
        self.duration = duration * 1000  # in ms
        #self.NotifyTimer = QTimer(self)
        self.media_obj.NotifyTimer.timeout.connect(self.endListener)

    def setImage(self, colRange):
        # takes in a piece of spectrogram and produces a pair of images
        # colRange: list [colStart, colEnd]
        # TODO Could be smoother to separate out setLevels
        # from setImage here, so that colours could be adjusted without
        # redrawing - like in other review dialogs. But this also helps
        # to trigger repaint upon scrolling etc, esp on Macs.
        im, alpha = fn.makeARGB(self.spec, lut=self.lut, levels=colRange)
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
            if self.guides is not None:
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
                elif self.mark == "red" or self.mark == "blue":
                    painter.setOpacity(0.5)
            painter.drawImage(rect, self.im1)
            if not self.cluster:
                painter.drawLine(self.line1)
                painter.drawLine(self.line2)

            if self.guides is not None:
                for gi in range(len(self.guidelines)):
                    painter.setPen(QPen(self.guidecol[gi], 2))
                    painter.drawLine(self.guidelines[gi])

            # draw decision mark
            fontsize = int(self.im1.size().height() * 0.65)
            if self.mark == "green":
                pass
            elif self.mark == "yellow" and not self.cluster:
                painter.setOpacity(0.9)
                painter.setPen(QPen(QColor(220,220,0)))
                painter.setFont(QFont("Helvetica", fontsize))
                painter.drawText(rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, "?")
            elif self.mark == "yellow" and self.cluster:
                painter.setOpacity(0.9)
                painter.setPen(QPen(QColor(220, 220, 0)))
                painter.setFont(QFont("Helvetica", fontsize))
                painter.drawText(rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, "\u2713")
                #painter.drawText(rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, "√")
            elif self.mark == "blue":
                painter.setOpacity(0.9)
                painter.setPen(QPen(QColor(0,0,220)))
                painter.setFont(QFont("Helvetica", fontsize))
                painter.drawText(rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, "+")
            elif self.mark == "red":
                painter.setOpacity(0.8)
                painter.setPen(QPen(QColor(220,0,0)))
                painter.setFont(QFont("Helvetica", fontsize))
                painter.drawText(rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, "\u274C")
                #painter.drawText(rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, "X")
            else:
                print("ERROR: unrecognised segment mark")
                return

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
            self.media_obj.playSeg(0,self.duration*1000,audiodata=self.audiodata)
            #self.NotifyTimer.start(30)
            #self.media_obj.loadArray(self.audiodata)

    def endListener(self):
        timeel = self.media_obj.elapsedUSecs() // 1000
        if timeel > self.duration:
            if self.media_obj.loop:
                self.media_obj.pressedStop()
                self.media_obj.loadArray(self.audiodata)
            else:
                self.stopPlayback()
                if not self.mouseIn:
                    self.playButton.hide()

    def stopPlayback(self):
        self.media_obj.pressedStop()
        #self.NotifyTimer.stop()
        self.playButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

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
                self.mark = "blue"
            elif self.mark == "blue":
                self.mark = "green"
        self.paintEvent(ev)
        #self.update()
        self.repaint()
        QApplication.processEvents()


class Layout(pg.LayoutWidget):
    # Layout for the clustering that allows drag and drop
    buttonDragged = QtCore.pyqtSignal(int,object)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, ev):
        ev.accept()

    def dropEvent(self, ev):
        self.buttonDragged.emit(ev.position().y(),ev.source())
        ev.setDropAction(Qt.DropAction.MoveAction)
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
        self.setMinimumWidth(150)

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
            self.listOfFiles = QDir(soundDir).entryInfoList(['..','*.wav','*.bmp','*.flac'],filters=QDir.Filter.AllDirs | QDir.Filter.NoDot | QDir.Filter.Files,sort=QDir.SortFlag.DirsFirst)
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
                        numflacs = 0
                        for root, dirs, files in os.walk(file.filePath()):
                            numwavs += sum(f.lower().endswith('.wav') for f in files)
                            numbmps += sum(f.lower().endswith('.bmp') for f in files)
                            numflacs += sum(f.lower().endswith('.flac') for f in files)
                        # keep these strings as short as possible
                        if numbmps==0:
                            item.setText("%s/\t\t(%d wav files)" % (file.fileName(), numwavs))
                        elif numwavs==0:
                            item.setText("%s/\t\t(%d bmp files)" % (file.fileName(), numbmps))
                        elif numflacs==0:
                            item.setText("%s/\t\t(%d flac files)" % (file.fileName(), numflacs))
                        else:
                            item.setText("%s/\t\t(%d wav, %d bmp, %d flac files)" % (file.fileName(), numwavs, numbmps, numflacs))
                    else:
                        item.setText(file.fileName() + "/")

                    # We still might need to walk the subfolders for sp lists and wav formats!
                    if not recursive:
                        continue
                    for root, dirs, files in os.walk(file.filePath()):
                        for filename in files:
                            filenamef = os.path.join(root, filename)
                            if filename.lower().endswith('.wav') or filename.lower().endswith('.bmp') or filename.lower().endswith('.flac'):
                                if readFmt:
                                    if filename.lower().endswith('.wav'):
                                        try:
                                            #samplerate = wavio.readFmt(filenamef)[0]
                                            #wavobj = wavio.read(filenamef, 0, 0)
                                            #samplerate = wavobj.rate
                                            info = sf.info(filenamef)
                                            samplerate = info.samplerate
                                            self.fsList.add(samplerate)
                                        except Exception as e:
                                            print("Warning: could not parse format of WAV file", filenamef)
                                            print(e)
                                    elif filename.lower().endswith('.flac'):
                                        try:
                                            info = sf.info(filenamef)
                                            samplerate = info.samplerate
                                            self.fsList.add(samplerate)
                                        except Exception as e:
                                            print("Warning: could not parse format of FLAC file", filenamef)
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
                        if fullname.lower().endswith('.wav'):
                            try:
                                #samplerate = wavio.readFmt(fullname)[0]
                                info = sf.info(fullname)
                                samplerate = info.samplerate
                                self.fsList.add(samplerate)
                            except Exception as e:
                                print("Warning: could not parse format of WAV file", fullname)
                                print(e)
                        elif fullname.lower().endswith('.flac'):
                            try:
                                info = sf.info(fullname)
                                samplerate = info.samplerate
                                self.fsList.add(samplerate)
                            except Exception as e:
                                print("Warning: could not parse format of FLAC file", fullname)
                                print(e)
                        else:
                            # For bitmaps, using hardcoded samplerate as there's no readFmt
                            self.fsList.add(176000)

        if readFmt:
            print("Found the following Fs:", self.fsList)

        # mark the current file or first row (..), if not found
        if fileName:
            # for matching dirs:
            # index = self.findItems(fileName+"\/",Qt.MatchExactly)
            index = self.findItems(fileName,Qt.MatchFlag.MatchExactly)
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
        index = self.findItems(fileName,Qt.MatchFlag.MatchExactly)
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


class MainPushButton(QPushButton):
    """ QPushButton with a standard styling """
    def __init__(self, *args, **kwargs):
        super(MainPushButton, self).__init__(*args, **kwargs)
        self.setStyleSheet("""
          MainPushButton { font-weight: bold; font-size: 14px; padding: 3px 3px 3px 7px; }
        """)
        # would like to add more stuff such as:
        #    border: 2px solid #8f8f91; background-color: #dddddd}
        #  MainPushButton:disabled { border: 2 px solid #cccccc }
        #  MainPushButton:hover { background-color: #eeeeee }
        #  MainPushButton:pressed { background-color: #cccccc }
        # But any such change overrides default drawing style entirely.
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
            self.brightSlider.setValue(brightness)
        else:
            self.brightSlider.setValue(100-brightness)
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
        # TODO -- transformation mode was 1, not sure if this right option
        labelBr.setPixmap(QPixmap('img/brightstr24.png').scaled(18, 18, transformMode=Qt.TransformationMode.SmoothTransformation))

        labelCo = QLabel()
        labelCo.setPixmap(QPixmap('img/contrstr24.png').scaled(18, 18, transformMode=Qt.TransformationMode.SmoothTransformation))

        self.volIcon = QLabel()
        self.volIcon.setPixmap(QPixmap('img/volume.png').scaled(18, 18, transformMode=Qt.TransformationMode.SmoothTransformation))
        # Layout
        if horizontal:
            labelCo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            labelBr.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.volIcon.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            box = QHBoxLayout()
            box.addSpacing(5)  # outer margin
            box.addWidget(self.volIcon)
            box.addWidget(self.volSlider)
            box.addSpacing(10)
            box.addWidget(labelBr)
            box.addWidget(self.brightSlider)
            box.addSpacing(10)
            box.addWidget(labelCo)
            box.addWidget(self.contrSlider)
            box.addStretch(3)

            box.setStretch(2,4)
            box.setStretch(5,5)  # color sliders should stretch more than vol
            box.setStretch(8,5)
        else:
            box = QGridLayout()
            box.addWidget(self.volIcon, 0, 0)
            box.addWidget(self.volSlider, 0, 1)
            box.setRowMinimumHeight(0, 30)

            box.addWidget(labelBr, 1, 0)
            box.addWidget(QLabel("Brightness"), 1, 1)
            box.addWidget(self.brightSlider, 2, 0, 1, 2)

            box.addWidget(labelCo, 3, 0)
            box.addWidget(QLabel("Contrast"), 3, 1)
            box.addWidget(self.contrSlider, 4, 0, 1, 2)

            # Could also set icon scale to 16
            labelCo.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom)
            labelBr.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom)
            self.volIcon.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            box.setColumnStretch(1,3)
            box.setHorizontalSpacing(20)
            box.setContentsMargins(0, 5, 0, 10)

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
            pos = event.position().x() if self.orientation() == Qt.Orientation.Horizontal else event.position().y()
            new_value = self.minimum() + (pos / self.width() * (self.maximum() - self.minimum())) if self.orientation() == Qt.Orientation.Horizontal else self.minimum() + ((self.height() - pos) / self.height() * (self.maximum() - self.minimum()))
            self.setValue(int(new_value))
        super().mousePressEvent(event)

class MenuStayOpen(QMenu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            action = self.activeAction()
            if action is not None:
                action.trigger()
                return True
        return False

class BirdSelectionMenu(QMenu):
    """ Custom menu which has the list of birds"""
    # signal
    addSpecies = pyqtSignal(int)
    addCallname = pyqtSignal(str,int)
    labelsUpdated = pyqtSignal(object,str,str,int)

    def __init__(self, shortBirdList, longBirdList, knownCalls, currentLabels, parent=None, unsure=False, multipleBirds=False):
        super(BirdSelectionMenu, self).__init__(parent)
        self.currentLabels = copy.deepcopy(currentLabels)
        self.parent = parent
        self.unsure = unsure
        self.multipleBirds = multipleBirds
        self.position = None
        self.unsure = unsure
        self.installEventFilter(self)
        self.createShortMenu(shortBirdList,knownCalls)
        self.createLongMenu(longBirdList,knownCalls)

    def createShortMenu(self,shortBirdList,knownCalls):
        currentSpecies = [label['species'] for label in self.currentLabels]
        for item in shortBirdList:
            if not item=="":
                if item=="Don't Know":
                    action = self.addAction("Don't Know")
                    action.triggered.connect(partial(self.birdAndCallSelected, "Don't Know", "Not Specified"))
                    if "Don't Know" in currentSpecies:
                        action.setText("\u2714 Don't Know")
                    self.addAction(action)
                else:
                    species, cert = self.parse_short_list_item(item)
                    calltypes = knownCalls[species] if species in knownCalls else []
                    calltypes = ["Not Specified"] + calltypes + ["Add"]
                    birdMenu = MenuStayOpen(species,self) if self.multipleBirds else QMenu(species,self)
                    anyChecked = False
                    for calltype in calltypes:
                        label = calltype if not self.unsure else calltype+'?'
                        action = birdMenu.addAction(label)
                        action.setCheckable(True)
                        action.triggered.connect(partial(self.birdAndCallSelected, species, calltype))
                        if not len(self.currentLabels)==0:
                            if calltype == "Not Specified" or calltype is None:
                                if species in currentSpecies and not 'calltype' in self.currentLabels[currentSpecies.index(species)]: # current call is empty
                                    action.setChecked(True)
                                    anyChecked = True
                            else:
                                if not calltype=="Add":
                                    if species in currentSpecies:
                                        if 'calltype' in self.currentLabels[currentSpecies.index(species)]:
                                            if calltype == self.currentLabels[currentSpecies.index(species)]['calltype']:
                                                action.setChecked(True)
                                                anyChecked = True
                    if anyChecked:
                        birdMenu.setTitle("\u2714 "+species)
                    self.addMenu(birdMenu)
        
    def createLongMenu(self,longBirdList,knownCalls):
        menuBirdAll = QMenu("See all",self)
        allBirdTree = {}
        if longBirdList is not None:
            for longBirdEntry in longBirdList:
                # Add ? marks if Ctrl menu is called
                if '>' in longBirdEntry:
                    speciesLevel1,speciesLevel2 = longBirdEntry.split('>')
                else:
                    speciesLevel1,speciesLevel2 = longBirdEntry, ""
                
                species = speciesLevel1 if speciesLevel2=="" else speciesLevel1 + " ("+speciesLevel2+")"
                calls = ["Not Specified"]
                if species in knownCalls:
                    calls+=knownCalls[species].copy()
                calls.append("Add")

                firstLetter = speciesLevel1[0].upper()

                if not firstLetter in allBirdTree:
                    allBirdTree[firstLetter] = {}
                
                if not speciesLevel1 in allBirdTree[firstLetter]:
                    allBirdTree[firstLetter][speciesLevel1] = {}
                
                if speciesLevel2 == "":
                    allBirdTree[firstLetter][speciesLevel1] = {None: calls}
                else:
                    if not speciesLevel2 in allBirdTree[firstLetter][speciesLevel1]:
                        allBirdTree[firstLetter][speciesLevel1][speciesLevel2] = calls

        for letter in allBirdTree:
            letterMenu = QMenu(letter,menuBirdAll)
            for speciesLevel1 in allBirdTree[letter]:
                speciesLevel1Menu = QMenu(speciesLevel1,letterMenu)
                if None in allBirdTree[letter][speciesLevel1]: # no species, go straight to call
                    for call in allBirdTree[letter][speciesLevel1][None]:
                        label = call if not self.unsure else call+'?'
                        callAction = speciesLevel1Menu.addAction(label)
                        callAction.triggered.connect(partial(self.birdAndCallSelected, speciesLevel1, call))
                else:
                    for speciesLevel2 in allBirdTree[letter][speciesLevel1]:
                        species = speciesLevel1 + " (" + speciesLevel2 + ")"
                        speciesLevel2Menu = QMenu(speciesLevel2,speciesLevel1Menu)
                        for call in allBirdTree[letter][speciesLevel1][speciesLevel2]:
                            label = call if not self.unsure else call+'?'
                            callAction = speciesLevel2Menu.addAction(label)
                            callAction.triggered.connect(partial(self.birdAndCallSelected, species, call))
                        speciesLevel1Menu.addMenu(speciesLevel2Menu)
                letterMenu.addMenu(speciesLevel1Menu)
            menuBirdAll.addMenu(letterMenu)
        new_species_action = menuBirdAll.addAction("Add")
        new_species_action.triggered.connect(partial(self.birdAndCallSelected, "Add", "Not Specified"))
        self.addMenu(menuBirdAll)

    def popup(self, pos):
        if self.position is None:
            self.position = pos
        super().popup(pos)
    
    def parse_short_list_item(self,item):
        # Determine certainty
        # Add ? marks if Ctrl menu is called
        searchForSpecies = re.search(r' \((.*?)\)$',item)
        if searchForSpecies is not None:
            species = searchForSpecies.group(1)
            genus = item.split(" ("+species+")")[0]
        else:
            # try > format
            itemSplit = item.split('>')
            if len(itemSplit)==1:
                species = None
                genus = item
            else:
                species = itemSplit[1]
                genus = itemSplit[0]
        if species is None:
            mergedSpeciesName = genus
        else:
            mergedSpeciesName = genus + " (" + species + ")"
        if self.unsure and item != "Don't Know":
            cert = 50
            mergedSpeciesName = mergedSpeciesName+'?'
        elif item == "Don't Know":
            cert = 0
        else:
            cert = 100
        return mergedSpeciesName, cert
    
    def birdAndCallSelected(self,species, callname):
        """ Collects the label for a bird from the context menu and processes it.
        Has to update the overview segments in case their colour should change.
        Copes with two level names (with a > in).
        Also handles getting the name through a message box if necessary.
        """
        if type(species) is not str:
            species = species.text()
        if species is None or species=='':
            return
        
        if species=="Don't Know":
            certainty = 0
        elif self.unsure:
            species = species[:-1]
            certainty = 50
        else:
            certainty = 100

        if species == 'Add':
            self.addSpecies.emit(certainty)
            return

        if callname == 'Add':
            self.addCallname.emit(species,certainty)
            return
        
        self.updateLabels(species,callname,certainty)
        self.updateMenu(species)
    
    def updateLabels(self,species,callname,certainty):
        currentSpecies = [label['species'] for label in self.currentLabels]
        if callname=="Not Specified": callname=None
        if species=="Don't Know":
            self.currentLabels = [{'species':"Don't Know",'certainty':certainty}]
            currentSpecies = ["Don't Know"]
        else:
            if species in currentSpecies: # remove if already in
                fullLabel = self.currentLabels[currentSpecies.index(species)]
                oldCallname = fullLabel['calltype'] if 'calltype' in fullLabel else None
                self.currentLabels.pop(currentSpecies.index(species))
                currentSpecies.pop(currentSpecies.index(species))
                if not callname==oldCallname:
                    if callname is None:
                        self.currentLabels.append({'species':species,'certainty':certainty})
                        currentSpecies.append(species)
                    else:
                        self.currentLabels.append({'species':species,'calltype':callname,'certainty':certainty})
                        currentSpecies.append(species)
                if len(self.currentLabels)==0:
                    self.currentLabels.append({'species':"Don't Know",'certainty':0})
                    currentSpecies.append("Don't Know")
            else: # add
                if not self.multipleBirds: # clear if new species and not multiple birds
                    self.currentLabels = []
                    currentSpecies = []

                if callname is None:
                    self.currentLabels.append({'species':species,'certainty':certainty})
                    currentSpecies.append(species)
                else:
                    self.currentLabels.append({'species':species,'calltype':callname,'certainty':certainty})
                    currentSpecies.append(species)

                if "Don't Know" in currentSpecies:
                    self.currentLabels.pop(currentSpecies.index("Don't Know"))
                    currentSpecies.pop(currentSpecies.index("Don't Know"))                
        
        self.labelsUpdated.emit(copy.deepcopy(self.currentLabels),species,callname,certainty)

    def updateMenu(self,species):
        for birdMenu in self.findChildren(QMenu):
            # remove current checks
            if "\u2714 " == birdMenu.title()[:2]:
                birdMenu.setTitle(birdMenu.title()[2:])
            # clear actions if we have Don't Know
            if species=="Don't Know":
                for action in birdMenu.actions():
                    action.setChecked(False)
            else:
                # go through each bird in the segment and check the right calltype, then set the title
                for label in self.currentLabels:
                    species_X = label['species']
                    calltype_X = label['calltype'] if 'calltype' in label else None
                    if birdMenu.title() == species_X or birdMenu.title() == "\u2714 "+species_X:
                        for action in birdMenu.actions():
                            calltype_X = "Not Specified" if calltype_X is None else calltype_X
                            if action.text() == calltype_X:
                                action.setChecked(True)
                            else:
                                action.setChecked(False)
                        birdMenu.setTitle("\u2714 " + birdMenu.title())

        # update the Don't Know action
        for action in self.actions():
            if action.text() == "Don't Know" and "Don't Know" in [label['species'] for label in self.currentLabels]:
                action.setText("\u2714 " + action.text())
            if action.text() == "\u2714 Don't Know" and not "Don't Know" in [label['species'] for label in self.currentLabels]:
                action.setText(action.text()[2:])

    def eventFilter(self, obj, event):
        # don't hide if an action is clicked
        if self.multipleBirds:
            if event.type() == QEvent.Type.MouseButtonPress:
                action = self.activeAction()
                if action is not None:
                    action.trigger()
                    return True
        return False

class BatSelectionMenu(QMenu):
    """ Custom menu which has the list of birds"""
    # signal
    labelsUpdated = pyqtSignal(object,str,int)

    def __init__(self, batList, currentLabels, parent=None, unsure=False, multipleBirds=False):
        super(BatSelectionMenu, self).__init__(parent)
        self.currentLabels = copy.deepcopy(currentLabels)
        self.parent = parent
        self.unsure = unsure
        self.multipleBirds = multipleBirds
        self.position = None
        self.unsure = unsure
        self.installEventFilter(self)
        self.createBatMenu(batList)

    def createBatMenu(self,batList):
        currentSpecies = [label['species'] for label in self.currentLabels]        
        for item in batList:
            species, cert = self.parse_short_list_item(item)
            bat = self.addAction(species)
            bat.setCheckable(True)
            if species in currentSpecies:
                bat.setChecked(True)
            bat.triggered.connect(partial(self.batSelected, species))
            self.addAction(bat)

    def popup(self, pos):
        if self.position is None:
            self.position = pos
        super().popup(pos)
    
    def parse_short_list_item(self,item):
        # Determine certainty
        # Add ? marks if Ctrl menu is called
        searchForSpecies = re.search(r' \((.*?)\)$',item)
        if searchForSpecies is not None:
            species = searchForSpecies.group(1)
            genus = item.split(" ("+species+")")[0]
        else:
            # try > format
            itemSplit = item.split('>')
            if len(itemSplit)==1:
                species = None
                genus = item
            else:
                species = itemSplit[1]
                genus = itemSplit[0]
        if species is None:
            mergedSpeciesName = genus
        else:
            mergedSpeciesName = genus + " (" + species + ")"
        if self.unsure and item != "Don't Know":
            cert = 50
            mergedSpeciesName = mergedSpeciesName+'?'
        elif item == "Don't Know":
            cert = 0
        else:
            cert = 100
        return mergedSpeciesName, cert
    
    def batSelected(self,species):
        """ Collects the label for a bird from the context menu and processes it.
        Has to update the overview segments in case their colour should change.
        Copes with two level names (with a > in).
        Also handles getting the name through a message box if necessary.
        """
        if type(species) is not str:
            species = species.text()
        if species is None or species=='':
            return
        
        if species=="Don't Know":
            certainty = 0
        elif self.unsure:
            species = species[:-1]
            certainty = 50
        else:
            certainty = 100
        
        self.updateLabels(species,certainty)
        self.updateMenu(species)
    
    def updateLabels(self,species,certainty):
        currentSpecies = [label['species'] for label in self.currentLabels]
        if species=="Don't Know":
            self.currentLabels = [{'species':"Don't Know",'certainty':certainty}]
            currentSpecies = ["Don't Know"]
        else:
            if species in currentSpecies: # remove if already in
                self.currentLabels.pop(currentSpecies.index(species))
                currentSpecies.pop(currentSpecies.index(species))
                if len(self.currentLabels)==0:
                    self.currentLabels.append({'species':"Don't Know",'certainty':certainty})
                    currentSpecies.append("Don't Know")
            else: # add
                if not self.multipleBirds: # clear if new species and not multiple birds
                    self.currentLabels = []
                    currentSpecies = []

                self.currentLabels.append({'species':species,'certainty':certainty})
                currentSpecies.append(species)
                
                if "Don't Know" in currentSpecies:
                    self.currentLabels.pop(currentSpecies.index("Don't Know"))
                    currentSpecies.pop(currentSpecies.index("Don't Know"))                
        
        self.labelsUpdated.emit(copy.deepcopy(self.currentLabels),species,certainty)

    def updateMenu(self,species):
        for action in self.actions():
            action.setChecked(False)
        
        if species=="Don't Know":
            for action in self.actions():
                if action.text() == species:
                    action.setChecked(True)
        else:
            currentSpecies = [label['species'] for label in self.currentLabels]
            for action in self.actions():
                for species_X in currentSpecies:
                    if action.text() == species_X:
                        action.setChecked(True)

    def eventFilter(self, obj, event):
        # don't hide if an action is clicked
        if self.multipleBirds:
            if event.type() == QEvent.Type.MouseButtonPress:
                action = self.activeAction()
                if action is not None:
                    action.trigger()
                    return True
        return False