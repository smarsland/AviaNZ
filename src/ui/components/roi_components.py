# coding=latin-1

# roi_components.py
# Region of Interest components for the AviaNZ program


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

from functools import partial
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import Qt, pyqtSignal
import pyqtgraph.functions as fn


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
                r.handleMoveFinished()
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
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.isMoving = True
        elif ev.isFinish():
            if self.translatable:
                for segment in self.parent.segments:
                    if segment[4] == self:
                        self.parent.segmentMoved(segment)
            return

        if self.translatable and self.isMoving and ev.buttons() != self.parent.MouseDrawingButton:
            snap = True if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier) else None
            #snap = True if (ev.modifiers() & QtCore.Qt.ControlModifier) else None
            newPos = self.mapToParent(ev.pos()) + self.cursorOffset
            self.translate(newPos - self.pos(), snap=snap, finish=False)


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
                if tomove.x()<=self.bounds[0] or tomove.x()>=self.bounds[1]:
                    newcenter.setX(self.startPositions[i].x() + (self.cursorOffsets[i].x()*-1))

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
                rgn[1] = self.bounds[0] + (rgn[1]-rgn[0])
                rgn[0] = self.bounds[0]
        if self.bounds[1] is not None:
            if rgn[1]>self.bounds[1]:
                rgn[0] = self.bounds[1] - (rgn[1]-rgn[0])
                rgn[1] = self.bounds[1]
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


# Apply the custom mouse drag events to PyQtGraph classes
pg.graphicsItems.ROI.Handle.mouseDragEvent = mouseDragEventFlexible
pg.graphicsItems.InfiniteLine.InfiniteLine.mouseDragEvent = mouseDragEventFlexibleLine