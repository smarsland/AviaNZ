# Support classes for the AviaNZ program
# Mostly subclassed from pyqtgraph
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.functions as fn


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        # Overwrite the axis tick code
        return [QTime().addSecs(value).toString('mm:ss') for value in values]

class ShadedROI(pg.ROI):
    def paint(self, p, opt, widget):
        #brush = QtGui.QBrush(QtGui.QColor(0, 0, 255, 50))
        if not hasattr(self, 'currentBrush'):
            self.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 50)))

        p.save()
        r = self.boundingRect()
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(fn.mkPen(None))
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

class ShadedRectROI(ShadedROI):
    def __init__(self, pos, size, centered=False, sideScalers=False, **args):
        #QtGui.QGraphicsRectItem.__init__(self, 0, 0, size[0], size[1])
        pg.ROI.__init__(self, pos, size, **args)
        if centered:
            center = [0.5, 0.5]
        else:
            center = [0, 0]

        #self.addTranslateHandle(center)
        self.addScaleHandle([1, 1], center)
        if sideScalers:
            self.addScaleHandle([1, 0.5], [center[0], 0.5])
            self.addScaleHandle([0.5, 1], [0.5, center[1]])

class DragViewBox(pg.ViewBox):
    # Normal ViewBox, but with ability to drag the segments
    sigMouseDragged = QtCore.Signal(object,object,object)

    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)

    def mouseDragEvent(self, ev):
        ## if axis is specified, event will only affect that axis.
        ev.accept()  ## we accept all buttons
        if self.state['mouseMode'] != pg.ViewBox.RectMode or ev.button() == QtCore.Qt.RightButton:
            ev.ignore()

        if ev.isFinish():  ## This is the final move in the drag; draw the actual box
            self.rbScaleBox.hide()
            self.sigMouseDragged.emit(ev.buttonDownScenePos(ev.button()),ev.scenePos(),ev.screenPos())
        else:
            ## update shape of scale box
            self.updateScaleBox(ev.buttonDownPos(), ev.pos())

    def keyPressEvent(self,ev):
        # TODO: This catches the keypresses and sends out a signal
        #print ev.key(), ev.text()
        self.emit(SIGNAL("keyPressed"),ev)

class PicButton(QAbstractButton):
    # Class for HumanClassify dialogs to put spectrograms on buttons
    def __init__(self, index, im1, im2, parent=None):
        super(PicButton, self).__init__(parent)
        self.index = index
        self.im1 = im1
        self.im2 = im2
        self.buttonClicked = False
        self.clicked.connect(self.changePic)

    def paintEvent(self, event):
        im = self.im2 if self.buttonClicked else self.im1

        if type(event) is not bool:
            painter = QPainter(self)
            #painter.drawPixmap(event.rect(), pix)
            painter.drawImage(event.rect(), im)

    def sizeHint(self):
        return self.im1.size()

    def changePic(self,event):
        self.buttonClicked = not(self.buttonClicked)
        self.paintEvent(event)
        self.update()