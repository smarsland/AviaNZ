
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

# Layout and container widgets for the AviaNZ program

import pyqtgraph as pg
from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QLayout, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal


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

    def _doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.spacing() + wid.style().layoutSpacing(
                QtGui.QSizePolicy.ControlType.PushButton,
                QtGui.QSizePolicy.ControlType.PushButton,
                QtCore.Qt.Orientation.Horizontal)

            spaceY = self.spacing() + wid.style().layoutSpacing(
                QtGui.QSizePolicy.ControlType.PushButton,
                QtGui.QSizePolicy.ControlType.PushButton,
                QtCore.Qt.Orientation.Vertical)

            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()


class Layout(pg.LayoutWidget):
    # Layout for the clustering that allows drag and drop
    buttonDragged = QtCore.pyqtSignal(float, object)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, ev):
        ev.accept()

    def dropEvent(self, ev):
        # Get the drop position relative to this widget
        dropPos = ev.position() if hasattr(ev, 'position') else ev.posF()
        yPos = dropPos.y()
        print(f"DROP EVENT: y={yPos}, source={ev.source()}")
        self.buttonDragged.emit(yPos, ev.source())
        ev.setDropAction(Qt.DropAction.MoveAction)
        ev.accept()


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