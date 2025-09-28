# coding=latin-1

# axis_widgets.py
# Axis-related components for the AviaNZ program


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

import math
import pyqtgraph as pg
from PyQt6.QtWidgets import QAbstractButton, QSizePolicy
from PyQt6.QtCore import Qt, QTime, QPoint, QSize, QLine
from PyQt6.QtGui import QPainter, QPen, QColor, QFont


class TimeAxisHour(pg.AxisItem):
    # Time axis (at bottom of spectrogram)
    # Writes the time as hh:mm:ss, and can add an offset
    def __init__(self, *args, **kwargs):
        super(TimeAxisHour, self).__init__(*args, **kwargs)
        self.offset = 0
        self.setLabel('Time', units='hh:mm:ss')
        self.showMS = False
        self.range = [0, 1]  # default range

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
        self.update()
    
    def setRange(self, minimum, maximum):
        """Set the range for the time axis"""
        # Store the range for use in tick calculations
        self.range = [minimum, maximum]
        # Call the parent's setRange method to properly set the axis range
        super().setRange(minimum, maximum)
        # Force an update to redraw the axis
        self.update()
    
    def clear(self):
        """Clear the axis and reset its state"""
        self.offset = 0
        self.showMS = False
        self.range = [0, 1]  # default range
        # Clear any cached data
        if hasattr(self, '_tickStrings'):
            self._tickStrings = []
        self.update()


class TimeAxisMin(pg.AxisItem):
    # Time axis (at bottom of spectrogram)
    # Writes the time as mm:ss, and can add an offset
    def __init__(self, *args, **kwargs):
        super(TimeAxisMin, self).__init__(*args, **kwargs)
        self.offset = 0
        self.setLabel('Time', units='mm:ss.z')
        self.showMS = False
        self.range = [0, 1]  # default range

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
                vstr2 = []
                for h in vstr1:
                    hour_add = int(vs[vstr1.index(h)]//3600)
                    vstr2.append(str(hour_add) + ":" + h)
                vstr1 = vstr2
            return vstr1
        else:
            self.setLabel('Time', units='mm:ss')
            # SRM: bug? (int)
            vstr1 = [QTime(0,0,0).addSecs(int(value)).toString('mm:ss') for value in vs]
            # check if we need to add hours:
            if vs[-1]>=3600:
                vstr2 = []
                for h in vstr1:
                    hour_add = int(vs[vstr1.index(h)]//3600)
                    vstr2.append(str(hour_add) + ":" + h)
                vstr1 = vstr2
            return vstr1

    def setOffset(self,offset):
        self.offset = offset
        self.update()
    
    def setRange(self, minimum, maximum):
        """Set the range for the time axis"""
        # Store the range for use in tick calculations
        self.range = [minimum, maximum]
        # Call the parent's setRange method to properly set the axis range
        super().setRange(minimum, maximum)
        # Force an update to redraw the axis
        self.update()
    
    def clear(self):
        """Clear the axis and reset its state"""
        self.offset = 0
        self.showMS = False
        self.range = [0, 1]  # default range
        # Clear any cached data
        if hasattr(self, '_tickStrings'):
            self._tickStrings = []
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
                currFrq += (self.maxFreq-self.minFreq)/4
                tickmark.translate(0, -self.sgsize//4)
                painter.drawLine(tickmark)
                painter.drawText(int(tickmark.x2()-fontOffset), int(tickmark.y2()+1), "%.1f" % currFrq)
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
                timeFormat = "%.0f"
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
                painter.drawText(int(tickmark.x1()-fontOffset*0.7), int(tickmark.y1()+fontOffset), timeFormat % currTime)
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