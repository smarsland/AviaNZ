
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

from PyQt6 import QtGui
from PyQt6.QtGui import *
from PyQt6.QtWidgets import QDialog, QPushButton, QHBoxLayout, QVBoxLayout, QToolButton # listing some explicitly to make syntax checks lighter
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QSize

import pyqtgraph as pg




pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class StartScreen(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowIcon(QIcon('src/resources/images/AviaNZ.ico'))
        self.setWindowTitle('AviaNZ - Choose Task')
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowCloseButtonHint)
        self.setAutoFillBackground(False)
        self.setMinimumSize(860, 350)
        self.setStyleSheet("QDialog {background-image: url(src/resources/images/AviaNZ_SW_V2.jpg); background-repeat: no-repeat; background-color: #242021; background-position: top center;}")
        self.activateWindow()

        # #242021 for the bgcolor of that image
        # btn_style='QPushButton {background-color: #A3C1DA; color: white; font-size:20px; font-weight: bold; font-family: "Arial"}'
        btn_style=""" QAbstractButton {background-color: #242021;
                    border-color: #b2c8da; border-width:2px; border-style: outset;
                    color: white; font-size:21px; font-weight: bold; font-family: "Arial"; padding: 3px;}
                    QAbstractButton:pressed {border-style: inset;}
                    """
        b1 = QPushButton("   Manual Processing   ")
        b2 = QPushButton("     Batch Processing     ")
        b3 = QPushButton("  Review Batch Results  ")
        b1.setStyleSheet(btn_style)
        b2.setStyleSheet(btn_style)
        b3.setStyleSheet(btn_style)
        bclose = QToolButton()
        bclose.setIcon(QtGui.QIcon('src/resources/images/close.png'))
        bclose.setIconSize(QSize(40, 40))
        bclose.setToolTip("Close")
        bclose.setStyleSheet(btn_style)
        bclose.clicked.connect(self.reject)

        hboxclose = QHBoxLayout()
        hboxclose.addWidget(bclose, alignment=Qt.AlignmentFlag.AlignRight)

        hbox = QHBoxLayout()
        hbox.addStretch(5)
        hbox.addWidget(b1)
        hbox.addStretch(4)
        hbox.addWidget(b2)
        hbox.addStretch(4)
        hbox.addWidget(b3)
        hbox.addStretch(5)

        vbox = QVBoxLayout()
        vbox.addLayout(hboxclose)
        vbox.addSpacing(180)
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addStretch(1)

        self.setLayout(vbox)

        b1.clicked.connect(self.manualSeg)
        b2.clicked.connect(self.findSpecies)
        b3.clicked.connect(self.reviewSeg)

        self.task = -1

    def manualSeg(self):
        self.task = 1
        self.accept()

    def findSpecies(self):
        self.task = 2
        self.accept()

    def reviewSeg(self):
        self.task = 3
        self.accept()

    #def utilities(self):
        #self.task = 4
        #self.accept()

    def getValues(self):
        return self.task
