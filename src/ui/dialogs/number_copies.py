
# This is part of the AviaNZ interface
# Holds most of the code for the various dialog boxes

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

# Dialogs used by the AviaNZ program
# Since most of them just get user selections, they are mostly just a mess of UI things

from PyQt6.QtGui import *
from PyQt6.QtWidgets import QLabel, QDialog, QPushButton, QVBoxLayout, QSpinBox # listing some explicitly to make syntax checks lighter
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt

import pyqtgraph as pg




pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class GetNumberCopiesPlus(QDialog):
    # Select number of max copies, check for wind, rain, metadata, operator/reviewer stats
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Multiple Calls')
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(300)

        self.numCopies = QSpinBox()
        self.numCopies.setRange(0,10)
        self.numCopies.setSingleStep(1)
        self.numCopies.setValue(1)

        button = QPushButton("OK")
        button.clicked.connect(self.accept)

        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("How many calls in this segment?"))
        vbox.addWidget(self.numCopies)
        vbox.addWidget(button)

        self.setLayout(vbox)

    def getValues(self):
        return self.numCopies.value()
