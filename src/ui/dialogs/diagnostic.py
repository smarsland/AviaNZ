
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

from PyQt6.QtGui import *
from PyQt6.QtWidgets import QLabel, QDialog, QComboBox, QCheckBox, QPushButton, QVBoxLayout, QRadioButton, QButtonGroup # listing some explicitly to make syntax checks lighter
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt

import pyqtgraph as pg




pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class Diagnostic(QDialog):
    # Class for the diagnostic dialog box
    def __init__(self, filters, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Diagnostic Plot Options')
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.setMinimumWidth(300)
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)

        # species / filter
        self.filterLabel = QLabel("Select recogniser to use")
        self.filter = QComboBox()
        # add filter file names to combobox
        self.filter.addItems(list(filters.keys()))

        # antialiasing
        self.aaLabel = QLabel("Select antialiasing type:")
        self.aaGroup = QButtonGroup()
        aaButtons = [QRadioButton("No AA"), QRadioButton("Fast partial AA"), QRadioButton("Full AA")]
        aaButtons[0].setChecked(True)
        for a in aaButtons:
            self.aaGroup.addButton(a)

        # spec or energy plot
        # self.plotLabel = QLabel("Select plot type:")
        # self.plotGroup = QButtonGroup()
        # plotButtons = [QRadioButton("Filter band energy"), QRadioButton("Reconstructed spectrogram")]
        # plotButtons[0].setChecked(True)
        # for a in plotButtons:
        #     self.plotGroup.addButton(a)

        # mark calls on spectrogram?
        self.mark = QCheckBox("Mark calls on spectrogram")
        self.mark.setChecked(True)

        # buttons
        self.activate = QPushButton("Make plots")
        self.clear = QPushButton("Clear plots")

        # layout
        Box = QVBoxLayout()
        Box.addWidget(self.filterLabel)
        Box.addWidget(self.filter)

        Box.addWidget(self.aaLabel)
        for a in aaButtons:
            Box.addWidget(a)

        # Box.addWidget(self.plotLabel)
        # for a in plotButtons:
        #     Box.addWidget(a)

        Box.addWidget(self.mark)

        Box.addWidget(self.activate)
        Box.addWidget(self.clear)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        return [self.filter.currentText(), self.aaGroup.checkedId(), self.mark.isChecked()]
