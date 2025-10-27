
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

from PyQt6.QtGui import *
from PyQt6.QtWidgets import QLabel, QDialog, QComboBox, QPushButton, QVBoxLayout, QFormLayout, QDoubleSpinBox # listing some explicitly to make syntax checks lighter
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt

import pyqtgraph as pg

pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class ShapesDialog(QDialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Shape analysis')
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(300)

        # if len(segments) == 0:
        #     print("No segments to analyse!")
        #     return

        # self.sampleRate = sampleRate
        # self.config = config
        # self.segments = segments

        self.detectorCombo = QComboBox()
        self.detectorCombo.addItems(['stupidShaper', 'fundFreqShaper', 'instantShaper1', 'instantShaper2'])
        self.detectorCombo.currentIndexChanged.connect(self.changeBoxes)

        self.activate = QPushButton('Detect shapes in segments')

        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("Choose shape detection method"))
        vbox.addWidget(self.detectorCombo)

        #add parameter selection for instan frequency methods

        self.IF1alpha = QDoubleSpinBox()
        self.IF1alpha.setRange(0.0, 2.0)
        self.IF1alpha.setSingleStep(0.05)
        self.IF1alpha.setValue(1)

        self.IF2alpha = QDoubleSpinBox()
        self.IF2alpha.setRange(0.0, 2.0)
        self.IF2alpha.setSingleStep(0.05)
        self.IF2alpha.setValue(1)

        self.IF2beta = QDoubleSpinBox()
        self.IF2beta.setRange(0.0, 2.0)
        self.IF2beta.setSingleStep(0.05)
        self.IF2beta.setValue(1)

        self.IF1Layout = QFormLayout()
        self.IF1Layout.addRow("Alpha:", self.IF1alpha)

        self.IF2Layout = QFormLayout()
        self.IF2Layout.addRow("Alpha:", self.IF2alpha)
        self.IF2Layout.addRow("Beta:", self.IF2beta)


        vbox.addLayout(self.IF1Layout)
        vbox.addLayout(self.IF2Layout)
        vbox.addWidget(self.activate)

        # Now put everything into the frame,
        # hide and reopen the default
        for w in range(vbox.count()):
            item = vbox.itemAt(w)
            if item.widget() is not None:
                item.widget().hide()
            else:
                # it is a layout, so loop again:
                for ww in range(item.layout().count()):
                    item.layout().itemAt(ww).widget().hide()
        self.setLayout(vbox)
        self.detectorCombo.show()
        self.activate.show()

    def changeBoxes(self, method):
        # This does the hiding and showing of the options as the algorithm changes
        # hide and reopen the default
        for w in range(self.layout().count()):
            item = self.layout().itemAt(w)
            if item.widget() is not None:
                item.widget().hide()
            else:
            # it is a layout, so loop again:
                for ww in range(item.layout().count()):
                    item.layout().itemAt(ww).widget().hide()
        self.detectorCombo.show()
        self.activate.show()

        if method == "instantShaper1":
            for ww in range(self.IF1Layout.count()):
                self.IF1Layout.itemAt(ww).widget().show()

        elif method == "instantShaper2":
            for ww in range(self.IF2Layout.count()):
                self.IF2Layout.itemAt(ww).widget().show()


    def getValues(self):
        method = self.detectorCombo.currentText()
        if method=="instantShaper1":
            pars = [self.IF1alpha.value()]
        elif method=="instantShaper2":
            pars = [self.IF2alpha.value(),self.IF2beta.value()]
        else:
            pars=[]
        print('pars', pars)
        return(method, pars)
    
