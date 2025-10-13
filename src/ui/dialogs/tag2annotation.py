
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
from PyQt6.QtWidgets import QLabel, QDialog, QPushButton, QLineEdit, QFileDialog, QHBoxLayout, QVBoxLayout # listing some explicitly to make syntax checks lighter
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QDir

import pyqtgraph as pg

from src.ui.components.buttons_and_controls import MainPushButton
from src.ui.components.popups import MessagePopup

pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class Tag2Annotation(QDialog):
    # Class for XML Tag to AviaNZ annotation
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Generate annotations from XML (Freebird)')
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(700)

        self.txtSession = QLineEdit()
        self.txtSession.setMinimumWidth(400)
        self.txtSession.setText('')
        self.btnBrowseSession = QPushButton("&Choose Session")
        self.btnBrowseSession.setFixedWidth(220)
        self.btnBrowseSession.clicked.connect(self.browseSession)

        #self.txtDuration = QLineEdit()
        #self.txtDuration.setMinimumWidth(400)
        #self.txtDuration.setText('')
        #lblDuration = QLabel("Duration (sec) of a recording")
        #lblDuration.setFixedWidth(220)
        #lblDuration.setAlignment(Qt.AlignCenter)

        self.btnGenerateAnnot = MainPushButton("Generate AviaNZ Annotation")

        Box = QVBoxLayout()
        Box.addWidget(QLabel())
        Box1 = QHBoxLayout()
        Box1.addWidget(self.btnBrowseSession)
        Box1.addWidget(self.txtSession)
        #Box2 = QHBoxLayout()
        #Box2.addWidget(lblDuration)
        #Box2.addWidget(self.txtDuration)
        Box.addLayout(Box1)
        #Box.addLayout(Box2)
        Box.addWidget(QLabel())
        Box.addWidget(self.btnGenerateAnnot)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        if self.txtSession.text(): # and self.txtDuration.text():
            return self.txtSession.text()
            #return [self.txtSession.text(), self.txtDuration.text()]
        else:
            msg = MessagePopup("w", "Folder ", "Need a folder of Freebird tags.")
            msg.exec()
            return []

    def browseSession(self):
        #dirName = QFileDialog.getExistingDirectory(self, 'Choose .session folder with .tag and .setting')
        d = QFileDialog(self)
        d.setFilter(QDir.Filter.AllDirs | QDir.Filter.Hidden | QDir.Filter.NoDotAndDotDot)
        d.setFileMode(QFileDialog.FileMode.Directory)
        if(d.exec()):
            dirName = d.selectedFiles()[0]
            self.txtSession.setText(dirName)
