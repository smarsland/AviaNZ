
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
from PyQt6.QtCore import Qt

import pyqtgraph as pg

from src.ui.components.buttons_and_controls import MainPushButton
from src.ui.components.popups import MessagePopup

pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class BackupAnnotation(QDialog):
    # Backup AviaNZ annotations into another folder. Simply copies all of the .data files with directory hierarchy preserved
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Backup annotations')
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(700)

        self.txtSrc = QLineEdit()
        self.txtSrc.setMinimumWidth(400)
        self.txtSrc.setText('')
        self.btnBrowseSrc = QPushButton("&Choose Source Directory")
        self.btnBrowseSrc.setFixedWidth(220)
        self.btnBrowseSrc.clicked.connect(self.browseSrc)

        self.txtDst = QLineEdit()
        self.txtDst.setMinimumWidth(400)
        self.txtDst.setText('')
        self.btnBrowseDst = QPushButton("&Choose Destination Directory")
        self.btnBrowseDst.setFixedWidth(220)
        self.btnBrowseDst.clicked.connect(self.browseDst)

        self.btnCopyAnnot = MainPushButton("Copy Annotations")

        Box = QVBoxLayout()
        Box.addWidget(QLabel('This allows you to get a copy of your annotations while preserving the directory hierarchy, only copy the .data files.\nSelect the directory you want to backup the annotations from and create a destination directory to copy the annotations'))
        Box.addWidget(QLabel())
        Box1 = QHBoxLayout()
        Box1.addWidget(self.btnBrowseSrc)
        Box1.addWidget(self.txtSrc)
        Box2 = QHBoxLayout()
        Box2.addWidget(self.btnBrowseDst)
        Box2.addWidget(self.txtDst)
        Box.addLayout(Box1)
        Box.addLayout(Box2)
        Box.addWidget(QLabel())
        Box.addWidget(self.btnCopyAnnot)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        if self.txtSrc.text() and self.txtDst.text():
            return [self.txtSrc.text(), self.txtDst.text()]
        else:
            msg = MessagePopup("t", "Need both source and target directories", "Need both source and target directories")
            msg.exec()
            return []

    def browseSrc(self):
        dirName = QFileDialog.getExistingDirectory(self, 'Choose the source folder to backup')
        self.txtSrc.setText(dirName)

    def browseDst(self):
        dirName = QFileDialog.getExistingDirectory(self, 'Choose the destination folder')
        self.txtDst.setText(dirName)