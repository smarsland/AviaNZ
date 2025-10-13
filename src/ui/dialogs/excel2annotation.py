
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

import os

from PyQt6.QtGui import *
from PyQt6.QtWidgets import QLabel, QDialog, QComboBox, QPushButton, QLineEdit, QFileDialog, QHBoxLayout, QVBoxLayout # listing some explicitly to make syntax checks lighter
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt

import pyqtgraph as pg

from src.ui.components.buttons_and_controls import MainPushButton
from src.ui.components.popups import MessagePopup
import openpyxl

pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class Excel2Annotation(QDialog):
    # Class for Excel to AviaNZ annotation
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Generate annotations from Excel')
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(700)

        self.txtExcel = QLineEdit()
        self.txtExcel.setMinimumWidth(400)
        self.txtExcel.setText('')
        self.btnBrowseExcel = QPushButton("&Choose Excel file")
        self.btnBrowseExcel.setFixedWidth(220)
        self.btnBrowseExcel.clicked.connect(self.browseExcel)

        lblHeader = QLabel('Choose Columns')
        lblHeader.setFixedWidth(220)
        lblHeader.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.comboStart = QComboBox()
        self.comboEnd = QComboBox()
        self.comboLow = QComboBox()
        self.comboHigh = QComboBox()

        self.txtAudio = QLineEdit()
        self.txtAudio.setMinimumWidth(400)
        self.txtAudio.setText('')
        self.btnBrowseAudio = QPushButton("Choose Corresponding Audio")
        self.btnBrowseAudio.setFixedWidth(220)
        self.btnBrowseAudio.setToolTip("Select corresponding .wav")
        self.btnBrowseAudio.clicked.connect(self.browseAudio)

        self.txtSpecies = QLineEdit()
        self.txtSpecies.setMinimumWidth(400)
        self.txtSpecies.setText('')
        lblSpecies = QLabel("Species Name")
        lblSpecies.setFixedWidth(220)
        lblSpecies.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.txtCalltype = QLineEdit()
        self.txtCalltype.setMinimumWidth(400)
        self.txtCalltype.setText('')
        lblCalltype = QLabel("Calltype (optional)")
        lblCalltype.setFixedWidth(220)
        lblCalltype.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btnGenerateAnnot = MainPushButton("Generate AviaNZ Annotation")

        # Show a template
        tableWidget = QTableWidget()
        tableWidget.setRowCount(4)
        tableWidget.setColumnCount(4)
        tableWidget.setHorizontalHeaderLabels("A;B;C;D".split(";"))
        tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tableWidget.setItem(0, 0, QTableWidgetItem("Start time (sec)"))
        tableWidget.setItem(0, 1, QTableWidgetItem("End time (sec)"))
        tableWidget.setItem(0, 2, QTableWidgetItem("Lower frequency (Hz)"))
        tableWidget.setItem(0, 3, QTableWidgetItem("Upper frequency (Hz)"))
        tableWidget.setItem(1, 0, QTableWidgetItem("42.15"))
        tableWidget.setItem(1, 1, QTableWidgetItem("48.24"))
        tableWidget.setItem(1, 2, QTableWidgetItem("546.26"))
        tableWidget.setItem(1, 3, QTableWidgetItem("7492.35"))
        tableWidget.setItem(2, 0, QTableWidgetItem("88.54"))
        tableWidget.setItem(2, 1, QTableWidgetItem("95.25"))
        tableWidget.setItem(2, 2, QTableWidgetItem("550.74"))
        tableWidget.setItem(2, 3, QTableWidgetItem("7505.25"))
        tableWidget.setItem(3, 0, QTableWidgetItem("684.15"))
        tableWidget.setItem(3, 1, QTableWidgetItem("699.74"))
        tableWidget.setItem(3, 2, QTableWidgetItem("560.25"))
        tableWidget.setItem(3, 3, QTableWidgetItem("8000.30"))
        tableWidget.setMinimumWidth(700)
        tableWidget.setStyleSheet("QTableWidget { color : #808080; }")
        tableWidget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        Box = QVBoxLayout()
        Box.addWidget(QLabel('Sample Excel:'))
        Box.addWidget(tableWidget)
        Box.addWidget(QLabel())
        Box1 = QHBoxLayout()
        Box1.addWidget(self.btnBrowseExcel)
        Box1.addWidget(self.txtExcel)
        Box10 = QHBoxLayout()
        Box11 = QVBoxLayout()
        Box11.addWidget(QLabel('Start time'))
        Box11.addWidget(self.comboStart)
        Box12 = QVBoxLayout()
        Box12.addWidget(QLabel('End time'))
        Box12.addWidget(self.comboEnd)
        Box13 = QVBoxLayout()
        Box13.addWidget(QLabel('Lower frequency'))
        Box13.addWidget(self.comboLow)
        Box14 = QVBoxLayout()
        Box14.addWidget(QLabel('Higher frequency'))
        Box14.addWidget(self.comboHigh)
        Box10.addWidget(lblHeader)
        Box10.addLayout(Box11)
        Box10.addLayout(Box12)
        Box10.addLayout(Box13)
        Box10.addLayout(Box14)

        Box2 = QHBoxLayout()
        Box2.addWidget(self.btnBrowseAudio)
        Box2.addWidget(self.txtAudio)
        Box3 = QHBoxLayout()
        Box3.addWidget(lblSpecies)
        Box3.addWidget(self.txtSpecies)
        Box3.addWidget(lblCalltype)
        Box3.addWidget(self.txtCalltype)
        Box.addLayout(Box1)
        Box.addLayout(Box10)
        Box.addLayout(Box2)
        Box.addLayout(Box3)
        Box.addWidget(QLabel())
        Box.addWidget(self.btnGenerateAnnot)

        # Now put everything into the frame
        self.setLayout(Box)

    def getValues(self):
        if self.txtSpecies.text() and self.txtExcel.text() and self.txtAudio.text():
            return [self.txtExcel.text(), self.txtAudio.text(), self.txtSpecies.text(), self.txtCalltype.text(), self.headers[self.comboStart.currentIndex()], self.headers[self.comboEnd.currentIndex()], self.headers[self.comboLow.currentIndex()], self.headers[self.comboHigh.currentIndex()]]
        else:
            msg = MessagePopup("t", "All fields are required ", "All fields are required ")
            msg.exec()
            return []

    def browseExcel(self):
        try:
            if not self.txtAudio.text():
                userDir = os.path.expanduser("~")
            else:
                userDir, _ = os.path.split(self.txtAudio.text())
            excelfile, _ = QFileDialog.getOpenFileName(self, 'Open file', userDir, "Excel (*.xlsx *.xls)")
            self.txtExcel.setText(excelfile)
            self.txtExcel.setReadOnly(True)
            # Read the excel to get the headers
            book = openpyxl.load_workbook(excelfile)
            sheet = book.active
            headers = [value for value in sheet.iter_rows(min_row=1, max_row=1)]
            headers = [h for h in headers[0]]
            self.headers = [h.column for h in headers]
            values = [h.value for h in headers]
            self.comboStart.addItems(values)
            self.comboStart.setCurrentText(values[0])
            self.comboEnd.addItems(values)
            self.comboEnd.setCurrentText(values[1])
            self.comboLow.addItems(values)
            self.comboLow.setCurrentText(values[2])
            self.comboHigh.addItems(values)
            self.comboHigh.setCurrentText(values[3])
        except Exception as e:
            print("ERROR: failed with error:")
            print(e)
            return

    def browseAudio(self):
        try:
            if not self.txtExcel.text():
                userDir = os.path.expanduser("~")
            else:
                userDir, _ = os.path.split(self.txtExcel.text())
            audiofile, _ = QFileDialog.getOpenFileName(self, 'Open file', userDir, "Audio (*.wav)")
            self.txtAudio.setText(audiofile)
            self.txtAudio.setReadOnly(True)
        except Exception as e:
            print("ERROR: failed with error:")
            print(e)
            return