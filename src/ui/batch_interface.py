# Version 3.4 18/12/24
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

# This contains all the GUI parts for batch running of AviaNZ.

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

from PyQt6 import QtGui
from PyQt6.QtGui import QIcon, QPixmap, QColor, QScreen
from PyQt6.QtWidgets import QMessageBox, QMainWindow, QLabel, QPlainTextEdit, QPushButton, QRadioButton, QTimeEdit, QSpinBox, QApplication, QComboBox, QLineEdit, QSlider, QListWidget, QListWidgetItem, QCheckBox, QGroupBox, QGridLayout, QHBoxLayout, QVBoxLayout, QProgressDialog, QFileDialog, QDoubleSpinBox, QFormLayout, QStyle, QAbstractItemView, QButtonGroup
from PyQt6.QtCore import Qt, QDir, QSize, QThread, QWaitCondition, QObject, QMutex, pyqtSignal, pyqtSlot

import os, platform, sys, webbrowser, re, traceback
import pyqtgraph as pg
from pyqtgraph.dockarea import Dock, DockArea

from src.core import SupportClasses
from src.core.AviaNZ_batch import AviaNZ_batchProcess, GentleExitException
from src.ui.components.popups import MessagePopup
from src.ui.components.file_list import LightedFileList
from src.ui.components.buttons_and_controls import MainPushButton
from src.ui.dialogs.export_bats import ExportBats

import webbrowser, copy

import soundfile as sf

class BatchInterface(QMainWindow):
    def __init__(self, configdir=''):
        # Allow the user to browse a folder and push a button to process that folder to find a target species
        # and sets up the window.
        super(BatchInterface, self).__init__()

        self.msgClosed = QWaitCondition()

        # read config and filters from user location
        # recogniser - filter file name without ".txt"
        # (Duplicated w/ the worker, but is needed here as well)
        self.configdir = configdir
        self.configfile = os.path.join(configdir, "AviaNZconfig.txt")
        self.ConfigLoader = SupportClasses.ConfigLoader()
        self.config = self.ConfigLoader.config(self.configfile)

        filtersDir = os.path.join(configdir, self.config['FiltersDir'])
        self.FilterDicts = self.ConfigLoader.filters(filtersDir)
        if "NZ Bats" in self.FilterDicts:
            del self.FilterDicts["NZ Bats"]

        self.dirName=''
        self.statusBar().showMessage("Select a directory to process")

        self.setWindowTitle('AviaNZ - Batch Processing')
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.createMenu()
        self.createFrame()
        self.centre()

    def createFrame(self):
        # Make the window and set its size
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.setMinimumSize(1200, 800)

        # Make the docks
        self.d_detection = Dock("Automatic Detection",size=(900, 800))
        self.d_files = Dock("File list", size=(300, 800))

        self.area.addDock(self.d_detection, 'right')
        self.area.addDock(self.d_files, 'left')

        self.w_browse = QPushButton("  Choose Folder")
        self.w_browse.setToolTip("Select a folder to process (may contain sub folders)")
        self.w_browse.setFixedSize(165, 50)
        self.w_browse.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.w_browse.setStyleSheet('QPushButton {font-weight: bold; padding: 3px 3px 3px 3px}')
        self.w_dir = QPlainTextEdit()
        self.w_dir.setFixedHeight(50)
        self.w_dir.setReadOnly(True)
        self.w_dir.setPlainText('')
        self.w_dir.setToolTip("The folder being processed")
        self.w_dir.setStyleSheet("color : #808080;")

        # SRM: TODO 
        # There will be some effort needed to tidy up the sampling rate, etc.
        # How to get the size of the filter list right?
        # Make AnySound separate? Or just in list?
        # GMF:  
        #   I am not sure about what needs tidying on the sample rate front.
        #   I can't see any issues with the size of the filter list, it seems to work with small & large lists for me. 
        #   AnySound at the moment is still in the list. It might be nice to put it seperately, but for now I don't think it is an issue. 
        self.process = QButtonGroup()
        self.process.setExclusive(True)
        self.usefilters = QRadioButton("Specify filters")
        self.process.addButton(self.usefilters)
        self.usefilters.setChecked(True)
        #self.anysound = QRadioButton("Any sound")
        #self.process.addButton(self.anysound)
        self.batfilter = QRadioButton("NZ Bats")
        self.process.addButton(self.batfilter)
        #self.anysound.clicked.connect(self.useFilters)
        self.batfilter.clicked.connect(self.useFilters)
        self.usefilters.clicked.connect(self.useFilters)
        self.hasFilters = False
        self.hasFiles = False

        self.w_speLabel1 = QLabel("Select one or more recognisers to use:")
        self.w_spe1 = QListWidget()
        #self.w_spe1.setMinimumSize(800,500)
        self.w_spe1.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

        spp = sorted(list(self.FilterDicts.keys()))
        self.w_spe1.addItems(spp)
        self.w_spe1.addItem("Any sound")
        self.w_spe1.itemClicked.connect(self.countFilters)

        self.subset = QCheckBox("Process all recordings") 
        self.subset.clicked.connect(self.showTime)
        self.subset.setChecked(True)
        self.w_timeLabel = QLabel("Select start and end times for processing")
        self.w_timeStart = QTimeEdit()
        self.w_timeStart.setDisplayFormat('hh:mm:ss')
        self.w_timeEnd = QTimeEdit()
        self.w_timeEnd.setDisplayFormat('hh:mm:ss')

        # Intermittent Sampling controls
        self.intermittent = QCheckBox("Process all of each recording")
        self.intermittent.setChecked(True)
        self.intermittent.clicked.connect(self.showIntermittent)
        self.intermittentLabel = QLabel("Specify the length and frequency of the sections to process")
        self.protocolSize = QSpinBox()
        self.protocolSize.setRange(1, 180)
        self.protocolSize.setValue(int(self.config['protocolSize']))
        self.protocolInterval = QSpinBox()
        self.protocolInterval.setRange(5, 3600)
        self.protocolInterval.setValue(int(self.config['protocolInterval']))

        #self.windfilter = QCheckBox("Perform wind filtering")
        #self.windfilter.setChecked(True)
        #self.windfilter.clicked.connect(self.showWind)
        windlabel = QLabel("Specify wind filter (or select None). Only used with chp filters.")
        self.windfilter = QComboBox()
        self.windfilter.addItems(["OLS wind filter (recommended)", "Robust wind filter (experimental, slow)", "None"])
        self.overwrite = QCheckBox("Overwrite existing annotations")
        self.overwrite.setChecked(True)

        self.mergesyllables = QCheckBox("Merge Syllables For 'Any sound'")
        self.mergesyllables.setChecked(False)
        #self.mergesyllables2 = QCheckBox("Specify merge parameters")
        #self.mergesyllables2.setChecked(False)
        self.mergesyllables.clicked.connect(self.showPost)
        self.mergesyllables.setEnabled(False)
        self.mergesyllables.hide()
        self.maxgap = QDoubleSpinBox()
        self.maxgap.setRange(0.05, 10.0)
        self.maxgap.setSingleStep(0.5)
        self.maxgap.setValue(1.0)
        self.maxgaplbl = QLabel("Maximum gap between syllables (s)")

        # Spinboxes in second scale
        self.minlen = QDoubleSpinBox()
        self.minlen.setRange(0.02, 20.0)
        self.minlen.setSingleStep(1.0)
        self.minlen.setValue(0.5)
        self.minlenlbl = QLabel("Minimum segment length (s)")

        #self.mergesegments = QCheckBox("Split Segments")
        #self.mergesegements.setChecked(True)
        self.maxlen = QDoubleSpinBox()
        self.maxlen.setRange(0.05, 120.0)
        self.maxlen.setSingleStep(2.0)
        self.maxlen.setValue(10.0)
        self.maxlenlbl = QLabel("Maximum segment length (s)")

        self.w_processButton = MainPushButton(" Process Folder")
        self.w_processButton.setIcon(QIcon(QPixmap('src/resources/images/process.png')))
        self.w_processButton.clicked.connect(self.detect)
        self.w_processButton.setFixedWidth(165)
        self.w_processButton.setEnabled(False)
        self.w_browse.clicked.connect(self.browse)

        self.d_detection.addWidget(self.w_dir, row=0, col=0, colspan=3)
        self.d_detection.addWidget(self.w_browse, row=0, col=3)
        #self.d_detection.addWidget(w_speLabel1, row=1, col=0)

        #self.warning = QLabel("Warning!\n\"Any sound\" mode will delete ALL the existing annotations\nin the selected folder")
        #self.warning.setStyleSheet('QLabel {font-size:14px; color:red;}')

        # Filter selection group
        self.boxSp = QGroupBox("")
        self.formSp = QVBoxLayout()
        self.buttonSp = QHBoxLayout()
        self.buttonSp.addWidget(self.usefilters)
        #self.buttonSp.addWidget(self.anysound)
        self.buttonSp.addWidget(self.batfilter)
        self.formSp.addLayout(self.buttonSp)
        self.formSp.addWidget(self.w_speLabel1)
        self.formSp.addWidget(self.w_spe1)
        self.boxSp.setLayout(self.formSp)
        self.d_detection.addWidget(self.boxSp, row=1, col=0, colspan=4)

        # Time Settings group
        self.d_detection.addWidget(self.subset,row=2,col=0, colspan=4)
        self.boxTime = QGroupBox()
        formTime = QGridLayout()
        formTime.addWidget(self.w_timeLabel, 0, 0, 1, 2)
        formTime.addWidget(QLabel("Start time (hh:mm:ss)"), 1, 0)
        formTime.addWidget(self.w_timeStart, 1, 1)
        formTime.addWidget(QLabel("End time (hh:mm:ss)"), 2, 0)
        formTime.addWidget(self.w_timeEnd, 2, 1)
        self.boxTime.setLayout(formTime)
        self.d_detection.addWidget(self.boxTime, row=3, col=0, colspan=4)
        self.boxTime.hide()

        # intermittent sampling group, layout
        self.d_detection.addWidget(self.intermittent,row=4,col=0, colspan=4)
        self.boxIntermittent = QGroupBox()
        #self.boxIntermit = QGroupBox("Intermittent sampling")
        formIntermit = QFormLayout()
        formIntermit.addRow("Length of window", self.protocolSize)
        formIntermit.addRow("Frequency", self.protocolInterval)
        self.boxIntermittent.setLayout(formIntermit)
        self.d_detection.addWidget(self.boxIntermittent, row=5, col=0, colspan=4)
        self.boxIntermittent.hide()

        # Post Proc checkbox group
        self.d_detection.addWidget(windlabel,row=6,col=0)
        self.d_detection.addWidget(self.windfilter,row=6,col=1, colspan=2)
        #self.d_detection.addWidget(self.windfilter2,row=6,col=2, colspan=2)
        #self.boxWind = QGroupBox()
        ##self.boxPost = QGroupBox("Post processing")
        #formWind = QGridLayout()
        #formWind.addWidget(self.w_wind, 0, 1)
        #self.boxWind.setLayout(formWind)
        #self.d_detection.addWidget(self.boxWind, row=8, col=0, colspan=4)
        #self.boxWind.hide()
        self.d_detection.addWidget(self.overwrite,row=7,col=0, colspan=2)
        self.d_detection.addWidget(self.mergesyllables,row=9,col=0, colspan=2)
        #self.d_detection.addWidget(self.mergesyllables2,row=9,col=2, colspan=2)
        self.boxPost = QGroupBox()
        #self.boxPost = QGroupBox("Post processing")
        formPost = QGridLayout()
        formPost.addWidget(self.maxgaplbl, 2, 0)
        formPost.addWidget(self.maxgap, 2, 1)
        formPost.addWidget(self.minlenlbl, 3, 0)
        formPost.addWidget(self.minlen, 3, 1)
        formPost.addWidget(self.maxlenlbl, 4, 0)
        formPost.addWidget(self.maxlen, 4, 1)
        self.boxPost.setLayout(formPost)
        self.d_detection.addWidget(self.boxPost, row=10, col=0, colspan=4)
        self.boxPost.hide()

        self.d_detection.addWidget(self.w_processButton, row=11, col=3)

        self.w_files = pg.LayoutWidget()
        self.d_files.addWidget(self.w_files)

        # List to hold the list of files
        colourNone = QColor(self.config['ColourNone'][0], self.config['ColourNone'][1], self.config['ColourNone'][2], self.config['ColourNone'][3])
        colourPossibleDark = QColor(self.config['ColourPossible'][0], self.config['ColourPossible'][1], self.config['ColourPossible'][2], 255)
        colourNamed = QColor(self.config['ColourNamed'][0], self.config['ColourNamed'][1], self.config['ColourNamed'][2], self.config['ColourNamed'][3])
        self.listFiles = LightedFileList(colourNone, colourPossibleDark, colourNamed)
        self.listFiles.itemDoubleClicked.connect(self.listLoadFile)

        # TODO: Remove?
        # GMF: I think this is actually quite nice if the user has a number of folders they want to process.
        #      Getting rid of it would make things simpler though. I guess in that case the user just has to choose another folder the usual way.
        self.w_files.addWidget(QLabel('Double click to select a folder'), row=0, col=0)
        self.w_files.addWidget(self.listFiles, row=2, col=0)

        self.d_detection.layout.setContentsMargins(20, 20, 20, 20)
        self.d_detection.layout.setSpacing(20)
        self.d_files.layout.setContentsMargins(10, 10, 10, 10)
        self.d_files.layout.setSpacing(10)
        #self.fillSpeciesBoxes()  # update the boxes to match the initial position
        self.show()

    def createMenu(self):
        """ Create the basic menu.
        """
        helpMenu = self.menuBar().addMenu("&Help")
        helpMenu.addAction("Help","Ctrl+H", self.showHelp)
        aboutMenu = self.menuBar().addMenu("&About")
        aboutMenu.addAction("About","Ctrl+A", self.showAbout)
        quitMenu = self.menuBar().addMenu("&Quit")
        quitMenu.addAction("Restart program", self.restart)
        quitMenu.addAction("Quit","Ctrl+Q", QApplication.quit)
        #helpMenu = self.menuBar().addMenu("&Help")
        #helpMenu.addAction("Help", self.showHelp,"Ctrl+H")
        #aboutMenu = self.menuBar().addMenu("&About")
        #aboutMenu.addAction("About", self.showAbout,"Ctrl+A")
        #quitMenu = self.menuBar().addMenu("&Quit")
        #quitMenu.addAction("Restart program", self.restart)
        #quitMenu.addAction("Quit", QApplication.quit, "Ctrl+Q")


    def showAbout(self):
        """ Create the About Message Box. Text is set in MessagePopup"""
        msg = MessagePopup("a", "About", ".")
        msg.exec()
        return

    def showHelp(self):
        """ Show the user manual (a pdf file)"""
        webbrowser.open_new(r'file://' + os.path.realpath('./Docs/AviaNZManual.pdf'))
        # webbrowser.open_new(r'http://avianz.net/docs/AviaNZManual.pdf')

    def restart(self):
        print("Restarting")
        QApplication.exit(1)

    def detect(self):
        # 1. Parses GUI
        # 2. Creates and starts the batch worker
        if not self.dirName:
            msg = MessagePopup("w", "Select Folder", "Please select a folder to process!")
            msg.exec()
            return(1)

        # TODO: SRM: Needs testing
        # GMF: I have tested this and it seems to work. 
        #      But I am unsure NZ Bats mode is doing what is intended.
        #      At the moment the batch worker ends up doing:
        #           self.exportToBatSearch()
        #           self.outputBatPasses()
        #           self.exportToDOCDB()
        #      Is that what we want? I figured it would just be doing the detection. 

        if self.batfilter.isChecked():
            species = "NZ Bats"
            self.w_processButton.setEnabled(True)
        else:
            selected = self.w_spe1.selectedItems()
            self.w_processButton.setEnabled(False)
            species = []
            for s in selected:
                species.append(s.text())
        print("Recognisers:", species)

        # Create the worker and move it to its thread
        # NOTE: any communication w/ batchProc from this thread
        # must be via signals, if at all necessary
        timeWindow_s = self.w_timeStart.time().hour() * 3600 + self.w_timeStart.time().minute() * 60 + self.w_timeStart.time().second()
        timeWindow_e = self.w_timeEnd.time().hour() * 3600 + self.w_timeEnd.time().minute() * 60 + self.w_timeEnd.time().second()
        self.batchProc = BatchProcessWorker(self, mode="GUI", configdir=self.configdir, sdir=self.dirName, recognisers=species, subset=self.subset.isChecked(), intermittent=not(self.intermittent.isChecked()), wind=self.windfilter.currentText(), mergeSyllables=self.mergesyllables.isChecked(), overwrite=self.overwrite.isChecked(), timeWindow_s=timeWindow_s, timeWindow_e=timeWindow_e, protocolSize=self.protocolSize.value(), protocolInterval=self.protocolInterval.value(), maxgap=self.maxgap.value(), minlen=self.minlen.value(), maxlen=self.maxlen.value())

        # NOTE: must be on self. to maintain the reference
        self.batchThread = QThread()
        self.batchProc.moveToThread(self.batchThread)
        # NOTE: any connections should be done after moveToThread
        self.batchProc.finished.connect(self.batchThread.quit)
        self.batchProc.completed.connect(self.completed_fileproc)
        self.batchProc.stopped.connect(self.stopped_fileproc)
        self.batchProc.failed.connect(self.error_fileproc)
        self.batchProc.need_msg.connect(self.check_msg)
        self.batchProc.need_clean_UI.connect(self.clean_UI)
        self.batchProc.need_update.connect(self.update_progress)
        self.batchProc.need_bat_info.connect(self.bat_survey_form)
        self.batchThread.started.connect(self.batchProc.detect)
        self.batchThread.start()  # a signal connected to batchProc.detect()

    def check_msg(self,title,text):
        msg = MessagePopup("t", title, text)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        response = msg.exec()

        if response == QMessageBox.StandardButton.Cancel:
            # a fall back basically
            self.msg_response = 2
        elif response == QMessageBox.StandardButton.No:
            # catches Esc as well
            self.msg_response = 1
        else:
            self.msg_response = 0
        # to utilize Esc, need to add another standard button, and then do:
        # msg.setEscapeButton(QMessageBox.Cancel)
        self.msgClosed.wakeAll()

    def bat_survey_form(self,operator,easting,northing,recorder):
        exportForm = ExportBats(os.path.join(self.dirName, "BatDB.csv"),operator,easting,northing,recorder)
        response = exportForm.exec()
        if response==1:
            self.batFormResults = exportForm.getValues()
        else:
            self.batFormResults = None
        # ping the batch worker that form was accepted or rejected
        self.msgClosed.wakeAll()

    def clean_UI(self,total,cnt):
        self.w_processButton.setEnabled(False)
        self.update()
        self.repaint()

        self.dlg = QProgressDialog("Analysing file %d / %d. Time remaining: ? h ?? min" % (cnt+1, total), "Cancel run", 0, total+1, self)
        self.dlg.setFixedSize(350, 100)
        self.dlg.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.dlg.setWindowTitle("AviaNZ - running Batch Analysis")
        self.dlg.setWindowFlags(self.dlg.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint ^ Qt.WindowType.WindowCloseButtonHint)
        self.dlg.canceled.connect(self.stopping_fileproc)
        # should be the default, but to make sure:
        self.dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.dlg.open()
        self.dlg.setValue(cnt)
        self.dlg.update()
        self.dlg.repaint()
        QApplication.processEvents()
        # ping the batch worker that dlg is ready
        self.msgClosed.wakeAll()

    def error_fileproc(self,e):
        # Pops an error message with string e
        self.statusBar().showMessage("Analysis stopped due to error")
        if hasattr(self, 'dlg'):
            self.dlg.setValue(self.dlg.maximum())
        msg = MessagePopup("w", "Analysis error!", e)
        msg.setStyleSheet("QMessageBox QLabel{color: #cc0000}")
        msg.exec()
        self.w_processButton.setEnabled(True)

    def completed_fileproc(self):
        # All files successfully processed
        self.statusBar().showMessage("Processed all %d files" % (self.dlg.maximum()-1))
        self.dlg.setValue(self.dlg.maximum())
        self.w_processButton.setEnabled(True)

        text = "Finished processing.\nWould you like to return to the start screen?"
        msg = MessagePopup("t", "Finished", text)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        reply = msg.exec()
        if reply==QMessageBox.StandardButton.Yes:
            QApplication.exit(1)
        else:
            return(0)

    def stopping_fileproc(self):
        # When "cancel" is pressed on the progress dialog, it hides,
        # but it may take a while for the worker thread to do the check and stop.
        # This function fills this period with Busy cursor.
        self.dlg.show()
        self.dlg.setLabelText("Stopping...")
        self.statusBar().showMessage("Stopping...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def stopped_fileproc(self):
        # Processing gently stopped (worker thread has now halted, and UI can continue).
        # Process any earlier requests, in particular the "stopping" signal:
        # NOTE: this might still lead to race condition as the "stopping" and "stopped" are
        # emitted by two different threads. Might need to re-emit self.dlg.canceled, or bloody sleep here.
        QApplication.processEvents()
        self.statusBar().showMessage("Analysis cancelled")
        if hasattr(self, 'dlg'):
            self.dlg.hide()
        self.w_processButton.setEnabled(True)
        # in case there was a busy cursor
        try:
            QApplication.restoreOverrideCursor()
        except Exception:
            pass

    def update_progress(self,cnt,progrtext):
        self.dlg.setValue(cnt)
        self.dlg.setLabelText(progrtext)
        self.statusBar().showMessage(progrtext)
        self.dlg.update()

    def centre(self):
        # Geometry of the main window
        qr = self.frameGeometry()
        # Centre point of screen
        cp = self.screen().availableGeometry().center()
        # Move rectangle's centre point to screen's centre point
        qr.moveCenter(cp)
        # Top left of rectangle becomes top left of window centring it
        self.move(qr.topLeft())

    def browse(self):
        if self.dirName:
            self.dirName = QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirName))
        else:
            self.dirName = QFileDialog.getExistingDirectory(self,'Choose Folder to Process')
        self.w_dir.setPlainText(self.dirName)
        self.w_dir.setReadOnly(True)

        # Populate file list and update rest of interface:
        if self.fillFileList()==0 and self.hasFilters:
            self.statusBar().showMessage("Ready for processing")
            self.w_processButton.setEnabled(True)
        elif self.hasFilters:
            self.statusBar().showMessage("Select a directory to process")
            self.w_processButton.setEnabled(False)
        else: 
            self.statusBar().showMessage("Select filters to use")
            self.w_processButton.setEnabled(False)

    def useFilters(self):
        """Enable or disable selection of filters"""
        if self.usefilters.isChecked():
            self.w_speLabel1.setStyleSheet("color: black")
            self.w_spe1.setEnabled(True)
        else:
            # Bats 
            self.w_speLabel1.setStyleSheet("color: gray")
            for i in range(self.w_spe1.count()):
                it = self.w_spe1.item(i)
                it.setSelected(False)
            self.w_spe1.setDisabled(True)
        self.countFilters()

    def countFilters(self):
        """ Update process message and buttons based on whether filters and files are selected"""
        if len(self.w_spe1.selectedItems()) > 0 or self.batfilter.isChecked(): # or self.anysound.isChecked():
            self.hasFilters = True
        else:
            self.hasFilters = False
        
        if 'Any sound' in [x.text() for x in self.w_spe1.selectedItems()]:
            self.mergesyllables.show()
            if not self.mergesyllables.isEnabled():
                self.mergesyllables.setEnabled(True)
                self.mergesyllables.setChecked(True)
                self.boxPost.show()
        else:
            self.mergesyllables.hide()
            self.mergesyllables.setEnabled(False)
            self.mergesyllables.setChecked(False)
            self.boxPost.hide()

        if self.hasFiles and self.hasFilters:
            self.statusBar().showMessage("Ready for processing")
            self.w_processButton.setEnabled(True)
        elif self.hasFilters:
            self.statusBar().showMessage("Select a directory to process")
            self.w_processButton.setEnabled(False)
        else: 
            self.statusBar().showMessage("Select filters to use")
            self.w_processButton.setEnabled(False)

    def showTime(self):
        if self.subset.isChecked():
            self.boxTime.hide()
        else:
            self.boxTime.show()

    def showIntermittent(self):
        if self.intermittent.isChecked():
            self.boxIntermittent.hide()
        else:
            self.boxIntermittent.show()

    #def showWind(self):
        #if self.windfilter.isChecked():
            #self.boxWind.show()
        #else:
            #self.boxWind.hide()

    def showPost(self):
        if self.mergesyllables.isChecked():
            self.boxPost.show()
        else:
            self.boxPost.hide()

    def fillFileList(self, fileName=None):
        """ Populates the list of files for the file listbox.
            Returns an error code if the specified directory is bad.
        """
        if not os.path.isdir(self.dirName):
            print("ERROR: directory %s doesn't exist" % self.dirName)
            self.listFiles.clear()
            return(1)

        self.listFiles.fill(self.dirName, fileName)

        # update the "Browse" field text
        self.w_dir.setPlainText(self.dirName)
        self.hasFiles = True
        return(0)

    def listLoadFile(self,current):
        """ Listener for when the user clicks on an item in filelist """

        # Need name of file
        if type(current) is QListWidgetItem:
            current = current.text()
            current = re.sub(r'\/.*', '', current)

        self.previousFile = current

        # Update the file list to show the right one
        i=0
        lof = self.listFiles.listOfFiles
        while i<len(lof)-1 and lof[i].fileName() != current:
            i+=1
        if lof[i].isDir() or (i == len(lof)-1 and lof[i].fileName() != current):
            dir = QDir(self.dirName)
            dir.cd(lof[i].fileName())
            # Now repopulate the listbox
            self.dirName=str(dir.absolutePath())
            self.previousFile = None
            self.fillFileList(current)
            # Show the selected file
            index = self.listFiles.findItems(os.path.basename(current), Qt.MatchFlag.MatchExactly)
            if len(index) > 0:
                self.listFiles.setCurrentItem(index[0])
        return(0)
    
    def closeEvent(self, event=None):
        """ Catch the user closing the window by clicking the Close button or otherwise."""
        print("Quitting")
        QApplication.exit(0)

class BatchProcessWorker(AviaNZ_batchProcess, QObject):
    # adds QObject functionality to standard batchProc,
    # so that it could be moved to a separate thread when multithreading.
    finished = pyqtSignal()
    completed = pyqtSignal()
    stopped = pyqtSignal()
    failed = pyqtSignal(str)
    need_msg = pyqtSignal(str, str)
    need_clean_UI = pyqtSignal(int, int)
    need_update = pyqtSignal(int, str)
    need_bat_info = pyqtSignal(str, str, str, str)

    def __init__(self, *args, **kwargs):
        # this is supposedly not OK if somebody was to ever
        # further multiply-inherit this class.
        AviaNZ_batchProcess.__init__(self, *args, **kwargs)
        QObject.__init__(self)
        self.mutex = QMutex()

    @pyqtSlot()
    def detect(self):
        try:
            AviaNZ_batchProcess.detect(self)
            self.completed.emit()
        except GentleExitException:
            # for clean exits, such as stops via progress dialog
            self.stopped.emit()
        except Exception as e:
            # we have UI, so just cleanly present the error;
            # in other modes this will CTD
            e = "Encountered error:\n" + traceback.format_exc()
            self.failed.emit(e)
        self.finished.emit()  # this is to prompt generic actions like stopping the event loop
