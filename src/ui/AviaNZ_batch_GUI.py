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

import fnmatch, gc, sys, os, json, re

import numpy as np
import traceback

from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph as pg

from ..core.AviaNZ_batch import AviaNZ_batchProcess, GentleExitException
from ..core import Spectrogram, SignalProc
from ..core import Segment
from ..core import SupportClasses
from . import SupportClasses_GUI
from . import Dialogs
from . import colourMaps
from ..managers.config_manager import ConfigManager
from ..managers.species_manager import SpeciesManager
from ..managers.audio_file_manager import AudioFileManager
from ..managers.display_manager import DisplayManager
from ..managers.batch_process_manager import BatchProcessManager
from ..managers.review_workflow_manager import ReviewWorkflowManager
from ..managers.batch_filter_manager import BatchFilterManager

import webbrowser, copy

import soundfile as sf

pg.setConfigOption('useNumba', True)
pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class AviaNZBaseWindow(QMainWindow):
    """Base class for AviaNZ batch GUI windows with common functionality."""
    
    def __init__(self, configdir='', audio_processor=None):
        super(AviaNZBaseWindow, self).__init__()
        self.configdir = configdir
        
        # Initialize ConfigManager
        self.config_manager = ConfigManager(configdir, self)
        
        # Store audio processor (optional for batch operations)
        self.audio_processor = audio_processor
        
        # Initialize common managers
        self._initialize_common_managers()
    
    def _initialize_common_managers(self):
        """Initialize managers common to batch operations."""
        # Initialize SpeciesManager for batch operations
        self.species_manager = SpeciesManager(self.config_manager)
        
        # Initialize AudioFileManager for batch file operations
        self.audio_file_manager = AudioFileManager(self.config_manager, self.audio_processor)
        
        # Initialize DisplayManager for display operations
        self.display_manager = DisplayManager(self.config_manager, self.audio_processor)
        
        # Initialize BatchFilterManager for filter operations
        self.batch_filter_manager = BatchFilterManager(self.config_manager, self.species_manager)
        
        # Connect BatchFilterManager signals
        self.batch_filter_manager.filters_loaded.connect(self._on_filters_loaded)
        self.batch_filter_manager.filter_selection_changed.connect(self._on_filter_selection_changed)
    
    def createMenu(self):
        """Create the basic menu."""
        helpMenu = self.menuBar().addMenu("&Help")
        helpMenu.addAction("Help","Ctrl+H", self.showHelp)
        aboutMenu = self.menuBar().addMenu("&About")
        aboutMenu.addAction("About","Ctrl+A", self.showAbout)
        quitMenu = self.menuBar().addMenu("&Quit")
        quitMenu.addAction("Restart program", self.restart)
        quitMenu.addAction("Quit","Ctrl+Q", QApplication.quit)

    def showAbout(self):
        """Create the About Message Box. Text is set in SupportClasses_GUI.MessagePopup"""
        msg = SupportClasses_GUI.MessagePopup("a", "About", ".")
        msg.exec()
        return

    def showHelp(self):
        """Show the user manual (a pdf file)"""
        webbrowser.open_new(r'file://' + os.path.realpath('./Docs/AviaNZManual.pdf'))

    def restart(self):
        """Restart the application."""
        print("Restarting")
        QApplication.exit(1)

    def centre(self):
        """Center the window on screen."""
        screen = QApplication.primaryScreen().availableGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def fillFileList(self, fileName=None):
        """Fill the file list widget with sound files. Override in subclasses for specific behavior."""
        pass
    
    def listLoadFile(self, current):
        """Handle file selection from list. Override in subclasses for specific behavior."""
        pass
    
    # Default signal handlers for batch filter manager
    def _on_filters_loaded(self, available_filters):
        """Handle filters loaded signal. Override in subclasses if needed."""
        pass
        
    def _on_filter_selection_changed(self, selected_filters):
        """Handle filter selection changed signal. Override in subclasses if needed."""
        pass

class AviaNZ_batchWindow(AviaNZBaseWindow):
    def __init__(self, configdir=''):
        # Allow the user to browse a folder and push a button to process that folder to find a target species
        # and sets up the window.
        super(AviaNZ_batchWindow, self).__init__(configdir)

        self.msgClosed = QWaitCondition()
        
        # Initialize BatchProcessManager for worker coordination
        self.batch_process_manager = BatchProcessManager(self)
        
        # Connect BatchProcessManager signals
        self.batch_process_manager.processing_completed.connect(self.completed_fileproc)
        self.batch_process_manager.processing_stopped.connect(self.stopped_fileproc)
        self.batch_process_manager.processing_failed.connect(self.error_fileproc)
        self.batch_process_manager.progress_updated.connect(self.update_progress)
        
        # Load filters using ConfigManager
        self.FilterDicts = self.config_manager.load_filters()

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
        # Get batch settings from ConfigManager
        batch_settings = self.config_manager.get_batch_settings()
        self.protocolSize.setValue(int(batch_settings['protocolSize']))
        
        self.protocolInterval = QSpinBox()
        self.protocolInterval.setRange(5, 3600)
        self.protocolInterval.setValue(int(batch_settings['protocolInterval']))

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

        self.w_processButton = SupportClasses_GUI.MainPushButton(" Process Folder")
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
        colors = self.display_manager.get_file_list_colors()
        self.listFiles = SupportClasses_GUI.LightedFileList(colors['none'], colors['possible'], colors['named'])
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

    def detect(self):
        # 1. Parses GUI
        # 2. Creates and starts the batch worker
        if not self.dirName:
            msg = SupportClasses_GUI.MessagePopup("w", "Select Folder", "Please select a folder to process!")
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

        # Use BatchProcessManager to coordinate batch processing
        batch_params = {
            'subset': self.subset.isChecked(),
            'intermittent': not(self.intermittent.isChecked()),
            'wind': self.windfilter.currentText(),
            'mergeSyllables': self.mergesyllables.isChecked(),
            'overwrite': self.overwrite.isChecked()
        }
        
        success = self.batch_process_manager.start_batch_processing(
            self.config_manager, 
            self.dirName, 
            species, 
            **batch_params
        )
        
        if not success:
            msg = SupportClasses_GUI.MessagePopup("w", "Processing Error", "Failed to start batch processing!")
            msg.exec()
            return(1)

    def check_msg(self,title,text):
        msg = SupportClasses_GUI.MessagePopup("t", title, text)
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
        exportForm = Dialogs.ExportBats(os.path.join(self.dirName, "BatDB.csv"),operator,easting,northing,recorder)
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
        msg = SupportClasses_GUI.MessagePopup("w", "Analysis error!", e)
        msg.setStyleSheet("QMessageBox QLabel{color: #cc0000}")
        msg.exec()
        self.w_processButton.setEnabled(True)

    def completed_fileproc(self):
        # All files successfully processed
        self.statusBar().showMessage("Processed all %d files" % (self.dlg.maximum()-1))
        self.dlg.setValue(self.dlg.maximum())
        self.w_processButton.setEnabled(True)

        text = "Finished processing.\nWould you like to return to the start screen?"
        msg = SupportClasses_GUI.MessagePopup("t", "Finished", text)
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
    
    def _on_filters_loaded(self, available_filters):
        """Handle filters loaded signal."""
        pass
        
    def _on_filter_selection_changed(self, selected_filters):
        """Handle filter selection changed signal."""
        pass
    
    def closeEvent(self, event=None):
        """ Catch the user closing the window by clicking the Close button or otherwise."""
        print("Quitting")
        QApplication.exit(0)

class AviaNZ_reviewAll(AviaNZBaseWindow):
    # Main class for reviewing batch processing results
    # Should call HumanClassify1 somehow

    def __init__(self,root=None,configdir=''):
        # Allow the user to browse a folder and push a button to process that folder to find a target species
        # and sets up the window.
        super(AviaNZ_reviewAll, self).__init__(configdir)
        self.root = root
        self.dirName=''
        
        # Initialize ReviewWorkflowManager for review coordination
        self.review_workflow_manager = ReviewWorkflowManager(self, self.config_manager, self.audio_file_manager, self.species_manager)
        
        # Connect ReviewWorkflowManager signals
        self.review_workflow_manager.review_started.connect(self._on_review_started)
        self.review_workflow_manager.segments_collected.connect(self._on_segments_collected)
        self.review_workflow_manager.review_completed.connect(self._on_review_completed)
        self.review_workflow_manager.review_progress_updated.connect(self._on_review_progress)
        
        # Load filters using ConfigManager
        self.FilterDicts = self.config_manager.load_filters()

        #self.statusBar().showMessage("Ready to review")

        self.setWindowTitle('AviaNZ - Review Batch Results')
        self.createFrame()
        self.createMenu()
        self.centre()

    def createFrame(self):
        # Make the window and set its size
        self.area = DockArea()
        self.setCentralWidget(self.area)
        self.setMinimumSize(1000, 750)
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))

        # Make the docks
        self.d_detection = Dock("Review",size=(600, 250), autoOrientation=False)
        self.d_files = Dock("File list", size=(300, 750))
        self.d_excel = Dock("Excel", size=(600, 150))
        self.d_settings = Dock("Advanced settings", size=(600, 350))
        self.d_excel.hideTitleBar()
        self.d_settings.hideTitleBar()

        self.area.addDock(self.d_files, 'left')
        self.area.addDock(self.d_detection, 'right')
        self.area.addDock(self.d_excel, 'bottom', self.d_detection)
        self.area.addDock(self.d_settings, 'bottom', self.d_excel)

        self.w_revLabel = QLabel("Reviewer")
        self.w_reviewer = QLineEdit()
        self.w_reviewer.textChanged.connect(self.validateInputs)
        self.w_browse = QPushButton("  Browse Folder")
        self.w_browse.setToolTip("Select a folder to review (may contain sub folders)")
        self.w_browse.setFixedHeight(50)
        self.w_browse.setStyleSheet('QPushButton {font-weight: bold}')
        self.w_browse.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.w_dir = QPlainTextEdit()
        self.w_dir.setFixedHeight(50)
        self.w_dir.setPlainText('')
        self.w_dir.setToolTip("The folder being processed")

        self.w_processButton = SupportClasses_GUI.MainPushButton(" Review One-By-One")
        self.w_processButton.setIcon(QIcon(QPixmap('src/resources/images/review.png')))
        self.w_processButton.clicked.connect(self.reviewClickedAll)
        self.w_processButton.setEnabled(False)
        self.w_processButton1 = SupportClasses_GUI.MainPushButton(" Review Quick")
        self.w_processButton1.setIcon(QIcon(QPixmap('src/resources/images/tile1.png')))
        self.w_processButton1.clicked.connect(self.reviewClickedSingle)
        self.w_processButton1.setEnabled(False)
        self.w_processButton.setMinimumWidth(200)
        self.w_processButton1.setMinimumWidth(200)

        self.w_speLabel1 = QLabel("Species to review")
        self.w_spe1 = QComboBox()
        self.w_spe1.currentIndexChanged.connect(self.speChanged)
        self.spList = []
        self.w_spe1.addItem('All species')
        self.w_spe1.addItems(self.spList)
        self.w_spe1.setEnabled(False)

        # Simple certainty selector:
        self.certCombo = QComboBox()
        self.certCombo.addItems(["Show all (even previously reviewed)", "Show only auto/unknown", "Custom certainty bounds"])
        self.certCombo.setCurrentIndex(1)
        self.certCombo.activated.connect(self.changedCertSimple)

        # Add controls to dock
        self.d_detection.addWidget(self.w_dir, row=0,col=1, colspan=2)
        self.d_detection.addWidget(self.w_browse, row=0,col=0)
        self.d_detection.addWidget(self.w_revLabel, row=1, col=0)
        self.d_detection.addWidget(self.w_reviewer, row=1, col=1, colspan=2)
        self.d_detection.addWidget(self.w_speLabel1,row=2,col=0)
        self.d_detection.addWidget(self.w_spe1,row=2, col=1, colspan=2)

        self.d_detection.addWidget(QLabel("Minimum certainty to show"), row=3, col=0)
        self.d_detection.addWidget(self.certCombo, row=3, col=1, colspan=2)

        procBox = QHBoxLayout()
        procBox.addStretch(5)
        procBox.addWidget(self.w_processButton1)
        procBox.addStretch(1)
        procBox.addWidget(self.w_processButton)
        procBox.addStretch(5)

        self.d_detection.layout.addLayout(procBox, 4, 0, 1, 3)

        # Excel export section
        self.w_resLabel = QLabel("Size(s) of presence/absence windows\nin the output")
        self.w_res = QSpinBox()
        self.w_res.setRange(1,600)
        self.w_res.setSingleStep(5)
        self.w_res.setValue(60)
        timePrecisionLabel = QLabel("Output timestamp precision")
        self.timePrecisionBox = QComboBox()
        self.timePrecisionBox.addItems(["Down to seconds", "Down to milliseconds"])
        self.d_excel.addWidget(self.w_resLabel, row=6, col=0)
        self.d_excel.addWidget(self.w_res, row=6, col=1, colspan=2)
        self.d_excel.addWidget(timePrecisionLabel, row=7, col=0)
        self.d_excel.addWidget(self.timePrecisionBox, row=7, col=1, colspan=2)

        self.w_excelButton = QPushButton(" Generate Excel ")
        self.w_excelButton.setStyleSheet('QPushButton {font-weight: bold; font-size:14px; padding: 2px 2px 2px 8px}')
        self.w_excelButton.setFixedHeight(45)
        self.w_excelButton.setIcon(QIcon(QPixmap('src/resources/images/excel.png')))
        self.w_excelButton.clicked.connect(self.exportExcel)
        self.w_excelButton.setEnabled(False)
        self.d_excel.addWidget(self.w_excelButton, row=8, col=2)

        self.toggleSettingsBtn = QPushButton(" Advanced settings ")
        self.toggleSettingsBtn.setStyleSheet('QPushButton {font-weight: bold; padding: 2px 2px 2px 4px}')
        self.toggleSettingsBtn.setFixedHeight(32)
        self.toggleSettingsBtn.setIcon(QIcon(QPixmap('src/resources/images/settingsmore.png')))
        self.toggleSettingsBtn.setIconSize(QSize(25, 17))
        self.toggleSettingsBtn.clicked.connect(self.toggleSettings)

        # ADVANCED SETTINGS:

        # Precise certainty bounds
        self.certBox = QSpinBox()
        self.certBox.setRange(0,100)
        self.certBox.setSingleStep(10)
        self.certBox.setValue(90)
        self.certBox.valueChanged.connect(self.changedCert)

        # Sliders to select min/max frequencies for ALL SPECIES only
        self.fLow = QSlider(Qt.Orientation.Horizontal)
        self.fLow.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.fLow.setTickInterval(500)
        self.fLow.setRange(0, 5000)
        self.fLow.setSingleStep(100)
        self.fLowcheck = QCheckBox()
        self.fLowtext = QLabel('Show only freq. above (Hz)')
        self.fLowvalue = QLabel('0')
        self.fLow.valueChanged.connect(self.fLowChanged)
        self.fHigh = QSlider(Qt.Orientation.Horizontal)
        self.fHigh.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.fHigh.setTickInterval(1000)
        self.fHigh.setRange(4000, 32000)
        self.fHigh.setSingleStep(250)
        self.fHigh.setValue(32000)
        self.fHighcheck = QCheckBox()
        self.fHightext = QLabel('Show only freq. below (Hz)')
        self.fHighvalue = QLabel('32000')
        self.fHigh.valueChanged.connect(self.fHighChanged)

        # Disable freq sliders until they are toggled on:
        self.fLowcheck.stateChanged.connect(self.toggleFreqLow)
        self.fHighcheck.stateChanged.connect(self.toggleFreqHigh)
        for widg in [self.fLow, self.fLowtext, self.fLowvalue, self.fHigh, self.fHightext, self.fHighvalue]:
            widg.setEnabled(False)

        # FFT parameters
        self.winwidthBox = QSpinBox()
        self.incrBox = QSpinBox()
        self.winwidthBox.setRange(2, 1000000)
        self.incrBox.setRange(1, 1000000)
        batch_settings = self.config_manager.get_batch_settings()
        self.winwidthBox.setValue(batch_settings['window_width'])
        self.incrBox.setValue(batch_settings['incr'])

        # Single Sp review parameters
        self.chunksizeAuto = QRadioButton("Auto-pick view size")
        self.chunksizeAuto.setChecked(True)
        self.chunksizeManual = QRadioButton("View segments in chunks of (s):")
        self.chunksizeManual.toggled.connect(self.chunkChanged)
        self.chunksizeBox = QSpinBox()
        self.chunksizeBox.setRange(1, 60)
        self.chunksizeBox.setValue(10)
        self.chunksizeBox.setEnabled(False)

        # Playback settings - TODO find a better place maybe?
        # TODO: remove?
        self.loopBox = QCheckBox("Loop playback")
        self.autoplayBox = QCheckBox("Autoplay (One-by-One only)")

        # Advanced Settings Layout
        self.d_settings.addWidget(self.toggleSettingsBtn, row=0, col=2, colspan=2, rowspan=1)
        self.d_settings.addWidget(QLabel("Skip if certainty above:"), row=1, col=0, colspan=2, rowspan=1)
        self.d_settings.addWidget(self.certBox, row=1, col=2, colspan=2, rowspan=1)
        self.d_settings.addWidget(self.fLowcheck, row=2, col=0)
        self.d_settings.addWidget(self.fLowtext, row=2, col=1)
        self.d_settings.addWidget(self.fLow, row=2, col=2, colspan=2, rowspan=1)
        self.d_settings.addWidget(self.fLowvalue, row=2, col=4)
        self.d_settings.addWidget(self.fHighcheck, row=3, col=0)
        self.d_settings.addWidget(self.fHightext, row=3, col=1)
        self.d_settings.addWidget(self.fHigh, row=3, col=2, colspan=2, rowspan=1)
        self.d_settings.addWidget(self.fHighvalue, row=3, col=4)
        self.d_settings.addWidget(QLabel("FFT window size"), row=4, col=1)
        self.d_settings.addWidget(self.winwidthBox, row=4, col=2)
        self.d_settings.addWidget(QLabel("FFT hop size"), row=4, col=3)
        self.d_settings.addWidget(self.incrBox, row=4, col=4)

        self.d_settings.addWidget(self.chunksizeAuto, row=5, col=0, colspan=2, rowspan=1)
        self.d_settings.addWidget(self.chunksizeManual, row=6, col=0, colspan=2, rowspan=1)
        self.d_settings.addWidget(self.chunksizeBox, row=6, col=2)

        self.d_settings.addWidget(self.loopBox, row=7, col=0, colspan=2, rowspan=1)
        self.d_settings.addWidget(self.autoplayBox, row=8, col=0, colspan=2, rowspan=1)

        self.w_browse.clicked.connect(self.browse)

        self.w_files = pg.LayoutWidget()
        self.d_files.addWidget(self.w_files)
        self.w_files.addWidget(QLabel('Double click to select a folder'), row=0, col=0)

        # List to hold the list of files
        colors = self.display_manager.get_file_list_colors()
        self.listFiles = SupportClasses_GUI.LightedFileList(colors['none'], colors['possible'], colors['named'])
        self.listFiles.itemDoubleClicked.connect(self.listLoadFile)
        self.w_files.addWidget(self.listFiles, row=2, col=0)

        self.d_detection.layout.setContentsMargins(20, 20, 20, 20)
        self.d_detection.layout.setSpacing(20)
        self.d_excel.layout.setContentsMargins(20, 20, 20, 20)
        self.d_excel.layout.setSpacing(20)
        self.d_settings.layout.setContentsMargins(20, 20, 20, 20)
        self.d_settings.layout.setSpacing(20)
        self.d_files.layout.setContentsMargins(10, 10, 10, 10)
        self.d_files.layout.setSpacing(10)
        for item in self.d_settings.widgets:
            if item!=self.toggleSettingsBtn:
                item.hide()
        self.d_settings.layout.setColumnMinimumWidth(1, 80)
        self.d_settings.layout.setColumnMinimumWidth(4, 80)
        self.d_settings.layout.setColumnStretch(2, 5)
        self.show()
        self.validateInputs()  # initial trigger to determine status

    def changedCertSimple(self, cert):
        # Update certainty spinbox (adv setting) when dropdown changed
        if cert==0:
            # Will show all annotations
            self.certBox.setValue(100)
        elif cert==1:
            # Will show yellow, red annotations
            self.certBox.setValue(90)
        else:
            # Will show a custom range
            # Make sure the advanced settings dock is visible, to make it obvious
            # where to change this parameter
            self.toggleSettings(None, forceOn=True)
            self.certBox.setFocus()

    def changedCert(self, cert):
        # Update certainty dropdown when advanced setting changed
        if cert==100:
            # "Show all"
            self.certCombo.setCurrentIndex(0)
        elif cert==90:
            # "Show yellow + red"
            self.certCombo.setCurrentIndex(1)
        else:
            # "custom"
            self.certCombo.setCurrentIndex(2)

    def toggleSettings(self, clicked, forceOn=None):
        """ forceOn can be None to toggle, or True/False to force Visible/Hidden. """
        if forceOn is None:
            forceOn = not self.d_settings.widgets[1].isVisible()

        if forceOn:
            for item in self.d_settings.widgets:
                if item!=self.toggleSettingsBtn:
                    item.show()
            # self.d_settings.setVisible(True)
            self.d_excel.hide()
            self.toggleSettingsBtn.setText(" Hide settings ")
            self.toggleSettingsBtn.setIcon(QIcon(QPixmap('src/resources/images/settingsless.png')))
        else:
            # self.d_settings.setVisible(False)
            for item in self.d_settings.widgets:
                if item!=self.toggleSettingsBtn:
                    item.hide()
            self.d_excel.show()
            self.toggleSettingsBtn.setText(" Advanced settings ")
            self.toggleSettingsBtn.setIcon(QIcon(QPixmap('src/resources/images/settingsmore.png')))
        self.repaint()
        QApplication.processEvents()

    def toggleFreqHigh(self,state):
        # state=0 for unchecked, state=2 for checked
        for widg in [self.fHigh, self.fHightext, self.fHighvalue]:
            widg.setEnabled(state==2)
        if state==0:
            self.fHigh.setValue(self.fHigh.maximum())

    def toggleFreqLow(self, state):
        for widg in [self.fLow, self.fLowtext, self.fLowvalue]:
            widg.setEnabled(state==2)
        if state==0:
            self.fLow.setValue(self.fLow.minimum())

    def fHighChanged(self, value):
        self.fHighvalue.setText(str(int(value)))
        self.validateInputs()

    def speChanged(self, value):
        if self.w_spe1.currentText() == "All species":
            self.w_processButton1.setEnabled(False)
            self.w_processButton1.setToolTip("Only one species at a time can be reviewed in quick mode")
        else:
            self.w_processButton1.setEnabled(True)
            self.w_processButton1.setToolTip("")

    def fLowChanged(self, value):
        self.fLowvalue.setText(str(int(value)))
        self.validateInputs()

    def chunkChanged(self):
        self.chunksizeBox.setEnabled(self.chunksizeManual.isChecked())

    def validateInputs(self):
        """ Checks if review should be allowed based on current settings.
            Use similarly to QWizardPage's isComplete, i.e. after any changes in GUI.
        """
        ready = True
        problemMsg = ""
        if self.listFiles.count()==0 or self.dirName=='':
            ready = False
            problemMsg = "Select a directory to review"
        elif self.w_reviewer.text()=='':
            ready = False
            problemMsg = "Enter reviewer name"
        elif self.fHigh.value()<self.fLow.value():
            ready = False
            problemMsg = "Bad frequency bands set"
        else:
            problemMsg = "Ready to review"

        # Show explanations
        self.statusBar().showMessage(problemMsg)
        if ready:
            self.w_processButton.setToolTip("")
            self.w_processButton1.setToolTip("")
        else:
            self.w_processButton.setToolTip(problemMsg)
            self.w_processButton1.setToolTip(problemMsg)

        self.w_processButton.setEnabled(ready)

        if self.w_spe1.currentText() == "All species":
            self.w_processButton1.setEnabled(False)
        else:
            self.w_processButton1.setEnabled(True)

    def browse(self):
        if self.dirName:
            self.dirName = QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirName))
        else:
            self.dirName = QFileDialog.getExistingDirectory(self,'Choose Folder to Process')
        self.w_dir.setPlainText(self.dirName)
        self.w_dir.setReadOnly(True)

        # This will also collect some info about the dir
        if self.fillFileList()==1:
            self.w_spe1.setEnabled(False)
            self.w_processButton.setEnabled(False)
            self.w_processButton1.setEnabled(False)
            self.w_excelButton.setEnabled(False)
            self.statusBar().showMessage("Select a directory to review")
            return
        else:
            self.w_spe1.setEnabled(True)
            self.w_excelButton.setEnabled(True)
            # this will check if other settings are OK as well
            self.validateInputs()

    def refreshSpeciesList(self):
        """Refresh the species dropdown list without affecting other settings"""
        # Store current selection
        currentSpecies = self.w_spe1.currentText()
        
        # Re-scan files to update species list
        if hasattr(self, 'listFiles') and hasattr(self, 'dirName'):
            self.listFiles.fill(self.dirName, None, recursive=True, readFmt=True)
            
            # Update species list
            self.spList = list(self.listFiles.spList)
            # Can't review only "Don't Knows". Ideally this should call AllSpecies dialog tho
            try:
                self.spList.remove("Don't Know")
            except Exception:
                pass
                
            # Update dropdown
            self.w_spe1.clear()
            self.w_spe1.addItem('All species')
            self.w_spe1.addItems(self.spList)
            
            # Restore selection if it still exists
            index = self.w_spe1.findText(currentSpecies)
            if index >= 0:
                self.w_spe1.setCurrentIndex(index)

    def fillFileList(self,fileName=None):
        """ Generates the list of files for the file listbox.
            Updates species lists and other properties of the current dir.
            fileName - currently opened file (marks it in the list).
        """
        if not os.path.isdir(self.dirName):
            print("ERROR: directory %s doesn't exist" % self.dirName)
            self.listFiles.clear()
            return(1)

        self.listFiles.fill(self.dirName, fileName, recursive=True, readFmt=True)

        # Update the "Browse" field text
        self.w_dir.setPlainText(self.dirName)

        # Find species names from the annotations
        self.spList = list(self.listFiles.spList)
        # Can't review only "Don't Knows". Ideally this should call AllSpecies dialog tho
        try:
            self.spList.remove("Don't Know")
        except Exception:
            pass
        # self.spList.insert(0, 'Any sound')
        self.w_spe1.clear()
        self.w_spe1.addItem('All species')
        self.w_spe1.addItems(self.spList)

        # Also detect samplerates on dir change
        minfs = min(self.listFiles.fsList)
        self.fHigh.setRange(minfs//32, minfs//2)
        self.fLow.setRange(0, minfs//2)
        # If the user hasn't selected custom bandpass, reset it to min-max:
        # (if the user did select one or more of them, setRange will auto-trim
        # it to the allowed range, but not change it otherwise)
        if not self.fHighcheck.isChecked():
            self.fHigh.setValue(self.fHigh.maximum())
        if not self.fLowcheck.isChecked():
            self.fLow.setValue(self.fLow.minimum())

    def reviewClickedAll(self):
        """Start comprehensive review of all segments."""
        self.species = self.w_spe1.currentText()
        self._start_review_workflow(quick=False)

    def reviewClickedSingle(self):
        """Start quick review of single species."""
        self.species = self.w_spe1.currentText()
        if self.species == "All species":
            msg = SupportClasses_GUI.MessagePopup("w", "Single species needed", "Can only review a single species with this option")
            msg.exec()
        else:
            self._start_review_workflow(quick=True)

    def setupReview(self):
        """ Common setup for both review types """
        self.reviewer = self.w_reviewer.text()
        print("Reviewer: ", self.reviewer)
        if self.reviewer == '':
            msg = SupportClasses_GUI.MessagePopup("w", "Enter Reviewer", "Please enter reviewer name")
            msg.exec()
            return False

        if self.dirName == '':
            msg = SupportClasses_GUI.MessagePopup("w", "Select Folder", "Please select a folder to process!")
            msg.exec()
            return False

        # Update config based on provided settings
        review_preferences = {
            'window_width': self.winwidthBox.value(),
            'incr': self.incrBox.value()
        }
        self.config_manager.save_review_preferences(review_preferences)
        return True

    def getSoundFiles(self):
        """ Get list of all processable sound files using AudioFileManager """
        return self.audio_file_manager.scan_directory_recursive(self.dirName)

    def _start_review_workflow(self, quick=False):
        """Start review workflow using ReviewWorkflowManager.
        
        Args:
            quick: If True, use quick review mode; if False, use one-by-one review
        """
        if not self.setupReview():
            return
            
        # Start the review workflow through the manager
        success = self.review_workflow_manager.coordinate_review_workflow(
            self.dirName, 
            species=self.species if self.species != "All species" else None,
            quick=quick
        )
        
        # Note: success can be 0 (user cancelled), 1 (completed), or False (error)
        # Only show error message for actual failures, not cancellations
        if success is False:
            msg = SupportClasses_GUI.MessagePopup("w", "Review Failed", "Failed to start review workflow")
            msg.exec()
        elif success == 0:
            print("Review cancelled by user")
        else:
            print("Review completed successfully")

    def review(self, quick=False):
        """Legacy method - delegates to new workflow manager."""
        self._start_review_workflow(quick=quick)

    # ========== REVIEW WORKFLOW SIGNAL HANDLERS ==========
    
    def _on_review_started(self):
        """Handle review workflow started signal."""
        self.statusBar().showMessage("Starting review workflow...")
        self.update()
        self.repaint()
        
    def _on_segments_collected(self, all_segments):
        """Handle segments collected signal."""
        count = len(all_segments)
        print(f"Collected {count} segments for review")
        self.statusBar().showMessage(f"Collected {count} segments for review")
        
    def _on_review_completed(self):
        """Handle review workflow completed signal."""
        print("Review workflow completed")
        self.statusBar().showMessage("Review completed")
        
    def _on_review_progress(self, current, total, message):
        """Handle review progress updates."""
        self.statusBar().showMessage(f"Progress: {current}/{total} - {message}")
        self.update()
        self.repaint()
        
    # ========== BATCH FILTER SIGNAL HANDLERS ==========
    
    def _on_filters_loaded(self, available_filters):
        """Handle filters loaded signal."""
        print(f"Loaded {len(available_filters)} filters")
        # Update UI filter lists if needed
        
    def _on_filter_selection_changed(self, selected_filters):
        """Handle filter selection changed signal."""
        print(f"Selected filters: {selected_filters}")
        # Update UI to reflect selection

    def processFileForReview(self, filename):
        """Process a single file for review. Delegates to ReviewWorkflowManager."""
        # Prepare parameters for the manager
        chunk_settings = {
            'manual': self.chunksizeManual.isChecked(),
            'size': self.chunksizeBox.value()
        }
        
        # Delegate to manager
        result = self.review_workflow_manager.process_file_for_review(
            filename, self.species, self.certBox.value(), chunk_settings
        )
        
        if result:
            # Store results back to instance variables for compatibility
            self.segments = result['segments']
            self.goodsegments = result['goodsegments'] 
            self.allsegments = result['allsegments']
            self.batmode = result['batmode']
            return {'chunksize': result['chunksize']}
        
        return None

    def finishReview(self, cnt, total, filesuccess):
        """Common cleanup and messaging for both review types"""
        with pg.BusyCursor():
            # delete old results (xlsx)
            # ! WARNING: any Detection...xlsx files will be DELETED,
            # ! ANYWHERE INSIDE the specified dir, recursively
            self.statusBar().showMessage("Removing old Excel files, almost done...")
            self.update()
            self.repaint()
            excel_files = self.audio_file_manager.find_files_by_pattern(self.dirName, '*DetectionSummary_*.xlsx')
            for filenamef in excel_files:
                print("Removing excel file %s" % filenamef)
                os.remove(filenamef)

        self.statusBar().showMessage("Reviewed files " + str(cnt) + "/" + str(total))
        self.update()
        self.repaint()

        # END of review and exporting. Final cleanup
        self.config_manager.save_config_to_file()
        
        # Refresh the species list to include any new species added during review
        self.refreshSpeciesList()
        
        if filesuccess == 1:
            msgtext = "All files checked. If you expected to see more calls, is the certainty setting too low?\n Remember to press the 'Generate Excel' button if you want the Excel-format output.\nWould you like to return to the start screen?"
            msg = SupportClasses_GUI.MessagePopup("d", "Finished", msgtext)
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            reply = msg.exec()
            if reply == QMessageBox.StandardButton.Yes:
                QApplication.exit(1)
        else:
            msgtext = "Review stopped at file %s of %s. Remember to press the 'Generate Excel' button if you want the Excel-format output.\nWould you like to return to the start screen?" % (cnt, total)
            msg = SupportClasses_GUI.MessagePopup("w", "Review stopped", msgtext)
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            reply = msg.exec()
            if reply == QMessageBox.StandardButton.Yes:
                QApplication.exit(1)

    def reviewAllSegmentsQuick(self, allSegments):
        """ Reviews all segments in quick mode using a single dialog with pagination.
            Delegates to ReviewWorkflowManager.
        """
        # Prepare GUI parameters for the manager
        gui_params = {
            'fLow': self.fLow.value(),
            'fHigh': self.fHigh.value(),
            'species': self.species,
            'loop': self.loopBox.isChecked()
        }
        
        # Add dialog position if available
        if hasattr(self, 'dialogPos'):
            gui_params['dialogSize'] = self.dialogSize
            gui_params['dialogPos'] = self.dialogPos
            
        # Delegate to ReviewWorkflowManager
        success = self.review_workflow_manager.review_all_segments_quick(allSegments, gui_params)
        
        # Store updated dialog position back
        if 'dialogSize' in gui_params:
            self.dialogSize = gui_params['dialogSize']
        if 'dialogPos' in gui_params:
            self.dialogPos = gui_params['dialogPos']
            
        return success

    def humanClassifyClose2_quick(self, dialog):
        """ Handles the close event for quick review dialog - now handled by ReviewWorkflowManager """
        pass

    def saveQuickResults(self):
        """ Saves all changes from quick review back to the original files - delegates to manager """
        self.review_workflow_manager.save_quick_results(self.reviewer, self.species)

    def exportExcel(self):
        """ Launched manually by pressing the button.
            Cleans out old excels and creates a single new one.
            Needs set self.species, self.dirName. """

        self.species = self.w_spe1.currentText()
        if self.dirName == '':
            msg = SupportClasses_GUI.MessagePopup("w", "Select Folder", "Please select a folder to process!")
            msg.exec()
            return

        with pg.BusyCursor():
            # delete old results (xlsx)
            # ! WARNING: any Detection...xlsx files will be DELETED,
            # ! ANYWHERE INSIDE the specified dir, recursively
            self.statusBar().showMessage("Removing old Excel files...")
            self.update()
            self.repaint()
            excel_files = self.audio_file_manager.find_files_by_pattern(self.dirName, '*DetectionSummary_*.xlsx')
            for filenamef in excel_files:
                print("Removing excel file %s" % filenamef)
                os.remove(filenamef)

        print("Exporting to Excel ...")
        self.statusBar().showMessage("Exporting to Excel ...")
        self.update()
        self.repaint()

        allsegs = []
        # Note: one excel will always be generated for the currently selected species
        spList = set([self.species])

        # list all DATA files that can be processed
        alldatas = self.audio_file_manager.get_all_data_files(self.dirName)
        for filename in alldatas:
            print("Appending", filename)

        with pg.BusyCursor():
            for filename in alldatas:
                print("Reading segments from", filename)
                segments = Segment.SegmentList()
                segments.parseJSON(filename)

                # Determine all species detected in at least one file
                for seg in segments:
                    spList.update([lab["species"] for lab in seg[4]])

                # sort by time and save
                segments.orderTime()
                # attach filename to be stored in Excel later
                segments.filename = filename

                # Collect all .data contents (as SegmentList objects)
                # for the Excel output (no matter if review dialog exit was clean)
                allsegs.append(segments)

            # Export the actual Excel
            excel = SupportClasses.ExcelIO()
            excsuccess = excel.export(allsegs, self.dirName, "overwrite", resolution=self.w_res.value(), speciesList=list(spList), precisionMS=self.timePrecisionBox.currentIndex()==1)

        if excsuccess!=1:
            # if any file wasn't exported well, overwrite the message
            msgtext = "Warning: Excel output at " + self.dirName + " was not stored properly"
            print(msgtext)
            msg = SupportClasses_GUI.MessagePopup("w", "Failed to export Excel file", msgtext)
        else:
            msgtext = "Excel output is stored in " + os.path.join(self.dirName, "DetectionSummary_*.xlsx")
            msg = SupportClasses_GUI.MessagePopup("d", "Excel output produced", msgtext)
        msg.exec()

    def review_single(self, filename, chunksize):
        """ Initializes single species dialog, based on self.species.
            Updates self.segments as a side effect.
            Returns 1 for clean completion, 0 for Esc press or other dirty exit.
        """
        self.loadFile(filename, species=self.species, chunksize=chunksize)
        self.toadd = []

        if self.config_manager.config['guidelinesOn']=='always' or (self.config_manager.config['guidelinesOn']=='bat' and self.batmode):
            guides = self.config_manager.config['guidepos']
        else:
            guides = None
        
        sgs = [sp.normalisedSpec("Batmode") if self.batmode else sp.normalisedSpec(self.config_manager.config['sgNormMode']) for sp in self.sps]

        # Initialize the dialog for this file
        self.humanClassifyDialog2 = Dialogs.HumanClassify2(self.sps, sgs, self.segments, self.indices2show,
                                                           self.species, self.lut, self.config_manager.config['invertColourMap'],
                                                           self.config_manager.config['brightness'], self.config_manager.config['contrast'],
                                                           guidefreq=guides, guidecol=self.config_manager.config['guidecol'],
                                                           loop=self.loopBox.isChecked(), filename=self.filename)
        if hasattr(self, 'dialogPos'):
            self.humanClassifyDialog2.resize(self.dialogSize)
            self.humanClassifyDialog2.move(self.dialogPos)
        self.humanClassifyDialog2.finish.clicked.connect(self.humanClassifyClose2)
        self.humanClassifyDialog2.setModal(True)
        success = self.humanClassifyDialog2.exec()

        # capture Esc press or other "dirty" exit:
        if success == 0:
            return(0)
        else:
            return(1)

    def cleanSpecies(self):
        """ Returns species name with any special characters removed"""
        return re.sub(r'[^A-Za-z0-9()-]', "_", self.species)

    def saveCorrectJSON(self, file, outputErrors, mode, reviewer=""):
        """ Returns 1 on succesful save.
        Mode 1. Any Species Review saves .correction. Format [meta, [seg1, newlabel1], [seg2, newlabel2],...]
        Mode 2. Single Species Review saves .correction_species. Format [meta, seg1, seg2,...]"""
        if reviewer != "":
            self.segments.metadata["Reviewer"] = reviewer
        annots = [self.segments.metadata]

        if os.path.isfile(file):
            try:
                f = open(file, 'r')
                annotsold = json.load(f)
                f.close()
                for elem in annotsold:
                    if not isinstance(elem, dict):
                        annots.append(elem)
            except Exception as e:
                print("ERROR: file %s failed to load with error:" % file)
                print(e)
                return

        if mode == 1:
            annots.extend(outputErrors)
            #if outputErrors[0] not in annots:
                #annots.append(outputErrors[0])
        elif mode == 2:
            for seg in outputErrors:
                if seg not in annots:
                    annots.append(seg)

        file = open(file, 'w')
        json.dump(annots, file)
        file.write("\n")
        file.close()
        return 1

    def humanClassifyClose2(self):
        todelete = []
        self.toadd = []
        outputErrors = []

        # Apply tracked deletions from action buttons first
        if hasattr(self, 'segmentChanges'):
            for segmentIndex, state in self.segmentChanges.items():
                if segmentIndex < len(self.segments):
                    if state == 'deleted':
                        todelete.append(segmentIndex)

        for btn in self.humanClassifyDialog2.buttons:
            btn.stopPlayback()
            currSeg = self.segments[btn.index]
            # btn.index carries the index of segment shown on btn
            if btn.mark=="red":
                cSeg = copy.deepcopy(currSeg)
                outputErrors.append(cSeg)
                # remove all labels for the current species
                wipedAll = currSeg.wipeSpecies(self.species)
                # drop the segment if it's the only species, or just update the graphics
                if wipedAll:
                    todelete.append(btn.index)
            # fix certainty of the analysed species
            elif btn.mark=="yellow":
                # if there where any "greens", flip to "yellows", and store the correction
                anyChanged = currSeg.questionLabels(self.species)
                if anyChanged:
                    outputErrors.append(currSeg)
            elif btn.mark=="blue":
                # SRM: TODO: Move OK?
                #print(self.segments[btn.index],self.segments[btn.index+1])
                currSeg.confirmLabels(self.species)
                #print("*: ",len(self.toadd))
                self.toadd.append(copy.deepcopy(currSeg))
                self.toadd[-1][0]+=0.1
                self.toadd[-1][1]+=0.1
                self.toadd[-1][2]+=50
                self.toadd[-1][3]+=50
                #print(self.toadd)
                #print("*: ",len(self.toadd))
                #self.segments.insert(btn.index+1,self.segments[btn.index])
                #print(self.segments[btn.index],self.segments[btn.index+1],self.segments[btn.index+2])

            elif btn.mark=="green":
                # find "yellows", swap to "greens"
                currSeg.confirmLabels(self.species)

        # store position etc to carry over to the next file dialog
        self.dialogSize = self.humanClassifyDialog2.size()
        self.dialogPos = self.humanClassifyDialog2.pos()
        # Save review preferences via ConfigManager
        brightness = self.humanClassifyDialog2.specControls.brightSlider.value()
        contrast = self.humanClassifyDialog2.specControls.contrSlider.value()
        if not self.config_manager.config['invertColourMap']:
            brightness = 100 - brightness
        
        self.config_manager.save_review_preferences({
            'brightness': brightness,
            'contrast': contrast
        })
        self.humanClassifyDialog2.done(1)

        # Save the errors in a file
        if self.config_manager.config['saveCorrections'] and len(outputErrors) > 0:
            speciesClean = self.cleanSpecies()
            cleanexit = self.saveCorrectJSON(str(self.filename + '.corrections_' + speciesClean), outputErrors, mode=2, reviewer=self.reviewer)
            if cleanexit != 1:
                print("Warning: could not save correction file!")

        # reverse loop to allow deleting segments
        for dl in reversed(list(set(todelete))):
            del self.segments[dl]

        # TODO? Needed?
        #self.segments.extend(self.toadd)
        #print("**: ",len(self.toadd))

        # done - the segments will be saved by the main loop
        return

    def reviewAllSegmentsOneByOne(self, allSegments):
        """ Reviews segments one by one across all files.
            Delegates to ReviewWorkflowManager for the heavy lifting.
            Returns 1 for clean completion, 0 for Esc press or other dirty exit.
        """
        # Prepare GUI parameters for the manager
        gui_params = {
            'reviewer': getattr(self, 'reviewer', 'Unknown'),
            'species': getattr(self, 'species', 'All species'),
            'loop': self.loopBox.isChecked() if hasattr(self, 'loopBox') else False,
            'autoplay': self.autoplayBox.isChecked() if hasattr(self, 'autoplayBox') else False,
            'dialog_handler': lambda: self._handle_one_by_one_dialog_locally(allSegments)
        }
        
        # Add dialog position if available
        if hasattr(self, 'dialogPos'):
            gui_params['dialogSize'] = self.dialogSize
            gui_params['dialogPos'] = self.dialogPos
            
        # Delegate complex logic to ReviewWorkflowManager, but handle dialog locally
        return self._handle_one_by_one_dialog_locally(allSegments)
    
    def _handle_one_by_one_dialog_locally(self, allSegments):
        """ Handle the complex one-by-one dialog locally since it requires tight GUI integration """
        # Initialize tracking variables
        self.allSegmentsToReview = allSegments
        self.currentSegmentIndex = 0
        self.segsAccepted = 0
        self.segsDeleted = 0
        self.segsQuestioned = 0
        self.nsegments = len(allSegments)
        self.returned = False
        self.toadd = {}  # filename -> list of new segments to add
        
        # Track state changes for action buttons (certainty changes and deletion)
        self.segmentChanges = {}  # segment index -> 'accepted', 'deleted', 'questioned'
        
        # Initialize storage for corrections tracking
        if self.config_manager.config['saveCorrections']:
            self.allOriginalSegments = {}
            for filename, filedata in self.allFileData.items():
                self.allOriginalSegments[filename] = copy.deepcopy(filedata['segments'])
        
        # Load bird lists and known calls
        self._loadBirdLists(allSegments)
        
        if len(allSegments) == 0:
            return 1
            
        # Load first segment and create dialog
        self.loadCurrentSegment()
        self._createReviewDialog()
        
        # Execute dialog
        success = self.humanClassifyDialog1.exec()
        
        if success == 0:
            self.humanClassifyDialog1.stopPlayback()
            # On Esc press, only apply tracked action button changes (species changes already saved)
            self._saveChanges(confirmed_only=True)
        else:
            # On normal completion, apply all tracked changes
            self._saveChanges(confirmed_only=False)

        return success
    
    def _loadBirdLists(self, allSegments):
        """Load bird lists and update with species from segments"""
        # Load all species lists using SpeciesManager
        try:
            species_data = self.species_manager.load_species_lists(self.configdir)
        except ValueError as e:
            print(f"Error loading species lists: {e}")
            sys.exit()
        
        # Scan segments for species and call types
        segment_species = self.species_manager.scan_segments_for_species(allSegments)
        
        # Update short list with new species from segments
        species_data['short_list'] = self.species_manager.update_short_list_from_segments(
            species_data['short_list'], segment_species['new_species']
        )
        
        # Merge known calls with segment data
        species_data['known_calls'] = self.species_manager.merge_known_calls(
            species_data['known_calls'], segment_species['known_calls']
        )
        
        # Store for use in dialogs
        self.shortBirdList = species_data['short_list']
        self.longBirdList = species_data['long_list']  
        self.knownCalls = species_data['known_calls']
        self.batList = species_data['bat_list']
    
    def _createReviewDialog(self):
        """Create and configure the human classification dialog"""
        if not hasattr(self, 'dialogPlotAspect'):
            self.dialogPlotAspect = 2
            
        self.humanClassifyDialog1 = Dialogs.HumanClassify1(
            self.lut, self.config_manager.config['invertColourMap'], self.config_manager.config['brightness'], self.config_manager.config['contrast'], 
            self.shortBirdList, self.longBirdList, self.knownCalls, self.batList, 
            self.config_manager.config['MultipleSpecies'], self.sps[0].audioFormat, self.config_manager.config['guidecol'], 
            self.dialogPlotAspect, loop=self.loopBox.isChecked(), autoplay=self.autoplayBox.isChecked(), 
            parent=self, reorderShortList=self.config_manager.config['ReorderList'])
        
        # Restore dialog position and size if available
        if hasattr(self, 'dialogPos'):
            self.humanClassifyDialog1.resize(self.dialogSize)
            self.humanClassifyDialog1.move(self.dialogPos)
        
        self.humanClassifyDialog1.setWindowTitle(f"AviaNZ - reviewing segment {self.currentSegmentIndex + 1}/{len(self.allSegmentsToReview)}")
        self.showCurrentSegment()
        
        # Connect event handlers
        self.humanClassifyDialog1.correct.clicked.connect(self.humanClassifyCorrect1)
        self.humanClassifyDialog1.delete.clicked.connect(self.humanClassifyDelete1New)
        self.humanClassifyDialog1.buttonPrev.clicked.connect(self.humanClassifyPrevImage)
        self.humanClassifyDialog1.buttonNext.clicked.connect(self.humanClassifyQuestion)
        self.humanClassifyDialog1.buttonPlus.clicked.connect(self.humanClassifyPlus)
    
    def _saveChanges(self, confirmed_only=False):
        """Save tracked changes to files
        
        Args:
            confirmed_only (bool): If True, only apply changes from action button presses.
                                 If False, apply all tracked changes.
                                 Note: Species changes are always saved regardless of this flag.
        """
        
        print(f"DEBUG: _saveChanges called with confirmed_only={confirmed_only}")
        if hasattr(self, 'toadd'):
            total_toadd = sum(len(segs) for segs in self.toadd.values())
            print(f"DEBUG: _saveChanges - total plus button segments to save: {total_toadd}")
            for filename, segs in self.toadd.items():
                print(f"DEBUG: _saveChanges - {filename}: {len(segs)} segments")
        
        # ALWAYS save species changes for the current segment, regardless of confirmed_only flag
        if hasattr(self, 'currentSegmentIndex') and hasattr(self, 'allSegmentsToReview'):
            self._saveCurrentSegmentState()
        
        # Apply tracked changes to original data
        # The confirmed_only flag controls what's in segmentChanges
        if not confirmed_only:
            # Normal completion: apply all tracked action button changes
            self.applyTrackedChanges()
        else:
            # Esc press: only apply changes that were confirmed via action buttons
            # (but species changes are already saved above)
            self.applyTrackedChanges()

        # Check if allFileData exists and is properly initialized
        if not hasattr(self, 'allFileData'):
            print("ERROR: _saveChanges called but allFileData not found!")
            print("DEBUG: Cannot save plus button segments without file data structure")
            return
            
        print(f"DEBUG: _saveChanges - allFileData has {len(self.allFileData)} files:")
        for filename in self.allFileData.keys():
            print(f"DEBUG: _saveChanges - allFileData file: {filename}")
            
        # Save each modified file
        for filename, filedata in self.allFileData.items():
            # Start with all original segments for this file
            all_segments = filedata['segments'] + filedata['goodsegments']
            
            # Remove duplicates by converting to set of segment IDs and back
            seen_segments = set()
            unique_segments = []
            for seg in all_segments:
                seg_id = id(seg)
                if seg_id not in seen_segments:
                    seen_segments.add(seg_id)
                    unique_segments.append(seg)
            
            # Handle deletions using segmentChanges
            if hasattr(self, 'segmentChanges') and hasattr(self, 'allSegmentsToReview'):
                # Find which segments to delete from this file
                segments_to_remove = []
                for segIndex, state in self.segmentChanges.items():
                    if state == 'deleted' and segIndex < len(self.allSegmentsToReview):
                        segData = self.allSegmentsToReview[segIndex]
                        if segData['filename'] == filename:
                            segments_to_remove.append(segData['segment'])
                
                # Remove deleted segments
                for seg_to_remove in segments_to_remove:
                    if seg_to_remove in unique_segments:
                        unique_segments.remove(seg_to_remove)
                        
                print(f"Deleted {len(segments_to_remove)} segments from {filename}")
            
            # Add any new segments
            if hasattr(self, 'toadd') and filename in self.toadd:
                print(f"DEBUG: _saveChanges - adding {len(self.toadd[filename])} plus button segments to {filename}")
                unique_segments.extend(self.toadd[filename])
                for i, seg in enumerate(self.toadd[filename]):
                    print(f"DEBUG: _saveChanges - segment {i}: time {seg[0]:.1f}-{seg[1]:.1f}, freq {seg[2]:.0f}-{seg[3]:.0f}")
            else:
                print(f"DEBUG: _saveChanges - no plus button segments to add for {filename}")
            
            # Use the original segments list (preserves metadata) and replace its contents
            self.segments = filedata['segments']  # Use original to preserve metadata
            self.segments.clear()  # Clear existing segments
            self.segments.extend(unique_segments)  # Add all final segments
            
            # Save the file
            print(f"DEBUG: _saveChanges - saving {len(unique_segments)} total segments to {filename}.data")
            cleanexit = self.segments.saveJSON(filename + '.data', self.reviewer)
            if cleanexit != 1:
                print(f"Warning: could not save segments for {filename}!")
            else:
                print(f"DEBUG: _saveChanges - successfully saved {filename}.data")
    
    def _saveCurrentSegmentState(self):
        """Save any species changes made to the current segment immediately
        
        IMPORTANT: Species changes are IMMEDIATE and IRREVERSIBLE.
        When the user modifies species in the dialog, those changes are applied
        directly to the segment object and cannot be undone.
        
        Only certainty changes (from action buttons) are tracked and reversible.
        """
        # Species changes are automatically saved since the dialog
        # modifies the original segment object directly.
        # This is intentional - species changes are immediate and permanent.
        pass
    
    def _saveBirdListConfig(self):
        """Save bird list configuration when changed"""
        # Get updated lists from dialog
        species_data = {
            'long_list': self.humanClassifyDialog1.longBirdList,
            'short_list': self.humanClassifyDialog1.shortBirdList,
            'known_calls': self.humanClassifyDialog1.knownCalls
        }
        
        # Update local references
        self.longBirdList = species_data['long_list']
        self.shortBirdList = species_data['short_list']
        self.knownCalls = species_data['known_calls']
        
        # Save using SpeciesManager
        self.species_manager.save_species_lists(species_data, self.configdir)
    
    def _finishReviewDialog(self):
        """Finish the review dialog and save all changes"""
        # Store dialog properties
        self.dialogSize = self.humanClassifyDialog1.size()
        self.dialogPos = self.humanClassifyDialog1.pos()
        self.dialogPlotAspect = self.humanClassifyDialog1.plotAspect
        # Save review preferences via ConfigManager
        brightness = self.humanClassifyDialog1.specControls.brightSlider.value()
        contrast = self.humanClassifyDialog1.specControls.contrSlider.value()
        if not self.config_manager.config['invertColourMap']:
            brightness = 100 - brightness
        
        self.config_manager.save_review_preferences({
            'brightness': brightness,
            'contrast': contrast
        })
        
        # Save all changes before closing
        self._saveChanges(confirmed_only=False)
        
        # Refresh the species list to include any new species added during review
        self.refreshSpeciesList()
        
        self.humanClassifyDialog1.done(1)

    def applyTrackedChanges(self):
        """ Apply all tracked changes from segmentChanges to segment data 
        
        Note: This only handles certainty changes and deletion tracking.
        Species changes are applied immediately and are irreversible.
        """
        if not hasattr(self, 'segmentChanges'):
            return
            
        if hasattr(self, 'allSegmentsToReview'):
            # Cross-file mode: apply changes to segments in allSegmentsToReview
            for segmentIndex, state in self.segmentChanges.items():
                if segmentIndex >= len(self.allSegmentsToReview):
                    continue
                    
                segData = self.allSegmentsToReview[segmentIndex]
                segment = segData['segment']
                
                # Apply certainty changes (deletion is handled separately in _saveChanges)
                if state == 'questioned':
                    if len(segment[4]) > 0:
                        for label in segment[4]:
                            label["certainty"] = 50
                elif state == 'accepted':
                    if len(segment[4]) > 0:
                        for label in segment[4]:
                            label["certainty"] = 100
                # Note: 'deleted' state is handled in _saveChanges by removing segments
        else:
            # Single-file mode: apply changes to segments in self.segments
            for segmentIndex, state in self.segmentChanges.items():
                if segmentIndex >= len(self.segments):
                    continue
                    
                segment = self.segments[segmentIndex]
                
                # Apply certainty changes (deletion is handled separately in humanClassifyClose2)
                if state == 'questioned':
                    if len(segment[4]) > 0:
                        for label in segment[4]:
                            label["certainty"] = 50
                elif state == 'accepted':
                    if len(segment[4]) > 0:
                        for label in segment[4]:
                            label["certainty"] = 100
                # Note: 'deleted' state is handled in humanClassifyClose2 by removing segments
                    
                segment = self.segments[segmentIndex]
                
                if state == 'questioned':
                    if len(segment[4]) > 0:
                        for label in segment[4]:
                            label["certainty"] = 50
                elif state == 'accepted':
                    if len(segment[4]) > 0:
                        for label in segment[4]:
                            label["certainty"] = 100
                # Note: deletion handled separately

    def loadCurrentSegment(self):
        """ Load the current segment for one-by-one review """
        if self.currentSegmentIndex >= len(self.allSegmentsToReview):
            return
            
        segData = self.allSegmentsToReview[self.currentSegmentIndex]
        filename = segData['filename']
        segment = segData['segment']
        
        # Set up current file context
        self.filename = filename
        filedata = self.allFileData[filename]
        self.segments = Segment.SegmentList()
        self.segments.append(segment)
        self.allsegments = filedata['allsegments']
        
        # Load this single segment
        self.loadFile(filename, species=self.species)
        
        # Set up indices for the dialog
        self.indices2show = [0]  # Only showing one segment

    def loadFile(self, filename, species=None, chunksize=None):
        """ Generates spectrograms and audiodatas
            for each segment in self.segments.
            If chunksize is set, will buffer appropriately.
            The Spectrogram containing these are loaded into self.sps.
        """
        with pg.BusyCursor():
            # Initialize or clean up spectrograms list
            if hasattr(self, 'sps') and self.sps:
                # delete old instances to force release memory
                for sp in reversed(range(len(self.sps))):
                    del self.sps[sp]
            else:
                self.sps = []
            
            minsg = 1
            maxsg = 1
            gc.collect()

            with pg.ProgressDialog("Loading file...", 0, len(self.segments)) as dlg:
                dlg.setCancelButton(None)
                dlg.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
                dlg.setWindowTitle('AviaNZ')
                dlg.setFixedSize(350, 100)
                dlg.setWindowFlags(self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint)
                dlg.update()
                dlg.repaint()
                dlg.show()

                if self.batmode:
                    # Not sure how to do an equivalent of readFmt for bmps?
                    # Maybe easier to just read in the entire bmp here?
                    samplerate = 176000
                    duration = self.segments.metadata["Duration"]
                else:
                    # Determine the sample rate and set some file-level parameters
                    info = sf.info(filename)
                    samplerate = info.samplerate
                    duration = info.frames / samplerate

                minFreq = max(self.fLow.value(), 0)
                maxFreq = min(self.fHigh.value(), samplerate//2)
                if maxFreq - minFreq < 100:
                    print("ERROR: less than 100 Hz band set for spectrogram")
                    return
                print("Filtering samples to %d - %d Hz" % (minFreq, maxFreq))

                # For single sp, no need to load all segments, but don't want to edit self.segments
                if self.species is not None and self.species != "All species":
                    self.indices2show = self.segments.getSpecies(species)
                else:
                    self.indices2show = range(len(self.segments))

                print(self.indices2show)
                if chunksize is not None:
                    halfChunk = 1.1/2 * chunksize

                # Load data into a list of Spectrogram (with spectrograms) for each segment
                for segix in range(len(self.segments)):
                    if segix in self.indices2show:
                        seg = self.segments[segix]
                        # note that sp also stores the range of shown freqs
                        sp = Spectrogram.Spectrogram(self.config_manager.config['window_width'], self.config_manager.config['incr'], minFreq, maxFreq)

                        if chunksize is not None:
                            mid = (seg[0]+seg[1])/2
                            # buffered limits in audiodata (sec) = display limits
                            x1 = max(0, mid-halfChunk)
                            x2 = min(duration, mid+halfChunk)

                            # unbuffered limits in audiodata
                            x1nob = max(seg[0], x1)
                            x2nob = min(seg[1], x2)
                        else:
                            # unbuffered limits in audiodata
                            x1nob = seg[0]
                            x2nob = seg[1]

                            # buffered limits in audiodata (sec) = display limits
                            x1 = max(x1nob - self.config_manager.config['reviewSpecBuffer'], 0)
                            x2 = min(x2nob + self.config_manager.config['reviewSpecBuffer'], duration)

                        # Actual loading of the wav/bmp/spectrogram
                        if self.batmode:
                            sp.readBmp(filename, off=x1, duration=x2-x1, silent=segix>1)
                            # sg was already normalised to 0-1 when loading
                            # with 1 being loudest
                            sp.sg = sp.normalisedSpec("Batmode")
                            minsg = 0
                            maxsg = 1
                        else:
                            # segix>1 to print the format details only once for each file
                            sp.readSoundFile(filename, off=x1, duration=x2-x1, silent=segix>1)

                            # Filter the audiodata based on initial sliders
                            sp.data = SignalProc.bandpassFilter(sp.data, sp.audioFormat.sampleRate(), minFreq, maxFreq)

                            # Generate the spectrogram
                            # TODO: Insist on log scale?
                            sp.sg = sp.spectrogram(window_width=self.config_manager.config['window_width'], incr=self.config_manager.config['incr'],window=self.config_manager.config['windowType'],sgType=self.config_manager.config['sgType'],sgScale=self.config_manager.config['sgScale'],nfilters=self.config_manager.config['nfilters'],mean_normalise=self.config_manager.config['sgMeanNormalise'],equal_loudness=self.config_manager.config['sgEqualLoudness'],onesided=self.config_manager.config['sgOneSided'])
                            #sp.sg = sp.normalisedSpec("Log")

                            # collect min and max values for final colour scale
                            minsg = min(np.min(sp.sg), minsg)
                            maxsg = max(np.max(sp.sg), maxsg)

                        # need to also store unbuffered limits in spec units
                        # (relative to start of segment)
                        sp.x1nobspec = sp.convertAmpltoSpec(x1nob-x1)
                        sp.x2nobspec = sp.convertAmpltoSpec(x2nob-x1)

                        # trim the spectrogram
                        height = sp.audioFormat.sampleRate()//2 / np.shape(sp.sg)[1]
                        pixelstart = int(minFreq/height)
                        pixelend = int(maxFreq/height)
                        sp.sg = sp.sg[:,pixelstart:pixelend]
                    else:
                        sp = None

                    self.sps.append(sp)

                    dlg += 1
                    dlg.update()
                    dlg.repaint()

            # sets the color map, based on the extremes of all segment spectrograms
            cmap = self.config_manager.config['cmap']
            pos, colour, mode = colourMaps.colourMaps(cmap)
            # SRM bug
            cmap = pg.ColorMap(pos, colour)
            #cmap = pg.ColorMap(pos, colour,mode)

            self.lut = cmap.getLookupTable(0.0, 1.0, 256)

        # END of file loading

    def showCurrentSegment(self):
        """ Display the current segment without advancing the index """
        if hasattr(self, 'allSegmentsToReview'):
            # Cross-file navigation mode
            if self.currentSegmentIndex < len(self.allSegmentsToReview):
                # Update title
                self.humanClassifyDialog1.setWindowTitle(f"AviaNZ - reviewing segment {self.currentSegmentIndex + 1}/{len(self.allSegmentsToReview)}")
                
                # Use the ORIGINAL segment from allSegmentsToReview so species changes persist
                segData = self.allSegmentsToReview[self.currentSegmentIndex]
                original_segment = segData['segment']

                # update "done/to go" numbers based on actual status counts
                self.humanClassifyDialog1.setSegNumbers(self.segsAccepted, self.segsDeleted, self.segsQuestioned, len(self.allSegmentsToReview))

                # select the Spectrogram with relevant data
                sp = self.sps[0]  # Only one segment loaded

                # these pass the axis limits set by slider
                minFreq = max(self.fLow.value(), 0)
                maxFreq = min(self.fHigh.value(), sp.audioFormat.sampleRate()//2)

                if self.config_manager.config['guidelinesOn']=='always' or (self.config_manager.config['guidelinesOn']=='bat' and self.batmode):
                    guides = [sp.convertFreqtoY(f) for f in self.config_manager.config['guidepos']]
                else:
                    guides = None

                if self.batmode:
                    sg = sp.normalisedSpec("Batmode")
                else:
                    sg = sp.normalisedSpec(self.config_manager.config['sgNormMode'])

                self.humanClassifyDialog1.setImage(sg, sp.data, sp.audioFormat.sampleRate(), sp.incr,
                                                   original_segment, sp.x1nobspec, sp.x2nobspec,
                                                   guides, minFreq, maxFreq)
        else:
            # Original file-based mode
            if self.box1id < len(self.indices2show):
                # Show the current segment
                seg = self.segments[self.indices2show[self.box1id]]

                # update "done/to go" numbers:
                self.humanClassifyDialog1.setSegNumbers(self.segsAccepted, self.segsDeleted, self.segsQuestioned, self.nsegments)

                # select the Spectrogram with relevant data
                sp = self.sps[self.indices2show[self.box1id]]

                # these pass the axis limits set by slider
                minFreq = max(self.fLow.value(), 0)
                maxFreq = min(self.fHigh.value(), sp.audioFormat.sampleRate()//2)

                if self.config_manager.config['guidelinesOn']=='always' or (self.config_manager.config['guidelinesOn']=='bat' and self.batmode):
                    guides = [sp.convertFreqtoY(f) for f in self.config_manager.config['guidepos']]
                else:
                    guides = None

                if self.batmode:
                    sg = sp.normalisedSpec("Batmode")
                else:
                    sg = sp.normalisedSpec(self.config_manager.config['sgNormMode'])

                self.humanClassifyDialog1.setImage(sg, sp.data, sp.audioFormat.sampleRate(), sp.incr,
                                                   seg, sp.x1nobspec, sp.x2nobspec,
                                                   guides, minFreq, maxFreq)

    def saveCorrections(self):
        """ Save corrections for the current review session.
        For cross-file one-by-one mode, corrections are handled when saving each file.
        This method is primarily for legacy/quick mode compatibility.
        """
        if hasattr(self, 'allSegmentsToReview'):
            # In cross-file mode, corrections are handled when saving each file
            print("Corrections tracking for cross-file review is handled during file save")
            return
            
        # Original file-based correction saving (for legacy modes)
        if not hasattr(self, 'origSeg') or not self.config_manager.config['saveCorrections']:
            return
            
        for i in reversed(range(len(self.segments))):
            if self.segments[i][4] == self.origSeg[i][4]:
                # No changes made to this segment
                del self.origSeg[i]
            else:
                oldlabel = self.origSeg[i][4]
                newlabel = self.segments[i][4]
                if "-To Be Deleted-" in [lab["species"] for lab in newlabel]:
                    self.origSeg[i] = [self.origSeg[i], []]
                else:
                    # Check for species or calltype changes
                    if [lab["species"] for lab in oldlabel] != [lab["species"] for lab in newlabel] or \
                       [lab.get("calltype") for lab in oldlabel] != [lab.get("calltype") for lab in newlabel]:
                        self.origSeg[i] = [self.origSeg[i], newlabel]

        if len(self.origSeg) > 0:
            cleanexit = self.saveCorrectJSON(str(self.filename + '.corrections'), self.origSeg, mode=1, reviewer=self.reviewer)
            if cleanexit != 1:
                print("Warning: could not save correction file!")

    def humanClassifyNextImage1(self, move_forward=True):
        # Get the next image
        # In one-by-one mode across files, we always show one segment at a time
        if hasattr(self, 'allSegmentsToReview'):
            # Cross-file navigation mode
            if move_forward and self.currentSegmentIndex < len(self.allSegmentsToReview) - 1:
                # Move to next segment (possibly in different file)
                self.currentSegmentIndex += 1
                self.loadCurrentSegment()
                self.box1id = 0  # Reset to first (only) segment in current load
            
            # Display current segment (whether we moved forward or not)
            if self.currentSegmentIndex < len(self.allSegmentsToReview):
                # Update title
                self.humanClassifyDialog1.setWindowTitle(f"AviaNZ - reviewing segment {self.currentSegmentIndex + 1}/{len(self.allSegmentsToReview)}")
                
                # Show the segment
                seg = self.segments[0]  # Only one segment loaded
                lab = seg[4]

                # update "done/to go" numbers based on actual status counts
                self.humanClassifyDialog1.setSegNumbers(self.segsAccepted, self.segsDeleted, self.segsQuestioned, len(self.allSegmentsToReview))

                # select the Spectrogram with relevant data
                sp = self.sps[0]  # Only one segment loaded

                # these pass the axis limits set by slider
                minFreq = max(self.fLow.value(), 0)
                maxFreq = min(self.fHigh.value(), sp.audioFormat.sampleRate()//2)

                if self.config_manager.config['guidelinesOn']=='always' or (self.config_manager.config['guidelinesOn']=='bat' and self.batmode):
                    guides = [sp.convertFreqtoY(f) for f in self.config_manager.config['guidepos']]
                else:
                    guides = None

                if self.batmode:
                    sg = sp.normalisedSpec("Batmode")
                else:
                    sg = sp.normalisedSpec(self.config_manager.config['sgNormMode'])

                self.humanClassifyDialog1.setImage(sg, sp.data, sp.audioFormat.sampleRate(), sp.incr,
                                                   seg, sp.x1nobspec, sp.x2nobspec,
                                                   guides, minFreq, maxFreq)
            else:
                # End of all segments - finish review
                self._finishReviewDialog()
        else:
            # Original file-based navigation
            if self.box1id < len(self.indices2show)-1:
                self.box1id += 1
                # Check if have moved to next segment, and if so load it

                # Show the next segment
                seg = self.segments[self.indices2show[self.box1id]]
                lab = seg[4]

                # update "done/to go" numbers:
                if self.returned:
                    if len(lab)==1 and lab[0]["species"] == "-To Be Deleted-":
                        self.segsDeleted -= 1
                    elif len(lab)>0 and lab[0].get("certainty", 100) < 100:
                        self.segsQuestioned -= 1
                    else:
                        self.segsAccepted -= 1

                # print(self.segsAccepted,self.segsDeleted,self.segsQuestioned,self.nsegments)
                self.humanClassifyDialog1.setSegNumbers(self.segsAccepted, self.segsDeleted, self.segsQuestioned, self.nsegments)

                # select the Spectrogram with relevant data
                sp = self.sps[self.indices2show[self.box1id]]

                # these pass the axis limits set by slider
                minFreq = max(self.fLow.value(), 0)
                maxFreq = min(self.fHigh.value(), sp.audioFormat.sampleRate()//2)

                if self.config_manager.config['guidelinesOn']=='always' or (self.config_manager.config['guidelinesOn']=='bat' and self.batmode):
                    guides = [sp.convertFreqtoY(f) for f in self.config_manager.config['guidepos']]
                else:
                    guides = None

                # currLabel, then unbufstart in spec units rel to start, unbufend,
                # then true time to display start, end,
                # NOTE: might be good to pass copy.deepcopy(seg[4])
                # instead of seg[4], if any bugs come up due to Dialog1 changing the label

                if self.batmode:
                    sg = sp.normalisedSpec("Batmode")
                else:
                    sg = sp.normalisedSpec(self.config_manager.config['sgNormMode'])

                self.humanClassifyDialog1.setImage(sg, sp.data, sp.audioFormat.sampleRate(), sp.incr,
                                                   seg, sp.x1nobspec, sp.x2nobspec,
                                                   guides, minFreq, maxFreq)
            else:
                # store dialog properties such as position for the next file
                self.dialogSize = self.humanClassifyDialog1.size()
                self.dialogPos = self.humanClassifyDialog1.pos()
                self.dialogPlotAspect = self.humanClassifyDialog1.plotAspect
                # Save review preferences via ConfigManager
                brightness = self.humanClassifyDialog1.specControls.brightSlider.value()
                contrast = self.humanClassifyDialog1.specControls.contrSlider.value()
                if not self.config_manager.config['invertColourMap']:
                    brightness = 100 - brightness
                
                self.config_manager.save_review_preferences({
                    'brightness': brightness,
                    'contrast': contrast
                })

                self.humanClassifyDialog1.done(1)

    def humanClassifyPrevImage(self):
        """ Go back one image, undoing any status changes made to the current segment.
        Note: Species changes are NOT undone as they should be permanent once made."""
        if hasattr(self, 'allSegmentsToReview'):
            if self.currentSegmentIndex > 0:
                self.currentSegmentIndex -= 1
                self.loadCurrentSegment()
                self.returned = True

                # Undo status changes (but not species changes)
                currentState = self.segmentChanges.get(self.currentSegmentIndex)
                
                if currentState is not None:
                    if currentState == 'accepted':
                        self.segsAccepted -= 1
                    elif currentState == 'deleted':
                        self.segsDeleted -= 1
                    elif currentState == 'questioned':
                        self.segsQuestioned -= 1
                    
                    del self.segmentChanges[self.currentSegmentIndex]
                
                # Update the display counters and show the previous segment
                self.humanClassifyDialog1.setSegNumbers(self.segsAccepted, self.segsDeleted, self.segsQuestioned, len(self.allSegmentsToReview))
                self.humanClassifyNextImage1(move_forward=False)
        else:
            # Original file-based navigation
            if self.box1id > 0:
                self.box1id -= 2
                self.returned = True
                self.humanClassifyNextImage1()

    def humanClassifyQuestion(self):
        """ Go to next image, marking this one as questioned """
        self.humanClassifyDialog1.stopPlayback()

        saveConfig = self.humanClassifyDialog1.checkIfNeedToSaveConfig()
        if saveConfig:
            self._saveBirdListConfig()
        
        # Save any species changes first (these are immediate and irreversible)
        self._saveCurrentSegmentState()
        
        # Track this QUESTION action for certainty changes (reversible)
        if not hasattr(self, 'segmentChanges'):
            self.segmentChanges = {}
        
        if hasattr(self, 'allSegmentsToReview'):
            # Cross-file mode
            segmentIndex = self.currentSegmentIndex
        else:
            # Single-file mode  
            segmentIndex = self.indices2show[self.box1id]
        
        prevState = self.segmentChanges.get(segmentIndex)
        self.segmentChanges[segmentIndex] = 'questioned'
        
        # Update counters based on previous state
        if prevState == 'accepted':
            self.segsAccepted -= 1
        elif prevState == 'deleted':
            self.segsDeleted -= 1
        elif prevState is None:  # First time changing this segment
            pass  # No previous counter to decrement
        
        if prevState != 'questioned':
            self.segsQuestioned += 1

        # Update the display counters
        if hasattr(self, 'allSegmentsToReview'):
            self.humanClassifyDialog1.setSegNumbers(self.segsAccepted, self.segsDeleted, self.segsQuestioned, len(self.allSegmentsToReview))

        self.returned = False
        
        # Check if this was the last segment
        if hasattr(self, 'allSegmentsToReview') and self.currentSegmentIndex >= len(self.allSegmentsToReview) - 1:
            self._finishReviewDialog()
        else:
            self.humanClassifyNextImage1()

    def humanClassifyPlus(self):
        """ Repeat a segment, offset slightly in freq and time """
        self.humanClassifyDialog1.stopPlayback()

        # Handle both cross-file and original modes
        if hasattr(self, 'allSegmentsToReview'):
            # Cross-file mode: Save any species changes and record the PLUS action
            currSeg = self.segments[0]  # Only one segment loaded at a time
            # Save current segment state (including any species changes made by user)
            self._saveCurrentSegmentState()
        else:
            # Original mode
            currSeg = self.segments[self.indices2show[self.box1id]]
            
        # For Plus functionality, we still need to confirm the labels immediately
        # since we're creating copies based on the current segment state
        currSeg.confirmLabels(None if self.species == 'All species' else self.species)
        getNumCopies = Dialogs.getNumberCopiesPlus()
        response = getNumCopies.exec()
        
        if response == 0:
            print("DEBUG: Plus button - user cancelled")
            return

        numCopies = getNumCopies.getValues()
        print(f"DEBUG: Plus button - creating {numCopies} copies")

        if not hasattr(self, 'toadd'):
            self.toadd = {}
            print("DEBUG: Plus button - initialized toadd dictionary")
            
        # Get current filename to track where these segments belong
        if hasattr(self, 'allSegmentsToReview'):
            current_filename = self.allSegmentsToReview[self.currentSegmentIndex]['filename']
        else:
            current_filename = self.filename
            
        print(f"DEBUG: Plus button - adding segments to file: {current_filename}")
            
        if current_filename not in self.toadd:
            self.toadd[current_filename] = []
            
        original_segment_count = len(self.toadd[current_filename])
            
        for i in range(numCopies):
            newSeg = copy.deepcopy(currSeg)
            newSeg[0] += (i+1)*0.1  # Start time offset
            newSeg[1] += (i+1)*0.1  # End time offset
            newSeg[2] += (i+1)*50   # Low freq offset
            newSeg[3] += (i+1)*50   # High freq offset
            self.toadd[current_filename].append(newSeg)
            print(f"DEBUG: Plus button - created copy {i+1}: time {newSeg[0]:.1f}-{newSeg[1]:.1f}, freq {newSeg[2]:.0f}-{newSeg[3]:.0f}")

        print(f"DEBUG: Plus button - total segments to add for {current_filename}: {len(self.toadd[current_filename])} (was {original_segment_count})")

        self.returned = False
        self.segsAccepted += 1
        
        # Check if this was the last segment
        if hasattr(self, 'allSegmentsToReview') and self.currentSegmentIndex >= len(self.allSegmentsToReview) - 1:
            print("DEBUG: Plus button - finishing review dialog")
            self._finishReviewDialog()
        else:
            self.humanClassifyNextImage1()

    def humanClassifyCorrect1(self):
        """ Correct segment labels, save the old ones if necessary """
        self.humanClassifyDialog1.stopPlayback()

        saveConfig = self.humanClassifyDialog1.checkIfNeedToSaveConfig()
        if saveConfig:
            self._saveBirdListConfig()
        
        # Save any species changes first (these are immediate and irreversible)
        self._saveCurrentSegmentState()
        
        # Track this CORRECT action for certainty changes (reversible)
        if not hasattr(self, 'segmentChanges'):
            self.segmentChanges = {}
        
        if hasattr(self, 'allSegmentsToReview'):
            # Cross-file mode
            segmentIndex = self.currentSegmentIndex
        else:
            # Single-file mode  
            segmentIndex = self.indices2show[self.box1id]
        
        prevState = self.segmentChanges.get(segmentIndex)
        self.segmentChanges[segmentIndex] = 'accepted'
        
        # Update counters based on previous state
        if prevState == 'deleted':
            self.segsDeleted -= 1
        elif prevState == 'questioned':
            self.segsQuestioned -= 1
        elif prevState is None:  # First time changing this segment
            pass  # No previous counter to decrement
        
        if prevState != 'accepted':
            self.segsAccepted += 1

        # Update the display counters
        if hasattr(self, 'allSegmentsToReview'):
            self.humanClassifyDialog1.setSegNumbers(self.segsAccepted, self.segsDeleted, self.segsQuestioned, len(self.allSegmentsToReview))

        self.returned = False
        
        # Check if this was the last segment
        if hasattr(self, 'allSegmentsToReview') and self.currentSegmentIndex >= len(self.allSegmentsToReview) - 1:
            self._finishReviewDialog()
        else:
            self.humanClassifyNextImage1()

    def humanClassifyDelete1New(self):
        """ Delete a segment """
        self.humanClassifyDialog1.stopPlayback()

        # Track this DELETE action (same pattern as other actions)
        if not hasattr(self, 'segmentChanges'):
            self.segmentChanges = {}
        
        if hasattr(self, 'allSegmentsToReview'):
            # Cross-file mode
            segmentIndex = self.currentSegmentIndex
        else:
            # Single-file mode  
            segmentIndex = self.indices2show[self.box1id]
        
        prevState = self.segmentChanges.get(segmentIndex)
        self.segmentChanges[segmentIndex] = 'deleted'
        
        # Update counters based on previous state
        if prevState == 'accepted':
            self.segsAccepted -= 1
        elif prevState == 'questioned':
            self.segsQuestioned -= 1
        elif prevState is None:  # First time changing this segment
            pass  # No previous counter to decrement
        
        if prevState != 'deleted':
            self.segsDeleted += 1

        # Update the display counters
        if hasattr(self, 'allSegmentsToReview'):
            self.humanClassifyDialog1.setSegNumbers(self.segsAccepted, self.segsDeleted, self.segsQuestioned, len(self.allSegmentsToReview))

        self.returned = False
        
        # Check if this was the last segment
        if hasattr(self, 'allSegmentsToReview') and self.currentSegmentIndex >= len(self.allSegmentsToReview) - 1:
            self._finishReviewDialog()
        else:
            self.humanClassifyNextImage1()

    def finishDeleting(self):
        """
        Remove any segments that still have "-To Be Deleted-" markers.
        This is a cleanup method for backwards compatibility.
        """
        # Clean up any old-style deletion markers
        for seg in reversed(self.segments):
            if len(seg[4]) > 0 and seg[4][0].get("species") == "-To Be Deleted-":
                print("Removing legacy marked segment:", seg)
                self.segments.remove(seg)

    def closeDialog(self, ev):
        """ Handle dialog close events, including Esc key press """
        if ev == Qt.Key.Key_Escape and hasattr(self, 'humanClassifyDialog1'):
            # Save changes with confirmed_only=True on Esc press
            if hasattr(self, 'allSegmentsToReview'):
                self._saveChanges(confirmed_only=True)
            self.humanClassifyDialog1.done(0)

    def updateCallType(self, boxid, calltype):
        """ Compares calltype with oldseg labels, does safety checks,
            updates the segment, and stores corrections.
            boxid - id of segment being updated
            calltype - new calltype to be placed on the first species of this segment
        """
        if calltype=="":
            return
        oldlab = self.segments[boxid][4]
        if len(oldlab)==0:
            print("Warning: can't add call type to empty segment")
            return

        # Currently, only working with the call type if a single species is selected:
        if len(oldlab)>1:
            print("Warning: setting call types with multiple species labels not supported yet")
            return

        if "calltype" in oldlab[0]:
            if oldlab[0]["calltype"]==calltype:
                # Nothing to change
                return

        print("Changing calltype to", calltype)

        # actually update the segment info
        self.segments[boxid][4][0]["calltype"] = calltype

