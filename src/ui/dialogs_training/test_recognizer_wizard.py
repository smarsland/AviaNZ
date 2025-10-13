
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

# Dialogs used for filter training / testing.
# These are relatively complicated wizards which also do file I/O

import os
import platform

from PyQt6.QtGui import QIcon, QColor
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton, QComboBox, QWizard, QWizardPage, QLineEdit, QSizePolicy, QFormLayout, QVBoxLayout, QHBoxLayout, QFileDialog, QAbstractItemView

import pyqtgraph as pg

from src.ui.components.file_list import LightedFileList
from core import config_loader
from core import training



class TestRecWizard(QWizard):
    class WPageData(QWizardPage):
        def __init__(self, config, filterlist, filter=None, parent=None):
            super(TestRecWizard.WPageData, self).__init__(parent)
            self.setTitle('Testing data')
            self.setSubTitle('Select the folder with testing data, then choose species')

            self.setMinimumSize(250, 150)
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            self.adjustSize()

            # the combobox will default to this filter initially if provided
            self.initialFilter = filter

            # grab the full filter list
            self.filterlist = filterlist

            self.testDirName = QLineEdit()
            self.testDirName.setReadOnly(True)
            self.btnBrowse = QPushButton('Browse')
            self.btnBrowse.clicked.connect(self.browseTestData)

            colourNone = QColor(config['ColourNone'][0], config['ColourNone'][1], config['ColourNone'][2], config['ColourNone'][3])
            colourPossibleDark = QColor(config['ColourPossible'][0], config['ColourPossible'][1], config['ColourPossible'][2], 255)
            colourNamed = QColor(config['ColourNamed'][0], config['ColourNamed'][1], config['ColourNamed'][2], config['ColourNamed'][3])
            self.listFiles = LightedFileList(colourNone, colourPossibleDark, colourNamed)
            self.listFiles.setMinimumHeight(275)
            self.listFiles.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

            selectSpLabel = QLabel("Choose the recogniser that you want to test")
            self.recognisers = QComboBox()  # fill during browse
            self.recognisers.addItems(['Choose recogniser...'])

            space = QLabel()
            space.setFixedHeight(20)

            # data selection page layout
            layout1 = QHBoxLayout()
            layout1.addWidget(self.testDirName)
            layout1.addWidget(self.btnBrowse)
            layout = QVBoxLayout()
            layout.addWidget(space)
            layout.addLayout(layout1)
            layout.addWidget(self.listFiles)
            layout.addWidget(space)
            layout.addWidget(selectSpLabel)
            layout.addWidget(self.recognisers)
            layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            self.setLayout(layout)
            self.setButtonText(QWizard.WizardButton.NextButton, 'Test >')

        #def initializePage(self):
            #filternames = [key + ".txt" for key in self.wizard().filterlist.keys()]
            #self.recognisers.addItems(sorted(filternames))
            #if self.initialFilter is not None:
                #self.recognisers.setCurrentText(self.initialFilter)

        def browseTestData(self):
            dirName = QFileDialog.getExistingDirectory(self, 'Choose folder for testing')
            self.testDirName.setText(dirName)

            self.listFiles.fill(dirName, fileName=None, readFmt=False, addWavNum=True, recursive=True)

            # while reading the file, we also collected a list of species present there
            spList = list(self.listFiles.spList)

            recogniserList = []
            # loop through the filters and get every filter where the species is in spList
            for key, value in self.filterlist.items():
                if value['species'] in spList:
                    recogniserList.append(key)
            
            print("found recognisers",recogniserList)

            recogniserList.insert(0, 'Choose recogniser...')
            self.recognisers.clear()
            self.recognisers.addItems(recogniserList)
            if len(spList)==2:
                self.recognisers.setCurrentIndex(1)
            if self.initialFilter is not None and self.initialFilter in spList:
                self.recognisers.setCurrentText(self.initialFilter)

    class WPageMain(QWizardPage):
        def __init__(self, configdir, filterdir, parent=None):
            super(TestRecWizard.WPageMain, self).__init__(parent)
            self.setTitle('Summary of testing results')

            self.setMinimumSize(300, 300)
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            self.adjustSize()
            self.configdir = configdir
            self.filterdir = filterdir

            self.lblTestDir = QLabel()
            self.lblTestDir.setStyleSheet("QLabel { color : #808080; }")
            self.lblTestFilter = QLabel()
            self.lblTestFilter.setStyleSheet("QLabel { color : #808080; }")
            self.lblSpecies = QLabel()
            self.lblSpecies.setStyleSheet("QLabel { color : #808080; }")
            space = QLabel()
            space.setFixedHeight(25)

            self.lblWFsummary = QLabel()
            # self.lblWFNNsummary = QLabel()
            # self.lblWFsummary.setStyleSheet("QLabel { color : #808080; }")
            # self.lblWFNNsummary.setStyleSheet("QLabel { color : #808080; }")
            self.lblOutfile = QLabel()

            # page layout
            vboxHead = QFormLayout()
            vboxHead.addRow("Testing data:", self.lblTestDir)
            vboxHead.addRow("Filter name:", self.lblTestFilter)
            vboxHead.addRow("Species name:", self.lblSpecies)
            vboxHead.addWidget(space)
            vbox = QVBoxLayout()
            vbox.addLayout(vboxHead)
            vbox.addWidget(self.lblWFsummary)
            # vbox.addWidget(self.lblWFNNsummary)
            vbox.addWidget(self.lblOutfile)
            self.setLayout(vbox)

        def initializePage(self):
            # Testing results will be stored there
            #testresfile = os.path.join(self.field("testDir"), "test-results.txt")
            # Run the actual testing here:
            with pg.BusyCursor():
                self.currfilt = self.wizard().filterlist[self.field("recognisers")]
                #self.currfilt = self.wizard().filterlist[self.field("recognisers")[:-4]]

                self.lblTestDir.setText(self.field("testDir"))
                self.lblTestFilter.setText(self.field("recognisers"))
                self.lblSpecies.setText(self.currfilt['species'])

                test = training.NNTest(self.field("testDir"), self.currfilt, self.field("recognisers"), self.configdir,self.filterdir)
                #test = Training.NNTest(self.field("testDir"), self.currfilt, self.field("recognisers")[:-4], self.configdir,self.filterdir)
                text = test.getOutput()

            if text == 0:
                self.lblWFsummary.setText("No segments for recognisers \'%s\' found!" % self.field("recognisers"))
                #self.lblWFsummary.setText("No segments for recognisers \'%s\' found!" % self.field("recognisers")[:-4])
                return

            self.lblWFsummary.setText(text)
            resfile = os.path.join(self.field("testDir"), "test-results.txt")
            self.lblOutfile.setText("The detailed results have been saved in file\n%s" % resfile)

        def cleanupPage(self):
            self.lblWFsummary.setText('')
            # self.lblWFNNsummary.setText('')
            self.lblRecognisers.setText('')
            self.lblTestDir.setText('')
            self.lblTestFilter.setText('')

    # extra page to display the full results?
    # class WPageFull(QWizardPage):
    #     def __init__(self, parent=None):
    #         super(TestRecWizard.WPageFull, self).__init__(parent)
    #         self.setTitle('Detailed testing results')

    #         self.setMinimumSize(300, 300)
    #         self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    #         self.adjustSize()

    #         self.results = QTextEdit()
    #         self.results.setReadOnly(True)

    #         vbox = QVBoxLayout()
    #         vbox.addWidget(self.results)
    #         self.setLayout(vbox)

    #     def initializePage(self):
    #         resfile = os.path.join(self.field("testDir"), "test-results.txt")
    #         resstream = open(resfile, 'r')
    #         self.results.setPlainText(resstream.read())
    #         resstream.close()

    #     def cleanupPage(self):
    #         self.results.setPlainText('')

    # Main init of the testing wizard
    def __init__(self, filtdir, configdir, filter=None, parent=None):
        super(TestRecWizard, self).__init__()
        self.setWindowTitle("Test Recogniser")
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        if platform.system() == 'Linux':
            self.setWindowFlags(self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint)
        else:
            self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOptions(QWizard.WizardOption.NoBackButtonOnStartPage)

        cl = ConfigLoader.ConfigLoader()
        self.filterlist = cl.filters(filtdir, bats=False)
        configfile = os.path.join(configdir, "AviaNZconfig.txt")
        ConfigLoader = ConfigLoader.ConfigLoader()
        config = ConfigLoader.config(configfile)
        browsedataPage = TestRecWizard.WPageData(config, self.filterlist, filter=filter)
        browsedataPage.registerField("testDir*", browsedataPage.testDirName)
        browsedataPage.registerField("recognisers*", browsedataPage.recognisers, "currentText", browsedataPage.recognisers.currentTextChanged)
        self.addPage(browsedataPage)

        pageMain = TestRecWizard.WPageMain(configdir, filtdir)
        self.addPage(pageMain)

        # extra page to show more details
        # self.addPage(TestRecWizard.WPageFull())
