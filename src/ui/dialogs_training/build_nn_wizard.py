
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
import platform

from PyQt6.QtGui import QIcon, QValidator, QPixmap, QColor
from PyQt6.QtCore import QDir, Qt

from src.ui.components.validators import FiltValidator
from PyQt6.QtWidgets import QLabel, QSlider, QPushButton, QListWidget, QListWidgetItem, QComboBox, QWizard, QWizardPage, QLineEdit, QSizePolicy, QVBoxLayout, QHBoxLayout, QCheckBox, QRadioButton, QGridLayout, QFileDialog, QAbstractItemView

import pyqtgraph as pg

import numpy as np
from src.ui.colourMaps import colourMaps
from src.ui.components.buttons_and_controls import CustomSlider, PicButton
from src.ui.components.file_list import LightedFileList
from src.core import config_loader
from src.core import spectrogram
from src.core import annotation
from src.core import training



class BuildNNWizard(QWizard):
    # Main init of the NN training wizard
    def __init__(self, filtdir, config, configdir, parent=None):
        super(BuildNNWizard, self).__init__()
        self.setWindowTitle("Train NN")
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        if platform.system() == 'Linux':
            self.setWindowFlags(self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint)
        else:
            self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOptions(QWizard.WizardOption.NoBackButtonOnStartPage)

        self.rocpages = []

        self.nntrain = training.NNTrain(configdir, filtdir)

        # P1
        self.browsedataPage = BuildNNWizard.WPageData(self.nntrain, config)
        self.browsedataPage.registerField("trainDir1*", self.browsedataPage.trainDirName1)
        self.browsedataPage.registerField("trainDir2*", self.browsedataPage.trainDirName2)
        self.browsedataPage.registerField("filter*", self.browsedataPage.speciesCombo, "currentText", self.browsedataPage.speciesCombo.currentTextChanged)
        self.addPage(self.browsedataPage)

        # P2
        self.confirminputPage = BuildNNWizard.WPageConfirminput(self.nntrain, configdir)
        self.addPage(self.confirminputPage)

        # P3
        self.parameterPage = BuildNNWizard.WPageParameters(config)
        self.parameterPage.registerField("frqMasked*", self.parameterPage.cbfrange, "isChecked")
        self.parameterPage.registerField("f1*", self.parameterPage.f1, "value", self.parameterPage.f1.valueChanged)
        self.parameterPage.registerField("f2*", self.parameterPage.f2, "value", self.parameterPage.f2.valueChanged)
        self.parameterPage.registerField("model*", self.parameterPage.modelArchitecture, "value", self.parameterPage.modelArchitecture.currentTextChanged)
        self.addPage(self.parameterPage)

        # add the Save & Test button
        self.saveTestBtn = QPushButton("Save and Test")
        self.setButton(QWizard.WizardButton.CustomButton1, self.saveTestBtn)
        self.setButtonLayout( [QWizard.WizardButton.Stretch, QWizard.WizardButton.BackButton, QWizard.WizardButton.NextButton, QWizard.WizardButton.CustomButton1, QWizard.WizardButton.FinishButton, QWizard.WizardButton.CancelButton])
        self.setOptions(QWizard.WizardOption.NoBackButtonOnStartPage | QWizard.WizardOption.HaveCustomButton1)
        self.saveTestBtn.setVisible(False)

    # page 1 - select train data
    class WPageData(QWizardPage):
        def __init__(self, nntrain, config, parent=None):
            super(BuildNNWizard.WPageData, self).__init__(parent)
            self.setTitle('Select data')
            self.setSubTitle('Choose the recogniser that you want to extend with NN, then select training data.')

            self.setMinimumSize(350, 700)
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            self.adjustSize()

            self.nntrain = nntrain

            self.splist1 = []
            self.splist2 = []
            self.anntlevel = "Some"

            self.trainDirName1 = QLineEdit()
            self.trainDirName1.setReadOnly(True)
            self.btnBrowseTrain1 = QPushButton('Browse')
            self.btnBrowseTrain1.clicked.connect(self.browseTrainData1)
            self.trainDirName2 = QLineEdit()
            self.trainDirName2.setReadOnly(True)
            self.btnBrowseTrain2 = QPushButton('Browse')
            self.btnBrowseTrain2.clicked.connect(self.browseTrainData2)

            colourNone = QColor(config['ColourNone'][0], config['ColourNone'][1], config['ColourNone'][2], config['ColourNone'][3])
            colourPossibleDark = QColor(config['ColourPossible'][0], config['ColourPossible'][1], config['ColourPossible'][2], 255)
            colourNamed = QColor(config['ColourNamed'][0], config['ColourNamed'][1], config['ColourNamed'][2], config['ColourNamed'][3])
            self.listFilesTrain2 = LightedFileList(colourNone, colourPossibleDark, colourNamed)
            self.listFilesTrain2.setMinimumWidth(350)
            self.listFilesTrain2.setMinimumHeight(175)
            self.listFilesTrain2.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            self.listFilesTrain1 = LightedFileList(colourNone, colourPossibleDark, colourNamed)
            self.listFilesTrain1.setMinimumWidth(350)
            self.listFilesTrain1.setMinimumHeight(175)
            self.listFilesTrain1.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            self.listFilesTest = LightedFileList(colourNone, colourPossibleDark, colourNamed)
            self.listFilesTest.setMinimumWidth(150)
            self.listFilesTest.setMinimumHeight(175)
            self.listFilesTest.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

            self.speciesCombo = QComboBox()  # fill during browse
            self.speciesCombo.addItems(['Choose recogniser...'])

            self.rbtn1 = QRadioButton("Just from batch review data")
            self.rbtn1.setChecked(True)
            self.rbtn1.annt = "Some"
            self.rbtn1.toggled.connect(self.onClicked)
            self.rbtn1Desc = QLabel("Finds those segments that were detected by the recogniser, but then marked as incorrect by a reviewer. See 'corrections' file.")
            self.rbtn1Desc.setWordWrap(True)
            self.rbtn1Desc.setStyleSheet("font-style: italic")
            self.rbtn1Desc.setIndent(20)
            self.rbtn2 = QRadioButton("From batch review data and detected segments")
            self.rbtn2.annt = "All"
            self.rbtn2.toggled.connect(self.onClicked)
            self.rbtn2Desc = QLabel("In addition to the above this will also run the recogniser over any manual data. Any segments which are found without being near an existing label will be treated as the noise class. If you haven't labelled all calls in the manual data this runs the risk of mislabelling a true call.")
            self.rbtn2Desc.setWordWrap(True)
            self.rbtn2Desc.setStyleSheet("font-style: italic")
            self.rbtn2Desc.setIndent(20)
            self.rbtn3 = QRadioButton("From batch review data and all unlabelled sound")
            self.rbtn3.annt = "All-nowt"
            self.rbtn3.toggled.connect(self.onClicked)
            self.rbtn3Desc = QLabel("Same as above but instead of detecting segments we just use anything that has not been labelled.")
            self.rbtn3Desc.setWordWrap(True)
            self.rbtn3Desc.setStyleSheet("font-style: italic")
            self.rbtn3Desc.setIndent(20)

            space = QLabel()
            space.setFixedHeight(10)
            space.setFixedWidth(40)

            main_layout = QVBoxLayout()

            grid_layout = QGridLayout()
            grid_layout.addWidget(QLabel('<b>Recogniser</b>'), 0, 0)
            grid_layout.addWidget(QLabel("Recogniser that you want to train NN for"), 1, 0)
            grid_layout.addWidget(self.speciesCombo, 1, 1)
            grid_layout.addWidget(space, 2, 0)
            grid_layout.addWidget(QLabel('<b>TRAINING data</b>'), 3, 0)
            grid_layout.addWidget(QLabel('<i>Manually annotated</i>'), 4, 0)
            grid_layout.addWidget(self.btnBrowseTrain1, 5, 0)
            grid_layout.addWidget(self.trainDirName1, 6, 0)
            grid_layout.addWidget(self.listFilesTrain1, 7, 0)
            grid_layout.addWidget(QLabel('<i>Auto processed & Batch reviewed</i>'), 4, 1)
            grid_layout.addWidget(self.btnBrowseTrain2, 5, 1)
            grid_layout.addWidget(self.trainDirName2, 6, 1)
            grid_layout.addWidget(self.listFilesTrain2, 7, 1)
            grid_layout.addWidget(QLabel("How do you want to generate noise samples?"), 8, 0)
            grid_layout.addWidget(space, 3, 2)
            main_layout.addLayout(grid_layout)

            btn_layout = QVBoxLayout()
            btn_layout.addWidget(self.rbtn1)
            btn_layout.addWidget(self.rbtn1Desc)
            btn_layout.addWidget(self.rbtn2)
            btn_layout.addWidget(self.rbtn2Desc)
            btn_layout.addWidget(self.rbtn3)
            btn_layout.addWidget(self.rbtn3Desc)

            main_layout.addLayout(btn_layout)

            self.setLayout(main_layout)

        def initializePage(self):
            filternames = [key + ".txt" for key in self.nntrain.FilterDict.keys()]
            filternames = sorted(filternames)
            self.speciesCombo.addItems(filternames)

        def browseTrainData2(self):
            dirName = QFileDialog.getExistingDirectory(self, 'Choose folder with auto-processed and reviewed training data')
            self.trainDirName2.setText(dirName)

            self.listFilesTrain2.fill(dirName, fileName=None, readFmt=False, addWavNum=True, recursive=True)
            # while reading the file, we also collected a list of species present there
            self.splist2 = list(self.listFilesTrain2.spList)
            self.completeChanged.emit()

        def browseTrainData1(self):
            dirName = QFileDialog.getExistingDirectory(self, 'Choose folder with manually annotated training data')
            self.trainDirName1.setText(dirName)

            self.listFilesTrain1.fill(dirName, fileName=None, readFmt=False, addWavNum=True, recursive=True)
            # while reading the file, we also collected a list of species present there
            self.splist1 = list(self.listFilesTrain1.spList)
            self.completeChanged.emit()

        def onClicked(self):
            radioBtn = self.sender()
            if radioBtn.isChecked():
                self.anntlevel = radioBtn.annt
            self.completeChanged.emit()

        def isComplete(self):
            if self.speciesCombo.currentText() != "Choose recogniser..." and (self.trainDirName1.text() or self.trainDirName2.text()):
                self.nntrain.setP1(self.trainDirName1.text(),self.trainDirName2.text(),self.speciesCombo.currentText(),self.anntlevel)
                return True
            else:
                return False

    # page 2 - data confirm page
    class WPageConfirminput(QWizardPage):
        def __init__(self, nntrain, configdir, parent=None):
            super(BuildNNWizard.WPageConfirminput, self).__init__(parent)
            self.setTitle('Confirm data input')
            self.setSubTitle('When ready, press \"Next\" to start preparing images and train the NN.')
            self.setMinimumSize(350, 275)
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            self.adjustSize()

            self.nntrain = nntrain
            self.certainty1 = True
            self.certainty2 = True
            self.hasant1 = False
            self.hasant2 = False
            cl = config_loader.ConfigLoader()
            self.LearningDict = cl.learningParams(os.path.join(configdir, "LearningParams.txt"))

            self.msgmdir = QLabel("")
            self.msgmdir.setFixedWidth(600)
            self.msgmdir.setWordWrap(True)
            self.msgmdir.setStyleSheet("QLabel { color : #808080; }")
            self.warnnoannt1 = QLabel("")
            self.warnnoannt1.setStyleSheet("QLabel { color : #800000; }")
            self.msgadir = QLabel("")
            self.msgadir.setFixedWidth(600)
            self.msgadir.setWordWrap(True)
            self.msgadir.setStyleSheet("QLabel { color : #808080; }")
            self.warnnoannt2 = QLabel("")
            self.warnnoannt2.setStyleSheet("QLabel { color : #800000; }")
            self.imgDirwarn = QLabel('')
            self.imgDirwarn.setStyleSheet("QLabel { color : #800000; }")

            self.msgrecfilter = QLabel("")
            self.msgrecfilter.setStyleSheet("QLabel { color : #808080; }")
            self.msgrecspp = QLabel("")
            self.msgrecspp.setStyleSheet("QLabel { color : #808080; }")
            self.msgreccts = QLabel("")
            self.msgreccts.setStyleSheet("QLabel { color : #808080; }")
            self.msgrecclens = QLabel("")
            self.msgrecclens.setStyleSheet("QLabel { color : #808080; }")
            self.msgrecfs = QLabel("")
            self.msgrecfs.setStyleSheet("QLabel { color : #808080; }")
            self.msgrecfrange = QLabel("")
            self.msgrecfrange.setStyleSheet("QLabel { color : #808080; }")
            self.warnLabel = QLabel("")
            self.warnLabel.setStyleSheet("QLabel { color : #800000; }")
            self.warnoise = QLabel("")
            self.warnoise.setStyleSheet("QLabel { color : #800000; }")
            self.msgseg = QLabel("")
            self.msgseg.setFixedWidth(600)
            self.msgseg.setWordWrap(True)
            self.msgseg.setStyleSheet("QLabel { color : #808080; }")
            lblmsgseg = QLabel("<b>Segments detected<b>")
            lblmsgseg.setStyleSheet("QLabel { color : #808080; }")
            self.warnseg = QLabel("")
            self.warnseg.setStyleSheet("QLabel { color : #800000; }")
            space = QLabel()
            space.setFixedHeight(20)
            space.setFixedWidth(10)

            # page layout
            layout = QGridLayout()
            layout.addWidget(QLabel("<b>Selected TRAINING data</b>"), 0, 0)
            layout.addWidget(self.msgadir, 0, 2)
            layout.addWidget(self.warnnoannt2, 1, 2)
            layout.addWidget(self.msgmdir, 2, 2)
            layout.addWidget(self.warnnoannt1, 3, 2)
            layout.addWidget(space, 4, 0)
            layout.addWidget(self.warnoise, 5, 2)
            layout.addWidget(lblmsgseg, 6, 2)
            layout.addWidget(self.msgseg, 7, 2)
            layout.addWidget(self.warnseg, 8, 2)
            layout.addWidget(space, 9, 0)
            layout.addWidget(space, 12, 0)
            layout.addWidget(QLabel("<b>Selected Recogniser</b>"), 13, 0)
            layout.addWidget(self.msgrecfilter, 13, 2)
            layout.addWidget(self.msgrecspp, 14, 2)
            layout.addWidget(self.msgreccts, 15, 2)
            layout.addWidget(self.msgrecclens, 16, 2)
            layout.addWidget(self.msgrecfs, 17, 2)
            layout.addWidget(self.msgrecfrange, 18, 2)
            layout.addWidget(self.warnLabel, 19, 2)
            layout.addWidget(self.imgDirwarn, 20, 2)
            self.setLayout(layout)

        def initializePage(self):
            self.certainty1 = True
            self.certainty2 = True

            with pg.BusyCursor():
                self.nntrain.readFilter()

            # Error checking

            # Check if it already got a NN model
            if "NN" in self.nntrain.currfilt:
                self.warnLabel.setText("Warning: This recogniser already has a NN.")
            else:
                self.warnLabel.setText("")

            warn = ""
            # Check the annotation certainty
            if self.field("trainDir1"):
                self.certainty1 = self.getCertainty(self.field("trainDir1"))
                if not self.certainty1:
                    warn += "Warning: Detected uncertain segments\n"

            # Check if there are annotations from the target species at all
            if self.field("trainDir1"):
                if self.nntrain.species not in self.wizard().browsedataPage.splist1:
                    warn += "Warning: No annotations of " + self.nntrain.species + " detected\n"
                    self.hasant1 = False
                else:
                    self.hasant1 = True

            self.warnnoannt1.setText(warn)

            warn = ""
            # Check the annotation certainty
            if self.field("trainDir2"):
                self.certainty2 = self.getCertainty(self.field("trainDir2"))
                if not self.certainty2:
                    warn += "Warning: Detected uncertain segments\n"

            # Check if there are annotations from the target species at all
            if self.field("trainDir2"):
                if self.nntrain.species not in self.wizard().browsedataPage.splist2:
                    warn += "Warning: No annotations of " + self.nntrain.species + " detected\n"
                    self.hasant2 = False
                else:
                    self.hasant2 = True

            self.warnnoannt2.setText(warn)

            if self.field("trainDir1"):
                self.msgmdir.setText("<b>Manually annotated:</b> %s" % (self.field("trainDir1")))
            if self.field("trainDir2"):
                self.msgadir.setText("\n<b>Auto processed and reviewed:</b> %s" % (self.field("trainDir2")))

            # Get training data
            with pg.BusyCursor():
                self.nntrain.genSegmentDataset(self.hasant1)

            self.msgrecfilter.setText("<b>Recogniser:</b> %s" % (self.field("filter")))
            self.msgrecspp.setText("<b>Species:</b> %s" % (self.nntrain.species))
            self.msgreccts.setText("<b>Call types:</b> %s" % (self.nntrain.calltypes))
            self.msgrecclens.setText("<b>Call length:</b> %.2f - %.2f sec" % (self.nntrain.mincallength, self.nntrain.maxcallength))
            self.msgrecfs.setText("<b>Sample rate:</b> %d Hz" % (self.nntrain.fs))
            self.msgrecfrange.setText("<b>Frequency range:</b> %d - %d Hz" % (self.nntrain.f1, self.nntrain.f2))

            for i in range(len(self.nntrain.calltypes)):
                self.msgseg.setText("%s:\t%d\t" % (self.msgseg.text() + self.nntrain.calltypes[i], self.nntrain.trainN[i]))
            self.msgseg.setText("%s:\t%d" % (self.msgseg.text() + "Noise", self.nntrain.trainN[-1]))

            # We need at least some number of segments from each class to proceed
            if min(self.nntrain.trainN) < self.LearningDict['minPerClass']:
                print('Warning: Need at least %d segments from each class to train NN' % self.LearningDict['minPerClass'])
                self.warnseg.setText('<b>Warning: Need at least %d segments from each class to train NN\n\n</b>' % self.LearningDict['minPerClass'])

            if not self.nntrain.correction and self.wizard().browsedataPage.anntlevel == 'Some':
                self.warnoise.setText('Warning: No segments found for Noise class\n(no correction segments/fully (manual) annotations)')

            freeGB,totalbytes = self.nntrain.checkDisk()

            if freeGB < 10:
                self.imgDirwarn.setText('Warning: Free space in the user directory is %.2f GB/ %.2f GB, you may run out of space' % (freeGB, totalbytes))

        def getCertainty(self, dirname):
            minCertainty = 100
            for root, dirs, files in os.walk(dirname):
                for file in files:
                    soundFile = os.path.join(root, file)
                    if (file.lower().endswith('.wav') or file.lower().endswith('.flac')) and os.stat(soundFile).st_size != 0 and file + '.data' in files:
                        segments = annotation.SegmentList()
                        segments.parseJSON(soundFile + '.data')
                        cert = [lab["certainty"] if lab["species"] == self.nntrain.species else 100 for seg in segments for lab in seg.labels]
                        if cert:
                            mincert = min(cert)
                            if minCertainty > mincert:
                                minCertainty = mincert
            if minCertainty < 100:
                return False
            else:
                return True

        def cleanupPage(self):
            self.imgDirwarn.setText('')
            self.msgmdir.setText('')
            self.msgadir.setText('')
            self.warnnoannt1.setText('')
            self.warnLabel.setText('')
            self.warnnoannt2.setText('')
            self.warnoise.setText('')
            self.msgseg.setText('')
            self.warnseg.setText('')
            self.msgrecfilter.setText('')
            self.msgrecspp.setText('')
            self.msgreccts.setText('')
            self.msgrecclens.setText('')
            self.msgrecfs.setText('')

        def isComplete(self):
            return (self.hasant1 or self.hasant2) and min(self.nntrain.trainN) >= self.LearningDict['minPerClass']

    # page 3 - set parameters, generate data and train
    class WPageParameters(QWizardPage):
        def __init__(self, config, parent=None):
            super(BuildNNWizard.WPageParameters, self).__init__(parent)
            self.setTitle('Choose call length and model')
            self.setSubTitle('When ready, press \"Generate NN images and Train\" to start preparing data for NN and training.\nThe process may take a long time.')

            self.setMinimumSize(350, 200)
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            self.adjustSize()

            # self.nntrain = nntrain
            self.config = config
            self.indx = np.ndarray(0)

            # Make pages to plot ROC for each call type OR automatically select thresholds
            self.redopages = True

            # Parameter/s
            self.imgsec = CustomSlider(Qt.Orientation.Horizontal)
            self.imgsec.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.imgsec.setTickInterval(25)
            self.imgsec.setRange(0, 600)  # 0-6 sec
            self.imgsec.setValue(25)
            self.imgsec.valueChanged.connect(self.imglenChange)
            self.imgsec.sliderClicked.connect(self.reloadImgs)
            self.imgsec.sliderReleased.connect(self.reloadImgs)

            self.imgtext = QLabel('0.25 sec')

            self.cbfrange = QCheckBox("Limit frequency range")
            self.cbfrange.setStyleSheet("QCheckBox { font-weight: bold; }")
            self.cbfrange.toggled.connect(self.onClickedFrange)

            self.f1 = CustomSlider(Qt.Orientation.Horizontal)
            self.f1.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.f1.setTickInterval(1000)
            # self.f1.setRange(0, self.nntrain.fs)  # 0-6 sec
            # self.f1.setValue(self.nntrain.f1)
            # self.f1.valueChanged.connect(self.f1Change)
            self.f1text = QLabel('')

            self.f2 = CustomSlider(Qt.Orientation.Horizontal)
            self.f2.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.f2.setTickInterval(1000)
            # self.f2.setRange(0, self.nntrain.fs)  # 0-6 sec
            # self.f2.setValue(self.nntrain.f2)
            # self.f2.valueChanged.connect(self.f2Change)
            self.f2text = QLabel('')

            self.modelArchitecture = QComboBox()
            self.modelArchitecture.addItems(["CNN"]) #,"AudioSpectogramTransformer","AudioSpectogramTransformer (pre-trained ViT)"])

            space = QLabel()
            space.setFixedSize(10, 30)
            msglayout = QVBoxLayout()
            self.msgspp = QLabel('')
            self.msgspp.setStyleSheet("QLabel { color : #808080; }")
            self.msgtrain1 = QLabel('')
            self.msgtrain1.setStyleSheet("QLabel { color : #808080; }")
            self.msgtrain2 = QLabel('')
            self.msgtrain2.setStyleSheet("QLabel { color : #808080; }")
            self.msgtest1 = QLabel('')
            self.msgtest1.setStyleSheet("QLabel { color : #808080; }")
            msglayout.addWidget(self.msgspp)
            msglayout.addWidget(self.msgtrain2)
            msglayout.addWidget(self.msgtrain1)
            msglayout.addWidget(self.msgtest1)

            layout0 = QVBoxLayout()
            layout0.addLayout(msglayout)
            # layout0.addWidget(space)
            layout0.addWidget(QLabel('<b>Choose call length (sec) you want to show to NN</b>'))
            layout0.addWidget(QLabel('Make sure an image covers at least couple of syllables when appropriate'))
            # layout0.addWidget(space)
            layout0.addWidget(self.imgtext)
            layout0.addWidget(self.imgsec)
            layout0.addWidget(self.cbfrange)
            layout0a = QHBoxLayout()
            layout0a1 = QVBoxLayout()
            # layout0a1.addWidget(QLabel('Lower frq. limit (Hz)'))
            layout0a1.addWidget(self.f1text)
            layout0a1.addWidget(self.f1)
            layout0a2 = QVBoxLayout()
            # layout0a2.addWidget(QLabel('Upper frq. limit (Hz)'))
            layout0a2.addWidget(self.f2text)
            layout0a2.addWidget(self.f2)
            layout0a.addLayout(layout0a1)
            layout0a.addLayout(layout0a2)
            layout0.addLayout(layout0a)

            layout2 = QVBoxLayout()
            layout2.addWidget(QLabel('<i>Example images from your dataset</i>'))
            self.flowLayout = QHBoxLayout()
            self.img1 = QLabel()
            self.img1.setFixedHeight(175)
            self.img2 = QLabel()
            self.img2.setFixedHeight(175)
            self.img3 = QLabel()
            self.img3.setFixedHeight(175)
            self.flowLayout.addWidget(self.img1)
            self.flowLayout.addWidget(self.img2)
            self.flowLayout.addWidget(self.img3)
            layout2.addLayout(self.flowLayout)

            self.cbAutoThr = QCheckBox("Tick if you want AviaNZ to decide threshold/s")
            self.cbAutoThr.setStyleSheet("QCheckBox { font-weight: bold; }")
            self.cbAutoThr.toggled.connect(self.onClicked)

            layout3 = QVBoxLayout()
            layout3.addWidget(QLabel('<b>Choose model architecture</b>'))
            layout3.addWidget(self.modelArchitecture)

            layout1 = QVBoxLayout()
            layout1.addLayout(layout0)
            layout1.addLayout(layout2)
            layout1.addWidget(self.cbAutoThr)
            layout1.addLayout(layout3)
            self.setLayout(layout1)
            self.setButtonText(QWizard.WizardButton.NextButton, 'Generate NN images and Train>')

        def initializePage(self):
            self.nntrain = self.wizard().confirminputPage.nntrain
            self.nntrain.windowWidth = 512
            self.nntrain.windowInc = 256
            self.f1.setRange(0, self.nntrain.fs//2)
            self.f1.setValue(0)
            self.f1text.setText('Lower frq. limit 0 Hz')
            self.f2.setRange(0, self.nntrain.fs//2)
            self.f2.setValue(self.nntrain.fs//2)
            self.f2text.setText('Upper frq. limit ' + str(self.nntrain.fs//2) + ' Hz')
            self.f1.valueChanged.connect(self.f1Change)
            self.f2.valueChanged.connect(self.f2Change)
            self.f1.sliderClicked.connect(self.reloadImgs)
            self.f1.sliderReleased.connect(self.reloadImgs)
            self.f2.sliderClicked.connect(self.reloadImgs)
            self.f2.sliderReleased.connect(self.reloadImgs)
            self.cbfrange.setChecked(False)
            self.f1text.setEnabled(False)
            self.f1.setEnabled(False)
            self.f2text.setEnabled(False)
            self.f2.setEnabled(False)
            self.modelArchitecture.currentTextChanged.connect(self.modelArchitectureChange)

            self.wizard().button(QWizard.WizardButton.NextButton).setDefault(False)
            self.msgspp.setText("<b>Species:</b> %s" % (self.nntrain.species))

            if self.field("trainDir1"):
                self.msgtrain1.setText("<b>Training data (Manually annotated):</b> %s" % (self.field("trainDir1")))
            if self.field("trainDir2"):
                self.msgtrain2.setText("<b>Training data (Auto processed and reviewed):</b> %s" % (self.field("trainDir2")))

            # Ideally, the image length should be bigger than the max gap between syllables
            if np.max(self.nntrain.maxgaps) * 2 <= 6:
                self.imgtext.setText(str(np.max(self.nntrain.maxgaps) * 2) + ' sec')
                self.imgsec.setValue(int(np.max(self.nntrain.maxgaps) * 2 * 100))
            elif np.max(self.nntrain.maxgaps) * 1.5 <= 6:
                self.imgtext.setText(str(np.max(self.nntrain.maxgaps) * 1.5) + ' sec')
                self.imgsec.setValue(int(np.max(self.nntrain.maxgaps) * 1.5 * 100))
            elif np.max(self.nntrain.mincallength) <= 6:
                self.imgtext.setText(str(np.max(self.nntrain.mincallength)) + ' sec')
                self.imgsec.setValue(int(np.max(self.nntrain.mincallength) * 100))
            self.nntrain.imgWidth = self.imgsec.value() / 100

            self.setWindowInc()
            self.showimg()
            self.completeChanged.emit()

        def onClicked(self):
            cbutton = self.sender()
            if cbutton.isChecked():
                self.nntrain.autoThr = True
            else:
                self.nntrain.autoThr = False
            self.redopages = True
            self.completeChanged.emit()

        def onClickedFrange(self):
            cbutton = self.sender()
            if cbutton.isChecked():
                self.f1.setEnabled(True)
                self.f1text.setEnabled(True)
                self.f2.setEnabled(True)
                self.f2text.setEnabled(True)
                if self.f1.value() == 0 and self.f2.value() == self.nntrain.fs/2:
                    self.f1.setValue(self.nntrain.f1)
                    self.f2.setValue(self.nntrain.f2)
                    self.f1text.setText('Lower frq. limit ' + str(self.nntrain.f1) + ' Hz')
                    self.f2text.setText('Upper frq. limit ' + str(self.nntrain.f2) + ' Hz')
            else:
                self.f1.setValue(0)
                self.f2.setValue(self.nntrain.fs/2)
                self.f1text.setText('Lower frq. limit ' + str(0) + ' Hz')
                self.f2text.setText('Upper frq. limit ' + str(self.nntrain.fs/2) + ' Hz')
                self.f1.setEnabled(False)
                self.f1text.setEnabled(False)
                self.f2.setEnabled(False)
                self.f2text.setEnabled(False)

        def showimg(self, indices=[]):
            ''' Show example spectrogram (random ct segments from train dataset)
            '''
            i = 0
            # SM
            #trainsegments = self.nntrain.trainsegments
            if len(indices) == 0:
                target = [rec[-1] for rec in self.nntrain.traindata]
                indxs = [list(np.where(np.array(target) == i)[0]) for i in range(len(self.nntrain.calltypes))]
                indxs = [i for sublist in indxs for i in sublist]
                self.indx = np.random.choice(indxs, 3, replace=False)
            else:
                self.indx = indices
            for ind in self.indx:
                filename=self.nntrain.traindata[ind][0]
                duration = self.imgsec.value()/100
                if duration == 0:
                    duration = None
                offset = self.nntrain.traindata[ind][1][0]
                sp = spectrogram.Spectrogram(self.nntrain.windowWidth, self.nntrain.windowInc)
                sp.readSoundFile(filename, duration, offset)
                sp.resample(self.nntrain.fs)
                sp.audio_data.sample_rate = self.nntrain.fs
                sgRaw = sp.spectrogram(self.nntrain.windowWidth, self.nntrain.windowInc)
                sgRaw = sgRaw[:self.nntrain.imgsize[0]]
                # Frequency masking
                f1 = self.f1.value()
                f2 = self.f2.value()
                # Mask out of band elements
                bin_width = self.nntrain.fs / 2 / np.shape(sgRaw)[1]
                lb = int(np.ceil(f1 / bin_width))
                ub = int(np.floor(f2 / bin_width))
                maxsg = np.min(sgRaw)
                sgRaw[:, 0:lb] = 0.0
                sgRaw[:, ub:] = 0.0
                sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))

                # determine colour map
                self.lut = colourMaps.getLookupTable(self.config['cmap'])

                picbtn = PicButton(1, np.fliplr(sg), sp, sp.audio_data, self.imgsec.value(), 0, 0, self.lut, cluster=True)
                if i == 0:
                    pic = QPixmap.fromImage(picbtn.im1)
                    self.img1.setPixmap(pic.scaledToHeight(175))
                    self.flowLayout.update()
                    i += 1
                elif i == 1:
                    pic = QPixmap.fromImage(picbtn.im1)
                    self.img2.setPixmap(pic.scaledToHeight(175))
                    self.flowLayout.update()
                    i += 1
                elif i == 2:
                    pic = QPixmap.fromImage(picbtn.im1)
                    self.img3.setPixmap(pic.scaledToHeight(175))
                    self.flowLayout.update()
                    i += 1
                else:
                    break
            if i == 0:
                self.img1.setText('<no image to show>')
                self.img2.setText('')
                self.img3.setText('')
                self.flowLayout.update()

        def setWindowInc(self):
            """
                What is the increment such that we get the correct image width?
            """
            self.nntrain.windowWidth = self.nntrain.imgsize[0] * 2
            duration = self.imgsec.value()/100
            totalLength = duration * self.nntrain.fs
            lengthIgnoringLast = totalLength - self.nntrain.windowWidth
            gapRequiredToFitOthers = lengthIgnoringLast / (self.nntrain.imgsize[1] - 1) # minus 1 as we ignore the last one
            self.nntrain.windowInc = int(np.floor(gapRequiredToFitOthers))
            print('window and increment set: ', self.nntrain.windowWidth, self.nntrain.windowInc)

        def imglenChange(self):
            value = self.imgsec.value()
            if value < 10:
                self.imgsec.setValue(10)
                self.imgtext.setText('0.1 sec')
            else:
                self.imgtext.setText(str(value / 100) + ' sec')
            self.nntrain.imgWidth = self.imgsec.value()/100

        def f1Change(self):
            value = self.f1.value()
            value = value//10*10
            if value < 0:
                value = 0
            # self.nntrain.f1 = value
            self.f1text.setText('Lower frq. limit ' + str(value) + ' Hz')

        def f2Change(self):
            value = self.f2.value()
            value = value//10*10
            if value < 0:
                value = 0
            # self.nntrain.f2 = value
            self.f2text.setText('Upper frq. limit ' + str(value) + ' Hz')
        
        def reloadImgs(self):
            self.setWindowInc()
            self.showimg(self.indx)
        
        def modelArchitectureChange(self):
            print("setting training architecture to ",self.modelArchitecture.currentText())
            self.nntrain.modelArchitecture = self.modelArchitecture.currentText()

        def cleanupPage(self):
            self.img1.setText('')
            self.img2.setText('')
            self.img3.setText('')

        def validatePage(self):
            with pg.BusyCursor():
                self.nntrain.f1 = self.f1.value()
                self.nntrain.f2 = self.f2.value()
                self.nntrain.train()
            return True

        def isComplete(self):
            if self.img1.text() == '<no image to show>':
                return False
            if self.redopages:
                self.redopages = False
                self.wizard().redoROCPages(self.nntrain)
            return True

    # page 4 - ROC curve
    class WPageROC(QWizardPage):
        def __init__(self, nntrain, ct, parent=None):
            super(BuildNNWizard.WPageROC, self).__init__(parent)
            self.setTitle('Training results')
            self.setSubTitle('Click on the graph at the point where you would like the classifier to trade-off false positives with false negatives. Points closest to the top-left are best.')
            self.setMinimumSize(350, 200)
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            self.adjustSize()

            self.nntrain = nntrain
            self.ct = ct

            self.lblSpecies = QLabel()
            self.lblSpecies.setStyleSheet("QLabel { color : #808080; }")
            self.lblCalltype = QLabel()
            self.lblCalltype.setStyleSheet("QLabel { color : #808080; }")
            self.lblUpdate = QLabel()

            self.layout = QGridLayout()
            self.layout.addWidget(self.lblSpecies, 0, 0)
            self.layout.addWidget(self.lblCalltype, 1, 0)
            self.setLayout(self.layout)

        def initializePage(self):
            # self.nntrain = self.wizard().parameterPage.nntrain
            self.thrs = self.nntrain.Thrs
            self.TPR = self.nntrain.TPRs[self.ct]
            self.FPR = self.nntrain.FPRs[self.ct]
            self.Precision = self.nntrain.Precisions[self.ct]
            self.Acc = self.nntrain.Accs[self.ct]
            print('ROC page, TPR: ', self.TPR)
            print('ROC page, FPR: ', self.FPR)

            # This is the Canvas Widget that displays the plot
            self.figCanvas = ROCCanvas(self)
            self.figCanvas.plotme()

            self.marker = self.figCanvas.ax.plot([0, 1], [0, 1], marker='o', color='black', linestyle='dotted')[0]
            # self.marker.set_visible(False)
            self.figCanvas.plotmeagain(self.TPR, self.FPR, NN=True)

            if self.ct == len(self.nntrain.calltypes):
                self.lblCalltype.setText('Noise (treat same as call types)')
            else:
                self.lblCalltype.setText('Call type: ' + self.nntrain.calltypes[self.ct])
            self.lblSpecies.setText('Species: ' + self.nntrain.species)

            # Figure click handler
            def onclick(event):
                fpr_cl = event.xdata
                tpr_cl = event.ydata
                if tpr_cl is None or fpr_cl is None:
                    return

                # Get thr for closest point
                distarr = (tpr_cl - self.TPR) ** 2 + (fpr_cl - self.FPR) ** 2
                thr_min_ind = np.unravel_index(np.argmin(distarr), distarr.shape)[0]
                tpr_near = self.TPR[thr_min_ind]
                fpr_near = self.FPR[thr_min_ind]
                self.marker.set_visible(False)
                self.figCanvas.draw()
                self.marker.set_xdata([fpr_cl, fpr_near])
                self.marker.set_ydata([tpr_cl, tpr_near])
                self.marker.set_visible(True)
                self.figCanvas.ax.draw_artist(self.marker)
                self.figCanvas.update()

                print("fpr_cl, tpr_cl: ", fpr_near, tpr_near)

                # Update sidebar info
                self.lblUpdate.setText(
                    '\tTrue Positive Rate: %.2f\n\tFalse Positive Rate: %.2f\n\tPrecision: %.2f\n\tAccuracy: %.2f' % (
                        self.TPR[thr_min_ind], self.FPR[thr_min_ind], self.Precision[thr_min_ind], self.Acc[thr_min_ind]))

                # This will save the best lower thr
                self.nntrain.bestThr[self.ct][0] = self.nntrain.thrs[thr_min_ind]
                self.nntrain.bestThrInd[self.ct] = thr_min_ind

                self.completeChanged.emit()

            self.figCanvas.figure.canvas.mpl_connect('button_press_event', onclick)

            self.layout.addWidget(self.figCanvas, 2, 0)
            self.layout.addWidget(self.lblUpdate, 2, 1)

        def cleanupPage(self):
            pass
            #try:
                #self.wizard().parameterPage.tmpdir1.cleanup()
                #self.wizard().parameterPage.tmpdir2.cleanup()
            #except:
                #pass

        def isComplete(self):
            if self.lblUpdate.text() == '':
                return False
            else:
                return True

    # page 5 - Summary
    class WPageSummary(QWizardPage):
        def __init__(self, nntrain, parent=None):
            super(BuildNNWizard.WPageSummary, self).__init__(parent)
            self.setTitle('Training Summary')
            self.setSubTitle('If you are happy with the NN performance, press \"Save the Recogniser.\"')
            self.setMinimumSize(250, 150)
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            self.adjustSize()

            self.nntrain = nntrain

            self.space = QLabel('').setFixedSize(20, 20)
            self.msgfilter = QLabel('')
            self.msgfilter.setStyleSheet("QLabel { color : #808080; }")
            self.msgspp = QLabel('')
            self.msgspp.setStyleSheet("QLabel { color : #808080; }")

            # page layout
            self.layout = QGridLayout()
            self.layout.addWidget(self.msgfilter, 0, 0)
            self.layout.addWidget(self.msgspp, 1, 0)
            self.layout.addWidget(self.space, 2, 0)

            self.setButtonText(QWizard.WizardButton.NextButton, 'Save the Recogniser>')
            self.setLayout(self.layout)

        def initializePage(self):
            self.msgfilter.setText("<b>Current recogniser:</b> %s" % (self.field("filter")))
            self.msgspp.setText("<b>Species:</b> %s" % (self.nntrain.species))

            row = 3
            for ct in range(len(self.nntrain.calltypes)):
                lblct = QLabel('Call type: ' + self.nntrain.calltypes[ct])
                lblct.setStyleSheet("QLabel { color : #808080; font-weight: bold; }")
                self.layout.addWidget(lblct, row, 0, alignment=Qt.AlignmentFlag.AlignTop)
                lblctsumy = QLabel('True Positive Rate: %.2f\nFalse Positive Rate: %.2f\nPrecision: %.2f\nAccuracy: %.2f'
                                   % (self.nntrain.TPRs[ct][self.nntrain.bestThrInd[ct]],
                                      self.nntrain.FPRs[ct][self.nntrain.bestThrInd[ct]],
                                      self.nntrain.Precisions[ct][self.nntrain.bestThrInd[ct]],
                                      self.nntrain.Accs[ct][self.nntrain.bestThrInd[ct]]))
                lblctsumy.setStyleSheet("QLabel { color : #808080; }")
                self.layout.addWidget(self.space, row, 1)
                self.layout.addWidget(lblctsumy, row, 2)
                row += 1
            self.layout.update()

        def cleanupPage(self):
            wgts = []
            for ct in range(len(self.nntrain.calltypes) ):
                if self.layout.itemAtPosition(ct+3, 0):
                    wgts.append(self.layout.itemAtPosition(ct+3, 0).widget())
                if self.layout.itemAtPosition(ct+3, 1):
                    wgts.append(self.layout.itemAtPosition(ct+3, 1).widget())
                if self.layout.itemAtPosition(ct+3, 2):
                    wgts.append(self.layout.itemAtPosition(ct+3, 2).widget())

            for i in reversed(range(len(wgts))):
                self.layout.removeWidget(wgts[i])
                wgts[i].deleteLater()
                del wgts[i]

    # page 6 - Save Filter
    class WPageSave(QWizardPage):
        def __init__(self, nntrain, parent=None):
            super(BuildNNWizard.WPageSave, self).__init__(parent)
            self.setTitle('Save Recogniser')
            self.setSubTitle('If you are happy with the NN performance, save the recogniser.')
            self.setMinimumSize(250, 150)
            self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            self.adjustSize()

            self.nntrain = nntrain
            self.filterfile = ''
            self.saveoption = 'New'

            # filter dir listbox
            self.listFiles = QListWidget()
            self.listFiles.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            self.listFiles.setMinimumHeight(200)
            filtdir = QDir(self.nntrain.filterdir).entryList(filters=QDir.Filter.NoDotAndDotDot | QDir.Filter.Files)
            for file in filtdir:
                if file.endswith('.txt'):
                    item = QListWidgetItem(self.listFiles)
                    item.setText(file)
            # filter file name
            self.enterFiltName = QLineEdit()
            self.enterFiltName.textChanged.connect(self.textChanged)

            trainFiltValid = FiltValidator(self.listFiles, check_reserved_m=True)
            self.enterFiltName.setValidator(trainFiltValid)
            space = QLabel('').setFixedSize(30, 50)

            self.msgfilter = QLabel('')
            self.msgfilter.setStyleSheet("QLabel { color : #808080; }")
            self.warnfilter = QLabel('')
            self.warnfilter.setStyleSheet("QLabel { color : #800000; }")
            self.msgspp = QLabel('')
            self.msgspp.setStyleSheet("QLabel { color : #808080; }")

            self.rbtn1 = QRadioButton('New recogniser (enter name below)')
            self.rbtn1.setChecked(True)
            self.rbtn1.val = "New"
            self.rbtn1.toggled.connect(self.onClicked)
            self.rbtn2 = QRadioButton('Update existing')
            self.rbtn2.val = "Update"
            self.rbtn2.toggled.connect(self.onClicked)

            # page layout
            layout = QGridLayout()
            layout.addWidget(self.msgfilter, 0, 0)
            layout.addWidget(self.warnfilter, 0, 1)
            layout.addWidget(self.msgspp, 1, 0)
            layout.addWidget(space, 2, 0)
            layout.addWidget(QLabel('<b>How do you want to save it?</b>'), 3, 0)
            layout.addWidget(self.rbtn1, 4, 1)
            layout.addWidget(self.rbtn2, 5, 1)
            layout.addWidget(space, 6, 0)
            layout.addWidget(QLabel("<i>Currently available recognisers</i>"), 7, 0)
            layout.addWidget(self.listFiles, 8, 0, 1, 2)
            layout.addWidget(space, 9, 0)
            layout.addWidget(QLabel("Enter file name if you choose to save the recogniser as a new file (must be unique)"), 10, 0, 1, 2)
            layout.addWidget(self.enterFiltName, 12, 0, 1, 2)

            self.setButtonText(QWizard.WizardButton.FinishButton, 'Save and Finish')
            self.setLayout(layout)

        def initializePage(self):
            self.msgfilter.setText("<b>Current recogniser:</b> %s" % (self.field("filter")))
            if "NN" in self.nntrain.currfilt:
                self.warnfilter.setText("Warning: The recogniser already has a NN.")
            self.msgspp.setText("<b>Species:</b> %s" % (self.nntrain.species))
            self.rbtn2.setText('Update existing (' + self.field("filter") + ')')

            self.nntrain.addNNFilter()

            self.wizard().saveTestBtn.setVisible(True)
            self.wizard().saveTestBtn.setEnabled(False)
            self.completeChanged.emit()

        def refreshCustomBtn(self):
            if self.isComplete():
                self.wizard().saveTestBtn.setEnabled(True)
            else:
                self.wizard().saveTestBtn.setEnabled(False)
            self.completeChanged.emit()

        def onClicked(self):
            radioBtn = self.sender()
            if radioBtn.isChecked():
                self.saveoption = radioBtn.val
            self.refreshCustomBtn()
            self.completeChanged.emit()

        def textChanged(self, text):
            self.refreshCustomBtn()
            self.completeChanged.emit()

        # def validatePage(self):
        #     with pg.BusyCursor():
        #         self.nntrain.saveFilter()
        #     return True

        def cleanupPage(self):
            self.wizard().saveTestBtn.setEnabled(False)
            self.enterFiltName.setText('')
            self.rbtn1.setChecked(True)
            self.saveoption = "New"
            self.wizard().saveTestBtn.setVisible(False)

        def isComplete(self):
            if self.saveoption == 'New' and self.enterFiltName.text() != '' and self.enterFiltName.text() != '.txt':
                self.nntrain.setP6(self.enterFiltName.text())
                return True
            elif self.saveoption == "Update":
                # SM
                self.nntrain.setP6(self.enterFiltName.text())
                return True
            else:
                return False

    def redoROCPages(self, nntrain):
        # clean any existing pages
        for page in self.rocpages:
            # for each calltype, remove roc page
            self.removePage(page)
        self.rocpages = []

        if not nntrain.autoThr:
            for i in range(len(nntrain.calltypes)):
                print("adding ROC page for class:", nntrain.calltypes[i])
                page4 = BuildNNWizard.WPageROC(nntrain, i)
                pageid = self.addPage(page4)
                self.rocpages.append(pageid)

        self.summaryPage = BuildNNWizard.WPageSummary(nntrain)
        pageid = self.addPage(self.summaryPage)
        self.rocpages.append(pageid)

        self.savePage = BuildNNWizard.WPageSave(nntrain)
        pageid = self.addPage(self.savePage)
        self.rocpages.append(pageid)

        self.parameterPage.setFinalPage(False)
        self.parameterPage.completeChanged.emit()

    def undoROCPages(self): # TODO: not using, delete
        # clean any existing pages
        for page in self.rocpages:
            # for each calltype, remove roc page
            self.removePage(page)
        self.rocpages = []

        self.summaryPage = BuildNNWizard.WPageSummary(self.nntrain)
        self.addPage(self.summaryPage)

        self.savePage = BuildNNWizard.WPageSave(self.nntrain)
        self.addPage(self.savePage)

        self.parameterPage.setFinalPage(False)
        self.parameterPage.completeChanged.emit()