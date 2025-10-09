
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
import time
import platform
import copy
from shutil import copyfile
import json

from PyQt6.QtGui import QIcon, QValidator, QPixmap, QColor
from PyQt6.QtCore import QDir, Qt, QEvent, QSize, pyqtSignal
from PyQt6.QtWidgets import QLabel, QSlider, QPushButton, QListWidget, QListWidgetItem, QComboBox, QDialog, QWizard, QWizardPage, QLineEdit, QSizePolicy, QFormLayout, QVBoxLayout, QHBoxLayout, QCheckBox, QLayout, QApplication, QRadioButton, QGridLayout, QFileDialog, QScrollArea, QWidget, QAbstractItemView

import matplotlib.markers as mks
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg

import numpy as np
from src.ui.colourMaps import colourMaps
from src.ui.components.buttons_and_controls import BrightContrVol, CustomSlider, PicButton
from src.ui.components.popups import MessagePopup
from src.ui.components.file_list import LightedFileList
from src.ui.components.layout_widgets import Layout
from src.core import SupportClasses
from src.core import Spectrogram
from src.core import WaveletSegment
from src.core import WaveletFunctions
from src.core import Segment
from src.core import Clustering
from src.core import Training

from src.models import NNModels

import math

class FilterCustomiseROC(QDialog):
    class LabelSlider(QWidget):
        valueChanged = pyqtSignal()
        # Creates a 0.001 precision slider with a label
        # args:
        # initial: initial value for the label & slider
        # minimum-maximum: range for the slider, if chosen
        # slider: bool, if False, only adds Qlabels
        def __init__(self, initial, minimum=0, maximum=0, slider=False, parent=None):
            super(FilterCustomiseROC.LabelSlider, self).__init__(parent)

            self.oldval = round(initial*1000)/1000 # store for comparison.
            # Storing as string to allow easy comparison w/ self.lbl

            self.lbl = QLabel(str(self.oldval))
            self.lbl.setMinimumWidth(40)

            oldlbl = QLabel("(current: %s)" % self.oldval)
            oldlbl.setMinimumWidth(40)
            oldlbl.setStyleSheet("QLabel { color: #808080}")

            if slider:
                slid = QSlider(Qt.Orientation.Horizontal)
                slid.setMinimum(int(minimum*1000))
                slid.setMaximum(int(maximum*1000))
                slid.setValue(round(initial*1000))
                slid.setTickInterval(1000)
                slid.setTickPosition(QSlider.TickPosition.TicksBelow)
                slid.valueChanged.connect(self.updatelbl)

            box = QHBoxLayout()
            box.setContentsMargins(0, 0, 0, 0)
            if slider:
                box.addWidget(slid)
            box.addWidget(self.lbl)
            box.addWidget(oldlbl)
            self.setLayout(box)

        def updatelbl(self, value):
            self.lbl.setText(str(value/1000))
            self.valueChanged.emit()

        def value(self):
            return(float(self.lbl.text()))

        def setValue(self, value):
            self.lbl.setText(str(value))

        def hasChanged(self):
            return(self.value()!=self.oldval)

    def __init__(self, filtdir, parent=None):
        super(FilterCustomiseROC, self).__init__(parent)
        self.setWindowTitle("Customise a recogniser (use existing ROC)")
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))

        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.filtdir = filtdir
        self.saveoption = "New"
        self.ROCWF = False
        self.RONN = False
        self.newthr = 0
        self.calltypes = []
        self.form = QGridLayout()
        self.form.setSpacing(25)

        # filter dir contents
        self.listFiles = QListWidget()
        self.listFiles.setFixedWidth(300)
        self.listFiles.setFixedHeight(450)
        self.listFiles.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.listFiles.itemSelectionChanged.connect(self.readFilter)

        self.readContents()

        # new filter name
        self.enterFiltName = QLineEdit()
        self.btnSave = QPushButton('Save')

        class FiltValidator(QValidator):
            def validate(self, input, pos):
                if not input.endswith('.txt'):
                    input = input+'.txt'
                if input==".txt" or input=="":
                    return(QValidator.State.Intermediate, input, pos)
                elif input=="M.txt":
                    print("filter name \"M\" reserved for manual annotations")
                    return(QValidator.State.Intermediate, input, pos)
                elif self.listFiles.findItems(input, Qt.MatchFlag.MatchExactly):
                    print("duplicated input", input)
                    return(QValidator.State.Intermediate, input, pos)
                else:
                    return(QValidator.State.Acceptable, input, pos)

        renameFiltValid = FiltValidator()
        renameFiltValid.listFiles = self.listFiles
        self.enterFiltName.setValidator(renameFiltValid)

        self.listFiles.itemSelectionChanged.connect(self.onFilterSelect)
        self.enterFiltName.textChanged.connect(self.refreshSaveButton)

        # layouts
        self.rbtn1 = QRadioButton('New recogniser (enter a unique name):')
        self.rbtn2 = QRadioButton('Update existing recogniser')
        self.rbtn1.toggled.connect(self.onClicked)
        self.lblsave1 = QLabel('How do you want to save the changes?')

        savegrid = QGridLayout()
        savegrid.addWidget(self.lblsave1, 0, 0, 2, 1)
        savegrid.addWidget(self.rbtn1, 0, 1)
        savegrid.addWidget(self.rbtn2, 1, 1)
        savegrid.addWidget(self.enterFiltName, 0, 2)
        savegrid.addWidget(self.btnSave, 1, 2)
        savegrid.setColumnStretch(2, 3)

        self.recgrid = QGridLayout()
        self.lblselected = QLabel("")
        self.lblselected.setStyleSheet("QLabel { font-size:10pt; font-weight: bold; background-color: #e0e0e0}")
        self.lblselected.setMinimumWidth(600)
        #self.lblselected.setVisible(False)
        self.cbct = QComboBox()
        self.cbct.setVisible(False)
        self.cbct.currentTextChanged.connect(self.loadROC)
        self.lblctText = QLabel("")
        self.cbmode = QComboBox()
        self.cbmode.setVisible(False)
        self.cbmode.currentTextChanged.connect(self.loadROC)
        self.lblmodeText = QLabel("")
        self.recgrid.addWidget(self.lblselected, 0, 0, 1, 4)
        self.recgrid.addWidget(self.lblctText, 1, 0)
        self.recgrid.addWidget(self.cbct, 1, 1)
        self.recgrid.addWidget(self.lblmodeText, 1, 2)
        self.recgrid.addWidget(self.cbmode, 1, 3)

        layout = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.listFiles)
        hbox.addLayout(self.recgrid)
        lbltitle = QLabel("The following recognisers are present. Select the recogniser to customise. Click on the graph at the point where you would like the classifier to trade-off false positives with false negatives. Points closest to the top-left are best.")
        lbltitle.setWordWrap(True)
        layout.addWidget(lbltitle)
        layout.addLayout(hbox)
        layout.addLayout(self.form)
        layout.addLayout(savegrid)
        layout.setSpacing(25)
        self.setLayout(layout)
        # This will trigger initial button enabling etc
        self.rbtn1.setChecked(True)

    def readContents(self):
        self.listFiles.clear()
        cl = SupportClasses.ConfigLoader()
        self.FilterDict = cl.filters(self.filtdir, bats=True)
        for file in self.FilterDict:
            item = QListWidgetItem(self.listFiles)
            item.setText(file)

    def readFilter(self):
        self.filter = self.FilterDict[self.listFiles.currentItem().text()]
        self.newfilter = copy.deepcopy(self.filter)
        self.species = self.filter['species']
        self.calltypes = []
        self.newthr = 0
        self.cleangrid()
        self.enterFiltName.clear()
        self.cbct.clear()
        self.cbmode.clear()
        self.lblmodeText.clear()
        self.lblctText.clear()
        self.cbmode.setVisible(False)
        self.cbct.setVisible(False)
        self.ROCWF = False
        self.RONN = False
        self.btnSave.setEnabled(False)
        
        # Store the widget pointers here
        self.WThrSliders = []
        self.NNThr1Sliders = []
        self.NNThr2Sliders = []

        # Check if there is a saved ROC
        if 'RONN' in self.filter:
            if os.path.exists(os.path.join(self.filtdir, self.filter['RONN'] + '.json')):
                self.RONN = True
                self.lblmodeText.setText('Select mode')
                self.cbmode.addItem('NN')
                self.cbmode.setVisible(True)
        if 'ROCWF' in self.filter:
            if os.path.exists(os.path.join(self.filtdir, self.filter['ROCWF'] + '.json')):
                self.ROCWF = True
                self.lblmodeText.setText('Select mode')
                self.cbmode.addItem('WF')
                self.cbmode.setVisible(True)
        lblCT = QLabel('Call type')
        lblWTnew = QLabel('Wavelet threshold')
        lblNNTnew = QLabel('Lower NN threshold')
        lblNNT2new = QLabel('Upper NN threshold')
        lblCT.setStyleSheet("QLabel { font-weight: bold}")
        lblWTnew.setStyleSheet("QLabel { font-weight: bold}")
        lblNNTnew.setStyleSheet("QLabel { font-weight: bold}")
        lblNNT2new.setStyleSheet("QLabel { font-weight: bold}")

        self.form.addWidget(lblCT, 0, 0)
        self.form.addWidget(lblWTnew, 0, 1)
        if 'NN' in self.filter:
            self.form.addWidget(lblNNTnew, 0, 2)
            self.form.addWidget(lblNNT2new, 0, 3)

        ROCyes = self.RONN or self.ROCWF

        if ROCyes:
            self.lblctText.setText('Select call type')
            self.cbct.setVisible(True)
        else:
            self.lblctText.setText('There is no ROC saved for this recogniser. You can still change thresholds manually.')
            self.cbmode.setVisible(False)

        for i in range(len(self.filter['Filters'])):
            self.calltypes.append(self.filter['Filters'][i]['calltype'])
            ct = QLabel(self.filter['Filters'][i]['calltype'])
            ct.setStyleSheet("QLabel { font-style: italic}")
            self.form.addWidget(ct, i+1, 0)
            if ROCyes:
                self.cbct.addItem(self.filter['Filters'][i]['calltype'])

            newWThr = FilterCustomiseROC.LabelSlider(self.filter['Filters'][i]['WaveletParams']['thr'], 0.05, 5.0, slider=not ROCyes)
            newWThr.valueChanged.connect(self.refreshSaveButton)
            self.form.addWidget(newWThr, i + 1, 1)
            self.WThrSliders.append(newWThr)

            if 'NN' in self.filter:
                if self.filter['Filters'][i]['calltype']=='Bat':
                    newNNThr1 = FilterCustomiseROC.LabelSlider(self.filter['NN']['thr'][0], 0.1, 100.0, slider=not ROCyes)
                    newNNThr2 = FilterCustomiseROC.LabelSlider(self.filter['NN']['thr'][1], 0.1, 100.0, slider=not ROCyes)
                else:
                    newNNThr1 = FilterCustomiseROC.LabelSlider(self.filter['NN']['thr'][i][0], 0.1, 10.0, slider=not ROCyes)
                    newNNThr2 = FilterCustomiseROC.LabelSlider(self.filter['NN']['thr'][i][1], 0.1, 10.0, slider=not ROCyes)
                
                newNNThr1.valueChanged.connect(self.refreshSaveButton)
                self.form.addWidget(newNNThr1, i + 1, 2)
                self.NNThr1Sliders.append(newNNThr1)

                newNNThr2.valueChanged.connect(self.refreshSaveButton)
                self.form.addWidget(newNNThr2, i + 1, 3)
                self.NNThr2Sliders.append(newNNThr2)

    def loadROC(self):
        if self.cbmode.currentText() == "" or self.cbct.currentText() == "":
            # this is a bit dumb but didn't find a better way to
            # clear ROCCanvas nicely.
            try:
                self.recgrid.removeWidget(self.figCanvas)
                self.figCanvas.deleteLater()
                self.figCanvas.setParent(None)
            except:
                pass
            return

        with pg.BusyCursor():
            try:
                self.recgrid.removeWidget(self.figCanvas)
                self.figCanvas.deleteLater()
                self.figCanvas.setParent(None)
            except:
                pass
            self.figCanvas = ROCCanvas(self)
            self.recgrid.addWidget(self.figCanvas, 2, 0, 1, 4)
            self.figCanvas.plotme()
            self.figCanvas.show()

            self.marker = self.figCanvas.ax.plot([0, 1], [0, 1], marker='o', color='black', linestyle='dotted')[0]

            # figure click handler
            def onclick(event):
                fpr_cl = event.xdata
                tpr_cl = event.ydata
                if tpr_cl is None or fpr_cl is None:
                    return

                if self.cbmode.currentText() == 'WF':
                    # get M and thr for closest point
                    distarr = (tpr_cl - self.TPR) ** 2 + (fpr_cl - self.FPR) ** 2
                    M_min_ind, thr_min_ind = np.unravel_index(np.argmin(distarr), distarr.shape)
                    self.tpr_near = self.TPR[M_min_ind, thr_min_ind]
                    self.fpr_near = self.FPR[M_min_ind, thr_min_ind]
                    self.marker.set_visible(False)
                    self.figCanvas.draw()
                    self.marker.set_xdata([fpr_cl, self.fpr_near])
                    self.marker.set_ydata([tpr_cl, self.tpr_near])
                    self.marker.set_visible(True)
                    self.figCanvas.ax.draw_artist(self.marker)
                    self.figCanvas.update()
                elif self.cbmode.currentText() == 'NN':
                    # get thr for closest point
                    distarr = (tpr_cl - self.TPR) ** 2 + (fpr_cl - self.FPR) ** 2
                    M_min_ind, thr_min_ind = np.unravel_index(np.argmin(distarr), distarr.shape)
                    self.tpr_near = self.TPR[M_min_ind, thr_min_ind]
                    self.fpr_near = self.FPR[M_min_ind, thr_min_ind]
                    self.marker.set_visible(False)
                    self.figCanvas.draw()
                    self.marker.set_xdata([fpr_cl, self.fpr_near])
                    self.marker.set_ydata([tpr_cl, self.tpr_near])
                    self.marker.set_visible(True)
                    self.figCanvas.ax.draw_artist(self.marker)
                    self.figCanvas.update()

                print("fpr_cl, tpr_cl: ", self.fpr_near, self.tpr_near)

                if self.cbmode.currentText() == 'WF':
                    print('thr: ', self.thrList[thr_min_ind])
                    print('nodes: ', self.nodes[thr_min_ind])
                    self.newthr = round(self.thrList[thr_min_ind], 4)
                    self.refreshSaveButton()
                elif self.cbmode.currentText() == 'NN':
                    print('thr: ', self.thrList[thr_min_ind])
                    self.newthr = round(self.thrList[thr_min_ind], 4)
                    self.refreshSaveButton()

            self.figCanvas.figure.canvas.mpl_connect('button_press_event', onclick)

            if self.cbmode.currentText() == 'WF':
                jsonfile = open(os.path.join(self.filtdir, self.filter['ROCWF'] + '.json'), 'r')
                self.roc = json.loads(jsonfile.read())
                jsonfile.close()
                self.TPR = np.asarray([self.roc[self.cbct.currentText()][0]], dtype=np.float32)
                self.FPR = np.asarray([self.roc[self.cbct.currentText()][1]], dtype=np.float32)
                self.thrList = self.roc["thr"]
                self.nodes = self.roc[self.cbct.currentText()][2]
                self.figCanvas.plotmeagain(self.TPR, self.FPR)
            elif self.cbmode.currentText() == 'NN':
                jsonfile = open(os.path.join(self.filtdir, self.filter['RONN'] + '.json'), 'r')
                self.roc = json.loads(jsonfile.read())
                jsonfile.close()
                self.TPR = np.asarray([self.roc["TPR"][self.calltypes.index(self.cbct.currentText())]], dtype=np.float32)
                self.FPR = np.asarray([self.roc["FPR"][self.calltypes.index(self.cbct.currentText())]], dtype=np.float32)
                self.thrList = self.roc["thr"]
                self.figCanvas.plotmeagain(self.TPR, self.FPR)

    def onFilterSelect(self):
        if len(self.listFiles.selectedItems()) == 0:
            self.btnSave.setEnabled(False)
        else:
            self.lblselected.setText(" Selected recogniser: " + self.listFiles.currentItem().text() + '.txt\n' + ' Species name: ' + self.species)
            self.lblselected.setVisible(True)

    def refreshSaveButton(self):
        if self.ROCWF or self.RONN:
            self.refreshSaveButtonWithROC()
        else:
            self.refreshSaveButtonWithoutROC()

    def refreshSaveButtonWithROC(self):
        # only allow saving if any values have changed from the stored one:
        anyChanged = False

        # NOTE: for NNs, currently ROC adjusts only the lower thr.

        if self.newthr != 0:
            for idx in range(len(self.calltypes)):
                if 'NN' in self.filter:
                    sliderCL = self.NNThr1Sliders[idx]
                    sliderCU = self.NNThr2Sliders[idx]
                    # parse the ct and mode of currently edited ROC:
                    if self.filter['Filters'][idx]['calltype'] == self.cbct.currentText() and self.cbmode.currentText() == 'NN':
                        sliderCL.setValue(self.newthr)
                        # sanity check
                        if sliderCL.value() > sliderCU.value():
                            sliderCU.setValue(1.0)

                    if self.calltypes[idx]=='Bat':
                        self.newfilter['NN']['thr'][0] = sliderCL.value()
                        self.newfilter['NN']['thr'][1] = sliderCU.value()
                    else:
                        self.newfilter['NN']['thr'][idx][0] = sliderCL.value()
                        self.newfilter['NN']['thr'][idx][1] = sliderCU.value()
                    if sliderCL.hasChanged() or sliderCU.hasChanged():
                        anyChanged = True

                sliderW = self.WThrSliders[idx]
                # parse the ct and mode of currently edited ROC:
                if self.filter['Filters'][idx]['calltype'] == self.cbct.currentText() and self.cbmode.currentText() == 'WF':
                    sliderW.setValue(self.newthr)

                self.newfilter['Filters'][idx]['WaveletParams']['thr'] = sliderW.value()
                if sliderW.hasChanged():
                    anyChanged = True

        btnState = anyChanged and (self.saveoption == "Update" or self.enterFiltName.hasAcceptableInput())
        self.btnSave.setEnabled(btnState)

    def refreshSaveButtonWithoutROC(self):
        # only allow saving if any values have changed from the stored one:
        anyChanged = False

        self.btnSave.setEnabled(False)
        for idx in range(len(self.calltypes)):
            if 'NN' in self.filter:
                sliderCL = self.NNThr1Sliders[idx]
                sliderCU = self.NNThr2Sliders[idx]
                if self.calltypes[idx]=='Bat':
                    self.newfilter['NN']['thr'][0] = sliderCL.value()
                    self.newfilter['NN']['thr'][1] = sliderCU.value()
                else:
                    self.newfilter['NN']['thr'][idx][0] = sliderCL.value()
                    self.newfilter['NN']['thr'][idx][1] = sliderCU.value()
                if sliderCL.hasChanged() or sliderCU.hasChanged():
                    anyChanged = True
            sliderW = self.WThrSliders[idx]
            self.newfilter['Filters'][idx]['WaveletParams']['thr'] = sliderW.value()
            if sliderW.hasChanged():
                anyChanged = True

        btnState = anyChanged and (self.saveoption == "Update" or self.enterFiltName.hasAcceptableInput())
        self.btnSave.setEnabled(btnState)

    def onClicked(self, checked):
        # This should only be connected to the New btn
        self.saveoption = "New" if checked else "Update"
        self.enterFiltName.setEnabled(checked)
        self.refreshSaveButton()

    def cleangrid(self):
        while self.form.count():
            item = self.form.takeAt(0)
            widget = item.widget()
            widget.deleteLater()
            widget.setParent(None)