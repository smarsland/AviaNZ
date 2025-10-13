
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
from PyQt6.QtWidgets import QLabel, QDialog, QComboBox, QCheckBox, QPushButton, QLineEdit, QSlider, QVBoxLayout, QFormLayout, QRadioButton, QSpinBox, QDoubleSpinBox # listing some explicitly to make syntax checks lighter
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt

import pyqtgraph as pg




pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

class SegmentationDialog(QDialog):
    # Class for the segmentation dialog box
    # TODO: add the wavelet params
    # TODO: work out how to return varying size of params, also process them
    # TODO: test and play
    def __init__(self, maxv, DOC=False, species=None, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Segmentation Options')
        self.setWindowIcon(QIcon('src/resources/images/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowType.WindowContextHelpButtonHint) | Qt.WindowType.WindowCloseButtonHint)
        self.setMinimumWidth(350)

        self.algs = QComboBox()
        if DOC:
            self.algs.addItems(["WV Changepoint", "Wavelet Filter", "FIR", "Median Clipping"])
        else:
            self.algs.addItems(["Default","Median Clipping","Fundamental Frequency","FIR","Harma","Power","WV Changepoint","Wavelet Filter","Cross-Correlation"])
        self.algs.currentTextChanged.connect(self.changeBoxes)
        #self.algs.currentIndexChanged.connect(self.changeBoxes)
        self.undo = QPushButton("Undo")
        self.activate = QPushButton("Segment")
        #self.save = QPushButton("Save segments")

        # Define the whole set of possible options for the dialog box here, just to have them together.
        # Then hide and show them as required as the algorithm chosen changes.

        # Spin box for amplitude threshold
        self.ampThr = QDoubleSpinBox()
        self.ampThr.setRange(0.001,maxv+0.001)
        self.ampThr.setSingleStep(0.002)
        self.ampThr.setDecimals(4)
        self.ampThr.setValue(maxv+0.001)

        self.HarmaThr1 = QSpinBox()
        self.HarmaThr1.setRange(10,90)
        self.HarmaThr1.setSingleStep(1)
        self.HarmaThr1.setValue(10)
        self.HarmaThr2 = QDoubleSpinBox()
        self.HarmaThr2.setRange(0.1,0.95)
        self.HarmaThr2.setSingleStep(0.05)
        self.HarmaThr2.setDecimals(2)
        self.HarmaThr2.setValue(0.9)

        self.PowerThr = QDoubleSpinBox()
        self.PowerThr.setRange(0.0,2.0)
        self.PowerThr.setSingleStep(0.1)
        self.PowerThr.setValue(1.0)

        self.Fundminfreqlabel = QLabel("Min Frequency")
        self.Fundminfreq = QLineEdit()
        self.Fundminfreq.setText('100')
        self.Fundminperiodslabel = QLabel("Min Number of periods")
        self.Fundminperiods = QSpinBox()
        self.Fundminperiods.setRange(1,10)
        self.Fundminperiods.setValue(3)
        self.Fundthrlabel = QLabel("Threshold")
        self.Fundthr = QDoubleSpinBox()
        self.Fundthr.setRange(0.1,1.0)
        self.Fundthr.setDecimals(1)
        self.Fundthr.setValue(0.5)
        self.Fundwindowlabel = QLabel("Window size (will be rounded up as appropriate)")
        self.Fundwindow = QSpinBox()
        self.Fundwindow.setRange(300,5000)
        self.Fundwindow.setSingleStep(500)
        self.Fundwindow.setValue(1000)

        self.medThr = QDoubleSpinBox()
        self.medThr.setRange(0.2,6)
        self.medThr.setSingleStep(1)
        self.medThr.setDecimals(2)
        self.medThr.setValue(3)

        # set min seg size for median clipping
        self.medSize = QSlider(Qt.Orientation.Horizontal)
        self.medSize.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.medSize.setTickInterval(100)
        self.medSize.setRange(100,2000)
        self.medSize.setSingleStep(100)
        self.medSize.setValue(1000)
        self.medSize.valueChanged.connect(self.medSizeChange)
        self.medSizeText = QLabel("Minimum length: 1000 ms")

        self.ecThr = QDoubleSpinBox()
        self.ecThr.setRange(0.001,6)
        self.ecThr.setSingleStep(1)
        self.ecThr.setDecimals(3)
        self.ecThr.setValue(1)

        self.FIRThr1text = QLabel("Set threshold")
        self.FIRThr1 = QDoubleSpinBox()
        self.FIRThr1.setRange(0.0,2.0) #setRange(0.0,1.0)
        self.FIRThr1.setSingleStep(0.05)
        self.FIRThr1.setValue(0.1)

        self.CCThr1 = QDoubleSpinBox()
        self.CCThr1.setRange(0.0,2.0) #setRange(0.0,1.0)
        self.CCThr1.setSingleStep(0.1)
        self.CCThr1.setValue(0.4)

        Box = QVBoxLayout()
        Box.addWidget(self.algs)
        Box.addStretch(1)

        # Labels
        #self.amplabel = QLabel("Set threshold amplitude")
        #Box.addWidget(self.amplabel)

        self.Harmalabel = QLabel("Set decibel threshold")
        Box.addWidget(self.Harmalabel)

        #self.Onsetslabel = QLabel("Onsets: No parameters")
        #Box.addWidget(self.Onsetslabel)

        self.medlabel = QLabel("Set median threshold")
        self.medlabel.show()

        self.eclabel = QLabel("Set energy curve threshold")
        self.ecthrtype = [QRadioButton("N standard deviations"), QRadioButton("Threshold")]

        self.specieslabel = QLabel("Species")
        self.species_wv = QComboBox()
        self.species_chp = QComboBox()

        self.specieslabel_cc = QLabel("Species")
        self.species_cc = QComboBox()
        self.species_cc.addItems(["Choose species...", "Bittern"])

        # parse the provided filter list into wavelet and changepoint boxes
        self.filters = species
        spp_chp = []
        spp_wv = []
        for key, item in species.items():
            if item.get("method")=="chp":
                spp_chp.append(key)
            else:
                spp_wv.append(key)
        spp_chp = sorted(spp_chp)
        spp_wv = sorted(spp_wv)
        spp_chp.insert(0, "Choose species...")
        spp_wv.insert(0, "Choose species...")
        self.species_wv.addItems(spp_wv)
        self.species_chp.addItems(spp_chp)
        self.species_chp.currentTextChanged.connect(self.filterChange)

        Box.addWidget(self.specieslabel)
        Box.addWidget(self.species_wv)
        Box.addWidget(self.species_chp)

        Box.addWidget(self.specieslabel_cc)
        Box.addWidget(self.species_cc)

        Box.addWidget(self.HarmaThr1)
        Box.addWidget(self.HarmaThr2)
        Box.addWidget(self.PowerThr)

        Box.addWidget(self.medlabel)
        Box.addWidget(self.medThr)
        Box.addWidget(self.medSizeText)
        Box.addWidget(self.medSize)

        Box.addWidget(self.eclabel)
        for i in range(len(self.ecthrtype)):
            Box.addWidget(self.ecthrtype[i])
        Box.addWidget(self.ecThr)

        Box.addWidget(self.FIRThr1text)
        Box.addWidget(self.FIRThr1)

        Box.addWidget(self.Fundminfreqlabel)
        Box.addWidget(self.Fundminfreq)
        Box.addWidget(self.Fundminperiodslabel)
        Box.addWidget(self.Fundminperiods)
        Box.addWidget(self.Fundthrlabel)
        Box.addWidget(self.Fundthr)
        Box.addWidget(self.Fundwindowlabel)
        Box.addWidget(self.Fundwindow)

        Box.addWidget(self.CCThr1)

        self.medSize = QSlider(Qt.Orientation.Horizontal)
        self.medSize.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.medSize.setTickInterval(100)
        self.medSize.setRange(100,2000)
        self.medSize.setSingleStep(100)
        self.medSize.setValue(1000)
        self.medSize.valueChanged.connect(self.medSizeChange)

        # Parameter selectors for changepoint methods
        self.chpalpha = QDoubleSpinBox()
        self.chpalpha.setRange(0.1, 20)
        self.chpalpha.setValue(3)

        self.chpwin = QDoubleSpinBox()
        self.chpwin.setDecimals(3)
        self.chpwin.setRange(0.005, 3)
        self.chpwin.setValue(0.5)

        self.maxlen = QDoubleSpinBox()
        self.maxlen.setDecimals(3)
        self.maxlen.setRange(0.05, 100)
        self.maxlen.setValue(10)

        # Sliders for minlen and maxgap are in ms scale
        self.minlen = QSlider(Qt.Orientation.Horizontal)
        self.minlen.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.minlen.setTickInterval(500)
        self.minlen.setRange(100, 10000)
        self.minlen.setSingleStep(100)
        self.minlen.setValue(500)
        self.minlen.valueChanged.connect(self.minLenChange)
        self.minlenlbl = QLabel("Minimum segment length: 0.5 sec")

        self.maxgap = QSlider(Qt.Orientation.Horizontal)
        self.maxgap.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.maxgap.setTickInterval(250)
        self.maxgap.setRange(50, 4000)
        self.maxgap.setSingleStep(50)
        self.maxgap.setValue(1000)
        self.maxgap.valueChanged.connect(self.maxGapChange)
        self.maxgaplbl = QLabel("Maximum gap between syllables: 1 sec")

        self.wind = QComboBox()
        self.wind.addItems(["OLS wind filter (recommended)", "Robust wind filter (experimental, slow)", "None"])

        self.chpLayout = QFormLayout()
        self.chpLayout.addRow("Threshold:", self.chpalpha)
        self.chpLayout.addRow("Window size (s):", self.chpwin)
        self.chpLayout.addRow("Max length (s):", self.maxlen)
        self.chpLayout.addRow("Wind denoising:", self.wind)

        self.rain = QCheckBox("Remove rain")
        Box.addWidget(self.rain)
        Box.addWidget(self.maxgaplbl)
        Box.addWidget(self.maxgap)
        Box.addWidget(self.minlenlbl)
        Box.addWidget(self.minlen)
        Box.addLayout(self.chpLayout)
        Box.addStretch(2)
        Box.addWidget(self.undo)
        self.undo.setEnabled(False)
        Box.addWidget(self.activate)
        #Box.addWidget(self.save)

        # Now put everything into the frame,
        # hide and reopen the default
        self.setLayout(Box)
        self.hideAll()
        self.algs.show()
        self.undo.show()
        self.activate.show()
        if DOC:
            self.changeBoxes("WV Changepoint")
        else:
            self.changeBoxes("Default")

    def hideAll(self):
        for w in range(self.layout().count()):
            item = self.layout().itemAt(w)
            if item.widget() is not None:
                # it is a widget
                item.widget().hide()
            elif item.layout() is not None:
                # it is a layout, so loop again:
                for ww in range(item.layout().count()):
                    item.layout().itemAt(ww).widget().hide()
            # pure items (stretch/spacer) get skipped


    def changeBoxes(self,alg):
        # This does the hiding and showing of the options as the algorithm changes
        self.hideAll()
        self.algs.show()
        # self.rain.show()
        self.minlenlbl.show()
        self.minlen.show()
        self.maxgaplbl.show()
        self.maxgap.show()
        self.undo.show()
        self.activate.show()

        if alg == "Default":
            pass
        elif alg == "Energy Curve":
            self.eclabel.show()
            self.ecThr.show()
            for i in range(len(self.ecthrtype)):
                self.ecthrtype[i].show()
            self.ecThr.show()
        elif alg == "Harma":
            self.Harmalabel.show()
            self.HarmaThr1.show()
            self.HarmaThr2.show()
        elif alg == "Power":
            self.PowerThr.show()
        elif alg == "Median Clipping":
            self.medlabel.show()
            self.medThr.show()
            # self.medSize.show()
            # self.medSizeText.show()
        elif alg == "Fundamental Frequency":
            self.Fundminfreq.show()
            self.Fundminperiods.show()
            self.Fundthr.show()
            self.Fundwindow.show()
            self.Fundminfreqlabel.show()
            self.Fundminperiodslabel.show()
            self.Fundthrlabel.show()
            self.Fundwindowlabel.show()
        #elif alg == "Onsets":
        #    self.Onsetslabel.show()
        elif alg == "FIR":
            self.FIRThr1text.show()
            self.FIRThr1.show()
        elif alg == "Cross-Correlation":
            self.CCThr1.show()
            self.specieslabel_cc.show()
            self.species_cc.show()
        else:
            #"Wavelet Filter" or "WV Changepoint"
            self.specieslabel.show()
            if alg == "WV Changepoint":
                for ww in range(self.chpLayout.count()):
                    self.chpLayout.itemAt(ww).widget().show()
                self.species_chp.show()
            else:
                self.species_wv.show()
            self.maxgaplbl.hide()
            self.maxgap.hide()
            self.minlenlbl.hide()
            self.minlen.hide()

    def filterChange(self, species):
        """ Override UI with parameters from the requested filter. """
        subfilt = self.filters[species]["Filters"][0]
        self.chpalpha.setValue(subfilt["WaveletParams"]["thr"])
        self.medThr.setValue(subfilt["WaveletParams"]["thr"])
        self.chpwin.setValue(subfilt["WaveletParams"]["win"])
        self.maxlen.setValue(subfilt["TimeRange"][1])

    def medSizeChange(self,value):
        self.medSizeText.setText("Minimum length: %s ms" % value)

    def minLenChange(self,value):
        self.minlenlbl.setText("Minimum segment length: %s sec" % str(round(int(value)/1000, 2)))

    def maxGapChange(self,value):
        self.maxgaplbl.setText("Maximum gap between syllables: %s sec" % str(round(int(value)/1000, 2)))

    def getValues(self):
        # TODO: check: self.medSize.value() is not used, should we keep it?
        alg = self.algs.currentText()
        if alg=="Wavelet Filter":
            filtname = self.species_wv.currentText()
        elif alg=="WV Changepoint":
            filtname = self.species_chp.currentText()
        elif alg=="Cross-Correlation":
            filtname = self.species_cc.currentText()
        else:
            filtname = None
        settings = {"medThr": self.medThr.value(), "medSize": self.medSize.value(), "HarmaThr1": self.HarmaThr1.text(), "HarmaThr2": self.HarmaThr2.text(), "PowerThr": self.PowerThr.text(),
                    "FFminfreq": self.Fundminfreq.text(), "FFminperiods": self.Fundminperiods.text(), "Yinthr": self.Fundthr.text(), "FFwindow": self.Fundwindow.text(), "FIRThr1": self.FIRThr1.text(),
                    "CCThr1": self.CCThr1.text(), "filtname": filtname, "rain": self.rain.isChecked(),
                    "maxgap": int(self.maxgap.value())/1000, "minlen": int(self.minlen.value())/1000, "chpalpha": self.chpalpha.value(), "chpwindow": self.chpwin.value(), "maxlen": self.maxlen.value(),
                    "wind": self.wind.currentText()}
        return(alg, settings)
