# Self-standing executable for tall comparison mode

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QPushButton, QPlainTextEdit, QWidget, QGridLayout, QSpinBox, QGroupBox, QSizePolicy, QSpacerItem, QLayout, QProgressDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QStyle, QComboBox
from PyQt5.QtCore import QDir, Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap

import sys
import os
import datetime as dt
import itertools

import pyqtgraph as pg

import SupportClasses
import Segment

class CompareCalls(QMainWindow):
    def __init__(self):
        super(CompareCalls, self).__init__()
        print("Starting...")
        self.setWindowTitle("AviaNZ call comparator")

        # menu bar
        fileMenu = self.menuBar()#.addMenu("&File")
        fileMenu.addAction("About", lambda: SupportClasses.MessagePopup("a", "About", ".").exec_())
        fileMenu.addAction("Quit", lambda: QApplication.quit())
        # do we need this?
        # if platform.system() == 'Darwin':
        #    helpMenu.addAction("About",self.showAbout,"Ctrl+A")

        # main dock setup
        area = QWidget()
        mainbox = QVBoxLayout()
        area.setLayout(mainbox)
        self.setCentralWidget(area)

        ## Data dir selection
        self.dirName = ""
        label = QLabel("Select input folder containing files named in one of the following formats:")
        formatlabel = QLabel("recordername_YYMMDD_hhmmss.wav\n or \nrecordername_YYYYMMDD_hhmmss.wav")
        formatlabel.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("QLabel {color: #303030}")
        formatlabel.setStyleSheet("QLabel {color: #303030}")

        self.w_browse = QPushButton(" Browse Folder")
        self.w_browse.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.w_browse.setStyleSheet('QPushButton {background-color: #c4ccd3; font-weight: bold; font-size:14px; padding: 3px 3px 3px 3px}')
        self.w_browse.setMinimumSize(170, 40)
        self.w_browse.setSizePolicy(QSizePolicy(1,1))
        self.w_browse.clicked.connect(self.browse)

        # area showing the selected folder
        self.w_dir = QPlainTextEdit()
        self.w_dir.setFixedHeight(40)
        self.w_dir.setPlainText('')
        self.w_dir.setSizePolicy(QSizePolicy(3,1))

        # feedback label
        self.labelDatasIcon = QLabel()
        self.labelDatas = QLabel()

        ## Clock shift adjustment
        labelClockSp = QLabel("Select species to analyse:")
        self.clockSpBox = QComboBox()

        self.suggestAdjBtn = QPushButton("Auto-suggest adjustment")
        self.suggestAdjBtn.setStyleSheet('QPushButton {background-color: #c4ccd3; font-weight: bold; font-size:14px; padding: 3px 3px 3px 3px}')
        self.suggestAdjBtn.clicked.connect(self.suggestAdjustment)

        self.labelAdjust = QLabel()
        self.adjustmentsOut = QPlainTextEdit()

        ### Layouts for each box
        # input dir
        inputGroup = QGroupBox("Input")
        inputGrid = QGridLayout()
        inputGroup.setLayout(inputGrid)
        inputGroup.setStyleSheet("QGroupBox:title{color: #505050; font-weight: 50}")

        inputGrid.addWidget(label, 0, 0, 1, 4)
        inputGrid.addWidget(formatlabel, 1, 0, 1, 4)
        inputGrid.addWidget(self.w_dir, 2, 0, 1, 3)
        inputGrid.addWidget(self.w_browse, 2, 3, 1, 1)
        hboxLabel = QHBoxLayout()
        hboxLabel.addWidget(self.labelDatasIcon)
        hboxLabel.addWidget(self.labelDatas)
        hboxLabel.addStretch(10)
        inputGrid.addLayout(hboxLabel, 3, 0, 1, 4)

        # clock adjustment
        clockadjGroup = QGroupBox("Clock adjustment")
        clockadjGrid = QGridLayout()
        clockadjGroup.setLayout(clockadjGrid)
        clockadjGroup.setStyleSheet("QGroupBox:title{color: #505050; font-weight: 50}")

        clockadjGrid.addWidget(labelClockSp, 0, 0, 1, 1)
        clockadjGrid.addWidget(self.clockSpBox, 0, 1, 1, 3)
        clockadjGrid.addWidget(self.suggestAdjBtn, 1, 0, 1, 4)
        clockadjGrid.addWidget(self.labelAdjust, 2, 0, 1, 4)
        clockadjGrid.addWidget(self.adjustmentsOut, 3, 0, 1, 4)

        # call comparator
        comparisonGroup = QGroupBox("Call comparison")
        comparisonGrid = QGridLayout()
        comparisonGroup.setLayout(comparisonGrid)
        comparisonGroup.setStyleSheet("QGroupBox:title{color: #505050; font-weight: 50}")

        # output save btn and settings
        outputGroup = QGroupBox("Output")
        outputGrid = QGridLayout()
        outputGroup.setLayout(outputGrid)
        outputGroup.setStyleSheet("QGroupBox:title{color: #505050; font-weight: 50}")

        ### Main Layout
        mainbox.addWidget(inputGroup)
        mainbox.addWidget(clockadjGroup)
        mainbox.addWidget(comparisonGroup)
        mainbox.addWidget(outputGroup)

    def browse(self):
        if self.dirName:
            self.dirName = QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirName))
        else:
            self.dirName = QFileDialog.getExistingDirectory(self,'Choose Folder to Process')

        # Reset interface so that it is empty/disabled in case an error is encountered
        self.w_dir.setPlainText("")
        self.w_dir.setReadOnly(True)
        self.labelDatas.setText("")
        pm = self.style().standardIcon(QStyle.SP_DialogCancelButton).pixmap(QSize(12,12))
        self.labelDatasIcon.setPixmap(pm)
        self.suggestAdjBtn.setEnabled(False)
        self.clockSpBox.clear()

        self.annots = []
        self.allrecs = set()

        # this will load the folder and populate annots
        with pg.BusyCursor():
            succ = self.checkInputDir()
            self.allrecs = list(self.allrecs)
            if succ>0:
                return

        self.w_dir.setPlainText(self.dirName)
        self.w_dir.setReadOnly(True)

        # Check if any DATA files were read, enable GUI for starting analysis
        if len(self.annots)==0:
            # can't do clock adjustment, nor call comparison
            self.labelDatas.setText("No DATA files detected")
            self.labelDatas.setStyleSheet("QLabel {color: #ff2020}")
            print("ERROR: no WAV or DATA files found")
            return(1)
        else:
            self.labelDatas.setText("Read %d data files from %d recorders" % (len(self.annots), len(self.allrecs)))
            self.labelDatas.setStyleSheet("QLabel {color: #000000}")
            pm = self.style().standardIcon(QStyle.SP_DialogApplyButton).pixmap(QSize(12,12))
            self.labelDatasIcon.setPixmap(pm)
            print("Read %d data files from %d recorders" % (len(self.annots), len(self.allrecs)))

            # allow analysis to be started
            spList = []
            for sl in self.annots:
                if len(sl)>0:
                    filesp = [lab["species"] for seg in sl for lab in seg[4]]
                    spList.extend(filesp)
            spList = list(set(spList))
            self.clockSpBox.addItems(spList)
            self.suggestAdjBtn.setEnabled(True)

    def checkInputDir(self):
        """ Checks the input file dir filenames etc. for validity
            Returns an error code if the specified directory is bad.
        """
        if not os.path.isdir(self.dirName):
            print("ERROR: directory %s doesn't exist" % self.dirName)
            return(1)

        # list all datas that will be processed
        alldatas = []
        try:
            for root, dirs, files in os.walk(str(self.dirName)):
                for filename in files:
                    if filename.lower().endswith('.wav.data'):
                        alldatas.append(os.path.join(root, filename))
        except Exception as e:
            print("ERROR: could not load dir %s" % self.dirName)
            print(e)
            return(1)

        # read in all datas to self.annots
        for f in alldatas:
            # must have correct naming format:
            infilestem = os.path.basename(f)[:-9]
            try:
                recname, filedate, filetime = infilestem.split("_")  # get [date, time]
                datestamp = filedate + '_' + filetime  # make "date_time"
                # check both 4-digit and 2-digit codes (century that produces closest year to now is inferred)
                try:
                    d = dt.datetime.strptime(datestamp, "%Y%m%d_%H%M%S")
                except ValueError:
                    d = dt.datetime.strptime(datestamp, "%y%m%d_%H%M%S")

                # timestamp identified, so read this file:
                segs = Segment.SegmentList()
                try:
                    segs.parseJSON(f, silent=True)
                except Exception as e:
                    print("Warning: could not read file %s" % f)
                    print(e)
                    continue

                # store the wav filename
                segs.wavname = f[:-5]
                segs.recname = recname
                segs.datetime = d
                self.annots.append(segs)

                # also keep track of the different recorders
                self.allrecs.add(recname)
            except ValueError:
                print("Could not identify timestamp in", f)
                continue
        print("Detected recorders:", self.allrecs)
        return(0)

    def calculateOverlap(self, dtl1, dtl2):
        """ Calculates total overlap between two lists of pairs of datetime obj:
            [(s1, e1), (s2, e2)] + [(s3, e3), (s4, e4)] -> total overl. in s
            Assumes that each tuple is in correct order (start, end).
        """
        overl = 0
        for dt1 in dtl1:
            for dt2 in dtl2:
                laststart = max(dt1[0], dt2[0])
                firstend = min(dt1[1], dt2[1])
                if firstend>laststart:
                    overl = overl + (firstend - laststart).seconds
        return(overl)

    def suggestAdjustment(self):
        """ Goes over all possible pairs of recorders and suggests best adjustment
            (based on max annotation overlap) for them.
        """
        print("Starting clock adjustment")

        species = self.clockSpBox.currentText()
        # do shifts in this resolution (seconds)
        hop = 1

        # generate all recorder pairs
        if len(self.allrecs)>30:
            print("ERROR: using more than 30 recorders disabled for safety")
            return(1)

        with pg.BusyCursor():
            # gather all annotations for each rec and this species
            # and convert them into: [(datetime_start, datetime_end), (ds, de), ...] for each recorder
            print("Collecting annotations for species %s" % species)
            speciesAnnots = []
            for i in range(len(self.allrecs)):
                rec1 = self.allrecs[i]
                thisRecAnnots = []
                for sl in self.annots:
                    if sl.recname!=rec1:
                        continue

                    # indices of segments in this file that contain the right species in at least one label:
                    six = sl.getSpecies(species)
                    for ix in six:
                        # convert in-file annot times into absolute time (in datetime obj)
                        absstart = sl.datetime + dt.timedelta(seconds=sl[ix][0])
                        absend = sl.datetime + dt.timedelta(seconds=sl[ix][1])

                        # segments from old versions can still be reversed
                        if absstart > absend:
                            absstart, absend = absend, absstart

                        thisRecAnnots.append((absstart, absend))

                print("Found %d annotations" % len(thisRecAnnots))
                if len(thisRecAnnots)>100:
                    print("ERROR: using more than 100 annotations per recorder disabled for safety")
                    return(1)

                speciesAnnots.append(thisRecAnnots)

            # output: vector with 1 best time shift for each rec
            adjustments = []
            for i in range(len(self.allrecs)):
                rec1 = self.allrecs[i]
                rec1annots = speciesAnnots[i]
                print("Working on recorder", rec1)
                print("Found %d annotations" % len(rec1annots))

                # find best adjustment for rec1
                for j in range(i+1, len(self.allrecs)):
                    rec2 = self.allrecs[j]
                    rec2annots = speciesAnnots[j]
                    print("Comparing with recorder", rec2)
                    overl = self.calculateOverlap(rec1annots, rec2annots)
                    print("Calculated overlap: %d seconds" % overl)

                    # find deltat that maximizes overlap between rec1annots and rec2annots
                    # for deltat in range(-300, 301):
                    #    calculateOverlap, rec1annots, rec2annots, deltat)
                    bestShift = 0

                adjustments.append(bestShift)

        print("Clock adjustment complete")
        self.labelAdjust.setText("Clock adjustment complete")
        self.adjustmentsOut.setPlainText(",".join(map(str, adjustments)))

#### MAIN LAUNCHER

print("Starting AviaNZ call comparator")
app = QApplication(sys.argv)
mainwindow = CompareCalls()
mainwindow.show()
app.exec_()
print("Processing complete, closing AviaNZ call comparator")
QApplication.closeAllWindows()
