# Self-standing executable for tall comparison mode

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QPushButton, QPlainTextEdit, QWidget, QGridLayout, QDoubleSpinBox, QGroupBox, QSizePolicy, QSpacerItem, QLayout, QProgressDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QStyle, QComboBox, QDialog, QToolButton, QCheckBox
from PyQt5.QtCore import QDir, Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap

import sys
import os
import datetime as dt
import numpy as np

import pyqtgraph as pg

import SupportClasses
import Segment
import SignalProc
import colourMaps


class CompareCalls(QMainWindow):
    def __init__(self):
        super(CompareCalls, self).__init__()
        print("Starting...")
        self.setWindowTitle("AviaNZ call comparator")
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

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
        self.w_dir.setReadOnly(True)
        self.w_dir.setFixedHeight(40)
        self.w_dir.setPlainText('')
        self.w_dir.setSizePolicy(QSizePolicy(3,1))

        # feedback label
        self.labelDatasIcon = QLabel()
        self.labelDatas = QLabel()

        ## Clock shift adjustment
        labelClockSp = QLabel("Select species to analyse:")
        self.clockSpBox = QComboBox()

        labelHop = QLabel("Clock shift resolution (s)")
        self.hopSpin = QDoubleSpinBox()
        self.hopSpin.setSingleStep(0.1)
        self.hopSpin.setDecimals(3)
        self.hopSpin.setValue(1.0)

        self.suggestAdjBtn = QPushButton("Auto-suggest adjustment")
        self.suggestAdjBtn.setStyleSheet('QPushButton {background-color: #c4ccd3; font-weight: bold; font-size:14px; padding: 3px 3px 3px 3px}')
        self.suggestAdjBtn.clicked.connect(self.suggestAdjustment)
        self.suggestAdjBtn.setEnabled(False)

        self.labelAdjustIcon = QLabel()
        self.labelAdjust = QLabel()

        self.reviewAdjBtn = QPushButton("Review suggested adjustments")
        self.reviewAdjBtn.setStyleSheet('QPushButton {background-color: #c4ccd3; font-weight: bold; font-size:14px; padding: 3px 3px 3px 3px}')
        self.reviewAdjBtn.clicked.connect(self.showAdjustmentsDialog)
        self.reviewAdjBtn.setEnabled(False)

        # call review
        self.compareBtn = QPushButton("Compare calls")
        self.compareBtn.setStyleSheet('QPushButton {background-color: #c4ccd3; font-weight: bold; font-size:14px; padding: 3px 3px 3px 3px}')
        self.compareBtn.clicked.connect(self.showComparisonDialog)
        self.compareBtn.setEnabled(False)
        self.compareRecSelector = QComboBox()

        self.adjustmentsOut = QPlainTextEdit()
        self.adjustmentsOut.setReadOnly(True)
        self.componentsOut = QPlainTextEdit()
        self.componentsOut.setReadOnly(True)

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
        clockadjGrid.addWidget(labelHop, 1, 0, 1, 1)
        clockadjGrid.addWidget(self.hopSpin, 1, 1, 1, 3)
        clockadjGrid.addWidget(self.suggestAdjBtn, 2, 0, 1, 4)
        hboxLabel2 = QHBoxLayout()
        hboxLabel2.addWidget(self.labelAdjustIcon)
        hboxLabel2.addWidget(self.labelAdjust)
        hboxLabel2.addStretch(10)
        clockadjGrid.addLayout(hboxLabel2, 3, 0, 1, 4)
        clockadjGrid.addWidget(self.reviewAdjBtn, 4, 0, 1, 4)

        # call comparator
        comparisonGroup = QGroupBox("Call comparison")
        comparisonGrid = QGridLayout()
        comparisonGroup.setLayout(comparisonGrid)
        comparisonGroup.setStyleSheet("QGroupBox:title{color: #505050; font-weight: 50}")

        comparisonGrid.addWidget(QLabel("Compare recorders:"), 0, 0, 1, 2)
        comparisonGrid.addWidget(self.compareRecSelector, 0, 2, 1, 2)
        comparisonGrid.addWidget(self.compareBtn, 1,0,1,4)

        # output save btn and settings
        outputGroup = QGroupBox("Output")
        outputGrid = QGridLayout()
        outputGroup.setLayout(outputGrid)
        outputGroup.setStyleSheet("QGroupBox:title{color: #505050; font-weight: 50}")
        outputGrid.addWidget(self.adjustmentsOut, 0, 0, 1, 4)
        outputGrid.addWidget(self.componentsOut, 1, 0, 1, 4)

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
        self.reviewAdjBtn.setEnabled(False)
        self.compareBtn.setEnabled(False)
        self.clockSpBox.clear()
        self.compareRecSelector.clear()

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
            for i in range(len(self.allrecs)):
                for j in range(i+1, len(self.allrecs)):
                    pair = self.allrecs[i] + "-" + self.allrecs[j]
                    self.compareRecSelector.addItem(pair)

            self.pairwiseShifts = np.zeros((len(self.allrecs), len(self.allrecs)))

            self.suggestAdjBtn.setEnabled(True)
            self.compareBtn.setEnabled(True)

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

    def calculateOverlap(self, dtl1, dtl2, shift=0):
        """ Calculates total overlap between two lists of pairs of datetime obj:
            [(s1, e1), (s2, e2)] + [(s3, e3), (s4, e4)] -> total overl. in s
            Shift: shifts dt1 by this many seconds (+ for lead, - for lag)
            Assumes that each tuple is in correct order (start, end).
        """
        overl = dt.timedelta()
        for dt1 in dtl1:
            st1 = dt1[0]+shift
            end1 = dt1[1]+shift
            for dt2 in dtl2:
                laststart = max(st1, dt2[0])
                firstend = min(end1, dt2[1])
                if firstend>laststart:
                    overl = firstend - laststart + overl
        # convert from datetime timedelta to seconds
        overl = overl.total_seconds()
        return(overl)

    def suggestAdjustment(self):
        """ Goes over all possible pairs of recorders and suggests best adjustment
            (based on max annotation overlap) for them.
        """
        print("Starting clock adjustment")

        species = self.clockSpBox.currentText()
        # do shifts in this resolution (seconds)
        hop = self.hopSpin.value()

        # Disable GUI in case of errors
        self.labelAdjust.setText("Clock adjustment not done")
        pm = self.style().standardIcon(QStyle.SP_DialogCancelButton).pixmap(QSize(12,12))
        self.labelAdjustIcon.setPixmap(pm)
        self.reviewAdjBtn.setEnabled(False)

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

            # matrix of pairwise connectivity between recorders
            self.recConnections = np.ones((len(self.allrecs), len(self.allrecs)))

            # determine 1 best time shift for each pair of recorders
            self.pairwiseShifts = np.zeros((len(self.allrecs), len(self.allrecs)))
            self.overlaps_forplot = []
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

                    overlap = np.zeros(601)
                    for shifti in range(-300, 301):
                        shift = dt.timedelta(seconds=shifti * hop)
                        overlap[shifti+300] = self.calculateOverlap(rec1annots, rec2annots, shift)

                    bestOverlap = np.max(overlap)
                    bestShift = (np.argmax(overlap)-300) * hop
                    self.pairwiseShifts[i, j] = bestShift
                    self.pairwiseShifts[j, i] = -bestShift
                    print("Calculated best overlap: %.1f seconds at deltat = %.2f" % (bestOverlap, bestShift))

                    # The full series of overlaps for each delta is needed for review, so store
                    self.overlaps_forplot.append({'series': overlap, 'bestOverlap': bestOverlap, 'bestShift': bestShift, 'recorders': (rec1, rec2)})

        # At this point, we have a matrix of best pairwise adjustments (in s)

        # Extract connected components (subgraphs, each is a list of rec indices):
        # will also determine the total shift for each recorder relative to component start
        # (=sum of edge weights from pairwiseShifts adj. matrix)
        # The shifts take into account user-set connectedness b/c shifts are added when traversing over recConnections.
        self.cclist = self.connectedComponents()

        print("Final global shifts:", self.allrecs, self.cclist)

        print("Clock adjustment complete")
        self.labelAdjust.setText("Clock adjustment complete")
        pm = self.style().standardIcon(QStyle.SP_DialogApplyButton).pixmap(QSize(12,12))
        self.labelAdjustIcon.setPixmap(pm)
        self.reviewAdjBtn.setEnabled(True)
        self.refreshMainOut()

    def showAdjustmentsDialog(self):
        self.adjrevdialog = ReviewAdjustments(self.overlaps_forplot, self.hopSpin.value(), self)
        self.adjrevdialog.exec_()

        self.cclist = self.connectedComponents()

        # update GUI after the dialog closes
        self.refreshMainOut()

    def showComparisonDialog(self):
        print("Extracting overlapping calls...")

        # self.allrecs and annots were set after browsing the input dir,
        # so here we just select the right annotations from them and pass to the dialog
        rec1, rec2 = self.compareRecSelector.currentText().split("-")
        annots1 = [sl for sl in self.annots if sl.recname==rec1]
        annots2 = [sl for sl in self.annots if sl.recname==rec2]

        # find the currently suggested shift for rec2 w.r.t rec1
        i = self.allrecs.index(rec1)
        j = self.allrecs.index(rec2)
        rec2shift = self.pairwiseShifts[i, j]

        # find the matching annotations from that pair of Segment Lists
        # ideally need the one with biggest overlap to the other
        # Or just: show an annotation if there is at least some overlap with another
        # TODO in the dialog maybe?
        annotspaired1 = []
        annotspaired2 = []
        for sl in annots1:
            slpaired1 = []
            slpaired2 = []

            # extract the right sl from annots2 (matching timestamp)
            # TODO this is very bodgy & might miss some annotation pairs:
            # as we're only looking for segment matches in the first overlapping file from annots2
            # - extreme case: file 00:00-15:00 cam have sl2match file -14:59-00:01
            # and then very few segment-level matches will be found.
            sl2match = None
            filelen1 = dt.timedelta(seconds=sl.metadata["Duration"])
            for sl2 in annots2:
                filelen2 = dt.timedelta(seconds=sl2.metadata["Duration"])
                if sl2.datetime-filelen1 < sl.datetime < sl2.datetime + filelen2:
                    sl2match = sl2
                    break

            if sl2match is None:
                print("Warning: no match for file %s found" % sl.wavname)
                continue

            # identify the matching segment from sl2
            for seg in sl:
                # find matching pair at segment level
                seg[0] = seg[0] + rec2shift
                seg[1] = seg[1] + rec2shift
                seg2 = self.findMatchIn(sl2match, seg)
                # store the pair in a temp list (including filenames)
                if len(seg2)>0:
                    seg.recname = rec1
                    seg2.recname = rec2
                    seg.wavname = sl.wavname
                    seg2.wavname = sl2.wavname
                    slpaired1.append(seg)
                    slpaired2.append(seg2)

            # add this set of segments to the main long list
            if len(slpaired1)>0:
                annotspaired1.extend(slpaired1)
                annotspaired2.extend(slpaired2)

        if len(annotspaired1)==0:
            print("Warning: no overlapping annotations found!")
            return

        print("Reviewing calls...")
        self.comparedialog = CompareCallsDialog(annotspaired1, annotspaired2, rec2shift, self)
        self.comparedialog.exec_()
        print("Call comparison complete")

    def findMatchIn(self, seglist, seg):
        """ Returns one segment from seglist that has the most overlap with seg. """
        overlaps = [min(seg[1], seg2[1]) - max(seg[0], seg2[0]) for seg2 in seglist]
        mpos = np.argmax(overlaps)
        if overlaps[mpos]<=0:
            print("No match found for segment", seg)
            return([])
        m = seglist[mpos]
        m.overlap = overlaps[mpos]
        return(m)

    def refreshMainOut(self):
        """ Collect information from self about shifts and connectivity
            and print out nicely. """
        adjstr = []
        for i in range(len(self.allrecs)):
            for j in range(i+1, len(self.allrecs)):
                pairtext = "Shift %s w.r.t. %s: %.2f" % (self.allrecs[i], self.allrecs[j], self.pairwiseShifts[i,j])
                if self.recConnections[i,j]==1:
                    adjstr.append(pairtext)
                else:
                    adjstr.append(pairtext + " (not connected)")

        self.adjustmentsOut.setPlainText("\n".join(adjstr))
        self.adjustmentsOut.setReadOnly(True)

        # cclist is a list of vertex ids
        ccstr = []
        for comp in self.cclist:
            recnames = []
            shifts = []
            for v in comp:
                recnames.append(self.allrecs[v[0]])
                shifts.append(v[1])
            recnames = "-".join(recnames)  # ZA-ZB-ZC
            shifts = "  ".join(map(str, shifts))  # 1  -2  4
            ccstr.append(recnames + "\t|\t" + shifts)
        ccstr = ";\n".join(ccstr)  # ZA-ZB-ZC  | 1  -2  4; ZD-ZE |  5 -10
        self.componentsOut.setPlainText(ccstr)
        self.componentsOut.setReadOnly(True)

    # depth-first search for finding a connected subgraph
    # and relative shift for each recorder within this subgraph
    # Args:
    # 1. list where to store found nodes
    # 2. starting node index
    # 3. Bool list of visited nodes
    # Return: list of tuples [(rec number, shift)]
    def DFS(self, temp, v, visited):
        # Mark the current vertex as visited
        visited[v] = True

        # Add this vertex and its shift to the current component
        if len(temp)>0:
            prevv = temp[-1][0]
            shift = temp[-1][1] + self.pairwiseShifts[v, prevv]
            print("total shift %.2f at %d / %s" %(shift, v, self.allrecs[v]))
        else:
            shift = 0
        temp.append((v, shift))

        # Run DFS for all vertices adjacent to this
        listOfNeighbours = np.nonzero(self.recConnections[v,])[0]
        for i in listOfNeighbours:
            if not visited[i]:
                temp = self.DFS(temp, i, visited)
        return temp

    # Retrieve connected components from an undirected graph
    # (and calculate within-component shifts as we are traversing the graph anyway)
    def connectedComponents(self):
        visited = [False] * len(self.allrecs)
        componentList = []
        for v in range(len(visited)):
            if not visited[v]:
                component = []
                componentList.append(self.DFS(component, v, visited))
                # Additionally, total sum of absolute adjustments could be minimized if we de-median the shifts
                # just need some testing before doing this
                # component[,1] = component[,1] - np.median(component[,1])

        print("Found the following components and shifts:")
        print(componentList)
        return componentList


class ReviewAdjustments(QDialog):
    def __init__(self, adj, hop, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Check Clock Adjustments')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

        self.parent = parent

        # list of dicts for each pair: {overlaps for each t, best overlap, best shift, (rec pair names)}
        self.shifts = adj
        self.xs = np.arange(-300*hop, 301*hop, hop)

        # top section
        self.labelCurrPage = QLabel("Page %s of %s" %(1, len(self.shifts)))
        self.labelRecs = QLabel()

        # middle section
        # The buttons to move through the overview
        self.leftBtn = QToolButton()
        self.leftBtn.setArrowType(Qt.LeftArrow)
        self.leftBtn.clicked.connect(self.moveLeft)
        self.leftBtn.setToolTip("View previous pair")
        self.rightBtn = QToolButton()
        self.rightBtn.setArrowType(Qt.RightArrow)
        self.rightBtn.clicked.connect(self.moveRight)
        self.rightBtn.setToolTip("View next pair")
        self.leftBtn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
        self.rightBtn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)

        self.connectCheckbox = QCheckBox("Recorders are connected")
        self.connectCheckbox.setChecked(self.parent.recConnections[0,1]==1)
        self.connectCheckbox.toggled.connect(self.connectToggled)

        # main graphic
        mainpl = pg.PlotWidget()
        self.p1 = mainpl.plot()
        self.p1.setPen(pg.mkPen('k', width=1))
        col = pg.mkColor(20, 255, 20, 150)
        self.bestLineO = pg.InfiniteLine(angle=0, movable=False, pen={'color': col, 'width': 2})
        self.bestLineSh = pg.InfiniteLine(angle=90, movable=False, pen={'color': col, 'width': 2})
        mainpl.addItem(self.bestLineO)
        mainpl.addItem(self.bestLineSh)

        # init plot with page 1 data
        self.currpage = 1
        self.leftBtn.setEnabled(False)
        if len(self.shifts)==1:
            self.rightBtn.setEnabled(False)
        self.setData()

        # accept / close
        closeBtn = QPushButton("Close")
        closeBtn.clicked.connect(self.accept)
        # cancelBtn = QPushButton("Cancel")
        # cancelBtn.clicked.connect(self.reject)

        # Layouts
        box = QVBoxLayout()
        topHBox = QHBoxLayout()
        topHBox.addWidget(self.labelCurrPage, stretch=1)
        topHBox.addWidget(self.labelRecs, stretch=3)
        box.addLayout(topHBox)

        midHBox = QHBoxLayout()
        midHBox.addWidget(self.leftBtn)
        midHBox.addWidget(mainpl, stretch=10)
        midHBox.addWidget(self.rightBtn)
        box.addLayout(midHBox)
        box.addWidget(self.connectCheckbox)

        bottomHBox = QHBoxLayout()
        bottomHBox.addWidget(closeBtn)
        # bottomHBox.addWidget(cancelBtn)
        box.addLayout(bottomHBox)
        self.setLayout(box)

    def connectToggled(self, ischecked):
        # convert recorder names to indices
        i = self.parent.allrecs.index(self.rec1)
        j = self.parent.allrecs.index(self.rec2)

        # change that connectivity edge
        if ischecked:
            self.parent.recConnections[i, j] = 1
        else:
            self.parent.recConnections[i, j] = 0

    def moveLeft(self):
        self.currpage = self.currpage - 1
        self.rightBtn.setEnabled(True)
        if self.currpage==1:
            self.leftBtn.setEnabled(False)
        self.setData()

    def moveRight(self):
        self.currpage = self.currpage + 1
        self.leftBtn.setEnabled(True)
        if self.currpage==len(self.shifts):
            self.rightBtn.setEnabled(False)
        self.setData()

    def setData(self):
        # update plot etc
        currdata = self.shifts[self.currpage-1]
        self.p1.setData(y=currdata['series'], x=self.xs)
        self.bestLineO.setValue(currdata['bestOverlap'])
        self.bestLineSh.setValue(currdata['bestShift'])
        self.labelCurrPage.setText("Page %s of %s" %(self.currpage, len(self.shifts)))
        self.rec1 = currdata['recorders'][0]
        self.rec2 = currdata['recorders'][1]
        self.labelRecs.setText("Suggested shift for recorder %s w.r.t. %s" % (self.rec1, self.rec2))
        i = self.parent.allrecs.index(self.rec1)
        j = self.parent.allrecs.index(self.rec2)
        self.connectCheckbox.setChecked(self.parent.recConnections[i, j]==1)


class CompareCallsDialog(QDialog):
    # Init args:
    # 1. list of SegmentLists for rec1
    # 2. list of SegmentLists for rec2
    # 3. best shift for rec2
    def __init__(self, annots1, annots2, rec2shift, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Compare Call Pairs')
        self.setWindowIcon(QIcon('img/Avianz.ico'))
        self.setWindowFlags((self.windowFlags() ^ Qt.WindowContextHelpButtonHint) | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

        self.parent = parent
        self.rec1 = annots1[0].recname
        self.rec2 = annots2[0].recname

        self.annots1 = annots1
        self.annots2 = annots2

        # Set colour map
        # cmap = self.config['cmap']
        cmap = "Grey"
        pos, colour, mode = colourMaps.colourMaps(cmap)
        cmap = pg.ColorMap(pos, colour,mode)
        self.lut = cmap.getLookupTable(0.0, 1.0, 256)

        self.topLabel = QLabel("Comparing recorders %s (top, shifted by %.1f) and %s (bottom)" %(self.rec1, rec2shift, self.rec2))
        self.labelPair = QLabel()

        # spectrograms
        self.wPlot = pg.GraphicsLayoutWidget()
        self.pPlot1 = self.wPlot.addViewBox(enableMouse=False, row=0, col=1)
        self.pPlot2 = self.wPlot.addViewBox(enableMouse=False, row=1, col=1)

        self.plot1 = pg.ImageItem()
        self.plot2 = pg.ImageItem()
        self.pPlot1.addItem(self.plot1)
        self.pPlot2.addItem(self.plot2)

        # Y (freq) axis
        self.sg_axis1 = pg.AxisItem(orientation='left')
        self.sg_axis1.linkToView(self.pPlot1)
        self.sg_axis2 = pg.AxisItem(orientation='left')
        self.sg_axis2.linkToView(self.pPlot2)
        self.wPlot.addItem(self.sg_axis1, row=0, col=0)
        self.wPlot.addItem(self.sg_axis2, row=1, col=0)

        # The buttons to move through the overview
        self.leftBtn = QToolButton()
        self.leftBtn.setArrowType(Qt.LeftArrow)
        self.leftBtn.clicked.connect(self.moveLeft)
        self.leftBtn.setToolTip("View previous pair")
        self.rightBtn = QToolButton()
        self.rightBtn.setArrowType(Qt.RightArrow)
        self.rightBtn.clicked.connect(self.moveRight)
        self.rightBtn.setToolTip("View next pair")
        self.leftBtn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
        self.rightBtn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)

        # accept / close
        closeBtn = QPushButton("Close")
        closeBtn.clicked.connect(self.accept)

        # init GUI with page 1 data
        self.currpage = 1
        self.leftBtn.setEnabled(False)
        if len(self.annots1)==1:
            self.rightBtn.setEnabled(False)
        self.setData()

        # layout
        box = QVBoxLayout()
        box.addWidget(self.topLabel)
        box.addWidget(self.labelPair)
        midHBox = QHBoxLayout()
        midHBox.addWidget(self.leftBtn)
        midHBox.addWidget(self.wPlot, stretch=10)
        midHBox.addWidget(self.rightBtn)
        box.addLayout(midHBox)
        box.addWidget(closeBtn)
        self.setLayout(box)

    def moveLeft(self):
        self.currpage = self.currpage - 1
        self.rightBtn.setEnabled(True)
        if self.currpage==1:
            self.leftBtn.setEnabled(False)
        self.setData()

    def moveRight(self):
        self.currpage = self.currpage + 1
        self.leftBtn.setEnabled(True)
        if self.currpage==len(self.annots1):
            self.rightBtn.setEnabled(False)
        self.setData()

    def setData(self):
        # update plot etc
        currseg1 = self.annots1[self.currpage-1]
        currseg2 = self.annots2[self.currpage-1]

        wav1 = currseg1.wavname
        wav2 = currseg2.wavname

        wav1start = currseg1[0]
        wav2start = currseg2[0]
        wav1len = currseg1[1]
        wav2len = currseg2[1]

        self.sp1 = SignalProc.SignalProc(256, 128)
        self.sp1.readWav(wav1, off=wav1start, len=wav1len)
        self.sp2 = SignalProc.SignalProc(256, 128)
        self.sp2.readWav(wav2, off=wav2start, len=wav2len)
        _ = self.sp1.spectrogram()
        _ = self.sp2.spectrogram()

        self.plot1.setImage(self.sp1.sg)
        self.plot2.setImage(self.sp2.sg)
        self.plot1.setLookupTable(self.lut)
        self.plot2.setLookupTable(self.lut)
        # TODO no color sliders yet
        # self.setColourLevels()
        # No scroll area yet
        # self.scroll.horizontalScrollBar().setValue(0)

        # axes
        minFreq = 0
        maxFreq = min(self.sp1.sampleRate, self.sp2.sampleRate)
        FreqRange = (maxFreq-minFreq)/1000.

        SgSize1 = np.shape(self.sp1.sg)[1]
        SgSize2 = np.shape(self.sp2.sg)[1]
        ticks1 = [(0,minFreq/1000.), (SgSize1/4, minFreq/1000.+FreqRange/4.), (SgSize1/2, minFreq/1000.+FreqRange/2.), (3*SgSize1/4, minFreq/1000.+3*FreqRange/4.), (SgSize1,minFreq/1000.+FreqRange)]
        ticks1 = [[(tick[0], "%.1f" % tick[1] ) for tick in ticks1]]
        ticks2 = [(0,minFreq/1000.), (SgSize2/4, minFreq/1000.+FreqRange/4.), (SgSize2/2, minFreq/1000.+FreqRange/2.), (3*SgSize2/4, minFreq/1000.+3*FreqRange/4.), (SgSize2,minFreq/1000.+FreqRange)]
        ticks2 = [[(tick[0], "%.1f" % tick[1] ) for tick in ticks2]]
        self.sg_axis1.setTicks(ticks1)
        self.sg_axis2.setTicks(ticks2)
        self.sg_axis1.setLabel('kHz')
        self.sg_axis2.setLabel('kHz')

        # info fields
        self.labelPair.setText("Showing calls at %.1f-%.1f s (adjusted) and %.1f-%.1f s" % (currseg1[0], currseg1[1], currseg2[0], currseg2[1]))

        # self.labelCurrPage.setText("Page %s of %s" %(self.currpage, len(self.shifts)))
        # i = self.parent.allrecs.index(self.rec1)
        # j = self.parent.allrecs.index(self.rec2)
        # self.connectCheckbox.setChecked(self.parent.recConnections[i, j]==1)

#### MAIN LAUNCHER

pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
pg.setConfigOption('antialias',True)

print("Starting AviaNZ call comparator")
app = QApplication(sys.argv)
mainwindow = CompareCalls()
mainwindow.show()
app.exec_()
print("Processing complete, closing AviaNZ call comparator")
QApplication.closeAllWindows()
