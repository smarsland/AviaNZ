# -*- coding: utf-8 -*-

# Wrapper script to SplitWav audio splitter.
# Splits wavs, and AviaNZ-format annotation files.

# Version 3.0 14/09/20
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti
# This file: Julius Juodakis

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2020

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


from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QPushButton, QPlainTextEdit, QWidget, QGridLayout, QSpinBox, QGroupBox, QSizePolicy, QSpacerItem, QLayout, QProgressDialog, QStyle
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QIcon
import sys
import os
import datetime as dt

# sys.path.append('..')
from ext import SplitLauncher
import Segment
import SupportClasses_GUI


class SplitData(QMainWindow):
    def __init__(self):
        super(SplitData, self).__init__()
        print("Starting AviaNZ WAV splitter")
        self.setWindowTitle("AviaNZ WAV splitter")

        self.dirName = []
        self.dirO = []
        self.indirOk = False
        self.outdirOk = False
        self.cutLen = 1

        # menu bar
        fileMenu = self.menuBar()#.addMenu("&File")
        fileMenu.addAction("About", lambda: SupportClasses_GUI.MessagePopup("a", "About", ".").exec_())
        fileMenu.addAction("Quit", QApplication.quit)
        # do we need this?
        # if platform.system() == 'Darwin':
        #    helpMenu.addAction("About",self.showAbout,"Ctrl+A")

        # main dock setup
        area = QWidget()
        grid = QGridLayout()
        area.setLayout(grid)
        self.setCentralWidget(area)


        ## input
        label = QLabel("Select input folder with files to split:")
        self.w_browse = QPushButton(" Browse Folder")
        self.w_browse.setToolTip("Warning: files inside subfolders will not be processed!")
        self.w_browse.setMinimumSize(170, 40)
        self.w_browse.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.w_browse.setStyleSheet('QPushButton {background-color: #c4ccd3; font-weight: bold; font-size:14px; padding: 3px 3px 3px 3px}')
        self.w_browse.clicked.connect(self.browse)
        self.w_browse.setSizePolicy(QSizePolicy(1,1))

        # area showing the selected folder
        self.w_dir = QPlainTextEdit()
        self.w_dir.setFixedHeight(40)
        self.w_dir.setPlainText('')
        self.w_dir.setToolTip("The folder being processed")
        self.w_dir.setSizePolicy(QSizePolicy(3,1))


        ## output
        labelO = QLabel("Select folder for storing split output:")
        self.w_browseO = QPushButton(" Browse Folder")
        self.w_browseO.setMinimumSize(170, 40)
        self.w_browseO.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.w_browseO.setStyleSheet('QPushButton {background-color: #c4ccd3; font-weight: bold; font-size:14px; padding: 3px 3px 3px 3px}')
        self.w_browseO.clicked.connect(self.browseO)
        self.w_browseO.setSizePolicy(QSizePolicy(1,1))

        # area showing the selected folder
        self.w_dirO = QPlainTextEdit()
        self.w_dirO.setFixedHeight(40)
        self.w_dirO.setPlainText('')
        self.w_dirO.setToolTip("Split files will be placed here")
        self.w_dirO.setSizePolicy(QSizePolicy(3,1))


        ## split length
        self.titleCutLen = QLabel("Set split file duration, in seconds:")
        self.labelCutLen = QLabel("")
        self.boxCutLen = QSpinBox()
        self.boxCutLen.setRange(1,3600*24)
        self.boxCutLen.setValue(60)

        ## start
        self.labelWavs = QLabel("")
        self.labelWavs.setWordWrap(True)
        self.labelDatas = QLabel("")
        self.labelDatas.setWordWrap(True)
        self.labelDirs = QLabel("")
        self.labelDirs.setWordWrap(True)
        self.labelOut = QLabel("")
        self.labelSum = QLabel("")
        self.boxCutLen.valueChanged.connect(self.setCutLen)
        self.setCutLen(self.boxCutLen.value())

        self.splitBut = QPushButton(" &Split!")
        self.splitBut.setFixedHeight(40)
        self.splitBut.setStyleSheet('QPushButton {background-color: #95b5ee; font-weight: bold; font-size:14px} QPushButton:disabled {background-color :#B3BCC4}')
        self.splitBut.clicked.connect(self.split)
        self.splitBut.setEnabled(False)


        ## groups
        inputGroup = QGroupBox("Input")
        inputGrid = QGridLayout()
        inputGroup.setLayout(inputGrid)
        inputGrid.addWidget(label, 0, 0, 1, 4)
        inputGrid.addWidget(self.w_browse, 1, 0, 1, 1)
        inputGrid.addWidget(self.w_dir, 1, 1, 1, 3)
        inputGrid.addWidget(self.labelWavs, 2, 0, 1, 4)
        inputGrid.addWidget(self.labelDatas, 3, 0, 1, 4)
        inputGrid.addWidget(self.labelDirs, 4, 0, 1, 4)

        outputGroup = QGroupBox("Output")
        outputGrid = QGridLayout()
        outputGroup.setLayout(outputGrid)
        outputGrid.addWidget(labelO, 0, 0, 1, 4)
        outputGrid.addWidget(self.w_browseO, 1, 0, 1, 1)
        outputGrid.addWidget(self.w_dirO, 1, 1, 1, 3)
        outputGrid.addWidget(self.titleCutLen, 2, 0, 1, 4)
        outputGrid.addWidget(self.boxCutLen, 3, 0, 1, 1)
        outputGrid.addWidget(self.labelCutLen, 3, 1, 1, 3)
        outputGrid.addWidget(self.labelOut, 4, 0, 1, 4)

        ## add everything to the main layout
        grid.addWidget(inputGroup, 0, 0, 2, 4)
        grid.addWidget(outputGroup, 2, 0, 2, 4)
        grid.addItem(QSpacerItem(4, 0, 1, 4))
        grid.addWidget(self.labelSum, 5, 0, 1, 4)
        grid.addWidget(self.splitBut, 6, 1, 1, 2)

        #inputGrid.setSizeConstraint(QLayout.SetFixedSize)
        #outputGrid.setSizeConstraint(QLayout.SetFixedSize)
        inputGroup.setSizePolicy(QSizePolicy(1,5))
        inputGroup.setMinimumSize(400, 220)
        outputGroup.setSizePolicy(QSizePolicy(1,5))
        outputGroup.setMinimumSize(400, 180)
        grid.setSizeConstraint(QLayout.SetMinimumSize)
        area.setSizePolicy(QSizePolicy(1,5))
        area.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy(1,1))
        self.setMinimumSize(200, 400)
        self.show()

    def browse(self):
        if self.dirName:
            self.dirName = QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirName))
        else:
            self.dirName = QFileDialog.getExistingDirectory(self,'Choose Folder to Process')
        self.w_dir.setPlainText(self.dirName)
        self.w_dir.setReadOnly(True)
        self.fillFileList()
        if self.indirOk and self.outdirOk:
            self.splitBut.setEnabled(True)
        else:
            self.splitBut.setEnabled(False)

    def browseO(self):
        if self.dirO:
            self.dirO = QFileDialog.getExistingDirectory(self,'Choose Folder to Process',str(self.dirO))
        else:
            self.dirO = QFileDialog.getExistingDirectory(self,'Choose Folder to Process')
        self.w_dirO.setPlainText(self.dirO)
        self.w_dirO.setReadOnly(True)

        # Ideally, should check if output file names are free
        if not os.access(self.dirO, os.W_OK | os.X_OK):
            self.labelOut.setText("ERROR: selected output folder not writeable")
            self.outdirOk = False
        elif not QDir(self.dirO).isEmpty():
            self.labelOut.setText("Warning: selected output folder not empty")
            self.outdirOk = True
        else:
            self.labelOut.setText("Folder looks good")
            self.outdirOk = True

        if self.indirOk and self.outdirOk:
            self.splitBut.setEnabled(True)
        else:
            self.splitBut.setEnabled(False)

    def setCutLen(self, time):
        """ Parses the split length spinbox value """
        if time==0:
            print("ERROR: cannot set cut length to 0!")
            return
        self.cutLen = int(time)
        min, s = divmod(time, 60)
        hr, min = divmod(min, 60)
        self.labelCutLen.setText("= %d hr %02d min %02d s" % (hr, min, s))
        if self.indirOk:
            self.labelSum.setText("Will split %d WAV files and %d DATA files into pieces of %d min %d s." % (len(self.listOfWavs), len(self.listOfDataFiles), self.cutLen // 60, self.cutLen % 60))
        else:
            self.labelSum.setText("Please select files to split")

    def fillFileList(self):
        """ Generates the list of files for the file listbox.
        Most of the work is to deal with directories in that list.
        It only sees *.data and *.wav files."""

        if not os.path.isdir(self.dirName):
            print("ERROR: directory %s doesn't exist" % self.dirName)
            return

        listOfDirs = QDir(self.dirName).entryList(['..'],filters=QDir.AllDirs | QDir.NoDotAndDotDot )
        self.listOfWavs = QDir(self.dirName).entryList(['*.wav'])
        self.listOfDataFiles = QDir(self.dirName).entryList(['*.wav.data'])

        # check if files have timestamps:
        haveTime = 0
        for f in self.listOfWavs:
            infilestem = f[:-4]
            try:
                datestamp = infilestem.split("_")[-2:] # get [date, time]
                datestamp = '_'.join(datestamp) # make "date_time"
                # check both 4-digit and 2-digit codes (century that produces closest year to now is inferred)
                try:
                    d = dt.datetime.strptime(datestamp, "%Y%m%d_%H%M%S")
                except ValueError:
                    d = dt.datetime.strptime(datestamp, "%y%m%d_%H%M%S")
                haveTime += 1
            except ValueError:
                print("Could not identify timestamp in", f)

        for f in self.listOfDataFiles:
            infilestem = f[:-9]
            try:
                datestamp = infilestem.split("_")[-2:] # get [date, time]
                datestamp = '_'.join(datestamp) # make "date_time"
                # check both 4-digit and 2-digit codes (century that produces closest year to now is inferred)
                try:
                    d = dt.datetime.strptime(datestamp, "%Y%m%d_%H%M%S")
                except ValueError:
                    d = dt.datetime.strptime(datestamp, "%y%m%d_%H%M%S")
                haveTime += 1
            except ValueError:
                print("Could not identify timestamp in", f)
        # Currently, haveTime sums are not used anywhere...

        # check the selected dir and print info
        if len(listOfDirs)==0:
            self.labelDirs.setText("Folder looks good (no subfolders)")
        elif len(listOfDirs)<4:
            self.labelDirs.setText("Warning: detected subfolders will not be processed: %s" % ", ".join(listOfDirs))
        else:
            self.labelDirs.setText("Warning: detected subfolders will not be processed: %s..." % ", ".join(listOfDirs[:3]))

        if len(self.listOfWavs)==0:
            self.labelWavs.setText("ERROR: no WAV files detected!")
            noWav = True
        elif len(self.listOfWavs)<4:
            self.labelWavs.setText("Found <b>%d</b> WAV files: %s" % (len(self.listOfWavs), ", ".join(self.listOfWavs)))
            noWav = False
        else:
            self.labelWavs.setText("Found <b>%d</b> WAV files: %s..." % (len(self.listOfWavs), ", ".join(self.listOfWavs[:3])))
            noWav = False

        if len(self.listOfDataFiles)==0:
            self.labelDatas.setText("No DATA files detected")
            noData = True
        elif len(self.listOfDataFiles)<4:
            self.labelDatas.setText("Found <b>%d</b> DATA files: %s" % (len(self.listOfDataFiles), ", ".join(self.listOfDataFiles)))
            noData = False
        else:
            self.labelDatas.setText("Found <b>%d</b> DATA files: %s..." % (len(self.listOfDataFiles), ", ".join(self.listOfDataFiles[:3])))
            noData = False

        self.indirOk = not (noData and noWav)

        if self.indirOk:
            self.labelSum.setText("Will split %d WAV files and %d DATA files into pieces of %d min %d s." % (len(self.listOfWavs), len(self.listOfDataFiles), self.cutLen // 60, self.cutLen % 60))
        else:
            self.labelSum.setText("Please select files to split")


    def split(self):
        """ This function is connected to the main button press """
        # setup progress bar etc
        print("Starting to split...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        totalfiles = len(self.listOfDataFiles) + len(self.listOfWavs)
        dlg = QProgressDialog("Splitting...", "", 0, totalfiles, self)
        donefiles = 0
        dlg.setCancelButton(None)
        dlg.setWindowIcon(QIcon('img/Avianz.ico'))
        dlg.setWindowTitle('AviaNZ')
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(1)
        dlg.forceShow()

        # do the wav files
        for f in self.listOfWavs:
            # output is passed as the same file name in different dir -
            # the splitter will figure out if numbers or times need to be attached
            infile_c = os.path.join(self.dirName, f).encode('ascii')

            # To avoid dealing with strptime too much which is missing on Win,
            # we check the format here - but we can't really pass the C-format struct entirely
            wavHasDt = int(0)
            try:
                wavstring = f[:-4].split("_")  # get [recorder, date, time]
                wavdt = '_'.join(wavstring[-2:])  # make "date_time"
                # check both 4-digit and 2-digit codes (century that produces closest year to now is inferred)
                try:
                    wavdt = dt.datetime.strptime(wavdt, "%Y%m%d_%H%M%S")
                except ValueError:
                    wavdt = dt.datetime.strptime(wavdt, "%y%m%d_%H%M%S")
                print(f, "identified as timestamp", wavdt)
                wavHasDt = int(1)
                # Here, we remake the out file name to always have 4 digit years, to make life easier in the C part
                outfile_c = '_'.join(wavstring[:-2])
                outfile_c =  outfile_c + '_' + dt.datetime.strftime(wavdt, "%Y%m%d_%H%M%S") + '.wav'
                outfile_c = os.path.join(self.dirO, outfile_c).encode('ascii')
            except ValueError:
                print("Could not identify timestamp in", f)
                outfile_c = os.path.join(self.dirO, f).encode('ascii')
                wavHasDt = int(0)

            if os.path.isfile(infile_c) and os.stat(infile_c).st_size>100:
                # check if file is formatted correctly
                with open(infile_c, 'br') as f:
                    if f.read(4) != b'RIFF':
                        print("Warning: file %s not formatted correctly, skipping" % infile_c)
                        return

                succ = SplitLauncher.launchCython(infile_c, outfile_c, self.cutLen, wavHasDt)
                if succ!=0:
                    print("ERROR: C splitter failed on file", f)
                    return
            else:
                print("Warning: input file %s does not exist or is empty, skipping", infile_c)

            donefiles += 1
            QApplication.processEvents()
            dlg.repaint()
            dlg.forceShow()
            dlg.setValue(donefiles)

        # do the data files
        for f in self.listOfDataFiles:
            self.splitData(os.path.join(self.dirName,f), self.dirO, self.cutLen)
            donefiles += 1
            QApplication.processEvents()
            dlg.repaint()
            dlg.forceShow()
            dlg.setValue(donefiles)

        print("processed %d files" % donefiles)
        QApplication.restoreOverrideCursor()
        if donefiles==totalfiles:
            msg = SupportClasses_GUI.MessagePopup("d", "Finished", "Folder processed successfully!")
            msg.exec_()


    def splitData(self, infile, outdir, cutlen):
        """ Args: input filename, output folder, split duration.
            Determines the original input length from the metadata segment[1].
        """
        print("Splitting data file", infile)
        segs = Segment.SegmentList()
        try:
            segs.parseJSON(infile)
        except Exception as e:
            print(e)
            print("ERROR: could not parse file", infile)
            return

        infile = os.path.basename(infile)[:-9]
        try:
            outprefix = '_'.join(infile.split("_")[:-2])
            datestamp = infile.split("_")[-2:]  # get [date, time]
            datestamp = '_'.join(datestamp)  # make "date_time"
            try:
                time = dt.datetime.strptime(datestamp, "%Y%m%d_%H%M%S")
            except ValueError:
                time = dt.datetime.strptime(datestamp, "%y%m%d_%H%M%S")
            print(infile, "identified as timestamp", time)
        except ValueError:
            outprefix = infile
            print("Could not identify timestamp in", infile)
            time = 0

        maxtime = segs.metadata["Duration"]
        if maxtime<=0:
            print("ERROR: bad audio duration %s read from .data" % maxtime)
            return
        elif maxtime>24*3600:
            print("ERROR: audio duration %s in .data exceeds 24 hr limit" % maxtime)
            return

        # repeat initial meta-segment for each output file
        # (output is determined by ceiling division)
        all = []
        for i in range(int(maxtime-1) // cutlen + 1):
            onelist = Segment.SegmentList()
            onelist.metadata = segs.metadata.copy()
            onelist.metadata["Duration"] = min(self.cutLen, maxtime-i*self.cutLen)
            all.append(onelist)

        # separate segments into output files and adjust segment timestamps
        for b in segs:
            filenum, adjst = divmod(b[0], cutlen)
            adjend = b[1] - filenum*cutlen
            # a segment can jut out past the end of a split file, so we trim it:
            # [a------|---b] -> [a-----f1end] [f2start----b]
            # If it's super long, it'll go back to the list to be trimmed again.
            if adjend > cutlen:
                print("trimming segment")
                # cut at the end of the starting file
                adjend = (filenum+1)*cutlen
                # keep rest for later
                segs.append([adjend, b[1], b[2], b[3], b[4]])

            all[int(filenum)].addSegment([adjst, adjend, b[2], b[3], b[4]])

        # save files, while increasing the filename datestamps
        for a in range(len(all)):
            if time!=0:
                f2 = str(outprefix) + '_' + dt.datetime.strftime(time, "%Y%m%d_%H%M%S") + '.wav.data'
                f2 = os.path.join(outdir, f2)
                print("outputting to", f2)
                time = time + dt.timedelta(seconds=cutlen)
            else:
                f2 = str(outprefix) + '_' + str(a) + '.wav.data'
                f2 = os.path.join(outdir, f2)
                print("outputting to", f2)
            all[a].saveJSON(f2)


#### MAIN LAUNCHER, for standalone exe version:

# print("Starting AviaNZ WAV splitter")
# app = QApplication(sys.argv)
# splitter = SplitData()
# splitter.show()
# app.exec_()
# print("Processing complete, closing AviaNZ WAV splitter")
# QApplication.closeAllWindows()
