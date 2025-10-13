
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

# File management widgets for the AviaNZ program

import os
import pyqtgraph as pg
import pyqtgraph.functions as fn
import soundfile as sf
from PyQt6.QtWidgets import QListWidget, QListWidgetItem
from PyQt6.QtCore import Qt, QDir
from PyQt6.QtGui import QPixmap, QPainter, QIcon, QColor
from src.core import annotation


class LightedFileList(QListWidget):
    """ File list with traffic light icons.
        On init (or after any change), pass the red, darkyellow, and green colors.
    """
    def __init__(self, ColourNone, ColourPossibleDark, ColourNamed):
        super().__init__()
        self.ColourNone = ColourNone
        self.ColourPossibleDark = ColourPossibleDark
        self.ColourNamed = ColourNamed
        self.soundDir = None
        self.spList = set()
        self.fsList = set()
        self.listOfFiles = []
        self.minCertainty = 100
        self.setMinimumWidth(150)

        # for the traffic light icons
        self.blackpen = fn.mkPen(color=(160,160,160,255), width=2)
        self.tempsl = annotation.SegmentList()

    def fill(self, soundDir, fileName, recursive=False, readFmt=False, addWavNum=False):
        """ read folder contents, populate the list widget.
            soundDir: current dir
            fileName: file which should be selected, or None
            recursive: should we read the species list/format info from subdirs as well?
            readFmt: should we read the wav header as well?
            addWavNum: add extra info to the end of dir names
        """
        # clear current listbox
        self.clearSelection()
        self.clearFocus()
        self.clear()
        # collect some additional info about the current dir
        self.spList = set()
        self.fsList = set()
        self.listOfFiles = []
        self.minCertainty = 100     # TODO: not used in training, can remove?

        with pg.BusyCursor():
            # Read contents of current dir
            self.listOfFiles = QDir(soundDir).entryInfoList(['..','*.wav','*.bmp','*.flac'],filters=QDir.Filter.AllDirs | QDir.Filter.NoDot | QDir.Filter.Files,sort=QDir.SortFlag.DirsFirst)
            self.soundDir = soundDir

            for file in self.listOfFiles:
                # add entry to the list
                item = QListWidgetItem(self)

                if file.isDir():
                    if file.fileName()=="..":
                        item.setText(file.fileName() + "/")
                        continue

                    # detailed dir view can be used for non-clickable instances
                    if addWavNum:
                        # count wavs in this dir:
                        numbmps = 0
                        numwavs = 0
                        numflacs = 0
                        for root, dirs, files in os.walk(file.filePath()):
                            numwavs += sum(f.lower().endswith('.wav') for f in files)
                            numbmps += sum(f.lower().endswith('.bmp') for f in files)
                            numflacs += sum(f.lower().endswith('.flac') for f in files)
                        # keep these strings as short as possible
                        if numbmps==0:
                            item.setText("%s/\t\t(%d wav files)" % (file.fileName(), numwavs))
                        elif numwavs==0:
                            item.setText("%s/\t\t(%d bmp files)" % (file.fileName(), numbmps))
                        elif numflacs==0:
                            item.setText("%s/\t\t(%d flac files)" % (file.fileName(), numflacs))
                        else:
                            item.setText("%s/\t\t(%d wav, %d bmp, %d flac files)" % (file.fileName(), numwavs, numbmps, numflacs))
                    else:
                        item.setText(file.fileName() + "/")

                    # We still might need to walk the subfolders for sp lists and wav formats!
                    if not recursive:
                        continue
                    for root, dirs, files in os.walk(file.filePath()):
                        for filename in files:
                            filenamef = os.path.join(root, filename)
                            if filename.lower().endswith('.wav') or filename.lower().endswith('.bmp') or filename.lower().endswith('.flac'):
                                if readFmt:
                                    if filename.lower().endswith('.wav'):
                                        try:
                                            info = sf.info(filenamef)
                                            samplerate = info.samplerate
                                            self.fsList.add(samplerate)
                                        except Exception as e:
                                            print("Warning: could not parse format of WAV file", filenamef)
                                            print(e)
                                    elif filename.lower().endswith('.flac'):
                                        try:
                                            info = sf.info(filenamef)
                                            samplerate = info.samplerate
                                            self.fsList.add(samplerate)
                                        except Exception as e:
                                            print("Warning: could not parse format of FLAC file", filenamef)
                                            print(e)
                                    else:
                                        # For bitmaps, using hardcoded samplerate as there's no readFmt
                                        self.fsList.add(176000)

                                # Data files can accompany either wavs or bmps
                                dataf = filenamef + '.data'
                                if os.path.isfile(dataf):
                                    try:
                                        self.tempsl.parseJSON(dataf, silent=True)
                                        if len(self.tempsl)>0:
                                            # collect any species present
                                            filesp = [lab["species"] for seg in self.tempsl for lab in seg.labels]
                                            self.spList.update(filesp)
                                            # min certainty
                                            cert = [lab["certainty"] for seg in self.tempsl for lab in seg.labels]
                                            if cert:
                                                mincert = min(cert)
                                                if self.minCertainty > mincert:
                                                    self.minCertainty = mincert
                                    except Exception as e:
                                        # .data exists, but unreadable
                                        print("Could not read DATA file", dataf)
                                        print(e)
                else:
                    item.setText(file.fileName())

                    # check for a data file here and color this entry based on that
                    fullname = os.path.join(soundDir, file.fileName())
                    # (also updates the directory info sets, and minCertainty)
                    self.paintItem(item, fullname+'.data')
                    # format collection only implemented for WAVs currently
                    if readFmt:
                        if fullname.lower().endswith('.wav'):
                            try:
                                info = sf.info(fullname)
                                samplerate = info.samplerate
                                self.fsList.add(samplerate)
                            except Exception as e:
                                print("Warning: could not parse format of WAV file", fullname)
                                print(e)
                        elif fullname.lower().endswith('.flac'):
                            try:
                                info = sf.info(fullname)
                                samplerate = info.samplerate
                                self.fsList.add(samplerate)
                            except Exception as e:
                                print("Warning: could not parse format of FLAC file", fullname)
                                print(e)
                        else:
                            # For bitmaps, using hardcoded samplerate as there's no readFmt
                            self.fsList.add(176000)

        if readFmt:
            print("Found the following Fs:", self.fsList)

        # mark the current file or first row (..), if not found
        if fileName:
            # for matching dirs:
            # index = self.findItems(fileName+"\/",Qt.MatchExactly)
            index = self.findItems(fileName,Qt.MatchFlag.MatchExactly)
            if len(index)>0:
                self.setCurrentItem(index[0])
            else:
                self.setCurrentRow(0)

    def refreshFile(self, fileName, cert):
        """ Repaint a single file icon with the provided certainty.
            fileName: file stem (dir will be read from self)
            cert:     0-100, or -1 if no annotations
        """
        # for matching dirs - not sure if needed:
        # index = self.findItems(fileName+"\/",Qt.MatchExactly)
        index = self.findItems(fileName,Qt.MatchFlag.MatchExactly)
        if len(index)==0:
            return

        curritem = index[0]
        # Create a new pixmap for each icon to avoid concurrent painter access
        pixmap = QPixmap(10, 10)
        # Repainting identical to paintItem
        if cert == -1:
            # .data exists, but no annotations
            pixmap.fill(QColor(255,255,255,0))
            painter = QPainter(pixmap)
            painter.setPen(self.blackpen)
            painter.drawRect(pixmap.rect())
            painter.end()
            curritem.setIcon(QIcon(pixmap))
            # no change to self.minCertainty
        elif cert == 0:
            pixmap.fill(self.ColourNone)
            curritem.setIcon(QIcon(pixmap))
            self.minCertainty = 0
        elif cert < 100:
            pixmap.fill(self.ColourPossibleDark)
            curritem.setIcon(QIcon(pixmap))
            self.minCertainty = min(self.minCertainty, cert)
        else:
            pixmap.fill(self.ColourNamed)
            curritem.setIcon(QIcon(pixmap))
            # self.minCertainty cannot be changed by a cert=100 segment

    def paintItem(self, item, datafile):
        """ Read the JSON and draw the traffic light for a single item """
        filesp = []
        if os.path.isfile(datafile):
            # Try loading the segments to get min certainty
            try:
                self.tempsl.parseJSON(datafile, silent=True)
                if len(self.tempsl)==0:
                    # .data exists, but empty - "file was looked at"
                    mincert = -1
                else:
                    cert = [lab["certainty"] for seg in self.tempsl for lab in seg.labels]
                    if cert:
                        mincert = min(cert)
                    else:
                        mincert = -1
                    # also collect any species present
                    filesp = [lab["species"] for seg in self.tempsl for lab in seg.labels]
            except Exception as e:
                # .data exists, but unreadable
                print("Could not determine certainty for file", datafile)
                print(e)
                mincert = -1

            # Create a new pixmap for each icon to avoid concurrent painter access
            pixmap = QPixmap(10, 10)
            if mincert == -1:
                # .data exists, but no annotations
                pixmap.fill(QColor(255,255,255,0))
                painter = QPainter(pixmap)
                painter.setPen(self.blackpen)
                painter.drawRect(pixmap.rect())
                painter.end()
                item.setIcon(QIcon(pixmap))

                # no change to self.minCertainty
            elif mincert == 0:
                pixmap.fill(self.ColourNone)
                item.setIcon(QIcon(pixmap))
                self.minCertainty = 0
            elif mincert < 100:
                pixmap.fill(self.ColourPossibleDark)
                item.setIcon(QIcon(pixmap))
                self.minCertainty = min(self.minCertainty, mincert)
            else:
                pixmap.fill(self.ColourNamed)
                item.setIcon(QIcon(pixmap))
                # self.minCertainty cannot be changed by a cert=100 segment
        else:
            # no .data for this sound file
            pixmap = QPixmap(10, 10)
            pixmap.fill(QColor(255,255,255,0))
            item.setIcon(QIcon(pixmap))

        # collect some extra info about this file as we've read it anyway
        self.spList.update(filesp)