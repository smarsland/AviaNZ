# AviaNZ_batch.py
#
# Runs filters over lots of data

# Version 4.0 9/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

# This is the processing class for the batch AviaNZ interface

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

import gc, os, re, fnmatch
import time
import datetime as dt
import numpy as np

from src.core import Spectrogram
from src.core import SignalProc
from src.core import Segment
from src.core import WaveletSegment
from src.core import SupportClasses
from core.BirdDetector import BirdDetector
from core.BatDetector import BatDetector

import traceback
import math
import copy

import soundfile as sf

class AviaNZ_batchProcess():
    # Main class for batch processing
    # Parent: AviaNZ_batchWindow
    # mode: "GUI/CLI/test". If GUI, must provide the parent
        # Recogniser - filter file name without ".txt" 
        # TODO: allow CLI to have multiple recognisers and other options

    def __init__(self, parent, mode="GUI", configdir='', sdir='', recognisers=None, subset=False, intermittent=False, wind="None", mergeSyllables=False, overwrite=True, timeWindow_s=0, timeWindow_e=0, protocolSize=15, protocolInterval=300, maxgap=1, minlen=0.2, maxlen=10):
        # Read config and filters from user location
        # recognisers - list of filter file names without ".txt"
        # timeWindow_s, timeWindow_e - time window in seconds from midnight (0 to 86400)
        # protocolSize - length of segments for intermittent sampling in seconds
        # protocolInterval - interval between segments for intermittent sampling in seconds
        # maxgap - max gap to join syllables in seconds
        # minlen - minimum syllable length in seconds
        # maxlen - maximum syllable length in seconds
        self.configdir = configdir
        self.configfile = os.path.join(configdir, "AviaNZconfig.txt")
        self.ConfigLoader = SupportClasses.ConfigLoader()
        self.config = self.ConfigLoader.config(self.configfile)

        self.filtersDir = os.path.join(configdir, self.config['FiltersDir'])
        self.FilterDicts = self.ConfigLoader.filters(self.filtersDir)

        self.overwrite = overwrite

        if mode=="GUI":
            self.CLI = False
            self.testmode = False
            if parent is None:
                print("ERROR: must provide a parent UI or specify CLI/testmode")
                return
            self.ui = parent
        elif mode=="CLI":
            self.CLI = True
            self.testmode = False
        elif mode=="test":
            self.CLI = False
            self.testmode = True
        elif mode=="export":
            self.CLI = False
            self.testmode=False
        else:
            print("ERROR: unrecognised mode ", mode)
            return

        self.dirName = sdir
        
        # Build options list for logging
        self.options = ["Wind: ", wind]
        if subset:
            self.options += ["Subset: ", timeWindow_s, timeWindow_e]
        else:
            self.options += ["","",""]
        if intermittent:
            self.options += ["Intermittent: ", protocolSize, protocolInterval]
        else:
            self.options += ["","",""]
        if mergeSyllables:
            self.options += ["Merge syllables: ", maxgap, minlen, maxlen]
        else:
            self.options += ["","","",""]

        print(self.options)

        if isinstance(recognisers, list):
            self.species = recognisers
        else:
            self.species = [recognisers]

        self.anySound = False
        if "Any sound" in self.species:
            self.anySound = True
            self.species.remove("Any sound")

        print(self.species)

        # Initialize detector classes
        self.bird_detector = BirdDetector(self.config, self.configdir)
        self.bat_detector = BatDetector()

        # In CLI/test modes, immediately run detection on init.
        # Otherwise GUI will ping that once it is moved to the right thread
        if self.CLI or self.testmode:
            self.detect()

    # from memory_profiler import profile
    # fp = open('memory_profiler_batch.log', 'w+')
    # @profile(stream=fp)
    def detect(self):
        # This is the function that gets things going
        # Loads the filters and gets the list of files to process

        # REQUIRES: [species], dirName, and processing options (wind, intermittent sampling, time-limited) must be set on self

        filters = [self.FilterDicts[name] for name in self.species]
        
        # Get all unique sample rates required by the filters
        samplerate = set([filt["SampleRate"] for filt in filters])
        if len(samplerate) > 1:
            print("Multiple sample rates required: ", samplerate)
            print("Audio will be resampled as needed for each filter group")

        # convert list to string
        speciesStr = " & ".join(self.species)

        # load target NN models (currently stored in the same dir as filters)
        # format: {filtername: [model, win, inputdim, output]}
        self.NNDicts = self.ConfigLoader.getNNmodels(self.FilterDicts, self.filtersDir, self.species)

        # LIST ALL FILES that will be processed (either wav or bmp, depending on mode)
        allsoundfiles = []
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                # Check if any bat species is being processed
                isBatMode = any("NZ Bats" in species for species in self.species)
                
                if not isBatMode and (filename.lower().endswith('.wav') or filename.lower().endswith('.flac')):
                    # Bird/other species: look for audio files
                    allsoundfiles.append(os.path.join(root, filename))
                elif isBatMode:
                    # Bat species: look for both audio files (.wav) and spectrogram files (.bmp)
                    if filename.lower().endswith('.wav') or filename.lower().endswith('.flac') or filename.lower().endswith('.bmp'):
                        allsoundfiles.append(os.path.join(root, filename))
        total = len(allsoundfiles)

        # LOG FILE is read here
        # note: important to log all analysis options here
        self.filesDone = []
        if not self.testmode:
            # TODO: Check
            self.log = SupportClasses.Log(os.path.join(self.dirName, 'LastAnalysisLog.txt'), speciesStr,self.options)

            # Ask for RESUME CONFIRMATION here
            if self.log.possibleAppend:
                filesExistAndDone = self.log.getDoneFiles(allsoundfiles)
                text = "Previous analysis found in this folder (analysed " + str(len(filesExistAndDone)) + " out of " + str(total) + " files in this folder).\nWould you like to resume that analysis?"
                if not self.CLI:
                    # this is super noodly but it assumes that self.CLI always means
                    # that this class was extended with the Qt-specific things.
                    self.mutex.lock()
                    self.need_msg.emit("Resume previous batch analysis?", text)
                    self.ui.msgClosed.wait(self.mutex)
                    self.mutex.unlock()
                    confirmedResume = self.ui.msg_response

                    if confirmedResume==0:
                        self.filesDone = filesExistAndDone
                    elif confirmedResume==1:
                        self.filesDone = []
                    else:  # (cancel/Esc)
                        print("Analysis cancelled")
                        raise GentleExitException
                else:
                    confirmedResume = input(text)
                    if confirmedResume.lower() == 'yes' or confirmedResume.lower() == 'y':
                        # ignore files in log
                        self.filesDone = filesExistAndDone
                    else:
                        # process all files
                        self.filesDone = []
                #if len(filesExistAndDone) == total:
                    # TODO: might want to redo?
                    #print("All files appear to have previous analysis results")
                    #return
            else:
                # work on all files
                self.filesDone = []

            # Ask for FINAL USER CONFIRMATION here
            cnt = len(self.filesDone)
            opts = ','.join(map(str, self.options))
            text = "Species: " + speciesStr + ", options: " + opts + ".\nNumber of files to analyse: " + str(total) + ", " + str(cnt) + " done so far.\n"

            text += "Log file stored in " + self.dirName + "/LastAnalysisLog.txt.\n"

            if self.overwrite:
                text += "\nWarning: any previous annotations for the selected species in these files will be deleted!\n"

            text = "Analysis will be launched with these options:\n" + text + "\nConfirm?"

            if not self.CLI:
                self.mutex.lock()
                self.need_msg.emit("Launch batch analysis",text)
                self.ui.msgClosed.wait(self.mutex)
                self.mutex.unlock()
                confirmedLaunch = self.ui.msg_response==0
            else:
                confirmedLaunch = input(text+"[y/n]")
                #print(confirmedLaunch.lower())
                if confirmedLaunch.lower() == 'yes' or confirmedLaunch.lower() == 'y':
                    confirmedLaunch = True
                else:
                    confirmedLaunch = False

            if not confirmedLaunch:
                print("Analysis cancelled")
                raise GentleExitException

            # update log: delete everything (by opening in overwrite mode),
            # reprint old headers,
            # print current header (or old if resuming),
            # print old file list if resuming.
            self.log.file = open(self.log.filepath, 'w')
            #if speciesStr not in ["Any sound", "Intermittent sampling"]:
                #self.log.reprintOld()
                # else single-sp runs should be deleted anyway

            self.log.appendHeader(header=None, species=self.log.species, settings=self.log.settings)

        if not self.CLI and not self.testmode:
            # clean up the UI before entering the long loop
            # and wait to confirm that all the dialogs are in place
            self.mutex.lock()
            self.need_clean_UI.emit(total, cnt)
            self.ui.msgClosed.wait(self.mutex)
            self.mutex.unlock()

            import pyqtgraph as pg
            with pg.BusyCursor():
                self.mainloop(allsoundfiles,total,speciesStr,filters)
        else:
            self.mainloop(allsoundfiles,total,speciesStr,filters)

        if not self.testmode:
            # delete old results (xlsx)
            # ! WARNING: any Detection...xlsx files will be DELETED,
            # ! ANYWHERE INSIDE the specified dir, recursively
            # NOTE: We currently do not export any excels automatically in this mode,
            # the user must do that manually (through Batch Review).

            print("Removing old Excel files...")
            if not self.CLI:
                self.need_update.emit(total,"Removing old Excel files, almost done...")

            for root, dirs, files in os.walk(str(self.dirName)):
                for filename in files:
                    filenamef = os.path.join(root, filename)
                    if fnmatch.fnmatch(filenamef, '*DetectionSummary_*.xlsx'):
                        print("Removing Excel file %s" % filenamef)
                        os.remove(filenamef)

            # At the end, if processing bats, export BatSearch xml automatically and check if want to export DOC database (in CLI mode, do it automatically, with missing data!)
            if any("NZ Bats" in species for species in self.species):
                # TODO: Check if detected any
                try:
                    self.exportBatResults(self.dirName, format='xml', threshold1=100, threshold2=None)
                    self.exportBatResults(self.dirName, format='passes')
                    self.exportToDOCDB()
                except Exception as e:
                    print(f"Warning: Error during bat export: {e}")
            # END of processing and exporting. Final cleanup
            self.log.file.close()

        print("Processed all %d files" % total)
        return(0)

    def mainloop(self,allsoundfiles,total,speciesStr,filters):
        processingTime = 0
        cleanexit = 0
        cnt = 0

        timeWindow_s = self.options[3]
        timeWindow_e = self.options[4]

        for filename in allsoundfiles:
            # get remaining run time in min
            processingTimeStart = time.time()
            hh,mm = divmod(processingTime * (total-cnt) / 60, 60)
            cnt = cnt+1
            progrtext = "file %d / %d. Time remaining: %d h %.2f min" % (cnt, total, hh, mm)

            print("*** Processing" + progrtext + " ***")

            # if it was processed previously (stored in log)
            if filename in self.filesDone:
                print("File %s processed previously, skipping" % filename)
                if not self.testmode:
                    self.log.appendFile(filename)
                continue

            # check if file not empty
            if os.stat(filename).st_size < 1000:
                print("File %s empty, skipping" % filename)
                if not self.testmode:
                    self.log.appendFile(filename)
                continue

            # check if file is formatted correctly
            # Check if any bat species is being processed
            isBatMode = any("NZ Bats" in species for species in self.species)
            
            with open(filename, 'br') as f:
                first2char = f.read(2)
                f.seek(0)
                first4char = f.read(4)
                
                # For bat mode: accept both BMP files (b'BM') and audio files (b'RIFF' or b'fLaC')
                # For bird mode: only accept audio files
                isValidFormat = False
                
                if isBatMode:
                    # Bat mode: accept BMP or audio files
                    if first2char == b'BM' or first4char == b'RIFF' or first4char == b'fLaC':
                        isValidFormat = True
                else:
                    # Bird mode: only audio files
                    if first4char == b'RIFF' or first4char == b'fLaC':
                        isValidFormat = True
                
                if not isValidFormat:
                    print("File %s not formatted correctly, skipping" % filename)
                    if not self.testmode:
                        self.log.appendFile(filename)
                    continue
            
            # check if there is a .corrections file and remove it
            if os.path.exists(filename + ".corrections"):
                print("Removing old corrections file")
                os.remove(filename + ".corrections")

            # test the selected time window if it is a DOC recording
            DOCRecording = re.search(r'(\d{6})_(\d{6})', os.path.basename(filename))
            if DOCRecording:
                startTime = DOCRecording.group(2)
                sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
                if timeWindow_s == timeWindow_e:
                    # (no time window set)
                    inWindow = True
                elif timeWindow_s < timeWindow_e:
                    # for day times ("8 to 17")
                    inWindow = (sTime >= timeWindow_s and sTime <= timeWindow_e)
                else:
                    # for times that include midnight ("17 to 8")
                    inWindow = (sTime >= timeWindow_s or sTime <= timeWindow_e)
            else:
                inWindow = True

            if DOCRecording and not inWindow:
                print("Skipping out-of-time-window recording %s" % filename)
                if not self.testmode:
                    self.log.appendFile(filename)
                continue

            # ALL SYSTEMS GO: process this file
            print("Loading file...")
            self.currentFilename = filename  # Track current file for bat processing
            self.segments = Segment.SegmentList()
            self.filesuccess = True  # Initialize file processing status
            
            # Check if any bat species is being processed
            isBatMode = any("NZ Bats" in species for species in self.species)
            
            try:
                self.loadFile(filename, isBatMode)
            except Exception as e:
                print(f"ERROR: Failed to load file {filename}: {e}")
                self.filesuccess = False
                continue
                
            if self.overwrite:
                print("Clearing old segments")
                self.segments = Segment.SegmentList()
            print('Segments in this file: ', self.segments)
            startCount = len(self.segments)

            if self.testmode:
                self.segments_nonn = Segment.SegmentList()
            if self.options[5] == "Intermittent: ":
                try:
                    self.addRegularSegments(filename,self.options[6],self.options[7])
                except Exception:
                    estr = "Encountered error:\n" + traceback.format_exc()
                    print("ERROR: ", estr)
                    self.log.file.close()
                    raise
            else:
                try:
                    print("Segmenting...")
                    self.detectFile(speciesStr, filters)
                except GentleExitException:
                    raise
                except Exception:
                    estr = "Encountered error:\n" + traceback.format_exc()
                    print("ERROR: ", estr)
                    if hasattr(self, 'log'):
                        self.log.file.close()
                    raise

            # export segments
            print("%d new segments marked" % (len(self.segments)-startCount))
            if self.testmode:
                # save separately With and without NN
                cleanexit = self.saveAnnotation(filename,self.segments, suffix=".tmpdata")
                cleanexit = self.saveAnnotation(filename,self.segments_nonn, suffix=".tmp2data")
            else:
                cleanexit = self.saveAnnotation(filename,self.segments)
            if cleanexit != 1:
                print("Warning: could not save segments!")

            # Log success for this file and update ProgrDlg
            if not self.testmode:
                self.log.appendFile(filename)
                if not self.CLI:
                    self.need_update.emit(cnt,"Analysed "+progrtext)
                    # TODO sprinkle more of these checks
                    if self.ui.dlg.wasCanceled():
                        print("Analysis cancelled")
                        self.log.file.close()
                        raise GentleExitException
            # track how long it took to process one file:
            processingTime = time.time() - processingTimeStart
            print("File processed in", processingTime)
            # END of audio batch processing

    def addRegularSegments(self,filename,length,interval):
        """ Perform the Hartley bodge: add fixed length segments at specified interval. """
        info = sf.info(filename)
        samplerate = info.samplerate
        nseconds = info.frames / samplerate
        self.segments.metadata = dict()
        self.segments.metadata["Operator"] = "Auto"
        self.segments.metadata["Reviewer"] = ""
        self.segments.metadata["Duration"] = nseconds
        i = 0
        segments = []
        print("Adding segments (%d s every %d s) to %s" %(length,interval, str(filename)))
        while i < nseconds:
            end_time = min(i + length, nseconds)
            segments.append([i, end_time])
            i += interval
        post = Segment.PostProcess(configdir=self.configdir, audioData=None, sampleRate=0, segments=segments, subfilter={}, cert=0)
        self.makeSegments(self.segments, post.segments)

    def detectFile(self, speciesStr, filters):
        """ Actual worker for a file in the detection loop.
        Args:
            speciesStr: String representation of species being detected
            filters: List of filter dictionaries, each with "SampleRate" key
            
        Does not return anything - results stored in self.segments
        Use with external try/catch for error handling
        """
        # Setup UI callback for cancellation checking
        ui_callback = None
        if not self.CLI:
            ui_callback = lambda: self.ui.dlg.wasCanceled()
        
        # Check if this is bat processing
        if any('NZ Bats' in species for species in self.species):
            # Use bat detector
            self.bat_detector.detectBatsInFile(
                sp=self.sp,
                segments=self.segments,
                currentFilename=self.currentFilename,
                filters=filters,
                NNDicts=self.NNDicts,
                testmode=self.testmode,
                ui_callback=ui_callback
            )
        else:
            # Use bird detector
            self.bird_detector.detectBirdsInFile(
                sp=self.sp,
                segments=self.segments,
                species=self.species,
                filters=filters,
                NNDicts=self.NNDicts,
                options=self.options,
                anySound=self.anySound,
                testmode=self.testmode,
                segments_nonn=getattr(self, 'segments_nonn', None),
                ui_callback=ui_callback
            )

    def makeSegments(self, segmentsList, segmentsNew, filtName=None, species=None, subfilter=None):
        """ Adds segmentsNew to segmentsList """
        if species == "NZ Bats":
            # Batmode: segmentsNew should be already prepared as: [x1, x2, labels]
            y1 = 0
            y2 = 0
            if len(segmentsNew)!=3:
                print("Warning: segment format does not match bat mode")
            segment = Segment.Segment([segmentsNew[0], segmentsNew[1], y1, y2, segmentsNew[2]])
            segmentsList.addSegment(segment)
        elif subfilter is not None:
            # for wavelet segments: (same as self.species!="Any sound")
            y1 = subfilter["FreqRange"][0]
            y2 = min(subfilter["FreqRange"][1], self.sp.audio_data.sample_rate//2)
            for s in segmentsNew:
                segment = Segment.Segment([s[0][0], s[0][1], y1, y2, [{"species": species, "certainty": s[1], "filter": filtName, "calltype": subfilter["calltype"]}]])
                segmentsList.addSegment(segment)
        else:
            # for generic all-species segments:
            y1 = 0
            y2 = 0
            species = "Don't Know"
            cert = 0.0
            segmentsList.addBasicSegments(segmentsNew, [y1, y2], species=species, certainty=cert)

    def saveAnnotation(self, filename, segmentList, suffix=".data"):
        """ Generates default batch-mode metadata,
            and saves the segmentList to a .data file.
            suffix arg can be used to export .tmpdata during testing.
        """
        if not hasattr(segmentList, "metadata"):
            segmentList.metadata = dict()
        segmentList.metadata["Operator"] = "Auto"
        segmentList.metadata["Reviewer"] = ""
        
        # Calculate duration based on file type
        if hasattr(self.sp, 'sg') and self.sp.sg is not None and (self.sp.audio_data is None or self.sp.audio_data.data is None):
            # BMP file - calculate duration from spectrogram dimensions
            duration = self.sp.sg.shape[1] * 0.002909090909090909
        else:
            # Audio file - calculate from audio data
            duration = float(len(self.sp.audio_data.data))/self.sp.audio_data.sample_rate
        
        segmentList.metadata["Duration"] = duration
        segmentList.metadata["noiseLevel"] = None
        segmentList.metadata["noiseTypes"] = []

        segmentList.saveJSON(str(filename) + suffix)
        return 1

    def loadFile(self, filename, bats=False, anysound=False, impMask=False):
        """ Load audio file and prepare for processing.
        
        Args:
            filename: Path to audio file
            bats: True if processing bat spectrograms (BMP files)
            anysound: True if using "Any sound" generic detection
            impMask: True to apply impulse masking (experimental)
            
        Note: Resampling is handled per-filter-group in detectFile() for efficiency.
        """
        # Create an instance of the Spectrogram class
        if not hasattr(self, 'sp'):
            print("LOADING SP 2")
            self.sp = Spectrogram.Spectrogram(self.config['window_width'], self.config['incr'])

        # Read audiodata or spectrogram
        if bats:
            self.sp.readBmp(filename, rotate=False)
        else:
            self.sp.readSoundFile(filename)

        print("Read %d samples, %f s at %d Hz" % (len(self.sp.audio_data.data), float(len(self.sp.audio_data.data))/self.sp.audio_data.sample_rate, self.sp.audio_data.sample_rate))

        # Read in stored segments (useful when doing multi-species)
        self.segments = Segment.SegmentList()
        
        # Calculate duration based on file type
        if bats:
            # For BMP files, calculate duration from spectrogram dimensions
            # Assuming standard BMP parameters: dt = 0.002909090909090909 s per column
            duration = self.sp.sg.shape[1] * 0.002909090909090909
        else:
            # For audio files, calculate from audio data
            duration = float(len(self.sp.audio_data.data))/self.sp.audio_data.sample_rate
        
        if bats or anysound or not os.path.isfile(filename + '.data'):
            # Initialize default metadata values
            self.segments.metadata = dict()
            self.segments.metadata["Operator"] = "Auto"
            self.segments.metadata["Reviewer"] = ""
            self.segments.metadata["Duration"] = duration
            # wipe all segments:
            print("Wiping all previous segments")
            self.segments.clear()
        else:
            hasmetadata = self.segments.parseJSON(filename+'.data', duration)
            if not hasmetadata:
                    # TODO: Should save this...
                    self.segments.metadata["Operator"] = "Auto"
                    self.segments.metadata["Reviewer"] = ""
            # wipe same species:
            for spec in self.species:
                # shorthand for double-checking that it's not "Any Sound" etc
                if spec in self.FilterDicts:
                    spname = self.FilterDicts[spec]["species"]
                    print("Wiping species", spname)
                    oldsegs = self.segments.getSpecies(spname)
                    for i in reversed(oldsegs):
                        wipeAll = self.segments[i].wipeSpecies(spname)
                        if wipeAll:
                            del self.segments[i]
            print("%d segments loaded from .data file" % len(self.segments))

        # impulse masking (off by default)
        # TODO
        #if impMask:
            #if anysound:
                #self.sp.data = SignalProc.impMask(self.sp.data, self.sp.sampleRate, engp=70, fp=0.50)
            #else:
                #self.sp.data = SignalProc.impMask(self.sp.data, self.sp.sampleRate) 
            #self.audiodata = self.sp.data
    
    def exportBatResults(self, dirName, format='xml', savefile=None, threshold1=0.85, threshold2=0.7):
        """
        Unified bat export method supporting multiple output formats.
        
        Args:
            dirName: Directory containing bat detection results
            format: Output format - 'xml' (BatSearch), 'csv' (BatSearch CSV), or 'passes' (bat passes)
            savefile: Output filename (defaults based on format if None)
            threshold1: Primary certainty threshold
            threshold2: Secondary certainty threshold (can be None)
            
        Returns:
            1 on success, 0 on failure
        """
        if not os.path.isdir(dirName):
            print("Folder doesn't exist")
            return 0
        
        # Set default savefile based on format
        if savefile is None:
            if format == 'xml':
                savefile = 'BatData.xml'
            elif format == 'csv':
                savefile = 'BatResults.csv'
            elif format == 'passes':
                savefile = 'BatPasses.csv'
            else:
                print(f"Unknown format: {format}")
                return 0
        
        # Metadata
        operator = "AviaNZ 3.4"
        site = "Nowhere"
        namedict = {"Unassigned":0, "Non-bat":1, "Unknown":2, "Long Tail":3, "Short Tail":4, 
                    "Possible LT":5, "Possible ST":6, "Both":7}
        
        if format == 'xml':
            return self.exportBatXML(dirName, savefile, threshold1, threshold2, operator, site, namedict)
        elif format == 'csv':
            return self.exportBatCSV(dirName, savefile, threshold1, threshold2, operator)
        elif format == 'passes':
            return self.exportBatPasses(dirName, savefile)
        else:
            print(f"Unknown format: {format}")
            return 0
    
    def exportBatXML(self, dirName, savefile, threshold1, threshold2, operator, site, namedict):
        """Export bat results to BatSearch XML format."""
        from lxml import etree
        
        for root, dirs, files in os.walk(dirName, topdown=True):
            if any(fnmatch.fnmatch(filename, '*.bmp') for filename in files):
                # Set up the XML structure
                start = etree.Element("ArrayOfBatRecording", 
                                     nsmap={'xsi': "http://www.w3.org/2001/XMLSchema-instance", 
                                           'xsd': "http://www.w3.org/2001/XMLSchema"})
                
                for filename in files:
                    if filename.endswith('.data'):
                        s1 = etree.SubElement(start, "BatRecording")
                        segments = Segment.SegmentList()
                        segments.parseJSON(os.path.join(root, filename))
                        
                        # Determine label from segments
                        label = self.getBatLabel(segments, threshold1, threshold2)
                        
                        # Create XML elements
                        etree.SubElement(s1, "AssignedBatCategory").text = str(namedict[label])
                        etree.SubElement(s1, "AssignedSite").text = site
                        etree.SubElement(s1, "AssignedUser").text = operator
                        etree.SubElement(s1, "RecTime").text = self.parseTimeDate(filename)
                        etree.SubElement(s1, "RecordingFileName").text = filename[:-5]
                        etree.SubElement(s1, "RecordingFolderName").text = ".\\" + os.path.split(root)[-1]
                        etree.SubElement(s1, "MeasureTimeFrom").text = str(0)
                
                # Write XML file
                print("writing to", os.path.join(root, savefile))
                with open(os.path.join(root, savefile), "wb") as f:
                    f.write(etree.tostring(etree.ElementTree(start), pretty_print=True, 
                                         xml_declaration=True, encoding='utf-8'))
        return 1
    
    def exportBatCSV(self, dirName, savefile, threshold1, threshold2, operator):
        """Export bat results to BatSearch CSV format."""
        f = open(os.path.join(dirName, savefile), 'w')
        f.write('Date,Time,AssignedSite,Category,Foldername,Filename,Observer\n')
        
        for root, dirs, files in os.walk(dirName):
            dirs.sort()
            files.sort()
            for filename in files:
                if filename.endswith('.data'):
                    segments = Segment.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
                    
                    label = self.getBatLabel(segments, threshold1, threshold2)
                    if label == 'Non-bat':
                        label = ''
                    
                    # Parse date and time (assumes DOC format)
                    d = filename[6:8] + '/' + filename[4:6] + '/' + filename[:4] + ','
                    if d[0] == '0':
                        d = d[1:]
                    
                    if int(filename[9:11]) < 13:
                        if filename[9:11] == '00':
                            t = str(int(filename[9:11]) + 12) + ':' + filename[11:13] + ':' + filename[13:15] + ' a.m.,'
                        else:
                            t = filename[9:11] + ':' + filename[11:13] + ':' + filename[13:15] + ' a.m.,'
                    else:
                        t = str(int(filename[9:11]) - 12) + ':' + filename[11:13] + ':' + filename[13:15] + ' p.m.,'
                    if t[0] == '0':
                        t = t[1:]
                    
                    rec = root.split('/')[-3] if label != '' else ''
                    date = '.\\' + root.split('/')[-1]
                    op = operator if label != '' else ''
                    
                    f.write(d + t + ',' + label + ',' + date + ',' + filename[:-5] + ',' + op + '\n')
        
        f.close()
        return 1
    
    def exportBatPasses(self, dirName, savefile):
        """Export bat passes summary."""
        if not hasattr(self, 'sp'):
            self.sp = Spectrogram.Spectrogram(self.config['window_width'], self.config['incr'])
        
        f = open(os.path.join(dirName, savefile), 'w')
        f.write("Tally,Night,Site,Detector,Detector Name,Bat species (L or S), Time of bat pass (24 hour clock e.g. 23:41:11),Length of bat pass (s),Feeding buzz present (yes/no)\n")
        
        dt = 0.002909090909090909
        tally = 0
        
        for root, dirs, files in os.walk(dirName, topdown=True):
            for filename in files:
                if filename.endswith('.data'):
                    segments = Segment.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
                    
                    label = 'Non-bat'
                    length = "0"
                    
                    if len(segments) > 0:
                        fn = filename[:-5]
                        self.sp.readBmp(os.path.join(root, fn), rotate=False, silent=True)
                        res = self.bat_detector.clickSearch(self.sp, None, virginia=False)
                        if res is not None:
                            length = "{:.2f}".format((res[1] - res[0]) * dt)
                        
                        seg = segments[0]
                        c = [lab["certainty"] for lab in seg[4]]
                        s = [lab["species"] for lab in seg[4]]
                        
                        if len(c) > 1:
                            label = 'Both'
                        elif c[0] > 50:
                            if s[0] == 'Long-tailed bat':
                                label = 'L'
                            elif s[0] == 'Short-tailed bat':
                                label = 'S'
                    
                    # Parse date/time from directory and filename
                    night = root[-2:] + "/" + root[-4:-2] + "/" + root[-6:-4]
                    folder = root.split("/")[-2]
                    detname = ""
                    time = filename[9:11] + ":" + filename[11:13] + ":" + filename[13:15]
                    
                    f.write(f"{tally},{night},,,{detname},{label},{time},{length},\n")
                    tally += 1
        
        f.close()
        return 1
    
    def getBatLabel(self, segments, threshold1, threshold2):
        """Helper method to determine bat label from segments."""
        if len(segments) == 0:
            return 'Non-bat'
        
        seg = segments[0]
        c = [lab["certainty"] for lab in seg[4]]
        s = [lab["species"] for lab in seg[4]]
        
        if len(c) > 1:
            return 'Both'
        
        if c[0] >= threshold1:
            if s[0] == 'Long-tailed bat':
                return 'Long Tail'
            elif s[0] == 'Short-tailed bat':
                return 'Short Tail'
        elif threshold2 is not None and c[0] > threshold2:
            if s[0] == 'Long-tailed bat':
                return 'Possible LT'
            elif s[0] == 'Short-tailed bat':
                return 'Possible ST'
        
        return 'Non-bat'
    
    def parseTimeDate(self, filename):
        """Helper method to parse time/date from filename for BatSearch format."""
        if len(filename.split('_')[0]) == 6:
            # ddmmyy format
            return "20" + filename[4:6] + "-" + filename[2:4] + "-" + filename[0:2] + "T" + \
                   filename[7:9] + ":" + filename[9:11] + ":" + filename[11:13]
        elif len(filename.split('_')[0]) == 8:
            # yyyymmdd format
            return filename[:4] + "-" + filename[4:6] + "-" + filename[6:8] + "T" + \
                   filename[9:11] + ":" + filename[11:13] + ":" + filename[13:15]
        else:
            print("Error: time unknown")
            return ""

    def exportBatSurvey(self,dirName,responses,threshold1=0.85):
        # Export an excel file for the Bat survey database
        # TODO: turn into full excel?
        if responses is None:
            responses = ['',self.config['operator'],'','ABM','','','','','']

        dates = []
        for root, dirs, files in os.walk(dirName):
            # Read the dates
            for d in dirs:
                if d.isdigit():
                    dates.append(d)

        if len(dates)==0:
            print("ERROR: no suitable folders found")
            return 0
        else:
            print("Dates:", dates)

        dates = np.array(dates)
        dates = np.unique(dates)
        dates = np.sort(dates)

        # skip unparseable strings
        dates_formatted = []
        for d in dates:
            try:
                d_f = dt.datetime.strptime(d, '%Y%m%d').date()
                dates_formatted.append(d_f)
            except ValueError:
                print("Warning: directory %s does not look like a date" % d)

        if len(dates_formatted)==0:
            print("ERROR: none of the directory names were date-like")
            return 0

        # get first, last, and total number of nights present in the data
        start = dates_formatted[0]
        end = dates_formatted[-1]
        totalnights = len(dates_formatted)

        # LT then ST
        species = np.zeros(2,dtype=int)

        for root, dirs, files in os.walk(dirName,topdown=True):
            for filename in files:
                if filename.endswith('.data'):
                    segments = Segment.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
                    if len(segments)>0:
                        seg = segments[0]
                        c = [lab["certainty"] for lab in seg[4]]
                        s = [lab["species"] for lab in seg[4]]
                        if len(c)>1:
                            species[0] += 1
                            species[1] += 1
                        else:
                            # ignoring possibles, since there should be some definites if it is real.
                            if c[0]>threshold1:
                                if s[0] == 'Long-tailed bat':
                                    species[0] += 1
                                elif s[0] == 'Short-tailed bat':
                                    species[1] += 1

        f = open(os.path.join(dirName,'BatDB.csv'),'w')

        f.write('Data Source,Observer,Survey method,Species,Passes,Date,Detector type,Date recorder put out,Date recorder collected,No. of nights out,Effective nights out,Notes,Eastings,Northings,Site name,Region\n')

        # TODO: Get effective days (how?) I think it is temperature > 7 degrees
        line = responses[0]+','+responses[1]+','+responses[2]+','
        if species[0] > 0 and species[1] > 0:
            line = line + 'Both species detected'+','+str(species[0]+species[1])+','
        elif species[0] > 0:
            line = line + 'Chalinolobus tuberculatus'+','+str(species[0])+','
        elif species[1] > 0:
            line = line + 'Mystacina tuberculata'+','+str(species[1])+','
        else:
            line = line + 'No bat species detected'+','+'0'+','
        line = line + str(start)+','+responses[3]+','+str(start)+','+str(end)+','+str(totalnights)+','+str(totalnights)+','+responses[4]+','+responses[5]+','+responses[6]+','+responses[7]+','+responses[8]+'\n'
        f.write(line)
        f.close()

    def exportToDOCDB(self):
        if not self.CLI:
            # TODO: what if you start from a different folder?
            # I think that this is OK, but need to check -- it should (?) put a BatDB file in each folder, just like the log files.
            # Then it's up to the user to sort them. Or maybe not?
            # TODO: autofill some metadata if user has filled it in once?
            easting = ""
            northing = ""
            try:
                f = open(os.path.join(self.dirName,'log.txt'),'r')
                # Find a line that contains GPS (lat, long),
                # And read the two numbers after it
                # This version just returns the last ones
                for line in f.readlines():
                    if 'GPS (lat,long)' in line:
                        ll = line.strip()
                        y = ll.split(",")
                        x = ll[-2].split(":")
                        easting = x[-1]
                        northing = y[-1]
                    elif 'GPS:' in line:
                        ll = line.strip()
                        y = ll.split("=")
                        x = y[-2].split(",")
                        easting = x[-2]
                        northing = y[-1]
            except FileNotFoundError:
                pass
            except Exception as e:
                print("Warning: could not read GPS data, ", e)

            recorder = os.path.split(self.dirName)[-1]

            # ping UI to show the survey form
            self.mutex.lock()
            self.need_bat_info.emit(self.config['operator'],easting,northing,recorder)
            self.ui.msgClosed.wait(self.mutex)
            self.mutex.unlock()

            # now, the form was either rejected, setting results to None, or accepted:
            if self.ui.batFormResults is not None:
                self.exportBatSurvey(self.dirName, self.ui.batFormResults)
        else:
            self.exportBatSurvey(self.dirName, None)


class GentleExitException(Exception):
    """ To allow tracking user-requested aborts, instead of using C-style returns. """
    pass


