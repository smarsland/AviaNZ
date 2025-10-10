# batch_processor.py
#
# Clean core batch processing without UI dependencies

# Version 4.0 9/10/25  
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

# This is the core processing class for batch AviaNZ interface
# Separated from UI concerns for better modularity

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2024

import gc, os, re, fnmatch
import time
import datetime as dt
import numpy as np
import soundfile as sf
import traceback
import math
import copy
from typing import Optional, Callable, List, Dict, Any

from src.core import Spectrogram
from src.core import SignalProc
from src.core import Segment
from src.core import WaveletSegment
from src.core import SupportClasses
from src.utils.exceptions import GentleExitException
from src.core.BirdDetector import BirdDetector
from src.core.BatDetector import BatDetector

class BatchProcessorCallbacks:
    """Interface for user interaction callbacks - allows different implementations for CLI vs GUI"""
    
    def ask_resume_analysis(self, message: str) -> bool:
        """Ask user if they want to resume previous analysis. Returns True to resume."""
        raise NotImplementedError
        
    def confirm_analysis_launch(self, message: str) -> bool:
        """Ask user to confirm analysis parameters. Returns True to proceed."""
        raise NotImplementedError
        
    def update_progress(self, current: int, total: int, message: str) -> None:
        """Update progress display"""
        pass
        
    def check_cancelled(self) -> bool:
        """Check if user has requested cancellation. Returns True if cancelled."""
        return False
        
    def get_bat_survey_info(self, operator: str, easting: str, northing: str, recorder: str) -> Optional[List[str]]:
        """Get bat survey information from user. Returns None if cancelled."""
        return None

class BatchProcessor:
    """Clean core batch processor without UI dependencies"""
    
    def __init__(self, configdir: str, directory: str, recognisers: List[str], 
                 callbacks: BatchProcessorCallbacks,
                 subset: bool = False, intermittent: bool = False, 
                 wind: str = "None", mergeSyllables: bool = False, 
                 overwrite: bool = True, timeWindow_s: int = 0, timeWindow_e: int = 0,
                 protocolSize: int = 15, protocolInterval: int = 300, 
                 maxgap: float = 1, minlen: float = 0.2, maxlen: float = 10):
        
        # Configuration
        self.configdir = configdir
        self.configfile = os.path.join(configdir, "AviaNZconfig.txt")
        self.ConfigLoader = SupportClasses.ConfigLoader()
        self.config = self.ConfigLoader.config(self.configfile)
        
        self.filtersDir = os.path.join(configdir, self.config['FiltersDir'])
        self.FilterDicts = self.ConfigLoader.filters(self.filtersDir)
        
        # Parameters
        self.dirName = directory
        self.overwrite = overwrite
        self.callbacks = callbacks
        
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

        # Process species list
        if isinstance(recognisers, list):
            self.species = recognisers.copy()
        else:
            self.species = [recognisers]

        self.anySound = False
        if "Any sound" in self.species:
            self.anySound = True
            self.species.remove("Any sound")

        # Initialize detector classes
        self.bird_detector = BirdDetector(self.config, self.configdir)
        self.bat_detector = BatDetector()

    def process_files(self) -> int:
        """Main processing method. Returns 0 on success, 1 on error."""
        try:
            # Load filters and validate
            filters = [self.FilterDicts[name] for name in self.species]
            
            # Get all unique sample rates required by the filters
            samplerate = set([filt["SampleRate"] for filt in filters])
            if len(samplerate) > 1:
                print("Multiple sample rates required: ", samplerate)
                print("Audio will be resampled as needed for each filter group")

            # convert list to string  
            speciesStr = " & ".join(self.species)

            # load target NN models
            self.NNDicts = self.ConfigLoader.getNNmodels(self.FilterDicts, self.filtersDir, self.species)

            # Get list of files to process
            allsoundfiles = self._get_files_to_process()
            total = len(allsoundfiles)

            # Handle log file and resume logic
            self.filesDone = []
            self.log = SupportClasses.Log(os.path.join(self.dirName, 'LastAnalysisLog.txt'), speciesStr, self.options)
            
            if self.log.possibleAppend:
                filesExistAndDone = self.log.getDoneFiles(allsoundfiles)
                message = f"Previous analysis found in this folder (analysed {len(filesExistAndDone)} out of {total} files in this folder).\nWould you like to resume that analysis?"
                
                if self.callbacks.ask_resume_analysis(message):
                    self.filesDone = filesExistAndDone
                else:
                    self.filesDone = []

            # Get final confirmation
            cnt = len(self.filesDone)
            opts = ','.join(map(str, self.options))
            message = f"Species: {speciesStr}, options: {opts}.\nNumber of files to analyse: {total}, {cnt} done so far.\n"
            message += f"Log file stored in {self.dirName}/LastAnalysisLog.txt.\n"
            
            if self.overwrite:
                message += "\nWarning: any previous annotations for the selected species in these files will be deleted!\n"
                
            message = "Analysis will be launched with these options:\n" + message + "\nConfirm?"
            
            if not self.callbacks.confirm_analysis_launch(message):
                print("Analysis cancelled")
                return 1

            # Update log file
            self.log.file = open(self.log.filepath, 'w') 
            self.log.appendHeader(header=None, species=self.log.species, settings=self.log.settings)

            # Notify about starting processing (so UI can setup progress dialog)
            self.callbacks.update_progress(cnt, total, "Preparing for analysis...")

            # Process all files
            return self._process_file_loop(allsoundfiles, total, speciesStr, filters)
            
        except Exception as e:
            print(f"Processing failed: {e}")
            return 1
        finally:
            if hasattr(self, 'log') and hasattr(self.log, 'file'):
                self.log.file.close()

    def _get_files_to_process(self) -> List[str]:
        """Get list of all files that will be processed"""
        allsoundfiles = []
        
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                isBatMode = any("NZ Bats" in species for species in self.species)
                
                if not isBatMode and (filename.lower().endswith('.wav') or filename.lower().endswith('.flac')):
                    allsoundfiles.append(os.path.join(root, filename))
                elif isBatMode:
                    if filename.lower().endswith('.bmp'):
                        allsoundfiles.append(os.path.join(root, filename))
                    
        return allsoundfiles

    def _process_file_loop(self, allsoundfiles: List[str], total: int, speciesStr: str, filters: List[Dict]) -> int:
        """Main file processing loop"""
        processingTime = 0
        cnt = 0
        
        timeWindow_s = self.options[3]
        timeWindow_e = self.options[4]

        for filename in allsoundfiles:
            # Check for cancellation
            if self.callbacks.check_cancelled():
                print("Processing cancelled by user")
                return 1
                
            processingTimeStart = time.time()
            hh, mm = divmod(processingTime * (total-cnt) / 60, 60)
            cnt = cnt + 1
            progrtext = f"file {cnt} / {total}. Time remaining: {int(hh)} h {mm:.2f} min"
            
            self.callbacks.update_progress(cnt, total, progrtext)
            print(f"*** Processing {progrtext} ***")

            # Skip if already processed
            if filename in self.filesDone:
                print(f"File {filename} processed previously, skipping")
                continue

            # Validate file
            if not self._validate_file(filename):
                continue

            # Check time window for DOC recordings
            if not self._check_time_window(filename, timeWindow_s, timeWindow_e):
                continue

            # Process this file
            try:
                success = self._process_single_file(filename, speciesStr, filters)
                if success:
                    self.log.appendFile(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

            # Track processing time
            processingTime = time.time() - processingTimeStart
            print(f"File processed in {processingTime}")

        print(f"Processed all {total} files")
        return 0

    def _validate_file(self, filename: str) -> bool:
        """Validate that file exists, has content, and is properly formatted"""
        # Check file size
        if os.stat(filename).st_size < 1000:
            print(f"File {filename} empty, skipping")
            return False

        # Check file format
        isBatMode = any("NZ Bats" in species for species in self.species)
        
        with open(filename, 'br') as f:
            first2char = f.read(2)
            f.seek(0)
            first4char = f.read(4)
            
            isValidFormat = False
            
            if isBatMode:
                isValidFormat = (first2char == b'BM') or (first4char == b'RIFF') or (first4char == b'fLaC')
            else:
                isValidFormat = (first4char == b'RIFF') or (first4char == b'fLaC')
            
            if not isValidFormat:
                print(f"File {filename} is not a valid audio/BMP file, skipping")
                return False

        return True

    def _check_time_window(self, filename: str, timeWindow_s: int, timeWindow_e: int) -> bool:
        """Check if DOC recording falls within specified time window"""
        DOCRecording = re.search(r'(\d{6})_(\d{6})', os.path.basename(filename))
        if not DOCRecording:
            return True
            
        startTime = DOCRecording.group(2)
        sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
        
        if timeWindow_s == timeWindow_e:
            inWindow = True
        elif timeWindow_s < timeWindow_e:
            inWindow = timeWindow_s <= sTime <= timeWindow_e
        else:
            inWindow = sTime >= timeWindow_s or sTime <= timeWindow_e
            
        if not inWindow:
            print(f"Skipping out-of-time-window recording {filename}")
            
        return inWindow

    def _process_single_file(self, filename: str, speciesStr: str, filters: List[Dict]) -> bool:
        """Process a single file. Returns True on success."""
        print("Loading file...")
        self.currentFilename = filename
        self.segments = Segment.SegmentList()
        
        # Check if any bat species is being processed
        isBatMode = any("NZ Bats" in species for species in self.species)
        
        try:
            self.loadFile(filename, isBatMode)
        except Exception as e:
            print(f"ERROR: Failed to load file {filename}: {e}")
            return False
            
        if self.overwrite:
            print("Clearing old segments")
            self.segments = Segment.SegmentList()
            
        print('Segments in this file: ', self.segments)
        startCount = len(self.segments)

        # Perform detection
        try:
            if self.options[5] == "Intermittent: ":
                self.addRegularSegments(filename, self.options[6], self.options[7])
            else:
                self.detectFile(speciesStr, filters)
        except Exception as e:
            print(f"Detection failed for {filename}: {e}")
            return False

        # Save results
        print(f"{len(self.segments)-startCount} new segments marked")
        self.saveAnnotation(filename, self.segments)
        
        return True

    # The rest of the methods from the original class can be copied here
    # (detectFile, loadFile, saveAnnotation, addRegularSegments, makeSegments, etc.)
    # but without any UI dependencies

    def detectFile(self, speciesStr, filters):
        """Actual worker for a file in the detection loop."""
        # Check if this is bat processing
        if any('NZ Bats' in species for species in self.species):
            # Use bat detector
            self.bat_detector.detectBatsInFile(
                sp=self.sp,
                segments=self.segments,
                currentFilename=self.currentFilename,
                filters=filters,
                NNDicts=self.NNDicts,
                testmode=False,
                ui_callback=self.callbacks.check_cancelled
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
                testmode=False,
                segments_nonn=None,
                ui_callback=self.callbacks.check_cancelled
            )

    def loadFile(self, filename, bats=False, anysound=False, impMask=False):
        """Load audio file and prepare for processing."""
        # Create an instance of the Spectrogram class
        if not hasattr(self, 'sp'):
            print("LOADING SP 2")
            self.sp = Spectrogram.Spectrogram(self.config['window_width'], self.config['incr'])

        # Read audiodata or spectrogram
        if bats:
            self.sp.readBmp(filename, rotate=False)
        else:
            self.sp.readSoundFile(filename)

        # Print file info - handle BMP files which don't have audio data
        if self.sp.audio_data.data is not None:
            print("Read %d samples, %f s at %d Hz" % (len(self.sp.audio_data.data), float(len(self.sp.audio_data.data))/self.sp.audio_data.sample_rate, self.sp.audio_data.sample_rate))
        else:
            # BMP file - calculate duration from spectrogram
            duration = self.sp.sg.shape[1] * 0.002909090909090909
            print("Read BMP spectrogram: %d x %d pixels, %f s at %d Hz" % (self.sp.sg.shape[0], self.sp.sg.shape[1], duration, self.sp.audio_data.sample_rate))

        # Read in stored segments (useful when doing multi-species)
        self.segments = Segment.SegmentList()
        
        # Calculate duration based on file type
        if bats:
            # For BMP files, calculate duration from spectrogram dimensions
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
            print("Wiping all previous segments")
            self.segments.clear()
        else:
            hasmetadata = self.segments.parseJSON(filename+'.data', duration)
            if not hasmetadata:
                self.segments.metadata = dict()
                self.segments.metadata["Operator"] = "Auto"
                self.segments.metadata["Reviewer"] = ""
                self.segments.metadata["Duration"] = duration
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

    def saveAnnotation(self, filename, segmentList, suffix=".data"):
        """Generates default batch-mode metadata and saves the segmentList to a .data file."""
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

    def addRegularSegments(self, filename, length, interval):
        """Perform the Hartley bodge: add fixed length segments at specified interval."""
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

    def makeSegments(self, segmentsList, segmentsNew, filtName=None, species=None, subfilter=None):
        """Adds segmentsNew to segmentsList"""
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

    def exportBatSurvey(self, dirName, responses, threshold1=0.85):
        """Export an excel file for the Bat survey database"""
        if responses is None:
            responses = ['', self.config['operator'], '', 'ABM', '', '', '', '', '']

        dates = []
        for root, dirs, files in os.walk(dirName):
            # Read the dates
            for d in dirs:
                if d.isdigit():
                    dates.append(d)

        if len(dates) == 0:
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
                import datetime as dt
                d_f = dt.datetime.strptime(d, '%Y%m%d').date()
                dates_formatted.append(d_f)
            except ValueError:
                print("Warning: directory %s does not look like a date" % d)

        if len(dates_formatted) == 0:
            print("ERROR: none of the directory names were date-like")
            return 0

        # get first, last, and total number of nights present in the data
        start = dates_formatted[0]
        end = dates_formatted[-1]
        totalnights = len(dates_formatted)

        # LT then ST
        species = np.zeros(2, dtype=int)

        for root, dirs, files in os.walk(dirName, topdown=True):
            for filename in files:
                if filename.endswith('.data'):
                    segments = Segment.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
                    label = self.getBatLabel(segments, threshold1, None)
                    if label == 'Long Tail' or label == 'Possible LT':
                        species[0] += 1
                    elif label == 'Short Tail' or label == 'Possible ST':
                        species[1] += 1
                    elif label == 'Both':
                        species[0] += 1
                        species[1] += 1

        f = open(os.path.join(dirName, 'BatDB.csv'), 'w')
        f.write('Data Source,Observer,Survey method,Species,Passes,Date,Detector type,Date recorder put out,Date recorder collected,No. of nights out,Effective nights out,Notes,Eastings,Northings,Site name,Region\n')

        # TODO: Get effective days (how?) I think it is temperature > 7 degrees
        line = responses[0] + ',' + responses[1] + ',' + responses[2] + ','
        if species[0] > 0 and species[1] > 0:
            line = line + 'Both species detected' + ',' + str(species[0] + species[1]) + ','
        elif species[0] > 0:
            line = line + 'Chalinolobus tuberculatus' + ',' + str(species[0]) + ','
        elif species[1] > 0:
            line = line + 'Mystacina tuberculata' + ',' + str(species[1]) + ','
        else:
            line = line + 'No bat species detected' + ',' + '0' + ','
        line = line + str(start) + ',' + responses[3] + ',' + str(start) + ',' + str(end) + ',' + str(totalnights) + ',' + str(totalnights) + ',' + responses[4] + ',' + responses[5] + ',' + responses[6] + ',' + responses[7] + ',' + responses[8] + '\n'
        f.write(line)
        f.close()
        
        return 1

    def exportToDOCDB(self):
        """Export to DOC database - requires user interaction for survey info"""
        # This method needs callback for getting survey info from user
        survey_info = self.callbacks.get_bat_survey_info(
            self.config['operator'], "", "", os.path.split(self.dirName)[-1]
        )
        
        if survey_info is not None:
            return self.exportBatSurvey(self.dirName, survey_info)
        else:
            print("Bat survey export cancelled")
            return 0