
# Version 4.0 09/10/25
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

# BatchProcessor.py
#
# Core batch processing without UI dependencies

import os, re
import time
import soundfile as sf

from src.core import Spectrogram
from src.core import Annotation
from src.core import SupportClasses
from src.core import BirdDetector
from src.core import BatDetector
from src.core import Segmentation

# Constants
SAMPLES_PER_PAGE_16KHZ = 900 * 16000
MIN_FILE_SIZE_BYTES = 1000
SAMPLES_PER_300S_PAGE = 300

class BatchProcessorCallbacks:
    """Abstract interface for user interaction callbacks - allows different implementations for CLI vs GUI"""
    
    def ask_resume_analysis(self, message):
        """Ask user if they want to resume previous analysis. Returns True to resume."""
        raise NotImplementedError
        
    def confirm_analysis_launch(self, message):
        """Ask user to confirm analysis parameters. Returns True to proceed."""
        raise NotImplementedError
        
    def update_progress(self, current, total, message):
        """Update progress display"""
        pass
        
    def check_cancelled(self):
        """Check if user has requested cancellation. Returns True if cancelled."""
        return False

class BatchProcessor:
    """Core batch processor for automated species detection across multiple audio files"""

    def __init__(self, configdir='', directory='', recognisers=None, 
                 callbacks=None,
                 subset=False, intermittent=False, 
                 wind="None", mergeSyllables=False, 
                 overwrite=None, overwriteSpecies=True, overwriteAll=False,
                 timeWindow_s=0, timeWindow_e=0,
                 protocolSize=15, protocolInterval=300, 
                 maxgap=1, minlen=0.2, maxlen=10,
                 testmode=False):
        
        # Configuration
        self.configdir = configdir
        self.configfile = os.path.join(configdir, "AviaNZconfig.txt")
        self.ConfigLoader = SupportClasses.ConfigLoader()
        self.config = self.ConfigLoader.config(self.configfile)
        
        self.filtersDir = os.path.join(configdir, self.config['FiltersDir'])
        self.FilterDicts = self.ConfigLoader.filters(self.filtersDir)
        
        # Parameters
        self.dirName = directory
        # Backward compatibility: if overwrite is specified, use it for overwriteSpecies
        if overwrite is not None:
            self.overwriteSpecies = overwrite
            self.overwriteAll = False
        else:
            self.overwriteSpecies = overwriteSpecies
            self.overwriteAll = overwriteAll
        self.callbacks = callbacks
        self.testmode = testmode
        
        # Processing options stored as dictionary with named keys
        self.options = {
            'wind': wind,
            'subset': subset,
            'timeWindow_s': timeWindow_s,
            'timeWindow_e': timeWindow_e,
            'intermittent': intermittent,
            'protocolSize': protocolSize,
            'protocolInterval': protocolInterval,
            'mergeSyllables': mergeSyllables,
            'maxgap': maxgap,
            'minlen': minlen,
            'maxlen': maxlen
        }

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
        self.bird_detector = BirdDetector.BirdDetector(self.config, self.configdir)
        self.bat_detector = BatDetector.BatDetector()

    def process_files(self):
        """Main processing method. Returns 0 on success, 1 on error."""
        filters = [self.FilterDicts[name] for name in self.species]
        
        samplerate = set([filt["SampleRate"] for filt in filters])
        if len(samplerate) > 1:
            print("Multiple sample rates required: ", samplerate)
            print("Audio will be resampled as needed for each filter group")

        speciesStr = " & ".join(self.species)

        self.NNDicts = self.ConfigLoader.getNNmodels(self.FilterDicts, self.filtersDir, self.species)

        allsoundfiles = self.get_files_to_process()
        total = len(allsoundfiles)

        self.filesDone = []
        self.log = SupportClasses.Log(os.path.join(self.dirName, 'LastAnalysisLog.txt'), speciesStr, self.options)
        
        if self.log.possibleAppend:
            filesExistAndDone = self.log.getDoneFiles(allsoundfiles)
            message = f"Previous analysis found in this folder (analysed {len(filesExistAndDone)} out of {total} files in this folder).\nWould you like to resume that analysis?"
            
            if self.callbacks.ask_resume_analysis(message):
                self.filesDone = filesExistAndDone
            else:
                self.filesDone = []

        cnt = len(self.filesDone)
        # Format options for display
        opts_parts = []
        if self.options['wind'] != "None":
            opts_parts.append(f"Wind: {self.options['wind']}")
        if self.options['subset']:
            opts_parts.append(f"Subset: {self.options['timeWindow_s']}-{self.options['timeWindow_e']}")
        if self.options['intermittent']:
            opts_parts.append(f"Intermittent: {self.options['protocolSize']}s every {self.options['protocolInterval']}s")
        if self.options['mergeSyllables']:
            opts_parts.append(f"Merge syllables: gap={self.options['maxgap']}, min={self.options['minlen']}, max={self.options['maxlen']}")
        opts = ', '.join(opts_parts) if opts_parts else 'None'
        
        message = f"Species: {speciesStr}, options: {opts}.\nNumber of files to analyse: {total}, {cnt} done so far.\n"
        message += f"Log file stored in {self.dirName}/LastAnalysisLog.txt.\n"
        
        if self.overwriteAll:
            message += "\nWarning: ALL previous annotations in these files will be deleted!\n"
        elif self.overwriteSpecies:
            message += "\nWarning: any previous annotations for the selected species in these files will be deleted!\n"
            
        message = "Analysis will be launched with these options:\n" + message + "\nConfirm?"
        
        if not self.callbacks.confirm_analysis_launch(message):
            print("Analysis cancelled")
            return 1

        self.log.file = open(self.log.filepath, 'w') 
        self.log.appendHeader(header=None, species=self.log.species, settings=self.log.settings)

        self.callbacks.update_progress(cnt, total, "Preparing for analysis...")

        return self.process_file_loop(allsoundfiles, total, filters)

    def get_files_to_process(self):
        """Get list of all files that will be processed"""
        allsoundfiles = []
        
        for root, dirs, files in os.walk(str(self.dirName)):
            for filename in files:
                isBatMode = any("NZ Bats" in species or species == "NZ Bats_NP" for species in self.species)
                
                if not isBatMode and (filename.lower().endswith('.wav') or filename.lower().endswith('.flac')):
                    allsoundfiles.append(os.path.join(root, filename))
                elif isBatMode:
                    if filename.lower().endswith('.bmp'):
                        allsoundfiles.append(os.path.join(root, filename))
                    
        return allsoundfiles

    def process_file_loop(self, allsoundfiles, total, filters):
        """Main file processing loop"""
        processingTime = 0
        cnt = 0
        
        timeWindow_s = self.options['timeWindow_s']
        timeWindow_e = self.options['timeWindow_e']

        for filename in allsoundfiles:
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
            if not self.validate_file(filename):
                continue

            # Check time window for DOC recordings
            if not self.check_time_window(filename, timeWindow_s, timeWindow_e):
                continue

            success = self.process_single_file(filename, filters)
            if success:
                self.log.appendFile(filename)

            processingTime = time.time() - processingTimeStart
            print(f"File processed in {processingTime}")

        print(f"Processed all {total} files")
        return 0

    def validate_file(self, filename):
        """Validate that file exists, has content, and is properly formatted"""
        if os.stat(filename).st_size < MIN_FILE_SIZE_BYTES:
            print(f"File {filename} empty, skipping")
            return False

        isBatMode = any("NZ Bats" in species or species == "NZ Bats_NP" for species in self.species)
        
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

    def check_time_window(self, filename, timeWindow_s, timeWindow_e):
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

    def process_single_file(self, filename, filters):
        """Process a single file. Returns True on success."""
        print("Loading file...")
        self.currentFilename = filename
        
        isBatMode = any("NZ Bats" in species or species == "NZ Bats_NP" for species in self.species)
        
        self.loadFile(filename, isBatMode)
        
        print('Segments in this file: ', self.segments)
        startCount = len(self.segments)

        # Initialize segments_nonn for testmode
        if self.testmode:
            self.segments_nonn = Annotation.SegmentList()

        if self.options['intermittent']:
            self.addRegularSegments(filename, self.options['protocolSize'], self.options['protocolInterval'])
        else:
            self.detectFile(filters)

        print(f"{len(self.segments)-startCount} new segments marked")
        
        # Save annotations
        if self.testmode:
            # Save separately with and without NN
            self.saveAnnotation(filename, self.segments, suffix=".tmpdata")
            self.saveAnnotation(filename, self.segments_nonn, suffix=".tmp2data")
        else:
            self.saveAnnotation(filename, self.segments)
        
        return True

    def detectFile(self, filters):
        """Actual worker for a file in the detection loop."""
        # Check if this is bat processing
        if any('NZ Bats' in species or species == "NZ Bats_NP" for species in self.species):
            self.bat_detector.detectBatsInFile(
                sp=self.sp,
                segments=self.segments,
                currentFilename=self.currentFilename,
                filters=filters,
                NNDicts=self.NNDicts,
                testmode=self.testmode,
                check_cancelled=self.callbacks.check_cancelled
            )
        else:
            # Pass segments_nonn only in testmode
            segments_nonn = self.segments_nonn if self.testmode else None
            
            self.bird_detector.detectBirdsInFile(
                sp=self.sp,
                segments=self.segments,
                species=self.species,
                filters=filters,
                NNDicts=self.NNDicts,
                options=self.options,
                anySound=self.anySound,
                testmode=self.testmode,
                segments_nonn=segments_nonn,
                check_cancelled=self.callbacks.check_cancelled
            )

    def loadFile(self, filename, bats=False, anysound=False, impMask=False):
        """Load audio file and prepare for processing."""
        self.sp = Spectrogram.Spectrogram(self.config['window_width'], self.config['incr'])

        if bats:
            self.sp.readSoundFile(filename, rotate=False)
        else:
            self.sp.readSoundFile(filename)

        if self.sp.audio_data.data is not None:
            print("Read %d samples, %f s at %d Hz" % (len(self.sp.audio_data.data), float(len(self.sp.audio_data.data))/self.sp.audio_data.sample_rate, self.sp.audio_data.sample_rate))
        else:
            duration = self.sp.get_duration()
            print("Read BMP spectrogram: %d x %d pixels, %f s at %d Hz" % (self.sp.sg.shape[0], self.sp.sg.shape[1], duration, self.sp.audio_data.sample_rate))

        self.segments = Annotation.SegmentList()
        
        duration = self.sp.get_duration()
        
        # If overwriteAll is set, or if we're in bat/anysound mode, or no .data file exists:
        # wipe everything
        if self.overwriteAll or bats or anysound or not os.path.isfile(filename + '.data'):
            self.segments.metadata["Operator"] = "Auto"
            self.segments.metadata["Reviewer"] = ""
            self.segments.metadata["Duration"] = duration
            print("Wiping all previous segments")
            self.segments.clear()
        else:
            # Load existing annotations
            hasmetadata = self.segments.parseJSON(filename+'.data', duration)
            if not hasmetadata:
                self.segments.metadata["Operator"] = "Auto"
                self.segments.metadata["Reviewer"] = ""
                self.segments.metadata["Duration"] = duration
            
            # If overwriteSpecies is set, remove annotations for the selected species
            if self.overwriteSpecies:
                for species in self.species:
                    if species in self.FilterDicts:
                        spname = self.FilterDicts[species]["species"]
                        print("Wiping species", spname)
                        oldsegs = self.segments.getSpecies(spname)
                        for i in reversed(oldsegs):
                            wipeAll = self.segments[i].wipeSpecies(spname)
                            if wipeAll:
                                del self.segments[i]
            # If neither overwrite option is set, keep all existing annotations
            # and just add new detections
            print("%d segments loaded from .data file" % len(self.segments))

    def saveAnnotation(self, filename, segmentList, suffix=".data"):
        """Generates default batch-mode metadata and saves the segmentList to a .data file."""
        segmentList.metadata["Operator"] = "Auto"
        segmentList.metadata["Reviewer"] = ""
        segmentList.metadata["Duration"] = self.sp.get_duration()
        segmentList.metadata["noiseLevel"] = None
        segmentList.metadata["noiseTypes"] = []
        segmentList.saveJSON(str(filename) + suffix)
        return 1

    def addRegularSegments(self, filename, length, interval):
        """Perform the Hartley bodge: add fixed length segments at specified interval."""
        info = sf.info(filename)
        samplerate = info.samplerate
        nseconds = info.frames / samplerate
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
        post = Segmentation.PostProcess(configdir=self.configdir, audioData=None, sampleRate=0, segments=segments, subfilter={}, cert=0)
        self.segments.addFromTimeRanges(post.segments, 0, 0, species="Don't Know", certainty=0.0)