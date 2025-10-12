
# Version 3.4 18/12/24
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

import numpy as np

from src.core import Spectrogram
from src.core import Segmentation
from src.core import Annotation
from src.core import WaveletSegment
from src.core import SupportClasses

import traceback
import time

import math
import copy

import soundfile as sf


# Need to work through this and ensure:
# 1. care taken for resampling for different filters
# 2. intermittent sampling and time of files is correct
# 3. simplify if possible

class AviaNZ_batchProcess():
    # Main class for batch processing
    # Contains the algorithms, not the GUI, so that it can be run from the commandline or the GUI
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
                if (not ("NZ Bats" in self.species) and (filename.lower().endswith('.wav') or filename.lower().endswith('.flac'))) or ("NZ Bats" in self.species  and filename.lower().endswith('.bmp')):
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
            if "NZ Bats" in self.species:
                # TODO: Check if detected any
                self.exportBatResults(self.dirName, format='xml', threshold1=100, threshold2=None)
                self.exportBatResults(self.dirName, format='passes')
                self.exportToDOCDB()
            # END of processing and exporting. Final cleanup
            self.log.file.close()

        print("Processed all %d files" % total)
        return(0)

    def mainloop(self,allsoundfiles,total,speciesStr,filters):
        # MAIN PROCESSING starts here
        # TODO: This will need a bit of work to deal with different filters with non-matching sample rates
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
                # skip the processing:
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
            # TODO: speciesStr or self.species?
            with open(filename, 'br') as f:
                first2char = f.read(2)
                f.seek(0)
                first4char = f.read(4)
                if ("NZ Bats" in self.species and first2char != b'BM') or (not("NZ Bats" in self.species) and first4char != b'RIFF' and first4char != b'fLaC'):
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
            self.segments = Annotation.SegmentList()
            self.loadFile(filename,"NZ Bats" in self.species)
            if self.overwrite:
                print("Clearing old segments")
                self.segments = Annotation.SegmentList()
            print('Segments in this file: ', self.segments)
            startCount = len(self.segments)

            if self.testmode:
                self.segments_nonn = Annotation.SegmentList()
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
            segments.append([i, i + length])
            i += interval
        post = Segmentation.PostProcess(configdir=self.configdir, audioData=None, sampleRate=0, segments=segments, subfilter={}, cert=0)
        self.makeSegments(self.segments, post.segments)

    def detectFile(self, speciesStr, filters):
        """ Actual worker for a file in the detection loop.
        Args:
            speciesStr: String representation of species being detected
            filters: List of filter dictionaries, each with "SampleRate" key
            
        Does not return anything - results stored in self.segments
        Use with external try/catch for error handling
        """
        # Segment over pages separately, to allow dealing with large files smoothly:
        # Page size depends on the original audio sample rate (not filter target rates)
        # - Low sample rates (<=4000 Hz, e.g., bittern): smaller pages as low freqs are noisier
        # - Standard rates: ~15 min pages (900s * 16000 Hz standard reference)
        if hasattr(self, 'sp'):
            if self.sp.audio_data.sample_rate <= 4000:
                # Low frequency recordings (e.g., bittern at 4000 Hz)
                samplesInPage = 300 * self.sp.audio_data.sample_rate  # ~5 min pages
            else:
                # Standard recordings: use 15 min pages normalized to 16 kHz
                samplesInPage = 900 * 16000
        
        elif not("NZ Bats" in self.species):
            # For bird detection without pre-loaded spectrogram
            # If using changepoint detection with very short windows,
            # we want enough windows for reliable statistics (~5000 windows)
            # Current implementation: use standard 15 min pages for all cases
            winsize = [subf["WaveletParams"].get("win", 1) for f in filters for subf in f["Filters"]]
            winsize = min(winsize)
            if winsize < 0.05:
                # Short window changepoint detection
                # Could use: int(4500 * 0.05 * sample_rate) for proportional sizing
                # But currently using standard page size for consistency
                samplesInPage = 900 * 16000
            else:
                # Standard window sizes
                samplesInPage = 900 * 16000
        else:
            # Bat processing: use standard 15 min pages
            samplesInPage = 900 * 16000

        # (ceil division for large integers)
        numPages = (len(self.sp.audio_data.data) - 1) // samplesInPage + 1

        # Actual segmentation happens here:
        for page in range(numPages):
            print("Segmenting page %d / %d" % (page+1, numPages))
            start = page*samplesInPage
            end = min(start+samplesInPage, len(self.sp.audio_data.data))
            # TODO: Still self.sp problems!
            thisPageLen = (end-start) / self.sp.audio_data.sample_rate
            #thisPageLen = (end-start) /16000 # self.sp.sampleRate

            if thisPageLen < 2 and not("NZ Bats" in self.species):
                print("Warning: can't process short file ends (%.2f s)" % thisPageLen)
                continue

            # Process
            if self.anySound:
                # Create spectrogram for median clipping etc
                if not hasattr(self, 'sp'):
                    print(self.config['window_width'], self.config['incr'])
                    print("LOADING SP 1")
                    self.sp = Spectrogram.Spectrogram(self.config['window_width'], self.config['incr'])
                _ = self.sp.spectrogram(window_width=self.config['window_width'], incr=self.config['incr'],window=self.config['windowType'],sgType=self.config['sgType'],sgScale=self.config['sgScale'],nfilters=self.config['nfilters'],mean_normalise=self.config['sgMeanNormalise'],equal_loudness=self.config['sgEqualLoudness'],onesided=self.config['sgOneSided'],start=start,stop=end)
                self.seg = Segmentation.Segmenter(self.sp, self.sp.audio_data.sample_rate)
                # thisPageSegs = self.seg.bestSegments()
                thisPageSegs = self.seg.medianClip(thr=3.5)
                # Post-process
                print("Segments detected: ", len(thisPageSegs))
                print("Post-processing...")
                post = Segmentation.PostProcess(configdir=self.configdir, audioData=self.sp.audio_data.data[start:end], sampleRate=self.sp.audio_data.sample_rate, segments=thisPageSegs, subfilter={}, cert=0)
                #post = Segmentation.PostProcess(configdir=self.configdir, audioData=self.audiodata[start:end], sampleRate=self.sp.sampleRate, segments=thisPageSegs, subfilter={}, cert=0)
                if self.options[8] != "":
                    post.joinGaps(self.options[9])
                    post.deleteShort(self.options[10])
                    # avoid extra long segments 
                    post.splitLong(self.options[11])

                # adjust segment starts for 15min "pages"
                if start != 0:
                    for seg in post.segments:
                        seg[0][0] += start/self.sp.audio_data.sample_rate
                        seg[0][1] += start/self.sp.audio_data.sample_rate
                # attach mandatory "Don't Know"s etc and put on self.segments
                self.makeSegments(self.segments, post.segments)
                del self.seg
                gc.collect()
                # After each page, check for interrupts:
                if not self.CLI:
                    if self.ui.dlg.wasCanceled():
                        print("Analysis cancelled")
                        self.log.file.close()
                        raise GentleExitException

            # Group filters by required sample rate to minimize resampling operations
            uniqueSampleRates = set([filt["SampleRate"] for filt in filters])
            
            for targetSampleRate in uniqueSampleRates:
                # Get all filters that need this sample rate
                filtersAtSampleRate = [filters[i] for i in range(len(filters)) 
                                      if filters[i]["SampleRate"] == targetSampleRate]
                speciesAtSampleRate = [self.species[i] for i in range(len(filters)) 
                                      if filters[i]["SampleRate"] == targetSampleRate]
                
                print(f"Processing sample rate {targetSampleRate} Hz for species: {speciesAtSampleRate}")
                
                # Initialize wavelet segmentation for bird processing only
                # (WaveletSegment.readBatch handles its own resampling efficiently)
                if not("NZ Bats" in speciesAtSampleRate) and len(speciesAtSampleRate) > 0:
                    self.ws = WaveletSegment.WaveletSegment(wavelet='dmey2')
                    useWind = self.options[1] in ["OLS wind filter (recommended)", "Robust wind filter (experimental, slow)"]
                    # Note: readBatch will intelligently handle resampling from self.sp.audio_data.sample_rate
                    # to targetSampleRate, including node adjustment optimizations for 2x/4x ratios
                    self.ws.readBatch(self.sp.audio_data.data[start:end], self.sp.audio_data.sample_rate, 
                                     d=False, spInfo=filtersAtSampleRate, wpmode="new", wind=useWind)
                
                for speciesix in range(len(filtersAtSampleRate)):
                    print("Working with recogniser:", filtersAtSampleRate[speciesix])
                    spInfo = filtersAtSampleRate[speciesix]
                    
                    # Check if this is bat processing
                    if "NZ Bats" in speciesAtSampleRate:
                        # === BAT PROCESSING PATH ===
                        # Get NN model for bats
                        NNmodel = None
                        if 'NN' in spInfo and spInfo['NN']['NN_name'] in self.NNDicts.keys():
                            NNmodel = self.NNDicts[spInfo['NN']['NN_name']]
                        
                        if NNmodel is None:
                            print("Warning: No NN model found for bat detection")
                            continue
                        
                        if not self.testmode:
                            # Process bat file with unified method
                            batSegments = self.processBatFile(self.currentFilename, start, end, thisPageLen, NNmodel)
                            if len(batSegments) > 0:
                                self.makeSegments(self.segments, batSegments[0])
                        else:
                            print("Warning: Bat detection not fully supported in test mode")
                            # Could add test mode support here if needed
                    else:
                        # === BIRD PROCESSING PATH ===
                        # Bird detection by wavelets. Choose the right wavelet method:
                        if "method" not in spInfo or spInfo["method"]=="wv":
                            # note: using 'recaa' mode = partial antialias
                            thisPageSegs = self.ws.waveletSegment(speciesix, wpmode="new")
                        elif spInfo["method"]=="chp":
                            # note that only allowing alg2 = nuisance-robust chp detection
                            thisPageSegs = self.ws.waveletSegmentChp(speciesix, alg=2, wind=self.options[1])
                        else:
                            print("ERROR: unrecognised method", spInfo["method"])
                            raise Exception

                        print("Segments detected (all subfilters): ", thisPageSegs)
                        if not self.testmode:
                            print("Post-processing...")
                        
                        # Process each subfilter
                        for filtix in range(len(spInfo['Filters'])):
                            NNmodel = None
                            if 'NN' in spInfo:
                                if spInfo['NN']['NN_name'] in self.NNDicts.keys():
                                    NNmodel = self.NNDicts[spInfo['NN']['NN_name']]

                            if not self.testmode:
                                # Bird-style NN and other processing:
                                postsegs = self.postProcFull(thisPageSegs, spInfo, filtix, start, end, NNmodel)
                                # attach filter info and put on self.segments:
                                self.makeSegments(self.segments, postsegs, speciesAtSampleRate[speciesix], spInfo["species"], spInfo['Filters'][filtix])

                                # After each subfilter is done, check for interrupts:
                                if not self.CLI:
                                    if self.ui.dlg.wasCanceled():
                                        print("Analysis cancelled")
                                        self.log.file.close()
                                        raise GentleExitException
                            else:
                                # THIS IS testmode for birds
                                # test without nn:
                                postsegs = self.postProcFull(copy.deepcopy(thisPageSegs), spInfo, filtix, start, end, NNmodel=None)
                                # stash these segments before any NN/postproc:
                                self.makeSegments(self.segments_nonn, postsegs, speciesAtSampleRate[speciesix], spInfo["species"], spInfo['Filters'][filtix])

                                # test with nn:
                                postsegs = self.postProcFull(copy.deepcopy(thisPageSegs), spInfo, filtix, start, end, NNmodel)
                                # attach filter info and put on self.segments:
                                self.makeSegments(self.segments, postsegs, speciesAtSampleRate[speciesix], spInfo["species"], spInfo['Filters'][filtix])

    def postProcFull(self, segments, spInfo, filtix, start, end, NNmodel):
        """ Full bird-style postprocessing (NN, joinGaps, fundamental frequency, etc.)
        
        Args:
            segments: list of segments over calltypes
            spInfo: species info dict containing filter config including target SampleRate
            filtix: index of current subfilter
            start, end: start and end of this page, in samples (at original sample rate)
            NNmodel: None or a neural network model for species classification
            
        Returns:
            list of processed segments
            
        Note: PostProcess will handle resampling if needed (sampleRate != tgtsampleRate).
        This is typically only required for NN classification or fundamental frequency detection.
        """
        subfilter = spInfo["Filters"][filtix]
        
        # PostProcess handles any needed resampling from current rate to target rate
        post = Segmentation.PostProcess(configdir=self.configdir, audioData=self.sp.audio_data.data[start:end],
                            sampleRate=self.sp.audio_data.sample_rate, 
                            tgtsampleRate=spInfo["SampleRate"],
                            segments=segments[filtix], subfilter=subfilter,
                            NNmodel=NNmodel, cert=50)
        print("Segments detected after WF: ", len(segments[filtix]))

        if NNmodel:
            print('Post-processing with NN')
            post.NN()

        # Fund freq and merging. Only do for standard wavelet filter currently:
        # (for median clipping, gap joining and some short segment cleanup was already done in WaveletSegment)
        if "method" not in spInfo or spInfo["method"]=="wv":
            if 'F0' in subfilter and 'F0Range' in subfilter and subfilter["F0"]:
                print("Checking for fundamental frequency...")
                post.fundamentalFrq()

            post.joinGaps(maxgap=subfilter['TimeRange'][3])

        # delete short segments, if requested:
        if subfilter['TimeRange'][0]>0:
            post.deleteShort(minlength=subfilter['TimeRange'][0])

        # adjust segment starts for 15min "pages"
        if start != 0:
            for seg in post.segments:
                seg[0][0] += start/self.sp.audio_data.sample_rate
                seg[0][1] += start/self.sp.audio_data.sample_rate
        print("After post-processing: ", post.segments)
        return(post.segments)

    def makeSegments(self, segmentsList, segmentsNew, filtName=None, species=None, subfilter=None):
        """ Adds segmentsNew to segmentsList """
        if species == "NZ Bats":
            # Batmode: segmentsNew should be already prepared as: [x1, x2, labels]
            y1 = 0
            y2 = 0
            if len(segmentsNew)!=3:
                print("Warning: segment format does not match bat mode")
            segment = Annotation.Segment([segmentsNew[0], segmentsNew[1], y1, y2, segmentsNew[2]])
            segmentsList.addSegment(segment)
        elif subfilter is not None:
            # for wavelet segments: (same as self.species!="Any sound")
            y1 = subfilter["FreqRange"][0]
            y2 = min(subfilter["FreqRange"][1], self.sp.audio_data.sample_rate//2)
            for s in segmentsNew:
                segment = Annotation.Segment([s[0][0], s[0][1], y1, y2, [{"species": species, "certainty": s[1], "filter": filtName, "calltype": subfilter["calltype"]}]])
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
        segmentList.metadata["Duration"] = float(len(self.sp.audio_data.data))/self.sp.audio_data.sample_rate
        segmentList.metadata["noiseLevel"] = None
        segmentList.metadata["noiseTypes"] = []

        segmentList.saveJSON(str(filename) + suffix)
        return 1

    def _needsResampling(self, originalRate, targetRate):
        """
        Helper method to determine if resampling is actually needed.
        Returns True if resampling required, False if rates match or node adjustment can be used.
        
        Args:
            originalRate: Original audio sample rate
            targetRate: Target sample rate required by filter
            
        Returns:
            bool: True if full resampling needed, False otherwise
        """
        if originalRate == targetRate:
            return False
        
        # WaveletSegment can handle 2x and 4x upsampling via node adjustment (more efficient)
        # This is already implemented in WaveletSegment.readBatch
        ratio = targetRate / originalRate
        if ratio in [2.0, 4.0]:
            print(f"Will use node adjustment instead of resampling ({originalRate} Hz -> {targetRate} Hz)")
            return False
        
        return True

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
            #self.sp.readBmp(self.filename, rotate=True, repeat=False)
        else:
            self.sp.readSoundFile(filename)

        print("Read %d samples, %f s at %d Hz" % (len(self.sp.audio_data.data), float(len(self.sp.audio_data.data))/self.sp.audio_data.sample_rate, self.sp.audio_data.sample_rate))

        # Read in stored segments (useful when doing multi-species)
        self.segments = Annotation.SegmentList()
        if bats or anysound or not os.path.isfile(filename + '.data'):
            # Initialize default metadata values
            self.segments.metadata = dict()
            self.segments.metadata["Operator"] = "Auto"
            self.segments.metadata["Reviewer"] = ""
            self.segments.metadata["Duration"] = float(len(self.sp.audio_data.data))/self.sp.audio_data.sample_rate
            # wipe all segments:
            print("Wiping all previous segments")
            self.segments.clear()
        else:
            hasmetadata = self.segments.parseJSON(filename+'.data', float(len(self.sp.audio_data.data))/self.sp.audio_data.sample_rate)
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

    # ========== BAT PROCESSING METHODS ==========
    # All bat-specific detection, classification, and processing methods are grouped here
    
    def processBatFile(self, filename, start, end, thisPageLen, NNmodel):
        """
        Unified bat processing method that handles click detection, NN classification, and labeling.
        
        Args:
            filename: path to the audio file
            start: start sample of the current page
            end: end sample of the current page  
            thisPageLen: length of the page in seconds
            NNmodel: list containing [model, win, inputdim, output, ..., [thr1, thr2]]
            
        Returns:
            List of segments in format [start_time, end_time, labels] or empty list if no bats detected
        """
        print("Processing bat file...")
        
        # Step 1: Detect clicks in spectrogram
        click_label, data_test, count = self.ClickSearch(self.sp.sg, filename, virginia=True)
        print(f"Click detection: {click_label}, {count} clicks found")
        
        if click_label != 'Click' or count == 0:
            print("No clicks detected")
            return []
        
        # Step 2: Prepare data for NN classification
        model = NNmodel[0]
        thr1 = NNmodel[5][0]
        thr2 = NNmodel[5][1]
        
        sg_test = np.ndarray(shape=(np.shape(data_test)[0], np.shape(data_test[0][0])[0], 
                                    np.shape(data_test[0][0])[1]), dtype=float)
        spec_id = []
        print(f'Number of file spectrograms: {np.shape(data_test)[0]}')
        
        for j in range(np.shape(data_test)[0]):
            maxg = np.max(data_test[j][0][:])
            sg_test[j][:] = data_test[j][0][:] / maxg
            spec_id.append(data_test[j][1:3])
        
        # Step 3: Run NN classification
        x_test = sg_test
        print(f"Shape before reshape: {x_test.shape}")
        test_images = x_test.reshape(x_test.shape[0], 6, 512, 1)
        print(f"Shape after reshape: {test_images.shape}")
        test_images = test_images.astype('float32')
        
        predictions = model.predict(test_images)
        
        # Step 4: Generate labels from predictions
        print('Assessing file label...')
        labels = self.labelBatFile(predictions, thr1=thr1, thr2=thr2)
        print('NN detected:', labels)
        
        if len(labels) == 0:
            return []
        
        # Step 5: Create segment with labels
        thisPageStart = start / self.sp.audio_data.sample_rate
        return [[thisPageStart, thisPageLen, labels]]
    
    def updateDataset(self, file_name, featuress, count, spectrogram, click_start, click_end, dt=None):
        """
        Update Dataset with current segment
        It take a piece of the spectrogram with fixed length centered in the click
        """
        win_pixel=1
        ls = np.shape(spectrogram)[1]-1
        click_center=int((click_start+click_end)/2)

        start_pixel=click_center-win_pixel
        if start_pixel<0:
            win_pixel2=win_pixel+np.abs(start_pixel)
            start_pixel=0
        else:
            win_pixel2=win_pixel

        end_pixel=click_center+win_pixel2
        if end_pixel>ls:
            start_pixel-=end_pixel-ls+1
            end_pixel=ls-1
            # this code above fails for sg less than 4 pixels wide
        sgRaw=spectrogram[:,start_pixel:end_pixel+1]  # note I am saving the spectrogram in the right dimension
        sgRaw=np.repeat(sgRaw,2,axis=1)
        sgRaw=(np.flipud(sgRaw)).T  # flipped spectrogram to make it consistent with Niro Method
        featuress.append([sgRaw.tolist(), file_name, count])  # not storing segment and label informations

        count += 1

        return featuress, count

    def ClickSearch(self, imspec, file,virginia=True):
        """
        searches for clicks in the provided imspec, saves dataset
        returns click_label, dataset and count of detections

        The search is made on the spectrogram image that we know to be generated
        with parameters (1024,512)
        Click presence is assessed for each spectrogram column: if the mean in the
        frequency band [f0, f1] (*) is bigger than a treshold we have a click
        thr=mean(all_spec)+std(all_spec) (*)

        The clicks are discarded if longer than 0.05 sec

        Clicks are stored into featuress using updateDataset

        imspec: unrotated spectrogram (rows=time)
        file: NOTE originally was basename, now full filename

        The virginia flag is because I've (SM) added the other functionality, which just finds the first and last clicks to get the time
        """
        featuress = []
        count = 0

        df=self.sp.audio_data.sample_rate//2 /(np.shape(imspec)[0]+1)  # frequency increment
        dt=self.sp.incr/self.sp.audio_data.sample_rate  # self.sp.incr is set to 512 for bats
        # dt=0.002909090909090909
        # up_len=math.ceil(0.05/dt) #0.5 second lenth in indices divided by 11
        up_len=17
        # up_len=math.ceil((0.5/11)/dt)

        # Frequency band
        f0=24000
        index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
        f1=54000
        index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up

        # Mean in the frequency band
        mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0)

        # Threshold
        mean_spec_all=np.mean(imspec, axis=0)[2:]
        thr_spec=(np.mean(mean_spec_all)+np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))

        ## clickfinder
        # check when the mean is bigger than the threshold
        # clicks is an array which elements are equal to 1 only where the sum is bigger
        # than the mean, otherwise are equal to 0
        clicks = mean_spec>thr_spec
        
        if virginia:
            clicks_indices = np.nonzero(clicks)
            # check: if I have found somenthing
            if np.shape(clicks_indices)[1]==0:
                click_label='None'
                return click_label, featuress, count
                # not saving spectrograms

            # Discarding segments too long or too short and saving spectrogram images
            click_start=clicks_indices[0][0]
            click_end=clicks_indices[0][0]
            for i in range(1,np.shape(clicks_indices)[1]):
                if clicks_indices[0][i]==click_end+1:
                    click_end=clicks_indices[0][i]
                else:
                    if click_end-click_start+1>up_len:
                        clicks[click_start:click_end+1] = False
                    else:
                        # savedataset
                        featuress, count = self.updateDataset(file, featuress, count, imspec, click_start, click_end, dt)
                    # update
                    click_start=clicks_indices[0][i]
                    click_end=clicks_indices[0][i]

            # checking last loop with end
            if click_end-click_start+1>up_len:
                clicks[click_start:click_end+1] = False
            else:
                featuress, count = self.updateDataset(file, featuress, count, imspec, click_start, click_end, dt)

            # Assigning: click label
            if np.any(clicks):
                click_label='Click'
            else:
                click_label='None'

            return click_label, featuress, count
        else:
            inds = np.where(clicks>0)[0]
            if (len(inds)) > 0:
                # Have found something, now find first that isn't too long
                flag = False
                start = inds[0]
                while flag:
                    i=1
                    while inds[i]-inds[i-1] == 1:
                        i+=1
                    end = i
                    if end-start<up_len:
                        flag=True
                    else:
                        start = inds[end+1]

                first = start

                # And last that isn't too long
                flag = False
                end = inds[-1]
                while flag:
                    i=len(inds)-1
                    while inds[i]-inds[i-1] == 1:
                        i-=1
                    start = i
                    if end-start<up_len:
                        flag=True
                    else:
                        end = inds[start-1]
                last = end
                return [first,last]
            else:
                return None

    def labelBatFile(self, predictions, thr1, thr2):
        """
        uses the predictions made by the NN to update the filewise annotations
        when we have 3 labels: 0 (LT), 1(ST), 2 (Noise)

        METHOD: evaluation of probability over files combining mean of probability
            + best3mean of probability against thr1 and thr2, respectively

        Returns: species labels (list of dicts), compatible w/ the label format on Segments
        """

        # Assessing file label
        # inizialization
        # vectors storing classes probabilities
        LT_prob=[]  # class 0
        ST_prob=[]  # class 1
        NT_prob=[]  # class 2
        spec_num=0   # counts number of spectrograms per file
        # flag: if no click detected no spectrograms
        click_detected_flag=False
        # looking for all the spectrogram related to this file

        for k in range(np.shape(predictions)[0]):
            click_detected_flag=True
            spec_num+=1
            LT_prob.append(predictions[k][0])
            ST_prob.append(predictions[k][1])
            NT_prob.append(predictions[k][2])

        # if no clicks => automatically Noise
        label = []

        if click_detected_flag:
            # mean
            LT_mean=np.mean(LT_prob)*100
            ST_mean=np.mean(ST_prob)*100

            # best3mean
            LT_best3mean=0
            ST_best3mean=0

            # LT
            ind = np.array(LT_prob).argsort()[-3:][::-1]
            # adding len ind in order to consider also the cases when we do not have 3 good examples
            if len(ind)==1:
                # this means that there is only one prob!
                LT_best3mean+=LT_prob[0]
            else:
                for j in range(len(ind)):
                    LT_best3mean+=LT_prob[ind[j]]
            LT_best3mean/= 3
            LT_best3mean*=100

            # ST
            ind = np.array(ST_prob).argsort()[-3:][::-1]
            # adding len ind in order to consider also the cases when we do not have 3 good examples
            if len(ind)==1:
                # this means that there is only one prob!
                ST_best3mean+=ST_prob[0]
            else:
                for j in range(len(ind)):
                    ST_best3mean+=ST_prob[ind[j]]
            ST_best3mean/= 3
            ST_best3mean*=100

            # ASSESSING FILE LABEL
            hasST = ST_mean>=thr1 or ST_best3mean>=thr2
            hasLT = LT_mean>=thr1 or LT_best3mean>=thr2
            hasSTlow = ST_mean<thr1 and ST_best3mean<thr2
            hasLTlow = LT_mean<thr1 and LT_best3mean<thr2
            reallyHasST = ST_mean>=thr1 and ST_best3mean>=thr2
            reallyHasLT = LT_mean>=thr1 and LT_best3mean>=thr2
            HasBat = LT_mean>=thr1 and ST_mean>=thr1

            if reallyHasLT and hasSTlow:
                label.append({"species": "Long-tailed bat", "certainty": 100})
            elif reallyHasLT and reallyHasST:
                label.append({"species": "Long-tailed bat", "certainty": 100})
            elif hasLT and ST_mean<thr1:
                label.append({"species": "Long-tailed bat", "certainty": 50})
            elif HasBat:
                label.append({"species": "Long-tailed bat", "certainty": 50})

            if reallyHasST and hasLTlow:
                label.append({"species": "Short-tailed bat", "certainty": 100})
            elif reallyHasLT and reallyHasST:
                label.append({"species": "Short-tailed bat", "certainty": 100})
            elif hasST and LT_mean<thr1:
                label.append({"species": "Short-tailed bat", "certainty": 50})
            elif HasBat:
                label.append({"species": "Short-tailed bat", "certainty": 50})

        return label
    
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
            return self._exportBatXML(dirName, savefile, threshold1, threshold2, operator, site, namedict)
        elif format == 'csv':
            return self._exportBatCSV(dirName, savefile, threshold1, threshold2, operator)
        elif format == 'passes':
            return self._exportBatPasses(dirName, savefile)
        else:
            print(f"Unknown format: {format}")
            return 0
    
    def _exportBatXML(self, dirName, savefile, threshold1, threshold2, operator, site, namedict):
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
                        segments = Annotation.SegmentList()
                        segments.parseJSON(os.path.join(root, filename))
                        
                        # Determine label from segments
                        label = self._getBatLabel(segments, threshold1, threshold2)
                        
                        # Create XML elements
                        etree.SubElement(s1, "AssignedBatCategory").text = str(namedict[label])
                        etree.SubElement(s1, "AssignedSite").text = site
                        etree.SubElement(s1, "AssignedUser").text = operator
                        etree.SubElement(s1, "RecTime").text = self._parseTimeDate(filename)
                        etree.SubElement(s1, "RecordingFileName").text = filename[:-5]
                        etree.SubElement(s1, "RecordingFolderName").text = ".\\" + os.path.split(root)[-1]
                        etree.SubElement(s1, "MeasureTimeFrom").text = str(0)
                
                # Write XML file
                print("writing to", os.path.join(root, savefile))
                with open(os.path.join(root, savefile), "wb") as f:
                    f.write(etree.tostring(etree.ElementTree(start), pretty_print=True, 
                                         xml_declaration=True, encoding='utf-8'))
        return 1
    
    def _exportBatCSV(self, dirName, savefile, threshold1, threshold2, operator):
        """Export bat results to BatSearch CSV format."""
        f = open(os.path.join(dirName, savefile), 'w')
        f.write('Date,Time,AssignedSite,Category,Foldername,Filename,Observer\n')
        
        for root, dirs, files in os.walk(dirName):
            dirs.sort()
            files.sort()
            for filename in files:
                if filename.endswith('.data'):
                    segments = Annotation.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
                    
                    label = self._getBatLabel(segments, threshold1, threshold2)
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
    
    def _exportBatPasses(self, dirName, savefile):
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
                    segments = Annotation.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
                    
                    label = 'Non-bat'
                    length = "0"
                    
                    if len(segments) > 0:
                        fn = filename[:-5]
                        self.sp.readBmp(os.path.join(root, fn), rotate=False, silent=True)
                        res = self.ClickSearch(self.sp.sg, None, virginia=False)
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
    
    def _getBatLabel(self, segments, threshold1, threshold2):
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
    
    def _parseTimeDate(self, filename):
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
        import datetime as dt
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
                    segments = Annotation.SegmentList()
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


