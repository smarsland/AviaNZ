
# Version 4.0 9/10/25
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

# BirdDetector.py
#
# Bird detection logic

import gc
import copy

from src.core import Segmentation
from src.core import Annotation
from src.core import WaveletSegment
from src.utils.exceptions import GentleExitException


class BirdDetector:
    """
    Handles bird detection using wavelet segmentation and post-processing.
    Extracted from AviaNZ_batchProcess to separate bird-specific logic.
    """
    
    def __init__(self, config, configdir):
        """Initialize bird detector with configuration."""
        self.config = config
        self.configdir = configdir
    
    def detectBirdsInFile(self, sp, segments, species, filters, NNDicts, options, anySound=False, 
                         testmode=False, segments_nonn=None, check_cancelled=None):
        """Detect birds in audio file using wavelet segmentation and optional NN classification."""
        # Calculate page size based on audio sample rate
        samplesInPage = self.calculatePageSize(sp, species, filters)
        numPages = (len(sp.audio_data.data) - 1) // samplesInPage + 1
        
        # Process each page
        for page in range(numPages):
            print("Segmenting page %d / %d" % (page+1, numPages))
            start = page * samplesInPage
            end = min(start + samplesInPage, len(sp.audio_data.data))
            thisPageLen = (end - start) / sp.audio_data.sample_rate
            
            if thisPageLen < 2:
                print("Warning: can't process short file ends (%.2f s)" % thisPageLen)
                continue
            
            # Process "Any Sound" detection if enabled
            if anySound:
                self.processAnySound(sp, segments, start, end, options, check_cancelled)
            
            # Process bird-specific filters
            self.processBirdFilters(sp, segments, species, filters, NNDicts, start, end, 
                                   options, testmode, segments_nonn, check_cancelled)
    
    def calculatePageSize(self, sp, species, filters):
        """Calculate appropriate page size based on audio characteristics and filters."""
        if hasattr(sp, 'audio_data') and sp.audio_data:
            if sp.audio_data.sample_rate <= 4000:
                # Low frequency recordings (e.g., bittern at 4000 Hz)
                return 300 * sp.audio_data.sample_rate  # ~5 min pages
        
        # Standard recordings: use 15 min pages normalized to 16 kHz
        return 900 * 16000
        
    def processAnySound(self, sp, segments, start, end, options, check_cancelled):
        """Process generic "Any sound" detection using median clipping."""
        # Create spectrogram for median clipping
        if not hasattr(sp, 'sg') or sp.sg is None:
            _ = sp.spectrogram(window_width=self.config['window_width'], 
                             incr=self.config['incr'],
                             window=self.config['windowType'],
                             sgType=self.config['sgType'],
                             sgScale=self.config['sgScale'],
                             nfilters=self.config['nfilters'],
                             mean_normalise=self.config['sgMeanNormalise'],
                             equal_loudness=self.config['sgEqualLoudness'],
                             onesided=self.config['sgOneSided'],
                             start=start, stop=end)
        
        seg = Segmentation.Segmenter(sp, sp.audio_data.sample_rate)
        thisPageSegs = seg.medianClip(thr=3.5)
        
        # Post-process
        print("Segments detected: ", len(thisPageSegs))
        print("Post-processing...")
        post = Segmentation.PostProcess(configdir=self.configdir, 
                                 audioData=sp.audio_data.data[start:end], 
                                 sampleRate=sp.audio_data.sample_rate, 
                                 segments=thisPageSegs, 
                                 subfilter={}, cert=0)
        
        if options['mergeSyllables']:
            post.joinGaps(options['maxgap'])
            post.deleteShort(options['minlen'])
            post.splitLong(options['maxlen'])
        
        # Adjust segment starts for pages
        if start != 0:
            for segment in post.segments:
                segment[0][0] += start / sp.audio_data.sample_rate
                segment[0][1] += start / sp.audio_data.sample_rate
        
        # Add to segments with generic labels
        self.makeGenericSegments(segments, post.segments)
        
        del seg
        gc.collect()
        
        # Check for cancellation
        if check_cancelled and check_cancelled():
            raise GentleExitException
    
    def processBirdFilters(self, sp, segments, species, filters, NNDicts, start, end, 
                          options, testmode, segments_nonn, check_cancelled):
        """Process bird-specific filters using wavelet segmentation."""
        # Group filters by required sample rate
        uniqueSampleRates = set([filt["SampleRate"] for filt in filters])
        
        for targetSampleRate in uniqueSampleRates:
            # Get all filters that need this sample rate
            filtersAtSampleRate = [filters[i] for i in range(len(filters)) 
                                  if filters[i]["SampleRate"] == targetSampleRate]
            speciesAtSampleRate = [species[i] for i in range(len(filters)) 
                                  if filters[i]["SampleRate"] == targetSampleRate]
            
            # Skip if this is bat processing
            if "NZ Bats" in speciesAtSampleRate:
                continue
            
            if len(speciesAtSampleRate) == 0:
                continue
                
            print(f"Processing sample rate {targetSampleRate} Hz for species: {speciesAtSampleRate}")
            
            # Initialize wavelet segmentation for bird processing
            ws = WaveletSegment.WaveletSegment(wavelet='dmey2')
            useWind = options['wind'] in ["OLS wind filter (recommended)", "Robust wind filter (experimental, slow)"]
            ws.readBatch(sp.audio_data.data[start:end], sp.audio_data.sample_rate, 
                        d=False, spInfo=filtersAtSampleRate, wpmode="new", wind=useWind)
            
            for speciesix in range(len(filtersAtSampleRate)):
                print("Working with recogniser:", filtersAtSampleRate[speciesix])
                spInfo = filtersAtSampleRate[speciesix]
                
                # Bird detection by wavelets
                if "method" not in spInfo or spInfo["method"] == "wv":
                    thisPageSegs = ws.waveletSegment(speciesix, wpmode="new")
                elif spInfo["method"] == "chp":
                    thisPageSegs = ws.waveletSegmentChp(speciesix, alg=2, wind=options['wind'])
                else:
                    print("ERROR: unrecognised method", spInfo["method"])
                    raise Exception
                
                print("Segments detected (all subfilters): ", thisPageSegs)
                if not testmode:
                    print("Post-processing...")
                
                # Process each subfilter
                for filtix in range(len(spInfo['Filters'])):
                    NNmodel = None
                    if 'NN' in spInfo and spInfo['NN']['NN_name'] in NNDicts.keys():
                        NNmodel = NNDicts[spInfo['NN']['NN_name']]
                    
                    if not testmode:
                        # Normal processing
                        postsegs = self.postProcFull(thisPageSegs, spInfo, filtix, start, end, 
                                                   NNmodel, sp)
                        self.makeBirdSegments(segments, postsegs, speciesAtSampleRate[speciesix], 
                                             spInfo["species"], spInfo['Filters'][filtix], 
                                             sp.audio_data.sample_rate)
                        
                        # Check for cancellation
                        if check_cancelled and check_cancelled():
                            raise GentleExitException
                    else:
                        # Test mode: process both with and without NN
                        if segments_nonn is not None:
                            postsegs_nonn = self.postProcFull(copy.deepcopy(thisPageSegs), spInfo, 
                                                            filtix, start, end, None, sp)
                            self.makeBirdSegments(segments_nonn, postsegs_nonn, 
                                                 speciesAtSampleRate[speciesix], 
                                                 spInfo["species"], spInfo['Filters'][filtix],
                                                 sp.audio_data.sample_rate)
                        
                        postsegs = self.postProcFull(copy.deepcopy(thisPageSegs), spInfo, filtix, 
                                                   start, end, NNmodel, sp)
                        self.makeBirdSegments(segments, postsegs, speciesAtSampleRate[speciesix], 
                                             spInfo["species"], spInfo['Filters'][filtix],
                                             sp.audio_data.sample_rate)
    
    def postProcFull(self, segments, spInfo, filtix, start, end, NNmodel, sp):
        """Apply full post-processing: NN classification, gap joining, fundamental frequency detection."""
        subfilter = spInfo["Filters"][filtix]
        
        # PostProcess handles any needed resampling from current rate to target rate
        post = Segmentation.PostProcess(configdir=self.configdir, 
                                 audioData=sp.audio_data.data[start:end],
                                 sampleRate=sp.audio_data.sample_rate, 
                                 tgtsampleRate=spInfo["SampleRate"],
                                 segments=segments[filtix], subfilter=subfilter,
                                 NNmodel=NNmodel, cert=50)
        print("Segments detected after WF: ", len(segments[filtix]))

        if NNmodel:
            print('Post-processing with NN')
            post.NN()

        # Fund freq and merging. Only do for standard wavelet filter currently:
        if "method" not in spInfo or spInfo["method"] == "wv":
            if 'F0' in subfilter and 'F0Range' in subfilter and subfilter["F0"]:
                print("Checking for fundamental frequency...")
                post.fundamentalFrq()

            post.joinGaps(maxgap=subfilter['TimeRange'][3])

        # delete short segments, if requested:
        if subfilter['TimeRange'][0] > 0:
            post.deleteShort(minlength=subfilter['TimeRange'][0])

        # adjust segment starts for 15min "pages"
        if start != 0:
            for seg in post.segments:
                seg[0][0] += start / sp.audio_data.sample_rate
                seg[0][1] += start / sp.audio_data.sample_rate
        
        print("After post-processing: ", post.segments)
        return post.segments
    
    def makeGenericSegments(self, segmentsList, segmentsNew):
        """Add generic "Don't Know" segments."""
        y1 = 0
        y2 = 0
        species = "Don't Know"
        cert = 0.0
        segmentsList.addFromTimeRanges(segmentsNew, y1, y2, species=species, certainty=cert)
    
    def makeBirdSegments(self, segmentsList, segmentsNew, filtName, species, subfilter, sampleRate):
        """Add bird segments with species labels and metadata."""
        y1 = subfilter["FreqRange"][0]
        y2 = min(subfilter["FreqRange"][1], sampleRate // 2)
        
        for s in segmentsNew:
            segment = Annotation.Segment(start_time=s[0][0], end_time=s[0][1], freq_low=y1, freq_high=y2, 
                                     labels=[{"species": species, "certainty": s[1], 
                                       "filter": filtName, "calltype": subfilter["calltype"]}])
            segmentsList.addSegment(segment)