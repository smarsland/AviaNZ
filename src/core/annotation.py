
# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2025

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

# Data objects for handling segments

from src.core import message_popup

import numpy as np
import json
import math
import copy
import soundfile as sf

class Segment:
    """ A single AviaNZ annotation ("segment" or "box" type).
        
        Attributes:
            start_time: Start time in seconds (float, >= 0)
            end_time: End time in seconds (float, >= 0)
            freq_low: Lower frequency bound in Hz (int, >= 0)
            freq_high: Upper frequency bound in Hz (int, >= 0)
            labels: List of label dicts with 'species', 'certainty', and optional 'filter', 'calltype'
    """
    def __init__(self, start_time, end_time, freq_low, freq_high, labels):
        if start_time < 0 or end_time < 0:
            raise ValueError("Segment times must be positive or 0")
        if freq_low < 0 or freq_high < 0:
            raise ValueError("Segment frequencies must be positive or 0")
        if not isinstance(labels, list):
            raise ValueError("Segment labels must be a list")

        for lab in labels:
            if not isinstance(lab, dict):
                raise ValueError("Segment label must be a dict")
            if "species" not in lab or not isinstance(lab["species"], str):
                raise ValueError("species bad or missing from label")
            if "certainty" not in lab or not isinstance(lab["certainty"], (int, float)):
                raise ValueError("certainty bad or missing from label")
            if "filter" in lab and lab["filter"]!="M" and "calltype" not in lab:
                raise ValueError("calltype required when automated filter provided in label")

        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.freq_low = int(freq_low)
        self.freq_high = int(freq_high)
        self.labels = labels

        self.keys = [lab['species'] for lab in self.labels]
        if len(self.keys) > len(set(self.keys)):
            raise ValueError("non-unique species detected")
    
    @classmethod
    def from_list(cls, data):
        """Create a Segment from legacy list format [start_time, end_time, freq_low, freq_high, labels].
        This is only used when parsing old JSON files."""
        if not isinstance(data, (list, tuple)):
            raise ValueError("from_list expects a list or tuple")
        if len(data) != 5:
            raise ValueError("from_list requires 5 elements: [start_time, end_time, freq_low, freq_high, labels], got %d" % len(data))
        return cls(start_time=data[0], end_time=data[1], freq_low=data[2], freq_high=data[3], labels=data[4])
    
    def __repr__(self):
        """String representation for debugging."""
        return f"Segment({self.start_time}, {self.end_time}, {self.freq_low}, {self.freq_high}, {len(self.labels)} labels)"
    
    def __str__(self):
        """ Returns a nicely-formatted string of this segment's info."""
        s = []
        for lab in self.labels:
            labs = "sp.: %s, cert.: %d%%" % (lab["species"], lab["certainty"])
            if "filter" in lab and lab["filter"]!="M":
                labs += ", filter: " + lab["filter"]
            if "calltype" in lab:
                labs += ", call: " + lab["calltype"]
            s.append(labs)
        return "; ".join(s)
    
    def to_list(self):
        """Convert Segment back to list format for JSON serialization."""
        return [self.start_time, self.end_time, self.freq_low, self.freq_high, self.labels]
    
    def hasLabel(self, species):
        """ Check if label identified by species is present in this segment. """
        return species in self.keys

    def addLabel(self, species, certainty, **label):
        """ Adds a label to this segment.
            Species and certainty are required.
            Any further label properties (filter, calltype...) must be passed as keyword args:
              addLabel("LSK", 100, filter="M"...)
        """
        if "filter" in label and label["filter"]!="M" and "calltype" not in label:
            raise ValueError("calltype required when automated filter provided in label")
        if self.hasLabel(species):
            raise ValueError("this species label already present")
        label["species"] = species
        label["certainty"] = certainty

        self.labels.append(label)
        self.keys.append(species)
    
    def getKeys(self):
        return [lab['species'] for lab in self.labels]
    
    def getKeysWithCalltypes(self):
        return [(lab['species'], lab['calltype'] if 'calltype' in lab else None) for lab in self.labels]
    
    def getCalltype(self, species):
        for lab in self.labels:
            if lab["species"]==species:
                if 'calltype' in lab:
                    return lab['calltype']
        return None

    ### --- couple functions to process all labels for a given species ---

    def wipeSpecies(self, species):
        """ Remove all labels for species, return True if all labels were wiped
            (and the interface should delete the segment).
        """
        deletedAll = list(set([lab["species"] for lab in self.labels])) == [species]
        for lab in reversed(self.labels):
            if lab["species"]==species:
                print("Wiping label", lab)
                self.removeLabel(lab["species"])
        return deletedAll

    def confirmLabels(self, species=None):
        """ Raise the certainty of this segment's uncertain labels to 100.
            Affects all species (if None) or indicated species.
            Ignores "Don't Know" labels.
        """
        toremove = []
        for labix in range(len(self.labels)):
            lab = self.labels[labix]
            if (species is None or lab["species"]==species) and lab["certainty"] < 100 and lab["species"]!="Don't Know":
                if (lab["species"], 100) in self.keys:
                    toremove.append(lab)
                else:
                    lab["certainty"] = 100
        for trlab in toremove:
            self.removeLabel(trlab["species"])

    def questionLabels(self, species=None):
        """ Lower the certainty of this segment's certain labels to 50.
            Affects all species (if None) or indicated species.
            Ignores "Don't Know" labels.
            Returns True if it changed any labels.
        """
        anyChanged = False
        toremove = []
        for labix in range(len(self.labels)):
            lab = self.labels[labix]
            if (species is None or lab["species"]==species) and lab["certainty"]==100 and lab["species"]!="Don't Know":
                otherLabels = [k[0]==lab["species"] and k[1]<100 for k in self.keys]
                if any(otherLabels):
                    toremove.append(lab)
                else:
                    lab["certainty"] = 50
                anyChanged = True
        for trlab in toremove:
            self.removeLabel(trlab["species"])
        return anyChanged

    def removeLabel(self, species):
        """ Removes label from this segment.
            Does not delete the actual segment - that's left for the interface to take care of.
        """
        deleted = False
        for lab in self.labels:
            if lab["species"]==species:
                self.labels.remove(lab)
                try:
                    self.keys.remove(species)
                except Exception as e:
                    text = "************ WARNING ************\n"
                    text += str(e)
                    text += "\nWhile trying to remove key "+str(species)+" from "+ str(self.labels)
                    text += "\nWhich had keys" + str(self.keys)
                    try:
                        msg = message_popup.MessagePopup("w", "ERROR - please report", text)
                        if hasattr(msg, 'exec'):
                            msg.exec()
                    except Exception:
                        print('ERROR - please report:', text)
                if len(self.labels)==0:
                    self.addLabel("Don't Know", 0)
                deleted = True
                break

        if not deleted:
            print("ERROR: could not find species to remove:", species)
    
    def clearLabels(self):
        self.labels = []
        self.keys = []


class SegmentList(list):
    """ List of Segments. Deals with I/O - parsing JSON,
        and retrieving the right Segment from this list.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize SegmentList with metadata attribute."""
        super().__init__(*args, **kwargs)
        self.metadata = {}

    def readDurationFromAudio(self, file):
        """Read duration from audio file. Returns 0 for bitmap files."""
        audio_file = file.removesuffix('.data')
        if audio_file.lower().endswith('.bmp'):
            return 0
        info = sf.info(audio_file)
        return info.frames / info.samplerate

    def parseMetadataOldFormat(self, annots, file, silent):
        """Parse metadata from old list-based metadata format."""
        if annots[0][0] == -1:
            self.metadata = {"Operator": annots[0][2], "Reviewer": annots[0][3]}
            
            if isinstance(annots[0][1], (int, float)) and 0 < annots[0][1] < 100000:
                self.metadata["Duration"] = annots[0][1]
            else:
                self.metadata["Duration"] = self.readDurationFromAudio(file)
            
            if len(annots[0]) >= 5 and isinstance(annots[0][4], list):
                self.metadata["noiseLevel"] = annots[0][4][0]
                self.metadata["noiseTypes"] = annots[0][4][1]
            else:
                self.metadata["noiseLevel"] = None
                self.metadata["noiseTypes"] = []
            
            del annots[0]
        else:
            if not silent:
                print("very old format metadata detected")
            self.metadata["Duration"] = self.readDurationFromAudio(file)
            self.metadata["Operator"] = ""
            self.metadata["Reviewer"] = ""

    def parseMetadataNewFormat(self, annots, file):
        """Parse metadata from new dict-based format."""
        self.metadata = annots[0]
        
        if "Duration" not in self.metadata:
            self.metadata["Duration"] = self.readDurationFromAudio(file)
        
        if "Operator" not in self.metadata:
            self.metadata["Operator"] = ""
        if "Reviewer" not in self.metadata:
            self.metadata["Reviewer"] = ""
        
        del annots[0]

    def migrateLabelFormat(self, label):
        """Convert old label formats (strings) to new dict format."""
        if isinstance(label, dict):
            return label
        if label == "Don't Know":
            return {"species": "Don't Know", "certainty": 0}
        if label.endswith('?'):
            return {"species": label[:-1], "certainty": 50}
        return {"species": label, "certainty": 100}

    def migrateFrequencyFormat(self, annot, file):
        """Convert old 0-1 frequency format to Hz."""
        if 0 < annot[2] < 1.1 and 0 < annot[3] < 1.1:
            print("Warning: updating old-format frequency marks")
            info = sf.info(file[:-5])
            annot[2] *= info.samplerate
            annot[3] *= info.samplerate

    def parseJSON(self, file, silent=False):
        """Read JSON annotation file and populate segments.
        
        Args:
            file: Path to .data JSON file
            silent: Suppress info messages
        """
        with open(file, 'r') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"Empty JSON file: {file}")
            annots = json.loads(content)

        if len(annots) == 0:
            raise ValueError("Empty annotation file")

        self.metadata = {}
        
        if isinstance(annots[0], list):
            if not silent:
                print("old format metadata detected")
            self.parseMetadataOldFormat(annots, file, silent)
        elif isinstance(annots[0], dict):
            self.parseMetadataNewFormat(annots, file)
        else:
            raise ValueError("Unrecognized metadata format")

        self.clear()
        for annot in annots:
            if not isinstance(annot, list) or len(annot) != 5:
                raise ValueError(f"Annotation in wrong format: {annot}")

            self.migrateFrequencyFormat(annot, file)
            
            if isinstance(annot[4], str):
                annot[4] = [annot[4]]
            
            if isinstance(annot[4], list):
                annot[4] = [self.migrateLabelFormat(lab) for lab in annot[4]]

            if len(annot[4]) == 0:
                annot[4] = [{"species": "Don't Know", "certainty": 0}]
            
            segment = Segment.from_list(annot)
            self.addSegment(segment)
        
        if not silent:
            print("%d segments read" % len(self))

    def addSegment(self, segment):
        """Add a Segment object to the list."""
        if not isinstance(segment, Segment):
            raise TypeError("addSegment only accepts Segment objects")
        self.append(segment)

    def addFromTimeRanges(self, time_ranges, freq_low, freq_high, **kwd):
        """Create and add Segments from time ranges (PostProcess output format).
        
        Args:
            time_ranges: List of [[start_time, end_time], probability] pairs from PostProcess
            freq_low: Lower frequency bound in Hz
            freq_high: Upper frequency bound in Hz
            **kwd: Label properties (species, certainty, filter, calltype, etc.)
        """
        if freq_low < 0 or freq_high < 0:
            raise ValueError(f"Invalid frequencies: {freq_low}, {freq_high}")

        for time_range in time_ranges:
            start_time, end_time = time_range[0]
            segment = Segment(start_time, end_time, freq_low, freq_high, [kwd])
            self.addSegment(segment)

    def getSpecies(self, species):
        """ Returns indices of all segments that have the indicated species in label. """
        out = []
        for segi in range(len(self)):
            # check each label in this segment:
            seg = self[segi]
            labs = seg.labels
            for lab in labs:
                if lab["species"] == species:
                    out.append(segi)
                    # go to next seg
                    break
        return(out)

    def getCalltype(self, species, calltype):
        """ Returns indices of all segments that have the indicated species & calltype in label. """
        out = []
        for segi in range(len(self)):
            # check each label in this segment:
            seg = self[segi]
            labs = seg.labels
            for lab in labs:
                try:
                    if lab["species"] == species and lab["calltype"] == calltype:
                        out.append(segi)
                        # go to next seg
                        break
                except:
                    pass
        return(out)

    def saveJSON(self, file, reviewer=""):
        """Save segments to JSON file."""
        if reviewer != "":
            self.metadata["Reviewer"] = reviewer
        annots = [self.metadata]
        for seg in self:
            if not isinstance(seg, Segment):
                raise TypeError(f"SegmentList can only contain Segment objects, found {type(seg)}")
            annots.append(seg.to_list())

        with open(file, 'w') as f:
            json.dump(annots, f)
            f.write("\n")

    def orderTime(self):
        """ Returns the order of segments in this list sorted by start time.
            Sorts itself using the order. Can then be used to sort any additional lists
            in matching order (graphics etc). """
        sttimes = [s.start_time for s in self]
        sttimes = np.argsort(sttimes)
        self.sort(key=lambda s: s.start_time)

        return(sttimes)

    def splitLongSeg(self, maxlen=10, species=None):
        """
        Splits long segments (> maxlen) evenly
        Operates on segment data structure
        [1,5,a,b, [{}]] -> [1,3,a,b, [{}]], [3,5,a,b, [{}]]
        """
        toadd = []
        for seg in self:
            # if species is given, only split segments where it is present:
            if species is not None and species not in [lab["species"] for lab in seg.labels]:
                continue
            l = seg.end_time - seg.start_time
            if l > maxlen:
                n = int(np.ceil(l/maxlen))
                d = l/n
                # adjust current seg to be the first piece
                seg.end_time = seg.start_time + d
                for i in range(1,n):
                    end = min(l, d * (i+1))
                    segpiece = copy.deepcopy(seg)
                    segpiece.start_time = seg.start_time + d*i
                    segpiece.end_time = seg.start_time + end
                    # store further pieces to be added
                    toadd.append(segpiece)
        # now add them, to avoid messing with the loop length above
        for seg in toadd:
            self.addSegment(seg)

    def mergeSplitSeg(self):
        """ Inverse of the above: merges overlapping segments.
            Merges only segments with identical labels,
            so e.g. [kiwi, morepork] [kiwi] will not be merged.
            Unlike analogs in Segmenter and PostProcess,
            merges segments that only touch ([1,2][2,3]->[1,3]).

            DOES NOT DELETE segments - returns indices to be deleted,
            so an external handler needs to do the required interface updates.

            ASSUMES sorted input!
        """
        todelete = []
        if len(self)==0:
            return []

        # ideally, we'd loop over different labels, but not easy since they're unhashable.
        # so we use a marker array to keep track of checked segments:
        done = np.zeros(len(self))
        while not np.all(done):
            firstsegi = None
            for segi in range(len(self)):
                # was this already reviewed (when mergin another sp combo)?
                if done[segi]==1:
                    continue
                # sets the first segment of this label
                # (and the sp combo that will be merged now)
                if firstsegi is None:
                    firstsegi = segi
                    done[segi] = 1
                    continue
                # ignore segments with labels other than the current one
                if self[segi].labels != self[firstsegi].labels:
                    continue
                # for subsequent segs, see if this can be merged to the previous one
                if self[segi].start_time <= self[firstsegi].end_time:
                    self[firstsegi].end_time = max(self[segi].end_time, self[firstsegi].end_time)
                    done[segi] = 1
                    # mark this for deleting
                    todelete.append(segi)
                else:
                    firstsegi = segi
                    done[segi] = 1
                    # no need to delete anything
        # avoid duplicates in output to make life easier for later deletion
        todelete = list(set(todelete))
        todelete.sort(reverse=True)
        return todelete

    def getSummaries(self):
        """ Calculates some summary parameters relevant for populating training dialogs.
            and returns other parameters for populating the training dialogs.
        """
        if len(self)==0:
            print("ERROR: no annotations for this calltype found")
            return

        # get parameter limits for populating training dialogs:
        # FreqRange, in Hz
        fLow = np.min([seg.freq_low for seg in self])
        fHigh = np.max([seg.freq_high for seg in self])
        # TimeRange, in s
        lenMin = np.min([seg.end_time - seg.start_time for seg in self])
        lenMax = np.max([seg.end_time - seg.start_time for seg in self])

        return(lenMin, lenMax, fLow, fHigh)

    def exportGT(self, filename, species, resolution):
        """ Given the AviaNZ annotations, exports a 0/1 ground truth as a txt file
        filename - current wav file name.
        species - string, will export the annotations for it.
        resolution - resolution at which to dichotomize the timestamps.
           set this to match the analysis resolution
           (i.e. window, inc in waveletSegment).
           E.g. with integer parameters can use
           resolution = math.gcd(window, inc)
        """
        # number of segments of width window at inc overlap
        # Use floor to match wavelet extraction (which can't handle partial windows)
        duration = int(np.floor(self.metadata["Duration"] / resolution))
        filenameNoExtension = filename.rsplit('.', 1)[0]
        eFile = filenameNoExtension + '-GT.txt'

        # deal with empty files
        thisSpSegs = self.getSpecies(species)

        GT = np.tile([0, 0, None], (duration,1))
        # fill first column with "time"
        GT[:,0] = range(1, duration+1)
        GT[:,0] = GT[:,0] * resolution

        print("exporting GT with resolution", resolution)

        for segix in thisSpSegs:
            seg = self[segix]
            # start and end in resolution base
            s = int(max(0, math.floor(seg.start_time / resolution)))
            e = int(min(duration, math.ceil(seg.end_time / resolution)))
            for i in range(s, e):
                GT[i,1] = 1
                GT[i,2] = species
        GT = GT.tolist()

        if len(GT)==1:
            print("Warning: no annotations found in file!!!")

        # now save the resulting txt:
        with open(eFile, "w") as f:
            for l, el in enumerate(GT):
                string = '\t'.join(map(str,el))
                for item in string:
                    f.write(item)
                f.write('\n')
            f.write('\n')
            print("output successfully saved to file", eFile)
