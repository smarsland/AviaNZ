
# Segment.py
# A variety of segmentation algorithms for AviaNZ

# Version 3.4 18/12/24
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
import Spectrogram
import SupportClasses
import Shapes

import numpy as np
import scipy.ndimage as spi
from scipy import signal
import librosa
import time
from ext import ce_denoise as ce
import json
import os
import math
import copy
from scipy.interpolate import interp1d
from scipy.signal import medfilt
import skimage.measure as skm
import tensorflow as tf
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

import soundfile as sf

class Segment(list):
    """ A single AviaNZ annotation ("segment" or "box" type).
        Deals with identifying the right Label from this list.

        Labels should be added either when initiating Segment,
        or through Segment.addLabel.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self) != 5:
            print("ERROR: incorrect number of args provided to Segment (need 5, not %d)" % len(self))
            return
        if self[0]<0 or self[1]<0:
            print("ERROR: Segment times must be positive or 0")
            return
        if self[2]<0 or self[3]<0:
            print("ERROR: Segment frequencies must be positive or 0")
            return
        if not isinstance(self[4], list):
            print("ERROR: Segment labels must be a list")
            return

        # check if labels have the right structure
        for lab in self[4]:
            if not isinstance(lab, dict):
                print("ERROR: Segment label must be a dict")
                return
            if "species" not in lab or not isinstance(lab["species"], str):
                print("ERROR: species bad or missing from label")
                return
            if "certainty" not in lab or not isinstance(lab["certainty"], (int, float)):
                print("ERROR: certainty bad or missing from label")
                return
            if "filter" in lab and lab["filter"]!="M" and "calltype" not in lab:
                print("ERROR: calltype required when automated filter provided in label")
                return

        # fix types to avoid numpy types etc
        self[0] = float(self[0])
        self[1] = float(self[1])
        self[2] = int(self[2])
        self[3] = int(self[3])

        self.keys = [lab['species'] for lab in self[4]]
        if len(self.keys)>len(set(self.keys)):
            print("ERROR: non-unique species detected")
            return
    
    def hasLabel(self, species):
        """ Check if label identified by species is present in this segment. """
        return species in self.keys

    def addLabel(self, species, certainty, **label):
        """ Adds a label to this segment.
            Species and certainty are required and passed positionally.
            Any further label properties (filter, calltype...) must be passed as keyword args:
              addLabel("LSK", 100, filter="M"...)
        """
        if not isinstance(species, str):
            print("ERROR: bad species provided")
            return
        if not isinstance(certainty, (int, float)):
            print("ERROR: bad certainty provided")
            return
        if "filter" in label and label["filter"]!="M" and "calltype" not in label:
            print("ERROR: calltype required when automated filter provided in label")
            return
        if self.hasLabel(species):
            print("ERROR: this species label already present")
            return
        label["species"] = species
        label["certainty"] = certainty

        self[4].append(label)
        self.keys.append(species)
    
    def getKeys(self):
        return [lab['species'] for lab in self[4]]
    
    def getKeysWithCalltypes(self):
        return [(lab['species'], lab['calltype'] if 'calltype' in lab else None) for lab in self[4]]
    
    def getCalltype(self,species): # a species can only have 1 calltype in a segment
        for lab in self[4]:
            if lab["species"]==species:
                if 'calltype' in lab:
                    return lab['calltype']
        return None

    ### --- couple functions to process all labels for a given species ---

    def wipeSpecies(self, species):
        """ Remove all labels for species, return True if all labels were wiped
            (and the interface should delete the segment).
        """
        deletedAll = list(set([lab["species"] for lab in self[4]])) == [species]
        # note that removeLabel will re-add a Don't Know in the end, so can't just check the final label.
        for lab in reversed(self[4]):
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
        for labix in range(len(self[4])):
            lab = self[4][labix]
            # check if this label is yellow:
            if (species is None or lab["species"]==species) and lab["certainty"] < 100 and lab["species"]!="Don't Know":
                # check if this segment has a green label for this species already
                if (lab["species"], 100) in self.keys:
                    # then just delete the yellow label
                    toremove.append(lab)
                else:
                    lab["certainty"] = 100
        for trlab in toremove:
            self.removeLabel(trlab["species"])

    def questionLabels(self, species=None):
        """ Lower the certainty of this segment's certain labels to 50.
            Affects all species (if None) or indicated species.
            Ignores "Don't Know" labels.
            (Could be merged with the above at some point.)
            Returns True if it changed any labels.
        """
        anyChanged = False
        toremove = []
        for labix in range(len(self[4])):
            lab = self[4][labix]
            # check if this label is green:
            if (species is None or lab["species"]==species) and lab["certainty"]==100 and lab["species"]!="Don't Know":
                # check if this segment has a yellow label for this species already
                otherLabels = [k[0]==lab["species"] and k[1]<100 for k in self.keys]
                if any(otherLabels):
                    # then just delete this label
                    toremove.append(lab)
                else:
                    lab["certainty"] = 50
                anyChanged = True
        for trlab in toremove:
            self.removeLabel(trlab["species"])

        return(anyChanged)

    def removeLabel(self, species):
        """ Removes label from this segment.
            Does not delete the actual segment - that's left for the interface to take care of.
        """
        deleted = False
        for lab in self[4]:
            if lab["species"]==species:
                self[4].remove(lab)
                try:
                    self.keys.remove(species)
                except Exception as e:
                    text = "************ WARNING ************\n"
                    text += str(e)
                    text += "\nWhile trying to remove key "+str(species)+" from "+ str(self[4])
                    text += "\nWhich had keys" + str(self.keys)
                    import SupportClasses_GUI
                    msg = SupportClasses_GUI.MessagePopup("w", "ERROR - please report", text)
                    msg.exec()
                # if that was the last label, flip to Don't Know
                if len(self[4])==0:
                    self.addLabel("Don't Know", 0)
                deleted = True
                break

        if not deleted:
            print("ERROR: could not find species to remove:", species)
            return
    
    def clearLabels(self):
        self[4]=[]
        self.keys =[]

    def infoString(self):
        """ Returns a nicely-formatted string of this segment's info."""
        s = []
        for lab in self[4]:
            labs = "sp.: %s, cert.: %d%%" % (lab["species"], lab["certainty"])
            if "filter" in lab and lab["filter"]!="M":
                labs += ", filter: " + lab["filter"]
            if "calltype" in lab:
                labs += ", call: " + lab["calltype"]
            s.append(labs)
        return "; ".join(s)


class SegmentList(list):
    """ List of Segments. Deals with I/O - parsing JSON,
        and retrieving the right Segment from this list.
    """

    def parseJSON(self, file, duration=0, silent=False):
        """ Takes in a filename and reads metadata to self.metadata,
            and other segments to just the main body of self.
            If wav file is loaded, pass the true duration in s to check
            (it will override any duration read from the JSON).
        """
        try:
            f = open(file, 'r')
            annots = json.load(f)
            f.close()
        except Exception as e:
            print("ERROR: file %s failed to load with error:" % file)
            print(e)
            return

        hasmetadata = True
        # first segment stores metadata
        self.metadata = dict()
        # -----
        if isinstance(annots[0],list):
            print(annots[0])
        #if isinstance(annots[0], list) and annots[0][0] == -1:
        if isinstance(annots[0], list):
            if not silent:
                print("old format metadata detected")
            if annots[0][0] == -1:
                self.metadata = {"Operator": annots[0][2], "Reviewer": annots[0][3]}
                # when file is loaded, true duration can be passed. Otherwise,
                # some old files have duration in samples, so need a rough check
                if duration>0:
                    self.metadata["Duration"] = duration
                elif isinstance(annots[0][1], (int, float)) and annots[0][1]>0 and annots[0][1]<100000:
                    self.metadata["Duration"] = annots[0][1]
                else:
                    # fallback to reading the wav:
                    try:
                        info = sf.info(file[:-5])
                        sample_rate = info.samplerate
                        fileduration = info.frames / sample_rate
                        self.metadata["Duration"] = fileduration
                    except Exception as e:
                        print("ERROR: duration not found in metadata, arguments, or read from wav")
                        print(file)
                        print(e)
                        return
                # noise metadata
                if len(annots[0])<5 or not isinstance(annots[0][4], list):
                    self.metadata["noiseLevel"] = None
                    self.metadata["noiseTypes"] = []
                else:
                    self.metadata["noiseLevel"] = annots[0][4][0]
                    self.metadata["noiseTypes"] = annots[0][4][1]
                del annots[0]
            else:
                # Very old version
                try:
                    info = sf.info(file[:-5])
                    sample_rate = info.samplerate
                    fileduration = info.frames / sample_rate
                    self.metadata["Duration"] = fileduration
                except Exception as e:
                    print("ERROR: can't read duration from wav")
                    print(file)
                    print(e)
                    return
                self.metadata["Operator"] = ""
                self.metadata["Reviewer"] = ""
                hasmetadata = False
        elif isinstance(annots[0], dict):
            # New format has 3 required fields:
            self.metadata = annots[0]
            if duration>0:
                self.metadata["Duration"] = duration
            if "Operator" not in self.metadata or "Reviewer" not in self.metadata or "Duration" not in self.metadata:
                if "Duration" not in self.metadata:
                    try:
                        info = sf.info(file[:-5])
                        sample_rate = info.samplerate
                        fileduration = info.frames / sample_rate
                        self.metadata["Duration"] = fileduration
                    except Exception as e:
                        print("ERROR: can't read duration from wav")
                        print(file)
                        print(e)
                    return
                self.metadata["Operator"] = ""
                self.metadata["Reviewer"] = ""
                hasmetadata = False
                #print("ERROR: required metadata fields not found")
                #return
            del annots[0]

        # read the segments
        self.clear()
        for annot in annots:
            if not isinstance(annot, list) or len(annot)!=5:
                print("ERROR: annotation in wrong format:", annot)
                return

            # This could be turned on to skip segments outside Duration bounds,
            # but may result in deleting actually useful annotations if Duration was wrong
            # if annot[0] > self.metadata["Duration"] and annot[1] > self.metadata["Duration"]:
            #     print("Warning: ignoring segment outside set duration", annot)
            #     continue

            # deal with old formats here, so that the Segment class
            # could require (and validate) clean input

            # Early version of AviaNZ stored freqs as values between 0 and 1.
            # The .1 is to take care of rounding errors
            if 0 < annot[2] < 1.1 and 0 < annot[3] < 1.1:
                print("Warning: updating old-format frequency marks")
                info = sf.info(file[:-5])
                sample_rate = info.samplerate
                fileduration = info.frames / sample_rate
                annot[2] *= rate
                annot[3] *= rate

            # single string-type species labels
            if isinstance(annot[4], str):
                annot[4] = [annot[4]]
            # for list-type labels, parse each into certainty and species
            if isinstance(annot[4], list):
                listofdicts = []
                for lab in annot[4]:
                    # new format:
                    if isinstance(lab, dict):
                        labdict = lab
                    # old format parsing:
                    elif lab == "Don't Know":
                        labdict = {"species": "Don't Know", "certainty": 0}
                    elif lab.endswith('?'):
                        labdict = {"species": lab[:-1], "certainty": 50}
                    else:
                        labdict = {"species": lab, "certainty": 100}
                    listofdicts.append(labdict)
                # if no labels were present, i.e. "[]", addSegment will create a Don't Know
                annot[4] = listofdicts

            self.addSegment(annot)
        if not silent:
            print("%d segments read" % len(self))

        return hasmetadata

    def addSegment(self, segment):
        """ Just a cleaner wrapper to allow adding segments quicker.
            Passes a list "segment" to the Segment class.
        """
        # allows passing empty label list - creates "Don't Know" then
        if len(segment[4]) == 0:
            segment[4] = [{"species": "Don't Know", "certainty": 0}]
        self.append(Segment(segment))

    def addBasicSegments(self, seglist, freq=[0,0], **kwd):
        """ Allows to add bunch of basic segments from segmentation
            with identical species/certainty/freq values.
            seglist - list of 2-col segments [[t1, t2],prob]
            label is built from kwd.
            These will be converted to [[t1, t2, freq[0], freq[1], label], ...]
            and stored.
        """
        if not isinstance(freq, list) or freq[0]<0 or freq[1]<0:
            print("ERROR: cannot use frequencies", freq)
            return

        for seg in seglist:
            newseg = [seg[0][0], seg[0][1], freq[0], freq[1], [kwd]]
            self.addSegment(newseg)

    def getSpecies(self, species):
        """ Returns indices of all segments that have the indicated species in label. """
        out = []
        for segi in range(len(self)):
            # check each label in this segment:
            labs = self[segi][4]
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
            labs = self[segi][4]
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
        """ Returns 1 on succesful save."""
        if reviewer != "":
            self.metadata["Reviewer"] = reviewer
        annots = [self.metadata]
        for seg in self:
            annots.append(seg)

        file = open(file, 'w')
        json.dump(annots, file)
        file.write("\n")
        file.close()
        return 1

    def orderTime(self):
        """ Returns the order of segments in this list sorted by start time.
            Sorts itself using the order. Can then be used to sort any additional lists
            in matching order (graphics etc). """
        sttimes = [s[0] for s in self]
        sttimes = np.argsort(sttimes)
        self.sort(key=lambda s: s[0])

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
            if species is not None and species not in [lab["species"] for lab in seg[4]]:
                continue
            l = seg[1]-seg[0]
            if l > maxlen:
                n = int(np.ceil(l/maxlen))
                d = l/n
                # adjust current seg to be the first piece
                seg[1] = seg[0]+d
                for i in range(1,n):
                    end = min(l, d * (i+1))
                    segpiece = copy.deepcopy(seg)
                    segpiece[0] = seg[0] + d*i
                    segpiece[1] = seg[0] + end
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
                if self[segi][4]!=self[firstsegi][4]:
                    continue
                # for subsequent segs, see if this can be merged to the previous one
                if self[segi][0]<=self[firstsegi][1]:
                    self[firstsegi][1] = max(self[segi][1], self[firstsegi][1])
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
        fLow = np.min([seg[2] for seg in self])
        fHigh = np.max([seg[3] for seg in self])
        # TimeRange, in s
        lenMin = np.min([seg[1] - seg[0] for seg in self])
        lenMax = np.max([seg[1] - seg[0] for seg in self])

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
        duration = int(np.ceil(self.metadata["Duration"] / resolution))
        filenameNoExtension = filename.rsplit('.', 1)[0]
        eFile = filenameNoExtension + '-GT.txt'

        # deal with empty files
        thisSpSegs = self.getSpecies(species)
        # if len(thisSpSegs)==0:
        #     print("Warning: no annotations for this species found in file", filename)
        #     # delete the file to avoid problems with old GT files
        #     try:
        #         os.remove(eFile)
        #     except Exception:
        #         pass
        #     return

        GT = np.tile([0, 0, None], (duration,1))
        # fill first column with "time"
        GT[:,0] = range(1, duration+1)
        GT[:,0] = GT[:,0] * resolution

        print("exporting GT with resolution", resolution)

        for segix in thisSpSegs:
            seg = self[segix]
            # start and end in resolution base
            s = int(max(0, math.floor(seg[0] / resolution)))
            e = int(min(duration, math.ceil(seg[1] / resolution)))
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

class Segmenter:
    """ This class implements six forms of segmentation for the AviaNZ interface:
    Amplitude threshold (rubbish)
    Energy threshold
    Harma
    Median clipping of spectrogram
    Fundamental frequency using yin
    FIR

    It also computes ways to merge them

    Important parameters:
        mingap: the smallest space between two segments (otherwise merge them)
        minlength: the smallest size of a segment (otherwise delete it)
        ignoreInsideEnvelope: whether you keep the superset of a set of segments or the individuals when merging
        maxlength: the largest size of a segment (currently unused)
        threshold: generally this is of the form mean + threshold * std dev and provides a way to filter

    And two forms of recognition:
    Cross-correlation
    DTW

    Each returns start and stop times for each segment (in seconds) as a Python list of pairs.
    It is up to the caller to convert these to a true SegmentList.
    See also the species-specific segmentation in WaveletSegment
    """

    def __init__(self, sp=None, fs=0, mingap=0.3, minlength=0.2):
        # This is the reference to Spectrogram
        self.sp = sp
        self.fs = fs
        # Spectrogram
        if hasattr(sp, 'sg'):
            self.sg = sp.sg
        else:
            self.sg = None
        # These are the spectrogram params. Needed to compute times.
        if sp:
            self.data = sp.data
            self.fs = sp.audioFormat.sampleRate()
            #self.fs = sp.sampleRate
            self.window_width = sp.window_width
            self.incr = sp.incr
        self.mingap = mingap
        self.minlength = minlength

    def setNewData(self, sp):
        # To be called when a new sound file is loaded
        self.sp = sp
        self.data = sp.data
        self.fs = sp.audioFormat.sampleRate()
        #self.fs = sp.sampleRate
        self.sg = sp.sg
        self.window_width = sp.window_width
        self.incr = sp.incr

    def bestSegments(self,FIRthr=0.7, medianClipthr=3.0, yinthr=0.9):
        """ A reasonably good segmentaion - a merged version of FIR, median clipping, and fundamental frequency using yin
        """
        segs1 = self.segmentByFIR(FIRthr)
        segs2 = self.medianClip(medianClipthr)
        segs3 = self.yinSegs(100, thr=yinthr)
        segs1 = self.mergeSegments(segs1, segs2)
        segs = self.mergeSegments(segs1, segs3)
        # mergeSegments also sorts, so not needed:
        # segs = sorted(segs, key=lambda x: x[0])
        return segs

    def mergeSegments(self, segs1, segs2=None):
        """ Given two segmentations of the same file, join them,
        and if one wasn't empty, merge any overlapping segments.
        format: [[1,3] [2,4] [5,7] [7,8]] -> [[1,4] [5,7] [7,8]]
        Can take in one or two lists. """
        if segs1 == [] and segs2 == []:
            return []
        elif segs1 == []:
            return segs2
        elif segs2 == []:
            return segs1

        if segs2 is not None:
            segs1.extend(segs2)
        out = self.checkSegmentOverlap(segs1)
        return out


    def convert01(self, presabs, window=1):
        """ Turns a list of presence/absence [0 1 1 1]
            into a list of start-end segments [[1,4]].
            Can use non-1 s units of pres/abs.
        """
        # squeeze any extra axes except axis 0 (don't make scalars)
        presabs = np.reshape(presabs, (np.shape(presabs)[0]))
        out = []
        t = 0
        while t < len(presabs):
            if presabs[t]==1:
                start = t
                while t<len(presabs) and presabs[t]!=0:
                    t += 1
                out.append([start*window, t*window])
            t += 1
        return out

    def deleteShort(self, segments, minlength=0.25):
        """ Checks whether segments are long enough.
            Operates on start-end list:
            [[1,3], [4,5]] -> [[1,3]].
        """
        out = []
        if minlength == 0:
            minlength = self.minlength
        for seg in segments:
            if seg[1]-seg[0] >= minlength:
                out.append(seg)
        return out

    def deleteShort3(self, segments, minlength=0.25):
        """ Checks whether segments are long enough.
            Operates on start-end list with probs:
            [[[1,3], 50], [[4,5], 90]] -> [[[1,3], 50]].
        """
        out = []
        if minlength == 0:
            minlength = self.minlength
        for seg in segments:
            if seg[0][1]-seg[0][0] >= minlength:
                out.append(seg)
        return out

    def splitLong3(self, segments, maxlen=10):
        """
        Splits long segments (> maxlen) evenly
        Operates on list of 3-element segments:
        [[[1,5], 50]] -> [[[1,3], 50], [[3,5], 50]]
        """
        out = []
        for seg in segments:
            l = seg[0][1]-seg[0][0]
            if l > maxlen:
                n = int(np.ceil(l/maxlen))
                d = l/n
                for i in range(n):
                    end = min(l, d * (i+1))
                    out.append([[seg[0][0] + d*i, seg[0][0] + end], seg[1]])
            else:
                out.append(seg)
        return out

    ##  MERGING ALGORITHMS: do the same with very small settings variations
    def checkSegmentOverlap(self, segments):
        """ Merges overlapping segments.
            Does not merge if the segments only touch.
            Operates on start-end list [[1,3], [2,4]] -> [[1,4]]
        """
        # Needs to be python array, not np array
        # Sort by increasing start times
        if isinstance(segments, np.ndarray):
            segments = segments.tolist()
        segments = sorted(segments)
        segments = np.array(segments)

        # Loop over segs until the start value of next segment
        #  is not inside the end value of the previous
        out = []
        i = 0
        while i < len(segments):
            start = segments[i][0]
            end = segments[i][1]
            while i+1 < len(segments) and segments[i+1][0]-end < 0:
                # there is overlap, so merge
                i += 1
                end = max(end, segments[i][1])
            # no more overlap, so store the current:
            out.append([start, end])
            i += 1
        return out

    def joinGaps(self, segments, maxgap=3):
        """ Merges segments within maxgap units.
            Identical to above, except merges touching segments, and allows a gap.
            Operates on start-end list [[1,2], [3,4]] -> [[1,4]]
        """
        if isinstance(segments, np.ndarray):
            segments = segments.tolist()
        if len(segments)==0:
            return []

        segments.sort(key=lambda seg: seg[0])

        out = []
        i = 0
        while i < len(segments):
            start = segments[i][0]
            end = segments[i][1]
            while i+1 < len(segments) and segments[i+1][0]-end <= maxgap:
                i += 1
                end = max(end, segments[i][1])
            out.append([start, end])
            i += 1
        return out

    def checkSegmentOverlap3(self, segments):
        """ Merges overlapping segments.
            Does not merge if the segments only touch.
            Identical to above, but operates on list of 3-element segments:
            [[[1,3], 50], [[2,5], 70]] -> [[[1,5], 60]]
            Currently, certainties are just averaged over the number of segs.
        """
        if isinstance(segments, np.ndarray):
            segments = segments.tolist()
        if len(segments)==0:
            return []

        segments.sort(key=lambda seg: seg[0][0])
        # sorted() appears to work fine as well

        out = []
        i = 0
        while i < len(segments):
            start = segments[i][0][0]
            end = segments[i][0][1]
            cert = [segments[i][1]]
            while i+1 < len(segments) and segments[i+1][0][0]-end < 0:
                i += 1
                end = max(end, segments[i][0][1])
                cert.append(segments[i][1])
            out.append([[start, end], np.mean(cert)])
            i += 1
        return out

    def joinGaps3(self, segments, maxgap=3):
        """ Merges segments within maxgap units.
            Operates on list of 3-element segments:
            [[[1,2], 50], [[3,5], 70]] -> [[[1,5], 60]]
            Identical to above, except merges touching segments, and allows a gap.
            Currently, certainties are just averaged over the number of segs.
        """
        if isinstance(segments, np.ndarray):
            segments = segments.tolist()
        if len(segments)==0:
            return []

        segments.sort(key=lambda seg: seg[0][0])

        out = []
        i = 0
        while i < len(segments):
            start = segments[i][0][0]
            end = segments[i][0][1]
            cert = [segments[i][1]]
            while i+1 < len(segments) and segments[i+1][0][0]-end <= maxgap:
                i += 1
                end = max(end, segments[i][0][1])
                cert.append(segments[i][1])
            out.append([[start, end], np.mean(cert)])
            i += 1
        return out

    def segmentByFIR(self, threshold):
        """ Segmentation using FIR envelope.
        """
        nsecs = len(self.data) / float(self.fs)
        fftrate = int(np.shape(self.sg)[0]) / nsecs
        upperlimit = 100
        FIR = [0.078573000000000004, 0.053921000000000004, 0.041607999999999999, 0.036006000000000003, 0.031521,
               0.029435000000000003, 0.028122000000000001, 0.027286999999999999, 0.026241000000000004,
               0.025225999999999998, 0.024076, 0.022926999999999999, 0.021703999999999998, 0.020487000000000002,
               0.019721000000000002, 0.019015000000000001, 0.018563999999999997, 0.017953, 0.01753,
               0.017077000000000002, 0.016544, 0.015762000000000002, 0.015056, 0.014456999999999999, 0.013913,
               0.013299, 0.012879, 0.012568000000000001, 0.012454999999999999, 0.012056000000000001, 0.011634,
               0.011077, 0.010707, 0.010217, 0.0098840000000000004, 0.0095959999999999986, 0.0093607000000000013,
               0.0090197999999999997, 0.0086908999999999997, 0.0083841000000000002, 0.0081481999999999995,
               0.0079185000000000002, 0.0076363000000000004, 0.0073406000000000009, 0.0070686999999999998,
               0.0068438999999999991, 0.0065873000000000008, 0.0063688999999999994, 0.0061700000000000001,
               0.0059743000000000001, 0.0057561999999999995, 0.0055351000000000003, 0.0053633999999999991,
               0.0051801, 0.0049743000000000001, 0.0047431000000000001, 0.0045648999999999993,
               0.0043972000000000004, 0.0042459999999999998, 0.0041016000000000004, 0.0039503000000000003,
               0.0038013000000000005, 0.0036351, 0.0034856000000000002, 0.0033270999999999999,
               0.0032066999999999998, 0.0030569999999999998, 0.0029206999999999996, 0.0027760000000000003,
               0.0026561999999999996, 0.0025301999999999998, 0.0024185000000000001, 0.0022967,
               0.0021860999999999998, 0.0020696999999999998, 0.0019551999999999998, 0.0018563,
               0.0017562000000000001, 0.0016605000000000001, 0.0015522000000000001, 0.0014482999999999998,
               0.0013492000000000001, 0.0012600000000000001, 0.0011788, 0.0010909000000000001, 0.0010049,
               0.00091527999999999998, 0.00082061999999999999, 0.00074465000000000002, 0.00067159000000000001,
               0.00060258999999999996, 0.00053370999999999996, 0.00046135000000000002, 0.00039071,
               0.00032736000000000001, 0.00026183000000000001, 0.00018987999999999999, 0.00011976000000000001,
               6.0781000000000006e-05, 0.0]
        f = interp1d(np.arange(0, len(FIR)), np.squeeze(FIR))
        samples = f(np.arange(1, upperlimit, float(upperlimit) / int(fftrate / 10.)))
        padded = np.concatenate((np.zeros(int(fftrate / 10.)), np.mean(self.sg, axis=1), np.zeros(int(fftrate / 10.))))
        envelope = spi.filters.convolve(padded, samples, mode='constant')[:-int(fftrate / 10.)]
        ind = envelope > np.median(envelope) + threshold * np.std(envelope)
        segs = self.convert01(ind, self.incr / self.fs)
        return segs

    def segmentByAmplitude(self, threshold, usePercent=True):
        """ Bog standard amplitude segmentation.
        A straw man, do not use.
        """
        if usePercent:
            threshold = threshold*np.max(self.data)
        seg = np.abs(self.data)>threshold
        seg = self.convert01(seg, self.fs)
        return seg

    def segmentByEnergy(self, thr, width, min_width=450):
        """ Based on description in Jinnai et al. 2012 paper in Acoustics
        Computes the 'energy curve' as windowed sum of absolute values of amplitude
        I median filter it, 'cos it's very noisy
        And then threshold it (no info on their threshold) and find max in each bit above threshold
        I also check for width of those (they don't say anything)
        They then return the max-width:max+width segments for each max
        """
        data = np.abs(self.data)
        E = np.zeros(len(data))
        E[width] = np.sum(data[:2*width+1])
        for i in range(width+1,len(data)-width):
            E[i] = E[i-1] - data[i-width-1] + data[i+width]
        E = E/(2*width)

        # TODO: Automatic energy gain (normalisation method)

        # This thing is noisy, so I'm going to median filter it. SoundID doesn't seem to?
        Em = np.zeros(len(data))
        for i in range(width,len(data)-width):
            Em[i] = np.median(E[i-width:i+width])
        for i in range(width):
            Em[i] = np.median(E[0:2*i])
            Em[-i] = np.median(E[-2 * i:])

        # TODO: Better way to do this?
        threshold = np.mean(Em) + thr*np.std(Em)

        # Pick out the regions above threshold and the argmax of each, assuming they are wide enough
        starts = []
        ends = []
        insegment = False
        for i in range(len(data)-1):
            if not insegment:
                if Em[i]<threshold and Em[i+1]>threshold:
                    starts.append(i)
                    insegment = True
            if insegment:
                if Em[i]>threshold and Em[i+1]<threshold:
                    ends.append(i)
                    insegment = False
        if insegment:
            ends.append(len(data))
        maxpoints = []
        Emm = np.zeros(len(data))
        for i in range(len(starts)):
            if ends[i] - starts[i] > min_width:
                maxpoints.append(np.argmax(Em[starts[i]:ends[i]]))
                Emm[starts[i]:ends[i]] = Em[starts[i]:ends[i]]

        # TODO: SoundID appears to now compute the 44 LPC coeffs for each [midpoint-width:midpoint+width]
        # TODO: And then compute the geometric distance to templates

        segs = []
        for i in range(len(starts)):
            segs.append([float(starts[i])/self.fs,float(ends[i])/self.fs])
        return segs

    def Harma(self, thr=10., stop_thr=0.8, minSegment=50):
        """ Harma's method, but with a different stopping criterion
        # Assumes that spectrogram is not normalised
        maxFreqs = 10. * np.log10(np.max(self.sg, axis = 1))
        """
        maxFreqs = 10. * np.log10(np.max(self.sg, axis=1))
        maxFreqs = medfilt(maxFreqs,21)
        biggest = np.max(maxFreqs)
        segs = []

        while np.max(maxFreqs)>stop_thr*biggest:
            t0 = np.argmax(maxFreqs)
            a_n = maxFreqs[t0]

            # Go backwards looking for where the syllable stops
            t = t0
            while maxFreqs[t] > a_n - thr and t>0:
                t -= 1
            t_start = t

            # And forwards
            t = t0
            while maxFreqs[t] > a_n - thr and t<len(maxFreqs)-1:
                t += 1
            t_end = t

            # Set the syllable just found to 0
            maxFreqs[t_start:t_end] = 0
            if float(t_end - t_start)*self.incr/self.fs*1000.0 > minSegment:
                segs.append([float(t_start)* self.incr / self.fs,float(t_end)* self.incr / self.fs])

        return segs

    def segmentByPower(self, thr=1.):
        """ Segmentation simply on the power
        """
        maxFreqs = 10. * np.log10(np.max(self.sg, axis=1))
        maxFreqs = medfilt(maxFreqs, 21)
        ind = maxFreqs > (np.mean(maxFreqs)+thr*np.std(maxFreqs))
        segs = self.convert01(ind, self.incr / self.fs)
        return segs

    def medianClip(self, thr=3.0, medfiltersize=5, minaxislength=5, minSegment=70):
        """ Median clipping for segmentation
        Based on Lasseck's method
        minaxislength - min "length of the minor axis of the ellipse that has the same normalized second central moments as the region", based on skm.
        minSegment - min number of pixels exceeding thr to declare an area as segment.
        This version only clips in time, ignoring frequency
        And it opens up the segments to be maximal (so assumes no overlap).
        The multitaper spectrogram helps a lot

        """
        #tt = time.time()
        sg = self.sg/np.max(self.sg)

        # This next line gives an exact match to Lasseck, but screws up bitterns!
        #sg = sg[4:232, :]

        rowmedians = np.median(sg, axis=1)
        colmedians = np.median(sg, axis=0)

        clipped = np.zeros(np.shape(sg),dtype=int)
        for i in range(np.shape(sg)[0]):
            for j in range(np.shape(sg)[1]):
                if (sg[i, j] > thr * rowmedians[i]) and (sg[i, j] > thr * colmedians[j]):
                    clipped[i, j] = 1
        print("Found", np.sum(clipped), "pixels")

        # This is the stencil for the closing and dilation. It's a 5x5 diamond. Can also use a 3x3 diamond
        diamond = np.zeros((5,5),dtype=int)
        diamond[2,:] = 1
        diamond[:,2] = 1
        diamond[1,1] = diamond[1,3] = diamond[3,1] = diamond[3,3] = 1
        #diamond[2, 1:4] = 1
        #diamond[1:4, 2] = 1

        clipped = spi.binary_closing(clipped,structure=diamond).astype(int)
        clipped = spi.binary_dilation(clipped,structure=diamond).astype(int)
        clipped = spi.median_filter(clipped,size=medfiltersize)
        clipped = spi.binary_fill_holes(clipped)

        blobs = skm.regionprops(skm.label(clipped.astype(int)))

        # Delete blobs that are too small
        keep = []
        for i in range(len(blobs)):
            if blobs[i].filled_area > minSegment and blobs[i].minor_axis_length > minaxislength:
                keep.append(i)

        out = []
        blobs = [blobs[i] for i in keep]

        # convert bounding box pixels to milliseconds:
        for l in blobs:
            out.append([float(l.bbox[0] * self.incr / self.fs), float(l.bbox[2] * self.incr / self.fs)])
        return out

    def checkSegmentOverlapCentroids(self, blobs, minSegment=50):
        # Delete overlapping boxes by computing the centroids and picking out overlaps
        # Could also look at width and so just merge boxes that are about the same size
        # Note: no mingap parameter is used right now
        centroids = []
        for i in blobs:
            centroids.append((i[1] - i[0])/2)
        centroids = np.array(centroids)
        ind = np.argsort(centroids)
        centroids = centroids[ind]
        blobs = np.array(blobs)[ind]

        current = 0
        centroid = centroids[0]
        count = 0
        list = []
        list.append(blobs[0])
        for i in centroids:
            # TODO: replace this with simple overlap?
            if (i - centroid)*1000 < minSegment / 2. * 10:
                if blobs[ind[count]][0] < list[current][0]:
                    list[current][0] = blobs[ind[count]][0]
                if blobs[ind[count]][1] > list[current][1]:
                    list[current][1] = blobs[ind[count]][1]
            else:
                current += 1
                centroid = centroids[count]
                list.append([blobs[ind[count]][0], blobs[ind[count]][1]])
            count += 1

        segments = []
        for i in list:
            if float(i[1] - i[0])*1000 > minSegment:
                segments.append([i[0], i[1]])
        return segments

    def onsets(self, thr=3.0):
        """ Segmentation using the onset times from librosa.
        There are no offset times -- compute an energy drop?
        A straw man really.
        """
        o_env = librosa.onset.onset_strength(self.data, sr=self.fs, aggregate=np.median)
        cutoff = np.mean(o_env) + thr * np.std(o_env)
        o_env = np.where(o_env > cutoff, o_env, 0)
        onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=self.fs)
        times = librosa.frames_to_time(np.arange(len(o_env)), sr=self.fs)

        segments = []
        for i in range(len(onsets)):
            segments.append([times[onsets[i]],times[onsets[i]]+0.2])
        return segments

    def yinSegs(self, minfreq=100, minperiods=3, thr=0.5, W=1000):
        """ Segmentation by computing the fundamental frequency.
            Uses the Yin algorithm of de Cheveigne and Kawahara (2002).
            Args:
            minfreq: lowest freq (Hz) to consider as plausible.
            thr: the threshold for accepting F0,
              necessarily higher than the 0.1 in the paper.
            W: the window in samples used.
        """
        # Window width W should be at least 3*period.
        # A sample rate of 16000 and a min fundamental frequency of 100Hz would then therefore suggest reasonably short windows
        minwin = float(self.fs) / minfreq * minperiods
        if W < minwin:
            print("Extending window width to ", minwin)
            W = int(minwin)

        # returns pitch in Hz for each window of Wsamples/2.
        # As this uses the full self.data, it is up to caller to adjust times
        # to real seconds if self.data only contained e.g. a segment
        shape = Shapes.fundFreqShaper(self.data, W, thr, self.fs)

        pitch = shape.y
        if len(pitch)==0:
            return np.array([])
        units = shape.tunit

        # drop any pitch under minfreq
        ind = pitch > minfreq
        segs = self.convert01(ind, units)
        return segs

    def findCCMatches(self, seg, sg, thr):
        """ Cross-correlation. Takes a segment and looks for others that match it to within thr.
        match_template computes fast normalised cross-correlation
        """
        from skimage.feature import match_template

        # seg and sg have the same $y$ size, so the result of match_template is 1D
        #m = match_template(sg,seg)
        matches = np.squeeze(match_template(sg, seg))

        import peakutils
        md = np.shape(seg)[0]/2
        threshold = thr*np.max(matches)
        indices = peakutils.indexes(matches, thres=threshold, min_dist=md)
        return indices

    def findDTWMatches(self, seg, data):
        # TODO: This is slow and crap. Note all the same length, for a start, and the fact that it takes forever!
        # Use MFCC first?
        d = np.zeros(len(data))
        for i in range(len(data)):
            d[i] = self.dtw(seg, data[i:i+len(seg)])
        return d

    def dtw(self, x, y, wantDistMatrix=False):
        # Compute the dynamic time warp between two 1D arrays
        dist = np.zeros((len(x)+1,len(y)+1))
        dist[1:, :] = np.inf
        dist[:, 1:] = np.inf
        for i in range(len(x)):
            for j in range(len(y)):
                dist[i+1, j+1] = np.abs(x[i]-y[j]) + min(dist[i, j+1], dist[i+1, j], dist[i, j])
        if wantDistMatrix:
            return dist
        else:
            return dist[-1, -1]

    def dtw_path(self, d):
        # Shortest path through DTW matrix
        i = np.shape(d)[0]-2
        j = np.shape(d)[1]-2
        xpath = [i]
        ypath = [j]
        while i>0 or j>0:
                next = np.argmin((d[i,j],d[i+1,j],d[i,j+1]))
                if next == 0:
                    i -= 1
                    j -= 1
                elif next == 1:
                    j -= 1
                else:
                    i -= 1
                xpath.insert(0,i)
                ypath.insert(0,j)
        return xpath, ypath

    # def testDTW(self):
    #     x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
    #     y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
    #
    #     d = self.dtw(x,y,wantDistMatrix=True)
    #     print self.dtw_path(d)


class PostProcess:
    """ This class implements few post processing methods basically to avoid false positives.
    Operates on detections from a single subfilter.

    segments:   wavelet filter detections in format [[s1,e1], [s2,e2],...]
        Will be converted to a list of [[s1, e1], prob] upon load,
        and subsequent functions deal with certainty values.
    subfilter:  AviaNZ format subfilter
    cert:       Default certainty to attach to the segments
    """

    def __init__(self, configdir, audioData=None, sampleRate=0, tgtsampleRate=0, segments=[], subfilter={}, NNmodel=None, cert=0):
        self.configdir = configdir
        self.audioData = audioData
        self.sampleRate = sampleRate
        self.subfilter = subfilter

        # Convert to [[s1, e1], cert]
        self.segments = []
        for seg in segments:
            if len(seg) != 2:
                continue
            if seg[0]<0 or seg[1]<0:
                continue
            self.segments.append([seg, cert])

        if NNmodel:
            cl = SupportClasses.ConfigLoader()
            self.LearningDict = cl.learningParams(os.path.join(configdir, "LearningParams.txt"))

            self.NNmodel = NNmodel[0]    # NNmodel is a list [model, win, inputdim, outputdict, windowInc, thrs]
            self.NNwindow = NNmodel[1][0]  # size of each frame
            # self.NNhop = NNmodel[1][1]
            self.NNhop = self.LearningDict['hopScaling']*self.NNwindow
            self.NNinputdim = NNmodel[2]
            self.NNoutputs = NNmodel[3]
            self.NNwindowInc = NNmodel[4]  # [window,incr] for making the spec
            self.NNthrs = NNmodel[5]
            if NNmodel[6]:
                self.NNfRange = NNmodel[7]
            else:
                self.NNfRange = None
            self.tgtsampleRate = tgtsampleRate
        else:
            self.NNmodel = None

        if subfilter != {}:
            self.minLen = subfilter['TimeRange'][0]
            self.maxLen = subfilter['TimeRange'][1]
            if 'F0Range' in subfilter:
                self.F0 = subfilter['F0Range']
            self.fLow = subfilter['FreqRange'][0]
            self.fHigh = subfilter['FreqRange'][1]
            self.minLen = subfilter['TimeRange'][0]
            self.calltype = subfilter['calltype']
            self.syllen = subfilter['TimeRange'][2]
        else:
            self.minLen = 0.25
            self.fLow = 0
            self.fHigh = 0

    def getCertainty(self, meanprob, ctkey):
        '''
        Calculate the certainty of a segment.
        :param meanprob: best n mean probabilities
        :param ctkey: current call type key
        :return:
        '''
        if meanprob[ctkey] >= self.NNthrs[ctkey][-1]:
            certainty = 90
        elif meanprob[ctkey] >= self.NNthrs[ctkey][0]:
            certainty = 50
        else:
            certainty = 0  # TODO: set certainty to 20, when AviaNZ interface is ready to hide uncertain segments?

        return certainty

    def NN(self):
        """
        Post-proc with NN model, self.segments get updated
        """
        if not self.NNmodel:
            print("ERROR: no NN model specified")
            return
        if len(self.segments) == 0:
            print("No segments to classify by NN")
            return
        ctkey = int(list(self.NNoutputs.keys())[list(self.NNoutputs.values()).index(self.calltype)])
        print('call type: ', self.calltype)

        batchsize = 5   # TODO: read from learning parameters file

        for ix in reversed(range(len(self.segments))):
            seg = self.segments[ix]
            print('\n--- Segment', seg)
            # expand the segment if it's smaller than 1 frame
            mincalllength = self.NNwindow
            duration = seg[0][1] - seg[0][0]
            if mincalllength >= duration:
                extend_by = (mincalllength-duration)/2 + 0.005
                seg[0][0] -= extend_by
                seg[0][1] += extend_by
                if seg[0][0] < 0:
                    seg[0][0] = 0
                    seg[0][1] = mincalllength + 0.01
                elif seg[0][1]*self.sampleRate > len(self.audioData):
                    seg[0][0] = len(self.audioData)/self.sampleRate - mincalllength - 0.01
                    seg[0][1] = len(self.audioData)/self.sampleRate
                duration = seg[0][1] - seg[0][0]

            # Extract the audiodata corresponding to the segment
            data = self.audioData[int(seg[0][0] * self.sampleRate):int(seg[0][1] * self.sampleRate)]

            # Generate features for NN, overlapped windows
            sp = Spectrogram.Spectrogram(window_width=self.NNwindowInc[0],
                                        incr=self.NNwindowInc[1])
            sp.data = data
            sp.audioFormat.setSampleRate(self.sampleRate)
            if self.sampleRate != self.tgtsampleRate:
                sp.resample(self.tgtsampleRate)

            featuress = sp.generateFeaturesNN(seglen=duration, real_spec_width=self.NNinputdim[1], frame_size=self.NNwindow, frame_hop=self.NNhop, NNfRange=self.NNfRange)
            featuress = featuress.astype('float32')

            # assert shape
            if featuress.shape != (featuress.shape[0], self.NNinputdim[0], self.NNinputdim[1], 1):
                print("ERROR: features shape incorrect", featuress.shape)
                raise AssertionError
            numframes = featuress.shape[0]

            # predict with NN
            if numframes > 0:
                # probs = self.NNmodel(tf.convert_to_tensor(featuress, dtype=tf.float32))  # This might lead to OOM error, therefore show batches
                probs = np.empty((numframes, len(self.NNoutputs)))
                for start in range(0, numframes, batchsize):
                    end = min(numframes, start + batchsize)
                    p = self.NNmodel(tf.convert_to_tensor(featuress[start:end, :, :, :], dtype=tf.float32))
                    probs[start:end, :] = p

                # convert probs to certainties for each frame
                if self.activelength(probs[:, ctkey], self.NNthrs[ctkey][-1]) >= self.subfilter['TimeRange'][0]:
                    certainty = 90
                elif self.activelength(probs[:, ctkey], self.NNthrs[ctkey][0]) >= self.subfilter['TimeRange'][0]:
                    certainty = 50
                else:
                    certainty = 0
            else:
                # Zero images from this segment, very unlikely to be a true seg.
                probs = 0
                certainty = 0
            print("probabilities: ", probs)

            if certainty == 0:
                print('Deleted by NN')
                del self.segments[ix]
            else:
                print('Not deleted by NN')
                self.segments[ix][-1] = certainty

        print("Segments remaining after NN: ", len(self.segments))

    def activelength(self, probs, thr):
        """
        Returns the max length (secs) above thr given the probabilities of the images (overlapped)
        """
        binaryout = np.asarray(probs>=thr, dtype=int)
        segmenter = Segmenter()
        subsegs = segmenter.convert01(binaryout)
        lengths = [seg[1]-seg[0] for seg in subsegs]
        if lengths:
            return max(lengths)*self.NNhop
        else:
            return 0

    def NNDiagnostic(self):
        """
        Return the raw probabilities
        """
        if not self.NNmodel:
            print("ERROR: no NN model specified")
            return
        if len(self.segments)==0:
            print("No segments to classify by NN")
            return

        self.NNhop = self.NNwindow
        for ix in reversed(range(len(self.segments))):
            seg = self.segments[ix]

            if self.NNwindow >= seg[0][1] - seg[0][0]:
                print('Current page is smaller than NN input (%f)' % (self.NNwindow))
            else:
                # data = self.audioData[int(seg[0][0]*self.sampleRate):int(seg[0][1]*self.sampleRate)]
                data = self.audioData
            # generate features for NN
            sp = Spectrogram.Spectrogram(window_width=self.NNwindowInc[0],
                                        incr=self.NNwindowInc[1])
            sp.data = data
            sp.audioFormat.setSampleRate(self.sampleRate)
            if self.sampleRate != self.tgtsampleRate:
                sp.resample(self.tgtsampleRate)

            # frame_hop can be set to self.NNhop for overlap
            featuress = sp.generateFeaturesNN(seglen=seg[0][1]-seg[0][0], real_spec_width=self.NNinputdim[1], frame_size=self.NNwindow, frame_hop=None, NNfRange=self.NNfRange)
            # or multichannel:
            # featuress = sp.generateFeaturesNN2(seglen=seg[0][1]-seg[0][0], real_spec_width=self.NNinputdim[1], frame_size=self.NNwindow, frame_hop=None)
            featuress = featuress.astype('float32')
            # predict with NN
            if np.shape(featuress)[0] > 0:
                probs = self.NNmodel.predict(featuress)
            else:
                probs = 0
        return self.NNwindow, probs

    def wind_cal(self, data, sampleRate, fn_peak=0.35):
        """ Calculate wind
        Adopted from Automatic Identification of Rainfall in Acoustic Recordings by Carol Bedoya et al.
        :param data: audio data
        :param sampleRate: sample rate
        :param fn_peak: min height of a peak to be considered it as a significant peak
        :return: mean and std of wind, a binary indicator of false negative
        """
        wind_lower = 2.0 * 50 / sampleRate
        wind_upper = 2.0 * 500 / sampleRate
        f, p = signal.welch(data, fs=sampleRate, window='hamming', nperseg=512, detrend=False)
        p = np.log10(p)

        limite_inf = int(round(p.__len__() * wind_lower))
        limite_sup = int(round(
            p.__len__() * wind_upper))
        a_wind = p[limite_inf:limite_sup]  # section of interest of the power spectral density

        fn = False  # is this a false negative?
        if self.fLow > 500 or self.fLow == 0:   # fLow==0 when non-species-specific
            # Check the presence/absence of the target species/call type > 500 Hz.
            ind = np.abs(f - 500).argmin()
            if self.fLow == 0:
                ind_fLow = ind
            else:
                ind_fLow = np.abs(f - self.fLow).argmin() - ind
            if self.fHigh == 0:
                ind_fHigh = len(f) - 1
            else:
                ind_fHigh = np.abs(f - self.fHigh).argmin() - ind
            p = p[ind:]

            peaks, _ = signal.find_peaks(p)
            peaks = [i for i in peaks if (ind_fLow <= i <= ind_fHigh)]
            prominences = signal.peak_prominences(p, peaks)[0]
            # If there is at least one significant prominence in the target frequency band, then it could be a FN
            if len(prominences) > 0 and np.max(prominences) > fn_peak:
                fn = True

        return np.mean(a_wind), np.std(a_wind), fn    # mean of the PSD in the frequency band of interest.Upper part of
                                                      # the step 3 in Algorithm 2.1

    def wind(self, windT=2.5, fn_peak=0.35):
        """
        Delete wind corrupted segments, mainly wind gust
        :param windT: wind threshold
        :param fn_peak: min height of a peak to be considered it as a significant peak
        :return: self.segments get updated
        """
        if len(self.segments) == 0:
            print("No segments to remove wind from")
            return

        newSegments = copy.deepcopy(self.segments)
        for seg in self.segments:
            data = self.audioData[int(seg[0][0]*self.sampleRate):int(seg[0][1]*self.sampleRate)]
            ind = np.flatnonzero(data).tolist()     # eliminate impulse masked sections
            data = np.asarray(data)[ind].tolist()
            if len(data) == 0:
                continue
            m, _, fn = self.wind_cal(data=data, sampleRate=self.sampleRate, fn_peak=fn_peak)
            if m > windT and not fn:
                print(seg[0], m, 'windy, deleted')
                newSegments.remove(seg)
            elif m > windT and fn:
                print(seg[0], m, 'windy, but possible bird call')
            else:
                print(seg[0], m, 'not windy/possible bird call')
        self.segments = newSegments
        print("Segments remaining after wind: ", len(self.segments))

    def rainClick(self):
        """
        delete random clicks e.g. rain.
        """
        # TODO
        return
        newSegments = copy.deepcopy(self.segments)
        if newSegments.__len__() > 1:
            # Get avg energy
            sp = Spectrogram.Spectrogram()
            sp.data = self.audioData
            sp.audioFormat.setSampleRate(self.sampleRate)
            rawsg = sp.spectrogram()
            # Normalise
            rawsg -= np.mean(rawsg, axis=0)
            rawsg /= np.max(np.abs(rawsg), axis=0)
            mean = np.mean(rawsg)
            std = np.std(rawsg)
            thr = mean - 2 * std  # thr for the recording

            for seg in self.segments:
                if seg[0] == -1:
                    continue
                else:
                    data = self.audioData[int(seg[0]*self.sampleRate):int(seg[1]*self.sampleRate)]
                    rawsg = sp.spectrogram(data)
                    # Normalise
                    rawsg -= np.mean(rawsg, axis=0)
                    rawsg /= np.max(np.abs(rawsg), axis=0)
                    if np.min(rawsg) < thr:
                        newSegments.remove(seg)
        self.segments = newSegments

    def fundamentalFrq(self, fileName=None):
        '''
        Check for fundamental frequency of the segments, discard the segments that do not indicate the species.
        '''
        # F0 detection parameters:
        Wsamples = 1024
        minfreq = 100
        thr = 0.5

        for segix in reversed(range(len(self.segments))):
            seg = self.segments[segix][0]

            # read the sound segment and check fundamental frq.
            secs = int(seg[1] - seg[0])
            # Got to read from the source instead of using self.audioData - ff is wrong if you use self.audioData somehow
            sp = Spectrogram.Spectrogram(256, 128)
            if fileName:
                sp.readSoundFile(fileName, secs, seg[0])
                self.sampleRate = sp.audioFormat.sampleRate()
                self.audioData = sp.data
            else:
                sp.data = self.audioData
                sp.audioFormat.setSampleRate(self.sampleRate)

            # TODO: could denoise before fundamental frq. extraction

            # Ensure window width W is at least 3*period:
            # (generally fine for 100 Hz minfreq = 0.3 s minwin)
            minwin = 3 * sp.audioFormat.sampleRate() / minfreq
            if Wsamples < minwin:
                print("Extending window width to ", minwin)
                Wsamples = int(minwin)

            # returns pitch in Hz for each window of Wsamples/2.
            pitch = Shapes.fundFreqShaper(sp.data, Wsamples, thr, sp.audioFormat.sampleRate())
            pitch = pitch.y
            ind = np.squeeze(np.where(pitch > minfreq))
            pitch = pitch[ind]

            if pitch.size == 0:
                print('Segment ', seg, ' *++ no fundamental freq detected, could be faded call or noise')
                del self.segments[segix]
            else:
                meanF0 = np.mean(pitch)
                if (meanF0 < self.F0[0]) or (meanF0 > self.F0[1]):
                    print('segment* ', seg, meanF0, pitch, ' *-- fundamental freq is out of range, could be noise')
                    del self.segments[segix]
        print("Segments remaining after fundamental frequency: ", len(self.segments))

    # The following are just wrappers for easier parsing of 3-element segment lists:
    # Segmenter class still has its own joinGaps etc which operate on 2-element lists
    def joinGaps(self, maxgap):
        seg = Segmenter()
        self.segments = seg.joinGaps3(self.segments, maxgap=maxgap)
        print("Segments remaining after merge (<=%.2f secs): %d" % (maxgap, len(self.segments)))

    def deleteShort(self, minlength):
        seg = Segmenter()
        self.segments = seg.deleteShort3(self.segments, minlength=minlength)
        print("Segments remaining after deleting short (<%.2f secs): %d" % (minlength, len(self.segments)))

    def splitLong(self, maxlen):
        seg = Segmenter()
        self.segments = seg.splitLong3(self.segments, maxlen=maxlen)
        print('Segments after splitting long segments (>%.2f secs): %d' % (maxlen, len(self.segments)))

    def checkSegmentOverlap(self):
        # Used for merging call types or different segmenter outputs
        seg = Segmenter()
        self.segments = seg.checkSegmentOverlap3(self.segments)
        print("Segments produced after merging: %d" % len(self.segments))
