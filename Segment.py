
# Segment.py
#
# A variety of segmentation algorithms for AviaNZ

# Version 1.3 23/10/18
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis

#    AviaNZ birdsong analysis program
#    Copyright (C) 2017--2018

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
import numpy as np
import scipy.ndimage as spi
import time
from ext import ce_denoise as ce
import json
import os
import re
import math
from intervaltree import IntervalTree

from PyQt5.QtCore import QTime
from openpyxl import load_workbook, Workbook
from openpyxl.styles import colors
from openpyxl.styles import Font


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

        self.keys = [(lab['species'], lab['certainty']) for lab in self[4]]
        if len(self.keys)>len(set(self.keys)):
            print("ERROR: non-unique species/certainty combo detected")
            return

    def hasLabel(self, species, certainty):
        """ Check if label identified by species-cert combo is present in this segment. """
        return (species, certainty) in self.keys

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
        if self.hasLabel(species, certainty):
            print("ERROR: this species-certainty label already present")
            return
        label["species"] = species
        label["certainty"] = certainty

        self[4].append(label)
        self.keys.append((species, certainty))

    ### --- couple functions to process all labels for a given species ---

    def wipeSpecies(self, species):
        """ Remove all labels for species, return True if all labels were wiped
            (and the interface should delete the segment).
        """
        deletedAll = list(set([lab["species"] for lab in self[4]])) == [species]
        # note that removeLabel will re-add a Don't Know in the end, so can't just check the final label.
        for lab in self[4]:
            self.removeLabel(lab["species"], lab["certainty"])
        return deletedAll

    def confirmLabels(self, species=None):
        """ Raise the certainty of this segment's uncertain labels to 100.
            Affects all species (if None) or indicated species.
        """
        for labix in range(len(self[4])):
            lab = self[4][labix]
            if (species is None or lab["species"]==species) and lab["certainty"] < 100:
                lab["certainty"] = 100
                self.keys[labix] = (lab["species"], lab["certainty"])

    def removeLabel(self, species, certainty):
        """ Removes label from this segment.
            Does not delete the actual segment - that's left for the interface to take care of.
        """
        deleted = False
        for lab in self[4]:
            if lab["species"]==species and lab["certainty"]==certainty:
                self[4].remove(lab)
                self.keys.remove((species, certainty))
                # if that was the last label, flip to Don't Know
                if len(self[4])==0:
                    self.addLabel("Don't Know", 0)
                deleted = True
                break

        if not deleted:
            print("ERROR: could not find species-certainty combo to remove:", species, certainty)
            return

    def infoString(self):
        """ Returns a nicely-formatted string of this segment's info."""
        s = []
        for lab in self[4]:
            labs = "sp.: {}, cert.: {}%".format(lab["species"], lab["certainty"])
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

    def parseJSON(self, file, duration=0):
        """ Takes in a filename and reads metadata to self.metadata,
            and other segments to just the main body of self.
            If wav file is loaded, pass the true duration in s to check
            (it will override any duration read from the JSON).
        """
        try:
            file = open(file, 'r')
            annots = json.load(file)
            file.close()
        except Exception as e:
            print("ERROR: file %s failed to load with error:" % file)
            print(e)
            return

        # first segment stores metadata
        self.metadata = dict()
        if isinstance(annots[0], list) and annots[0][0] == -1:
            print("old format metadata detected")
            self.metadata = {"Operator": annots[0][2], "Reviewer": annots[0][3]}
            # when file is loaded, true duration can be passed. Otherwise,
            # some old files have duration in samples, so need a rough check
            if duration>0:
                self.metadata["Duration"] = duration
            elif isinstance(annots[0][1],str):
                # TODO: Best thing to do for this?
                self.metadata["Duration"] = 0
            elif annots[0][1]>0 and annots[0][1]<100000:
                self.metadata["Duration"] = annots[0][1]
            else:
                print("ERROR: duration not found in metadata, need to supply as argument")
                return
            del annots[0]

        elif isinstance(annots[0], dict):
            self.metadata = annots[0]
            if duration>0:
                self.metadata["Duration"] = duration
            del annots[0]

        # original code also stored+parsed noise data from array-format metadata,
        # should we keep this?
        # if type(self.segments[0][4]) is int:
        #     self.noiseLevel = None
        #     self.noiseTypes = []
        # else:
        #     self.noiseLevel = self.segments[0][4][0]
        #     self.noiseTypes = self.segments[0][4][1]

        # read the segments
        self.clear()
        for annot in annots:
            if not isinstance(annot, list) or len(annot)!=5:
                print("ERROR: annotation in wrong format:", annot)
                return

            # deal with old formats here, so that the Segment class
            # could require (and validate) clean input

            # Early version of AviaNZ stored freqs as values between 0 and 1.
            # The .1 is to take care of rounding errors
            if 0 < annot[2] < 1.1 and 0 < annot[3] < 1.1:
                print("Warning: ignoring old-format frequency marks")
                annot[2] = 0
                annot[3] = 0

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
        print("%d segments read" % len(self))

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
            seglist - list of 2-col segments [t1, t2]
            label is built from kwd.
            These will be converted to [t1, t2, freq[0], freq[1], label]
            and stored.
        """
        if not isinstance(freq, list) or freq[0]<0 or freq[1]<0:
            print("ERROR: cannot use frequencies", freq)
            return

        for seg in seglist:
            newseg = [seg[0], seg[1], freq[0], freq[1], [kwd]]
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

    def saveJSON(self, file):
        """ Returns 1 on succesful save."""
        annots = [self.metadata]
        for seg in self:
            annots.append(seg)

        file = open(file, 'w')
        json.dump(annots, file)
        file.write("\n")
        file.close()
        return 1

    def exportGT(self, filename, species, window=1, inc=None):
        """ Given the AviaNZ annotations, exports a 0/1 ground truth as a txt file,
            and returns other parameters for populating the training dialogs.
        filename - current wav file name.
        species - string, will export the annotations for it.
        Window and inc defined as in waveletSegment.
        """

        if inc is None:
            inc = window
        resolution = math.gcd(int(100*window), int(100*inc)) / 100

        # number of segments of width window at inc overlap
        duration = int(np.ceil(self.metadata["Duration"] / resolution))
        eFile = filename[:-4] + '-res' + str(float(resolution)) + 'sec.txt'

        # TODO: empty files (no annotations or no sound) will lead to problems
        thisSpSegs = self.getSpecies(species)
        if len(thisSpSegs)==0:
            print("Warning: no annotations for this species found in file", filename)
            # delete the file to avoid problems with old GT files
            os.remove(eFile)
            # return some default constants
            return((100, 0, 32000, 0, 32000))

        GT = np.tile([0, 0, None], (duration,1))
        # fill first column with "time"
        GT[:,0] = range(1, duration+1)
        GT[:,0] = GT[:,0] * resolution

        for segix in thisSpSegs:
            seg = self[segix]
            # start and end in resolution base
            s = int(math.floor(seg[0] / resolution))
            e = int(math.ceil(seg[1] / resolution))
            for i in range(s, e):
                GT[i,1] = 1
                GT[i,2] = species
        GT = GT.tolist()

        # now save the resulting txt:
        with open(eFile, "w") as f:
            for l, el in enumerate(GT):
                string = '\t'.join(map(str,el))
                for item in string:
                    f.write(item)
                f.write('\n')
            f.write('\n')
            print("output successfully saved to file", eFile)

        # get parameter limits for populating training dialogs:
        # FreqRange, in Hz
        fLow = np.min([self[segix][2] for segix in thisSpSegs])
        fHigh = np.max([self[segix][3] for segix in thisSpSegs])
        # TimeRange, in s
        lenMin = np.min([self[segix][1] - self[segix][0] for segix in thisSpSegs])
        lenMax = np.max([self[segix][1] - self[segix][0] for segix in thisSpSegs])

        return((lenMin, lenMax, fLow, fHigh))

    def exportExcel(self, dirName, filename, action, pagelen, numpages=1, speciesList=[], startTime=0, resolution=1):
        """ Exports the annotations to xlsx, with three sheets:
        time stamps, presence/absence, and per second presence/absence.
        Saves each species into a separate workbook,
        + an extra workbook for all species (to function as a readable segment printout).
        It makes the workbook if necessary.

        Inputs
            dirName:    xlsx will be stored here
            filename:   name of the wav file, to be recorded inside the xlsx
            action:     "append" or "overwrite" any found Excels
            pagelen:    page length, seconds (for filling out absence)
            numpages:   number of pages in this file (of size pagelen)
            speciesList:    list of species that are currently processed -- will force an xlsx output even if none were detected
            startTime:  timestamp for cell names
            resolution: output resolution on excel (sheet 3) in seconds. Default is 1
        """

        pagelen = math.ceil(pagelen)

        # will export species present in self, + passed as arg, + "all species" excel
        speciesList = set(speciesList)
        for seg in self:
            speciesList.update([lab["species"] for lab in seg[4]])
        speciesList.add("All species")
        print("The following species were detected for export:", speciesList)

        # ideally, we store relative paths, but that's not possible across drives:
        try:
            relfname = str(os.path.relpath(filename, dirName))
        except Exception as e:
            print("Falling back to absolute paths. Encountered exception:")
            print(e)
            relfname = str(os.path.abspath(filename))

        # functions for filling out the excel sheets:
        def writeToExcelp1(wb, segix, currsp):
            ws = wb['Time Stamps']
            r = ws.max_row + 1
            # Print the filename
            ws.cell(row=r, column=1, value=relfname)
            # Loop over the segments
            for segi in segix:
                seg = self[segi]
                ws.cell(row=r, column=2, value=str(QTime(0,0,0).addSecs(seg[0]+startTime).toString('hh:mm:ss')))
                ws.cell(row=r, column=3, value=str(QTime(0,0,0).addSecs(seg[1]+startTime).toString('hh:mm:ss')))
                if seg[3]!=0:
                    ws.cell(row=r, column=4, value=int(seg[2]))
                    ws.cell(row=r, column=5, value=int(seg[3]))
                if currsp=="All species":
                    text = [lab["species"] for lab in seg[4]]
                    ws.cell(row=r, column=6, value=", ".join(text))
                r += 1

        def writeToExcelp2(wb, segix):
            ws = wb['Presence Absence']
            r = ws.max_row + 1
            ws.cell(row=r, column=1, value=relfname)
            ws.cell(row=r, column=2, value='_')
            if len(segix)>0:
                ws.cell(row=r, column=2, value='Yes')
            else:
                ws.cell(row=r, column=2, value='No')

        def writeToExcelp3(wb, detected, pagenum):
            # writes binary output DETECTED (per s) from page PAGENUM of length PAGELEN
            starttime = pagenum * pagelen
            ws = wb['Per Time Period']
            # print resolution "header"
            r = ws.max_row + 1
            ws.cell(row=r, column=1, value=str(resolution) + ' secs resolution')
            ft = Font(color=colors.DARKYELLOW)
            ws.cell(row=r, column=1).font=ft
            # print file name and page number
            ws.cell(row=r+1, column=1, value=relfname)
            ws.cell(row=r+1, column=2, value=str(pagenum+1))
            # fill the header and detection columns
            c = 3
            for t in range(0, len(detected), resolution):
                # absolue (within-file) times:
                win_start = starttime + t
                win_end = min(win_start+resolution, int(pagelen * numpages))
                ws.cell(row=r, column=c, value=str(win_start) + '-' + str(win_end))
                ws.cell(row=r, column=c).font = ft
                # within-page times:
                det = 1 if np.sum(detected[t:win_end-starttime])>0 else 0
                ws.cell(row=r+1, column=c, value=det)
                c += 1

        # now, generate the actual files, SEPARATELY FOR EACH SPECIES:
        for species in speciesList:
            print("Exporting species %s" % species)
            # clean version for filename
            speciesClean = re.sub(r'\W', "_", species)

            # setup output files:
            # if an Excel exists, append (so multiple files go into one worksheet)
            # if not, create new
            eFile = dirName + '/DetectionSummary_' + speciesClean + '.xlsx'

            if action == "overwrite" or not os.path.isfile(eFile):
                # make a new workbook:
                wb = Workbook()
                wb.create_sheet(title='Time Stamps', index=1)
                wb.create_sheet(title='Presence Absence', index=2)
                wb.create_sheet(title='Per Time Period', index=3)

                # First sheet
                ws = wb['Time Stamps']
                ws.cell(row=1, column=1, value="File Name")
                ws.cell(row=1, column=2, value="start (hh:mm:ss)")
                ws.cell(row=1, column=3, value="end (hh:mm:ss)")
                ws.cell(row=1, column=4, value="min freq. (Hz)")
                ws.cell(row=1, column=5, value="max freq. (Hz)")
                if species=="All_species":
                    ws.cell(row=1, column=6, value="species")

                # Second sheet
                ws = wb['Presence Absence']
                ws.cell(row=1, column=1, value="File Name")
                ws.cell(row=1, column=2, value="Presence/Absence")

                # Third sheet
                ws = wb['Per Time Period']
                ws.cell(row=1, column=1, value="File Name")
                ws.cell(row=1, column=2, value="Page")
                ws.cell(row=1, column=3, value="Presence=1, Absence=0")

                # Hack to delete original sheet
                del wb['Sheet']
            elif action == "append":
                try:
                    wb = load_workbook(eFile)
                except Exception as e:
                    print("ERROR: cannot open file %s to append" % eFile)  # no read permissions or smth
                    print(e)
                    return 0
            else:
                print("ERROR: unrecognized action", action)
                return 0

            # extract segments for the current species
            # if species=="All", take ALL segments.
            if species=="All species":
                speciesSegs = range(len(self))
            else:
                speciesSegs = self.getSpecies(species)

            # export segments
            writeToExcelp1(wb, speciesSegs, species)
            # export presence/absence
            writeToExcelp2(wb, speciesSegs)

            # Generate per second binary output
            # (assuming all pages are of same length as current data)
            for p in range(0, numpages):
                detected = np.zeros(pagelen)
                for segi in speciesSegs:
                    seg = self[segi]
                    for t in range(pagelen):
                        # convert within-page time to segment (within-file) time
                        truet = t + p*pagelen
                        if math.floor(seg[0]) <= truet and truet < math.ceil(seg[1]):
                            detected[t] = 1
                # write page p to xlsx
                writeToExcelp3(wb, detected, p)

            # Save the file
            try:
                wb.save(eFile)
            except Exception as e:
                print("ERROR: could not create new file %s" % eFile)  # no read permissions or smth
                print(e)
                return 0
        return 1


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

    def __init__(self, data, sg, sp, fs, window_width=256, incr=128, mingap=0.3, minlength=0.2):
        self.data = data
        self.fs = fs
        # Spectrogram
        self.sg = sg
        # This is the reference to SignalProc
        self.sp = sp
        # These are the spectrogram params. Needed to compute times.
        self.window_width = window_width
        self.incr = incr
        self.mingap = mingap
        self.minlength = minlength

    def setNewData(self, data, sg, fs, window_width, incr):
        # To be called when a new sound file is loaded
        self.data = data
        self.fs = fs
        self.sg = sg
        self.window_width = window_width
        self.incr = incr

    def bestSegments(self,FIRthr=0.7,medianClipthr=3.0,yinthr=0.9,mingap=0, minlength=0, maxlength=5.0):
        # Have a go at performing generally reasonably segmentation
        # TODO: Decide on this!
        segs1 = self.checkSegmentLength(self.segmentByFIR(FIRthr),mingap,minlength,maxlength)
        segs2 = self.checkSegmentLength(self.medianClip(medianClipthr),mingap,minlength,maxlength)
        segs3, p, t = self.yin(100, thr=yinthr, returnSegs=True)
        segs3 = self.checkSegmentOverlap(segs3,mingap)
        segs3 = self.checkSegmentLength(segs3,mingap,minlength,maxlength)
        segs1 = self.mergeSegments(segs1, segs2)
        segs = self.mergeSegments(segs1,segs3)
        segs = segs[::-1]
        return segs

    def mergeSegments(self,segs1,segs2,ignoreInsideEnvelope=True):
        """ Given two segmentations of the same file, return the merged set of them
        Two similar segments should be replaced by their union
        Those that are inside another should be removed (?) or the too-large one deleted?
        If ignoreInsideEnvelope is true this is the first of those, otherwise the second
        """

        t = IntervalTree()

        # Put the first set into the tree
        for s in segs1:
            t[s[0]:s[1]] = s

        # Decide whether or not to put each segment in the second set in
        for s in segs2:
            overlaps = t.search(s[0],s[1])
            # If there are no overlaps, add it
            if len(overlaps)==0:
                t[s[0]:s[1]] = s
            else:
                # Search for any enveloped, if there are remove and add the new one
                envelops = t.search(s[0],s[1],strict=True)
                if len(envelops) > 0:
                    if ignoreInsideEnvelope:
                        # Remove any inside the envelope of the test point
                        t.remove_envelop(s[0],s[1])
                        overlaps = t.search(s[0], s[1])
                        #print s[0], s[1], overlaps
                        # Open out the region, delete the other
                        for o in overlaps:
                            if o.begin < s[0]:
                                s[0] = o.begin
                                t.remove(o)
                            if o.end > s[1]:
                                s[1] = o.end
                                t.remove(o)
                        t[s[0]:s[1]] = s
                else:
                    # Check for those that intersect the ends, widen them out a bit
                    for o in overlaps:
                        if o.begin > s[0]:
                            t[s[0]:o[1]] = (s[0],o[1])
                            t.remove(o)
                        if o.end < s[1]:
                            t[o[0]:s[1]] = (o[0],s[1])
                            t.remove(o)

        segs = []
        for a in t:
            segs.append([a[0],a[1]])
        return segs

    def checkSegmentLength(self,segs, mingap=0, minlength=0, maxlength=5.0):
        """ Checks whether start/stop segments are long enough
        These are species specific!
        """
        if mingap == 0:
            mingap = self.mingap
        if minlength == 0:
            minlength = self.minlength
        # TODO: Doesn't currently use maxlength
        for i in range(len(segs))[-1::-1]:
            if i<len(segs)-1:
                if np.abs(segs[i][1] - segs[i+1][0]) < mingap:
                    segs[i][1] = segs[i+1][1]
                    del segs[i+1]
            if np.abs(segs[i][1] - segs[i][0]) < minlength:
                del segs[i]
        return segs

    def identifySegments(self, seg, maxgap=1, minlength=1, notSpec=False):
        """ Turns presence/absence segments into a list of start/stop times
        Note the two parameters
        """
        segments = []
        start = seg[0]
        for i in range(1, len(seg)):
            if seg[i] <= seg[i - 1] + maxgap:
                pass
            else:
                # See if segment is long enough to be worth bothering with
                if (seg[i - 1] - start) > minlength:
                    if notSpec:
                        segments.append([start, seg[i - 1]])
                    else:
                        segments.append([float(start) * self.incr / self.fs, float(seg[i - 1]) * self.incr / self.fs])
                start = seg[i]
        if seg[-1] - start > minlength:
            if notSpec:
                segments.append([start, seg[i-1]])
            else:
                segments.append([float(start) * self.incr / self.fs, float(seg[-1]) * self.incr / self.fs])

        return segments

    def segmentByFIR(self, threshold):
        """ Segmentation using FIR envelope.
        """
        from scipy.interpolate import interp1d
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
        ind = np.squeeze(np.where(envelope > np.median(envelope) + threshold * np.std(envelope)))
        return self.identifySegments(ind, minlength=10)

    def segmentByAmplitude(self, threshold, usePercent=True):
        """ Bog standard amplitude segmentation.
        A straw man, do not use.
        """
        if usePercent:
            threshold = threshold*np.max(self.data)
        seg = np.where(np.abs(self.data)>threshold)
        if np.shape(np.squeeze(seg))[0]>0:
            return self.identifySegments(np.squeeze(seg)/float(self.incr))
        else:
            return []

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


    def Harma(self,thr=10.,stop_thr=0.8,minSegment=50):
        """ Harma's method, but with a different stopping criterion
        # Assumes that spectrogram is not normalised
        maxFreqs = 10. * np.log10(np.max(self.sg, axis = 1))
        """
        maxFreqs = 10. * np.log10(np.max(self.sg, axis=1))
        from scipy.signal import medfilt
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
        from scipy.signal import medfilt
        maxFreqs = medfilt(maxFreqs, 21)
        ind = np.squeeze(np.where(maxFreqs > (np.mean(maxFreqs)+thr*np.std(maxFreqs))))
        return self.identifySegments(ind, minlength=10)

    def medianClip(self, thr=3.0, medfiltersize=5, minaxislength=5, minSegment=70):
        """ Median clipping for segmentation
        Based on Lasseck's method
        minaxislength - min "length of the minor axis of the ellipse that has the same normalized second central moments as the region", based on skm.
        minSegment - min number of pixels exceeding thr to declare an area as segment.
        This version only clips in time, ignoring frequency
        And it opens up the segments to be maximal (so assumes no overlap).
        The multitaper spectrogram helps a lot

        """
        tt = time.time()
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

        # This is the stencil for the closing and dilation. It's a 5x5 diamond. Can also use a 3x3 diamond
        diamond = np.zeros((5,5),dtype=int)
        diamond[2,:] = 1
        diamond[:,2] = 1
        diamond[1,1] = diamond[1,3] = diamond[3,1] = diamond[3,3] = 1
        #diamond[2, 1:4] = 1
        #diamond[1:4, 2] = 1

        import scipy.ndimage as spi
        clipped = spi.binary_closing(clipped,structure=diamond).astype(int)
        clipped = spi.binary_dilation(clipped,structure=diamond).astype(int)
        clipped = spi.median_filter(clipped,size=medfiltersize)
        clipped = spi.binary_fill_holes(clipped)

        import skimage.measure as skm
        blobs = skm.regionprops(skm.label(clipped.astype(int)))

        # Delete blobs that are too small
        keep = []
        for i in range(len(blobs)):
            if blobs[i].filled_area > minSegment and blobs[i].minor_axis_length > minaxislength:
                keep.append(i)

        list = []
        blobs = [blobs[i] for i in keep]

        # convert bounding box pixels to milliseconds:
        for l in blobs:
            list.append([float(l.bbox[0] * self.incr / self.fs),
                    float(l.bbox[2] * self.incr / self.fs)])
        return list

    def checkSegmentOverlap(self,segs,minSegment=50):
        # Needs to be python array, not np array
        # Sort by increasing start times
        if isinstance(segs, np.ndarray):
            segs = segs.tolist()
        segs = sorted(segs)
        segs = np.array(segs)

        newsegs = []
        # Loop over segs until the start value of 1 is not inside the end value of the previous
        s=0
        while s<len(segs):
            i = s
            end = segs[i,1]
            while i < len(segs)-1 and segs[i+1,0] < end:
                i += 1
                end = max(end, segs[i,1])
            newsegs.append([segs[s,0],end])
            s = i+1

        return newsegs

    def mergeshort(self, segs, minlen):
        newsegs = []
        # loop over and check for short segs, merge
        i = 0
        while i < len(segs):
            if segs[i][1]-segs[i][0] < minlen and i+1 < len(segs):
                newsegs.append([segs[i][0], segs[i+1][1]])
                i += 2
            else:
                newsegs.append(segs[i])
                i += 1

        l = len(newsegs)
        if newsegs[l-1][1]-newsegs[l-1][0] < minlen and l > 1:
            temp = [newsegs[l-2][0], newsegs[l-1][1]]
            del newsegs[-1]
            newsegs.append(temp)

        return newsegs


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
            #print(i)
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

    def onsets(self,thr=3.0):
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

    def yin(self,minfreq=100, minperiods=3, thr=0.5, W=1000, returnSegs=False):
        """ Segmentation by computing the fundamental frequency.
        Uses the Yin algorithm of de Cheveigne and Kawahara (2002)
        """
        if self.data.dtype == 'int16':
            data = self.data.astype(float)/32768.0
        else:
            data = self.data

        # The threshold is necessarily higher than the 0.1 in the paper

        # Window width W should be at least 3*period.
        # A sample rate of 16000 and a min fundamental frequency of 100Hz would then therefore suggest reasonably short windows
        minwin = float(self.fs) / minfreq * minperiods
        if minwin > W:
            print("Extending window width to ", minwin)
            W = minwin

        # Make life easier, and make W be a function of the spectrogram window width
        W = int(round(W/self.window_width)*self.window_width)

        # work on a copy of the main data, just to be safer
        data2 = np.zeros(len(data))
        data2[:] = data[:]

        # Now, compute squared diffs between signal and shifted signal
        # (Yin fund freq estimator)
        # over all tau<W, for each start position.
        # (starts are shifted by half a window size), i.e.:
        starts = range(0, len(data) - 2 * W, W // 2)

        # Will return pitch as self.fs/besttau for each window
        pitch = ce.FundFreqYin(data2, W, thr, self.fs)

        if returnSegs:
            ind = np.squeeze(np.where(pitch > minfreq))
            segs = self.identifySegments(ind, notSpec=True)
            for s in segs:
                s[0] = float(s[0])/len(pitch) * np.shape(self.sg)[0]/self.fs*self.incr # W / self.window_width
                s[1] = float(s[1])/len(pitch) * np.shape(self.sg)[0]/self.fs*self.incr # W / self.window_width
            return segs, pitch, np.array(starts)
        else:
            return pitch, np.array(starts), minfreq, W

    def findCCMatches(self,seg,sg,thr):
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

    def findDTWMatches(self,seg,data,thr):
        # TODO: This is slow and crap. Note all the same length, for a start, and the fact that it takes forever!
        # Use MFCC first?
        d = np.zeros(len(data))
        for i in range(len(data)):
            d[i] = self.dtw(seg,data[i:i+len(seg)])
        return d

    def dtw(self,x,y,wantDistMatrix=False):
        # Compute the dynamic time warp between two 1D arrays
        dist = np.zeros((len(x)+1,len(y)+1))
        dist[1:,:] = np.inf
        dist[:,1:] = np.inf
        for i in range(len(x)):
            for j in range(len(y)):
                dist[i+1,j+1] = np.abs(x[i]-y[j]) + min(dist[i,j+1],dist[i+1,j],dist[i,j])
        if wantDistMatrix:
            return dist
        else:
            return dist[-1,-1]

    def dtw_path(self,d):
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

# Below are test functions for the segmenters.
def convertAmpltoSpec(x,fs,inc):
    """ Unit conversion """
    return x*fs/inc

def testMC():
    import wavio
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui

    #wavobj = wavio.read('Sound Files/kiwi_1min.wav')
    wavobj = wavio.read('Sound Files/tril1.wav')
    fs = wavobj.rate
    data = wavobj.data#[:20*fs]

    if data.dtype is not 'float':
        data = data.astype('float')  #/ 32768.0

    if np.shape(np.shape(data))[0] > 1:
        data = data[:, 0]

    import SignalProc
    sp = SignalProc.SignalProc(data,fs,256,128)
    sg = sp.spectrogram(data=data,window_width=256,incr=128,window='Hann',mean_normalise=True,onesided=True,multitaper=False,need_even=False)
    s = Segmenter(data,sg,sp,fs)

    #print np.shape(sg)

    #s1 = s.medianClip()
    s1,p,t = s.yin(returnSegs=True)
    app = QtGui.QApplication([])

    mw = QtGui.QMainWindow()
    mw.show()
    mw.resize(800, 600)

    win = pg.GraphicsLayoutWidget()
    mw.setCentralWidget(win)
    vb1 = win.addViewBox(enableMouse=False, enableMenu=False, row=0, col=0)
    im1 = pg.ImageItem(enableMouse=False)
    vb1.addItem(im1)
    im1.setImage(10.*np.log10(sg))

    # vb2 = win.addViewBox(enableMouse=False, enableMenu=False, row=1, col=0)
    # im2 = pg.ImageItem(enableMouse=False)
    # vb2.addItem(im2)
    # im2.setImage(c)

    vb3 = win.addViewBox(enableMouse=False, enableMenu=False, row=1, col=0)
    im3 = pg.ImageItem(enableMouse=False)
    vb3.addItem(im3)
    im3.setImage(10.*np.log10(sg))

    vb4 = win.addViewBox(enableMouse=False, enableMenu=False, row=2, col=0)
    im4 = pg.PlotDataItem(enableMouse=False)
    vb4.addItem(im4)
    im4.setData(data)

    for seg in s1:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        #a.setRegion([seg[0],seg[1]])
        vb3.addItem(a, ignoreBounds=True)

    QtGui.QApplication.instance().exec_()


def showSegs():
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    import wavio
    import WaveletSegment
    from time import time

    #wavobj = wavio.read('Sound Files/tril1.wav')
    #wavobj = wavio.read('Sound Files/010816_202935_p1.wav')
    #wavobj = wavio.read('Sound Files/20170515_223004 piping.wav')
    wavobj = wavio.read('Sound Files/kiwi_1min.wav')
    fs = wavobj.rate
    data = wavobj.data#[:20*fs]

    if data.dtype is not 'float':
        data = data.astype('float') # / 32768.0

    if np.shape(np.shape(data))[0] > 1:
        data = data[:, 0]

    import SignalProc
    sp = SignalProc.SignalProc(data,fs,256,128)
    sg = sp.spectrogram(data,multitaper=False)
    s = Segment(data,sg,sp,fs,50)

    # FIR: threshold doesn't matter much, but low is better (0.01).
    # Amplitude: not great, will have to work on width and abs if want to use it (threshold about 0.6)
    # Power: OK, but threshold matters (0.5)
    # Median clipping: OK, threshold of 3 fine.
    # Onsets: Threshold of 4.0 was fine, lower not. Still no offsets!
    # Yin: Threshold 0.9 is pretty good
    # Energy: Not great, but thr 1.0
    ts = time()
    s1=s.checkSegmentLength(s.segmentByFIR(0.1))
    s2=s.checkSegmentLength(s.segmentByFIR(0.01))
    s3= s.checkSegmentLength(s.medianClip(3.0))
    s4= s.checkSegmentLength(s.medianClip(2.0))
    s5,p,t=s.yin(100, thr=0.5,returnSegs=True)
    s5 = s.checkSegmentLength(s5)
    s6=s.mergeSegments(s2,s4)
    ws = WaveletSegment.WaveletSegment()
    s7= ws.waveletSegment_test(None, data, fs, None, 'Kiwi', False)
    #print('Took {}s'.format(time() - ts))
    #s7 = s.mergeSegments(s1,s.mergeSegments(s3,s4))

    #s4, samp = s.segmentByFIR(0.4)
    #s4 = s.checkSegmentLength(s4)
    #s2 = s.segmentByAmplitude1(0.6)
    #s5 = s.checkSegmentLength(s.segmentByPower(0.3))
    #s6, samp = s.segmentByFIR(0.6)
    #s6 = s.checkSegmentLength(s6)
    #s7 = []
    #s5 = s.onsets(3.0)
    #s6 = s.segmentByEnergy(1.0,500)

    #s5 = s.Harma(5.0,0.8)
    #s4 = s.Harma(10.0,0.8)
    #s7 = s.Harma(15.0,0.8)

    #s2 = s.segmentByAmplitude1(0.7)
    #s3 = s.segmentByPower(1.)
    #s4 = s.medianClip(3.0)
    #s5 = s.onsets(3.0)
    #s6, p, t = s.yin(100,thr=0.5,returnSegs=True)
    #s7 = s.Harma(10.0,0.8)

    app = QtGui.QApplication([])

    mw = QtGui.QMainWindow()
    mw.show()
    mw.resize(800, 600)

    win = pg.GraphicsLayoutWidget()
    mw.setCentralWidget(win)
    vb1 = win.addViewBox(enableMouse=False, enableMenu=False, row=0, col=0)
    im1 = pg.ImageItem(enableMouse=False)
    vb1.addItem(im1)
    im1.setImage(10.*np.log10(sg))

    vb2 = win.addViewBox(enableMouse=False, enableMenu=False, row=1, col=0)
    im2 = pg.ImageItem(enableMouse=False)
    vb2.addItem(im2)
    im2.setImage(10.*np.log10(sg))

    vb3 = win.addViewBox(enableMouse=False, enableMenu=False, row=2, col=0)
    im3 = pg.ImageItem(enableMouse=False)
    vb3.addItem(im3)
    im3.setImage(10.*np.log10(sg))

    vb4 = win.addViewBox(enableMouse=False, enableMenu=False, row=3, col=0)
    im4 = pg.ImageItem(enableMouse=False)
    vb4.addItem(im4)
    im4.setImage(10.*np.log10(sg))

    vb5 = win.addViewBox(enableMouse=False, enableMenu=False, row=4, col=0)
    im5 = pg.ImageItem(enableMouse=False)
    vb5.addItem(im5)
    im5.setImage(10.*np.log10(sg))

    vb6 = win.addViewBox(enableMouse=False, enableMenu=False, row=5, col=0)
    im6 = pg.ImageItem(enableMouse=False)
    vb6.addItem(im6)
    im6.setImage(10.*np.log10(sg))

    vb7 = win.addViewBox(enableMouse=False, enableMenu=False, row=6, col=0)
    im7 = pg.ImageItem(enableMouse=False)
    vb7.addItem(im7)
    im7.setImage(10.*np.log10(sg))

    print("====")
    print(s1)
    for seg in s1:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb1.addItem(a, ignoreBounds=True)

    print(s2)
    for seg in s2:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb2.addItem(a, ignoreBounds=True)

    print(s3)
    for seg in s3:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb3.addItem(a, ignoreBounds=True)

    print(s4)
    for seg in s4:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb4.addItem(a, ignoreBounds=True)

    print(s5)
    for seg in s5:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb5.addItem(a, ignoreBounds=True)

    print(s6)
    for seg in s6:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb6.addItem(a, ignoreBounds=True)

    print(s7)
    for seg in s7:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0],fs,128), convertAmpltoSpec(seg[1],fs,128)])
        vb7.addItem(a, ignoreBounds=True)

    QtGui.QApplication.instance().exec_()

def showSpecDerivs():
    import SignalProc
    reload(SignalProc)
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    import wavio

    #wavobj = wavio.read('Sound Files/tril1.wav')
    #wavobj = wavio.read('Sound Files/010816_202935_p1.wav')
    #wavobj = wavio.read('Sound Files/20170515_223004 piping.wav')
    wavobj = wavio.read('Sound Files/kiwi_1min.wav')
    fs = wavobj.rate
    data = wavobj.data[:20*fs]

    if data.dtype is not 'float':
        data = data.astype('float')     # / 32768.0

    if np.shape(np.shape(data))[0] > 1:
        data = data[:, 0]

    import SignalProc
    sp = SignalProc.SignalProc(data, fs, 256, 128)
    sg = sp.spectrogram(data, multitaper=False)

    h,v,b = sp.spectralDerivatives()
    h = np.abs(np.where(h == 0, 0.0, 10.0 * np.log10(h)))
    v = np.abs(np.where(v == 0, 0.0, 10.0 * np.log10(v)))
    b = np.abs(np.where(b == 0, 0.0, 10.0 * np.log10(b)))
    s = Segment(data, sg, sp, fs, 50)

    hm = np.max(h[:, 10:], axis=1)
    inds = np.squeeze(np.where(hm > (np.mean(h[:,10:]+2.5*np.std(h[:, 10:])))))
    segmentsh = s.identifySegments(inds, minlength=10)

    vm = np.max(v[:, 10:], axis=1)
    inds = np.squeeze(np.where(vm > (np.mean(v[:, 10:]+2.5*np.std(v[:, 10:])))))
    segmentsv = s.identifySegments(inds, minlength=10)

    bm = np.max(b[:, 10:], axis=1)
    segs = np.squeeze(np.where(bm > (np.mean(b[:, 10:]+2.5*np.std(b[:, 10:])))))
    segmentsb = s.identifySegments(segs, minlength=10)
    #print np.mean(h), np.max(h)
    #print np.where(h>np.mean(h)+np.std(h))

    app = QtGui.QApplication([])

    mw = QtGui.QMainWindow()
    mw.show()
    mw.resize(800, 600)

    win = pg.GraphicsLayoutWidget()
    mw.setCentralWidget(win)
    vb1 = win.addViewBox(enableMouse=False, enableMenu=False, row=0, col=0)
    im1 = pg.ImageItem(enableMouse=False)
    vb1.addItem(im1)
    im1.setImage(10.*np.log10(sg))

    vb2 = win.addViewBox(enableMouse=False, enableMenu=False, row=1, col=0)
    im2 = pg.ImageItem(enableMouse=False)
    vb2.addItem(im2)
    im2.setImage(h)
    for seg in segmentsh:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0], fs, 128), convertAmpltoSpec(seg[1], fs, 128)])
        vb2.addItem(a, ignoreBounds=True)

    vb3 = win.addViewBox(enableMouse=False, enableMenu=False, row=2, col=0)
    im3 = pg.ImageItem(enableMouse=False)
    vb3.addItem(im3)
    im3.setImage(v)
    for seg in segmentsv:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0], fs, 128), convertAmpltoSpec(seg[1], fs, 128)])
        vb3.addItem(a, ignoreBounds=True)

    vb4 = win.addViewBox(enableMouse=False, enableMenu=False, row=3, col=0)
    im4 = pg.ImageItem(enableMouse=False)
    vb4.addItem(im4)
    im4.setImage(b)
    for seg in segmentsb:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0], fs, 128), convertAmpltoSpec(seg[1], fs, 128)])
        vb4.addItem(a, ignoreBounds=True)
    QtGui.QApplication.instance().exec_()

def detectClicks():
    import SignalProc
    # reload(SignalProc)
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    import wavio
    from scipy.signal import medfilt

    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\Clicks\\1ex\Lake_Thompson__01052018_SOUTH1047849_01052018_High_20180509_'
    #                     '20180509_183506.wav')  # close kiwi and rain
    wavobj = wavio.read('D:\AviaNZ\Sound_Files\Clicks\Lake_Thompson__01052018_SOUTH1047849_01052018_High_20180508_'
                        '20180508_200506.wav')  # very close kiwi with steady wind
    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\Clicks\\1ex\Murchison_Kelper_Heli_25042018_SOUTH7881_25042018_High_'
    #                     '20180405_20180405_211007.wav')
    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\\Noise examples\\Noise_10s\Rain_010.wav')
    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\Clicks\Ponui_SR2_Jono_20130911_021920.wav')   #
    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\Clicks\CL78_BIRM_141120_212934.wav')   #
    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\Clicks\CL78_BIRD_141120_212934.wav')   # Loud click
    # wavobj = wavio.read('D:\AviaNZ\Sound_Files\Tier1\Tier1 dataset\positive\DE66_BIRD_141011_005829.wav')   # close kiwi
    # wavobj = wavio.read('Sound Files/010816_202935_p1.wav')
    #wavobj = wavio.read('Sound Files/20170515_223004 piping.wav')
    # wavobj = wavio.read('Sound Files/test/DE66_BIRD_141011_005829.wav')
    #wavobj = wavio.read('/Users/srmarsla/DE66_BIRD_141011_005829_wb.wav')
    #wavobj = wavio.read('/Users/srmarsla/ex1.wav')
    #wavobj = wavio.read('/Users/srmarsla/ex2.wav')
    fs = wavobj.rate
    data = wavobj.data #[:20*fs]

    if data.dtype is not 'float':
        data = data.astype('float')     # / 32768.0

    if np.shape(np.shape(data))[0] > 1:
        data = data[:, 0]

    import SignalProc
    sp = SignalProc.SignalProc(data, fs, 128, 128)
    sg = sp.spectrogram(data, multitaper=False)
    s = Segment(data, sg, sp, fs, 128)

    # for each frq band get sections where energy exceeds some (90%) percentile
    # and generate a binary spectrogram
    sgb = np.zeros((np.shape(sg)))
    for y in range(np.shape(sg)[1]):
        ey = sg[:, y]
        # em = medfilt(ey, 15)
        ep = np.percentile(ey, 90)
        sgb[np.where(ey > ep), y] = 1

    # If lots of frq bands got 1 then predict a click
    clicks = []
    for x in range(np.shape(sg)[0]):
        if np.sum(sgb[x, :]) > np.shape(sgb)[1]*0.75:
            clicks.append(x)

    app = QtGui.QApplication([])

    mw = QtGui.QMainWindow()
    mw.show()
    mw.resize(1200, 500)

    win = pg.GraphicsLayoutWidget()
    mw.setCentralWidget(win)
    vb1 = win.addViewBox(enableMouse=False, enableMenu=False, row=0, col=0)
    im1 = pg.ImageItem(enableMouse=False)
    vb1.addItem(im1)
    im1.setImage(sgb)

    if len(clicks) > 0:
        clicks = s.identifySegments(clicks, minlength=1)

    for seg in clicks:
        a = pg.LinearRegionItem()
        a.setRegion([convertAmpltoSpec(seg[0], fs, 128), convertAmpltoSpec(seg[1], fs, 128)])
        vb1.addItem(a, ignoreBounds=True)

    QtGui.QApplication.instance().exec_()


    # energy = np.sum(sg, axis=1)
    # energy = medfilt(energy, 15)
    # e2 = np.percentile(energy, 50)*2
    # # Step 1: clicks have high energy
    # clicks = np.squeeze(np.where(energy > e2))
    # print(clicks)
    # if len(clicks) > 0:
    #     clicks = s.identifySegments(clicks, minlength=1)
    #
    # app = QtGui.QApplication([])
    #
    # mw = QtGui.QMainWindow()
    # mw.show()
    # mw.resize(800, 600)
    #
    # win = pg.GraphicsLayoutWidget()
    # mw.setCentralWidget(win)
    # vb1 = win.addViewBox(enableMouse=False, enableMenu=False, row=0, col=0)
    # im1 = pg.ImageItem(enableMouse=False)
    # vb1.addItem(im1)
    # im1.setImage(10.*np.log10(sg))
    #
    # for seg in clicks:
    #     a = pg.LinearRegionItem()
    #     a.setRegion([convertAmpltoSpec(seg[0], fs, 128), convertAmpltoSpec(seg[1],fs,128)])
    #     vb1.addItem(a, ignoreBounds=True)
    #
    # QtGui.QApplication.instance().exec_()

# detectClicks()
