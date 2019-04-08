# generate annotation.py
# Virginia Listanti
# Script to generate annotations for sliding windows.
# To use for filter training

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
import wavio
import os, json
import math, re
import datetime


def genGT(dirName,species='Kiwi',duration=0,window=1, inc=None):
    """
    Given the directory where sound files and the annotations along with the species being considered,
    it generates the ground truth txt files using 'annotation2GT'
    If you know the duration of a recording pass it through 'duration'.
    'window' gives the desired length of the window. It is the resolution of the list.
    'inc' gives the increment. If given it is the "real" resolution.
    """
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.endswith('.wav'):
                filename = root + '/' + filename
                annotation2GT_OvWin(filename,species,duration=duration,window=window,inc=inc)
    print ("Generated GT")

def annotation2GT_OvWin(wavFile, species, duration=0,window=1, inc=None, notargetsp=False):
    """
    This generates the ground truth for a given sound file
    Given the AviaNZ annotation, returns the ground truth as a txt file
    """
    #Virginia:now it generate a text file for overlapping window

    # Virginia: set increment and resolution
    if inc==None:
        inc=window
        resol=window
    else:
    # Virginia: resolution is the "gcd" between window and inc. In this way I'm hoping to solve the case with
    # 75% overlap
        resol=(math.gcd(int(100*window),int(100*inc)))/100

    datFile = wavFile + '.data'
    #Virginia:changed file name appearence
    eFile = datFile[:-9] +'-res'+str(resol)+'sec.txt'

    if duration == 0:
        wavobj = wavio.read(wavFile)
        sampleRate = wavobj.rate
        # Virginia: number of sample for increment
        res_sr= resol*sampleRate
        data = wavobj.data
        duration = int(np.ceil(len(data) / res_sr))
        # Virginia: number of expected segments = number of segments of length resol

    GT = np.zeros((duration, 4))
    GT = GT.tolist()
    GT[:][1] = str(0)
    GT[:][2] = ''
    GT[:][3] = ''

    # fHigh and fLow for text boxes
    fLow = sampleRate/2
    fHigh = 0
    lenMin = duration
    lenMax =0
    if os.path.isfile(datFile):
        # print(datFile)
        with open(datFile) as f:
            segments = json.load(f)
        for seg in segments:
            if seg[0] == -1:
                continue
            if not species.title() in seg[4][0]:
                continue
            else:
                # print("lenMin, seg[1]-seg[0]", lenMin, seg[1]-seg[0])
                #Virginia: added this variable so the machine don't have to calculate it every rime
                dur_segm=seg[1]-seg[0]
                if lenMin > dur_segm:
                    lenMin = dur_segm
                if lenMax < dur_segm:
                    lenMax = dur_segm
                if fLow > seg[2]:
                    fLow = seg[2]
                if fHigh < seg[3]:
                    fHigh = seg[3]
                # Record call type for evaluation purpose
                if species == 'Kiwi (Nth Is Brown)' or species == 'Kiwi' or species == 'Kiwi(Tokoeka Fiordland)':
                    # check male, female, duet calls
                    if '(M)' in str(seg[4][0]):
                        type = 'M'
                    elif '(F)' in str(seg[4][0]):
                        type = 'F'
                    elif '(D)' in str(seg[4][0]):
                        type = 'D'
                    else:
                        type = 'K'
                elif species == 'Morepork':
                    # check mp, tril, weow, roro calls
                    if '(Mp)' in str(seg[4][0]):
                        type = 'Mp'
                    elif '(Tril)' in str(seg[4][0]):
                        type = 'Tril'
                    elif '(Weow)' in str(seg[4][0]):
                        type = 'Weow'
                    elif '(Roro)' in str(seg[4][0]):
                        type = 'Roro'
                elif species == 'Robin':
                    type = 'Robin'
                elif species == 'Kakapo(B)':
                    type = 'B'
                elif species == 'Kakapo(C)':
                    type = 'C'
                elif species == 'Bittern':
                    type = 'Bittern'
                # Record call quality for evaluation purpose
                if re.search('1', seg[4][0]):
                    quality = '1'  # v close
                elif re.search('2', seg[4][0]):
                    quality = '2'  # close
                elif re.search('3', seg[4][0]):
                    quality = '3'  # fade
                elif re.search('4', seg[4][0]):
                    quality = '4'  # v fade
                elif re.search('5', seg[4][0]):
                    quality = '5'  # v v fade
                #Virginia: start and end must be read in resol base
                s=int(math.floor(seg[0]/resol))
                e=int(math.ceil(seg[1]/resol))
                print("start and end: ", s, e)
                for i in range(s, e):
                    # when there are overlapping calls priority for good quality one
                    if GT[i][1] == '1' and GT[i][3] >= quality:
                        continue
                    else:
                        GT[i][1] = str(1)
                        GT[i][2] = type
                        GT[i][3] = quality

    # Empty files cannot be used now, and lead to problems
    # if len(GT)==0:
    #     print("ERROR: no calls for this species in file", datFile)
    #     return

    for line in GT:
        if line[1] == 0.0:
            line[1] = '0'
        if line[2] == 0.0:
            line[2] = ''
        if line[3] == 0.0:
            line[3] = ''
    # now save GT as a .txt file
    # Virginia: from index reconstruct time
    for i in range(1, duration + 1):
        GT[i - 1][0] = str(i*resol)  # add time as the first column to make GT readable
    # strings = (str(item) for item in GT)
    with open(eFile, "w") as f:
        for l, el in enumerate(GT):
            string = '\t'.join(map(str, el))
            for item in string:
                f.write(item)
            f.write('\n')
        f.write('\n')
    print(eFile)
    print(lenMin, lenMax, fLow, fHigh, sampleRate)
    #return [lenMin, lenMax, fLow, fHigh, sampleRate]

#Virginia:change directory name
# genGT('D:\AviaNZ\Sound Files\Fiordland kiwi\Dataset\\Negative',species='Kiwi(Tokoeka Fiordland)',window=1, inc=1)

