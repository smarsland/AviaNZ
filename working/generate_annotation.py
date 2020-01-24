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
    #☺duration=900
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.endswith('.wav'):
            #1if filename.endswith('.data'):
                filename = root + '/' + filename
                annotation2GT_OvWin(filename,species,duration=duration,window=window,inc=inc)
    print ("Generated GT")

def annotation2GT_OvWin(wavFile, species, duration=0,window=1, inc=None, notargetsp=False):
<<<<<<< Updated upstream
=======
#def annotation2GT_OvWin(datFile, species, duration=0,window=1, inc=None, notargetsp=False):
>>>>>>> Stashed changes
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
    eFile = datFile[:-9] +'-res'+str(float(resol))+'sec.txt'
    print(eFile)

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
        print(datFile)
        with open(datFile) as f:
            segments = json.load(f)
        for seg in segments:
            if seg[0] == -1:
                continue
            #ORIGINAL Version
            #virginia: changed because I had problem on this
            #if not species.title() in seg[4]:
                #continue
            #elif seg[4]!=['Noise']:
            #if "Morepork" in str(seg[4]):
            if species in str(seg[4]):
            #else:
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
                    elif '(Trill)' in str(seg[4][0]):
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

<<<<<<< Updated upstream
#Virginia:change directory name
# genGT('D:\AviaNZ\Sound Files\Fiordland kiwi\Dataset\\Negative',species='Kiwi(Tokoeka Fiordland)',window=1, inc=1)
=======
def splitGT(dirName, window=1, inc=None):

#From Nirosha

    # Virginia: set increment and resolution
    if inc==None:
        inc=window
        resol=window
    else:
    # Virginia: resolution is the "gcd" between window and inc. In this way I'm hoping to solve the case with
    # 75% overlap
        resol=(math.gcd(int(100*window),int(100*inc)))/100
    slot1=int(math.ceil(300/resol))
    slot2=int(math.ceil(600/resol))
    slot3=int(math.ceil(900/resol))
    new_dir='D:\Desktop\Documents\Work\Data\Filter experiment\Ruru\Test-5min'
    for root, dirs, files in os.walk(str(dirName)):
        for file in files:
            if file.endswith('.txt'):
                filename = root + '/' + file
                filename2= new_dir + '/'+file
                fileAnnotations = []
                # Get the segmentation from the txt file
                f = open(filename, "r")
                if resol==0.25:
                     f1 = filename2[:-15] + '_0' + filename2[-15:]
                     f2 = filename2[:-15] + '_1' + filename2[-15:]
                     f3 = filename2[:-15] + '_2' + filename2[-15:]
                else:
                    f1 = filename2[:-14] + '_0' + filename2[-14:]
                    f2 = filename2[:-14] + '_1' + filename2[-14:]
                    f3 = filename2[:-14] + '_2' + filename2[-14:]
                f1out = open(f1, 'w')
                f2out = open(f2, 'w')
                f3out = open(f3, 'w')
                i = 0
                for line in f:
                    #if i<300:
                    if i<slot1:
                        f1out.write(line)
                    #elif i<600:
                    elif i<slot2:
                        f2out.write(line)
                    #elif i<900:
                    elif i<slot3:
                        f3out.write(line)
                    i = i+1
                f1out.close()
                f2out.close()
                f3out.close()


#Virginia:change directory name 
#genGT('/home/listanvirg/Data/Filter experiment/Ruru',species='Morepork',window=1)
genGT('/home/listanvirg/Data/Filter experiment/BKiwi/Ponui',species='Kiwi',window=4, inc=3)
#genGT('D:\Desktop\Documents\Work\Data\Filter experiment\Ruru GT\Test',species='Morepork',window=0.5, inc=0.25)
#splitGT('D:\Desktop\Documents\Work\Data\Filter experiment\Ruru GT\Test',window=0.5, inc=0.25)

>>>>>>> Stashed changes

