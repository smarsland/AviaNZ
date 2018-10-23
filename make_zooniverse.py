# Processing to create zooniverse images and sound files

# Reads a data file, gets the appropriate times for 10 second sound files, clips those and saves them
# See also associated shell file

import wavio
import math
import numpy as np
import json
import os
import click

def loadFile(filename):

    # Load any previous segments stored
    if os.path.isfile(filename + '.data'):
        file = open(filename + '.data', 'r')
        segments = json.load(file)
        file.close()
        if len(segments) > 0:
            if segments[0][0] == -1:
                del segments[0]
        else:
            return None, None, 0, 0, 0, 0
    else:
        return None, None, 0, 0, 0, 0

    if os.stat(filename).st_size != 0: # avoid files with no data (Tier 1 has 0Kb .wavs)
        wavobj = wavio.read(filename)

        # Parse wav format details based on file header:
        sampleRate = wavobj.rate
        audiodata = wavobj.data
        minFreq = 0
        maxFreq = sampleRate / 2.
        fileLength = wavobj.nframes

        if audiodata.dtype is not 'float':
            audiodata = audiodata.astype('float')  # / 32768.0

        if np.shape(np.shape(audiodata))[0] > 1:
            audiodata = audiodata[:, 0]
            datalength = np.shape(audiodata)[0]
            datalengthSec = datalength / sampleRate
            #print("Length of file is ", datalengthSec, " seconds (", datalength, "samples) loaded from ", fileLength / sampleRate, "seconds (", fileLength, " samples) with sample rate ",sampleRate, " Hz.")

        return segments, audiodata, sampleRate, minFreq, maxFreq, datalength

def save_selected_sound(audiodata,sampleRate,t1,t2,filename):
        # t1, t2 in seconds
        t1 = math.floor(t1 * sampleRate)
        t2 = math.floor(t2 * sampleRate)
        wavio.write(str(filename) + '.wav', audiodata[int(t1):int(t2)].astype('int16'), sampleRate, scale='dtype-limits', sampwidth=2)

@click.command()
@click.option('-s', '--species', type=str, help='Species')
@click.option('-i', '--infile', type=click.Path(), help='Input folder')
@click.option('-o', '--outfile', type=click.Path(), help='Output folder')
def make_zooniverse(species,infile,outfile):

# Load datafile
#species = "Kiwi, Nth Is Brown"
#infile = "Sound Files/Batch/"
#outfile = "TestOut/KNIB"

    files = [f for f in os.listdir(infile) if f[-4:]=='.wav']

    filecount = 0
    for f in files:
        segmentcount = 0
        segments, audiodata, sampleRate, minFreq, maxFreq, datalength = loadFile(infile+f)

        if segments is None:
            pass
        else:
            for s in segments:
                # Check for segments with correct label
                if species in s[4] or species+'?' in s[4]:
                    if s[1] - s[0] < 10:
                        # If segment is less than 10s, put it evenly in the middle
                        excess = (10 - s[1] + s[0]) / 2
                        if s[0] - excess < 0:
                            t1 = 0
                            t2 = 10
                        else:
                            t1 = s[0] - excess
                            t2 = s[1] + excess
                
                        filename = outfile+"_"+str(filecount)+"_"+str(segmentcount)
                        save_selected_sound(audiodata,sampleRate,t1,t2,filename)
                        segmentcount += 1
                    else:
                        # Otherwise, take the first 10s and the last 10s as 2 segments
                        filename = outfile+"_"+str(filecount)+"_"+str(segmentcount)
                        save_selected_sound(audiodata,sampleRate,s[0],s[0]+10,filename)
                        filename = outfile+"_"+str(filecount)+"_"+str(segmentcount+1)
                        save_selected_sound(audiodata,sampleRate,s[1]-10,s[1],filename)
                        segmentcount += 2
    
        filecount += 1

make_zooniverse()
