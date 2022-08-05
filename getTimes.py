
# Given a folder that contains more folders birdID/year/... 
# use the data file to get the length of each syllable and each pause
# and save as a csv file

#import SignalProc 
import Segment
import os
import string, math
#import wavio
import numpy as np
import pylab as pl
#from ext import ce_denoise
import csv

def getTimes(path):
    birds = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.isdigit()]
    print(birds)
    extracted = os.path.join(path,"extracted")

    # For each folder
    times  = []
    for bird in birds:
        birddir = os.path.join(path,bird)
        # For each year
        years = [d for d in os.listdir(birddir) if os.path.isdir(os.path.join(birddir, d))]
        for year in years:
            # For each file
            yeardir = os.path.join(birddir,year)
            files = [f for f in os.listdir(yeardir) if f.lower().endswith(".data")]
            #print(files)
            callID = 0
            for call in files:
                filename = os.path.join(yeardir,call)
                segments = Segment.SegmentList()
                print(filename)
                if os.path.isfile(filename):
                    segments.parseJSON(filename)
                segments.orderTime()
    
                #print(segments)
                # Extract each syllable
                syllID = 0
                a = [bird+"_"+year+"_"+str(callID)]
                a.append(call)
                end = 0
                for s in segments:
                    print(s)
                    if end>0:
                        a.append(s[0]-end)
                        a.append(s[1]-s[0])
                    else:
                        a.append(s[1]-s[0])
                    end = s[1]
                
                    syllID += 1
                times.append(a)

                callID += 1

    # Choose name
    name = os.path.join(extracted,"times.csv")
    with open(name, 'w') as f:
        write = csv.writer(f)
        write.writerows(times)
    
    
    
getTimes("/home/marslast/Dropbox/Kiwi_IndividualID")
