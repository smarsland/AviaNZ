#THIS SCRIPT converts a data file with samplerate 1 to a data file with samplerate2

#Author: Virginia Listanti

import json

import SignalProc as sp
import pylab as pl
import numpy as np
import cv2  # image -processing
import os  # linux shell comands
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wavio
import math

#dirName='/home/listanvirg/Data/Bat/BAT/TRAIN_DATA/LT'
# dirName='D:\Desktop\Documents\Work\Data\Bat\BAT\TRAIN_DATA\ST'
dirName='D:\Desktop\Documents\Work\Zohara files\TEST'
samplerate1=8000
samplerate2=4000
k=samplerate1/samplerate2

for root, dirs, files in os.walk(str(dirName)):
    for file in files:
        #works directly on all the data files in the folder
        if file.endswith('.wav.data'):
            annotation_file=root+'/'+file
            print(annotation_file)
            with open(annotation_file) as f2:
                segments = json.load(f2)
                for seg in segments:
                    print(seg)
                    if "Operator" in seg:
                        print(seg['Duration'])
                        seg['Duration'] = seg['Duration']*k
                    elif seg[0]==-1:
                        seg[1] = seg[1]*k
                    else:
                        #print(seg[0])
                        seg[0]=seg[0]*k
                        #print(seg[0])
                        seg[1] = seg[1]*k
                        #rint(seg[2])
                        seg[2] = seg[2] / k
                        #print(seg[2])
                        seg[3] = seg[3]/ k
            new_annotation_file=annotation_file[:-9]+'_timeexpanded'+str(samplerate2)+'.wav.data'
            with open(new_annotation_file, 'w') as f2:
                json.dump(segments,f2)
