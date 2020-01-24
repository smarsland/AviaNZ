# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:59:38 2019

@author: Virginia Listanti
"""
#This script is intended to transform the segment annotations stored in .data 
# files into file wise annotations stored into one .data file

#Labels:
# LT
# ST
# Both 
# None if empty or only Noise annotations 


import json
#
#import SignalProc as sp
#import pylab as pl
import numpy as np
#import cv2  # image -processing
import os  # linux shell comands
#from scipy import misc
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import wavio
#import math
import Segment

#dirName='/home/listanvirg/Data/Bat/BAT/TRAIN_DATA/LT'
dirName='D:\Desktop\Documents\Work\Data\Bat\BAT\CNN experiment\TEST2'

dataset=[]
for root, dirs, files in os.walk(str(dirName)):
    for file in files:
        #Work on .wav data
        if file.endswith('.bmp'):
            sound_file=root+'/'+file[:-3]
#            print(sound_file)
            if file[:-3] +'wav.data' in files: 
                annotation_file=sound_file+'wav.data'
#                print(annotation_file)
                segments = Segment.SegmentList()
                segments.parseJSON(annotation_file)
                thisSpSegs = np.arange(len(segments)).tolist()
                # Now find syllables within each segment, median clipping
                label='Noise' #inizialization
                for segix in thisSpSegs:
                    seg = segments[segix]

                    # Find the GT label for the syllables from this segment
                    # 0 -Bat (Long Tailed)
                    # 1 - Bat (Short Tailed)
                    if isinstance(seg[4][0], dict):
                        # new format
                        if 'Bat (Long Tailed)' == seg[4][0]["species"]:
                            if label=='Noise':
                                label = 'LT'
                            elif label=='ST':
                                label= 'Both'
                                break
                        elif 'Bat (Short Tailed)' == seg[4][0]["species"]:
                            if label=='Noise':
                                label = 'ST'
                            elif label=='LT':
                                label= 'Both' 
                                break
                        else:
                            continue
                    elif isinstance(seg[4][0], str):
                        # old format
                       if 'Bat (Long Tailed)' == seg[4][0]["species"]:
                            if label=='Noise':
                                label = 'LT'
                            elif label=='ST':
                                label= 'Both'
                                break
                       elif 'Bat (Short Tailed)' == seg[4][0]["species"]:
                            if label=='Noise':
                                label = 'ST'
                            elif label=='LT':
                                label= 'Both' 
                                break
                       else:
                            continue
            else:
                label='Noise'
#                dataset.append([os.path.join(root, file), label])
            dataset.append([file, label])
#check on dataset
print ('number of file',len(dataset)) 
print(dataset[0][0],dataset[0][1])               
#save dataset                
with open(dirName+'\Test_dataset_images.data', 'w') as f2:
    json.dump(dataset,f2)
            
            
            
            
            
            

