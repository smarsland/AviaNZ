# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 21:52:42 2020

@author: Virginia Listanti

Script to recover how many files we are actually using for CNN training
"""


import json
import numpy as np
import wavio
import SignalProc
import math
import Segment
import pyqtgraph as pg
import pyqtgraph.exporters as pge
import os

#tensorflow libraries
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims

import librosa
import WaveletSegment
import WaveletFunctions

import cv2  # image -processing
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def ClickSearch(dirName, file, featuress, count, Train=False):
    """
    ClickSearch search for clicks into file in directory dirName, saves 
    dataset, and return click_label and dataset
    
    The search is made on the spectrogram image that we know to be generated 
    with parameters (1024,512)
    Click presence is assested for each spectrogram column: if the mean in the
    frequency band [33000, 55000] (*) is bigger than a treshold we have a click
    thr=mean(all_spec)+std(all_spec) (*)
    
    The clicks are discarded if longer than 0.05 sec
    
    Clicks are stored into featuress using updateDataset or updateDataset2
    
    """
    
    print("Click search on ",file)
    filename = dirName + '\\' + file
    
    img = mpimg.imread(filename) #read image
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    img2[-1, :] = 254 * np.ones((np.shape(img2[1]))) #cut last row
    imspec=np.repeat(img2,8, axis=0) #repeat rows 7 times to fit invertspectrogram
    imspec = -(imspec - 254.0 * np.ones(np.shape(imspec)))  # reverse value having the black as the most intense
#    imspec=np.flipud(imspec) #reverse up and down of the spectrogram -> see AviaNZ spectrogram
    imspec = imspec/np.max(imspec) #normalization
    imspec = imspec[:, 1:np.shape(img2)[1]]  # Cutting first column because it only contains the scale and cutting last columns
    
#    #Read audiodata
#   keeping this just to read duration -> HACK
    fs=16000
    audiodata = wavio.read(filename[:-3]+'wav')
    sp = SignalProc.SignalProc(1024, 512) #outside?
    sp.data = audiodata.data
    duration=audiodata.nframes/fs
    
    #copyed from sp.wavRead to make everything consistent
    # take only left channel
    if np.shape(np.shape(sp.data))[0] > 1:
        sp.data = sp.data[:, 0]
    sp.audioFormat.setChannelCount(1)
     # force float type
    if sp.data.dtype != 'float':
        sp.data = sp.data.astype('float')
    sp.audioFormat.setSampleSize(audiodata.sampwidth * 8)
    
#    #Spectrogram
#    sp.samplerate= fs
#    sgraw= sp.spectrogram(1024, 512, 'Blackman')
#    imspec=(10.*np.log10(sgraw)).T #transpose
#    imspec=np.flipud(imspec) #updown 

    df=88000/(np.shape(imspec)[0]+1) #frequency increment 
    dt=(duration/11)/(np.shape(imspec)[1]+1) #timeincrement
#    print("file ", file, "dt " , dt)
#    dt=0.002909090909090909
    up_len=math.ceil(0.05/dt) #0.5 second lenth in indices divided by 11
#    up_len=17
#    up_len=math.ceil((0.5/11)/dt)
    
    
    #Frequency band
#    f0=33000
    f0=24000
    index_f0=-1+math.floor(f0/df) #lower bound needs to be rounded down
#    print(f0,index_f0)
#    f1=55000
    f1=54000
    index_f1=-1+math.ceil(f1/df) #upper bound needs to be rounded up
    
    #Mean in the frequency band
    mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0) #added 0.01 to avoid divition by 0

    #Threshold
    mean_spec_all=np.mean(imspec, axis=0)[2:]
    thr_spec=(np.mean(mean_spec_all)+np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))
    
    ##clickfinder
    #check when the mean is bigger than the threshold
    #clicks is an array which elements are equal to 1 only where the sum is bigger 
    #than the mean, otherwise are equal to 0
    clicks=np.where(mean_spec>thr_spec,1,0)
    clicks_indices=np.nonzero(clicks)
    print(np.shape(clicks_indices))
    #check: if I have found somenthing
    if np.shape(clicks_indices)[1]==0:
        #If not: label = None 
        click_label='None'
        #check if I need to return something different
        return click_label, featuress, count
        #not saving spectrograms
    
#    DIscarding segments too long or too shorts and saving spectrogram images
    
    #Read annotation file: if in Train mode
    if Train==True:
        annotation_file=filename[:-3] +'wav.data'
        if os.path.isfile(annotation_file):
            segments = Segment.SegmentList()
            segments.parseJSON(annotation_file)
            thisSpSegs = np.arange(len(segments)).tolist()
        else:
            segments=[]
            thisSpSegs=[]
    else:
        segments=[]
        thisSpSegs=[]
 
#    DIscarding segments too long or too shorts and saving spectrogram images        
    click_start=clicks_indices[0][0]
    click_end=clicks_indices[0][0]  
    for i in range(1,np.shape(clicks_indices)[1]):
        if clicks_indices[0][i]==click_end+1:
            click_end=clicks_indices[0][i]
        else:
            if click_end-click_start+1>up_len:
                clicks[click_start:click_end+1]=0
            else:
                #savedataset
                featuress, count=updateDataset(file, dirName, featuress, count, imspec,  segments, thisSpSegs, click_start, click_end, dt, Train)
            #update
            click_start=clicks_indices[0][i]
            click_end=clicks_indices[0][i] 
                              
    #checking last loop with end
    if click_end-click_start+1>up_len:
        clicks[click_start:click_end+1]=0
    else:
        featuress, count = updateDataset(file, dirName, featuress, count, imspec, segments, thisSpSegs, click_start, click_end, dt, Train)
        
    #updating click_inidice
    clicks_indices=np.nonzero(clicks)
    
    #Assigning: click label
    if np.shape(clicks_indices)[1]==0:
        click_label='None'
    else:
        click_label='Click'
    
    return click_label, featuress, count


def updateDataset(file_name, dirName, featuress, count, spectrogram, segments, thisSpSegs, click_start, click_end, dt=None, Train=False):
    """
    Update Dataset with current segment
    It take a piece of the spectrogram with fixed length centered in the
    click 
    
    We are using AviaNZ annotations but are  reading click_start and click_end
    in seconds multiplicating by 11
    
    TRAIN MODE => stores the lables as well
    A spectrogram is labeled is the click is inside a segment
    We have 3 labels:
        0 => LT
        1 => ST
        2 => Noise
    """
    #I assign a label t the spectrogram only for Train Dataset
    # we convert evereting to Avianz time scale
    click_start_sec=(click_start*dt)*11
    click_end_sec=(click_end*dt)*11
    if Train==True:
        assigned_flag=False #control flag
        for segix in thisSpSegs:
            seg = segments[segix]
            if isinstance(seg[4][0], dict):
#                print('UpdateDataset Check')
                if seg[0]<=click_start_sec and seg[1]>=click_end_sec:
                    if 'Bat (Long Tailed)' == seg[4][0]["species"]:
                        spec_label = 0
                        assigned_flag=True
                        break
                    elif 'Bat (Short Tailed)' == seg[4][0]["species"]:
                        spec_label = 1
                        assigned_flag=True
                        break
                    elif 'Noise' == seg[4][0]["species"]:
                        spec_label = 2    
                        assigned_flag=True
                        break
                    else:
                        continue
                    
            elif isinstance(seg[4][0], str):
                # old format
                print('UpdateDataset Check')
                if seg[0]<=click_start_sec and seg[1]>=click_end_sec:
                    if 'Bat (Long Tailed)' == seg[4][0]["species"]:
                        spec_label = 0
                        assigned_flag=True
                        break
                    elif 'Bat (Short Tailed)' == seg[4][0]["species"]:
                        spec_label = 1
                        assigned_flag=True
                        break
                    elif 'Noise' == seg[4][0]["species"]:
                        spec_label = 2   
                        assigned_flag=True
                        break
                    else:
                        continue
        if assigned_flag==False:
            spec_label=2
    
# slice spectrogram   

    win_pixel=1 
    ls = np.shape(spectrogram)[1]-1
    click_center=int((click_start+click_end)/2)

    start_pixel=click_center-win_pixel
    if start_pixel<0:
        win_pixel2=win_pixel+np.abs(start_pixel)
        start_pixel=0
    else:
        win_pixel2=win_pixel
    
    end_pixel=click_center+win_pixel2
    if end_pixel>ls:
        start_pixel-=end_pixel-ls+1
        end_pixel=ls-1
        #this code above fails for sg less than 4 pixels wide   
    sgRaw=spectrogram[:,start_pixel:end_pixel+1] #not I am saving the spectrogram in the right dimension
    sgRaw=np.repeat(sgRaw,2,axis=1)
    sgRaw=(np.flipud(sgRaw)).T #flipped spectrogram to make it consistent with Niro Mewthod
    if Train==True:
        featuress.append([sgRaw.tolist(), file_name, count, spec_label])
    else:
        featuress.append([sgRaw.tolist(), file_name, count]) #not storing segment and label informations

    count += 1

    return featuress, count


##MAIN
 
#Create train dataset for CNN from the results of clicksearch   
train_dir = "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TRAIN4" #changed directory
annotation_file_train= "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TRAIN4\\Train_dataset_images.data"
with open(annotation_file_train) as f:
    segments_filewise_train = json.load(f)
file_number_train=np.shape(segments_filewise_train)[0]

#inizializations of counters
count=0
train_featuress =[]
TD=0
FD=0
TND=0
FND=0

effective_number=0

#search clicks
for i in range(file_number_train):
    file = segments_filewise_train[i][0]
    click_label, train_featuress, count = ClickSearch(train_dir, file, train_featuress, count, Train=True)
    if click_label=='Click':
        effective_number+=1

print('Number of file actually used out of ', file_number_train, ' = ', effective_number)