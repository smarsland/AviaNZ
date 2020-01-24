# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:05:05 2019

@author: Virginia Listanti
"""

# Bat Search looks for clicks in files and then use a CNN to classify files
# as LT Bat, ST bat, Both or None.

#Part of the code is based on Nirosha Priyadarshani scripts makeTrainingData,py
# and CNN_keras

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

import librosa
import WaveletSegment
import WaveletFunctions

import cv2  # image -processing
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def ClickSearch(dirName, file, fs, featuress, count, Train=False):
    """
    ClickSearch search for clicks into file in directory dirName, saves 
    dataset, and return click_label and dataset
    
    The search is made on the spectrogram image generated with parameters
    (1024,512)
    Click presence is assested for each spectrogram column: if the mean in the
    frequency band [3000, 5000] (*) is bigger than a treshold we have a click
    thr=mean(all_spec)+std(all_spec) (*)
    
    The clicks are discarded if longer than 0.5 sec
    
    Clicks are stored into featuress using updateDataset
    
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
    imspec = imspec[:, 1:np.shape(img2)[1]]  # Cutting first column because it only contains the scale
    
#    #Read audiodata
#   keeping this just to read duration -> HACK
    audiodata = wavio.read(filename[:-3]+'wav')
#    sp = SignalProc.SignalProc(1024, 512) #outside?
#    sp.data = audiodata.data
    duration=audiodata.nframes/fs
#    
#    #copyed from sp.wavRead to make everything consistent
#    # take only left channel
#    if np.shape(np.shape(sp.data))[0] > 1:
#        sp.data = sp.data[:, 0]
#    sp.audioFormat.setChannelCount(1)
#     # force float type
#    if sp.data.dtype != 'float':
#        sp.data = sp.data.astype('float')
#    sp.audioFormat.setSampleSize(audiodata.sampwidth * 8)
#    
#    #Spectrogram
#    sp.samplerate= fs
#    sgraw= sp.spectrogram(1024, 512, 'Blackman')
#    imspec=(10.*np.log10(sgraw)).T #transpose
#    imspec=np.flipud(imspec) #updown 

    df=88000/(np.shape(imspec)[0]+1) #frequency increment 
    dt=duration/(np.shape(imspec)[1]+1) #timeincrement
    up_len=math.ceil(0.5/dt) #0.5 second lenth in indices  
    
    #Frequency band
    f0=3000
    index_f0=-1+math.floor(f0/df) #lower bound needs to be rounded down
#    print(f0,index_f0)
    f1=5000
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
        annotation_file=filename +'.data'
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
                featuress, count=updateDataset3(file, dirName, featuress, count, imspec,  segments, thisSpSegs, click_start, click_end, dt, Train)
                #update
                click_start=clicks_indices[0][i]
                click_end=clicks_indices[0][i] 
                              
    #checking last loop with end
    if click_end-click_start+1>up_len:
        clicks[click_start:click_end+1]=0
    else:
        featuress, count = updateDataset3(file, dirName, featuress, count, imspec, segments, thisSpSegs, click_start, click_end, dt, Train)
        
    #updating click_inidice
    clicks_indices=np.nonzero(clicks)
    
    #Assigning: click label
    if np.shape(clicks_indices)[1]==0:
        click_label='None'
    else:
        click_label='Click'
    
    return click_label, featuress, count

def ClickSearch2(dirName, file, fs, featuress, count, Train=False):
    """
    ClickSearch search for clicks into file in directory dirName, saves 
    dataset, and return click_label and dataset
    
    The search is made on the spectrogram image generated with parameters
    (1024,512)
    Click presence is assested for each spectrogram column: if the mean in the
    frequency band [3000, 5000] (*) is bigger than a treshold we have a click
    thr=mean(all_spec) (*)
    
    The clicks are discarded if longer than 0.5 sec
    
     Clicks are stored into featuress using updateDataset
    
    """
    
    print("Click search on ",file)
    filename = dirName + '\\' + file
    
    #Read audiodata
    audiodata = wavio.read(filename)
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
    
    #Spectrogram
    sp.samplerate= fs
    sgraw= sp.spectrogram(1024, 512, 'Blackman')
    imspec=(10.*np.log10(sgraw)).T #transpose
    imspec=np.flipud(imspec) #updown 

    df=16000/(np.shape(imspec)[0]+1) #frequency increment 
    dt=duration/(np.shape(imspec)[1]+1) #timeincrement
    up_len=math.ceil(0.5/dt) #0.5 second lenth in indices  
    
    #Frequency band
    f0=3000
    index_f0=-1+math.floor(f0/df) #lower bound needs to be rounded down
    f1=5000
    index_f1=-1+math.ceil(f1/df) #upper bound needs to be rounded up
    
    #Mean in the frequency band
    mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0) #added 0.01 to avoid divition by 0
    #Threshold
    mean_spec_all=np.mean(imspec, axis=0)[2:]
#    thr_spec=(np.mean(mean_spec_all)+np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))
    thr_spec=np.mean(mean_spec_all)*np.ones((np.shape(mean_spec)))
    
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
        annotation_file=filename +'.data'
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
                featuress, count=updateDataset3(file, dirName, featuress, count, imspec,  segments, thisSpSegs, click_start, click_end, dt, Train)
                #update
                click_start=clicks_indices[0][i]
                click_end=clicks_indices[0][i] 
                              
    #checking last loop with end
    if click_end-click_start+1>up_len:
        clicks[click_start:click_end+1]=0
    else:
        featuress, count = updateDataset3(file, dirName, featuress, count, imspec, segments, thisSpSegs, click_start, click_end, dt, Train)
        
    #updating click_inidice
    clicks_indices=np.nonzero(clicks)
    
    #Assigning: click label
    if np.shape(clicks_indices)[1]==0:
        click_label='None'
    else:
        click_label='Click'
    
    return click_label, featuress, count

def updateDataset(file_name, dirName, featuress, count, spectrogram, segments, thisSpSegs, click_start, click_end, dt, Train=False):
    """
        Update Dataset with current segment
        It slices the current segments into smaller spectrograms and stores them
        into featuress.
        
        TRAIN MODE => stores the lables as well
         A spectrogram is labeled is the click is inside a segment
        We have 3 labels:
            0 => LT
            1 => ST
            2 => Noise
    """
    #I assign a label t the spectrogram only for Train Dataset
    click_start_sec=click_start*dt
    click_end_sec=click_end*dt
    
    #checking annotations in order to assignlabel
    if Train==True:
        assigned_flag=False #control flag
        for segix in thisSpSegs:
            seg = segments[segix]
            
            if isinstance(seg[4][0], dict):
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
    win_pixel=2
    duration = click_end -click_start +1
    if duration > win_pixel:
        
        n = math.ceil(duration/win_pixel)
    #         inizialization
        start_pixel=click_start

        for i in range(n):
            end_pixel=start_pixel+win_pixel
            sgRaw=spectrogram[:,start_pixel:end_pixel] #not I am saving the spectrogram in the right dimension
            sgRaw=np.repeat(sgRaw,6,axis=1)
            sgRaw=(np.flipud(sgRaw)).T #flipped spectrogram to make it consistent with Niro Mewthod
            if Train==True:
                featuress.append([sgRaw.tolist(), file_name, count, spec_label])
            else:
         #if testing: do not save label
                featuress.append([sgRaw.tolist(), file_name, count]) 
            start_pixel=end_pixel
            count += 1
         
    return featuress, count
            
def updateDataset2(file_name, dirName, featuress, count, spectrogram, segments, thisSpSegs, click_start, click_end, dt, Train=False):
    """
    Update Dataset with current segment
    It take a piece of the spectrogram with fixed length starting from the
    click start
    
    TRAIN MODE => stores the lables as well
     A spectrogram is labeled is the click is inside a segment
    We have 3 labels:
        0 => LT
        1 => ST
        2 => Noise
    """
     #I assign a label t the spectrogram only for Train Dataset
    click_start_sec=click_start*dt
    click_end_sec=click_end*dt
    if Train==True:
        assigned_flag=False #control flag
        for segix in thisSpSegs:
            seg = segments[segix]
            
            if isinstance(seg[4][0], dict):
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

    win_pixel=3
    ls = np.shape(spectrogram)[1]-1
    duration=click_end-click_start+1
    #n number of spectrograms we can get from the detected clicks
    if duration<2*win_pixel+1:
        n=1 #trick to get ate least one spectrogram
    else:
        n=math.ceil(duration/(2*win_pixel+1))
    
    for i in range(n):
        start_pixel=click_start-win_pixel
        if start_pixel<0:
            win_pixel2=win_pixel+np.abs(start_pixel)
            start_pixel=0
        else:
            win_pixel2=win_pixel
        
        end_pixel=click_start+win_pixel2
        if end_pixel>ls:
            start_pixel-=end_pixel-ls+1
            end_pixel=ls-1
        if end_pixel-start_pixel != 6:
            print("*******************************************",end_pixel,start_pixel)
            #this code above fails for sg less than 5 pixels wide
    
        sgRaw=spectrogram[:,start_pixel:end_pixel+1] #not I am saving the spectrogram in the right dimension
        #sgRaw=np.repeat(sgRaw,6,axis=1)
        sgRaw=(np.flipud(sgRaw)).T #flipped spectrogram to make it consistent with Niro Mewthod
        if Train==True:
            featuress.append([sgRaw.tolist(), file_name, count, spec_label])
        else:
     #if testing: do not save label
            featuress.append([sgRaw.tolist(), file_name, count]) #not storing segment and label informations
        
        click_start+=win_pixel2*2+1
        count += 1

    return featuress, count

def updateDataset3(file_name, dirName, featuress, count, spectrogram, segments, thisSpSegs, click_start, click_end, dt=None, Train=False):
    """
    Update Dataset with current segment
    It take a piece of the spectrogram with fixed length centered in the
    click 
    
    TRAIN MODE => stores the lables as well
    A spectrogram is labeled is the click is inside a segment
    We have 3 labels:
        0 => LT
        1 => ST
        2 => Noise
    """
    #I assign a label t the spectrogram only for Train Dataset
    
    if Train==True:
        click_start_sec=click_start*dt
        click_end_sec=click_end*dt
        assigned_flag=False #control flag
        for segix in thisSpSegs:
            seg = segments[segix]
            if isinstance(seg[4][0], dict):
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

    win_pixel=4 
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
#    if end_pixel-start_pixel != 10:
#        print("*******************************************",end_pixel,start_pixel)
        #this code above fails for sg less than 4 pixels wide   
    sgRaw=spectrogram[:,start_pixel:end_pixel+1] #not I am saving the spectrogram in the right dimension
#    sgRaw=np.repeat(sgRaw,2,axis=1)
    sgRaw=(np.flipud(sgRaw)).T #flipped spectrogram to make it consistent with Niro Mewthod
    if Train==True:
        featuress.append([sgRaw.tolist(), file_name, count, spec_label])
    else:
        featuress.append([sgRaw.tolist(), file_name, count]) #not storing segment and label informations

    count += 1

    return featuress, count


def updateDataset4(file_name, dirName, featuress, count, spectrogram, segments, thisSpSegs, click_start, click_end, dt, Train=False):
    """
    Update Dataset with current segment
    It take a piece of the spectrogram with fixed length centered in the
    click 
    
    TRAIN MODE => stores the lables as well
    A spectrogram is labeled is the click it overlap a segment
    We have 3 labels:
        0 => LT
        1 => ST
        2 => Noise
    """
    #I assign a label t the spectrogram only for Train Dataset
    click_start_sec=click_start*dt
    click_end_sec=click_end*dt
    if Train==True:
        assigned_flag=False #control flag
        for segix in thisSpSegs:
            seg = segments[segix]
            
            if isinstance(seg[4][0], dict):
                if (seg[0]<=click_start_sec and seg[1]>=click_end_sec) or (seg[0]<=click_start_sec and seg[1]>=click_start_sec) or  (seg[1]>=click_end_sec and seg[0]<=click_end_sec):
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
                if  (seg[0]<=click_start_sec and seg[1]>=click_end_sec) or (seg[0]<=click_start_sec and seg[1]>=click_start_sec) or  (seg[1]>=click_end_sec and seg[0]<=click_end_sec):
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
    #find click center
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
#    if end_pixel-start_pixel != 10:
#        print("*******************************************",end_pixel,start_pixel)
        #this code above fails for sg less than 4 pixels wide
#    print(start_pixel, end_pixel)    
    sgRaw=spectrogram[:,start_pixel:end_pixel+1] #not I am saving the spectrogram in the right dimension
    sgRaw=np.repeat(sgRaw,2,axis=1)
    sgRaw=(np.flipud(sgRaw)).T #flipped spectrogram to make it consistent with Niro Mewthod
    if Train==True:
        featuress.append([sgRaw.tolist(), file_name, count, spec_label])
    else:
 #if testing: do not save label
        featuress.append([sgRaw.tolist(), file_name, count]) #not storing segment and label informations

    count += 1

    return featuress, count

#First Option: 3 lables, arg max * majority vote
def File_label(predictions, spec_id, segments_filewise_test, filewise_output, file_number ):
    """
    FIle_label use the predictions made by the CNN to update the filewise annotations
    when we have 3 labels: 0 (LT), 1(ST), 2 (Noise)
    
    METHOD: ARGMAX probabilities labels + majority vote
        Majority Vote:
            if No LT or ST => Noise
            if  LT>70% of good spectrogram in file => LT
            if  ST>70% of good spectrogram in file => ST
            otherwise => Both
        
    """
    
    predicted_label= np.argmax(predictions,axis=1)
    if len(predicted_label)!=np.shape(spec_id)[0]:
        print('ERROR: Number of labels is not equal to number of spectrograms' )
    
    # Assesting file label and updating metrics  
    for i in range(file_number):
        file = segments_filewise_test[i][0]
        #inizializing counters
        LT_count=0
        ST_count=0
        Other_count=0
        spec_num=0   #counts number of spectrograms per file
        #flag: if no click detected no spectrograms
        click_detected_flag=False
#        looking for all the spectrogram related to this file
        #count majority
        for k in range(np.shape(spec_id)[0]):
            if spec_id[k][0]==file:
                click_detected_flag= True
                if predicted_label[k]==0:
                    LT_count+=1
                elif predicted_label[k]==1:
                    ST_count+=1
                else:
                    Other_count+=1
                spec_num+=1
                
        #assign label to file
        if click_detected_flag==True:
            #this makes sense only if there were spectrograms
#            keeping differen majority vote options
    #        if Other_count>LT_count+ST_count:
    #        if (Other_count/spec_num)*100>90:
            if LT_count+ST_count==0:
                label='Noise'
            else:
                LT_perc=(LT_count/(spec_num-Other_count))*100 #percentage of LT over "good clicks" clicks
                ST_perc=(ST_count/(spec_num-Other_count))*100 #percentage of LT over "good clicks" clicks
                if LT_perc>70:
                    label='LT'
                elif ST_perc>70:
                    label='ST'
                else:
                    label='Both'
        else:
            #if no click automatically we have Noise
            label='Noise'
        filewise_output[i][3] = label

    return filewise_output   


#2nd option: 3 labels, evaluate it filewise
def File_label2(predictions, spec_id, segments_filewise_test, filewise_output, file_number ):
    """
    FIle_label2 use the predictions made by the CNN to update the filewise annotations
    when we have 3 labels: 0 (LT), 1(ST), 2 (Noise)
    
    METHOD: evaluation of probability over files
        P(2)>50% => Noise
        P(0)>70 => LT
        P(1)>70 => ST
        else => Both
    
     TODO: how can I had possible?
    """
   
    
    if len(predictions)!=np.shape(spec_id)[0]:
        print('ERROR: Number of labels is not equal to number of spectrograms' )
    
    # Assesting file label
    for i in range(file_number):
        file = segments_filewise_test[i][0]
        file_prob=np.zeros((1,3))
        spec_num=0   #counts number of spectrograms per file
        #flag: if no click detected no spectrograms
        click_detected_flag=False
        #        looking for all the spectrogram related to this file

        for k in range(np.shape(spec_id)[0]):
            if spec_id[k][0]==file:
                click_detected_flag=True
                spec_num+=1
                file_prob[0][:]+=predictions[k][:]
#                print('check file_prob',file_prob)
        if click_detected_flag==True:
            file_prob/=spec_num
            file_prob*=100
            if file_prob[0][2]>90:
            #if file_prob[0][0]<5 and file_prob[0][1]<5:
                label='Noise'
            elif file_prob[0][0]-file_prob[0][1]>10:
                label='LT'
            elif file_prob[0][1]-file_prob[0][0]>10:
                label='ST'
#            elif file_prob[0][0]>80:
#                label='LT'
#            elif file_prob[0][1]>80:
#                label='ST'
            else:
                label='Both'
            
        else:
#            if no clicks => automatically Noise
            label='Noise'
            
        print('check file_prob',file_prob)
        filewise_output[i][3] = label
        
    return filewise_output

#3rd option: 2 lables evaluate it foe each spectrogram
def File_label3(predictions, spec_id, segments_filewise_test, filewise_output, file_number ):
    """
    FIle_label2 use the predictions made by the CNN to update the filewise annotations
    when we have 2 labels: 0 (LT), 1 (ST)
    
    METHOD: evaluation of probability over files, we work directly with predictions
        1) Assign spectrogram Label
            P(0)>0.5 => LT
            P(1)>0.5 => ST
            otherwise Noise
    
        2) Majority vote:
            if No LT or ST => Noise
            if  LT>70% of good spectrogram in file => LT
            if  ST>70% of good spectrogram in file => ST
            otherwise => Both
    
    TODO: how can I had possible?
    """
       
    # Assesting file label 
    for i in range(file_number):
        file = segments_filewise_test[i][0]
        #inizializing counters
        LT_count=0
        ST_count=0
        Other_count=0
        spec_num=0   #counts number of spectrograms per file
        #flag: if no click detected no spectrograms
        click_detected_flag=False
        #count LT, ST and others occurrence
        for k in range(np.shape(spec_id)[0]):
            if spec_id[k][0]==file:
                click_detected_flag= True
                if predictions[k][0]*100>50:
                    LT_count+=1
                elif predictions[k][1]*100>50:
                    ST_count+=1
                else:
                    Other_count+=1
                spec_num+=1
                
        #assign label
        if click_detected_flag==True:
            #this makes sense only if there were spectrograms
#            keepung differen spectrograms 
    #        if Other_count>LT_count+ST_count:
    #        if (Other_count/spec_num)*100>90:
            if LT_count+ST_count==0:
                label='Noise'
            else:
                LT_perc=(LT_count/(spec_num-Other_count))*100 #percentage of LT over "good clicks" clicks
                ST_perc=(ST_count/(spec_num-Other_count))*100 #percentage of LT over "good clicks" clicks
                if LT_perc>70:
                    label='LT'
                elif ST_perc>70:
                    label='ST'
                else:
                    label='Both'
        else:
            #if no click automatically we have Noise
            label='Noise'
            
        filewise_output[i][3] = label
   
    return filewise_output

#2nd option: 2 labels, evaluate it filewise
def File_label4(predictions, spec_id, segments_filewise_test, filewise_output, file_number ):
    """
    FIle_label4 use the predictions made by the CNN to update the filewise annotations
    when we have 2 labels: 0 (LT), 1(ST)
    
    METHOD: evaluation of probability over files
        P(0)>70 => LT
        P(1)>70 => ST
        P(0)>30 and P(1)>30 => Both
        else => Noise
    
     TODO: how can I had possible?
    """
   
    
    if len(predictions)!=np.shape(spec_id)[0]:
        print('ERROR: Number of labels is not equal to number of spectrograms' )
    
    # Assesting file label and updating metrics  
    for i in range(file_number):
        file = segments_filewise_test[i][0]
        file_prob=np.zeros((2,1))
        #inizializing counters
        spec_num=0   #counts number of spectrograms per file
        #flag: if no click detected no spectrograms
        click_detected_flag=False
        for k in range(np.shape(spec_id)[0]):
            if spec_id[k][0]==file:
                click_detected_flag=True
                spec_num+=1
                file_prob+=predictions[k]
        if click_detected_flag==True:
            file_prob/=spec_num
            file_prob*=100
            if file_prob[0]>70:
                label='LT'
            elif file_prob[1]>70:
                label='ST'
            elif file_prob[1]>30 and file_prob[0]>30:
                label='Both'
            else:
                label='Noise'
            
        else:
            label='Noise'
        
        filewise_output[i][3] = label
        
    return filewise_output

##MAIN
 
#Create train dataset for CNN from the results of clicksearch   
train_dir = "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TRAIN2" #changed directory
fs = 16000
annotation_file_train= "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TRAIN2\\Train_dataset_images.data"
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

#search clicks
for i in range(file_number_train):
    file = segments_filewise_train[i][0]
    click_label, train_featuress, count = ClickSearch(train_dir, file, fs, train_featuress, count, Train=True)
    if segments_filewise_train[i][1]=='LT' or segments_filewise_train[i][1]=='ST' or segments_filewise_train[i][1]=='Both':
        if click_label == 'Click':
            TD+=1
        else:
            FND+=1
    else:
        if click_label == 'Click':
            FD+=1
        else:
            TND+=1

#printng metrics
print("-------------------------------------------")
print("Click Detector stats on Training Data")
Recall= TD/(TD+FND)*100
print('Recall ', Recall)
Precision= TD/(TD+FD)*100
print('Precision ', Precision)
Accuracy = (TD+TND)/(TD+TND+FD+FND)*100
print('Accuracy', Accuracy)
TD_rate= (TD/file_number_train)*100
print('True Detected rate', TD_rate)
FD_rate= (FD/file_number_train)*100
print('False Detected rate', FD_rate)
FND_rate= (FND/file_number_train)*100
print('False Negative Detected rate', FND_rate)
TND_rate= (TND/file_number_train)*100
print('True Negative Detected rate', TND_rate)
print("-------------------------------------------")

#saving dataset
with open(os.path.join(train_dir, 'sgramdata_train.json'), 'w') as outfile:
    json.dump(train_featuress, outfile)
    
# Detect clicks in Test Dataset and save it without labels 
    
test_dir = "D:\Desktop\Documents\Work\Data\Bat\BAT\CNN experiment\TEST2" #changed directory
annotation_file_test= "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\Test_dataset_images.data"
test_fold= "BAT SEARCH TESTS\Test_86" #Test folder where to save all the stats
os.mkdir(test_dir+ '/' + test_fold)
with open(annotation_file_test) as f:
    segments_filewise_test = json.load(f)
file_number=np.shape(segments_filewise_test)[0]

#storing train and test dataset into test folder
with open(test_dir+ '/' + test_fold+'\Train_dataset.data', 'w') as f2:
    json.dump(segments_filewise_train,f2)
    
with open(test_dir+ '/' + test_fold+'\Test_dataset.data', 'w') as f2:
    json.dump(segments_filewise_test,f2)

#inizializations
count_start=0
test_featuress =[]
filewise_output=[] #here we store; file, click detected (true/false), #of spectrogram, final label
TD=0
FD=0
TND=0
FND=0
control_spec=0 #this variable count the file where the click detector found a click that are not providing spectrograms for CNN

#search clicks
for i in range(file_number):
    file = segments_filewise_test[i][0]
    control='False'
    click_label, test_featuress, count_end = ClickSearch(test_dir, file, fs, test_featuress, count_start, Train=False)
    gen_spec= count_end-count_start # numb. of generated spectrograms
    
    #update stored information on test file
    filewise_output.append([file, click_label, gen_spec, 'Noise', segments_filewise_test[i][1]]) #note final label inizialized to 'Noise'
    #if I have a click but not a spectrogram I update
    if click_label=='Click' and gen_spec==0:
        control_spec+=1
        
    #updating metrics count    
    if segments_filewise_test[i][1]=='LT' or segments_filewise_test[i][1]=='ST' or segments_filewise_test[i][1]=='Both':
        if click_label == 'Click':
            TD+=1
        else:
            FND+=1
    else:
        if click_label == 'Click':
            FD+=1
        else:
            TND+=1
    count_start=count_end

#printing metrics
print("-------------------------------------------")
print('Number of detected click', count_start, 'in ', file_number, ' files')
print("Click Detector stats on Testing Data")
Recall= TD/(TD+FND)*100
print('Recall ', Recall)
Precision= TD/(TD+FD)*100
print('Precision ', Precision)
Accuracy = (TD+TND)/(TD+TND+FD+FND)*100
print('Accuracy', Accuracy)
TD_rate= (TD/file_number)*100
print('True Detected rate', TD_rate)
FD_rate= (FD/file_number)*100
print('False Detected rate', FD_rate)
FND_rate= (FND/file_number)*100
print('False Negative Detected rate', FND_rate)
TND_rate= (TND/file_number)*100
print('True Negative Detected rate', TND_rate)
print("-------------------------------------------")

#saving Click Detector Stats
cd_metrics_file=test_dir+ '/' + test_fold + '\click_detector_stats.txt'
file1=open(cd_metrics_file,"w")
L0=["Number of file %5d \n"  %file_number]
L1=["Number of detected clicks %5d \n" %count_start ]
L2=["NUmber of file with detected clicks but not spectrograms for CNN = %5d \n " %control_spec]
L3=["Recall = %3.7f \n" %Recall,"Precision = %3.7f \n" %Precision, "Accuracy = %3.7f \n" %Accuracy, "True Detected rate = %3.7f \n" %TD_rate, "False Detected rate = %3.7f \n" %FD_rate, "True Negative Detected rate = %3.7f \n" %TND_rate, "False Negative Detected rate = %3.7f \n" %FND_rate ]
file1.writelines(np.concatenate((L0,L1,L2,L3)))
file1.close()
#saving dataset
with open(test_dir+'\\'+test_fold +'\\sgramdata_test.json', 'w') as outfile:
    json.dump(test_featuress, outfile)
    
#Train CNN

data_train=train_featuress
print('check',np.shape(data_train))
sg_train=np.ndarray(shape=(np.shape(data_train)[0],np.shape(data_train[0][0])[0], np.shape(data_train[0][0])[1]), dtype=float) #check
target_train = np.zeros((np.shape(data_train)[0], 1)) #label train
for i in range(np.shape(data_train)[0]):
    maxg = np.max(data_train[i][0][:])
    sg_train[i][:] = data_train[i][0][:]/maxg
    target_train[i][0] = data_train[i][-1]
 
#Check: how many files I am using for training
print("-------------------------------------------")
print('Number of spectrograms', np.shape(target_train)[0]) 
print("Spectrograms for LT: ", np.shape(np.nonzero(target_train==0))[1])
print("Spectrograms for ST: ", np.shape(np.nonzero(target_train==1))[1])
print("Spectrograms for Noise: ", np.shape(np.nonzero(target_train==2))[1])

#Save training info into a file
cnn_train_info_file=test_dir+'/'+test_fold+'\CNN_train_data_info.txt'
file1=open(cnn_train_info_file,'w')
L=['Number of spectrograms = %5d \n' %np.shape(target_train)[0], 
   "Spectrograms for LT: %5d \n" %np.shape(np.nonzero(target_train==0))[1],
   "Spectrograms for ST: %5d \n" %np.shape(np.nonzero(target_train==1))[1],
   "Spectrograms for Noise: %5d \n" %np.shape(np.nonzero(target_train==2))[1]]
file1.writelines(L)
file1.close()

#recover test data
#note: I am not giving target!
data_test= test_featuress
sg_test=np.ndarray(shape=(np.shape(data_test)[0],np.shape(data_test[0][0])[0], np.shape(data_test[0][0])[1]), dtype=float)
spec_id=[]
print('Number of test spectrograms', np.shape(data_test)[0])
for i in range(np.shape(data_test)[0]):
    maxg = np.max(data_test[i][0][:])
    sg_test[i][:] = data_test[i][0][:]/maxg
    spec_id.append(data_test[i][1:3])
    
#check:
print('check on spec_id', np.shape(spec_id))    

# Using different train and test datasets
x_train = sg_train
y_train = target_train
x_test = sg_test
#y_test = target_test

train_images = x_train.reshape(x_train.shape[0],9, 512, 1) #changed image dimensions
test_images = x_test.reshape(x_test.shape[0],9, 512, 1)
input_shape = (9, 512, 1)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

num_labels=3 #change this variable, when changing number of labels
train_labels = tensorflow.keras.utils.to_categorical(y_train, num_labels)
#test_labels = tensorflow.keras.utils.to_categorical(y_test, 8)   #change this to set labels  

accuracies=np.zeros((10,1)) #initializing accuracies array
model_paths=[] #initializing list where to stor model path
for i in range(5):
    #Build CNN architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), #I don't think this nees to be changed
                     activation='relu',
                     input_shape=input_shape))
    # 64 3x3 kernels
    model.add(Conv2D(64, (3, 3), activation='relu')) #I don't think this nees to be changed
    # Reduce by taking the max of each 2x2 block
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Dropout to avoid overfitting
    model.add(Dropout(0.25))
    # Flatten the results to one dimension for passing into our final layer
    model.add(Flatten())
    # A hidden layer to learn with
    model.add(Dense(128, activation='relu'))
    # Another dropout
    model.add(Dropout(0.5))
    # Final categorization from 0-9 with softmax
    #Virginia: changed 8->3 because I have 3 classes (????)
    model.add(Dense(num_labels, activation='softmax')) #this needs to be changed if I have only 2 labeles
    
    model.summary()
    
    #set the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    #train the model
    #I am not giving validation_data
    print('Training n', i)
    history = model.fit(train_images, train_labels,
                        batch_size=32,
                        epochs=30,
                        verbose=2)
    #save reached accuracy
    accuracies[i]=history.history['acc'][-1]
    print('Accuracy reached',accuracies[i])
    #save model
    modelpath=test_dir+ '/' + test_fold + '/model_'+str(i)+'.h5' #aid variable
    model.save(modelpath)
    model_paths.append(modelpath)

#check what modelgave us better accuracy
index_best_model=np.argmax(accuracies) 
print('Best CNN is ', index_best_model)
print('Best accuracy reached ',accuracies[index_best_model])
modelpath=model_paths[index_best_model]    
#recover model
model=load_model(modelpath)
#recovering labels
predictions =model.predict(test_images)
#predictions is an array #imagesX #of classes which entries are the probabilities
#for each classes

filewise_output=File_label2(predictions, spec_id, segments_filewise_test, filewise_output, file_number )

#compare predicted_annotations with segments_filewise_test
#evaluate metrics
    
# inizializing
TD=0
FD=0
FND=0
TND=0
CoCl=0 #correctly classified
NCoCl=0
comparison_annotations = []
confusion_matrix=np.zeros((4,4))
#confusion_matrix[0][:]=['', 'LT', 'ST', 'Both', 'Noise']
#confusion_matrix[:][0]=['', 'LT', 'ST', 'Both', 'Noise']
print('Estimating metrics')
for i in range(file_number):
    assigned_label= filewise_output[i][3]
    correct_label=segments_filewise_test[i][1]
    if correct_label==assigned_label:
        CoCl+=1
        if correct_label=='Noise':
            TND+=1
            confusion_matrix[3][3]+=1
        else:
            TD+=1
            if correct_label=='LT':
                confusion_matrix[0][0]+=1
            elif correct_label=='ST':
                confusion_matrix[1][1]+=1
            elif correct_label=='Both':
                confusion_matrix[2][2]+=1
    else:
        NCoCl+=1
        if correct_label=='Noise':
            FD+=1
            if assigned_label=='LT':
                confusion_matrix[0][3]+=1
            elif assigned_label=='ST':
                confusion_matrix[1][3]+=1
            elif assigned_label=='Both':
                confusion_matrix[2][3]+=1
        elif assigned_label=='Noise':
            FND+=1
            if correct_label=='LT':
                confusion_matrix[3][0]+=1
            elif correct_label=='ST':
                confusion_matrix[3][1]+=1
            elif correct_label=='Both':
                confusion_matrix[3][2]+=1
        else:
            TD+=1
            if correct_label=='LT':
                if assigned_label=='ST':
                    confusion_matrix[1][0]+=1
                elif assigned_label=='Both':
                    confusion_matrix[2][0]+=1
            elif correct_label=='ST':
                if assigned_label=='LT':
                    confusion_matrix[0][1]+=1
                elif assigned_label=='Both':
                    confusion_matrix[2][1]+=1
            elif correct_label=='Both':
                if assigned_label=='LT':
                    confusion_matrix[0][2]+=1
                elif assigned_label=='ST':
                    confusion_matrix[1][2]+=1
            
    comparison_annotations.append([filewise_output[i][0], segments_filewise_test[i][1], assigned_label])
 
#chck
print('number of files =', file_number)
print('TD =',TD)
print('FD =',FD)
print('TND =',TND)
print('FND =',FND)
print('Correct classifications =', CoCl)
print('uncorrect classifications =', NCoCl)
#printng metrics
print("-------------------------------------------")
print("Click Detector stats on Testing Data")
if TD==0:
    Recall=0
else:
    Recall= TD/(TD+FND)*100
print('Recall ', Recall)
if TD==0:
    Precision= 0
else:
    Precision= TD/(TD+FD)*100
print('Precision ', Precision)
if CoCl==0:
    Accuracy=0
else:
    Accuracy = CoCl/(CoCl+NCoCl)*100
print('Accuracy', Accuracy)
TD_rate= (TD/file_number)*100
print('True Detected rate', TD_rate)
FD_rate= (FD/file_number)*100
print('False Detected rate', FD_rate)
FND_rate= (FND/file_number)*100
print('False Negative Detected rate', FND_rate)
TND_rate= (TND/file_number)*100
print('True Negative Detected rate', TND_rate)
CoCl_rate= (CoCl/file_number)*100
print('Correctly Classified rate', CoCl_rate)
NCoCl_rate= (NCoCl/file_number)*100
print('Uncorrectly Classified rate', NCoCl_rate)
print(confusion_matrix)
print("-------------------------------------------")

#saving Click Detector Stats
cd_metrics_file=test_dir+'\\'+test_fold+'\\bat_detector_stats.txt'
file1=open(cd_metrics_file,"w")
L1=["Bat Detector stats on Testing Data \n"]
L2=['Number of files = %5d \n' %file_number]
L3=['TD = %5d \n' %TD]
L4=['FD = %5d \n' %FD]
L5=['TND = %5d \n' %TND]
L6=['FND = %5d \n' %FND]
L7=['Correctly classified files= %5d \n' %CoCl]
L8=['Uncorrectly classified files= %5d \n' %NCoCl]
L9=["Recall = %3.7f \n" %Recall,"Precision = %3.7f \n" %Precision, "Accuracy = %3.7f \n" %Accuracy, "True Detected rate = %3.7f \n" %TD_rate, "False Detected rate = %3.7f \n" %FD_rate, "True Negative Detected rate = %3.7f \n" %TND_rate, "False Negative Detected rate = %3.7f \n" %FND_rate, "Correctly Classified rate =%3.7f \n" %CoCl_rate, "Uncorrectly Classified rate =%3.7f \n" %NCoCl_rate ]
#L10=["Confusion matrix \n %5d" %confusion_matrix ]
L10=['Model used %5d \n' %index_best_model]
L11=['Training accuracy for the model %3.7f \n' %accuracies[index_best_model]]
file1.writelines(np.concatenate((L1,L2,L3,L4, L5, L6, L7, L8, L9, L10, L11)))
file1.close()
       
#saving compared labels
with open(test_dir+'\\' +test_fold+'\\Test_annotations_comparison.data', 'w') as f:
    json.dump(comparison_annotations,f)


#saving compared labels
with open(test_dir+'\\' +test_fold+'\\Test_filewise_output.data', 'w') as f:
    json.dump(filewise_output,f)
          

        
    
            
        
