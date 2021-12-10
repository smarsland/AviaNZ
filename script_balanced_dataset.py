"""
Script to assamble a balanced dataset
count # of spectrogram for ST and LT 
and add noise files until we have 
#spec_noise=max(LT,ST)
or
#spec_noise=2*mean(LT,ST)

This will generate 6 Datasets.
3 pools for Noise: 
Noise
New_1
New_2
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
import csv
from shutil import copy

import librosa
import WaveletSegment
import WaveletFunctions

from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def ClickSearch(imspec, window, f0, f1, segments, featuress, Train=False):
    """
    Searches for clicks in the provided imspec, saves segments in .data file
    returns click_label, dataset and count of detections

    The search is made on the spectrogram image that we know to be generated
    with parameters (1024,512)
    Click presence is assessed for each spectrogram column: if the mean in the
    frequency band [f0, f1] (*) is bigger than a treshold we have a click
    thr=mean(all_spec)+std(all_spec) (*)

    The clicks are discarded if longer than 0.05 sec

    imspec: time=number of columns
    file: NOTE originally was basename, now full filename
    """
    
    #inizializations
    #featuress=[]
    count = 0

    df=sp.sampleRate//2 /(np.shape(imspec)[0]+1)  # frequency increment
    dt=sp.incr/sp.sampleRate  # self.sp.incr is set to 512 for bats dt=temporal increment in samples
    duration=dt*np.shape(imspec)[1] #duration=dt*num_columns

    up_len=math.ceil(0.075/dt) #max length in columns

    # Frequency band
    #f0=24000
    index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
    #f1=54000
    index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up

    # Mean in the frequency band
    mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0)

    # Threshold
    mean_spec_all=np.mean(imspec, axis=0)[2:]
    thr_spec=(np.mean(mean_spec_all))*np.ones((np.shape(mean_spec)))

    ## clickfinder
    # check when the mean is bigger than the threshold
    # clicks is an array which elements are equal to 1 only where the sum is bigger
    # than the mean, otherwise are equal to 0
    clicks = mean_spec>thr_spec
    clicks_indices = np.nonzero(clicks)
    # check: if I have found somenthing
    if np.shape(clicks_indices)[1]==0:
        label='None'
        return label, featuress
        # not saving spectrograms

    # Discarding segments too long or too short and saving spectrogram images
    click_start=clicks_indices[0][0]
    click_end=clicks_indices[0][0]
    for i in range(1,np.shape(clicks_indices)[1]):
        if clicks_indices[0][i]==click_end+1:
            click_end=clicks_indices[0][i]
        else:
            if click_end-click_start+1>up_len:
                clicks[click_start:click_end+1] = 0
            else:
                # update annotations
                count+=1
                #savedataset
                featuress=updateDataset(imspec, segments, window, click_start, click_end, featuress, Train)
                
            # update
            click_start=clicks_indices[0][i]
            click_end=clicks_indices[0][i]

    # checking last loop with end
    if click_end-click_start+1>up_len:
        clicks[click_start:click_end+1] = 0
    else:
        count+=1
        #savedataset
        featuress=updateDataset(imspec, segments, window, click_start, click_end, featuress, Train)

    #Assigning: click label
    if count==0:
        label='None'
    else:
        label='Click'

    print(count, ' clicks found')
        
    return label, featuress 


def updateDataset(spectrogram, segments, window, click_start, click_end, featuress, Train=False):
    """
    Update Dataset with current segment
    It take a piece of the spectrogram with fixed length centered in the
    click 
    
    We are using AviaNZ annotations but are  reading click_start and click_end
    in seconds 
    
    TRAIN MODE => stores the lables as well
    A spectrogram is labeled is the click is inside a segment
    We have 3 labels:
        0 => LT
        1 => ST
        2 => Noise
    """
    #I assign a label t the spectrogram only for Train Dataset
    # we convert evereting to Avianz time scale
    dt=sp.incr/sp.sampleRate 
    click_start_sec=click_start*dt
    click_end_sec= click_end*dt

    thisSpSegs = np.arange(len(segments)).tolist()
    if Train==True:
        assigned_flag=False #control flag
        for segix in thisSpSegs:
            seg = segments[segix]
            if isinstance(seg[4][0], dict):
#                print('UpdateDataset Check')
                if seg[0]<=click_start_sec and seg[1]>=click_end_sec:
                    if 'Long-tailed bat' == seg[4][0]["species"]:
                        spec_label = 0
                        assigned_flag=True
                        break
                    elif 'Short-tailed bat' == seg[4][0]["species"]:
                        spec_label = 1
                        assigned_flag=True
                        break
                    elif "Don't Know" == seg[4][0]["species"]:
                        spec_label = 2    
                        assigned_flag=True
                        break
                    else:
                        continue
                    
            elif isinstance(seg[4][0], str):
                # old format
                print('UpdateDataset Check')
                if seg[0]<=click_start_sec and seg[1]>=click_end_sec:
                    if 'Long-tailed bat' == seg[4][0]["species"]:
                        spec_label = 0
                        assigned_flag=True
                        break
                    elif 'Short-tailed bat' == seg[4][0]["species"]:
                        spec_label = 1
                        assigned_flag=True
                        break
                    elif "Don't Know" == seg[4][0]["species"]:
                        spec_label = 2   
                        assigned_flag=True
                        break
                    else:
                        continue
        if assigned_flag==False:
            spec_label=2
    
# slice spectrogram   

    win_pixel=int(np.floor(window/2))
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
    if window==3:
        #we need reapeted columns only if window=3
        sgRaw=np.repeat(sgRaw,2,axis=1)
    sgRaw=(np.flipud(sgRaw)).T #flipped spectrogram to make it consistent with Niro Mewthod
    #print('spec features dim', np.shape(sgRaw))
    if Train==True:
        featuress.append([sgRaw.tolist(), spec_label])
    else:
        featuress.append([sgRaw.tolist()]) #not storing segment and label informations

    return featuress

def segm_counting(file_list):
    """
    Count LT, ST and Noise segments generated by files in file list
    """

    f0=24000
    f1=54000
    window=3
    train_featuress=[]

    for k in range(len(file_list)):
        file_path = file_list[k]
        print('Analizing file ', file_path)

        #read image
        print('Uploading file', file_path)
        sp.readBmp(file_path, rotate=False)
        #read annotaion file
        #Read annotation file: if in Train mode
        annotation_file=file_path+'.data'
        if os.path.isfile(annotation_file):
            segments = Segment.SegmentList()
            segments.parseJSON(annotation_file)
            #thisSpSegs = np.arange(len(segments)).tolist()
        else:
                segments=[]
                #thisSpSegs=[]
        #CLickSearch
        print('Click search on file ', file_path)
        click_label, train_featuress=ClickSearch(sp.sg, window, f0, f1, segments, train_featuress, Train=True)

           
    #print('checks')
    #print(np.array(train_featuress)[:,1])
    #print(np.shape(np.nonzero(np.array(train_featuress)[:,-1]==1)))
    count_LT= np.shape(np.nonzero(np.array(train_featuress)[:,-1]==0))[1]
    count_ST= np.shape(np.nonzero(np.array(train_featuress)[:,-1]==1))[1]
    count_Noise= np.shape(np.nonzero(np.array(train_featuress)[:,-1]==2))[1]

    return count_LT, count_ST, count_Noise

##################################################################################################################################################################################
##############################################################   MAIN    #########################################################################################################
##################################################################################################################################################################################

#inizialization

origin_root = "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\Old" #directory with train files
new_root="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\New_Train_Datasets" #new train dataset folder path
#dataset_count=0 #counter for new datasets

sp=SignalProc.SignalProc(1024,512)
f0=24000
f1=54000
window=3

#randomization
new_noise_0="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\New\\20191106"
new_noise_files=[]
for f in os.listdir(new_noise_0):
    if f.endswith('.bmp'):
        new_noise_files.append(new_noise_0+'\\'+f)

new_noise_1="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\New\\20191109"
for f in os.listdir(new_noise_1):
    if f.endswith('.bmp'):
        new_noise_files.append(new_noise_1+'\\'+f)
#print(len(files))
x = np.arange(len(new_noise_files)-1)
np.random.shuffle(x)


for i in range(6):
    print('\n ------------------------------------------------ \n')
    print('Creating dataset ', i)
    dataset_fold='Train_'+str(i)
    if dataset_fold not in os.listdir(new_root):
        os.mkdir(new_root+'\\'+dataset_fold)
        os.mkdir(new_root+'\\'+dataset_fold+'\\LT')
        os.mkdir(new_root+'\\'+dataset_fold+'\\ST')
        os.mkdir(new_root+'\\'+dataset_fold+'\\NOISE')

    if i<2:
        train_fold="TRAIN100"
    elif i<4:
        train_fold="TRAIN250"
    else:
        train_fold="TRAIN500"

    if i%2==0:

        #counters inizialization
        seg_count_LT=0 
        seg_count_ST=0
        seg_count_Noise=0

        #LT randomization
        LT_files=[]
        LT_dir_path=origin_root+'\\'+train_fold+'\\LT'
        for f in os.listdir(LT_dir_path):
            if f.endswith('.bmp'):
                LT_files.append(LT_dir_path+'\\'+f) 

        #count LT segments
        count_LT, count_ST, count_Noise = segm_counting(LT_files)
        seg_count_LT+=count_LT
        seg_count_ST+=count_ST
        seg_count_Noise+=count_Noise

        #ST randomization
        ST_files=[]
        ST_dir_path=origin_root+'\\'+train_fold+'\\ST'
        for f in os.listdir(ST_dir_path):
            if f.endswith('.bmp'):
                ST_files.append(ST_dir_path+'\\'+f)

        #count ST segments
        count_LT, count_ST, count_Noise = segm_counting(ST_files)
        seg_count_LT+=count_LT
        seg_count_ST+=count_ST
        seg_count_Noise+=count_Noise
        print('count LT= ', count_LT)
        print('count ST= ', count_ST)
        print('count Noise= ', count_Noise)
        

        #Noise randomization
        Noise_files=[]
        Noise_dir_path=origin_root+'\\'+train_fold+'\\NOISE'
        for f in os.listdir(Noise_dir_path):
            if f.endswith('.bmp'):
                Noise_files.append(Noise_dir_path+'\\'+f)
        y=np.arange(len(Noise_files)-1)
        np.random.shuffle(y)
        
        Noise_segm_lim=np.maximum(seg_count_LT,seg_count_ST)

    else:
        Noise_segm_lim=2*np.mean((seg_count_LT, seg_count_ST))

    #create dataset

    #copy LT files
    for file_path in LT_files:
        if os.path.isfile(file_path+'.data'):
            copy(file_path, new_root+'\\'+dataset_fold+'\\LT')
            copy(file_path+'.data', new_root+'\\'+dataset_fold+'\\LT')

    #copy ST files
    for file_path in ST_files:
        if os.path.isfile(file_path+'.data'):
            copy(file_path, new_root+'\\'+dataset_fold+'\\ST')
            copy(file_path+'.data', new_root+'\\'+dataset_fold+'\\ST')

    #manage noise files
    Noise_check=seg_count_Noise
    index_track=0
    while Noise_check<=Noise_segm_lim:
        print('Noise_check= ', Noise_check)
        print('Noise_segm_lim= ',Noise_segm_lim)
        print('seg_count_LT= ',seg_count_LT)
        print('seg_count_ST= ',seg_count_ST)
        if index_track%2==1 and (index_track-1)/2/2<len(x):
            print('index_track=', index_track)
            print('len x ',  len(x))
            print('x index=', int((index_track-1)/2))
            file_path=new_noise_files[x[int((index_track-1)/2)]-1]
            
        else:
            print('index_track=', index_track)
            print('len y ',  len(y))
            print('y index=', int(index_track/2))
            file_path=Noise_files[y[int(index_track/2)]-1]
        train_featuress=[]
        #read image
        print('Uploading file', file_path)
        sp.readBmp(file_path, rotate=False)
        #read annotaion file
        #Read annotation file: if in Train mode
        annotation_file=file_path+'.data'
        if os.path.isfile(annotation_file):
            segments = Segment.SegmentList()
            segments.parseJSON(annotation_file)
            #thisSpSegs = np.arange(len(segments)).tolist()
        else:
                segments=[]
                #thisSpSegs=[]
        #CLickSearch
        print('Click search on file ', file_path)
        click_label, train_featuress=ClickSearch(sp.sg, window, f0, f1, segments, train_featuress, Train=True)
        if click_label=='None':
                print('No click detected in ', file_path)
                count_Noise=0
        else:
            count_Noise= np.shape(np.nonzero(np.array(train_featuress)[:,-1]==2))[1]
            

            #copy
            copy(file_path, new_root+'\\'+dataset_fold+'\\NOISE')
            copy(file_path+'.data', new_root+'\\'+dataset_fold+'\\NOISE')

        #update
        Noise_check+=count_Noise
        index_track+=1

    #save dataset info
    dataset_info_file=new_root+'\\'+dataset_fold+'\\train_dataset_info.txt'
    file1=open(dataset_info_file,'w')
    
    total_count=Noise_check+seg_count_LT+seg_count_ST
    L00=["Dataset info \n "]
    L0=['Click search parameters: f0= %5d, f1= %5d \n' %(f0,f1) ]
    L=['Number of spectrograms =  %5d \n' %total_count, 
    "Spectrograms for LT: %5d \n" %seg_count_LT,
    "Spectrograms for ST: %5d \n" %seg_count_ST,
    "Spectrograms for Noise: %5d \n" %Noise_check]
    file1.writelines(np.concatenate((L00,L0,L)))
    file1.close()

