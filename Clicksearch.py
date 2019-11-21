# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:58:19 2019

@author: Virginia Listanti
"""

# CLicksearch looks for click long 0.1/0.5 seconds  and then evaluates metrics
#Clicks are found using mean of the energy

import json
import numpy as np
import wavio
import SignalProc
import math

dirName='D:\Desktop\Documents\Work\Data\Bat\BAT\CNN experiment\TEST'
annotation_file= "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST\\Test_dataset.data"
with open(annotation_file) as f:
    segments = json.load(f)
file_number=np.shape(segments)[0]
clicks_annotations = [] #inizializations
comparison_annotations = []
fs=16000

#inizializing
TD=0
FD=0
TND=0
FND=0

for k in range(file_number):
    
    file = segments[k][0]
    print(file)
    filename = dirName + '/' + file
    audiodata = wavio.read(filename)
    sp = SignalProc.SignalProc(1024, 512) 
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
    up_len=math.ceil(0.5/dt) #1 second lenth in indices  
    low_len=math.floor(0.07/dt)
    print(low_len, up_len)
    
    #sum along colums
    f0=3000
    index_f0=-1+math.ceil(f0/df) #lower bound needs to be rounded down
#    print(f0,index_f0)
    f1=5000
    index_f1=-1+math.floor(f1/df) #upper bound needs to be rounded up
#    print(f1,index_f1)
#    print(np.shape(imspec[index_f0:index_f1,:]))
#    sum_spec=np.sum(imspec[index_f0:index_f1,:], axis=0) #added 0.01 to avoid divition by 0
    mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0) #added 0.01 to avoid divition by 0
    x_axis=np.arange(np.shape(imspec)[1])
    mean_spec_all=np.mean(imspec, axis=0)[2:]
    thr_spec=(np.mean(mean_spec_all)+np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))
#    print(np.max(mean_spec),thr_spec,np.std(mean_spec_all))
#    thr_spec=(np.mean(sum_spec[2:])+np.std(sum_spec[2:]))*np.ones((np.shape(sum_spec)))
    
    ##clickfinder
    #check when the sum is bigger than the mean
    #clicks is an array which elements are equal to 1 only where the sum is bigger 
    #than the mean, otherwise are equal to 0

    clicks=np.where(mean_spec>thr_spec,1,0)
    clicks_indices=np.nonzero(clicks)
    print(np.shape(clicks_indices))
    #check: if I have found somenthing
    if np.shape(clicks_indices)[1]==0:
        #If not: label = None 
        label='None'
    else:
        click_start=clicks_indices[0][0]
    #    print(int(click_start))
        click_end=clicks_indices[0][0]
    #    print(int(click_end))
    #    DIscarding segments too long or too shorts
        for i in range(1,np.shape(clicks_indices)[1]):
            if clicks_indices[0][i]==click_end+1:
                click_end=clicks_indices[0][i]
            else:
#                if click_end-click_start+1>up_len or click_end-click_start+1<low_len:
                if click_end-click_start+1>up_len:
                    #print('check')
                    clicks[click_start:click_end+1]=0
                click_start=clicks_indices[0][i]
                click_end=clicks_indices[0][i]            
        #checking last loop with end
        if click_end-click_start+1>up_len:
            clicks[click_start:click_end+1]=0   
#        elif click_end-click_start+1<low_len:
#            clicks[click_start:click_end+1]=0
        clicks_indices=np.nonzero(clicks)
        
        #update label
        if np.shape(clicks_indices)[1]==0:
            label='None'
        else:
            label='Click'
            
    
    #updating poistive/negatives counts
    print(label,segments[k][1])
    if segments[k][1]=='LT' or segments[k][1]=='ST' or segments[k][1]=='Both':
        if label == 'Click':
            TD+=1
#            print(TD)
        else:
            FND+=1
#            print(FND)
    else:
        if label == 'Click':
            FD+=1
#            print(FD)
        else:
            TND+=1
#            print(TND)
    
#    Update clicks dataset
    clicks_annotations.append([file, label])
    comparison_annotations.append([file, segments[k][1], label])
    
#check
if len(clicks_annotations)!=len(segments):
    print('Dataset with 2 different lengths check!')
    
with open(dirName+'\Test_annotations.data', 'w') as f2:
    json.dump(clicks_annotations,f2)
    
with open(dirName+'\Test_annotations_comparison.data', 'w') as f:
    json.dump(comparison_annotations,f)
   
print('TD =',TD)
print( ' FD=', FD)
#print('TND'= TND)
print( 'FND=', FND)
#metrics    
Recall= TD/(TD+FND)
print('Recall ', Recall)
Precision= TD/(TD+FD)
print('Precision ', Precision)
Accuracy = (TD+TND)/(TD+TND+FD+FND)
print('Accuracy', Accuracy)
TD_rate= (TD/file_number)*100
print('True Detected rate', TD_rate)
FD_rate= (FD/file_number)*100
print('False Detected rate', FD_rate)
FND_rate= (FND/file_number)*100
print('False Negative Detected rate', FND_rate)
TND_rate= (TND/file_number)*100
print('True Negative Detected rate', TND_rate)
            
        
            


