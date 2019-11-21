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

import librosa
import WaveletSegment
import WaveletFunctions


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
#    low_len=math.floor(0.07/dt)
#    print(low_len, up_len)
    
    #Frequency band
    f0=3000
    index_f0=-1+math.floor(f0/df) #lower bound needs to be rounded down
#    print(f0,index_f0)
    f1=5000
    index_f1=-1+math.ceil(f1/df) #upper bound needs to be rounded up
#    print(f1,index_f1)
    
#    print(np.shape(imspec[index_f0:index_f1,:]))
#    sum_spec=np.sum(imspec[index_f0:index_f1,:], axis=0) #added 0.01 to avoid divition by 0
    
    #Mean in the frequency band
    mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0) #added 0.01 to avoid divition by 0
#    x_axis=np.arange(np.shape(imspec)[1])
    #Threshold
    mean_spec_all=np.mean(imspec, axis=0)[2:]
    thr_spec=(np.mean(mean_spec_all)+np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))
#    print(np.max(mean_spec),thr_spec,np.std(mean_spec_all))
#    thr_spec=(np.mean(sum_spec[2:])+np.std(sum_spec[2:]))*np.ones((np.shape(sum_spec)))
    
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
#    print(int(click_start))
    click_end=clicks_indices[0][0]
#    print(int(click_end))    
    for i in range(1,np.shape(clicks_indices)[1]):
        if clicks_indices[0][i]==click_end+1:
            click_end=clicks_indices[0][i]
        else:
#                if click_end-click_start+1>up_len or click_end-click_start+1<low_len:
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
    "Update Dataset with current segment"
    #I assign a label t the spectrogram only for Train Dataset
    click_start_sec=click_start*dt
    click_end_sec=click_end*dt
    if Train==True:
        assigned_flag=False #control flag
#        click_seg=[]
        for segix in thisSpSegs:
            seg = segments[segix]
            
            if isinstance(seg[4][0], dict):
                if seg[0]<=click_start_sec and seg[1]>=click_end_sec:
                    if 'Bat (Long Tailed)' == seg[4][0]["species"]:
                        spec_label = 0
                        assigned_flag=True
#                        click_seg =seg
                        break
                    elif 'Bat (Short Tailed)' == seg[4][0]["species"]:
                        spec_label = 1
                        assigned_flag=True
#                        click_seg =seg
                        break
                    elif 'Noise' == seg[4][0]["species"]:
                        spec_label = 2    
                        assigned_flag=True
#                        click_seg =seg
                        break
                    else:
                        continue
                    
            elif isinstance(seg[4][0], str):
                # old format
                if seg[0]<=click_start_sec and seg[1]>=click_end_sec:
                    if 'Bat (Long Tailed)' == seg[4][0]["species"]:
                        spec_label = 0
                        assigned_flag=True
#                        click_seg=seg
                        break
                    elif 'Bat (Short Tailed)' == seg[4][0]["species"]:
                        spec_label = 1
                        assigned_flag=True
#                        click_seg=seg
                        break
                    elif 'Noise' == seg[4][0]["species"]:
                        spec_label = 2   
                        assigned_flag=True
#                        click_seg=seg
                        break
                    else:
                        continue
        if assigned_flag==False:
            spec_label=2
#            click_seg=[click_start, click_end, 0, 8000]
    
# slice spectrogram   
    #win = 0.005 #dt is stable
#    win_pixel=math.ceil(win/dt)
    win_pixel=2
    duration = click_end -click_start +1
    if duration > win_pixel:
        
        n = math.ceil(duration/win_pixel)
    #         inizialization
        start_pixel=click_start
        
    #    imagewindow = pg.image()
        for i in range(n):
            end_pixel=start_pixel+win_pixel
            sgRaw=spectrogram[:,start_pixel:end_pixel] #not I am saving the spectrogram in the right dimension
            sgRaw=np.repeat(sgRaw,6,axis=1)
            sgRaw=(np.flipud(sgRaw)).T #flipped spectrogram to make it consistent with Niro Mewthod
            if Train==True:
    #            print('Train==True')
                featuress.append([sgRaw.tolist(), file_name, count, spec_label])
            else:
         #if testing: do not save label
                featuress.append([sgRaw.tolist(), file_name, count]) #not storing segment and label informations
            start_pixel=end_pixel
            #not storing images
    #        maxsg = np.min(sgRaw)
    #        sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))
    #
    #        img = pg.ImageItem(sg)
    #        imagewindow.clear()
    #        imagewindow.setImage(np.flip(sg, 1)) 
    #        exporter = pge.ImageExporter(imagewindow.view)
    #        if Train==True:
    #            exporter.export(os.path.join(dirName, 'img', str(spec_label) + '_' + "%04d" % count + '.png'))
    #        else:
    #            exporter.export(os.path.join(dirName, 'img', "%04d" % count + '.png')) #not saving label for testing
            count += 1
#    imagewindow.close()
             
    return featuress, count
            
def updateDataset2(file_name, dirName, featuress, count, spectrogram, segments, thisSpSegs, click_start, click_end, dt, Train=False):
    "Update Dataset with current segment"
    #I assign a label t the spectrogram only for Train Dataset
    click_start_sec=click_start*dt
    click_end_sec=click_end*dt
    if Train==True:
        assigned_flag=False #control flag
#        click_seg=[]
        for segix in thisSpSegs:
            seg = segments[segix]
            
            if isinstance(seg[4][0], dict):
                if seg[0]<=click_start_sec and seg[1]>=click_end_sec:
                    if 'Bat (Long Tailed)' == seg[4][0]["species"]:
                        spec_label = 0
                        assigned_flag=True
#                        click_seg =seg
                        break
                    elif 'Bat (Short Tailed)' == seg[4][0]["species"]:
                        spec_label = 1
                        assigned_flag=True
#                        click_seg =seg
                        break
                    elif 'Noise' == seg[4][0]["species"]:
                        spec_label = 2    
                        assigned_flag=True
#                        click_seg =seg
                        break
                    else:
                        continue
                    
            elif isinstance(seg[4][0], str):
                # old format
                if seg[0]<=click_start_sec and seg[1]>=click_end_sec:
                    if 'Bat (Long Tailed)' == seg[4][0]["species"]:
                        spec_label = 0
                        assigned_flag=True
#                        click_seg=seg
                        break
                    elif 'Bat (Short Tailed)' == seg[4][0]["species"]:
                        spec_label = 1
                        assigned_flag=True
#                        click_seg=seg
                        break
                    elif 'Noise' == seg[4][0]["species"]:
                        spec_label = 2   
                        assigned_flag=True
#                        click_seg=seg
                        break
                    else:
                        continue
        if assigned_flag==False:
            spec_label=2
#            click_seg=[click_start, click_end, 0, 8000]
    
# slice spectrogram   
    #win = 0.005 #dt is stable
#    win_pixel=math.ceil(win/dt)
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
#            print('Train==True')
            featuress.append([sgRaw.tolist(), file_name, count, spec_label])
        else:
     #if testing: do not save label
            featuress.append([sgRaw.tolist(), file_name, count]) #not storing segment and label informations
        
        click_start+=win_pixel2*2+1
        count += 1

    return featuress, count

def updateDataset3(file_name, dirName, featuress, count, spectrogram, segments, thisSpSegs, click_start, click_end, dt, Train=False):
    "Update Dataset with current segment"
    "This function generate I spectrogram long 3 pixels and then duplicate it"
    #I assign a label t the spectrogram only for Train Dataset
    click_start_sec=click_start*dt
    click_end_sec=click_end*dt
    if Train==True:
        assigned_flag=False #control flag
#        click_seg=[]
        for segix in thisSpSegs:
            seg = segments[segix]
            
            if isinstance(seg[4][0], dict):
                if seg[0]<=click_start_sec and seg[1]>=click_end_sec:
                    if 'Bat (Long Tailed)' == seg[4][0]["species"]:
                        spec_label = 0
                        assigned_flag=True
#                        click_seg =seg
                        break
                    elif 'Bat (Short Tailed)' == seg[4][0]["species"]:
                        spec_label = 1
                        assigned_flag=True
#                        click_seg =seg
                        break
                    elif 'Noise' == seg[4][0]["species"]:
                        spec_label = 2    
                        assigned_flag=True
#                        click_seg =seg
                        break
                    else:
                        continue
                    
            elif isinstance(seg[4][0], str):
                # old format
                if seg[0]<=click_start_sec and seg[1]>=click_end_sec:
                    if 'Bat (Long Tailed)' == seg[4][0]["species"]:
                        spec_label = 0
                        assigned_flag=True
#                        click_seg=seg
                        break
                    elif 'Bat (Short Tailed)' == seg[4][0]["species"]:
                        spec_label = 1
                        assigned_flag=True
#                        click_seg=seg
                        break
                    elif 'Noise' == seg[4][0]["species"]:
                        spec_label = 2   
                        assigned_flag=True
#                        click_seg=seg
                        break
                    else:
                        continue
        if assigned_flag==False:
            spec_label=2
#            click_seg=[click_start, click_end, 0, 8000]
    
# slice spectrogram   
    #win = 0.005 #dt is stable
#    win_pixel=math.ceil(win/dt)
    win_pixel=1 
    ls = np.shape(spectrogram)[1]-1
#    duration=click_end-click_start+1
    click_center=int((click_start+click_end)/2)
#    print(click_start, click_end,click_center)


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
    if end_pixel-start_pixel != 2:
        print("*******************************************",end_pixel,start_pixel)
        #this code above fails for sg less than 5 pixels wide
#    print(start_pixel, end_pixel)    
    sgRaw=spectrogram[:,start_pixel:end_pixel+1] #not I am saving the spectrogram in the right dimension
    sgRaw=np.repeat(sgRaw,2,axis=1)
    sgRaw=(np.flipud(sgRaw)).T #flipped spectrogram to make it consistent with Niro Mewthod
    if Train==True:
#            print('Train==True')
        featuress.append([sgRaw.tolist(), file_name, count, spec_label])
    else:
 #if testing: do not save label
        featuress.append([sgRaw.tolist(), file_name, count]) #not storing segment and label informations

    count += 1

    return featuress, count
##MAIN
 
#Create train dataset for CNN from the results of clicksearch   
train_dir = "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TRAIN" #changed directory
fs = 16000
annotation_file_train= "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TRAIN\\Train_dataset.data"
with open(annotation_file_train) as f:
    segments_filewise_train = json.load(f)
file_number_train=np.shape(segments_filewise_train)[0]

#inizializations
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
            

#if xxxx == 2:
#    sulk

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
    
test_dir = "D:\Desktop\Documents\Work\Data\Bat\BAT\CNN experiment\TEST" #changed directory
annotation_file_test= "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST\\Test_dataset.data"
test_fold= "BAT SEARCH TESTS\Test_26" #Test folder where to save all the stats
with open(annotation_file_test) as f:
    segments_filewise_test = json.load(f)
file_number=np.shape(segments_filewise_test)[0]

#inizializations
count_start=0
test_featuress =[]
#test_filelist=[] #Here we store file name, spectrograms id,label
filewise_output=[] #here we store; file, click detected (true/false), #of spectrogram, final label
TD=0
FD=0
TND=0
FND=0
#spec_flag=[]
control_spec=0 #this variable count the file where the click detector found a click that are not providing spectrograms for CNN

#search clicks
for i in range(file_number):
    file = segments_filewise_test[i][0]
    control='False'
    click_label, test_featuress, count_end = ClickSearch(test_dir, file, fs, test_featuress, count_start, Train=False)
    gen_spec= count_end-count_start # numb. of generated spectrograms
    #update stored information on test file
#    if count_start!=count_end:
#        control='True'
#    for k in range(count_start,count_end):
#        test_filelist.append([file,k,2]) #note:label inizialized to 2
#        control='True'
#    spec_flag.append([file,control,count_end-count_start])
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

#recover train data
#with open("D:\Desktop\Documents\Work\Data\Bat\BAT\CNN experiment\TRAIN\sgramdata_train.json") as f:
#    data_train = json.load(f)
#print(np.shape(data_train))

data_train=train_featuress
#print(np.shape(data_train[0][0]))
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
#file1.write('Number of spectrograms = %5d \n', np.shape(target_train))
#file1.write("Spectrograms for LT: %5d \n", np.shape(np.nonzero(target_train==0))[1])
#file1.write("Spectrograms for ST: %5d \n", np.shape(np.nonzero(target_train==1))[1])
#L1=['Number of spectrograms = %5d \n', np.shape(target_train)[0]]
#L2=["Spectrograms for LT: %5d \n", np.shape(np.nonzero(target_train==0))[1]]
#L3=["Spectrograms for ST: %5d \n", np.shape(np.nonzero(target_train==1))[1]]
#L4=["Spectrograms for Noise: %5d \n", np.shape(np.nonzero(target_train==2))[1]]
#L1.append(L2)
#L1.append(L3)
#L1.append(L4)
L=['Number of spectrograms = %5d \n' %np.shape(target_train)[0], 
   "Spectrograms for LT: %5d \n" %np.shape(np.nonzero(target_train==0))[1],
   "Spectrograms for ST: %5d \n" %np.shape(np.nonzero(target_train==1))[1],
   "Spectrograms for Noise: %5d \n" %np.shape(np.nonzero(target_train==2))[1]]
file1.writelines(L)
file1.close()

#recover test data
#note: I am not giving target!
#with open("D:\Desktop\Documents\Work\Data\Bat\BAT\CNN experiment\TEST\sgramdata_test.json") as f:
#    data_test = json.load(f)
#print(np.shape(data_test))
data_test= test_featuress

sg_test=np.ndarray(shape=(np.shape(data_test)[0],np.shape(data_test[0][0])[0], np.shape(data_test[0][0])[1]), dtype=float)
#sg_test=np.ndarray(shape=(np.shape(data_test)[0],7, 512), dtype=float) #check
#target_test = np.zeros((np.shape(data_test)[0], 1)) #labels test
#spec_id=np.zeros((np.shape(data_test)[0], 2)) #spectrogram idintification number associated at file name
spec_id=[]
print('Number of test spectrograms', np.shape(data_test)[0])
for i in range(np.shape(data_test)[0]):
    maxg = np.max(data_test[i][0][:])
    sg_test[i][:] = data_test[i][0][:]/maxg
    spec_id.append(data_test[i][1:3])
#    spec_id[i][:] = data_test[i][1:]
#    target_test[i][0] = data_test[i][-1]
#print(np.shape(np.zeros(np.shape(spec_id)[0])))
print('check on spec_id', np.shape(spec_id))    
#spec_id=np.concatenate((spec_id, np.zeros((np.shape(spec_id)[0],1))), axis=1) #inizialize label to 0
#print('check on spec_id', np.shape(spec_id))
# Using different train and test datasets
x_train = sg_train
y_train = target_train
x_test = sg_test
#y_test = target_test

train_images = x_train.reshape(x_train.shape[0],6, 512, 1) #changed image dimensions
test_images = x_test.reshape(x_test.shape[0],6, 512, 1)
input_shape = (6, 512, 1)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_labels = tensorflow.keras.utils.to_categorical(y_train, 3)
#test_labels = tensorflow.keras.utils.to_categorical(y_test, 8)   #change this to set labels  

#Build CNN architecture

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# 64 3x3 kernels
model.add(Conv2D(64, (3, 3), activation='relu'))
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
model.add(Dense(3, activation='softmax'))

model.summary()

#set the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#train the model
#I am not giving validation_data
history = model.fit(train_images, train_labels,
                    batch_size=32,
                    epochs=25,
                    verbose=2)

#recovering labels
predictions =model.predict(test_images)
#predictions is an array #imagesX #of classes which entries are the probabilities
#for each classes
#print('predictions -> ', np.shape(predictions))
#print(predictions)
#why 8 labels? why the labels in predicted_label are just 0 and 1?
predicted_label=np.argmax(predictions,axis=1)
print('predicted label -> ', np.amax(predicted_label), np.amin(predicted_label))
#print(predicted_label)
if len(predicted_label)!=np.shape(spec_id)[0]:
    print('ERROR: Number of labels is not equal to number of spectrograms' )

#updateding spec_id with predicted label
#for i in range(np.shape(spec_id)[0]):
#    spec_id[i][2]=predicted_label[i]

# Assesting file label and updating metrics   
for i in range(file_number):
    file = segments_filewise_test[i][0]
#    print('Assesting label of file ', file)
    #inizializing counters
    LT_count=0
    ST_count=0
    Other_count=0
    spec_num=0   #counts number of spectrograms per file
    #flag: if no click detected no spectrograms
    click_detected_flag=False
    #count majority
    for k in range(np.shape(spec_id)[0]):
        if spec_id[k][0]==file:
            click_detected_flag= True
#            if spec_id[k][2]==0:
#                LT_count+=1
#            elif spec_id[k][2]==1:
#                ST_count+=1
            if predicted_label[k]==0:
                LT_count+=1
            elif predicted_label[k]==1:
                ST_count+=1
            else:
                Other_count+=1
            spec_num+=1
    #assign label
#    print('check count', LT_count, ST_count, Other_count)
    if click_detected_flag==True:
        #this makes sense only if there were spectrograms
#        if Other_count>LT_count+ST_count:
#        if (Other_count/spec_num)*100>80:
        if LT_count+ST_count==0:
            label='Noise'
        else:
            LT_perc=LT_count/(spec_num-Other_count)*100 #percentage of LT over "good clicks" clicks
            ST_perc=ST_count/(spec_num-Other_count)*100 #percentage of LT over "good clicks" clicks
            if LT_perc>70:
                label='LT'
            elif ST_perc>70:
                label='ST'
            else:
                label='Both'
    else:
        #if no click automatically we have Noise
        label='Noise'
#    print(file, label, click_detected_flag)
    filewise_output[i][3] = label

    

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
#    if predicted_annotations[i][0]==segments_filewise_test[i][0]:
#        print('file order is correct',i)
#    for k in range(file_number):
#    if predicted_annotations[i][0]==segments_filewise_test[i][0]:
    correct_label=segments_filewise_test[i][1]
#            print('found correct label')
#            break
#        else:
#            print('Somenthing wrong check')
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
file1.writelines(np.concatenate((L1,L2,L3,L4, L5, L6, L7, L8, L9)))
file1.close()
#
#confusion_matrix=json.dump(confusion_matrix)
#with open(test_dir+'\\' +test_fold+'\\confusion_matrix.txt','wb') as f:
#    json.dump(confusion_matrix,f)
#    for line in confusion_matrix:
#        np.savetxt(f, line, fmt='%5d')
#    for i in range(4):
#        f.writeline(["|----------------------------|\n"])
#        for j in range(4):
#            f.write(confusion_matrix[i][j])
#        f.writeline(["|----------------------------|\n"])
    
    
#saving compared labels
with open(test_dir+'\\' +test_fold+'\\Test_annotations_comparison.data', 'w') as f:
    json.dump(comparison_annotations,f)


#saving compared labels
with open(test_dir+'\\' +test_fold+'\\Test_filewise_output.data', 'w') as f:
    json.dump(filewise_output,f)
          

        
    
            
        
