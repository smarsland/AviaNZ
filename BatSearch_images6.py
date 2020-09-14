# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:05:05 2019

@author: Virginia Listanti

# Bat Search looks for clicks in files and then use a CNN to classify files
# as LT Bat, ST bat, Both or None.

#Part of the code is based on Nirosha Priyadarshani scripts makeTrainingData,py
# and CNN_keras
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
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
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


def updateDataset2(file_name, dirName, featuress, count, spectrogram, segments, thisSpSegs, click_start, click_end, dt=None, Train=False):
    """
    Update Dataset with current segment
    It take a piece of the spectrogram with fixed length centered in the
    click 
    
    We are using AviaNZ annotations but are  reading click_start and click_end
    in seconds multiplicating by 11
    
    We assign a label to the click if there is an overlap between the click and 
    the label
    
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
                if (seg[0]<=click_start_sec and seg[1]>=click_end_sec) or (seg[0]<=click_start_sec and seg[1]<click_end_sec) or (seg[0]>click_start_sec and seg[1]>=click_end_sec) or (seg[0]>=click_start_sec and seg[1]<=click_end_sec):
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
                if (seg[0]<=click_start_sec and seg[1]>=click_end_sec) or (seg[0]<=click_start_sec and seg[1]<click_end_sec) or (seg[0]>click_start_sec and seg[1]>=click_end_sec) or (seg[0]>=click_start_sec and seg[1]<=click_end_sec):
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

#Asses file label: 3 labels, evaluate probability filewise
def File_label(predictions, spec_id, segments_filewise_test, filewise_output, file_number ):
    """
    FIle_label use the predictions made by the CNN to update the filewise annotations
    when we have 3 labels: 0 (LT), 1(ST), 2 (Noise)
    
    METHOD: evaluation of probability over files combining mean of probability
        + best3mean of probability

    File labels:
        LT
        LT?
        ST
        ST?
        Both
        Both?
        Noise

    """
   
    
    if len(predictions)!=np.shape(spec_id)[0]:
        print('ERROR: Number of labels is not equal to number of spectrograms' )
        
#    thresholds for assessing label
    thr1=10
    thr2=70
    
    # Assesting file label
    for i in range(file_number):
        file = segments_filewise_test[i][0]
        #inizialization
        #vectors storing classes probabilities
        LT_prob=[] #class 0
        ST_prob=[] #class 1
        NT_prob=[] #class 2
        spec_num=0   #counts number of spectrograms per file
        #flag: if no click detected no spectrograms
        click_detected_flag=False
        #        looking for all the spectrogram related to this file

        for k in range(np.shape(spec_id)[0]):
            if spec_id[k][0]==file:
                click_detected_flag=True
                spec_num+=1
                LT_prob.append(predictions[k][0])
                ST_prob.append(predictions[k][1])
                NT_prob.append(predictions[k][2])
        if click_detected_flag==True:
            #mean
            LT_mean=np.mean(LT_prob)*100
            ST_mean=np.mean(ST_prob)*100
    
            #best3mean
            #inizialization
            LT_best3mean=0
            ST_best3mean=0
            
            #LT
            ind = np.array(LT_prob).argsort()[-3:][::-1]
        #    adding len ind in order to consider also the cases when we do not have 3 good examples
            if len(ind)==1:
                #this means that there is only one prob!
                LT_best3mean+=LT_prob[0]
            else:
                for j in range(len(ind)):
                    LT_best3mean+=LT_prob[ind[j]]
            LT_best3mean/= 3
            LT_best3mean*=100
            
            #ST
            ind = np.array(ST_prob).argsort()[-3:][::-1]
        #    adding len ind in order to consider also the cases when we do not have 3 good examples
            if len(ind)==1:
                #this means that there is only one prob!
                ST_best3mean+=ST_prob[0]
            else:
                for j in range(len(ind)):
                    ST_best3mean+=ST_prob[ind[j]]
            ST_best3mean/= 3
            ST_best3mean*=100
            
            #ASSESSING FILE LABEL
            #Noise
            if LT_mean<thr1 and ST_mean<thr1 and LT_best3mean<thr2 and ST_best3mean<thr2:
                label='Noise'
            elif LT_mean>=thr1 and ST_mean<thr1 and LT_best3mean>=thr2 and ST_best3mean<thr2:
                label='LT'
            elif LT_mean>=thr1 and ST_mean<thr1 and LT_best3mean<thr2:
                label='LT?'
            elif LT_mean>=thr1 and ST_mean<thr1 and LT_best3mean>=thr2 and ST_best3mean>=thr2:
                label='LT?'
            elif LT_mean<thr1 and ST_mean<thr1 and LT_best3mean>=thr2 and ST_best3mean<thr2:
                label='LT?'
            elif LT_mean<thr1 and ST_mean>=thr1 and LT_best3mean<thr2 and ST_best3mean>=thr2:
                label='ST'
            elif LT_mean<thr1 and ST_mean>=thr1 and LT_best3mean>=thr2:
                label='ST?'
            elif LT_mean<thr1 and ST_mean>=thr1 and LT_best3mean<thr2 and ST_best3mean<thr2:
                label='ST?'
            elif LT_mean<thr1 and ST_mean<thr1 and LT_best3mean<thr2 and ST_best3mean>=thr2:
                label='ST?'
            elif LT_mean>=thr1 and ST_mean>=thr1 and LT_best3mean>=thr2 and ST_best3mean>=thr2:
                label='Both'
            elif LT_mean>=thr1 and ST_mean>=thr1 and LT_best3mean<thr2:
                label='Both?'
            elif LT_mean>=thr1 and ST_mean>=thr1 and LT_best3mean>=thr2 and ST_best3mean<thr2:
                label='Both?'
            elif LT_mean<thr1 and ST_mean<thr1 and LT_best3mean>=thr2 and ST_best3mean>=thr2:
                label='Both?'
            
        else:
#            if no clicks => automatically Noise
            label='Noise'

        filewise_output[i][3] = label
        
    return filewise_output

def metrics(confusion_matrix, file_num):
    """
    Compute Recall, Precision, Accuracy pre and post possible classes check
    for each method
    
    INPUT:
        confusion_matrix is a tensor (3, 7, 4) that stores the confusion matrix
                         for each method
                         
    OUTPUT:
        Recall -> vector, recall for each method
                  TD/TD+FND this metric doesn't change before and after check
                  
        Precision_pre -> vector, precision for each method before check
                         TD/TD+FD 
                         
        Precision_post -> vector, precision for each method after check
                         TD/TD+FD 
                         
        Accuracy_pre -> vector, accuracy for each method before check
                         #correct classified/#files
                         for correct classifications we don't count possible 
                         classes
                         
       Accuracy_post -> vector, accuracy for each method after check
                         #correct classified/#files
                         for correct classifications we count possible classes
                                                  
    """
    
    #inizialization
    Recall=0
    Precision_pre=0
    Precision_post=0
    Accuracy_pre1=0
    Accuracy_pre2=0
    Accuracy_post=0
    #counting
    TD=np.sum(confusion_matrix[0][0:3])+np.sum(confusion_matrix[1][0:3])+np.sum(confusion_matrix[2][0:3])+ np.sum(confusion_matrix[3][0:3])+np.sum(confusion_matrix[4][0:3])+ np.sum(confusion_matrix[5][0:3])
    FND=np.sum(confusion_matrix[6][0:3])
    FPD_pre=confusion_matrix[0][3]+confusion_matrix[1][3]+confusion_matrix[2][3]+confusion_matrix[3][3]+confusion_matrix[4][3]+confusion_matrix[5][3]
    FPD_post=confusion_matrix[0][3]+confusion_matrix[2][3] + confusion_matrix[4][3]
    CoCla_pre1= confusion_matrix[0][0]+confusion_matrix[2][1] + confusion_matrix[4][2]+confusion_matrix[6][3]
    CoCla_pre2=CoCla_pre1+confusion_matrix[1][0]+confusion_matrix[3][1]+confusion_matrix[5][2]
    CoCla_post=CoCla_pre1+np.sum(confusion_matrix[1][:])+np.sum(confusion_matrix[3][:])+np.sum(confusion_matrix[5][:])
        
    #print
    #chck
    print('number of files =', file_num)
    print('TD =',TD)
    print('FND =',FND)
    print('FPD_pre =', FPD_pre)
    print('FPD_post =', FPD_post)
    print('Correct classifications pre1 =', CoCla_pre1)
    print('Correct classifications pre2 =', CoCla_pre2)
    print('Correct classifications post =', CoCla_post)
    #printng metrics
    print("-------------------------------------------")
    print("Click Detector stats on Testing Data")
    if TD==0:
        Recall=0
        Precision_pre= 0
        Precision_post=0
    else:
        Recall= TD/(TD+FND)*100
        Precision_pre=TD/(TD+FPD_pre)*100
        Precision_post=TD/(TD+FPD_post)*100
    print('Recall ', Recall)
    print('Precision_pre ', Precision_pre)
    print('Precision_post ', Precision_post)
    
    if CoCla_pre1==0:
        Accuracy_pre1=0
    else:
       Accuracy_pre1=(CoCla_pre1/file_num)*100
       
    if CoCla_pre2==0:
         Accuracy_pre2=0
    else:
       Accuracy_pre2=(CoCla_pre2/file_num)*100
       
    if CoCla_post==0:
        Accuracy_post=0
    else:
       Accuracy_post=(CoCla_post/file_num)*100
   
    print('Accuracy_pre1', Accuracy_pre1)
    print('Accuracy_pre2', Accuracy_pre2)
    print('Accuracy_post', Accuracy_post)
    

    return Recall, Precision_pre, Precision_post, Accuracy_pre1,Accuracy_pre2, Accuracy_post, TD, FPD_pre, FPD_post, FND, CoCla_pre1, CoCla_pre2, CoCla_post



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

#search clicks
for i in range(file_number_train):
    file = segments_filewise_train[i][0]
    click_label, train_featuress, count = ClickSearch(train_dir, file, train_featuress, count, Train=True)
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
    
# Detect clicks in Test Dataset and save it without labels 
    
test_dir = "D:\Desktop\Documents\Work\Data\Bat\BAT\CNN experiment\TEST2" #changed directory
annotation_file_test= "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\Test_dataset_images.data"
test_fold= "BAT SEARCH TESTS\Test_Spec_58" #Test folder where to save all the stats
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
    click_label, test_featuress, count_end = ClickSearch(test_dir, file, test_featuress, count_start, Train=False)
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
    
#saving dataset
with open(test_dir+'\\'+test_fold +'\\sgramdata_train.json', 'w') as outfile:
    json.dump(train_featuress, outfile)
    
#Train CNN

data_train=train_featuress
print('check',np.shape(data_train))
#sg_train=np.ndarray(shape=(np.shape(data_train)[0],np.shape(data_train[0][0])[0], np.shape(data_train[0][0])[1]), dtype=float) #check
#target_train = np.zeros((np.shape(data_train)[0], 1)) #label train
sg_train_0=[]
sg_train_1=[]
sg_train_2=[]
for i in range(np.shape(data_train)[0]):
    maxg = np.max(data_train[i][0][:])
    if data_train[i][-1]==0:
        sg_train_0.append(data_train[i][0][:]/maxg)
    elif data_train[i][-1]==1:
        sg_train_1.append(data_train[i][0][:]/maxg)
    elif data_train[i][-1]==2:
        sg_train_2.append(data_train[i][0][:]/maxg)
#    target_train[i][0] = data_train[i][-1]
    
#### DATA AGUMENTATION ######################
# create image data augmentation generator for in-build
datagen1 = ImageDataGenerator(width_shift_range=0.5, fill_mode='nearest')
batch_size = 32

# class_0
samples = expand_dims(sg_train_0, np.shape(sg_train_0)[0])
# prepare iterator
it1 = datagen1.flow(samples, batch_size=batch_size)
# generate samples
batch_train_0 = it1.next()
for i in range(int(np.round((1500-np.shape(sg_train_0)[0])/batch_size))):
    batch = it1.next()
    batch_train_0 = np.vstack((batch_train_0, batch))
    
print('sg_train_0 shape', np.shape(sg_train_0))
    
print('batch_train_0 shape', np.shape(batch_train_0))
#add to original ones
batch_train_0=np.reshape(batch_train_0,np.shape(batch_train_0)[0:3])
print('batch_train_0 shape after reshape', np.shape(batch_train_0))
sg_train_0_ag=np.concatenate((sg_train_0,batch_train_0), axis=0)
#label
target_train_0=np.zeros(np.shape(sg_train_0_ag)[0])

# class_1
samples = expand_dims(sg_train_1, np.shape(sg_train_1)[0])
# prepare iterator
it1 = datagen1.flow(samples, batch_size=batch_size)
# generate samples
batch_train_1 = it1.next()
for i in range(int(np.round((1500-np.shape(sg_train_1)[0])/batch_size))):
    batch = it1.next()
    batch_train_1 = np.vstack((batch_train_1, batch))

print('sg_train_1 shape', np.shape(sg_train_1))
    
print('batch_train_1 shape', np.shape(batch_train_1))
#add to original ones
batch_train_1=np.reshape(batch_train_1,np.shape(batch_train_1)[0:3])
print('batch_train_1 shape after reshape', np.shape(batch_train_1))
#add to original ones
sg_train_1_ag=np.concatenate((sg_train_1,batch_train_1), axis=0)
#label
target_train_1=np.ones(np.shape(sg_train_1_ag)[0])
# class_2
samples = expand_dims(sg_train_2, np.shape(sg_train_2)[0])
# prepare iterator
it1 = datagen1.flow(samples, batch_size=batch_size)
# generate samples
batch_train_2 = it1.next()
for i in range(int(np.round((9000-np.shape(sg_train_2)[0])/batch_size))):
    batch = it1.next()
    batch_train_2 = np.vstack((batch_train_2, batch))
    
print('sg_train_2 shape', np.shape(sg_train_2))
    
print('batch_train_2 shape', np.shape(batch_train_2))
#add to original ones
batch_train_2=np.reshape(batch_train_2,np.shape(batch_train_2)[0:3])
print('batch_train_2 shape after reshape', np.shape(batch_train_2))
#add to original ones
sg_train_2_ag=np.concatenate((sg_train_2,batch_train_2), axis=0)
#label
target_train_2=2*np.ones(np.shape(sg_train_2_ag)[0])

#unify train dataset
sg_train=np.concatenate((sg_train_0_ag, sg_train_1_ag, sg_train_2_ag), axis=0)
#sg_train.extend(batch_train_1)
#sg_train.extend(batch_train_2)
sg_train=np.asarray(sg_train, dtype=float)
print('check sg_train shape', np.shape(sg_train))
#unify labels
target_train=np.concatenate((target_train_0, target_train_1, target_train_2), axis=0)
print('check target_train shape', np.shape(target_train))
#target_train=[target_train_0[:][0], target_train_1[:][0], target_train_2[:][0]]
#target_train.extend(target_train_1)
#target_train.extend(target_train_2)

#validation data
# randomly choose 90% train data and keep the rest as validation data
idxs = np.random.permutation(np.shape(sg_train)[0])
x_train = sg_train[idxs[0:int(len(idxs)*0.9)]]
y_train = target_train[idxs[0:int(len(idxs)*0.9)]]
x_validation = sg_train[idxs[int(len(idxs)*0.9):]]
y_validation = target_train[idxs[int(len(idxs)*0.9):]]
 
#Check: how many files I am using for training
print("-------------------------------------------")
print('Number of spectrograms', np.shape(target_train)[0]) 
print('Spectrogram used for training ',np.shape(y_train)[0])
print("Spectrograms for LT: ", np.shape(np.nonzero(y_train==0))[1])
print("Spectrograms for ST: ", np.shape(np.nonzero(y_train==1))[1])
print("Spectrograms for Noise: ", np.shape(np.nonzero(y_train==2))[1])
print('\n Spectrogram used for validation ',np.shape(y_validation)[0])
print("Spectrograms for LT: ", np.shape(np.nonzero(y_validation==0))[1])
print("Spectrograms for ST: ", np.shape(np.nonzero(y_validation==1))[1])
print("Spectrograms for Noise: ", np.shape(np.nonzero(y_validation==2))[1])

#Save training info into a file
cnn_train_info_file=test_dir+'/'+test_fold+'\CNN_train_data_info.txt'
file1=open(cnn_train_info_file,'w')
L=['Number of spectrograms = %5d \n' %np.shape(target_train)[0], 
   "Spectrogram used for training = %5d \n" %np.shape(y_train)[0],
   "Spectrograms for LT: %5d \n" %np.shape(np.nonzero(y_train==0))[1],
   "Spectrograms for ST: %5d \n" %np.shape(np.nonzero(y_train==1))[1],
   "Spectrograms for Noise: %5d \n" %np.shape(np.nonzero(y_train==2))[1],
   "\n Spectrogram used for validation = %5d \n" %np.shape(y_validation)[0],
   "Spectrograms for LT: %5d \n" %np.shape(np.nonzero(y_validation==0))[1],
   "Spectrograms for ST: %5d \n" %np.shape(np.nonzero(y_validation==1))[1],
   "Spectrograms for Noise: %5d \n" %np.shape(np.nonzero(y_validation==2))[1]]
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
#x_train = sg_train
#y_train = target_train
x_test = sg_test
#y_test = target_test

train_images = x_train.reshape(x_train.shape[0],6, 512, 1) #changed image dimensions
validation_images = x_validation.reshape(x_validation.shape[0],6, 512, 1)
test_images = x_test.reshape(x_test.shape[0],6, 512, 1)
input_shape = (6, 512, 1)

train_images = train_images.astype('float32')
validation_images = validation_images.astype('float32')
test_images = test_images.astype('float32')

num_labels=3 #change this variable, when changing number of labels
train_labels = tensorflow.keras.utils.to_categorical(y_train, num_labels)
validation_labels = tensorflow.keras.utils.to_categorical(y_validation, num_labels)
#test_labels = tensorflow.keras.utils.to_categorical(y_test, 8)   #change this to set labels  

accuracies=np.zeros((10,1)) #initializing accuracies array
model_paths=[] #initializing list where to stor model path
for i in range(10):
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
    
    #adding early stopping
    checkpoint = ModelCheckpoint(test_dir+ '/' + test_fold+"/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')
    history = model.fit(train_images, train_labels,
                    batch_size=32,
                    epochs=35,
                    verbose=2,
                    validation_data=(validation_images, validation_labels),
                    callbacks=[checkpoint, early],
                    shuffle=True)

        #recovering labels
    predictions =model.predict(test_images)
    #predictions is an array #imagesX #of classes which entries are the probabilities
    #for each classes
    
    filewise_output=File_label(predictions, spec_id, segments_filewise_test, filewise_output, file_number )
    
    #compare predicted_annotations with segments_filewise_test
    #evaluate metrics
        
    # inizializing
    confusion_matrix=np.zeros((7,4))
    print('Estimating metrics')
    for j in range(file_number):
        assigned_label= filewise_output[j][3]
        correct_label=segments_filewise_test[j][1]
        if correct_label==assigned_label:
            if correct_label=='Noise':
                confusion_matrix[6][3]+=1
            else:
                if correct_label=='LT':
                    confusion_matrix[0][0]+=1
                elif correct_label=='ST':
                    confusion_matrix[2][1]+=1
                elif correct_label=='Both':
                    confusion_matrix[4][2]+=1
        else:
            if correct_label=='Noise':
                if assigned_label=='LT':
                    confusion_matrix[0][3]+=1
                elif assigned_label=='LT?':
                    confusion_matrix[1][3]+=1
                elif assigned_label=='ST':
                    confusion_matrix[2][3]+=1
                elif assigned_label=='ST?':
                    confusion_matrix[3][3]+=1
                elif assigned_label=='Both':
                    confusion_matrix[4][3]+=1
                elif assigned_label=='Both?':
                    confusion_matrix[5][3]+=1
            elif assigned_label=='Noise':
                if correct_label=='LT':
                    confusion_matrix[6][0]+=1
                elif correct_label=='ST':
                    confusion_matrix[6][1]+=1
                elif correct_label=='Both':
                    confusion_matrix[6][2]+=1
            else:
                if correct_label=='LT':
                    if assigned_label=='LT?':
                        confusion_matrix[1][0]+=1
                    elif assigned_label=='ST':
                        confusion_matrix[2][0]+=1
                    elif assigned_label=='ST?':
                        confusion_matrix[3][0]+=1
                    elif assigned_label=='Both':
                        confusion_matrix[4][0]+=1
                    elif assigned_label=='Both?':
                        confusion_matrix[5][0]+=1
                elif correct_label=='ST':
                    if assigned_label=='LT':
                        confusion_matrix[0][1]+=1
                    elif assigned_label=='LT?':
                        confusion_matrix[1][1]+=1
                    elif assigned_label=='ST?':
                        confusion_matrix[3][1]+=1
                    elif assigned_label=='Both':
                        confusion_matrix[4][1]+=1
                    elif assigned_label=='Both?':
                        confusion_matrix[5][1]+=1
                elif correct_label=='Both':
                    if assigned_label=='LT':
                        confusion_matrix[0][2]+=1
                    elif assigned_label=='LT?':
                        confusion_matrix[1][2]+=1
                    elif assigned_label=='ST':
                        confusion_matrix[2][2]+=1
                    elif assigned_label=='ST?':
                        confusion_matrix[3][2]+=1
                    elif assigned_label=='Both?':
                        confusion_matrix[5][2]+=1
                     
    Recall, Precision_pre, Precision_post, Accuracy_pre1,Accuracy_pre2, Accuracy_post, TD, FPD_pre, FPD_post, FND, CoCla_pre1, CoCla_pre2, CoCla_post=metrics(confusion_matrix, file_number)

    #save reached accuracy
    accuracies[i]=Accuracy_post
    print('Accuracy reached',accuracies[i])
    #save model
    modelpath=test_dir+ '\\' + test_fold + '\\model_'+str(i)+'.h5' #aid variable
    model.save(modelpath)
    model_paths.append(modelpath)

#check what modelgave us better accuracy
index_best_model=np.argmax(accuracies) 
print('Best CNN is ', index_best_model)
print('Best accuracy reached ',accuracies[index_best_model])
modelpath=model_paths[index_best_model] 
   
#recover model
#modelpath= "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\model_3.h5"
model=load_model(modelpath)

#recovering labels
predictions =model.predict(test_images)
#predictions is an array #imagesX #of classes which entries are the probabilities
#for each classes

filewise_output=File_label(predictions, spec_id, segments_filewise_test, filewise_output, file_number )

#compare predicted_annotations with segments_filewise_test
#evaluate metrics
    
# inizializing
comparison_annotations = []
confusion_matrix=np.zeros((7,4))
print('Estimating metrics')
for i in range(file_number):
    assigned_label= filewise_output[i][3]
    correct_label=segments_filewise_test[i][1]
    if correct_label==assigned_label:
        if correct_label=='Noise':
            confusion_matrix[6][3]+=1
        else:
            if correct_label=='LT':
                confusion_matrix[0][0]+=1
            elif correct_label=='ST':
                confusion_matrix[2][1]+=1
            elif correct_label=='Both':
                confusion_matrix[4][2]+=1
    else:
        if correct_label=='Noise':
            if assigned_label=='LT':
                confusion_matrix[0][3]+=1
            elif assigned_label=='LT?':
                confusion_matrix[1][3]+=1
            elif assigned_label=='ST':
                confusion_matrix[2][3]+=1
            elif assigned_label=='ST?':
                confusion_matrix[3][3]+=1
            elif assigned_label=='Both':
                confusion_matrix[4][3]+=1
            elif assigned_label=='Both?':
                confusion_matrix[5][3]+=1
        elif assigned_label=='Noise':
            if correct_label=='LT':
                confusion_matrix[6][0]+=1
            elif correct_label=='ST':
                confusion_matrix[6][1]+=1
            elif correct_label=='Both':
                confusion_matrix[6][2]+=1
        else:
            if correct_label=='LT':
                if assigned_label=='LT?':
                    confusion_matrix[1][0]+=1
                elif assigned_label=='ST':
                    confusion_matrix[2][0]+=1
                elif assigned_label=='ST?':
                    confusion_matrix[3][0]+=1
                elif assigned_label=='Both':
                    confusion_matrix[4][0]+=1
                elif assigned_label=='Both?':
                    confusion_matrix[5][0]+=1
            elif correct_label=='ST':
                if assigned_label=='LT':
                    confusion_matrix[0][1]+=1
                elif assigned_label=='LT?':
                    confusion_matrix[1][1]+=1
                elif assigned_label=='ST?':
                    confusion_matrix[3][1]+=1
                elif assigned_label=='Both':
                    confusion_matrix[4][1]+=1
                elif assigned_label=='Both?':
                    confusion_matrix[5][1]+=1
            elif correct_label=='Both':
                if assigned_label=='LT':
                    confusion_matrix[0][2]+=1
                elif assigned_label=='LT?':
                    confusion_matrix[1][2]+=1
                elif assigned_label=='ST':
                    confusion_matrix[2][2]+=1
                elif assigned_label=='ST?':
                    confusion_matrix[3][2]+=1
                elif assigned_label=='Both?':
                    confusion_matrix[5][2]+=1
            
    comparison_annotations.append([filewise_output[i][0], segments_filewise_test[i][1], assigned_label])
 
Recall, Precision_pre, Precision_post, Accuracy_pre1,Accuracy_pre2, Accuracy_post, TD, FPD_pre, FPD_post, FND, CoCla_pre1, CoCla_pre2, CoCla_post=metrics(confusion_matrix, file_number)

TD_rate= (TD/(file_number-np.sum(confusion_matrix[:][3])))*100
print('True Detected rate', TD_rate)
FPD_pre_rate= (FPD_pre/file_number)*100
print('False Detected rate pre', FPD_pre_rate)
FPD_post_rate= (FPD_post/file_number)*100
print('False Detected rate post', FPD_post_rate)
FND_rate= (FND/file_number)*100
print('False Negative Detected rate', FND_rate)
print(confusion_matrix)
print("-------------------------------------------")
    
#saving Click Detector Stats
cd_metrics_file=test_dir+'\\'+test_fold+'\\bat_detector_stats.txt'
file1=open(cd_metrics_file,"w")
L1=["Bat Detector stats on Testing Data \n"]
L2=['Number of files = %5d \n' %file_number]
L3=['TD = %5d \n' %TD]
L4=['FPD_pre = %5d \n' %FPD_pre]
L5=['FPD_post= %5d \n' %FPD_post]
L6=['FND = %5d \n' %FND]
L7=['Correctly classified files before check (1) = %5d \n' %CoCla_pre1]
L8=['Correctly classified files before check (2) = %5d \n' %CoCla_pre2, 'Correctly classified files after check= %5d \n' %CoCla_post]
L9=["Recall = %3.7f \n" %Recall,"Precision pre = %3.7f \n" %Precision_pre, "Precision post = %3.7f \n" %Precision_post, "Accuracy pre 1= %3.7f \n" %Accuracy_pre1, "Accuracy pre 2= %3.7f \n" %Accuracy_pre2, "Accuracy post = %3.7f \n" %Accuracy_post,  "True Detected rate = %3.7f \n" %TD_rate, "False Detected rate pre = %3.7f \n" %FPD_pre_rate, "False Detected rate post = %3.7f \n" %FPD_post_rate, "False Negative Detected rate = %3.7f \n" %FND_rate]
L10=['Model used %5d \n' %index_best_model]
L11=['Training accuracy for the model %3.7f \n' %accuracies[index_best_model]]
file1.writelines(np.concatenate((L1,L2,L3,L4, L5, L6, L7, L8, L9, L10, L11)))
#file1.writelines(np.concatenate((L1,L2,L3,L4, L5, L6, L7, L8, L9)))

file1.close()
       
#saving compared labels
with open(test_dir+'\\' +test_fold+'\\Test_annotations_comparison.data', 'w') as f:
    json.dump(comparison_annotations,f)
    
with open(test_dir+'\\' +test_fold+"\\Confusion_matrix.txt",'w') as f:
    f.write("Confusion Matrix \n\n")
    np.savetxt(f, confusion_matrix, fmt='%d')

#saving compared labels
with open(test_dir+'\\' +test_fold+'\\Test_filewise_output.data', 'w') as f:
    json.dump(filewise_output,f)

        
    
            
        
