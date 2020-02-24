# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:46:58 2020

@author: Virginia Listanti

This script process automatically the spectrogram images in pre-labelled
directories provided by Moira Pryde and then saves a .txt files with the wrongly
assigned files and other metrics

It uses the CNN model:
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

import librosa
import WaveletSegment
import WaveletFunctions

import cv2  # image -processing
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def generate_filewise_annotations(dirpath, dirname):
    """
    This function generate the filewise annotation for the dataset
    the label comes form directory name
    """
    dataset=[]
    for root, dirs, files in os.walk(os.path.join(dirpath, dirname)):
        for file in files:
            #Work on .bmp data
            if file.endswith('.bmp'):
                if dirname=='Both':
                    label='Both'
                elif dirname=='Long tail':
                    label='LT'
                elif dirname=='Non-bat':
                    label='Noise'
                elif dirname=='Possible LT':
                    label='LT?'
                elif dirname=='Possible ST':
                    label='ST?'
                elif dirname=='Short tail':
                    label='ST'
                elif dirname=='Unassigned':
                    label='Noise'
                elif dirname=='Unknown':
                    label='Noise'
                dataset.append([file, label])              
        
    return dataset
        
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
    
    filename = dirName + '\\' + file
    
    img = mpimg.imread(filename) #read image
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    img2[-1, :] = 254 * np.ones((np.shape(img2[1]))) #cut last row
    imspec=np.repeat(img2,8, axis=0) #repeat rows 7 times to fit invertspectrogram
    imspec = -(imspec - 254.0 * np.ones(np.shape(imspec)))  # reverse value having the black as the most intense
#    imspec=np.flipud(imspec) #reverse up and down of the spectrogram -> see AviaNZ spectrogram
    imspec = imspec/np.max(imspec) #normalization
    imspec = imspec[:, 1:np.shape(img2)[1]]  # Cutting first column because it only contains the scale and cutting last columns
    
    df=88000/(np.shape(imspec)[0]+1) #frequency increment 
    dt=0.002909090909090909
#    up_len=math.ceil(0.05/dt) #0.5 second lenth in indices divided by 11
    up_len=17
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
#    print(np.shape(clicks_indices))
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


#Asses file label: 3 labels, evaluate probability filewise
def File_label(predictions):
    """
    FIle_label use the predictions made by the CNN to update the filewise annotations
    when we have 3 labels: 0 (LT), 1(ST), 2 (Noise)
    
    This version works file by file
    
    segment_filewise_test contains the file annotation
    
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
        
#    thresholds for assessing label
    thr1=10
    thr2=70
    
    # Assesting file label
    #inizialization
    #vectors storing classes probabilities
    LT_prob=[] #class 0
    ST_prob=[] #class 1
    NT_prob=[] #class 2
    spec_num=0   #counts number of spectrograms per file
    #flag: if no click detected no spectrograms
    click_detected_flag=False
    #        looking for all the spectrogram related to this file


# Actually,  We don't need spec_id any more
    
#    for k in range(np.shape(spec_id)[0]):
#        if spec_id[k][0]==file:
#            click_detected_flag=True
#            spec_num+=1
#            LT_prob.append(predictions[k][0])
#            ST_prob.append(predictions[k][1])
#            NT_prob.append(predictions[k][2])
    
    for k in range(np.shape(predictions)[0]):
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
        
    return label

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
    Accuracy_pre=0
    Accuracy_post=0
    #counting
    TD=np.sum(confusion_matrix[0][0:5])+np.sum(confusion_matrix[1][0:5])+np.sum(confusion_matrix[2][0:5])+ np.sum(confusion_matrix[3][0:5])+np.sum(confusion_matrix[4][0:5])+ np.sum(confusion_matrix[5][0:5])
    FND=np.sum(confusion_matrix[6][0:5])
    FPD_pre=confusion_matrix[0][5]+confusion_matrix[1][5]+confusion_matrix[2][5]+confusion_matrix[3][5]+confusion_matrix[4][5]+confusion_matrix[5][5]
    FPD_post=confusion_matrix[0][5]+confusion_matrix[2][5] + confusion_matrix[4][5]
    CoCla_pre= confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1]+confusion_matrix[2][2]+confusion_matrix[2][3] +confusion_matrix[3][3]+ confusion_matrix[3][2]+confusion_matrix[4][4]+confusion_matrix[5][4]+confusion_matrix[6][5]
    CoCla_post=CoCla_pre+np.sum(confusion_matrix[1][2:])+np.sum(confusion_matrix[3][0:2])+np.sum(confusion_matrix[3][4:])+np.sum(confusion_matrix[5][0:4])+confusion_matrix[5][5]
        
    #print
    #chck
    print('number of files =', file_num)
    print('TD =',TD)
    print('FND =',FND)
    print('FPD_pre =', FPD_pre)
    print('FPD_post =', FPD_post)
    print('Correct classifications pre =', CoCla_pre)
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
    
    if CoCla_pre==0:
        Accuracy_pre=0
    else:
       Accuracy_pre=(CoCla_pre/file_num)*100
              
    if CoCla_post==0:
        Accuracy_post=0
    else:
       Accuracy_post=(CoCla_post/file_num)*100
   
    print('Accuracy_pre1', Accuracy_pre)
    print('Accuracy_post', Accuracy_post)
    

    return Recall, Precision_pre, Precision_post, Accuracy_pre, Accuracy_post, TD, FPD_pre, FPD_post, FND, CoCla_pre, CoCla_post


def update_confusion_matrix(file, assigned_label, correct_label, confusion_matrix, comparison_annotations, missed_files, false_positives, misclassified_files):
    """
    This function update the confusion matrix and return 
    """

    if correct_label==assigned_label:
        if correct_label=='Noise':
            confusion_matrix[6][5]+=1
        else:
            if correct_label=='LT':
                confusion_matrix[0][0]+=1
            elif correct_label=='LT?':
                confusion_matrix[1][1]+=1
            elif correct_label=='ST':
                confusion_matrix[2][2]+=1
            elif correct_label=='ST?':
                confusion_matrix[3][3]+=1
            elif correct_label=='Both':
                confusion_matrix[4][4]+=1
    else:
        if correct_label=='Noise':
            false_positives.append([file, correct_label, assigned_label])
            if assigned_label=='LT':
                confusion_matrix[0][5]+=1
            elif assigned_label=='LT?':
                confusion_matrix[1][5]+=1
            elif assigned_label=='ST':
                confusion_matrix[2][5]+=1
            elif assigned_label=='ST?':
                confusion_matrix[3][5]+=1
            elif assigned_label=='Both':
                confusion_matrix[4][5]+=1
            elif assigned_label=='Both?':
                confusion_matrix[5][5]+=1
        elif assigned_label=='Noise':
            missed_files.append([file, correct_label, assigned_label])
            if correct_label=='LT':
                confusion_matrix[6][0]+=1
            elif correct_label=='LT?':
                confusion_matrix[6][1]+=1
            elif correct_label=='ST':
                confusion_matrix[6][2]+=1
            elif correct_label=='ST?':
                confusion_matrix[6][3]+=1
            elif correct_label=='Both':
                confusion_matrix[6][4]+=1
        else:
            if correct_label=='LT':
                if assigned_label=='LT?':
                    confusion_matrix[1][0]+=1
                elif assigned_label=='ST':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[2][0]+=1
                elif assigned_label=='ST?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[3][0]+=1
                elif assigned_label=='Both':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[4][0]+=1
                elif assigned_label=='Both?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[5][0]+=1
            elif correct_label=='LT?':
                if assigned_label=='LT':
                    confusion_matrix[0][1]+=1
                elif assigned_label=='ST':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[2][1]+=1
                elif assigned_label=='ST?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[3][1]+=1
                elif assigned_label=='Both':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[4][1]+=1
                elif assigned_label=='Both?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[5][1]+=1
            elif correct_label=='ST':
                if assigned_label=='LT':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[0][2]+=1
                elif assigned_label=='LT?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[1][2]+=1
                elif assigned_label=='ST?':
                    confusion_matrix[3][2]+=1
                elif assigned_label=='Both':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[4][2]+=1
                elif assigned_label=='Both?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[5][2]+=1
            elif correct_label=='ST?':
                if assigned_label=='LT':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[0][3]+=1
                elif assigned_label=='LT?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[1][3]+=1
                elif assigned_label=='ST':
                    confusion_matrix[2][3]+=1
                elif assigned_label=='Both':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[4][3]+=1
                elif assigned_label=='Both?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[5][3]+=1
            elif correct_label=='Both':
                if assigned_label=='LT':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[0][4]+=1
                elif assigned_label=='LT?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[1][4]+=1
                elif assigned_label=='ST':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[2][4]+=1
                elif assigned_label=='ST?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[3][4]+=1
                elif assigned_label=='Both?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[5][4]+=1
                
        comparison_annotations.append([file, correct_label, assigned_label])
        
    return confusion_matrix, comparison_annotations, missed_files, false_positives, misclassified_files
##############################################################################
    

######################### Main ###############################################
    
bat_dir="D:\Desktop\Documents\Work\Bats\Smaller dataset"
test_dir = "D:\Desktop\Documents\Work\Bats"
test_fold= "BAT SEARCH TESTS SMALL\Test_03" #Test folder where to save all the stats
os.mkdir(test_dir+ '\\' + test_fold)
#list of directory
dirs=os.listdir(bat_dir)
#recover model
modelpath= "D:\Desktop\Documents\Work\Data\Bat\BAT\CNN experiment\TEST2\BAT SEARCH TESTS\Test_Spec_30\model_5.h5"
model=load_model(modelpath)
#inizialization confusion_matrix
confusion_matrix_tot=np.zeros((7,6))
file_number_tot=0
for dirname in dirs:
    print('Analizing folder ', dirname)
        
    #create folder to store results inside the folder
    os.mkdir(bat_dir+ '\\' + dirname+ '\\'+ test_fold)
    
    print('generate dataset for ', dirname)
    segments_filewise_test=generate_filewise_annotations(bat_dir, dirname)
        #save dataset  on dataset directory              
    with open(bat_dir+ '\\' + dirname+ '\\'+ test_fold+'\\dataset.data', 'w') as f2:
        json.dump(segments_filewise_test,f2)
     #save dataset  on test direcotory
    with open(test_dir+ '\\' + test_fold+'\\'+dirname+'_dataset.data', 'w') as f2:
        json.dump(segments_filewise_test,f2)
     
    file_number=len(segments_filewise_test)
  
#    check: if there is noting in the directory: skip 
    if file_number==0:
        print('The folder is empty')
        continue

    #inizializations
   
    filewise_output=[] #here we store; file, click detected (true/false), #of spectrogram, final label
    TD=0
    FD=0
    TND=0
    FND=0
    control_spec=0 #this variable count the file where the click detector found a click that are not providing spectrograms for CNN
    count_cliks=0 #this variable counts clicks in the folder
    
    #inizializing annotations
    comparison_annotations = []
    missed_files=[]
    false_positives=[]
    misclassified_files=[]
    # inizializing folder confusion_matrix
    confusion_matrix=np.zeros((7,6))

#   index_corrupted files is a list of indexes of corrupted files that we will erase
    index_corrupted_files=[]
#    while corrupted_files is the list of the corrupted files
    corrupted_files=[]
    #search clicks
    for i in range(file_number):
        #features are stored for each file
        data_test =[]
        count_start=0
        file = segments_filewise_test[i][0]
        print("Click search on ",file)
        control='False'
#        handling the possibility that we have a problem loading file image
        try:
            click_label, data_test, count_end = ClickSearch(bat_dir+'\\'+dirname, file, data_test, count_start, Train=False)
        except OSError:
            print('Error loading file ', file)
#            if problem loading file we keep track of this 
            index_corrupted_files.append([i])
            corrupted_files.append([file])
#            we assest the rest to make it work
#            we store the information but we will not add the file to the stats
            click_label='None'
            filewise_output.append([file, 'None', 0, 'Corrupted file', segments_filewise_test[i][1]])
#            we need to updat this in order to not screw with the indeces
            print('File classification = Corrupted file')
            continue
        except:
            print('Error loading file ', file)
#            if problem loading file we keep track of this 
            index_corrupted_files.append([i])
            corrupted_files.append([file])
#            we assest the rest to make it work
#            we store the information but we will not add the file to the stats
            click_label='None'
            filewise_output.append([file, 'None', 0, 'Corrupted file', segments_filewise_test[i][1]])
#            we need to updat this in order to not screw with the indeces
            print('File classification = Corrupted file')
            continue
        gen_spec= count_end-count_start # numb. of generated spectrograms
        count_cliks+=gen_spec
            
        print('number of detected clicks = ', gen_spec)
        
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
        
        #recover test data
        #note: I am not giving target!
        if click_label=='Click':
            #we enter in the cnn only if we got a click
            print('check data_test shape ', np.shape(data_test))
            sg_test=np.ndarray(shape=(np.shape(data_test)[0],np.shape(data_test[0][0])[0], np.shape(data_test[0][0])[1]), dtype=float)
            spec_id=[]
            print('Number of file spectrograms', np.shape(data_test)[0])
            for j in range(np.shape(data_test)[0]):
                maxg = np.max(data_test[j][0][:])
                sg_test[j][:] = data_test[j][0][:]/maxg
                spec_id.append(data_test[j][1:3])
    #            
    #        #check:
    #        print('check on spec_id', np.shape(spec_id))  
            
            #CNN classification of clicks
            x_test = sg_test
            test_images = x_test.reshape(x_test.shape[0],6, 512, 1)
            input_shape = (6, 512, 1)
            test_images = test_images.astype('float32')
            
            print('Clicks classification')
            #recovering labels
            predictions =model.predict(test_images)
            #predictions is an array #imagesX #of classes which entries are the probabilities
            #for each classes
            
            print('Assessing file label')
            label=File_label(predictions)
        else:
            label='Noise'
        
        filewise_output[i][3] = label
        print('File classification = ', label)
        #updating confusion matrix
        print ('Updating confusion_matrix')
        confusion_matrix, comparison_annotations, missed_files, false_positives, misclassified_files= update_confusion_matrix(file, label, segments_filewise_test[i][1], confusion_matrix, comparison_annotations, missed_files, false_positives, misclassified_files)
        
    #check if there where corrupted files
#    in case: update segment_filewise_test removing them from it
#              update file numer
    if len(index_corrupted_files)!=0:
        file_number-=len(index_corrupted_files)
        for index in index_corrupted_files:
#            in this way we will have an empty list and not the expected list and we are not screwing with indeces
            del segments_filewise_test[index][:]
        #update dataset  on dataset directory              
        with open(bat_dir+ '\\' + dirname+ '\\'+ test_fold+'\\updated_dataset.data', 'w') as f2:
            json.dump(segments_filewise_test,f2)
         #update dataset  on test direcotory
        with open(test_dir+ '\\' + test_fold+'\\'+dirname+'_updated_dataset.data', 'w') as f2:
            json.dump(segments_filewise_test,f2)
            
        #store corrupted files on dataset directory            
        with open(bat_dir+ '\\' + dirname+ '\\'+ test_fold+'\\corrupted_files.data', 'w') as f2:
            json.dump(corrupted_files,f2)
         #store corrupted files on test direcotory
        with open(test_dir+ '\\' + test_fold+'\\'+dirname+'_corrupted_files.data', 'w') as f2:
            json.dump(corrupted_files,f2)
            
#    ############## ATTENTION #######################
#    This update should match with  filewise_output (finger crossed)
#    #############################################        
    #updating total file_number    
    file_number_tot+=file_number
    #updating total confusion_matrix
    confusion_matrix_tot+=confusion_matrix 
    #printing metrics
    print("-------------------------------------------")
    print('Number of detected click', count_cliks, 'in ', dirname, 'with ', file_number, ' files')
    print("Click Detector stats on Testing Data")
    if TD==0:
        Recall=0
    else:
        Recall= TD/(TD+FND)*100
    print('Recall ', Recall)
    if TD==0:
        Precision=0
    else:
        Precision= TD/(TD+FD)*100
    print('Precision ', Precision)
    if TD==0 and TND==0:
        Accuracy=0
    else:
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
    cd_metrics_file=bat_dir+ '\\' + dirname+ '\\'+ test_fold+'\\click_detector_stats.txt'
    file1=open(cd_metrics_file,"w")
    L0=["Number of file %5d \n"  %file_number]
    L1=["Number of detected clicks %5d \n" %count_cliks ]
    L2=["NUmber of file with detected clicks but not spectrograms for CNN = %5d \n " %control_spec]
    L3=["Recall = %3.7f \n" %Recall,"Precision = %3.7f \n" %Precision, "Accuracy = %3.7f \n" %Accuracy, "True Detected rate = %3.7f \n" %TD_rate, "False Detected rate = %3.7f \n" %FD_rate, "True Negative Detected rate = %3.7f \n" %TND_rate, "False Negative Detected rate = %3.7f \n" %FND_rate ]
    file1.writelines(np.concatenate((L0,L1,L2,L3)))
    file1.close()
    
       
    #saving compared labels
    with open(bat_dir+ '\\' + dirname+ '\\'+ test_fold+'\\annotations_comparison.data', 'w') as f:
        json.dump(comparison_annotations,f)
        
    #saving missed_fils
    with open(bat_dir+ '\\' + dirname+ '\\'+ test_fold+'\\missed_files.data', 'w') as f:
        json.dump(missed_files,f)

    #saving false positives
    with open(bat_dir+ '\\' + dirname+ '\\'+ test_fold+'\\false_positives.data', 'w') as f:
        json.dump(false_positives,f)        
    
    #saving misclassified files
    with open(bat_dir+ '\\' + dirname+ '\\'+ test_fold+'\\misclassified_files.data', 'w') as f:
        json.dump(misclassified_files,f)
        
    #saving compared labels
    with open(bat_dir+ '\\' + dirname+ '\\'+ test_fold+'\\filewise_output.data', 'w') as f:
        json.dump(filewise_output,f)
    
    
    # Evaluating metrics for this folder
    print('Evaluating BatSearch performance on ', dirname)
    Recall, Precision_pre, Precision_post, Accuracy_pre, Accuracy_post, TD, FPD_pre, FPD_post, FND, CoCla_pre, CoCla_post=metrics(confusion_matrix, file_number)
    
    #print metrics
    print("-------------------------------------------")
    print('Classification performance on ', dirname)
    TD_rate= (TD/(file_number-np.sum(confusion_matrix[:][5])))*100
    print('True Detected rate', TD_rate)
    FPD_pre_rate= (FPD_pre/file_number)*100
    print('False Detected rate pre', FPD_pre_rate)
    FPD_post_rate= (FPD_post/file_number)*100
    print('False Detected rate post', FPD_post_rate)
    FND_rate= (FND/file_number)*100
    print('False Negative Detected rate', FND_rate)
    print(confusion_matrix)
    print("-------------------------------------------")
    
    with open(bat_dir+ '\\' + dirname+ '\\'+ test_fold+'\\Confusion_matrix.txt','w') as f:
        f.write("Confusion Matrix \n\n")
        np.savetxt(f, confusion_matrix, fmt='%d')
        
    #saving Click Detector Stats
    cd_metrics_file=bat_dir+ '\\' + dirname+'\\'+test_fold+'\\bat_detector_stats.txt'
    file1=open(cd_metrics_file,"w")
    L1=["Bat Detector stats on Testing Data \n"]
    L2=['Number of files = %5d \n' %file_number]
    L3=['TD = %5d \n' %TD]
    L4=['FPD_pre = %5d \n' %FPD_pre]
    L5=['FPD_post= %5d \n' %FPD_post]
    L6=['FND = %5d \n' %FND]
    L7=['Correctly classified files before check = %5d \n' %CoCla_pre]
    L8=['Correctly classified files after check= %5d \n' %CoCla_post]
    L9=["Recall = %3.7f \n" %Recall,"Precision pre = %3.7f \n" %Precision_pre, "Precision post = %3.7f \n" %Precision_post, "Accuracy pre = %3.7f \n" %Accuracy_pre,  "Accuracy post = %3.7f \n" %Accuracy_post,  "True Detected rate = %3.7f \n" %TD_rate, "False Detected rate pre = %3.7f \n" %FPD_pre_rate, "False Detected rate post = %3.7f \n" %FPD_post_rate, "False Negative Detected rate = %3.7f \n" %FND_rate]
    L10=['Model used ', modelpath, '\n']
    #L11=['Training accuracy for the model %3.7f \n' %accuracies[index_best_model]]
    file1.writelines(np.concatenate((L1,L2,L3,L4, L5, L6, L7, L8, L9, L10)))
    #file1.writelines(np.concatenate((L1,L2,L3,L4, L5, L6, L7, L8, L9)))
    file1.close()
    
# Evaluating metrics at the end of the process
print('Evualuating Bat Search performance on the entire Dataset')
Recall, Precision_pre, Precision_post, Accuracy_pre, Accuracy_post, TD, FPD_pre, FPD_post, FND, CoCla_pre, CoCla_post=metrics(confusion_matrix_tot, file_number_tot)

#print metrics
print("-------------------------------------------")
print('Classification performance on the entire dataset')
print('Number of analized files =', file_number_tot)
TD_rate= (TD/(file_number_tot-np.sum(confusion_matrix_tot[:][5])))*100
print('True Detected rate', TD_rate)
FPD_pre_rate= (FPD_pre/file_number_tot)*100
print('False Detected rate pre', FPD_pre_rate)
FPD_post_rate= (FPD_post/file_number_tot)*100
print('False Detected rate post', FPD_post_rate)
FND_rate= (FND/file_number_tot)*100
print('False Negative Detected rate', FND_rate)
print(confusion_matrix_tot)
print("-------------------------------------------")

with open(test_dir+'\\' +test_fold+"\\Confusion_matrix.txt",'w') as f:
    f.write("Confusion Matrix \n\n")
    np.savetxt(f, confusion_matrix_tot, fmt='%d')
    
#saving Click Detector Stats
cd_metrics_file=test_dir+'\\'+test_fold+'\\bat_detector_stats.txt'
file1=open(cd_metrics_file,"w")
L1=["Bat Detector stats on Testing Data \n"]
L2=['Number of files = %5d \n' %file_number_tot]
L3=['TD = %5d \n' %TD]
L4=['FPD_pre = %5d \n' %FPD_pre]
L5=['FPD_post= %5d \n' %FPD_post]
L6=['FND = %5d \n' %FND]
L7=['Correctly classified files before check = %5d \n' %CoCla_pre]
L8=['Correctly classified files after check= %5d \n' %CoCla_post]
L9=["Recall = %3.7f \n" %Recall,"Precision pre = %3.7f \n" %Precision_pre, "Precision post = %3.7f \n" %Precision_post, "Accuracy pre = %3.7f \n" %Accuracy_pre,  "Accuracy post = %3.7f \n" %Accuracy_post,  "True Detected rate = %3.7f \n" %TD_rate, "False Detected rate pre = %3.7f \n" %FPD_pre_rate, "False Detected rate post = %3.7f \n" %FPD_post_rate, "False Negative Detected rate = %3.7f \n" %FND_rate]
L10=['Model used ', modelpath, '\n']
#L11=['Training accuracy for the model %3.7f \n' %accuracies[index_best_model]]
file1.writelines(np.concatenate((L1,L2,L3,L4, L5, L6, L7, L8, L9, L10)))
#file1.writelines(np.concatenate((L1,L2,L3,L4, L5, L6, L7, L8, L9)))
file1.close()