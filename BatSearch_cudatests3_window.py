"""
Created on 1/10/2020

Updated 19/10/2020

@author: Virginia Listanti

This code generate different trained CNN and test them on the folder Moira2020

Different test configuration:
- click search freq. bands: [24k, 54k], [21k, 60k]
- window: 3pxl doubled, 7 pxl, 17,  31 pxl
[- augumentation y/n]


Pipeline:
-create train dataset: clicksearch + read annotations
- 75% training, 25% validation
- [agumentation(y/n)]
- Cnn train
- test on dataset
- generate annotations

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
                featuress=updateDataset(imspec, window, segments, click_start, click_end, featuress, Train)
                
            # update
            click_start=clicks_indices[0][i]
            click_end=clicks_indices[0][i]

    # checking last loop with end
    if click_end-click_start+1>up_len:
        clicks[click_start:click_end+1] = 0
    else:
        count+=1
        #savedataset
        featuress=updateDataset(imspec, window, segments, click_start, click_end, featuress, Train)

    #Assigning: click label
    if count==0:
        label='None'
    else:
        label='Click'

    print(count, ' clicks found')
        
    return label, featuress 


def updateDataset(spectrogram, window, segments, click_start, click_end, featuress, Train=False):
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

def File_label_0(predictions, thr1, thr2, dir_2_save, bmpfile):
    """
    Standard File Label.

    uses the predictions made by the CNN to update the filewise annotations
    when we have 3 labels: 0 (LT), 1(ST), 2 (Noise)

    METHOD: evaluation of probability over files combining mean of probability
        + best3mean of probability against thr1 and thr2, respectively

    Returns: species labels (list of dicts), compatible w/ the label format on Segments
    """

    # Assessing file label
    # inizialization
    # vectors storing classes probabilities
    LT_prob=[]  # class 0
    ST_prob=[]  # class 1
    NT_prob=[]  # class 2
    spec_num=0   # counts number of spectrograms per file
    # flag: if no click detected no spectrograms
    click_detected_flag=False
    # looking for all the spectrogram related to this file

    for k in range(np.shape(predictions)[0]):
        click_detected_flag=True
        spec_num+=1
        LT_prob.append(predictions[k][0])
        ST_prob.append(predictions[k][1])
        NT_prob.append(predictions[k][2])


    # if no clicks => automatically Noise
    label = []

    # best3mean
    LT_best3mean=0
    ST_best3mean=0

    if click_detected_flag:
        # mean
        LT_mean=np.mean(LT_prob)*100
        ST_mean=np.mean(ST_prob)*100

        # LT
        ind = np.array(LT_prob).argsort()[-3:][::-1]
        # adding len ind in order to consider also the cases when we do not have 3 good examples
        if len(ind)==1:
            # this means that there is only one prob!
            LT_best3mean+=LT_prob[0]
        else:
            for j in range(len(ind)):
                LT_best3mean+=LT_prob[ind[j]]
        LT_best3mean/= 3
        LT_best3mean*=100

        # ST
        ind = np.array(ST_prob).argsort()[-3:][::-1]
        # adding len ind in order to consider also the cases when we do not have 3 good examples
        if len(ind)==1:
            # this means that there is only one prob!
            ST_best3mean+=ST_prob[0]
        else:
            for j in range(len(ind)):
                ST_best3mean+=ST_prob[ind[j]]
        ST_best3mean/= 3
        ST_best3mean*=100

        # ASSESSING FILE LABEL
        hasST = ST_mean>=thr1 or ST_best3mean>=thr2
        hasLT = LT_mean>=thr1 or LT_best3mean>=thr2
        hasSTlow = ST_mean<thr1 and ST_best3mean<thr2
        hasLTlow = LT_mean<thr1 and LT_best3mean<thr2
        reallyHasST = ST_mean>=thr1 and ST_best3mean>=thr2
        reallyHasLT = LT_mean>=thr1 and LT_best3mean>=thr2
        HasBat = LT_mean>=thr1 and ST_mean>=thr1

        if reallyHasLT and hasSTlow:
            label.append({"species": "Long-tailed bat", "certainty": 100})
        elif reallyHasLT and reallyHasST:
            label.append({"species": "Long-tailed bat", "certainty": 100})
        elif hasLT and ST_mean<thr1:
            label.append({"species": "Long-tailed bat", "certainty": 50})
        elif HasBat:
            label.append({"species": "Long-tailed bat", "certainty": 50})

        if reallyHasST and hasLTlow:
            label.append({"species": "Short-tailed bat", "certainty": 100})
        elif reallyHasLT and reallyHasST:
            label.append({"species": "Short-tailed bat", "certainty": 100})
        elif hasST and LT_mean<thr1:
            label.append({"species": "Short-tailed bat", "certainty": 50})
        elif HasBat:
            label.append({"species": "Short-tailed bat", "certainty": 50})

    if len(label)==0:
        file_label='Noise'
        cert_label=100
    elif len(label)==2:
        file_label='Both'
        cert_label=np.maximum(label[0]['certainty'],label[1]['certainty'])
    else:
        file_label=label[0]['species']
        cert_label=label[0]['certainty']


    #path to save
    image_path=dir_2_save+'/'+bmpfile[:-4]+"_click_prob.png"

    #plot clicks probabilities + save images
        #Each row has a differen plot: LT, ST NT
        # Lt first columin, st second, nt third
    fig, ax = plt.subplots()
    ax.plot(LT_prob, 'ro', label='LT' )
    ax.plot(ST_prob, 'ob', label='ST')
    ax.plot(NT_prob,'og', label='Not-bat')
       
    ax.set_title('File classified as '+file_label+' cert ='+str(cert_label)+', num. clicks = '+str(len(LT_prob)))
        
    legend = ax.legend(loc='upper right')
        
    plt.savefig(image_path)
 

    return label, LT_best3mean, ST_best3mean

def File_label_1(predictions, thr1, thr2, dir_2_save, bmpfile):
    """
    Stephen File Label.

    uses the predictions made by the CNN to update the filewise annotations
    when we have 3 labels: 0 (LT), 1(ST), 2 (Noise)

    METHOD: 
    - if #clicks<4 => Noise
    - if best4mean(class k)> 85 => class k prob 100%
    - elif best4mean(class k)> 85 => class k prob 50%

    Returns: species labels (list of dicts), compatible w/ the label format on Segments
    """

    # Assessing file label
    # inizialization
    # vectors storing classes probabilities
    LT_prob=[]  # class 0
    ST_prob=[]  # class 1
    NT_prob=[]  # class 2
    spec_num=0   # counts number of spectrograms per file
    # flag: if no click detected no spectrograms
    click_detected_flag=False
    # looking for all the spectrogram related to this file

    num_clicks=np.shape(predictions)[0]

    # if no clicks => automatically Noise
    label = []

    # best3mean
    LT_best4mean=0
    ST_best4mean=0

    for k in range(num_clicks):
            click_detected_flag=True
            spec_num+=1
            LT_prob.append(predictions[k][0])
            ST_prob.append(predictions[k][1])
            NT_prob.append(predictions[k][2])


    if num_clicks>=4:

        if click_detected_flag:
            # mean
            LT_mean=np.mean(LT_prob)*100
            ST_mean=np.mean(ST_prob)*100


            # LT
            ind = np.array(LT_prob).argsort()[-4:][::-1]
            # adding len ind in order to consider also the cases when we do not have 3 good examples
            if len(ind)==1:
                # this means that there is only one prob!
                LT_best4mean+=LT_prob[0]
            else:
                for j in range(len(ind)):
                    LT_best4mean+=LT_prob[ind[j]]
            LT_best4mean/= 4
            LT_best4mean*=100

            # ST
            ind = np.array(ST_prob).argsort()[-4:][::-1]
            # adding len ind in order to consider also the cases when we do not have 3 good examples
            if len(ind)==1:
                # this means that there is only one prob!
                ST_best4mean+=ST_prob[0]
            else:
                for j in range(len(ind)):
                    ST_best4mean+=ST_prob[ind[j]]
            ST_best4mean/= 4
            ST_best4mean*=100

            # ASSESSING FILE LABEL
            hasST = ST_best4mean>=thr2
            hasLT = LT_best4mean>=thr2
            hasSTlow =ST_best4mean>thr1
            hasLTlow = LT_best4mean>thr1
            
            if reallyHasLT:
                label.append({"species": "Long-tailed bat", "certainty": 100})
            elif hasLTlow:
                label.append({"species": "Long-tailed bat", "certainty": 50})

            if reallyHasST:
                label.append({"species": "Short-tailed bat", "certainty": 100})
            elif hasSTlow:
                label.append({"species": "Short-tailed bat", "certainty": 50})

    if len(label)==0:
        file_label='Noise'
        cert_label=100
    elif len(label)==2:
        file_label='Both'
        cert_label=np.maximum(label[0]['certainty'],label[1]['certainty'])
    else:
        file_label=label[0]['species']
        cert_label=label[0]['certainty']


    #path to save
    image_path=dir_2_save+'/'+bmpfile[:-4]+"_click_prob.png"

    #plot clicks probabilities + save images
        #Each row has a differen plot: LT, ST NT
        # Lt first columin, st second, nt third
    fig, ax = plt.subplots()
    ax.plot(LT_prob, 'ro', label='LT' )
    ax.plot(ST_prob, 'ob', label='ST')
    ax.plot(NT_prob,'og', label='Not-bat')
       
    ax.set_title('File classified as '+file_label+' cert ='+str(cert_label)+', num. clicks = '+str(len(LT_prob)))
        
    legend = ax.legend(loc='upper right')
        
    plt.savefig(image_path)
 
    return label, LT_best4mean, ST_best4mean


def File_label_2(predictions, thr1, thr2, thr3, dir_2_save, bmpfile):
    """
    My File Label.

    uses the predictions made by the CNN to update the filewise annotations
    when we have 3 labels: 0 (LT), 1(ST), 2 (Noise)

    METHOD: 
    - if #clicks<4 => Noise
    - if best4mean(class k)> 85 => class k prob 100%
    - elif best4mean(class k)> 85 => class k prob 50%

    Returns: species labels (list of dicts), compatible w/ the label format on Segments
    """

    # Assessing file label
    # inizialization
    # vectors storing classes probabilities
    LT_prob=[]  # class 0
    ST_prob=[]  # class 1
    NT_prob=[]  # class 2
    spec_num=0   # counts number of spectrograms per file
    # flag: if no click detected no spectrograms
    click_detected_flag=False
    # looking for all the spectrogram related to this file

    num_clicks=np.shape(predictions)[0]

    # if no clicks => automatically Noise
    label = []

    #counting feasible LT/ST clicks
    LT_clicks=0
    ST_clicks=0

    #inizializing mean
    LT_mean=0
    ST_mean=0

    for k in range(num_clicks):
        click_detected_flag=True
        spec_num+=1
        LT_prob.append(predictions[k][0])
        ST_prob.append(predictions[k][1])
        NT_prob.append(predictions[k][2])

        if predictions[k][0]*100>thr1:
            LT_mean+=predictions[k][0]
            LT_clicks+=1

        if predictions[k][1]*100>thr1:
            ST_mean+=predictions[k][1]
            ST_clicks+=1
    
    if LT_mean>0:
        LT_mean/=LT_clicks
        LT_mean*=100

    if ST_mean>0:
        ST_mean/=ST_clicks
        ST_mean*=100
    #is it a LT?

    if LT_mean>thr3:
        if (LT_clicks/num_clicks)*100 >thr2:
            label.append({"species": "Long-tailed bat", "certainty": 100})
        else:
            label.append({"species": "Long-tailed bat", "certainty": 50})

    #is it a ST?

    if ST_mean>thr3:
        if (ST_clicks/num_clicks)*100 >thr2:
            label.append({"species": "Short-tailed bat", "certainty": 100})
        else:
            label.append({"species": "Short-tailed bat", "certainty": 50})
                

    if len(label)==0:
        file_label='Noise'
        cert_label=100
    elif len(label)==2:
        file_label='Both'
        cert_label=np.maximum(label[0]['certainty'],label[1]['certainty'])
    else:
        file_label=label[0]['species']
        cert_label=label[0]['certainty']


    #path to save
    image_path=dir_2_save+'/'+bmpfile[:-4]+"_click_prob.png"

    #plot clicks probabilities + save images
        #Each row has a differen plot: LT, ST NT
        # Lt first columin, st second, nt third
    fig, ax = plt.subplots()
    ax.plot(LT_prob, 'ro', label='LT' )
    ax.plot(ST_prob, 'ob', label='ST')
    ax.plot(NT_prob,'og', label='Not-bat')
       
    ax.set_title('File classified as '+file_label+' cert ='+str(cert_label)+', num. clicks = '+str(len(LT_prob)))
        
    legend = ax.legend(loc='upper right')
        
    plt.savefig(image_path)
 
    return label, LT_mean, ST_mean

def update_confusion_matrix(saving_dir, dir, file, assigned_label, correct_label, confusion_matrix, LT_check, ST_check, thr_check, click_num):
    """
    This function update the confusion matrix and return 
    """

    if correct_label==assigned_label:
        if correct_label=='Noise':
            confusion_matrix[6][5]+=1
        else:
            if correct_label=='Long-tailed bat':
                confusion_matrix[0][0]+=1
            elif correct_label=='Long-tailed bat?':
                confusion_matrix[1][1]+=1
            elif correct_label=='Short-tailed bat':
                confusion_matrix[2][2]+=1
            elif correct_label=='Short-tailed bat?':
                confusion_matrix[3][3]+=1
            elif correct_label=='Both':
                confusion_matrix[4][4]+=1
    else:

        if correct_label=='Noise':
            if LT_check>=thr_check or ST_check>=thr_check:
                problem='CNN'
            elif click_num<4:
                problem='CD'
            else:
                problem='Label'
            update_csv(saving_dir+'/false_positives.csv', dir, file, correct_label, assigned_label, problem)
            if assigned_label=='Long-tailed bat':
                confusion_matrix[0][5]+=1
            elif assigned_label=='Long-tailed bat?':
                confusion_matrix[1][5]+=1
            elif assigned_label=='Short-tailed bat':
                confusion_matrix[2][5]+=1
            elif assigned_label=='Short-tailed bat?':
                confusion_matrix[3][5]+=1
            elif assigned_label=='Both':
                confusion_matrix[4][5]+=1
            elif assigned_label=='Both?':
                confusion_matrix[5][5]+=1
        elif assigned_label=='Noise':
            if LT_check<thr_check or ST_check<thr_check:
                problem='CNN'
            elif click_num<4:
                problem='CD'
            else:
                problem='Label'
            update_csv(saving_dir+'/missed_files.csv', dir, file, correct_label, assigned_label, problem)
            if correct_label=='Long-tailed bat':
                confusion_matrix[6][0]+=1
            elif correct_label=='Long-tailed bat?':
                confusion_matrix[6][1]+=1
            elif correct_label=='Short-tailed bat':
                confusion_matrix[6][2]+=1
            elif correct_label=='Short-tailed bat?':
                confusion_matrix[6][3]+=1
            elif correct_label=='Both':
                confusion_matrix[6][4]+=1
        else:
            if correct_label=='Long-tailed bat':
                if assigned_label=='Long-tailed bat?':
                    confusion_matrix[1][0]+=1
                else:
                    if LT_check<thr_check or ST_check>=thr_check:
                        problem='CNN'
                    elif click_num<4:
                        problem='CD'
                    else:
                        problem='Label'
                    update_csv(saving_dir+'/misclassified_files.csv', dir, file, correct_label, assigned_label, problem)
                    if assigned_label=='Short-tailed bat':
                        confusion_matrix[2][0]+=1
                    elif assigned_label=='Short-tailed bat?':
                        confusion_matrix[3][0]+=1
                    elif assigned_label=='Both':
                        confusion_matrix[4][0]+=1
                    elif assigned_label=='Both?':
                        confusion_matrix[5][0]+=1
            elif correct_label=='Long-tailed bat?':
                if assigned_label=='Long-tailed bat':
                    confusion_matrix[0][1]+=1
                else:
                    if LT_check<thr_check or ST_check>=thr_check:
                        problem='CNN'
                    elif click_num<4:
                        problem='CD'
                    else:
                        problem='Label'
                    update_csv(saving_dir+'/misclassified_files.csv', dir, file, correct_label, assigned_label, problem)
                    if assigned_label=='Short-tailed bat':
                        confusion_matrix[2][1]+=1
                    elif assigned_label=='Short-tailed bat?':
                        confusion_matrix[3][1]+=1
                    elif assigned_label=='Both':
                        confusion_matrix[4][1]+=1
                    elif assigned_label=='Both?':
                        confusion_matrix[5][1]+=1
            elif correct_label=='Short-tailed bat':
                if assigned_label=='Short-tailed bat?':
                    confusion_matrix[3][2]+=1
                else:
                    if LT_check>=thr_check or ST_check<thr_check:
                        problem='CNN'
                    elif click_num<4:
                        problem='CD'
                    else:
                        problem='Label'
                    update_csv(saving_dir+'/misclassified_files.csv', dir, file, correct_label, assigned_label, problem)
                    if assigned_label=='Long-tailed bat':
                        confusion_matrix[0][2]+=1
                    elif assigned_label=='Long-tailed bat?':
                        confusion_matrix[1][2]+=1  
                    elif assigned_label=='Both':
                        confusion_matrix[4][2]+=1
                    elif assigned_label=='Both?':
                        confusion_matrix[5][2]+=1
            elif correct_label=='Short-tailed bat?':
                if assigned_label=='Short-tailed bat':
                    confusion_matrix[2][3]+=1
                else:
                    if LT_check>=thr_check or ST_check<thr_check:
                        problem='CNN'
                    elif click_num<4:
                        problem='CD'
                    else:
                        problem='Label'
                    update_csv(saving_dir+'/misclassified_files.csv', dir, file, correct_label, assigned_label, problem)
                    if assigned_label=='Long-tailed bat':
                        confusion_matrix[0][3]+=1
                    elif assigned_label=='Long-tailed bat?':
                        confusion_matrix[1][3]+=1             
                    elif assigned_label=='Both':
                        confusion_matrix[4][3]+=1
                    elif assigned_label=='Both?':
                        confusion_matrix[5][3]+=1
            elif correct_label=='Both':
                if assigned_label=='Both?':
                    confusion_matrix[5][4]+=1
                else:
                    if LT_check>=thr_check or ST_check>=thr_check:
                        problem='CNN'
                    elif click_num<4:
                        problem='CD'
                    else:
                        problem='Label'
                    update_csv(saving_dir+'/misclassified_files.csv', dir, file, correct_label, assigned_label, problem)
                    if assigned_label=='Long-tailed bat':
                        confusion_matrix[0][4]+=1
                    elif assigned_label=='Long-tailed bat?':
                        confusion_matrix[1][4]+=1
                    elif assigned_label=='Short-tailed bat':
                        confusion_matrix[2][4]+=1
                    elif assigned_label=='Short-tailed bat?':
                        confusion_matrix[3][4]+=1     
    return confusion_matrix


def update_csv(csvfilename, dir, file, correct_label, assigned_label, problem):
    """
    This function update the cvsfile with a new row storing dir, file, true and ans assigned label information
    """
    print('Updating file ', csvfilename)
    fieldnames=['Directory', 'Filename', 'True Label', 'Assigned Label', 'Possible Error']
    with open(csvfilename, 'a', newline='') as csvfile:
        writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Directory':dir, 'Filename': file, 'True Label':correct_label, 'Assigned Label': assigned_label, 'Possible Error': problem})

    return

def metrics(confusion_matrix, file_num):
    """
    Compute Recall, Precision, Accuracy pre and post possible classes check
    for each method
    
    INPUT:
        confusion_matrix is a matrix(3, 7) that stores the confusion matrix
                         
    OUTPUT:
        Recall -> number, recall 
                  TD/TD+FND this metric doesn't change before and after check
                  
        Precision_pre -> number, precision  before check
                         TD/TD+FD 
                         
        Precision_post -> number, precision  after check
                         TD/TD+FD 
                         
        Accuracy_pre -> number, accuracy before check
                         #correct classified/#files
                         for correct classifications we don't count possible 
                         classes
                         
       Accuracy_post -> number, accuracy after check
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



##################################################################################################################################################################################
##############################################################   MAIN    #########################################################################################################
##################################################################################################################################################################################

#inizialization

train_dir = "C:\\Users\\Virginia\\Documents\\Work\Data\\Bats\\New_Train_Datasets" #directory with train files
test_dataset_dir="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Moira 2020\\Raw files\\Bat_tests\\Test_dataset" #directory where to find test dataset files
test_count=1 #counter for test number
#test_dir = "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Results\\20201016_tests"

test_general_results_dir = "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Results\\New_CNN_train_bats" #directory to store test result
test_dataset_storage_dir="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Moira 2020\\Raw files\\Bat_tests\\Test_dataset" #local directory where to store annotations 

sp=SignalProc.SignalProc(1024,512)
thr1_0=10
thr2_0=70
thr1_1=60
thr2_1=85
thr1_2=20
thr2_2=50
thr3_2=85
f0=24000
f1=54000

print('Starting test ', test_count)
test_fold= "Test_New_"+str(test_count) #Test folder where to save all the stats
if test_fold not in os.listdir(test_general_results_dir):
    os.mkdir(test_general_results_dir+ '/' + test_fold)
    

for i in range(2):
    if test_count==1:
        i=0
    #identify train dataset
    train_fold='Train_'+str(i)

    #filelist train
    file_list_train=[]
    for sub_dir in os.listdir(train_dir+'/'+train_fold):
        if sub_dir.endswith('.txt'):
            continue
        for f in os.listdir(train_dir+'/'+train_fold+'/'+sub_dir):
            if f.endswith('.bmp'):
                file_list_train.append(sub_dir+'/'+f)

    file_number_train=len(file_list_train)
    for j in range(4):

        if test_count==1:
            j=0
        #inizializations of counters
        train_featuress =[]
        
        if j==0:
            window=3
        elif j==1:
            window=11
        elif j==2:
            window=17
        else:
            window=31

        #Create train dataset for CNN from the results of clicksearch   
        #search clicks
        for k in range(file_number_train):
            file = file_list_train[k]
            print('Analizing file ', file)
            filepath=train_dir+'/'+train_fold+'/'+file
            #read image
            print('Uploading file', file)
            sp.readBmp(filepath, rotate=False)
            #read annotaion file
            #Read annotation file: if in Train mode
            annotation_file=filepath+'.data'
            if os.path.isfile(annotation_file):
                segments = Segment.SegmentList()
                segments.parseJSON(annotation_file)
                #thisSpSegs = np.arange(len(segments)).tolist()
            else:
                    segments=[]
                    #thisSpSegs=[]
            #CLickSearch
            print('Click search on file ', file)
            click_label, train_featuress=ClickSearch(sp.sg, window, f0, f1, segments, train_featuress, Train=True)
            
            if click_label=='None':
                print('No click detected in ', file)
            else:
                #train_featuress.append(featuress)
                print('Clicks detected in ', file)

        #saving dataset
        #with open(test_dir+'/'+test_fold +'/sgramdata_train.json', 'w') as outfile:
        #    json.dump(train_featuress, outfile)

        #check
        print(np.shape(train_featuress))
        if window==3:
            first_dim=6
        else:
            first_dim=window

        #Train CNN
        # with and without agumentation
        
        #ag_flag=False
        data_train=train_featuress
        sg_train=np.ndarray(shape=(np.shape(data_train)[0],np.shape(data_train[0][0])[0], np.shape(data_train[0][0])[1]), dtype=float) #check
        target_train = np.zeros((np.shape(data_train)[0], 1)) #label train
        for k in range(np.shape(data_train)[0]):
            maxg = np.max(data_train[k][0][:])
            sg_train[k][:] = data_train[k][0][:]/maxg
            target_train[k][0] = data_train[k][-1]

        #validation data
        # randomly choose 75% train data and keep the rest as validation data
        idxs = np.random.permutation(np.shape(sg_train)[0])
        x_train = sg_train[idxs[0:int(len(idxs)*0.75)]]
        y_train = target_train[idxs[0:int(len(idxs)*0.75)]]
        x_validation = sg_train[idxs[int(len(idxs)*0.75):]]
        y_validation = target_train[idxs[int(len(idxs)*0.75):]]
 
        #Check: how many files I am using for training
        print("-------------------------------------------")
        print("Training model n ", test_count)
        print("Click search parameter: f0= ", f0, " , f1= ", f1)
        if window==3:
            print('window used 3x2')
        else:
            print('window used ', window)
        print('Number of spectrograms', np.shape(target_train)[0]) 
        print('Spectrograms used for training ',np.shape(y_train)[0])
        print("Spectrograms for LT: ", np.shape(np.nonzero(y_train==0))[1])
        print("Spectrograms for ST: ", np.shape(np.nonzero(y_train==1))[1])
        print("Spectrograms for Noise: ", np.shape(np.nonzero(y_train==2))[1])
        print('\n Spectrograms used for validation ',np.shape(y_validation)[0])
        print("Spectrograms for LT: ", np.shape(np.nonzero(y_validation==0))[1])
        print("Spectrograms for ST: ", np.shape(np.nonzero(y_validation==1))[1])
        print("Spectrograms for Noise: ", np.shape(np.nonzero(y_validation==2))[1])

        
        train_images = x_train.reshape(x_train.shape[0],first_dim, 512, 1) #changed image dimensions
        validation_images = x_validation.reshape(x_validation.shape[0],first_dim, 512, 1)
        input_shape = (first_dim, 512, 1)

        train_images = train_images.astype('float32')
        validation_images = validation_images.astype('float32')

        # CNN Training
        num_labels=3 #change this variable, when changing number of labels
        train_labels = tensorflow.keras.utils.to_categorical(y_train, num_labels)
        validation_labels = tensorflow.keras.utils.to_categorical(y_validation, num_labels)
        #test_labels = tensorflow.keras.utils.to_categorical(y_test, 8)   #change this to set labels  

        accuracies=np.zeros((10,1)) #initializing accuracies array
        model_paths=[] #initializing list where to stor model path
        #repeat training 10 times and take the best one on validation accuracy
        for k in range(10):
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
            print('Training n', k)
    
            #adding early stopping
            checkpoint = ModelCheckpoint(test_general_results_dir+ '/' + test_fold+"/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
            early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=1, mode='auto')
            history = model.fit(train_images, train_labels,
                            batch_size=32,
                            epochs=35,
                            verbose=2,
                            validation_data=(validation_images, validation_labels),
                            callbacks=[checkpoint, early],
                            shuffle=True)

            accuracies[k]=history.history['val_accuracy'][-1]
            print('Accuracy reached',accuracies[k])
            #save model
            modelpath=test_general_results_dir+ '/' + test_fold + '/model_'+str(k)+'.h5' #aid variable
            model.save(modelpath)
            model_paths.append(modelpath)
            #erasewigths
            for filelist in os.listdir(test_general_results_dir+ '/' + test_fold):
                if filelist.endswith('hdf5'):
                    erase_path=test_general_results_dir+ '/' + test_fold+ '/'+filelist
                    if erase_path!=modelpath:
                        os.remove(erase_path)

        #check what modelgave us better accuracy
        index_best_model=np.argmax(accuracies) 
        print('Training ended')
        print('Best CNN is ', index_best_model)
        print('Best accuracy reached ',accuracies[index_best_model])
        modelpath=model_paths[index_best_model] 

        #erase models
        for k in range(10):
            if k!=index_best_model:
                os.remove(model_paths[k])
   
        #recover model
        model=load_model(modelpath)

        #Saving Training info 
        #Save training info into a file
        cnn_train_info_file=test_general_results_dir+'/'+test_fold+'/CNN_train_data_info.txt'
        file1=open(cnn_train_info_file,'w')
        
        L00=["Training dataset used "+train_dir]
        L0=['Click search parameters: f0= %5d, f1= %5d \n' %(f0,f1) ]
        if window==3:
            L1=['window used 3x2 \n']
        else:
            L1=['window used %2d \n' %window]
        L=['Number of spectrograms =  %5d \n' %np.shape(target_train)[0], 
        "Spectrograms used for training = %5d \n" %np.shape(y_train)[0],
        "Spectrograms for LT: %5d \n" %np.shape(np.nonzero(y_train==0))[1],
        "Spectrograms for ST: %5d \n" %np.shape(np.nonzero(y_train==1))[1],
        "Spectrograms for Noise: %5d \n" %np.shape(np.nonzero(y_train==2))[1],
        "\n Spectrograms used for validation = %5d \n" %np.shape(y_validation)[0],
        "Spectrograms for LT: %5d \n" %np.shape(np.nonzero(y_validation==0))[1],
        "Spectrograms for ST: %5d \n" %np.shape(np.nonzero(y_validation==1))[1],
        "Spectrograms for Noise: %5d \n" %np.shape(np.nonzero(y_validation==2))[1]]
        L2=['\n Model used %5d \n' %index_best_model]
        L3=['Training accuracy for the model %3.7f \n' %accuracies[index_best_model]]
        L4=['Path best model '+ model_paths[index_best_model]]
        file1.writelines(np.concatenate((L00,L0,L1,L, L2, L3, L4)))
        file1.close()

        ## 3 test with differen Label strategies
        for label_index in range(3):
            if test_count==1:
                label_index=1
            ########## TEST TRAINED MODEL ###########################
         
            print('\n ----------------------------------------------- \n')
            print('Starting test ', test_count)
            dir_list=os.listdir(test_dataset_dir)

            #inizialization global metrics
            confusion_matrix_tot=np.zeros((7,6))
            file_number_tot=0

            for Moira_dir in dir_list:
                dir_path=test_dataset_dir+'/'+Moira_dir+'/Bat'
                print('Analising directory ', dir_path)
                dir_path_local=test_dataset_storage_dir+'/'+Moira_dir+'/Bat'
                test_sub_dir=test_dataset_storage_dir+'/'+Moira_dir+'/'+test_fold
    
                if test_fold not in os.listdir(test_dataset_storage_dir+'/'+ Moira_dir):
                    os.mkdir(test_sub_dir)

                #inizialization folder metrics
                file_number=0 #count number of files in folder
                #metrics for click detector evaluation
                TD_cd=0
                FD_cd=0
                TND_cd=0
                FND_cd=0
                count_clicks=0 #this variable counts files in folder where a click is detected

                #initializing header
                fieldnames=['Directory', 'Filename', 'True Label', 'Assigned Label', 'Possible Error']

                #creating .csv files for FN
                with open(test_sub_dir+'/missed_files.csv', 'w', newline='') as csvfile:
                     writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                     writer.writeheader()

                #creating .csv files for FP
                with open(test_sub_dir+'/false_positives.csv', 'w', newline='') as csvfile:
                     writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                     writer.writeheader()

                #creating .csv files for Misclassified files
                with open(test_sub_dir+'/misclassified_files.csv', 'w', newline='') as csvfile:
                     writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                     writer.writeheader()


                # inizializing folder confusion_matrix
                confusion_matrix=np.zeros((7,6))

                for root, dirs, files in os.walk(dir_path):
                    #for dir in dirs:
                    #    print('Analising dir ', dir)
                    for dir in dirs:
                        #creating dir where to save created annotations
                        if dir[0]==str(2):
                            if test_fold not in os.listdir(dir_path_local+'/'+dir):
                                os.mkdir(dir_path_local+'/'+dir+'/'+test_fold)
                            #removing .png file
                            for filelist in os.listdir(dir_path_local+'/'+dir+'/'+test_fold):
                                if filelist.endswith('png'):
                                    erase_path=dir_path_local+'/'+dir+'/'+test_fold+ '/'+filelist
                                    print('Removing previously stored .png file ', erase_path)
                                    os.remove(erase_path)

                                elif filelist.endswith('data'):
                                    erase_path=dir_path_local+'/'+dir+'/'+test_fold+ '/'+filelist
                                    print('Removing previously stored data file ', erase_path)
                                    os.remove(erase_path)
                        
                    for file in files:

                        if file.endswith('.bmp'):
                
                            bat_dir=root
                            bat_file=file
                            data_test=[]
                            print('Analising file ', file, ' in dir ', bat_dir)
                            filepath=bat_dir+'/'+file
                            bat_dir_local_storage=dir_path_local+'/'+bat_dir[-8:]+'/'+test_fold #local folder where to store generated annotations and png
                        
                            annotation_path=bat_dir_local_storage+'/'+file+'.data'
                            #read file
                            try:
                                #read image
                                sp.readBmp(filepath, rotate=False)
                            except OSError:
                                print('Error loading file ', file) 
                                print('File classification = Corrupted file')
                                continue
                            except:
                                print('Error loading file ', file)
                                print('File classification = Corrupted file')
                                continue
                            file_number+=1
                
                            #read GT annotation

                            GT_path=bat_dir+'/GT'
                            GT_annotations=os.listdir(GT_path)

                            if file+'.data' in GT_annotations:
                                GT_annotation_file=GT_path+'/'+file+'.data'
                                print('Annotation file found')
                                GT_segments = Segment.SegmentList()
                                GT_segments.parseJSON(GT_annotation_file)
                                print('GT annotations ', GT_segments)
                            else:
                                print('Annotation file not found')
                                GT_segments=[] 
                    

                            #recover GT label
                            if len(GT_segments)==0:
                                GT_label='Noise'
                            elif len(GT_segments)==2:
                                GT_Label='Both'
                                if np.minimum(GT_segments[0][4][0]["certainty"],GT_segments[0][4][1]["certainty"])==50:
                                    GT_label+='?'
                            else:
                                if GT_segments[0][4][0]['certainty']==50:
                                    GT_label=GT_segments[0][4][0]["species"]+'?'
                                else:
                                    GT_label=GT_segments[0][4][0]["species"]
                
                            #inizialization
                            segments=[]
                            #CLickSearch
                            print('Click Search')
                            click_label, data_test=ClickSearch(sp.sg, window, f0, f1, segments, data_test, Train=False)
                            #initialize annotations
                            dt=sp.incr/sp.sampleRate  # self.sp.incr is set to 512 for bats dt=temporal increment in samples
                            duration=dt*np.shape(sp.sg)[1] #duration=dt*num_columns
                            generated_annotation=[{"Operator": "Auto", "Reviewer": "", "Duration": duration, "noiseLevel": [], "noiseTypes": []}]
                            if click_label=='Click':
                                count_clicks+=1
                                if len(GT_segments)==0:
                                    FD_cd+=1
                                else:
                                    TD_cd+=1 
                                print('Click Detected')
                                #we enter in the cnn only if we got a click
                                print('check data_test shape ', np.shape(data_test))
                                sg_test=np.ndarray(shape=(np.shape(data_test)[0],np.shape(data_test[0][0])[0], np.shape(data_test[0][0])[1]), dtype=float)
                                print('Number of file spectrograms', np.shape(data_test)[0])
                                for k in range(np.shape(data_test)[0]):
                                    maxg = np.max(data_test[k][0][:])
                                    sg_test[k][:] = data_test[k][0][:]/maxg
            
                                #CNN classification of clicks
                                x_test = sg_test
                                test_images = x_test.reshape(x_test.shape[0],first_dim, 512, 1)
                                input_shape = (first_dim, 512, 1)
                                test_images = test_images.astype('float32')
                                #recovering labels
                                predictions =model.predict(test_images)
                                num_clicks=np.shape(predictions)[0]
                                #predictions is an array #imagesX #of classes which entries are the probabilities
                                #for each classes
                                if label_index==0:
                                    label, LT_check, ST_check=File_label_0(predictions, thr1_0, thr2_0, bat_dir_local_storage, file)
                                    thr_check=thr2_0
                                elif label_index==1:
                                    label, LT_check, ST_check=File_label_1(predictions, thr1_1, thr2_1, bat_dir_local_storage, file)
                                    thr_check=thr1_1
                                elif label_index==2:
                                    label, LT_check, ST_check=File_label_2(predictions, thr1_2, thr2_2, thr3_2, bat_dir_local_storage, file)
                                    thr_check=thr3_2
                                print('CNN detected: ', label)
                                if len(label)>0:
                                    #update annotations
                                    generated_annotation.append([0, duration, 0, 0, label])                               
                            else:
                                num_clicks=0
                                LT_check=0
                                ST_check=0
                                thr_check=0
                                # do not create any segments
                                print("Nothing detected")
                                #count_clicks+=1
                                if len(GT_segments)==0:
                                    TND_cd+=1
                                else:
                                    FND_cd+=1 

                            #save segments in datafile
                            print('Storing annotation in ', annotation_path)
                            f = open(annotation_path, 'w')
                            json.dump(generated_annotation, f)
                            f.close()

                            #determine assigned label
                            #print(generated_annotation)
                            #print(len(generated_annotation))
                            if len(generated_annotation)==1:
                                assigned_label='Noise'
                            elif len(generated_annotation[1][4])==2:
                                assigned_label='Both'
                                if np.minimum(generated_annotation[1][4][0]['certainty'],generated_annotation[1][4][1]['certainty'])==50:
                                    assigned_label+='?'
                            else:
                                if generated_annotation[1][4][0]['certainty']==50:
                                    assigned_label=generated_annotation[1][4][0]["species"]+'?'
                                else:
                                    assigned_label=generated_annotation[1][4][0]["species"]
                
                            confusion_matrix= update_confusion_matrix(test_sub_dir, bat_dir[-8:], file, assigned_label, GT_label , confusion_matrix, LT_check, ST_check, thr_check, num_clicks)

                ##updating total file_number    
                file_number_tot+=file_number
                ##updating total confusion_matrix
                confusion_matrix_tot+=confusion_matrix 
                #printing metrics
                print("-------------------------------------------")
                print('Number files with detected clicks detected ', count_clicks, 'in ', Moira_dir, 'with ', file_number, ' files')
                print("Click Detector stats on Testing Data")
                if TD_cd==0:
                    Recall_cd=0
                else:
                    Recall_cd= TD_cd/(TD_cd+FND_cd)*100
                print('Recall ', Recall_cd)
                if TD_cd==0:
                    Precision_cd=0
                else:
                    Precision_cd= TD_cd/(TD_cd+FD_cd)*100
                print('Precision ', Precision_cd)
                if TD_cd==0 and TND_cd==0:
                    Accuracy_cd=0
                else:
                    Accuracy_cd = (TD_cd+TND_cd)/(TD_cd+TND_cd+FD_cd+FND_cd)*100
                print('Accuracy', Accuracy_cd)
                TD_cd_rate= (TD_cd/file_number)*100
                print('True Detected rate', TD_cd_rate)
                FD_cd_rate= (FD_cd/file_number)*100
                print('False Detected rate', FD_cd_rate)
                FND_cd_rate= (FND_cd/file_number)*100
                print('False Negative Detected rate', FND_cd_rate)
                TND_cd_rate= (TND_cd/file_number)*100
                print('True Negative Detected rate', TND_cd_rate)
                print("-------------------------------------------")
    
                #saving Click Detector Stats
                cd_metrics_file=test_sub_dir+'/click_detector_stats.txt'
                file1=open(cd_metrics_file,"w")
                L0=["Number of file %5d \n"  %file_number]
                L1=["Number of files with detected clicks %5d \n" %count_clicks ]
                L3=["Recall = %3.7f \n" %Recall_cd,"Precision = %3.7f \n" %Precision_cd, "Accuracy = %3.7f \n" %Accuracy_cd, "True Detected rate = %3.7f \n" %TD_cd_rate, "False Detected rate = %3.7f \n" %FD_cd_rate, "True Negative Detected rate = %3.7f \n" %TND_cd_rate, "False Negative Detected rate = %3.7f \n" %FND_cd_rate ]
                file1.writelines(np.concatenate((L0,L1,L3)))
                file1.close()

        
                # Evaluating metrics for this folder
                print('Evaluating BatSearch performance on ', Moira_dir)
                Recall, Precision_pre, Precision_post, Accuracy_pre, Accuracy_post, TD, FPD_pre, FPD_post, FND, CoCla_pre, CoCla_post=metrics(confusion_matrix, file_number)
    
                #print metrics
                print("-------------------------------------------")
                print('Classification performance on ', Moira_dir)
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

                #storing confusion matrix into a csv file
                fieldnames=['  ', 'LT','LT?', 'ST','ST?', 'BT', 'Noise' ]
                column_fieldnames=['LT','LT?', 'ST','ST?', 'BT','BT?', 'Noise' ]
                with open(test_sub_dir+'/confusion_matrix.csv', 'w', newline='') as csvfile:
                     writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                     writer.writeheader()
                     for i in range(7):
                        writer.writerow({'  ':column_fieldnames[i], 'LT':confusion_matrix[i][0], 'LT?':confusion_matrix[i][1], 'ST':confusion_matrix[i][2], 'ST?':confusion_matrix[i][3], 'BT':confusion_matrix[i][4], 'Noise':confusion_matrix[i][5]})
    
     

                #saving Click Detector Stats
                cd_metrics_file=test_sub_dir+'/bat_detector_stats.txt'
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

            #storing confusion matrix into a csv file
            fieldnames=['  ', 'LT','LT?', 'ST','ST?', 'BT', 'Noise' ]
            column_fieldnames=['LT','LT?', 'ST','ST?', 'BT','BT?', 'Noise' ]
            with open(test_general_results_dir+ '/' + test_fold+'/confusion_matrix.csv', 'w', newline='') as csvfile:
                    writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in range(7):
                        writer.writerow({'  ':column_fieldnames[row], 'LT':confusion_matrix_tot[row][0], 'LT?':confusion_matrix_tot[row][1], 'ST':confusion_matrix_tot[row][2], 'ST?':confusion_matrix_tot[row][3], 'BT':confusion_matrix_tot[row][4], 'Noise':confusion_matrix_tot[row][5]})
    
            #saving Click Detector Stats
            cd_metrics_file=test_general_results_dir+ '/' + test_fold+'/bat_detector_stats.txt'
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
            if test_count<72:
                test_count+=1
                test_fold= "Test_New_"+str(test_count) #Test folder where to save all the stats
                if test_fold not in os.listdir(test_general_results_dir):
                    os.mkdir(test_general_results_dir+ '/' + test_fold)
                print('\n\n Starting test ', test_count)
        
                

    







    

