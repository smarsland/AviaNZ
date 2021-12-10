"""
This script:
- create a list of LT, ST and Noise file from a selected datasete
- randomly select 100 files for each class
- runs click search
- runs CNN on clicks
- for each file stores: path, filename, mean(LT), mean(ST), true label

NOTE: only if #clicks >=4

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


def random_select(filelist,file_number, output_dir ):
    """
    This function shuffle indeces of filelist, randomly select the first file_number ones
    and copy selected files in output_dir
    """

    #shuffle indexes
    x = np.arange(len(filelist)-1)
    np.random.shuffle(x)


    count=0 
    i=0
    #copy to output directory
    while count<file_number:
    #print('x[i] = ', x[i])
    #print('file= ', files[x[i]])
        file_path=filelist[x[i]-1][0]+'\\'+filelist[x[i]-1][1]
        annotation_file=filelist[x[i]-1][0]+'\\GT\\'+filelist[x[i]-1][1]+'.data'
        copy(file_path,output_dir)
        if os.path.isfile(annotation_file):
            copy(annotation_file, output_dir)
        count+=1
        i+=1

    return 
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
                featuress=updateDataset(imspec, segments, click_start, click_end, featuress, Train)
                
            # update
            click_start=clicks_indices[0][i]
            click_end=clicks_indices[0][i]

    # checking last loop with end
    if click_end-click_start+1>up_len:
        clicks[click_start:click_end+1] = 0
    else:
        count+=1
        #savedataset
        featuress=updateDataset(imspec, segments, click_start, click_end, featuress, Train)

    #Assigning: click label
    if count==0:
        label='None'
    else:
        label='Click'

    print(count, ' clicks found')
        
    return label, featuress 


def updateDataset(spectrogram, segments, click_start, click_end, featuress, Train=False):
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



######################## MAIN #############################################

Test_dataset="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Moira 2020\\Raw files\\Bat_tests"
modelpath="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Results\\20201016_tests\\Test_1\\model_6.h5"
model=load_model(modelpath)
sp=SignalProc.SignalProc(1024,512)
window=3
if window==3:
    first_dim=6
else:
    first_dim=window

f0=24000
f1=54000
## Create LT, ST, and Noise file list
#inizialization
LT_files=[]
ST_files=[]
Noise_files=[]

#tree search
for root, dirs, files in os.walk(Test_dataset):
    dirs.sort()
    files.sort()
    for filename in files:
        if filename.endswith('.bmp'):
            annotation_file=root+'\\GT\\'+filename+'.data'
            print('Annotation path', annotation_file)
            if os.path.isfile(annotation_file):
                print('Annotation file found')
                GT_segments = Segment.SegmentList()
                GT_segments.parseJSON(annotation_file)
                print('GT annotations ', GT_segments)
            else:
                print('Annotation file not found')
                GT_segments=[] 
            #recover GT label
            if len(GT_segments)==0:
                GT_label='Noise'
            elif len(GT_segments)==2:
                GT_label='Both'
                if np.minimum(GT_segments[0][4][0]["certainty"],GT_segments[0][4][1]["certainty"])==50:
                    GT_label+='?'
            else:
                if GT_segments[0][4][0]['certainty']==50:
                    GT_label=GT_segments[0][4][0]["species"]+'?'
                else:
                    GT_label=GT_segments[0][4][0]["species"]
            if GT_label=='Long-tailed bat':
                LT_files.append([root, filename])
                print('[', root,' , ', filename,'] added to LT_files')
            elif GT_label=='Short-tailed bat':
                ST_files.append([root, filename])
                print('[', root,' , ', filename,'] added to ST_files')
            elif GT_label=='Noise':
                Noise_files.append([root, filename])
                print('[', root,' , ', filename,'] added to Noise_files')

#print
print('Number of detected LT files = ', len(LT_files))
print('Number of detected ST files = ', len(ST_files))
print('Number of detected Noise files = ', len(Noise_files))


validation_fold= Test_dataset+"\\Validation_data" #Validation
if "Validation_data" not in os.listdir(Test_dataset):
    os.mkdir(validation_fold)

#pick randomly 
val_num=75
random_select(LT_files,val_num, validation_fold )
random_select(ST_files,val_num, validation_fold )
random_select(Noise_files,val_num, validation_fold )

#initializing csv file
#initializing header
fieldnames=['Filename', 'True Label', 'Click Number', 'Class LT mean', 'Class LT best 4 mean', 'Class ST mean', 'Class ST best 4 mean']

csv_filename=validation_fold+"\\prob_table.csv"
#creating .csv files for FN
with open(csv_filename, 'w', newline='') as csvfile:
        writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


 #CNN 
for root, dirs, files in os.walk(validation_fold):
    for file in files:

        if file.endswith('.bmp'):
                
            bat_dir=root
            #bat_file=file
            data_test=[]
            print('Analising file ', file, ' in dir ', bat_dir)
            
            filepath=bat_dir+'\\'+file    
            print(filepath)

            #read file
            #try:
                #read image
            sp.readBmp(filepath, rotate=False)
            #except OSError:
            #    print('Error loading file ', file) 
            #    print('File classification = Corrupted file')
            #    continue
            #except:
            #    print('Error loading file ', file)
            #    print('File classification = Corrupted file')
            #    continue
                
            #read GT annotation

            GT_path=bat_dir+'\\'+file+'.data'
            print(GT_path)

            if os.path.isfile(GT_path):
                print('Annotation file found')
                GT_segments = Segment.SegmentList()
                GT_segments.parseJSON(GT_path)
                print('GT annotations ', GT_segments)
            else:
                print('Annotation file not found')
                GT_segments=[] 
                    

            #recover GT label
            if len(GT_segments)==0:
                GT_label='Noise'
            elif len(GT_segments)==2:
                GT_label='Both'
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
            #generated_annotation=[{"Operator": "Auto", "Reviewer": "", "Duration": duration, "noiseLevel": [], "noiseTypes": []}]
            if click_label=='Click':
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
                #predictions is an array #imagesX #of classes which entries are the probabilities
                click_number=np.shape(predictions)[0]
                LT_prob=[]  # class 0
                ST_prob=[]
                for k in range(click_number):
                    #spec_num+=1
                    LT_prob.append(predictions[k][0])
                    ST_prob.append(predictions[k][1])

                # mean
                LT_mean=np.mean(LT_prob)*100
                ST_mean=np.mean(ST_prob)*100

                # best3mean
                LT_best4mean=0
                ST_best4mean=0

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
                       
            else:
                # do not create any segments
                print("Nothing detected")
                click_number=0
                LT_mean=0
                ST_mean=0
                LT_best4mean=0
                ST_best4mean=0

            #update csv
            with open(csv_filename, 'a', newline='') as csvfile:
                writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'Filename': file, 'True Label':GT_label, 'Click Number': click_number,'Class LT mean':LT_mean, 'Class LT best 4 mean': LT_best4mean, 'Class ST mean':ST_mean, 'Class ST best 4 mean': ST_best4mean })