"""
Created on 1/10/2020

Updated 19/10/2020

@author: Virginia Listanti

This code generate different trained CNN and test them on the folder Moira2020

Different test configuration:
- 100/250/500 files for each class each [3 diffent datasets
- window: 3pxl doubled, 7 pxl, 31 pxl
- augumentation y/n


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


def ClickSearch(imspec, window, segments, featuress, Train=False):
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
    f0=24000
    index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
    f1=54000
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

def File_label(predictions, thr1, thr2):
    """
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

    if click_detected_flag:
        # mean
        LT_mean=np.mean(LT_prob)*100
        ST_mean=np.mean(ST_prob)*100

        # best3mean
        LT_best3mean=0
        ST_best3mean=0

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

    return label

def exportCSV(dirName, Dataset_dir, writefile):
    #from PyQt5.QtCore import QTime

    # list all DATA files that can be processed
    #writefile = "Results.csv"
    f = open(os.path.join(dirName,writefile),'w')
    f.write('Date,Time,AssignedSite,Category,Foldername,Filename,Observer\n')
    for root, dirs, files in os.walk(Dataset_dir):
        dirs.sort()
        files.sort()
        for filename in files:
            if filename.endswith('.data'):
                print("Appending" ,filename)
                segments = Segment.SegmentList()
                segments.parseJSON(os.path.join(root, filename))
                if len(segments)>0:
                    seg = segments[0]
                    c = [lab["certainty"] for lab in seg[4]]
                    if c[0]==100:
                        s = [lab["species"] for lab in seg[4]]
                        # TODO: what if both?
                        if s[0] == 'Long-tailed bat':
                            s = 'Long tail,'
                        elif s[0] == 'Short-tailed bat':
                            s = 'Short tail,'
                    else:
                        s = ''
                else:
                    s = ''
                # Assumes DOC format
                d = filename[6:8]+'/'+filename[4:6]+'/'+filename[:4]+','
                if d[0] == '0':
                    d = d[1:]
                if int(filename[9:11]) < 13:
                    if filename[9:11] == '00':
                        t = str(int(filename[9:11])+12)+':'+filename[11:13]+':'+filename[13:15]+' a.m.,'
                    else:
                        t = filename[9:11]+':'+filename[11:13]+':'+filename[13:15]+' a.m.,'
                else:
                    t = str(int(filename[9:11])-12)+':'+filename[11:13]+':'+filename[13:15]+' p.m.,'
                if t[0] == '0':
                    t = t[1:]
                # Assume that directory structure is recorder - date
                if s == '':
                    rec = ',Unassigned'
                    op = ''
                else:
                    rec = root.split('/')[-3]
                    op = 'Moira Pryde'
                date = '.\\'+root.split('/')[-1]
                #dd/mm/yyyy,(h)h:mm:ss a.m.,R?,Long tail,.\20191110,.\file.bmp,Moira Pryde
                f.write(d+t+rec+','+s+date+','+'.\\'+filename[:-5]+','+op+'\n')

                #delete .datafile
                print ("Deleting ", filename)
                os.remove(os.path.join(root, filename))

    f.close()


##MAIN
 

train_dir = "/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Battybats/Train_Datasets" #directory with train files
CNN_test_dir = "/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Virginia_From_Moira_2020/Raw_files" #directory with test files
test_count=8 #counter for test number
test_dir = "/am/state-opera/home1/listanvirg/Documents/Bat_TESTS"
#test_dir = "/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Battybats/Experiments_Results" #directory to store test result
test_fold= "Test_"+str(test_count) #Test folder where to save all the stats
os.mkdir(test_dir+ '/' + test_fold)

print('Starting test ', test_count)

sp=SignalProc.SignalProc(1024,512)
thr1=10
thr2=70

#Create train dataset for CNN from the results of clicksearch   

for i in range(2,3):
    if i==0:
        train_fold='TRAIN100'
    elif i==1:
        train_fold='TRAIN250'
    else:
        train_fold='TRAIN500'
    #filelist
    file_list_train=[]
    for sub_dir in os.listdir(train_dir+'/'+train_fold):
        for f in os.listdir(train_dir+'/'+train_fold+'/'+sub_dir):
            if f.endswith('.bmp'):
                file_list_train.append(sub_dir+'/'+f)

    file_number_train=len(file_list_train)

    for j in range(2,3):

        if test_count==1:
            j=1
        #inizializations of counters
        train_featuress =[]
        
        if j==0:
            window=3
        elif j==1:
            window=11
        else:
            window=31

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
            click_label, train_featuress=ClickSearch(sp.sg, window, segments, train_featuress, Train=True)
            
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
        
        #for ag in range(2):
        #    #if test_count==1:
        #    #    ag=1
        #    data_train=train_featuress
        #    if ag%2==1:
        #        #### data agumentation ######################
        #        # create image data augmentation generator for in-build
        #        ag_flag=True
        #        #sg_train_0=[]
        #        #sg_train_1=[]
        #        #g_train_2=[]
        #        sg_train_0=np.ndarray(shape=(np.shape(data_train)[0], first_dim, 512), dtype=float)
        #        sg_train_1=np.ndarray(shape=(np.shape(data_train)[0], first_dim, 512), dtype=float)
        #        sg_train_2=np.ndarray(shape=(np.shape(data_train)[0], first_dim, 512), dtype=float)
        #        print('check shape',np.shape(data_train)[0])
        #        for k in range(np.shape(data_train)[0]):
        #            maxg = np.max(data_train[k][0][:])
        #            if data_train[k][-1]==0:
        #                sg_train_0[k][:]=data_train[k][0][:]/maxg
        #            elif data_train[k][-1]==1:
        #                sg_train_1[k][:]=data_train[k][0][:]/maxg
        #            elif data_train[k][-1]==2:
        #                sg_train_2[k][:]=data_train[k][0][:]/maxg
        #        datagen1 = ImageDataGenerator(width_shift_range=0.5, fill_mode='nearest')
        #        batch_size = 32

        #        # class_0
        #        print('check shape',np.shape(sg_train_0)[0])
        #        samples = expand_dims(sg_train_0, np.shape(sg_train_0)[0])
        #        # prepare iterator
        #        it1 = datagen1.flow(samples, batch_size=batch_size)
        #        # generate samples
        #        batch_train_0 = it1.next()
        #        for k in range(int(np.round((1500-np.shape(sg_train_0)[0])/batch_size))):
        #            batch = it1.next()
        #            batch_train_0 = np.vstack((batch_train_0, batch))
    
        #        print('sg_train_0 shape', np.shape(sg_train_0))
    
        #        print('batch_train_0 shape', np.shape(batch_train_0))
        #        #add to original ones
        #        batch_train_0=np.reshape(batch_train_0,np.shape(batch_train_0)[0:3])
        #        print('batch_train_0 shape after reshape', np.shape(batch_train_0))
        #        sg_train_0_ag=np.concatenate((sg_train_0,batch_train_0), axis=0)
        #        #label
        #        target_train_0=np.zeros(np.shape(sg_train_0_ag)[0])

        #        # class_1
        #        samples = expand_dims(sg_train_1, np.shape(sg_train_1)[0])
        #        # prepare iterator
        #        it1 = datagen1.flow(samples, batch_size=batch_size)
        #        # generate samples
        #        batch_train_1 = it1.next()
        #        for k in range(int(np.round((1500-np.shape(sg_train_1)[0])/batch_size))):
        #            batch = it1.next()
        #            batch_train_1 = np.vstack((batch_train_1, batch))

        #        print('sg_train_1 shape', np.shape(sg_train_1))
    
        #        print('batch_train_1 shape', np.shape(batch_train_1))
        #        #add to original ones
        #        batch_train_1=np.reshape(batch_train_1,np.shape(batch_train_1)[0:3])
        #        print('batch_train_1 shape after reshape', np.shape(batch_train_1))
        #        #add to original ones
        #        sg_train_1_ag=np.concatenate((sg_train_1,batch_train_1), axis=0)
        #        #label
        #        target_train_1=np.ones(np.shape(sg_train_1_ag)[0])
        #        # class_2
        #        samples = expand_dims(sg_train_2, np.shape(sg_train_2)[0])
        #        # prepare iterator
        #        it1 = datagen1.flow(samples, batch_size=batch_size)
        #        # generate samples
        #        batch_train_2 = it1.next()
        #        for k in range(int(np.round((9000-np.shape(sg_train_2)[0])/batch_size))):
        #            batch = it1.next()
        #            batch_train_2 = np.vstack((batch_train_2, batch))
    
        #        print('sg_train_2 shape', np.shape(sg_train_2))
    
        #        print('batch_train_2 shape', np.shape(batch_train_2))
        #        #add to original ones
        #        batch_train_2=np.reshape(batch_train_2,np.shape(batch_train_2)[0:3])
        #        print('batch_train_2 shape after reshape', np.shape(batch_train_2))
        #        #add to original ones
        #        sg_train_2_ag=np.concatenate((sg_train_2,batch_train_2), axis=0)
        #        #label
        #        target_train_2=2*np.ones(np.shape(sg_train_2_ag)[0])

        #        #unify train dataset
        #        sg_train=np.concatenate((sg_train_0_ag, sg_train_1_ag, sg_train_2_ag), axis=0)
        #        sg_train=np.asarray(sg_train, dtype=float)
        #        print('check sg_train shape', np.shape(sg_train))
        #        #unify labels
        #        target_train=np.concatenate((target_train_0, target_train_1, target_train_2), axis=0)
        #        print('check target_train shape', np.shape(target_train))
        #    else:
        ag_flag=False
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
        if ag_flag:
            print('Agumentation used')
        else:
            print('Agumentation NOT used')
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
            checkpoint = ModelCheckpoint(test_dir+ '/' + test_fold+"/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
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
            modelpath=test_dir+ '/' + test_fold + '/model_'+str(k)+'.h5' #aid variable
            model.save(modelpath)
            model_paths.append(modelpath)
            #erasewigths
            for filelist in os.listdir(test_dir+ '/' + test_fold):
                if filelist.endswith('hdf5'):
                    erase_path=test_dir+ '/' + test_fold+ '/'+filelist
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
        cnn_train_info_file=test_dir+'/'+test_fold+'/CNN_train_data_info.txt'
        file1=open(cnn_train_info_file,'w')
        if ag_flag:
            L0=['Agumentation used \n']
        else:
            L0=['Agumentation NOT used\n']
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
        file1.writelines(np.concatenate((L0,L1,L, L2, L3, L4)))
        file1.close()

        ########## TEST TRAINED MODEL ###########################
        print('\n ----------------------------------------------- \n')
        print('Starting test')
        for root, dirs, files in os.walk(CNN_test_dir):
            #for dir in dirs:
            #    print('Analising dir ', dir)
            for file in files:
                if file.endswith('.bmp'):
                    bat_dir=root
                    bat_file=file
                    data_test=[]
                    print('Analising file ', file, ' in dir ', bat_dir)
                    filepath=bat_dir+'/'+file
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
                    
                
                    #inizialization
                    segments=[]
                    #CLickSearch
                    print('Click Search')
                    click_label, data_test=ClickSearch(sp.sg, window, segments, data_test, Train=False)
                    #initialize annotations
                    dt=sp.incr/sp.sampleRate  # self.sp.incr is set to 512 for bats dt=temporal increment in samples
                    duration=dt*np.shape(sp.sg)[1] #duration=dt*num_columns
                    generated_annotation=[{"Operator": "Auto", "Reviewer": "", "Duration": duration, "noiseLevel": [], "noiseTypes": []}]
                    if click_label=='Click':
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
                        #predictions is an array #imagesX #of classes which entries are the probabilities
                        #for each classes
                        label=File_label(predictions, thr1, thr2)
                        print('CNN detected: ', label)
                        if len(label)>0:
                            #update annotations
                            generated_annotation.append([0, duration, 0, 0, label])
                                     
                    else:
                        # do not create any segments
                        print("Nothing detected")
                    #save segments in datafile
                    annotation_file=filepath + '.data'
                    print('Storing annotation in ', annotation_file)
                    f = open(annotation_file, 'w')
                    json.dump(generated_annotation, f)
                    f.close()
                    print('test: are we storing annotations?')
                    segments = Segment.SegmentList()
                    segments.parseJSON(annotation_file)
                    print(segments)
            
        #savecsvs
        print('\n Saving cvs files with results')
        for k in range(1,22):
            MoiraDir = CNN_test_dir+'/R'+str(k)
            print('Generating .cvs for ', MoiraDir)
            exportCSV(test_dir+ '/' + test_fold, MoiraDir, 'R'+str(k)+'_Results.csv')
            #update here
        if test_count<17:
            test_count+=1
            test_fold= "Test_"+str(test_count) #Test folder where to save all the stats
            os.mkdir(test_dir+ '/' + test_fold)
            print('\n\n Starting test ', test_count)
                

    







    

