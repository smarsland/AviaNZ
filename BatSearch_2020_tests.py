"""
Created on 15/10/2020

@author: Virginia Listanti

This code test different BatSearch options on Moira folders:
-R1 (with rodents, annotated by me)
-R13 (annotated by me)
-R18 (annotated by Moira)

Then generates a .txt with the following stats:
-percentage of detected files with click detector
-stats after CNN: Recall, Precision, Accuracy

For FP and FN plot probabilities.



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

def File_label(predictions, thr1, thr2, dir_2_save, bmpfile):
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
        image_path=dir_2_save+'\\'+bmpfile[:-4]+"_click_prob.png"

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
 

    return label


def update_confusion_matrix(file, assigned_label, correct_label, confusion_matrix, missed_files, false_positives, misclassified_files):
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
            false_positives.append([file, correct_label, assigned_label])
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
            missed_files.append([file, correct_label, assigned_label])
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
                if assigned_label=='?':
                    confusion_matrix[1][0]+=1
                elif assigned_label=='Short-tailed bat':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[2][0]+=1
                elif assigned_label=='Short-tailed bat?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[3][0]+=1
                elif assigned_label=='Both':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[4][0]+=1
                elif assigned_label=='Both?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[5][0]+=1
            elif correct_label=='Long-tailed bat?':
                if assigned_label=='Long-tailed bat':
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
            elif correct_label=='Short-tailed bat':
                if assigned_label=='Long-tailed bat':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[0][2]+=1
                elif assigned_label=='Long-tailed bat?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[1][2]+=1
                elif assigned_label=='Short-tailed bat?':
                    confusion_matrix[3][2]+=1
                elif assigned_label=='Both':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[4][2]+=1
                elif assigned_label=='Both?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[5][2]+=1
            elif correct_label=='Short-tailed bat?':
                if assigned_label=='Long-tailed bat':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[0][3]+=1
                elif assigned_label=='Long-tailed bat?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[1][3]+=1
                elif assigned_label=='Short-tailed bat':
                    confusion_matrix[2][3]+=1
                elif assigned_label=='Both':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[4][3]+=1
                elif assigned_label=='Both?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[5][3]+=1
            elif correct_label=='Both':
                if assigned_label=='Long-tailed bat':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[0][4]+=1
                elif assigned_label=='Long-tailed bat?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[1][4]+=1
                elif assigned_label=='Short-tailed bat':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[2][4]+=1
                elif assigned_label=='Short-tailed bat?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[3][4]+=1
                elif assigned_label=='Both?':
                    misclassified_files.append([file, correct_label, assigned_label])
                    confusion_matrix[5][4]+=1
                
   
        
    return confusion_matrix, missed_files, false_positives, misclassified_files


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

##MAIN
 

#train_dir = "/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Battybats/Train_Datasets" #directory with train files
CNN_test_dir = "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\Moira 2020\\Raw files\\Bat_tests" #directory with test files
#CNN_test_dir="/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Battybats/Test_dataset"
test_count=0 #counter for test number
test_dir = "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Results\\20201016_tests"
#test_dir = "/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Battybats/Results/20201016_tests" #directory to store test result
test_fold= "Test_"+str(test_count) #Test folder where to save all the stats
if test_fold not in os.listdir(test_dir):
    os.mkdir(test_dir+ '/' + test_fold)

print('Starting test ', test_count)

sp=SignalProc.SignalProc(1024,512)
thr1=10
thr2=70
window=3

if window==3:
    first_dim=6
else:
    first_dim=window

#recover model
modelpath="C:\\Users\\Virginia\\AppData\\Roaming\\AviaNZ\\Filters\\NZ Bats.h5"
#modelpath="/am/state-opera/home1/listanvirg/sourcecode/AviaNZ/Filters/NZ Bats.h5"
model=load_model(modelpath)

 ########## TEST TRAINED MODEL ###########################
print('\n ----------------------------------------------- \n')
print('Starting test')
dir_list=os.listdir(CNN_test_dir)

#inizialization global metrics
confusion_matrix_tot=np.zeros((7,6))
file_number_tot=0

for Moira_dir in dir_list:
    dir_path=CNN_test_dir+'/'+Moira_dir+'/Bat'
    test_sub_dir=CNN_test_dir+'/'+Moira_dir+'/'+test_fold
    
    if test_fold not in os.listdir(CNN_test_dir+'/'+ Moira_dir):
        
        os.mkdir(test_sub_dir)

    #inizialization folder metrics
    file_number=0 #count number of files in folder
    #metrics for click detector evaluation
    TD_cd=0
    FD_cd=0
    TND_cd=0
    FND_cd=0
    count_clicks=0 #this variable counts files in folder where a click is detected
    
    #inizializing annotations
    missed_files=[]
    false_positives=[]
    misclassified_files=[]
    # inizializing folder confusion_matrix
    confusion_matrix=np.zeros((7,6))

    for root, dirs, files in os.walk(dir_path):
        #for dir in dirs:
        #    print('Analising dir ', dir)
        for dir in dirs:
            #creating dir where to save created annotations
            if dir[0]==str(2):
                if test_fold not in os.listdir(root+'/'+dir):
                    os.mkdir(root+'/'+dir+'/'+test_fold)
        for file in files:

            if file.endswith('.bmp'):
                
                bat_dir=root
                bat_file=file
                data_test=[]
                print('Analising file ', file, ' in dir ', bat_dir)
                filepath=bat_dir+'/'+file
                annotation_path=bat_dir+'/'+test_fold+'/'+file+'.data'
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
                click_label, data_test=ClickSearch(sp.sg, window, segments, data_test, Train=False)
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
                    #predictions is an array #imagesX #of classes which entries are the probabilities
                    #for each classes
                    label=File_label(predictions, thr1, thr2, bat_dir+'/'+test_fold, file)
                    print('CNN detected: ', label)
                    if len(label)>0:
                        #update annotations
                        generated_annotation.append([0, duration, 0, 0, label])                               
                else:
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
                
                confusion_matrix, missed_files, false_positives, misclassified_files= update_confusion_matrix(dir+'/'+file, assigned_label, GT_label, confusion_matrix, missed_files, false_positives, misclassified_files)
        
    #updating total file_number    
    file_number_tot+=file_number
    #updating total confusion_matrix
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

    #saving missed_fils
    with open(test_sub_dir+'/missed_files.data', 'w') as f:
        json.dump(missed_files,f)

    #saving false positives
    with open(test_sub_dir+'/false_positives.data', 'w') as f:
        json.dump(false_positives,f)        
    
    #saving misclassified files
    with open(test_sub_dir+'/misclassified_files.data', 'w') as f:
        json.dump(misclassified_files,f)
        
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
    
    with open(test_sub_dir+'/Confusion_matrix.txt','w') as f:
        f.write("Confusion Matrix \n\n")
        np.savetxt(f, confusion_matrix, fmt='%d')
        
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

with open(test_dir+'/' +test_fold+"/Confusion_matrix.txt",'w') as f:
    f.write("Confusion Matrix \n\n")
    np.savetxt(f, confusion_matrix_tot, fmt='%d')
    
#saving Click Detector Stats
cd_metrics_file=test_dir+'/'+test_fold+'/bat_detector_stats.txt'
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