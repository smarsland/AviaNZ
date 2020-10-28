""""
29/09/2020

Author: Virginia Listanti

This script:
- finds Click using ClickSearch built for bats
- saves clicks as segments into .data file 

I am using some of the functions build by Julius from my code to be quicker.

"""

import SignalProc
import os
import math
import numpy as np
import json
import Segment




def ClickSearch(imspec, label):
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
    
    count = 0

    if label=="Don't know":
        certanty=0
    else:
        certanty=50

    df=sp.sampleRate//2 /(np.shape(imspec)[0]+1)  # frequency increment
    dt=sp.incr/sp.sampleRate  # self.sp.incr is set to 512 for bats dt=temporal increment in samples
    duration=dt*np.shape(imspec)[1] #duration=dt*num_columns

    #inizialize annotations
    detected_clicks=[{"Operator": "Auto", "Reviewer": "", "Duration": duration, "noiseLevel": [], "noiseTypes": []}]

    #check this
    up_len=math.ceil(0.075/dt) #max length in columns
    print('check up_len= ', up_len)
    #up_len=17
    # up_len=math.ceil((0.5/11)/dt)

    # Frequency band
    f0=24000 #21000
    index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
    f1=54000 #54000
    index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up

    # Mean in the frequency band
    mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0)

    # Threshold
    mean_spec_all=np.mean(imspec, axis=0)[2:]
    #thr_spec=(np.mean(mean_spec_all)+np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))
    thr_spec=np.mean(mean_spec_all)
    #mean_spec_all=np.mean(imspec, axis=0)
    #thr_spec=mean_spec_all-0.5*np.std(mean_spec_all)

    ## clickfinder
    # check when the mean is bigger than the threshold
    # clicks is an array which elements are equal to 1 only where the sum is bigger
    # than the mean, otherwise are equal to 0
    clicks = mean_spec>thr_spec
    clicks_indices = np.nonzero(clicks)
    # check: if I have found somenthing
    if np.shape(clicks_indices)[1]==0:
        #count='None'
        return detected_clicks, count
        # not saving spectrograms

    # Discarding segments too long or too short and saving spectrogram images
    click_start=clicks_indices[0][0]
    click_end=clicks_indices[0][0]
    for i in range(1,np.shape(clicks_indices)[1]):
        if clicks_indices[0][i]==click_end+1:
            click_end=clicks_indices[0][i]
        else:
            if click_end-click_start+1>up_len:
                clicks[click_start:click_end+1] = False
            else:
                # update annotations
                count+=1
                detected_clicks.append([float(click_start*dt), float((click_end+1)*dt), float(f0), float(f1), [{"species": label, "certainty": certanty, "filter": "ClickSearch", "calltype": "Click"}]])
                
            # update
            click_start=clicks_indices[0][i]
            click_end=clicks_indices[0][i]

    # checking last loop with end
    if click_end-click_start+1>up_len:
        clicks[click_start:click_end+1] = False
    else:
        count+=1
        detected_clicks.append([float(click_start*dt), float((click_end+1)*dt), float(f0), float(f1), [{"species": label, "certainty": certanty, "filter": "ClickSearch", "calltype": "Click"}]])
        
    return detected_clicks, count

def ClickSearch2(imspec, label):
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
    
    count = 0

    if label=="Don't know":
        certanty=0
    else:
        certanty=50

    df=sp.sampleRate//2 /(np.shape(imspec)[0]+1)  # frequency increment
    dt=sp.incr/sp.sampleRate  # self.sp.incr is set to 512 for bats dt=temporal increment in samples
    duration=dt*np.shape(imspec)[1] #duration=dt*num_columns

    #inizialize annotations
    detected_clicks=[{"Operator": "Auto", "Reviewer": "", "Duration": duration, "noiseLevel": [], "noiseTypes": []}]

    #check this
    up_len=math.ceil(0.075/dt) #max length in columns
    #print('check up_len= ', up_len)
    #up_len=17
    # up_len=math.ceil((0.5/11)/dt)

    # Threshold
    #mean_spec_all=np.mean(imspec, axis=0)[2:]
    #thr_spec=(np.mean(mean_spec_all)+np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))
    #thr_spec=(np.mean(mean_spec_all))*np.ones((np.shape(mean_spec_all)))
    #thr_spec=np.mean(mean_spec_all)
    mean_spec_all=np.mean(imspec, axis=0)
    thr_spec=mean_spec_all+np.std(mean_spec_all)

    # 3 Frequency bands we want to check
    #[21k,36k]
    f0=21000 #21000
    index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
    f1=36000 #54000
    index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up

    # Mean in the frequency band
    mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0)
    #check thr in freq band
    clicks_0 = np.where(mean_spec>thr_spec, True, False)

    #[36k,50k]
    f0=36000 #21000
    index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
    f1=50000 #54000
    index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up

    # Mean in the frequency band
    mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0)
    #check thr in freq band
    clicks_1= np.where(mean_spec>thr_spec, True, False)

    #[50k,60k]
    f0=50000 #21000
    index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
    f1=60000 #54000
    index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up

    # Mean in the frequency band
    mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0)
    #check thr in freq band
    clicks_2= np.where(mean_spec>thr_spec, True, False)
    

    ## clickfinder
    # check when the mean is bigger than the threshold
    # clicks is an array which elements are equal to True where the thr is overomed in one of the freq bands
    clicks=clicks_0+clicks_1+clicks_2
    clicks_indices = np.nonzero(clicks)
    # check: if I have found somenthing
    if np.shape(clicks_indices)[1]==0:
        #count='None'
        return detected_clicks, count
        # not saving spectrograms

    # Discarding segments too long or too short and saving spectrogram images
    click_start=clicks_indices[0][0]
    click_end=clicks_indices[0][0]
    for i in range(1,np.shape(clicks_indices)[1]):
        if clicks_indices[0][i]==click_end+1:
            click_end=clicks_indices[0][i]
        else:
            if click_end-click_start+1>up_len:
                clicks[click_start:click_end+1] = False
            else:
                # update annotations
                count+=1
                detected_clicks.append([float(click_start*dt), float((click_end+1)*dt), float(21000), float(60000), [{"species": label, "certainty": certanty, "filter": "ClickSearch", "calltype": "Click"}]])
                
            # update
            click_start=clicks_indices[0][i]
            click_end=clicks_indices[0][i]

    # checking last loop with end
    if click_end-click_start+1>up_len:
        clicks[click_start:click_end+1] = False
    else:
        count+=1
        detected_clicks.append([float(click_start*dt), float((click_end+1)*dt), float(21000), float(60000), [{"species": label, "certainty": certanty, "filter": "ClickSearch", "calltype": "Click"}]])
        
    return detected_clicks, count


def ClickSearch3(imspec, label):
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
    
    count = 0

    if label=="Don't know":
        certanty=0
    else:
        certanty=50

    df=sp.sampleRate//2 /(np.shape(imspec)[0]+1)  # frequency increment
    dt=sp.incr/sp.sampleRate  # self.sp.incr is set to 512 for bats dt=temporal increment in samples
    duration=dt*np.shape(imspec)[1] #duration=dt*num_columns

    #inizialize annotations
    detected_clicks=[{"Operator": "Auto", "Reviewer": "", "Duration": duration, "noiseLevel": [], "noiseTypes": []}]

    #check this
    up_len=math.ceil(0.01/dt) #max length in columns
    #print('check up_len= ', up_len)
    #up_len=17
    # up_len=math.ceil((0.5/11)/dt)

    # Threshold
    #thr_spec=(np.mean(mean_spec_all)+np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))
    #thr_spec=(np.mean(mean_spec_all))*np.ones((np.shape(mean_spec_all)))
    #mean_spec_all=np.mean(imspec, axis=0)[2:]
    #thr_spec=np.mean(mean_spec_all)*0.7
    mean_spec_all=np.mean(imspec, axis=0)
    thr_spec=mean_spec_all*0.75

    # 2 Frequency bands we want to check
    #[21k,36k]+[50k,60k]
    f0=21000 #21000
    index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
    f1=36000 #54000
    index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up
    f0_1=50000 #21000
    index_f0_1=-1+math.floor(f0_1/df)  # lower bound needs to be rounded down
    f1_1=60000 #54000
    index_f1_1=-1+math.ceil(f1_1/df)  # upper bound needs to be rounded up

    # Mean in the frequency band
    mean_spec=np.mean(np.concatenate((imspec[index_f0:index_f1,:],imspec[index_f0_1:index_f1_1,:])), axis=0)
    #check thr in freq band
    clicks_0 = np.where(mean_spec>thr_spec, True, False)

    #[36k,50k]
    f0=36000 #21000
    index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
    f1=50000 #54000
    index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up

    # Mean in the frequency 
    mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0)
    #check thr in freq band
    clicks_1= np.where(mean_spec>thr_spec, True, False)
    
    ## clickfinder
    # check when the mean is bigger than the threshold
    # clicks is an array which elements are equal to True where the thr is overomed in one of the freq bands
    clicks=clicks_0+clicks_1
    clicks_indices = np.nonzero(clicks)
    # check: if I have found somenthing
    if np.shape(clicks_indices)[1]==0:
        #count='None'
        return detected_clicks, count
        # not saving spectrograms

    # Discarding segments too long or too short and saving spectrogram images
    click_start=clicks_indices[0][0]
    click_end=clicks_indices[0][0]
    for i in range(1,np.shape(clicks_indices)[1]):
        if clicks_indices[0][i]==click_end+1:
            click_end=clicks_indices[0][i]
        else:
            if click_end-click_start+1>up_len:
                clicks[click_start:click_end+1] = False
            else:
                # update annotations
                count+=1
                detected_clicks.append([float(click_start*dt), float((click_end+1)*dt), float(21000), float(60000), [{"species": label, "certainty": certanty, "filter": "ClickSearch", "calltype": "Click"}]])
                
            # update
            click_start=clicks_indices[0][i]
            click_end=clicks_indices[0][i]

    # checking last loop with end
    if click_end-click_start+1>up_len:
        clicks[click_start:click_end+1] = False
    else:
        count+=1
        detected_clicks.append([float(click_start*dt), float((click_end+1)*dt), float(21000), float(60000), [{"species": label, "certainty": certanty, "filter": "ClickSearch", "calltype": "Click"}]])
        
    return detected_clicks, count

def ClickSearch4(imspec, label):
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
    
    count = 0

    if label=="Don't know":
        certanty=0
    else:
        certanty=50

    df=sp.sampleRate//2 /(np.shape(imspec)[0]+1)  # frequency increment
    dt=sp.incr/sp.sampleRate  # self.sp.incr is set to 512 for bats dt=temporal increment in samples
    duration=dt*np.shape(imspec)[1] #duration=dt*num_columns

    #inizialize annotations
    detected_clicks=[{"Operator": "Auto", "Reviewer": "", "Duration": duration, "noiseLevel": [], "noiseTypes": []}]

    #check this
    up_len=math.ceil(0.01/dt) #max length in columns
    #print('check up_len= ', up_len)
    #up_len=17
    # up_len=math.ceil((0.5/11)/dt)

    # Threshold
    #thr_spec=(np.mean(mean_spec_all)+np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))
    #thr_spec=(np.mean(mean_spec_all))*np.ones((np.shape(mean_spec_all)))
    mean_spec_all=np.mean(np.log(imspec), axis=0)[2:]
    thr_spec=np.mean(mean_spec_all)-np.std(mean_spec_all)
    #mean_spec_all=np.mean(imspec, axis=0)
    #thr_spec=mean_spec_all*0.75

    # Frequency band
    f0=24000 #21000
    index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
    f1=54000 #54000
    index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up

    # Mean in the frequency band
    mean_spec=np.mean(np.log(imspec[index_f0:index_f1,:]), axis=0)

    ## clickfinder
    # check when the mean is bigger than the threshold
    # clicks is an array which elements are equal to 1 only where the sum is bigger
    # than the mean, otherwise are equal to 0
    clicks = mean_spec>thr_spec

    ## 2 Frequency bands we want to check
    ##[21k,36k]+[50k,60k]
    #f0=21000 #21000
    #index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
    #f1=36000 #54000
    #index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up
    #f0_1=50000 #21000
    #index_f0_1=-1+math.floor(f0_1/df)  # lower bound needs to be rounded down
    #f1_1=60000 #54000
    #index_f1_1=-1+math.ceil(f1_1/df)  # upper bound needs to be rounded up

    ## Mean in the frequency band
    #mean_spec=np.mean(np.concatenate((imspec[index_f0:index_f1,:],imspec[index_f0_1:index_f1_1,:])), axis=0)
    ##check thr in freq band
    #clicks_0 = np.where(mean_spec>thr_spec, True, False)

    ##[36k,50k]
    #f0=36000 #21000
    #index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
    #f1=50000 #54000
    #index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up

    ## Mean in the frequency 
    #mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0)
    ##check thr in freq band
    #clicks_1= np.where(mean_spec>thr_spec, True, False)
    
    ### clickfinder
    ## check when the mean is bigger than the threshold
    ## clicks is an array which elements are equal to True where the thr is overomed in one of the freq bands
    #clicks=clicks_0+clicks_1
    clicks_indices = np.nonzero(clicks)
    # check: if I have found somenthing
    if np.shape(clicks_indices)[1]==0:
        #count='None'
        return detected_clicks, count
        # not saving spectrograms

    # Discarding segments too long or too short and saving spectrogram images
    click_start=clicks_indices[0][0]
    click_end=clicks_indices[0][0]
    for i in range(1,np.shape(clicks_indices)[1]):
        if clicks_indices[0][i]==click_end+1:
            click_end=clicks_indices[0][i]
        else:
            if click_end-click_start+1>up_len:
                clicks[click_start:click_end+1] = False
            else:
                # update annotations
                count+=1
                detected_clicks.append([float(click_start*dt), float((click_end+1)*dt), float(21000), float(60000), [{"species": label, "certainty": certanty, "filter": "ClickSearch", "calltype": "Click"}]])
                
            # update
            click_start=clicks_indices[0][i]
            click_end=clicks_indices[0][i]

    # checking last loop with end
    if click_end-click_start+1>up_len:
        clicks[click_start:click_end+1] = False
    else:
        count+=1
        detected_clicks.append([float(click_start*dt), float((click_end+1)*dt), float(21000), float(60000), [{"species": label, "certainty": certanty, "filter": "ClickSearch", "calltype": "Click"}]])
        
    return detected_clicks, count

                                ################### MAIN ############################


"""
Work flow on Bat_tests:
- entry root directory 
- navigate tree
- ClickSearch
- save .data file
- evaluate Recall and Precision
"""

#root_path='C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\BattyBats'
root_path="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Moira 2020\\Raw files\\Bat_tests"
test_num=0
Test_fold="Genral_Test_"+str(test_num)
#data_fold=root_path+"\\Data"
storing_directory="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Moira 2020\\Raw files\\Bat_tests\\ClickSearch_GeneralTest"

if Test_fold not in os.listdir(storing_directory):
    os.mkdir(storing_directory+"\\"+Test_fold)

sp=SignalProc.SignalProc(1024,512)
TD=0
FD=0
TND=0
FND=0
file_num=0

list_dir=os.listdir(root_path)

for dir in list_dir:
    if dir.startswith('R'):
        if Test_fold not in os.listdir(root_path+'\\'+dir):
            os.mkdir(root_path+'\\'+dir+"\\"+Test_fold)

        for root, dirs, files in os.walk(root_path+'\\'+dir):

            for file in files:
                if not file.endswith('.bmp'):
                    print(file, "is not a .bmp file")
                    continue
                #count=0 #counts number of clicks detected
                print('Analizing file ', file)
                filepath=root+'\\'+file

                #read GT annotation

                GT_path=root+'\\GT'
                GT_annotations=os.listdir(GT_path)

                if file+'.data' in GT_annotations:
                    GT_annotation_file=GT_path+'\\'+file+'.data'
                    print('Annotation file found')
                    GT_segments = Segment.SegmentList()
                    GT_segments.parseJSON(GT_annotation_file)
                    print('GT annotations ', GT_segments)
                else:
                    print('Annotation file not found')
                    GT_segments=[] 

                if len(GT_segments)==0:
                    label="Don't know"
                else:
                    label=GT_segments[0][4][0]["species"]
            

                #read image
                sp.readBmp(filepath, rotate=False)
                print('Spectrogram dimensions:', np.shape(sp.sg))
                file_num+=1

                #CLickSearch
                detected_clicks, count=ClickSearch(sp.sg,label)
                print(count, '  clicks detected')

                #updating count
                if label=="Don't know":
                    if count==0:
                        TND+=1
                    else:
                        FD+=1
                else:
                    if count==0:
                        FND+=1
                    else:
                        TD+=1

                #save segments
                #save segments in datafile
                f = open(root_path+'\\'+dir+"\\"+Test_fold+"\\"+file +'.data', 'w')
                json.dump(detected_clicks, f)
                f.close()

#evaluating metrics
Recall=TD/(TD+FND)*100
Precision=TD/(TD+FD)*100
TD_rate=(TD/file_num)*100
FD_rate=(FD/file_num)*100
TND_rate=(TND/file_num)*100
FND_rate=(FND/file_num)*100

print("-------------------------------------")
print("Metrics:")
print("Recall = ", Recall)
print("Precision = ", Precision)
print("TD rate = ", TD_rate)
print("FD rate = ", FD_rate)
print("TND rate = ", TND_rate)
print("FND rate = ", FND_rate)
print("TD = ", TD)
print("FD = ", FD)
print("TND = ", TND)
print("FND = ", FND)

#save Stats
#saving Click Detector Stats
cd_metrics_file=storing_directory+"\\"+Test_fold+'\\click_detector_stats.txt'
file1=open(cd_metrics_file,"w")
L0=["Number of file %5d \n"  %file_num]
L1=["Number of true detected files %5d \n" %TD, "Number of false detected files %5d \n" %FD, "Number of true negative detected files %5d \n" %TND, "Number of false negative detected files %5d \n" %FND]
L3=["Recall = %3.7f \n" %Recall,"Precision = %3.7f \n" %Precision, "True Detected rate = %3.7f \n" %TD_rate, "False Detected rate = %3.7f \n" %FD_rate, "True Negative Detected rate = %3.7f \n" %TND_rate, "False Negative Detected rate = %3.7f \n" %FND_rate ]
file1.writelines(np.concatenate((L0,L3)))
file1.close()