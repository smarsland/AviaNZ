""""
29/07/2020

Author: Virginia Listanti

This script:
- finds click compatible with Bigeye pops 
- evaluates performance on file annotated by Zohara

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

import librosa
import WaveletSegment
import WaveletFunctions

def ClickSearch(dirName, file, resol, window=None):
    """
    ClickSearch search for clicks into file in directory dirName, 
    
    The search is made on the spectrogram image generated from the file

    The search is done on a moving window of size window (default=1sec)

    Click presence is assested for each spectrogram column: if the mean in the
    frequency band [] (*) is bigger than a treshold we have a click
    thr=mean(all_spec)+std(all_spec) (*)
    
    The clicks are discarded if longer than 0.05 sec
    
    Detection are stored into the array detections which is discretized using resol
    which are the output
    
    """
    # Default: window=1 sec
    if window==None:
        window=1

    print("Click search on ",file)
    filename = dirName + '\\' + file
    
    
#    #Read audiodata
    sp = SignalProc.SignalProc(1024, 512) #outside?
    sp.readWav(filename)
    audiodata = sp.data
    sampleRate = sp.sampleRate
    datalength = np.shape(audiodata)[0]
    datalengthSec = datalength / sampleRate

    #detection vector inizialization
    det_length=np.ceil(datalengthSec/res)
    print('det_length = ', det_length)
    detections=np.zeros((1,int(det_length)))

    #work on 1 sec windows
    for i in range(int(np.ceil(datalengthSec))):
        #start and end in samples
        start=i*sampleRate
        #index in resol scale
        start_resol=int(math.floor(i/resol))
        if i>=datalengthSec:
            end=datalength
            #index in resol scale
            end_resol=int(math.ceil(datalengthSec/resol))
        else:
            end=(i+1)*sampleRate
            end_resol=int(math.ceil((i+1)/resol))
        
        #generate spectrogram
        sp.data=audiodata[start:end]
        sgraw= sp.spectrogram(256, 128, 'Blackman')
        imspec=(10.*np.log10(sgraw)).T #transpose
        imspec=np.flipud(imspec) #updown 


        # spectrogram parameter
        df=sampleRate/(np.shape(imspec)[0]+1) #frequency increment 
        dt= 1/(np.shape(imspec)[1]+1) #timeincrement
        # print("file ", file, "dt " , dt)
        up_len=math.ceil(0.05/dt) #0.05 second lenth in indices divided by 11
    
        #Frequency band
        f0=300
        index_f0=-1+math.floor(f0/df) #lower bound needs to be rounded down
    #    print(f0,index_f0)
    #    f1=55000
        f1=600
        index_f1=-1+math.ceil(f1/df) #upper bound needs to be rounded up
        
        #Mean in the frequency band
        mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0) #added 0.01 to avoid divition by 0

        #Threshold
        mean_spec_all=np.mean(imspec, axis=0)
        thr_spec=(np.mean(mean_spec_all)+np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))
    
        ##clickfinder
        #check when the mean is bigger than the threshold
        #clicks is an array which elements are equal to 1 only where the sum is bigger 
        #than the mean, otherwise are equal to 0
        clicks=np.where(mean_spec>thr_spec,1,0)
        clicks_indices=np.nonzero(clicks)
        # print(np.shape(clicks_indices))
        #check: if I have found somenthing
        if np.shape(clicks_indices)[1]==0:
            continue
    
        #DIscarding segments too long or too shorts and saving spectrogram images       
        click_start=clicks_indices[0][0]
        click_end=clicks_indices[0][0]  
        for j in range(1,np.shape(clicks_indices)[1]):
            if clicks_indices[0][j]==click_end+1:
                click_end=clicks_indices[0][j]
            else:
                if click_end-click_start+1>up_len:
                    clicks[click_start:click_end+1]=0
                else:
                    #update detections
                    click_start_res=start_resol+int(math.floor((click_start*dt)/resol))
                    click_end_res=i+int(math.ceil((click_end*dt)/resol))
                    detections[0][click_start_res:click_end_res]=1
                    
                #update
                click_start=clicks_indices[0][j]
                click_end=clicks_indices[0][j] 
                                
        #checking last loop with end
        if click_end-click_start+1>up_len:
            clicks[click_start:click_end+1]=0
        else:
            click_start_res=i+int(math.floor((click_start*dt)/resol))
            click_end_res=np.minimum(i+int(math.ceil((click_end*dt)/resol)),i+int(np.ceil(datalengthSec)))
            detections[0][click_start_res:click_end_res]=1
            
    
    return detections


def comparison(det, ann):
    """
    Comparison between detections and annotations to evaluate

    TP True Positives
    TN True Negatives
    FP False Positives
    FN False Negatives
    """

    #inizialization 
    TP=0
    TN=0
    FP=0
    FN=0

    print('shape det', np.shape(det))
    print('len det', len(det))

    for i in range(np.shape(det)[1]):
        if det[0][i]==0:
            if ann[0][i]==0:
                TN+=1
            else:
                FN+=1
        else:
            if ann[0][i]==0:
                FP+=1
            else:
                TP+=1

    print('len(det) =',len(det))
    print('TP =', TP)
    print('TN =', TN)
    print('FP =', FP)
    print('FN =', FN)
    print('TP+TN+FP+FN =', TP+TN+FP+FN)

    return TP, TN, FP, FN


def metrics(TP, TN, FP, FN):
    """
    This function evaluates Recall Precision and Accuracy
    """

    if TP==0:
        Recall=0
    else:
        Recall= TP/(TP+FN)

    return Recall, Precision, Accuracy




################### MAIN ############################
"""
Work flow:
- entry directory + file
- read annotation
- ClickSearch
- Evaluate and save metrics
"""

dirname='D:\\Desktop\\Documents\\Work\\Zohara files\\TEST'
filename = '67375127.140303193211_downsampled8000.wav'
window = 1 #length in sec. of window used for click search
res = 0.05 #annotation resolution

#    #Read audiodata
#just for duration: think something more intelligent
sp = SignalProc.SignalProc(1024, 512) #outside?
sp.readWav(dirname+'\\'+filename)
audiodata=sp.data
sampleRate = sp.sampleRate
datalength = np.shape(audiodata)[0]
datalengthSec = datalength / sampleRate


#read annotation file: from WaveletSegment
annotation_file=dirname+'\\'+filename[:-4]+'-res'+str(float(res))+'sec.txt'
fileAnnotations = []
annotation=[]
# Get the segmentation from the txt file
with open(annotation_file) as f:
    reader = csv.reader(f, delimiter="\t")
    d = list(reader)

print(d[300])
if d[-1] == []:
    d = d[:-1]
# if len(d) != datalengthSec/res:
#     print("ERROR: annotation length %d does not match file duration %d!" % (len(d), n))
#     self.annotation = []
#     return False

# for each second, store 0/1 presence:
presblocks = 0
for row in d:
    fileAnnotations.append(int(row[1]))
    presblocks += int(row[1])

annotation.append(np.array(fileAnnotations))

totalblocks = sum([len(a) for a in annotation])
print("File length %f, annotations with %f resolution \n" % (datalengthSec,res))
print(" %d blocks read, %d presence blocks found.\n" % ( totalblocks, presblocks))

# call ClickSearch
detections=ClickSearch(dirname, filename, res, window)

#Check
if np.shape(detections)[1]!=np.shape(annotation)[1]:
    print("Detections and annotations length do not match")
    print("Detections size=", np.shape(detections))
    print("Annotation size=", np.shape(annotation))


# Detection and annotation comparison
TP, TN, FP, FN = comparison(detections, annotation)

# metrics
# Recall, Precision, Accuracy = metrics(TP, TN, FP, FN)

# print metrics

