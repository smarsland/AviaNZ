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
    up_len=math.ceil(0.05/dt) #max length in columns
    print('check up_len= ', up_len)
    #up_len=17
    # up_len=math.ceil((0.5/11)/dt)

    # Frequency band
    f0=24000
    index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
    f1=54000
    index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up

    # Mean in the frequency band
    mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0)

    # Threshold
    mean_spec_all=np.mean(imspec, axis=0)[2:]
    #thr_spec=(np.mean(mean_spec_all)+np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))
    thr_spec=(np.mean(mean_spec_all))*np.ones((np.shape(mean_spec)))

    ## clickfinder
    # check when the mean is bigger than the threshold
    # clicks is an array which elements are equal to 1 only where the sum is bigger
    # than the mean, otherwise are equal to 0
    clicks = mean_spec>thr_spec
    clicks_indices = np.nonzero(clicks)
    # check: if I have found somenthing
    if np.shape(clicks_indices)[1]==0:
        count='None'
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


################### MAIN ############################
"""
Work flow:
- entry root directory 
- navigate tree
- ClickSearch
- save .data file
"""

root_path='C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\BattyBats'

#list of directory
dirs=os.listdir(root_path)

sp=SignalProc.SignalProc(1024,512)


for dirname in dirs:
    #pre assessing label
    print('Analizing folder ', dirname)
    if dirname=='LT':
        label='Long-tailed bat'
    elif dirname=='ST':
        label='Short-tailed bat'
    else:
        label="Don't Know"

    for _, _, files in os.walk(os.path.join(root_path, dirname)):

        for file in files:


            if not file.endswith('.bmp'):
                print(file, 'is not a .bmp file')
                continue
            #count=0 #counts number of clicks detected
            print('Analizing file ', file)
            filepath=root_path+'\\'+dirname+'\\'+file

            #read image
            sp.readBmp(filepath, rotate=False)
            print('Spectrogram dimensions:', np.shape(sp.sg))

            #CLickSearch
            detected_clicks, count=ClickSearch(sp.sg,label)
            print(count, '  clicks detected')

            #save segments
            #save segments in datafile
            f = open(filepath + '.data', 'w')
            json.dump(detected_clicks, f)
            f.close()
        
