"""
This script read DOC csv file for bats and generate .data files
"""

import csv
#import Segment
import SignalProc
import json
import numpy as np


dirName="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Moira 2020\\Raw files\\Bat_tests\\R18\\Bat"
sp=SignalProc.SignalProc(1024,512)
with open(dirName+"\\BatSearch.csv", 'r') as csvfile:
    csvreader=csv.DictReader(csvfile)
    for row in csvreader:
        annotation_dir=dict(row)['Foldername'][2:]
        annotation_file=dict(row)['Filename'][2:]
        file_path=dirName+'\\'+annotation_dir+'\\'+annotation_file
        try:
        #read image
            sp.readBmp(file_path, rotate=False)
        except OSError:
            print('Error loading file ', file_path) 
            print('File classification = Corrupted file')
            continue
        except:
            print('Error loading file ', file_path)
            print('File classification = Corrupted file')
            continue
        dt=sp.incr/sp.sampleRate  # self.sp.incr is set to 512 for bats dt=temporal increment in samples
        duration=dt*np.shape(sp.sg)[1] #duration=dt*num_columns
        annotation=[{"Operator": "Auto", "Reviewer": "", "Duration": duration, "noiseLevel": [], "noiseTypes": []}]

        #recover label
        label=[]
        if dict(row)['Category']=='Long tail':
            label.append({"species": "Long-tailed bat", "certainty": 100})
        elif dict(row)['Category']=='Short tail':
            label.append({"species": "Short-tailed bat", "certainty": 100})
        elif dict(row)['Category']=='Both':
            label.append({"species": "Long-tailed bat", "certainty": 100})
            label.append({"species": "Short-tailed bat", "certainty": 100})
        elif dict(row)['Category']=='Possible LT':
            label.append({"species": "Long-tailed bat", "certainty": 50})
        elif dict(row)['Category']=='Possible ST':
            label.append({"species": "Short-tailed bat", "certainty": 50})

        if len(label)>0:
            #update annotations
            annotation.append([0, duration, 0, 0, label]) 

        #save segments in datafile
        annotation_path=file_path+'.data'
        print('Storing annotation in ', annotation_path)
        f = open(annotation_path, 'w')
        json.dump(annotation, f)
        f.close()
