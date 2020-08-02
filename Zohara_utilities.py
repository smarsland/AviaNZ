"""
Created on 24/07/2020

Author: Virginia Listanti

Script with utilities for Zohara files
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

import librosa
import WaveletSegment
import WaveletFunctions

# #Read audiodata
# #fs=16000
# sp = SignalProc.SignalProc(1024, 512) #outside?
# audiofile="D:\\Desktop\\Documents\\Work\\Zohara files\\67375127.140303193211.wav"
# sp.readWav(audiofile)
# audiodata = sp.data
# sampleRate = sp.sampleRate
# datalength = np.shape(audiodata)[0]
# datalengthSec = datalength / sampleRate
# original_sample_res=datalengthSec/datalength
# print("Length of file is ", datalengthSec, " seconds (", datalength, "samples) with sample rate ",sampleRate, " Hz.")
# print("Each sample correspond to ", original_sample_res, "seconds")

# #resample
# fs_resample=8000
# sp.resample(fs_resample)
# #update
# sampleRate = sp.sampleRate
# audiodata = sp.data
# datalength = np.shape(audiodata)[0]
# datalengthSec = datalength / sampleRate
# new_sample_res=datalengthSec/datalength
# print('\n After downsampling \n')
# print("Length of file is ", datalengthSec, " seconds (", datalength, "samples) with sample rate ",sampleRate, " Hz.")
# print("Each sample correspond to ", new_sample_res, "seconds")

# #save new_file
# wavFile = audiofile[:-4] + '_downsampled'+str(fs_resample)+'.wav'
# wavio.write(wavFile, audiodata, sampleRate, sampwidth=2)

###time-expand

#Read audiodata
#fs=16000
sp = SignalProc.SignalProc(1024, 512) #outside?
audiofile="D:\\Desktop\\Documents\\Work\\Zohara files\\TEST\\67375127.140303193211_downsampled8000.wav"
sp.readWav(audiofile)
audiodata = sp.data
sampleRate = sp.sampleRate
datalength = np.shape(audiodata)[0]
datalengthSec = datalength / sampleRate
print("Length of file is ", datalengthSec, " seconds (", datalength, "samples) with sample rate ",sampleRate, " Hz.")


#save new_file
samplerate2=4000
wavFile = audiofile[:-4] + '_timeexpanded'+str(samplerate2)+'.wav'
wavio.write(wavFile, audiodata, samplerate2, sampwidth=2)
sp.readWav(wavFile)
audiodata = sp.data
sampleRate = sp.sampleRate
datalength = np.shape(audiodata)[0]
datalengthSec = datalength / sampleRate
new_sample_res=datalengthSec/datalength
print("After timeexpansion")
print("Length of file is ", datalengthSec, " seconds (", datalength, "samples) with sample rate ",sampleRate, " Hz.")
print("Each sample correspond to ", new_sample_res, "seconds")


##Raven to excel
def ravenexcel2data(wav,excelFile):
    import os
    import openpyxl
    # read excel file
    book = openpyxl.load_workbook(excelFile)
    sheet = book.active
    starttime = sheet['A2': 'A231']
    endtime = sheet['B2': 'B231']
    flow = sheet['C2': 'C231']
    fhigh = sheet['D2': 'D231']
    _, duration, _, _ = wavio.readFmt(wav)
    annotation = []
    for i in range(len(starttime)):
        annotation.append([float(starttime[i][0].value), float(endtime[i][0].value), float(flow[i][0].value), float(fhigh[i][0].value),
                           [{"species": "Bigeye", "certainty": 100.0, "filter": "M", "calltype": "Pop"}]])
    annotation.insert(0,{"Operator": "Zohara", "Reviewer": "", "Duration": duration})
    file = open(wav + '.data', 'w')
    json.dump(annotation, file)
    file.close()

