# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 13:47:34 2020

@author: Virginia Listanti

aid script used for developing BatSearch_images.py
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

fs=1600
filename="D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\290116_001439.bmp"
img = mpimg.imread(filename) #read image
print('image shape', np.shape(img))
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
img2[-1, :] = 254 * np.ones((np.shape(img2[1]))) #cut last row
imspec_bmp=np.repeat(img2,8, axis=0) #repeat rows 7 times to fit invertspectrogram
imspec_bmp = -(imspec_bmp - 254.0 * np.ones(np.shape(imspec_bmp)))  # reverse value having the black as the most intense
#    imspec=np.flipud(imspec) #reverse up and down of the spectrogram -> see AviaNZ spectrogram
imspec_bmp = imspec_bmp/np.max(imspec_bmp) #normalization
imspec_bmp = imspec_bmp[:, 1:np.shape(img2)[1]]  # Cutting first column because it only contains the scale



audiodata = wavio.read(filename[:-3]+'wav')
sp = SignalProc.SignalProc(1024, 512) #outside?
sp.data = audiodata.data
duration=audiodata.nframes/fs

#copyed from sp.wavRead to make everything consistent
# take only left channel
if np.shape(np.shape(sp.data))[0] > 1:
    sp.data = sp.data[:, 0]
sp.audioFormat.setChannelCount(1)
 # force float type
if sp.data.dtype != 'float':
    sp.data = sp.data.astype('float')
sp.audioFormat.setSampleSize(audiodata.sampwidth * 8)

#Spectrogram
sp.samplerate= fs
sgraw_wav= sp.spectrogram(1024, 512, 'Blackman')
imspec_wav=(10.*np.log10(sgraw_wav)).T #transpose
imspec_wav=np.flipud(imspec_wav) #updown 


df_wav=16000/(np.shape(imspec_wav)[0]+1) #frequency increment 
dt_wav=duration/(np.shape(imspec_wav)[1]+1) #timeincrement
print('\n Wav file \n')
print('Wav spectrogram shape', np.shape(imspec_wav))
print('df', df_wav)
print('dt', dt_wav)



df_bmp=176000/(np.shape(imspec_bmp)[0]+1) #frequency increment 
dt_bmp=(duration/11)/(np.shape(imspec_bmp)[1]+1) #timeincrement
print('\n Bmp file \n')
print('Bmp spectrogram shape', np.shape(imspec_bmp))
print('df', df_bmp)
print('dt', dt_bmp)

print('\n Checks \n')
print('df_wav=', df_wav)
print('df_bmp / 11 =', df_bmp/11)
print('dt_wav =', dt_wav)
print('dt_bmp * 11 =', dt_bmp*11)

fs=1600
filename="D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\061216_013023.bmp"
img = mpimg.imread(filename) #read image
print('image shape', np.shape(img))
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
img2[-1, :] = 254 * np.ones((np.shape(img2[1]))) #cut last row
imspec_bmp=np.repeat(img2,8, axis=0) #repeat rows 7 times to fit invertspectrogram
imspec_bmp = -(imspec_bmp - 254.0 * np.ones(np.shape(imspec_bmp)))  # reverse value having the black as the most intense
#    imspec=np.flipud(imspec) #reverse up and down of the spectrogram -> see AviaNZ spectrogram
imspec_bmp = imspec_bmp/np.max(imspec_bmp) #normalization
imspec_bmp = imspec_bmp[:, 1:np.shape(img2)[1]]  # Cutting first column because it only contains the scale



audiodata = wavio.read(filename[:-3]+'wav')
sp = SignalProc.SignalProc(1024, 512) #outside?
sp.data = audiodata.data
duration=audiodata.nframes/fs

#copyed from sp.wavRead to make everything consistent
# take only left channel
if np.shape(np.shape(sp.data))[0] > 1:
    sp.data = sp.data[:, 0]
sp.audioFormat.setChannelCount(1)
 # force float type
if sp.data.dtype != 'float':
    sp.data = sp.data.astype('float')
sp.audioFormat.setSampleSize(audiodata.sampwidth * 8)

#Spectrogram
sp.samplerate= fs
sgraw_wav= sp.spectrogram(1024, 512, 'Blackman')
imspec_wav=(10.*np.log10(sgraw_wav)).T #transpose
imspec_wav=np.flipud(imspec_wav) #updown 


df_wav=16000/(np.shape(imspec_wav)[0]+1) #frequency increment 
dt_wav=duration/(np.shape(imspec_wav)[1]+1) #timeincrement
print('\n Wav file \n')
print('Wav spectrogram shape', np.shape(imspec_wav))
print('df', df_wav)
print('dt', dt_wav)



df_bmp=176000/(np.shape(imspec_bmp)[0]+1) #frequency increment 
dt_bmp=(duration/11)/(np.shape(imspec_bmp)[1]+1) #timeincrement
print('\n Bmp file \n')
print('Bmp spectrogram shape', np.shape(imspec_bmp))
print('df', df_bmp)
print('dt', dt_bmp)

print('\n Checks \n')
print('df_wav=', df_wav)
print('df_bmp / 11 =', df_bmp/11)
print('dt_wav =', dt_wav)
print('dt_bmp * 11 =', dt_bmp*11)
