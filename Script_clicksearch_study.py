# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:35:19 2019

@author: Virginia Listanti
"""
#This script is intended as a tool to study a way to find clicks on resempled
#recordings

import SignalProc
import cv2
import scipy
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import WaveletFunctions
import wavio
import math
import pyqtgraph as pg
import pyqtgraph.exporters as pge


#def loadFile(filename, duration=0, offset=0, fs=0, denoise=False, f1=0, f2=0):
#    """
#    Read audio file and preprocess as required.
#    """
#
#    print(filename)
#    if duration == 0:
#        duration = None
#
#    sp = SignalProc.SignalProc(1024, 512) # changed fuorier parameters
#    wavobj=wavio.read(filename)
#    sp.data=wavobj.data
#    sp.sampleRate=16000
##    print(len(wavobj))
#    #sp.resample(fs)
#    # sampleRate=16000
#    #sampleRate = sp.sampleRate
#    audiodata = sp.data
#
#    # # pre-process
#    if denoise:
#        WF = WaveletFunctions.WaveletFunctions(data=audiodata, wavelet='dmey2', maxLevel=10, samplerate=fs)
#        audiodata = WF.waveletDenoise(thresholdType='soft', maxLevel=10)
#
#    if f1 != 0 and f2 != 0:
#        # audiodata = sp.ButterworthBandpass(audiodata, sampleRate, f1, f2)
#        audiodata = sp.bandpassFilter(audiodata, sampleRate, f1, f2)
#
#    return audiodata

fs=16000
## LT
#filelist=["D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\002506.wav", 
#          "D:\\Desktop\\Documents\\Work\\Data\Bat\\BAT\\CLICK SEARCH TEST\\005115.wav",
#          "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\056475.wav",
#          "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\171215_005049.wav",
#          "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\100216_061715.wav"]

##ST
#filelist=["D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\140269.wav", 
#          "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\149085.wav",
#          "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\196675.wav",
#          "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\181217_015712.wav",
#          "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\201217_015533.wav",
#          "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\010316_002730.wav"]


##Noise
filelist=["D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\000013.wav",
          "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\000027.wav",
          "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\000680.wav",
          "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\001321.wav",
          "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\230416_032954.wav",
          "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\090216_231740.wav"]


#inizializing cont_row and cont_column to plot
count_row=0
count_column=0
fig, axes=plt.subplots(6,3,sharex='col')
for k in range(len(filelist)):

    filename=filelist[k]
    
    audiodata = wavio.read(filename)
    sp = SignalProc.SignalProc(1024, 512) 
    sp.data = audiodata.data
    duration=audiodata.nframes/fs
    print(duration) #of seconds
    
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
    sgraw= sp.spectrogram(1024, 512, 'Blackman')
    imspec=(10.*np.log10(sgraw)).T #transpose
    imspec=np.flipud(imspec) #updown
    print(np.shape(imspec))
    #print(np.shape(imspec)[0])
    #print(np.shape(imspec)[1])
    df=16000/(np.shape(imspec)[0]+1) #frequency increment 
    dt=duration/(np.shape(imspec)[1]+1) #timeincrement
    up_len=math.ceil(0.5/dt) #1 second lenth in indices  
    low_len=math.floor(0.1/dt)
    print(low_len, up_len)
    
    #sum along colums
    f0=2000
    index_f0=-1+math.ceil(f0/df) #lower bound needs to be rounded down
    print(f0,index_f0)
    f1=6000
    index_f1=-1+math.floor(f1/df) #upper bound needs to be rounded up
    print(f1,index_f1)
    print(np.shape(imspec[index_f0:index_f1,:]))
#    sum_spec=np.sum(imspec[index_f0:index_f1,:], axis=0) #added 0.01 to avoid divition by 0
    mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0) #added 0.01 to avoid divition by 0
    x_axis=np.arange(np.shape(imspec)[1])
    mean_spec_all=np.mean(imspec, axis=0)[2:]
    thr_spec=(np.mean(mean_spec_all)+np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))
    print(np.max(mean_spec),thr_spec,np.std(mean_spec_all))
#    thr_spec=(np.mean(sum_spec[2:])+np.std(sum_spec[2:]))*np.ones((np.shape(sum_spec)))
    
    ##clickfinder
    #check when the sum is bigger than the mean
    #clicks is an array which elements are equal to 1 only where the sum is bigger 
    #than the mean, otherwise are equal to 0
#    clicks=np.where(sum_spec>thr_spec,1,0) 
    clicks=np.where(mean_spec>thr_spec,1,0)
    clicks_indices=np.nonzero(clicks)
#    print(np.shape(clicks_indices))
    print(np.shape(clicks_indices)[1])
    
    click_start=clicks_indices[0][0]
#    print(int(click_start))
    click_end=clicks_indices[0][0]
#    print(int(click_end))
    for i in range(1,np.shape(clicks_indices)[1]):
        if clicks_indices[0][i]==click_end+1:
            click_end=clicks_indices[0][i]
        else:
            if click_end-click_start+1>up_len or click_end-click_start+1<low_len:
                #print('check')
                clicks[click_start:click_end+1]=0
            click_start=clicks_indices[0][i]
            click_end=clicks_indices[0][i]
                
    #checking last loop with end
    if click_end-click_start+1>up_len:
        clicks[click_start:click_end+1]=0   
    elif click_end-click_start+1<low_len:
        clicks[click_start:click_end+1]=0
    clicks_indices=np.nonzero(clicks)
    print(np.shape(clicks_indices)[1])
    
    #plot
    if k<3:
        count_row=0
    else:
        count_row=3
    if k==0 or k==3:
        count_column=0
    elif k==1 or k==4:
        count_column=1
    else:
        count_column=2
#    fig, axes=plt.subplots(5,3,sharex='col')
    axes[count_row][count_column].imshow(imspec,aspect='auto')
    axes[count_row+1][count_column].plot(x_axis,mean_spec, thr_spec,'r')
    axes[count_row+2][count_column].plot(clicks)
    
plt.show()
exporter = pge.ImageExporter(fig.view)
exporter.export("D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CLICK SEARCH TEST\\LT_test.png")


#ax1=plt.subplot(2, 1, 1, xscale='linear')
#plt.imshow(imspec)
#
#ax2=plt.subplot(2, 1, 2, sharex=ax1, xscale='linear')
#plt.plot(sum_spec)
#
#plt.show()


#plt.figure()
#imgplot=plt.imshow(imspec)
#plt.show()
