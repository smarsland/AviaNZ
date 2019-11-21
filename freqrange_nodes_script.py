# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:51:55 2019

@author: User
"""

import WaveletFunctions
import wavio

data=wavio.read('D:\Desktop\Documents\Work\Data\Bat\BAT\TRAIN_DATA\LT\002506.wav')
wf= WaveletFunctions(data,'dmey2',5,16000)
f1=wf.getWCFreq(28,16000)
print('28 is ',f1)