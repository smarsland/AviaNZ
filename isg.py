
# convert_image_2sound.py

# Script to invert bat echolocation spectrograms

# Version 1.3 23/10/18
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis

#    AviaNZ birdsong analysis program
#    Copyright (C) 2017--2018

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
import SignalProc as sp
import pylab as pl
import numpy as np
#import cv2  # image -processing
import os  # linux shell comands
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wavio
from PyQt5.QtGui import QImage

"""
This script works in batch. It generate sound files from .bmp spectrogram images.
There are two option (not used is commented)
1) Audible file -> freqnecy shift to hear bat echolocation [ON]
2) Same frequency band -> just the spectrogram inverted [OFF]
NOTE: we need appropriate number of frequency bins in order to make invertSpectrogram work
"""

#dirName='/home/listanvirg/Data/Bat/BAT/TEST_DATA/'
#dirName='D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\TRAIN_DATA\\NONE'

for file in ['/Users/marslast/20191106_022512.bmp']:
    if file.endswith('.bmp'):
        #bmpFile = root + '/' + file[:-4]
        #imgFile=root + '/'+file
        #print(imgFile)
        #img = mpimg.imread(imgFile) #read image
        #img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale

        img = QImage(file, "BMP")
        h = img.height()
        w = img.width()
        colc = img.colorCount()
        img.convertTo(QImage.Format_Grayscale8)
        ptr = img.constBits()
        ptr.setsize(h*w*1)
        img2 = np.array(ptr).reshape(h, w)
        img2 = img2.T

        a = sp.SignalProc(window_width=1024, incr=512)
        print(np.shape(img2))

        img2[-1, :] = 254 * np.ones((np.shape(img2)[1])) #cut last row

        # OPTION 1: AUFIBLE FILE
        row_dim = 7 * np.shape(img2)[0]
        appo = np.zeros((row_dim, np.shape(img2)[1]))
        spec = np.concatenate((appo, img2))
        print(np.shape(spec))
        samplerate = 16000

        #OPTION 2: same frequency band
        #spec=np.repeat(img2,8, axis=0) #repeat rows 7 times to fit invertspectrogram
        # samplerate = 176000

        spec = -(spec - 254.0 * np.ones(np.shape(spec)))  # reverse value having the black as the most intense
        spec=np.flipud(spec) #reverse up and down of the spectrogram -> see AviaNZ spectrogram
        spec = spec/np.max(spec) #normalization
        spec = spec[:, 1:np.shape(img2)[1]]  # Cutting first column because it only contains the scale
        spec2 = spec.T
        (n, m) = np.shape(spec)
        print(np.shape(spec2))
        wave = a.invertSpectrogram(spec2, 1024, 512)

        #wave = a.bandpassFilter(wave, samplerate, 1000, 22000) bandpass?
        soundFile = 'test.wav'
        #soundFile=bmpFile+'_audible.wav' #Other option
        wavio.write(soundFile, wave, samplerate, sampwidth=2)
