import SignalProc as sp
import pylab as pl
import numpy as np
import cv2  # image -processing
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
#dirName='D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TRAIN4'
dirName= "C:\\Users\\Virginia\\Documents\\GitHub\\AviaNZ\\Sound Files\\"
a = sp.SignalProc(window_width=1024, incr=512)


for root, dirs, files in os.walk(str(dirName)):
    for file in files:
        if file.endswith('.bmp'):
            bmpFile = root + '/' + file[:-4]
            imgFile=root + '/'+file
            print(imgFile)
            #img = mpimg.imread(imgFile) #read image
            #img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            #img2[-1, :] = 254 * np.ones((np.shape(img2)[1])) #cut last row

            #from Julius readBmp: this works

            #silent=False
            #img = QImage(bmpFile, "BMP")
            #h = img.height()
            #w = img.width()
            #colc = img.colorCount()
            #if h==0 or w==0:
            #    print("ERROR: image was not loaded")

            ## Check color format and convert to grayscale
            #if not silent and (not img.allGray() or colc>256):
            #    print("Warning: image provided not in 8-bit grayscale, information will be lost")
            #img.convertTo(QImage.Format_Grayscale8)

            ## Convert to numpy
            # # (remember that pyqtgraph images are column-major)
            #ptr = img.constBits()
            #ptr.setsize(h*w*1)
            #img2 = np.array(ptr).reshape(h, w)

            #calling signal proc routine directly

            a.readBmp(bmpFile)
            img2=a.sg
            print(img2)
            print(np.shape(img2))
            img2 = np.rot90(img2, -1, (1,0)) #undo rotation
            print(np.shape(img2))
            
            img2=img2[::8,:]
            print(np.shape(img2))
                
            # OPTION 1: AUFIBLE FILE
            row_dim = 7 * np.shape(img2)[0]
            #appo = 254 * np.ones((row_dim, np.shape(img2)[1]))
            appo = np.zeros((row_dim, np.shape(img2)[1]))
            spec = np.concatenate((appo, img2))
            samplerate = 176000

            #OPTION 2: same frequency band
            #spec=np.repeat(img2,8, axis=0) #repeat rows 7 times to fit invertspectrogram
            # samplerate = 176000

            #spec = -(spec - 254.0 * np.ones(np.shape(spec)))  # reverse value having the black as the most intense
            spec=np.flipud(spec) #reverse up and down of the spectrogram -> see AviaNZ spectrogram
            #spec = spec/np.max(spec) #normalization
            #spec = spec[:, 1:np.shape(img2)[1]]  # Cutting first column because it only contains the scale
            spec2 = spec.T
            print(spec2)
            (n, m) = np.shape(spec)
            
            wave = a.invertSpectrogram(spec2, 1024, 512, window='Blackman')

            #wave = a.bandpassFilter(wave, samplerate, 1000, 22000) bandpass?
            wavFile = bmpFile + '.wav'
            #wavFile=bmpFile+'_audible.wav' #Other option
            wavio.write(wavFile, wave, samplerate, sampwidth=2)

