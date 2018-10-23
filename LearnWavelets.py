
# LearnWavelets.py
#
# One set of wavelet classes

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
import pywt
import numpy as np
import WaveletSegment

# mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
# os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
# import xgboost as xgb
# from sklearn.externals import joblib
# clf_maleKiwi = joblib.load('maleKiwiClassifier.pkl')
# clf_femaleKiwi = joblib.load('femaleKiwiClassifier.pkl')
# clf_ruru = joblib.load('ruruClassifier.pkl')


# def test(nodelist):
#     for filename in glob.glob(os.path.join('E:/SONGSCAPE/birdscapeConda2/Sound Files/test','*.wav')):
#         findCalls_test(nodelist,filename[:-4])
#
# test(nodelist_kiwi) # TESTING

# #Survey data processing
# #First split audio into 5-min to a subfolder '5min'
# ws=WaveletSeg()
# ws.splitAudio(folder_to_process='Sound Files/survey')
# print "5-min splitting done"
# Now to process survey data
# detected=processFolder(folder_to_process='Sound Files/survey/5min', species='kiwi')
# genReport(folder_to_process='Sound Files/survey/5min',detected=detected)
# print detected

# print findCalls_train('E:/SONGSCAPE/birdscapeConda2/Sound Files/train/kiwi/train1',species='kiwi')

#ws = WaveletSeg()
#ws.loadData('Wavelet Segmentation/kiwi/train/train1')

#fs = 16000
#if ws.sampleRate != fs:
    #ws.data = librosa.core.audio.resample(ws.data, ws.sampleRate, fs)
    #ws.sampleRate = fs

# Get the five level wavelet decomposition
#wData = ws.denoise(ws.data, thresholdType='soft', maxlevel=5)
# librosa.output.write_wav('train/kiwi/D/', wData, sampleRate, norm=False)

# Bandpass filter
# fwData = bandpass(wData,sampleRate)
# TODO: Params in here!
# bittern
# fwData = ButterworthBandpass(wData,sampleRate,low=100,high=400)
# kiwi
#fwData = ws.ButterworthBandpass(wData, ws.sampleRate, low=1100, high=7500)
#print fwData

#fwData = ws.data
# fwData = data
#waveletCoefs = ws.computeWaveletEnergy(fwData, ws.sampleRate)

#np.savetxt('waveout.txt',waveletCoefs)

# findCalls_train('E:/SONGSCAPE/birdscapeConda2/Sound Files/train/kiwi/train1')

# processFolder_train()  #TRAINING


#***************************************************************************************
# Ready DATA SET (using 5 min recordings) FOR ML

# def read_tgt(file):
#     annotation = np.zeros(300)
#     count = 0
#     import xlrd
#     wb=xlrd.open_workbook(filename = file)
#     ws=wb.sheet_by_index(0)
#     col=ws.col(1)
#     for row in range(1,301):
#         annotation[count]=col[row].value
#         count += 1
#     return annotation
#
# data=np.zeros((8*300,63))    #8 rec =8*300=2400 data points, 62 nodes + target (0/1)
# # train1
# d = np.loadtxt("Sound Files/MLPdata/train/train1_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train1-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[0:300,0:62]=d.transpose()
# data[0:300,62]=t
# # train2
# d = np.loadtxt("Sound Files/MLPdata/train/train2_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train2-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[300:600,0:62]=d.transpose()
# data[300:600,62]=t
# # train3
# d = np.loadtxt("Sound Files/MLPdata/train/train3_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train3-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[600:900,0:62]=d.transpose()
# data[600:900,62]=t
# # train4
# d = np.loadtxt("Sound Files/MLPdata/train/train4_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train4-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[900:1200,0:62]=d.transpose()
# data[900:1200,62]=t
# # train5
# d = np.loadtxt("Sound Files/MLPdata/train/train5_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train5-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[1200:1500,0:62]=d.transpose()
# data[1200:1500,62]=t
# # train6
# d = np.loadtxt("Sound Files/MLPdata/train/train6_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train6-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[1500:1800,0:62]=d.transpose()
# data[1500:1800,62]=t
# # train7
# d = np.loadtxt("Sound Files/MLPdata/train/train7_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train7-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[1800:2100,0:62]=d.transpose()
# data[1800:2100,62]=t
# # train8
# d = np.loadtxt("Sound Files/MLPdata/train/train8_kiwi_DF_E_all.dat",delimiter=',')
# t=read_tgt("Sound Files/MLPdata/train/train8-sec.xlsx")
# print np.shape(d)
# print np.shape(t)
# data[2100:2400,0:62]=d.transpose()
# data[2100:2400,62]=t
#
# # Save it as 2400 sec *63 [62 nodes + tgt]
# np.savetxt('Sound Files/MLPdata/train/DF_E_all.dat', data, delimiter=',')

#***************************************************************************************
# Create DATA SET FOR ML - using annotations made with AviaNZ
def CreateDataSet(directory,species='kiwi',choice='all',denoise=False):
    #Generate the wavelet energy (all nodes)give the directory with sound and annotation
    ws=WaveletSegment.WaveletSegment()
    if choice=='all' and denoise==False:
        filename='wEnergyAll.data'
    elif choice=='all' and denoise==True:
        filename='wEnergyAllDenoised.data'
    elif choice=='bandpass' and denoise==False:
        filename='wEnergyBandpass.data'
    elif choice=='bandpass' and denoise==True:
        filename='wEnergyBandpassDenoised.data'
    f2=open(str(directory)+'/'+filename,'a')
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.wav'):
                if not os.path.isfile(root+'/'+filename+'.data'): # if no AviaNZ annotation then skip
                    continue
                ws.loadData(root+'/'+filename[:-4],trainTest=False)
                if species !='boom' and ws.sampleRate!=16000:
                    ws.data=librosa.core.audio.resample(ws.data,ws.sampleRate,16000)
                    ws.sampleRate=16000
                # Read the AviaNZ annotation
                with open(root+'/'+filename+".data") as f:
                    segments = json.load(f)
                segments=segments[1:]
                for seg in segments:
                    # If the length of a segment is less than 1 sec make it 1sec
                    if seg[1]-seg[0]<1:
                        seg[1]=seg[0]+1
                    # Discard the tail (<1 sec)
                    seg[1]=seg[0]+np.floor(seg[1]-seg[0])
                    n=int(seg[1]-seg[0])
                    for i in range(n):
                        current=ws.data[(int(seg[0])+i)*ws.sampleRate:(int(seg[0])+(i+1))*ws.sampleRate]
                        # Compute wavelet energy for this second
                        E=computeWaveletEnergy_1s(current,'dmey2',choice,denoise)
                        # E=genWEnergy(filename[:-4],species=species) # try later with bp filter e.g when trainig for kiwi male use 1200-7500
                        E=E.tolist()
                        spp=str(seg[4])
                        if 'Noise' in spp:
                            target=0
                        else:
                            target=1
                        E.append(spp)
                        E.append(target)
                        f2.write(str(E)[1:-1]+"\n")
    f2.close()

def CreateDataSet_data(directory,species='kiwi',choice='all',denoise=False):
    # Generate the wavelet energy (all nodes)give the directory with sound and annotation
    # This also saves all the 1sec data along with their labels to be used in findCalls_train_learning
    # to find the wavelet nodes
    ws=WaveletSegment.WaveletSegment()
    if choice=='all' and denoise==False:
        filename='wE.data'
    elif choice=='all' and denoise==True:
        filename='wED.data'
    elif choice=='bandpass' and denoise==False:
        filename='wEB.data'
    elif choice=='bandpass' and denoise==True:
        filename='wEDB.data'
    f0=open(str(directory)+'/data-1s.data','a')   # 1 sec data
    f1=open(str(directory)+'/label-1s','a')      # 1 sec labels
    f2=open(str(directory)+'/'+filename,'a')     # wavelet segments

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.wav'):
                if not os.path.isfile(root+'/'+filename+'.data'): # if no AviaNZ annotation then skip
                    continue
                ws.loadData(root+'/'+filename[:-4],trainTest=False)
                if species !='boom' and ws.sampleRate!=16000:
                    ws.data=librosa.core.audio.resample(ws.data,ws.sampleRate,16000)
                    ws.sampleRate=16000
                # Read the AviaNZ annotation
                with open(root+'/'+filename+".data") as f:
                    segments = json.load(f)
                for seg in segments:
                    # If the length of a segment is less than 1 sec make it 1sec
                    if seg[1]-seg[0]<1:
                        seg[1]=seg[0]+1
                    # Discard the tail (<1 sec)
                    seg[1]=seg[0]+np.floor(seg[1]-seg[0])
                    n=int(seg[1]-seg[0])
                    for i in range(n):
                        current=ws.data[(int(seg[0])+i)*ws.sampleRate:(int(seg[0])+(i+1))*ws.sampleRate]
                        # Compute wavelet energy for this second
                        E=computeWaveletEnergy_1s(current,'dmey2',choice,denoise)
                        # E=genWEnergy(filename[:-4],species=species) # try later with bp filter e.g when trainig for kiwi male use 1200-7500
                        E=E.tolist()
                        spp=str(seg[4])
                        if 'Noise' in spp:
                            target=0
                        else:
                            target=1
                        E.append(spp)
                        E.append(target)
                        current=current.tolist()
                        f0.write(str(current)[1:-1]+"\n")
                        f1.write(str(spp)+"\n")
                        f2.write(str(E)[1:-1]+"\n")
    f2.close()

# CreateDataSet(directory= 'E:/AviaNZ/Sound Files/MLdata',choice='all',denoise=False)
# CreateDataSet(directory= 'E:/AviaNZ/Sound Files/MLdata',choice='bandpass',denoise=False)
# CreateDataSet(directory= 'E:/AviaNZ/Sound Files/MLdata',choice='bandpass',denoise=True)

# CreateDataSet(directory= 'E:/AviaNZ/Sound Files/testsmall',choice='all',denoise=False)


# CreateDataSet_data(directory= 'E:/AviaNZ/Sound Files/MLdata/',choice='all',denoise=False)

# findCalls_train_learning(species='kiwi')
# moretest()

