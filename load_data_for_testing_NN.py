import numpy as np
import SupportClasses
import Training
import NN
import shutil
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import NNModels
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
from PIL import Image
from itertools import cycle
from tensorflow.keras.applications import EfficientNetV2S

class DataLoader:
    # This will take a folder with segments and process it into spectrogram images.
    # Each segment will produce a number of different versions corresponding to slightly different starting points.
    # Files will be saved in a folder structure like:
    # your_folder/
    # ├── 0/
    # │   ├── segment_0/
    # │   │   ├── 0_segment-000000_version-000000_140517_181503.npy
    # │   │   ├── 0_segment-000000_version-000001_140517_181503.npy
    # │   │   └── 0_segment-000000_version-000002_140517_181503.npy
    # │   └── segment_1/
    # │       └── ...
    # ├── 1/
    # │   ├── segment_8/
    # │   │   └── ...
    # │   └── segment_23/
    # │       └── ...
    # └...
    def __init__(self, filterName, segmentDataFolder, imageFolder, duration, imgsize):
        configdir = os.path.expanduser("~/.avianz/")
        configfile = os.path.join(configdir, "AviaNZconfig.txt")
        ConfigLoader = SupportClasses.ConfigLoader()
        config = ConfigLoader.config(configfile)
        filtdir = os.path.join(configdir, config['FiltersDir'])
        self.filt = ConfigLoader.filters(filtdir)[filterName]
        self.segmentDataFolder = segmentDataFolder
        self.imageFolder = imageFolder
        self.duration = duration
        self.imgsize = imgsize

    def loadSegmentsToImageDataset(self):
        # set the spectrogram window
        windowWidth = self.imgsize[0] * 2
        totalLength = self.duration * self.filt["SampleRate"]
        windowInc = int(np.floor((totalLength - windowWidth) / (self.imgsize[1] - 1)))
        f1 = np.min([fi['FreqRange'][0] for fi in self.filt["Filters"]])
        f2 = np.max([fi['FreqRange'][1] for fi in self.filt["Filters"]])
        
        # load segments
        DataGen = NN.GenerateData(self.filt, windowWidth, windowInc, f1, f2)
        traindata = []
        nCalltypes = len(self.filt["Filters"])
        for i in range(nCalltypes):
            traindata = traindata + DataGen.findCTsegments(self.segmentDataFolder, i)
        traindata = traindata + DataGen.findNoisesegments(self.segmentDataFolder)[:int(len(traindata)/nCalltypes)] # only take as much noise as there are other classes.

        # make the folders
        if os.path.exists(self.imageFolder):
            shutil.rmtree(self.imageFolder)
        os.makedirs(self.imageFolder)
        for ct in range(nCalltypes + 1):
            os.makedirs(os.path.join(self.imageFolder, str(ct)))

        # save data
        N = DataGen.generateFeatures(dirName=self.imageFolder, dataset=traindata)


if __name__=="__main__":
    filterName = "Kiwi (Great Spotted)"
    segmentDataFolder = "/home/giotto/Desktop/AviaNZ/Sound Files/learning/train"
    imageFolder = "/home/giotto/Desktop/AviaNZ/Sound Files/learning/imageDataset"
    duration = 20
    imgsize = [224,224]

    loader = DataLoader(filterName,segmentDataFolder, imageFolder, duration,imgsize)
    loader.loadSegmentsToImageDataset()