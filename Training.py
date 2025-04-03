
# Version 3.4 18/12/24
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2024

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

# Holds most of the code for training NNs

import os, gc, re, json, tempfile
from shutil import copyfile
from shutil import disk_usage

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
import matplotlib.pyplot as plt
from time import strftime, gmtime
import math

import SupportClasses
import Spectrogram
import NN
import Segment, WaveletSegment
import AviaNZ_batch

import soundfile as sf

import NNModels
from tensorflow.keras.utils import custom_object_scope

class NNtrain:

    def __init__(self, configdir, filterdir, folderTrain1=None, folderTrain2=None, recogniser=None, imgWidth=0, CLI=False):
        # Two important things: 
        # 1. LearningParams.txt, which a dictionary of parameters *** including spectrogram parameters
        # 2. CLI: whether it runs off the command line, which makes picking the ROC curve parameters hard
        # Qn: what is imgWidth? Why not a learning param? Ans: it is a slider in the UI and sets the input data shape

        self.filterdir = filterdir
        self.configdir =configdir
        cl = SupportClasses.ConfigLoader()
        self.FilterDict = cl.filters(filterdir, bats=False)
        self.LearningDict = cl.learningParams(os.path.join(configdir, "LearningParams.txt"))
        self.sp = Spectrogram.Spectrogram(self.LearningDict['sgramWindowWidth'], self.LearningDict['sgramHop'])

        self.imgsize = [self.LearningDict['imgX'], self.LearningDict['imgY']]
        self.tmpdir1 = False
        self.tmpdir2 = False
        self.ROCdata = {}

        self.CLI = CLI
        if CLI:
            self.filterName = recogniser
            self.folderTrain1 = folderTrain1
            self.folderTrain2 = folderTrain2
            self.imgWidth = imgWidth
            self.autoThr = True
            self.correction = True
            self.annotatedAll = True
        else:
            self.autoThr = False
            self.correction = False
            self.imgWidth = imgWidth
        
        self.modelArchitecture = "CNN"

    def setP1(self, folderTrain1, folderTrain2, recogniser, annotationLevel):
        # This is a function that the Wizard calls to set parameters
        self.folderTrain1 = folderTrain1
        self.folderTrain2 = folderTrain2
        self.filterName = recogniser
        self.annotatedAll = annotationLevel

    def setP6(self, recogniser):
        # This is a function that the Wizard calls to set parameters
        self.newFilterName = recogniser
            
    def cliTrain(self):
        # This is the main training function for CLI-based learning.
        # It proceeds very much like the wizard 
            
        # Get info from wavelet filter
        self.readFilter()

        # Load data
        # Note: no error checking in the CLI version
        # Find segments belong to each class in the training data
        self.genSegmentDataset(hasAnnotation=True)

        # Check on memory space
        self.checkDisk()

        # OK, WTF?
        self.windowWidth = self.imgsize[0] * self.LearningDict['windowScaling']
        self.windowInc = int(np.ceil(self.imgWidth * self.fs / (self.imgsize[1] - 1)) )

        # Train
        self.train()

        # Save the output
        self.saveFilter()

    def readFilter(self):
        # Read the current (wavelet) filter and get the details
        if self.filterName.lower().endswith('.txt'):
            self.currfilt = self.FilterDict[self.filterName[:-4]]
        else:
            self.currfilt = self.FilterDict[self.filterName]

        self.fs = self.currfilt["SampleRate"]
        self.species = self.currfilt["species"]

        mincallengths = []
        maxcallengths = []
        f1 = []
        f2 = []
        self.maxgaps = []
        self.calltypes = []
        for fi in self.currfilt['Filters']:
            self.calltypes.append(fi['calltype'])
            mincallengths.append(fi['TimeRange'][0])
            maxcallengths.append(fi['TimeRange'][1])
            self.maxgaps.append(fi['TimeRange'][3])
            f1.append(fi['FreqRange'][0])
            f2.append(fi['FreqRange'][1])
        self.mincallength = np.max(mincallengths)
        self.maxcallength = np.max(maxcallengths)
        self.f1 = np.min(f1)
        self.f2 = np.max(f2)

        print("Manually annotated: %s" % self.folderTrain1)
        print("Auto processed and reviewed: %s" % self.folderTrain2)
        print("Recogniser: %s" % self.currfilt)
        print("Species: %s" % self.species)
        print("Call types: %s" % self.calltypes)
        print("Call length: %.2f - %.2f sec" % (self.mincallength, self.maxcallength))
        print("Sample rate: %d Hz" % self.fs)
        print("Frequency range: %d - %d Hz" % (self.f1, self.f2))

    def checkDisk(self):
        # Check disk usage
        totalbytes, usedbytes, freebytes = disk_usage(os.path.expanduser("~"))
        freeGB = freebytes/1024/1024/1024
        print('\nFree space in the user directory: %.2f GB/ %.2f GB\n' % (freeGB, totalbytes/1024/1024/2014))
        if freeGB < 10:
            print('Warning: You may run out of space in the user directory!')
        return freeGB, totalbytes/1024/1024/1024

    def genSegmentDataset(self, hasAnnotation):
        # Prepares segments for input to the learners
        self.traindata = []
        self.DataGen = NN.GenerateData(self.currfilt, 0, 0, 0, 0, 0, 0, 0)

        # For manually annotated data where the user is confident about full annotation, 
        # choose anything else in the spectrograms as noise examples
        if not self.folderTrain1=="":
            if self.annotatedAll=="All":
                self.noisedata1 = self.DataGen.findNoisesegments(self.folderTrain1)
                print('----noise data1:')
                for x in self.noisedata1:
                    self.traindata.append(x)
            if self.annotatedAll=="All-nowt":
                self.noisedata1 = self.DataGen.findAllsegments(self.folderTrain1)
                print('----noise data1:')
                for x in self.noisedata1:
                    self.traindata.append(x)

            # Call type segments
            print('----CT data1:')
            if hasAnnotation:
                for i in range(len(self.calltypes)):
                    ctdata = self.DataGen.findCTsegments(self.folderTrain1, i)
                    print(self.calltypes[i])
                    for x in ctdata:
                        self.traindata.append(x)

        if not self.folderTrain2=="":
            # For wavelet outputs that have been manually verified get noise segments from .corrections
            if os.path.isdir(self.folderTrain2):
                for root, dirs, files in os.walk(str(self.folderTrain2)):
                    for file in files:
                        print("File: ", file)
                        if (file.lower().endswith('.wav') or file.lower().endswith('.flac')) and file + '.corrections' in files:
                            # Read the .correction (from allspecies review)
                            cfile = os.path.join(root, file + '.corrections')
                            soundfile = os.path.join(root, file)
                            try:
                                f = open(cfile, 'r')
                                annots = json.load(f)
                                f.close()
                            except Exception as e:
                                print("ERROR: file %s failed to load with error:" % file)
                                print(e)
                                return
                            for seg in annots:
                                if isinstance(seg, dict):
                                    continue
                                if len(seg) != 2: # correction not needed
                                    continue
                                oldlabel = seg[0][4]
                                # check in cases like: [kiwi] -> [kiwi, morepork]
                                # (these will be stored in .corrections, but aren't incorrect detections)
                                newsp = [lab["species"] for lab in seg[1]]
                                if len(oldlabel) != 1:
                                    # this was made manually
                                    print("Warning: ignoring labels with multiple species")
                                    continue
                                if oldlabel[0]['species'] == self.species and self.species not in newsp:
                                    # store this as "noise" calltype
                                    print("adding noise")
                                    self.traindata.append([soundfile, seg[0][:2], len(self.calltypes)])
                                    self.correction = True
                        elif (file.lower().endswith('.wav') or file.lower().endswith('.flac')) and file + '.corrections_' + self.cleanSpecies(self.species) in files:
                            # Read the .correction (from single sp review)
                            cfile = os.path.join(root, file + '.corrections_' + self.cleanSpecies(self.species))
                            soundfile = os.path.join(root, file)
                            try:
                                f = open(cfile, 'r')
                                annots = json.load(f)
                                f.close()
                            except Exception as e:
                                print("ERROR: file %s failed to load with error:" % file)
                                print(e)
                                return
                            for seg in annots:
                                if isinstance(seg, dict):
                                    continue
                                else:
                                    # store this as "noise" calltype
                                    self.traindata.append([soundfile, seg[:2], len(self.calltypes)])
                                    self.correction = True

            # Call type segments
            print('----CT data2:')
            for i in range(len(self.calltypes)):
                ctdata = self.DataGen.findCTsegments(self.folderTrain2, i)
                print(self.calltypes[i])
                for x in ctdata:
                    self.traindata.append(x)

        # How many of each class
        target = np.array([rec[-1] for rec in self.traindata])
        self.trainN = [np.sum(target == i) for i in range(len(self.calltypes) + 1)]

    def genImgDataset(self, hop):
        ''' Generate training images  for each calltype and noise'''
        for ct in range(len(self.calltypes) + 1):
            os.makedirs(os.path.join(self.tmpdir1.name, str(ct)))
        self.imgsize[1], self.Nimg = self.DataGen.generateFeatures(dirName=self.tmpdir1.name, dataset=self.traindata, hop=hop)

    def train(self):
        # Create temp dir to hold img data and model
        try:
            if self.tmpdir1:
                self.tmpdir1.cleanup()
            if self.tmpdir2:
                self.tmpdir2.cleanup()
        except:
            pass
        self.tmpdir1 = tempfile.TemporaryDirectory(prefix='NN_')
        print('Temporary img dir:', self.tmpdir1.name)
        self.tmpdir2 = tempfile.TemporaryDirectory(prefix='NN_')
        print('Temporary model dir:', self.tmpdir2.name)

        # Find train segments belong to each class
        self.DataGen = NN.GenerateData(self.currfilt, self.imgWidth, self.windowWidth, self.windowInc, self.imgsize[0], self.imgsize[1], self.f1, self.f2)

        # Find how many images with default hop (=imgWidth), adjust hop to make a good number of images also keep space
        # for some in-built augmenting (width-shift)
        hop = [self.imgWidth for i in range(len(self.calltypes)+1)]
        imgN = self.DataGen.getImgCount(dirName=self.tmpdir1.name, dataset=self.traindata, hop=hop)
        print('Expected number of images when no overlap: ', imgN)
        print('Updating hop...')
        hop = self.updateHop(imgN, hop)
        imgN = self.DataGen.getImgCount(dirName=self.tmpdir1.name, dataset=self.traindata, hop=hop)
        print('Expected number of images with updated hop: ', imgN)

        print('Generating NN images...')
        self.genImgDataset(hop)
        print('\nGenerated images:\n')
        for i in range(len(self.calltypes)):
            print("\t%s:\t%d\n" % (self.calltypes[i], self.Nimg[i]))
        print("\t%s:\t%d\n" % ("Noise", self.Nimg[-1]))

        # NN training
        nn = NN.NN(self.configdir, self.species, self.calltypes, self.fs, self.imgWidth, self.windowWidth, self.windowInc, self.imgsize[0], self.imgsize[1], self.modelArchitecture)

        # 1. Data augmentation
        print('Data augmenting...')
        filenames, labels = nn.getImglist(self.tmpdir1.name)
        labels = np.argmax(labels, axis=1)
        ns = [np.shape(np.where(labels == i)[0])[0] for i in range(len(self.calltypes) + 1)]
        # create image data augmentation generator in-build
        datagen = ImageDataGenerator(width_shift_range=0.3, fill_mode='nearest')
        # Data augmentation for each call type
        for ct in range(len(self.calltypes) + 1):
            if self.LearningDict['t'] - ns[ct] > self.LearningDict['batchsize']:
                # load this ct images
                samples = nn.loadCTImg(os.path.join(self.tmpdir1.name, str(ct)))
                # prepare iterator
                it = datagen.flow(samples, batch_size=self.LearningDict['batchsize'])
                # generate samples
                batch = it.next()
                for j in range(int((self.LearningDict['t'] - ns[ct]) / self.LearningDict['batchsize'])):
                    newbatch = it.next()
                    batch = np.vstack((batch, newbatch))
                # Save augmented data
                k = 0
                for sgRaw in batch:
                    np.save(os.path.join(self.tmpdir1.name, str(ct), str(ct) + '_aug' + "%06d" % k + '.npy'),
                            sgRaw)
                    k += 1
                try:
                    del batch
                    del samples
                    del newbatch
                except:
                    pass
                gc.collect()

        # 2. TRAIN - use custom image generator
        filenamesall, labelsall = nn.getImglist(self.tmpdir1.name)
        print('Final NN images...')
        labelsalld = np.argmax(labelsall, axis=1)
        ns = [np.shape(np.where(labelsalld == i)[0])[0] for i in range(len(self.calltypes) + 1)]
        for i in range(len(self.calltypes)):
            print("\t%s:\t%d\n" % (self.calltypes[i], ns[i]))
        print("\t%s:\t%d\n" % ("Noise", ns[-1]))

        filenamesall, labelsall = shuffle(filenamesall, labelsall)
        
        X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(filenamesall, labelsall, test_size=self.LearningDict['test_size'], random_state=1)
        training_batch_generator = NN.CustomGenerator(X_train_filenames, y_train, self.LearningDict['batchsize'], self.tmpdir1.name, nn.imageheight, nn.imagewidth, 1)
        validation_batch_generator = NN.CustomGenerator(X_val_filenames, y_val, self.LearningDict['batchsize'], self.tmpdir1.name, nn.imageheight, nn.imagewidth, 1)

        print('Creating NN architecture...')
        nn.createArchitecture()

        print('Training...')
        nn.train(modelsavepath=self.tmpdir2.name, training_batch_generator=training_batch_generator, validation_batch_generator=validation_batch_generator)
        print('Training complete!')

        self.bestThr = [[0, 0] for i in range(len(self.calltypes))]
        self.bestThrInd = [0 for i in range(len(self.calltypes))]

        # 3. Prepare ROC plots
        print('Generating ROC statistics...')
        # Load the model
        # Find best weights
        weights = []
        epoch = []
        for r, d, files in os.walk(self.tmpdir2.name):
            for f in files:
                if f.endswith('.h5') and 'weights' in f:
                    epoch.append(int(f.split('weights.')[-1][:2]))
                    weights.append(f)
            j = np.argmax(epoch)
            weightfile = weights[j]
        model = os.path.join(self.tmpdir2.name, 'model.json')
        self.bestweight = os.path.join(self.tmpdir2.name, weightfile)
        # Load the model and prepare
        jsonfile = open(model, 'r')
        loadedModelJson = jsonfile.read()
        jsonfile.close()
        with custom_object_scope(NNModels.customObjectScopes):
            try:
                model = model_from_json(loadedModelJson)
            except:
                print('Error in loading model from json. Are you linking all custom layers in NNModels.customObjectScopes?')
                return False
        # Load weights into new model
        model.load_weights(self.bestweight)
        # Compile the model
        model.compile(loss=self.LearningDict['loss'], optimizer=self.LearningDict['optimizer'],
                      metrics=self.LearningDict['metrics'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('Loaded NN model from ', self.tmpdir2.name)

        TPs = [0 for i in range(len(self.calltypes) + 1)]
        FPs = [0 for i in range(len(self.calltypes) + 1)]
        TNs = [0 for i in range(len(self.calltypes) + 1)]
        FNs = [0 for i in range(len(self.calltypes) + 1)]
        CTps = [[[] for i in range(len(self.calltypes) + 1)] for j in range(len(self.calltypes) + 1)]
        # Do all the plots based on Validation set (eliminate augmented?)
        # N = len(filenames)
        N = len(X_val_filenames)
        y_val = np.argmax(y_val, axis=1)
        print('Validation data: ', N)
        if os.path.isdir(self.tmpdir2.name):
            print('Model directory exists')
        else:
            print('Model directory DOES NOT exist')
        if os.path.isdir(self.tmpdir1.name):
            print('Img directory exists')
        else:
            print('Img directory DOES NOT exist')
        
        for i in range(int(np.ceil(N / self.LearningDict['batchsize_ROC']))):
            # imagesb = nn.loadImgBatch(filenames[i * self.LearningDict['batchsize_ROC']:min((i + 1) * self.LearningDict['batchsize_ROC'], N)])
            # labelsb = labels[i * self.LearningDict['batchsize_ROC']:min((i + 1) * self.LearningDict['batchsize_ROC'], N)]
            imagesb = nn.loadImgBatch(X_val_filenames[i * self.LearningDict['batchsize_ROC']:min((i + 1) * self.LearningDict['batchsize_ROC'], N)])
            labelsb = y_val[i * self.LearningDict['batchsize_ROC']:min((i + 1) * self.LearningDict['batchsize_ROC'], N)]
            for ct in range(len(self.calltypes) + 1):
                res, ctp = self.testCT(ct, imagesb, labelsb, model)  # res=[thrlist, TPs, FPs, TNs, FNs], ctp=[[0to0 probs], [0to1 probs], [0to2 probs]]
                for j in range(len(self.calltypes) + 1):
                    CTps[ct][j] += ctp[j]
                if TPs[ct] == 0:
                    TPs[ct] = res[1]
                    FPs[ct] = res[2]
                    TNs[ct] = res[3]
                    FNs[ct] = res[4]
                else:
                    TPs[ct] = [TPs[ct][i] + res[1][i] for i in range(len(TPs[ct]))]
                    FPs[ct] = [FPs[ct][i] + res[2][i] for i in range(len(FPs[ct]))]
                    TNs[ct] = [TNs[ct][i] + res[3][i] for i in range(len(TNs[ct]))]
                    FNs[ct] = [FNs[ct][i] + res[4][i] for i in range(len(FNs[ct]))]
        self.Thrs = res[0]
        print('Thrs: ', self.Thrs)
        print('validation TPs[0]: ', TPs[0])

        self.TPRs = [[0.0 for i in range(len(self.Thrs))] for j in range(len(self.calltypes) + 1)]
        self.FPRs = [[0.0 for i in range(len(self.Thrs))] for j in range(len(self.calltypes) + 1)]
        self.Precisions = [[0.0 for i in range(len(self.Thrs))] for j in range(len(self.calltypes) + 1)]
        self.Accs = [[0.0 for i in range(len(self.Thrs))] for j in range(len(self.calltypes) + 1)]

        plt.style.use('ggplot')
        fig, axs = plt.subplots(len(self.calltypes) + 1, len(self.calltypes) + 1, sharey=True, sharex='col')

        for ct in range(len(self.calltypes) + 1):
            self.TPRs[ct] = [TPs[ct][i] / (TPs[ct][i] + FNs[ct][i]) for i in range(len(self.Thrs))]
            self.FPRs[ct] = [FPs[ct][i] / (TNs[ct][i] + FPs[ct][i]) for i in range(len(self.Thrs))]
            self.Precisions[ct] = [0.0 if (TPs[ct][i] + FPs[ct][i]) == 0 else TPs[ct][i] / (TPs[ct][i] + FPs[ct][i]) for i in range(len(self.Thrs))]
            self.Accs[ct] = [(TPs[ct][i] + TNs[ct][i]) / (TPs[ct][i] + TNs[ct][i] + FPs[ct][i] + FNs[ct][i]) for
                             i in range(len(self.Thrs))]

            # Temp plot is saved in train data directory - prediction probabilities for instances of current ct
            for i in range(len(self.calltypes) + 1):
                CTps[ct][i] = sorted(CTps[ct][i], key=float)
                axs[i, ct].plot(CTps[ct][i], 'k')
                axs[i, ct].plot(CTps[ct][i], 'bo')
                if ct == i == len(self.calltypes):
                    axs[i, 0].set_ylabel('Noise')
                    axs[0, ct].set_title('Noise')
                elif ct == i:
                    axs[i, 0].set_ylabel(str(self.calltypes[ct]))
                    axs[0, ct].set_title(str(self.calltypes[ct]))
                if i == len(self.calltypes):
                    axs[i, ct].set_xlabel('Number of samples')
        fig.suptitle('Human')
        if self.folderTrain1:
            fig.savefig(os.path.join(self.folderTrain1, 'validation-plots.png'))
            print('Validation plot is saved: ', os.path.join(self.folderTrain1, 'validation-plots.png'))
        else:
            fig.savefig(os.path.join(self.folderTrain2, 'validation-plots.png'))
            print('Validation plot is saved: ', os.path.join(self.folderTrain2, 'validation-plots.png'))
        plt.close()

        # Collate ROC daaa
        self.ROCdata["TPR"] = self.TPRs
        self.ROCdata["FPR"] = self.FPRs
        self.ROCdata["thr"] = self.Thrs
        print('TPR: ', self.ROCdata["TPR"])
        print('FPR: ', self.ROCdata["FPR"])

        # 4. Auto select the upper threshold (fpr = 0)
        for ct in range(len(self.calltypes)):
            try:
                self.bestThr[ct][1] = self.Thrs[self.FPRs[ct].index(0.0)]
            except:
                self.bestThr[ct][1] = self.Thrs[len(self.FPRs[ct]) - 1]

        # 5. Auto select lower threshold IF the user asked so
        if self.autoThr:
            for ct in range(len(self.calltypes)):
                # Get min distance to ROC from (0 FPR, 1 TPR)
                distarr = (np.float64(1) - self.TPRs[ct]) ** 2 + (np.float64(0) - self.FPRs[ct]) ** 2
                self.thr_min_ind = np.unravel_index(np.argmin(distarr), distarr.shape)[0]
                self.bestThr[ct][0] = self.Thrs[self.thr_min_ind]
                self.bestThrInd[ct] = self.thr_min_ind
        return True

    def updateHop(self, imgN, hop):
        ''' Update hop'''
        # Compare against the expected number of total images per class (t)
        for i in range(len(self.calltypes) + 1):
            fillratio1 = imgN[i] / (self.LearningDict['t'] - self.LearningDict['tWidthShift'])
            fillratio2 = imgN[i] / self.LearningDict['t']
            if fillratio1 < 0.75:   # too less, decrease hop
                if i == len(self.calltypes):
                    print('Noise: only %d images, adjusting hop from %.2f to %.2f' % (imgN[i], hop[i], hop[i]*fillratio1))
                else:
                    print('%s: only %d images, adjusting hop from %.2f to %.2f' % (self.calltypes[i], imgN[i], hop[i], hop[i]*fillratio1))
                hop[i] = hop[i]*fillratio1
            elif fillratio1 > 1 and fillratio2 > 0.75:  # increase hop and make room for augmenting
                if i == len(self.calltypes):
                    print('Noise: %d images, adjusting hop from %.2f to %.2f' % (imgN[i], hop[i], hop[i]*fillratio1))
                else:
                    print('%s: %d images, adjusting hop from %.2f to %.2f' % (self.calltypes[i], imgN[i], hop[i], hop[i]*fillratio1))
                hop[i] = hop[i]*fillratio1
            elif fillratio2 > 1:    # too many, avoid hop
                if i == len(self.calltypes):
                    print('Noise: %d images, adjusting hop from %.2f to %.2f' % (imgN[i], hop[i], hop[i]*fillratio2))
                else:
                    print('%s: %d images, adjusting hop from %.2f to %.2f' % (self.calltypes[i], imgN[i], hop[i], hop[i]*fillratio2))
                hop[i] = hop[i]*fillratio2
        return hop

    def testCT(self, ct, testimages, targets, model):
        '''
        :param ct: integer relevant to call type
        :return: [thrlist, TPs, FPs, TNs, FNs], ctprob
        '''

        self.thrs = []
        self.TPs = []
        self.FPs = []
        self.TNs = []
        self.FNs = []

        # Predict and temp plot
        pre = model.predict(testimages)
        ctprob = [[] for i in range(len(self.calltypes) + 1)]
        for i in range(len(targets)):
            if targets[i] == ct:
                for ind in range(len(self.calltypes) + 1):
                    ctprob[ind].append(pre[i][ind])

        # Get the stats over different thr
        labels = [i for i in range(len(self.calltypes) + 1)]
        for thr in np.linspace(0.00001, 1, 100):
            predictions = [self.pred(p, thr=thr, ct=ct) for p in pre]
            CM = confusion_matrix(predictions, targets, labels=labels)
            TP = CM[ct][ct]
            FP = np.sum(CM[ct][:]) - TP
            colct = 0
            for i in range(len(self.calltypes) + 1):
                colct += CM[i][ct]
            FN = colct - TP
            TN = np.sum(CM) - FP - FN - TP

            self.thrs.append(thr)
            self.TPs.append(TP)
            self.FPs.append(FP)
            self.TNs.append(TN)
            self.FNs.append(FN)

        return [self.thrs, self.TPs, self.FPs, self.TNs, self.FNs], ctprob

    def pred(self, p, thr, ct):
        if p[ct] > thr:
            prediction = ct
        elif ct == len(self.calltypes):
            prediction = 0
        else:
            prediction = len(self.calltypes)
        return prediction

    def saveFilter(self):
        # Add NN component to the current filter
        self.addNNFilter()
        # NNdic = {}
        # NNdic["NN_name"] = "NN_name"
        # NNdic["loss"] = self.LearningDict['loss']
        # NNdic["optimizer"] = self.LearningDict['optimizer']
        # NNdic["windowInc"] = [self.windowWidth,self.windowInc]
        # NNdic["win"] = [self.imgWidth,self.imgWidth/5]     # TODO: remove hop
        # NNdic["inputdim"] = self.imgsize
        # output = {}
        # thr = []
        # for ct in range(len(self.calltypes)):
        #     output[str(ct)] = self.calltypes[ct]
        #     thr.append(self.bestThr[ct])
        # output[str(len(self.calltypes))] = "Noise"
        # # thr.append(self.wizard().parameterPage.bestThr[len(self.calltypes)])
        # NNdic["output"] = output
        # NNdic["thr"] = thr
        # print(NNdic)
        # self.currfilt["NN"] = NNdic

        if self.CLI:
            # write out the filter and NN model
            modelsrc = os.path.join(self.tmpdir2.name, 'model.json')
            NN_name = self.species + strftime("_%H-%M-%S", gmtime())
            self.currfilt["NN"]["NN_name"] = NN_name
            rocfilename = self.species + "_RONN" + strftime("_%H-%M-%S", gmtime())
            self.currfilt["RONN"] = rocfilename
            rocfilename = os.path.join(self.filterdir, rocfilename + '.json')

            modelfile = os.path.join(self.filterdir, NN_name + '.json')
            weightsrc = self.bestweight
            weightfile = os.path.join(self.filterdir, NN_name + '.h5')

            filename = os.path.join(self.filterdir, self.filterName)
            if not filename.lower().endswith('.txt'):
                filename = filename + '.txt'
            print("Updating the existing recogniser ", filename)
            f = open(filename, 'w')
            f.write(json.dumps(self.currfilt))
            f.close()
            # Actually copy the model
            copyfile(modelsrc, modelfile)
            copyfile(weightsrc, weightfile)
            # save ROC
            f = open(rocfilename, 'w')
            f.write(json.dumps(self.ROCdata))
            f.close()
            # And remove temp dirs
            self.tmpdir1.cleanup()
            self.tmpdir2.cleanup()
            print("Recogniser saved, don't forget to test it!")

    def addNNFilter(self):
        # Add NN component to the current filter
        NNdic = {}
        NNdic["NN_name"] = "NN_name"
        NNdic["loss"] = self.LearningDict['loss']
        NNdic["optimizer"] = self.LearningDict['optimizer']
        NNdic["windowInc"] = [self.windowWidth,self.windowInc]
        NNdic["win"] = [self.imgWidth,self.imgWidth/5]     # TODO: remove hop
        NNdic["inputdim"] = [int(self.imgsize[0]), int(self.imgsize[1])]
        if self.f1 == 0 and self.f2 == self.fs/2:
            print('no frequency masking used')
        else:
            print('frequency masking used', self.f1, self.f2)
            NNdic["fRange"] = [int(self.f1), int(self.f2)]
        output = {}
        thr = []
        for ct in range(len(self.calltypes)):
            output[str(ct)] = self.calltypes[ct]
            thr.append(self.bestThr[ct])
        output[str(len(self.calltypes))] = "Noise"
        # thr.append(self.wizard().parameterPage.bestThr[len(self.calltypes)])
        NNdic["output"] = output
        NNdic["thr"] = thr
        print(NNdic)
        self.currfilt["NN"] = NNdic

    def cleanSpecies(self, species):
        """ Returns cleaned species name"""
        return re.sub(r'[^A-Za-z0-9()-]', "_", species)


class NNtest:
    # Test a previously-trained NN

    def __init__(self,testDir,currfilt,filtname,configdir,filterdir,CLI=False):
        """ currfilt: the recognizer to be used (dict) """
        self.testDir = testDir
        self.outfile = open(os.path.join(self.testDir, "test-results.txt"),"w")

        self.currfilt = currfilt
        self.filtname = filtname

        self.configdir = configdir
        self.filterdir = filterdir
        # Note: this is just the species name, unlike the self.species in Batch mode
        species = self.currfilt['species']
        self.sampleRate = self.currfilt['SampleRate']
        self.calltypes = []
        for fi in self.currfilt['Filters']:
            self.calltypes.append(fi['calltype'])

        self.outfile.write("Recogniser name: %s\n" %(filtname))
        self.outfile.write("Species name: %s\n" % (species))
        self.outfile.write("Using data: %s\n" % (self.testDir))

        # 0. Generate GT files from annotations in test folder
        self.manSegNum = 0
        self.window = 1
        inc = None
        print('Generating GT...')
        for root, dirs, files in os.walk(self.testDir):
            for file in files:
                soundFile = os.path.join(root, file)
                if (file.lower().endswith('.wav') or file.lower().endswith('.flac')) and os.stat(soundFile).st_size != 0 and file + '.data' in files:
                    segments = Segment.SegmentList()
                    segments.parseJSON(soundFile + '.data')
                    self.manSegNum += len(segments.getSpecies(species))
                    # Currently, we ignore call types here and just
                    # look for all calls for the target species.
                    segments.exportGT(soundFile, species, resolution=self.window)

        if self.manSegNum == 0:
            print("ERROR: no segments for species %s found" % species)
            self.text = 0
            return

        # 1. Run Batch Processing upto WF and generate .tempdata files (no post-proc)
        avianz_batch = AviaNZ_batch.AviaNZ_batchProcess(parent=None, configdir=self.configdir, mode="test", sdir=self.testDir, recognisers=filtname, wind="None")
        # NOTE: will use wind-robust detection

        # 2. Report statistics of WF followed by general post-proc steps (no NN but wind-merge neighbours-delete short)
        self.text = self.getSummary(NN=False)

        # 3. Report statistics of WF followed by post-proc steps (wind-NN-merge neighbours-delete short)
        if "NN" in self.currfilt:
            cl = SupportClasses.ConfigLoader()
            filterlist = cl.filters(self.filterdir, bats=False)
            NNDicts = cl.getNNmodels(filterlist, self.filterdir, [filtname])
            #if len(NNDicts.keys()) == 1:
            # SM: NN testing used wrong name
            #if filtname in NNDicts.keys():
                #for filtname in NNDicts.keys():
                    #NNmodel = NNDicts[filtname]
                    #self.text = self.getSummary(NN=True)
            #else:
                #print("ERROR: Couldn't find a matching NN!")
            # Providing one filter, so only one NN should be returned:
            if len(NNDicts)!=1:
                print("ERROR: Couldn't find a unique matching NN!")
                self.outfile.write("No matching NN found!\n")
                self.outfile.write("-- End of testing --\n")
                self.outfile.close()
                return
            NNmodel = list(NNDicts)[0]
            self.text = self.getSummary(NN=True)
        self.outfile.write("-- End of testing --\n")
        self.outfile.close()

        print("Testing output written to " + os.path.join(self.testDir, "test-results.txt"))

        # Tidy up
        # for root, dirs, files in os.walk(self.testDir):
        #     for file in files:
        #         if file.endswith('.tmpdata'):
        #             os.remove(os.path.join(root, file))

    def getOutput(self):
        return self.text

    def findCTsegments(self, datafile, calltypei):
        calltypeSegments = []
        species = self.currfilt["species"]
        segments = Segment.SegmentList()
        segments.parseJSON(datafile)
        if len(self.calltypes) == 1:
            ctSegments = segments.getSpecies(species)
        else:
            ctSegments = segments.getCalltype(species, self.calltypes[calltypei])
        calltypeSegments = [segments[indx][:2] for indx in ctSegments]

        return calltypeSegments

    def getSummary(self, NN=False):
        autoSegCTnum = [0] * len(self.calltypes)
        ws = WaveletSegment.WaveletSegment()
        TP = FP = TN = FN = 0
        for root, dirs, files in os.walk(self.testDir):
            for file in files:
                soundFile = os.path.join(root, file)
                filenameNoExtension = file.rsplit('.', 1)[0]
                if (file.lower().endswith('.wav') or file.lower().endswith('.flac')) and os.stat(soundFile).st_size != 0 and \
                        file + '.tmpdata' in files and filenameNoExtension + '-GT.txt' in files:
                    # Extract all segments and back-convert to 0/1:
                    if file.lower().endswith('.wav'):
                        info = sf.info(soundFile)
                        duration = info.frames / info.samplerate
                    else:
                        info = sf.info(soundFile)
                        duration = info.frames / info.samplerate

                    duration = math.ceil(duration)
                    det01 = np.zeros(duration)

                    for i in range(len(self.calltypes)):
                        if NN:
                            # read segments
                            ctsegments = self.findCTsegments(soundFile+'.tmpdata', i)
                        else:
                            # read segments from an identical postproc pipeline w/o NN
                            ctsegments = self.findCTsegments(soundFile+'.tmp2data', i)
                        autoSegCTnum[i] += len(ctsegments)

                        for seg in ctsegments:
                            det01[math.floor(seg[0]):math.ceil(seg[1])] = 1

                    # get and parse the agreement metrics
                    GT = self.loadGT(os.path.join(root, filenameNoExtension + '-GT.txt'), duration)
                    _, _, tp, fp, tn, fn = ws.fBetaScore(GT, det01)
                    TP += tp
                    FP += fp
                    TN += tn
                    FN += fn
        # Summary
        total = TP + FP + TN + FN
        if total == 0:
            print("ERROR: failed to find any testing data")
            return

        if TP + FN != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0
        if TP + FP != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0
        if TN + FP != 0:
            specificity = TN / (TN + FP)
        else:
            specificity = 0
        accuracy = (TP + TN) / (TP + FP + TN + FN)

        if NN:
            self.outfile.write("\n\n-- Wavelet Pre-Processor + NN detection summary --\n")
        else:
            self.outfile.write("\n-- Wavelet Pre-Processor detection summary --\n")
        self.outfile.write("TP | FP | TN | FN seconds:\t %.2f | %.2f | %.2f | %.2f\n" % (TP, FP, TN, FN))
        self.outfile.write("Specificity:\t\t%.2f %%\n" % (specificity * 100))
        self.outfile.write("Recall (sensitivity):\t%.2f %%\n" % (recall * 100))
        self.outfile.write("Precision (PPV):\t%.2f %%\n" % (precision * 100))
        self.outfile.write("Accuracy:\t\t%.2f %%\n\n" % (accuracy * 100))
        self.outfile.write("Manually labelled segments:\t%d\n" % (self.manSegNum))
        for i in range(len(self.calltypes)):
            self.outfile.write("Auto suggested \'%s\' segments:\t%d\n" % (self.calltypes[i], autoSegCTnum[i]))
        self.outfile.write("Total auto suggested segments:\t%d\n\n" % sum(autoSegCTnum))

        if NN:
            text = "Wavelet Pre-Processor + NN detection summary\n\n\tTrue Positives:\t%d seconds (%.2f %%)\n\tFalse Positives:\t%d seconds (%.2f %%)\n\tTrue Negatives:\t%d seconds (%.2f %%)\n\tFalse Negatives:\t%d seconds (%.2f %%)\n\n\tSpecificity:\t%.2f %%\n\tRecall:\t\t%.2f %%\n\tPrecision:\t%.2f %%\n\tAccuracy:\t%.2f %%\n" \
                   % (TP, TP * 100 / total, FP, FP * 100 / total, TN, TN * 100 / total, FN, FN * 100 / total,
                      specificity * 100, recall * 100, precision * 100, accuracy * 100)
        else:
            text = "Wavelet Pre-Processor detection summary\n\n\tTrue Positives:\t%d seconds (%.2f %%)\n\tFalse Positives:\t%d seconds (%.2f %%)\n\tTrue Negatives:\t%d seconds (%.2f %%)\n\tFalse Negatives:\t%d seconds (%.2f %%)\n\n\tSpecificity:\t%.2f %%\n\tRecall:\t\t%.2f %%\n\tPrecision:\t%.2f %%\n\tAccuracy:\t%.2f %%\n" \
                   % (TP, TP * 100 / total, FP, FP * 100 / total, TN, TN * 100 / total, FN, FN * 100 / total,
                      specificity * 100, recall * 100, precision * 100, accuracy * 100)
        return text

    def loadGT(self, filename, length):
        import csv
        annotation = []
        # Get the segmentation from the txt file
        with open(filename) as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)
        if d[-1] == []:
            d = d[:-1]
        if len(d) != length:
            print("ERROR: annotation length %d does not match file duration %d!" % (len(d), length))
            self.annotation = []
            return False

        # for each second, store 0/1 presence:
        for row in d:
            annotation.append(int(row[1]))

        return annotation

