
# This is part of the AviaNZ interface
# Holds most of the code for training CNNs

# Version 3.0 14/09/20
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2020

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

# The separated code for CNN training

import os, gc, re, json, tempfile
from shutil import copyfile
from shutil import disk_usage

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
import matplotlib.pyplot as plt
from time import strftime, gmtime

import SupportClasses
import SignalProc
import CNN
import Segment, WaveletSegment
import AviaNZ_batch

class CNNtrain:

    def __init__(self, configdir, filterdir, folderTrain1=None, folderTrain2=None, recogniser=None, imgWidth=0, CLI=False):

        self.filterdir = filterdir
        self.configdir =configdir
        cl = SupportClasses.ConfigLoader()
        self.FilterDict = cl.filters(filterdir, bats=False)
        self.LearningDict = cl.learningParams(os.path.join(configdir, "LearningParams.txt"))
        self.sp = SignalProc.SignalProc(self.LearningDict['sgramWindowWidth'], self.LearningDict['sgramHop'])

        self.imgsize = [self.LearningDict['imgX'], self.LearningDict['imgY']]
        self.tmpdir1 = False
        self.tmpdir2 = False

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

    def setP1(self, folderTrain1, folderTrain2, recogniser, annotationLevel):
        self.folderTrain1 = folderTrain1
        self.folderTrain2 = folderTrain2
        self.filterName = recogniser
        self.annotatedAll = annotationLevel

    def setP6(self, recogniser):
        self.newFilterName = recogniser
            
    def cliTrain(self):
        # This proceeds very much like the wizard 
            
        self.readFilter()
        # Load data
        # Note: no error checking in the CLI version
        # Find segments belong to each class in the training data
        self.genSegmentDataset(hasAnnotation=True)

        self.checkDisk()
        self.windowWidth = self.imgsize[0] * self.LearningDict['windowScaling']
        self.windowInc = int(np.ceil(self.imgWidth * self.fs / (self.imgsize[1] - 1)) )

        # Train
        self.train()

        self.saveFilter()

    def readFilter(self):
        if self.filterName.lower().endswith('.txt'):
            self.currfilt = self.FilterDict[self.filterName[:-4]]
        else:
            self.currfilt = self.FilterDict[self.filterName]

        self.fs = self.currfilt["SampleRate"]
        self.species = self.currfilt["species"]

        mincallengths = []
        maxcallengths = []
        self.maxgaps = []
        self.calltypes = []
        for fi in self.currfilt['Filters']:
            self.calltypes.append(fi['calltype'])
            mincallengths.append(fi['TimeRange'][0])
            maxcallengths.append(fi['TimeRange'][1])
            self.maxgaps.append(fi['TimeRange'][3])
        self.mincallength = np.max(mincallengths)
        self.maxcallength = np.max(maxcallengths)

        print("Manually annotated: %s" % self.folderTrain1)
        print("Auto processed and reviewed: %s" % self.folderTrain2)
        print("Recogniser: %s" % self.currfilt)
        print("Species: %s" % self.species)
        print("Call types: %s" % self.calltypes)
        print("Call length: %.2f - %.2f sec" % (self.mincallength, self.maxcallength))
        print("Sample rate: %d Hz" % self.fs)

    def checkDisk(self):
        # Check disk usage
        totalbytes, usedbytes, freebytes = disk_usage(os.path.expanduser("~"))
        freeGB = freebytes/1024/1024/1024
        print('\nFree space in the user directory: %.2f GB/ %.2f GB\n' % (freeGB, totalbytes/1024/1024/2014))
        if freeGB < 10:
            print('Warning: You may run out of space in the user directory!')
        return freeGB, totalbytes/1024/1024/1024

    def genSegmentDataset(self, hasAnnotation):
        self.traindata = []
        self.DataGen = CNN.GenerateData(self.currfilt, 0, 0, 0, 0, 0)
        # Dir1 - manually annotated
        # Find noise segments if the user is confident about full annotation
        if self.annotatedAll:
            self.noisedata1 = self.DataGen.findNoisesegments(self.folderTrain1)
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

        # Dir2 - auto reviewed
        # Get noise segments from .corrections
        if os.path.isdir(self.folderTrain2):
            for root, dirs, files in os.walk(str(self.folderTrain2)):
                for file in files:
                    if file.lower().endswith('.wav') and file + '.corrections' in files:
                        # Read the .correction (from allspecies review)
                        cfile = os.path.join(root, file + '.corrections')
                        wavfile = os.path.join(root, file)
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
                            if len(seg) != 2:
                                print("Warning: old format corrections detected")
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
                                self.traindata.append([wavfile, seg[0][:2], len(self.calltypes)])
                                self.correction = True
                    elif file.lower().endswith('.wav') and file + '.corrections_' + self.cleanSpecies(self.species) in files:
                        # Read the .correction (from single sp review)
                        cfile = os.path.join(root, file + '.corrections_' + self.cleanSpecies(self.species))
                        wavfile = os.path.join(root, file)
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
                                self.traindata.append([wavfile, seg[:2], len(self.calltypes)])
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
        ''' Generate training images'''
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
        self.tmpdir1 = tempfile.TemporaryDirectory(prefix='CNN_')
        # self.tmpdir1 = tempfile.TemporaryDirectory(prefix='CNN_', dir="/local/tmp/juodakjuli/cnntmp")
        print('Temporary img dir:', self.tmpdir1.name)
        self.tmpdir2 = tempfile.TemporaryDirectory(prefix='CNN_')
        # self.tmpdir2 = tempfile.TemporaryDirectory(prefix='CNN_', dir="/local/tmp/juodakjuli/cnntmp")
        print('Temporary model dir:', self.tmpdir2.name)

        # Find train segments belong to each class
        self.DataGen = CNN.GenerateData(self.currfilt, self.imgWidth, self.windowWidth, self.windowInc, self.imgsize[0], self.imgsize[1])

        # Find how many images with default hop (=imgWidth), adjust hop to make a good number of images also keep space
        # for some in-built augmenting (width-shift)
        hop = [self.imgWidth for i in range(len(self.calltypes)+1)]
        imgN = self.DataGen.getImgCount(dirName=self.tmpdir1.name, dataset=self.traindata, hop=hop)
        print('Expected number of images when no overlap: ', imgN)
        print('Updating hop...')
        hop = self.updateHop(imgN, hop)
        imgN = self.DataGen.getImgCount(dirName=self.tmpdir1.name, dataset=self.traindata, hop=hop)
        print('Expected number of images with updated hop: ', imgN)

        print('Generating CNN images...')
        self.genImgDataset(hop)
        print('\nGenerated images:\n')
        for i in range(len(self.calltypes)):
            print("\t%s:\t%d\n" % (self.calltypes[i], self.Nimg[i]))
        print("\t%s:\t%d\n" % ("Noise", self.Nimg[-1]))

        # CNN training
        cnn = CNN.CNN(self.configdir, self.species, self.calltypes, self.fs, self.imgWidth, self.windowWidth, self.windowInc, self.imgsize[0], self.imgsize[1])

        # 1. Data augmentation
        print('Data augmenting...')
        filenames, labels = cnn.getImglist(self.tmpdir1.name)
        labels = np.argmax(labels, axis=1)
        ns = [np.shape(np.where(labels == i)[0])[0] for i in range(len(self.calltypes) + 1)]
        # create image data augmentation generator in-build
        datagen = ImageDataGenerator(width_shift_range=0.3, fill_mode='nearest')
        # Data augmentation for each call type
        for ct in range(len(self.calltypes) + 1):
            if self.LearningDict['t'] - ns[ct] > self.LearningDict['batchsize']:
                # load this ct images
                samples = cnn.loadCTImg(os.path.join(self.tmpdir1.name, str(ct)))
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
        filenamesall, labelsall = cnn.getImglist(self.tmpdir1.name)
        print('Final CNN images...')
        labelsalld = np.argmax(labelsall, axis=1)
        ns = [np.shape(np.where(labelsalld == i)[0])[0] for i in range(len(self.calltypes) + 1)]
        for i in range(len(self.calltypes)):
            print("\t%s:\t%d\n" % (self.calltypes[i], ns[i]))
        print("\t%s:\t%d\n" % ("Noise", ns[-1]))

        filenamesall, labelsall = shuffle(filenamesall, labelsall)
        
        X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(filenamesall, labelsall, test_size=self.LearningDict['test_size'], random_state=1)
        training_batch_generator = CNN.CustomGenerator(X_train_filenames, y_train, self.LearningDict['batchsize'], self.tmpdir1.name, cnn.imageheight, cnn.imagewidth, 1)
        validation_batch_generator = CNN.CustomGenerator(X_val_filenames, y_val, self.LearningDict['batchsize'], self.tmpdir1.name, cnn.imageheight, cnn.imagewidth, 1)

        print('Creating CNN architecture...')
        cnn.createArchitecture()

        print('Training...')
        cnn.train(modelsavepath=self.tmpdir2.name, training_batch_generator=training_batch_generator, validation_batch_generator=validation_batch_generator)
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
        loadedmodeljson = jsonfile.read()
        jsonfile.close()
        model = model_from_json(loadedmodeljson)
        # Load weights into new model
        model.load_weights(self.bestweight)
        # Compile the model
        model.compile(loss=self.LearningDict['loss'], optimizer=self.LearningDict['optimizer'],
                      metrics=self.LearningDict['metrics'])
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        TPs = [0 for i in range(len(self.calltypes) + 1)]
        FPs = [0 for i in range(len(self.calltypes) + 1)]
        TNs = [0 for i in range(len(self.calltypes) + 1)]
        FNs = [0 for i in range(len(self.calltypes) + 1)]
        CTps = [[[] for i in range(len(self.calltypes) + 1)] for j in range(len(self.calltypes) + 1)]
        # Do all the plots based on Validation set (eliminate augmented?)
        # N = len(filenames)
        N = len(X_val_filenames)
        y_val = np.argmax(y_val, axis=1)
        
        for i in range(int(np.ceil(N / self.LearningDict['batchsize_ROC']))):
            # imagesb = cnn.loadImgBatch(filenames[i * self.LearningDict['batchsize_ROC']:min((i + 1) * self.LearningDict['batchsize_ROC'], N)])
            # labelsb = labels[i * self.LearningDict['batchsize_ROC']:min((i + 1) * self.LearningDict['batchsize_ROC'], N)]
            imagesb = cnn.loadImgBatch(X_val_filenames[i * self.LearningDict['batchsize_ROC']:min((i + 1) * self.LearningDict['batchsize_ROC'], N)])
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
        else:
            fig.savefig(os.path.join(self.folderTrain2, 'validation-plots.png'))

                # # Individual plots
                # fig = plt.figure()
                # ax = plt.axes()
                # ax.plot(CTps[ct][i], 'k')
                # ax.plot(CTps[ct][i], 'bo')
                # plt.xlabel('Number of samples')
                # plt.ylabel('Probability')
                # if ct == len(self.calltypes):
                #     plt.title('Class: Noise')
                # else:
                #     plt.title('Class: ' + str(self.calltypes[ct]))
                # if self.folderTrain1:
                #     fig.savefig(os.path.join(self.folderTrain1, str(ct) + '-' + str(i) + '.png'))
                # else:
                #     fig.savefig(os.path.join(self.folderTrain2, str(ct) + '-' + str(i) + '.png'))
                # plt.close()

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

        # Predict and temp plot (just for me)
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
        # Add CNN component to the current filter
        self.addCNNFilter()
        # CNNdic = {}
        # CNNdic["CNN_name"] = "CNN_name"
        # CNNdic["loss"] = self.LearningDict['loss']
        # CNNdic["optimizer"] = self.LearningDict['optimizer']
        # CNNdic["windowInc"] = [self.windowWidth,self.windowInc]
        # CNNdic["win"] = [self.imgWidth,self.imgWidth/5]     # TODO: remove hop
        # CNNdic["inputdim"] = self.imgsize
        # output = {}
        # thr = []
        # for ct in range(len(self.calltypes)):
        #     output[str(ct)] = self.calltypes[ct]
        #     thr.append(self.bestThr[ct])
        # output[str(len(self.calltypes))] = "Noise"
        # # thr.append(self.wizard().parameterPage.bestThr[len(self.calltypes)])
        # CNNdic["output"] = output
        # CNNdic["thr"] = thr
        # print(CNNdic)
        # self.currfilt["CNN"] = CNNdic

        if self.CLI:
            # write out the filter and CNN model
            modelsrc = os.path.join(self.tmpdir2.name, 'model.json')
            CNN_name = self.species + strftime("_%H-%M-%S", gmtime())
            self.currfilt["CNN"]["CNN_name"] = CNN_name

            modelfile = os.path.join(self.filterdir, CNN_name + '.json')
            weightsrc = self.bestweight
            weightfile = os.path.join(self.filterdir, CNN_name + '.h5')

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
            # And remove temp dirs
            self.tmpdir1.cleanup()
            self.tmpdir2.cleanup()
            print("Recogniser saved, don't forget to test it!")

    def addCNNFilter(self):
        # Add CNN component to the current filter
        CNNdic = {}
        CNNdic["CNN_name"] = "CNN_name"
        CNNdic["loss"] = self.LearningDict['loss']
        CNNdic["optimizer"] = self.LearningDict['optimizer']
        CNNdic["windowInc"] = [self.windowWidth,self.windowInc]
        CNNdic["win"] = [self.imgWidth,self.imgWidth/5]     # TODO: remove hop
        CNNdic["inputdim"] = self.imgsize
        output = {}
        thr = []
        for ct in range(len(self.calltypes)):
            output[str(ct)] = self.calltypes[ct]
            thr.append(self.bestThr[ct])
        output[str(len(self.calltypes))] = "Noise"
        # thr.append(self.wizard().parameterPage.bestThr[len(self.calltypes)])
        CNNdic["output"] = output
        CNNdic["thr"] = thr
        print(CNNdic)
        self.currfilt["CNN"] = CNNdic

    def cleanSpecies(self, species):
        """ Returns cleaned species name"""
        return re.sub(r'[^A-Za-z0-9()-]', "_", species)


class CNNtest:

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
                wavFile = os.path.join(root, file)
                if file.lower().endswith('.wav') and os.stat(wavFile).st_size != 0 and file + '.data' in files:
                    segments = Segment.SegmentList()
                    segments.parseJSON(wavFile + '.data')
                    self.manSegNum += len(segments.getSpecies(species))
                    # Currently, we ignore call types here and just
                    # look for all calls for the target species.
                    segments.exportGT(wavFile, species, window=self.window, inc=inc)

        if self.manSegNum == 0:
            print("ERROR: no segments for species %s found" % species)
            self.text = 0
            return

        # 1. Run Batch Processing upto WF and generate .tempdata files (no post-proc)
        avianz_batch = AviaNZ_batch.AviaNZ_batchProcess(parent=None, configdir=self.configdir, mode="test",
                                                        sdir=self.testDir, recogniser=filtname, wind=True)

        # 2. Report statistics of WF followed by general post-proc steps (no CNN but wind-merge neighbours-delete short)
        self.text = self.getSummary(avianz_batch, CNN=False)

        # 3. Report statistics of WF followed by post-proc steps (wind-CNN-merge neighbours-delete short)
        if "CNN" in self.currfilt:
            cl = SupportClasses.ConfigLoader()
            filterlist = cl.filters(self.filterdir, bats=False)
            CNNDicts = cl.CNNmodels(filterlist, self.filterdir, [filtname])
            if filtname in CNNDicts.keys():
                CNNmodel = CNNDicts[filtname]
                self.text = self.getSummary(avianz_batch, CNN=True, CNNmodel=CNNmodel)
            else:
                print("ERROR: Couldn't find a matching CNN!")
                self.outfile.write("No matching CNN found!\n")
                self.outfile.write("-- End of testing --\n")
                self.outfile.close()
                return
        self.outfile.write("-- End of testing --\n")
        self.outfile.close()

        print("Testing output written to " + os.path.join(self.testDir, "test-results.txt"))

        # Tidy up
        for root, dirs, files in os.walk(self.testDir):
            for file in files:
                if file.endswith('.tmpdata'):
                    os.remove(os.path.join(root, file))

    def getOutput(self):
        return self.text

    def findCTsegments(self, file, calltypei):
        calltypeSegments = []
        species = self.currfilt["species"]
        if file.lower().endswith('.wav') and os.path.isfile(file + '.tmpdata'):
            segments = Segment.SegmentList()
            segments.parseJSON(file + '.tmpdata')
            if len(self.calltypes) == 1:
                ctSegments = segments.getSpecies(species)
            else:
                ctSegments = segments.getCalltype(species, self.calltypes[calltypei])
            for indx in ctSegments:
                seg = segments[indx]
                calltypeSegments.append(seg[:2])

        return calltypeSegments

    def getSummary(self, avianz_batch, CNN=False, CNNmodel=None):
        autoSegNum = 0
        autoSegCT = [[] for i in range(len(self.calltypes))]
        ws = WaveletSegment.WaveletSegment()
        TP = FP = TN = FN = 0
        for root, dirs, files in os.walk(self.testDir):
            for file in files:
                wavFile = os.path.join(root, file)
                if file.lower().endswith('.wav') and os.stat(wavFile).st_size != 0 and \
                        file + '.tmpdata' in files and file[:-4] + '-res' + str(float(self.window)) + 'sec.txt' in files:
                    autoSegCTCurrent = [[] for i in range(len(self.calltypes))]
                    avianz_batch.filename = os.path.join(root, file)
                    avianz_batch.loadFile([self.filtname], anysound=False)
                    duration = int(np.ceil(len(avianz_batch.audiodata) / avianz_batch.sampleRate))
                    for i in range(len(self.calltypes)):
                        ctsegments = self.findCTsegments(avianz_batch.filename, i)
                        post = Segment.PostProcess(configdir=self.configdir, audioData=avianz_batch.audiodata,
                                                   sampleRate=avianz_batch.sampleRate,
                                                   tgtsampleRate=self.sampleRate, segments=ctsegments,
                                                   subfilter=self.currfilt['Filters'][i], CNNmodel=CNNmodel, cert=50)
                        post.wind()
                        if CNN and CNNmodel:
                            post.CNN()
                        if 'F0' in self.currfilt['Filters'][i] and 'F0Range' in self.currfilt['Filters'][i]:
                            if self.currfilt['Filters'][i]["F0"]:
                                print("Checking for fundamental frequency...")
                                post.fundamentalFrq()
                        post.joinGaps(maxgap=self.currfilt['Filters'][i]['TimeRange'][3])
                        post.deleteShort(minlength=self.currfilt['Filters'][i]['TimeRange'][0])
                        if post.segments:
                            for seg in post.segments:
                                autoSegCTCurrent[i].append(seg[0])
                                autoSegCT[i].append(seg[0])
                                autoSegNum += 1
                    # back-convert to 0/1:
                    det01 = np.zeros(duration)
                    for i in range(len(self.calltypes)):
                        for seg in autoSegCTCurrent[i]:
                            det01[int(seg[0]):int(seg[1])] = 1
                    # get and parse the agreement metrics
                    GT = self.loadGT(os.path.join(root, file[:-4] + '-res' + str(float(self.window)) + 'sec.txt'),
                                     duration)
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

        if CNN:
            self.outfile.write("\n\n-- Wavelet Pre-Processor + CNN detection summary --\n")
        else:
            self.outfile.write("\n-- Wavelet Pre-Processor detection summary --\n")
        self.outfile.write("TP | FP | TN | FN seconds:\t %.2f | %.2f | %.2f | %.2f\n" % (TP, FP, TN, FN))
        self.outfile.write("Specificity:\t\t%.2f %%\n" % (specificity * 100))
        self.outfile.write("Recall (sensitivity):\t%.2f %%\n" % (recall * 100))
        self.outfile.write("Precision (PPV):\t%.2f %%\n" % (precision * 100))
        self.outfile.write("Accuracy:\t\t%.2f %%\n\n" % (accuracy * 100))
        self.outfile.write("Manually labelled segments:\t%d\n" % (self.manSegNum))
        for i in range(len(self.calltypes)):
            self.outfile.write("Auto suggested \'%s\' segments:\t%d\n" % (self.calltypes[i], len(autoSegCT[i])))
        self.outfile.write("Total auto suggested segments:\t%d\n\n" % (autoSegNum))

        if CNN:
            text = "Wavelet Pre-Processor + CNN detection summary\n\n\tTrue Positives:\t%d seconds (%.2f %%)\n\tFalse Positives:\t%d seconds (%.2f %%)\n\tTrue Negatives:\t%d seconds (%.2f %%)\n\tFalse Negatives:\t%d seconds (%.2f %%)\n\n\tSpecificity:\t%.2f %%\n\tRecall:\t\t%.2f %%\n\tPrecision:\t%.2f %%\n\tAccuracy:\t%.2f %%\n" \
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
            print("ERROR: annotation length %d does not match file duration %d!" % (len(d), n))
            self.annotation = []
            return False

        # for each second, store 0/1 presence:
        for row in d:
            annotation.append(int(row[1]))

        return annotation

