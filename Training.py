
# This is part of the AviaNZ interface
# Holds most of the code for training CNNs
# Version 2.0 18/11/19
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2019

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

import os, time, platform, copy, re, json, tempfile, csv
from shutil import copyfile
from shutil import disk_usage

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
import matplotlib.pyplot as plt

import SupportClasses
import SignalProc
import CNN

class CNNtrain:

    def __init__(self,filterdir,folderTrain1=None,folderTrain2=None,recogniser=None,imgWidth=0,CLI=False):

        self.filterdir = filterdir
        cl = SupportClasses.ConfigLoader()
        self.FilterDict = cl.filters(filterdir, bats=False)
        # TODO: hard-coded params
        self.sp = SignalProc.SignalProc(256, 128)

        self.imgsize = [256, 256]
        self.tmpdir1 = False
        self.tmpdir2 = False

        self.CLI = CLI
        if CLI:
            self.species = recogniser
            self.folderTrain1 = folderTrain1
            self.folderTrain2 = folderTrain2
            self.imgWidth = imgWidth
            self.autoThr = True
            self.correction = True
        else:
            self.autoThr = False
            self.correction = False

    def setP1(self,folderTrain1,folderTrain2,recogniser,annotationLevel):
        self.folderTrain1 = folderTrain1
        self.folderTrain2 = folderTrain2
        self.filterName = recogniser
        self.annotatedAll = annotationLevel
        print('All ',self.annotatedAll)

    #def setP2(self,conf1,conf2):
        #self.userConfident = conf1
        #self.userAnnotated = conf2

    def setP3(self,imgWidth,windowWidth,windowInc):
        self.imgWidth = imgWidth
        self.windowWidth = windowWidth
        self.windowInc = windowInc

    #def setP4(self,thr1,thr2):
        #self.
    def setP6(self,recogniser):
        self.newFilterName = recogniser
            
    #def setWidth(self,value):
        #imgWidth = value

    def cliTrain(self):
        # This proceeds very much like the wizard 
            
        # Load data
        self.loadData()

        # Parameters
        self.checkDisk()
        self.windowWidth = self.cnntrain.imgsize[0] * 2
        self.windowInc = int(np.ceil(self.imgsec.value() * self.cnntrain.fs / (self.cnntrain.imgsize[1] - 1)) / 100)

        # Train
        self.train()

        self.saveFilter()

    def loadData(self,hasAnnotation=True):
        self.currfilt = self.FilterDict[self.filterName[:-4]]
        #self.currfilt = self.FilterDicts[self.field("filter")[:-4]]

        self.fs = self.currfilt["SampleRate"]
        self.species = self.currfilt["species"]

        mincallengths = []
        maxcallengths = []
        self.calltypes = []
        for fi in self.currfilt['Filters']:
            self.calltypes.append(fi['calltype'])
            mincallengths.append(fi['TimeRange'][0])
            maxcallengths.append(fi['TimeRange'][1])
        self.mincallength = np.max(mincallengths)
        self.maxcallength = np.max(maxcallengths)

        # Note: no error checking in the CLI version

        print("Manually annotated: %s" % self.folderTrain1)
        print("Auto processed and reviewed: %s" % self.folderTrain2)
        print("Recogniser: %s" % self.currfilt)
        print("Species: %s" % self.species)
        print("Call types: %s" % self.calltypes)
        print("Call length: %.2f - %.2f sec" % (self.mincallength, self.maxcallength))
        print("Sample rate: %d Hz" % self.fs)

        # Find segments belong to each class - Train data
        self.genSegmentDataset(hasAnnotation)

        # TODO: param
        # We need at least some number of segments from each class to proceed
        if min(self.trainN) < 5:    
            print('Warning: Need at least 5 segments from each class to train CNN')

    def checkDisk(self):
        # Check disk usage
        totalbytes, usedbytes, freebytes = disk_usage(os.path.expanduser("~"))
        freeGB = freebytes/1024/1024/1024
        print('\nFree space in the user directory: %.2f GB/ %.2f GB\n' % (freeGB, totalbytes/1024/1024/2014))
        if freeGB < 10:
            print('Warning: You may run out of space in the user directory!')
        return freeGB, totalbytes/1024/1024/1024

    def genSegmentDataset(self,hasAnnotation):
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
                            if len(seg)!=2:
                                print("Warning: old format corrections detected")
                                continue
                            oldlabel = seg[0][4]
                            # check in cases like: [kiwi] -> [kiwi, morepork]
                            # (these will be stored in .corrections, but aren't incorrect detections)
                            newsp = [lab["species"] for lab in seg[1]]
                            if len(oldlabel)!=1:
                                # this was made manually
                                print("Warning: ignoring labels with multiple species")
                                continue
                            if oldlabel[0]['species'] == self.species and self.species not in newsp:
                                # store this as "noise" calltype
                                data.append([wavfile, seg[0][:2], len(self.calltypes)])
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

    # TODO: hop param
    def genImgDataset(self):
        ''' Generate training images'''
        for ct in range(len(self.calltypes) + 1):
            os.makedirs(os.path.join(self.tmpdir1.name, str(ct)))
        self.imgsize[1], self.Nimg = self.DataGen.generateFeatures(dirName=self.tmpdir1.name, dataset=self.traindata, hop=self.imgWidth/5)
        #self.wizard().parameterPage.imgsize[1], N = self.DataGen.generateFeatures(dirName=self.tmpdir1.name, dataset=segments, hop=self.imgsec.value() / 500)

    def train(self):
        # Create temp dir to hold img data and model
        if self.tmpdir1:
            self.tmpdir1.cleanup()
        if self.tmpdir2:
            self.tmpdir2.cleanup()
        self.tmpdir1 = tempfile.TemporaryDirectory(prefix='CNN_')
        print('Temporary img dir:', self.tmpdir1.name)
        self.tmpdir2 = tempfile.TemporaryDirectory(prefix='CNN_')
        print('Temporary model dir:', self.tmpdir2.name)

        # Find train segments belong to each class
        # TODO: params here!
        self.DataGen = CNN.GenerateData(self.currfilt, self.imgWidth, self.windowWidth, self.windowInc, self.imgsize[0], self.imgsize[1])
        # TODO
        #self.segments = self.wizard().confirminputPage.trainsegments

        print('Generating CNN images...')
        self.genImgDataset()
        print('\nGenerated images:\n')
        for i in range(len(self.calltypes)):
            print("\t%s:\t%d\n" % (self.calltypes[i], self.Nimg[i]))
        print("\t%s:\t%d\n" % ("Noise", self.Nimg[-1]))

        # CNN training
        cnn = CNN.CNN(self.species, self.calltypes, self.fs, self.imgWidth, self.windowWidth, self.windowInc, self.imgsize[0], self.imgsize[1])
        #cnn = CNN.CNN(self.species, self.calltypes, self.fs, self.imgsec.value() / 100, self.windowidth, self.incwidth, self.imgsize[0], self.imgsize[1])
        # TODO: hard-coded parameters?
        batchsize = 32

        # 1. Data augmentation
        filenames, labels = cnn.getImglist(self.tmpdir1.name)
        labels = np.argmax(labels, axis=1)
        ns = [np.shape(np.where(labels == i)[0])[0] for i in range(len(self.calltypes) + 1)]
        # create image data augmentation generator in-build
        datagen = ImageDataGenerator(width_shift_range=0.3, fill_mode='nearest')
        # TODO
        t = 1000
        # Data augmentation for each call type
        for ct in range(len(self.calltypes) + 1):
            if t - ns[ct] > batchsize:
                # load this ct images
                samples = cnn.loadCTImg(os.path.join(self.tmpdir1.name, str(ct)))
                # prepare iterator
                it = datagen.flow(samples, batch_size=batchsize)
                # generate samples
                batch = it.next()
                for j in range(int((t - ns[ct]) / batchsize)):
                    newbatch = it.next()
                    batch = np.vstack((batch, newbatch))
                # Save augmented data
                k = 0
                for sgRaw in batch:
                    np.save(os.path.join(self.tmpdir1.name, str(ct), str(ct) + '_aug' + "%06d" % k + '.npy'),
                            sgRaw)
                    k += 1

        # 2. TRAIN - use custom image generator
        filenamesall, labelsall = cnn.getImglist(self.tmpdir1.name)
        filenamesall, labelsall = shuffle(filenamesall, labelsall)
        # TODO: params?
        X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(filenamesall, labelsall, test_size=0.10, random_state=1)
        training_batch_generator = CNN.CustomGenerator(X_train_filenames, y_train, batchsize, self.tmpdir1.name, cnn.imageheight, cnn.imagewidth, 1)
        validation_batch_generator = CNN.CustomGenerator(X_val_filenames, y_val, batchsize, self.tmpdir1.name, cnn.imageheight, cnn.imagewidth, 1)

        print('Creating CNN architecture...')
        cnn.createArchitecture()

        print('Training...')
        cnn.train(modelsavepath=self.tmpdir2.name, training_batch_generator=training_batch_generator, validation_batch_generator=validation_batch_generator, batch_size=batchsize)
        print('Training complete!')

        self.bestThr = [[0, 0] for i in range(len(self.calltypes) + 1)]
        self.bestThrInd = [0 for i in range(len(self.calltypes) + 1)]

        # 3. Prepare ROC plots
        print('Generating ROC statistics...')
        TPs = [0 for i in range(len(self.calltypes) + 1)]
        FPs = [0 for i in range(len(self.calltypes) + 1)]
        TNs = [0 for i in range(len(self.calltypes) + 1)]
        FNs = [0 for i in range(len(self.calltypes) + 1)]
        CTps = [[] for i in range(len(self.calltypes) + 1)]
        N = len(filenames)
        batchsize = 100
        for i in range(int(np.ceil(N / batchsize))):
            imagesb = cnn.loadImgBatch(filenames[i * batchsize:min((i + 1) * batchsize, N)])
            labelsb = labels[i * batchsize:min((i + 1) * batchsize, N)]
            for ct in range(len(self.calltypes) + 1):
                res, ctp = self.testCT(ct, imagesb, labelsb)  # res=[thrlist, TPs, FPs, TNs, FNs]
                CTps[ct] += ctp
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

        for ct in range(len(self.calltypes) + 1):
            self.TPRs[ct] = [TPs[ct][i] / (TPs[ct][i] + FNs[ct][i]) for i in range(len(self.Thrs))]
            self.FPRs[ct] = [FPs[ct][i] / (TNs[ct][i] + FPs[ct][i]) for i in range(len(self.Thrs))]
            self.Precisions[ct] = [0.0 if (TPs[ct][i] + FPs[ct][i]) == 0 else TPs[ct][i] / (TPs[ct][i] + FPs[ct][i]) for i in range(len(self.Thrs))]
            self.Accs[ct] = [(TPs[ct][i] + TNs[ct][i]) / (TPs[ct][i] + TNs[ct][i] + FPs[ct][i] + FNs[ct][i]) for
                             i in range(len(self.Thrs))]

            # Temp plot is saved in train data directory - prediction probabilities for instances of current ct
            CTps[ct] = sorted(CTps[ct], key=float)
            fig = plt.figure()
            ax = plt.axes()
            ax.plot(CTps[ct], 'k')
            ax.plot(CTps[ct], 'bo')
            plt.xlabel('Number of samples')
            plt.ylabel('Probability')
            if ct == len(self.calltypes):
                plt.title('Class: Noise')
            else:
                plt.title('Class: ' + str(self.calltypes[ct]))
            if self.folderTrain1:
                fig.savefig(os.path.join(self.folderTrain1, str(ct) + '.png'))
            else:
                fig.savefig(os.path.join(self.folderTrain2, str(ct) + '.png'))
            plt.close()

        # 4. Auto select the upper threshold (fpr = 0)
        for ct in range(len(self.calltypes) + 1):
            try:
                self.bestThr[ct][1] = self.Thrs[self.FPRs[ct].index(0.0)]
            except:
                self.bestThr[ct][1] = self.Thrs[len(self.FPRs) - 1]

        # 5. Auto select lower threshold IF the user asked so
        if self.autoThr:
            for ct in range(len(self.calltypes) + 1):
                # Get min distance to ROC from (0 FPR, 1 TPR)
                distarr = (np.float64(1) - self.TPRs[ct]) ** 2 + (np.float64(0) - self.FPRs[ct]) ** 2
                self.thr_min_ind = np.unravel_index(np.argmin(distarr), distarr.shape)[0]
                self.bestThr[ct][0] = self.Thrs[self.thr_min_ind]
                self.bestThrInd[ct] = self.thr_min_ind
        return True

    def testCT(self, ct, testimages, targets):
        '''
        :param ct: integer relevant to call type
        :return: [thrlist, TPs, FPs, TNs, FNs], ctprob
        '''

        self.thrs = []
        self.TPs = []
        self.FPs = []
        self.TNs = []
        self.FNs = []

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
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Predict and temp plot (just for me)
        pre = model.predict(testimages)
        ctprob = []
        for i in range(len(targets)):
            if targets[i] == ct:
                ctprob.append(pre[i][ct])

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
        CNNdic = {}
        CNNdic["CNN_name"] = "CNN_name"
        CNNdic["loss"] = "binary_crossentropy"
        CNNdic["optimizer"] = "adam"
        CNNdic["windowInc"] = [self.windowWidth,self.windowInc] #[self.wizard().parameterPage.windowidth, self.wizard().parameterPage.incwidth]
        CNNdic["win"] = [self.imgWidth,self.imgWidth/5] #[self.wizard().parameterPage.imgsec.value() / 100, self.wizard().parameterPage.imgsec.value() / 500]
        CNNdic["inputdim"] = self.imgsize #self.wizard().parameterPage.imgsize
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

        # TODO: save it!

    def cleanSpecies(self, species):
        """ Returns cleaned species name"""
        return re.sub(r'[^A-Za-z0-9()-]', "_", species)

