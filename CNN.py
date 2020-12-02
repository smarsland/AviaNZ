
# CNN.py
#
# CNN for the AviaNZ program

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
#     from PyQt5.QtGui import QIcon, QPixmap

import tensorflow as tf
from skimage.transform import resize

import json, os
import numpy as np
import math
import gc
from time import gmtime, strftime

import SignalProc
import WaveletSegment
import Segment
import WaveletFunctions
import SupportClasses
import librosa
import wavio

# from sklearn.metrics import confusion_matrix
# from numpy import expand_dims
# from keras_preprocessing.image import ImageDataGenerator
# import pyqtgraph as pg
# import SupportClasses

class CNN:
    """ This class implements CNN training and data augmentation in AviaNZ.
    """

    def __init__(self, configdir, species, calltypes, fs, length, windowwidth, inc, imageheight, imagewidth):
        self.species = species
        self.length = length
        self.windowwidth = windowwidth
        self.inc = inc
        self.imageheight = imageheight
        self.imagewidth = imagewidth
        self.calltypes = calltypes
        self.fs = fs

        cl = SupportClasses.ConfigLoader()
        self.LearningDict = cl.learningParams(os.path.join(configdir, "LearningParams.txt"))

    # Custom data augmentation
    def addNoise(self, image, noise_image):
        ''' Add random percentage of noiseImage to image.
        :param image: original image
        :param noiseImage: noise image
        :return: new image
        '''
        new_image = image + noise_image*np.random.uniform(0.2, 0.8)
        return new_image

    def genBatchNoise(self, images, noise_pool, n):
        ''' Generate a batch of n new images
        :param images: a set of original images
        :param noise_pool: noise pool images
        :param n: number of new images to generate
        :return: new images
        '''
        new_images = np.ndarray(shape=(n, self.imageheight, self.imagewidth, 1), dtype=float)
        for i in range(0, n):
            # pick a random image and add a random % of random noise from the noise_pool
            new_images[i][:] = self.addNoise(images[np.random.randint(0, np.shape(images)[0])],
                                             noise_pool[np.random.randint(0, np.shape(noise_pool)[0])])
        return new_images

    def genBatchNoise2(self, audios, noise_pool, n):
        ''' Generate a batch of n new images
        :param images: a set of original images
        :param noise_pool: noise pool images
        :param n: number of new images to generate
        :return: new images
        '''
        new_audios = np.ndarray(shape=(n, self.fs*self.length), dtype=float)
        for i in range(0, n):
            # pick a random image and add a random % of random noise from the noise_pool
            new_audios[i][:] = self.addNoise(audios[np.random.randint(0, np.shape(audios)[0])],
                                             noise_pool[np.random.randint(0, np.shape(noise_pool)[0])])

        new_images = np.ndarray(shape=(n, self.imageheight, self.imagewidth), dtype=float)
        for i in range(0, n):
            new_images[i][:] = self.generateImage(new_audios[i][:])
        return new_images.reshape(new_images.shape[0], self.imageheight, self.imagewidth, 1)

    def timeStretch(self, data, rate):
        ''' Time stretch audio data by given rate
        :param data: audio data
        :param rate: stretch rate
        :return: new audio data
        '''
        input_length = len(data)
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

    def generateImage(self, audiodata):
        ''' Generate spectrogram image'''
        sp = SignalProc.SignalProc(self.windowwidth, self.inc)
        sp.data = audiodata
        sp.sampleRate = self.fs
        sgRaw = sp.spectrogram(self.windowwidth, self.inc)
        maxg = np.max(sgRaw)
        return np.rot90(sgRaw / maxg).tolist()

    def changeSpeed(self, audiodata):
        ''' Change the speed of the audio data (time stretch) and then generate spectrogram image
        :param data: audio data
        :return: new image
        '''
        # choose rate
        mu, sigma = 1, 0.05  # mean and standard deviation
        s = np.random.normal(mu, sigma, 1000)
        rate = s[int(np.random.random() * 1000)]
        newdata = self.timeStretch(audiodata, rate)
        img = self.generateImage(newdata)
        return img

    def genBatchChangeSpeed(self, audios, n):
        ''' Generate a batch of n new images, change speed
        :param audios:
        :param n:
        :return:
        '''
        new_images = np.ndarray(shape=(np.shape(audios)[0], self.imageheight, self.imagewidth, 1), dtype=float)
        for i in range(0, n):
            # pick a random audio to time stretch
            new_images[i][:] = self.changeSpeed(audios[np.random.randint(0, np.shape(audios)[0])])
        return new_images

    # def pitchShift(self, audiodata):
    #     '''
    #     :param audiodata:
    #     :return:
    #     '''
    #     mu, sigma = 0, ?  # mean and standard deviation
    #     s = np.random.normal(mu, sigma, 1000)
    #     pitch_factor = s[int(np.random.random() * 1000)]
    #     return librosa.effects.pitch_shift(audiodata, fs, pitch_factor)

    def genBatchPitchShift(self, audios, n):
        ''' Generate a batch of n new images
        :param audios:
        :param n:
        :return:
        '''
        new_images = np.ndarray(shape=(np.shape(audios)[0], self.imageheight, self.imagewidth, 1), dtype=float)
        for i in range(0, n):
            # pick a random audio to time stretch
            new_images[i][:] = self.pitchShift(audios[np.random.randint(0, np.shape(audios)[0])])
        return new_images

    def loadCTImg(self, dirName):
        ''' Returns images of the call type subdirectory dirName'''
        filenames, labels = self.getImglist(dirName)

        return np.array([resize(np.load(file_name), (self.imageheight, self.imagewidth, 1)) for file_name in
                         filenames])

    def loadImgBatch(self, filenames):
        ''' Returns images given the list of file names'''
        return np.array([resize(np.load(file_name), (self.imageheight, self.imagewidth, 1)) for file_name in
                         filenames])

    def loadImageData(self, file, noisepool=False):
        '''
        :param file: JSON file with extracted features and labels
        :return:
        '''
        npzfile = file
        dataz = np.load(npzfile)
        numarrays = len(dataz)

        labfile = file[:-4] + '_labels.json'
        with open(labfile) as f:
            labels = json.load(f)

        # initialize output
        features = np.ndarray(shape=(numarrays, self.imageheight, self.imagewidth), dtype=float)

        badind = []
        if noisepool:
            i = 0
            for key in dataz.files:
                if np.shape(dataz[key]) == (self.imageheight, self.imagewidth):
                    features[i][:] = dataz[key][:]
                else:
                    badind.append(i)
                i += 1
            features = np.delete(features, badind, 0)
            return features
        else:
            targets = np.zeros((numarrays, 1))
            i = 0
            for key in dataz.files:
                try:
                    if np.shape(dataz[key]) == (self.imageheight, self.imagewidth):
                        features[i][:] = dataz[key][:]
                        targets[i][0] = labels[key]
                    else:
                        badind.append(i)
                    i += 1
                except Exception as e:
                    print("Error: failed to load image because:", e)

            features = np.delete(features, badind, 0)
            targets = np.delete(targets, badind, 0)
            return features, targets

    def loadAudioData(self, file, noisepool=False):
        '''
        :param file: JSON file with extracted features and labels
        :return:
        '''
        with open(file) as f:
            data = json.load(f)
        nsamp = self.fs*self.length
        features = np.ndarray(shape=(np.shape(data)[0], nsamp), dtype=float)
        badind = []
        if noisepool:
            for i in range(0, np.shape(data)[0]):
                if len(data[i][0]) == nsamp:
                    features[i][:] = data[i][0][:]
                elif len(data[i][0]) > nsamp:
                    features[i][:] = data[i][0][:nsamp]
                else:
                    badind.append(i)
            features = np.delete(features, badind, 0)
            return features
        else:
            targets = np.zeros((np.shape(data)[0], 1))
            for i in range(0, np.shape(data)[0]):
                if len(data[i][0]) == nsamp:
                    features[i][:] = data[i][0][:]
                    targets[i][0] = data[i][-1]
                elif len(data[i][0]) > nsamp:
                    features[i][:] = data[i][0][:nsamp]
                    targets[i][0] = data[i][-1]
                else:
                    badind.append(i)
            features = np.delete(features, badind, 0)
            targets = np.delete(targets, badind, 0)
            return features, targets

    def loadAllImageData(self, dirName):
        ''' Read datasets from dirName, return a list of ct arrays'''
        sg = None
        target = None
        pos = 0
        for root, dirs, files in os.walk(str(dirName)):
            for file in files:
                if file.endswith('.npz'):
                    print('reading ', file)
                    sg1, target1 = self.loadImageData(os.path.join(dirName, file))
                    if not pos:
                        sg = sg1
                        target = target1
                        pos += np.shape(target1)[0]
                    else:
                        sg = np.vstack((sg, sg1))
                        target = np.vstack((target, target1))
                        pos += np.shape(target1)[0]

        # Separate into classes
        ns = [np.shape(np.where(target == i)[0])[0] for i in range(len(self.calltypes) + 1)]
        sgCT = [np.empty((n, self.imageheight, self.imagewidth), dtype=float) for n in ns]
        idxs = [np.random.permutation(np.where(target == i)[0]).tolist() for i in range(len(self.calltypes) + 1)]
        for ct in range(len(self.calltypes) + 1):
            i = 0
            for j in idxs[ct]:
                sgCT[ct][i][:] = sg[j][:]
                i += 1
        return sgCT, ns

    def getImglist(self, dirName):
        ''' Returns the image filenames and labels in dirName:
        '''
        filenames = []
        labels = []

        for root, dirs, files in os.walk(dirName):
            for file in files:
                if file.endswith('.npy'):
                    filenames.append(os.path.join(root, file))
                    lbl = file.split('_')[0]
                    labels.append(int(lbl))

        # One hot vector representation of the labels
        labels = tf.keras.utils.to_categorical(np.array(labels), len(self.calltypes) + 1)

        return filenames, labels

    def createArchitecture(self):
        '''
        Sets self.model
        '''
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=[self.imageheight, self.imagewidth, 1], padding='Same'))
        self.model.add(tf.keras.layers.Conv2D(64, (7, 7), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(tf.keras.layers.Dropout(0.2))
        # Flatten the results to one dimension for passing into our final layer
        self.model.add(tf.keras.layers.Flatten())
        # A hidden layer to learn with
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        # Another dropout
        self.model.add(tf.keras.layers.Dropout(0.5))
        # Final categorization from 0-ct+1 with softmax
        self.model.add(tf.keras.layers.Dense(len(self.calltypes)+1, activation='softmax'))
        self.model.summary()

    def train2(self, modelsavepath):
        ''' Train the model - keep all in memory '''

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        if not os.path.exists(modelsavepath):
            os.makedirs(modelsavepath)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            modelsavepath + "/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.h5",
            monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto',
            save_freq='epoch')
        early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')
        self.history = self.model.fit(self.train_images, self.train_labels,
                                      batch_size=32,
                                      epochs=50,
                                      verbose=2,
                                      validation_data=(self.val_images, self.val_labels),
                                      callbacks=[checkpoint, early],
                                      shuffle=True)
        # Save the model
        # Serialize model to JSON
        model_json = self.model.to_json()
        with open(modelsavepath + "/model.json", "w") as json_file:
            json_file.write(model_json)
        # # just serialize final weights to H5, not necessary
        # self.model.save_weights(modelsavepath + "/weights.h5")
        print("Saved model to ", modelsavepath)

    def train(self, modelsavepath, training_batch_generator, validation_batch_generator):
        ''' Train the model - use image generator '''

        # self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.compile(loss=self.LearningDict['loss'], optimizer=self.LearningDict['optimizer'], metrics=self.LearningDict['metrics'])

        if not os.path.exists(modelsavepath):
            os.makedirs(modelsavepath)
        # checkpoint = tf.keras.callbacks.ModelCheckpoint(modelsavepath + "/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
        # early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=1, mode='auto')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(modelsavepath + "/weights.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.h5", monitor=self.LearningDict['monitor'], verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
        early = tf.keras.callbacks.EarlyStopping(monitor=self.LearningDict['monitor'], min_delta=0, patience=self.LearningDict['patience'], verbose=1, mode='auto')

        epochs = self.LearningDict['epochs']
        self.history = self.model.fit(training_batch_generator,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=validation_batch_generator,
                                      callbacks=[checkpoint, early])

        # Save the model
        # Serialize model to JSON
        model_json = self.model.to_json()
        with open(modelsavepath + "/model.json", "w") as json_file:
            json_file.write(model_json)
        print("Saved model to ", modelsavepath)

class GenerateData:
    """ This class implements CNN data preparation. There are different ways:
    1. when manually annotated recordings are presented (.wav and GT.data along with call type info). In this case run
    the existing recogniser (WF) over the data set and get the diff to find FP segments (Noise class). And .data has TP/
    call type segments
    2. auto processed and batch reviewed data (.wav, .data, .correction). .data has TP/call type segments while
    .correction has segments for the noise class
    3. when extracted pieces of sounds (of call types and noise) are presented TODO
    """
    def __init__(self, filter, length, windowwidth, inc, imageheight, imagewidth):
        self.filter = filter
        self.species = filter["species"]
        # not sure if this is needed?
        ind = self.species.find('>')
        if ind != -1:
            self.species = self.species.replace('>', '(')
            self.species = self.species + ')'
        self.calltypes = []
        for fi in filter['Filters']:
            self.calltypes.append(fi['calltype'])
        self.fs = filter["SampleRate"]
        self.length = length
        self.windowwidth = windowwidth
        self.inc = inc
        self.imageheight = imageheight
        self.imagewidth = imagewidth

    def findCTsegments(self, dirName, calltypei):
        ''' dirName got reviewed.data or manual.data
            Find calltype segments
            :returns ct segments [[filename, seg, label], ...]
        '''

        calltypeSegments = []
        for root, dirs, files in os.walk(dirName):
            for file in files:
                wavFile = os.path.join(root, file)
                if file.lower().endswith('.wav') and file + '.data' in files:
                    segments = Segment.SegmentList()
                    segments.parseJSON(wavFile + '.data')
                    if len(self.calltypes) == 1:
                        ctSegments = segments.getSpecies(self.species)
                    else:
                        ctSegments = segments.getCalltype(self.species, self.calltypes[calltypei])
                    for indx in ctSegments:
                        seg = segments[indx]
                        # skip uncertain segments
                        cert = [lab["certainty"] if lab["species"] == self.species else 100 for lab in seg[4]]
                        if cert:
                            mincert = min(cert)
                            if mincert == 100:
                                calltypeSegments.append([wavFile, seg[:2], calltypei])

        return calltypeSegments

    def findNoisesegments(self, dirName):
        ''' dirName got manually annotated GT.data
        Find noise segments by diff of auto segments and GT.data
        :returns noise segments [[filename, seg, label], ...]
        '''

        manSegNum = 0
        window = 1
        inc = None
        noiseSegments = []
        # Generate GT files from annotations in dir1
        print('Generating GT...')
        for root, dirs, files in os.walk(dirName):
            for file in files:
                wavFile = os.path.join(root, file)
                if file.lower().endswith('.wav') and os.stat(wavFile).st_size != 0 and file + '.data' in files:
                    segments = Segment.SegmentList()
                    segments.parseJSON(wavFile + '.data')
                    sppSegments = segments.getSpecies(self.species)
                    manSegNum += len(sppSegments)

                    # Currently, we ignore call types here and just
                    # look for all calls for the target species.
                    segments.exportGT(wavFile, self.species, window=window, inc=inc)
        if manSegNum == 0:
            print("ERROR: no segments for species %s found" % self.species)
            return

        ws = WaveletSegment.WaveletSegment(self.filter, 'dmey2')
        autoSegments = ws.waveletSegment_cnn(dirName, self.filter)  # [(filename, [segments]), ...]

        #  now the diff between segment and autoSegments
        print("autoSeg", autoSegments)
        for item in autoSegments:
            print(item[0])
            wavFile = item[0]
            if os.stat(wavFile).st_size != 0:
                sppSegments = []
                if os.path.isfile(wavFile + '.data'):
                    segments = Segment.SegmentList()
                    segments.parseJSON(wavFile + '.data')
                    sppSegments = segments.getSpecies(self.species)
                for segAuto in item[1]:
                    overlappedwithGT = False
                    for ind in sppSegments:
                        segGT = segments[ind]
                        if self.Overlap(segGT, segAuto):
                            overlappedwithGT = True
                            break
                        else:
                            continue
                    if not overlappedwithGT:
                        noiseSegments.append([wavFile, segAuto, len(self.calltypes)])
        return noiseSegments

    def Overlap(self, segGT, seg):
        # return True if the two segments, segGT and seg overlap
        if segGT[1] >= seg[0] >= segGT[0]:
            return True
        elif segGT[1] >= seg[1] >= segGT[0]:
            return True
        elif segGT[1] <= seg[1] and segGT[0] >= seg[0]:
            return True
        else:
            return False

    def getImgCount(self, dirName, dataset, hop):
        '''
        Read the segment library and estimate the number of CNN images per class
        :param dataset: segments in the form of [[file, [segment], label], ..]
        :param hop: list of hops for different classes
        :return: a list
        '''
        dhop = hop
        eps = 0.0005
        N = [0 for i in range(len(self.calltypes) + 1)]

        for record in dataset:
            # Compute number of images, also consider tiny segments because this would be the case for song birds.
            duration = record[1][1] - record[1][0]
            hop = dhop[record[-1]]
            if duration < self.length:
                fileduration = wavio.readFmt(record[0])[1]
                record[1][0] = record[1][0] - (self.length - duration)/2 - eps
                record[1][1] = record[1][1] + (self.length - duration)/2 + eps
                if record[1][0] < 0:
                    record[1][0] = 0
                    record[1][1] = self.length + eps
                elif record[1][1] > fileduration:
                    record[1][1] = fileduration
                    record[1][0] = fileduration - duration - eps
                if 0 <= record[1][0] and record[1][1] <= fileduration:
                    n = 1
                else:
                    n = 0
            else:
                n = math.ceil((record[1][1] - record[1][0] - self.length) / hop + 1)
            N[record[-1]] += n

        return N

    def generateFeatures(self, dirName, dataset, hop):
        '''
        Read the segment library and generate features, training
        :param dataset: segments in the form of [[file, [segment], label], ..]
        :param hop:
        :return: save the preferred features into JSON files + save images. Currently the spectrogram images.
        '''
        count = 0
        dhop = hop
        eps = 0.0005
        specFrameSize = len(range(0, int(self.length * self.fs - self.windowwidth), self.inc))
        N = [0 for i in range(len(self.calltypes) + 1)]

        for record in dataset:
            # Compute features, also consider tiny segments because this would be the case for song birds.
            duration = record[1][1] - record[1][0]
            hop = dhop[record[-1]]
            if duration < self.length:
                fileduration = wavio.readFmt(record[0])[1]
                record[1][0] = record[1][0] - (self.length - duration) / 2 - eps
                record[1][1] = record[1][1] + (self.length - duration) / 2 + eps
                if record[1][0] < 0:
                    record[1][0] = 0
                    record[1][1] = self.length + eps
                elif record[1][1] > fileduration:
                    record[1][1] = fileduration
                    record[1][0] = fileduration - self.length - eps
                if record[1][0] <= 0 and record[1][1] <= fileduration:
                    n = 1
                    hop = self.length
                    duration = self.length + eps
                else:
                    continue
            else:
                n = math.ceil((record[1][1]-record[1][0]-self.length) / hop + 1)
            print('* hop:', hop, 'n:', n, 'label:', record[-1])
            try:
                audiodata = self.loadFile(filename=record[0], duration=duration, offset=record[1][0], fs=self.fs, denoise=False)
            except Exception as e:
                print("Warning: failed to load audio because:", e)
                continue
            N[record[-1]] += n
            sp = SignalProc.SignalProc(self.windowwidth, self.inc)
            sp.data = audiodata
            sp.sampleRate = self.fs
            sgRaw = sp.spectrogram(self.windowwidth, self.inc)

            for i in range(int(n)):
                print('**', record[0], self.length, record[1][0]+hop*i, self.fs, '************************************')
                # start = int(hop * i * fs)
                # end = int(hop * i * fs + length * fs)
                # if end > len(audiodata):
                #     end = len(audiodata)
                #     start = int(len(audiodata) - length * fs)
                # audiodata_i = audiodata[start: end]
                # audiodata_i = audiodata_i.tolist()
                # featuresa.append([audiodata_i, record[-1]])

                # Sgram images
                sgstart = int(hop * i * self.fs / sp.incr)
                sgend = sgstart + specFrameSize
                if sgend > np.shape(sgRaw)[0]:
                    sgend = np.shape(sgRaw)[0]
                    sgstart = np.shape(sgRaw)[0] - specFrameSize
                if sgstart < 0:
                    continue
                sgRaw_i = sgRaw[sgstart:sgend, :]
                maxg = np.max(sgRaw_i)
                # Normalize and rotate
                sgRaw_i = np.rot90(sgRaw_i / maxg)
                print(np.shape(sgRaw_i))

                # Save train data: individual images as npy
                np.save(os.path.join(dirName, str(record[-1]),
                        str(record[-1]) + '_' + "%06d" % count + '_' + record[0].split(os.sep)[-1][:-4] + '.npy'),
                        sgRaw_i)
                count += 1

        print('\n\nCompleted feature extraction')
        return specFrameSize, N

    def loadFile(self, filename, duration=0.0, offset=0, fs=0, denoise=False, f1=0, f2=0):
        """
        Read audio file and preprocess as required.
        """
        if duration == 0:
            duration = None

        sp = SignalProc.SignalProc(256, 128)
        sp.readWav(filename, duration, offset)
        sp.resample(fs)
        sampleRate = sp.sampleRate
        audiodata = sp.data

        # # pre-process
        if denoise:
            WF = WaveletFunctions.WaveletFunctions(data=audiodata, wavelet='dmey2', maxLevel=10, samplerate=fs)
            audiodata = WF.waveletDenoise(thresholdType='soft', maxLevel=10)

        if f1 != 0 and f2 != 0:
            # audiodata = sp.ButterworthBandpass(audiodata, sampleRate, f1, f2)
            audiodata = sp.bandpassFilter(audiodata, sampleRate, f1, f2)

        return audiodata


class CustomGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, traindir, imghight, imgwidth, channels):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.train_dir = traindir
        self.imgheight = imghight
        self.imgwidth = imgwidth
        self.channels = channels

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        # return np.array([resize(imread(os.path.join(self.train_dir , str(file_name))), (self.imgheight, self.imgwidth, self.channels)) for file_name in batch_x]) / 255.0, np.array(batch_y)
        return np.array([resize(np.load(file_name), (self.imgheight, self.imgwidth, self.channels)) for file_name in batch_x]), np.array(batch_y)
