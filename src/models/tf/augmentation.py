# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2025

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

# TensorFlow-based data augmentation and training

import tensorflow as tf
from skimage.transform import resize
import json, os
import numpy as np
import librosa

from src.core import spectrogram
from src.core import config_loader
from src.core import audio_data
from src.models.tf import architectures


class NN:
    """ This class implements NN training and data augmentation in AviaNZ. """

    def __init__(self, configdir, species, calltypes, fs, length, windowwidth, inc, imageheight, imagewidth, modelArchitecture):
        self.species = species
        self.length = length
        self.windowwidth = windowwidth
        self.inc = inc
        self.imageheight = imageheight
        self.imagewidth = imagewidth
        self.calltypes = calltypes
        self.fs = fs
        self.modelArchitecture = modelArchitecture

        cl = config_loader.ConfigLoader()
        self.LearningDict = cl.learningParams(os.path.join(configdir, "LearningParams.txt"))

    def addNoise(self, image, noise_image):
        """ Add random percentage of noiseImage to image.
        :param image: original image
        :param noiseImage: noise image
        :return: new image
        """
        new_image = image + noise_image*np.random.uniform(0.2, 0.8)
        return new_image

    def genBatchNoise(self, images, noise_pool, n):
        """ Generate a batch of n new images
        :param images: a set of original images
        :param noise_pool: noise pool images
        :param n: number of new images to generate
        :return: new images
        """
        new_images = np.ndarray(shape=(n, self.imageheight, self.imagewidth, 1), dtype=float)
        for i in range(0, n):
            new_images[i][:] = self.addNoise(images[np.random.randint(0, np.shape(images)[0])],
                                             noise_pool[np.random.randint(0, np.shape(noise_pool)[0])])
        return new_images

    def genBatchNoise2(self, audios, noise_pool, n):
        """ Generate a batch of n new images, add audios -> generate images
        :param images: a set of original images
        :param noise_pool: noise pool images
        :param n: number of new images to generate
        :return: new images
        """
        new_audios = np.ndarray(shape=(n, self.fs*self.length), dtype=float)
        for i in range(0, n):
            new_audios[i][:] = self.addNoise(audios[np.random.randint(0, np.shape(audios)[0])],
                                             noise_pool[np.random.randint(0, np.shape(noise_pool)[0])])

        new_images = np.ndarray(shape=(n, self.imageheight, self.imagewidth), dtype=float)
        for i in range(0, n):
            new_images[i][:] = self.generateImage(new_audios[i][:])
        return new_images.reshape(new_images.shape[0], self.imageheight, self.imagewidth, 1)

    def timeStretch(self, data, rate):
        """ Time stretch audio data by given rate
        :param data: audio data
        :param rate: stretch rate
        :return: new audio data
        """
        input_length = len(data)
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

    def generateImage(self, audiodata):
        """ Generate spectrogram image"""
        sp = spectrogram.Spectrogram(self.windowwidth, self.inc)
        sp.audio_data = audio_data.AudioData(data=audiodata, sample_rate=self.fs, 
                                   sample_format='float32', sample_size=32, channels=1)
        sgRaw = sp.spectrogram(self.windowwidth, self.inc)
        maxg = np.max(sgRaw)
        return np.rot90(sgRaw / maxg).tolist()

    def changeSpeed(self, audiodata):
        """ Change the speed of the audio data (time stretch) and then generate spectrogram image
        :param data: audio data
        :return: new image
        """
        mu, sigma = 1, 0.05
        s = np.random.normal(mu, sigma, 1000)
        rate = s[int(np.random.random() * 1000)]
        newdata = self.timeStretch(audiodata, rate)
        img = self.generateImage(newdata)
        return img

    def genBatchChangeSpeed(self, audios, n):
        """ Generate a batch of n new images, change speed
        :param audios:
        :param n:
        :return:
        """
        new_images = np.ndarray(shape=(np.shape(audios)[0], self.imageheight, self.imagewidth, 1), dtype=float)
        for i in range(0, n):
            new_images[i][:] = self.changeSpeed(audios[np.random.randint(0, np.shape(audios)[0])])
        return new_images

    def genBatchPitchShift(self, audios, n):
        """ Generate a batch of n new images
        :param audios:
        :param n:
        :return:
        """
        new_images = np.ndarray(shape=(np.shape(audios)[0], self.imageheight, self.imagewidth, 1), dtype=float)
        for i in range(0, n):
            new_images[i][:] = self.pitchShift(audios[np.random.randint(0, np.shape(audios)[0])])
        return new_images

    def loadCTImg(self, dirName):
        """ Returns images of the call type subdirectory dirName"""
        filenames, labels = self.getImglist(dirName)

        return np.array([resize(np.load(file_name), (self.imageheight, self.imagewidth, 1)) for file_name in
                         filenames])

    def loadImgBatch(self, filenames):
        """ Returns images given the list of file names"""
        return np.array([resize(np.load(file_name), (self.imageheight, self.imagewidth, 1)) for file_name in
                         filenames])

    def loadImageData(self, file, noisepool=False):
        """
        :param file: JSON file with extracted features and labels
        :return:
        """
        npzfile = file
        dataz = np.load(npzfile)
        numarrays = len(dataz)

        fileMinusExtension = file.rsplit('.', 1)[0]
        labfile = fileMinusExtension + '_labels.json'
        with open(labfile) as f:
            labels = json.load(f)

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
                if np.shape(dataz[key]) == (self.imageheight, self.imagewidth):
                    features[i][:] = dataz[key][:]
                    targets[i][0] = labels[key]
                else:
                    badind.append(i)
                i += 1

            features = np.delete(features, badind, 0)
            targets = np.delete(targets, badind, 0)
            return features, targets

    def loadAudioData(self, file, noisepool=False):
        """
        :param file: JSON file with extracted features and labels
        :return:
        """
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
        """ Read datasets from dirName, return a list of ct arrays"""
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
        """ Returns the image filenames and labels in dirName:
        """
        filenames = []
        labels = []

        for root, dirs, files in os.walk(dirName):
            for file in files:
                if file.endswith('.npy'):
                    filenames.append(os.path.join(root, file))
                    lbl = file.split('_')[0]
                    labels.append(int(lbl))

        labels = tf.keras.utils.to_categorical(np.array(labels), len(self.calltypes) + 1)

        return filenames, labels

    def getOriginalImglist(self, dirName):
        """ Returns only the original image filenames and labels in dirName:
        """
        filenames = []
        labels = []

        for root, dirs, files in os.walk(dirName):
            for file in files:
                if file.endswith('.npy') and '_aug' not in file:
                    filenames.append(os.path.join(root, file))
                    lbl = file.split('_')[0]
                    labels.append(int(lbl))

        labels = tf.keras.utils.to_categorical(np.array(labels), len(self.calltypes) + 1)

        return filenames, labels

    def createArchitecture(self):
        """
        Sets self.model
        """
        if self.modelArchitecture == 'CNN':
            self.model = architectures.CNNModel(self.imageheight, self.imagewidth, len(self.calltypes)+1)
        else:
            raise ValueError("Model architecture not supported")

    def train2(self, modelsavepath):
        """ Train the model - keep all in memory """

        if not os.path.exists(modelsavepath):
            os.makedirs(modelsavepath)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            modelsavepath + "/{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.weights.h5",
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
        model_json = self.model.to_json()
        with open(modelsavepath + "/model.json", "w") as json_file:
            json_file.write(model_json)
        print("Saved model to ", modelsavepath)

    def train(self, modelsavepath, training_batch_generator, validation_batch_generator):
        """ Train the model - use image generator """

        if not os.path.exists(modelsavepath):
            os.makedirs(modelsavepath)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(modelsavepath + "/{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.weights.h5", monitor=self.LearningDict['monitor'], verbose=1, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
        early = tf.keras.callbacks.EarlyStopping(monitor=self.LearningDict['monitor'], min_delta=0, patience=self.LearningDict['patience'], verbose=1, mode='auto')

        epochs = self.LearningDict['epochs']
        self.history = self.model.fit(training_batch_generator,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=validation_batch_generator,
                                      callbacks=[checkpoint, early])

        model_json = self.model.to_json()
        with open(modelsavepath + "/model.json", "w") as json_file:
            json_file.write(model_json)
        print("Saved model to ", modelsavepath)
