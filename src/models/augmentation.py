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

# PyTorch-based data augmentation and training

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage.transform import resize
import json, os
import numpy as np
import librosa

from src.core import spectrogram
from src.core import config_loader
from src.core import audio_data
from src.models import architectures


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        labels_onehot = np.zeros((len(labels), len(self.calltypes) + 1))
        for i, lbl in enumerate(labels):
            labels_onehot[i, lbl] = 1

        return filenames, labels_onehot

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

        labels_onehot = np.zeros((len(labels), len(self.calltypes) + 1))
        for i, lbl in enumerate(labels):
            labels_onehot[i, lbl] = 1

        return filenames, labels_onehot

    def createArchitecture(self):
        """
        Sets self.model
        """
        if self.modelArchitecture == 'CNN':
            self.model = architectures.CNNModel(self.imageheight, self.imagewidth, len(self.calltypes)+1)
            self.model = self.model.to(self.device)
        else:
            raise ValueError("Model architecture not supported")

    def train2(self, modelsavepath):
        """ Train the model - keep all in memory """

        if not os.path.exists(modelsavepath):
            os.makedirs(modelsavepath)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters())
        
        train_data = torch.from_numpy(self.train_images).float().permute(0, 3, 1, 2)
        train_labels = torch.from_numpy(self.train_labels).float()
        val_data = torch.from_numpy(self.val_images).float().permute(0, 3, 1, 2)
        val_labels = torch.from_numpy(self.val_labels).float()
        
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(50):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, labels_idx = torch.max(batch_y, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == labels_idx).sum().item()
            
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    _, labels_idx = torch.max(batch_y, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == labels_idx).sum().item()
            
            val_acc = val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 
                          os.path.join(modelsavepath, f"{epoch:02d}-{val_loss/len(val_loader):.2f}-{val_acc:.2f}.pth"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_correct/train_total:.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        model_config = {
            'model_type': 'CNN',
            'imageHeight': self.imageheight,
            'imageWidth': self.imagewidth,
            'outputDim': len(self.calltypes) + 1
        }
        with open(os.path.join(modelsavepath, "model.json"), "w") as json_file:
            json.dump(model_config, json_file)
        print("Saved model to ", modelsavepath)

    def train(self, modelsavepath, training_batch_generator, validation_batch_generator):
        """ Train the model - use image generator """

        if not os.path.exists(modelsavepath):
            os.makedirs(modelsavepath)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters())
        
        epochs = self.LearningDict['epochs']
        monitor = self.LearningDict['monitor']
        patience = self.LearningDict['patience']
        
        best_val_metric = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for i in range(len(training_batch_generator)):
                batch_x, batch_y = training_batch_generator[i]
                batch_x = torch.from_numpy(batch_x).float().permute(0, 3, 1, 2).to(self.device)
                batch_y = torch.from_numpy(batch_y).float().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, labels_idx = torch.max(batch_y, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == labels_idx).sum().item()
            
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for i in range(len(validation_batch_generator)):
                    batch_x, batch_y = validation_batch_generator[i]
                    batch_x = torch.from_numpy(batch_x).float().permute(0, 3, 1, 2).to(self.device)
                    batch_y = torch.from_numpy(batch_y).float().to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    _, labels_idx = torch.max(batch_y, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == labels_idx).sum().item()
            
            val_acc = val_correct / val_total
            
            if 'accuracy' in monitor and val_acc > best_val_metric:
                best_val_metric = val_acc
                torch.save(self.model.state_dict(), 
                          os.path.join(modelsavepath, f"{epoch:02d}-{val_loss/len(validation_batch_generator):.2f}-{val_acc:.2f}.pth"))
                patience_counter = 0
            elif 'loss' in monitor and (best_val_metric == 0 or val_loss < best_val_metric):
                best_val_metric = val_loss
                torch.save(self.model.state_dict(), 
                          os.path.join(modelsavepath, f"{epoch:02d}-{val_loss/len(validation_batch_generator):.2f}-{val_acc:.2f}.pth"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            print(f"Epoch {epoch+1}/{epochs}: Val Loss: {val_loss/len(validation_batch_generator):.4f}, Val Acc: {val_acc:.4f}")
        
        model_config = {
            'model_type': 'CNN',
            'imageHeight': self.imageheight,
            'imageWidth': self.imagewidth,
            'outputDim': len(self.calltypes) + 1
        }
        with open(os.path.join(modelsavepath, "model.json"), "w") as json_file:
            json.dump(model_config, json_file)
        print("Saved model to ", modelsavepath)
