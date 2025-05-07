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

class BalancedGenerator(tf.keras.utils.Sequence):
    # This class will generate images ready for the network.
    # It will sample images at random such that the number of images in each class is roughly the same.
    # It will then reshape into the minimum size and apply a random shift before cropping to (imgwidth,imgheight)
    def __init__(self, image_filenames, labels, batch_size, num_batches, traindir, imgheight, imgwidth, channels):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.train_dir = traindir
        self.imgheight = imgheight
        self.imgwidth = imgwidth
        self.channels = channels
        classCounts = dict(zip(*np.unique([str(l) for l in labels], return_counts=True))) # zips the labels to the counts
        relative_likelihoods = np.array([1/classCounts[str(l)] for l in labels])
        self.image_likelihoods = relative_likelihoods / np.sum(relative_likelihoods)
    
    def applyPaddingAndAddChannels(self, array):
        if len(array.shape) == 2:
            array = np.expand_dims(array, axis=-1)

        h, w, c = array.shape
        if h < self.imgheight:
            pad_h = self.imgheight - h
            array = np.concatenate([array, np.zeros((pad_h, w, c))], axis=0)
        if w < self.imgwidth:
            pad_w = self.imgwidth - w
            array = np.concatenate([array, np.zeros((array.shape[0], pad_w, c))], axis=1)

        if c < self.channels:
            pad_c = self.channels - c
            array = np.concatenate([array, np.zeros((*array.shape[:2], pad_c))], axis=-1)
        elif c > self.channels:
            array = array[:, :, :self.channels]

        return array
    
    def applyRandomShift(self, array):
        h, w, c = array.shape
        start_row = np.random.randint(0, array.shape[0] - self.imgheight + 1) if array.shape[0] > self.imgheight else 0
        start_col = np.random.randint(0, array.shape[1] - self.imgwidth + 1) if array.shape[1] > self.imgwidth else 0
        array = array[start_row:start_row + self.imgheight, start_col:start_col + self.imgwidth]
        return array

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        choices = np.random.choice(len(self.image_filenames),self.batch_size, p=self.image_likelihoods)
        batch_x_files = [self.image_filenames[c] for c in choices]
        batch_x = [self.applyRandomShift(self.applyPaddingAndAddChannels(np.load(file_name))) for file_name in batch_x_files]
        batch_y = [self.labels[c] for c in choices]
        return np.array(batch_x), np.array(batch_y)

class ModelTester:
    # This will take a folder containing spectrograms and train different models on it.
    def __init__(self, imageFolder, outputFolder, imgsize, validationShare, batchSize, numBatches, epochs):
        self.imageFolder = imageFolder
        self.outputFolder = outputFolder
        self.imgsize = imgsize
        self.validationShare = validationShare
        self.batchSize = batchSize
        self.numBatches = numBatches
        self.epochs = epochs

        # Split the segments into a training and validation set
        classLabels = os.listdir(self.imageFolder)
        self.nclasses = len(classLabels)
        X_train_filenames, y_train, X_val_filenames, y_val = [], [], [], []

        for classLabel in classLabels:
            segments = shuffle(os.listdir(os.path.join(self.imageFolder,classLabel)))
            splitPoint = int(len(segments)*(1-validationShare))
            trainSegments, testSegments = segments[:splitPoint], segments[splitPoint:]
            for segment in trainSegments:
                for file in os.listdir(os.path.join(self.imageFolder,classLabel,segment)):
                    X_train_filenames.append(os.path.join(self.imageFolder, classLabel, segment, file))
                    y_train.append(int(classLabel))
            for segment in testSegments:
                for file in os.listdir(os.path.join(self.imageFolder,classLabel,segment)):
                    X_val_filenames.append(os.path.join(self.imageFolder, classLabel, segment, file))
                    y_val.append(int(classLabel))

        y_train = tf.keras.utils.to_categorical(np.array(y_train), self.nclasses).astype(int)
        y_val = tf.keras.utils.to_categorical(np.array(y_val), self.nclasses).astype(int)

        X_train_filenames, y_train = shuffle(X_train_filenames, y_train)
        X_val_filenames, y_val = shuffle(X_val_filenames, y_val)
        
        # make the generators
        self.training_batch_generator = BalancedGenerator(X_train_filenames, y_train, self.batchSize, self.numBatches, self.imageFolder, self.imgsize[0], self.imgsize[1], 1)
        self.validation_batch_generator = BalancedGenerator(X_val_filenames, y_val, self.batchSize, int(self.numBatches*self.validationShare), self.imageFolder, self.imgsize[0], self.imgsize[1], 1)
    
    def loadModel(self, modelName):
        if modelName=="CNN":
            model = NNModels.CNNModel(self.imgsize[0],self.imgsize[1],self.nclasses)
        elif modelName=="ViT":
            model = NNModels.AudioSpectogramTransformer(imageHeight=self.imgsize[0], imageWidth=self.imgsize[1], patchSize=16, patchOverlap=0, outputDim=self.nclasses, embeddingDim=192, transformerLayers=12, transformerHeads=3, transformerNNDim=768)
        elif modelName=="PretrainedViT":
            if not (self.imgsize[0]==224 and self.imgsize[1]==224):
                print("Error! PretrainedViT needs an exact input size of 224 x 224")
            model = NNModels.AudioSpectogramTransformer(imageHeight=224, imageWidth=224, patchSize=16, patchOverlap=0, outputDim=self.nclasses, embeddingDim=192, transformerLayers=12, transformerHeads=3, transformerNNDim=768)
            model.load_weights("pre-trained_ViT_weights.h5", by_name=True, skip_mismatch=True)
        elif modelName=="EfficientNetV2S-LSTMHead":
            if not (self.imgsize[0]==224 and self.imgsize[1]==224):
                print("Error! EfficientNetV2S-LSTMHead needs an exact input size of 224 x 224")
            inputs = tf.keras.Input(shape=(224, 224, 1))
            x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])  # shape becomes (224, 224, 3)
            base_model = EfficientNetV2S(include_top=False, weights='imagenet')
            base_model.trainable = False
            x = base_model(x, training=False)
            x = tf.keras.layers.Reshape((-1, 1280))(x)  # Turns (7,7,1280) → (49, 1280)
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256))(x)
            outputs = tf.keras.layers.Dense(self.nclasses, activation='softmax')(x)
            model = tf.keras.Model(inputs, outputs)
            model.summary()
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model

    def runModel(self, modelName):
        model = self.loadModel(modelName)
        history = model.fit(self.training_batch_generator, epochs=self.epochs, verbose=1, validation_data=self.validation_batch_generator)
        os.makedirs(self.outputFolder,exist_ok=True)
        with open(os.path.join(self.outputFolder,modelName+"History.json"), "w") as f:
            json.dump(history.history, f)

class Plotter:
    def __init__(self, outputFolder):
        self.outputFolder = outputFolder
    
    def plotHistories(self):
        # Make plots
        histories = {}
        for filename in os.listdir(self.outputFolder):
            if filename.endswith("History.json"):
                modelType = filename.replace("History.json", "")
                with open(os.path.join(self.outputFolder, filename), "r") as f:
                    histories[modelType] = json.load(f)

        plt.figure()
        colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        for modelType, history in histories.items():
            color = next(colors)
            plt.plot(history["accuracy"], label=f"{modelType} - train", color=color)
            plt.plot(history["val_accuracy"], label=f"{modelType} - val", color=color, linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training and Validation Accuracy")
        plt.show()

if __name__=="__main__":
    imageFolder = "/home/giotto/Desktop/AviaNZ/Sound Files/learning/imageDataset"
    testOutputFolder = "/home/giotto/Desktop/AviaNZ/Sound Files/learning/modelTestingOutputs"
    imgsize = [224,224] # this must be [224,224] for the ViT and EfficientNetV2S models
    validationShare = 0.2 # what share of segments are for validation
    batchSize = 32
    numBatches = 100
    epochs = 20

    tester = ModelTester(imageFolder, testOutputFolder, imgsize, validationShare, batchSize, numBatches, epochs)
    tester.runModel("CNN")
    tester.runModel("ViT")
    tester.runModel("PretrainedViT")
    tester.runModel("EfficientNetV2S-LSTMHead")

    plotter = Plotter(testOutputFolder)
    plotter.plotHistories()