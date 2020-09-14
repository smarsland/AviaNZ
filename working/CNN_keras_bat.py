#Script from Nirosha Priyadarshani
#Changes by Virginia Listanti

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
import json
import numpy as np
#sklearn.manifold

#Virginia: different train and test sets

# spectrogram data train
with open("D:\Desktop\Documents\Work\Data\Bat\BAT\CNN experiment\TRAIN\sgramdata.json") as f:
    data_train = json.load(f)
print(np.shape(data_train))

sg_train=np.ndarray(shape=(np.shape(data_train)[0],6, 512), dtype=float) #changed image dimensions
target_train = np.zeros((np.shape(data_train)[0], 1)) #label train
for i in range(np.shape(data_train)[0]):
    maxg = np.max(data_train[i][0][:])
    sg_train[i][:] = data_train[i][0][:]/maxg
    target_train[i][0] = data_train[i][-1]

# spectrogram data test
with open("D:\Desktop\Documents\Work\Data\Bat\BAT\CNN experiment\TEST\sgramdata.json") as f:
    data_test = json.load(f)
print(np.shape(data_test))

sg_test=np.ndarray(shape=(np.shape(data_test)[0], 6, 512), dtype=float) #changed image dimensions
target_test = np.zeros((np.shape(data_test)[0], 1)) #labels test
for i in range(np.shape(data_test)[0]):
    maxg = np.max(data_test[i][0][:])
    sg_test[i][:] = data_test[i][0][:]/maxg
    target_test[i][0] = data_test[i][-1]

# x = sg.reshape(np.shape(data)[0], 3840)
    
# Using different train and test datasets
x_train = sg_train
y_train = target_train
x_test = sg_test
y_test = target_test

train_images = x_train.reshape(x_train.shape[0], 6, 512, 1) #changed image dimensions
test_images = x_test.reshape(x_test.shape[0], 6, 512, 1)
input_shape = (6, 512, 1)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_labels = tensorflow.keras.utils.to_categorical(y_train, 8)
test_labels = tensorflow.keras.utils.to_categorical(y_test, 8)   #change this to set labels  

#(ninputs * 3) 0 0 1
#(ninputs, 2) 0 0 or 0 1 or 1 0 what we want

## randomly choose 60% train data and keep the rest for testing
#idxs = np.random.permutation(np.shape(sg)[0])
#x_train = sg[idxs[0:int(len(idxs)*0.6)]]
#y_train = target[idxs[0:int(len(idxs)*0.6)]]
#x_test = sg[idxs[int(len(idxs)*0.6):]]
#y_test = target[idxs[int(len(idxs)*0.6):]]
#
#train_images = x_train.reshape(x_train.shape[0], 30, 128, 1)
#test_images = x_test.reshape(x_test.shape[0], 30, 128, 1)
#input_shape = (30, 128, 1)
#
#train_images = train_images.astype('float32')
#test_images = test_images.astype('float32')
#
#train_labels = tensorflow.keras.utils.to_categorical(y_train, 8)
#test_labels = tensorflow.keras.utils.to_categorical(y_test, 8)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# 64 3x3 kernels
model.add(Conv2D(64, (3, 3), activation='relu'))
# Reduce by taking the max of each 2x2 block
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout to avoid overfitting
model.add(Dropout(0.25))
# Flatten the results to one dimension for passing into our final layer
model.add(Flatten())
# A hidden layer to learn with
model.add(Dense(128, activation='relu'))
# Another dropout
model.add(Dropout(0.5))
# Final categorization from 0-9 with softmax
model.add(Dense(8, activation='softmax'))

model.summary()

#set the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#look at loss functions 
#check keras.io
#train the model
history = model.fit(train_images, train_labels,
                    batch_size=32,
                    epochs=20,
                    verbose=2,
                    validation_data=(test_images, test_labels))

#evaluate performace
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])