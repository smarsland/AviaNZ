import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
import json
import numpy as np

# spectrogram data
with open("sgramdata.json") as f:
    data = json.load(f)
print(np.shape(data))

sg=np.ndarray(shape=(np.shape(data)[0],30,128), dtype=float)
target = np.zeros((np.shape(data)[0], 1))
for i in range(np.shape(data)[0]):
    maxg = np.max(data[i][0][:])
    sg[i][:] = data[i][0][:]/maxg
    target[i][0] = data[i][-1]

# x = sg.reshape(np.shape(data)[0], 3840)

# randomly choose 60% train data and keep the rest for testing
idxs = np.random.permutation(np.shape(sg)[0])
x_train = sg[idxs[0:int(len(idxs)*0.6)]]
y_train = target[idxs[0:int(len(idxs)*0.6)]]
x_test = sg[idxs[int(len(idxs)*0.6):]]
y_test = target[idxs[int(len(idxs)*0.6):]]

train_images = x_train.reshape(x_train.shape[0], 30, 128, 1)
test_images = x_test.reshape(x_test.shape[0], 30, 128, 1)
input_shape = (30, 128, 1)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_labels = tensorflow.keras.utils.to_categorical(y_train, 8)
test_labels = tensorflow.keras.utils.to_categorical(y_test, 8)

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

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,
                    batch_size=32,
                    epochs=10,
                    verbose=2,
                    validation_data=(test_images, test_labels))

score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])