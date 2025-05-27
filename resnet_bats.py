
import os,math
import tensorflow as tf
import matplotlib.pyplot as pl
from tensorflow.keras import datasets, layers, models, losses, Model
import numpy as np
import SignalProc

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.transform import resize

import NNModels
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import custom_object_scope

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
physical_devices = tf.config.list_physical_devices('GPU') 
print(physical_devices)
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
def trainNet(dirName,netName,imgWidth=128,imgHeight=64,incr=32,nLearnLayer=500,nClasses=3,batchSize=32,epochs=50,trainAmount=0.6,channels=3):
        # Use pre-trained networks and the generator to get the images
        modelsavepath='.'

        # Get the pre-trained network
        #PARAMS
        base_model = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,channels))
        #base_model = tf.keras.applications.ResNet50V2(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,channels))
        #base_model = tf.keras.applications.ResNet152(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,channels))
        # Inception needs at least 75 x 75
        #base_model = tf.keras.applications.InceptionV3(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,channels))
        #base_model = tf.keras.applications.InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,channels))
        # Xception needs at least 71 x 71
        #base_model = tf.keras.applications.Xception(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,channels))

        for layer in base_model.layers:
                layer.trainable = False
        
        # Add output layers
        # ?? Choice of activations?
        newlayer = layers.Flatten()(base_model.output)
        newlayer = layers.Dense(nLearnLayer, activation='relu')(newlayer) 
        predictions = layers.Dense(nClasses, activation = 'softmax')(newlayer)
        
        # Get the training data from images
        filenames = []
        labels = []
        #print(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(incr)+"_"+str(imgHeight)))

        files = os.listdir(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)))
        files.sort()
        for file in files:
            if file.endswith('.npy'):
                filenames.append(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr), file))
            #if file.endswith('.txt'):
                #labels = np.loadtxt(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr),file))

        ext = "img"+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)+"/label_tensorflow.txt"
        labels = np.loadtxt(os.path.join(dirName,ext))
        labels.astype('int')
        labels = tf.keras.utils.to_categorical(labels, 3)
        filenames, labels = shuffle(filenames, labels)

        X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(filenames, labels, test_size=1-trainAmount, random_state=1)
        X_val_filenames, x_test_filenames, y_val, y_test = train_test_split(X_val_filenames,y_val, test_size=0.5, random_state=1)
        training_batch_generator = CustomGenerator(X_train_filenames, y_train, batchSize, os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)), imgWidth, imgHeight, channels)
        validation_batch_generator = CustomGenerator(X_val_filenames, y_val, batchSize, os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)), imgWidth, imgHeight, channels)

        #a,b = training_batch_generator.__getitem__(0)

        # Expand the dimensionality to 3 by duplication
        #x = tf.expand_dims(x, axis=3, name=None)
        #x = tf.repeat(x, 3, axis=3)

        # Train the output layers
        # ?? Choice of optimiser and metrics?
        head_model = Model(inputs = base_model.input, outputs = predictions)
        head_model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

        if not os.path.exists(modelsavepath):
            os.makedirs(modelsavepath)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            modelsavepath + "/{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.weights.h5",
            monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto',
            save_freq='epoch')

        early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')
        history = head_model.fit(training_batch_generator, batch_size=batchSize, epochs=epochs, verbose=1, validation_data=validation_batch_generator,callbacks=[checkpoint, early],shuffle=True)

        fig, axs = pl.subplots(2, 1, figsize=(15,15))
        axs[0].plot(history.history['loss'])
        axs[0].plot(history.history['val_loss'])
        axs[0].title.set_text('Training Loss vs Validation Loss')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend(['Train','Val'])
        axs[1].plot(history.history['accuracy'])
        axs[1].plot(history.history['val_accuracy'])
        axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend(['Train', 'Val'])
        pl.savefig(netName+'learning.png')
        print("Training complete")

        head_model.save(netName+"_"+str(imgWidth)+"_"+str(imgHeight)+"_"+str(batchSize)+"_"+str(nLearnLayer))
        #head_model.save(dirName+'.net')

        # Test the network
        x_test = np.array([np.load(file_name) for file_name in x_test_filenames])
        x_test = tf.expand_dims(x_test, axis=3, name=None)
        x_test = tf.repeat(x_test, channels, axis=3)
        head_model.evaluate(x_test, y_test)

        print(head_model.predict(x_test),y_test)
        #head_model.summary()

        # Fine tuning
        #base_model.trainable = True

        #base_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss = losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        #head_model.summary()
        #head_model.fit(training_batch_generator,batch_size=batchSize,epochs=epochs,verbose=1,validation_data=validation_batch_generator)

        #head_model.evaluate(x_test, y_test)
        #print(head_model.predict(x_test),y_test)
        #base_model.save(netName+"_"+str(imgWidth)+"_"+str(imgHeight)+"_"+str(batchSize)+"_"+str(nLearnLayer)+"all")


def checkNet(dirName,netName,imgWidth=128,imgHeight=64,nLearnLayer=500,batchSize=32,epochs=10):
        # Load tuned or untuned model...
        #base_model = tf.keras.applications.ResNet50V2(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,3))
        #base_model = tf.keras.applications.InceptionV3(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,3))
        #base_model = tf.keras.applications.InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,3))
        base_model = tf.keras.models.load_model(netName+'_'+str(imgWidth)+'_'+str(nLearnLayer)+'_'+str(batchSize)+"_"+str(imgHeight)+"all")
        head_model = tf.keras.models.load_model(netName+'_'+str(imgWidth)+'_'+str(nLearnLayer)+'_'+str(batchSize)+"_"+str(imgHeight))

       # Get the training data from images
        filenames = []
        labels = []

        files = os.listdir(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)))
        files.sort()
        for file in files:
            if file.endswith('.npy'):
                filenames.append(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr), file))
            if file.endswith('.txt'):
                labels = np.loadtxt(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr),file))
        labels.astype('int')
        filenames, labels = shuffle(filenames, labels)

        X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(filenames, labels, test_size=0.4, random_state=1)
        X_val_filenames, x_test_filenames, y_val, y_test = train_test_split(X_val_filenames,y_val, test_size=0.5, random_state=1)
        training_batch_generator = CustomGenerator(X_train_filenames, y_train, batchSize, os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)), imgWidth, imgHeight, channels)
        validation_batch_generator = CustomGenerator(X_val_filenames, y_val, batchSize, os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)), imgWidth, imgHeight, channels)

        # Test the network
        x_test = np.array([np.load(file_name) for file_name in x_test_filenames])
        x_test = tf.expand_dims(x_test, axis=3, name=None)
        x_test = tf.repeat(x_test, 3, axis=3)
        head_model.evaluate(x_test, y_test)

        # Fine tuning
        # Size bug here
        #base_model.trainable = True

        #base_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss = losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        #head_model.fit(training_batch_generator,batch_size=batchSize,epochs=epochs,verbose=1,validation_data=validation_batch_generator)

        #head_model.evaluate(x_test, y_test)
        #base_model.save(netName+"_"+str(imgWidth)+"_"+str(nLearnLayer)+"_"+str(batchSize)+"base")

def useNet(netName,dirName,imgWidth,imgHeight,incr,nClasses,thr1,thr2,channels=3):

        #if netName.find('sIR_')>-1:
            #base_model = tf.keras.applications.InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,3))
        #elif netName.find('sI_')>-1:
            #base_model = tf.keras.applications.InceptionV3(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,3))
        #elif netName.find('sR_')>-1:
            #base_model = tf.keras.applications.ResNet50V2(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,3))
        #elif netName.find('sRR_')>-1:
            #base_model = tf.keras.applications.ResNet152(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,3))
        #elif netName.find('sX_')>-1:
            #base_model = tf.keras.applications.Xception(weights = 'imagenet', include_top = False, input_shape = (imgWidth,imgHeight,3))
        #elif netName.find('sNP_')>-1:
            #print("todo")
        #else:
            #print("Not recognised")

        if netName[-5:]=='NP.h5':
                print("Loading original filter")
                print(os.path.join('/home/marslast/.avianz/Filters/',netName))
                json_file = open(os.path.join('/home/marslast/.avianz/Filters/',netName[:-3]) + '.json', 'r')
                loadedModelJson = json_file.read()
                json_file.close()
                with custom_object_scope(NNModels.customObjectScopes):
                    try:
                        model = model_from_json(loadedModelJson)
                    except Exception as e:
                        print(e)
                        print('Error in loading model from json. Are you linking all custom layers in NNModels.customObjectScopes?')
                        return False
                head_model = models.load_model(os.path.join('/home/marslast/.avianz/Filters/',netName))
                imgHeight=64
                imgWidth=343
        else:
                head_model = models.load_model(netName)

        sp = SignalProc.SignalProc(1024,512)

        # csv file for output
        #outf = open('results'+netName+'.csv','w')

        for root, dirs, files in os.walk(str(dirName)):
                for filename in files:
                    if filename.lower().endswith('.bmp'):
                        filename = os.path.join(root, filename)
        
                        # check if file not empty
                        if os.stat(filename).st_size > 1000:
                        # check if file is formatted correctly
                            with open(filename, 'br') as f:
                                if f.read(2) == b'BM':
                                    sp.readBmp(filename,repeat=False,rotate=False,silent=True)
                                    # Should probably only do this between clicks?
                                    # Or not?!
                                    res = ClickSearch(sp.sg,sp.audioFormat.sampleRate())
                                    #print(filename,np.shape(sp.sg))
                                    if res is not None:
                                        starts = range(res[0], res[1]-imgWidth, incr)
                                    else:
                                        starts = range(0, np.shape(sp.sg)[1] - imgWidth, incr)
                                    print(filename,len(starts))
                                    if len(starts)>0:
                                        x = np.zeros((len(starts),imgWidth,imgHeight))
                                        for s in range(len(starts)):
                                            x[s,:,:64] = sp.sg[:,starts[s]:starts[s]+imgWidth].T
                                        if channels>1:
                                            x = tf.expand_dims(x, axis=3, name=None)
                                            x = tf.repeat(x, channels, axis=3)
                                        else:
                                            x = resize(x,(len(starts),imgWidth,imgHeight,channels))
                                            if netName[-5:]=='NP.h5':
                                                x = x.transpose(0,2,1,3)
                                        y = head_model.predict(x)
                                        print(y)
                                        np.save(filename+'_'+netName,y)
                                    #print(y)
                                    #outf.write(filename)
                                    #outf.write(np.array_str(y))
                                    
                                    # Average of best 5 of each class
                                    # ????? How to do ??? Threshold for classes, otherwise noise?
                                    #means = np.zeros(nClasses)
                                    #for c in range(nClasses):
                                            #means[c] = np.mean(np.partition(y[:,c],-5)[-5:])
                                    #outf.write(filename)
                                    #outf.write(np.array_str(means))
                                    #if means[0] > thr1:
                                        #outf.write("LT")
                                    #elif means[0] > thr2:
                                        #outf.write("LT?")
                                    #if means[1] > thr1:
                                        #outf.write("ST")
                                    #elif means[1] > thr2:
                                        #outf.write("ST?")
        
                                    
        #outf.close()

def findThreshold(netID,dirName,imgWidth,imgHeight,incr,nClasses,thr1,thr2):

        for root, dirs, files in os.walk(str(dirName)):
                for filename in files:
                    if filename.find(netID)>-1:
                        filename = os.path.join(root, filename)
                        y = np.loadtxt(filename)
                        if np.ndim(y)>1:
                            means = np.zeros(nClasses)
                            # TODO: param
                            nInclude = min(np.shape(y)[0],5)
                            for c in range(nClasses):
                                means[c] = np.mean(np.partition(y[:,c],-nInclude)[-nInclude:])
                            print(filename, means)
                        else:          
                            print(filename, "ouch")
        

def trainNet_VGG(dirName,netName,imgWidth=224,imgHeight=64,incr=32,nLearnLayer=500,nClasses=3,batchSize=32,epochs=50,trainAmount=0.6,channels=3):
    modelsavepath='.'
    # This is tha basic VGG16 network
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(input_shape=(imgWidth,imgHeight,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
    model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
    model.add(tf.keras.layers.Dense(units=nClasses, activation="softmax"))
    model.summary()

    model.compile(loss = losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    # Get the training data from images
    filenames = []
    labels = []

    files = os.listdir(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)))
    files.sort()
    for file in files:
        if file.endswith('.npy'):
            filenames.append(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr), file))
        #if file.endswith('.txt'):
            #labels = np.loadtxt(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr),file))
    ext = "img"+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)+"/lab2.txt"
    labels = np.loadtxt(os.path.join(dirName,ext))
    labels.astype('int')
    #labels = np.loadtxt(os.path.join(dirName,"img128_64_32/lab2.txt"))
    #labels = int(labels[-1])
    #print(labels)

    print(np.shape(filenames),np.shape(labels))
    filenames, labels = shuffle(filenames, labels)

    X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(filenames, labels, test_size=1-trainAmount, random_state=1)
    X_val_filenames, x_test_filenames, y_val, y_test = train_test_split(X_val_filenames,y_val, test_size=0.5, random_state=1)
    training_batch_generator = CustomGenerator(X_train_filenames, y_train, batchSize, os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)), imgWidth, imgHeight, channels)
    validation_batch_generator = CustomGenerator(X_val_filenames, y_val, batchSize, os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)), imgWidth, imgHeight, channels)

    if not os.path.exists(modelsavepath):
        os.makedirs(modelsavepath)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        modelsavepath + "/{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.weights.h5",
        monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto',
        save_freq='epoch')
    early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')
    model.history = model.fit(training_batch_generator, batch_size=batchSize, epochs=epochs, verbose=1, validation_data=validation_batch_generator, callbacks=[checkpoint, early], shuffle=True)

    fig, axs = pl.subplots(2, 1, figsize=(15,15))
    axs[0].plot(model.history.history['loss'])
    axs[0].plot(model.history.history['val_loss'])
    axs[0].title.set_text('Training Loss vs Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend(['Train','Val'])
    axs[1].plot(model.history.history['accuracy'])
    axs[1].plot(model.history.history['val_accuracy'])
    axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(['Train', 'Val'])
    pl.savefig(netName+'learning.png')

    model.save(netName+"_"+str(imgWidth)+"_"+str(imgHeight)+"_"+str(batchSize)+"_"+str(nLearnLayer))


def trainNet_NP(dirName,netName,imgWidth=343,imgHeight=64,incr=32,nLearnLayer=500,nClasses=3,batchSize=32,epochs=50,trainAmount=0.6,channels=1):
    tf.keras.backend.set_image_data_format('channels_last')
    modelsavepath='.'

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=[imgWidth, imgHeight, 1], padding='Same'))
    model.add(tf.keras.layers.Conv2D(64, (7, 7), activation='relu', padding='Same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), padding='Same'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='Same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='Same'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='Same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='Same'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='Same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='Same'))
    model.add(tf.keras.layers.Dropout(0.2))
    # Flatten the results to one dimension for passing into our final layer
    model.add(tf.keras.layers.Flatten())
    # A hidden layer to learn with
    model.add(tf.keras.layers.Dense(nLearnLayer, activation='relu'))
    # Another dropout
    model.add(tf.keras.layers.Dropout(0.5))
    # Final categorization from 0-ct+1 with softmax
    model.add(tf.keras.layers.Dense(nClasses, activation='softmax'))
    model.summary()

    model.compile(loss = losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Get the training data from images
    filenames = []
    labels = []
    print(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(incr)+"_"+str(imgHeight)))

    files = os.listdir(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)))
    files.sort()
    for file in files:
        if file.endswith('.npy'):
            filenames.append(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr), file))
        #if file.endswith('.txt'):
            #labels = np.loadtxt(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr),file))
        
    ext = "img"+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)+"/lab2.txt"
    labels = np.loadtxt(os.path.join(dirName,ext))
    labels.astype('int')
    #labels = tf.keras.utils.to_categorical(labels, 3)
    print(len(filenames))
    filenames, labels = shuffle(filenames, labels)

    X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(filenames, labels, test_size=1-trainAmount, random_state=1)
    X_val_filenames, x_test_filenames, y_val, y_test = train_test_split(X_val_filenames,y_val, test_size=0.5, random_state=1)
    training_batch_generator = CustomGenerator(X_train_filenames, y_train, batchSize, os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)), imgWidth, imgHeight, channels)
    validation_batch_generator = CustomGenerator(X_val_filenames, y_val, batchSize, os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr)), imgWidth, imgHeight, channels)

    if not os.path.exists(modelsavepath):
        os.makedirs(modelsavepath)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        modelsavepath + "/{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.weights.h5",
        monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto',
        save_freq='epoch')
    early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')
    model.history = model.fit(training_batch_generator, batch_size=batchSize, epochs=epochs, verbose=1, validation_data=validation_batch_generator, callbacks=[checkpoint, early], shuffle=True)

    fig, axs = pl.subplots(2, 1, figsize=(15,15))
    axs[0].plot(model.history.history['loss'])
    axs[0].plot(model.history.history['val_loss'])
    axs[0].title.set_text('Training Loss vs Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend(['Train','Val'])
    axs[1].plot(model.history.history['accuracy'])
    axs[1].plot(model.history.history['val_accuracy'])
    axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(['Train', 'Val'])
    pl.savefig(netName+'learning.png')


    model.save(netName+"_"+str(imgWidth)+"_"+str(imgHeight)+"_"+str(batchSize)+"_"+str(nLearnLayer))

def trainNet_loadAllData(filename):
        # Load the data matrix
        x = np.load(filename)
        y = x['y']
        x = x['x']

        # Randomise order
        shuff = np.arange(np.shape(x)[0])
        np.random.shuffle(shuff)
        x = x[shuff,:,:]
        y = y[shuff]

        # Expand the dimensionality to 3 by duplication
        x = tf.expand_dims(x, axis=3, name=None)
        x = tf.repeat(x, 3, axis=3)

        # Separate into training, validation, testing
        # PARAM: amounts!
        ndatapoints = np.shape(x)[0]
        nval = int(0.2*ndatapoints)
        ntest = int(0.2*ndatapoints)
        
        x_val = x[-nval:,:,:,:]
        y_val = y[-nval:]
        x_test = x[-nval-ntest:-nval,:,:,:]
        y_test = y[-nval-ntest:-nval]
        x = x[:-nval-ntest,:,:,:]
        y = y[:-nval-ntest]

        # Get the pre-trained network
        base_model = tf.keras.applications.ResNet50V2(weights = 'imagenet', include_top = False, input_shape = (np.shape(x)[1],np.shape(x)[2],3))
        #base_model = tf.keras.applications.ResNet152(weights = 'imagenet', include_top = False, input_shape = (np.shape(x)[1],np.shape(x)[2],3))
        # Inception needs at least 75 x 75
        #base_model = tf.keras.applications.InceptionV3(weights = 'imagenet', include_top = False, input_shape = (np.shape(x)[1],np.shape(x)[2],3))
        #base_model = tf.keras.applications.InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = (np.shape(x)[1],np.shape(x)[2],3))
        # Xception needs at least 71 x 71
        #base_model = tf.keras.applications.Xception(weights = 'imagenet', include_top = False, input_shape = (np.shape(x)[1],np.shape

        for layer in base_model.layers:
                layer.trainable = False
        
        # Add output layers
        # PARAMS here
        newlayer = layers.Flatten()(base_model.output)
        newlayer = layers.Dense(1000, activation='relu')(newlayer) 
        # PARAM here
        predictions = layers.Dense(3, activation = 'softmax')(newlayer)
        
        # Train the output layers
        #PARAMS here
        head_model = Model(inputs = base_model.input, outputs = predictions)
        head_model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        history = head_model.fit(x, y, batch_size=12, epochs=10, validation_data=(x_val, y_val)) 

        head_model.evaluate(x_test, y_test)
        filenameNoExtension = filename.rsplit('.', 1)[0]
        head_model.save(filenameNoExtension)

        # Fine tuning
        # Size bug here
        #base_model.trainable = True

        #base_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss = losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        #base_model.fit(x,y,batch_size=12,epochs=10,validation_data=(x_val,y_val))

        #base_model.evaluate(x_test, y_test)
        #base_model.save(filename[:-4]+'x')

class CustomGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, traindir, imgWidth, imgHeight, channels):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.train_dir = traindir
        self.imgHeight = imgHeight
        self.imgWidth = imgWidth
        self.channels = channels

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        if self.channels>1:
            x = np.array([np.load(file_name) for file_name in batch_x])
            x = tf.expand_dims(x, axis=3, name=None)
            x = tf.repeat(x, self.channels, axis=3)
        else:
            x = np.array([resize(np.load(file_name), (self.imgWidth, self.imgHeight, self.channels)) for file_name in batch_x])
        return x, np.array(batch_y)
        # return np.array([resize(imread(os.path.join(self.train_dir , str(file_name))), (self.imgheight, self.imgwidth, self.channels)) for file_name in batch_x]) / 255.0, np.array(batch_y)
        #return np.array([resize(np.load(file_name), (self.imgheight, self.imgwidth, self.channels)) for file_name in batch_x]), np.array(batch_y)
        #return np.array([resize(np.load(file_name), (self.imgheight, self.imgwidth, self.channels)) for file_name in batch_x]), np.array(batch_y)


imgWidth=192
incr=32
nLearnLayer=50
nClasses=3
batchSize=32
epochs=2
trainAmount=0.8
imgHeight=64

trainNet('/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BatTraining/Check','batsVG_pre',imgWidth=imgWidth,imgHeight=imgHeight,incr=incr,nLearnLayer=nLearnLayer,nClasses=nClasses,batchSize=batchSize,epochs=epochs,trainAmount=trainAmount,channels=3)
#trainNet('/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/New_Train_Datasets/Train_5','batsVG_pre',imgWidth=imgWidth,imgHeight=imgHeight,incr=incr,nLearnLayer=nLearnLayer,nClasses=nClasses,batchSize=batchSize,epochs=epochs,trainAmount=trainAmount,channels=3)
#useNet('batsVG_343_64_32_200','/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/Test_dataset/Test/',imgWidth,imgHeight,incr,nClasses,0.8,0.6) #R1/Bat/20191105/20191105_235907.bmp',0.8,0.6) # LT
#useNet('batsVG_343_64_32_200all','/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/Test_dataset/Test/',imgWidth,imgHeight,incr,nClasses,0.8,0.6) #R1/Bat/20191105/20191105_235907.bmp',0.8,0.6) # LT

#trainNet_NP('/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BatTraining/Full','batsNP',imgWidth=imgWidth,imgHeight=imgHeight,incr=incr,nLearnLayer=nLearnLayer,nClasses=nClasses,batchSize=batchSize,epochs=epochs,trainAmount=trainAmount,channels=1)



#trainNet_NP('/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/New_Train_Datasets/Train_5','batsNP',imgWidth=imgWidth,imgHeight=imgHeight,incr=incr,nLearnLayer=nLearnLayer,nClasses=nClasses,batchSize=batchSize,epochs=epochs,trainAmount=trainAmount,channels=1)
#useNet('batsNP_343_64_32_200','/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/Test_dataset/Test/',imgWidth,imgHeight,incr,nClasses,0.8,0.6,channels=1)

#trainNet_VGG('/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BatTraining/Full','batsVGG',imgWidth=imgWidth,imgHeight=imgHeight,incr=incr,nLearnLayer=nLearnLayer,nClasses=nClasses,batchSize=batchSize,epochs=epochs,trainAmount=trainAmount,channels=3)
#useNet('batsVGG_343_64_32_200','/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/Test_dataset/Test/',imgWidth,imgHeight,incr,nClasses,0.8,0.6,channels=3)

#findThreshold('sI_224_1000','/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/Train_Datasets/TRAIN100/',imgWidth,imgHeight,incr,nClasses,0.8,0.6)
#print("---")
#findThreshold('sIR_224_1000','/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/Train_Datasets/TRAIN100/',imgWidth,imgHeight,incr,nClasses,0.8,0.6)
#print("---")
#findThreshold('sIR_224_500','/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/Train_Datasets/TRAIN100/',imgWidth,imgHeight,incr,nClasses,0.8,0.6)
#print("---")
#findThreshold('sR_224_1000','/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/Train_Datasets/TRAIN100/',imgWidth,imgHeight,incr,nClasses,0.8,0.6)
#print("---")
#findThreshold('sRR_224_1000','/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/Train_Datasets/TRAIN100/',imgWidth,imgHeight,incr,nClasses,0.8,0.6)
#print("---")
#findThreshold('sX_224_1000','/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/Train_Datasets/TRAIN100/',imgWidth,imgHeight,incr,nClasses,0.8,0.6)
#print("---")
#findThreshold('sX_224_500','/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/Train_Datasets/TRAIN100/',imgWidth,imgHeight,incr,nClasses,0.8,0.6)
#print("---")
#findThreshold('sNP_224_1000','/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/Train_Datasets/TRAIN100/',imgWidth,imgHeight,incr,nClasses,0.8,0.6)

        
