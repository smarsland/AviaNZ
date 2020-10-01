"""
Created on 1/10/2020

@author: Virginia Listanti

This code generate different trained CNN and test them on the folder Moira2020

Different test configuration:
- 100/250/500 files for each class each [3 diffent datasets
- window: 3pxl doubled, 7 pxl, 11 pxl
- augumentation y/n


Pipeline:
-create train dataset: clicksearch + read annotations
- 75% training, 25% validation
- agumentation(y/n)
- Cnn train
- test on dataset
- generate annotations

"""

import json
import numpy as np
import wavio
import SignalProc
import math
import Segment
import pyqtgraph as pg
import pyqtgraph.exporters as pge
import os

#tensorflow libraries
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from numpy import expand_dims

import librosa
import WaveletSegment
import WaveletFunctions

import cv2  # image -processing
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg





##MAIN
 
#Create train dataset for CNN from the results of clicksearch   
train_dir = "/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Battybats" #changed directory

#filelist
file_list_train=[]
for f in listdir(train_dir):
    if f.endswith('.bmp'):
        file_list_train.append(f)

file_number_train=len(file_list_train)

#inizializations of counters
count=0
train_featuress =[]

#search clicks
for i in range(file_number_train):
    file = file_list_train[i]
    click_label, train_featuress, count = ClickSearch(train_dir, file, train_featuress, count, window, Train=True)

#saving dataset
with open(test_dir+'\\'+test_fold +'\\sgramdata_train.json', 'w') as outfile:
    json.dump(train_featuress, outfile)

#Train CNN

data_train=train_featuress
print('check',np.shape(data_train))
sg_train_0=[]
sg_train_1=[]
sg_train_2=[]
for i in range(np.shape(data_train)[0]):
    maxg = np.max(data_train[i][0][:])
    if data_train[i][-1]==0:
        sg_train_0.append(data_train[i][0][:]/maxg)
    elif data_train[i][-1]==1:
        sg_train_1.append(data_train[i][0][:]/maxg)
    elif data_train[i][-1]==2:
        sg_train_2.append(data_train[i][0][:]/maxg)
#    target_train[i][0] = data_train[i][-1]
    
#### DATA AGUMENTATION ######################
# create image data augmentation generator for in-build
datagen1 = ImageDataGenerator(width_shift_range=0.5, fill_mode='nearest')
batch_size = 32

# class_0
samples = expand_dims(sg_train_0, np.shape(sg_train_0)[0])
# prepare iterator
it1 = datagen1.flow(samples, batch_size=batch_size)
# generate samples
batch_train_0 = it1.next()
for i in range(int(np.round((1500-np.shape(sg_train_0)[0])/batch_size))):
    batch = it1.next()
    batch_train_0 = np.vstack((batch_train_0, batch))
    
print('sg_train_0 shape', np.shape(sg_train_0))
    
print('batch_train_0 shape', np.shape(batch_train_0))
#add to original ones
batch_train_0=np.reshape(batch_train_0,np.shape(batch_train_0)[0:3])
print('batch_train_0 shape after reshape', np.shape(batch_train_0))
sg_train_0_ag=np.concatenate((sg_train_0,batch_train_0), axis=0)
#label
target_train_0=np.zeros(np.shape(sg_train_0_ag)[0])

# class_1
samples = expand_dims(sg_train_1, np.shape(sg_train_1)[0])
# prepare iterator
it1 = datagen1.flow(samples, batch_size=batch_size)
# generate samples
batch_train_1 = it1.next()
for i in range(int(np.round((1500-np.shape(sg_train_1)[0])/batch_size))):
    batch = it1.next()
    batch_train_1 = np.vstack((batch_train_1, batch))

print('sg_train_1 shape', np.shape(sg_train_1))
    
print('batch_train_1 shape', np.shape(batch_train_1))
#add to original ones
batch_train_1=np.reshape(batch_train_1,np.shape(batch_train_1)[0:3])
print('batch_train_1 shape after reshape', np.shape(batch_train_1))
#add to original ones
sg_train_1_ag=np.concatenate((sg_train_1,batch_train_1), axis=0)
#label
target_train_1=np.ones(np.shape(sg_train_1_ag)[0])
# class_2
samples = expand_dims(sg_train_2, np.shape(sg_train_2)[0])
# prepare iterator
it1 = datagen1.flow(samples, batch_size=batch_size)
# generate samples
batch_train_2 = it1.next()
for i in range(int(np.round((9000-np.shape(sg_train_2)[0])/batch_size))):
    batch = it1.next()
    batch_train_2 = np.vstack((batch_train_2, batch))
    
print('sg_train_2 shape', np.shape(sg_train_2))
    
print('batch_train_2 shape', np.shape(batch_train_2))
#add to original ones
batch_train_2=np.reshape(batch_train_2,np.shape(batch_train_2)[0:3])
print('batch_train_2 shape after reshape', np.shape(batch_train_2))
#add to original ones
sg_train_2_ag=np.concatenate((sg_train_2,batch_train_2), axis=0)
#label
target_train_2=2*np.ones(np.shape(sg_train_2_ag)[0])

#unify train dataset
sg_train=np.concatenate((sg_train_0_ag, sg_train_1_ag, sg_train_2_ag), axis=0)
#sg_train.extend(batch_train_1)
#sg_train.extend(batch_train_2)
sg_train=np.asarray(sg_train, dtype=float)
print('check sg_train shape', np.shape(sg_train))
#unify labels
target_train=np.concatenate((target_train_0, target_train_1, target_train_2), axis=0)
print('check target_train shape', np.shape(target_train))
#target_train=[target_train_0[:][0], target_train_1[:][0], target_train_2[:][0]]
#target_train.extend(target_train_1)
#target_train.extend(target_train_2)

#validation data
# randomly choose 75% train data and keep the rest as validation data
idxs = np.random.permutation(np.shape(sg_train)[0])
x_train = sg_train[idxs[0:int(len(idxs)*0.75)]]
y_train = target_train[idxs[0:int(len(idxs)*0.75)]]
x_validation = sg_train[idxs[int(len(idxs)*0.75):]]
y_validation = target_train[idxs[int(len(idxs)*0.75):]]
 
#Check: how many files I am using for training
print("-------------------------------------------")
print('Number of spectrograms', np.shape(target_train)[0]) 
print('Spectrogram used for training ',np.shape(y_train)[0])
print("Spectrograms for LT: ", np.shape(np.nonzero(y_train==0))[1])
print("Spectrograms for ST: ", np.shape(np.nonzero(y_train==1))[1])
print("Spectrograms for Noise: ", np.shape(np.nonzero(y_train==2))[1])
print('\n Spectrogram used for validation ',np.shape(y_validation)[0])
print("Spectrograms for LT: ", np.shape(np.nonzero(y_validation==0))[1])
print("Spectrograms for ST: ", np.shape(np.nonzero(y_validation==1))[1])
print("Spectrograms for Noise: ", np.shape(np.nonzero(y_validation==2))[1])

#Save training info into a file
cnn_train_info_file=test_dir+'/'+test_fold+'\CNN_train_data_info.txt'
file1=open(cnn_train_info_file,'w')
L=['Number of spectrograms = %5d \n' %np.shape(target_train)[0], 
   "Spectrogram used for training = %5d \n" %np.shape(y_train)[0],
   "Spectrograms for LT: %5d \n" %np.shape(np.nonzero(y_train==0))[1],
   "Spectrograms for ST: %5d \n" %np.shape(np.nonzero(y_train==1))[1],
   "Spectrograms for Noise: %5d \n" %np.shape(np.nonzero(y_train==2))[1],
   "\n Spectrogram used for validation = %5d \n" %np.shape(y_validation)[0],
   "Spectrograms for LT: %5d \n" %np.shape(np.nonzero(y_validation==0))[1],
   "Spectrograms for ST: %5d \n" %np.shape(np.nonzero(y_validation==1))[1],
   "Spectrograms for Noise: %5d \n" %np.shape(np.nonzero(y_validation==2))[1]]
file1.writelines(L)
file1.close()

train_images = x_train.reshape(x_train.shape[0],6, 512, 1) #changed image dimensions
validation_images = x_validation.reshape(x_validation.shape[0],6, 512, 1)
input_shape = (6, 512, 1)

train_images = train_images.astype('float32')
validation_images = validation_images.astype('float32')
    
# Detect clicks in Test Dataset and save it without labels 
    
test_dir = "/am/state-opera/home1/listanvirg/Documents/Bat_TESTS/" #changed directory
test_fold= "Test_"+str(test_cont) #Test folder where to save all the stats
os.mkdir(test_dir+ '\' + test_fold)
with open(annotation_file_test) as f:
    segments_filewise_test = json.load(f)
file_number=np.shape(segments_filewise_test)[0]



#inizializations
count_start=0
test_featuress =[]
filewise_output=[] #here we store; file, click detected (true/false), #of spectrogram, final label
TD=0
FD=0
TND=0
FND=0
control_spec=0 #this variable count the file where the click detector found a click that are not providing spectrograms for CNN

#search clicks
for i in range(file_number):
    file = segments_filewise_test[i][0]
    click_label, test_featuress, count_end = ClickSearch(test_dir, file, test_featuress, count_start, Train=False)
  
    
    
#saving dataset
with open(test_dir+'\\'+test_fold +'\\sgramdata_test.json', 'w') as outfile:
    json.dump(test_featuress, outfile)
    

    


#recover test data
#note: I am not giving target!
data_test= test_featuress
sg_test=np.ndarray(shape=(np.shape(data_test)[0],np.shape(data_test[0][0])[0], np.shape(data_test[0][0])[1]), dtype=float)
spec_id=[]
print('Number of test spectrograms', np.shape(data_test)[0])
for i in range(np.shape(data_test)[0]):
    maxg = np.max(data_test[i][0][:])
    sg_test[i][:] = data_test[i][0][:]/maxg
    spec_id.append(data_test[i][1:3])
    
#check:
print('check on spec_id', np.shape(spec_id))    

# Using different train and test datasets
#x_train = sg_train
#y_train = target_train
x_test = sg_test
#y_test = target_test


test_images = x_test.reshape(x_test.shape[0],6, 512, 1)

test_images = test_images.astype('float32')

num_labels=3 #change this variable, when changing number of labels
train_labels = tensorflow.keras.utils.to_categorical(y_train, num_labels)
validation_labels = tensorflow.keras.utils.to_categorical(y_validation, num_labels)
#test_labels = tensorflow.keras.utils.to_categorical(y_test, 8)   #change this to set labels  

accuracies=np.zeros((10,1)) #initializing accuracies array
model_paths=[] #initializing list where to stor model path
for i in range(10):
    #Build CNN architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), #I don't think this nees to be changed
                     activation='relu',
                     input_shape=input_shape))
    # 64 3x3 kernels
    model.add(Conv2D(64, (3, 3), activation='relu')) #I don't think this nees to be changed
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
    #Virginia: changed 8->3 because I have 3 classes (????)
    model.add(Dense(num_labels, activation='softmax')) #this needs to be changed if I have only 2 labeles
    
    model.summary()
    
    #set the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    #train the model
    #I am not giving validation_data
    print('Training n', i)
    
    #adding early stopping
    checkpoint = ModelCheckpoint(test_dir+ '/' + test_fold+"/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')
    history = model.fit(train_images, train_labels,
                    batch_size=32,
                    epochs=35,
                    verbose=2,
                    validation_data=(validation_images, validation_labels),
                    callbacks=[checkpoint, early],
                    shuffle=True)

        #recovering labels
    predictions =model.predict(test_images)
    #predictions is an array #imagesX #of classes which entries are the probabilities
    #for each classes
    
    filewise_output=File_label(predictions, spec_id, segments_filewise_test, filewise_output, file_number )
    
    #compare predicted_annotations with segments_filewise_test
    #evaluate metrics
        
    # inizializing
    confusion_matrix=np.zeros((7,4))
    print('Estimating metrics')
    for j in range(file_number):
        assigned_label= filewise_output[j][3]
        correct_label=segments_filewise_test[j][1]
        if correct_label==assigned_label:
            if correct_label=='Noise':
                confusion_matrix[6][3]+=1
            else:
                if correct_label=='LT':
                    confusion_matrix[0][0]+=1
                elif correct_label=='ST':
                    confusion_matrix[2][1]+=1
                elif correct_label=='Both':
                    confusion_matrix[4][2]+=1
        else:
            if correct_label=='Noise':
                if assigned_label=='LT':
                    confusion_matrix[0][3]+=1
                elif assigned_label=='LT?':
                    confusion_matrix[1][3]+=1
                elif assigned_label=='ST':
                    confusion_matrix[2][3]+=1
                elif assigned_label=='ST?':
                    confusion_matrix[3][3]+=1
                elif assigned_label=='Both':
                    confusion_matrix[4][3]+=1
                elif assigned_label=='Both?':
                    confusion_matrix[5][3]+=1
            elif assigned_label=='Noise':
                if correct_label=='LT':
                    confusion_matrix[6][0]+=1
                elif correct_label=='ST':
                    confusion_matrix[6][1]+=1
                elif correct_label=='Both':
                    confusion_matrix[6][2]+=1
            else:
                if correct_label=='LT':
                    if assigned_label=='LT?':
                        confusion_matrix[1][0]+=1
                    elif assigned_label=='ST':
                        confusion_matrix[2][0]+=1
                    elif assigned_label=='ST?':
                        confusion_matrix[3][0]+=1
                    elif assigned_label=='Both':
                        confusion_matrix[4][0]+=1
                    elif assigned_label=='Both?':
                        confusion_matrix[5][0]+=1
                elif correct_label=='ST':
                    if assigned_label=='LT':
                        confusion_matrix[0][1]+=1
                    elif assigned_label=='LT?':
                        confusion_matrix[1][1]+=1
                    elif assigned_label=='ST?':
                        confusion_matrix[3][1]+=1
                    elif assigned_label=='Both':
                        confusion_matrix[4][1]+=1
                    elif assigned_label=='Both?':
                        confusion_matrix[5][1]+=1
                elif correct_label=='Both':
                    if assigned_label=='LT':
                        confusion_matrix[0][2]+=1
                    elif assigned_label=='LT?':
                        confusion_matrix[1][2]+=1
                    elif assigned_label=='ST':
                        confusion_matrix[2][2]+=1
                    elif assigned_label=='ST?':
                        confusion_matrix[3][2]+=1
                    elif assigned_label=='Both?':
                        confusion_matrix[5][2]+=1
                     
    Recall, Precision_pre, Precision_post, Accuracy_pre1,Accuracy_pre2, Accuracy_post, TD, FPD_pre, FPD_post, FND, CoCla_pre1, CoCla_pre2, CoCla_post=metrics(confusion_matrix, file_number)

    #save reached accuracy
    accuracies[i]=Accuracy_post
    print('Accuracy reached',accuracies[i])
    #save model
    modelpath=test_dir+ '\\' + test_fold + '\\model_'+str(i)+'.h5' #aid variable
    model.save(modelpath)
    model_paths.append(modelpath)

#check what modelgave us better accuracy
index_best_model=np.argmax(accuracies) 
print('Best CNN is ', index_best_model)
print('Best accuracy reached ',accuracies[index_best_model])
modelpath=model_paths[index_best_model] 
   
#recover model
#modelpath= "D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\model_3.h5"
model=load_model(modelpath)

#recovering labels
predictions =model.predict(test_images)
#predictions is an array #imagesX #of classes which entries are the probabilities
#for each classes

filewise_output=File_label(predictions, spec_id, segments_filewise_test, filewise_output, file_number )

#compare predicted_annotations with segments_filewise_test
#evaluate metrics
    
# inizializing
comparison_annotations = []
confusion_matrix=np.zeros((7,4))
print('Estimating metrics')
for i in range(file_number):
    assigned_label= filewise_output[i][3]
    correct_label=segments_filewise_test[i][1]
    if correct_label==assigned_label:
        if correct_label=='Noise':
            confusion_matrix[6][3]+=1
        else:
            if correct_label=='LT':
                confusion_matrix[0][0]+=1
            elif correct_label=='ST':
                confusion_matrix[2][1]+=1
            elif correct_label=='Both':
                confusion_matrix[4][2]+=1
    else:
        if correct_label=='Noise':
            if assigned_label=='LT':
                confusion_matrix[0][3]+=1
            elif assigned_label=='LT?':
                confusion_matrix[1][3]+=1
            elif assigned_label=='ST':
                confusion_matrix[2][3]+=1
            elif assigned_label=='ST?':
                confusion_matrix[3][3]+=1
            elif assigned_label=='Both':
                confusion_matrix[4][3]+=1
            elif assigned_label=='Both?':
                confusion_matrix[5][3]+=1
        elif assigned_label=='Noise':
            if correct_label=='LT':
                confusion_matrix[6][0]+=1
            elif correct_label=='ST':
                confusion_matrix[6][1]+=1
            elif correct_label=='Both':
                confusion_matrix[6][2]+=1
        else:
            if correct_label=='LT':
                if assigned_label=='LT?':
                    confusion_matrix[1][0]+=1
                elif assigned_label=='ST':
                    confusion_matrix[2][0]+=1
                elif assigned_label=='ST?':
                    confusion_matrix[3][0]+=1
                elif assigned_label=='Both':
                    confusion_matrix[4][0]+=1
                elif assigned_label=='Both?':
                    confusion_matrix[5][0]+=1
            elif correct_label=='ST':
                if assigned_label=='LT':
                    confusion_matrix[0][1]+=1
                elif assigned_label=='LT?':
                    confusion_matrix[1][1]+=1
                elif assigned_label=='ST?':
                    confusion_matrix[3][1]+=1
                elif assigned_label=='Both':
                    confusion_matrix[4][1]+=1
                elif assigned_label=='Both?':
                    confusion_matrix[5][1]+=1
            elif correct_label=='Both':
                if assigned_label=='LT':
                    confusion_matrix[0][2]+=1
                elif assigned_label=='LT?':
                    confusion_matrix[1][2]+=1
                elif assigned_label=='ST':
                    confusion_matrix[2][2]+=1
                elif assigned_label=='ST?':
                    confusion_matrix[3][2]+=1
                elif assigned_label=='Both?':
                    confusion_matrix[5][2]+=1
            
    comparison_annotations.append([filewise_output[i][0], segments_filewise_test[i][1], assigned_label])
 
Recall, Precision_pre, Precision_post, Accuracy_pre1,Accuracy_pre2, Accuracy_post, TD, FPD_pre, FPD_post, FND, CoCla_pre1, CoCla_pre2, CoCla_post=metrics(confusion_matrix, file_number)

TD_rate= (TD/(file_number-np.sum(confusion_matrix[:][3])))*100
print('True Detected rate', TD_rate)
FPD_pre_rate= (FPD_pre/file_number)*100
print('False Detected rate pre', FPD_pre_rate)
FPD_post_rate= (FPD_post/file_number)*100
print('False Detected rate post', FPD_post_rate)
FND_rate= (FND/file_number)*100
print('False Negative Detected rate', FND_rate)
print(confusion_matrix)
print("-------------------------------------------")
    
#saving Click Detector Stats
cd_metrics_file=test_dir+'\\'+test_fold+'\\bat_detector_stats.txt'
file1=open(cd_metrics_file,"w")
L1=["Bat Detector stats on Testing Data \n"]
L2=['Number of files = %5d \n' %file_number]
L3=['TD = %5d \n' %TD]
L4=['FPD_pre = %5d \n' %FPD_pre]
L5=['FPD_post= %5d \n' %FPD_post]
L6=['FND = %5d \n' %FND]
L7=['Correctly classified files before check (1) = %5d \n' %CoCla_pre1]
L8=['Correctly classified files before check (2) = %5d \n' %CoCla_pre2, 'Correctly classified files after check= %5d \n' %CoCla_post]
L9=["Recall = %3.7f \n" %Recall,"Precision pre = %3.7f \n" %Precision_pre, "Precision post = %3.7f \n" %Precision_post, "Accuracy pre 1= %3.7f \n" %Accuracy_pre1, "Accuracy pre 2= %3.7f \n" %Accuracy_pre2, "Accuracy post = %3.7f \n" %Accuracy_post,  "True Detected rate = %3.7f \n" %TD_rate, "False Detected rate pre = %3.7f \n" %FPD_pre_rate, "False Detected rate post = %3.7f \n" %FPD_post_rate, "False Negative Detected rate = %3.7f \n" %FND_rate]
L10=['Model used %5d \n' %index_best_model]
L11=['Training accuracy for the model %3.7f \n' %accuracies[index_best_model]]
file1.writelines(np.concatenate((L1,L2,L3,L4, L5, L6, L7, L8, L9, L10, L11)))
#file1.writelines(np.concatenate((L1,L2,L3,L4, L5, L6, L7, L8, L9)))

file1.close()
       
#saving compared labels
with open(test_dir+'\\' +test_fold+'\\Test_annotations_comparison.data', 'w') as f:
    json.dump(comparison_annotations,f)
    
with open(test_dir+'\\' +test_fold+"\\Confusion_matrix.txt",'w') as f:
    f.write("Confusion Matrix \n\n")
    np.savetxt(f, confusion_matrix, fmt='%d')

#saving compared labels
with open(test_dir+'\\' +test_fold+'\\Test_filewise_output.data', 'w') as f:
    json.dump(filewise_output,f)
