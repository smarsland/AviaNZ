
import numpy as np
from shutil import copy
from os import listdir
import json

#files = [f for f in listdir('Long tail')]

#x = np.arange(len(files)-1)
#np.random.shuffle(x)

#i=0
#while i<1000:
#    copy('Long tail/'+files[i],'/Volumes/ECS_acoustic_02/BattyBats/LT/')
#    i+=1
    
#files = [f for f in listdir('Short tail')]
#x = np.arange(len(files)-1)
#np.random.shuffle(x)

#i=0
#while i<1000:
#    copy('Short tail/'+files[i],'/Volumes/ECS_acoustic_02/BattyBats/ST/')
#    i+=1


#dirName="D:\\Moria Bat Data\\Bat recoding files by category\\Non-bat"    
#files = [f for f in listdir(dirName)]
#print(len(files))
#x = np.arange(len(files)-1)
#np.random.shuffle(x)

#i=0
#while i<1000:
#    print('x[i] = ', x[i])
#    print('file= ', files[x[i]])
#    copy(dirName+'\\'+files[x[i]-1],"Z:\\BattyBats\\Noise\\")
#    i+=1
    

#copying only if clicks detected


#Long-Tailed

dirName="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\BattyBats\\LT" 
files=[]
for f in listdir(dirName):
    if f.endswith('.bmp'):
        files.append(f)
print(len(files))
x = np.arange(len(files)-1)
np.random.shuffle(x)

#100 files

i=0
count=0 
while count<100:
    #print('x[i] = ', x[i])
    #print('file= ', files[x[i]])
    annotation_file=files[x[i]-1]+'.data'
    f=open(dirName+'\\'+annotation_file)
    segments=json.load(f)
    f.close()
    if len(segments)>1:
        copy(dirName+'\\'+files[x[i]-1],"C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN100\\LT")
        copy(dirName+'\\'+annotation_file, "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN100\\LT")
        count+=1
    i+=1


#250 files
i=0
count=0 
while count<250:
    #print('x[i] = ', x[i])
    #print('file= ', files[x[i]])
    annotation_file=files[x[i]-1]+'.data'
    f=open(dirName+'\\'+annotation_file)
    segments=json.load(f)
    f.close()
    if len(segments)>1:
        copy(dirName+'\\'+files[x[i]-1],"C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN250\\LT")
        copy(dirName+'\\'+annotation_file, "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN250\\LT")
        count+=1
    i+=1

#500 files

i=0
count=0 
while count<500:
    #print('x[i] = ', x[i])
    #print('file= ', files[x[i]])
    annotation_file=files[x[i]-1]+'.data'
    f=open(dirName+'\\'+annotation_file)
    segments=json.load(f)
    f.close()
    if len(segments)>1:
        copy(dirName+'\\'+files[x[i]-1],"C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN500\\LT")
        copy(dirName+'\\'+annotation_file, "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN500\\LT")
        count+=1
    i+=1


#Short-Tailed

dirName="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\BattyBats\\ST" 
files=[]
for f in listdir(dirName):
    if f.endswith('.bmp'):
        files.append(f)
print(len(files))
x = np.arange(len(files)-1)
np.random.shuffle(x)

#100 files

i=0
count=0 
while count<=100:
    #print('x[i] = ', x[i])
    #print('file= ', files[x[i]])
    annotation_file=files[x[i]-1]+'.data'
    f=open(dirName+'\\'+annotation_file)
    segments=json.load(f)
    f.close()
    if len(segments)>1:
        copy(dirName+'\\'+files[x[i]-1],"C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN100\\ST")
        copy(dirName+'\\'+annotation_file, "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN100\\ST")
        count+=1
    i+=1


#250 files
i=0
count=0 
while count<250:
    #print('x[i] = ', x[i])
    #print('file= ', files[x[i]])
    annotation_file=files[x[i]-1]+'.data'
    f=open(dirName+'\\'+annotation_file)
    segments=json.load(f)
    f.close()
    if len(segments)>1:
        copy(dirName+'\\'+files[x[i]-1],"C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN250\\ST")
        copy(dirName+'\\'+annotation_file, "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN250\\ST")
        count+=1
    i+=1

#500 files

i=0
count=0 
while count<500:
    #print('x[i] = ', x[i])
    #print('file= ', files[x[i]])
    annotation_file=files[x[i]-1]+'.data'
    f=open(dirName+'\\'+annotation_file)
    segments=json.load(f)
    f.close()
    if len(segments)>1:
        copy(dirName+'\\'+files[x[i]-1],"C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN500\\ST")
        copy(dirName+'\\'+annotation_file, "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN500\\ST")
        count+=1
    i+=1

#Noise

dirName="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\BattyBats\\Noise" 
files=[]
for f in listdir(dirName):
    if f.endswith('.bmp'):
        files.append(f)
print(len(files))
x = np.arange(len(files)-1)
np.random.shuffle(x)

#100 files

i=0
count=0 
while count<100:
    #print('x[i] = ', x[i])
    #print('file= ', files[x[i]])
    annotation_file=files[x[i]-1]+'.data'
    f=open(dirName+'\\'+annotation_file)
    segments=json.load(f)
    f.close()
    if len(segments)>1:
        copy(dirName+'\\'+files[x[i]-1],"C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN100\\NOISE")
        copy(dirName+'\\'+annotation_file, "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN100\\NOISE")
        count+=1
    i+=1


#250 files
i=0
count=0 
while count<250:
    #print('x[i] = ', x[i])
    #print('file= ', files[x[i]])
    annotation_file=files[x[i]-1]+'.data'
    f=open(dirName+'\\'+annotation_file)
    segments=json.load(f)
    f.close()
    if len(segments)>1:
        copy(dirName+'\\'+files[x[i]-1],"C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN250\\NOISE")
        copy(dirName+'\\'+annotation_file, "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN250\\NOISE")
        count+=1
    i+=1

#500 files

i=0
count=0 
while count<500:
    #print('x[i] = ', x[i])
    #print('file= ', files[x[i]])
    annotation_file=files[x[i]-1]+'.data'
    f=open(dirName+'\\'+annotation_file)
    segments=json.load(f)
    f.close()
    if len(segments)>1:
        copy(dirName+'\\'+files[x[i]-1],"C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN500\\NOISE")
        copy(dirName+'\\'+annotation_file, "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Train_Datasets\\TRAIN500\\NOISE")
        count+=1
    i+=1

