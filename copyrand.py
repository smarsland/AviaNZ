
import numpy as np
from shutil import copy
from os import listdir

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
    

dirName="D:\\Moria Bat Data\\Bat recoding files by category\\Long tail"    
files = [f for f in listdir(dirName)]
print(len(files))
x = np.arange(len(files)-1)
np.random.shuffle(x)

i=0
while i<1000:
    #print('x[i] = ', x[i])
    #print('file= ', files[x[i]])
    copy(dirName+'\\'+files[x[i]-1],"Z:\\BattyBats\\LT\\")
    i+=1


dirName="D:\\Moria Bat Data\\Bat recoding files by category\\Short tail"    
files = [f for f in listdir(dirName)]
print(len(files))
x = np.arange(len(files)-1)
np.random.shuffle(x)

i=0
while i<1000:
    #print('x[i] = ', x[i])
    #print('file= ', files[x[i]])
    copy(dirName+'\\'+files[x[i]-1],"Z:\\BattyBats\\ST\\")
    i+=1