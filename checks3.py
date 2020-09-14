"""
Authors: Stephen Marsland, Virginia Listanti

This code recover information from files and then plot max, min, mean and mean for 
the best 3 in  4different plots: LT, ST, NT, BT with different classification 
conditions

It does this for all file classification with all possible spectrogram 
probabilities

ACTUAL CONDITIONS:
    LT -> P(0)>=65 and P(1)<65
    ST -> P(1)>=65 and P(0)<65
    BT -> P(0)>=65 and P(1)>=65
    NT -> else

LEGEND :) :
    ST -> Short Tailed
    LT -> Long Tailed
    NT -> NO Tailed
    BT -> Both tailed  
"""
import numpy as np
import json
#import pylab as pl

import matplotlib.pyplot as plt
import pyqtgraph as pg
import pyqtgraph.exporters as pge

#READ INFORMATION FROM FILES
f = open('D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\ST_spec_prob.data')
a = json.load(f)
f.close()

f = open('D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\LT_spec_prob.data')
b = json.load(f)
f.close()

f = open('D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Noise_spec_prob.data')
c = json.load(f)
f.close()


f = open('D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Both_spec_prob.data')
d = json.load(f)
f.close()
# a[i][0] is filename, [1] is the species label, [2] is the np array

#St files divided vy divided by clicks probabilities
## Clicks statistics in ST files
ST_ST = [] #save informations for ST clicks in ST files
ST_LT = [] #save informations for LT clicks in ST files
ST_NT = [] #save informations for ST clicks in ST files
count = 0
#aid variables
st_st=[]
st_nt=[]
st_lt=[]
while count<len(a):
    file = a[count][0]
    while count<len(a) and a[count][0] == file:
        st_st.append(a[count][1][1])
        st_nt.append(a[count][1][2])
        st_lt.append(a[count][1][0])
        count+=1
    ST_ST.append([file,st_st])
    ST_NT.append([file,st_nt])
    ST_LT.append([file,st_lt])
    st_st = []
    st_nt =[]
    st_lt =[]

#metrics for ST_ST files
ST_ST_max=np.zeros((len(ST_ST),1))
ST_ST_min=np.zeros((len(ST_ST),1))
ST_ST_mean=np.zeros((len(ST_ST),1))
ST_ST_best3mean=np.zeros((len(ST_ST),1))
ST_filenames=[]
#print('check ST_ST')
#print(ST_ST[0])
for i in range(len(ST_ST)):
#    print(ST_ST[i])
    ST_filenames.append(ST_ST[i][0])
    ST_ST_max[i] = np.max(ST_ST[i][1])
    ST_ST_min[i] = np.min(ST_ST[i][1])
    ST_ST_best3mean [i]= 0
    ind = np.array(ST_ST[i][1]).argsort()[-3:][::-1]
#    print(ind, len(ind))
#    adding len ind in order to consider also the cases when we do not have 3 good examples
    if len(ind)==1:
        #this means that there is only one prob!
        ST_ST_best3mean+=ST_ST[i][1]
    else:
        for j in range(len(ind)):
            ST_ST_best3mean[i]+=ST_ST[i][1][ind[j]]
    ST_ST_best3mean[i]/= len(ind)
    ST_ST_mean[i]=np.mean(ST_ST[i][1])
#    print(ST[i][0],stmax,stmean,np.mean(ST[i][1]))

#metrics for ST_LT files
ST_LT_max=np.zeros((len(ST_LT),1))
ST_LT_min=np.zeros((len(ST_LT),1))
ST_LT_mean=np.zeros((len(ST_LT),1))
ST_LT_best3mean=np.zeros((len(ST_LT),1))
#print(ST_ST)
for i in range(len(ST_LT)):
#    print(ST_ST[i])
    ST_LT_max[i] = np.max(ST_LT[i][1])
    ST_LT_min[i] = np.min(ST_LT[i][1])
    ST_LT_best3mean [i]= 0
    ind = np.array(ST_LT[i][1]).argsort()[-3:][::-1]
#    print(ind, len(ind))
#    adding len ind in order to consider also the cases when we do not have 3 good examples
    if len(ind)==1:
        #this means that there is only one prob!
        ST_LT_best3mean+=ST_LT[i][1]
    else:
        for j in range(len(ind)):
            ST_LT_best3mean[i]+=ST_LT[i][1][ind[j]]
    ST_LT_best3mean[i]/= len(ind)
    ST_LT_mean[i]=np.mean(ST_LT[i][1])
    
#metrics for ST_NT files
ST_NT_max=np.zeros((len(ST_NT),1))
ST_NT_min=np.zeros((len(ST_NT),1))
ST_NT_mean=np.zeros((len(ST_NT),1))
ST_NT_best3mean=np.zeros((len(ST_NT),1))
#BT_filenames=[]
for i in range(len(ST_NT)):
#    BT_filenames.append(ST_BT[i][0])
    ST_NT_max[i] = np.max(ST_NT[i][1])
    ST_NT_min[i] = np.min(ST_NT[i][1])
    ST_NT_best3mean[i] = 0
    ind = np.array(ST_NT[i][1]).argsort()[-3:][::-1]
    if len(ind)==1:
        #this means that there is only one prob!
        ST_NT_best3mean+=ST_NT[i][1]
    else:
        for j in range(len(ind)):
            ST_NT_best3mean[i]+=ST_NT[i][1][ind[j]]
    ST_NT_best3mean[i]/= len(ind)
    ST_NT_mean[i]=np.mean(ST_NT[i][1])

###LT files divided vy divided by clicks probabilities   
LT_LT = [] #save informations for LT clicks in LT files
LT_ST = [] #save informations for ST clicks in LT files
LT_NT = [] #save informations for NT clicks in LT files
count = 0
#aid variables
lt_lt=[]
lt_st=[]
lt_nt=[]
while count<len(b):
    file = b[count][0]
    while count<len(b) and b[count][0] == file:
        lt_lt.append(b[count][1][0])
        lt_st.append(b[count][1][1])
        lt_nt.append(b[count][1][2])
        count+=1
    LT_LT.append([file,lt_lt])
    LT_ST.append([file,lt_st])
    LT_NT.append([file,lt_nt])
    lt_lt = []
    lt_st = []
    lt_nt =[]

#metrics for LT_LT files
    #note LT_LT len is equal to LT_NT len
LT_LT_max=np.zeros((len(LT_LT),1))
LT_LT_min=np.zeros((len(LT_LT),1))
LT_LT_mean=np.zeros((len(LT_LT),1))
LT_LT_best3mean=np.zeros((len(LT_LT),1))
LT_filenames=[]
#print(np.shape(np.asarray(LT_LT)))
for i in range(len(LT_LT)):
#    print(LT_LT[i][0])
#    print(LT_LT[i][1])
    LT_filenames.append(LT_LT[i][0])
    LT_LT_max[i] = np.max(LT_LT[i][1])
    LT_LT_min[i] = np.min(LT_LT[i][1])
#    LT_LT_in = min(LT_LT[i][1])
    LT_LT_best3mean[i]= 0
    ind = np.array(LT_LT[i][1]).argsort()[-3:][::-1]
    if len(ind)==1:
        #this means that there is only one prob!
        LT_LT_best3mean[i]=LT_LT[i][1]
    else:
        for j in range(len(ind)):
            LT_LT_best3mean[i]+=LT_LT[i][1][ind[j]]
    LT_LT_best3mean[i]/= len(ind)
    LT_LT_mean[i]=np.mean(LT_LT[i][1]) 

#print(LT_LT_max)
#print(LT_LT_min)
#print(LT_LT_mean)
#print(LT_LT_best3mean)
#print(LT_filenames)


#metrics for LT_ST files
#print('check LT_ST')
#print(LT_ST[1])
#print(np.shape(LT_ST))
LT_ST_max=np.zeros((len(LT_ST),1))
LT_ST_min=np.zeros((len(LT_ST),1))
LT_ST_mean=np.zeros((len(LT_ST),1))
LT_ST_best3mean=np.zeros((len(LT_ST),1))
for i in range(len(LT_ST)):
#    print(LT_ST[i][0])
#    print(LT_ST[i][1])
    LT_ST_max[i] = np.max(LT_ST[i][1])
    LT_ST_min[i] = np.min(LT_ST[i][1])
#    LT_LT_in = min(LT_LT[i][1])
    LT_ST_best3mean[i]= 0
    ind = np.array(LT_ST[i][1]).argsort()[-3:][::-1]
    if len(ind)==1:
        #this means that there is only one prob!
        LT_ST_best3mean[i]+=LT_ST[i][1]
    else:
        for j in range(len(ind)):
            LT_ST_best3mean[i]+=LT_ST[i][1][ind[j]]
    LT_ST_best3mean[i]/= len(ind)
    LT_ST_mean[i]=np.mean(LT_ST[i][1])     
    
#print('check on ST probabilities')
#print(LT_ST_max)
#print(LT_ST_min)
#print(LT_ST_mean)
#print(LT_ST_best3mean)
#print(LT_filenames) 
    
#metrics for LT_BT files
LT_NT_max=np.zeros((len(LT_NT),1))
LT_NT_min=np.zeros((len(LT_NT),1))
LT_NT_mean=np.zeros((len(LT_NT),1))
LT_NT_best3mean=np.zeros((len(LT_NT),1))
for i in range(len(LT_NT)):
    LT_NT_max[i] = np.max(LT_NT[i][1])
    LT_NT_min[i] =np.min(LT_NT[i][1])
#    LT_BT_in = min(LT_BT[i][1])
    LT_NT_best3mean[i] = 0
    ind = np.array(LT_NT[i][1]).argsort()[-3:][::-1]
    if len(ind)==1:
        #this means that there is only one prob!
        LT_NT_best3mean[i]+=LT_NT[i][1]
    else:
        for j in range(len(ind)):
            LT_NT_best3mean[i]+=LT_NT[i][1][ind[j]]
    LT_NT_best3mean[i]/= len(ind)
    LT_NT_mean[i]=np.mean(LT_NT[i][1])    

## NOISE flies divided by clicks probabilities
NT_LT = [] #save informations for LT clicks into Noise files
NT_ST = [] #save informations for ST clicks into Noise files
NT_NT = [] #save informations for Noise clicks into Noise files
count = 0
#aid variable: I just need one because the file label it is consistent in all files
nt_nt=[]
nt_lt=[]
nt_st=[]
while count<len(c):
    file = c[count][0]
    while count<len(c) and c[count][0] == file :
#        if c[count][0]!=0:
        c[count][1]=np.reshape(c[count][1],(3,))
#        print(c[count][1])
#        print(np.shape(c[count][1]))
        nt_nt.append(c[count][1][2])
        nt_lt.append(c[count][1][0])
        nt_st.append(c[count][1][1])
        count+=1
        
    NT_NT.append([file,nt_nt])
    NT_LT.append([file,nt_lt])
    NT_ST.append([file,nt_st])
    nt_nt=[]
    nt_lt=[]
    nt_st=[]

#metrics for NT_NT files
NT_NT_max=np.zeros((len(NT_NT),1))
NT_NT_min=np.zeros((len(NT_NT),1))
NT_NT_mean=np.zeros((len(NT_NT),1))
NT_NT_best3mean=np.zeros((len(NT_NT),1))
NT_filenames=[]
for i in range(len(NT_NT)):
    NT_filenames.append(NT_NT[i][0])
    NT_NT_max[i] = np.max(NT_NT[i][1])
    NT_NT_min[i] = np.min(NT_NT[i][1])
#    NT_NT_in = min(NT_NT[i][1])
    NT_NT_best3mean[i] = 0
    ind = np.array(NT_NT[i][1]).argsort()[-3:][::-1]
    if len(ind)==1:
        #this means that there is only one prob!
        NT_NT_best3mean[i]+=NT_NT[i][1]
    else:
        for j in range(len(ind)):
            NT_NT_best3mean[i]+=NT_NT[i][1][ind[j]]
    NT_NT_best3mean[i]/= len(ind)
    NT_NT_mean[i]=np.mean(NT_NT[i][1])    

print('check NT_NT_best3mean')
print('max', max(NT_NT_best3mean))   
  
#metrics for NT_LT files
NT_LT_max=np.zeros((len(NT_LT),1))
NT_LT_min=np.zeros((len(NT_LT),1))
NT_LT_mean=np.zeros((len(NT_LT),1))
NT_LT_best3mean=np.zeros((len(NT_LT),1))
for i in range(len(NT_LT)):
    NT_LT_max[i] = np.max(NT_LT[i][1])
    NT_LT_min[i] = np.min(NT_LT[i][1])
#    NT_NT_in = min(NT_NT[i][1])
    NT_LT_best3mean[i] = 0
    ind = np.array(NT_LT[i][1]).argsort()[-3:][::-1]
    if len(ind)==1:
        #this means that there is only one prob!
        NT_LT_best3mean[i]+=NT_LT[i][1]
    else:
        for j in range(len(ind)):
            NT_LT_best3mean[i]+=NT_LT[i][1][ind[j]]
    NT_LT_best3mean[i]/= len(ind)
    NT_LT_mean[i]=np.mean(NT_LT[i][1]) 
    
#metrics for NT_ST files
NT_ST_max=np.zeros((len(NT_ST),1))
NT_ST_min=np.zeros((len(NT_ST),1))
NT_ST_mean=np.zeros((len(NT_ST),1))
NT_ST_best3mean=np.zeros((len(NT_ST),1))
for i in range(len(NT_ST)):
    NT_ST_max[i] = np.max(NT_ST[i][1])
    NT_ST_min[i] = np.min(NT_ST[i][1])
#    NT_NT_in = min(NT_NT[i][1])
    NT_ST_best3mean[i] = 0
    ind = np.array(NT_ST[i][1]).argsort()[-3:][::-1]
    if len(ind)==1:
        #this means that there is only one prob!
        NT_ST_best3mean[i]+=NT_ST[i][1]
    else:
        for j in range(len(ind)):
            NT_ST_best3mean[i]+=NT_ST[i][1][ind[j]]
    NT_ST_best3mean[i]/= len(ind)
    NT_ST_mean[i]=np.mean(NT_ST[i][1]) 
    
##clicks in BT files    
BT_LT = [] #save informations for LT clicks into BT files
BT_ST = [] #save informations for ST clicks into BT files
BT_NT = [] #save informations for Noise clicks into BT files
count = 0
#aid variables
bt_lt=[]
bt_st=[]
bt_nt=[]
while count<len(d):
    file = d[count][0]
    while count<len(d) and d[count][0] == file:
        bt_lt.append(d[count][1][0])
        bt_st.append(d[count][1][1])
        bt_nt.append(d[count][1][2])
        count+=1
    BT_LT.append([file,bt_lt])
    BT_ST.append([file,bt_st])
    BT_NT.append([file,bt_nt])
    bt_lt = []
    bt_st = []
    bt_nt = []

#print('BT_LT')
#print(BT_LT)
#print('BT_ST')
#print(BT_ST)
#print('BT_NT')
#print(BT_NT)
#metrics for BT_LT files
    #note  BT_LT len is equal to BT_NT and to BT_ST len
BT_LT_max=np.zeros((len(BT_LT),1))
BT_LT_min=np.zeros((len(BT_LT),1))
BT_LT_mean=np.zeros((len(BT_LT),1))
BT_LT_best3mean=np.zeros((len(BT_LT),1))
BT_filenames=[]
for i in range(len(BT_LT)):
#    print(np.shape(BT_LT[i][1]))
#    print(BT_LT[i][1])
    BT_filenames.append(BT_LT[i][0])
    BT_LT_max[i] = np.max(BT_LT[i][1])
    BT_LT_min[i] = np.min(BT_LT[i][1])
#    LT_LT_in = min(LT_LT[i][1])
    BT_LT_best3mean[i]= 0
    ind = np.array(BT_LT[i][1]).argsort()[-3:][::-1]
    if len(ind)==1:
        #this means that there is only one prob!
        BT_LT_best3mean[i]+=BT_LT[i][1]
    else:
        for j in range(len(ind)):
            BT_LT_best3mean[i]+=BT_LT[i][1][ind[j]]
    BT_LT_best3mean[i]/= len(ind)
    BT_LT_mean[i]=np.mean(BT_LT[i][1]) 
    
#metrics for BT_NT files
BT_NT_max=np.zeros((len(BT_NT),1))
BT_NT_min=np.zeros((len(BT_NT),1))
BT_NT_mean=np.zeros((len(BT_NT),1))
BT_NT_best3mean=np.zeros((len(BT_NT),1))
for i in range(len(BT_NT)):
#    print(np.shape(BT_NT[i][1]))
    BT_NT_max[i] = np.max(BT_NT[i][1])
    BT_NT_min[i] = np.min(BT_NT[i][1])
#    LT_BT_in = min(LT_BT[i][1])
    BT_NT_best3mean[i] = 0
    ind = np.array(BT_NT[i][1]).argsort()[-3:][::-1]
    if len(ind)==1:
        #this means that there is only one prob!
        BT_NT_best3mean[i]+=BT_NT[i][1]
    else:
        for j in range(len(ind)):
            BT_NT_best3mean[i]+=BT_NT[i][1][ind[j]]
    BT_NT_best3mean[i]/= len(ind)
    BT_NT_mean[i]=np.mean(BT_NT[i][1])  

#metrics for BT_ST files
BT_ST_max=np.zeros((len(BT_ST),1))
BT_ST_min=np.zeros((len(BT_ST),1))
BT_ST_mean=np.zeros((len(BT_ST),1))
BT_ST_best3mean=np.zeros((len(BT_ST),1))
for i in range(len(BT_ST)):
    BT_ST_max[i] = np.max(BT_ST[i][1])
    BT_ST_min[i] = np.min(BT_ST[i][1])
#    LT_BT_in = min(LT_BT[i][1])
    BT_ST_best3mean[i] = 0
    ind = np.array(BT_ST[i][1]).argsort()[-3:][::-1]
    if len(ind)==1:
        #this means that there is only one prob!
        BT_ST_best3mean[i]+=BT_ST[i][1]
    else:
        for j in range(len(ind)):
            BT_ST_best3mean[i]+=BT_ST[i][1][ind[j]]
    BT_ST_best3mean[i]/= len(ind)
    BT_ST_mean[i]=np.mean(BT_ST[i][1])      

##LT files detections and plot

#detect LT in LT files
index_LT_best3mean=np.nonzero(LT_LT_best3mean*100>=65)
LT_LT_detected_best3mean=np.zeros((np.shape(LT_LT_best3mean)))
LT_LT_detected_best3mean[index_LT_best3mean]=np.where(LT_ST_best3mean[index_LT_best3mean]*100<65,1,0)

index_LT_max=np.nonzero(LT_LT_max*100>=65)
LT_LT_detected_max=np.zeros((np.shape(LT_LT_max)))
LT_LT_detected_max[index_LT_max]=np.where(LT_ST_max[index_LT_max]*100<65,1,0)

index_LT_mean=np.nonzero(LT_LT_mean*100>=65)
LT_LT_detected_mean=np.zeros((np.shape(LT_LT_mean)))
LT_LT_detected_mean[index_LT_mean]=np.where(LT_ST_mean[index_LT_mean]*100<65,1,0) 

#detect ST in LT files
index_ST_best3mean=np.nonzero(LT_ST_best3mean*100>=65)
LT_ST_detected_best3mean=np.zeros((np.shape(LT_ST_best3mean)))
LT_ST_detected_best3mean[index_ST_best3mean]=np.where(LT_LT_best3mean[index_ST_best3mean]*100<65,1,0)

index_ST_max=np.nonzero(LT_ST_max*100>=65)
LT_ST_detected_max=np.zeros((np.shape(LT_ST_max)))
LT_ST_detected_max[index_ST_max]=np.where(LT_LT_max[index_ST_max]*100<65,1,0)

index_ST_mean=np.nonzero(LT_ST_mean*100>=65)
LT_ST_detected_mean=np.zeros((np.shape(LT_ST_mean)))
LT_ST_detected_mean[index_ST_mean]=np.where(LT_LT_mean[index_ST_mean]*100<65,1,0) 

#detect BT in LT files
LT_BT_detected_best3mean=np.zeros(np.shape(LT_LT_detected_best3mean))
#in this way we are asking both the condition to be satisfied
LT_BT_detected_best3mean[index_LT_best3mean]=np.where(LT_ST_best3mean[index_LT_best3mean]*100>=65 ,1,0) 

LT_BT_detected_max=np.zeros(np.shape(LT_LT_detected_max))
#in this way we are asking both the condition to be satisfied
LT_BT_detected_max[index_LT_max]=np.where(LT_ST_max[index_LT_max]*100>=65 ,1,0)

LT_BT_detected_mean=np.zeros(np.shape(LT_LT_detected_mean))
#in this way we are asking both the condition to be satisfied
LT_BT_detected_mean[index_LT_mean]=np.where(LT_ST_mean[index_LT_mean]*100>=65 ,1,0)

#Detected NT in LT files
LT_NT_detected_best3mean=np.ones((np.shape(LT_LT_best3mean)))
LT_NT_detected_best3mean[np.nonzero(LT_LT_detected_best3mean==1)]-=LT_LT_detected_best3mean[np.nonzero(LT_LT_detected_best3mean==1)]
LT_NT_detected_best3mean[np.nonzero(LT_ST_detected_best3mean==1)]-=LT_ST_detected_best3mean[np.nonzero(LT_ST_detected_best3mean==1)]
LT_NT_detected_best3mean[np.nonzero(LT_BT_detected_best3mean==1)]-=LT_BT_detected_best3mean[np.nonzero(LT_BT_detected_best3mean==1)]
#LT_NT_detected_best3mean=np.where(LT_NT_detected_best3mean>0,1,0) it should not be necessary now
LT_NT_detected_max=np.ones((np.shape(LT_LT_max)))
LT_NT_detected_max[np.nonzero(LT_LT_detected_max==1)]-=LT_LT_detected_max[np.nonzero(LT_LT_detected_max==1)]
LT_NT_detected_max[np.nonzero(LT_ST_detected_max==1)]-=LT_ST_detected_max[np.nonzero(LT_ST_detected_max==1)]
LT_NT_detected_max[np.nonzero(LT_BT_detected_max==1)]-=LT_BT_detected_max[np.nonzero(LT_BT_detected_max==1)]
#LT_NT_detected_max=np.where(LT_NT_detected_max>0,1,0)
LT_NT_detected_mean=np.ones((np.shape(LT_LT_mean)))
LT_NT_detected_mean[np.nonzero(LT_LT_detected_mean==1)]-=LT_LT_detected_mean[np.nonzero(LT_LT_detected_mean==1)]
LT_NT_detected_mean[np.nonzero(LT_ST_detected_mean==1)]-=LT_ST_detected_mean[np.nonzero(LT_ST_detected_mean==1)]
LT_NT_detected_mean[np.nonzero(LT_BT_detected_mean==1)]-=LT_BT_detected_mean[np.nonzero(LT_BT_detected_mean==1)]
#print(np.shape(np.where(LT_NT_detected_mean>0,1,0)))
#LT_NT_detected_mean=np.where(LT_NT_detected_mean>0,1,0)

## find missed LT files with different methods
# a missed file is a file that  it will be classified as noise

LT_filenames_a=np.reshape(LT_filenames,(len(LT_filenames),1))
LT_missed_best3mean=LT_filenames_a[np.nonzero(LT_NT_detected_best3mean)]
LT_missed_max=LT_filenames_a[np.nonzero(LT_NT_detected_max)]
LT_missed_mean=LT_filenames_a[np.nonzero(LT_NT_detected_mean)]

#plot
fig, axes=plt.subplots(4,3,sharex='all', sharey='all')
fig.suptitle('Detections in LT files')
#LT first row
axes[0][0].plot(LT_filenames, LT_LT_best3mean, 'b', LT_LT_detected_best3mean, 'r')
axes[0][0].set_title('Best 3 mean')
axes[0][0].set_ylabel('LT', rotation=0, size='large')
axes[0][1].plot(LT_filenames, LT_LT_max, 'b', LT_LT_detected_max, 'r')
axes[0][1].set_title('Max', rotation=0)
axes[0][2].plot(LT_filenames, LT_LT_mean, 'b', LT_LT_detected_mean, 'r')
axes[0][2].set_title('Mean')
axes[0][2].axes.xaxis.set_ticklabels([])

#ST second row
axes[1][0].plot(LT_filenames, LT_ST_best3mean, 'g', LT_ST_detected_best3mean, 'r')
axes[1][0].set_ylabel('ST', rotation=0, size='large')
axes[1][1].plot(LT_filenames, LT_ST_max, 'g', LT_ST_detected_max, 'r')
axes[1][2].plot(LT_filenames, LT_ST_mean, 'g', LT_ST_detected_mean, 'r')
axes[1][2].axes.xaxis.set_ticklabels([])

#BT Third row
axes[2][0].plot(LT_filenames, LT_LT_best3mean, 'b', LT_ST_best3mean, 'g', LT_BT_detected_best3mean, 'r')
axes[2][0].set_ylabel('BT', rotation=0, size='large')
axes[2][1].plot(LT_filenames,LT_LT_max, 'b', LT_ST_max, 'g',  LT_BT_detected_max, 'r')
axes[2][2].plot(LT_filenames,LT_LT_mean, 'b', LT_ST_mean, 'g', LT_BT_detected_mean, 'r')
axes[2][2].axes.xaxis.set_ticklabels([])

#NT Forth row
axes[3][0].plot(LT_filenames, LT_NT_best3mean, 'b', LT_NT_detected_best3mean, 'r')
axes[3][0].set_ylabel('NT', rotation=0, size='large')
axes[3][1].plot(LT_filenames, LT_NT_max, 'b', LT_NT_detected_max, 'r')
axes[3][2].plot(LT_filenames, LT_NT_mean, 'b', LT_NT_detected_mean, 'r')
axes[3][2].axes.xaxis.set_ticklabels([])
#hide labels for inner plots
for ax in axes.flat:
    ax.label_outer()
#plt.show()
plt.savefig("D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Test02\\LT_detections.png")

#save
f= open('D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Test02\\LT_missed.txt', 'w')
f.write('BEST 3 MEAN \n')
json.dump(LT_missed_best3mean.tolist(),f)
f.write('\n MAX \n')
json.dump(LT_missed_max.tolist(),f)
f.write('\n Mean \n')
json.dump(LT_missed_mean.tolist(),f)
f.close()

##ST files plot
#detect LT in ST files
index_LT_best3mean=np.nonzero(ST_LT_best3mean*100>=65)
ST_LT_detected_best3mean=np.zeros((np.shape(ST_LT_best3mean)))
ST_LT_detected_best3mean[index_LT_best3mean]=np.where(LT_ST_best3mean[index_LT_best3mean]*100<65,1,0)

index_LT_max=np.nonzero(ST_LT_max*100>=65)
ST_LT_detected_max=np.zeros((np.shape(ST_LT_max)))
ST_LT_detected_max[index_LT_max]=np.where(ST_ST_max[index_LT_max]*100<65,1,0)

index_LT_mean=np.nonzero(ST_LT_mean*100>=65)
ST_LT_detected_mean=np.zeros((np.shape(ST_LT_mean)))
ST_LT_detected_mean[index_LT_mean]=np.where(ST_ST_mean[index_LT_mean]*100<65,1,0) 

#detect ST in ST files
index_ST_best3mean=np.nonzero(ST_ST_best3mean*100>=65)
ST_ST_detected_best3mean=np.zeros((np.shape(ST_ST_best3mean)))
ST_ST_detected_best3mean[index_ST_best3mean]=np.where(ST_LT_best3mean[index_ST_best3mean]*100<65,1,0)

index_ST_max=np.nonzero(ST_ST_max*100>=65)
ST_ST_detected_max=np.zeros((np.shape(ST_ST_max)))
ST_ST_detected_max[index_ST_max]=np.where(ST_LT_max[index_ST_max]*100<65,1,0)

index_ST_mean=np.nonzero(ST_ST_mean*100>=65)
ST_ST_detected_mean=np.zeros((np.shape(ST_ST_mean)))
ST_ST_detected_mean[index_ST_mean]=np.where(ST_LT_mean[index_ST_mean]*100<65,1,0) 

#detect BT in ST files
ST_BT_detected_best3mean=np.zeros(np.shape(ST_LT_detected_best3mean))
#in this way we are asking both the condition to be satisfied
ST_BT_detected_best3mean[index_LT_best3mean]=np.where(ST_ST_best3mean[index_LT_best3mean]*100>=65 ,1,0) 

ST_BT_detected_max=np.zeros(np.shape(ST_LT_detected_max))
#in this way we are asking both the condition to be satisfied
ST_BT_detected_max[index_LT_max]=np.where(ST_ST_max[index_LT_max]*100>=65 ,1,0)

ST_BT_detected_mean=np.zeros(np.shape(ST_LT_detected_mean))
#in this way we are asking both the condition to be satisfied
ST_BT_detected_mean[index_LT_mean]=np.where(ST_ST_mean[index_LT_mean]*100>=65 ,1,0)

#Detected NT in ST files
ST_NT_detected_best3mean=np.ones((np.shape(ST_LT_best3mean)))
ST_NT_detected_best3mean[np.nonzero(ST_LT_detected_best3mean==1)]-=ST_LT_detected_best3mean[np.nonzero(ST_LT_detected_best3mean==1)]
ST_NT_detected_best3mean[np.nonzero(ST_ST_detected_best3mean==1)]-=ST_ST_detected_best3mean[np.nonzero(ST_ST_detected_best3mean==1)]
ST_NT_detected_best3mean[np.nonzero(ST_BT_detected_best3mean==1)]-=ST_BT_detected_best3mean[np.nonzero(ST_BT_detected_best3mean==1)]
#LT_NT_detected_best3mean=np.where(LT_NT_detected_best3mean>0,1,0) it should not be necessary now
ST_NT_detected_max=np.ones((np.shape(ST_LT_max)))
ST_NT_detected_max[np.nonzero(ST_LT_detected_max==1)]-=ST_LT_detected_max[np.nonzero(ST_LT_detected_max==1)]
ST_NT_detected_max[np.nonzero(ST_ST_detected_max==1)]-=ST_ST_detected_max[np.nonzero(ST_ST_detected_max==1)]
ST_NT_detected_max[np.nonzero(ST_BT_detected_max==1)]-=ST_BT_detected_max[np.nonzero(ST_BT_detected_max==1)]
#LT_NT_detected_max=np.where(LT_NT_detected_max>0,1,0)
ST_NT_detected_mean=np.ones((np.shape(ST_LT_mean)))
ST_NT_detected_mean[np.nonzero(ST_LT_detected_mean==1)]-=ST_LT_detected_mean[np.nonzero(ST_LT_detected_mean==1)]
ST_NT_detected_mean[np.nonzero(ST_ST_detected_mean==1)]-=ST_ST_detected_mean[np.nonzero(ST_ST_detected_mean==1)]
ST_NT_detected_mean[np.nonzero(ST_BT_detected_mean==1)]-=ST_BT_detected_mean[np.nonzero(ST_BT_detected_mean==1)]

## find missed ST files with different methods
# a missed file is a file that  it will be classified as noise

ST_filenames_a=np.reshape(ST_filenames,(len(ST_filenames),1))
ST_missed_best3mean=ST_filenames_a[np.nonzero(ST_NT_detected_best3mean)]
ST_missed_max=ST_filenames_a[np.nonzero(ST_NT_detected_max)]
ST_missed_mean=ST_filenames_a[np.nonzero(ST_NT_detected_mean)]

#plot
fig, axes=plt.subplots(4,3,sharex='all', sharey='all')
fig.suptitle('Detections in ST files')
#LT first row
axes[0][0].plot(ST_filenames, ST_LT_best3mean, 'b', ST_LT_detected_best3mean, 'r')
axes[0][0].set_title('Best 3 mean')
axes[0][0].set_ylabel('LT', rotation=0, size='large')
axes[0][1].plot(ST_filenames, ST_LT_max, 'b', ST_LT_detected_max, 'r')
axes[0][1].set_title('Max', rotation=0)
axes[0][2].plot(ST_filenames, ST_LT_mean, 'b', ST_LT_detected_mean, 'r')
axes[0][2].set_title('Mean')
axes[0][2].axes.xaxis.set_ticklabels([])

#ST second row
axes[1][0].plot(ST_filenames, ST_ST_best3mean, 'g', ST_ST_detected_best3mean, 'r')
axes[1][0].set_ylabel('ST', rotation=0, size='large')
axes[1][1].plot(ST_filenames, ST_ST_max, 'g', ST_ST_detected_max, 'r')
axes[1][2].plot(ST_filenames, ST_ST_mean, 'g', ST_ST_detected_mean, 'r')
axes[1][2].axes.xaxis.set_ticklabels([])

#BT Third row
axes[2][0].plot(ST_filenames, ST_LT_best3mean, 'b', ST_ST_best3mean, 'g', ST_BT_detected_best3mean, 'r')
axes[2][0].set_ylabel('BT', rotation=0, size='large')
axes[2][1].plot(ST_filenames,ST_LT_max, 'b', ST_ST_max, 'g',  ST_BT_detected_max, 'r')
axes[2][2].plot(ST_filenames,ST_LT_mean, 'b', ST_ST_mean, 'g', ST_BT_detected_mean, 'r')
axes[2][2].axes.xaxis.set_ticklabels([])

#NT Forth row
axes[3][0].plot(ST_filenames, ST_NT_best3mean, 'b', ST_NT_detected_best3mean, 'r')
axes[3][0].set_ylabel('NT', rotation=0, size='large')
axes[3][1].plot(ST_filenames, ST_NT_max, 'b', ST_NT_detected_max, 'r')
axes[3][2].plot(ST_filenames, ST_NT_mean, 'b', ST_NT_detected_mean, 'r')
axes[3][2].axes.xaxis.set_ticklabels([])
#hide labels for inner plots
for ax in axes.flat:
    ax.label_outer()
#plt.show()
plt.savefig("D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Test02\\ST_detections.png")

#save
f= open('D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Test02\\ST_missed.txt', 'w')
f.write('BEST 3 MEAN \n')
json.dump(ST_missed_best3mean.tolist(),f)
f.write('\n MAX \n')
json.dump(ST_missed_max.tolist(),f)
f.write('\n Mean \n')
json.dump(ST_missed_mean.tolist(),f)
f.close()

##NT files plot

#detect LT in NT files
index_LT_best3mean=np.nonzero(NT_LT_best3mean*100>=65)
NT_LT_detected_best3mean=np.zeros((np.shape(NT_LT_best3mean)))
NT_LT_detected_best3mean[index_LT_best3mean]=np.where(NT_ST_best3mean[index_LT_best3mean]*100<65,1,0)

index_LT_max=np.nonzero(NT_LT_max*100>=65)
NT_LT_detected_max=np.zeros((np.shape(NT_LT_max)))
NT_LT_detected_max[index_LT_max]=np.where(NT_ST_max[index_LT_max]*100<65,1,0)

index_LT_mean=np.nonzero(NT_LT_mean*100>=65)
NT_LT_detected_mean=np.zeros((np.shape(NT_LT_mean)))
NT_LT_detected_mean[index_LT_mean]=np.where(NT_ST_mean[index_LT_mean]*100<65,1,0) 

#detect ST in NT files
index_ST_best3mean=np.nonzero(NT_ST_best3mean*100>=65)
NT_ST_detected_best3mean=np.zeros((np.shape(NT_ST_best3mean)))
NT_ST_detected_best3mean[index_ST_best3mean]=np.where(NT_LT_best3mean[index_ST_best3mean]*100<65,1,0)

index_ST_max=np.nonzero(NT_ST_max*100>=65)
NT_ST_detected_max=np.zeros((np.shape(NT_ST_max)))
NT_ST_detected_max[index_ST_max]=np.where(NT_LT_max[index_ST_max]*100<65,1,0)

index_ST_mean=np.nonzero(NT_ST_mean*100>=65)
NT_ST_detected_mean=np.zeros((np.shape(NT_ST_mean)))
NT_ST_detected_mean[index_ST_mean]=np.where(NT_LT_mean[index_ST_mean]*100<65,1,0) 

#detect BT in NT files
NT_BT_detected_best3mean=np.zeros(np.shape(NT_LT_detected_best3mean))
#in this way we are asking both the condition to be satisfied
NT_BT_detected_best3mean[index_LT_best3mean]=np.where(NT_ST_best3mean[index_LT_best3mean]*100>=65 ,1,0) 

NT_BT_detected_max=np.zeros(np.shape(NT_LT_detected_max))
#in this way we are asking both the condition to be satisfied
NT_BT_detected_max[index_LT_max]=np.where(NT_ST_max[index_LT_max]*100>=65 ,1,0)

NT_BT_detected_mean=np.zeros(np.shape(NT_LT_detected_mean))
#in this way we are asking both the condition to be satisfied
NT_BT_detected_mean[index_LT_mean]=np.where(NT_ST_mean[index_LT_mean]*100>=65 ,1,0)

#Detected NT in NT files
NT_NT_detected_best3mean=np.ones((np.shape(NT_LT_best3mean)))
NT_NT_detected_best3mean[np.nonzero(NT_LT_detected_best3mean==1)]-=NT_LT_detected_best3mean[np.nonzero(NT_LT_detected_best3mean==1)]
NT_NT_detected_best3mean[np.nonzero(NT_ST_detected_best3mean==1)]-=NT_ST_detected_best3mean[np.nonzero(NT_ST_detected_best3mean==1)]
NT_NT_detected_best3mean[np.nonzero(NT_BT_detected_best3mean==1)]-=NT_BT_detected_best3mean[np.nonzero(NT_BT_detected_best3mean==1)]
#LT_NT_detected_best3mean=np.where(LT_NT_detected_best3mean>0,1,0) it should not be necessary now
NT_NT_detected_max=np.ones((np.shape(NT_LT_max)))
NT_NT_detected_max[np.nonzero(NT_LT_detected_max==1)]-=NT_LT_detected_max[np.nonzero(NT_LT_detected_max==1)]
NT_NT_detected_max[np.nonzero(NT_ST_detected_max==1)]-=NT_ST_detected_max[np.nonzero(NT_ST_detected_max==1)]
NT_NT_detected_max[np.nonzero(NT_BT_detected_max==1)]-=NT_BT_detected_max[np.nonzero(NT_BT_detected_max==1)]
#LT_NT_detected_max=np.where(LT_NT_detected_max>0,1,0)
NT_NT_detected_mean=np.ones((np.shape(NT_LT_mean)))
NT_NT_detected_mean[np.nonzero(NT_LT_detected_mean==1)]-=NT_LT_detected_mean[np.nonzero(NT_LT_detected_mean==1)]
NT_NT_detected_mean[np.nonzero(NT_ST_detected_mean==1)]-=NT_ST_detected_mean[np.nonzero(NT_ST_detected_mean==1)]
NT_NT_detected_mean[np.nonzero(NT_BT_detected_mean==1)]-=NT_BT_detected_mean[np.nonzero(NT_BT_detected_mean==1)]

#plot
fig, axes=plt.subplots(4,3,sharex='all', sharey='all')
fig.suptitle('Detections in NT files')
#LT first row
axes[0][0].plot(NT_filenames, NT_LT_best3mean, 'b', NT_LT_detected_best3mean, 'r')
axes[0][0].set_title('Best 3 mean')
axes[0][0].set_ylabel('LT', rotation=0, size='large')
axes[0][1].plot(NT_filenames, NT_LT_max, 'b', NT_LT_detected_max, 'r')
axes[0][1].set_title('Max', rotation=0)
axes[0][2].plot(NT_filenames, NT_LT_mean, 'b', NT_LT_detected_mean, 'r')
axes[0][2].set_title('Mean')
axes[0][2].axes.xaxis.set_ticklabels([])

#ST second row
axes[1][0].plot(NT_filenames, NT_ST_best3mean, 'g', NT_ST_detected_best3mean, 'r')
axes[1][0].set_ylabel('ST', rotation=0, size='large')
axes[1][1].plot(NT_filenames, NT_ST_max, 'g', NT_ST_detected_max, 'r')
axes[1][2].plot(NT_filenames, NT_ST_mean, 'g', NT_ST_detected_mean, 'r')
axes[1][2].axes.xaxis.set_ticklabels([])

#BT Third row
axes[2][0].plot(NT_filenames,NT_LT_best3mean, 'b', NT_ST_best3mean, 'g', NT_BT_detected_best3mean, 'r')
axes[2][0].set_ylabel('BT', rotation=0, size='large')
axes[2][1].plot(NT_filenames, NT_LT_max, 'b', NT_ST_max, 'g', NT_BT_detected_max, 'r')
axes[2][2].plot(NT_filenames, NT_LT_mean, 'b', NT_ST_mean, 'g', NT_BT_detected_mean, 'r')
axes[2][2].axes.xaxis.set_ticklabels([])

#NT Forth row
axes[3][0].plot(NT_filenames, NT_NT_best3mean, 'b', NT_NT_detected_best3mean, 'r')
axes[3][0].set_ylabel('NT', rotation=0, size='large')
axes[3][1].plot(NT_filenames, NT_NT_max, 'b', NT_NT_detected_max, 'r')
axes[3][2].plot(NT_filenames, NT_NT_mean, 'b', NT_NT_detected_mean, 'r')
axes[3][2].axes.xaxis.set_ticklabels([])
#hide labels for inner plots
for ax in axes.flat:
    ax.label_outer()
#plt.show()
plt.savefig("D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Test02\\NT_detections.png")

#save false positives

NT_filenames_a=np.reshape(NT_filenames,(len(NT_filenames),1))
#ST
FP_ST_best3mean=NT_filenames_a[np.nonzero(NT_ST_detected_best3mean)]
FP_ST_max=NT_filenames_a[np.nonzero(NT_ST_detected_max)]
FP_ST_mean=NT_filenames_a[np.nonzero(NT_ST_detected_mean)]
#LT
FP_LT_best3mean=NT_filenames_a[np.nonzero(NT_LT_detected_best3mean)]
FP_LT_max=NT_filenames_a[np.nonzero(NT_LT_detected_max)]
FP_LT_mean=NT_filenames_a[np.nonzero(NT_LT_detected_mean)]
#BT
FP_BT_best3mean=NT_filenames_a[np.nonzero(NT_BT_detected_best3mean)]
FP_BT_max=[np.nonzero(NT_BT_detected_max)]
FP_BT_mean=NT_filenames_a[np.nonzero(NT_BT_detected_mean)]

#save on a file
f= open('D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Test02\\false_positives.txt', 'w')
f.write('BEST 3 MEAN \n')
f.write('LT \n')
json.dump(FP_LT_best3mean.tolist(),f)
f.write('\n ST \n')
json.dump(FP_ST_best3mean.tolist(),f)
f.write('\n BT \n')
json.dump(FP_BT_best3mean.tolist(),f)
f.write('\n\n MAX \n')
f.write('LT \n')
json.dump(FP_LT_max.tolist(),f)
f.write('\n ST \n')
json.dump(FP_ST_max.tolist(),f)
f.write('\n BT \n')
json.dump(FP_ST_max.tolist(),f)
f.write('\n\n  Mean \n')
f.write(' LT \n')
json.dump(FP_LT_mean.tolist(),f)
f.write('\n ST \n')
json.dump(FP_ST_mean.tolist(),f)
f.write('\n BT \n')
json.dump(FP_BT_mean.tolist(),f)
f.close()

##BT files plot

#detect LT in BT files
index_LT_best3mean=np.nonzero(BT_LT_best3mean*100>=65)
BT_LT_detected_best3mean=np.zeros((np.shape(BT_LT_best3mean)))
BT_LT_detected_best3mean[index_LT_best3mean]=np.where(BT_ST_best3mean[index_LT_best3mean]*100<65,1,0)

index_LT_max=np.nonzero(BT_LT_max*100>=65)
BT_LT_detected_max=np.zeros((np.shape(BT_LT_max)))
BT_LT_detected_max[index_LT_max]=np.where(BT_ST_max[index_LT_max]*100<65,1,0)

index_LT_mean=np.nonzero(BT_LT_mean*100>=65)
BT_LT_detected_mean=np.zeros((np.shape(BT_LT_mean)))
BT_LT_detected_mean[index_LT_mean]=np.where(BT_ST_mean[index_LT_mean]*100<65,1,0) 

#detect ST in BT files
index_ST_best3mean=np.nonzero(BT_ST_best3mean*100>=65)
BT_ST_detected_best3mean=np.zeros((np.shape(BT_ST_best3mean)))
BT_ST_detected_best3mean[index_ST_best3mean]=np.where(BT_LT_best3mean[index_ST_best3mean]*100<65,1,0)

index_ST_max=np.nonzero(BT_ST_max*100>=65)
BT_ST_detected_max=np.zeros((np.shape(BT_ST_max)))
BT_ST_detected_max[index_ST_max]=np.where(BT_LT_max[index_ST_max]*100<65,1,0)

index_ST_mean=np.nonzero(BT_ST_mean*100>=65)
BT_ST_detected_mean=np.zeros((np.shape(BT_ST_mean)))
BT_ST_detected_mean[index_ST_mean]=np.where(BT_LT_mean[index_ST_mean]*100<65,1,0) 

#detect BT in BT files
BT_BT_detected_best3mean=np.zeros(np.shape(BT_LT_detected_best3mean))
#in this way we are asking both the condition to be satisfied
BT_BT_detected_best3mean[index_LT_best3mean]=np.where(BT_ST_best3mean[index_LT_best3mean]*100>=65 ,1,0) 

BT_BT_detected_max=np.zeros(np.shape(BT_LT_detected_max))
#in this way we are asking both the condition to be satisfied
BT_BT_detected_max[index_LT_max]=np.where(BT_ST_max[index_LT_max]*100>=65 ,1,0)

BT_BT_detected_mean=np.zeros(np.shape(BT_LT_detected_mean))
#in this way we are asking both the condition to be satisfied
BT_BT_detected_mean[index_LT_mean]=np.where(BT_ST_mean[index_LT_mean]*100>=65 ,1,0)

#Detected NT in BT files
BT_NT_detected_best3mean=np.ones((np.shape(BT_LT_best3mean)))
BT_NT_detected_best3mean[np.nonzero(BT_LT_detected_best3mean==1)]-=BT_LT_detected_best3mean[np.nonzero(BT_LT_detected_best3mean==1)]
BT_NT_detected_best3mean[np.nonzero(BT_ST_detected_best3mean==1)]-=BT_ST_detected_best3mean[np.nonzero(BT_ST_detected_best3mean==1)]
BT_NT_detected_best3mean[np.nonzero(BT_BT_detected_best3mean==1)]-=BT_BT_detected_best3mean[np.nonzero(BT_BT_detected_best3mean==1)]
#LT_NT_detected_best3mean=np.where(LT_NT_detected_best3mean>0,1,0) it should not be necessary now
BT_NT_detected_max=np.ones((np.shape(BT_LT_max)))
BT_NT_detected_max[np.nonzero(BT_LT_detected_max==1)]-=BT_LT_detected_max[np.nonzero(BT_LT_detected_max==1)]
BT_NT_detected_max[np.nonzero(BT_ST_detected_max==1)]-=BT_ST_detected_max[np.nonzero(BT_ST_detected_max==1)]
BT_NT_detected_max[np.nonzero(BT_BT_detected_max==1)]-=BT_BT_detected_max[np.nonzero(BT_BT_detected_max==1)]
#LT_NT_detected_max=np.where(LT_NT_detected_max>0,1,0)
BT_NT_detected_mean=np.ones((np.shape(BT_LT_mean)))
BT_NT_detected_mean[np.nonzero(BT_LT_detected_mean==1)]-=BT_LT_detected_mean[np.nonzero(BT_LT_detected_mean==1)]
BT_NT_detected_mean[np.nonzero(BT_ST_detected_mean==1)]-=BT_ST_detected_mean[np.nonzero(BT_ST_detected_mean==1)]
BT_NT_detected_mean[np.nonzero(BT_BT_detected_mean==1)]-=BT_BT_detected_mean[np.nonzero(BT_BT_detected_mean==1)]

## find missed BT files with different methods
# a missed file is a file that  it will be classified as noise

BT_filenames_a=np.reshape(BT_filenames,(len(BT_filenames),1))
BT_missed_best3mean=BT_filenames_a[np.nonzero(BT_NT_detected_best3mean)]
BT_missed_max=BT_filenames_a[np.nonzero(BT_NT_detected_max)]
BT_missed_mean=BT_filenames_a[np.nonzero(BT_NT_detected_mean)]

#plot
fig, axes=plt.subplots(4,3,sharex='all', sharey='all')
fig.suptitle('Detections in BT files')
#LT first row
axes[0][0].plot(BT_filenames, BT_LT_best3mean, 'bs', BT_LT_detected_best3mean, 'rx')
axes[0][0].set_title('Best 3 mean')
axes[0][0].set_ylabel('LT', rotation=0, size='large')
axes[0][1].plot(BT_filenames, BT_LT_max, 'bs', BT_LT_detected_max, 'rx')
axes[0][1].set_title('Max', rotation=0)
axes[0][2].plot(BT_filenames, BT_LT_mean, 'bs', BT_LT_detected_mean, 'rx')
axes[0][2].set_title('Mean')
axes[0][2].axes.xaxis.set_ticklabels([])

#ST second row
axes[1][0].plot(BT_filenames, BT_ST_best3mean, 'go', BT_ST_detected_best3mean, 'rx')
axes[1][0].set_ylabel('ST', rotation=0, size='large')
axes[1][1].plot(BT_filenames, BT_ST_max, 'go', BT_ST_detected_max, 'rx')
axes[1][2].plot(BT_filenames, BT_ST_mean, 'go', BT_ST_detected_mean, 'rx')
axes[1][2].axes.xaxis.set_ticklabels([])

#BT Third row
axes[2][0].plot(BT_filenames, BT_LT_best3mean,'bs', BT_ST_best3mean, 'go',  BT_BT_detected_best3mean, 'rx')
axes[2][0].set_ylabel('BT', rotation=0, size='large')
axes[2][1].plot(BT_filenames, BT_LT_max,'bs', BT_ST_max, 'go',  BT_BT_detected_max, 'rx')
axes[2][2].plot(BT_filenames, BT_LT_mean,'bs', BT_ST_mean, 'go',  BT_BT_detected_mean, 'rx')
axes[2][2].axes.xaxis.set_ticklabels([])

#NT Forth row
axes[3][0].plot(BT_filenames, BT_NT_best3mean, 'bs', BT_NT_detected_best3mean, 'xr')
axes[3][0].set_ylabel('NT', rotation=0, size='large')
axes[3][1].plot(BT_filenames, BT_NT_max, 'bs', BT_NT_detected_max, 'rx')
axes[3][2].plot(BT_filenames, BT_NT_mean, 'bs', BT_NT_detected_mean, 'rx')
axes[3][2].axes.xaxis.set_ticklabels([])
#hide labels for inner plots
for ax in axes.flat:
    ax.label_outer()
#plt.show()
plt.savefig("D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Test02\\BT_detections.png")

#save
f= open('D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Test02\\BT_missed.txt', 'w')
f.write('BEST 3 MEAN \n')
json.dump(BT_missed_best3mean.tolist(),f)
f.write('\n MAX \n')
json.dump(BT_missed_max.tolist(),f)
f.write('\n Mean \n')
json.dump(BT_missed_mean.tolist(),f)
f.close()
