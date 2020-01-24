"""
Authors: Stephen Marsland, Virginia Listanti

This code recover information from files and then plot max, min and mean for 
the best 3 in 4 different plots: LT, ST, NT and BT

It does this for all file classification with all possible spectrogram classification

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
f = open('D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study2\\ST_spec_prob.data')
a = json.load(f)
f.close()

f = open('D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study2\LT_spec_prob.data')
b = json.load(f)
f.close()

f = open('D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study2\\Noise_spec_prob.data')
c = json.load(f)
f.close()


f = open('D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study2\\Both_spec_prob.data')
d = json.load(f)
f.close()
# a[i][0] is filename, [1] is the species label, [2] is the np array

## Clicks statistics in ST files
ST_ST = [] #save informations for ST file with ST clicks
ST_NT = [] #save informations for ST file with Noise clicks
count = 0
#aid variables
st_st=[]
st_nt=[]
while count<len(a):
    file = a[count][0]
    label=a[count][1]
    while count<len(a) and a[count][0] == file:
        if label==1:
            st_st.append(a[count][2][1])
        elif label==2:
            st_nt.append(a[count][2][2])
        count+=1
    #if lists are  empty assigning zero ->it means no clicks of that kind detected
    if st_st==[]:
        st_st=0
    if st_nt==[]:
        st_nt=0
    ST_ST.append([file,st_st])
    ST_NT.append([file,st_nt])
    st_st = []
    st_nt =[]

#metrics for ST_ST files
ST_ST_max=np.zeros((len(ST_ST),1))
ST_ST_min=np.zeros((len(ST_ST),1))
ST_ST_mean=np.zeros((len(ST_ST),1))
ST_ST_best3mean=np.zeros((len(ST_ST),1))
ST_filenames=[]
#print(ST_ST)
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

##LT CLICKS BY FILE    
LT_LT = [] #save informations for LT file with LT clicks
LT_NT = [] #save informations for LT file with Noise clicks
count = 0
#aid variables
lt_lt=[]
lt_nt=[]
while count<len(b):
    file = b[count][0]
    label=b[count][1]
    while count<len(b) and b[count][0] == file:
        if label==0:
            lt_lt.append(b[count][2][0])
        elif label==2:
            lt_nt.append(b[count][2][2])
        count+=1
    #if lists are  empty assigning zero ->it means no clicks of that kind detected
    if lt_lt==[]:
        lt_lt=0
    if lt_nt==[]:
        lt_nt=0
    LT_LT.append([file,lt_lt])
    LT_NT.append([file,lt_nt])
    lt_lt = []
    lt_nt =[]

#metrics for LT_LT files
    #note LT_LT len is equal to LT_NT len
LT_LT_max=np.zeros((len(LT_LT),1))
LT_LT_min=np.zeros((len(LT_LT),1))
LT_LT_mean=np.zeros((len(LT_LT),1))
LT_LT_best3mean=np.zeros((len(LT_LT),1))
LT_filenames=[]
for i in range(len(LT_LT)):
    LT_filenames.append(LT_LT[i][0])
    LT_LT_max[i] = np.max(LT_LT[i][1])
    LT_LT_min[i] = np.min(LT_LT[i][1])
#    LT_LT_in = min(LT_LT[i][1])
    LT_LT_best3mean[i]= 0
    ind = np.array(LT_LT[i][1]).argsort()[-3:][::-1]
    if len(ind)==1:
        #this means that there is only one prob!
        LT_LT_best3mean+=LT_LT[i][1]
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
        LT_NT_best3mean+=LT_NT[i][1]
    else:
        for j in range(len(ind)):
            LT_NT_best3mean[i]+=LT_NT[i][1][ind[j]]
    LT_NT_best3mean[i]/= len(ind)
    LT_NT_mean[i]=np.mean(LT_NT[i][1])    

## NOISE CLICKS ordered by file

NT_NT = [] #save informations for Noise file with Noise clicks
count = 0
#aid variable: I just need one because the file label it is consistent in all files
nt=[]
while count<len(c):
    file = c[count][0]
    label=c[count][1]
    while count<len(c) and c[count][0] == file :
#        if c[count][0]!=0:
        c[count][2]=np.reshape(c[count][2],(3,))
        print(c[count][2])
        print(np.shape(c[count][2]))
        nt.append(c[count][2][2])
        count+=1
        
    NT_NT.append([file,nt])
    nt = []

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
        NT_NT_best3mean+=NT_NT[i][1]
    else:
        for j in range(len(ind)):
            NT_NT_best3mean[i]+=NT_NT[i][1][ind[j]]
    NT_NT_best3mean[i]/= len(ind)
    NT_NT_mean[i]=np.mean(NT_NT[i][1])    
    
##clicks in BT files    
BT_LT = [] #save informations for BT file with LT clicks
BT_ST = [] #save informations for BT file with ST clicks
BT_NT = [] #save informations for BT file with Noise clicks
count = 0
#aid variables
bt_lt=[]
bt_st=[]
bt_nt=[]
while count<len(d):
    file = d[count][0]
    label=d[count][1]
    while count<len(d) and d[count][0] == file:
        if label==0:
            bt_lt.append(d[count][2][0])
        elif label==0:
            bt_st.append(d[count][2][1])
        elif label==2:
            bt_nt.append(d[count][2][2])
        count+=1
    #if lists are  empty assigning zero  ->it means no clicks of that kind detected
    if bt_lt==[]:
        print('check')
        bt_lt=0
    if bt_nt==[]:
        print('check')
        bt_nt=0
    if bt_st==[]:
        bt_st=0
    BT_LT.append([file,bt_lt])
    BT_ST.append([file,bt_st])
    BT_NT.append([file,bt_nt])
    bt_lt = []
    bt_st = []
    bt_nt = []

print('BT_LT')
print(BT_LT)
print('BT_ST')
print(BT_ST)
print('BT_NT')
print(BT_NT)
#metrics for BT_LT files
    #note  BT_LT len is equal to BT_NT and to BT_ST len
BT_LT_max=np.zeros((len(BT_LT),1))
BT_LT_min=np.zeros((len(BT_LT),1))
BT_LT_mean=np.zeros((len(BT_LT),1))
BT_LT_best3mean=np.zeros((len(BT_LT),1))
BT_filenames=[]
for i in range(len(BT_LT)):
    print(np.shape(BT_LT[i][1]))
    print(BT_LT[i][1])
    BT_filenames.append(BT_LT[i][0])
    BT_LT_max[i] = np.max(BT_LT[i][1])
    BT_LT_min[i] = np.min(BT_LT[i][1])
#    LT_LT_in = min(LT_LT[i][1])
    BT_LT_best3mean[i]= 0
    ind = np.array(BT_LT[i][1]).argsort()[-3:][::-1]
    if len(ind)==1:
        #this means that there is only one prob!
        BT_LT_best3mean+=BT_LT[i][1]
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
    print(np.shape(BT_NT[i][1]))
    BT_NT_max[i] = np.max(BT_NT[i][1])
    BT_NT_min[i] = np.min(BT_NT[i][1])
#    LT_BT_in = min(LT_BT[i][1])
    BT_NT_best3mean[i] = 0
    ind = np.array(BT_NT[i][1]).argsort()[-3:][::-1]
    if len(ind)==1:
        #this means that there is only one prob!
        BT_NT_best3mean+=BT_NT[i][1]
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
        BT_ST_best3mean+=BT_ST[i][1]
    else:
        for j in range(len(ind)):
            BT_ST_best3mean[i]+=BT_ST[i][1][ind[j]]
    BT_ST_best3mean[i]/= len(ind)
    BT_ST_mean[i]=np.mean(BT_ST[i][1])      
    
#PLOTS FOR LT files
#print(LT_LT)
#print('LT_mean shape', np.shape(LT_LT_mean), np.shape(LT_LT_best3mean))
#print('file name shape', np.shape(LT_filenames))
#print('max shape', np.shape(LT_LT_max))
#print('min shape',print(np.shape(LT_LT_min)))
#fig, axes=plt.subplots(4,2,sharex='col', sharey='all')
#fig.suptitle('LT files')
##fig.autofmt_xdate(rotation=45, size='small')
#axes[0][0].plot(LT_filenames, LT_LT_max, 'k')
#axes[0][0].set_title('LT clicks',size='large')
#axes[0][0].set_ylabel('Maxima', rotation=0, size='large')
#axes[1][0].plot(LT_filenames, LT_LT_min, 'r')
#axes[1][0].set_ylabel('Minima', rotation=0, size='large')
#axes[2][0].plot(LT_filenames, LT_LT_best3mean, 'g')
#axes[2][0].set_ylabel('Best 3 mean', rotation=0, size='large')
#axes[3][0].plot(LT_filenames, LT_LT_mean, 'b')
#axes[3][0].set_ylabel('Mean', rotation=0, size='large')
##hide x-axis label
#axes[3][0].axes.xaxis.set_ticklabels([])
##axes[3][0].set_xticklabels(LT_filenames, rotation=45, ha='right')
##axes[3][0].set_xticks(LT_filenames, rotation=45)
#axes[0][1].plot(LT_filenames, LT_NT_max, 'k')
#axes[0][1].set_title('Noise clicks',size='large')
#axes[1][1].plot(LT_filenames, LT_NT_min, 'r')
#axes[2][1].plot(LT_filenames, LT_NT_best3mean, 'g')
#axes[3][1].plot(LT_filenames, LT_NT_mean, 'b')
#axes[3][1].axes.xaxis.set_ticklabels([])
#
##hide labels for inner plots
#for ax in axes.flat:
#    ax.label_outer()
##plt.show()
#plt.savefig("D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study2\\LT_files.png")
##exporter = pge.ImageExporter(fig.view)
##exporter.export("D:\Desktop\Documents\Work\Data\Bat\BAT\CNN experiment\TEST2\BAT SEARCH TESTS\Test_79\LT_files.png")
#
##PLOTS FOR ST files
##print(LT_LT)
#fig, axes=plt.subplots(4,2,sharex='col', sharey='all')
#fig.suptitle('ST files')
##fig.autofmt_xdate(rotation=45, size='small')
#axes[0][0].plot(ST_filenames, ST_ST_max, 'k')
#axes[0][0].set_title('ST clicks',size='large')
#axes[0][0].set_ylabel('Maxima', rotation=0, size='large')
#axes[1][0].plot(ST_filenames, ST_ST_min, 'r')
#axes[1][0].set_ylabel('Minima', rotation=0, size='large')
#axes[2][0].plot(ST_filenames, ST_ST_best3mean, 'g')
#axes[2][0].set_ylabel('Best 3 mean', rotation=0, size='large')
#axes[3][0].plot(ST_filenames, ST_ST_mean, 'b')
#axes[3][0].set_ylabel('Mean', rotation=0, size='large')
##hide x-axis label
#axes[3][0].axes.xaxis.set_ticklabels([])
##axes[3][0].set_xticklabels(LT_filenames, rotation=45, ha='right')
##axes[3][0].set_xticks(LT_filenames, rotation=45)
#axes[0][1].plot(ST_filenames, ST_NT_max, 'k')
#axes[0][1].set_title('Noise clicks',size='large')
#axes[1][1].plot(ST_filenames, ST_NT_min, 'r')
#axes[2][1].plot(ST_filenames, ST_NT_best3mean, 'g')
#axes[3][1].plot(ST_filenames, ST_NT_mean, 'b')
#axes[3][1].axes.xaxis.set_ticklabels([])
#
##hide labels for inner plots
#for ax in axes.flat:
#    ax.label_outer()
##plt.show()
#plt.savefig("D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study1\\ST_files.png")
#
##PLOTS FOR BT files
##print(LT_LT)
#print('BT_LT mean shape', np.shape(BT_LT_mean), np.shape(BT_LT_best3mean))
#print('file name shape', np.shape(BT_filenames))
#print('max shape', np.shape(BT_LT_max))
#print('min shape',np.shape(BT_LT_min))
#fig, axes=plt.subplots(4,3,sharex='col', sharey='all')
#fig.suptitle('Both files')
##fig.autofmt_xdate(rotation=45, size='small')
#axes[0][0].plot(BT_filenames, BT_ST_max, 'k')
#axes[0][0].set_title('ST clicks',size='large')
#axes[0][0].set_ylabel('Maxima', rotation=0, size='large')
#axes[1][0].plot(BT_filenames, BT_ST_min, 'r')
#axes[1][0].set_ylabel('Minima', rotation=0, size='large')
#axes[2][0].plot(BT_filenames, BT_ST_best3mean, 'g')
#axes[2][0].set_ylabel('Best 3 mean', rotation=0, size='large')
#axes[3][0].plot(BT_filenames, BT_ST_mean, 'b')
#axes[3][0].set_ylabel('Mean', rotation=0, size='large')
##hide x-axis label
#axes[3][0].axes.xaxis.set_ticklabels([])
#axes[0][1].plot(BT_filenames, BT_LT_max, 'k')
#axes[0][1].set_title('LT clicks',size='large')
#axes[1][1].plot(BT_filenames, BT_LT_min, 'r')
#axes[2][1].plot(BT_filenames, BT_LT_best3mean, 'g')
#axes[3][1].plot(BT_filenames, BT_LT_mean, 'b')
##hide x-axis label
#axes[3][1].axes.xaxis.set_ticklabels([])
##axes[3][0].set_xticklabels(LT_filenames, rotation=45, ha='right')
##axes[3][0].set_xticks(LT_filenames, rotation=45)
#axes[0][2].plot(BT_filenames, BT_NT_max, 'k')
#axes[0][2].set_title('Noise clicks',size='large')
#axes[1][2].plot(BT_filenames, BT_NT_min, 'r')
#axes[2][2].plot(BT_filenames, BT_NT_best3mean, 'g')
#axes[3][2].plot(BT_filenames, BT_NT_mean, 'b')
#axes[3][2].axes.xaxis.set_ticklabels([])
#
##hide labels for inner plots
#for ax in axes.flat:
#    ax.label_outer()
##plt.show()
#plt.savefig("D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study1\\Both_files.png")

#fig, axes=plt.subplots(4,sharex='col')
#fig.suptitle('Noise files')
##fig.autofmt_xdate(rotation=45, size='small')
#axes[0].plot(NT_filenames, NT_NT_max, 'k')
#axes[0].set_title('NT clicks',size='large')
#axes[0].set_ylabel('Maxima', rotation=0, size='large')
#axes[1].plot(NT_filenames, NT_NT_min, 'r')
#axes[1].set_ylabel('Minima', rotation=0, size='large')
#axes[2].plot(NT_filenames, NT_NT_best3mean, 'g')
#axes[2].set_ylabel('Best 3 mean', rotation=0, size='large')
#axes[3].plot(NT_filenames, NT_NT_mean, 'b')
#axes[3].set_ylabel('Mean', rotation=0, size='large')
##hide x-axis label
#axes[3].axes.xaxis.set_ticklabels([])
##hide labels for inner plots
#for ax in axes.flat:
#    ax.label_outer()
##plt.show()
#plt.savefig("D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study1\\Noise_files.png")

#detect LT
LT_detected_best3mean=np.where(LT_LT_best3mean*100>70,1,0)
LT_detected_max=np.where(LT_LT_max*100>70,1,0)
LT_detected_mean=np.where(LT_LT_mean*100>70,1,0)
fig, axes=plt.subplots(3,sharex='col')
fig.suptitle('LT detected')
axes[0].plot(LT_filenames, LT_LT_best3mean, 'b', LT_detected_best3mean, 'r')
axes[0].set_ylabel('Best 3 mean', rotation=0, size='large')
axes[1].plot(LT_filenames, LT_LT_max, 'b', LT_detected_max, 'r')
axes[1].set_ylabel('Max', rotation=0, size='large')
axes[2].plot(LT_filenames, LT_LT_mean, 'b', LT_detected_mean, 'r')
axes[2].set_ylabel('Mean', rotation=0, size='large')
axes[2].axes.xaxis.set_ticklabels([])
#hide labels for inner plots
for ax in axes.flat:
    ax.label_outer()
#plt.show()
plt.savefig("D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study2\\Test01\\LT_detections.png")

#detect ST
ST_detected_best3mean=np.where(ST_ST_best3mean*100>70,1,0)
ST_detected_max=np.where(ST_ST_max*100>70,1,0)
ST_detected_mean=np.where(ST_ST_mean*100>70,1,0)
fig, axes=plt.subplots(3,sharex='col')
fig.suptitle('ST detected')
axes[0].plot(ST_filenames, ST_ST_best3mean, 'b', ST_detected_best3mean, 'r')
axes[0].set_ylabel('Best 3 mean', rotation=0, size='large')
axes[1].plot(ST_filenames, ST_ST_max, 'b', ST_detected_max, 'r')
axes[1].set_ylabel('Max', rotation=0, size='large')
axes[2].plot(ST_filenames, ST_ST_mean, 'b', ST_detected_mean, 'r')
axes[2].set_ylabel('Mean', rotation=0, size='large')
axes[2].axes.xaxis.set_ticklabels([])
#hide labels for inner plots
for ax in axes.flat:
    ax.label_outer()
#plt.show()
plt.savefig("D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study2\\Test01\\ST_detections.png")

#detect LT
BT_detected_best3mean=np.where(BT_LT_best3mean*100>50,1,0)+np.where(BT_ST_best3mean*100>50,1,0)
BT_detected_max=np.where(BT_LT_max*100>50,1,0)+np.where(BT_ST_max*100>50,1,0)
BT_detected_mean=np.where(BT_LT_mean*100>50,1,0)+np.where(BT_ST_mean*100>50,1,0)
fig, axes=plt.subplots(3,sharex='col')
fig.suptitle('BT detected')
axes[0].plot(BT_filenames, BT_LT_best3mean, 'b',  BT_ST_best3mean, 'g', BT_detected_best3mean, 'r')
axes[0].set_ylabel('Best 3 mean', rotation=0, size='large')
axes[1].plot(BT_filenames, BT_LT_max, 'b', BT_ST_max, 'k', BT_detected_max, 'r')
axes[1].set_ylabel('Max', rotation=0, size='large')
axes[2].plot(BT_filenames, BT_LT_mean, 'b', BT_ST_mean, 'k', BT_detected_mean, 'r')
axes[2].set_ylabel('Mean', rotation=0, size='large')
axes[2].axes.xaxis.set_ticklabels([])
#hide labels for inner plots
for ax in axes.flat:
    ax.label_outer()
#plt.show()
plt.savefig("D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study2\\Test01\\BT_detections.png")