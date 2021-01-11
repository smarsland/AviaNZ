"""
Script to visualize results from Bat Tests

"""

import csv
#import Segment
import SignalProc
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

################## MAIN ####################################


##Read TEST RESULTS
#result_csv_file="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Results\\20201016_tests\\Results.csv"
result_csv_file="C:\\Users\\Virginia\\Documents\\Work\Data\\Bats\\Results\\20201016_tests\\Results_row64.csv"
#result_csv_file="/am/state-opera/home1/listanvirg/Documents/Experiments_result/Results.csv"



Recall=[]
Precision_pre=[]
Precision_post=[]
train_id=[]
test_id=[]

with open(result_csv_file, 'r') as csvfile:
    csvreader=csv.DictReader(csvfile)
    for row in csvreader:
        Recall.append(dict(row)["RECALL"])
        Precision_pre.append(dict(row)["PRECISION_PRE"])
        Precision_post.append(dict(row)["PRECISION_POST"])
        print(dict(row)["PRECISION_POST"])
        #print(dict(row)["TEST"])
        test_id.append(dict(row)["TEST_ID"])
        train_id.append(dict(row)["TRAIN DATASET"])

#index_conf_0=np.arange(0,73,12)
#index_conf_0=np.array(index_conf_0,dtype='int
#test_id=np.array(test_id)
Recall=np.array(Recall, dtype=np.float32)
Precision_pre=np.array(Precision_pre, dtype=np.float32)
Precision_post=np.array(Precision_post, dtype=np.float32)

#READ ERROR tables
R1_path='C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Moira 2020\\Raw files\\Bat_tests\\Test_dataset\\R1'
R13_path='C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Moira 2020\\Raw files\\Bat_tests\\Test_dataset\\R13'
R18_path='C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Moira 2020\\Raw files\\Bat_tests\\Test_dataset\\R18'
dir_path_list=[R1_path, R13_path, R18_path]
#print('len(test_id)=', len(test_id))
count=0
CNN=np.zeros((len(train_id),1))
CD=np.zeros((len(train_id),1))
CD_p=np.zeros((len(train_id),1))
Lab=np.zeros((len(train_id),1))
for test in test_id:
    CNN[count]=0
    CD[count]=0
    CD_p[count]=0
    Lab[count]=0
    for dir in dir_path_list:
        fp_file=dir+'\\Test_New_row64_'+str(test)+'\\false_positives.csv'
        fn_file=dir+'\\Test_New_row64_'+str(test)+'\\missed_files.csv'
        misc_file=dir+'\\Test_New_row64_'+str(test)+'\\misclassified_files.csv'
        file_list=[fp_file, fn_file, misc_file]
        for file in file_list:
            with open(file, 'r') as csvfile:
                csvreader=csv.DictReader(csvfile)
                for row in csvreader:
                    if 'CNN' in dict(row)["Possible Error"]:
                        CNN[count]+=1
                    elif 'CD' in dict(row)["Possible Error"]:
                        CD[count]+=1
                    elif '?' in dict(row)["Possible Error"]:
                        CD_p[count]+=1
                    elif 'Label' in dict(row)["Possible Error"]:
                        Lab[count]+=1
    count+=1
print(np.shape(CNN))
print(type(CNN))

im_file=['win_3','win_11','win_17','win_31']
title=['Window 3x2', 'Window 11', 'Window 17', 'Window 31']

for k in range(0,4):
    #path to save
    #image_path="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Results\\20201016_tests\\Results_test_"+im_file[k]+".png"
    image_path="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Results\\20201016_tests\\Results_test_"+im_file[k]+"_row64.png"

    #fig.set_ylim(0,100)
    fig, ax = plt.subplots(2,3, sharey='row', sharex='col')
    fig.suptitle(title[k])
    for i in range(0,3):
    #ax.set_ylim(40,100)
        index_conf_0=slice(3*k+i,48,12)
        print(i, index_conf_0)
        ax[0,i].set_title('File Label'+str(i))
        ax[0,i].plot(train_id[index_conf_0],Recall[index_conf_0], 'r', label='Rec')
        ax[0,i].plot(train_id[index_conf_0], Precision_pre[index_conf_0], 'b', label='Pre_pre')
        ax[0,i].plot(train_id[index_conf_0], Precision_post[index_conf_0], 'g', label='Pre_post')
        ax[0,i].legend(loc='lower right')

        x = np.arange(len(train_id[index_conf_0]))  # the label locations
        width = 0.15  # the width of the bars
        rects1 = ax[1,i].bar(x - (3/2)*width, CNN[index_conf_0,0], width, label='CNN')
        rects2 = ax[1,i].bar(x - width/2, CD[index_conf_0,0], width, label='CD')
        rects3 = ax[1,i].bar(x + width/2, CD_p[index_conf_0,0], width, label='CD?')
        rects4 = ax[1,i].bar(x + (3/2)*width, Lab[index_conf_0,0], width, label='Label')
        ax[1,i].legend()

    plt.savefig(image_path)





       
