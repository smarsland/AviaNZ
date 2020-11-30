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

result_csv_file="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Results\\20201016_tests\\Results.csv"


Recall=[]
Precision_pre=[]
Precision_post=[]
with open(result_csv_file, 'r') as csvfile:
    csvreader=csv.DictReader(csvfile)
    for row in csvreader:
        Recall.append(dict(row)["RECALL"])
        Precision_pre.append(dict(row)["PRECISION_PRE"])
        Precision_post.append(dict(row)["PRECISION_POST"])


#Recall=np.array(Recall)
#Precision_pre=np.array(Precision_pre)
#Precision_post=np.array(Precision_post)

print(Recall[0], Recall[12], Recall[24], Recall[36])
print(Precision_pre[0], Precision_pre[12], Precision_pre[24], Precision_pre[36])
print(Precision_post[0], Precision_post[12], Precision_post[24], Precision_post[36])
x_axis_0=[0,1,2,3]
x_axis_1=[0,1,2,3, 3.1]
x_axis_2=[0,1,2]


#path to save
image_path="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Results\\20201016_tests\\Results.png"

#plot clicks probabilities + save images
    #Each row has a differen plot: LT, ST NT
    # Lt first columin, st second, nt third
fig, axes = plt.subplots(2,3)
#plt.ylim(0,100)
#first row
#axes[0,0].set_ylim([0,100])
axes[0,0].set_title("win=3x2 File_lab=0")
#axes[0,0].plot(x_axis_0,[Recall[0], Recall[12], Recall[24], Recall[36]],'r', x_axis_0, [Precision_pre[0], Precision_pre[12], Precision_pre[24], Precision_pre[36]],'b', x_axis_0,[Precision_post[0], Precision_post[12], Precision_post[24], Precision_post[36]],'g')
l1,= axes[0,0].plot(x_axis_0,[Recall[0], Recall[12], Recall[24], Recall[36]],'r')
l2, =axes[0,0].plot(x_axis_0,[Precision_pre[0], Precision_pre[12], Precision_pre[24], Precision_pre[36]],'b')
l3, =axes[0,0].plot(x_axis_0,[Precision_post[0], Precision_post[12], Precision_post[24], Precision_post[36]],'g')
#axes[0,0].set_ylim((0,100))

axes[0,1].set_title("win=3x2 File_lab=1")
axes[0,1].plot(x_axis_1,[Recall[1], Recall[13], Recall[25], Recall[37], Recall[38]],'r')
axes[0,1].plot(x_axis_1,[Precision_pre[1], Precision_pre[13], Precision_pre[25], Precision_pre[37], Precision_pre[38]],'b')
axes[0,1].plot(x_axis_1,[Precision_post[1], Precision_post[13], Precision_post[25], Precision_post[37], Precision_post[38]],'g')

axes[0,2].set_title("win=3x2 File_lab=2")
axes[0,2].plot(x_axis_0,[Recall[2], Recall[14], Recall[26], Recall[39]],'r')
axes[0,2].plot(x_axis_0,[Precision_pre[2], Precision_pre[14], Precision_pre[26], Precision_pre[39]],'b')
axes[0,2].plot(x_axis_0,[Precision_post[2], Precision_post[14], Precision_post[26], Precision_post[39]],'g')

#second row
axes[1,0].set_title("win=11 File_lab=0")
axes[1,0].plot(x_axis_0,[Recall[3], Recall[15], Recall[27], Recall[40]],'r')
axes[1,0].plot(x_axis_0,[Precision_pre[3], Precision_pre[15], Precision_pre[27], Precision_pre[40]],'b')
axes[1,0].plot(x_axis_0,[Precision_post[3], Precision_post[15], Precision_post[27], Precision_post[40]],'g')

axes[1,1].set_title("win=11 File_lab=1")
axes[1,1].plot(x_axis_0,[Recall[4], Recall[16], Recall[28], Recall[41]],'r')
axes[1,1].plot(x_axis_0,[Precision_pre[4], Precision_pre[16], Precision_pre[28], Precision_pre[41]],'b')
axes[1,1].plot(x_axis_0,[Precision_post[4], Precision_post[16], Precision_post[28], Precision_post[41]],'g')

axes[1,2].set_title("win=11 File_lab=2")
axes[1,2].plot(x_axis_0,[Recall[5], Recall[17], Recall[29], Recall[42]],'r')
axes[1,2].plot(x_axis_0,[Precision_pre[5], Precision_pre[17], Precision_pre[29], Precision_pre[42]],'b')
axes[1,2].plot(x_axis_0,[Precision_post[5], Precision_post[17], Precision_post[29], Precision_post[42]],'g')

plt.tight_layout()

plt.savefig(image_path)


#path to save
image_path="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Results\\20201016_tests\\Results2.png"

#plot clicks probabilities + save images
    #Each row has a differen plot: LT, ST NT
    # Lt first columin, st second, nt third
fig, axes = plt.subplots(2,3, sharey="row")

#2 plots
#third row
axes[0,0].set_title("win=17 File_lab=0")
axes[0,0].plot(x_axis_0,[Recall[6], Recall[18], Recall[30], Recall[43]],'r')
axes[0,0].plot(x_axis_0,[Precision_pre[6], Precision_pre[18], Precision_pre[30], Precision_pre[43]],'b')
axes[0,0].plot(x_axis_0,[Precision_post[6], Precision_post[18], Precision_post[30], Precision_post[43]],'g')

axes[0,1].set_title("win=17 File_lab=1")
axes[0,1].plot(x_axis_0,[Recall[7], Recall[19], Recall[31], Recall[44]],'r')
axes[0,1].plot(x_axis_0,[Precision_pre[7], Precision_pre[19], Precision_pre[31], Precision_pre[44]],'b')
axes[0,1].plot(x_axis_0,[Precision_post[7], Precision_post[19], Precision_post[31], Precision_post[44]],'g')

axes[0,2].set_title("win=17 File_lab=2")
axes[0,2].plot(x_axis_0,[Recall[8], Recall[20], Recall[32],Recall[45]],'r')
axes[0,2].plot(x_axis_0,[Precision_pre[8], Precision_pre[20], Precision_pre[32], Precision_pre[45]],'b')
axes[0,2].plot(x_axis_0,[Precision_post[8], Precision_post[20], Precision_post[32], Precision_post[45]],'g')

#forth row
axes[1,0].set_title("win=31 File_lab=0")
axes[1,0].plot(x_axis_2,[Recall[9], Recall[21], Recall[33]],'r')
axes[1,0].plot(x_axis_2,[Precision_pre[9], Precision_pre[21], Precision_pre[33]],'b')
axes[1,0].plot(x_axis_2,[Precision_post[9], Precision_post[21], Precision_post[33]],'g')

axes[1,1].set_title("win=31 File_lab=1")
axes[1,1].plot(x_axis_2,[Recall[10], Recall[22], Recall[34]],'r')
axes[1,1].plot(x_axis_2,[Precision_pre[10], Precision_pre[22], Precision_pre[34]],'b')
axes[1,1].plot(x_axis_2,[Precision_post[10], Precision_post[22], Precision_post[34]],'g')

axes[1,2].set_title("win=31 File_lab=2")
axes[1,2].plot(x_axis_2,[Recall[11], Recall[23], Recall[35]],'r')
axes[1,2].plot(x_axis_2,[Precision_pre[11], Precision_pre[23], Precision_pre[35]],'b')
axes[1,2].plot(x_axis_2,[Precision_post[11], Precision_post[23], Precision_post[35]],'g')


#ax.set_title('File classified as '+file_label+' cert ='+str(cert_label)+', num. clicks = '+str(len(LT_prob)))
        
#legend = ax.legend(loc='upper right')

plt.savefig(image_path)

#image_path="C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Results\\20201016_tests\\Test.png"
#fig, ax = plt.subplots()
#ax.plot(x_axis_0,rec,'r', label='Recall' )
#ax.plot(x_axis_0,pre1,'b', label='Precision_pre')
#ax.plot(x_axis_0,pre2,'g', label='Precision_post')
       
##ax.set_title('File classified as '+file_label+' cert ='+str(cert_label)+', num. clicks = '+str(len(LT_prob)))
        
#legend = ax.legend(loc='upper right')
        
#plt.savefig(image_path)
       
