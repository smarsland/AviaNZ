"""
Authors: Stephen Marsland, Virginia Listanti

This code recover information from files and then plot max, min, mean and mean for 
the best 3 in  4different plots: LT, ST, NT, BT with different classification 
conditions

It does this for all file classification with all possible spectrogram 
probabilities

ACTUAL CONDITIONS:
    NT  -> mean(0)<10 and mean(1)<10
    LT  -> mean(0)>=10 and P(0)>=65 and P(1)<65
    LT? -> mean(0)>=10 and P(0)<65
    ST  -> mean(1)>=10 and P(1)>=65 and P(0)<65
    ST? -> mean(1)>=10 and P(1)<65
    BT  -> mean(0)>=10 and mean(1)>=10 P(0)>=65 and P(1)>=65
    BT? -> mean(0)>=10 and mean(1)>=10 else
    

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

def detections_main(file_type, fun_list, filenames, thr1, thr2, LT_mean,
                    ST_mean, LT_best3mean, ST_best3mean, LT_max, ST_max,
                    confusion_matrix):
    """
    Manage the call for detections and plot for each file_type
    
    file_type= LT or ST or BT or NT
    fun_list= best3mean, max, mean
    filenames = list of all file with names
    
    thr1= threshold for mean
    thr2= threshold for fun
    
    RETURN (update):
        confusion_matrix  -> tensor
        
        
    Function -> index:
        best3mean -> 0
        max -> 1
        mean ->2
                      
    
    """
    #indeces where mean LT is >=10
    index_LT=np.nonzero(LT_mean*100>=thr1)
    #indeces where mean ST is >=10
    index_ST=np.nonzero(ST_mean*100>=thr1)
    #indeces where mean ST and LT are >=10
    supp=np.zeros((np.shape(ST_mean))) #aid vector
    supp[index_ST]=np.where(LT_mean[index_ST]*100>=thr1,1,0)
    index_LT_ST=np.nonzero(supp)
    #support index =>where mean LT >=10 but mean ST <10
    supp=np.zeros((np.shape(LT_mean))) #aid vector
    supp[index_LT]=np.where(ST_mean[index_LT]*100<thr1,1,0)
    index_sup_LT=np.nonzero(supp)
    #support index =>where mean ST >=10 but mean LT <10
    supp=np.zeros((np.shape(ST_mean))) #aid vector
    supp[index_ST]=np.where(LT_mean[index_ST]*100<thr1,1,0)
    index_sup_ST=np.nonzero(supp)
    #indeces where mean ST and LT are <10
    supp=np.zeros((np.shape(ST_mean))) #aid vector
    index_sup=np.nonzero(LT_mean*100<thr1)
    supp[index_sup]=np.where(ST_mean[index_sup]*100<thr1,1,0)
    index_not_LT_ST=np.nonzero(supp)
    
    print('checks '+ file_type +' FILES')
    print('index_LT', np.shape(index_LT)[1])
    print('index_ST', np.shape(index_ST)[1])
    print('index_LT_ST', np.shape(index_LT_ST)[1])
    print('index_sup_LT', np.shape(index_sup_LT)[1])
    print('index_sup_ST', np.shape(index_sup_ST)[1])
    print('index_not_LT_ST', np.shape(index_not_LT_ST)[1])
    print('number of file',len(LT_mean))
    
    for fun in fun_list:
        if fun=='best3mean':
            #evaluate indeces
            [index_LT_fun, index_BT_fun, index_LT_not_fun, index_not_LT_ST_fun,
             index_BT_not_fun, index_ST_fun, index_ST_not_fun, 
             index_not_LT_ST_not_fun] = indices(thr2, LT_best3mean, 
                                         index_LT,  index_ST, index_LT_ST,
                                         index_sup_LT, index_sup_ST, index_not_LT_ST)
            
            # detected function
            [NT_detected_best3mean, LT_detected_best3mean, LT_pos_detected_best3mean,
            ST_detected_best3mean, ST_pos_detected_best3mean, BT_detected_best3mean, 
            BT_pos_detected_best3mean] = detect_classes(thr2, ST_best3mean, index_LT_fun, 
            index_BT_fun, index_LT_not_fun, index_not_LT_ST_fun, index_BT_not_fun, 
            index_ST_fun, index_ST_not_fun, index_not_LT_ST_not_fun)
            
            plot_detections(filenames, file_type, fun, LT_best3mean, ST_best3mean, LT_mean, ST_mean, NT_detected_best3mean, LT_detected_best3mean, LT_pos_detected_best3mean,
            ST_detected_best3mean, ST_pos_detected_best3mean, BT_detected_best3mean, 
            BT_pos_detected_best3mean)
            
            #update confusion_matrix
            confusion_matrix[0]=update_confusion_matrix(confusion_matrix[0], file_type, NT_detected_best3mean, 
                                    LT_detected_best3mean, LT_pos_detected_best3mean, ST_detected_best3mean, 
                                    ST_pos_detected_best3mean, BT_detected_best3mean, BT_pos_detected_best3mean)
            
        elif fun=='max':
            #evaluate indeces
            [index_LT_fun, index_BT_fun, index_LT_not_fun,
            index_not_LT_ST_fun, index_BT_not_fun, index_ST_fun,
            index_ST_not_fun, index_not_LT_ST_not_fun ]= indices(thr2, LT_max, 
                                                         index_LT,  index_ST, index_LT_ST,
                                                         index_sup_LT, index_sup_ST, index_not_LT_ST)
            
            # detected function
            [NT_detected_max, LT_detected_max, LT_pos_detected_max,
            ST_detected_max, ST_pos_detected_max, BT_detected_max, 
            BT_pos_detected_max] = detect_classes(thr2, ST_max, index_LT_fun, 
            index_BT_fun, index_LT_not_fun, index_not_LT_ST_fun, index_BT_not_fun, 
            index_ST_fun, index_ST_not_fun, index_not_LT_ST_not_fun)
            
            plot_detections(filenames, file_type, fun, LT_max, ST_max, LT_mean, ST_mean, 
                            NT_detected_max, LT_detected_max, LT_pos_detected_max,
                            ST_detected_max, ST_pos_detected_max, BT_detected_max, 
                            BT_pos_detected_max)
            
            confusion_matrix[1]=update_confusion_matrix(confusion_matrix[1], file_type, NT_detected_max, 
                                    LT_detected_max, LT_pos_detected_max, ST_detected_max, 
                                    ST_pos_detected_max, BT_detected_max, BT_pos_detected_max)
            
        elif fun=='mean':
            #evaluate indeces
            [index_LT_fun, index_BT_fun, index_LT_not_fun, index_not_LT_ST_fun,
             index_BT_not_fun, index_ST_fun, index_ST_not_fun, 
             index_not_LT_ST_not_fun] = indices(thr2, LT_mean, 
                                         index_LT,  index_ST, index_LT_ST,
                                         index_sup_LT, index_sup_ST, index_not_LT_ST)
            
            # detected function
            [NT_detected_mean, LT_detected_mean, LT_pos_detected_mean,
            ST_detected_mean, ST_pos_detected_mean, BT_detected_mean, 
            BT_pos_detected_mean] = detect_classes(thr2, ST_mean, index_LT_fun, 
            index_BT_fun, index_LT_not_fun, index_not_LT_ST_fun, index_BT_not_fun, 
            index_ST_fun, index_ST_not_fun, index_not_LT_ST_not_fun)
            
            plot_detections(filenames, file_type, fun, LT_best3mean, ST_best3mean, LT_mean, ST_mean, NT_detected_best3mean, LT_detected_best3mean, LT_pos_detected_best3mean,
            ST_detected_best3mean, ST_pos_detected_best3mean, BT_detected_best3mean, 
            BT_pos_detected_best3mean)
            
            confusion_matrix[2]=update_confusion_matrix(confusion_matrix[2], file_type, NT_detected_mean, 
                                    LT_detected_mean, LT_pos_detected_mean, ST_detected_mean, 
                                    ST_pos_detected_mean, BT_detected_mean, BT_pos_detected_mean)
            
    test_dir="D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Test10\\"        
    if file_type=='NT':
        
        #save false positives
        filenames_a=np.reshape(filenames,(len(filenames),1))
        #ST
        FP_ST_best3mean=filenames_a[np.nonzero(ST_detected_best3mean)]
        FP_ST_max=filenames_a[np.nonzero(ST_detected_max)]
        FP_ST_mean=filenames_a[np.nonzero(ST_detected_mean)]
        #LT
        FP_LT_best3mean=filenames_a[np.nonzero(LT_detected_best3mean)]
        FP_LT_max=filenames_a[np.nonzero(LT_detected_max)]
        FP_LT_mean=filenames_a[np.nonzero(LT_detected_mean)]
        #BT
        FP_BT_best3mean=filenames_a[np.nonzero(BT_detected_best3mean)]
        FP_BT_max=filenames_a[np.nonzero(BT_detected_max)]
        FP_BT_mean=filenames_a[np.nonzero(BT_detected_mean)]
        
        #save on a file
        f= open(test_dir+'false_positives.txt', 'w')
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
        json.dump(FP_BT_max.tolist(),f)
        f.write('\n\n  Mean \n')
        f.write(' LT \n')
        json.dump(FP_LT_mean.tolist(),f)
        f.write('\n ST \n')
        json.dump(FP_ST_mean.tolist(),f)
        f.write('\n BT \n')
        json.dump(FP_BT_mean.tolist(),f)
        f.close()
    else:
        ## find missed files with different methods
    # a missed file is a file that  it will be classified as noise
        filenames_a=np.reshape(filenames,(len(filenames),1))
        missed_best3mean=filenames_a[np.nonzero(NT_detected_best3mean)]
        missed_max=filenames_a[np.nonzero(NT_detected_max)]
        missed_mean=filenames_a[np.nonzero(NT_detected_mean)]
        #save txt file
        f= open(test_dir+file_type+'_missed.txt', 'w')
        f.write('BEST 3 MEAN \n')
        json.dump(missed_best3mean.tolist(),f)
        f.write('\n MAX \n')
        json.dump(missed_max.tolist(),f)
        f.write('\n Mean \n')
        json.dump(missed_mean.tolist(),f)
        f.close()
    
    return confusion_matrix

def indices(thr2, LT_fun, index_LT,  index_ST, index_LT_ST, index_sup_LT, index_sup_ST, index_not_LT_ST):
    """
    Evaluate indexes based on fun
    """
#    index_LT_fun = mean(0)>=thr1 mean(1)<thr1 F(0)>=thr2
    supp=np.zeros((np.shape(LT_fun))) #aid vector
    supp[index_sup_LT]=np.where(LT_fun[index_sup_LT]*100>=thr2,1,0)
    index_LT_fun=np.nonzero(supp)
    
    #    index_ST_fun = mean(0)<thr1 mean(1)>=thr1 F(0)>=thr2
    supp=np.zeros((np.shape(LT_fun))) #aid vector
    supp[index_sup_ST]=np.where(LT_fun[index_sup_ST]*100>=thr2,1,0)
    index_ST_fun=np.nonzero(supp)
    
     #    index_BT_fun = mean(0)<thr1 mean(1)>=thr1 F(0)>=thr2
    supp=np.zeros((np.shape(LT_fun))) #aid vector
    supp[index_LT_ST]=np.where(LT_fun[index_LT_ST]*100>=thr2,1,0)
    index_BT_fun=np.nonzero(supp)
    
    #    index_not_LT_ST_fun = mean(0)<thr1 mean(1)<thr1 F(0)>=thr2
    supp=np.zeros((np.shape(LT_fun))) #aid vector
    supp[index_not_LT_ST]=np.where(LT_fun[index_not_LT_ST]*100>=thr2,1,0)
    index_not_LT_ST_fun=np.nonzero(supp)
    
    #    index_LT_not_fun = mean(0)>=thr1 mean(1)<thr1 F(0)<thr2
    supp=np.zeros((np.shape(LT_fun))) #aid vector
    supp[index_sup_LT]=np.where(LT_fun[index_sup_LT]*100<thr2,1,0)
    index_LT_not_fun=np.nonzero(supp)
    
    #    index_ST_not_fun = mean(0)<thr1 mean(1)>=thr1 F(0)>=thr2
    supp=np.zeros((np.shape(LT_fun))) #aid vector
    supp[index_sup_ST]=np.where(LT_fun[index_sup_ST]*100<thr2,1,0)
    index_ST_not_fun=np.nonzero(supp)
    
     #    index_BT_not_fun = mean(0)<thr1 mean(1)>=thr1 F(0)>=thr2
    supp=np.zeros((np.shape(LT_fun))) #aid vector
    supp[index_LT_ST]=np.where(LT_fun[index_LT_ST]*100<thr2,1,0)
    index_BT_not_fun=np.nonzero(supp)
    
    #    index_not_LT_ST_not_fun = mean(0)<thr1 mean(1)<thr1 F(0)>=thr2
    supp=np.zeros((np.shape(LT_fun))) #aid vector
    supp[index_not_LT_ST]=np.where(LT_fun[index_not_LT_ST]*100<thr2,1,0)
    index_not_LT_ST_not_fun=np.nonzero(supp)
    
#    print('Check indices:')
#    print('index_LT_fun', np.shape(index_LT_fun)[1])
#    print('index_LT_not_fun', np.shape(index_LT_not_fun)[1])
#    print('index_ST_fun', np.shape(index_ST_fun)[1])
#    print('index_ST_not_fun', np.shape(index_ST_not_fun)[1])
#    print('index_BT_fun', np.shape(index_BT_fun)[1])
#    print('index_BT_not_fun', np.shape(index_BT_not_fun)[1])
#    print('index_not_LT_ST_fun', np.shape(index_not_LT_ST_fun)[1])
#    print('index_not_LT_ST_not_fun', np.shape(index_not_LT_ST_not_fun)[1])
#    print('number of file',len(LT_fun))
    
      
    return index_LT_fun, index_BT_fun, index_LT_not_fun, index_not_LT_ST_fun, index_BT_not_fun, index_ST_fun, index_ST_not_fun, index_not_LT_ST_not_fun



def detect_classes(thr2, ST_fun, index_LT_fun, index_BT_fun, index_LT_not_fun, index_not_LT_ST_fun, index_BT_not_fun, index_ST_fun, index_ST_not_fun, index_not_LT_ST_not_fun):
    """
    Find the detection function for each class
    """
    
    #NT
    NT_detected_fun=np.zeros((np.shape(ST_fun)))
    NT_detected_fun[index_not_LT_ST_not_fun]=np.where(ST_fun[index_not_LT_ST_not_fun]*100<thr2,1,0)
    
    #LT
    LT_detected_fun=np.zeros((np.shape(ST_fun)))
    LT_detected_fun[index_LT_fun]=np.where(ST_fun[index_LT_fun]*100<thr2,1,0)
    
    #LT?
    LT_pos_detected_fun=np.zeros((np.shape(ST_fun)))
    LT_pos_detected_fun[index_LT_not_fun]=np.ones((np.shape(LT_pos_detected_fun[index_LT_not_fun])))
#    LT_pos_detected_fun[index_LT_not_fun]=1
    supp=np.zeros((np.shape(ST_fun)))
    supp[index_LT_fun]=np.where(ST_fun[index_LT_fun]*100>=thr2,1,0)
    LT_pos_detected_fun+=supp
    supp=np.zeros((np.shape(ST_fun)))
    supp[index_not_LT_ST_fun]=np.where(ST_fun[index_not_LT_ST_fun]*100<thr2,1,0)
    LT_pos_detected_fun+=supp
    
    #ST
    ST_detected_fun=np.zeros((np.shape(ST_fun)))
    ST_detected_fun[index_ST_not_fun]=np.where(ST_fun[index_ST_not_fun]*100>=thr2,1,0)
    
    #ST?
    ST_pos_detected_fun=np.zeros((np.shape(ST_fun)))
    ST_pos_detected_fun[index_ST_fun]=np.ones((np.shape(ST_pos_detected_fun[index_ST_fun])))
#    ST_pos_detected_fun[index_ST_fun]=1
    supp=np.zeros((np.shape(ST_fun)))
    supp[index_ST_not_fun]=np.where(ST_fun[index_ST_not_fun]*100<thr2,1,0)
    ST_pos_detected_fun+=supp
    supp=np.zeros((np.shape(ST_fun)))
    supp[index_not_LT_ST_not_fun]=np.where(ST_fun[index_not_LT_ST_not_fun]*100>=thr2,1,0)
    ST_pos_detected_fun+=supp
    
    #BT
    BT_detected_fun=np.zeros((np.shape(ST_fun)))
    BT_detected_fun[index_BT_fun]=np.where(ST_fun[index_BT_fun]*100>=thr2,1,0)
    
    #LT?
    BT_pos_detected_fun=np.zeros((np.shape(ST_fun)))
    print('index_BT_not_fun',index_BT_not_fun)
    BT_pos_detected_fun[index_BT_not_fun]=np.ones((np.shape(BT_pos_detected_fun[index_BT_not_fun])))
#    BT_pos_detected_fun[index_BT_not_fun]=1
    supp=np.zeros((np.shape(ST_fun)))
    supp[index_BT_fun]=np.where(ST_fun[index_BT_fun]*100<thr2,1,0)
    BT_pos_detected_fun+=supp
    supp=np.zeros((np.shape(ST_fun)))
    supp[index_not_LT_ST_fun]=np.where(ST_fun[index_not_LT_ST_fun]*100>=thr2,1,0)
    BT_pos_detected_fun+=supp
       
    return NT_detected_fun, LT_detected_fun, LT_pos_detected_fun, ST_detected_fun, ST_pos_detected_fun, BT_detected_fun, BT_pos_detected_fun


def plot_detections(filenames, file_type, fun, LT_fun, ST_fun, LT_mean, ST_mean, NT_detected_fun, LT_detected_fun, LT_pos_detected_fun, ST_detected_fun, ST_pos_detected_fun, BT_detected_fun, BT_pos_detected_fun):
    """
    Manage plot of detections functions 
    fyle_type -> str with LT, ST, BT or NT
    fun -> str with funtion name
    
    """
    
       #check that we got everything
    print('check that we got everything')
    print('check shape', len(filenames)  )
    print('File type: '+file_type)
    print('Function: ' +fun)
    print('Number of files', len(filenames))
    print('number of LT detected', np.shape(np.nonzero(LT_detected_fun))[1] )
    print('number of LT? detected', np.shape(np.nonzero(LT_pos_detected_fun))[1] )
    print('number of ST detected', np.shape(np.nonzero(ST_detected_fun))[1] )
    print('number of ST? detected', np.shape(np.nonzero(ST_pos_detected_fun))[1] )
    print('number of BT detected', np.shape(np.nonzero(BT_detected_fun))[1] )
    print('number of BT? detected', np.shape(np.nonzero(BT_pos_detected_fun))[1] )
    print('number of NT detected', np.shape(np.nonzero(NT_detected_fun))[1] )
    sum_detections= np.shape(np.nonzero(LT_detected_fun))[1]+np.shape(np.nonzero(LT_pos_detected_fun))[1]+ np.shape(np.nonzero(ST_detected_fun))[1]+ np.shape(np.nonzero(ST_pos_detected_fun))[1]+ np.shape(np.nonzero(BT_detected_fun))[1]+np.shape(np.nonzero(BT_pos_detected_fun))[1]+ np.shape(np.nonzero(NT_detected_fun))[1]
    print('sum', sum_detections)
    
    
    fig, axes=plt.subplots(4,2,sharex='all', sharey='col')
    fig.suptitle('Detections in '+file_type+' files with '+ fun)
    #different plot if BT files
    if file_type=='BT':
        #LT first row
        axes[0][0].plot(filenames, LT_fun, 'bo', LT_detected_fun, 'rx')
        axes[0][0].set_ylabel('LT',  rotation=0, size='large')
        axes[0][1].plot(filenames, LT_fun, 'bo', LT_pos_detected_fun, 'rx')
        axes[0][1].set_title('? categories')
        #ST second row
        axes[1][0].plot(filenames, ST_fun, 'gs', ST_detected_fun, 'rx')
        axes[1][0].set_ylabel('ST',  rotation=0, size='large')
        axes[1][1].plot(filenames, ST_fun, 'gs', ST_pos_detected_fun, 'rx')
        
        #BT tird row
        axes[2][0].plot(filenames, LT_fun, 'bo', ST_fun, 'gs', BT_detected_fun, 'rx')
        axes[2][0].set_ylabel('BT',  rotation=0, size='large')
        axes[2][1].plot(filenames, LT_fun, 'bo', ST_fun, 'gs', BT_pos_detected_fun, 'rx')
        
        #NT last row
        axes[3][0].plot(filenames, LT_mean, 'bo', ST_mean, 'gs', NT_detected_fun, 'rx')
        axes[3][0].axes.xaxis.set_ticklabels([])
        
    elif file_type=='NT':
        #LT first row
        axes[0][0].plot(filenames, LT_mean, 'b', LT_detected_fun, 'r')
        axes[0][0].set_ylabel('LT',  rotation=0, size='large')
        axes[0][1].plot(filenames, LT_mean, 'b', LT_pos_detected_fun, 'r')
        axes[0][1].set_title('? categories')
        #ST second row
        axes[1][0].plot(filenames, ST_mean, 'g', ST_detected_fun, 'r')
        axes[1][0].set_ylabel('ST',  rotation=0, size='large')
        axes[1][1].plot(filenames, ST_mean, 'g', ST_pos_detected_fun, 'r')
        
        #BT tird row
        axes[2][0].plot(filenames, LT_mean, 'b', ST_fun, 'g', BT_detected_fun, 'r')
        axes[2][0].set_ylabel('BT',  rotation=0, size='large')
        axes[2][1].plot(filenames, LT_mean, 'b', ST_fun, 'g', BT_pos_detected_fun, 'r')
        
        #NT last row
        axes[3][0].plot(filenames, LT_mean, 'b', ST_mean, 'g', NT_detected_fun, 'r')
        axes[3][0].axes.xaxis.set_ticklabels([]) 
        
    else: 
        #LT first row
        axes[0][0].plot(filenames, LT_fun, 'b', LT_detected_fun, 'r')
        axes[0][0].set_ylabel('LT',  rotation=0, size='large')
        axes[0][1].plot(filenames, LT_fun, 'b', LT_pos_detected_fun, 'r')
        axes[0][1].set_title('? categories')
        #ST second row
        axes[1][0].plot(filenames, ST_fun, 'g', ST_detected_fun, 'r')
        axes[1][0].set_ylabel('ST',  rotation=0, size='large')
        axes[1][1].plot(filenames, ST_fun, 'g', ST_pos_detected_fun, 'r')
        
        #BT tird row
        axes[2][0].plot(filenames, LT_fun, 'b', ST_fun, 'g', BT_detected_fun, 'r')
        axes[2][0].set_ylabel('BT',  rotation=0, size='large')
        axes[2][1].plot(filenames, LT_fun, 'b', ST_fun, 'g', BT_pos_detected_fun, 'r')
        
        #NT last row
        axes[3][0].plot(filenames, LT_mean, 'b', ST_mean, 'g', NT_detected_fun, 'r')
        axes[3][0].axes.xaxis.set_ticklabels([]) 

    #hide labels for inner plots
    for ax in axes.flat:
        ax.label_outer()
    #plt.show()
    test_dir="D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Test10\\"
    plt.savefig(test_dir+file_type+"detections_"+fun+".png")

    return  

def update_confusion_matrix(confusion_matrix, file_type, NT_detected, LT_detected, 
                            LT_pos_detected, ST_detected, ST_pos_detected,
                            BT_detected, BT_pos_detected):
    """
    Update confusion_matrix 
    
    file_type -> matrix column
        LT -> 0
        ST -> 1
        BT -> 2
        NT -> 3

    """
    
    #recover column index fol file type
    if file_type=='LT':
        column_index=0
    elif file_type=='ST':
        column_index=1
    elif file_type=='BT':
        column_index=2
    elif file_type=='NT':
        column_index=3
    
    confusion_matrix[0][column_index]=np.shape(np.nonzero(LT_detected))[1]
    confusion_matrix[1][column_index]=np.shape(np.nonzero(LT_pos_detected))[1]
    confusion_matrix[2][column_index]=np.shape(np.nonzero(ST_detected))[1]
    confusion_matrix[3][column_index]=np.shape(np.nonzero(ST_pos_detected))[1]
    confusion_matrix[4][column_index]=np.shape(np.nonzero(BT_detected))[1]
    confusion_matrix[5][column_index]=np.shape(np.nonzero(BT_pos_detected))[1]
    confusion_matrix[6][column_index]=np.shape(np.nonzero(NT_detected))[1]
    
    return confusion_matrix

def metrics(confusion_matrix, file_num):
    """
    Compute Recall, Precision, Accuracy pre and post possible classes check
    for each method
    
    INPUT:
        confusion_matrix is a tensor (3, 7, 4) that stores the confusion matrix
                         for each method
                         
    OUTPUT:
        Recall -> vector, recall for each method
                  TD/TD+FND this metric doesn't change before and after check
                  
        Precision_pre -> vector, precision for each method before check
                         TD/TD+FD 
                         
        Precision_post -> vector, precision for each method after check
                         TD/TD+FD 
                         
        Accuracy_pre -> vector, accuracy for each method before check
                         #correct classified/#files
                         for correct classifications we don't count possible 
                         classes
                         
       Accuracy_post -> vector, accuracy for each method after check
                         #correct classified/#files
                         for correct classifications we count possible classes
                                                  
    """
    
    #inizialization
    Recall=np.zeros((3,))
    Precision_pre=np.zeros((3,))
    Precision_post=np.zeros((3,))
    Accuracy_pre=np.zeros((3,))
    Accuracy_post=np.zeros((3,))
    
    for i in range(3):
        #counting
        TD=np.sum(confusion_matrix[i][0][0:3])+np.sum(confusion_matrix[i][1][0:3])+np.sum(confusion_matrix[i][2][0:3])+ np.sum(confusion_matrix[i][3][0:3])+np.sum(confusion_matrix[i][4][0:3])+ np.sum(confusion_matrix[i][5][0:3])
        print('TD', TD)
        FND=np.sum(confusion_matrix[i][6][0:3])
        print('FND', FND)
#        TND_pre=confusion_matrix[i][7][3]
#        TND_post=confusion_matrix[i][1][3]+confusion_matrix[i][3][3] + confusion_matrix[i][5][3]+confusion_matrix[i][6][3]
        FPD_pre=confusion_matrix[i][0][3]+confusion_matrix[i][1][3]+confusion_matrix[i][2][3]+confusion_matrix[i][3][3]+confusion_matrix[i][4][3]+confusion_matrix[i][5][3]
        print('FPD_pre', FPD_pre)
        FPD_post=confusion_matrix[i][0][3]+confusion_matrix[i][2][3] + confusion_matrix[i][4][3]
        print('FPD_post', FPD_post)
        CoCla_pre= confusion_matrix[i][0][0]+confusion_matrix[i][2][1] + confusion_matrix[i][4][2]+confusion_matrix[i][6][3]
        print('CoCla_pre', CoCla_pre)
        CoCla_post=CoCla_pre+np.sum(confusion_matrix[i][1][:])+np.sum(confusion_matrix[i][3][:])+np.sum(confusion_matrix[i][5][:])
        print('CoCla_post', CoCla_post)
        
        #metrics
        Recall[i]=TD/(TD+FND)
        Precision_pre[i]=TD/(TD+FPD_pre)
        Precision_post[i]=TD/(TD+FPD_post)
        Accuracy_pre[i]=CoCla_pre/file_num
        Accuracy_post[i]=CoCla_post/file_num
    return Recall, Precision_pre, Precision_post, Accuracy_pre, Accuracy_post

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
#    ST_ST_best3mean[i]/= len(ind)
    ST_ST_best3mean[i]/= 3
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
#    ST_LT_best3mean[i]/= len(ind)
    ST_LT_best3mean[i]/=3
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
#    ST_NT_best3mean[i]/= len(ind)
    ST_NT_best3mean[i]/=3
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
#    LT_LT_best3mean[i]/= len(ind)
    LT_LT_best3mean[i]/=3
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
#    LT_ST_best3mean[i]/= len(ind)
    LT_ST_best3mean[i]/=3
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
#    LT_NT_best3mean[i]/= len(ind)
    LT_NT_best3mean[i]/=3
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
#    NT_NT_best3mean[i]/= len(ind)
    NT_NT_best3mean[i]/=3
    NT_NT_mean[i]=np.mean(NT_NT[i][1])    

#print('check NT_NT_best3mean')
#print('max', max(NT_NT_best3mean))   
  
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
#    NT_LT_best3mean[i]/= len(ind)
    NT_LT_best3mean[i]/=3
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
#    NT_ST_best3mean[i]/= len(ind)
    NT_ST_best3mean[i]/= 3
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
#    BT_LT_best3mean[i]/= len(ind)
    BT_LT_best3mean[i]/=3
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
#    BT_NT_best3mean[i]/= len(ind)
    BT_NT_best3mean[i]/=3
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
#    BT_ST_best3mean[i]/= len(ind)
    BT_ST_best3mean[i]/=3
    BT_ST_mean[i]=np.mean(BT_ST[i][1])      

#Parameters
fun_list=['best3mean', 'max','mean']
thr1=10
thr2=70

file_num=len(LT_filenames)+len(ST_filenames)+len(BT_filenames)+len(NT_filenames)
#print('We are testing on '+str(file_num) +' files')

#inizializing confusion_matrix
confusion_matrix=np.zeros((3,7,4))

##LT files detections and plot
confusion_matrix=detections_main('LT', fun_list, LT_filenames, thr1, thr2, LT_LT_mean, LT_ST_mean, LT_LT_best3mean, LT_ST_best3mean, LT_LT_max, LT_ST_max, confusion_matrix)

###ST files detections and plot
confusion_matrix=detections_main('ST', fun_list, ST_filenames, thr1, thr2, ST_LT_mean, ST_ST_mean, ST_LT_best3mean, ST_ST_best3mean, ST_LT_max, ST_ST_max, confusion_matrix)

###NT files detections and plot
confusion_matrix=detections_main('NT', fun_list, NT_filenames, thr1, thr2, NT_LT_mean, NT_ST_mean, NT_LT_best3mean, NT_ST_best3mean, NT_LT_max, NT_ST_max, confusion_matrix)

###BT files plot
confusion_matrix=detections_main('BT', fun_list, BT_filenames, thr1, thr2, BT_LT_mean, BT_ST_mean, BT_LT_best3mean, BT_ST_best3mean, BT_LT_max, BT_ST_max, confusion_matrix)

print('Check confusion matrix')
print('We are testing on '+str(file_num) +' files')
print('Best 3 Mean', np.sum(np.sum(confusion_matrix[0],0)))
print('MAx', np.sum(confusion_matrix[1]))
print('Mean', np.sum(confusion_matrix[2]))


#Save confusion matrix for each method
test_dir="D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Test10\\"
with open(test_dir+"Confusion_matrix.txt",'w') as f:
    f.write("Best 3 mean \n\n")
    np.savetxt(f, confusion_matrix[0], fmt='%d')
    f.write("\n\n Max \n\n")
    np.savetxt(f, confusion_matrix[1], fmt='%d')
    f.write("\n\n Mean \n\n")
    np.savetxt(f, confusion_matrix[2], fmt='%d')

#Evaluate metrics
Recall, Precision_pre, Precision_post, Accuracy_pre, Accuracy_post = metrics(confusion_matrix, file_num)

#Save metrics on files
with open(test_dir+"metrics.txt",'w') as f:
    for i in range(3):
        if i==0:
            f.write('Best 3 mean \n')
        elif i==1:
            f.write('\n\n Max \n')
        elif i==2:
            f.write('\n\n Mean \n')
        f.write('Recall = %2.2f \n' %(Recall[i]*100))
        f.write('\nPre-check on possible classes: \n')
        f.write('Precision = %2.2f \n' %(Precision_pre[i]*100))
        f.write('Accuracy = %2.2f \n' %(Accuracy_pre[i]*100))
        f.write('\n Post-check on possible classes: \n')
        f.write('Precision = %2.2f \n' %(Precision_post[i]*100))
        f.write('Accuracy = %2.2f \n' %(Accuracy_post[i]*100))
        