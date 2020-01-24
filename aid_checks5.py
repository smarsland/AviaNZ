# -*- coding: utf-8 -*-
"""
Author: Virginia Listanti

Aid Script for developing functions for checks5.py

"""

import numpy as np

def detections_main(file_type, fun_list, filenames, thr1, thr2, LT_mean, ST_mean, LT_best3mean, ST_best3mean, LT_max, ST_max):
    """
    Manage the call for detections and plot for each file_type
    
    file_type= LT or ST or BT or NT
    fun_list= best3mean, max, mean
    filenames = list of all file with names
    
    thr1= threshold for mean
    thr2= threshold for fun
    
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
    index_sup=np.nonzero(LT_mean*100<10)
    supp[index_sup]=np.where(ST_mean[index_sup]*100<thr1,1,0)
    index_not_LT_ST=np.nonzero(supp)
    
    print('checks'+ file_type +'FILES')
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
            
        elif fun=='max':
            #evaluate indeces
            index_LT_fun, index_BT_fun, index_LT_not_fun,
            index_not_LT_ST_fun, index_BT_not_fun, index_ST_fun,
            index_ST_not_fun, index_not_LT_ST_not_fun = indices(thr2, LT_max, 
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
            
    
    
    
    return

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
    [index_not_LT_ST]=np.where(LT_fun[index_not_LT_ST]*100>=thr2,1,0)
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
    
      
    return index_LT_fun, index_BT_fun, index_LT_not_fun, index_not_LT_ST_fun, index_BT_not_fun, index_ST_fun, index_ST_not_fun, index_not_LT_ST_not_fun



def detect_classes(thr2, ST_fun, index_LT_fun, index_BT_fun, index_LT_not_fun, index_not_LT_ST_fun, index_BT_not_fun, index_ST_fun, index_ST_not_fun, index_not_LT_ST_not_fun):
    """
    Find the detection function for each class
    """
    
    #NT
    NT_detected_fun=np.zeros((np.shape(ST_fun)))
    NT_detected_fun[index_not_LT_ST_fun]=np.where(ST_fun[index_not_LT_ST_fun]*100<thr2,1,0)
    
    #LT
    LT_detected_fun=np.zeros((np.shape(ST_fun)))
    LT_detected_fun[index_LT_fun]=np.where(ST_fun[index_LT_fun]*100<thr2,1,0)
    
    #LT?
    LT_pos_detected_fun=np.zeros((np.shape(ST_fun)))
    LT_pos_detected_fun[index_LT_not_fun]=1
    supp=np.zeros((np.shape(ST_fun)))
    supp[index_LT_fun]=np.where(ST_fun[index_LT_fun]>=thr2,1,0)
    LT_pos_detected_fun+=supp
    supp=np.zeros((np.shape(ST_fun)))
    supp[index_not_LT_ST_fun]=np.where(ST_fun[index_not_LT_ST_fun]<thr2,1,0)
    LT_pos_detected_fun+=supp
    
    #ST
    ST_detected_fun=np.zeros((np.shape(ST_fun)))
    ST_detected_fun[index_ST_not_fun]=np.where(ST_fun[index_ST_not_fun]*100<thr2,1,0)
    
    #ST?
    ST_pos_detected_fun=np.zeros((np.shape(ST_fun)))
    ST_pos_detected_fun[index_ST_fun]=1
    supp=np.zeros((np.shape(ST_fun)))
    supp[index_ST_not_fun]=np.where(ST_fun[index_ST_not_fun]<thr2,1,0)
    ST_pos_detected_fun+=supp
    supp=np.zeros((np.shape(ST_fun)))
    supp[index_not_LT_ST_not_fun]=np.where(ST_fun[index_not_LT_ST_not_fun]>=thr2,1,0)
    ST_pos_detected_fun+=supp
    
    #BT
    BT_detected_fun=np.zeros((np.shape(ST_fun)))
    BT_detected_fun[index_BT_fun]=np.where(ST_fun[index_BT_fun]*100>=thr2,1,0)
    
    #LT?
    BT_pos_detected_fun=np.zeros((np.shape(ST_fun)))
    BT_pos_detected_fun[index_BT_not_fun]=1
    supp=np.zeros((np.shape(ST_fun)))
    supp[index_BT_fun]=np.where(ST_fun[index_BT_fun]<thr2,1,0)
    BT_pos_detected_fun+=supp
    supp=np.zeros((np.shape(ST_fun)))
    supp[index_not_LT_ST_fun]=np.where(ST_fun[index_not_LT_ST_fun]>=thr2,1,0)
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
    print('File type: '+file_type)
    print('Function. ' +fun)
    print('Number of files', np.shape(LT_detected_fun)[1])
    print('number of LT detected', np.shape(np.nonzero(LT_detected_fun))[1] )
    print('number of LT? detected', np.shape(np.nonzero(LT_pos_detected_fun))[1] )
    print('number of ST detected', np.shape(np.nonzero(ST_detected_fun))[1] )
    print('number of ST? detected', np.shape(np.nonzero(ST_pos_detected_fun))[1] )
    print('number of BT detected', np.shape(np.nonzero(BT_detected_fun))[1] )
    print('number of BT? detected', np.shape(np.nonzero(BT_pos_detected_fun))[1] )
    print('sum', np.shape(np.nonzero(LT_detected_fun))[1]+np.shape(np.nonzero(LT_pos_detected_fun))[1]+
        np.shape(np.nonzero(ST_detected_fun))[1]+np.shape(np.nonzero(BT_pos_detected_fun))[1]+
       np.shape(np.nonzero(NT_detected_fun))[1])
    
    fig, axes=plt.subplots(4,2,sharex='all', sharey='col')
    fig.suptitle('Detections in'+file_type+' files with '+ fun)

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
    plt.savefig("D:\\Desktop\\Documents\\Work\\Data\\Bat\\BAT\\CNN experiment\\TEST2\\BAT SEARCH TESTS\\Test_79\\study3\\Test03\\"+file_type+"detections_"+fun+".png")

    return    