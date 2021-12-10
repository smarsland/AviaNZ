"""
Script to evaluate total metrics

"""




def metrics(confusion_matrix, file_num):
    """
    Compute Recall, Precision, Accuracy pre and post possible classes check
    for each method
    
    INPUT:
        confusion_matrix is a matrix(3, 7) that stores the confusion matrix
                         
    OUTPUT:
        Recall -> number, recall 
                  TD/TD+FND this metric doesn't change before and after check
                  
        Precision_pre -> number, precision  before check
                         TD/TD+FD 
                         
        Precision_post -> number, precision  after check
                         TD/TD+FD 
                         
        Accuracy_pre -> number, accuracy before check
                         #correct classified/#files
                         for correct classifications we don't count possible 
                         classes
                         
       Accuracy_post -> number, accuracy after check
                         #correct classified/#files
                         for correct classifications we count possible classes
                                                  
    """
    
    #inizialization
    Recall=0
    Precision_pre=0
    Precision_post=0
    Accuracy_pre=0
    Accuracy_post=0
    #counting
    TD=np.sum(confusion_matrix[0][0:5])+np.sum(confusion_matrix[1][0:5])+np.sum(confusion_matrix[2][0:5])+ np.sum(confusion_matrix[3][0:5])+np.sum(confusion_matrix[4][0:5])+ np.sum(confusion_matrix[5][0:5])
    FND=np.sum(confusion_matrix[6][0:5])
    FPD_pre=confusion_matrix[0][5]+confusion_matrix[1][5]+confusion_matrix[2][5]+confusion_matrix[3][5]+confusion_matrix[4][5]+confusion_matrix[5][5]
    FPD_post=confusion_matrix[0][5]+confusion_matrix[2][5] + confusion_matrix[4][5]
    CoCla_pre= confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1]+confusion_matrix[2][2]+confusion_matrix[2][3] +confusion_matrix[3][3]+ confusion_matrix[3][2]+confusion_matrix[4][4]+confusion_matrix[5][4]+confusion_matrix[6][5]
    CoCla_post=CoCla_pre+np.sum(confusion_matrix[1][2:])+np.sum(confusion_matrix[3][0:2])+np.sum(confusion_matrix[3][4:])+np.sum(confusion_matrix[5][0:4])+confusion_matrix[5][5]
        
    #print
    #chck
    print('number of files =', file_num)
    print('TD =',TD)
    print('FND =',FND)
    print('FPD_pre =', FPD_pre)
    print('FPD_post =', FPD_post)
    print('Correct classifications pre =', CoCla_pre)
    print('Correct classifications post =', CoCla_post)
    #printng metrics
    print("-------------------------------------------")
    print("Click Detector stats on Testing Data")
    if TD==0:
        Recall=0
        Precision_pre= 0
        Precision_post=0
    else:
        Recall= TD/(TD+FND)*100
        Precision_pre=TD/(TD+FPD_pre)*100
        Precision_post=TD/(TD+FPD_post)*100
    print('Recall ', Recall)
    print('Precision_pre ', Precision_pre)
    print('Precision_post ', Precision_post)
    
    if CoCla_pre==0:
        Accuracy_pre=0
    else:
       Accuracy_pre=(CoCla_pre/file_num)*100
              
    if CoCla_post==0:
        Accuracy_post=0
    else:
       Accuracy_post=(CoCla_post/file_num)*100
   
    print('Accuracy_pre1', Accuracy_pre)
    print('Accuracy_post', Accuracy_post)
    

    return Recall, Precision_pre, Precision_post, Accuracy_pre, Accuracy_post, TD, FPD_pre, FPD_post, FND, CoCla_pre, CoCla_post


########## MAIN ################

test_count=0 #counter for test number
test_dir = "C:\\Users\\Virginia\\Documents\\Work\\Data\\Bats\\Results\\20201016_tests"
#test_dir = "/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/Battybats/Results/20201016_tests" #directory to store test result
test_fold= "Test_"+str(test_count) #Test folder where to save all the stats
confusion_matrix_tot=[]
file_number_tot=
# Evaluating metrics at the end of the process
print('Evualuating Bat Search performance on the entire Dataset')
Recall, Precision_pre, Precision_post, Accuracy_pre, Accuracy_post, TD, FPD_pre, FPD_post, FND, CoCla_pre, CoCla_post=metrics(confusion_matrix_tot, file_number_tot)

#print metrics
print("-------------------------------------------")
print('Classification performance on the entire dataset')
print('Number of analized files =', file_number_tot)
TD_rate= (TD/(file_number_tot-np.sum(confusion_matrix_tot[:][5])))*100
print('True Detected rate', TD_rate)
FPD_pre_rate= (FPD_pre/file_number_tot)*100
print('False Detected rate pre', FPD_pre_rate)
FPD_post_rate= (FPD_post/file_number_tot)*100
print('False Detected rate post', FPD_post_rate)
FND_rate= (FND/file_number_tot)*100
print('False Negative Detected rate', FND_rate)
print(confusion_matrix_tot)
print("-------------------------------------------")

with open(test_dir+'/' +test_fold+"/Confusion_matrix.txt",'w') as f:
    f.write("Confusion Matrix \n\n")
    np.savetxt(f, confusion_matrix_tot, fmt='%d')
    
#saving Click Detector Stats
cd_metrics_file=test_dir+'/'+test_fold+'/bat_detector_stats.txt'
file1=open(cd_metrics_file,"w")
L1=["Bat Detector stats on Testing Data \n"]
L2=['Number of files = %5d \n' %file_number_tot]
L3=['TD = %5d \n' %TD]
L4=['FPD_pre = %5d \n' %FPD_pre]
L5=['FPD_post= %5d \n' %FPD_post]
L6=['FND = %5d \n' %FND]
L7=['Correctly classified files before check = %5d \n' %CoCla_pre]
L8=['Correctly classified files after check= %5d \n' %CoCla_post]
L9=["Recall = %3.7f \n" %Recall,"Precision pre = %3.7f \n" %Precision_pre, "Precision post = %3.7f \n" %Precision_post, "Accuracy pre = %3.7f \n" %Accuracy_pre,  "Accuracy post = %3.7f \n" %Accuracy_post,  "True Detected rate = %3.7f \n" %TD_rate, "False Detected rate pre = %3.7f \n" %FPD_pre_rate, "False Detected rate post = %3.7f \n" %FPD_post_rate, "False Negative Detected rate = %3.7f \n" %FND_rate]
L10=['Model used ', modelpath, '\n']
#L11=['Training accuracy for the model %3.7f \n' %accuracies[index_best_model]]
file1.writelines(np.concatenate((L1,L2,L3,L4, L5, L6, L7, L8, L9, L10)))
#file1.writelines(np.concatenate((L1,L2,L3,L4, L5, L6, L7, L8, L9)))
file1.close()