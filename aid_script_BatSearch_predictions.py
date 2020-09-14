# -*- coding: utf-8 -*-
"""
Aid file for reading predictions functions

@Virginia Listanti
"""

#First Option: 3 lables, arg max * majority vote
def File_label(predictions, spec_id, segments_filewise_test, filewise_output, file_number ):
    """
    FIle_label use the predictions made by the CNN to update the filewise annotations
    when we have 3 labels: 0 (LT), 1(ST), 2 (Noise)
    
    METHOD: ARGMAX probabilities labels + majority vote
        Majority Vote:
            if No LT or ST => Noise
            if  LT>70% of good spectrogram in file => LT
            if  ST>70% of good spectrogram in file => ST
            otherwise => Both
        
    """
    
    predicted_label= np.argmax(predictions,axis=1)
    if len(predicted_label)!=np.shape(spec_id)[0]:
        print('ERROR: Number of labels is not equal to number of spectrograms' )
    
    # Assesting file label and updating metrics  
    for i in range(file_number):
        file = segments_filewise_test[i][0]
        #inizializing counters
        LT_count=0
        ST_count=0
        Other_count=0
        spec_num=0   #counts number of spectrograms per file
        #flag: if no click detected no spectrograms
        click_detected_flag=False
#        looking for all the spectrogram related to this file
        #count majority
        for k in range(np.shape(spec_id)[0]):
            if spec_id[k][0]==file:
                click_detected_flag= True
                if predicted_label[k]==0:
                    LT_count+=1
                elif predicted_label[k]==1:
                    ST_count+=1
                else:
                    Other_count+=1
                spec_num+=1
                
        #assign label to file
        if click_detected_flag==True:
            #this makes sense only if there were spectrograms
#            keeping differen majority vote options
    #        if Other_count>LT_count+ST_count:
    #        if (Other_count/spec_num)*100>90:
            if LT_count+ST_count==0:
                label='Noise'
            else:
                LT_perc=(LT_count/(spec_num-Other_count))*100 #percentage of LT over "good clicks" clicks
                ST_perc=(ST_count/(spec_num-Other_count))*100 #percentage of LT over "good clicks" clicks
                if LT_perc>70:
                    label='LT'
                elif ST_perc>70:
                    label='ST'
                else:
                    label='Both'
        else:
            #if no click automatically we have Noise
            label='Noise'
        filewise_output[i][3] = label

    return filewise_output   


#2nd option: 3 labels, evaluate it filewise
def File_label2(predictions, spec_id, segments_filewise_test, filewise_output, file_number ):
    """
    FIle_label2 use the predictions made by the CNN to update the filewise annotations
    when we have 3 labels: 0 (LT), 1(ST), 2 (Noise)
    
    METHOD: evaluation of probability over files
        P(2)>50% => Noise
        P(0)>70 => LT
        P(1)>70 => ST
        else => Both
    
     TODO: how can I had possible?
    """
   
    
    if len(predictions)!=np.shape(spec_id)[0]:
        print('ERROR: Number of labels is not equal to number of spectrograms' )
    
    # Assesting file label
    for i in range(file_number):
        file = segments_filewise_test[i][0]
        file_prob=np.zeros((3,1))
        spec_num=0   #counts number of spectrograms per file
        #flag: if no click detected no spectrograms
        click_detected_flag=False
        #        looking for all the spectrogram related to this file

        for k in range(np.shape(spec_id)[0]):
            if spec_id[k][0]==file:
                click_detected_flag=True
                spec_num+=1
                file_prob+=predictions[k]
        if click_detected_flag==True:
            file_prob/=spec_num
            file_prob*=100
            if file_prob[2]>50:
                label='Noise'
            elif file_prob[0]>70:
                label='LT'
            elif file_prob[1]>70:
                label='ST'
            else:
                label='Both'
            
        else:
#            if no clicks => automatically Noise
            label='Noise'
        
        filewise_output[i][3] = label
        
    return filewise_output

#3rd option: 2 lables evaluate it foe each spectrogram
def File_label3(predictions, spec_id, segments_filewise_test, filewise_output, file_number ):
    """
    FIle_label2 use the predictions made by the CNN to update the filewise annotations
    when we have 2 labels: 0 (LT), 1 (ST)
    
    METHOD: evaluation of probability over files, we work directly with predictions
        1) Assign spectrogram Label
            P(0)>0.5 => LT
            P(1)>0.5 => ST
            otherwise Noise
    
        2) Majority vote:
            if No LT or ST => Noise
            if  LT>70% of good spectrogram in file => LT
            if  ST>70% of good spectrogram in file => ST
            otherwise => Both
    
    TODO: how can I had possible?
    """
       
    # Assesting file label 
    for i in range(file_number):
        file = segments_filewise_test[i][0]
        #inizializing counters
        LT_count=0
        ST_count=0
        Other_count=0
        spec_num=0   #counts number of spectrograms per file
        #flag: if no click detected no spectrograms
        click_detected_flag=False
        #count LT, ST and others occurrence
        for k in range(np.shape(spec_id)[0]):
            if spec_id[k][0]==file:
                click_detected_flag= True
                if predictions[k][0]*100>50:
                    LT_count+=1
                elif predictions[k][1]*100>50:
                    ST_count+=1
                else:
                    Other_count+=1
                spec_num+=1
                
        #assign label
        if click_detected_flag==True:
            #this makes sense only if there were spectrograms
#            keepung differen spectrograms 
    #        if Other_count>LT_count+ST_count:
    #        if (Other_count/spec_num)*100>90:
            if LT_count+ST_count==0:
                label='Noise'
            else:
                LT_perc=(LT_count/(spec_num-Other_count))*100 #percentage of LT over "good clicks" clicks
                ST_perc=(ST_count/(spec_num-Other_count))*100 #percentage of LT over "good clicks" clicks
                if LT_perc>70:
                    label='LT'
                elif ST_perc>70:
                    label='ST'
                else:
                    label='Both'
        else:
            #if no click automatically we have Noise
            label='Noise'
            
        filewise_output[i][3] = label
   
    return filewise_output

#2nd option: 2 labels, evaluate it filewise
def File_label4(predictions, spec_id, segments_filewise_test, filewise_output, file_number ):
    """
    FIle_label4 use the predictions made by the CNN to update the filewise annotations
    when we have 2 labels: 0 (LT), 1(ST)
    
    METHOD: evaluation of probability over files
        P(0)>70 => LT
        P(1)>70 => ST
        P(0)>30 and P(1)>30 => Both
        else => Noise
    
     TODO: how can I had possible?
    """
   
    
    if len(predictions)!=np.shape(spec_id)[0]:
        print('ERROR: Number of labels is not equal to number of spectrograms' )
    
    # Assesting file label and updating metrics  
    for i in range(file_number):
        file = segments_filewise_test[i][0]
        file_prob=np.zeros((2,1))
        #inizializing counters
        spec_num=0   #counts number of spectrograms per file
        #flag: if no click detected no spectrograms
        click_detected_flag=False
        for k in range(np.shape(spec_id)[0]):
            if spec_id[k][0]==file:
                click_detected_flag=True
                spec_num+=1
                file_prob+=predictions[k]
        if click_detected_flag==True:
            file_prob/=spec_num
            file_prob*=100
            if file_prob[0]>70:
                label='LT'
            elif file_prob[1]>70:
                label='ST'
            elif file_prob[1]>30 and file_prob[0]>30:
                label='Both'
            else:
                label='Noise'
            
        else:
            label='Noise'
        
        filewise_output[i][3] = label
        
    return filewise_output