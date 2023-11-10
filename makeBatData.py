
import os,math
import numpy as np
import Spectrogram

def ClickSearch(imspec, sampleRate):
    """
    searches for clicks in the provided imspec, saves dataset
    returns click_label, dataset and count of detections

    The search is made on the spectrogram image that we know to be generated
    with parameters (1024,512)
    Click presence is assessed for each spectrogram column: if the mean in the
    frequency band [f0, f1] (*) is bigger than a treshold we have a click
    thr=mean(all_spec)+std(all_spec) (*)

    The clicks are discarded if longer than 0.05 sec

    imspec: unrotated spectrogram (rows=time)
    file: NOTE originally was basename, now full filename
    """
    df=sampleRate//2 /(np.shape(imspec)[0]+1)  # frequency increment
    #dt=sp.incr/sampleRate  # self.sp.incr is set to 512 for bats
    # up_len=math.ceil(0.05/dt) #0.5 second lenth in indices divided by 11
    up_len=17
    # up_len=math.ceil((0.5/11)/dt)

    # Frequency band
    f0=24000
    index_f0=-1+math.floor(f0/df)  # lower bound needs to be rounded down
    f1=54000
    index_f1=-1+math.ceil(f1/df)  # upper bound needs to be rounded up

    # Mean in the frequency band
    mean_spec=np.mean(imspec[index_f0:index_f1,:], axis=0)

    # Threshold
    mean_spec_all=np.mean(imspec, axis=0)[2:]
    thr_spec=(np.mean(mean_spec_all)+np.std(mean_spec_all))*np.ones((np.shape(mean_spec)))

    ## clickfinder
    # check when the mean is bigger than the threshold
    # clicks is an array which elements are equal to 1 only where the sum is bigger
    # than the mean, otherwise are equal to 0
    clicks = mean_spec>thr_spec
    inds = np.where(clicks>0)[0]
    if (len(inds)) > 0:
        # Have found something, now find first that isn't too long
        flag = False
        start = inds[0]
        while flag:
            i=1
            while inds[i]-inds[i-1] == 1:
                i+=1
            end = i
            if end-start<up_len:
                flag=True
            else:
                start = inds[end+1]

        first = start

        # And last that isn't too long
        flag = False
        end = inds[-1]
        while flag:
            i=len(inds)-1
            while inds[i]-inds[i-1] == 1:
                i-=1
            start = i
            if end-start<up_len:
                flag=True
            else:
                end = inds[start-1]
        last = end
        return [first,last]
    else:
        return None

def makeBatData(dirName,imgWidth,imgHeight,incr,img,scale=True):
    # Make bat data matrix. It will be size nimages * imgHeight * imgWidth
    # This is intended to run once and save the images
    # Assumes that the data are in folders called ST LT NOISE
    # img parameter is whether to assemble a data matrix or save images

    # You need to say how many images to make, which is a bit crap

    # PARAM: # noise images
    nnoise = 2000//500

    lots = 100000
    train_x = np.zeros((lots,imgWidth,imgHeight))
    train_y = np.zeros(lots)
    count = 0

    sp = Spectrogram.Spectrogram(1024,512)
    dt=0.002909090909090909
    if img:
        imgsavepath = os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr))
        if not os.path.exists(imgsavepath):
                os.makedirs(imgsavepath)
        x = np.zeros((imgWidth,imgHeight))
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.lower().endswith('.bmp'):
                filename = os.path.join(root, filename)
                #print(filename)

                # check if file not empty
                if os.stat(filename).st_size > 1000:
                    # check if file is formatted correctly
                    with open(filename, 'br') as f:
                        if f.read(2) == b'BM':
                            sp.readBmp(filename,repeat=False,rotate=False,silent=True)
                            # Optional scale to full 0-1
                            if scale: 
                                sp.sg -= np.min(sp.sg)
                                sp.sg /= np.max(sp.sg)
                            #print(np.shape(sp.sg))
                            classname = os.path.split(root)[-1] 
                            #print(classname)
                            if classname != "LT" and classname != "ST":
                                # Different for noise
                                # Random param: number of noise sections
                                starts = np.random.randint(np.shape(sp.sg)[1]-2*imgWidth,size=nnoise)+imgWidth
                                for s in starts:
                                    if img: 
                                        if imgHeight==64:
                                            np.save(os.path.join(imgsavepath,str(2) + '_' + "%06d" % count + '.npy'), sp.sg[:,s:s+imgWidth].T)
                                        else:
                                            x[:,:64] = sp.sg[:,s:s+imgWidth].T
                                            np.save(os.path.join(imgsavepath,str(2) + '_' + "%06d" % count + '.npy'), x)
                                    else:
                                        train_x[count,:,:] = sp.sg[:,s:s+imgWidth].T
                                    train_y[count] = 2
                                    count+=1
                            else:
                                # What if no clicks found? 
                                # For now: ignore that file
                                res = ClickSearch(sp.sg,sp.sampleRate)
                                if res is not None:
                                    # Assemble a set of images
                                    # PARAM: Start a bit before the start, finish a bit before the last, if possible
                                    start = max(res[0]-incr,0)
                                    end = min(res[1]+incr,np.shape(sp.sg)[1])
                                    i = 0
                                    #print(start,start+(i+1)*imgWidth,end)
                                    while start+(i+1)*imgWidth < end:
                                        hasClicks = ((res[0] > start) & (res[0]< start+imgWidth)).any()
                                        if hasClicks:
                                            if classname == "LT":
                                                train_y[count] = 0
                                            elif classname == "ST":
                                                train_y[count] = 1
                                            else:
                                                print("ERROR!")
                                            if img:
                                                if imgHeight==64:
                                                    np.save(os.path.join(imgsavepath,str(int(train_y[count])) + '_' + "%06d" % count + '.npy'), sp.sg[:,start+i*imgWidth:start+(i+1)*imgWidth].T)
                                                else:
                                                    x[:,:64] = sp.sg[:,start+i*imgWidth:start+(i+1)*imgWidth].T
                                                    np.save(os.path.join(imgsavepath,str(int(train_y[count])) + '_' + "%06d" % count + '.npy'), x)
                                            else:
                                                train_x[count,:,:] = sp.sg[:,start+i*imgWidth:start+(i+1)*imgWidth].T
                                            count+=1
                                            i+=1

    train_y = train_y[:count]
    if img:
        np.savetxt(os.path.join(dirName,'img'+str(imgWidth)+"_"+str(imgHeight)+"_"+str(incr),'label_tensorflow.txt'),train_y)
    else:
        train_x = train_x[:count,:,:]
        name = 'bat_data_'+str(imgWidth)+'_'+str(imgHeight)+str(incr)+'.npz'
        np.savez_compressed(name, x=train_x,y=train_y)                                    
    nLT = np.shape(np.where(train_y==0))[1]
    nST = np.shape(np.where(train_y==1))[1]
    nN = np.shape(np.where(train_y==2))[1]
    print(count,nLT,nST,nN)

#makeBatData('/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/New_Train_Datasets/Train_5',imgWidth=240,imgHeight=75,incr=incr,img=True)
#makeBatData('/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BattyBats/New_Train_Datasets/Train_5',imgWidth=224,imgHeight=64,incr=32,img=True,scale=False)
makeBatData('/media/smb-vuwstocoissrin1.vuw.ac.nz-ECS_acoustic_02/BatTraining/Check',imgWidth=192,imgHeight=64,incr=32,img=True,scale=False)

