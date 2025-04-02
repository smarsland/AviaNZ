import os, math
import numpy as np
import SignalProc 

def makeBatData(dirName,imgWidth,imgHeight,incr,img,scale=False):
    # Make bat data matrix. It will be size nimages * imgHeight * imgWidth
    # This is intended to run once and save the images
    # Assumes that the data are in folders called ST LT NOISE
    # img parameter is whether to assemble a data matrix or save images

    # You need to say how many images to make, which is a bit crap

    # PARAM: # noise images
    nnoise = 15

    lots = 100000
    train_x = np.zeros((lots,imgWidth,imgHeight))
    train_y = []
    train_y2 = []
    count = 0

    sp = SignalProc.SignalProc(1024,512)
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
                                y=2
                                for s in starts:
                                    if img: 
                                        if imgHeight==64:
                                            np.save(os.path.join(imgsavepath,str(y) + '_' + "%06d" % count + '.npy'), sp.sg[:,s:s+imgWidth].T)
                                        else:
                                            # Blank section at top
                                            x[:,:64] = sp.sg[:,s:s+imgWidth].T
                                            np.save(os.path.join(imgsavepath,str(y) + '_' + "%06d" % count + '.npy'), x)
                                    else:
                                        train_x[count,:,:] = sp.sg[:,s:s+imgWidth].T
                                    train_y.append(str(y) + '_' + "%06d" % count + '.npy, '+str(y))
                                    train_y2.append(y)
                                    count+=1
                            else:
                                # What if no clicks found? 
                                # For now: ignore it
                                res = sp.clickSearch(returnAll=True)
                                if classname == "LT":
                                    y = 0
                                elif classname == "ST":
                                    y = 1
                                else:
                                    print("ERROR!")
                                if res is not None:
                                    # Assemble a set of images
                                    # PARAM: Start a bit before the start, finish a bit before the last, if possible
                                    start = max(res[1]-incr,0)
                                    end = min(res[2]+incr,np.shape(sp.sg)[1])
                                    i = 0
                                    #print(start,start+(i+1)*imgWidth,end)
                                    while start+imgWidth < end:
                                        # If there is a click in the section keep it, otherwise don't bother
                                        hasClicks = ((res[0] > start) & (res[0]< start+imgWidth)).any()
                                        #print(hasClicks,res[0],start+i*imgWidth)
                                        if hasClicks:
                                            train_y.append(str(y) + '_' + "%06d" % count + '.npy, '+str(y))
                                            train_y2.append(y)
                                            if img:
                                                if imgHeight==64:
                                                        np.save(os.path.join(imgsavepath,str(y) + '_' + "%06d" % count + '.npy'), sp.sg[:,start+i*imgWidth:start+(i+1)*imgWidth].T)
                                                else:
                                                        x[:,:64] = sp.sg[:,start+i*imgWidth:start+(i+1)*imgWidth].T
                                                        np.save(os.path.join(imgsavepath,str(y) + '_' + "%06d" % count + '.npy'), x)
                                            else:
                                                train_x[count,:,:] = sp.sg[:,start+i*imgWidth:start+(i+1)*imgWidth].T
                                            count+=1
                                        start = start+incr
                                #else:
                                    #print(classname,"None")
                                    #start = int(0.2*np.shape(sp.sg)[1])
                                    #end = int(0.6*np.shape(sp.sg)[1])

    train_y = train_y[:count]
    train_y2 = train_y2[:count]
    if img:
        np.savetxt(os.path.join(imgsavepath,'lab2.txt'),train_y2)
        with open(os.path.join(imgsavepath,'labels.txt'), 'w') as f:
            for y in train_y:
                f.write(y + '\n')
        
    else:
        train_x = train_x[:count,:,:]
        name = 'bat_data_'+str(imgWidth)+'_'+str(imgHeight)+str(incr)+'.npz'
        np.savez_compressed(name, x=train_x,y=train_y)                                    
    #nLT = np.shape(np.where(train_y==0))[1]
    #nST = np.shape(np.where(train_y==1))[1]
    #nN = np.shape(np.where(train_y==2))[1]
    #print(count,nLT,nST,nN)

makeBatData('Full',imgWidth=128,imgHeight=64,incr=32,img=True,scale=False)
#makeBatData('Check',imgWidth=128,imgHeight=64,incr=32,img=True,scale=True)
