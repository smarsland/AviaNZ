
import os,math
import SignalProc
import numpy as np

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

def run(dirName):
    sp = SignalProc.SignalProc(1024,512)
    dt=0.002909090909090909
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.lower().endswith('.bmp'):
                filename = os.path.join(root, filename)

                # check if file not empty
                if os.stat(filename).st_size < 1000:
                    print("File %s empty, skipping" % filename)

                # check if file is formatted correctly
                with open(filename, 'br') as f:
                    if f.read(2) != b'BM':
                        print("Warning: file %s not formatted correctly, skipping" % filename)
                    else:
                        sp.readBmp(filename, rotate=False,silent=True)
                        res = ClickSearch(sp.sg,sp.audioFormat.sampleRate())
                        if res is not None:
                            print(filename,(res[1]-res[0])*dt)

run('.')
