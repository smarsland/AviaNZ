import WaveletSegment
import SupportClasses
import SignalProc
import Segment
import wavio
import numpy as np
import os, re, json
import math
import librosa
from scipy import signal

# ------ wavelet detection - Filter 0
def detect(dirName='',trainTest=False, species=''):
    """
    Wavelet detection - batch
    """
    cnt=0
    for root, dirs, files in os.walk(str(dirName)):
        for file in files:
            Night = False
            DOCRecording = re.search('(\d{6})_(\d{6})', file)
            if DOCRecording:
                startTime = DOCRecording.group(2)
                if int(startTime[:2]) > 17 or int(startTime[:2]) < 6:   #if int(startTime[:2]) > 18 or int(startTime[:2]) < 6:   #   6pm to 6am as night
                    Night=True
            if file.endswith('.wav') and os.stat(root + '/' + file).st_size != 0 and (Night or not DOCRecording): # avoid day recordings and files with no data (Tier 1 has 0Kb .wavs)
                if file + '.data' not in files:  # skip already processed files
                    filename = root + '/' + file
                    # load wav and annotation
                    wSeg = WaveletSegment.WaveletSegment(species=species)
                    wSeg.loadData(fName=filename[:-4],trainTest=trainTest)
                    datalength = np.shape(wSeg.data)[0]
                    if species!='all':
                        import librosa
                        if (species == 'Kiwi' or species == 'Ruru') and wSeg.sampleRate != 16000:
                            wSeg.data = librosa.core.audio.resample(wSeg.data, wSeg.sampleRate, 16000)
                            wSeg.sampleRate = 16000
                            datalength = np.shape(wSeg.data)[0]

                        # ws = WaveletSegment.WaveletSegment(species=species, annotation=annotation)
                        segments_possible = wSeg.waveletSegment_test(fName=None, data=wSeg.data,
                                                             sampleRate=wSeg.sampleRate, species=species,
                                                             trainTest=trainTest)
                        if type(segments_possible) == tuple:
                            segments_possible = segments_possible[0]
                        # detected=np.ones(900)
                        if len(segments_possible) > 0:
                            post = SupportClasses.postProcess(wSeg.data, wSeg.sampleRate, segments_possible)
                            # post.detectClicks()
                            post.eRatioConfd2()
                            segments_withConf = post.confirmedSegments

                    else:
                        sp = SignalProc.SignalProc()
                        sgRaw = sp.spectrogram(data=wSeg.data, window_width=256, incr=128, window='Hann', mean_normalise=True, onesided=True,multitaper=False, need_even=False)
                        seg = Segment.Segment(wSeg.data, sgRaw, sp, wSeg.sampleRate)
                        segments_possible = seg.bestSegments()

                    if trainTest == True:
                        # turn into binary format to compute with GT
                        detected = np.zeros(len(wSeg.annotation))
                        for seg in segments_possible:
                            for a in range(len(detected)):
                                if math.floor(seg[0]) <= a and a < math.ceil(seg[1]):
                                    detected[a] = 1
                        wSeg.fBetaScore(wSeg.annotation, detected)
                    else:
                        # Save the excel file
                        if DOCRecording:
                            sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
                        else:
                            sTime = 0
                        out = SupportClasses.exportSegments(segments=segments_possible, species=species, startTime=sTime,
                                                        dirName=dirName, filename=filename,
                                                        datalength=datalength, sampleRate=wSeg.sampleRate,
                                                        method='Wavelets', resolution=60, trainTest=trainTest,
                                                        withConf=False,operator="Nirosha",reviewer="Nirosha", minLen=3)
                        out.excel() # all possible segments
                        # out.saveAnnotation()
                        out.withConf = True
                        out.segments = segments_withConf
                        out.confirmedSegments = segments_withConf
                        out.segmentstoCheck = segments_possible
                        out.excel() # only the segments those passed eRatio test
                        # Save the annotation
                        out.saveAnnotation()
                        cnt=cnt+1
                        print "current: ", cnt

#----- Utility function. Given the annotation, this code generates the excel.
def annotation2excel(dirName='', species=''):
    """ This is to generate the excel output given a set of annotation files.
    """
    cnt = 0
    for root, dirs, files in os.walk(str(dirName)):
        for file in files:
            Night = False
            DOCRecording = re.search('(\d{6})_(\d{6})', file)
            if DOCRecording:
                startTime = DOCRecording.group(2)
                if int(startTime[:2]) > 18 or int(startTime[:2]) < 6:  # 6pm to 6am as night
                    Night = True
                    sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
            else:
                sTime = 0
            if file.endswith('.data') and (Night or not DOCRecording):     #and os.stat(root + '/' + file).st_size != 0:  # avoid day recordings and files with no data (Tier 1 has 0Kb .wavs)
                filename = root + '/' + file[:-5]
                file = open(filename + '.data', 'r')
                segments = json.load(file)
                file.close()
                if len(segments) > 0:
                    if segments[0][0] == -1:
                        operator = segments[0][2]
                        reviewer = segments[0][3]
                        del segments[0]
                # now seperate the segments into possible and withConfidence
                segments_possible=[]
                segments_withConf=[]
                for seg in segments:
                    if seg[4]=='Kiwi':
                        segments_possible.append([seg[0],seg[1]])
                        segments_withConf.append([seg[0],seg[1]])
                    elif seg[4]=='Kiwi?':
                        segments_possible.append([seg[0],seg[1]])

                out = SupportClasses.exportSegments(segments=segments_possible, species='Kiwi', startTime=sTime,
                                                    dirName=dirName, filename=filename,
                                                    datalength=14400000, sampleRate=16000,
                                                    method='Wavelets', resolution=60, trainTest=False,
                                                    withConf=False)
                out.excel()  # all possible segments
                # out.saveAnnotation()
                out.withConf = True
                out.segments = segments_withConf
                out.seg_pos = segments_possible
                out.excel()  # only the segments those passed eRatio test
                cnt=cnt+1
                print "current: ", cnt

# ----- Utility function. When no kiwi was detected just make empty data file.
def emptyData(dirName='',trainTest=False, species=''):
    cnt=1
    for root, dirs, files in os.walk(str(dirName)):
        for file in files:
            Night = False
            DOCRecording = re.search('(\d{6})_(\d{6})', file)
            if DOCRecording:
                startTime = DOCRecording.group(2)
                if int(startTime[:2]) > 18 or int(startTime[:2]) < 6:   #   6pm to 6am as night
                    Night=True
                    # print "Night recording...", file
            if file.endswith('.wav') and Night and os.stat(root + '/' + file).st_size != 0: # avoid day recordings and files with no data (Tier 1 has 0Kb .wavs)
                if file + '.data' not in files:  # skip already processed files
                    # Save the excel file
                    print file
                    filename = root + '/' + file
                    sTime= int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
                    out = SupportClasses.exportSegments(segments=[], species=species, startTime=sTime,
                                                    dirName=dirName, filename=filename,
                                                    datalength=14400000, sampleRate=16000,
                                                    method='Wavelets', resolution=60, trainTest=trainTest,withConf=False)
                    #out.excel() # all possible segments
                    # out.saveAnnotation()
                    #out.withConf=True
                    out.segments=[]
                    out.seg_pos=[]
                    #out.excel() # only the segments those passed eRatio test
                    # Save the annotation
                    out.saveAnnotation()
                    cnt=cnt+1
                    print "current: ", cnt

# ----- Remove segments < 4 seconds long from the annotations (.data) - Filter 1
def deleteShort(dirName='', minLen=4):
    """
    This will delete short segments from the annotation
    """
    cnt = 0
    for root, dirs, files in os.walk(str(dirName)):
        for file in files:
            if file.endswith('.data') and file[:-5] in files:   # skip GT annotations
                file = root + '/' + file
                with open(file) as f:
                    segments = json.load(f)
                    newSegments=[]
                    chg = False
                    for seg in segments:
                        if seg[0] == -1:
                            newSegments.append(seg)
                        elif seg[1]-seg[0] < minLen:
                            chg = True
                            continue
                        else:
                            newSegments.append(seg)
                if chg:
                    file = open(file, 'w')
                    json.dump(newSegments, file)
                cnt += 1
                print file, cnt

def eRatio(dirName, thr=2.5):
    for root, dirs, files in os.walk(str(dirName)):
        for file in files:
            if file.endswith('.wav'):
                wavobj = wavio.read(root + '\\' + file)
                sampleRate = wavobj.rate
                data = wavobj.data
                if data is not 'float':
                    data = data.astype('float')  # data / 32768.0
                if np.shape(np.shape(data))[0] > 1:
                    data = data[:, 0]
                post = SupportClasses.postProcess(data, sampleRate, [])
                post.eRatioConfd(seg=None, thr=2.5)

# eRatio('E:\AviaNZ\Sound Files\Kiwi\\test\Tier1-TP and FP segments', thr=2.5)

# ----- Delete wind/rain corrupted segments - Filter 2
def deleteWindRain(dirName, windTest=True, rainTest=False, Tmean_wind = 1e-9, T_ERatio=1.5):
    """
    Given the directory of sounds this deletes the annotation segments with wind/rain corrupted files.
    Targeting moderate wind and above. Check to make sure the segment to delete has no sign of kiwi
    Automatic Identification of Rainfall in Acoustic Recordings by Carol Bedoya, Claudia Isaza, Juan M.Daza, and Jose D.Lopez
    """
    #Todo: find thrs
    import copy
    Tmean_rain = 1e-8   # Mean threshold
    Tsnr_rain = 10     # SNR threshold

    # Tmean_wind = 1e-9   # Mean threshold
    # Tsnr_wind = 0.5     # SNR threshold

    cnt = 0
    for root, dirs, files in os.walk(str(dirName)):
        for file in files:
            if file.endswith('.data') and file[:-5] in files:
                # go through each segment
                file = root + '/' + file
                with open(file) as f:
                    segments = json.load(f)
                    newSegments=copy.deepcopy(segments)
                    chg = False
                    for seg in segments:
                        if seg[0] == -1:
                            continue
                        else:
                            # read the sound segment and check for wind
                            secs = seg[1]-seg[0]
                            wavobj = wavio.read(file[:-5], nseconds=secs, offset=seg[0])
                            sampleRate = wavobj.rate
                            data = wavobj.data
                            if data is not 'float':
                                data = data / 32768.0
                            data = data[:,0].squeeze()

                            wind_lower = 2.0 * 100 / sampleRate
                            wind_upper = 2.0 * 250 / sampleRate
                            rain_lower = 2.0 * 600 / sampleRate
                            rain_upper = 2.0 * 1200 / sampleRate

                            f, p = signal.welch(data, fs=sampleRate, window='hamming', nperseg=512, detrend=False)

                            if windTest:
                                limite_inf = int(round(len(p) * wind_lower)) # minimum frequency of the rainfall frequency band 0.00625(in
                                                                     # normalized frequency); in Hz = 0.00625 * (44100 / 2) = 100 Hz
                                limite_sup = int(round(len(p) * wind_upper)) # maximum frequency of the rainfall frequency band 0.03125(in
                                                                     # normalized frequency); in Hz = 0.03125 * (44100 / 2) = 250 Hz
                                a_wind = p[limite_inf:limite_sup] # section of interest of the power spectral density.Step 2 in Algorithm 2.1

                                mean_a_wind = np.mean(a_wind) # mean of the PSD in the frequency band of interest.Upper part of the step 3 in Algorithm 2.1
                                std_a_wind = np.std(a_wind)  # standar deviation of the PSD in the frequency band of the interest. Lower part of the step 3 in Algorithm 2.1

                                # c_wind = mean_a_wind / std_a_wind  # signal to noise ratio of the analysed recording. step 3 in Algorithm 2.1


                                if mean_a_wind > Tmean_wind:
                                    # check if it is not kiwi
                                    if '?' in str(seg[4]):  # 'Kiwi?' segments didn't pass eRatio
                                        # x = eRatioConfd(file[:-5], seg, thr=T_ERatio)
                                        potentialCall = False
                                    else:
                                        potentialCall = True
                                    if not potentialCall:
                                        print file, seg, "--> windy"
                                        newSegments.remove(seg)
                                        chg = True
                                else:
                                    print file, seg, "--> not windy"
                            if rainTest:
                                limite_inf = int(round(len(p) * rain_lower)) # minimum frequency of the rainfall frequency band 0.0272 (in
                                                                             # normalized frequency); in Hz=0.0272*(44100/2)=599.8  Hz
                                limite_sup = int(round(len(p) * rain_upper)) # maximum frequency of the rainfall frequency band 0.0544 (in
                                                                             # normalized frequency); in Hz=0.0544*(44100/2)=1199.5 Hz
                                a_rain = p[limite_inf:limite_sup]   # section of interest of the power spectral density.Step 2 in Algorithm 2.1

                                mean_a_rain = np.mean(a_rain)   # mean of the PSD in the frequency band of interest.Upper part of the step 3 in Algorithm 2.1
                                std_a_rain = np.std(a_rain)     # standar deviation of the PSD in the frequency band of the interest. Lower part of the step 3 in Algorithm 2.1

                                c_rain = mean_a_rain / std_a_rain   # signal to noise ratio of the analysed recording. step 3 in Algorithm 2.1

                                if mean_a_rain > Tmean_rain and c_rain > Tsnr_rain:
                                    print file, "--> rainy"
                                    # rainy.append(mean_a_rain)
                                else:
                                    # rainy.append(0)
                                    print file, "--> not rainy"

                    if chg:
                        file = open(file, 'w')
                        json.dump(newSegments, file)
                cnt += 1
                print file, cnt

# deleteWindRain('E:\AviaNZ\Sound Files\\tier1-test\\notWindy', windTest=True, rainTest=False)

#----- after each filter assess the accuracy against the ground truth
def accuracy(dirName):
    """
    compare the annotation with its GT tex file
    """
    TP=FP=TN=FN=0.0
    for root, dirs, files in os.walk(str(dirName)):
        for file in files:
            if file.endswith('.data') and file[:-5] in files:
                # f = open(root + '/' + file[:-9]+'-sec.txt', 'r')
                # GT = f.read()
                # f.close()
                with open(root + '/' + file[:-9]+'-sec.txt') as f:
                    GT = []
                    for line in f:
                        x = int(line.split()[1])
                        GT.append(x)
                with open(root + '/' + file) as f:
                    segments = json.load(f)
                res = np.zeros((900)).astype(int)
                for seg in segments:
                    if seg[0] == -1:
                        continue
                    res[int(np.ceil(seg[0])-1):int(np.ceil(seg[1])-1)] = 1
                wSeg = WaveletSegment.WaveletSegment(species='Kiwi')
                print file
                fB, recall, tp, fp, tn, fn = wSeg.fBetaScore(np.asarray(GT), res)
                TP += tp
                TN += tn
                FP += fp
                FN += fn
    print '------SUMMARY TP, FP, TN, FN, recall, precision, specificity, accuracy-------'
    print int(TP), int(FP), int(TN), int(FN), TP/(TP+FN), TP/(TP+FP), TN/(TN+FP), (TP + TN)/ (TP + TN + FP + FN) # TP, FP, TN, FN, recall, precision, specificity, accuracy

# filter 1
# deleteShort(dirName='E:\AviaNZ\Sound Files\Kiwi\\test\Tier1', minLen=4)
# accuracy('E:\AviaNZ\Sound Files\Kiwi\\test\Tier1')
# filter 2


#----- post process with DTW and MFCC
def loadFile(filename):
    wavobj = wavio.read(filename)
    sampleRate = wavobj.rate
    audiodata = wavobj.data

    # None of the following should be necessary for librosa
    if audiodata.dtype is not 'float':
        audiodata = audiodata.astype('float') #/ 32768.0
    if np.shape(np.shape(audiodata))[0]>1:
        audiodata = audiodata[:,0]

    # if sampleRate != 16000:
    #     audiodata = librosa.core.audio.resample(audiodata, sampleRate, 16000)
    #     sampleRate=16000

    # pre-process
    sc = SupportClasses.preProcess(audioData=audiodata, sampleRate=sampleRate, species='Kiwi', df=False)
    audiodata,sampleRate = sc.denoise_filter()
    return audiodata,sampleRate

def MFCC(y,sr):
    # Calculate MFCC
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=24, n_fft=2048, hop_length=512)  # n_fft=10240, hop_length=2560
    mfcc_delta=librosa.feature.delta(mfcc)
    mfcc=np.concatenate((mfcc,mfcc_delta),axis=0)
    # Remove mean and normalize each column of MFCC
    import copy
    def preprocess_mfcc(mfcc):
        mfcc_cp = copy.deepcopy(mfcc)
        for i in xrange(mfcc.shape[1]):
            mfcc_cp[:, i] = mfcc[:, i] - np.mean(mfcc[:, i])
            mfcc_cp[:, i] = mfcc_cp[:, i] / np.max(np.abs(mfcc_cp[:, i]))
        return mfcc_cp

    mfcc = preprocess_mfcc(mfcc)
    # average MFCC over all frames
    mfcc = mfcc.mean(1)
    return mfcc

def DTW(mfcc1, mfcc2):
    # Calculate the distances from the test signal to ref
    d, wp = librosa.dtw(mfcc1, mfcc2, metric='euclidean')
    return d[d.shape[0] - 1][d.shape[1] - 1]

def MFCC_pool(dirName):
    ''' This is to prepare reference templates: convert sound into MFCC and save as a txt file
    '''
    mfcc_all=[]
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.endswith('.wav'):
                filename = root + '/' + filename  # [:-4]
                y, sr = loadFile(filename)
                mfcc = MFCC(y, sr)
                mfcc = mfcc.transpose()
                mfcc = mfcc.tolist()
                mfcc_all.append(mfcc)
                # mfcc_all.append('\n')
    MFCCTable = np.zeros((len(mfcc_all),len(mfcc_all[0])))
    i=0
    for item in mfcc_all:
        j=0
        for x in item:
            MFCCTable[i,j]=x
            j += 1
        i += 1

    with open(dirName + '\MFCC.txt', 'w') as f:
        f.write(json.dumps(MFCCTable.tolist()))

# MFCC_pool(dirName='..\Sound Files\dtw_mfcc\kiwi')

def isKiwi_dtw_mfcc_batch(dirName, refDir, species='Kiwi'):
    # Post process a batch of recordings using MFCC and DTW, update the annotation
    for root, dirs, files in os.walk(str(dirName)):
        for file in files:
            Night = False
            DOCRecording = re.search('(\d{6})_(\d{6})', file)
            if DOCRecording:
                startTime = DOCRecording.group(2)
                if int(startTime[:2]) > 18 or int(startTime[:2]) < 6:  # 6pm to 6am as night
                    Night = True
                    sTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
            else:
                sTime=0
            if file.endswith('.data') and (Night or not DOCRecording):     #and os.stat(root + '/' + file).st_size != 0:  # avoid day recordings and files with no data (Tier 1 has 0Kb .wavs)
                filename = root + '/' + file[:-5]
                annotation_file = open(filename + '.data', 'r')
                segments = json.load(annotation_file)
                annotation_file.close()

                new_segments = []
                if len(segments)>0 and segments[0][0] == -1:
                    new_segments.append(segments[0])
                    del segments[0]
                if len(segments) == 0:
                    continue
                # now it got some segments to consider, so load the corresponding wav
                # filename_wav = filename[:-5] + '.wav'
                if file[:-5] not in files:
                    continue
                else:
                    print filename
                    audioData, sampleRate = loadFile(filename)  # read, filter, and re-sample to 16 kHz

                for seg in segments:
                    if str(seg[4]) == 'Kiwi' or str(seg[4]) == 'Kiwi?':
                        # extract that bit and do MFCC DTW if it is not a too short segment (2 sec)
                        if seg[1]-seg[0]<3:
                            continue
                        s = int(seg[0]) * sampleRate
                        e = int(seg[1]) * sampleRate
                        isKiwi = isKiwi_dtw_mfcc(refDir, audioData[s:e], sampleRate,mfccFromFile=True)
                        if isKiwi:
                            new_segments.append([seg[0],seg[1], 0, 0, 'Kiwi'])
                # save the nw segments
                if isinstance(filename, str):
                    file = open(filename + '.data', 'w')
                else:
                    file = open(str(filename) + '.data', 'w')
                json.dump(new_segments, file)

def isKiwi_dtw_mfcc(dirName, yTest, srTest, mfccFromFile=False):
    ''' Input the set of kiwi templates (a folder) and the test file
    :return: a binary value (kiwi or not)
    '''
    dList=[]
    mfccTest = MFCC(yTest, srTest)
    if mfccFromFile:
        filename = dirName + '/MFCC.txt'
        f = open(filename, 'r')
        mfccs = json.load(f)
        f.close()

        for mfccRef in mfccs:
            d = DTW(mfccRef, mfccTest)
            dList.append(d)
    else:
        for root, dirs, files in os.walk(str(dirName)):
            for filename in files:
                if filename.endswith('.wav'):
                    filename = root + '/' + filename #[:-4]
                    y, sr = loadFile(filename)
                    mfccRef = MFCC(y,sr)
                    d = DTW(mfccRef,mfccTest)
                    dList.append(d)
    if sorted(dList)[1]<0.6:
        print 'it is KIWI',sorted(dList)[1]
        return True
    else:
        print 'it is NOT kiwi', sorted(dList)[1]
        return False
    # return dList

def mfcc_dtw(y, sr, yTest, srTest):
    # Calculate MFCC of test and reference, return the DTW distance between them
    # First convert the data to mfcc:
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=24,n_fft=2048, hop_length=512) # n_fft=10240, hop_length=2560
    mfccTest = librosa.feature.mfcc(yTest, srTest, n_mfcc=24, n_fft=2048, hop_length=512)
    # get delta mfccs
    mfcc_delta=librosa.feature.delta(mfcc)
    mfccTest_delta=librosa.feature.delta(mfccTest)
    # then merge
    mfcc=np.concatenate((mfcc,mfcc_delta),axis=0)
    mfccTest = np.concatenate((mfccTest, mfccTest_delta), axis=0)

    # Remove mean and normalize each column of MFCC
    import copy
    def preprocess_mfcc(mfcc):
        mfcc_cp = copy.deepcopy(mfcc)
        for i in xrange(mfcc.shape[1]):
            mfcc_cp[:, i] = mfcc[:, i] - np.mean(mfcc[:, i])
            mfcc_cp[:, i] = mfcc_cp[:, i] / np.max(np.abs(mfcc_cp[:, i]))
        return mfcc_cp

    mfcc = preprocess_mfcc(mfcc)
    mfccTest = preprocess_mfcc(mfccTest)

    #average MFCC over all frames
    mfcc=mfcc.mean(1)
    mfccTest=mfccTest.mean(1)

    # Calculate the distances from the test signal
    d, wp = librosa.dtw(mfccTest, mfcc, metric='euclidean')
    return d[d.shape[0] - 1][d.shape[1] - 1]
    # d = dtw(mfccTest,mfcc,wantDistMatrix=False)
    # return d

def dtw(x,y,wantDistMatrix=False):
    # Compute the dynamic time warp between two 1D arrays
    # same as librosa dtw
    # TODO: stealed from Segments
    dist = np.zeros((len(x)+1,len(y)+1))
    dist[1:,:] = np.inf
    dist[:,1:] = np.inf
    for i in range(len(x)):
        for j in range(len(y)):
            dist[i+1,j+1] = np.abs(x[i]-y[j]) + min(dist[i,j+1],dist[i+1,j],dist[i,j])
    if wantDistMatrix:
        return dist
    else:
        return dist[-1,-1]

# isKiwi_dtw_mfcc_batch(dirName='sound\Ponui-test', refDir='..\Sound Files\dtw_mfcc\kiwi')
# isKiwi_dtw_mfcc_batch(dirName='E:/Employ/Halema/Survey2/Card 1', refDir='..\Sound Files\dtw_mfcc\kiwi')