import numpy as np
import wavio
import os, json
import math, re
import datetime

#----------------------------generate .data from Freebird tag files
def tag2data(dir,birdlist):
    import os
    import openpyxl
    import xml.etree.ElementTree as ET

    # read freebird bird list
    book = openpyxl.load_workbook(birdlist)
    sheet = book.active
    name = sheet['A2': 'A353']
    code = sheet['B2': 'B353']
    spName=[]
    spCode=[]
    for i in range(len(name)):
        spName.append(str(name[i][0].value))
    for i in range(len(code)):
        spCode.append(int(code[i][0].value))

    spDict = dict(zip(spCode, spName))

    # generate the .data files from .tag
    for root, dirs, files in os.walk(str(dir)):
        for f1 in files:
            if f1.endswith('.tag'):
                DOCRecording = re.search('(\d{6})_(\d{6})', f1)
                if DOCRecording:
                    startTime = DOCRecording.group(2)
                annotation=[]
                tagFile = root + '/' + f1
                tree = ET.parse(tagFile)
                troot = tree.getroot()
                for elem in troot:
                    sp = spDict[int(elem[0].text)]
                    annotation.append([float(elem[1].text),float(elem[1].text)+float(elem[2].text),0,0,sp])
                annotation.insert(0,[-1, startTime, "Doc_Operator", "Doc_Reviewer", -1])
                file = open(tagFile[:-4] + '.wav.data', 'w')
                json.dump(annotation, file)

# tag2data('E:\AviaNZ\Sound Files\Kiwi\\test\Tier1 dataset\DOC annotations\North_Island_tags_2011_16','E:\AviaNZ\Sound Files\Kiwi\\test\Tier1 dataset\DOC annotations\Freebird_species_ list.xlsx')

# what was deleted by Fund frq. filter?
def genDiffAnnotation(dir1, dir2):  # e.g. dir1 is Filter3 and dir2 is Filter4
    for root, dirs, files in os.walk(str(dir1)):
        for f1 in files:
            if f1.endswith('.data'):
                annotation1 = dir1 + '/' + f1
                try:
                    with open(annotation1) as ann1:
                        segments1 = json.load(ann1)
                    annotation2 = dir2 + '/' + f1
                    with open(annotation2) as ann2:
                        segments2 = json.load(ann2)
                except:
                    pass
                annotation = []
                for seg in segments1:
                    if seg[0] == -1:
                        annotation.append(seg)
                    if seg in segments2:
                        continue
                    else:
                        annotation.append(seg)

                file = open(str(annotation1), 'w')
                json.dump(annotation, file)

#genDiffAnnotation('E:\AviaNZ\Sound Files\Kiwi\\test\Tier1 dataset\positive', 'E:\AviaNZ\Sound Files\Kiwi\\test\Tier1 dataset\positive\\filter4-v1')

#------------------------------------------------- code to count segments
def countSegments(dir):
    """
    This counts the segments in the data files in the given folder
    """
    # wavFile=datFile[:-5]
    cnt=0
    for root, dirs, files in os.walk(str(dir)):
        for f1 in files:
            if f1.endswith('.data'):
                annotation = dir + '/' + f1
                try:
                    with open(annotation) as ann:
                        segments = json.load(ann)
                except:
                    pass
                for seg in segments:
                    if seg[0] == -1:
                        pass
                    else:
                        cnt += 1
    print (cnt)

# countSegments('/home/nirosha/Avianz_serverScript/test/')

#------------------------------------------------- delete empty .data files
def delEmpAnn(dir):
    """
    This deletes empty data files in a given folder
    helps when manually reviewing auto detections, because reviwer can skip empty recordings without opening it (red and black color code from the list)
    """
    import os
    for root, dirs, files in os.walk(str(dir)):
        for f1 in files:
            if f1.endswith('.data'):
                annotation = root + '/' + f1
                try:
                    with open(annotation) as ann:
                        segments = json.load(ann)
                except:
                    break
                if len(segments)==0:
                    os.remove(annotation)
                elif len(segments)==1 and segments[0][0]==-1:
                    os.remove(annotation)

# delEmpAnn('F:\Tier1-2014-15')

#------------------------------------------------- code to extract segments
def extractSegments(wavFile, destination, copyName, species):
    """
    This extracts the sound segments given the annotation and the corresponding wav file. (Isabel's experiment data extraction)
    """
    datFile=wavFile+'.data'
    try:
        wavobj = wavio.read(wavFile)
        sampleRate = wavobj.rate
        data = wavobj.data
        if os.path.isfile(datFile):
            with open(datFile) as f:
                segments = json.load(f)
            cnt = 1
            for seg in segments:
                if seg[0] == -1:
                    continue
                if copyName:    # extract all - extracted sounds are saved with the same name as the corresponding segment in the annotation (e.g. Rawhiti exp.)
                    filename = destination + '\\' + seg[4] + '.wav'
                    s = int(seg[0] * sampleRate)
                    e = int(seg[1] * sampleRate)
                    temp = data[s:e]
                    wavio.write(filename, temp.astype('int16'), sampleRate, scale='dtype-limits', sampwidth=2)
                elif not species:   # extract all - extracted sounds are saved with the original file name followed by an index starting 1
                    ind = wavFile.rindex('/')
                    filename = destination + '\\' + str(wavFile[ind + 1:-4]) + '-' + str(cnt) + '.wav'
                    cnt += 1
                    s = int(seg[0] * sampleRate)
                    e = int(seg[1] * sampleRate)
                    temp = data[s:e]
                    wavio.write(filename, temp.astype('int16'), sampleRate, scale='dtype-limits', sampwidth=2)
                elif species == seg[4]:   # extract only specific calls - extracted sounds are saved with with the original file name followed by an index starting 1
                    ind = wavFile.rindex('/')
                    ind2 = wavFile.rindex('\\')
                    filename = destination + '\\' + str(wavFile[ind2+1:ind]) + '-' + str(wavFile[ind + 1:-4]) + '-' + str(seg[4]) + '-' + str(cnt) + '.wav'
                    cnt += 1
                    s = int((seg[0]-1) * sampleRate)
                    e = int((seg[1]+1) * sampleRate)
                    temp = data[s:e]
                    wavio.write(filename, temp.astype('int16'), sampleRate, scale='dtype-limits', sampwidth=2)
    except:
        print ("unsupported file: ", wavFile)
        # pass

# extractSegments('E:\ISABEL\Rawhiti Experiment Data Sorting NP\Station7-DOC16-Night-Expert\St7-Night-Expert-Trial1to7.wav', 'E:\ISABEL\Rawhiti Experiment Data Sorting NP\Station7-DOC16-Night-Expert\songs')

def extractSegments_batch(dirName, destination, copyName = True, species = None):
    """
    This extracts the sound segments in a directory (Isabel's Rawithi experiment data extraction)
    copyName is True when extracted sounds are saved with the same name as the corresponding segment in the annotation
    Specify 'species' when you want to extract specific calls, e.g. 'kiwi(M)1' to extract all v close brown kiwi male calls
    """
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.endswith('.wav') and filename+'.data' in files:
                filename = root + '/' + filename
                extractSegments(filename, destination, copyName=copyName, species = species)

# extractSegments_batch('E:\ISABEL\Rawhiti Experiment Night Data Sorting NP\Rawhiti experiment Top recorder\Top Evening\New folder\New folder', 'E:\ISABEL\Rawhiti Experiment Night Data Sorting NP\Rawhiti experiment Top recorder\Top Evening\New folder\New folder\songss')

# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkf-close', copyName=False, species = 'Bkf1')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkf-close', copyName=False, species = 'Bkf2')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkf-other', copyName=False, species = 'Bkf3')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkf-other', copyName=False, species = 'Bkf4')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkf-other', copyName=False, species = 'Bkf5')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkm-close', copyName=False, species = 'Bkm1')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkm-close', copyName=False, species = 'Bkm2')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkm-other', copyName=False, species = 'Bkm3')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkm-other', copyName=False, species = 'Bkm4')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkm-other', copyName=False, species = 'Bkm5')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkd-close', copyName=False, species = 'Bkd1')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkd-close', copyName=False, species = 'Bkd2')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkd-other', copyName=False, species = 'Bkd3')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkd-other', copyName=False, species = 'Bkd4')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkd-other', copyName=False, species = 'Bkd5')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkr-close', copyName=False, species = 'Bkr1')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkr-close', copyName=False, species = 'Bkr2')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkr-other', copyName=False, species = 'Bkr3')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkr-other', copyName=False, species = 'Bkr4')
# extractSegments_batch('E:\Employ\Halema\Survey2', 'E:\AviaNZ Kiwi sound pool\Taranaki-bkr-other', copyName=False, species = 'Bkr5')


#------------------------------------------------- code to rename the annotations e.g. Kiwi(M)1 into bkm1
def renameAnnotation(dirName, frm, to):
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.endswith('.data') and filename[:-5] in files:
                filename = root + '/' + filename
                with open(filename) as f:
                    segments = json.load(f)
                    chg = False
                    for seg in segments:
                        if seg[0] == -1:
                            continue
                        elif frm == seg[4]:
                            seg[4] = to
                            chg = True
                if chg:
                    file = open(str(filename), 'w')
                    json.dump(segments, file)
                    file.close()

# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi(M)1', 'bkm1')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi(M)2', 'bkm2')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi(M)3', 'bkm3')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi(M)4', 'bkm4')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi(M)5', 'bkm5')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi(F)1', 'bkf1')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi(F)2', 'bkf2')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi(F)3', 'bkf3')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi(F)4', 'bkf4')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi(F)5', 'bkf5')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi1', 'bk1')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi2', 'bk2')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi3', 'bk3')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi4', 'bk4')
# renameAnnotation('E:\Employ\Halema\Survey2', 'Kiwi5', 'bk5')

# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi(M)1', 'bkm1')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi(M)2', 'bkm2')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi(M)3', 'bkm3')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi(M)4', 'bkm4')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi(M)5', 'bkm5')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi(F)1', 'bkf1')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi(F)2', 'bkf2')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi(F)3', 'bkf3')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi(F)4', 'bkf4')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi(F)5', 'bkf5')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi1', 'bk1')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi2', 'bk2')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi3', 'bk3')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi4', 'bk4')
# renameAnnotation('F:\Isabel-Coromendal kiwi-October_2017', 'Kiwi5', 'bk5')

#---------------------------generate GT batch
def annotation2GT(wavFile,species,duration=0):
    """
    This generates the ground truth for a given sound file (currently for kiwi and bittern).
    Given the AviaNZ annotation, returns the ground truth as a txt file
    """
    # wavFile=datFile[:-5]
    datFile=wavFile+'.data'
    eFile = datFile[:-9]+'-sec.txt'
    if duration ==0:
        wavobj = wavio.read(wavFile)
        sampleRate = wavobj.rate
        data = wavobj.data
        duration=len(data)/sampleRate   # number of secs
    GT=np.zeros((duration,4))
    GT=GT.tolist()
    GT[:][1]=str(0)
    GT[:][2]=''
    GT[:][3]=''
    if os.path.isfile(datFile):
        print (datFile)
        with open(datFile) as f:
            segments = json.load(f)
        for seg in segments:
            if seg[0]==-1:
                continue
            # x = re.search(species, str(seg[4]))
            # print x
            if not re.search(species, seg[4]):
                continue
            elif species=='Kiwi' or 'Gsk':
                # check M/F
                if '(M)' in str(seg[4]):        # if re.search('(M)', seg[4]):
                    type = 'M'
                elif '(F)' in str(seg[4]):      #if re.search('(F)', seg[4]):
                    type='F'
                elif '(D)' in str(seg[4]):
                    type='D'
                else:
                    type='K'
            elif species=='Bittern':
                # check boom/inhalation
                if '(B)' in str(seg[4]):
                    type = 'B'
                elif '(I)' in str(seg[4]):
                    type='I'
                else:
                    type=''

            # check quality
            if re.search('1', seg[4]):
                quality = '*****'   # v close
            elif re.search('2', seg[4]):
                quality = '****'    # close
            elif re.search('3', seg[4]):
                quality = '***' # fade
            elif re.search('4', seg[4]):
                quality = '**'  # v fade
            elif re.search('5', seg[4]):
                quality = '*'   # v v fade
            else:
                quality = ''

            s=int(math.floor(seg[0]))
            e=int(math.ceil(seg[1]))
            for i in range(s,e):
                GT[i][1] = str(1)
                GT[i][2] = type
                GT[i][3] = quality
    for line in GT:
        if line[1]==0.0:
            line[1]='0'
        if line[2]==0.0:
            line[2]=''
        if line[3]==0.0:
            line[3]=''
    # now save GT as a .txt file
    for i in range(1, duration + 1):
        GT[i-1][0]=str(i)   # add time as the first column to make GT readable
    out = file(eFile, "w")
    for line in GT:
        print >> out, "\t".join(line)
    out.close()

def genGT(dirName,species='Kiwi',duration=0):
    """
    Given the directory where sound files and the annotations along with the species being considered,
    it generates the ground truth txt files using 'annotation2GT'
    If you know the duration of a recording pass it through 'duration'.
    """
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.endswith('.wav'):
                filename = root + '/' + filename
                annotation2GT(filename,species,duration=duration)
    print ("Generated GT")

# genGT("E:\kiwi-Fiordland roroa-Sandy and Robin\GSKtrain", species='Gsk')
# genGT("Sound Files\\test\\test", species='Kiwi',duration=300)
# genGT('E:\Employ\Kessel ecology\Bittern-Kessels Ecology-Wiea-2017-11-19\DownSampled\KA13_Oct 17-24_down', species='Bittern',duration=900)
# genGT('E:\Employ\Halema\Survey2', species='Kiwi',duration=900)
# genGT('E:\AviaNZ\Sound Files\Kiwi\\test\Tier1-new\\negative', species='Kiwi',duration=900)

# ------------------------------------------------  SIPO
# generate GT for SIPO
# annotation2GT('E:\Rebecca SIPO\\train\Mt Cass coastal SIPO 07072012.wav.data')
# annotation2GT('E:\Rebecca SIPO\\train\\170617_163003_train.wav.data')

# Train
# nodes= ws.waveletSegment_train('E:\Rebecca SIPO\\train\Mt Cass coastal SIPO 07072012',species='SIPO')
# print "Nodes for SIPO: ",nodes
# # # nodes_SIPO=[54,26,2]
# #           = [61L, 53L, 60L, 57L]
#
# nodes = ws.waveletSegment_train('E:\Rebecca SIPO\\train\\170617_163003_train', species='SIPO')
# print "Nodes for SIPO: ",nodes
# [59L, 54L, 51L, 60L, 58L, 49L, 47L]

# d=ws.waveletSegment_test(fName='E:\Rebecca SIPO\\train\Mt Cass coastal SIPO 07072012',data=None, sampleRate=None, species='SIPO',trainTest=False)
# print d

#------------------------------------------------- code to convert the anotations to excel report batch
def genReport(dirName,species='Kiwi'):
    '''
    Convert the kiwi annotations into an excel summary using annotation2Report (for Sumudu)
    '''
    from openpyxl import load_workbook, Workbook
    eFile = dirName + '/report.xlsx'
    wb = Workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="File name")
    ws.cell(row=1, column=2, value="start")
    ws.cell(row=1, column=3, value="end")
    ws.cell(row=1, column=4, value="M/F")
    ws.cell(row=1, column=5, value="quality")
    r = 2
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.endswith('.data'):
                print (filename)
                fName=filename
                filename = root + '/' + filename
                with open(filename) as f:
                    segments = json.load(f)
                if len(segments)==0:
                    continue
                if len(segments)==1 and segments[0][0]==-1:
                    continue
                # Check if the filename is in standard DOC format
                # Which is xxxxxx_xxxxxx.wav or ccxx_cccc_xxxxxx_xxxxxx.wav (c=char, x=0-9), could have _ afterward
                # So this checks for the 6 ints _ 6 ints part anywhere in string
                DOCRecording = re.search('(\d{6})_(\d{6})', filename[-22:-9])

                if DOCRecording:
                    startTime = DOCRecording.group(2)

                    if int(startTime[:2]) > 8 or int(startTime[:2]) < 8:
                        print ("Night time DOC recording")
                    else:
                        print ("Day time DOC recording")
                        # TODO: And modify the order of the bird list
                    startTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
                    print (startTime)
                    print (datetime.timedelta(seconds=startTime))

                c = 1
                ws.cell(row=r, column=c, value=str(fName[:-9]))
                c = c + 1
                for seg in segments:
                    if seg[0] == -1:
                        continue
                    x = re.search('Kiwi', seg[4])
                    if not re.search('Kiwi', seg[4]):
                        continue
                    s = int(math.floor(seg[0]))
                    s = datetime.timedelta(seconds=startTime+s)
                    e = int(math.ceil(seg[1]))
                    e = datetime.timedelta(seconds=startTime+e)

                    ws.cell(row=r, column=c, value=str(s))
                    c=c+1
                    ws.cell(row=r, column=c, value=str(e))
                    c = c + 1
                    if len(seg[4])==5:                          # 'Kiwi?' and 'Kiwi5'
                        gend='K'
                        if str(seg[4][4])=='?':
                            quality = ''
                        elif int(seg[4][4]) == 1 or int(seg[4][4]) == 2 or int(seg[4][4]) == 3 or int(seg[4][4]) == 4 or int(seg[4][4]) == 5 or int(seg[4][4]) == 6:
                            quality = '*' * int(seg[4][4])
                    elif len(seg[4])==6:                        # 'Kiwi5?'
                        gend='?'
                        quality = '*' * int(seg[4][4])
                    elif len(seg[4]) == 8:                      # 'Kiwi(M)1'
                        gend = seg[4][5]
                        quality = '*' * int(seg[4][7])
                    ws.cell(row=r, column=c, value=gend)
                    c = c + 1
                    ws.cell(row=r, column=c, value = quality)
                    c = 2
                    r = r + 1
                # r = r + 1
    wb.save(str(eFile))
    print ("Generated Report")

# genReport('E:\Employ\Halema\Survey2\Card 4',species='Kiwi')
#genReport('G:\Isabel-Summit Forest (Northland)-Karen Lucich 22-11-17\Kiwi Recorder card 5',species='Kiwi')

def genReportBittern(dirName):
    '''
    Convert the bittern annotations into an excel summary using annotation2Report (for Sumudu)
    '''
    from openpyxl import load_workbook, Workbook
    eFile = dirName + '/report.xlsx'
    wb = Workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="File name")
    ws.cell(row=1, column=2, value="start")
    ws.cell(row=1, column=3, value="end")
    ws.cell(row=1, column=4, value="B/I")
    ws.cell(row=1, column=5, value="quality")
    r = 2
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.endswith('.data'):
                fName=filename
                filename = root + '/' + filename
                with open(filename) as f:
                    segments = json.load(f)
                if len(segments)==0:
                    continue
                if len(segments)==1 and segments[0][0]==-1:
                    continue
                # Check if the filename is in standard DOC format
                # Which is xxxxxx_xxxxxx.wav or ccxx_cccc_xxxxxx_xxxxxx.wav (c=char, x=0-9), could have _ afterward
                # So this checks for the 6 ints _ 6 ints part anywhere in string
                DOCRecording = re.search('(\d{6})_(\d{6})', filename)

                if DOCRecording:
                    startTime = DOCRecording.group(2)

                    if int(startTime[:2]) > 8 or int(startTime[:2]) < 8:
                        print ("Night time DOC recording")
                    else:
                        print ("Day time DOC recording")
                        # TODO: And modify the order of the bird list
                    startTime = int(startTime[:2]) * 3600 + int(startTime[2:4]) * 60 + int(startTime[4:6])
                    print (startTime)
                    print (datetime.timedelta(seconds=startTime))

                c = 1
                ws.cell(row=r, column=c, value=str(fName[:-9]))
                c = c + 1
                for seg in segments:
                    if seg[0] == -1:
                        continue
                    # x = re.search('Bittern', seg[4])
                    if not re.search('Bittern', seg[4]):
                        continue
                    else:
                        # check Boom/Inhalation
                        if '(B)' in str(seg[4]):
                            type = 'B'
                        elif '(I)' in str(seg[4]):
                            type = 'I'
                        # quality
                        if int(str(seg[4])[-1:])==1 or int(str(seg[4])[-1:])==2 or int(str(seg[4])[-1:])==3 or int(str(seg[4])[-1:])==4 or int(str(seg[4])[-1:])==5:
                            quality = '*' * int(str(seg[4])[-1:])

                    s = int(math.floor(seg[0]))
                    s = datetime.timedelta(seconds=startTime+s)
                    e = int(math.ceil(seg[1]))
                    e = datetime.timedelta(seconds=startTime+e)

                    ws.cell(row=r, column=c, value=str(s))
                    c=c+1
                    ws.cell(row=r, column=c, value=str(e))
                    c = c + 1
                    ws.cell(row=r, column=c, value=type)
                    c = c + 1
                    ws.cell(row=r, column=c, value = quality)
                    c = 2
                    r = r + 1
                # r = r + 1
    wb.save(str(eFile))
    print ("Generated Report")

# genReportBittern('E:\Employ\Kessel ecology\Bittern-Kessels Ecology-Wiea-2017-11-19\DownSampled\KA13 Oct 25-31 down')

#------------------------------------------------- code to evaluate batch
def batch_fB(dirName,species,length):
    ''' This is to assess the detection results. Input the directory to assess containing GT (.txt)
    and annotation (.data) made by the detector
    length: length of a recording in sec
    '''
    import os, WaveletSegment
    TP=FP=TN=FN=0
    detected = np.zeros(length)
    ws = WaveletSegment.WaveletSegment(wavelet='dmey')
    for root, dirs, files in os.walk(str(dirName)):
        for filename in files:
            if filename.endswith('.data'):
                filename = root + '/' + filename
                # load the .data file
                with open(filename) as f:
                    segments = json.load(f)
                if len(segments) == 0:
                    detected = detected
                if len(segments) == 1 and segments[0][0] == -1:
                    detected = detected
                # convert it into a binary thing

                # load the GT
                # use fB
                if not train:
                    print ("***", filename)
                    det, tp, fp, tn, fn = ws.waveletSegment_test(filename, listnodes=listnodes, species=species, trainTest=True,df=df)
                    TP+=tp
                    FP+=fp
                    TN+=tn
                    FN+=fn

    if train:
        print ('----- wavelet nodes for the species', nodeList)
    else:
        print ("-----TP   FP  TN  FN")
        print (TP, FP, TN, FN)