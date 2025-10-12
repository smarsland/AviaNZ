# Deprecated bat export methods from AviaNZ_batch.py
# These methods were replaced by the unified exportBatResults() method
# Kept here for reference and for use by old archive scripts

# To use these methods, you would need to:
# 1. Import necessary modules (os, fnmatch, numpy, etc.)
# 2. Have access to Segment, Spectrogram classes
# 3. Have self.config and other class attributes available

# For active code, use: AviaNZ_batchProcess.exportBatResults(dirName, format='xml'|'csv'|'passes')

def outputBatPasses(self,dirName,savefile='BatPasses.csv'):
    """DEPRECATED: Use exportBatResults(dirName, format='passes') instead."""
    # A bit ad hoc for now. Assumes that the directory structure ends with 'Bat detname date/date/'
    if not hasattr(self, 'sp'):
        print("LOADING SP 3")
        from src.core import Spectrogram
        self.sp = Spectrogram.Spectrogram(self.config['window_width'], self.config['incr'])
    start = "Tally,Night,Site,Detector,Detector Name,Bat species (L or S), Time of bat pass (24 hour clock e.g. 23:41:11),Length of bat pass (s),Feeding buzz present (yes/no)\n"
    output = start
    dt=0.002909090909090909
    if not os.path.isdir(dirName):
       print("Folder doesn't exist")
       return 0
    tally = 0
    import os
    from src.core import Annotation
    for root, dirs, files in os.walk(dirName, topdown=True):
        nfiles = len(files)
        if nfiles > 0:
            for count in range(nfiles):
                filename = files[count]
                if filename.endswith('.data'):
                    segments = Annotation.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
                    # TODO:Should be able to remove this...
                    label = 'Non-bat'
                    if len(segments)>0:
                        # Get the length of the clicks from the spectrogram
                        fn = filename[:-5]
                        #print(fn,os.path.join(root, fn))
                        self.sp.readBmp(os.path.join(root, fn), rotate=False,silent=True)
                        #self.sampleRate = self.sp.sampleRate
                        res = self.ClickSearch(self.sp.sg,None,virginia=False)
                        if res is not None:
                            length = "{:.2f}".format((res[1]-res[0])*dt)
                        else:
                            length = str(0)
                        #print("Length "+length)
                        seg = segments[0]
                        #print(seg)
                        c = [lab["certainty"] for lab in seg[4]]
                        s = [lab["species"] for lab in seg[4]]
                        if len(c)>1:
                            label = 'L,S'
                        else:
                            if s[0] == 'Long-tailed bat':
                                label = 'L'
                            elif s[0] == 'Short-tailed bat':
                                label = 'S'
                    else:
                        length = "0"
                        label = ''
                    #print("label "+label)

                    # DOC format
                    # night comes from the directory
                    night = root[-2:]+"/"+root[-4:-2]+"/"+root[-6:-4]
                    print(night)
                    folder = root.split("/")[-2]
                    print(folder)
                    # TODO -- Note sure what this is doing?!
                    #detname = folder.split(" ")[-2]
                    detname = ""
                    #print(detname,folder)
                    #print("night "+night)
                    #night = filename[6:8]+"/"+filename[4:6]+"/"+filename[2:4]
                    # time comes from file
                    time = filename[9:11]+":"+filename[11:13]+":"+filename[13:15]
                    #print("time "+time)
                    
                    output+= str(tally)+","+night+",,,"+detname+","+label+","+time+","+length+",\n"
                    tally += 1
    # Now write the file if necessary
    if output != start:
        file = open(os.path.join(dirName, savefile), 'w')
        print("writing to", os.path.join(dirName, savefile))
        file.write(output)
        file.write("\n")
        file.close()
        output = start

def exportToBatSearch(self,dirName,savefile='BatData.xml',threshold1=0.85,threshold2=0.7):
    """DEPRECATED: Use exportBatResults(dirName, format='xml') instead."""
    # Write out a BatData.xml that can be used for BatSearch import
    # The format of Bat searches is <Survey> / <Site> / Bat / <Date> / files ----- the word Bat is fixed
    # The BatData.xml goes in the Date folder
    # TODO: No error checking!
    # TODO: Check date
    from lxml import etree 
    import os
    import fnmatch
    from src.core import Annotation

    # TODO: Get version label!
    operator = "AviaNZ 3.0"
    site = "Nowhere"

    # BatSeach codes
    namedict = {"Unassigned":0, "Non-bat":1, "Unknown":2, "Long Tail":3, "Short Tail":4, "Possible LT":5, "Possible ST":6, "Both":7}
    if not os.path.isdir(dirName):
        print("Folder doesn't exist")
        return 0
    for root, dirs, files in os.walk(dirName, topdown=True):
        #nfiles = len(files)
        #if nfiles > 0:
        if any(fnmatch.fnmatch(filename, '*.bmp') for filename in files):
            # Set up the XML start
            schema = etree.QName("http://www.w3.org/2001/XMLSchema-instance", "schema")
            start = etree.Element("ArrayOfBatRecording", nsmap={'xsi': "http://www.w3.org/2001/XMLSchema-instance", 'xsd':"http://www.w3.org/2001/XMLSchema"})

            for filename in files:
            #for count in range(nfiles):
                #filename = files[count]
                if filename.endswith('.data'):
                    s1 = etree.SubElement(start,"BatRecording")
                    segments = Annotation.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
                    # TODO:Should be able to remove this...
                    label = 'Non-bat'
                    if len(segments)>0:
                        seg = segments[0]
                        #print(seg)
                        c = [lab["certainty"] for lab in seg[4]]
                        s = [lab["species"] for lab in seg[4]]
                        if len(c)>1:
                            label = 'Both'
                        else:
                            if c[0]>=threshold1:
                                if s[0] == 'Long-tailed bat':
                                    label = 'Long Tail'
                                elif s[0] == 'Short-tailed bat':
                                    label = 'Short Tail'
                            elif threshold2 is not None:
                                if c[0]>threshold2:
                                    if s[0] == 'Long-tailed bat':
                                        label = 'Possible LT'
                                    elif s[0] == 'Short-tailed bat':
                                        label = 'Possible ST'
                            elif threshold2 is None:
                                if s[0] == 'Long-tailed bat':
                                    label = 'Possible LT'
                                elif s[0] == 'Short-tailed bat':
                                    label = 'Possible ST'
                            else:
                                label = 'Non-bat'
                    else:
                        # TODO: which?
                        label = 'Non-bat'
                        #label = 'Unassigned'
                    # This is the text for the file
                    s2 = etree.SubElement(s1,"AssignedBatCategory")
                    s3 = etree.SubElement(s1,"AssignedSite")
                    s4 = etree.SubElement(s1,"AssignedUser")
                    s5 = etree.SubElement(s1,"RecTime")
                    s6 = etree.SubElement(s1,"RecordingFileName")
                    s7 = etree.SubElement(s1,"RecordingFolderName")
                    s8 = etree.SubElement(s1,"MeasureTimeFrom")

                    # TODO: which?
                    #s2.text = str(label)
                    s2.text = str(namedict[label])
                    s3.text = site
                    s4.text = operator
                    # DOC format -- BatSearch wants yyyy-mm-ddThh:mm:ss
                    if len(filename.split('_')[0]) == 6:
                        # ddmmyy
                        timedate = "20"+filename[4:6]+"-"+filename[2:4]+"-"+filename[0:2]+"T"+filename[7:9]+":"+filename[9:11]+":"+filename[11:13]
                    elif len(filename.split('_')[0]) == 8:
                        # yyyymmdd
                        timedate = filename[:4]+"-"+filename[4:6]+"-"+filename[6:8]+"T"+filename[9:11]+":"+filename[11:13]+":"+filename[13:15]
                    else:
                        print("Error: time unknown")
                        timedate = ""
                    s5.text = timedate

                    s6.text = filename[:-5]
                    s7.text = ".\\"+os.path.split(root)[-1]
                    #s7.text = ".\\"+os.path.relpath(root, dirName)
                    s8.text = str(0)

            # Now write the file 
            print("writing to", os.path.join(root, savefile))
            with open(os.path.join(root, savefile), "wb") as f:
                f.write(etree.tostring(etree.ElementTree(start), pretty_print=True, xml_declaration=True, encoding='utf-8'))
    return 1

def exportToBatSearch_2(self,dirName,savefile='BatData.xml',threshold1=0.85,threshold2=0.7):
    """DEPRECATED: Use exportBatResults(dirName, format='xml') instead."""
    # Write out a file that can be used for BatSearch import
    # For now, looks like the xml file used there
    # Assumes that dirName is a survey folder and the structure beneath is something like Rx/Bat/Date
    # TODO: No error checking!
    # TODO: Use xml properly
    # TODO: Check date
    import os
    from src.core import Annotation
    
    operator = "AviaNZ 3.0"
    site = "Nowhere"
    # BatSeach codes
    namedict = {"Unassigned":0, "Non-bat":1, "Unknown":2, "Long Tail":3, "Short Tail":4, "Possible LT":5, "Possible ST":6, "Both":7}
    # File header
    start = "<?xml version=\"1.0\"?>\n<ArrayOfBatRecording xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\">"
    output = start
    if not os.path.isdir(dirName):
        print("Folder doesn't exist")
        return 0
    for root, dirs, files in os.walk(dirName, topdown=True):
        nfiles = len(files)
        if nfiles > 0:
            for count in range(nfiles):
                filename = files[count]
                if filename.endswith('.data'):
                    segments = Annotation.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
                    # TODO:Should be able to remove this...
                    label = 'Non-bat'
                    if len(segments)>0:
                        seg = segments[0]
                        print(seg)
                        c = [lab["certainty"] for lab in seg[4]]
                        s = [lab["species"] for lab in seg[4]]
                        if len(c)>1:
                            label = 'Both'
                        else:
                            if c[0]>=threshold1:
                                if s[0] == 'Long-tailed bat':
                                    label = 'Long Tail'
                                elif s[0] == 'Short-tailed bat':
                                    label = 'Short Tail'
                            elif threshold2 is not None:
                                if c[0]>threshold2:
                                    if s[0] == 'Long-tailed bat':
                                        label = 'Possible LT'
                                    elif s[0] == 'Short-tailed bat':
                                        label = 'Possible ST'
                            elif threshold2 is None:
                                if s[0] == 'Long-tailed bat':
                                    label = 'Possible LT'
                                elif s[0] == 'Short-tailed bat':
                                    label = 'Possible ST'
                            else:
                                label = 'Non-bat'
                    else:
                        # TODO: which?
                        label = 'Non-bat'
                        #label = 'Unassigned'
                    # This is the text for the file
                    s1 = "<BatRecording>\n"
                    s2 = "<AssignedBatCategory>"+str(namedict[label])+"</AssignedBatCategory>\n"
                    s3 = "<AssignedSite>"+site+"</AssignedSite>\n"
                    s4 = "<AssignedUser>"+operator+"</AssignedUser>\n"
                    # DOC format -- BatSearch wants yyyy-mm-ddThh:mm:ss
                    if len(filename.split('_')[0]) == 6:
                        # ddmmyy
                        s5 = "<RecTime>"+"20"+filename[4:6]+"-"+filename[2:4]+"-"+filename[0:2]+"T"+filename[7:9]+":"+filename[9:11]+":"+filename[11:13]+"</RecTime>\n"
                    elif len(filename.split('_')[0]) == 8:
                        # yyyymmdd
                        s5 = "<RecTime>"+filename[:4]+"-"+filename[4:6]+"-"+filename[6:8]+"T"+filename[9:11]+":"+filename[11:13]+":"+filename[13:15]+"</RecTime>\n"
                    else:
                        print("Error: time unknown")
                        s5 = "<RecTime>"+"</RecTime>\n"

                    #s5 = "<RecTime>"+filename[:4]+"-"+filename[4:6]+"-"+filename[6:8]+"T"+filename[9:11]+":"+filename[11:13]+":"+filename[13:15]+"</RecTime>\n"
                    s6 = "<RecordingFileName>"+filename[:-5]+"</RecordingFileName>\n"
                    s7 = "<RecordingFolderName>.\\"+os.path.relpath(root, dirName)+"</RecordingFolderName>\n"
                    s8 = "<MeasureTimeFrom>0</MeasureTimeFrom>\n"
                    s9 = "</BatRecording>\n"
                    output+= s1+s2+s3+s4+s5+s6+s7+s8+s9
            # Now write the file if necessary
            if output != start:
                output += "</ArrayOfBatRecording>\n"
                file = open(os.path.join(root, savefile), 'w')
                print("writing to", os.path.join(root, savefile))
                file.write(output)
                file.write("\n")
                file.close()
                output = start

    return 1

def exportToBatSearch_1(self, dirName, savefile='BatData.xml'):
    """DEPRECATED: Use exportBatResults(dirName, format='xml') instead."""
    # Write out a file that can be used for BatSearch import
    # For now, looks like the xml file used there
    # Assumes that dirName is a survey folder and the structure beneath is something like Rx/Bat/Date
    # No error checking
    import os
    from src.core import Annotation
    
    operator = "AviaNZ 3.1"
    site = "Nowhere"
    # BatSeach codes
    namedict = {"Unassigned": 0, "Non-bat": 1, "Unknown": 2, "Long Tail": 3, "Short Tail": 4, "Possible LT": 5,
                "Possible ST": 6, "Both": 7}
    # File header
    start = "<?xml version=\"1.0\"?>\n<ArrayOfBatRecording xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\">"
    output = start
    if not os.path.isdir(dirName):
        print("Folder doesn't exist")
        return 0
    for root, dirs, files in os.walk(dirName, topdown=True):
        nfiles = len(files)
        if nfiles > 0:
            for count in range(nfiles):
                filename = files[count]
                if filename.endswith('.data'):
                    segments = Annotation.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
                    # TODO:Should be able to remove this...
                    label = 'Non-bat'
                    if len(segments) > 0:
                        seg = segments[0]
                        print(seg)
                        c = [lab["certainty"] for lab in seg[4]]
                        s = [lab["species"] for lab in seg[4]]
                        if c[0] == 100:
                            if s[0] == 'Long-tailed bat':
                                label = 'Long Tail'
                            elif s[0] == 'Short-tailed bat':
                                label = 'Short Tail'
                        else:
                            if s[0] == 'Long-tailed bat':
                                label = 'Possible LT'
                            elif s[0] == 'Short-tailed bat':
                                label = 'Possible ST'
                    else:
                        label = 'Non-bat'
                        # label = 'Unassigned'
                    # This is the text for the file
                    s1 = "<BatRecording>\n"
                    s2 = "<AssignedBatCategory>" + str(namedict[label]) + "</AssignedBatCategory>\n"
                    s3 = "<AssignedSite>" + site + "</AssignedSite>\n"
                    s4 = "<AssignedUser>" + operator + "</AssignedUser>\n"
                    # DOC format
                    s5 = "<RecTime>" + filename[:4] + "-" + filename[4:6] + "-" + filename[6:8] + "T" + filename[
                                                                                                        9:11] + ":" + filename[
                                                                                                                      11:13] + ":" + filename[
                                                                                                                                     13:15] + "</RecTime>\n"
                    s6 = "<RecordingFileName>" + filename[:-5] + "</RecordingFileName>\n"
                    s7 = "<RecordingFolderName>.\\" + os.path.basename(root) + "</RecordingFolderName>\n"
                    s8 = "<MeasureTimeFrom>0</MeasureTimeFrom>\n"
                    s9 = "</BatRecording>\n"
                    output += s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9
            # Now write the file if necessary
            if output != start:
                output += "</ArrayOfBatRecording>\n"
                file = open(os.path.join(root, savefile), 'w')
                print("writing to", os.path.join(root, savefile))
                file.write(output)
                file.write("\n")
                file.close()
                output = start

    return 1

def exportToBatSearchCSV(self,dirName,writefile="BatResults.csv",threshold1=0.85,threshold2=0.7):
    """DEPRECATED: Use exportBatResults(dirName, format='csv') instead."""
    # This produces a csv file that looks like the one from Bat Search. 
    import os
    from src.core import Annotation

    f = open(os.path.join(dirName,writefile),'w')
    f.write('Date,Time,AssignedSite,Category,Foldername,Filename,Observer\n')
    for root, dirs, files in os.walk(dirName):
        dirs.sort()
        files.sort()
        for filename in files:
            if filename.endswith('.data'):
                segments = Annotation.SegmentList()
                segments.parseJSON(os.path.join(root, filename))
                if len(segments)>0:
                    seg = segments[0]
                    c = [lab["certainty"] for lab in seg[4]]
                    s = [lab["species"] for lab in seg[4]]
                    if len(c)>1:
                        label = 'Both'
                    else:
                        if c[0]>threshold1:
                            if s[0] == 'Long-tailed bat':
                                label = 'Long Tail'
                            elif s[0] == 'Short-tailed bat':
                                label = 'Short Tail'
                        elif c[0]>threshold2:
                            if s[0] == 'Long-tailed bat':
                                label = 'Possible LT'
                            elif s[0] == 'Short-tailed bat':
                                label = 'Possible ST'
                        else:
                            label = '' #Non-bat'
                else:
                    label = '' #'Non-bat'
                # Assumes DOC format
                d = filename[6:8]+'/'+filename[4:6]+'/'+filename[:4]+','
                if d[0] == '0':
                    d = d[1:]
                if int(filename[9:11]) < 13:
                    if filename[9:11] == '00':
                        t = str(int(filename[9:11])+12)+':'+filename[11:13]+':'+filename[13:15]+' a.m.,'
                    else:
                        t = filename[9:11]+':'+filename[11:13]+':'+filename[13:15]+' a.m.,'
                else:
                    t = str(int(filename[9:11])-12)+':'+filename[11:13]+':'+filename[13:15]+' p.m.,'
                if t[0] == '0':
                    t = t[1:]
                # Assume that directory structure is recorder - date
                if label == '':
                    rec = ',Unassigned'
                    op = ''
                else:
                    rec = root.split('/')[-3]
                    op = 'Moira Pryde'
                date = '.\\'+root.split('/')[-1]
                f.write(d+t+rec+','+label+','+date+','+filename[:-5]+','+op+'\n')
    f.close()
    return 1
