import os,json, Segment
from fnmatch import fnmatch

def exportToBatSearch(dirName,savefile='BatData.xml',threshold1=0.85,threshold2=0.7):
    # Write out a BatData.xml that can be used for BatSearch import
    # The format of Bat searches is <Survey> / <Site> / Bat / <Date> / files ----- the word Bat is fixed
    # The BatData.xml goes in the Date folder
    # TODO: No error checking!
    # TODO: Check date
    from lxml import etree 

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
        if any(fnmatch(filename, '*.bmp') for filename in files):
            # Set up the XML start
            schema = etree.QName("http://www.w3.org/2001/XMLSchema-instance", "schema")
            start = etree.Element("ArrayOfBatRecording", nsmap={'xsi': "http://www.w3.org/2001/XMLSchema-instance", 'xsd':"http://www.w3.org/2001/XMLSchema"})

        #if nfiles > 0:
            for filename in files:
            #for count in range(nfiles):
                #filename = files[count]
                if filename.endswith('.data'):
                    s1 = etree.SubElement(start,"BatRecording")
                    segments = Segment.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
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
                    print(s7)
                    #s7.text = ".\\"+os.path.relpath(root, dirName)
                    s8.text = str(0)

            # Now write the file 
            print("writing to", os.path.join(root, savefile))
            with open(os.path.join(root, savefile), "wb") as f:
                f.write(etree.tostring(etree.ElementTree(start), pretty_print=True, xml_declaration=True, encoding='utf-8'))
    return 1

#exportToBatSearch('/home/marslast/Temp/R1/',savefile='BatData.xml',threshold1=0.85,threshold2=0.7)
exportToBatSearch('/home/marslast/Downloads/Bat/',savefile='BatData.xml',threshold1=0.85,threshold2=0.7)
