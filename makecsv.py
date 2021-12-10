
#dd/mm/yyyy,(h)h:mm:ss a.m.,R?,Long tail,.\20191110,.\file.bmp,Moira Pryde
import os
import Segment

def exportCSV(dirName):
    #from PyQt5.QtCore import QTime

    # list all DATA files that can be processed
    writefile = "Results.csv"
    f = open(os.path.join(dirName,writefile),'w')
    f.write('Date,Time,AssignedSite,Category,Foldername,Filename,Observer\n')
    for root, dirs, files in os.walk(dirName):
        dirs.sort()
        files.sort()
        for filename in files:
            if filename.endswith('.data'):
                #print("Appending" ,filename)
                segments = Segment.SegmentList()
                segments.parseJSON(os.path.join(root, filename))
                if len(segments)>0:
                    seg = segments[0]
                    c = [lab["certainty"] for lab in seg[4]]
                    if c[0]==100:
                        s = [lab["species"] for lab in seg[4]]
                        # TODO: what if both?
                        if s[0] == 'Long-tailed bat':
                            s = 'Long tail,'
                        elif s[0] == 'Short-tailed bat':
                            s = 'Short tail,'
                    else:
                        s = ''
                else:
                    s = ''
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
                if s == '':
                    rec = ',Unassigned'
                    op = ''
                else:
                    rec = root.split('/')[-3]
                    op = 'Moira Pryde'
                date = '.\\'+root.split('/')[-1]
                #dd/mm/yyyy,(h)h:mm:ss a.m.,R?,Long tail,.\20191110,.\file.bmp,Moira Pryde
                f.write(d+t+rec+','+s+date+','+'.\\'+filename[:-5]+','+op+'\n')

    f.close()

            # Print the filename
            #ws.cell(row=r, column=1, value=segsl.filename)

            # Time limits
            #ws.cell(row=r, column=2, value=str(QTime(0,0,0).addSecs(seg[0]+startTime).toString('hh:mm:ss')))
            #ws.cell(row=r, column=3, value=str(QTime(0,0,0).addSecs(seg[1]+startTime).toString('hh:mm:ss')))
            # print species and certainty
            #text = [lab["species"] for lab in seg[4]]
            #ws.cell(row=r, column=6, value=", ".join(text))
            #text = [str(lab["certainty"]) for lab in seg[4]]
            #ws.cell(row=r, column=7, value=", ".join(text))

for i in range(1,22):
    dirName = "/home/marslast/Dropbox/BatResults/R"+str(i)
    exportCSV(dirName)
