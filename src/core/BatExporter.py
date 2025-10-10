# BatExporter.py
#
# Bat detection results export functionality

# Version 4.0 10/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2024

import os
import fnmatch
import numpy as np

from src.core import Segment
from src.core import Spectrogram

class BatExporter:
    """Handles export of bat detection results to various formats"""
    
    def __init__(self, config, bat_detector=None):
        """
        Initialize bat exporter.
        
        Args:
            config: Configuration dictionary
            bat_detector: Optional BatDetector instance for click search
        """
        self.config = config
        self.bat_detector = bat_detector
        self.sp = None
    
    def exportResults(self, dirName, format='xml', savefile=None, threshold1=0.85, threshold2=0.7):
        """
        Unified bat export method supporting multiple output formats.
        
        Args:
            dirName: Directory containing bat detection results
            format: Output format - 'xml' (BatSearch), 'csv' (BatSearch CSV), or 'passes' (bat passes)
            savefile: Output filename (defaults based on format if None)
            threshold1: Primary certainty threshold
            threshold2: Secondary certainty threshold (can be None)
            
        Returns:
            1 on success, 0 on failure
        """
        if not os.path.isdir(dirName):
            print("Folder doesn't exist")
            return 0
        
        if savefile is None:
            if format == 'xml':
                savefile = 'BatData.xml'
            elif format == 'csv':
                savefile = 'BatResults.csv'
            elif format == 'passes':
                savefile = 'BatPasses.csv'
            else:
                print(f"Unknown format: {format}")
                return 0
        
        operator = "AviaNZ 3.4"
        site = "Nowhere"
        namedict = {"Unassigned":0, "Non-bat":1, "Unknown":2, "Long Tail":3, "Short Tail":4, 
                    "Possible LT":5, "Possible ST":6, "Both":7}
        
        if format == 'xml':
            return self.exportXML(dirName, savefile, threshold1, threshold2, operator, site, namedict)
        elif format == 'csv':
            return self.exportCSV(dirName, savefile, threshold1, threshold2, operator)
        elif format == 'passes':
            return self.exportPasses(dirName, savefile)
        else:
            print(f"Unknown format: {format}")
            return 0
    
    def exportXML(self, dirName, savefile, threshold1, threshold2, operator, site, namedict):
        """Export bat results to BatSearch XML format."""
        from lxml import etree
        
        for root, dirs, files in os.walk(dirName, topdown=True):
            if any(fnmatch.fnmatch(filename, '*.bmp') for filename in files):
                start = etree.Element("ArrayOfBatRecording", 
                                     nsmap={'xsi': "http://www.w3.org/2001/XMLSchema-instance", 
                                           'xsd': "http://www.w3.org/2001/XMLSchema"})
                
                for filename in files:
                    if filename.endswith('.data'):
                        s1 = etree.SubElement(start, "BatRecording")
                        segments = Segment.SegmentList()
                        segments.parseJSON(os.path.join(root, filename))
                        
                        label = self.getBatLabel(segments, threshold1, threshold2)
                        
                        etree.SubElement(s1, "AssignedBatCategory").text = str(namedict[label])
                        etree.SubElement(s1, "AssignedSite").text = site
                        etree.SubElement(s1, "AssignedUser").text = operator
                        etree.SubElement(s1, "RecTime").text = self.parseTimeDate(filename)
                        etree.SubElement(s1, "RecordingFileName").text = filename[:-5]
                        etree.SubElement(s1, "RecordingFolderName").text = ".\\" + os.path.split(root)[-1]
                        etree.SubElement(s1, "MeasureTimeFrom").text = str(0)
                
                print("writing to", os.path.join(root, savefile))
                with open(os.path.join(root, savefile), "wb") as f:
                    f.write(etree.tostring(etree.ElementTree(start), pretty_print=True, 
                                         xml_declaration=True, encoding='utf-8'))
        return 1
    
    def exportCSV(self, dirName, savefile, threshold1, threshold2, operator):
        """Export bat results to BatSearch CSV format."""
        f = open(os.path.join(dirName, savefile), 'w')
        f.write('Date,Time,AssignedSite,Category,Foldername,Filename,Observer\n')
        
        for root, dirs, files in os.walk(dirName):
            dirs.sort()
            files.sort()
            for filename in files:
                if filename.endswith('.data'):
                    segments = Segment.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
                    
                    label = self.getBatLabel(segments, threshold1, threshold2)
                    if label == 'Non-bat':
                        label = ''
                    
                    d = filename[6:8] + '/' + filename[4:6] + '/' + filename[:4] + ','
                    if d[0] == '0':
                        d = d[1:]
                    
                    if int(filename[9:11]) < 13:
                        if filename[9:11] == '00':
                            t = str(int(filename[9:11]) + 12) + ':' + filename[11:13] + ':' + filename[13:15] + ' a.m.,'
                        else:
                            t = filename[9:11] + ':' + filename[11:13] + ':' + filename[13:15] + ' a.m.,'
                    else:
                        t = str(int(filename[9:11]) - 12) + ':' + filename[11:13] + ':' + filename[13:15] + ' p.m.,'
                    if t[0] == '0':
                        t = t[1:]
                    
                    rec = root.split('/')[-3] if label != '' else ''
                    date = '.\\' + root.split('/')[-1]
                    op = operator if label != '' else ''
                    
                    f.write(d + t + ',' + label + ',' + date + ',' + filename[:-5] + ',' + op + '\n')
        
        f.close()
        return 1
    
    def exportPasses(self, dirName, savefile):
        """Export bat passes summary."""
        if self.sp is None:
            self.sp = Spectrogram.Spectrogram(self.config['window_width'], self.config['incr'])
        
        f = open(os.path.join(dirName, savefile), 'w')
        f.write("Tally,Night,Site,Detector,Detector Name,Bat species (L or S), Time of bat pass (24 hour clock e.g. 23:41:11),Length of bat pass (s),Feeding buzz present (yes/no)\n")
        
        tally = 0
        
        for root, dirs, files in os.walk(dirName, topdown=True):
            for filename in files:
                if filename.endswith('.data'):
                    segments = Segment.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
                    
                    label = 'Non-bat'
                    length = "0"
                    
                    if len(segments) > 0:
                        fn = filename[:-5]
                        self.sp.readSoundFile(os.path.join(root, fn), rotate=False, silent=True)
                        if self.bat_detector:
                            res = self.bat_detector.clickSearch(self.sp, None, virginia=False)
                            if res is not None:
                                length = "{:.2f}".format((res[1] - res[0]) * Spectrogram.BAT_SPECTROGRAM_TIME_PER_PIXEL)
                        
                        seg = segments[0]
                        c = [lab["certainty"] for lab in seg[4]]
                        s = [lab["species"] for lab in seg[4]]
                        
                        if len(c) > 1:
                            label = 'Both'
                        elif c[0] > 50:
                            if s[0] == 'Long-tailed bat':
                                label = 'L'
                            elif s[0] == 'Short-tailed bat':
                                label = 'S'
                    
                    night = root[-2:] + "/" + root[-4:-2] + "/" + root[-6:-4]
                    folder = root.split("/")[-2]
                    detname = ""
                    time = filename[9:11] + ":" + filename[11:13] + ":" + filename[13:15]
                    
                    f.write(f"{tally},{night},,,{detname},{label},{time},{length},\n")
                    tally += 1
        
        f.close()
        return 1
    
    def exportSurvey(self, dirName, responses, threshold1=0.85):
        """Export an excel file for the Bat survey database"""
        if responses is None:
            responses = ['', self.config['operator'], '', 'ABM', '', '', '', '', '']

        dates = []
        for root, dirs, files in os.walk(dirName):
            for d in dirs:
                if d.isdigit():
                    dates.append(d)

        if len(dates) == 0:
            print("ERROR: no suitable folders found")
            return 0
        else:
            print("Dates:", dates)

        dates = np.array(dates)
        dates = np.unique(dates)
        dates = np.sort(dates)

        dates_formatted = []
        for d in dates:
            import datetime as dt
            d_f = dt.datetime.strptime(d, '%Y%m%d').date()
            dates_formatted.append(d_f)

        if len(dates_formatted) == 0:
            print("ERROR: none of the directory names were date-like")
            return 0

        start = dates_formatted[0]
        end = dates_formatted[-1]
        totalnights = len(dates_formatted)

        species = np.zeros(2, dtype=int)

        for root, dirs, files in os.walk(dirName, topdown=True):
            for filename in files:
                if filename.endswith('.data'):
                    segments = Segment.SegmentList()
                    segments.parseJSON(os.path.join(root, filename))
                    label = self.getBatLabel(segments, threshold1, None)
                    if label == 'Long Tail' or label == 'Possible LT':
                        species[0] += 1
                    elif label == 'Short Tail' or label == 'Possible ST':
                        species[1] += 1
                    elif label == 'Both':
                        species[0] += 1
                        species[1] += 1

        f = open(os.path.join(dirName, 'BatDB.csv'), 'w')
        f.write('Data Source,Observer,Survey method,Species,Passes,Date,Detector type,Date recorder put out,Date recorder collected,No. of nights out,Effective nights out,Notes,Eastings,Northings,Site name,Region\n')

        line = responses[0] + ',' + responses[1] + ',' + responses[2] + ','
        if species[0] > 0 and species[1] > 0:
            line = line + 'Both species detected' + ',' + str(species[0] + species[1]) + ','
        elif species[0] > 0:
            line = line + 'Chalinolobus tuberculatus' + ',' + str(species[0]) + ','
        elif species[1] > 0:
            line = line + 'Mystacina tuberculata' + ',' + str(species[1]) + ','
        else:
            line = line + 'No bat species detected' + ',' + '0' + ','
        line = line + str(start) + ',' + responses[3] + ',' + str(start) + ',' + str(end) + ',' + str(totalnights) + ',' + str(totalnights) + ',' + responses[4] + ',' + responses[5] + ',' + responses[6] + ',' + responses[7] + ',' + responses[8] + '\n'
        f.write(line)
        f.close()
        
        return 1
    
    def getBatLabel(self, segments, threshold1, threshold2):
        """Helper method to determine bat label from segments."""
        if len(segments) == 0:
            return 'Non-bat'
        
        seg = segments[0]
        c = [lab["certainty"] for lab in seg[4]]
        s = [lab["species"] for lab in seg[4]]
        
        if len(c) > 1:
            return 'Both'
        
        if c[0] >= threshold1:
            if s[0] == 'Long-tailed bat':
                return 'Long Tail'
            elif s[0] == 'Short-tailed bat':
                return 'Short Tail'
        elif threshold2 is not None and c[0] > threshold2:
            if s[0] == 'Long-tailed bat':
                return 'Possible LT'
            elif s[0] == 'Short-tailed bat':
                return 'Possible ST'
        
        return 'Non-bat'
    
    def parseTimeDate(self, filename):
        """Helper method to parse time/date from filename for BatSearch format."""
        if len(filename.split('_')[0]) == 6:
            return "20" + filename[4:6] + "-" + filename[2:4] + "-" + filename[0:2] + "T" + \
                   filename[7:9] + ":" + filename[9:11] + ":" + filename[11:13]
        elif len(filename.split('_')[0]) == 8:
            return filename[:4] + "-" + filename[4:6] + "-" + filename[6:8] + "T" + \
                   filename[9:11] + ":" + filename[11:13] + ":" + filename[13:15]
        else:
            print("Error: time unknown")
            return ""
