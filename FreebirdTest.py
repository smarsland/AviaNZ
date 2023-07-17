import os, csv
#import openpyxl
import xml.etree.ElementTree as ET
import Segment

# TODO: Remove duration from dialog
# Read freebird bird list
# TODO: make it part of the setup
spName = []
spCode = []

#try:
    #print(os.path.join('/home/marslast/.avianz/', "Freebird_species_list.xlsx"))
    #book = openpyxl.load_workbook(os.path.join('/home/marslast/.avianz/', "Freebird_species_list.xlsx"))
    #sheet = book.active
#except:
    #print("Warning: Did not find Freebird species list")

#name = sheet['A2': 'A' + str(sheet.max_row)]
#code = sheet['B2': 'B' + str(sheet.max_row)]
#for i in range(len(name)):
    #spName.append(str(name[i][0].value))
#for i in range(len(code)):
    #if code[i][0].value is not None:
        #spCode.append(int(code[i][0].value))
    #else:
        #spCode.append(-1)

try:
    with open('/home/marslast/.avianz/Freebird_species_list.csv', mode='r') as f:
        cs = csv.DictReader(f)
        for l in cs:
            if l['FreebirdCode'] != '':
                spName.append(l['SpeciesName'])
                spCode.append(int(l['FreebirdCode']))

    f.close()
except:
    print("Warning: Did not find Freebird species list")

spDict = dict(zip(spCode, spName))

# Generate the .data files from .tag, read operator/reviewer from the corresponding .setting file
for root, dirs, files in os.walk('Freebird'):
    for file in files:
        if file.endswith('.tag'):
            tagFile = os.path.join(root, file)
            tagSegments = Segment.SegmentList()

            # First get the metadata
            operator = ""
            reviewer = ""
            duration = ""
            try:
                stree = ET.parse(tagFile[:-4] + '.setting')
                stroot = stree.getroot()
                for elem in stroot:
                    if elem.tag == 'Operator':
                        operator = elem.text
                    if elem.tag == 'Reviewer' and elem.text:
                        reviewer = elem.text
            except:
                print("Can't read %s.setting or missing data" %tagFile[:-4])
            try:
                # Read the duration from the sample if possible
                ptree = ET.parse(tagFile[:-4] + '.p')
                ptroot = ptree.getroot()
                for elem in ptroot:
                    for elem2 in elem:
                        if elem2.tag == 'DurationSecond':
                            duration = elem2.text
            except:
                print("Can't read %s.p or missing data" %tagFile[:-4])
                # Otherwise, load the wav file
                import SignalProc 
                sp = SignalProc.SignalProc(512,256, 0, 0)
                sp.readWav(tagFile[:-4] + '.wav', 0, 0)
                duration = sp.fileLength / sp.sampleRate

            tagSegments.metadata = {"Operator": operator, "Reviewer": reviewer, "Duration": duration}
                
            try:
                tree = ET.parse(tagFile)
                troot = tree.getroot()

                for elem in troot:
                    try:
                        print(elem)
                        print(elem[0])
                        species = [{"species": spDict[int(elem[0].text)], "certainty": 100, "filter": "M"}]
                        # TODO: Get the size right! Something weird about the freqs
                        #newSegment = Segment.Segment([float(elem[1].text), float(elem[1].text) + float(elem[2].text), 0,0, species])
                        #newSegment = Segment.Segment([float(elem[1].text), float(elem[1].text) + float(elem[2].text), float(elem[3].text), float(elem[4].text), species])
                        newSegment = Segment.Segment([float(elem[1].text), float(elem[1].text) + float(elem[2].text), 36000-float(elem[3].text), 36000-float(elem[4].text), species])
                        tagSegments.append(newSegment)
                        #print(tagSegments)
                    except KeyError:
                        print("{0} not in bird list for file {1}".format(elem[0].text,tagFile))
            except Exception as e:
                print("Can't read %s or missing data" %tagFile)
                print("Warning: Generating annotation from %s failed with error:" % (tagFile))
                print(e)

            # save .data, possible over-writing
            tagSegments.saveJSON(tagFile[:-4] + '.wav.data')
            #file = open(tagFile[:-4] + '.wav.data', 'w')
            #json.dump(annotation, file)
            #file.close()

