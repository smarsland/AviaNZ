
import importlib

import os

import Clustering
cluster = Clustering.Clustering([], [], 0)
s, c = cluster.getSyllables('/home/marslast/Projects/AviaNZ/Sound Files/TrainX','Kiwi (Nth Is Brown)',16000)

trainDir = '/home/marslast/Projects/AviaNZ/Sound Files/TrainX'
listOfDataFiles = []
listOfSoundFiles = []
for root, dirs, files in os.walk(trainDir):
    for file in files:
        if file.lower().endswith('.data'):
            listOfDataFiles.append(os.path.join(root, file))
        elif file.lower().endswith('.wav') or file.lower().endswith('.flac'):
            listOfSoundFiles.append(os.path.join(root, file))

print(listOfDataFiles)
print(listOfSoundFiles)

species = 'Kiwi (Nth Is Brown)'
import Segment
for file in listOfDataFiles:
    if file[:-5] in listOfSoundFiles:
        segments = Segment.SegmentList()
        segments.parseJSON(os.path.join(trainDir, file))
        soundfile = os.path.join(trainDir, file[:-5])
        SpSegs = segments.getSpecies(species)
	print(SpSegs)

