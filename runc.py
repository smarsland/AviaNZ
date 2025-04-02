
import importlib

import os

import Clustering
cluster = Clustering.Clustering([], [], 0)
s, c = cluster.getSyllables('/home/marslast/Projects/AviaNZ/Sound Files/TrainX','Kiwi (Nth Is Brown)',16000)

trainDir = '/home/marslast/Projects/AviaNZ/Sound Files/TrainX'
listOfDataFiles = []
listOfWavFiles = []
for root, dirs, files in os.walk(trainDir):
    for file in files:
        if file[-5:].lower() == '.data':
            listOfDataFiles.append(os.path.join(root, file))
        elif file[-4:].lower() == '.wav':
            listOfWavFiles.append(os.path.join(root, file))

print(listOfDataFiles)
print(listOfWavFiles)

species = 'Kiwi (Nth Is Brown)'
import Segment
for file in listOfDataFiles:
    if file[:-5] in listOfWavFiles:
        segments = Segment.SegmentList()
        segments.parseJSON(os.path.join(trainDir, file))
        wavfile = os.path.join(trainDir, file[:-5])
        SpSegs = segments.getSpecies(species)
	print(SpSegs)

