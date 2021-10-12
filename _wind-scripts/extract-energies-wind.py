# Small script for extracting energies or other measures from wav files
import sys
# set your path to AviaNZ folder here:
sys.path.append('/home/julius/Documents/gitrep/birdscape')

# set the directory containing the pilotdata set here:
# (.energies files will also be exported there)
DIR = '/home/julius/Documents/kiwis/wind/deposited/pilotdata/'

import wavio
import os
import numpy as np
import re
import librosa
import WaveletFunctions


def loadData(file, len, off):
    data = wavio.read(file, len, off)
    samplerate = data.rate
    data = data.data
    data = data[:,0].astype("float")

    # downsample
    data = librosa.core.audio.resample(data, samplerate, 16000)
    print("File %s loaded" % file)
    return(data)


def rawEnergies(data, filename, nodes, wv='dmey2'):
    WF = WaveletFunctions.WaveletFunctions(data=data, wavelet=wv, maxLevel=5, samplerate=16000)
    WF.WaveletPacket(nodes, 'symmetric', False)
    n0, realwin = WF.extractE(1, 0.1)

    datalen = len(n0)
    E = np.zeros((datalen, len(nodes)))
    i = 0
    for node in nodes:
        C, _ = WF.extractE(node, 0.1)
        E[:,i] = C[:datalen]
        i += 1
        print("node %d extracted" % node)

    out = os.path.join(filename+"-"+wv+".energies")
    print("Saving to", out)
    print(np.shape(E))
    np.savetxt(out, E, delimiter="\t")

for root, dirs, files in os.walk(DIR):
    for file in files:
        if re.search("_2300.*wav$", file):
            print("working on" + file)
            ff = os.path.join(root, file)
            wav = loadData(ff, 1*60, 0*60)  # file, len, off
            rawEnergies(wav, ff, range(1,63), 'dmey2')
            rawEnergies(wav, ff, range(1,63), 'sym8')
