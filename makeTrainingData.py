
import numpy as np
import os, wavio, json
import librosa

import WaveletSegment
import WaveletFunctions
import SignalProc
import Segment

import pyqtgraph as pg
import pyqtgraph.exporters as pge

def loadFile(filename, duration=0, offset=0, fs=0, denoise=False, f1=0, f2=0):
    """
    Read audio file and preprocess as required.
    """
    if duration == 0:
        duration = None

    sp = SignalProc.SignalProc(256, 128)
    sp.readWav(filename, duration, offset)
    sp.resample(fs)
    sampleRate = sp.sampleRate
    audiodata = sp.data

    # # pre-process
    if denoise:
        WF = WaveletFunctions.WaveletFunctions(data=audiodata, wavelet='dmey2', maxLevel=10, samplerate=fs)
        audiodata = WF.waveletDenoise(thresholdType='soft', maxLevel=10)

    if f1 != 0 and f2 != 0:
        # audiodata = sp.ButterworthBandpass(audiodata, sampleRate, f1, f2)
        audiodata = sp.bandpassFilter(audiodata, sampleRate, f1, f2)

    return audiodata

dir = "/Users/marslast/Projects/AviaNZ/Sound Files/Train2/"
species = "Kiwi (Nth Is Brown)"
fs = 16000
dataset = []
if os.path.isdir(dir):
    for root, dirs, files in os.walk(str(dir)):
        for file in files:
            if file.lower().endswith('.wav') and file + '.data' in files:
                # Read the annotation
                segments = Segment.SegmentList()
                segments.parseJSON(os.path.join(root, file + '.data'))
                if species:
                    thisSpSegs = segments.getSpecies(species)
                else:
                    thisSpSegs = np.arange(len(segments)).tolist()
                # Now find syllables within each segment, median clipping
                for segix in thisSpSegs:
                    seg = segments[segix]
                    audiodata = loadFile(filename=os.path.join(root, file), duration=seg[1] - seg[0], offset=seg[0], fs=fs, denoise=False)
                    # minlen = minlen * fs
                    start = seg[0]
                    # start = int(seg[0] * fs)
                    sp = SignalProc.SignalProc(256, 128)
                    sp.data = audiodata
                    sp.sampleRate = fs
                    _ = sp.spectrogram(256, 128)
                    segment = Segment.Segmenter(sp, fs)
                    syls = segment.medianClip(thr=3, medfiltersize=5, minaxislength=9, minSegment=50)
                    if len(syls) == 0:  # Sanity check
                        segment = Segment.Segmenter(sp, fs)
                        syls = segment.medianClip(thr=2, medfiltersize=5, minaxislength=9, minSegment=50)
                    syls = segment.checkSegmentOverlap(syls)  # merge overlapped segments
                    #syls = segment.joinGaps(syls, minlen)
                    # syls = [[int(s[0] * fs) + start, int(s[1] * fs + start)] for s in syls]
                    syls = [[s[0] + start, s[1] + start] for s in syls]

                    # Sanity check, e.g. when user annotates syllables tight, median clipping may not detect it.
                    if len(syls) == 0:
                        syls = [[start, seg[1]]]
                    # if len(syls) > 1:
                    #     syls = segment.joinGaps(syls, minlen)  # Merge short segments
                    if len(syls) == 1 and syls[0][1] - syls[0][0] < minlen:  # Sanity check
                        syls = [[start, seg[1]]]
                    # syls = [[x[0] / fs, x[1] / fs] for x in syls]
                    # print('\nCurrent:', seg, '--> Median clipping ', syls)
                    for syl in syls:
                        dataset.append([os.path.join(root, file), seg, syl])

# Read the syllables and generate features, also zero padding short syllables
featuresw = []
featuresm = []
featuresc = []
featuress = []
featuresa = []
count=0
imagewindow = pg.image()
for record in dataset:
    audiodata = loadFile(filename=record[0], duration=record[2][1] - record[2][0], offset=record[2][0],fs=fs, denoise=False, f1=0, f2=0)
    audiodata = audiodata.tolist()

    featuresa.append(audiodata)

    mfcc = librosa.feature.mfcc(y=np.asarray(audiodata), sr=fs, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc, mode='nearest')
    mfcc = np.concatenate((mfcc, mfcc_delta), axis=0)
    mfcc = [i for sublist in mfcc for i in sublist]
    featuresm.append(mfcc)

    ws = WaveletSegment.WaveletSegment(spInfo={})
    we = ws.computeWaveletEnergy(data=audiodata, sampleRate=fs, nlevels=5, wpmode='new')
    featuresw.append(we.tolist())

    chroma = librosa.feature.chroma_cqt(y=np.asarray(audiodata), sr=fs)
    # chroma = librosa.feature.chroma_stft(y=data, sr=fs)
    featuresc.append(chroma.tolist())

    # Sgram images
    sp.data = audiodata
    sp.sampleRate = fs
    sgRaw = sp.spectrogram(256,128)
    featuress.append(sgRaw.tolist())
    maxsg = np.min(sgRaw)
    sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))

    img = pg.ImageItem(sg)
    imagewindow.clear()
    imagewindow.setImage(sg)
    exporter = pge.ImageExporter(imagewindow.view)
    exporter.export(species+"%03d"%count+'.png')
    count+=1

with open('waveletdata.json', 'w') as outfile:
    json.dump(featuresw, outfile)
with open('mfccdata.json', 'w') as outfile:
    json.dump(featuresm, outfile)
with open('chromadata.json', 'w') as outfile:
    json.dump(featuresc, outfile)
with open('sgramdata.json', 'w') as outfile:
    json.dump(featuress, outfile)
with open('audiodata.json', 'w') as outfile:
    json.dump(featuresa, outfile)

