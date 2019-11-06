
import numpy as np
import os, json
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

dir = "E:\ClusterData\BrownKiwi\Train"
species = None
fs = 16000
dataset = []
if os.path.isdir(dir):
    for root, dirs, files in os.walk(str(dir)):
        for file in files:
            if file.endswith('.wav') and file + '.data' in files:
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

                    # Find the GT label for the syllables from this segment
                    # 0 - brown kiwi, male
                    # 1 - brown kiwi, female
                    # 2 - LSK, male
                    # 3 - LSK, female
                    # 4 - morepork, more-pork
                    # 5 - morepork, trill
                    # 6 - morepork, weow
                    # 7 - rooster
                    # 8 - squeeck
                    if isinstance(seg[4][0], dict):
                        # new format
                        if 'Kiwi(M)' == seg[4][0]["species"] or 'Kiwi(M)1' == seg[4][0]["species"] or 'Kiwi(M)2' == seg[4][0]["species"] or 'Kiwi(M)3' == seg[4][0]["species"] or 'Kiwi(M)4' == seg[4][0]["species"]:
                            label = 0
                        elif 'Kiwi(F)' == seg[4][0]["species"] or 'Kiwi(F)1' == seg[4][0]["species"] or 'Kiwi(F)2' == seg[4][0]["species"] or 'Kiwi(F)3' == seg[4][0]["species"] or 'Kiwi(F)4' == seg[4][0]["species"]:
                            label = 1
                        elif 'Lsk(M)' in seg[4][0]["species"]:
                            label = 2
                        elif 'Lsk(F)' in seg[4][0]["species"]:
                            label = 3
                        elif 'Morepork(Mp)' == seg[4][0]["species"] or 'Morepork(Mp)1' == seg[4][0]["species"] or 'Morepork(Mp)2' == seg[4][0]["species"] or 'Morepork(Mp)3' == seg[4][0]["species"] or 'Morepork(Mp)4' == seg[4][0]["species"]:
                            label = 4
                        elif 'Morepork(Tril)' == seg[4][0]["species"] or 'Morepork(Tril)1' == seg[4][0]["species"] or 'Morepork(Tril)2' == seg[4][0]["species"] or 'Morepork(Tril)3' == seg[4][0]["species"] or 'Morepork(Tril)4' == seg[4][0]["species"]:
                            label = 5
                        elif 'Morepork(Weow)' == seg[4][0]["species"] or 'Morepork(Weow)1' == seg[4][0]["species"] or 'Morepork(Weow)2' == seg[4][0]["species"] or 'Morepork(Weow)3' == seg[4][0]["species"] or 'Morepork(Weow)4' == seg[4][0]["species"]:
                            label = 6
                        elif 'Rooster' in seg[4][0]["species"]:
                            label = 7
                        else:
                            continue
                    elif isinstance(seg[4][0], str):
                        # old format
                        if 'Kiwi(M)' == seg[4][0] or 'Kiwi(M)1' == seg[4][0] or 'Kiwi(M)2' == seg[4][0] or 'Kiwi(M)3' == seg[4][0] or 'Kiwi(M)4' == seg[4][0]:
                            label = 0
                        elif 'Kiwi(F)' == seg[4][0] or 'Kiwi(F)1' == seg[4][0] or 'Kiwi(F)2' == seg[4][0] or 'Kiwi(F)3' == seg[4][0] or 'Kiwi(F)4' == seg[4][0]:
                            label = 1
                        elif 'Lsk(M)' in seg[4][0]:
                            label = 2
                        elif 'Lsk(F)' in seg[4][0]:
                            label = 3
                        elif 'Morepork(Mp)' == seg[4][0] or 'Morepork(Mp)1' == seg[4][0] or 'Morepork(Mp)2' == seg[4][0] or 'Morepork(Mp)3' == seg[4][0] or 'Morepork(Mp)4' == seg[4][0]:
                            label = 4
                        elif 'Morepork(Tril)' == seg[4][0] or 'Morepork(Tril)1' == seg[4][0] or 'Morepork(Tril)2' == seg[4][0] or 'Morepork(Tril)3' == seg[4][0] or 'Morepork(Tril)4' == seg[4][0]:
                            label = 5
                        elif 'Morepork(Weow)' == seg[4][0] or 'Morepork(Weow)1' == seg[4][0] or 'Morepork(Weow)2' == seg[4][0] or 'Morepork(Weow)3' == seg[4][0] or 'Morepork(Weow)4' == seg[4][0]:
                            label = 6
                        elif 'Rooster' in seg[4][0]:
                            label = 7
                        else:
                            continue

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
                    # if len(syls) == 1 and syls[0][1] - syls[0][0] < minlen:  # Sanity check
                    #     syls = [[start, seg[1]]]
                    # syls = [[x[0] / fs, x[1] / fs] for x in syls]
                    # print('\nCurrent:', seg, '--> Median clipping ', syls)
                    for syl in syls:
                        if syl[1]-syl[0] < 2.5:
                            dataset.append([os.path.join(root, file), seg, syl, label])

lengths = []
for record in dataset:
    lengths.append(record[2][1] - record[2][0])
print("min length:", min(lengths))
print("Max length:", max(lengths))
print("Mean:", np.mean(lengths))
print("Median:", np.median(lengths))


# Read the syllables and generate features, also zero padding short syllables
nlevels = 5
featuresw = []
n_mfcc = 40
n_bins = 32
n_chroma = 12
featuresm = []
featuresc = []
# featuresc = []
featuress = []
featuresa = []
labels = []
count = 0
imagewindow = pg.image()
for record in dataset:
    # # 1) compute features over whole syllables
    # audiodata = loadFile(filename=record[0], duration=record[2][1] - record[2][0], offset=record[2][0], fs=fs,
    #                      denoise=False, f1=0, f2=0)
    # audiodata = audiodata.tolist()
    # featuresa.append([audiodata, record[1][2], record[1][3], record[-1]])
    #
    # mfcc = librosa.feature.mfcc(y=np.asarray(audiodata), sr=fs, n_mfcc=40)
    # mfcc_delta = librosa.feature.delta(mfcc, mode='nearest')
    # mfcc = np.concatenate((mfcc, mfcc_delta), axis=0)
    # mfcc = [i for sublist in mfcc for i in sublist]
    # featuresm.append([mfcc, record[1][2], record[1][3], record[-1]])
    #
    # ws = WaveletSegment.WaveletSegment(spInfo={})
    # we = ws.computeWaveletEnergy(data=audiodata, sampleRate=fs, nlevels=5, wpmode='new',
    #                              window=record[2][1] - record[2][0], inc=record[2][1] - record[2][0],
    #                              resol=record[2][1] - record[2][0])
    # we = [i[0] for i in we]
    # featuresw.append([we, record[1][2], record[1][3], record[-1]])
    #
    # chroma = librosa.feature.chroma_cqt(y=np.asarray(audiodata), sr=fs)
    # # chroma = librosa.feature.chroma_stft(y=data, sr=fs)
    # chroma = [i for sublist in chroma for i in sublist]
    # featuresc.append([chroma, record[1][2], record[1][3], record[-1]])
    #
    # # Sgram images
    # sp.data = audiodata
    # sp.sampleRate = fs
    # sgRaw = sp.spectrogram(256,128)
    # featuress.append([sgRaw.tolist(),record[1][2], record[1][3], record[-1]])
    #
    # maxsg = np.min(sgRaw)
    # sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))
    #
    # img = pg.ImageItem(sg)
    # imagewindow.clear()
    # imagewindow.setImage(np.flip(sg, 1))
    # exporter = pge.ImageExporter(imagewindow.view)
    # exporter.export(os.path.join(dir, 'img', str(record[-1])+'_'+"%04d"%count+'.png'))
    # count+=1

    #2) compute features over each 1/4 sec, ignore tiny syllables
    win = 0.25
    duration = record[2][1] - record[2][0]
    if duration < win:
        continue    # ignore tiny syllables
    n = duration//win
    for i in range(int(n)):
        audiodata = loadFile(filename=record[0], duration=win, offset=record[2][0]+win*i, fs=fs, denoise=False, f1=0, f2=0)
        audiodata = audiodata.tolist()
        featuresa.append([audiodata, record[1][2], record[1][3], record[-1]])

        mfcc = librosa.feature.mfcc(y=np.asarray(audiodata), sr=fs, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc, mode='nearest')
        mfcc = np.concatenate((mfcc, mfcc_delta), axis=0)
        mfcc = [i for sublist in mfcc for i in sublist]
        featuresm.append([mfcc, record[1][2], record[1][3], record[-1]])

        ws = WaveletSegment.WaveletSegment(spInfo={})
        we = ws.computeWaveletEnergy(data=audiodata, sampleRate=fs, nlevels=5, wpmode='new',
                                 window=record[2][1] - record[2][0], inc=record[2][1] - record[2][0],
                                 resol=record[2][1] - record[2][0])
        we = [i[0] for i in we]
        featuresw.append([we, record[1][2], record[1][3], record[-1]])

        chroma = librosa.feature.chroma_cqt(y=np.asarray(audiodata), sr=fs)
        # chroma = librosa.feature.chroma_stft(y=data, sr=fs)
        chroma = [i for sublist in chroma for i in sublist]
        featuresc.append([chroma, record[1][2], record[1][3], record[-1]])

        # Sgram images
        sp.data = audiodata
        sp.sampleRate = fs
        sgRaw = sp.spectrogram(256, 128)
        featuress.append([sgRaw.tolist(), record[1][2], record[1][3], record[-1]])

        maxsg = np.min(sgRaw)
        sg = np.abs(np.where(sgRaw == 0, 0.0, 10.0 * np.log10(sgRaw / maxsg)))

        img = pg.ImageItem(sg)
        imagewindow.clear()
        imagewindow.setImage(np.flip(sg, 1))
        exporter = pge.ImageExporter(imagewindow.view)
        exporter.export(os.path.join(dir, 'img', str(record[-1]) + '_' + "%04d" % count + '.png'))
        count += 1

with open(os.path.join(dir, 'waveletdata.json'), 'w') as outfile:
    json.dump(featuresw, outfile)
with open(os.path.join(dir, 'mfccdata.json'), 'w') as outfile:
    json.dump(featuresm, outfile)
with open(os.path.join(dir, 'chromadata.json'), 'w') as outfile:
    json.dump(featuresc, outfile)
with open(os.path.join(dir, 'sgramdata.json'), 'w') as outfile:
    json.dump(featuress, outfile)
with open(os.path.join(dir, 'audiodata.json'), 'w') as outfile:
    json.dump(featuresa, outfile)

