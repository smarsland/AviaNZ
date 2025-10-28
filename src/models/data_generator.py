# Version 3.5 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2025

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# PyTorch-based data generation and preparation for training

from torch.utils.data import Dataset
from skimage.transform import resize
import os
import numpy as np
import math
import soundfile as sf

from src.core import annotation
from src.core import wavelet_segment
from src.core import segmentation
from src.core import spectrogram


class GenerateData:
    """ This class implements NN data preparation. There are different ways:
    1. when manually annotated recordings are presented (.wav and GT.data along with call type info). In this case run
    the existing recogniser (WF) over the data set and get the diff to find FP segments (Noise class). And .data has TP/
    call type segments
    2. auto processed and batch reviewed data (.wav, .data, .correction). .data has TP/call type segments while
    .correction has segments for the noise class
    3. when extracted pieces of sounds (of call types and noise) are presented TODO
    """
    def __init__(self, filter, length, windowwidth, inc, imageheight, imagewidth, f1, f2):
        self.filter = filter
        self.species = filter["species"]
        ind = self.species.find('>')
        if ind != -1:
            self.species = self.species.replace('>', '(')
            self.species = self.species + ')'
        self.calltypes = []
        for fi in filter['Filters']:
            self.calltypes.append(fi['calltype'])
        self.fs = filter["SampleRate"]
        self.f1 = f1
        self.f2 = f2
        self.length = length
        self.windowwidth = windowwidth
        self.inc = inc
        self.imageheight = imageheight
        self.imagewidth = imagewidth

    def findCTsegments(self, dirName, calltypei):
        """ dirName got reviewed.data or manual.data
            Find calltype segments
            :returns ct segments [[filename, seg, label], ...]
        """

        calltypeSegments = []
        for root, dirs, files in os.walk(dirName):
            for file in files:
                soundFile = os.path.join(root, file)
                if (file.lower().endswith('.wav') or file.lower().endswith('.flac')) and file + '.data' in files:
                    segments = annotation.SegmentList()
                    segments.parseJSON(soundFile + '.data')
                    if len(self.calltypes) == 1:
                        ctSegments = segments.getSpecies(self.species)
                    else:
                        ctSegments = segments.getCalltype(self.species, self.calltypes[calltypei])
                    for indx in ctSegments:
                        seg = segments[indx]
                        cert = [lab["certainty"] if lab["species"] == self.species else 100 for lab in seg.labels]
                        if cert:
                            mincert = min(cert)
                            if mincert == 100:
                                calltypeSegments.append([soundFile, [seg.start_time, seg.end_time], calltypei])

        return calltypeSegments

    def findNoisesegments(self, dirName):
        """ dirName got manually annotated GT.data
        Generates auto segments by running wavelet detection
        Find noise segments by diff of auto segments and GT.data
        :returns noise segments [[filename, seg, label], ...]
        """
        manSegNum = 0
        noiseSegments = []
        print('Generating GT...')
        for root, dirs, files in os.walk(dirName):
            for file in files:
                soundFile = os.path.join(root, file)
                if (file.lower().endswith('.wav') or file.lower().endswith('.flac')) and os.stat(soundFile).st_size != 0 and file + '.data' in files:
                    segments = annotation.SegmentList()
                    segments.parseJSON(soundFile + '.data')
                    sppSegments = segments.getSpecies(self.species)
                    manSegNum += len(sppSegments)

                    segments.exportGT(soundFile, self.species, resolution=1.0)
        if manSegNum == 0:
            print("ERROR: no segments for species %s found" % self.species)
            return []

        ws = wavelet_segment.WaveletSegment(self.filter, 'dmey2')
        autoSegments = ws.waveletSegment_nn(dirName, self.filter)

        print("autoSeg", autoSegments)
        for item in autoSegments:
            print(item[0])
            soundFile = item[0]
            if os.stat(soundFile).st_size != 0:
                sppSegments = []
                if os.path.isfile(soundFile + '.data'):
                    segments = annotation.SegmentList()
                    segments.parseJSON(soundFile + '.data')
                    sppSegments = [segments[i] for i in segments.getSpecies(self.species)]
                for segAuto in item[1]:
                    overlappedwithGT = False
                    for segGT in sppSegments:
                        if self.Overlap(segGT, segAuto):
                            overlappedwithGT = True
                            break
                    if not overlappedwithGT:
                        noiseSegments.append([soundFile, segAuto, len(self.calltypes)])
        return noiseSegments

    def findAllsegments(self, dirName):
        """ dirName got manually annotated GT.data
        Generates noise segments as the complement to GT segments
        (i.e. every not marked second is used as noise)
        :returns noise segments [[filename, seg, label], ...]
        """
        manSegNum = 0
        noiseSegments = []
        segmenter = segmentation.Segmenter()
        print('Generating GT...')
        for root, dirs, files in os.walk(dirName):
            for file in files:
                soundFile = os.path.join(root, file)
                if (file.lower().endswith('.wav') or file.lower().endswith('.flac')) and os.stat(soundFile).st_size != 0 and file + '.data' in files:
                    segments = annotation.SegmentList()
                    segments.parseJSON(soundFile + '.data')
                    sppSegments = segments.getSpecies(self.species)
                    manSegNum += len(sppSegments)

                    segments.exportGT(soundFile, self.species, resolution=1.0)

                    print('Determining noise...')
                    autoseg = []
                    for sec in range(math.floor(segments.metadata["Duration"])-1):
                        if not any([sec >= seg.start_time and sec <= seg.end_time for seg in segments]):
                            autoseg.append([sec, sec+1])
                    autoSegments = segmenter.joinGaps(autoseg, maxgap=0)

                    print("autoSeg, file", soundFile, autoSegments)
                    for segAuto in autoSegments:
                        noiseSegments.append([soundFile, segAuto, len(self.calltypes)])

        if manSegNum == 0:
            print("ERROR: no segments for species %s found" % self.species)
            return []

        return noiseSegments

    def Overlap(self, segGT, seg):
        return seg[0]<=segGT[1] and seg[1]>=segGT[0]

    def getImgCount(self, dirName, dataset, hop):
        """
        Read the segment library and estimate the number of NN images per class
        :param dataset: segments in the form of [[file, [segment], label], ..]
        :param hop: list of hops for different classes
        :return: a list
        """
        dhop = hop
        eps = 0.0005
        N = [0 for i in range(len(self.calltypes) + 1)]

        for record in dataset:
            duration = record[1][1] - record[1][0]
            hop = dhop[record[-1]]
            if duration < self.length:
                try:
                    info = sf.info(record[0])
                except Exception as e:
                    print(f"ERROR: Could not read audio file {record[0]}: {e}")
                    print(f"Skipping record from file {record[0]}")
                    continue
                sample_rate = info.samplerate
                fileduration = info.frames / sample_rate
                
                record[1][0] = record[1][0] - (self.length - duration)/2 - eps
                record[1][1] = record[1][1] + (self.length - duration)/2 + eps
                if record[1][0] < 0:
                    record[1][0] = 0
                    record[1][1] = self.length + eps
                elif record[1][1] > fileduration:
                    record[1][1] = fileduration
                    record[1][0] = fileduration - duration - eps
                if 0 <= record[1][0] and record[1][1] <= fileduration:
                    n = 1
                else:
                    n = 0
            else:
                n = math.ceil((record[1][1] - record[1][0] - self.length) / hop + 1)
            N[record[-1]] += n

        return N

    def generateFeatures(self, dirName, dataset, hop, specFrameSize, verbose=False):
        """
        Read the segment library and generate features, training.
        Similar to SignalProc.generateFeaturesNN, except this one saves images
            to disk instead of returning them.
        :param dataset: segments in the form of [[file, [segment], label], ..]
        :param hop:
        :param specFrameSize: size of the spectrogram frame. We can't just use the window width, because that has been rounded to an integer,
            and we want the final image to be a set width. 
        :return: save the preferred features into JSON files + save images. Currently the spectrogram images.
        """
        count = 0
        dhop = hop
        eps = 0.0005
        N = [0 for i in range(len(self.calltypes) + 1)]
        sp = spectrogram.Spectrogram(self.windowwidth, self.inc)

        for record in dataset:
            duration = record[1][1] - record[1][0]
            hop = dhop[record[-1]]
            if duration < self.length:
                try:
                    info = sf.info(record[0])
                except Exception as e:
                    print(f"ERROR: Could not read audio file {record[0]}: {e}")
                    print(f"Skipping record from file {record[0]}")
                    continue
                sample_rate = info.samplerate
                fileduration = info.frames / sample_rate

                record[1][0] = record[1][0] - (self.length - duration) / 2 - eps
                record[1][1] = record[1][1] + (self.length - duration) / 2 + eps
                if record[1][0] < 0:
                    record[1][0] = 0
                    record[1][1] = self.length + eps
                elif record[1][1] > fileduration:
                    record[1][1] = fileduration
                    record[1][0] = fileduration - self.length - eps
                if record[1][0] <= 0 and record[1][1] <= fileduration:
                    n = 1
                    hop = self.length
                    duration = self.length + eps
                else:
                    continue
            else:
                n = math.ceil((record[1][1]-record[1][0]-self.length) / hop + 1)
            print('* hop:', hop, 'n:', n, 'label:', record[-1])

            try:
                sp.readSoundFile(record[0], duration=duration, off=record[1][0])
                sp.resample(self.fs)
                sgRaw = sp.spectrogram()
            except Exception as e:
                print("Warning: failed to load audio because:", e)
                continue

            N[record[-1]] += n

            bin_width = self.fs / 2 / np.shape(sgRaw)[1]
            lb = int(np.ceil(self.f1 / bin_width))
            ub = int(np.floor(self.f2 / bin_width))
            sgRaw[:, 0:lb] = 0.0
            sgRaw[:, ub:] = 0.0

            for i in range(int(n)):
                if verbose:
                    print('**', record[0], self.length, record[1][0]+hop*i, self.fs, '**')
                
                sgRaw_i, success = sp.extractSpectrogramFrame(
                    sgRaw, i, hop, specFrameSize, self.fs, adjust_last=(i == int(n) - 1)
                )
                
                if not success:
                    print("Warning: skipping incomplete frame", i, "for", record[0])
                    continue

                np.save(os.path.join(dirName, str(record[-1]),
                        str(record[-1]) + '_' + "%06d" % count + '_' + record[0].split(os.sep)[-1].rsplit('.', 1)[0] + '.npy'),
                        sgRaw_i)
                count += 1

        print('\n\nCompleted feature extraction')
        return N


class CustomGenerator(Dataset):

    def __init__(self, image_filenames, labels, batch_size, traindir, imghight, imgwidth, channels):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.train_dir = traindir
        self.imgheight = imghight
        self.imgwidth = imgwidth
        self.channels = channels

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int64)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        return np.array([resize(np.load(file_name), (self.imgheight, self.imgwidth, self.channels)) for file_name in batch_x]), np.array(batch_y)
