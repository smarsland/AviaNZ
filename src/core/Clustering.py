
# Version 4.0 9/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2024

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

# Clustering.py
#
# Cluster segments

import numpy as np
import os
import librosa

from src.core import WaveletSegment
from src.core import WaveletFunctions
from src.core import Spectrogram
from src.core import Annotation
from src.core import Segmentation
from src.core import SignalProc
from src.core import AudioData

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering

from statistics import mode
from sklearn.metrics.pairwise import pairwise_distances

class Clustering:
    # This class implements various clustering algorithms and performance measures for the AviaNZ interface
    # Based on scikit-learn

    def __init__(self, features, labels, nclusters):
        if not features == []:
            features = StandardScaler().fit_transform(features)
        self.features = features
        self.targets = labels
        self.n_clusters = nclusters

    def custom_dist(self, x, y):
            d, _ = librosa.sequence.dtw(x, y, metric='euclidean')
            return d[d.shape[0] - 1][d.shape[1] - 1]

    # def DBscan(self, eps=0.5, min_samples=5, metric='euclidean'):
    def DBscan(self, eps=0.5, min_samples=5):
        """ Density-Based Spatial Clustering of Applications with Noise. An extension to mean shift clustering.
            Finds core samples of high density and expands clusters from them.
            Usecase: non-flat geometry, uneven cluster sizes
        """
        model = DBSCAN(metric='precomputed')
        d = pairwise_distances(self.features, self.features, metric=self.custom_dist)
        model.fit(d)

        return model

    def birch(self, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True):
        """ Builds a tree called the Characteristic Feature Tree (CFT) for the given data. The data is essentially lossy
            compressed to a set of Characteristic Feature nodes (CF Nodes).
            Usecase: large dataset, outlier removal, data reduction
        """
        model = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters,
                      compute_labels=compute_labels, copy=copy)
        model.fit(self.features)

        return model

    def agglomerativeClustering(self, n_clusters=3, distance_threshold=None, linkage='ward', affinity='euclidean',
                                compute_full_tree=False):
        """ A Hierarchical clustering using a bottom up approach: each observation starts in its own cluster, and
            clusters are successively merged together.
            Usecase: many clusters, possibly connectivity constraints, non Euclidean distances.
        """
        min_clusters = min(n_clusters, len(self.features))
        model = AgglomerativeClustering(n_clusters=min_clusters, distance_threshold=distance_threshold, linkage=linkage,
                                        metric=affinity, compute_full_tree=compute_full_tree)
        d = pairwise_distances(self.features, self.features, metric=self.custom_dist)
        model.fit(d)

        return model

    def cluster(self, dataset, dirname, fs, species, feature='we', n_mels=24, minlen=0.2, denoise=False, alg='agglomerative'):
        """
        Cluster segments during training to make sub-filters.

        This is called by training wizards (BuildRecAdvWizard), after getSyllables has run

        Given wav + annotation files, plus set of syllables
            1) make them fixed-length by padding or clipping
            2) cluster with some clustering algorithm

        :param dataset: syllables output by findSyllables
        :param dir: path to directory with wav & wav.data files
        :param fs: sample rate
        :param species: string, will train on segments containing this label
        :param feature: 'we' (wavelet energy), 'mfcc', or 'chroma'
        :param n_mels: number of mel coeff when feature='mfcc'
        :param minlen: min syllable length in secs
        :param denoise: True/False
        :param alg: clustering algorithm to use, default to agglomerative
        :return: clustered segments - a list of lists [[file1, seg1, [syl1, syl2], [features1, features2], predict], ...]
                 fs, nclasses, syllable duration (median)
        """

        self.alg = alg
        nlevels = 6
        weInds = []

        # 1. Get the frequency band and sampling frequency from annotations
        # TODO
        f1, f2 = self.getFrqRange(dirname, species, fs)
        print("Clustering using sampling rate", fs)

        print("f1",f1,"f2",f2)

        # 2. Find the lower and upper bounds (relevant to the frq range)
        if feature == 'mfcc' and f1 != 0 and f2 != 0:
            mels = librosa.core.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fs / 2, htk=False)
            ind_flow = (np.abs(mels - f1)).argmin()
            ind_fhigh = (np.abs(mels - f2)).argmin()
        elif feature == 'we' and f1 != 0 and f2 != 0:
            weInds = self.nodesInRange(nlevels, f1, f2, fs)
        else:
            print("Unknown feature type")
            return

        # 3. Clustering at syllable level, therefore find the syllables in each segment
        # TODO: The input from DialogTraining is the syllables? Is this called from elsewhere?
        # dataset = self.findSyllables(dirname, species, minlen, fs, f1, f2, denoise)
        # dataset format: [[file1, seg1, syl1], [file1, seg1, syl2], [file1, seg2, syl1],..]

        # Make syllables fixed-length (again to have same sized feature matrices) and generate features
        lengths = []
        for data in dataset:
            lengths.append(data[2][1] - data[2][0])
        duration = np.median(lengths)
        print("Setting duration to", duration)
        # duration is going to be the fixed length of a syllable, if a syllable is too long clip it
        # TODO: Could make >1 syllable instead...
        for record in dataset:
            if record[2][1] - record[2][0] > duration:
                middle = (record[2][1] + record[2][0]) / 2
                record[2][0] = middle - duration / 2
                record[2][1] = middle + duration / 2

        # 4. Read the syllables and generate features, also zero padding short syllables
        # TODO: Here...
        features = []
        for record in dataset:
            audiodata = self.loadFile(filename=record[0], duration=record[2][1] - record[2][0], offset=record[2][0], fs=fs, denoise=denoise, f1=f1, f2=f2, silent=True)
            audiodata = audiodata.tolist()
            if record[2][1] - record[2][0] < duration:
                # Zero padding both ends to have fixed duration
                gap = int((duration * fs - len(audiodata)) // 2)
                z = [0] * gap
                audiodata.extend(z)
                z.extend(audiodata)
                audiodata = z
            if feature == 'mfcc':  # MFCC
                mfcc = librosa.feature.mfcc(y=np.asarray(audiodata), sr=fs, n_mfcc=n_mels)
                if f1 != 0 and f2 != 0:
                    mfcc = mfcc[ind_flow:ind_fhigh, :]  # Limit the frequency to the fixed range [f1, f2]
                mfcc_delta = librosa.feature.delta(mfcc, mode='nearest')
                mfcc = np.concatenate((mfcc, mfcc_delta), axis=0)
                mfcc = scale(mfcc, axis=1)
                mfcc = [i for sublist in mfcc for i in sublist]
                features.append(mfcc)
                record.insert(3, mfcc)
            elif feature == 'we':  # Wavelet Energy
                ws = WaveletSegment.WaveletSegment(spInfo={})
                we = ws.computeWaveletEnergy(data=audiodata, sampleRate=fs, nlevels=nlevels, wpmode='new')
                we = we.mean(axis=1)
                if weInds:
                    we = we[weInds]
                features.append(we)
                record.insert(3, we)
            elif feature == 'chroma':
                chroma = librosa.feature.chroma_cqt(y=audiodata, sr=fs)
                chroma = scale(chroma, axis=1)
                features.append(chroma)
                record.insert(3, chroma)

        # 5. Actual clustering
        self.features = features

        model = self.trainModel()
        predicted_labels = model.labels_

        # Attach the label to each syllable
        for i in range(len(predicted_labels)):
            dataset[i].insert(4, predicted_labels[i])   # dataset format [[file1, seg1, syl1, features, predict], ...]

        clustered_dataset = []
        for record in dataset:
            if record[:2] not in clustered_dataset:
                clustered_dataset.append(record[:2])    # clustered_dataset [[file1, seg1], ...]

        labels = [[] for i in range(len(clustered_dataset))]
        for i in range(len(predicted_labels)):
            ind = clustered_dataset.index(dataset[i][:2])
            labels[ind].append(predicted_labels[i])

        # Majority voting when multiple syllables in a segment
        for i in range(len(labels)):
            try:
                labels[i] = mode(labels[i])
            except:
                labels[i] = labels[i][0]

        # Add the detected syllables
        for record in clustered_dataset:
            record.insert(2, [])
            for rec in dataset:
                if record[:2] == rec[:2]:
                    record[2].append(rec[2])

        # Add the features
        for record in clustered_dataset:
            record.insert(3, [])
            for rec in dataset:
                if record[:2] == rec[:2]:
                    record[3].append(rec[3])

        # Make the labels continuous, e.g. agglomerative may have produced 0, 2, 3, ...
        ulabels = list(set(labels))
        nclasses = len(ulabels)
        dic = []
        for i in range(nclasses):
            dic.append((ulabels[i], i))
        dic = dict(dic)

        # Update the labels
        for i in range(len(clustered_dataset)):
            clustered_dataset[i].insert(4, dic[labels[i]])

        print("clustered_dataset", clustered_dataset)
        print("nclasses", nclasses)
        print("duration", duration)

        return clustered_dataset, nclasses, duration

    def nodesInRange(self, nlevels, f1, f2, fs):
        ''' Return the indices (nodes) to keep
        '''
        allnodes = range(1, 2 ** (nlevels + 1) - 1)
        inband = []
        for i in allnodes:
            flow, fhigh = WaveletFunctions.getWCFreq(i, fs)
            if flow < f2 and fhigh > f1:
                inband.append(i-1)

        return inband

    def getFrqRange(self, dirname, species, fs):
        ''' Get the frequency band and sampling frequency from annotations
        '''
        lowlist = []
        highlist = []

        # Directory mode (from the training dialog)
        if os.path.isdir(dirname):
            for root, dirs, files in os.walk(str(dirname)):
                for file in files:
                    if (file.lower().endswith('.wav') or file.lower().endswith('.flac')) and file + '.data' in files:
                        # Read the annotation
                        segments = Annotation.SegmentList()
                        segments.parseJSON(os.path.join(root, file + '.data'))
                        # keep the right species
                        if species:
                            thisSpSegs = segments.getSpecies(species)
                        else:
                            thisSpSegs = np.arange(len(segments)).tolist()
                        for segix in thisSpSegs:
                            seg = segments[segix]
                            lowlist.append(seg.freq_low)
                            highlist.append(seg.freq_high)

        # File mode (from the main interface)
        elif os.path.isfile(dirname):
            if (dirname.lower().endswith('.wav') or dirname.lower().endswith('.flac')) and os.path.exists(dirname + '.data'):
                # Read the annotation
                segments = Annotation.SegmentList()
                segments.parseJSON(dirname + '.data')
                # keep the right species
                if species:
                    thisSpSegs = segments.getSpecies(species)
                else:
                    thisSpSegs = np.arange(len(segments)).tolist()
                for segix in thisSpSegs:
                    seg = segments[segix]
                    lowlist.append(seg.freq_low)
                    highlist.append(seg.freq_high)

            if len(thisSpSegs) < self.n_clusters:
                self.n_clusters = len(thisSpSegs)//2
                print('Setting number of clusters to ', self.n_clusters)

        # Sampling rate is coming from the first page in the wavelet training wizard
        # # Set sampling frequency based on segments and min samp. frq from the file list
        # arr = [4000, 8000, 16000, 32000, 48000]
        # pos = np.abs(arr - np.median(highlist) * 2).argmin()
        # fs = arr[pos]
        # if fs > np.min(srlist):
        #     fs = np.min(srlist)

        # Find frequency limits
        # TODO: Made fixed in order to have same sized feature matrices, can we vary this to use segment frequency limits?
        if len(lowlist) > 0:
            f1 = np.min(lowlist)
            f2 = np.median(highlist)
        else:
            f1 = 0
            f2 = fs/2

        if fs < f2 * 2 + 50:
            f2 = fs // 2 - 50

        if f2 < f1:
            f2 = np.mean(highlist)

        return f1, f2

    def findSyllables(self, dirname, species, minlen, fs, f1, f2, denoise):
        """ Find the syllables
        :param dirname: directory with the sound and annotation files OR a single wave file (having its .data)
        :param species: target species
        :param minlen: minimum length of a segment
        :param fs: sampling frequency
        :param f1: lower frequency bound
        :param f2: higher frequency bound
        :param denoise: denoise or not binary
        :return: a list of lists [[file1, seg1, syl1], [file1, seg1, syl2], [file1, seg2, syl1],..]
        """
        dataset = []
        if os.path.isdir(dirname):
            for root, dirs, files in os.walk(str(dirname)):
                for file in files:
                    if (file.lower().endswith('.wav') or file.lower().endswith('.flac')) and file + '.data' in files:
                        # Read the annotation
                        segments = Annotation.SegmentList()
                        segments.parseJSON(os.path.join(root, file + '.data'))
                        if species:
                            thisSpSegs = segments.getSpecies(species)
                        else:
                            thisSpSegs = np.arange(len(segments)).tolist()
                        # Now find syllables within each segment, median clipping
                        for segix in thisSpSegs:
                            seg = segments[segix]
                            syls = self.findSyllablesSeg(os.path.join(root, file), seg, fs, denoise, minlen)
                            for syl in syls:
                                dataset.append([os.path.join(root, file), seg, syl])
        elif os.path.isfile(dirname):
            if (dirname.lower().endswith('.wav') or dirname.lower().endswith('.flac')) and os.path.exists(dirname + '.data'):
                # Read the annotation
                segments = Annotation.SegmentList()
                segments.parseJSON(dirname + '.data')
                if species:
                    thisSpSegs = segments.getSpecies(species)
                else:
                    thisSpSegs = np.arange(len(segments)).tolist()
                # Now find syllables within each segment, median clipping
                for segix in thisSpSegs:
                    seg = segments[segix]
                    syls = self.findSyllablesSeg(dirname, seg, fs, denoise, minlen)
                    for syl in syls:
                        dataset.append([dirname, seg, syl])
        return dataset

    def findSyllablesSeg(self, filename, seg, fs=None, denoise=False, minlen=10):
        """ Find syllables in the segment using median clipping - single segment
        :return: syllables list
        """
        audiodata = self.loadFile(filename=filename, duration=seg.end_time - seg.start_time, offset=seg.start_time, fs=fs, denoise=denoise)
        start = seg.start_time
        self.sp.audio_data = AudioData.AudioData(data=audiodata, sample_rate=fs,
                                        sample_format='float32', sample_size=32, channels=1)
        _ = self.sp.spectrogram()
            
        # Show only the segment frequencies to the median clipping and avoid overlapping noise
        linear = np.linspace(0, fs / 2, int(self.sp.window_width/2))
        # check segment type to determine if upper freq bound is OK
        if seg.freq_high==0:
            print("Warning: auto-detecting freq bound for full-height segments")
            fhigh = fs//2
        else:
            fhigh = seg.freq_high
        ind_flow = (np.abs(linear - seg.freq_low)).argmin()
        ind_fhigh = (np.abs(linear - fhigh)).argmin()
        self.sp.sg = self.sp.sg[:, ind_flow:ind_fhigh]

        segmentation = Segmentation.Segmenter(self.sp, fs)

        syls = segmentation.medianClip(thr=3, medfiltersize=5, minaxislength=9, minSegment=50)
        if len(syls) == 0:  # Sanity check
            # Try again with lower threshold
            # TODO: Why reinitialise?
            segmentation = Segmentation.Segmenter(self.sp, fs)
            syls = segmentation.medianClip(thr=2, medfiltersize=5, minaxislength=9, minSegment=50)

        # Merge overlapped segments
        syls = segmentation.checkSegmentOverlap(syls)
        syls = segmentation.deleteShort(syls, minlen)
        syls = [[s[0] + start, s[1] + start] for s in syls]

        # Sanity check, e.g. when user annotates syllables tightly, median clipping may not detect it
        if len(syls) == 0:
            syls = [[start, seg.end_time]]
        if len(syls) == 1 and syls[0][1] - syls[0][0] < minlen:  
            syls = [[start, seg.end_time]]

        return syls

    def trainModel(self):
        """ Clustering model"""
        # TODO: More parameters :(
        if self.alg == 'DBSCAN':
            print('\nDBSCAN--------------------------------------')
            model = self.DBscan(eps=0.3, min_samples=3)

        elif self.alg == 'Birch':
            print('\nBirch----------------------------------------')
            if not self.n_clusters:
                model = self.birch(threshold=0.5, n_clusters=self.n_clusters)
            else:
                model = self.birch(threshold=0.88, n_clusters=None)

        if self.alg == 'agglomerative':
            print('\nAgglomerative Clustering----------------------')
            # Either set n_clusters=None and compute_full_tree=T or distance_threshold=None
            if not self.n_clusters:
                model = self.agglomerativeClustering(n_clusters=None, distance_threshold=0.5, linkage='average', affinity='precomputed')
            else:
                model = self.agglomerativeClustering(n_clusters=self.n_clusters, distance_threshold=None, linkage='average', affinity='precomputed')

        return model

    def getClusterCenter(self, cluster, fs, f1, f2, feature, duration, n_mels=24, denoise=False):
        """
        Compute cluster centre of a cluster
        :param cluster: segments of a cluster - a list of lists, each sublist represents a segment
                        [parent_audio_file, [segment], [syllables], [features], class_label]
        :param feature: 'we' or 'mfcc' or 'chroma'
        :param duration: the fixed duration of a syllable
        :return: cluster centre, an array
        """
        # Re-compute features to match with frquency range [f1, f2]
        # Find the lower and upper bounds (relevant to the frq range), when the range is given
        if feature == 'mfcc' and f1 != 0 and f2 != 0:
            mels = librosa.core.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fs / 2, htk=False)
            ind_flow = (np.abs(mels - f1)).argmin()
            ind_fhigh = (np.abs(mels - f2)).argmin()

        elif feature == 'we' and f1 != 0 and f2 != 0:
            linear = np.linspace(0, fs / 2, 62)
            ind_flow = (np.abs(linear - f1)).argmin()
            ind_fhigh = (np.abs(linear - f2)).argmin()

        fc = []
        for record in cluster:
            # Compute the features of each syllable in this segment
            for syl in record[2]:
                audiodata = self.loadFile(filename=record[0], duration=syl[1] - syl[0], offset=syl[0], fs=fs, denoise=denoise, f1=f1, f2=f2, silent=True)
                audiodata = audiodata.tolist()
                if syl[1] - syl[0] < duration:
                    # Zero padding both ends to have fixed duration
                    gap = int((duration * fs - len(audiodata)) // 2)
                    z = [0] * gap
                    audiodata.extend(z)
                    z.extend(audiodata)
                    audiodata = z
                if feature == 'mfcc':  # MFCC
                    mfcc = librosa.feature.mfcc(y=np.asarray(audiodata), sr=fs, n_mfcc=n_mels)
                    if f1 != 0 and f2 != 0:
                        mfcc = mfcc[ind_flow:ind_fhigh, :]  # Limit the frequency to the fixed range [f1, f2]
                    mfcc_delta = librosa.feature.delta(mfcc, mode='nearest')
                    mfcc = np.concatenate((mfcc, mfcc_delta), axis=0)
                    mfcc = scale(mfcc, axis=1)
                    mfcc = [i for sublist in mfcc for i in sublist]
                    fc.append(mfcc)
                elif feature == 'we':  # Wavelet Energy
                    ws = WaveletSegment.WaveletSegment(spInfo={})
                    we = ws.computeWaveletEnergy(data=audiodata, sampleRate=fs, nlevels=5, wpmode='new')
                    we = we.mean(axis=1)
                    if f1 != 0 and f2 != 0:
                        we = we[ind_flow:ind_fhigh]  # Limit the frequency to a fixed range f1, f2
                    fc.append(we)
                elif feature == 'chroma':
                    chroma = librosa.feature.chroma_cqt(y=audiodata, sr=fs)
                    chroma = scale(chroma, axis=1)
                    fc.append(chroma)
        return np.mean(fc, axis=0)

    def loadFile(self, filename, duration=0, offset=0, fs=0, denoise=False, f1=0, f2=0, silent=False):
        """
        Read audio file and preprocess as required.
        """
        if duration == 0:
            duration = None

        self.sp = Spectrogram.Spectrogram(256, 128)
        print(filename,duration,offset)
        self.sp.readSoundFile(filename, duration, offset, silent=silent)
        
        sampleRate = self.sp.audio_data.sample_rate
        audiodata = self.sp.audio_data.data

        # Pre-process
        if denoise:
            WF = WaveletFunctions.WaveletFunctions(data=audiodata, wavelet='dmey2', maxLevel=10, samplerate=fs)
            audiodata = WF.waveletDenoise(thresholdType='soft', maxLevel=10)

        if f1 != 0 and f2 != 0:
            audiodata = SignalProc.bandpassFilter(audiodata, sampleRate, f1, f2)

        return audiodata

    def class_create(self, label, syl, features, f_low, f_high, segs, single=False, dist_method='dtw'):
        """ Create a new class
        :param label: label of the new class
        :param syl: syllables
        :param features:
        :param f_low:
        :param f_high:
        :param segs:
        :param single: True if only one syllable from the segment goes to the class templates
        :return:
        """
        from scipy import signal
        dist = np.zeros((len(features), len(features)))
        shift = 0
        for i in range(len(features)):
            shift += 1
            for j in range(shift, len(features)):
                if dist_method == 'dtw':
                    d, _ = librosa.sequence.dtw(features[i], features[j], metric='euclidean')
                    dist[i, j] = d[d.shape[0] - 1][d.shape[1] - 1]
                elif dist_method == 'xcor':
                    corr = signal.correlate(features[i], features[j], mode='full')
                    dist[i, j] = np.sum(corr) / max(len(features[i]), len(features[j]))

        if np.count_nonzero(dist) > 0:
            nonzero = dist > 0
            inclass_d = np.percentile(dist[nonzero], 10)  # TODO: max? mean? a percentile?
        else:
            inclass_d = 0

        if single:
            features = [features[len(features) // 2]]  # get the features of the middle syllable

        newclass = {
            "label": label,
            "d": inclass_d,
            "syl": syl,
            "features": features,
            "f_low": f_low,
            "f_high": f_high,
            "segs": segs
        }
        return newclass

    def class_update(self, cluster, newfeatures, newf_low, newf_high, newsyl, newseg, single, dist_method='dtw'):
        """ Update an existing class
        :param cluster: the class to update
        :param newfeatures:
        :param newf_low:
        :param newf_high:
        :param newsyl:
        :param newsegs:
        :return: the updated cluster
        """
        from scipy import signal

        # Get in-class distance
        f_c = cluster["features"]  # features of the current class c

        if single:
            newfeatures = [newfeatures[len(newfeatures) // 2]]
            newsyl = [newsyl[len(newsyl) // 2]]

        for i in range(len(newfeatures)):
            f_c.append(newfeatures[i])

        dist_c = np.zeros((len(f_c), len(f_c)))  # distances to the current class c
        shift = 0
        for i in range(len(f_c)):
            shift += 1
            for j in range(shift, len(f_c)):
                if dist_method == 'dtw':
                    d, _ = librosa.sequence.dtw(f_c[i], f_c[j], metric='euclidean')
                    dist_c[i, j] = d[d.shape[0] - 1][d.shape[1] - 1]
                elif dist_method == 'xcor':
                    corr = signal.correlate(f_c[i], f_c[j], mode='full')
                    dist_c[i, j] = np.sum(corr) / max(len(f_c[i]), len(f_c[j]))

        if np.count_nonzero(dist_c) > 0:
            nonzero = dist_c > 0
            inclass_d = np.percentile(dist_c[nonzero], 10)  # TODO: max? mean? a percentile?
        else:
            inclass_d = 0

        for s in newsyl:
            cluster["syl"].append(s)
        for fe in newfeatures:
            cluster["features"].append(fe)
        cluster["d"] = inclass_d
        cluster["f_low"] = (newf_low + cluster["f_low"]) / 2  # not sure if this is correct
        cluster["f_high"] = (newf_high + cluster["f_high"]) / 2
        cluster["segs"].append(newseg)
        print('Updated Class ', "'", cluster["label"], "'" '\tin-class_d: ',
              cluster["d"], '\tf_low: ', cluster["f_low"], '\tf_high: ',
              cluster["f_high"])
        return cluster
