# Clustering.py
#
# Cluster segments

# Version 1.5 09/09/19
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis

#    AviaNZ birdsong analysis program
#    Copyright (C) 2017--2018

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

import numpy as np
import random
import os, wavio
import librosa

import WaveletSegment
import WaveletFunctions
import SignalProc
import Segment

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
# from sklearn.cluster import OPTICS
# from sklearn import cluster_optics_dbscan
from sklearn import metrics
from sklearn.manifold import TSNE
from statistics import mode


class Clustering:
    # This class implements various clustering algorithms and performance measures for the AviaNZ interface
    # Based on scikit-learn

    def __init__(self, features, labels):
        if not features == []:
            features = StandardScaler().fit_transform(features)
        self.features = features
        self.targets = labels

    def clusteringScore1(self, labels_true, labels):
        """ Evaluate clustering performance using different scores when ground truth labels are present.
        """
        arc = self.adjustedRandScore(labels_true, labels)
        ami = self.adjustedMutualInfo(labels_true, labels)
        h = self.homogeneityScore(labels_true, labels)
        c = self.completenessScore(labels_true, labels)
        v = self.vMeasureScore(labels_true, labels)

        return arc, ami, h, c, v

    def clusteringScore2(self, features, labels):
        """ Evaluate clustering performance using different scores when ground truth labels are NOT present.
        """
        sc = self.silhouetteCoef(features, labels)

        return sc

    def homogeneityScore(self, labels_true, labels):
        """ Homogeneity: each cluster contains only members of a single class.
            score - between 0.0 and 1.0.
            1.0 perfectly homogeneous
        """
        hs = metrics.homogeneity_score(labels_true, labels)
        print("Homogeneity: %0.3f" % hs)

        return hs

    def completenessScore(self, labels_true, labels):
        """ Completeness: all members of a given class are assigned to the same cluster.
            score - between 0.0 and 1.0.
            1.0 perfectly complete
        """
        cs = metrics.completeness_score(labels_true, labels)
        print("Completeness: %0.3f" % cs)

        return cs

    def vMeasureScore(self, labels_true, labels):
        """ V-measure is the harmonic mean between homogeneity and completeness.
            score - between 0.0 and 1.0.
            1.0 perfectly complete labeling
        """
        vs = metrics.v_measure_score(labels_true, labels)
        print("V-measure: %0.3f" % vs)

        return vs

    def adjustedRandScore(self, labels_true, labels):
        """ Measures the similarity of the two assignments, ignoring permutations and with chance normalization.
            score - between -1.0 and 1.0.
            Random labelings will have score close to 0.0.
            1.0 perfect match.
        """
        ari = metrics.adjusted_rand_score(labels_true, labels)
        print("Adjusted Rand Index: %0.3f" % ari)

        return ari

    def adjustedMutualInfo(self, labels_true, labels):
        """ Adjusted Mutual Information between two clusterings. Measures the agreement of the two assignments,
            ignoring permutations.
            score - =< 1.0.
            1.0 perfect match.
        """
        ami = metrics.adjusted_mutual_info_score(labels_true, labels)
        print("Adjusted Mutual Information: %0.3f" % ami)

        return ami

    def silhouetteCoef(self, features, labels):
        """ When the ground truth labels are not present.
            Mean Silhouette Coefficient of all samples.
            Calculated using the mean intra-cluster distance and the mean nearest-cluster distance for each
            sample.
            score - between -1.0 and 1.0 (perfect).
            score close to zero: overlapping clusters.
            negative score: a sample has been assigned to the wrong cluster, as a different cluster is more similar.
        """
        sc = metrics.silhouette_score(features, labels)
        print("Silhouette Coefficient: %0.3f" % sc)

        return sc

    def kMeans(self, init='k-means++', n_clusters=8, n_init=10):
        """ K-Means clustering.
            Useful when: general-purpose, even cluster size, flat geometry, not too many clusters.
        """
        model = KMeans(init=init, n_clusters=n_clusters, n_init=n_init)
        model.fit(self.features)

        return model

    def miniBatchKmeans(self, n_clusters=8, init='k-means++', max_iter=100, batch_size=25):
        """ Variant of the K-Means algorithm, uses mini-batches to reduce the computation time.
        """
        model = MiniBatchKMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, batch_size=batch_size)
        model.fit(self.features)

        return model

    def meanShift(self):
        """ A sliding-window-based algorithm that attempts to find dense areas of data points.
            Usecase: many clusters, uneven cluster size, non-flat geometry.
        """
        model = MeanShift()
        model.fit(self.features)

        return model

    def DBscan(self, eps=0.5, min_samples=5, metric='euclidean'):
        """ Density-Based Spatial Clustering of Applications with Noise. An extension to mean shift clustering.
            Finds core samples of high density and expands clusters from them.
            Usecase: non-flat geometry, uneven cluster sizes
        """
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        model.fit(self.features)

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

    def spectralClustering(self, n_clusters=8, eigen_solver=None, random_state=None, n_init=10, gamma=1.0,
                           affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3,
                           coef0=1, kernel_params=None, n_jobs=None):
        """ Requires the number of clusters to be specified. Good for small number of classes.
            Usecase: few clusters, even cluster size, non-flat geometry.
        """
        model = SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver, random_state=random_state,
                                   n_init=n_init, gamma=gamma, affinity=affinity, n_neighbors=n_neighbors,
                                   eigen_tol=eigen_tol, assign_labels=assign_labels, degree=degree, coef0=coef0,
                                   kernel_params=kernel_params, n_jobs=n_jobs)
        model.fit(self.features)

        return model

    def agglomerativeClustering(self, n_clusters=3, distance_threshold=None, linkage='ward', affinity='euclidean',
                                compute_full_tree=False):
        """ A Hierarchical clustering using a bottom up approach: each observation starts in its own cluster, and
            clusters are successively merged together.
            Usecase: many clusters, possibly connectivity constraints, non Euclidean distances.
        """
        model = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold, linkage=linkage,
                                        affinity=affinity, compute_full_tree=compute_full_tree)
        model.fit(self.features)

        return model

    def GMM(self, n_components=3, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1,
            init_params='kmeans'):
        """ Gaussian mixture model. Not scalable.
            Usecase: flat geometry, good for density estimation.
        """
        model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, tol=tol,
                                reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params)
        model.fit(self.features)
        model.labels_ = model.predict(self.features)

        return model

    def affinityPropagation(self, damping=0.5, max_iter=200, convergence_iter=15):
        """ Affinity Propagation.
            Usecase: many clusters, uneven cluster size, non-flat geometry.
        """
        model = AffinityPropagation(damping=damping, max_iter=max_iter, convergence_iter=convergence_iter)
        model.fit(self.features)

        return model

    def som(self, mapsize):
        """ Self Organising Map
        """
        import sompy
        som = sompy.SOMFactory.build(self.features, [], mask=None, mapshape='planar', lattice='rect', normalization='var',
                                     initialization='pca', neighborhood='gaussian', training='batch', name='sompy')
        som.train()

        return som

    def cluster(self, dir, species=None, feature='we', n_mels=24, minlen=0.2, denoise=False,
                alg='agglomerative', n_clusters=None):
        """
        Cluster segments during training to make sub-filters.
        Given wav + annotation files,
            1) identify syllables using median clipping/ FIR
            2) make them to fixed-length by padding or clipping
            3) use existing clustering algorithems
        :param dir: path to directory with wav & wav.data files
        :param species: string, will train on segments containing this label
        :param feature: 'we' (wavelet energy), 'mfcc', or 'chroma'
        :param n_mels: number of mel coeff when feature='mfcc'
        :param minlen: min syllable length in secs
        :param denoise: True/False
        :param alg: algorithm to use, default to agglomerative
        :param n_clusters: number of clusters, optional
        :return: clustered segments
        """
        self.alg = alg
        self.n_clusters = n_clusters
        # Get the frequency band from annotations
        lowlist = []
        highlist = []
        srlist = []
        # Directory mode
        if os.path.isdir(dir):
            for root, dirs, files in os.walk(str(dir)):
                for file in files:
                    if file.endswith('.wav') and file + '.data' in files:
                        wavobj = wavio.read(os.path.join(root, file))
                        srlist.append(wavobj.rate)
                        # Read the annotation
                        segments = Segment.SegmentList()
                        segments.parseJSON(os.path.join(root, file + '.data'))
                        # keep the right species
                        if species:
                            thisSpSegs = segments.getSpecies(species)
                        else:
                            thisSpSegs = np.arange(len(segments)).tolist()
                        for segix in thisSpSegs:
                            seg = segments[segix]
                            lowlist.append(seg[2])
                            highlist.append(seg[3])
        # File mode
        elif os.path.isfile(dir):
            if dir.endswith('.wav') and os.path.exists(dir + '.data'):
                wavobj = wavio.read(dir)
                srlist.append(wavobj.rate)
                # Read the annotation
                segments = Segment.SegmentList()
                segments.parseJSON(dir + '.data')
                # keep the right species
                if species:
                    thisSpSegs = segments.getSpecies(species)
                else:
                    thisSpSegs = np.arange(len(segments)).tolist()
                for segix in thisSpSegs:
                    seg = segments[segix]
                    lowlist.append(seg[2])
                    highlist.append(seg[3])

        if len(thisSpSegs) < self.n_clusters:
            self.n_clusters = len(thisSpSegs)*2 + 1
            # why this is more than #segments? self.n_clusters relevents to syllable level clusters, differs to segment
            # level clusters with majority voting

        f1 = np.min(lowlist)
        f2 = np.median(highlist)

        arr = [2000, 4000, 8000, 16000, 32000]
        pos = np.abs(arr - np.median(highlist) * 2).argmin()
        fs = arr[pos]

        # TODO: is this necessary?
        if fs > np.min(srlist):
            fs = np.min(srlist)

        if fs < f2 * 2 + 50:
            f2 = fs // 2 - 50

        if f2 < f1:
            f2 = np.mean(highlist)

        # Find the lower and upper bounds (relevant to the frq range), when the range is given
        if feature == 'mfcc' and f1 != 0 and f2 != 0:
            mels = librosa.core.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fs / 2, htk=False)
            ind_flow = (np.abs(mels - f1)).argmin()
            ind_fhigh = (np.abs(mels - f2)).argmin()

        elif feature == 'we' and f1 != 0 and f2 != 0:
            linear = np.linspace(0, fs / 2, 62)
            ind_flow = (np.abs(linear - f1)).argmin()
            ind_fhigh = (np.abs(linear - f2)).argmin()

        # Clustering at syllable level, therefore find the syllables in each segment
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
                            audiodata = self.loadFile(filename=os.path.join(root, file), duration=seg[1] - seg[0],
                                                     offset=seg[0], fs=fs, denoise=denoise, f1=f1, f2=f2)
                            minlen = minlen * fs
                            start = int(seg[0] * fs)
                            sp = SignalProc.SignalProc(256, 128)
                            sp.data = audiodata
                            sp.sampleRate = fs
                            sgRaw = sp.spectrogram(256, 128)
                            segment = Segment.Segmenter(sp, fs)
                            syls = segment.medianClip(thr=3, medfiltersize=5, minaxislength=9, minSegment=50)
                            if len(syls) == 0:  # Sanity check
                                segment = Segment.Segmenter(sp, fs)
                                syls = segment.medianClip(thr=2, medfiltersize=5, minaxislength=9, minSegment=50)
                            syls = segment.checkSegmentOverlap(syls)  # merge overlapped segments
                            syls = [[int(s[0] * fs) + start, int(s[1] * fs + start)] for s in syls]

                            # Sanity check, e.g. when user annotates syllables tight, median clipping may not detect it.
                            if len(syls) == 0:
                                syls = [[start, int(seg[1] * fs)]]
                            if len(syls) > 1:
                                syls = segment.mergeshort(syls, minlen)  # Merge short segments
                            if len(syls) == 1 and syls[0][1] - syls[0][0] < minlen:  # Sanity check
                                syls = [[start, int(seg[1] * fs)]]
                            syls = [[x[0] / fs, x[1] / fs] for x in syls]
                            # print('\nCurrent:', seg, '--> Median clipping ', syls)
                            for syl in syls:
                                dataset.append([os.path.join(root, file), seg, syl])
        elif os.path.isfile(dir):
            if dir.endswith('.wav') and os.path.exists(dir + '.data'):
                # Read the annotation
                segments = Segment.SegmentList()
                segments.parseJSON(dir + '.data')
                if species:
                    thisSpSegs = segments.getSpecies(species)
                else:
                    thisSpSegs = np.arange(len(segments)).tolist()
                # Now find syllables within each segment, median clipping
                for segix in thisSpSegs:
                    seg = segments[segix]
                    audiodata = self.loadFile(filename=dir, duration=seg[1] - seg[0],
                                              offset=seg[0], fs=fs, denoise=denoise, f1=f1, f2=f2)
                    minlen = minlen * fs
                    start = int(seg[0] * fs)
                    sp = SignalProc.SignalProc(256, 128)
                    sp.data = audiodata
                    sp.sampleRate = fs
                    sgRaw = sp.spectrogram(256, 128)
                    segment = Segment.Segmenter(sp, fs)
                    syls = segment.medianClip(thr=3, medfiltersize=5, minaxislength=9, minSegment=50)
                    if len(syls) == 0:  # Sanity check
                        segment = Segment.Segmenter(sp, fs)
                        syls = segment.medianClip(thr=2, medfiltersize=5, minaxislength=9, minSegment=50)
                    syls = segment.checkSegmentOverlap(syls)  # merge overlapped segments
                    syls = [[int(s[0] * fs) + start, int(s[1] * fs + start)] for s in syls]

                    # Sanity check, e.g. when user annotates syllables tight, median clipping may not detect it.
                    if len(syls) == 0:
                        syls = [[start, int(seg[1] * fs)]]
                    if len(syls) > 1:
                        syls = segment.mergeshort(syls, minlen)  # Merge short segments
                    if len(syls) == 1 and syls[0][1] - syls[0][0] < minlen:  # Sanity check
                        syls = [[start, int(seg[1] * fs)]]
                    syls = [[x[0] / fs, x[1] / fs] for x in syls]
                    # print('\nCurrent:', seg, '--> Median clipping ', syls)
                    for syl in syls:
                        dataset.append([dir, seg, syl])

        # Make syllables fixed-length
        lengths = []
        for data in dataset:
            lengths.append(data[2][1] - data[2][0])
        duration = np.median(lengths)
        # duration is going to be the fixed length of a syllable, if a syllable too long clip it
        for record in dataset:
            if record[2][1] - record[2][0] > duration:
                middle = (record[2][1] + record[2][0]) / 2
                record[2][0] = middle - duration / 2
                record[2][1] = middle + duration / 2

        # Read the syllables and generate features, also zero padding short syllables
        features = []
        for record in dataset:
            audiodata = self.loadFile(filename=record[0], duration=record[2][1] - record[2][0], offset=record[2][0],
                                    fs=fs, denoise=denoise, f1=f1, f2=f2)
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
                we = ws.computeWaveletEnergy(data=audiodata, sampleRate=fs, nlevels=5, wpmode='new')
                we = we.mean(axis=1)
                if f1 != 0 and f2 != 0:
                    we = we[ind_flow:ind_fhigh]  # Limit the frequency to a fixed range f1, f2
                features.append(we)
                record.insert(3, we)
            elif feature == 'chroma':
                chroma = librosa.feature.chroma_cqt(y=audiodata, sr=fs)
                # chroma = librosa.feature.chroma_stft(y=data, sr=fs)
                chroma = scale(chroma, axis=1)
                features.append(chroma)
                record.insert(3, chroma)

        features = TSNE().fit_transform(features)
        # learners = Clustering(features, [])
        self.features = features

        model = self.trainModel()
        predicted_labels = model.labels_
        # clusters = len(set(model.labels_))

        # Attach the label to each syllable
        for i in range(len(predicted_labels)):
            dataset[i].insert(4, predicted_labels[i])

        clustered_dataset = []
        for record in dataset:
            if record[:2] not in clustered_dataset:
                clustered_dataset.append(record[:2])

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

        # Make the labels continous, e.g. agglomerative may have produced 0, 2, 3, ...
        ulabels = list(set(labels))
        nclasses = len(ulabels)
        dic = []
        for i in range(nclasses):
            dic.append((ulabels[i], i))
        dic = dict(dic)

        # Update the labels
        for i in range(len(clustered_dataset)):
            clustered_dataset[i].insert(4, dic[labels[i]])

        return clustered_dataset, fs, nclasses, duration

    def trainModel(self,):
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
                model = self.agglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=0.5,
                                                     linkage='complete')
            else:
                model = self.agglomerativeClustering(n_clusters=self.n_clusters, compute_full_tree=False,
                                                     distance_threshold=None, linkage='complete')
            model.fit_predict(self.features)
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
                audiodata = self.loadFile(filename=record[0], duration=syl[1] - syl[0], offset=syl[0], fs=fs,
                                          denoise=denoise, f1=f1, f2=f2)
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
                    # chroma = librosa.feature.chroma_stft(y=data, sr=fs)
                    chroma = scale(chroma, axis=1)
                    fc.append(chroma)
        return np.mean(fc, axis=0)


    def loadFile(self, filename, duration=0, offset=0, fs=0, denoise=False, f1=0, f2=0):
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
