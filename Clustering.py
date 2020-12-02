# Clustering.py
#
# Cluster segments

# Version 3.0 14/09/20
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2020

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

    # def DBscan(self, eps=0.5, min_samples=5, metric='euclidean'):
    def DBscan(self, eps=0.5, min_samples=5):
        """ Density-Based Spatial Clustering of Applications with Noise. An extension to mean shift clustering.
            Finds core samples of high density and expands clusters from them.
            Usecase: non-flat geometry, uneven cluster sizes
        """
        # model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        # model = DBSCAN(eps=eps, min_samples=min_samples, metric=self.custom_dist)
        model = DBSCAN(metric='precomputed')
        d = pairwise_distances(self.features, self.features, metric=self.custom_dist)
        # model.fit(self.features)
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
        d = pairwise_distances(self.features, self.features, metric=self.custom_dist)
        model.fit(d)
        # model.fit(self.features)

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

    # def cluster(self, dirname, fs, species=None, feature='we', n_mels=24, minlen=0.2, denoise=False, alg='agglomerative'):
    def cluster(self, dirname, fs, species=None, feature='we', n_mels=24, minlen=0.2, denoise=False,
                    alg='agglomerative'):
        """
        Cluster segments during training to make sub-filters.
        Given wav + annotation files,
            1) identify syllables using median clipping/ FIR
            2) make them to fixed-length by padding or clipping
            3) use existing clustering algorithems
        :param dir: path to directory with wav & wav.data files
        :param fs: sample rate
        :param species: string, optional. will train on segments containing this label
        :param feature: 'we' (wavelet energy), 'mfcc', or 'chroma'
        :param n_mels: number of mel coeff when feature='mfcc'
        :param minlen: min syllable length in secs
        :param denoise: True/False
        :param alg: algorithm to use, default to agglomerative
        :return: clustered segments - a list of lists [[file1, seg1, [syl1, syl2], [features1, features2], predict], ...]
                 fs, nclasses, syllable duration (median)
        """

        self.alg = alg
        nlevels = 6
        weInds = []

        # 1. Get the frequency band and sampling frequency from annotations
        f1, f2 = self.getFrqRange(dirname, species, fs)
        print("Clustering using sampling rate", fs)

        # 2. Find the lower and upper bounds (relevant to the frq range)
        if feature == 'mfcc' and f1 != 0 and f2 != 0:
            mels = librosa.core.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fs / 2, htk=False)
            ind_flow = (np.abs(mels - f1)).argmin()
            ind_fhigh = (np.abs(mels - f2)).argmin()

        elif feature == 'we' and f1 != 0 and f2 != 0:
            weInds = self.nodesInRange(nlevels, f1, f2, fs)

        # 3. Clustering at syllable level, therefore find the syllables in each segment
        dataset = self.findSyllables(dirname, species, minlen, fs, f1, f2, denoise)
        # dataset format: [[file1, seg1, syl1], [file1, seg1, syl2], [file1, seg2, syl1],..]

        # Make syllables fixed-length (again to have same sized feature matrices) and generate features
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

        # 4. Read the syllables and generate features, also zero padding short syllables
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
                we = ws.computeWaveletEnergy(data=audiodata, sampleRate=fs, nlevels=nlevels, wpmode='new')
                we = we.mean(axis=1)
                if weInds:
                    we = we[weInds]
                # if f1 != 0 and f2 != 0:
                #     we = we[ind_flow:ind_fhigh]  # Limit the frequency to a fixed range f1, f2
                features.append(we)
                record.insert(3, we)
            elif feature == 'chroma':
                chroma = librosa.feature.chroma_cqt(y=audiodata, sr=fs)
                # chroma = librosa.feature.chroma_stft(y=data, sr=fs)
                chroma = scale(chroma, axis=1)
                features.append(chroma)
                record.insert(3, chroma)

        # 5. Actual clustering
        # features = TSNE().fit_transform(features)
        self.features = features

        model = self.trainModel()
        predicted_labels = model.labels_
        print(predicted_labels)
        # clusters = len(set(model.labels_))

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
            # clustered_dataset format: [[file1, seg1, [syl1, syl2], [features1, features2], predict], ...]

        return clustered_dataset, nclasses, duration

    def nodesInRange(self, nlevels, f1, f2, fs):
        ''' Return the indices (nodes) to keep
        '''
        allnodes = range(1, 2 ** (nlevels + 1) - 1)
        inband = []
        WF = WaveletFunctions.WaveletFunctions(data=[], wavelet='dmey2', maxLevel=1, samplerate=fs)
        for i in allnodes:
            flow, fhigh = WF.getWCFreq(i, fs)
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
                    if file.lower().endswith('.wav') and file + '.data' in files:
                        # wavrate = wavio.readFmt(os.path.join(root, file))[0]
                        # srlist.append(wavrate)
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

        # File mode (from the main interface)
        elif os.path.isfile(dirname):
            if dirname.lower().endswith('.wav') and os.path.exists(dirname + '.data'):
                # wavrate = wavio.readFmt(dirname)[0]
                # srlist.append(wavrate)
                # Read the annotation
                segments = Segment.SegmentList()
                segments.parseJSON(dirname + '.data')
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
                            syls = self.findSyllablesSeg(os.path.join(root, file), seg, fs, f1, f2, denoise, minlen)
                            for syl in syls:
                                dataset.append([os.path.join(root, file), seg, syl])
        elif os.path.isfile(dirname):
            if dirname.lower().endswith('.wav') and os.path.exists(dirname + '.data'):
                # Read the annotation
                segments = Segment.SegmentList()
                segments.parseJSON(dirname + '.data')
                if species:
                    thisSpSegs = segments.getSpecies(species)
                else:
                    thisSpSegs = np.arange(len(segments)).tolist()
                # Now find syllables within each segment, median clipping
                for segix in thisSpSegs:
                    seg = segments[segix]
                    syls = self.findSyllablesSeg(dirname, seg, fs, f1, f2, denoise, minlen)
                    for syl in syls:
                        dataset.append([dirname, seg, syl])
        return dataset

    def findSyllablesSeg(self, file, seg, fs, f1, f2, denoise, minlen):
        """ Find syllables in the segment using median clipping - single segment
        :return: syllables list
        """
        # TODO: Use f1 and f2 to restrict spectrogram in median clipping to skip some of the noise
        # audiodata = self.loadFile(filename=file, duration=seg[1] - seg[0], offset=seg[0], fs=fs, denoise=denoise, f1=f1, f2=f2)
        audiodata = self.loadFile(filename=file, duration=seg[1] - seg[0], offset=seg[0], fs=fs, denoise=denoise)
        start = seg[0]
        sp = SignalProc.SignalProc()
        sp.data = audiodata
        sp.sampleRate = fs
        _ = sp.spectrogram()
        # Show only the segment frequencies to the median clipping and avoid overlapping noise - better than filtering when loading audiodata (it could make aliasing effect)
        linear = np.linspace(0, fs / 2, sp.window_width/2)
        # ind_flow = (np.abs(linear - f1)).argmin()
        # ind_fhigh = (np.abs(linear - f2)).argmin()
        ind_flow = (np.abs(linear - seg[2])).argmin()
        ind_fhigh = (np.abs(linear - seg[3])).argmin()
        sp.sg = sp.sg[:, ind_flow:ind_fhigh]

        segment = Segment.Segmenter(sp, fs)

        syls = segment.medianClip(thr=3, medfiltersize=5, minaxislength=9, minSegment=50)
        if len(syls) == 0:  # Sanity check
            # Try again with lower threshold
            segment = Segment.Segmenter(sp, fs)
            syls = segment.medianClip(thr=2, medfiltersize=5, minaxislength=9, minSegment=50)

        # Merge overlapped segments
        syls = segment.checkSegmentOverlap(syls)
        syls = segment.deleteShort(syls, minlen)
        syls = [[s[0] + start, s[1] + start] for s in syls]

        # Sanity check, e.g. when user annotates syllables tight, median clipping may not detect it
        if len(syls) == 0:
            syls = [[start, seg[1]]]
        if len(syls) == 1 and syls[0][1] - syls[0][0] < minlen:  # Sanity check
            syls = [[start, seg[1]]]

        return syls

    def trainModel(self):
        """ Clustering model"""
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

            # # Either set n_clusters=None and compute_full_tree=T or distance_threshold=None
            # if not self.n_clusters:
            #     model = self.agglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=0.5,
            #                                          linkage='complete')
            # else:
            #     model = self.agglomerativeClustering(n_clusters=self.n_clusters, compute_full_tree=False,
            #                                          distance_threshold=None, linkage='complete')
            # # model.fit_predict(self.features)
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

    def cluster_by_dist(self, dir, species, feature='we', n_mels=24, fs=0, minlen=0.2, f_1=0, f_2=0, denoise=False, single=False,
                        distance='dtw', max_clusters=10):
        """
        Given wav + annotation files,
            1) identify syllables using median clipping/ FIR
            2) generate features WE/MFCC/chroma
            3) calculate DTW distances and decide class/ generate new class
        :param dir: directory of audio and annotations
        :param feature: 'WE' or 'MFCC' or 'chroma'
        :param n_mels: number of mel coefs for MFCC
        :param fs: prefered sampling frequency, 0 leads to calculate it from the anotations
        :param minlen: min syllable length in secs
        :param f_1: lower frequency bound, 0 leads to calculate it from the anotations
        :param f_2: upper frequency bound, 0 leads to calculate it from the anotations
        :param denoise: wavelet denoise
        :param single: True means when there are multiple syllables in a segment, add only one syllable to the cluster info
        :param distance: 'dtw' or 'xcor'
        :return: possible clusters
        """
        import Segment
        import SignalProc
        from scipy import signal

        # Get flow and fhigh for bandpass from annotations
        lowlist = []
        highlist = []
        srlist = []
        for root, dirs, files in os.walk(str(dir)):
            for file in files:
                if file.lower().endswith('.wav') and file + '.data' in files:
                    wavrate = wavio.readFmt(os.path.join(root, file))[0]
                    srlist.append(wavrate)
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
        print(lowlist)
        print(highlist)
        print(srlist)
        if f_1 == 0:
            f_1 = np.min(lowlist)
        if f_2 == 0:
            f_2 = np.median(highlist)

        if fs == 0:
            arr = [4000, 8000, 16000]
            pos = np.abs(arr - np.median(highlist) * 2).argmin()
            fs = arr[pos]

        print('fs: ', fs)

        if fs > np.min(srlist):
            print(fs)
            fs = np.min(srlist)

        if fs < f_2 * 2 + 50:
            f_2 = fs // 2 - 50

        minlen_samples = minlen * fs

        print('Frequency band:', f_1, '-', f_2)
        print('fs: ', fs)

        # Find the lower and upper bounds (relevant to the frq range), when the range is given
        if feature == 'mfcc' and f_1 != 0 and f_2 != 0:
            mels = librosa.core.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fs / 2, htk=False)
            ind_flow = (np.abs(mels - f_1)).argmin()
            ind_fhigh = (np.abs(mels - f_2)).argmin()

        elif feature == 'we' and f_1 != 0 and f_2 != 0:
            linear = np.linspace(0, fs / 2, 62)
            ind_flow = (np.abs(linear - f_1)).argmin()
            ind_fhigh = (np.abs(linear - f_2)).argmin()

        # Ready for clustering
        max_clusters = max_clusters
        n_clusters = 0
        clusters = []
        for root, dirs, files in os.walk(str(dir)):
            for file in files:
                if file.lower().endswith('.wav') and file + '.data' in files:
                    # Read the annotation
                    segments = Segment.SegmentList()
                    segments.parseJSON(os.path.join(root, file + '.data'))
                    # keep the right species
                    if species:
                        thisSpSegs = segments.getSpecies(species)
                    else:
                        thisSpSegs = np.arange(len(segments)).tolist()

                    # Sort the segments longest to shortest, would be a good idea to avoid making first class with only
                    # one member :)
                    segments_len = [segments[segix][1] - segments[segix][0] for segix in thisSpSegs]
                    inds = np.argsort(segments_len)[::-1]
                    sortedsegments = [segments[i] for i in inds]

                    # Now find syllables within each segment, median clipping
                    for seg in sortedsegments:
                        if seg[0] == -1:
                            continue
                        audiodata = self.loadFile(filename=os.path.join(root, file), duration=seg[1] - seg[0],
                                                      offset=seg[0], fs=fs, denoise=denoise, f1=f_1, f2=f_2)
                        start = int(seg[0] * fs)
                        sp = SignalProc.SignalProc(256, 128)
                        sp.data = audiodata
                        sp.sampleRate = fs
                        sgRaw = sp.spectrogram(256, 128)
                        segment = Segment.Segmenter(sp=sp, fs=fs)
                        syls = segment.medianClip(thr=3, medfiltersize=5, minaxislength=9, minSegment=50)
                        if len(syls) == 0:  # Try again with FIR
                            syls = segment.segmentByFIR(threshold=0.05)
                        syls = segment.checkSegmentOverlap(syls)  # merge overlapped segments
                        syls = [[int(s[0] * fs), int(s[1] * fs)] for s in syls]

                        if len(syls) == 0:  # Sanity check, when annotating syllables tight,
                            syls = [[0, int((seg[1] - seg[0]) * fs)]]  # median clipping doesn't detect it.
                        if len(syls) > 1:
                            # TODO: samples to seconds
                            syls = segment.joinGaps(syls, minlen_samples)  # Merge short segments
                        if len(syls) == 1 and syls[0][1] - syls[0][0] < minlen_samples:  # Sanity check
                            syls = [[0, int((seg[1] - seg[0]) * fs)]]
                        temp = [[np.round((x[0] + start) / fs, 2), np.round((x[1] + start) / fs, 2)] for x in syls]
                        print('\nCurrent:', seg, '--> syllables >', minlen, 'secs ', temp)

                        # Calculate features of the syllables in the current segment.
                        f = []
                        for s in syls:
                            data = audiodata[s[0]:s[1]]
                            if feature == 'mfcc':  # MFCC
                                mfcc = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=n_mels)
                                if f_1 != 0 and f_2 != 0:
                                    mfcc = mfcc[ind_flow:ind_fhigh, :]  # Limit the frequency to the fixed range [f_1, f_2]
                                mfcc_delta = librosa.feature.delta(mfcc, mode='nearest')
                                mfcc = np.concatenate((mfcc, mfcc_delta), axis=0)
                                mfcc = scale(mfcc, axis=1)
                                # librosa.display.specshow(mfcc, sr=fs, x_axis='time')
                                # m = [i for sublist in mfcc for i in sublist]
                                f.append(mfcc)

                            elif feature == 'we':  # Wavelet Energy
                                ws = WaveletSegment.WaveletSegment(spInfo={})
                                we = ws.computeWaveletEnergy(data=data, sampleRate=fs, nlevels=5, wpmode='new')
                                we = we.mean(axis=1)
                                if f_1 != 0 and f_2 != 0:
                                    we = we[ind_flow:ind_fhigh]  # Limit the frequency to a fixed range f_1, f_2
                                f.append(we)
                            elif feature == 'chroma':
                                chroma = librosa.feature.chroma_cqt(y=data, sr=fs)
                                # chroma = librosa.feature.chroma_stft(y=data, sr=fs)
                                chroma = scale(chroma, axis=1)
                                f.append(chroma)

                        matched = False
                        if n_clusters == 0:
                            print('**Case 1: First class')
                            newclass = self.class_create(label=n_clusters, syl=syls, features=f, f_low=seg[2],
                                                         f_high=seg[3], segs=[(os.path.join(root, file), seg)],
                                                         single=single, dist_method=distance)
                            clusters.append(newclass)
                            n_clusters += 1
                            print('Created new class: Class ', "'", newclass["label"], "'", ',\tIn-class_d: ',
                                  newclass["d"], '\tf_low: ', newclass["f_low"], '\tf_high: ', newclass["f_high"])
                            matched = True
                        if not matched:
                            # See if the syllables in the current seg match with any existing class
                            min_ds = []  # Keep track of the minimum distances to each class
                            clusters = random.sample(clusters, len(clusters))  # Shuffle the clusters to avoid bias
                            for c in range(len(clusters)):
                                f_c = clusters[c]["features"]  # features of the current class c
                                dist_c = np.zeros((len(f_c), len(f)))  # distances to the current class c
                                for i in range(len(f_c)):
                                    for j in range(len(f)):
                                        if distance == 'dtw':
                                            d, _ = librosa.sequence.dtw(f_c[i], f[j], metric='euclidean')
                                            dist_c[i, j] = d[d.shape[0] - 1][d.shape[1] - 1]
                                        elif distance == 'xcor':
                                            corr = signal.correlate(f_c[i], f[j], mode='full')
                                            dist_c[i, j] = np.sum(corr) / max(len(f_c[i]), len(f[j]))

                                # Min distance to the current class
                                print('Distance to Class ', clusters[c]["label"], ': ', np.amin(dist_c[dist_c != 0]),
                                      '( In-class distance: ', clusters[c]["d"], ')')
                                min_ds.append(np.amin(dist_c[dist_c != 0]))

                            # Now get the clusters sorted according to the min dist
                            ind = np.argsort(min_ds)
                            min_ds = np.sort(min_ds)
                            # make the cluster order
                            clusters = [clusters[i] for i in ind]
                            for c in range(len(clusters)):
                                if (clusters[c]["d"] != 0) and min_ds[c] < (clusters[c]["d"] + clusters[c]["d"] * 0.1):
                                    print('**Case 2: Found a match with a class > one syllable')
                                    print('Class ', clusters[c]["label"], ', dist ', min_ds[c])
                                    # Update this class
                                    clusters[c] = self.class_update(cluster=clusters[c], newfeatures=f, newf_low=seg[2],
                                                               newf_high=seg[3], newsyl=syls,
                                                               newseg=(os.path.join(root, file), seg), single=single,
                                                               dist_method=distance)
                                    matched = True
                                    break  # found a match, exit from the for loop, go to the next segment

                                elif c < len(clusters) - 1:
                                    continue  # continue to the next class

                        # Checked most of the classes by now, if still no match found, check the classes with only one
                        # data point (clusters[c]["d"] == 0).
                        # Note the arbitrary thr.
                        if not matched:
                            if distance == 'dtw':
                                thr = 25
                            elif distance == 'xcor':
                                thr = 1000
                            for c in range(len(clusters)):
                                if clusters[c]["d"] == 0 and min_ds[c] < thr:
                                    print('**Case 3: In-class dist of ', clusters[c]["label"], '=', clusters[c]["d"],
                                          'and this example < ', thr, ' dist')
                                    print('Class ', clusters[c]["label"], ', dist ', min_ds[c])
                                    # Update this class
                                    clusters[c] = self.class_update(cluster=clusters[c], newfeatures=f, newf_low=seg[2],
                                                               newf_high=seg[3], newsyl=syls,
                                                               newseg=(os.path.join(root, file), seg), single=single,
                                                               dist_method=distance)
                                    matched = True
                                    break  # Break the search and go to the next segment

                        # If no match found yet, check the max clusters
                        if not matched:
                            if n_clusters == max_clusters:
                                print('**Case 4: Reached max classes, therefore adding current seg to the closest '
                                      'class... ')
                                # min_ind = np.argmin(min_ds)
                                # classes are sorted in ascending order of distance already
                                for c in range(len(clusters)):
                                    if min_ds[c] <= 4 * clusters[c]["d"] or clusters[c]["d"] == 0:
                                        print('Class ', clusters[c]["label"], ', dist ', min_ds[c],
                                              '(in-class distance:', clusters[c]["d"], ')')
                                        # Update this class
                                        clusters[c] = self.class_update(cluster=clusters[c], newfeatures=f, newf_low=seg[2],
                                                                   newf_high=seg[3], newsyl=syls,
                                                                   newseg=(os.path.join(root, file), seg),
                                                                   single=single,
                                                                   dist_method=distance)
                                        matched = True
                                        break
                                if not matched:
                                    print('Class ', clusters[0]["label"], ', dist ', min_ds[0],
                                          '(in-class distance:', clusters[0]["d"], ')')
                                    # Update this class
                                    # TODO: don't update the class as it is an outlier?
                                    clusters[0] = self.class_update(cluster=clusters[0], newfeatures=f, newf_low=seg[2],
                                                               newf_high=seg[3], newsyl=syls,
                                                               newseg=(os.path.join(root, file), seg), single=single,
                                                               dist_method=distance)
                                    matched = True
                                continue  # Continue to next segment

                        #  If still no luck, create a new class
                        if not matched:
                            print('**Case 5: None of Case 1-4')
                            newclass = self.class_create(label=n_clusters, syl=syls, features=f, f_low=seg[2], f_high=seg[3],
                                                    segs=[(os.path.join(root, file), seg)], single=single,
                                                    dist_method=distance)
                            print('Created a new class: Class ', n_clusters + 1)
                            clusters.append(newclass)
                            n_clusters += 1
                            print('Created new class: Class ', "'", newclass["label"], "'", ',\tin-class_d: ',
                                  newclass["d"], '\tf_low: ', newclass["f_low"], '\tf_high: ', newclass["f_high"])

        print('\n\n--------------Clusters created-------------------')
        clustered_segs = []
        for c in range(len(clusters)):
            print('Class ', clusters[c]['label'], ': ', len(clusters[c]['segs']))
            for s in range(len(clusters[c]['segs'])):
                print('\t', clusters[c]['segs'][s])
                if single:
                    clustered_segs.append([clusters[c]['segs'][s][0], clusters[c]['segs'][s][1],
                                           [clusters[c]['features'][s]], clusters[c]['label']])
                else:
                    clustered_segs.append([clusters[c]['segs'][s][0], clusters[c]['segs'][s][1], clusters[c]['label']])

        # Clustered segments
        print('\n\n################### Clustered segments ############################')
        for s in clustered_segs:
            print(s)
        return clustered_segs, fs, n_clusters, 1
        # return clustered_dataset, fs, nclasses, duration

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
