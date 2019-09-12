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


class Clustering:
    # This class implements various clustering algorithms and performance measures for the AviaNZ interface
    # Based on scikit-learn

    def __init__(self, features, labels):
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


def loadFile(filename, duration=0, offset=0, fs=0, denoise=False, f1=0, f2=0):
    """
    Read audio file and preprocess as required
    :param filename:
    :param fs:
    :param f1:
    :param f2:
    :return:
    """
    if offset == 0 and duration == 0:
        wavobj = wavio.read(filename)
    else:
        wavobj = wavio.read(filename, duration, offset)
    sampleRate = wavobj.rate
    audiodata = wavobj.data

    if audiodata.dtype is not 'float':
        audiodata = audiodata.astype('float')   #/ 32768.0
    if np.shape(np.shape(audiodata))[0] > 1:
        audiodata = audiodata[:, 0]

    if fs != 0 and sampleRate != fs:
        audiodata = librosa.core.audio.resample(audiodata, sampleRate, fs)
        sampleRate = fs

    # # pre-process
    if denoise:
        WF = WaveletFunctions.WaveletFunctions(data=audiodata, wavelet='dmey2', maxLevel=10, samplerate=fs)
        audiodata = WF.waveletDenoise(thresholdType='soft', maxLevel=10)

    if f1 != 0 and f2 != 0:
        sp = SignalProc.SignalProc([], 0, 256, 128)
        # audiodata = sp.ButterworthBandpass(audiodata, sampleRate, f1, f2)
        audiodata = sp.bandpassFilter(audiodata, sampleRate, f1, f2)

    return audiodata, sampleRate


def cluster_by_dist(dir, feature='we', n_mels=24, fs=0, minlen=0.2, f_1=0, f_2=0, denoise=False, single=False,
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
            if file.endswith('.wav') and file+'.data' in files:
                wavobj = wavio.read(os.path.join(root, file))
                srlist.append(wavobj.rate)
                # Read the annotation
                segments = Segment.SegmentList()
                segments.parseJSON(os.path.join(root, file+'.data'))
                for seg in segments:
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
        pos = np.abs(arr - np.median(highlist)*2).argmin()
        fs = arr[pos]

    print('fs: ', fs)

    if fs > np.min(srlist):
        print(fs)
        fs = np.min(srlist)

    if fs < f_2 * 2 + 50:
        f_2 = fs//2 - 50

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
            if file.endswith('.wav') and file+'.data' in files:
                # Read the annotation
                segments = Segment.SegmentList()
                segments.parseJSON(os.path.join(root, file+'.data'))

                # Sort the segments longest to shortest, would be a good idea to avoid making first class with only
                # one member :)
                if len(segments) > 0 and segments[0][0] == -1:
                    del segments[0]
                segments_len = [seg[1]-seg[0] for seg in segments]
                inds = np.argsort(segments_len)[::-1]
                sortedsegments = [segments[i] for i in inds]

                # Now find syllables within each segment, median clipping
                for seg in sortedsegments:
                    if seg[0] == -1:
                        continue
                    audiodata, sr = loadFile(filename=os.path.join(root, file), duration=seg[1]-seg[0], offset=seg[0],
                                             fs=fs, denoise=denoise, f1=f_1, f2=f_2)
                    start = int(seg[0] * fs)
                    sp = SignalProc.SignalProc(audiodata, fs, 256, 128)
                    sgRaw = sp.spectrogram(audiodata, 256, 128)
                    segment = Segment.Segmenter(data=audiodata, sg=sgRaw, sp=sp, fs=fs, window_width=256, incr=128)
                    syls = segment.medianClip(thr=3, medfiltersize=5, minaxislength=9, minSegment=50)
                    if len(syls) == 0:      # Try again with FIR
                        syls = segment.segmentByFIR(threshold=0.05)
                    syls = segment.checkSegmentOverlap(syls)    # merge overlapped segments
                    syls = [[int(s[0] * sr), int(s[1] * fs)] for s in syls]

                    if len(syls) == 0:                                  # Sanity check, when annotating syllables tight,
                        syls = [[0, int((seg[1]-seg[0])*fs)]]           # median clipping doesn't detect it.
                    if len(syls) > 1:
                        syls = segment.mergeshort(syls, minlen_samples)             # Merge short segments
                    if len(syls) == 1 and syls[0][1]-syls[0][0] < minlen_samples:   # Sanity check
                        syls = [[0, int((seg[1]-seg[0])*fs)]]
                    temp = [[np.round((x[0] + start) / fs, 2), np.round((x[1] + start) / fs, 2)] for x in syls]
                    print('\nCurrent:', seg, '--> syllables >', minlen, 'secs ', temp)

                    # Calculate features of the syllables in the current segment.
                    f = []
                    for s in syls:
                        data = audiodata[s[0]:s[1]]
                        if feature == 'mfcc':    # MFCC
                            mfcc = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=n_mels)
                            if f_1 != 0 and f_2 != 0:
                                mfcc = mfcc[ind_flow:ind_fhigh, :]  # Limit the frequency to the fixed range [f_1, f_2]
                            mfcc_delta = librosa.feature.delta(mfcc, mode='nearest')
                            mfcc = np.concatenate((mfcc, mfcc_delta), axis=0)
                            mfcc = scale(mfcc, axis=1)
                            # librosa.display.specshow(mfcc, sr=fs, x_axis='time')
                            # m = [i for sublist in mfcc for i in sublist]
                            f.append(mfcc)

                        elif feature == 'we':    # Wavelet Energy
                            ws = WaveletSegment.WaveletSegment(spInfo=[])
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
                        newclass = class_create(label=n_clusters, syl=syls, features=f, f_low=seg[2], f_high=seg[3],
                                                segs=[(os.path.join(root, file), seg)], single=single,
                                                dist_method=distance)
                        clusters.append(newclass)
                        n_clusters += 1
                        print('Created new class: Class ', "'", newclass["label"], "'", ',\tIn-class_d: ',
                              newclass["d"], '\tf_low: ', newclass["f_low"], '\tf_high: ', newclass["f_high"])
                        matched = True
                    if not matched:
                        # See if the syllables in the current seg match with any existing class
                        min_ds = []     # Keep track of the minimum distances to each class
                        clusters = random.sample(clusters, len(clusters))   # Shuffle the clusters to avoid bias
                        for c in range(len(clusters)):
                            f_c = clusters[c]["features"]   # features of the current class c
                            dist_c = np.zeros((len(f_c), len(f)))   # distances to the current class c
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
                                clusters[c] = class_update(cluster=clusters[c], newfeatures=f, newf_low=seg[2],
                                                           newf_high=seg[3], newsyl=syls,
                                                           newseg=(os.path.join(root, file), seg), single=single,
                                                           dist_method=distance)
                                matched = True
                                break       # found a match, exit from the for loop, go to the next segment

                            elif c < len(clusters)-1:
                                continue    # continue to the next class

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
                                print('**Case 3: In-class dist of ', clusters[c]["label"],  '=', clusters[c]["d"],
                                      'and this example < ',  thr, ' dist')
                                print('Class ', clusters[c]["label"], ', dist ', min_ds[c])
                                # Update this class
                                clusters[c] = class_update(cluster=clusters[c], newfeatures=f, newf_low=seg[2],
                                                           newf_high=seg[3], newsyl=syls,
                                                           newseg=(os.path.join(root, file), seg), single=single,
                                                           dist_method=distance)
                                matched = True
                                break    # Break the search and go to the next segment

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
                                    clusters[c] = class_update(cluster=clusters[c], newfeatures=f, newf_low=seg[2],
                                                               newf_high=seg[3], newsyl=syls,
                                                               newseg=(os.path.join(root, file), seg), single=single,
                                                               dist_method=distance)
                                    matched = True
                                    break
                            if not matched:
                                print('Class ', clusters[0]["label"], ', dist ', min_ds[0],
                                      '(in-class distance:', clusters[0]["d"], ')')
                                # Update this class
                                # TODO: don't update the class as it is an outlier?
                                clusters[0] = class_update(cluster=clusters[0], newfeatures=f, newf_low=seg[2],
                                                           newf_high=seg[3], newsyl=syls,
                                                           newseg=(os.path.join(root, file), seg), single=single,
                                                           dist_method=distance)
                                matched = True
                            continue    # Continue to next segment

                    #  If still no luck, create a new class
                    if not matched:
                        print('**Case 5: None of Case 1-4')
                        newclass = class_create(label=n_clusters, syl=syls, features=f, f_low=seg[2], f_high=seg[3],
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
    return clustered_segs, fs, n_clusters


def class_create(label, syl, features, f_low, f_high, segs, single=False, dist_method='dtw'):
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
                dist[i,j] = np.sum(corr)/max(len(features[i]), len(features[j]))

    if np.count_nonzero(dist) > 0:
        nonzero = dist > 0
        inclass_d = np.percentile(dist[nonzero], 10)  # TODO: max? mean? a percentile?
    else:
        inclass_d = 0

    if single:
        features = [features[len(features)//2]]     # get the features of the middle syllable

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


def class_update(cluster, newfeatures, newf_low, newf_high, newsyl, newseg, single, dist_method='dtw'):
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
        newfeatures = [newfeatures[len(newfeatures)//2]]
        newsyl = [newsyl[len(newsyl)//2]]

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


def cluster(dir, feature='we', n_mels=24, minlen=0.2, denoise=False,
                    alg='agglomerative', n_clusters=None):
    """
    Given wav + annotation files,
        1) identify syllables using median clipping/ FIR
        2) make them to fixed-length by padding or clipping
        3) use existing clustering algorithems
    :param n_mels: number of mel coeff
    :param fs: prefered sampling frequency
    :param minlen: min syllable length in secs
    :param f_1:
    :param f_2:
    :param denoise:
    :param alg: algorithm to use
    :param n_clusters: number of clusters
    :return: possible clusters
    """
    # Get flow and fhigh for bandpass from annotations
    lowlist = []
    highlist = []
    srlist = []
    for root, dirs, files in os.walk(str(dir)):
        for file in files:
            if file.endswith('.wav') and file+'.data' in files:
                wavobj = wavio.read(os.path.join(root, file))
                srlist.append(wavobj.rate)
                # Read the annotation
                segments = Segment.SegmentList()
                segments.parseJSON(os.path.join(root, file+'.data'))
                for seg in segments:
                    lowlist.append(seg[2])
                    highlist.append(seg[3])

    f_1 = np.min(lowlist)
    f_2 = np.median(highlist)

    arr = [4000, 8000, 16000, 32000]
    pos = np.abs(arr - np.median(highlist)*2).argmin()
    fs = arr[pos]

    # TODO: is this necessary?
    if fs > np.min(srlist):
        fs = np.min(srlist)

    if fs < f_2 * 2 + 50:
        f_2 = fs//2 - 50

    if f_2 < f_1:
        f_2 = np.mean(highlist)

    # Find the lower and upper bounds (relevant to the frq range), when the range is given
    if feature == 'mfcc' and f_1 != 0 and f_2 != 0:
        mels = librosa.core.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fs / 2, htk=False)
        ind_flow = (np.abs(mels - f_1)).argmin()
        ind_fhigh = (np.abs(mels - f_2)).argmin()

    elif feature == 'we' and f_1 != 0 and f_2 != 0:
        linear = np.linspace(0, fs / 2, 62)
        ind_flow = (np.abs(linear - f_1)).argmin()
        ind_fhigh = (np.abs(linear - f_2)).argmin()

    # Find the syllables
    dataset = []
    for root, dirs, files in os.walk(str(dir)):
        for file in files:
            if file.endswith('.wav') and file+'.data' in files:
                # Read the annotation
                segments = Segment.SegmentList()
                segments.parseJSON(os.path.join(root, file+'.data'))
                # Now find syllables within each segment, median clipping
                for seg in segments:
                    audiodata, sr = loadFile(filename=os.path.join(root, file), duration=seg[1]-seg[0], offset=seg[0],
                                             fs=fs, denoise=denoise, f1=f_1, f2=f_2)
                    assert sr == fs
                    minlen = minlen * fs
                    start = int(seg[0] * fs)
                    sp = SignalProc.SignalProc(audiodata, fs, 256, 128)
                    sgRaw = sp.spectrogram(audiodata, 256, 128)
                    segment = Segment.Segmenter(data=audiodata, sg=sgRaw, sp=sp, fs=fs, window_width=256, incr=128)
                    syls = segment.medianClip(thr=3, medfiltersize=5, minaxislength=9, minSegment=50)
                    if len(syls) == 0:      # Sanity check
                        segment = Segment.Segmenter(audiodata, sgRaw, sp, fs, 256, 128)
                        syls = segment.medianClip(thr=2, medfiltersize=5, minaxislength=9, minSegment=50)
                    syls = segment.checkSegmentOverlap(syls)    # merge overlapped segments
                    syls = [[int(s[0] * sr) + start, int(s[1] * fs + start)] for s in syls]

                    if len(syls) == 0:                                  # Sanity check, when annotating syllables tight,
                        syls = [[start, int(seg[1]*fs)]]           # median clipping doesn't detect it.
                    if len(syls) > 1:
                        syls = segment.mergeshort(syls, minlen)             # Merge short segments
                    if len(syls) == 1 and syls[0][1]-syls[0][0] < minlen:   # Sanity check
                        syls = [[start, int(seg[1]*fs)]]
                    syls = [[x[0] / fs, x[1] / fs] for x in syls]
                    # print('\nCurrent:', seg, '--> Median clipping ', syls)
                    for syl in syls:
                        dataset.append([os.path.join(root, file), seg, syl])

    # Make syllables fixed-length
    lengths = []
    for data in dataset:
        lengths.append(data[2][1]-data[2][0])
    # print('min: ', min(lengths), ' max: ', max(lengths), ' median: ', np.median(lengths))
    # This is going to be the fixed length of a syllable, if a syllable too long clip it otherwise padding with zero
    duration = np.median(lengths)
    # Now calculate the features
    for record in dataset:
        if record[2][1]-record[2][0] > duration:
            # get the middle 'duration' secs
            middle = (record[2][1]+record[2][0]) / 2
            record[2][0] = middle - duration/2
            record[2][1] = middle + duration / 2

    # lengths = []
    # for data in dataset:
    #     lengths.append(data[2][1] - data[2][0])
    # print('min: ', min(lengths), ' max: ', max(lengths), ' median: ', np.median(lengths))

    # Read the syllables and generate features
    features = []
    for record in dataset:
        audiodata, _ = loadFile(filename=record[0], duration=record[2][1] - record[2][0], offset=record[2][0],
                                fs=fs, denoise=denoise, f1=f_1, f2=f_2)
        audiodata = audiodata.tolist()
        if record[2][1] - record[2][0] < duration:
            # Zero padding both ends to have fixed duration
            gap = int((duration*fs - len(audiodata))//2)
            z = [0] * gap
            audiodata.extend(z)
            z.extend(audiodata)
            audiodata = z
        if feature == 'mfcc':  # MFCC
            mfcc = librosa.feature.mfcc(y=np.asarray(audiodata), sr=fs, n_mfcc=n_mels)
            if f_1 != 0 and f_2 != 0:
                mfcc = mfcc[ind_flow:ind_fhigh, :]  # Limit the frequency to the fixed range [f_1, f_2]
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
            if f_1 != 0 and f_2 != 0:
                we = we[ind_flow:ind_fhigh]  # Limit the frequency to a fixed range f_1, f_2
            features.append(we)
            record.insert(3, we)
        elif feature == 'chroma':
            chroma = librosa.feature.chroma_cqt(y=audiodata, sr=fs)
            # chroma = librosa.feature.chroma_stft(y=data, sr=fs)
            chroma = scale(chroma, axis=1)
            features.append(chroma)
            record.insert(3, chroma)
    # print(np.shape(features))

    features = TSNE().fit_transform(features)
    learners = Clustering(features, [])

    if alg =='DBSCAN':
        print('\nDBSCAN--------------------------------------')
        model_dbscan = learners.DBscan(eps=0.3, min_samples=3)
        predicted_labels = model_dbscan.labels_
        clusters = set(model_dbscan.labels_)

    elif alg == 'Birch':
        print('\nBirch----------------------------------------')
        if not n_clusters:
            model_birch = learners.birch(threshold=0.5, n_clusters=n_clusters)
        else:
            model_birch = learners.birch(threshold=0.88, n_clusters=None)
        predicted_labels = model_birch.labels_
        clusters = set(model_birch.labels_)

    if alg == 'agglomerative':
        print('\nAgglomerative Clustering----------------------')
        if not n_clusters:
            model_agg = learners.agglomerativeClustering(n_clusters=None, compute_full_tree=True,
                                                         distance_threshold=0.5, linkage='complete')
        else:
            model_agg = learners.agglomerativeClustering(n_clusters=n_clusters, compute_full_tree=False,
                                                         distance_threshold=None, linkage='complete')
        # Either set n_clusters=None and compute_full_tree=T or distance_threshold=None

        model_agg.fit_predict(learners.features)
        predicted_labels = model_agg.labels_
        clusters = set(model_agg.labels_)

    # print('predicted labels\n', predicted_labels)
    # print('clusters:', clusters)
    # print('# clusters :', len(clusters))

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
    from statistics import mode
    for i in range(len(labels)):
        try:
            labels[i] = mode(labels[i])
        except:
            labels[i] = labels[i][0]

    # add the features
    for record in clustered_dataset:
        record.insert(2, [])
        for rec in dataset:
            if record[:2] == rec[:2]:
                record[2].append(rec[3])

    # make the labels continous
    ulabels = list(set(labels))
    # print('ulabels:', ulabels)
    nclasses = len(ulabels)
    dic = []
    for i in range(nclasses):
        dic.append((ulabels[i], i))

    # print('[old, new] labels')
    dic = dict(dic)
    # print(dic)

    # add the labels
    for i in range(len(clustered_dataset)):
        clustered_dataset[i].insert(3, dic[labels[i]])

    # for record in clustered_dataset:
    #     print(record[3])

    clustercentres = getClusterCenters(clustered_dataset, nclasses)

    return clustered_dataset, fs, nclasses, clustercentres


def getClusterCenters(clusters, nclasses):
    """
    Computer cluster-centres
    :param clusters: clustered segments, a list of lists, each sublist represents a
                        segment [parent_audio_file, [segment], [features], class_label]
    :param nclasses: number of clusters, cluster labels are always 0, 1, 2, ..., nclasses-1
    :return:
    """
    # I simply compute the mean of each cluster here
    clustercentres = {}
    fc = []
    for c in range(nclasses):
        for seg in clusters:
            if seg[-1] == c:
                for f in seg[-2]:
                    fc.append(f)
        clustercentres[c] = np.mean(fc, axis=0)

    return clustercentres
