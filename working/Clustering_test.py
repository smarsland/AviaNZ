# Clustering_test.py
#
# Cluster testing

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

import Clustering
import numpy as np
import pandas as pd
import os, wavio
import librosa
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def testClustering():
    # Simple test using Iris data
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    data = load_iris()
    learners = Clustering.Clustering(data.data, data.target)

    print('**************Iris dataset ******************')

    print('\nK-means-------------------------------------')
    model = learners.kMeans(n_clusters=3)
    learners.clusteringScore1(learners.targets, model.labels_)

    print('\nMini batch K-means--------------------------')
    model = learners.miniBatchKmeans(n_clusters=3)
    learners.clusteringScore1(learners.targets, model.labels_)

    print('\nDBSCAN--------------------------------------')
    model = learners.DBscan(eps=0.5, min_samples=5)
    learners.clusteringScore1(learners.targets, model.labels_)
    # plot
    # core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
    # core_samples_mask[model.core_sample_indices_] = True
    # unique_labels = set(model.labels_)
    # colors = [plt.cm.Spectral(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (model.labels_ == k)
    #
    #     xy = learners.features[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
    #
    #     xy = learners.features[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=6)
    # plt.title('DBSCAN')
    # plt.show()

    print('\nBirch----------------------------------------')
    model = learners.birch(threshold=0.95, n_clusters=None)
    learners.clusteringScore1(learners.targets, model.labels_)
    print('# clusters', len(set(model.labels_)))

    print('\nSpectral Clustering--------------------------')
    model = learners.spectralClustering()
    learners.clusteringScore1(learners.targets, model.labels_)

    print('\nMeanShift Clustering-------------------------')
    model = learners.meanShift()
    learners.clusteringScore1(learners.targets, model.labels_)

    print('\nAgglomerative Clustering----------------------')
    model = learners.agglomerativeClustering(n_clusters=None, distance_threshold=1, compute_full_tree=True,
                                             linkage='complete')
    learners.clusteringScore1(learners.targets, model.labels_)
    # spanner = learners.get_cluster_spanner(model)
    # newick_tree = learners.build_Newick_tree(model.children_, model.n_leaves_, learners.features, model.labels_, spanner)
    #
    # tree = ete3.Tree(newick_tree)
    # tree.show()

    print('\nGMM------------------------------------------')
    model = learners.GMM(n_components=3)
    learners.clusteringScore1(learners.targets, model.labels_)

    print('\nAffinity Propagation--------------------------')
    model = learners.affinityPropagation()
    learners.clusteringScore1(learners.targets, model.labels_)


def cluster_ruru(sampRate):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    import SignalProc
    import wavio

    d = pd.read_csv('D:\AviaNZ\Sound_Files\Denoising_paper_data\Primary_dataset\\ruru\we2.tsv', sep="\t", header=None)
    data = d.values

    target = data[:, -1]
    fnames = data[:, 0]
    data = data[:, 1:-1]
    # dim reduction before clustering
    # pca = PCA(n_components=0.9)
    # data = pca.fit_transform(data)
    data = TSNE().fit_transform(data)
    learners = Clustering.Clustering(data, target)

    print('\n**************Ruru dataset******************')
    # Only choose algorithms that does not require n_clusters
    m = []
    print('\nDBSCAN--------------------------------------')
    model_dbscan = learners.DBscan(eps=0.5, min_samples=5)
    # print(model_dbscan.labels_)
    print('# clusters', len(set(model_dbscan.labels_)))
    m.append(learners.clusteringScore1(learners.targets, model_dbscan.labels_))

    print('\nBirch----------------------------------------')
    model_birch = learners.birch(threshold=0.88, n_clusters=None)
    # print(model_birch.labels_)
    print('# clusters', len(set(model_birch.labels_)))
    m.append(learners.clusteringScore1(learners.targets, model_birch.labels_))

    print('\nAgglomerative Clustering----------------------')
    model_agg = learners.agglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=4.4,
                                             linkage='complete')    # Either set n_clusters=None and compute_full_tree=T
                                                                    # or distance_threshold=None
    model_agg.fit_predict(learners.features)
    # print(model_agg.labels_)
    print('# clusters', len(set(model_agg.labels_)))
    m.append(learners.clusteringScore1(learners.targets, model_agg.labels_))

    print('\nAffinity Propagation--------------------------')
    model_aff = learners.affinityPropagation(damping=0.8, max_iter=400, convergence_iter=50)
    # print(model_aff.labels_)
    print('# clusters', len(set(model_aff.labels_)))
    m.append(learners.clusteringScore1(learners.targets, model_aff.labels_))

    best_m = np.argmax(m, axis=0).tolist()      # Get algorithm with the best performance on each index
    best_alg = max(set(best_m), key=best_m.count)   # Get the overall best alg

    # Analysis
    if best_alg == 0:
        model_best = model_dbscan
        print('\n***best clustering by: DBSCAN')
        print('predicted:\n', model_dbscan.labels_)
        print('actual:\n', learners.targets)
    elif best_alg == 1:
        model_best = model_birch
        print('\n***best clustering by: Birch')
        print('predicted:\n', model_birch.labels_)
        print('actual:\n', learners.targets)
    elif best_alg == 2:
        model_best = model_agg
        print('\n***best clustering by: Agglomerative')
        print('predicted:\n', model_agg.labels_)
        print('actual:\n', learners.targets)

    elif best_alg == 3:
        model_best = model_aff
        print('\n***best clustering by: Affinity')
        print('predicted:\n', model_aff.labels_)
        print('actual:\n', learners.targets)

    # plot the examples using the best clustering model
    # n_clusters = len(set(model_best.labels_))
    # get indices and plot them
    labels = list(set(model_best.labels_))

    app = QtGui.QApplication([])

    for label in labels:

        inds = np.where(model_best.labels_ == label)[0].tolist()

        mw = QtGui.QMainWindow()
        mw.show()
        mw.resize(1200, 800)

        win = pg.GraphicsLayoutWidget()
        mw.setCentralWidget(win)

        row = 0
        col = 0

        for i in inds:
            wavobj = wavio.read(fnames[i])
            fs = wavobj.rate
            audiodata = wavobj.data
            if audiodata.dtype is not 'float':
                audiodata = audiodata.astype('float')
            if np.shape(np.shape(audiodata))[0] > 1:
                audiodata = audiodata[:, 0]

            if fs != sampRate:
                audiodata = librosa.core.audio.resample(audiodata, fs, sampRate)
                fs = sampRate

            sp = SignalProc.SignalProc(audiodata, fs, 128, 128)
            sg = sp.spectrogram(audiodata, multitaper=False)

            vb = win.addViewBox(enableMouse=False, enableMenu=False, row=row, col=col, invertX=True)
            vb2 = win.addViewBox(enableMouse=False, enableMenu=False, row=row+1, col=col)
            im = pg.ImageItem(enableMouse=False)
            txt = fnames[i].split("/")[-1][:-4]
            lbl = pg.LabelItem(txt, rotateAxis=(1,0), angle=179)
            vb.addItem(lbl)
            vb2.addItem(im)
            im.setImage(sg)
            im.setBorder('w')
            mw.setWindowTitle("Class " + str(label) + ' - ' + str(np.shape(inds)[0]) + ' calls')

            if row == 8:
                row = 0
                col += 1
            else:
                row += 2

        QtGui.QApplication.instance().exec_()


def loadFile(filename, duration=0, offset=0, fs=0, denoise=False, f1=0, f2=0):
    """
    Read audio file and preprocess as required
    :param filename:
    :param fs:
    :param f1:
    :param f2:
    :return:
    """
    import WaveletFunctions
    import SignalProc
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


def within_cluster_dist(dir):
    """ First attempt to generate clusters by dist
        use more-pork (200 data points) and tril (100 data points) syllable dataset
    """
    # find within/between cluster distance distance with mfcc+ and dtw
    features = []
    for root, dirs, files in os.walk(str(dir)):
        for filename in files:
            if filename.endswith('.wav'):
                filename = os.path.join(root, filename)
                print(filename)
                data, fs = loadFile(filename)
                mfcc = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=24, n_fft=2048, hop_length=512)
                mfcc = mfcc[1:, :]
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc = np.concatenate((mfcc, mfcc_delta), axis=0)
                mfcc = scale(mfcc, axis=1)
                # librosa.display.specshow(mfcc, sr=fs, x_axis='time')
                m = [i for sublist in mfcc for i in sublist]
                features.append(m)

    print(np.shape(features))
    # now calculate dtw
    dist = np.zeros((np.shape(features)[0], np.shape(features)[0]))
    shift = 0
    for i in range(np.shape(features)[0]):
        shift += 1
        for j in range(shift, np.shape(features)[0]):
            d, wp = librosa.sequence.dtw(features[i], features[j], metric='euclidean')
            dist[i, j] = d[d.shape[0] - 1][d.shape[1] - 1]
            print(i, j, dist[i, j])

    print(dist)
    print('max:', np.max(dist))
    print('51% percentile:', np.percentile(dist, 51))
    print('80% percentile:', np.percentile(dist, 80))
    print('90% percentile:', np.percentile(dist, 90))
    print('95% percentile:', np.percentile(dist, 95))

# within_cluster_dist('D:\AviaNZ\Sound_Files\Denoising_paper_data\Primary_dataset\\ruru')

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
                        # TODO: samples to seconds
                        syls = segment.joinGaps(syls, minlen_samples)             # Merge short segments
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
