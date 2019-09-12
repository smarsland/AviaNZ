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