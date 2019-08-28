# WaveletSegment.py
#
# Wavelet Segmentation

# Version 1.4 26/06/19
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
import matplotlib.pyplot as plt
import pandas as pd
import random
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
import xgboost as xgb

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

import json, time, os, math, csv, gc, wavio
import WaveletSegment
import librosa

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

# TODO:
# Put some stuff in here!
# Needs decision trees and recurrent NN as comparisons to Digby and Bagnall
# Random forests from Lasseck
# GMM from lots of people
# Plus whatever scikit-learn has :)
# Some deep learning stuff?
# xgboost

# Also, consider another module for HMMs and related -> syllable ordering, etc.

class Learning:
    # This class implements various learning algorithms for the AviaNZ interface
    # Mostly based on scikit-learn

    def __init__(self, features, labels, testFraction=0.4):
        features = StandardScaler().fit_transform(features)
        if testFraction == 1:
            self.test = features
            self.testTgt = labels
        elif testFraction == 0:
            self.train = features
            self.trainTgt = labels
        else:
            self.train, self.test, self.trainTgt, self.testTgt = train_test_split(features, labels,
                                                                                  test_size=testFraction, shuffle=True)

    def performTest(self, model):
        testOut = model.predict(self.test)
        testError = np.mean(self.testTgt.ravel() == testOut.ravel()) * 100

        print("Testing error: ", testError)
        CM = confusion_matrix(self.testTgt, testOut)
        print(CM)
        print(CM[1][1], CM[0][1], CM[0][0], CM[1][0])

    def trainMLP(self, structure=(100,), learningrate=0.001, solver='adam', epochs=200, alpha=1, shuffle=True,
                 early_stopping=False):
        model = MLPClassifier(hidden_layer_sizes=structure, solver=solver, alpha=alpha, max_iter=epochs,
                              learning_rate_init=learningrate, shuffle=shuffle, early_stopping=early_stopping)
        model.fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model

    def trainKNN(self, K=5):
        model = KNeighborsClassifier(K)
        model.fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model

    def trainSVM(self, kernel='rbf', C=1, gamma='auto'):
        model = SVC(kernel=kernel, C=C, gamma=gamma)
        model.fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model

    def trainGP(self,kernel="RBF", param=1.0):
        model = GaussianProcessClassifier(1.0 * RBF(1.0))
        model.fit(self.train,self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ",trainError)

        return model

    def trainDecisionTree(self, maxDepth=5):
        model = DecisionTreeClassifier(max_depth=maxDepth)
        model.fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model

    def trainRandomForest(self, maxDepth=5, nTrees=10, maxFeatures=2):
        model = RandomForestClassifier(max_depth=maxDepth, n_estimators=nTrees, max_features=maxFeatures)
        model.fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model

    def trainBoosting(self, n_estimators=100):
        model = AdaBoostClassifier(n_estimators=n_estimators)
        model.fit(self.train,self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ",trainError)

        return model

    def trainXGBoost(self, nRounds=10):
        model = xgb.XGBClassifier().fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model

    def trainGMM(self, nClasses=2, covType='spherical', maxIts=20):
        model = GaussianMixture(n_components=nClasses, covariance_type=covType, max_iter=maxIts)
        model.means_init = np.array([self.train[self.trainTgt == i].mean(axis=0) for i in range(nClasses)])
        model.fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model

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


#For each sound class an ensemble of randomized decision trees (sklearn.ensemble.ExtraTreesRegressor) is applied. The number of estimators is chosen to be twice the number of selected features per class but not greater than 500. The winning solution considers 4 features when looking for the best split and requires a minimum of 3 samples to split an internal node.

class Validate:
    # This class implements cross-validation and learning curves
    # based on scikit-learn

    def __init__(self, estimator, title, features, labels, param_name=None, param_range=None,scoring=None):
        features = StandardScaler().fit_transform(features)
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a v
        self.cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
        self.estimator = estimator
        self.title = title
        self.X = features
        self.y = labels
        self.param_name = param_name
        self.param_range = param_range
        self.scoring = scoring

    def plot_validation_curve(self):
        '''
        Plot validation curve for different hyper-parameter value
        '''
        train_scores, test_scores = validation_curve(self.estimator, self.X, self.y, param_name=self.param_name,
                                                     param_range=self.param_range,
                                                     cv=self.cv, scoring=self.scoring, n_jobs=1)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.title(self.title)
        plt.xlabel(self.param_name)
        plt.ylabel(self.scoring)
        plt.ylim(0.0, 1.1)
        lw = 2

        plt.semilogx(self.param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(self.param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(self.param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(self.param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")
        return plt

    def plot_learning_curve(self, ylim=None, cv=None,
                            n_jobs=None, train_sizes=np.linspace(.01, 1.0, 10)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - :term:`CV splitter`,
              - An iterable yielding (train, test) splits as arrays of indices.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        train_sizes : array-like, shape (n_ticks,), dtype float or int
            Relative or absolute numbers of training examples that will be used to
            generate the learning curve. If the dtype is float, it is regarded as a
            fraction of the maximum size of the training set (that is determined
            by the selected validation method), i.e. it has to be within (0, 1].
            Otherwise it is interpreted as absolute sizes of the training sets.
            Note that for classification the number of samples usually have to
            be big enough to contain at least one sample from each class.
            (default: np.linspace(0.1, 1.0, 5))
        """
        plt.figure()
        plt.title(self.title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel(self.scoring)
        train_sizes, train_scores, test_scores = learning_curve(
            self.estimator, self.X, self.y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=self.scoring)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt


def testClustering():
    # Simple test using Iris data
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    data = load_iris()
    learners = Clustering(data.data, data.target)

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
    learners = Clustering(data, target)

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

def cluster_by_dist(dir, feature='mfcc', n_mels=24, fs=0, minlen=0.2, f_1=0, f_2=0, denoise=False, single=False,
                    displayallinone=True, distance='dtw'):
    """
    Given wav + annotation files,
        1) identify syllables using median clipping
        2) generate features
        3) calculate DTW distances and decide class/ generate new class
    :param n_mels: number of mel coeff
    :param fs: prefered sampling frequency
    :param minlen: min syllable length in secs
    :param f_1:
    :param f_2:
    :param denoise:
    :param single:
    :param displayallinone:
    :param distance:
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
                with open(os.path.join(root, file + '.data')) as f1:
                    segments = json.load(f1)
                    for seg in segments:
                        if seg[0] == -1:
                            continue
                        else:
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
    max_clusters = 10
    n_clusters = 0
    clusters = []
    for root, dirs, files in os.walk(str(dir)):
        for file in files:
            if file.endswith('.wav') and file+'.data' in files:
                # Read the annotation
                with open(os.path.join(root, file + '.data')) as f1:
                    segments = json.load(f1)

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
                        newclass= class_create(label=n_clusters+1, syl=syls, features=f, f_low=seg[2], f_high=seg[3],
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
                            if (clusters[c]["d"] != 0) and \
                                    min_ds[c] < (clusters[c]["d"] + clusters[c]["d"] * 0.1):
                                print('**Case 2: Found a match with a class > one syllable')
                                print('Class ', clusters[c]["label"], ', dist ', min_ds[c])
                                # Update this class
                                clusters[c] = class_update(cluster=clusters[c], newfeatures=f, newf_low=seg[2],
                                                           newf_high=seg[3], newsyl=syls,
                                                           newsegs=(os.path.join(root, file), seg), single=single,
                                                           dist_method=distance)
                                matched = True
                                break       # found match, exit from the search, go to the next segment

                            elif c < len(clusters)-1:
                                continue    # continue to the next class

                    # Checked most of the classes by now, if still no match found, check the classes with only one
                    # syllable.
                    # Note the arbitrary thr, 50 for the case with clusters with only one data point.
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
                                                           newsegs=(os.path.join(root, file), seg), single=single,
                                                           dist_method=distance)
                                matched = True
                                break    # Break the search and go to the next segment

                    # No match found yet, check the max clusters
                    if not matched:
                        if n_clusters == max_clusters:
                            print('**Case 4: Reached max classes, therefore adding current seg to the closest '
                                  'class... ')
                            # min_ind = np.argmin(min_ds)
                            # classes are sorted in accesnding order of distance already
                            for c in range(len(clusters)):
                                if min_ds[c] <= 4 * clusters[c]["d"] or clusters[c]["d"] == 0:
                                    print('Class ', clusters[c]["label"], ', dist ', min_ds[c],
                                          '(in-class distance:', clusters[c]["d"], ')')
                                    # Update this class
                                    clusters[c] = class_update(cluster=clusters[c], newfeatures=f, newf_low=seg[2],
                                                               newf_high=seg[3], newsyl=syls,
                                                               newsegs=(os.path.join(root, file), seg), single=single,
                                                               dist_method=distance)
                                    matched = True
                                    break
                            if not matched:
                                print('Class ', clusters[0]["label"], ', dist ', min_ds[0],
                                      '(in-class distance:', clusters[0]["d"], ')')
                                # Update this class
                                clusters[0] = class_update(cluster=clusters[0], newfeatures=f, newf_low=seg[2],
                                                           newf_high=seg[3], newsyl=syls,
                                                           newsegs=(os.path.join(root, file), seg), single=single,
                                                           dist_method=distance)
                                matched = True
                            continue    # Continue to next segment

                    #  If still no luck, create a new class
                    if not matched:
                        print('**Case 5: None of Case 1-4')
                        newclass = class_create(label=n_clusters+1, syl=syls, features=f, f_low=seg[2], f_high=seg[3],
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
            clustered_segs.append([clusters[c]['segs'][s][0], clusters[c]['segs'][s][1], clusters[c]['label']])

    # Clustered segments
    print('\n\n################### Clustered segments ############################')
    for s in clustered_segs:
        print(s)
    # return clustered_segs, fs

    # Display the segs
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    if displayallinone:
        app = QtGui.QApplication([])

        mw = QtGui.QMainWindow()
        mw.show()
        mw.resize(1200, 800)
        mw.setWindowTitle("Clustered segments (each row is a Class) - Feature: " + feature + " ; Denoise: " + str(denoise) +
                          " ; Bandpass: " + str(f_1) + "-" + str(f_2))

        win = pg.GraphicsLayoutWidget()
        mw.setCentralWidget(win)
        row = 0

        for c in range(len(clusters)):
            col = 0
            for s in clusters[c]["segs"]:
                audiodata, _ = loadFile(s[0], s[1][1]-s[1][0], s[1][0], fs)
                sp = SignalProc.SignalProc(audiodata, fs, 512, 256)
                sg = sp.spectrogram(audiodata, multitaper=False)
                maxsg = np.min(sg)
                sg = np.abs(np.where(sg == 0, 0.0, 10.0 * np.log10(sg / maxsg)))
                # Make it readable
                minsg = np.min(sg)
                maxsg = np.max(sg)
                colourStart = (20 / 100.0 * 20 / 100.0) * (maxsg - minsg) + minsg
                colourEnd = (maxsg - minsg) * (1.0 - 20 / 100.0) + colourStart

                vb = win.addViewBox(enableMouse=False, enableMenu=False, row=row, col=col, invertX=True)
                vb2 = win.addViewBox(enableMouse=False, enableMenu=False, row=row+1, col=col)
                im = pg.ImageItem(enableMouse=False)
                vb2.addItem(im)
                im.setImage(sg)
                im.setBorder('w')
                im.setLevels([colourStart, colourEnd])

                txt = s[1][4][0]
                # txt = s[0].split('\\')[-1]+'-'+s[1][4][0]
                lbl = pg.LabelItem(txt, rotateAxis=(1,0), angle=179)
                vb.addItem(lbl)
                col += 1
            row += 2

        QtGui.QApplication.instance().exec_()

    else:
        app = QtGui.QApplication([])
        for c in range(len(clusters)):
            print(clusters[c]["label"])
            # app = QtGui.QApplication([])

            mw = QtGui.QMainWindow()
            mw.show()
            mw.resize(1200, 800)

            win = pg.GraphicsLayoutWidget()
            mw.setCentralWidget(win)
            row = 0
            col = 0
            for s in clusters[c]["segs"]:
                audiodata, _ = loadFile(s[0], s[1][1]-s[1][0], s[1][0], fs)
                sp = SignalProc.SignalProc(audiodata, fs, 512, 256)
                sg = sp.spectrogram(audiodata, multitaper=False)
                maxsg = np.min(sg)
                sg = np.abs(np.where(sg == 0, 0.0, 10.0 * np.log10(sg / maxsg)))
                # Make it readable
                minsg = np.min(sg)
                maxsg = np.max(sg)
                colourStart = (20 / 100.0 * 20 / 100.0) * (maxsg - minsg) + minsg
                colourEnd = (maxsg - minsg) * (1.0 - 20 / 100.0) + colourStart

                vb = win.addViewBox(enableMouse=False, enableMenu=False, row=row, col=col, invertX=True)
                vb2 = win.addViewBox(enableMouse=False, enableMenu=False, row=row + 1, col=col)
                im = pg.ImageItem(enableMouse=False)
                vb2.addItem(im)
                im.setImage(sg)
                im.setBorder('w')
                im.setLevels([colourStart, colourEnd])

                txt = s[1][4][0]
                lbl = pg.LabelItem(txt, rotateAxis=(1, 0), angle=179)
                vb.addItem(lbl)
                if row == 6:
                    row = 0
                    col += 1
                else:
                    row += 2
            QtGui.QApplication.instance().exec_()


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
        features = [features[len(features)//2]]

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


def class_update(cluster, newfeatures, newf_low, newf_high, newsyl, newsegs, single, dist_method='dtw'):
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
    cluster["segs"].append(newsegs)
    print('Updated Class ', "'", cluster["label"], "'" '\tin-class_d: ',
          cluster["d"], '\tf_low: ', cluster["f_low"], '\tf_high: ',
          cluster["f_high"])
    return cluster


def cluster_by_agg(dir, feature='mfcc', n_mels=24, fs=0, minlen=0.2, f_1=0, f_2=0, denoise=False,
                    displayallinone=True, alg='agglomerative'):
    """
    Given wav + annotation files,
        1) identify syllables using a general method (median clipping)
        2) make them to fixed-length by padding or clipping
        3) use existing clustering algorithems
    :param n_mels: number of mel coeff
    :param fs: prefered sampling frequency
    :param minlen: min syllable length in secs
    :param f_1:
    :param f_2:
    :param denoise:
    :param displayallinone:
    :return: possible clusters
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    import Segment
    import SignalProc

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
                with open(os.path.join(root, file + '.data')) as f1:
                    segments = json.load(f1)
                    for seg in segments:
                        if seg[0] == -1:
                            continue
                        else:
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

    if f_2 < f_1:
        f_2 = np.mean(highlist)

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

    # Find the syllables
    dataset = []
    for root, dirs, files in os.walk(str(dir)):
        for file in files:
            if file.endswith('.wav') and file+'.data' in files:
                # Read the annotation
                with open(os.path.join(root, file + '.data')) as f1:
                    segments = json.load(f1)

                # Now find syllables within each segment, median clipping
                for seg in segments:
                    if seg[0] == -1:
                        continue
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
                    print('\nCurrent:', seg, '--> Median clipping ', syls)
                    for syl in syls:
                        dataset.append([os.path.join(root, file), seg, syl])

    # Make them fixed-length
    lengths = []
    for data in dataset:
        lengths.append(data[2][1]-data[2][0])
    print('min: ', min(lengths), ' max: ', max(lengths), ' median: ', np.median(lengths))
    duration = np.median(lengths)   # This is the fixed length of a syllable, if a syllable too long clip it otherwise
                                    # padding with zero
    # Now calculate the features
    for record in dataset:
        if record[2][1]-record[2][0] > duration:
            # get the middle 'duration' secs
            middle = (record[2][1]+record[2][0]) / 2
            record[2][0] = middle - duration/2
            record[2][1] = middle + duration / 2

    lengths = []
    for data in dataset:
        lengths.append(data[2][1] - data[2][0])
    print('min: ', min(lengths), ' max: ', max(lengths), ' median: ', np.median(lengths))

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
        elif feature == 'we':  # Wavelet Energy
            ws = WaveletSegment.WaveletSegment(spInfo=[])
            we = ws.computeWaveletEnergy(data=audiodata, sampleRate=fs, nlevels=5, wpmode='new')
            we = we.mean(axis=1)
            if f_1 != 0 and f_2 != 0:
                we = we[ind_flow:ind_fhigh]  # Limit the frequency to a fixed range f_1, f_2
            features.append(we)
        elif feature == 'chroma':
            chroma = librosa.feature.chroma_cqt(y=audiodata, sr=fs)
            # chroma = librosa.feature.chroma_stft(y=data, sr=fs)
            chroma = scale(chroma, axis=1)
            features.append(chroma)
    print(np.shape(features))

    features = TSNE().fit_transform(features)
    learners = Clustering(features, [])

    if alg =='DBSCAN':
        print('\nDBSCAN--------------------------------------')
        model_dbscan = learners.DBscan(eps=0.3, min_samples=3)
        predicted_labels = model_dbscan.labels_
        print('# clusters', len(set(model_dbscan.labels_)))

    elif alg == 'Birch':
        print('\nBirch----------------------------------------')
        model_birch = learners.birch(threshold=0.88, n_clusters=None)
        predicted_labels = model_birch.labels_
        print('# clusters', len(set(model_birch.labels_)))

    if alg == 'agglomerative':
        print('\nAgglomerative Clustering----------------------')
        # model_agg = learners.agglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=0.5,
        #                                          linkage='complete')
        model_agg = learners.agglomerativeClustering(n_clusters=6, compute_full_tree=False, distance_threshold=None,
                                                     linkage='complete')
        # Either set n_clusters=None and compute_full_tree=T or distance_threshold=None

        model_agg.fit_predict(learners.features)
        predicted_labels = model_agg.labels_
        print('# clusters', len(set(model_agg.labels_)))

    print(predicted_labels)
    # Attach the label to each syllable in the dataset
    for i in range(len(predicted_labels)):
        dataset[i].insert(3, predicted_labels[i])

    for record in dataset:
        print(record)

    # Majority voting
    clustered_dataset = []
    for record in dataset:
        if record[:2] not in clustered_dataset:
            clustered_dataset.append(record[:2])

    labels = [[] for i in range(len(clustered_dataset))]

    for i in range(len(predicted_labels)):
        ind = clustered_dataset.index(dataset[i][:2])
        labels[ind].append(predicted_labels[i])

    from statistics import mode
    for i in range(len(labels)):
        try:
            labels[i] = mode(labels[i])
        except:
            labels[i] = labels[i][0]

    for i in range(len(clustered_dataset)):
        print(clustered_dataset[i], labels[i])

    classes = set(labels).__len__()

    for i in range(len(clustered_dataset)):
        clustered_dataset[i].insert(2, labels[i])

    return clustered_dataset, fs, classes

    # # Sort the clusters and display
    # classes = set(labels)
    #
    # # Display the segs
    # import pyqtgraph as pg
    # from pyqtgraph.Qt import QtCore, QtGui
    # if displayallinone:
    #     app = QtGui.QApplication([])
    #
    #     mw = QtGui.QMainWindow()
    #     mw.show()
    #     mw.resize(1200, 800)
    #     mw.setWindowTitle(
    #         "Clustered segments (each row is a Class) - Feature: " + feature + " ; Denoise: " + str(denoise) +
    #         " ; Bandpass: " + str(f_1) + "-" + str(f_2))
    #
    #     win = pg.GraphicsLayoutWidget()
    #     mw.setCentralWidget(win)
    #     row = 0
    #
    #     for c in classes:
    #         col = 0
    #         indc = np.where(labels == c)[0].tolist()
    #         print('Class ', c, ': ', np.shape(indc)[0], ' segments')
    #         clusterc = []
    #         for i in indc:
    #             print(clustered_dataset[i])
    #             clusterc.append(clustered_dataset[i])
    #         for x in clusterc:
    #             audiodata, _ = loadFile(x[0], x[1][1]-x[1][0], x[1][0], fs)
    #             sp = SignalProc.SignalProc(audiodata, fs, 1024, 512)
    #             sg = sp.spectrogram(audiodata, multitaper=False)
    #             maxsg = np.min(sg)
    #             sg = np.abs(np.where(sg == 0, 0.0, 10.0 * np.log10(sg / maxsg)))
    #             # Make it readable
    #             minsg = np.min(sg)
    #             maxsg = np.max(sg)
    #             colourStart = (20 / 100.0 * 20 / 100.0) * (maxsg - minsg) + minsg
    #             colourEnd = (maxsg - minsg) * (1.0 - 20 / 100.0) + colourStart
    #
    #             vb = win.addViewBox(enableMouse=False, enableMenu=False, row=row, col=col, invertX=True)
    #             vb2 = win.addViewBox(enableMouse=False, enableMenu=False, row=row + 1, col=col)
    #             im = pg.ImageItem(enableMouse=False)
    #             vb2.addItem(im)
    #             im.setImage(sg)
    #             im.setBorder('w')
    #             im.setLevels([colourStart, colourEnd])
    #
    #             txt = x[1][4][0]
    #             # txt = s[0].split('\\')[-1]+'-'+s[1][4][0]
    #             lbl = pg.LabelItem(txt, rotateAxis=(1, 0), angle=179)
    #             vb.addItem(lbl)
    #             col += 1
    #         row += 2
    #
    #     QtGui.QApplication.instance().exec_()
    # else:
    #     app = QtGui.QApplication([])
    #     for c in classes:
    #         indc = np.where(labels == c)[0].tolist()
    #         print('Class ', c, ': ', np.shape(indc)[0], ' segments')
    #         clusterc = []
    #         for i in indc:
    #             print(clustered_dataset[i])
    #             clusterc.append(clustered_dataset[i])
    #         mw = QtGui.QMainWindow()
    #         mw.show()
    #         mw.resize(1200, 800)
    #
    #         win = pg.GraphicsLayoutWidget()
    #         mw.setCentralWidget(win)
    #         row = 0
    #         col = 0
    #
    #         for x in clusterc:
    #             audiodata, _ = loadFile(x[0], x[1][1] - x[1][0], x[1][0], fs)
    #             sp = SignalProc.SignalProc(audiodata, fs, 1024, 512)
    #             sg = sp.spectrogram(audiodata, multitaper=False)
    #             maxsg = np.min(sg)
    #             sg = np.abs(np.where(sg == 0, 0.0, 10.0 * np.log10(sg / maxsg)))
    #             # Make it readable
    #             minsg = np.min(sg)
    #             maxsg = np.max(sg)
    #             colourStart = (20 / 100.0 * 20 / 100.0) * (maxsg - minsg) + minsg
    #             colourEnd = (maxsg - minsg) * (1.0 - 20 / 100.0) + colourStart
    #
    #             # vb = win.addViewBox(enableMouse=False, enableMenu=False, row=row, col=col, invertX=True)
    #             vb2 = win.addViewBox(enableMouse=False, enableMenu=False, row=row + 1, col=col)
    #             im = pg.ImageItem(enableMouse=False)
    #             vb2.addItem(im)
    #             im.setImage(sg)
    #             im.setBorder('w')
    #             im.setLevels([colourStart, colourEnd])
    #
    #             # txt = x[1][0][4]
    #             # lbl = pg.LabelItem(txt, rotateAxis=(1, 0), angle=179)
    #             # vb.addItem(lbl)
    #             if row == 6:
    #                 row = 0
    #                 col += 1
    #             else:
    #                 row += 2
    #         QtGui.QApplication.instance().exec_()


# cluster_by_dist('D:\AviaNZ\Sound_Files\demo\morepork', feature='we', denoise=False)
# cluster_by_agg('D:\AviaNZ\Sound_Files\demo\morepork', feature='we')


def testLearning1():
    # Very simple test
    import Learning
    from sklearn.datasets import make_classification
    features, labels = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1,
                                           n_clusters_per_class=1)
    learners = Learning.Learning(features, labels)

    model = learners.trainMLP()
    learners.performTest(model)
    model = learners.trainKNN()
    learners.performTest(model)
    model = learners.trainSVM()
    learners.performTest(model)
    model = learners.trainGP()
    learners.performTest(model)
    model = learners.trainDecisionTree()
    learners.performTest(model)
    model = learners.trainRandomForest()
    learners.performTest(model)
    model = learners.trainBoosting()
    learners.performTest(model)
    model = learners.trainXGBoost()
    learners.performTest(model)
    model = learners.trainGMM()
    learners.performTest(model)


def testLearning2():
    # Iris data
    import Learning
    # from sklearn.datasets import load_iris
    # data = load_iris()

    d = pd.read_csv('D:\AviaNZ\Sound_Files\Denoising_paper_data\Primary_dataset\kiwi\mfcc.tsv', sep="\t", header=None)
    data = d.values

    target = data[:, -1]
    data = data[:, 0:-1]
    learners = Learning.Learning(data, target)

    model = learners.trainMLP()
    learners.performTest(model)
    model = learners.trainKNN()
    learners.performTest(model)
    model = learners.trainSVM()
    learners.performTest(model)
    model = learners.trainGP()
    learners.performTest(model)
    model = learners.trainDecisionTree()
    learners.performTest(model)
    model = learners.trainRandomForest()
    learners.performTest(model)
    model = learners.trainBoosting()
    learners.performTest(model)
    model = learners.trainXGBoost()
    learners.performTest(model)
    model = learners.trainGMM()
    learners.performTest(model)

# testLearning2()

def learninigCurve(dataFile, clf, score=None):
    ''' Choose a classifier and plot the learning curve
    dataFile: dataset including features and targets
    clf: classifier to consider
    score: customise the scoring (default in sklearn is 'accuracy')
    '''

    # Let's use fB(F2) score
    if score is None:
        from sklearn.metrics import fbeta_score, make_scorer
        score = make_scorer(fbeta_score, beta=2)
    d = pd.read_csv(dataFile, sep="\t", header=None)
    data = d.values

    # Balance the data set
    targets = data[:, -1]
    data = data[:, 0:-1]
    posTargetInd = np.where(targets == 1)
    negTargetInd = np.where(targets == 0)
    # randomly select n negative rows
    n = min(np.shape(posTargetInd)[1], np.shape(negTargetInd)[1])
    posTargetInd = posTargetInd[0].tolist()
    posTargetInd = random.sample(posTargetInd, n)
    negTargetInd = negTargetInd[0].tolist()
    negTargetInd = random.sample(negTargetInd, n)
    inds = posTargetInd + negTargetInd
    data = data[inds, :]
    targets = targets[inds]
    indices = np.arange(targets.shape[0])
    np.random.shuffle(indices)
    data, targets = data[indices], targets[indices]
    if clf == 'GaussianNB':
        from sklearn.naive_bayes import GaussianNB
        estimator = GaussianNB()
    elif clf == 'SVM':
        from sklearn.svm import SVC
        estimator = SVC(gamma=0.0077)
    elif clf == 'MLP':
        estimator = MLPClassifier(hidden_layer_sizes=(250,), max_iter=100, early_stopping=True)
    elif clf == 'kNN':
        estimator = KNeighborsClassifier(3)
    elif clf == 'GP':
        estimator = GaussianProcessClassifier(1.0 * RBF(1.0))
    elif clf == 'DT':
        estimator = DecisionTreeClassifier(max_depth=5)
    elif clf == 'RF':
        estimator = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2)
    elif clf == 'Boost':
        estimator = AdaBoostClassifier()
    elif clf == 'XGB':
        estimator = xgb.XGBClassifier()
    elif clf == 'GMM':
        estimator = GaussianMixture(n_components=2, covariance_type='spherical', max_iter=20)
    title = "Learning Curves - " + clf
    v = Validate(estimator, title, data, targets, scoring=score)
    plt = v.plot_learning_curve()
    plt.show()


def validationCurve(dataFile, clf, nClasses=2, score=None):
    ''' Choose a classifier and plot the validation curve
    Score against different values for a selected hyperparameter to see the influence of a single hyperparameter
    dataFile: dataset including features and targets
    clf: classifier to consider
    score: customise the scoring (default in sklearn is 'accuracy')
    '''

    # Let's use fB(F2) score
    if score is None:
        from sklearn.metrics import fbeta_score, make_scorer
        score = make_scorer(fbeta_score, beta=2)
    d = pd.read_csv(dataFile, sep="\t", header=None)
    data = d.values

    # Balance the data set
    targets = data[:, -1]
    data = data[:, 0:-1]
    if nClasses == 2:
        posTargetInd = np.where(targets == 1)
        negTargetInd = np.where(targets == 0)
        # randomly select n negative rows
        n = min(np.shape(posTargetInd)[1], np.shape(negTargetInd)[1])
        posTargetInd = posTargetInd[0].tolist()
        posTargetInd = random.sample(posTargetInd, n)
        negTargetInd = negTargetInd[0].tolist()
        negTargetInd = random.sample(negTargetInd, n)
        inds = posTargetInd + negTargetInd
    elif nClasses == 3:
        c1TargetInd = np.where(targets == 0)    # c1=noise
        c2TargetInd = np.where(targets == 1)    # c2=male
        c3TargetInd = np.where(targets == 2)    # c3=female
        # randomly select n negative rows
        n = min(np.shape(c1TargetInd)[1], np.shape(c2TargetInd)[1], np.shape(c3TargetInd)[1])
        c1TargetInd = c1TargetInd[0].tolist()
        c1TargetInd = random.sample(c1TargetInd, n)
        c2TargetInd = c2TargetInd[0].tolist()
        c2TargetInd = random.sample(c2TargetInd, n)
        c3TargetInd = c3TargetInd[0].tolist()
        c3TargetInd = random.sample(c3TargetInd, n)
        inds = c1TargetInd + c2TargetInd + c3TargetInd
    data = data[inds, :]
    targets = targets[inds]
    indices = np.arange(targets.shape[0])
    np.random.shuffle(indices)
    data, targets = data[indices], targets[indices]
    if clf == 'GaussianNB':
        from sklearn.naive_bayes import GaussianNB
        estimator = GaussianNB()
    elif clf == 'SVM':
        estimator = SVC(C=1)
        param_name = "gamma"
        param_range = np.logspace(-6, 1, 10)
        # param_name = "C"
        # param_range = np.linspace(0.01, 1, 5)
    elif clf == 'MLP':
        estimator = MLPClassifier()
        param_name = "alpha"
        param_range = 10.0 ** -np.arange(1, 7)
        # param_name = "max_iter"
        # param_range = [100, 200, 300, 400, 500]
    elif clf == 'kNN':
        estimator = KNeighborsClassifier()
        param_name = "n_neighbors"
        param_range = [1, 2, 3, 4, 5, 6]
    elif clf == 'GP':
        estimator = GaussianProcessClassifier(1.0 * RBF(1.0))
    elif clf == 'DT':
        estimator = DecisionTreeClassifier(max_depth=5)
    elif clf == 'RF':
        estimator = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2)
    elif clf == 'Boost':
        estimator = AdaBoostClassifier()
    elif clf == 'XGB':
        estimator = xgb.XGBClassifier()
    elif clf == 'GMM':
        estimator = GaussianMixture(n_components=2, covariance_type='spherical', max_iter=20)

    title = "Validation Curves - " + clf
    v = Validate(estimator, title, data, targets, param_name=param_name, param_range=param_range, scoring=score)
    plt = v.plot_validation_curve()
    plt.show()


def fit_GridSearchCV(dataFile, clf, nClasses=2):
    '''
    Grid search over specified parameter values for a classifier and return the best
    :param dataFile: dataset including features and targets
    :param clf: classifier
    :param nClasses: 2 if binary classification (number of classes)
    :return: the best estimator
    '''
    d = pd.read_csv(dataFile, sep="\t", header=None)
    data = d.values

    # Balance the data set
    targets = data[:, -1]
    data = data[:, 0:-1]
    if nClasses==2:
        posTargetInd = np.where(targets == 1)
        negTargetInd = np.where(targets == 0)
        # randomly select n negative rows
        n = min(np.shape(posTargetInd)[1], np.shape(negTargetInd)[1])
        posTargetInd = posTargetInd[0].tolist()
        posTargetInd = random.sample(posTargetInd, n)
        negTargetInd = negTargetInd[0].tolist()
        negTargetInd = random.sample(negTargetInd, n)
        inds = posTargetInd + negTargetInd
    elif nClasses == 3:
        c1TargetInd = np.where(targets == 0)    # c1=noise
        c2TargetInd = np.where(targets == 1)    # c2=male
        c3TargetInd = np.where(targets == 2)    # c3=female
        # randomly select n negative rows
        n = min(np.shape(c1TargetInd)[1], np.shape(c2TargetInd)[1], np.shape(c3TargetInd)[1])
        c1TargetInd = c1TargetInd[0].tolist()
        c1TargetInd = random.sample(c1TargetInd, n)
        c2TargetInd = c2TargetInd[0].tolist()
        c2TargetInd = random.sample(c2TargetInd, n)
        c3TargetInd = c3TargetInd[0].tolist()
        c3TargetInd = random.sample(c3TargetInd, n)
        inds = c1TargetInd + c2TargetInd + c3TargetInd
    data = data[inds, :]
    targets = targets[inds]
    indices = np.arange(targets.shape[0])
    np.random.shuffle(indices)
    data, targets = data[indices], targets[indices]
    # Choose an estimator
    if clf == 'GaussianNB':
        from sklearn.naive_bayes import GaussianNB
        estimator = GaussianNB()
    elif clf == 'SVM':
        param_grid = {
            'C': np.linspace(0.01, 1, 4),
            'gamma': np.logspace(-6, 1, 5)
        }
        estimator = GridSearchCV(
            SVC(), param_grid=param_grid)
    elif clf == 'MLP':
        param_grid = {
            'hidden_layer_sizes': [(5, 5), (125,), (250,), (125, 5), (250, 2), (25, 5)],
            'max_iter': [100, 200, 300, 400, 500],
            # 'solver': ['lbfgs', 'sgd', 'adam'],
            # 'activation': ['identity', 'logistic', 'tanh', 'relu']
        }
        estimator = GridSearchCV(
            MLPClassifier(solver='adam', activation='tanh', learning_rate='constant', learning_rate_init=0.001, early_stopping=True, shuffle=True),
            param_grid=param_grid)
    elif clf == 'kNN':
        param_grid = {
            'n_neighbors': [3, 4, 5, 6, 10]
        }
        estimator = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid)
        # estimator = KNeighborsClassifier(3)
    elif clf == 'GP':
        estimator = GaussianProcessClassifier(1.0 * RBF(1.0))
    elif clf == 'DT':
        estimator = DecisionTreeClassifier(max_depth=5)
    elif clf == 'RF':
        estimator = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=2)
    elif clf == 'Boost':
        param_grid = {
            'n_estimators': [25, 50, 75, 100, 200]
        }
        estimator = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid)
    elif clf == 'XGB':
        estimator = xgb.XGBClassifier()
    elif clf == 'GMM':
        param_grid = {
            'n_components': [2, 3, 4, 5],
            'covariance_type': ['full', 'tied', 'diag', 'spherical'],
            'max_iter': [100, 200, 300, 400]
        }
        estimator = GridSearchCV(GaussianMixture(), param_grid=param_grid)
    estimator.fit(data, targets)
    print(estimator.best_estimator_)


def TrainClassifier(dir, species, feature, clf=None, pca=False):
    '''
    Use wavelet energy/MFCC as features, train, and save the classifiers for later use
    Recommended to use fit_GridSearchCV and plot validation/learning curves to determine hyper-parameter values
    and see how learning improves with more data, at what point it gets stable
    Choose what features to show to the classifier. Currently lots of variations of WE and MFCC.
    (1) Wavelet Energies - All 62 nodes, extracted from raw recordings (feature = 'weraw_all')
    (2) Wavelet Energies - Limit nodes to match frequency range of the species, extracted from raw recordings
    (3) Wavelet Energies - Limit to optimum nodes for species, extracted from raw recordings

    (4) Wavelet Energies - All 62 nodes, extracted with bandpass filter
    (5) Wavelet Energies - Limit nodes to match frequency range of the species, extracted with bandpass filter
    (6) Wavelet Energies - Limit to optimum nodes for species, extracted with bandpass filter

    (7) Wavelet Energies - All 62 nodes, extracted from denoised
    (8) Wavelet Energies - Limit nodes to match frequency range of the species, extracted from denoised
    (9) Wavelet Energies - Limit to optimum nodes for species, extracted from denoised

    (10) Wavelet Energies - All 62 nodes, extracted from denoised + bandpassed
    (11) Wavelet Energies - Limit nodes to match frequency range of the species, extracted from denoised + bandpassed
    (12) Wavelet Energies - Limit to optimum nodes for species, extracted from denoised + bandpassed

    (13) MFCC - Full range extracted from raw ('mfccraw_all')
    (14) MFCC - Limit to match frquency range of the species extracted from raw ('mfccraw_band')
    (15) MFCC - Full range extracted from bandpassed ('mfccbp_all')
    (16) MFCC - Limit to match frquency range of the species extracted from bandpassed
    (17) MFCC - Full range extracted from denoised
    (18) MFCC - Limit to match frquency range of the species extracted from denoised
    (19) MFCC - Full range extracted from bandpassed + denoised
    (20) MFCC - Limit to match frquency range of the species extracted from bandpassed + denoised

    :param dir: path to the dataset
    :param species: species name so that the classifier can be saved accordingly
    :param feature: 'WEraw_all', 'WEraw_band', 'WEraw_spnodes',
                    'WEbp_all', 'WEbp_band', 'WEbp_spnodes',
                    'WEd_all', 'WEd_band', 'WEd_spnodes',
                    'WEbpd_all', 'WEbpd_band', 'WEbpd_spnodes',
                    'MFCCraw_all', 'mfccraw_band',
                    'MFCCbp_all', 'mfccbp_band',
                    'MFCCd_all', 'MFCCd_band',
                    'MFCCbpd_all', 'MFCCbpd_band'
    :param clf: name of the classifier to train
    :return: save the trained classifier in dirName e.g. kiwi_SVM.joblib
    '''
    # Read previously stored data as required
    # d = pd.read_csv(os.path.join(dir, 'Kiwi (Tokoeka Fiordland)_WE_spnodes_seg_train.tsv'), sep=",", header=None)
    d = pd.read_csv(os.path.join(dir, species + '_' + feature + '.tsv'), sep="\t", header=None)
    data = d.values

    # Balance the data set
    targets = data[:, -1]
    data = data[:, 0:-1]
    posTargetInd = np.where(targets == 1)
    negTargetInd = np.where(targets == 0)
    # randomly select n negative rows
    n = min(np.shape(posTargetInd)[1], np.shape(negTargetInd)[1])
    posTargetInd = posTargetInd[0].tolist()
    posTargetInd = random.sample(posTargetInd, n)
    negTargetInd = negTargetInd[0].tolist()
    negTargetInd = random.sample(negTargetInd, n)
    inds = posTargetInd + negTargetInd
    data = data[inds, :]
    # use PCA if selected
    if pca:
        pca1 = PCA(n_components=0.8)   # will retain 90% of the variance
        data = pca1.fit_transform(data)
    targets = targets[inds]

    learners = Learning(data, targets, testFraction=0.5)  # use whole data set for training
    # OR learn with optimum nodes, for kiwi it is [35, 43, 36, 45]
    # kiwiNodes = [35, 43, 36, 45]
    # kiwiNodes = [34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 55]
    # kiwiNodes = [n - 1 for n in kiwiNodes]
    # nodes = list(range(63))
    # # nonKiwiNodes = list(set(nodes) - set(kiwiNodes))
    # # print(nonKiwiNodes)
    # learners = Learning(data[:, kiwiNodes], targets)
    # learners = Learning(data[:, nonKiwiNodes], data[:, -1])
    # learners = Learning(data[:, 33:61], data[:, -1])

    if clf == None: # then train all the classifiers (expensive option)
        print("MLP--------------------------------")
        # model = learners.trainMLP(structure=(100,), learningrate=0.001, solver='adam', epochs=200, alpha=1,
        #                           shuffle=True, early_stopping=False)
        model = learners.trainMLP(structure=(25,), learningrate=0.001, solver='adam', epochs=200, alpha=1,
                                  shuffle=True, early_stopping=False)
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_MLP.joblib'))
        learners.performTest(model)
        print("kNN--------------------------------")
        model = learners.trainKNN(K=3)
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_kNN.joblib'))
        learners.performTest(model)
        print("SVM--------------------------------")
        # model = learners.trainSVM(kernel="rbf", C=1, gamma=0.0077)
        model = learners.trainSVM(kernel="rbf", C=1, gamma=0.03)
        learners.performTest(model)
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_SVM.joblib'))
        learners.performTest(model)
        print("GP--------------------------------")
        model = learners.trainGP()
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_GP.joblib'))
        learners.performTest(model)
        print("DT--------------------------------")
        model = learners.trainDecisionTree()
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_DT.joblib'))
        learners.performTest(model)
        print("RF--------------------------------")
        model = learners.trainRandomForest()
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_RF.joblib'))
        learners.performTest(model)
        print("Boosting--------------------------------")
        model = learners.trainBoosting()
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_Boost.joblib'))
        learners.performTest(model)
        print("XGB--------------------------------")
        model = learners.trainXGBoost()
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_XGB.joblib'))
        learners.performTest(model)
        # print("GMM--------------------------------")
        # model = learners.trainGMM(covType='full', maxIts=200, nClasses=4)
        # # Save the model
        # dump(model, os.path.join(dir,species+'_'+feature+'_GMM.joblib'))
        print("######################################################")
    elif clf == 'MLP':
        print("MLP--------------------------------")
        model = learners.trainMLP(structure=(250, ), learningrate=0.001, solver='adam', epochs=200, alpha=1,
                                  shuffle=True, early_stopping=True)
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_MLP.joblib'))
    elif clf == 'kNN':
        print("kNN--------------------------------")
        model = learners.trainKNN(K=3)
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_kNN.joblib'))
    elif clf == 'SVM':
        print("SVM--------------------------------")
        model = learners.trainSVM(kernel="rbf", C=1, gamma=0.00018)
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_SVM.joblib'))
    elif clf == 'GP':
        print("GP--------------------------------")
        model = learners.trainGP()
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_GP.joblib'))
    elif clf == 'DT':
        print("DT--------------------------------")
        model = learners.trainDecisionTree()
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_DT.joblib'))
    elif clf == 'RF':
        print("RF--------------------------------")
        model = learners.trainRandomForest()
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_RF.joblib'))
    elif clf == 'Boost':
        print("Boosting--------------------------------")
        model = learners.trainBoosting()
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_Boost.joblib'))
    elif clf == 'XGB':
        print("XGB--------------------------------")
        model = learners.trainXGBoost()
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_XGB.joblib'))
    elif clf == 'GMM':
        print("GMM--------------------------------")
        model = learners.trainGMM(covType='full', maxIts=200, nClasses=4)
        # Save the model
        dump(model, os.path.join(dir, species+'_'+feature+'_GMM.joblib'))


def testClassifiers(dir_clf, dir_test, species, feature, clf=None, pca=False):
    '''
    Load previously trained classifiers and test on a completely new data set.
    :param dir_clf: path to the saved classifiers
    :param dir_test: path to the test dataset
    :param species: species name
    :param feature: 'WEraw_all', 'WEraw_band', 'WEraw_spnodes' ...
    :param clf: classifier name e.g. 'SVM'

    :return: print out confusion matrix
    '''
    # read test dataset
    d = pd.read_csv(os.path.join(dir_test, species + '_' + feature + '.tsv'), sep="\t", header=None)
    # d = pd.read_csv(os.path.join(dir_test, 'Kiwi (Tokoeka Fiordland)_WE_spnodes_seg_test.tsv'), sep=",", header=None)
    data = d.values
    targets = data[:, -1]
    data = data[:, 0:-1]
    # use PCA if selected
    if pca:
        pca1 = PCA(n_components=0.8)   # will retain 90% of the variance
        data = pca1.fit_transform(data)
    # Test with all 62 nodes
    learners = Learning(data, targets, testFraction=1)     # use all data for testing
    # # OR test with optimum nodes, for kiwi it is [35, 43, 36, 45]
    # # kiwiNodes = [35, 43, 36, 45]
    # kiwiNodes = [34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 55]
    # kiwiNodes = [n - 1 for n in kiwiNodes]
    # nodes = list(range(63))
    # nonKiwiNodes = list(set(nodes) - set(kiwiNodes))
    # learners = Learning(data[:, kiwiNodes], data[:, -1], testFraction=1)
    # # learners = Learning.Learning(data[:, nonKiwiNodes], data[:, -1])
    # # learners = Learning.Learning(data[:, 33:61], data[:, -1])
    if clf == None:
        print("MLP--------------------------------")
        # Load the model
        model = load(os.path.join(dir_clf, species + '_' + feature + '_MLP.joblib'))
        learners.performTest(model)
        print("kNN--------------------------------")
        model = load(os.path.join(dir_clf, species + '_' + feature + '_kNN.joblib'))
        learners.performTest(model)
        print("SVM--------------------------------")
        model = load(os.path.join(dir_clf, species + '_' + feature + '_SVM.joblib'))
        learners.performTest(model)
        print("GP--------------------------------")
        model = load(os.path.join(dir_clf, species + '_' + feature + '_GP.joblib'))
        learners.performTest(model)
        print("DT--------------------------------")
        model = load(os.path.join(dir_clf, species + '_' + feature + '_DT.joblib'))
        learners.performTest(model)
        print("RF--------------------------------")
        model = load(os.path.join(dir_clf, species + '_' + feature + '_RF.joblib'))
        learners.performTest(model)
        print("Boosting--------------------------------")
        model = load(os.path.join(dir_clf, species + '_' + feature + '_Boost.joblib'))
        learners.performTest(model)
        print("XGB--------------------------------")
        model = load(os.path.join(dir_clf, species + '_' + feature + '_XGB.joblib'))
        learners.performTest(model)
        # print("GMM--------------------------------")
        # model = load(os.path.join(dir_clf, species + '_' + feature + '_GMM.joblib'))
        # learners.performTest(model)
        print("######################################################")
    else:
        model = load(os.path.join(dir_clf, species+'_' + feature + '_' + clf + '.joblib'))
        learners.performTest(model)
