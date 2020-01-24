# Learning.py
#
# Learning

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
import random
from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
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