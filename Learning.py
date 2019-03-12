# Version 0.1 30/5/16
# Author: Stephen Marsland

import numpy as np
import sklearn as sk
import WaveletSegment
import matplotlib.pyplot as plt
import pandas as pd
import random
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
#import xgboost as xgb
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import json, time, os, math, csv, gc, wavio
import WaveletSegment
import librosa


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
            self.train, self.test, self.trainTgt, self.testTgt = train_test_split(features, labels, test_size=testFraction, shuffle=True)

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

    def trainKNN(self, K):
        model = KNeighborsClassifier(K)
        model.fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model

    def trainSVM(self, kernel, C, gamma):
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

# Diagnostic plots
class Validate:
    # This class implements cross-validation and learning curves for the AviaNZ interface
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


def testLearning1():
    # Very simple test
    import Learning
    from sklearn.datasets import make_classification
    features, labels = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    learners = Learning.Learning(features,labels)
    
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
    from sklearn.datasets import load_iris
    data = load_iris()
    learners = Learning.Learning(data.data,data.target)
    
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
    d = pd.read_csv(os.path.join(dir,species + '_' + feature + '.tsv'), sep="\t", header=None)
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

    learners = Learning(data, targets, testFraction=0.0)  # use whole data set for training
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
        dump(model, os.path.join(dir,species+'_'+feature+'_MLP.joblib'))
        learners.performTest(model)
        print("kNN--------------------------------")
        model = learners.trainKNN(K=3)
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_kNN.joblib'))
        learners.performTest(model)
        print("SVM--------------------------------")
        # model = learners.trainSVM(kernel="rbf", C=1, gamma=0.0077)
        model = learners.trainSVM(kernel="rbf", C=1, gamma=0.03)
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_SVM.joblib'))
        learners.performTest(model)
        print("GP--------------------------------")
        model = learners.trainGP()
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_GP.joblib'))
        learners.performTest(model)
        print("DT--------------------------------")
        model = learners.trainDecisionTree()
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_DT.joblib'))
        learners.performTest(model)
        print("RF--------------------------------")
        model = learners.trainRandomForest()
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_RF.joblib'))
        learners.performTest(model)
        print("Boosting--------------------------------")
        model = learners.trainBoosting()
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_Boost.joblib'))
        learners.performTest(model)
        print("XGB--------------------------------")
        model = learners.trainXGBoost()
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_XGB.joblib'))
        learners.performTest(model)
        print("GMM--------------------------------")
        model = learners.trainGMM(covType='full', maxIts=200, nClasses=4)
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_GMM.joblib'))
        learners.performTest(model)
        print("######################################################")
    elif clf == 'MLP':
        print("MLP--------------------------------")
        model = learners.trainMLP(structure=(250, ), learningrate=0.001, solver='adam', epochs=200, alpha=1,
                                  shuffle=True, early_stopping=True)
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_MLP.joblib'))
    elif clf == 'kNN':
        print("kNN--------------------------------")
        model = learners.trainKNN(K=3)
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_kNN.joblib'))
    elif clf == 'SVM':
        print("SVM--------------------------------")
        model = learners.trainSVM(kernel="rbf", C=1, gamma=0.00018)
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_SVM.joblib'))
    elif clf == 'GP':
        print("GP--------------------------------")
        model = learners.trainGP()
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_GP.joblib'))
    elif clf == 'DT':
        print("DT--------------------------------")
        model = learners.trainDecisionTree()
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_DT.joblib'))
    elif clf == 'RF':
        print("RF--------------------------------")
        model = learners.trainRandomForest()
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_RF.joblib'))
    elif clf == 'Boost':
        print("Boosting--------------------------------")
        model = learners.trainBoosting()
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_Boost.joblib'))
    elif clf == 'XGB':
        print("XGB--------------------------------")
        model = learners.trainXGBoost()
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_XGB.joblib'))
    elif clf == 'GMM':
        print("GMM--------------------------------")
        model = learners.trainGMM(covType='full', maxIts=200, nClasses=4)
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_GMM.joblib'))
    if not clf:
        # Save the model
        dump(model, os.path.join(dir,species+'_'+feature+'_'+clf+'.joblib'))
        learners.performTest(model)

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
    d = pd.read_csv(os.path.join(dir_test,species + '_' + feature + '.tsv'), sep="\t", header=None)
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
        model = load(os.path.join(dir_clf,species+'_MLP.joblib'))
        learners.performTest(model)
        print("kNN--------------------------------")
        model = load(os.path.join(dir_clf,species+'_kNN.joblib'))
        learners.performTest(model)
        print("SVM--------------------------------")
        model = load(os.path.join(dir_clf,species+'_SVM.joblib'))
        learners.performTest(model)
        print("GP--------------------------------")
        model = load(os.path.join(dir_clf,species+'_GP.joblib'))
        learners.performTest(model)
        print("DT--------------------------------")
        model = load(os.path.join(dir_clf,species+'_DT.joblib'))
        learners.performTest(model)
        print("RF--------------------------------")
        model = load(os.path.join(dir_clf,species+'_RF.joblib'))
        learners.performTest(model)
        print("Boosting--------------------------------")
        model = load(os.path.join(dir_clf,species+'_Boost.joblib'))
        learners.performTest(model)
        print("XGB--------------------------------")
        model = load(os.path.join(dir_clf,species+'_XGB.joblib'))
        learners.performTest(model)
        print("GMM--------------------------------")
        model = load(os.path.join(dir_clf,species+'_GMM.joblib'))
        learners.performTest(model)
        print("######################################################")
    else:
        model = load(os.path.join(dir_clf,species+'_' + feature + '_' + clf + '.joblib'))
        learners.performTest(model)

def generateDataset(dir_src, feature, species, filemode, wpmode, dir_out):
    '''
    Generates different data sets for ML - variations of WE and MFCC
    Can be continuous wav files or extracted segments
    Continuous files + GT annotations OR
    Extracted segments + tell if they are TPs or not
    :param dir_src: path to the directory with recordings + GT annotations
    :param feature: 'WEraw_all', 'WEbp_all', 'WEd_all', 'WEbpd_all', 'MFCCraw_all', 'MFCCbp_all', 'MFCCd_all',
                    'MFCCbpd_all',
                    'WE+MFCCraw_all', 'WE+MFCCbp_all', 'WE+MFCCd_all', 'WE+MFCCbpd_all'
    :param species: species name (should be able to find species filter in dir_src)
    :param filemode: 'long', 'segpos', 'segneg'
    :param wpmode: 'pywt' or 'new' or 'aa'
    :param dir_out: path to the output dir

    :return: saves the data file to out-dir
    '''

    annotation = []
    if 'WE' in feature:
        nlevels = 5
        waveletCoefs = np.array([]).reshape(2**(nlevels+1)-2, 0)
    if 'MFCC' in feature:
        n_mfcc = 48
        # n_bins = 8
        n_bins = 32
        delta = True
        if delta:
            MFCC = np.array([]).reshape(0, n_mfcc * 2 * n_bins)
        else:
            MFCC = np.array([]).reshape(0, n_mfcc * n_bins)
    # Find the species filter
    speciesData = json.load(open(os.path.join(dir_src, species + '.txt')))

    for root, dirs, files in os.walk(str(dir_src)):
        for file in files:
            if file.endswith('.wav') and os.stat(root + '/' + file).st_size != 0 and file[:-4] + '-res1.0sec.txt' in files or file.endswith('.wav') and filemode!='long' and os.stat(root + '/' + file).st_size > 150:
                opstartingtime = time.time()
                wavFile = root + '/' + file[:-4]
                print(wavFile)
                data, currentannotation, sampleRate = loadData(wavFile, filemode)

                ws = WaveletSegment.WaveletSegment(data=data, sampleRate=sampleRate)
                if feature == 'WEraw_all' or feature == 'MFCCraw_all' or feature == 'WE+MFCCraw_all':
                    data = ws.preprocess(speciesData, d=False, f=False)
                elif feature == 'WEbp_all' or feature == 'MFCCbp_all' or feature == 'WE+MFCCbp_all':
                    data = ws.preprocess(speciesData, d=False, f=True)
                elif feature == 'WEd_all' or feature == 'MFCCd_all' or feature == 'WE+MFCCd_all':
                    data = ws.preprocess(speciesData, d=True, f=False)
                elif feature == 'WEbpd_all' or feature == 'MFCCbpd_all' or feature == 'WE+MFCCbpd_all':
                    data = ws.preprocess(speciesData, d=True, f=True)

                # Compute energy in each WP node and store
                if 'WE' in feature:
                    currWCs = ws.computeWaveletEnergy(data=data, sampleRate=ws.sampleRate, nlevels=nlevels,
                                                      wpmode=wpmode)
                    waveletCoefs = np.column_stack((waveletCoefs, currWCs))
                if 'MFCC' in feature:
                    currMFCC = computeMFCC(data=data, sampleRate=ws.sampleRate, n_mfcc=n_mfcc, n_bins=n_bins, delta=delta)
                    MFCC = np.concatenate((MFCC, currMFCC), axis=0)
                annotation.extend(currentannotation)
                print("file loaded in", time.time() - opstartingtime)

    annotation = np.array(annotation)
    ann = np.reshape(annotation, (len(annotation), 1))
    if 'WE' in feature and 'MFCC' not in feature:
        # Prepare WC data and annotation targets into a matrix for saving
        WC = np.transpose(waveletCoefs)
        # ann = np.reshape(annotation,(len(annotation),1))
        MLdata = np.append(WC, ann, axis=1)
    elif 'MFCC' in feature and 'WE' not in feature:
        # ann = np.reshape(annotation, (len(annotation), 1))
        MLdata = np.append(MFCC, ann, axis=1)
    elif 'WE' in feature and 'MFCC' in feature:
        WC = np.transpose(waveletCoefs)
        WE_MFCC = np.append(WC, MFCC, axis=1)
        MLdata = np.append(WE_MFCC, ann, axis=1)
    np.savetxt(os.path.join(dir_out, species + '_' + feature + '.tsv'), MLdata, delimiter="\t")
    print("Directory loaded. %d/%d presence blocks found.\n" % (np.sum(annotation), len(annotation)))

def loadData(fName, filemode):
    '''
    Load wav and GT for ML data set generation
    :param fName:
    :param filemode: 'long' or 'segpos' or 'segneg'
    :return: audio data, GT, sampleRate
    '''
    filename = fName+'.wav'
    filenameAnnotation = fName+'-res1.0sec.txt'
    try:
        wavobj = wavio.read(filename)
    except:
        print("unsupported file: ", filename)
        pass
    sampleRate = wavobj.rate
    data = wavobj.data
    if data.dtype is not 'float':
        data = data.astype('float') #/ 32768.0
    if np.shape(np.shape(data))[0]>1:
        data = np.squeeze(data[:,0])
    n = math.ceil(len(data)/sampleRate)

    if filemode=='long':
        # GT from the txt file
        fileAnnotation = []
        with open(filenameAnnotation) as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)
        if d[-1]==[]:
            d = d[:-1]
        if len(d) != n:
            print("ERROR: annotation length %d does not match file duration %d!" %(len(d), n))
            return
        # for each second, store 0/1 presence:
        sum = 0
        for row in d:
            fileAnnotation.append(int(row[1]))
            sum += int(row[1])
    elif filemode=='segpos':
        fileAnnotation = np.ones((math.ceil(len(data) / sampleRate), 1))
    elif filemode=='segneg':
        fileAnnotation = np.zeros((math.ceil(len(data) / sampleRate), 1))
    return data, np.array(fileAnnotation), sampleRate

def computeMFCC(data, sampleRate, n_mfcc, n_bins, delta):
    '''
    Compute MFCC for each second of data and return as a matrix
    :param data: audio data
    :param sampleRate: sample rate
    :param delta: True/False
    :return: MFCC metrix
    '''
    n = math.ceil(len(data) / sampleRate)
    if delta:
        mfcc = np.zeros((n, n_mfcc * 2 * n_bins))
    else:
        mfcc=np.zeros((n, n_mfcc*n_bins))
    i = 0
    for t in range(n):
        end = min(len(data), (t + 1) * sampleRate)
        mfcc1 = librosa.feature.mfcc(y=data[t * sampleRate:end], sr=sampleRate, n_mfcc=n_mfcc, n_fft=2048,
                                        hop_length=512)  # n_fft=10240, hop_length=2560
        if delta:
            if n_bins == 8:
                mfcc1_delta = librosa.feature.delta(mfcc1, width=5)
            else:
                mfcc1_delta = librosa.feature.delta(mfcc1)
            mfcc1 = np.concatenate((mfcc1, mfcc1_delta), axis=0)
        # Normalize
        mfcc1 -= np.mean(mfcc1, axis=0)
        mfcc1 /= np.max(np.abs(mfcc1), axis=0)
        mfcc1 = np.reshape(mfcc1, np.shape(mfcc1)[0] * np.shape(mfcc1)[1])
        mfcc1 = mfcc1.flatten()
        mfcc[i, :] = mfcc1
        i += 1
    return mfcc


# generateDataset(dir_src="D:\WaveletDetection\DATASETS\\NIbrownkiwi\Test_5min", feature='WE+MFCCbp_all',
#                 species='Kiwi', filemode='long', wpmode='new',
#                 dir_out="D:\WaveletDetection\DATASETS\\NIbrownkiwi\Test_5min\ML")
# generateDataset(dir_src="D:\WaveletDetection\DATASETS\\NIbrownkiwi\Train_5min", feature='WE+MFCCbp_all',
#                 species='Kiwi', filemode='long', wpmode='new',
#                 dir_out="D:\WaveletDetection\DATASETS\\NIbrownkiwi\Train_5min\ML")
TrainClassifier('D:\WaveletDetection\DATASETS\Morepork\Train-5min\ML', 'Morepork', 'WEbp_all', clf='kNN',
                pca=True)
testClassifiers(dir_clf='D:\WaveletDetection\DATASETS\Morepork\Train-5min\ML',
                dir_test='D:\WaveletDetection\DATASETS\Morepork\Test-5min\ML', species='Morepork',
                feature='WEbp_all', clf='kNN', pca=True)

# generateDataset(dir_src="D:\WaveletDetection\DATASETS\Kakapo\KakapoC\\test-5min", feature='WEbp_all',
#                 species='KakapoC', filemode='long', wpmode='new',
#                 dir_out="D:\WaveletDetection\DATASETS\Kakapo\KakapoC\\test-5min\ML")
# generateDataset(dir_src="D:\WaveletDetection\DATASETS\Kakapo\KakapoC\\train-5min\\train", feature='WEbp_all',
#                 species='KakapoC', filemode='long', wpmode='new',
#                 dir_out="D:\WaveletDetection\DATASETS\Kakapo\KakapoC\\train-5min\\train\ML")
# TrainClassifier('D:\WaveletDetection\DATASETS\Kakapo\KakapoB\\train-5min\ML', 'KakapoB', 'WE+MFCCraw_all', clf='SVM',
#                 pca=False)
# testClassifiers(dir_clf='D:\WaveletDetection\DATASETS\Kakapo\KakapoB\\train-5min\ML',
#                 dir_test='D:\WaveletDetection\DATASETS\Kakapo\KakapoB\\test-5min\ML', species='KakapoB',
#                 feature='WE+MFCCraw_all', clf='SVM', pca=False)

# validationCurve(dataFile='D:\WaveletDetection\DATASETS\Morepork\Train-5min\ML\Morepork_MFCCbp_all.tsv', clf='kNN', nClasses=2, score=None)

generateDataset(dir_src="/Users/marslast/Projects/AviaNZ/Sound Files/Train", feature='MFCCraw_all', species='Morepork',filemode='long', wpmode='new', dir_out="/Users/marslast/Projects/AviaNZ/Sound Files/Train")
TrainClassifier('/Users/marslast/Projects/AviaNZ/Sound Files/Train', 'Morepork', 'MFCCraw_all', clf='kNN')
testClassifiers(dir_clf='/Users/marslast/Projects/AviaNZ/Sound Files/Train', dir_test='/Users/marslast/Projects/AviaNZ/Sound Files/Train', species='Morepork', feature='MFCCraw_all',clf='kNN')











# IGNORE THE FOLLOWING CODE FOR NOW
def testScoring():
    from sklearn import svm, datasets
    from sklearn.model_selection import cross_val_score
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf = svm.SVC(gamma='scale', random_state=0)
    # cross_val_score(clf, X, y, scoring='recall_macro', cv=5)
    model = svm.SVC()
    print(cross_val_score(model, X, y, cv=5, scoring='recall_macro'))
# testScoring()


def brownKiwi_segmentbased_train(clf=None, dirName="D:\\Nirosha\CHAPTER5\kiwi\\brown_segmentbased"):
    d_male = pd.read_csv('D:\\Nirosha\CHAPTER5\kiwi\\brown_segmentbased\male\energies.tsv', sep="\t", header=None)
    data_male = d_male.values
    d_female = pd.read_csv('D:\\Nirosha\CHAPTER5\kiwi\\brown_segmentbased\\female\energies.tsv', sep="\t", header=None)
    data_female = d_female.values
    d_noise = pd.read_csv('D:\\Nirosha\CHAPTER5\kiwi\\brown_segmentbased\\noise\energies.tsv', sep="\t", header=None)
    data_noise = d_noise.values
    # Set the last column noise=0, male=1, female=2
    data_male[:,-1] = 1
    data_female[:,-1] = 2
    # merge into one dataset
    data = np.concatenate((data_noise,data_male),axis=0)
    data = np.concatenate((data, data_female), axis=0)
    targets = data[:, -1]
    data = data[:, 0:-1]
    # Balance the data set
    noiseTargetInd = np.where(targets == 0)
    maleTargetInd = np.where(targets == 1)
    femaleTargetInd = np.where(targets == 2)
    # randomly select n negative rows
    n = min(np.shape(noiseTargetInd)[1], np.shape(maleTargetInd)[1], np.shape(femaleTargetInd)[1])
    noiseTargetInd = noiseTargetInd[0].tolist()
    noiseTargetInd = random.sample(noiseTargetInd, n)
    maleTargetInd = maleTargetInd[0].tolist()
    maleTargetInd = random.sample(maleTargetInd, n)
    femaleTargetInd = femaleTargetInd[0].tolist()
    femaleTargetInd = random.sample(femaleTargetInd, n)
    inds = noiseTargetInd + maleTargetInd + femaleTargetInd
    data = data[inds, :]
    targets = targets[inds]
    # Learn with all 62 nodes
    learners = Learning(data, targets, testFraction=0)  # use whole data set for training
    if clf == None:
        print("MLP--------------------------------")
        model = learners.trainMLP(structure=(100,), learningrate=0.001, solver='adam', epochs=200, alpha=1,
                                  shuffle=True, early_stopping=False)
        # Save the model
        dump(model, dirName+'\\'+'bk_MLP.joblib')
        # learners.performTest(model)
        print("kNN--------------------------------")
        model = learners.trainKNN(K=5)
        # Save the model
        dump(model, dirName+'\\'+'bk_kNN.joblib')
        # learners.performTest(model)
        print("SVM--------------------------------")
        # model = learners.trainSVM(kernel="rbf", C=1, gamma=0.0077)
        model = learners.trainSVM(kernel="rbf", C=1, gamma='auto')
        # Save the model
        dump(model, dirName+'\\'+ 'bk_SVM.joblib')
        # learners.performTest(model)
        print("GP--------------------------------")
        model = learners.trainGP()
        # Save the model
        dump(model, dirName+'\\'+'bk_GP.joblib')
        # learners.performTest(model)
        print("DT--------------------------------")
        model = learners.trainDecisionTree()
        # Save the model
        dump(model, dirName+'\\'+'bk_DT.joblib')
        # learners.performTest(model)
        print("RF--------------------------------")
        model = learners.trainRandomForest()
        # Save the model
        dump(model, dirName+'\\'+'bk_RF.joblib')
        # learners.performTest(model)
        print("Boosting--------------------------------")
        model = learners.trainBoosting()
        # Save the model
        dump(model, dirName+'\\'+'bk_Boost.joblib')
        # learners.performTest(model)
        print("XGB--------------------------------")
        model = learners.trainXGBoost()
        # Save the model
        dump(model, dirName+'\\'+'bk_XGB.joblib')
        # learners.performTest(model)
        print("GMM--------------------------------")
        model = learners.trainGMM(covType='full', maxIts=200, nClasses=4)
        # Save the model
        dump(model, dirName+'\\'+'bk_GMM.joblib')
        # learners.performTest(model)
        print("######################################################")
    elif clf == 'MLP':
        print("MLP--------------------------------")
        model = learners.trainMLP(structure=(250, ), learningrate=0.001, solver='adam', epochs=100, alpha=1,
                                  shuffle=True, early_stopping=True)
    elif clf == 'kNN':
        print("kNN--------------------------------")
        model = learners.trainKNN()
    elif clf == 'SVM':
        print("SVM--------------------------------")
        model = learners.trainSVM(kernel="rbf", C=1, gamma=0.0032)
    elif clf == 'GP':
        print("GP--------------------------------")
        model = learners.trainGP()
    elif clf == 'DT':
        print("DT--------------------------------")
        model = learners.trainDecisionTree()
    elif clf == 'RF':
        print("RF--------------------------------")
        model = learners.trainRandomForest()
    elif clf == 'Boost':
        print("Boosting--------------------------------")
        model = learners.trainBoosting(n_estimators=200)
    elif clf == 'XGB':
        print("XGB--------------------------------")
        model = learners.trainXGBoost()
    elif clf == 'GMM':
        print("GMM--------------------------------")
        model = learners.trainGMM(covType='full', maxIts=200, nClasses=4)
    if clf:
        # Save the model
        dump(model, dirName + '\\' + 'bk_' + clf + '.joblib')
        # learners.performTest(model)

def brownKiwi_segmentbased_test(clf=None, dir_clf="D:\\Nirosha\CHAPTER5\kiwi\\brown_segmentbased", species='bk'):
    d_male = pd.read_csv('D:\\Nirosha\CHAPTER5\kiwi\\brown_segmentbased\\TestData_Tier1\energies_Tier1_male.tsv', sep="\t", header=None)
    data_male = d_male.values
    d_female = pd.read_csv('D:\\Nirosha\CHAPTER5\kiwi\\brown_segmentbased\\TestData_Tier1\energies_Tier1_female.tsv', sep="\t", header=None)
    data_female = d_female.values
    d_noise = pd.read_csv('D:\\Nirosha\CHAPTER5\kiwi\\brown_segmentbased\\\\TestData_Tier1\energies_Tier1_noise.tsv', sep="\t", header=None)
    data_noise = d_noise.values
    # Set the last column noise=0, male=1, female=2
    data_male[:,-1] = 1
    data_female[:,-1] = 2
    # merge into one dataset
    data = np.concatenate((data_noise,data_male),axis=0)
    data = np.concatenate((data, data_female), axis=0)
    # ##
    # # test on train data and see
    # d = pd.read_csv('D:\\Nirosha\CHAPTER5\kiwi\\brown_segmentbased\energies_train.tsv', sep="\t", header=None)
    # data = d.values
    # # ##
    targets = data[:, -1]
    data = data[:, 0:-1]
    # Learn with all 62 nodes
    learners = Learning(data, targets, testFraction=1.0)  # use whole data set for testing
    if clf == None:
        print("MLP--------------------------------")
        # Load the model
        model = load(dir_clf+'\\'+species+'_MLP.joblib')
        learners.performTest(model)
        print("kNN--------------------------------")
        model = load(dir_clf+'\\'+species+'_kNN.joblib')
        learners.performTest(model)
        print("SVM--------------------------------")
        model = load(dir_clf+'\\'+species+'_SVM.joblib')
        learners.performTest(model)
        print("GP--------------------------------")
        model = load(dir_clf+'\\'+species+'_GP.joblib')
        learners.performTest(model)
        print("DT--------------------------------")
        model = load(dir_clf+'\\'+species+'_DT.joblib')
        learners.performTest(model)
        print("RF--------------------------------")
        model = load(dir_clf+'\\'+species+'_RF.joblib')
        learners.performTest(model)
        print("Boosting--------------------------------")
        model = load(dir_clf+'\\'+species+'_Boost.joblib')
        learners.performTest(model)
        print("XGB--------------------------------")
        model = load(dir_clf+'\\'+species+'_XGB.joblib')
        learners.performTest(model)
        print("GMM--------------------------------")
        model = load(dir_clf+'\\'+species+'_GMM.joblib')
        learners.performTest(model)
        print("######################################################")
    else:
        model = load(dir_clf + '\\' + species + '_' + clf + '.joblib')
        learners.performTest(model)

# brownKiwi_segmentbased_train('XGB')
# brownKiwi_segmentbased_test(clf='XGB')

# validationCurve(dataFile='D:\\Nirosha\CHAPTER5\kiwi\\brown_segmentbased\energies_train.tsv', clf='MLP', nClasses=3, score='accuracy')
# learninigCurve('D:\\nirosha\CHAPTER5\kiwi\\brown\\train-national\energies.tsv', 'XGB')
# fit_GridSearchCV('D:\\nirosha\CHAPTER5\kiwi\\brown_segmentbased\energies_train.tsv', clf='Boost', nClasses=3)

# testLearning3(dirName='D:\\nirosha\CHAPTER5\kiwi\\brown\\train-national', species='kiwi', clf='kNN')
# testClassifiers(dir_clf='D:\\nirosha\CHAPTER5\kiwi\\brown\\train-national', dir_test='D:\\nirosha\CHAPTER5\kiwi\\tier1',
#                 species='kiwi', clf='kNN')

