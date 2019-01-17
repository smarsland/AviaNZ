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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


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

    def __init__(self, features, labels, testFraction=0.6):
        # from sklearn.model_selection import train_test_split
        # from sklearn.preprocessing import StandardScaler
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
        # from sklearn.metrics import confusion_matrix
        testOut = model.predict(self.test)
        testError = np.mean(self.testTgt.ravel() == testOut.ravel()) * 100

        print("Testing error: ", testError)
        CM = confusion_matrix(self.testTgt, testOut)
        print(CM)
        # Get the performance metrics
        ws = WaveletSegment.WaveletSegment()
        fB, recall, TP, FP, TN, FN = ws.fBetaScore(self.testTgt, testOut)
        # Risk_grand = np.shape(np.where(abs(self.testTgt-testOut)==1))[1]/len(self.testTgt)
        Risk_FN = (self.testTgt-testOut).tolist().count(1)
        Risk_FP = (self.testTgt-testOut).tolist().count(-1)
        Risk_weighted = (2*Risk_FN+Risk_FP)/(3*len(self.testTgt))
        # Risk_weighted=((1.+2**2)*Risk_FN*Risk_FP)/(Risk_FN + 2**2*Risk_FP)
        # print("Risk M grand: ", Risk_grand*100)
        print("Risk M FN: ", Risk_FN*100/len(self.testTgt))
        print("Risk M FP: ", Risk_FP*100/len(self.testTgt))
        print("Risk M weighted (R2): ", Risk_weighted*100)

    def trainMLP(self, structure=(100,), learningrate=0.001, solver='adam', epochs=200, alpha=1, shuffle=True,
                 early_stopping=False):
        # from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=structure, solver=solver, alpha=alpha, max_iter=epochs,
                              learning_rate_init=learningrate, shuffle=shuffle, early_stopping=early_stopping)
        model.fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model

    def trainKNN(self, K=3):
        # from sklearn.neighbors import KNeighborsClassifier

        model = KNeighborsClassifier(K)
        model.fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model

    def trainSVM(self, kernel, C, gamma):
        # from sklearn.svm import SVC
        # model = SVC(kernel=kernel, C=C)
        model = SVC(kernel=kernel, C=C, gamma=gamma)
        model.fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model

    def trainGP(self,kernel="RBF", param=1.0):
        # from sklearn.gaussian_process import GaussianProcessClassifier
        # from sklearn.gaussian_process.kernels import RBF
        model = GaussianProcessClassifier(1.0 * RBF(1.0))
        model.fit(self.train,self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ",trainError)

        return model

    def trainDecisionTree(self, maxDepth=5):
        # from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(max_depth=maxDepth)
        model.fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model

    def trainRandomForest(self, maxDepth=5, nTrees=10, maxFeatures=2):
        # from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(max_depth=maxDepth, n_estimators=nTrees, max_features=maxFeatures)
        model.fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model
    
    def trainBoosting(self, n_estimators=100):
        # from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=n_estimators)
        model.fit(self.train,self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ",trainError)

        return model

    def trainXGBoost(self, nRounds=10):
        # import xgboost as xgb
        model = xgb.XGBClassifier().fit(self.train, self.trainTgt)
    
        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model

    def trainGMM(self, nClasses=2, covType='spherical', maxIts=20):
        # from sklearn.mixture import GaussianMixture

        model = GaussianMixture(n_components=nClasses, covariance_type=covType, max_iter=maxIts)
        model.means_init = np.array([self.train[self.trainTgt == i].mean(axis=0) for i in range(nClasses)])
        model.fit(self.train, self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ", trainError)

        return model
    
#For each sound class an ensemble of randomized decision trees (sklearn.ensemble.ExtraTreesRegressor) is applied. The number of estimators is chosen to be twice the number of selected features per class but not greater than 500. The winning solution considers 4 features when looking for the best split and requires a minimum of 3 samples to split an internal node.

# Diagnostic plots
class Validation:
    # This class implements cross-validation and learning curves for the AviaNZ interface
    # based on scikit-learn

    def __init__(self, estimator, title, features, labels, param_name=None, param_range=None,scoring=None):
        from sklearn.model_selection import ShuffleSplit
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
    ''' Choose a classifier and plot the learning curve '''

    # Create fB score
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
    # Choose an estimator
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
    v = Validation(estimator, title, data, targets, scoring=score)
    plt = v.plot_learning_curve()
    plt.show()

def validationCurve(dataFile, clf, nClasses=2, score=None):
    # Plot validation curve
    # Score against different values for a selected hyperparameter to see the influence of a single hyperparameter

    # Create fB score
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
    # Choose an estimator
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

    title = "Validation Curves - " + clf
    v = Validation(estimator, title, data, targets, param_name=param_name, param_range=param_range, scoring=score)
    plt = v.plot_validation_curve()
    plt.show()

def fit_GridSearchCV(dataFile, clf, nClasses):
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


def testLearning3(dirName, species, clf=None):
    '''
    Use wavelet energy as features, train and save the classifiers
    '''
    d = pd.read_csv(dirName+'\energies.tsv', sep="\t", header=None)
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
    # Learn with all 62 nodes
    learners = Learning(data, targets, testFraction=0)  # use whole data set for training
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

    if clf == None:
        print("MLP--------------------------------")
        model = learners.trainMLP(structure=(100,), learningrate=0.001, solver='adam', epochs=200, alpha=1,
                                  shuffle=True, early_stopping=False)
        # Save the model
        dump(model, dirName+'\\'+species+'_MLP.joblib')
        # learners.performTest(model)
        print("kNN--------------------------------")
        model = learners.trainKNN()
        # Save the model
        dump(model, dirName+'\\'+species+'_kNN.joblib')
        # learners.performTest(model)
        print("SVM--------------------------------")
        model = learners.trainSVM(kernel="rbf", C=1, gamma=0.0077)
        # Save the model
        dump(model, dirName+'\\'+species+'_SVM.joblib')
        # learners.performTest(model)
        print("GP--------------------------------")
        model = learners.trainGP()
        # Save the model
        dump(model, dirName+'\\'+species+'_GP.joblib')
        # learners.performTest(model)
        print("DT--------------------------------")
        model = learners.trainDecisionTree()
        # Save the model
        dump(model, dirName+'\\'+species+'_DT.joblib')
        # learners.performTest(model)
        print("RF--------------------------------")
        model = learners.trainRandomForest()
        # Save the model
        dump(model, dirName+'\\'+species+'_RF.joblib')
        # learners.performTest(model)
        print("Boosting--------------------------------")
        model = learners.trainBoosting()
        # Save the model
        dump(model, dirName+'\\'+species+'_Boost.joblib')
        # learners.performTest(model)
        print("XGB--------------------------------")
        model = learners.trainXGBoost()
        # Save the model
        dump(model, dirName+'\\'+species+'_XGB.joblib')
        # learners.performTest(model)
        print("GMM--------------------------------")
        model = learners.trainGMM(covType='full', maxIts=200, nClasses=4)
        # Save the model
        dump(model, dirName+'\\'+species+'_GMM.joblib')
        # learners.performTest(model)
        print("######################################################")
    elif clf == 'MLP':
        print("MLP--------------------------------")
        model = learners.trainMLP(structure=(250, ), learningrate=0.001, solver='adam', epochs=200, alpha=1,
                                  shuffle=True, early_stopping=True)
        # Save the model
        dump(model, dirName + '\\' + species + '_' + clf + '.joblib')
    elif clf == 'kNN':
        print("kNN--------------------------------")
        model = learners.trainKNN()
        # Save the model
        dump(model, dirName+'\\'+species+'_kNN.joblib')
    elif clf == 'SVM':
        print("SVM--------------------------------")
        model = learners.trainSVM(kernel="rbf", C=1, gamma=0.0077)
        # Save the model
        dump(model, dirName+'\\'+species+'_SVM.joblib')
    elif clf == 'GP':
        print("GP--------------------------------")
        model = learners.trainGP()
        # Save the model
        dump(model, dirName+'\\'+species+'_GP.joblib')
    elif clf == 'DT':
        print("DT--------------------------------")
        model = learners.trainDecisionTree()
        # Save the model
        dump(model, dirName+'\\'+species+'_DT.joblib')
    elif clf == 'RF':
        print("RF--------------------------------")
        model = learners.trainRandomForest()
        # Save the model
        dump(model, dirName+'\\'+species+'_RF.joblib')
    elif clf == 'Boost':
        print("Boosting--------------------------------")
        model = learners.trainBoosting()
        # Save the model
        dump(model, dirName+'\\'+species+'_Boost.joblib')
    elif clf == 'XGB':
        print("XGB--------------------------------")
        model = learners.trainXGBoost()
        # Save the model
        dump(model, dirName+'\\'+species+'_XGB.joblib')
    elif clf == 'GMM':
        print("GMM--------------------------------")
        model = learners.trainGMM(covType='full', maxIts=200, nClasses=4)
        # Save the model
        dump(model, dirName+'\\'+species+'_GMM.joblib')
    if clf:
        # Save the model
        dump(model, dirName + '\\' + species + '_' + clf + '.joblib')

def testClassifiers(dir_clf, dir_test, species, clf=None):
    '''
    Load previously trained classifiers and test on new data.
    dirName has the classifiers to use
    '''
    d = pd.read_csv(dir_test+'\energies.tsv', sep="\t", header=None)
    data = d.values
    # Test with all 62 nodes
    learners = Learning(data[:, 0:-1], data[:, -1], testFraction=1)     # use all data for testing
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
        model = learners.trainSVM(kernel="rbf", C=1, gamma=0.0077)
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

brownKiwi_segmentbased_train('XGB')
brownKiwi_segmentbased_test(clf='XGB')

# validationCurve(dataFile='D:\\Nirosha\CHAPTER5\kiwi\\brown_segmentbased\energies_train.tsv', clf='MLP', nClasses=3, score='accuracy')
# learninigCurve('D:\\nirosha\CHAPTER5\kiwi\\brown\\train-national\energies.tsv', 'XGB')
# fit_GridSearchCV('D:\\nirosha\CHAPTER5\kiwi\\brown_segmentbased\energies_train.tsv', clf='Boost', nClasses=3)

# testLearning3(dirName='D:\\nirosha\CHAPTER5\kiwi\\brown\\train-national', species='kiwi', clf='kNN')
# testClassifiers(dir_clf='D:\\nirosha\CHAPTER5\kiwi\\brown\\train-national', dir_test='D:\\nirosha\CHAPTER5\kiwi\\tier1',
#                 species='kiwi', clf='kNN')

