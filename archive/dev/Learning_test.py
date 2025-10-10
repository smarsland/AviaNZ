# Learning_test.py
#
# Learning tests

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

import Learning

import pandas as pd
import numpy as np
import random
from joblib import dump, load
import os

from sklearn.decomposition import PCA
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
from sklearn.model_selection import GridSearchCV


def testLearning1():
    # Very simple test
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
    v = Learning.Validate(estimator, title, data, targets, scoring=score)
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
    v = Learning.Validate(estimator, title, data, targets, param_name=param_name, param_range=param_range, scoring=score)
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