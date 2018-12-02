# Version 0.1 30/5/16
# Author: Stephen Marsland

import numpy as np
import sklearn as sk

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

    def __init__(self,features,labels,testFraction=0.4):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        features = StandardScaler().fit_transform(features)
        self.train, self.test, self.trainTgt, self.testTgt = train_test_split(features, labels, test_size=testFraction)

    def performTest(self,model):
        from sklearn.metrics import confusion_matrix
        testOut = model.predict(self.test)
        testError = np.mean(self.testTgt.ravel() == testOut.ravel()) * 100

        print("Testing error: ",testError)
        print(confusion_matrix(self.testTgt,testOut))

    def trainMLP(self):
    #def trainMLP(self,structure,learningrate,epochs):
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(alpha=1)
        model.fit(self.train,self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ",trainError)

        return model

    def trainKNN(self,K=3):
        from sklearn.neighbors import KNeighborsClassifier

        model = KNeighborsClassifier(K)
        model.fit(self.train,self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ",trainError)

        return model

    def trainSVM(self,kernel="linear",C=0.025):
        from sklearn.svm import SVC
        model = SVC(kernel=kernel, C=C)
        model.fit(self.train,self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ",trainError)

        return model

    def trainGP(self,kernel="RBF",param=1.0):
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        model = GaussianProcessClassifier(1.0 * RBF(1.0))
        model.fit(self.train,self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ",trainError)

        return model

    def trainDecisionTree(self,maxDepth=5):
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(max_depth=maxDepth)
        model.fit(self.train,self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ",trainError)

        return model

    def trainRandomForest(self,maxDepth=5,nTrees=10,maxFeatures=2):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(max_depth=maxDepth, n_estimators=nTrees, max_features=maxFeatures)
        model.fit(self.train,self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ",trainError)

        return model
    
    def trainBoosting(self):
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier()
        model.fit(self.train,self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ",trainError)

        return model

    def trainXGBoost(self,nRounds=10):
        import xgboost as xgb
        model = xgb.XGBClassifier().fit(self.train,self.trainTgt)
    
        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ",trainError)

        return model

    def trainGMM(self,nClasses=2,covType='spherical',maxIts=20):
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(n_components=nClasses, covariance_type=covType, max_iter=maxIts)
        model.means_init = np.array([self.train[self.trainTgt == i].mean(axis=0) for i in range(nClasses)])
        model.fit(self.train,self.trainTgt)

        trainOut = model.predict(self.train)
        trainError = np.mean(self.trainTgt.ravel() == trainOut.ravel()) * 100
        print("Training Error: ",trainError)

        return model
    
#For each sound class an ensemble of randomized decision trees (sklearn.ensemble.ExtraTreesRegressor) is applied. The number of estimators is chosen to be twice the number of selected features per class but not greater than 500. The winning solution considers 4 features when looking for the best split and requires a minimum of 3 samples to split an internal node.

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
