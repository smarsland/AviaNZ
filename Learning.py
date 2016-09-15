# Version 0.1 30/5/16
# Author: Stephen Marsland

import numpy as np
import pywt
from scipy.io import wavfile
import pylab as pl
import matplotlib
import sklearn

# TODO:
# Put some stuff in here!
# Needs decision trees and recurrent NN as comparisons to Digby and Bagnall
# Random forests from Lasseck
# GMM from lots of people
# Plus whatever scikit-learn has :)
# Some deep learning stuff?
#xgboost

# Also, consider another module for HMMs and related -> syllable ordering, etc.

class Learning:
    # This class implements various learning algorithms for the AviaNZ interface
    # Based on scikit-learn

    def __init__(self,features):
        pass

#For each sound class an ensemble of randomized decision trees (sklearn.ensemble.ExtraTreesRegressor) is applied. The number of estimators is chosen to be twice the number of selected features per class but not greater than 500. The winning solution considers 4 features when looking for the best split and requires a minimum of 3 samples to split an internal node.
