
# This is to quickly test the wavelet energies with a variety of learners
# Column 62 is a binary index (very unbalanced),  column 63 is a class index 

import numpy as np
from sklearn.metrics import confusion_matrix

# Load the dataset, put it into a numpy array
# Kiwi are 6 (M) and 7 (F)
#f = np.genfromtxt("first.data",delimiter=',',dtype=None)
# Kiwi are 10 (M) and 11 (F), Ruru are 0, 6, 14, 15
#f = np.genfromtxt("wEnergyAll.data",delimiter=',',dtype=None)
#f = np.genfromtxt("wEnergyBandpass.data",delimiter=',',dtype=None)
f = np.genfromtxt("wEnergyBandpassDenoised.data",delimiter=',',dtype=None)

ld = len(f[0])
data = np.zeros((len(f),ld))

names = []

for i in range(len(f)):
    for j in range(ld-2):
        data[i,j] = f[i][j]
    data[i,ld-2] = f[i][ld-1]
    if not f[i][ld-2] in names:
        names.append(f[i][ld-2])
        data[i,ld-1] = len(names)
    else:
        data[i,ld-1] = names.index(f[i][ld-2])
        
# Decide on a class to be the 1 to detect
data[:,62] = 0
#inds = np.where(data[:,63] == 7)
#data[inds,62] = 1
inds = np.where(data[:,63] == 0)
data[inds,62] = 1
inds = np.where(data[:,63] == 6)
data[inds,62] = 1
inds = np.where(data[:,63] == 14)
data[inds,62] = 1
inds = np.where(data[:,63] == 15)
data[inds,62] = 1

np.savetxt('wEBD_ruru.txt',data)
#np.savetxt('wE_kf.txt',data)

def test_all_ruru():

    data = np.loadtxt('wE_ruru.txt')
    # Split into training and testing and check of positive class examples in the split
    ind = np.random.permutation(np.shape(data)[0])
    print("Energy all")
    test_classifiers(data,ind)

    data = np.loadtxt('wEB_ruru.txt')
    print("Energy bandpass ")
    test_classifiers(data,ind)

    data = np.loadtxt('wEBD_ruru.txt')
    print("Energy bandpass, denoised ")
    test_classifiers(data,ind)

def test_all_kiwi():

    #nodes = [44, 43, 16, 62]
    #nodes = [44, 43, 40, 61, 60, 34, 16,62]
    data = np.loadtxt('wE_km.txt')
    # Split into training and testing and check of positive class examples in the split
    ind = np.random.permutation(np.shape(data)[0])
    print("Energy all, male")
    test_classifiers2(data[:,nodes],ind)

    data = np.loadtxt('wE_kf.txt')
    print("Energy all, female")
    test_classifiers2(data[:,nodes],ind)

    data = np.loadtxt('wEB_km.txt')
    print("Energy bandpass, male")
    test_classifiers2(data[:,nodes],ind)

    data = np.loadtxt('wEB_kf.txt')
    print("Energy bandpass, female")
    test_classifiers2(data[:,nodes],ind)

    data = np.loadtxt('wEBD_km.txt')
    print("Energy bandpass, denoised, male")
    test_classifiers2(data[:,nodes],ind)

    data = np.loadtxt('wEBD_kf.txt')
    print("Energy bandpass, denoised, female")
    test_classifiers2(data[:,nodes],ind)

print np.where(data[ind[:1000],62]==1)
print np.where(data[ind[1000:],62]==1)

# A variety of learners. So if they use col 62, they are binary, col 63 is multiclass
from sklearn.svm import SVC
clf = SVC()

clf.fit(data[ind[:1000],:62], data[ind[:1000],62]) 
clf.score(data[ind[1000:],:62], data[ind[1000:],62])

out = clf.predict(data[ind[1000:],:62])
print clf, falseneg, falsepos
#falseneg = np.shape(np.where(out-data[ind[1000:],62]<0))[1]
#falsepos = np.shape(np.where(out-data[ind[1000:],62]>0))[1]
#print clf, falseneg, falsepos

#clf = SVC(decision_function_shape='ovr',class_weight='balanced',verbose=True)
#clf.fit(data[ind[:1000],:62], data[ind[:1000],62]) 

def test_classifiers(data,ind):
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(data[ind[:1000],:62], data[ind[:1000],62]) 
    print clf.score(data[ind[1000:],:62], data[ind[1000:],62])
    out = clf.predict(data[ind[1000:],:62])
    print(confusion_matrix(data[ind[1000:],62], out))
    
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    clf.fit(data[ind[:1000],:62], data[ind[:1000],62])
    print clf.score(data[ind[1000:],:62], data[ind[1000:],62])
    out = clf.predict(data[ind[1000:],:62])
    print(confusion_matrix(data[ind[1000:],62], out))
    
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(10, 10), random_state=1)
    clf.fit(data[ind[:1000],:62], data[ind[:1000],62])
    print clf.score(data[ind[1000:],:62], data[ind[1000:],62])
    out = clf.predict(data[ind[1000:],:62])
    print(confusion_matrix(data[ind[1000:],62], out))
    
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier().fit(data[ind[:1000],:62],data[ind[:1000],62])
    out = xgb_model.predict(data[ind[1000:],:62])
    a = confusion_matrix(data[ind[1000:],62], out)
    print float(a[0,0]+a[1,1])/np.sum(a) 
    print a

def test_classifiers2(data,ind):
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(data[ind[:1000],:-1], data[ind[:1000],-1]) 
    print clf.score(data[ind[1000:],:-1], data[ind[1000:],-1])
    out = clf.predict(data[ind[1000:],:-1])
    print(confusion_matrix(data[ind[1000:],-1], out))
    
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    clf.fit(data[ind[:1000],:-1], data[ind[:1000],-1])
    print clf.score(data[ind[1000:],:-1], data[ind[1000:],-1])
    out = clf.predict(data[ind[1000:],:-1])
    print(confusion_matrix(data[ind[1000:],-1], out))
    
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(10, 10), random_state=1)
    clf.fit(data[ind[:1000],:-1], data[ind[:1000],-1])
    print clf.score(data[ind[1000:],:-1], data[ind[1000:],-1])
    out = clf.predict(data[ind[1000:],:-1])
    print(confusion_matrix(data[ind[1000:],-1], out))
    
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier().fit(data[ind[:1000],:-1],data[ind[:1000],-1])
    out = xgb_model.predict(data[ind[1000:],:-1])
    a = confusion_matrix(data[ind[1000:],-1], out)
    print float(a[0,0]+a[1,1])/np.sum(a) 
    print a

def best_classifiers(data,ind):
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier().fit(data[ind[:1000],:62],data[ind[:1000],62])
    out = xgb_model.predict(data[ind[1000:],:62])
    a = confusion_matrix(data[ind[1000:],62], out)
    print float(a[0,0]+a[1,1])/np.sum(a)
    print a

    from sklearn.externals import joblib
    #joblib.dump(xgb_model,'femaleKiwiClassifier.pkl')
    #joblib.dump(xgb_model,'maleKiwiClassifier.pkl')
    joblib.dump(xgb_model,'ruruClassifier.pkl')
    #clf2 = joblib.load('maleKiwiClassifier.pkl')
    #out = clf2.predict(data[ind[1000:],:62])
    #a = confusion_matrix(data[ind[1000:],62], out)
    #print a


def computeWaveletEnergy(fwData,sampleRate):
    # Get the energy of the nodes in the wavelet packet decomposition
    # There are 62 coefficients up to level 5 of the wavelet tree (without root), and 300 seconds in 5 mins
    # The energy is the sum of the squares of the data in each node divided by the total in the tree
    totalTime = int(float(len(fwData))/sampleRate)
    coefs = np.zeros((62, totalTime))
    for t in range(totalTime):
        E = []
        for level in range(1,6):
            wp = pywt.WaveletPacket(data=fwData[t * sampleRate:(t + 1) * sampleRate], wavelet=wavelet, mode='symmetric', maxlevel=level)
            e = np.array([np.sum(n.data**2) for n in wp.get_level(level, "natural")])
            if np.sum(e)>0:
                e = 100.0*e/np.sum(e)
            E = np.concatenate((E, e),axis=0)
        coefs[:, t] = E
    return coefs

import wavio
wavobj = wavio.read('Sound Files/tril1.wav')
sampleRate = wavobj.rate
data = np.squeeze(wavobj.data)
coefs = computeWaveletEnergy(data,sampleRate)

clf = joblib.load('ruruClassifier.pkl')
out = clf.predict(coefs)
print out
# a = confusion_matrix(data[ind[1000:],62], out)
# print a