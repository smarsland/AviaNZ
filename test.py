import Training
import NN
import tempfile
import numpy as np

def makeModel(architecture,configdir,filtdir,trainDirName,recogniser,anntlevel):
    nntrain = Training.NNtrain(configdir, filtdir)
    nntrain.setP1(trainDirName,"",recogniser,anntlevel)
    nntrain.readFilter()
    nntrain.genSegmentDataset(True)
    nntrain.windowWidth = nntrain.imgsize[0] * 2
    nntrain.windowInc = int(np.ceil(100 * nntrain.fs / (nntrain.imgsize[1] - 1)) / 100)
    nntrain.imgWidth = 1
    nntrain.modelArchitecture = architecture
    return nntrain

configdir = "/home/giotto/.avianz/"
filtdir = "/home/giotto/.avianz/Filters/"
trainDirName = "/home/giotto/Desktop/AviaNZ/Sound Files/learning/trainandtest/"
recogniser = "KakaCHP"
anntlevel = "All-nowt"

# CNN = makeModel("CNN",configdir,filtdir,trainDirName,recogniser,anntlevel)
# CNN.train()

AST = makeModel("AudioSpectogramTransformer (pre-trained ViT)",configdir,filtdir,trainDirName,recogniser,anntlevel)
AST.train()