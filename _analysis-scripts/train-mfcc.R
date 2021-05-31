library(dplyr)
library(tidyr)
library(ggplot2)
library(glmnet)
setwd("~/Documents/audiodata/DCASE/tmp/")


## TRAIN DATA
mfccAll = list()
gtAll = c()
for(f in c("Y0632OqvXrwg_7.000_17.000.wav", "Y0DtJdRFPmS4_30.000_40.000.wav", "Y0E6Uaq_e6OA_20.000_30.000.wav",
           "Y0F40MJSfsDw_30.000_40.000.wav", "Y0uab4-3d6MM_30.000_40.000.wav",
           "Y-_OktjRkoSk_110.000_120.000.wav", "Y4oQdHNxBFFY_20.000_30.000.wav",  # speech-only files
           "Y-3cmXbOJOoc_30.000_40.000.wav", "Y-XLzWTb3sUg_30.000_40.000.wav", "Y0XDz4i5L4nE_30.000_40.000.wav"
           )){
    print(f)
    # read in the features (a matrix over time windows)
    mfcc = t(read.table(paste0(f, ".mfcc")))
    mfcc = scale(mfcc)
    mfccAll[[length(mfccAll)+1]] = mfcc
    # read in the ground truth (0/1 for each window)
    gt = as.numeric(readLines(paste0(f, ".txt")))
    gtAll = c(gtAll, gt)
}
mfccAll = do.call(rbind, mfccAll)

image(mfccAll)

# Create all pairwise interactions
interpairsAll = combn(1:ncol(mfccAll), 2)
interpairsAll = apply(interpairsAll, 2, function(pair) mfccAll[,pair[1]] * mfccAll[,pair[2]])
interpairsAll = scale(interpairsAll)

# Fit a LASSO-penalized logistic regression model, with CV to set penalty strength
cvLasso = cv.glmnet(cbind(mfccAll, interpairsAll), gtAll, alpha=1, family="binomial")
plot(cvLasso)
regrLasso = glmnet(cbind(mfccAll, interpairsAll), gtAll, alpha=1, family="binomial", lambda=cvLasso$lambda.1se)
coef(regrLasso)

# just a check to see how accurate it is on the same training data
predsOnTrain = as.numeric(predict(regrLasso, cbind(mfccAll, interpairsAll), type="response")>0.5)
table(predsOnTrain, gtAll)

# store the coefs
writeLines(as.character(as.numeric(coef(regrLasso))), "~/.avianz/model-mfcc-zcr-lasso.txt")


## TEST DATA
# Optional: can also extract the features from some files from the testing data
# to see if the model generalizes.

# mfcc = t(read.table("Y0QtmEs-OOsA_21.000_31.000.wav.mfcc"))
# gt = as.numeric(readLines("Y0QtmEs-OOsA_21.000_31.000.wav.txt"))
# 
# mfcc = t(read.table("Y0GGO5npJzWI_0.000_10.000.wav.mfcc"))
# gt = as.numeric(readLines("Y0GGO5npJzWI_0.000_10.000.wav.txt"))
# 
# mfccS = scale(mfcc)
# 
# mfccTest = mfccS[, 2:24]
# interpairsNew = combn(1:ncol(mfccTest), 2)
# interpairsNew = apply(interpairsNew, 2, function(pair) mfccTest[,pair[1]] * mfccTest[,pair[2]])
# interpairsNew = scale(interpairsNew)
# 
# posPreds = which(predict(regrLasso, newx=cbind(mfccTest, interpairsNew), type="response")>0.5)
# image(mfccS)
# points(x=posPreds/length(gt), y=rep(0, length(posPreds)), col="red", pch=12)
