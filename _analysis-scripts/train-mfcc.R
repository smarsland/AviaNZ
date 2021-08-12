# USAGE: Rscript train-mfcc.R INDIR
# with INDIR - a dir that contains .mfcc and .txt files
args = commandArgs(TRUE)
INDIR = args[1]
# or:
# INDIR = "~/Documents/audiodata/DCASE/tmp/"

library(glmnet)
setwd(INDIR)

filelist = c("Y0GGO5npJzWI_0.000_10.000.wav", "Y0G-Qx3Gv01s_0.000_8.000.wav", "Y0UKzyBqyhGc_27.000_37.000.wav",
             "Y0QtmEs-OOsA_21.000_31.000.wav", "Y0SjvusZqA3w_0.000_6.000.wav", "Y0Sq8ge7D4wU_3.000_13.000.wav",
             "Y0W_gYR6p-D0_30.000_40.000.wav",  # test files
             "Y0632OqvXrwg_7.000_17.000.wav", "Y0DtJdRFPmS4_30.000_40.000.wav", "Y0E6Uaq_e6OA_20.000_30.000.wav",
             "Y0F40MJSfsDw_30.000_40.000.wav", "Y0uab4-3d6MM_30.000_40.000.wav", # wf train files
             "Y-_OktjRkoSk_110.000_120.000.wav", "Y4oQdHNxBFFY_20.000_30.000.wav",  # speech-only files
             "Y-3cmXbOJOoc_30.000_40.000.wav", "Y-XLzWTb3sUg_30.000_40.000.wav", "Y0XDz4i5L4nE_30.000_40.000.wav",
             "Y0BSidlvh4Ds_30.000_40.000.wav", "Y0F1zmrJb1tI_30.000_40.000.wav",
             "Y1GUZ1qlXK8M_70.000_80.000.wav", "Y2gSMufsNQKA_30.000_40.000.wav", "Y3BcwNM72zcQ_10.000_20.000.wav"
)

## TRAIN DATA
mfccAll = list()
interpairsAll = list()
gtAll = c()
for(f in filelist){
    print(f)
    # read in the features (a matrix over time windows)
    mfcc = t(read.table(paste0(f, ".mfcc")))
    mfcc = scale(mfcc)
    mfccAll[[length(mfccAll)+1]] = mfcc
    # read in the ground truth (0/1 for each window)
    gt = as.numeric(readLines(paste0(f, ".txt")))
    gtAll = c(gtAll, gt)
    # Create all pairwise interactions
    interpairs = combn(1:ncol(mfcc), 2)
    interpairs = apply(interpairs, 2, function(pair) mfcc[,pair[1]] * mfcc[,pair[2]])
    interpairsAll[[length(interpairsAll)+1]] = interpairs
}
mfccAll = do.call(rbind, mfccAll)
interpairsAll = do.call(rbind, interpairsAll)

image(mfccAll)

# Fit a LASSO-penalized logistic regression model, with CV to set penalty strength
cvLasso = cv.glmnet(cbind(mfccAll, interpairsAll), gtAll, alpha=1, family="binomial", maxit=2000)
plot(cvLasso); cvLasso$lambda.1se  # around 0.0025-0.0035
regrLasso = glmnet(cbind(mfccAll, interpairsAll), gtAll, alpha=1, maxit=3000,
                   family="binomial", lambda=cvLasso$lambda[42])
sum(coef(regrLasso)==0)
head(coef(regrLasso))

# just a check to see how accurate it is on the same training data
predsOnTrain = as.numeric(predict(regrLasso, cbind(mfccAll, interpairsAll), type="response")>0.5)
table(predsOnTrain, gtAll)

# store the coefs
writeLines(as.character(as.numeric(coef(regrLasso))), "~/.avianz/model-mfcc.txt")
