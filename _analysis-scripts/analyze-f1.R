
options(stringsAsFactors = F)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(rjson)

## SETTINGS AND HELPERS
indir = "~/Documents/kiwis/soundchppaper/p3surveys/"

# Reads all AviaNZ style annotations from directory dir,
# over recorders recs (string vector).
# Does some basic date conversions and returns a df.
# LSK version
readAnnots_K = function(dir, recs){
    annot = data.frame()
    for(rec in recs){
        gooddata = list.files(dir, pattern=paste0(rec, "_.*wav.data"), recursive=T)
        for(f in gooddata){
            a = fromJSON(file=paste(dir, f, sep="/"))
            if(length(a)>1){
                a = a[-1] # drop metadata
                a = data.frame(t(sapply(a, c))) # to dataframe
                tstamp = gsub("Z[A-Z]_(.*).wav.data", "\\1", f)
                a$time = parse_date_time(tstamp, "Ymd_HMS")
                a$rec = rec
                annot = rbind(annot, a)
            }
        }
        if(length(gooddata)==0) print(paste("Warning: no files found for recorder", rec))
    }
    if(nrow(annot)==0) print("Warning: no files read!")
    
    # actual start and end of call
    annot$start = annot$time + seconds(annot$X1)
    annot$end = annot$time + seconds(annot$X2)
    
    # JSON annotations are read in as list columns, so convert/drop those]
    annot$species = unlist(lapply(annot$X5, function(x) x[[1]]$species))
    annot$calltype = unlist(lapply(annot$X5, function(x) x[[1]]$calltype))
    annot = annot[,6:ncol(annot)]
    
    annot$calllength = annot$end - annot$start
    
    return(annot)
}
# Reads all AviaNZ style annotations from directory dir,
# over recorders recs (string vector).
# Does some basic date conversions and returns a df.
# Bittern version
readAnnots_B = function(dir, recs){
    annot = data.frame()
    for(rec in recs){
        gooddata = list.files(paste(dir, rec, sep="/"), pattern=".*wav.data", recursive=T)
        for(f in gooddata){
            a = fromJSON(file=paste(dir, rec, f, sep="/"))
            if(length(a)>1){
                a = a[-1] # drop metadata
                a = data.frame(t(sapply(a, c))) # to dataframe
                tstamp = gsub("(.*).wav.data", "\\1", f)
                if(nchar(tstamp)>13){
                    a$time = parse_date_time(tstamp, "Ymd_HMS")  
                } else {
                    a$time = parse_date_time(tstamp, "dmy_HMS")
                }
                a$rec = rec
                annot = rbind(annot, a)
            }
        }
        if(length(gooddata)==0) print(paste("Warning: no files found for recorder", rec))
    }
    if(nrow(annot)==0) print("Warning: no files read!")
    
    # actual start and end of call
    annot$start = annot$time + seconds(annot$X1)
    annot$end = annot$time + seconds(annot$X2)
    
    # JSON annotations are read in as list columns, so convert/drop those
    annot$species = unlist(lapply(annot$X5, function(x) x[[1]]$species))
    annot = annot[,6:ncol(annot)]
    
    annot$calllength = annot$end - annot$start
    
    return(annot)
}

getTPs = function(ref, pred){
    refmatched = rep(0, nrow(ref))
    predmatched = rep(0, nrow(pred))
    # take each predicted event and find a ref match, if possible
    for(i in 1:nrow(pred)){
        print(i)
        for(j in 1:nrow(ref)){
            # check if the event hasn't been assigned yet, and then for overlap
            if(refmatched[j]==0 & pred$rec[i]==ref$rec[j] & pred$time[i]==ref$time[j] &
               pred$start[i]<=ref$end[j] & pred$end[i]>=ref$start[j]){
                # print("overlapping")
                # print(pred[i,])
                # print(ref[j,])
                refmatched[j]=i
                predmatched[i]=j
            }
        }
    }
    TP = sum(refmatched>0)
    return(TP)
}

# filters GT dataframe to match the recorder
# passed as a vector of repeated values in summarise
extr_helper = function(column, gtdf, preddf){
    gtdf = filter(gtdf, rec==unique(column))
    preddf = filter(preddf, rec==unique(column))
    tp = getTPs(gtdf, preddf)
    return(tp)
}


## READ IN ANNOTATIONS (raw and reviewed)
recs = c("ZA", "ZC", "ZE", "ZG", "ZH", "ZI", "ZJ", "ZK")
ans_raw_wf = readAnnots_K(paste(indir, "rawannots", "WF", "LSK", sep="/"), recs)
ans_raw_chp = readAnnots_K(paste(indir, "rawannots", "CHP", "LSK", sep="/"), recs)
ans_raw_mc = readAnnots_K(paste(indir, "rawannots", "MC", "LSK", sep="/"), recs)
ans_gt = readAnnots_K(paste(indir, "reviewed", "manual", "LSK", sep="/"), recs)

# basically apply our function over groups (recs)
# with dplyr handling the group mapping-reducing
scores_wf = group_by(ans_raw_wf, rec) %>%
    summarize(det="wf", tps = extr_helper(rec, ans_gt, .), npreds=nrow(.))
scores_mc = group_by(ans_raw_mc, rec) %>%
    summarize(det="mc", tps = extr_helper(rec, ans_gt, .), npreds=nrow(.))
scores_chp = group_by(ans_raw_chp, rec) %>%
    summarize(det="chp", tps = extr_helper(rec, ans_gt, .), npreds=nrow(.))
scores_K = bind_rows(scores_wf, scores_mc, scores_chp) %>%
    group_by(det) %>%
    summarize(tps=sum(tps), precision=sum(tps)/unique(npreds),
          recall=sum(tps)/nrow(ans_gt), f1=2*precision*recall/(precision+recall))
scores_K


# Repeat all for bittern
recs = paste0("BIT", 1:9)
ans_raw_wf = readAnnots_B(paste(indir, "rawannots", "WF", "Bittern", sep="/"), recs)
ans_raw_chp = readAnnots_B(paste(indir, "rawannots", "CHP", "Bittern", sep="/"), recs)
ans_raw_mc = readAnnots_B(paste(indir, "rawannots", "MC", "Bittern", sep="/"), recs)
ans_gt = readAnnots_B(paste(indir, "reviewed", "manual", "Bittern", sep="/"), recs)

scores_wf = group_by(ans_raw_wf, rec) %>%
    summarize(det="wf", tps = extr_helper(rec, ans_gt, .), npreds=nrow(.))
scores_mc = group_by(ans_raw_mc, rec) %>%
    summarize(det="mc", tps = extr_helper(rec, ans_gt, .), npreds=nrow(.))
scores_chp = group_by(ans_raw_chp, rec) %>%
    summarize(det="chp", tps = extr_helper(rec, ans_gt, .), npreds=nrow(.))
scores_B = bind_rows(scores_wf, scores_mc, scores_chp) %>%
    group_by(det) %>%
    summarize(tps=sum(tps), precision=sum(tps)/unique(npreds),
              recall=sum(tps)/nrow(ans_gt), f1=2*precision*recall/(precision+recall))
scores_B
