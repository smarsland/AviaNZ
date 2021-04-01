## This script erases frequency information to help plotting.
## (Standard AviaNZ annotations for band-specific filters
## produce band-specific annotations, but those are not good for
## visualizing, so this script converts annotations to full-spectrum ones.)

options(stringsAsFactors = F)
library(dplyr)
library(tidyr)
library(ggplot2)
library(rjson)

setwd("~/Documents/audiodata/p3samples/examples/")
datafs = c("20130605_021919.wav.data",
           "mixture_devtrain_gunshot_404_5cbccf7f42125fda77a0d8cb110c6ccd.wav.data",
           "Y4Yo_DkHXXjI_170.000_180.000.wav.data",
           "Y-6-rh8kbZf0_40.000_50.000.wav.data")

# parse annotations
for(f in datafs){
  print(paste("working on file", f))
  a = fromJSON(file=f)
  for(segi in 2:length(a)){
    a[[segi]][[3]] = 0
    a[[segi]][[4]] = 0
  }
  writeLines(toJSON(a), f) 
}
