## Plots the annotations for the applications examples (speech, gunshot etc.).
## Use the script erase-freq-info.R beforehand, to convert
## band-specific annotations into full-spectrum ones.

## Filters are distributed with AviaNZ; for sound file sources,
## see the publication. Applying the filters to these sound files via AviaNZ
## Segment function will recreate the annotations.
## To export spectrogram PNGs, use "Export spectrogram image" function of AviaNZ.

options(stringsAsFactors = F)
library(dplyr)
library(tidyr)
library(ggplot2)
library(rjson)
library(raster)
library(cowplot)

setwd("~/Documents/kiwis/soundchppaper/examples_results/")
# datafs = c("20130605_021919.wav.data",
#            "mixture_devtrain_gunshot_404_5cbccf7f42125fda77a0d8cb110c6ccd.wav.data",
#            "Y4Yo_DkHXXjI_170.000_180.000.wav.data",
#            "Y-6-rh8kbZf0_40.000_50.000.wav.data",
#            "ST_10750016.wav.data")


## load pngs: FIRST SPEECH EXAMPLE
f = "Y-6-rh8kbZf0_40.000_50.000.wav.data"

spec = raster(gsub(".wav.data", ".png", f))
spec  = flip(spec, 'y')
timescaling = 10/(extent(spec)@xmax-extent(spec)@xmin)
freqscaling = 8/(extent(spec)@ymax-extent(spec)@ymin)
spec = data.frame(rasterToPoints(spec))
colnames(spec)[3] = "vol"

a = fromJSON(file=gsub(".wav", "_STANDARD.wav", f))
a = data.frame(t(sapply(a[-1], function(x) c(x[1:4]))))
a = unnest(a)
colnames(a) = c("start", "end", "fl", "fu")
a$filename = f

p1 = ggplot(a) + 
  geom_raster(data=spec, aes(x=x*timescaling,y=y*freqscaling,fill=vol)) +
  geom_rect(aes(xmin=start, xmax=end, ymin=-0.2, ymax=8.2), col="tomato", fill="red", alpha=0.1) + 
  scale_fill_gradient(low="#000000", high="#ffffff", guide="none") +
  scale_x_continuous(expand=c(0.01,0)) + scale_y_continuous(expand=c(0,0)) + 
  theme_minimal() + xlab(NULL) + ylab("kHz") + theme(axis.ticks = element_line(colour="grey20"))

a = fromJSON(file=gsub(".wav", "_PROPOSED.wav", f))
a = data.frame(t(sapply(a[-1], function(x) c(x[1:4]))))
a = unnest(a)
colnames(a) = c("start", "end", "fl", "fu")
a$filename = f

p2 = ggplot(a) + 
  geom_raster(data=spec, aes(x=x*timescaling,y=y*freqscaling,fill=vol)) +
  geom_rect(aes(xmin=start, xmax=end, ymin=-0.2, ymax=8.2), col="tomato", fill="red", alpha=0.1) + 
  scale_fill_gradient(low="#000000", high="#ffffff", guide="none") +
  scale_x_continuous(expand=c(0.01,0)) + scale_y_continuous(expand=c(0,0), breaks=c(0,2,4,6)) + 
  theme_minimal() + xlab(NULL) + ylab(NULL) + theme(axis.ticks = element_line(colour="grey20"))


## load pngs: SECOND SPEECH EXAMPLE
f = "Y4Yo_DkHXXjI_170.000_180.000.wav.data"

spec = raster(gsub(".wav.data", ".png", f))
spec  = flip(spec, 'y')
timescaling = 10/(extent(spec)@xmax-extent(spec)@xmin)
freqscaling = 8/(extent(spec)@ymax-extent(spec)@ymin)
spec = data.frame(rasterToPoints(spec))
colnames(spec)[3] = "vol"

a = fromJSON(file=gsub(".wav", "_STANDARD.wav", f))
a = data.frame(t(sapply(a[-1], function(x) c(x[1:4]))))
a = unnest(a)
colnames(a) = c("start", "end", "fl", "fu")
a$filename = f

p3 = ggplot(a) + 
  geom_raster(data=spec, aes(x=x*timescaling,y=y*freqscaling,fill=vol)) +
  geom_rect(aes(xmin=start, xmax=end, ymin=-0.2, ymax=8.2), col="tomato", fill="red", alpha=0.1) + 
  scale_fill_gradient(low="#000000", high="#ffffff", guide="none") +
  scale_x_continuous(expand=c(0.01,0)) + scale_y_continuous(expand=c(0,0)) + 
  theme_minimal() + xlab(NULL) + ylab("kHz") + theme(axis.ticks = element_line(colour="grey20"))

a = fromJSON(file=gsub(".wav", "_PROPOSED.wav", f))
a = data.frame(t(sapply(a[-1], function(x) c(x[1:4]))))
a = unnest(a)
colnames(a) = c("start", "end", "fl", "fu")
a$filename = f

p4 = ggplot(a) + 
  geom_raster(data=spec, aes(x=x*timescaling,y=y*freqscaling,fill=vol)) +
  geom_rect(aes(xmin=start, xmax=end, ymin=-0.2, ymax=8.2), col="tomato", fill="red", alpha=0.1) + 
  scale_fill_gradient(low="#000000", high="#ffffff", guide="none") +
  scale_x_continuous(expand=c(0.01,0)) + scale_y_continuous(expand=c(0,0), breaks=c(0,2,4,6)) + 
  theme_minimal() + xlab(NULL) + ylab(NULL) + theme(axis.ticks = element_line(colour="grey20"))


## load pngs: KIWI CALLS
f = "20130605_021919.wav.data"

spec = raster(gsub(".wav.data", ".png", f))
spec  = flip(spec, 'y')
# crop a bit, to reduce the raster size. This does complicate the plotting expressions a bit
# tocrop = extent(spec)
# wid = (tocrop@xmax - tocrop@xmin)
# tocrop@xmin = 0.25*wid
# tocrop@xmax = 0.75*wid
# spec = crop(spec, tocrop)
timescalingK = 130/(extent(spec)@xmax-extent(spec)@xmin)
freqscalingK = 8/(extent(spec)@ymax-extent(spec)@ymin)
spec = data.frame(rasterToPoints(spec))
colnames(spec)[3] = "vol"

a = fromJSON(file=gsub(".wav", "_LONG.wav", f))
a = data.frame(t(sapply(a[-1], function(x) c(x[1:4]))))
a = unnest(a)
colnames(a) = c("start", "end", "fl", "fu")
a$filename = f

p5 = ggplot(a) + 
  geom_raster(data=spec, aes(x=x*timescalingK,y=y*freqscalingK,fill=vol)) +
  geom_rect(aes(xmin=start, xmax=end, ymin=-0.2, ymax=8.2), col="tomato", fill="red", alpha=0.1) + 
  scale_fill_gradient(low="#000000", high="#ffffff", guide="none") +
  scale_x_continuous(limits=c(30, 90), expand=c(0.01,0)) + scale_y_continuous(expand=c(0,0)) + 
  theme_minimal() + xlab(NULL) + ylab("kHz") + theme(axis.ticks = element_line(colour="grey20"))


a = fromJSON(file=gsub(".wav", "_SHORT.wav", f))
a = data.frame(t(sapply(a[-1], function(x) c(x[1:4]))))
a = unnest(a)
colnames(a) = c("start", "end", "fl", "fu")
a$filename = f

p6 = ggplot(a) + 
  geom_raster(data=spec, aes(x=x*timescalingK,y=y*freqscalingK,fill=vol)) +
  geom_rect(aes(xmin=start, xmax=end, ymin=-0.2, ymax=8.2), col="tomato", fill="red", alpha=0.1) + 
  scale_fill_gradient(low="#000000", high="#ffffff", guide="none") +
  scale_x_continuous(limits=c(30, 90), expand=c(0.01,0)) + scale_y_continuous(expand=c(0,0), breaks=c(0,2,4,6)) + 
  theme_minimal() + xlab(NULL) + ylab(NULL) + theme(axis.ticks = element_line(colour="grey20"))


## load pngs: GUNSHOTS
f = "mixture_devtrain_gunshot_404_5cbccf7f42125fda77a0d8cb110c6ccd.wav.data"

spec = raster(gsub(".wav.data", ".png", f))
spec  = flip(spec, 'y')
timescalingG = 30/(extent(spec)@xmax-extent(spec)@xmin)
freqscalingG = 8/(extent(spec)@ymax-extent(spec)@ymin)
spec = data.frame(rasterToPoints(spec))
colnames(spec)[3] = "vol"

a = fromJSON(file=gsub(".wav", "_PROPOSED.wav", f))
a = data.frame(t(sapply(a[-1], function(x) c(x[1:4]))))
a = unnest(a)
colnames(a) = c("start", "end", "fl", "fu")
a$filename = f

# note that we actually manually make the mark less accurate for visualization
p7 = ggplot(a) + 
  geom_raster(data=spec, aes(x=x*timescalingG,y=y*freqscalingG,fill=vol)) +
  geom_rect(aes(xmin=start-0.05, xmax=end+0.05, ymin=-0.2, ymax=8.2), col="tomato", fill="red", alpha=0.1, size=0.5) + 
  scale_fill_gradient(low="#000000", high="#ffffff", guide="none") +
  scale_x_continuous(limits=c(10, 24), expand=c(0.01,0)) + scale_y_continuous(expand=c(0,0)) + 
  theme_minimal() + xlab("Time (s)") + ylab("kHz") + theme(axis.ticks = element_line(colour="grey20"))


## load pngs: BATS
f = "ST_10750016.wav.data"

spec = raster(gsub(".wav.data", ".png", f))
spec  = flip(spec, 'y')
timescalingB = 2.407/(extent(spec)@xmax-extent(spec)@xmin)
freqscalingB = 61.8/(extent(spec)@ymax-extent(spec)@ymin)
spec = data.frame(rasterToPoints(spec))
colnames(spec)[3] = "vol"

a = fromJSON(file=gsub(".wav", "_PROPOSED.wav", f))
a = data.frame(t(sapply(a[-1], function(x) c(x[1:4]))))
a = unnest(a)
colnames(a) = c("start", "end", "fl", "fu")
a$filename = f

# note that we actually manually make the mark less accurate for visualization
p8 = ggplot(a) + 
  geom_raster(data=spec, aes(x=x*timescalingB,y=y*freqscalingB,fill=vol)) +
  geom_rect(aes(xmin=start-0.01, xmax=end+0.01, ymin=-1.5, ymax=63.2), col="tomato", fill="red", alpha=0.1, size=0.5) + 
  scale_fill_gradient(low="#000000", high="#ffffff", guide="none") +
  scale_x_continuous(expand=c(0.01,0)) + scale_y_continuous(expand=c(0,0)) + 
  theme_minimal() + xlab("Time (s)") + ylab("kHz") + theme(axis.ticks = element_line(colour="grey20"))


## SAVE OUTPUT
plot_grid(p1, p2, p3, p4, p5, p6, p7, p8, nrow=4, ncol=2, labels="AUTO")
ggsave("fig_allexamples.pdf")
