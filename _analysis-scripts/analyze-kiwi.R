## Analyse the results of LSK Zealandia survey:
## compare the raw and reviewed annotations obtained by three detection methods
## and run SCR.

## Requires:
## * raw and reviewed annotations in .data format
##   (in dirs structured like rawannots/CHP/LSK/, reviewed/MC/LSK/, ...)
## * clock adjustment vector determined with CompareCalls.py
## * recorder GPS positions and Zealandia shapefile in this repo

options(stringsAsFactors = F)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(rjson)
library(rgdal)
library(ascr)

## SETTINGS AND HELPERS
indir = "~/Documents/kiwis/soundchppaper/p3surveys/"
outdir = "~/Documents/kiwis/soundchppaper/p3surveys/LSK/"

# Reads all AviaNZ style annotations from directory dir,
# over recorders recs (string vector).
# Does some basic date conversions and returns a df.
readAnnots = function(dir, recs){
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
  
  # JSON annotations are read in as list columns, so convert/drop those
  annot$species = unlist(lapply(annot$X5, function(x) x[[1]]$species))
  annot$calltype = unlist(lapply(annot$X5, function(x) x[[1]]$calltype))
  annot = annot[,6:11]
  
  annot$calllength = annot$end - annot$start
  
  return(annot)
}

# ggplot syntax for plotting raw & reviewed annotations.
# only prints one day, and expects a $start column
plotDayAnnots = function(df){
  if(min(day(df$start))==5 & max(day(df$start))==5){
    minbreaks = seq(ymd_hms("2018-11-05 22:00:00"), ymd_hms("2018-11-06 00:00:00"), by=60)
  } else if(min(day(df$start))==6 & max(day(df$start))==6) {
    minbreaks = seq(ymd_hms("2018-11-06 02:00:00"), ymd_hms("2018-11-06 04:00:00"), by=60)
  } else {
    stop("Error: supplied multiple days")
  }
  
  p = ggplot(df) + geom_segment(aes(x=start, xend=end, y=rec, yend=rec, col=stage), size=5) +
    facet_wrap(~quarter, scales="free_x", nrow=4) +
    scale_color_manual(values=c("goldenrod1", "green4")) +
    scale_x_datetime(expand=c(0.01,0.01)) + 
    theme_minimal() + theme(panel.grid.major.y = element_blank(), strip.background = element_blank(),
                            strip.text = element_blank())
  print(p)
}

# Converts start-end type annotations into
# presence-absence over each second in the survey.
# Requires columns $daystart and $startadj, and $calllength.
convtoPresAbs = function(df){
  # create a vector of seconds along the entire survey
  df = mutate(df, relday=day(start)-min(day(start)))
  if(length(unique(df$relday))==1){
    timerange = list(seq(0, hms::as_hms(round(max(df$endadj)))-hms::as_hms(round(min(df$startadj)))))
  } else {
    # apply converts hms to chars, so this is a bit ugly:
    timerange = group_by(df, relday) %>%
      summarize(mint = hms::as_hms(round(min(startadj))), maxt = hms::as_hms(round(max(endadj)))) %>%
      apply(., 1, function(x) seq(0, hms::parse_hms(x[3])-hms::parse_hms(x[2])))
  }
  
  # this is a list of dfs for each day:
  presence = lapply(seq_along(timerange), function(id) data.frame(secs = paste(id, timerange[[id]], sep="_"),
                                                                  ZA=0, ZC=0, ZE=0, ZG=0, ZH=0, ZI=0, ZJ=0, ZK=0))
  
  # convert start and end to relative times for that day
  df = group_by(df, relday) %>%
    mutate(start = as.numeric(startadj-min(startadj), units="secs")) %>%
    mutate(end = start + calllength)
  # Convert into 0/1 by second
  for(r in 1:nrow(df)){
    callstart = floor(df$start[r])
    callend = ceiling(df$end[r])
    day = df$relday[r]+1
    rec = df$rec[r]
    presence[[day]][callstart:callend, rec] = 1
  }
  # convert to a single df (flatten over days):
  presence = bind_rows(presence)
  return(presence)
}

# Converts to make.capt style capture history.
# From a presence-absence over each time and recorder,
# extracts a call detection matrix: callId x recorder,
# where callId is the start time of a contiguous presence region
identifyCalls = function(df){
  calls = data.frame()
  df = mutate(df, anycall = ZA | ZC | ZE | ZG | ZH | ZI | ZJ | ZK)
  i = 1
  while(i <= nrow(df)){
    ## find start of call
    if(df$anycall[i]){
      ## call detected on any of the recorders
      callTime = df$secs[i]
      j = i+1
      ## find when this call ends, assuming contiguous call across all recs
      while(df$anycall[j] & j<=nrow(df)){
        j = j+1
      }
      ## find recorders which had that call
      callRecs = names(which(sapply(df[i:j-1, 2:8], any)))
      for(rec in callRecs){
        calls = rbind(calls, data.frame(session=1, id=callTime, occ=1, rec=rec))
      }
      i = j
    }
    i = i+1
  }
  return(calls)
}


## READ IN ANNOTATIONS (raw and reviewed)
recs = c("ZA", "ZC", "ZE", "ZG", "ZH", "ZI", "ZJ", "ZK")
ans_rev_wf = readAnnots(paste(indir, "reviewed", "WF", "LSK", sep="/"), recs)
ans_rev_chp = readAnnots(paste(indir, "reviewed", "CHP", "LSK", sep="/"), recs)
ans_rev_mc = readAnnots(paste(indir, "reviewed", "MC", "LSK", sep="/"), recs)
ans_raw_wf = readAnnots(paste(indir, "rawannots", "WF", "LSK", sep="/"), recs)
ans_raw_chp = readAnnots(paste(indir, "rawannots", "CHP", "LSK", sep="/"), recs)
ans_raw_mc = readAnnots(paste(indir, "rawannots", "MC", "LSK", sep="/"), recs)

ans_rev_wf$method = "WF"
ans_raw_wf$method = "WF"
ans_rev_chp$method = "CHP"
ans_raw_chp$method = "CHP"
ans_rev_mc$method = "MC"
ans_raw_mc$method = "MC"
ans_rev_wf$stage = "reviewed"
ans_raw_wf$stage = "raw"
ans_rev_chp$stage = "reviewed"
ans_raw_chp$stage = "raw"
ans_rev_mc$stage = "reviewed"
ans_raw_mc$stage = "raw"

ans_all = bind_rows(ans_rev_wf, ans_raw_wf, ans_rev_chp, ans_raw_chp, ans_rev_mc, ans_raw_mc)
head(ans_all)
table(ans_all$method, ans_all$stage, day(ans_all$time))

# total num TP/FP seconds by method
group_by(ans_all, method, stage) %>%
  summarize(P=sum(calllength))

# attach this for plotting
ans_all = mutate(ans_all, quarter=pmin(3, ifelse(day(start)==5,
                            difftime(start, ymd_hms("181105 22:00:00"), units="mins"),
                            difftime(start, ymd_hms("181106 02:00:00"), units="mins")) %/% 30))

# clock adjustments, estimated using the CallCompare branch:
clockadjs = tibble(rec=c("ZA", "ZC", "ZE", "ZG", "ZH", "ZI", "ZJ", "ZK"), adjs=c(159,132,0, 82,102,31, 0,34))
ans_all = left_join(ans_all, clockadjs, by="rec") %>%
  mutate(startadj = start+adjs, endadj=end+adjs) %>%
  arrange(rec, stage)


## PLOT ALL CALLS
# some raw data, without syncing clock times:
ans_all %>%
  filter(method=="WF", day(start)==5) %>%
  plotAnnots

# Same plot, times adjusted for clock drift:
ans_all %>%
  filter(method=="WF", day(start)==5) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ans_all %>%
  filter(method=="WF", day(start)==6) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots

# adjusted times, MC method:
ans_all %>%
  filter(method=="MC", day(start)==5) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ans_all %>%
  filter(method=="MC", day(start)==6) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots

# adjusted times, CHP method:
ans_all %>%
  filter(method=="CHP", day(start)==5) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ggsave(paste0(outdir, "detections_CHP_05.png"), width=8, height=6)
ans_all %>%
  filter(method=="CHP", day(start)==6) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ggsave(paste0(outdir, "detections_CHP_06.png"), width=8, height=6)


# equalize review effort (downsample methods with more reported detections)
dsfactor_wf = nrow(ans_raw_chp)/nrow(ans_raw_wf)
dsfactor_mc = nrow(ans_raw_chp)/nrow(ans_raw_mc)

# subset by time: from each 5 min window 04:00-04:05,04:05-04:10,..., take first X %
# where X is chosen to produce equal numbers of annotations for all methods.
# create a relative time counter from that day's start
ans_all = mutate(ans_all, secsfromstart=ifelse(day(start)==5,
                            difftime(startadj, ymd_hms("181105 22:00:00"), units="secs"),
                            difftime(startadj, ymd_hms("181106 02:00:00"), units="secs")))

ans_final_wf = filter(ans_all, method=="WF", stage=="reviewed") %>%
  filter(secsfromstart %% 300 < dsfactor_wf*300)
ans_final_mc = filter(ans_all, method=="MC", stage=="reviewed") %>%
  filter(secsfromstart %% 300 < dsfactor_mc*300)
ans_final_chp = filter(ans_all, method=="CHP", stage=="reviewed")

# Same plots, but WF and MC effort reduced to match CHP effort:
ans_all %>%
  filter(method=="WF", day(start)==5) %>%
  filter(secsfromstart %% 300 < dsfactor_wf*300) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ggsave(paste0(outdir, "detections_WF_05.png"), width=8, height=6)

ans_all %>%
  filter(method=="WF", day(start)==6) %>%
  filter(secsfromstart %% 300 < dsfactor_wf*300) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ggsave(paste0(outdir, "detections_WF_06.png"), width=8, height=6)

ans_all %>%
  filter(method=="MC", day(start)==5) %>%
  filter(secsfromstart %% 300 < dsfactor_mc*300) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ggsave(paste0(outdir, "detections_MC_05.png"), width=8, height=6)

ans_all %>%
  filter(method=="MC", day(start)==6) %>%
  filter(secsfromstart %% 300 < dsfactor_mc*300) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ggsave(paste0(outdir, "detections_MC_06.png"), width=8, height=6)


## CONVERT TO BINARY DETECTION MATRIX

# extract the right df with the adjusted times
presence_wf = convtoPresAbs(ans_final_wf)
presence_chp = convtoPresAbs(ans_final_chp)
presence_mc = convtoPresAbs(ans_final_mc)

# identify unique contiguous calls and their matches across recorders
calls_wf = identifyCalls(presence_wf)
calls_wf

calls_chp = identifyCalls(presence_chp)
calls_chp

calls_mc = identifyCalls(presence_mc)
calls_mc

# number of calls x recorders
nrow(calls_wf)  # 100
nrow(calls_chp) # 116
nrow(calls_mc)  # 52
# we can see that the ratio enforced at annotation level matches ratio of call detections:
nrow(calls_wf)/nrow(calls_chp); dsfactor_wf
nrow(calls_mc)/nrow(calls_chp); dsfactor_mc


# ------- SCR -------

# GPS DATA:
# create recorder grid, using fixed distances for now
gpspos = read.table("ZealandiaPoints.txt", sep=" ", nrows=10, h=F)
gpspos$V1 = paste0("Z", substr(gpspos$V1, 2, 2))
gpspos = filter(gpspos, V1 %in% recs)
ggplot(gpspos, aes(y=V2, x=V3)) + geom_point() + geom_text(aes(label=V1), nudge_y=0.0005) +
  theme(aspect.ratio = 1)

# Zealandia fence polygon, NZTM
fence = readOGR("zealandia-polygon-merged.shp")

# project lat / long to easting / northing in meters, NZTM
coordinates(gpspos) = c("V3", "V2")
proj4string(gpspos) = CRS("+proj=longlat +datum=WGS84")
gpsposM = spTransform(gpspos,
            CRS("+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"))
gpsposM = data.frame(gpsposM)
colnames(gpsposM) = c("rec", "east", "north", "optional")

# center everything for simpler plotting later:
gpsposM = mutate(gpsposM, east=east-mean(east), north=north-mean(north), optional=NULL)
ggplot(gpsposM, aes(y=north, x=east)) +
  geom_point() + geom_text(aes(label=rec), nudge_y=50) +
  theme_minimal() + theme(aspect.ratio = 1)

# create trap coord matrix and the integration mask
scr_mask_bufsize = 1200
trapgrid = gpsposM %>% mutate(recid=1:n())
traps = as.matrix(trapgrid[,c("east", "north")])
mask = create.mask(traps, buffer=scr_mask_bufsize)

# crop mask to the inside of Zealandia.
# this requires converting the mask into SpatialPoints:
maskNonCentred = mask
maskNonCentred[,1] = maskNonCentred[,1] + mapcentroidE
maskNonCentred[,2] = maskNonCentred[,2] + mapcentroidN
maskDF = SpatialPoints(maskNonCentred, proj4string=CRS(proj4string(fence)))
maskOutside = is.na(over(maskDF, fence)$ID)

# fencepolygon = fence@polygons[[1]]@Polygons[[1]]@coords
# fencepolygon = fencepolygon - matrix(rep(c(mapcentroidE, mapcentroidN), each=nrow(fencepolygon)), ncol=2)

# plot before cropping:
plot(mask)
points(traps, col="red", pch=4)
points(mask[maskOutside,], col="seagreen")

# actually crop the mask:
mask = mask[!maskOutside,]


## RUNNING WITH WF ANNOTS
# Create the call history
capt_wf = inner_join(calls_wf, trapgrid, by="rec")[,c("session", "id", "occ", "recid")]
capt_wf = create.capt(as.data.frame(capt_wf), traps=traps)

# inspect the call history over recorders
capt_wf$bincapt
nrow(capt_wf$bincapt)
table(rowSums(capt_wf$bincapt))

# Fit the model, accounting for the shorter duration
cr = fit.ascr(capt_wf, traps, mask, survey.length = dsfactor_wf)
summary(cr)

detfnparams_wf = cr$coefficients
precision_wf = cr$se

show.detfn(cr, xlim=c(0, scr_mask_bufsize))
show.detsurf(cr)

# Bootstrap to get good SE estimates
boot_wf = boot.ascr(cr, N=500, n.cores=4)
summary(boot_wf)
stdEr(boot_wf, mce=T)
save(boot_wf, file=paste0(outdir, "boot_wf.RData"))


## RUNNING WITH MC ANNOTS
# Create the call history
capt_mc = inner_join(calls_mc, trapgrid, by="rec")[,c("session", "id", "occ", "recid")]
capt_mc = create.capt(as.data.frame(capt_mc), traps=traps)

# inspect the call history over recorders
capt_mc$bincapt
nrow(capt_mc$bincapt)
table(rowSums(capt_mc$bincapt))

cr = fit.ascr(capt_mc, traps, mask, survey.length = dsfactor_mc)
summary(cr)

detfnparams_mc = cr$coefficients
precision_mc = cr$se

show.detfn(cr, xlim=c(0, scr_mask_bufsize))
show.detsurf(cr)

# Bootstrap to get good SE estimates
boot_mc = boot.ascr(cr, N=500, n.cores=4)
summary(boot_mc)
stdEr(boot_mc, mce=T)
save(boot_mc, file=paste0(outdir, "boot_mc.RData"))


## RUNNING WITH CHP ANNOTS
# Create the call history
capt_chp = inner_join(calls_chp, trapgrid, by="rec")[,c("session", "id", "occ", "recid")]
capt_chp = create.capt(as.data.frame(capt_chp), traps=traps)

# inspect the call history over recorders
capt_chp$bincapt
nrow(capt_chp$bincapt)
table(rowSums(capt_chp$bincapt))

cr = fit.ascr(capt_chp, traps, mask, survey.length = 1)
summary(cr)

detfnparams_chp = cr$coefficients
precision_chp = cr$se

show.detfn(cr, xlim=c(0, scr_mask_bufsize))
show.detsurf(cr)

# Bootstrap to get good SE estimates
boot_chp = boot.ascr(cr, N=500, n.cores=4)
summary(boot_chp)
stdEr(boot_chp, mce=T)
save(boot_chp, file=paste0(outdir, "boot_chp.RData"))
