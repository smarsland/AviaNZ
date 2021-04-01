## Analyse the results of Bittern LE survey:
## compare the raw and reviewed annotations obtained by three detection methods
## and run SCR.

## Requires:
## * raw and reviewed annotations in .data format
##   (in dirs structured like rawannots/CHP/Bittern/, reviewed/MC/Bittern/, ...)
## * clock adjustment vector determined with CompareCalls.py
## * Lake Ellesmere shape file from LINZ
## * recorder and speaker GPS positions in this repo

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
outdir = "~/Documents/kiwis/soundchppaper/p3surveys/BitternLE/"

# Reads all AviaNZ style annotations from directory dir,
# over recorders recs (string vector).
# Does some basic date conversions and returns a df.
readAnnots = function(dir, recs){
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
  annot = annot[,6:10]
  
  annot$calllength = annot$end - annot$start
  
  return(annot)
}

# ggplot syntax for plotting raw & reviewed annotations.
# only prints one day, and expects a $start column
plotDayAnnots = function(df){
  if(min(day(df$start))==6 & max(day(df$start))==6){
    minbreaks = seq(ymd_hms("2019-12-06 04:00:00"), ymd_hms("2019-12-06 05:00:00"), by=60)
  } else if(min(day(df$start))==7 & max(day(df$start))==7) {
    minbreaks = seq(ymd_hms("2019-12-07 04:00:00"), ymd_hms("2019-12-07 05:00:00"), by=60)
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
                        BIT1=0, BIT2=0, BIT3=0, BIT4=0, BIT5=0, BIT6=0, BIT8=0))
  
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
  df = mutate(df, anycall = BIT1 | BIT2 | BIT3 | BIT4 | BIT5 | BIT6 | BIT8)
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
recs = paste0("BIT", 1:9)
ans_rev_wf = readAnnots(paste(indir, "reviewed", "WF", "Bittern", sep="/"), recs)
ans_rev_chp = readAnnots(paste(indir, "reviewed", "CHP", "Bittern", sep="/"), recs)
ans_rev_mc = readAnnots(paste(indir, "reviewed", "MC", "Bittern", sep="/"), recs)
ans_raw_wf = readAnnots(paste(indir, "rawannots", "WF", "Bittern", sep="/"), recs)
ans_raw_chp = readAnnots(paste(indir, "rawannots", "CHP", "Bittern", sep="/"), recs)
ans_raw_mc = readAnnots(paste(indir, "rawannots", "MC", "Bittern", sep="/"), recs)

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
table(ans_all$method, ans_all$stage)

# total num TP/FP seconds by method
group_by(ans_all, method, stage) %>%
  summarize(P=sum(calllength))

# attach this for plotting
ans_all = mutate(ans_all, quarter=pmin(3, ifelse(day(start)==6,
      difftime(start, ymd_hms("191206 22:00:00"), units="mins"),
      difftime(start, ymd_hms("191207 04:00:00"), units="mins")) %/% 15))

# clock adjustments, estimated using the CallCompare branch:
clockadjs = tibble(rec=paste0("BIT", 1:9), adjs=c(31.5,0.5,18, 30,0,-24, 0,-24.5,0))
ans_all = left_join(ans_all, clockadjs, by="rec") %>%
  mutate(startadj = start+adjs, endadj=end+adjs) %>%
  arrange(rec, stage)


## PLOT ALL CALLS
# some raw data, without syncing clock times:
ans_all %>%
  filter(method=="WF", day(start)==7) %>%
  plotAnnots

# Same plot, times adjusted for clock drift:
ans_all %>%
  filter(method=="WF", day(start)==6) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ans_all %>%
  filter(method=="WF", day(start)==7) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots

# adjusted times, MC method:
ans_all %>%
  filter(method=="MC", day(start)==6) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ans_all %>%
  filter(method=="MC", day(start)==7) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots

# adjusted times, CHP method:
ans_all %>%
  filter(method=="CHP", day(start)==6) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ggsave(paste0(outdir, "detections_CHP_06.png"), width=8, height=6)
ans_all %>%
  filter(method=="CHP", day(start)==7) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ggsave(paste0(outdir, "detections_CHP_07.png"), width=8, height=6)


# equalize review effort (downsample methods with more reported detections)
dsfactor_wf = nrow(ans_raw_chp)/nrow(ans_raw_wf)
dsfactor_mc = nrow(ans_raw_chp)/nrow(ans_raw_mc)

# subset by time: from each 5 min window 04:00-04:05,04:05-04:10,..., take first X %
# where X is chosen to produce equal numbers of annotations for all methods.
# create a relative time counter from that day's start
ans_all = mutate(ans_all, secsfromstart=ifelse(day(start)==6,
                  difftime(startadj, ymd_hms("191206 22:00:00"), units="secs"),
                  difftime(startadj, ymd_hms("191207 04:00:00"), units="secs")))

ans_final_wf = filter(ans_all, method=="WF", stage=="reviewed") %>%
  filter(secsfromstart %% 300 < dsfactor_wf*300)
ans_final_mc = filter(ans_all, method=="MC", stage=="reviewed") %>%
  filter(secsfromstart %% 300 < dsfactor_mc*300)
ans_final_chp = filter(ans_all, method=="CHP", stage=="reviewed")

# Same plots, but WF and MC effort reduced to match CHP effort:
ans_all %>%
  filter(method=="WF", day(start)==6) %>%
  filter(secsfromstart %% 300 < dsfactor_wf*300) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ggsave(paste0(outdir, "detections_WF_06.png"), width=8, height=6)

ans_all %>%
  filter(method=="WF", day(start)==7) %>%
  filter(secsfromstart %% 300 < dsfactor_wf*300) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ggsave(paste0(outdir, "detections_WF_07.png"), width=8, height=6)

ans_all %>%
  filter(method=="MC", day(start)==6) %>%
  filter(secsfromstart %% 300 < dsfactor_mc*300) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ggsave(paste0(outdir, "detections_MC_06.png"), width=8, height=6)

ans_all %>%
  filter(method=="MC", day(start)==7) %>%
  filter(secsfromstart %% 300 < dsfactor_mc*300) %>%
  mutate(start=startadj, end=endadj) %>%
  plotAnnots
ggsave(paste0(outdir, "detections_MC_07.png"), width=8, height=6)


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
nrow(calls_wf)  # 93
nrow(calls_chp) # 204
nrow(calls_mc)  # 138
# we can see that the ratio enforced at annotation level matches ratio of call detections:
nrow(calls_wf)/nrow(calls_chp); dsfactor_wf
nrow(calls_mc)/nrow(calls_chp); dsfactor_mc


# ------- SCR -------

## MAP & GPS PREPARATIONS
# read recorder positions (in NZTM)
gpspos = read.table("recordergps_NZTM.txt", header=T)

# lake shapefile from LINZ, NZTM2000
lake = readOGR(paste0(mapdir, "map_photo/lake-polygon-hydro-190k-1350k.shp"))

# speaker positions (in NZ TM already)
speakerpos = read.table("speakergps.txt", sep="\t", h=T)

# create trap coord matrix and the integration mask
scr_mask_bufsize = 1200
trapgrid = gpspos %>% mutate(recid=as.numeric(substr(gsub("8", "7", rec), 4,4))) %>% arrange(recid)
traps = as.matrix(trapgrid[,c("east", "north")])
mask = create.mask(traps, buffer=scr_mask_bufsize)

# crop lake parts out of the mask.
# this requires converting the mask into SpatialPoints:
maskDF = SpatialPoints(mask, proj4string=CRS(proj4string(lake)))
maskInLand = is.na(over(maskDF, lake)$fidn)

# plot before cropping:
plot(mask)
points(traps, col="red", pch=4)
points(mask[!maskInLand,], col="seagreen")

# actually crop the mask:
mask = mask[maskInLand,]

# center everything for simpler plotting later:
mapcentroidE = mean(gpspos$east)
mapcentroidN = mean(gpspos$north)

gpsposM = mutate(gpspos, east=east-mapcentroidE, north=north-mapcentroidN, optional=NULL)
trapsM = traps - matrix(rep(c(mapcentroidE, mapcentroidN), each=nrow(traps)), ncol=2)
maskM = mask - matrix(rep(c(mapcentroidE, mapcentroidN), each=nrow(mask)), ncol=2)

# create a "speaker" covariate (field that is 1 near a speaker position, 0 elsewhere)
distToSpeaker1 = sqrt((maskM[,1] - (speakerpos[1,1]-mapcentroidE))^2 + (maskM[,2] - (speakerpos[1,2]-mapcentroidN))^2)
distToSpeaker2 = sqrt((maskM[,1] - (speakerpos[2,1]-mapcentroidE))^2 + (maskM[,2] - (speakerpos[2,2]-mapcentroidN))^2)
speakerPresent = list(data.frame(speakerPres=pmin(distToSpeaker1, distToSpeaker2)<20))  # 20 m distance cutoff

lakepolygon = lake@polygons[[1]]@Polygons[[1]]@coords
lakepolygon = lakepolygon - matrix(rep(c(mapcentroidE, mapcentroidN), each=nrow(lakepolygon)), ncol=2)

ggplot(gpsposM, aes(y=north, x=east)) +
  geom_point() + geom_text(aes(label=rec), nudge_y=50) +
  theme_minimal() + theme(aspect.ratio = 1) +
  coord_cartesian(xlim=c(-1000, 1000), ylim=c(-1000,1000))


## RUNNING WITH WF ANNOTS
# Create the call history
capt_wf = inner_join(calls_wf, trapgrid, by="rec")[,c("session", "id", "occ", "recid")]
capt_wf = create.capt(as.data.frame(capt_wf), traps=traps)

# inspect the call history over recorders
capt_wf$bincapt
nrow(capt_wf$bincapt)
table(rowSums(capt_wf$bincapt))

# Fit the model, accounting for the shorter duration
cr = fit.ascr(capt_wf, trapsM, maskM, survey.length = dsfactor_wf)
summary(cr)

detfnparams_wf = cr$coefficients
precision_wf = cr$se

show.detfn(cr, xlim=c(0, scr_mask_bufsize))
show.detsurf(cr)

# example locations
pdf(paste0(outdir, "locations_wf.pdf"))
par(mfrow=c(2,3), mar=c(0,0,1,1))
for(i in 1:nrow(capt_wf$bincapt)){
  locations(cr, i, xlim=c(-500,500), ylim=c(-500, 500), levels=c(0.1, 0.3, 0.5, 0.7, 0.9), plot.estlocs = TRUE)
  lines(lakepolygon, col="lightseagreen", lwd=3)
}
dev.off()
par(mfrow=c(1,1), mar=c(5,5,4,4))

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

cr = fit.ascr(capt_mc, trapsM, maskM, survey.length = dsfactor_mc)
summary(cr)

detfnparams_mc = cr$coefficients
precision_mc = cr$se

show.detfn(cr, xlim=c(0, scr_mask_bufsize))
show.detsurf(cr)

# example locations
pdf(paste0(outdir, "locations_mc.pdf"))
par(mfrow=c(2,3), mar=c(0,0,1,1))
for(i in 1:nrow(capt_mc$bincapt)){
  locations(cr, i, xlim=c(-500,500), ylim=c(-500, 500), levels=c(0.1, 0.3, 0.5, 0.7, 0.9), plot.estlocs = TRUE)
  lines(lakepolygon, col="lightseagreen", lwd=3)
}
dev.off()
par(mfrow=c(1,1), mar=c(5,5,4,4))

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

cr = fit.ascr(capt_chp, trapsM, maskM, survey.length = 1)
summary(cr)

detfnparams_chp = cr$coefficients
precision_chp = cr$se

show.detfn(cr, xlim=c(0, scr_mask_bufsize))
show.detsurf(cr)

# example locations
pdf(paste0(outdir, "locations_chp.pdf"))
par(mfrow=c(2,3), mar=c(0,0,1,1))
for(i in 1:nrow(capt_chp$bincapt)){
  locations(cr, i, xlim=c(-500,500), ylim=c(-500, 500), levels=c(0.1, 0.3, 0.5, 0.7, 0.9), plot.estlocs = TRUE)
  lines(lakepolygon, col="lightseagreen", lwd=3)
}
dev.off()
par(mfrow=c(1,1), mar=c(5,5,4,4))

# Bootstrap to get good SE estimates
boot_chp = boot.ascr(cr, N=500, n.cores=4)
summary(boot_chp)
stdEr(boot_chp, mce=T)
save(boot_chp, file=paste0(outdir, "boot_chp.RData"))


## ALTERNATIVE MODELS
# Can account for speaker positions by setting it as density covariate:
cr = fit.ascr(capt_wf, trapsM, maskM, survey.length = dsfactor_wf,
              ihd.opts = list(model=~speakerPres, covariates=speakerPresent, scale=F))
summary(cr)
cr = fit.ascr(capt_mc, trapsM, maskM, survey.length = dsfactor_mc,
              ihd.opts = list(model=~speakerPres, covariates=speakerPresent, scale=F))
summary(cr)
cr = fit.ascr(capt_chp, trapsM, maskM, survey.length = 1,
              ihd.opts = list(model=~speakerPres, covariates=speakerPresent, scale=F))
summary(cr)
